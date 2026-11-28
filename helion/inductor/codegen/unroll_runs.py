"""Unroll the runs of a persistent reduction that reads its rdim by runs.

A kernel that reads its rdim as Q runs of W elements (HelionKernel.rdim_runs())
tiles the runs as one block of an outer loop, tile_q, beside one run of the
rdim, and reduces each value over the run, then over the runs. A BatchNorm in
training over an NCHW [16, 256, 14, 14] bf16 (better-benchmark's
var_mean_0c1c9fc832f4_b2063b0a), 16 runs of 196 per channel, with its running
statistics updated in place:

    for tile_x, tile_q in hl.tile([256, 16], block_size=[None, 16]):
        tmp0 = in0[tile_x, tile_q, :].to(torch.float32)
        ...
        tmp5 = torch.sum(tmp4, -1)
        tmp6 = torch.sum(tmp5, -1)
        tmp10 = tmp6.reshape([tile_x, 1, 1])
        ...
        out2[tile_x, tile_q, :] = tmp52.to(torch.bfloat16)
        in1[tile_x] = tmp37

The CuTe backend refuses that kernel at every config: "threads sharing a tile
axis race on a tensor in a body a later pass rewrites (_helion_lane_reduce),
where the barrier ordering them cannot be placed". unroll_runs() writes it
with one value per run instead, and adds the runs' values elementwise before
one reduction over the run:

    for tile_x in hl.tile(256):
        tmp0_0 = in0[tile_x, 0, :].to(torch.float32)
        ...
        tmp0_15 = in0[tile_x, 15, :].to(torch.float32)
        ...
        tmp6 = torch.sum(tmp4_0 + ... + tmp4_15, -1)
        tmp10 = tmp6.reshape([tile_x, 1])
        ...
        out2[tile_x, 0, :] = tmp52_0.to(torch.bfloat16)
        ...
        in1[tile_x] = tmp37

which CuTe renders with each run in packets. Anything the rewrite does not
recognize leaves the kernel as it was (None): a statement but an assignment,
tile_q other than as an index, a tensor method of the run's index alone, a call
that names dims by position, a reduction over each run other than a sum, or a
value reduced over each run that anything but its sum over the runs reads.
"""

from __future__ import annotations

import ast
import copy
from typing import cast

UNROLL_TILE = "tile_q"


def _pairwise_sum(terms: list[ast.expr]) -> ast.expr:
    """The sum of `terms` added in pairs, (v0 + v1) + (v2 + v3): each lane's
    sum of 16 runs rounds through 4 additions rather than 15, as the
    reduction over the run's lanes after it does. In sequence, the values of
    a log-uniform input over 16 runs (DenseNet's BatchNorm over a cat of 80
    channels, float32) came out 2.1 ulp off against eager's 0.2."""
    if len(terms) == 1:
        return terms[0]
    half = len(terms) // 2
    return ast.BinOp(
        left=_pairwise_sum(terms[:half]),
        op=ast.Add(),
        right=_pairwise_sum(terms[half:]),
    )


def _names(node: ast.AST) -> set[str]:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


# Calls whose result or arguments name dims by position, which the unrolled
# kernel numbers one lower past the runs' dim: their meaning there is not
# rewritten, so the rewrite declines them.
POSITIONAL_DIM_CALLS = frozenset(
    [
        "broadcast_to",
        "cat",
        "chunk",
        "expand",
        "expand_as",
        "flatten",
        "flip",
        "gather",
        "index_select",
        "movedim",
        "narrow",
        "permute",
        "repeat",
        "roll",
        "select",
        "split",
        "squeeze",
        "stack",
        "transpose",
        "unsqueeze",
        "view",
    ]
)
# Reductions, which the unrolled kernel takes over its last dim as the tile
# of the runs does (torch.sum(v, -1)).
REDUCTION_CALLS = frozenset(
    [
        "amax",
        "amin",
        "any",
        "all",
        "argmax",
        "argmin",
        "max",
        "mean",
        "min",
        "prod",
        "sum",
    ]
)


def _is_last_dim(node: ast.expr) -> bool:
    return (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, ast.USub)
        and isinstance(node.operand, ast.Constant)
        and node.operand.value == 1
    )


def _reduction_over_last(node: ast.expr) -> tuple[str, ast.Name] | None:
    """The kind and operand of torch.<kind>(name, -1), a reduction over the
    last dim, else None."""
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in REDUCTION_CALLS
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "torch"
        and len(node.args) == 2
        and not node.keywords
        and isinstance(node.args[0], ast.Name)
        and _is_last_dim(node.args[1])
    ):
        return node.func.attr, node.args[0]
    return None


def _is_tile(node: ast.AST) -> bool:
    return isinstance(node, ast.Name) and node.id == UNROLL_TILE


class _PerRun(ast.NodeTransformer):
    """A copy of an expression for run `index`: per-run names take their
    run's suffix; tile_q as an index (x[tile_x, tile_q, :], hl.load(x,
    [tile_x, tile_q, slice(None)])) and tile_q.index, reshaped or not, take
    the run's index; reshapes, and the size of a load of a run, drop the
    runs' dim. tile_q anywhere else (a shape, tile_q.begin), a call that
    names dims by position, a reduction over another dim than the last, and
    tile_q in a copy for no run (`index` None) decline."""

    def __init__(
        self, per_run: set[str], index: int | None, q_pos: int, params: set[str]
    ) -> None:
        self.per_run = per_run
        self.index = index
        self.q_pos = q_pos
        self.params = params

    def run_index(self, node: ast.expr) -> ast.expr:
        if self.index is None:
            raise _Decline
        return ast.copy_location(ast.Constant(value=self.index), node)

    def run_indices(self, elts: list[ast.expr]) -> list[ast.expr]:
        return [self.run_index(elt) if _is_tile(elt) else elt for elt in elts]

    def visit_Name(self, node: ast.Name) -> ast.expr:
        if node.id == UNROLL_TILE:
            raise _Decline
        if self.index is not None and node.id in self.per_run:
            return ast.copy_location(
                ast.Name(id=f"{node.id}_{self.index}", ctx=node.ctx), node
            )
        return node

    def visit_Subscript(self, node: ast.Subscript) -> ast.expr:
        # in0[tile_x, tile_q, :] reads in0[tile_x, k, :] for run k. Only the
        # kernel's tensors are subscripted: tmp[:, None] would name dims by
        # position.
        if not (isinstance(node.value, ast.Name) and node.value.id in self.params):
            raise _Decline
        if isinstance(node.slice, ast.Tuple):
            node.slice.elts = self.run_indices(node.slice.elts)
        self.generic_visit(node)
        return node

    def visit_Attribute(self, node: ast.Attribute) -> ast.expr:
        if _is_tile(node.value):
            if node.attr != "index":
                raise _Decline
            return self.run_index(node)
        self.generic_visit(node)
        if not _names(node.value):
            # A tensor method of an expression of the run's index alone,
            # tile_q.index.reshape([tile_q, 1]).to(torch.int64) or
            # (784*tile_q.index).to(torch.int64), which the run's number, a
            # Python int, lacks.
            raise _Decline
        return node

    def visit_Call(self, node: ast.Call) -> ast.expr:
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else None
        if name in POSITIONAL_DIM_CALLS:
            raise _Decline
        # A run's own reduction (over the run) takes the sum over the runs
        # that follows it as its partner, which _unroll() matches; one in a
        # copy for a run has none.
        if name in REDUCTION_CALLS and (
            self.index is not None
            or not (
                len(node.args) == 2 and not node.keywords and _is_last_dim(node.args[1])
            )
        ):
            raise _Decline
        if name == "reshape" and isinstance(func, ast.Attribute):
            receiver = func.value
            # tile_q.index.reshape([tile_q, 1]): run k's index, a scalar.
            if isinstance(receiver, ast.Attribute) and _is_tile(receiver.value):
                return self.visit_Attribute(receiver)
        if (
            name == "load"
            and isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Name)
            and func.value.id == "hl"
            and len(node.args) >= 2
            and isinstance(node.args[1], ast.List)
        ):
            node.args[1].elts = self.run_indices(node.args[1].elts)
        size_dim = None
        if (
            name == "size"
            and isinstance(func, ast.Attribute)
            and len(node.args) == 1
            and isinstance(node.args[0], ast.Constant)
            and type(node.args[0].value) is int
        ):
            # rdim = in0[tile_x, tile_q, :].size(2): a run's load drops the
            # runs' dim, in0[tile_x, k, :].size(1).
            receiver = func.value
            subscript = (
                receiver.slice.elts
                if isinstance(receiver, ast.Subscript)
                and isinstance(receiver.slice, ast.Tuple)
                else []
            )
            runs_dim = next((i for i, e in enumerate(subscript) if _is_tile(e)), None)
            size_dim = node.args[0].value
            if runs_dim is not None:
                if size_dim == runs_dim:
                    raise _Decline
                size_dim -= size_dim > runs_dim
            elif _names(receiver) & (self.per_run | {UNROLL_TILE}):
                raise _Decline
        self.generic_visit(node)
        if size_dim is not None:
            node.args[0] = ast.Constant(value=size_dim)
        if (
            name == "reshape"
            and len(node.args) == 1
            and isinstance(node.args[0], ast.List)
        ):
            dims = node.args[0].elts
            if len(dims) <= self.q_pos:
                raise _Decline
            dim = dims[self.q_pos]
            if not (isinstance(dim, ast.Constant) and dim.value == 1):
                raise _Decline
            node.args[0] = ast.List(
                elts=[d for i, d in enumerate(dims) if i != self.q_pos],
                ctx=ast.Load(),
            )
        return node


class _Decline(Exception):
    pass


def _rewrite(
    node: ast.expr, per_run: set[str], index: int | None, q_pos: int, params: set[str]
) -> ast.expr:
    return cast(
        "ast.expr", _PerRun(per_run, index, q_pos, params).visit(copy.deepcopy(node))
    )


def unroll_runs(source: str, runs: int) -> str | None:
    """`source`, a kernel of one loop whose tiles include tile_q over `runs`
    runs in one block, with the runs unrolled (module docstring); None where it
    holds anything the rewrite does not recognize."""
    try:
        return _unroll(source, runs)
    except _Decline:
        return None


def _unroll(source: str, runs: int) -> str | None:
    tree = ast.parse(source)
    kernel = next(
        (n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "kernel"),
        None,
    )
    if (
        kernel is None
        or len(kernel.body) != 1
        or not isinstance(kernel.body[0], ast.For)
    ):
        return None
    loop = kernel.body[0]
    params = {arg.arg for arg in kernel.args.args}
    target = loop.target
    if not isinstance(target, ast.Tuple):
        return None
    tiles = [elt.id if isinstance(elt, ast.Name) else None for elt in target.elts]
    if UNROLL_TILE not in tiles:
        return None
    q_pos = tiles.index(UNROLL_TILE)
    call = loop.iter
    if not (
        isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == "tile"
        and len(call.args) == 1
        and isinstance(call.args[0], ast.List)
        and len(call.args[0].elts) == len(tiles)
    ):
        return None
    sizes = call.args[0].elts
    q_size = sizes[q_pos]
    if not (isinstance(q_size, ast.Constant) and q_size.value == runs):
        return None
    keywords = [k for k in call.keywords if k.arg != "block_size"]
    blocks = next((k.value for k in call.keywords if k.arg == "block_size"), None)
    q_block = blocks.elts[q_pos] if isinstance(blocks, ast.List) else None
    # The block of tile_q holds every run: the runs' power of two, whose
    # lanes past the runs the tile masks, and the copies, one per run, have
    # none of.
    if not (
        isinstance(blocks, ast.List)
        and isinstance(q_block, ast.Constant)
        and isinstance(q_block.value, int)
        and q_block.value >= runs
        and all(
            isinstance(b, ast.Constant) and b.value is None
            for i, b in enumerate(blocks.elts)
            if i != q_pos
        )
    ):
        return None
    kept = [t for i, t in enumerate(target.elts) if i != q_pos]
    kept_sizes = [s for i, s in enumerate(sizes) if i != q_pos]
    loop.target = kept[0] if len(kept) == 1 else ast.Tuple(elts=kept, ctx=ast.Store())
    call.args = [
        kept_sizes[0]
        if len(kept_sizes) == 1
        else ast.List(elts=kept_sizes, ctx=ast.Load())
    ]
    call.keywords = keywords

    per_run: set[str] = set()
    # torch.sum(v, -1) of a per-run v, the sum over each run, by the name it
    # assigns: the sum over the runs that follows reads the runs' values
    # added elementwise instead, and drops it (dropped).
    partial: dict[str, str] = {}
    dropped: set[str] = set()
    body: list[ast.stmt] = []
    for stmt in loop.body:
        if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
            return None
        (store,) = stmt.targets
        value = stmt.value
        reads = _names(value)
        if isinstance(store, ast.Name):
            reduction = _reduction_over_last(value)
            kind, operand = reduction if reduction is not None else (None, None)
            if operand is not None and operand.id in per_run:
                if kind != "sum":
                    return None
                partial[store.id] = operand.id
                continue
            if operand is not None and operand.id in partial:
                if kind != "sum":
                    return None
                dropped.add(operand.id)
                source_name = partial.pop(operand.id)
                folded = cast("ast.Call", copy.deepcopy(value))
                folded.args[0] = _pairwise_sum(
                    [
                        ast.Name(id=f"{source_name}_{index}", ctx=ast.Load())
                        for index in range(runs)
                    ]
                )
                body.append(ast.Assign(targets=[store], value=folded))
                continue
            if reads & (set(partial) | dropped):
                return None
            if (
                isinstance(value, ast.Call)
                and isinstance(value.func, ast.Attribute)
                and value.func.attr == "size"
                and not reads & per_run
            ):
                # rdim = in0[tile_x, tile_q, :].size(2): every run's load has
                # one size, rdim = in0[tile_x, 0, :].size(1).
                body.append(
                    ast.Assign(
                        targets=[store],
                        value=_rewrite(value, per_run, 0, q_pos, params),
                    )
                )
                continue
            if reads & per_run or UNROLL_TILE in reads:
                for index in range(runs):
                    body.append(
                        ast.Assign(
                            targets=[
                                ast.Name(id=f"{store.id}_{index}", ctx=ast.Store())
                            ],
                            value=_rewrite(value, per_run, index, q_pos, params),
                        )
                    )
                per_run.add(store.id)
                continue
            body.append(
                ast.Assign(
                    targets=[store], value=_rewrite(value, per_run, None, q_pos, params)
                )
            )
            continue
        if not isinstance(store, ast.Subscript):
            return None
        if (reads | _names(store)) & (set(partial) | dropped):
            return None
        if reads & per_run and UNROLL_TILE not in _names(store):
            # A run's values stored where no run's index tells them apart.
            return None
        if (reads | _names(store)) & per_run or UNROLL_TILE in _names(store):
            for index in range(runs):
                body.append(
                    ast.Assign(
                        targets=[_rewrite(store, per_run, index, q_pos, params)],
                        value=_rewrite(value, per_run, index, q_pos, params),
                    )
                )
            continue
        body.append(
            ast.Assign(
                targets=[_rewrite(store, per_run, None, q_pos, params)],
                value=_rewrite(value, per_run, None, q_pos, params),
            )
        )
    if partial:
        return None
    loop.body = body
    return ast.unparse(ast.fix_missing_locations(tree)) + "\n"
