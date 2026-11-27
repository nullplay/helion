"""Issue the loads of one-trip tile loops first and keep their packets in registers.

A kernel that reduces a row and then reads it again (a BatchNorm or a
LayerNorm: the mean, the deviations, then the normalized values) loops over
the row in each pass.  When the row fits one tile, every such loop runs once,
yet each pass still loads what it reads when it gets there.  The CuTe code of
a BatchNorm over [96, 256, 8, 8] at 256 threads, one 8192-element tile per
channel (4 lanes of 8 bf16 per thread)::

    for tile_offset_1 in range(cutlass.Int32(0), cutlass.Int32(6144), cutlass.Int32(_BLOCK_SIZE_1)):
        for lane_1 in cutlass.range_constexpr(4):          # the sum
            lane_base_1 = tile_offset_1 + (cutlass.Int32(lane_1) * 256 + tid) * 8
            if lane_base_1 < 6144 and ...:
                x = cute.arch.load(in0.iterator + ADDR(lane_base_1), VT)
                ...
        mean = block_sum(...) / 6144
        for lane_1 in cutlass.range_constexpr(4):          # the deviations
            lane_base_1 = ...
            if lane_base_1 < 6144 and ...:
                x = cute.arch.load(in0.iterator + ADDR(lane_base_1), VT)   # again
                ...
    (out0.iterator + ...).store(mean)
    tmp37 = (in1.iterator + indices_0 * ...).load()        # running stats, after both reductions
    ...
    for tile_offset_2 in range(cutlass.Int32(0), cutlass.Int32(6144), cutlass.Int32(_BLOCK_SIZE_2)):
        for lane_2 in range(4):                             # the normalize, one lane per trip
            lane_base_2 = tile_offset_2 + (cutlass.Int32(lane_2) * 256 + tid) * 8
            if lane_base_2 < 6144 and ...:
                x = cute.arch.load(in0.iterator + ADDR(lane_base_2), VT)  # a third time
                r = cute.arch.load(in5.iterator + ADDR(lane_base_2), VT)  # first touch, from DRAM
                ... out2 packet store

Each later load waits for what precedes it: the residual ``r`` and the
running stats only leave for DRAM after both block reductions, and the
normalize's runtime lane loop issues one lane's packets per trip, after the
previous trip's store.  Triton's persistent kernel of the same computation
issues every load first.  This pass gives the CuTe code that schedule:

* a per-tile scalar load (an address of the tile coordinates alone, such as
  a running statistic or a channel weight) moves up to the end of the
  kernel's leading definitions, once (``_hoist_tile_scalars``), when a
  runtime lane loop or a multi-trip loop repeats it, or when a block
  reduction (a CTA barrier) precedes it, as the running stats above, whose
  latency then overlaps the reductions;
* the tile loops that run once (``range(0, E, B)`` with a constexpr block
  ``B >= E``) are inlined, and the packets their lane loops load are loaded
  once per lane, in the first lane loop of the same lane layout (a guarded
  packet predicated, ``_cute_load_v16b_if``: zeros where the guard is false,
  no branch; an unguarded one by its own ``cute.arch.load``); every
  later lane loop of that layout reads them from registers, and its runtime
  lane loop becomes a trace-time one::

        _pk0 = []
        _pk1 = []
        for lane_1 in cutlass.range_constexpr(4):  # the sum
            lane_base_1 = ...
            _pk0.append(_cute_load_v16b_if(in0.iterator + ADDR(lane_base_1), GUARD, VT))
            _pk1.append(_cute_load_v16b_if(in5.iterator + ADDR(lane_base_1), GUARD, VT))
            if GUARD:
                x = _pk0[lane_1]
                ...
        ...  # the deviations read _pk0 too
        for lane_2 in cutlass.range_constexpr(4):  # the normalize
            lane_base_2 = ...
            if GUARD:
                x = _pk0[lane_2]
                r = _pk1[lane_2]
                ...

A packet is shared only when its tensor is never written by the kernel and is
proven disjoint from every tensor the kernel writes
(``proven_disjoint_tensor_pairs``), its address and guard are the same
register computation of the lane in both places (no load or other memory read
in them, names of one value only), and the packets of a thread fit
``_register_budget``; the others keep their loads.

``sink_vector_loops._reuse_sibling_packets`` serves a later sibling lane loop
from an earlier one's packets too, but only for identical load statements
directly in the bodies of two lane loops of one statement list; the loops
this pass joins sit in separate tile loops (the reduction's and the
normalize's), their packets may sit in ``if`` blocks, and the normalize's
residual is loaded nowhere before it, so it is moved (prefetched), not
reused.
"""

from __future__ import annotations

import ast
from collections import Counter
from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING
from typing import cast

from ..ast_extension import expr_from_string
from ..ast_extension import statement_from_string
from ._ast_pass_utils import _is_proven_relocatable_call
from ._ast_pass_utils import deep_clone
from .cache_policy_loads import MASKED_PACKET_LOAD_HELPERS
from .fuse_two_pass_loads import _CuteFuseTwoPassLoads
from .fuse_two_pass_loads import _tensor_arg_roots
from .packet_access import call_write_roots
from .packet_access import match_packet_access
from .packet_access import render_masked_packet_load

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Collection
    from collections.abc import Iterator

# The bytes of packets a thread keeps in registers for the later lane loops
# (64 32-bit registers), and at most a quarter of what the register file
# gives each thread of the CTA (65536 registers: 64 each at 1024 threads, so
# 64 bytes there).  The packets past it keep their loads.
REGISTER_BUDGET_BYTES = 256
_REGISTER_FILE_BYTES = 65536 * 4


def _register_budget(threads: int | None) -> int:
    if not threads:
        return REGISTER_BUDGET_BYTES
    return min(REGISTER_BUDGET_BYTES, _REGISTER_FILE_BYTES // threads // 4)


_ITEMSIZES = {
    "cutlass.Uint8": 1,
    "cutlass.Int8": 1,
    "cutlass.Uint16": 2,
    "cutlass.Int16": 2,
    "cutlass.BFloat16": 2,
    "cutlass.Float16": 2,
    "cutlass.Uint32": 4,
    "cutlass.Int32": 4,
    "cutlass.Float32": 4,
}
_COMPOUND = (ast.For, ast.If, ast.While, ast.With, ast.Try)
_LAUNCH_COORDINATES = frozenset(
    (
        "cute.arch.thread_idx",
        "cute.arch.block_idx",
        "cute.arch.block_dim",
        "cute.arch.grid_dim",
    )
)


def issue_packets_first(
    body: list[ast.stmt],
    *,
    constexpr_values: dict[str, int],
    tensor_names: Collection[str],
    proven_disjoint_tensor_pairs: Collection[frozenset[str]],
    rename_groups: dict[str, str] | None = None,
    thread_block_dims: tuple[int, int, int] | None = None,
) -> list[ast.stmt]:
    """The kernel ``body`` with the per-tile scalar loads its loops repeat at
    the top and the packets of its one-trip tile loops loaded first (module
    docstring).

    ``rename_groups`` maps a name to the one the final rename gives it: a
    name merged with another (a loop-carried value and its accumulator) is
    no single definition: a packet whose key reads such a name, and a scalar
    load that reads or assigns one, keep their loads."""
    tensors = set(tensor_names)
    disjoint = set(proven_disjoint_tensor_pairs)
    groups = rename_groups or {}
    spellings: dict[str, set[str]] = {}
    for node in ast.walk(ast.Module(body=body, type_ignores=[])):
        if isinstance(node, ast.Name):
            spellings.setdefault(groups.get(node.id, node.id), set()).add(node.id)
    merged = {n for names in spellings.values() if len(names) > 1 for n in names}
    writes = _writes(body, tensors)
    if writes is None or any(
        isinstance(node, (ast.Return, ast.Raise))
        for node in ast.walk(ast.Module(body=body, type_ignores=[]))
    ):
        # After an early exit a statement need not run whenever the kernel
        # does, which both parts assume of the loads they move earlier.
        return body
    body = _hoist_tile_scalars(
        body, tensors, disjoint, writes, merged, constexpr_values
    )
    threads = (
        None
        if thread_block_dims is None
        else thread_block_dims[0] * thread_block_dims[1] * thread_block_dims[2]
    )
    return _keep_one_trip_packets(
        body, constexpr_values, tensors, disjoint, merged, _register_budget(threads)
    )


# --------------------------------------------------------------------------
# Program order and memory writes
# --------------------------------------------------------------------------


def _statements(body: list[ast.stmt]) -> Iterator[ast.stmt]:
    """Every statement of ``body``, nested ones included, in program order."""
    for statement in body:
        yield statement
        for name in ("body", "orelse", "finalbody"):
            inner = getattr(statement, name, None)
            if isinstance(inner, list):
                yield from _statements(inner)
        for handler in getattr(statement, "handlers", ()):
            yield from _statements(handler.body)


def _order(body: list[ast.stmt]) -> dict[int, int]:
    """Statement id -> its position in program order."""
    return {id(s): i for i, s in enumerate(_statements(body))}


@dataclass
class _Write:
    statement: ast.stmt
    roots: frozenset[str]


def _writes(body: list[ast.stmt], tensors: set[str]) -> list[_Write] | None:
    """The memory writes of ``body`` with the tensors each may write; None
    when a call or a subscript store may write memory it does not name."""
    found = []
    containers = _local_containers(body)
    for statement in _statements(body):
        for node in _own_nodes(statement):
            if isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Store):
                # A subscript store into a kernel tensor, ``T[i] = v``, writes
                # T as ``(T.iterator + i).store(v)`` does.
                named = frozenset(
                    child.id
                    for child in ast.walk(node.value)
                    if isinstance(child, ast.Name) and child.id in tensors
                )
                if named:
                    found.append(_Write(statement, named))
                    continue
                if isinstance(node.value, ast.Name) and node.value.id in containers:
                    continue
                # Any other target may write a kernel tensor: after
                # ``alias = y`` or ``held = [y]``, ``alias[pid] = acc`` and
                # ``held[0][pid] = acc`` write y, and a packet of y loaded
                # first would miss it.
                return None
            if not isinstance(node, ast.Call):
                continue
            roots = call_write_roots(node, lambda e: _tensor_arg_roots(e, tensors))
            if roots is None:
                return None
            if roots:
                found.append(_Write(statement, roots))
    return found


def _local_containers(body: list[ast.stmt]) -> set[str]:
    """The names that ``body`` binds only to registers or shared memory, so
    that a store into one of their elements, ``NAME[i] = v``, writes no
    kernel tensor: each of their bindings is ``NAME = cute.make_rmem_tensor(
    ...)``, a trace-time list literal, or ``NAME = cute.make_tensor(p, ...)``
    of a pointer ``p`` bound only to ``cute.arch.alloc_smem(...)`` (a shared
    packet's staging). A store through an element, ``NAME[0][i] = v``, may
    write a tensor the list holds (``_writes()``)."""
    module = ast.Module(body=body, type_ignores=[])
    stores = Counter(
        node.id
        for node in ast.walk(module)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    )
    values: dict[str, list[ast.expr]] = {}
    for node in ast.walk(module):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                values.setdefault(target.id, []).append(node.value)

    def only(name: str, accept: Callable[[ast.expr], bool]) -> bool:
        bound = values.get(name, [])
        return len(bound) == stores[name] > 0 and all(accept(v) for v in bound)

    def calls(value: ast.expr, path: str) -> bool:
        return isinstance(value, ast.Call) and ast.unparse(value.func) == path

    def shared_view(value: ast.expr) -> bool:
        if not calls(value, "cute.make_tensor"):
            return False
        args = cast("ast.Call", value).args
        return (
            bool(args)
            and isinstance(args[0], ast.Name)
            and only(args[0].id, lambda v: calls(v, "cute.arch.alloc_smem"))
        )

    return {
        name
        for name in values
        if only(
            name,
            lambda v: (
                calls(v, "cute.make_rmem_tensor")
                or isinstance(v, ast.List)
                or shared_view(v)
            ),
        )
    }


def _own_nodes(statement: ast.stmt) -> Iterator[ast.AST]:
    """The nodes of ``statement`` outside its nested statement lists."""
    stack: list[ast.AST] = [statement]
    while stack:
        node = stack.pop()
        yield node
        for name, value in ast.iter_fields(node):
            if node is statement and name in (
                "body",
                "orelse",
                "finalbody",
                "handlers",
            ):
                continue
            if isinstance(value, list):
                stack.extend(v for v in value if isinstance(v, ast.AST))
            elif isinstance(value, ast.AST):
                stack.append(value)


def _disjoint(
    tensor: str, roots: frozenset[str], disjoint: set[frozenset[str]]
) -> bool:
    return all(r != tensor and frozenset((tensor, r)) in disjoint for r in roots)


def _stored_names(body: list[ast.stmt]) -> Counter[str]:
    module = ast.Module(body=body, type_ignores=[])
    return Counter(
        n.id
        for n in ast.walk(module)
        if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del))
    )


def _names(node: ast.AST) -> set[str]:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def _reads_no_memory(node: ast.AST) -> bool:
    """Whether ``node`` is register arithmetic over names and launch
    coordinates: every call proven pure and no load (a load moved with it
    would read its tensor at another point of the kernel, where a write may
    stand between), and no subscript but a tensor's layout extents, spelled
    ``T[i]`` or ``operator.getitem(T, i)``."""

    def indexes_registers(value: ast.AST) -> bool:
        return (
            isinstance(value, ast.Attribute) and value.attr in ("stride", "shape")
        ) or (
            isinstance(value, ast.Call)
            and ast.unparse(value.func) in _LAUNCH_COORDINATES
        )

    for child in ast.walk(node):
        if (
            isinstance(child, ast.Call)
            and ast.unparse(child.func) == "operator.getitem"
            and not (child.args and indexes_registers(child.args[0]))
        ):
            return False
        if isinstance(child, ast.Call) and not _is_proven_relocatable_call(
            child, allow_load=False
        ):
            return False
        if isinstance(child, ast.Subscript) and not indexes_registers(child.value):
            return False
    return True


# --------------------------------------------------------------------------
# Part 1: per-tile scalar loads at the top
# --------------------------------------------------------------------------


def _scalar_load(statement: ast.stmt) -> str | None:
    """``NAME`` of ``NAME = (pointer).load()``."""
    if not (
        isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
    ):
        return None
    access = match_packet_access(statement.value)
    if access is None or access.kind != "load":
        return None
    return statement.targets[0].id


def _always_runs(text: str, block: list[ast.stmt], trips: dict[int, int]) -> bool:
    """Whether a statement spelled ``text`` runs whenever ``block`` does: it
    is in ``block``, in both arms of an ``if`` of it, or in the body of a
    loop of it that runs at least once (``trips``)."""
    for statement in block:
        if ast.unparse(statement) == text:
            return True
        if isinstance(statement, ast.If) and statement.orelse:
            if _always_runs(text, statement.body, trips) and _always_runs(
                text, statement.orelse, trips
            ):
                return True
        if (
            isinstance(statement, ast.For)
            and trips.get(id(statement), 0) >= 1
            and not _cuts_loop(statement)
        ):
            if _always_runs(text, statement.body, trips):
                return True
    return False


def _hoist_tile_scalars(
    body: list[ast.stmt],
    tensors: set[str],
    disjoint: set[frozenset[str]],
    writes: list[_Write],
    merged: set[str],
    constexpr_values: dict[str, int],
) -> list[ast.stmt]:
    """Move each scalar load whose address reads only the kernel's leading
    definitions (the tile coordinates), and that a loop repeats or a CTA
    barrier precedes, to the end of those definitions, once.

    A load moves when one of its places sits in a runtime loop that can run
    more than one trip (a runtime lane loop, a multi-trip tile loop; not a
    ``cutlass.range_constexpr`` loop, ``_trace_time_loop``), or a statement
    that
    synchronizes the CTA (a block or cluster reduction, a barrier:
    ``_synchronizes_cta``) runs between those definitions and its first
    place; when every statement assigning its name is that load, it runs
    whenever the kernel does (``_always_runs``), and no write before its last
    place, nor any write in a loop around one of its places, may touch its
    tensor (a write of another tensor proven disjoint may).  Moving a load
    earlier then reads what it read: no write of this or another thread
    reaches the address between the two places (the kernel's other writes
    that follow still follow).

    Another straight-line load stays: it runs once where it is, and moved
    above lane loops and warp reductions it would only hold a register across
    them, with no wait for its latency to overlap.  Above a barrier, the
    load's round trip overlaps the wait (a BatchNorm's running statistics,
    read after its two block reductions)."""
    prefix_end = next(
        (i for i, s in enumerate(body) if isinstance(s, _COMPOUND)), len(body)
    )
    stored = _stored_names(body)
    prefix = {
        name
        for s in body[:prefix_end]
        for name in _stored_names([s])
        if stored[name] == 1 and name not in merged
    }
    order = _order(body)
    trips = _trip_counts(body, constexpr_values)
    enclosing: dict[int, list[ast.stmt]] = {}

    def note_enclosing(block: list[ast.stmt], outer: list[ast.stmt]) -> None:
        for statement in block:
            enclosing[id(statement)] = outer
            for name in ("body", "orelse"):
                inner = getattr(statement, name, None)
                if isinstance(inner, list):
                    note_enclosing(inner, [*outer, statement])

    note_enclosing(body, [])
    loads: dict[str, list[ast.Assign]] = {}
    for statement in _statements(body[prefix_end:]):
        found = _scalar_load(statement)
        if found is not None:
            loads.setdefault(found, []).append(cast("ast.Assign", statement))
    moved: list[ast.stmt] = []
    taken: set[int] = set()
    for name, places in loads.items():
        text = ast.unparse(places[0].value)
        call = places[0].value
        assert isinstance(call, ast.Call)
        pointer = match_packet_access(call)
        assert pointer is not None
        roots = _tensor_arg_roots(pointer.pointer, tensors)
        if (
            name in prefix
            or name in merged
            or stored[name] != len(places)
            or any(ast.unparse(p.value) != text for p in places)
            or roots is None
            or len(roots) != 1
            or not _names(pointer.pointer) <= prefix | tensors | _globals(body, stored)
            or not _reads_no_memory(pointer.pointer)
            or not _always_runs(ast.unparse(places[0]), body[prefix_end:], trips)
        ):
            continue
        first = min(order[id(p)] for p in places)
        repeated = any(
            isinstance(outer, ast.While)
            or (
                isinstance(outer, ast.For)
                and not _trace_time_loop(outer)
                and trips.get(id(outer)) != 1
            )
            for p in places
            for outer in enclosing[id(p)]
        )
        if not repeated and not any(
            order[id(s)] < first and _synchronizes_cta(s)
            for s in _statements(body[prefix_end:])
        ):
            continue
        (tensor,) = roots
        last = max(order[id(p)] for p in places)
        loops = {
            id(outer)
            for p in places
            for outer in enclosing[id(p)]
            if isinstance(outer, (ast.For, ast.While))
        }
        if any(
            not _disjoint(tensor, w.roots, disjoint)
            and (
                order[id(w.statement)] < last
                or any(id(o) in loops for o in enclosing[id(w.statement)])
            )
            for w in writes
        ):
            continue
        # Read before its first place: the name would change meaning.
        if any(
            isinstance(n, ast.Name) and n.id == name and isinstance(n.ctx, ast.Load)
            for s in _statements(body)
            if order[id(s)] < first
            for n in _own_nodes(s)
        ):
            continue
        moved.append(deep_clone(places[0]))
        taken.update(id(p) for p in places)
    if not moved:
        return body

    def strip(block: list[ast.stmt]) -> list[ast.stmt]:
        kept = []
        for statement in block:
            if id(statement) in taken:
                continue
            for name in ("body", "orelse"):
                inner = getattr(statement, name, None)
                if isinstance(inner, list) and inner:
                    setattr(statement, name, strip(inner) or [ast.Pass()])
            kept.append(statement)
        return kept

    rest = strip(body[prefix_end:])
    return [*body[:prefix_end], *moved, *rest]


# Calls that synchronize the CTA or its cluster: barriers, and the block and
# cluster reductions of ``reduce_helpers`` (shared-memory exchanges between
# barriers).  The warp reductions (``_cute_grouped_reduce_warp*``,
# ``cute.arch.warp_reduction_*``) do not.
_CTA_BARRIER_CALLS = frozenset(
    {
        "cute.arch.sync_threads",
        "cute.arch.barrier",
        "cute.arch.cluster_arrive",
        "cute.arch.cluster_wait",
    }
)
_CTA_REDUCTION_CALL_PREFIXES = (
    "_cute_grouped_reduce_shared",
    "_cute_grouped_reduce_block",
    "_cute_grouped_reduce_cluster",
    "_cute_cooperative_column_sums",
    "_cute_grid_barrier",
)


def _trace_time_loop(loop: ast.For) -> bool:
    """Whether ``loop`` unrolls at trace time (``cutlass.range_constexpr``):
    its trips are copies of the body in straight-line code, whose copies of a
    load the compiler may merge.  Moving such a load gains no loop overhead,
    and a 0-d load in one (after a warp reduction) measured slower moved."""
    return (
        isinstance(loop.iter, ast.Call)
        and ast.unparse(loop.iter.func) == "cutlass.range_constexpr"
    )


def _synchronizes_cta(statement: ast.stmt) -> bool:
    """Whether ``statement`` itself (not a statement nested in it) calls a
    barrier or a block or cluster reduction."""
    for node in _own_nodes(statement):
        if isinstance(node, ast.Call):
            name = ast.unparse(node.func)
            if name in _CTA_BARRIER_CALLS or name.startswith(
                _CTA_REDUCTION_CALL_PREFIXES
            ):
                return True
    return False


def _globals(body: list[ast.stmt], stored: Counter[str]) -> set[str]:
    """Names ``body`` reads and never assigns: the module's (``cutlass``,
    constexprs, helpers) and the kernel's parameters."""
    module = ast.Module(body=body, type_ignores=[])
    return {
        n.id for n in ast.walk(module) if isinstance(n, ast.Name) and stored[n.id] == 0
    }


# --------------------------------------------------------------------------
# Part 2: one-trip tile loops' packets
# --------------------------------------------------------------------------


def _int_value(node: ast.expr, constexpr_values: dict[str, int]) -> int | None:
    """The int ``node`` spells: a literal or constexpr name, bare or in
    ``cutlass.Int32(...)``."""
    if (
        isinstance(node, ast.Call)
        and ast.unparse(node.func) == "cutlass.Int32"
        and len(node.args) == 1
        and not node.keywords
    ):
        node = node.args[0]
    if isinstance(node, ast.Constant) and type(node.value) is int:
        return node.value
    if isinstance(node, ast.Name) and node.id in constexpr_values:
        value = constexpr_values[node.id]
        return value if type(value) is int else None
    return None


def _range(
    loop: ast.For, constexpr_values: dict[str, int]
) -> tuple[int, int, int] | None:
    """``(start, stop, step)`` of ``for _ in range(...)`` or
    ``cutlass.range_constexpr(...)`` over ints."""
    call = loop.iter
    if not (
        isinstance(call, ast.Call)
        and ast.unparse(call.func) in ("range", "cutlass.range_constexpr")
        and not call.keywords
        and 1 <= len(call.args) <= 3
        and not loop.orelse
        and isinstance(loop.target, ast.Name)
    ):
        return None
    values = [_int_value(a, constexpr_values) for a in call.args]
    if any(v is None for v in values):
        return None
    ints = [v for v in values if v is not None]
    if len(ints) == 1:
        return 0, ints[0], 1
    if len(ints) == 2:
        return ints[0], ints[1], 1
    return ints[0], ints[1], ints[2]


def _trips(bounds: tuple[int, int, int]) -> int:
    start, stop, step = bounds
    if step <= 0:
        return 0
    return max(0, -(-(stop - start) // step))


def _trip_counts(
    body: list[ast.stmt], constexpr_values: dict[str, int] | None = None
) -> dict[int, int]:
    counts = {}
    for statement in _statements(body):
        if isinstance(statement, ast.For):
            bounds = _range(statement, constexpr_values or {})
            if bounds is not None:
                counts[id(statement)] = _trips(bounds)
    return counts


@dataclass
class _Packet:
    """One ``NAME = cute.arch.load(pointer, vector_type)`` of a lane loop,
    under ``guard`` (the test of the ``if`` it heads, None when unguarded)."""

    statement: ast.Assign
    tensor: str
    pointer: ast.expr
    vector_type: ast.expr
    guard: ast.expr | None
    nbytes: int
    key: tuple[str, str, str]
    width: int
    carrier: str
    # The predicated load's priority argument (``CachePolicy.masked_priority``:
    # an L1 priority, "cs" for streaming, "" for none).
    priority: str = ""

    @property
    def keywords(self) -> list[ast.keyword]:
        call = self.statement.value
        assert isinstance(call, ast.Call)
        return call.keywords


@dataclass
class _LaneLoop:
    loop: ast.For
    tile: ast.For
    base: str
    lanes: int
    layout: str
    packets: list[_Packet] = field(default_factory=list)
    # Where the first lane loop of a layout loads the shared packets: before
    # this statement of its body.
    anchor: ast.stmt | None = None


@dataclass
class _Choice:
    """A packet the plan shares: its places (lane loop, load) and its pointer
    and guard spelled in the producer."""

    places: list[tuple[_LaneLoop, _Packet]]
    pointer: ast.expr
    guard: ast.expr | None


@dataclass
class _Plan:
    """The first lane loop of a layout (``producer``), the lane loops of that
    layout, and the packets they share."""

    producer: _LaneLoop
    members: list[_LaneLoop]
    choices: list[_Choice]


class _Canon(ast.NodeTransformer):
    """Spell an index expression by what it computes: names assigned once
    are replaced by their definitions, ``cutlass.Int32(e)`` by ``e``, and
    the names in ``fixed`` by the given expressions."""

    def __init__(
        self, definitions: dict[str, ast.expr], fixed: dict[str, ast.expr]
    ) -> None:
        self.definitions = definitions
        self.fixed = fixed
        self.depth = 0

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if node.id in self.fixed:
            return deep_clone(self.fixed[node.id])
        if node.id in self.definitions and self.depth < 32:
            self.depth += 1
            try:
                return self.visit(deep_clone(self.definitions[node.id]))
            finally:
                self.depth -= 1
        return node

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        if (
            ast.unparse(node.func) == "cutlass.Int32"
            and len(node.args) == 1
            and not node.keywords
        ):
            return node.args[0]
        return node


def _spell(
    node: ast.expr, definitions: dict[str, ast.expr], fixed: dict[str, ast.expr]
) -> str:
    return ast.unparse(_Canon(definitions, fixed).visit(deep_clone(node)))


# The most lanes a lane loop may have to share packets: a runtime lane loop
# is unrolled for them, and a body of up to 8 lanes keeps the code size of
# the normalize near what the rolled loop had in registers anyway.
MAX_UNROLLED_LANES = 8


def _cuts_loop(loop: ast.For) -> bool:
    """Whether a ``break`` or ``continue`` of ``loop`` itself sits in its
    body: such a loop cannot be inlined, and a statement after the exit need
    not run in every iteration.  One of a loop nested in it ends only that
    loop, unless it sits in the nested loop's ``else``, which runs in
    ``loop``."""

    def exits(node: ast.AST) -> bool:
        if isinstance(node, (ast.Break, ast.Continue)):
            return True
        if isinstance(node, (ast.For, ast.AsyncFor, ast.While)):
            return any(exits(child) for child in node.orelse)
        if isinstance(node, (ast.FunctionDef, ast.Lambda)):
            return False
        return any(exits(child) for child in ast.iter_child_nodes(node))

    return any(exits(statement) for statement in loop.body)


def _lane_loops(
    body: list[ast.stmt],
    constexpr_values: dict[str, int],
    definitions: dict[str, ast.expr],
    tensors: set[str],
    stable: set[str],
) -> list[_LaneLoop]:
    """The lane loops directly inside the top-level one-trip tile loops, in
    program order, with their packets."""
    found = []
    for tile in body:
        if (
            not isinstance(tile, ast.For)
            or _cuts_loop(tile)
            # A reduction loop over an rdim (``roffset_N``) keeps its packets
            # as ``cute_reduction_reloads`` asks (``fuse_two_pass_loads``,
            # ``prefetch_reduction_packets``): "gmem" reloads on purpose.
            or _CuteFuseTwoPassLoads._reduction_block_id(tile) is not None
        ):
            continue
        bounds = _range(tile, constexpr_values)
        if bounds is None or bounds[0] != 0 or _trips(bounds) != 1:
            continue
        assert isinstance(tile.target, ast.Name)
        for loop in tile.body:
            if not isinstance(loop, ast.For) or not loop.body:
                continue
            lane_bounds = _range(loop, constexpr_values)
            if (
                lane_bounds is None
                or lane_bounds[0] != 0
                or lane_bounds[2] != 1
                or lane_bounds[1] > MAX_UNROLLED_LANES
                or _cuts_loop(loop)
            ):
                continue
            first = loop.body[0]
            assert isinstance(loop.target, ast.Name)
            lane = loop.target.id
            if not (
                isinstance(first, ast.Assign)
                and len(first.targets) == 1
                and isinstance(first.targets[0], ast.Name)
                and lane in _names(first.value)
            ):
                continue
            fixed: dict[str, ast.expr] = {
                tile.target.id: ast.Constant(0),
                lane: ast.Name("LANE", ast.Load()),
            }
            base = first.targets[0].id
            layout = _spell(first.value, definitions, fixed)
            fixed[base] = cast("ast.expr", expr_from_string(layout))
            lane_loop = _LaneLoop(loop, tile, base, lane_bounds[1], layout)
            for statement in loop.body[1:]:
                if isinstance(statement, ast.If):
                    heads = []
                    for inner in statement.body:
                        packet = _packet(
                            inner, statement.test, tensors, definitions, fixed, stable
                        )
                        if packet is None:
                            break
                        heads.append(packet)
                    if heads and lane_loop.anchor is None:
                        lane_loop.anchor = statement
                    lane_loop.packets.extend(heads)
                else:
                    packet = _packet(
                        statement, None, tensors, definitions, fixed, stable
                    )
                    if packet is not None:
                        if lane_loop.anchor is None:
                            lane_loop.anchor = statement
                        lane_loop.packets.append(packet)
            found.append(lane_loop)
    return found


def _packet(
    statement: ast.stmt,
    guard: ast.expr | None,
    tensors: set[str],
    definitions: dict[str, ast.expr],
    fixed: dict[str, ast.expr],
    stable: set[str],
) -> _Packet | None:
    """The packet ``statement`` loads, keyed by what its pointer, vector type
    and guard compute (``_spell``).  None for any other statement, and for
    one whose key still reads a name that is not ``stable`` (a value
    assigned more than once, which may differ between two lane loops).  A
    guarded packet's cache hint goes to the predicated load's priority
    argument."""
    if not (
        isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
    ):
        return None
    access = match_packet_access(statement.value)
    if (
        access is None
        or access.kind != "packet"
        or access.width is None
        or access.carrier not in _ITEMSIZES
    ):
        return None
    policy = access.cache_policy()
    if access.keywords and policy is None:
        # Keywords that are no cache hint keep the packet's load.
        return None
    roots = _tensor_arg_roots(access.pointer, tensors)
    if roots is None or len(roots) != 1:
        return None
    nbytes = access.width * _ITEMSIZES[access.carrier]
    if nbytes not in MASKED_PACKET_LOAD_HELPERS:
        return None
    vector_type = access.vector_type
    assert vector_type is not None
    spelled = [_Canon(definitions, fixed).visit(deep_clone(access.pointer))]
    if guard is not None:
        spelled.append(_Canon(definitions, fixed).visit(deep_clone(guard)))
    if not all(_names(e) <= stable | {"LANE"} and _reads_no_memory(e) for e in spelled):
        return None
    keywords = ", ".join(ast.unparse(k) for k in access.keywords)
    key = (
        ast.unparse(spelled[0]),
        f"{ast.unparse(vector_type)}; {keywords}",
        "" if guard is None else ast.unparse(spelled[1]),
    )
    (tensor,) = roots
    return _Packet(
        statement,
        tensor,
        access.pointer,
        vector_type,
        guard,
        nbytes,
        key,
        access.width,
        access.carrier,
        "" if policy is None else policy.masked_priority,
    )


def _keep_one_trip_packets(
    body: list[ast.stmt],
    constexpr_values: dict[str, int],
    tensors: set[str],
    disjoint: set[frozenset[str]],
    merged: set[str],
    budget: int,
) -> list[ast.stmt]:
    stored = _stored_names(body)
    for name in merged:
        stored[name] += 2
    definitions = {
        s.targets[0].id: s.value
        for s in _statements(body)
        if isinstance(s, ast.Assign)
        and len(s.targets) == 1
        and isinstance(s.targets[0], ast.Name)
        and stored[s.targets[0].id] == 1
    }
    # Names with one value wherever they are read: the module's and the
    # kernel's (never assigned), and those assigned once by a top-level
    # statement.  A packet's key reads only these, besides its lane.
    stable = {
        s.targets[0].id
        for s in body
        if isinstance(s, ast.Assign)
        and len(s.targets) == 1
        and isinstance(s.targets[0], ast.Name)
        and stored[s.targets[0].id] == 1
    }
    stable |= _globals(body, stored)
    loops = _lane_loops(body, constexpr_values, definitions, tensors, stable)
    groups: dict[tuple[int, str], list[_LaneLoop]] = {}
    for lane_loop in loops:
        groups.setdefault((lane_loop.lanes, lane_loop.layout), []).append(lane_loop)
    writes = _writes(body, tensors)
    assert writes is not None
    order = _order(body)
    written = {r for w in writes for r in w.roots}
    plans: list[_Plan] = []
    for members in groups.values():
        producer = members[0]
        if len(members) < 2 or producer.anchor is None:
            continue
        uses: dict[tuple[str, str, str], list[tuple[_LaneLoop, _Packet]]] = {}
        for member in members:
            for packet in member.packets:
                uses.setdefault(packet.key, []).append((member, packet))
        chosen: list[_Choice] = []
        for _, places in sorted(
            uses.items(), key=lambda kv: (-len(kv[1]), order[id(kv[1][0][1].statement)])
        ):
            if len(places) < 2 and places[0][0] is producer:
                continue
            packet = places[0][1]
            # Read-only, and apart from every tensor the kernel writes: no
            # thread of any CTA writes what the packet holds, whenever it
            # loads.
            if packet.tensor in written or not all(
                _disjoint(packet.tensor, w.roots, disjoint) for w in writes
            ):
                continue
            cost = producer.lanes * packet.nbytes
            if cost > budget:
                continue
            source = _at_producer(producer, places, definitions, stored, body, order)
            if source is None:
                continue
            budget -= cost
            chosen.append(_Choice(places, *source))
        if chosen:
            plans.append(_Plan(producer, members, chosen))
    if not plans:
        return body
    taken = _stored_names(body) | Counter(
        _names(ast.Module(body=body, type_ignores=[]))
    )
    serial = 0
    tiles: set[int] = set()
    for plan in plans:
        producer, members = plan.producer, plan.members
        assert producer.anchor is not None
        cache_lists = []
        for choice in plan.choices:
            places, pointer, guard = choice.places, choice.pointer, choice.guard
            while (
                f"_one_trip_packets_{serial}" in taken
                or f"_one_trip_packets_{serial}_lane" in taken
            ):
                serial += 1
            cache = f"_one_trip_packets_{serial}"
            taken[cache] += 1
            cache_lists.append(cache)
            packet = places[0][1]
            helper = MASKED_PACKET_LOAD_HELPERS[packet.nbytes]
            vector_type = ast.unparse(packet.vector_type)
            if guard is None:
                keywords = "".join(f", {ast.unparse(k)}" for k in packet.keywords)
                load = (
                    f"cute.arch.load({ast.unparse(pointer)}, {vector_type}{keywords})"
                )
            else:
                load = render_masked_packet_load(
                    helper,
                    ast.unparse(pointer),
                    ast.unparse(guard),
                    packet.width,
                    packet.carrier,
                    packet.priority,
                )
            index = producer.loop.body.index(producer.anchor)
            # The lane's packet under a name of its own, then into the list:
            # a load stays a ``NAME = load(...)`` statement, as everywhere.
            producer.loop.body[index:index] = [
                statement_from_string(f"{cache}_lane = {load}"),
                statement_from_string(f"{cache}.append({cache}_lane)"),
            ]
            for member, used in places:
                assert isinstance(member.loop.target, ast.Name)
                used.statement.value = cast(
                    "ast.expr", expr_from_string(f"{cache}[{member.loop.target.id}]")
                )
        position = body.index(producer.tile)
        for cache in cache_lists:
            body.insert(position, statement_from_string(f"{cache} = []"))
            position += 1
        used = {id(producer)} | {
            id(m) for choice in plan.choices for m, _ in choice.places
        }
        for member in members:
            if id(member) not in used:
                continue
            tiles.add(id(member.tile))
            call = member.loop.iter
            assert isinstance(call, ast.Call)
            if ast.unparse(call.func) == "range":
                member.loop.iter = cast(
                    "ast.expr",
                    expr_from_string(f"cutlass.range_constexpr({member.lanes})"),
                )
    inlined: list[ast.stmt] = []
    for statement in body:
        if isinstance(statement, ast.For) and id(statement) in tiles:
            assert isinstance(statement.target, ast.Name)
            inlined.append(
                statement_from_string(f"{statement.target.id} = cutlass.Int32(0)")
            )
            inlined.extend(statement.body)
        else:
            inlined.append(statement)
    return [ast.fix_missing_locations(s) for s in inlined]


def _at_producer(
    producer: _LaneLoop,
    places: list[tuple[_LaneLoop, _Packet]],
    definitions: dict[str, ast.expr],
    stored: Counter[str],
    body: list[ast.stmt],
    order: dict[int, int],
) -> tuple[ast.expr, ast.expr | None] | None:
    """The packet's pointer and guard spelled where the first lane loop of its
    layout loads the shared packets (before its anchor): the producer's own
    load when it has one, else the first place's with the producer's lane and
    base names; either way the names not yet bound there are expanded into
    their definitions, and None when a name stays out of reach (a name the
    producer binds after its anchor, or one assigned more than once)."""
    member, packet = next(((m, p) for m, p in places if m is producer), places[0])
    assert isinstance(member.loop.target, ast.Name)
    assert isinstance(producer.loop.target, ast.Name)
    assert isinstance(member.tile.target, ast.Name)
    fixed: dict[str, ast.expr] = (
        {}
        if member is producer
        else {
            member.loop.target.id: cast(
                "ast.expr", expr_from_string(producer.loop.target.id)
            ),
            member.base: cast("ast.expr", expr_from_string(producer.base)),
            member.tile.target.id: cast(
                "ast.expr", expr_from_string("cutlass.Int32(0)")
            ),
        }
    )
    anchor = order[id(producer.anchor)]
    visible = {
        s.targets[0].id
        for s in [*body, *producer.tile.body, *producer.loop.body]
        if isinstance(s, ast.Assign)
        and len(s.targets) == 1
        and isinstance(s.targets[0], ast.Name)
        and stored[s.targets[0].id] == 1
        and order.get(id(s), anchor) < anchor
    } | {producer.base, producer.loop.target.id}

    class Expand(ast.NodeTransformer):
        failed = False

        def visit_Name(self, node: ast.Name) -> ast.AST:
            if node.id in fixed:
                return deep_clone(fixed[node.id])
            if node.id in visible or (stored[node.id] == 0):
                return node
            if node.id in definitions:
                return self.visit(deep_clone(definitions[node.id]))
            self.failed = True
            return node

    expand = Expand()
    pointer = expand.visit(deep_clone(packet.pointer))
    guard = None if packet.guard is None else expand.visit(deep_clone(packet.guard))
    if expand.failed:
        return None
    return cast("ast.expr", pointer), (
        None if guard is None else cast("ast.expr", guard)
    )
