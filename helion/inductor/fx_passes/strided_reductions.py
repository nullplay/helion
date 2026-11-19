"""The rewrite of reductions over dims that memory keeps apart, which
HelionPostGradPass applies: BatchNorm's var_mean over N, H and W of an NCHW
tensor reduces H and W, which are adjacent and contiguous, and N, which C
separates from them. Inductor lowers it to one flat reduction dim, and the
first kernel of its split reduction over [32, 256, 56, 56] reads

    for tile_y, tile_x in hl.tile([4, 256]):
        for tile_r in hl.tile(25088):
            tmp0 = in0[3136*tile_x.index.reshape([tile_x, 1])
                + 802816*(tile_r.index // 3136)
                + 6422528*tile_y.index.reshape([tile_y, 1, 1])
                + ((tile_r.index % 3136))]

with a division and a remainder in each index, which keep the loads from
being vectors. Reduced first over H and W, the rows of 3136 are contiguous:

    for tile_y, tile_x in hl.tile([32, 256]):
        for tile_r in hl.tile(3136):
            tmp0 = in0[tile_y, tile_x, tile_r]

and a second reduction over N combines each channel's 32 partials."""

from __future__ import annotations

import math
import operator
from typing import TYPE_CHECKING
from typing import Any

import torch
from torch.fx.operator_schemas import normalize_function

from .graph import insert_call

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Container
    from collections.abc import Sequence

    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.fx.node import Argument

aten = torch.ops.aten

# The fewest values in a row of the first reduction: the partials it writes,
# a float32 per row for a sum, cost what the row costs to read past that.
MIN_INNER_RUN = 32

REDUCTIONS = (aten.sum.dim_IntList, aten.mean.dim, aten.var_mean.correction)

HALF_DTYPES = (torch.float16, torch.bfloat16)


def split_reduction_dims(
    value: torch.Tensor, dims: Sequence[int]
) -> tuple[list[int], list[int]] | None:
    """The innermost reduced dims that are contiguous in memory, the first
    reduction's, and the other reduced dims, the second's; None when the
    reduced dims are one contiguous run, or the innermost dim is kept, as in
    var_mean over N, H and W of an NHWC tensor (one flat reduction dim of
    stride C), or the run is shorter than MIN_INNER_RUN. Dims of size 1 do not
    count."""
    sizes, strides = value.shape, value.stride()
    if not all(isinstance(n, int) for n in (*sizes, *strides)):
        return None
    reduced = {d % value.dim() for d in dims}
    order = sorted(
        (d for d in range(value.dim()) if sizes[d] != 1),
        key=lambda d: strides[d],
        reverse=True,
    )
    inner: list[int] = []
    for dim in reversed(order):
        if dim not in reduced or (
            inner and strides[dim] != strides[inner[-1]] * sizes[inner[-1]]
        ):
            break
        inner.append(dim)
    outer = sorted(d for d in reduced if d not in inner and sizes[d] != 1)
    if (
        not inner
        or not outer
        or strides[inner[0]] != 1
        or math.prod(sizes[d] for d in inner) < MIN_INNER_RUN
    ):
        return None
    return sorted(inner), outer


def split_strided_reductions(
    graph: torch.fx.Graph, devices: Container[str] = ("cuda",)
) -> bool:
    """Rewrites each sum, mean and var_mean whose dims split_reduction_dims()
    splits into a reduction over the inner dims, with keepdim, and one over the
    outer dims of its result. Every row of the first has as many values, so
    the mean of the means is the mean, and the variance is the mean of the
    rows' variances plus the variance of their means. A float16 or bfloat16
    reduction keeps its partials in float32, as Inductor's split reductions
    do, and rounds once at the end: a sum of halves of 2048 and -2048 over a
    float16 [2, 3, 8, 8] is 0, where float16 partials of +-131072 are inf.
    Returns whether it rewrote one."""
    changed = False
    for node in list(graph.nodes):
        if node.op != "call_function" or node.target not in REDUCTIONS:
            continue
        normalized = normalize_function(
            node.target,
            node.args,
            node.kwargs,
            normalize_to_only_use_kwargs=True,
        )
        assert normalized is not None
        kwargs = dict(normalized.kwargs)
        x, dims = kwargs.pop("input"), kwargs.pop("dim")
        value = x.meta["val"]
        # The reductions Helion's kernels compute, of CUDA tensors; an empty
        # or 0-d one has no rows to split.
        if (
            value.device.type not in devices
            or not all(isinstance(n, int) for n in value.shape)
            or value.numel() == 0
            or value.dim() == 0
        ):
            continue
        if dims is None or (split := split_reduction_dims(value, dims)) is None:
            continue
        rewrite = Rewrite(
            graph,
            node,
            value.fake_mode,
            *split,
            sorted({d % value.dim() for d in dims}),
            kwargs.pop("keepdim"),
        )
        if node.target is aten.var_mean.correction:
            if not value.dtype.is_floating_point or not rewrite.var_mean(
                x, value, kwargs.get("correction")
            ):
                continue
        else:
            out_dtype = node.meta["val"].dtype
            if out_dtype in HALF_DTYPES:
                kwargs["dtype"] = torch.float32
            first = rewrite.call(node.target, x, rewrite.inner, True, **kwargs)
            second = rewrite.call(node.target, first, rewrite.outer, True, **kwargs)
            node.replace_all_uses_with(rewrite.finish(rewrite.cast(second, out_dtype)))
        graph.erase_node(node)
        changed = True
    return changed


class Rewrite:
    """The nodes that replace `node`, a reduction over `dims`, `inner` and
    `outer` of them reduced in turn (dims of size 1 in neither), inserted before
    it with their meta["val"] from `fake_mode`."""

    def __init__(
        self,
        graph: torch.fx.Graph,
        node: torch.fx.Node,
        fake_mode: FakeTensorMode,
        inner: list[int],
        outer: list[int],
        dims: list[int],
        keepdim: bool,
    ) -> None:
        self.graph = graph
        self.node = node
        self.fake_mode = fake_mode
        self.inner = inner
        self.outer = outer
        self.dims = dims
        self.keepdim = keepdim

    def call(
        self, target: Callable[..., Any], *args: Argument, **kwargs: Argument
    ) -> torch.fx.Node:
        return insert_call(
            self.graph, self.node, self.fake_mode, target, *args, **kwargs
        )

    def cast(self, result: torch.fx.Node, dtype: torch.dtype) -> torch.fx.Node:
        """`result` in `dtype`, as the reduction returned it."""
        if result.meta["val"].dtype == dtype:
            return result
        return self.call(torch.ops.prims.convert_element_type.default, result, dtype)

    def finish(self, result: torch.fx.Node) -> torch.fx.Node:
        """`result`, with keepdim, as the reduction returned it: without every
        reduced dim, those of size 1 too, unless it keeps them."""
        if self.keepdim:
            return result
        return self.call(aten.squeeze.dims, result, self.dims)

    def var_mean(
        self, x: torch.fx.Node, value: torch.Tensor, correction: float | None
    ) -> bool:
        """var_mean over the inner dims, then the mean over the outer dims of
        the rows' means, corrected by their mean deviation from it (of rows of
        100553424, whose mean rounds to 100553432, the correction is -8), and
        of their variances plus the variance of their means: the mean of their
        squared deviations from the corrected mean less the square of their
        mean deviation, as the corrected mean can round as well (of rows of
        1e8 and 1e8 + 8, to 1e8, leaving deviations 0 and 8, of variance 32 -
        16), all of the values less one of each output's own, which keeps the
        rows' float32 means at the spread's scale; the mean returned is the
        unshifted values', corrected the same way. False, rewriting nothing,
        when the correction leaves no degrees of freedom."""
        correction = 1 if correction is None else correction
        count = math.prod(value.shape[d] for d in self.inner + self.outer)
        if count <= correction:
            return False
        call, outer, dtype = self.call, self.outer, value.dtype
        if dtype in HALF_DTYPES:
            x = self.cast(x, torch.float32)
        rows_per_mean = math.prod(value.shape[d] for d in outer)
        # Shifted data: the values less one of each output's own, its first
        # (x[0, c, 0, 0] of BatchNorm's channel c), near the mean wherever
        # |mean| dwarfs the spread, so that the rows' float32 means round at
        # the spread's scale for the variance between them: rows of 1e8 and
        # 1e8 + 8 average 4, where 1e8 + 4 would round to 1e8 and lose the
        # rows' spread.
        shift = x
        for dim in self.dims:
            shift = call(aten.slice.Tensor, shift, dim, 0, 1)
        shifted = call(aten.sub.Tensor, x, shift)

        def mean(t: torch.fx.Node) -> torch.fx.Node:
            # Each term scaled before the sum, as Welford's means are, so
            # that rows of 1e37 do not sum to inf.
            scaled = call(aten.mul.Tensor, t, 1 / rows_per_mean)
            return call(aten.sum.dim_IntList, scaled, outer, True)

        # Each row's variance and mean, Welford's (sound whatever the mean),
        # and its mean less the shift, of the same rows in the same kernel,
        # whose spread over the rows is the variance between them.
        rows = call(aten.var_mean.correction, x, self.inner, correction=0, keepdim=True)
        row_var = call(operator.getitem, rows, 0)
        plain = call(operator.getitem, rows, 1)
        per_row = math.prod(value.shape[d] for d in self.inner)
        row_mean = call(
            aten.sum.dim_IntList,
            call(aten.mul.Tensor, shifted, 1 / per_row),
            self.inner,
            True,
        )
        overall = mean(row_mean)
        drift = mean(call(aten.sub.Tensor, row_mean, overall))
        mean_ = call(aten.add.Tensor, overall, drift)
        # The mean returned is the rows' unshifted one, corrected likewise: a
        # shift far from the rest, an outlier first value of 1e8 among ones,
        # rounds them away in x - shift, where the variance does not miss
        # them.
        plain_mean = mean(plain)
        plain_mean = call(
            aten.add.Tensor,
            plain_mean,
            mean(call(aten.sub.Tensor, plain, plain_mean)),
        )
        # The deviations from mean_, which can round too, less their mean,
        # each scaled before it is squared: rows of 1e20 and 0 square to
        # 1e40, past float32's largest value.
        centered = call(aten.sub.Tensor, row_mean, mean_)
        offset = mean(centered)
        scaled = call(aten.mul.Tensor, centered, rows_per_mean**-0.5)
        between = call(
            aten.sub.Tensor,
            call(
                aten.sum.dim_IntList,
                call(aten.mul.Tensor, scaled, scaled),
                outer,
                True,
            ),
            call(aten.mul.Tensor, offset, offset),
        )
        var = call(
            aten.add.Tensor,
            mean(row_var),
            call(aten.clamp_min.default, between, 0.0),
        )
        if correction != 0:
            var = call(aten.mul.Tensor, var, count / (count - correction))
        replacements = {
            0: self.finish(self.cast(var, dtype)),
            1: self.finish(self.cast(plain_mean, dtype)),
        }
        for user in list(self.node.users):
            index = user.args[1]
            assert user.target is operator.getitem and isinstance(index, int)
            user.replace_all_uses_with(replacements[index])
            self.graph.erase_node(user)
        return True
