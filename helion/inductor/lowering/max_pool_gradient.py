"""The gradient of a max pool of non-overlapping windows as a gather.

torch._decomp's max_pool2d_with_indices_backward computes a max pool's
gradient as a scatter_add: the input's gradient starts as zeros, and each
pooled position adds its gradient at the input position its window took the
max from. The torch nightlies of this backend (2.16.0.dev20261118 on) do so
for padded, overlapping or dilated pools, and gather for the others (the
decomposition's nonoverlap branch); graphs traced with earlier ones, as
better-benchmark's repros, scatter for every pool. With Inductor's
low-memory max pool, the input position comes from the window's offset:

    full = torch.full([32768, 784], 0.0)                    # [N * C, H * W]
    indices = _low_memory_max_pool_offsets_to_indices(offsets, [2, 2], [28, 28], [2, 2], [0, 0], [1, 1])
    grad = torch.scatter_add(full, 1, indices.view(32768, 196), grad_out.view(32768, 196).float())

(better-benchmark's sum_cf864ca94168, a max pool of VGG16 in training, N =
64, C = 512). Compiled as written, that is a kernel filling the zeros, a
kernel adding into them with an atomic per pooled position, and the consumers
reading them back.

Where the windows do not overlap (stride >= kernel, as VGG's 2 x 2 pools),
each input position lies in one window at most, so its gradient is a gather
instead: its window's gradient where the window's offset is the position's
own offset in it, else 0 (window_gather()). The consumers then compute it in
their own kernel (the ReLU backward's where and the bias gradient's sum),
without the zeros, the atomics or the fp32 buffer (VGG16's [64, 512, 28, 28],
bf7c8e5d):

    for tile_y, tile_x in hl.tile([256, 512]):
        acc = hl.zeros([tile_y, tile_x], dtype=torch.float32)
        for tile_r in hl.tile(196):
            tmp0 = in0[tile_y, tile_r, tile_x]                   # the ReLU's mask
            tmp2 = in2[512*(((tile_r.index... // 2) % 14)) + ... + tile_x.index]
            tmp12 = in3[...the same...].to(torch.float32)        # its window's gradient
            tmp3 = tmp2.to(torch.int32)                          # its window's offset
            ...
            tmp9 = (2*(((... // 28) % 2)) + ((tile_r.index % 2))).to(torch.int32)
            tmp11 = tmp8 == tmp10                                # the window's max?
            tmp14 = torch.where(tmp11, tmp12, tmp13)
            ...
            tmp16 = torch.where(tmp0, tmp1, tmp15)
            out0[tile_y, tile_r, tile_x] = tmp16.to(torch.bfloat16)
            acc = acc + torch.sum(tmp16, -2)
        out1[tile_y, tile_x] = acc

Inductor's own max_pool2d_with_indices_backward lowering gathers the same
values, but clamps each window index to the pool (torch.minimum of each with
the window's last one) and decodes each offset into the input position it
names, in int64, to compare it with the position's own: at VGG16's [64, 256,
56, 56] (b3ea8ee2) its kernel took 127.5 us, 94.8 without the clamps, where
window_gather()'s takes 50.2 (GB300, each after a quick search, cold). The
lowering takes window_gather() where the offsets are an input of the graph
(the saved offsets a backward graph reads), and Inductor's lowering over the
decoded indices for other offsets (the forward's own, in a graph that pools
and backpropagates).

Overlapping windows (AlexNet's and SqueezeNet's kernel 3, stride 2) put an
input position in up to four windows, each a gathered offset and gradient
with its own index arithmetic: such a gather took 237 us at AlexNet's [128,
64, 55, 55], against 81 us for the whole scatter graph, so the lowering
leaves them to the scatter.

So the lowering takes the padded non-overlapping pools of today's graphs
(kernel 3, stride 3, padding 1) and, in graphs traced before the
decomposition gathered, unpadded ones too. It takes the gather only for
exactly that form: a float scatter_add over dim 1 of +0.0 [P, H * W], which
nothing writes into before it, at the indices of a 2-D max pool's offsets
(prims._low_memory_max_pool_offsets_to_indices over an input of H x W,
through a contiguous clone, views and unsqueezes as [P, PH * PW], P the
pool's leading dims), which nothing writes into after the pool decodes them,
with static sizes, no dilation and windows that do not overlap; it declines
everything else, which Inductor scatters as before. Each input position takes
+0.0 plus one value or plus 0, as the scatter added that value to +0.0."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING
from typing import Any
from typing import cast

import sympy
import torch
from torch._inductor import config as inductor_config
from torch._inductor import lowering as inductor_lowering
from torch._inductor.ir import Pointwise
from torch._inductor.ir import TensorBox
from torch._inductor.virtualized import V
from torch._inductor.virtualized import ops
from torch.utils._sympy.functions import FloorDiv
from torch.utils._sympy.functions import ModularIndexing

from .transposed_pads import _writes_between
from .transposed_pads import _written

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Sequence

aten = torch.ops.aten
prims = torch.ops.prims

SCATTER_ADD = aten.scatter_add.default
OFFSETS_TO_INDICES = prims._low_memory_max_pool_offsets_to_indices.default
# The ops between the pool's indices and the scatter_add's index that keep
# its values in order: a contiguous clone, views of it, and the unsqueeze of
# torch._decomp's unbatched [C, H, W] form.
VIEWS = (
    aten.clone.default,
    aten.view.default,
    aten._unsafe_view.default,
    aten.reshape.default,
    aten.unsqueeze.default,
)


def static_ints(values: object) -> list[int] | None:
    """`values` as Python ints where each is a constant (an int, or a sympy
    Integer of an IR node's size), else None."""
    if not isinstance(values, (list, tuple)) or not all(
        isinstance(v, (int, sympy.Integer)) and not isinstance(v, bool) for v in values
    ):
        return None
    return [int(v) for v in values]


def exprs(*values: int) -> list[sympy.Expr]:
    """Sizes as the sympy integers Inductor's view lowering takes."""
    return [sympy.Integer(v) for v in values]


def pool_of(index: object) -> torch.fx.Node | None:
    """The offsets_to_indices node `index` is a view of, through VIEWS."""
    while isinstance(index, torch.fx.Node) and index.target in VIEWS:
        index = index.args[0]
    if isinstance(index, torch.fx.Node) and index.target is OFFSETS_TO_INDICES:
        return index
    return None


def window_count(kernel_size: list[int], stride: list[int]) -> int:
    """The most windows that hold one input position: Inductor's
    max_pool2d_with_indices_backward's h_window_size * w_window_size, 1 where
    the windows do not overlap."""
    count = 1
    for k, s in zip(kernel_size, stride, strict=True):
        count *= max(max(h // s - max(0, (h - k) // s), 1) for h in range(k * 2))
    return count


def max_pool_gradient(
    x: object, dim: object, index: object, src: object
) -> TensorBox | None:
    """The gather of scatter_add(x, dim, index, src) where it is a max pool's
    gradient (the module docstring), or None."""
    if inductor_config.cuda_backend != "helion":
        return None
    node = V.graph.current_node
    if not (
        isinstance(x, TensorBox)
        and isinstance(index, TensorBox)
        and isinstance(src, TensorBox)
        and (device := x.get_device()) is not None
        and device.type == "cuda"
        and node is not None
        and node.target is SCATTER_ADD
        and len(node.args) == 4
        and not node.kwargs
    ):
        return None
    fill, _, index_node, _ = node.args
    pool = pool_of(index_node)
    # The gather never reads x: it must be the zeros the decomposition makes,
    # +0.0 (a -0.0 the scatter keeps where no window adds), and nothing may
    # write into its buffer before the scatter_add (a user Triton kernel's
    # write in place, which Inductor's reinplacing points at the full's
    # buffer). Nor into the indices after the pool decodes them: the gather
    # looks for each input position in its own window alone, where the
    # scatter adds wherever an index points. A user Triton kernel that swaps
    # two windows' indices in place before the full (a
    # triton_kernel_wrapper_mutation of the indices' buffer after
    # reinplacing) makes the scatter add each gradient at the other window's
    # max, which the gather would never read: the two gradients would be
    # lost. (A setitem into the indices stays a select_scatter, which
    # pool_of() does not see through.) A float scatter only: Inductor's
    # gather starts its sum at a float 0.0, which would add an integer src in
    # float32.
    if not (
        isinstance(fill, torch.fx.Node)
        and fill.target is aten.full.default
        and len(fill.args) >= 2
        and positive_zero(fill.args[1])
        and x.get_dtype().is_floating_point
        and src.get_dtype() == x.get_dtype()
        and not _writes_between(fill, node)
        and ((name := x.maybe_get_name()) is None or not _written([name]))
        and pool is not None
        and len(pool.args) == 6
        and not _writes_between(pool, node)
        and ((name := index.maybe_get_name()) is None or not _written([name]))
    ):
        return None
    offsets, *params = pool.args
    kernel_size, input_size, stride, padding, dilation = (
        static_ints(p) for p in params
    )
    sizes = static_ints(list(x.get_size()))
    index_sizes = static_ints(list(index.get_size()))
    src_sizes = static_ints(list(src.get_size()))
    if (
        kernel_size is None
        or input_size is None
        or stride is None
        or padding is None
        or dilation is None
        or sizes is None
        or index_sizes is None
        or src_sizes is None
        or not isinstance(offsets, torch.fx.Node)
        or not isinstance(offsets_value := offsets.meta.get("val"), torch.Tensor)
        or len(kernel_size) != 2
        or any(d != 1 for d in dilation)
        or window_count(kernel_size, stride) != 1
        or dim not in (1, -1)
        or len(sizes) != 2
        or index_sizes != src_sizes
    ):
        return None
    height, width = input_size
    shape = static_ints(list(offsets_value.shape))
    planes = sizes[0]
    if (
        shape is None
        or len(shape) not in (3, 4)
        or math.prod(shape[:-2]) != planes
        or sizes != [planes, height * width]
        or index_sizes != [planes, shape[-2] * shape[-1]]
    ):
        return None
    # The pool's own [N, C, PH, PW] (an unbatched [C, PH, PW] as [1, C, PH,
    # PW]): its planes' gradients in the pool's dims, the input's [N, C, H, W]
    # that the consumers read back.
    lead, pooled = ([1] * (4 - len(shape)) + shape[:-2]), shape[-2:]
    view = inductor_lowering.view
    # The offsets as the graph's input (a max pool's saved offsets, in the
    # backward graph that reads them), which window_gather() reads rather than
    # the indices they decode to; other offsets (the forward's own, in a
    # graph that pools and backpropagates) take Inductor's lowering over the
    # decoded indices, which clamps each window index and compares the
    # decoded index with the position's own.
    lowered = (
        V.graph.graph_inputs.get(str(offsets.target))
        if offsets.op == "placeholder"
        else None
    )
    if isinstance(lowered, TensorBox):
        gradient = window_gather(
            view(src, exprs(*lead, *pooled)),
            view(lowered, exprs(*lead, *pooled)),
            [*lead, height, width],
            kernel_size,
            stride,
            padding,
        )
    else:
        gradient = inductor_lowering.lowerings[aten.max_pool2d_with_indices_backward](
            view(src, exprs(*lead, *pooled)),
            view(x, exprs(*lead, height, width)),
            kernel_size,
            stride,
            padding,
            dilation,
            False,
            view(index, exprs(*lead, *pooled)),
        )
        # +0.0 + g, as the scatter added g into +0.0: a -0.0 gradient comes
        # out +0.0 there, and here too (the gather alone would keep its sign).
        gradient = inductor_lowering.lowerings[aten.add.Tensor](gradient, 0.0)
    return view(gradient, exprs(*sizes))


def window_gather(
    src: TensorBox,
    offsets: TensorBox,
    sizes: list[int],
    kernel_size: list[int],
    stride: list[int],
    padding: list[int],
) -> TensorBox:
    """The gradient at each input position of `sizes` ([N, C, H, W]) of a max
    pool of windows that do not overlap, from its pooled gradient `src` and
    the offsets of its maxima within their windows, both [N, C, PH, PW]: the
    gradient of the position's window where the window's offset is the
    position's own offset in it, else 0, plus +0.0 (as the scatter added it
    into +0.0: a -0.0 gradient comes out +0.0). The window of input row h is
    (h + padding) // stride, and its offset there (h + padding) % stride, so
    a 2 x 2 pool's gradient reads its window's offset and gradient at the
    position's own // 2, with no other index arithmetic,

        tmp2 = in2[... 512*((tile_r.index // 2) % 14) + ... + tile_x.index]
        tmp8 = in3[... same ...]
        tmp12 = tmp2.to(torch.int32) == 2*(... % 2) + (tile_r.index % 2)

    where Inductor's max_pool2d_with_indices_backward lowering clamps both
    window indices (torch.minimum of each with the window's last one, which
    no 2 x 2 pool of an even input needs) and decodes each offset into the
    input position it names, in int64, to compare it with the position's
    own. Positions past the last window of a dim (ceil mode over an odd
    input, a padded pool's) or between windows (stride > kernel) are in no
    window: their windows' indices are clamped, and they take 0."""
    (kernel_h, kernel_w), (stride_h, stride_w) = kernel_size, stride
    *_, pooled_h, pooled_w = (int(size) for size in src.get_size())
    *_, height, width = sizes
    dtype = src.get_dtype()
    grad_loader = src.make_loader()
    offset_loader = offsets.make_loader()
    window = kernel_h * kernel_w
    # Every input position in a window, each window index within the pool.
    exact = (
        padding == [0, 0]
        and stride == kernel_size
        and height == pooled_h * stride_h
        and width == pooled_w * stride_w
    )

    def expr(value: object) -> sympy.Expr:
        # sympy's functions (FloorDiv, ModularIndexing, Min) build Exprs.
        return cast("sympy.Expr", value)

    def inner_fn(idx: Sequence[sympy.Expr]) -> object:
        *prefix, h, w = idx
        rows = expr(sympy.Add(h, sympy.Integer(padding[0])))
        cols = expr(sympy.Add(w, sympy.Integer(padding[1])))
        window_h = expr(FloorDiv(rows, stride_h))
        window_w = expr(FloorDiv(cols, stride_w))
        at_h = expr(ModularIndexing(rows, 1, stride_h))
        at_w = expr(ModularIndexing(cols, 1, stride_w))
        if exact:
            at = [*prefix, window_h, window_w]
        else:
            at = [
                *prefix,
                expr(sympy.Min(window_h, pooled_h - 1)),
                expr(sympy.Min(window_w, pooled_w - 1)),
            ]
        offset = ops.to_dtype(offset_loader(at), torch.int32)
        # An offset below 0 names window + offset, as Inductor's decoding reads
        # it (ops.indirect_indexing's wrap of an int8 offset of a large pool).
        offset = ops.where(
            ops.lt(offset, ops.constant(0, torch.int32)),
            ops.add(offset, ops.constant(window, torch.int32)),
            offset,
        )
        own = expr(sympy.Add(sympy.Mul(at_h, sympy.Integer(kernel_w)), at_w))
        hit = ops.eq(offset, ops.index_expr(own, torch.int32))
        if not exact:
            for position, bound in (
                (at_h, kernel_h),
                (at_w, kernel_w),
                (window_h, pooled_h),
                (window_w, pooled_w),
            ):
                inside = ops.lt(
                    ops.index_expr(position, torch.int32),
                    ops.constant(bound, torch.int32),
                )
                hit = ops.and_(hit, inside)
        zero = ops.constant(0.0, dtype)
        return ops.add(ops.where(hit, grad_loader(at), zero), zero)

    return Pointwise.create(
        device=src.get_device(),
        dtype=dtype,
        inner_fn=inner_fn,
        ranges=exprs(*sizes),
    )


def positive_zero(value: object) -> bool:
    """Whether a full's fill is 0 or +0.0 (not -0.0, nor False)."""
    if isinstance(value, bool):
        return False
    if isinstance(value, float):
        return value == 0.0 and math.copysign(1.0, value) > 0
    return isinstance(value, int) and value == 0


def lowering_of_max_pool_gradient(
    replaced: Callable[..., Any] | None,
) -> Callable[..., Any]:
    """The user lowering of aten.scatter_add: the replaced entry's, else
    max_pool_gradient()'s gather, else None (Inductor's scatter)."""

    def lowering(*args: object, **kwargs: object) -> object:
        out = replaced(*args, **kwargs) if replaced is not None else None
        if out is None and len(args) == 4 and not kwargs:
            out = max_pool_gradient(*args)
        return out

    return lowering


def register_max_pool_gradient_lowering() -> None:
    """Register lowering_of_max_pool_gradient() for aten.scatter_add, once:
    registering again (helion.inductor._register() runs again in tests)
    keeps the one wrapper."""
    current = inductor_lowering.user_lowerings.get(SCATTER_ADD)
    if getattr(current, _MAX_POOL_GRADIENT_WRAPPER, False):
        return
    lowering = lowering_of_max_pool_gradient(current)
    setattr(lowering, _MAX_POOL_GRADIENT_WRAPPER, True)
    inductor_lowering.user_lowerings[SCATTER_ADD] = lowering


# Marks the user lowering register_max_pool_gradient_lowering() installed.
_MAX_POOL_GRADIENT_WRAPPER = "_helion_max_pool_gradient_lowering"
