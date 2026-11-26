from __future__ import annotations

from types import SimpleNamespace
import unittest

import pytest
import sympy
import torch
from torch._inductor.codegen.common import CSEVariable
from torch._inductor.codegen.simd_kernel_features import SIMDKernelFeatures
from torch._inductor.graph import GraphLowering
from torch._inductor.ir import FixedLayout
from torch._inductor.ir import InputBuffer
from torch._inductor.utils import sympy_index_symbol
from torch._inductor.virtualized import V
from torch.utils._ordered_set import OrderedSet
from torch.utils._sympy.functions import FloorDiv
from torch.utils._sympy.functions import ModularIndexing
from torch.utils._sympy.value_ranges import ValueRanges

from helion._compat import requires_torch_version
from helion._testing import TestCase

if not requires_torch_version("2.16"):
    pytest.skip("helion.inductor needs torch 2.16", allow_module_level=True)

from helion.inductor.codegen.kernel import HelionKernel

pytestmark = pytest.mark.cpu_only


def features(numel: int) -> SIMDKernelFeatures:
    """The features of a pointwise kernel of `numel` elements and no nodes."""
    return SIMDKernelFeatures([], sympy.Integer(numel))


class TestIndexWidening(TestCase):
    def test_sum_without_values(self):
        # 2**30*x + 2**30*y + (2**30 + 1)*tmp0 prints its terms without the
        # value first: each fits int32, but their sum, taken in int32 before the
        # int64 value joins, reaches 2**31, so the tile indices widen.
        graph = GraphLowering(torch.fx.symbolic_trace(lambda x: x + 1))
        with V.set_graph_handler(graph):
            kernel = HelionKernel({"y": 2, "x": 2}, features=features(4))
            with V.set_kernel_handler(kernel):
                value = kernel.cse.generate(
                    kernel.compute, "src", dtype=torch.int64, shape=("tile_y", "tile_x")
                )
                tmp = sympy_index_symbol(str(value))
                kernel.indirect[tmp] = value
                x, y = kernel.symbols["x"], kernel.symbols["y"]
                index = 2**30 * x + 2**30 * y + (2**30 + 1) * tmp
                self.assertEqual(
                    kernel.index_tensor_expr(index, ["y", "x"]),
                    "1073741824*tile_x.index.to(torch.int64)"
                    " + 1073741824*tile_y.index.reshape([tile_y, 1]).to(torch.int64)"
                    " + 1073741825*tmp0",
                )

    def test_scalar_value(self):
        # A 0-d value, as in x.repeat(2**26).view(-1, 4).gather(0, i) with one
        # i, does not promote the int32 tile index it is added to: in int32,
        # (tile_x.index + 4*tmp0) % 100 wraps for tmp0 = 2**29.
        graph = GraphLowering(torch.fx.symbolic_trace(lambda x: x + 1))
        with V.set_graph_handler(graph):
            kernel = HelionKernel({"x": 4}, features=features(4))
            with V.set_kernel_handler(kernel):
                value = kernel.cse.generate(
                    kernel.compute, "src", dtype=torch.int64, shape=()
                )
                tmp = sympy_index_symbol(str(value))
                kernel.indirect[tmp] = value
                x = kernel.symbols["x"]
                index = ModularIndexing(x + 4 * tmp, 1, 100)
                self.assertEqual(
                    kernel.index_tensor_expr(index, ["x"]),
                    "((tile_x.index.to(torch.int64) + 4*tmp0) % 100)",
                )


class TestFlatLane(TestCase):
    def check(self, index_of, expected, vector=8):
        graph = GraphLowering(torch.fx.symbolic_trace(lambda x: x + 1))
        with V.set_graph_handler(graph):
            sizes = {"z0_": 128, "y": 4, "x": 160}
            kernel = HelionKernel(sizes, features=features(128 * 4 * 160))
            with V.set_kernel_handler(kernel):
                # A gathered index read from a tensor (an embedding's), [tile_y].
                value = kernel.cse.generate(
                    kernel.compute, "src", dtype=torch.int64, shape=("tile_y",)
                )
                tmp = sympy_index_symbol(str(value))
                kernel.indirect[tmp] = value
                symbols = kernel.symbols
                index = index_of(symbols["z0_"], symbols["y"], symbols["x"], tmp)
                self.assertEqual(
                    kernel.index_tensor_expr(index, list(sizes), vector=vector),
                    expected,
                )

    def test_lane_added_last(self):
        # The H/W-swapped operand of better-benchmark's sum_9683f50f2dcf, read
        # flat (at a pointwise shape): whole 16-byte vectors beside the
        # innermost tile's index, which goes last, where CuTe's flat-gather
        # match takes it (cute/flat_gather_lane.py).
        self.check(
            lambda z, y, x, tmp: (
                x + 160 * FloorDiv(y, 2) + 4480 * ModularIndexing(y, 1, 2) + 125440 * z
            ),
            "(160*(tile_y.index.reshape([tile_y, 1]) // 2)"
            " + 4480*((tile_y.index.reshape([tile_y, 1]) % 2))"
            " + 125440*tile_z.index.reshape([tile_z, 1, 1])) + tile_x.index",
        )

    def test_gather_by_a_loaded_index_adds_the_lane_last(self):
        # Elements gathered by a loaded index (an embedding's lookup, tmp0 of
        # [tile_y]): the rest is the value's term, whole vectors of 160.
        self.check(
            lambda z, y, x, tmp: x + 160 * tmp + 20480 * z,
            "(160*tmp0.reshape([tile_y, 1])"
            " + 20480*tile_z.index.reshape([tile_z, 1, 1])) + tile_x.index",
        )

    def test_rest_of_partial_vectors_keeps_the_order(self):
        # A cat's part at -4 + 364*y: no rest of whole bf16 vectors.
        self.check(
            lambda z, y, x, tmp: x - 4 + 364 * y + 2048 * z,
            "(-4) + tile_x.index + 364*tile_y.index.reshape([tile_y, 1])"
            " + 2048*tile_z.index.reshape([tile_z, 1, 1])",
        )

    def test_scaled_lane_keeps_the_order(self):
        # Every other element along x: no lane term with a unit coefficient.
        self.check(
            lambda z, y, x, tmp: 2 * x + 320 * y + 2048 * z,
            "2*tile_x.index + 320*tile_y.index.reshape([tile_y, 1])"
            " + 2048*tile_z.index.reshape([tile_z, 1, 1])",
        )

    def test_vector_width_decides_the_lane(self):
        # A rest of 164*y is whole 4-element (fp32) vectors but no whole 8-element
        # (bf16) ones.
        index_of = lambda z, y, x, tmp: x + 164 * y + 2048 * z  # noqa: E731
        self.check(
            index_of,
            "(164*tile_y.index.reshape([tile_y, 1])"
            " + 2048*tile_z.index.reshape([tile_z, 1, 1])) + tile_x.index",
            vector=4,
        )
        self.check(
            index_of,
            "tile_x.index + 164*tile_y.index.reshape([tile_y, 1])"
            " + 2048*tile_z.index.reshape([tile_z, 1, 1])",
        )

    def test_a_scaled_innermost_tile_moves_the_next_one_last(self):
        # The innermost tile scaled by 16 and y added once: y is the lane.
        self.check(
            lambda z, y, x, tmp: 16 * x + y + 2048 * z,
            "(16*tile_x.index + 2048*tile_z.index.reshape([tile_z, 1, 1]))"
            " + tile_y.index.reshape([tile_y, 1])",
        )

    def test_a_rest_whose_partial_sums_may_wrap_widens(self):
        # A term read from a tensor beside int32 terms whose sum fits int32
        # but whose partial sums need not: -2**31 - 8*((y + 1) // 2) leaves
        # it before 64*((y + 4) // 3) brings it back, and 16*tmp0 (int64) can
        # join between them, which the printer's order did with the lane
        # first at x < 8*((y + 1) // 2) (the index 56 of x = 0, y = 1, tmp0 =
        # 2**27 read as 4294967352). The tile indices widen to int64, so no
        # order wraps, and the lane goes last.
        self.check(
            lambda z, y, x, tmp: (
                x - 2**31 - 8 * FloorDiv(y + 1, 2) + 16 * tmp + 64 * FloorDiv(y + 4, 3)
            ),
            "(((-2147483648)"
            " + (-8)*((1 + tile_y.index.reshape([tile_y, 1]).to(torch.int64)) // 2)"
            " + 16*tmp0.reshape([tile_y, 1])"
            " + 64*((4 + tile_y.index.reshape([tile_y, 1]).to(torch.int64)) // 3))"
            " + tile_x.index.to(torch.int64)).expand(tile_z, tile_y, tile_x)",
        )

    def test_a_value_in_a_nested_sum_widens(self):
        # The same sum inside a ModularIndexing, whose dividend wraps the
        # same way before 16*tmp0 joins (56 % 257 read as 4294967352 % 257,
        # 57): the tile indices widen too. The index adds no tile at its top,
        # so no lane moves.
        self.check(
            lambda z, y, x, tmp: ModularIndexing(
                x - 2**31 - 8 * FloorDiv(y + 1, 2) + 16 * tmp + 64 * FloorDiv(y + 4, 3),
                1,
                257,
            ),
            "((((-2147483648) + tile_x.index.to(torch.int64)"
            " + (-8)*((1 + tile_y.index.reshape([tile_y, 1]).to(torch.int64)) // 2)"
            " + 16*tmp0.reshape([tile_y, 1])"
            " + 64*((4 + tile_y.index.reshape([tile_y, 1]).to(torch.int64)) // 3))"
            " % 257)).expand(tile_z, tile_y, tile_x)",
        )

    def test_a_rest_whose_partial_sums_fit_adds_the_lane_last(self):
        # The same kind of rest with no negative term but the constant (8*y):
        # every partial sum of the int32 terms stays in [-2**31, -2**31 +
        # 24], and the lane goes last.
        self.check(
            lambda z, y, x, tmp: x - 2**31 + 8 * y + 16 * tmp,
            "(((-2147483648) + 8*tile_y.index.reshape([tile_y, 1])"
            " + 16*tmp0.reshape([tile_y, 1])) + tile_x.index)"
            ".expand(tile_z, tile_y, tile_x)",
        )

    def test_a_persistent_rdim_is_no_lane(self):
        # A persistent reduction's r is an arange of the whole rdim, no tile
        # whose index can be a vector lane: r + 8*x has none (x is scaled),
        # x + 8*r has x.
        graph = GraphLowering(torch.fx.symbolic_trace(lambda x: x + 1))
        with V.set_graph_handler(graph):
            reduction = SIMDKernelFeatures([], sympy.Integer(64), sympy.Integer(128))
            kernel = HelionKernel(
                {"x": 64, "r0_": 128},
                features=reduction,
                override_persistent_reduction=True,
            )
            x, r = kernel.symbols["x"], kernel.symbols["r0_"]
            self.assertIsNone(kernel.flat_lane(r + 8 * x, ["x", "r0_"], 8))
            self.assertEqual(kernel.flat_lane(x + 8 * r, ["x", "r0_"], 8), (x, 8 * r))

    def test_a_repeated_lane_keeps_the_order(self):
        # The lane in the rest too (x // 4): no rest independent of the lane.
        self.check(
            lambda z, y, x, tmp: x + 8 * FloorDiv(x, 4) + 2048 * z,
            "(tile_x.index + 8*(tile_x.index // 4)"
            " + 2048*tile_z.index.reshape([tile_z, 1, 1])).expand(tile_z, tile_y, tile_x)",
        )

    def test_flat_subscript_derives_the_vector_from_the_dtype(self):
        # flat_subscript() asks for 16 bytes of the buffer's dtype: a rest of
        # 164*y is whole vectors of 4 fp32 elements, not of 8 bf16 ones.
        for dtype, expected in (
            (torch.float32, "(164*tile_y.index.reshape([tile_y, 1])) + tile_x.index"),
            (torch.bfloat16, "tile_x.index + 164*tile_y.index.reshape([tile_y, 1])"),
        ):
            graph = GraphLowering(torch.fx.symbolic_trace(lambda x: x + 1))
            with V.set_graph_handler(graph):
                sizes = {"y": 4, "x": 160}
                kernel = HelionKernel(sizes, features=features(4 * 160))
                buffer = InputBuffer(
                    name="arg0_1",
                    layout=FixedLayout(torch.device("cuda"), dtype, [656], [1]),
                )
                with V.set_kernel_handler(kernel):
                    y, x = kernel.symbols["y"], kernel.symbols["x"]
                    _, coordinates, _ = kernel.flat_subscript(
                        buffer, x + 164 * y, list(sizes)
                    )
                    self.assertEqual(coordinates, [expected], dtype)


class TestSpanSubscript(TestCase):
    """HelionKernel.span_subscript(): a flat index that adds the innermost
    tile, with coefficient 1, to a multiple of its extent the other tiles
    select reads a view of spans of that extent."""

    def span(
        self, sizes, index, numel, dtype=torch.bfloat16, persistent=None, axes=None
    ):
        axes = list(sizes) if axes is None else axes
        graph = GraphLowering(torch.fx.symbolic_trace(lambda x: x + 1))
        with V.set_graph_handler(graph):
            total = 1
            for size in sizes.values():
                total *= size
            reduction = sizes.get("r0_")
            kernel = HelionKernel(
                sizes,
                features=SIMDKernelFeatures(
                    [],
                    sympy.Integer(total // (reduction or 1)),
                    sympy.Integer(reduction or 1),
                ),
                **(
                    {}
                    if persistent is None
                    else {"override_persistent_reduction": persistent}
                ),
            )
            buffer = InputBuffer(
                name="arg0_1",
                layout=FixedLayout(torch.device("cuda"), dtype, [numel], [1]),
            )
            with V.set_kernel_handler(kernel):
                found = kernel.span_subscript(buffer, index(**kernel.symbols), axes)
                # A store's subscript, and one with spans off, as store() asks
                # within a kernel emitted over segments.
                stores = [
                    kernel.subscript(
                        buffer,
                        index(**kernel.symbols),
                        axes,
                        output=True,
                        spans=spans,
                    )[1]
                    for spans in (True, False)
                ]
        if found is None:
            return None, stores
        arg, coordinates, shape = found
        layout = arg.operand.get_layout()
        layout_facts = (list(layout.size), list(layout.stride), layout.offset)
        return (*layout_facts, coordinates, shape), stores

    def test_swin_window_partition(self):
        # Swin's window partition after a roll of 3 over [28672, 7, 128]
        # tiles: x plus 128 times the rolled, windowed token, of a [64, 56,
        # 56, 128] input, reads the [200704, 128] view by the token's span.
        token = lambda z, y: (  # noqa: E731
            ModularIndexing(3 + y + 7 * ModularIndexing(z, 1, 8), 1, 56)
            + 56
            * ModularIndexing(
                3 + 7 * ModularIndexing(z, 56, 8) + ModularIndexing(z, 8, 7), 1, 56
            )
            + 3136 * FloorDiv(z, 448)
        )
        found, stored = self.span(
            {"z0_": 28672, "y": 7, "x": 128},
            lambda z0_, y, x: x + 128 * token(z0_, y),
            25690112,
        )
        assert found is not None
        sizes, strides, offset, coordinates, shape = found
        self.assertEqual((sizes, strides, offset), ([200704, 128], [128, 1], 0))
        self.assertEqual(coordinates[1], "tile_x")
        self.assertNotIn("tile_x", coordinates[0])
        self.assertIn("tile_z.index.reshape([tile_z, 1])", coordinates[0])
        self.assertEqual(shape, ("tile_z", "tile_y", "tile_x"))
        # Its store writes the span view too, but within a kernel emitted over
        # segments (store() turns spans off) through the flat index: the CuTe
        # backend cannot tell the segments' spans of one output apart.
        span_store, flat_store = stored
        self.assertEqual(len(span_store), 2)
        self.assertEqual(span_store[1], "tile_x")
        self.assertEqual(len(flat_store), 1)
        self.assertTrue(flat_store[0].endswith(" + tile_x.index"))

    def test_persistent_rdim_and_offset(self):
        # A persistent rdim is the innermost tile: a cat's second part, read
        # from 128 elements before x's span (an offset of -128 = -1 span),
        # and a part starting 32 elements into a span (an offset that is no
        # multiple of the extent) keeps a view starting there.
        found, _ = self.span(
            {"x": 1000, "r0_": 128},
            lambda x, r0_: r0_ - 128 + 128 * ModularIndexing(x, 1, 500),
            128000,
            persistent=True,
        )
        assert found is not None
        sizes, strides, offset, coordinates, shape = found
        self.assertEqual((sizes, strides, offset), ([1000, 128], [128, 1], 0))
        self.assertEqual(coordinates[1], ":")
        self.assertEqual(shape, ("tile_x", "tile_r"))
        found, _ = self.span(
            {"x": 999, "r0_": 128},
            lambda x, r0_: r0_ + 32 + 128 * ModularIndexing(x, 1, 500),
            128000,
            persistent=True,
        )
        assert found is not None
        self.assertEqual(found[:3], ([999, 128], [128, 1], 32))

    def test_declines(self):
        # An offset no multiple of the extent, a scaled innermost tile, a
        # looped rdim (the backend reads its flat index as a lane of the
        # loop's tile, match_flat_gather_lane()) and a span outside the view's
        # whole spans keep the flat index.
        cases = [
            (
                {"y": 7, "x": 128},
                lambda y, x: x + 100 * ModularIndexing(y, 1, 5),
                1024,
                None,
            ),
            (
                {"y": 7, "x": 64},
                lambda y, x: 2 * x + 128 * ModularIndexing(y, 1, 5),
                1024,
                None,
            ),
            (
                {"x": 64, "r0_": 4096},
                lambda x, r0_: r0_ + 4096 * ModularIndexing(x, 1, 32),
                262144,
                False,
            ),
            # F.pad(x, (0, 4)) of a [4, 17, 304] tensor's [:, :16, :300]: its
            # last tokens' span (15 + 17 * 3 = 66) reaches past the 66 whole
            # spans of 304 its 20364-element storage holds.
            (
                {"z0_": 4, "y": 16, "r0_": 304},
                lambda z0_, y, r0_: r0_ + 304 * (y + 17 * z0_),
                20364,
                True,
            ),
            # F.pad(x, (96, 0)) of a [2, 128] tensor's [:, :32]: a view from
            # 32 elements in, whose span y - 1 is -1 for y = 0, where the
            # left pad's lanes read the storage's first 32 elements.
            (
                {"y": 2, "r0_": 128},
                lambda y, r0_: r0_ - 96 + 128 * y,
                160,
                True,
            ),
        ]
        for sizes, index, numel, persistent in cases:
            found, _ = self.span(sizes, index, numel, persistent=persistent)
            self.assertIsNone(found, sizes)
        # So does any access in the loop over tile_r, whose flat index (an
        # offset plus the lane) the backend's V-loop sink reads in packets:
        # a load there over tiles of y and x only, x + 128 * (y % 4), which
        # outside the loop reads the [4, 128] view by span.
        index = lambda y, x, r0_: x + 128 * ModularIndexing(y, 1, 4)  # noqa: E731
        sizes = {"y": 8, "x": 128, "r0_": 4096}
        for persistent in (False, True):
            found, _ = self.span(
                sizes, index, 512, persistent=persistent, axes=["y", "x"]
            )
            self.assertEqual(found is None, not persistent)

    def test_spans_on_16_byte_boundaries(self):
        # Spans of 6156 elements: 24624 bytes in fp32, on 16-byte boundaries,
        # read the view; 12312 bytes in bf16, every other span 8 bytes off
        # them, and spans of 257 bf16 elements keep the flat index.
        index = lambda y, x: x + 6156 * ModularIndexing(y, 1, 4)  # noqa: E731
        found, _ = self.span({"y": 8, "x": 6156}, index, 24624, dtype=torch.float32)
        assert found is not None
        self.assertEqual(found[:3], ([4, 6156], [6156, 1], 0))
        found, _ = self.span({"y": 8, "x": 6156}, index, 24624)
        self.assertIsNone(found)
        found, _ = self.span(
            {"z0_": 512, "x": 257},
            lambda z0_, x: x + 257 * ModularIndexing(z0_, 1, 256),
            65792,
        )
        self.assertIsNone(found)

    def test_masked_off_load_keeps_the_flat_index(self):
        # b's load of cat([a, b], 1), a and b bf16 [4, 256, 64], over tiles of
        # a's 256 y: under its mask y >= 256, which the tiles' ranges prove
        # false, it reads nothing and keeps its flat index (load()); under a
        # mask they prove nothing of, it reads b's [1024, 64] view by span.
        layout = FixedLayout(
            torch.device("cuda"), torch.bfloat16, [4, 256, 64], [16384, 64, 1]
        )
        graph = GraphLowering(torch.fx.symbolic_trace(lambda x: x + 1))
        graph.name_to_buffer["buf0"] = InputBuffer(name="buf0", layout=layout)
        masks = [
            CSEVariable(name, ValueRanges.unknown(), torch.bool, ("tile_y",))
            for name in ("off", "unknown")
        ]
        interpreter = SimpleNamespace(current_node=None)
        with V.set_graph_handler(graph), V.set_interpreter_handler(interpreter):
            features = SIMDKernelFeatures([], sympy.Integer(65536), sympy.Integer(1))
            kernel = HelionKernel({"z0_": 4, "y": 256, "x": 64}, features=features)
            z, y, x = (kernel.symbols[axis] for axis in ("z0_", "y", "x"))
            kernel.predicates["off"] = [sympy.Ge(y, 256, evaluate=False)]
            with V.set_kernel_handler(kernel):
                for mask in masks:
                    kernel.load_masks.append(mask)
                    kernel.load("buf0", x + 64 * (y - 256) + 16384 * z)
                    kernel.load_masks.pop()
        off, unknown = kernel.compute.getvalue().splitlines()
        self.assertIn(
            "[((-16384) + 64*tile_y.index.reshape([tile_y, 1])"
            " + 16384*tile_z.index.reshape([tile_z, 1, 1])) + tile_x.index]",
            off,
        )
        self.assertIn("extra_mask=off.reshape([tile_y, 1])", off)
        self.assertIn(
            "[(-256) + tile_y.index + 256*tile_z.index.reshape([tile_z, 1]), tile_x]",
            unknown,
        )


class TestAxisSplit(TestCase):
    """HelionKernel.axis_split() and split_axis(): a pointwise axis that the
    kernel's indices no view reads divide, or take modulo, by one size K that
    divides its extent is tiled as extent / K by K, over which they are
    affine."""

    def split(self, sizes, *indices, cuts=(), then=None):
        """The split axis_split() takes after the kernel over `sizes` saw
        `indices` (note_axis_divisors()), with `cuts` axes that segments cut;
        and `then` over the tile indices once split_axis() took it."""
        graph = GraphLowering(torch.fx.symbolic_trace(lambda x: x + 1))
        with V.set_graph_handler(graph):
            total = 1
            for size in sizes.values():
                total *= size
            kernel = HelionKernel(
                sizes,
                features=SIMDKernelFeatures([], sympy.Integer(total), sympy.Integer(1)),
            )
            for axis in cuts:
                kernel.axis_cuts[axis] = OrderedSet([1])
            with V.set_kernel_handler(kernel):
                for index in indices:
                    kernel.note_axis_divisors(index(**kernel.symbols))
                found = kernel.axis_split()
                if found is None or then is None:
                    return found, None
                symbols = dict(kernel.symbols)
                kernel.split_axis(*found)
                index = kernel.tile_index(then(**symbols))
                axes = kernel.outer_axes
                extents = [kernel.extents[axis] for axis in axes]
                # The index over tile_w (w) and the other tiles, by axis.
                names = {kernel.symbols[axis]: axis.rstrip("0_") for axis in axes}
                return found, (
                    str(
                        index.xreplace(
                            {
                                symbol: sympy.Symbol(name)
                                for symbol, name in names.items()
                            }
                        )
                    ),
                    axes,
                    extents,
                )

    def test_unfold(self):
        # MobileViT's unfold, a cat of three trimmed [512, 4, 256, 40] parts
        # copied to [512, 256, 3, 4, 36]: the store divides y of 1024 by 256
        # and takes it modulo 256, so y is tiled as tile_w of 4 by tile_y of
        # 256, over which the store and the load are affine.
        store = lambda z0_, y, x: (  # noqa: E731
            x + 36 * FloorDiv(y, 256) + 432 * ModularIndexing(y, 1, 256) + 110592 * z0_
        )
        load = lambda z0_, y, x: x + 40 * y + 40960 * z0_  # noqa: E731
        found, split = self.split(
            {"z0_": 512, "y": 1024, "x": 36}, store, load, then=store
        )
        self.assertEqual(found, ("y", 256))
        self.assertEqual(
            split,
            (
                "36*w + x + 432*y + 110592*z",
                ("z0_", "w_", "y", "x"),
                [512, 4, 256, 36],
            ),
        )
        # The load: 40 * y is 10240 * tile_w + 40 * tile_y.
        _, split = self.split({"z0_": 512, "y": 1024, "x": 36}, store, then=load)
        self.assertEqual(split[0], "10240*w + x + 40*y + 40960*z")

    def test_declines(self):
        sizes = {"y": 1024, "x": 36}
        # Two sizes of one axis: no split makes both affine.
        two = lambda y, x: FloorDiv(y, 256) + ModularIndexing(y, 1, 128)  # noqa: E731
        self.assertIsNone(self.split(sizes, two)[0])
        # A size that does not divide the extent.
        self.assertIsNone(self.split(sizes, lambda y, x: FloorDiv(y, 300))[0])
        # Two outer axes, each divided: one split at most.
        both = lambda z0_, y, x: FloorDiv(y, 256) + 7 * ModularIndexing(z0_, 1, 4)  # noqa: E731
        self.assertIsNone(self.split({"z0_": 64, **sizes}, both)[0])
        # The innermost axis stays whole: split, its tile would hold runs of
        # 256 elements instead of the axis's whole runs.
        inner = lambda y, x: FloorDiv(x, 256) + 7 * ModularIndexing(x, 1, 256)  # noqa: E731
        self.assertIsNone(self.split({"y": 36, "x": 1024}, inner)[0])
        # Nor does an outer axis split beside it: the innermost axis's
        # accesses would stay flat.
        outer = lambda y, x: FloorDiv(y, 256) + 7 * ModularIndexing(x, 1, 4)  # noqa: E731
        self.assertIsNone(self.split(sizes, outer)[0])
        # An axis the segments cut, which they make affine.
        self.assertIsNone(
            self.split(sizes, lambda y, x: FloorDiv(y, 256), cuts=["y"])[0]
        )
        # A size of the whole extent divides nothing: y // 1024 is 0.
        whole = lambda y, x: FloorDiv(y, 1024) + FloorDiv(y, 256)  # noqa: E731
        self.assertEqual(self.split(sizes, whole)[0], ("y", 256))


class TestPersistentLoad(TestCase):
    def test_masked_load_through_layout(self):
        # In a persistent reduction a masked load through its buffer's own
        # layout lists the whole row as slice(None), as [tile_x, :] is a
        # SyntaxError in hl.load's index list (HelionKernel.index_list()).
        # Inductor's cat and pad index their inputs through Identity(), e.g.
        # F.pad(x[:32], (0, 0, 0, 32)) reads x at r0_1 + 32*Identity(x0), which
        # matches no layout and reads through a flat view, so this emits such a
        # load directly.
        layout = FixedLayout(torch.device("cuda"), torch.float32, [64, 300], [300, 1])
        graph = GraphLowering(torch.fx.symbolic_trace(lambda x: x + 1))
        graph.name_to_buffer["buf0"] = InputBuffer(name="buf0", layout=layout)
        mask = CSEVariable("mask", ValueRanges.unknown(), torch.bool, ("tile_x",))
        interpreter = SimpleNamespace(current_node=None)
        with V.set_graph_handler(graph), V.set_interpreter_handler(interpreter):
            reduction = SIMDKernelFeatures([], sympy.Integer(64), sympy.Integer(300))
            kernel = HelionKernel(
                {"x": 64, "r0_": 300},
                features=reduction,
                override_persistent_reduction=True,
            )
            with V.set_kernel_handler(kernel):
                kernel.load_masks.append(mask)
                x, r = kernel.symbols["x"], kernel.symbols["r0_"]
                kernel.load("buf0", 300 * x + r)
        self.assertEqual(
            kernel.compute.getvalue(),
            "tmp0 = hl.load(__arg0__, [tile_x, slice(None)],"
            " extra_mask=mask.reshape([tile_x, 1]))\n",
        )


if __name__ == "__main__":
    unittest.main()
