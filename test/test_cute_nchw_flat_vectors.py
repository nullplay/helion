"""Packets of an NCHW tensor read through a flat tile over (n, h, w).

A channel's elements sit at ``c * HW + (r // HW) * (C * HW) + r % HW`` for
the tile index ``r``: every lane's pointer has the lane under the ``//`` and
the ``%``.  With ``HW`` a multiple of the vector width, the affine vectorizer
reads the pointer on the copy whose quotients and remainders divide the lane
base (``lane_invariant_division.divided_vector_loops``), where the lane has
coefficient one, and loads and stores the V lanes as one packet; a load of
the channel's own parameter, one address for every lane, stays scalar.
Rendered on CPU.
"""

from __future__ import annotations

import ast
import re

import pytest
import torch

import helion
from helion._compiler.cute.affine_vector_io import _masked_packet
from helion._compiler.cute.persistent_branch_vec import _DTYPE_INFO
from helion._compiler.cute.persistent_branch_vec import _pointer_integer_affine_form
from helion._testing import DEVICE
from helion._testing import cpu_render_target
from helion._testing import skipIfNotCUDA
from helion._testing import skipUnlessBackends
from helion._testing.cute_codegen import cpu_bind
import helion.language as hl

pytestmark = pytest.mark.cpu_only


def _channel_sums(
    x: torch.Tensor, out: torch.Tensor, c: hl.constexpr, hw: hl.constexpr
) -> None:
    for tile_x in hl.tile(c):
        acc = hl.zeros([tile_x], dtype=torch.float32)
        for tile_r in hl.tile(x.size(0) // c):
            v = x[
                hw * tile_x.index.reshape([tile_x, 1])
                + (c * hw) * (tile_r.index // hw)
                + tile_r.index % hw
            ].to(torch.float32)
            acc = acc + torch.sum(v, -1)
        out[tile_x] = acc


def _scaled_channels(
    x: torch.Tensor,
    w: torch.Tensor,
    out: torch.Tensor,
    c: hl.constexpr,
    hw: hl.constexpr,
) -> None:
    for tile_x in hl.tile(c):
        for tile_r in hl.tile(x.size(0) // c):
            index = (
                hw * tile_x.index.reshape([tile_x, 1])
                + (c * hw) * (tile_r.index // hw)
                + tile_r.index % hw
            )
            v = x[index].to(torch.float32) * w[tile_x].reshape([tile_x, 1])
            out[index] = v.to(out.dtype)


def _cat_channel_sums(
    a: torch.Tensor,
    b: torch.Tensor,
    out: torch.Tensor,
    c0: hl.constexpr,
    c1: hl.constexpr,
    hw: hl.constexpr,
) -> None:
    # The channel sums of cat([a, b], 1) of two NCHW parts: each channel reads
    # one part, under a mask that is one value for a channel's tile.
    for tile_x in hl.tile(c0 + c1):
        acc = hl.zeros([tile_x], dtype=torch.float32)
        for tile_r in hl.tile(a.size(0) // c0):
            first = (tile_x.index < c0).reshape([tile_x, 1])
            run = (tile_r.index // hw) * hw
            va = hl.load(
                a,
                [hw * tile_x.index.reshape([tile_x, 1]) + c0 * run + tile_r.index % hw],
                extra_mask=first,
            )
            vb = hl.load(
                b,
                [
                    hw * (tile_x.index.reshape([tile_x, 1]) - c0)
                    + c1 * run
                    + tile_r.index % hw
                ],
                extra_mask=~first,
            )
            acc = acc + torch.sum(torch.where(first, va, vb).to(torch.float32), -1)
        out[tile_x] = acc


def _broadcast_plane_sums(
    x: torch.Tensor,
    out: torch.Tensor,
    c: hl.constexpr,
    hw: hl.constexpr,
    n: hl.constexpr,
) -> None:
    # The channel sums of a [C, H, W] image broadcast over n samples (stride
    # 0 on n): the lane sits under a remainder and no quotient.
    for tile_x in hl.tile(c):
        acc = hl.zeros([tile_x], dtype=torch.float32)
        for tile_r in hl.tile(n * hw):
            v = x[hw * tile_x.index.reshape([tile_x, 1]) + tile_r.index % hw].to(
                torch.float32
            )
            acc = acc + torch.sum(v, -1)
        out[tile_x] = acc


_SUMS = helion.kernel(_channel_sums, backend="cute", static_shapes=True)
_CAT_SUMS = helion.kernel(_cat_channel_sums, backend="cute", static_shapes=True)
_SCALED = helion.kernel(_scaled_channels, backend="cute", static_shapes=True)
_BROADCAST_SUMS = helion.kernel(
    _broadcast_plane_sums, backend="cute", static_shapes=True
)


def _config(width: int, **options: object) -> helion.Config:
    return helion.Config(
        block_sizes=[1, 256],
        num_threads=[1, 256 // width],
        cute_vector_widths=[1, width],
        **options,
    )


def _code(
    kernel: object, args: tuple[object, ...], width: int, **options: object
) -> str:
    # A GB300's shared memory for the reduction's two-stage combine.
    with cpu_render_target(reduction_smem_budget_bytes=232448):
        bound = cpu_bind(kernel, args)
        return bound.to_code(
            bound.config_spec.normalized_config(_config(width, **options))
        )


def _packets(code: str, call: str) -> list[str]:
    """The source of each ``call`` (``cute.arch.load`` of a vector type or a
    vector store helper) in ``code``."""
    found = []
    for node in ast.walk(ast.parse(code)):
        if not isinstance(node, ast.Call):
            continue
        source = ast.unparse(node)
        if ast.unparse(node.func) == call and (
            call != "cute.arch.load" or "ir.VectorType.get(" in source
        ):
            found.append(source)
    return found


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize(
    ("dtype", "width", "hw"),
    [
        (torch.bfloat16, 4, 196),  # 14 x 14: a multiple of 4, not of 8
        (torch.float32, 4, 196),  # one 16-byte packet of fp32
        (torch.float16, 8, 64),  # 8 x 8: one 16-byte packet of fp16
    ],
)
def test_nchw_flat_loads_are_packets(dtype: torch.dtype, width: int, hw: int) -> None:
    # n * hw = 3 * hw: the last tile of 256 is partial.
    c = 6
    x = torch.randn(3 * c * hw).to(dtype)
    code = _code(_SUMS, (x, torch.empty(c), c, hw), width)
    loads = _packets(code, "cute.arch.load")
    assert loads, code
    assert all(f"ir.VectorType.get([{width}]" in load for load in loads), loads


@skipUnlessBackends(["cute"])
def test_nchw_flat_store_is_a_packet_beside_a_channel_parameter() -> None:
    c, hw = 6, 196
    x = torch.randn(3 * c * hw).to(torch.bfloat16)
    args = (x, torch.randn(c), torch.empty_like(x), c, hw)
    code = _code(_SCALED, args, 4)
    assert len(_packets(code, "cute.arch.load")) == 1, code
    assert len(_packets(code, "_cute_store_u16_vec")) == 1, code
    # The channel's parameter: one scalar load, at the same address per lane.
    assert "(w.iterator + cutlass.Int32(indices_0)" in code, code


@skipUnlessBackends(["cute"])
def test_exclusive_part_masks_load_predicated_packets() -> None:
    """The two parts' masks never hold together: each part loads its packet
    under its own mask (zeros where it is false), which is one value for a
    channel's fragment, instead of a fragment guard no fragment passes."""
    c0, c1, hw = 4, 2, 196
    a = torch.randn(3 * c0 * hw).to(torch.bfloat16)
    b = torch.randn(3 * c1 * hw).to(torch.bfloat16)
    code = _code(_CAT_SUMS, (a, b, torch.empty(c0 + c1), c0, c1, hw), 4)
    assert code.count("_cute_load_v8b_if(") == 2, code
    assert not _packets(code, "cute.arch.load"), code


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize("dtype", sorted(_DTYPE_INFO))
def test_a_predicated_packet_loads_its_store_carrier(dtype: str) -> None:
    """Every element type the vectorizer packs, unsigned ones included, loads
    a predicated sixteen-byte packet as the integers its stores use."""
    carrier, itemsize, _ = _DTYPE_INFO[dtype]
    width = 16 // itemsize
    load = _masked_packet("ptr", "pred", width, dtype)
    assert load.startswith("_cute_load_v16b_if(ptr, pred, "), load
    assert f"[{width}], {carrier}.mlir_type" in load, load


@skipUnlessBackends(["cute"])
def test_a_remainder_alone_divides_the_lane_base() -> None:
    """A vector loop whose only lane-invariant identity is a remainder (an
    image broadcast over the samples, ``x[c * HW + r % HW]``) reads its
    pointer on the divided copy too, and loads packets."""
    c, hw = 6, 196
    x = torch.randn(c * hw).to(torch.bfloat16)
    code = _code(_BROADCAST_SUMS, (x, torch.empty(c), c, hw, 3), 4)
    loads = _packets(code, "cute.arch.load")
    assert loads, code
    assert all("ir.VectorType.get([4]" in load for load in loads), loads


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize("uniform_compares", [False, True])
def test_a_rebuilt_flat_index_compares_its_base(uniform_compares: bool) -> None:
    """An fp64 image's loads stay scalar, with the bound check of the flat
    index ``c * HW + C * HW * (r // HW) + r % HW < numel``.  Under
    ``cute_uniform_lane_compares`` the remainder's factor carries into the
    sum, which compares its base once per vector; without the knob every
    lane compares its own index."""
    c, hw = 6, 196
    x = torch.randn(3 * c * hw, dtype=torch.float64)
    code = _code(
        _SUMS,
        (x, torch.empty(c), c, hw),
        4,
        cute_uniform_lane_compares=uniform_compares,
    )
    base = re.compile(
        r"cutlass\.Int32\(v_\d+ \+ _cute_python_mod\(cutlass\.Int32\(lane_base_1\), "
        r"cutlass\.Int32\(v_\d+\)\)\) < 3528"
    )
    assert bool(base.search(code)) is uniform_compares, code
    assert (
        bool(re.search(r"cutlass\.Int32\(v_\d+\) < 3528", code)) is not uniform_compares
    )


def test_a_lane_free_product_is_an_atom_unless_a_stride_is_unproven() -> None:
    def form(source: str, strides: dict[tuple[str, int], int]) -> object:
        return _pointer_integer_affine_form(
            ast.parse(source, mode="eval").body, "lane", strides, set(), "pointer"
        )

    product = ast.dump(ast.parse("a * b", mode="eval").body)
    assert form("lane + a * b", {}) == (1, {product: 1}, 0)
    assert form("(lane + a) * b", {}) is None
    stride = "cutlass.Int32(t.layout.stride[0])"
    assert form(f"lane + a * {stride}", {}) is None
    atom = ast.dump(ast.parse("a", mode="eval").body)
    assert form(f"lane + a * {stride}", {("t", 0): 8}) == (1, {atom: 8}, 0)


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize(
    ("hw", "width"),
    [
        (49, 4),  # 7 x 7: a vector straddles the next sample's plane
        (196, 8),  # 196 is a multiple of 4, not of 8
    ],
)
def test_a_plane_no_multiple_of_the_width_keeps_scalar_loads(
    hw: int, width: int
) -> None:
    c = 6
    x = torch.randn(3 * c * hw).to(torch.bfloat16)
    code = _code(_SUMS, (x, torch.empty(c), c, hw), width)
    assert not _packets(code, "cute.arch.load"), code


@pytest.mark.cuda
@skipUnlessBackends(["cute"])
@skipIfNotCUDA()
@pytest.mark.parametrize("width", [4, 8])
def test_cuda_nchw_flat_packets_match_eager(width: int) -> None:
    c, hw, n = 6, 64, 5
    image = torch.randn(n, c, 8, 8, device=DEVICE, dtype=torch.bfloat16)
    w = torch.randn(c, device=DEVICE)
    x = image.reshape(-1)
    sums = helion.kernel(
        _channel_sums, backend="cute", static_shapes=True, config=_config(width)
    )
    scaled = helion.kernel(
        _scaled_channels, backend="cute", static_shapes=True, config=_config(width)
    )
    out = torch.empty(c, device=DEVICE)
    sums(x, out, c, hw)
    torch.testing.assert_close(out, image.float().sum((0, 2, 3)), rtol=1e-4, atol=1e-3)
    y = torch.empty_like(x)
    scaled(x, w, y, c, hw)
    cat_sums = helion.kernel(
        _cat_channel_sums, backend="cute", static_shapes=True, config=_config(width)
    )
    parts = image[:, :4].contiguous(), image[:, 4:].contiguous()
    cat_out = torch.empty(c, device=DEVICE)
    cat_sums(parts[0].reshape(-1), parts[1].reshape(-1), cat_out, 4, c - 4, hw)
    torch.testing.assert_close(
        cat_out, image.float().sum((0, 2, 3)), rtol=1e-4, atol=1e-3
    )
    torch.testing.assert_close(
        y.view_as(image), (image.float() * w[None, :, None, None]).to(image.dtype)
    )
