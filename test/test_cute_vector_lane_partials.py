"""A cross-thread reduce that ``hoist_warp_reduce`` leaves in a constexpr
V-loop combines one V lane of every thread: a partial of the reduction over
the loop's block.  Values the same in every lane and later reduces may fold
the partials; per-element values computed from one may not, and such a
config is refused (``_per_lane_reduce_escape``)."""

from __future__ import annotations

import ast
import textwrap

import pytest
import torch

import helion
from helion import exc
from helion._compiler.cute.hoist_warp_reduce import hoist_warp_reduce_from_vloop
from helion._testing import cpu_render_target
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import mock_cuda_unavailable
import helion.language as hl

pytestmark = pytest.mark.cpu_only


def _hoist(src: str) -> str:
    body = hoist_warp_reduce_from_vloop(ast.parse(textwrap.dedent(src)).body)
    return ast.unparse(ast.Module(body=body, type_ignores=[]))


def test_a_partial_stored_per_element_is_refused() -> None:
    """``v - v.sum()``: each element would be stored minus one lane's sum."""
    src = """
    for vec_lane_1 in cutlass.range_constexpr(4):
        v = cutlass.Float32(_v0[vec_lane_1])
        s = cutlass.Float32(cute.arch.warp_reduction_sum(v, threads_in_group=4))
        d = v - s
        vals.append(d)
    """
    with pytest.raises(exc.BackendUnsupported, match="other than an assignment reads"):
        _hoist(src)


def test_a_partial_carried_across_the_lanes_is_refused() -> None:
    """A carry updated in every lane with one lane's sum adds the partials
    where the program adds the whole sum once per element."""
    src = """
    for vec_lane_1 in cutlass.range_constexpr(4):
        v = cutlass.Float32(_v0[vec_lane_1])
        s = cutlass.Float32(cute.arch.warp_reduction_sum(v, threads_in_group=4))
        acc = acc + cute.math.exp(v) + s
    """
    with pytest.raises(exc.BackendUnsupported, match="acc carries one V lane"):
        _hoist(src)


def test_a_reduce_in_a_branch_of_the_lane_body_is_refused() -> None:
    """A branch around a reduce of the lane's element writes values that mix
    the two; the check does not look into it and refuses the loop."""
    src = """
    for vec_lane_1 in cutlass.range_constexpr(4):
        acc_copy = acc
        v = cutlass.Float32(_v0[vec_lane_1])
        if beta:
            acc = acc_copy + cute.math.exp(v)
        else:
            s = cutlass.Float32(cute.arch.warp_reduction_sum(v, threads_in_group=4))
            acc = acc_copy + s
    """
    with pytest.raises(exc.BackendUnsupported, match="other than an assignment reads"):
        _hoist(src)


def test_partials_folded_by_later_reduces_and_lane_uniform_merges_stay() -> None:
    """Welford over a chunk per V lane: the centered squares fold into the
    second reduce and the merge reads values the same in every lane, so the
    loop stays as it is (the count's Int64 reduce input keeps the hoist out,
    as in the emitted Welford of test_cute_sibling_layout_safety.py)."""
    src = textwrap.dedent(
        """
        for vec_lane_1 in cutlass.range_constexpr(4):
            chunk = cutlass.Float32(_v0[vec_lane_1])
            valid = cutlass.Int32(lane_base + vec_lane_1 < n)
            n_c = cutlass.Int64(cute.arch.warp_reduction_sum(cutlass.Int64(valid), threads_in_group=4))
            total = cutlass.Float32(cute.arch.warp_reduction_sum(chunk, threads_in_group=4))
            mean_c = total / cutlass.Float32(n_c)
            centered = chunk - mean_c
            square = centered * centered
            m2_c = cutlass.Float32(cute.arch.warp_reduction_sum(square, threads_in_group=4))
            delta = mean_c - mean
            count = cnt + cutlass.Float32(n_c)
            mean = mean + delta * cutlass.Float32(n_c) / count
            m2 = m2 + m2_c + delta * delta * cnt * cutlass.Float32(n_c) / count
            cnt = count
        """
    )
    assert _hoist(src) == ast.unparse(ast.parse(src))


@helion.kernel(backend="cute", static_shapes=True)
def _center_rows(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    bn = hl.register_block_size(n)
    out = torch.empty_like(x)
    for tile_m in hl.tile(m):
        for tile_n in hl.tile(n, block_size=bn):
            v = x[tile_m, tile_n]
            out[tile_m, tile_n] = v - v.sum(-1, keepdim=True)
    return out


@pytest.mark.parametrize("vector", [1, 4])
def test_a_tile_sum_read_per_element_is_not_split_by_vector_lanes(vector: int) -> None:
    """Four threads hold a 16-column tile: as a lane loop (one element per
    lane) the sum is complete before its elements read it; as four vector
    lanes the reduce would stay in the V-loop and sum 4 of the 16."""
    x = torch.empty(64, 64, dtype=torch.float32)
    config = helion.Config(
        block_sizes=[16, 1], num_threads=[4, 1], cute_vector_widths=[vector, 1]
    )
    with mock_cuda_unavailable(), cpu_render_target():
        bound = cpu_bind(_center_rows, (x,))
        config = bound.config_spec.normalized_config(config)
        if vector == 1:
            assert "warp_reduction_sum(sum_1_lane_acc" in bound.to_code(config)
            return
        with pytest.raises(exc.BackendUnsupported, match="vector lane loop"):
            bound.to_code(config)


def test_a_partial_combined_with_the_tiles_extent_is_refused() -> None:
    """Inductor's looped Welford, its V-loop kept by an element store beside
    two reductions: each lane's sum, divided by the tile's count that the
    statements ahead of the loop compute from tile_end, is taken for the
    tile's mean and merged with the tile's weight once per lane."""
    src = """
    for tile_offset_1 in range(0, 3136, _BLOCK_SIZE_1):
        tile_end = cutlass.Int32(tile_offset_1 + _BLOCK_SIZE_1)
        count = cutlass.Float32(1.0 * (tile_end + -1 * tile_offset_1 + 0.0))
        scale = 1.0 / (_BLOCK_SIZE_1 + 0.0)
        weight = count * scale
        share = count / (acc2 + count)
        for vec_lane_1 in cutlass.range_constexpr(4):
            v = cutlass.Float32(_v0[vec_lane_1])
            (out0.iterator + cutlass.Int32(lane_base + vec_lane_1)).store(cutlass.BFloat16(v))
            scaled = v * scale
            s = cutlass.Float32(cute.arch.warp_reduction_sum(scaled, threads_in_group=32))
            mean_c = s / weight
            d = v - mean_c
            square = d * d
            m2_c = cutlass.Float32(cute.arch.warp_reduction_sum(square, threads_in_group=32))
            acc = acc + (mean_c - acc) * share
            acc1 = acc1 + m2_c
        acc2 = acc2 + count
    """
    with pytest.raises(exc.BackendUnsupported, match="combined with the tile's extent"):
        _hoist(src)


def test_a_partial_combined_with_a_count_the_loop_computes_is_refused() -> None:
    """The same merge with the tile's count computed inside the V-loop from
    tile_end; a mask compared against tile_end (under a cast) is no count."""
    src = """
    for tile_offset_1 in range(0, 3136, _BLOCK_SIZE_1):
        tile_end = cutlass.Int32(tile_offset_1 + _BLOCK_SIZE_1)
        for vec_lane_1 in cutlass.range_constexpr(4):
            v = cutlass.Float32(_v0[vec_lane_1])
            (out0.iterator + cutlass.Int32(lane_base + vec_lane_1)).store(cutlass.BFloat16(v))
            inside = cutlass.Boolean(lane_base + vec_lane_1 < tile_end)
            kept = v if inside else cutlass.Float32(0)
            s = cutlass.Float32(cute.arch.warp_reduction_sum(kept, threads_in_group=32))
            count = cutlass.Float32(tile_end - tile_offset_1)
            mean_c = s / count
            d = v - mean_c
            square = d * d
            m2_c = cutlass.Float32(cute.arch.warp_reduction_sum(square, threads_in_group=32))
            acc = acc + mean_c
            acc1 = acc1 + m2_c
    """
    with pytest.raises(exc.BackendUnsupported, match=r"tile's extent \(count\)"):
        _hoist(src)


def test_a_sum_of_partials_of_scaled_elements_stays() -> None:
    """Elements scaled by the tile's count ahead of their reduce: each lane's
    partial is a sum, and adding every lane's into the carry gives the
    tile's, so the loop stays (the partial is no extent of the tile)."""
    src = textwrap.dedent(
        """
        for tile_offset_1 in range(0, 3136, _BLOCK_SIZE_1):
            tile_end = cutlass.Int32(tile_offset_1 + _BLOCK_SIZE_1)
            w = cutlass.Float32(1.0) / cutlass.Float32(tile_end - tile_offset_1)
            for vec_lane_1 in cutlass.range_constexpr(4):
                v = cutlass.Float32(_v0[vec_lane_1])
                (out0.iterator + cutlass.Int32(lane_base + vec_lane_1)).store(cutlass.BFloat16(v))
                t = v * w
                s = cutlass.Float32(cute.arch.warp_reduction_sum(t, threads_in_group=32))
                square = t * t
                q = cutlass.Float32(cute.arch.warp_reduction_sum(square, threads_in_group=32))
                total = total + s
                total2 = total2 + q
        """
    )
    assert _hoist(src) == ast.unparse(ast.parse(src))


@helion.kernel(backend="cute", static_shapes=True)
def _welford_with_copy(
    x: torch.Tensor, copy: torch.Tensor, out_mean: torch.Tensor, out_m2: torch.Tensor
) -> None:
    # Inductor's looped Welford of a BatchNorm over its NCHW rdim N * H * W
    # (4 x 96 of each of 8 channels), whose chunk loop also copies the values
    # it reads: x and copy are flat [4 * 8 * 96].
    for tile_x in hl.tile(8):
        acc = hl.zeros([tile_x], dtype=torch.float32)
        acc1 = hl.zeros([tile_x], dtype=torch.float32)
        acc2 = hl.zeros([tile_x], dtype=torch.float32)
        for tile_r in hl.tile(384):
            index = (
                96 * tile_x.index.reshape([tile_x, 1])
                + 768 * (tile_r.index // 96)
                + tile_r.index % 96
            )
            v = x[index]
            copy[index] = v
            weight = 1.0 * (tile_r.end - tile_r.begin)
            scale = 1.0 / tile_r.block_size
            block_sum = torch.sum(v * scale, -1)
            full = hl.full([], weight, dtype=torch.float32)
            block_mean = block_sum / (full * scale)
            d = v - block_mean.reshape([tile_x, 1])
            block_m2 = torch.sum(d * d, -1)
            new_weight = acc2 + weight
            delta = torch.where(acc == block_mean, 0.0, block_mean - acc)
            share = torch.where(new_weight == 0.0, 0.0, full / new_weight)
            acc = acc + delta * share
            acc1 = acc1 + block_m2 + delta * (acc2 * share) * delta
            acc2 = new_weight
        out_mean[tile_x] = acc
        out_m2[tile_x] = acc1


@pytest.mark.parametrize("vector", [1, 4])
def test_a_welford_chunk_loop_that_stores_is_not_split_by_vector_lanes(
    vector: int,
) -> None:
    """With one element per lane the Welford of each chunk is the chunk's;
    as four vector lanes beside the copy's stores, each lane's block sum
    would be read as the chunk's (on GB300 the mean of 3136 values of x + 1
    came out 0.2): the config is refused."""
    x = torch.empty(4 * 8 * 96, dtype=torch.float32)
    args = (x, torch.empty_like(x), torch.empty(8), torch.empty(8))
    config = helion.Config(
        block_sizes=[1, 512], num_threads=[1, 128], cute_vector_widths=[1, vector]
    )
    with cpu_render_target(reduction_smem_budget_bytes=232448):
        bound = cpu_bind(_welford_with_copy, args)
        config = bound.config_spec.normalized_config(config)
        if vector == 1:
            assert "_cute_grouped_reduce" in bound.to_code(config)
            return
        with pytest.raises(exc.BackendUnsupported, match="the tile's extent"):
            bound.to_code(config)
