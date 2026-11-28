"""CuTe lane folds across guarded packets: helion.inductor's NCHW BatchNorm
reads each channel's rdim through ``r // H*W`` and ``r % H*W``, which the
affine vectorizer loads in packets behind a guard; at several lanes per
thread the cross-lane reduction fold has to repeat that guard per lane.
``test_cute_lane_fold_reductions.py`` runs the kernel."""

from __future__ import annotations

import pytest

import helion
from helion._testing import cpu_render_target
from helion._testing.cute_batch_norm import BATCH_NORM_LANES_CONFIG
from helion._testing.cute_batch_norm import batch_norm_inputs
from helion._testing.cute_batch_norm import batch_norm_training
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import tcgen05_mma_support

pytestmark = pytest.mark.cpu_only

REDUCE = "_cute_grouped_reduce_shared_two_stage("


def test_guarded_packet_lanes_fold() -> None:
    """At 64 threads x V8 the 2048 elements of a channel take 4 lanes, each
    loading its packet behind ``if lane_base < 2048 and index < 32768:``.
    Both reductions of the loop (the mean, then the deviations from it) fold
    across the lanes: the guard is register arithmetic, which the fold may
    repeat per lane (it refused the whole config)."""
    with cpu_render_target(
        mma_support=tcgen05_mma_support(), reduction_smem_budget_bytes=232448
    ):
        bound = cpu_bind(batch_norm_training, batch_norm_inputs())
        config = {**bound.config_spec.default_config(), **BATCH_NORM_LANES_CONFIG}
        code = bound.to_code(helion.Config(**config))
    kernel = code[code.index("@cute.kernel") :]
    first = kernel[: kernel.index("out0.iterator")]
    assert first.count(REDUCE) == 2
    assert first.count("_fold_parts_") >= 4
    assert first.count("cutlass.range_constexpr(4)") >= 2
    assert "_helion_affine_load" in first
