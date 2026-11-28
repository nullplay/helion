"""Metal thread axes of the device loops of a kernel with several roots,
rendered on CPU without a Metal device."""

from __future__ import annotations

import pytest
import torch

import helion
from helion._testing import skipUnlessBackends
from helion._testing.cute_codegen import forbid_cuda_work
import helion.language as hl

pytestmark = [pytest.mark.cpu_only, skipUnlessBackends(["metal"])]


@helion.kernel(backend="metal", static_shapes=True)
def _nested_loops_beside_a_root(
    x: torch.Tensor, z: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    out = torch.empty_like(x)
    out_z = torch.empty_like(z)
    for tile_m, tile_n in hl.tile([x.size(0), x.size(1)], block_size=[2, 2]):
        for tile_k in hl.tile(x.size(2), block_size=2):
            for tile_l in hl.tile(x.size(3), block_size=2):
                out[tile_m, tile_n, tile_k, tile_l] = x[tile_m, tile_n, tile_k, tile_l]
    for tile_p in hl.tile(z.size(0), block_size=2):
        out_z[tile_p] = z[tile_p]
    return out, out_z


@forbid_cuda_work()
def test_nested_device_loops_beside_another_root_stay_within_three_axes() -> None:
    """Metal has no tile-loop paths, so a device loop's budget counts every
    root's grid axes: the 2-D root's two device loops beside a 1-D root keep
    to lane loops instead of taking a fourth thread axis, which Metal's
    thread_index refuses."""
    x = torch.randn(4, 4, 4, 4)
    z = torch.randn(4)
    bound = _nested_loops_beside_a_root.bind((x, z))
    code = bound.to_code(helion.Config(num_warps=1))
    assert "tid[3]" not in code
    # tile_k and tile_l run as lane loops of two
    assert code.count(" in range(2):") == 2
