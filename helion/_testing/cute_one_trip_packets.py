"""The kernel ``test_cute_one_trip_packets{,_gpu}.py`` generate code for: a
mean over each row of ``x`` and a normalize that reads ``x`` again beside a
residual ``r`` and a per-row weight ``w`` (the shape of a BatchNorm's or a
LayerNorm's normalize, which ``issue_packets_first`` targets)."""

from __future__ import annotations

import torch

import helion
import helion.language as hl


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def normalize_with_residual(
    x: torch.Tensor,
    r: torch.Tensor,
    w: torch.Tensor,
    out: torch.Tensor,
    mean_out: torch.Tensor,
) -> None:
    for tile_x in hl.tile(x.size(0)):
        acc = hl.zeros([tile_x], dtype=torch.float32)
        for tile_r in hl.tile(x.size(1)):
            acc = acc + torch.sum(x[tile_x, tile_r].to(torch.float32), -1)
        mean = acc / x.size(1)
        mean_out[tile_x] = mean
        wv = w[tile_x]
        for tile_r in hl.tile(x.size(1)):
            v = (x[tile_x, tile_r].to(torch.float32) - mean[:, None]) * wv[:, None] + r[
                tile_x, tile_r
            ].to(torch.float32)
            out[tile_x, tile_r] = v.to(out.dtype)


def reference(
    x: torch.Tensor, r: torch.Tensor, w: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(out, mean)`` of ``normalize_with_residual`` in float64."""
    mean = x.double().mean(-1)
    out = (x.double() - mean[:, None]) * w.double()[:, None] + r.double()
    return out, mean


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def running_mean_update(x: torch.Tensor, stat: torch.Tensor) -> None:
    """``stat = 0.9 * stat + 0.1 * x.mean(-1)`` in place, as a BatchNorm
    updates its running mean: ``stat`` is read after the reduction and
    written back."""
    for tile_x in hl.tile(x.size(0)):
        acc = hl.zeros([tile_x], dtype=torch.float32)
        for tile_r in hl.tile(x.size(1)):
            acc = acc + torch.sum(x[tile_x, tile_r], -1)
        stat[tile_x] = stat[tile_x] * 0.9 + acc / x.size(1) * 0.1


def running_mean_reference(x: torch.Tensor, stat: torch.Tensor) -> torch.Tensor:
    return stat.double() * 0.9 + x.double().mean(-1) * 0.1
