"""A BatchNorm in training over an NCHW [32, 16, 8, 8] bf16 tensor, as
helion.inductor emits it (better-benchmark var_mean_0c1c9fc832f4 at a smaller
shape), for the CuTe tests of its multi-lane configs
(``test_cute_guarded_packet_fold.py``, ``test_cute_lane_fold_reductions.py``):
flat views, each channel's rdim of N * H * W = 2048 read through r // 64 and
r % 64 (guarded packets), mean and deviations in one loop over the rdim, the
running variance and mean (in1, in2) updated in place, and a normalize that
reads the input again beside the weight and bias (in3, in4) and a residual
(in5)."""

from __future__ import annotations

import torch

import helion
import helion.language as hl

BN_N, BN_C, BN_HW = 32, 16, 64
# 64 threads x 8-element packets: 4 lanes per thread over one 2048-element
# tile of each loop.
BATCH_NORM_LANES_CONFIG = {
    "block_sizes": [1, 2048, 2048],
    "num_threads": [1, 64, 64],
    "cute_vector_widths": [1, 8, 8],
    "cute_lane_layouts": ["blocked", "strided", "strided"],
}


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def batch_norm_training(
    in0: torch.Tensor,
    in1: torch.Tensor,
    in2: torch.Tensor,
    in3: torch.Tensor,
    in4: torch.Tensor,
    in5: torch.Tensor,
    out0: torch.Tensor,
    out1: torch.Tensor,
    out2: torch.Tensor,
) -> None:
    for tile_x in hl.tile(16):
        acc = hl.zeros([tile_x], dtype=torch.float32)
        acc1 = hl.zeros([tile_x], dtype=torch.float32)
        acc2 = hl.zeros([tile_x], dtype=torch.float32)
        for tile_r in hl.tile(2048):
            tmp0 = in0[
                64 * tile_x.index[:, None]
                + 1024 * (tile_r.index // 64)
                + (tile_r.index % 64)
            ].to(torch.float32)
            tmp1 = 0.0
            tmp2 = 1.0
            tmp3 = tmp2 * (tile_r.end - tile_r.begin)
            tmp4 = tmp2 / tile_r.block_size
            tmp5 = tmp0 * tmp4
            tmp6 = torch.sum(tmp5, -1)
            tmp7 = hl.full([], tmp3, dtype=torch.float32)
            tmp8 = tmp7 * tmp4
            tmp9 = tmp6 / tmp8
            tmp10 = tmp9[:, None]
            tmp11 = tmp0 - tmp10
            tmp12 = tmp11 * tmp11
            tmp13 = torch.sum(tmp12, -1)
            tmp14 = acc2 + tmp3
            tmp15 = acc == tmp9
            tmp16 = tmp9 - acc
            tmp17 = torch.where(tmp15, tmp1, tmp16)
            tmp18 = tmp14 == tmp1
            tmp19 = tmp7 / tmp14
            tmp20 = torch.where(tmp18, tmp1, tmp19)
            tmp21 = tmp17 * tmp20
            tmp22 = acc + tmp21
            tmp23 = acc1 + tmp13
            tmp24 = acc2 * tmp20
            tmp25 = tmp17 * tmp24
            tmp26 = tmp25 * tmp17
            tmp27 = tmp23 + tmp26
            acc = tmp22
            acc1 = tmp27
            acc2 = tmp14
        out0[tile_x] = acc
        tmp37 = in1[tile_x]
        tmp42 = in2[tile_x]
        tmp28 = 2048.0
        tmp29 = acc1 / tmp28
        tmp30 = 1e-05
        tmp31 = tmp29 + tmp30
        tmp32 = torch.rsqrt(tmp31)
        tmp33 = 1.0004885197850513
        tmp34 = tmp29 * tmp33
        tmp35 = 0.1
        tmp36 = tmp34 * tmp35
        tmp38 = 0.9
        tmp39 = tmp37 * tmp38
        tmp40 = tmp36 + tmp39
        tmp41 = acc * tmp35
        tmp43 = tmp42 * tmp38
        tmp44 = tmp41 + tmp43
        out1[tile_x] = tmp32
        in1[tile_x] = tmp40
        in2[tile_x] = tmp44
        for tile_r in hl.tile(2048):
            tmp45 = in0[
                64 * tile_x.index[:, None]
                + 1024 * (tile_r.index // 64)
                + (tile_r.index % 64)
            ].to(torch.float32)
            tmp50 = in3[tile_x]
            tmp53 = in4[tile_x]
            tmp56 = in5[
                64 * tile_x.index[:, None]
                + 1024 * (tile_r.index // 64)
                + (tile_r.index % 64)
            ].to(torch.float32)
            tmp46 = acc[:, None]
            tmp47 = tmp45 - tmp46
            tmp48 = tmp32[:, None]
            tmp49 = tmp47 * tmp48
            tmp51 = tmp50[:, None]
            tmp52 = tmp49 * tmp51
            tmp54 = tmp53[:, None]
            tmp55 = tmp52 + tmp54
            tmp57 = tmp55 + tmp56
            tmp58 = torch.relu(tmp57)
            out2[
                64 * tile_x.index[:, None]
                + 1024 * (tile_r.index // 64)
                + (tile_r.index % 64)
            ] = tmp58.to(torch.bfloat16)


def batch_norm_inputs(device: torch.device | str = "cpu") -> tuple[torch.Tensor, ...]:
    numel = BN_N * BN_C * BN_HW
    x = torch.randn(numel, device=device).to(torch.bfloat16)
    res = torch.randn(numel, device=device).to(torch.bfloat16)
    stats = [torch.randn(BN_C, device=device) for _ in range(4)]
    out0 = torch.empty(BN_C, device=device)
    out1 = torch.empty(BN_C, device=device)
    out2 = torch.empty_like(x)
    return (x, *stats, res, out0, out1, out2)


def batch_norm_reference(
    inputs: tuple[torch.Tensor, ...],
) -> dict[str, torch.Tensor]:
    """The float64 results of ``batch_norm_training`` on (copies of) its
    ``inputs``: the outputs and the updated running statistics."""
    x, rv, rm, w, b, res = (t.double() for t in inputs[:6])
    xs = x.view(BN_N, BN_C, BN_HW)
    mean = xs.mean((0, 2))
    var = xs.var((0, 2), unbiased=False)
    rstd = torch.rsqrt(var + 1e-5)
    y = (xs - mean[None, :, None]) * rstd[None, :, None] * w[None, :, None]
    y = (y + b[None, :, None] + res.view(BN_N, BN_C, BN_HW)).relu()
    n = BN_N * BN_HW
    return {
        "out0": mean,
        "out1": rstd,
        "out2": y.reshape(-1),
        "in1": var * (n / (n - 1)) * 0.1 + rv * 0.9,
        "in2": mean * 0.1 + rm * 0.9,
    }
