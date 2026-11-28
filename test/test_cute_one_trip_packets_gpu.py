"""CuTe ``issue_packets_first`` on a GPU: kernels whose one-trip tile loops
share packets, and the cases it must leave alone (an aliased or in-place
residual, packets past the register budget, multi-trip loops), against a
float64 reference.  ``test_cute_one_trip_packets.py`` checks the code."""

from __future__ import annotations

import pytest
import torch

from helion._testing import DEVICE
from helion._testing import code_and_output
from helion._testing import onlyBackends
from helion._testing import skipIfNotCUDA
from helion._testing import skipUnlessCuteAvailable
from helion._testing.cute_batch_norm import batch_norm_inputs
from helion._testing.cute_batch_norm import batch_norm_reference
from helion._testing.cute_batch_norm import batch_norm_training
from helion._testing.cute_one_trip_packets import normalize_with_residual
from helion._testing.cute_one_trip_packets import reference
from helion._testing.cute_one_trip_packets import running_mean_reference
from helion._testing.cute_one_trip_packets import running_mean_update


def _config(threads: int, block: int, vector: int = 8) -> dict[str, object]:
    return {
        "block_sizes": [1, block, block],
        "num_threads": [1, threads, threads],
        "cute_vector_widths": [1, vector, vector],
        "cute_lane_layouts": ["blocked", "strided", "strided"],
    }


def _check(out: torch.Tensor, mean: torch.Tensor, x, r, w) -> None:
    ref_out, ref_mean = reference(x, r, w)
    torch.testing.assert_close(mean.double(), ref_mean, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(out.double(), ref_out, rtol=2e-2, atol=2e-2)


@skipIfNotCUDA()
@skipUnlessCuteAvailable("requires CuTe DSL")
@onlyBackends(["cute"])
@pytest.mark.parametrize(
    ("threads", "vector", "block", "rdim", "shared"),
    [
        # One trip, 4 lanes: x and the residual come from registers.
        (64, 8, 2048, 1536, True),
        # 8 lanes of 32 bytes: x fills the register budget, the residual
        # reloads.
        (16, 16, 2048, 1536, True),
        # Three trips of 512: nothing is shared.
        (64, 8, 512, 1536, False),
        # An rdim no packet divides: the tail lanes keep their scalar path.
        (64, 8, 2048, 1500, True),
    ],
)
def test_normalize_with_residual(
    threads: int, vector: int, block: int, rdim: int, shared: bool
) -> None:
    torch.manual_seed(0)
    x = torch.randn(8, rdim, device=DEVICE, dtype=torch.bfloat16)
    r = torch.randn(8, rdim, device=DEVICE, dtype=torch.bfloat16)
    w = torch.randn(8, device=DEVICE, dtype=torch.float32)
    out = torch.empty_like(x)
    mean = torch.empty(8, device=DEVICE, dtype=torch.float32)
    code, _ = code_and_output(
        normalize_with_residual,
        (x, r, w, out, mean),
        **_config(threads, block, vector),
    )
    assert ("_one_trip_packets_0" in code) == shared
    _check(out, mean, x, r, w)


@skipIfNotCUDA()
@skipUnlessCuteAvailable("requires CuTe DSL")
@onlyBackends(["cute"])
def test_in_place_residual_keeps_its_loads() -> None:
    # out is r: the kernel writes the residual, so its packets load where
    # they are read (the x packets, read-only and disjoint, are shared).
    torch.manual_seed(0)
    x = torch.randn(8, 1536, device=DEVICE, dtype=torch.bfloat16)
    r = torch.randn(8, 1536, device=DEVICE, dtype=torch.bfloat16)
    w = torch.randn(8, device=DEVICE, dtype=torch.float32)
    r0 = r.clone()
    mean = torch.empty(8, device=DEVICE, dtype=torch.float32)
    code, _ = code_and_output(
        normalize_with_residual, (x, r, w, r, mean), **_config(64, 2048)
    )
    assert "_one_trip_packets_0" in code
    assert "_one_trip_packets_1" not in code
    _check(r, mean, x, r0, w)


@skipIfNotCUDA()
@skipUnlessCuteAvailable("requires CuTe DSL")
@onlyBackends(["cute"])
def test_overlapping_views_are_not_shared() -> None:
    # x and the output are views of one storage (out = x[:, 1:] shifted by
    # one element): no packet of x is proven apart from the writes.
    torch.manual_seed(0)
    base = torch.randn(8, 1537, device=DEVICE, dtype=torch.bfloat16)
    x = base[:, :1536]
    out = base[:, 1:]
    expected_x = x.clone()
    r = torch.randn(8, 1536, device=DEVICE, dtype=torch.bfloat16)
    w = torch.randn(8, device=DEVICE, dtype=torch.float32)
    mean = torch.empty(8, device=DEVICE, dtype=torch.float32)
    code, _ = code_and_output(
        normalize_with_residual, (x, r, w, out, mean), **_config(64, 2048)
    )
    # The residual (read-only, apart from every write) is still prefetched;
    # x keeps its loads in the normalize.
    shared = [
        line
        for line in code.splitlines()
        if "_one_trip_packets_" in line and "_lane = " in line
    ]
    assert shared
    assert all("x.iterator" not in line for line in shared)
    normalize = code[code.index("tile_offset_2 = cutlass.Int32(0)") :]
    assert "x.iterator" in normalize
    ref_mean = expected_x.double().mean(-1)
    torch.testing.assert_close(mean.double(), ref_mean, rtol=1e-5, atol=1e-5)


@skipIfNotCUDA()
@skipUnlessCuteAvailable("requires CuTe DSL")
@onlyBackends(["cute"])
@pytest.mark.parametrize("threads", [64, 128])
def test_running_mean_read_first_and_written_last(threads: int) -> None:
    # A running statistic read after the block reduction and written back in
    # place: its load moves to the top, its store stays last.
    torch.manual_seed(0)
    x = torch.randn(16, 1024, device=DEVICE, dtype=torch.float32)
    stat = torch.randn(16, device=DEVICE, dtype=torch.float32)
    expected = running_mean_reference(x, stat)
    code, _ = code_and_output(
        running_mean_update,
        (x, stat),
        block_sizes=[1, 1024],
        num_threads=[1, threads],
        cute_vector_widths=[1, 4],
        cute_lane_layouts=["blocked", "strided"],
    )
    kernel = code[code.index("@cute.kernel") :]
    assert kernel.index("stat.iterator") < kernel.index("_cute_grouped_reduce")
    torch.testing.assert_close(stat.double(), expected, rtol=1e-5, atol=1e-5)


@skipIfNotCUDA()
@skipUnlessCuteAvailable("requires CuTe DSL")
@onlyBackends(["cute"])
def test_batch_norm_training() -> None:
    threads = 64
    # Running statistics read first and written last, input and residual
    # packets shared by the mean, the deviations and the normalize.
    torch.manual_seed(0)
    inputs = batch_norm_inputs(DEVICE)
    expected = batch_norm_reference(inputs)
    code, _ = code_and_output(
        batch_norm_training,
        inputs,
        block_sizes=[1, 2048, 2048],
        num_threads=[1, threads, threads],
        cute_vector_widths=[1, 8, 8],
        cute_lane_layouts=["blocked", "strided", "strided"],
    )
    assert "_one_trip_packets_0" in code
    names = ("in0", "in1", "in2", "in3", "in4", "in5", "out0", "out1", "out2")
    got = dict(zip(names, inputs, strict=True))
    for name in ("out0", "out1", "in1", "in2"):
        torch.testing.assert_close(
            got[name].double(), expected[name], rtol=1e-4, atol=1e-5
        )
    torch.testing.assert_close(
        got["out2"].double(), expected["out2"], rtol=1e-2, atol=2e-2
    )
