from __future__ import annotations

import functools
import pathlib
import unittest

import pytest
import torch
from torch._inductor import config
from torch._inductor.custom_graph_pass import CustomGraphModulePass
from torch.fx.experimental.proxy_tensor import make_fx

from helion._compat import requires_torch_version
from helion._testing import TestCase

if not requires_torch_version("2.16"):
    pytest.skip("helion.inductor needs torch 2.16", allow_module_level=True)

import helion.inductor
from helion.inductor import config as helion_config
from helion.inductor.fx_passes.post_grad import HelionPostGradPass
from helion.inductor.fx_passes.strided_reductions import split_strided_reductions

pytestmark = pytest.mark.cpu_only

aten = torch.ops.aten


class Recorder(CustomGraphModulePass):
    def __init__(self, uuid: object = "recorder") -> None:
        self.graphs: list[torch.fx.GraphModule] = []
        self._uuid = uuid

    def __call__(self, gm: torch.fx.GraphModule) -> None:
        self.graphs.append(gm)

    def uuid(self) -> object:
        return self._uuid


class TestSourceHash(TestCase):
    def test_covers_subpackages(self):
        # _source_hash, in Inductor's FX graph cache key, hashes every module of
        # helion.inductor: a change to the codegen in codegen/ or to a pass in
        # fx_passes/ misses the cache, as one to a module at the top does.
        package = pathlib.Path(helion.inductor.__file__).parent
        hashed = {
            path.relative_to(package).as_posix()
            for path in helion_config._source_files()
        }
        expected = {
            "codegen/kernel.py",
            "codegen/overrides.py",
            "codegen/scheduling.py",
            "fx_passes/online_softmax.py",
            "fx_passes/post_grad.py",
            "fx_passes/strided_reductions.py",
        }
        self.assertLessEqual(expected, hashed)


class TestPostGradPass(TestCase):
    def test_chain(self):
        # The pass registered before it still runs, also for cuda_backend
        # "triton", and its uuid stays in the FX graph cache key.
        inner = Recorder()
        helion_pass = HelionPostGradPass(inner)
        gm = torch.fx.symbolic_trace(lambda x: x + 1)
        with config.patch(cuda_backend="triton"):
            helion_pass(gm)
        self.assertEqual(inner.graphs, [gm])
        self.assertEqual(helion_pass.uuid()[1], "recorder")
        # An inner pass that asks to skip caching (uuid None) still does.
        self.assertIsNone(HelionPostGradPass(Recorder(uuid=None)).uuid())


class TestStridedReductions(TestCase):
    def split(self, fn, x):
        """fn traced with fake tensors, after split_strided_reductions(): its
        reductions, as (target, dims) pairs, and its result on x against
        fn(x)'s."""
        gm = make_fx(fn, tracing_mode="fake")(x)
        split_strided_reductions(gm.graph, devices=("cpu",))
        gm.recompile()
        torch.testing.assert_close(gm(x), fn(x), rtol=1e-5, atol=1e-5)
        return [
            (node.target.__name__, node.args[1])
            for node in gm.graph.nodes
            if node.op == "call_function"
            and node.target
            in (aten.var_mean.correction, aten.sum.dim_IntList, aten.mean.dim)
        ]

    def test_var_mean_nchw(self):
        # BatchNorm's var_mean over N, H and W: H and W are one contiguous run,
        # reduced first, and N is reduced over the rows' means and variances:
        # the mean of the means, of their deviations from it, of the squares
        # of those, and of the variances.
        x = torch.randn(32, 3, 8, 8) * 3 + 5
        for correction in (0, 1):
            fn = functools.partial(
                aten.var_mean.correction,
                dim=[0, 2, 3],
                correction=correction,
                keepdim=True,
            )
            self.assertEqual(
                self.split(fn, x),
                [
                    ("var_mean.correction", [2, 3]),
                    ("sum.dim_IntList", [2, 3]),
                    ("sum.dim_IntList", [0]),
                    ("sum.dim_IntList", [0]),
                    ("sum.dim_IntList", [0]),
                    ("sum.dim_IntList", [0]),
                    ("sum.dim_IntList", [0]),
                    ("sum.dim_IntList", [0]),
                    ("sum.dim_IntList", [0]),
                ],
            )

    def test_half_partials(self):
        # float16 partials of the rows of 64 values of 2048 would be inf; in
        # float32 the sum over N, H and W is 0.
        x = torch.full((2, 3, 8, 8), 2048.0, dtype=torch.float16)
        x[1] = -2048.0
        self.split(functools.partial(aten.sum.dim_IntList, dim=[0, 2, 3]), x)
        self.split(
            functools.partial(aten.sum.dim_IntList, dim=[0, 2, 3], dtype=torch.float16),
            x,
        )

    def test_rounded_mean(self):
        # Rows of 100553424, whose mean of 1536 values rounds to 100553432:
        # the variance is 0 and the mean within a float32 ulp.
        x = torch.full((24, 3, 8, 8), 100553424.0)
        fn = functools.partial(aten.var_mean.correction, dim=[0, 2, 3], correction=0)
        gm = make_fx(fn, tracing_mode="fake")(x)
        split_strided_reductions(gm.graph, devices=("cpu",))
        gm.recompile()
        var, mean = gm(x)
        torch.testing.assert_close(var, torch.zeros(3), rtol=0, atol=0)
        torch.testing.assert_close(mean, x[0, :, 0, 0], rtol=1.2e-7, atol=0)
        # Rows of 1e8 and 1e8 + 8, whose mean, 1e8 + 4, rounds to 1e8 twice:
        # the deviations from it are 0 and 8, of variance 16.
        x = torch.full((2, 3, 1, 1024), 1e8)
        x[1] += 8
        gm = make_fx(fn, tracing_mode="fake")(x)
        split_strided_reductions(gm.graph, devices=("cpu",))
        gm.recompile()
        var, _ = gm(x)
        torch.testing.assert_close(var, torch.full((3,), 16.0))
        # Rows of 1e20 and 0, whose deviations square past float32's largest
        # value, and an empty reduction with a negative correction.
        x = torch.zeros(64, 3, 1, 32)
        x[0] = 1e20
        gm = make_fx(fn, tracing_mode="fake")(x)
        split_strided_reductions(gm.graph, devices=("cpu",))
        gm.recompile()
        torch.testing.assert_close(gm(x), fn(x))
        # Ones but for a first value of 1e8 and of -1e8, which x - shift
        # would round the ones away against: the mean is the ones', 0.999,
        # to the rows' float32 means (97657.25 for 97657.249).
        x = torch.ones(2, 3, 1, 1024)
        x[0, :, 0, 0], x[1, :, 0, 0] = 1e8, -1e8
        gm = make_fx(fn, tracing_mode="fake")(x)
        split_strided_reductions(gm.graph, devices=("cpu",))
        gm.recompile()
        torch.testing.assert_close(gm(x)[1], fn(x)[1], rtol=0, atol=2e-3)
        empty = functools.partial(
            aten.var_mean.correction, dim=[0, 2, 3], correction=-1
        )
        x = torch.zeros(0, 3, 8, 8)
        gm = make_fx(empty, tracing_mode="fake")(x)
        self.assertFalse(split_strided_reductions(gm.graph, devices=("cpu",)))
        # Rows of 1e8 and 1e8 + 8 each, and of 1e8 alone, whose means would
        # round to 1e8 unshifted, losing the rows' spread (8 against 12).
        for x in (torch.full((2, 3, 1, 32), 1e8), torch.full((64, 3, 1, 32), 1e8)):
            x[1::2, ..., 1::2] += 8
            gm = make_fx(fn, tracing_mode="fake")(x)
            split_strided_reductions(gm.graph, devices=("cpu",))
            gm.recompile()
            torch.testing.assert_close(gm(x), fn(x))
        # A 0-d input, whose sum over dim 0 is valid.
        gm = make_fx(
            functools.partial(aten.sum.dim_IntList, dim=[0]), tracing_mode="fake"
        )(torch.tensor(1.0))
        self.assertFalse(split_strided_reductions(gm.graph, devices=("cpu",)))
        # Rows of 3e38 and 0, whose shifted rows would sum past float32's
        # largest value: the variance is inf, as eager's.
        x = torch.zeros(2, 3, 8, 8)
        x[0] = 3e38
        gm = make_fx(fn, tracing_mode="fake")(x)
        split_strided_reductions(gm.graph, devices=("cpu",))
        gm.recompile()
        torch.testing.assert_close(gm(x), fn(x))
        # Rows of 1e37, whose 64 means sum past float32's largest value.
        x = torch.full((64, 3, 8, 8), 1e37)
        gm = make_fx(fn, tracing_mode="fake")(x)
        split_strided_reductions(gm.graph, devices=("cpu",))
        gm.recompile()
        var, mean = gm(x)
        torch.testing.assert_close(var, torch.zeros(3), rtol=0, atol=0)
        torch.testing.assert_close(mean, x[0, :, 0, 0])

    def test_size_one_dim(self):
        # keepdim=False drops every reduced dim, the one of size 1 too.
        x = torch.randn(4, 3, 1, 64)
        for target in (aten.sum.dim_IntList, aten.mean.dim):
            self.split(functools.partial(target, dim=[0, 2, 3]), x)

    def test_sum_nchw(self):
        # keepdim=False squeezes the reduced dims at the end.
        x = torch.randn(4, 3, 8, 8)
        fn = lambda x: aten.sum.dim_IntList(x, [0, 2, 3])  # noqa: E731
        self.assertEqual(
            self.split(fn, x), [("sum.dim_IntList", [2, 3]), ("sum.dim_IntList", [0])]
        )

    def test_unsplit(self):
        # NHWC's N, H and W are one run (of stride C), as are the last dims of
        # a row reduction; a run of 16 values is too short.
        cases = [
            (torch.randn(4, 3, 8, 8).to(memory_format=torch.channels_last), [0, 2, 3]),
            (torch.randn(4, 3, 8, 8), [1, 2, 3]),
            (torch.randn(4, 3, 4, 4), [0, 2, 3]),
        ]
        for x, dims in cases:
            fn = functools.partial(aten.sum.dim_IntList, dim=dims)
            self.assertEqual(self.split(fn, x), [("sum.dim_IntList", dims)])


if __name__ == "__main__":
    unittest.main()
