from __future__ import annotations

import unittest

import torch

import helion
from helion._testing import DEVICE
from helion._testing import TestCase
import helion.language as hl


class TestSparseTile(TestCase):
    def test_value_returns_correct_shape(self) -> None:
        """tile_k.value and x[tile_k] both produce [P0, P1] shaped tensors."""
        ptr = torch.tensor([0, 2, 3, 5], dtype=torch.int64, device=DEVICE)
        coord = torch.tensor([0, 2, 1, 0, 3], dtype=torch.int64, device=DEVICE)
        values = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device=DEVICE)
        A = hl.SparseTensor(ptr=ptr, coord=coord, values=values, shape=(3, 4))
        x = torch.ones(4, device=DEVICE)

        @helion.kernel
        def spmv(A: hl.SparseTensor, x: torch.Tensor) -> torch.Tensor:
            out = torch.zeros(A.shape[0], dtype=x.dtype, device=x.device)
            for tile_m in hl.sparse_tile(A, dim=0, levelformat="Dense"):
                acc = hl.zeros([tile_m], dtype=x.dtype)
                for tile_k in hl.sparse_tile(tile_m, dim=1, levelformat="Compressed"):
                    x_val = x[tile_k+1]          # [P0, P1] gather from x
                    a_val = tile_k.value + 2       # [P0, P1] sparse values
                    acc = acc + (a_val * x_val).sum(dim=-1)
                out[tile_m] = acc
            return out

        # bind() runs type propagation — if it completes without error the
        # shapes of x_val, a_val, and acc are all consistent.
        bound = spmv.bind((A, x))
        # Print the typed AST + device IR graphs for inspection.
        # debug_str() calls CompileEnvironment.current() so must run inside env.
        with bound.env:
            print(bound.host_function.debug_str(), flush=True)
        # Outer (tile_m) + inner (tile_k) block sizes must both be registered.
        assert len(bound.config_spec.block_sizes) >= 2


if __name__ == "__main__":
    unittest.main()
