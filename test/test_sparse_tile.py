from __future__ import annotations

import unittest

import torch

import helion
from helion._testing import DEVICE
from helion._testing import TestCase
import helion.language as hl


# Logical 2x3 matrix A:
#   A = [[1, 0, 2],
#        [0, 3, 0]]
# The four fixtures below encode this same matrix in DD/DC/CD/CC layouts.
# `values` is always a flat 1-D tensor; ptrs/coords are ordered by
# loop-nesting order (outer-first).
_SHAPE = (2, 3)


def _build_dd() -> hl.SparseTensor:
    values = torch.tensor([1.0, 0.0, 2.0, 0.0, 3.0, 0.0], device=DEVICE)
    return hl.SparseTensor(
        values=values, shape=_SHAPE, ptrs=(None, None), coords=(None, None)
    )


def _build_dc() -> hl.SparseTensor:
    ptrs1 = torch.tensor([0, 2, 3], dtype=torch.int64, device=DEVICE)
    coords1 = torch.tensor([0, 2, 1], dtype=torch.int64, device=DEVICE)
    values = torch.tensor([1.0, 2.0, 3.0], device=DEVICE)
    return hl.SparseTensor(
        values=values,
        shape=_SHAPE,
        ptrs=(None, ptrs1),
        coords=(None, coords1),
    )


def _build_cd() -> hl.SparseTensor:
    ptrs0 = torch.tensor([0, 2], dtype=torch.int64, device=DEVICE)
    coords0 = torch.tensor([0, 1], dtype=torch.int64, device=DEVICE)
    values = torch.tensor([1.0, 0.0, 2.0, 0.0, 3.0, 0.0], device=DEVICE)
    return hl.SparseTensor(
        values=values,
        shape=_SHAPE,
        ptrs=(ptrs0, None),
        coords=(coords0, None),
    )


def _build_cc() -> hl.SparseTensor:
    ptrs_rows = torch.tensor([0, 2], dtype=torch.int64, device=DEVICE)
    coords_rows = torch.tensor([0, 1], dtype=torch.int64, device=DEVICE)
    ptrs_cols = torch.tensor([0, 2, 3], dtype=torch.int64, device=DEVICE)
    coords_cols = torch.tensor([0, 2, 1], dtype=torch.int64, device=DEVICE)
    values = torch.tensor([1.0, 2.0, 3.0], device=DEVICE)
    return hl.SparseTensor(
        values=values,
        shape=_SHAPE,
        ptrs=(ptrs_rows, ptrs_cols),
        coords=(coords_rows, coords_cols),
    )


@helion.kernel(config=helion.Config(block_sizes=[2, 32]))
def spmv_dd(A: hl.SparseTensor, x: torch.Tensor) -> torch.Tensor:
    out = torch.zeros(A.shape[0], dtype=x.dtype, device=x.device)
    for tile_m in hl.sparse_tile(A, dim=0, levelformat="Dense"):
        acc = hl.zeros([tile_m], dtype=x.dtype)
        for tile_k in hl.sparse_tile(tile_m, dim=1, levelformat="Dense"):
            x_val = x[tile_k]
            a_val = tile_k.value 
            acc = acc + (a_val * x_val).sum(dim=-1)
        out[tile_m] = acc
    return out


@helion.kernel(config=helion.Config(block_sizes=[2, 32]))
def spmv_dc(A: hl.SparseTensor, x: torch.Tensor) -> torch.Tensor:
    out = torch.zeros(A.shape[0], dtype=x.dtype, device=x.device)
    for tile_m in hl.sparse_tile(A, dim=0, levelformat="Dense"):
        acc = hl.zeros([tile_m], dtype=x.dtype)
        for tile_k in hl.sparse_tile(tile_m, dim=1, levelformat="Compressed"):
            x_val = x[tile_k]
            a_val = tile_k.value 
            acc = acc + (a_val * x_val).sum(dim=-1)
        out[tile_m] = acc
    return out


@helion.kernel(config=helion.Config(block_sizes=[2, 32]))
def spmv_cd(A: hl.SparseTensor, x: torch.Tensor) -> torch.Tensor:
    out = torch.zeros(A.shape[0], dtype=x.dtype, device=x.device)
    for tile_m in hl.sparse_tile(A, dim=0, levelformat="Compressed"):
        acc = hl.zeros([tile_m], dtype=x.dtype)
        for tile_k in hl.sparse_tile(tile_m, dim=1, levelformat="Dense"):
            x_val = x[tile_k]
            a_val = tile_k.value
            acc = acc + (a_val * x_val).sum(dim=-1)
        out[tile_m] = acc
    return out


@helion.kernel(config=helion.Config(block_sizes=[2, 32]))
def spmv_cc(A: hl.SparseTensor, x: torch.Tensor) -> torch.Tensor:
    out = torch.zeros(A.shape[0], dtype=x.dtype, device=x.device)
    for tile_m in hl.sparse_tile(A, dim=0, levelformat="Compressed"):
        acc = hl.zeros([tile_m], dtype=x.dtype)
        for tile_k in hl.sparse_tile(tile_m, dim=1, levelformat="Compressed"):
            x_val = x[tile_k]
            a_val = tile_k.value
            acc = acc + (a_val * x_val).sum(dim=-1)
        out[tile_m] = acc
    return out


_DENSE_A = torch.tensor([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]], device=DEVICE)


class TestSparseTile(TestCase):
    def _run(self, spmv, A: hl.SparseTensor) -> None:
        x = torch.tensor([1.0, 2.0, 3.0], device=DEVICE)
        bound = spmv.bind((A, x))
        with bound.env:
            print(bound.host_function.debug_str(), flush=True)
        assert len(bound.config_spec.block_sizes) >= 2
        expected = _DENSE_A @ x
        got = spmv(A, x)
        torch.testing.assert_close(got, expected)

    def test_dense_dense(self) -> None:
        self._run(spmv_dd, _build_dd())

    def test_dense_compressed(self) -> None:
        self._run(spmv_dc, _build_dc())


if __name__ == "__main__":
    unittest.main()
