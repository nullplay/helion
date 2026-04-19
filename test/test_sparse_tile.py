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
        acc = hl.zeros([tile_m.size(0)], dtype=x.dtype)
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
        acc = hl.zeros([tile_m.size(0)], dtype=x.dtype)
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
        acc = hl.zeros([tile_m.size(0)], dtype=x.dtype)
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
        acc = hl.zeros([tile_m.size(0)], dtype=x.dtype)
        for tile_k in hl.sparse_tile(tile_m, dim=1, levelformat="Compressed"):
            x_val = x[tile_k]
            a_val = tile_k.value
            acc = acc + (a_val * x_val).sum(dim=-1)
        out[tile_m] = acc
    return out


_DENSE_A = torch.tensor([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]], device=DEVICE)
_B = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], device=DEVICE)


@helion.kernel(config=helion.Config(block_sizes=[2, 2, 32]))
def spmm_cc(A: hl.SparseTensor, B: torch.Tensor) -> torch.Tensor:
    M = A.shape[0]
    N = B.size(1)
    C = torch.zeros(M, N, dtype=B.dtype, device=B.device)
    for tile_n in hl.tile(N):
        for tile_m in hl.sparse_tile(A, dim=0, levelformat="Compressed"):
            acc = hl.zeros([tile_m.size(0), tile_n], dtype=B.dtype)
            for tile_k in hl.sparse_tile(tile_m, dim=1, levelformat="Compressed"):
                a_val = tile_k.value
                b_val = B[tile_k[:, :, None], tile_n.index[None, None, :]]
                acc = acc + (a_val.unsqueeze(-1) * b_val).sum(dim=1)
            C[tile_m, tile_n] = acc
    return C


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

    def test_compressed_dense(self) -> None:
        self._run(spmv_cd, _build_cd())

    def test_compressed_compressed(self) -> None:
        self._run(spmv_cc, _build_cc())

    def test_spmm_cc(self) -> None:
        A = _build_cc()
        expected = _DENSE_A @ _B
        got = spmm_cc(A, _B)
        torch.testing.assert_close(got, expected)


# --- 3D fixture: DDC (Dense, Dense, Compressed along dim=2 per (i,j)) ---------
# Logical A[2, 2, 3]:
#   A[0,0,:] = [1, 0, 2]
#   A[0,1,:] = [0, 3, 0]
#   A[1,0,:] = [0, 0, 5]
#   A[1,1,:] = [4, 6, 0]
# Per-(i,j) nnz in row-major (i,j) order: 2, 1, 1, 2 → total nnz = 6.
_SHAPE_3D = (2, 2, 3)


def _build_ddc() -> hl.SparseTensor:
    # DCD layout: Compressed at level 1 (dim=1). All 4 rows are non-zero.
    ptrs1 = torch.tensor([0, 2, 4], dtype=torch.int64, device=DEVICE)
    coords1 = torch.tensor([0, 1, 0, 1], dtype=torch.int64, device=DEVICE)
    # 4 stored rows × 3 cols (Dense at dim=2), row-major:
    values = torch.tensor(
        [1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 0.0, 0.0, 5.0, 4.0, 6.0, 0.0],
        device=DEVICE,
    )
    return hl.SparseTensor(
        values=values,
        shape=_SHAPE_3D,
        ptrs=(None, ptrs1, None),
        coords=(None, coords1, None),
    )


_DENSE_A_3D = torch.tensor(
    [
        [[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]],
        [[0.0, 0.0, 5.0], [4.0, 6.0, 0.0]],
    ],
    device=DEVICE,
)


@helion.kernel(config=helion.Config(block_sizes=[2, 2, 32]))
def sdot_ddc(A: hl.SparseTensor, B: torch.Tensor) -> torch.Tensor:
    I = A.shape[0]
    J = A.shape[1]
    C = torch.zeros(I * J, dtype=B.dtype, device=B.device)
    for tile_i in hl.sparse_tile(A, dim=0, levelformat="Dense"):
        for tile_j in hl.sparse_tile(tile_i, dim=1, levelformat="Compressed"):
            acc = hl.zeros([tile_i.size(0), tile_j.size(1)], dtype=B.dtype)
            for tile_k in hl.sparse_tile(tile_j, dim=2, levelformat="Dense"):
                a_val = tile_k.value
                b_val = B[tile_k]
                acc = acc + (a_val * b_val).sum(dim=-1)
            flat_idx = tile_i[:, None] * J + tile_j
            C[flat_idx] = acc
    return C.view(I, J)


def _build_ccd() -> hl.SparseTensor:
    # CCD layout: Compressed at level 0 (dim=0) and level 1 (dim=1), Dense at
    # level 2 (dim=2). All 2 outer rows and all 4 (i,j) rows are non-zero.
    ptrs0 = torch.tensor([0, 2], dtype=torch.int64, device=DEVICE)
    coords0 = torch.tensor([0, 1], dtype=torch.int64, device=DEVICE)
    ptrs1 = torch.tensor([0, 2, 4], dtype=torch.int64, device=DEVICE)
    coords1 = torch.tensor([0, 1, 0, 1], dtype=torch.int64, device=DEVICE)
    # 4 stored (i,j) rows × 3 cols (Dense at dim=2), row-major:
    values = torch.tensor(
        [1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 0.0, 0.0, 5.0, 4.0, 6.0, 0.0],
        device=DEVICE,
    )
    return hl.SparseTensor(
        values=values,
        shape=_SHAPE_3D,
        ptrs=(ptrs0, ptrs1, None),
        coords=(coords0, coords1, None),
    )


@helion.kernel(config=helion.Config(block_sizes=[2, 2, 32]))
def sdot_ccd(A: hl.SparseTensor, B: torch.Tensor) -> torch.Tensor:
    I = A.shape[0]
    J = A.shape[1]
    C = torch.zeros(I * J, dtype=B.dtype, device=B.device)
    for tile_i in hl.sparse_tile(A, dim=0, levelformat="Compressed"):
        for tile_j in hl.sparse_tile(tile_i, dim=1, levelformat="Compressed"):
            acc = hl.zeros([tile_i.size(0), tile_j.size(1)], dtype=B.dtype)
            for tile_k in hl.sparse_tile(tile_j, dim=2, levelformat="Dense"):
                a_val = tile_k.value
                b_val = B[tile_k]
                acc = acc + (a_val * b_val).sum(dim=-1)
            flat_idx = tile_i[:, None] * J + tile_j
            C[flat_idx] = acc
    return C.view(I, J)


class TestSparseTile3D(TestCase):
    def test_ddc_sdot(self) -> None:
        A = _build_ddc()
        B = torch.tensor([1.0, 2.0, 3.0], device=DEVICE)
        expected = torch.einsum("ijk,k->ij", _DENSE_A_3D, B)
        got = sdot_ddc(A, B)
        torch.testing.assert_close(got, expected)

    def test_ccd_sdot(self) -> None:
        A = _build_ccd()
        B = torch.tensor([1.0, 2.0, 3.0], device=DEVICE)
        expected = torch.einsum("ijk,k->ij", _DENSE_A_3D, B)
        got = sdot_ccd(A, B)
        torch.testing.assert_close(got, expected)


if __name__ == "__main__":
    unittest.main()
