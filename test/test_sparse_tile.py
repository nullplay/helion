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


def _build_dp() -> hl.SparseTensor:
    # DP (Dense-Padded, ELL-style): pad_size = max row-nnz = 2.
    # Padding convention: coord = 0, value = 0 → multiplied slot contributes 0.
    # row 0: non-zeros at cols 0, 2 → coord [0, 2], value [1, 2]
    # row 1: non-zero at col 1  → coord [1, 0 (pad)], value [3, 0 (pad)]
    coords1 = torch.tensor([[0, 2], [1, 0]], dtype=torch.int64, device=DEVICE)
    values = torch.tensor([1.0, 2.0, 3.0, 0.0], device=DEVICE)
    return hl.SparseTensor(
        values=values,
        shape=_SHAPE,
        ptrs=(None, None),
        coords=(None, coords1),
    )


def _build_cp() -> hl.SparseTensor:
    # CP (Compressed-Padded): outer Compressed picks the non-empty rows, inner
    # Padded stores pad_size=2 (coord, value) pairs per stored row.  Both rows
    # are non-empty here, so coords1 shape is (nnz_rows=2, pad_size=2).
    ptrs0 = torch.tensor([0, 2], dtype=torch.int64, device=DEVICE)
    coords0 = torch.tensor([0, 1], dtype=torch.int64, device=DEVICE)
    coords1 = torch.tensor([[0, 2], [1, 0]], dtype=torch.int64, device=DEVICE)
    values = torch.tensor([1.0, 2.0, 3.0, 0.0], device=DEVICE)
    return hl.SparseTensor(
        values=values,
        shape=_SHAPE,
        ptrs=(ptrs0, None),
        coords=(coords0, coords1),
    )


@helion.kernel(config=helion.Config(block_sizes=[2, 32]))
def spmv(
    A: hl.SparseTensor,
    x: torch.Tensor,
    fmt0: hl.constexpr,
    fmt1: hl.constexpr,
) -> torch.Tensor:
    out = torch.zeros(A.shape[0], dtype=x.dtype, device=x.device)
    for tile_m in hl.sparse_tile(A, dim=0, levelformat=fmt0):
        acc = hl.zeros([tile_m.size(0)], dtype=x.dtype)
        for tile_k in hl.sparse_tile(tile_m, dim=1, levelformat=fmt1):
            x_val = x[tile_k]
            a_val = tile_k.value
            acc = acc + (a_val * x_val).sum(dim=-1)
        out[tile_m] = acc
    return out


_SPMV_LAYOUTS = {
    ("Dense", "Dense"): _build_dd,
    ("Dense", "Compressed"): _build_dc,
    ("Compressed", "Dense"): _build_cd,
    ("Compressed", "Compressed"): _build_cc,
    ("Dense", "Padded"): _build_dp,
    ("Compressed", "Padded"): _build_cp,
}


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
    def test_spmv_layouts(self) -> None:
        x = torch.tensor([1.0, 2.0, 3.0], device=DEVICE)
        expected = _DENSE_A @ x
        for fmt, builder in _SPMV_LAYOUTS.items():
            with self.subTest(fmt=fmt):
                got = spmv(builder(), x, *fmt)
                torch.testing.assert_close(got, expected)

    def test_spmm_cc(self) -> None:
        A = _build_cc()
        expected = _DENSE_A @ _B
        got = spmm_cc(A, _B)
        torch.testing.assert_close(got, expected)


# --- 3D logical fixture shared by DCD / DDC / DCC / CCD layouts ----------------
# Logical A[2, 2, 3]:
#   A[0,0,:] = [1, 0, 2]
#   A[0,1,:] = [0, 3, 0]
#   A[1,0,:] = [0, 0, 5]
#   A[1,1,:] = [4, 6, 0]
# Per-(i,j) nnz in row-major (i,j) order: 2, 1, 1, 2 → total nnz = 6.
_SHAPE_3D = (2, 2, 3)


def _build_dcd() -> hl.SparseTensor:
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


def _build_ddc() -> hl.SparseTensor:
    # DDC layout: Dense at levels 0/1, Compressed at level 2 (per row-major (i,j)).
    # ptrs2 indexes the I*J rows in row-major order; lengths under each (i,j) are
    # data-dependent → exercises a 2-D parent jagged_tile.
    ptrs2 = torch.tensor([0, 2, 3, 4, 6], dtype=torch.int64, device=DEVICE)
    coords2 = torch.tensor([0, 2, 1, 2, 0, 1], dtype=torch.int64, device=DEVICE)
    values = torch.tensor([1.0, 2.0, 3.0, 5.0, 4.0, 6.0], device=DEVICE)
    return hl.SparseTensor(
        values=values,
        shape=_SHAPE_3D,
        ptrs=(None, None, ptrs2),
        coords=(None, None, coords2),
    )


def _build_dcc() -> hl.SparseTensor:
    # DCC layout: Dense at level 0, Compressed at levels 1 and 2.
    # All (i,j) rows happen to be non-empty so coords1 enumerates [0,1,0,1].
    ptrs1 = torch.tensor([0, 2, 4], dtype=torch.int64, device=DEVICE)
    coords1 = torch.tensor([0, 1, 0, 1], dtype=torch.int64, device=DEVICE)
    ptrs2 = torch.tensor([0, 2, 3, 4, 6], dtype=torch.int64, device=DEVICE)
    coords2 = torch.tensor([0, 2, 1, 2, 0, 1], dtype=torch.int64, device=DEVICE)
    values = torch.tensor([1.0, 2.0, 3.0, 5.0, 4.0, 6.0], device=DEVICE)
    return hl.SparseTensor(
        values=values,
        shape=_SHAPE_3D,
        ptrs=(None, ptrs1, ptrs2),
        coords=(None, coords1, coords2),
    )


_DENSE_A_3D = torch.tensor(
    [
        [[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]],
        [[0.0, 0.0, 5.0], [4.0, 6.0, 0.0]],
    ],
    device=DEVICE,
)


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
def sdot(
    A: hl.SparseTensor,
    B: torch.Tensor,
    fmt0: hl.constexpr,
    fmt1: hl.constexpr,
    fmt2: hl.constexpr,
) -> torch.Tensor:
    I = A.shape[0]
    J = A.shape[1]
    C = torch.zeros(I * J, dtype=B.dtype, device=B.device)
    for tile_i in hl.sparse_tile(A, dim=0, levelformat=fmt0):
        for tile_j in hl.sparse_tile(tile_i, dim=1, levelformat=fmt1):
            acc = hl.zeros([tile_i.size(0), tile_j.size(1)], dtype=B.dtype)
            for tile_k in hl.sparse_tile(tile_j, dim=2, levelformat=fmt2):
                a_val = tile_k.value
                b_val = B[tile_k]
                acc = acc + (a_val * b_val).sum(dim=-1)
            flat_idx = tile_i[:, None] * J + tile_j
            C[flat_idx] = acc
    return C.view(I, J)


def _build_dpd() -> hl.SparseTensor:
    # DPD layout: Dense-Padded-Dense. Every (i, j) is non-empty in the 3D
    # fixture (J=2, pad_size=2), so coords1 just enumerates [0, 1] per i.
    # Values are flat row-major of A[i, padded[i, :], k].
    coords1 = torch.tensor([[0, 1], [0, 1]], dtype=torch.int64, device=DEVICE)
    values = torch.tensor(
        [1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 0.0, 0.0, 5.0, 4.0, 6.0, 0.0],
        device=DEVICE,
    )
    return hl.SparseTensor(
        values=values,
        shape=_SHAPE_3D,
        ptrs=(None, None, None),
        coords=(None, coords1, None),
    )


def _build_ddp() -> hl.SparseTensor:
    # DDP layout: Dense-Dense-Padded.  pad_size = max over-(i,j) of non-zeros
    # per fixture row = 2.  coord is the natural 3-D shape ``(I, J, pad_size)
    # == (2, 2, 2)``; the lowering flattens it internally via FlattenOrigin
    # so a single flat ``self_position`` can load it.  Pad slots use coord=0,
    # value=0 so masked multiplications contribute zero.
    #   (i=0,j=0) nz at k=0,2 → [0, 2], vals [1, 2]
    #   (i=0,j=1) nz at k=1   → [1, 0 (pad)], vals [3, 0]
    #   (i=1,j=0) nz at k=2   → [2, 0 (pad)], vals [5, 0]
    #   (i=1,j=1) nz at k=0,1 → [0, 1], vals [4, 6]
    coords2 = torch.tensor(
        [[[0, 2], [1, 0]], [[2, 0], [0, 1]]], dtype=torch.int64, device=DEVICE
    )
    values = torch.tensor(
        [1.0, 2.0, 3.0, 0.0, 5.0, 0.0, 4.0, 6.0], device=DEVICE
    )
    return hl.SparseTensor(
        values=values,
        shape=_SHAPE_3D,
        ptrs=(None, None, None),
        coords=(None, None, coords2),
    )


_SDOT_LAYOUTS = {
    ("Dense", "Compressed", "Dense"): _build_dcd,
    ("Dense", "Dense", "Compressed"): _build_ddc,
    ("Dense", "Compressed", "Compressed"): _build_dcc,
    ("Compressed", "Compressed", "Dense"): _build_ccd,
    ("Dense", "Padded", "Dense"): _build_dpd,
    ("Dense", "Dense", "Padded"): _build_ddp,
}


class TestSparseTile3D(TestCase):
    def test_sdot_layouts(self) -> None:
        B = torch.tensor([1.0, 2.0, 3.0], device=DEVICE)
        expected = torch.einsum("ijk,k->ij", _DENSE_A_3D, B)
        for fmt, builder in _SDOT_LAYOUTS.items():
            with self.subTest(fmt=fmt):
                got = sdot(builder(), B, *fmt)
                torch.testing.assert_close(got, expected)


if __name__ == "__main__":
    unittest.main()
