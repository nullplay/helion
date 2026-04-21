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
    # Padding convention: coord = -1 sentinel → masked out via Padded
    # augment, so ``values`` at pad slots is never touched and may hold
    # any garbage (sentinel 777.0 here catches broken masks).
    # row 0: non-zeros at cols 0, 2 → coord [0, 2],  value [1, 2]
    # row 1: non-zero  at col  1    → coord [1, -1], value [3, 777]
    coords1 = torch.tensor([[0, 2], [1, -1]], dtype=torch.int64, device=DEVICE)
    values = torch.tensor([1.0, 2.0, 3.0, 777.0], device=DEVICE)
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
    # Padding convention: coord = -1 sentinel masked out via Padded augment;
    # pad-slot value is garbage (777.0 sentinel catches broken masks).
    ptrs0 = torch.tensor([0, 2], dtype=torch.int64, device=DEVICE)
    coords0 = torch.tensor([0, 1], dtype=torch.int64, device=DEVICE)
    coords1 = torch.tensor([[0, 2], [1, -1]], dtype=torch.int64, device=DEVICE)
    values = torch.tensor([1.0, 2.0, 3.0, 777.0], device=DEVICE)
    return hl.SparseTensor(
        values=values,
        shape=_SHAPE,
        ptrs=(ptrs0, None),
        coords=(coords0, coords1),
    )


def _build_dj() -> hl.SparseTensor:
    # DJ (Dense-Jagged): outer Dense enumerates every row; inner Jagged uses
    # ptrs1 for variable prefix length per row.  No coord tensor — coord ==
    # tile index within the row's prefix, so stored values include explicit
    # zeros wherever the matrix has a zero before its last non-zero in that
    # row.  Row 0 stores cols [0,1,2] = [1,0,2]; row 1 stores cols [0,1] = [0,3].
    ptrs1 = torch.tensor([0, 3, 5], dtype=torch.int64, device=DEVICE)
    values = torch.tensor([1.0, 0.0, 2.0, 0.0, 3.0], device=DEVICE)
    return hl.SparseTensor(
        values=values,
        shape=_SHAPE,
        ptrs=(None, ptrs1),
        coords=(None, None),
    )


def _build_cj() -> hl.SparseTensor:
    # CJ (Compressed-Jagged): outer Compressed picks non-empty rows; inner
    # Jagged uses ptrs1 on the compressed-row ordering.
    ptrs0 = torch.tensor([0, 2], dtype=torch.int64, device=DEVICE)
    coords0 = torch.tensor([0, 1], dtype=torch.int64, device=DEVICE)
    ptrs1 = torch.tensor([0, 3, 5], dtype=torch.int64, device=DEVICE)
    values = torch.tensor([1.0, 0.0, 2.0, 0.0, 3.0], device=DEVICE)
    return hl.SparseTensor(
        values=values,
        shape=_SHAPE,
        ptrs=(ptrs0, ptrs1),
        coords=(coords0, None),
    )


# --- Bitmap fixtures ----------------------------------------------------------
# DB is over the (2, 3) logical A; BD / BB switch to a 3x3 fixture with an
# empty middle row so the outer Bitmap has something real to mask out.
#   A_BD = [[1, 0, 2], [0, 0, 0], [0, 3, 0]]
# The masked-out slots are filled with a sentinel (_GARBAGE) in ``values`` so
# a broken mask would leak the sentinel into the result and fail the check.
_GARBAGE = 777.0
_SHAPE_BD = (3, 3)
_DENSE_A_BD = torch.tensor(
    [[1.0, 0.0, 2.0], [0.0, 0.0, 0.0], [0.0, 3.0, 0.0]], device=DEVICE
)


def _build_db() -> hl.SparseTensor:
    # DB (Dense-Bitmap): dense outer row, inner Bitmap masks the (2, 3)
    # sparsity pattern of _DENSE_A. Zeros in A become sentinels in values.
    values = torch.tensor(
        [1.0, _GARBAGE, 2.0, _GARBAGE, 3.0, _GARBAGE], device=DEVICE
    )
    bitmap1 = torch.tensor(
        [[True, False, True], [False, True, False]], device=DEVICE
    )
    return hl.SparseTensor(
        values=values,
        shape=_SHAPE,
        ptrs=(None, None),
        coords=(None, None),
        bitmaps=(None, bitmap1),
    )


def _build_bd() -> hl.SparseTensor:
    # BD (Bitmap-Dense): outer Bitmap masks row 1 (all zeros) of the 3x3
    # fixture, inner Dense stores all K=3 slots per outer row.  The
    # masked-out row holds sentinels end-to-end.
    values = torch.tensor(
        [
            1.0, 0.0, 2.0,
            _GARBAGE, _GARBAGE, _GARBAGE,
            0.0, 3.0, 0.0,
        ],
        device=DEVICE,
    )
    bitmap0 = torch.tensor([True, False, True], device=DEVICE)
    return hl.SparseTensor(
        values=values,
        shape=_SHAPE_BD,
        ptrs=(None, None),
        coords=(None, None),
        bitmaps=(bitmap0, None),
    )


def _build_bb() -> hl.SparseTensor:
    # BB (Bitmap-Bitmap) over the same 3x3 fixture.  Outer Bitmap masks the
    # empty middle row; inner Bitmap additionally masks the exact sparsity
    # pattern of the two non-empty rows.  Every False slot in values is a
    # sentinel.
    values = torch.tensor(
        [
            1.0, _GARBAGE, 2.0,
            _GARBAGE, _GARBAGE, _GARBAGE,
            _GARBAGE, 3.0, _GARBAGE,
        ],
        device=DEVICE,
    )
    bitmap0 = torch.tensor([True, False, True], device=DEVICE)
    bitmap1 = torch.tensor(
        [
            [True, False, True],
            [False, False, False],
            [False, True, False],
        ],
        device=DEVICE,
    )
    return hl.SparseTensor(
        values=values,
        shape=_SHAPE_BD,
        ptrs=(None, None),
        coords=(None, None),
        bitmaps=(bitmap0, bitmap1),
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


_DENSE_A = torch.tensor([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]], device=DEVICE)


_SPMV_LAYOUTS = {
    ("Dense", "Dense"): (_build_dd, _DENSE_A),
    ("Dense", "Compressed"): (_build_dc, _DENSE_A),
    ("Compressed", "Dense"): (_build_cd, _DENSE_A),
    ("Compressed", "Compressed"): (_build_cc, _DENSE_A),
    ("Dense", "Padded"): (_build_dp, _DENSE_A),
    ("Compressed", "Padded"): (_build_cp, _DENSE_A),
    ("Dense", "Jagged"): (_build_dj, _DENSE_A),
    ("Compressed", "Jagged"): (_build_cj, _DENSE_A),
    ("Dense", "Bitmap"): (_build_db, _DENSE_A),
    ("Bitmap", "Dense"): (_build_bd, _DENSE_A_BD),
    ("Bitmap", "Bitmap"): (_build_bb, _DENSE_A_BD),
}
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
        for fmt, (builder, dense_a) in _SPMV_LAYOUTS.items():
            with self.subTest(fmt=fmt):
                got = spmv(builder(), x, *fmt)
                expected = dense_a @ x
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
    # so a single flat ``self_position`` can load it.  Padding convention:
    # coord = -1 sentinel masked out via Padded augment; pad-slot value is
    # garbage (777.0 catches broken masks).
    #   (i=0,j=0) nz at k=0,2 → [0, 2],  vals [1, 2]
    #   (i=0,j=1) nz at k=1   → [1, -1], vals [3, 777]
    #   (i=1,j=0) nz at k=2   → [2, -1], vals [5, 777]
    #   (i=1,j=1) nz at k=0,1 → [0, 1],  vals [4, 6]
    coords2 = torch.tensor(
        [[[0, 2], [1, -1]], [[2, -1], [0, 1]]], dtype=torch.int64, device=DEVICE
    )
    values = torch.tensor(
        [1.0, 2.0, 3.0, 777.0, 5.0, 777.0, 4.0, 6.0], device=DEVICE
    )
    return hl.SparseTensor(
        values=values,
        shape=_SHAPE_3D,
        ptrs=(None, None, None),
        coords=(None, None, coords2),
    )


def _build_djj() -> hl.SparseTensor:
    # DJJ layout: Dense-Jagged-Jagged.  Every i has J=2 j-entries
    # (ptrs1 = [0, 2, 4]); each (i, j) stores a k-prefix up to its last
    # non-zero with explicit zeros for gaps.
    #   (i=0,j=0) last nz at k=2 → prefix len 3, vals [1, 0, 2]
    #   (i=0,j=1) last nz at k=1 → prefix len 2, vals [0, 3]
    #   (i=1,j=0) last nz at k=2 → prefix len 3, vals [0, 0, 5]
    #   (i=1,j=1) last nz at k=1 → prefix len 2, vals [4, 6]
    ptrs1 = torch.tensor([0, 2, 4], dtype=torch.int64, device=DEVICE)
    ptrs2 = torch.tensor([0, 3, 5, 8, 10], dtype=torch.int64, device=DEVICE)
    values = torch.tensor(
        [1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 0.0, 5.0, 4.0, 6.0], device=DEVICE
    )
    return hl.SparseTensor(
        values=values,
        shape=_SHAPE_3D,
        ptrs=(None, ptrs1, ptrs2),
        coords=(None, None, None),
    )


def _build_dpj() -> hl.SparseTensor:
    # DPJ layout: Dense-Padded-Jagged.  Outer Dense over i, middle Padded
    # over j with pad_size=2 (every (i) row stores both j=0,1), inner
    # Jagged over k with the same ragged prefix as DJJ.  ptrs2 is indexed
    # by the Padded level's flat position i * pad_size + pad_j, which
    # enumerates (0,0), (0,1), (1,0), (1,1) → ptrs2 = [0, 3, 5, 8, 10].
    coords1 = torch.tensor([[0, 1], [0, 1]], dtype=torch.int64, device=DEVICE)
    ptrs2 = torch.tensor([0, 3, 5, 8, 10], dtype=torch.int64, device=DEVICE)
    values = torch.tensor(
        [1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 0.0, 5.0, 4.0, 6.0], device=DEVICE
    )
    return hl.SparseTensor(
        values=values,
        shape=_SHAPE_3D,
        ptrs=(None, None, ptrs2),
        coords=(None, coords1, None),
    )


# --- 3D Bitmap fixtures -------------------------------------------------------
# Shared (3, 2, 3) base with an all-zero i=1 plane so outer Bitmap has
# something to mask.  Masked-out slots hold a sentinel to catch broken masks.
#   A_3D_B[0] = [[1, 0, 2], [0, 3, 0]]
#   A_3D_B[1] = [[0, 0, 0], [0, 0, 0]]   <- masked by Bitmap at level 0
#   A_3D_B[2] = [[0, 0, 5], [4, 6, 0]]
_SHAPE_3D_B = (3, 2, 3)
_DENSE_A_3D_B = torch.tensor(
    [
        [[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]],
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        [[0.0, 0.0, 5.0], [4.0, 6.0, 0.0]],
    ],
    device=DEVICE,
)


def _build_ddb() -> hl.SparseTensor:
    # DDB: inner Bitmap masks the full sparsity pattern at level 2.  Values
    # are fully dense I*J*K=18; sentinels sit wherever bitmap[2] is False.
    G = _GARBAGE
    values = torch.tensor(
        [
            1.0, G, 2.0,   G, 3.0, G,
            G,   G, G,     G, G,   G,
            G,   G, 5.0,   4.0, 6.0, G,
        ],
        device=DEVICE,
    )
    bitmap2 = torch.tensor(
        [
            [[True, False, True], [False, True, False]],
            [[False, False, False], [False, False, False]],
            [[False, False, True], [True, True, False]],
        ],
        device=DEVICE,
    )
    return hl.SparseTensor(
        values=values,
        shape=_SHAPE_3D_B,
        ptrs=(None, None, None),
        coords=(None, None, None),
        bitmaps=(None, None, bitmap2),
    )


def _build_dbd() -> hl.SparseTensor:
    # DBD: middle Bitmap masks whole (i, j) rows of the empty i=1 plane.
    # Values are dense I*J*K=18; the two masked (i=1, j=*) rows are sentinels.
    G = _GARBAGE
    values = torch.tensor(
        [
            1.0, 0.0, 2.0,   0.0, 3.0, 0.0,
            G,   G,   G,     G,   G,   G,
            0.0, 0.0, 5.0,   4.0, 6.0, 0.0,
        ],
        device=DEVICE,
    )
    bitmap1 = torch.tensor(
        [[True, True], [False, False], [True, True]], device=DEVICE
    )
    return hl.SparseTensor(
        values=values,
        shape=_SHAPE_3D_B,
        ptrs=(None, None, None),
        coords=(None, None, None),
        bitmaps=(None, bitmap1, None),
    )


def _build_bdd() -> hl.SparseTensor:
    # BDD: outer Bitmap masks the empty i=1 plane.  Values are dense
    # I*J*K=18; the whole i=1 plane is sentinels.
    G = _GARBAGE
    values = torch.tensor(
        [
            1.0, 0.0, 2.0,   0.0, 3.0, 0.0,
            G,   G,   G,     G,   G,   G,
            0.0, 0.0, 5.0,   4.0, 6.0, 0.0,
        ],
        device=DEVICE,
    )
    bitmap0 = torch.tensor([True, False, True], device=DEVICE)
    return hl.SparseTensor(
        values=values,
        shape=_SHAPE_3D_B,
        ptrs=(None, None, None),
        coords=(None, None, None),
        bitmaps=(bitmap0, None, None),
    )


def _build_bbb() -> hl.SparseTensor:
    # BBB: Bitmap at every level.  Outer masks i=1, middle stays all True
    # over the live planes, inner matches the exact sparsity.  Sentinels fill
    # both the masked i=1 plane and the False slots in non-empty planes.
    G = _GARBAGE
    values = torch.tensor(
        [
            1.0, G, 2.0,   G, 3.0, G,
            G,   G, G,     G, G,   G,
            G,   G, 5.0,   4.0, 6.0, G,
        ],
        device=DEVICE,
    )
    bitmap0 = torch.tensor([True, False, True], device=DEVICE)
    bitmap1 = torch.tensor(
        [[True, True], [True, True], [True, True]], device=DEVICE
    )
    bitmap2 = torch.tensor(
        [
            [[True, False, True], [False, True, False]],
            [[False, False, False], [False, False, False]],
            [[False, False, True], [True, True, False]],
        ],
        device=DEVICE,
    )
    return hl.SparseTensor(
        values=values,
        shape=_SHAPE_3D_B,
        ptrs=(None, None, None),
        coords=(None, None, None),
        bitmaps=(bitmap0, bitmap1, bitmap2),
    )


_SDOT_LAYOUTS = {
    ("Dense", "Compressed", "Dense"): (_build_dcd, _DENSE_A_3D),
    ("Dense", "Dense", "Compressed"): (_build_ddc, _DENSE_A_3D),
    ("Dense", "Compressed", "Compressed"): (_build_dcc, _DENSE_A_3D),
    ("Compressed", "Compressed", "Dense"): (_build_ccd, _DENSE_A_3D),
    ("Dense", "Padded", "Dense"): (_build_dpd, _DENSE_A_3D),
    ("Dense", "Dense", "Padded"): (_build_ddp, _DENSE_A_3D),
    ("Dense", "Jagged", "Jagged"): (_build_djj, _DENSE_A_3D),
    ("Dense", "Padded", "Jagged"): (_build_dpj, _DENSE_A_3D),
    ("Dense", "Dense", "Bitmap"): (_build_ddb, _DENSE_A_3D_B),
    ("Dense", "Bitmap", "Dense"): (_build_dbd, _DENSE_A_3D_B),
    ("Bitmap", "Dense", "Dense"): (_build_bdd, _DENSE_A_3D_B),
    ("Bitmap", "Bitmap", "Bitmap"): (_build_bbb, _DENSE_A_3D_B),
}


class TestSparseTile3D(TestCase):
    def test_sdot_layouts(self) -> None:
        B = torch.tensor([1.0, 2.0, 3.0], device=DEVICE)
        for fmt, (builder, dense_a) in _SDOT_LAYOUTS.items():
            with self.subTest(fmt=fmt):
                got = sdot(builder(), B, *fmt)
                expected = torch.einsum("ijk,k->ij", dense_a, B)
                torch.testing.assert_close(got, expected)


if __name__ == "__main__":
    unittest.main()
