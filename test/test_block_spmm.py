from __future__ import annotations

import unittest

import torch

import helion
from helion._testing import DEVICE
from helion._testing import TestCase
import helion.language as hl

# ----------------------------------------------------------------------------
# Block-sparse SPMM tests.
#
# ``A`` is a SparseTensor whose per-leaf payload is a fixed-shape dense block
# rather than a scalar. The coord tree (``A.shape``) enumerates *blocks*;
# the payload shape is ``A.value_shape`` (inferred from ``values.shape[1:]``).
# ``.value`` on the leaf tile returns ``(*leaf_coord_shape, *A.value_shape)``.
#
# Cases exercised:
#   * 2D block payload ``(BX, BY)``: A is block-sparse in both M and K.
#     Logical A shape = ``(M_BLK * BX, K_BLK * BY)``.
#   * 1D block payload ``(BX,)``: A is block-sparse in M only; K is unblocked.
#     Logical A shape = ``(M_BLK * BX, K)``.
#
# Each payload shape is run under Compressed-root and Dense-root layouts with
# a Compressed inner, giving 4 (kernel, format) combos.  B is a plain dense
# matrix; all contractions go through ``hl.dot`` (2D block packs (P_k, BY)
# into the reduction axis, 1D block contracts directly over P_k).
# ----------------------------------------------------------------------------


# Non-zero block coords (m_blk, k_idx) re-used under every layout.  Irregular
# per-row nnz plus one empty row (row 3) stresses jagged-tile boundaries for
# Compressed inner and empty-row elision for Compressed root.
_NNZ_COORDS: list[tuple[int, int]] = [
    (0, 0),
    (0, 2),
    (1, 1),
    (1, 3),
    (1, 4),
    (2, 0),
    (2, 2),
    # row 3: empty
]


def _build_block_sparse(
    fmt0: str,
    level_shape: tuple[int, int],
    value_shape: tuple[int, ...],
    nnz_coords: list[tuple[int, int]],
    block_values: torch.Tensor,
) -> hl.SparseTensor:
    """Encode ``block_values`` under ``(fmt0, 'Compressed')`` levels.

    ``block_values`` has shape ``(len(nnz_coords), *value_shape)``; entry
    ``i`` is the block stored at coord ``nnz_coords[i]``.
    """
    M_L0 = level_shape[0]
    by_row: list[list[tuple[int, int]]] = [[] for _ in range(M_L0)]
    for i, (m, k) in enumerate(nnz_coords):
        by_row[m].append((k, i))

    if fmt0 == "Compressed":
        row_slots = [r for r in range(M_L0) if by_row[r]]
        coords0: torch.Tensor | None = torch.tensor(
            row_slots, dtype=torch.int64, device=DEVICE
        )
        ptrs0: torch.Tensor | None = torch.tensor(
            [0, len(row_slots)], dtype=torch.int64, device=DEVICE
        )
    elif fmt0 == "Dense":
        row_slots = list(range(M_L0))
        coords0 = None
        ptrs0 = None
    else:
        raise AssertionError(fmt0)

    ptrs_list: list[int] = [0]
    coord_list: list[int] = []
    reordered: list[torch.Tensor] = []
    for r in row_slots:
        for k, i in by_row[r]:
            coord_list.append(k)
            reordered.append(block_values[i])
        ptrs_list.append(len(coord_list))
    ptrs1 = torch.tensor(ptrs_list, dtype=torch.int64, device=DEVICE)
    coords1 = torch.tensor(coord_list, dtype=torch.int64, device=DEVICE)
    if reordered:
        values = torch.stack(reordered, dim=0)
    else:
        values = torch.zeros(
            (0, *value_shape), dtype=block_values.dtype, device=DEVICE
        )

    return hl.SparseTensor(
        values=values,
        shape=tuple(level_shape),
        ptrs=(ptrs0, ptrs1),
        coords=(coords0, coords1),
        bitmaps=(None, None),
    )


def _dense_reference_2d(
    M_BLK: int,
    K_BLK: int,
    BX: int,
    BY: int,
    nnz: list[tuple[int, int]],
    block_values: torch.Tensor,
) -> torch.Tensor:
    """Reconstruct the logical ``(M_BLK*BX, K_BLK*BY)`` dense matrix."""
    A = torch.zeros(
        M_BLK * BX, K_BLK * BY, dtype=block_values.dtype, device=block_values.device
    )
    for i, (m, k) in enumerate(nnz):
        A[m * BX : (m + 1) * BX, k * BY : (k + 1) * BY] = block_values[i]
    return A


def _dense_reference_1d(
    M_BLK: int,
    K: int,
    BX: int,
    nnz: list[tuple[int, int]],
    block_values: torch.Tensor,
) -> torch.Tensor:
    """Reconstruct the logical ``(M_BLK*BX, K)`` dense matrix."""
    A = torch.zeros(
        M_BLK * BX, K, dtype=block_values.dtype, device=block_values.device
    )
    for i, (m, k) in enumerate(nnz):
        A[m * BX : (m + 1) * BX, k] = block_values[i]
    return A


# ----------------------------------------------------------------------------
# Kernels.
#
# Both kernels treat ``tile_m`` as (P_m,) block-row coords (value is the
# block-row index under Compressed root, the tile-index 0..M_BLK-1 under
# Dense root — either way ``tile_m * BX + arange(BX)`` produces the
# logical row ids) and ``tile_k`` as (P_m, P_k) block-col coords inherited
# from the parent Compressed inner.
# ----------------------------------------------------------------------------


@helion.kernel(config=helion.Config(block_sizes=[4, 2, 4]))
def block_spmm_2d(
    A: hl.SparseTensor,
    B: torch.Tensor,
    BX: hl.constexpr,
    BY: hl.constexpr,
    fmt0: hl.constexpr,
) -> torch.Tensor:
    """SPMM with 2D ``(BX, BY)`` block payload.

    Inner contraction packs the (P_k, BY) axes into a single reduction dim
    and drops them through a 3D batched ``hl.dot``:
      ``a_blk  : (P_m, P_k, BX, BY) -> (P_m, BX, P_k*BY)``
      ``b_val  : (P_m, P_k, BY, P_n) -> (P_m, P_k*BY, P_n)``
      ``acc    : (P_m, BX, P_n)``
    """
    M_BLK = A.shape[0]
    N = B.size(1)
    C = torch.zeros(M_BLK * BX, N, dtype=B.dtype, device=B.device)
    for tile_n in hl.tile(N):
        for tile_m in hl.sparse_tile(A, dim=0, levelformat=fmt0):
            acc = hl.zeros([tile_m.size(0), BX, tile_n], dtype=B.dtype)
            for tile_k in hl.sparse_tile(tile_m, dim=1, levelformat="Compressed"):
                a_blk = tile_k.value  # (P_m, P_k, BX, BY)
                col_off = (
                    tile_k[:, :, None] * BY + hl.arange(0, BY)[None, None, :]
                )  # (P_m, P_k, BY)
                b_val = B[
                    col_off[:, :, :, None], tile_n.index[None, None, None, :]
                ]  # (P_m, P_k, BY, P_n)
                a_flat = a_blk.permute(0, 2, 1, 3).reshape(
                    [tile_m.size(0), BX, tile_k.size(1) * BY]
                )
                b_flat = b_val.reshape(
                    [tile_m.size(0), tile_k.size(1) * BY, tile_n]
                )
                acc = hl.dot(a_flat, b_flat, acc=acc)
            row_idx = (
                tile_m[:, None] * BX + hl.arange(0, BX)[None, :]
            )  # (P_m, BX)
            C[row_idx[:, :, None], tile_n.index[None, None, :]] = acc
    return C


@helion.kernel(config=helion.Config(block_sizes=[4, 2, 4]))
def block_spmm_1d(
    A: hl.SparseTensor,
    B: torch.Tensor,
    BX: hl.constexpr,
    fmt0: hl.constexpr,
) -> torch.Tensor:
    """SPMM with 1D ``(BX,)`` block payload — block only in M; K is unblocked.

    Inner contraction is a straight batched ``hl.dot`` over P_k:
      ``a_blk  : (P_m, P_k, BX) -> (P_m, BX, P_k)`` (permute)
      ``b_val  : (P_m, P_k, P_n)``
      ``acc    : (P_m, BX, P_n)``
    """
    M_BLK = A.shape[0]
    N = B.size(1)
    C = torch.zeros(M_BLK * BX, N, dtype=B.dtype, device=B.device)
    for tile_n in hl.tile(N):
        for tile_m in hl.sparse_tile(A, dim=0, levelformat=fmt0):
            acc = hl.zeros([tile_m.size(0), BX, tile_n], dtype=B.dtype)
            for tile_k in hl.sparse_tile(tile_m, dim=1, levelformat="Compressed"):
                a_blk = tile_k.value  # (P_m, P_k, BX)
                b_val = B[
                    tile_k[:, :, None], tile_n.index[None, None, :]
                ]  # (P_m, P_k, P_n)
                a_perm = a_blk.permute(0, 2, 1)  # (P_m, BX, P_k)
                acc = hl.dot(a_perm, b_val, acc=acc)
            row_idx = tile_m[:, None] * BX + hl.arange(0, BX)[None, :]
            C[row_idx[:, :, None], tile_n.index[None, None, :]] = acc
    return C


# ----------------------------------------------------------------------------
# Tests.
# ----------------------------------------------------------------------------


class TestBlockSpMM(TestCase):
    def setUp(self) -> None:
        super().setUp()
        torch.manual_seed(0)

    def test_value_shape_and_nnz_properties(self) -> None:
        """SparseTensor exposes ``value_shape`` / ``nnz`` without the user
        ever touching the raw ``values`` storage."""
        blocks = torch.randn(7, 4, 3, device=DEVICE)
        A = hl.SparseTensor(
            values=blocks,
            shape=(3, 5),
            ptrs=(None, None),
            coords=(None, None),
            bitmaps=(None, None),
        )
        self.assertEqual(A.value_shape, (4, 3))
        self.assertEqual(A.nnz, 7)

        # Scalar payload → empty value_shape.
        A_scalar = hl.SparseTensor(
            values=torch.randn(12, device=DEVICE),
            shape=(3, 4),
            ptrs=(None, None),
            coords=(None, None),
            bitmaps=(None, None),
        )
        self.assertEqual(A_scalar.value_shape, ())
        self.assertEqual(A_scalar.nnz, 12)

    def _run_2d(self, fmt0: str) -> None:
        M_BLK, K_BLK, BX, BY, N = 4, 5, 4, 4, 8
        block_values = torch.randn(
            len(_NNZ_COORDS), BX, BY, dtype=torch.float32, device=DEVICE
        )
        A = _build_block_sparse(
            fmt0,
            level_shape=(M_BLK, K_BLK),
            value_shape=(BX, BY),
            nnz_coords=_NNZ_COORDS,
            block_values=block_values,
        )
        self.assertEqual(A.value_shape, (BX, BY))
        self.assertEqual(A.nnz, len(_NNZ_COORDS))

        B = torch.randn(K_BLK * BY, N, dtype=torch.float32, device=DEVICE)
        A_dense = _dense_reference_2d(M_BLK, K_BLK, BX, BY, _NNZ_COORDS, block_values)
        expected = A_dense @ B

        C = block_spmm_2d(A, B, BX, BY, fmt0)
        # TF32 on modern GPUs → loosen fp32 matmul tolerance.
        torch.testing.assert_close(C, expected, atol=1e-2, rtol=1e-2)

    def _run_1d(self, fmt0: str) -> None:
        M_BLK, K, BX, N = 4, 5, 4, 8
        block_values = torch.randn(
            len(_NNZ_COORDS), BX, dtype=torch.float32, device=DEVICE
        )
        A = _build_block_sparse(
            fmt0,
            level_shape=(M_BLK, K),
            value_shape=(BX,),
            nnz_coords=_NNZ_COORDS,
            block_values=block_values,
        )
        self.assertEqual(A.value_shape, (BX,))
        self.assertEqual(A.nnz, len(_NNZ_COORDS))

        B = torch.randn(K, N, dtype=torch.float32, device=DEVICE)
        A_dense = _dense_reference_1d(M_BLK, K, BX, _NNZ_COORDS, block_values)
        expected = A_dense @ B

        C = block_spmm_1d(A, B, BX, fmt0)
        # TF32 on modern GPUs → loosen fp32 matmul tolerance.
        torch.testing.assert_close(C, expected, atol=1e-2, rtol=1e-2)

    def test_2d_block_compressed_root(self) -> None:
        self._run_2d("Compressed")

    def test_2d_block_dense_root(self) -> None:
        self._run_2d("Dense")

    def test_1d_block_compressed_root(self) -> None:
        self._run_1d("Compressed")

    def test_1d_block_dense_root(self) -> None:
        self._run_1d("Dense")


if __name__ == "__main__":
    unittest.main()
