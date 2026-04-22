from __future__ import annotations

import unittest

import torch

import helion
from helion._testing import DEVICE
from helion._testing import TestCase
import helion.language as hl

# ----------------------------------------------------------------------------
# 3-D sparse-tile SDOT tests.
#
# SDOT computes ``C[i, j] = sum_k A[i, j, k] * x[k]`` — a per-(i, j) sparse
# dot product against a dense 1-D vector ``x``.  One rich fixture with
# irregular k-nnz per (i, j) plus one empty (i, j) plane is re-encoded into
# 30 format combos spanning all positions.
#
# A single kernel covers every combo.  The inner body is a pure element-wise
# multiply-then-sum, so PyTorch broadcasting makes it rank-polymorphic in
# ``x_val``: when the innermost level is Dense, ``tile_k`` is 1-D and
# ``x[tile_k]`` has shape ``(K,)`` which left-pads to ``(1, 1, K)`` against
# the parent-inherited ``a_val`` of shape ``(I, J, K)``; for any non-Dense
# innermost level, ``tile_k`` is ND and ``x[tile_k]`` is already ``(I, J, K)``.
# Both paths produce ``(I, J, K) → .sum(dim=-1) → (I, J)`` identically.
# ----------------------------------------------------------------------------

# Logical (I=4, J=3, K=6) tensor.  Irregular k-nnz per (i, j), one wholly
# empty (i, j) plane, empty i=2 row at root for Bitmap/Padded masking.
#   A[0, 0, :] = [1, 0, 2, 0, 0, 3]   nnz=3
#   A[0, 1, :] = [0, 0, 0, 0, 0, 0]   nnz=0  ← empty (i, j) plane
#   A[0, 2, :] = [0, 4, 0, 0, 0, 0]   nnz=1
#   A[1, 0, :] = [5, 0, 0, 6, 7, 8]   nnz=4  ← max k-nnz
#   A[1, 1, :] = [0, 0, 9, 0, 10, 0]  nnz=2
#   A[1, 2, :] = [0, 0, 0, 0, 0, 11]  nnz=1
#   A[2, *, :] = 0                    nnz=0  ← empty root i=2
#   A[3, 0, :] = [12, 0, 13, 0, 0, 0] nnz=2
#   A[3, 1, :] = [0, 14, 0, 0, 15, 0] nnz=2
#   A[3, 2, :] = [0, 0, 0, 16, 0, 17] nnz=2
_DENSE_A_3D = torch.tensor(
    [
        [
            [1.0, 0.0, 2.0, 0.0, 0.0, 3.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 4.0, 0.0, 0.0, 0.0, 0.0],
        ],
        [
            [5.0, 0.0, 0.0, 6.0, 7.0, 8.0],
            [0.0, 0.0, 9.0, 0.0, 10.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 11.0],
        ],
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        [
            [12.0, 0.0, 13.0, 0.0, 0.0, 0.0],
            [0.0, 14.0, 0.0, 0.0, 15.0, 0.0],
            [0.0, 0.0, 0.0, 16.0, 0.0, 17.0],
        ],
    ],
    device=DEVICE,
)
_SHAPE_3D = (4, 3, 6)
_I, _J, _K = _SHAPE_3D
_X = torch.arange(_K, dtype=torch.float32, device=DEVICE) * 0.1 + 1.0  # (K,)
_GARBAGE = 777.0


def _int64(xs):
    return torch.tensor(xs, dtype=torch.int64, device=DEVICE)


def _bool(xs):
    return torch.tensor(xs, dtype=torch.bool, device=DEVICE)


def _build_sparse_3d(fmt0: str, fmt1: str, fmt2: str) -> hl.SparseTensor:
    """Re-encode ``_DENSE_A_3D`` under a given ``(fmt0, fmt1, fmt2)`` triple."""
    dense = _DENSE_A_3D
    I, J, K = dense.shape

    def nz_k(i: int, j: int) -> list[int]:
        return (dense[i, j] != 0).nonzero(as_tuple=False).flatten().tolist()

    def i_has_nnz(i: int) -> bool:
        return bool((dense[i] != 0).any())

    def ij_has_nnz(i: int, j: int) -> bool:
        return bool((dense[i, j] != 0).any())

    # ========== Root (i) ==========
    ptrs0 = coords0 = bitmaps0 = None
    if fmt0 == "Dense":
        root_slots = [(i, True) for i in range(I)]
    elif fmt0 == "Compressed":
        nonempty = [i for i in range(I) if i_has_nnz(i)]
        ptrs0 = _int64([0, len(nonempty)])
        coords0 = _int64(nonempty)
        root_slots = [(i, True) for i in nonempty]
    elif fmt0 == "Bitmap":
        bmp = [i_has_nnz(i) for i in range(I)]
        bitmaps0 = _bool(bmp)
        root_slots = [(i, bmp[i]) for i in range(I)]
    else:
        raise AssertionError(fmt0)

    num_root = len(root_slots)

    # ========== Middle (j per i) ==========
    # ``mid_slot_lists[s]`` is a list of ``(j, present)`` for root slot s.
    ptrs1 = coords1 = bitmaps1 = None
    mid_slot_lists: list[list[tuple[int, bool]]] = []

    if fmt1 == "Dense":
        for (i, present) in root_slots:
            if present and i != -1:
                mid_slot_lists.append([(j, True) for j in range(J)])
            else:
                mid_slot_lists.append([(j, False) for j in range(J)])
    elif fmt1 == "Compressed":
        ptrs_l = [0]
        coords_l: list[int] = []
        for (i, present) in root_slots:
            if present and i != -1:
                nz_j = [j for j in range(J) if ij_has_nnz(i, j)]
                mid_slot_lists.append([(j, True) for j in nz_j])
                coords_l.extend(nz_j)
            else:
                mid_slot_lists.append([])
            ptrs_l.append(len(coords_l))
        ptrs1 = _int64(ptrs_l)
        coords1 = _int64(coords_l)
    elif fmt1 == "Padded":
        pad_size_1 = max(
            (
                sum(1 for j in range(J) if ij_has_nnz(i, j))
                for (i, present) in root_slots
                if present and i != -1
            ),
            default=1,
        )
        pad_size_1 = max(pad_size_1, 1)
        coord_buf = torch.full(
            (num_root, pad_size_1), -1, dtype=torch.int64, device=DEVICE
        )
        for s, (i, present) in enumerate(root_slots):
            if present and i != -1:
                nz_j = [j for j in range(J) if ij_has_nnz(i, j)]
                for k_, j in enumerate(nz_j):
                    coord_buf[s, k_] = j
                slots = [(j, True) for j in nz_j] + [(-1, False)] * (
                    pad_size_1 - len(nz_j)
                )
            else:
                slots = [(-1, False)] * pad_size_1
            mid_slot_lists.append(slots)
        coords1 = coord_buf
    elif fmt1 == "Jagged":
        # Jagged's prefix semantics include every slot in ``[0, last+1)`` as
        # storage-present (no coord / mask at this level); the inner storage
        # records ``dense[i, j]`` verbatim so logically-zero planes inside
        # the prefix contribute zero naturally.  Marking them ``present=True``
        # (even when the plane is all zeros) is what keeps Dense inner from
        # leaking ``_GARBAGE`` at (i, j) positions Jagged doesn't mask.
        ptrs_l = [0]
        for (i, present) in root_slots:
            if present and i != -1:
                nz = [j for j in range(J) if ij_has_nnz(i, j)]
                last = nz[-1] if nz else -1
                slots = [(j, True) for j in range(last + 1)]
            else:
                slots = []
            mid_slot_lists.append(slots)
            ptrs_l.append(ptrs_l[-1] + len(slots))
        ptrs1 = _int64(ptrs_l)
    elif fmt1 == "Bitmap":
        bmp_buf = torch.zeros((num_root, J), dtype=torch.bool, device=DEVICE)
        for s, (i, present) in enumerate(root_slots):
            if present and i != -1:
                slots = [(j, ij_has_nnz(i, j)) for j in range(J)]
                for j, pres in slots:
                    bmp_buf[s, j] = pres
            else:
                slots = [(j, False) for j in range(J)]
            mid_slot_lists.append(slots)
        bitmaps1 = bmp_buf
    else:
        raise AssertionError(fmt1)

    # ========== Inner (k per (i, j)) ==========
    # Build ptrs2 / coords2 / bitmaps2 and flat values.
    ptrs2 = coords2 = bitmaps2 = None
    values: torch.Tensor

    # Enumerate every "inner slot owner" = (root_idx, mid_idx) with its
    # logical (i, j) and its presence (from both root and mid).
    inner_owners: list[tuple[int, bool]] = []
    for s, mid_slots in enumerate(mid_slot_lists):
        i_logical, root_present = root_slots[s]
        for (j_logical, mid_present) in mid_slots:
            present = root_present and mid_present and i_logical != -1 and j_logical != -1
            if present:
                inner_owners.append((s, True))
            else:
                inner_owners.append((s, False))
    num_owners = len(inner_owners)

    # Flatten root/mid indices so we can recover (i_logical, j_logical) per owner.
    owner_ij: list[tuple[int, int, bool]] = []
    for s, mid_slots in enumerate(mid_slot_lists):
        i_logical, root_present = root_slots[s]
        for (j_logical, mid_present) in mid_slots:
            present = root_present and mid_present and i_logical != -1 and j_logical != -1
            owner_ij.append((i_logical, j_logical, present))

    if fmt2 == "Dense":
        buf = torch.full((num_owners, K), _GARBAGE, device=DEVICE)
        for o, (i, j, present) in enumerate(owner_ij):
            if present:
                buf[o] = dense[i, j]
        values = buf.flatten()
    elif fmt2 == "Compressed":
        ptrs_l = [0]
        coords_l: list[int] = []
        vals_l: list[float] = []
        for (i, j, present) in owner_ij:
            if present:
                nz = nz_k(i, j)
                coords_l.extend(nz)
                vals_l.extend(float(dense[i, j, c]) for c in nz)
            ptrs_l.append(len(coords_l))
        ptrs2 = _int64(ptrs_l)
        coords2 = _int64(coords_l)
        values = torch.tensor(vals_l, device=DEVICE) if vals_l else torch.zeros(
            0, device=DEVICE
        )
    elif fmt2 == "Padded":
        pad_size_2 = max(
            (len(nz_k(i, j)) for (i, j, present) in owner_ij if present),
            default=1,
        )
        pad_size_2 = max(pad_size_2, 1)
        coord_buf = torch.full(
            (num_owners, pad_size_2), -1, dtype=torch.int64, device=DEVICE
        )
        val_buf = torch.full((num_owners, pad_size_2), _GARBAGE, device=DEVICE)
        for o, (i, j, present) in enumerate(owner_ij):
            if present:
                nz = nz_k(i, j)
                for p, c in enumerate(nz):
                    coord_buf[o, p] = c
                    val_buf[o, p] = dense[i, j, c]
        coords2 = coord_buf
        values = val_buf.flatten()
    elif fmt2 == "Jagged":
        ptrs_l = [0]
        vals_l: list[float] = []
        for (i, j, present) in owner_ij:
            if present:
                nz = nz_k(i, j)
                if nz:
                    last = nz[-1]
                    for c in range(last + 1):
                        vals_l.append(float(dense[i, j, c]))
                    ptrs_l.append(len(vals_l))
                else:
                    ptrs_l.append(len(vals_l))
            else:
                ptrs_l.append(len(vals_l))
        ptrs2 = _int64(ptrs_l)
        values = torch.tensor(vals_l, device=DEVICE) if vals_l else torch.zeros(
            0, device=DEVICE
        )
    elif fmt2 == "Bitmap":
        bmp_buf = torch.zeros((num_owners, K), dtype=torch.bool, device=DEVICE)
        val_buf = torch.full((num_owners, K), _GARBAGE, device=DEVICE)
        for o, (i, j, present) in enumerate(owner_ij):
            if present:
                val_buf[o] = dense[i, j]
                for c in nz_k(i, j):
                    bmp_buf[o, c] = True
        bitmaps2 = bmp_buf
        values = val_buf.flatten()
    else:
        raise AssertionError(fmt2)

    kwargs = dict(
        values=values,
        shape=_SHAPE_3D,
        ptrs=(ptrs0, ptrs1, ptrs2),
        coords=(coords0, coords1, coords2),
    )
    if any(b is not None for b in (bitmaps0, bitmaps1, bitmaps2)):
        kwargs["bitmaps"] = (bitmaps0, bitmaps1, bitmaps2)
    return hl.SparseTensor(**kwargs)


@helion.kernel(config=helion.Config(block_sizes=[4, 4, 8]))
def sdot_kernel(
    A: hl.SparseTensor,
    x: torch.Tensor,
    fmt0: hl.constexpr,
    fmt1: hl.constexpr,
    fmt2: hl.constexpr,
) -> torch.Tensor:
    """Format-invariant SDOT via element-wise mul-then-sum.  ``a_val`` is
    always ``(I, J, K)`` via parent-chain inheritance; ``x_val`` is either
    ``(I, J, K)`` (non-Dense innermost: ND ``tile_k`` gather) or ``(K,)``
    (Dense innermost: 1-D ``tile_k``).  Broadcasting left-pads the 1-D case
    so both paths collapse to ``(I, J, K) → (I, J)`` after the reduction."""
    I = A.shape[0]
    J = A.shape[1]
    C = torch.zeros(I * J, dtype=x.dtype, device=x.device)
    for tile_i in hl.sparse_tile(A, dim=0, levelformat=fmt0):
        for tile_j in hl.sparse_tile(tile_i, dim=1, levelformat=fmt1):
            acc = hl.zeros([tile_i.size(0), tile_j.size(-1)], dtype=x.dtype)
            for tile_k in hl.sparse_tile(tile_j, dim=2, levelformat=fmt2):
                a_val = tile_k.value
                x_val = x[tile_k]
                acc = acc + (a_val * x_val).sum(dim=-1)
            flat_idx = tile_i[:, None] * J + tile_j
            C[flat_idx] = acc
    return C.view(I, J)


# Curated 30-combo sweep covering every format at every position.  Root is
# restricted to ``Dense``/``Compressed``/``Bitmap`` (Padded / Jagged are
# forbidden at root).
_LAYOUTS: list[tuple[str, str, str]] = [
    # root=Dense (12)
    ("Dense", "Dense", "Dense"),
    ("Dense", "Dense", "Compressed"),
    ("Dense", "Dense", "Padded"),
    ("Dense", "Dense", "Jagged"),
    ("Dense", "Dense", "Bitmap"),
    ("Dense", "Compressed", "Dense"),
    ("Dense", "Compressed", "Compressed"),
    ("Dense", "Padded", "Dense"),
    ("Dense", "Jagged", "Dense"),
    ("Dense", "Bitmap", "Dense"),
    ("Dense", "Padded", "Bitmap"),
    ("Dense", "Padded", "Jagged"),
    # root=Compressed (10)
    ("Compressed", "Dense", "Dense"),
    ("Compressed", "Compressed", "Compressed"),
    ("Compressed", "Compressed", "Dense"),
    ("Compressed", "Compressed", "Padded"),
    ("Compressed", "Padded", "Compressed"),
    ("Compressed", "Jagged", "Jagged"),
    ("Compressed", "Bitmap", "Dense"),
    ("Compressed", "Dense", "Bitmap"),
    ("Compressed", "Bitmap", "Bitmap"),
    ("Compressed", "Compressed", "Jagged"),
    # root=Bitmap (8)
    ("Bitmap", "Dense", "Dense"),
    ("Bitmap", "Compressed", "Compressed"),
    ("Bitmap", "Bitmap", "Bitmap"),
    ("Bitmap", "Dense", "Jagged"),
    ("Bitmap", "Jagged", "Bitmap"),
    ("Bitmap", "Padded", "Padded"),
    ("Bitmap", "Dense", "Compressed"),
    ("Bitmap", "Padded", "Dense"),
]


class TestSparseTile3D(TestCase):
    def test_sdot_all_layouts(self) -> None:
        # Reference: einsum over k of dense A and x.
        expected = torch.einsum("ijk,k->ij", _DENSE_A_3D, _X)
        for fmt in _LAYOUTS:
            with self.subTest(fmt=fmt):
                A = _build_sparse_3d(*fmt)
                fmt0, fmt1, fmt2 = fmt
                got = sdot_kernel(A, _X, fmt0, fmt1, fmt2)
                torch.testing.assert_close(got, expected)


if __name__ == "__main__":
    unittest.main()
