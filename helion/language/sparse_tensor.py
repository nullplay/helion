from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


@dataclasses.dataclass(eq=False)
class SparseTensor:
    """A sparse tensor described level-by-level.

    Storage is uniform across formats:

    * ``values`` is a tensor of shape ``(nnz, *value_shape)``. Each leaf
      position in the coordinate tree owns one block of shape
      ``value_shape``; when ``value_shape == ()`` the block degenerates to
      a scalar (so ``values`` is flat of length ``nnz``). Leaf positions
      are laid out along axis 0 in the order determined by the per-level
      lowering, so the same storage works for DD / DC / CD / CC / ... .
      The payload shape is inferred from ``values.shape[1:]`` and is
      exposed to user code as ``A.value_shape``; the leaf count is
      exposed as ``A.nnz``.
    * ``shape`` is positional: ``shape[d]`` is the size of dim ``d``.
    * ``ptrs`` / ``coords`` / ``bitmaps`` have length ``len(shape)`` and
      are aligned to **tensor level order** (outer → inner in the user's
      ``hl.sparse_tile`` loop nest). Dense levels store ``None`` in all
      three.  Compressed levels store a 1-D ``ptr`` tensor and a 1-D
      ``coord`` tensor. Padded levels store ``None`` for ``ptr`` and a
      2-D ``coord`` tensor of shape ``(flat_parent_count, pad_size)`` —
      the leading dim enumerates the parent's flat-storage positions and
      the trailing dim is the fixed pad width (e.g. ``(M, pad_size)``
      for ELL, ``(nnz_parent, pad_size)`` when nested under a Compressed
      level).  A root Padded level — no parent — stores a 1-D
      ``(pad_size,)`` coord instead.  Jagged levels store a 1-D ``ptr``
      tensor (same layout as Compressed) and ``None`` for ``coord``: the
      per-parent segment length is ``ptr[parent+1] - ptr[parent]`` and
      the coord exposed in the loop body is the local tile index itself
      (0..length-1), so no coord tensor is needed.  Useful for ragged
      tensors whose non-zeros form a contiguous column prefix per row.
      Bitmap levels store ``None`` for both ``ptr`` and ``coord`` and a
      ``torch.bool`` tensor in ``bitmaps[level]`` of shape
      ``(len(parent's storage), shape[level])`` — the leading dim
      enumerates the parent's flat-storage positions (``prod(shape[:level])``
      for a Dense parent chain, ``nnz_parent`` under Compressed, etc.)
      and the trailing dim is the full dense extent of this level.
      Addressing mirrors Dense (``parent_pos * shape[level] + local``);
      the bitmap is loaded per-tile and AND'd into the tile mask so
      padded slots are masked out of loads/stores/reductions.  A root
      Bitmap level — no parent — stores a 1-D ``(shape[0],)`` bitmap
      instead.

    The per-level format (Dense / Compressed / Padded / Jagged / Bitmap)
    is given at the call site via ``hl.sparse_tile(..., levelformat=...)``
    — it is not stored on the tensor. The tensor just supplies whatever
    arrays the chosen formats need.
    """

    values: torch.Tensor  # shape (nnz, *value_shape); flat when value_shape == ()
    shape: tuple[int, ...]
    # length == len(shape); None for Dense (and for Padded's ptrs slot,
    # and for Jagged's coords slot, and for Bitmap's ptrs/coords slots),
    # 1-D tensor for Compressed/Jagged ptrs and Compressed coords,
    # N-D tensor for Padded coords.
    ptrs: tuple[torch.Tensor | None, ...] = ()
    coords: tuple[torch.Tensor | None, ...] = ()
    # length == len(shape); None for non-Bitmap levels. Bitmap levels
    # store a torch.bool tensor of shape
    # ``(len(parent's storage), shape[level])`` (root: ``(shape[0],)``).
    bitmaps: tuple[torch.Tensor | None, ...] = ()

    @property
    def value_shape(self) -> tuple[int, ...]:
        """Per-leaf payload shape. ``()`` means scalar payload."""
        return tuple(self.values.shape[1:])

    @property
    def nnz(self) -> int:
        """Number of stored leaf positions (``values.shape[0]``)."""
        return int(self.values.shape[0])
