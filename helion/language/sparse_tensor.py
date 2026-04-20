from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


@dataclasses.dataclass(eq=False)
class SparseTensor:
    """A sparse tensor described level-by-level.

    Storage is uniform across formats:

    * ``values`` is always a flat 1-D tensor of length ``nnz`` (the total
      number of leaf positions). The way to turn a user-visible multi-axis
      value into a flat position is determined by the per-level lowering,
      so the same flat array works for DD / DC / CD / CC / ... .
    * ``shape`` is positional: ``shape[d]`` is the size of dim ``d``.
    * ``ptrs`` / ``coords`` have length ``len(shape)`` and are aligned to
      **tensor level order** (outer → inner in the user's
      ``hl.sparse_tile`` loop nest). Dense levels store ``None`` in both.
      Compressed levels store a 1-D ``ptr`` tensor and a 1-D ``coord``
      tensor. Padded levels store ``None`` for ``ptr`` and a 2-D
      ``coord`` tensor of shape ``(flat_parent_count, pad_size)`` — the
      leading dim enumerates the parent's flat-storage positions and the
      trailing dim is the fixed pad width (e.g. ``(M, pad_size)`` for
      ELL, ``(nnz_parent, pad_size)`` when nested under a Compressed
      level).  A root Padded level — no parent — stores a 1-D
      ``(pad_size,)`` coord instead.

    The per-level format (Dense / Compressed / Padded) is given at the
    call site via ``hl.sparse_tile(..., levelformat=...)`` — it is not
    stored on the tensor. The tensor just supplies whatever arrays the
    chosen formats need.
    """

    values: torch.Tensor  # flat, length nnz
    shape: tuple[int, ...]
    # length == len(shape); None for Dense (and for Padded's ptrs slot),
    # 1-D tensor for Compressed ptrs/coords, N-D tensor for Padded coords.
    ptrs: tuple[torch.Tensor | None, ...] = ()
    coords: tuple[torch.Tensor | None, ...] = ()
