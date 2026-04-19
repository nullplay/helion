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
      ``hl.sparse_tile`` loop nest). Compressed levels store their
      ``ptr`` / ``coord`` tensor; Dense levels store ``None``.

    The per-level format (Dense / Compressed) is given at the call site
    via ``hl.sparse_tile(..., levelformat=...)`` — it is not stored on
    the tensor. The tensor just supplies whatever arrays the chosen
    formats need.
    """

    values: torch.Tensor  # flat, length nnz
    shape: tuple[int, ...]
    # length == len(shape); None for Dense levels, tensor for Compressed.
    ptrs: tuple[torch.Tensor | None, ...] = ()
    coords: tuple[torch.Tensor | None, ...] = ()
