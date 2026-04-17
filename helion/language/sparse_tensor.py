from __future__ import annotations

import dataclasses

import torch


@dataclasses.dataclass
class SparseTensor:
    """A sparse tensor described level-by-level.

    Holds the three standard arrays (ptr, coord, values) and the logical shape.
    Level formats (Dense, Compressed, Jagged, Padded) are specified at call time
    via ``sparse_tile``.

    Example::

        ptr   = torch.tensor([0, 2, 3, 5], device="cuda")   # 3 rows, 5 nnz
        coord = torch.tensor([0, 2, 1, 0, 3], device="cuda")
        val   = torch.tensor([1., 2., 3., 4., 5.], device="cuda")
        A = hl.SparseTensor(ptr=ptr, coord=coord, values=val, shape=(3, 4))

    Iteration is expressed with ``hl.sparse_tile(source, dim=..., levelformat=...)``,
    where ``source`` is either this tensor (for the root level) or an outer
    ``SparseTile`` produced by a previous ``hl.sparse_tile`` call.
    """

    ptr: torch.Tensor  # [n0+1]  position / row-pointer array
    coord: torch.Tensor  # [nnz]   coordinate (column-index) array
    values: torch.Tensor  # [nnz]  nonzero values
    shape: tuple[int, ...]
