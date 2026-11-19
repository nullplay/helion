"""Helpers the graph passes share."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

import torch
from torch.utils import _pytree as pytree

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.fx.node import Argument


def insert_call(
    graph: torch.fx.Graph,
    before: torch.fx.Node,
    fake_mode: FakeTensorMode,
    target: Callable[..., Any],
    *args: Argument,
    **kwargs: Argument,
) -> torch.fx.Node:
    """A call of `target` inserted before `before`, its meta["val"] computed
    under `fake_mode` from its arguments', as Inductor's lowering reads it."""
    with graph.inserting_before(before):
        new = graph.call_function(target, args, kwargs)
    fake_args, fake_kwargs = pytree.tree_map_only(
        torch.fx.Node, lambda n: n.meta["val"], (args, kwargs)
    )
    with fake_mode:
        new.meta["val"] = target(*fake_args, **fake_kwargs)
    return new
