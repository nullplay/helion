"""The post-grad pass helion.inductor adds to Inductor's, for cuda_backend
"helion": Inductor's own passes keep some rewrites to Triton, which Helion
kernels can take too."""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch._inductor import config as inductor_config
from torch._inductor.custom_graph_pass import CustomGraphModulePass
from torch._inductor.custom_graph_pass import get_hash_for_files

from .online_softmax import online_softmax_patterns
from .strided_reductions import split_strided_reductions

if TYPE_CHECKING:
    import torch


class HelionPostGradPass(CustomGraphModulePass):
    """Runs after Inductor's post-grad passes on a graph with a CUDA tensor, as
    the CUDA device's custom backend pass, after the one that was registered
    before it (`inner`). With cuda_backend "helion" it rewrites softmax to
    online softmax, as Inductor does for Triton: torch.softmax(x, -1) is then
    one pass over each row for the max and the sum of exps (online_softmax_reduce),
    and one that writes exp(x - max) / sum, rather than three. It also splits
    a reduction over dims that memory keeps apart into one over the innermost
    contiguous run of them and one over the rest (split_strided_reductions()),
    as BatchNorm's over N, H and W of an NCHW tensor."""

    def __init__(self, inner: CustomGraphModulePass | None) -> None:
        self.inner = inner

    def __call__(self, gm: torch.fx.GraphModule) -> None:
        if self.inner is not None:
            self.inner(gm)
        # As Inductor applies its replacements, under pattern_matcher.
        if not (
            inductor_config.cuda_backend == "helion" and inductor_config.pattern_matcher
        ):
            return
        changed = split_strided_reductions(gm.graph)
        if inductor_config.online_softmax and online_softmax_patterns().apply(gm.graph):
            changed = True
        if changed:
            gm.graph.lint()
            gm.recompile()

    def uuid(self) -> object:
        # Inductor's FX graph cache key holds it, with cuda_backend,
        # online_softmax and pattern_matcher, which __call__ reads. None, which
        # asks to skip caching, stays None.
        inner = None
        if self.inner is not None and (inner := self.inner.uuid()) is None:
            return None
        return (get_hash_for_files((__file__,)), inner)
