"""Graph passes helion.inductor adds to Inductor's, for cuda_backend "helion":
rewrites of the post-grad FX graph that run before lowering. post_grad.py
holds the pass Inductor runs (HelionPostGradPass), and each pattern family it
applies has a module of its own (online_softmax.py, strided_reductions.py),
and graph.py holds the helpers they share."""

from __future__ import annotations
