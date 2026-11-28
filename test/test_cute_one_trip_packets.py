"""CuTe ``issue_packets_first``: one-trip tile loops' packets loaded first and kept in registers.

Codegen checks bind on CPU; ``test_cute_one_trip_packets_gpu.py`` runs the
kernels.
"""

from __future__ import annotations

import ast
import textwrap

import pytest
import torch

import helion
from helion._compiler.cute.one_trip_packets import issue_packets_first
from helion._testing import cpu_render_target
from helion._testing.cute_batch_norm import batch_norm_inputs
from helion._testing.cute_batch_norm import batch_norm_training
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import tcgen05_mma_support
from helion._testing.cute_one_trip_packets import normalize_with_residual

pytestmark = pytest.mark.cpu_only

VT = "ir.VectorType.get([8], cutlass.Uint16.mlir_type)"


def _code(config: dict[str, object], rows: int = 8, rdim: int = 1536) -> str:
    x = torch.randn(rows, rdim, dtype=torch.bfloat16)
    r = torch.randn(rows, rdim, dtype=torch.bfloat16)
    w = torch.randn(rows, dtype=torch.float32)
    args = (x, r, w, torch.empty_like(x), torch.empty(rows, dtype=torch.float32))
    with cpu_render_target(
        mma_support=tcgen05_mma_support(), reduction_smem_budget_bytes=232448
    ):
        bound = cpu_bind(normalize_with_residual, args)
        base = dict(bound.config_spec.default_config())
        base.update(config)
        return bound.to_code(helion.Config(**base))


def _config(threads: int, block: int, vector: int = 8) -> dict[str, object]:
    return {
        "block_sizes": [1, block, block],
        "num_threads": [1, threads, threads],
        "cute_vector_widths": [1, vector, vector],
        "cute_lane_layouts": ["blocked", "strided", "strided"],
    }


def test_one_trip_lane_loops_share_their_packets() -> None:
    # 64 threads x 8 lanes of a vector cover 512 elements per lane; one
    # 2048-element tile covers the rdim of 1536 in one trip of each loop.
    code = _code(_config(64, 2048))
    kernel = code[code.index("@cute.kernel") :]
    # The tile loops are inlined and both packets of a lane are loaded once,
    # in the reduction's lane loop, before the block reduction.
    assert "range(cutlass.Int32(0), cutlass.Int32(1536)" not in kernel
    first = kernel.index("_one_trip_packets_0.append(")
    assert kernel.index("_one_trip_packets_1.append(") < kernel.index(
        "_cute_grouped_reduce"
    )
    assert "r.iterator" in kernel[first : kernel.index("_cute_grouped_reduce")]
    # The normalize reads both from registers, in a trace-time lane loop.
    normalize = kernel[kernel.index("tile_offset_2 = cutlass.Int32(0)") :]
    assert "for lane_2 in cutlass.range_constexpr(4)" in normalize
    assert "= _one_trip_packets_0[lane_2]" in normalize
    assert "= _one_trip_packets_1[lane_2]" in normalize
    assert "cute.arch.load(" not in normalize.split("else:")[0]
    # The weight, read after the block reduction in the source, loads first.
    assert kernel.index("w.iterator") < first


def test_multi_trip_tile_loops_keep_their_loads() -> None:
    # A 512-element tile loops three times over 1536: nothing is shared.
    code = _code(_config(64, 512))
    assert "_one_trip_packets_0" not in code
    assert (
        "range(cutlass.Int32(0), cutlass.Int32(1536), cutlass.Int32(_BLOCK_SIZE_1))"
        in code
    )


def test_packets_past_the_register_budget_keep_their_loads() -> None:
    # 16 threads x V16: 8 lanes of 32 bytes per packet stream, 256 bytes. The
    # x packets (read by both loops) fill REGISTER_BUDGET_BYTES; the
    # residual's stay in the normalize.
    code = _code(_config(16, 2048, vector=16))
    kernel = code[code.index("@cute.kernel") :]
    assert "_one_trip_packets_0" in kernel
    assert (
        "r.iterator" not in kernel[: kernel.index("tile_offset_2 = cutlass.Int32(0)")]
    )
    normalize = kernel[kernel.index("tile_offset_2 = cutlass.Int32(0)") :]
    assert "r.iterator" in normalize


def test_lane_loops_past_eight_lanes_keep_their_loads() -> None:
    # 16 threads x V8: 16 lanes, more than MAX_UNROLLED_LANES: the loops keep
    # their form and loads.
    code = _code(_config(16, 2048))
    assert "_one_trip_packets_0" not in code


def test_batch_norm_loads_first() -> None:
    # helion.inductor's BatchNorm at 64 threads x V8: 4 lanes of guarded
    # packets over a 2048-element rdim, one trip of each loop.
    inputs = batch_norm_inputs()
    with cpu_render_target(
        mma_support=tcgen05_mma_support(), reduction_smem_budget_bytes=232448
    ):
        bound = cpu_bind(batch_norm_training, inputs)
        config = dict(bound.config_spec.default_config())
        config.update(
            block_sizes=[1, 2048, 2048],
            num_threads=[1, 64, 64],
            cute_vector_widths=[1, 8, 8],
            cute_lane_layouts=["blocked", "strided", "strided"],
        )
        code = bound.to_code(helion.Config(**config))
    kernel = code[code.index("@cute.kernel") :]
    reduce = kernel.index("_cute_grouped_reduce")
    # The running statistics (straight-line loads after both block
    # reductions), and the weight and bias (loaded in every lane of the
    # normalize), load once before the first block reduction; the input and
    # residual packets of every lane are predicated loads in the first lane
    # loop.
    for tensor in ("in1", "in2", "in3", "in4"):
        assert kernel.index(f"({tensor}.iterator") < reduce
    for tensor in ("in3", "in4"):
        assert kernel.count(f"({tensor}.iterator") == 1
    assert (
        kernel.index("_one_trip_packets_1_lane = _cute_load_v16b_if(in5.iterator")
        < reduce
    )
    # The deviations and the normalize read them back: no other in0 or in5
    # packet load is left.
    assert kernel.count("cute.arch.load(in0.iterator") == 0
    assert kernel.count("cute.arch.load(in5.iterator") == 0
    assert "for lane_2 in cutlass.range_constexpr(4)" in kernel


def _body(text: str) -> list[ast.stmt]:
    return ast.parse(textwrap.dedent(text)).body


SWEEPS = f"""
pid = cutlass.Int32(cute.arch.block_idx()[0])
for t1 in range(cutlass.Int32(0), cutlass.Int32(96), cutlass.Int32(_B1)):
    for lane_1 in cutlass.range_constexpr(2):
        base_1 = t1 + (cutlass.Int32(lane_1) * 8 + cutlass.Int32(cute.arch.thread_idx()[0])) * 8
        if base_1 < 96:
            p = cute.arch.load(x.iterator + pid * 96 + base_1, {VT})
            acc = acc + p[0]
STORE
for t2 in range(cutlass.Int32(0), cutlass.Int32(96), cutlass.Int32(_B2)):
    for lane_2 in range(2):
        base_2 = t2 + (cutlass.Int32(lane_2) * 8 + cutlass.Int32(cute.arch.thread_idx()[0])) * 8
        if base_2 < 96:
            q = cute.arch.load(x.iterator + pid * 96 + base_2, {VT})
            s = cute.arch.load(y.iterator + pid * 96 + base_2, {VT})
            _cute_store_u16_vec(out.iterator + pid * 96 + base_2, [q[0], s[0]])
"""


def _run(text: str, disjoint: set[frozenset[str]]) -> str:
    body = issue_packets_first(
        _body(text),
        constexpr_values={"_B1": 128, "_B2": 128},
        tensor_names={"x", "y", "out", "m"},
        proven_disjoint_tensor_pairs=disjoint,
    )
    return ast.unparse(ast.Module(body=body, type_ignores=[]))


ALL_DISJOINT = {
    frozenset(pair)
    for pair in (("x", "out"), ("y", "out"), ("x", "m"), ("y", "m"), ("out", "m"))
}


def test_sweeps_share_packets_of_read_only_disjoint_tensors() -> None:
    out = _run(SWEEPS.replace("STORE", "(m.iterator + pid).store(acc)"), ALL_DISJOINT)
    assert "_one_trip_packets_0_lane = _cute_load_v16b_if(x.iterator" in out
    assert "_one_trip_packets_1_lane = _cute_load_v16b_if(y.iterator" in out
    assert "q = _one_trip_packets_0[lane_2]" in out
    assert "s = _one_trip_packets_1[lane_2]" in out
    assert "for lane_2 in cutlass.range_constexpr(2)" in out


def test_a_tensor_not_proven_disjoint_from_a_write_keeps_its_loads() -> None:
    # y may alias m, which the kernel writes between the two loops: y's
    # packet stays where it was; x's (disjoint from every write) is shared.
    disjoint = ALL_DISJOINT - {frozenset(("y", "m"))}
    out = _run(SWEEPS.replace("STORE", "(m.iterator + pid).store(acc)"), disjoint)
    assert "q = _one_trip_packets_0[lane_2]" in out
    assert "s = cute.arch.load(y.iterator" in out


def test_a_tensor_the_kernel_writes_keeps_its_loads() -> None:
    out = _run(SWEEPS.replace("STORE", "(y.iterator + pid).store(acc)"), ALL_DISJOINT)
    assert "s = cute.arch.load(y.iterator" in out


def test_scalar_load_a_loop_repeats_moves_above_the_loops() -> None:
    text = """
    pid = cutlass.Int32(cute.arch.block_idx()[0])
    for t1 in range(cutlass.Int32(0), cutlass.Int32(96), cutlass.Int32(64)):
        acc = acc + cutlass.Float32(1)
    (out.iterator + pid).store(acc)
    for t2 in range(cutlass.Int32(0), cutlass.Int32(96), cutlass.Int32(64)):
        w = (y.iterator + pid).load()
        acc = acc + w
    (m.iterator + pid).store(acc)
    """
    out = _run(text, ALL_DISJOINT)
    lines = out.splitlines()
    assert lines[1] == "w = (y.iterator + pid).load()"
    assert out.count("y.iterator") == 1
    # Not when an earlier write may reach it.
    out = _run(text, ALL_DISJOINT - {frozenset(("y", "out"))})
    assert out.splitlines()[1].startswith("for t1")


def test_straight_line_scalar_load_after_a_block_reduction_moves() -> None:
    # Above the block reduction, the load's round trip overlaps the wait.
    text = """
    pid = cutlass.Int32(cute.arch.block_idx()[0])
    for t1 in range(cutlass.Int32(0), cutlass.Int32(96), cutlass.Int32(64)):
        acc = acc + cutlass.Float32(1)
    s = _cute_grouped_reduce_shared_two_stage(acc, 'sum', cutlass.Float32(0), lane, lane_in_group, pre_lane, pre=1, group_span=64, group_count=1)
    w = (y.iterator + pid).load()
    (m.iterator + pid).store(s + w)
    """
    out = _run(text, ALL_DISJOINT)
    assert out.splitlines()[1] == "w = (y.iterator + pid).load()"
    # After a warp reduction only, it stays.
    out = _run(
        text.replace(
            "_cute_grouped_reduce_shared_two_stage", "cute.arch.warp_reduction_sum"
        ),
        ALL_DISJOINT,
    )
    assert out.splitlines()[1].startswith("for t1")


def test_scalar_load_in_a_trace_time_loop_stays() -> None:
    # A cutlass.range_constexpr loop is straight-line code once traced: a load
    # in it, under a one-trip runtime lane loop and after a warp reduction,
    # repeats nothing at run time and stays.
    text = """
    pid = cutlass.Int32(cute.arch.block_idx()[0])
    for lane_1 in range(1):
        s = cute.arch.warp_reduction_sum(acc, threads_in_group=32)
        for vec_lane_1 in cutlass.range_constexpr(2):
            w = (y.iterator + pid).load()
            acc = acc + w * s
    (m.iterator + pid).store(acc)
    """
    out = _run(text, ALL_DISJOINT)
    assert out.splitlines()[1].startswith("for lane_1")
    # The same load in a runtime loop of two trips moves.
    out = _run(text.replace("cutlass.range_constexpr(2)", "range(2)"), ALL_DISJOINT)
    assert out.splitlines()[1] == "w = (y.iterator + pid).load()"


def test_straight_line_scalar_load_stays() -> None:
    # With no barrier before it, a load moved up would only hold a register
    # across the loop before its use; under a one-trip loop the same.
    for text in (
        """
        pid = cutlass.Int32(cute.arch.block_idx()[0])
        for t1 in range(cutlass.Int32(0), cutlass.Int32(96), cutlass.Int32(64)):
            acc = acc + cutlass.Float32(1)
        w = (y.iterator + pid).load()
        (m.iterator + pid).store(acc + w)
        """,
        """
        pid = cutlass.Int32(cute.arch.block_idx()[0])
        for t1 in range(cutlass.Int32(0), cutlass.Int32(96), cutlass.Int32(64)):
            acc = acc + cutlass.Float32(1)
        for t2 in cutlass.range_constexpr(1):
            w = (y.iterator + pid).load()
            acc = acc + w
        (m.iterator + pid).store(acc)
        """,
    ):
        out = _run(text, ALL_DISJOINT)
        assert out.splitlines()[1].startswith("for t1")


def test_scalar_load_in_one_branch_stays() -> None:
    text = """
    pid = cutlass.Int32(cute.arch.block_idx()[0])
    for t1 in range(cutlass.Int32(0), cutlass.Int32(96), cutlass.Int32(64)):
        if acc > 0:
            w = (y.iterator + pid).load()
            acc = acc + w
    """
    out = _run(text, ALL_DISJOINT)
    assert out.splitlines()[1].startswith("for t1")


def test_a_producer_packet_reads_only_names_bound_before_the_shared_loads() -> None:
    # The first lane loop's second packet reads ``m``, which it binds after
    # its first packet: the shared load, issued before the first packet,
    # spells ``m`` by its definition.
    text = f"""
    pid = cutlass.Int32(cute.arch.block_idx()[0])
    for t1 in range(cutlass.Int32(0), cutlass.Int32(96), cutlass.Int32(_B1)):
        for lane_1 in cutlass.range_constexpr(2):
            base_1 = t1 + (cutlass.Int32(lane_1) * 8 + cutlass.Int32(cute.arch.thread_idx()[0])) * 8
            if base_1 < 96:
                p = cute.arch.load(x.iterator + pid * 96 + base_1, {VT})
                acc = acc + p[0]
            m = base_1 + pid * 96
            if base_1 < 96:
                q = cute.arch.load(y.iterator + m, {VT})
                acc = acc + q[0]
    for t2 in range(cutlass.Int32(0), cutlass.Int32(96), cutlass.Int32(_B2)):
        for lane_2 in range(2):
            base_2 = t2 + (cutlass.Int32(lane_2) * 8 + cutlass.Int32(cute.arch.thread_idx()[0])) * 8
            if base_2 < 96:
                s = cute.arch.load(y.iterator + (base_2 + pid * 96), {VT})
                _cute_store_u16_vec(out.iterator + pid * 96 + base_2, [s[0]])
    """
    out = _run(text, ALL_DISJOINT)
    lines = out.splitlines()
    shared = next(
        i
        for i, line in enumerate(lines)
        if "_lane = _cute_load_v16b_if(y.iterator" in line
    )
    assert "base_1 + pid * 96" in lines[shared]
    assert shared < next(
        i for i, line in enumerate(lines) if line.strip().startswith("m = ")
    )
    assert "s = _one_trip_packets_" in out


def test_a_key_that_reads_memory_keeps_its_load() -> None:
    # The normalize's guard reads k = m[pid], which the kernel writes between
    # the loops: moving the y packet (and its guard) into the first lane loop
    # would read m before that write. Its load stays.
    text = SWEEPS.replace(
        "STORE", "(m.iterator + pid).store(acc)\nk = (m.iterator + pid).load()"
    ).replace("        if base_2 < 96:", "        if base_2 < k:")
    out = _run(text, ALL_DISJOINT)
    assert "s = cute.arch.load(y.iterator" in out
    assert "(m.iterator + pid).load()" not in out.split("(m.iterator + pid).store")[0]


def test_a_key_read_through_getitem_keeps_its_load() -> None:
    # The same key spelled operator.getitem(m, pid), a read of m too.
    text = SWEEPS.replace(
        "STORE", "(m.iterator + pid).store(acc)\nk = operator.getitem(m, pid)"
    ).replace("        if base_2 < 96:", "        if base_2 < k:")
    out = _run(text, ALL_DISJOINT)
    assert "s = cute.arch.load(y.iterator" in out


def test_a_subscript_store_counts_as_a_write() -> None:
    # y[pid] = acc writes y as (y.iterator + pid).store(acc) does, so y's
    # packets keep their loads, as test_a_tensor_the_kernel_writes_keeps_its_loads.
    out = _run(SWEEPS.replace("STORE", "y[pid] = acc"), ALL_DISJOINT)
    assert "s = cute.arch.load(y.iterator" in out


def test_a_subscript_store_through_an_alias_counts_as_a_write() -> None:
    # alias = y makes alias[pid] = acc a write of y that names no kernel
    # tensor, and so does held[0][pid] = acc through a list that holds y: a
    # store but into an element of a container bound only to registers or
    # shared memory may write any tensor, so the kernel keeps every load where
    # it was.
    for store in ("alias = y\nalias[pid] = acc", "held = [y]\nheld[0][pid] = acc"):
        out = _run(SWEEPS.replace("STORE", store), ALL_DISJOINT)
        assert "s = cute.arch.load(y.iterator" in out
    # A register fragment or a shared packet's staging is no such container.
    for store in (
        "regs = cute.make_rmem_tensor(1, cutlass.Float32)\nregs[0] = acc",
        (
            "sp = cute.arch.alloc_smem(cutlass.Float32, 2)\n"
            "staged = cute.make_tensor(sp, (2,))\nstaged[0] = acc"
        ),
    ):
        out = _run(SWEEPS.replace("STORE", store), ALL_DISJOINT)
        assert "s = cute.arch.load(y.iterator" not in out


def test_a_pointer_that_reads_memory_keeps_its_load() -> None:
    # The y packet's address goes through idx = m[pid], read after the kernel
    # writes m: the packet keeps its load where it was.
    text = SWEEPS.replace(
        "STORE", "(m.iterator + pid).store(acc)\nidx = (m.iterator + pid).load()"
    ).replace(
        "s = cute.arch.load(y.iterator + pid * 96 + base_2,",
        "s = cute.arch.load(y.iterator + idx * 96 + base_2,",
    )
    out = _run(text, ALL_DISJOINT)
    assert "s = cute.arch.load(y.iterator + idx * 96 + base_2" in out


def test_scalar_load_after_a_continue_stays() -> None:
    text = """
    pid = cutlass.Int32(cute.arch.block_idx()[0])
    for t1 in range(2):
        if skip:
            continue
        w = (y.iterator + pid).load()
        acc = acc + w
    """
    out = _run(text, ALL_DISJOINT)
    assert out.splitlines()[1].startswith("for t1")


def test_scalar_load_after_an_early_return_stays() -> None:
    text = """
    pid = cutlass.Int32(cute.arch.block_idx()[0])
    if pid >= 7:
        return
    w = (y.iterator + pid).load()
    (out.iterator + pid).store(w)
    """
    out = _run(text, ALL_DISJOINT)
    assert out.splitlines()[1].startswith("if pid >= 7")


def test_scalar_load_after_a_nested_loop_else_continue_stays() -> None:
    # The inner loop's ``else`` runs in the outer loop: its ``continue``
    # always skips the load, which must not move above the loop.
    text = """
    pid = cutlass.Int32(cute.arch.block_idx()[0])
    for t in range(2):
        for k in cutlass.range_constexpr(1):
            pass
        else:
            continue
        w = (y.iterator + pid).load()
        acc = acc + w
    """
    out = _run(text, ALL_DISJOINT)
    assert out.splitlines()[1].startswith("for t")


def test_packets_stay_in_a_kernel_with_an_early_return() -> None:
    # A return between the loops: the residual's packet must not load before
    # it.
    text = SWEEPS.replace("STORE", "if pid >= 7:\n    return")
    out = _run(text, ALL_DISJOINT)
    assert "_one_trip_packets_" not in out
