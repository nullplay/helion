from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import sympy
from torch._inductor import config as inductor_config
from torch._inductor.choices import FusionScore
from torch._inductor.choices import InductorChoices
from torch._inductor.choices import Sortable
from torch._inductor.ir import ComputedBuffer
from torch._inductor.scheduler import FusedMixOrderReductions
from torch._inductor.scheduler import MixOrderReduction
from torch._inductor.virtualized import V
from torch.utils._ordered_set import OrderedSet

from . import config
from .support import MATMUL_DTYPES
from .support import mix_order_split
from .support import names_row_dim
from .support import numerics_modes
from .support import python_wrapper

if TYPE_CHECKING:
    from torch._inductor.codegen.simd_kernel_features import SIMDKernelFeatures
    from torch._inductor.ir import IRNode
    from torch._inductor.scheduler import BaseSchedulerNode
    from torch._inductor.scheduler import Scheduler


# The reductions that keep Triton's rule for a persistent row
# (should_use_persistent_reduction()): a matmul's dot, whose K loop accumulates
# into acc, and those whose loop over tile_r reads each row once, combining as
# it goes, where the row written as one block reduces it twice.
LOOPED_REDUCTIONS = OrderedSet(
    ["dot", "welford_reduce", "welford_combine", "online_softmax_reduce"]
)

# The longest row a kernel reads as one block: a row that no full row of an
# operand names (HelionKernel.codegen_rdim()), read only through a flat or
# gathered load, gets the dim of a factory, which Helion cannot roll, and past
# 2**20 elements, Triton's largest tensor, does not compile.
MAX_PERSISTENT_ROW = 2**20


def static_layout(node: IRNode) -> bool:
    """Whether node's sizes are constants, as HelionKernel's tiles need
    (supports_tiling()), and for a buffer or view, its offset and the strides of
    its dims of more than one element, as its indices do (unsupported()'s
    dynamic_index). An operand that is still a Pointwise, a prologue, has no
    layout yet."""
    exprs = [*node.get_size()]
    if (layout := node.maybe_get_layout()) is not None:
        dims = zip(layout.size, layout.stride, strict=True)
        exprs += [stride for size, stride in dims if size != 1]
        exprs.append(layout.offset)
    return all(isinstance(expr, (int, sympy.Integer)) for expr in exprs)


class HelionChoices(InductorChoices):
    """Inductor's heuristics, except that with cuda_backend "helion" the matmuls a
    Helion kernel emits are lowered to dot reductions. relu(a @ b + c) is then one
    Helion kernel:

        helion_fused_add_mm_relu_0(arg0_1, arg1_1, arg2_1, buf1)

    instead of cuBLAS and a Helion epilogue:

        extern_kernels.mm(arg0_1, arg1_1, out=buf0)
        helion_fused_add_relu_0(reinterpret_tensor(buf1, ...), arg2_1, ...)
    """

    def use_native_matmul(self, mat1: IRNode, mat2: IRNode) -> bool:
        """With cuda_backend "helion", lower mat1 @ mat2 to a dot when a Helion
        kernel emits it; otherwise decide as InductorChoices does.

        A dot that HelionScheduling cannot emit fails the compile, so this
        declines what is known at lowering time not to become a Helion kernel,
        which keeps its extern kernel: e.g. relu(a @ b) with a dynamic dim of a,
        or relu(h[:, -1] @ b) with a dynamic dim 1 of h, whose size s is in the
        stride (64*s) and offset (64*s - 64) of h[:, -1]. What only scheduling
        shows still fails: an operand with an indirect load that needs a bounds
        check, as in a[i] @ b with an i of unknown range, or a fused epilogue
        that Helion does not emit, as in a @ b + torch.tensor(2.0) with the
        scalar on the CPU."""
        device = mat1.get_device_or_error()
        if not (device.type == "cuda" and inductor_config.cuda_backend == "helion"):
            return super().use_native_matmul(mat1, mat2)
        return (
            config.native_matmul
            and python_wrapper()
            and not numerics_modes()
            and mat1.get_dtype() in MATMUL_DTYPES
            and static_layout(mat1)
            and static_layout(mat2)
            # InductorChoices' default declines dims of size 1 too: Inductor's
            # native matmul does not take them yet. The sizes are static here.
            and min(int(n) for n in (*mat1.get_size()[-2:], mat2.get_size()[-1])) > 1
        )

    @staticmethod
    def should_use_persistent_reduction(
        features: SIMDKernelFeatures, cooperative_reduction: bool
    ) -> bool:
        """Whether a reduction kernel reads each row as one block, as
        HelionKernel.should_use_persistent_reduction() asks. With cuda_backend
        "helion" the answer is yes for a kernel of sums, maxes and the like,
        whatever the row's length or stride: the kernel writes the row as one
        block and leaves the choice between reading it whole and a loop over
        chunks of it to Config.reduction_loops, which the autotuner picks.
        x.sum(-1) of a [8192, 262144] x:

            for tile_x in hl.tile(8192):
                tmp0 = in0[tile_x, :]
                tmp1 = torch.sum(tmp0, -1)

        where the loop Triton's rule asks for (an inner reduction of up to 1024
        elements, another of up to 64) fixes the loop, its chunk and the
        reduction of each chunk in the generated code:

            for tile_x in hl.tile(8192):
                acc = hl.zeros([tile_x], dtype=torch.float32)
                for tile_r in hl.tile(262144):
                    tmp0 = in0[tile_x, tile_r]
                    acc = acc + torch.sum(tmp0, -1)

        A kernel with a matmul keeps Triton's rule, which loops over K, and so
        does one with a Welford or online-softmax reduction: its loop reads each
        row once, combining as it goes, where a row written as one block
        computes the mean before the variance (and the max before the sum of
        exps), so a loop over the row reads it twice. So does the first kernel
        of a split reduction, whose chunks of the row run past its end: its
        loads are masked and flat, so no full row of an operand names the row's
        dim, and a row named by a factory's dim cannot be rolled
        (HelionKernel.row_dim()): the chunk of 2047 strided rows of a column
        sum over [1152000, 512] would be read as one block. So does a row
        past MAX_PERSISTENT_ROW. Other devices and backends keep
        Triton's rule, and config.triton.persistent_reductions off still asks
        for loops. The kernel keeps a row it cannot write as one block in a
        loop, whatever this says (names_row_dim()), and so does a row past
        UNROLLED_ROW_MAX that no input's full row names
        (HelionScheduling.create_kernel_choices())."""
        devices = {
            device.type
            for node in features.scheduler_nodes()
            if (device := node.get_device()) is not None
        }
        if (
            devices == {"cuda"}
            and inductor_config.cuda_backend == "helion"
            and inductor_config.triton.persistent_reductions
            and not cooperative_reduction
            and V.graph.sizevars.statically_known_leq(
                features.reduction_numel, MAX_PERSISTENT_ROW
            )
            and not any(
                isinstance(node.node, ComputedBuffer)
                and (
                    node.node.get_reduction_type() in LOOPED_REDUCTIONS
                    or node.node._split_size is not None
                )
                for node in features.scheduler_nodes()
            )
        ):
            return True
        return InductorChoices.should_use_persistent_reduction(
            features, cooperative_reduction
        )

    @staticmethod
    def mix_order_reductions(
        node1: BaseSchedulerNode, node2: BaseSchedulerNode
    ) -> bool:
        """Whether, with cuda_backend "helion", node1 and node2 reduce one
        [rows, cols] array along its two dims, the cols of each row (a row
        reduction) and the rows of each col (a column reduction), which
        HelionScheduling fuses into one kernel that reads the array once
        (codegen_mix_order_reduction()). RMSNorm's backward reads grad and x
        in its row reduction, sum(grad * w * x, -1), and in its weight's
        gradient, sum(grad * x * rstd, 0): unfused, the column reduction reads
        them again. Inductor's check (MixOrderReduction.can_fuse()), which
        it applies to its Triton kernels only: sibling reductions that read a
        common buffer in full, one along contiguous rows of up to 16384, the
        other a sum or product in a float dtype, over enough data that the
        second read would not come from L2 (5 * 2**20 elements, in 4096 rows
        or more), with config.triton.mix_order_reduction on. Helion's
        conditions: a row Helion reads as one block (names_row_dim()) and a
        split of the rows that divides them (mix_order_split()). Neither node
        may already be in a mix-order kernel."""
        device = node1.get_device()
        if not (
            node1.is_reduction()
            and node2.is_reduction()
            and device is not None
            and device.type == "cuda"
            and inductor_config.cuda_backend == "helion"
            and not any(
                isinstance(node.scheduler.get_fused_node(n), FusedMixOrderReductions)
                for node in (node1, node2)
                for n in node.get_nodes()
            )
        ):
            return False
        # Inductor's check, whose gate on the Triton backend
        # (_is_gpu_triton_backend()) is all that keeps it from Helion's
        # kernels.
        with inductor_config.patch(cuda_backend="triton"):
            if not MixOrderReduction.can_fuse(node1, node2):
                return False
        row = node1 if MixOrderReduction.is_contiguous_node(node1) else node2
        rows, cols = MixOrderReduction.get_numel_rnumel(row)
        return (
            isinstance(rows, sympy.Integer)
            and isinstance(cols, sympy.Integer)
            and names_row_dim(cols)
            and mix_order_split(int(rows), device) is not None
        )

    @staticmethod
    def can_fuse(
        scheduler: Scheduler,
        node1: BaseSchedulerNode,
        node2: BaseSchedulerNode,
        shared_data_score: int,
    ) -> bool:
        """InductorChoices.can_fuse(), which declines nodes that share no data
        by its count: a mix-order pair (mix_order_reductions()) reads a common
        buffer along different dims, which the count misses. It counts for the
        pair the elements of its array, as Inductor counts its Triton pairs."""
        if HelionChoices.mix_order_reductions(node1, node2):
            shared_data_score = max(
                shared_data_score, MixOrderReduction.get_fusion_score(node1, node2)
            )
        return InductorChoices.can_fuse(scheduler, node1, node2, shared_data_score)

    @staticmethod
    def can_fuse_horizontal(
        scheduler: Scheduler,
        node1: BaseSchedulerNode,
        node2: BaseSchedulerNode,
        shared_data_score: int,
    ) -> bool:
        """InductorChoices.can_fuse_horizontal(), and a mix-order pair
        (mix_order_reductions()) whatever the data it shares."""
        return HelionChoices.mix_order_reductions(
            node1, node2
        ) or InductorChoices.can_fuse_horizontal(
            scheduler, node1, node2, shared_data_score
        )

    @staticmethod
    def score_fusion(
        scheduler: Scheduler, node1: BaseSchedulerNode, node2: BaseSchedulerNode
    ) -> Sortable:
        """InductorChoices.score_fusion(), which ranks a mix-order pair
        (mix_order_reductions()) by the elements of its array, as Inductor
        ranks its Triton pairs."""
        score = InductorChoices.score_fusion(scheduler, node1, node2)
        if isinstance(score, FusionScore) and HelionChoices.mix_order_reductions(
            node1, node2
        ):
            return dataclasses.replace(
                score,
                template_score=1
                + (node1.is_template() == inductor_config.epilogue_fusion_first),
                node_type_score=1,
                memory_score=MixOrderReduction.get_fusion_score(node1, node2),
            )
        return score

    def uuid(self) -> None:
        # The FX graph cache key already holds what use_native_matmul() reads:
        # Inductor's cuda_backend, and helion.inductor.config, which
        # helion.inductor registers with Inductor.
        return None
