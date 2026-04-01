#ifndef GROUPED_MATMUL_WEIGHT_QUANT_BASIC_BLOCK_KERNEL_KERNEL_H
#define GROUPED_MATMUL_WEIGHT_QUANT_BASIC_BLOCK_KERNEL_KERNEL_H

#include "../utils/integral_constant.h"

namespace Kernel {

template <class ProblemShape_, class BlockMmad_, class BlockEpilogue_, class BlockScheduler_ = void,
          class BlockPrologue_ = void, class Enable = void>
class GroupedMatmul {
    static_assert(AscendC::Std::always_false_v<ProblemShape_>, "GroupedMatmul is not implemented for this combination");
};

} // namespace Kernel

#include "grouped_matmul_mxfp8fp4_kernel_resplit.h"

#endif
