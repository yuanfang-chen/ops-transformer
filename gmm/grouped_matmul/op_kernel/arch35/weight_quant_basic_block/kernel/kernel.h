#ifndef GROUPED_MATMUL_WEIGHT_QUANT_BASIC_BLOCK_KERNEL_KERNEL_H
#define GROUPED_MATMUL_WEIGHT_QUANT_BASIC_BLOCK_KERNEL_KERNEL_H

namespace Kernel {

template <class ProblemShape_, class BlockMmad_, class BlockEpilogue_, class BlockScheduler_ = void,
          class BlockPrologue_ = void, class Enable = void>
class GroupedMatmul;

} // namespace Kernel

#endif
