#ifndef GROUPED_MATMUL_WEIGHT_QUANT_BASIC_BLOCK_POLICY_DISPATCH_POLICY_H
#define GROUPED_MATMUL_WEIGHT_QUANT_BASIC_BLOCK_POLICY_DISPATCH_POLICY_H

#include <cstdint>

namespace GROUPED_MATMUL {

struct KernelMixDynamicKL1NTailResplit {
    inline static constexpr uint64_t ubMte2BufferNum = 4;
};

} // namespace GROUPED_MATMUL

#endif
