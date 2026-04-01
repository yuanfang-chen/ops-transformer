#ifndef GROUPED_MATMUL_WEIGHT_QUANT_BASIC_BLOCK_POLICY_DISPATCH_POLICY_H
#define GROUPED_MATMUL_WEIGHT_QUANT_BASIC_BLOCK_POLICY_DISPATCH_POLICY_H

#include "../block/basic_block_config.h"

namespace GROUPED_MATMUL {

struct KernelMixDynamicKL1NTailResplit {
    inline static constexpr WeightQuantBatchMatmulV2::Arch35::WqmmConfig kWqmmConfig = {
        false, true, CubeFormat::NZ};
    inline static constexpr WeightQuantBatchMatmulV2::Arch35::VecAntiQuantConfig kVecAntiQuantConfig = {4};
};

} // namespace GROUPED_MATMUL

#endif
