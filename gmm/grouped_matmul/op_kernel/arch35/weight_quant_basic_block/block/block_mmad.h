#ifndef GROUPED_MATMUL_WEIGHT_QUANT_BASIC_BLOCK_BLOCK_MMAD_H
#define GROUPED_MATMUL_WEIGHT_QUANT_BASIC_BLOCK_BLOCK_MMAD_H

#include "../policy/dispatch_policy.h"
#include "grouped_matmul_mxfp8fp4_block_mmad_resplit.h"

namespace Block {

template <class DispatchPolicy, class L1TileShape, class L0TileShape, class AType, class LayoutA, class BType,
          class LayoutB, class CType, class LayoutC, typename = void>
class BlockMmad {
    static_assert(AscendC::Std::always_false_v<DispatchPolicy>, "BlockMmad is not implemented for this DispatchPolicy");
};

template <class L1TileShape, class L0TileShape, class AType, class LayoutA, class BType, class LayoutB, class CType,
          class LayoutC>
class BlockMmad<GROUPED_MATMUL::KernelMixDynamicKL1NTailResplit, L1TileShape, L0TileShape, AType, LayoutA, BType,
                LayoutB, CType, LayoutC, void>
    : public WeightQuantBatchMatmulV2::Arch35::GroupedMatmulMxFp8Fp4BlockMmadResplit<
          AType, BType, DTYPE_ANTIQUANT_SCALE, DTYPE_SCALE, DTYPE_PER_TOKEN_SCALE, DTYPE_BIAS, CType,
          GROUPED_MATMUL::KernelMixDynamicKL1NTailResplit::kWqmmConfig,
          GROUPED_MATMUL::KernelMixDynamicKL1NTailResplit::kVecAntiQuantConfig> {
public:
    using Base = WeightQuantBatchMatmulV2::Arch35::GroupedMatmulMxFp8Fp4BlockMmadResplit<
        AType, BType, DTYPE_ANTIQUANT_SCALE, DTYPE_SCALE, DTYPE_PER_TOKEN_SCALE, DTYPE_BIAS, CType,
        GROUPED_MATMUL::KernelMixDynamicKL1NTailResplit::kWqmmConfig,
        GROUPED_MATMUL::KernelMixDynamicKL1NTailResplit::kVecAntiQuantConfig>;
    using Base::Base;

    using XType = AType;
    using WeightType = BType;
    using YType = CType;
};

} // namespace Block

#endif
