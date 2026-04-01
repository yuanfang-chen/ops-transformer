#ifndef GROUPED_MATMUL_WEIGHT_QUANT_BASIC_BLOCK_PROLOGUE_BLOCK_PROLOGUE_H
#define GROUPED_MATMUL_WEIGHT_QUANT_BASIC_BLOCK_PROLOGUE_BLOCK_PROLOGUE_H

#include "../policy/dispatch_policy.h"
#include "grouped_matmul_mxfp8fp4_prologue_mx_cast_w.h"

namespace Block {

template <class DispatchPolicy, class... Args>
class BlockPrologue {
    static_assert(AscendC::Std::always_false_v<DispatchPolicy>, "BlockPrologue is not implemented for this DispatchPolicy");
};

template <class XType, class WeightType, class AntiQuantScaleType, class ScaleType, class PerTokenScaleType,
          class BiasType, class YType>
class BlockPrologue<GROUPED_MATMUL::KernelMixDynamicKL1NTailResplit, XType, WeightType, AntiQuantScaleType, ScaleType,
                    PerTokenScaleType, BiasType, YType>
    : public WeightQuantBatchMatmulV2::Arch35::WeightQuantMatmulBasicBlockAiv<
          XType, WeightType, AntiQuantScaleType, ScaleType, PerTokenScaleType, BiasType, YType,
          GROUPED_MATMUL::KernelMixDynamicKL1NTailResplit::kWqmmConfig,
          GROUPED_MATMUL::KernelMixDynamicKL1NTailResplit::kVecAntiQuantConfig> {
public:
    using Base = WeightQuantBatchMatmulV2::Arch35::WeightQuantMatmulBasicBlockAiv<
        XType, WeightType, AntiQuantScaleType, ScaleType, PerTokenScaleType, BiasType, YType,
        GROUPED_MATMUL::KernelMixDynamicKL1NTailResplit::kWqmmConfig,
        GROUPED_MATMUL::KernelMixDynamicKL1NTailResplit::kVecAntiQuantConfig>;
    using Base::Base;

    using XDataType = XType;
    using WeightDataType = WeightType;
    using AntiQuantScaleDataType = AntiQuantScaleType;
    using ScaleDataType = ScaleType;
    using PerTokenScaleDataType = PerTokenScaleType;
    using BiasDataType = BiasType;
    using YDataType = YType;
};

} // namespace Block

#endif
