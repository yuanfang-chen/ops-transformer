/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file grouped_matmul_mxfp8fp4_kernel_resplit.h
 * \brief
 */
#ifndef GROUPED_MATMUL_MXFP8FP4_KERNEL_RESPLIT_H
#define GROUPED_MATMUL_MXFP8FP4_KERNEL_RESPLIT_H

#include "../../grouped_matmul_tiling_data_apt.h"
#include "../block/block_mmad.h"
#include "../prologue/block_prologue.h"
#include "include/experimental/tensor_api/tensor.h"

using WeightQuantBatchMatmulV2::Arch35::A_L1_MAX_SIZE_WITH_BIAS_QUANT;
using WeightQuantBatchMatmulV2::Arch35::CeilDivide;
using WeightQuantBatchMatmulV2::Arch35::GetKBUnit;
using GMMWeightQuantParam = GroupedMatmulTilingData::GMMWeightQuantParam;

namespace Kernel {

#define GROUPED_MATMUL_RESPLIT_KERNEL_TEMPLATE_PARAM                                                                    \
    template <class ProblemShape, class L1TileShape, class L0TileShape, class XType, class LayoutA, class WeightType, \
              class LayoutB, class YType, class LayoutC, class BlockEpilogue, class BlockScheduler,                    \
              class AntiQuantScaleType, class ScaleType, class PerTokenScaleType, class BiasType, class Enable>

#define GROUPED_MATMUL_RESPLIT_KERNEL_CLASS                                                                             \
    GroupedMatmul<                                                                                                      \
        ProblemShape,                                                                                                   \
        Block::BlockMmad<GROUPED_MATMUL::KernelMixDynamicKL1NTailResplit, L1TileShape, L0TileShape, XType, LayoutA,   \
                         WeightType, LayoutB, YType, LayoutC>,                                                          \
        BlockEpilogue, BlockScheduler,                                                                                  \
        Block::BlockPrologue<GROUPED_MATMUL::KernelMixDynamicKL1NTailResplit, XType, WeightType, AntiQuantScaleType,  \
                             ScaleType, PerTokenScaleType, BiasType, YType>,                                           \
        Enable>

GROUPED_MATMUL_RESPLIT_KERNEL_TEMPLATE_PARAM
class GROUPED_MATMUL_RESPLIT_KERNEL_CLASS {
public:
    using BlockMmad = Block::BlockMmad<GROUPED_MATMUL::KernelMixDynamicKL1NTailResplit, L1TileShape, L0TileShape,
                                       XType, LayoutA, WeightType, LayoutB, YType, LayoutC>;
    using BlockPrologue = Block::BlockPrologue<GROUPED_MATMUL::KernelMixDynamicKL1NTailResplit, XType, WeightType,
                                               AntiQuantScaleType, ScaleType, PerTokenScaleType, BiasType, YType>;

    __aicore__ inline GroupedMatmul() = delete;
    __aicore__ inline GroupedMatmul(GM_ADDR x, GM_ADDR weight, GM_ADDR antiquantScale, GM_ADDR bias, GM_ADDR groupList,
                                    GM_ADDR perTokenScale, GM_ADDR y,
                                    const GMMWeightQuantParam *__restrict baseTiling,
                                    const TCubeTiling *__restrict mmTiling);
    __aicore__ inline void operator()();

    struct Params {
        ProblemShape problemShape;
        typename BlockMmad::Params mmad;
        typename BlockPrologue::Params prologue;
        uint64_t groupType;
        uint64_t groupListType;
    };
private:
    __aicore__ inline uint64_t GetSplitValueFromGroupList(uint64_t groupIdx);
    __aicore__ inline void CalcDynamicKBlock(uint64_t mL1Size, uint64_t nL1Size, uint64_t &kaL1Size,
                                             uint64_t &kbL1Size) const;
    __aicore__ inline void UpdateGmAddr(uint64_t mSize, uint64_t kSize, uint64_t nSize);
    __aicore__ inline void PrefetchA(uint64_t mSize, uint64_t kSize);

    const GMMWeightQuantParam *gmmBaseTiling_;
    const TCubeTiling *mmTiling_;

    __gm__ XType *xGm_;
    __gm__ WeightType *weightGm_;
    __gm__ AntiQuantScaleType *antiquantScaleGm_;
    __gm__ BiasType *biasGm_ = nullptr;
    __gm__ YType *yGm_;
    __gm__ PerTokenScaleType *perTokenScaleGm_;
    GlobalTensor<int64_t> groupListGm_;
    BlockMmad blockMmad_;
    BlockPrologue blockPrologue_;

    uint64_t preOffset_ = 0;
    static constexpr uint64_t MX_A8W4_L1_K_CONFIG_256 = 256;
    static constexpr uint64_t MX_A8W4_L1_K_CONFIG_512 = 512;
    static constexpr uint64_t MX_A8W4_L1_K_DYNAMIC_CONFIG_N_THRESHOLD = 128;
    static constexpr uint64_t MX_A8W4_L1_K_DYNAMIC_CONFIG_M_THRESHOLD_256 = 256;
    static constexpr uint64_t MX_A8W4_L1_K_DYNAMIC_CONFIG_M_THRESHOLD_240 = 240;
    uint64_t mxA8W4L1KDynamicConfigMThreshold_;
};

GROUPED_MATMUL_RESPLIT_KERNEL_TEMPLATE_PARAM
__aicore__ inline GROUPED_MATMUL_RESPLIT_KERNEL_CLASS::GroupedMatmul(
    GM_ADDR x, GM_ADDR weight, GM_ADDR antiquantScale, GM_ADDR bias, GM_ADDR groupList, GM_ADDR perTokenScale,
    GM_ADDR y, const GMMWeightQuantParam *__restrict baseTiling,
    const TCubeTiling *__restrict mmTiling)
    : blockMmad_(baseTiling->hasBias, 0, mmTiling), blockPrologue_(baseTiling->hasBias, 0, mmTiling)
{
    gmmBaseTiling_ = baseTiling;
    mmTiling_ = mmTiling;

    if (groupList != nullptr) {
        groupListGm_.SetGlobalBuffer((__gm__ int64_t *)groupList);
    }

}

GROUPED_MATMUL_RESPLIT_KERNEL_TEMPLATE_PARAM
__aicore__ inline void GROUPED_MATMUL_RESPLIT_KERNEL_CLASS::operator()(const Params& params)
{
    using TensorLayoutA = typename AscendC::Te::NDLayoutFormat<XType>;
    using TensorLayoutC = typename AscendC::Te::NDLayoutFormat<YType>;
    using TensorLayoutScaleA = typename AscendC::Te::ScaleANDLayoutFormat<fp8_e8m0_t>;
    using TensorLayoutScaleB = typename AscendC::Te::ScaleBDNLayoutFormat<fp8_e8m0_t>;

    if ASCEND_IS_AIV {
        weightGm_ = GROUPED_MATMUL::GetTensorAddr<BlocPrologue::wType>(0, params.blockPrologue.ptrB);
        if (params.blockPrologue.hasBias) {
            biasGm_ = GROUPED_MATMUL::GetTensorAddr<BlocPrologue::biasType>(0, params.blockPrologue.ptrBias);
        }
    }
    if ASCEND_IS_AIC {
        xGm_ = GROUPED_MATMUL::GetTensorAddr<BlockMmad::xType>(0, params.blockMmad.ptrA);
        perTokenScaleGm_ = reinterpret_cast<__gm__ BlockMmad::perTokenScaleType *>(params.blockMmad.ptrScaleA);
        antiquantScaleGm_ = GROUPED_MATMUL::GetTensorAddr<BlockMmad::antiQuantScaleType>(0, params.blockMmad.ptrScaleB);
        yGm_ = GROUPED_MATMUL::GetTensorAddr<<BlockMmad::YType>(0, params.blockMmad.ptrC);
    }

    // TODO 移动到block/prologue层，由block/prologue层动态计算kaL1,kbL1大小
    mxA8W4L1KDynamicConfigMThreshold_ = gmmBaseTiling_->hasBias ? MX_A8W4_L1_K_DYNAMIC_CONFIG_M_THRESHOLD_240 :
                                                                  MX_A8W4_L1_K_DYNAMIC_CONFIG_M_THRESHOLD_256;

    uint32_t cubeBlockIdx = GetBlockIdx();
    if ASCEND_IS_AIV {
        cubeBlockIdx = cubeBlockIdx >> 1;
    }

    const uint64_t kSize = params.problemShape.kSize;
    const uint64_t nSize = params.problemShape.nSize;
    const uint64_t nAlign = AscendC::CeilAlign(nSize, static_cast<uint64_t>(BLOCK_CUBE));
    const bool isCacheLineUnaligned = kSize % 256 != 0;
    const uint64_t scaleKSize = CeilDivide(kSize, static_cast<uint64_t>(64)) * 2;
    BlockScheduler scheduler(params.scheduler);
    for (uint32_t groupIdx = 0, startBasicBlockId = 0; groupIdx < params.problemShape.groupNum; ++groupIdx) {
        uint64_t mSize = GetSplitValueFromGroupList(groupIdx);
        if (mSize > 0 && nSize > 0) {
            auto tensorAGm =
                AscendC::Te::MakeTensor(AscendC::Te::MakeGMmemPtr(xGm_), TensorLayoutA{}(mSize, kSize));
            auto tensorYGm =
                AscendC::Te::MakeTensor(AscendC::Te::MakeGMmemPtr(yGm_), TensorLayoutC{}(mSize, nSize));
            auto tensorScaleAGm =
                AscendC::Te::MakeTensor(AscendC::Te::MakeGMmemPtr(perTokenScaleGm_), TensorLayoutScaleA{}(mSize, scaleKSize));
            auto tensorScaleBGm = AscendC::Te::MakeTensor(AscendC::Te::MakeGMmemPtr(antiquantScaleGm_),
                                                          TensorLayoutScaleB{}(scaleKSize, nSize));

            decltype(tensorAGm) tensorBlockAGm;
            decltype(tensorScaleAGm) tensorBlockScaleAGm;
            auto callbackM = [&](uint64_t mOffset, uint64_t mL1Size) {
                if ASCEND_IS_AIC {
                    tensorBlockAGm = tensorAGm(AscendC::Te::MakeCoord(mOffset, 0), AscendC::Te::MakeShape(mL1Size, kSize));
                    tensorBlockScaleAGm = tensorScaleAGm(AscendC::Te::MakeCoord(mOffset, 0), AscendC::Te::MakeShape(mL1Size, scaleKSize));
                }
            };
            
            auto callbackMN = [&](uint64_t mOffset, uint64_t mL1Size, uint64_t nOffset, uint64_t nL1Size) {
                if ASCEND_IS_AIC {
                    auto tensorBlockYGm = tensorYGm(AscendC::Te::MakeCoord(mOffset, nOffset), AscendC::Te::MakeShape(mL1Size, nL1Size));
                    auto tensorBlockScaleBGm = tensorScaleBGm(AscendC::Te::MakeCoord(0, nOffset), AscendC::Te::MakeShape(scaleKSize, nL1Size));
                    blockMmad_(tensorBlockAGm, tensorBlockYGm, tensorBlockScaleAGm, tensorBlockScaleBGm);
                } else {
                    blockPrologue_(weightGm_, biasGm_, kSize, nL1Size, nOffset, nAlign);
                }
            };

            scheduler(startBasicBlockId, mSize, callbackM, callbackMN);

            xGm_ += mSize * kSize;
            // 4bit，地址偏移单位为8bit
            weightGm_ += (nSize * kSize) >> 1;
            antiquantScaleGm_ += nSize * scaleKSize;
            perTokenScaleGm_ += mSize * scaleKSize;
            if (gmmBaseTiling_->hasBias) {
                biasGm_ += nSize;
            }
            yGm_ += mSize * nSize;
        }
    }

    if ASCEND_IS_AIC {
        blockMmad_.End();
    } else {
        blockPrologue_.End();
    }
}

GROUPED_MATMUL_RESPLIT_KERNEL_TEMPLATE_PARAM
__aicore__ inline void GROUPED_MATMUL_RESPLIT_KERNEL_CLASS::UpdateGmAddr(
    uint64_t mSize, uint64_t kSize, uint64_t nSize)
{
    xGm_ += mSize * kSize;
    // 4bit，地址偏移单位为8bit
    weightGm_ += (nSize * kSize) >> 1;
    antiquantScaleGm_ += nSize * CeilDivide(kSize, static_cast<uint64_t>(gmmBaseTiling_->groupSize));
    perTokenScaleGm_ += mSize * CeilDivide(kSize, static_cast<uint64_t>(gmmBaseTiling_->groupSize));
    if (gmmBaseTiling_->hasBias) {
        biasGm_ += nSize;
    }
    yGm_ += mSize * nSize;
}

GROUPED_MATMUL_RESPLIT_KERNEL_TEMPLATE_PARAM
__aicore__ inline void GROUPED_MATMUL_RESPLIT_KERNEL_CLASS::PrefetchA(
    uint64_t mSize, uint64_t kSize)
{
    if ASCEND_IS_AIV {
        return;
    }
    if (params.scheduler.mainBlockCount == 0 &&
        params.scheduler.firstTailBlockCount + params.scheduler.secondTailBlockCount <
            params.scheduler.cubeNumBlocksN) {
        return;
    }
    uint64_t aSize = mSize * kSize * sizeof(XType);
    if (mSize <= 512 && aSize <= static_cast<uint64_t>(params.scheduler.cubeNumBlocksN) * A_L1_MAX_SIZE_WITH_BIAS_QUANT &&
        (params.scheduler.coreNum % params.scheduler.cubeNumBlocksN == 0)) {
        uint64_t aPrefetchSize =
            AscendC::CeilAlign(CeilDivide(mSize * kSize, static_cast<uint64_t>(params.scheduler.cubeNumBlocksN)), 64UL);
        blockMmad_.PrefetchA(aPrefetchSize, mSize * kSize);
    }
}

GROUPED_MATMUL_RESPLIT_KERNEL_TEMPLATE_PARAM
__aicore__ inline uint64_t GROUPED_MATMUL_RESPLIT_KERNEL_CLASS::GetSplitValueFromGroupList(
    uint64_t groupIdx)
{
    uint64_t splitValue = 0;
    if (likely(params.groupType != -1)) {
        if (params.groupListType == 0) {
            uint64_t offset = static_cast<uint64_t>(groupListGm_.GetValue(groupIdx));
            splitValue = offset - preOffset_;
            preOffset_ = offset;
        } else {
            splitValue = static_cast<uint64_t>(groupListGm_.GetValue(groupIdx));
        }
    }
    return splitValue;
}

#undef GROUPED_MATMUL_RESPLIT_KERNEL_CLASS
#undef GROUPED_MATMUL_RESPLIT_KERNEL_TEMPLATE_PARAM
} // namespace Kernel

#endif  // GROUPED_MATMUL_MXFP8FP4_KERNEL_RESPLIT_H
