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

private:
    __aicore__ inline uint64_t GetSplitValueFromGroupList(uint64_t groupIdx);
    __aicore__ inline void CalcDynamicKBlock(uint64_t mL1Size, uint64_t nL1Size, uint64_t &kaL1Size,
                                             uint64_t &kbL1Size) const;
    template <typename TensorA, typename TensorY, typename TensorScaleA, typename TensorScaleB>
    __aicore__ inline void RunBlockRange(const TensorA &tensorBlockAGm, const TensorY &tensorYGm,
                                         const TensorScaleA &tensorBlockScaleAGm, const TensorScaleB &tensorScaleBGm,
                                         uint64_t mOffset, uint64_t mL1Size, uint64_t kSize, uint64_t scaleKSize,
                                         uint64_t nAlign, bool weightL2Cacheable, uint64_t blockCount,
                                         uint64_t blockSize, uint64_t nBaseOffset, uint64_t basicBlockLimit,
                                         uint64_t &curBasicBlockId);
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

    xGm_ = GROUPED_MATMUL::GetTensorAddr<XType>(0, x);
    weightGm_ = GROUPED_MATMUL::GetTensorAddr<WeightType>(0, weight);
    antiquantScaleGm_ = GROUPED_MATMUL::GetTensorAddr<AntiQuantScaleType>(0, antiquantScale);
    if (gmmBaseTiling_->hasBias) {
        biasGm_ = GROUPED_MATMUL::GetTensorAddr<BiasType>(0, bias);
    }
    perTokenScaleGm_ = reinterpret_cast<__gm__ PerTokenScaleType *>(perTokenScale);
    yGm_ = GROUPED_MATMUL::GetTensorAddr<YType>(0, y);
    if (groupList != nullptr) {
        groupListGm_.SetGlobalBuffer((__gm__ int64_t *)groupList);
    }
    mxA8W4L1KDynamicConfigMThreshold_ = gmmBaseTiling_->hasBias ? MX_A8W4_L1_K_DYNAMIC_CONFIG_M_THRESHOLD_240 :
                                                                  MX_A8W4_L1_K_DYNAMIC_CONFIG_M_THRESHOLD_256;
}

GROUPED_MATMUL_RESPLIT_KERNEL_TEMPLATE_PARAM
__aicore__ inline void GROUPED_MATMUL_RESPLIT_KERNEL_CLASS::operator()()
{
    using TensorLayoutA = typename AscendC::Te::NDLayoutFormat<XType>;
    using TensorLayoutC = typename AscendC::Te::NDLayoutFormat<YType>;
    using TensorLayoutScaleA = typename AscendC::Te::ScaleANDLayoutFormat<fp8_e8m0_t>;
    using TensorLayoutScaleB = typename AscendC::Te::ScaleBDNLayoutFormat<fp8_e8m0_t>;

    uint32_t cubeBlockIdx = GetBlockIdx();
    if ASCEND_IS_AIV {
        cubeBlockIdx = cubeBlockIdx >> 1;
    }

    const uint64_t kSize = gmmBaseTiling_->kSize;
    const uint64_t nSize = gmmBaseTiling_->nSize;
    const uint64_t nAlign = AscendC::CeilAlign(nSize, static_cast<uint64_t>(BLOCK_CUBE));
    const bool isCacheLineUnaligned = kSize % 256 != 0;
    const uint64_t scaleKSize = CeilDivide(kSize, static_cast<uint64_t>(64)) * 2;
    for (uint32_t groupIdx = 0, startBasicBlockId = 0; groupIdx < gmmBaseTiling_->groupNum; ++groupIdx) {
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

            uint64_t mBlkNum = CeilDivide(mSize, static_cast<uint64_t>(mmTiling_->baseM));
            uint64_t mL1Step = CeilDivide(mSize, mBlkNum);
            bool weightL2Cacheable = mL1Step < mSize || isCacheLineUnaligned;
            uint64_t curBasicBlockId =
                cubeBlockIdx >= startBasicBlockId ? cubeBlockIdx : cubeBlockIdx + gmmBaseTiling_->coreNum;
            uint64_t basicBlockLimit = startBasicBlockId;
            for (uint64_t mOffset = 0; mOffset < mSize; mOffset += mL1Step) {
                uint64_t mL1Size = mOffset + mL1Step > mSize ? mSize - mOffset : mL1Step;
                auto tensorBlockAGm =
                    tensorAGm(AscendC::Te::MakeCoord(mOffset, 0), AscendC::Te::MakeShape(mL1Size, kSize));
                auto tensorBlockScaleAGm = tensorScaleAGm(AscendC::Te::MakeCoord(mOffset, 0),
                                                          AscendC::Te::MakeShape(mL1Size, scaleKSize));
                uint64_t nBaseOffset = 0;
                RunBlockRange(tensorBlockAGm, tensorYGm, tensorBlockScaleAGm, tensorScaleBGm, mOffset, mL1Size, kSize,
                              scaleKSize, nAlign, weightL2Cacheable, gmmBaseTiling_->mainBlockCount,
                              gmmBaseTiling_->mainBlockSize, nBaseOffset, basicBlockLimit, curBasicBlockId);
                nBaseOffset += gmmBaseTiling_->mainBlockSize * gmmBaseTiling_->mainBlockCount;
                basicBlockLimit += gmmBaseTiling_->mainBlockCount;
                RunBlockRange(tensorBlockAGm, tensorYGm, tensorBlockScaleAGm, tensorScaleBGm, mOffset, mL1Size, kSize,
                              scaleKSize, nAlign, weightL2Cacheable, gmmBaseTiling_->firstTailBlockCount,
                              gmmBaseTiling_->firstTailBlockSize, nBaseOffset, basicBlockLimit, curBasicBlockId);
                nBaseOffset += gmmBaseTiling_->firstTailBlockSize * gmmBaseTiling_->firstTailBlockCount;
                basicBlockLimit += gmmBaseTiling_->firstTailBlockCount;
                RunBlockRange(tensorBlockAGm, tensorYGm, tensorBlockScaleAGm, tensorScaleBGm, mOffset, mL1Size, kSize,
                              scaleKSize, nAlign, weightL2Cacheable, gmmBaseTiling_->secondTailBlockCount,
                              gmmBaseTiling_->secondTailBlockSize, nBaseOffset, basicBlockLimit, curBasicBlockId);
                nBaseOffset += gmmBaseTiling_->secondTailBlockSize * gmmBaseTiling_->secondTailBlockCount;
                basicBlockLimit += gmmBaseTiling_->secondTailBlockCount;
            }
            startBasicBlockId = basicBlockLimit % gmmBaseTiling_->coreNum;
        }
        UpdateGmAddr(mSize, kSize, nSize);
    }

    if ASCEND_IS_AIC {
        blockMmad_.End();
    } else {
        blockPrologue_.End();
    }
}

GROUPED_MATMUL_RESPLIT_KERNEL_TEMPLATE_PARAM
__aicore__ inline void GROUPED_MATMUL_RESPLIT_KERNEL_CLASS::CalcDynamicKBlock(
    uint64_t mL1Size, uint64_t nL1Size, uint64_t &kaL1Size, uint64_t &kbL1Size) const
{
    kbL1Size = mmTiling_->baseK * mmTiling_->stepKb;
    kbL1Size = (mL1Size <= mxA8W4L1KDynamicConfigMThreshold_ &&
                nL1Size <= MX_A8W4_L1_K_DYNAMIC_CONFIG_N_THRESHOLD) ?
                   MX_A8W4_L1_K_CONFIG_512 :
                   MX_A8W4_L1_K_CONFIG_256;
    if (mL1Size < nL1Size) {
        uint64_t aL1Size = gmmBaseTiling_->hasBias ? 124 * GetKBUnit<XType>() : 128 * GetKBUnit<XType>();
        uint64_t mL1Align = AscendC::CeilAlign(mL1Size, static_cast<uint64_t>(BLOCK_CUBE));
        kaL1Size = aL1Size / (mL1Align * kbL1Size) * kbL1Size;
    } else {
        kaL1Size = kbL1Size;
    }
}

GROUPED_MATMUL_RESPLIT_KERNEL_TEMPLATE_PARAM
template <typename TensorA, typename TensorY, typename TensorScaleA, typename TensorScaleB>
__aicore__ inline void GROUPED_MATMUL_RESPLIT_KERNEL_CLASS::RunBlockRange(
    const TensorA &tensorBlockAGm, const TensorY &tensorYGm, const TensorScaleA &tensorBlockScaleAGm,
    const TensorScaleB &tensorScaleBGm, uint64_t mOffset, uint64_t mL1Size, uint64_t kSize, uint64_t scaleKSize,
    uint64_t nAlign, bool weightL2Cacheable, uint64_t blockCount, uint64_t blockSize, uint64_t nBaseOffset,
    uint64_t basicBlockLimit, uint64_t &curBasicBlockId)
{
    if (blockCount == 0) {
        return;
    }
    for (; curBasicBlockId < basicBlockLimit + blockCount; curBasicBlockId += gmmBaseTiling_->coreNum) {
        uint64_t nOffset =
            nBaseOffset + ((curBasicBlockId - basicBlockLimit) % blockCount) * blockSize;
        uint64_t nL1Size = nOffset + blockSize > gmmBaseTiling_->nSize ? gmmBaseTiling_->nSize - nOffset : blockSize;
        uint64_t kaL1Size = 0;
        uint64_t kbL1Size = 0;
        CalcDynamicKBlock(mL1Size, nL1Size, kaL1Size, kbL1Size);
        auto tensorBlockYGm = tensorYGm(AscendC::Te::MakeCoord(mOffset, nOffset),
                                        AscendC::Te::MakeShape(mL1Size, nL1Size));
        auto tensorBlockScaleBGm = tensorScaleBGm(AscendC::Te::MakeCoord(0, nOffset),
                                                  AscendC::Te::MakeShape(scaleKSize, nL1Size));
        if ASCEND_IS_AIC {
            blockMmad_(tensorBlockAGm, tensorBlockYGm, tensorBlockScaleAGm, tensorBlockScaleBGm, kaL1Size,
                           kbL1Size);
        } else {
            blockPrologue_(weightGm_, biasGm_, weightL2Cacheable, kSize, nL1Size, nOffset, kbL1Size, nAlign);
        }
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
    if (gmmBaseTiling_->mainBlockCount == 0 &&
        gmmBaseTiling_->firstTailBlockCount + gmmBaseTiling_->secondTailBlockCount <
            gmmBaseTiling_->cubeNumBlocksN) {
        return;
    }
    uint64_t aSize = mSize * kSize * sizeof(XType);
    if (mSize <= 512 && aSize <= static_cast<uint64_t>(gmmBaseTiling_->cubeNumBlocksN) * A_L1_MAX_SIZE_WITH_BIAS_QUANT &&
        (gmmBaseTiling_->coreNum % gmmBaseTiling_->cubeNumBlocksN == 0)) {
        uint64_t aPrefetchSize =
            AscendC::CeilAlign(CeilDivide(mSize * kSize, static_cast<uint64_t>(gmmBaseTiling_->cubeNumBlocksN)), 64UL);
        blockMmad_.PrefetchA(aPrefetchSize, mSize * kSize);
    }
}

GROUPED_MATMUL_RESPLIT_KERNEL_TEMPLATE_PARAM
__aicore__ inline uint64_t GROUPED_MATMUL_RESPLIT_KERNEL_CLASS::GetSplitValueFromGroupList(
    uint64_t groupIdx)
{
    uint64_t splitValue = 0;
    if (likely(gmmBaseTiling_->groupType != -1)) {
        if (gmmBaseTiling_->groupListType == 0) {
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
