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
#include "../block/grouped_matmul_scheduler_n_resplit.h"
#include "../prologue/block_prologue.h"
#include "include/experimental/tensor_api/tensor.h"

using WeightQuantBatchMatmulV2::Arch35::A_L1_MAX_SIZE_WITH_BIAS_QUANT;
using WeightQuantBatchMatmulV2::Arch35::CeilDivide;
using WeightQuantBatchMatmulV2::Arch35::GetKBUnit;
using GMMWeightQuantParam = GroupedMatmulTilingData::GMMWeightQuantParam;

namespace Kernel {

#define GROUPED_MATMUL_RESPLIT_KERNEL_TEMPLATE_PARAM                                                                    \
    template <class ProblemShape, class L1TileShape, class L0TileShape, class XType, class LayoutA, class WeightType, \
              class LayoutB, class YType, class LayoutC, class BlockEpilogue, class BlockScheduler, class BiasType,   \
              class Enable>

#define GROUPED_MATMUL_RESPLIT_KERNEL_CLASS                                                                             \
    GroupedMatmul<                                                                                                      \
        ProblemShape,                                                                                                   \
        Block::BlockMmad<GROUPED_MATMUL::KernelMixDynamicKL1NTailResplit, L1TileShape, L0TileShape, XType, LayoutA,   \
                         WeightType, LayoutB, YType, LayoutC>,                                                          \
        BlockEpilogue, BlockScheduler,                                                                                  \
        Block::BlockPrologue<GROUPED_MATMUL::KernelMixDynamicKL1NTailResplit, XType, WeightType, BiasType>,           \
        Enable>

GROUPED_MATMUL_RESPLIT_KERNEL_TEMPLATE_PARAM
class GROUPED_MATMUL_RESPLIT_KERNEL_CLASS {
public:
    using KernelImpl = GROUPED_MATMUL_RESPLIT_KERNEL_CLASS;
    using BlockMmad = Block::BlockMmad<GROUPED_MATMUL::KernelMixDynamicKL1NTailResplit, L1TileShape, L0TileShape,
                                       XType, LayoutA, WeightType, LayoutB, YType, LayoutC>;
    using BlockPrologue =
        Block::BlockPrologue<GROUPED_MATMUL::KernelMixDynamicKL1NTailResplit, XType, WeightType, BiasType>;
    using AntiQuantScaleType = typename BlockMmad::antiQuantScaleType;
    using PerTokenScaleType = typename BlockMmad::perTokenScaleType;

    struct Params {
        GM_ADDR x;
        GM_ADDR weight;
        GM_ADDR antiquantScale;
        GM_ADDR bias;
        GM_ADDR groupList;
        GM_ADDR perTokenScale;
        GM_ADDR y;
        const GMMWeightQuantParam *__restrict baseTiling;
        const TCubeTiling *__restrict mmTiling;
    };

    __aicore__ inline GroupedMatmul() = default;
    __aicore__ inline void operator()(const Params &params);
private:
    template <typename TensorA, typename TensorScaleA, typename TensorY, typename TensorScaleB>
    struct AicScheduleHandler {
        BlockMmad &blockMmad;
        const GMMWeightQuantParam &gmmBaseTiling;
        const TensorA &tensorAGm;
        const TensorScaleA &tensorScaleAGm;
        const TensorY &tensorYGm;
        const TensorScaleB &tensorScaleBGm;
        TensorA &tensorBlockAGm;
        TensorScaleA &tensorBlockScaleAGm;
        uint64_t kSize;
        uint64_t scaleKSize;

        __aicore__ inline void OnM(uint64_t mOffset, uint64_t mL1Size) const
        {
            tensorBlockAGm = tensorAGm(AscendC::Te::MakeCoord(mOffset, 0), AscendC::Te::MakeShape(mL1Size, kSize));
            tensorBlockScaleAGm =
                tensorScaleAGm(AscendC::Te::MakeCoord(mOffset, 0), AscendC::Te::MakeShape(mL1Size, scaleKSize));
            if (gmmBaseTiling.mainBlockCount == 0 &&
                gmmBaseTiling.firstTailBlockCount + gmmBaseTiling.secondTailBlockCount < gmmBaseTiling.cubeNumBlocksN) {
                return;
            }
            uint64_t aSize = mL1Size * kSize * sizeof(XType);
            if (mL1Size <= 512 &&
                aSize <= static_cast<uint64_t>(gmmBaseTiling.cubeNumBlocksN) * A_L1_MAX_SIZE_WITH_BIAS_QUANT &&
                (gmmBaseTiling.coreNum % gmmBaseTiling.cubeNumBlocksN == 0)) {
                uint64_t aPrefetchSize = AscendC::CeilAlign(
                    CeilDivide(mL1Size * kSize, static_cast<uint64_t>(gmmBaseTiling.cubeNumBlocksN)), 64UL);
                blockMmad.PrefetchA(aPrefetchSize, mL1Size * kSize);
            }
        }

        __aicore__ inline void OnMN(uint64_t mOffset, uint64_t mL1Size, uint64_t nOffset, uint64_t nL1Size) const
        {
            auto tensorBlockYGm =
                tensorYGm(AscendC::Te::MakeCoord(mOffset, nOffset), AscendC::Te::MakeShape(mL1Size, nL1Size));
            auto tensorBlockScaleBGm =
                tensorScaleBGm(AscendC::Te::MakeCoord(0, nOffset), AscendC::Te::MakeShape(scaleKSize, nL1Size));
            blockMmad(tensorBlockAGm, tensorBlockYGm, tensorBlockScaleAGm, tensorBlockScaleBGm);
        }
    };

    struct AivScheduleHandler {
        BlockPrologue &blockPrologue;
        __gm__ WeightType *weightGm;
        __gm__ BiasType *biasGm;
        bool &weightL2Cacheable;
        uint64_t mSize;
        uint64_t kSize;
        uint64_t nAlign;
        bool isCacheLineUnaligned;

        __aicore__ inline void OnM(uint64_t, uint64_t mL1Size) const
        {
            weightL2Cacheable = mL1Size < mSize || isCacheLineUnaligned;
        }

        __aicore__ inline void OnMN(uint64_t, uint64_t mL1Size, uint64_t nOffset, uint64_t nL1Size) const
        {
            blockPrologue(weightGm, biasGm, weightL2Cacheable, mL1Size, kSize, nL1Size, nOffset, nAlign);
        }
    };

    __aicore__ inline uint64_t GetSplitValueFromGroupList(uint64_t groupIdx);
    __aicore__ inline void UpdateGmAddr(uint64_t mSize, uint64_t kSize, uint64_t nSize);

    const GMMWeightQuantParam *gmmBaseTiling_;
    const TCubeTiling *mmTiling_;

    __gm__ XType *xGm_;
    __gm__ WeightType *weightGm_;
    __gm__ AntiQuantScaleType *antiquantScaleGm_;
    __gm__ BiasType *biasGm_ = nullptr;
    __gm__ YType *yGm_;
    __gm__ PerTokenScaleType *perTokenScaleGm_;
    GlobalTensor<int64_t> groupListGm_;

    uint64_t preOffset_ = 0;
};

GROUPED_MATMUL_RESPLIT_KERNEL_TEMPLATE_PARAM
__aicore__ inline void GROUPED_MATMUL_RESPLIT_KERNEL_CLASS::operator()(const Params &params)
{
    gmmBaseTiling_ = params.baseTiling;
    mmTiling_ = params.mmTiling;
    preOffset_ = 0;

    xGm_ = GROUPED_MATMUL::GetTensorAddr<XType>(0, params.x);
    weightGm_ = GROUPED_MATMUL::GetTensorAddr<WeightType>(0, params.weight);
    antiquantScaleGm_ = GROUPED_MATMUL::GetTensorAddr<AntiQuantScaleType>(0, params.antiquantScale);
    biasGm_ = nullptr;
    if (gmmBaseTiling_->hasBias) {
        biasGm_ = GROUPED_MATMUL::GetTensorAddr<BiasType>(0, params.bias);
    }
    perTokenScaleGm_ = reinterpret_cast<__gm__ PerTokenScaleType *>(params.perTokenScale);
    yGm_ = GROUPED_MATMUL::GetTensorAddr<YType>(0, params.y);
    if (params.groupList != nullptr) {
        groupListGm_.SetGlobalBuffer((__gm__ int64_t *)params.groupList);
    }

    using TensorLayoutA = typename AscendC::Te::NDLayoutFormat<XType>;
    using TensorLayoutC = typename AscendC::Te::NDLayoutFormat<YType>;
    using TensorLayoutScaleA = typename AscendC::Te::ScaleANDLayoutFormat<fp8_e8m0_t>;
    using TensorLayoutScaleB = typename AscendC::Te::ScaleBDNLayoutFormat<fp8_e8m0_t>;

    const uint64_t kSize = gmmBaseTiling_->kSize;
    const uint64_t nSize = gmmBaseTiling_->nSize;
    const uint64_t nAlign = AscendC::CeilAlign(nSize, static_cast<uint64_t>(BLOCK_CUBE));
    const bool isCacheLineUnaligned = kSize % 256 != 0;
    const uint64_t scaleKSize = CeilDivide(kSize, static_cast<uint64_t>(64)) * 2;
    typename BlockScheduler::Params schedulerParams = {
        gmmBaseTiling_->mainBlockCount,     gmmBaseTiling_->mainBlockSize,      gmmBaseTiling_->firstTailBlockCount,
        gmmBaseTiling_->firstTailBlockSize, gmmBaseTiling_->secondTailBlockCount,
        gmmBaseTiling_->secondTailBlockSize, gmmBaseTiling_->coreNum,           gmmBaseTiling_->cubeNumBlocksN,
        static_cast<uint64_t>(mmTiling_->baseM), nSize};
    BlockScheduler scheduler(schedulerParams);
    if ASCEND_IS_AIC {
        BlockMmad blockMmad(gmmBaseTiling_->hasBias, 0, mmTiling_);
        uint64_t startBasicBlockId = 0;
        for (uint32_t groupIdx = 0; groupIdx < gmmBaseTiling_->groupNum; ++groupIdx) {
            uint64_t mSize = GetSplitValueFromGroupList(groupIdx);
            if (mSize > 0 && nSize > 0) {
                auto tensorAGm =
                    AscendC::Te::MakeTensor(AscendC::Te::MakeGMmemPtr(xGm_), TensorLayoutA{}(mSize, kSize));
                auto tensorYGm =
                    AscendC::Te::MakeTensor(AscendC::Te::MakeGMmemPtr(yGm_), TensorLayoutC{}(mSize, nSize));
                auto tensorScaleAGm = AscendC::Te::MakeTensor(AscendC::Te::MakeGMmemPtr(perTokenScaleGm_),
                                                              TensorLayoutScaleA{}(mSize, scaleKSize));
                auto tensorScaleBGm = AscendC::Te::MakeTensor(AscendC::Te::MakeGMmemPtr(antiquantScaleGm_),
                                                              TensorLayoutScaleB{}(scaleKSize, nSize));
                decltype(tensorAGm) tensorBlockAGm;
                decltype(tensorScaleAGm) tensorBlockScaleAGm;
                AicScheduleHandler<decltype(tensorAGm), decltype(tensorScaleAGm), decltype(tensorYGm),
                                   decltype(tensorScaleBGm)>
                    handler = {
                        blockMmad,
                        *gmmBaseTiling_,
                        tensorAGm,
                        tensorScaleAGm,
                        tensorYGm,
                        tensorScaleBGm,
                        tensorBlockAGm,
                        tensorBlockScaleAGm,
                        kSize,
                        scaleKSize};
                scheduler(startBasicBlockId, mSize, handler);
            }
            UpdateGmAddr(mSize, kSize, nSize);
        }
        blockMmad.End();
    } else {
        BlockPrologue blockPrologue(gmmBaseTiling_->hasBias, 0, mmTiling_);
        uint64_t startBasicBlockId = 0;
        for (uint32_t groupIdx = 0; groupIdx < gmmBaseTiling_->groupNum; ++groupIdx) {
            uint64_t mSize = GetSplitValueFromGroupList(groupIdx);
            if (mSize > 0 && nSize > 0) {
                bool weightL2Cacheable = isCacheLineUnaligned;
                AivScheduleHandler handler = {
                    blockPrologue,
                    weightGm_,
                    biasGm_,
                    weightL2Cacheable,
                    mSize,
                    kSize,
                    nAlign,
                    isCacheLineUnaligned};
                scheduler(startBasicBlockId, mSize, handler);
            }
            UpdateGmAddr(mSize, kSize, nSize);
        }
        blockPrologue.End();
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
