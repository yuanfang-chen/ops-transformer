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
 * \file grouped_matmul_weight_quant_resplit_controller.h
 * \brief
 */
#ifndef GROUPED_MATMUL_WEIGHT_QUANT_RESPLIT_CONTROLLER_H
#define GROUPED_MATMUL_WEIGHT_QUANT_RESPLIT_CONTROLLER_H

#include "../grouped_matmul_tiling_data_apt.h"
#include "include/experimental/tensor_api/tensor.h"

using WeightQuantBatchMatmulV2::Arch35::A_L1_MAX_SIZE_WITH_BIAS_QUANT;
using WeightQuantBatchMatmulV2::Arch35::BasicBlockControlParam;
using WeightQuantBatchMatmulV2::Arch35::BasicBlockOffsetParam;
using WeightQuantBatchMatmulV2::Arch35::CeilDivide;
using WeightQuantBatchMatmulV2::Arch35::VecAntiQuantConfig;
using WeightQuantBatchMatmulV2::Arch35::WqmmConfig;
using WeightQuantBatchMatmulV2::Arch35::GetKBUnit;
using GMMWeightQuantParam = GroupedMatmulTilingData::GMMWeightQuantParam;

namespace GROUPED_MATMUL {
template <typename xType, typename wType, typename antiQuantScaleType, typename scaleType, typename perTokenScaleType,
          typename biasType, typename yType,
          template <typename, typename, typename, typename, typename, typename, typename, const WqmmConfig &,
                    const VecAntiQuantConfig &>
          class BasicBlock,
          const WqmmConfig &wqmmConfig, const VecAntiQuantConfig &vecConfig>
class GMMWeightQuantResplitController {
public:
    __aicore__ inline GMMWeightQuantResplitController(){};
    __aicore__ inline GMMWeightQuantResplitController(GM_ADDR x, GM_ADDR weight, GM_ADDR scale, GM_ADDR antiquantScale,
                                                      GM_ADDR antiquantOffset, GM_ADDR bias, GM_ADDR groupList,
                                                      GM_ADDR perTokenScale, GM_ADDR y,
                                                      const GMMWeightQuantParam *__restrict baseTiling,
                                                      const TCubeTiling *__restrict mmTiling, GM_ADDR tiling,
                                                      TPipe *tPipe);
    __aicore__ inline void operator()();

private:
    __aicore__ inline void InitOffsetParam(BasicBlockOffsetParam &offsetParam);
    __aicore__ inline void RunBlockRange(BasicBlockOffsetParam &offsetParam, BasicBlockControlParam &ctrlParam,
                                         uint64_t basicBlockCount, uint64_t basicBlockSize);
    __aicore__ inline void SplitNByMultiCore(BasicBlockOffsetParam &offsetParam, BasicBlockControlParam &ctrlParam,
                                             uint64_t basicBlockCount, uint64_t basicBlockSize);
    __aicore__ inline uint64_t GetSplitValueFromGroupList(uint64_t groupIdx);
    __aicore__ inline void UpdateGmAddr(uint64_t mSize, uint64_t kSize, uint64_t nSize);
    __aicore__ inline void PrefetchA(uint64_t mSize, uint64_t kSize);

    const GMMWeightQuantParam *gmmBaseTiling_;
    const TCubeTiling *mmTiling_;

    __gm__ xType *xGm_;
    __gm__ wType *weightGm_;
    __gm__ antiQuantScaleType *antiquantScaleGm_;
    __gm__ xType *antiquantOffsetGm_;
    __gm__ biasType *biasGm_;
    __gm__ yType *yGm_;
    __gm__ perTokenScaleType *perTokenScaleGm_;
    __gm__ scaleType *scaleGm_;
    GlobalTensor<int64_t> groupListGm_;
    BasicBlock<xType, wType, antiQuantScaleType, scaleType, perTokenScaleType, biasType, yType, wqmmConfig, vecConfig>
        basicBlock_;

    uint64_t preOffset_ = 0;
    static constexpr uint64_t MX_A8W4_L1_K_CONFIG_256 = 256;
    static constexpr uint64_t MX_A8W4_L1_K_CONFIG_512 = 512;
    static constexpr uint64_t MX_A8W4_L1_K_DYNAMIC_CONFIG_N_THRESHOLD = 128;
    static constexpr uint64_t MX_A8W4_L1_K_DYNAMIC_CONFIG_M_THRESHOLD_256 = 256;
    static constexpr uint64_t MX_A8W4_L1_K_DYNAMIC_CONFIG_M_THRESHOLD_240 = 240;
    uint64_t mxA8W4L1KDynamicConfigMThreshold_; // m轴依赖空间划分，无法静态配置
};

template <typename xType, typename wType, typename antiQuantScaleType, typename scaleType, typename perTokenScaleType,
          typename biasType, typename yType,
          template <typename, typename, typename, typename, typename, typename, typename, const WqmmConfig &,
                    const VecAntiQuantConfig &>
          class BasicBlock,
          const WqmmConfig &wqmmConfig, const VecAntiQuantConfig &vecConfig>
__aicore__ inline GMMWeightQuantResplitController<xType, wType, antiQuantScaleType, scaleType, perTokenScaleType,
                                                  biasType, yType, BasicBlock, wqmmConfig, vecConfig>::
    GMMWeightQuantResplitController(GM_ADDR x, GM_ADDR weight, GM_ADDR scale, GM_ADDR antiquantScale,
                                    GM_ADDR antiquantOffset, GM_ADDR bias, GM_ADDR groupList,
                                    GM_ADDR perTokenScale, GM_ADDR y,
                                    const GMMWeightQuantParam *__restrict baseTiling,
                                    const TCubeTiling *__restrict mmTiling, GM_ADDR tiling, TPipe *tPipe)
    : basicBlock_(baseTiling->hasBias, 0, mmTiling, tPipe)
{
    (void)tiling;
    gmmBaseTiling_ = baseTiling;

    mmTiling_ = mmTiling;

    xGm_ = GetTensorAddr<xType>(0, x);
    weightGm_ = GetTensorAddr<wType>(0, weight);
    antiquantScaleGm_ = GetTensorAddr<antiQuantScaleType>(0, antiquantScale);
    antiquantOffsetGm_ = GetTensorAddr<xType>(0, antiquantOffset);
    biasGm_ = GetTensorAddr<biasType>(0, bias);
    scaleGm_ = GetTensorAddr<scaleType>(0, scale);
    perTokenScaleGm_ = reinterpret_cast<__gm__ perTokenScaleType *>(perTokenScale);
    yGm_ = GetTensorAddr<yType>(0, y);
    if (groupList != nullptr) {
        groupListGm_.SetGlobalBuffer((__gm__ int64_t *)groupList);
    }
    mxA8W4L1KDynamicConfigMThreshold_ = gmmBaseTiling_->hasBias ? MX_A8W4_L1_K_DYNAMIC_CONFIG_M_THRESHOLD_240 :
                                                                  MX_A8W4_L1_K_DYNAMIC_CONFIG_M_THRESHOLD_256;
}

template <typename xType, typename wType, typename antiQuantScaleType, typename scaleType, typename perTokenScaleType,
          typename biasType, typename yType,
          template <typename, typename, typename, typename, typename, typename, typename, const WqmmConfig &,
                    const VecAntiQuantConfig &>
          class BasicBlock,
          const WqmmConfig &wqmmConfig, const VecAntiQuantConfig &vecConfig>
__aicore__ inline void GMMWeightQuantResplitController<xType, wType, antiQuantScaleType, scaleType, perTokenScaleType,
                                                       biasType, yType, BasicBlock, wqmmConfig, vecConfig>::operator()()
{
    uint32_t cubeBlockIdx = GetBlockIdx();
    if ASCEND_IS_AIV {
        cubeBlockIdx = cubeBlockIdx >> 1;
    }

    BasicBlockOffsetParam offsetParam;
    InitOffsetParam(offsetParam);

    bool isCacheLineUnaligned = offsetParam.kSize % 128 != 0;  // 缓存大小128B，对应8bit为128个元素
    if constexpr (IsSameType<wType, int4b_t>::value || IsSameType<wType, fp4x2_e2m1_t>::value ||
                  IsSameType<wType, fp4x2_e1m2_t>::value) {
        isCacheLineUnaligned = offsetParam.kSize % 256 != 0;  // 缓存大小128B，对应4bit为256个元素
    }

    BasicBlockControlParam ctrlParam;
    for (uint32_t groupIdx = 0, startBasicBlockId = 0; groupIdx < gmmBaseTiling_->groupNum; ++groupIdx) {
        ctrlParam.mSize = GetSplitValueFromGroupList(groupIdx);
        if (ctrlParam.mSize > 0 && offsetParam.nSize > 0) {
            uint64_t mBlkNum = CeilDivide(ctrlParam.mSize, static_cast<uint64_t>(mmTiling_->baseM));
            ctrlParam.mL1Size = CeilDivide(ctrlParam.mSize, mBlkNum);
            basicBlock_.UpdateGlobalAddr(ctrlParam.mSize, offsetParam.kSize, offsetParam.nSize, xGm_, weightGm_, antiquantScaleGm_, antiquantOffsetGm_, scaleGm_,
                                         perTokenScaleGm_, biasGm_, yGm_, mmTiling_->isBias,
                                         ctrlParam.mL1Size < ctrlParam.mSize || isCacheLineUnaligned);
            PrefetchA(ctrlParam.mSize, offsetParam.kSize);
            ctrlParam.curBasicBlockId =
                cubeBlockIdx >= startBasicBlockId ? cubeBlockIdx : cubeBlockIdx + gmmBaseTiling_->coreNum;
            ctrlParam.basicBlockLimit = startBasicBlockId;
            for (ctrlParam.mOffset = 0; ctrlParam.mOffset < ctrlParam.mSize; ctrlParam.mOffset += ctrlParam.mL1Size) {
                ctrlParam.nOffset = 0;
                RunBlockRange(offsetParam, ctrlParam, gmmBaseTiling_->mainBlockCount, gmmBaseTiling_->mainBlockSize);
                RunBlockRange(offsetParam, ctrlParam, gmmBaseTiling_->firstTailBlockCount,
                              gmmBaseTiling_->firstTailBlockSize);
                RunBlockRange(offsetParam, ctrlParam, gmmBaseTiling_->secondTailBlockCount,
                              gmmBaseTiling_->secondTailBlockSize);
            }
            startBasicBlockId = ctrlParam.basicBlockLimit % gmmBaseTiling_->coreNum;
        }
        UpdateGmAddr(ctrlParam.mSize, offsetParam.kSize, offsetParam.nSize);
    }

    basicBlock_.End();
}

template <typename xType, typename wType, typename antiQuantScaleType, typename scaleType, typename perTokenScaleType,
          typename biasType, typename yType,
          template <typename, typename, typename, typename, typename, typename, typename, const WqmmConfig &,
                    const VecAntiQuantConfig &>
          class BasicBlock,
          const WqmmConfig &wqmmConfig, const VecAntiQuantConfig &vecConfig>
__aicore__ inline void GMMWeightQuantResplitController<xType, wType, antiQuantScaleType, scaleType, perTokenScaleType,
                                                       biasType, yType, BasicBlock, wqmmConfig, vecConfig>::InitOffsetParam(
    BasicBlockOffsetParam &offsetParam)
{
    const uint64_t kbL1Size = mmTiling_->baseK * mmTiling_->stepKb;
    const uint64_t kAlign = CeilAlign(gmmBaseTiling_->kSize, static_cast<uint64_t>(BLOCK_CUBE));
    const uint64_t nAlign = CeilAlign(gmmBaseTiling_->nSize, static_cast<uint64_t>(BLOCK_CUBE));
    offsetParam.kbL1Size = kbL1Size;
    offsetParam.kaL1Size = kbL1Size;  // 当前实现a矩阵切分保持b矩阵一致
    offsetParam.kSize = gmmBaseTiling_->kSize;
    offsetParam.nSize = gmmBaseTiling_->nSize;
    offsetParam.kAlign = kAlign;
    offsetParam.nAlign = nAlign;
}

template <typename xType, typename wType, typename antiQuantScaleType, typename scaleType, typename perTokenScaleType,
          typename biasType, typename yType,
          template <typename, typename, typename, typename, typename, typename, typename, const WqmmConfig &,
                    const VecAntiQuantConfig &>
          class BasicBlock,
          const WqmmConfig &wqmmConfig, const VecAntiQuantConfig &vecConfig>
__aicore__ inline void GMMWeightQuantResplitController<xType, wType, antiQuantScaleType, scaleType, perTokenScaleType,
                                                       biasType, yType, BasicBlock, wqmmConfig, vecConfig>::RunBlockRange(
    BasicBlockOffsetParam &offsetParam, BasicBlockControlParam &ctrlParam, uint64_t basicBlockCount,
    uint64_t basicBlockSize)
{
    SplitNByMultiCore(offsetParam, ctrlParam, basicBlockCount, basicBlockSize);
    ctrlParam.basicBlockLimit += basicBlockCount;
}

template <typename xType, typename wType, typename antiQuantScaleType, typename scaleType, typename perTokenScaleType,
          typename biasType, typename yType,
          template <typename, typename, typename, typename, typename, typename, typename, const WqmmConfig &,
                    const VecAntiQuantConfig &>
          class BasicBlock,
          const WqmmConfig &wqmmConfig, const VecAntiQuantConfig &vecConfig>
__aicore__ inline void GMMWeightQuantResplitController<xType, wType, antiQuantScaleType, scaleType, perTokenScaleType,
                                                       biasType, yType, BasicBlock, wqmmConfig, vecConfig>::SplitNByMultiCore(
    BasicBlockOffsetParam &offsetParam, BasicBlockControlParam &ctrlParam, uint64_t basicBlockCount,
    uint64_t basicBlockSize)
{
    for (; ctrlParam.curBasicBlockId < ctrlParam.basicBlockLimit + basicBlockCount;
         ctrlParam.curBasicBlockId += gmmBaseTiling_->coreNum) {
        offsetParam.mSize = ctrlParam.mSize;
        offsetParam.mOffset = ctrlParam.mOffset;
        offsetParam.mL1Size = ctrlParam.mOffset + ctrlParam.mL1Size > ctrlParam.mSize ? ctrlParam.mSize - ctrlParam.mOffset
                                                                                       : ctrlParam.mL1Size;
        offsetParam.nOffset =
            ctrlParam.nOffset +
            ((ctrlParam.curBasicBlockId - ctrlParam.basicBlockLimit) % basicBlockCount) * basicBlockSize;
        offsetParam.nL1Size =
            offsetParam.nOffset + basicBlockSize > gmmBaseTiling_->nSize ? gmmBaseTiling_->nSize - offsetParam.nOffset
                                                                         : basicBlockSize;
        offsetParam.yGmAddr = reinterpret_cast<GM_ADDR>(yGm_);

        // MxA8W4场景，切换低阶api，支持kernel内动态调整k轴切分
        offsetParam.kbL1Size =
            (offsetParam.mL1Size <= mxA8W4L1KDynamicConfigMThreshold_ &&
                offsetParam.nL1Size <= MX_A8W4_L1_K_DYNAMIC_CONFIG_N_THRESHOLD) ?
                MX_A8W4_L1_K_CONFIG_512 :
                MX_A8W4_L1_K_CONFIG_256;
        if (offsetParam.mL1Size < offsetParam.nL1Size) {
            // L1的空间，在有Bias时，预留124 Kb, 其他场景预留128 Kb
            uint64_t aL1Size = gmmBaseTiling_->hasBias ? 124 * GetKBUnit<xType>() : 128 * GetKBUnit<xType>();
            uint64_t mL1Align = CeilAlign(offsetParam.mL1Size, BLOCK_CUBE);
            // 当前切分mL1比nL1小的场景，可以尝试L1上多倍载入kaL1,提升A矩阵载入效率 
            offsetParam.kaL1Size = aL1Size / (mL1Align * offsetParam.kbL1Size) * offsetParam.kbL1Size;
        } else {
            offsetParam.kaL1Size = offsetParam.kbL1Size;
        }
        basicBlock_.ComputeBasicBlock(offsetParam);
    }
    ctrlParam.nOffset += basicBlockSize * basicBlockCount;
}

template <typename xType, typename wType, typename antiQuantScaleType, typename scaleType, typename perTokenScaleType,
          typename biasType, typename yType,
          template <typename, typename, typename, typename, typename, typename, typename, const WqmmConfig &,
                    const VecAntiQuantConfig &>
          class BasicBlock,
          const WqmmConfig &wqmmConfig, const VecAntiQuantConfig &vecConfig>
__aicore__ inline void GMMWeightQuantResplitController<xType, wType, antiQuantScaleType, scaleType, perTokenScaleType,
                                                       biasType, yType, BasicBlock, wqmmConfig, vecConfig>::UpdateGmAddr(
    uint64_t mSize, uint64_t kSize, uint64_t nSize)
{
    xGm_ += mSize * kSize;
    if constexpr (IsSameType<wType, int4b_t>::value || IsSameType<wType, fp4x2_e2m1_t>::value ||
                  IsSameType<wType, fp4x2_e1m2_t>::value) {
        weightGm_ += (nSize * kSize) >> 1;
    } else {
        weightGm_ += nSize * kSize;
    }

    antiquantScaleGm_ += nSize * CeilDivide(kSize, static_cast<uint64_t>(gmmBaseTiling_->groupSize));
    antiquantOffsetGm_ += nSize * CeilDivide(kSize, static_cast<uint64_t>(gmmBaseTiling_->groupSize));

    scaleGm_ += nSize;

    perTokenScaleGm_ += mSize * CeilDivide(kSize, static_cast<uint64_t>(gmmBaseTiling_->groupSize));

    biasGm_ += nSize;
    yGm_ += mSize * nSize;
}

template <typename xType, typename wType, typename antiQuantScaleType, typename scaleType, typename perTokenScaleType,
          typename biasType, typename yType,
          template <typename, typename, typename, typename, typename, typename, typename, const WqmmConfig &,
                    const VecAntiQuantConfig &>
          class BasicBlock,
          const WqmmConfig &wqmmConfig, const VecAntiQuantConfig &vecConfig>
__aicore__ inline void GMMWeightQuantResplitController<xType, wType, antiQuantScaleType, scaleType, perTokenScaleType,
                                                       biasType, yType, BasicBlock, wqmmConfig, vecConfig>::PrefetchA(
    uint64_t mSize, uint64_t kSize)
{
    if ASCEND_IS_AIV {
        return;
    }
    // mxA8W4场景，Dn2nz严重阻塞流水，在weight较小的时候不启用prefetch策略
    if (gmmBaseTiling_->mainBlockCount == 0 &&
        gmmBaseTiling_->firstTailBlockCount + gmmBaseTiling_->secondTailBlockCount <
            gmmBaseTiling_->cubeNumBlocksN) {
        return;
    }
    uint64_t aSize = mSize * kSize * sizeof(xType);

    /*
     * 准入条件：
     * 1. m <= 512
     * 2. A的大小在cubeNumBlocksN上可被一条mte2指令均分载入
     * 3. 核数是N分核数的倍数
     */
    if (mSize <= 512 && aSize <= static_cast<uint64_t>(gmmBaseTiling_->cubeNumBlocksN) * A_L1_MAX_SIZE_WITH_BIAS_QUANT &&
        (gmmBaseTiling_->coreNum % gmmBaseTiling_->cubeNumBlocksN == 0)) {
        uint64_t aPrefetchSize =
            CeilAlign(CeilDivide(mSize * kSize, static_cast<uint64_t>(gmmBaseTiling_->cubeNumBlocksN)),
                      64UL); // 64 表示128B的cacheline对齐
        basicBlock_.PrefetchA(aPrefetchSize, mSize * kSize);
    }
}

template <typename xType, typename wType, typename antiQuantScaleType, typename scaleType, typename perTokenScaleType,
          typename biasType, typename yType,
          template <typename, typename, typename, typename, typename, typename, typename, const WqmmConfig &,
                    const VecAntiQuantConfig &>
          class BasicBlock,
          const WqmmConfig &wqmmConfig, const VecAntiQuantConfig &vecConfig>
__aicore__ inline uint64_t GMMWeightQuantResplitController<xType, wType, antiQuantScaleType, scaleType,
                                                           perTokenScaleType, biasType, yType, BasicBlock,
                                                           wqmmConfig, vecConfig>::GetSplitValueFromGroupList(
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

}  // namespace GROUPED_MATMUL

#endif  // GROUPED_MATMUL_WEIGHT_QUANT_RESPLIT_CONTROLLER_H
