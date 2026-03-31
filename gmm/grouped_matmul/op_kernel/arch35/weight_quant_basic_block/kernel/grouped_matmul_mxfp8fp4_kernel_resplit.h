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
#include "include/experimental/tensor_api/tensor.h"

using WeightQuantBatchMatmulV2::Arch35::A_L1_MAX_SIZE_WITH_BIAS_QUANT;
using WeightQuantBatchMatmulV2::Arch35::BasicBlockOffsetParam;
using WeightQuantBatchMatmulV2::Arch35::CeilDivide;
using WeightQuantBatchMatmulV2::Arch35::VecAntiQuantConfig;
using WeightQuantBatchMatmulV2::Arch35::WqmmConfig;
using WeightQuantBatchMatmulV2::Arch35::GetKBUnit;
using GMMWeightQuantParam = GroupedMatmulTilingData::GMMWeightQuantParam;

namespace GROUPED_MATMUL {

struct BasicBlockControlParam {
    uint64_t mSize;
    uint64_t mL1Size;
    uint64_t curBasicBlockId;
    uint64_t basicBlockLimit;
    uint64_t mOffset;
    uint64_t nOffset;
};

template <typename xType, typename wType, typename antiQuantScaleType, typename scaleType, typename perTokenScaleType,
          typename biasType, typename yType,
          template <typename, typename, typename, typename, typename, typename, typename, const WqmmConfig &,
                    const VecAntiQuantConfig &>
          class AicBasicBlock,
          template <typename, typename, typename, typename, typename, typename, typename, const WqmmConfig &,
                    const VecAntiQuantConfig &>
          class AivBasicBlock,
          const WqmmConfig &wqmmConfig, const VecAntiQuantConfig &vecConfig>
class GMMWeightQuantResplitController {
public:
    __aicore__ inline GMMWeightQuantResplitController() = delete;
    __aicore__ inline GMMWeightQuantResplitController(GM_ADDR x, GM_ADDR weight, GM_ADDR scale, GM_ADDR antiquantScale,
                                                      GM_ADDR antiquantOffset, GM_ADDR bias, GM_ADDR groupList,
                                                      GM_ADDR perTokenScale, GM_ADDR y,
                                                      const GMMWeightQuantParam *__restrict baseTiling,
                                                      const TCubeTiling *__restrict mmTiling, GM_ADDR tiling);
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
    __gm__ biasType *biasGm_;
    __gm__ yType *yGm_;
    __gm__ perTokenScaleType *perTokenScaleGm_;
    GlobalTensor<int64_t> groupListGm_;
    AicBasicBlock<xType, wType, antiQuantScaleType, scaleType, perTokenScaleType, biasType, yType, wqmmConfig, vecConfig>
        aicBasicBlock_;
    AivBasicBlock<xType, wType, antiQuantScaleType, scaleType, perTokenScaleType, biasType, yType, wqmmConfig, vecConfig>
        aivBasicBlock_;

    uint64_t preOffset_ = 0;
    static constexpr uint64_t MX_A8W4_L1_K_CONFIG_256 = 256;
    static constexpr uint64_t MX_A8W4_L1_K_CONFIG_512 = 512;
    static constexpr uint64_t MX_A8W4_L1_K_DYNAMIC_CONFIG_N_THRESHOLD = 128;
    static constexpr uint64_t MX_A8W4_L1_K_DYNAMIC_CONFIG_M_THRESHOLD_256 = 256;
    static constexpr uint64_t MX_A8W4_L1_K_DYNAMIC_CONFIG_M_THRESHOLD_240 = 240;
    uint64_t mxA8W4L1KDynamicConfigMThreshold_;
};

template <typename xType, typename wType, typename antiQuantScaleType, typename scaleType, typename perTokenScaleType,
          typename biasType, typename yType,
          template <typename, typename, typename, typename, typename, typename, typename, const WqmmConfig &,
                    const VecAntiQuantConfig &>
          class AicBasicBlock,
          template <typename, typename, typename, typename, typename, typename, typename, const WqmmConfig &,
                    const VecAntiQuantConfig &>
          class AivBasicBlock,
          const WqmmConfig &wqmmConfig, const VecAntiQuantConfig &vecConfig>
__aicore__ inline GMMWeightQuantResplitController<xType, wType, antiQuantScaleType, scaleType, perTokenScaleType,
                                                  biasType, yType, AicBasicBlock, AivBasicBlock, wqmmConfig,
                                                  vecConfig>::GMMWeightQuantResplitController(
    GM_ADDR x, GM_ADDR weight, GM_ADDR scale, GM_ADDR antiquantScale, GM_ADDR antiquantOffset, GM_ADDR bias,
    GM_ADDR groupList, GM_ADDR perTokenScale, GM_ADDR y, const GMMWeightQuantParam *__restrict baseTiling,
    const TCubeTiling *__restrict mmTiling, GM_ADDR tiling)
    : aicBasicBlock_(baseTiling->hasBias, 0, mmTiling), aivBasicBlock_(baseTiling->hasBias, 0, mmTiling)
{
    (void)tiling;
    gmmBaseTiling_ = baseTiling;
    mmTiling_ = mmTiling;

    xGm_ = GetTensorAddr<xType>(0, x);
    weightGm_ = GetTensorAddr<wType>(0, weight);
    antiquantScaleGm_ = GetTensorAddr<antiQuantScaleType>(0, antiquantScale);
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
          class AicBasicBlock,
          template <typename, typename, typename, typename, typename, typename, typename, const WqmmConfig &,
                    const VecAntiQuantConfig &>
          class AivBasicBlock,
          const WqmmConfig &wqmmConfig, const VecAntiQuantConfig &vecConfig>
__aicore__ inline void GMMWeightQuantResplitController<xType, wType, antiQuantScaleType, scaleType, perTokenScaleType,
                                                       biasType, yType, AicBasicBlock, AivBasicBlock, wqmmConfig,
                                                       vecConfig>::operator()()
{
    uint32_t cubeBlockIdx = GetBlockIdx();
    if ASCEND_IS_AIV {
        cubeBlockIdx = cubeBlockIdx >> 1;
    }

    BasicBlockOffsetParam offsetParam;
    InitOffsetParam(offsetParam);

    // 缓存大小128B，对应4bit为256个元素
    bool isCacheLineUnaligned = offsetParam.kSize % 256 != 0;
    uint64_t scaleKSize = Cgmct::Gemm::CeilDiv(kSize, 64) * 2;

    BasicBlockControlParam ctrlParam;
    for (uint32_t groupIdx = 0, startBasicBlockId = 0; groupIdx < gmmBaseTiling_->groupNum; ++groupIdx) {
        ctrlParam.mSize = GetSplitValueFromGroupList(groupIdx);
        if (ctrlParam.mSize > 0 && offsetParam.nSize > 0) {
            auto tensorAGm = AscendC::Te::MakeTensor(AscendC::Te::MakeGMemPtr(xGm_), AscendC::Te::MakeNDLayout(mSize, kSize));
            auto tensorYGm = AscendC::Te::MakeTensor(AscendC::Te::MakeGMemPtr(yGm_), AscendC::Te::MakeNDLayout(mSize, nSize));
            auto tensorScaleAGm = AscendC::Te::MakeTensor(AscendC::Te::MakeGMemPtr(perTokenScaleGm_), AscendC::Te::MakeScaleANDLayout<fp8_e8m0_t>{}(mSize, scaleKSize));
            auto tensorScaleBGm = AscendC::Te::MakeTensor(AscendC::Te::MakeGMemPtr(antiquantScaleGm_), AscendC::Te::MakeScaleBDNLayout<fp8_e8m0_t>{}(scaleKSize, nSize));

            uint64_t mBlkNum = CeilDivide(ctrlParam.mSize, static_cast<uint64_t>(mmTiling_->baseM));
            ctrlParam.mL1Size = CeilDivide(ctrlParam.mSize, mBlkNum);

            bool weightL2Cacheable = ctrlParam.mL1Size < ctrlParam.mSize || isCacheLineUnaligned;
            ctrlParam.curBasicBlockId =
                cubeBlockIdx >= startBasicBlockId ? cubeBlockIdx : cubeBlockIdx + gmmBaseTiling_->coreNum;
            ctrlParam.basicBlockLimit = startBasicBlockId;
            for (ctrlParam.mOffset = 0; ctrlParam.mOffset < ctrlParam.mSize; ctrlParam.mOffset += ctrlParam.mL1Size) {
                ctrlParam.nOffset = 0;

                offsetParam.mSize = ctrlParam.mSize;
                offsetParam.mL1Size = ctrlParam.mOffset + ctrlParam.mL1Size > ctrlParam.mSize ? ctrlParam.mSize - ctrlParam.mOffset :
                                                                                            ctrlParam.mL1Size;
                auto tensorBlockAGm = tensorAGm(AscendC::Te::MakeCoord(mOffset, 0), AscendC::Te::MakeShape(mL1Size, kSize));
                auto tensorBlockScaleAGm = tensorScaleAGm(AscendC::Te::MakeCoord(mOffset, 0), AscendC::Te::MakeShape(mL1Size, scaleKSize));

                SplitNByMultiCore(offsetParam, ctrlParam, gmmBaseTiling_->mainBlockCount, gmmBaseTiling_->mainBlockSize);
                ctrlParam.basicBlockLimit += gmmBaseTiling_->mainBlockCount;
                SplitNByMultiCore(offsetParam, ctrlParam, gmmBaseTiling_->firstTailBlockCount,
                              gmmBaseTiling_->firstTailBlockSize);
                ctrlParam.basicBlockLimit += gmmBaseTiling_->firstTailBlockCount;
                SplitNByMultiCore(offsetParam, ctrlParam, gmmBaseTiling_->secondTailBlockCount,
                              gmmBaseTiling_->secondTailBlockSize);
                ctrlParam.basicBlockLimit += gmmBaseTiling_->secondTailBlockCount;
            }
            startBasicBlockId = ctrlParam.basicBlockLimit % gmmBaseTiling_->coreNum;
        }
        UpdateGmAddr(ctrlParam.mSize, offsetParam.kSize, offsetParam.nSize);
    }

    if ASCEND_IS_AIC {
        aicBasicBlock_.End();
    } else {
        aivBasicBlock_.End();
    }
}

template <typename xType, typename wType, typename antiQuantScaleType, typename scaleType, typename perTokenScaleType,
          typename biasType, typename yType,
          template <typename, typename, typename, typename, typename, typename, typename, const WqmmConfig &,
                    const VecAntiQuantConfig &>
          class AicBasicBlock,
          template <typename, typename, typename, typename, typename, typename, typename, const WqmmConfig &,
                    const VecAntiQuantConfig &>
          class AivBasicBlock,
          const WqmmConfig &wqmmConfig, const VecAntiQuantConfig &vecConfig>
__aicore__ inline void GMMWeightQuantResplitController<xType, wType, antiQuantScaleType, scaleType, perTokenScaleType,
                                                       biasType, yType, AicBasicBlock, AivBasicBlock, wqmmConfig,
                                                       vecConfig>::InitOffsetParam(BasicBlockOffsetParam &offsetParam)
{
    const uint64_t kbL1Size = mmTiling_->baseK * mmTiling_->stepKb;
    const uint64_t nAlign = CeilAlign(gmmBaseTiling_->nSize, static_cast<uint64_t>(BLOCK_CUBE));
    offsetParam.kbL1Size = kbL1Size;
    offsetParam.kSize = gmmBaseTiling_->kSize;
    offsetParam.nSize = gmmBaseTiling_->nSize;
    offsetParam.nAlign = nAlign;
}

template <typename xType, typename wType, typename antiQuantScaleType, typename scaleType, typename perTokenScaleType,
          typename biasType, typename yType,
          template <typename, typename, typename, typename, typename, typename, typename, const WqmmConfig &,
                    const VecAntiQuantConfig &>
          class AicBasicBlock,
          template <typename, typename, typename, typename, typename, typename, typename, const WqmmConfig &,
                    const VecAntiQuantConfig &>
          class AivBasicBlock,
          const WqmmConfig &wqmmConfig, const VecAntiQuantConfig &vecConfig>
__aicore__ inline void GMMWeightQuantResplitController<xType, wType, antiQuantScaleType, scaleType, perTokenScaleType,
                                                       biasType, yType, AicBasicBlock, AivBasicBlock, wqmmConfig,
                                                       vecConfig>::SplitNByMultiCore(const TensorA &tensorBlockAGm, const TensorScaleA &tensorBlockScaleAGm, const TensorScaleB &tensorScaleBGm, const TensorY &tensorYGm,
    BasicBlockOffsetParam &offsetParam, BasicBlockControlParam &ctrlParam, uint64_t basicBlockCount,
    uint64_t basicBlockSize)
{
    for (; ctrlParam.curBasicBlockId < ctrlParam.basicBlockLimit + basicBlockCount;
         ctrlParam.curBasicBlockId += gmmBaseTiling_->coreNum) {        
        offsetParam.nOffset =
            ctrlParam.nOffset +
            ((ctrlParam.curBasicBlockId - ctrlParam.basicBlockLimit) % basicBlockCount) * basicBlockSize;
        offsetParam.nL1Size =
            offsetParam.nOffset + basicBlockSize > gmmBaseTiling_->nSize ? gmmBaseTiling_->nSize - offsetParam.nOffset :
                                                                         basicBlockSize;

        auto tensorBlockYGm = tensorYGm(AscendC::Te::MakeCoord(mOffset, nOffset), AscendC::Te::MakeShape(mL1Size, nL1Size));
        auto tensorBlockScaleBGm = tensorScaleBGm(AscendC::Te::MakeCoord(0, nOffset), AscendC::Te::MakeShape(scaleKSize, nL1Size));

        if ASCEND_IS_AIC {
            aicBasicBlock_(tensorBlockAGm, tensorBlockYGm, tensorBlockScaleAGm, tensorBLockScaleBGm, offsetParam);
        } else {
            aivBasicBlock_(offsetParam);
        }
    }
    ctrlParam.nOffset += basicBlockSize * basicBlockCount;
}

template <typename xType, typename wType, typename antiQuantScaleType, typename scaleType, typename perTokenScaleType,
          typename biasType, typename yType,
          template <typename, typename, typename, typename, typename, typename, typename, const WqmmConfig &,
                    const VecAntiQuantConfig &>
          class AicBasicBlock,
          template <typename, typename, typename, typename, typename, typename, typename, const WqmmConfig &,
                    const VecAntiQuantConfig &>
          class AivBasicBlock,
          const WqmmConfig &wqmmConfig, const VecAntiQuantConfig &vecConfig>
__aicore__ inline void GMMWeightQuantResplitController<xType, wType, antiQuantScaleType, scaleType, perTokenScaleType,
                                                       biasType, yType, AicBasicBlock, AivBasicBlock, wqmmConfig,
                                                       vecConfig>::UpdateGmAddr(uint64_t mSize, uint64_t kSize,
                                                                                uint64_t nSize)
{
    xGm_ += mSize * kSize;
    // 4bit，地址偏移单位为8bit
    weightGm_ += (nSize * kSize) >> 1;
    antiquantScaleGm_ += nSize * CeilDivide(kSize, static_cast<uint64_t>(gmmBaseTiling_->groupSize));
    perTokenScaleGm_ += mSize * CeilDivide(kSize, static_cast<uint64_t>(gmmBaseTiling_->groupSize));
    biasGm_ += nSize;
    yGm_ += mSize * nSize;
}

template <typename xType, typename wType, typename antiQuantScaleType, typename scaleType, typename perTokenScaleType,
          typename biasType, typename yType,
          template <typename, typename, typename, typename, typename, typename, typename, const WqmmConfig &,
                    const VecAntiQuantConfig &>
          class AicBasicBlock,
          template <typename, typename, typename, typename, typename, typename, typename, const WqmmConfig &,
                    const VecAntiQuantConfig &>
          class AivBasicBlock,
          const WqmmConfig &wqmmConfig, const VecAntiQuantConfig &vecConfig>
__aicore__ inline void GMMWeightQuantResplitController<xType, wType, antiQuantScaleType, scaleType, perTokenScaleType,
                                                       biasType, yType, AicBasicBlock, AivBasicBlock, wqmmConfig,
                                                       vecConfig>::PrefetchA(uint64_t mSize, uint64_t kSize)
{
    if ASCEND_IS_AIV {
        return;
    }
    if (gmmBaseTiling_->mainBlockCount == 0 &&
        gmmBaseTiling_->firstTailBlockCount + gmmBaseTiling_->secondTailBlockCount <
            gmmBaseTiling_->cubeNumBlocksN) {
        return;
    }
    uint64_t aSize = mSize * kSize * sizeof(xType);
    if (mSize <= 512 && aSize <= static_cast<uint64_t>(gmmBaseTiling_->cubeNumBlocksN) * A_L1_MAX_SIZE_WITH_BIAS_QUANT &&
        (gmmBaseTiling_->coreNum % gmmBaseTiling_->cubeNumBlocksN == 0)) {
        uint64_t aPrefetchSize =
            CeilAlign(CeilDivide(mSize * kSize, static_cast<uint64_t>(gmmBaseTiling_->cubeNumBlocksN)), 64UL);
        aicBasicBlock_.PrefetchA(aPrefetchSize, mSize * kSize);
    }
}

template <typename xType, typename wType, typename antiQuantScaleType, typename scaleType, typename perTokenScaleType,
          typename biasType, typename yType,
          template <typename, typename, typename, typename, typename, typename, typename, const WqmmConfig &,
                    const VecAntiQuantConfig &>
          class AicBasicBlock,
          template <typename, typename, typename, typename, typename, typename, typename, const WqmmConfig &,
                    const VecAntiQuantConfig &>
          class AivBasicBlock,
          const WqmmConfig &wqmmConfig, const VecAntiQuantConfig &vecConfig>
__aicore__ inline uint64_t GMMWeightQuantResplitController<xType, wType, antiQuantScaleType, scaleType,
                                                           perTokenScaleType, biasType, yType, AicBasicBlock,
                                                           AivBasicBlock, wqmmConfig, vecConfig>::GetSplitValueFromGroupList(
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

#endif  // GROUPED_MATMUL_MXFP8FP4_KERNEL_RESPLIT_H
