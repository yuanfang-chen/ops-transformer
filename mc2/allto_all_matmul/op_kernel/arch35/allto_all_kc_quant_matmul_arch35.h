/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file allto_all_kc_quant_matmul_arch35.h
 * \brief
 */

#ifndef ALLTO_ALL_KC_QUANT_MATMUL_ARCH35_H
#define ALLTO_ALL_KC_QUANT_MATMUL_ARCH35_H

#include "allto_all_matmul_tiling_data.h"

#define KC_DYN_QUANT_FP8E5M2 35
#define KC_DYN_QUANT_FP8E4M3 36

namespace AlltoAllMatmulImpl {
using namespace AscendC;

template <typename SchedulerType, typename SchedulerContextType, typename AlltoAllMatmulTilingDataType>
class AlltoAllKcQuantMatmulArch35 {
public:
    __aicore__ inline AlltoAllKcQuantMatmulArch35(SchedulerType *pipeLine) : pipeLine_(pipeLine){};
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR bias, GM_ADDR y, GM_ADDR all2all_out,
                                GM_ADDR smooth_scale, GM_ADDR x2_scale, GM_ADDR x2_offset, GM_ADDR workspaceGM,
                                AlltoAllMatmulTilingDataType *tilingData, TPipe *tPipe);
    __aicore__ inline void Process();

private:
    SchedulerType *pipeLine_;
    SchedulerContextType pipeLineContext_;
    AlltoAllMatmulTilingDataType *tilingData_;
    TPipe *tPipe_;
    GM_ADDR x1_;
    GM_ADDR x2_;
    GM_ADDR y_;
    GM_ADDR bias_;
    GM_ADDR smoothScale_;
    GM_ADDR x2Scale_;
    GM_ADDR x2Offset_;
    GM_ADDR workspaceGM_;
    GM_ADDR commOutGM_;
    GM_ADDR transOutGM_;
    GM_ADDR x1ScaleGM_;
    GM_ADDR quantOutGM_;

private:
    __aicore__ inline void ProcessTile(uint32_t taskCnt);
    __aicore__ inline void ProcessTail(uint32_t taskCnt);
};

template <typename SchedulerType, typename SchedulerContextType, typename AlltoAllMatmulTilingDataType>
__aicore__ inline void
AlltoAllKcQuantMatmulArch35<SchedulerType, SchedulerContextType, AlltoAllMatmulTilingDataType>::Init(
    GM_ADDR x1, GM_ADDR x2, GM_ADDR bias, GM_ADDR y, GM_ADDR all2all_out, GM_ADDR smooth_scale, GM_ADDR x2_scale,
    GM_ADDR x2_offset, GM_ADDR workspaceGM, AlltoAllMatmulTilingDataType *tilingData, TPipe *tPipe)
{
    // 获取tilingdata数据
    tilingData_ = tilingData;
    auto &&mc2Tiling_ = tilingData_->alltoAllKcQuantMatmulTilingInfo;
    tPipe_ = tPipe;
    x1_ = x1;
    x2_ = x2;
    y_ = y;
    bias_ = bias;
    smoothScale_ = smooth_scale;
    x2Scale_ = x2_scale;
    x2Offset_ = x2_offset;

    workspaceGM_ = workspaceGM;
    commOutGM_ = workspaceGM;
    transOutGM_ = all2all_out;
    if (all2all_out == nullptr) {
        transOutGM_ = (GM_ADDR)((uint64_t)commOutGM_ + mc2Tiling_.commLen);
    }
    x1ScaleGM_ = (GM_ADDR)((uint64_t)commOutGM_ + mc2Tiling_.commLen + mc2Tiling_.permuteLen);
    quantOutGM_ = (GM_ADDR)((uint64_t)x1ScaleGM_ + mc2Tiling_.x1ScaleOptionalLen);

    // 初始化流水线
    pipeLine_->Init();
}

template <typename SchedulerType, typename SchedulerContextType, typename AlltoAllMatmulTilingDataType>
__aicore__ inline void
AlltoAllKcQuantMatmulArch35<SchedulerType, SchedulerContextType, AlltoAllMatmulTilingDataType>::Process()
{
    auto &&mc2Tiling_ = tilingData_->alltoAllKcQuantMatmulTilingInfo;
    // 启动主块流水
    if (mc2Tiling_.tileCnt > 0) {
        ProcessTile(mc2Tiling_.tileCnt);
    }

    // 启动尾块流水
    if (mc2Tiling_.tailCnt > 0) {
        ProcessTail(mc2Tiling_.tailCnt);
    }

    // 结束流水线
    pipeLine_->End();
}

template <typename SchedulerType, typename SchedulerContextType, typename AlltoAllMatmulTilingDataType>
__aicore__ inline void
AlltoAllKcQuantMatmulArch35<SchedulerType, SchedulerContextType, AlltoAllMatmulTilingDataType>::ProcessTile(
    uint32_t taskCnt)
{
    auto &&mc2Tiling_ = tilingData_->alltoAllKcQuantMatmulTilingInfo;
    // 复用的变量
    uint64_t tileMMultiRankK = (uint64_t)mc2Tiling_.tileM * (uint64_t)mc2Tiling_.rankK;

    // 通信相关地址和偏移
    pipeLineContext_.communicationContext->taskCnt = mc2Tiling_.tileCnt;
    pipeLineContext_.communicationContext->sendBuffer = x1_;
    pipeLineContext_.communicationContext->recvBuffer = commOutGM_;
    pipeLineContext_.communicationContext->sendOffset = tileMMultiRankK * (uint64_t)sizeof(DTYPE_X1);
    pipeLineContext_.communicationContext->recvOffset = pipeLineContext_.communicationContext->sendOffset;
    pipeLineContext_.communicationContext->sendCount = tileMMultiRankK;
    pipeLineContext_.communicationContext->strideCount =
        (uint64_t)mc2Tiling_.rankM * (uint64_t)mc2Tiling_.rankK / (uint64_t)mc2Tiling_.rankDim;
    pipeLineContext_.communicationContext->hcclDataType = mc2Tiling_.hcclDataType;

    // 转置相关地址和偏移
    pipeLineContext_.transposeContext->transposeSrcAddr = commOutGM_;
    pipeLineContext_.transposeContext->transposeDstAddr = transOutGM_;
    pipeLineContext_.transposeContext->transposeSrcOffset = tileMMultiRankK * (uint64_t)sizeof(DTYPE_X1);
    pipeLineContext_.transposeContext->transposeDstOffset = tileMMultiRankK * (uint64_t)mc2Tiling_.rankDim * (uint64_t)sizeof(DTYPE_X1);
    pipeLineContext_.transposeContext->nextSrcBlockOffset =
        (uint64_t)mc2Tiling_.rankM * (uint64_t)mc2Tiling_.rankK / (uint64_t)mc2Tiling_.rankDim;
    pipeLineContext_.transposeContext->nextDstBlockOffset = (uint64_t)mc2Tiling_.rankK;
    pipeLineContext_.transposeContext->rankCnt = (uint64_t)mc2Tiling_.rankDim;
    pipeLineContext_.transposeContext->innerAxis = (uint64_t)mc2Tiling_.rankK;
    pipeLineContext_.transposeContext->transM = (uint64_t)mc2Tiling_.tileM;

    // 动态量化相关地址和偏移
    pipeLineContext_.quantizationContext->quantOutputScaleAddr = x1ScaleGM_;
    pipeLineContext_.quantizationContext->quantOutputScaleAddrOffset = (uint64_t)mc2Tiling_.tileM * sizeof(float);
    pipeLineContext_.quantizationContext->quantInputAddr = transOutGM_;
    pipeLineContext_.quantizationContext->quantInputAddrOffset = tileMMultiRankK * (uint64_t)sizeof(DTYPE_X1) * mc2Tiling_.rankDim;
    pipeLineContext_.quantizationContext->quantOutputAddr = quantOutGM_;
    pipeLineContext_.quantizationContext->quantOutputAddrOffset = tileMMultiRankK * (uint64_t)sizeof(DTYPE_X2) * mc2Tiling_.rankDim;
    pipeLineContext_.quantizationContext->rowNum = (uint64_t)mc2Tiling_.tileM;
    pipeLineContext_.quantizationContext->colNum = (uint64_t)mc2Tiling_.rankK * mc2Tiling_.rankDim;

    // quantMatmul相关的地址和偏移
    pipeLineContext_.computationContext->baseData.aGM = quantOutGM_;
    pipeLineContext_.computationContext->baseData.bGM = x2_;
    pipeLineContext_.computationContext->baseData.cGM = y_;
    pipeLineContext_.computationContext->baseData.biasGM = bias_;
    pipeLineContext_.computationContext->baseData.a_offset = tileMMultiRankK * (uint64_t)mc2Tiling_.rankDim * (uint64_t)sizeof(DTYPE_X2);
    pipeLineContext_.computationContext->baseData.b_offset = (uint64_t)0UL;
    pipeLineContext_.computationContext->baseData.c_offset =
        (uint64_t)mc2Tiling_.tileM * (uint64_t)mc2Tiling_.rankN * (uint64_t)sizeof(DTYPE_Y);
    pipeLineContext_.computationContext->additionalData.x1_scale = x1ScaleGM_;
    pipeLineContext_.computationContext->additionalData.x1_scale_offset = (uint64_t)mc2Tiling_.tileM * sizeof(float);
    pipeLineContext_.computationContext->additionalData.x2_scale = x2Scale_;
    pipeLineContext_.computationContext->additionalData.x2_offset = x2Offset_;
    pipeLineContext_.computationContext->tilingDataPtr = &(tilingData_->mc2KcQuantMmTileTilingData);

    pipeLine_->Process(taskCnt);
}

template <typename SchedulerType, typename SchedulerContextType, typename AlltoAllMatmulTilingDataType>
__aicore__ inline void
AlltoAllKcQuantMatmulArch35<SchedulerType, SchedulerContextType, AlltoAllMatmulTilingDataType>::ProcessTail(
    uint32_t taskCnt)
{
    auto &&mc2Tiling_ = tilingData_->alltoAllKcQuantMatmulTilingInfo;
    uint64_t tailMMultiRankK = (uint64_t)mc2Tiling_.tailM * (uint64_t)mc2Tiling_.rankK;
    uint64_t tileCntMultitileMMultiRankK =
        (uint64_t)mc2Tiling_.tileCnt * (uint64_t)mc2Tiling_.tileM * (uint64_t)mc2Tiling_.rankK;

    // 通信相关地址和偏移
    pipeLineContext_.communicationContext->taskCnt = mc2Tiling_.tailCnt;
    pipeLineContext_.communicationContext->sendBuffer = x1_ + tileCntMultitileMMultiRankK * sizeof(DTYPE_X1);
    pipeLineContext_.communicationContext->recvBuffer = commOutGM_ + tileCntMultitileMMultiRankK * sizeof(DTYPE_X1);
    pipeLineContext_.communicationContext->sendOffset = tailMMultiRankK * sizeof(DTYPE_X1);
    pipeLineContext_.communicationContext->recvOffset = pipeLineContext_.communicationContext->sendOffset;
    pipeLineContext_.communicationContext->sendCount = tailMMultiRankK;

    // 转置相关的地址和偏移
    pipeLineContext_.transposeContext->transposeSrcAddr = commOutGM_ + tileCntMultitileMMultiRankK * sizeof(DTYPE_X1);
    pipeLineContext_.transposeContext->transposeDstAddr =
        transOutGM_ + tileCntMultitileMMultiRankK * (uint64_t)mc2Tiling_.rankDim * sizeof(DTYPE_X1);
    pipeLineContext_.transposeContext->transposeSrcOffset = tailMMultiRankK * (uint64_t)sizeof(DTYPE_X1);
    pipeLineContext_.transposeContext->transposeDstOffset = tailMMultiRankK * (uint64_t)mc2Tiling_.rankDim * sizeof(DTYPE_X1);
    pipeLineContext_.transposeContext->transM = (uint64_t)mc2Tiling_.tailM;

    // 动态量化相关地址和偏移
    pipeLineContext_.quantizationContext->quantOutputScaleAddr =
        x1ScaleGM_ + (uint64_t)mc2Tiling_.tileCnt * mc2Tiling_.tileM * sizeof(float);
    pipeLineContext_.quantizationContext->quantOutputScaleAddrOffset = (uint64_t)mc2Tiling_.tailM * sizeof(float);
    pipeLineContext_.quantizationContext->quantInputAddr = transOutGM_ + tileCntMultitileMMultiRankK * sizeof(DTYPE_X1) * mc2Tiling_.rankDim;
    pipeLineContext_.quantizationContext->quantInputAddrOffset = tailMMultiRankK * sizeof(DTYPE_X1) * mc2Tiling_.rankDim;
    pipeLineContext_.quantizationContext->quantOutputAddr =
        quantOutGM_ + tileCntMultitileMMultiRankK * sizeof(DTYPE_X2) * mc2Tiling_.rankDim;
    pipeLineContext_.quantizationContext->quantOutputAddrOffset = tailMMultiRankK * sizeof(DTYPE_X2) * mc2Tiling_.rankDim;
    pipeLineContext_.quantizationContext->rowNum = (uint64_t)mc2Tiling_.tailM;

    // quantMatmul相关的地址和偏移
    pipeLineContext_.computationContext->baseData.aGM = quantOutGM_ + tileCntMultitileMMultiRankK * sizeof(DTYPE_X2) * mc2Tiling_.rankDim;
    pipeLineContext_.computationContext->baseData.bGM = x2_;
    pipeLineContext_.computationContext->baseData.cGM = y_ + (uint64_t)mc2Tiling_.tileCnt * mc2Tiling_.tileM * mc2Tiling_.rankN * sizeof(DTYPE_Y);
    pipeLineContext_.computationContext->baseData.a_offset = tailMMultiRankK * mc2Tiling_.rankDim * sizeof(DTYPE_X2);
    pipeLineContext_.computationContext->baseData.b_offset = (uint64_t)0UL;
    pipeLineContext_.computationContext->baseData.c_offset = (uint64_t)mc2Tiling_.tailM * mc2Tiling_.rankN * (uint64_t)sizeof(DTYPE_Y);
    pipeLineContext_.computationContext->additionalData.x1_scale = x1ScaleGM_ + mc2Tiling_.tileCnt * mc2Tiling_.tileM * sizeof(float);
    pipeLineContext_.computationContext->additionalData.x1_scale_offset = (uint64_t)mc2Tiling_.tailM * sizeof(float);
    pipeLineContext_.computationContext->additionalData.x2_scale = x2Scale_;
    pipeLineContext_.computationContext->additionalData.x2_offset = x2Offset_;
    pipeLineContext_.computationContext->tilingDataPtr = &(tilingData_->mc2KcQuantMmTailTilingData);

    pipeLine_->Process(taskCnt);
}
} // namespace AlltoAllMatmulImpl
#endif