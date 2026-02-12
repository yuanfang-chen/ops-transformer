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
 * \file add_rms_norm_dynamic_quant_all_gather_qbmm.h
 * \brief add_rms_norm_dynamic_quant_all_gather_qbmm mte通信kernel代码逻辑
 */

#ifndef ADD_RMS_NORM_DYNAMIC_ALL_GATHER_QBMM_H
#define ADD_RMS_NORM_DYNAMIC_ALL_GATHER_QBMM_H

#include "basic_api/kernel_basic_intf.h"
#include "adv_api/hccl/hccl.h"
#include "adv_api/reduce/sum.h"
#include "adv_api/pad/broadcast.h"
#include "kernel_tiling/kernel_tiling.h"
#include "add_rms_norm_dynamic_quant_all_gather_qbmm_tiling.h"
#include "utils.h"
#include "mte_comm.h"
#include "vec_comp.h"
#include "add_rms_norm_dynamic_quant_v2_helper.h"
#if __has_include("../common/inc/kernel/mc2_kernel_utils.h")
#include "../common/inc/kernel/mc2_kernel_utils.h"
#else
#include "../../common/inc/kernel/mc2_kernel_utils.h"
#endif

namespace AddRmsNormDynamicQuantAllGatherQbmmImpl {

// TODO
#define TemplateMC2TypeClass typename X1Type, bool IsScaleExist, bool IsSmoothScaleExist
#define TemplateMC2TypeFunc X1Type, IsScaleExist, IsSmoothScaleExist, 
// using namespace QuantMTECommImpl;
// using namespace VectorComputeImpl;
using namespace AscendC;

// 之后可修改成从tiling侧获取数据切块大小
constexpr static uint32_t X_PRE_BLOCK_NUM = 1024U;  // 当前一次搬运一个x数据块，x dtype为 8bit 时对应 1024个x数据. 对于fp4需要另外算
constexpr static uint64_t MX_SCALES_LAST_DIM = 2U; // MX量化scales最后一维的大小

template<TemplateMC2TypeClass>
class AddRmsNormDynamicQuantAllGatherQbmm {
public:
    __aicore__ inline AddRmsNormDynamicQuantAllGatherQbmm() {};
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR residual, GM_ADDR y, GM_ADDR gamma, GM_ADDR scale,
                                GM_ADDR smoothScale, GM_ADDR bias, GM_ADDR output, GM_ADDR z, GM_ADDR workspaceGM,
                                TPipe *pipe, const AddRmsNormDynamicQuantAllGatherQbmmTilingData *tilingData);
    __aicore__ inline void Process();
private:
    __aicore__ inline void AddRmsNormDynamicQuantProcess();
    
    TPipe *tpipe_{nullptr};
    GlobalTensor<X1Type> x1GMTensor_;
    GlobalTensor<int8_t> x2GMTensor_;
    GlobalTensor<X1Type> residualGMTensor_;
    GlobalTensor<X1Type> yGMTensor_;
    GlobalTensor<float> gammaGMTensor_;
    GlobalTensor<float> scaleGMTensor_; // 类型确定
    GlobalTensor<float> smoothScaleGMTensor_;
    GlobalTensor<int32_t> biasGMTensor_; // 类型确定
    
    LocalTensor<X1Type> x1Tensor_;
    LocalTensor<int8_t> x2Tensor_;
    LocalTensor<float> gammaTensor_;
    LocalTensor<float> scaleTensor_;
    LocalTensor<float> smoothScaleTensor_;
    LocalTensor<int32_t> biasTensor_;
    LocalTensor<float> scaleLocalTensor_;//dynamic quant

    TBuf<> smoothScaleBuf_;    // 搬入smoothScale
    TBuf<> gammaBuf_;
    TBuf<> x1BufFp32_;
    TBuf<> yBufFp32_;
    TBuf<> scalesBuf_;

    TQue<QuePosition::VECIN, 1> inQueue_; // 
    TQue<QuePosition::VECOUT, 1> x1OutQueue_; //
    TQue<QuePosition::VECOUT, 1> zOutQueue_; // 

    uint32_t aivId_{0};
    uint32_t rankId_{0};

    __gm__ HcclOpResParam *winContext_{nullptr};
    
};

template<TemplateMC2TypeClass>
__aicore__ inline void AddRmsNormDynamicQuantAllGatherQbmm<TemplateMC2TypeFunc>::Init(
    GM_ADDR x1, GM_ADDR x2, GM_ADDR residual, GM_ADDR y, GM_ADDR gamma, GM_ADDR scale, GM_ADDR smoothScale, GM_ADDR bias,
    GM_ADDR output, GM_ADDR z, GM_ADDR workspaceGM, TPipe *pipe, const AddRmsNormDynamicQuantAllGatherQbmmTilingData *tilingData)
{
    tpipe_ = pipe;
    aivId_ = GetBlockIdx();
    InitBaseParams(tilingData);
    numRowsAligned_ = (rowStep_ + ELEM_PER_BLK_FP32 - 1) / ELEM_PER_BLK_FP32 * ELEM_PER_BLK_FP32;
    ubAligned_ = (numLastDimAligned_ - numLastDim_) >= ELEM_PER_BLK_FP16;

    x1GMTensor_.SetGlobalBuffer((__gm__ X1Type*)(x1) + aivGMOffset_);
    x2GMTensor_.SetGlobalBuffer((__gm__ int8_t*)x2);
    residualGMTensor_.SetGlobalBuffer((__gm__ int32_t*)(residual) + aivGMOffset_);
    yGMTensor_.SetGlobalBuffer((__gm__ int32_t*)(y) + aivGMOffset_);
    gammaGMTensor_.SetGlobalBuffer((__gm__ int32_t*)gamma);
    // 可选输入
    scaleGMTensor_.SetGlobalBuffer((__gm__ int32_t*)scale);
    smoothScaleGMTensor_.SetGlobalBuffer((__gm__ int32_t*)smoothScale);
    biasGMTensor_.SetGlobalBuffer((__gm__ bool*)bias);

    // tpipe_->InitBuffer(inQueue_, BUFFER_NUM, 3 * axisKaAlignSize_); // 修改
    tpipe_->InitBuffer(inQueue_, BUFFER_NUM, 3 * rowStep_ * numLastDimAligned_ * sizeof(X1Type)); // 修改
    // tpipe_->InitBuffer(x1InQueue_, BUFFER_NUM, 3 * rowStep_ * axisKaAlignSize_); // 修改
    // tpipe_->InitBuffer(residualInQueue_, BUFFER_NUM, 3 * rowStep_ * axisKaAlignSize_); // 修改
    // tpipe_->InitBuffer(yInQueue_, BUFFER_NUM, 3 * rowStep_ * axisKaAlignSize_); // 修改
    tpipe_->InitBuffer(zOutQueue_, BUFFER_NUM, rowStep_ * numLastDimAligned_ * sizeof(X1Type)); // 修改
    tpipe_->InitBuffer(x1OutQueue_, BUFFER_NUM, rowStep_ * numLastDimAligned_ * sizeof(int8_t)); // 修改
    tpipe_->InitBuffer(gammaBuf_, axisKaAlignSize_); // 对齐32B
    tpipe_->InitBuffer(smoothScaleBuf_, axisKaAlignSize_); // 对齐32B
    tpipe_->InitBuffer(x1BufFp32_, rowStep_ * numLastDimAligned_ * sizeof(float));
    tpipe_->InitBuffer(yBufFp32_, rowStep_ * numLastDimAligned_ * sizeof(float));
    tpipe_->InitBuffer(scalesBuf_, numRowsAligned_ * sizeof(float));
    gammaTensor_ = gammaBuf_.Get<float>();
    smoothScaleTensor_ = smoothScaleBuf_.Get<float>();
}

template<TemplateMC2TypeClass>
__aicore__ inline void AddRmsNormDynamicQuantAllGatherQbmm<TemplateMC2TypeFunc>::InitBaseParams(const AddRmsNormDynamicQuantAllGatherQbmmTilingData *tilingData)
{
    aivId_ = GetBlockIdx();
    auto contextGM0 = AscendC::GetHcclContext<HCCL_GROUP_ID_0>();
    winContext_ = (__gm__ HcclOpResParam *)AscendC::GetHcclContext<HCCL_GROUP_ID_0>();
    rankId_ = winContext_->localUsrRankId;
    rankSize_ = 4; // 修改

    axisM_ = tilingData->M;
    axisKa_ = tilingData->Ka;
    axisN_ = tilingData->N;
    // aivNum_ = tilingData->aivNum;
    axisMPerCore_ = tilingData->firstDimPerCore;
    numCore_ = tilingData->useCore;
    numFirstDim_ = tilingData->numFirstDim;
    numLastDim_ = tilingData->numLastDim;
    numLastDimAligned_ = tilingData->numLastDimAligned; // Quantize better be aligned to 32 elements

    firstDimPerCore_ = tilingData->firstDimPerCore;
    firstDimPerCoreTail_ = tilingData->firstDimPerCoreTail;
    firstDimPerLoop_ = tilingData->firstDimPerLoop;

    lastDimSliceLen_ = tilingData->lastDimSliceLen;
    lastDimLoopNum_ = tilingData->lastDimLoopNum;
    lastDimSliceLenTail_ = tilingData->lastDimSliceLenTail;

    eps_ = tilingData->epsilon;
    aveNum_ = tilingData->avgFactor;

    if (aivId_ != numCore - 1) {
        rowWork_ = firstDimPerCore_;
        rowStep_ = firstDimPerLoop_;
    } else {
        rowWork_ = firstDimPerCoreTail_;
        rowStep_ = TWO_NUMS_MIN(firstDimPerLoop_, rowWork_);
    }
    rowTail_ = (rowWork_ % rowStep_ == 0) ? rowStep_ : (rowWork_ % rowStep_);
    gmOffset_ = firstDimPerCore_ * numLastDim_;

    // smooth1Exist = tilingData->smoothNum >= 1;
    // // 2 dynamic quant operator required 2 scale buffer.
    // smooth2Exist = tilingData->smoothNum == 2;

    copyPerCoreCnt_ = axisMPerCore_ * axisKa_;
    aivGMOffset_ = aivId_ * copyInPerCoreCnt_;
    axisKaAlignSize_ = Ceil(axisKa_ * sizeof(X1Type), UB_ALIGN) * UB_ALIGN;
    axisKaAlignSize_ = Ceil(axisKa_ * sizeof(float), UB_ALIGN) * UB_ALIGN;
    copyInCntAlignSize_ = Ceil(copyPerCoreCnt_ * sizeof(X1Type), UB_ALIGN) * UB_ALIGN;
    copyOutCntAlignSize_ = Ceil(copyPerCoreCnt_ * sizeof(int8_t), UB_ALIGN) * UB_ALIGN;
}

// template<TemplateMC2TypeClass>
// __aicore__ inline void AddRmsNormDynamicQuantAllGatherQbmm<TemplateMC2TypeFunc>::ComputeDynamicQuant()
// {
    
// }

template<TemplateMC2TypeClass>
__aicore__ inline void AddRmsNormDynamicQuantAllGatherQbmm<TemplateMC2TypeFunc>::Add2RmsNormDynamicQuantCompute(
    int32_t gmOffset, int32_t gmOffsetScale, int32_t rowCount, int32_t elementCount)
{
    LocalTensor<X1Type> copyInLocalTensor = inQueue_.AllocTensor<X1Type>();
    LocalTensor<X1Type> zOutLocalTensor = zOutQueue_.AllocTensor<X1Type>();

    DataCopyEx(copyInLocalTensor[0], x1GMTensor_[gmOffset], axisKa_, rowCount, ubAligned_);
    DataCopyEx(copyInLocalTensor[elementCount], residualGMTensor_[gmOffset], axisKa_, rowCount, ubAligned_);
    DataCopyEx(copyInLocalTensor[elementCount * 2], yGMTensor_[gmOffset], axisKa_, rowCount, ubAligned_);
    inQueue_.EnQue(copyInLocalTensor);
    copyInLocalTensor = inQueue_.DeQue<X1Type>();
    LocalTensor<X1Type> x1LocalTensor = copyInLocalTensor[0];
    LocalTensor<X1Type> residualLocalTensor = copyInLocalTensor[elementCount];
    LocalTensor<X1Type> yLocalTensor = copyInLocalTensor[elementCount * 2];
    LocalTensor<float> xLocalTensorFp32 = x1BufFp32_.Get<float>();
    LocalTensor<float> yLocalTensorFp32 = yBufFp32_.Get<float>();

    // Add
    Cast(xLocalTensorFp32, x1LocalTensor, RoundMode::CAST_NONE, elementCount);
    Cast(yLocalTensorFp32, residualLocalTensor, RoundMode::CAST_NONE, elementCount);
    PipeBarrier<PIPE_V>();
    Add(xLocalTensorFp32, xLocalTensorFp32, yLocalTensorFp32, elementCount);
    PipeBarrier<PIPE_V>();
    Cast(yLocalTensorFp32, yLocalTensor, RoundMode::CAST_NONE, elementCount);
    PipeBarrier<PIPE_V>();
    Add(xLocalTensorFp32, xLocalTensorFp32, yLocalTensorFp32, elementCount);
    inQueue_.FreeTensor<X1Type>(copyInLocalTensor);
    PipeBarrier<PIPE_V>();

    // CopyOut z
    if constexpr (is_same<X1Type, half>::value) {
        Cast(zOutLocalTensor, xLocalTensorFp32, RoundMode::CAST_NONE, elementCount);
    } else { // BF16
        Cast(zOutLocalTensor, xLocalTensorFp32, RoundMode::CAST_RINT, elementCount);
    }
    zOutQueue_.EnQue(zOutLocalTensor);
    zOutLocalTensor = zOutQueue_.DeQue<X1Type>();
    DataCopyEx(zGMTensor_[gmOffset], zOutLocalTensor, axisKa_, rowCount, ubAligned_);
    zOutQueue_.FreeTensor<X1Type>(zOutLocalTensor);

    // RMS Norm
    PipeBarrier<PIPE_V>();
    Mul(yLocalTensorFp32, xLocalTensorFp32, xLocalTensorFp32, elementCount); // yLocalFp32 <- x ** 2
    PipeBarrier<PIPE_V>();

    for (int32_t rid = 0; rid < rowCount; ++rid) {
        int32_t roundOffset = rid * numLastDimAligned_;
        float squareSumTemp = ReduceSumHalfInterval(yLocalTensorFp32[roundOffset], axisKa_); // aveLocalTemp <-- E(x**2)
        float rstdLocalTemp = 1 / sqrt(squareSumTemp * aveNum_ + eps_);
        SyncFunc<AscendC::HardEvent::V_S>();
        Muls(xLocalTensorFp32[roundOffset], xLocalTensorFp32[roundOffset], rstdLocalTemp, axisKa_); // xLocalFp32 <- x * rstd
    }

    // reduce#1 for mean
    for (int32_t rid = 0; rid < rowCount; ++rid) {
        auto roundOffset = rid * numLastDimAligned_;
        float squareSumTemp = ReduceSumHalfInterval(yLocalTensorFp32[roundOffset], axisKa_); // aveLocalTemp <-- E(x**2)
        float rstdLocalTemp = 1 / sqrt(squareSumTemp * aveNum_ + eps_);
        SyncFunc<AscendC::HardEvent::V_S>();
        Muls(xLocalTensorFp32[roundOffset], xLocalTensorFp32[roundOffset], rstdLocalTemp, axisKa_); // xLocalFp32 <- x * rstd
    }
    PipeBarrier<PIPE_V>();
    Cast(yLocalTensorFp32, gammaTensor_, RoundMode::CAST_NONE, axisKa_); // yLocalFp32 <- gamma
    PipeBarrier<PIPE_V>();
    for (int32_t rid = 0; rid < rowCount; ++rid) {
        auto roundOffset = rid * numLastDimAligned_;
        Mul(xLocalTensorFp32[roundOffset], xLocalTensorFp32[roundOffset], yLocalTensorFp32, axisKa_); // xLocalFp32 <- x * rstd * gamma
    }
    PipeBarrier<PIPE_V>();

    // DynamicQuant smooth
    auto smoothScaleTensorFp32 = yLocalTensorFp32[(rowCount - 1) * numLastDimAligned_];
    Cast(smoothScaleTensorFp32, smoothScaleTensor_, RoundMode::CAST_NONE, axisKa_);
    PipeBarrier<PIPE_V>();
    for (int32_t rid = 0; rid < rowCount; ++rid) {
        Mul(yLocalTensorFp32[rid * numLastDimAligned_], xLocalTensorFp32[numLastDimAligned_], smoothScaleTensorFp32, axisKa_);
    }
    PipeBarrier<PIPE_V>();
    
    // scale
    float maxTemp;
    float scaleTemp;
    Abs(xLocalTensorFp32, yLocalTensorFp32, elementCount); // tmpLocal <-- |y * smooth1|
    PipeBarrier<PIPE_V>();
    for (int32_t rid = 0; rid < rowCount; ++rid) {
        ReduceMaxInplace(xLocalTensorFp32[rid * numLastDimAligned_], axisKa_);
        SyncFunc<AscendC::HardEvent::V_S>();
        maxTemp = xLocalTensorFp32[rid * numLastDimAligned_].GetValue(0); // Reduce
        scaleTemp = float(127.0) / maxTemp;
        scaleLocalTensor_.SetValue(rid, 1 / scaleTemp);
        SyncFunc<AscendC::HardEvent::S_V>();
        Muls(yLocalTensorFp32[rid * numLastDimAligned_], yLocalTensorFp32[rid * numLastDimAligned_], scaleTemp, axisKa_);
    }
    PipeBarrier<PIPE_V>();

    // copy out 到win区
    LocalTensor<int8_t> x1OutLocalTensor = x1OutQueue_.AllocTensor<int8_t>();
    RoundFloat2Int8(x1OutLocalTensor, yLocalTensorFp32, elementCount);
    x1OutQueue_.EnQue(x1OutLocalTensor);
    x1OutLocalTensor = x1OutQueue_.DeQue<int8_t>();
    DataCopyEx(x1WinGMTensor_[gmOffset], x1OutLocalTensor, axisKa_, rowCount);
    DataCopyEx(scaleWinGMTensor_[gmOffsetScale], scaleLocalTensor_, rowCount);
    x1OutQueue_.FreeTensor(x1OutLocalTensor);
}

template<TemplateMC2TypeClass>
__aicore__ inline void AddRmsNormDynamicQuantAllGatherQbmm<TemplateMC2TypeFunc>::Add2RmsNormDynamicQuantProcess()
{
    int32_t rowMoveCnt = CEIL_DIV(rowWork, rowStep);
    scaleLocalTensor_ = scalesBuf_.Get<float>();
    DataCopyEx(gammaTensor_, gammaGMTensor_, axisKa_);
    DataCopyEx(smoothScaleTensor_, smoothScaleGMTensor_, axisKa_);

    int32_t gmOffset = 0;
    int32_t gmOffsetScale = 0;
    int32_t elementCount = numLastDimAligned_ * rowStep_;
    uint64_t winOffset = Ceil(rankSize_ * axisM_ * axisKa_ * sizeof(int8_t), WIN_ALIGN) * WIN_ALIGN;
    GM_ADDR selfRankAddr = (GM_ADDR)(winContext_->localWindowsIn);
    GM_ADDR x1WinGM = (__gm__ uint8_t*)(selfRankAddr + rankId_ * axisM_ * axisKa_ * sizeof(int8_t));
    GM_ADDR dynamicScaleWinGM = (__gm__ uint8_t*)(selfRankAddr + winOffset + rankId_ * axisM_ * sizeof(float));

    for (int32_t rowIdx = 0; rowIdx < rowMoveCnt - 1; ++rowIdx) {
        Add2RmsNormDynamicQuantCompute(gmOffset, gmOffsetScale, rowStep_, elementCount)
        gmOffset += rowStep * numLastDim;
        gmOffsetScale += rowStep;
    }

    // tail
    Add2RmsNormDynamicQuantCompute(gmOffset, gmOffsetScale, rowTail_, elementCount)
}

template<TemplateMC2TypeClass>
__aicore__ inline void AddRmsNormDynamicQuantAllGatherQbmm<TemplateMC2TypeFunc>::SetRemoteFlag()
{
    

}

template<TemplateMC2TypeClass>
__aicore__ inline void AddRmsNormDynamicQuantAllGatherQbmm<TemplateMC2TypeFunc>::WaitRemoteFlag()
{
    

}

template<TemplateMC2TypeClass>
__aicore__ inline void AddRmsNormDynamicQuantAllGatherQbmm<TemplateMC2TypeFunc>::MatmulProcess()
{
    

}

template<TemplateMC2TypeClass>
__aicore__ inline void AddRmsNormDynamicQuantAllGatherQbmm<TemplateMC2TypeFunc>::Process()
{
    if ASCEND_IS_AIV {
        Add2RmsNormDynamicQuantProcess();
        SyncAll<true>();
        PipeBarrier<PIPE_MTE3>();
    }
    
    if ASCEND_IS_AIC {
        MatmulProcess();
    }

}
} // AddRmsNormDynamicQuantAllGatherQbmmImpl
#endif  // ADD_RMS_NORM_DYNAMIC_ALL_GATHER_QBMM_H