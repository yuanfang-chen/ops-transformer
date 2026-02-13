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
#include "add_rms_norm_dynamic_quant_all_gather_qbmm_tiling_data.h"
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
constexpr uint8_t BUFFER_NUM = 2; // 多Buf
constexpr uint32_t UB_ALIGN = 32; // UB按32字节对齐
constexpr uint32_t WIN_ALIGN = 512; // win offset 512字节对齐

template<TemplateMC2TypeClass>
class AddRmsNormDynamicQuantAllGatherQbmm {
public:
    __aicore__ inline AddRmsNormDynamicQuantAllGatherQbmm() {};
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR residual, GM_ADDR y, GM_ADDR gamma, GM_ADDR scale,
                                GM_ADDR smoothScale, GM_ADDR bias, GM_ADDR output, GM_ADDR z, GM_ADDR workspaceGM,
                                TPipe *pipe, const AddRmsNormDynamicQuantAllGatherQbmmTilingData *tilingData);
    __aicore__ inline void Process();
private:
    __aicore__ inline void InitBaseParams(const AddRmsNormDynamicQuantAllGatherQbmmTilingData *tilingData);
    __aicore__ inline void SplitToCore(uint32_t curSendCnt, uint32_t curUseAivNum, uint32_t startId, uint32_t endId, uint32_t sendNum);
    __aicore__ inline void Add2RmsNormDynamicQuantCompute(int32_t gmOffset, int32_t rowIdx, int32_t startRowId, int32_t elementCount);
    __aicore__ inline void Add2RmsNormDynamicQuantProcess();
    
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
    LocalTensor<float> dynamicScaleLocalTensor_;//dynamic quant
    LocalTensor<float> xLocalTensorFp32_;
    LocalTensor<float> yLocalTensorFp32_;
    LocalTensor<int8_t> x1OutLocalTensor_;

    TBuf<> smoothScaleBuf_;    // 搬入smoothScale
    TBuf<> gammaBuf_;
    TBuf<> x1TempBuf_;
    TBuf<> yTempBuf_;
    TBuf<> weightTempBuf_;
    TBuf<> dynamicScaleBuf_;

    TQue<QuePosition::VECIN, 1> inQueue_; // 
    TQue<QuePosition::VECOUT, 1> x1OutQueue_; //
    TQue<QuePosition::VECOUT, 1> zOutQueue_; // 

    uint32_t aivId_{0};
    uint32_t rankId_{0};
    uint32_t axisM_{0};
    uint32_t axisKa_{0};
    uint32_t axisN_{0};
    uint32_t aivNum_{0};
    uint32_t rankSize_{0};
    uint32_t eps_{0};
    uint32_t aveNum_{0};
    uint64_t axisKaAlignSize_{0};
    uint64_t axisKaAlignFloatSize_{0};
    uint64_t axisKaAlignInt8Size_{0};

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

    x1GMTensor_.SetGlobalBuffer((__gm__ X1Type*)x1);
    x2GMTensor_.SetGlobalBuffer((__gm__ int8_t*)x2);
    residualGMTensor_.SetGlobalBuffer((__gm__ int32_t*)residual);
    yGMTensor_.SetGlobalBuffer((__gm__ int32_t*)y);
    gammaGMTensor_.SetGlobalBuffer((__gm__ int32_t*)gamma);
    // 可选输入
    scaleGMTensor_.SetGlobalBuffer((__gm__ int32_t*)scale);
    smoothScaleGMTensor_.SetGlobalBuffer((__gm__ int32_t*)smoothScale);
    biasGMTensor_.SetGlobalBuffer((__gm__ bool*)bias);

    // tpipe_->InitBuffer(inQueue_, BUFFER_NUM, 3 * axisKaAlignSize_); // 修改
    tpipe_->InitBuffer(inQueue_, BUFFER_NUM, 3 * axisKaAlignSize_); // 修改
    tpipe_->InitBuffer(zOutQueue_, BUFFER_NUM, axisKaAlignSize_); // 修改
    tpipe_->InitBuffer(x1OutQueue_, BUFFER_NUM, axisKaAlignInt8Size_); // 修改
    tpipe_->InitBuffer(gammaBuf_, axisKaAlignFloatSize_); // 对齐32B
    tpipe_->InitBuffer(smoothScaleBuf_, axisKaAlignFloatSize_); // 对齐32B
    tpipe_->InitBuffer(x1TempBuf_, axisKaAlignFloatSize_);
    tpipe_->InitBuffer(yTempBuf_, axisKaAlignFloatSize_);
    gammaTensor_ = gammaBuf_.Get<float>();
    smoothScaleTensor_ = smoothScaleBuf_.Get<float>();
    xLocalTensorFp32_ = x1TempBuf_.Get<float>();
    yLocalTensorFp32_ = yTempBuf_.Get<float>();

    uint64_t winOffset = Ceil(rankSize_ * axisM_ * axisKa_ * sizeof(int8_t), WIN_ALIGN) * WIN_ALIGN;
    GM_ADDR selfRankAddr = (GM_ADDR)(winContext_->localWindowsIn);
    GM_ADDR x1WinGM = (__gm__ uint8_t*)(selfRankAddr + rankId_ * axisM_ * axisKa_ * sizeof(int8_t));
    GM_ADDR dynamicScaleWinGM = (__gm__ uint8_t*)(selfRankAddr + winOffset + rankId_ * axisM_ * sizeof(float));
    x1WinGMTensor_.SetGlobalBuffer((__gm__ int8_t*)x1WinGM);
    scaleWinGMTensor_.SetGlobalBuffer((__gm__ float*)dynamicScaleWinGM);
}

template<TemplateMC2TypeClass>
__aicore__ inline void AddRmsNormDynamicQuantAllGatherQbmm<TemplateMC2TypeFunc>::InitBaseParams(const AddRmsNormDynamicQuantAllGatherQbmmTilingData *tilingData)
{
    aivId_ = GetBlockIdx();
    auto contextGM0 = AscendC::GetHcclContext<HCCL_GROUP_ID_0>();
    winContext_ = (__gm__ HcclOpResParam *)AscendC::GetHcclContext<HCCL_GROUP_ID_0>();
    rankId_ = winContext_->localUsrRankId;
    // rankSize_ = 4; // 修改

    axisM_ = tilingData->M;
    axisKa_ = tilingData->Ka;
    axisN_ = tilingData->N;
    aivNum_ = tilingData->aivNum;
    rankSize_ = tilingData->rankSize;
    eps_ = tilingData->epsilon;
    aveNum_ = tilingData->avgFactor;

    axisKaAlignSize_ = Ceil(axisKa_ * sizeof(X1Type), UB_ALIGN) * UB_ALIGN;
    axisKaAlignFloatSize_ = Ceil(axisKa_ * sizeof(float), UB_ALIGN) * UB_ALIGN;
    axisKaAlignInt8Size_ = Ceil(axisKa_ * sizeof(int8_t), UB_ALIGN) * UB_ALIGN;
}

template<TemplateMC2TypeClass>
__aicore__ inline void AddRmsNormDynamicQuantAllGatherQbmm<TemplateMC2TypeFunc>::SplitToCore(uint32_t curSendCnt, uint32_t curUseAivNum, uint32_t &startId, uint32_t &endId, uint32_t &sendNum)
{
    sendNum = curSendCnt / curUseAivNum; // 每个aiv需要处理的数
    uint32_t remainderNum = curSendCnt % curUseAivNum; // 余数
    startId = sendNum * aivId_; // 每个aiv发送时的起始id
    if (aivId_ < remainderNum) { // 前remainderNum个aiv需要多处理1个数据
        sendNum += 1;
        startId += aivId_;
    } else {
        startId += remainderNum;
    }
    endId = startId + sendNum;
}

// template<TemplateMC2TypeClass>
// __aicore__ inline void AddRmsNormDynamicQuantAllGatherQbmm<TemplateMC2TypeFunc>::ComputeDynamicQuant()
// {
    
// }

template<TemplateMC2TypeClass>
__aicore__ inline void AddRmsNormDynamicQuantAllGatherQbmm<TemplateMC2TypeFunc>::Add2RmsNormDynamicQuantCompute(
    int32_t gmOffset, int32_t rowIdx, int32_t startRowId, int32_t elementCount)
{
    LocalTensor<X1Type> copyInLocalTensor = inQueue_.AllocTensor<X1Type>();
    LocalTensor<X1Type> zOutLocalTensor = zOutQueue_.AllocTensor<X1Type>();

    DataCopyEx(copyInLocalTensor[0], x1GMTensor_[gmOffset], axisKa_);
    DataCopyEx(copyInLocalTensor[elementCount], residualGMTensor_[gmOffset], axisKa_);
    DataCopyEx(copyInLocalTensor[elementCount * 2], yGMTensor_[gmOffset], axisKa_);
    inQueue_.EnQue(copyInLocalTensor);
    copyInLocalTensor = inQueue_.DeQue<X1Type>();
    LocalTensor<X1Type> x1LocalTensor = copyInLocalTensor[0];
    LocalTensor<X1Type> residualLocalTensor = copyInLocalTensor[elementCount];
    LocalTensor<X1Type> yLocalTensor = copyInLocalTensor[elementCount * 2];

    // Add
    Cast(xLocalTensorFp32_, x1LocalTensor, RoundMode::CAST_NONE, axisKa_);
    Cast(yLocalTensorFp32_, residualLocalTensor, RoundMode::CAST_NONE, axisKa_);
    PipeBarrier<PIPE_V>();
    Add(xLocalTensorFp32_, xLocalTensorFp32_, yLocalTensorFp32_, axisKa_);
    PipeBarrier<PIPE_V>();
    Cast(yLocalTensorFp32_, yLocalTensor, RoundMode::CAST_NONE, axisKa_);
    PipeBarrier<PIPE_V>();
    Add(xLocalTensorFp32_, xLocalTensorFp32_, yLocalTensorFp32_, axisKa_);
    inQueue_.FreeTensor<X1Type>(copyInLocalTensor);
    PipeBarrier<PIPE_V>();

    // CopyOut z
    if constexpr (is_same<X1Type, half>::value) {
        Cast(zOutLocalTensor, xLocalTensorFp32_, RoundMode::CAST_NONE, axisKa_);
    } else { // BF16
        Cast(zOutLocalTensor, xLocalTensorFp32_, RoundMode::CAST_RINT, axisKa_);
    }
    zOutQueue_.EnQue(zOutLocalTensor);
    zOutLocalTensor = zOutQueue_.DeQue<X1Type>();
    DataCopyEx(zGMTensor_[gmOffset], zOutLocalTensor, axisKa_);
    zOutQueue_.FreeTensor<X1Type>(zOutLocalTensor);

    // RMS Norm
    PipeBarrier<PIPE_V>();
    Mul(yLocalTensorFp32_, xLocalTensorFp32_, xLocalTensorFp32_, axisKa_); // yLocalTensorFp32_ <- x ** 2
    PipeBarrier<PIPE_V>();

    // reduce#1 for mean
    float squareSumTemp = ReduceSumHalfInterval(yLocalTensorFp32_, axisKa_); // aveLocalTemp <-- E(x**2)
    float rstdLocalTemp = 1 / sqrt(squareSumTemp * aveNum_ + eps_);
    SyncFunc<AscendC::HardEvent::V_S>();
    Muls(xLocalTensorFp32_, xLocalTensorFp32_, rstdLocalTemp, axisKa_); // xLocalTensorFp32_ <- x * rstd
    PipeBarrier<PIPE_V>();
    if (rowIdx == startRowId) {
        SyncFunc<AscendC::HardEvent::MTE2_V>();
    }
    Mul(xLocalTensorFp32_, xLocalTensorFp32_, gammaTensor_, axisKa_); // xLocalTensorFp32_ <- x * rstd * gamma
    PipeBarrier<PIPE_V>();

    // DynamicQuant smooth
    Mul(yLocalTensorFp32_, xLocalTensorFp32_, smoothScaleTensor_, axisKa_); // y * smooth
    PipeBarrier<PIPE_V>();
 
    // scale
    float maxTemp;
    float scaleTemp;
    Abs(xLocalTensorFp32_, yLocalTensorFp32_, axisKa_); // xLocalTensorFp32_ <-- |y * smooth|
    PipeBarrier<PIPE_V>();
    ReduceMaxInplace(xLocalTensorFp32_, axisKa_);
    SyncFunc<AscendC::HardEvent::V_S>();
    maxTemp = xLocalTensorFp32_.GetValue(0); // Reduce
    scaleTemp = float(127.0) / maxTemp;
    dynamicScaleLocalTensor_.SetValue(rowIdx, 1 / scaleTemp);
    SyncFunc<AscendC::HardEvent::S_V>();
    Muls(yLocalTensorFp32_, yLocalTensorFp32_, scaleTemp, axisKa_);
    PipeBarrier<PIPE_V>();
}

template<TemplateMC2TypeClass>
__aicore__ inline void AddRmsNormDynamicQuantAllGatherQbmm<TemplateMC2TypeFunc>::Add2RmsNormDynamicQuantProcess()
{
    DataCopyEx(gammaTensor_, gammaGMTensor_, axisKa_);
    DataCopyEx(smoothScaleTensor_, smoothScaleGMTensor_, axisKa_);

    uint32_t startRowId = 0;
    uint32_t endRowId = 0;
    uint32_t rowNum = 0;
    SplitToCore(axisM_, aivNum_, startRowId, endRowId, rowNum);
    if (startRowId > axisM_) {
        return;
    }

    uint32_t rowNumSize = Ceil(rowNum * sizeof(float), UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(dynamicScaleBuf_, rowNumSize);
    dynamicScaleLocalTensor_ = dynamicScaleBuf_.Get<float>();

    int32_t gmOffset = 0;
    int32_t elementCount = axisKaAlignSize_ / sizeof(X1Type);
    for (int32_t rowIdx = startRowId; rowIdx < endRowId; ++rowIdx) {
        Add2RmsNormDynamicQuantCompute(gmOffset, rowIdx, startRowId, elementCount);
        // copy out 到win区?  修改
        x1OutLocalTensor_ = x1OutQueue_.AllocTensor<int8_t>();
        RoundFloat2Int8(x1OutLocalTensor_, yLocalTensorFp32_, axisKa_);
        x1OutQueue_.EnQue(x1OutLocalTensor_);
        x1OutLocalTensor_ = x1OutQueue_.DeQue<int8_t>();
        DataCopyEx(x1WinGMTensor_[gmOffset], x1OutLocalTensor_, axisKa_);
        x1OutQueue_.FreeTensor(x1OutLocalTensor_);
        gmOffset += axisKa_;
    }
    DataCopyEx(scaleWinGMTensor_[startRowId], dynamicScaleLocalTensor_, rowNum); //修改
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