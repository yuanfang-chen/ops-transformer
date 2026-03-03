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
#include "all_gather_mte.h"
#include "add_rms_norm_dynamic_quant_v2_helper.h"
#if __has_include("../common/inc/kernel/moe_distribute_base.h")
#include "../common/inc/kernel/moe_distribute_base.h"
#include "../common/inc/kernel/mc2_kernel_utils.h"
#else
#include "../../common/inc/kernel/moe_distribute_base.h"
#include "../../common/inc/kernel/mc2_kernel_utils.h"
#endif
#include "quant_batch_matmul_v3_base.h"

namespace AddRmsNormDynamicQuantAllGatherQbmmImpl {

#define TemplateMC2TypeClass typename X1Type, bool IsScaleExist, bool IsSmoothScaleExist
#define TemplateMC2TypeFunc X1Type, IsScaleExist, IsSmoothScaleExist
using namespace AscendC;
using namespace AllGatherImpl;

// 之后可修改成从tiling侧获取数据切块大小
constexpr static uint32_t X_PRE_BLOCK_NUM = 1024U;  // 当前一次搬运一个x数据块，x dtype为 8bit 时对应 1024个x数据. 对于fp4需要另外算
constexpr static uint64_t MX_SCALES_LAST_DIM = 2U; // MX量化scales最后一维的大小
constexpr static uint8_t BUFFER_NUM = 2; // 多Buf
constexpr static uint32_t UB_ALIGN = 32; // UB按32字节对齐
constexpr static uint32_t WIN_ALIGN = 512; // win offset 512字节对齐
constexpr static uint32_t SINGLE_CORE_K = 512; // Matmul切K轴后每份长度
constexpr static uint64_t SYNC_AIC_TO_AIV = 5;

template<TemplateMC2TypeClass>
class AddRmsNormDynamicQuantAllGatherQbmm {
public:
    __aicore__ inline AddRmsNormDynamicQuantAllGatherQbmm() {};
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR residual, GM_ADDR y, GM_ADDR gamma, GM_ADDR scale,
                                GM_ADDR smoothScale, GM_ADDR bias, GM_ADDR output, GM_ADDR z, GM_ADDR addRmsNormOut,
                                GM_ADDR dynamicQuantOut, GM_ADDR allGatherDataOut, GM_ADDR allGatherScalesOut,
                                GM_ADDR workspaceGM, TPipe *pipe,
                                const AddRmsNormDynamicQuantAllGatherQbmmTilingData *tilingData);
    __aicore__ inline void Process();

    using AMatmulType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, int8_t, false>;
    using BMatmulType = matmul::MatmulType<TPosition::GM, CubeFormat::NZ, int8_t, false>;
    using CMatmulType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, int32_t>;
    using BiasMatmulType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, int32_t>;
    matmul::MatmulImpl<AMatmulType, BMatmulType, CMatmulType, BiasMatmulType, MM_DEFAULT_MDL_CFG> mm_;

private:
    __aicore__ inline void InitBaseParams(const AddRmsNormDynamicQuantAllGatherQbmmTilingData *tilingData);
    __aicore__ inline void SplitToCore(uint32_t curSendCnt, uint32_t curUseAivNum, uint32_t &startId, uint32_t &endId, uint32_t &sendNum);
    __aicore__ inline void Add2RmsNormCompute(int32_t gmOffset, int32_t elementCount);
    __aicore__ inline void GammaWeightAndCopyOut(int32_t gmOffset);
    __aicore__ inline void DynamicQuant(int32_t offset);
    __aicore__ inline void Add2RmsNormDynamicQuantProcess();
    __aicore__ inline void CheckCvFlagReady(uint32_t mBlockIdx, uint32_t kBlockIdx);
    __aicore__ inline void CalcOffset(uint32_t nDimStartIdx, uint32_t mCoreIndx, uint32_t nCoreIndx);
    __aicore__ inline void SetWorkspace();
    __aicore__ inline void MMCompute(uint32_t singleCoreM, uint32_t singleCoreN, uint32_t kBlockIdx);
    __aicore__ inline void MatmulProcess();
    __aicore__ inline void InitTilingData(const AddRmsNormDynamicQuantAllGatherQbmmTilingData *tilingData);
    __aicore__ inline void DequantInit();
    __aicore__ inline void Bf16ScaleGm2Ub(LocalTensor<float> &scaleLocal, GlobalTensor<float> &scaleGm_,
        DataCopyPadParams padParams, uint64_t baseNOfffset, uint32_t curAivN);
    __aicore__ inline void DequantCompute(GlobalTensor<int32_t> &curMmOutGm, uint64_t baseMOfffset,
        uint64_t baseNOfffset, uint32_t curAicM, uint32_t curAicN);
    __aicore__ inline void DequantProcess();

    TPipe *tpipe_{nullptr};
    AllGatherMte<int8_t, float, int8_t> allGatherMte_;  // allGather 相关实现
    GlobalTensor<X1Type> x1GMTensor_;
    GlobalTensor<int8_t> x2GMTensor_;
    GlobalTensor<X1Type> residualGMTensor_;
    GlobalTensor<X1Type> yGMTensor_;
    GlobalTensor<float> gammaGMTensor_;
    GlobalTensor<float> scaleGMTensor_; // 类型确定
    GlobalTensor<float> smoothScaleGMTensor_;
    GlobalTensor<int32_t> biasGMTensor_; // 类型确定
    GlobalTensor<int8_t> x1WinGMTensor_; // 类型确定
    GlobalTensor<float> scaleWinGMTensor_; // 类型确定
    GlobalTensor<X1Type> outputGMTensor_;
    GlobalTensor<X1Type> zGMTensor_;
    GlobalTensor<X1Type> addRmsNormOutGMTensor_;   // TODO: 仅调试用
    GlobalTensor<int8_t> dynamicQuantOutGMTensor_;   // TODO: 仅调试用
    GlobalTensor<int8_t> allGatherDataOutGMTensor_;   // TODO: 仅调试用
    GlobalTensor<float> allGatherScalesOutGMTensor_;   // TODO: 仅调试用
    GM_ADDR allGatherDataOutAddr_;    // TODO: 仅调试用
    GM_ADDR allGatherScalesOutAddr_;    // TODO: 仅调试用
    
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
    TBuf<> stateResetBuf_;

    TQue<QuePosition::VECIN, 1> inQueue_;
    TQue<QuePosition::VECOUT, 1> x1OutQueue_;
    TQue<QuePosition::VECOUT, 1> zOutQueue_;
    TQue<QuePosition::VECOUT, 1> addRmsNormOutQueue_;

    uint32_t aivId_{0};
    uint32_t aicId_{0};
    uint32_t rankId_{0};
    uint32_t axisM_{0};
    uint32_t axisKa_{0};
    uint32_t axisN_{0};
    uint32_t aivNum_{0};
    uint32_t rankSize_{0};
    uint32_t tileK_{0};
    uint32_t sendCoreNumPerRank_{0};
    float eps_{0};
    float aveNum_{0};
    uint64_t axisKaAlignSize_{0};
    uint64_t axisKaAlignFloatSize_{0};
    uint64_t axisKaAlignInt8Size_{0};

    __gm__ HcclOpResParam *winContext_{nullptr};

    uint32_t blockIdx_;
    uint32_t m_;
    uint32_t n_;
    uint32_t k_;
    uint32_t singleCoreM_;
    uint32_t singleCoreN_;
    uint32_t singleTimeM_;
    uint32_t singleTimeN_;
    uint32_t singleCoreK_;
    uint32_t usedCoreNum_;
    uint32_t baseM_;
    uint32_t baseN_;
    uint32_t baseK_;
    // bool isMouter_;

    // vector dequant
    uint32_t ubCalcM_;
    uint32_t ubCalcN_;
    uint32_t ubTmpBuffer_;

    uint64_t offsetA_ = 0;
    uint64_t offsetB_ = 0;
    uint64_t offsetC_ = 0;
    uint64_t offsetBias_ = 0;
    uint64_t offsetScale_ = 0;
    uint64_t offsetPertokenScale_ = 0;

    GlobalTensor<int32_t> mmOutGm_;
    GlobalTensor<int32_t> workspaceGm_;
    GM_ADDR workspaceAddr_;
    // TODO: 手动管理CO1，用FIXP做NZ2ND
    // TQue<QuePosition::CO1, 1> co1Queue_;
    // LocalTensor<int32_t> l0cTensor_;

    // define the que
    TQue<QuePosition::VECIN, 1> vecQueSrc_;
    TQue<QuePosition::VECIN, 1> vecQueScale_;
    TQue<QuePosition::VECIN, 1> vecQuePertokenScale_;
    TQue<QuePosition::VECIN, 1> vecQueBias_;
    TBuf<TPosition::VECCALC> vecQueTmp_;
    TQue<QuePosition::VECOUT, 1> vecQueOut_;
    TBuf<TPosition::VECCALC> broadcastFp32Tmp_;
    // used when bias type is bf16/fp16/fp32, deqaunt result should be fp32
    TBuf<TPosition::VECCALC> biasFp32Tmp_;
    TBuf<TPosition::VECCALC> outFp32Tmp_;

    const AddRmsNormDynamicQuantAllGatherQbmmTilingData *tilingData_;
};

template<TemplateMC2TypeClass>
__aicore__ inline void AddRmsNormDynamicQuantAllGatherQbmm<TemplateMC2TypeFunc>::Init(
    GM_ADDR x1, GM_ADDR x2, GM_ADDR residual, GM_ADDR y, GM_ADDR gamma, GM_ADDR scale, GM_ADDR smoothScale,
    GM_ADDR bias, GM_ADDR output, GM_ADDR z, GM_ADDR addRmsNormOut, GM_ADDR dynamicQuantOut,
    GM_ADDR allGatherDataOut, GM_ADDR allGatherScalesOut, GM_ADDR workspaceGM, TPipe *pipe,
    const AddRmsNormDynamicQuantAllGatherQbmmTilingData *tilingData)
{
    tpipe_ = pipe;
    InitBaseParams(tilingData);

    x1GMTensor_.SetGlobalBuffer((__gm__ X1Type*)x1);
    x2GMTensor_.SetGlobalBuffer((__gm__ int8_t*)x2);
    residualGMTensor_.SetGlobalBuffer((__gm__ X1Type*)residual);
    yGMTensor_.SetGlobalBuffer((__gm__ X1Type*)y);
    gammaGMTensor_.SetGlobalBuffer((__gm__ float*)gamma);
    // 可选输入
    scaleGMTensor_.SetGlobalBuffer((__gm__ float*)scale);
    smoothScaleGMTensor_.SetGlobalBuffer((__gm__ float*)smoothScale);
    biasGMTensor_.SetGlobalBuffer((__gm__ int32_t*)bias);
    // 输出
    outputGMTensor_.SetGlobalBuffer((__gm__ X1Type*)output);
    zGMTensor_.SetGlobalBuffer((__gm__ X1Type*)z);
    addRmsNormOutGMTensor_.SetGlobalBuffer((__gm__ X1Type*)addRmsNormOut);   // TODO: 仅调试用
    dynamicQuantOutGMTensor_.SetGlobalBuffer((__gm__ int8_t*)dynamicQuantOut);   // TODO: 仅调试用
    allGatherDataOutGMTensor_.SetGlobalBuffer((__gm__ int8_t*)allGatherDataOut);   // TODO: 仅调试用
    allGatherScalesOutGMTensor_.SetGlobalBuffer((__gm__ float*)allGatherScalesOut);   // TODO: 仅调试用
    
    allGatherDataOutAddr_ = allGatherDataOut;   // TODO: 仅调试用
    allGatherScalesOutAddr_ = allGatherScalesOut;   // TODO: 仅调试用

    // tpipe_->InitBuffer(inQueue_, BUFFER_NUM, 3 * axisKaAlignSize_); // 修改
    tpipe_->InitBuffer(inQueue_, BUFFER_NUM, 3 * axisKaAlignSize_); // 修改
    tpipe_->InitBuffer(zOutQueue_, BUFFER_NUM, axisKaAlignSize_); // 修改
    tpipe_->InitBuffer(addRmsNormOutQueue_, BUFFER_NUM, axisKaAlignSize_); // 修改
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

    InitTilingData(tilingData);
    tilingData_ = tilingData;
    workspaceAddr_ = workspaceGM;
    mmOutGm_.SetGlobalBuffer((__gm__ int32_t*)output);
    workspaceGm_.SetGlobalBuffer((__gm__ int32_t*)workspaceAddr_);

    if ASCEND_IS_AIC {
        return;
    }

    // 清空状态区100K之后的2*tileK*64B数据
    if (aivId_ == 0) {
        // 已保证32B对齐
        uint64_t sizeToBeCleaned = tileK_ * CV_STATE_ROW_NUM * CV_STATE_ALIGN;
        LocalTensor<int32_t> stateResetTensor;
        tpipe_->InitBuffer(stateResetBuf_, sizeToBeCleaned);
        stateResetTensor = stateResetBuf_.Get<int32_t>();
        Duplicate<int32_t>(stateResetTensor, 0, sizeToBeCleaned / sizeof(int32_t));
        SyncFunc<AscendC::HardEvent::V_MTE3>();
        GM_ADDR cvFlagAddr = (GM_ADDR)(winContext_->localWindowsExp) + CV_SYNC_START_OFFSET;
        GlobalTensor<int32_t> winCvExpTensor;
        winCvExpTensor.SetGlobalBuffer((__gm__ int32_t *)cvFlagAddr);
        DataCopy(winCvExpTensor, stateResetTensor, sizeToBeCleaned / sizeof(int32_t));
        PipeBarrier<PIPE_V>();
    }
}

template<TemplateMC2TypeClass>
__aicore__ inline void AddRmsNormDynamicQuantAllGatherQbmm<TemplateMC2TypeFunc>::InitBaseParams(const AddRmsNormDynamicQuantAllGatherQbmmTilingData *tilingData)
{
    aivId_ = GetBlockIdx();
    aicId_ = aivId_;    // CV核1:1
    winContext_ = (__gm__ HcclOpResParam *)AscendC::GetHcclContext<HCCL_GROUP_ID_0>();
    rankId_ = winContext_->localUsrRankId;

    axisM_ = tilingData->addRmsNormDynamicQuantAllGatherTilingData.M;
    axisKa_ = tilingData->addRmsNormDynamicQuantAllGatherTilingData.Ka;
    axisN_ = tilingData->addRmsNormDynamicQuantAllGatherTilingData.N;
    aivNum_ = tilingData->addRmsNormDynamicQuantAllGatherTilingData.aivNum;
    rankSize_ = tilingData->addRmsNormDynamicQuantAllGatherTilingData.rankSize;
    eps_ = tilingData->addRmsNormDynamicQuantAllGatherTilingData.epsilon;
    aveNum_ = tilingData->addRmsNormDynamicQuantAllGatherTilingData.avgFactor;

    tileK_ = Ceil(axisKa_, SINGLE_CORE_K);
    axisKaAlignSize_ = Ceil(axisKa_ * sizeof(X1Type), UB_ALIGN) * UB_ALIGN;
    axisKaAlignFloatSize_ = Ceil(axisKa_ * sizeof(float), UB_ALIGN) * UB_ALIGN;
    axisKaAlignInt8Size_ = Ceil(axisKa_ * sizeof(int8_t), UB_ALIGN) * UB_ALIGN;

    sendCoreNumPerRank_ = aivNum_ / rankSize_;
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

template<TemplateMC2TypeClass>
__aicore__ inline void AddRmsNormDynamicQuantAllGatherQbmm<TemplateMC2TypeFunc>::DynamicQuant(int32_t offset)
{
    // smooth
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
    dynamicScaleLocalTensor_.SetValue(offset, 1 / scaleTemp);
    SyncFunc<AscendC::HardEvent::S_V>();
    Muls(yLocalTensorFp32_, yLocalTensorFp32_, scaleTemp, axisKa_);
    PipeBarrier<PIPE_V>();
}

template<TemplateMC2TypeClass>
__aicore__ inline void AddRmsNormDynamicQuantAllGatherQbmm<TemplateMC2TypeFunc>::Add2RmsNormCompute(int32_t gmOffset, int32_t elementCount)
{
    // Copy In
    LocalTensor<X1Type> copyInLocalTensor = inQueue_.AllocTensor<X1Type>();
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

    // Copy out z
    LocalTensor<X1Type> zOutLocalTensor = zOutQueue_.AllocTensor<X1Type>();
    Cast(zOutLocalTensor, xLocalTensorFp32_, RoundMode::CAST_RINT, axisKa_);
    PipeBarrier<PIPE_V>();
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
    SyncFunc<AscendC::HardEvent::S_V>();
    Muls(xLocalTensorFp32_, xLocalTensorFp32_, rstdLocalTemp, axisKa_); // xLocalTensorFp32_ <- x * rstd
    PipeBarrier<PIPE_V>();
}

template<TemplateMC2TypeClass>
__aicore__ inline void AddRmsNormDynamicQuantAllGatherQbmm<TemplateMC2TypeFunc>::GammaWeightAndCopyOut(int32_t gmOffset)
{
    // Gamma Weight
    Mul(xLocalTensorFp32_, xLocalTensorFp32_, gammaTensor_, axisKa_); // xLocalTensorFp32_ <- x * rstd * gamma
    PipeBarrier<PIPE_V>();

    // CopyOut addRmsNormOut
    LocalTensor<X1Type> addRmsNormOutLocalTensor = addRmsNormOutQueue_.AllocTensor<X1Type>();
    if constexpr (is_same<X1Type, half>::value) {
        Cast(addRmsNormOutLocalTensor, xLocalTensorFp32_, RoundMode::CAST_NONE, axisKa_);
    } else { // BF16
        Cast(addRmsNormOutLocalTensor, xLocalTensorFp32_, RoundMode::CAST_RINT, axisKa_);
    }
    PipeBarrier<PIPE_V>();
    addRmsNormOutQueue_.EnQue(addRmsNormOutLocalTensor);
    addRmsNormOutLocalTensor = addRmsNormOutQueue_.DeQue<X1Type>();
    DataCopyEx(addRmsNormOutGMTensor_[gmOffset], addRmsNormOutLocalTensor, axisKa_);
    addRmsNormOutQueue_.FreeTensor<X1Type>(addRmsNormOutLocalTensor);
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

    int32_t gmOffset = startRowId * axisKa_;
    int32_t elementCount = axisKaAlignSize_ / sizeof(X1Type);
    for (int32_t rowIdx = startRowId; rowIdx < endRowId; ++rowIdx) {
        Add2RmsNormCompute(gmOffset, elementCount);
        if (rowIdx == startRowId) {
            SyncFunc<AscendC::HardEvent::MTE2_V>(); // wait gammaTensor_ and smoothScaleTensor_
        }
        GammaWeightAndCopyOut(gmOffset);
        DynamicQuant(rowIdx - startRowId);
        // copy out 到本卡win区
        x1OutLocalTensor_ = x1OutQueue_.AllocTensor<int8_t>();
        RoundFloat2Int8(x1OutLocalTensor_, yLocalTensorFp32_, axisKa_);
        x1OutQueue_.EnQue(x1OutLocalTensor_);
        x1OutLocalTensor_ = x1OutQueue_.DeQue<int8_t>();
        DataCopyEx(x1WinGMTensor_[gmOffset], x1OutLocalTensor_, axisKa_);
        // copy out to dynamicQuantOut, too
        DataCopyEx(dynamicQuantOutGMTensor_[gmOffset], x1OutLocalTensor_, axisKa_);
        x1OutQueue_.FreeTensor(x1OutLocalTensor_);
        gmOffset += axisKa_;
    }
    SyncFunc<AscendC::HardEvent::S_MTE3>();
    DataCopyEx(scaleWinGMTensor_[startRowId], dynamicScaleLocalTensor_, rowNum);
}

template<TemplateMC2TypeClass>
__aicore__ inline void AddRmsNormDynamicQuantAllGatherQbmm<TemplateMC2TypeFunc>::CheckCvFlagReady(
    uint32_t mBlockIdx, uint32_t kBlockIdx)
{
    GlobalTensor<int32_t> winCvExp;
    GM_ADDR cvFlagAddr = allGatherMte_.CalcCvFlagAddr(mBlockIdx, kBlockIdx);
    winCvExp.SetGlobalBuffer((__gm__ int32_t *)cvFlagAddr);
    int32_t targetCount = axisM_ * rankSize_ / CV_STATE_ROW_NUM;
    while (true) {
        DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(winCvExp);
        int32_t flagCount = winCvExp.GetValue(0);
        if (flagCount == targetCount) {
            break;
        }
        __asm__ volatile("NOP");    // Necessary
    }
}

// template<TemplateMC2TypeClass>
// __aicore__ inline void AddRmsNormDynamicQuantAllGatherQbmm<TemplateMC2TypeFunc>::MatmulProcess()
// {
//     mm_.SetOrgShape(252, 3072, 5120); // m_, n_, k_
//     mm_.SetTail(252, 128, 512); // singleCoreM, singleCoreN, singleCoreK

//     // mm计算: x和pertoken_scale在本端win区，weight和scale在inputGM
//     GlobalTensor<int8_t> x1WinGlobalTensor;
//     // GM_ADDR localDataGm = (GM_ADDR)(winContext_->localWindowsIn);
//     // x1WinGlobalTensor.SetGlobalBuffer((__gm__ int8_t*)localDataGm);
//     x1WinGlobalTensor.SetGlobalBuffer((__gm__ int8_t*)outputAddr_);
//     uint64_t offsetA = 0;
//     uint64_t offsetB = aicId_ * 128 * 5120; // baseN * k_
//     uint32_t targetRankIdx = aicId_ / tileM_;
//     uint32_t mBlockIdx = aicId_ % tileM_;
//     for (uint32_t kBlockIdx = 0; kBlockIdx < tileK_; kBlockIdx++) {
//         CheckCvFlagReady(targetRankIdx, mBlockIdx, kBlockIdx);
//         mm_.SetTensorA(x1WinGlobalTensor[offsetA], false);
//         mm_.SetTensorB(x2GMTensor_[offsetB], false);
//         // 暂不支持bias
//         // if (hasBias_ != 0 && biasDtype_ == DT_INT32) {
//         //     mm_.SetBias(biasGmInt32_[offsetBias_]);
//         // }
//         mm_.template Iterate<false>(true); // <sync=false>(enPartialSum=true)
//         offsetA += 512; // baseK
//         offsetB += (512 * 32); // baseK * n0
//     }

//     mmOutGm = mm_.GetTensorC(); // 获取异步场景用于缓存结果的Workspace上的C矩阵
//     PipeBarrier<PIPE_ALL>();
//     mm_.End();

//     // 通知AIV
//     CrossCoreSetFlag<0x2, PIPE_FIX>(SYNC_AIC_TO_AIV);
// }

template<TemplateMC2TypeClass>
__aicore__ inline void AddRmsNormDynamicQuantAllGatherQbmm<TemplateMC2TypeFunc>::CalcOffset(
    uint32_t nDimStartIdx, uint32_t mCoreIndx, uint32_t nCoreIndx)
{
    uint64_t mOffset = static_cast<uint64_t>(mCoreIndx * singleCoreM_);
    uint64_t nOffset = static_cast<uint64_t>(nCoreIndx * singleCoreN_);

    // AMatmulType::format == CubeFormat::ND
    offsetA_ = mOffset * k_;

    // BMatmulType::format == CubeFormat::NZ
    offsetB_ = DequantBmm::Align(nOffset, K0_INT8) * DequantBmm::Align(k_, BMM_BLOCK_NUM) +
        nDimStartIdx * DequantBmm::Align(singleCoreN_, K0_INT8) * DequantBmm::Align(k_, BMM_BLOCK_NUM);

    // the output of mm only support ND/ND_ALIGN for vector
    offsetC_ = nDimStartIdx * static_cast<uint64_t>(singleCoreN_) + mOffset * n_ + nOffset;

    offsetPertokenScale_ = mOffset;
    offsetScale_ = nDimStartIdx * static_cast<uint64_t>(singleCoreN_) + nOffset;
}

template<TemplateMC2TypeClass>
__aicore__ inline void AddRmsNormDynamicQuantAllGatherQbmm<TemplateMC2TypeFunc>::MMCompute(
    uint32_t singleCoreM, uint32_t singleCoreN, uint32_t kBlockIdx)
{
    // mm计算: x和pertoken_scale在本端win区，weight和scale在inputGM
    GlobalTensor<int8_t> x1WinGlobalTensor;
    GM_ADDR localDataGm = (GM_ADDR)(winContext_->localWindowsIn);
    x1WinGlobalTensor.SetGlobalBuffer((__gm__ int8_t*)localDataGm);
    // x1WinGlobalTensor.SetGlobalBuffer((__gm__ int8_t*)outputAddr_);

    mm_.SetTail(singleCoreM, singleCoreN, singleCoreK_); // singleCoreK=512
    mm_.SetSingleShape(singleCoreM, singleCoreN, singleCoreK_); // singleCoreK=512
    mm_.SetTensorA(x1WinGlobalTensor[offsetA_], false);
    mm_.SetTensorB(x2GMTensor_[offsetB_], false);
    mm_.DisableBias();
    // 暂不支持bias
    // if (hasBias_ != 0 && biasDtype_ == DT_INT32) {
    //     mm_.SetBias(biasGmInt32_[offsetBias_]);
    // }

    if (kBlockIdx == 0) {
        mm_.template Iterate<false>(false); // <sync=false>(enPartialSum=false)
    } else {
        mm_.template Iterate<false>(true); // <sync=false>(enPartialSum=true)
    }

    // if (kBlockIdx == 0) {
    //     mm_.template Iterate<false>(false, l0cTensor_); // <sync=false>(enPartialSum=false, l0cTensor_)
    // } else {
    //     mm_.template Iterate<false>(true, l0cTensor_); // <sync=false>(enPartialSum=false, l0cTensor_)
    // }
}

template<TemplateMC2TypeClass>
__aicore__ inline void AddRmsNormDynamicQuantAllGatherQbmm<TemplateMC2TypeFunc>::MMGetCo1Tensor(
    uint32_t singleCoreM, uint32_t singleCoreN)
{}

template<TemplateMC2TypeClass>
__aicore__ inline void AddRmsNormDynamicQuantAllGatherQbmm<TemplateMC2TypeFunc>::MatmulProcess()
{
    uint32_t mDim = DequantBmm::CeilDiv(m_, singleCoreM_);
    uint32_t nDimNeed = DequantBmm::CeilDiv(n_, singleCoreN_);
    uint32_t nDimReal = usedCoreNum_ / mDim;
    uint32_t nDimLoops = DequantBmm::CeilDiv(nDimNeed, nDimReal);

    uint32_t mCoreIndx = blockIdx_ % mDim;  // 必须沿着N 轴方向输出
    uint32_t nCoreIndx = blockIdx_ / mDim;

    uint32_t gmUseM = m_ - mCoreIndx * singleCoreM_;
    uint32_t singleCoreMUpdate = gmUseM < singleCoreM_ ? gmUseM : singleCoreM_;
    uint32_t gmUseN = n_ - nCoreIndx * singleCoreN_;
    uint32_t singleCoreNUpdate = gmUseN < singleCoreN_ ? gmUseN : singleCoreN_;

    mm_.SetOrgShape(m_, n_, k_); // 252, 3072, 5120

    uint32_t targetRankIdx = aicId_ / tileM_;
    uint32_t mBlockIdx = aicId_ % tileM_;
    CalcOffset(0, mCoreIndx, nCoreIndx);
    SetWorkspace();
    uint32_t mBlockIdx = aicId_ % CV_STATE_ROW_NUM;
    for (uint32_t kBlockIdx = 0; kBlockIdx < tileK_; kBlockIdx++) {
        CheckCvFlagReady(mBlockIdx, kBlockIdx);
        // enPartialSum 要求 singleCoreM == baseM, singleCoreN == baseN（当前N方向没有尾块）
        MMCompute(singleCoreM_, singleCoreNUpdate, kBlockIdx);
        offsetA_ += singleCoreK_; // 512
        offsetB_ += (singleCoreK_ * K0_INT8); // 512*32
    }
    mm_.GetTensorC<false>(mmOutGm_[offsetC_]);
    CrossCoreSetFlag<0x2, PIPE_FIX>(SYNC_AIC_TO_AIV);

    for (uint32_t nDimLoopIdx = 1; nDimLoopIdx < nDimLoops; nDimLoopIdx++) {
        CalcOffset(nDimLoopIdx * nDimReal, mCoreIndx, nCoreIndx);
        SetWorkspace();
        for (uint32_t kBlockIdx = 0; kBlockIdx < tileK_; kBlockIdx++) {
            // enPartialSum 要求 singleCoreM == baseM, singleCoreN == baseN（当前N方向没有尾块）
            MMCompute(singleCoreM_, singleCoreNUpdate, kBlockIdx);
            offsetA_ += singleCoreK_; // 512
            offsetB_ += (singleCoreK_ * K0_INT8); // 512*32
        }
        mm_.GetTensorC<false>(mmOutGm_[offsetC_]);
        CrossCoreSetFlag<0x2, PIPE_FIX>(SYNC_AIC_TO_AIV);
    }

    PipeBarrier<PIPE_ALL>();
    mm_.End();
}

template<TemplateMC2TypeClass>
__aicore__ inline void AddRmsNormDynamicQuantAllGatherQbmm<TemplateMC2TypeFunc>::InitTilingData(
    const AddRmsNormDynamicQuantAllGatherQbmmTilingData *tilingData)
{
    blockIdx_ = GetBlockIdx();
    blockIdx_ /= GetTaskRation();

    m_ = tilingData->matmulTiling.M;
    n_ = tilingData->matmulTiling.N;
    k_ = tilingData->matmulTiling.Ka;
    singleTimeM_ = tilingData->matmulTiling.singleCoreM;  // calcM of each mm iterate
    singleTimeN_ = tilingData->matmulTiling.singleCoreN;  // calcN of each mm iterate
    singleCoreK_ = tilingData->matmulTiling.singleCoreK;
    usedCoreNum_ = tilingData->matmulTiling.usedCoreNum;
    singleCoreM_ = singleTimeM_;
    singleCoreN_ = singleTimeN_;

    baseM_ = tilingData->matmulTiling.baseM;
    baseN_ = tilingData->matmulTiling.baseN;
    baseK_ = tilingData->matmulTiling.baseK;
    // ubCalcM_ = tilingData->qbmmParams.ubCalcM;
    // ubCalcN_ = tilingData->qbmmParams.ubCalcN;
    // ubTmpBuffer_ = tilingData->qbmmParams.needUbBuffer;
    ubCalcM_ = 8;
    ubCalcN_ = baseN_;
    ubTmpBuffer_ = BUFFER_NUM * ubCalcM_ * ubCalcN_ * sizeof(float);
}

template<TemplateMC2TypeClass>
__aicore__ inline void AddRmsNormDynamicQuantAllGatherQbmm<TemplateMC2TypeFunc>::DequantInit()
{
    // blockIdx_ = GetBlockIdx();
    // blockIdx_ /= GetTaskRation();

    // InitTilingData(tilingData);
    if (blockIdx_ >= usedCoreNum_ || GetSubBlockIdx() == 1) {
        return;
    }
    // init global buffer
    // UpdateGlobalAddr(x1, x2, bias, scale, pertokenScale, y, workSpace);
    // init ub local buffer
    tpipe_->Reset();
    tpipe_->InitBuffer(vecQueSrc_, BUFFER_NUM, ubCalcM_ * ubCalcN_ * sizeof(int32_t));
    tpipe_->InitBuffer(vecQueTmp_, ubTmpBuffer_);
    tpipe_->InitBuffer(vecQueOut_, BUFFER_NUM, ubCalcM_ * ubCalcN_ * sizeof(X1Type)); // yType=X1Type
    // if (biasDtype_ != DT_INT32) {
    //     tpipe_->InitBuffer(biasFp32Tmp_, ubCalcN_ * sizeof(float));
    //     tpipe_->InitBuffer(vecQueBias_, BUFFER_NUM, ubCalcN_ * biasDtypeSize_);
    // }
    // if (!isPerTensor_) {
    //     tpipe_->InitBuffer(vecQueScale_, BUFFER_NUM, ubCalcN_ * sizeof(scaleType));
    // }
    tpipe_->InitBuffer(vecQueScale_, BUFFER_NUM, ubCalcN_ * sizeof(float));
    // pertoken
    tpipe_->InitBuffer(vecQuePertokenScale_, BUFFER_NUM, DequantBmm::Align(ubCalcM_, 8U) * sizeof(float));
    tpipe_->InitBuffer(broadcastFp32Tmp_, ubCalcM_ * ubCalcN_ * sizeof(float));
    tpipe_->InitBuffer(outFp32Tmp_, ubCalcM_ * ubCalcN_ * sizeof(float));
}

template<TemplateMC2TypeClass>
__aicore__ inline void AddRmsNormDynamicQuantAllGatherQbmm<TemplateMC2TypeFunc>::Bf16ScaleGm2Ub(
    LocalTensor<float> &scaleLocal, GlobalTensor<float> &scaleGm_,
    DataCopyPadParams padParams, uint64_t baseNOfffset, uint32_t curAivN)
{
    DataCopyParams scale2UbParams{1, 0, 0, 0};
    scale2UbParams.blockLen = curAivN * sizeof(float);
    uint64_t scaleOffset = offsetScale_ + baseNOfffset;
    DataCopyPad(scaleLocal, scaleGm_[scaleOffset], scale2UbParams, padParams);
}

template<TemplateMC2TypeClass>
__aicore__ inline void AddRmsNormDynamicQuantAllGatherQbmm<TemplateMC2TypeFunc>::DequantCompute(
    GlobalTensor<int32_t> &curMmOutGm, uint64_t baseMOfffset,
    uint64_t baseNOfffset, uint32_t curAicM, uint32_t curAicN)
{
    LocalTensor<float> dstLocalFp32;
    LocalTensor<float> biasFp32;
    LocalTensor<bfloat16_t> oriBiasBf16;
    LocalTensor<half> oriBiasFp16;
    LocalTensor<float> oriBiasFp32;
    uint32_t curAivM = ubCalcM_;
    // calcN in ub is equal to aicN
    uint32_t curAivN = curAicN;
    uint32_t mUbLoops = DequantBmm::CeilDiv(curAicM, ubCalcM_);
    DataCopyParams gm2UbParams{1, 0, 0, 0};
    DataCopyExtParams ub2GmParams{1, 0, 0, 0, 0};
    DataCopyPadParams padParams;
    DequantParams dequantParams;
    DequantBmm::CalcDequantParams(mUbLoops == 1 ? curAicM : ubCalcM_, curAicN, dequantParams);
    dstLocalFp32 = outFp32Tmp_.Get<float>();
    for (uint32_t mUbLoopIdx = 0; mUbLoopIdx < mUbLoops; ++mUbLoopIdx) {
        if (mUbLoopIdx == mUbLoops - 1) {
            curAivM = curAicM - ubCalcM_ * (mUbLoops - 1);
            DequantBmm::CalcDequantParams(curAivM, curAicN, dequantParams, mUbLoops != 1 && curAivM != ubCalcM_);
        }
        LocalTensor<int32_t> srcLocal = vecQueSrc_.AllocTensor<int32_t>();
        LocalTensor<X1Type> dstLocal = vecQueOut_.AllocTensor<X1Type>();
        LocalTensor<uint8_t> tmpLocal = vecQueTmp_.Get<uint8_t>();
        // datacopypad 32B aligned
        gm2UbParams.blockLen = curAivN * sizeof(int32_t);
        gm2UbParams.blockCount = curAivM;
        gm2UbParams.srcStride = (n_ - curAivN) * sizeof(int32_t);
        uint32_t curAicAivOffset = mUbLoopIdx * ubCalcM_ * n_;
        DataCopyPad(srcLocal, mmOutGm_[offsetC_ + curAicAivOffset], gm2UbParams, padParams);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        // if (biasDtype_ != DT_INT32) {
        //     BiasTensorInit(dstLocalFp32, biasFp32, oriBiasBf16, oriBiasFp16, oriBiasFp32);
        //     BiasGm2Ub(oriBiasBf16, oriBiasFp16, oriBiasFp32, padParams, baseNOfffset, curAicN);
        // }
        // if (isPerTensor_) {
        //     AscendDequant(dstLocalFp32, srcLocal, scaleScalar_, tmpLocal, dequantParams);
        // } else {
            LocalTensor<float> scaleLocal = vecQueScale_.AllocTensor<float>();
            Bf16ScaleGm2Ub(scaleLocal, scaleGMTensor_, padParams, baseNOfffset, curAicN);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
            AscendDequant(dstLocalFp32, srcLocal, scaleLocal, tmpLocal, dequantParams);
            vecQueScale_.FreeTensor(scaleLocal);
        // }

        DataCopyParams scale2UbParams{1, 0, 0, 0};
        scale2UbParams.blockLen = curAivM * sizeof(float);
        uint64_t scaleOffset = offsetPertokenScale_ + mUbLoopIdx * ubCalcM_ + baseMOfffset / n_;

        uint32_t computedAivN = DequantBmm::Align(curAivN, 8U);  // 8: 32B aligned for float
        uint32_t ubResAlignedN = DequantBmm::Align(curAivN);     // 16: sizeof(yType) is 2, 32B / 2
        const uint32_t broadCastDst[M_N_TWO_DIMS] = {curAivM, computedAivN};
        const uint32_t broadCastSrc[M_N_TWO_DIMS] = {curAivM, 1};

        LocalTensor<float> broadcastFp32 = broadcastFp32Tmp_.Get<float>();
        LocalTensor<float> pertokenScaleLocal = vecQuePertokenScale_.AllocTensor<float>();

        GlobalTensor<float> pertokenScaleGm;
        // GM_ADDR pertokenScaleGmAddr = (GM_ADDR)((winContext_->localWindowsIn) + 252*5120);
        // pertokenScaleGlobalTensor.SetGlobalBuffer((__gm__ float*)pertokenScaleGmAddr);
        pertokenScaleGm.SetGlobalBuffer((__gm__ float*)zAddr_);
        DataCopyPad(pertokenScaleLocal, pertokenScaleGm[scaleOffset], scale2UbParams, padParams);
        vecQuePertokenScale_.EnQue<float>(pertokenScaleLocal);
        pertokenScaleLocal = vecQuePertokenScale_.DeQue<float>();

        BroadCast<float, M_N_TWO_DIMS, 1>(broadcastFp32, pertokenScaleLocal, broadCastDst, broadCastSrc);

        AscendC::PipeBarrier<PIPE_V>();
        LocalTensor<float> tmpdstLocal = vecQueTmp_.Get<float>();
        if (computedAivN == ubResAlignedN) {
            Mul(tmpdstLocal, broadcastFp32, dstLocalFp32, computedAivN * curAivM);
        } else {
            for (auto i = 0; i < curAivM; i++) {
                Mul(tmpdstLocal[ubResAlignedN * i], broadcastFp32[computedAivN * i], dstLocalFp32[computedAivN * i],
                    computedAivN);
            }
        }
        vecQuePertokenScale_.FreeTensor(pertokenScaleLocal);

        // if (biasDtype_ != DT_INT32) {
        //     CalBiasAdd(tmpdstLocal, biasFp32, oriBiasBf16, oriBiasFp16, oriBiasFp32, curAivN, curAivM);
        // }
        AscendC::PipeBarrier<PIPE_V>();
        Cast(dstLocal, tmpdstLocal, RoundMode::CAST_RINT, curAivM * ubResAlignedN);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID2);
        vecQueSrc_.FreeTensor(srcLocal);
        // dst from ub -> gm
        ub2GmParams.blockLen = curAivN * sizeof(X1Type); // yType=X1Type
        ub2GmParams.blockCount = curAivM;
        ub2GmParams.dstStride = (n_ - curAivN) * sizeof(X1Type); // yType=X1Type
        uint64_t aivOffset = mUbLoopIdx * ubCalcM_ * n_;
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID2);
        DataCopyPad(outputGMTensor_[offsetC_ + baseMOfffset + baseNOfffset + aivOffset], dstLocal, ub2GmParams);
        vecQueOut_.FreeTensor(dstLocal);
    }
}

template<TemplateMC2TypeClass>
__aicore__ inline void AddRmsNormDynamicQuantAllGatherQbmm<TemplateMC2TypeFunc>::DequantProcess()
{
    uint32_t mDim = DequantBmm::CeilDiv(m_, singleCoreM_);
    uint32_t nDimNeed = DequantBmm::CeilDiv(n_, singleCoreN_);
    uint32_t nDimReal = usedCoreNum_ / mDim;
    uint32_t nDimLoops = DequantBmm::CeilDiv(nDimNeed, nDimReal);

    uint32_t mCoreIndx = blockIdx_ % mDim;  // 必须沿着N 轴方向输出
    uint32_t nCoreIndx = blockIdx_ / mDim;

    uint32_t gmUseM = m_ - mCoreIndx * singleCoreM_;
    uint32_t singleCoreMUpdate = gmUseM < singleCoreM_ ? gmUseM : singleCoreM_;
    uint32_t gmUseN = n_ - nCoreIndx * singleCoreN_;
    uint32_t singleCoreNUpdate = gmUseN < singleCoreN_ ? gmUseN : singleCoreN_;

    DequantInit();
    for (uint32_t nDimLoopIdx = 0; nDimLoopIdx < nDimLoops; nDimLoopIdx++) {
        CalcOffset(nDimLoopIdx * nDimReal, mCoreIndx, nCoreIndx);
        CrossCoreWaitFlag(SYNC_AIC_TO_AIV);
        // DequantCompute(mmOutGm, 0, 0, singleCoreMUpdate, singleCoreNUpdate);
    }
}

template<TemplateMC2TypeClass>
__aicore__ inline void AddRmsNormDynamicQuantAllGatherQbmm<TemplateMC2TypeFunc>::Process()
{
    if ASCEND_IS_AIV {
        Add2RmsNormDynamicQuantProcess();
        SyncAll<true>();
        PipeBarrier<PIPE_MTE3>();
        tpipe_->Reset();
        allGatherMte_.Init(tpipe_, axisM_, axisKa_, aivNum_, rankSize_);
        allGatherMte_.SetRemoteFlag();
        allGatherMte_.WaitRemoteFlag();
        allGatherMte_.ExecuteAllGather(allGatherDataOutAddr_, allGatherScalesOutAddr_);
        DequantProcess();
    }
    
    if ASCEND_IS_AIC {        
        allGatherMte_.Init(tpipe_, axisM_, axisKa_, aivNum_, rankSize_);
        mm_.SetSubBlockIdx(0);
        mm_.Init(&(tilingData_->matmulTiling), tpipe_);
        MatmulProcess();
    }
    // AscendC::PRINTF("[Kernel] Over!!!");
}
} // AddRmsNormDynamicQuantAllGatherQbmmImpl
#endif  // ADD_RMS_NORM_DYNAMIC_ALL_GATHER_QBMM_H