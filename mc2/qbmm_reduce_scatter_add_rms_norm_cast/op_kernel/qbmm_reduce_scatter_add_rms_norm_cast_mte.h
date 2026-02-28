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
 * \file qbmm_reduce_scatter_add_rms_norm_cast_mte.h
 * \brief qbmm_reduce_scatter_add_rms_norm_cast mte通信kernel代码逻辑
 */

#ifndef QBMM_REDUCE_SCATTER_ADD_RMS_NORM_CAST_MTE_H
#define QBMM_REDUCE_SCATTER_ADD_RMS_NORM_CAST_MTE_H

#include "basic_api/kernel_basic_intf.h"
#include "adv_api/hccl/hccl.h"
#include "adv_api/reduce/sum.h"
#include "adv_api/pad/broadcast.h"
#include "kernel_tiling/kernel_tiling.h"
#include "qbmm_reduce_scatter_add_rms_norm_cast_tiling_data.h"
#include "kernel_operator.h"
#if __has_include("../common/inc/kernel/moe_distribute_base.h")
#include "../common/inc/kernel/moe_distribute_base.h"
#include "../common/inc/kernel/mc2_kernel_utils.h"
#else
#include "../../common/inc/kernel/moe_distribute_base.h"
#include "../../common/inc/kernel/mc2_kernel_utils.h"
#endif

namespace QbmmReduceScatterAddRmsNormCastImpl {
using namespace AscendC;

constexpr uint32_t BUFFER_NUM = 1U;
constexpr uint32_t UB_ALIGN_BYTES = 32U;
constexpr uint32_t FLOAT_UB_ALIGN_NUM = 8U;
constexpr uint32_t BLOCK_LENGTH = 5120U;
constexpr float ONE = 1;

template <AscendC::HardEvent event>
__aicore__ inline void SyncFunc() {
    int32_t eventID = static_cast<int32_t>(GetTPipePtr()->FetchEventID(event));
    AscendC::SetFlag<event>(eventID);
    AscendC::WaitFlag<event>(eventID);
}

class QbmmReduceScatterAddRmsNormCastMte {
public:
    __aicore__ inline QbmmReduceScatterAddRmsNormCastMte() {};
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR gamma, GM_ADDR scale, GM_ADDR bias, GM_ADDR perTokenScale, GM_ADDR y1Out,
                GM_ADDR y2Out, GM_ADDR xOut, GM_ADDR workspaceGM, TPipe *pipe, const QbmmReduceScatterAddRmsNormCastTilingData *tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline GM_ADDR GetWindAddrByRankId(const int32_t rankId);
    __aicore__ inline void SplitToCore(const uint32_t curSendCnt, const uint32_t curUseAivNum, const uint32_t coreId, uint32_t &startId, uint32_t &endId, uint32_t &sendNum);
    __aicore__ inline GM_ADDR GetWindStateAddrByRankId(const int32_t rankId);
    __aicore__ inline void DequantCompute(uint64_t baseMOffset, uint64_t baseNOffset, uint32_t curAicM, uint32_t curAicN, uint64_t row, uint64_t col);
    __aicore__ inline void DequantNOuterSplitAndSendDataToRemote(uint32_t singleM, uint32_t singleN, uint32_t fixpMtimes, uint32_t fixpNtimes, uint32_t mOffset, uint32_t nOffset);
    __aicore__ inline void ReadRemoteDataAdd();
    __aicore__ inline void WriteStatusToWin();
    __aicore__ inline void ReadStatus();
    __aicore__ inline void AddRmsNormAddCompute(uint32_t tokenIndex, uint32_t numCol,
                                                LocalTensor<float>& x1TmpFloatLocal,
                                                LocalTensor<float>& x2TmpFloatLocal,
                                                LocalTensor<float>& addOutTmpFloatLocal,
                                                const DataCopyExtParams& copyExtParams,
                                                const DataCopyPadExtParams<bfloat16_t>& copyPadExtParams);
    __aicore__ inline void AddRmsNormRmsNormCompute(uint32_t tokenIndex, uint32_t numCol,
                                                    LocalTensor<float>& xFp32, LocalTensor<float>& sqx,
                                                    LocalTensor<float>& gammaLocal,
                                                    const DataCopyExtParams& copyExtParams);
    TPipe *tpipe_{nullptr};
    TBuf<> writeStateBuf_;
    TBuf<> readStateBuf_;
    TBuf<> sumFp32Buf_;
    TBuf<> tokenFp32Buf_;

    // addrmsnormcast
    TBuf<> tokenBuf_;
    TBuf<> gammaBuf_;
    TBuf<> rowTmpFloatBuf_;
    TBuf<> mulBuf_;
    TBuf<> resFp32Buf_;
    TBuf<TPosition::VECCALC> reduceFp32Buf_;

    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> tmpQueue_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> tokenQueue_;

    uint32_t coreVid_{0};
    uint32_t coreCid_{0};
    GlobalTensor<int8_t> x1GM_;
    GlobalTensor<int8_t> x2GM_;
    // 用来存储通信tensor
    GlobalTensor<bfloat16_t> yGM_;
    // 用来存储reducescatter输出
    GlobalTensor<bfloat16_t> xOutGM_;
    GlobalTensor<bfloat16_t> y2OutGM_;
    GlobalTensor<float> y1OutGM_;
    GlobalTensor<float> gammaGM_;

    LocalTensor<bfloat16_t> tmpTensor_;
    LocalTensor<bfloat16_t> tokenTensor_;
    LocalTensor<float> rowTmpFloatLocal_;
    LocalTensor<float> mulBufLocal_;

    uint32_t m_;
    uint32_t n_;
    uint32_t singleTpSize_;
    uint32_t baseM_;
    uint32_t baseN_;
    uint32_t singleM_;
    uint32_t singleN_;
    uint32_t tileN_;
    uint32_t tileNRemainder_;
    uint32_t tpWorldSize_;
    uint32_t rankId_{0};
    uint32_t aicNum_{0};
    uint32_t aivNum_{0};
    float armAvgFactor_;
    float epsilon_;
    __gm__ HcclOpResParam *winContext_{nullptr};
};

__aicore__ inline void QbmmReduceScatterAddRmsNormCastMte::Init(GM_ADDR x, GM_ADDR x2, GM_ADDR y, GM_ADDR gamma, GM_ADDR scale, 
    GM_ADDR bias, GM_ADDR perTokenScale, GM_ADDR y1Out, GM_ADDR y2Out, GM_ADDR xOut, GM_ADDR workspaceGM, TPipe *pipe, const QbmmReduceScatterAddRmsNormCastTilingData *tilingData)
{
    // PRINTF("kernel init doing.");
    tpipe_ = pipe;
    coreVid_ = GetBlockIdx();
    coreCid_ = coreVid_ / 2;
    yGM_.SetGlobalBuffer((__gm__ bfloat16_t*) y);
    gammaGM_.SetGlobalBuffer((__gm__ float*) gamma);
    xOutGM_.SetGlobalBuffer((__gm__ bfloat16_t*) xOut);
    y1OutGM_.SetGlobalBuffer((__gm__ float*) y1Out);
    y2OutGM_.SetGlobalBuffer((__gm__ bfloat16_t*) y2Out);
    singleM_ = 126;
    singleN_ = 128;
    baseM_ = 126;
    baseN_ = 128;

    m_ = 252;
    n_ = 5120;
    aicNum_ = 24;
    aivNum_ = 48;
    tpWorldSize_ = 4;
    singleTpSize_ = (singleM_ / 2) * n_;
    armAvgFactor_ = 1.0f / n_;
    epsilon_ = 1e-6f;
    
    // init ub local buffer
    tpipe_->InitBuffer(tmpQueue_, BUFFER_NUM, 63 * 128 * 2);
    tpipe_->InitBuffer(tokenQueue_, BUFFER_NUM, 5120 * 2);
    tpipe_->InitBuffer(writeStateBuf_, UB_ALIGN_BYTES);
    tpipe_->InitBuffer(readStateBuf_, UB_ALIGN_BYTES);
    tpipe_->InitBuffer(sumFp32Buf_, BLOCK_LENGTH * 4);
    tpipe_->InitBuffer(tokenFp32Buf_, BLOCK_LENGTH * 4);
    tpipe_->InitBuffer(tokenBuf_, BLOCK_LENGTH * 2);
    tpipe_->InitBuffer(rowTmpFloatBuf_, BLOCK_LENGTH * 2);
    tpipe_->InitBuffer(mulBuf_, BLOCK_LENGTH * 4);
    tpipe_->InitBuffer(reduceFp32Buf_, 256);
    tpipe_->InitBuffer(resFp32Buf_, BLOCK_LENGTH * 4);
    tpipe_->InitBuffer(gammaBuf_, BLOCK_LENGTH * 4);
    winContext_ = (__gm__ HcclOpResParam*)AscendC::GetHcclContext<HCCL_GROUP_ID_0>();
    rankId_ = winContext_->localUsrRankId;
    
}

__aicore__ inline GM_ADDR QbmmReduceScatterAddRmsNormCastMte::GetWindAddrByRankId(const int32_t rankId)
{
    if (rankId == rankId_) {
        return (GM_ADDR)(winContext_->localWindowsIn);
    }
    return (GM_ADDR)(((HcclRankRelationResV2*)(winContext_->remoteRes[rankId].nextDevicePtr))->windowsIn);   // 先找到某个rank的首地址，然后再偏移到具体的处理data的地方
}

__aicore__ inline GM_ADDR QbmmReduceScatterAddRmsNormCastMte::GetWindStateAddrByRankId(const int32_t rankId)
{
    if (rankId == rankId_) {
        return (GM_ADDR)(winContext_->localWindowsExp);
    }
    return (GM_ADDR)(((HcclRankRelationResV2*)(winContext_->remoteRes[rankId].nextDevicePtr))->windowsExp);
}

__aicore__ inline void QbmmReduceScatterAddRmsNormCastMte::SplitToCore(const uint32_t curSendCnt, const uint32_t curUseAivNum, const uint32_t coreId, uint32_t &startId, uint32_t &endId, uint32_t &sendNum)
{
    sendNum = curSendCnt / curUseAivNum;
    uint32_t remainderNum = curSendCnt % curUseAivNum;
    startId = sendNum * coreId;
    if (coreId < remainderNum) {
        sendNum += 1;
        startId += coreId;
    } else {
        startId += remainderNum;
    }
    endId = startId + sendNum;
}

__aicore__ inline void QbmmReduceScatterAddRmsNormCastMte::WriteStatusToWin()
{
    if (coreVid_ >= tpWorldSize_) {
        return;
    }
    uint32_t curOffset = rankId_ * FLOAT_UB_ALIGN_NUM;
    // 写入状态到对端，每个核写一个状态到一个rank
    LocalTensor<float> statusTensor = writeStateBuf_.Get<float>();
    statusTensor(0) = (float)1;
    int32_t curDstId = coreVid_;
    GM_ADDR remoteWinStateGM = GetWindStateAddrByRankId(curDstId);
    GlobalTensor<float> stateGMTensor;
    stateGMTensor.SetGlobalBuffer((__gm__ float*)remoteWinStateGM);
    SyncFunc<AscendC::HardEvent::S_MTE3>();
    DataCopy(stateGMTensor[curOffset], statusTensor, FLOAT_UB_ALIGN_NUM);
}

__aicore__ inline void QbmmReduceScatterAddRmsNormCastMte::ReadStatus()
{
    // 粗粒度状态区逻辑，只有tpwordsize个状态，所需的v核只需要4个
    if (coreVid_ >= tpWorldSize_) {
        return;
    }
    GM_ADDR stateGM = GetWindStateAddrByRankId(rankId_);
    GlobalTensor<float> selfStateWinTensor;
    // 获取当前核所需读取状态的头地址
    selfStateWinTensor.SetGlobalBuffer((__gm__ float*)(stateGM));
    uint32_t offset = coreVid_ * FLOAT_UB_ALIGN_NUM;
    LocalTensor<float> statusTensor = readStateBuf_.Get<float>();
    float flag = -1;
    uint32_t statusCnt = FLOAT_UB_ALIGN_NUM;
    float minTarget = (float)0.5;
    float maxTarget = (float)1.5;
    // 读取statusCnt个数据求和
    while ((flag < minTarget) || (flag > maxTarget)) {
        SyncFunc<AscendC::HardEvent::S_MTE2>();
        DataCopy(statusTensor, selfStateWinTensor[offset], statusCnt);
        SyncFunc<AscendC::HardEvent::MTE2_S>();
        flag = statusTensor(0);
    }   
}

__aicore__ inline void QbmmReduceScatterAddRmsNormCastMte::DequantCompute(uint64_t baseMOffset, uint64_t baseNOffset, uint32_t curAicM, uint32_t curAicN, uint64_t row, uint64_t col)
{
    uint32_t curAivM = curAicM / 2;
    uint32_t curAivN = curAicN;
    uint32_t mUbLoops = 1;
    // 一个block是256字节 = 8 datablock， 一行5120 * 2 = 10240字节 = 320 block， 320 - 8 = 312
    DataCopyParams gm2UbParams{63, 8, 312, 0};
    DataCopyParams ub2GmParams{63, 8, 0, 312};
    // 只循环一次
    GlobalTensor<bfloat16_t> remoteTensor;
    for (uint32_t mUbLoopIdx = 0; mUbLoopIdx < mUbLoops; ++mUbLoopIdx) {
        tmpTensor_ = tmpQueue_.AllocTensor<bfloat16_t>();
        uint32_t curAicAivOffset = row * curAicM * n_ + (coreVid_ % 2) * curAivM * n_ + col * singleN_;
        DataCopy(tmpTensor_, yGM_[curAicAivOffset], gm2UbParams);
        tmpQueue_.EnQue(tmpTensor_);
        tmpTensor_ = tmpQueue_.DeQue<bfloat16_t>();
        // 计算当前正处于的TP域
        int32_t remoteRankId = curAicAivOffset / (m_ * n_ / tpWorldSize_);
        GM_ADDR remoteWinAddr = GetWindAddrByRankId(remoteRankId);
        remoteTensor.SetGlobalBuffer((__gm__ bfloat16_t*)remoteWinAddr);
        uint64_t tpOffset = (m_ * n_ / tpWorldSize_) * rankId_;
        uint64_t nOffset = col * baseN_;
        DataCopy(remoteTensor[tpOffset + nOffset], tmpTensor_, ub2GmParams);
        tmpQueue_.FreeTensor<bfloat16_t>(tmpTensor_);
    }
}

__aicore__ inline void QbmmReduceScatterAddRmsNormCastMte::DequantNOuterSplitAndSendDataToRemote(uint32_t singleM, uint32_t singleN, uint32_t fixpMtimes,
                                              uint32_t fixpNtimes, uint32_t mOffset, uint32_t nOffset)
{
    uint32_t curAicOuter = baseN_;
    uint32_t curAicInner = baseM_;
    for (uint32_t fixpOuterIdx = 0; fixpOuterIdx < fixpNtimes; ++fixpOuterIdx) {
        for (uint32_t fixpInnerIdx = 0; fixpInnerIdx < fixpMtimes; ++fixpInnerIdx) {
            // 目前只执行一次
            DequantCompute(static_cast<uint64_t>(fixpInnerIdx) * baseM_ * n_, fixpOuterIdx * baseN_,
                            curAicInner, curAicOuter, mOffset, nOffset);
        }
    }
}

__aicore__ inline void QbmmReduceScatterAddRmsNormCastMte::AddRmsNormAddCompute(uint32_t tokenIndex, uint32_t numCol,
                                                                                LocalTensor<float>& x1TmpFloatLocal,
                                                                                LocalTensor<float>& x2TmpFloatLocal,
                                                                                LocalTensor<float>& addOutTmpFloatLocal,
                                                                                const DataCopyExtParams& copyExtParams,
                                                                                const DataCopyPadExtParams<bfloat16_t>& copyPadExtParams)
{
    // 计算x + residual_x
    LocalTensor<bfloat16_t> x2 = tokenBuf_.Get<bfloat16_t>();
    DataCopyPad(x2, yGM_[tokenIndex * n_], copyExtParams, copyPadExtParams);
    SyncFunc<AscendC::HardEvent::MTE2_V>();
    Cast(x2TmpFloatLocal, x2, AscendC::RoundMode::CAST_NONE, numCol);
    PipeBarrier<PIPE_V>();
    AscendC::Add(addOutTmpFloatLocal, x1TmpFloatLocal, x2TmpFloatLocal, numCol);
}

__aicore__ inline void QbmmReduceScatterAddRmsNormCastMte::AddRmsNormRmsNormCompute(uint32_t tokenIndex, uint32_t numCol,
                                                    LocalTensor<float>& xFp32, LocalTensor<float>& sqx,
                                                    LocalTensor<float>& gammaLocal,
                                                    const DataCopyExtParams& copyExtParams)
{
    // 计算rstd
    LocalTensor<float> reduceBufLocal = reduceFp32Buf_.Get<float>();
    LocalTensor<float> resFp32Tensor = resFp32Buf_.Get<float>();
    Mul(sqx, xFp32, xFp32, numCol);
    PipeBarrier<PIPE_V>();
    Muls(sqx, sqx, armAvgFactor_, numCol);
    PipeBarrier<PIPE_V>();
    ReduceSum(sqx, sqx, reduceBufLocal, numCol);
    PipeBarrier<PIPE_V>();
    Adds(sqx, sqx, epsilon_, 1);
    PipeBarrier<PIPE_V>();
    Sqrt(sqx, sqx, 1);
    Duplicate(reduceBufLocal, ONE, 1);
    PipeBarrier<PIPE_V>();
    Div(reduceBufLocal, reduceBufLocal, sqx, 1);
    // 计算y
    SyncFunc<AscendC::HardEvent::V_S>();
    float rstdValue = reduceBufLocal.GetValue(0);
    SyncFunc<AscendC::HardEvent::S_V>();
    Muls(xFp32, xFp32, rstdValue, numCol);
    PipeBarrier<PIPE_V>();
    LocalTensor<bfloat16_t> yLocal = rowTmpFloatBuf_.Get<bfloat16_t>();
    Cast(yLocal, xFp32, RoundMode::CAST_RINT, numCol);
    PipeBarrier<PIPE_V>();
    Cast(xFp32, yLocal, RoundMode::CAST_NONE, numCol);
    PipeBarrier<PIPE_V>();
    Mul(xFp32, xFp32, gammaLocal, numCol);
    PipeBarrier<PIPE_V>();
    Cast(yLocal, xFp32, RoundMode::CAST_RINT, numCol);
    // y结果搬出
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopy(y2OutGM_[tokenIndex * n_], yLocal, n_);
    PipeBarrier<PIPE_V>();
    Cast(resFp32Tensor, yLocal, RoundMode::CAST_NONE, numCol);
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopy(y1OutGM_[tokenIndex * n_], resFp32Tensor, n_);
}

__aicore__ inline void QbmmReduceScatterAddRmsNormCastMte::ReadRemoteDataAdd()
{
    uint32_t startRowId = 0;
    uint32_t endRowId = 0;
    uint32_t rowNum = 0;
    uint32_t tokenIndex = 0;
    SplitToCore(singleM_ / 2, aivNum_, coreVid_, startRowId, endRowId, rowNum);

    LocalTensor<float> sumFp32Tensor = sumFp32Buf_.Get<float>();
    LocalTensor<float> tokenFp32Tensor = tokenFp32Buf_.Get<float>();
    mulBufLocal_ = mulBuf_.Get<float>();
    uint32_t curVidOffset = startRowId * n_;
    GlobalTensor<bfloat16_t> localWinTensor;
    localWinTensor.SetGlobalBuffer((__gm__ bfloat16_t*)GetWindAddrByRankId(rankId_));

    DataCopyExtParams expandXCopyParams{1U, static_cast<uint32_t>(n_ * 2), 0U, 0U, 0U};
    DataCopyExtParams gammaCopyParams{1U, static_cast<uint32_t>(n_ * 4), 0U, 0U, 0U};
    const DataCopyPadExtParams<bfloat16_t> copyPadXTypeParams{false, 0U, 0U, 0U};
    const DataCopyPadExtParams<float> copyPadFloatParams{false, 0U, 0U, 0U};

    for(uint32_t tileIdx = 0; tileIdx < rowNum; ++tileIdx) {
        Duplicate<float>(sumFp32Tensor, (float)0.0, BLOCK_LENGTH);
        tokenIndex = startRowId + tileIdx;
        uint32_t curVidTileOffset = curVidOffset + tileIdx * n_;
        for(int tpIndex = 0; tpIndex < tpWorldSize_; ++tpIndex) {
            uint32_t curOffset = curVidTileOffset + tpIndex * singleTpSize_;
            tokenTensor_ = tokenQueue_.AllocTensor<bfloat16_t>();
            DataCopy(tokenTensor_, localWinTensor[curOffset], n_);
            tokenQueue_.EnQue(tokenTensor_);
            tokenTensor_ = tokenQueue_.DeQue<bfloat16_t>();
            SyncFunc<AscendC::HardEvent::MTE2_V>();
            Cast(tokenFp32Tensor, tokenTensor_, RoundMode::CAST_NONE, BLOCK_LENGTH);
            PipeBarrier<PIPE_V>();
            Add(sumFp32Tensor, sumFp32Tensor, tokenFp32Tensor, BLOCK_LENGTH);
            tokenQueue_.FreeTensor<bfloat16_t>(tokenTensor_);
        }
        AddRmsNormAddCompute(tokenIndex, n_, sumFp32Tensor, rowTmpFloatLocal_, sumFp32Tensor, expandXCopyParams, copyPadXTypeParams);
        // 执行AddRmsNorm--Add，输出x
        LocalTensor<bfloat16_t> sumBufLocal = tokenBuf_.Get<bfloat16_t>();
        PipeBarrier<PIPE_V>();
        Cast(sumBufLocal, sumFp32Tensor, AscendC::RoundMode::CAST_RINT, n_);
        SyncFunc<AscendC::HardEvent::V_MTE3>();
        DataCopyPad(xOutGM_[tokenIndex * n_], sumBufLocal, expandXCopyParams);

        // 执行AddRmsNorm--RmsNormCast，输出x最终结果的fp16搬出到y2，float32搬出到y1
        LocalTensor<float> gammaLocal = gammaBuf_.Get<float>();
        DataCopyPad(gammaLocal, gammaGM_, gammaCopyParams, copyPadFloatParams);
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        AddRmsNormRmsNormCompute(tokenIndex, n_, sumFp32Tensor, mulBufLocal_, gammaLocal, expandXCopyParams);
    }
}

__aicore__ inline void QbmmReduceScatterAddRmsNormCastMte::Process()
{
    if ASCEND_IS_AIV {
        uint32_t fixpMTimes = singleM_ / baseM_;
        uint32_t fixpNTimes = singleN_ / baseN_;
        uint32_t tileTotalNum = (n_ / baseN_) * (m_ / baseM_);
        uint32_t startBlockIdx = 0;
        uint32_t endBlockIdx = 0;
        uint32_t tileNum = 0;
        SplitToCore(tileTotalNum, aicNum_, coreCid_, startBlockIdx, endBlockIdx, tileNum);
        // 发数据
        for(uint32_t idx = 0; idx < tileNum; idx++) {
            // 计算当前index(126, 128) <=> 前置块的个数
            //计算yGM具体偏移 = 当前核处理的数据块idx， 计算idx的row和col
            uint32_t blockIdx = startBlockIdx + idx;
            uint32_t mOffset = blockIdx / (n_ / singleN_);
            uint32_t nOffset = blockIdx % (n_ / singleN_);
            DequantNOuterSplitAndSendDataToRemote(singleM_, singleN_, fixpMTimes, fixpNTimes, mOffset, nOffset);
        }
        SyncAll<true>();
        // 当前die的数据已经发完
        WriteStatusToWin();
        ReadStatus();
        SyncAll<true>();    //确保前4个核都等到了状态，即数据区完全ready
        ReadRemoteDataAdd();
    }
}
}; // QbmmReduceScatterAddRmsNormCastImpl
#endif  // QBMM_REDUCE_SCATTER_ADD_RMS_NORM_CAST_MTE_H