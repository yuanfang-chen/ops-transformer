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
 * \file quant_reduce_scatter_mte.h
 * \brief quant_reduce_scatter mte通信kernel代码逻辑
 */

#ifndef QUANT_REDUCE_SCATTER_MTE_H
#define QUANT_REDUCE_SCATTER_MTE_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "quant_reduce_scatter_tiling_data.h"
#include "utils.h"

namespace QuantReduceScatterImpl {

using namespace AscendC;
constexpr static uint32_t UB_ALIGN = 32; // UB按32字节对齐
constexpr uint32_t FLOAT_PER_UB_ALIGN = 8U;
constexpr uint32_t BUFFER_NUM = 2U;
constexpr static uint32_t PT_SCALE_SINGLE_NUM = 8U; // 当前PT量化 scale单次搬运8个
constexpr static uint32_t MXFP_SCALE_SINGLE_NUM = 32U; // scale单次搬运的个数 8 个数
constexpr static uint32_t SCALE_SINGLE_SIZE = 32U; // 当前MX量化scale单次搬运8个
constexpr static uint32_t X_SINGLE_SIZE = 1024U;  // 当前固定一次拷贝1024B数据，后续调优
constexpr static uint32_t SINGLE_NUM = 1024U;     // 当前固定一次拷贝1024个数据，后续调优
constexpr static uint32_t PER_GROUP_SIZE = 128U;  // PT量化一个scale对应的数的个数
constexpr static uint32_t MX_SIZE = 32U;          // MX量化一个scale对应的数的个数
constexpr static uint32_t TWO_DIMS = 2U;          // boardcast数组维度
constexpr static uint64_t MX_SCALES_LAST_DIM = 2U; // MX量化scales最后一维的大小

#define TemplateQuantReduceScatterTypeClass typename XType, typename ScalesType, typename OutputType
#define TemplateQuantReduceScatterType XType, ScalesType, OutputType

template<TemplateQuantReduceScatterTypeClass>
class QuantReduceScatterMte {
public:
    __aicore__ inline QuantReduceScatterMte() {};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR scales, GM_ADDR output,
        TPipe *pipe, const QuantReduceScatterTilingData *tilingData);
    __aicore__ inline void Process();
private:
    __aicore__ inline void WriteStatusToWin();
    __aicore__ inline void ReadStatus();
    __aicore__ inline void CopyDataToWin(uint64_t xOffset, uint64_t scaleOffset);
    __aicore__ inline void ReadRomoteDataAddDequant(uint64_t xOffset, uint64_t scaleOffset);
    __aicore__ inline void CastToFloat(LocalTensor<XType> &xTmpTensor, LocalTensor<ScalesType> &scaleTmpTensor);

    TPipe *tpipe_{nullptr};
    __gm__ HcclA5OpResParam *hcclContext_;
    GlobalTensor<XType> xGMTensor_;
    GlobalTensor<ScalesType> scalesGMTensor_;
    GlobalTensor<XType> localWinXGMTensor_;
    GlobalTensor<ScalesType> localWinScaleGMTensor_;
    GlobalTensor<XType> remoteWinXTensor_;
    GlobalTensor<ScalesType> remoteWinScaleTensor_;
    GlobalTensor<OutputType> outputTensor_;

    LocalTensor<XType> xTmpTensor_;
    LocalTensor<ScalesType> scaleTmpTensor_;
    LocalTensor<float> scaleCalTensor_;
    LocalTensor<float> sumTensor_;
    LocalTensor<float> stateResetTensor_;
    LocalTensor<bfloat16_t> tempBf16Scale_;
    LocalTensor<float> xCastTemp_;
    LocalTensor<OutputType> xOutTensor_;

    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> xQueue_, scaleQueue_;
    TQue<QuePosition::VECIN, 1> xInQueue_, scaleInQue;
    TQue<QuePosition::VECOUT, 1> xOutQueue_;
    TBuf<> writeStateBuf_;
    TBuf<> readStateBuf_;
    TBuf<> xCastBuf_;
    TBuf<> xCastBf16Buf_;
    TBuf<> xCastfp16Buf_;
    TBuf<> tempBf16ScaleBuf_;
    TBuf<> castScaleBuf_;
    TBuf<> sumBuf_;
    TBuf<> brcbBuf_;
    TBuf<> outputCastBuf_;
    TBuf<> stateResetBuf_;

    uint32_t aivId_{0};
    uint32_t rsTotalBlocks_{0};
    uint32_t rsRound_{0};
    uint32_t rsResRound_{0};
    uint32_t rsBlockNums_{0};
    uint64_t xSize_{0};
    uint64_t scaleSize_{0};
    uint64_t xSliceSize_{0};
    uint64_t scaleSingleNums_{0};
    uint64_t scaleSliceNums_{0};
};

template <TemplateQuantReduceScatterTypeClass>
__aicore__ inline void QuantReduceScatterMte<TemplateQuantReduceScatterType>::Init(GM_ADDR x, GM_ADDR scales,
    GM_ADDR output, TPipe *tPipe, const QuantReduceScatterTilingData *tilingData)
{
    hcclContext_ = (__gm__ HcclA5OpResParam*)GetHcclContext<HCCL_GROUP_ID_0>();
    aivId_ = GetBlockIdx();
    auto&& tiliingDataIinfo = tilingData->quantReduceScatterTilingInfo;

    // windows分区写入数据
    xSize_ = tiliingDataIinfo.bs * tiliingDataIinfo.hiddenSize * sizeof(XType);
    scaleSize_ = tiliingDataIinfo.bs * tiliingDataIinfo.scaleHiddenSize * sizeof(ScalesType);
    // 对于mx的scale是三维，最后一维为2，总scales的数据量需要再乘以2
    if constexpr(AscendC::IsSameType<ScalesType, fp8_e8m0_t>::value) {
        scaleSize_ *= MX_SCALES_LAST_DIM;
    }

    xGMTensor_.SetGlobalBuffer((__gm__ XType*)x);
    scalesGMTensor_.SetGlobalBuffer((__gm__ ScalesType*)scales);
    outputTensor_.SetGlobalBuffer((__gm__ OutputType*)output);
    // 获取本地win区地址,将数据写到本卡
    // 通过rankId获取对应卡的数据区域
    // 获取本卡地址写数据
    GM_ADDR localDataSpaceGm = (GM_ADDR)(hcclContext_->windowsIn[hcclContext_->rankId]);
    localWinXGMTensor_.SetGlobalBuffer((__gm__ XType*)localDataSpaceGm);
    localWinScaleGMTensor_.SetGlobalBuffer((__gm__ ScalesType*)(localDataSpaceGm + xSize_));

    xSliceSize_ = xSize_ / hcclContext_->rankDim; // 每块的数据量大小
    scaleSliceNums_ = scaleSize_ / (hcclContext_->rankDim * sizeof(ScalesType)); // 每块的数据量个数
    rsTotalBlocks_ = CeilDiv(xSliceSize_, X_SINGLE_SIZE);             // 1/rank 数据需要搬运的块数

    scaleSingleNums_ = SCALE_SINGLE_SIZE / sizeof(ScalesType);
    rsRound_ = rsTotalBlocks_ / tiliingDataIinfo.aivNum;  // 1/rank 数据需要搬运的所有核搬运的轮次
    rsResRound_ = rsTotalBlocks_ % tiliingDataIinfo.aivNum; // 1/rank 数据需要搬运的所有核整数轮次后剩余的块数
    rsBlockNums_ = aivId_ < rsResRound_ ? rsRound_ + 1 : rsRound_;
    tPipe->Reset();
    tPipe->InitBuffer(xInQueue_, BUFFER_NUM, X_SINGLE_SIZE);
    tPipe->InitBuffer(scaleInQue, BUFFER_NUM, UB_ALIGN);
    tPipe->InitBuffer(xQueue_, BUFFER_NUM, X_SINGLE_SIZE);    // 每次拷贝 1024个数据
    tPipe->InitBuffer(scaleQueue_, BUFFER_NUM, UB_ALIGN);     // 每次拷贝32B
    tPipe->InitBuffer(xOutQueue_, BUFFER_NUM, X_SINGLE_SIZE * sizeof(OutputType));
    tPipe->InitBuffer(writeStateBuf_, UB_ALIGN);              // 32B 对齐
    tPipe->InitBuffer(readStateBuf_, hcclContext_->rankDim * UB_ALIGN);    // 32B 对齐
    tPipe->InitBuffer(stateResetBuf_, hcclContext_->rankDim * UB_ALIGN);   // 清理状态区
    tPipe->InitBuffer(sumBuf_, SINGLE_NUM * sizeof(float));
    tPipe->InitBuffer(brcbBuf_, SINGLE_NUM * sizeof(float));
    tPipe->InitBuffer(xCastBuf_, SINGLE_NUM * sizeof(float));
    tPipe->InitBuffer(xCastBf16Buf_, SINGLE_NUM * sizeof(bfloat16_t));
    tPipe->InitBuffer(xCastfp16Buf_, SINGLE_NUM * sizeof(float16_t));
    // 对于mx量化，fp8_e8m0_t数据类型转换成float数据类型，需要进行转换：float_e8m0_t -> bfloat16_t -> float
    if constexpr(AscendC::IsSameType<ScalesType, fp8_e8m0_t>::value) {
        tPipe->InitBuffer(tempBf16ScaleBuf_, MXFP_SCALE_SINGLE_NUM * sizeof(bfloat16_t));
        tPipe->InitBuffer(castScaleBuf_, MXFP_SCALE_SINGLE_NUM * sizeof(float));
    } else {
        tPipe->InitBuffer(castScaleBuf_, PT_SCALE_SINGLE_NUM * sizeof(float));
    }
    tPipe->InitBuffer(outputCastBuf_, SINGLE_NUM * sizeof(OutputType));
    sumTensor_ = sumBuf_.Get<float>();
    stateResetTensor_ = stateResetBuf_.Get<float>();
    Duplicate<float>(stateResetTensor_, (float)0.0, static_cast<uint32_t>(hcclContext_->rankDim * FLOAT_PER_UB_ALIGN));
}

template <TemplateQuantReduceScatterTypeClass>
__aicore__ inline void QuantReduceScatterMte<TemplateQuantReduceScatterType>::WriteStatusToWin()
{
    uint32_t coreOffset = aivId_ * hcclContext_->rankDim;
    for (uint32_t curRank = 0; curRank < hcclContext_->rankDim; ++curRank) {
        // 写入状态到对端，每个核写一个状态，表示自己的数据块已经写完
        LocalTensor<float> statusTensor = writeStateBuf_.Get<float>();
        DataCopy<float>(statusTensor, stateResetTensor_, FLOAT_PER_UB_ALIGN); // 先重置statusTensor数据
        SyncFunc<AscendC::HardEvent::MTE2_S>(); 
        statusTensor(0) = (float)1.0;
        GM_ADDR remoteWinStateGM = (GM_ADDR)hcclContext_->windowsOut[curRank];  // 当前读取的那片数据对应的卡的位置
        GlobalTensor<float> stateGMTensor;
        stateGMTensor.SetGlobalBuffer((__gm__ float*)remoteWinStateGM);
        // 不同卡上的核的状态写到相邻位置，读时可以一次读rankDim个状态
        uint32_t curOffset = (coreOffset + hcclContext_->rankId) * FLOAT_PER_UB_ALIGN;
        SyncFunc<AscendC::HardEvent::S_MTE3>();
        DataCopy(stateGMTensor[curOffset], statusTensor, FLOAT_PER_UB_ALIGN);  // 按32对齐拷贝, 32B, 卡偏移 + 当前核的偏移
    }
}

template <TemplateQuantReduceScatterTypeClass>
__aicore__ inline void QuantReduceScatterMte<TemplateQuantReduceScatterType>::ReadStatus()
{
    GM_ADDR stateGM = (GM_ADDR)hcclContext_->windowsOut[hcclContext_->rankId];  // 当前核读取rankDim张卡对应核的状态位
    GlobalTensor<float> selfStatusWinTensor;
    uint32_t offset = aivId_ * hcclContext_->rankDim * FLOAT_PER_UB_ALIGN;
    selfStatusWinTensor.SetGlobalBuffer((__gm__ float*)(stateGM));
    LocalTensor<float> statusTensor = readStateBuf_.Get<float>();
    float flag = 0;
    uint32_t statusCnt = hcclContext_->rankDim * FLOAT_PER_UB_ALIGN;
    SumParams sumParams{1, statusCnt, statusCnt};
    float minTarget = hcclContext_->rankDim - (float)0.5;
    float maxTarget = hcclContext_->rankDim + (float)0.5;
    // 读取statusCnt个数据求和
    while ((flag < minTarget) || (flag > maxTarget)) {
        SyncFunc<AscendC::HardEvent::S_MTE2>();
        DataCopy<float>(statusTensor, selfStatusWinTensor[offset], statusCnt);
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        Sum(statusTensor, statusTensor, sumParams);
        SyncFunc<AscendC::HardEvent::V_S>();
        flag = statusTensor(0);
    }
    SyncFunc<AscendC::HardEvent::S_MTE3>();
    DataCopy<float>(selfStatusWinTensor[offset], stateResetTensor_, statusCnt);
}

template <TemplateQuantReduceScatterTypeClass>
__aicore__ inline void QuantReduceScatterMte<TemplateQuantReduceScatterType>::CopyDataToWin(uint64_t xOffset,
    uint64_t scaleOffset)
{
    // 每个核需要搬运的数据块数
    for(uint64_t curBlock = 0; curBlock < rsBlockNums_; ++curBlock) {
        uint64_t curXOffset = xOffset + curBlock * X_SINGLE_SIZE;
        uint64_t curScaleOffset = scaleOffset + curBlock * scaleSingleNums_;
        for(uint64_t curRank = 0; curRank < hcclContext_->rankDim; ++curRank) {
            // 先拷贝data数据， 再拷贝scales
            uint64_t curRankXOffset = curXOffset + curRank * xSliceSize_;
            uint64_t curRankScaleOffset = curScaleOffset + curRank * scaleSliceNums_;
            xTmpTensor_ = xQueue_.AllocTensor<XType>();
            DataCopy(xTmpTensor_, xGMTensor_[curRankXOffset], X_SINGLE_SIZE);
            xQueue_.EnQue(xTmpTensor_);
            xTmpTensor_ = xQueue_.DeQue<XType>();
            DataCopy(localWinXGMTensor_[curRankXOffset], xTmpTensor_, X_SINGLE_SIZE);
            xQueue_.FreeTensor<XType>(xTmpTensor_);
            scaleTmpTensor_ = scaleQueue_.AllocTensor<ScalesType>();
            DataCopy(scaleTmpTensor_, scalesGMTensor_[curRankScaleOffset], scaleSingleNums_);
            scaleQueue_.EnQue(scaleTmpTensor_);
            scaleTmpTensor_ = scaleQueue_.DeQue<ScalesType>();
            DataCopy(localWinScaleGMTensor_[curRankScaleOffset], scaleTmpTensor_, scaleSingleNums_);
            scaleQueue_.FreeTensor<ScalesType>(scaleTmpTensor_);
        }
    }
}

template <TemplateQuantReduceScatterTypeClass>
__aicore__ inline void QuantReduceScatterMte<TemplateQuantReduceScatterType>::CastToFloat(LocalTensor<XType> &xTmpTensor,
    LocalTensor<ScalesType> &scaleTmpTensor)
{
    xCastTemp_ = xCastBuf_.Get<float>();
    scaleCalTensor_ = brcbBuf_.Get<float>();
    LocalTensor<bfloat16_t> xTempBf16Tensor = xCastBf16Buf_.Get<bfloat16_t>();
    LocalTensor<float16_t> xTempfp16Tensor = xCastfp16Buf_.Get<float16_t>();
    LocalTensor<float> castLocalScale = castScaleBuf_.Get<float>();
    
    if constexpr (AscendC::IsSameType<XType, fp4x2_e2m1_t>::value ||
        AscendC::IsSameType<XType, fp4x2_e1m2_t>::value) {
        // 数据类型转换 fp4x2_e2m1_t/fp4x2_e1m2_t -> bfloat16_t -> float
        Duplicate<bfloat16_t>(xTempBf16Tensor, (bfloat16_t)0.0, SINGLE_NUM);
        Cast(xTempBf16Tensor, xTmpTensor, RoundMode::CAST_NONE, SINGLE_NUM);
        PipeBarrier<PIPE_V>();
        Cast(xCastTemp_, xTempBf16Tensor, RoundMode::CAST_NONE, SINGLE_NUM);
        PipeBarrier<PIPE_V>();
    } else if (AscendC::IsSameType<XType, int8_t>::value) {
        // 数据类型转换 int8_t -> float16_t -> float
        Duplicate<float16_t>(xTempfp16Tensor, (float16_t)0.0, SINGLE_NUM);
        Cast(xTempfp16Tensor, xTmpTensor, RoundMode::CAST_NONE, SINGLE_NUM);
        PipeBarrier<PIPE_V>();
        Cast(xCastTemp_, xTempfp16Tensor, RoundMode::CAST_NONE, SINGLE_NUM);
        PipeBarrier<PIPE_V>();
    } else {
        // 数据类型转换 fp8/hifloat8 -> float
        Duplicate<float>(xCastTemp_, (float)0.0, SINGLE_NUM);
        PipeBarrier<PIPE_V>();
        Cast(xCastTemp_, xTmpTensor, RoundMode::CAST_NONE, SINGLE_NUM);
        PipeBarrier<PIPE_V>();
    }

    if constexpr (AscendC::IsSameType<ScalesType, fp8_e8m0_t>::value) {
        // 数据类型转换 fp8_e8m0_t -> bfloat16_t -> float
        // 当前Cast接口不支持fp8_e8m0_t类型转换只能使用微指令实现
        tempBf16Scale_ = tempBf16ScaleBuf_.Get<bfloat16_t>();
        Duplicate<bfloat16_t>(tempBf16Scale_, (bfloat16_t)0.0, MX_SIZE);
        PipeBarrier<PIPE_V>();
        __local_mem__ fp8_e8m0_t* srcPtr = (__local_mem__ fp8_e8m0_t*)scaleTmpTensor.GetPhyAddr();
        __local_mem__ bfloat16_t* tempBf16ScalePtr = (__local_mem__ bfloat16_t*)tempBf16Scale_.GetPhyAddr();
        // 当前Cast接口不支持fp8_e8m0_t类型转换只能使用微指令实现
        VF_CALL<CastVf>(tempBf16ScalePtr, srcPtr, MX_SIZE);
        PipeBarrier<PIPE_V>();
        Cast(castLocalScale, tempBf16Scale_, RoundMode::CAST_NONE, MX_SIZE);
        PipeBarrier<PIPE_V>();
        // MX量化将scale广播成32
        const uint32_t broadcastDst[TWO_DIMS]{MXFP_SCALE_SINGLE_NUM, MX_SIZE};
        const uint32_t broadcastSrc[TWO_DIMS]{MXFP_SCALE_SINGLE_NUM, 1};
        BroadCast<float, TWO_DIMS, 1>(scaleCalTensor_, castLocalScale, broadcastDst, broadcastSrc);
    } else {
        castLocalScale = scaleTmpTensor.template ReinterpretCast<float>();
        // PT量化将scale广播成128
        const uint32_t broadcastDst[TWO_DIMS]{PT_SCALE_SINGLE_NUM, PER_GROUP_SIZE};
        const uint32_t broadcastSrc[TWO_DIMS]{PT_SCALE_SINGLE_NUM, 1};
        BroadCast<float, TWO_DIMS, 1>(scaleCalTensor_, castLocalScale, broadcastDst, broadcastSrc);
    }
}

// 使用Que进行数据搬运
template <TemplateQuantReduceScatterTypeClass>
__aicore__ inline void QuantReduceScatterMte<TemplateQuantReduceScatterType>::ReadRomoteDataAddDequant(uint64_t xOffset,
    uint64_t scaleOffset)
{
    ReadStatus();
    for(uint64_t curBlock = 0; curBlock < rsBlockNums_; ++curBlock) {
        uint64_t curXOffset = xOffset + curBlock * X_SINGLE_SIZE;  // 对于fp4需要另外算偏移
        uint64_t curScaleOffset = scaleOffset + curBlock * scaleSingleNums_;
        Duplicate<float>(sumTensor_, (float)0.0, SINGLE_NUM);
        // 读取远端数据进行dequant和sum操作
        for(int remoteRankId = 0; remoteRankId < hcclContext_->rankDim; ++remoteRankId) {
            uint64_t curRankXOffset = curXOffset + hcclContext_->rankId * xSliceSize_;
            uint64_t curRankScaleOffset = curScaleOffset + hcclContext_->rankId * scaleSliceNums_;
            GlobalTensor<XType> remoteWinGMTensor_;
            GM_ADDR remoteXWin = (GM_ADDR)(hcclContext_->windowsIn[remoteRankId]);
            remoteWinXTensor_.SetGlobalBuffer((__gm__ XType*)remoteXWin);
            GM_ADDR remoteScaleWin = (GM_ADDR)(hcclContext_->windowsIn[remoteRankId] + xSize_);
            remoteWinScaleTensor_.SetGlobalBuffer((__gm__ ScalesType*)remoteScaleWin);

            LocalTensor<XType> xTmpTensor = xInQueue_.AllocTensor<XType>();
            xInQueue_.EnQue(xTmpTensor);
            DataCopy(xTmpTensor, remoteWinXTensor_[curRankXOffset], X_SINGLE_SIZE);
            xTmpTensor = xInQueue_.DeQue<XType>();
            LocalTensor<ScalesType> scaleTmpTensor = scaleInQue.AllocTensor<ScalesType>();
            DataCopy(scaleTmpTensor, remoteWinScaleTensor_[curRankScaleOffset], scaleSingleNums_);
            scaleInQue.EnQue(scaleTmpTensor);
            scaleTmpTensor = scaleInQue.DeQue<ScalesType>();

            // Cast成float数据计算
            CastToFloat(xTmpTensor, scaleTmpTensor);
            PipeBarrier<PIPE_ALL>();
            Mul(xCastTemp_, xCastTemp_, scaleCalTensor_, SINGLE_NUM); 
            PipeBarrier<PIPE_ALL>();
            Add(sumTensor_, sumTensor_, xCastTemp_, SINGLE_NUM);
            xInQueue_.FreeTensor(xTmpTensor);
            scaleInQue.FreeTensor(scaleTmpTensor);
            PipeBarrier<PIPE_ALL>();
        }
        // 将计算好的数据拷贝到输出tensor，如果是非float数据类型需要先转换成目标数据类型
        xOutTensor_ = xOutQueue_.AllocTensor<OutputType>();
        if constexpr (AscendC::IsSameType<OutputType, float>::value) {
            DataCopy(xOutTensor_, sumTensor_, SINGLE_NUM);
        } else {
            Cast(xOutTensor_, sumTensor_, RoundMode::CAST_RINT, SINGLE_NUM);
        }
        xOutQueue_.EnQue(xOutTensor_);
        xOutTensor_ = xOutQueue_.DeQue<OutputType>();
        DataCopy(outputTensor_[curXOffset], xOutTensor_, SINGLE_NUM);
        xOutQueue_.FreeTensor(xOutTensor_);
    }
}

template <TemplateQuantReduceScatterTypeClass>
__aicore__ inline void QuantReduceScatterMte<TemplateQuantReduceScatterType>::Process()
{
    if ASCEND_IS_AIC {
        return;
    }
    // 当前核访问连续地址处理, 尾块顺序分配给前面核处理
    uint64_t blockOffset = aivId_ >= rsResRound_ ? aivId_ * rsRound_ + rsResRound_ :
        aivId_ * rsRound_ + aivId_ % rsResRound_;
    uint64_t xOffset = blockOffset * X_SINGLE_SIZE;
    uint64_t scaleOffset = blockOffset * scaleSingleNums_;
    // 一次性拷贝完所有数据到本地卡win区
    CopyDataToWin(xOffset, scaleOffset);
    PipeBarrier<PIPE_ALL>();
    // 写入状态到对端
    WriteStatusToWin();
    
    ReadRomoteDataAddDequant(xOffset, scaleOffset);
}
} // QuantReduceScatterImpl
#endif  // QUANT_REDUCE_SCATTER_MTE_H