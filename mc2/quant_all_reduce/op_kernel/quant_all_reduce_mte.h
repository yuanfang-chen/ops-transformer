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
 * \file quant_all_reduce_mte.h
 * \brief quant_all_reduce mte通信kernel代码逻辑
 */

#ifndef QUANT_ALL_REDUCE_MTE_H
#define QUANT_ALL_REDUCE_MTE_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "quant_all_reduce_tiling_data.h"
#include "utils.h"

namespace QuantAllReduceImpl {

using namespace AscendC;
constexpr static uint32_t UB_ALIGN = 32U; // UB按32B对齐
constexpr uint32_t FLOAT_PER_UB_ALIGN = 8U; // float格式下32B对齐需要 32/4 =8个
constexpr uint32_t BUFFER_NUM = 2U; // double buffer
constexpr static uint32_t PT_SCALE_SINGLE_NUM = 8U; // PT量化时， scale单次搬运8个
constexpr static uint32_t MXFP_SCALE_SINGLE_NUM = 32U; // MX量化时，scale单次搬运32个
constexpr static uint32_t SCALE_SINGLE_SIZE = 32U; // 一个sclae数据块固定32B，搬运要求32B对齐
constexpr static uint32_t X_SINGLE_SIZE = 1024U;  // 当前一个x数据块固定1024B = 128 * 8，为穿刺取值
constexpr static uint32_t SINGLE_NUM = 1024U;     // 当前一次搬运1024个x数据，为穿刺取值
constexpr static uint32_t PER_GROUP_SIZE = 128U;  // PT量化时，128个x数据共有一个scale
constexpr static uint32_t MX_SIZE = 32U;          // MX量化时，32个x数据共有一个scale
constexpr static uint32_t TWO_DIMS = 2U;          // 用于sclae反量化时boardcast的数组维度
constexpr static uint64_t MX_SCALES_LAST_DIM = 2U; // MX量化scales最后一维的大小

#define TemplateQuantAllReduceTypeClass typename XType, typename ScalesType, typename OutputType
#define TemplateQuantAllReduceType XType, ScalesType, OutputType

template<TemplateQuantAllReduceTypeClass>
class QuantAllReduceMte {
public:
    __aicore__ inline QuantAllReduceMte() {};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR scales, GM_ADDR output,
        TPipe *pipe, const QuantAllReduceTilingData *tilingData);
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
    uint64_t alignedXSize_{0};
    uint64_t scaleSize_{0};
    uint64_t xSliceSize_{0};
    uint64_t scaleSingleNums_{0};
    uint64_t scaleSliceNums_{0};
};

template <TemplateQuantAllReduceTypeClass>
__aicore__ inline void QuantAllReduceMte<TemplateQuantAllReduceType>::Init(GM_ADDR x, GM_ADDR scales,
    GM_ADDR output, TPipe *tPipe, const QuantAllReduceTilingData *tilingData)
{
    hcclContext_ = (__gm__ HcclA5OpResParam*)GetHcclContext<HCCL_GROUP_ID_0>();
    aivId_ = GetBlockIdx();
    auto&& tiliingDataIinfo = tilingData->quantAllReduceTilingInfo;

    // windows分区写入数据
    xSize_ = tiliingDataIinfo.bs * tiliingDataIinfo.hiddenSize * sizeof(XType);
    scaleSize_ = tiliingDataIinfo.bs * tiliingDataIinfo.scaleHiddenSize * sizeof(ScalesType);
    xSliceSize_ = xSize_ / hcclContext_->rankDim; // 每块的数据量大小
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
    alignedXSize_ = CeilAlign(xSliceSize_, SINGLE_NUM) * hcclContext_->rankDim;
    localWinScaleGMTensor_.SetGlobalBuffer((__gm__ ScalesType*)(localDataSpaceGm + alignedXSize_)); // GM上sclae数据跟在x后

    scaleSliceNums_ = scaleSize_ / (hcclContext_->rankDim * sizeof(ScalesType)); // 每块的数据量个数
    rsTotalBlocks_ = CeilDiv(xSliceSize_, X_SINGLE_SIZE); // 按每次搬运x的数据量分块，得到的总块数
    scaleSingleNums_ = SCALE_SINGLE_SIZE / sizeof(ScalesType); // 一块scale固定32B, 计算包含多少个数据
    rsRound_ = rsTotalBlocks_ / tiliingDataIinfo.aivNum;  // 计算总的数据分给所有核搬运需要的轮次数
    rsResRound_ = rsTotalBlocks_ % tiliingDataIinfo.aivNum; // 搬运的尾块数
    rsBlockNums_ = aivId_ < rsResRound_ ? rsRound_ + 1 : rsRound_; // 顺序分核，序号小的核多搬一轮
    tPipe->Reset();
    tPipe->InitBuffer(xInQueue_, BUFFER_NUM, X_SINGLE_SIZE); // 每次拷贝 1024B x; 128 * 8
    tPipe->InitBuffer(xQueue_, BUFFER_NUM, X_SINGLE_SIZE);    
    tPipe->InitBuffer(scaleInQue, BUFFER_NUM, UB_ALIGN);  // 每次拷贝 32B scale；4 * 8
    tPipe->InitBuffer(scaleQueue_, BUFFER_NUM, UB_ALIGN);     
    tPipe->InitBuffer(xOutQueue_, BUFFER_NUM, X_SINGLE_SIZE * sizeof(OutputType));
    tPipe->InitBuffer(writeStateBuf_, UB_ALIGN); // 状态位每一个按32B对齐
    tPipe->InitBuffer(readStateBuf_, hcclContext_->rankDim * UB_ALIGN);    // 每次读 rankDim 个状态位
    tPipe->InitBuffer(stateResetBuf_, hcclContext_->rankDim * UB_ALIGN);   // 用于清理状态区
    tPipe->InitBuffer(sumBuf_, SINGLE_NUM * sizeof(float)); // 用于Reduce_sum 求和，1024 * 4 = 4k
    tPipe->InitBuffer(brcbBuf_, SINGLE_NUM * sizeof(float)); // 用于scale BroadCast，1024 * 4 = 4k
    tPipe->InitBuffer(xCastBuf_, SINGLE_NUM * sizeof(float)); // 用于x Cast 成 fp32, 1024 * 4 = 4k
    tPipe->InitBuffer(xCastBf16Buf_, SINGLE_NUM * sizeof(bfloat16_t)); // 用于 x Cast 成 bf16, 1024 *2 = 2k
    tPipe->InitBuffer(xCastfp16Buf_, SINGLE_NUM * sizeof(float16_t)); // 用于 x Cast 成 fp16, 1024 *2 = 2k

    // 对于mx量化，fp8_e8m0_t数据类型转换成float数据类型，需要进行转换：float_e8m0_t -> bfloat16_t -> float
    if constexpr (AscendC::IsSameType<ScalesType, fp8_e8m0_t>::value) {
        tPipe->InitBuffer(tempBf16ScaleBuf_, MXFP_SCALE_SINGLE_NUM * sizeof(bfloat16_t));
        tPipe->InitBuffer(castScaleBuf_, MXFP_SCALE_SINGLE_NUM * sizeof(float));
    } else { // PT量化
        tPipe->InitBuffer(castScaleBuf_, PT_SCALE_SINGLE_NUM * sizeof(float));
    }

    tPipe->InitBuffer(outputCastBuf_, SINGLE_NUM * sizeof(OutputType));
    sumTensor_ = sumBuf_.Get<float>();
    stateResetTensor_ = stateResetBuf_.Get<float>();
    Duplicate<float>(stateResetTensor_, (float)0.0, static_cast<uint32_t>(hcclContext_->rankDim * FLOAT_PER_UB_ALIGN)); // 用于状态区清零
}

template <TemplateQuantAllReduceTypeClass>
__aicore__ inline void QuantAllReduceMte<TemplateQuantAllReduceType>::CopyDataToWin(uint64_t xOffset,
    uint64_t scaleOffset)
{
    // 每个核需要搬运的数据块数
    for(uint64_t curBlock = 0; curBlock < rsBlockNums_; ++curBlock) {
        uint64_t curXOffset = xOffset + curBlock * X_SINGLE_SIZE; // 计算现在搬第几个x
        uint64_t curScaleOffset = scaleOffset + curBlock * scaleSingleNums_; // 计算现在搬第几个scale
        for(uint64_t curRank = 0; curRank < hcclContext_->rankDim; ++curRank) {
        // 先拷贝data数据， 再拷贝scales
            uint64_t curRankXOffset = curXOffset + curRank * xSliceSize_; 
            uint64_t curRankScaleOffset = curScaleOffset + curRank * scaleSliceNums_;

            ////////// x 从 GM -> UB -> Win ////////////
            xTmpTensor_ = xQueue_.AllocTensor<XType>();
            DataCopy(xTmpTensor_, xGMTensor_[curRankXOffset], X_SINGLE_SIZE);
            xQueue_.EnQue(xTmpTensor_);
            xTmpTensor_ = xQueue_.DeQue<XType>();
            DataCopy(localWinXGMTensor_[curRankXOffset], xTmpTensor_, X_SINGLE_SIZE);
            xQueue_.FreeTensor<XType>(xTmpTensor_);

            ////////// scale 从 GM -> UB -> Win ////////////
            scaleTmpTensor_ = scaleQueue_.AllocTensor<ScalesType>();
            DataCopy(scaleTmpTensor_, scalesGMTensor_[curRankScaleOffset], scaleSingleNums_);
            scaleQueue_.EnQue(scaleTmpTensor_);
            scaleTmpTensor_ = scaleQueue_.DeQue<ScalesType>();
            DataCopy(localWinScaleGMTensor_[curRankScaleOffset], scaleTmpTensor_, scaleSingleNums_);
            scaleQueue_.FreeTensor<ScalesType>(scaleTmpTensor_);
        }
    }
}

template <TemplateQuantAllReduceTypeClass>
__aicore__ inline void QuantAllReduceMte<TemplateQuantAllReduceType>::WriteStatusToWin()
{
    uint32_t coreOffset = aivId_ * hcclContext_->rankDim; // Win区大小为 aivNum * rankDim, 此处计算核偏移
    // 遍历每一张卡，给每一张卡都要写入状态
    for (uint32_t curRank = 0; curRank < hcclContext_->rankDim; ++curRank) {
        // 写入状态到对端，每个核写一个状态，表示自己的数据块已经写完
        LocalTensor<float> statusTensor = writeStateBuf_.Get<float>();
        DataCopy<float>(statusTensor, stateResetTensor_, FLOAT_PER_UB_ALIGN); // 先重置statusTensor数据
        SyncFunc<AscendC::HardEvent::MTE2_S>();
        statusTensor(0) = (float)1.0;  // 用1标识
        GM_ADDR remoteWinStateGM = (GM_ADDR)hcclContext_->windowsOut[curRank];  // 获取当前要写对端卡的状态区地址
        GlobalTensor<float> stateGMTensor;
        stateGMTensor.SetGlobalBuffer((__gm__ float*)remoteWinStateGM);
        // 不同卡上的核的状态写到相邻位置，读时可以一次读rankDim个状态, 状态区大小设计为 aivNum * ranDim
        uint64_t curOffset = (coreOffset + hcclContext_->rankId) * FLOAT_PER_UB_ALIGN; // 当前核偏移 + 卡偏移， 按32B对齐
        SyncFunc<AscendC::HardEvent::S_MTE3>();
        DataCopy(stateGMTensor[curOffset], statusTensor, FLOAT_PER_UB_ALIGN);  // 按32B对齐拷贝
    }
}

template <TemplateQuantAllReduceTypeClass>
__aicore__ inline void QuantAllReduceMte<TemplateQuantAllReduceType>::ReadStatus()
{
    GM_ADDR stateGM = (GM_ADDR)hcclContext_->windowsOut[hcclContext_->rankId];  // 获取本卡的状态区用于读取
    GlobalTensor<float> selfStatusWinTensor;
    uint32_t offset = aivId_ * hcclContext_->rankDim * FLOAT_PER_UB_ALIGN; // 获取当前核所需读取状态位的头地址，状态按32B对齐
    selfStatusWinTensor.SetGlobalBuffer((__gm__ float*)(stateGM));
    LocalTensor<float> statusTensor = readStateBuf_.Get<float>();
    float flag = 0; // 用于计算状态和
    uint32_t statusCnt = hcclContext_->rankDim * FLOAT_PER_UB_ALIGN; // 一次读rankDim个，按32B对齐
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
    DataCopy<float>(selfStatusWinTensor[offset], stateResetTensor_, statusCnt); // 相关状态区重新置零
}

template <TemplateQuantAllReduceTypeClass>
__aicore__ inline void QuantAllReduceMte<TemplateQuantAllReduceType>::CastToFloat(LocalTensor<XType> &xTmpTensor,
    LocalTensor<ScalesType> &scaleTmpTensor)
{
    xCastTemp_ = xCastBuf_.Get<float>();
    scaleCalTensor_ = brcbBuf_.Get<float>();
    LocalTensor<bfloat16_t> xTempBf16Tensor = xCastBf16Buf_.Get<bfloat16_t>();
    LocalTensor<float16_t> xTempfp16Tensor = xCastfp16Buf_.Get<float16_t>();
    LocalTensor<float> castLocalScale = castScaleBuf_.Get<float>();
    

    if constexpr (AscendC::IsSameType<XType, fp4x2_e2m1_t>::value ||
        AscendC::IsSameType<XType, fp4x2_e1m2_t>::value) {
        // fp4 -> bf16 -> fp32
        Duplicate<bfloat16_t>(xTempBf16Tensor, (bfloat16_t)0.0, SINGLE_NUM);
        Cast(xTempBf16Tensor, xTmpTensor, RoundMode::CAST_NONE, SINGLE_NUM);
        PipeBarrier<PIPE_V>();
        Cast(xCastTemp_, xTempBf16Tensor, RoundMode::CAST_NONE, SINGLE_NUM);
        PipeBarrier<PIPE_V>();
    } else if (AscendC::IsSameType<XType, int8_t>::value) { 
        // int8 -> fp16 -> fp32
        Duplicate<float16_t>(xTempfp16Tensor, (float16_t)0.0, SINGLE_NUM);
        Cast(xTempfp16Tensor, xTmpTensor, RoundMode::CAST_NONE, SINGLE_NUM);
        PipeBarrier<PIPE_V>();
        Cast(xCastTemp_, xTempfp16Tensor, RoundMode::CAST_NONE, SINGLE_NUM);
        PipeBarrier<PIPE_V>();
    } else {
        // fp8 -> fp32
        Duplicate<float>(xCastTemp_, (float)0.0, SINGLE_NUM);
        PipeBarrier<PIPE_V>();
        Cast(xCastTemp_, xTmpTensor, RoundMode::CAST_NONE, SINGLE_NUM);
        PipeBarrier<PIPE_V>();
    }

    if constexpr (AscendC::IsSameType<ScalesType, fp8_e8m0_t>::value) {
        // fp8_e8m0 -> bf16 -> fp32
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
template <TemplateQuantAllReduceTypeClass>
__aicore__ inline void QuantAllReduceMte<TemplateQuantAllReduceType>::ReadRomoteDataAddDequant(uint64_t xOffset,
    uint64_t scaleOffset)
{
    ReadStatus(); // 读状态位，软同步
    // allgather 需要全部数据
    for (uint64_t rankIdx = 0; rankIdx < hcclContext_ -> rankDim; ++rankIdx) {
        // 遍历需要搬运的数据块
        for (uint64_t curBlock = 0; curBlock < rsBlockNums_; ++curBlock) {
            uint64_t curXOffset = xOffset + curBlock * X_SINGLE_SIZE;  // 对于fp4需要另外算偏移
            uint64_t curScaleOffset = scaleOffset + curBlock * scaleSingleNums_;
            Duplicate<float>(sumTensor_, (float)0.0, SINGLE_NUM); // sumTensor 清零
            // 遍历每张卡，读取其Win区的数据，采取错卡序读取，从自己卡上读起
            // rank0: [0,1]
            // rank1: [1,0]
            int startRankId = hcclContext_->rankId;
            for (int i = 0; i < hcclContext_->rankDim; ++i) { 
                int remoteRankId = (startRankId + i) % hcclContext_->rankDim;
                uint64_t curRankXOffset = curXOffset + rankIdx * xSliceSize_;
                uint64_t curRankScaleOffset = curScaleOffset + rankIdx * scaleSliceNums_;

                // 获取对端Win区中 x 和 sclae的地址
                GlobalTensor<XType> remoteWinGMTensor_;
                GM_ADDR remoteXWin = (GM_ADDR)(hcclContext_->windowsIn[remoteRankId]);
                remoteWinXTensor_.SetGlobalBuffer((__gm__ XType*)remoteXWin);
                GM_ADDR remoteScaleWin = (GM_ADDR)(hcclContext_->windowsIn[remoteRankId] + alignedXSize_);
                remoteWinScaleTensor_.SetGlobalBuffer((__gm__ ScalesType*)remoteScaleWin);

                ////////////// 读取 x 从 Win -> UB //////////
                LocalTensor<XType> xTmpTensor = xInQueue_.AllocTensor<XType>();
                xInQueue_.EnQue(xTmpTensor);
                DataCopy(xTmpTensor, remoteWinXTensor_[curRankXOffset], SINGLE_NUM);
                xTmpTensor = xInQueue_.DeQue<XType>();

                ////////////// 读取 scale 从 Win -> UB //////////
                LocalTensor<ScalesType> scaleTmpTensor = scaleInQue.AllocTensor<ScalesType>();
                DataCopy(scaleTmpTensor, remoteWinScaleTensor_[curRankScaleOffset], scaleSingleNums_);
                scaleInQue.EnQue(scaleTmpTensor);
                scaleTmpTensor = scaleInQue.DeQue<ScalesType>();

                // Cast成float计算
                CastToFloat(xTmpTensor, scaleTmpTensor);
                PipeBarrier<PIPE_V>();
                // 反量化
                Mul(xCastTemp_, xCastTemp_, scaleCalTensor_, SINGLE_NUM);
                PipeBarrier<PIPE_V>();
                // 累加
                Add(sumTensor_, sumTensor_, xCastTemp_, SINGLE_NUM);
                xInQueue_.FreeTensor(xTmpTensor);
                scaleInQue.FreeTensor(scaleTmpTensor);
                PipeBarrier<PIPE_V>();
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
            DataCopy(outputTensor_[curXOffset + rankIdx * xSliceSize_], xOutTensor_, SINGLE_NUM);
            xOutQueue_.FreeTensor(xOutTensor_);
        }
    }
}

template <TemplateQuantAllReduceTypeClass>
__aicore__ inline void QuantAllReduceMte<TemplateQuantAllReduceType>::Process()
{
    // 纯AIV过程
    if ASCEND_IS_AIC {
        return;
    }

    // 顺序切块，序列小的核处理尾块
    // 例子：20个数据块，9个核，rsRound = 2轮，尾块rsResRound = 2块
    // 此时 0 核搬运数据块 0，1，2；1 核搬运数据块 3，4，5；2 核搬运数据块 6，7。 .....
    uint64_t blockOffset = aivId_ * rsRound_ + (aivId_ < rsResRound_ ? aivId_ : rsResRound_); // 计算首块偏移
    uint64_t xOffset = blockOffset * X_SINGLE_SIZE; // 块数 * 一块有多少数据，得到要搬第几个x数据
    uint64_t scaleOffset = blockOffset * scaleSingleNums_; // 计算要搬第几个 scale
    // 一次性拷贝完所有数据到本地卡win区
    CopyDataToWin(xOffset, scaleOffset);
    PipeBarrier<PIPE_ALL>();
    // 写入状态到对端
    WriteStatusToWin();
    // 等待状态区同步，进行反量化ReduceSum
    ReadRomoteDataAddDequant(xOffset, scaleOffset);
}
} // QuantAllReduceImpl
#endif  // QUANT_ALL_REDUCE_MTE_H