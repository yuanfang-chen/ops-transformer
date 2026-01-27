/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fp8_dynamic_quant_pertoken.h
 * \brief
 */

#ifndef FP8_DYNAMIC_QUANT_PERTOKEN_H
#define FP8_DYNAMIC_QUANT_PERTOKEN_H

namespace MC2KernelTemplate {
using namespace AscendC;
template <typename quantInputDataType, typename quantOutputDataType>
class Fp8DynamicQuantPertoken {
protected:
    static constexpr uint32_t ALIGN_NUM = 8;
    static constexpr uint32_t TWO_FACTOR = 2;
    static constexpr uint32_t ONE_FACTOR = 1;
    static constexpr uint32_t UB_DATABLOCK = 32;
    static constexpr uint32_t COMPARE_ALIGN_LEN = 256;
    static constexpr uint32_t VECTOR_UB_SIZE = AscendC::TOTAL_UB_SIZE;

    static constexpr float FP8_E5M2_MAX_VALUE = 57344.0f;
    static constexpr float FP8_E4M3FN_MAX_VALUE = 448.0f;

    template <typename T>
    __aicore__ static inline T Max(T a, T b)
    {
        return (a > b) ? (a) : (b);
    }

    template <typename T>
    __aicore__ static inline T Min(T a, T b)
    {
        return (a > b) ? (b) : (a);
    }

    template <typename T>
    __aicore__ static inline T Ceil(T x, T y)
    {
        return (x + y - 1) / y;
    }

    template <AscendC::HardEvent event>
    __aicore__ static inline void SyncFunc()
    {
        int32_t eventID = static_cast<int32_t>(GetTPipePtr()->FetchEventID(event));
        AscendC::SetFlag<event>(eventID);
        AscendC::WaitFlag<event>(eventID);
    }

    TPipe *tPipe_;

    GM_ADDR quantInputAddr_;
    GM_ADDR smoothScaleAddr_;
    GM_ADDR quantOutputAddr_;
    GM_ADDR quantOutputScaleAddr_;

    GlobalTensor<quantInputDataType> quantInputGM_;
    GlobalTensor<quantOutputDataType> quantOutputGM_;
    GlobalTensor<float> quantOutputScaleGM_;

    TQue<QuePosition::VECIN, 1> rawInputQue_;     // 存放原始 float16 数据
    TQue<QuePosition::VECOUT, 1> quantOutputQue_; // 存放量化后的 fp8 数据
    TQue<QuePosition::VECOUT, 1> quantScaleQue_;  // 存放输出的 scale

    TBuf<TPosition::VECCALC> floatDataBuf_;
    TBuf<TPosition::VECCALC> workBuf_;


    uint64_t rowNum_ = 0; // 二维Tensor的第二维
    uint64_t colNum_ = 0; // 二维Tensor的第二维

    uint64_t usedCoreAivNum_ = 0; // 使用的aiv核数

    uint64_t calBuffSize_ = 0;      // tiling侧预先计算出的占用的UB空间
    uint64_t rowsThisCore_ = 0;     // 当前核负责的总行数
    uint64_t startRowThisCore_ = 0; // 当前核负责的起始行索引

    float recipFP8MaxLimit_ = 0.0f;
    float fp8MaxLimit_ = 0.0f;

    __aicore__ inline void SetMaxValue();

    __aicore__ inline void ProcessOneToken();

public:
    __aicore__ inline Fp8DynamicQuantPertoken(TPipe *tPipe) : tPipe_(tPipe)
    {
    }

    __aicore__ inline void Init(GM_ADDR quantInputAddr, GM_ADDR smoothScaleAddr, GM_ADDR quantOutputAddr,
                                GM_ADDR quantOutputScaleAddr, uint64_t rowNum, uint64_t colNum, uint64_t calBuffSize);

    __aicore__ inline void Process();

    __aicore__ inline void Destroy();
};

template <typename quantInputDataType, typename quantOutputDataType>
__aicore__ inline void Fp8DynamicQuantPertoken<quantInputDataType, quantOutputDataType>::Init(
    GM_ADDR quantInputAddr, GM_ADDR smoothScaleAddr, GM_ADDR quantOutputAddr, GM_ADDR quantOutputScaleAddr,
    uint64_t rowNum, uint64_t colNum, uint64_t calBuffSize)
{
    tPipe_->Reset();
    if ASCEND_IS_AIC {
        return;
    }

    // 变量初始化
    this->rowNum_ = rowNum;
    this->colNum_ = colNum;
    // 暂时没有使用，考虑删除
    this->calBuffSize_ = calBuffSize;
    this->quantInputAddr_ = quantInputAddr;
    // 预留参数，实际外部调用传入为空
    this->smoothScaleAddr_ = smoothScaleAddr;
    this->quantOutputAddr_ = quantOutputAddr;
    this->quantOutputScaleAddr_ = quantOutputScaleAddr;

    uint64_t totalCores = static_cast<uint64_t>(GetBlockNum() * TWO_FACTOR);
    this->usedCoreAivNum_ = (rowNum < totalCores) ? rowNum : totalCores;

    // 2. 均匀分配任务到各个核
    uint32_t coreIdx = GetBlockIdx();
    uint64_t avgRows = rowNum / this->usedCoreAivNum_;
    uint32_t tailRows = rowNum % this->usedCoreAivNum_;

    // 每个核计算自己的任务范围，如10行数据，使用4个核（0,1,2,3），第1个aiv核处理3,4,5行，startRowThisCore_的起始索引为3
    this->rowsThisCore_ = avgRows + (coreIdx < tailRows ? 1 : 0);
    this->startRowThisCore_ = coreIdx * avgRows + (coreIdx < tailRows ? coreIdx : tailRows);

    SetMaxValue();
}

template <typename quantInputDataType, typename quantOutputDataType>
__aicore__ inline void Fp8DynamicQuantPertoken<quantInputDataType, quantOutputDataType>::Process()
{
    if (GetBlockIdx() >= this->usedCoreAivNum_) {
        return;
    }
    ProcessOneToken();
}

template <typename quantInputDataType, typename quantOutputDataType>
__aicore__ inline void Fp8DynamicQuantPertoken<quantInputDataType, quantOutputDataType>::Destroy()
{
    rawInputQue_.FreeAllEvent();
    quantOutputQue_.FreeAllEvent();
    quantScaleQue_.FreeAllEvent();
}

template <typename quantInputDataType, typename quantOutputDataType>
__aicore__ inline void Fp8DynamicQuantPertoken<quantInputDataType, quantOutputDataType>::SetMaxValue()
{
    if constexpr (IsSameType<quantOutputDataType, fp8_e5m2_t>::value) {
        this->recipFP8MaxLimit_ = static_cast<float>(1.0) / FP8_E5M2_MAX_VALUE;
        this->fp8MaxLimit_ = FP8_E5M2_MAX_VALUE;
    } else if constexpr (IsSameType<quantOutputDataType, fp8_e4m3fn_t>::value) {
        this->recipFP8MaxLimit_ = static_cast<float>(1.0) / FP8_E4M3FN_MAX_VALUE;
        this->fp8MaxLimit_ = FP8_E4M3FN_MAX_VALUE;
    }
}

/**
 * @brief 单核一次处理一行（UB空间足够的情况）
 *
 * @return __aicore__
 */
template <typename quantInputDataType, typename quantOutputDataType>
__aicore__ inline void Fp8DynamicQuantPertoken<quantInputDataType, quantOutputDataType>::ProcessOneToken()
{
    // 1.初始化资源，计算对齐后的系数
    uint32_t inputSize =
        Ceil(static_cast<uint32_t>(this->colNum_ * sizeof(quantInputDataType)), UB_DATABLOCK) * UB_DATABLOCK;
    uint32_t outputSize =
        Ceil(static_cast<uint32_t>(this->colNum_ * sizeof(quantOutputDataType)), UB_DATABLOCK) * UB_DATABLOCK;
    //
    uint32_t floatBufSize = Ceil(static_cast<uint32_t>(this->colNum_ * sizeof(float)), UB_DATABLOCK) * UB_DATABLOCK;

    quantInputGM_.SetGlobalBuffer((__gm__ quantInputDataType *)this->quantInputAddr_);
    quantOutputGM_.SetGlobalBuffer((__gm__ quantOutputDataType *)this->quantOutputAddr_);
    quantOutputScaleGM_.SetGlobalBuffer((__gm__ float *)this->quantOutputScaleAddr_);

    tPipe_->InitBuffer(workBuf_, UB_DATABLOCK); // Reduce 32
    tPipe_->InitBuffer(floatDataBuf_, floatBufSize);
    tPipe_->InitBuffer(rawInputQue_, ONE_FACTOR, inputSize);
    tPipe_->InitBuffer(quantOutputQue_, ONE_FACTOR, outputSize);
    tPipe_->InitBuffer(quantScaleQue_, ONE_FACTOR,
                       Ceil(static_cast<uint32_t>(this->rowsThisCore_ * sizeof(float)), UB_DATABLOCK) * UB_DATABLOCK);

    LocalTensor<float> workData = workBuf_.Get<float>();
    LocalTensor<float> floatBufData = floatDataBuf_.Get<float>();

    LocalTensor<float> coreQuantScales = quantScaleQue_.AllocTensor<float>();
    LocalTensor<quantInputDataType> rawInputTensor = rawInputQue_.AllocTensor<quantInputDataType>();
    LocalTensor<quantOutputDataType> quantOut = quantOutputQue_.AllocTensor<quantOutputDataType>();

    // 2.逐行处理
    for (uint64_t r = 0; r < this->rowsThisCore_; ++r) {
        float maxValue, minValue;
        uint64_t globalRowIdx = this->startRowThisCore_ + r;
        PipeBarrier<PIPE_V>();
        DataCopyPad<quantInputDataType, PaddingMode::Normal>(
            rawInputTensor, quantInputGM_[globalRowIdx * this->colNum_],
            {1, static_cast<uint32_t>(this->colNum_ * sizeof(quantInputDataType)), 0, 0, 0}, {false, 0, 0, 0});
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        Cast(floatBufData, rawInputTensor, RoundMode::CAST_NONE, this->colNum_);
        PipeBarrier<PIPE_V>();

        AscendC::ReduceMax<float>(workData, floatBufData, floatBufData, this->colNum_, false);
        SyncFunc<AscendC::HardEvent::V_S>();
        maxValue = workData.GetValue(0);
        AscendC::ReduceMin<float>(workData, floatBufData, floatBufData, this->colNum_, false);
        SyncFunc<AscendC::HardEvent::V_S>();
        minValue = workData.GetValue(0);

        float rowMax = Max(maxValue, (-minValue));
        // 量化系数
        float scale = (rowMax > 0.0f) ? (rowMax * this->recipFP8MaxLimit_) : 1.0f;
        float recipScale = (rowMax > 0.0f) ? (this->fp8MaxLimit_ / rowMax) : 1.0f;
        coreQuantScales.SetValue(r, scale);
        SyncFunc<AscendC::HardEvent::S_V>();

        // 开始量化
        Muls(floatBufData, floatBufData, recipScale, this->colNum_);
        PipeBarrier<PIPE_V>();
        Cast(quantOut, floatBufData, RoundMode::CAST_RINT, this->colNum_);
        PipeBarrier<PIPE_V>();
        // 搬出量化结果
        SyncFunc<AscendC::HardEvent::V_MTE3>();
        DataCopyPad<quantOutputDataType, PaddingMode::Normal>(
            quantOutputGM_[globalRowIdx * this->colNum_], quantOut,
            {1, static_cast<uint32_t>(this->colNum_ * sizeof(quantOutputDataType)), 0, 0, 0});
    }
    rawInputQue_.FreeTensor(rawInputTensor);

    DataCopyExtParams outScaleParams = {1, static_cast<uint32_t>(this->rowsThisCore_ * sizeof(float)), 0, 0, 0};
    DataCopyPad<float, PaddingMode::Normal>(quantOutputScaleGM_[this->startRowThisCore_], coreQuantScales,
                                            outScaleParams);
    SyncFunc<AscendC::HardEvent::MTE3_S>();

    quantOutputQue_.FreeTensor(quantOut);
    quantScaleQue_.FreeTensor(coreQuantScales);
}

} // namespace MC2KernelTemplate
#endif
