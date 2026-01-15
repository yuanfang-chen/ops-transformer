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

constexpr uint32_t ALIGN_NUM = 8;
constexpr uint32_t TWO_FACTOR = 2;
constexpr uint32_t ONE_FACTOR = 1;
constexpr uint32_t UB_DATABLOCK = 32;
constexpr uint32_t COMPARE_ALIGN_LEN = 256;
constexpr uint32_t VECTOR_UB_SIZE = AscendC::TOTAL_UB_SIZE;

constexpr float FP8_E5M2_MAX_VALUE = 57344.0f;
constexpr float FP8_E4M3FN_MAX_VALUE = 448.0f;

template <typename T>
__aicore__ inline T Max(T a, T b)
{
    return (a > b) ? (a) : (b);
}

template <typename T>
__aicore__ inline T Min(T a, T b)
{
    return (a > b) ? (b) : (a);
}

template <typename T1, typename T2>
__aicore__ inline T2 Ceil(T1 x, T1 y)
{
    return (x + y - 1) / y;
}

template <typename quantInputDataType, typename quantOutputDataType>
class Fp8DynamicQuantPertoken {
protected:
    TPipe *tPipe_;

    GM_ADDR quantInputAddr_;
    GM_ADDR smoothScaleAddr_;
    GM_ADDR quantOutputAddr_;
    GM_ADDR quantOutputScaleAddr_;

    GlobalTensor<quantInputDataType> quantInputGM_;
    GlobalTensor<quantInputDataType> smoothScaleGM_;
    GlobalTensor<quantOutputDataType> quantOutputGM_;
    GlobalTensor<float> quantOutputScaleGM_;

    TQue<QuePosition::VECIN, 1> rawInputQue_;     // 存放原始 float16 数据
    TQue<QuePosition::VECIN, 1> smoothScaleQue_;  // 存放平滑系数
    TQue<QuePosition::VECOUT, 1> quantOutputQue_; // 存放量化后的 fp8 数据
    TQue<QuePosition::VECOUT, 1> quantScaleQue_;  // 存放输出的 scale

    TBuf<TPosition::VECCALC> floatDataBuf_;
    TBuf<TPosition::VECCALC> workBuf_;
    TBuf<TPosition::VECCALC> tempStatBuf_;
    TBuf<TPosition::VECCALC> maskBuf_; // 存储掩码块


    uint64_t rowNum_ = 0; // 二维Tensor的第二维
    uint64_t colNum_ = 0; // 二维Tensor的第二维

    uint64_t usedCoreAivNum_ = 0; // 使用的aiv核数
    bool hasSmooth_ = false;      // 是否有平滑系数

    uint64_t outLoopNum_ = 0;      // 表示循环执行量化操作的次数
    uint64_t tailUsedCoreNum_ = 0; // 表示无法平分的余数
    uint32_t procRows_ = 0;
    uint32_t maxProcRows_ = 0; // 当前UB空间可以容纳aiv单次处理的最大长度
    uint64_t procRowsFirstTail_ = 0;
    uint64_t calBuffSize_ = 0;      // tiling侧预先计算出的占用的UB空间
    uint64_t rowsThisCore_ = 0;     // 当前核负责的总行数
    uint64_t startRowThisCore_ = 0; // 当前核负责的起始行索引

    float recipFP8MaxLimit_ = 0.0f;
    float fp8MaxLimit_ = 0.0f;

public:
    __aicore__ inline Fp8DynamicQuantPertoken(TPipe *tPipe) : tPipe_(tPipe)
    {
    }

    __aicore__ inline void SetMaxValue()
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
     * @brief 获取当前剩余UB空间，用于判断vector单次能够处理多少行
     *
     */
    __aicore__ inline uint32_t GetMaxProcRows()
    {
        uint32_t curUbSize =
            VECTOR_UB_SIZE - this->calBuffSize_ - (COMPARE_ALIGN_LEN - 1 + UB_DATABLOCK - 1) * TWO_FACTOR;
        // 对应每一行需要的UB空间大小，Tque,TBuf对应的空间
        uint32_t ubSizePerRow =
            (3 * sizeof(float) + sizeof(quantInputDataType) + sizeof(quantOutputDataType)) * this->colNum_ +
            sizeof(float) + sizeof(float) + sizeof(uint8_t);
        ubSizePerRow *= TWO_FACTOR;
        return curUbSize / ubSizePerRow;
    }

    /**
     * @brief 单个vector一次处理一行数据所需要的空间
     *
     */
    __aicore__ inline uint32_t GetMaxProcCols()
    {
        uint32_t curUbSize =
            VECTOR_UB_SIZE - this->calBuffSize_ - (COMPARE_ALIGN_LEN - 1 + UB_DATABLOCK - 1) * TWO_FACTOR;
        // 单行切分时，每列数据需要的空间
        uint32_t ubSizePerCol =
            (3 * sizeof(float) + sizeof(quantInputDataType) + sizeof(quantOutputDataType)) * TWO_FACTOR;
        return curUbSize / ubSizePerCol;
    }

    /**
     * @brief 对长度为count的float类型的LocalTensor进行inplace求最大值
     *
     */
    __aicore__ inline void ReduceMaxInplace(const LocalTensor<float> &srcLocal, uint32_t count)
    {
        uint64_t repsFp32 = count >> 6;       // 6 is count / elemPerRefFp32
        uint64_t offsetsFp32 = repsFp32 << 6; // 6 is repsFp32 * elemPerRefFp32
        uint64_t remsFp32 = count & 0x3f;     // 0x3f 63, count % elemPerRefFp32
        const uint64_t elemPerRefFp32 = 64UL; // 256 bit / sizeof(float)
        if (likely(repsFp32 > 1)) {
            // 8 is rep stride
            Max(srcLocal, srcLocal[elemPerRefFp32], srcLocal, elemPerRefFp32, repsFp32 - 1, {1, 1, 1, 0, 8, 0});
            PipeBarrier<PIPE_V>();
        }
        if (unlikely(remsFp32 > 0) && unlikely(offsetsFp32 > 0)) {
            Max(srcLocal, srcLocal[offsetsFp32], srcLocal, remsFp32, 1, {1, 1, 1, 0, 8, 0});
            PipeBarrier<PIPE_V>();
        }
        uint32_t mask = (repsFp32 > 0) ? elemPerRefFp32 : count;
        // 8 is rep stride
        WholeReduceMax(srcLocal, srcLocal, mask, 1, 8, 1, 8);
    }

    /**
     * @brief 动态量化的计算, 支持多行
     *
     */
    __aicore__ inline void DynamicQuantMultiToken(uint32_t rowBatch, uint32_t totalDataCount,
                                                  LocalTensor<quantInputDataType> localInputRaw,
                                                  LocalTensor<float> localSmoothScale,
                                                  LocalTensor<quantOutputDataType> localOutput,
                                                  LocalTensor<float> localScale)
    {
        // 绑定成员变量到局部 Tensor
        LocalTensor<float> floatData = floatDataBuf_.Get<float>();
        LocalTensor<float> workBuf = workBuf_.Get<float>();
        LocalTensor<float> tempStat = tempStatBuf_.Get<float>();
        LocalTensor<uint8_t> mask = maskBuf_.Get<uint8_t>();
        // 使用tensor前n个数据参与计算，设置count时，需要保证count个元素所占空间256字节对齐，Co
        uint32_t compareCnt = (Ceil(rowBatch * sizeof(float), COMPARE_ALIGN_LEN) * COMPARE_ALIGN_LEN) / sizeof(float);

        // 1. 输入数据类型转换为FLOAT
        Cast(floatData, localInputRaw, RoundMode::CAST_NONE, totalDataCount);
        PipeBarrier<PIPE_V>();

        const uint32_t columnsPerRow = totalDataCount / rowBatch;
        const uint32_t broadcastShape[TWO_FACTOR] = {rowBatch, columnsPerRow};
        const uint32_t sourceShape[TWO_FACTOR] = {rowBatch, 1};

        // 2. 乘平滑系数，按行平滑
        if (this->hasSmooth_) {
            Cast(tempStat, localSmoothScale, RoundMode::CAST_NONE, rowBatch);
            Broadcast<float, TWO_FACTOR, 1, false>(workBuf, tempStat, broadcastShape, sourceShape);
            Mul(floatData, floatData, workBuf, totalDataCount);
            PipeBarrier<PIPE_V>();
        }

        // 3. 计算绝对值和单行最大值
        Abs(tempStat, floatData, totalDataCount);
        PipeBarrier<PIPE_V>();
        // 结果存放在 workBuf (形状为 [rowBatch, 1])
        ReduceMax<float, AscendC::Pattern::Reduce::AR, true>(workBuf, tempStat, broadcastShape, false);
        // 防除零操作1，找到最大值为0的位置
        CompareScalar(mask, workBuf, 0.0f, AscendC::CMPMODE::NE, compareCnt);
        PipeBarrier<PIPE_V>();

        // 4. 获取结果量化参数  rowMax / fp8Max
        Duplicate(tempStat, this->recipFP8MaxLimit_, rowBatch);
        // 防除零操作2，直接将最大值为0的位置设置为最大值，这样相乘后得到的量化系数为1，且对于第五步计算得到的量化乘数也为1
        Select(workBuf, mask, workBuf, this->fp8MaxLimit_, AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, rowBatch);
        Mul(localScale, workBuf, tempStat, rowBatch);
        PipeBarrier<PIPE_V>();

        // 5. 计算量化的乘数 fp8Max / rowMax
        Duplicate(tempStat, this->fp8MaxLimit_, rowBatch);
        Div(tempStat, tempStat, workBuf, rowBatch);
        PipeBarrier<PIPE_V>();

        // 6. 获取量化结果 x * (fp8Max/rowMax) -> fp8
        Broadcast<float, TWO_FACTOR, 1, false>(workBuf, tempStat, broadcastShape, sourceShape);
        Mul(floatData, floatData, workBuf, totalDataCount);
        PipeBarrier<PIPE_V>();

        Cast(localOutput, floatData, RoundMode::CAST_RINT, totalDataCount);
    }

    __aicore__ inline void Init(GM_ADDR quantInputAddr, GM_ADDR smoothScaleAddr, GM_ADDR quantOutputAddr,
                                GM_ADDR quantOutputScaleAddr, uint64_t rowNum, uint64_t colNum, uint64_t calBuffSize)
    {
        if ASCEND_IS_AIC {
            return;
        }

        // 变量初始化
        this->rowNum_ = rowNum;
        this->colNum_ = colNum;
        this->calBuffSize_ = calBuffSize;
        this->quantInputAddr_ = quantInputAddr;
        this->smoothScaleAddr_ = smoothScaleAddr;
        this->quantOutputAddr_ = quantOutputAddr;
        this->quantOutputScaleAddr_ = quantOutputScaleAddr;
        this->hasSmooth_ = (smoothScaleAddr != nullptr);


        uint64_t totalCores = static_cast<uint64_t>(GetBlockNum() * TWO_FACTOR);
        this->usedCoreAivNum_ = (rowNum < totalCores) ? rowNum : totalCores;

        this->maxProcRows_ = GetMaxProcRows();

        // 2. 均匀分配任务到各个核
        uint32_t coreIdx = GetBlockIdx();
        uint64_t avgRows = rowNum / this->usedCoreAivNum_;
        uint32_t tailRows = rowNum % this->usedCoreAivNum_;

        // 每个核计算自己的任务范围，如10行数据，使用4个核（0,1,2,3），第1个aiv核处理3,4,5行，startRowThisCore_的起始索引为3
        this->rowsThisCore_ = avgRows + (coreIdx < tailRows ? 1 : 0);
        this->startRowThisCore_ = coreIdx * avgRows + (coreIdx < tailRows ? coreIdx : tailRows);

        // 3. 确定内部循环步长（受UB限制）
        if (this->maxProcRows_ > 0) {
            this->procRows_ = (this->rowsThisCore_ < this->maxProcRows_) ? this->rowsThisCore_ : this->maxProcRows_;
        }

        SetMaxValue();
    }

    __aicore__ inline void Process()
    {
        if (GetBlockIdx() >= this->usedCoreAivNum_) {
            return;
        }
        if (this->maxProcRows_ > 0) {
            ProcessMultiTokens();
        } else {
            ProcessLargeRows();
        }
    }

    __aicore__ inline void ProcessMultiTokens()
    {
        for (uint64_t offset = 0; offset < this->rowsThisCore_; offset += this->procRows_) {
            uint32_t curRows =
                static_cast<uint32_t>(Min(static_cast<uint64_t>(this->procRows_), this->rowsThisCore_ - offset));

            InitTileResources(this->startRowThisCore_ + offset, curRows);
            ProcessTile(curRows, this->colNum_);
        }
    }

    __aicore__ inline void InitTileResources(uint64_t globalRowOffset, uint64_t curRows)
    {
        tPipe_->Reset();

        uint64_t inputOffset = globalRowOffset * this->colNum_;
        uint64_t outputOffset = globalRowOffset * this->colNum_;
        uint64_t scaleOffset = globalRowOffset;

        // 重新设置 Global Buffer 地址
        quantInputGM_.SetGlobalBuffer((__gm__ quantInputDataType *)this->quantInputAddr_ + inputOffset,
                                      curRows * this->colNum_);
        quantOutputGM_.SetGlobalBuffer((__gm__ quantOutputDataType *)this->quantOutputAddr_ + outputOffset,
                                       curRows * this->colNum_);
        quantOutputScaleGM_.SetGlobalBuffer((__gm__ float *)this->quantOutputScaleAddr_ + scaleOffset, curRows);
        // smooth的数据类型与input一致
        if (this->hasSmooth_) {
            smoothScaleGM_.SetGlobalBuffer((__gm__ quantInputDataType *)this->smoothScaleAddr_ + scaleOffset, curRows);
        }

        // ALIGN_NUM为8，目的是为了保证bufDatCnt*sizeof(float)与32B对齐
        uint32_t alignedColNum = Ceil(this->colNum_, ALIGN_NUM) * ALIGN_NUM;
        uint32_t bufDataCnt = curRows * alignedColNum;

        // 初始化成员 TBuf
        tPipe_->InitBuffer(floatDataBuf_, bufDataCnt * sizeof(float));
        tPipe_->InitBuffer(workBuf_, bufDataCnt * sizeof(float));
        tPipe_->InitBuffer(tempStatBuf_, bufDataCnt * sizeof(float));
        uint32_t maskCnt = (Ceil(curRows, ALIGN_NUM) * ALIGN_NUM);
        tPipe_->InitBuffer(maskBuf_, maskCnt * sizeof(uint8_t));

        // 初始化关键 Queue
        tPipe_->InitBuffer(rawInputQue_, TWO_FACTOR, bufDataCnt * sizeof(quantInputDataType));
        tPipe_->InitBuffer(quantOutputQue_, TWO_FACTOR, bufDataCnt * sizeof(quantOutputDataType));
        tPipe_->InitBuffer(quantScaleQue_, TWO_FACTOR, Ceil(curRows * sizeof(float), UB_DATABLOCK) * UB_DATABLOCK);

        if (this->hasSmooth_) {
            tPipe_->InitBuffer(smoothScaleQue_, TWO_FACTOR,
                               Ceil(curRows * sizeof(quantInputDataType), UB_DATABLOCK) * UB_DATABLOCK);
        }
    }

    /**
     * @brief pertoken动态量化一个形为（curRows,colNum)的数据块
     *
     * @param curRows 当前处理的总行数
     * @param colNum  一行数据包含的元素个数
     */
    __aicore__ inline void ProcessTile(uint64_t curRows, uint32_t colNum)
    {
        LocalTensor<quantInputDataType> rawInLocal = rawInputQue_.AllocTensor<quantInputDataType>();
        // 保证32对齐
        uint32_t inputAlign = UB_DATABLOCK / sizeof(quantInputDataType);
        uint32_t alignedColNum = Ceil(this->colNum_, inputAlign) * inputAlign;

        // GM->UB
        DataCopyExtParams copyParams = {
            static_cast<uint16_t>(curRows), static_cast<uint32_t>(colNum * sizeof(quantInputDataType)), 0,
            static_cast<uint32_t>(alignedColNum - colNum) * sizeof(quantInputDataType) / UB_DATABLOCK};
        DataCopyPadExtParams<quantInputDataType> padExtParams{false, 0, 0, 0};
        DataCopyPad(rawInLocal, quantInputGM_, copyParams, padExtParams);
        rawInputQue_.EnQue(rawInLocal);

        if (this->hasSmooth_) {
            // GM->UB
            LocalTensor<quantInputDataType> sScale = smoothScaleQue_.AllocTensor<quantInputDataType>();
            DataCopyExtParams sParams{1, static_cast<uint32_t>(curRows * sizeof(quantInputDataType)), 0, 0};
            DataCopyPad(sScale, smoothScaleGM_, sParams, padExtParams);
            smoothScaleQue_.EnQue(sScale);
        }
        ComputeAndMoveOut(curRows, curRows * alignedColNum, colNum, alignedColNum);
    }

    __aicore__ inline void ComputeAndMoveOut(uint64_t curRows, uint32_t padCalCnt, uint32_t colNum,
                                             uint32_t alignedColNum)
    {
        LocalTensor<quantInputDataType> rawIn = rawInputQue_.DeQue<quantInputDataType>();
        LocalTensor<quantOutputDataType> quantOut = quantOutputQue_.AllocTensor<quantOutputDataType>();
        LocalTensor<float> scaleOut = quantScaleQue_.AllocTensor<float>();

        LocalTensor<quantInputDataType> smoothScale;
        if (this->hasSmooth_) {
            smoothScale = smoothScaleQue_.DeQue<quantInputDataType>();
        }

        // 1. 执行量化计算 (UB -> UB)
        DynamicQuantMultiToken(static_cast<uint32_t>(curRows), padCalCnt, rawIn, smoothScale, quantOut, scaleOut);

        // 2. 将结果数据写回 GM (UB -> GM)
        DataCopyExtParams outDataParams = {
            static_cast<uint16_t>(curRows), static_cast<uint32_t>(colNum * sizeof(quantOutputDataType)),
            static_cast<uint32_t>((alignedColNum - colNum) * sizeof(quantOutputDataType) / UB_DATABLOCK), 0};
        DataCopyPad(quantOutputGM_, quantOut, outDataParams);

        // 3. 将 Scale 写回 GM (UB -> GM)
        DataCopyExtParams outScaleParams = {1, static_cast<uint32_t>(curRows * sizeof(float)), 0, 0};
        DataCopyPad(quantOutputScaleGM_, scaleOut, outScaleParams);

        // 4. 释放资源
        rawInputQue_.FreeTensor(rawIn);
        quantOutputQue_.FreeTensor(quantOut);
        quantScaleQue_.FreeTensor(scaleOut);
        if (this->hasSmooth_) {
            smoothScaleQue_.FreeTensor(smoothScale);
        }
    }

    __aicore__ inline void ProcessLargeRows()
    {
        uint32_t maxCols = GetMaxProcCols();
        maxCols = (maxCols / ALIGN_NUM) * ALIGN_NUM;

        tPipe_->Reset();
        InitSegmentResources(maxCols);

        uint32_t smoothScaleDataSize =
            Ceil(this->rowsThisCore_ * sizeof(quantInputDataType), UB_DATABLOCK) * UB_DATABLOCK;
        uint32_t quantScaleDataSize = Ceil(this->rowsThisCore_ * sizeof(float), UB_DATABLOCK) * UB_DATABLOCK;

        tPipe_->InitBuffer(smoothScaleQue_, ONE_FACTOR, smoothScaleDataSize);
        tPipe_->InitBuffer(quantScaleQue_, ONE_FACTOR, quantScaleDataSize);

        LocalTensor<quantInputDataType> coreSmoothScales = smoothScaleQue_.AllocTensor<quantInputDataType>();
        LocalTensor<float> coreQuantScales = quantScaleQue_.AllocTensor<float>();

        if (this->hasSmooth_) {
            DataCopyExtParams sParams{1, static_cast<uint32_t>(this->rowsThisCore_ * sizeof(quantInputDataType)), 0, 0};
            DataCopyPadExtParams<quantInputDataType> padExtParams{false, 0, 0, 0};
            // GM->UB
            DataCopyPad(coreSmoothScales,
                        GlobalTensor<quantInputDataType>((__gm__ quantInputDataType *)this->smoothScaleAddr_ +
                                                         this->startRowThisCore_),
                        sParams, padExtParams);
        }

        for (uint64_t r = 0; r < this->rowsThisCore_; ++r) {
            float s = this->hasSmooth_ ? coreSmoothScales.GetValue(r) : 1.0f;
            float rowMax = ProcessRowMaxAcrossSegments(this->startRowThisCore_ + r, maxCols, s);
            // 除零保护
            float scale = (rowMax > 0.0f) ? (rowMax * this->recipFP8MaxLimit_) : 1.0f;
            float recipScale = (rowMax > 0.0f) ? (this->fp8MaxLimit_ / rowMax) : 1.0f;
            ProcessRowQuantAcrossSegments(this->startRowThisCore_ + r, maxCols, s, recipScale);
            coreQuantScales.SetValue(r, (float)scale);
        }
        // UB ->GM
        DataCopyExtParams outScaleParams = {1, static_cast<uint32_t>(this->rowsThisCore_ * sizeof(float)), 0, 0};
        DataCopyPad(GlobalTensor<float>((__gm__ float *)this->quantOutputScaleAddr_ + this->startRowThisCore_),
                    coreQuantScales, outScaleParams);

        smoothScaleQue_.FreeTensor(coreSmoothScales);
        quantScaleQue_.FreeTensor(coreQuantScales);
    }

    __aicore__ inline float ProcessRowMaxAcrossSegments(uint64_t globalRowIdx, uint32_t maxCols, float smoothScalar)
    {
        float rowMax = 0.0f;
        for (uint32_t cOffset = 0; cOffset < this->colNum_; cOffset += maxCols) {
            uint32_t curCols = Min(maxCols, (uint32_t)this->colNum_ - cOffset);

            LocalTensor<float> floatData = floatDataBuf_.Get<float>();
            LocalTensor<quantInputDataType> rawIn = rawInputQue_.AllocTensor<quantInputDataType>();
            // GM->UB
            DataCopyPad(rawIn,
                        GlobalTensor<quantInputDataType>((__gm__ quantInputDataType *)this->quantInputAddr_ +
                                                         globalRowIdx * this->colNum_ + cOffset),
                        {1, static_cast<uint32_t>(curCols * sizeof(quantInputDataType)), 0, 0}, {false, 0, 0, 0});

            Cast(floatData, rawIn, RoundMode::CAST_NONE, curCols);
            if (this->hasSmooth_)
                Muls(floatData, floatData, smoothScalar, curCols);
            Abs(floatData, floatData, curCols);
            PipeBarrier<PIPE_V>();
            ReduceMaxInplace(floatData, curCols);
            rowMax = Max(rowMax, floatData.GetValue(0));
            rawInputQue_.FreeTensor(rawIn);
        }
        return rowMax;
    }

    __aicore__ inline void ProcessRowQuantAcrossSegments(uint64_t globalRowIdx, uint32_t maxCols, float smoothScalar,
                                                         float recipScale)
    {
        for (uint32_t cOffset = 0; cOffset < this->colNum_; cOffset += maxCols) {
            uint32_t curCols = Min(maxCols, (uint32_t)this->colNum_ - cOffset);

            LocalTensor<float> floatData = floatDataBuf_.Get<float>();
            LocalTensor<quantInputDataType> rawIn = rawInputQue_.AllocTensor<quantInputDataType>();
            LocalTensor<quantOutputDataType> quantOut = quantOutputQue_.AllocTensor<quantOutputDataType>();
            // GM->UB
            DataCopyPad(rawIn,
                        GlobalTensor<quantInputDataType>((__gm__ quantInputDataType *)this->quantInputAddr_ +
                                                         globalRowIdx * this->colNum_ + cOffset),
                        {1, (uint32_t)(curCols * sizeof(quantInputDataType)), 0, 0}, {false, 0, 0, 0});

            Cast(floatData, rawIn, RoundMode::CAST_NONE, curCols);
            if (this->hasSmooth_)
                Muls(floatData, floatData, smoothScalar, curCols);
            Muls(floatData, floatData, recipScale, curCols);
            Cast(quantOut, floatData, RoundMode::CAST_RINT, curCols);
            // UB->GM
            DataCopyPad(GlobalTensor<quantOutputDataType>((__gm__ quantOutputDataType *)this->quantOutputAddr_ +
                                                          globalRowIdx * this->colNum_ + cOffset),
                        quantOut, {1, (uint32_t)(curCols * sizeof(quantOutputDataType)), 0, 0});

            rawInputQue_.FreeTensor(rawIn);
            quantOutputQue_.FreeTensor(quantOut);
        }
    }

    __aicore__ inline void InitSegmentResources(uint32_t curCols)
    {
        uint32_t alignedCols = Ceil(curCols, ALIGN_NUM) * ALIGN_NUM;
        uint32_t alignedBytes = alignedCols * sizeof(float);

        tPipe_->InitBuffer(floatDataBuf_, alignedBytes);
        tPipe_->InitBuffer(workBuf_, alignedBytes);
        tPipe_->InitBuffer(tempStatBuf_, alignedBytes);

        tPipe_->InitBuffer(rawInputQue_, ONE_FACTOR, alignedCols * sizeof(quantInputDataType));
        tPipe_->InitBuffer(quantOutputQue_, ONE_FACTOR, alignedCols * sizeof(quantOutputDataType));
    }
};
} // namespace MC2KernelTemplate
#endif
