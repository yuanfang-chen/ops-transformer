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
 * \file fia_block_vec_nonquant.h
 * \brief
 */
#ifndef COMPRESSOR_BLOCK_VEC_H
#define FIA_BLOCK_VEC_NONQUANT_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "compressor_comm.h"

using namespace Compressor;
using AscendC::CrossCoreSetFlag;
using AscendC::CrossCoreWaitFlag;

template <typename COMP> class CompressorBlockVector{
public:
    // =================================类型定义区=================================
    // 中间计算数据类型为float，高精度模式
    using T = float;
    static constexpr uint64_t BLOCK_VEC_BASE_BUFFER_SIZE = 32 * 1024; // 32k

    __aicore__ inline CompressorBlockVector(){};
    // =================================设置参数=================================
    __aicore__ inline void InitParams(const ConstInfo &constInfo);
    __aicore__ inline void Init( 
        __gm__ uint8_t *x,
        __gm__ uint8_t *wKv,
        __gm__ uint8_t *wGate,
        __gm__ uint8_t *kvState,
        __gm__ uint8_t *scoreState,
        __gm__ uint8_t *ape,
        __gm__ uint8_t *normWeight,
        __gm__ uint8_t *ropeSin,
        __gm__ uint8_t *ropeCos,
        __gm__ uint8_t *blockTable,
        __gm__ uint8_t *cuSeqlens,
        __gm__ uint8_t *seqUsed,
        __gm__ uint8_t *startPos,
        __gm__ uint8_t *cmpKvOut,
        __gm__ uint8_t *kvStateOut,
        __gm__ uint8_t *scoreStateOut);
    // =================================资源管理=================================
    __aicore__ inline void InitBuffers(TPipe *pipe);
    // =================================执行计算=================================
    __aicore__ inline void ComputeVec1(const Compressor::RunInfo &info);
    __aicore__ inline uint32_t GetBasicNum();
    __aicore__ inline void SetMSplitInfo(const Compressor::RunInfo &info);
    __aicore__ inline void InitVec1GlobalTensor(GlobalTensor<T> preMm1ResGm, GlobalTensor<T> curMm1ResGm, GlobalTensor<T> vec1ResGm);
    __aicore__ inline void ComputeVec2(const Compressor::RunInfo &info);
    __aicore__ inline void WriteToCacheState(GlobalTensor<T> &state, LocalTensor<T> &input, uint32_t batchIdx, uint32_t startSeqIdx, uint32_t endSeqIdx, uint32_t dStart, uint32_t dEnd);
    __aicore__ inline void ReadFromCacheState(LocalTensor<T> &output, GlobalTensor<T> &state, uint32_t batchIdx, uint64_t startSeqIdx, uint64_t endSeqIdx, uint32_t dStart, uint32_t dEnd);
    __aicore__ inline void ProcessSingleBatch(uint32_t batchIdx);

protected:
    GlobalTensor<T> vec1ResGm_;
    GlobalTensor<T> preMm1ResGm_;
    GlobalTensor<T> curMm1ResGm_;
    TBuf<TPosition::VECIN> mm1ResUb;
    LocalTensor<T> mm1ResTensor;
    TBuf<TPosition::VECCALC> shareBuffer_;

private:
    __aicore__ inline uint32_t GetStartPos(uint32_t bIdx);
    __aicore__ inline uint32_t GetSeqLength(uint32_t bStart, uint32_t bIdx);
    uint32_t cmpRatio_ = 0U;
    uint32_t coff_ = 0U;
    uint32_t curStartPos_ = 0;
    uint32_t preStartPosIdx_ = 0;
    uint32_t accSeqLength_ = 0;
    uint32_t curActSeqLength_ = 0;
    uint32_t preActSeqIdx_ = 0;
    ConstInfo constInfo_ = {};
    MSplitInfo mSplitInfo = {};
    GlobalTensor<int32_t> startPosGm_;
    GlobalTensor<int32_t> cuSeqlensGm_;
    GlobalTensor<int32_t> blockTableGm_;
};

template <typename COMP>
__aicore__ inline void CompressorBlockVector<COMP>::InitParams(const ConstInfo &constInfo)
{
    this->constInfo_ = constInfo;
}

template <typename COMP> 
__aicore__ inline void CompressorBlockVector<COMP>::Init(
        __gm__ uint8_t *x,
        __gm__ uint8_t *wKv,
        __gm__ uint8_t *wGate,
        __gm__ uint8_t *kvState,
        __gm__ uint8_t *scoreState,
        __gm__ uint8_t *ape,
        __gm__ uint8_t *normWeight,
        __gm__ uint8_t *ropeSin,
        __gm__ uint8_t *ropeCos,
        __gm__ uint8_t *blockTable,
        __gm__ uint8_t *cuSeqlens,
        __gm__ uint8_t *seqUsed,
        __gm__ uint8_t *startPos,
        __gm__ uint8_t *cmpKvOut,
        __gm__ uint8_t *kvStateOut,
        __gm__ uint8_t *scoreStateOut)
{
    startPosGm_.SetGlobalBuffer((__gm__ int32_t *)startPos);
    cuSeqlensGm_.SetGlobalBuffer((__gm__ int32_t *)cuSeqlens);
    blockTableGm_.SetGlobalBuffer((__gm__ int32_t *)blockTable);

    // TODO 封装成类来使用GetStartPos、GetSeqLength
    // 初始化curStartPos_和accSeqLength_
    if (COMP::xLayout == X_LAYOUT::TH) {
        curActSeqLength_ = cuSeqlensGm_.GetValue(1) - cuSeqlensGm_.GetValue(0);
        accSeqLength_ = curActSeqLength_;
        // printf("[Init] curActSeqLength:%u\n", curActSeqLength);
    }
    curStartPos_ = startPosGm_.GetValue(0);
}

template <typename COMP> 
__aicore__ inline void CompressorBlockVector<COMP>::InitBuffers(TPipe *pipe)
{
    // UB
    pipe->InitBuffer(mm1ResUb, 128 * 1024);
    mm1ResTensor = mm1ResUb.Get<T>();
    pipe->InitBuffer(shareBuffer_, BLOCK_VEC_BASE_BUFFER_SIZE);
}

template <typename COMP> 
__aicore__ inline void CompressorBlockVector<COMP>::InitVec1GlobalTensor(GlobalTensor<T> preMm1ResGm, GlobalTensor<T> curMm1ResGm, GlobalTensor<T> vec1ResGm) {
    this->preMm1ResGm_ = preMm1ResGm;
    this->curMm1ResGm_ = curMm1ResGm;
    this->vec1ResGm_ = vec1ResGm;
}

template <typename COMP>
__aicore__ inline uint32_t CompressorBlockVector<COMP>::GetStartPos(uint32_t bIdx)
{
    if (preStartPosIdx_ != bIdx) {
        curStartPos_ = startPosGm_.GetValue(bIdx);
        preStartPosIdx_ = bIdx;
        return curStartPos_;
    } else {
        return curStartPos_;
    }
}

// TODO 在kernel侧是两次顺序访问，这里如果不能顺序访问则不能用这种方式
template <typename COMP>
__aicore__ inline uint32_t CompressorBlockVector<COMP>::GetSeqLength(uint32_t bStart, uint32_t bIdx)
{
    if (COMP::xLayout == X_LAYOUT::TH) {
        if (preActSeqIdx_ != bIdx) {
            preActSeqIdx_ = bIdx;
            if (bIdx == bStart) {
                accSeqLength_ = cuSeqlensGm_.GetValue(bIdx + 1) - cuSeqlensGm_.GetValue(bIdx);
                return accSeqLength_;
            } else {
                uint32_t tmpSeqLength = accSeqLength_;
                accSeqLength_ = cuSeqlensGm_.GetValue(bIdx + 1);
                return accSeqLength_ - tmpSeqLength;
            }
        } else {
            return curActSeqLength_;
        }
    } else {
        return constInfo_.sSize;
    }
}

template <typename COMP>
__aicore__ inline void CompressorBlockVector<COMP>::WriteToCacheState(GlobalTensor<T> &state, LocalTensor<T> &input, uint32_t batchIdx, uint32_t startSeqIdx, uint32_t endSeqIdx, uint32_t dStart, uint32_t dEnd)
{
    uint64_t blockTablebaseOffset = batchIdx * constInfo_.maxBlockNumPerBatch;
    uint32_t curSeqIdx = 0;
    uint32_t copyFinishRowCnt = 0;
    uint32_t seqCnt = endSeqIdx - startSeqIdx;
    while (copyFinishRowCnt < seqCnt) {
        curSeqIdx = startSeqIdx + copyFinishRowCnt;
        uint64_t blockIdOffset = curSeqIdx / constInfo_.blockSize;
        uint64_t remainRowCnt = curSeqIdx % constInfo_.blockSize;
        uint64_t idInBlockTable = blockTableGm_.GetValue(blockTablebaseOffset + blockIdOffset);
        uint32_t copyRowCnt = constInfo_.blockSize - remainRowCnt;
        if (copyFinishRowCnt + copyRowCnt > seqCnt) {
            copyRowCnt = seqCnt - copyFinishRowCnt;
        }
        uint64_t stateOffset = idInBlockTable * constInfo_.blockSize * (dEnd - dStart) + remainRowCnt * (dEnd - dStart) + dStart;
        if (idInBlockTable == 0) {
            DataCopy(state[stateOffset], input, copyRowCnt);
        }
        copyFinishRowCnt += copyRowCnt;
        curSeqIdx += copyRowCnt;
    }
}

template <typename COMP>
__aicore__ inline void CompressorBlockVector<COMP>::ReadFromCacheState(LocalTensor<T> &output, GlobalTensor<T> &state, uint32_t batchIdx, uint64_t startSeqIdx, uint64_t endSeqIdx, uint32_t dStart, uint32_t dEnd)
{
    uint64_t blockTablebaseOffset = batchIdx * constInfo_.maxBlockNumPerBatch;
    uint32_t curSeqIdx = 0;
    uint32_t copyFinishRowCnt = 0;
    uint32_t seqCnt = endSeqIdx - startSeqIdx;
    while (copyFinishRowCnt < seqCnt) {
        curSeqIdx = startSeqIdx + copyFinishRowCnt;
        uint64_t blockIdOffset = curSeqIdx / constInfo_.blockSize;
        uint64_t remainRowCnt = curSeqIdx % constInfo_.blockSize;
        uint64_t idInBlockTable = blockTableGm_.GetValue(blockTablebaseOffset + blockIdOffset);
        uint32_t copyRowCnt = constInfo_.blockSize - remainRowCnt;
        if (copyFinishRowCnt + copyRowCnt > seqCnt) {
            copyRowCnt = seqCnt - copyFinishRowCnt;
        }
        if (idInBlockTable == 0) {
            // error log
        }
        uint64_t stateOffset = idInBlockTable * constInfo_.blockSize * (dEnd - dStart) + remainRowCnt * (dEnd - dStart) + dStart;
        DataCopy(output, state[stateOffset], copyRowCnt);
        copyFinishRowCnt += copyRowCnt;
        curSeqIdx += copyRowCnt;
    }
}

template <typename COMP> 
__aicore__ inline void CompressorBlockVector<COMP>::ProcessSingleBatch(uint32_t batchIdx)
{
    uint32_t seqIdx = 0;
    uint64_t seqUsed = GetSeqLength(0, batchIdx);
    uint64_t startPos = GetStartPos(batchIdx);
    uint64_t startSeqIdx = 0;
    uint64_t endSeqIdx = 0;
    while (seqIdx < seqUsed) {
        startSeqIdx = startPos + seqIdx;
        endSeqIdx = startSeqIdx / constInfo_.cmpRatio * constInfo_.cmpRatio + constInfo_.cmpRatio;
        if (endSeqIdx > (startPos + seqUsed)) {
            endSeqIdx = startPos + seqUsed;
        }
        uint64_t baseOffset = batchIdx * constInfo_.sSize;
        uint64_t startOffset = baseOffset + (startSeqIdx - startPos);
        uint64_t endOffset = baseOffset + (endSeqIdx - startPos);
    }
}

template <typename COMP>
__aicore__ inline uint32_t CompressorBlockVector<COMP>::GetBasicNum() {
    // 获取 m方向上对应基本单元Tc的个数
    uint32_t curBasicNum = 0;
    uint32_t headSize = 0;
    if (curStartPos_ % constInfo_.cmpRatio != 0) {
        headSize = constInfo_.cmpRatio - curStartPos_ % constInfo_.cmpRatio;
        headSize = headSize > curActSeqLength_ ? curActSeqLength_ : headSize;
        curBasicNum++;
    }
    // 加上中间整块及尾块
    curBasicNum += (curActSeqLength_ - headSize + constInfo_.cmpRatio - 1) / constInfo_.cmpRatio;
    return curBasicNum;
}

template <typename COMP>
 __aicore__ inline void CompressorBlockVector<COMP>::SetMSplitInfo(const Compressor::RunInfo &info)
 {
    // TODO 处理0块需要考虑？
    // VEC0需要处理的大小
    mSplitInfo.dealTcNum = (info.dealTcNum + 1) / 2;
    mSplitInfo.vecStartB = info.bStart;
    mSplitInfo.vecStartS = info.sStart;
    uint32_t curBEnd = 0;
    uint32_t curSEnd = 0;
    
    // VEC1处理的大小
    uint32_t accBasicNum = 0;
    for (int bIdx = mSplitInfo.vecStartB; bIdx < constInfo_.batchSize; ++bIdx) {
        curBEnd = bIdx;
        // 计算起始batch的剩余块
        if (bIdx == mSplitInfo.vecStartB) {
            curActSeqLength_ = GetSeqLength(mSplitInfo.vecStartB, bIdx);
            curStartPos_ = GetStartPos(bIdx);
            uint32_t curRemainTcNum = 0;
            // 计算起始batch的剩余seq长度 起始位置计算头块
            uint32_t headSize = 0;
            if (curStartPos_ % constInfo_.cmpRatio != 0) {
                headSize = (constInfo_.cmpRatio - curStartPos_ % constInfo_.cmpRatio);
                headSize = headSize > curActSeqLength_ ? curActSeqLength_ : headSize;
            }
            if (mSplitInfo.vecStartS == 0) {
                curRemainTcNum = (curActSeqLength_ - headSize + constInfo_.cmpRatio - 1) / constInfo_.cmpRatio;
                curRemainTcNum = headSize == 0 ? curRemainTcNum : curRemainTcNum + 1;
            } else {
                curRemainTcNum = (curActSeqLength_ - mSplitInfo.vecStartS + constInfo_.cmpRatio - 1) / constInfo_.cmpRatio;
            }
            // printf("[GetEndIdx]  bIdx:%u accBasicNum:%u mSplitInfo.dealTcNum:%u curRemainTcNum:%u headSize:%u curStartPos_:%u curActSeqLength_:%u \n", bIdx, accBasicNum, mSplitInfo.dealTcNum, curRemainTcNum, headSize, curStartPos_, curActSeqLength_);
            if (curRemainTcNum > mSplitInfo.dealTcNum) {
                if (mSplitInfo.vecStartS == 0) {
                    if (headSize == 0) {
                        curSEnd = mSplitInfo.vecStartS + mSplitInfo.dealTcNum * constInfo_.cmpRatio;
                    } else {
                        curSEnd = mSplitInfo.vecStartS + headSize + (mSplitInfo.dealTcNum - 1) * constInfo_.cmpRatio;
                    }
                    break;
                } else {
                    curSEnd = mSplitInfo.vecStartS + mSplitInfo.dealTcNum * constInfo_.cmpRatio;
                    break;
                }
            } else if (curRemainTcNum == mSplitInfo.dealTcNum || bIdx == constInfo_.batchSize - 1) {
                curSEnd = curActSeqLength_;
                break;
            } else {
                accBasicNum += curRemainTcNum;
            }
        } else {
            curActSeqLength_ = GetSeqLength(mSplitInfo.vecStartB, bIdx);
            curStartPos_ = GetStartPos(bIdx);
            uint32_t curBasicNum = GetBasicNum();
            // printf("[GetEndIdx] accBasicNum:%u curBasicNum:%u dealTcNum:%u\n", accBasicNum, curBasicNum, dealTcNum);
            if (accBasicNum + curBasicNum > mSplitInfo.dealTcNum) {
                uint32_t headSize = 0;
                if (curStartPos_ % constInfo_.cmpRatio != 0) {
                    headSize = constInfo_.cmpRatio - curStartPos_ % constInfo_.cmpRatio;
                    // 处理seq不足head大小的情况
                    headSize = headSize > curActSeqLength_ ? curActSeqLength_ : headSize;
                }
                // 在当前batch结束tc块索引
                uint32_t curBasicNumEnd = mSplitInfo.dealTcNum - accBasicNum;
                if (headSize == 0) {
                    curSEnd = curBasicNumEnd * constInfo_.cmpRatio;
                } else {
                    curSEnd = headSize + (curBasicNumEnd - 1) * constInfo_.cmpRatio;
                }
                curSEnd = curSEnd > curActSeqLength_ ? curActSeqLength_ : curSEnd;
                return;
            } else if (accBasicNum + curBasicNum == mSplitInfo.dealTcNum) {
                curSEnd = curActSeqLength_;
                return;
            }
            accBasicNum += curBasicNum;
        }
    }
    mSplitInfo.vecEndS = curSEnd;
    mSplitInfo.vecEndB = curBEnd;
    if (GetBlockIdx() % 2 == 1) {
        mSplitInfo.vecStartB = curBEnd;
        mSplitInfo.vecStartS = curSEnd;
        mSplitInfo.dealTcNum = info.dealTcNum - mSplitInfo.dealTcNum;
        if (curSEnd == curActSeqLength_ && mSplitInfo.dealTcNum > 0) {
             mSplitInfo.vecStartB++;
             mSplitInfo.vecStartS = 0;
        }
        mSplitInfo.vecEndB = info.bEnd;
        mSplitInfo.vecEndS = info.sEnd;
    }
 }

template <typename COMP>
 __aicore__ inline void CompressorBlockVector<COMP>::ComputeVec1(const Compressor::RunInfo &info)
{
    // DumpTensor(mm1ResTensor, 1, 128 * 256);
    // TODO 1分核
    SetMSplitInfo(info);
    uint32_t scLoopTimes = 0;
    uint32_t dLoopTimes = 0;
    uint32_t splitSize = BLOCK_VEC_BASE_BUFFER_SIZE / (constInfo_.cmpRatio * static_cast<uint32_t>(COMP::coff) * sizeof(T));
    if (splitSize < constInfo_.headDim) {
        scLoopTimes = 1;
        dLoopTimes = (constInfo_.headDim + (splitSize - 1)) / splitSize;
    } else {
        dLoopTimes = 1;
        scLoopTimes = splitSize / constInfo_.headDim;
    }
    for (uint32_t i = 0; i < scLoopTimes; i++) {
        for (uint32_t j = 0; j < dLoopTimes; j++) {
            for (uint32_t k = info.bStart; k <= info.bEnd; k++) {
                // 计算当前batch的seq 开始结束索引
                curActSeqLength_ = GetSeqLength(info.bStart, k);
                uint32_t sStart = 0;
                uint32_t sEnd = curActSeqLength_;
                if (k == info.bStart) {
                    sStart = info.sStart;
                }
                if (k == info.bEnd) {
                    sEnd = info.sEnd;
                }
                printf("[IDX] b:%u sStart:%u sEnd:%u\n", k, sStart, sEnd);
                // 从UB拷贝到32k空间
                // 存state
                // 从state取
                // overlap
                
            }
        }
    }

}

template <typename COMP> 
__aicore__ inline void CompressorBlockVector<COMP>::ComputeVec2(const Compressor::RunInfo &info)
{
    
}

#endif // COMPRESSOR_BLOCK_VECTOR_H