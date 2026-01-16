/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file compressor_block_vec.h
 * \brief
 */

#ifndef COMPRESSOR_BLOCK_VEC_H
#define COMPRESSOR_BLOCK_VEC_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "../compressor_comm.h"

using namespace AscendC;

namespace Compressor 
{
using AscendC::CrossCoreSetFlag;
using AscendC::CrossCoreWaitFlag;

static constexpr uint64_t BLOCK_VEC_BASE_BUFFER_SIZE = 32 * 1024; // 32k
template <typename COMP> class CompressorBlockVector {
public:
    static constexpr bool X_DTYPE = COMP::xDtype == X_DTYPE::BF16;
    // =================================类型定义区=================================
    // 中间计算数据类型为float，高精度模式
    using T = float;
    using X_T = typename AscendC::Conditional<X_DTYPE, bfloat16_t, half>::type;

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
    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();
    // =================================执行计算=================================
    __aicore__ inline void ComputeVec1(const Compressor::RunInfo &info);
    __aicore__ inline uint32_t GetBasicNum();
    __aicore__ inline uint32_t GetScSize();
    __aicore__ inline void GetScIdxInfo(uint32_t bStart, uint32_t scStart, uint32_t dealScSize, uint32_t v2TcStart, uint32_t v2TcEnd, 
                                                            uint32_t &outputBStart, uint32_t &outputSStart, uint32_t &outputScSize);
    __aicore__ inline void CalcTcEndIdx(uint32_t bStart, uint32_t sStart, uint32_t dealTcNum, uint32_t &bEnd, uint32_t &sEnd);
    __aicore__ inline void CalcScEndIdx(uint32_t bStart, uint32_t scStart, uint32_t dealScSize, uint32_t &bEnd, uint32_t &scEnd);
    __aicore__ inline void SetMSplitInfo(const Compressor::RunInfo &info);
    __aicore__ inline void InitVec1GlobalTensor(GlobalTensor<T> preMm1ResGm, GlobalTensor<T> curMm1ResGm, GlobalTensor<T> vec1ResGm);
    __aicore__ inline void ComputeVec2(const Compressor::RunInfo &info);
    __aicore__ inline void WriteToCacheState(GlobalTensor<T> &state, LocalTensor<T> &input, uint32_t batchIdx, uint32_t startSeqIdx, uint32_t endSeqIdx, uint32_t dStart, uint32_t dEnd);
    __aicore__ inline void ReadFromCacheState(LocalTensor<T> &output, GlobalTensor<T> &state, uint32_t batchIdx, uint64_t startSeqIdx, uint64_t endSeqIdx, uint32_t dStart, uint32_t dEnd);
    __aicore__ inline void ProcessSingleBatch(uint32_t batchIdx);
    __aicore__ inline void ProcessSingleBatch(uint32_t batchIdx, uint64_t batchStartSeqIdx, uint64_t batchEndSeqIdx, uint32_t dLoop, uint32_t dealDSize,
        LocalTensor<T> &mmResRight, LocalTensor<T> &mmResLeft);

protected:
    GlobalTensor<T> vec1ResGm_;
    GlobalTensor<T> preMm1ResGm_;
    GlobalTensor<T> curMm1ResGm_;
    TBuf<TPosition::VECIN> mm1ResUb;
    LocalTensor<T> mm1ResTensor;
    TBuf<TPosition::VECCALC> kvBuff_;
    TBuf<TPosition::VECCALC> scoreBuff_;
    TBuf<> inputBuf1; // 32K * 2

private:
    __aicore__ inline uint32_t GetStartPos(uint32_t bIdx);
    __aicore__ inline uint32_t GetSeqLength(uint32_t bIdx);
    __aicore__ inline uint32_t GetBsLength(uint32_t index);
    __aicore__ inline void SplitCoreV2(const Compressor::RunInfo& info);
    __aicore__ inline void CopyFinalResultOut(const Compressor::RunInfo& info, LocalTensor<X_T> &ComOutUb);
    __aicore__ inline void DealVec2BaseBlock(const Compressor::RunInfo& info, uint32_t startRow, uint32_t dealRowCount);
    __aicore__ inline void RmsNorm(const Compressor::RunInfo& info, uint32_t startRow, uint32_t dealRowCount);
    __aicore__ inline void CalRope(const Compressor::RunInfo& info);
    static constexpr uint64_t SYNC_INPUT_BUF1_FLAG = 2;
    static constexpr uint64_t SYNC_INPUT_BUF1_PONG_FLAG = 3;
    uint32_t cmpRatio_ = 0U;
    uint32_t coff_ = 0U;
    uint32_t curStartPos_ = 0;
    uint32_t curActSeqLength_ = 0;
    // vec2
    uint32_t v2MBaseSize = 16; // Tc块数量：32 * 1024 / (512 * 4)
    uint32_t v2TcStartIdx = 0U;
    uint32_t v2TcEndIdx = 0U;
    uint32_t mmResColSize_ = 128;
    int64_t vec1ResGmStart = 0U;
    uint32_t usedCoreNum = 16;
    uint32_t OutputBStartIdx, OutputSStartIdx, OutputSize;
    uint32_t pingpongFlag = 0U;
    uint32_t mmResBaseOffset_ = 0;
    ConstInfo constInfo_ = {};
    MSplitInfo mSplitInfo = {};
    GlobalTensor<int32_t> startPosGm_;
    GlobalTensor<int32_t> cuSeqlensGm_;
    GlobalTensor<int32_t> blockTableGm_;
    GlobalTensor<T> kvStateGm_;
    GlobalTensor<T> scoreStateGm_;
    GlobalTensor<T> apeGm_;
};

template <typename COMP>
__aicore__ inline void CompressorBlockVector<COMP>::InitParams(const ConstInfo &constInfo)
{
    this->constInfo_ = constInfo;
    v2MBaseSize = BLOCK_VEC_BASE_BUFFER_SIZE / (constInfo_.headDim * sizeof(float));
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
    kvStateGm_.SetGlobalBuffer((__gm__ T *)kvStateOut);
    scoreStateGm_.SetGlobalBuffer((__gm__ T *)scoreStateOut);
    apeGm_.SetGlobalBuffer((__gm__ T *)ape);
}

template <typename COMP> 
__aicore__ inline void CompressorBlockVector<COMP>::InitBuffers(TPipe *pipe)
{
    // UB
    pipe->InitBuffer(mm1ResUb, 128 * 1024);
    mm1ResTensor = mm1ResUb.Get<T>();
    pipe->InitBuffer(kvBuff_, BLOCK_VEC_BASE_BUFFER_SIZE);
    pipe->InitBuffer(scoreBuff_, BLOCK_VEC_BASE_BUFFER_SIZE);
}

template <typename COMP> 
__aicore__ inline void CompressorBlockVector<COMP>::AllocEventID()
{
    SetFlag<HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG);
    SetFlag<HardEvent::V_MTE2>(SYNC_INPUT_BUF1_PONG_FLAG);
}

template <typename COMP> 
__aicore__ inline void CompressorBlockVector<COMP>::FreeEventID()
{
    WaitFlag<HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG);
    WaitFlag<HardEvent::V_MTE2>(SYNC_INPUT_BUF1_PONG_FLAG);
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
    return startPosGm_.GetValue(bIdx);
}

// TODO 使用这种方式获取seq的约束为顺序访问，随机访问不可用
template <typename COMP>
__aicore__ inline uint32_t CompressorBlockVector<COMP>::GetSeqLength(uint32_t bIdx)
{
    if (COMP::xLayout == X_LAYOUT::TH) {
        return cuSeqlensGm_.GetValue(bIdx + 1) - cuSeqlensGm_.GetValue(bIdx);
    } else {
        return constInfo_.sSize;
    }
}

template <typename COMP>
__aicore__ inline uint32_t CompressorBlockVector<COMP>::GetBsLength(uint32_t index)
{
    if (COMP::xLayout == X_LAYOUT::TH) {
        return cuSeqlensGm_.GetValue(index);
    } else {
        return index * constInfo_.sSize;
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
            // 拷贝参数还需要调整
            DataCopyParams copyParams {static_cast<uint16_t>(copyRowCnt), static_cast<uint16_t>(dEnd - dStart), constInfo_.dBaseSize, static_cast<uint16_t>(2 * (dEnd - dStart))};
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
            // print error log
        }
        uint64_t stateOffset = idInBlockTable * constInfo_.blockSize * (dEnd - dStart) + remainRowCnt * (dEnd - dStart) + dStart;
        // 拷贝参数还需要调整
        DataCopyParams copyParams {static_cast<uint16_t>(copyRowCnt), static_cast<uint16_t>(dEnd - dStart),
            static_cast<uint16_t>(2 * (dEnd - dStart)), static_cast<uint16_t>(2 * (dEnd - dStart))};
        DataCopy(output, state[stateOffset], copyRowCnt);
        copyFinishRowCnt += copyRowCnt;
        curSeqIdx += copyRowCnt;
    }
}

template <typename COMP> __aicore__ inline void CompressorBlockVector<COMP>::ProcessSingleBatch(uint32_t batchIdx, uint64_t batchStartSeqIdx, uint64_t batchEndSeqIdx,
    uint32_t dLoop, uint32_t dealDSize, LocalTensor<T> &mmResRight, LocalTensor<T> &mmResLeft)
{
    uint32_t seqIdx = batchStartSeqIdx;
    uint64_t seqUsed = GetSeqLength(batchIdx); // seqused有效的话还需要调整
    uint64_t startPos = GetStartPos(batchIdx);
    uint64_t startSeqIdx = 0;
    uint64_t endSeqIdx = 0;
    uint32_t dStartIdx = 0;
    uint32_t dEndIdx = 0;
    uint64_t baseOffset = GetBsLength(batchIdx);
    uint64_t compressSeqId = (startPos + seqUsed) / constInfo_.cmpRatio * constInfo_.cmpRatio;
    uint32_t copyUbBaseOffset = 0;
    LocalTensor<T> kvLocal = kvBuff_.Get<T>();
    LocalTensor<T> scoreLocal = scoreBuff_.Get<T>();
    while (seqIdx < batchEndSeqIdx) {
        // 首次处理半个r，中间处理完整r，结尾半个r
        startSeqIdx = startPos + seqIdx;
        endSeqIdx = startSeqIdx / constInfo_.cmpRatio * constInfo_.cmpRatio + constInfo_.cmpRatio;
        // seqIdx和batchStartSeqIdx为局部索引
        uint64_t startOffset = mmResBaseOffset_ + (seqIdx - batchStartSeqIdx) * constInfo_.dBaseSize; //(baseOffset + (startSeqIdx - startPos)) * constInfo_.dBaseSize;
        // uint64_t endOffset = baseOffset + (endSeqIdx - startPos);
        uint64_t endOffset = baseOffset + (endSeqIdx - startPos);
        bool isSaveState = false;
        bool isCompress = false;
        bool isCopy = true;
        if (endSeqIdx > (startPos + seqUsed) && (batchEndSeqIdx == seqUsed)) {
            // 1.说明当前batch尾部在处理的基本块中，且结尾非完整r块，本次不需要压缩，不需要拷贝到UB,其他场景都需要拷贝
            endSeqIdx = startPos + seqUsed;
            isCopy = false;
        }
        if (startPos == startSeqIdx && (endSeqIdx - startSeqIdx) < constInfo_.cmpRatio) {
            // batch的头小于r,用来和上个的尾结合为一个r，需要将上面补充的数据拷贝过来
            DataCopyParams copyParams {(startPos % constInfo_.cmpRatio), static_cast<uint16_t>(dealDSize), mmResColSize_, static_cast<uint16_t>(2 * dealDSize)};
            DataCopy(kvLocal[mmResBaseOffset_ + copyUbBaseOffset + dealDSize], mmResRight[startOffset + dealDSize * dLoop + constInfo_.dBaseSize], copyParams); // 将kv的右侧从mm中拷贝过来，源数据d从0或者32
            DataCopy(scoreLocal[mmResBaseOffset_ + copyUbBaseOffset + dealDSize], mmResRight[startOffset + dealDSize * dLoop + constInfo_.dBaseSize], copyParams); // 将score的右侧从mm中拷贝过来,源数据d从64或者96
            DataCopy(kvLocal[mmResBaseOffset_ + copyUbBaseOffset], mmResLeft[startOffset + dealDSize * dLoop], copyParams); // 将kv的左侧从mm中拷贝过来,源数据d从0或者32
            DataCopy(scoreLocal[mmResBaseOffset_ + copyUbBaseOffset], mmResLeft[startOffset + dealDSize * dLoop ], copyParams); // 将score的左侧从mm中拷贝过来,源数据d从64或者96
            copyUbBaseOffset += (startPos % constInfo_.cmpRatio) * dealDSize;        }
        if (isCopy) {
            // 除尾块之外的用于本次压缩的存到UB中, 以右侧索引为准，左侧拷贝同样的seqidx
            DataCopyParams copyParams {static_cast<uint16_t>(endSeqIdx - startSeqIdx), static_cast<uint16_t>(dealDSize), mmResColSize_, static_cast<uint16_t>(2 * dealDSize)};
            DataCopy(kvLocal[mmResBaseOffset_ + copyUbBaseOffset], mmResRight[startOffset + dealDSize * dLoop + constInfo_.dBaseSize], copyParams); // 将kv的右侧从mm中拷贝过来，源数据d从0或者32
            DataCopy(scoreLocal[mmResBaseOffset_ + copyUbBaseOffset], mmResRight[startOffset + dealDSize * dLoop + constInfo_.dBaseSize], copyParams); // 将score的右侧从mm中拷贝过来,源数据d从64或者96
            DataCopy(kvLocal[mmResBaseOffset_ + copyUbBaseOffset + dealDSize], mmResLeft[startOffset + dealDSize * dLoop], copyParams); // 将kv的左侧从mm中拷贝过来,源数据d从0或者32
            DataCopy(scoreLocal[mmResBaseOffset_ + copyUbBaseOffset + dealDSize], mmResLeft[startOffset + dealDSize * dLoop], copyParams); // 将score的左侧从mm中拷贝过来,源数据d从64或者96
            copyUbBaseOffset += (endSeqIdx - startSeqIdx) * dealDSize; 
        }
        dStartIdx = dealDSize * dLoop;
        dEndIdx = dStartIdx + dealDSize;
        if (startSeqIdx >= (compressSeqId - constInfo_.cmpRatio) && endSeqIdx <= (batchEndSeqIdx + startPos) && (batchEndSeqIdx <= seqUsed)) {
            // 第1种场景：尾块在当前处理的基本块中，最后一个尾块和上一个r块需要存入state, 
            isSaveState = true;
        }
        if ((batchEndSeqIdx + startPos) == endSeqIdx && startSeqIdx < (batchEndSeqIdx + startPos) && (seqUsed - batchEndSeqIdx) <  constInfo_.cmpRatio) {
            // 第2种场景：最后一个完整的r在本次处理的基本块中
            isSaveState = true;
        }
        if (batchEndSeqIdx - batchStartSeqIdx == constInfo_.cmpRatio && ((seqUsed - batchEndSeqIdx < constInfo_.cmpRatio) || (seqUsed == batchEndSeqIdx))) {
            // 第3种场景：本次处理的首尾只有一个r
            isSaveState = true;
            endSeqIdx = (batchEndSeqIdx + startPos);
        }
        // todo:第四种场景，最后一个batch的尾块在第一个batch的首位的左边
        if (startSeqIdx < compressSeqId) {
            isCompress = true;
        }
        if (isSaveState) {
            // 左和右都需要考虑
            // 1.存右边
            WriteToCacheState(kvStateGm_, mmResRight[startOffset + dealDSize * dLoop], batchIdx, startSeqIdx, endSeqIdx, dStartIdx, dEndIdx);
            WriteToCacheState(scoreStateGm_, mmResRight[startOffset + dealDSize * dLoop + constInfo_.dBaseSize], batchIdx, startSeqIdx, endSeqIdx, dStartIdx, dEndIdx);
            // 2.存左边
            WriteToCacheState(kvStateGm_, mmResLeft[startOffset + dealDSize * dLoop - constInfo_.cmpRatio * constInfo_.dBaseSize], batchIdx, startSeqIdx, endSeqIdx, dStartIdx, dEndIdx);
            WriteToCacheState(scoreStateGm_, mmResLeft[startOffset + dealDSize * dLoop + constInfo_.dBaseSize], batchIdx, startSeqIdx, endSeqIdx, dStartIdx, dEndIdx);
            // 2. 右边存最后一个尾块
            if ((batchStartSeqIdx + startPos) == startPos) {
                // 第4种情况，左上角有个r块需要存
                WriteToCacheState(kvStateGm_, mmResLeft[startOffset], batchIdx, startSeqIdx, endSeqIdx, dStartIdx, dEndIdx);
                WriteToCacheState(scoreStateGm_, mmResLeft[startOffset + constInfo_.dBaseSize], batchIdx, startSeqIdx, endSeqIdx, dStartIdx, dEndIdx);  
            }
        }
        if (isCompress) {
            uint32_t copyStartSeqId = 0;
            uint32_t copyEndSeqId = 0;
            uint32_t coffId = coff_ - 1;
            uint32_t dStart = coffId * constInfo_.dBaseSize + dLoop * dealDSize;
            uint32_t dEnd = (coffId + 1) * constInfo_.dBaseSize + dLoop * dealDSize;
            uint32_t cntFromState = 0;
            // 拷贝右边数据
            if (startPos == startSeqIdx) {
                // batch从起始位置开始
                cntFromState = startPos % constInfo_.cmpRatio;
                if (cntFromState > 0) {
                    copyStartSeqId = startPos - cntFromState;
                    copyEndSeqId = startPos;
                    // 第一块拷贝
                    ReadFromCacheState(kvLocal[mmResBaseOffset_ + dealDSize], kvStateGm_, batchIdx, copyStartSeqId, copyEndSeqId, dStart, dEnd); // 拷贝的右边
                    ReadFromCacheState(scoreLocal[mmResBaseOffset_ + dealDSize], scoreStateGm_, batchIdx, copyStartSeqId, copyEndSeqId, dStart, dEnd); // 拷贝的右边
                }
            }
            if (coff_ == 2) {  // 拷贝左边数据
                coffId = 0;
                dStart = coffId * constInfo_.dBaseSize  + dLoop * dealDSize ;
                dEnd = (coffId + 1) * constInfo_.dBaseSize  + dLoop * dealDSize;
                cntFromState = 0;
                if (startPos == startSeqIdx) {
                    // 拷贝左边第一块完整块
                    cntFromState = constInfo_.cmpRatio;
                    if (startPos >= constInfo_.cmpRatio) {
                        copyStartSeqId = startPos - startPos % constInfo_.cmpRatio;
                        copyEndSeqId = copyStartSeqId + cntFromState;
                        ReadFromCacheState(kvLocal[mmResBaseOffset_], kvStateGm_, batchIdx, copyStartSeqId, copyEndSeqId, dStart, dEnd);
                        ReadFromCacheState(scoreLocal[mmResBaseOffset_], scoreStateGm_, batchIdx, copyStartSeqId, copyEndSeqId, dStart, dEnd);
                    }
                } else if (startSeqIdx - constInfo_.cmpRatio < startPos) {
                        // 左边数据需要拷贝一个尾块和一个完整r块
                        cntFromState = startPos % constInfo_.cmpRatio;
                        if (cntFromState > 0) {
                            copyStartSeqId = startPos - startPos % constInfo_.cmpRatio;
                            copyEndSeqId = startPos + constInfo_.cmpRatio;
                            ReadFromCacheState(kvLocal[(copyEndSeqId - copyStartSeqId) * dealDSize], kvStateGm_, batchIdx, copyStartSeqId, copyEndSeqId, dStart, dEnd);
                            ReadFromCacheState(scoreLocal[(copyEndSeqId - copyStartSeqId) * dealDSize], scoreStateGm_, batchIdx, copyStartSeqId, copyEndSeqId, dStart, dEnd);
                        }
                    }
            }
        }
        seqIdx = endSeqIdx - startSeqIdx;
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
__aicore__ inline uint32_t CompressorBlockVector<COMP>::GetScSize() {
    // 获取 m方向上对应基本单元Tc的个数
    uint32_t curBasicNum = 0;
    uint32_t headSize = 0;
    if (curStartPos_ % constInfo_.cmpRatio != 0) {
        headSize = constInfo_.cmpRatio - curStartPos_ % constInfo_.cmpRatio;
        headSize = headSize > curActSeqLength_ ? curActSeqLength_ : headSize;
        curBasicNum++;
    }
    // 加上中间整块及尾块
    curBasicNum += (curActSeqLength_ - headSize) / constInfo_.cmpRatio;
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
    uint32_t bEnd = 0;
    uint32_t sEnd = 0;

    CalcTcEndIdx(mSplitInfo.vecStartB, mSplitInfo.vecStartS, mSplitInfo.dealTcNum, bEnd, sEnd);
    mSplitInfo.vecEndB = bEnd;
    mSplitInfo.vecEndS = sEnd;
    if (GetBlockIdx() % 2 == 1) {
        mSplitInfo.vecStartB = bEnd;
        mSplitInfo.vecStartS = sEnd;
        mSplitInfo.dealTcNum = info.dealTcNum - mSplitInfo.dealTcNum;
        if (sEnd == curActSeqLength_ && mSplitInfo.dealTcNum > 0) {
             mSplitInfo.vecStartB++;
             mSplitInfo.vecStartS = 0;
        }
        mSplitInfo.vecEndB = info.bEnd;
        mSplitInfo.vecEndS = info.sEnd;
    }
}

// 根据计算Tc开始结束索引
template <typename COMP>
__aicore__ inline void CompressorBlockVector<COMP>::CalcTcEndIdx(uint32_t bStart, uint32_t sStart, uint32_t dealTcNum, uint32_t &bEnd, uint32_t &sEnd) {
    uint32_t accBasicNum = 0;
    for (int bIdx = bStart; bIdx < constInfo_.batchSize; ++bIdx) {
        bEnd = bIdx;
        // 计算起始batch的剩余块
        if (bIdx == bStart) {
            curActSeqLength_ = GetSeqLength(bIdx);
            curStartPos_ = GetStartPos(bIdx);
            uint32_t curRemainTcNum = 0;
            // 计算起始batch的剩余seq长度 起始位置计算头块
            uint32_t headSize = 0;
            if (curStartPos_ % constInfo_.cmpRatio != 0) {
                headSize = (constInfo_.cmpRatio - curStartPos_ % constInfo_.cmpRatio);
                headSize = headSize > curActSeqLength_ ? curActSeqLength_ : headSize;
            }
            if (sStart == 0) {
                curRemainTcNum = (curActSeqLength_ - headSize + constInfo_.cmpRatio - 1) / constInfo_.cmpRatio;
                curRemainTcNum = headSize == 0 ? curRemainTcNum : curRemainTcNum + 1;
            } else {
                curRemainTcNum = (curActSeqLength_ - sStart + constInfo_.cmpRatio - 1) / constInfo_.cmpRatio;
            }
            // printf("[GetEndIdx]  bIdx:%u accBasicNum:%u dealTcNum:%u curRemainTcNum:%u headSize:%u curStartPos_:%u curActSeqLength_:%u \n", bIdx, accBasicNum, dealTcNum, curRemainTcNum, headSize, curStartPos_, curActSeqLength_);
            if (curRemainTcNum > dealTcNum) {
                if (sStart == 0) {
                    if (headSize == 0) {
                        sEnd = sStart + dealTcNum * constInfo_.cmpRatio;
                    } else {
                        sEnd = sStart + headSize + (dealTcNum - 1) * constInfo_.cmpRatio;
                    }
                    return;
                } else {
                    sEnd = sStart + dealTcNum * constInfo_.cmpRatio;
                    return;
                }
            } else if (curRemainTcNum == dealTcNum || bIdx == constInfo_.batchSize - 1) {
                sEnd = curActSeqLength_;
                return;
            } else {
                accBasicNum += curRemainTcNum;
            }
        } else {
            curActSeqLength_ = GetSeqLength(bIdx);
            curStartPos_ = GetStartPos(bIdx);
            uint32_t curBasicNum = GetBasicNum();
            // printf("[GetEndIdx] accBasicNum:%u curBasicNum:%u dealTcNum:%u\n", accBasicNum, curBasicNum, dealTcNum);
            if (accBasicNum + curBasicNum > dealTcNum) {
                uint32_t headSize = 0;
                if (curStartPos_ % constInfo_.cmpRatio != 0) {
                    headSize = constInfo_.cmpRatio - curStartPos_ % constInfo_.cmpRatio;
                    // 处理seq不足head大小的情况
                    headSize = headSize > curActSeqLength_ ? curActSeqLength_ : headSize;
                }
                uint32_t curBasicNumEnd = dealTcNum - accBasicNum;
                if (headSize == 0) {
                    sEnd = curBasicNumEnd * constInfo_.cmpRatio;
                } else {
                    sEnd = headSize + (curBasicNumEnd - 1) * constInfo_.cmpRatio;
                }
                sEnd = sEnd > curActSeqLength_ ? curActSeqLength_ : sEnd;
                return;
            } else if (accBasicNum + curBasicNum == dealTcNum) {
                sEnd = curActSeqLength_;
                return;
            }
            accBasicNum += curBasicNum;
        }
    }
}

// 根据计算Tc开始结束索引
template <typename COMP>
__aicore__ inline void CompressorBlockVector<COMP>::CalcScEndIdx(uint32_t bStart, uint32_t scStart, uint32_t dealScSize, uint32_t &bEnd, uint32_t &scEnd) {
    uint32_t accScSize = 0;
    for (int bIdx = bStart; bIdx < constInfo_.batchSize; ++bIdx) {
        bEnd = bIdx;
        // 计算起始batch的剩余块
        if (bIdx == bStart) {
            curActSeqLength_ = GetSeqLength(bIdx);
            curStartPos_ = GetStartPos(bIdx);
            accScSize += GetScSize();
            // printf("[GetEndIdx]  bIdx:%u accScSize:%u dealScSize:%u headSize:%u curStartPos_:%u curActSeqLength_:%u \n", bIdx, accScSize, dealScSize, headSize, curStartPos_, curActSeqLength_);
            if (accScSize >= dealScSize) {
                scEnd = scStart + dealScSize;
                return;
            }
        } else {
            curActSeqLength_ = GetSeqLength(bIdx);
            curStartPos_ = GetStartPos(bIdx);
            uint32_t curBasicNum = GetScSize();
            uint32_t curBasicNumEnd = dealScSize - accScSize;
            
            // printf("[GetEndIdx] accScSize:%u curBasicNum:%u dealScSize:%u\n", accScSize, curBasicNum, dealScSize);
            if (accScSize + curBasicNum >= dealScSize) {
                scEnd = curBasicNumEnd;
                return;
            }
            accScSize += curBasicNum;
        }
    }
}

// 根据sc的开始索引计算vec输出时的b、sc的索引
template <typename COMP>
__aicore__ inline void CompressorBlockVector<COMP>::GetScIdxInfo(uint32_t bStart, uint32_t scStart, uint32_t dealScSize, uint32_t v2TcStart, uint32_t v2TcEnd, 
                                                            uint32_t &outputBStart, uint32_t &outputSStart, uint32_t &outputScSize)
{
    outputScSize = v2TcEnd - v2TcStart;
    uint32_t scEnd = 0;
    uint32_t bEnd = 0;
    CalcScEndIdx(bStart, scStart, v2TcStart, bEnd, scEnd);
    outputSStart = scEnd;
    outputBStart = bEnd;
    // 处理跳batch
    curActSeqLength_ = GetSeqLength(bEnd);
    curStartPos_ = GetStartPos(bEnd);
    uint32_t curScSize = GetScSize();
    if (curScSize == scEnd) {
        outputSStart = 0;
        outputBStart++;
    }
}

template <typename COMP>
 __aicore__ inline void CompressorBlockVector<COMP>::ComputeVec1(const RunInfo &info)
{
    DumpTensor(mm1ResTensor, 1, 128 * 256);
    // TODO 1分核
    SetMSplitInfo(info);
    uint32_t scLoopTimes = 0;
    uint32_t dLoopTimes = 0;
    uint32_t splitSize = BLOCK_VEC_BASE_BUFFER_SIZE / (constInfo_.cmpRatio * static_cast<uint32_t>(COMP::coff) * sizeof(T));
    if (splitSize < constInfo_.dBaseSize) {
        scLoopTimes = 1;
        dLoopTimes = (constInfo_.dBaseSize + (splitSize - 1)) / splitSize;
    } else {
        dLoopTimes = 1;
        scLoopTimes = splitSize / constInfo_.dBaseSize;
    }
    uint32_t sStart = mSplitInfo.vecStartS;
    uint32_t sEnd = 0;
    uint32_t bStart = mSplitInfo.vecStartB;
    uint32_t bEnd = 0;
    uint32_t remaindTcNum = mSplitInfo.dealTcNum;             // 剩余需要处理的Tc块
    uint32_t tmpTcNum = 0;
    // printf("[SetMSplitInfo] vecStartB:%u vecEndB:%u vecStartS:%u vecEndS:%u dealTcNum:%u\n", mSplitInfo.vecStartB, mSplitInfo.vecEndB, mSplitInfo.vecStartS, mSplitInfo.vecEndS, mSplitInfo.dealTcNum);
    for (uint32_t i = 0; i < scLoopTimes; i++) {
        for (uint32_t j = 0; j < dLoopTimes; j++) {
            // 计算当前需要处理的splitSize个Tc块的b、s的开始结束索引
            if (remaindTcNum == 0) {
                return;
            }
            tmpTcNum = splitSize <= remaindTcNum ? splitSize : remaindTcNum;
            remaindTcNum -= tmpTcNum;
            CalcTcEndIdx(bStart, sStart, tmpTcNum, bEnd, sEnd);
            // printf("[CalcTcEndIdx] bStart:%u bEnd:%u sStart:%u sEnd:%u\n", bStart, bEnd, sStart, sEnd);
            for (uint32_t k = bStart; k <= bEnd; k++) {
                // 计算当前batch的seq 开始结束索引
                curActSeqLength_ = GetSeqLength(k);
                uint32_t curSStart = 0;
                uint32_t curSEnd = curActSeqLength_;
                if (k == bStart) {
                    curSStart = sStart;
                }
                if (k == bEnd) {
                    curSEnd = sEnd;
                }
                // printf("[IDX] b:%u sStart:%u sEnd:%u curSStart:%u curSEnd:%u\n", k, sStart, sEnd, curSStart, curSEnd);
                // 从UB拷贝到32k空间
                // 存state
                // 从state取
                // overlap
                
            }
            sStart = sEnd;
            bStart = bEnd;
            // 处理刚好是结尾跳batch
            if (sEnd == curActSeqLength_) {
                sStart = 0;
                bStart ++;
            }
        }
    }

}

template <typename COMP> 
__aicore__ inline void CompressorBlockVector<COMP>::ComputeVec2(const Compressor::RunInfo &info)
{
    SplitCoreV2(info);
    uint32_t vec2DealM = v2TcEndIdx - v2TcStartIdx;
    uint32_t loopCount = (vec2DealM + v2MBaseSize - 1) / v2MBaseSize;
    for (uint32_t v2LoopIdx = 0, dealSize = v2MBaseSize; v2LoopIdx < loopCount; ++v2LoopIdx) {
        if (v2LoopIdx == loopCount - 1) {
            dealSize = vec2DealM - (v2LoopIdx - 1) * v2MBaseSize;
        }
        // DealVec2BaseBlock(info, v2LoopIdx * v2MBaseSize, dealSize);
        pingpongFlag ^= 1;
    }
}

template <typename COMP> 
__aicore__ inline void CompressorBlockVector<COMP>::DealVec2BaseBlock(const Compressor::RunInfo& info, uint32_t startRow, uint32_t dealRowCount)
{
    uint32_t computeSize = dealRowCount * constInfo_.headDim;
    uint64_t inGmOffset = vec1ResGmStart + startRow * constInfo_.headDim;
    // CopyIn
    LocalTensor<T> vec1ResUb =  inputBuf1.Get<T>()[pingpongFlag * BLOCK_VEC_BASE_BUFFER_SIZE];
    LocalTensor<X_T> outputUb;
    WaitFlag<HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG + pingpongFlag);
    DataCopy(vec1ResUb, vec1ResGm_[inGmOffset], computeSize);

    // RmsNorm
    RmsNorm(info, startRow, dealRowCount);
    // rope
    CalRope(info);
    // CopyOut
    CopyFinalResultOut(info, outputUb);
    SetFlag<HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG + pingpongFlag);
}

template <typename COMP> 
__aicore__ inline void CompressorBlockVector<COMP>::RmsNorm(const Compressor::RunInfo& info, uint32_t startRow, uint32_t dealRowCount)
{
    struct RmsNormParam rmsNormParams;
    // RmsNormVF(const LocalTensor<T> &outputLocal, const LocalTensor<T> &inputLocal, const LocalTensor<GammaType> &gammaLocal,
    //     rmsNormParams);
}

template <typename COMP> 
__aicore__ inline void CompressorBlockVector<COMP>::CalRope(const Compressor::RunInfo& info)
{

}

template <typename COMP> 
__aicore__ inline void CompressorBlockVector<COMP>::SplitCoreV2(const Compressor::RunInfo& info)
{   
    // 累积N个基本块数据后做vec2，N=2，传入的RunInfo包含该组核处理的数据块的bStart、bEnd、sStart、sEnd以及dealTcCount；
    // 每组核切M方向将C1/V1后的数据分8 * 2个vec核上进行V2计算
    // 每次进行v2计算都会根据当前情况将workspace中的每组核处理的数据重新分到当前组的vec核

    // Input: syncAll前每组cube核处理的实际数据块在batch及s方向的起止idx及实际数据量(m方向)
    // Output: 每个vec核的处理数据块在m方向的起止位置及输出到Gm上的起始位置
    uint32_t coreNum = constInfo_.dBasicBlockNum * 2; // 组中有多少个vec核:16
    uint32_t currCoreIdx = GetBlockIdx(); // 当前vec核ID
    uint32_t curVecCoreGroupIdx = currCoreIdx / coreNum; // 当前vec核所在组ID
    vec1ResGmStart = curVecCoreGroupIdx * constInfo_.nSize * constInfo_.tcBaseSize * constInfo_.headDim;
    // 1.计算总vec2基本块数量
    // uint64_t totalBaseNum = (constInfo_.coreGroupNum * constInfo_.nSize * constInfo_.tcBaseSize + v2MBaseSize - 1) / v2MBaseSize; // TODO:不是按照实际数据量计算，暂时按照m方向完整基本块计算数据量 
    uint64_t totalBaseNum = info.dealTcNum; // 当前组核累积的实际数据量
    // 2.每个vec核上分到的数据量
    uint32_t avgBaseNum = 1;
    if (totalBaseNum > coreNum) {
        avgBaseNum = (totalBaseNum + coreNum - 1) / coreNum;
        // uint32_t remainder = totalBaseNum % coreNum;
        // avgBaseNum = (currCoreIdx % coreNum) < remainder ? avgBaseNum + 1 : avgBaseNum;
    } else {
        usedCoreNum = totalBaseNum;
    }
    if (currCoreIdx % coreNum >= usedCoreNum) {
        return;
    }
    // 3.计算每个vec核的起始结束位置
    uint32_t accumBaseNum = 0; // 当前累积的基本块数
    uint32_t targetBaseNum = (currCoreIdx % coreNum + 1) * avgBaseNum; // 当前vec核目标要达到的基本块数量
    uint32_t targetStartBaseNum = targetBaseNum - avgBaseNum; // 分当前vec核时前面已经完成分核的基本块数量
    bool setStart = false;
    for (uint32_t i = 0; i < totalBaseNum; ++i) {
        if (accumBaseNum >= totalBaseNum) {
            return;
        }
        accumBaseNum += 1;
        if (!setStart && (accumBaseNum >= targetStartBaseNum)) {
            v2TcStartIdx = i;
            setStart = true;
        } 
        if (accumBaseNum >= targetBaseNum || i == (totalBaseNum - 1)) {
            // 更新当前核的End分核信息
            v2TcEndIdx = i + 1;
            GetScIdxInfo(info.bStart, info.sStart, info.dealScSize, v2TcStartIdx, v2TcEndIdx,
                OutputBStartIdx, OutputSStartIdx, OutputSize);
            return;
        }
    }
}

template <typename COMP> 
__aicore__ inline void CompressorBlockVector<COMP>::CopyFinalResultOut(const Compressor::RunInfo& info, LocalTensor<X_T> &ComOutUb)
{
    int64_t outOffset;

}
} // namespace Compressor

#endif // COMPRESSOR_BLOCK_VECTOR_H