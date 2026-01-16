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
 * \file sparse_attn_sharedkv_scfa_kernel.h
 * \brief
 */

#ifndef SPARSE_ATTN_SHAREDKV_SCFA_KERNEL_H
#define SPARSE_ATTN_SHAREDKV_SCFA_KERNEL_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "../sparse_attn_sharedkv_common.h"
#include "sparse_attn_sharedkv_scfa_block_cube.h"
#include "sparse_attn_sharedkv_scfa_block_vector.h"

using namespace matmul;
using AscendC::CacheMode;
using AscendC::CrossCoreSetFlag;
using AscendC::CrossCoreWaitFlag;

// 由于S2循环前，RunInfo还没有赋值，使用Bngs1Param临时存放B、N、S1轴相关的信息；同时减少重复计算
struct TempLoopInfo {
    uint32_t bn2IdxInCurCore = 0;
    uint32_t bIdx = 0U;
    uint32_t n2Idx = 0U;
    uint64_t s2BasicSizeTail = 0U; // S2方向循环的尾基本块大小
    uint32_t s2LoopTimes = 0U;     // S2方向循环的总次数，无论TND还是BXXD都是等于实际次数，不用减1
    uint64_t actS2Size = 0ULL;
    uint64_t actS2SizeOri = 0ULL;
    bool curActSeqLenIsZero = false;
    int32_t nextTokensPerBatch = 0;

    uint64_t actS1Size = 1ULL;     // TND场景下当前Batch循环处理的S1轴的大小
    uint32_t tndCoreStartKVSplitPos;
    bool tndIsS2SplitCore;

    uint32_t gS1Idx = 0U;
    uint32_t s1StartIdx = 0;
    uint32_t s1EndIdx = 0;
    uint64_t mBasicSizeTail = 0U;  // gS1方向循环的尾基本块大小


    uint32_t cmpLoopTimes = 0;
    uint32_t oriLoopTimes = 0;

    
    // sparsemode = 4
    int64_t oriMaskRight = 0; // 对应ori mask的时候 右边那条线
    int64_t oriMaskLeft = 0; // 对应ori mask的时候 左边那条线

    // sparsemode = 3
    int64_t cmpMaskRight = 0; // 对应cmp mask的时候 右边那个线
};

template <typename SAST> class SparseAttnSharedkvScfa {
public:
    // 中间计算数据类型为float，高精度模式
    using T = float;
    using Q_T = typename SAST::queryType;
    using KV_T = typename SAST::kvType;
    using OUT_T = typename SAST::outputType;
    using SINKS_T = float;
    using UPDATE_T = T;
    using MM1_OUT_T = T;
    using MM2_OUT_T = T;

    __aicore__ inline SparseAttnSharedkvScfa(){};
    __aicore__ inline void Init(__gm__ uint8_t *query, __gm__ uint8_t *oriKV, __gm__ uint8_t *cmpKV,
                                __gm__ uint8_t *cmpSparseIndices, __gm__ uint8_t* oriBlockTable,
                                __gm__ uint8_t* cmpBlockTable, __gm__ uint8_t *cuSeqlensQ,
                                __gm__ uint8_t *seqUsedKV, __gm__ uint8_t *sinks,
                                __gm__ uint8_t *metadata, __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace, 
                                const SparseAttnSharedkvTilingData *__restrict tiling, __gm__ uint8_t *gmTiling, TPipe *tPipe);

    __aicore__ inline void Process();

private:
    static constexpr bool PAGE_ATTENTION = SAST::pageAttention;
    static constexpr int TEMPLATE_MODE = SAST::templateMode;
    static constexpr bool FLASH_DECODE = SAST::flashDecode;
    static constexpr SAS_LAYOUT LAYOUT_T = SAST::layout;
    static constexpr SAS_LAYOUT KV_LAYOUT_T = SAST::kvLayout;

    static constexpr uint32_t PRELOAD_NUM = 2;
    static constexpr uint32_t N_BUFFER_M_BASIC_SIZE = 256;
    static constexpr uint32_t SAS_PRELOAD_TASK_CACHE_SIZE = 3;

    static constexpr uint32_t SYNC_V0_C1_FLAG = 6;
    static constexpr uint32_t SYNC_C1_V1_FLAG = 7;
    static constexpr uint32_t SYNC_V1_C2_FLAG = 8;
    static constexpr uint32_t SYNC_C2_V2_FLAG = 9;
    static constexpr uint32_t SYNC_C2_V1_FLAG = 4;
    static constexpr uint32_t SYNC_V1_NUPDATE_C2_FLAG = 5;

    static constexpr uint64_t SYNC_MM2RES_BUF1_FLAG = 10;
    static constexpr uint64_t SYNC_MM2RES_BUF2_FLAG = 11;
    static constexpr uint64_t SYNC_FDOUTPUT_BUF_FLAG = 12;

    // static constexpr uint32_t BLOCK_ELEMENT_NUM = SASVectorBlock<SAST>::BYTE_BLOCK / sizeof(T);

    static constexpr uint64_t kvHeadNum = 1ULL;
    static constexpr uint64_t headDim = 512ULL;
    static constexpr uint64_t headDimAlign = 512ULL;
    static constexpr uint32_t msdIterNum = 2U;
    static constexpr uint64_t headDimRope = 64ULL;

    static constexpr uint32_t dbWorkspaceRatio = PRELOAD_NUM;

    const SparseAttnSharedkvTilingData *__restrict tilingData = nullptr;

    TPipe *pipe = nullptr;

    uint64_t mSizeVStart = 0ULL;
    int64_t threshold = 0;
    uint64_t topKBaseOffset = 0ULL;
    uint64_t s2BatchBaseOffset = 0;
    uint64_t tensorACoreOffset = 0ULL;
    uint64_t tensorARopeCoreOffset = 0ULL;
    uint64_t tensorBCoreOffset = 0ULL;
    uint64_t tensorBRopeCoreOffset = 0ULL;
    uint64_t attenOutOffset = 0ULL;

    uint32_t tmpBlockIdx = 0U;
    uint32_t aiCoreIdx = 0U;
    uint32_t usedCoreNum = 24U;

    ConstInfo constInfo{};
    TempLoopInfo tempLoopInfo{};

    SASCubeBlock<SAST> cubeBlock;
    SASVectorBlock<SAST> vectorBlock;

    GlobalTensor<Q_T> queryGm;
    GlobalTensor<KV_T> oriKvGm;
    GlobalTensor<KV_T> cmpKvGm;
    GlobalTensor<SINKS_T> sinksGm;

    GlobalTensor<OUT_T> attentionOutGm;
    GlobalTensor<int32_t> oriBlockTableGm;
    GlobalTensor<int32_t> cmpBlockTableGm;
    GlobalTensor<int32_t> topKGm;

    GlobalTensor<int32_t> actualSeqLengthsQGm;
    GlobalTensor<int32_t> actualSeqLengthsKVGm;

    // workspace
    GlobalTensor<MM1_OUT_T> mm1ResGm;
    GlobalTensor<KV_T> vec1ResGm;
    GlobalTensor<MM2_OUT_T> mm2ResGm;
    GlobalTensor<KV_T> kvMergeGm_;
    GlobalTensor<int32_t> kvValidSizeGm_;

    GlobalTensor<int32_t> mm2ResInt32Gm;
    GlobalTensor<UPDATE_T> vec2ResGm;

    GlobalTensor<T> accumOutGm;
    GlobalTensor<T> lseSumFdGm;
    GlobalTensor<T> lseMaxFdGm;

    // ================================Init functions==================================
    __aicore__ inline void InitTilingData();
    __aicore__ inline void InitCalcParamsEach();
    __aicore__ inline void InitBuffers();
    __aicore__ inline void InitActualSeqLen(__gm__ uint8_t *actualSeqLengthsQ, __gm__ uint8_t *actualSeqLengths);
    __aicore__ inline void InitOutputSingleCore();
    // ================================Process functions================================
    __aicore__ inline void ProcessBalance();
    __aicore__ inline void PreloadPipeline(uint32_t loop, uint64_t s2Start, uint64_t s2LoopIdx,
                                           RunInfo extraInfo[SAS_PRELOAD_TASK_CACHE_SIZE]);
    __aicore__ inline bool OriSkip(uint32_t s2LoopIdx);
    __aicore__ inline bool CmpSkip(uint32_t s2LoopIdx);
    __aicore__ inline bool IsSkip(uint32_t s2LoopIdx);
    // ================================Offset Calc=====================================
    __aicore__ inline void GetActualSeqLen(uint32_t bIdx, uint32_t s1Idx = 0);
    __aicore__ inline void GetSparseActualSeqLen();
    __aicore__ inline void UpdateInnerLoopCond();
    __aicore__ inline void DealActSeqLenIsZero(uint32_t bIdx, uint32_t s1Idx, uint32_t n2Idx);
    __aicore__ inline void CalcParams(uint32_t loop, uint64_t s2Start, uint32_t s2LoopIdx, RunInfo &info);
    __aicore__ inline void GetAxisStartIdx(uint32_t bN2EndPrev, uint32_t gS1EndPrev, uint32_t s2EndPrev);
    __aicore__ inline uint64_t GetBalanceActualSeqLengths(GlobalTensor<int32_t> &actualSeqLengths, uint32_t bIdx);
    __aicore__ inline uint32_t GetActualSeqLenKV(uint32_t bIdx);
    __aicore__ inline void GetBN2Idx(uint32_t bN2Idx, uint32_t &bIdx, uint32_t &n2Idx);
    __aicore__ inline void UpdateInner(uint32_t &s2End, uint32_t &curS2End, uint32_t s1Idx, bool isEnd);
    __aicore__ inline void GetPreNextTokensLeftUp();
    // ================================Mm1==============================================
    __aicore__ inline void ComputeMm1(const RunInfo &info);
    // ================================Mm2==============================================
    __aicore__ inline void ComputeMm2(const RunInfo &info);
    __aicore__ inline void InitAllZeroOutput(uint32_t bIdx, uint32_t s1Idx, uint32_t n2Idx);
};

template <typename SAST> __aicore__ inline void SparseAttnSharedkvScfa<SAST>::InitTilingData()
{
    // singleCoreParams
    usedCoreNum = tilingData->baseParams.usedCoreNum;
    // singleCoreTensorSize
    constInfo.mmResUbSize = 64 * 512;
    constInfo.bmm2ResUbSize = 64 * 512;
    constInfo.vec1ResUbSize = 64 * 512;
    // baseParams
    constInfo.batchSize = tilingData->baseParams.batchSize;
    constInfo.qHeadNum = constInfo.gSize = tilingData->baseParams.nNumOfQInOneGroup;
    constInfo.kvSeqSize = tilingData->baseParams.kvSeqSize;
    constInfo.qSeqSize = tilingData->baseParams.qSeqSize;
    constInfo.oriMaxBlockNumPerBatch = tilingData->baseParams.oriMaxBlockNumPerBatch;
    constInfo.kvCacheBlockSize = tilingData->baseParams.paBlockSize;
    constInfo.outputLayout = static_cast<SAS_LAYOUT>(tilingData->baseParams.outputLayout);
    constInfo.kvHeadNum = kvHeadNum;
    constInfo.headDim = headDim;
    constInfo.sparseBlockSize = tilingData->baseParams.sparseBlockSize;
    constInfo.oriMaskMode = tilingData->baseParams.oriMaskMode;
    constInfo.oriWinLeft = tilingData->baseParams.oriWinLeft;
    constInfo.oriWinRight = tilingData->baseParams.oriWinRight;

    constInfo.actualLenDimsQ = tilingData->baseParams.actualLenDimsQ;
    constInfo.actualLenDimsKV = tilingData->baseParams.actualLenDimsKV;
    // innerSplitParams
    constInfo.mBaseSize = 64;
    constInfo.s2BaseSize = 512;

    constInfo.attentionMode = ATTENTION_MODE::MLA_ABSORB;
    constInfo.combineHeadDim = headDim;

    constInfo.preLoadNum = PRELOAD_NUM;
    constInfo.nBufferMBaseSize = N_BUFFER_M_BASIC_SIZE;
    constInfo.syncV0C1 = SYNC_V0_C1_FLAG;
    constInfo.syncC1V1 = SYNC_C1_V1_FLAG;
    constInfo.syncV1C2 = SYNC_V1_C2_FLAG;
    constInfo.syncC2V2 = SYNC_C2_V2_FLAG;
    constInfo.syncV1NupdateC2 = SYNC_V1_NUPDATE_C2_FLAG;

    // cmp
    constInfo.cmpRatio = tilingData->cmpParams.cmpRatio;
    constInfo.sparseBlockCount = tilingData->cmpParams.sparseBlockCount;
    constInfo.cmpMaskMode = tilingData->cmpParams.cmpMaskMode;
    constInfo.cmpMaxBlockNumPerBatch = tilingData->cmpParams.cmpMaxBlockNumPerBatch;
    // ori

}

template <typename SAST> __aicore__ inline void SparseAttnSharedkvScfa<SAST>::InitBuffers()
{
    if ASCEND_IS_AIV {
        vectorBlock.InitBuffers(pipe);
    } else {
        cubeBlock.InitBuffers(pipe);
    }
}

template <typename SAST>
__aicore__ inline void
SparseAttnSharedkvScfa<SAST>::InitActualSeqLen(__gm__ uint8_t *actualSeqLengthsQ,
                                                          __gm__ uint8_t *actualSeqLengths)
{
    if (constInfo.actualLenDimsKV != 0) {
        actualSeqLengthsKVGm.SetGlobalBuffer((__gm__ int32_t *)actualSeqLengths, constInfo.actualLenDimsKV);
    }
    if (constInfo.actualLenDimsQ != 0) {
        actualSeqLengthsQGm.SetGlobalBuffer((__gm__ int32_t *)actualSeqLengthsQ, constInfo.actualLenDimsQ);
    }
}

template <typename SAST>
__aicore__ inline void SparseAttnSharedkvScfa<SAST>::InitAllZeroOutput(uint32_t bIdx, uint32_t s1Idx,
                                                                                  uint32_t n2Idx)
{
    if (constInfo.outputLayout == SAS_LAYOUT::TND) {
        uint32_t tBase = bIdx == 0 ? 0 : actualSeqLengthsQGm.GetValue(bIdx - 1);
        uint32_t s1Count = tempLoopInfo.actS1Size;

        uint64_t attenOutOffset = (tBase + s1Idx) * kvHeadNum * constInfo.gSize * headDim +   // T轴、s1轴偏移
                                    n2Idx * constInfo.gSize * headDim;                        // N2轴偏移
        matmul::InitOutput<OUT_T>(attentionOutGm[attenOutOffset], constInfo.gSize * headDim, 0);
    } else if (constInfo.outputLayout == SAS_LAYOUT::BSND) {
        uint64_t attenOutOffset = bIdx * constInfo.qSeqSize * kvHeadNum * constInfo.gSize * headDim +
                                    s1Idx * kvHeadNum * constInfo.gSize * headDim + // B轴、S1轴偏移
                                    n2Idx * constInfo.gSize * headDim;              // N2轴偏移
        matmul::InitOutput<OUT_T>(attentionOutGm[attenOutOffset], constInfo.gSize * headDim, 0);
    }
}

template <typename SAST>
__aicore__ inline void SparseAttnSharedkvScfa<SAST>::InitOutputSingleCore()
{
    uint32_t coreNum = GetBlockNum();
    if (coreNum != 0) {
        uint64_t totalOutputSize = constInfo.batchSize * constInfo.qHeadNum * constInfo.qSeqSize * constInfo.headDim;
        uint64_t singleCoreSize = (totalOutputSize + (2 * coreNum) - 1) / (2 * coreNum);  // 2 means c:v = 1:2
        uint64_t tailSize = totalOutputSize - tmpBlockIdx * singleCoreSize;
        uint64_t singleInitOutputSize = tailSize < singleCoreSize ? tailSize : singleCoreSize;
        if (singleInitOutputSize > 0) {
            matmul::InitOutput<OUT_T>(attentionOutGm[tmpBlockIdx * singleCoreSize], singleInitOutputSize, 0);
        }
        SyncAll();
    }
}

template <typename SAST>
__aicore__ inline void SparseAttnSharedkvScfa<SAST>::GetActualSeqLen(uint32_t bIdx, uint32_t s1Idx)
{
    tempLoopInfo.actS2SizeOri = GetActualSeqLenKV(bIdx);
    tempLoopInfo.actS1Size = GetBalanceActualSeqLengths(actualSeqLengthsQGm, bIdx);
}

template <typename SAST>
__aicore__ inline void SparseAttnSharedkvScfa<SAST>::GetSparseActualSeqLen()
{
    // 行无效通过ori部分判断, ori部分如果有行无效那么ori和cmp都有
    if (tempLoopInfo.oriMaskRight < 0 && tempLoopInfo.s1EndIdx < -tempLoopInfo.oriMaskRight) {
        tempLoopInfo.actS2Size = 0;
        return;
    }
    
    // 对于cmp部分还有top k, tempLoopInfo.actS2Size只针对cmp
    int64_t threshold = tempLoopInfo.actS2SizeOri;
    if (constInfo.cmpMaskMode == 3) {
        threshold = static_cast<int64_t>(tempLoopInfo.cmpMaskRight) + static_cast<int64_t>(tempLoopInfo.s1EndIdx) + 1;
    }
    tempLoopInfo.actS2Size = (constInfo.sparseBlockCount * constInfo.sparseBlockSize > threshold) ?
                                           threshold :
                                           constInfo.sparseBlockCount * constInfo.sparseBlockSize;
}

template <typename SAST>
__aicore__ inline uint32_t SparseAttnSharedkvScfa<SAST>::GetActualSeqLenKV(uint32_t bIdx)
{
    if constexpr (KV_LAYOUT_T == SAS_LAYOUT::TND) {
        if (bIdx >= 0) {
            return actualSeqLengthsKVGm.GetValue(bIdx + 1) - actualSeqLengthsKVGm.GetValue(bIdx);
        } else {
            return 0;
        }
    } else {
        if (constInfo.actualLenDimsKV == 0) {
            return constInfo.kvSeqSize;
        } else if (constInfo.actualLenDimsKV == 1) {
            return actualSeqLengthsKVGm.GetValue(0);
        } else {
            return actualSeqLengthsKVGm.GetValue(bIdx);
        }
    }
}

template <typename SAST>
__aicore__ inline void SparseAttnSharedkvScfa<SAST>::DealActSeqLenIsZero(uint32_t bIdx, uint32_t s1Idx,
                                                                                    uint32_t n2Idx)
{
    if ASCEND_IS_AIV {
        InitAllZeroOutput(bIdx, s1Idx, n2Idx);
    }
}

template <typename SAST>
__aicore__ inline void SparseAttnSharedkvScfa<SAST>::GetPreNextTokensLeftUp()
{
    if (constInfo.cmpMaskMode == 3) {
        tempLoopInfo.nextTokensPerBatch =
            static_cast<int32_t>(tempLoopInfo.actS2SizeOri) - static_cast<int32_t>(tempLoopInfo.actS1Size);
    }
}

template <typename SAST> __aicore__ inline void SparseAttnSharedkvScfa<SAST>::UpdateInnerLoopCond()
{
    if ((tempLoopInfo.actS2Size == 0) || (tempLoopInfo.actS1Size == 0)) {
        tempLoopInfo.curActSeqLenIsZero = true;
        return;
    }
    tempLoopInfo.curActSeqLenIsZero = false;
    tempLoopInfo.mBasicSizeTail = (tempLoopInfo.actS1Size * constInfo.gSize) % constInfo.mBaseSize;
    tempLoopInfo.mBasicSizeTail =
        (tempLoopInfo.mBasicSizeTail == 0) ? constInfo.mBaseSize : tempLoopInfo.mBasicSizeTail;
    tempLoopInfo.s2LoopTimes = 0;
}

template <typename SAST>
__aicore__ inline void SparseAttnSharedkvScfa<SAST>::UpdateInner(uint32_t &s2End, uint32_t &curS2End,
                                                                            uint32_t s1Idx, bool isEnd)
{
    if ((tempLoopInfo.actS2Size == 0) || (tempLoopInfo.actS1Size == 0)) {
        tempLoopInfo.curActSeqLenIsZero = true;
        return;
    }
    tempLoopInfo.curActSeqLenIsZero = false;
    tempLoopInfo.mBasicSizeTail = (tempLoopInfo.actS1Size * constInfo.gSize) % constInfo.mBaseSize;
    tempLoopInfo.mBasicSizeTail =
        (tempLoopInfo.mBasicSizeTail == 0) ? constInfo.mBaseSize : tempLoopInfo.mBasicSizeTail;
    tempLoopInfo.s2LoopTimes = 0;
}


template <typename SAST>
__aicore__ inline bool SparseAttnSharedkvScfa<SAST>::OriSkip(uint32_t s2LoopIdx)
{
    // 左闭右闭, 只针对sparse mode = 4, 可扩展, 更新left和right计算公式即可
    uint32_t s1StartIdx = tempLoopInfo.s1StartIdx;
    uint32_t s2StartIdx = s2LoopIdx * constInfo.s2BaseSize;
    uint32_t s1EndIdx = tempLoopInfo.s1EndIdx;
    uint32_t s2EndIdx = ((s2StartIdx + constInfo.s2BaseSize) < tempLoopInfo.actS2SizeOri ? (s2StartIdx + constInfo.s2BaseSize) : tempLoopInfo.actS2SizeOri) - 1;
    int64_t min_diff = static_cast<int64_t>(s2StartIdx) - static_cast<int64_t>(s1EndIdx);
    int64_t max_diff = static_cast<int64_t>(s2EndIdx) - static_cast<int64_t>(s1StartIdx);
    // 满足条件, 不跳过
    if (min_diff <= tempLoopInfo.oriMaskRight && max_diff >= tempLoopInfo.oriMaskLeft) {
        return false;
    }
    return true;
}

template <typename SAST>
__aicore__ inline bool SparseAttnSharedkvScfa<SAST>::CmpSkip(uint32_t s2LoopIdx)
{   
    // sparse mode = 3, 
    // 对于Cmp的s2 loop times为 tempLoopInfo.actS2SizeOri // ratio, 不够除就丢掉
    // 恢复压缩前, 只对S2进行压缩
    uint32_t oS2BaseSize = constInfo.s2BaseSize * constInfo.cmpRatio;
    uint32_t s1StartIdx = tempLoopInfo.s1StartIdx;
    uint32_t s2StartIdx = s2LoopIdx * oS2BaseSize;
    uint32_t s1EndIdx = tempLoopInfo.s1EndIdx;
    uint32_t s2EndIdx = ((s2StartIdx + oS2BaseSize) < tempLoopInfo.actS2SizeOri ? (s2StartIdx + oS2BaseSize) : tempLoopInfo.actS2SizeOri) - 1;
    int64_t min_diff = static_cast<int64_t>(s2StartIdx) - static_cast<int64_t>(s1EndIdx);
    int64_t max_diff = static_cast<int64_t>(s2EndIdx) - static_cast<int64_t>(s1StartIdx);
    if (min_diff <= tempLoopInfo.cmpMaskRight) {
        // 满足mask条件也不能跳, 还需要满足一个条件
        // 还需要判断s1EndIdx处, 当前基本块里面处理的数据除以ratio是否大于0
        uint32_t s2NoMaskSize = s1EndIdx - tempLoopInfo.actS1Size + tempLoopInfo.actS2SizeOri + 1;
        if ((s2NoMaskSize - s2StartIdx) / constInfo.cmpRatio > 0) {
            return false;
        }
    }
    return true;
}



template <typename SAST>
__aicore__ inline bool SparseAttnSharedkvScfa<SAST>::IsSkip( uint32_t s2LoopIdx)
{

    bool isSkip = false;
    // 一个基本块只能是ori或者cmp
    if (s2LoopIdx < tempLoopInfo.oriLoopTimes) {
        isSkip = OriSkip(s2LoopIdx);
    } else {
        isSkip = CmpSkip(s2LoopIdx - tempLoopInfo.oriLoopTimes);
    }
    return isSkip;
}

template <typename SAST>
__aicore__ inline void SparseAttnSharedkvScfa<SAST>::Init(
                                __gm__ uint8_t *query, __gm__ uint8_t *oriKV, __gm__ uint8_t *cmpKV,
                                __gm__ uint8_t *cmpSparseIndices, __gm__ uint8_t* oriBlockTable,
                                __gm__ uint8_t* cmpBlockTable, __gm__ uint8_t *cuSeqlensQ,
                                __gm__ uint8_t *seqUsedKV, __gm__ uint8_t *sinks,
                                __gm__ uint8_t *metadata, __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace, 
                                const SparseAttnSharedkvTilingData *__restrict tiling, __gm__ uint8_t *gmTiling, TPipe *tPipe)
{
    if ASCEND_IS_AIV {
        tmpBlockIdx = GetBlockIdx(); // vec:0-47
        aiCoreIdx = tmpBlockIdx / 2;
    } else {
        tmpBlockIdx = GetBlockIdx(); // cube:0-23
        aiCoreIdx = tmpBlockIdx;
    }

    // init tiling data
    tilingData = tiling;

    InitTilingData();
    InitActualSeqLen(cuSeqlensQ, seqUsedKV);

    // 初始化计算参数 分和函數沒有
    InitCalcParamsEach();
    pipe = tPipe;

    // init global buffer
    queryGm.SetGlobalBuffer((__gm__ Q_T *)query);
    oriKvGm.SetGlobalBuffer((__gm__ KV_T *)oriKV);
    cmpKvGm.SetGlobalBuffer((__gm__ KV_T *)cmpKV);
    
    if (sinks != nullptr) {
        sinksGm.SetGlobalBuffer((__gm__ SINKS_T *)sinks);
    }

    attentionOutGm.SetGlobalBuffer((__gm__ OUT_T *)attentionOut);

    if ASCEND_IS_AIV {
        if (LAYOUT_T != SAS_LAYOUT::TND) {
            if (constInfo.needInit) {
                InitOutputSingleCore();
            }
        }
    }

    if constexpr (PAGE_ATTENTION) {
        oriBlockTableGm.SetGlobalBuffer((__gm__ int32_t *)oriBlockTable);
        cmpBlockTableGm.SetGlobalBuffer((__gm__ int32_t *)cmpBlockTable);
    }
    topKGm.SetGlobalBuffer((__gm__ int32_t *)cmpSparseIndices);

    // workspace 内存排布
    // |Q--|mm1ResGm|vec1ResGm|mm2ResGm|vec2ResGm
    // |Core0_Q1-Core0_Q2-Core1_Q1-Core1_Q2....Core32_Q1-Core32_Q2|Core0_mmRes
    uint64_t offset = 0;
    mm1ResGm.SetGlobalBuffer(
        (__gm__ MM1_OUT_T *)(workspace + offset +
                             aiCoreIdx * dbWorkspaceRatio * constInfo.mmResUbSize * sizeof(MM1_OUT_T)));
    offset += GetBlockNum() * dbWorkspaceRatio * constInfo.mmResUbSize * sizeof(MM1_OUT_T);

    vec1ResGm.SetGlobalBuffer(
        (__gm__ Q_T *)(workspace + offset + aiCoreIdx * dbWorkspaceRatio * constInfo.mmResUbSize * sizeof(KV_T)));
    offset += GetBlockNum() * dbWorkspaceRatio * constInfo.mmResUbSize * sizeof(KV_T);

    mm2ResGm.SetGlobalBuffer(
        (__gm__ MM2_OUT_T *)(workspace + offset +
                             aiCoreIdx * dbWorkspaceRatio * constInfo.bmm2ResUbSize * sizeof(MM2_OUT_T)));
    offset += GetBlockNum() * dbWorkspaceRatio * constInfo.bmm2ResUbSize * sizeof(MM2_OUT_T);
    mm2ResInt32Gm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(mm2ResGm.GetPhyAddr(0)));

    vec2ResGm.SetGlobalBuffer((__gm__ T *)(workspace + offset +
                              aiCoreIdx * dbWorkspaceRatio * constInfo.bmm2ResUbSize * sizeof(T)));
    offset += GetBlockNum() * dbWorkspaceRatio * constInfo.bmm2ResUbSize * sizeof(T);
    
    // v模板: s2  d+rope bufNum
    kvMergeGm_.SetGlobalBuffer((__gm__ KV_T *)(workspace + offset + aiCoreIdx * 512 * 512 * 4 * sizeof(KV_T)));
    offset += GetBlockNum() * 512 * 512 * 4 * sizeof(KV_T);

    kvValidSizeGm_.SetGlobalBuffer(
        (__gm__ int32_t *)(workspace + offset + (aiCoreIdx * 2) * 128 * 4 * sizeof(int32_t)));

    if ASCEND_IS_AIV {
        vectorBlock.InitParams(constInfo, tilingData);
        vectorBlock.InitVec0GlobalTensor(kvValidSizeGm_, kvMergeGm_, oriKvGm, cmpKvGm, oriBlockTableGm, cmpBlockTableGm);
        vectorBlock.InitVec1GlobalTensor(mm1ResGm, vec1ResGm, actualSeqLengthsQGm, actualSeqLengthsKVGm, lseMaxFdGm, lseSumFdGm, topKGm, sinksGm);
        vectorBlock.InitVec2GlobalTensor(accumOutGm, vec2ResGm, mm2ResGm, attentionOutGm);
    }

    if ASCEND_IS_AIC {
        cubeBlock.InitParams(constInfo);
        cubeBlock.InitMm1GlobalTensor(queryGm, oriKvGm, cmpKvGm, mm1ResGm);
        cubeBlock.InitMm2GlobalTensor(vec1ResGm, mm2ResGm, attentionOutGm);
    }
    // 要在InitParams之后执行
    if (pipe != nullptr) {
        InitBuffers();
    }
}

template <typename SAST> __aicore__ inline void SparseAttnSharedkvScfa<SAST>::InitCalcParamsEach()
{
    // TODO: 针对decode首case处理
    constInfo.bN2Start = 0;
    constInfo.gS1Start = 0;
    constInfo.s2Start = 0;

    if (aiCoreIdx == 0) {
        constInfo.bN2End = 2;
        constInfo.gS1End = 1; // 右开
        constInfo.s2End = 1;
    } else {
        constInfo.bN2End = 0;
        constInfo.gS1End = 0;
    }
}


template <typename SAST>
__aicore__ inline void SparseAttnSharedkvScfa<SAST>::CalcParams(uint32_t loop, uint64_t s2Start,
                                                                           uint32_t s2LoopIdx, RunInfo &info)
{
    info.loop = loop;
    info.bIdx = tempLoopInfo.bIdx;
    info.gS1Idx = tempLoopInfo.gS1Idx;
    info.s2Idx = s2LoopIdx;
    info.curSInnerLoopTimes = tempLoopInfo.s2LoopTimes;

    info.tndIsS2SplitCore = tempLoopInfo.tndIsS2SplitCore;
    info.tndCoreStartKVSplitPos = tempLoopInfo.tndCoreStartKVSplitPos;
    info.isBmm2Output = false;

    info.actS1Size = tempLoopInfo.actS1Size;
    
    
    info.actMBaseSize = constInfo.mBaseSize;
    uint32_t remainedGS1Size = tempLoopInfo.actS1Size * constInfo.gSize - tempLoopInfo.gS1Idx;
    if (remainedGS1Size <= constInfo.mBaseSize && remainedGS1Size > 0) {
        info.actMBaseSize = tempLoopInfo.mBasicSizeTail;
    }

    info.isValid = s2LoopIdx < tempLoopInfo.s2LoopTimes;

    if ASCEND_IS_AIV {
        info.mSize = info.actMBaseSize;
        info.mSizeV = (info.mSize <= 16) ? info.mSize : (((info.mSize + 15) / 16 + 1) / 2 * 16);
        info.mSizeVStart = 0;
        if (tmpBlockIdx % 2 == 1) {
            info.mSizeVStart = info.mSizeV;
            info.mSizeV = info.mSize - info.mSizeV;
        }
    }

    info.isChangeBatch = false;

    info.isFirstSInnerLoop = s2LoopIdx == s2Start;
    if (info.isFirstSInnerLoop) {
        tempLoopInfo.bn2IdxInCurCore++;
    }
    info.isLastS2Loop = s2LoopIdx == tempLoopInfo.s2LoopTimes - 1;
    info.bn2IdxInCurCore = tempLoopInfo.bn2IdxInCurCore - 1;
    uint64_t actualSeqQPrefixSum;
    if constexpr (LAYOUT_T == SAS_LAYOUT::TND) {
        actualSeqQPrefixSum = (info.bIdx <= 0) ? 0 : actualSeqLengthsQGm.GetValue(info.bIdx - 1);
    } else {
        actualSeqQPrefixSum = (info.bIdx <= 0) ? 0 : info.bIdx * constInfo.qSeqSize;
    }
    info.tndBIdxOffsetForQ = actualSeqQPrefixSum * constInfo.qHeadNum * headDim;

    uint64_t actualSeqKVPrefixSum;
    if constexpr (KV_LAYOUT_T == SAS_LAYOUT::TND) {
        actualSeqKVPrefixSum = (info.bIdx <= 0) ? 0 : actualSeqLengthsKVGm.GetValue(info.bIdx - 1);
    } else {
        actualSeqKVPrefixSum = (info.bIdx <= 0) ? 0 : info.bIdx * constInfo.kvSeqSize;
    }
    info.tndBIdxOffsetForKV = actualSeqKVPrefixSum * constInfo.kvHeadNum * headDim;

    if (info.isFirstSInnerLoop) {
        uint64_t tndBIdxRopeOffsetForQ = actualSeqQPrefixSum * constInfo.qHeadNum * headDimRope;
        tensorACoreOffset = info.tndBIdxOffsetForQ + info.gS1Idx * headDim;
        tensorARopeCoreOffset = tndBIdxRopeOffsetForQ + info.gS1Idx * headDimRope;
        
        uint64_t tndBIdxRopeOffsetForK = actualSeqKVPrefixSum * constInfo.kvHeadNum * headDimRope;
        tensorBCoreOffset = info.tndBIdxOffsetForKV + info.n2Idx * headDim;
        tensorBRopeCoreOffset = tndBIdxRopeOffsetForK + info.n2Idx * headDimRope;
        if (constInfo.cmpMaskMode == 3) {
            threshold = static_cast<int64_t>(tempLoopInfo.nextTokensPerBatch) + info.gS1Idx / constInfo.gSize + 1;
        } else {
            threshold = tempLoopInfo.actS2SizeOri;
        }
        if constexpr(LAYOUT_T == SAS_LAYOUT::BSND) {     // B,S1,N2 K
            topKBaseOffset = info.bIdx * constInfo.qSeqSize * constInfo.kvHeadNum * constInfo.sparseBlockCount +
                            info.gS1Idx / constInfo.gSize * constInfo.kvHeadNum * constInfo.sparseBlockCount +
                            info.n2Idx * constInfo.sparseBlockCount;
        } else if (LAYOUT_T == SAS_LAYOUT::TND) {        // T N2 K
            topKBaseOffset = info.tndBIdxOffsetForQ / constInfo.gSize / constInfo.headDim * constInfo.kvHeadNum *
                             constInfo.sparseBlockCount + info.n2Idx * constInfo.sparseBlockCount +
                             info.gS1Idx / constInfo.gSize * constInfo.kvHeadNum * constInfo.sparseBlockCount;
        } else {                                         // B N2 S1 K
            topKBaseOffset = info.bIdx * constInfo.kvHeadNum * constInfo.qSeqSize * constInfo.sparseBlockCount +
                            info.n2Idx * constInfo.qSeqSize * constInfo.sparseBlockCount +
                            info.gS1Idx / constInfo.gSize * constInfo.sparseBlockCount;
        }
    }
    info.topKBaseOffset = topKBaseOffset;
    info.threshold = threshold;
    info.tensorAOffset = tensorACoreOffset;
    info.tensorARopeOffset = tensorARopeCoreOffset;
    info.tensorBOffset = tensorBCoreOffset;
    info.tensorBRopeOffset = tensorBRopeCoreOffset;
    info.attenOutOffset = tensorACoreOffset;
    
    if (s2LoopIdx < tempLoopInfo.oriLoopTimes) {
        info.isOri = true;
        uint64_t sInnerOffsetDataSize = info.s2Idx * constInfo.s2BaseSize;
        if (s2LoopIdx + 1 == tempLoopInfo.oriLoopTimes) {
            info.actualSingleProcessSInnerSize = tempLoopInfo.actS2SizeOri - sInnerOffsetDataSize;
        } else {
            info.actualSingleProcessSInnerSize = constInfo.s2BaseSize;
        }

    } else {
        info.isOri = false;
        uint64_t sInnerOffsetDataSize = (s2LoopIdx - tempLoopInfo.oriLoopTimes) * constInfo.s2BaseSize;
        info.actualSingleProcessSInnerSize = tempLoopInfo.actS2Size / constInfo.cmpRatio - sInnerOffsetDataSize;
        if (info.actualSingleProcessSInnerSize <= 0) {
            info.actualSingleProcessSInnerSize = 0;
        } else {
            info.actualSingleProcessSInnerSize = info.actualSingleProcessSInnerSize < constInfo.s2BaseSize ?
                            info.actualSingleProcessSInnerSize : constInfo.s2BaseSize;
        }
    }
    info.actualSingleProcessSInnerSizeAlign =
        SASAlign((uint32_t)info.actualSingleProcessSInnerSize, (uint32_t)SASVectorBlock<SAST>::BYTE_BLOCK);
}

template <typename SAST>
__aicore__ inline void SparseAttnSharedkvScfa<SAST>::ComputeMm1(const RunInfo &info)
{
    uint32_t nBufferLoopTimes = (info.actMBaseSize + constInfo.nBufferMBaseSize - 1) / constInfo.nBufferMBaseSize;
    uint32_t nBufferTail = info.actMBaseSize - (nBufferLoopTimes - 1) * constInfo.nBufferMBaseSize;
    for (uint32_t i = 0; i < nBufferLoopTimes; i++) {
        MSplitInfo mSplitInfo;
        mSplitInfo.nBufferStartM = i * constInfo.nBufferMBaseSize;
        mSplitInfo.nBufferDealM = (i + 1 != nBufferLoopTimes) ? constInfo.nBufferMBaseSize : nBufferTail;
        cubeBlock.ComputeMm1(info, mSplitInfo);
        CrossCoreSetFlag<ConstInfo::SAS_SYNC_MODE2, PIPE_FIX>(constInfo.syncC1V1);
    }
}

template <typename SAST>
__aicore__ inline void SparseAttnSharedkvScfa<SAST>::ComputeMm2(const RunInfo &info)
{
    uint32_t nBufferLoopTimes = (info.actMBaseSize + constInfo.nBufferMBaseSize - 1) / constInfo.nBufferMBaseSize;
    uint32_t nBufferTail = info.actMBaseSize - (nBufferLoopTimes - 1) * constInfo.nBufferMBaseSize;
    for (uint32_t i = 0; i < nBufferLoopTimes; i++) {
        MSplitInfo mSplitInfo;
        mSplitInfo.nBufferStartM = i * constInfo.nBufferMBaseSize;
        mSplitInfo.nBufferDealM = (i + 1 != nBufferLoopTimes) ? constInfo.nBufferMBaseSize : nBufferTail;
        CrossCoreWaitFlag(constInfo.syncV1C2);
        cubeBlock.ComputeMm2(info, mSplitInfo);
        CrossCoreSetFlag<ConstInfo::SAS_SYNC_MODE2, PIPE_FIX>(constInfo.syncC2V2);
        CrossCoreSetFlag<ConstInfo::SAS_SYNC_MODE2, PIPE_FIX>(constInfo.syncC2V1);
    }
}

template <typename SAST> __aicore__ inline void SparseAttnSharedkvScfa<SAST>::Process()
{
    if (aiCoreIdx < usedCoreNum) {
        if ASCEND_IS_AIV {
            vectorBlock.AllocEventID();
            vectorBlock.InitSoftmaxDefaultBuffer();
        } else {
            cubeBlock.AllocEventID();
        }
        ProcessBalance();
        if ASCEND_IS_AIV {
            vectorBlock.FreeEventID();
        } else {
            cubeBlock.FreeEventID();
        }
    }
}

template <typename SAST>
__aicore__ inline void SparseAttnSharedkvScfa<SAST>::GetBN2Idx(uint32_t bN2Idx, uint32_t &bIdx, uint32_t &n2Idx)
{
    bIdx = bN2Idx / kvHeadNum;
    n2Idx = bN2Idx % kvHeadNum;
}

template <typename SAST> __aicore__ inline void SparseAttnSharedkvScfa<SAST>::ProcessBalance()
{
    RunInfo extraInfo[SAS_PRELOAD_TASK_CACHE_SIZE];
    uint32_t gloop = 0;
    uint32_t gS1LoopEnd;
    bool globalLoopStart = true;
    if ASCEND_IS_AIC {
        CrossCoreSetFlag<ConstInfo::SAS_SYNC_MODE2, PIPE_FIX>(constInfo.syncC2V1);
        CrossCoreSetFlag<ConstInfo::SAS_SYNC_MODE2, PIPE_MTE2>(3);
        CrossCoreSetFlag<ConstInfo::SAS_SYNC_MODE2, PIPE_MTE2>(3);
        CrossCoreSetFlag<ConstInfo::SAS_SYNC_MODE2, PIPE_MTE2>(3);
        CrossCoreSetFlag<ConstInfo::SAS_SYNC_MODE2, PIPE_MTE2>(3);
    }
    // 适配左闭右开
    if (constInfo.bN2Start == constInfo.bN2End) {
        if (constInfo.gS1Start != constInfo.gS1End || constInfo.s2Start != constInfo.s2End) {
            constInfo.bN2End += 1;
        }
    }
    for (uint32_t bN2LoopIdx = constInfo.bN2Start; bN2LoopIdx < constInfo.bN2End; bN2LoopIdx++) {
        GetBN2Idx(bN2LoopIdx, tempLoopInfo.bIdx, tempLoopInfo.n2Idx);
        GetActualSeqLen(tempLoopInfo.bIdx); // 获取actualSeqLength及ActualSeqLengthKV
        GetPreNextTokensLeftUp();
        tempLoopInfo.oriMaskRight = static_cast<int64_t>(tempLoopInfo.actS2SizeOri) - static_cast<int64_t>(tempLoopInfo.actS1Size) + constInfo.oriWinRight;
        tempLoopInfo.oriMaskLeft = static_cast<int64_t>(tempLoopInfo.actS2SizeOri)  - static_cast<int64_t>(tempLoopInfo.actS1Size) - constInfo.oriWinLeft;
        tempLoopInfo.cmpMaskRight = static_cast<int64_t>(tempLoopInfo.actS2SizeOri) - static_cast<int64_t>(tempLoopInfo.actS1Size);
        if (tempLoopInfo.actS1Size == 0) {
            continue;
        }
        int gS1SplitNum = (tempLoopInfo.actS1Size * constInfo.gSize + constInfo.mBaseSize - 1) / constInfo.mBaseSize;
        // 当处于最后一个BN2时, 且gS1End为0时, 说明当前BN2里的所有数据都在当前核处理
        gS1LoopEnd = (bN2LoopIdx == constInfo.bN2End - 1 && constInfo.gS1End != 0) ? constInfo.gS1End : gS1SplitNum;
        for (uint32_t gS1LoopIdx = constInfo.gS1Start; gS1LoopIdx < gS1LoopEnd; gS1LoopIdx++) {
            // 计算需要的数据, 避免重复计算
            tempLoopInfo.gS1Idx = gS1LoopIdx * constInfo.mBaseSize;
            tempLoopInfo.s1StartIdx = tempLoopInfo.gS1Idx / constInfo.gSize;
            tempLoopInfo.s1EndIdx = Min((tempLoopInfo.gS1Idx + constInfo.mBaseSize) / constInfo.gSize, tempLoopInfo.actS1Size) - 1;
            //
            GetSparseActualSeqLen(); // TopK值sparse完后的ActualSeqLengthKV
            UpdateInnerLoopCond();

            if (tempLoopInfo.curActSeqLenIsZero) {
                DealActSeqLenIsZero(tempLoopInfo.bIdx, gS1LoopIdx, tempLoopInfo.n2Idx);
            }
            uint32_t oriSplitNum = (tempLoopInfo.actS2SizeOri + constInfo.s2BaseSize - 1) / constInfo.s2BaseSize;
            uint32_t cmpSplitNum = (tempLoopInfo.actS2Size / constInfo.cmpRatio + constInfo.s2BaseSize - 1) / constInfo.s2BaseSize;
            uint32_t s2SplitNum = oriSplitNum + cmpSplitNum;
            bool isEnd = (bN2LoopIdx + 1 == constInfo.bN2End) && (gS1LoopIdx + 1 == constInfo.gS1End);
            tempLoopInfo.s2LoopTimes = s2SplitNum;
            tempLoopInfo.oriLoopTimes = oriSplitNum;
            tempLoopInfo.cmpLoopTimes = cmpSplitNum;
            uint32_t s2LoopEnd = (isEnd && constInfo.s2End != 0) ? constInfo.s2End : tempLoopInfo.s2LoopTimes;
            tempLoopInfo.s2LoopTimes = s2LoopEnd;
            // 分核修改后需要打开
            // 当前s2是否被切，决定了输出是否要写到attenOut上
            tempLoopInfo.tndIsS2SplitCore =
                ((constInfo.s2Start == 0) && (s2LoopEnd == s2SplitNum)) ? false : true;
            tempLoopInfo.tndCoreStartKVSplitPos = globalLoopStart ? constInfo.coreStartKVSplitPos : 0;
            uint32_t extraLoop = isEnd ? 2 : 0;
            uint32_t curTopKIdx = 0;
            uint64_t curOffsetInSparseBlock = 0;
            
            for (uint32_t s2LoopIdx = constInfo.s2Start; s2LoopIdx < (s2LoopEnd + extraLoop); s2LoopIdx++) {
                // PreloadPipeline loop初始值要求为 PRELOAD_NUM
                if (IsSkip(s2LoopIdx) && s2LoopIdx < s2LoopEnd) {
                    continue;
                }
                PreloadPipeline(gloop, constInfo.s2Start, s2LoopIdx, extraInfo);
                ++gloop;
            }
            globalLoopStart = false;
            constInfo.s2Start = 0;
        }
        constInfo.gS1Start = 0;
    }
    if ASCEND_IS_AIV {
        CrossCoreWaitFlag(constInfo.syncC2V1);
        CrossCoreWaitFlag(3);
        CrossCoreWaitFlag(3);
        CrossCoreWaitFlag(3);
        CrossCoreWaitFlag(3);
    }
}

template <typename SAST>
__aicore__ inline void
SparseAttnSharedkvScfa<SAST>::PreloadPipeline(uint32_t loop, uint64_t s2Start, uint64_t s2LoopIdx,
                                                         RunInfo extraInfo[SAS_PRELOAD_TASK_CACHE_SIZE])
{
    RunInfo &extraInfo0 = extraInfo[loop % SAS_PRELOAD_TASK_CACHE_SIZE];         // 本轮任务
    RunInfo &extraInfo2 = extraInfo[(loop + 2) % SAS_PRELOAD_TASK_CACHE_SIZE]; // 上一轮任务
    RunInfo &extraInfo1 = extraInfo[(loop + 1) % SAS_PRELOAD_TASK_CACHE_SIZE]; // 上两轮任务

    CalcParams(loop, s2Start, s2LoopIdx, extraInfo0);
    if (extraInfo0.isValid) {
        if ASCEND_IS_AIC {
            CrossCoreWaitFlag(constInfo.syncV0C1);
            ComputeMm1(extraInfo0);
        } else {
            CrossCoreWaitFlag(3);
            vectorBlock.MergeKv(extraInfo0);
            CrossCoreSetFlag<ConstInfo::SAS_SYNC_MODE2, PIPE_MTE3>(constInfo.syncV0C1);
        }
    }
    if (extraInfo2.isValid) {
        if ASCEND_IS_AIV {
            vectorBlock.ProcessVec1L(extraInfo2);
        }
        if ASCEND_IS_AIC {
            ComputeMm2(extraInfo2);
            CrossCoreSetFlag<ConstInfo::SAS_SYNC_MODE2, PIPE_MTE2>(3);
        }
    }
    if (extraInfo1.isValid) {
        if ASCEND_IS_AIV {
            vectorBlock.ProcessVec2L(extraInfo1);
        }
        extraInfo1.isValid = false;
    }
}

template <typename SAST>
__aicore__ inline uint64_t
SparseAttnSharedkvScfa<SAST>::GetBalanceActualSeqLengths(GlobalTensor<int32_t> &actualSeqLengths, uint32_t bIdx)
{
    if constexpr (LAYOUT_T == SAS_LAYOUT::TND) {
        if (bIdx >= 0) {
            return actualSeqLengths.GetValue(bIdx + 1) - actualSeqLengths.GetValue(bIdx);
        } else {
            return 0;
        }
    } else {
        if (constInfo.actualLenDimsQ == 0) {
            return constInfo.qSeqSize;
        } else if (constInfo.actualLenDimsQ == 1) {
            return actualSeqLengths.GetValue(0);
        } else {
            return actualSeqLengths.GetValue(bIdx);
        }
    }
}

template <typename SAST>
__aicore__ inline void SparseAttnSharedkvScfa<SAST>::GetAxisStartIdx(uint32_t bN2EndPrev,
                                                                                uint32_t s1GEndPrev,
                                                                                uint32_t s2EndPrev)
{
    uint32_t bEndPrev = bN2EndPrev / kvHeadNum;
    uint32_t actualSeqQPrev = GetBalanceActualSeqLengths(actualSeqLengthsQGm, bEndPrev);
    uint32_t s1GPrevBaseNum = (actualSeqQPrev * constInfo.gSize + constInfo.mBaseSize - 1) / constInfo.mBaseSize;
    constInfo.bN2Start = bN2EndPrev;
    constInfo.gS1Start = s1GEndPrev;
    
    constInfo.s2Start = 0;
    if (s1GEndPrev >= s1GPrevBaseNum - 1) { // 上个核把S1G处理完了
        constInfo.gS1Start = 0;
        constInfo.bN2Start++;
    } else {
        constInfo.gS1Start++;
    }
}
#endif // KV_QUANT_SPARSE_FLASH_ATTENTION_KERNEL_MLA_H