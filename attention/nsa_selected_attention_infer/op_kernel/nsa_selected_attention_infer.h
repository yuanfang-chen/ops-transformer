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
 * \file nsa_selected_attention_infer.h
 * \brief
 */

#pragma once

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "nsa_public_define.h"

using namespace matmul;
using AscendC::CacheMode;
using AscendC::CrossCoreSetFlag;
using AscendC::CrossCoreWaitFlag;
using AscendC::PipeBarrier;

#define PRE_LOAD_NUM 4

struct ExtraInfo {
    uint32_t loop = 0;
    uint32_t bIdx = 0;
    uint32_t n2Idx = 0;
    uint32_t s1Idx = 0;
    uint32_t s2Idx = 0;
    uint32_t bn2IdxInCurCore = 0;
    uint32_t bn2IdxInPreCore = 0;
    uint32_t curSInnerLoopTimes = 0;
    uint64_t tensorAOffset = 0;
    uint64_t tensorBOffset = 0;
    uint64_t tensorARopeOffset = 0;
    uint64_t tensorBRopeOffset = 0;
    uint64_t attenOutOffset = 0;

    uint32_t actualSingleProcessSInnerSize = 0;
    uint32_t actualSingleProcessSInnerSizeAlign = 0;
    bool isFirstSInnerLoop = false;
    bool needSetOrgShape = false;
    uint32_t s2BatchOffset = 0;
    ExtraInfo() {
        loop = 0;
        bIdx = 0;
        n2Idx = 0;
        s1Idx = 0;
        s2Idx = 0;
        bn2IdxInCurCore = 0;
        bn2IdxInPreCore = 0;
        curSInnerLoopTimes = 0;
        tensorAOffset = 0;
        tensorBOffset = 0;
        attenOutOffset = 0;

        actualSingleProcessSInnerSize = 0;
        actualSingleProcessSInnerSizeAlign = 0;
        isFirstSInnerLoop = false;
        needSetOrgShape = false;
        s2BatchOffset = 0;
    }
};

template <typename NSAT> class NsaSelectAttentionInfer {
public:
    __aicore__ inline NsaSelectAttentionInfer(){};
    __aicore__ inline void Init(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
                                __gm__ uint8_t *topkIndices, __gm__ uint8_t *attenMask, __gm__ uint8_t *blockTable,
                                __gm__ uint8_t *actualQSeqLengths, __gm__ uint8_t *actualSeqLengths,
                                __gm__ uint8_t *attentionOut,
                                __gm__ uint8_t *workspace,
                                const NsaSelectAttentionInferTilingData *__restrict tiling, __gm__ uint8_t *gmTiling,
                                TPipe *tPipe);
    
    __aicore__ inline void Process();
    __aicore__ inline bool IsFinish(uint32_t loop);

    // 中间计算数据类型为float，高精度模式
    using T = float;

    using Q_T = typename NSAT::queryType;
    using KV_T = typename NSAT::kvType;
    using OUT_T = typename NSAT::outputType;
    using ORIGIN_T = typename NSAT::orginalType;
    static constexpr LAYOUT LAYOUT_T = NSAT::layout; //change back by wy

    using MM_OUT_T = float;

protected:
    const NsaSelectAttentionInferTilingData *__restrict tilingData = nullptr;
    TPipe *pipe = nullptr;

    GlobalTensor<Q_T> queryGm;
    GlobalTensor<KV_T> keyGm;
    GlobalTensor<KV_T> valueGm;
    GlobalTensor<OUT_T> attentionOutGm;
    GlobalTensor<int32_t> blockTableGm;
    GlobalTensor<int32_t> topKGm;
    GlobalTensor<uint64_t> actualSeqLengthsGm;
    GlobalTensor<uint64_t> actualQSeqLengthsGm;
    // workspace
    GlobalTensor<MM_OUT_T> mm1ResGm;
    GlobalTensor<KV_T> vec1ResGm;
    GlobalTensor<MM_OUT_T> mm2ResGm;
    GlobalTensor<T> vec2ResGm;
    GlobalTensor<T> accumOutGm; // no
    GlobalTensor<T> lseSumFdGm; // no
    GlobalTensor<T> lseMaxFdGm; // no

    // L1
    // KP复用kpL1Tensor
    TBuf<TPosition::A1> kpBufL1;
    LocalTensor<KV_T> kpL1Tensor;

    TBuf<TPosition::A1> valueBufL1;
    LocalTensor<KV_T> vL1Tensor;

    TBuf<TPosition::A1> queryBufL1;
    LocalTensor<Q_T> qL1Tensor;

    // L1 buffer size
    static constexpr uint32_t L1_KP_SIZE = (128 * (128 + 64) * sizeof(KV_T));
    static constexpr uint32_t L1_V_SIZE = (2 * 128 * 128 * sizeof(KV_T));
    static constexpr uint32_t L1_Q_SIZE = (128 * (128 + 64) * sizeof(Q_T));

    // L0 buffer size
    static constexpr uint32_t L0A_PP_SIZE = (32 * 1024); // 128*128*2
    static constexpr uint32_t L0B_PP_SIZE = (32 * 1024);
    static constexpr uint32_t L0C_PP_SIZE = (64 * 1024); // 128 * 128 *4 

    // mte2 <> mte1 EventID
    static constexpr uint32_t KP_EVENT0 = EVENT_ID4;
    static constexpr uint32_t KP_EVENT1 = EVENT_ID5;
    static constexpr uint32_t V_EVENT0 = EVENT_ID2;
    static constexpr uint32_t Q_EVENT1 = EVENT_ID3;

    // m <> mte1 EventID
    static constexpr uint32_t L0A_EVENT0 = EVENT_ID3;
    static constexpr uint32_t L0A_EVENT1 = EVENT_ID4;
    static constexpr uint32_t L0B_EVENT0 = EVENT_ID5;
    static constexpr uint32_t L0B_EVENT1 = EVENT_ID6;

    // fix <> m EventID
    static constexpr uint32_t L0C_EVENT0 = EVENT_ID3;
    static constexpr uint32_t L0C_EVENT1 = EVENT_ID4;

    uint32_t kpL1BufIter = 0; // double buffer
    uint32_t vL1BufIter = 0;
    uint32_t aL0BufIter = 0;
    uint32_t bL0BufIter = 0;
    uint32_t cL0BufIter = 0;

    // queue
    TQue<QuePosition::VECIN, 1> inputQue1;   // 32K, inque
    TQue<QuePosition::VECIN, 1> inputQue2;   // 16K, inque
    TQue<QuePosition::VECOUT, 1> outputQue1; // 32K, outque
    TQue<QuePosition::A1, 1> inputQueL1A;
    TQue<QuePosition::B1, 1> inputQueL1B;
    // L0_A
    TBuf<TPosition::A2> tmpBufL0A;
    LocalTensor<KV_T> aL0TensorPingPong;
    // L0_B
    TBuf<TPosition::B2> tmpBufL0B;
    LocalTensor<KV_T> bL0TensorPingPong;
    // L0_C
    TBuf<TPosition::CO1> tmpBufL0C;
    LocalTensor<MM_OUT_T> cL0TensorPingPong;

    // 临时tbuf
    TBuf<> tmpBuff1; // 32K
    TBuf<> tmpBuff2; // 32K

    // 常驻tbuf
    TBuf<> softmaxMaxBuff[PRE_LOAD_NUM]; // PRE_LOAD_NUM * 2K
    TBuf<> softmaxExpBuff[PRE_LOAD_NUM]; // PRE_LOAD_NUM * 2K
    TBuf<> softmaxSumBuff[PRE_LOAD_NUM]; // PRE_LOAD_NUM * 2K
    TBuf<> softmaxMaxDefaultBuff;     // 2K
    TBuf<> softmaxSumDefaultBuff;     // 2K

    LocalTensor<T> softmaxMaxUb[PRE_LOAD_NUM];
    LocalTensor<T> softmaxSumUb[PRE_LOAD_NUM];
    LocalTensor<T> softmaxExpUb[PRE_LOAD_NUM];
    LocalTensor<T> softmaxMaxDefaultUb;
    LocalTensor<T> softmaxSumDefaultUb;

    static constexpr uint64_t SYNC_MODE2 = 2;
    static constexpr uint64_t SYNC_C1_V1_FLAG = 7;
    static constexpr uint64_t SYNC_V1_C2_FLAG = 8;
    static constexpr uint64_t SYNC_C2_V2_FLAG = 9;

    static constexpr uint32_t HALF_BLOCK_NUM = 16;
    static constexpr uint32_t BLOCK_ELEMENT_NUM = BYTE_BLOCK / sizeof(T);
    static constexpr uint32_t BASE_BLOCK_MAX_ELEMENT_NUM = BUFFER_SIZE_BYTE_32K / sizeof(T);
    static constexpr T SOFTMAX_MIN_NUM = -2e38;
    uint32_t msdIterNum = 0U;
    uint64_t sUnitSize = 0;

    // for workspace pingpong
    const uint32_t dbWorkspaceRatio = PRE_LOAD_NUM;

    __gm__ uint8_t *keyPtr = nullptr;
    __gm__ uint8_t *valuePtr = nullptr;

    uint32_t tmpBlockIdx = 0U;
    uint32_t aiCoreIdx = 0U;
    __gm__ uint8_t *blocktablePtr = nullptr;

    // tilingdata
    uint64_t singleProcessSInnerSize = 0U;
    uint32_t sInnerLoopTimes = 0U;
    uint64_t singleProcessSInnerSizeTail = 0U;
    uint32_t formerCoreNum = 0U;
    uint32_t usedCoreNum = 0U;
    uint32_t bIdx = 0U;
    uint32_t n2Idx = 0U;
    // split Q
    uint32_t s1Idx = 0U;
    uint32_t mmResUbSize = 0U;
    uint32_t bmm2ResUbSize = 0U;

    uint64_t batchSize = 0ULL;
    uint64_t qHeadNum = 0ULL;
    uint64_t kvHeadNum = 0ULL;
    uint64_t gSize = 0ULL;
    uint64_t gSizeVector = 0ULL;
    uint64_t gSizeCube = 0ULL;
    uint64_t gSizeStart = 0ULL;
    uint64_t kvSeqSize = 0ULL;
    uint64_t headDim = 0ULL;
    uint64_t headDimV = 0ULL;
    uint64_t headDimAlign = 0ULL;
    uint64_t headDimVAlign = 0ULL;
    uint64_t selectedBlockSize = 0;
    uint64_t selectedBlockCount = 0;

    // pageAttention
    uint32_t kvCacheBlockSize = 0;
    uint32_t maxBlockNumPerBatch = 0;

    // offset
    uint64_t tensorACoreOffset = 0ULL;
    uint64_t tensorAttenOutCoreOffset = 0ULL;
    uint64_t tensorBOffset = 0ULL;
    uint64_t valueOffset = 0ULL;
    uint64_t attenOutOffset = 0ULL;

    // splitKV
    uint32_t splitKVNum = 0U;
    uint32_t s2Idx = 0U;
    uint32_t s2IdxFD = 0U;
    uint64_t sInnerLoopSize = 0ULL;

    uint64_t curActualSeqLen = 0ULL;
    uint64_t curActualSeqLenOri = 0ULL;
    uint64_t curSingleProcessSInnerSizeAlign = 0ULL;
    uint64_t actualSingleProcessSInnerSize = 0ULL;
    uint64_t actualSingleProcessSInnerSizeAlign = 0ULL;
    uint32_t beforeBlockSplitBn2Nums = 0U;
    uint32_t bn2LoopTimes = 0U;

    uint32_t actualLenDims = 0U;
    uint32_t actualLenQDims = 0U;
    // 记录当前轮的bIdx nIdx s2Idx s1Idx actualLen
    uint32_t curBIdx = 0;
    uint32_t curN2Idx = 0;
    uint32_t curS2Idx = 0;
    uint32_t curS1Idx = 0;
    uint32_t curSInnerLoopTimes = 0;

    uint32_t bn2IdxInCurCore = 0;
    uint32_t bn2s2LoopTimes = 0;
    uint32_t numDouble = 2;
    uint32_t numEight = 8;
    ExtraInfo extraInfo[PRE_LOAD_NUM]{};

    // topk索引取数参数
    // mm1
    int64_t x = 0; // 上一轮topk索引已处理数据量
    int64_t currentTopDeal = -1; // 上一轮topk索引最大处理数据量
    int64_t topKIdOffset = 0; // 当前topk索引偏移
    int64_t idIntopKRecord = -1; // 当前topk索引偏移
    // mm2
    int64_t xMM2 = 0;
    int64_t currentTopDealMM2 = -1;
    int64_t topKIdOffsetMM2 = 0;
    int64_t idIntopKRecordMM2 = -1;
    bool isMtpFlag = false;
    uint32_t qSeqSize = 1;
    uint32_t curBatchQseqlen = 1;
    __aicore__ inline void CalcParams(uint32_t loop);
    __aicore__ inline void ComputeMm1(uint32_t loop);
    __aicore__ inline void ProcessVec1L(uint32_t loop);
    __aicore__ inline void ComputeMm2(uint32_t loop);
    __aicore__ inline void ProcessVec2L(uint32_t loop);
    __aicore__ inline void SetLoopTimes();
    __aicore__ inline void CopyInMm1AToL1(LocalTensor<KV_T>& aL1Tensor, ExtraInfo& info);
    __aicore__ inline void CopyInMm1BToL1ForPA(LocalTensor<KV_T>& bL1Tensor, uint64_t keyGmBaseOffset,
                                uint32_t copyTotalRowCnt, uint32_t copyStartRowCnt, uint32_t nActCopyRowCount);
    __aicore__ inline void LoadDataMm1A(LocalTensor<KV_T>& aL0Tensor, LocalTensor<KV_T>& aL1Tensor, uint32_t idx,
        uint32_t copyColCnt);
    __aicore__ inline void LoadDataMm1B(LocalTensor<KV_T>& bl0Tensor, LocalTensor<KV_T>& bl1Tensor, uint32_t k,
        uint32_t copyColCnt, uint32_t i, uint32_t copyRowCnt);
    __aicore__ inline void CopyInMm2AToL1(LocalTensor<KV_T>& aL1Tensor, ExtraInfo& info, uint32_t kCopyIdx,
        uint32_t kCopyRowCount, uint32_t kActCopyRowCountAlign, uint32_t kActCopyRowCount);
    __aicore__ inline void CopyInMm2BToL1ForPA(LocalTensor<KV_T>& bL1Tensor, uint64_t valueGmBaseOffset,
        uint32_t copyTotalRowCnt, uint32_t copyStartRowCnt, uint32_t kActCopyRowCount);
    __aicore__ inline void LoadDataMm2A(LocalTensor<KV_T> aL0Tensor, LocalTensor<KV_T> aL1Tensor, uint32_t kSize);

    bool curActSeqLenIsZero = false;

    template <typename T> __aicore__ inline T Align(T num, T rnd)
    {
        return (((rnd) == 0) ? 0 : (((num) + (rnd)-1) / (rnd) * (rnd)));
    }
    __aicore__ inline void InitTilingData();
    __aicore__ inline void InitCalcParamsEach();
    __aicore__ inline void InitBuffers();
    __aicore__ inline void InitActualSeqLen(__gm__ uint8_t *actualSeqLengths);
    __aicore__ inline void InitActualQSeqLen(__gm__ uint8_t *actualQSeqLengths);
    __aicore__ inline void GetActualSeqLen();
    __aicore__ inline void UpdateInnerLoopCond();
    __aicore__ inline void DealActSeqLenIsZero(uint32_t bIdx, uint32_t n2Idx);

    __aicore__ inline void GetBN2id(const uint32_t bn2Idx);

    __aicore__ inline void DealBmm1ResBaseBlock(const uint32_t loop, uint32_t startRow, uint32_t dealRowCount,
                                                uint32_t columnCount, uint32_t actualColumnCount);
    __aicore__ inline void ProcessVec1Inner(uint32_t loop);

    __aicore__ inline void DealBmm2ResBaseBlock(const uint32_t loop, uint32_t startRow, uint32_t dealRowCount,
                                                uint32_t columnCount, uint32_t actualColumnCount);
    
    __aicore__ inline void ProcessVec2Inner(uint32_t loop);

    __aicore__ inline void SoftmaxFlashV2Compute(uint32_t loop, LocalTensor<T> &mmResUb, LocalTensor<uint8_t> &softmaxTmpUb,
                                                uint32_t startRow, uint32_t dealRowCount, uint32_t columnCount,
                                                uint32_t actualColumnCount);

    __aicore__ inline void Bmm2DataCopyOut(uint64_t attenOutOffset, LocalTensor<OUT_T> &attenOutUb, uint32_t startRow, uint32_t dealRowCount,
                                        uint32_t columnCount, uint32_t actualColumnCount);
    __aicore__ inline void Bmm2CastAndCopyOut(ExtraInfo& info, LocalTensor<T> &bmm2ResUb, uint32_t startRow, uint32_t dealRowCount,
                                            uint32_t columnCount, uint32_t actualColumnCount);

    __aicore__ inline void InitAllZeroOutput(uint32_t bIdx, uint32_t n2Idx);

    __aicore__ inline void ElewiseCompute(uint32_t loop, LocalTensor<T> &mmResUb, uint32_t dealRowCount, uint32_t columnCount);
};

#include "nsa_selected_attention_infer_impl.h"