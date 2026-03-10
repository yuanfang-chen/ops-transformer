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
 * \file nsa_compress_attention_infer.h
 * \brief
 */

#ifndef NSA_COMPRESS_ATTENTION_INFER_H
#define NSA_COMPRESS_ATTENTION_INFER_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "ncai_public_define.h"
#include "common_func.h"
#include "hardware.h"
#include "common.h"
#include "iterator.h"

using namespace AscendC;
using namespace matmul;

namespace NSA_COMPRESS_ATTENTION_INFER {

#ifdef __DAV_C220_CUBE__

template <typename NCAIType>
class NsaCompressAttentionInferAic {
public:
    using Q_T = typename NCAIType::queryType;
    using KV_T = typename NCAIType::kvType;
    using OUT_T = typename NCAIType::outputType;
    static constexpr LAYOUT LAYOUT_T = NCAIType::layout;
    
    __aicore__ inline NsaCompressAttentionInferAic(){};

    __aicore__ inline void Init(GM_ADDR query, GM_ADDR key, GM_ADDR value, 
        GM_ADDR blockTable, GM_ADDR actualQSeqLen, GM_ADDR actualKvSeqLen, GM_ADDR output, GM_ADDR topkIndicesOut, 
        GM_ADDR workspace, const NsaCompressAttentionInferTilingData *__restrict tilingData);

    __aicore__ inline void Process();

protected:

    /* tiling data */
    NsaCompressAttentionInferBaseParams baseTilingData;
    NsaCompressAttentionInferSplitBNParams splitBNTilingData;
    
    /* variable */
    AscendC::GlobalTensor<Q_T> queryGm;
    AscendC::GlobalTensor<KV_T> keyGm;
    AscendC::GlobalTensor<KV_T> valueGm;
    AscendC::GlobalTensor<OUT_T> outGm;  

    GlobalTensor<int32_t> blockTableGm;
    GlobalTensor<int64_t> actualQSeqLenGm;
    GlobalTensor<int64_t> actualKvSeqLenGm;
    GlobalTensor<float> mm1ResGm;
    GlobalTensor<float> scoreInGm;
    GlobalTensor<Q_T> mm2InGm;

    AsdopsBuffer<ArchType::ASCEND_V220> buf;
    LocalTensor<Q_T> l1qBufAddrTensor;
    LocalTensor<KV_T> l1kBufAddrTensor;
    LocalTensor<KV_T> l1vBufAddrTensor;
    LocalTensor<Q_T> l1pBufAddrTensor;
    LocalTensor<Q_T> l0aBufTensor;
    LocalTensor<KV_T> l0bBufTensor;
    LocalTensor<float> l0cBufTensor;

    __gm__ uint8_t *blocktablePtr = nullptr;

    uint32_t l1qBufAddrOffset;
    uint32_t l1kBufAddrOffset;
    uint32_t l1vBufAddrOffset;
    uint32_t l1pBufAddrOffset;

    uint32_t vecBlockIdx;
    uint32_t cubeBlockIdx;
    uint32_t bn2LoopTimes;
    uint32_t beforeBlockSplitBn2Nums;

    uint32_t bIdx;

    // 基础参数
    uint32_t batchSize;
    uint32_t qSeqSize;
    uint32_t seqSize;
    uint32_t qHeadNum;
    uint32_t kvHeadNum;
    uint32_t headSizeQk;
    uint32_t headSizeVo;
    uint32_t blockSize;
    uint32_t groupSize;

    // 新增
    uint32_t maxBlockNumPerBatch;
    uint32_t workSpaceElemNum;
    uint32_t mm1ResWorkSpaceSize;
    uint32_t mm2InWorkSpaceSize;
    uint32_t scoreInWorkSpaceSize;

    uint32_t processPerBatch;
    uint32_t kvHeadSplitSize;
    uint32_t kvHeadSplitNum;
    uint32_t qSeqLenSplitSize;
    uint32_t qSeqLenOffset;
    uint32_t qSeqLenCurProcess;
    uint32_t qSeqLenCumSum;
    uint32_t kvHeadOffset;
    uint32_t curKvHeadNum;
    uint32_t processNum;
    uint32_t aicNum;

    uint32_t coreNumUsed;
    uint32_t bnPerHeadCore;
    uint32_t bnPerTailCore;

    __aicore__ inline void InitTilingData(const NsaCompressAttentionInferTilingData *__restrict tilingData);
    __aicore__ inline void CopyQToL1(const uint32_t qRowNum, const uint32_t qRowNumRound,
                                     const uint32_t headSizeQkRound, const uint32_t actualKvHeadIdx,
                                     const uint32_t l1qPingPongFlag);
    __aicore__ inline void CopyKToL1(const uint32_t kvSeqTile, const uint32_t kvSeqTileRound,
                                     const uint32_t strideK, const uint32_t kCoreOffset,
                                     const uint32_t l1kPingPongFlag);
    __aicore__ inline void CopyPVToL1(const uint32_t pRowNum, const uint32_t pRowNumRound, const int64_t curSeqlen,
                                      const uint32_t kvSeqTile, const uint32_t kvSeqTileRound, const uint32_t strideV,
                                      const uint64_t pCoreOffset, const uint32_t vCoreOffset,
                                      const uint32_t l0abPingPongFlag, const uint32_t l1pvPingPongFlag);
    __aicore__ inline void LoadQKToL0(const uint32_t qRowNum, const uint32_t qRowNumRound,
                                      const uint32_t headSizeQkRound, const uint32_t kvSeqTileRound,
                                      const uint32_t l1qPingPongFlag, const uint32_t l1kPingPongFlag,
                                      const uint32_t sIdx, const uint32_t sLoop);
    __aicore__ inline void LoadPVToL0(const uint32_t pRowNum, const uint32_t pRowNumRound,
                                      const uint32_t headSizeVoRound, const uint32_t kvSeqTileRound,
                                      const uint32_t l1pvPingPongFlag, const uint32_t l0abPingPongFlag);
    __aicore__ inline void PerformQKMmad(const uint32_t qRowNum, const uint32_t kvSeqTile,
                                         const uint32_t l0cPingPongFlag);
    __aicore__ inline void PerformPVMmad(const uint32_t pRowNum, const uint32_t kvSeqTile,
                                         const uint32_t l0abPingPongFlag, const uint32_t l0cPingPongFlag,
                                         const uint32_t sIdx);
    __aicore__ inline void CopySToWorkSpace(const uint32_t qRowNum, const uint32_t qRowNumRound,
                                            const uint32_t kvSeqTileRound, const uint32_t curSeqlenRound,
                                            const uint32_t curKvHeadIdx, const uint32_t sIdx,
                                            const uint32_t l0cPingPongFlag, const uint32_t mm1ResPingPongFlag);
    __aicore__ inline void CopyOToGm(const uint32_t pRowNum, const uint32_t pRowNumRound,
                                     const uint32_t oCoreOffset, const uint32_t l0cPingPongFlag);
    __aicore__ inline void ProcessMm1(const uint32_t processIdx);
    __aicore__ inline void ProcessMm2(const uint32_t processIdx);
    __aicore__ inline void PreProcess(const uint32_t processIdx);
};

#elif __DAV_C220_VEC__

template <typename NCAIType>
class NsaCompressAttentionInferAiv {
public:
    using Q_T = typename NCAIType::queryType;
    using KV_T = typename NCAIType::kvType;
    using OUT_T = typename NCAIType::outputType;
    static constexpr LAYOUT LAYOUT_T = NCAIType::layout;

    __aicore__ inline NsaCompressAttentionInferAiv(){};

    __aicore__ inline void Init(GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR blockTable, GM_ADDR actualQSeqLen, 
        GM_ADDR actualKvSeqLen, GM_ADDR actualSelKvSeqLen, GM_ADDR output, GM_ADDR topkIndicesOut,
        GM_ADDR workspace, const NsaCompressAttentionInferTilingData *__restrict tilingData);

    __aicore__ inline void Process();

protected:

    /* tiling data */
    NsaCompressAttentionInferBaseParams baseTilingData;
    NsaCompressAttentionInferSplitBNParams splitBNTilingData;

    /* variable */
    AscendC::GlobalTensor<int32_t> topkOutGm;

    GlobalTensor<int64_t> actualQSeqLenGm;
    GlobalTensor<int64_t> actualSelKvSeqLenGm;
    GlobalTensor<int64_t> actualKvSeqLenGm;
    GlobalTensor<float> mm1ResGm;
    GlobalTensor<float> scoreInGm;
    GlobalTensor<float> topKInGm;
    GlobalTensor<Q_T> mm2InGm;

    AsdopsBuffer<ArchType::ASCEND_V220> buf;
    __gm__ uint8_t *blocktablePtr = nullptr;

    SoftMaxTiling softmaxTilingData;
    TopkTiling topkTilingData;

    uint32_t vecBlockIdx;
    uint32_t cubeBlockIdx;
    uint32_t bIdx;

    // 基础参数
    uint32_t batchSize;
    uint32_t qSeqSize;
    uint32_t seqSize;
    uint32_t blockSize;
    uint32_t groupSize;
    uint32_t qHeadNum;
    uint32_t kvHeadNum;
    uint32_t headSizeQk;
    uint32_t headSizeVo;
    float scaleValue;
    uint32_t selectNum;

    // 新增
    uint32_t maxBlockNumPerBatch;
    uint32_t workSpaceElemNum;
    uint32_t mm1ResWorkSpaceSize;
    uint32_t mm2InWorkSpaceSize;
    uint32_t scoreInWorkSpaceSize;
    uint32_t topKInWorkSpaceSize;

    uint32_t processPerBatch;
    uint32_t kvHeadSplitSize;
    uint32_t kvHeadSplitNum;
    uint32_t qSeqLenSplitSize;
    uint32_t qSeqLenOffset;
    uint32_t qSeqLenCurProcess;
    uint32_t qSeqLenCumSum;
    uint32_t kvHeadOffset;
    uint32_t curKvHeadNum;
    uint32_t attenMaskFlag;
    uint32_t processNum;
    uint32_t aicNum;

    uint32_t coreNumUsed;
    uint32_t bnPerHeadCore;
    uint32_t bnPerTailCore;
    uint32_t rowNumVec0;      // 第1个核处理的行数
    uint32_t rowNumVec1;      // 第2个核处理的行数
    uint32_t basicRowNumSMVec0;   // 第1个核每次计算的行数
    uint32_t basicRowNumSMVec1;   // 第2个每次计算的行数
    uint32_t softmaxRowLenPerCore;
    uint32_t softmaxBasicRowLen;
    uint32_t softmaxRowLoop;
    uint32_t basicRowLenCal;
    int64_t curSeqlen;
    uint8_t softmaxRightPadding;
    uint64_t softmaxTmpVecGmOffset;
    uint64_t mm2InWorkSpaceOffset;
    uint64_t scoreInWorkSpaceOffset;
    uint32_t alignedColLen;
    uint32_t softmaxTileLength;
    uint32_t halfTileLength;
    uint32_t groupNumPerSubCore; // 每个vector核分到的N2数量
    uint32_t currentQSeqLenOffset; //当前softmaxloop属于哪一个token

    AscendC::LocalTensor<float> softmaxInputbufTensor;
    AscendC::LocalTensor<float> softmaxOut32bufTensor;
    AscendC::LocalTensor<Q_T> softmaxOut16bufTensor;
    AscendC::LocalTensor<uint8_t> sharedTmpBuffer;

    // importance score params
    uint32_t ubSize = 0;
    uint32_t baseBlockSize = 0;
    uint32_t gHeadNums = 0;
    uint32_t gHeadNumsAlign;
    uint32_t outS2 = 0;
    uint32_t maxOutS2 = 0;
    uint32_t maxBaseS2 = 0;
    uint32_t maxN = 0;
    uint32_t maxM = 0;
    uint32_t maxBaseJ = 0;
    uint32_t strideOut = 0;
    uint32_t compSizeL = 0;
    uint32_t compStrideD = 0;
    uint32_t selectSize = 0;
    uint64_t impScoreInputGmOffset;
    uint64_t impScoreOutputGmOffset;
    uint32_t impSocreCoreRowCount = 0;
    uint32_t maxRowCountPerLoop = 0;
    int32_t FLOAT_TYPE_MASK = 0x7f800000; // 0111 1111 1000 0000 0000 0000 0000 0000
    float FLOAT_MIN_MASK = -3e38;
    AscendC::LocalTensor<float> pslcTensor;
    AscendC::LocalTensor<float> pslcCalcTensor;
    AscendC::LocalTensor<float> pslcTmpTensor;
    AscendC::LocalTensor<uint8_t> pslcSharedBuffer;

    //topk params
    uint8_t topkRightPadding;
    uint32_t alignedoutS2;
    uint32_t alignedTok32;
    uint32_t alignedTokloop;
    AscendC::LocalTensor<float> topkInputbufTensor;
    AscendC::LocalTensor<float> topkoutvalueLocal;
    AscendC::LocalTensor<int32_t> topkoutindexLocal;
    AscendC::LocalTensor<int32_t> topksrcindexLocal;
    AscendC::LocalTensor<bool> topksrcLocalFinish;
    uint64_t topkInputGmOffset;
    uint64_t topkOutputGmOffset;
    AscendC::LocalTensor<uint8_t> topksharedTmpBuffer;
    AscendC::LocalTensor<int32_t> arithbuffer;

    __aicore__ inline void InitTilingData(const NsaCompressAttentionInferTilingData *__restrict tilingData);
    __aicore__ inline void PreProcessOffset(const uint32_t processIdx);
    __aicore__ inline void PreProcess(const uint32_t processIdx);
    __aicore__ inline void ProcessSoftmax(uint32_t processIdx);
    __aicore__ inline void SoftmaxComputeVecInGmOffset(uint32_t bn2Idx, uint32_t ridx);
    __aicore__ inline void SoftmaxCopyIn(DataCopyParams &splitCopyinParams,uint32_t ridx);
    __aicore__ inline void SoftmaxCompute(uint32_t ridx);
    __aicore__ inline void SoftmaxCopyOut(DataCopyParams &splitCopyoutParams, DataCopyParams &splitCopyout32Params, uint32_t ridx);
    __aicore__ inline void ComputeCurrentTokenOffset(uint32_t startIdx);
    __aicore__ inline void AddMask(uint32_t ridx, uint32_t startIdx, uint32_t endIdx);
    __aicore__ inline void MaskOperation(uint32_t ridx, uint32_t startIdx, uint32_t endIdx);
    __aicore__ inline void MaskApply(uint32_t ridx);

    // importance score
    __aicore__ inline void InitImportanceScoreParams();
    __aicore__ inline void ProcessImportanceScore(uint32_t processIdx);
    __aicore__ inline void ProcessImpScoreS2Loop(uint32_t loopCnt, uint32_t processIdx, uint32_t taskId, uint32_t startRowIdx, uint32_t endRowIdx);
    __aicore__ inline void ComputeImpScoreValue(uint32_t loopCnt, uint32_t loopOutS2, uint32_t taskId, uint32_t baseS2Id, uint32_t startRowIdx,
                                                uint32_t endRowIdx, uint32_t startJ);
    __aicore__ inline void ImpScoreDataCopyIn(int64_t offset, uint16_t blockCount, uint32_t blockLength);
    __aicore__ inline void ImpScoreDataCopyOut(uint32_t taskId, uint32_t baseS2Id, int64_t offset, uint16_t blockCount, uint32_t blockLength,
                                               uint32_t srcStride);
    __aicore__ inline void ImpScoreTransposeTensor(uint32_t height, uint32_t width);
    __aicore__ inline void ImpScoreAccumulation(uint32_t startJ, uint32_t rowCount, uint32_t offsetS2Split);
    __aicore__ inline void ImpScoreReduceSum(uint32_t startJ, uint32_t rowCount, uint32_t outNewS2);

    //topk
    __aicore__ inline void ProcessTopK();
    __aicore__ inline void TopkComputeVecInGmOffset();
    __aicore__ inline void TopkCopyIn(DataCopyParams &splitCopyintopkParams);
    __aicore__ inline void TopkCompute();
    __aicore__ inline void TopkCopyOut(DataCopyParams &splitCopyouttopkParams);
};

#endif

template <typename NCAIType>
class NsaCompressAttentionInfer {
public:

    __aicore__ inline NsaCompressAttentionInfer(){};

    __aicore__ inline void Run(GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR blockTable, GM_ADDR actualQSeqLen, 
        GM_ADDR actualKvSeqLen, GM_ADDR actualSelKvSeqLen, GM_ADDR output, GM_ADDR topkIndicesOut,
        GM_ADDR workspace, const NsaCompressAttentionInferTilingData *__restrict tilingData) {
        #ifdef __DAV_C220_CUBE__
            NsaCompressAttentionInferAic<NCAIType> opAic;
            opAic.Init(query, key, value, blockTable, actualQSeqLen, actualKvSeqLen, output, topkIndicesOut, 
                workspace, tilingData);
            opAic.Process();

        #elif __DAV_C220_VEC__
            NsaCompressAttentionInferAiv<NCAIType> opAiv;
            opAiv.Init(query, key, value, blockTable, actualQSeqLen, actualKvSeqLen, actualSelKvSeqLen, output, topkIndicesOut, 
                workspace, tilingData);
            opAiv.Process();
        #endif
    }
};

#include "nsa_compress_attention_infer_init.h"
#include "nsa_compress_attention_infer_copy.h"
#include "nsa_compress_attention_infer_compute.h"
#include "nsa_compress_attention_infer_process.h"

} // namespace NSA_COMPRESS_ATTENTION_INFER
#endif  // NSA_COMPRESS_ATTENTION_INFER_H
