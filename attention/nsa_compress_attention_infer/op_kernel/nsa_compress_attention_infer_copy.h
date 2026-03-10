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
 * \file nsa_compress_attention_infer_copy.h
 * \brief
 */

#include "nsa_compress_attention_infer.h"

#pragma once

namespace NSA_COMPRESS_ATTENTION_INFER {

#ifdef __DAV_C220_CUBE__

template <typename NCAIType>
__aicore__ inline void NsaCompressAttentionInferAic<NCAIType>::CopyQToL1(const uint32_t qRowNum,
                                                                         const uint32_t qRowNumRound,
                                                                         const uint32_t headSizeQkRound,
                                                                         const uint32_t actualKvHeadIdx,
                                                                         const uint32_t l1qPingPongFlag)
{
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1qPingPongFlag + 6);
    for (uint32_t qSeqLenCurProcessIdx = 0; qSeqLenCurProcessIdx < qSeqLenCurProcess; qSeqLenCurProcessIdx++) {
        uint64_t qCoreOffset = (qSeqLenCumSum + qSeqLenOffset + qSeqLenCurProcessIdx) * qHeadNum * headSizeQk + actualKvHeadIdx * groupSize * headSizeQk;
        if (qRowNum != 1) {
            AscendC::DataCopy(l1qBufAddrTensor[l1qPingPongFlag * BASE_L1_BLOCK_ELEM_NUM_QKV + qSeqLenCurProcessIdx * groupSize * 16],
                            queryGm[qCoreOffset],
                            AscendC::Nd2NzParams(1,           // ndNum
                                                groupSize, // nValue
                                                headSizeQk, // dValue
                                                0,           // srcNdMatrixStride, unused
                                                headSizeQk,        // srcDValue
                                                qRowNumRound,   // dstNzC0Stride
                                                1,           // dstNzNStride
                                                0));         // dstNzMatrixStride, unused
        } else {
            AscendC::DataCopy(l1qBufAddrTensor[l1qPingPongFlag * BASE_L1_BLOCK_ELEM_NUM_QKV],
                            queryGm[qCoreOffset],
                            AscendC::DataCopyParams(1,
                                                    CeilDiv(headSizeQkRound, BLOCK_SIZE), // nValue
                                                    0,           // dstNzNStride
                                                    0));         // dstNzMatrixStride, unused
        }
    }
    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(0);
}

template <typename NCAIType>
__aicore__ inline void NsaCompressAttentionInferAic<NCAIType>::CopyKToL1(const uint32_t kvSeqTile,
                                                                         const uint32_t kvSeqTileRound,
                                                                         const uint32_t strideK,
                                                                         const uint32_t kCoreOffset,
                                                                         const uint32_t l1kPingPongFlag)
{
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1kPingPongFlag);
    AscendC::DataCopy(l1kBufAddrTensor[l1kPingPongFlag * BASE_L1_BLOCK_ELEM_NUM_QKV],
                      keyGm[kCoreOffset],
                      AscendC::Nd2NzParams(1,           // ndNum
                                           kvSeqTile, // nValue
                                           headSizeQk, // dValue
                                           0,           // srcNdMatrixStride, unused
                                           strideK,        // srcDValue
                                           kvSeqTileRound,   // dstNzC0Stride
                                           1,           // dstNzNStride
                                           0));         // dstNzMatrixStride, unused
    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(1);
}

template <typename NCAIType>
__aicore__ inline void NsaCompressAttentionInferAic<NCAIType>::CopyPVToL1(const uint32_t pRowNum,
                                                                          const uint32_t pRowNumRound,
                                                                          const int64_t curSeqlen,
                                                                          const uint32_t kvSeqTile,
                                                                          const uint32_t kvSeqTileRound,
                                                                          const uint32_t strideV,
                                                                          const uint64_t pCoreOffset,
                                                                          const uint32_t vCoreOffset,
                                                                          const uint32_t l0abPingPongFlag,
                                                                          const uint32_t l1pvPingPongFlag)
{
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1pvPingPongFlag + 2);
    if (pRowNum != 1) {
        AscendC::DataCopy(
                    l1pBufAddrTensor[l1pvPingPongFlag * BASE_L1_BLOCK_ELEM_NUM_QKV],
                    mm2InGm[pCoreOffset],
                    AscendC::Nd2NzParams(1,           // ndNum
                                         pRowNum, // nValue
                                         kvSeqTile, // dValue
                                         0,           // srcNdMatrixStride, unused
                                         curSeqlen,        // srcDValue
                                         pRowNumRound,   // dstNzC0Stride
                                         1,           // dstNzNStride
                                         0));         // dstNzMatrixStride, unused
    } else {
                AscendC::DataCopy(
                    l1pBufAddrTensor[l1pvPingPongFlag * BASE_L1_BLOCK_ELEM_NUM_QKV],
                    mm2InGm[pCoreOffset],
                    AscendC::DataCopyParams(1,
                                            CeilDiv(kvSeqTileRound, BLOCK_SIZE),
                                            0,
                                            0));
    }
    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l0abPingPongFlag + 2);

    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1pvPingPongFlag + 4); //
    AscendC::DataCopy(
        l1vBufAddrTensor[l1pvPingPongFlag * BASE_L1_BLOCK_ELEM_NUM_QKV],
        valueGm[vCoreOffset],
        AscendC::Nd2NzParams(1,           // ndNum
                             kvSeqTile, // nValue
                             headSizeVo, // dValue
                             0,           // srcNdMatrixStride, unused
                             strideV,        // srcDValue
                             kvSeqTileRound,   // dstNzC0Stride
                             1,           // dstNzNStride
                             0));
    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l0abPingPongFlag + 4);
}

template <typename NCAIType>
__aicore__ inline void NsaCompressAttentionInferAic<NCAIType>::LoadQKToL0(const uint32_t qRowNum,
                                                                          const uint32_t qRowNumRound,
                                                                          const uint32_t headSizeQkRound,
                                                                          const uint32_t kvSeqTileRound,
                                                                          const uint32_t l1qPingPongFlag,
                                                                          const uint32_t l1kPingPongFlag,
                                                                          const uint32_t sIdx,
                                                                          const uint32_t sLoop)
{
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(0);
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(1);
    if (qRowNum != 1) {
        for (uint64_t l0aLoadIdx = 0; l0aLoadIdx < qRowNumRound / BLOCK_SIZE; ++l0aLoadIdx) {
            AscendC::LoadData(
                l0aBufTensor[l0aLoadIdx * headSizeQkRound * BLOCK_SIZE],
                l1qBufAddrTensor[l1qPingPongFlag * BASE_L1_BLOCK_ELEM_NUM_QKV + l0aLoadIdx * 256],
                AscendC::LoadData2dParams(0, headSizeQkRound / BLOCK_SIZE, qRowNumRound / BLOCK_SIZE,
                                          0, 0, false, 0));
        }
    } else {
        AscendC::LoadData(
            l0aBufTensor,
            l1qBufAddrTensor[l1qPingPongFlag * BASE_L1_BLOCK_ELEM_NUM_QKV],
            AscendC::LoadData2dParams(0, CeilDiv(headSizeQkRound, 256), 1,
                                      0, 0, false, 0));
    }
    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(0);
    if (sIdx == sLoop - 1) {
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1qPingPongFlag + 6);
    }

    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(1);
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(2);
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(3);
    AscendC::LoadData(
        l0bBufTensor,
        l1kBufAddrTensor[l1kPingPongFlag * BASE_L1_BLOCK_ELEM_NUM_QKV],
        AscendC::LoadData2dParams(0, headSizeQkRound * kvSeqTileRound / 256, 1,
                                  0, 0, false, 0));
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1kPingPongFlag);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(1);
}

template <typename NCAIType>
__aicore__ inline void NsaCompressAttentionInferAic<NCAIType>::LoadPVToL0(const uint32_t pRowNum,
                                                                          const uint32_t pRowNumRound,
                                                                          const uint32_t headSizeVoRound,
                                                                          const uint32_t kvSeqTileRound,
                                                                          const uint32_t l1pvPingPongFlag,
                                                                          const uint32_t l0abPingPongFlag)
{
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l0abPingPongFlag + 2);
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0abPingPongFlag);
    if (pRowNum != 1) {
        for (uint64_t l0aLoadIdx = 0; l0aLoadIdx < pRowNumRound / BLOCK_SIZE; ++l0aLoadIdx) {
            AscendC::LoadData(
                l0aBufTensor[l0abPingPongFlag * BASE_L0AB_BLOCK_ELEM_NUM + l0aLoadIdx * kvSeqTileRound * BLOCK_SIZE],
                l1pBufAddrTensor[l1pvPingPongFlag * BASE_L1_BLOCK_ELEM_NUM_QKV + l0aLoadIdx * 256],
                AscendC::LoadData2dParams(0, kvSeqTileRound / BLOCK_SIZE, pRowNumRound / BLOCK_SIZE,
                                          0, 0, false, 0));
        }
    } else {
        AscendC::LoadData(
            l0aBufTensor[l0abPingPongFlag * BASE_L0AB_BLOCK_ELEM_NUM],
            l1pBufAddrTensor[l1pvPingPongFlag * BASE_L1_BLOCK_ELEM_NUM_QKV],
            AscendC::LoadData2dParams(0, CeilDiv(kvSeqTileRound, 256), 1,
                                      0, 0, false, 0));
    }
    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0abPingPongFlag + 2);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1pvPingPongFlag + 2);

    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l0abPingPongFlag + 4);
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0abPingPongFlag + 2);
    AscendC::LoadData2dTransposeParams loadDataParams;
    loadDataParams.startIndex = 0;
    loadDataParams.dstFracGap = 0;
    loadDataParams.repeatTimes = headSizeVoRound / BLOCK_SIZE;
    loadDataParams.srcStride = kvSeqTileRound / BLOCK_SIZE;
    loadDataParams.dstGap = 0;
    for (uint32_t l0bLoadIdx = 0; l0bLoadIdx < kvSeqTileRound / BLOCK_SIZE; ++l0bLoadIdx) {
        AscendC::LoadDataWithTranspose(
            l0bBufTensor[l0abPingPongFlag * BASE_L0AB_BLOCK_ELEM_NUM + l0bLoadIdx * headSizeVoRound * BLOCK_SIZE],
            l1vBufAddrTensor[l1pvPingPongFlag * BASE_L1_BLOCK_ELEM_NUM_QKV + l0bLoadIdx * 256],
            loadDataParams);
    }
    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0abPingPongFlag + 4);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1pvPingPongFlag + 4);
}

template <typename NCAIType>
__aicore__ inline void NsaCompressAttentionInferAic<NCAIType>::CopySToWorkSpace(const uint32_t qRowNum,
                                                                                const uint32_t qRowNumRound,
                                                                                const uint32_t kvSeqTileRound,
                                                                                const uint32_t curSeqlenRound,
                                                                                const uint32_t curKvHeadIdx,
                                                                                const uint32_t sIdx,
                                                                                const uint32_t l0cPingPongFlag,
                                                                                const uint32_t mm1ResPingPongFlag)
{
    AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(l0cPingPongFlag);
    auto intriParams = AscendC::FixpipeParamsV220(
        kvSeqTileRound, // nSize
        qRowNum, // mSize
        qRowNumRound,   // srcStride
        curSeqlenRound,   // dstStride
        false);      // enRelu
    intriParams.quantPre = QuantMode_t::NoQuant;
    AscendC::Fixpipe<float, float, AscendC::CFG_ROW_MAJOR>(
        mm1ResGm[mm1ResPingPongFlag * (mm1ResWorkSpaceSize / DOUBLE_BUFFER / sizeof(float)) + // db offset
                 cubeBlockIdx * workSpaceElemNum +                    // core offset
                 curKvHeadIdx * qSeqLenCurProcess * groupSize * curSeqlenRound +          // row offset
                 sIdx * blockSize],                                   // column offset
        l0cBufTensor[l0cPingPongFlag * BASE_L0C_BLOCK_ELEM_NUM],
        intriParams);
    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0cPingPongFlag);
}

template <typename NCAIType>
__aicore__ inline void NsaCompressAttentionInferAic<NCAIType>::CopyOToGm(const uint32_t pRowNum,
                                                                         const uint32_t pRowNumRound,
                                                                         const uint32_t oCoreOffset,
                                                                         const uint32_t l0cPingPongFlag)
{
    AscendC::SetFlag<AscendC::HardEvent::M_FIX>(l0cPingPongFlag);
    AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(l0cPingPongFlag);
    auto intriParams = AscendC::FixpipeParamsV220(
        headSizeVo, // nSize
        pRowNum, // mSize
        pRowNumRound,   // srcStride
        headSizeVo,   // dstStride
        false);      // enRelu
    if (std::is_same<OUT_T, __bf16>::value) {
        intriParams.quantPre = QuantMode_t::F322BF16;
    } else {
        intriParams.quantPre = QuantMode_t::F322F16;
    }
    AscendC::Fixpipe<OUT_T, float, AscendC::CFG_ROW_MAJOR>(
        outGm[oCoreOffset], l0cBufTensor[l0cPingPongFlag * BASE_L0C_BLOCK_ELEM_NUM], intriParams);
    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0cPingPongFlag + 2);
}

#elif __DAV_C220_VEC__

template <typename NCAIType>
__aicore__ inline void NsaCompressAttentionInferAiv<NCAIType>::SoftmaxCopyIn(DataCopyParams &splitCopyinParams,uint32_t ridx)
{
    // 非对齐拷贝，不填充
    DataCopyPadParams padParams{false, 0, 0, 0};
    DataCopyPad(softmaxInputbufTensor, mm1ResGm[softmaxTmpVecGmOffset], splitCopyinParams, padParams);
}

template <typename NCAIType>
__aicore__ inline void NsaCompressAttentionInferAiv<NCAIType>::SoftmaxCopyOut(DataCopyParams &splitCopyoutParams, DataCopyParams &splitCopyout32Params, uint32_t ridx)
{   
    DataCopyPad(mm2InGm[mm2InWorkSpaceOffset], softmaxOut16bufTensor, splitCopyoutParams);

    if (ridx == 0) {
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(0); // softmax搬出等待score搬入结束
    }
    DataCopyPad(scoreInGm[scoreInWorkSpaceOffset], softmaxOut32bufTensor, splitCopyout32Params);
}

template <typename NCAIType>
__aicore__ inline void NsaCompressAttentionInferAiv<NCAIType>::ImpScoreDataCopyIn(int64_t offset, uint16_t blockCount, uint32_t blockLength) {
    uint32_t srcStride = curSeqlen * 4 - blockLength;
    DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
    uint32_t dstStride = 0;
    if ((blockLength % 64) != 0) {
        padParams.isPad = true;
        uint32_t paddingNum = (64 - (blockLength % 64)) / 4;
        if (paddingNum * 4 > 32) { // padding最大不能超过32字节
            paddingNum = paddingNum - 8;
            dstStride = 1;
        }
        padParams.rightPadding = paddingNum;
    }
    DataCopyExtParams copyParams{blockCount, blockLength, srcStride, dstStride, 0};

    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(1); // score搬入等score搬出结束，由于ub上使用了同一块地址

    DataCopyPad(pslcTensor, scoreInGm[offset], copyParams, padParams);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(1); // score计算等score搬入结束
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(1); // score计算等score搬入结束
}

template <typename NCAIType>
__aicore__ inline void NsaCompressAttentionInferAiv<NCAIType>::ImpScoreDataCopyOut(uint32_t taskId, uint32_t baseS2Id, int64_t offset, uint16_t blockCount,
                                                                                   uint32_t blockLength, uint32_t srcStride) {
    if (taskId == 0 && baseS2Id == 0) {
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(1); // score搬出等待topk搬入结束
    }
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(1); // score搬出等待score计算结束
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(1); // score搬出等待score计算结束

    uint32_t dstStride = outS2 * 4 - blockLength;

    DataCopyExtParams copyParams{blockCount, blockLength, srcStride, dstStride, 0};
    DataCopyPad(topKInGm[offset], pslcTensor, copyParams);

    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(1); // score搬入等score搬出结束，由于ub上使用了同一块地址
}

template <typename NCAIType>
__aicore__ inline void NsaCompressAttentionInferAiv<NCAIType>::TopkCopyIn(DataCopyParams &splitCopyintopkParams)
{
    PipeBarrier<PIPE_V>();
    AscendC::Duplicate<float>(topkInputbufTensor, static_cast<float>(0), BASE_TOPK_ELEM_NUM_OFFSET);
    PipeBarrier<PIPE_V>();
    DataCopyPadParams padParams{true, 0, topkRightPadding, 0};
    DataCopyPad(topkInputbufTensor, topKInGm[topkInputGmOffset], splitCopyintopkParams, padParams);
}

template <typename NCAIType>
__aicore__ inline void NsaCompressAttentionInferAiv<NCAIType>::TopkCopyOut(DataCopyParams &splitCopyouttopkParams)
{   
    DataCopyPad(topkOutGm[topkOutputGmOffset], topkoutindexLocal[0], splitCopyouttopkParams);
}

#endif
}