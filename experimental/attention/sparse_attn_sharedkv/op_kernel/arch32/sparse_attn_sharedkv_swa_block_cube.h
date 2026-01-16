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
 * \file sparse_flash_attention_service_cube_mla.h
 * \brief use 7 buffer for matmul l1, better pipeline
 */
#ifndef SPARSE_ATTN_SHAREDKV_SWA_BLOCK_CUBE_H
#define SPARSE_ATTN_SHAREDKV_SWA_BLOCK_CUBE_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "../sparse_attn_sharedkv_common.h"

template <typename SAST> class SWACubeBlock {
public:
    // 中间计算数据类型为float, 高精度模式
    using T = float;
    using Q_T = typename SAST::queryType;
    using KV_T = typename SAST::kvType;
    using OUT_T = typename SAST::outputType;
    using MM_OUT_T = T;

    __aicore__ inline SWACubeBlock(){};
    __aicore__ inline void InitParams(const ConstInfo &constInfo);
    __aicore__ inline void InitMm1GlobalTensor(GlobalTensor<Q_T> queryGm, GlobalTensor<KV_T> oriKvGm,
                                               GlobalTensor<KV_T> cmpKV, GlobalTensor<MM_OUT_T> mm1ResGm);
    __aicore__ inline void InitMm2GlobalTensor(GlobalTensor<KV_T> vec1ResGm, GlobalTensor<MM_OUT_T> mm2ResGm, 
                                               GlobalTensor<OUT_T> attentionOutGm);
    __aicore__ inline void InitBuffers(TPipe *pipe);
    __aicore__ inline void UpdateKey(GlobalTensor<KV_T> keyGm);
    __aicore__ inline void UpdateValue(GlobalTensor<KV_T> valueGm);

    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();
    __aicore__ inline void ComputeMm1(const RunInfo &info, const MSplitInfo mSplitInfo);
    __aicore__ inline void ComputeMm2(const RunInfo &info, const MSplitInfo mSplitInfo);

private:
    static constexpr bool PAGE_ATTENTION = SAST::pageAttention;
    static constexpr int TEMPLATE_MODE = SAST::templateMode;
    static constexpr bool FLASH_DECODE = SAST::flashDecode;
    static constexpr SAS_LAYOUT LAYOUT_T = SAST::layout;
    static constexpr SAS_LAYOUT KV_LAYOUT_T = SAST::kvLayout;

    static constexpr uint32_t M_SPLIT_SIZE = 128;     // m方向切分
    static constexpr uint32_t N_SPLIT_SIZE = 128;     // n方向切分
    static constexpr uint32_t N_WORKSPACE_SIZE = 512; // n方向切分

    static constexpr uint32_t L1_BLOCK_SIZE = (64 * 512 * sizeof(Q_T));
    static constexpr uint32_t L1_BLOCK_OFFSET = 64 * 512;

    static constexpr uint32_t L0A_PP_SIZE = (32 * 1024);
    static constexpr uint32_t L0B_PP_SIZE = (32 * 1024);
    static constexpr uint32_t L0C_PP_SIZE = (64 * 1024);

    // mte2 <> mte1 EventID
    // L1 3buf, 使用3个eventId
    static constexpr uint32_t L1_EVENT0 = EVENT_ID2;
    static constexpr uint32_t L1_EVENT1 = EVENT_ID3;
    static constexpr uint32_t L1_EVENT2 = EVENT_ID4;
    static constexpr uint32_t L1_EVENT3 = EVENT_ID5;
    static constexpr uint32_t L1_EVENT4 = EVENT_ID6;
    static constexpr uint32_t L1_EVENT5 = EVENT_ID7;
    static constexpr uint32_t L1_EVENT6 = EVENT_ID1;

    // m <> mte1 EventID
    static constexpr uint32_t L0AB_EVENT0 = EVENT_ID3;
    static constexpr uint32_t L0AB_EVENT1 = EVENT_ID4;

    static constexpr IsResetLoad3dConfig LOAD3DV2_CONFIG = {true, true};                    // isSetFMatrix isSetPadding;
    static constexpr uint32_t mte21QPIds[4] = {L1_EVENT0, L1_EVENT1, L1_EVENT2, L1_EVENT3}; // mte12复用
    static constexpr uint32_t mte21KVIds[3] = {L1_EVENT4, L1_EVENT5, L1_EVENT6};

    uint32_t kvCacheBlockSize = 0;
    uint32_t maxBlockNumPerBatch = 0;
    ConstInfo constInfo{};

    // L1分成3块buf, 用于记录
    uint32_t qpL1BufIter = 0;
    uint32_t kvL1BufIter = -1;
    uint32_t abL0BufIter = 0;
    uint32_t cL0BufIter = 0;

    // mm1
    GlobalTensor<Q_T> queryGm;
    GlobalTensor<Q_T> qRopeGm;
    GlobalTensor<KV_T> keyGm;
    GlobalTensor<KV_T> kRopeGm;
    GlobalTensor<MM_OUT_T> mm1ResGm;
    GlobalTensor<KV_T> kvMergeGm_;
    GlobalTensor<KV_T> oriKvGm;
    GlobalTensor<KV_T> cmpKvGm;

    // mm2
    GlobalTensor<KV_T> vec1ResGm;
    GlobalTensor<KV_T> valueGm;
    GlobalTensor<MM_OUT_T> mm2ResGm;
    GlobalTensor<OUT_T> attentionOutGm;

    // block_table
    GlobalTensor<int32_t> blockTableGm;

    TBuf<TPosition::A1> bufQPL1;
    TBuf<TPosition::A1> bufKVL1;
    TBuf<TPosition::A2> tmpBufL0A;
    TBuf<TPosition::B2> tmpBufL0B;
    TBuf<TPosition::CO1> tmpBufL0C;

    LocalTensor<Q_T> l1QPTensor;
    LocalTensor<Q_T> l1KVTensor;
    LocalTensor<KV_T> aL0TensorPingPong;
    LocalTensor<KV_T> bL0TensorPingPong;
    LocalTensor<MM_OUT_T> cL0TensorPingPong;

    // L0AB m <> mte1 EventID
    __aicore__ inline uint32_t Mte1MmABEventId(uint32_t idx)
    {
        return (L0AB_EVENT0 + idx);
    }

    __aicore__ inline uint32_t GetQPL1RealIdx(uint32_t mIdx, uint32_t k1Idx)
    {
        uint32_t idxMap[] = {0, 2}; // 确保0块和1块连在一起, 2和3块连在一起, 来保证同一m块的地址相连
        return idxMap[mIdx % 2] + k1Idx;
    }

    __aicore__ inline void CopyGmToL1(LocalTensor<KV_T> &l1Tensor, GlobalTensor<KV_T> &gmSrcTensor, uint32_t srcN,
                                      uint32_t srcD, uint32_t srcDstride);
    __aicore__ inline void CopyInMm1AToL1(LocalTensor<KV_T> &aL1Tensor, const RunInfo &info, uint32_t mSeqIdx,
                                          uint32_t mSizeAct, uint32_t headSize, uint32_t headOffset);
    __aicore__ inline void CopyInMm1ARopeToL1(LocalTensor<KV_T> &aL1Tensor, const RunInfo &info, uint32_t mSeqIdx,
                                              uint32_t mSizeAct);
    __aicore__ inline void CopyInMm1BToL1(LocalTensor<KV_T> &bL1Tensor, const uint64_t keyGmBaseOffset,
                                               uint32_t copyTotalRowCntAlign, uint32_t copyStartRowCnt,
                                               uint32_t nActCopyRowCount, uint32_t headSize);
    __aicore__ inline void CopyInMm1BRopeToL1(LocalTensor<KV_T> &bL1Tensor, const uint64_t keyGmBaseOffset,
                                                   uint32_t copyTotalRowCntAlign, uint32_t copyStartRowCnt,
                                                   uint32_t nActCopyRowCount, uint32_t headSize);
    __aicore__ inline void CopyInMm2AToL1(LocalTensor<KV_T> &aL1Tensor, const RunInfo &info, uint32_t mSeqIdx,
                                          uint32_t subMSizeAct, uint32_t nSize, uint32_t nOffset);
    __aicore__ inline void CopyInMm2BToL1(LocalTensor<KV_T> &bL1Tensor, const uint64_t valueGmBaseOffset,
                                               uint32_t copyTotalRowCntAlign, uint32_t copyStartRowCnt,
                                               uint32_t nActCopyRowCount, uint32_t copyStartColumnCount,
                                               uint32_t copyColumnCount);
    __aicore__ inline void LoadDataMm1A(LocalTensor<KV_T> &aL0Tensor, LocalTensor<KV_T> &aL1Tensor, uint32_t idx,
                                        uint32_t kSplitSize, uint32_t mSize, uint32_t kSize);
    __aicore__ inline void LoadDataMm1B(LocalTensor<KV_T> &bL0Tensor, LocalTensor<KV_T> &bL1Tensor, uint32_t idx,
                                        uint32_t kSplitSize, uint32_t kSize, uint32_t nSize);
};


template <typename SAST> __aicore__ inline void SWACubeBlock<SAST>::InitParams(const ConstInfo &constInfo)
{
    this->constInfo = constInfo;
}

template <typename SAST>
__aicore__ inline void
SWACubeBlock<SAST>::InitMm1GlobalTensor(GlobalTensor<Q_T> queryGm, GlobalTensor<KV_T> oriKvGm,
                                                   GlobalTensor<KV_T> cmpKvGm, GlobalTensor<MM_OUT_T> mm1ResGm)
{
    // mm1
    this->queryGm = queryGm;
    this->oriKvGm = oriKvGm;
    this->cmpKvGm = cmpKvGm;
    this->mm1ResGm = mm1ResGm;
}

template <typename SAST>
__aicore__ inline void
SWACubeBlock<SAST>::InitMm2GlobalTensor(GlobalTensor<KV_T> vec1ResGm, GlobalTensor<MM_OUT_T> mm2ResGm, 
                                        GlobalTensor<OUT_T> attentionOutGm)
{
    // mm2
    this->vec1ResGm = vec1ResGm;
    this->mm2ResGm = mm2ResGm;
    this->attentionOutGm = attentionOutGm;
}

template <typename SAST> __aicore__ inline void SWACubeBlock<SAST>::InitBuffers(TPipe *pipe)
{
    pipe->InitBuffer(bufQPL1, L1_BLOCK_SIZE * 4);
    l1QPTensor = bufQPL1.Get<Q_T>();
    pipe->InitBuffer(bufKVL1, L1_BLOCK_SIZE * 3);
    l1KVTensor = bufKVL1.Get<KV_T>();

    // L0A
    pipe->InitBuffer(tmpBufL0A, L0A_PP_SIZE * 2); // 64K
    aL0TensorPingPong = tmpBufL0A.Get<KV_T>();
    // L0B
    pipe->InitBuffer(tmpBufL0B, L0B_PP_SIZE * 2); // 64K
    bL0TensorPingPong = tmpBufL0B.Get<KV_T>();
    // L0C
    pipe->InitBuffer(tmpBufL0C, L0C_PP_SIZE * 2); // 128K
    cL0TensorPingPong = tmpBufL0C.Get<MM_OUT_T>();
}

template <typename SAST> __aicore__ inline void SWACubeBlock<SAST>::UpdateKey(GlobalTensor<KV_T> keyGm)
{
    this->keyGm = keyGm;
}

template <typename SAST> __aicore__ inline void SWACubeBlock<SAST>::UpdateValue(GlobalTensor<KV_T> valueGm)
{
    this->valueGm = valueGm;
}

template <typename SAST> __aicore__ inline void SWACubeBlock<SAST>::AllocEventID()
{
    SetFlag<HardEvent::MTE1_MTE2>(L1_EVENT0);
    SetFlag<HardEvent::MTE1_MTE2>(L1_EVENT1);
    SetFlag<HardEvent::MTE1_MTE2>(L1_EVENT2);
    SetFlag<HardEvent::MTE1_MTE2>(L1_EVENT3);
    SetFlag<HardEvent::MTE1_MTE2>(L1_EVENT4);
    SetFlag<HardEvent::MTE1_MTE2>(L1_EVENT5);
    SetFlag<HardEvent::MTE1_MTE2>(L1_EVENT6);
    SetFlag<HardEvent::M_MTE1>(L0AB_EVENT0);
    SetFlag<HardEvent::M_MTE1>(L0AB_EVENT1);
}

template <typename SAST> __aicore__ inline void SWACubeBlock<SAST>::FreeEventID()
{
    WaitFlag<HardEvent::MTE1_MTE2>(L1_EVENT0);
    WaitFlag<HardEvent::MTE1_MTE2>(L1_EVENT1);
    WaitFlag<HardEvent::MTE1_MTE2>(L1_EVENT2);
    WaitFlag<HardEvent::MTE1_MTE2>(L1_EVENT3);
    WaitFlag<HardEvent::MTE1_MTE2>(L1_EVENT4);
    WaitFlag<HardEvent::MTE1_MTE2>(L1_EVENT5);
    WaitFlag<HardEvent::MTE1_MTE2>(L1_EVENT6);
    WaitFlag<HardEvent::M_MTE1>(L0AB_EVENT0);
    WaitFlag<HardEvent::M_MTE1>(L0AB_EVENT1);
}

template <typename SAST>
__aicore__ inline void SWACubeBlock<SAST>::CopyGmToL1(LocalTensor<KV_T> &l1Tensor,
                                                                 GlobalTensor<KV_T> &gmSrcTensor, uint32_t srcN,
                                                                 uint32_t srcD, uint32_t srcDstride)
{
    Nd2NzParams nd2nzPara;
    nd2nzPara.ndNum = 1;
    nd2nzPara.nValue = srcN; // 行数
    nd2nzPara.dValue = srcD;
    nd2nzPara.srcDValue = srcDstride;
    nd2nzPara.dstNzC0Stride = (srcN + 15) / 16 * 16; // 对齐到16 单位block
    nd2nzPara.dstNzNStride = 1;
    nd2nzPara.srcNdMatrixStride = 0;
    nd2nzPara.dstNzMatrixStride = 0;
    DataCopy(l1Tensor, gmSrcTensor, nd2nzPara);
}

template <typename SAST>
__aicore__ inline void SWACubeBlock<SAST>::CopyInMm1AToL1(LocalTensor<KV_T> &l1Tensor, const RunInfo &info,
                                                                     uint32_t mSeqIdx, uint32_t mSizeAct,
                                                                     uint32_t headSize, uint32_t headOffset)
{
    auto srcGm = queryGm[info.tensorAOffset + mSeqIdx * constInfo.headDim + headOffset];
    CopyGmToL1(l1Tensor, srcGm, mSizeAct, headSize, constInfo.headDim);
}

template <typename SAST>
__aicore__ inline void SWACubeBlock<SAST>::CopyInMm1ARopeToL1(LocalTensor<KV_T> &l1Tensor,
                                                                         const RunInfo &info, uint32_t mSeqIdx,
                                                                         uint32_t mSizeAct)
{
    auto srcGm = qRopeGm[info.tensorARopeOffset + mSeqIdx * constInfo.headDimRope];
    CopyGmToL1(l1Tensor, srcGm, mSizeAct, constInfo.headDimRope, constInfo.headDimRope);
}

template <typename SAST>
__aicore__ inline void
SWACubeBlock<SAST>::CopyInMm1BToL1(LocalTensor<KV_T> &bL1Tensor, const uint64_t keyGmBaseOffset,
                                                   uint32_t copyTotalRowCntAlign, uint32_t copyStartRowCnt,
                                                   uint32_t nActCopyRowCount, uint32_t headSize)
{
    uint64_t dStride = constInfo.headDim;
    if constexpr (KV_LAYOUT_T == SAS_LAYOUT::BSND || KV_LAYOUT_T == SAS_LAYOUT::TND) {
        dStride = constInfo.headDim * constInfo.kvHeadNum;
    }

    uint32_t blockElementCnt = 32 / sizeof(KV_T);

    Nd2NzParams mm1Nd2NzParamsForB;
    mm1Nd2NzParamsForB.ndNum = 1;
    mm1Nd2NzParamsForB.nValue = nActCopyRowCount;
    mm1Nd2NzParamsForB.dValue = headSize;
    mm1Nd2NzParamsForB.srcDValue = dStride;
    mm1Nd2NzParamsForB.dstNzC0Stride = copyTotalRowCntAlign;
    mm1Nd2NzParamsForB.dstNzNStride = 1;
    mm1Nd2NzParamsForB.srcNdMatrixStride = 0;
    mm1Nd2NzParamsForB.dstNzMatrixStride = 0;
    DataCopy(bL1Tensor[copyStartRowCnt * blockElementCnt], keyGm[keyGmBaseOffset], mm1Nd2NzParamsForB);
}

template <typename SAST>
__aicore__ inline void
SWACubeBlock<SAST>::CopyInMm1BRopeToL1(LocalTensor<KV_T> &bL1Tensor, const uint64_t kRopeGmBaseOffset,
                                                       uint32_t copyTotalRowCntAlign, uint32_t copyStartRowCnt,
                                                       uint32_t nActCopyRowCount, uint32_t headSize)
{
    uint64_t dStride = constInfo.headDimRope;
    if constexpr (KV_LAYOUT_T == SAS_LAYOUT::BSND || KV_LAYOUT_T == SAS_LAYOUT::TND) {
        dStride = constInfo.headDimRope * constInfo.kvHeadNum;
    }

    uint32_t blockElementCnt = 32 / sizeof(KV_T);

    Nd2NzParams mm1Nd2NzParamsForB;
    mm1Nd2NzParamsForB.ndNum = 1;
    mm1Nd2NzParamsForB.nValue = nActCopyRowCount;
    mm1Nd2NzParamsForB.dValue = headSize;
    mm1Nd2NzParamsForB.srcDValue = dStride;
    mm1Nd2NzParamsForB.dstNzC0Stride = copyTotalRowCntAlign;
    mm1Nd2NzParamsForB.dstNzNStride = 1;
    mm1Nd2NzParamsForB.srcNdMatrixStride = 0;
    mm1Nd2NzParamsForB.dstNzMatrixStride = 0;
    DataCopy(bL1Tensor[copyStartRowCnt * blockElementCnt], kRopeGm[kRopeGmBaseOffset], mm1Nd2NzParamsForB);
}

template <typename SAST>
__aicore__ inline void SWACubeBlock<SAST>::LoadDataMm1A(LocalTensor<KV_T> &aL0Tensor,
                                                                   LocalTensor<KV_T> &aL1Tensor, uint32_t idx,
                                                                   uint32_t kSplitSize, uint32_t mSize, uint32_t kSize)
{
    LocalTensor<KV_T> srcTensor = aL1Tensor[mSize * kSplitSize * idx];
    LoadData3DParamsV2<KV_T> loadData3DParams;
    // SetFmatrixParams
    loadData3DParams.l1H = mSize / 16; // Hin=M1=8
    loadData3DParams.l1W = 16;         // Win=M0
    loadData3DParams.padList[0] = 0;
    loadData3DParams.padList[1] = 0;
    loadData3DParams.padList[2] = 0;
    loadData3DParams.padList[3] = 255; // 尾部数据不影响滑窗的结果

    // SetLoadToA0Params
    loadData3DParams.mExtension = mSize; // M
    loadData3DParams.kExtension = kSize; // K
    loadData3DParams.mStartPt = 0;
    loadData3DParams.kStartPt = 0;
    loadData3DParams.strideW = 1;
    loadData3DParams.strideH = 1;
    loadData3DParams.filterW = 1;
    loadData3DParams.filterSizeW = (1 >> 8) & 255;
    loadData3DParams.filterH = 1;
    loadData3DParams.filterSizeH = (1 >> 8) & 255;
    loadData3DParams.dilationFilterW = 1;
    loadData3DParams.dilationFilterH = 1;
    loadData3DParams.enTranspose = 0;
    loadData3DParams.fMatrixCtrl = 0;
    loadData3DParams.channelSize = kSize; // Cin=K
    LoadData<KV_T, LOAD3DV2_CONFIG>(aL0Tensor, srcTensor, loadData3DParams);
}

template <typename SAST>
__aicore__ inline void SWACubeBlock<SAST>::LoadDataMm1B(LocalTensor<KV_T> &l0Tensor,
                                                                   LocalTensor<KV_T> &l1Tensor, uint32_t idx,
                                                                   uint32_t kSplitSize, uint32_t kSize, uint32_t nSize)
{
    // N 方向全载
    LocalTensor<KV_T> srcTensor = l1Tensor[nSize * kSplitSize * idx];

    LoadData2DParams loadData2DParams;
    loadData2DParams.startIndex = 0;
    loadData2DParams.repeatTimes = (nSize + 15) / 16 * kSize / (32 / sizeof(KV_T));
    loadData2DParams.srcStride = 1;
    loadData2DParams.dstGap = 0;
    loadData2DParams.ifTranspose = false;
    LoadData(l0Tensor, srcTensor, loadData2DParams);
}

template <typename SAST>
__aicore__ inline void SWACubeBlock<SAST>::CopyInMm2AToL1(LocalTensor<KV_T> &aL1Tensor, const RunInfo &info,
                                                                     uint32_t mSeqIdx, uint32_t subMSizeAct,
                                                                     uint32_t nSize, uint32_t nOffset)
{
    auto srcGm = vec1ResGm[(info.loop % constInfo.preLoadNum) * constInfo.mmResUbSize +
                           mSeqIdx * info.actualSingleProcessSInnerSizeAlign + nOffset];
    CopyGmToL1(aL1Tensor, srcGm, subMSizeAct, nSize, info.actualSingleProcessSInnerSizeAlign);
}

template <typename SAST>
__aicore__ inline void SWACubeBlock<SAST>::CopyInMm2BToL1(
    LocalTensor<KV_T> &bL1Tensor, const uint64_t valueGmBaseOffset, uint32_t copyTotalRowCntAlign,
    uint32_t copyStartRowCnt, uint32_t nActCopyRowCount, uint32_t copyStartColumnCount, uint32_t copyColumnCount)
{
    uint64_t step = constInfo.headDim;
    if constexpr (KV_LAYOUT_T == SAS_LAYOUT::BSND || KV_LAYOUT_T == SAS_LAYOUT::TND) {
        step = constInfo.headDim * constInfo.kvHeadNum;
    }

    uint32_t blockElementCnt = 32 / sizeof(KV_T);

    Nd2NzParams mm1Nd2NzParamsForB;
    mm1Nd2NzParamsForB.ndNum = 1;
    mm1Nd2NzParamsForB.nValue = nActCopyRowCount;
    mm1Nd2NzParamsForB.dValue = copyColumnCount;
    mm1Nd2NzParamsForB.srcDValue = step;
    mm1Nd2NzParamsForB.dstNzC0Stride = copyTotalRowCntAlign;
    mm1Nd2NzParamsForB.dstNzNStride = 1;
    mm1Nd2NzParamsForB.srcNdMatrixStride = 0;
    mm1Nd2NzParamsForB.dstNzMatrixStride = 0;
    DataCopy(bL1Tensor[copyStartRowCnt * blockElementCnt], valueGm[valueGmBaseOffset + copyStartColumnCount],
             mm1Nd2NzParamsForB);
}


template <typename SAST>
__aicore__ inline void SWACubeBlock<SAST>::ComputeMm1(const RunInfo &info, const MSplitInfo mSplitInfo)
{
    // 最外层还需要一层m的循环
    uint32_t mSize = mSplitInfo.nBufferDealM;
    uint32_t mSizeAlign = (mSize + 16 - 1) / 16;
    uint32_t mL1Loops = (mSize + M_SPLIT_SIZE - 1) / M_SPLIT_SIZE;
    uint32_t mL1SizeAlign = M_SPLIT_SIZE; // 16对齐
    uint32_t mL1Size = M_SPLIT_SIZE;      // m的实际大小

    uint32_t nSize = BlockAlign<KV_T>(constInfo.headDim);
    uint32_t nL1Loops = (nSize + N_SPLIT_SIZE - 1) / N_SPLIT_SIZE;
    uint32_t nL1SizeAlign = N_SPLIT_SIZE; // 16对齐
    uint32_t nL1Size = N_SPLIT_SIZE;      // n的实际大小

    uint32_t kSize = info.actualSingleProcessSInnerSize;
    uint32_t kL1Size = 256;
    uint32_t kL1SizeAlign = SASAlign(kL1Size, 16U);
    uint32_t kL1Loops = (kSize + kL1Size - 1) / kL1Size;
    uint32_t kL0Size = 128;
    uint32_t kL0Loops = (kL1Size + kL0Size - 1) / kL0Size;
    uint32_t kL0SizeAlign = kL0Size;
    LocalTensor<KV_T> bL1Tensor;
    LocalTensor<KV_T> subvTensor;

    uint32_t ka = 0, kb = 0;
    uint32_t mBaseIdx = qpL1BufIter;
    for (uint32_t nL1 = 0; nL1 < nL1Loops; nL1++) { // n切L1

        for (uint32_t k1 = 0; k1 < kL1Loops; k1++) { // k切L1, 这里套了一层l0来操作
            for (uint32_t mL1 = 0; mL1 < mL1Loops; mL1++) {
                if (k1 == (kL1Loops - 1)) {
                    if (nL1 == 0 && mL1 == 0) { // 第一次Fixpipe前等待
                        CrossCoreWaitFlag(constInfo.syncV1NupdateC2);
                    }
                }
            }
        }
        // cL0BufIter已经不在使用
        if (mL1Loops == 1) {
            cL0BufIter++;
        }
    }
    qpL1BufIter += mL1Loops;
    printf("process mm1 \n");
}


template <typename SAST>
__aicore__ inline void SWACubeBlock<SAST>::ComputeMm2(const RunInfo &info, const MSplitInfo mSplitInfo)
{
    printf("process mm2 \n");
}

#endif