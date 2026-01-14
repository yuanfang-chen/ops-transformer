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
 * \file compressor_block_cube.h
 * \brief
 */

#ifndef COMPRESSOR_BLOCK_CUBE_H
#define COMPRESSOR_BLOCK_CUBE_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "compressor_comm.h"

using namespace AscendC;


namespace Compressor {


struct BatchInfo {
    int32_t batchId;
    int32_t cuSeqLens;
    int32_t seqUsed;
    int32_t startPos;
};

template<typename COMP> class CompressorBlockCube {
public:
    static constexpr bool X_DTYPE = COMP::xDtype == X_DTYPE::BF16;
    using X_T = typename AscendC::Conditional<X_DTYPE, bfloat16_t, half>::type;

    __aicore__ inline CompressorBlockCube(){};
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
    __aicore__ inline void InitBuffers(TPipe *pipe_);
    __aicore__ inline void ComputeMm1(const RunInfo &info);

    __aicore__ inline uint32_t GetSeqLength(uint32_t index);
    __aicore__ inline uint32_t GetStartPos(uint32_t index);
    __aicore__ inline void GetBatchInfo(BatchInfo &batchInfo, uint32_t batchId);
    __aicore__ inline void LoadDataToL0A(size_t l0Offset, size_t l1Offset);
    __aicore__ inline void LoadDataToL0B(size_t l0Offset, size_t l1Offset);
    __aicore__ inline void CopyNDGMToL1X(size_t startT, size_t sizeT, size_t rOffset,size_t mL1, 
        size_t kL1, size_t l1Offset);
    __aicore__ inline void CopyNDGmToL1W(size_t startD, size_t coffOffset, size_t kL1, size_t l1Offset,
        const GlobalTensor<X_T> &gmTensor);
    __aicore__ inline void LeftRight(size_t kL1, size_t coffOffset, size_t mOffset);
    __aicore__ inline void OnlyRight(size_t kL1, size_t wOffset, const GlobalTensor<X_T> &wGm);
    __aicore__ inline void CopyOutToUB();
    __aicore__ inline void FreeBuffers(TPipe *pipe_);
    __aicore__ inline void InnerLoop(size_t kL1, size_t mOffset, size_t l0COffset);

private:
    uint32_t cmpRatio_ = 0U;
    uint32_t coff_ = 0U;
    uint32_t curStartPos_ = 0;
    uint32_t preStartPosIdx_ = 0;
    uint32_t accSeqLength_ = 0;
    uint32_t curActSeqLength_ = 0;
    uint32_t preActSeqIdx_ = 0;
    uint32_t curActSeqLength = 0; // 当前B在整个T中的长度
    ConstInfo constInfo_ = {};

    // mte2 <> mte1
    int32_t mte2ToMte1A[2];
    int32_t mte2ToMte1B[2];
    int32_t mte1ToMte2A[2];
    int32_t mte1ToMte2B[2];

    // m <> mte1
    int32_t mToMte1A[2];
    int32_t mToMte1B[2];
    int32_t mte1ToMA[2];
    int32_t mte1ToMB[2];

    // fix <> m
    int32_t fixToM; // L0C占满，无法double buffer
    int32_t mToFix; // L0C占满，无法double buffer

    // GM
    GlobalTensor<X_T> xGm_;
    GlobalTensor<X_T> wkvGm_;
    GlobalTensor<X_T> wgateGm_;
    GlobalTensor<int32_t> cuSeqlensGm_;
    GlobalTensor<int32_t> sequsedGm_;
    GlobalTensor<int32_t> startPosGm_;

    // L1
    TBuf<TPosition::A1> aBufL1_;
    TBuf<TPosition::B1> bBufL1_;
    TBuf<TPosition::A1> xBufL1_;

    // L0
    TBuf<TPosition::A2> aBufL0_;
    TBuf<TPosition::B2> bBufL0_;
    TBuf<TPosition::CO1> cBufL0_;

    TBuf<TPosition::VECIN> ubBuf_;

    LocalTensor<X_T> aL1Tensor_;
    LocalTensor<X_T> bL1Tensor_;
    LocalTensor<X_T> aL0Tensor_;
    LocalTensor<X_T> bL0Tensor_;
    LocalTensor<float> cL0Tensor_;
    LocalTensor<float> ubTensor_;

    int32_t l1AIter = 0;
    int32_t l0AIter = 0;
    int32_t l1BIter = 0;
    int32_t l0BIter = 0;
    int32_t l0CIter = 0;

    // 定义L1所要使用的大小
    // L1A_PP_SIZE = 128 * 3 (T序列长度) 128 (baseK) 2 (bf16)
    static constexpr size_t L1A_PP_SIZE = (128 * 3) * 128 * 2; // 96KB
    // L1B_PP_SIZE = 128 (baseD kv+state) 128 (baseK) 2 (bf16)
    static constexpr size_t L1B_PP_SIZE = 128 * 128 * 2; // 32KB

    // 定义L0所使用的大小
    static constexpr size_t L0A_PP_SIZE = (32 * 1024); // 128 * 128 * sizeof(bf16)
    static constexpr size_t L0B_PP_SIZE = (32 * 1024); // 128 * 128 * sizeof(bf16)
    static constexpr size_t L0C_PP_SIZE = (128 * 1024); // 128 * 256(left_kv right_gate right_kv right_gate) * sizeof(float)
};

template <typename COMP>
__aicore__ inline void CompressorBlockCube<COMP>::InitParams(const ConstInfo &constInfo)
{
    this->constInfo_ = constInfo;
}

template <typename COMP> __aicore__ inline void CompressorBlockCube<COMP>::Init(
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
    xGm_.SetGlobalBuffer((__gm__ X_T *)x);
    wkvGm_.SetGlobalBuffer((__gm__ X_T *)wKv);
    wgateGm_.SetGlobalBuffer((__gm__ X_T *)wGate);
    cuSeqlensGm_.SetGlobalBuffer((__gm__ int32_t *)cuSeqlens);
    sequsedGm_.SetGlobalBuffer((__gm__ int32_t *)seqUsed);
    startPosGm_.SetGlobalBuffer((__gm__ int32_t *)startPos);
}

template <typename COMP>
__aicore__ inline void CompressorBlockCube<COMP>::InitBuffers(TPipe *pipe_)
{
    // MTE1<->MTE2
    mte1ToMte2A[0] = pipe_->AllocEventID<HardEvent::MTE1_MTE2>();
    mte1ToMte2A[1] = pipe_->AllocEventID<HardEvent::MTE1_MTE2>();
    mte1ToMte2B[0] = pipe_->AllocEventID<HardEvent::MTE1_MTE2>();
    mte1ToMte2B[1] = pipe_->AllocEventID<HardEvent::MTE1_MTE2>();
    SetFlag<HardEvent::MTE1_MTE2>(mte1ToMte2A[0]);
    SetFlag<HardEvent::MTE1_MTE2>(mte1ToMte2A[1]);
    SetFlag<HardEvent::MTE1_MTE2>(mte1ToMte2B[0]);
    SetFlag<HardEvent::MTE1_MTE2>(mte1ToMte2B[1]);

    // MTE2<->MTE1
    mte2ToMte1A[0] = pipe_->AllocEventID<HardEvent::MTE2_MTE1>();
    mte2ToMte1A[1] = pipe_->AllocEventID<HardEvent::MTE2_MTE1>();
    mte2ToMte1B[0] = pipe_->AllocEventID<HardEvent::MTE2_MTE1>();
    mte2ToMte1B[1] = pipe_->AllocEventID<HardEvent::MTE2_MTE1>();

    // M<->MTE1
    mToMte1A[0] = pipe_->AllocEventID<HardEvent::M_MTE1>();
    mToMte1A[1] = pipe_->AllocEventID<HardEvent::M_MTE1>();
    mToMte1B[0] = pipe_->AllocEventID<HardEvent::M_MTE1>();
    mToMte1B[1] = pipe_->AllocEventID<HardEvent::M_MTE1>();
    SetFlag<HardEvent::M_MTE1>(mToMte1A[0]);
    SetFlag<HardEvent::M_MTE1>(mToMte1A[1]);
    SetFlag<HardEvent::M_MTE1>(mToMte1B[0]);
    SetFlag<HardEvent::M_MTE1>(mToMte1B[1]);

    // MTE1<->M
    mte1ToMA[0] = pipe_->AllocEventID<HardEvent::MTE1_M>();
    mte1ToMA[1] = pipe_->AllocEventID<HardEvent::MTE1_M>();
    mte1ToMB[0] = pipe_->AllocEventID<HardEvent::MTE1_M>();
    mte1ToMB[1] = pipe_->AllocEventID<HardEvent::MTE1_M>();

    // FIX<->M
    fixToM = pipe_->AllocEventID<HardEvent::FIX_M>();
    SetFlag<HardEvent::FIX_M>(fixToM);

    // M<->FIX
    mToFix = pipe_->AllocEventID<HardEvent::M_FIX>();

    // L1
    pipe_->InitBuffer(aBufL1_, L1A_PP_SIZE * 2); // 192KB
    pipe_->InitBuffer(bBufL1_, L1B_PP_SIZE * 2); // 64KB
    aL1Tensor_ = aBufL1_.Get<X_T>();
    bL1Tensor_ = bBufL1_.Get<X_T>();

    // L0 
    pipe_->InitBuffer(aBufL0_, L0A_PP_SIZE * 2); // 64KB
    pipe_->InitBuffer(bBufL0_, L0B_PP_SIZE * 2); // 64KB
    pipe_->InitBuffer(cBufL0_, L0C_PP_SIZE * 2); // 128KB
    aL0Tensor_ = aBufL0_.Get<X_T>();
    bL0Tensor_ = bBufL0_.Get<X_T>();
    cL0Tensor_ = cBufL0_.Get<float>();

    // UB
    pipe_->InitBuffer(ubBuf_, L0C_PP_SIZE);
    ubTensor_ = ubBuf_.Get<float>();
}

template <typename COMP>
__aicore__ inline void CompressorBlockCube<COMP>::FreeBuffers(TPipe *pipe_)
{
    WaitFlag<HardEvent::MTE1_MTE2>(mte1ToMte2A[0]);
    WaitFlag<HardEvent::MTE1_MTE2>(mte1ToMte2A[1]);
    WaitFlag<HardEvent::MTE1_MTE2>(mte1ToMte2B[0]);
    WaitFlag<HardEvent::MTE1_MTE2>(mte1ToMte2B[1]);
    pipe_->ReleaseEventID<HardEvent::MTE1_MTE2>(mte1ToMte2A[0]);
    pipe_->ReleaseEventID<HardEvent::MTE1_MTE2>(mte1ToMte2A[1]);
    pipe_->ReleaseEventID<HardEvent::MTE1_MTE2>(mte1ToMte2B[0]);
    pipe_->ReleaseEventID<HardEvent::MTE1_MTE2>(mte1ToMte2B[1]);

    pipe_->ReleaseEventID<HardEvent::MTE2_MTE1>(mte2ToMte1A[0]);
    pipe_->ReleaseEventID<HardEvent::MTE2_MTE1>(mte2ToMte1A[1]);
    pipe_->ReleaseEventID<HardEvent::MTE2_MTE1>(mte2ToMte1B[0]);
    pipe_->ReleaseEventID<HardEvent::MTE2_MTE1>(mte2ToMte1B[1]);

    WaitFlag<HardEvent::M_MTE1>(mToMte1A[0]);
    WaitFlag<HardEvent::M_MTE1>(mToMte1A[1]);
    WaitFlag<HardEvent::M_MTE1>(mToMte1B[0]);
    WaitFlag<HardEvent::M_MTE1>(mToMte1B[1]);
    pipe_->ReleaseEventID<HardEvent::M_MTE1>(mToMte1A[0]);
    pipe_->ReleaseEventID<HardEvent::M_MTE1>(mToMte1A[1]);
    pipe_->ReleaseEventID<HardEvent::M_MTE1>(mToMte1B[0]);
    pipe_->ReleaseEventID<HardEvent::M_MTE1>(mToMte1B[1]);

    pipe_->ReleaseEventID<HardEvent::MTE1_M>(mte1ToMA[0]);
    pipe_->ReleaseEventID<HardEvent::MTE1_M>(mte1ToMA[1]);
    pipe_->ReleaseEventID<HardEvent::MTE1_M>(mte1ToMB[0]);
    pipe_->ReleaseEventID<HardEvent::MTE1_M>(mte1ToMB[1]);

    SetFlag<HardEvent::FIX_M>(fixToM);
    pipe_->ReleaseEventID<HardEvent::FIX_M>(fixToM);

    pipe_->ReleaseEventID<HardEvent::M_FIX>(mToFix);
}

template <typename COMP>
__aicore__ inline uint32_t CompressorBlockCube<COMP>::GetStartPos(uint32_t index)
{
    return startPosGm_.GetValue(index);
}

template <typename COMP>
__aicore__ inline uint32_t CompressorBlockCube<COMP>::GetSeqLength(uint32_t index)
{
    if (COMP::xLayout == X_LAYOUT::TH) {
        return cuSeqlensGm_.GetValue(index + 1) - cuSeqlensGm_.GetValue(index);
    } else {
        return constInfo_.sSize;
    }
}

template <typename COMP>
__aicore__ inline void CompressorBlockCube<COMP>::GetBatchInfo(BatchInfo &batchInfo, uint32_t batchId)
{
    batchInfo.batchId = batchId;
    if constexpr (COMP::xLayout == X_LAYOUT::TH) {
        batchInfo.cuSeqLens = cuSeqlensGm_.GetValue(batchId);
        batchInfo.seqUsed = GetSeqLength(batchId);
        batchInfo.startPos = GetStartPos(batchId);
    // } else {
    //     batchInfo.cuSeqlens = constInfo_.batchSize * batchId;
    //     batchInfo.seqUsed = seqUsedGm_.GetValue(batchId);
    //     batchInfo.startPos = startPosGm_.GetValue(batchId);
    }
}

#if __CCE_AICORE__ == 310
template <typename COMP>
__aicore__ inline void CompressorBlockCube<COMP>::LoadDataToL0A(size_t l0Offset, size_t l1Offset)
{
    // NZ2ZZ
    LoadData2DParamsV2 loadData2DParamsA;
    // 以M*K矩阵为例，源矩阵M轴方向的起始位置，单位为16 element
    loadData2DParamsA.mStartPosition = 0; // 算到 l1Offset 中
    // 以M*K矩阵为例，源矩阵K轴方向的起始位置，单位为32B
    loadData2DParamsA.kStartPosition = 0;
    // 是否启用转置功能，对每个分型矩阵进行转置
    loadData2DParamsA.ifTranspose = false;
    // 以M*K矩阵为例,源矩阵M轴方向搬运长度(S1向上对齐分形(512B),16*16个f16->向上对齐16)，单位为16 element,取值范围：mStep属于[0,255]
    loadData2DParamsA.mStep = 128 / 16; // mL0Base = 128
    // 以M*K矩阵为例,源矩阵K轴方向搬运长度(qkD个f16)，单位为32B,取值范围：nStep属于[0,255]
    loadData2DParamsA.kStep = 128 * sizeof(X_T) / 32;
    loadData2DParamsA.srcStride = ((constInfo_.cmpRatio + 256) + 15) / 16;
    loadData2DParamsA.dstStride = 1;
    LoadData(aL0Tensor_[l0Offset], aL1Tensor_[l1Offset], loadData2DParamsA);
}

// L1->L0B + 切k/切M/全载
template <typename COMP>
__aicore__ inline void CompressorBlockCube<COMP>::LoadDataToL0B(size_t l0Offset, size_t l1Offset)
{
    const size_t baseD = constInfo_.dBaseSize;
    constexpr size_t baseK = 128;

    // NZ2ZN(带转置)
    LoadData2DParamsV2 loadData2DParamsB;
    // 以M*K矩阵为例，源矩阵M轴方向的起始位置，单位为16 element
    loadData2DParamsB.mStartPosition = 0;
    // 以M*K矩阵为例，源矩阵K轴方向的起始位置，单位为32B
    loadData2DParamsB.kStartPosition = 0;
    // 是否启用转置功能，对每个分型矩阵进行转置
    loadData2DParamsB.ifTranspose = true;
    // 以M*K矩阵为例,源矩阵M轴方向搬运长度(S1向上对齐分形(512B),16*16个f16->向上对齐16)，单位为16 element,取值范围：mStep属于[0,255]
    loadData2DParamsB.mStep = baseD / 16;
    // 以M*K矩阵为例,源矩阵K轴方向搬运长度(qkD个f16)，单位为32B,取值范围：nStep属于[0,255]
    loadData2DParamsB.kStep = baseK * sizeof(X_T) / 32;
    // 以M*K矩阵为例，源矩阵K方向前一个分形起始地址与后一个分形起始地址的间隔，单位：512B
    loadData2DParamsB.srcStride = baseD / 16;
    // 以M*K矩阵为例，目标矩阵K方向前一个分形起始地址与后一个分形起始地址的间隔，单位：512B
    loadData2DParamsB.dstStride = 1;
    LoadData(bL0Tensor_[l0Offset], bL1Tensor_[l1Offset], loadData2DParamsB);
}

#elif
template <typename COMP>
__aicore__ inline void CompressorBlockCube<COMP>::LoadDataToL0A(size_t l0Offset, size_t l1Offset)
{
    constexpr IsResetLoad3dConfig LOAD3DV2_CONFIG = {true, true};
    constexpr uint32_t LOAD3D_L1W_SIZE = 16;
    LoadData3DParamsV2<X_T> loadData3DParams;
    loadData3DParams.l1H = 128 / LOAD3D_L1W_SIZE; // 源操作数height
    loadData3DParams.l1W = LOAD3D_L1W_SIZE; // 源操作数weight
    loadData3DParams.padList[0] = 0;
    loadData3DParams.padList[1] = 0;
    loadData3DParams.padList[2] = 0;
    loadData3DParams.padList[3] = 255; // 尾部数据不影响滑窗的结果

    loadData3DParams.mExtension = ((constInfo_.cmpRatio + 256) + 15) / 16 * 16; // 在目的操作数height维度的传输长度
    loadData3DParams.kExtension = 128; // 在目的操作数width维度的传输长度 (kBase)
    loadData3DParams.mStartPt = 0; // 卷积核在目的操作数width维度的起点
    loadData3DParams.kStartPt = 0; // 卷积核在目的操作数height维度的起点
    loadData3DParams.strideW = 1; // 卷积核在源操作数width维度滑动的步长
    loadData3DParams.strideH = 1; // 卷积核在源操作数height维度滑动的步长
    loadData3DParams.filterW = 1; // 卷积核width
    loadData3DParams.filterSizeW = false; // 是否在filterW的基础上将卷积核width增加256个元素
    loadData3DParams.filterH = 1; // 卷积核height
    loadData3DParams.filterSizeH = false; // 是否在filterH的基础上将卷积核height增加256个元素
    loadData3DParams.dilationFilterW = 1; // 卷积核width膨胀系数
    loadData3DParams.dilationFilterH = 1; // 卷积核height膨胀系数
    loadData3DParams.enTranspose = 0; // 是否启用转置功能，对整个目标矩阵进行转置
    loadData3DParams.fMatrixCtrl = 0;
    loadData3DParams.channelSize = 128; // 源操作数的通道数。膨胀系数为1时，目的weight为filterW*filterH*channelSize
    LoadData<X_T, LOAD3DV2_CONFIG>(aL0Tensor_[l0Offset], aL1Tensor_[l1Offset], loadData3DParams);
}

template <typename COMP>
__aicore__ inline void CompressorBlockCube<COMP>::LoadDataToL0B(size_t l0Offset, size_t l1Offset)
{
    // NZ2ZN(带转置)
    LoadData2DParams loadData2DParams;
    loadData2DParams.startIndex = 0;
    loadData2DParams.repeatTimes = 128 * 128 * sizeof(X_T) / 512;
    loadData2DParams.srcStride = 1;
    loadData2DParams.dstGap = 0;
    loadData2DParams.ifTranspose = true;
    LoadData(bL0Tensor_[l0Offset], bL1Tensor_[l1Offset], loadData2DParams);
}
#endif

template <typename COMP>
__aicore__ inline void CompressorBlockCube<COMP>::CopyNDGMToL1X(size_t startT, size_t sizeT, size_t rOffset,
    size_t mL1, size_t kL1, size_t l1Offset)
{
    size_t baseK = 128;
    // l1Offset 传入指示是哪个buffer
    constexpr size_t nzBaseWidth = 32 / sizeof(X_T);
    l1Offset += (mL1 + rOffset) * nzBaseWidth; // mL1 为压缩块开始，rOffset是压缩块中数据的起始相对位置
    size_t gmOffset = startT * constInfo_.hSize + kL1;
    Nd2NzParams nd2nzPara;
    nd2nzPara.ndNum = 1;
    nd2nzPara.nValue = sizeT;
    nd2nzPara.dValue = baseK; // 一次 只能做 baseK = 128
    nd2nzPara.srcDValue = constInfo_.hSize;
    // cmpRatio == 2 / 4 / 8 的时候需要对齐 16，否则L1->L0拷贝 LoadData2D 不好表示
    nd2nzPara.dstNzC0Stride = ((constInfo_.cmpRatio + 256) + 15) / 16 * 16 - 1;
    nd2nzPara.dstNzNStride = 1;
    nd2nzPara.srcNdMatrixStride = 0;
    nd2nzPara.dstNzMatrixStride = 0;
    DataCopy(aL1Tensor_[l1Offset], xGm_[gmOffset], nd2nzPara);
}

template <typename COMP>
__aicore__ inline void CompressorBlockCube<COMP>::CopyNDGmToL1W(size_t startD, size_t coffOffset, size_t kL1, size_t l1Offset,
    const GlobalTensor<X_T> &gmTensor)
{
    size_t baseK = 128;
    // l1Offset 传入指示是哪个buffer
    constexpr size_t nzBaseWidth = 32 / sizeof(X_T);
    l1Offset += (startD + coffOffset * constInfo_.dBaseSize) * nzBaseWidth; // mL1 为压缩块开始，rOffset是压缩块中数据的起始相对位置
    size_t gmOffset = (startD + coffOffset * constInfo_.dBaseSize) * constInfo_.hSize + kL1;
    Nd2NzParams nd2nzPara;
    nd2nzPara.ndNum = 1;
    nd2nzPara.nValue = constInfo_.dBaseSize;
    nd2nzPara.dValue = baseK;
    nd2nzPara.srcDValue = constInfo_.hSize;
    nd2nzPara.dstNzC0Stride = (constInfo_.dBaseSize + 15) / 16 * 16  - 1;
    nd2nzPara.dstNzNStride = 1;
    nd2nzPara.srcNdMatrixStride = 0;
    nd2nzPara.dstNzMatrixStride = 0;
    DataCopy(bL1Tensor_[l1Offset], wkvGm_[gmOffset], nd2nzPara);
}

template <typename COMP>
__aicore__ inline void CompressorBlockCube<COMP>::InnerLoop(size_t kL1, size_t mOffset, size_t l0COffset)
{
    size_t l1BOffset = L1B_PP_SIZE / sizeof(X_T) * (l1BIter & 1);
    size_t l0BOffset = L0B_PP_SIZE / sizeof(X_T) * (l0BIter & 1);

    WaitFlag<HardEvent::MTE2_MTE1>(mte2ToMte1B[l1BIter & 1]);
    WaitFlag<HardEvent::M_MTE1>(mToMte1B[l0BIter & 1]);
    LoadDataToL0B(l0BOffset, l1BOffset);
    // MTE1B拷贝完，释放L1B
    SetFlag<HardEvent::MTE1_MTE2>(mte1ToMte2B[l1BIter & 1]);
    l1BIter++;

    SetFlag<HardEvent::MTE1_M>(mte1ToMB[l0BIter & 1]);
    WaitFlag<HardEvent::MTE1_M>(mte1ToMB[l0BIter & 1]);

    const size_t mL1Base = 128;
    const size_t mL1Split = 256;

    const size_t nL1Base = 128;
    const size_t kL1Base = 128;

    for (size_t mL1 = 0; mL1 < mL1Split; mL1 += mL1Base) {
        size_t l1AOffsetPP = L1A_PP_SIZE / sizeof(X_T) * (l1AIter & 1);
        size_t l0AOffsetPP = L0A_PP_SIZE / sizeof(X_T) * (l0AIter & 1);
        size_t l1AOffset = l1AOffsetPP + 32 / sizeof(X_T) * (mL1 + mOffset);
        size_t l0AOffset = l0AOffsetPP;
        WaitFlag<HardEvent::M_MTE1>(mToMte1A[l0AIter & 1]);
        LoadDataToL0A(l0AOffset, l1AOffset);
        SetFlag<HardEvent::MTE1_M>(mte1ToMA[l0AIter & 1]);
        WaitFlag<HardEvent::MTE1_M>(mte1ToMA[l0AIter & 1]);
        MmadParams mmadParams;
        mmadParams.m = mL1Base;
        mmadParams.n = nL1Base;
        mmadParams.k = kL1Base;
        mmadParams.cmatrixInitVal = (kL1 == 0);
        mmadParams.cmatrixSource = false;
        mmadParams.unitFlag = 0;
        size_t l0COffsetInner = L0C_PP_SIZE / 2 / sizeof(float) * (l0CIter & 1);
        Mmad(cL0Tensor_[l0COffset + l0COffsetInner], aL0Tensor_[l0AOffset], bL0Tensor_[l0BOffset], mmadParams);
        SetFlag<HardEvent::M_MTE1>(mToMte1A[l0AIter & 1]);
        l0CIter++; // 前后 128 刚好错开
        l0AIter++;
    }
        // MMAD执行完成，释放L0B
    SetFlag<HardEvent::M_MTE1>(mToMte1B[l0BIter & 1]);
    l0BIter++;
}

template <typename COMP>
__aicore__ inline void CompressorBlockCube<COMP>::LeftRight(size_t kL1, size_t coffOffset, size_t mOffset)
{
    WaitFlag<HardEvent::MTE1_MTE2>(mte1ToMte2B[l1BIter & 1]);
    size_t l1BOffset = L1B_PP_SIZE / sizeof(X_T) * (l1BIter & 1);
    size_t kvSize = L1B_PP_SIZE / sizeof(X_T) / 2;
    const size_t startD = constInfo_.dIdx * constInfo_.dBaseSize;
    CopyNDGmToL1W(startD, coffOffset, kL1, l1BOffset, wkvGm_);
    CopyNDGmToL1W(startD, coffOffset, kL1, l1BOffset + kvSize, wgateGm_);
    SetFlag<HardEvent::MTE2_MTE1>(mte2ToMte1B[l1BIter & 1]);

    size_t l0COffset = L0C_PP_SIZE / sizeof(float) * coffOffset;
    InnerLoop(kL1, mOffset, l0COffset);
}

template <typename COMP>
__aicore__ inline void CompressorBlockCube<COMP>::OnlyRight(size_t kL1, size_t wOffset, const GlobalTensor<X_T> &wGm)
{
    WaitFlag<HardEvent::MTE1_MTE2>(mte1ToMte2B[l1BIter & 1]);
    size_t l1BOffset = L1B_PP_SIZE / sizeof(X_T) * (l1BIter & 1);
    const size_t startD = constInfo_.dIdx * constInfo_.dBaseSize;
    CopyNDGmToL1W(startD, 0, kL1, l1BOffset, wGm);
    SetFlag<HardEvent::MTE2_MTE1>(mte2ToMte1B[l1BIter & 1]);

    size_t l0COffset = L0C_PP_SIZE / sizeof(float) * wOffset;
    InnerLoop(kL1, 0, l0COffset);
}

#if __CCE_AICORE__ == 310
template <typename COMP>
__aicore__ inline void CompressorBlockCube<COMP>::CopyOutToUB()
{
    static constexpr FixpipeConfig config = {CO2Layout::ROW_MAJOR, true};
    FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;
    size_t nSize = constInfo_.dBaseSize;
    if constexpr (COMP::coff == COFF::OVERLAP) {
        nSize = constInfo_.dBaseSize * 2;
    }
    fixpipeParams.nSize = nSize * 2;
    fixpipeParams.mSize = constInfo_.mBaseSize / 2;
    fixpipeParams.srcStride = constInfo_.mBaseSize / 2; 
    fixpipeParams.dstStride = nSize * 2;
    fixpipeParams.dualDstCtl = 2; // 切 N 的双发
    fixpipeParams.params.ndNum = 1;
    fixpipeParams.params.srcNdStride = 0;
    fixpipeParams.params.dstNdStride = 0;

    // coff=1 vec0:front_kv     vec1:back_kv
    // coff=2 vec0:front_left   vec1:back_left
    Fixpipe<float, float, config>(ubTensor_, cL0Tensor_, fixpipeParams);

    // coff=1 vec0:front_gate   vec1:back_gate
    // coff=2 vec0:front_right  vec1:back_right
    const size_t ubOffset = L0C_PP_SIZE / sizeof(float) / 2;
    const size_t cL0Offset = L0C_PP_SIZE / sizeof(float);
    Fixpipe<float, float, config>(ubTensor_[ubOffset], cL0Tensor_[cL0Offset], fixpipeParams);
}
#elif
template <typename COMP>
__aicore__ inline void CompressorBlockCube<COMP>::CopyOutToGm(const GlobalTensor<float> &workspace,
    const LocalTensor<float> &local)
{
    FixpipeParamsV220 fixParams;
    fixParams.mSize = 128;
    fixParams.nSize = 128;
    fixParams.srcStride = 16;
    fixParams.dstStride = 128;
    fixParams.ndNum = 1;
    Fixpipe(workspace, local, fixParams);
}
#endif

template <typename COMP> __aicore__ inline void CompressorBlockCube<COMP>::ComputeMm1(const RunInfo &info)
{
    // 当前计算到了第几次256大的基本块的循环

    // 分batch进行搬运
    constexpr size_t kL1Base = 128; // k再分128分别搬入L1
    constexpr size_t nL1Split = 256; // L1内n轴的大小
    constexpr size_t nL1Base = 128; // 往L1内每次搬运的总大小（Dkv+Dgate）
    constexpr size_t mL1Split = 256; // L1内m轴的大小
    constexpr size_t mL1Base = 128; // 往L1内每次搬运的token大小

    // fixpipe 同步（因为UB空间有限，无法double buffer；GM workspace可以做 double buffer）
    WaitFlag<HardEvent::FIX_M>(fixToM);

    BatchInfo batchInfo;
    GetBatchInfo(batchInfo, info.bStart);
    const size_t totalStartT = batchInfo.cuSeqLens + info.sStart;
    const size_t totalEndT = cuSeqlensGm_.GetValue(info.bEnd) + info.sEnd;
    for (size_t kL1 = 0; kL1 < constInfo_.hSize; kL1 += kL1Base) { // K 循环
        size_t rCount = mL1Split / constInfo_.cmpRatio;
        size_t rIndex = 0;
        if constexpr (COMP::coff == COFF::OVERLAP) {
            rCount += 1;
            if (totalStartT == batchInfo.cuSeqLens) {
                // 这意味着 batch 开头，不需要拷贝，而是需要空一格 cmpRatio 的大小
                rIndex = 1;
            }
        }
        size_t curT = totalStartT;
        BatchInfo curBatchInfo = batchInfo;
        WaitFlag<HardEvent::MTE1_MTE2>(mte1ToMte2A[l1AIter & 1]);
        for (; rIndex < rCount; rIndex++) {
            if (curT >= totalEndT) {
                // 意味着结束拷贝
                break;
            }
            // 计算拷贝开头相对与压缩块开头的偏移量
            size_t rOffset = (curBatchInfo.startPos + curT - curBatchInfo.cuSeqLens) % constInfo_.cmpRatio;
            // 计算拷贝长度
            size_t len = constInfo_.cmpRatio - rOffset;
            bool changeBatchFlag = false;
            if (curT + len >= curBatchInfo.cuSeqLens + curBatchInfo.seqUsed) {
                // 意味着需要换 batch（有可能刚好到batch序列边界，这种也需要换batch）
                len = curBatchInfo.cuSeqLens + curBatchInfo.seqUsed - curT;
                changeBatchFlag = true;
            }
            CopyNDGMToL1X(curT, len, rOffset, (constInfo_.cmpRatio * rIndex),
                kL1, L1A_PP_SIZE / sizeof(X_T) * (l1AIter & 1));
            if (changeBatchFlag) {
                if (curBatchInfo.batchId + 1 >= constInfo_.batchSize) {
                    // 最后一batch
                    break;
                }
                // 获取下一个 batch 的信息
                GetBatchInfo(curBatchInfo, curBatchInfo.batchId + 1);
                // 刷新到下一个 batch 的开头
                curT = curBatchInfo.cuSeqLens;
            } else {
                curT += len;
            }
        }
        SetFlag<HardEvent::MTE2_MTE1>(mte2ToMte1A[l1AIter & 1]);
        WaitFlag<HardEvent::MTE2_MTE1>(mte2ToMte1A[l1AIter & 1]);
        if constexpr (COMP::coff == COFF::OVERLAP) {
            LeftRight(kL1, 0, 0);
            LeftRight(kL1, 1, constInfo_.cmpRatio); // Right 需要向后偏移一个 cmpRatio
        } else {
            OnlyRight(kL1, 0, wkvGm_);
            OnlyRight(kL1, 1, wgateGm_);
        }
        // 释放 L1A buffer
        SetFlag<HardEvent::MTE1_MTE2>(mte1ToMte2A[l1AIter & 1]);
        l1AIter++;
    }

    // 往外拷贝，LC0->UB
    SetFlag<HardEvent::M_FIX>(mToFix);
    WaitFlag<HardEvent::M_FIX>(mToFix);
    CopyOutToUB();
    SetFlag<HardEvent::FIX_M>(fixToM);
}

} // namespace Compressor

#endif // COMPRESSOR_BLOCK_CUBE_H