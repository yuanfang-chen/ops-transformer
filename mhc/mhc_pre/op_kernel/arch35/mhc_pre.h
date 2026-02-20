/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
 * the software repository for the full text of the License.
 */

/*!
 * \file mhc_pre.h
 * \brief
 */

#ifndef __mhc_pre_KERNEL_H_
#define __mhc_pre_KERNEL_H_

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "mhc_pre_utils.h"

namespace MhcPre {

using namespace matmul;
using namespace AscendC;

constexpr MicroAPI::CastTrait ctHalf2Fp32Zero = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
                                                 MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

struct InitParams {
    GM_ADDR x;
    GM_ADDR phi;
    GM_ADDR alpha;
    GM_ADDR bias;
    GM_ADDR gamma;
    GM_ADDR hin;
    GM_ADDR h_post;
    GM_ADDR h_res;
    GM_ADDR inv_rms;
    GM_ADDR mm_res;
    GM_ADDR h_pre;
    GM_ADDR workspace;
    TPipe *tPipeIn;
    MhcPreTilingData *tilingData;
};

struct MatrixInfo {
    uint64_t totalLength = 0; // 总长度 (batch * sequence 或 T)
    uint64_t nD = 0;          // n * D
    uint64_t fusionSize = 0;  // phi的第二维
    float normEps = 0.0f;     // 归一化epsilon
    float hcEps = 0.0f;       // hyper connection epsilon
};

struct VectorOffsetParams {
    uint64_t globalOffsetM = 0;
    uint64_t singleCoreM = 0;
    uint64_t offsetMStart = 0;
    uint64_t offsetMEnd = 0;
};

struct MNConfig {
    uint64_t m = 0;
    uint64_t n = 0;
    uint64_t k = 0;
    uint64_t baseM = 0;
    uint64_t baseN = 0;
    uint64_t baseK = 0;
    uint64_t curbaseM = 0;
    uint64_t curBaseN = 0;
    uint64_t curBaseK = 0;
    uint64_t singleCoreM;
    uint64_t singleCoreN;
    uint64_t singleCoreK;
    uint64_t curSingleCoreM;
    uint64_t curSingleCoreN;
    uint64_t curSingleCoreK;
};

// ========== 编译时常量 ==========
// 分块大小常量
static constexpr uint32_t V0_BASE_T = 16;  // 分块大小，暂定设置
static constexpr uint64_t V1_BASE_T = 16;  // T维度分块大小 TODO
static constexpr uint64_t V1_BASE_D = 32;  // D维度分块大小
static constexpr uint32_t ND_LENGTH = 256; // nD维度分块长度

// 同步标志常量 - PingPong 流水
static constexpr uint64_t SYNC_V2C_P0 = 0x1;  // Vector→Cube, buffer P0 就绪
static constexpr uint64_t SYNC_C2V_P0 = 0x2;  // Cube→Vector, buffer P0 可复用
static constexpr uint64_t SYNC_V2C_P1 = 0x3;  // Vector→Cube, buffer P1 就绪
static constexpr uint64_t SYNC_C2V_P1 = 0x4;  // Cube→Vector, buffer P1 可复用
static constexpr uint64_t SYNC_C2V1 = 0x5;    // Cube→Vector, 矩阵乘法全部完成
static constexpr uint64_t VEC1_FLAG_OFFSET = MhcPreUtils::VEC1_FLAG_ID_OFFSET; // subBlock1 flag 偏移

// 其他常量
static constexpr uint32_t parallNum_ = 2; // 并行数量
static constexpr uint32_t AL1_PINGPONG = 2; // aL1 double buffer 数量

using aT = MatmulType<TPosition::TSCM, CubeFormat::ND, float32_t>; // TPosition::TSCM
using bT = MatmulType<TPosition::GM, CubeFormat::ND, float32_t, true>;
using cT = MatmulType<TPosition::GM, CubeFormat::ND, float32_t>;
using MT = matmul::MatmulImpl<aT, bT, cT>;

template <class T, class P>
class MhcPreKernel {
public:
    __aicore__ inline MhcPreKernel(MT &matmul) : mm(matmul)
    {
    }
    __aicore__ inline void Init(InitParams initParams);
    __aicore__ inline void Process();
    __aicore__ inline void AICProcess();
    __aicore__ inline void InitUbBuffers();
    __aicore__ inline void InitCubeBuffers();
    __aicore__ inline void VectorComputeOffset();
    __aicore__ inline void V0Process(uint32_t curblock, uint32_t tblockNum);
    __aicore__ inline LocalTensor<P> PrepareV0NdBlock(uint32_t offsetNd, uint32_t &curNdLen);
    __aicore__ inline void RunV0XGammaCompute(LocalTensor<P> &aL1Ub, LocalTensor<P> &invRmsUb, uint32_t curMLen,
                                              uint32_t curNdLen, uint32_t offsetNd);
    __aicore__ inline void ComputeV0MBlock(LocalTensor<P> &curAL1Buf, uint32_t offsetNd, uint32_t curNdLen,
                                           uint32_t offsetM);
    __aicore__ inline void DataCopyX(uint32_t curMLen, uint32_t curNdLen, uint32_t offsetM, uint32_t offsetNd);
    __aicore__ inline void DataCopyOutInvRmsUb(uint32_t curMLen, uint32_t offsetM);
    __aicore__ inline void DataCopyOutToWorkSpace(LocalTensor<P> &x, uint32_t curMLen, uint32_t curNdLen,
                                                  uint32_t offsetM, uint32_t offsetNd, LocalTensor<P> &aL1Buf);
    __aicore__ inline void DataCopyGamma(uint32_t curNdLen, uint32_t offsetNd);
    __aicore__ inline void AIVPreLoad();
    __aicore__ inline void HMixCopyIn(uint64_t offset, uint64_t lenT);
    __aicore__ inline void AIV1ProcessHPost(uint64_t offsetT, uint64_t lenT, uint64_t lenD);
    __aicore__ inline void AIV1ProcessHPre(uint64_t offsetT, uint64_t lenT);
    __aicore__ inline void AIV1Prologue(uint64_t offsetT, uint64_t lenT, uint64_t singleCoreOffset);
    __aicore__ inline void AIV1Process(uint64_t curBlock, uint64_t tBlockNum);
    __aicore__ inline void BiasCopyIn();
    __aicore__ inline void AIV1GetHSliceOffset();
    __aicore__ inline void AIV1ProcessHIn(uint64_t offsetT, uint64_t lenT, uint64_t offsetD, uint64_t lenD);
    template <bool hasGamma, bool isFirstND>
    __aicore__ inline void VFDoV0ProcessXIn(__ubuf__ P *xDst, __ubuf__ P *invRmsDst, __ubuf__ T *xIn, __ubuf__ P *gamma,
                                            uint16_t mSize, uint16_t nSize);
    __aicore__ inline void VFDoV0ProcessInvRms(__ubuf__ P *invRms, uint16_t nSize, float scaleMean, float normEps);
    __aicore__ inline uint64_t GetVecFlagOffset() const;
    __aicore__ inline void NotifyCube(uint64_t syncFlag);
    __aicore__ inline void WaitForCube(uint64_t syncFlag);
    __aicore__ inline void EndWaitForCube(uint32_t ndBlockNum);
    __aicore__ inline void WaitForVector(uint64_t syncFlag);
    __aicore__ inline void NotifyVector(uint64_t syncFlag);

private:
    MT &mm;
    MNConfig mnConfig_;
    uint32_t coreNum_;
    uint64_t totalLength_;
    uint64_t N_;
    uint64_t D_;
    bool outFlag_;
    bool hasGamma_;

    GlobalTensor<T> xGm_;      // 输入 x
    GlobalTensor<P> phiGm_;    // 输入 phi
    GlobalTensor<P> alphaGm_;  // 输入 alpha
    GlobalTensor<P> biasGm_;   // 输入 bias
    GlobalTensor<P> gammaGm_;  // 输入 gamma
    GlobalTensor<T> hinGm_;    // 输出 hin
    GlobalTensor<P> hPostGm_;  // 输出 h_post
    GlobalTensor<P> hResGm_;   // 输出 h_res
    GlobalTensor<P> invRmsGm_; // 输出 h_res
    GlobalTensor<P> mmResGm_;  // 输出 h_res
    GlobalTensor<P> hPreGm_;   // 输出 h_res
    GlobalTensor<P> xFloatGm_;

    // TODO: 使用LocalTensor构造函数和同步指令
    TQue<QuePosition::VECIN, 1> xInQueue_;
    TQue<QuePosition::VECIN, 1> biasQueue_;
    TQue<QuePosition::VECIN, 1> gammaInQueue_;
    TQue<QuePosition::VECOUT, 1> invRmsOutQueue_;
    TQue<QuePosition::VECOUT, 1> outQueue_;
    TQue<QuePosition::VECIN, 1> biasInQue_;

    TBuf<TPosition::VECCALC> tmpBuff_;
    TBuf<TPosition::VECCALC> alphaBuf_;

    LocalTensor<P> inputBuff_;
    LocalTensor<P> hPreBuff_;
    LocalTensor<P> hPreBrcb_;
    LocalTensor<P> hPostBuff_;
    LocalTensor<P> hResBuff_;
    LocalTensor<uint8_t> brcbBuff_;

    LocalTensor<P> alphaInUb_;
    LocalTensor<P> biasInUb_;
    LocalTensor<P> broadCastTmpUb_;
    LocalTensor<P> matmulRes_;

    LocalTensor<T> xLocal_;
    LocalTensor<P> invRmsUb_;
    LocalTensor<P> gammaUb_;

    LocalTensor<uint32_t> preOffsetBuf_;
    LocalTensor<uint32_t> postOffsetBuf_;
    LocalTensor<uint32_t> resOffsetBuf_;

    // aL1 Double Buffer for PingPong
    LocalTensor<P> aL1Buf0_; // L1 buffer P0
    LocalTensor<P> aL1Buf1_; // L1 buffer P1
    uint8_t curAL1BufIdx_ = 0; // 当前使用的 buffer 索引

    TPipe *pipe_;
    MatrixInfo matrixInfo_;
    VectorOffsetParams vectorOffset_;

    const MhcPreTilingData *tiling_;

    // 运行时状态变量
    uint32_t chunTSize_ = 192;
    uint32_t v1ChunkDSize_ = 5120;
    uint32_t curSingleT_ = 192; // 当前块的实际大小（可能小于singleM_，如果是最后一个块）
    uint32_t coreIdx_ = 0;
    uint32_t subBlockIdx_ = 0;
    float scaleMean_ = 0.0f; // 1/nD
    uint64_t globalOffsetM_ = 0;
    uint32_t vectorCount_ = 0;
    uint32_t cubeCount_ = 0;
    uint64_t mmCount_ = 0;
    uint64_t vec1Count_ = 0;
    uint32_t minT_ = 1;
};

template <class T, class P>
__aicore__ inline void MhcPreKernel<T, P>::Init(InitParams initParams)
{
    // 1. 绑定GlobalTensor
    xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(initParams.x));
    phiGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P *>(initParams.phi));
    alphaGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P *>(initParams.alpha));
    biasGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P *>(initParams.bias));
    gammaGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P *>(initParams.gamma));
    hinGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(initParams.hin));
    hPostGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P *>(initParams.h_post));
    hResGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P *>(initParams.h_res));

    invRmsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P *>(initParams.inv_rms));
    hPreGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P *>(initParams.h_pre));

    xFloatGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P *>(initParams.workspace));

    // 2. 获取TilingData进行初始化
    tiling_ = initParams.tilingData;
    matrixInfo_.totalLength = tiling_->totalLength;
    matrixInfo_.nD = tiling_->nD;

    N_ = tiling_->N;
    D_ = tiling_->D;
    matrixInfo_.fusionSize = tiling_->fusionSize;
    matrixInfo_.normEps = tiling_->normEps;
    matrixInfo_.hcEps = tiling_->hcEps;
    coreNum_ = tiling_->coreNum;
    totalLength_ = tiling_->totalLength;
    outFlag_ = (tiling_->outFlag != 0);
    scaleMean_ = tiling_->scaleMean;
    chunTSize_ = tiling_->chunkTSize;
    v1ChunkDSize_ = tiling_->v1ChunkDSize;
    hasGamma_ = (tiling_->hasGamma != 0);

    if (N_ == 4) {
        minT_ = 2;
    }
    if (N_ == 6) {
        minT_ = 4;
    }

    mnConfig_.m = matrixInfo_.totalLength;
    mnConfig_.n = matrixInfo_.fusionSize;
    mnConfig_.k = matrixInfo_.nD;
    mnConfig_.singleCoreM = chunTSize_;
    mnConfig_.singleCoreN = mnConfig_.n;
    mnConfig_.singleCoreK = 256;
    mnConfig_.curSingleCoreM = mnConfig_.singleCoreM;
    mnConfig_.curSingleCoreN = mnConfig_.singleCoreN;
    mnConfig_.curSingleCoreK = mnConfig_.singleCoreK;
    curSingleT_ = chunTSize_;

    if (outFlag_) {
        mmResGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P *>(initParams.mm_res));
    } else {
        mmResGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P *>(
            initParams.workspace + mnConfig_.singleCoreM * mnConfig_.singleCoreK * parallNum_ * sizeof(P) * coreNum_));
    }

    // 3. 申请UB
    pipe_ = initParams.tPipeIn;
    coreIdx_ = GetBlockIdx();
    subBlockIdx_ = GetSubBlockIdx();

    InitUbBuffers();
    InitCubeBuffers();

    SyncAll<false>();
}

template <class T, class P>
__aicore__ inline void MhcPreKernel<T, P>::InitCubeBuffers()
{
    uint64_t bufSize = mnConfig_.singleCoreM * mnConfig_.singleCoreK;
    aL1Buf0_ = LocalTensor<P>(TPosition::TSCM, 0, bufSize);
    aL1Buf1_ = LocalTensor<P>(TPosition::TSCM, bufSize * sizeof(P), bufSize);

    if ASCEND_IS_NOT_AIC {
        return;
    }
}

template <class T, class P>
__aicore__ inline void MhcPreKernel<T, P>::InitUbBuffers()
{
    if ASCEND_IS_NOT_AIV {
        return;
    }

    pipe_->InitBuffer(xInQueue_, 1, 20 * 1024);                           // 20KB
    pipe_->InitBuffer(outQueue_, 1, 20 * 1024);                           // 32KB
    pipe_->InitBuffer(invRmsOutQueue_, 1, (curSingleT_ / 2) * sizeof(P)); // 1KB

    if (hasGamma_) {
        pipe_->InitBuffer(gammaInQueue_, 1, ND_LENGTH * sizeof(P)); // 1KB
    }

    // TODO: 全改RegBase后可以淘汰掉tmpBuff
    pipe_->InitBuffer(tmpBuff_, 105 * 1024); // 120KB

    pipe_->InitBuffer(biasInQue_, 1, mnConfig_.n * sizeof(P)); // 1KB
    pipe_->InitBuffer(alphaBuf_, mnConfig_.n * sizeof(P));     // 1KB
    alphaInUb_ = alphaBuf_.Get<P>();

    preOffsetBuf_ = tmpBuff_.GetWithOffset<uint32_t>(uint32_t(mnConfig_.n * V1_BASE_T), 0);
    postOffsetBuf_ = preOffsetBuf_[N_ * V1_BASE_T];
    resOffsetBuf_ = postOffsetBuf_[N_ * V1_BASE_T];
    uint64_t buffOffset = mnConfig_.n * V1_BASE_T * sizeof(uint32_t);

    // V1 hPreBuff这些都换成OutQueue buffer。
    buffOffset = mnConfig_.n * V1_BASE_T * sizeof(uint32_t) + (curSingleT_ / 2) * sizeof(uint32_t);
    hPreBuff_ = tmpBuff_.GetWithOffset<P>(uint32_t(V1_BASE_T * N_), buffOffset);
    buffOffset += V1_BASE_T * N_ * sizeof(P);
    inputBuff_ = tmpBuff_.GetWithOffset<P>(uint32_t(mnConfig_.n * V1_BASE_T * V1_BASE_D), buffOffset);
    buffOffset += mnConfig_.n * V1_BASE_T * V1_BASE_D * sizeof(P);
    broadCastTmpUb_ = tmpBuff_.GetWithOffset<P>(uint32_t(mnConfig_.n * V1_BASE_T), buffOffset); // 20KB
    buffOffset += mnConfig_.n * V1_BASE_T * sizeof(P);
    hPostBuff_ = tmpBuff_.GetWithOffset<P>(uint32_t(V1_BASE_T * N_), buffOffset);
    buffOffset += V1_BASE_T * N_ * sizeof(P);
    hResBuff_ = tmpBuff_.GetWithOffset<P>(uint32_t(V1_BASE_T * N_), buffOffset);
    buffOffset += V1_BASE_T * N_ * sizeof(P);
}

template <class T, class P>
__aicore__ inline void MhcPreKernel<T, P>::Process()
{
    // TODO: 实现核心计算逻辑
    // 根据tiling数据进行分块处理
    uint32_t tBlockNum = Ceil(totalLength_, chunTSize_);
    if ASCEND_IS_AIV {
        coreIdx_ = GetBlockIdx() / 2;
        AIVPreLoad();
    }

    for (uint64_t offset = coreIdx_; offset < tBlockNum; offset += coreNum_) {
        // 处理每个batch/sequence的数据块
        // 1. 计算 hin
        // 2. 计算 h_post
        // 3. 计算 h_res
        globalOffsetM_ = offset * chunTSize_; // chunTSize_ = 192

        if ASCEND_IS_AIV {
            V0Process(offset, tBlockNum);
            AIV1Process(offset, tBlockNum);
        }

        if ASCEND_IS_AIC {
            if (offset == tBlockNum - 1) {
                mnConfig_.curSingleCoreM = totalLength_ - globalOffsetM_; // 尾块处理
            }
            AICProcess();
        }
    }
    if ASCEND_IS_AIV {
        invRmsOutQueue_.FreeTensor(invRmsUb_);
        biasInQue_.FreeTensor(biasInUb_);
    }
}

template <class T, class P>
__aicore__ inline void MhcPreKernel<T, P>::AIV1GetHSliceOffset()
{
    uint32_t offset1 = 0;
    uint32_t offset2 = 0;
    uint32_t offset3 = 0;
    uint32_t curOffset = 0;
    uint32_t nSquare = N_ * N_;
    for (uint32_t i = 0; i < V1_BASE_T; i++) {
        for (uint32_t j = 0; j < N_; j++) {
            preOffsetBuf_.SetValue(offset1++, curOffset * sizeof(P));
            curOffset++;
        }
        for (uint32_t j = 0; j < N_; j++) {
            postOffsetBuf_.SetValue(offset2++, curOffset * sizeof(P));
            curOffset++;
        }
        for (uint32_t j = 0; j < nSquare; j++) {
            resOffsetBuf_.SetValue(offset3++, curOffset * sizeof(P));
            curOffset++;
        }
    }
}

template <class T, class P>
__aicore__ inline uint64_t MhcPreKernel<T, P>::GetVecFlagOffset() const
{
    return (subBlockIdx_ == 0) ? MhcPreUtils::VEC0_FLAG_ID_OFFSET : MhcPreUtils::VEC1_FLAG_ID_OFFSET;
}

template <class T, class P>
__aicore__ inline void MhcPreKernel<T, P>::NotifyCube(uint64_t syncFlag)
{
    CrossCoreSetFlag<MhcPreUtils::SYNC_AIC_AIV_MODE, PIPE_MTE3>(syncFlag + GetVecFlagOffset());
}

template <class T, class P>
__aicore__ inline void MhcPreKernel<T, P>::WaitForCube(uint64_t syncFlag)
{
    AscendC::CrossCoreWaitFlag<MhcPreUtils::SYNC_AIC_AIV_MODE, PIPE_MTE3>(syncFlag + GetVecFlagOffset());
}

template <class T, class P>
__aicore__ inline void MhcPreKernel<T, P>::EndWaitForCube(uint32_t ndBlockNum)
{
    if (ndBlockNum > 0) {
        WaitForCube(SYNC_C2V_P0);
    }
    if (ndBlockNum > 1) {
        WaitForCube(SYNC_C2V_P1);
    }
}

template <class T, class P>
__aicore__ inline void MhcPreKernel<T, P>::WaitForVector(uint64_t syncFlag)
{
    AscendC::CrossCoreWaitFlag<MhcPreUtils::SYNC_AIC_AIV_MODE, PIPE_MTE1>(syncFlag);
    AscendC::CrossCoreWaitFlag<MhcPreUtils::SYNC_AIC_AIV_MODE, PIPE_MTE1>(syncFlag + VEC1_FLAG_OFFSET);
}

template <class T, class P>
__aicore__ inline void MhcPreKernel<T, P>::NotifyVector(uint64_t syncFlag)
{
    CrossCoreSetFlag<MhcPreUtils::SYNC_AIC_AIV_MODE, PIPE_MTE1>(syncFlag);
    CrossCoreSetFlag<MhcPreUtils::SYNC_AIC_AIV_MODE, PIPE_MTE1>(syncFlag + VEC1_FLAG_OFFSET);
}

template <class T, class P>
__aicore__ inline void MhcPreKernel<T, P>::AICProcess()
{
    uint64_t outOffset = 0;
    if (outFlag_) {
        outOffset = globalOffsetM_ * mnConfig_.n;
    } else {
        outOffset = mnConfig_.singleCoreM * mnConfig_.singleCoreN * ((mmCount_ % 2) * coreNum_ + coreIdx_);
    }
    mnConfig_.curSingleCoreK = mnConfig_.singleCoreK;
    curAL1BufIdx_ = 0;

    for (uint64_t offsetNd = 0; offsetNd < mnConfig_.k; offsetNd += mnConfig_.singleCoreK) {
        uint64_t waitFlag = (curAL1BufIdx_ == 0) ? SYNC_V2C_P0 : SYNC_V2C_P1;
        WaitForVector(waitFlag);

        if (offsetNd + mnConfig_.singleCoreK > mnConfig_.k) {
            mnConfig_.curSingleCoreK = mnConfig_.k - offsetNd;
        }

        LocalTensor<P> curAL1Buf = (curAL1BufIdx_ == 0) ? aL1Buf0_ : aL1Buf1_;

        mm.SetOrgShape(mnConfig_.curSingleCoreM, mnConfig_.curSingleCoreN, mnConfig_.curSingleCoreK,
                       mnConfig_.k);
        mm.SetSingleShape(mnConfig_.curSingleCoreM, mnConfig_.curSingleCoreN,
                          mnConfig_.curSingleCoreK);
        mm.SetTensorA(curAL1Buf);
        mm.SetTensorB(phiGm_[offsetNd], true);
        mm.IterateAll(mmResGm_[outOffset], offsetNd == 0 ? 0 : 1);
        mm.End();
        cubeCount_++;

        uint64_t setFlag = (curAL1BufIdx_ == 0) ? SYNC_C2V_P0 : SYNC_C2V_P1;
        NotifyVector(setFlag);

        curAL1BufIdx_ = (curAL1BufIdx_ + 1) % AL1_PINGPONG;
    }
    mmCount_++;
    NotifyVector(SYNC_C2V1);
}

template <class T, class P>
__aicore__ inline void MhcPreKernel<T, P>::VectorComputeOffset()
{
    uint64_t aliginSingleM = Ceil(curSingleT_ / 2, 8) * 8; // 32Byte对齐
    vectorOffset_.singleCoreM = aliginSingleM < curSingleT_ ? aliginSingleM : curSingleT_;
    if (subBlockIdx_ == 0) {
        vectorOffset_.offsetMStart = 0;
        vectorOffset_.offsetMEnd = vectorOffset_.singleCoreM;
    } else {
        vectorOffset_.offsetMStart = vectorOffset_.singleCoreM;
        vectorOffset_.singleCoreM = curSingleT_ - vectorOffset_.singleCoreM;
        vectorOffset_.offsetMEnd = curSingleT_;
    }
}

template <class T, class P>
template <bool hasGamma, bool isFirstND>
__aicore__ inline void MhcPreKernel<T, P>::VFDoV0ProcessXIn(__ubuf__ P *xDst, __ubuf__ P *invRmsDst, __ubuf__ T *xIn,
                                                            __ubuf__ P *gamma, uint16_t mSize, uint16_t nSize)
{
    uint32_t eleNumPerVf = MhcPreUtils::GetVRegSize() / sizeof(P);
    // 计算两种数据类型经32B对齐后的Size
    uint32_t nSrcUbAligned = MhcPreUtils::Align(nSize, static_cast<uint16_t>(MhcPreUtils::UB_ALIGN_SIZE / sizeof(T)));
    uint32_t nDstUbAligned = MhcPreUtils::Align(nSize, static_cast<uint16_t>(MhcPreUtils::UB_ALIGN_SIZE / sizeof(P)));
    uint16_t nLoopCnt = MhcPreUtils::CeilDiv(nSize, eleNumPerVf);
    __VEC_SCOPE__
    {
        MicroAPI::MaskReg mask = MicroAPI::CreateMask<P>();
        for (uint16_t mIdx = 0; mIdx < mSize; mIdx++) {
            uint32_t elementNum = nSize; // 每次update会减去VL_T
            MicroAPI::RegTensor<P> sumReg;
            if constexpr (isFirstND) {
                MicroAPI::Duplicate(sumReg, 0);
            } else {
                // 非32B对齐
                MicroAPI::Load(sumReg, invRmsDst + mIdx);
            }
            for (uint16_t vfBlockIdx = 0; vfBlockIdx < nLoopCnt; vfBlockIdx++) {
                MicroAPI::RegTensor<T> xInReg;
                MicroAPI::RegTensor<P> gammaReg;
                MicroAPI::RegTensor<P> xFp32Reg, xMulReg, xSquaReg;
                MicroAPI::RegTensor<P> tmpSumReg;

                // 从UB搬运到Register，地址需要32B对齐，从GM搬运时保证
                uint32_t xInOffset = mIdx * nSrcUbAligned + vfBlockIdx * eleNumPerVf;
                // TODO: DataCopy换成LoadAlign和StoreAlign
                MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(xInReg, xIn + xInOffset);
                // cast b16 -> fp32
                MicroAPI::MaskReg maskN4B32 = MicroAPI::UpdateMask<P>(elementNum);
                MicroAPI::Cast<float, T, ctHalf2Fp32Zero>(xFp32Reg, xInReg, maskN4B32);
                // castX * gamma
                // regbase内判断都用constexpr
                if constexpr (hasGamma) {
                    MicroAPI::LoadAlign(gammaReg, gamma + vfBlockIdx * eleNumPerVf);
                    MicroAPI::Mul(xMulReg, gammaReg, xFp32Reg, maskN4B32);
                } else {
                    xMulReg = xFp32Reg;
                }
                // copy out x, 对齐搬出，这里可能要和Cube适配。
                uint32_t dstUbOffset = mIdx * nDstUbAligned + vfBlockIdx * eleNumPerVf;
                MicroAPI::StoreAlign(xDst + dstUbOffset, xMulReg, maskN4B32);

                // xFp32 * xFp32
                MicroAPI::Mul(xSquaReg, xFp32Reg, xFp32Reg, maskN4B32);

                // reducesum, 累加到sumReg第一个元素上
                MicroAPI::Reduce<MicroAPI::ReduceType::SUM>(tmpSumReg, xSquaReg, maskN4B32);
                MicroAPI::Add(sumReg, sumReg, tmpSumReg, maskN4B32);
            }

            // 搬出地址不32B对齐
            MicroAPI::Store(invRmsDst + mIdx, sumReg, 1);
        }
    }
}

template <class T, class P>
__aicore__ inline void MhcPreKernel<T, P>::VFDoV0ProcessInvRms(__ubuf__ P *invRms, uint16_t nSize, float scaleMean,
                                                               float normEps)
{
    uint32_t eleNumPerVf = MhcPreUtils::GetVRegSize() / sizeof(P);
    uint32_t nUbAligned = MhcPreUtils::Align(nSize, static_cast<uint16_t>(MhcPreUtils::UB_ALIGN_SIZE / sizeof(P)));
    uint16_t nLoopCnt = MhcPreUtils::CeilDiv(nSize, eleNumPerVf);
    __VEC_SCOPE__
    {
        uint32_t elementNum = nSize; // 每次update会减去VL_T
        MicroAPI::MaskReg mask = MicroAPI::CreateMask<P>();
        for (uint16_t vfBlockIdx = 0; vfBlockIdx < nLoopCnt; vfBlockIdx++) {
            MicroAPI::MaskReg maskN4B32 = MicroAPI::UpdateMask<P>(elementNum);
            MicroAPI::RegTensor<P> invrmsReg, onesReg;

            MicroAPI::LoadAlign(invrmsReg, invRms + vfBlockIdx * eleNumPerVf);
            MicroAPI::Muls(invrmsReg, invrmsReg, scaleMean, maskN4B32);
            MicroAPI::Adds(invrmsReg, invrmsReg, normEps, maskN4B32);
            MicroAPI::Sqrt(invrmsReg, invrmsReg, maskN4B32);
            MicroAPI::Duplicate(onesReg, 1);
            MicroAPI::Div(invrmsReg, onesReg, invrmsReg, maskN4B32);
            MicroAPI::StoreAlign(invRms + vfBlockIdx * eleNumPerVf, invrmsReg, maskN4B32);
        }
    }
}

template <class T, class P>
__aicore__ inline LocalTensor<P> MhcPreKernel<T, P>::PrepareV0NdBlock(uint32_t offsetNd, uint32_t &curNdLen)
{
    curNdLen = ND_LENGTH;
    if (offsetNd + ND_LENGTH >= matrixInfo_.nD) {
        curNdLen = matrixInfo_.nD - offsetNd;
    }
    if (offsetNd >= AL1_PINGPONG * ND_LENGTH) {
        uint64_t waitFlag = (curAL1BufIdx_ == 0) ? SYNC_C2V_P0 : SYNC_C2V_P1;
        WaitForCube(waitFlag);
    }
    return (curAL1BufIdx_ == 0) ? aL1Buf0_ : aL1Buf1_;
}

template <class T, class P>
__aicore__ inline void MhcPreKernel<T, P>::RunV0XGammaCompute(LocalTensor<P> &aL1Ub, LocalTensor<P> &invRmsUb,
                                                               uint32_t curMLen, uint32_t curNdLen, uint32_t offsetNd)
{
    if (hasGamma_) {
        gammaUb_ = gammaInQueue_.AllocTensor<P>();
        DataCopyGamma(curNdLen, offsetNd);
        gammaUb_ = gammaInQueue_.DeQue<P>();
        if (offsetNd == 0) {
            VFDoV0ProcessXIn<true, true>((__ubuf__ P *)aL1Ub.GetPhyAddr(), (__ubuf__ P *)invRmsUb.GetPhyAddr(),
                                         (__ubuf__ T *)xLocal_.GetPhyAddr(), (__ubuf__ P *)gammaUb_.GetPhyAddr(),
                                         curMLen, curNdLen);
        } else {
            VFDoV0ProcessXIn<true, false>((__ubuf__ P *)aL1Ub.GetPhyAddr(), (__ubuf__ P *)invRmsUb.GetPhyAddr(),
                                          (__ubuf__ T *)xLocal_.GetPhyAddr(), (__ubuf__ P *)gammaUb_.GetPhyAddr(),
                                          curMLen, curNdLen);
        }
        PipeBarrier<PIPE_V>();
        gammaInQueue_.FreeTensor(gammaUb_);
        return;
    }
    if (offsetNd == 0) {
        VFDoV0ProcessXIn<false, true>((__ubuf__ P *)aL1Ub.GetPhyAddr(), (__ubuf__ P *)invRmsUb.GetPhyAddr(),
                                      (__ubuf__ T *)xLocal_.GetPhyAddr(), nullptr, curMLen, curNdLen);
    } else {
        VFDoV0ProcessXIn<false, false>((__ubuf__ P *)aL1Ub.GetPhyAddr(), (__ubuf__ P *)invRmsUb.GetPhyAddr(),
                                       (__ubuf__ T *)xLocal_.GetPhyAddr(), nullptr, curMLen, curNdLen);
    }
}

template <class T, class P>
__aicore__ inline void MhcPreKernel<T, P>::ComputeV0MBlock(LocalTensor<P> &curAL1Buf, uint32_t offsetNd,
                                                            uint32_t curNdLen, uint32_t offsetM)
{
    uint32_t curMLen = V0_BASE_T;
    if (offsetM + V0_BASE_T >= vectorOffset_.offsetMEnd) {
        curMLen = vectorOffset_.offsetMEnd - offsetM;
    }
    uint64_t invRmsOffset = offsetM - vectorOffset_.offsetMStart;
    LocalTensor<P> invRmsUb = invRmsUb_[invRmsOffset];
    xLocal_ = xInQueue_.AllocTensor<T>();
    DataCopyX(curMLen, curNdLen, offsetM, offsetNd);
    xLocal_ = xInQueue_.DeQue<T>();
    LocalTensor<P> aL1Ub = outQueue_.AllocTensor<P>();
    RunV0XGammaCompute(aL1Ub, invRmsUb, curMLen, curNdLen, offsetNd);
    outQueue_.EnQue<P>(aL1Ub);
    aL1Ub = outQueue_.DeQue<P>();
    DataCopyOutToWorkSpace(aL1Ub, curMLen, curNdLen, offsetM, offsetNd, curAL1Buf);
    xInQueue_.FreeTensor(xLocal_);
    outQueue_.FreeTensor(aL1Ub);
}

template <class T, class P>
__aicore__ inline void MhcPreKernel<T, P>::V0Process(uint32_t curblock, uint32_t tBlockNum)
{
    curSingleT_ = chunTSize_;
    if (curblock == tBlockNum - 1) {
        curSingleT_ = matrixInfo_.totalLength - curblock * chunTSize_;
    }
    VectorComputeOffset();
    curAL1BufIdx_ = 0;

    for (uint32_t offsetNd = 0; offsetNd < matrixInfo_.nD; offsetNd += ND_LENGTH) {
        uint32_t curNdLen = ND_LENGTH;
        LocalTensor<P> curAL1Buf = PrepareV0NdBlock(offsetNd, curNdLen);
        for (uint32_t offsetM = vectorOffset_.offsetMStart; offsetM < vectorOffset_.offsetMEnd; offsetM += V0_BASE_T) {
            ComputeV0MBlock(curAL1Buf, offsetNd, curNdLen, offsetM);
        }
        uint64_t setFlag = (curAL1BufIdx_ == 0) ? SYNC_V2C_P0 : SYNC_V2C_P1;
        NotifyCube(setFlag);
        curAL1BufIdx_ = (curAL1BufIdx_ + 1) % AL1_PINGPONG;
    }
    uint32_t ndBlockNum = Ceil(matrixInfo_.nD, ND_LENGTH);
    EndWaitForCube(ndBlockNum);
    VFDoV0ProcessInvRms((__ubuf__ P *)invRmsUb_.GetPhyAddr(), vectorOffset_.singleCoreM, scaleMean_,
                        matrixInfo_.normEps);
    DataCopyOutInvRmsUb(vectorOffset_.singleCoreM, vectorOffset_.offsetMStart);
}

template <class T, class P>
__aicore__ inline void MhcPreKernel<T, P>::AIV1Process(uint64_t curBlock, uint64_t tBlockNum)
{
    // TODO: 可以提出去
    curSingleT_ = chunTSize_;
    if (curBlock == tBlockNum - 1) {
        curSingleT_ = matrixInfo_.totalLength - curBlock * chunTSize_;
    }
    VectorComputeOffset(); // 计算0核和1核的偏移
    if (vectorOffset_.singleCoreM <= 0) {
        return;
    }
    WaitForCube(SYNC_C2V1);
    uint64_t lenT = 0;
    uint64_t lenD = 0;
    uint64_t singleCoreOffset = 0;
    for (int offsetT = vectorOffset_.offsetMStart; offsetT < vectorOffset_.offsetMEnd; offsetT += V1_BASE_T) {
        lenT = V1_BASE_T < vectorOffset_.offsetMEnd - offsetT ? V1_BASE_T : vectorOffset_.offsetMEnd - offsetT;
        lenD = v1ChunkDSize_;
        // H 切分
        AIV1Prologue(offsetT, lenT, singleCoreOffset);
        AIV1ProcessHPost(offsetT, lenT, lenD);
        AIV1ProcessHPre(offsetT, lenT);
        auto hInDealBuf = inputBuff_;              // 最大80K
        auto reduceResult = hInDealBuf[lenD * N_]; // 最大10K
        PipeBarrier<PIPE_ALL>();

        for (uint32_t tIdx = 0; tIdx < lenT; tIdx++) {
            for (int offsetD = 0; offsetD < D_; offsetD += v1ChunkDSize_) {
                lenD = v1ChunkDSize_ < D_ - offsetD ? v1ChunkDSize_ : D_ - offsetD;


                uint32_t ubOffset = 0;
                const uint32_t srcReduceShape_[2] = {static_cast<uint32_t>(N_), static_cast<uint32_t>(lenD)};
                for (uint32_t nIdx = 0; nIdx < N_; nIdx++) {
                    uint64_t xGmOffset = (globalOffsetM_ + offsetT + tIdx) * N_ * D_ + nIdx * D_ + offsetD;

                    LocalTensor<T> xIn = xInQueue_.AllocTensor<T>();
                    // copy X
                    DataCopyExtParams copyParams;
                    copyParams.blockCount = static_cast<uint16_t>(1);
                    copyParams.blockLen = uint32_t(lenD * sizeof(T));
                    copyParams.srcStride = 0;
                    copyParams.dstStride = 0;

                    uint32_t rightPadNum = Ceil(lenD, 16) * 16 - lenD;
                    DataCopyPadExtParams<T> padParams{true, 0, static_cast<uint8_t>(rightPadNum), 0};
                    DataCopyPad(xIn, xGm_[xGmOffset], copyParams, padParams);

                    xInQueue_.EnQue(xIn);
                    xIn = xInQueue_.DeQue<T>();

                    Cast(hInDealBuf[ubOffset], xIn, RoundMode::CAST_NONE, lenD);
                    PipeBarrier<PIPE_V>();
                    xInQueue_.FreeTensor(xIn);

                    Muls(hInDealBuf[ubOffset], hInDealBuf[ubOffset], hPreBuff_.GetValue(tIdx * N_ + nIdx), lenD);
                    PipeBarrier<PIPE_V>();
                    ubOffset += lenD;
                }

                // reducesum
                ReduceSum<P, Pattern::Reduce::RA, true>(reduceResult, hInDealBuf, srcReduceShape_, true);
                PipeBarrier<PIPE_V>();

                // cast
                LocalTensor<T> hinOut = outQueue_.AllocTensor<T>();
                Cast(hinOut, reduceResult, RoundMode::CAST_RINT, lenD);
                PipeBarrier<PIPE_V>();

                outQueue_.EnQue(hinOut);
                hinOut = outQueue_.DeQue<T>();
                // copyout
                DataCopyExtParams copyParams;
                copyParams.blockCount = 1; // 行数
                copyParams.blockLen = uint32_t(lenD * sizeof(T));
                copyParams.srcStride = 0;
                copyParams.dstStride = 0;

                DataCopyPad(hinGm_[(globalOffsetM_ + offsetT + tIdx) * D_ + offsetD], hinOut, copyParams);
                PipeBarrier<PIPE_V>();
                outQueue_.FreeTensor(hinOut);
            }
        }

        singleCoreOffset += V1_BASE_T;
    }
}

template <class T, class P>
__aicore__ inline void MhcPreKernel<T, P>::AIV1Prologue(uint64_t offsetT, uint64_t lenT, uint64_t singleCoreOffset)
{
    // TODO: 优化点：先对H slice，再运算。+
    uint64_t offset = globalOffsetM_ + offsetT;
    uint64_t HMixOffset = 0;
    LocalTensor<P> hResOutLocal = outQueue_.AllocTensor<P>();
    if (outFlag_) {
        HMixOffset = (globalOffsetM_ + offsetT) * mnConfig_.n;
    } else {
        HMixOffset = mnConfig_.singleCoreM * mnConfig_.singleCoreN * ((vec1Count_ % 2) * coreNum_ + coreIdx_) +
                     offsetT * mnConfig_.n;
    }
    HMixCopyIn(HMixOffset, lenT);

    uint32_t hMixShape[] = {uint32_t(lenT), uint32_t(mnConfig_.n)};
    uint32_t alphaBiaShape[] = {uint32_t(1), uint32_t(mnConfig_.n)};
    uint32_t rShape[] = {uint32_t(lenT), uint32_t(1)};
    matmulRes_ = xInQueue_.DeQue<P>(); // TODO
    Broadcast<P, 2, 1>(broadCastTmpUb_, invRmsUb_[singleCoreOffset], hMixShape, rShape);
    PipeBarrier<PIPE_V>();
    ;
    matmulRes_ = matmulRes_ * broadCastTmpUb_;
    PipeBarrier<PIPE_V>();
    ;
    Broadcast<P, 2, 0>(broadCastTmpUb_, alphaInUb_, hMixShape, alphaBiaShape);
    PipeBarrier<PIPE_V>();
    ;
    matmulRes_ = matmulRes_ * broadCastTmpUb_;
    PipeBarrier<PIPE_V>();
    ;
    Broadcast<P, 2, 0>(broadCastTmpUb_, biasInUb_, hMixShape, alphaBiaShape);
    PipeBarrier<PIPE_V>();
    ;
    matmulRes_ = matmulRes_ + broadCastTmpUb_;
    PipeBarrier<PIPE_V>();
    ;

    SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);

    Gather(hPreBuff_, matmulRes_, preOffsetBuf_, uint32_t(0), lenT * N_);
    Gather(hPostBuff_, matmulRes_, postOffsetBuf_, uint32_t(0), lenT * N_);
    Gather(hResOutLocal, matmulRes_, resOffsetBuf_, uint32_t(0), lenT * N_ * N_);
    PipeBarrier<PIPE_V>();
    outQueue_.EnQue(hResOutLocal);
    hResOutLocal = outQueue_.DeQue<P>();
    DataCopyExtParams copyParams;
    copyParams.blockCount = static_cast<uint16_t>(1);
    copyParams.blockLen = uint32_t(lenT * N_ * N_ * sizeof(P));
    copyParams.srcStride = uint32_t(0);
    copyParams.dstStride = uint32_t(0);
    DataCopyPad(hResGm_[offset * N_ * N_], hResOutLocal, copyParams);
    outQueue_.FreeTensor(hResOutLocal);
    xInQueue_.FreeTensor(matmulRes_);
}

template <class T, class P>
__aicore__ inline void MhcPreKernel<T, P>::AIV1ProcessHPre(uint64_t offsetT, uint64_t lenT)
{
    uint64_t offset = globalOffsetM_ + offsetT;
    LocalTensor<P> hPreSigmoid = inputBuff_;

    Sigmoid(hPreSigmoid, hPreBuff_, lenT * N_);
    PipeBarrier<PIPE_V>();
    Adds(hPreBuff_, hPreSigmoid, matrixInfo_.hcEps, lenT * N_);
    PipeBarrier<PIPE_V>();
    if (outFlag_) {
        DataCopyExtParams copyParams;
        copyParams.blockCount = static_cast<uint16_t>(1); // 行数
        copyParams.blockLen = uint32_t(lenT * N_ * sizeof(P));
        copyParams.srcStride = uint32_t(0);
        copyParams.dstStride = uint32_t((0));
        SetFlag<HardEvent::V_MTE3>(EVENT_ID2);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID2);
        DataCopyPad(hPreGm_[offset * N_], hPreBuff_, copyParams);
    }
}

template <class T, class P>
__aicore__ inline void MhcPreKernel<T, P>::AIV1ProcessHIn(uint64_t offsetT, uint64_t lenT, uint64_t offsetD,
                                                          uint64_t lenD)
{
    uint64_t offset = globalOffsetM_ + offsetT;
    auto xCast = inputBuff_;
    // input x
    LocalTensor<T> xIn = xInQueue_.DeQue<T>();
    Cast(xCast, xIn, RoundMode::CAST_NONE, lenT * N_ * lenD);
    PipeBarrier<PIPE_V>();
    LocalTensor<T> hPreCast = outQueue_.AllocTensor<T>();

    hPreBrcb_ = xCast[lenT * N_ * lenD];
    const uint32_t srcShape_[2] = {static_cast<uint32_t>(minT_ * N_), 1};
    const uint32_t dstShape_[2] = {static_cast<uint32_t>(minT_ * N_), static_cast<uint32_t>(lenD)};
    const uint32_t srcReduceShape_[2] = {static_cast<uint32_t>(N_), static_cast<uint32_t>(lenD)};

    for (uint64_t i = 0; i < lenT; i += minT_) {
        Broadcast<P, 2, 1>(hPreBrcb_, hPreBuff_[i * N_], dstShape_, srcShape_);
        PipeBarrier<PIPE_V>();
        Mul(xCast[i * N_ * lenD], hPreBrcb_, xCast[i * N_ * lenD], minT_ * N_ * lenD);
        PipeBarrier<PIPE_V>();
        // 消n轴
        for (uint64_t j = 0; j < minT_; j++) {
            ReduceSum<P, Pattern::Reduce::RA, false>(hPreBrcb_[j * lenD], xCast[(i + j) * N_ * lenD], srcReduceShape_,
                                                     true);
            PipeBarrier<PIPE_V>();
        }
        SetFlag<HardEvent::MTE3_V>(EVENT_ID3);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID3);
        Cast(hPreCast, hPreBrcb_, RoundMode::CAST_RINT, minT_ * lenD);
        PipeBarrier<PIPE_V>();
        outQueue_.EnQue(hPreCast);
        hPreCast = outQueue_.DeQue<T>();
        // 写入 h_in
        DataCopyExtParams copyParams;
        copyParams.blockCount = static_cast<uint16_t>(minT_); // 行数
        copyParams.blockLen = uint32_t(lenD * sizeof(T));
        copyParams.srcStride = uint32_t(0);
        copyParams.dstStride = uint32_t((D_ - lenD) * sizeof(T));

        DataCopyPad(hinGm_[(offset + i) * D_ + offsetD], hPreCast, copyParams);
    }
    xInQueue_.FreeTensor(xIn);
    outQueue_.FreeTensor(hPreCast);
}

template <class T, class P>
__aicore__ inline void MhcPreKernel<T, P>::AIV1ProcessHPost(uint64_t offsetT, uint64_t lenT, uint64_t lenD)
{
    uint64_t offset = globalOffsetM_ + offsetT;
    LocalTensor<P> hPostSigmoid = broadCastTmpUb_;
    LocalTensor<P> hPostOutLocal = outQueue_.AllocTensor<P>();
    Sigmoid(hPostSigmoid, hPostBuff_);
    PipeBarrier<PIPE_V>();
    Muls(hPostOutLocal, hPostSigmoid, static_cast<float>(2.0), lenT * N_);
    PipeBarrier<PIPE_V>();
    outQueue_.EnQue(hPostOutLocal);
    hPostOutLocal = outQueue_.DeQue<P>();
    DataCopyExtParams copyParams;
    copyParams.blockCount = static_cast<uint16_t>(1);
    copyParams.blockLen = uint32_t(lenT * N_ * sizeof(P));
    copyParams.srcStride = uint32_t(0);
    copyParams.dstStride = uint32_t(0);
    DataCopyPad(hPostGm_[offset * N_], hPostOutLocal, copyParams);
    outQueue_.FreeTensor(hPostOutLocal);
}

template <class T, class P>
__aicore__ inline void MhcPreKernel<T, P>::AIVPreLoad()
{
    invRmsUb_ = invRmsOutQueue_.AllocTensor<P>();
    AIV1GetHSliceOffset();

    float alphaPre = alphaGm_.GetValue(0);
    float alphaPost = alphaGm_.GetValue(1);
    float alphaComb = alphaGm_.GetValue(2);
    for (uint64_t i = 0; i < N_; ++i) {
        alphaInUb_.SetValue(i, alphaPre);
        alphaInUb_.SetValue(i + N_, alphaPost);
        for (uint64_t j = 0; j < N_; ++j) {
            alphaInUb_.SetValue((2 + i) * N_ + j, alphaComb);
        }
    }
    BiasCopyIn();
    biasInUb_ = biasInQue_.DeQue<P>();
}

template <class T, class P>
__aicore__ inline void MhcPreKernel<T, P>::BiasCopyIn()
{
    LocalTensor<P> biasLocal = biasInQue_.AllocTensor<P>();

    DataCopyExtParams copyParams;
    copyParams.blockCount = static_cast<uint16_t>(1); // 行数
    copyParams.blockLen = uint32_t(matrixInfo_.fusionSize * sizeof(P));
    copyParams.srcStride = uint32_t(0); // 相邻块的间隔
    copyParams.dstStride = uint32_t(0); // 相邻块的间隔
    DataCopyPadExtParams<P> copyPadParams{true, 0, 0, 0};

    DataCopyPad(biasLocal, biasGm_, copyParams, copyPadParams);
    biasInQue_.EnQue(biasLocal);
}

template <class T, class P>
__aicore__ inline void MhcPreKernel<T, P>::DataCopyX(uint32_t curMLen, uint32_t curNdLen, uint32_t offsetM,
                                                     uint32_t offsetNd)
{
    DataCopyExtParams copyParams;
    copyParams.blockCount = static_cast<uint16_t>(curMLen);
    copyParams.blockLen = uint32_t(curNdLen * sizeof(T));
    copyParams.srcStride = uint32_t((matrixInfo_.nD - curNdLen) * sizeof(T));
    copyParams.dstStride = uint32_t(0);

    // 搬运每一块32B对齐
    uint32_t rightPadNum = Ceil(curNdLen, 16) * 16 - curNdLen;
    DataCopyPadExtParams<T> padParams{true, 0, static_cast<uint8_t>(rightPadNum), 0};

    uint64_t offset = globalOffsetM_ * matrixInfo_.nD + offsetM * matrixInfo_.nD + offsetNd;
    DataCopyPad(xLocal_, xGm_[offset], copyParams, padParams);
    xInQueue_.EnQue<T>(xLocal_);
}

template <class T, class P>
__aicore__ inline void MhcPreKernel<T, P>::DataCopyGamma(uint32_t curNdLen, uint32_t offsetNd)
{
    DataCopyExtParams copyParams;
    copyParams.blockCount = static_cast<uint16_t>(1);
    copyParams.blockLen = uint32_t(curNdLen * sizeof(P));
    copyParams.srcStride = uint32_t(0);
    copyParams.dstStride = uint32_t(0);

    uint32_t rightPadNum = Ceil(curNdLen, 16) * 16 - curNdLen;
    DataCopyPadExtParams<P> padParams{true, 0, static_cast<uint8_t>(rightPadNum), 0};

    uint64_t offset = offsetNd;

    DataCopyPad(gammaUb_, gammaGm_[offset], copyParams, padParams);
    gammaInQueue_.EnQue<P>(gammaUb_);
}

template <class T, class P>
__aicore__ inline void MhcPreKernel<T, P>::HMixCopyIn(uint64_t offset, uint64_t lenT)
{
    LocalTensor<P> hMixLocal = xInQueue_.AllocTensor<P>();

    DataCopyExtParams copyParams;
    copyParams.blockCount = static_cast<uint16_t>(1); // 行数
    copyParams.blockLen = uint32_t(lenT * mnConfig_.n * sizeof(P));
    copyParams.srcStride = uint32_t(0); // 相邻块的间隔
    copyParams.dstStride = uint32_t(0); // 相邻块的间隔
    DataCopyPadExtParams<P> copyPadParams{true, 0, 0, 0};

    DataCopyPad(hMixLocal, mmResGm_[offset], copyParams, copyPadParams);

    // TODO
    xInQueue_.EnQue(hMixLocal);
}

template <class T, class P>
__aicore__ inline void MhcPreKernel<T, P>::DataCopyOutInvRmsUb(uint32_t curMLen, uint32_t offsetM)
{
    invRmsOutQueue_.EnQue<P>(invRmsUb_);
    invRmsUb_ = invRmsOutQueue_.DeQue<P>();

    DataCopyParams copyParams;
    copyParams.blockCount = static_cast<uint16_t>(1);
    copyParams.blockLen = uint32_t(curMLen * sizeof(P));
    copyParams.srcStride = uint32_t(0);
    copyParams.dstStride = uint32_t(0);

    uint64_t offset = globalOffsetM_ + offsetM;

    DataCopyPad(invRmsGm_[offset], invRmsUb_, copyParams);
}

template <class T, class P>
__aicore__ inline void MhcPreKernel<T, P>::DataCopyOutToWorkSpace(LocalTensor<P> &x, uint32_t curMLen,
                                                                  uint32_t curNdLen, uint32_t offsetM,
                                                                  uint32_t offsetNd, LocalTensor<P> &aL1Buf)
{
    DataCopyParams copyParams;
    copyParams.blockCount = static_cast<uint16_t>(curMLen);
    copyParams.blockLen = uint32_t(curNdLen * sizeof(P));
    uint32_t alignedNdLen = Ceil(curNdLen, static_cast<uint32_t>(16)) * 16;
    copyParams.srcStride = uint32_t((alignedNdLen - curNdLen) * sizeof(P));
    copyParams.dstStride = uint32_t(0);

    uint64_t offset = offsetM * curNdLen;
    DataCopy(aL1Buf[offset], x, copyParams);
}

} // namespace MhcPre

#endif // __mhc_pre_KERNEL_H_
