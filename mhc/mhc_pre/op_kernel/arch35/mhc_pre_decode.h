/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
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

constexpr MicroAPI::CastTrait ctFp32To16 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
                                              MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
constexpr MicroAPI::CastTrait ctHalf2Fp32Zero = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
                                                 MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
constexpr MicroAPI::DivSpecificMode divMode = {MicroAPI::MaskMergeMode::ZEROING, true};

using namespace matmul;
using namespace AscendC;


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
    uint64_t totalLength = 0;  // 总长度 (batch * sequence 或 T)
    uint64_t nD = 0;           // n * D
    uint64_t fusionSize = 0;   // phi 的第二维
    float normEps = 0.0f;      // 归一化 epsilon
    float hcEps = 0.0f;        // hyper connection epsilon
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

// ========== 常量配置 ==========
// 分块处理相关配置


// 同步标志常量
static constexpr uint64_t SYNC_V0toV0 = 0x1;  // Vector -> Vector 同步标志
static constexpr uint64_t SYNC_V0toC = 0x2;   // Vector -> Cube 同步标志
static constexpr uint64_t SYNC_CtoC = 0x3;    // Cube -> Cube 同步标志
static constexpr uint64_t SYNC_CtoV1 = 0x4;   // Cube -> Vector 同步标志

// 其他常量
static constexpr uint32_t parallNum_ = 2;     // 并行数量

using aT = MatmulType<TPosition::GM, CubeFormat::ND, float32_t>;
using bT = MatmulType<TPosition::GM, CubeFormat::ND, float32_t, true>;
using cT = MatmulType<TPosition::GM, CubeFormat::ND, float32_t>;
using MT = matmul::MatmulImpl<aT, bT, cT>;

template <class T, class P>
class MhcPreKernel {
public:
    __aicore__ inline MhcPreKernel(MT &matmul) : mm(matmul) {}
    __aicore__ inline void Init(InitParams initParams);
    __aicore__ inline void Process();
    __aicore__ inline void AICProcess();
    __aicore__ inline void InitLocalBuffers();
    __aicore__ inline void VectorComputeOffset();
    __aicore__ inline void DataCopyX(uint32_t curMLen, uint32_t curNdLen, uint32_t offsetM, uint32_t offsetNd);
    __aicore__ inline void DataCopyOutInvRmsUb(uint32_t curMLen, uint32_t offsetM);
    __aicore__ inline void DataCopyOutToWorkSpace(LocalTensor<P> &x, uint32_t curMLen, uint32_t curNdLen, uint32_t offsetM, uint32_t offsetNd);
    __aicore__ inline void DataCopyGamma(uint32_t curNdLen, uint32_t offsetNd);
    __aicore__ inline void AIVPreLoad();
    __aicore__ inline void HMixProcess(uint64_t offsetT, uint64_t lenT);
    __aicore__ inline void AIV1ProcessHPost(uint64_t offsetT, uint64_t lenT);
    __aicore__ inline void AIV1ProcessHPre(uint64_t offsetT, uint64_t lenT);
    __aicore__ inline void AIV1Prologue(uint64_t offsetT, uint64_t lenT, uint64_t singleCoreOffset);
    __aicore__ inline void AIV1Process(uint64_t curBlock, uint64_t tBlockNum);
    __aicore__ inline void BiasCopyIn();
    __aicore__ inline void HMixCopyIn(uint64_t offset, uint64_t lenT);
    __aicore__ inline void AIV1GetHSliceOffset();
    __aicore__ inline void AIV1ProcessHIn(uint64_t offsetT, uint64_t lenT, uint64_t lenD);
    __aicore__ inline void VFDoV1ProcessHinForN4(__ubuf__ T* xInAddr, __ubuf__ T* hinOutAddr, uint32_t lenD, uint32_t tIdx);
    __aicore__ inline void VFDoV1ProcessHinForN6(__ubuf__ T* xInAddr, __ubuf__ T* hinOutAddr, uint32_t lenD, uint32_t tIdx);
    __aicore__ inline void VFDoV1ProcessHinForN8(__ubuf__ T* xInAddr, __ubuf__ T* hinOutAddr, uint32_t lenD, uint32_t tIdx);

    // V0
    __aicore__ inline void V0PostProcess();
    template <bool hasGamma, bool isFirstND>
    __aicore__ inline void VFDoV0ProcessXIn(__ubuf__ P *xDst, __ubuf__ P *invRmsDst, __ubuf__ T *xIn, __ubuf__ P *gamma, uint16_t mSize, uint16_t nSize);
    __aicore__ inline void VFDoV0ProcessInvRms(__ubuf__ P *invRms, uint16_t nSize, float scaleMean, float normEps);
    __aicore__ inline void V0Prologue();
    __aicore__ inline void DataCopyOutHPre(uint64_t offset, uint32_t totalElem);
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
    GlobalTensor<P> invRmsGm_; // 输出 inv_rms
    GlobalTensor<P> mmResGm_;  // 输出 mm_res
    GlobalTensor<P> hPreGm_;   // 输出 h_pre
    GlobalTensor<P> xFloatGm_;

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

    LocalTensor<float> oneUb_;

    TPipe *pipe_;
    MatrixInfo matrixInfo_;
    VectorOffsetParams vectorOffset_;

    const MhcPreTilingData *tiling_;

    // 运行时状态变量
    uint32_t chunTSize_ = 2;    // 1,2,3,...,6
    uint32_t chunNDSize_ = 320; // 320,640
    uint32_t v1ChunkDSize_ = 5120;
    uint32_t curSingleM_ = 2;  // 当前块的实际M长度（尾块可能更小）
    uint32_t coreIdx_ = 0;
    uint32_t subBlockIdx_ = 0;
    float scaleMean_ = 0.0f; // 1/nD
    uint64_t globalOffsetM_ = 0;
    uint32_t vectorCount_ = 0;
    uint32_t cubeCount_ = 0;
    uint64_t mmCount_ = 0;
    uint64_t vec1Count_ = 0;
    uint32_t minT_ = 1;
    uint32_t vectorCoreNum = 2;
    uint32_t V0_BASE_T = 1;      // V0 的 T 维分块
    uint64_t V1_BASE_T = 8;      // V1 的 T 维分块 TODO
    uint64_t V1_BASE_D = 32;     // V1 的 D 维分块 TODO
    uint32_t ND_LENGTH = 1024;   // nD 分块长度
};

template <class T, class P>
__aicore__ inline void MhcPreKernel<T, P>::Init(InitParams initParams)
{
    // 1. 绑定 GlobalTensor
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

    // 2. 获取 TilingData 并初始化
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
    // chunNDSize_ = tiling_->chunkNDSize;
    v1ChunkDSize_ = tiling_->v1ChunkDSize;
    hasGamma_ = (tiling_->hasGamma != 0);

    if (N_ == 4){
        minT_ = 2;
    }
    if (N_ == 6){
        minT_ = 4;
    }

    chunNDSize_ = Ceil(matrixInfo_.nD, coreNum_);
    chunTSize_ = Ceil(totalLength_, coreNum_);

    // V0_BASE_T = chunTSize_ / vectorCoreNum;      // V0 的 T 维分块
    // V1_BASE_T = chunTSize_ / vectorCoreNum;       // V1 的 T 维分块 TODO
    // V1_BASE_D = 32;     // V1 的 D 维分块 TODO
    // ND_LENGTH = (((8 * 1024) / V0_BASE_T) / 1024) * 1024;   // nD 分块长度

    mnConfig_.m = matrixInfo_.totalLength;
    mnConfig_.n = matrixInfo_.fusionSize;
    mnConfig_.k = matrixInfo_.nD;
    mnConfig_.singleCoreM = totalLength_;
    mnConfig_.singleCoreN = mnConfig_.n;
    mnConfig_.singleCoreK = chunNDSize_;
    mnConfig_.curSingleCoreM = mnConfig_.singleCoreM;
    mnConfig_.curSingleCoreN = mnConfig_.singleCoreN;
    mnConfig_.curSingleCoreK = mnConfig_.singleCoreK;
    curSingleM_ = chunTSize_;

    if (outFlag_) {
        mmResGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P *>(initParams.mm_res));
    }
    else {
        constexpr uint64_t kWorkspaceAlignBytes = 32UL;
        uint64_t xFloatWorkspaceBytes = totalLength_ * matrixInfo_.nD * sizeof(P);
        xFloatWorkspaceBytes = (xFloatWorkspaceBytes + kWorkspaceAlignBytes - 1UL) /
                            kWorkspaceAlignBytes * kWorkspaceAlignBytes;
        // 保持 mmRes workspace 偏移与xFloat 一致.
        mmResGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P *>(initParams.workspace + xFloatWorkspaceBytes));
    }

    // 3. 申请 UB
    pipe_ = initParams.tPipeIn;
    coreIdx_ = GetBlockIdx();
    subBlockIdx_ = GetSubBlockIdx();

    if ASCEND_IS_AIV {
        InitLocalBuffers();
        // 需移除：Duplicate<float>(oneUb_, 1.0f, curSingleM_ / 2);
        // 新增：oneUb_按Ceil(curSingleM_/2)长度填充，避免奇数长度时Div越界读取。
        // Duplicate<float>(oneUb_, 1.0f, Ceil(curSingleM_, 2)); // 全为1的tensor，用于计算倒数
    }

    SyncAll<false>();
}


template <class T, class P>
__aicore__ inline void MhcPreKernel<T, P>::InitLocalBuffers()
{
    pipe_->InitBuffer(xInQueue_, 2, 80 * 1024); // 2 × 80KB
    pipe_->InitBuffer(outQueue_, 2, 32 * 1024); // 2 × 32KB
    pipe_->InitBuffer(invRmsOutQueue_, 1, Ceil(curSingleM_, 2) * sizeof(P));

    if (hasGamma_) {
        pipe_->InitBuffer(gammaInQueue_, 1, ND_LENGTH * sizeof(P));
    }
    
    pipe_->InitBuffer(tmpBuff_, 20 * 1024); // 40KB

    pipe_->InitBuffer(biasInQue_, 1, mnConfig_.n * sizeof(P));
    pipe_->InitBuffer(alphaBuf_, mnConfig_.n * sizeof(P));
    alphaInUb_ = alphaBuf_.Get<P>();

    uint64_t buffOffset = 0;
    preOffsetBuf_ = tmpBuff_.GetWithOffset<uint32_t>(uint32_t(N_ * V1_BASE_T), buffOffset);
    buffOffset += N_ * V1_BASE_T * sizeof(uint32_t);
    buffOffset = Ceil(buffOffset, 32) * 32;

    postOffsetBuf_ = tmpBuff_.GetWithOffset<uint32_t>(uint32_t(N_ * V1_BASE_T), buffOffset);
    buffOffset += N_ * V1_BASE_T * sizeof(uint32_t);
    buffOffset = Ceil(buffOffset, 32) * 32;

    resOffsetBuf_ = tmpBuff_.GetWithOffset<uint32_t>(uint32_t(N_ * N_ * V1_BASE_T), buffOffset);
    buffOffset += N_ * N_ * V1_BASE_T * sizeof(uint32_t);
    buffOffset = Ceil(buffOffset, 32) * 32;

    // oneUb_ = tmpBuff_.GetWithOffset<float>(Ceil(curSingleM_, 2), buffOffset);
    // buffOffset += Ceil(curSingleM_, 2) * sizeof(float);
    // buffOffset = Ceil(buffOffset, 32) * 32;

    hPreBuff_ = tmpBuff_.GetWithOffset<P>(uint32_t(V1_BASE_T * N_), buffOffset);
    buffOffset += V1_BASE_T * N_ * sizeof(P);
    buffOffset = Ceil(buffOffset, 32) * 32;
    
    // inputBuff_ = tmpBuff_.GetWithOffset<P>(uint32_t(mnConfig_.n * V1_BASE_T * V1_BASE_D), buffOffset);
    // buffOffset += mnConfig_.n * V1_BASE_T * V1_BASE_D * sizeof(P);
    // buffOffset = Ceil(buffOffset, 32) * 32;

    // broadCastTmpUb_ = tmpBuff_.GetWithOffset<P>(uint32_t(mnConfig_.n * V1_BASE_T), buffOffset);
    // buffOffset += mnConfig_.n * V1_BASE_T * sizeof(P);
    // buffOffset = Ceil(buffOffset, 32) * 32;

    hPostBuff_ = tmpBuff_.GetWithOffset<P>(uint32_t(V1_BASE_T * N_), buffOffset);
    buffOffset += V1_BASE_T * N_ * sizeof(P);
    buffOffset = Ceil(buffOffset, 32) * 32;

    hResBuff_ = tmpBuff_.GetWithOffset<P>(uint32_t(V1_BASE_T * N_ * N_), buffOffset);
}

template <class T, class P>
__aicore__ inline void MhcPreKernel<T, P>::Process()
{
    if ASCEND_IS_AIV {
        coreIdx_ = GetBlockIdx() / 2;
        AIVPreLoad();

        uint32_t tBlockNum = Ceil(totalLength_, chunTSize_);
        if (coreIdx_ < tBlockNum) {
            globalOffsetM_ = coreIdx_ * chunTSize_;
            V0Prologue();
            AIV1Process(coreIdx_, tBlockNum);
        }
        else{
            AscendC::CrossCoreSetFlag<0x0, PIPE_MTE3>(SYNC_V0toV0);
            AscendC::CrossCoreWaitFlag(SYNC_V0toV0);
            AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(SYNC_V0toC);
        }
    }

    if ASCEND_IS_AIC {
        uint32_t ndBlockNum = Ceil(matrixInfo_.nD, chunNDSize_);
        if (coreIdx_ < ndBlockNum) {
            AICProcess();
        }
    }

    if ASCEND_IS_AIV {
        // V0PostProcess();
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
__aicore__ inline void MhcPreKernel<T, P>::AICProcess()
{
    AscendC::CrossCoreWaitFlag(SYNC_V0toC);
    // 尾块处理
    uint64_t offsetNd = coreIdx_ * chunNDSize_;
    if(offsetNd + mnConfig_.singleCoreK > mnConfig_.k){
        mnConfig_.curSingleCoreK = mnConfig_.k - offsetNd;
    }

    uint64_t Offset = chunNDSize_ * coreIdx_;
    uint64_t outOffset = mnConfig_.singleCoreM * mnConfig_.singleCoreN * coreIdx_;
    mm.SetOrgShape(mnConfig_.singleCoreM, mnConfig_.singleCoreN, mnConfig_.k);  // MNK
    mm.SetSingleShape(mnConfig_.singleCoreM, mnConfig_.singleCoreN, mnConfig_.curSingleCoreK); // SingleCoreMNK
    mm.SetTensorA(xFloatGm_[Offset]);
    mm.SetTensorB(phiGm_[Offset], true);

    mm.IterateAll(mmResGm_[outOffset], 0);
    mm.End();

    AscendC::CrossCoreSetFlag<0x0, PIPE_FIX>(SYNC_CtoC);
    AscendC::CrossCoreWaitFlag(SYNC_CtoC);
    AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(SYNC_CtoV1);
}

template <class T, class P>
__aicore__ inline void MhcPreKernel<T, P>::VectorComputeOffset()
{
    uint64_t aliginSingleM = Ceil(curSingleM_, 2); // 32Byte对齐
    vectorOffset_.singleCoreM = aliginSingleM <  curSingleM_ ? aliginSingleM : curSingleM_;
    if (subBlockIdx_ == 0) {
        vectorOffset_.offsetMStart = 0;
        vectorOffset_.offsetMEnd = vectorOffset_.singleCoreM;
    } else {
        vectorOffset_.offsetMStart = vectorOffset_.singleCoreM;
        vectorOffset_.singleCoreM = curSingleM_ - vectorOffset_.singleCoreM;
        vectorOffset_.offsetMEnd = curSingleM_;
    }
}

template <class T, class P>
template <bool hasGamma, bool isFirstND>
__aicore__ inline void MhcPreKernel<T, P>::VFDoV0ProcessXIn(__ubuf__ P *xDst, __ubuf__ P *invRmsDst, __ubuf__ T *xIn, __ubuf__ P *gamma, uint16_t mSize, uint16_t nSize)
{
    uint32_t eleNumPerVf = MhcPreUtils::GetVRegSize() / sizeof(P);
    uint32_t nSrcUbAligned =
        MhcPreUtils::Align(nSize, static_cast<uint16_t>(MhcPreUtils::UB_ALIGN_SIZE / sizeof(T)));
    uint32_t nDstUbAligned = MhcPreUtils::Align(nSize, static_cast<uint16_t>(MhcPreUtils::UB_ALIGN_SIZE / sizeof(P)));
    uint16_t nLoopCnt = MhcPreUtils::CeilDiv(nSize, eleNumPerVf);
    __VEC_SCOPE__
    {
        MicroAPI::MaskReg mask = MicroAPI::CreateMask<P>();
        for (uint16_t mIdx = 0; mIdx < mSize; mIdx++) {
            uint32_t elementNum = nSize;
            MicroAPI::RegTensor<P> sumReg;
            if constexpr (isFirstND) {
                MicroAPI::Duplicate(sumReg, 0);
            } else {
                MicroAPI::Load(sumReg, invRmsDst + mIdx);
            }
            for (uint16_t vfBlockIdx = 0; vfBlockIdx < nLoopCnt; vfBlockIdx++) {
                MicroAPI::RegTensor<T> xInReg;
                MicroAPI::RegTensor<P> gammaReg;
                MicroAPI::RegTensor<P> xFp32Reg, xMulReg, xSquaReg;
                MicroAPI::RegTensor<P> tmpSumReg;
                
                uint32_t xInOffset = mIdx * nSrcUbAligned + vfBlockIdx * eleNumPerVf;
                MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(xInReg, xIn + xInOffset);
                MicroAPI::MaskReg maskN4B32 = MicroAPI::UpdateMask<P>(elementNum);
                MicroAPI::Cast<float, T, ctHalf2Fp32Zero>(xFp32Reg, xInReg, maskN4B32);
                if constexpr (hasGamma) { 
                    MicroAPI::LoadAlign(gammaReg, gamma + vfBlockIdx * eleNumPerVf);
                    MicroAPI::Mul(xMulReg, gammaReg, xFp32Reg, maskN4B32);
                } else {
                    xMulReg = xFp32Reg;
                }
                uint32_t dstUbOffset = mIdx * nDstUbAligned + vfBlockIdx * eleNumPerVf;
                MicroAPI::StoreAlign(xDst + dstUbOffset, xMulReg, maskN4B32);

                MicroAPI::Mul(xSquaReg, xFp32Reg, xFp32Reg, maskN4B32);

                MicroAPI::Reduce<MicroAPI::ReduceType::SUM>(tmpSumReg, xSquaReg, maskN4B32);
                MicroAPI::Add(sumReg, sumReg, tmpSumReg, maskN4B32);
            }
            
            MicroAPI::Store(invRmsDst + mIdx, sumReg, 1);
        }
    }
}

template <class T, class P>
__aicore__ inline void MhcPreKernel<T, P>::VFDoV0ProcessInvRms(__ubuf__ P *invRms, uint16_t nSize, float scaleMean, float normEps)
{
    uint32_t eleNumPerVf = MhcPreUtils::GetVRegSize() / sizeof(P);
    uint32_t nUbAligned = MhcPreUtils::Align(nSize, static_cast<uint16_t>(MhcPreUtils::UB_ALIGN_SIZE / sizeof(P)));
    uint16_t nLoopCnt = MhcPreUtils::CeilDiv(nSize, eleNumPerVf);
    __VEC_SCOPE__
    {
        uint32_t elementNum = nSize;
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
__aicore__ inline void MhcPreKernel<T, P>::V0Prologue()
{
    VectorComputeOffset();
    if (vectorOffset_.singleCoreM == 0) {
        AscendC::CrossCoreSetFlag<0x0, PIPE_MTE3>(SYNC_V0toV0);
        AscendC::CrossCoreWaitFlag(SYNC_V0toV0);
        AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(SYNC_V0toC);
        return;
    }

    for (uint32_t offsetNd = 0; offsetNd < matrixInfo_.nD; offsetNd += ND_LENGTH) {
        uint32_t curNdLen = ND_LENGTH;
        if (offsetNd + ND_LENGTH >= matrixInfo_.nD) {
            curNdLen = matrixInfo_.nD - offsetNd;
        }

        for (uint32_t offsetM = vectorOffset_.offsetMStart; offsetM < vectorOffset_.offsetMEnd; offsetM += V0_BASE_T) {
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

            if (hasGamma_) {
                gammaUb_ = gammaInQueue_.AllocTensor<P>();
                DataCopyGamma(curNdLen, offsetNd);
                gammaUb_ = gammaInQueue_.DeQue<P>();

                if (offsetNd == 0) {
                    VFDoV0ProcessXIn<true, true>((__ubuf__ P *)aL1Ub.GetPhyAddr(), (__ubuf__ P *)invRmsUb.GetPhyAddr(), (__ubuf__ T *)xLocal_.GetPhyAddr(), (__ubuf__ P *)gammaUb_.GetPhyAddr(), curMLen, curNdLen);
                } else {
                    VFDoV0ProcessXIn<true, false>((__ubuf__ P *)aL1Ub.GetPhyAddr(), (__ubuf__ P *)invRmsUb.GetPhyAddr(), (__ubuf__ T *)xLocal_.GetPhyAddr(), (__ubuf__ P *)gammaUb_.GetPhyAddr(), curMLen, curNdLen);
                }
                gammaInQueue_.FreeTensor(gammaUb_);
            } else {
                if (offsetNd == 0) {
                    VFDoV0ProcessXIn<false, true>((__ubuf__ P *)aL1Ub.GetPhyAddr(), (__ubuf__ P *)invRmsUb.GetPhyAddr(), (__ubuf__ T *)xLocal_.GetPhyAddr(), nullptr, curMLen, curNdLen);
                } else {
                    VFDoV0ProcessXIn<false, false>((__ubuf__ P *)aL1Ub.GetPhyAddr(), (__ubuf__ P *)invRmsUb.GetPhyAddr(), (__ubuf__ T *)xLocal_.GetPhyAddr(), nullptr, curMLen, curNdLen);
                }
            }

            outQueue_.EnQue<P>(aL1Ub);
            aL1Ub = outQueue_.DeQue<P>();
            DataCopyOutToWorkSpace(aL1Ub, curMLen, curNdLen, offsetM, offsetNd);

            xInQueue_.FreeTensor(xLocal_);
            outQueue_.FreeTensor(aL1Ub);
        }
    }
    AscendC::CrossCoreSetFlag<0x0, PIPE_MTE3>(SYNC_V0toV0);
    AscendC::CrossCoreWaitFlag(SYNC_V0toV0);
    AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(SYNC_V0toC);

    VFDoV0ProcessInvRms((__ubuf__ P *)invRmsUb_.GetPhyAddr(), vectorOffset_.singleCoreM, scaleMean_, matrixInfo_.normEps);
    DataCopyOutInvRmsUb(vectorOffset_.singleCoreM, vectorOffset_.offsetMStart);
}

template <class T, class P>
__aicore__ inline void MhcPreKernel<T, P>::V0PostProcess()
{
    VFDoV0ProcessInvRms((__ubuf__ P *)invRmsUb_.GetPhyAddr(), vectorOffset_.singleCoreM, scaleMean_, matrixInfo_.normEps);
    DataCopyOutInvRmsUb(vectorOffset_.singleCoreM, vectorOffset_.offsetMStart);
}
template <class T, class P>
__aicore__ inline void MhcPreKernel<T, P>::AIV1Process(uint64_t curBlock, uint64_t tBlockNum)
{
    curSingleM_= chunTSize_;
    if (curBlock == tBlockNum - 1) {
        curSingleM_= matrixInfo_.totalLength - curBlock * chunTSize_;
    }
    VectorComputeOffset();
    if (vectorOffset_.singleCoreM <= 0) {
        return;
    }
    AscendC::CrossCoreWaitFlag(SYNC_CtoV1);
    uint64_t lenT = 0;
    uint64_t lenD = 0;
    uint64_t singleCoreOffset = 0;
    for (int offsetT = vectorOffset_.offsetMStart; offsetT < vectorOffset_.offsetMEnd; offsetT += V1_BASE_T) {
        lenT = V1_BASE_T < vectorOffset_.offsetMEnd - offsetT ? V1_BASE_T : vectorOffset_.offsetMEnd - offsetT;
        lenD = v1ChunkDSize_;
        AIV1Prologue(offsetT, lenT, singleCoreOffset);

        AIV1ProcessHPost(offsetT, lenT);
        
        AIV1ProcessHPre(offsetT, lenT);
        AIV1ProcessHIn(offsetT, lenT, lenD); 

        singleCoreOffset += V1_BASE_T;
    }
}

template <class T, class P>
__aicore__ inline void MhcPreKernel<T, P>::AIV1Prologue(uint64_t offsetT, uint64_t lenT, uint64_t singleCoreOffset)
{
    uint64_t offset = globalOffsetM_ + offsetT;
    uint64_t HMixOffset = 0;
    LocalTensor<P> hResOutLocal = outQueue_.AllocTensor<P>();
    if (outFlag_) {
        HMixOffset = (globalOffsetM_ + offsetT) * mnConfig_.n;
    } else {
        HMixOffset = mnConfig_.singleCoreM * mnConfig_.singleCoreN * ((vec1Count_ % 2) * coreNum_ + coreIdx_) + offsetT * mnConfig_.n;
    }
    HMixProcess(offsetT, lenT);

    // FIX：重新搬进来
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);

    HMixCopyIn(HMixOffset, lenT);
    matmulRes_ = xInQueue_.DeQue<P>(); 
    
    __ubuf__ P* matmulPtr = (__ubuf__ P*)matmulRes_.GetPhyAddr();
    __ubuf__ P* invRmsPtr = (__ubuf__ P*)invRmsUb_.GetPhyAddr();
    __ubuf__ P* biasInPtr = (__ubuf__ P*)biasInUb_.GetPhyAddr();
    __ubuf__ P* alphaInPtr = (__ubuf__ P*)alphaInUb_.GetPhyAddr();
    __ubuf__ P* hPreBuffPtr = (__ubuf__ P*)hPreBuff_.GetPhyAddr();
    __ubuf__ P* hPostBuffPtr = (__ubuf__ P*)hPostBuff_.GetPhyAddr();
    __ubuf__ P* hResOutLocalPtr = (__ubuf__ P*)hResOutLocal.GetPhyAddr();

    uint32_t eleNumPerVf = 256 / sizeof(P);
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<P> matmulResReg;
        MicroAPI::RegTensor<P> invRmsReg;
        MicroAPI::RegTensor<P> invRmsBroadReg;
        MicroAPI::RegTensor<P> biasInReg;
        MicroAPI::RegTensor<P> alphaInReg;

        for (uint16_t tIdx = 0; tIdx < static_cast<uint16_t>(lenT); tIdx++) {
            MicroAPI::Load<P>(invRmsReg, invRmsPtr + singleCoreOffset + tIdx);

            uint32_t loopCntPerN = (mnConfig_.n + eleNumPerVf - 1) / eleNumPerVf;
            uint32_t curLen = mnConfig_.n;
            for (uint16_t vfBlockIdx = 0; vfBlockIdx < (uint16_t)loopCntPerN; vfBlockIdx++) {
                uint32_t maskOffset = tIdx * mnConfig_.n + vfBlockIdx * eleNumPerVf;
                uint32_t alphaOffset = vfBlockIdx * eleNumPerVf;
                uint32_t biasOffset = vfBlockIdx * eleNumPerVf;
                MicroAPI::MaskReg curMask = MicroAPI::UpdateMask<P>(curLen);

                MicroAPI::LoadAlign<P>(matmulResReg, matmulPtr + maskOffset);
                MicroAPI::LoadAlign<P>(biasInReg, biasInPtr + biasOffset);
                MicroAPI::LoadAlign<P>(alphaInReg, alphaInPtr + alphaOffset);

                MicroAPI::Duplicate(invRmsBroadReg, invRmsReg, curMask);
                MicroAPI::Mul(matmulResReg, matmulResReg, invRmsBroadReg, curMask);

                MicroAPI::Mul(matmulResReg, matmulResReg, alphaInReg, curMask);

                MicroAPI::Add(matmulResReg, matmulResReg, biasInReg, curMask);

                MicroAPI::StoreAlign<P>(matmulPtr + maskOffset, matmulResReg, curMask);
            }
        }
    }

    

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
    __ubuf__ P *hPreBuffAddr = (__ubuf__ P *)hPreBuff_.GetPhyAddr();
    uint32_t totalElem = lenT * N_;
    uint32_t eleNumPerVf = 64;
    uint16_t nLoopCnt = Ceil(totalElem, eleNumPerVf);
    uint32_t curElemCnt = totalElem;
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<P> hPreReg;
        MicroAPI::RegTensor<P> negReg, expReg, addOneReg, sigmoidReg, resultReg, oneReg;

        for (uint16_t vfBlockIdx = 0; vfBlockIdx < nLoopCnt; vfBlockIdx++) {
            MicroAPI::MaskReg mask = MicroAPI::UpdateMask<P>(curElemCnt);
            MicroAPI::LoadAlign(hPreReg, hPreBuffAddr + vfBlockIdx * eleNumPerVf);
            MicroAPI::Neg(negReg, hPreReg, mask);
            MicroAPI::Exp(expReg, negReg, mask);
            MicroAPI::Adds(addOneReg, expReg, static_cast<P>(1.0), mask);
            MicroAPI::Duplicate(oneReg, static_cast<P>(1.0), mask);
            MicroAPI::Div<P, &divMode>(sigmoidReg, oneReg, addOneReg, mask);

            MicroAPI::Adds(resultReg, sigmoidReg, matrixInfo_.hcEps, mask);
            MicroAPI::StoreAlign(hPreBuffAddr + vfBlockIdx * eleNumPerVf, resultReg, mask);
        }
    }

    if (outFlag_) {
        uint64_t offset = globalOffsetM_ + offsetT;
        DataCopyOutHPre(offset, totalElem);
    }
}

template <class T, class P>
__aicore__ inline void MhcPreKernel<T, P>::AIV1ProcessHIn(uint64_t offsetT, uint64_t lenT, uint64_t lenD)
{
    // auto hInDealBuf = inputBuff_;
    for (uint32_t tIdx = 0; tIdx < lenT; tIdx++) { 
        for (int offsetD = 0; offsetD < D_; offsetD += v1ChunkDSize_) { 
            
            lenD = v1ChunkDSize_ < D_ - offsetD ? v1ChunkDSize_ : D_ - offsetD; 
            LocalTensor<T> xIn = xInQueue_.AllocTensor<T>();
            DataCopyExtParams xInCopyParams;
            xInCopyParams.blockCount = static_cast<uint16_t>(N_);
            xInCopyParams.blockLen = uint32_t(lenD * sizeof(T));
            xInCopyParams.srcStride = uint32_t((D_ - lenD) * sizeof(T));
            xInCopyParams.dstStride = 0;

            uint32_t rightPadNum = Ceil(lenD, 16) * 16 - lenD;
            DataCopyPadExtParams<T> padParams{true, 0, static_cast<uint8_t>(rightPadNum), 0};
            uint64_t xGmOffset = (globalOffsetM_ + offsetT + tIdx) * N_ * D_ + offsetD;
            DataCopyPad(xIn, xGm_[xGmOffset], xInCopyParams, padParams);

            xInQueue_.EnQue(xIn);
            xIn = xInQueue_.DeQue<T>();
            LocalTensor<T> hinOut = outQueue_.AllocTensor<T>();
            __ubuf__ T* xInAddr = (__ubuf__ T*)xIn.GetPhyAddr();
            __ubuf__ T* hinOutAddr = (__ubuf__ T*)hinOut.GetPhyAddr();  
            switch(N_) {
                case 4 :
                        VFDoV1ProcessHinForN4(xInAddr, hinOutAddr, lenD, tIdx);
                        break;
                case 6 :
                        VFDoV1ProcessHinForN6(xInAddr, hinOutAddr, lenD, tIdx);
                        break;
                case 8 :
                        VFDoV1ProcessHinForN8(xInAddr, hinOutAddr, lenD, tIdx);
                        break;
                default:
                        break;
            }

            outQueue_.EnQue(hinOut);
            hinOut = outQueue_.DeQue<T>();
            DataCopyExtParams copyParams;
            copyParams.blockCount = 1;
            copyParams.blockCount = 1; // 行数
            copyParams.blockLen = uint32_t(lenD * sizeof(T));
            copyParams.srcStride = 0;
            copyParams.dstStride = 0;

            DataCopyPad(hinGm_[(globalOffsetM_ + offsetT + tIdx) * D_ + offsetD], hinOut, copyParams);
            
            xInQueue_.FreeTensor(xIn);
            outQueue_.FreeTensor(hinOut);
        }
    } 
}

template <class T, class P>
__aicore__ inline void MhcPreKernel<T, P>::AIV1ProcessHPost(uint64_t offsetT, uint64_t lenT)
{
    uint64_t offset = globalOffsetM_ + offsetT;
    LocalTensor<P> hPostOutLocal = outQueue_.AllocTensor<P>();
    __ubuf__ P *hPostBuffAddr = (__ubuf__ P *)hPostBuff_.GetPhyAddr();
    __ubuf__ P *hPostOutAddr = (__ubuf__ P *)hPostOutLocal.GetPhyAddr();
    uint32_t totalElem = lenT * N_;
    uint32_t regCapacityFp32 = 64;
    uint16_t nLoopCnt = Ceil(totalElem, regCapacityFp32);
    float scalarValue = 2.0;
    uint32_t curElemCnt = totalElem;

    __VEC_SCOPE__
    {
        for (uint16_t vfBlockIdx = 0; vfBlockIdx < nLoopCnt; ++vfBlockIdx) {
            uint32_t elemOffset = vfBlockIdx * regCapacityFp32;
            MicroAPI::MaskReg mask = MicroAPI::UpdateMask<P>(curElemCnt);
            MicroAPI::RegTensor<P> hPostReg;
            MicroAPI::RegTensor<P> negReg, expReg, addOneReg, sigmoidReg, resultReg, oneReg;
            
            MicroAPI::LoadAlign(hPostReg, hPostBuffAddr + elemOffset);
            MicroAPI::Neg(negReg, hPostReg, mask);
            MicroAPI::Exp(expReg, negReg, mask);
            MicroAPI::Adds(addOneReg, expReg, 1.0f, mask);
            MicroAPI::Duplicate(oneReg, 1.0f, mask);
            MicroAPI::Div<P, &divMode>(sigmoidReg, oneReg, addOneReg, mask);

            MicroAPI::Muls(resultReg, sigmoidReg, scalarValue, mask);
            MicroAPI::StoreAlign(hPostOutAddr + elemOffset, resultReg, mask);
        }
    }
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
    for(uint64_t i = 0; i < N_; ++i){
        alphaInUb_.SetValue(i, alphaPre);
        alphaInUb_.SetValue(i + N_, alphaPost);
        for(uint64_t j = 0; j < N_; ++j){
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
__aicore__ inline void MhcPreKernel<T, P>::DataCopyX(uint32_t curMLen, uint32_t curNdLen, uint32_t offsetM, uint32_t offsetNd)
{
    DataCopyExtParams copyParams;
    copyParams.blockCount = static_cast<uint16_t>(curMLen);
    copyParams.blockLen = uint32_t(curNdLen * sizeof(T));
    copyParams.srcStride = uint32_t((matrixInfo_.nD - curNdLen) * sizeof(T));
    copyParams.dstStride = uint32_t(0);

    uint32_t rightPadNum = Ceil(curNdLen, 16) * 16 - curNdLen;
    DataCopyPadExtParams<T> padParams{true, 0, static_cast<uint8_t>(rightPadNum), 0};

    uint64_t offset = (globalOffsetM_ + offsetM) * matrixInfo_.nD + offsetNd;
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
__aicore__ inline void MhcPreKernel<T, P>::HMixProcess(uint64_t offsetT, uint64_t lenT)
{
    uint32_t mmResGmBlockNum = Ceil(matrixInfo_.nD, chunNDSize_);    // coreNum
    uint64_t HMixOffset = (globalOffsetM_ + offsetT) * mnConfig_.n;

    LocalTensor<P> hMixLocal = xInQueue_.AllocTensor<P>();
    DataCopyExtParams copyParams;
    copyParams.blockCount = static_cast<uint16_t>(mmResGmBlockNum); // 行数
    copyParams.blockLen = uint32_t(lenT * mnConfig_.n * sizeof(P));
    copyParams.srcStride = uint32_t((mnConfig_.curSingleCoreM * mnConfig_.curSingleCoreN - lenT * mnConfig_.n) * sizeof(P)); // 相邻块的间隔
    copyParams.dstStride = uint32_t(0);                                                   // 相邻块的间隔
    DataCopyPadExtParams<P> copyPadParams{true, 0, 0, 0};
    DataCopyPad(hMixLocal, mmResGm_[HMixOffset], copyParams, copyPadParams);
    xInQueue_.EnQue(hMixLocal);
    hMixLocal = xInQueue_.DeQue<P>();

    uint64_t addOffset = 0;
    for(uint32_t mmResGmBlockIdx = 1; mmResGmBlockIdx < mmResGmBlockNum; mmResGmBlockIdx++)
    {
        addOffset += lenT * mnConfig_.n;
        Add(hMixLocal, hMixLocal, hMixLocal[addOffset], lenT * mnConfig_.n);
        PipeBarrier<PIPE_V>();
    }
    // Add计算完后copyout
    SetFlag<HardEvent::V_MTE3>(EVENT_ID4);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID4);
    DataCopyExtParams MixOutCopyParams;
    MixOutCopyParams.blockCount = static_cast<uint16_t>(1); // 行数
    MixOutCopyParams.blockLen = uint32_t(lenT * mnConfig_.n * sizeof(P));
    MixOutCopyParams.srcStride = uint32_t(0); // 相邻块的间隔
    MixOutCopyParams.dstStride = uint32_t(0);                                                   // 相邻块的间隔
    DataCopyPad(mmResGm_[HMixOffset], hMixLocal, MixOutCopyParams);
    // xInQueue_.EnQue(hMixLocal);
    xInQueue_.FreeTensor(hMixLocal);
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
__aicore__ inline void MhcPreKernel<T, P>::DataCopyOutToWorkSpace(LocalTensor<P> &x, uint32_t curMLen, uint32_t curNdLen, uint32_t offsetM, uint32_t offsetNd)
{
    // outQueue_.EnQue<P>(x);
    // x = outQueue_.DeQue<P>();
    
    DataCopyParams copyParams;
    copyParams.blockCount = static_cast<uint16_t>(curMLen);
    copyParams.blockLen = uint32_t(curNdLen * sizeof(P));
    copyParams.srcStride = uint32_t(0);
    copyParams.dstStride = uint32_t((matrixInfo_.nD - curNdLen) * sizeof(P));

    uint64_t offset = (globalOffsetM_ + offsetM) * matrixInfo_.nD + offsetNd;
    DataCopyPad(xFloatGm_[offset], x, copyParams);
}

template <class T, class P>
__aicore__ inline void MhcPreKernel<T, P>::DataCopyOutHPre(uint64_t offset, uint32_t totalElem)
{
    DataCopyExtParams copyParams;
    copyParams.blockCount = static_cast<uint16_t>(1);
    copyParams.blockLen = uint32_t(totalElem * sizeof(P));
    copyParams.srcStride = uint32_t(0);
    copyParams.dstStride = uint32_t(0);
    SetFlag<HardEvent::V_MTE3>(EVENT_ID2);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID2);
    DataCopyPad(hPreGm_[offset * N_], hPreBuff_, copyParams);
}

template <class T, class P>
__aicore__ inline void MhcPreKernel<T, P>::VFDoV1ProcessHinForN4(__ubuf__ T* xInAddr, __ubuf__ T* hinOutAddr, uint32_t lenD, uint32_t tIdx)
{
    uint16_t eleNumPerVf = 64;
    uint16_t dLoopCnt = (lenD + eleNumPerVf - 1) / eleNumPerVf;
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<P> xFp32Reg;
        MicroAPI::RegTensor<T> xInReg;
        MicroAPI::RegTensor<P> accFp32Reg;
        MicroAPI::RegTensor<T> outB16Reg;
        uint32_t curLenD = lenD;

        P hPreValue0 = hPreBuff_.GetValue(tIdx * N_ + 0);
        P hPreValue1 = hPreBuff_.GetValue(tIdx * N_ + 1);
        P hPreValue2 = hPreBuff_.GetValue(tIdx * N_ + 2);
        P hPreValue3 = hPreBuff_.GetValue(tIdx * N_ + 3);
        for (uint16_t dIdx = 0; dIdx < dLoopCnt; dIdx++) {
            MicroAPI::MaskReg mask = MicroAPI::UpdateMask<P>(curLenD);
            MicroAPI::Duplicate<P>(accFp32Reg, static_cast<P>(0.0f), mask);

            MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(xInReg, xInAddr + 0 * lenD + dIdx * eleNumPerVf);
            MicroAPI::Cast<float, T, ctHalf2Fp32Zero>(xFp32Reg, xInReg, mask);
            MicroAPI::Muls(xFp32Reg, xFp32Reg, hPreValue0, mask);
            MicroAPI::Add(accFp32Reg, accFp32Reg, xFp32Reg, mask);

            MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(xInReg, xInAddr + 1 * lenD + dIdx * eleNumPerVf);
            MicroAPI::Cast<float, T, ctHalf2Fp32Zero>(xFp32Reg, xInReg, mask);
            MicroAPI::Muls(xFp32Reg, xFp32Reg, hPreValue1, mask);
            MicroAPI::Add(accFp32Reg, accFp32Reg, xFp32Reg, mask);

            MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(xInReg, xInAddr + 2 * lenD + dIdx * eleNumPerVf);
            MicroAPI::Cast<float, T, ctHalf2Fp32Zero>(xFp32Reg, xInReg, mask);
            MicroAPI::Muls(xFp32Reg, xFp32Reg, hPreValue2, mask);
            MicroAPI::Add(accFp32Reg, accFp32Reg, xFp32Reg, mask);

            MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(xInReg, xInAddr + 3 * lenD + dIdx * eleNumPerVf);
            MicroAPI::Cast<float, T, ctHalf2Fp32Zero>(xFp32Reg, xInReg, mask);
            MicroAPI::Muls(xFp32Reg, xFp32Reg, hPreValue3, mask);
            MicroAPI::Add(accFp32Reg, accFp32Reg, xFp32Reg, mask);

            MicroAPI::Cast<T, P, ctFp32To16>(outB16Reg, accFp32Reg, mask);
            MicroAPI::StoreAlign<T, MicroAPI::StoreDist::DIST_PACK_B32>(hinOutAddr + dIdx * eleNumPerVf, outB16Reg, mask);
        }
    }
}

template <class T, class P>
__aicore__ inline void MhcPreKernel<T, P>::VFDoV1ProcessHinForN6(__ubuf__ T* xInAddr, __ubuf__ T* hinOutAddr, uint32_t lenD, uint32_t tIdx)
{
    uint16_t eleNumPerVf = 64;
    uint16_t dLoopCnt = (lenD + eleNumPerVf - 1) / eleNumPerVf;
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<P> xFp32Reg;
        MicroAPI::RegTensor<T> xInReg;
        MicroAPI::RegTensor<P> accFp32Reg;
        MicroAPI::RegTensor<T> outB16Reg;
        uint32_t curLenD = lenD;

        P hPreValue0 = hPreBuff_.GetValue(tIdx * N_ + 0);
        P hPreValue1 = hPreBuff_.GetValue(tIdx * N_ + 1);
        P hPreValue2 = hPreBuff_.GetValue(tIdx * N_ + 2);
        P hPreValue3 = hPreBuff_.GetValue(tIdx * N_ + 3);
        P hPreValue4 = hPreBuff_.GetValue(tIdx * N_ + 4);
        P hPreValue5 = hPreBuff_.GetValue(tIdx * N_ + 5);
        for (uint16_t dIdx = 0; dIdx < dLoopCnt; dIdx++) {
            MicroAPI::MaskReg mask = MicroAPI::UpdateMask<P>(curLenD);
            MicroAPI::Duplicate<P>(accFp32Reg, static_cast<P>(0.0f), mask);

            MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(xInReg, xInAddr + 0 * lenD + dIdx * eleNumPerVf);
            MicroAPI::Cast<float, T, ctHalf2Fp32Zero>(xFp32Reg, xInReg, mask);
            MicroAPI::Muls(xFp32Reg, xFp32Reg, hPreValue0, mask);
            MicroAPI::Add(accFp32Reg, accFp32Reg, xFp32Reg, mask);

            MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(xInReg, xInAddr + 1 * lenD + dIdx * eleNumPerVf);
            MicroAPI::Cast<float, T, ctHalf2Fp32Zero>(xFp32Reg, xInReg, mask);
            MicroAPI::Muls(xFp32Reg, xFp32Reg, hPreValue1, mask);
            MicroAPI::Add(accFp32Reg, accFp32Reg, xFp32Reg, mask);

            MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(xInReg, xInAddr + 2 * lenD + dIdx * eleNumPerVf);
            MicroAPI::Cast<float, T, ctHalf2Fp32Zero>(xFp32Reg, xInReg, mask);
            MicroAPI::Muls(xFp32Reg, xFp32Reg, hPreValue2, mask);
            MicroAPI::Add(accFp32Reg, accFp32Reg, xFp32Reg, mask);

            MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(xInReg, xInAddr + 3 * lenD + dIdx * eleNumPerVf);
            MicroAPI::Cast<float, T, ctHalf2Fp32Zero>(xFp32Reg, xInReg, mask);
            MicroAPI::Muls(xFp32Reg, xFp32Reg, hPreValue3, mask);
            MicroAPI::Add(accFp32Reg, accFp32Reg, xFp32Reg, mask);

            MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(xInReg, xInAddr + 4 * lenD + dIdx * eleNumPerVf);
            MicroAPI::Cast<float, T, ctHalf2Fp32Zero>(xFp32Reg, xInReg, mask);
            MicroAPI::Muls(xFp32Reg, xFp32Reg, hPreValue4, mask);
            MicroAPI::Add(accFp32Reg, accFp32Reg, xFp32Reg, mask);

            MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(xInReg, xInAddr + 5 * lenD + dIdx * eleNumPerVf);
            MicroAPI::Cast<float, T, ctHalf2Fp32Zero>(xFp32Reg, xInReg, mask);
            MicroAPI::Muls(xFp32Reg, xFp32Reg, hPreValue5, mask);
            MicroAPI::Add(accFp32Reg, accFp32Reg, xFp32Reg, mask);

            MicroAPI::Cast<T, P, ctFp32To16>(outB16Reg, accFp32Reg, mask);
            MicroAPI::StoreAlign<T, MicroAPI::StoreDist::DIST_PACK_B32>(hinOutAddr + dIdx * eleNumPerVf, outB16Reg, mask);
        }
    }
}

template <class T, class P>
__aicore__ inline void MhcPreKernel<T, P>::VFDoV1ProcessHinForN8(__ubuf__ T* xInAddr, __ubuf__ T* hinOutAddr, uint32_t lenD, uint32_t tIdx)
{
    uint16_t eleNumPerVf = 64;
    uint16_t dLoopCnt = (lenD + eleNumPerVf - 1) / eleNumPerVf;
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<P> xFp32Reg;
        MicroAPI::RegTensor<T> xInReg;
        MicroAPI::RegTensor<P> accFp32Reg;
        MicroAPI::RegTensor<T> outB16Reg;
        uint32_t curLenD = lenD;

        P hPreValue0 = hPreBuff_.GetValue(tIdx * N_ + 0);
        P hPreValue1 = hPreBuff_.GetValue(tIdx * N_ + 1);
        P hPreValue2 = hPreBuff_.GetValue(tIdx * N_ + 2);
        P hPreValue3 = hPreBuff_.GetValue(tIdx * N_ + 3);
        P hPreValue4 = hPreBuff_.GetValue(tIdx * N_ + 4);
        P hPreValue5 = hPreBuff_.GetValue(tIdx * N_ + 5);
        P hPreValue6 = hPreBuff_.GetValue(tIdx * N_ + 6);
        P hPreValue7 = hPreBuff_.GetValue(tIdx * N_ + 7);
        for (uint16_t dIdx = 0; dIdx < dLoopCnt; dIdx++) {
            MicroAPI::MaskReg mask = MicroAPI::UpdateMask<P>(curLenD);
            MicroAPI::Duplicate<P>(accFp32Reg, static_cast<P>(0.0f), mask);

            MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(xInReg, xInAddr + 0 * lenD + dIdx * eleNumPerVf);
            MicroAPI::Cast<float, T, ctHalf2Fp32Zero>(xFp32Reg, xInReg, mask);
            MicroAPI::Muls(xFp32Reg, xFp32Reg, hPreValue0, mask);
            MicroAPI::Add(accFp32Reg, accFp32Reg, xFp32Reg, mask);

            MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(xInReg, xInAddr + 1 * lenD + dIdx * eleNumPerVf);
            MicroAPI::Cast<float, T, ctHalf2Fp32Zero>(xFp32Reg, xInReg, mask);
            MicroAPI::Muls(xFp32Reg, xFp32Reg, hPreValue1, mask);
            MicroAPI::Add(accFp32Reg, accFp32Reg, xFp32Reg, mask);

            MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(xInReg, xInAddr + 2 * lenD + dIdx * eleNumPerVf);
            MicroAPI::Cast<float, T, ctHalf2Fp32Zero>(xFp32Reg, xInReg, mask);
            MicroAPI::Muls(xFp32Reg, xFp32Reg, hPreValue2, mask);
            MicroAPI::Add(accFp32Reg, accFp32Reg, xFp32Reg, mask);

            MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(xInReg, xInAddr + 3 * lenD + dIdx * eleNumPerVf);
            MicroAPI::Cast<float, T, ctHalf2Fp32Zero>(xFp32Reg, xInReg, mask);
            MicroAPI::Muls(xFp32Reg, xFp32Reg, hPreValue3, mask);
            MicroAPI::Add(accFp32Reg, accFp32Reg, xFp32Reg, mask);

            MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(xInReg, xInAddr + 4 * lenD + dIdx * eleNumPerVf);
            MicroAPI::Cast<float, T, ctHalf2Fp32Zero>(xFp32Reg, xInReg, mask);
            MicroAPI::Muls(xFp32Reg, xFp32Reg, hPreValue4, mask);
            MicroAPI::Add(accFp32Reg, accFp32Reg, xFp32Reg, mask);

            MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(xInReg, xInAddr + 5 * lenD + dIdx * eleNumPerVf);
            MicroAPI::Cast<float, T, ctHalf2Fp32Zero>(xFp32Reg, xInReg, mask);
            MicroAPI::Muls(xFp32Reg, xFp32Reg, hPreValue5, mask);
            MicroAPI::Add(accFp32Reg, accFp32Reg, xFp32Reg, mask);

            MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(xInReg, xInAddr + 6 * lenD + dIdx * eleNumPerVf);
            MicroAPI::Cast<float, T, ctHalf2Fp32Zero>(xFp32Reg, xInReg, mask);
            MicroAPI::Muls(xFp32Reg, xFp32Reg, hPreValue6, mask);
            MicroAPI::Add(accFp32Reg, accFp32Reg, xFp32Reg, mask);

            MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(xInReg, xInAddr + 7 * lenD + dIdx * eleNumPerVf);
            MicroAPI::Cast<float, T, ctHalf2Fp32Zero>(xFp32Reg, xInReg, mask);
            MicroAPI::Muls(xFp32Reg, xFp32Reg, hPreValue7, mask);
            MicroAPI::Add(accFp32Reg, accFp32Reg, xFp32Reg, mask);

            MicroAPI::Cast<T, P, ctFp32To16>(outB16Reg, accFp32Reg, mask);
            MicroAPI::StoreAlign<T, MicroAPI::StoreDist::DIST_PACK_B32>(hinOutAddr + dIdx * eleNumPerVf, outB16Reg, mask);
        }
    }
}

} // namespace MhcPre

#endif // __mhc_pre_KERNEL_H_
