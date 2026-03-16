/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file chunk_gated_delta_rule_stage2.h
 * \brief
 */
#ifndef __CHUNK_GATED_DELTA_RULE_STAGE2_H_
#define __CHUNK_GATED_DELTA_RULE_STAGE2_H_

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "chunk_gated_delta_rule_tiling_data.h"

namespace ChunkGatedDeltaRule {
using namespace AscendC;
using namespace matmul;

using aT2 = MatmulType<TPosition::GM, CubeFormat::ND, float, true>;
using bT2 = MatmulType<TPosition::GM, CubeFormat::ND, float, true>;
using cT2 = MatmulType<TPosition::GM, CubeFormat::ND, float>;
using StageTwoMT = matmul::MatmulImpl<aT2, bT2, cT2>;

constexpr uint64_t BUFFER_NUM = 1;
constexpr uint64_t BLOCK_SIZE = 32;
static constexpr uint64_t V_MTE3_EVENT = 0;
static constexpr uint64_t MTE2_V_EVENT = 2;
static constexpr uint64_t MTE3_MTE2_EVENT = 4;

struct StageTwoParams {
    GlobalTensor<float> qPrime_;    // (Nv, Sp, Dk)
    GlobalTensor<float> vInner_;    // (Nv, Sp, Dv)
    GlobalTensor<float> gCumExp_;   // (Nv, Sp)
    GlobalTensor<float> kCumdecay_; // (Nv, Sp, Dk)
    GlobalTensor<float> curState_;  // (Nv, Dv, Dk)
    GlobalTensor<float> kg_;
    GlobalTensor<float> attnInter_;

    GM_ADDR ws;

    StageTwoMT *mm1_;

    TPipe *pipe_;

    ChunkGroup *cg;
    int64_t Nv_;
    int64_t Nk_;
    int64_t Dv_;
    int64_t Dk_;
};

template <typename inType, typename outType>
struct mm2Params {
    GlobalTensor<inType> x;
    GlobalTensor<inType> y;
    GlobalTensor<outType> z;
    int64_t m;
    int64_t n;
    int64_t k;
    int64_t singleM;
    int64_t singleN;
    int64_t singleK;
};

class Stage2 {
public:
    __aicore__ inline void Init(StageTwoParams *initParams, int32_t coreNum)
    {
        sTP_ = initParams;
        pipe_ = sTP_->pipe_;
        chunkSize_ = sTP_->cg->chunkSize;
        seqLength_ = sTP_->cg->length;
        Sp_ = (seqLength_ + chunkSize_ - 1) / chunkSize_  * chunkSize_;
        chunkNum_ = Sp_ / chunkSize_;
        coreNum_ = coreNum;
        Nv_ = sTP_->Nv_;
        Nk_ = sTP_->Nk_;
        Dv_ = sTP_->Dv_;
        Dk_ = sTP_->Dk_;
        curDk_ = Ceil(sTP_->Dk_, BLOCK_SIZE / sizeof(float)) * (BLOCK_SIZE / sizeof(float));
        curChunkSize_ = chunkSize_;
        InitLocalBuffers();
    }

    __aicore__ inline void InitLocalBuffers()
    {
        if ASCEND_IS_AIC {
            return;
        }
        pipe_->InitBuffer(inQueue_, BUFFER_NUM, chunkSize_ > Dv_ ? chunkSize_ * curDk_ * sizeof(float) : Dv_ * curDk_ * sizeof(float));
        uint64_t outQueueSize = AscendC::Std::max((uint64_t)chunkSize_ * chunkSize_ * sizeof(float), (uint64_t)Dv_ * curDk_ * sizeof(bfloat16_t));
        pipe_->InitBuffer(outQueue_, BUFFER_NUM, outQueueSize);
        pipe_->InitBuffer(tmpBuff_, (chunkSize_ > Dv_ ? chunkSize_ * curDk_ * sizeof(float) : Dv_ * curDk_ * sizeof(float)));
    }

    __aicore__ inline void Process()
    {
        int64_t coreId = GetBlockIdx();
        if ASCEND_IS_AIV {
            coreId /= 2;
        }
        int64_t nvPerCore = (Nv_ + coreNum_ - 1) / coreNum_;
        int64_t nvStart = coreId * nvPerCore;
        int64_t nvEnd = nvStart + nvPerCore;
        nvEnd = nvEnd > Nv_ ? Nv_ : nvEnd;
        int64_t lastChunkSize = seqLength_ % chunkSize_ == 0 ? chunkSize_ : seqLength_ % chunkSize_;
        for (int64_t nvId = nvStart; nvId < nvEnd; nvId++) {
            auto curState = sTP_->curState_[nvId * Dv_ * Dk_];
            for (int64_t cId = 0; cId < chunkNum_; cId++) {
                int64_t length = cId * chunkSize_;
                if (cId == chunkNum_ - 1) {
                    curChunkSize_ = lastChunkSize;
                }
                if ASCEND_IS_AIV {
                    if (GetSubBlockIdx() == 0) {
                        CopyIn<float>(curState, sTP_->Dv_, sTP_->Dk_);
                    }
                    CrossCoreWaitFlag(0x2);
                    if (GetSubBlockIdx() == 0) {
                        CalGCumExp(curState, sTP_->gCumExp_[nvId * seqLength_ + length]);
                    }
                    CrossCoreSetFlag<0x2, PIPE_MTE3>(0x3);  // 当前state非空，无法直接原子累加，需要覆盖写完通知AIC
                    CrossCoreWaitFlag(0x4);
                }
                if ASCEND_IS_AIC {
                    int mm_offset0 = nvId * seqLength_ * Dk_ + length * Dk_;
                    int mm_offset1 = nvId * seqLength_ * Dv_ + length * Dv_;
                    CalVPrime(sTP_->kCumdecay_[mm_offset0], curState, sTP_->vInner_[mm_offset1]);
                    CalAttnInter(sTP_->qPrime_[mm_offset0], curState, sTP_->attnInter_[mm_offset1]);
                    CrossCoreSetFlag<0x2, PIPE_FIX>(0x2);   // 读完之前AIV不能写
                    CrossCoreWaitFlag(0x3);
                    CalStateNew(sTP_->vInner_[mm_offset1], sTP_->kg_[mm_offset0], curState);
                    CrossCoreSetFlag<0x2, PIPE_FIX>(0x4);
                }
            }
        }
    }

    __aicore__ inline void CalGCumExp(GlobalTensor<float> stateNew, GlobalTensor<float> gCumExp)
    {
        float last_g_cum_exp = gCumExp.GetValue(curChunkSize_ - 1);
        auto state_in = inQueue_.DeQue<float>();
        auto state_out = outQueue_.AllocTensor<float>();
        SetFlag<HardEvent::MTE2_V>(MTE2_V_EVENT);
        WaitFlag<HardEvent::MTE2_V>(MTE2_V_EVENT);
        Muls(state_out, state_in, last_g_cum_exp, Dv_ * curDk_);
        SetFlag<HardEvent::V_MTE3>(V_MTE3_EVENT);
        WaitFlag<HardEvent::V_MTE3>(V_MTE3_EVENT);
        outQueue_.EnQue(state_out);
        CopyOut<float>(stateNew, Dv_, Dk_);
        inQueue_.FreeTensor(state_in);
    }

    __aicore__ inline void CalAttnInter(GlobalTensor<float> qPrime,
                                        GlobalTensor<float> state,
                                        GlobalTensor<float> attnInter)
    {
        // q_prime @ state.transpose(0, 1)
        sTP_->mm1_->SetOrgShape(curChunkSize_, Dv_, Dk_);    // MNK
        sTP_->mm1_->SetSingleShape(curChunkSize_, Dv_, Dk_); // SingleCoreMNK
        sTP_->mm1_->SetTensorA(qPrime, false);
        sTP_->mm1_->SetTensorB(state, true);
        sTP_->mm1_->IterateAll(attnInter);
        sTP_->mm1_->End();
    }

    __aicore__ inline void CalVPrime(GlobalTensor<float> kCumdecay,
                                     GlobalTensor<float> state,
                                     GlobalTensor<float> vPrime)
    {
        // v_inner += k_cumdecay @ state.transpose(0, 1)
        sTP_->mm1_->SetOrgShape(curChunkSize_, Dv_, Dk_);    // MNK
        sTP_->mm1_->SetSingleShape(curChunkSize_, Dv_, Dk_); // SingleCoreMNK
        sTP_->mm1_->SetTensorA(kCumdecay, false);
        sTP_->mm1_->SetTensorB(state, true);
        sTP_->mm1_->IterateAll(vPrime, 1);
        sTP_->mm1_->End();
    }

    __aicore__ inline void CalStateNew(GlobalTensor<float> vInner,
                                       GlobalTensor<float> kg,
                                       GlobalTensor<float> state)
    {
        // state_out = v_new.transpose(0, 1) @ kg 
        sTP_->mm1_->SetOrgShape(Dv_, Dk_, curChunkSize_);    // MNK
        sTP_->mm1_->SetSingleShape(Dv_, Dk_, curChunkSize_); // SingleCoreMNK
        sTP_->mm1_->SetTensorA(vInner, true);
        sTP_->mm1_->SetTensorB(kg, false);
        sTP_->mm1_->IterateAll(state, 1);
        sTP_->mm1_->End();
    }

    template <typename inType>
    __aicore__ inline void CopyIn(GlobalTensor<inType> tmpGM, int32_t row, int32_t col)
    {
        LocalTensor<inType> inLocal = inQueue_.AllocTensor<inType>();
        DataCopyPadExtParams<inType> padParams;
        DataCopyExtParams inParams{static_cast<uint16_t>(row),
                                   static_cast<uint32_t>(col * sizeof(inType)),                // 非对齐情况需要补0
                                   static_cast<uint32_t>(0), 
                                   0, 0};
        int padding = Ceil(col, 32 / sizeof(inType)) * (32 / sizeof(inType)) - col;
        DataCopyPadExtParams<inType> copyPadParams{true, 0, static_cast<uint8_t>(padding), 0};
        DataCopyPad(inLocal, tmpGM, inParams, padParams);
        inQueue_.EnQue(inLocal);
    }

    template <typename outType>
    __aicore__ inline void CopyOut(GlobalTensor<outType> tmpGM, int32_t row, int32_t col, bool setAtomic = false)
    {
        auto outLocal = outQueue_.DeQue<outType>();
        DataCopyExtParams copyParams;
        copyParams.blockCount = static_cast<uint16_t>(row);
        copyParams.blockLen = static_cast<uint32_t>(col * sizeof(outType));
        copyParams.srcStride = static_cast<uint32_t>(0);
        copyParams.dstStride = static_cast<uint32_t>((0) * sizeof(outType));
        if (setAtomic) {
            SetAtomicAdd<float>();
        }
        DataCopyPad(tmpGM, outLocal, copyParams);
        if (setAtomic) {
            SetAtomicNone();
        }
        outQueue_.FreeTensor(outLocal);
    }

private:
    StageTwoParams *sTP_;
    TPipe *pipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue_;
    TBuf<TPosition::VECCALC> tmpBuff_;
    GlobalTensor<float> curStateGM_;
    LocalTensor<float> DvDkFloat_;
    int64_t Nk_;
    int64_t Nv_;
    int64_t Dk_;
    int64_t Dv_;
    int64_t seqLength_;
    int32_t chunkSize_;
    int32_t curChunkSize_;
    int32_t curDk_;
    int64_t Sp_;
    int32_t chunkNum_;
    int32_t coreNum_;
};
} // namespace ChunkGatedDeltaRule
#endif