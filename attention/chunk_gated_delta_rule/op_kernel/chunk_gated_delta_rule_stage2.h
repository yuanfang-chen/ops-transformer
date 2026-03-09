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

struct StageTwoParams {
    // in
    GlobalTensor<bfloat16_t> qPrime_;       // (Nv, Sp, Dk)
    GlobalTensor<float> vInner_;            // (Nv, Sp, Dv)
    GlobalTensor<float> gCumExp_;           // (Nv, Sp)
    GlobalTensor<bfloat16_t> kCumdecay_;    // (Nv, Sp, Dk)
    GlobalTensor<bfloat16_t> initState_;        // (Nv, Dv, Dk)
    GlobalTensor<float> kg_;
    // out
    GlobalTensor<bfloat16_t> finalState_;
    GlobalTensor<float> attnInter_;
    GlobalTensor<float> vNew_;

    // Matmul
    MT0 *mm1_;
    MT1 *mm2_;

    // Pipe
    TPipe *pipe_;

    // attr
    ChunkGroup *cg;
    int32_t maxGroupLength_;
    int32_t Nv_;
    int32_t Nk_;
    int32_t Dv_;
    int32_t Dk_;
};

class Stage2 {
public:
    __aicore__ inline void Init(StageTwoParams *initParams)
    {
        sTP_ = initParams;
        pipe_ = sTP_->pipe_;
        Sp_ = cg->maxGroupLength;
        chunkSize_ = initParams->cg->chunkSize;
        chunkNum_ = Sp_ / chunkSize_;
        InitLocalBuffers();
    }

    __aicore__ inline void InitLocalBuffers()
    {
        if ASCEND_IS_AIC {
            return;
        }
        pipe_->InitBuffer(inQueue_, BUFFER_NUM, chunkSize_ > sTP_->Dv_ ? chunkSize_ * sTP_->Dk_ * sizeof(float) : sTP_->Dv_ * sTP_->Dk_ * sizeof(float));
        pipe_->InitBuffer(outQueue_, BUFFER_NUM, chunkSize_ * chunkSize_ * sizeof(float));
        pipe_->InitBuffer(tmpBuff_, (chunkSize_ > sTP_->Dv_ ? chunkSize_ * sTP_->Dk_ * sizeof(float) : sTP_->Dv_ * sTP_->Dk_ * sizeof(float)));
        uint32_t buffOffset = 0;
        DvDkFloat_ = sTP_->tmpBuff_->GetWithOffset<float>(static_cast<uint32_t>(sTP_->Dv_ * sTP_->Dk_), buffOffset);
    }

    __aicore__ inline void Process()
    {
        for (int nv_id = 0; nv_id < sTP_->Nv_; nv_id++) {
            int state_offset = nv_id * sTP_->Dv_ * sTP_->Dk_;
            for (int core_id = GetBlockIdx(); core_id < Cn_; core_id += 1) {   // 当前为1核
                auto cur_state = core_id == GetBlockIdx() ?
                    sTP_->state_[state_offset] : sTP_->finalState_[state_offset];
                int idx = core_id * sTP_->chunkSize_;
                if ASCEND_IS_AIV {
                    CrossCoreWaitFlag(0x4);
                    CalGCumExp(cur_state, sTP_->finalState_[state_offset]);
                    if (core_id + 1 < Cn_) {
                        CrossCoreSetFlag<0x2, PIPE_MTE3>(0x3);
                    }
                }
                if ASCEND_IS_AIC {
                    int mm_offset0 = nv_id * Sp_ * sTP_->Dk_ + idx * sTP_->Dk_;
                    int mm_offset1 = nv_id * Sp_ * sTP_->Dv_ + idx * sTP_->Dv_;
                    if (core_id != GetBlockIdx()) {
                        CrossCoreWaitFlag(0x3);
                    }
                    CalVPrime(sTP_->kCumdecay_[mm_offset0], cur_state, sTP_->vNew_[mm_offset1]);
                    CalAttnInter(sTP_->qPrime_[mm_offset0], cur_state, sTP_->attnInter_[mm_offset1]);
                    // 上面读完AIV才可以开始写
                    CrossCoreSetFlag<0x2, PIPE_FIX>(0x4);
                    CalStateNew(sTP_->vNew_[mm_offset1], sTP_->kg_[mm_offset0], sTP_->finalState_[state_offset]);
                }
            }
        }
    }

    __aicore__ inline void CalGCumExp(GlobalTensor<bfloat16_t> state_old, GlobalTensor<bfloat16_t> state_new)
    {
        float last_g_cum_exp = sTP_->gCumExp_.GetValue(sTP_->chunkSize_ - 1);
        CopyIn<bfloat16_t>(state_old, sTP_->Dv_, sTP_->Dk_);
        curDk_ = Ceil(sTP_->Dk_, 32 / sizeof(bfloat16_t)) * (32 / sizeof(bfloat16_t));
        auto state_in = sTP_->inQueue_->DeQue<bfloat16_t>();
        Cast(DvDkFloat_, state_in, RoundMode::CAST_NONE, sTP_->Dv_ * curDk_);
        PipeBarrier<PIPE_V>();
        Muls(DvDkFloat_, DvDkFloat_, last_g_cum_exp, sTP_->Dv_ * curDk_);
        auto state_out = sTP_->outQueue_->AllocTensor<bfloat16_t>();
        Cast(state_out, DvDkFloat_, RoundMode::CAST_RINT, sTP_->Dv_ * curDk_);
        sTP_->inQueue_->FreeTensor(state_in);
        sTP_->outQueue_->EnQue(state_out);
        CopyOut<bfloat16_t>(state_new, sTP_->Dv_, sTP_->Dk_, true);
    }

    __aicore__ inline void CalAttnInter(GlobalTensor<bfloat16_t> a, GlobalTensor<bfloat16_t> b, GlobalTensor<float> c)
    {
        mmParams<bfloat16_t, float> params{a, b, c,
            Sp_, sTP_->Dv_, sTP_->Dk_,
            sTP_->chunkSize_, sTP_->Dv_, sTP_->Dk_};
        AICProcess<bfloat16_t>(params, 0, false, true);
    }

    __aicore__ inline void CalVPrime(GlobalTensor<bfloat16_t> a, GlobalTensor<bfloat16_t> b, GlobalTensor<float> c)
    {
        mmParams<bfloat16_t, float> params{a, b, c,
            Sp_, sTP_->Dv_, sTP_->Dk_,
            sTP_->chunkSize_, sTP_->Dv_, sTP_->Dk_};
        AICProcess<bfloat16_t>(params, 0, false, true);
    }

    __aicore__ inline void CalStateNew(GlobalTensor<float> a, GlobalTensor<float> b, GlobalTensor<bfloat16_t> c)
    {
        mmParams<float, bfloat16_t> params{a, b, c,
            sTP_->Dv_, sTP_->Dk_, sTP_->chunkSize_,
            sTP_->Dv_, sTP_->Dk_, sTP_->chunkSize_};
        // state_new = (key * (g_cum_exp[-1, None] / g_cum_exp)[..., None]).transpose(-1, -2) @ v_new
        // CrossCoreWaitFlag(0x3);
        AICProcess<float>(params, 1, true, false);
    }

    template <typename inType, typename outType>
    __aicore__ inline void AICProcess(mmParams<inType, outType>& params,
        bool mmType=0, bool isTransposeA=false, bool isTransposeB=false)
    {
        if constexpr (std::is_same_v<inType, bfloat16_t>) {
            sTP_->mm1_->SetOrgShape(params.m, params.n, params.k);    // MNK
            sTP_->mm1_->SetSingleShape(params.singleM, params.singleN, params.singleK); // SingleCoreMNK
            sTP_->mm1_->SetTensorA(params.x, isTransposeA);
            sTP_->mm1_->SetTensorB(params.y, isTransposeB);
            sTP_->mm1_->IterateAll(params.z);
            sTP_->mm1_->End();
        }
        if constexpr (std::is_same_v<inType, float>) {
            sTP_->mm2_->SetOrgShape(params.m, params.n, params.k);    // MNK
            sTP_->mm2_->SetSingleShape(params.singleM, params.singleN, params.singleK); // SingleCoreMNK
            sTP_->mm2_->SetTensorA(params.x, isTransposeA);
            sTP_->mm2_->SetTensorB(params.y, isTransposeB);
            sTP_->mm2_->IterateAll(params.z, mmType);
            sTP_->mm2_->End();
        }
    }

    template <typename inType>
    __aicore__ inline void CopyIn(GlobalTensor<inType> tmpGM, int32_t row, int32_t col)
    {
        LocalTensor<inType> inLocal = sTP_->inQueue_->AllocTensor<inType>();
        DataCopyPadExtParams<inType> padParams;
        DataCopyExtParams inParams{static_cast<uint16_t>(row),
                                    static_cast<uint32_t>(col * sizeof(inType)),                // 非对齐情况需要补0
                                    static_cast<uint32_t>(0), 
                                    0, 0};
        int padding = Ceil(col, 32 / sizeof(inType)) * (32 / sizeof(inType)) - col;
        DataCopyPadExtParams<inType> copyPadParams{true, 0, static_cast<uint8_t>(padding), 0};
        DataCopyPad(inLocal, tmpGM, inParams, padParams);
        sTP_->inQueue_->EnQue(inLocal);
    }

    template <typename outType>
    __aicore__ inline void CopyOut(GlobalTensor<outType> tmpGM, int32_t row, int32_t col, bool setAtomic = false)
    {
        auto outLocal = sTP_->outQueue_->DeQue<outType>();
        DataCopyExtParams copyParams;
        copyParams.blockCount = static_cast<uint16_t>(row);
        copyParams.blockLen = static_cast<uint32_t>(col * sizeof(outType));
        copyParams.srcStride = static_cast<uint32_t>(0);
        copyParams.dstStride = static_cast<uint32_t>((0) * sizeof(outType));
        if (setAtomic) {
            SetAtomicAdd<bfloat16_t>();
        }
        DataCopyPad(tmpGM, outLocal, copyParams);
        if (setAtomic) {
            SetAtomicNone();
        }
        sTP_->outQueue_->FreeTensor(outLocal);
    }

private:
    StageTwoParams *sTP_;
    TPipe *pipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue_;
    TBuf<TPosition::VECCALC> tmpBuff_;
    LocalTensor<float> DvDkFloat_;
    int32_t chunkSize_; // Dk非对齐时补齐后长度
    int32_t curDk_; // Dk非对齐时补齐后长度
    int32_t Sp_;    // S非对齐时补齐后长度
    int32_t chunkNum_;    // S非对齐时补齐后Chunk个数
};
} // namespace ChunkGatedDeltaRule
#endif