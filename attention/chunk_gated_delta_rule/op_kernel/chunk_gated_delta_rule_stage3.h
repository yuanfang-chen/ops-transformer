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
 * \file chunk_gated_delta_rule_stage3.h
 * \brief
 */
#ifndef __CHUNK_GATED_DELTA_RULE_STAGE3_H_
#define __CHUNK_GATED_DELTA_RULE_STAGE3_H_

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "chunk_gated_delta_rule_tiling_data.h"

namespace ChunkGatedDeltaRule {
using namespace AscendC;
using namespace matmul;

using aT3 = MatmulType<TPosition::GM, CubeFormat::ND, float, true>;
using bT3 = MatmulType<TPosition::GM, CubeFormat::ND, float, true>;
using cT3 = MatmulType<TPosition::GM, CubeFormat::ND, float>;
using MT3 = matmul::MatmulImpl<aT3, bT3, cT3>;

template <typename inType, typename outType>
struct mm3Params {
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

struct StageThreeParams {
    // in
    GlobalTensor<float> qkt_;       // (Nv, Sp, Dk)
    GlobalTensor<float> gCumExp_;           // (Nv, Sp)
    GlobalTensor<float> attnInter_;
    GlobalTensor<float> vInner_;
    GlobalTensor<float> maskTensor_;

    // tmp
    GM_ADDR ws;

    // out
    GlobalTensor<bfloat16_t> attnOut_;

    // matmul
    MT3 *mm3_;

    // Pipe
    TPipe *pipe_;

    // attr
    ChunkGroup *cg;
    float scale_;
    int64_t maxGroupLength_;
    int64_t Nv_;
    int64_t Nk_;
    int64_t Dv_;
    int64_t Dk_;
};

class Stage3 {
public:
    __aicore__ inline void Init(StageThreeParams *initParams, int32_t coreNum)
    {
        sTP_ = initParams;
        pipe_ = sTP_->pipe_;
        chunkSize_ = sTP_->cg->chunkSize;
        Sp_ = (sTP_->cg->length + chunkSize_ - 1) / chunkSize_  * chunkSize_;
        chunkNum_ = Sp_ / chunkSize_;
        coreNum_ = coreNum;

        uint64_t workSpaceOffset = 0;
        cCFloatGM_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(initParams->ws + workSpaceOffset + chunkSize_ * chunkSize_ * sizeof(float)));

        if ASCEND_IS_AIC {
            return;
        }

        pipe_->InitBuffer(inQueue_, BUFFER_NUM, chunkSize_ > sTP_->Dv_ ? chunkSize_ * sTP_->Dk_ * sizeof(float) : sTP_->Dv_ * sTP_->Dk_ * sizeof(float));
        pipe_->InitBuffer(outQueue_, BUFFER_NUM, chunkSize_ * sTP_->Dv_ * sizeof(float));
        pipe_->InitBuffer(tmpBuff_, (4 * chunkSize_ * chunkSize_ * sizeof(float)));
        uint32_t buffOffset = 0;
        cCFloat_ = tmpBuff_.GetWithOffset<float>(static_cast<uint32_t>(chunkSize_ * chunkSize_), buffOffset);
        buffOffset += chunkSize_ * chunkSize_ * sizeof(float);
        cCFloat2_ = tmpBuff_.GetWithOffset<float>(static_cast<uint32_t>(chunkSize_ * chunkSize_), buffOffset);
    }

    __aicore__ inline void Process()
    {
        for (int nv_id = 0; nv_id < sTP_->Nv_; nv_id++) {
            int core_id = GetBlockIdx();
            if ASCEND_IS_AIV {
                core_id /= 2;
            }
            for (; core_id < chunkNum_; core_id += coreNum_) {
                int idx = core_id * chunkSize_;
                curChunkSize_ = idx + chunkSize_ > sTP_->cg->length ? sTP_->cg->length - idx : chunkSize_;  // 实际chunk
                if ASCEND_IS_AIV {
                    if (GetSubBlockIdx() == 1) {    // 单AIV计算
                        CrossCoreSetFlag<0x2, PIPE_MTE3>(0x4);
                        CrossCoreWaitFlag(0x3);
                        continue;
                    }
                    CalMaskedQKT(cCFloatGM_, nv_id, idx);
                    CrossCoreSetFlag<0x2, PIPE_MTE3>(0x4);
                    // DumpTensor(cCFloatGM_, 0, chunkSize_ * curChunkSize_);
                    CrossCoreWaitFlag(0x3);
                    CalAttnOut(sTP_->attnInter_[nv_id * Sp_ * sTP_->Dv_ + idx * sTP_->Dv_],
                               sTP_->attnOut_[idx * sTP_->Nv_ * sTP_->Dv_ + nv_id * sTP_->Dv_]);
                }
                if ASCEND_IS_AIC {
                    CrossCoreWaitFlag(0x4);
                    mm3Params<float, float> params{
                        cCFloatGM_,
                        sTP_->vInner_[nv_id * Sp_ * sTP_->Dv_ + idx * sTP_->Dv_],
                        sTP_->attnInter_[nv_id * Sp_ * sTP_->Dv_ + idx * sTP_->Dv_],
                        chunkSize_, sTP_->Dv_, chunkSize_, chunkSize_, sTP_->Dv_, chunkSize_};
                    AICProcess<float, float>(params, 1, false, false);
                    CrossCoreSetFlag<0x2, PIPE_FIX>(0x3);
                }
            }
        }
    }
    
    __aicore__ inline void CalMaskedQKT(GlobalTensor<float> outGM, int nv_id, int idx)
    {
        int offset0 = nv_id * Sp_ * chunkSize_;
        int offset1 = idx * chunkSize_;
        // g_cum_exp
        CopyIn<float>(sTP_->gCumExp_[nv_id * Sp_ + idx], 1, chunkSize_);
        auto g_cum_exp = inQueue_.DeQue<float>();
        // broadcast
        const uint32_t srcShape1[] = {static_cast<uint32_t>(chunkSize_), static_cast<uint32_t>(1)};
        const uint32_t srcShape2[] = {static_cast<uint32_t>(1), static_cast<uint32_t>(chunkSize_)};
        const uint32_t dstShape[] = {static_cast<uint32_t>(chunkSize_), static_cast<uint32_t>(chunkSize_)};
        Broadcast<float, 2, 1>(cCFloat_, g_cum_exp, dstShape, srcShape1);
        Broadcast<float, 2, 0>(cCFloat2_, g_cum_exp, dstShape, srcShape2);
        PipeBarrier<PIPE_V>();

        Div(cCFloat_, cCFloat_, cCFloat2_, chunkSize_ * chunkSize_);
        inQueue_.FreeTensor(g_cum_exp);
        
        // qkt
        CopyIn<float>(sTP_->qkt_[offset0 + offset1], chunkSize_, chunkSize_);
        auto qkt = inQueue_.DeQue<float>();
        auto scale_qkt = outQueue_.AllocTensor<float>();
        Muls(scale_qkt, qkt, sTP_->scale_, chunkSize_ * chunkSize_);
        Mul(scale_qkt, scale_qkt, cCFloat_, chunkSize_ * chunkSize_);
        inQueue_.FreeTensor(qkt);
        
        // mask
        CopyIn<float>(sTP_->maskTensor_, chunkSize_, chunkSize_);
        auto lower = inQueue_.DeQue<float>();
        Mul(scale_qkt, scale_qkt, lower, chunkSize_ * chunkSize_);

        outQueue_.EnQue(scale_qkt);
        CopyOut<float>(outGM, chunkSize_, chunkSize_);   // 处理非对齐内容
        inQueue_.FreeTensor(lower);
    }

    __aicore__ inline void CalAttnOut(GlobalTensor<float> inTensor, GlobalTensor<bfloat16_t> outTensor)
    {
        AttnCopyIn(inTensor, chunkSize_, sTP_->Dv_);
        curDv_ = Ceil(sTP_->Dv_, 32 / sizeof(bfloat16_t)) * (32 / sizeof(bfloat16_t));
        auto out = inQueue_.DeQue<float>();
        auto attn_out = outQueue_.AllocTensor<bfloat16_t>();
        Cast(attn_out, out, RoundMode::CAST_RINT, chunkSize_ * curDv_);
        outQueue_.EnQue(attn_out);
        AttnCopyOut(outTensor);
        inQueue_.FreeTensor(out);
    }

    template <typename inType, typename outType>
    __aicore__ inline void AICProcess(mm3Params<inType, outType>& params,
        bool mmType=0, bool isTransposeA=false, bool isTransposeB=false)
    {
        if constexpr (std::is_same_v<inType, float>) {
            sTP_->mm3_->SetOrgShape(params.m, params.n, params.k);    // MNK
            sTP_->mm3_->SetSingleShape(params.singleM, params.singleN, params.singleK); // SingleCoreMNK
            sTP_->mm3_->SetTensorA(params.x, isTransposeA);
            sTP_->mm3_->SetTensorB(params.y, isTransposeB);
            sTP_->mm3_->IterateAll(params.z, mmType);
            sTP_->mm3_->End();
        }
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
            SetAtomicAdd<bfloat16_t>();
        }
        DataCopyPad(tmpGM, outLocal, copyParams);
        if (setAtomic) {
            SetAtomicNone();
        }
        outQueue_.FreeTensor(outLocal);
    }

    __aicore__ inline void AttnCopyIn(GlobalTensor<float> tmpGM, int32_t row, int32_t col)
    {
        LocalTensor<float> inLocal = inQueue_.AllocTensor<float>();
        DataCopyPadExtParams<float> padParams;
        DataCopyExtParams inParams{static_cast<uint16_t>(row),
                                    static_cast<uint32_t>(col * sizeof(float)),                // 非对齐情况需要补0
                                    static_cast<uint32_t>(0), 
                                    0, 0};
        int padding = Ceil(col, 32 / sizeof(bfloat16_t)) * (32 / sizeof(bfloat16_t)) - col;
        DataCopyPadExtParams<float> copyPadParams{true, 0, static_cast<uint8_t>(padding), 0};
        DataCopyPad(inLocal, tmpGM, inParams, padParams);
        inQueue_.EnQue(inLocal);
    }

    __aicore__ inline void AttnCopyOut(GlobalTensor<bfloat16_t> tmpGM)
    {
        auto outLocal = outQueue_.DeQue<bfloat16_t>();

        DataCopyExtParams copyParams;
        copyParams.blockCount = static_cast<uint16_t>(curChunkSize_);
        copyParams.blockLen = static_cast<uint32_t>(sTP_->Dv_ * sizeof(bfloat16_t));
        copyParams.srcStride = static_cast<uint32_t>((0) * sizeof(bfloat16_t));
        copyParams.dstStride = static_cast<uint32_t>((sTP_->Nv_ * sTP_->Dv_ - sTP_->Dv_) * sizeof(bfloat16_t));

        DataCopyPad(tmpGM, outLocal, copyParams);
        outQueue_.FreeTensor(outLocal);
    }

private:
    StageThreeParams *sTP_;
    TPipe *pipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue_;
    TBuf<TPosition::VECCALC> tmpBuff_;
    GlobalTensor<float> cCFloatGM_;
    GlobalTensor<float> cDvFloatGM_;
    LocalTensor<float> cCFloat_;
    LocalTensor<float> cCFloat2_;
    int32_t curDk_; // Dk非对齐时补齐后长度
    int32_t curDv_; // Dv非对齐时补齐后长度
    int32_t curChunkSize_; // Dv非对齐时补齐后长度
    int32_t chunkSize_; // Dk非对齐时补齐后长度
    int64_t Sp_;    // S非对齐时补齐后长度
    int32_t chunkNum_;    // S非对齐时补齐后Chunk个数
    int32_t coreNum_;    // S非对齐时补齐后Chunk个数
};

} // namespace ChunkGatedDeltaRule
#endif