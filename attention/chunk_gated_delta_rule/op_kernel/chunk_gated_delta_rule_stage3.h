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
using StageThreeMT = matmul::MatmulImpl<aT3, bT3, cT3>;

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
    GlobalTensor<float> qkt;       // (Nv, Sp, Dk)
    GlobalTensor<float> gCumExp;           // (Nv, Sp)
    GlobalTensor<float> attnInter;
    GlobalTensor<float> vInner;
    GlobalTensor<float> maskTensor;
    GM_ADDR ws;
    GlobalTensor<bfloat16_t> attnOut;
    StageThreeMT *mm3;
    TPipe *pipe;
    ChunkGroup *cg;
    float scale_;
    int64_t Nv;
    int64_t Nk;
    int64_t Dv;
    int64_t Dk;
    bool gOptional;
};

class Stage3 {
public:
    __aicore__ inline void Init(StageThreeParams *initParams, int32_t coreNum)
    {
        sTP_ = initParams;
        pipe_ = sTP_->pipe;
        chunkSize_ = sTP_->cg->chunkSize;
        seqLength_ = sTP_->cg->length;
        Sp_ = (seqLength_ + chunkSize_ - 1) / chunkSize_  * chunkSize_;
        chunkNum_ = (seqLength_ + chunkSize_ - 1) / chunkSize_ ;
        coreNum_ = coreNum;
        Nv_ = sTP_->Nv;
        Nk_ = sTP_->Nk;
        Dv_ = sTP_->Dv;
        Dk_ = sTP_->Dk;
        gOptional_ = sTP_->gOptional;

        uint64_t workSpaceOffset = 0;
        tmpGM_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(initParams->ws + workSpaceOffset +
                                                                   coreNum_ * chunkSize_ * chunkSize_ * sizeof(float)));

        if ASCEND_IS_AIC {
            return;
        }

        pipe_->InitBuffer(inQueue_, BUFFER_NUM_ONE, 
               chunkSize_ > Dv_ ? chunkSize_ * Dk_ * sizeof(float) : Dv_ * Dk_ * sizeof(float));
        pipe_->InitBuffer(outQueue_, BUFFER_NUM_ONE, chunkSize_ * Dv_ * sizeof(float));
        pipe_->InitBuffer(tmpBuff_, (STAGE3_BUFFER_COUNT * chunkSize_ * chunkSize_ * sizeof(float)));
        uint32_t buffOffset = 0;
        tmpBuffer1_ = tmpBuff_.GetWithOffset<float>(static_cast<uint32_t>(chunkSize_ * chunkSize_), buffOffset);
        buffOffset += chunkSize_ * chunkSize_ * sizeof(float);
        tmpBuffer2_ = tmpBuff_.GetWithOffset<float>(static_cast<uint32_t>(chunkSize_ * chunkSize_), buffOffset);
    }

    __aicore__ inline void Process()
    {
        // Nv Nc 融合
        int64_t totalChunks = Nv_ * chunkNum_;
        int64_t chunksPerCore = (totalChunks + coreNum_ -1) / coreNum_;
        int64_t lastChunkSize = seqLength_ % chunkSize_ == 0 ? chunkSize_ : seqLength_ % chunkSize_;
        int coreId = GetBlockIdx();
        if ASCEND_IS_AIV {
            coreId /= 2;
        }
        int64_t startChunk = coreId * chunksPerCore;
        int64_t endChunk = startChunk + chunksPerCore > totalChunks ? totalChunks : startChunk + chunksPerCore;
        for (int64_t idx = startChunk; idx < endChunk; idx++) {
            int64_t nvId = idx / chunkNum_;
            int64_t chunkId = idx % chunkNum_;
            int64_t chunkPos = chunkId * chunkSize_;    // 当前chunk起始位置
            curChunkSize_ = (chunkId == chunkNum_ - 1) ? lastChunkSize : chunkSize_; // 尾块
            if ASCEND_IS_AIV {
                if (GetSubBlockIdx() == 0) {
                    CalMaskedQKT(tmpGM_[coreId * chunkSize_ * chunkSize_], nvId, chunkPos);
                }
                CrossCoreSetFlag<0x2, PIPE_MTE3>(0x4);
                CrossCoreWaitFlag(0x3);
                if (GetSubBlockIdx() == 0) {
                    ReadAttnOut(sTP_->attnInter[nvId * Sp_ * Dv_ + chunkPos * Dv_]);
                    CalAttnOut(sTP_->attnOut[chunkPos * Nv_ * Dv_ + nvId * Dv_]);
                }
            }

            if ASCEND_IS_AIC {
                CrossCoreWaitFlag(0x4);
                AICProcess(tmpGM_[coreId * chunkSize_ * chunkSize_],
                           sTP_->vInner[nvId * Sp_ * Dv_ + chunkPos * Dv_],
                           sTP_->attnInter[nvId * Sp_ * Dv_ + chunkPos * Dv_]);
                CrossCoreSetFlag<0x2, PIPE_FIX>(0x3);
            }
        }
    }
    
    __aicore__ inline void CalMaskedQKT(GlobalTensor<float> outGM, int nvId, int chunkPos)
    {
        int64_t paddingChunkSize = Ceil(curChunkSize_, BLOCK_SIZE / sizeof(float)) * (BLOCK_SIZE / sizeof(float));
        if (gOptional_) {
            CopyIn<float>(sTP_->gCumExp[nvId * Sp_ + chunkPos], 1, curChunkSize_);

            auto g_cum_exp = inQueue_.DeQue<float>();
            const uint32_t srcShape1[] = {static_cast<uint32_t>(paddingChunkSize), static_cast<uint32_t>(1)};
            const uint32_t srcShape2[] = {static_cast<uint32_t>(1), static_cast<uint32_t>(paddingChunkSize)};
            const uint32_t dstShape[] = {static_cast<uint32_t>(paddingChunkSize), 
                                         static_cast<uint32_t>(paddingChunkSize)};
            Broadcast<float, BROADCAST_AXIS, 1>(tmpBuffer1_, g_cum_exp, dstShape, srcShape1);
            Broadcast<float, BROADCAST_AXIS, 0>(tmpBuffer2_, g_cum_exp, dstShape, srcShape2);
            PipeBarrier<PIPE_V>();
            Div(tmpBuffer1_, tmpBuffer1_, tmpBuffer2_, curChunkSize_ * paddingChunkSize);
            inQueue_.FreeTensor(g_cum_exp);
        } else {
            Duplicate(tmpBuffer1_, static_cast<float>(1.0f), curChunkSize_ * paddingChunkSize);
            PipeBarrier<PIPE_V>();
        }
 
        // qkt
        CopyIn<float>(sTP_->qkt[nvId * Sp_ * chunkSize_ + chunkPos * chunkSize_], curChunkSize_, curChunkSize_);
        auto qkt = inQueue_.DeQue<float>();
        auto scale_qkt = outQueue_.AllocTensor<float>();
        Muls(scale_qkt, qkt, sTP_->scale_, curChunkSize_ * paddingChunkSize);
        Mul(scale_qkt, scale_qkt, tmpBuffer1_, curChunkSize_ * paddingChunkSize);
        inQueue_.FreeTensor(qkt);
 
        // mask
        LocalTensor<float> inLocal = inQueue_.AllocTensor<float>();
        DataCopyExtParams inParams{static_cast<uint16_t>(curChunkSize_),
                                   static_cast<uint32_t>(curChunkSize_ * sizeof(float)),
                                   static_cast<uint32_t>((chunkSize_ - curChunkSize_) * sizeof(float)),
                                   0, 0};
        int padding = Ceil(curChunkSize_, BLOCK_SIZE / sizeof(float)) * (BLOCK_SIZE / sizeof(float)) - curChunkSize_;
        DataCopyPadExtParams<float> copyPadParams{true, 0, static_cast<uint8_t>(padding), 0};
        DataCopyPad(inLocal, sTP_->maskTensor, inParams, copyPadParams);
        inQueue_.EnQue(inLocal);
        auto lower = inQueue_.DeQue<float>();
        Mul(scale_qkt, scale_qkt, lower, curChunkSize_ * paddingChunkSize);
        outQueue_.EnQue(scale_qkt);
        CopyOut<float>(outGM, curChunkSize_, curChunkSize_);
        inQueue_.FreeTensor(lower);
    }

    __aicore__ inline void ReadAttnOut(GlobalTensor<float> inTensor)
    {
        AttnCopyIn(inTensor, curChunkSize_, Dv_);
    }

    __aicore__ inline void CalAttnOut(GlobalTensor<bfloat16_t> outTensor)
    {
        curDv_ = Ceil(Dv_, BLOCK_SIZE / sizeof(bfloat16_t)) * (BLOCK_SIZE / sizeof(bfloat16_t));
        auto out = inQueue_.DeQue<float>();
        auto attn_out = outQueue_.AllocTensor<bfloat16_t>();
        Cast(attn_out, out, RoundMode::CAST_RINT, curChunkSize_ * curDv_);
        outQueue_.EnQue(attn_out);
        AttnCopyOut(outTensor);
        inQueue_.FreeTensor(out);
    }

    __aicore__ inline void AICProcess(GlobalTensor<float>tmpGM,
                                      GlobalTensor<float>vInner,
                                      GlobalTensor<float>attnInter)
    {
        // masked_qkt @ v_inner
        sTP_->mm3->SetOrgShape(curChunkSize_, Dv_, curChunkSize_);    // MNK
        sTP_->mm3->SetSingleShape(curChunkSize_, Dv_, curChunkSize_); // SingleCoreMNK
        sTP_->mm3->SetTensorA(tmpGM);
        sTP_->mm3->SetTensorB(vInner);
        sTP_->mm3->IterateAll(attnInter, 1);
        sTP_->mm3->End();
    }

    template <typename inType>
    __aicore__ inline void CopyIn(GlobalTensor<inType> tmpGM, int32_t row, int32_t col)
    {
        LocalTensor<inType> inLocal = inQueue_.AllocTensor<inType>();
        DataCopyExtParams inParams{static_cast<uint16_t>(row),
                                    static_cast<uint32_t>(col * sizeof(inType)),
                                    static_cast<uint32_t>(0), 
                                    0, 0};
        int padding = Ceil(col, BLOCK_SIZE / sizeof(inType)) * (BLOCK_SIZE / sizeof(inType)) - col;
        DataCopyPadExtParams<inType> copyPadParams{true, 0, static_cast<uint8_t>(padding), 0};
        DataCopyPad(inLocal, tmpGM, inParams, copyPadParams);
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
        DataCopyExtParams inParams{static_cast<uint16_t>(row),
                                    static_cast<uint32_t>(col * sizeof(float)),
                                    static_cast<uint32_t>(0), 
                                    0, 0};
        int padding = Ceil(col, BLOCK_SIZE / sizeof(bfloat16_t)) * (BLOCK_SIZE / sizeof(bfloat16_t)) - col;
        DataCopyPadExtParams<float> copyPadParams{true, 0, static_cast<uint8_t>(padding), 0};
        DataCopyPad(inLocal, tmpGM, inParams, copyPadParams);
        inQueue_.EnQue(inLocal);
    }

    __aicore__ inline void AttnCopyOut(GlobalTensor<bfloat16_t> tmpGM)
    {
        auto outLocal = outQueue_.DeQue<bfloat16_t>();

        DataCopyExtParams copyParams;
        copyParams.blockCount = static_cast<uint16_t>(curChunkSize_);
        copyParams.blockLen = static_cast<uint32_t>(Dv_ * sizeof(bfloat16_t));
        copyParams.srcStride = static_cast<uint32_t>((0) * sizeof(bfloat16_t));
        copyParams.dstStride = static_cast<uint32_t>((Nv_ * Dv_ - Dv_) * sizeof(bfloat16_t));

        DataCopyPad(tmpGM, outLocal, copyParams);
        outQueue_.FreeTensor(outLocal);
        PipeBarrier<PIPE_MTE3>();
    }

private:
    StageThreeParams *sTP_;
    TPipe *pipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM_ONE> inQueue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM_ONE> outQueue_;
    TBuf<TPosition::VECCALC> tmpBuff_;
    GlobalTensor<float> tmpGM_;
    LocalTensor<float> tmpBuffer1_;
    LocalTensor<float> tmpBuffer2_;
    int32_t curDv_;
    int32_t curChunkSize_; 
    int32_t chunkSize_;
    int64_t seqLength_;
    int64_t Sp_;
    int32_t chunkNum_;
    int32_t coreNum_;
    int64_t Nv_;
    int64_t Nk_;
    int64_t Dv_;
    int64_t Dk_;
    bool gOptional_;
};

} // namespace ChunkGatedDeltaRule
#endif