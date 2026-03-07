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
 * \file grouped_matmul_finalize_routing.h
 * \brief
 */
#ifndef __CHUNK_GATED_DELTA_RULE_H_
#define __CHUNK_GATED_DELTA_RULE_H_

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "chunk_gated_delta_rule_tiling_data.h"
#include "chunk_gated_delta_rule_stage1.h"

namespace ChunkGatedDeltaRule {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

struct CGDRInitParams {
    GM_ADDR query;
    GM_ADDR key;
    GM_ADDR value;
    GM_ADDR beta;
    GM_ADDR initState;
    GM_ADDR seqlens;
    GM_ADDR gOptional;
    GM_ADDR attnOut;
    GM_ADDR finalState;
};


template <typename lowType, typename highType>
class CGDR {
public:
    __aicore__ inline CGDR(TPipe *pipe, const ChunkGatedDeltaRuleTilingData *tilingData)
        : stageOneOp_(mmFp32_, mmBf16_)
    {
        pipe_ = pipe;
        tiling_ = tilingData;
    };

    __aicore__ inline void InitMatmul()
    {
        if ASCEND_IS_AIC {
            // 使用 tiling 中的 matmul tiling 数据初始化
            mmFp32_.Init(&tiling_->matmulTilingFp32, pipe_);
            mmBf16_.Init(&tiling_->matmulTilingBf16, pipe_);
        }
    }

    __aicore__ inline void Init(const CGDRInitParams &initParams, GM_ADDR user)
    {
        uint64_t dataSize = tiling_->t * tiling_->nk * tiling_->dk;
        query_.SetGlobalBuffer(reinterpret_cast<__gm__ lowType *>(initParams.query), dataSize);
        key_.SetGlobalBuffer(reinterpret_cast<__gm__ lowType *>(initParams.key), dataSize);

        dataSize = tiling_->t * tiling_->nv * tiling_->dv;
        value_.SetGlobalBuffer(reinterpret_cast<__gm__ lowType *>(initParams.value), dataSize);
        out_.SetGlobalBuffer(reinterpret_cast<__gm__ lowType *>(initParams.attnOut), dataSize);

        dataSize = tiling_->t * tiling_->nv;
        beta_.SetGlobalBuffer(reinterpret_cast<__gm__ lowType *>(initParams.beta), dataSize);
        if (initParams.gOptional != nullptr) {
            g_.SetGlobalBuffer(reinterpret_cast<__gm__ highType *>(initParams.gOptional), dataSize);
        }

        dataSize = tiling_->b * tiling_->nv * tiling_->dv * tiling_->dk;
        initState_.SetGlobalBuffer(reinterpret_cast<__gm__ lowType *>(initParams.initState), dataSize);
        finalState_.SetGlobalBuffer(reinterpret_cast<__gm__ lowType *>(initParams.finalState), dataSize);

        actualSeqLens_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(initParams.seqlens), tiling_->b);

        uint64_t offset = 0;
        gCumExp_.SetGlobalBuffer(reinterpret_cast<__gm__ highType *>(user + offset));
        offset += sizeof(highType) * tiling_->nv * tiling_->maxGroupLength;

        kCumDecay_.SetGlobalBuffer(reinterpret_cast<__gm__ lowType *>(user + offset));
        offset += sizeof(lowType) * tiling_->nv * tiling_->maxGroupLength * tiling_->dk;

        vInner_.SetGlobalBuffer(reinterpret_cast<__gm__ highType *>(user + offset));
        offset += sizeof(highType) * tiling_->nv * tiling_->maxGroupLength * tiling_->dv;

        qPrime_.SetGlobalBuffer(reinterpret_cast<__gm__ lowType *>(user + offset));
        offset += sizeof(lowType) * tiling_->nv * tiling_->maxGroupLength * tiling_->dk;

        attnInter_.SetGlobalBuffer(reinterpret_cast<__gm__ highType *>(user + offset));
        offset += sizeof(highType) * tiling_->nv * tiling_->maxGroupLength * tiling_->dv;

        vNew_.SetGlobalBuffer(reinterpret_cast<__gm__ highType *>(user + offset));
        offset += sizeof(highType) * tiling_->nv * tiling_->maxGroupLength * tiling_->dv;

        kg_.SetGlobalBuffer(reinterpret_cast<__gm__ highType *>(user + offset));
        offset += sizeof(highType) * tiling_->nv * tiling_->maxGroupLength * tiling_->dk;

        qkt_.SetGlobalBuffer(reinterpret_cast<__gm__ highType *>(user + offset));
        offset += sizeof(highType) * tiling_->nv * tiling_->maxGroupLength * tiling_->chunkSize;

        stageWsAddr_ = user + offset;
        InitMatmul();
    }

    __aicore__ inline void Process()
    {
        int64_t seqStart = 0;
        int64_t seqEnd = 0;
        ChunkGroup cg;
        cg.chunkSize = tiling_->chunkSize;
        for (int64_t bid = 0; bid < tiling_->b; bid++) {
            int32_t length = actualSeqLens_.GetValue(bid);
            seqStart = seqEnd;
            seqEnd = seqStart + (int64_t)length;
            GlobalTensor<lowType> curInitState = initState_[bid * tiling_->nv * tiling_->dv * tiling_->dk];
            GlobalTensor<lowType> curFinalState = finalState_[bid * tiling_->nv * tiling_->dv * tiling_->dk];
            for (int64_t pos = seqStart; pos < seqEnd; pos += tiling_->maxGroupLength) {
                // set chunk group
                cg.startPos = pos;
                if (pos + tiling_->maxGroupLength > seqEnd) {
                    cg.length = seqEnd - pos;
                } else {
                    cg.length = tiling_->maxGroupLength;
                }
                // compute this chunk group
                stage1(cg);
                SyncAll<false>();

                if (pos > seqStart) {
                    curInitState = curFinalState;
                }
                stage2(cg, curInitState, curFinalState);
                SyncAll<false>();

                stage3(cg);
                SyncAll<false>();
            }
        }
    }

private:
    __aicore__ inline void stage1(const ChunkGroup& cg)
    {
        // todo: stage1, release ub resource after computing
        GDRStageOneInitParams initStageOneParams {query_, key_, value_, beta_, g_,
                                                  gCumExp_, kCumDecay_, vInner_, qPrime_, kg_, qkt_, stageWsAddr_, cg};
        stageOneOp_.Init(initStageOneParams, pipe_, tiling_);
        stageOneOp_.Process();
        pipe_->Reset();
    }

    __aicore__ inline void stage2(
        const ChunkGroup& cg,
        GlobalTensor<lowType>& initState,
        GlobalTensor<lowType>& finalState)
    {
        // todo: stage2, release ub resource after computing
        if (GetBlockIdx() == 23)
        {
            // Dump stage1 outputs for debugging: print first 10 and last 10 values
        int64_t nvLen = cg.length * tiling_->nv;

        // gCumExp_: (Nv, maxGroupLength) - highType
        AscendC::printf("======================================================gCumExp_ first 10:");
        AscendC::DumpTensor(gCumExp_[0], 1001, 10);
        AscendC::DumpTensor(gCumExp_[nvLen - 10], 1002, 10);

        // kCumDecay_: (Nv, maxGroupLength, Dk) - lowType
        int64_t kCumDecayLen = cg.length * tiling_->nv * tiling_->dk;
        AscendC::PRINTF("======================================================kCumDecay_ first 10:");
        AscendC::DumpTensor(kCumDecay_[0], 2001, 10);
        AscendC::PRINTF("kCumDecay_ last 10:");
        AscendC::DumpTensor(kCumDecay_[kCumDecayLen - 10], 2002, 10);

        // vInner_: (Nv, maxGroupLength, Dv) - highType
        int64_t vInnerLen = cg.length * tiling_->nv * tiling_->dv;
        AscendC::PRINTF("======================================================vInner_ first 10:");
        AscendC::DumpTensor(vInner_[0], 3001, 10);
        AscendC::DumpTensor(vInner_[vInnerLen - 10], 3002, 10);

        // qPrime_: (Nv, maxGroupLength, Dk) - lowType
        int64_t qPrimeLen = cg.length * tiling_->nv * tiling_->dk;
        AscendC::PRINTF("======================================================qPrime_ first 10:");
        AscendC::DumpTensor(qPrime_[0], 4001, 10);
        AscendC::DumpTensor(qPrime_[qPrimeLen - 10], 4002, 10);

        // kg_: (Nv, maxGroupLength, Dk) - highType
        int64_t kgLen = cg.length * tiling_->nv * tiling_->dk;
        AscendC::PRINTF("======================================================kg_ first 10:");
        AscendC::DumpTensor(kg_[0], 5001, 10);
        AscendC::DumpTensor(kg_[kgLen - 10], 5002, 10);

        // qkt_: (Nv, maxGroupLength, C) - highType
        int64_t qktLen = cg.length * tiling_->nv * tiling_->chunkSize;
        AscendC::PRINTF("======================================================qkt_ first 10:");
        AscendC::DumpTensor(qkt_[0], 6001, 10);
        AscendC::DumpTensor(qkt_[qktLen - 10], 6002, 10);
        }
        
        
    }

    __aicore__ inline void stage3(const ChunkGroup& cg)
    {
        // todo: stage3, release ub resource after computing
    }

private:
    TPipe *pipe_;
    const ChunkGatedDeltaRuleTilingData *tiling_;
    GlobalTensor<lowType> query_;
    GlobalTensor<lowType> key_;
    GlobalTensor<lowType> value_;
    GlobalTensor<lowType> beta_;
    GlobalTensor<highType> g_;
    GlobalTensor<lowType> out_;
    float scale_;
    GlobalTensor<lowType> finalState_;
    GlobalTensor<lowType> initState_;
    GlobalTensor<int32_t> actualSeqLens_;

    GlobalTensor<highType> gCumExp_;      // (Nv, maxGroupLength)
    GlobalTensor<lowType> kCumDecay_;     // (Nv, maxGroupLength, Dk)
    GlobalTensor<highType> vInner_;       // (Nv, maxGroupLength, Dv)
    GlobalTensor<lowType> qPrime_;        // (Nv, maxGroupLength, Dk)
    GlobalTensor<highType> attnInter_;    // (Nv, maxGroupLength, Dv)
    GlobalTensor<highType> vNew_;         // (Nv, maxGroupLength, Dv)
    GlobalTensor<highType> kg_;           // (Nv, maxGroupLength, Dk)
    GlobalTensor<highType> qkt_;          // (Nv, maxGroupLength, C)
    GM_ADDR stageWsAddr_;                 // temporary space addr for stages

    // Matmul objects
    MT_FP32 mmFp32_;
    MT_BF16 mmBf16_;

    // Stage operators
    GDRStageOne stageOneOp_;

};

} // namespace ChunkGatedDeltaRule
#endif  // __CHUNK_GATED_DELTA_RULE_H_