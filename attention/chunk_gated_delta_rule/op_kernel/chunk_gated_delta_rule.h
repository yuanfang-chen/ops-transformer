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

namespace ChunkGatedDeltaRule {
    
using namespace AscendC;
    
constexpr int32_t BUFFER_NUM = 2;

struct ChunkGatedDeltaRuleInitParams {
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
class ChunkGatedDeltaRule {
public:
    __aicore__ inline ChunkGatedDeltaRule(TPipe *pipe, const RecurrentGatedDeltaRuleTilingData *tilingData)
    {
        pipe_ = pipe;
        tiling_ = tilingData;
    };

    __aicore__ inline void Init(const ChunkGatedDeltaRuleInitParams &initParams, GM_ADDR user)
    {
        uint64_t dataSize = tiling_.t * tiling_.nk * tiling_.dk;
        query_.SetGlobalBuffer(reinterpret_cast<__gm__ lowType *>(initParams.query), dataSize);
        key_.SetGlobalBuffer(reinterpret_cast<__gm__ lowType *>(initParams.key), dataSize);

        dataSize = tiling_.t * tiling_.nv * tiling_.dv;
        value_.SetGlobalBuffer(reinterpret_cast<__gm__ lowType *>(initParams.value), dataSize);
        out_.SetGlobalBuffer(reinterpret_cast<__gm__ lowType *>(initParams.attnOut), dataSize);

        dataSize = tiling_.t * tiling_.nv;
        beta_.SetGlobalBuffer(reinterpret_cast<__gm__ lowType *>(initParams.beta), dataSize);
        if (initParams.gOptional != nullptr) {
            g_.SetGlobalBuffer(reinterpret_cast<__gm__ lowType *>(initParams.gOptional), dataSize);
        }

        dataSize = tiling_.b * tiling_.nv * tiling_.dv * tiling_.dk;
        initState_.SetGlobalBuffer(reinterpret_cast<__gm__ lowType *>(initParams.initState), dataSize);
        finalState_.SetGlobalBuffer(reinterpret_cast<__gm__ lowType *>(initParams.finalState), dataSize);

        actualSeqLens_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(initParams.seqlens), tiling_.b);

        uint64_t offset = 0;
        gCumExp_.SetGlobalBuffer(reinterpret_cast<__gm__ highType *>(user + offset));
        offset += sizeof(highType) * tiling_.nv * tiling_.maxGroupLength;

        kCumDecay_.SetGlobalBuffer(reinterpret_cast<__gm__ lowType *>(user + offset));
        offset += sizeof(lowType) * tiling_.nv * tiling_.maxGroupLength * tiling_.dk;

        vInner_.SetGlobalBuffer(reinterpret_cast<__gm__ highType *>(user + offset));
        offset += sizeof(highType) * tiling_.nv * tiling_.maxGroupLength * tiling_.dv;

        qPrime_.SetGlobalBuffer(reinterpret_cast<__gm__ lowType *>(user + offset));
        offset += sizeof(lowType) * tiling_.nv * tiling_.maxGroupLength * tiling_.dk;

        attnInter_.SetGlobalBuffer(reinterpret_cast<__gm__ highType *>(user + offset));
        offset += sizeof(highType) * tiling_.nv * tiling_.maxGroupLength * tiling_.dv;

        vNew_.SetGlobalBuffer(reinterpret_cast<__gm__ highType *>(user + offset));
        offset += sizeof(highType) * tiling_.nv * tiling_.maxGroupLength * tiling_.dv;

        kg_.SetGlobalBuffer(reinterpret_cast<__gm__ highType *>(user + offset));
        offset += sizeof(highType) * tiling_.nv * tiling_.maxGroupLength * tiling_.dk;

        qkt_.SetGlobalBuffer(reinterpret_cast<__gm__ highType *>(user + offset));
        offset += sizeof(highType) * tiling_.nv * tiling_.maxGroupLength * tiling_.chunkSize;

        stageWsAddr_ = user + offset;
    }

    __aicore__ inline void Process()
    {
        int64_t seqStart = 0;
        int64_t seqEnd = 0;
        ChunkGroup cg;
        cg.chunkSize = tiling_.chunkSize;
        for (int64_t bid = 0; bid < tiling_.b; bid++) {
            int32_t length = actualSeqLens_.GetValue(bid);
            seqEnd = seqStart + (int64_t)length;
            GlobalTensor<lowType> curInitState = initState_[bid * tiling_.nv * tiling_.dv * tiling_.dk];
            GlobalTensor<lowType> curFinalState = finalState_[bid * tiling_.nv * tiling_.dv * tiling_.dk];
            for (int64_t pos = seqStart; pos < seqEnd; pos += tiling_.maxGroupLength) {
                // set chunk group
                cg.startPos = pos;
                if (pos + tiling_.maxGroupLength > seqEnd) {
                    cg.length = seqEnd - pos;
                } else {
                    cg.length = tiling_.maxGroupLength;
                }

                // compute this chunk group
                stage1(cg);

                if (pos > seqStart) {
                    curInitState = curFinalState;
                }
                stage2(cg, curInitState, curFinalState);

                stage3(cg);
            }
        }
    }

private:
    __aicore__ inline void stage1(const ChunkGroup& cg)
    {
        // todo: stage1, release ub resource after computing
    }

    __aicore__ inline void stage2(
        const ChunkGroup& cg,
        GlobalTensor<lowType>& initState,
        GlobalTensor<lowType>& finalState)
    {
        // todo: stage2, release ub resource after computing
    }

    __aicore__ inline void stage3(const ChunkGroup& cg)
    {
        // todo: stage3, release ub resource after computing
    }

private:
    TPipe *pipe_;
    const RecurrentGatedDeltaRuleTilingData *tiling_;
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

};

} // namespace ChunkGatedDeltaRule
#endif  // __CHUNK_GATED_DELTA_RULE_H_