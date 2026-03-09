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
 * \file moe_finalize_routing_v2_h_partial_load.h
 * \brief
 */

#ifndef MOE_FINALIZE_ROUTING_V2_H_PARTIAL_LOAD_H_
#define MOE_FINALIZE_ROUTING_V2_H_PARTIAL_LOAD_H_

#include "moe_finalize_routing_v2_base.h"
namespace MoeFinalizeRoutingV2Regbase {
using namespace AscendC;
template <typename T, typename S, int32_t dropPadMode>
class MoeFinalizeRoutingV2HPartialLoad
{
public:
    __aicore__ inline MoeFinalizeRoutingV2HPartialLoad()
    {}

    __aicore__ inline void Init(
        GM_ADDR expandedX, GM_ADDR expandedRowIdx, GM_ADDR x1, GM_ADDR x2, GM_ADDR bias, GM_ADDR scales,
        GM_ADDR expertIdx,  GM_ADDR x, GM_ADDR constExpertAlpha1, GM_ADDR constExpertAlpha2, GM_ADDR v,
        GM_ADDR y, GM_ADDR workspace, const MoeFinalizeRoutingV2RegbaseTilingData* tilingDataPtr,
        TPipe* pipePtr)
    {
        hasBiasAndExpertIdx = (bias != nullptr) && (expertIdx != nullptr);
        hasScales = scales != nullptr;

        pipe = pipePtr;
        tilingData = tilingDataPtr;
        hasX1 = x1 != nullptr;
        hasX2 = x2 != nullptr;

        hasX = x != nullptr;
        hasConstExpert = (constExpertAlpha1 != nullptr) && (constExpertAlpha2 != nullptr) && (v != nullptr);

        expandedXGm.SetGlobalBuffer((__gm__ T*)expandedX);
        expandedRowIdxGm.SetGlobalBuffer((__gm__ int32_t*)expandedRowIdx);
        biasGm.SetGlobalBuffer((__gm__ T*)bias);
        scalesGm.SetGlobalBuffer((__gm__ S*)scales);
        x1Gm.SetGlobalBuffer((__gm__ T*)x1);
        x2Gm.SetGlobalBuffer((__gm__ T*)x2);
        expertIdxGm.SetGlobalBuffer((__gm__ int32_t*)expertIdx);
        xGm.SetGlobalBuffer((__gm__ T*)x);
        constExpertAlpha1Gm.SetGlobalBuffer((__gm__ T*)constExpertAlpha1);
        constExpertAlpha2Gm.SetGlobalBuffer((__gm__ T*)constExpertAlpha2);
        vGm.SetGlobalBuffer((__gm__ T*)v);
        yGm.SetGlobalBuffer((__gm__ T*)y);

        int32_t hFactorAlignedT = RoundUp<T>(tilingData->hFactor);
        int32_t hFactorAlignedFloat = RoundUp<float>(tilingData->hFactor);
        pipe->InitBuffer(expandedXQue, DOUBLE_BUFFER, hFactorAlignedT * sizeof(T));
        pipe->InitBuffer(yQue, DOUBLE_BUFFER, hFactorAlignedFloat * sizeof(float));
        if (hasBiasAndExpertIdx) {
            pipe->InitBuffer(biasQue, DOUBLE_BUFFER, hFactorAlignedT * sizeof(T));
        }
        if (hasX1) {
            pipe->InitBuffer(x1Que, DOUBLE_BUFFER, hFactorAlignedT * sizeof(T));
        }
        if (hasX2) {
            pipe->InitBuffer(x2Que, DOUBLE_BUFFER, hFactorAlignedT * sizeof(T));
        }
        if (hasX) {
            // X的大小是row_num, h
            pipe->InitBuffer(xQue, DOUBLE_BUFFER, hFactorAlignedT * sizeof(T));
        }
        if (hasBiasAndExpertIdx) {
            pipe->InitBuffer(constExpertAlpha1Que, DOUBLE_BUFFER, hFactorAlignedT * sizeof(T));
            pipe->InitBuffer(constExpertAlpha2Que, DOUBLE_BUFFER, hFactorAlignedT * sizeof(T));
            pipe->InitBuffer(vQue, DOUBLE_BUFFER, hFactorAlignedT * sizeof(T));
        }
    }

    __aicore__ inline void Process()
    {
        int64_t rowOuterLoop =
            (GetBlockIdx() == GetBlockNum() - 1) ? tilingData->rowLoopOfTailBlock : tilingData->rowLoopOfFormerBlock;
        for (int64_t rowOuterIdx = 0; rowOuterIdx < rowOuterLoop; rowOuterIdx += 1) {
            for (int64_t hIdx = 0; hIdx < tilingData->hLoop; hIdx += 1) {
                int64_t hFactor = (hIdx == tilingData->hLoop - 1) ? tilingData->tailHFactor : tilingData->hFactor;
                if (hasX1) {
                    x1Local = x1Que.AllocTensor<T>();
                    int64_t x1GmOffset = GetBlockIdx() * tilingData->rowOfFormerBlock * tilingData->h +
                                         rowOuterIdx * tilingData->h + hIdx * tilingData->hFactor;
                    CopyIn(x1Gm[x1GmOffset], x1Local, 1, hFactor);
                    x1Que.EnQue(x1Local);
                    x1Local = x1Que.DeQue<T>();
                }
                if (hasX2) {
                    x2Local = x2Que.AllocTensor<T>();
                    int64_t x2GmOffset = GetBlockIdx() * tilingData->rowOfFormerBlock * tilingData->h +
                                         rowOuterIdx * tilingData->h + hIdx * tilingData->hFactor;
                    CopyIn(x2Gm[x2GmOffset], x2Local, 1, hFactor);
                    x2Que.EnQue(x2Local);
                    x2Local = x2Que.DeQue<T>();
                }

                yLocal = yQue.AllocTensor<float>();
                ProcessX1AndX2(yLocal, x1Local, x2Local, hFactor, hasX1, hasX2);
                ProcessYWithInput(rowOuterIdx, hIdx, hFactor);
                yQue.EnQue(yLocal);

                if (hasX1) {
                    x1Que.FreeTensor(x1Local);
                }
                if (hasX2) {
                    x2Que.FreeTensor(x2Local);
                }
                yLocal = yQue.DeQue<float>();
                int64_t yGmOffset = GetBlockIdx() * tilingData->rowOfFormerBlock * tilingData->h +
                                    rowOuterIdx * tilingData->h + hIdx * tilingData->hFactor;
                CopyOut(yLocal.template ReinterpretCast<T>(), yGm[yGmOffset], 1, hFactor);
                yQue.FreeTensor(yLocal);
            }
        }
    }

private:
    __aicore__ inline void ProcessYWithInput(int64_t rowOuterIdx, int64_t hIdx, int64_t hFactor)
    {
        for (int64_t kIdx = 0; kIdx < tilingData->k; kIdx += 1) {
            SetExpandedRowIdxOffset(rowOuterIdx, kIdx);
            int64_t expandedRowIdxGmValue = expandedRowIdxGm.GetValue(expandedRowIdxOffset);
            if (expandedRowIdxGmValue == INVALID_IDX) {
                continue;
            }
            if constexpr (dropPadMode != DROP_PAD_COLUMN && dropPadMode != DROP_PAD_ROW) {
                if (expandedRowIdxGmValue >= tilingData->activeNum) {
                    continue;
                }
            }
            expandedXLocal = expandedXQue.AllocTensor<T>();
            CopyIn(
                expandedXGm[expandedRowIdxGmValue * tilingData->h + hIdx * tilingData->hFactor], expandedXLocal, 1,
                hFactor);
            expandedXQue.EnQue(expandedXLocal);
            int64_t expertIdx = expertIdxGm.GetValue(expertIdxOffset);
            if (expertIdx >= tilingData->zeroExpertStart && expertIdx < tilingData->zeroExpertEnd) {
                continue;
            }
            if (expertIdx >= tilingData->copyExpertStart && expertIdx < tilingData->copyExpertEnd) {
                // x = x[i]
                int64_t i = expertIdxOffset / tilingData->k;
                int64_t xGmOffset = i * tilingData->h + hIdx * hFactor;
                CopyIn(xGm[xGmOffset], xLocal, 1, hFactor); 
                AscendC::Copy(expandedXLocal[validK * tilingData->hAligned], xLocal, tilingData->h);
            }
            if (expertIdx >= tilingData->constantExpertStart && expertIdx < tilingData->constantExpertEnd) {
                // x = a1 * x[i] +  a2 * v
                int64_t i = expertIdxOffset / tilingData->k;
                int64_t xGmOffset = i * tilingData->h + hIdx * hFactor;
                // 不需要有偏移，用完就下一个循环覆盖掉就行
                CopyIn(xGm[xGmOffset], xLocal, 1, hFactor); 
                int64_t constExpertGmOffset = (expertIdx - tilingData->constantExpertStart) * hFactor;
                CopyIn(constExpertAlpha1Gm[constExpertGmOffset], constExpertAlpha1Local, 1, hFactor); 
                CopyIn(constExpertAlpha2Gm[constExpertGmOffset], constExpertAlpha2Local, 1, hFactor); 
                CopyIn(vGm[constExpertGmOffset], vLocal, 1, hFactor); 
                vLocal = vLocal * constExpertAlpha2Local;
                xLocal = xLocal * constExpertAlpha1Local;
                xLocal = xLocal + vLocal;
                AscendC::Copy(expandedXLocal[validK * tilingData->hAligned], xLocal, tilingData->h);
                constExpertAlpha1Que.EnQue(constExpertAlpha1Local);
                constExpertAlpha1Local = constExpertAlpha1Que.DeQue<T>();
                constExpertAlpha2Que.EnQue(constExpertAlpha2Local);
                constExpertAlpha2Local = constExpertAlpha2Que.DeQue<T>();
                vQue.EnQue(vLocal);
                vLocal = vQue.DeQue<T>();
            }
            xQue.EnQue(xLocal);
            xLocal = xQue.DeQue<T>();
            if (hasBiasAndExpertIdx) {
                SetExpertIdxOffset(rowOuterIdx, kIdx);
                int64_t biasGmOffset =
                    expertIdxGm.GetValue(expertIdxOffset) * tilingData->h + hIdx * tilingData->hFactor;
                biasLocal = biasQue.AllocTensor<T>();
                CopyIn(biasGm[biasGmOffset], biasLocal, 1, hFactor);
                biasQue.EnQue(biasLocal);
            }
            if (hasScales) {
                SetScaleOffset(rowOuterIdx, kIdx);
                scale = scalesGm.GetValue(scaleOffset);
            }
            expandedXLocal = expandedXQue.DeQue<T>();
            if (hasBiasAndExpertIdx) {
                biasLocal = biasQue.DeQue<T>();
            }
            ProcessExpandXBiasScale<T, S>(
                yLocal, expandedXLocal, biasLocal, scale, hFactor, hasBiasAndExpertIdx, hasScales);
            expandedXQue.FreeTensor(expandedXLocal);
            if (hasBiasAndExpertIdx) {
                biasQue.FreeTensor(biasLocal);
            }
            if (hasX) {
                xQue.FreeTensor(xLocal);
            }
            if (hasConstExpert) {
                constExpertAlpha1Que.FreeTensor(constExpertAlpha1Local);
                constExpertAlpha2Que.FreeTensor(constExpertAlpha2Local);
                vQue.FreeTensor(vLocal);
            }
        }
        if constexpr (IsSameType<T, half>::value || IsSameType<T, bfloat16_t>::value) {
            Cast(yLocal.template ReinterpretCast<T>(), yLocal, RoundMode::CAST_RINT, hFactor);
        }
    }

    __aicore__ inline void SetExpandedRowIdxOffset(int64_t rowOuterIdx, int64_t kIdx)
    {
        if constexpr (dropPadMode == DROPLESS_COLUMN || dropPadMode == DROP_PAD_COLUMN) {
            expandedRowIdxOffset = GetBlockIdx() * tilingData->rowOfFormerBlock + rowOuterIdx + kIdx * tilingData->row;
        } else {
            expandedRowIdxOffset =
                GetBlockIdx() * tilingData->rowOfFormerBlock * tilingData->k + rowOuterIdx * tilingData->k + kIdx;
        }
    }

    __aicore__ inline void SetExpertIdxOffset(int64_t rowOuterIdx, int64_t kIdx)
    {
        if constexpr (dropPadMode == DROPLESS_COLUMN || dropPadMode == DROP_PAD_COLUMN) {
            expertIdxOffset =
                GetBlockIdx() * tilingData->rowOfFormerBlock * tilingData->k + rowOuterIdx * tilingData->k + kIdx;
        } else {
            expertIdxOffset = expandedRowIdxOffset;
        }
    }

    __aicore__ inline void SetScaleOffset(int64_t rowOuterIdx, int64_t kIdx)
    {
        if constexpr (dropPadMode == DROPLESS_COLUMN || dropPadMode == DROP_PAD_COLUMN) {
            scaleOffset =
                GetBlockIdx() * tilingData->rowOfFormerBlock * tilingData->k + rowOuterIdx * tilingData->k + kIdx;
        } else {
            scaleOffset = expandedRowIdxOffset;
        }
    }

private:
    TPipe* pipe;
    const MoeFinalizeRoutingV2RegbaseTilingData* tilingData;
    GlobalTensor<T> expandedXGm;
    GlobalTensor<int32_t> expandedRowIdxGm;
    GlobalTensor<int32_t> expertIdxGm;
    GlobalTensor<T> yGm;
    GlobalTensor<T> x1Gm;
    GlobalTensor<T> x2Gm;
    GlobalTensor<T> xGm;
    GlobalTensor<T> constExpertAlpha1Gm;
    GlobalTensor<T> constExpertAlpha2Gm;
    GlobalTensor<T> vGm;

    int64_t expandedRowIdxOffset{0};
    int64_t expertIdxOffset{0};
    int64_t scaleOffset{0};

    GlobalTensor<T> biasGm;
    GlobalTensor<S> scalesGm;

    LocalTensor<T> expandedXLocal;
    LocalTensor<T> x1Local;
    LocalTensor<T> x2Local;
    LocalTensor<T> biasLocal;
    LocalTensor<T> xLocal;
    LocalTensor<T> constExpertAlpha1Local;
    LocalTensor<T> constExpertAlpha2Local;
    LocalTensor<T> vLocal;
    LocalTensor<float> yLocal;

    TQue<QuePosition::VECIN, DOUBLE_BUFFER> expandedXQue;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> biasQue;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> x1Que;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> x2Que;

    TQue<QuePosition::VECIN, DOUBLE_BUFFER> xQue;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> constExpertAlpha1Que;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> constExpertAlpha2Que;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> vQue;

    TQue<QuePosition::VECOUT, DOUBLE_BUFFER> yQue;

    S scale;
    bool hasX1{false};
    bool hasX2{false};
    bool hasBiasAndExpertIdx{false};
    bool hasScales{false};
    bool hasX{false};
    bool hasConstExpert{false};
};
} // namespace MoeFinalizeRoutingV2Regbase

#endif