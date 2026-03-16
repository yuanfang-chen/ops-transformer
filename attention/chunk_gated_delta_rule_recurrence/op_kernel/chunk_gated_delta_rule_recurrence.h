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
 * \file chunk_gated_delta_rule_recurrence.h
 * \brief AIV-only kernel for ChunkGatedDeltaRuleRecurrence (Ascend 950 / DAV_3510)
 *
 * Algorithm per chunk i for task (bIdx, hIdx):
 *   v_prime   = k_cumdecay[h,i] @ state[b,h].T      [cs,dk] @ [dk,dv] -> [cs,dv]
 *   attn_out  = qgexp[h,i]      @ state[b,h].T       [cs,dk] @ [dk,dv] -> [cs,dv]
 *   v_new     = value[h,i]  - v_prime                element-wise [cs,dv]
 *   state    *= gexp[h,i,cs-1,0]                     scalar decay
 *   state    += v_new.T @ kgexp[h,i]                [dv,cs] @ [cs,dk] -> [dv,dk]
 *
 * Memory strategy:
 *  - Chunk matrices (kCumdecay/qgexp/kgexp) are loaded ONCE per chunk and held
 *    across all dvTile iterations, then freed at end of chunk.
 *  - C1/C2 results are computed directly into output-queue tensors to save UB.
 *  - tmpBuf_ holds only: stateUb_[dvTile_*alignDk_] + scratchUb_[alignDk_].
 *
 * Total UB = 3*alignCs*alignDk*4 + (3*alignCs + alignDk)*dvTile_*4 + alignDk*4 + reserve
 */

#ifndef __CHUNK_GATED_DELTA_RULE_RECURRENCE_KERNEL_H_
#define __CHUNK_GATED_DELTA_RULE_RECURRENCE_KERNEL_H_

#include "kernel_operator.h"
#include "chunk_gated_delta_rule_recurrence_tiling_data.h"

namespace ChunkGatedDeltaRuleRecurrence {

using namespace AscendC;

constexpr uint64_t BUFFER_NUM     = 1;
constexpr uint32_t FP32_PER_BLOCK = 8;   // 8 × 4B = 32B

class CGDR {
public:
    __aicore__ inline CGDR(const ChunkGatedDeltaRuleRecurrenceTilingData *td)
    {
        b_            = td->b;
        hv_           = td->hv;
        realDk_       = td->realDk;
        alignDk_      = td->alignDk;
        realDv_       = td->realDv;
        alignDv_      = td->alignDv;
        nChunks_      = td->nChunks;
        realCs_       = td->realChunkSize;
        alignCs_      = td->alignChunkSize;
        dvTile_       = td->dvTile;
        totalTasks_   = td->totalTasks;
        tasksPerCore_ = td->tasksPerCore;
    }

    __aicore__ inline void Init(GM_ADDR initialState, GM_ADDR kgexp, GM_ADDR value,
                                GM_ADDR kCumdecay, GM_ADDR qgexp, GM_ADDR gexp,
                                GM_ADDR cuSeqlens, GM_ADDR attnInterOut,
                                GM_ADDR vNewOut, TPipe *pipe)
    {
        blockIdx_ = GetBlockIdx();
        pipe_     = pipe;
        stateGm_.SetGlobalBuffer((__gm__ float *)initialState);
        kgexpGm_.SetGlobalBuffer((__gm__ float *)kgexp);
        valueGm_.SetGlobalBuffer((__gm__ float *)value);
        kCumdecayGm_.SetGlobalBuffer((__gm__ float *)kCumdecay);
        qgexpGm_.SetGlobalBuffer((__gm__ float *)qgexp);
        gexpGm_.SetGlobalBuffer((__gm__ float *)gexp);
        cuSeqlensGm_.SetGlobalBuffer((__gm__ int32_t *)cuSeqlens);
        attnInterOutGm_.SetGlobalBuffer((__gm__ float *)attnInterOut);
        vNewOutGm_.SetGlobalBuffer((__gm__ float *)vNewOut);
        InitBuffers();
    }

    __aicore__ inline void Process()
    {
        uint32_t taskBegin = blockIdx_ * tasksPerCore_;
        uint32_t taskEnd   = taskBegin + tasksPerCore_;
        if (taskEnd > totalTasks_) taskEnd = totalTasks_;
        for (uint32_t task = taskBegin; task < taskEnd; task++) {
            ProcessTask(task / hv_, task % hv_);
        }
    }

private:
    // ──────────────────────────────────────────────────────────────────────
    __aicore__ inline void InitBuffers()
    {
        pipe_->InitBuffer(kCumdecayQ_, BUFFER_NUM, alignCs_ * alignDk_ * sizeof(float));
        pipe_->InitBuffer(qgexpQ_,     BUFFER_NUM, alignCs_ * alignDk_ * sizeof(float));
        pipe_->InitBuffer(kgexpQ_,     BUFFER_NUM, alignCs_ * alignDk_ * sizeof(float));
        pipe_->InitBuffer(valueQ_,     BUFFER_NUM, alignCs_ * dvTile_ * sizeof(float));
        pipe_->InitBuffer(attnOutQ_,   BUFFER_NUM, alignCs_ * dvTile_ * sizeof(float));
        pipe_->InitBuffer(vNewOutQ_,   BUFFER_NUM, alignCs_ * dvTile_ * sizeof(float));
        // tmpBuf_: stateUb_ [dvTile_*alignDk_] + scratchUb_ [alignDk_]
        pipe_->InitBuffer(tmpBuf_, (dvTile_ * alignDk_ + alignDk_) * sizeof(float));
        uint32_t off = 0;
        stateUb_   = tmpBuf_.GetWithOffset<float>(dvTile_ * alignDk_, off); off += dvTile_ * alignDk_;
        scratchUb_ = tmpBuf_.GetWithOffset<float>(alignDk_,            off);
    }

    // ──────────────────────────────────────────────────────────────────────
    // Load chunk matrix [realCs_, realDk_] → local [alignCs_, alignDk_] with padding
    __aicore__ inline void CopyInChunkMat(TQue<QuePosition::VECIN, 1> &q,
                                           LocalTensor<float> &dst,
                                           GlobalTensor<float> &src,
                                           uint64_t gmOff)
    {
        LocalTensor<float> local = q.AllocTensor<float>();
        DataCopyExtParams       cp{static_cast<uint16_t>(realCs_),
                                   static_cast<uint32_t>(realDk_ * sizeof(float)),
                                   0, 0, 0};
        DataCopyPadExtParams<float> pp{true, 0,
                                       static_cast<uint8_t>(alignDk_ - realDk_), 0};
        DataCopyPad(local, src[gmOff], cp, pp);
        q.EnQue<float>(local);
        dst = q.DeQue<float>();
    }

    // Load value slice [realCs_, curDvTile] from GM (row-stride realDv_)
    // into local [alignCs_, dvTile_] (row-stride dvTile_) with end-of-row padding
    __aicore__ inline void CopyInValueSlice(LocalTensor<float> &dst,
                                             uint64_t gmOff, uint32_t curDvTile)
    {
        LocalTensor<float> local = valueQ_.AllocTensor<float>();
        DataCopyExtParams       cp{static_cast<uint16_t>(realCs_),
                                   static_cast<uint32_t>(curDvTile * sizeof(float)),
                                   static_cast<uint32_t>((realDv_ - curDvTile) * sizeof(float)),
                                   0, 0};
        DataCopyPadExtParams<float> pp{true, 0,
                                       static_cast<uint8_t>(dvTile_ - curDvTile), 0};
        DataCopyPad(local, valueGm_[gmOff], cp, pp);
        valueQ_.EnQue<float>(local);
        dst = valueQ_.DeQue<float>();
    }

    // Load state slice [curDvTile, realDk_] from GM into stateUb_ [dvTile_, alignDk_]
    __aicore__ inline void CopyInStateSlice(uint64_t gmOff, uint32_t curDvTile)
    {
        DataCopyExtParams       cp{static_cast<uint16_t>(curDvTile),
                                   static_cast<uint32_t>(realDk_ * sizeof(float)),
                                   0, 0, 0};
        DataCopyPadExtParams<float> pp{true, 0,
                                       static_cast<uint8_t>(alignDk_ - realDk_), 0};
        DataCopyPad(stateUb_, stateGm_[gmOff], cp, pp);
    }

    // Write stateUb_ [curDvTile, alignDk_] → GM [curDvTile, realDk_], one row at a time
    __aicore__ inline void CopyOutStateSlice(uint64_t gmOff, uint32_t curDvTile)
    {
        DataCopyParams cp{1, static_cast<uint16_t>(realDk_ * sizeof(float)), 0, 0};
        for (uint32_t d = 0; d < curDvTile; d++) {
            DataCopyPad(stateGm_[gmOff + static_cast<uint64_t>(d) * realDk_],
                        stateUb_[d * alignDk_], cp);
        }
    }

    // Write local [alignCs_, dvTile_] → GM [realCs_, realDv_], one row at a time
    __aicore__ inline void CopyOutSlice(GlobalTensor<float> &dst,
                                         LocalTensor<float> &src,
                                         uint64_t gmBaseOff, uint32_t curDvTile)
    {
        DataCopyParams cp{1, static_cast<uint16_t>(curDvTile * sizeof(float)), 0, 0};
        for (uint32_t s = 0; s < realCs_; s++) {
            DataCopyPad(dst[gmBaseOff + static_cast<uint64_t>(s) * realDv_],
                        src[s * dvTile_], cp);
        }
    }

    // ──────────────────────────────────────────────────────────────────────
    // One (chunk, dvTile) slice computation.
    // Chunk matrices (kCumdecayUb_, qgexpUb_, kgexpUb_) and gexpScalar are
    // pre-loaded by the caller and remain valid for this call.
    // ──────────────────────────────────────────────────────────────────────
    __aicore__ inline void ComputeChunkTile(uint32_t hIdx, uint32_t chunkIdx,
                                             uint32_t bIdx, uint32_t dvOff,
                                             uint32_t curDvTile, float gexpScalar)
    {
        // GM offsets for value/attn/vnew slices [hv, nChunks, cs, dv] column dvOff
        uint64_t valBase   = (static_cast<uint64_t>(hIdx) * nChunks_ + chunkIdx)
                             * static_cast<uint64_t>(realCs_) * realDv_ + dvOff;
        // GM offset for state [b, hv, dv, dk] row dvOff
        uint64_t stateBase = ((static_cast<uint64_t>(bIdx) * hv_ + hIdx) * realDv_ + dvOff)
                             * realDk_;

        // Load value and state slices
        LocalTensor<float> valueLocal;
        CopyInValueSlice(valueLocal, valBase, curDvTile);
        CopyInStateSlice(stateBase, curDvTile);

        // Allocate output tensors from queues
        LocalTensor<float> attnLocal = attnOutQ_.AllocTensor<float>();
        LocalTensor<float> vNewLocal = vNewOutQ_.AllocTensor<float>();

        // ── C1: vNewLocal[s, d] = vPrime = dot(kCumdecay[s,:], state[d,:])
        // ── C2: attnLocal[s, d] = dot(qgexp[s,:], state[d,:])
        // Both computed directly into their output queue tensors.
        // Layout: [realCs_, dvTile_] row-major, vNewLocal[s*dvTile_+d], attnLocal[s*dvTile_+d]
        for (uint32_t s = 0; s < realCs_; s++) {
            for (uint32_t d = 0; d < curDvTile; d++) {
                float accV = 0.0f, accA = 0.0f;
                for (uint32_t k = 0; k < realDk_; k++) {
                    float sk = stateUb_.GetValue(d * alignDk_ + k);
                    accV += kCumdecayUb_.GetValue(s * alignDk_ + k) * sk;
                    accA += qgexpUb_.GetValue(s * alignDk_ + k)     * sk;
                }
                vNewLocal.SetValue(s * dvTile_ + d, accV);
                attnLocal.SetValue(s * dvTile_ + d, accA);
            }
        }
        AscendC::PipeBarrier<PIPE_V>();

        // ── V1: vNew = value - vPrime (in-place in vNewLocal) ──
        Sub(vNewLocal, valueLocal, vNewLocal, realCs_ * dvTile_);
        AscendC::PipeBarrier<PIPE_V>();

        // ── V0: state *= gexpScalar ──
        Muls(stateUb_, stateUb_, gexpScalar, curDvTile * alignDk_);
        AscendC::PipeBarrier<PIPE_V>();

        // ── C3: state[d,:] += sum_s( vNew[s,d] * kgexp[s,:] )
        // Vectorized: for each (s,d), state[d,:] += scalar * kgexp[s,:]
        for (uint32_t s = 0; s < realCs_; s++) {
            for (uint32_t d = 0; d < curDvTile; d++) {
                float vnVal = vNewLocal.GetValue(s * dvTile_ + d);
                Muls(scratchUb_, kgexpUb_[s * alignDk_], vnVal, alignDk_);
                AscendC::PipeBarrier<PIPE_V>();
                Add(stateUb_[d * alignDk_], stateUb_[d * alignDk_], scratchUb_, alignDk_);
                AscendC::PipeBarrier<PIPE_V>();
            }
        }

        // ── Write outputs ──
        attnOutQ_.EnQue<float>(attnLocal);
        vNewOutQ_.EnQue<float>(vNewLocal);
        LocalTensor<float> attnOut = attnOutQ_.DeQue<float>();
        LocalTensor<float> vNewOut = vNewOutQ_.DeQue<float>();
        CopyOutSlice(attnInterOutGm_, attnOut, valBase, curDvTile);
        CopyOutSlice(vNewOutGm_,      vNewOut, valBase, curDvTile);
        attnOutQ_.FreeTensor(attnOut);
        vNewOutQ_.FreeTensor(vNewOut);

        // Write updated state back
        CopyOutStateSlice(stateBase, curDvTile);
        valueQ_.FreeTensor(valueLocal);
    }

    // ──────────────────────────────────────────────────────────────────────
    __aicore__ inline void ProcessTask(uint32_t bIdx, uint32_t hIdx)
    {
        int32_t chunkBegin = cuSeqlensGm_.GetValue(bIdx)     / static_cast<int32_t>(realCs_);
        int32_t chunkEnd   = cuSeqlensGm_.GetValue(bIdx + 1) / static_cast<int32_t>(realCs_);

        for (int32_t ci = chunkBegin; ci < chunkEnd; ci++) {
            // GM base for chunk matrices [hv, nChunks, cs, dk]
            uint64_t chunkBase = (static_cast<uint64_t>(hIdx) * nChunks_ + ci)
                                 * static_cast<uint64_t>(realCs_) * realDk_;

            // Load chunk matrices once (held across dvTile iterations)
            CopyInChunkMat(kCumdecayQ_, kCumdecayUb_, kCumdecayGm_, chunkBase);
            CopyInChunkMat(qgexpQ_,     qgexpUb_,     qgexpGm_,     chunkBase);
            CopyInChunkMat(kgexpQ_,     kgexpUb_,     kgexpGm_,     chunkBase);

            // gexp scalar: last token of chunk = gexp[h, ci, cs-1, 0]
            uint64_t gexpOff = (static_cast<uint64_t>(hIdx) * nChunks_ + ci)
                               * realCs_ + (realCs_ - 1);
            float gexpScalar = gexpGm_.GetValue(gexpOff);

            for (uint32_t dvOff = 0; dvOff < realDv_; dvOff += dvTile_) {
                uint32_t curDvTile = (dvOff + dvTile_ <= realDv_) ? dvTile_
                                                                   : (realDv_ - dvOff);
                ComputeChunkTile(hIdx, static_cast<uint32_t>(ci), bIdx,
                                 dvOff, curDvTile, gexpScalar);
            }

            // Release chunk matrices after all dvTile slices are done
            kCumdecayQ_.FreeTensor(kCumdecayUb_);
            qgexpQ_.FreeTensor(qgexpUb_);
            kgexpQ_.FreeTensor(kgexpUb_);
        }
    }

    // ──────────────────────────────────────────────────────────────────────
    GlobalTensor<float>   stateGm_;
    GlobalTensor<float>   kgexpGm_;
    GlobalTensor<float>   valueGm_;
    GlobalTensor<float>   kCumdecayGm_;
    GlobalTensor<float>   qgexpGm_;
    GlobalTensor<float>   gexpGm_;
    GlobalTensor<int32_t> cuSeqlensGm_;
    GlobalTensor<float>   attnInterOutGm_;
    GlobalTensor<float>   vNewOutGm_;

    TPipe *pipe_;

    TQue<QuePosition::VECIN, 1>  kCumdecayQ_;
    TQue<QuePosition::VECIN, 1>  qgexpQ_;
    TQue<QuePosition::VECIN, 1>  kgexpQ_;
    TQue<QuePosition::VECIN, 1>  valueQ_;
    TQue<QuePosition::VECOUT, 1> attnOutQ_;
    TQue<QuePosition::VECOUT, 1> vNewOutQ_;
    TBuf<TPosition::VECCALC>     tmpBuf_;

    LocalTensor<float> kCumdecayUb_;
    LocalTensor<float> qgexpUb_;
    LocalTensor<float> kgexpUb_;
    LocalTensor<float> stateUb_;
    LocalTensor<float> scratchUb_;

    uint32_t b_, hv_;
    uint32_t realDk_, alignDk_;
    uint32_t realDv_, alignDv_;
    uint32_t nChunks_;
    uint32_t realCs_, alignCs_;
    uint32_t dvTile_;
    uint32_t totalTasks_, tasksPerCore_;
    uint32_t blockIdx_;
};

} // namespace ChunkGatedDeltaRuleRecurrence
#endif // __CHUNK_GATED_DELTA_RULE_RECURRENCE_KERNEL_H_
