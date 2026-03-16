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
 * \brief Fused Cube+Vector kernel for ChunkGatedDeltaRuleRecurrence (ascend950 / DAV_3510)
 *
 * Algorithm per chunk i for task (bIdx, hIdx):
 *   vPrime   = k_cumdecay[h,i] @ state[b,h].T   [cs,dk] @ [dk,dv] → [cs,dv]  (AIC C1)
 *   attn     = qgexp[h,i]      @ state[b,h].T   [cs,dk] @ [dk,dv] → [cs,dv]  (AIC C2)
 *   vNew     = value[h,i] - vPrime               element-wise [cs,dv]          (AIV V1)
 *   state   *= gexp[h,i,-1,0]                    scalar decay                  (AIV V0)
 *   state   += vNew.T @ kgexp[h,i]               [dv,cs] @ [cs,dk] → [dv,dk]  (AIC C3)
 *   state    = state_before_V0 * scalar + delta                                (AIV Vadd)
 *
 * Core type assignment (KERNEL_TYPE_MIX_AIC_1_2):
 *   AIC: Cube matmuls C1, C2, C3 using MatmulImpl
 *   AIV: Vector ops V1 (vNew = value - vPrime), V0 (state *= scalar),
 *        Vadd (state += delta), plus attn copy to output
 *
 * Sync protocol per (task, chunk):
 *   AIC: C1 → signal(C1_DONE) → C2 → signal(C2_DONE) → wait(V1_DONE)
 *        → C3 → signal(C3_DONE) → wait(VADD_DONE)
 *   AIV primary:   wait(C1_DONE) → V1 loop → signal(V1_DONE) → wait(C2_DONE)
 *                  → attn copy loop → wait(C3_DONE)
 *                  → V0+Vadd loop → PipeBarrier<MTE3> → signal(VADD_DONE)
 *   AIV secondary: wait(C1_DONE) → signal(V1_DONE) → wait(C2_DONE)
 *                  → wait(C3_DONE) → signal(VADD_DONE)
 *
 * Workspace layout per AIC group (wsBase = aicGroupIdx * wsPerGroup):
 *   [0 .. cs*dv)          vPrimeNewWs  (AIC writes vPrime; AIV overwrites with vNew for C3)
 *   [cs*dv .. 2*cs*dv)    attnWs       (AIC writes attn_inter)
 *   [2*cs*dv .. 2*cs*dv + dv*dk)  deltaWs  (AIC writes state delta)
 *
 * AIV TBuf: size = 2 × max(alignCs, alignDk) × dvTile × sizeof(float)
 *   Phase V1:    vPrimeUb_[alignCs×dvTile] + valueUb_[alignCs×dvTile]
 *   Phase V0+Vadd: stateUb_[dvTile×alignDk] + deltaUb_[dvTile×alignDk]
 *   Phase attn copy: attnUb_[alignCs×dvTile]  (fits within the V1 region)
 */

#ifndef __CHUNK_GATED_DELTA_RULE_RECURRENCE_KERNEL_H_
#define __CHUNK_GATED_DELTA_RULE_RECURRENCE_KERNEL_H_

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "chunk_gated_delta_rule_recurrence_tiling_data.h"

namespace ChunkGatedDeltaRuleRecurrence {

using namespace AscendC;
using namespace matmul;

constexpr uint64_t BUFFER_NUM     = 1;
constexpr uint32_t FP32_PER_BLOCK = 8;    // 8 × 4B = 32B

// Cross-core sync flag IDs  (AIC↔AIV, mode 0x2)
constexpr uint32_t C1_DONE   = 1U;  // AIC → AIV: vPrimeWs ready
constexpr uint32_t V1_DONE   = 2U;  // AIV → AIC: vNewWs ready
constexpr uint32_t C2_DONE   = 3U;  // AIC → AIV: attnWs ready
constexpr uint32_t C3_DONE   = 4U;  // AIC → AIV: deltaWs ready
constexpr uint32_t VADD_DONE = 5U;  // AIV → AIC: state written back

// ──────────────────────────────────────────────────────────────────────────
template <typename MT>
class CGDR {
public:
    __aicore__ inline CGDR(MT &mmC12, MT &mmC3,
                            const ChunkGatedDeltaRuleRecurrenceTilingData *td)
        : mmC12_(mmC12), mmC3_(mmC3)
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
        wsPerGroup_   = td->wsPerGroup;
        scaleValue_   = td->scaleValue;
    }

    __aicore__ inline void Init(GM_ADDR initialState, GM_ADDR kgexp, GM_ADDR value,
                                GM_ADDR kCumdecay, GM_ADDR qgexp, GM_ADDR gexp,
                                GM_ADDR cuSeqlens, GM_ADDR attnInterOut,
                                GM_ADDR vNewOut, GM_ADDR workspace, TPipe *pipe)
    {
        pipe_ = pipe;
        stateGm_.SetGlobalBuffer((__gm__ float *)initialState);
        kgexpGm_.SetGlobalBuffer((__gm__ float *)kgexp);
        valueGm_.SetGlobalBuffer((__gm__ float *)value);
        kCumdecayGm_.SetGlobalBuffer((__gm__ float *)kCumdecay);
        qgexpGm_.SetGlobalBuffer((__gm__ float *)qgexp);
        gexpGm_.SetGlobalBuffer((__gm__ float *)gexp);
        cuSeqlensGm_.SetGlobalBuffer((__gm__ int32_t *)cuSeqlens);
        attnInterOutGm_.SetGlobalBuffer((__gm__ float *)attnInterOut);
        vNewOutGm_.SetGlobalBuffer((__gm__ float *)vNewOut);
        workspaceGm_.SetGlobalBuffer((__gm__ float *)workspace);

        if ASCEND_IS_AIC {
            aicGroupIdx_ = GetBlockIdx();
            // User workspace layout: per-group data starts at float offset 0
            // wsBase_ = float offset for this AIC group's workspace region
            wsBase_ = static_cast<uint64_t>(aicGroupIdx_) * wsPerGroup_;
            // Initialise both matmul objects with per-task tiling
            mmC12_.Init(&cubeTilingC12_, pipe_);
            mmC3_.Init(&cubeTilingC3_,  pipe_);
        }
        if ASCEND_IS_AIV {
            // aicGroupIdx for AIV = blockIdx / taskRation
            aicGroupIdx_ = GetBlockIdx() / GetTaskRation();
            wsBase_       = static_cast<uint64_t>(aicGroupIdx_) * wsPerGroup_;
            isPrimaryAiv_ = (GetSubBlockIdx() == 0);
            InitAivBuffers();
        }
    }

    // Set TCubeTiling pointers (called from kernel.cpp after tiling is loaded)
    __aicore__ inline void SetCubeTilings(const TCubeTiling &c12, const TCubeTiling &c3)
    {
        cubeTilingC12_ = c12;
        cubeTilingC3_  = c3;
    }

    __aicore__ inline void Process()
    {
        uint32_t taskBegin = aicGroupIdx_ * tasksPerCore_;
        uint32_t taskEnd   = taskBegin + tasksPerCore_;
        if (taskEnd > totalTasks_) taskEnd = totalTasks_;
        for (uint32_t task = taskBegin; task < taskEnd; task++) {
            ProcessTask(task / hv_, task % hv_);
        }
    }

private:
    // ──────────────────────────────────────────────────────────────── AIV buffers
    __aicore__ inline void InitAivBuffers()
    {
        // Single TBuf shared across all phases (phases are sequential, no overlap)
        // Size: 2 × max(alignCs, alignDk) × dvTile_ floats
        uint32_t maxDim = (alignDk_ > alignCs_) ? alignDk_ : alignCs_;
        pipe_->InitBuffer(tmpBuf_, 2U * maxDim * dvTile_ * sizeof(float));

        uint32_t off = 0U;
        // Phase V1 partition: [vPrimeUb_, valueUb_/attnUb_]
        vPrimeUb_ = tmpBuf_.GetWithOffset<float>(alignCs_ * dvTile_, off);
        off      += alignCs_ * dvTile_;
        valueUb_  = tmpBuf_.GetWithOffset<float>(alignCs_ * dvTile_, off);
        // Phase V0+Vadd partition: [stateUb_, deltaUb_] (same base as above, reused)
        off = 0U;
        stateUb_  = tmpBuf_.GetWithOffset<float>(dvTile_ * alignDk_, off);
        off      += dvTile_ * alignDk_;
        deltaUb_  = tmpBuf_.GetWithOffset<float>(dvTile_ * alignDk_, off);
    }

    // ─────────────────────────────────────────────────────── AIC compute: per chunk
    __aicore__ inline void AicProcessChunk(uint32_t bIdx, uint32_t hIdx, int32_t ci)
    {
        // GM offsets for chunk matrices [hv, nChunks, cs, dk]
        uint64_t chunkBase = (static_cast<uint64_t>(hIdx) * nChunks_ + ci)
                             * static_cast<uint64_t>(realCs_) * realDk_;
        // GM offset for state [b, hv, dv, dk]
        uint64_t stateBase = ((static_cast<uint64_t>(bIdx) * hv_ + hIdx) * realDv_)
                             * realDk_;
        // Workspace offsets (relative to wsBase_)
        uint64_t vPrimeOff = wsBase_;                               // [cs, dv]
        uint64_t attnOff   = wsBase_ + realCs_ * realDv_;           // [cs, dv]
        uint64_t deltaOff  = wsBase_ + 2UL * realCs_ * realDv_;     // [dv, dk]

        // gexp scalar: last token of chunk gexp[h, ci, cs-1, 0]
        // (Not needed by AIC — used by AIV; skip)

        // ── C1: kCumdecay[cs,dk] × state[dv,dk]^T → vPrimeWs[cs,dv] ──
        mmC12_.SetOrgShape(realCs_, realDv_, realDk_);
        mmC12_.SetSingleShape(realCs_, realDv_, realDk_);
        mmC12_.SetTensorA(kCumdecayGm_[chunkBase]);
        mmC12_.SetTensorB(stateGm_[stateBase]);
        while (mmC12_.Iterate()) {
            mmC12_.GetTensorC(workspaceGm_[vPrimeOff], 0, true);
        }
        AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(C1_DONE);

        // ── C2: qgexp[cs,dk] × state[dv,dk]^T → attnWs[cs,dv] ──
        // (Runs in parallel with AIV V1 computing vNew from vPrimeWs)
        mmC12_.SetOrgShape(realCs_, realDv_, realDk_);
        mmC12_.SetSingleShape(realCs_, realDv_, realDk_);
        mmC12_.SetTensorA(qgexpGm_[chunkBase]);
        mmC12_.SetTensorB(stateGm_[stateBase]);
        while (mmC12_.Iterate()) {
            mmC12_.GetTensorC(workspaceGm_[attnOff], 0, true);
        }
        AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(C2_DONE);

        // ── Wait V1_DONE: vNewWs (= overwritten vPrimeWs) is ready ──
        AscendC::CrossCoreWaitFlag(V1_DONE);

        // ── C3: vNew[cs,dv]^T × kgexp[cs,dk] → deltaWs[dv,dk] ──
        mmC3_.SetOrgShape(realDv_, realDk_, realCs_);
        mmC3_.SetSingleShape(realDv_, realDk_, realCs_);
        mmC3_.SetTensorA(workspaceGm_[vPrimeOff]);   // vNew stored at vPrimeWs slot
        mmC3_.SetTensorB(kgexpGm_[chunkBase]);
        while (mmC3_.Iterate()) {
            mmC3_.GetTensorC(workspaceGm_[deltaOff], 0, true);
        }
        AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(C3_DONE);

        // ── Wait VADD_DONE: AIV has written updated state back to GM ──
        AscendC::CrossCoreWaitFlag(VADD_DONE);
    }

    // ──────────────────────────────────────────────────── AIV helpers: strided copy
    // Load workspace[cs, curDvTile] (row stride realDv_) → localUb [alignCs, dvTile_]
    __aicore__ inline void LoadWsSlice(LocalTensor<float> &dst,
                                        uint64_t wsBase, uint32_t dvOff,
                                        uint32_t curDvTile)
    {
        DataCopyExtParams cp{static_cast<uint16_t>(realCs_),
                             static_cast<uint32_t>(curDvTile * sizeof(float)),
                             static_cast<uint32_t>((realDv_ - curDvTile) * sizeof(float)),
                             0U, 0U};
        DataCopyPadExtParams<float> pp{true, 0U,
                                       static_cast<uint8_t>(dvTile_ - curDvTile), 0U};
        DataCopyPad(dst, workspaceGm_[wsBase + dvOff], cp, pp);
        AscendC::PipeBarrier<PIPE_MTE2>();
    }

    // Load state[curDvTile, realDk_] from GM → stateUb_
    __aicore__ inline void LoadStateSlice(uint64_t stateBase, uint32_t curDvTile)
    {
        DataCopyExtParams cp{static_cast<uint16_t>(curDvTile),
                             static_cast<uint32_t>(realDk_ * sizeof(float)),
                             0U, 0U, 0U};
        DataCopyPadExtParams<float> pp{true, 0U,
                                       static_cast<uint8_t>(alignDk_ - realDk_), 0U};
        DataCopyPad(stateUb_, stateGm_[stateBase], cp, pp);
        AscendC::PipeBarrier<PIPE_MTE2>();
    }

    // Load deltaWs[curDvTile, realDk_] (row stride realDk_) → deltaUb_
    __aicore__ inline void LoadDeltaSlice(uint64_t deltaWsBase, uint32_t dvOff,
                                           uint32_t curDvTile)
    {
        // deltaWs rows are contiguous (each row = realDk floats, no gap)
        DataCopyExtParams cp{static_cast<uint16_t>(curDvTile),
                             static_cast<uint32_t>(realDk_ * sizeof(float)),
                             0U, 0U, 0U};
        DataCopyPadExtParams<float> pp{true, 0U,
                                       static_cast<uint8_t>(alignDk_ - realDk_), 0U};
        DataCopyPad(deltaUb_, workspaceGm_[deltaWsBase + dvOff * realDk_], cp, pp);
        AscendC::PipeBarrier<PIPE_MTE2>();
    }

    // Write stateUb_ [curDvTile, realDk_] → GM, row by row
    __aicore__ inline void StoreStateSlice(uint64_t stateBase, uint32_t curDvTile)
    {
        DataCopyParams cp{1U, static_cast<uint16_t>(realDk_ * sizeof(float)), 0U, 0U};
        for (uint32_t d = 0; d < curDvTile; d++) {
            DataCopyPad(stateGm_[stateBase + static_cast<uint64_t>(d) * realDk_],
                        stateUb_[d * alignDk_], cp);
        }
    }

    // Write valueUb_ [realCs, curDvTile] (row stride dvTile_) → GM [realCs, realDv]
    __aicore__ inline void StoreSliceToGm(GlobalTensor<float> &dst,
                                           uint64_t gmBaseOff, uint32_t curDvTile)
    {
        DataCopyParams cp{1U, static_cast<uint16_t>(curDvTile * sizeof(float)), 0U, 0U};
        for (uint32_t s = 0; s < realCs_; s++) {
            DataCopyPad(dst[gmBaseOff + static_cast<uint64_t>(s) * realDv_],
                        valueUb_[s * dvTile_], cp);
        }
    }

    // Write valueUb_ [realCs, curDvTile] → workspace (same layout, row stride realDv_)
    __aicore__ inline void StoreSliceToWs(uint64_t wsBase, uint32_t dvOff,
                                           uint32_t curDvTile)
    {
        DataCopyParams cp{1U, static_cast<uint16_t>(curDvTile * sizeof(float)), 0U, 0U};
        for (uint32_t s = 0; s < realCs_; s++) {
            DataCopyPad(workspaceGm_[wsBase + dvOff + static_cast<uint64_t>(s) * realDv_],
                        valueUb_[s * dvTile_], cp);
        }
    }

    // Load value GM [realCs, curDvTile] (row stride realDv_) → valueUb_ [alignCs, dvTile_]
    __aicore__ inline void LoadValueSlice(uint64_t valBase, uint32_t dvOff,
                                           uint32_t curDvTile)
    {
        DataCopyExtParams cp{static_cast<uint16_t>(realCs_),
                             static_cast<uint32_t>(curDvTile * sizeof(float)),
                             static_cast<uint32_t>((realDv_ - curDvTile) * sizeof(float)),
                             0U, 0U};
        DataCopyPadExtParams<float> pp{true, 0U,
                                       static_cast<uint8_t>(dvTile_ - curDvTile), 0U};
        DataCopyPad(valueUb_, valueGm_[valBase + dvOff], cp, pp);
        AscendC::PipeBarrier<PIPE_MTE2>();
    }

    // ─────────────────────────────────────────────── AIV compute: primary per chunk
    __aicore__ inline void AivPrimaryProcessChunk(uint32_t bIdx, uint32_t hIdx, int32_t ci)
    {
        // GM offset for value/output [hv, nChunks, cs, dv]
        uint64_t valBase   = (static_cast<uint64_t>(hIdx) * nChunks_ + ci)
                             * static_cast<uint64_t>(realCs_) * realDv_;
        // GM offset for state [b, hv, dv, dk]
        uint64_t stateBase = ((static_cast<uint64_t>(bIdx) * hv_ + hIdx) * realDv_)
                             * realDk_;
        // Workspace offsets
        uint64_t vPrimeOff = wsBase_;
        uint64_t attnOff   = wsBase_ + realCs_ * realDv_;
        uint64_t deltaOff  = wsBase_ + 2UL * realCs_ * realDv_;

        // gexp scalar: gexp[h, ci, cs-1, 0]
        uint64_t gexpOff  = (static_cast<uint64_t>(hIdx) * nChunks_ + ci)
                            * realCs_ + (realCs_ - 1);
        float gexpScalar  = gexpGm_.GetValue(gexpOff);

        // ── Wait C1_DONE: vPrimeWs is ready ──
        AscendC::CrossCoreWaitFlag(C1_DONE);

        // ── V1 loop: vNew = value - vPrime, per dvTile ──
        for (uint32_t dvOff = 0; dvOff < realDv_; dvOff += dvTile_) {
            uint32_t curDvTile = (dvOff + dvTile_ <= realDv_) ? dvTile_ : (realDv_ - dvOff);

            // Load vPrime slice from workspace
            LoadWsSlice(vPrimeUb_, vPrimeOff, dvOff, curDvTile);
            // Load value slice from GM
            LoadValueSlice(valBase, dvOff, curDvTile);

            // vNew = value - vPrime  (result in valueUb_)
            Sub(valueUb_, valueUb_, vPrimeUb_, realCs_ * dvTile_);
            AscendC::PipeBarrier<PIPE_V>();

            // Write vNew → vNewOutGm_
            StoreSliceToGm(vNewOutGm_, valBase + dvOff, curDvTile);
            // Write vNew → vPrimeWs (overwrite, so AIC C3 can read it)
            StoreSliceToWs(vPrimeOff, dvOff, curDvTile);
        }
        AscendC::PipeBarrier<PIPE_MTE3>();

        // ── Signal V1_DONE (primary — with MTE3 to ensure vNew is visible) ──
        AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(V1_DONE);

        // ── Wait C2_DONE: attnWs is ready ──
        AscendC::CrossCoreWaitFlag(C2_DONE);

        // ── Attn copy loop: attnWs → attnInterOutGm_ ──
        for (uint32_t dvOff = 0; dvOff < realDv_; dvOff += dvTile_) {
            uint32_t curDvTile = (dvOff + dvTile_ <= realDv_) ? dvTile_ : (realDv_ - dvOff);
            // Reuse valueUb_ as attn buffer
            LoadWsSlice(valueUb_, attnOff, dvOff, curDvTile);
            StoreSliceToGm(attnInterOutGm_, valBase + dvOff, curDvTile);
        }

        // ── Wait C3_DONE: deltaWs is ready ──
        AscendC::CrossCoreWaitFlag(C3_DONE);

        // ── V0 + Vadd loop: state *= scalar; state += delta ──
        for (uint32_t dvOff = 0; dvOff < realDv_; dvOff += dvTile_) {
            uint32_t curDvTile = (dvOff + dvTile_ <= realDv_) ? dvTile_ : (realDv_ - dvOff);

            LoadStateSlice(stateBase + static_cast<uint64_t>(dvOff) * realDk_, curDvTile);
            LoadDeltaSlice(deltaOff, dvOff, curDvTile);

            // V0: state *= gexpScalar
            Muls(stateUb_, stateUb_, gexpScalar, curDvTile * alignDk_);
            AscendC::PipeBarrier<PIPE_V>();
            // Vadd: state += delta
            Add(stateUb_, stateUb_, deltaUb_, curDvTile * alignDk_);
            AscendC::PipeBarrier<PIPE_V>();

            StoreStateSlice(stateBase + static_cast<uint64_t>(dvOff) * realDk_, curDvTile);
        }
        AscendC::PipeBarrier<PIPE_MTE3>();

        // ── Signal VADD_DONE (primary — with MTE3 to ensure state is visible) ──
        AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(VADD_DONE);
    }

    // ─────────────────────────────────────────── AIV compute: secondary (pass-through)
    __aicore__ inline void AivSecondaryProcessChunk()
    {
        // Wait C1_DONE → signal V1_DONE immediately (no actual work)
        AscendC::CrossCoreWaitFlag(C1_DONE);
        AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(V1_DONE);

        // Wait C2_DONE (no action needed)
        AscendC::CrossCoreWaitFlag(C2_DONE);

        // Wait C3_DONE → signal VADD_DONE immediately
        AscendC::CrossCoreWaitFlag(C3_DONE);
        AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(VADD_DONE);
    }

    // ───────────────────────────────────────────────────────── per-task dispatch
    __aicore__ inline void ProcessTask(uint32_t bIdx, uint32_t hIdx)
    {
        int32_t chunkBegin = cuSeqlensGm_.GetValue(bIdx)     / static_cast<int32_t>(realCs_);
        int32_t chunkEnd   = cuSeqlensGm_.GetValue(bIdx + 1) / static_cast<int32_t>(realCs_);

        for (int32_t ci = chunkBegin; ci < chunkEnd; ci++) {
            if ASCEND_IS_AIC {
                AicProcessChunk(bIdx, hIdx, ci);
            }
            if ASCEND_IS_AIV {
                if (isPrimaryAiv_) {
                    AivPrimaryProcessChunk(bIdx, hIdx, ci);
                } else {
                    AivSecondaryProcessChunk();
                }
            }
        }
    }

    // ──────────────────────────────────────────────────────────────── data members
    // Matmul objects (references to kernel-scope variables)
    MT &mmC12_;
    MT &mmC3_;
    TCubeTiling cubeTilingC12_;
    TCubeTiling cubeTilingC3_;

    TPipe *pipe_{nullptr};

    GlobalTensor<float>   stateGm_;
    GlobalTensor<float>   kgexpGm_;
    GlobalTensor<float>   valueGm_;
    GlobalTensor<float>   kCumdecayGm_;
    GlobalTensor<float>   qgexpGm_;
    GlobalTensor<float>   gexpGm_;
    GlobalTensor<int32_t> cuSeqlensGm_;
    GlobalTensor<float>   attnInterOutGm_;
    GlobalTensor<float>   vNewOutGm_;
    GlobalTensor<float>   workspaceGm_;

    // AIV TBuf (not used by AIC)
    TBuf<TPosition::VECCALC> tmpBuf_;

    // AIV local tensor aliases (partition of tmpBuf_)
    LocalTensor<float> vPrimeUb_;   // [alignCs × dvTile_] — V1 phase: vPrime
    LocalTensor<float> valueUb_;    // [alignCs × dvTile_] — V1/attn phase
    LocalTensor<float> stateUb_;    // [dvTile_ × alignDk_] — V0+Vadd phase
    LocalTensor<float> deltaUb_;    // [dvTile_ × alignDk_] — V0+Vadd phase

    // Shape params
    uint32_t b_, hv_;
    uint32_t realDk_, alignDk_;
    uint32_t realDv_, alignDv_;
    uint32_t nChunks_;
    uint32_t realCs_, alignCs_;
    uint32_t dvTile_;
    uint32_t totalTasks_, tasksPerCore_;
    uint32_t wsPerGroup_;
    float    scaleValue_;

    // Core identity
    uint32_t aicGroupIdx_{0};
    uint64_t wsBase_{0};         // workspace base offset (in floats) for this group
    bool     isPrimaryAiv_{false};
};

} // namespace ChunkGatedDeltaRuleRecurrence
#endif // __CHUNK_GATED_DELTA_RULE_RECURRENCE_KERNEL_H_
