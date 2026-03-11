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
 * \file chunk_gated_delta_rule_stage1.h
 * \brief
 */
#ifndef __CHUNK_GATED_DELTA_RULE_STAGE1_H_
#define __CHUNK_GATED_DELTA_RULE_STAGE1_H_

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "chunk_gated_delta_rule_tiling_data.h"

namespace ChunkGatedDeltaRule {
using namespace AscendC;
using namespace matmul;

using aT_FP32 = MatmulType<TPosition::GM, CubeFormat::ND, float32_t>;
using bT_FP32 = MatmulType<TPosition::GM, CubeFormat::ND, float32_t>;
using cT_FP32 = MatmulType<TPosition::GM, CubeFormat::ND, float32_t>;
using MT_FP32 = matmul::MatmulImpl<aT_FP32, bT_FP32, cT_FP32>;

using aT_BF16 = MatmulType<TPosition::GM, CubeFormat::ND, float32_t>;
using bT_BF16 = MatmulType<TPosition::GM, CubeFormat::ND, float32_t>;
using cT_BF16 = MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t>;
using MT_BF16 = matmul::MatmulImpl<aT_BF16, bT_BF16, cT_BF16>;

constexpr uint64_t UB_REST_BYTES = 100 * 1024;  // 100KB
constexpr uint64_t INVERSE_SHAPE = 32;          // 对角块边长
constexpr uint64_t INVERSE_COUNT = 5;           // 求逆所需空间
constexpr uint64_t STAGEONE_BUFFER_NUM = 1;

struct GDRStageOneInitParams {
    // input
    GlobalTensor<bfloat16_t> query;     // (T, Nk, Dk) 
    GlobalTensor<bfloat16_t> key;       // (T, Nk, Dk) 
    GlobalTensor<bfloat16_t> value;     // (T, Nv, Dv)
    GlobalTensor<bfloat16_t> beta;      // (T, Nv)
    GlobalTensor<float> g;              // (T, Nv)
    // ouput
    GlobalTensor<float> gCumExp;        // (Nv, cg_len)
    GlobalTensor<bfloat16_t> kCumdecay; // (Nv, cg_len, Dk)
    GlobalTensor<float> vInner;         // (Nv, cg_len, Dv)
    GlobalTensor<bfloat16_t> qG;        // (Nv, cg_len, Dk)
    GlobalTensor<float> kG;             // (Nv, cg_len, Dk)
    GlobalTensor<float> qK;             // (Nv, cg_len, C)
    // other
    GM_ADDR ws;
    GlobalTensor<float> stageOneMask;   // (Nv, cg_len, C)
    ChunkGroup cg;
};

class GDRStageOne {
public:
    __aicore__ inline GDRStageOne(MT_FP32 &mmFp32, MT_BF16 &mmBf16) : mmFp32(mmFp32), mmBf16(mmBf16) {}
    __aicore__ inline void SetGlobalTensors(const GDRStageOneInitParams &initParams) {
        queryBaseGm_ = initParams.query;
        keyBaseGm_ = initParams.key;
        valueBaseGm_ = initParams.value;
        betaBaseGm_ = initParams.beta;
        gBaseGm_ = initParams.g;

        outGCumExpBaseGm_ = initParams.gCumExp;
        outKCumdecayBaseGm_ = initParams.kCumdecay;
        outVInnerBaseGm_ = initParams.vInner;
        outQGBaseGm_ = initParams.qG;
        outKgBaseGm_ = initParams.kG;
        outQkBaseGm_ = initParams.qK;
        stageOneMask_ = initParams.stageOneMask;

        uint64_t workSpaceOffset = 0;
        GBKWsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(initParams.ws + workSpaceOffset + coreIdx_ * chunkSize_ * dk_ * sizeof(float)));

        workSpaceOffset += coreNum_ * chunkSize_ * dk_ * sizeof(float);
        kkWsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(initParams.ws + workSpaceOffset + coreIdx_ * chunkSize_ * chunkSize_ * sizeof(float)));

        workSpaceOffset += coreNum_ * chunkSize_ * chunkSize_ * sizeof(float);
        vBetaWsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(initParams.ws + workSpaceOffset + coreIdx_ * chunkSize_ * dv_ * sizeof(float)));

        workSpaceOffset += coreNum_ * chunkSize_ * dv_ * sizeof(float);
        AttnWsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(initParams.ws + workSpaceOffset + coreIdx_ * chunkSize_ * chunkSize_ * sizeof(float)));

        workSpaceOffset += coreNum_ * chunkSize_ * chunkSize_ * sizeof(float);
        queryContinousGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(initParams.ws + workSpaceOffset + coreIdx_ * chunkSize_ * dk_ * sizeof(float)));

        workSpaceOffset += coreNum_ * chunkSize_ * dk_ * sizeof(float);
        keyContinousGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(initParams.ws + workSpaceOffset + coreIdx_ * chunkSize_ * dk_ * sizeof(float)));
    }

    __aicore__ inline void InitLocalBuffers()
    {
        if ASCEND_IS_AIC {
            return;
        }
        uint32_t maxLen = AscendC::Std::max(AscendC::Std::max(dv_ / 2, dk_ / 2), chunkSize_);
        pipe_->InitBuffer(fp32InQueue_, STAGEONE_BUFFER_NUM, chunkSize_ * maxLen * sizeof(float));  // 16KB  maxLen=64
        pipe_->InitBuffer(fp32OutQueue_, STAGEONE_BUFFER_NUM, chunkSize_ * maxLen * sizeof(float));  // 16KB
        pipe_->InitBuffer(gOutQueue_, STAGEONE_BUFFER_NUM, chunkSize_ * sizeof(float));  // 1KB

        pipe_->InitBuffer(tmpBuff, UB_REST_BYTES);
        uint32_t buffOffset = 0;
        betaUbBfloat16 = tmpBuff.GetWithOffset<bfloat16_t>(static_cast<uint32_t>(halfChunkSize_), buffOffset);  // 1KB
        buffOffset += halfChunkSize_ * sizeof(bfloat16_t);
        
        gCumUbFloat = tmpBuff.GetWithOffset<float>(static_cast<uint32_t>(chunkSize_), buffOffset);  // 1KB
        buffOffset += chunkSize_ * sizeof(float);

        gBUbFloat = tmpBuff.GetWithOffset<float>(static_cast<uint32_t>(halfChunkSize_), buffOffset);   // 1KB
        buffOffset += halfChunkSize_ * sizeof(float);

        gEndBroadUbFloat = tmpBuff.GetWithOffset<float>(static_cast<uint32_t>(halfChunkSize_), buffOffset);  // 1KB
        buffOffset += halfChunkSize_ * sizeof(float);

        betaUbFloat = tmpBuff.GetWithOffset<float>(static_cast<uint32_t>(halfChunkSize_), buffOffset);  // 1KB
        buffOffset += halfChunkSize_ * sizeof(float);

        gBroadUbFloat = tmpBuff.GetWithOffset<float>(static_cast<uint32_t>(chunkSize_ * maxLen), buffOffset);    // 16KB
        gammaUbFloat = gBroadUbFloat;
        kUbFloat = gBroadUbFloat;
        valueUbFloat = gBroadUbFloat;
        qUbFloat = gBroadUbFloat;
        buffOffset += chunkSize_ * maxLen  * sizeof(float);
        
        gTransBroadUbFloat = tmpBuff.GetWithOffset<float>(static_cast<uint32_t>(chunkSize_ * maxLen), buffOffset);      // 16KB
        attnUbFloat = gTransBroadUbFloat;
        gCumExpBroadUbFloat = gTransBroadUbFloat;
        qPrimeUbFloat = gTransBroadUbFloat;
        buffOffset += chunkSize_ * maxLen * sizeof(float);

        qUbFloatCon = tmpBuff.GetWithOffset<float>(static_cast<uint32_t>(halfChunkSize_ * dk_), buffOffset);      // 16KB
        buffOffset += halfChunkSize_ * dk_ * sizeof(float);

        kUbFloatCon = tmpBuff.GetWithOffset<float>(static_cast<uint32_t>(halfChunkSize_ * dk_), buffOffset);      // 16KB
        buffOffset += halfChunkSize_ * dk_ * sizeof(float);

        inverseUbFloat = tmpBuff.GetWithOffset<float>(static_cast<uint32_t>(halfChunkSize_ * halfChunkSize_ * INVERSE_COUNT), buffOffset);      // 20KB
    }

    __aicore__ inline void Init(const GDRStageOneInitParams &initParams, TPipe *pipe, const ChunkGatedDeltaRuleTilingData *tilingData)
    {
        pipe_ = pipe;
        tiling_ = tilingData;
        nk_ = tiling_->nk;
        nv_ = tiling_->nv;
        chunkSize_ = tiling_->chunkSize;
        dk_ = tiling_->dk;
        dv_ = tiling_->dv;
        scale_ = tiling_->scale;
        coreNum_ = tiling_->aiCoreNum;
        cg_ = initParams.cg;
        validLen_ = chunkSize_;
        v_row_stride_ = nv_ * dv_;
        NumChunk_ = (cg_.length + chunkSize_ - 1) / chunkSize_;
        subBlockIdx_ = GetSubBlockIdx();
        halfChunkSize_ = chunkSize_ / 2;
        subOffset_ = subBlockIdx_ * halfChunkSize_;
        coreIdx_ = GetBlockIdx();
        if ASCEND_IS_AIV{
            coreIdx_ /= GetTaskRatio();
        }
        SetGlobalTensors(initParams);
        InitLocalBuffers();
    }

    __aicore__ inline void Process() 
    {
        uint32_t totalChunk = nv_ * NumChunk_;
        uint32_t tailChunkNum = totalChunk / coreNum_;   // tail核处理的块数
        uint32_t formerChunkNum = tailChunkNum + 1;     // former核处理的块数
        uint32_t formerCoreNum = totalChunk % coreNum_;  // former核数量
        uint32_t start, end;
        if(coreIdx_ < formerCoreNum){
            start = coreIdx_ * formerChunkNum;
            end = start + formerChunkNum;
        } else {
            start = formerCoreNum * formerChunkNum + (coreIdx_ - formerCoreNum) * tailChunkNum;
            end = start + tailChunkNum;
        }

        for (int32_t task_id = start; task_id < end; ++task_id) {
            uint64_t nid   = task_id % nv_;
            uint64_t cg_id = task_id / nv_;
            // 尾chunk处理
            uint64_t valid_len = chunkSize_;
            if (cg_id == NumChunk_ - 1 && cg_.length % chunkSize_ != 0) {
                validLen_ = cg_.length % chunkSize_;
            }
            // chunk在全局T上的起始行 = chunkGroup起始行 + chunk内偏移
            uint64_t chunk_start_row = cg_.startPos + cg_id * chunkSize_;
            SetChunkTensors(nid, cg_id, chunk_start_row);
            ProcessOneChunk();
        }
    }

private:
    // ----------------------------------------------------------
    // SetChunkTensors
    //   nid       : head 编号 (Nv 维度)
    //   local_cid : CG 内的 chunk 编号 (0 ~ CG_CHUNKS-1)
    //   chunk_start_row   : 当前 chunk 在全局 T 上的起始行
    // ----------------------------------------------------------
   __aicore__ inline void SetChunkTensors(uint64_t nid, uint64_t local_cid, uint64_t chunk_start_row)
    {
        uint64_t kid = nid * nk_ / nv_;
        uint64_t sub_row = chunk_start_row + subOffset_;
        uint64_t qk_base = sub_row * nk_ * dk_ + kid * dk_;
        queryGm_ = queryBaseGm_[qk_base];
        keyGm_   = keyBaseGm_[qk_base];

        uint64_t vOffset = chunk_start_row * nv_ * dv_ + nid * dv_;
        valueGm_ = valueBaseGm_[vOffset];

        uint64_t bgOffset = chunk_start_row * nv_ + nid;
        betaGm_ = betaBaseGm_[bgOffset];
        gGm_ = gBaseGm_[bgOffset];

        uint64_t cgLen_pad = (cg_.length + chunkSize_ - 1) / chunkSize_ * chunkSize_;
        uint64_t cb = nid * cgLen_pad + local_cid * chunkSize_;

        outGCumExpGm_ = outGCumExpBaseGm_[cb];
        outKCumdecayGm_ = outKCumdecayBaseGm_[cb * dk_];
        outQgGm_ = outQGBaseGm_[cb * dk_];
        outKgGm_ = outKgBaseGm_[cb * dk_];
        outVInnerGm_ = outVInnerBaseGm_[cb * dv_];
        outQkGm_ = outQkBaseGm_[cb * chunkSize_];
    }

    __aicore__ inline void ProcessOneChunk()
    {
        if ASCEND_IS_AIC {
            AscendC::CrossCoreWaitFlag(0x9);  //同步0
            // key @ key.transpose(-1,-2)
            AICProcess(keyContinousGm_, keyContinousGm_, kkWsGm_, 
                       chunkSize_, chunkSize_, dk_, chunkSize_, chunkSize_, dk_, true);
            AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(0x8);  //同步1
            // query @ key.transpose(-1,-2)   stage1 out
            AICProcess(queryContinousGm_, keyContinousGm_, outQkGm_, chunkSize_, chunkSize_, dk_, 
                       chunkSize_, chunkSize_, dk_, true);
            AscendC::CrossCoreWaitFlag(0x7);  //同步2
            // 求逆左下角矩阵
            AttnInverseMMCompute(INVERSE_SHAPE);
            AscendC::CrossCoreWaitFlag(0x6);  //同步3
            // attn @ k_cumdecay
            kCumDecayCompute();
            AscendC::CrossCoreWaitFlag(0x5);  //同步4
            // attn @ v_beta    stage1 out
            AICProcess(AttnWsGm_, vBetaWsGm_, outVInnerGm_, chunkSize_, dv_, chunkSize_, chunkSize_, dv_, chunkSize_);
        }
        if ASCEND_IS_AIV {
            // 获取连续QK
            QKPreProcess();
            AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(0x9);  //同步0
            // g_cum_exp = g.cumsum(dim=-1).exp()
            GCumExpCompute();
            // attn_1 = (g_cum_exp[:None] / g_cum_exp[None,:]) * mask
            GammaCompute();
            BetaCopyInWithStride();
            AscendC::CrossCoreWaitFlag(0x8);  //同步1
            // attn_1 = kkt * attn_1
            KKBetaCompute();
            // attn_1对角块求逆，对角块shape为INVERSE_SHAPE=32
            InverseCompute();
            AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(0x7);  //同步2
            // kg = key * (g_cum_exp[-1, None] / g_cum_exp)[..., None] && k_cumdecay = -1.0 * k * beta * g_cum_exp
            GBKCompute();
            AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(0x6);  //同步3
            // v_beta = value * beta.unsqueeze(-1)  # (C, Dv)
            vBetaCompute();
            AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(0x5);  //同步4
            // q_prime = query * scale_ * g_cum_exp[:, None]       # (C, Dk)
            QPrimeCompute();
        }
    }


    __aicore__ inline void QKPreProcessCompute(const GlobalTensor<bfloat16_t>& srcGm, const GlobalTensor<float>& dstGm,
                                                LocalTensor<float>& dstBuffer)
    {
        uint32_t rows = chunkSize_ / 2;
        uint64_t validRow = halfChunkSize_;
        if (validLen_ < halfChunkSize_) {
            validRow = (subBlockIdx_ == 0) ? validLen_ : 0;
        } else {
            validRow = (subBlockIdx_ == 0) ? halfChunkSize_ : validLen_ - halfChunkSize_;
        }
        LocalTensor<float> tmpTensor;
        // copyIn
        DataCopyInBf16WithStride(validRow, dk_, srcGm, nk_ *dk_);
        // compute
        LocalTensor<bfloat16_t> bf16Tensor = fp32InQueue_.DeQue<bfloat16_t>();
        Cast(dstBuffer, bf16Tensor, AscendC::RoundMode::CAST_NONE, validRow * dk_);
        PipeBarrier<PIPE_V>();
        fp32InQueue_.FreeTensor(bf16Tensor);

        if (validRow < rows) {
            Duplicate(dstBuffer[validRow * dk_], static_cast<float>(0.0f), (halfChunkSize_ - validRow) * dk_);
            PipeBarrier<PIPE_V>();
        }

        // copyOut
        tmpTensor = fp32OutQueue_.AllocTensor<float>();
        DataCopy(tmpTensor, dstBuffer, rows * dk_);
        fp32OutQueue_.EnQue(tmpTensor);
        tmpTensor = fp32OutQueue_.DeQue<float>();

        DataCopyExtParams outParams{static_cast<uint16_t>(rows),
                                    static_cast<uint32_t>(dk_ * sizeof(float)), 0, 0, 0};
        DataCopyPad(dstGm, tmpTensor, outParams);
        fp32OutQueue_.FreeTensor(tmpTensor);
    }

    __aicore__ inline void QKPreProcess(){
        if ASCEND_IS_AIC {
            return;
        }
        uint64_t outOffset = subOffset_ * dk_;
        QKPreProcessCompute(queryGm_, queryContinousGm_[outOffset], qUbFloatCon);
        QKPreProcessCompute(keyGm_, keyContinousGm_[outOffset], kUbFloatCon);
    }

    __aicore__ inline void GCumExpCompute()
    {
        if ASCEND_IS_AIC {
            return;
        }
        // Copy g
        GCopyInWithStride();
        // CumSum计算
        uint32_t outer = 1;
        uint32_t inner = chunkSize_;
        CumSumInfo cumSumInfo{outer, inner};
        CumSum<float>(gCumUbFloat, gCumUbFloat, gCumUbFloat, cumSumInfo);
        PipeBarrier<PIPE_V>();
        // Exp计算
        gCumExpUbFloat = gOutQueue_.AllocTensor<float>();
        Exp<float, 0, true>(gCumExpUbFloat, gCumUbFloat, chunkSize_);
        gOutQueue_.EnQue<float>(gCumExpUbFloat);
        DataCopyOutG(chunkSize_);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void GammaCompute()
    {
        if ASCEND_IS_AIC {
            return;
        }
        // BroadCast
        uint32_t divShape[2] = {chunkSize_, chunkSize_};
        uint32_t gShape[2] = {chunkSize_, 1};
        uint32_t gTransShape[2] = {1, chunkSize_};
        Broadcast<float, 2, 1>(gBroadUbFloat, gCumExpUbFloat, divShape, gShape);
        Broadcast<float, 2, 0>(gTransBroadUbFloat, gCumExpUbFloat, divShape, gTransShape);
        PipeBarrier<PIPE_V>();
        // div
        Div(gammaUbFloat, gBroadUbFloat, gTransBroadUbFloat, chunkSize_ * chunkSize_);
        PipeBarrier<PIPE_V>();
        // mask
        DataCopyInFp32(chunkSize_ * chunkSize_, stageOneMask_);
        kkLocal = fp32InQueue_.DeQue<float>();
        Mul(gammaUbFloat, gammaUbFloat, kkLocal, chunkSize_ * chunkSize_);
        fp32InQueue_.FreeTensor(kkLocal);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void KKBetaCompute()
    {
        // copy value
        uint32_t kkLength = chunkSize_ * halfChunkSize_;
        uint64_t kkBeginOffset = subOffset_ * chunkSize_;
        DataCopyInFp32(kkLength, kkWsGm_[kkBeginOffset]);
        kkLocal = fp32InQueue_.DeQue<float>();

        uint32_t betaShape[2] = {halfChunkSize_, 1};
        uint32_t kkShape[2] = {halfChunkSize_, chunkSize_};
        Broadcast<float, 2, 1>(attnUbFloat, betaUbFloat, kkShape, betaShape);
        PipeBarrier<PIPE_V>();
        Mul(attnUbFloat, kkLocal, attnUbFloat, chunkSize_ * halfChunkSize_);
        PipeBarrier<PIPE_V>();
        fp32InQueue_.FreeTensor(kkLocal);
    }

    __aicore__ inline void InverseCompute()
    {
        uint64_t curVecLen = chunkSize_ * halfChunkSize_;
        Mul(attnUbFloat, attnUbFloat, gammaUbFloat[subOffset_ * chunkSize_], curVecLen);
        PipeBarrier<PIPE_V>();

        inverseLocal = fp32OutQueue_.AllocTensor<float>();
        Muls(inverseLocal, attnUbFloat, static_cast<float>(-1.0), curVecLen);
        PipeBarrier<PIPE_V>();

        InverseAIV(subOffset_, INVERSE_SHAPE);
        fp32OutQueue_.EnQue(inverseLocal);
        DataCopyOutFp32(curVecLen, AttnWsGm_[subBlockIdx_ * curVecLen]);
    }

    __aicore__ inline void InverseAIV(uint64_t offset, uint32_t inverseVecLen)
    {
        PipeBarrier<PIPE_V>();
        uint64_t inverseBufferOffset = 0;
        auto row = inverseUbFloat[inverseBufferOffset];
        inverseBufferOffset += inverseVecLen * inverseVecLen;
        auto col = inverseUbFloat[inverseBufferOffset];
        inverseBufferOffset += inverseVecLen * inverseVecLen + inverseVecLen;
        auto yLocal = inverseUbFloat[inverseBufferOffset];
        inverseBufferOffset += inverseVecLen * inverseVecLen;
        auto ei = inverseUbFloat[inverseBufferOffset];
        inverseBufferOffset += inverseVecLen * inverseVecLen;
        auto colBufferGather = colBuffer[inverseBufferOffset];

        Duplicate(ei, static_cast<float>(0.0), inverseVecLen);
        Duplicate(yLocal, static_cast<float>(0.0), 2 * inverseVecLen * inverseVecLen); // yLocal清零
        inverseLocal.SetValue(offset, static_cast<float>(1.0));
        
        uint32_t srcShape[2] = {1, inverseVecLen};
        uint32_t offsetIdx = 0;
        for (uint32_t j = 0; j < inverseVecLen; ++j) {
            colBufferGather.SetValue<uint32_t>(offsetIdx++, (j * chunkSize_) * sizeof(float));
        }
        for (int i = 1; i < inverseVecLen; ++i) {
            uint32_t curI = i - 1;
            uint32_t validRows = inverseVecLen - i;
            Gather(col, attnUbFloat[offset + i * chunkSize_ + curI], colBufferGather, (uint32_t)0, validRows);

            uint32_t dstShape[2] = {validRows, inverseVecLen};
            uint32_t colSrcShape[2] = {validRows, 1};
            Broadcast<float, 2, 1>(col[inverseVecLen], col, dstShape, colSrcShape);
            Broadcast<float, 2, 0>(row, inverseLocal[offset + curI * chunkSize_], dstShape, srcShape);
            MulAddDst(yLocal[i * inverseVecLen], col[inverseVecLen], row, inverseVecLen * validRows);
            PipeBarrier<PIPE_V>();
            ei.SetValue(i - 1, static_cast<float>(0.0));
            ei.SetValue(i, static_cast<float>(1.0));
            // xi = (I - SUM) / Lii = I - SUM
            Sub(inverseLocal[offset + i * chunkSize_], ei, yLocal[i * inverseVecLen], inverseVecLen);
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void GBKCompute()
    {
        if ASCEND_IS_AIC {
            return;
        }
        // tmp = -1.0 * beta * g_cum_exp
        Mul(gBUbFloat, betaUbFloat, gCumExpUbFloat[subOffset_], halfChunkSize_);
        PipeBarrier<PIPE_V>();
        Muls(gBUbFloat, gBUbFloat, static_cast<float>(-1), halfChunkSize_);
        PipeBarrier<PIPE_V>();
        // k_cumdecay = k * tmp =  -1.0 * k * beta * g_cum_exp
        uint32_t betaShape[2] = {halfChunkSize_, 1};
        uint32_t kShape[2] = {halfChunkSize_, dk_};
        gBKLocal = fp32OutQueue_.AllocTensor<float>();
        Broadcast<float, 2, 1>(gBKLocal, gBUbFloat, kShape, betaShape);
        PipeBarrier<PIPE_V>();
        Mul(gBKLocal, gBKLocal, kUbFloatCon, halfChunkSize_ * dk_);
        fp32OutQueue_.EnQue<float>(gBKLocal);
        uint64_t GBKBeginOffset = subOffset_ * dk_;
        DataCopyOutFp32(halfChunkSize_ * dk_, GBKWsGm_[GBKBeginOffset]);
        PipeBarrier<PIPE_V>();
        // kg = k * (g_cum_exp[-1, None] / g_cum_exp)[..., None]
        uint32_t gEndShape[2] = {1, 1};
        uint32_t gBroadShape[2] = {halfChunkSize_, 1};
        Broadcast<float, 2, 0>(gEndBroadUbFloat, gCumExpUbFloat[chunkSize_ - 1], gBroadShape, gEndShape);
        PipeBarrier<PIPE_V>();
        Div(gEndBroadUbFloat, gEndBroadUbFloat, gCumExpUbFloat[subOffset_], halfChunkSize_);
        PipeBarrier<PIPE_V>();
        kgLocal = fp32OutQueue_.AllocTensor<float>();
        Broadcast<float, 2, 1>(kgLocal, gEndBroadUbFloat, kShape, gBroadShape);
        PipeBarrier<PIPE_V>();
        Mul(kgLocal, kgLocal, kUbFloatCon, halfChunkSize_ * dk_);
        PipeBarrier<PIPE_V>();
        fp32OutQueue_.EnQue<float>(kgLocal);
        uint64_t kgBeginOffset = subOffset_ * dk_;
        DataCopyOutFp32(halfChunkSize_ * dk_, outKgGm_[kgBeginOffset]);  // stage1 out
    }

    __aicore__ inline void vBetaCompute()
    {
        uint32_t valueLength = halfChunkSize_ * dv_;
        uint64_t vBeginOffset = subOffset_ * v_row_stride_;
        uint64_t validRow = halfChunkSize_;
        if (validLen_ < halfChunkSize_) {
            validRow = (subBlockIdx_ == 0) ? validLen_ : 0;
        } else {
            validRow = (subBlockIdx_ == 0) ? halfChunkSize_ : validLen_ - halfChunkSize_;
        }
        DataCopyInBf16WithStride(validRow, dv_, valueGm_[vBeginOffset], v_row_stride_);
        valueLocal = fp32InQueue_.DeQue<bfloat16_t>();
        vBetaLocal = fp32OutQueue_.AllocTensor<float>();
        Cast(valueUbFloat, valueLocal, AscendC::RoundMode::CAST_NONE, validRow * dv_);
        PipeBarrier<PIPE_V>();
        fp32InQueue_.FreeTensor(valueLocal);
        if (validLen_ < halfChunkSize_) {
            Duplicate(valueUbFloat[validLen_ * dv_], static_cast<float>(0.0f), (halfChunkSize_ - validLen_) * dv_);
            PipeBarrier<PIPE_V>();
        }
        
        uint32_t betaShape[2] = {halfChunkSize_, 1};
        uint32_t vShape[2] = {halfChunkSize_, dv_};
        Broadcast<float, 2, 1>(vBetaLocal, betaUbFloat, vShape, betaShape);
        PipeBarrier<PIPE_V>();
        Mul(vBetaLocal, valueUbFloat, vBetaLocal, chunkSize_ * dv_ / 2);
        PipeBarrier<PIPE_V>();
        fp32OutQueue_.EnQue<float>(vBetaLocal);
        DataCopyOutFp32(valueLength, vBetaWsGm_[subOffset_ * dv_]);
    }

    __aicore__ inline void QPrimeCompute()
    {
        if ASCEND_IS_AIC {
            return;
        }
        // query * scale
        Muls(qUbFloat, qUbFloatCon, scale_, chunkSize_ * dk_ / 2);
        PipeBarrier<PIPE_V>();
        uint32_t gCumExpShape[2] = {halfChunkSize_, 1};
        uint32_t qShape[2] = {halfChunkSize_, dk_};
        Broadcast<float, 2, 1>(gCumExpBroadUbFloat, gCumExpUbFloat[subOffset_], qShape, gCumExpShape);
        PipeBarrier<PIPE_V>();
         // query * scale * g_cum_exp[:, None]       # (C, Dk)
        Mul(qPrimeUbFloat, qUbFloat, gCumExpBroadUbFloat, chunkSize_ * dk_ / 2);
        PipeBarrier<PIPE_V>();
        qPrimeLocal = fp32OutQueue_.AllocTensor<bfloat16_t>();
        Cast(qPrimeLocal, qPrimeUbFloat, AscendC::RoundMode::CAST_RINT, chunkSize_ * dk_ / 2);
        fp32OutQueue_.EnQue<bfloat16_t>(qPrimeLocal);
        uint64_t qgBeginOffset = subOffset_ * dk_;
        DataCopyOutBf16(chunkSize_ * dk_ / 2, outQgGm_[qgBeginOffset]);  // stage1 out
        PipeBarrier<PIPE_V>();
        gOutQueue_.FreeTensor(gCumExpUbFloat);
    }

    __aicore__ inline void DataCopyInFp32(uint64_t len, GlobalTensor<float> y)
    {
        DataCopyPadExtParams<float> padParams;
        DataCopyExtParams kkParams{static_cast<uint16_t>(1), static_cast<uint32_t>(len * sizeof(float)), 0, 0, 0};
        fp32InLocal = fp32InQueue_.AllocTensor<float>();
        DataCopyPad(fp32InLocal, y, kkParams, padParams);
        fp32InQueue_.EnQue<float>(fp32InLocal);
    }

    __aicore__ inline void BetaCopyInWithStride()
    {
        uint64_t betaBeginOffset = subOffset_ * nv_;
        uint64_t validRow = halfChunkSize_;
        if (validLen_ < halfChunkSize_) {
            validRow = (subBlockIdx_ == 0) ? validLen_ : 0;
        } else {
            validRow = (subBlockIdx_ == 0) ? halfChunkSize_ : validLen_ - halfChunkSize_;
        }
        DataCopyInBf16WithStride(validRow, 1, betaGm_[betaBeginOffset], nv_);
        betaLocal = fp32InQueue_.DeQue<bfloat16_t>();
        constexpr uint32_t slot = 32 / sizeof(bfloat16_t);
        for (uint32_t i = 0; i < validRow; ++i) {
            betaUbBfloat16.SetValue(i, betaLocal.GetValue(i * slot));
        }
        for (uint32_t i = validRow; i < halfChunkSize_; ++i) {
            betaUbBfloat16.SetValue(i, bfloat16_t(0.0f));
        }
        Cast(betaUbFloat, betaUbBfloat16, AscendC::RoundMode::CAST_NONE, halfChunkSize_);
        PipeBarrier<PIPE_V>();
        fp32InQueue_.FreeTensor(betaLocal);
    }

    __aicore__ inline void GCopyInWithStride()
    {
        constexpr uint32_t slot = 32 / sizeof(float);
        uint64_t validRow = validLen_;
        DataCopyInFp32WithStride(validRow, 1, gGm_, nv_);
        gLocal = fp32InQueue_.DeQue<float>();
        for (uint32_t i = 0; i < validRow; ++i) {
            gCumUbFloat.SetValue(i, gLocal.GetValue(i * slot));
        }
        for (uint32_t i = validRow; i < chunkSize_; ++i) {
            gCumUbFloat.SetValue(i, float(0.0f));
        }
        PipeBarrier<PIPE_V>();
        fp32InQueue_.FreeTensor(gLocal);
    }

    __aicore__ inline void DataCopyInFp32WithStride(uint64_t rows,  // 要搬的行数
                                                    uint64_t cols,  // 每行的元素数
                                                    const GlobalTensor<float> src,
                                                    uint64_t srcRowStride) // GM上相邻行的间距(元素数)
    {
        DataCopyPadExtParams<float> padParams = {false, static_cast<uint8_t>(0), static_cast<uint8_t>(0),
                                                      static_cast<float>(0)};
        uint32_t srcGap = (srcRowStride - cols) * sizeof(float);
        DataCopyExtParams params{static_cast<uint16_t>(rows),
                                 static_cast<uint32_t>(cols * sizeof(float)),
                                 static_cast<uint32_t>(srcGap), 0, 0};
        fp32InLocal = fp32InQueue_.AllocTensor<float>();
        DataCopyPad(fp32InLocal, src, params, padParams);
        fp32InQueue_.EnQue<float>(fp32InLocal);
    }

    __aicore__ inline void DataCopyInBf16WithStride(uint64_t rows,  // 要搬的行数
                                                    uint64_t cols,  // 每行的元素数
                                                    GlobalTensor<bfloat16_t> src,
                                                    uint64_t srcRowStride, // GM上相邻行的间距(元素数)
                                                    uint64_t pad_rows = 0)
    {
        DataCopyPadExtParams<bfloat16_t> padParams = {false, static_cast<uint8_t>(0), static_cast<uint8_t>(0),
                                                      static_cast<float>(0)};
        uint32_t srcGap = (srcRowStride - cols) * sizeof(bfloat16_t);
        DataCopyExtParams params{static_cast<uint16_t>(rows),
                                 static_cast<uint32_t>(cols * sizeof(bfloat16_t)),
                                 static_cast<uint32_t>(srcGap), 0, 0};
        bf16InLocal = fp32InQueue_.AllocTensor<bfloat16_t>();
        DataCopyPad(bf16InLocal, src, params, padParams);
        fp32InQueue_.EnQue<bfloat16_t>(bf16InLocal);
    }

    __aicore__ inline void DataCopyOutFp32(uint32_t len, GlobalTensor<float> y)
    {
        fp32OutLocal = fp32OutQueue_.DeQue<float>();
        DataCopyExtParams yGMParams{static_cast<uint16_t>(1), static_cast<uint16_t>(len * sizeof(float)), 0, 0, 0};
        DataCopyPad(y, fp32OutLocal, yGMParams);
        fp32OutQueue_.FreeTensor(fp32OutLocal);
    }

    __aicore__ inline void DataCopyOutBf16(uint32_t len, GlobalTensor<bfloat16_t> y)
    {
        bf16OutLocal = fp32OutQueue_.DeQue<bfloat16_t>();
        DataCopyExtParams yGMParams{static_cast<uint16_t>(1), static_cast<uint16_t>(len * sizeof(bfloat16_t)), 0, 0, 0};
        DataCopyPad(y, bf16OutLocal, yGMParams);
        fp32OutQueue_.FreeTensor(bf16OutLocal);
    }

    __aicore__ inline void DataCopyOutG(uint64_t length)
    {
        gCumExpUbFloat = gOutQueue_.DeQue<float>();
        if (subBlockIdx_ == 0){
            DataCopyExtParams params{static_cast<uint16_t>(1),
                                    static_cast<uint16_t>(length * sizeof(float)),
                                    0, 0, 0};
            DataCopyPad(outGCumExpGm_, gCumExpUbFloat, params);  // stage1 out
        }
    }

    __aicore__ inline void AttnInverseMMCompute(uint64_t curLen)
    {
        uint64_t leftDown = chunkSize_ * curLen;
        uint64_t rightDown = leftDown + curLen;
        // 右矩阵左下角 @ 右矩阵左上角 -> 右矩阵左下角
        AICProcess(AttnWsGm_[leftDown], AttnWsGm_, AttnWsGm_[leftDown], chunkSize_, chunkSize_, chunkSize_, curLen, curLen, curLen);
        SetFlag<HardEvent::FIX_MTE2>(EVENT_ID1);
        WaitFlag<HardEvent::FIX_MTE2>(EVENT_ID1);
        // 右矩阵右下角 @ 右矩阵左下角 -> 右矩阵左下角
        AICProcess(AttnWsGm_[rightDown], AttnWsGm_[leftDown], AttnWsGm_[leftDown], chunkSize_, chunkSize_, chunkSize_, curLen, curLen, curLen);
        SetFlag<HardEvent::FIX_MTE2>(EVENT_ID1);
        WaitFlag<HardEvent::FIX_MTE2>(EVENT_ID1);
    }

    __aicore__ inline void AICProcess(GlobalTensor<float> x, GlobalTensor<float> y, GlobalTensor<float> z, 
                                      uint64_t m, uint64_t n, uint64_t k, uint64_t sm, uint64_t sn, uint64_t sk, bool transB=false)
    {
        mmFp32.SetOrgShape(m, n, k);
        mmFp32.SetSingleShape(sm, sn, sk);
        mmFp32.SetTensorA(x);
        mmFp32.SetTensorB(y, transB);
        mmFp32.IterateAll(z);
        mmFp32.End();
    }

    __aicore__ inline void kCumDecayCompute()
    {
        mmBf16.SetOrgShape(chunkSize_, dk_, chunkSize_);
        mmBf16.SetSingleShape(chunkSize_, dk_, chunkSize_);
        mmBf16.SetTensorA(AttnWsGm_);
        mmBf16.SetTensorB(GBKWsGm_);
        mmBf16.IterateAll(outKCumdecayGm_);  // stage1 out
        mmBf16.End();
    }
    
    TPipe *pipe_;
    MT_FP32 &mmFp32;
    MT_BF16 &mmBf16;
    const ChunkGatedDeltaRuleTilingData *tiling_;
    ChunkGroup cg_;
    uint32_t nk_;
    uint32_t nv_;
    uint32_t dk_;
    uint32_t dv_;
    uint32_t NumChunk_;
    uint64_t v_row_stride_;
    uint32_t halfChunkSize_;
    uint32_t subBlockIdx_;
    uint32_t subOffset_;
    uint32_t coreIdx_;
    uint32_t chunkSize_;
    uint32_t validLen_;
    uint32_t coreNum_;
    float scale_;

    // base GM pointers
    GlobalTensor<bfloat16_t> queryBaseGm_;
    GlobalTensor<bfloat16_t> keyBaseGm_;
    GlobalTensor<bfloat16_t> valueBaseGm_;
    GlobalTensor<bfloat16_t> betaBaseGm_;
    GlobalTensor<float> gBaseGm_;
    GlobalTensor<float> outGCumExpBaseGm_, outVInnerBaseGm_, outKgBaseGm_, outQkBaseGm_;
    GlobalTensor<bfloat16_t> outKCumdecayBaseGm_, outQGBaseGm_;

    // per-chunk GM pointers 
    GlobalTensor<bfloat16_t> queryGm_;
    GlobalTensor<bfloat16_t> keyGm_;
    GlobalTensor<bfloat16_t> valueGm_;
    GlobalTensor<bfloat16_t> betaGm_;
    GlobalTensor<float> gGm_;
    GlobalTensor<float> outGCumExpGm_;
    GlobalTensor<bfloat16_t> outKCumdecayGm_;
    GlobalTensor<float> outVInnerGm_;
    GlobalTensor<bfloat16_t> outQgGm_;
    GlobalTensor<float> outKgGm_;
    GlobalTensor<float> outQkGm_;

    GlobalTensor<float> vBetaWsGm_;
    GlobalTensor<float> kkWsGm_;
    GlobalTensor<float> AttnWsGm_;
    GlobalTensor<float> GBKWsGm_;
    GlobalTensor<float> queryContinousGm_;
    GlobalTensor<float> keyContinousGm_;
    GlobalTensor<float> querytmpGm_;
    GlobalTensor<float> stageOneMask_;

    // UB queues
    TQue<QuePosition::VECIN, 1> fp32InQueue_;
    TQue<QuePosition::VECOUT, 1> fp32OutQueue_;
    TQue<QuePosition::VECOUT, 1> gOutQueue_;

    TBuf<TPosition::VECCALC> tmpBuff;

    // UB tensors
    LocalTensor<bfloat16_t> betaUbBfloat16;
    LocalTensor<float> betaUbFloat;
    LocalTensor<float> valueUbFloat;
    LocalTensor<float> attnUbFloat;
    LocalTensor<float> inverseUbFloat;
    LocalTensor<float> inverseLocal;
    LocalTensor<float> gCumUbFloat;
    LocalTensor<float> gCumExpUbFloat;
    LocalTensor<float> gCumExpBroadUbFloat;
    LocalTensor<float> gBUbFloat;
    LocalTensor<float> kUbFloat;
    LocalTensor<float> qUbFloat;
    LocalTensor<float> qPrimeUbFloat;
    LocalTensor<float> gBroadUbFloat;
    LocalTensor<float> gTransBroadUbFloat;
    LocalTensor<float> gEndBroadUbFloat;
    LocalTensor<float> gammaUbFloat;
    LocalTensor<float> maskUbFloat;
    LocalTensor<uint32_t> colBuffer;
    LocalTensor<float> qUbFloatCon;
    LocalTensor<float> kUbFloatCon;

    LocalTensor<bfloat16_t> betaLocal;
    LocalTensor<bfloat16_t> valueLocal;
    LocalTensor<bfloat16_t> kLocal;
    LocalTensor<bfloat16_t> qLocal;
    LocalTensor<bfloat16_t> qPrimeLocal;
    LocalTensor<float> vBetaLocal;
    LocalTensor<float> kkLocal;
    LocalTensor<float> gLocal;
    LocalTensor<float> gBKLocal;
    LocalTensor<float> kgLocal;
    
    LocalTensor<bfloat16_t> bf16InLocal;
    LocalTensor<bfloat16_t> bf16OutLocal;
    LocalTensor<float> fp32InLocal;
    LocalTensor<float> fp32OutLocal;
};
} // namespace ChunkGatedDeltaRule
#endif