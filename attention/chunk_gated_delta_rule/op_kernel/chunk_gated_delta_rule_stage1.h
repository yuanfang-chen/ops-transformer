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
 * \file chunk_gated_delta_rule_stage1.h.h
 * \brief
 */
#ifndef __CHUNK_GATED_DELTA_RULE_STAGE1_H
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

        uint64_t workSpaceOffset = 0;
        GBKWsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(initParams.ws + workSpaceOffset + coreIdx * chunkSize * dk * sizeof(float)));

        workSpaceOffset += tiling_->aiCoreNum * chunkSize * dk * sizeof(float);
        kkWsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(initParams.ws + workSpaceOffset + coreIdx * chunkSize * chunkSize * sizeof(float)));

        workSpaceOffset += tiling_->aiCoreNum * chunkSize * chunkSize * sizeof(float);
        vBetaWsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(initParams.ws + workSpaceOffset + coreIdx * chunkSize * dv * sizeof(float)));

        workSpaceOffset += tiling_->aiCoreNum * chunkSize * dv * sizeof(float);
        AttnWsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(initParams.ws + workSpaceOffset + coreIdx * chunkSize * chunkSize * sizeof(float)));

        workSpaceOffset += tiling_->aiCoreNum * chunkSize * chunkSize * sizeof(float);
        queryContinousGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(initParams.ws + workSpaceOffset + coreIdx * chunkSize * dk * sizeof(float)));

        workSpaceOffset += tiling_->aiCoreNum * chunkSize * dk * sizeof(float);
        keyContinousGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(initParams.ws + workSpaceOffset + coreIdx * chunkSize * dk * sizeof(float)));
    }

    __aicore__ inline void InitLocalBuffers()
    {
        if ASCEND_IS_AIC {
            return;
        }
        uint32_t maxLen = AscendC::Std::max(AscendC::Std::max(dv, dk), chunkSize);
        pipe_->InitBuffer(fp32InQueue_, STAGEONE_BUFFER_NUM, chunkSize * chunkSize / 2 * sizeof(float));      // 8KB
        pipe_->InitBuffer(fp32OutQueue_, STAGEONE_BUFFER_NUM, chunkSize * maxLen / 2 * sizeof(float));        // 16KB
        pipe_->InitBuffer(gOutQueue_, STAGEONE_BUFFER_NUM, chunkSize * sizeof(float)); // 总24.25KB

        pipe_->InitBuffer(tmpBuff, UB_REST_BYTES);
        uint32_t buffOffset = 0;
        betaUbBfloat16 = tmpBuff.GetWithOffset<bfloat16_t>(static_cast<uint32_t>(chunkSize / 2), buffOffset);
        buffOffset += chunkSize / 2 * sizeof(bfloat16_t);
        
        gCumUbFloat = tmpBuff.GetWithOffset<float>(static_cast<uint32_t>(chunkSize), buffOffset);  // 0.25KB
        buffOffset += chunkSize * sizeof(float);

        gBUbFloat = tmpBuff.GetWithOffset<float>(static_cast<uint32_t>(chunkSize / 2), buffOffset);  // 0.125KB 
        buffOffset += chunkSize / 2 * sizeof(float);

        gEndBroadUbFloat = tmpBuff.GetWithOffset<float>(static_cast<uint32_t>(chunkSize / 2), buffOffset);  // 0.125KB
        buffOffset += chunkSize / 2 * sizeof(float);

        betaUbFloat = tmpBuff.GetWithOffset<float>(static_cast<uint32_t>(chunkSize / 2), buffOffset);   // 0.125KB
        buffOffset += chunkSize / 2 * sizeof(float);

        maxLen = AscendC::Std::max(AscendC::Std::max(dv / 2, dk / 2), chunkSize);
        gBroadUbFloat = tmpBuff.GetWithOffset<float>(static_cast<uint32_t>(chunkSize * maxLen), buffOffset);  // 16KB
        gammaUbFloat = gBroadUbFloat;   // 16KB
        kUbFloat = gBroadUbFloat;       // 32KB/2 = 16KB
        valueUbFloat = gBroadUbFloat;   // 32KB/2 = 16KB
        qUbFloat = gBroadUbFloat;       // 32KB/2 = 16KB
        buffOffset += chunkSize * maxLen  * sizeof(float);
        
        gTransBroadUbFloat = tmpBuff.GetWithOffset<float>(static_cast<uint32_t>(chunkSize * maxLen), buffOffset);    // 16KB
        attnUbFloat = gTransBroadUbFloat;           // 16KB/2 = 8KB
        inverseUbFloat = gTransBroadUbFloat[chunkSize * chunkSize / 2];
        gCumExpBroadUbFloat = gTransBroadUbFloat;   // 32KB/2 = 16KB
        qPrimeUbFloat = gTransBroadUbFloat;         // 16KB
        buffOffset += chunkSize * maxLen * sizeof(float);

        identityUbFloat = tmpBuff.GetWithOffset<float>(static_cast<uint32_t>(chunkSize * chunkSize / 2), buffOffset);  // 8KB
        buffOffset += chunkSize * chunkSize / 2 * sizeof(float);

        maskUbFloat = tmpBuff.GetWithOffset<float>(static_cast<uint32_t>(chunkSize), buffOffset);  // 8KB
        buffOffset += chunkSize * sizeof(float);

        qUbFloatCon = tmpBuff.GetWithOffset<float>(static_cast<uint32_t>(chunkSize / 2 * dk), buffOffset);  // 8KB
        buffOffset += chunkSize / 2 * dk * sizeof(float);

        kUbFloatCon = tmpBuff.GetWithOffset<float>(static_cast<uint32_t>(chunkSize / 2 * dk), buffOffset);  // 8KB
        buffOffset += chunkSize / 2 * dk * sizeof(float);

    }

    __aicore__ inline void Init(const GDRStageOneInitParams &initParams, TPipe *pipe, const ChunkGatedDeltaRuleTilingData *tilingData)
    {
        pipe_ = pipe;
        tiling_ = tilingData;
        Nk_ = tiling_->nk;
        Nv_ = tiling_->nv;
        chunkSize = tiling_->chunkSize;
        dk = tiling_->dk;
        dv = tiling_->dv;
        scale = tiling_->scale;
        validLen_ = chunkSize;
        cg_ = initParams.cg;
        v_row_stride_ = Nv_ * dv;
        NumChunk_ = (cg_.length + chunkSize - 1) / chunkSize;
        subBlockIdx = GetSubBlockIdx();
        coreIdx = GetBlockIdx();
        if ASCEND_IS_AIV{
            coreIdx /= GetTaskRatio();
        }
        coreNum = tiling_->aiCoreNum;
        SetGlobalTensors(initParams);
        InitLocalBuffers();
    }

    __aicore__ inline void Process() 
    {
        uint32_t totalChunk = Nv_ * NumChunk_;
        uint32_t tailChunkNum = totalChunk / coreNum;   // tail核处理的块数
        uint32_t formerChunkNum = tailChunkNum + 1;     // former核处理的块数
        uint32_t formerCoreNum = totalChunk % coreNum;  // former核数量
        uint32_t start, end;
        if(coreIdx < formerCoreNum){
            start = coreIdx * formerChunkNum;
            end = start + formerChunkNum;
        } else {
            start = formerCoreNum * formerChunkNum + (coreIdx - formerCoreNum) * tailChunkNum;
            end = start + tailChunkNum;
        }

        for (int32_t task_id = start; task_id < end; ++task_id) {
            uint64_t nid   = task_id % Nv_;
            uint64_t cg_id = task_id / Nv_;
            // 尾chunk处理
            uint64_t valid_len = chunkSize;
            if (cg_id == NumChunk_ - 1 && cg_.length % chunkSize != 0) {
                validLen_ = cg_.length % chunkSize;
            }
            // chunk在全局T上的起始行 = chunkGroup起始行 + chunk内偏移
            uint64_t chunk_start_row = cg_.startPos + cg_id * chunkSize;
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
        uint64_t kid = nid * Nk_ / Nv_;
        uint64_t sub_row = chunk_start_row + subBlockIdx * chunkSize / 2;
        uint64_t qk_base = sub_row * Nk_ * dk + kid * dk;
        queryGm_ = queryBaseGm_[qk_base];
        keyGm_   = keyBaseGm_[qk_base];

        uint64_t vOffset = chunk_start_row * Nv_ * dv + nid * dv;
        valueGm_ = valueBaseGm_[vOffset];

        uint64_t bgOffset = chunk_start_row * Nv_ + nid;
        betaGm_ = betaBaseGm_[bgOffset];
        gGm_ = gBaseGm_[bgOffset];

        uint64_t cgLen_pad = (cg_.length + chunkSize - 1) / chunkSize * chunkSize;
        uint64_t cb = nid * cgLen_pad + local_cid * chunkSize;

        outGCumExpGm_ = outGCumExpBaseGm_[cb];
        outKCumdecayGm_ = outKCumdecayBaseGm_[cb * dk];
        outQgGm_ = outQGBaseGm_[cb * dk];
        outKgGm_ = outKgBaseGm_[cb * dk];
        outVInnerGm_ = outVInnerBaseGm_[cb * dv];
        outQkGm_ = outQkBaseGm_[cb * chunkSize];
    }

    __aicore__ inline void ProcessOneChunk()
    {
        if ASCEND_IS_AIC {
            AscendC::CrossCoreWaitFlag(0x9);  //同步0
            AICProcess(keyContinousGm_, keyContinousGm_, kkWsGm_, chunkSize, chunkSize, dk, chunkSize, chunkSize, dk, true);    // KK
            AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(0x8);  //同步1
            AICProcess(queryContinousGm_, keyContinousGm_, outQkGm_, chunkSize, chunkSize, dk, chunkSize, chunkSize, dk, true);    //// 阶段一输出 用于调试
            AscendC::CrossCoreWaitFlag(0x7);  //同步2
            AttnInverseMMCompute();
            AscendC::CrossCoreWaitFlag(0x6);  //同步3
            kCumDecayCompute();
            AscendC::CrossCoreWaitFlag(0x5);  //同步4
            AICProcess(AttnWsGm_, vBetaWsGm_, outVInnerGm_, chunkSize, dv, chunkSize, chunkSize, dv, chunkSize); // 阶段一输出
        }
        if ASCEND_IS_AIV {
            QKPreProcess();
            AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(0x9);  //同步0
            GCumExpCompute();
            GammaCompute();
            BetaCopyInWithStride();
            AscendC::CrossCoreWaitFlag(0x8);  //同步1
            KKBetaCompute();
            InverseCompute();
            AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(0x7);  //同步2

            GBKCompute();
            AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(0x6);  //同步3
            vBetaCompute();
            AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(0x5);  //同步4
            QPrimeCompute();
        }
    }


    __aicore__ inline void QKPreProcessCompute(const GlobalTensor<bfloat16_t>& srcGm, const GlobalTensor<float>& dstGm,
                                                LocalTensor<float>& dstBuffer)
    {
        uint32_t rows = chunkSize / 2;
        uint64_t validRow = chunkSize / 2;
        if (validLen_ < chunkSize / 2) {
            validRow = (subBlockIdx == 0) ? validLen_ : 0;
        } else {
            validRow = (subBlockIdx == 0) ? chunkSize / 2 : validLen_ - chunkSize / 2;
        }
        LocalTensor<float> tmpTensor;
        //copyIn
        DataCopyInBf16WithStride(validRow, dk, srcGm, Nk_ *dk);

        // compute
        LocalTensor<bfloat16_t> bf16Tensor = fp32InQueue_.DeQue<bfloat16_t>();
        Cast(dstBuffer, bf16Tensor, AscendC::RoundMode::CAST_NONE, validRow * dk);
        PipeBarrier<PIPE_V>();
        fp32InQueue_.FreeTensor(bf16Tensor);

        if (validRow < rows) {
            Duplicate(dstBuffer[validRow * dk], static_cast<float>(0.0f), (chunkSize / 2 - validRow) * dk);
            PipeBarrier<PIPE_V>();
        }

        //copyOut
        tmpTensor = fp32OutQueue_.AllocTensor<float>();
        DataCopy(tmpTensor, dstBuffer, rows * dk);
        fp32OutQueue_.EnQue(tmpTensor);
        tmpTensor = fp32OutQueue_.DeQue<float>();

        DataCopyExtParams outParams{static_cast<uint16_t>(rows),
                                    static_cast<uint32_t>(dk * sizeof(float)), 0, 0, 0};
        DataCopyPad(dstGm, tmpTensor, outParams);
        fp32OutQueue_.FreeTensor(tmpTensor);
    }

    __aicore__ inline void QKPreProcess(){
        if ASCEND_IS_AIC {
            return;
        }
        uint32_t rows = chunkSize / 2;
        uint64_t outOffset = subBlockIdx * rows * dk;
        QKPreProcessCompute(queryGm_, queryContinousGm_[outOffset], qUbFloatCon); // [inOffset]
        QKPreProcessCompute(keyGm_, keyContinousGm_[outOffset], kUbFloatCon); // [inOffset]
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
        uint32_t inner = chunkSize;
        CumSumInfo cumSumInfo{outer, inner};  // outer-行, inner-列
        CumSum<float>(gCumUbFloat, gCumUbFloat, gCumUbFloat, cumSumInfo);
        PipeBarrier<PIPE_V>();
        // Exp计算
        gCumExpUbFloat = gOutQueue_.AllocTensor<float>();
        Exp<float, 0, true>(gCumExpUbFloat, gCumUbFloat, chunkSize);
        gOutQueue_.EnQue<float>(gCumExpUbFloat);
        DataCopyOutG(chunkSize);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void GammaCompute()
    {
        if ASCEND_IS_AIC {
            return;
        }
        // BroadCast
        uint32_t divShape[2] = {chunkSize, chunkSize};
        uint32_t gShape[2] = {chunkSize, 1};
        uint32_t gTransShape[2] = {1, chunkSize};
        Broadcast<float, 2, 1>(gBroadUbFloat, gCumExpUbFloat, divShape, gShape);
        Broadcast<float, 2, 0>(gTransBroadUbFloat, gCumExpUbFloat, divShape, gTransShape);
        PipeBarrier<PIPE_V>();
        // div
        Div(gammaUbFloat, gBroadUbFloat, gTransBroadUbFloat, chunkSize * chunkSize);
        PipeBarrier<PIPE_V>();
        // SetMaskNorm();
        // SetVectorMask<float, MaskMode::NORMAL>(); // 按行计算得算chunkSize次
        // tril(严格下三角) setvecormask
        ////////////////////////////////////////////////////////////////////////////////临时掩码
        Duplicate(maskUbFloat, static_cast<float>(0.0), chunkSize);
        PipeBarrier<PIPE_V>();
        if (subBlockIdx == 1){
            Duplicate(maskUbFloat, static_cast<float>(1.0), chunkSize / 2);
            PipeBarrier<PIPE_V>();
        }
        for (int i = 0; i < chunkSize / 2; ++i){    ///////////////////和SetVectorMask的优劣///////////////////
            Mul(gammaUbFloat[i * chunkSize + subBlockIdx * chunkSize * chunkSize / 2], gammaUbFloat[i * chunkSize + subBlockIdx * chunkSize * chunkSize / 2], maskUbFloat, chunkSize);
            PipeBarrier<PIPE_V>();
            maskUbFloat.SetValue(i + chunkSize * subBlockIdx / 2, 1);
            PipeBarrier<PIPE_V>();
        }
        PipeBarrier<PIPE_V>();
        /////////////////////////////////////////////////////////////////////////////////
    }

    __aicore__ inline void KKBetaCompute()
    {
        // copy value
        uint32_t kkLength = chunkSize * chunkSize / 2;
        uint64_t kkBeginOffset = subBlockIdx * chunkSize * chunkSize / 2;
        DataCopyInFp32(kkLength, kkWsGm_[kkBeginOffset]);
        kkLocal = fp32InQueue_.DeQue<float>();

        uint32_t betaShape[2] = {chunkSize / 2, 1};
        uint32_t kkShape[2] = {chunkSize / 2, chunkSize};
        uint64_t betaBeginOffset = subBlockIdx * chunkSize / 2;
        Broadcast<float, 2, 1>(attnUbFloat, betaUbFloat, kkShape, betaShape);
        PipeBarrier<PIPE_V>();
        Mul(attnUbFloat, kkLocal, attnUbFloat, chunkSize * chunkSize / 2);
        PipeBarrier<PIPE_V>();
        fp32InQueue_.FreeTensor(kkLocal);
    }

    __aicore__ inline void InverseCompute()
    {
        uint64_t curVecLen = chunkSize * chunkSize / 2;
        uint64_t attnBeginOffset = subBlockIdx * chunkSize / 2;

        Mul(attnUbFloat, attnUbFloat, gammaUbFloat[subBlockIdx * chunkSize * chunkSize / 2], curVecLen);
        PipeBarrier<PIPE_V>();

        uint32_t inverseVecLen = 16;
        inverseLocal = fp32OutQueue_.AllocTensor<float>();
        Muls(inverseLocal, attnUbFloat, static_cast<float>(-1.0), curVecLen);
        PipeBarrier<PIPE_V>();

        InverseAIV(attnBeginOffset, inverseVecLen);
        InverseAIV(attnBeginOffset + inverseVecLen * chunkSize + inverseVecLen, inverseVecLen);
        fp32OutQueue_.EnQue(inverseLocal);
        DataCopyOutFp32(curVecLen, AttnWsGm_[subBlockIdx * curVecLen]);
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
            colBufferGather.SetValue<uint32_t>(offsetIdx++, (j * chunkSize) * sizeof(float));
        }
        for (int i = 1; i < inverseVecLen; ++i) {
            uint32_t curI = i - 1;
            uint32_t validRows = inverseVecLen - i;
            Gather(col, attnUbFloat[offset + i * chunkSize + curI], colBufferGather, (uint32_t)0, validRows);

            uint32_t dstShape[2] = {validRows, inverseVecLen};
            uint32_t colSrcShape[2] = {validRows, 1};
            Broadcast<float, 2, 1>(col[inverseVecLen], col, dstShape, colSrcShape);
            Broadcast<float, 2, 0>(row, inverseLocal[offset + curI * chunkSize], dstShape, srcShape);
            MulAddDst(yLocal[i * inverseVecLen], col[inverseVecLen], row, inverseVecLen * validRows);
            PipeBarrier<PIPE_V>();
            ei.SetValue(i - 1, static_cast<float>(0.0));
            ei.SetValue(i, static_cast<float>(1.0));
            // xi = (I - SUM) / Lii = I - SUM
            Sub(inverseLocal[offset + i * chunkSize], ei, yLocal[i * inverseVecLen], inverseVecLen);
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void GBKCompute()
    {
        if ASCEND_IS_AIC {
            return;
        }
        // mul -> gBeta, Mul(dst, src0, src1, count)
        uint64_t gCumExpBeginOffset = subBlockIdx * chunkSize / 2;
        Mul(gBUbFloat, betaUbFloat, gCumExpUbFloat[gCumExpBeginOffset], chunkSize / 2); //  [C/2, 1]
        PipeBarrier<PIPE_V>();
        float minusOne = -1.0;
        Muls(gBUbFloat, gBUbFloat, minusOne, chunkSize / 2); // Muls(dst, src, scalar, count)
        PipeBarrier<PIPE_V>();
        // mul -> gBetaK   广播 [C, 1] -> [C, Dk]
        uint32_t betaShape[2] = {chunkSize / 2, 1};
        uint32_t kShape[2] = {chunkSize / 2, dk};
        // Broadcast<T, dim, axis, isReduceSource = false>(dst, src, dstshape, srcshape)
        gBKLocal = fp32OutQueue_.AllocTensor<float>();
        Broadcast<float, 2, 1>(gBKLocal, gBUbFloat, kShape, betaShape); // [C/2, 1] -> [C/2, Dk]
        PipeBarrier<PIPE_V>();
        Mul(gBKLocal, gBKLocal, kUbFloatCon, chunkSize * dk / 2);
        fp32OutQueue_.EnQue<float>(gBKLocal);
        uint64_t GBKBeginOffset = subBlockIdx * chunkSize * dk / 2;
        DataCopyOutFp32(chunkSize * dk / 2, GBKWsGm_[GBKBeginOffset]);
        PipeBarrier<PIPE_V>();
        // -----start【kg前置】
        uint32_t gEndShape[2] = {1, 1};
        uint32_t gBroadShape[2] = {chunkSize / 2, 1};
        Broadcast<float, 2, 0>(gEndBroadUbFloat, gCumExpUbFloat[chunkSize - 1], gBroadShape, gEndShape);
        PipeBarrier<PIPE_V>();
        uint64_t gEndBeginOffset = subBlockIdx * chunkSize / 2;
        Div(gEndBroadUbFloat, gEndBroadUbFloat, gCumExpUbFloat[gEndBeginOffset], chunkSize / 2); // Div(dst, src0, src1, count)
        PipeBarrier<PIPE_V>();
        kgLocal = fp32OutQueue_.AllocTensor<float>();
        Broadcast<float, 2, 1>(kgLocal, gEndBroadUbFloat, kShape, gBroadShape);
        PipeBarrier<PIPE_V>();
        Mul(kgLocal, kgLocal, kUbFloatCon, chunkSize * dk / 2);
        PipeBarrier<PIPE_V>();
        fp32OutQueue_.EnQue<float>(kgLocal);
        uint64_t kgBeginOffset = subBlockIdx * chunkSize * dk / 2;
        DataCopyOutFp32(chunkSize * dk / 2, outKgGm_[kgBeginOffset]);  // 阶段一输出
    }

    __aicore__ inline void vBetaCompute()
    {
        // copy value
        uint32_t valueLength = chunkSize / 2 * dv;
        uint64_t vBeginOffset = subBlockIdx * (chunkSize / 2) * v_row_stride_;
        uint64_t validRow = chunkSize / 2;
        if (validLen_ < chunkSize / 2) {
            validRow = (subBlockIdx == 0) ? validLen_ : 0;
        } else {
            validRow = (subBlockIdx == 0) ? chunkSize / 2 : validLen_ - chunkSize / 2;
        }
        DataCopyInBf16WithStride(validRow, dv, valueGm_[vBeginOffset], v_row_stride_);  // [c,dv]
        valueLocal = fp32InQueue_.DeQue<bfloat16_t>();
        vBetaLocal = fp32OutQueue_.AllocTensor<float>();
        Cast(valueUbFloat, valueLocal, AscendC::RoundMode::CAST_NONE, validRow * dv);
        PipeBarrier<PIPE_V>();
        fp32InQueue_.FreeTensor(valueLocal);
        if (validLen_ < chunkSize / 2) {
            Duplicate(valueUbFloat[validLen_ * dv], static_cast<float>(0.0f), (chunkSize / 2 - validLen_) * dv);
            PipeBarrier<PIPE_V>();
        }
        
        uint32_t betaShape[2] = {chunkSize / 2, 1};
        uint32_t vShape[2] = {chunkSize / 2, dv};
        uint64_t betaBeginOffset = subBlockIdx * chunkSize / 2;
        Broadcast<float, 2, 1>(vBetaLocal, betaUbFloat, vShape, betaShape); // [C/2, 1] -> [C/2, Dv]
        PipeBarrier<PIPE_V>();
        Mul(vBetaLocal, valueUbFloat, vBetaLocal, chunkSize * dv / 2);
        PipeBarrier<PIPE_V>();
        fp32OutQueue_.EnQue<float>(vBetaLocal);
        DataCopyOutFp32(valueLength, vBetaWsGm_[subBlockIdx * chunkSize / 2 * dv]); // 非阶段一输出
    }

    __aicore__ inline void QPrimeCompute()
    {
        if ASCEND_IS_AIC {
            return;
        }
        // Get scale
        // scale = scale_valueGm_.GetValue(0);
        Muls(qUbFloat, qUbFloatCon, scale, chunkSize * dk / 2);
        PipeBarrier<PIPE_V>();
        uint32_t gCumExpShape[2] = {chunkSize / 2, 1};
        uint32_t qShape[2] = {chunkSize / 2, dk};
        uint64_t gCumExpBeginOffset = subBlockIdx * chunkSize / 2;
        Broadcast<float, 2, 1>(gCumExpBroadUbFloat, gCumExpUbFloat[gCumExpBeginOffset], qShape, gCumExpShape);
        PipeBarrier<PIPE_V>();
        // Mul -> qPrimeLocal;
        Mul(qPrimeUbFloat, qUbFloat, gCumExpBroadUbFloat, chunkSize * dk / 2);
        PipeBarrier<PIPE_V>();
        qPrimeLocal = fp32OutQueue_.AllocTensor<bfloat16_t>();
        Cast(qPrimeLocal, qPrimeUbFloat, AscendC::RoundMode::CAST_RINT, chunkSize * dk / 2);
        fp32OutQueue_.EnQue<bfloat16_t>(qPrimeLocal);
        uint64_t qgBeginOffset = subBlockIdx * chunkSize * dk / 2;
        DataCopyOutBf16(chunkSize * dk / 2, outQgGm_[qgBeginOffset]);// 阶段一输出
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
        // betaShape(T, Nv), 行间距 Nv
        // stride版本: cols=1, stride=Nv
        uint64_t betaBeginOffset = subBlockIdx * (chunkSize / 2) * Nv_;
        uint64_t validRow = chunkSize / 2;
        if (validLen_ < chunkSize / 2) {
            validRow = (subBlockIdx == 0) ? validLen_ : 0;
        } else {
            validRow = (subBlockIdx == 0) ? chunkSize / 2 : validLen_ - chunkSize / 2;
        }
        DataCopyInBf16WithStride(validRow, 1, betaGm_[betaBeginOffset], Nv_);
        betaLocal = fp32InQueue_.DeQue<bfloat16_t>();
        constexpr uint32_t slot = 32 / sizeof(bfloat16_t);
        for (uint32_t i = 0; i < validRow; ++i) {
            betaUbBfloat16.SetValue(i, betaLocal.GetValue(i * slot));
        }
        for (uint32_t i = validRow; i < chunkSize / 2; ++i) {
            betaUbBfloat16.SetValue(i, bfloat16_t(0.0f));
        }
        Cast(betaUbFloat, betaUbBfloat16, AscendC::RoundMode::CAST_NONE, chunkSize / 2);
        PipeBarrier<PIPE_V>();
        fp32InQueue_.FreeTensor(betaLocal);
    }

    __aicore__ inline void GCopyInWithStride()
    {
        // g 是全局量，两个 subVec 共用，只需 subVec 0 搬全部 chunkSize 行?
        // if (subBlockIdx != 0) return;
        constexpr uint32_t slot = 32 / sizeof(float);
        uint64_t validRow = validLen_;
        DataCopyInFp32WithStride(validRow, 1, gGm_, Nv_);  // TODO [GetBlockIdx() / 2]
        gLocal = fp32InQueue_.DeQue<float>();
        for (uint32_t i = 0; i < validRow; ++i) {
            gCumUbFloat.SetValue(i, gLocal.GetValue(i * slot));
        }
        for (uint32_t i = validRow; i < chunkSize; ++i) {
            gCumUbFloat.SetValue(i, float(0.0f));
        }
        PipeBarrier<PIPE_V>();
        fp32InQueue_.FreeTensor(gLocal);
    }

    __aicore__ inline void DataCopyInFp32WithStride(uint64_t rows,  // 要搬的行数(=chunkSize 或 valid_len)
                                                    uint64_t cols,  // 每行的元素数(= dk 或 dv 或 1)
                                                    const GlobalTensor<float> src,
                                                    uint64_t srcRowStride) // GM上相邻行的间距(元素数), = Nk * Dk 或 Nv * Dv 或 Nv
    {
        DataCopyPadExtParams<float> padParams = {false, static_cast<uint8_t>(0), static_cast<uint8_t>(0),
                                                      static_cast<float>(0)};
        uint32_t srcGap = (srcRowStride - cols) * sizeof(float);
        DataCopyExtParams params{static_cast<uint16_t>(rows),
                                 static_cast<uint32_t>(cols * sizeof(float)),
                                 static_cast<uint32_t>(srcGap), 0, 0};  // srcStride
        fp32InLocal = fp32InQueue_.AllocTensor<float>();
        DataCopyPad(fp32InLocal, src, params, padParams);
        fp32InQueue_.EnQue<float>(fp32InLocal);
    }

    __aicore__ inline void DataCopyInBf16WithStride(uint64_t rows,  // 要搬的行数(=chunkSize 或 valid_len)
                                                    uint64_t cols,  // 每行的元素数(= dk 或 dv 或 1)
                                                    GlobalTensor<bfloat16_t> src,
                                                    uint64_t srcRowStride, // GM上相邻行的间距(元素数), = Nk * Dk 或 Nv * Dv 或 Nv{false, 0, 0, 0}
                                                    uint64_t pad_rows = 0)
    {
        DataCopyPadExtParams<bfloat16_t> padParams = {false, static_cast<uint8_t>(0), static_cast<uint8_t>(0),
                                                      static_cast<float>(0)};
        uint32_t srcGap = (srcRowStride - cols) * sizeof(bfloat16_t);
        DataCopyExtParams params{static_cast<uint16_t>(rows),
                                 static_cast<uint32_t>(cols * sizeof(bfloat16_t)),
                                 static_cast<uint32_t>(srcGap), 0, 0};  // srcStride
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
        DataCopyExtParams params{static_cast<uint16_t>(1),  // blockCount
                                    static_cast<uint16_t>(length * sizeof(float)),  // blockLen 
                                    0, 0, 0};  // srcStride, dstStride
        DataCopyPad(outGCumExpGm_, gCumExpUbFloat, params);  // dst, src   // 阶段一输出
    }

    __aicore__ inline void AttnInverseMMCompute()
    {
        uint32_t curLen = 16;
        uint64_t beginAddr = 0;
        UpdateLowerBlock(beginAddr, curLen);
        
        beginAddr = chunkSize * chunkSize / 2 + chunkSize / 2;
        UpdateLowerBlock(beginAddr, curLen);

        curLen = 32;
        beginAddr = 0;
        UpdateLowerBlock(beginAddr, curLen);
    }

    __aicore__ inline void UpdateLowerBlock(uint64_t beginAddr, uint64_t curLen)
    {
        uint64_t leftDown = beginAddr + chunkSize * curLen;
        uint64_t rightDown = leftDown + curLen;
        // 右矩阵左下角 @ 右矩阵左上角 -> 右矩阵左下角
        AICProcess(AttnWsGm_[leftDown], AttnWsGm_[beginAddr], AttnWsGm_[leftDown], chunkSize, chunkSize, chunkSize, curLen, curLen, curLen);
        SetFlag<HardEvent::FIX_MTE2>(EVENT_ID1);
        WaitFlag<HardEvent::FIX_MTE2>(EVENT_ID1);
        // 右矩阵右下角 @ 右矩阵左下角 -> 右矩阵左下角
        AICProcess(AttnWsGm_[rightDown], AttnWsGm_[leftDown], AttnWsGm_[leftDown], chunkSize, chunkSize, chunkSize, curLen, curLen, curLen);
        SetFlag<HardEvent::FIX_MTE2>(EVENT_ID1);
        WaitFlag<HardEvent::FIX_MTE2>(EVENT_ID1);
    }

    __aicore__ inline void AICProcess(GlobalTensor<float> x, GlobalTensor<float> y, GlobalTensor<float> z, 
                                      uint64_t m, uint64_t n, uint64_t k, uint64_t sm, uint64_t sn, uint64_t sk, bool transB=false)
    {
        mmFp32.SetOrgShape(m, n, k);    // MNK
        mmFp32.SetSingleShape(sm, sn, sk); // SingleCOreMNK
        mmFp32.SetTensorA(x);
        mmFp32.SetTensorB(y, transB);
        mmFp32.IterateAll(z);
        mmFp32.End();
    }

    __aicore__ inline void kCumDecayCompute()
    {
        mmBf16.SetOrgShape(chunkSize, dk, chunkSize);
        mmBf16.SetSingleShape(chunkSize, dk, chunkSize);
        mmBf16.SetTensorA(AttnWsGm_);
        mmBf16.SetTensorB(GBKWsGm_);
        mmBf16.IterateAll(outKCumdecayGm_);// 阶段一输出，
        mmBf16.End();
    }

    MT_FP32 &mmFp32;
    MT_BF16 &mmBf16;
    // TCubeTiling tilingData;
    const ChunkGatedDeltaRuleTilingData *tiling_;
    ChunkGroup cg_;
    uint32_t Nk_;
    uint32_t Nv_;
    uint32_t NumChunk_;
    uint64_t v_row_stride_;

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

    TPipe *pipe_;
    uint32_t chunkSize;
    uint32_t validLen_;
    uint32_t dk;
    uint32_t dv;
    uint32_t coreNum;
    float scale;

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
    LocalTensor<float> identityUbFloat;
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

    uint32_t subBlockIdx;
    uint32_t coreIdx;
};
} // namespace ChunkGatedDeltaRule