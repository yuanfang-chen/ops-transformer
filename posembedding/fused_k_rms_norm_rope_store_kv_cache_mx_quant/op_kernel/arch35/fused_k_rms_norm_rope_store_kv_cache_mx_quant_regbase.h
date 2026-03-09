/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fused_k_rms_norm_rope_store_kv_cache_mx_quant_regbase.h
 * \brief
 */

#ifndef FUSED_K_RMS_NORM_ROPE_STORE_KV_CACHE_MX_QUANT_REGBASE_H_
#define FUSED_K_RMS_NORM_ROPE_STORE_KV_CACHE_MX_QUANT_REGBASE_H_

namespace FusedKRmsNormRopeStoreKvCacheMxQuant {

using namespace AscendC;


/*
gm->ub拷入
lineNum：拷入的行数
srcLineLength：原Gm中数据一行的长度
copyLineLength：从原Gm中实际需要拷入的长度
拷入完成后dstUb中的数据为lineNum*copyLineLengthPadAlign
*/
template <typename T>
__aicore__ inline void CopyInLineAlign(
    const LocalTensor<T>& dstUb, const GlobalTensor<T>& srcGm, int64_t lineNum, int64_t srcLineLength,
    int64_t copyLineLength)
{
    DataCopyPadExtParams<T> padParams{true, 0, 0, 0};
    DataCopyExtParams copyInParams;
    copyInParams.blockCount = lineNum;
    copyInParams.blockLen = copyLineLength * sizeof(T);
    copyInParams.srcStride = (srcLineLength - copyLineLength) * sizeof(T);
    copyInParams.dstStride = 0;
    DataCopyPad(dstUb, srcGm, copyInParams, padParams);
}


template <typename T_QKV>
class FusedKRmsNormRopeStoreKvCacheMxQuantRegbase
{
public:
    __aicore__ inline FusedKRmsNormRopeStoreKvCacheMxQuantRegbase(
        TPipe* pipe, const FusedKRmsNormRopeStoreKvCacheMxQuantRegbaseTilingData* tiling)
    {
        tilingData_ = tiling;
        pipe_ = pipe;
    }

    __aicore__ inline void Init(
        GM_ADDR qkv, GM_ADDR cos, GM_ADDR sin, GM_ADDR gamma, GM_ADDR kv_slot_mapping, GM_ADDR v_scale_slot_mapping, GM_ADDR k_cache,
        GM_ADDR k_scale_cache, GM_ADDR v_cache, GM_ADDR v_scale_cache, GM_ADDR q, GM_ADDR q_scale)
    {
        // 获取分块操作中的单个块的大小。判断是否是最后一块，是最后一块，则等于剩余元素的数量，否则等于固定的单核处理的行数
        int64_t currentBlockFactor = tilingData_->blockFactor;
        if (this->blockIdx == (this->useCoreNum - 1)) {
            currentBlockFactor =
                tilingData_->batchSize * tilingData_->seqLength - (this->useCoreNum - 1) * tilingData_->blockFactor;
        }
        this->reciprocal = tilingData_->reciprocal;
        this->epsilon = tilingData_->epsilon;
        this->dv = tilingData_->dv;
        this->dvAlign = tilingData_->dvAlign;
        this->dvB8Align = tilingData_->dvB8Align;
        this->dkAlign = tilingData_->dkAlign;
        this->dk = tilingData_->dk;
        this->ubFactor = tilingData_->ubFactor;
 
        // ubLoop: 表示需要多少个子块来覆盖singleA大小的数据
        this->ubLoop = (currentBlockFactor + this->ubFactor - 1) / this->ubFactor;
        // ubLoop：最后一块
        this->ubTail = currentBlockFactor - (this->ubLoop - 1) * this->ubFactor;
        dKV = (tilingData_->dk + tilingData_->dv);
        kScaleCopyLen = tilingData_->kScaleType == 1 ? 1 : tilingData_->halfDk;
        kScaleCopyOffset = tilingData_->kScaleType == 1 ? 0 : tilingData_->halfDk;
        kOffsetCopyLen = tilingData_->kOffsetType == 1 ? 1 : tilingData_->halfDk;
        kOffsetCopyOffset = tilingData_->kOffsetType == 1 ? 0 : tilingData_->halfDk;
        vScaleCopyLen = tilingData_->vScaleType == 1 ? 1 : tilingData_->dv;
        vOffsetCopyLen = tilingData_->vOffsetType == 1 ? 1 : tilingData_->dv;

        kVFLoop = ops::FloorDiv(tilingData_->halfDk, static_cast<int64_t>(VL_FP32));
        kVFTail = tilingData_->halfDk % VL_FP32;

        this->dvLoopCount = ops::CeilDiv(tilingData_->dv, static_cast<int64_t>(VL_FP32));

        // init global memory
        qkvGm_.SetGlobalBuffer((__gm__ T_QKV*)qkv);
        cosGm_.SetGlobalBuffer((__gm__ T_QKV*)cos);
        sinGm_.SetGlobalBuffer((__gm__ T_QKV*)sin);
        gammaGm_.SetGlobalBuffer((__gm__ float*)gamma);
        kvSlotMappingGm_.SetGlobalBuffer((__gm__ int64_t*)kv_slot_mapping);
        vScaleSlotMappingGm_.SetGlobalBuffer((__gm__ int64_t*)v_scale_slot_mapping);
        kCacheGm_.SetGlobalBuffer((__gm__ uint8_t*)k_cache);
        vCacheGm_.SetGlobalBuffer((__gm__ uint8_t*)v_cache);
        kScaleCacheGm_.SetGlobalBuffer((__gm__ uint8_t*)k_scale_cache);
        vScaleCacheGm_.SetGlobalBuffer((__gm__ uint8_t*)v_scale_cache);
        qGm_.SetGlobalBuffer((__gm__ uint8_t*)q);
        qScaleGm_.SetGlobalBuffer((__gm__ uint8_t*)q_scale);

        // init pipe
        pipe_->InitBuffer(inQueueGamma, BUFFER_COUNT_SINGLE, tilingData_->dvAlign * sizeof(T_QKV));
        pipe_->InitBuffer(inQueueCosSin, BUFFER_COUNT_DOUBLE, this->ubFactor * COS_SIN_CHANNEL_COUNT * tilingData_->halfDkAlign * sizeof(T_QKV));
        pipe_->InitBuffer(inQueueX, BUFFER_COUNT_DOUBLE, this->ubFactor * tilingData_->inUbSize);
        pipe_->InitBuffer(outQueue, BUFFER_COUNT_DOUBLE, this->ubFactor * tilingData_->outUbSize);
        pipe_->InitBuffer(wsBuffer0, WORKSPACE_FLOAT_VECTOR_COUNT * VL_FP32 * sizeof(float));
        pipe_->InitBuffer(wsBuffer1, this->ubFactor * tilingData_->rmsNormWspSize);
    }


    /*
    输入：srcTensor   shape为[A,R] R=128
    输入：gammaTensor shape为[R]
    输出：dstTensor   shape为[A,R]
    T仅支持bfloat16_t
    */
    template <typename T>
    __aicore__ inline void DoRmsNorm(
        const LocalTensor<float>& dstTensor, const LocalTensor<T>& srcTensor, const LocalTensor<float>& gammaTensor,
        const int64_t aSize, const int64_t rSize)
    {
    }

    /*
    输入：srcTensor  shape为[S,N,D] D=128
    输入：cosTensor  shape为[S,1,D]
    输入：sinTensor  shape为[S,1,D]
    输出：dstTensor  shape为[S,N,D]
    T1支持float和bfloat16_t
    T1仅支持bfloat16_t
    */
    template <typename T1, T2>
    __aicore__ inline void DoRope(const LocalTensor<float> &dstTensor, const LocalTensor<T1> &srcTensor,
                                  const LocalTensor<T2> &cosTensor, const LocalTensor<T2> &sinTensor,
                                  const int64_t seqSize, const int64_t nSize, const int64_t dSize)
    {
    }

    /*
    输入：srcTensor  shape为[M,D] M为64的整数倍，D=128
    输出：outTensor  shape为[M,D] 量化后的fp8数据
    输出：outScaleTensor 量化后的scale，数据类型为fp8
    quantAxis支持0或者1，量化的BlockSize固定为32
    当quantAxis为1时 outScale的shape为[M, D//32 = 4 需要pad到32，需要block对齐后续scatter]
    当quantAxis为0时 outScale的shape为[M//32//2,D,2]
    */
    __aicore__ inline void DoMxQuant(const LocalTensor<uint8_t> &outTensor, const LocalTensor<uint8_t> &outScaleTensor,
                                     const LocalTensor<float> &srcTensor, const int64_t quantAxis,
                                     const int64_t mSize, const int64_t dSize)
    {
    }

    __aicore__ inline void Process()
    {
        // Phase 1 do k rope
        GetBlockIdx();
        for (int64_t tIdx = startTIndex; tIdx < endTIndex; tIdx += tilingData_->qUbFactor) {
            int64_t processSeqLen;
            if (tIdx + tilingData_->qUbFactor < endTIndex) {
                processSeqLen = ubInner;
            } else {
                processSeqLen = endTIndex - tIdx;
            }

        }


        // CopyIn gamma
        LocalTensor<float> gammaLocal = inQueueGamma.AllocTensor<float>();
        CopyInLineAlign<float>(gammaLocal, gammaGm_, 1, tilingData_->headDim, tilingData_->dv);
        inQueueGamma.EnQue(gammaLocal);
        gammaLocal = inQueueGamma.DeQue<T_QKV>();

        // 行，有几行相当于ubA
        int64_t calcRow = this->ubFactor;

        for (int64_t loopIdx = 0; loopIdx < this->ubLoop; ++loopIdx) {
            if (loopIdx == this->ubLoop - 1) {
                calcRow = this->ubTail;
            }
            kvGlobalMemoryOffset = loopIdx * this->ubFactor * dKV;
            kOutGmOffset = loopIdx * this->ubFactor * this->dk;
            vOutGmOffset = loopIdx * this->ubFactor * this->dv;
            // 当前子块的偏移量
            int64_t startIdx = this->blockIdx * tilingData_->blockFactor + loopIdx * this->ubFactor;

            LocalTensor<T_QKV> ropeLocal = inQueueX.AllocTensor<T_QKV>();
            LocalTensor<T_QKV> realCosLocal = inQueueCosSin.AllocTensor<T_QKV>();
            LocalTensor<T_QKV> imgCosLocal = realCosLocal[this->ubFactor * tilingData_->halfDkAlign];
            LocalTensor<T_QKV> realSinLocal = imgCosLocal[this->ubFactor * tilingData_->halfDkAlign];
            LocalTensor<T_QKV> imgSinLocal = realSinLocal[this->ubFactor * tilingData_->halfDkAlign];
            CopyInLineAlign<T_QKV>(
                ropeLocal, this->kvGm[kvGlobalMemoryOffset + tilingData_->dv], calcRow, dKV, tilingData_->dk);
            if (tilingData_->cosSinNeedBrc == 0) {
                freqGlobalMemoryOffset =
                    this->blockIdx * tilingData_->blockFactor * this->dk + loopIdx * this->ubFactor * tilingData_->dk;
                CopyInLineAlign<T_QKV>(
                    realCosLocal, this->cosGm[freqGlobalMemoryOffset], calcRow, tilingData_->dk, tilingData_->halfDk);
                CopyInLineAlign<T_QKV>(
                    imgCosLocal, this->cosGm[freqGlobalMemoryOffset + tilingData_->halfDk], calcRow, tilingData_->dk,
                    tilingData_->halfDk);
                CopyInLineAlign<T_QKV>(
                    realSinLocal, this->sinGm[freqGlobalMemoryOffset], calcRow, tilingData_->dk, tilingData_->halfDk);
                CopyInLineAlign<T_QKV>(
                    imgSinLocal, this->sinGm[freqGlobalMemoryOffset + tilingData_->halfDk], calcRow, tilingData_->dk,
                    tilingData_->halfDk);
            } else {
                int64_t batchId = 0;
                int64_t ubOffset = 0;
                for (int64_t i = 0; i < calcRow; i++) {
                    batchId = (startIdx + i) / tilingData_->seqLength;
                    freqGlobalMemoryOffset = batchId * tilingData_->dk;
                    ubOffset = i * tilingData_->halfDkAlign;
                    CopyInLineAlign<T_QKV>(
                        realCosLocal[ubOffset], this->cosGm[freqGlobalMemoryOffset], 1, tilingData_->dk,
                        tilingData_->halfDk);
                    CopyInLineAlign<T_QKV>(
                        imgCosLocal[ubOffset], this->cosGm[freqGlobalMemoryOffset + tilingData_->halfDk], 1,
                        tilingData_->dk, tilingData_->halfDk);
                    CopyInLineAlign<T_QKV>(
                        realSinLocal[ubOffset], this->sinGm[freqGlobalMemoryOffset], 1, tilingData_->dk,
                        tilingData_->halfDk);
                    CopyInLineAlign<T_QKV>(
                        imgSinLocal[ubOffset], this->sinGm[freqGlobalMemoryOffset + tilingData_->halfDk], 1,
                        tilingData_->dk, tilingData_->halfDk);
                }
            }
            inQueueX.EnQue(ropeLocal);
            inQueueCosSin.EnQue(realCosLocal);
            ropeLocal = inQueueX.DeQue<T_QKV>();
            realCosLocal = inQueueCosSin.DeQue<T_QKV>();
            // Calc: RoPE
            Rope(
                ropeLocal, realCosLocal, imgCosLocal, realSinLocal, imgSinLocal, realKScaleLocal, imgKScaleLocal,
                realKOffsetLocal, imgKOffsetLocal, workspaceBuffer0, calcRow);
            inQueueX.FreeTensor(ropeLocal);
            inQueueCosSin.FreeTensor(realCosLocal);
            // 拷出到kCache
            if (tilingData_->cacheMode <= PA_NZ_CACHE_MODE) {
                ScatterUpdateK(calcRow, startIdx);
            } else {
                ScatterBlkUpdateK(calcRow, startIdx);
            }

            // 拷出到kOut
            if (tilingData_->isOutputKv > 0) {
                CopyLineAlignOut<T_QKV>(this->kOutGm[kOutGmOffset], kOutLocal, calcRow, this->dk, this->dk);
            }
            // FreeTensor 区分场景
            if constexpr (IsSameType<T_K_CACHE, int8_t>::value || IsSameType<T_K_CACHE, hifloat8_t>::value ||
                          IsSameType<T_K_CACHE, fp8_e5m2_t>::value || IsSameType<T_K_CACHE, fp8_e4m3fn_t>::value) {
                if (tilingData_->isOutputKv > 0) {
                    outQueue.FreeTensor(kOutLocal);
                } else {
                    outQueue.FreeTensor(kQuantLocal);
                }
            } else {
                outQueue.FreeTensor(kOutLocal);
            }

            // CopyIn x: [ubFactor, RmsLength]
            LocalTensor<T_QKV> xLocal = inQueueX.AllocTensor<T_QKV>();
            CopyInLineAlign<T_QKV>(xLocal, this->kvGm[kvGlobalMemoryOffset], calcRow, dKV, tilingData_->dv);
            inQueueX.EnQue(xLocal);
            xLocal = inQueueX.DeQue<T_QKV>();

            // Calc: RmsNorm
            vOutLocal = outQueue.AllocTensor<T_QKV>();
            this->RmsNorm(vOutLocal, xLocal, gammaLocal, workspaceBuffer1, vScaleLocal, vOffsetLocal, calcRow);
            inQueueX.FreeTensor(xLocal);
            outQueue.EnQue(vOutLocal);
            vOutLocal = outQueue.DeQue<T_QKV>();
            // 拷出到vCache
            if (tilingData_->cacheMode <= PA_NZ_CACHE_MODE) {
                ScatterUpdateV(calcRow, startIdx);
            } else {
                ScatterBlkUpdateV(calcRow, startIdx);
            }
            // 拷出到vOut
            if (tilingData_->isOutputKv > 0) {
                CopyLineAlignOut<T_QKV>(this->vOutGm[vOutGmOffset], vOutLocal, calcRow, this->dv, this->dv);
            }
            // FreeTensor 不论什么场景，都是申请的vOutLocal
            outQueue.FreeTensor(vOutLocal);
        }
        inQueueGamma.FreeTensor(gammaLocal);
    }

private:
    const FusedKRmsNormRopeStoreKvCacheMxQuantRegbaseTilingData* tilingData_;
    TQue<QuePosition::VECIN, 1> inQueueX, inQueueGamma, inQueueCosSin;
    TQue<QuePosition::VECOUT, 1> outQueue;
    TBuf<TPosition::VECCALC> wsBuffer0, wsBuffer1;
    TPipe* pipe_ = nullptr;

    GlobalTensor<T_QKV> qkvGm_, cosGm_, sinGm_;
    GlobalTensor<float> gammaGm_;
    GlobalTensor<int64_t> kvSlotMappingGm_, vScaleSlotMappingGm_;
    GlobalTensor<uint8_t> kCacheGm_, vCacheGm_, kScaleCacheGm_, vScaleCacheGm_, qGm_, qScaleGm_;

    LocalTensor<T_QKV> kOutLocal;
    LocalTensor<T_K_CACHE> kQuantLocal;
    LocalTensor<T_QKV> vOutLocal;
    LocalTensor<T_V_CACHE> vQuantLocal;

    LocalTensor<float> realKScaleLocal;
    LocalTensor<float> imgKScaleLocal;
    LocalTensor<float> realKOffsetLocal;
    LocalTensor<float> imgKOffsetLocal;
    LocalTensor<float> vScaleLocal;
    LocalTensor<float> vOffsetLocal;

    event_t eventIDSToMTE3;

    int64_t dKV = 0;
    int64_t kScaleCopyLen = 0;
    int64_t kScaleCopyOffset = 0;
    int64_t kOffsetCopyLen = 0;
    int64_t kOffsetCopyOffset = 0;
    int64_t vScaleCopyLen = 0;
    int64_t vOffsetCopyLen = 0;
    int64_t dk0 = 0;
    int64_t dk1 = 0;
    int64_t dv0 = 0;
    int64_t dv1 = 0;

    uint16_t kVFLoop = 0;
    uint32_t kVFTail = 0;

    int64_t kvGlobalMemoryOffset = 0;
    int64_t freqGlobalMemoryOffset = 0;
    int64_t kOutGmOffset = 0;
    int64_t vOutGmOffset = 0;
};
} // namespace FusedKRmsNormRopeStoreKvCacheMxQuant

#endif // FUSED_K_RMS_NORM_ROPE_STORE_KV_CACHE_MX_QUANT_REGBASE_H_