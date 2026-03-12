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

#include "kernel_operator.h"
#include "platform.h"

namespace FusedKRmsNormRopeStoreKvCacheMxQuant {
using namespace AscendC;

constexpr static uint32_t VL_FP32 = static_cast<int64_t>(platform::GetVRegSize()) / sizeof(float);

constexpr static AscendC::MicroAPI::CastTrait CAST_B16_TO_B32 = {
    AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN, AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN};

constexpr static AscendC::MicroAPI::CastTrait CAST_FP32_TO_FP16 = {
    AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::NO_SAT, AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT};

template <typename T>
__aicore__ inline void LoadTensorForDtypeT(
    __local_mem__ T* input, AscendC::MicroAPI::RegTensor<float>& dst, AscendC::MicroAPI::MaskReg& preg, uint32_t offset)
{
    if constexpr (IsSameType<T, half>::value) {
        AscendC::MicroAPI::RegTensor<half> xFp16;
        DataCopy<half, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(xFp16, ((__local_mem__ half*)(input) + (offset)));
        Cast<float, half, CAST_B16_TO_B32>(dst, xFp16, preg);
    } else if constexpr (IsSameType<T, bfloat16_t>::value) {
        AscendC::MicroAPI::RegTensor<bfloat16_t> xBf16;
        DataCopy<bfloat16_t, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
            xBf16, ((__local_mem__ bfloat16_t*)(input) + (offset)));
        Cast<float, bfloat16_t, CAST_B16_TO_B32>(dst, xBf16, preg);
    } else {
        DataCopy(dst, ((__local_mem__ float*)(input) + (offset)));
    }
}

template <typename T>
__aicore__ inline void StoreTensorForDtypeTOut(
    __local_mem__ T* dst, AscendC::MicroAPI::RegTensor<float>& src, AscendC::MicroAPI::MaskReg& preg, uint32_t offset)
{
    if constexpr (IsSameType<T, float>::value) {
        DataCopy<T, AscendC::MicroAPI::StoreDist::DIST_NORM>(dst + offset, src, preg);
    } else {
        AscendC::MicroAPI::RegTensor<T> xB16;
        Cast<T, float, CAST_FP32_TO_FP16>(xB16, src, preg);
        DataCopy<T, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(dst + offset, xB16, preg);
    }
}

constexpr static int64_t QUANT_BLOCK_SIZE = 32;
constexpr static int64_t U8_BLOCK_ALIGN_NUM = 32;
constexpr static int64_t CONST_TWO = 2;

template <typename T_QKV>
class FusedKRmsNormRopeStoreKvCacheMxQuantRegbase
{
public:
    __aicore__ inline int64_t CEIL_DIV(int64_t x, int64_t y)
    {
        return (y != 0) ? (x + y - 1) / y : 0;
    }

    __aicore__ inline int64_t CEIL_ALIGN(int64_t x, int64_t y)
    {
        return CEIL_DIV(x, y) * y;
    }

    __aicore__ inline FusedKRmsNormRopeStoreKvCacheMxQuantRegbase(
        TPipe* pipe, const FusedKRmsNormRopeStoreKvCacheMxQuantRegbaseTilingData* tiling)
    {
        tilingData_ = tiling;
        pipe_ = pipe;
    }

    __aicore__ inline void Init(GM_ADDR qkv, GM_ADDR cos, GM_ADDR sin, GM_ADDR gamma, GM_ADDR kv_slot_mapping,
                                GM_ADDR v_scale_slot_mapping, GM_ADDR k_cache, GM_ADDR k_scale_cache, GM_ADDR v_cache,
                                GM_ADDR v_scale_cache, GM_ADDR q, GM_ADDR q_scale)
    {
        quantScaleLastDim_ = tilingData_->headDim / QUANT_BLOCK_SIZE;
        quantScaleLastDimBlockAlign_ = CEIL_ALIGN(quantScaleLastDim_, U8_BLOCK_ALIGN_NUM);
        vScaleCacheActualBs_ = tilingData_->blockSize / QUANT_BLOCK_SIZE / 2;
        allHiddenSize_ = tilingData_->qkvNumHead * tilingData_->headDim;
        qHiddenSize_ = tilingData_->qNumHead * tilingData_->headDim;
        kHiddenSize_ = tilingData_->kNumHead * tilingData_->headDim;
        // v的处理会对NumHead维度做Ub切分
        vUbHiddenSize_ = tilingData_->vNumHeadUbFactor * tilingData_->headDim;
        kCacheBlockOffset_ = tilingData_->kNumHead * tilingData_->blockSize * tilingData_->headDim;
        kScaleCacheBlockOffset_ = kCacheBlockOffset_ / QUANT_BLOCK_SIZE;
        vCacheBlockOffset_ = tilingData_->vNumHead * tilingData_->blockSize * tilingData_->headDim;
        vScaleCacheBlockOffset_ = vCacheBlockOffset_ / QUANT_BLOCK_SIZE;

        // init global memory
        qkvGm_.SetGlobalBuffer((__gm__ T_QKV *)qkv);
        cosGm_.SetGlobalBuffer((__gm__ T_QKV *)cos);
        sinGm_.SetGlobalBuffer((__gm__ T_QKV *)sin);
        gammaGm_.SetGlobalBuffer((__gm__ float *)gamma);
        kvSlotMappingGm_.SetGlobalBuffer((__gm__ int64_t *)kv_slot_mapping);
        vScaleSlotMappingGm_.SetGlobalBuffer((__gm__ int64_t *)v_scale_slot_mapping);
        kCacheGm_.SetGlobalBuffer((__gm__ uint8_t *)k_cache);
        vCacheGm_.SetGlobalBuffer((__gm__ uint8_t *)v_cache);
        kScaleCacheGm_.SetGlobalBuffer((__gm__ uint8_t *)k_scale_cache);
        vScaleCacheGm_.SetGlobalBuffer((__gm__ uint8_t *)v_scale_cache);
        qGm_.SetGlobalBuffer((__gm__ uint8_t *)q);
        qScaleGm_.SetGlobalBuffer((__gm__ uint8_t *)q_scale);

        // init pipe
        pipe_->InitBuffer(inQueueX, 2, this->ubFactor * tilingData_->inUbSize);
        pipe_->InitBuffer(inQueueGamma, 1, tilingData_->headDim * sizeof(float));
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
    __aicore__ inline void DoRmsNorm(const LocalTensor<float> &dstTensor, const LocalTensor<T> &srcTensor,
                                     const LocalTensor<float> &gammaTensor, const int64_t aSize, const int64_t rSize)
    {
        if (aSize <= 0 || rSize <= 0) {
            return;
        }
        if (rSize > CONST_TWO * VL_FP32) {
            return;
        }

        float epsilon = tilingData_->epsilon;
        float reciprocal = tilingData_->reciprocal;
        int64_t stride = rSize;
        uint16_t loopTimes = static_cast<uint16_t>(aSize);

        __local_mem__ float* dst = (__local_mem__ float*)dstTensor.GetPhyAddr();
        __local_mem__ T* x = (__local_mem__ T*)srcTensor.GetPhyAddr();
        __local_mem__ T* x_1 = (__local_mem__ T*)srcTensor.GetPhyAddr() + VL_FP32;
        __local_mem__ float* gamma = (__local_mem__ float*)gammaTensor.GetPhyAddr();
        __local_mem__ float* gamma_1 = (__local_mem__ float*)gammaTensor.GetPhyAddr() + VL_FP32;

        __VEC_SCOPE__
        {
            uint32_t count = static_cast<uint32_t>(rSize - VL_FP32);
            AscendC::MicroAPI::RegTensor<float> reg0, reg1, reg0_1, reg1_1, reg2, reg2_1;
            AscendC::MicroAPI::RegTensor<float> reg3, reg4, reg5, reg6, reg7, reg8, reg9;
            AscendC::MicroAPI::MaskReg pMask = AscendC::MicroAPI::UpdateMask<float>(count);
            AscendC::MicroAPI::MaskReg pFull =
                AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();

            for (uint16_t i = 0; i < loopTimes; ++i) {
                LoadTensorForDtypeT<T>(x, reg0, pFull, i * stride);
                LoadTensorForDtypeT<T>(x_1, reg0_1, pMask, i * stride);
                LoadTensorForDtypeT<float>(gamma, reg1, pFull, 0);
                LoadTensorForDtypeT<float>(gamma_1, reg1_1, pMask, 0);

                AscendC::MicroAPI::Mul(reg2, reg0, reg0, pFull);
                AscendC::MicroAPI::Mul(reg2_1, reg0_1, reg0_1, pMask);
                Add<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(reg2_1, reg2, reg2_1, pMask);
                Copy<float, AscendC::MicroAPI::MaskMergeMode::MERGING>(reg2, reg2_1, pMask);
                ReduceSum(reg2, reg2, pFull);

                AscendC::MicroAPI::Muls(reg3, reg2, reciprocal, pFull);
                AscendC::MicroAPI::Adds(reg4, reg3, epsilon, pFull);
                AscendC::MicroAPI::Sqrt(reg5, reg4, pFull);
                Duplicate(reg5, reg5, pFull);

                AscendC::MicroAPI::Div(reg6, reg0, reg5, pFull);
                AscendC::MicroAPI::Mul(reg7, reg1, reg6, pFull);
                StoreTensorForDtypeTOut<float>(dst, reg7, pFull, i * stride);

                AscendC::MicroAPI::Div(reg8, reg0_1, reg5, pMask);
                AscendC::MicroAPI::Mul(reg9, reg1_1, reg8, pMask);
                StoreTensorForDtypeTOut<float>(dst, reg9, pMask, i * stride + VL_FP32);
            }
        }
    }

    /*
    输入：srcTensor  shape为[S,N,D] D=128
    输入：cosTensor  shape为[S,1,D]
    输入：sinTensor  shape为[S,1,D]
    输出：dstTensor  shape为[S,N,D]
    T1支持float和bfloat16_t
    T2仅支持bfloat16_t
    */
    template <typename T1, T2>
    __aicore__ inline void DoRope(const LocalTensor<T2> &dstTensor, const LocalTensor<T1> &srcTensor,
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
    T仅支持bfloat16_t
    */
    template <typename T>
    __aicore__ inline void DoMxQuant(const LocalTensor<uint8_t> &outTensor, const LocalTensor<uint8_t> &outScaleTensor,
                                     const LocalTensor<T> &srcTensor, const int64_t quantAxis, const int64_t mSize,
                                     const int64_t dSize)
    {
    }

    /*
    gm->ub拷入,提取输入中的q或者k或者v，在ub中紧密排布
    输入：srcGm
    输出：dstUb
    dstUb = srcGm[:lineNum, :copyLineLength]
    */
    template <typename T>
    __aicore__ inline void GatherCopyIn(const LocalTensor<T> &dstUb, const GlobalTensor<T> &srcGm,
                                        const int64_t lineNum, const int64_t srcLineLength,
                                        const int64_t copyLineLength)
    {
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyExtParams copyInParams;
        copyInParams.blockCount = lineNum;
        copyInParams.blockLen = copyLineLength * sizeof(T);
        copyInParams.srcStride = (srcLineLength - copyLineLength) * sizeof(T);
        copyInParams.dstStride = 0;
        DataCopyPad(dstUb, srcGm, copyInParams, padParams);
    }

    /*
    ub->gm拷出
    输入：outTensor
    输出：outGm
    srcLineLength需要32B对齐
    */
    __aicore__ inline void CopyQuantOut(const GlobalTensor<uint8_t> &outGm, const LocalTensor<uint8_t> &outTensor,
                                        const int64_t lineNum, const int64_t srcLineLength,
                                        const int64_t copyLineLength)
    {
        int64_t copyLineLengthBlockAlign = CEIL_ALIGN(copyLineLength, U8_BLOCK_ALIGN_NUM);
        DataCopyExtParams copyOutParams;
        copyOutParams.blockCount = lineNum;
        copyOutParams.blockLen = copyLineLength * sizeof(uint8_t);
        // src在ub上单位为Block
        copyOutParams.srcStride = (srcLineLength - copyLineLengthBlockAlign) / U8_BLOCK_ALIGN_NUM;
        copyOutParams.dstStride = 0;
        DataCopyPad(dstGm, srcUb, copyOutParams);
    }

    __aicore__ inline void ScatterUpdateK(const LocalTensor<uint8_t> &outTensor,
                                          const LocalTensor<uint8_t> &outScaleTensor, const int64_t processSeqLen,
                                          const int64_t startTIdx)
    {
        event_t eventIDSToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
        DataCopyExtParams copyQuantOutParams;
        copyQuantOutParams.blockCount = tilingData_->kNumHead;
        copyQuantOutParams.blockLen = tilingData_->headDim * sizeof(uint8_t);
        copyQuantOutParams.srcStride = 0;
        copyQuantOutParams.dstStride = (tilingData_->blockSize - 1) * tilingData_->headDim * sizeof(uint8_t);

        DataCopyExtParams copyScaleOutParams;
        copyScaleOutParams.blockCount = tilingData_->kNumHead;
        copyScaleOutParams.blockLen = quantScaleLastDim_ * sizeof(uint8_t);
        copyScaleOutParams.srcStride = 0;
        copyScaleOutParams.dstStride = (tilingData_->blockSize - 1) * quantScaleLastDim_ * sizeof(uint8_t);

        for (int64_t i = 0; i < processSeqLen; ++i) {
            int64_t cacheIndex = kvSlotMappingGm_(startTIdx + i);
            // 计算Bn维度上的索引
            int64_t bnIndex = cacheIndex / tilingData_->blockSize;
            // 计算Bs维度上的索引
            int64_t bsIndex = cacheIndex % tilingData_->blockSize;
            SetFlag<HardEvent::S_MTE3>(eventIDSToMTE3);
            WaitFlag<HardEvent::S_MTE3>(eventIDSToMTE3);
            DataCopyPad(kCacheGm_[bnIndex * kCacheBlockOffset_ + bsIndex * tilingData_->headDim],
                        outTensor[i * kHiddenSize_], copyQuantOutParams);
            DataCopyPad(kScaleCacheGm_[bnIndex * kScaleCacheBlockOffset_ + bsIndex * quantScaleLastDim_],
                        outScaleTensor[i * tilingData_->kNumHead * quantScaleLastDimBlockAlign_], copyScaleOutParams);
        }
    }

    __aicore__ inline void ScatterUpdateV(const LocalTensor<uint8_t> &outTensor,
                                          const LocalTensor<uint8_t> &outScaleTensor, const int64_t processSeqLen,
                                          const int64_t startTIdx, const int64_t nIdx)
    {
        event_t eventIDSToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
        DataCopyExtParams copyQuantOutParams;
        copyQuantOutParams.blockCount = tilingData_->vNumHeadUbFactor;
        copyQuantOutParams.blockLen = tilingData_->headDim * sizeof(uint8_t);
        copyQuantOutParams.srcStride = 0;
        copyQuantOutParams.dstStride = (tilingData_->blockSize - 1) * tilingData_->headDim * sizeof(uint8_t);

        for (int64_t i = 0; i < processSeqLen; ++i) {
            int64_t cacheIndex = kvSlotMappingGm_(startTIdx + i);
            // 计算Bn维度上的索引
            int64_t bnIndex = cacheIndex / tilingData_->blockSize;
            // 计算Bs维度上的索引
            int64_t bsIndex = cacheIndex % tilingData_->blockSize;
            SetFlag<HardEvent::S_MTE3>(eventIDSToMTE3);
            WaitFlag<HardEvent::S_MTE3>(eventIDSToMTE3);
            DataCopyPad(vCacheGm_[bnIndex * vCacheBlockOffset_ + nIdx * tilingData_->blockSize * tilingData_->headDim +
                                  bsIndex * tilingData_->headDim],
                        outTensor[i * vUbHiddenSize_], copyQuantOutParams);
        }

        DataCopyExtParams copyScaleOutParams;
        copyScaleOutParams.blockCount = tilingData_->vNumHeadUbFactor;
        copyScaleOutParams.blockLen = tilingData_->headDim * 2 * sizeof(uint8_t);
        copyScaleOutParams.srcStride = 0;
        copyScaleOutParams.dstStride = (vScaleCacheActualBs_ - 1) * tilingData_->headDim * 2 * sizeof(uint8_t);
        // scalar 除 需要优化
        for (int64_t i = 0; i < processSeqLen / QUANT_BLOCK_SIZE / 2; ++i) {
            int64_t cacheIndex = vScaleSlotMappingGm_(startTIdx / QUANT_BLOCK_SIZE / 2 + i);
            // 计算Bn维度上的索引
            int64_t bnIndex = cacheIndex / vScaleCacheActualBs_;
            // 计算Bs维度上的索引
            int64_t bsIndex = cacheIndex % vScaleCacheActualBs_;
            DataCopyPad(vScaleCacheGm_[bnIndex * vScaleCacheBlockOffset_ +
                                       nIdx * vScaleCacheActualBs_ * tilingData_->headDim * 2 +
                                       bsIndex * tilingData_->headDim * 2],
                        outScaleTensor[i * vUbHiddenSize_ * 2], copyScaleOutParams);
        }

        __aicore__ inline void DoPhase1()
        {
            if (GetBlockIdx() >= tilingData_->qUsedCoreNum) {
                return;
            }
            in64_t startTIndex = GetBlockIdx() * tilingData_->qBlockFactor;
            in64_t endTIndex = startTIndex + tilingData_->qBlockFactor;
            // 尾核场景
            if (endTIndex > tilingData_->seqLengthSum) {
                endTIndex = tilingData_->seqLengthSum;
            }
            for (int64_t tIdx = startTIndex; tIdx < endTIndex; tIdx += tilingData_->qUbFactor) {
                int64_t processSeqLen = tilingData_->qUbFactor;
                // 尾块场景
                if (tIdx + processSeqLen > endTIndex) {
                    processSeqLen = endTIndex - tIdx;
                }
                LocalTensor<T_QKV> qTensor = inQueue.AllocTensor<T_QKV>();
                LocalTensor<T_QKV> cosTensor = qTensor[tilingData_->qUbFactor * qHiddenSize_];
                LocalTensor<T_QKV> sinTensor = cosTensor[tilingData_->qUbFactor * tilingData_->headDim];
                GatherCopyIn(qTensor, qkvGm_[tIdx * allHiddenSize_], processSeqLen,
                             allHiddenSize_, qHiddenSize_);
                GatherCopyIn(cosTensor, cosGm_[tIdx * tilingData_->headDim], processSeqLen, tilingData_->headDim,
                             tilingData_->headDim);
                GatherCopyIn(sinTensor, sinGm_[tIdx * tilingData_->headDim], processSeqLen, tilingData_->headDim,
                             tilingData_->headDim);
                inQueue.EnQue(qTensor);
                qTensor = inQueue.DeQue<T_QKV>();
                LocalTensor<T_QKV> ropeTensor = wsBuffer0.Get<T_QKV>();
                DoRope<T_QKV, T_QKV>(ropeTensor, qTensor, cosTensor, sinTensor, processSeqLen, tilingData_->qNumHead,
                                     tilingData_->headDim);
                inQueue.FreeTensor(qTensor);
                LocalTensor<uint8_t> outTensor = outQueue.AllocTensor<uint8_t>();
                LocalTensor<uint8_t> outScaleTensor = outTensor[tilingData_->qUbFactor * qHiddenSize_];
                DoMxQuant(outTensor, outScaleTensor, ropeTensor, 1, processSeqLen * tilingData_->qNumHead,
                          tilingData_->headDim);
                outQueue.EnQue(outTensor);
                outTensor = outQueue.DeQue<uint8_t>();
                CopyQuantOut(qGm_[tIdx * qHiddenSize_], outTensor, 1, processSeqLen * qHiddenSize_,
                             processSeqLen * qHiddenSize_);
                CopyQuantOut(qScaleGm_[tIdx * tilingData_->qNumHead * quantScaleLastDim_], outScaleTensor,
                             processSeqLen * tilingData_->qNumHead, quantScaleLastDimBlockAlign_, quantScaleLastDim_);
                outQueue.FreeTensor(outTensor);
            }
        }

        __aicore__ inline void DoPhase2()
        {
            if (GetBlockIdx() >= tilingData_->kUsedCoreNum) {
                return;
            }
            // CopyIn gamma
            LocalTensor<float> gammaTensor = inQueueGamma.AllocTensor<float>();
            GatherCopyIn(gammaLocal, gammaGm_, 1, tilingData_->headDim, tilingData_->headDim);
            inQueueGamma.EnQue(gammaTensor);
            gammaTensor = inQueueGamma.DeQue<float>();
            in64_t startTIndex = GetBlockIdx() * tilingData_->kBlockFactor;
            in64_t endTIndex = startTIndex + tilingData_->kBlockFactor;
            // 尾核场景
            if (endTIndex > tilingData_->seqLengthSum) {
                endTIndex = tilingData_->seqLengthSum;
            }
            for (int64_t tIdx = startTIndex; tIdx < endTIndex; tIdx += tilingData_->kUbFactor) {
                int64_t processSeqLen = tilingData_->kUbFactor;
                if (tIdx + processSeqLen > endTIndex) {
                    processSeqLen = endTIndex - tIdx;
                }
                LocalTensor<T_QKV> kTensor = inQueue.AllocTensor<T_QKV>();
                LocalTensor<T_QKV> cosTensor = kTensor[tilingData_->kUbFactor * kHiddenSize_];
                LocalTensor<T_QKV> sinTensor = cosTensor[tilingData_->kUbFactor * tilingData_->headDim];
                GatherCopyIn(kTensor, qkvGm_[tIdx * allHiddenSize_ + qHiddenSize_],
                             processSeqLen, allHiddenSize_, kHiddenSize_);
                GatherCopyIn(cosTensor, cosGm_[tIdx * tilingData_->headDim], processSeqLen, tilingData_->headDim,
                             tilingData_->headDim);
                GatherCopyIn(sinTensor, sinGm_[tIdx * tilingData_->headDim], processSeqLen, tilingData_->headDim,
                             tilingData_->headDim);
                inQueue.EnQue(kTensor);
                kTensor = inQueue.DeQue<T_QKV>();
                LocalTensor<float> rmsNormTensor = wsBuffer0.Get<float>();
                DoRmsNorm<T_QKV>(rmsNormTensor, kTensor, gammaTensor, processSeqLen * tilingData_->kNumHead,
                                 tilingData_->headDim);
                LocalTensor<T_QKV> ropeTensor = wsBuffer1.Get<T_QKV>();
                DoRope<float, T_QKV>(ropeTensor, rmsNormTensor, cosTensor, sinTensor, processSeqLen,
                                     tilingData_->kNumHead, tilingData_->headDim);
                inQueue.FreeTensor(kTensor);
                LocalTensor<uint8_t> outTensor = outQueue.AllocTensor<uint8_t>();
                LocalTensor<uint8_t> outScaleTensor = outTensor[tilingData_->kUbFactor * kHiddenSize_];
                DoMxQuant(outTensor, outScaleTensor, ropeTensor, 1, processSeqLen * tilingData_->kNumHead,
                          tilingData_->headDim);
                outQueue.EnQue(outTensor);
                outTensor = outQueue.DeQue<uint8_t>();
                ScatterUpdateK(outTensor, outScaleTensor, tIdx);
                outQueue.FreeTensor(outTensor);
            }
            inQueueGamma.FreeTensor(gammaLocal);
        }

        __aicore__ inline void DoPhase3()
        {
            if (GetBlockIdx() >= tilingData_->vUsedCoreNum) {
                return;
            }
            in64_t startTIndex = GetBlockIdx() * tilingData_->vBlockFactor;
            in64_t endTIndex = startTIndex + tilingData_->vBlockFactor;
            // 尾核场景
            if (endTIndex > tilingData_->seqLengthSum) {
                endTIndex = tilingData_->seqLengthSum;
            }
            for (int64_t tIdx = startTIndex; tIdx < endTIndex; tIdx += tilingData_->vBlockFactor) {
                int64_t processSeqLen = tilingData_->vBlockFactor;
                if (tIdx + processSeqLen > endTIndex) {
                    processSeqLen = endTIndex - tIdx;
                }
                for (int64_t nIdx = 0; nIdx < tilingData_->vNumHead; nIdx += tilingData_->vNumHeadUbFactor) {
                    LocalTensor<T_QKV> vTensor = inQueue.AllocTensor<T_QKV>();
                    GatherCopyIn(vTensor,
                                qkvGm_[tIdx * allHiddenSize_ +
                                        (tilingData_->qNumHead + tilingData_->kNumHead + nIdx) * tilingData_->headDim],
                                processSeqLen, allHiddenSize_, vUbHiddenSize_);
                    inQueue.EnQue(vTensor);
                    vTensor = inQueue.DeQue<T_QKV>();
                    LocalTensor<uint8_t> outTensor = outQueue.AllocTensor<uint8_t>();
                    LocalTensor<uint8_t> outScaleTensor = outTensor[tilingData_->vUbFactor * vUbHiddenSize_];
                    DoMxQuant(outTensor, outScaleTensor, vTensor, 0, processSeqLen * tilingData_->vNumHeadUbFactor,
                            tilingData_->headDim);
                    inQueue.FreeTensor(vTensor);
                    outQueue.EnQue(outTensor);
                    outTensor = outQueue.DeQue<uint8_t>();
                    ScatterUpdateV(outTensor, outScaleTensor, tIdx, nIdx);
                    outQueue.FreeTensor(outTensor);
                }
            }
        }

        __aicore__ inline void Process()
        {
            // 对q做rope，mxquant
            DoPhase1();
            // 对k做rms_norm，rope，mxquant，再scatter到cache中
            DoPhase2();
            // 对v做再scatter到cache中
            DoPhase3();
        }

    private:
        const FusedKRmsNormRopeStoreKvCacheMxQuantRegbaseTilingData *tilingData_;
        TQue<QuePosition::VECIN, 1> inQueue, inQueueGamma;
        TQue<QuePosition::VECOUT, 1> outQueue;
        TBuf<TPosition::VECCALC> wsBuffer0, wsBuffer1;
        TPipe *pipe_ = nullptr;

        GlobalTensor<T_QKV> qkvGm_, cosGm_, sinGm_;
        GlobalTensor<float> gammaGm_;
        GlobalTensor<int64_t> kvSlotMappingGm_, vScaleSlotMappingGm_;
        GlobalTensor<uint8_t> kCacheGm_, vCacheGm_, kScaleCacheGm_, vScaleCacheGm_, qGm_, qScaleGm_;

        int64_t quantScaleLastDim_ = 0;
        int64_t quantScaleLastDimBlockAlign_ = 0;
        int64_t vScaleCacheActualBs_ = 0;
        int64_t allHiddenSize_ = 0;
        int64_t qHiddenSize_ = 0;
        int64_t kHiddenSize_ = 0;
        int64_t vUbHiddenSize_ = 0;
        int64_t kCacheBlockOffset_ = 0;
        int64_t kScaleCacheBlockOffset_ = 0;
        int64_t vCacheBlockOffset_ = 0;
        int64_t vScaleCacheBlockOffset_ = 0;
    };
};
}// namespace FusedKRmsNormRopeStoreKvCacheMxQuant
#endif // FUSED_K_RMS_NORM_ROPE_STORE_KV_CACHE_MX_QUANT_REGBASE_H_