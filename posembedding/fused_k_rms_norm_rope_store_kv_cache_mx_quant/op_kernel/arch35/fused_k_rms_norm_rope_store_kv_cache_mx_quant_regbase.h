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

#define FLOAT_OVERFLOW_MODE_CTRL 60

#include "kernel_operator.h"
#include "op_kernel/platform_util.h"

namespace FusedKRmsNormRopeStoreKvCacheMxQuant {
using namespace AscendC;

constexpr static AscendC::MicroAPI::CastTrait CAST_B16_TO_B32 = {
    AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN, AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN};

constexpr static AscendC::MicroAPI::CastTrait CAST_FP32_TO_FP16 = {
    AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::NO_SAT, AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT};

template <typename T>
__aicore__ inline void LoadTensorForDtypeT(__local_mem__ T *input, AscendC::MicroAPI::RegTensor<float> &dst,
                                           AscendC::MicroAPI::MaskReg &preg, uint32_t offset)
{
    if constexpr (IsSameType<T, half>::value) {
        AscendC::MicroAPI::RegTensor<half> xFp16;
        DataCopy<half, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(xFp16, ((__local_mem__ half *)(input) + (offset)));
        Cast<float, half, CAST_B16_TO_B32>(dst, xFp16, preg);
    } else if constexpr (IsSameType<T, bfloat16_t>::value) {
        AscendC::MicroAPI::RegTensor<bfloat16_t> xBf16;
        DataCopy<bfloat16_t, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
            xBf16, ((__local_mem__ bfloat16_t *)(input) + (offset)));
        Cast<float, bfloat16_t, CAST_B16_TO_B32>(dst, xBf16, preg);
    } else {
        DataCopy(dst, ((__local_mem__ float *)(input) + (offset)));
    }
}

template <typename T>
__aicore__ inline void StoreTensorForDtypeTOut(__local_mem__ T *dst, AscendC::MicroAPI::RegTensor<float> &src,
                                               AscendC::MicroAPI::MaskReg &preg, uint32_t offset)
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
constexpr static uint32_t DIGIT_TWO = 2;
constexpr static uint32_t VL_B16 = Ops::Base::GetVRegSize() / sizeof(bfloat16_t);
constexpr static uint32_t HALF_VL_B16 = VL_B16 / DIGIT_TWO;
constexpr static uint32_t VL_FP32 = Ops::Base::GetVRegSize() / sizeof(float);
constexpr static int64_t COS_SIN_NUM = 2;
constexpr static int64_t DOUBLE_BUFFER = 2;
constexpr static uint16_t MAX_EXP_FOR_BF16 = 0x7f80;

constexpr static uint16_t BF16_EXP_BIAS = 0x7f00;
constexpr static uint16_t FP8_E4M3_MAX_EXP = 0x0400;
constexpr static uint16_t NAN_CUSTOMIZATION = 0x7f81;
constexpr static int16_t SHR_NUM_FOR_BF16 = 7;
constexpr static uint16_t MAX_EXP_FOR_FP8 = 0x00ff;
constexpr static uint16_t SPECIAL_EXP_THRESHOLD = 0x0040;
constexpr static uint32_t OUT_ALL = 256;
constexpr static uint16_t BLOCK_REDUCE_NUMS = Ops::Base::GetVRegSize() / Ops::Base::GetUbBlockSize();

template <typename T_QKV, typename T_OUT>
class FusedKRmsNormRopeStoreKvCacheMxQuantRegbase {
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
        TPipe *pipe, const FusedKRmsNormRopeStoreKvCacheMxQuantTilingData *tiling)
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
        vScaleCacheActualBs_ = tilingData_->blockSize / QUANT_BLOCK_SIZE / DIGIT_TWO;
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
        if (rSize > DIGIT_TWO * VL_FP32) {
            return;
        }

        float epsilon = tilingData_->epsilon;
        float reciprocal = tilingData_->reciprocal;
        int64_t stride = rSize;
        uint16_t loopTimes = static_cast<uint16_t>(aSize);

        __local_mem__ float *dst = (__local_mem__ float *)dstTensor.GetPhyAddr();
        __local_mem__ T *x = (__local_mem__ T *)srcTensor.GetPhyAddr();
        __local_mem__ T *x_1 = (__local_mem__ T *)srcTensor.GetPhyAddr() + VL_FP32;
        __local_mem__ float *gamma = (__local_mem__ float *)gammaTensor.GetPhyAddr();
        __local_mem__ float *gamma_1 = (__local_mem__ float *)gammaTensor.GetPhyAddr() + VL_FP32;

        __VEC_SCOPE__
        {
            uint32_t count = static_cast<uint32_t>(rSize - VL_FP32);
            uint32_t strideU32 = static_cast<uint32_t>(stride);
            AscendC::MicroAPI::RegTensor<float> reg0, reg1, reg0_1, reg1_1, reg2, reg2_1;
            AscendC::MicroAPI::RegTensor<float> reg3, reg4, reg5, reg6, reg7, reg8, reg9;
            AscendC::MicroAPI::MaskReg pMask = AscendC::MicroAPI::UpdateMask<float>(count);
            AscendC::MicroAPI::MaskReg pFull =
                AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();

            for (uint16_t i = 0; i < loopTimes; ++i) {
                LoadTensorForDtypeT<T>(x, reg0, pFull, i * strideU32);
                LoadTensorForDtypeT<T>(x_1, reg0_1, pMask, i * strideU32);
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
                StoreTensorForDtypeTOut<float>(dst, reg7, pFull, i * strideU32);

                AscendC::MicroAPI::Div(reg8, reg0_1, reg5, pMask);
                AscendC::MicroAPI::Mul(reg9, reg1_1, reg8, pMask);
                StoreTensorForDtypeTOut<float>(dst, reg9, pMask, i * strideU32 + VL_FP32);
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
    template <typename T1, typename T2>
    __aicore__ inline void DoRope(const LocalTensor<T2> &dstTensor, const LocalTensor<T1> &srcTensor,
                                  const LocalTensor<T2> &cosTensor, const LocalTensor<T2> &sinTensor,
                                  const int64_t seqSize, const int64_t nSize, const int64_t dSize)
    {
        if (seqSize <= 0 || nSize <= 0 || dSize <= 0) {
            return;
        }

        __local_mem__ T1 *srcUb = (__local_mem__ T1 *)srcTensor.GetPhyAddr();
        __local_mem__ T2 *cosUb = (__local_mem__ T2 *)cosTensor.GetPhyAddr();
        __local_mem__ T2 *sinUb = (__local_mem__ T2 *)sinTensor.GetPhyAddr();
        __local_mem__ T2 *dstUb = (__local_mem__ T2 *)dstTensor.GetPhyAddr();

        uint32_t dHalfOffset = dSize / DIGIT_TWO;

        __VEC_SCOPE__
        {
            uint32_t dSizeU32 = static_cast<uint32_t>(dSize);
            uint32_t nSizeU32 = static_cast<uint32_t>(nSize);
            AscendC::MicroAPI::RegTensor<float> inPart1Reg;
            AscendC::MicroAPI::RegTensor<float> inPart2Reg;
            AscendC::MicroAPI::RegTensor<float> cosPart1Reg;
            AscendC::MicroAPI::RegTensor<float> cosPart2Reg;
            AscendC::MicroAPI::RegTensor<float> sinPart1Reg;
            AscendC::MicroAPI::RegTensor<float> sinPart2Reg;
            AscendC::MicroAPI::RegTensor<float> tmp1Reg;
            AscendC::MicroAPI::RegTensor<float> tmp2Reg;
            AscendC::MicroAPI::RegTensor<float> tmp3Reg;
            AscendC::MicroAPI::RegTensor<float> tmp4Reg;
            AscendC::MicroAPI::MaskReg pFull =
                AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
            __local_mem__ T1 *currSrcUb;
            __local_mem__ T2 *currDstUb;
            __local_mem__ T2 *currCosUb;
            __local_mem__ T2 *currSinUb;

            for (uint16_t sIdx = 0; sIdx < static_cast<uint16_t>(seqSize); sIdx++) {
                // 对每个序列位置sIdx，cos和sin只需要加载一次
                currCosUb = cosUb + sIdx * dSizeU32;
                currSinUb = sinUb + sIdx * dSizeU32;
                LoadTensorForDtypeT<T2>(currCosUb, cosPart1Reg, pFull, 0);           // cos[0:D/2]
                LoadTensorForDtypeT<T2>(currCosUb, cosPart2Reg, pFull, dHalfOffset); // cos[D/2:D]

                LoadTensorForDtypeT<T2>(currSinUb, sinPart1Reg, pFull, 0);           // sin[0:D/2]
                LoadTensorForDtypeT<T2>(currSinUb, sinPart2Reg, pFull, dHalfOffset); // sin[D/2:D]

                for (uint16_t nIdx = 0; nIdx < static_cast<uint16_t>(nSize); nIdx++) {
                    // src需要对每个nIdx都加载
                    currSrcUb = srcUb + (sIdx * nSizeU32 + nIdx) * dSizeU32;
                    currDstUb = dstUb + (sIdx * nSizeU32 + nIdx) * dSizeU32;

                    LoadTensorForDtypeT<T1>(currSrcUb, inPart1Reg, pFull, 0);           // in[0:D/2]
                    LoadTensorForDtypeT<T1>(currSrcUb, inPart2Reg, pFull, dHalfOffset); // in[D/2:D]

                    // RoPE计算
                    // out[0:D/2] = in[0:D/2] * cos[0:D/2] - in[D/2:D] * sin[0:D/2]
                    Mul(tmp1Reg, inPart1Reg, cosPart1Reg, pFull); // temp1 = in[0:D/2] * cos[0:D/2]
                    Mul(tmp2Reg, inPart2Reg, sinPart1Reg, pFull); // temp2 = in[D/2:D] * sin[0:D/2]
                    Sub(tmp1Reg, tmp1Reg, tmp2Reg, pFull);        // out[0:D/2] = temp1 - temp2

                    // out[D/2:D] = in[D/2:D] * cos[D/2:D] + in[0:D/2] * sin[D/2:D]
                    Mul(tmp3Reg, inPart2Reg, cosPart2Reg, pFull); // temp3 = in[D/2:D] * cos[D/2:D]
                    Mul(tmp4Reg, inPart1Reg, sinPart2Reg, pFull); // temp4 = in[0:D/2] * sin[D/2:D]
                    Add(tmp3Reg, tmp3Reg, tmp4Reg, pFull);        // out[D/2:D] = temp3 + temp4

                    // 存储结果
                    StoreTensorForDtypeTOut<T2>(currDstUb, tmp1Reg, pFull, 0);           // 存储前半部分
                    StoreTensorForDtypeTOut<T2>(currDstUb, tmp3Reg, pFull, dHalfOffset); // 存储后半部分
                }
            }
        }
    }

    /*
    输入：srcTensor  shape为[M,D]
    输入：tmpTensor 存储中间结果的临时ub空间
    输出：outTensor  shape为[M,D] 量化后的fp8数据
    输出：outScaleTensor 量化后的scale，数据类型为fp8
    quantAxis支持0或者1，量化的BlockSize固定为32
    当quantAxis为1时 outScale的shape为[M, D/32 = 4 需要pad到32，需要block对齐后续scatter] D是128
    当quantAxis为0时 outScale的shape为[M/32/2,D,2] M为64的整数倍, D是128的整数倍
    T仅支持bfloat16_t
    */
    template <typename T>
    __aicore__ inline void DoMxQuant(const LocalTensor<uint8_t> &outTensor, const LocalTensor<uint8_t> &outScaleTensor,
                                     const LocalTensor<T> &srcTensor, const LocalTensor<T> &tmpTensor,
                                     const int64_t quantAxis, const int64_t mSize, const int64_t dSize)
    {
        if (quantAxis == 1) {
            DoMxQuantAxis1(outTensor, outScaleTensor, srcTensor, tmpTensor, mSize, dSize);
        } else if (quantAxis == 0) {
            DoMxQuantAxis0(outTensor, outScaleTensor, srcTensor, tmpTensor, mSize, dSize);
        } else {
            return;
        }
    }

    template <typename T>
    __aicore__ inline void CalcAxis1MaxExpOcpVF(__ubuf__ uint16_t *maxExpAddr, __ubuf__ T *srcAddr, uint32_t allNum,
                                                uint16_t loopNum)
    {
        __VEC_SCOPE__
        {
            AscendC::MicroAPI::RegTensor<T> vdExp0;
            AscendC::MicroAPI::RegTensor<T> vdExp1;
            AscendC::MicroAPI::RegTensor<bfloat16_t> vdExp0BF16;
            AscendC::MicroAPI::RegTensor<bfloat16_t> vdExp1BF16;
            AscendC::MicroAPI::RegTensor<uint16_t> vdExpSelect0;
            AscendC::MicroAPI::RegTensor<uint16_t> vdExpSelect1;
            AscendC::MicroAPI::RegTensor<uint16_t> vdExpExtract0;
            AscendC::MicroAPI::RegTensor<uint16_t> vdExpExtract1;

            AscendC::MicroAPI::RegTensor<uint16_t> expMaskBF16;
            AscendC::MicroAPI::Duplicate(expMaskBF16, MAX_EXP_FOR_BF16);

            AscendC::MicroAPI::RegTensor<uint16_t> vdMaxExp;
            AscendC::MicroAPI::MaskReg scaleMask1;
            AscendC::MicroAPI::MaskReg scaleMask2;
            AscendC::MicroAPI::UnalignReg u1;
            for (uint16_t i = 0; i < loopNum; i++) {
                scaleMask1 = AscendC::MicroAPI::UpdateMask<T>(allNum);
                // 一次loop处理2*VL_B16的数据，需要UpdateMask两次
                scaleMask2 = AscendC::MicroAPI::UpdateMask<T>(allNum);
                AscendC::MicroAPI::DataCopy<T, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                            AscendC::MicroAPI::LoadDist::DIST_DINTLV_B16>(vdExp0, vdExp1, srcAddr,
                                                                                          VL_B16 * DIGIT_TWO);
                AscendC::MicroAPI::And(vdExpExtract0, (AscendC::MicroAPI::RegTensor<uint16_t> &)vdExp0, expMaskBF16,
                                       scaleMask1);
                AscendC::MicroAPI::And(vdExpExtract1, (AscendC::MicroAPI::RegTensor<uint16_t> &)vdExp1, expMaskBF16,
                                       scaleMask1);
                AscendC::MicroAPI::Max(vdMaxExp, vdExpExtract0, vdExpExtract1, scaleMask1);
                AscendC::MicroAPI::ReduceMaxWithDataBlock(vdMaxExp, vdMaxExp, scaleMask1);
                AscendC::MicroAPI::DataCopyUnAlign<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    maxExpAddr, vdMaxExp, u1, BLOCK_REDUCE_NUMS);
            }
            AscendC::MicroAPI::DataCopyUnAlignPost(maxExpAddr, u1, 0);
        }
    }

    __aicore__ inline void CalcAxis1ScaleOcpVF(__ubuf__ uint16_t *mxScaleAddr, __ubuf__ uint16_t *halfScaleAddr,
                                               __ubuf__ uint16_t *maxExpAddr, uint32_t allScaleNum, uint16_t loopNum)
    {
        
        __VEC_SCOPE__
        {
            AscendC::MicroAPI::RegTensor<uint16_t> expMask;
            AscendC::MicroAPI::Duplicate(expMask, MAX_EXP_FOR_BF16);
            AscendC::MicroAPI::RegTensor<uint16_t> vdMaxExp;
            AscendC::MicroAPI::MaskReg cmpResult;
            AscendC::MicroAPI::MaskReg zeroMask;
            AscendC::MicroAPI::MaskReg cmpResultSub;
            AscendC::MicroAPI::MaskReg preMaskScale;
            AscendC::MicroAPI::RegTensor<uint16_t> maxExpValue;
            AscendC::MicroAPI::Duplicate(maxExpValue, FP8_E4M3_MAX_EXP);
            AscendC::MicroAPI::RegTensor<uint16_t> sharedExp;
            AscendC::MicroAPI::RegTensor<uint16_t> scaleValue;
            AscendC::MicroAPI::RegTensor<uint16_t> scaleBias;
            AscendC::MicroAPI::Duplicate(scaleBias, BF16_EXP_BIAS);
            AscendC::MicroAPI::RegTensor<uint16_t> halfScale;
            AscendC::MicroAPI::RegTensor<uint16_t> fp8NanRegTensor;
            AscendC::MicroAPI::Duplicate(fp8NanRegTensor, MAX_EXP_FOR_FP8);
            AscendC::MicroAPI::RegTensor<uint16_t> zeroRegTensor;
            AscendC::MicroAPI::Duplicate(zeroRegTensor, 0);
            AscendC::MicroAPI::RegTensor<uint16_t> nanRegTensor;
            AscendC::MicroAPI::Duplicate(nanRegTensor, NAN_CUSTOMIZATION);
            AscendC::MicroAPI::MaskReg invalidDataMask;
            AscendC::MicroAPI::MaskReg specialDataMask;
            AscendC::MicroAPI::RegTensor<uint16_t> specialExpRegTensor;
            AscendC::MicroAPI::Duplicate(specialExpRegTensor, SPECIAL_EXP_THRESHOLD);
            for (uint16_t i = 0; i < loopNum; i++) {
                preMaskScale = AscendC::MicroAPI::UpdateMask<uint16_t>(allScaleNum);
                AscendC::MicroAPI::DataCopy<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    vdMaxExp, maxExpAddr, VL_B16);
                AscendC::MicroAPI::Compare<uint16_t, CMPMODE::NE>(cmpResult, vdMaxExp, expMask,
                                                                  preMaskScale); // INF/NAN
                AscendC::MicroAPI::Compare<uint16_t, CMPMODE::NE>(zeroMask, vdMaxExp, zeroRegTensor, preMaskScale);
                AscendC::MicroAPI::Compare<uint16_t, CMPMODE::LE>(invalidDataMask, vdMaxExp, maxExpValue, preMaskScale);

                AscendC::MicroAPI::Select<uint16_t>(vdMaxExp, maxExpValue, vdMaxExp, invalidDataMask);

                AscendC::MicroAPI::Sub(sharedExp, vdMaxExp, maxExpValue, preMaskScale);
                AscendC::MicroAPI::ShiftRights(scaleValue, sharedExp, SHR_NUM_FOR_BF16, preMaskScale);

                AscendC::MicroAPI::Select<uint16_t>(scaleValue, scaleValue, fp8NanRegTensor, cmpResult);
                AscendC::MicroAPI::Select<uint16_t>(scaleValue, scaleValue, zeroRegTensor, zeroMask);

                AscendC::MicroAPI::DataCopy<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                            AscendC::MicroAPI::StoreDist::DIST_PACK_B16>(
                    mxScaleAddr, scaleValue, HALF_VL_B16, preMaskScale);

                AscendC::MicroAPI::Compare<uint16_t, CMPMODE::EQ>(specialDataMask, sharedExp, scaleBias, preMaskScale);
                AscendC::MicroAPI::Sub(halfScale, scaleBias, sharedExp, preMaskScale);
                AscendC::MicroAPI::Select<uint16_t>(halfScale, halfScale, nanRegTensor, cmpResult);
                AscendC::MicroAPI::Select<uint16_t>(halfScale, halfScale, zeroRegTensor, zeroMask);
                AscendC::MicroAPI::Select<uint16_t>(halfScale, specialExpRegTensor, halfScale, specialDataMask);

                AscendC::MicroAPI::DataCopy<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    halfScaleAddr, halfScale, VL_B16, preMaskScale);
            }
        }
    }

    template <typename T, typename U>
    __aicore__ inline void CalcAxis1QuantOutVF(__ubuf__ uint8_t *quantOutAddr, __ubuf__ T *srcAddr,
                                               __ubuf__ uint16_t *halfScaleAddr, uint16_t loopNum)
    {
        __VEC_SCOPE__
        {
            AscendC::MicroAPI::MaskReg dataMask1;
            AscendC::MicroAPI::MaskReg dataMask2;
            AscendC::MicroAPI::MaskReg dataMask3;
            AscendC::MicroAPI::MaskReg dataMask4;
            AscendC::MicroAPI::MaskReg dataMask5;
            AscendC::MicroAPI::RegTensor<uint16_t> halfScaleForMul;
            AscendC::MicroAPI::RegTensor<T> vdExp0;
            AscendC::MicroAPI::RegTensor<T> vdExp1;
            AscendC::MicroAPI::RegTensor<float> vdExp0FP32Zero;
            AscendC::MicroAPI::RegTensor<float> vdExp0FP32One;
            AscendC::MicroAPI::RegTensor<float> vdExp1FP32Zero;
            AscendC::MicroAPI::RegTensor<float> vdExp1FP32One;
            AscendC::MicroAPI::RegTensor<U> vdExp0FP8Zero;
            AscendC::MicroAPI::RegTensor<U> vdExp0FP8One;
            AscendC::MicroAPI::RegTensor<U> vdExp1FP8Zero;
            AscendC::MicroAPI::RegTensor<U> vdExp1FP8One;
            static constexpr AscendC::MicroAPI::CastTrait castTraitZero = {
                AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN,
                AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
            static constexpr AscendC::MicroAPI::CastTrait castTraitOne = {
                AscendC::MicroAPI::RegLayout::ONE, AscendC::MicroAPI::SatMode::UNKNOWN,
                AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
            static constexpr AscendC::MicroAPI::CastTrait castTrait32to80 = {
                AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::SAT,
                AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
            static constexpr AscendC::MicroAPI::CastTrait castTrait32to81 = {
                AscendC::MicroAPI::RegLayout::ONE, AscendC::MicroAPI::SatMode::SAT,
                AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
            static constexpr AscendC::MicroAPI::CastTrait castTrait32to82 = {
                AscendC::MicroAPI::RegLayout::TWO, AscendC::MicroAPI::SatMode::SAT,
                AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
            static constexpr AscendC::MicroAPI::CastTrait castTrait32to83 = {
                AscendC::MicroAPI::RegLayout::THREE, AscendC::MicroAPI::SatMode::SAT,
                AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
            dataMask1 = AscendC::MicroAPI::CreateMask<T>();
            dataMask2 = AscendC::MicroAPI::CreateMask<T>();
            dataMask3 = AscendC::MicroAPI::CreateMask<T>();
            dataMask4 = AscendC::MicroAPI::CreateMask<T>();
            dataMask5 = AscendC::MicroAPI::CreateMask<U>();
            for (uint16_t i = 0; i < loopNum; i++) {
                AscendC::MicroAPI::DataCopy<T, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                            AscendC::MicroAPI::LoadDist::DIST_DINTLV_B16>(vdExp0, vdExp1, srcAddr,
                                                                                          VL_B16 * DIGIT_TWO);
                AscendC::MicroAPI::DataCopy<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                            AscendC::MicroAPI::LoadDist::DIST_E2B_B16>(halfScaleForMul, halfScaleAddr,
                                                                                       BLOCK_REDUCE_NUMS);

                AscendC::MicroAPI::Mul(vdExp0, vdExp0, (AscendC::MicroAPI::RegTensor<T> &)halfScaleForMul, dataMask1);
                AscendC::MicroAPI::Mul(vdExp1, vdExp1, (AscendC::MicroAPI::RegTensor<T> &)halfScaleForMul, dataMask1);

                AscendC::MicroAPI::Cast<float, T, castTraitZero>(vdExp0FP32Zero, vdExp0, dataMask1);
                AscendC::MicroAPI::Cast<float, T, castTraitOne>(vdExp0FP32One, vdExp0, dataMask1);
                AscendC::MicroAPI::Cast<float, T, castTraitZero>(vdExp1FP32Zero, vdExp1, dataMask2);
                AscendC::MicroAPI::Cast<float, T, castTraitOne>(vdExp1FP32One, vdExp1, dataMask2);

                AscendC::MicroAPI::Cast<U, float, castTrait32to80>(vdExp0FP8Zero, vdExp0FP32Zero, dataMask3);
                AscendC::MicroAPI::Cast<U, float, castTrait32to82>(vdExp0FP8One, vdExp0FP32One, dataMask3);
                AscendC::MicroAPI::Cast<U, float, castTrait32to81>(vdExp1FP8Zero, vdExp1FP32Zero, dataMask4);
                AscendC::MicroAPI::Cast<U, float, castTrait32to83>(vdExp1FP8One, vdExp1FP32One, dataMask4);

                AscendC::MicroAPI::Add((AscendC::MicroAPI::RegTensor<uint8_t> &)vdExp0FP8Zero,
                                       (AscendC::MicroAPI::RegTensor<uint8_t> &)vdExp0FP8Zero,
                                       (AscendC::MicroAPI::RegTensor<uint8_t> &)vdExp0FP8One, dataMask5);
                AscendC::MicroAPI::Add((AscendC::MicroAPI::RegTensor<uint8_t> &)vdExp0FP8Zero,
                                       (AscendC::MicroAPI::RegTensor<uint8_t> &)vdExp0FP8Zero,
                                       (AscendC::MicroAPI::RegTensor<uint8_t> &)vdExp1FP8Zero, dataMask5);
                AscendC::MicroAPI::Add((AscendC::MicroAPI::RegTensor<uint8_t> &)vdExp0FP8Zero,
                                       (AscendC::MicroAPI::RegTensor<uint8_t> &)vdExp0FP8Zero,
                                       (AscendC::MicroAPI::RegTensor<uint8_t> &)vdExp1FP8One, dataMask5);

                AscendC::MicroAPI::DataCopy<uint8_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                            AscendC::MicroAPI::StoreDist::DIST_NORM_B8>(
                    quantOutAddr, (AscendC::MicroAPI::RegTensor<uint8_t> &)vdExp0FP8Zero, OUT_ALL,
                    dataMask5);
            }
        }
    }

    template <typename T>
    __aicore__ inline void PadMxScaleBlockAlign(__ubuf__ uint8_t *mxScaleBlockAlignAddr, __ubuf__ uint8_t *mxScaleAddr,
                                                uint16_t loopNum)
    {
        __VEC_SCOPE__
        {
            uint32_t quantScaleLastDimU32 = static_cast<uint32_t>(quantScaleLastDim_);
            uint32_t quantScaleLastDimBlockAlignU32 = static_cast<uint32_t>(quantScaleLastDimBlockAlign_);
            AscendC::MicroAPI::UnalignReg uIn;
            AscendC::MicroAPI::UnalignReg uOut;
            AscendC::MicroAPI::RegTensor<uint8_t> inputRegTensor;
            for (uint16_t i = 0; i < loopNum; i++) {
                AscendC::MicroAPI::DataCopyUnAlignPre(uIn, mxScaleAddr);
                AscendC::MicroAPI::DataCopyUnAlign<uint8_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    inputRegTensor, uIn, mxScaleAddr, quantScaleLastDimU32);
                AscendC::MicroAPI::DataCopyUnAlign<uint8_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    mxScaleBlockAlignAddr, inputRegTensor, uOut, quantScaleLastDimBlockAlignU32);
                AscendC::MicroAPI::DataCopyUnAlignPost(mxScaleBlockAlignAddr, uOut, 0);
            }
        }
    }

    template <typename T>
    __aicore__ inline void DoMxQuantAxis1(const LocalTensor<uint8_t> &outTensor,
                                          const LocalTensor<uint8_t> &outScaleTensor, const LocalTensor<T> &srcTensor,
                                          const LocalTensor<T> &tmpTensor, const int64_t mSize, const int64_t dSize)
    {
        uint32_t allNum = mSize * dSize;
        uint16_t loopNum = (allNum + VL_B16 * DIGIT_TWO - 1) / (VL_B16 * DIGIT_TWO);
        uint32_t allScaleNum = mSize * dSize / QUANT_BLOCK_SIZE;
        uint16_t scaleLoopNum = (allScaleNum + VL_B16 - 1) / VL_B16;

        // 输入输出addr
        auto srcAddr = reinterpret_cast<__ubuf__ T *>(srcTensor.GetPhyAddr());
        auto quantOutAddr = reinterpret_cast<__ubuf__ uint8_t *>(outTensor.GetPhyAddr());
        auto scaleBlockAlignAddr = reinterpret_cast<__ubuf__ uint8_t *>(outScaleTensor.GetPhyAddr());


        // 临时空间
        uint32_t allScaleNumBlockAlign = CEIL_ALIGN(allScaleNum, U8_BLOCK_ALIGN_NUM);
        auto maxExpAddr = reinterpret_cast<__ubuf__ uint16_t *>(tmpTensor.GetPhyAddr());
        auto halfScaleAddr = reinterpret_cast<__ubuf__ uint16_t *>(tmpTensor[allScaleNumBlockAlign].GetPhyAddr());
        auto mxScaleAddr = reinterpret_cast<__ubuf__ uint16_t *>(tmpTensor[2 * allScaleNumBlockAlign].GetPhyAddr());
        // mxScaleU8Addr与mxScaleAddr同地址
        auto mxScaleU8Addr = reinterpret_cast<__ubuf__ uint8_t *>(mxScaleAddr);

        CalcAxis1MaxExpOcpVF(maxExpAddr, srcAddr, allNum, loopNum);
        CalcAxis1ScaleOcpVF(mxScaleAddr, halfScaleAddr, maxExpAddr, allScaleNum, scaleLoopNum);
        CalcAxis1QuantOutVF<T, T_OUT>(quantOutAddr, srcAddr, halfScaleAddr, loopNum);
        PadMxScaleBlockAlign<T>(scaleBlockAlignAddr, mxScaleU8Addr, static_cast<uint16_t>(mSize));
    }

    __aicore__ inline void InterleaveScaleAxis0VF(__ubuf__ uint8_t *scaleOutAddr, __ubuf__ uint8_t *mxScaleU8ddr0,
                                                  const uint16_t scalePairNum, uint16_t colLoopNum, const int64_t dSize)
    {
        __VEC_SCOPE__
        {
            uint32_t dSizeU32 = static_cast<uint32_t>(dSize);
            AscendC::MicroAPI::RegTensor<uint8_t> scale0;
            AscendC::MicroAPI::RegTensor<uint8_t> scale1;
            AscendC::MicroAPI::RegTensor<uint8_t> first128;
            AscendC::MicroAPI::RegTensor<uint8_t> second128;
            AscendC::MicroAPI::MaskReg pregAllU8 =
                AscendC::MicroAPI::CreateMask<uint8_t, AscendC::MicroAPI::MaskPattern::ALL>();
            for (uint16_t pairIdx = 0; pairIdx < scalePairNum; pairIdx++) {
                // 两行scale为一个交织对，[2, dSize] -> [1, dSize, 2]
                for (uint16_t i = 0; i < colLoopNum; i++) {
                    AscendC::MicroAPI::DataCopy<uint8_t, AscendC::MicroAPI::LoadDist::DIST_NORM>(
                        scale0, mxScaleU8ddr0 + pairIdx * dSizeU32 * DIGIT_TWO + i * VL_B16);
                    AscendC::MicroAPI::DataCopy<uint8_t, AscendC::MicroAPI::LoadDist::DIST_NORM>(
                        scale1, mxScaleU8ddr0 + dSizeU32 + pairIdx * dSizeU32 * DIGIT_TWO + i * VL_B16);
                    AscendC::MicroAPI::Interleave(first128, second128, scale0, scale1);
                    AscendC::MicroAPI::DataCopy<uint8_t, AscendC::MicroAPI::StoreDist::DIST_NORM>(
                        scaleOutAddr + pairIdx * dSizeU32 * DIGIT_TWO + i * VL_B16 * DIGIT_TWO, first128, pregAllU8);
                }
            }
        }
    }

    template <typename T>
    __aicore__ inline void CalcAxis0MaxExpOcpVF(__ubuf__ uint8_t *mxScaleAddr, __ubuf__ uint16_t *tmpAddr,
                                                __ubuf__ T *xAddr, int64_t dSize, uint16_t rowLoopNum,
                                                uint16_t colLoopNum)
    {
        __VEC_SCOPE__
        {
            uint32_t dSizeU32 = static_cast<uint32_t>(dSize);
            AscendC::MicroAPI::RegTensor<T> xRegTensor;
            AscendC::MicroAPI::RegTensor<uint16_t> expRegTensor;
            AscendC::MicroAPI::RegTensor<uint16_t> expMaxRegTensor;
            AscendC::MicroAPI::RegTensor<uint16_t> mxScaleRegTensor;
            AscendC::MicroAPI::RegTensor<uint16_t> reversedShareExpRegTensor;
            AscendC::MicroAPI::RegTensor<uint8_t> mxScale;

            AscendC::MicroAPI::RegTensor<uint16_t> specialExpRegTensor;
            AscendC::MicroAPI::RegTensor<uint16_t> biasRegTensor;
            AscendC::MicroAPI::RegTensor<uint16_t> zeroRegTensor;
            AscendC::MicroAPI::RegTensor<uint16_t> nanRegTensor;
            AscendC::MicroAPI::RegTensor<uint16_t> maxEleRegTensor;
            AscendC::MicroAPI::RegTensor<uint16_t> fp8MaxExpRegTensor;
            AscendC::MicroAPI::RegTensor<uint16_t> fp8NanRegTensor;

            AscendC::MicroAPI::MaskReg infMask;
            AscendC::MicroAPI::MaskReg zeroMask;
            AscendC::MicroAPI::MaskReg invalidDataMask;
            AscendC::MicroAPI::MaskReg preg128 =
                AscendC::MicroAPI::CreateMask<uint8_t, AscendC::MicroAPI::MaskPattern::VL128>();
            AscendC::MicroAPI::MaskReg pregAll16 =
                AscendC::MicroAPI::CreateMask<uint16_t, AscendC::MicroAPI::MaskPattern::ALL>();

            AscendC::MicroAPI::Duplicate(maxEleRegTensor, MAX_EXP_FOR_BF16);
            AscendC::MicroAPI::Duplicate(fp8MaxExpRegTensor, FP8_E4M3_MAX_EXP);
            AscendC::MicroAPI::Duplicate(fp8NanRegTensor, MAX_EXP_FOR_FP8);
            AscendC::MicroAPI::Duplicate(biasRegTensor, BF16_EXP_BIAS);
            AscendC::MicroAPI::Duplicate(zeroRegTensor, 0);
            AscendC::MicroAPI::Duplicate(nanRegTensor, NAN_CUSTOMIZATION);
            AscendC::MicroAPI::Duplicate(specialExpRegTensor, SPECIAL_EXP_THRESHOLD);
            // 列方向循环
            for (uint16_t i = 0; i < colLoopNum; i++) {
                // 行方向循环
                AscendC::MicroAPI::Duplicate(expMaxRegTensor, 0);
                for (uint16_t j = 0; j < rowLoopNum; j++) {
                    DataCopy(xRegTensor, xAddr + j * dSizeU32 + i * VL_B16);
                    AscendC::MicroAPI::And(expRegTensor, (AscendC::MicroAPI::RegTensor<uint16_t> &)xRegTensor,
                                           maxEleRegTensor, pregAll16);
                    AscendC::MicroAPI::Max(expMaxRegTensor, expMaxRegTensor, expRegTensor, pregAll16);
                }

                AscendC::MicroAPI::Compare<uint16_t, CMPMODE::NE>(infMask, expMaxRegTensor, maxEleRegTensor, pregAll16);
                AscendC::MicroAPI::Compare<uint16_t, CMPMODE::NE>(zeroMask, expMaxRegTensor, zeroRegTensor, pregAll16);
                AscendC::MicroAPI::Compare<uint16_t, CMPMODE::LE>(invalidDataMask, expMaxRegTensor, fp8MaxExpRegTensor,
                                                                  pregAll16);
                AscendC::MicroAPI::Select<uint16_t>(expMaxRegTensor, fp8MaxExpRegTensor, expMaxRegTensor,
                                                    invalidDataMask);
                AscendC::MicroAPI::Sub(expMaxRegTensor, expMaxRegTensor, fp8MaxExpRegTensor, pregAll16);

                AscendC::MicroAPI::ShiftRights(mxScaleRegTensor, expMaxRegTensor, SHR_NUM_FOR_BF16, pregAll16);
                AscendC::MicroAPI::Select<uint16_t>(mxScaleRegTensor, mxScaleRegTensor, fp8NanRegTensor, infMask);
                AscendC::MicroAPI::Select<uint16_t>(mxScaleRegTensor, mxScaleRegTensor, zeroRegTensor, zeroMask);

                AscendC::MicroAPI::Pack<uint8_t, uint16_t, AscendC::MicroAPI::HighLowPart::LOWEST>(
                    (AscendC::MicroAPI::RegTensor<uint8_t> &)mxScale, mxScaleRegTensor);
                // 写出只写128个数
                DataCopy(mxScaleAddr + i * VL_B16, mxScale, preg128);
                // 求1/scale
                AscendC::MicroAPI::Compare<uint16_t, CMPMODE::EQ>(invalidDataMask, expMaxRegTensor, biasRegTensor,
                                                                  pregAll16);
                AscendC::MicroAPI::Sub(reversedShareExpRegTensor, biasRegTensor, expMaxRegTensor, pregAll16);
                AscendC::MicroAPI::Select<uint16_t>(reversedShareExpRegTensor, reversedShareExpRegTensor, nanRegTensor,
                                                    infMask);
                AscendC::MicroAPI::Select<uint16_t>(reversedShareExpRegTensor, reversedShareExpRegTensor, zeroRegTensor,
                                                    zeroMask);
                AscendC::MicroAPI::Select<uint16_t>(reversedShareExpRegTensor, specialExpRegTensor,
                                                    reversedShareExpRegTensor, invalidDataMask);
                DataCopy(tmpAddr + i * VL_B16, reversedShareExpRegTensor, pregAll16);
            }
        }
    }

    template <typename T, typename U>
    __aicore__ inline void CalcAxis0QuantOutVF(__ubuf__ uint8_t *yAddr, __ubuf__ T *xAddr, __ubuf__ uint16_t *tmpAddr,
                                               int64_t dSize, uint16_t rowLoopNum, uint16_t colLoopNum)
    {
        __VEC_SCOPE__
        {
            uint32_t dSizeU32 = static_cast<uint32_t>(dSize);
            AscendC::MicroAPI::RegTensor<T> xRegTensor;
            AscendC::MicroAPI::RegTensor<float> xFP32RegTensor0;
            AscendC::MicroAPI::RegTensor<float> xFP32RegTensor1;
            AscendC::MicroAPI::RegTensor<uint16_t> reversedShareExpRegTensor;
            AscendC::MicroAPI::RegTensor<float> reversedShareExpFP32RegTensor0;
            AscendC::MicroAPI::RegTensor<float> reversedShareExpFP32RegTensor1;
            AscendC::MicroAPI::RegTensor<U> yZeroFP8;
            AscendC::MicroAPI::RegTensor<U> yOneFP8;
            AscendC::MicroAPI::RegTensor<U> yZeroFP4;

            AscendC::MicroAPI::MaskReg preg128 =
                AscendC::MicroAPI::CreateMask<uint8_t, AscendC::MicroAPI::MaskPattern::VL128>();
            AscendC::MicroAPI::MaskReg pregAll16 =
                AscendC::MicroAPI::CreateMask<uint16_t, AscendC::MicroAPI::MaskPattern::ALL>();
            AscendC::MicroAPI::MaskReg pregAll32 =
                AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();

            static constexpr AscendC::MicroAPI::CastTrait castTraitXdtypetoFp32Zero = {
                AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN,
                AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
            static constexpr AscendC::MicroAPI::CastTrait castTraitXdtypetoFp32One = {
                AscendC::MicroAPI::RegLayout::ONE, AscendC::MicroAPI::SatMode::UNKNOWN,
                AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
            static constexpr AscendC::MicroAPI::CastTrait castTraitFp32toFp8 = {
                AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::SAT,
                AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

            // 列方向循环
            for (uint16_t i = 0; i < colLoopNum; i++) {
                // 行方向循环
                AscendC::MicroAPI::DataCopy<uint16_t, MicroAPI::LoadDist::DIST_NORM>(reversedShareExpRegTensor,
                                                                                     tmpAddr + i * VL_B16);
                AscendC::MicroAPI::Cast<float, bfloat16_t, castTraitXdtypetoFp32Zero>(
                    reversedShareExpFP32RegTensor0,
                    (AscendC::MicroAPI::RegTensor<bfloat16_t> &)reversedShareExpRegTensor, pregAll16);
                AscendC::MicroAPI::Cast<float, bfloat16_t, castTraitXdtypetoFp32One>(
                    reversedShareExpFP32RegTensor1,
                    (AscendC::MicroAPI::RegTensor<bfloat16_t> &)reversedShareExpRegTensor, pregAll16);
                for (uint16_t j = 0; j < rowLoopNum; j++) {
                    AscendC::MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(xRegTensor,
                                                                                  xAddr + j * dSizeU32 + i * VL_B16);

                    AscendC::MicroAPI::Cast<float, T, castTraitXdtypetoFp32Zero>(xFP32RegTensor0, xRegTensor,
                                                                                 pregAll16);
                    AscendC::MicroAPI::Cast<float, T, castTraitXdtypetoFp32One>(xFP32RegTensor1, xRegTensor, pregAll16);

                    AscendC::MicroAPI::Mul(xFP32RegTensor0, xFP32RegTensor0, reversedShareExpFP32RegTensor0, pregAll32);
                    AscendC::MicroAPI::Mul(xFP32RegTensor1, xFP32RegTensor1, reversedShareExpFP32RegTensor1, pregAll32);

                    AscendC::MicroAPI::Cast<U, float, castTraitFp32toFp8>(
                        yZeroFP8, (AscendC::MicroAPI::RegTensor<float> &)xFP32RegTensor0, pregAll32);
                    AscendC::MicroAPI::Pack<uint16_t, uint32_t, AscendC::MicroAPI::HighLowPart::LOWEST>(
                        (AscendC::MicroAPI::RegTensor<uint16_t> &)yZeroFP8,
                        (AscendC::MicroAPI::RegTensor<uint32_t> &)yZeroFP8);

                    AscendC::MicroAPI::Cast<U, float, castTraitFp32toFp8>(
                        yOneFP8, (AscendC::MicroAPI::RegTensor<float> &)xFP32RegTensor1, pregAll32);
                    AscendC::MicroAPI::Pack<uint16_t, uint32_t, AscendC::MicroAPI::HighLowPart::LOWEST>(
                        (AscendC::MicroAPI::RegTensor<uint16_t> &)yOneFP8,
                        (AscendC::MicroAPI::RegTensor<uint32_t> &)yOneFP8);

                    AscendC::MicroAPI::Interleave((AscendC::MicroAPI::RegTensor<uint16_t> &)yZeroFP8,
                                                  (AscendC::MicroAPI::RegTensor<uint16_t> &)yOneFP8,
                                                  (AscendC::MicroAPI::RegTensor<uint16_t> &)yZeroFP8,
                                                  (AscendC::MicroAPI::RegTensor<uint16_t> &)yOneFP8);

                    AscendC::MicroAPI::Pack<uint8_t, uint16_t, AscendC::MicroAPI::HighLowPart::LOWEST>(
                        (AscendC::MicroAPI::RegTensor<uint8_t> &)yZeroFP8,
                        (AscendC::MicroAPI::RegTensor<uint16_t> &)yZeroFP8);

                    DataCopy(yAddr + j * dSizeU32 + i * VL_B16, (AscendC::MicroAPI::RegTensor<uint8_t> &)yZeroFP8,
                             preg128);
                }
            }
        }
    }

    template <typename T>
    __aicore__ inline void DoMxQuantAxis0(const LocalTensor<uint8_t> &outTensor,
                                          const LocalTensor<uint8_t> &outScaleTensor, const LocalTensor<T> &srcTensor,
                                          const LocalTensor<T> &tmpTensor, const int64_t mSize, const int64_t dSize)
    {
        // quantAxis=0: 沿M维度量化，blockSize=32
        // mSize是64的整数倍，dSize是128的整数倍

        int64_t blockCount = mSize / QUANT_BLOCK_SIZE;

        // 输入输出addr
        auto srcAddr = reinterpret_cast<__ubuf__ T *>(srcTensor.GetPhyAddr());
        auto quantOutAddr = reinterpret_cast<__ubuf__ uint8_t *>(outTensor.GetPhyAddr());
        auto scaleOutAddr = reinterpret_cast<__ubuf__ uint8_t *>(outScaleTensor.GetPhyAddr());

        uint16_t rowLoopNum = QUANT_BLOCK_SIZE;
        uint16_t colLoopNum = dSize / VL_B16;
        uint16_t scalePairNum = blockCount / DIGIT_TWO;

        // 临时空间
        auto tmpAddr = reinterpret_cast<__ubuf__ uint16_t *>(tmpTensor.GetPhyAddr());
        auto mxScaleTmpAddr = reinterpret_cast<__ubuf__ uint8_t *>(tmpTensor[blockCount * dSize].GetPhyAddr());

        for (int64_t blockIdx = 0; blockIdx < blockCount; blockIdx++) {
            CalcAxis0MaxExpOcpVF(mxScaleTmpAddr, tmpAddr, srcAddr, dSize, rowLoopNum, colLoopNum);
            CalcAxis0QuantOutVF<T, T_OUT>(quantOutAddr, srcAddr, tmpAddr, dSize, rowLoopNum, colLoopNum);
            tmpAddr += dSize;
            mxScaleTmpAddr += dSize;
            srcAddr += QUANT_BLOCK_SIZE * dSize;
            quantOutAddr += QUANT_BLOCK_SIZE * dSize;
        }
        mxScaleTmpAddr = reinterpret_cast<__ubuf__ uint8_t *>(tmpTensor[blockCount * dSize].GetPhyAddr());
        InterleaveScaleAxis0VF(scaleOutAddr, mxScaleTmpAddr, scalePairNum, colLoopNum, dSize);
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
        DataCopyPad(outGm, outTensor, copyOutParams);
    }

    __aicore__ inline void ScatterUpdateK(const LocalTensor<uint8_t> &outTensor,
                                          const LocalTensor<uint8_t> &outScaleTensor, const int64_t processSeqLen,
                                          const int64_t startTIdx)
    {
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
            // 负索引跳过写
            if (cacheIndex < 0) {
                continue;
            }
            // 计算Bn维度上的索引
            int64_t bnIndex = cacheIndex / tilingData_->blockSize;
            // 计算Bs维度上的索引
            int64_t bsIndex = cacheIndex % tilingData_->blockSize;
            DataCopyPad(kCacheGm_[bnIndex * kCacheBlockOffset_ + bsIndex * tilingData_->headDim],
                        outTensor[i * kHiddenSize_], copyQuantOutParams);
            DataCopyPad(kScaleCacheGm_[bnIndex * kScaleCacheBlockOffset_ + bsIndex * quantScaleLastDim_],
                        outScaleTensor[i * tilingData_->kNumHead * quantScaleLastDimBlockAlign_], copyScaleOutParams);
        }
    }

    __aicore__ inline void ScatterUpdateV(const LocalTensor<uint8_t> &outTensor,
                                          const LocalTensor<uint8_t> &outScaleTensor, const int64_t processSeqLen,
                                          const int64_t processNLen, const int64_t startTIdx, const int64_t nIdx)
    {
        DataCopyExtParams copyQuantOutParams;
        copyQuantOutParams.blockCount = processNLen;
        copyQuantOutParams.blockLen = tilingData_->headDim * sizeof(uint8_t);
        copyQuantOutParams.srcStride = 0;
        copyQuantOutParams.dstStride = (tilingData_->blockSize - 1) * tilingData_->headDim * sizeof(uint8_t);

        for (int64_t i = 0; i < processSeqLen; ++i) {
            int64_t cacheIndex = kvSlotMappingGm_(startTIdx + i);
            // 负索引跳过写
            if (cacheIndex < 0) {
                continue;
            }
            // 计算Bn维度上的索引
            int64_t bnIndex = cacheIndex / tilingData_->blockSize;
            // 计算Bs维度上的索引
            int64_t bsIndex = cacheIndex % tilingData_->blockSize;
            DataCopyPad(vCacheGm_[bnIndex * vCacheBlockOffset_ + nIdx * tilingData_->blockSize * tilingData_->headDim +
                                  bsIndex * tilingData_->headDim],
                        outTensor[i * processNLen * tilingData_->headDim], copyQuantOutParams);
        }

        DataCopyExtParams copyScaleOutParams;
        copyScaleOutParams.blockCount = processNLen;
        copyScaleOutParams.blockLen = tilingData_->headDim * DIGIT_TWO * sizeof(uint8_t);
        copyScaleOutParams.srcStride = 0;
        copyScaleOutParams.dstStride = (vScaleCacheActualBs_ - 1) * tilingData_->headDim * DIGIT_TWO * sizeof(uint8_t);
        for (int64_t i = 0; i < processSeqLen / QUANT_BLOCK_SIZE / DIGIT_TWO; ++i) {
            int64_t cacheIndex = vScaleSlotMappingGm_(startTIdx / QUANT_BLOCK_SIZE / DIGIT_TWO + i);
            // 负索引跳过写
            if (cacheIndex < 0) {
                continue;
            }
            // 计算Bn维度上的索引
            int64_t bnIndex = cacheIndex / vScaleCacheActualBs_;
            // 计算Bs维度上的索引
            int64_t bsIndex = cacheIndex % vScaleCacheActualBs_;
            DataCopyPad(vScaleCacheGm_[bnIndex * vScaleCacheBlockOffset_ +
                                       nIdx * vScaleCacheActualBs_ * tilingData_->headDim * DIGIT_TWO +
                                       bsIndex * tilingData_->headDim * DIGIT_TWO],
                        outScaleTensor[i * processNLen * tilingData_->headDim * DIGIT_TWO], copyScaleOutParams);
        }
    }

    __aicore__ inline void DoPhase1()
    {
        if (GetBlockIdx() >= tilingData_->qUsedCoreNum) {
            return;
        }
#if (__NPU_ARCH__ == 3510)
    AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(0);
#endif
        // init pipe
        pipe_->InitBuffer(inQueue_, DOUBLE_BUFFER,
                        tilingData_->qUbFactor * (COS_SIN_NUM + tilingData_->qNumHead) * tilingData_->headDim *
                            sizeof(T_QKV));
        pipe_->InitBuffer(outQueue_, 1,
                        tilingData_->qUbFactor * (tilingData_->headDim + U8_BLOCK_ALIGN_NUM) * tilingData_->qNumHead *
                            sizeof(int8_t));
        pipe_->InitBuffer(wsBuffer0_, tilingData_->qUbFactor * qHiddenSize_ * sizeof(T_QKV));
        pipe_->InitBuffer(wsBuffer1_, tilingData_->qUbFactor * qHiddenSize_ * sizeof(T_QKV));
        int64_t startTIndex = GetBlockIdx() * tilingData_->qBlockFactor;
        int64_t endTIndex = startTIndex + tilingData_->qBlockFactor;
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
            LocalTensor<T_QKV> qTensor = inQueue_.AllocTensor<T_QKV>();
            LocalTensor<T_QKV> cosTensor = qTensor[tilingData_->qUbFactor * qHiddenSize_];
            LocalTensor<T_QKV> sinTensor = cosTensor[tilingData_->qUbFactor * tilingData_->headDim];
            GatherCopyIn(qTensor, qkvGm_[tIdx * allHiddenSize_], processSeqLen, allHiddenSize_, qHiddenSize_);
            GatherCopyIn(cosTensor, cosGm_[tIdx * tilingData_->headDim], processSeqLen, tilingData_->headDim,
                         tilingData_->headDim);
            GatherCopyIn(sinTensor, sinGm_[tIdx * tilingData_->headDim], processSeqLen, tilingData_->headDim,
                         tilingData_->headDim);
            inQueue_.EnQue(qTensor);
            qTensor = inQueue_.DeQue<T_QKV>();
            LocalTensor<T_QKV> ropeTensor = wsBuffer0_.Get<T_QKV>();
            DoRope<T_QKV, T_QKV>(ropeTensor, qTensor, cosTensor, sinTensor, processSeqLen, tilingData_->qNumHead,
                                 tilingData_->headDim);
            inQueue_.FreeTensor(qTensor);
            LocalTensor<uint8_t> outTensor = outQueue_.AllocTensor<uint8_t>();
            LocalTensor<uint8_t> outScaleTensor = outTensor[tilingData_->qUbFactor * qHiddenSize_];
            LocalTensor<T_QKV> tmpTensor = wsBuffer1_.Get<T_QKV>();
            DoMxQuant(outTensor, outScaleTensor, ropeTensor, tmpTensor, 1, processSeqLen * tilingData_->qNumHead,
                      tilingData_->headDim);
            outQueue_.EnQue(outTensor);
            outTensor = outQueue_.DeQue<uint8_t>();
            CopyQuantOut(qGm_[tIdx * qHiddenSize_], outTensor, 1, processSeqLen * qHiddenSize_,
                         processSeqLen * qHiddenSize_);
            CopyQuantOut(qScaleGm_[tIdx * tilingData_->qNumHead * quantScaleLastDim_], outScaleTensor,
                         processSeqLen * tilingData_->qNumHead, quantScaleLastDimBlockAlign_, quantScaleLastDim_);
            outQueue_.FreeTensor(outTensor);
        }
        pipe_->Reset();
    }

    __aicore__ inline void DoPhase2()
    {
        if (GetBlockIdx() >= tilingData_->kUsedCoreNum) {
            return;
        }
#if (__NPU_ARCH__ == 3510)
    AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(0);
#endif
        // init pipe
        pipe_->InitBuffer(inQueue_, DOUBLE_BUFFER,
                        tilingData_->kUbFactor * (COS_SIN_NUM + tilingData_->kNumHead) * tilingData_->headDim *
                            sizeof(T_QKV));
        pipe_->InitBuffer(inQueueGamma_, 1, tilingData_->headDim * sizeof(float));
        pipe_->InitBuffer(outQueue_, 1,
                        tilingData_->kUbFactor * (tilingData_->headDim + U8_BLOCK_ALIGN_NUM) * tilingData_->kNumHead *
                            sizeof(int8_t));
        pipe_->InitBuffer(wsBuffer0_, tilingData_->kUbFactor * qHiddenSize_ * sizeof(float));
        pipe_->InitBuffer(wsBuffer1_, tilingData_->kUbFactor * qHiddenSize_ * sizeof(T_QKV));
        // CopyIn gamma
        LocalTensor<float> gammaTensor = inQueueGamma_.AllocTensor<float>();
        GatherCopyIn(gammaTensor, gammaGm_, 1, tilingData_->headDim, tilingData_->headDim);
        inQueueGamma_.EnQue(gammaTensor);
        gammaTensor = inQueueGamma_.DeQue<float>();
        int64_t startTIndex = GetBlockIdx() * tilingData_->kBlockFactor;
        int64_t endTIndex = startTIndex + tilingData_->kBlockFactor;
        // 尾核场景
        if (endTIndex > tilingData_->seqLengthSum) {
            endTIndex = tilingData_->seqLengthSum;
        }
        for (int64_t tIdx = startTIndex; tIdx < endTIndex; tIdx += tilingData_->kUbFactor) {
            int64_t processSeqLen = tilingData_->kUbFactor;
            if (tIdx + processSeqLen > endTIndex) {
                processSeqLen = endTIndex - tIdx;
            }
            LocalTensor<T_QKV> kTensor = inQueue_.AllocTensor<T_QKV>();
            LocalTensor<T_QKV> cosTensor = kTensor[tilingData_->kUbFactor * kHiddenSize_];
            LocalTensor<T_QKV> sinTensor = cosTensor[tilingData_->kUbFactor * tilingData_->headDim];
            GatherCopyIn(kTensor, qkvGm_[tIdx * allHiddenSize_ + qHiddenSize_], processSeqLen, allHiddenSize_,
                         kHiddenSize_);
            GatherCopyIn(cosTensor, cosGm_[tIdx * tilingData_->headDim], processSeqLen, tilingData_->headDim,
                         tilingData_->headDim);
            GatherCopyIn(sinTensor, sinGm_[tIdx * tilingData_->headDim], processSeqLen, tilingData_->headDim,
                         tilingData_->headDim);
            inQueue_.EnQue(kTensor);
            kTensor = inQueue_.DeQue<T_QKV>();
            LocalTensor<float> rmsNormTensor = wsBuffer0_.Get<float>();
            DoRmsNorm<T_QKV>(rmsNormTensor, kTensor, gammaTensor, processSeqLen * tilingData_->kNumHead,
                             tilingData_->headDim);
            LocalTensor<T_QKV> ropeTensor = wsBuffer1_.Get<T_QKV>();
            DoRope<float, T_QKV>(ropeTensor, rmsNormTensor, cosTensor, sinTensor, processSeqLen, tilingData_->kNumHead,
                                 tilingData_->headDim);
            inQueue_.FreeTensor(kTensor);
            LocalTensor<uint8_t> outTensor = outQueue_.AllocTensor<uint8_t>();
            LocalTensor<uint8_t> outScaleTensor = outTensor[tilingData_->kUbFactor * kHiddenSize_];
            LocalTensor<T_QKV> tmpTensor = wsBuffer0_.Get<T_QKV>();
            DoMxQuant(outTensor, outScaleTensor, ropeTensor, tmpTensor, 1, processSeqLen * tilingData_->kNumHead,
                      tilingData_->headDim);
            outQueue_.EnQue(outTensor);
            outTensor = outQueue_.DeQue<uint8_t>();
            ScatterUpdateK(outTensor, outScaleTensor, processSeqLen, tIdx);
            outQueue_.FreeTensor(outTensor);
        }
        inQueueGamma_.FreeTensor(gammaTensor);
        pipe_->Reset();
    }

    __aicore__ inline void DoPhase3()
    {
        if (GetBlockIdx() >= tilingData_->vUsedCoreNum) {
            return;
        }
#if (__NPU_ARCH__ == 3510)
        AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(0);
#endif
        // init pipe
        pipe_->InitBuffer(inQueue_, DOUBLE_BUFFER, tilingData_->vTUbFactor * vUbHiddenSize_ * sizeof(T_QKV));
        pipe_->InitBuffer(outQueue_, 1,
                          (tilingData_->vTUbFactor + tilingData_->vTUbFactor / QUANT_BLOCK_SIZE) * vUbHiddenSize_ *
                              sizeof(int8_t));
        pipe_->InitBuffer(wsBuffer0_,
                          2 * (tilingData_->vTUbFactor / QUANT_BLOCK_SIZE * vUbHiddenSize_ * sizeof(T_QKV)));
        int64_t startTIndex = GetBlockIdx() * tilingData_->vBlockFactor;
        int64_t endTIndex = startTIndex + tilingData_->vBlockFactor;
        // 尾核场景
        if (endTIndex > tilingData_->seqLengthSum) {
            endTIndex = tilingData_->seqLengthSum;
        }
        for (int64_t tIdx = startTIndex; tIdx < endTIndex; tIdx += tilingData_->vTUbFactor) {
            int64_t processSeqLen = tilingData_->vTUbFactor;
            if (tIdx + processSeqLen > endTIndex) {
                processSeqLen = endTIndex - tIdx;
            }
            for (int64_t nIdx = 0; nIdx < tilingData_->vNumHead; nIdx += tilingData_->vNumHeadUbFactor) {
                int64_t processNLen = tilingData_->vNumHeadUbFactor;
                if (nIdx + processNLen > tilingData_->vNumHead) {
                    processNLen = tilingData_->vNumHead - nIdx;
                }
                LocalTensor<T_QKV> vTensor = inQueue_.AllocTensor<T_QKV>();
                GatherCopyIn(vTensor,
                             qkvGm_[tIdx * allHiddenSize_ +
                                    (tilingData_->qNumHead + tilingData_->kNumHead + nIdx) * tilingData_->headDim],
                             processSeqLen, allHiddenSize_, processNLen * tilingData_->headDim);
                inQueue_.EnQue(vTensor);
                vTensor = inQueue_.DeQue<T_QKV>();
                LocalTensor<uint8_t> outTensor = outQueue_.AllocTensor<uint8_t>();
                LocalTensor<uint8_t> outScaleTensor = outTensor[tilingData_->vTUbFactor * vUbHiddenSize_];
                LocalTensor<T_QKV> tmpTensor = wsBuffer0_.Get<T_QKV>();
                DoMxQuant(outTensor, outScaleTensor, vTensor, tmpTensor, 0, processSeqLen,
                          processNLen * tilingData_->headDim);
                inQueue_.FreeTensor(vTensor);
                outQueue_.EnQue(outTensor);
                outTensor = outQueue_.DeQue<uint8_t>();
                ScatterUpdateV(outTensor, outScaleTensor, processSeqLen, processNLen, tIdx, nIdx);
                outQueue_.FreeTensor(outTensor);
            }
        }
        pipe_->Reset();
    }

private:
    const FusedKRmsNormRopeStoreKvCacheMxQuantTilingData *tilingData_;
    TQue<QuePosition::VECIN, 1> inQueue_, inQueueGamma_;
    TQue<QuePosition::VECOUT, 1> outQueue_;
    TBuf<TPosition::VECCALC> wsBuffer0_, wsBuffer1_;
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
} // namespace FusedKRmsNormRopeStoreKvCacheMxQuant
#endif // FUSED_K_RMS_NORM_ROPE_STORE_KV_CACHE_MX_QUANT_REGBASE_H_