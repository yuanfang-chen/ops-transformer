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
 * \file moe_init_routing_v3_mx_quant_gather_mxfp8_quant.h
 * \brief GatherMxfp8Quant stage — gather x rows by sorted_row_idx and perform MX-FP8 quantization
 *
 * Template parameters:
 *   T = input dtype (bfloat16_t or half)
 *   U = output dtype (fp8_e4m3fn_t or fp8_e5m2_t)
 */
#ifndef MOE_INIT_ROUTING_V3_MX_QUANT_GATHER_MXFP8_QUANT_H
#define MOE_INIT_ROUTING_V3_MX_QUANT_GATHER_MXFP8_QUANT_H

#include "moe_init_routing_v3_mx_quant_common.h"
#include "op_kernel/platform_util.h"
#include "op_kernel/math_util.h"

namespace MoeInitRoutingV3MxQuantNs {
using namespace AscendC;

// ============================================================================
// vfComputeMaxExp: compute per-MX-block max exponent from input x (bf16/fp16)
// ============================================================================
template <typename T, typename U>
__simd_vf__ inline void vfComputeMaxExp(__ubuf__ T *xAddr, __ubuf__ uint16_t *maxExpOutAddr, uint32_t xElemNum,
                                        uint16_t vfLoopNum, uint32_t vlForT, uint32_t numVRegBlocks)
{
    using namespace AscendC::MicroAPI;

    // For fp16->bf16 cast (same width, truncation)
    static constexpr CastTrait traitFP16ToBF16 = {RegLayout::UNKNOWN, SatMode::UNKNOWN, MaskMergeMode::ZEROING,
                                                   RoundMode::CAST_TRUNC};

    // x0 stores odd-position elements, x1 stores even-position elements
    RegTensor<T> x0, x1;
    RegTensor<bfloat16_t> x0BF16, x1BF16;
    RegTensor<uint16_t> exp0, exp1, exp0FP16, exp1FP16, maxExp;
    // Exponent bit masks for FP16 and BF16
    RegTensor<uint16_t> emaskFP16, emaskBF16;
    Duplicate(emaskFP16, FP16_EMASK_AND_INF_VAL);
    Duplicate(emaskBF16, BF16_EMASK_AND_INF_VAL);
    // Full mask for 2-byte register width
    MaskReg maskAllB16 = CreateMask<uint16_t, MaskPattern::ALL>();
    MaskReg mask0, mask1, mask0FP16NanInf, mask1FP16NanInf;
    // Unaligned write-out register
    UnalignReg uReg;

    for (uint16_t i = 0; i < vfLoopNum; i++) {
        mask0 = UpdateMask<T>(xElemNum);
        mask1 = UpdateMask<T>(xElemNum);
        MaskDeInterleave<T>(mask0, mask1, mask0, mask1);

        // Double-load: odd-position elements -> x0, even-position -> x1
        DataCopy<T, PostLiteral::POST_MODE_UPDATE, LoadDist::DIST_DINTLV_B16>(x0, x1, xAddr, vlForT * 2);

        if constexpr (IsSameType<T, half>::value) {
            // Extract fp16 inf/nan elements
            And(exp0FP16, (RegTensor<uint16_t> &)x0, emaskFP16, mask0);
            And(exp1FP16, (RegTensor<uint16_t> &)x1, emaskFP16, mask1);
            Compare<uint16_t, CMPMODE::EQ>(mask0FP16NanInf, exp0FP16, emaskFP16, mask0);
            Compare<uint16_t, CMPMODE::EQ>(mask1FP16NanInf, exp1FP16, emaskFP16, mask1);

            // fp16 -> bf16 conversion first
            Cast<bfloat16_t, T, traitFP16ToBF16>(x0BF16, x0, mask0);
            Cast<bfloat16_t, T, traitFP16ToBF16>(x1BF16, x1, mask1);
            // Extract BF16 exponent bits
            And(exp0, (RegTensor<uint16_t> &)x0BF16, emaskBF16, mask0);
            And(exp1, (RegTensor<uint16_t> &)x1BF16, emaskBF16, mask1);

            // exp[expFP16==nan/inf] = nan/inf
            Select(exp0, emaskBF16, exp0, mask0FP16NanInf);
            Select(exp1, emaskBF16, exp1, mask1FP16NanInf);
        } else {
            // BF16 path: directly extract exponent bits
            And(exp0, (RegTensor<uint16_t> &)x0, emaskBF16, mask0);
            And(exp1, (RegTensor<uint16_t> &)x1, emaskBF16, mask1);
        }
        // Max of odd/even pairs (per 2 elements)
        Max(maxExp, exp0, exp1, mask0);
        // Reduce to per-block max (each 32B block -> 1 value)
        ReduceMaxWithDataBlock(maxExp, maxExp, maskAllB16);
        // Unaligned write-out: numVRegBlocks elements per iteration
        DataCopyUnAlign<uint16_t, PostLiteral::POST_MODE_UPDATE>(maxExpOutAddr, maxExp, uReg, numVRegBlocks);
    }
    // Finalize unaligned write
    DataCopyUnAlignPost(maxExpOutAddr, uReg, 0);
}

// ============================================================================
// vfComputeScale: compute mxScale (fp8_e8m0) and invScale (bf16) from maxExp
// ============================================================================
template <typename T, typename U>
__simd_vf__ inline void vfComputeScale(__ubuf__ uint16_t *maxExpInAddr, __ubuf__ uint16_t *mxScaleOutAddr,
                                       __ubuf__ uint16_t *invScaleOutAddr, uint32_t scaleElemNum, uint32_t validScaleElemNum,
                                       uint16_t vfLoopNum, uint32_t vlForT, uint16_t expLowerBoundValue)
{
    using namespace AscendC::MicroAPI;

    RegTensor<uint16_t> maxExp, sharedExp, mxScale, invScale;
    // BF16 inf exponent value
    RegTensor<uint16_t> infBF16;
    Duplicate(infBF16, BF16_EMASK_AND_INF_VAL);
    // Zero value (16-bit)
    RegTensor<uint16_t> zeroB16;
    Duplicate(zeroB16, 0);
    // Lower bound for maxExp
    RegTensor<uint16_t> expLowerBound;
    Duplicate(expLowerBound, expLowerBoundValue);
    // E8M0 NaN value (low 8 bits of uint16)
    RegTensor<uint16_t> nanForE8M0;
    Duplicate(nanForE8M0, FP8_E8M0_NAN_VAL);
    // For computing invScale = BF16_EXP_INVSUB - sharedExp
    RegTensor<uint16_t> invSub;
    Duplicate(invSub, BF16_EXP_INVSUB);
    // BF16 NaN value
    RegTensor<uint16_t> nanBF16;
    Duplicate(nanBF16, BF16_NAN_VAL);
    // E8M0 special minimum value
    RegTensor<uint16_t> specialMinE8M0;
    Duplicate(specialMinE8M0, FP8_E8M0_SPECIAL_MIN);

    // Loop masks
    MaskReg maskLoop, maskValid;
    // Compare result masks
    MaskReg maskInfBF16, maskZero, maskLowExp, maskSpecialMin;

    for (uint16_t i = 0; i < vfLoopNum; i++) {
        maskLoop = UpdateMask<uint16_t>(scaleElemNum);
        maskValid = UpdateMask<uint16_t>(validScaleElemNum);
        // Load maxExp computed by vfComputeMaxExp
        DataCopy<uint16_t, PostLiteral::POST_MODE_UPDATE>(maxExp, maxExpInAddr, vlForT);

        // 1. Compute and store mxScale (float8_e8m0)

        // Clamp maxExp below lower bound
        Compare<uint16_t, CMPMODE::LT>(maskLowExp, maxExp, expLowerBound, maskValid);
        Select<uint16_t>(maxExp, expLowerBound, maxExp, maskLowExp);

        // sharedExp = maxExp - expLowerBound
        Sub(sharedExp, maxExp, expLowerBound, maskValid);
        // mxScale = sharedExp >> BF16_EXP_SHR_BITS (shift exponent to low 8 bits for e8m0)
        ShiftRights(mxScale, sharedExp, BF16_EXP_SHR_BITS, maskValid);

        // Handle special cases: mxScale[maxExp==inf] = nanE8M0
        Compare<uint16_t, CMPMODE::EQ>(maskInfBF16, maxExp, infBF16, maskValid);
        Select<uint16_t>(mxScale, nanForE8M0, mxScale, maskInfBF16);
        // mxScale[maxExp==0] = 0
        Compare<uint16_t, CMPMODE::EQ>(maskZero, maxExp, zeroB16, maskValid);
        Select<uint16_t>(mxScale, zeroB16, mxScale, maskZero);

        // Store mxScale: pack uint16 -> uint8 (only low 8 bits), so element count is vlForT/2
        DataCopy<uint16_t, PostLiteral::POST_MODE_UPDATE, StoreDist::DIST_PACK_B16>(mxScaleOutAddr, mxScale, vlForT / 2,
                                                                                    maskLoop);

        // 2. Compute and store invScale (bfloat16)

        // invScale = invSub - sharedExp
        Sub<uint16_t>(invScale, invSub, sharedExp, maskValid);
        // invScale[maxExp==inf] = nanBF16
        Select<uint16_t>(invScale, nanBF16, invScale, maskInfBF16);
        // invScale[maxExp==0] = 0
        Select<uint16_t>(invScale, zeroB16, invScale, maskZero);
        // invScale[(invSub==sharedExp)] = FP8_E8M0_SPECIAL_MIN
        Compare<uint16_t, CMPMODE::EQ>(maskSpecialMin, invSub, sharedExp, maskValid);
        Select<uint16_t>(invScale, specialMinE8M0, invScale, maskSpecialMin);

        // Store invScale as uint16 (binary representation of bfloat16)
        DataCopy<uint16_t, PostLiteral::POST_MODE_UPDATE>(invScaleOutAddr, invScale, vlForT, maskLoop);
    }
}

// ============================================================================
// vfComputeData: quantize x using invScale -> fp8 output
// ============================================================================
template <typename T, typename U>
__simd_vf__ inline void vfComputeData(__ubuf__ T *xAddr, __ubuf__ uint16_t *invScaleInAddr,
                                      __ubuf__ int8_t *xQuantOutAddr, uint32_t xElemNum, uint16_t vfLoopNum,
                                      uint32_t vlForT, uint32_t numVRegBlocks)
{
    using namespace AscendC::MicroAPI;

    // fp16 -> bf16 (same width, truncation)
    static constexpr CastTrait traitFP16ToBF16 = {RegLayout::UNKNOWN, SatMode::UNKNOWN, MaskMergeMode::ZEROING,
                                                   RoundMode::CAST_TRUNC};
    // b16 -> b32: odd-position elements (layout 0)
    static constexpr CastTrait traitB16ToB32Layout0 = {RegLayout::ZERO, SatMode::UNKNOWN, MaskMergeMode::ZEROING,
                                                       RoundMode::UNKNOWN};
    // b16 -> b32: even-position elements (layout 1)
    static constexpr CastTrait traitB16ToB32Layout1 = {RegLayout::ONE, SatMode::UNKNOWN, MaskMergeMode::ZEROING,
                                                       RoundMode::UNKNOWN};
    // b32 -> b8: fp32 -> fp8, with saturation and rint rounding
    static constexpr CastTrait traitB32ToB8Layout0 = {RegLayout::ZERO, SatMode::SAT, MaskMergeMode::ZEROING,
                                                      RoundMode::CAST_RINT};

    RegTensor<uint16_t> invScale;
    RegTensor<float> invScaleFP32;
    RegTensor<T> x0, x1;
    RegTensor<float> x0FP32Layout0, x0FP32Layout1, x1FP32Layout0, x1FP32Layout1;
    RegTensor<U> xQuant0, xQuant1, xQuant2, xQuant3;

    MaskReg maskXQuant0B32, maskXQuant1B32, maskXQuant2B32, maskXQuant3B32;
    MaskReg maskAllB16 = CreateMask<uint16_t, MaskPattern::ALL>();
    MaskReg maskAllB32 = CreateMask<float, MaskPattern::ALL>();

    for (uint16_t i = 0; i < vfLoopNum; i++) {
        maskXQuant0B32 = UpdateMask<float>(xElemNum);
        maskXQuant1B32 = UpdateMask<float>(xElemNum);
        maskXQuant2B32 = UpdateMask<float>(xElemNum);
        maskXQuant3B32 = UpdateMask<float>(xElemNum);

        // Load x: deinterleaved double-load, odd -> x0, even -> x1
        DataCopy<T, PostLiteral::POST_MODE_UPDATE, LoadDist::DIST_DINTLV_B16>(x0, x1, xAddr, vlForT * 2);
        // Load invScale: each element broadcast to 32B (16 bf16 elements), numVRegBlocks elements total
        DataCopy<uint16_t, PostLiteral::POST_MODE_UPDATE, LoadDist::DIST_E2B_B16>(invScale, invScaleInAddr,
                                                                                  numVRegBlocks);
        if constexpr (IsSameType<T, half>::value) {
            // FP16 path: convert x and invScale to FP32, then multiply, then cast to FP8

            // 0. invScale bf16 -> fp32 (each element repeated 16 times in bf16 -> 8 times in fp32)
            Cast<float, bfloat16_t, traitB16ToB32Layout0>(invScaleFP32, (RegTensor<bfloat16_t> &)invScale, maskAllB16);

            // 1. Quantize odd-position elements x0:
            Cast<float, T, traitB16ToB32Layout0>(x0FP32Layout0, x0, maskAllB16);
            Cast<float, T, traitB16ToB32Layout1>(x0FP32Layout1, x0, maskAllB16);
            Mul(x0FP32Layout0, x0FP32Layout0, invScaleFP32, maskAllB32);
            Mul(x0FP32Layout1, x0FP32Layout1, invScaleFP32, maskAllB32);
            Interleave(x0FP32Layout0, x0FP32Layout1, x0FP32Layout0, x0FP32Layout1);

            // 2. Quantize even-position elements x1:
            Cast<float, T, traitB16ToB32Layout0>(x1FP32Layout0, x1, maskAllB16);
            Cast<float, T, traitB16ToB32Layout1>(x1FP32Layout1, x1, maskAllB16);
            Mul(x1FP32Layout0, x1FP32Layout0, invScaleFP32, maskAllB32);
            Mul(x1FP32Layout1, x1FP32Layout1, invScaleFP32, maskAllB32);
            Interleave(x1FP32Layout0, x1FP32Layout1, x1FP32Layout0, x1FP32Layout1);

            // 3. Restore original element order: x0L0, x1L0, x0L1, x1L1
            Interleave(x0FP32Layout0, x1FP32Layout0, x0FP32Layout0, x1FP32Layout0);
            Interleave(x0FP32Layout1, x1FP32Layout1, x0FP32Layout1, x1FP32Layout1);

            // 4. Cast fp32 -> fp8 target type
            Cast<U, float, traitB32ToB8Layout0>(xQuant0, x0FP32Layout0, maskXQuant0B32);
            Cast<U, float, traitB32ToB8Layout0>(xQuant1, x1FP32Layout0, maskXQuant1B32);
            Cast<U, float, traitB32ToB8Layout0>(xQuant2, x0FP32Layout1, maskXQuant2B32);
            Cast<U, float, traitB32ToB8Layout0>(xQuant3, x1FP32Layout1, maskXQuant3B32);
        } else {
            // BF16 path: multiply directly in bf16 domain, then convert to fp32 -> fp8

            // 1. x *= invScale (invScale is broadcast per 32 elements, matching MX block size)
            Mul(x0, x0, (RegTensor<T> &)invScale, maskAllB16);
            Mul(x1, x1, (RegTensor<T> &)invScale, maskAllB16);
            // Restore original order from deinterleaved
            Interleave(x0, x1, x0, x1);

            // 2. x0 part: bf16 -> fp32 -> fp8
            Cast<float, T, traitB16ToB32Layout0>(x0FP32Layout0, x0, maskAllB16);
            Cast<float, T, traitB16ToB32Layout1>(x0FP32Layout1, x0, maskAllB16);
            Interleave(x0FP32Layout0, x0FP32Layout1, x0FP32Layout0, x0FP32Layout1);
            Cast<U, float, traitB32ToB8Layout0>(xQuant0, x0FP32Layout0, maskXQuant0B32);
            Cast<U, float, traitB32ToB8Layout0>(xQuant1, x0FP32Layout1, maskXQuant1B32);

            // 3. x1 part: bf16 -> fp32 -> fp8
            Cast<float, T, traitB16ToB32Layout0>(x1FP32Layout0, x1, maskAllB16);
            Cast<float, T, traitB16ToB32Layout1>(x1FP32Layout1, x1, maskAllB16);
            Interleave(x1FP32Layout0, x1FP32Layout1, x1FP32Layout0, x1FP32Layout1);
            Cast<U, float, traitB32ToB8Layout0>(xQuant2, x1FP32Layout0, maskXQuant2B32);
            Cast<U, float, traitB32ToB8Layout0>(xQuant3, x1FP32Layout1, maskXQuant3B32);
        }

        // Store quantized output: pack 4:1 (fp32 layout -> fp8 byte), OUT_ELE_NUM_ONE_BLK=64 elements per chunk
        DataCopy<int8_t, PostLiteral::POST_MODE_UPDATE, StoreDist::DIST_PACK4_B32>(
            xQuantOutAddr, (RegTensor<int8_t> &)xQuant0, OUT_ELE_NUM_ONE_BLK, maskXQuant0B32);
        DataCopy<int8_t, PostLiteral::POST_MODE_UPDATE, StoreDist::DIST_PACK4_B32>(
            xQuantOutAddr, (RegTensor<int8_t> &)xQuant1, OUT_ELE_NUM_ONE_BLK, maskXQuant1B32);
        DataCopy<int8_t, PostLiteral::POST_MODE_UPDATE, StoreDist::DIST_PACK4_B32>(
            xQuantOutAddr, (RegTensor<int8_t> &)xQuant2, OUT_ELE_NUM_ONE_BLK, maskXQuant2B32);
        DataCopy<int8_t, PostLiteral::POST_MODE_UPDATE, StoreDist::DIST_PACK4_B32>(
            xQuantOutAddr, (RegTensor<int8_t> &)xQuant3, OUT_ELE_NUM_ONE_BLK, maskXQuant3B32);
    }
}

template <typename T, typename U>
class MoeGatherOutMxfp8Quant {
public:
    __aicore__ inline MoeGatherOutMxfp8Quant(){};
    __aicore__ inline void Init(GM_ADDR xAddr, GM_ADDR scaleAddr, GM_ADDR workspaceAddr,
                                GM_ADDR expandedRowIdxAddr, GM_ADDR yAddr, GM_ADDR mxscaleAddr,
                                GM_ADDR expandedScaleAddr,
                                const MoeInitRoutingV3MxQuantArch35TilingData *tilingData, TPipe *tPipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void InitKernelTiling(GM_ADDR workspaceAddr,
                                            const MoeInitRoutingV3MxQuantArch35TilingData *tilingData);
    __aicore__ inline void CopyInExpandedExpertIdx(int64_t progress);
    __aicore__ inline void CopyExpandedXandMXQuant(int64_t progress);
    __aicore__ inline void CopyIn(int64_t srcIdx, int64_t colIdx, int64_t loopCols);
    __aicore__ inline void Compute(uint32_t xElemNum, uint32_t scaleElemNum, uint32_t validScaleElemNum);
    __aicore__ inline void CopyOut(int64_t dstIdx, int64_t colIdx, int64_t loopCols, int64_t loopScaleCols);

private:
    TPipe *pipe_ = nullptr;
    TQue<QuePosition::VECIN, 1> xInQueue_;
    TQue<QuePosition::VECIN, 1> sortedRowIdxInQueue_;
    TQue<QuePosition::VECOUT, 1> xQuantOutQueue_;
    TQue<QuePosition::VECOUT, 1> mxScaleOutQueue_;
    TBuf<QuePosition::VECCALC> maxExpBuffer_;
    TBuf<QuePosition::VECCALC> invScaleBuffer_;
    TQue<QuePosition::VECIN, 1> scaleCopyInQueue_;   // for expanded_scale gather (optional)

    GlobalTensor<T> xInGm_;
    GlobalTensor<uint8_t> yOutGm_;           // y output (fp8, stored as uint8)
    GlobalTensor<uint8_t> mxscaleOutGm_;     // mxscale output (fp8_e8m0, stored as uint8)
    GlobalTensor<T> scaleInGm_;              // scale input (bf16, shape=[n, k], optional)
    GlobalTensor<T> expandedScaleOutGm_;     // expanded_scale output (bf16, shape=[num_expanded_tokens], optional)
    GlobalTensor<int32_t> sortedRowIdxGm_;
    GlobalTensor<int32_t> expertTotalCountGm_;

    const MoeV3Arch35GatherOutComputeTilingData *gatherOutTilingData_ = nullptr;

    int64_t needCoreNum_ = 0;
    int64_t blockIdx_ = 0;
    int64_t cols_ = 0;
    int64_t validScaleCols_ = 0;
    int64_t scaleCols_ = 0;
    int64_t n_ = 0;
    int64_t k_ = 0;
    int64_t perCoreRow_ = 0;
    int64_t currentLoopRows_ = 0;
    int64_t coreRows_ = 0;
    int64_t perLoopRows_ = 0;
    int64_t lastLoopRows_ = 0;
    int64_t rowLoops_ = 0;
    int64_t perLoopCols_ = 0;
    int64_t lastLoopCols_ = 0;
    int64_t colLoops_ = 0;
    int64_t perLoopScaleCols_ = 0;
    int64_t lastLoopValidScaleCols_ = 0;
    int64_t lastLoopScaleCols_ = 0;
    int64_t indicesOffset_ = 0;
    int64_t rowIdxType_ = 0;
    int64_t isInputScale_ = 0;

    // Lower bound of BF16 maxExp for the target fp8 type U
    uint16_t lowerBoundOfB16MaxExp_ = 0;

    // Hardware constants
    const uint32_t vRegSize_ = Ops::Base::GetVRegSize();
    const uint32_t ubBlockSize_ = Ops::Base::GetUbBlockSize();
    const uint32_t vlForB16_ = vRegSize_ / sizeof(T);
    const uint32_t numUbBlocksPerVReg_ = vRegSize_ / ubBlockSize_;  // typically 8
    const uint32_t numElemPerUbBlock_ = ubBlockSize_ / sizeof(T);
};

// ============================================================================
// Init: set up GM tensors, pipe/queue, read tiling parameters
// ============================================================================
template <typename T, typename U>
__aicore__ inline void MoeGatherOutMxfp8Quant<T, U>::Init(
    GM_ADDR xAddr, GM_ADDR scaleAddr, GM_ADDR workspaceAddr,
    GM_ADDR expandedRowIdxAddr, GM_ADDR yAddr, GM_ADDR mxscaleAddr,
    GM_ADDR expandedScaleAddr,
    const MoeInitRoutingV3MxQuantArch35TilingData *tilingData, TPipe *tPipe)
{
    // Note: OVERFLOW_MODE_CTRL is already set to 0 by the top-level _apt.cpp before Init() is called.

    pipe_ = tPipe;
    blockIdx_ = GetBlockIdx();
    InitKernelTiling(workspaceAddr, tilingData);

    // GM input: x
    xInGm_.SetGlobalBuffer((__gm__ T *)xAddr);

    // GM outputs: y (fp8) and mxscale (fp8_e8m0)
    yOutGm_.SetGlobalBuffer((__gm__ uint8_t *)yAddr);
    mxscaleOutGm_.SetGlobalBuffer((__gm__ uint8_t *)mxscaleAddr);

    // GM input: scale (optional, bf16 routing weight, shape=[n, k])
    scaleInGm_.SetGlobalBuffer((__gm__ T *)scaleAddr);

    // GM output: expandedScale (optional, bf16 routing weight, shape=[num_expanded_tokens])
    expandedScaleOutGm_.SetGlobalBuffer((__gm__ T *)expandedScaleAddr);

    // sorted_row_idx comes from workspace
    // workspace layout: [sorted_expert_idx (n*k) | sorted_row_idx (n*k) | expert_total_count (actualExpertNum) | ...]
    // For SCATTER mode, sorted_row_idx is from expandedRowIdx output; for GATHER mode, from workspace
    if (rowIdxType_ == SCATTER) {
        sortedRowIdxGm_.SetGlobalBuffer((__gm__ int32_t *)expandedRowIdxAddr + blockIdx_ * perCoreRow_,
                                        Align(perCoreRow_, sizeof(int32_t)));
    } else {
        // GATHER: sorted_row_idx is in workspace after sorted_expert_idx
        sortedRowIdxGm_.SetGlobalBuffer(
            (__gm__ int32_t *)workspaceAddr + Align(n_ * k_, sizeof(int32_t)) + blockIdx_ * perCoreRow_,
            Align(perCoreRow_, sizeof(int32_t)));
    }

    // Init buffers
    // sortedRowIdxInQueue: holds perLoopRows indices (int32)
    pipe_->InitBuffer(sortedRowIdxInQueue_, 1, AlignBytes(perLoopRows_, sizeof(int32_t)));
    // xInQueue: holds one row slice of perLoopCols elements (T = bf16/fp16, 2 bytes each)
    pipe_->InitBuffer(xInQueue_, 1, AlignBytes(perLoopCols_, sizeof(T)));
    // xQuantOutQueue: holds quantized output (fp8, 1 byte each), but stored as 4:1 pack from fp32
    pipe_->InitBuffer(xQuantOutQueue_, 1, AlignBytes(perLoopCols_ / 4, sizeof(int8_t)) * 4);
    // mxScaleOutQueue: holds mxscale output (fp8_e8m0, 1 byte each per MX block)
    pipe_->InitBuffer(mxScaleOutQueue_, 1, AlignBytes(perLoopScaleCols_, sizeof(int8_t)));
    // maxExpBuffer: intermediate buffer for maxExp computation (uint16, one per MX block)
    pipe_->InitBuffer(maxExpBuffer_, AlignBytes(perLoopScaleCols_, sizeof(T)));
    // invScaleBuffer: intermediate buffer for invScale computation (uint16/bf16, one per MX block)
    pipe_->InitBuffer(invScaleBuffer_, AlignBytes(perLoopScaleCols_, sizeof(T)));

    // scaleCopyInQueue: holds one scale value (T = bf16/fp16) for expanded_scale gather
    if (isInputScale_) {
        pipe_->InitBuffer(scaleCopyInQueue_, 1, AlignBytes(1, sizeof(T)));
    }

    // Set lower bound of maxExp based on target fp8 type
    if constexpr (IsSameType<U, fp8_e4m3fn_t>::value) {
        lowerBoundOfB16MaxExp_ = LOWER_BOUND_OF_MAX_EXP_FOR_E4M3;
    } else {
        lowerBoundOfB16MaxExp_ = LOWER_BOUND_OF_MAX_EXP_FOR_E5M2;
    }
}

// ============================================================================
// InitKernelTiling: extract tiling parameters and compute loop bounds
// ============================================================================
template <typename T, typename U>
__aicore__ inline void MoeGatherOutMxfp8Quant<T, U>::InitKernelTiling(
    GM_ADDR workspaceAddr,
    const MoeInitRoutingV3MxQuantArch35TilingData *tilingData)
{
    gatherOutTilingData_ = &(tilingData->gatherOutComputeParamsOp);
    cols_ = tilingData->cols;
    validScaleCols_ = Ops::Base::CeilDiv<int64_t>(cols_, MX_BLOCK_SIZE);
    scaleCols_ = Ops::Base::CeilAlign<int64_t>(validScaleCols_, 2);  // align to even
    n_ = tilingData->n;
    k_ = tilingData->k;
    rowIdxType_ = tilingData->rowIdxType;
    isInputScale_ = tilingData->isInputScale;

    // Read expertTotalCount from workspace
    // workspace layout: [sorted_expert_idx (n*k) | sorted_row_idx (n*k) | expert_total_count (actualExpertNum)]
    int64_t actualExpertNum = tilingData->actualExpertNum;
    expertTotalCountGm_.SetGlobalBuffer(
        (__gm__ int32_t *)workspaceAddr + Align(n_ * k_, sizeof(int32_t)) * 2 +
            Align(actualExpertNum, sizeof(int32_t)),
        actualExpertNum);
    AscendC::DataCacheCleanAndInvalid<int32_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
                                      AscendC::DcciDst::CACHELINE_OUT>(expertTotalCountGm_);
    int64_t expertTotalCount = expertTotalCountGm_.GetValue(0);

    // Core split: evenly distribute expanded tokens across cores
    perCoreRow_ = Ceil(expertTotalCount, tilingData->coreNum);
    needCoreNum_ = Ceil(expertTotalCount, perCoreRow_);
    int64_t lastCoreIndicesElements = expertTotalCount - (needCoreNum_ - 1) * perCoreRow_;

    // Inner core split: determine perLoopRows based on tiling params
    int64_t originPerLoopElements;
    if (blockIdx_ == needCoreNum_ - 1) {
        coreRows_ = lastCoreIndicesElements;
        originPerLoopElements = gatherOutTilingData_->lastCorePerLoopIndicesElements;
    } else {
        coreRows_ = perCoreRow_;
        originPerLoopElements = gatherOutTilingData_->perCorePerLoopIndicesElements;
    }
    perLoopRows_ = Min(coreRows_, originPerLoopElements);
    rowLoops_ = Ceil(coreRows_, perLoopRows_);
    lastLoopRows_ = coreRows_ - (rowLoops_ - 1) * perLoopRows_;

    // Column split: from tiling data (perLoopCols already aligned to MX_BLOCK_SIZE=32)
    perLoopCols_ = gatherOutTilingData_->perLoopCols;
    lastLoopCols_ = gatherOutTilingData_->lastLoopCols;
    colLoops_ = gatherOutTilingData_->colsLoops;
    perLoopScaleCols_ = perLoopCols_ / MX_BLOCK_SIZE;
    lastLoopValidScaleCols_ = validScaleCols_ - (colLoops_ - 1) * perLoopScaleCols_;
    lastLoopScaleCols_ = scaleCols_ - (colLoops_ - 1) * perLoopScaleCols_;
}

// ============================================================================
// Process: outer row loop x inner col loop
// ============================================================================
template <typename T, typename U>
__aicore__ inline void MoeGatherOutMxfp8Quant<T, U>::Process()
{
    if (blockIdx_ < needCoreNum_) {
        // Process all row loops except the last
        currentLoopRows_ = perLoopRows_;
        for (int64_t loop = 0; loop < rowLoops_ - 1; loop++) {
            CopyInExpandedExpertIdx(loop);
            CopyExpandedXandMXQuant(loop);
        }

        // Process the last row loop (may have fewer rows)
        currentLoopRows_ = lastLoopRows_;
        CopyInExpandedExpertIdx(rowLoops_ - 1);
        CopyExpandedXandMXQuant(rowLoops_ - 1);
    }
}

// ============================================================================
// CopyInExpandedExpertIdx: load a batch of sorted_row_idx from GM to UB
// ============================================================================
template <typename T, typename U>
__aicore__ inline void MoeGatherOutMxfp8Quant<T, U>::CopyInExpandedExpertIdx(int64_t progress)
{
    indicesOffset_ = progress * perLoopRows_;
    LocalTensor<int32_t> indicesLocal = sortedRowIdxInQueue_.AllocTensor<int32_t>();
    DataCopyExtParams dataCopyParams{1, static_cast<uint32_t>(currentLoopRows_ * sizeof(int32_t)), 0, 0, 0};
    DataCopyPadExtParams<int32_t> dataCopyPadParams{false, 0, 0, 0};
    DataCopyPad(indicesLocal, sortedRowIdxGm_[indicesOffset_], dataCopyParams, dataCopyPadParams);
    sortedRowIdxInQueue_.EnQue<int32_t>(indicesLocal);
}

// ============================================================================
// CopyExpandedXandMXQuant: for each row in current batch, do col-loop of CopyIn->Compute->CopyOut
// ============================================================================
template <typename T, typename U>
__aicore__ inline void MoeGatherOutMxfp8Quant<T, U>::CopyExpandedXandMXQuant(int64_t progress)
{
    LocalTensor<int32_t> indicesLocal = sortedRowIdxInQueue_.DeQue<int32_t>();
    SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);
    for (int64_t index = 0; index < currentLoopRows_; index++) {
        int32_t srcIdx = indicesLocal.GetValue(index);
        int64_t dstIdx = perCoreRow_ * blockIdx_ + perLoopRows_ * progress + index;

        // Gather expanded_scale: read scale[srcIdx] from GM and write to expandedScale[dstIdx]
        // srcIdx is in the (n*k) domain; scale shape=[n, k] stored flat, so scale[srcIdx] is correct
        if (isInputScale_) {
            LocalTensor<T> scaleLocal = scaleCopyInQueue_.AllocTensor<T>();
            DataCopyExtParams scaleCopyInParams{1, static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
            DataCopyPadExtParams<T> scalePadParams{false, 0, 0, 0};
            DataCopyPad(scaleLocal, scaleInGm_[srcIdx], scaleCopyInParams, scalePadParams);
            scaleCopyInQueue_.EnQue(scaleLocal);

            LocalTensor<T> scaleOutLocal = scaleCopyInQueue_.DeQue<T>();
            DataCopyExtParams scaleCopyOutParams{1, static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
            DataCopyPad(expandedScaleOutGm_[dstIdx], scaleOutLocal, scaleCopyOutParams);
            scaleCopyInQueue_.FreeTensor(scaleOutLocal);
        }

        for (int64_t j = 0; j < colLoops_; j++) {
            int64_t loopCols = (j == colLoops_ - 1) ? lastLoopCols_ : perLoopCols_;
            uint32_t loopScaleCols = (j == colLoops_ - 1) ? lastLoopScaleCols_ : perLoopScaleCols_;
            uint32_t loopValidScaleCols = (j == colLoops_ - 1) ? lastLoopValidScaleCols_ : perLoopScaleCols_;
            // srcIdx is expanded index (n*k domain), divide by k to get original row in x
            CopyIn(srcIdx / k_, j, loopCols);
            Compute(loopCols, loopScaleCols, loopValidScaleCols);
            CopyOut(dstIdx, j, loopCols, loopScaleCols);
        }
    }
}

// ============================================================================
// CopyIn: gather one row slice from x[GM] to UB based on sorted_row_idx
// ============================================================================
template <typename T, typename U>
__aicore__ inline void MoeGatherOutMxfp8Quant<T, U>::CopyIn(int64_t srcIdx, int64_t colIdx, int64_t loopCols)
{
    LocalTensor<T> inLocal = xInQueue_.AllocTensor<T>();
    DataCopyExtParams copyInParam = {1, static_cast<uint32_t>(loopCols * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams = {false, 0, 0, 0};
    // If loopCols is not aligned to MX_BLOCK_SIZE, pad to ensure complete MX blocks
    int64_t loopColsTail = loopCols % MX_BLOCK_SIZE;
    if (loopColsTail != 0) {
        padParams.isPad = true;
        if (loopColsTail > numElemPerUbBlock_) {
            padParams.rightPadding = numElemPerUbBlock_ - (loopColsTail - numElemPerUbBlock_);
        } else {
            padParams.rightPadding = numElemPerUbBlock_;
        }
    }
    DataCopyPad(inLocal, xInGm_[srcIdx * cols_ + colIdx * perLoopCols_], copyInParam, padParams);
    xInQueue_.EnQue(inLocal);
}

// ============================================================================
// Compute: MX-FP8 quantization via 3-step VF_CALL (maxExp -> scale -> quantize)
// ============================================================================
template <typename T, typename U>
__aicore__ inline void MoeGatherOutMxfp8Quant<T, U>::Compute(uint32_t xElemNum, uint32_t scaleElemNum,
                                                              uint32_t validScaleElemNum)
{
    // Deque input from xInQueue
    LocalTensor<T> xLocal = xInQueue_.DeQue<T>();
    auto xLocalAddr = reinterpret_cast<__ubuf__ T *>(xLocal.GetPhyAddr());

    // Alloc output tensors
    LocalTensor<int8_t> xQuantLocal = xQuantOutQueue_.AllocTensor<int8_t>();
    auto xQuantLocalAddr = reinterpret_cast<__ubuf__ int8_t *>(xQuantLocal.GetPhyAddr());
    LocalTensor<uint16_t> mxScaleLocal = mxScaleOutQueue_.AllocTensor<uint16_t>();
    auto mxScaleLocalAddr = reinterpret_cast<__ubuf__ uint16_t *>(mxScaleLocal.GetPhyAddr());

    // Get temporary buffers
    LocalTensor<uint16_t> maxExpLocal = maxExpBuffer_.Get<uint16_t>();
    auto maxExpLocalAddr = reinterpret_cast<__ubuf__ uint16_t *>(maxExpLocal.GetPhyAddr());
    LocalTensor<uint16_t> invScaleLocal = invScaleBuffer_.Get<uint16_t>();
    auto invScaleLocalAddr = reinterpret_cast<__ubuf__ uint16_t *>(invScaleLocal.GetPhyAddr());

    // Compute VF loop counts: double-load processes vlForB16_*2 elements per iteration
    uint16_t vfLoopNumForX = (xElemNum + vlForB16_ * 2 - 1) / (vlForB16_ * 2);
    uint16_t vfLoopNumForScale = (scaleElemNum + vlForB16_ - 1) / vlForB16_;

    // Step 1: Compute per-MX-block max exponent
    VF_CALL<vfComputeMaxExp<T, U>>(xLocalAddr, maxExpLocalAddr, xElemNum, vfLoopNumForX, vlForB16_,
                                   numUbBlocksPerVReg_);
    // Step 2: Compute mxScale (fp8_e8m0) and invScale (bf16) from maxExp
    VF_CALL<vfComputeScale<T, U>>(maxExpLocalAddr, mxScaleLocalAddr, invScaleLocalAddr, scaleElemNum,
                                  validScaleElemNum, vfLoopNumForScale, vlForB16_, lowerBoundOfB16MaxExp_);
    // Step 3: Quantize x using invScale -> fp8 output
    VF_CALL<vfComputeData<T, U>>(xLocalAddr, invScaleLocalAddr, xQuantLocalAddr, xElemNum, vfLoopNumForX, vlForB16_,
                                 numUbBlocksPerVReg_);

    // Free input
    xInQueue_.FreeTensor(xLocal);

    // Enque outputs
    xQuantOutQueue_.EnQue(xQuantLocal);
    mxScaleOutQueue_.EnQue(mxScaleLocal);
}

// ============================================================================
// CopyOut: write quantized y and mxscale from UB back to GM
// ============================================================================
template <typename T, typename U>
__aicore__ inline void MoeGatherOutMxfp8Quant<T, U>::CopyOut(int64_t dstIdx, int64_t colIdx,
                                                              int64_t loopCols, int64_t loopScaleCols)
{
    LocalTensor<uint8_t> mxScaleLocal = mxScaleOutQueue_.DeQue<uint8_t>();
    LocalTensor<uint8_t> outLocal = xQuantOutQueue_.DeQue<uint8_t>();

    // Write quantized y (fp8) to GM
    DataCopyExtParams copyOutParams = {1, static_cast<uint32_t>(loopCols * sizeof(uint8_t)), 0, 0, 0};
    DataCopyPad<uint8_t>(yOutGm_[dstIdx * cols_ + colIdx * perLoopCols_], outLocal, copyOutParams);

    // Write mxscale (fp8_e8m0) to GM
    DataCopyExtParams copyScaleParams = {1, static_cast<uint32_t>(loopScaleCols * sizeof(uint8_t)), 0, 0, 0};
    DataCopyPad<uint8_t>(mxscaleOutGm_[dstIdx * scaleCols_ + colIdx * perLoopScaleCols_], mxScaleLocal,
                         copyScaleParams);

    xQuantOutQueue_.FreeTensor(outLocal);
    mxScaleOutQueue_.FreeTensor(mxScaleLocal);
}

} // namespace MoeInitRoutingV3MxQuantNs
#endif // MOE_INIT_ROUTING_V3_MX_QUANT_GATHER_MXFP8_QUANT_H
