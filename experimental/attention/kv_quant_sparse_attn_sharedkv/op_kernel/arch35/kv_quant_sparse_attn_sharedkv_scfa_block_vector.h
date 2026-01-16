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
 * \file flash_attention_score_block_vec_base_scfa.h
 * \brief
 */
#ifndef KV_QUANT_SPARSE_ATTN_SHAREDKV_SCFA_BLOCK_VECTOR_H
#define KV_QUANT_SPARSE_ATTN_SHAREDKV_SCFA_BLOCK_VECTOR_H

#include "util_regbase.h"
#include "kv_quant_sparse_attn_sharedkv_common_arch35.h"
#include "common/buffers_policy.h"
#include "common/buffer_manager.h"
#include "common/buffer.h"
#include "kernel_operator_list_tensor_intf.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"

#include "vf/vf_mul_sel_softmaxflashv2_cast_nz_scfa.h"
#include "vf/vf_flashupdate_new_scfa.h"

using namespace AscendC;
using namespace SCFaVectorApi;
using namespace AscendC::Impl::Detail;
using namespace optiling;
using namespace optiling::detail;
using namespace regbaseutil;
using namespace matmul;

namespace BaseApi {
TEMPLATES_DEF
class SCFABlockVec {
public:
    /* =================编译期常量的基本块信息================= */
    static constexpr uint32_t s1BaseSize = 64; 
    static constexpr uint32_t s2BaseSize = 128; 
    static constexpr uint32_t vec1Srcstride = (s1BaseSize >> 1) + 1; 
    static constexpr uint32_t dVTemplateType = 512;
    static constexpr uint32_t dTemplateAlign64 = Align64Func((uint16_t)dVTemplateType);
    static constexpr float R0 = 1.0f;
    static constexpr uint64_t SYNC_SINKS_BUF_FLAG = 6;

    SasMetaData metadataVecLocal;
    bool isSinks = false;
    // ==================== Functions ======================
    __aicore__ inline SCFABlockVec() {};
    __aicore__ inline void InitVecBlock(TPipe *pipe, const KvQuantSparseAttnSharedkvTilingData *__restrict tiling,
        CVSharedParams &sharedParams, int32_t aicIdx, uint8_t subBlockIdx, SasMetaData &metadataLocal) {
        if ASCEND_IS_AIV {
            tPipe = pipe;
            tilingData = tiling;
            metadataVecLocal = metadataLocal;
            this->InitCubeVecSharedParams(sharedParams, aicIdx, subBlockIdx);
            this->GetExtremeValue(this->negativeFloatScalar);
        }
    }

    // 初始化LocalTensor
    __aicore__ inline void InitLocalBuffer(TPipe *pipe, ConstInfo &constInfo);
    // 初始化attentionOutGM
    __aicore__ inline void CleanOutput(__gm__ uint8_t *attentionOut, ConstInfo &constInfo, __gm__ uint8_t *cuSeqlensQ);
    __aicore__ inline void InitGlobalBuffer(__gm__ uint8_t *oriKV, __gm__ uint8_t *cmpKV, __gm__ uint8_t *cmpSparseIndices,
        __gm__ uint8_t *oriBlockTable, __gm__ uint8_t *cmpBlockTable, __gm__ uint8_t *cuSeqlensQ, __gm__ uint8_t *sequsedKv,
        __gm__ uint8_t *sinks);
    __aicore__ inline void InitOutputSingleCore(ConstInfo &constInfo, __gm__ uint8_t *cuSeqlensQ);

    // ==================== Vector0 ======================
    __aicore__ inline void ProcessSparseKv(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputL1, const RunInfo &runInfo);
    __aicore__ inline void ProcessVec0(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputL1, const RunInfo &runInfo);
    __aicore__ inline void ProcessNotSparseKv(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputL1, const RunInfo &runInfo);
    __aicore__ inline int64_t GetKeyBNBOffset(int64_t realS2Idx, const RunInfo &runInfo, int64_t s2IdLimit);
    __aicore__ inline void GetRealS2Idx(int64_t s2GmOffset, int64_t &realS2Idx, int64_t topkGmBaseOffset,
                                        const RunInfo &runInfo);
    __aicore__ inline void CopyInKvNotSparse(LocalTensor<KV_T> kvMergUb, int64_t v0Loop, int64_t dealRow,
        int64_t s2StartOffset, const RunInfo &runInfo);
    __aicore__ inline void CopyInKvSparse(int64_t &mte2Size, int64_t mte3Size, int64_t mergeMte3Idx, int64_t realS2Idx1,
                                    int64_t realS2Idx2, const RunInfo &runInfo);
    __aicore__ inline void DequantKv(LocalTensor<Q_T> antiKvTensorAsB16, LocalTensor<KV_T> srcTensor, int64_t dealRow, int64_t s2ProcessBaseSize);
    __aicore__ inline void CopyOutKvUb2L1(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputL1,
        LocalTensor<Q_T> antiKvTensorAsB16, int64_t v0Loop, int64_t dealRow, int64_t s2StartOffset);
    __aicore__ inline void CopyOutMrgeResult(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputL1, int64_t mte2Size, int64_t mte3Size, int64_t s2StartGmOffset,
                                             int64_t mergeMte3Idx, const RunInfo &runInfo);
    __aicore__ inline void CopyInSingleKv(int64_t &mte2Size, int64_t mte3Size, int64_t mergeMte3Idx, int64_t realS2Idx,
                                          int64_t keyBNBOffset, int64_t s2IdLimit, const RunInfo &runInfo);
    // ==================== Vector1 ======================
    __aicore__ inline void ProcessVec1(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputBuf,
        Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm1ResBuf, RunInfo &runInfo,
        ConstInfo &constInfo);
    __aicore__ inline void CopySinksIn(const LocalTensor<T>& maxTensor, int64_t sinksGmOffset ,uint32_t elementNum);

    using mm2ResPos = Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>;
    __aicore__ inline void ProcessVec2(mm2ResPos &bmm2ResBuf, RunInfo &runInfo,
        ConstInfo &constInfo);

    TPipe *tPipe;
    // V0
    // BLOCK和REPEAT的字节数
    static constexpr uint64_t BYTE_BLOCK = 32UL;
    static constexpr uint32_t REPEAT_BLOCK_BYTE = 256U;
    // BLOCK和REPEAT的FP32元素数
    static constexpr uint32_t FP32_BLOCK_ELEMENT_NUM = BYTE_BLOCK / sizeof(float);
    static constexpr uint32_t FP32_REPEAT_ELEMENT_NUM = REPEAT_BLOCK_BYTE / sizeof(float);
    // V0
    const KvQuantSparseAttnSharedkvTilingData *__restrict tilingData;
    
    // 在CleanOutput中初始化
    GlobalTensor<OUTPUT_T> attentionOutGm;
    GlobalTensor<half> attentionOutInitGm;
    GlobalTensor<KV_T> oriKVGm;
    GlobalTensor<KV_T> cmpKVGm;
    GlobalTensor<KV_T> keyGm_;
    GlobalTensor<int32_t> cmpSparseIndicesGm;
    GlobalTensor<int32_t> oriBlockTableGm;
    GlobalTensor<int32_t> cmpBlockTableGm;
    GlobalTensor<int32_t> blockTableGm_;
    GlobalTensor<T> sinksGm;
    // __gm__ int64_t *actualSeqQlenAddr;
    // __gm__ int64_t *actualSeqKvlenAddr;

    /* =====================V侧UB变量==================== */
    TBuf<> commonTBuf; // common的复用空间
    TQue<QuePosition::VECOUT, 1> stage1OutQue[2];
    TQue<QuePosition::VECIN, 2> stage0InQue; // for v0 input
    TQue<QuePosition::VECOUT, 2> stage0OutQue; // for v0 output
    TBuf<> stage2OutBuf;
    TEventID mte3ToVId[2]; // 存放MTE3_V的eventId, 2份表示可能存在pingpong
    TEventID vToMte3Id[2]; // 存放V_MTE3的eventId, 2份表示可能存在pingpong
    TBuf<> softmaxMaxBuf[2];
    TBuf<> softmaxSumBuf[2];
    TBuf<> softmaxExpBuf[2]; 
    /* =================初始化后不变的信息================= */
    T negativeFloatScalar;
protected:
/* VEC2_RES_T 表示bmm2ResUb当前的类型，VEC2_RES_T = Q_T那么不需要做Cast。另外，无效行场景当前默认需要做Cast */
    using VEC2_RES_T = T;
    template <typename VEC2_RES_T>
    __aicore__ inline void Bmm2DataCopyOut(RunInfo &runInfo, ConstInfo &constInfo,
        LocalTensor<VEC2_RES_T> &vec2ResUb, int64_t vec2S1Idx, int64_t vec2CalcSize = 0);
    template <typename VEC2_RES_T>
    __aicore__ inline void CopyOutAttentionOut(
    RunInfo &runInfo, ConstInfo &constInfo, LocalTensor<VEC2_RES_T> &vec2ResUb, int64_t vec2S1Idx, int64_t vec2CalcSize);

private:
    __aicore__ inline void SoftmaxInitBuffer();
    __aicore__ inline void InitCubeVecSharedParams(CVSharedParams &sharedParams, int32_t aicIdx, uint8_t subBlockIdx);
    __aicore__ inline void GetExtremeValue(T &negativeScalar);

    // for V0
    static constexpr uint64_t MERGE_CACHE_GM_BUF_NUM = 4;
    static constexpr uint64_t SYNC_OUTPUT_BUF1_FLAG = 4;
    static constexpr uint64_t SYNC_OUTPUT_BUF2_FLAG = 5;
    static constexpr uint32_t INPUT1_BUFFER_OFFSET = ConstInfo::BUFFER_SIZE_BYTE_32K;
    static constexpr uint32_t LIMIT_DEAL_ROW = 16U;

    ConstInfo constInfo_ = {};

    GlobalTensor<int32_t> actualSeqLengthsQGm;
    GlobalTensor<int32_t> actualSeqLengthsKVGm;

    GlobalTensor<Q_T> kvMergeGm_; // todo :改成L1
    GlobalTensor<int32_t> kvValidSizeGm_;

    // ================================Local Buffer区====================================
    // v0
    TBuf<> v0ValidSizeBuff;  // 8K
    TBuf<> dequantScaleBuff_;         // 32K

    LocalTensor<int32_t> v0ValidSizeUb_;
    // ============v0 compute, for debug==========
    TBuf<> inputBuff2;  // 32K
    TBuf<> outputBuff1; // 32K
    TBuf<> outputBuff2; // 4K

    TBuf<> tmpBuff2;         // 8K
    // ============v0 compute, for debug==========
};

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockVec<TEMPLATE_ARGS>::GetRealS2Idx(int64_t s2GmOffset, int64_t &realS2Idx,
                                                              int64_t topkGmBaseOffset, const RunInfo &runInfo)
{
    int64_t topkGmIdx = (s2GmOffset + runInfo.s2LoopCount * constInfo_.s2BaseSize) / constInfo_.sparseBlockSize;
    if (unlikely(topkGmIdx >= constInfo_.sparseBlockCount)) {
        realS2Idx = -1;
        return;
    }
    realS2Idx = cmpSparseIndicesGm.GetValue(topkGmBaseOffset + topkGmIdx) * static_cast<int64_t>(constInfo_.sparseBlockSize) +
                static_cast<int64_t>((s2GmOffset + runInfo.s2LoopCount * constInfo_.s2BaseSize) % constInfo_.sparseBlockSize);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline int64_t SCFABlockVec<TEMPLATE_ARGS>::GetKeyBNBOffset(int64_t realS2Idx,
                                                                    const RunInfo &runInfo, int64_t s2IdLimit)
{
    if (realS2Idx < 0 || realS2Idx >= s2IdLimit) {
        return -1;
    }
    int64_t realKeyBNBOffset = 0;
    if constexpr (isPa) {
        int64_t blkTableIdx = realS2Idx / constInfo_.blockSize;
        int64_t blkTableOffset = realS2Idx % constInfo_.blockSize;
        realKeyBNBOffset = blockTableGm_.GetValue(runInfo.boIdx * constInfo_.maxBlockNumPerBatch + blkTableIdx) *
                                static_cast<int64_t>(constInfo_.blockSize) *
                                static_cast<int64_t>(constInfo_.n2Size) +
                                blkTableOffset;
    } else {
        // realKeyBNBOffset = (runInfo.tensorBOffset +
        realKeyBNBOffset = (runInfo.keyOffset + // todo check if keyOffset is tensorBOffset
                           realS2Idx * constInfo_.n2Size * constInfo_.dSize) /
                           constInfo_.dSize;
    }
    return realKeyBNBOffset;
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void
SCFABlockVec<TEMPLATE_ARGS>::CopyInSingleKv(int64_t &mte2Size, int64_t mte3Size, int64_t mergeMte3Idx, int64_t realS2Idx,
                                         int64_t keyBNBOffset, int64_t s2IdLimit, const RunInfo &runInfo)
{
#if 0
    if (keyBNBOffset < 0) {
        return;
    }
    int64_t validS2Count =
        (realS2Idx + constInfo_.sparseBlockSize > s2IdLimit ? s2IdLimit - realS2Idx : constInfo_.sparseBlockSize);
    DataCopyExtParams intriParams;
    
    intriParams.blockCount = validS2Count;
    intriParams.dstStride = 0;
    intriParams.srcStride = 0;
    DataCopyPadExtParams<KV_T> padParams;
    // 当前仅支持COMBINE模式
    // if (constInfo_.quantScaleRepoMode == QUANT_SCALE_REPO_MODE::COMBINE) {
        uint32_t combineBytes = (constInfo_.dSizeNope * sizeof(KV_T) + constInfo_.dSizeRope * sizeof(Q_T) +
            constInfo_.dSizeNope / constInfo_.tileSize * sizeof(T));
        intriParams.blockLen = combineBytes;
        uint32_t combineDim = combineBytes / sizeof(KV_T);
        uint32_t combineDimAlign = CeilAlign(combineBytes, ConstInfo::BUFFER_SIZE_BYTE_32B) / sizeof(KV_T);
        padParams.isPad = true;
        padParams.leftPadding = 0;
        padParams.rightPadding = combineDimAlign - combineDim;
        padParams.paddingValue = 0;
        DataCopyPad(kvMergUb_[mergeMte3Idx % 2 * INPUT1_BUFFER_OFFSET / sizeof(KV_T)  + (mte2Size - mte3Size) *
                combineDimAlign], keyGm_[keyBNBOffset * combineDim], intriParams, padParams);
    // }
    mte2Size += validS2Count;
#endif
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockVec<TEMPLATE_ARGS>::CopyInKvSparse(int64_t &mte2Size, int64_t mte3Size, int64_t mergeMte3Idx,
                                                          int64_t realS2Idx1, int64_t realS2Idx2,
                                                          const RunInfo &runInfo)
{
#if 0
    // int64_t s2IdLimit = runInfo.curActualSeqLenOri;
    int64_t s2IdLimit = runInfo.s2RealSize;
    if (constInfo_.sparseMode == 3) {
        // s2IdLimit = runInfo.curActualSeqLenOri - runInfo.actualS1Size + runInfo.gS1Idx / constInfo_.gSize + 1; // todo del gS1Idx
        s2IdLimit = runInfo.s2RealSize - runInfo.actualS1Size + runInfo.s1oIdx + 1; // todo del gS1Idx
    }

    int64_t keyBNBOffset1 = GetKeyBNBOffset(realS2Idx1, runInfo, s2IdLimit);
    int64_t keyBNBOffset2 = GetKeyBNBOffset(realS2Idx2, runInfo, s2IdLimit);
    if (unlikely(keyBNBOffset1 < 0 && keyBNBOffset2 < 0)) {
        return;
    }

    int64_t sparseBlockSrcStride =
        ((keyBNBOffset1 > keyBNBOffset2 ? (keyBNBOffset1 - keyBNBOffset2) :
        (keyBNBOffset2 - keyBNBOffset1)) - constInfo_.sparseBlockSize);
    uint32_t combineBytes = (constInfo_.dSizeNope * sizeof(KV_T) +
                             constInfo_.dSizeRope * sizeof(Q_T) +
                             constInfo_.dSizeNope / constInfo_.tileSize * sizeof(T));
    int64_t keySrcStride = sparseBlockSrcStride * combineBytes;
    if (unlikely(keySrcStride >= INT32_MAX || keySrcStride < 0 ||
        realS2Idx1 + constInfo_.sparseBlockSize >= s2IdLimit ||
        realS2Idx2 + constInfo_.sparseBlockSize >= s2IdLimit) ||
        constInfo_.sparseBlockSize > 1) {
        // stride溢出、stride为负数、s2超长等异常场景，还原成2条搬运指令
        CopyInSingleKv(mte2Size, mte3Size, mergeMte3Idx, realS2Idx1, keyBNBOffset1, s2IdLimit, runInfo);
        CopyInSingleKv(mte2Size, mte3Size, mergeMte3Idx, realS2Idx2, keyBNBOffset2, s2IdLimit, runInfo);
    } else {
        DataCopyExtParams intriParams;
        intriParams.blockCount = (keyBNBOffset1 >= 0) + (keyBNBOffset2 >= 0);
        intriParams.dstStride = 0;
        intriParams.srcStride = keySrcStride;
        DataCopyPadExtParams<KV_T> padParams;

        int64_t startGmOffset = keyBNBOffset1 > -1 ? keyBNBOffset1 : keyBNBOffset2;
        if (keyBNBOffset2 > -1 && keyBNBOffset2 < keyBNBOffset1) {
            startGmOffset = keyBNBOffset2;
        }

        // 当前仅支持COMBINE模式
        // if (constInfo_.quantScaleRepoMode == QUANT_SCALE_REPO_MODE::COMBINE) {
            intriParams.blockLen = constInfo_.sparseBlockSize * combineBytes;
            uint32_t combineDim = combineBytes / sizeof(KV_T);
            uint32_t combineDimAlign = CeilAlign(combineBytes, ConstInfo::BUFFER_SIZE_BYTE_32B) / sizeof(KV_T);
            padParams.isPad = true;
            padParams.leftPadding = 0;
            padParams.rightPadding = combineDimAlign - combineDim;
            padParams.paddingValue = 0;
            DataCopyPad(kvMergUb_[mergeMte3Idx % 2 * INPUT1_BUFFER_OFFSET / sizeof(KV_T) + (mte2Size - mte3Size) *
                        combineDimAlign], keyGm_[startGmOffset * combineDim], intriParams, padParams);
        // }
        mte2Size += ((keyBNBOffset1 > -1) + (keyBNBOffset2 > -1)) * constInfo_.sparseBlockSize;
    }
#endif
}

// fp8->fp32
static constexpr MicroAPI::CastTrait castTraitFp8_1 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
                                                       MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
// fp8->fp32
static constexpr MicroAPI::CastTrait castTraitFp8_2 = {MicroAPI::RegLayout::ONE, MicroAPI::SatMode::UNKNOWN,
                                                       MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
// fp32->fp16
static constexpr MicroAPI::CastTrait castTraitFp8_3 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
                                                       MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
// fp32->fp16
static constexpr MicroAPI::CastTrait castTraitFp8_4 = {MicroAPI::RegLayout::ONE, MicroAPI::SatMode::NO_SAT,
                                                       MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
template <typename Q_T, typename KV_T>
__simd_vf__ void CastScaleImpl(__ubuf__ float* ubDstAddr, __ubuf__ int8_t* ubSrcAddr, uint32_t dealRowCount)
{
    MicroAPI::RegTensor<fp8_e8m0_t> vScale0;
    MicroAPI::RegTensor<fp8_e8m0_t> vScale1;
    MicroAPI::RegTensor<bfloat16_t> vScalebf16Res0;
    MicroAPI::RegTensor<bfloat16_t> vScalebf16Res1;
    MicroAPI::RegTensor<float> vScalefp32Res0;
    MicroAPI::RegTensor<float> vScalefp32Res1;
    __ubuf__ int8_t* ubScaleSrcAddrTemp = ubSrcAddr;
    __ubuf__ float* ubDstAddrTmp = ubDstAddr;
    MicroAPI::MaskReg bf16TypeMaskAll = MicroAPI::CreateMask<bfloat16_t, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg fp32MaskAll = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
    for (uint16_t i = 0; i < static_cast<uint16_t>(dealRowCount); i++) {
        // load scale
        MicroAPI::LoadAlign<int8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_UNPACK4_B8>(
            (MicroAPI::RegTensor<int8_t>&)vScale0, ubScaleSrcAddrTemp, 640);

        MicroAPI::Cast<bfloat16_t, fp8_e8m0_t, castTraitFp8_1>(vScalebf16Res0, vScale0, bf16TypeMaskAll); // todo mask type
        MicroAPI::Cast<float, bfloat16_t, castTraitFp8_1>(vScalefp32Res0, vScalebf16Res0, fp32MaskAll);

        MicroAPI::StoreAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            ubDstAddrTmp, vScalefp32Res0, 64, bf16TypeMaskAll);
    }
}
template <typename Q_T, typename KV_T>
__aicore__ inline void CastScale(LocalTensor<float>& outputUb,  LocalTensor<KV_T>& inputUb,
                                   uint32_t dealRowCount) {
    __ubuf__ float* ubDstAddr = (__ubuf__ float*)(outputUb.GetPhyAddr());
    __ubuf__ int8_t* ubScaleAddr = (__ubuf__ int8_t*)(inputUb[448 + 64 * 2].GetPhyAddr());

    CastScaleImpl<Q_T, KV_T>(ubDstAddr, ubScaleAddr, dealRowCount);
}
template <typename Q_T, typename KV_T>
__simd_vf__ void AntiquantVFImplFp8D448(__ubuf__ int8_t* ubSrcAddr, __ubuf__ Q_T* ubDstAddr, // output first
                                        __ubuf__ float* ubScaleSrcAddr, uint32_t dealRowCount)
{
    uint32_t combineDim = 640; // 128对齐
    MicroAPI::RegTensor<KV_T> vKvData0;
    MicroAPI::RegTensor<KV_T> vKvData1;
    MicroAPI::RegTensor<half> vKvDataHalf0;
    MicroAPI::RegTensor<half> vKvDataHalf1;
    MicroAPI::RegTensor<float> vCastFp32Res0;
    MicroAPI::RegTensor<float> vCastFp32Res1;
    MicroAPI::RegTensor<float> vMulRes0;
    MicroAPI::RegTensor<float> vMulRes1;
    MicroAPI::RegTensor<float> vScale0;
    MicroAPI::RegTensor<float> vScale1;
    MicroAPI::RegTensor<Q_T> vCastRes0;
    MicroAPI::RegTensor<Q_T> vCastRes1;
    MicroAPI::RegTensor<Q_T> vCastResPack0;
    MicroAPI::RegTensor<Q_T> vCastResPack1;

    MicroAPI::MaskReg kvTypeMaskAll = MicroAPI::CreateMask<KV_T, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg kvRopeTypeMaskAll = MicroAPI::CreateMask<Q_T, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg fp32MaskAll = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
    uint32_t blockStride = 17; // +1 to solve bank confict
    uint32_t repeatStride = 1;
    for (uint16_t j = 0; j < (512 / 128); j++) {
        // tilesize is 64, deal 128 b8 kv, deal 2 fp32 scale
        __ubuf__ int8_t* ubSrcTemp = ubSrcAddr + j * 128;
        __ubuf__ float* ubScaleSrcAddrTemp = ubScaleSrcAddr + j * 2;
        __ubuf__ Q_T* ubDstAddrTmp = ubDstAddr + j * 128 * blockStride;
        for (uint16_t i = 0; i < static_cast<uint16_t>(dealRowCount); i++) {
            // load scale
            MicroAPI::LoadAlign<int8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_UNPACK4_B8>(
                (MicroAPI::RegTensor<int8_t>&)vKvData0, ubSrcTemp, 64);
            MicroAPI::LoadAlign<int8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_UNPACK4_B8>(
                (MicroAPI::RegTensor<int8_t>&)vKvData1, ubSrcTemp, combineDim - 64);

            MicroAPI::LoadAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_BRC_B32>(
                (MicroAPI::RegTensor<float>&)vScale0, ubScaleSrcAddrTemp, 1);
            MicroAPI::LoadAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_BRC_B32>(
                (MicroAPI::RegTensor<float>&)vScale1, ubScaleSrcAddrTemp, 64 - 1);

            MicroAPI::Cast<float, KV_T, castTraitFp8_1>(vCastFp32Res0, vKvData0, fp32MaskAll); // todo mask type
            MicroAPI::Cast<float, KV_T, castTraitFp8_1>(vCastFp32Res1, vKvData1, fp32MaskAll);

            MicroAPI::Mul<float, MicroAPI::MaskMergeMode::ZEROING>(vMulRes0, vCastFp32Res0, vScale0, fp32MaskAll);
            MicroAPI::Mul<float, MicroAPI::MaskMergeMode::ZEROING>(vMulRes1, vCastFp32Res1, vScale1, fp32MaskAll);

            MicroAPI::Cast<Q_T, float, castTraitFp8_3>(vCastRes0, vMulRes0, fp32MaskAll);
            MicroAPI::Cast<Q_T, float, castTraitFp8_3>(vCastRes1, vMulRes1, fp32MaskAll);

            MicroAPI::DeInterleave(vCastResPack0, vCastResPack1, vCastRes0, vCastRes1);
            // todo copy nz
            MicroAPI::StoreAlign<Q_T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ubDstAddrTmp, vCastResPack0, blockStride, repeatStride, kvRopeTypeMaskAll);
        }
    }
    return;
}
template <typename Q_T, typename KV_T>
__aicore__ inline void AntiquantVFFp8D448(LocalTensor<Q_T>& outputUb,  LocalTensor<KV_T>& inputUb,
                                   LocalTensor<float>& scaleUb, uint32_t dealRowCount) {
    __ubuf__ int8_t* ubSrcAddr = (__ubuf__ int8_t*)(inputUb[64 * sizeof(Q_T)].GetPhyAddr());
    __ubuf__ Q_T* ubDstAddr = (__ubuf__ Q_T*)(outputUb.GetPhyAddr());
    __ubuf__ float* ubScaleAddr = (__ubuf__ float*)(scaleUb.GetPhyAddr());

    AntiquantVFImplFp8D448<Q_T, KV_T>(ubSrcAddr, ubDstAddr, ubScaleAddr, dealRowCount);
}
TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockVec<TEMPLATE_ARGS>::DequantKv(LocalTensor<Q_T> antiKvTensorAsB16,
    LocalTensor<KV_T> srcTensor, int64_t dealRow, int64_t s2ProcessBaseSize)
{
    // srcTensor是rope(448) + nope(64) + scale + pad, dstTensor是nope(448) + rope(64)
    LocalTensor<float> floatScale = dequantScaleBuff_.Get<float>();
    CastScale<Q_T, KV_T>(floatScale, srcTensor, dealRow);
    // PRINTF("dealrow is %d\n", dealRow);
    // DumpTensor(kvMergUb_, 20001, 1024);
    AntiquantVFFp8D448<Q_T, KV_T>(antiKvTensorAsB16, srcTensor, floatScale, dealRow);

    LocalTensor<Q_T> kRopeUb = srcTensor.template ReinterpretCast<Q_T>();
    LocalTensor<Q_T> kRopeUbNz = antiKvTensorAsB16[constInfo_.dSizeNope * (16 + 1)];

    Copy(kRopeUbNz, kRopeUb,
        constInfo_.dSizeRope, // mask 处理多少列数据
        static_cast<uint8_t>(dealRow), // repeatTime, 每次处理多少个block
        {
            17, // dst stride
            1, // src stride
            1, // dst repeat stride
            20 // src repeat stride, 640 / 32
        });

    // DumpTensor(antiKvTensorAsB16, 20009, 1024);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockVec<TEMPLATE_ARGS>::CopyOutKvUb2L1(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputL1,
    LocalTensor<Q_T> antiKvTensorAsB16, int64_t v0Loop, int64_t dealRow, int64_t s2StartOffset)
{
    uint64_t blockElementNum = 16;
    DataCopyParams dataCopyParams;
    dataCopyParams.blockCount = (constInfo_.dSizeNope + constInfo_.dSizeRope) / blockElementNum;
    dataCopyParams.blockLen = dealRow;
    dataCopyParams.srcGap = 17 - dealRow; // 16 + 1
    dataCopyParams.dstGap = constInfo_.s2BaseSize - dealRow;

    LocalTensor<Q_T> dst = outputL1.GetTensor<Q_T>();
    DataCopy(dst[s2StartOffset * blockElementNum], antiKvTensorAsB16, dataCopyParams);
    
    // keyGm_
    // Nd2NzParams nd2nzPara;
    // nd2nzPara.ndNum = 4;
    // nd2nzPara.nValue = dealRow; //nd矩阵的行数
    // nd2nzPara.dValue = 16; //nd矩阵的列数
    // nd2nzPara.srcDValue = 640; //同一nd矩阵相邻行起始地址间的偏移
    // nd2nzPara.dstNzC0Stride = dealRow;
    // nd2nzPara.dstNzNStride = 1;
    // nd2nzPara.srcNdMatrixStride = 64 * dealRow;
    // nd2nzPara.dstNzMatrixStride = 0;
    // DataCopy(outputL1.GetTensor<Q_T>()[448 * dealRow], keyGm_, nd2nzPara);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockVec<TEMPLATE_ARGS>::CopyOutMrgeResult(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputL1,
                                                                   int64_t mte2Size, int64_t mte3Size,
                                                                   int64_t s2GmStartOffset, int64_t mergeMte3Idx,
                                                                   const RunInfo &runInfo)
{
#if 0
    if (mte2Size <= mte3Size) {
        return;
    }
    int32_t dealRow = mte2Size - mte3Size;
    DequantKv(mergeMte3Idx, dealRow);
    uint64_t blockElementNum = FP32_BLOCK_ELEMENT_NUM * 2;
    int64_t nopeGmOffset = runInfo.loop % MERGE_CACHE_GM_BUF_NUM * 512 * 576 + (s2GmStartOffset +
        mte3Size) * blockElementNum;
    int64_t ropeGmOffset = runInfo.loop % MERGE_CACHE_GM_BUF_NUM * 512 * 576 + 512 * 512 + (s2GmStartOffset +
        mte3Size) * blockElementNum;
    CopyOutKvUb2L1(outputL1, mergeMte3Idx, nopeGmOffset, ropeGmOffset, dealRow);
#endif
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockVec<TEMPLATE_ARGS>::ProcessNotSparseKv(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputL1, const RunInfo &runInfo)
{
    int64_t s2ProcessBaseSize = 32;
    int64_t s2ProcessSize = s2ProcessBaseSize;
    int64_t s2V0LoopTimes = (runInfo.s2RealSize + s2ProcessBaseSize - 1) / s2ProcessBaseSize;
    int64_t s2Tail = runInfo.s2RealSize - (s2V0LoopTimes - 1) * s2ProcessBaseSize;
    for (uint32_t i = 0; i < s2V0LoopTimes; i++) {
        if (i == s2V0LoopTimes - 1) {
            s2ProcessSize = s2Tail;
        }
        int64_t dealRow = GetSubBlockIdx() == 0 ? CeilDiv(s2ProcessSize, 2L) : s2ProcessSize - CeilDiv(s2ProcessSize, 2L);
        int64_t s2StartOffset = GetSubBlockIdx() == 0 ? 0 : CeilDiv(s2ProcessSize, 2L);
        s2StartOffset += i * s2ProcessBaseSize;
        // 1、copy kv in, gm ->ub
        LocalTensor<KV_T> kvInUb = stage0InQue.AllocTensor<KV_T>();
        CopyInKvNotSparse(kvInUb, i, dealRow, s2StartOffset, runInfo);
        stage0InQue.EnQue(kvInUb);
        kvInUb = stage0InQue.DeQue<KV_T>();

        // 2、dequant by vf
        LocalTensor<Q_T> kvDequantOutUb = stage0OutQue.AllocTensor<Q_T>();
        DequantKv(kvDequantOutUb, kvInUb, dealRow, s2ProcessBaseSize);
        stage0InQue.FreeTensor(kvInUb);
        stage0OutQue.EnQue(kvDequantOutUb);
        kvDequantOutUb = stage0OutQue.DeQue<Q_T>();

        // 3、copy kv out, ub -> l1
        CopyOutKvUb2L1(outputL1, kvDequantOutUb, i, dealRow, s2StartOffset);
        stage0OutQue.FreeTensor(kvDequantOutUb);
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockVec<TEMPLATE_ARGS>::CopyInKvNotSparse(LocalTensor<KV_T> kvMergUb, int64_t v0Loop,
    int64_t dealRow, int64_t s2StartOffset, const RunInfo &runInfo)
{
    // todo 是否要计算前置偏移 s10Idx
    int64_t s2Idx = s2StartOffset + runInfo.s2LoopCount * constInfo_.s2BaseSize + runInfo.s2StartIdx;
    uint32_t combineBytes = constInfo_.dSizeVInput;
    uint32_t combineDim = combineBytes / sizeof(KV_T);
    uint32_t combineDimAlign = CeilAlign(combineBytes, ConstInfo::BUFFER_SIZE_BYTE_32B) / sizeof(KV_T);
    DataCopyExtParams intriParams;
    intriParams.blockCount = dealRow;
    intriParams.blockLen = combineBytes;
    intriParams.dstStride = 0;
    intriParams.srcStride = 0;
    DataCopyPadExtParams<KV_T> padParams;
    padParams.isPad = true;
    padParams.leftPadding = 0;
    padParams.rightPadding = combineDimAlign - combineDim;
    padParams.paddingValue = 0;
    if constexpr (isPa) {
        // PRINTF("PAGE_ATTENTION=====\n");
        uint64_t blockTableBaseOffset = runInfo.boIdx * constInfo_.maxBlockNumPerBatch;
        uint64_t dstOffset = 0;
        uint32_t copyFinishElmenCnt = 0;
        // uint32_t curSequence = runInfo.s2BatchOffset;
        uint32_t curSequence = s2Idx;
        while (copyFinishElmenCnt < dealRow) {
            // PRINTF("copyFinishElmenCnt(%d) < s2ProcessSize(%d)\n", copyFinishElmenCnt, s2ProcessSize);
            uint64_t blockIdOffset = curSequence / constInfo_.blockSize;
            uint64_t remainElmenCnt = curSequence % constInfo_.blockSize;
            uint64_t idInBlockTable = blockTableGm_.GetValue(blockTableBaseOffset + blockIdOffset);
            uint32_t copyElmenCnt = constInfo_.blockSize - remainElmenCnt;
            if (copyElmenCnt + copyFinishElmenCnt > dealRow) {
                copyElmenCnt = dealRow - copyFinishElmenCnt;
            }
            uint64_t srcOffset = idInBlockTable * constInfo_.blockSize * constInfo_.n2Size * combineBytes +
                remainElmenCnt * constInfo_.n2Size * combineBytes + (uint64_t)(runInfo.n2oIdx * combineBytes); // BlockNum, BlockSize, N, D
            intriParams.blockCount = copyElmenCnt; // base s2 size
            DataCopyPad(kvMergUb[dstOffset * combineDimAlign], keyGm_[srcOffset], intriParams, padParams);
            dstOffset += copyElmenCnt;
            copyFinishElmenCnt += copyElmenCnt;
            curSequence += copyElmenCnt;
        }
    } else {
        DataCopyPad(kvMergUb, keyGm_[s2Idx * combineDim], intriParams, padParams);
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockVec<TEMPLATE_ARGS>::ProcessVec0(
    Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputL1, const RunInfo &runInfo)
{
    outputL1.WaitCrossCore(); // 核间同步
    bool isCmp = runInfo.s2LoopCount > runInfo.oriKvLoopEndIdx; // todo:判断条件确认开闭
    isCmp = false;
    if (isCmp) {
        keyGm_ = cmpKVGm;
        blockTableGm_ = cmpBlockTableGm;
        // todo block size可以不同
    } else {
        keyGm_ = oriKVGm;
        blockTableGm_ = oriBlockTableGm;
    }

    if ((TEMPLATE_MODE == SASTemplateMode::SCFA_TEMPLATE_MODE) && (isCmp)) {
        ProcessSparseKv(outputL1, runInfo);
    } else {
        ProcessNotSparseKv(outputL1, runInfo);
    }
    outputL1.SetCrossCore(); // 核间同步
}
TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockVec<TEMPLATE_ARGS>::ProcessSparseKv(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputL1, const RunInfo &runInfo)
{
#if 0
    int64_t s2ProcessSize = runInfo.s2RealSize;
    int64_t s2Pair = CeilDiv(s2ProcessSize, 2L * constInfo_.sparseBlockSize);
    int64_t topkGmBaseOffset = 0;

    // if constexpr (LAYOUT_T == QSFA_LAYOUT::TND) { // zhj
    if (constInfo_.layoutType == static_cast<uint8_t>(SAS_LAYOUT::TND)) {
        uint64_t actualSeqQPrefixSum = actualSeqLengthsQGm.GetValue(runInfo.boIdx);
        // topkGmBaseOffset += (actualSeqQPrefixSum + runInfo.gS1Idx / constInfo_.gSize) * constInfo_.n2Size *
        topkGmBaseOffset += (actualSeqQPrefixSum + runInfo.s1oIdx) * constInfo_.n2Size *
                            constInfo_.sparseBlockCount + runInfo.n2oIdx * constInfo_.sparseBlockCount; // T, N2, K
    } else {
        topkGmBaseOffset += runInfo.boIdx * constInfo_.s1Size * constInfo_.sparseBlockCount +
                            // runInfo.gS1Idx / constInfo_.gSize * constInfo_.sparseBlockCount; // B, S1, N2, K
                            runInfo.s1oIdx * constInfo_.sparseBlockCount; // B, S1, N2, K
    }
    int64_t mergeMte3Idx = 0;
    int64_t mte2Size = 0;
    int64_t mte3Size = 0;
    int64_t s2IdxArray0 = -1;
    int64_t s2IdxArray1 = -1;
    bool needWaitMte3ToMte2 = true;
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(0);
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(1);
    int64_t s2GmStartOffset = GetSubBlockIdx() == 0 ? 0 : CeilDiv(s2Pair, 2L) * 2 * constInfo_.sparseBlockSize;
    int64_t s2GmLimit = GetSubBlockIdx() == 0 ? CeilDiv(s2Pair, 2L) * 2 * constInfo_.sparseBlockSize: s2ProcessSize;
    if (s2GmLimit > s2ProcessSize) {
        s2GmLimit = s2ProcessSize;
    }
    for (int64_t s2GmOffsetArray = s2GmStartOffset; s2GmOffsetArray < s2GmLimit; s2GmOffsetArray += 2 *
        constInfo_.sparseBlockSize) {
        if (needWaitMte3ToMte2) {
            WaitFlag<AscendC::HardEvent::MTE3_MTE2>(mergeMte3Idx % 2);
            needWaitMte3ToMte2 = false;
        }
        GetRealS2Idx(s2GmOffsetArray, s2IdxArray0, topkGmBaseOffset, runInfo);
        if (unlikely(s2IdxArray0 < 0)) {
            CopyOutMrgeResult(outputL1, mte2Size, mte3Size, s2GmStartOffset, mergeMte3Idx, runInfo);
            SetFlag<AscendC::HardEvent::MTE3_MTE2>(mergeMte3Idx % 2);
            mergeMte3Idx++;
            break;
        }
        GetRealS2Idx(s2GmOffsetArray + constInfo_.sparseBlockSize, s2IdxArray1, topkGmBaseOffset, runInfo);
        CopyInKvSparse(mte2Size, mte3Size, mergeMte3Idx, s2IdxArray0, s2IdxArray1, runInfo);
        if ((mte2Size - mte3Size + 2 * constInfo_.sparseBlockSize > 32) ||
            s2GmOffsetArray + 2 * constInfo_.sparseBlockSize >= s2GmLimit) {
            CopyOutMrgeResult(outputL1, mte2Size, mte3Size, s2GmStartOffset, mergeMte3Idx, runInfo);
            mte3Size = mte2Size;
            SetFlag<AscendC::HardEvent::MTE3_MTE2>(mergeMte3Idx % 2);
            mergeMte3Idx++;
            needWaitMte3ToMte2 = true;
        }
    }

    if (unlikely(s2GmStartOffset + mte2Size < s2GmLimit)) {
        uint64_t blockElementNum = FP32_BLOCK_ELEMENT_NUM * 2;
        SetFlag<AscendC::HardEvent::MTE3_V>(0);
        WaitFlag<AscendC::HardEvent::MTE3_V>(0);
        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(mergeMte3Idx & 1);
        LocalTensor<Q_T> mergeUb = kvMergUb_.template ReinterpretCast<Q_T>();
        Duplicate(mergeUb, static_cast<Q_T>(0.0), constInfo_.dSizeNope);
        SetFlag<AscendC::HardEvent::V_MTE3>(0);
        WaitFlag<AscendC::HardEvent::V_MTE3>(0);

        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = constInfo_.dSizeNope / blockElementNum;
        dataCopyParams.blockLen = blockElementNum * sizeof(Q_T);
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = (constInfo_.s2BaseSize - 1) * blockElementNum * sizeof(Q_T);
        for (int64_t s2GmOffset = s2GmStartOffset + mte2Size; s2GmOffset < s2GmLimit; s2GmOffset++) {
            // todo copy 2 l1
            DataCopyPad(kvMergeGm_[runInfo.loop % MERGE_CACHE_GM_BUF_NUM * 512 * 576 + s2GmOffset * blockElementNum],
                        mergeUb, dataCopyParams);
        }
        dataCopyParams.blockCount = constInfo_.dSizeRope / blockElementNum;
        for (int64_t s2GmOffset = s2GmStartOffset + mte2Size; s2GmOffset < s2GmLimit; s2GmOffset++) {
            DataCopyPad(kvMergeGm_[runInfo.loop % MERGE_CACHE_GM_BUF_NUM * 512 * 576 + 512 * constInfo_.dSize +
                                   s2GmOffset * blockElementNum],
                        mergeUb, dataCopyParams);
        }
        SetFlag<AscendC::HardEvent::MTE3_MTE2>(mergeMte3Idx & 1);
        mergeMte3Idx++;
    }
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0);
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(1);
    v0ValidSizeUb_.SetValue(runInfo.loop % MERGE_CACHE_GM_BUF_NUM, mte2Size);
    SetFlag<AscendC::HardEvent::S_MTE3>(1);
    WaitFlag<AscendC::HardEvent::S_MTE3>(1);
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = 1;
    dataCopyParams.blockLen = 128 * sizeof(int32_t);
    dataCopyParams.srcStride = 0;
    dataCopyParams.dstStride = 0;
    // 这里不出核，是否可以用ub给到v1
    DataCopyPad(kvValidSizeGm_[runInfo.loop % MERGE_CACHE_GM_BUF_NUM * (128 * 2) + GetSubBlockIdx() * 128],
                v0ValidSizeUb_, dataCopyParams);
    return;
#endif
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockVec<TEMPLATE_ARGS>::ProcessVec1(
    Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputBuf,
    Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm1ResBuf, RunInfo &runInfo, 
    ConstInfo &constInfo)
{
    bmm1ResBuf.WaitCrossCore();

    LocalTensor<float> sumUb = this->softmaxSumBuf[runInfo.multiCoreIdxMod2].template Get<float>();
    LocalTensor<float> maxUb = this->softmaxMaxBuf[runInfo.multiCoreIdxMod2].template Get<float>();
    LocalTensor<float> expUb = this->softmaxExpBuf[runInfo.taskIdMod2].template Get<T>();
    int64_t stage1Offset = runInfo.taskIdMod2;
    auto stage1CastTensor = this->stage1OutQue[stage1Offset].template AllocTensor<Q_T>();

    LocalTensor<T> apiTmpBuffer = this->commonTBuf.template Get<T>();
    LocalTensor<T> mmRes = bmm1ResBuf.template GetTensor<T>();

    // TODO v0尾块填充-inf处理
    // TODO cfa也要做sinks
    // loopCount = 0 但传入sinks时走update分支，maxUb通过sinks初始化，sumUb初始化为1.0
    if (runInfo.s2LoopCount == 0 && !isSinks) {
        if (likely(runInfo.s2RealSize == 128)) {
            ProcessVec1Vf<T, Q_T, false, s1BaseSize, s2BaseSize, SCFaVectorApi::EQ_128_SCFA>(
                stage1CastTensor, mmRes, sumUb, maxUb, maxUb, apiTmpBuffer, runInfo.halfMRealSize, runInfo.s2RealSize,
                static_cast<T>(constInfo.softmaxScale), negativeFloatScalar);
        } else if(runInfo.s2RealSize <= 64) {
            ProcessVec1Vf<T, Q_T, false, s1BaseSize, s2BaseSize, SCFaVectorApi::GT_0_AND_LTE_64_SCFA>(
                stage1CastTensor, mmRes, sumUb, maxUb, maxUb, apiTmpBuffer, runInfo.halfMRealSize, runInfo.s2RealSize,
                static_cast<T>(constInfo.softmaxScale), negativeFloatScalar);
        } else if(runInfo.s2RealSize < 128) {
            ProcessVec1Vf<T, Q_T, false, s1BaseSize, s2BaseSize, SCFaVectorApi::GT_64_AND_LTE_128_SCFA>(
                stage1CastTensor, mmRes, sumUb, maxUb, maxUb, apiTmpBuffer, runInfo.halfMRealSize, runInfo.s2RealSize,
                static_cast<T>(constInfo.softmaxScale), negativeFloatScalar);
        }
    } else {
        if (runInfo.s2LoopCount == 0 && isSinks) {
            // s1切1,vec0: 0 ~ halfMRealSize - 1, vec1: gSize - halfMRealSize ~ gSize
            int64_t sinksGmOffset = GetBlockIdx() % 2 == 0 ? 0 : constInfo.gSize - runInfo.halfMRealSize;
            CopySinksIn(maxUb, sinksGmOffset, runInfo.halfMRealSize);
            DuplicateSumWithR0<T>(sumUb, R0, runInfo.halfMRealSize);
        }
        if (likely(runInfo.s2RealSize == 128)) {
            ProcessVec1Vf<T, Q_T, true, s1BaseSize, s2BaseSize, SCFaVectorApi::EQ_128_SCFA>(
                stage1CastTensor, mmRes, sumUb, maxUb, maxUb, apiTmpBuffer, runInfo.halfMRealSize, runInfo.s2RealSize,
                static_cast<T>(constInfo.softmaxScale), negativeFloatScalar);
        } else if (runInfo.s2RealSize <= 64) {
            ProcessVec1Vf<T, Q_T, true, s1BaseSize, s2BaseSize, SCFaVectorApi::GT_0_AND_LTE_64_SCFA>(
                stage1CastTensor, mmRes, sumUb, maxUb, maxUb, apiTmpBuffer, runInfo.halfMRealSize, runInfo.s2RealSize,
                static_cast<T>(constInfo.softmaxScale), negativeFloatScalar);
        } else if(runInfo.s2RealSize < 128) {
            ProcessVec1Vf<T, Q_T, true, s1BaseSize, s2BaseSize, SCFaVectorApi::GT_64_AND_LTE_128_SCFA>(
                stage1CastTensor, mmRes, sumUb, maxUb, maxUb, apiTmpBuffer, runInfo.halfMRealSize, runInfo.s2RealSize,
                static_cast<T>(constInfo.softmaxScale), negativeFloatScalar);
        }
    }
    bmm1ResBuf.SetCrossCore();

    // ===================DataCopy to L1 ====================
    this->stage1OutQue[stage1Offset].template EnQue(stage1CastTensor);
    this->stage1OutQue[stage1Offset].template DeQue<Q_T>();

    outputBuf.WaitCrossCore();
    LocalTensor<Q_T> mm2AL1Tensor = outputBuf.GetTensor<Q_T>();
    if (likely(runInfo.halfMRealSize != 0)) {
        DataCopy(mm2AL1Tensor[constInfo.subBlockIdx * (BLOCK_BYTE / sizeof(Q_T)) * (runInfo.mRealSize - runInfo.halfMRealSize)], stage1CastTensor,
            {s2BaseSize / 16, (uint16_t)runInfo.halfMRealSize,
            (uint16_t)(vec1Srcstride - runInfo.halfMRealSize),
            (uint16_t)(s1BaseSize - runInfo.halfMRealSize)});
    }

    this->stage1OutQue[stage1Offset].template FreeTensor(stage1CastTensor);

    outputBuf.SetCrossCore();
    // ======================================================
    if (runInfo.s2LoopCount != 0 || (runInfo.s2LoopCount == 0 && isSinks)) {
        SCFAUpdateExpSumAndExpMax<T>(sumUb, maxUb, expUb, sumUb, maxUb, apiTmpBuffer, runInfo.halfMRealSize);
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockVec<TEMPLATE_ARGS>::CopySinksIn(const LocalTensor<T>& maxTensor, int64_t sinksGmOffset, uint32_t elementNum)
{
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = 1U;
    dataCopyParams.blockLen = elementNum * sizeof(T);
    dataCopyParams.srcStride = 0U;
    dataCopyParams.dstStride = 0U;
    DataCopyPadExtParams<T> padParams;
    DataCopyPad(maxTensor, this->sinksGm[sinksGmOffset], dataCopyParams, padParams);
    SetFlag<AscendC::HardEvent::MTE2_V>(SYNC_SINKS_BUF_FLAG);
    WaitFlag<AscendC::HardEvent::MTE2_V>(SYNC_SINKS_BUF_FLAG);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockVec<TEMPLATE_ARGS>::ProcessVec2(
    Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm2ResBuf, RunInfo &runInfo,
    ConstInfo &constInfo) {
    bmm2ResBuf.WaitCrossCore();
    if (unlikely(runInfo.vec2MBaseSize == 0)) {
        bmm2ResBuf.SetCrossCore();
        return;
    }
    
    // TOTO:s1切1，g方向当前无尾块  mm2Res: s1Base * dV
    runInfo.vec2S1RealSize = runInfo.vec2S1BaseSize;
    runInfo.vec2MRealSize = runInfo.vec2MBaseSize;
    int64_t vec2CalcSize = runInfo.vec2MRealSize * dTemplateAlign64;

    LocalTensor<T> vec2ResUb = this->stage2OutBuf.template Get<T>();
    LocalTensor<T> mmRes = bmm2ResBuf.template GetTensor<T>();
    WaitFlag<HardEvent::MTE3_V>(mte3ToVId[0]);
    if (unlikely(runInfo.s2LoopCount == 0)) {
        DataCopy(vec2ResUb, mmRes, vec2CalcSize);
    } else {
        LocalTensor<T> expUb = softmaxExpBuf[runInfo.taskIdMod2].template Get<T>();
        if (runInfo.s2LoopCount < runInfo.s2LoopLimit) {
            FlashUpdateNew<T, Q_T, OUTPUT_T, dTemplateAlign64>(
                    vec2ResUb, mmRes, vec2ResUb, expUb, runInfo.vec2MRealSize);
        } else {
            LocalTensor<float> sumUb = this->softmaxSumBuf[runInfo.multiCoreIdxMod2].template Get<float>();
            FlashUpdateLastNew<T, Q_T, OUTPUT_T, dTemplateAlign64>(
                vec2ResUb, mmRes, vec2ResUb, expUb, sumUb, runInfo.vec2MRealSize);
        }
    }

    bmm2ResBuf.SetCrossCore();
    if (runInfo.s2LoopCount == runInfo.s2LoopLimit) {
        if (unlikely(runInfo.s2LoopCount == 0)) {
            LocalTensor<float> sumUb = this->softmaxSumBuf[runInfo.multiCoreIdxMod2].template Get<float>();
            LastDivNew<T, Q_T, OUTPUT_T, dTemplateAlign64>(vec2ResUb, vec2ResUb, sumUb, runInfo.vec2MRealSize);
        }

        this->CopyOutAttentionOut(runInfo, constInfo, vec2ResUb, 0, vec2CalcSize);
    }
    SetFlag<HardEvent::MTE3_V>(mte3ToVId[0]);
}

TEMPLATES_DEF_NO_DEFAULT
template <typename VEC2_RES_T>
__aicore__ inline void SCFABlockVec<TEMPLATE_ARGS>::Bmm2DataCopyOut (
    RunInfo &runInfo, ConstInfo &constInfo, LocalTensor<VEC2_RES_T> &vec2ResUb, int64_t vec2S1Idx, int64_t vec2CalcSize)
{
    LocalTensor<OUTPUT_T> attenOut;
    int64_t dSizeAligned64 = (int64_t)dTemplateAlign64;

    if constexpr (!IsSameType<Q_T, VEC2_RES_T>::value) {
        attenOut.SetAddr(vec2ResUb.address_);
        Cast(attenOut, vec2ResUb, RoundMode::CAST_ROUND, vec2CalcSize);
        SetFlag<HardEvent::V_MTE3>(vToMte3Id[0]);
        WaitFlag<HardEvent::V_MTE3>(vToMte3Id[0]);
    }

    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockLen = constInfo.dSizeV * sizeof(OUTPUT_T);
    dataCopyParams.srcStride = (dSizeAligned64 - constInfo.dSizeV) >> 4;
    dataCopyParams.dstStride = constInfo.attentionOutStride;
    dataCopyParams.blockCount = runInfo.vec2MRealSize;

    DataCopyPad(this->attentionOutGm[runInfo.attentionOutOffset], attenOut, dataCopyParams);
}

TEMPLATES_DEF_NO_DEFAULT
template <typename VEC2_RES_T>
__aicore__ inline void SCFABlockVec<TEMPLATE_ARGS>::CopyOutAttentionOut(
    RunInfo &runInfo, ConstInfo &constInfo, LocalTensor<VEC2_RES_T> &vec2ResUb, int64_t vec2S1Idx, int64_t vec2CalcSize)
{
    this->Bmm2DataCopyOut(runInfo, constInfo, vec2ResUb, vec2S1Idx, vec2CalcSize);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockVec<TEMPLATE_ARGS>::InitOutputSingleCore(ConstInfo &constInfo, __gm__ uint8_t *cuSeqlensQ)
{
#if 0
    uint32_t coreNum = GetBlockNum();
    uint32_t vecBlockIdx = GetBlockIdx(); // vec:0-47
    uint64_t totalOutputSize = 0;
    GlobalTensor<int32_t>actualSeqQLenGM;
    if (cuSeqlensQ != nullptr) {
        actualSeqQLenGM.SetGlobalBuffer((__gm__ int32_t *)cuSeqlensQ);
    }

    // n2 = 1, n1 = gn2 = gSize
    if(constInfo.layoutType == static_cast<uint8_t>(SAS_LAYOUT::BSND)) {
        totalOutputSize = constInfo.bSize * constInfo.gSize * constInfo.s1Size * constInfo.dSizeV;
    }else if(constInfo.layoutType == static_cast<uint8_t>(SAS_LAYOUT::TND)) {
        totalOutputSize = actualSeqQLenGM.GetValue(constInfo.actualSeqLenSize) * constInfo.gSize * constInfo.dSizeV;
    }

    if (coreNum != 0) {
    uint64_t singleCoreSize = (totalOutputSize + (2 * coreNum) - 1) / (2 * coreNum);  // 2 means c:v = 1:2
    uint64_t tailSize = totalOutputSize - vecBlockIdx * singleCoreSize;
    uint64_t singleInitOutputSize = tailSize < singleCoreSize ? tailSize : singleCoreSize;
        if (singleInitOutputSize > 0) {
            matmul::InitOutput<OUTPUT_T>(this->attentionOutGm[vecBlockIdx * singleCoreSize], singleInitOutputSize, 0);
        }
    }
    SyncAll();
#endif
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockVec<TEMPLATE_ARGS>::CleanOutput(__gm__ uint8_t *attentionOut, ConstInfo &constInfo, __gm__ uint8_t *cuSeqlensQ) 
{
    if ASCEND_IS_AIV {
        this->attentionOutGm.SetGlobalBuffer((__gm__ OUTPUT_T *)attentionOut);
        if (constInfo.needInit == 1) {
            InitOutputSingleCore(constInfo, cuSeqlensQ);
        }
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockVec<TEMPLATE_ARGS>::InitGlobalBuffer(__gm__ uint8_t *oriKV, __gm__ uint8_t *cmpKV,
    __gm__ uint8_t *cmpSparseIndices, __gm__ uint8_t *oriBlockTable, __gm__ uint8_t *cmpBlockTable, __gm__ uint8_t *cuSeqlensQ,
    __gm__ uint8_t *sequsedKv, __gm__ uint8_t *sinks)
{
    oriKVGm.SetGlobalBuffer((__gm__ KV_T *)(oriKV));
    oriBlockTableGm.SetGlobalBuffer((__gm__ int32_t *)oriBlockTable);

    // if constexpr (TEMPLATE_MODE != SASTemplateMode::SWA_TEMPLATE_MODE) { // useless
    cmpKVGm.SetGlobalBuffer((__gm__ KV_T *)cmpKV);
    cmpBlockTableGm.SetGlobalBuffer((__gm__ int32_t *)cmpBlockTable);
    // }

    if constexpr (TEMPLATE_MODE == SASTemplateMode::SCFA_TEMPLATE_MODE) {
        cmpSparseIndicesGm.SetGlobalBuffer((__gm__ int32_t *)cmpSparseIndices);
    }

    if (cuSeqlensQ != nullptr) {
        // actualSeqQlenAddr = (__gm__ int64_t *)cuSeqlensQ;
        actualSeqLengthsQGm.SetGlobalBuffer((__gm__ int32_t *)cuSeqlensQ);
    }
    if (sequsedKv != nullptr) {
        // actualSeqKvlenAddr = (__gm__ int64_t *)sequsedKv;
        actualSeqLengthsKVGm.SetGlobalBuffer((__gm__ int32_t *)sequsedKv);
    }
    if (sinks != nullptr) {
        sinksGm.SetGlobalBuffer((__gm__ T *)sinks);
        this->isSinks = true;
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockVec<TEMPLATE_ARGS>::SoftmaxInitBuffer()
{
    tPipe->InitBuffer(softmaxSumBuf[0], 128); // 64/ 2*sizeof(float) = 128
    tPipe->InitBuffer(softmaxSumBuf[1], 128); 
    tPipe->InitBuffer(softmaxMaxBuf[0], 128); 
    tPipe->InitBuffer(softmaxMaxBuf[1], 128); 
    tPipe->InitBuffer(softmaxExpBuf[0], 128); 
    tPipe->InitBuffer(softmaxExpBuf[1], 128); 
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockVec<TEMPLATE_ARGS>::InitLocalBuffer(TPipe *pipe, ConstInfo &constInfo)
{
    // ub buffer
    // v0
    constInfo_ = constInfo;
    pipe->InitBuffer(v0ValidSizeBuff, ConstInfo::BUFFER_SIZE_BYTE_8K);
    pipe->InitBuffer(dequantScaleBuff_, 128 * 16 * 2 * sizeof(float));
    v0ValidSizeUb_ = v0ValidSizeBuff.Get<int32_t>();

    uint32_t mm1ResultSize = s1BaseSize / CV_RATIO * s2BaseSize * sizeof(T);
    uint32_t mm2ResultSize = s1BaseSize / CV_RATIO * dTemplateAlign64 * sizeof(T);

    SoftmaxInitBuffer();
    
    tPipe->InitBuffer(commonTBuf, 512);

    tPipe->InitBuffer(stage0InQue, 2, 640 * 16 * sizeof(KV_T));
    tPipe->InitBuffer(stage0OutQue, 2, 512 * (16 + 1) * sizeof(Q_T));

    tPipe->InitBuffer(stage1OutQue[0], 1, 8448); // （32 + 1） * 128 * 2(bf16)
    tPipe->InitBuffer(stage1OutQue[1], 1, 8448);
    tPipe->InitBuffer(stage2OutBuf, 32 * dTemplateAlign64 * sizeof(T)); //s1Base/cv_ratio * 512 * 4(float)

    mte3ToVId[0] = GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>();
    mte3ToVId[1] = GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>();

    vToMte3Id[0] = GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>();
    vToMte3Id[1] = GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>();
    SetFlag<HardEvent::MTE3_V>(mte3ToVId[0]);
    SetFlag<HardEvent::MTE3_V>(mte3ToVId[1]);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockVec<TEMPLATE_ARGS>::InitCubeVecSharedParams(
    CVSharedParams &sharedParams, int32_t aicIdx, uint8_t subBlockIdx)
{
    auto &sparseAttnSharedkvBaseParams = this->tilingData->baseParams;
    sharedParams.bSize = sparseAttnSharedkvBaseParams.batchSize;
    sharedParams.n2Size = 1;
    sharedParams.gSize = sparseAttnSharedkvBaseParams.nNumOfQInOneGroup; 
    sharedParams.s1Size = sparseAttnSharedkvBaseParams.qSeqSize;
    sharedParams.s2Size = sparseAttnSharedkvBaseParams.kvSeqSize;
    sharedParams.actualSeqLengthsSize = sparseAttnSharedkvBaseParams.actualLenDimsQ;
    sharedParams.actualSeqLengthsKVSize = sparseAttnSharedkvBaseParams.actualLenDimsKV;
    sharedParams.sparseBlockCount = sparseAttnSharedkvBaseParams.sparseBlockCount;
    sharedParams.sparseBlockSize = sparseAttnSharedkvBaseParams.sparseBlockSize;
    sharedParams.cmpRatio = sparseAttnSharedkvBaseParams.cmpRatio;
    sharedParams.oriMaskMode = sparseAttnSharedkvBaseParams.oriMaskMode;
    sharedParams.cmpMaskMode = sparseAttnSharedkvBaseParams.cmpMaskMode;
    sharedParams.oriWinLeft = sparseAttnSharedkvBaseParams.oriWinLeft;
    sharedParams.oriWinRight = sparseAttnSharedkvBaseParams.oriWinRight;
    sharedParams.layoutType = sparseAttnSharedkvBaseParams.outputLayout; 
    sharedParams.kvQuantMode = sparseAttnSharedkvBaseParams.kvQuantMode;
    sharedParams.tileSize = sparseAttnSharedkvBaseParams.tileSize;
    sharedParams.ropeHeadDim = sparseAttnSharedkvBaseParams.ropeHeadDim;
    sharedParams.softmaxScale = sparseAttnSharedkvBaseParams.softmaxScale; 
    sharedParams.dSize = sparseAttnSharedkvBaseParams.dSize;
    sharedParams.dSizeV = sparseAttnSharedkvBaseParams.dSizeV;
    sharedParams.dSizeVInput = sparseAttnSharedkvBaseParams.dSizeVInput;

    // pageAttention, rope在C侧搬运时使用
    if constexpr (isPa) {
        sharedParams.blockSize = sparseAttnSharedkvBaseParams.paBlockSize;
        sharedParams.oriMaxBlockNumPerBatch = sparseAttnSharedkvBaseParams.oriMaxBlockNumPerBatch; 
        sharedParams.cmpMaxBlockNumPerBatch = sparseAttnSharedkvBaseParams.cmpMaxBlockNumPerBatch;
    }
    
    // actQ->TND, actKV pa场景任意layout均有
    sharedParams.isActualSeqLengthsNull = 1U; // 非tnd true 
    sharedParams.isActualSeqLengthsKVNull = 0U; // 均flase 
    if constexpr (LAYOUT_T == SAS_LAYOUT::TND){
        sharedParams.isActualSeqLengthsNull = 0U; // flase  
    }

    sharedParams.coreNum = metadataVecLocal.usedCoreNum;
    /* 多核切分偏移计算 */
    sharedParams.needInit = 1; //

    if ASCEND_IS_AIV {
        if (subBlockIdx == 0) {
            auto tempTilingSSbuf = reinterpret_cast<__ssbuf__ uint32_t*>(0); // 从ssbuf的0地址开始拷贝
            auto tempTiling = reinterpret_cast<uint32_t *>(&sharedParams);
            #pragma unroll
            for (int i = 0; i < sizeof(CVSharedParams) / sizeof(uint32_t); ++i, ++tempTilingSSbuf, ++tempTiling) {
                *tempTilingSSbuf = *tempTiling;
            }
            CrossCoreSetFlag<SYNC_MODE, PIPE_S>(15);
        }
    }

}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockVec<TEMPLATE_ARGS>::GetExtremeValue(
    T &negativeScalar)
{
    uint32_t tmp1 = NEGATIVE_MIN_VAULE_FP32;
    negativeScalar = *((float *)&tmp1);
}

TEMPLATES_DEF
class SCFABlockVecDummy {
public:
    // TODO 是否需要补充其他函数
    __aicore__ inline SCFABlockVecDummy() {};
    __aicore__ inline void CleanOutput(__gm__ uint8_t *attentionOut, ConstInfo &constInfo, __gm__ uint8_t *cuSeqlensQ) {}
    __aicore__ inline void InitGlobalBuffer(__gm__ uint8_t *oriKV, __gm__ uint8_t *cmpKV, __gm__ uint8_t *cmpSparseIndices,
        __gm__ uint8_t *oriBlockTable, __gm__ uint8_t *cmpBlockTable, __gm__ uint8_t *cuSeqlensQ, __gm__ uint8_t *sequsedKv,
        __gm__ uint8_t *sinks) {}
    __aicore__ inline void InitVecBlock(TPipe *pipe, const KvQuantSparseAttnSharedkvTilingData *__restrict tiling,
        CVSharedParams &sharedParams, int32_t aicIdx, uint8_t subBlockIdx, SasMetaData &metadataLocal) {};
    __aicore__ inline void InitLocalBuffer(TPipe *pipe, ConstInfo &constInfo) {}
    __aicore__ inline void ProcessVec1(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputBuf,
        Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm1ResBuf, RunInfo &runInfo,
        ConstInfo &constInfo) {}

    using mm2ResPos = Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>;
    __aicore__ inline void ProcessVec2(mm2ResPos &bmm2ResBuf, RunInfo &runInfo,
        ConstInfo &constInfo) {}
};
}
#endif // KV_QUANT_SPARSE_ATTN_SHAREDKV_SCFA_BLOCK_VECTOR_H

