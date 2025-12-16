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
 * \file vf_antiquant_w8.h
 * \brief
 */
#ifndef VF_ANTIQUANT_W8
#define VF_ANTIQUANT_W8

#include "kernel_tensor.h"

namespace AscendC {
// w8转Q_T
static constexpr MicroAPI::CastTrait castTrait = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
                                                  MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
// fp32->Q_T
static constexpr MicroAPI::CastTrait castTrait0 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
                                                   MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
// fp16 -> bf16                                                      
static constexpr MicroAPI::CastTrait castTrait1 = {MicroAPI::RegLayout::UNKNOWN, MicroAPI::SatMode::UNKNOWN,
                                                      MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
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

template <typename Q_T, typename KV_T, typename ANTIQ_PARAMS_T, uint32_t baseSize, bool hasOffset = false>
__aicore__ inline void AntiquantVFImplW8Nz(LocalTensor<KV_T>& antiqInUb, LocalTensor<Q_T>& antiqResUb,
                                           LocalTensor<ANTIQ_PARAMS_T>& antiqOffsetUb,
                                           LocalTensor<ANTIQ_PARAMS_T>& antiqScaleUb,
                                           uint32_t dealRowCount) {
  static_assert(baseSize % 16 == 0);
  static_assert(IsSameType<KV_T, int8_t>::value || IsSameType<KV_T, hifloat8_t>::value,
                "antiquant w8, KV_T must be int8_t or hifloat8_t");
  __VEC_SCOPE__ {
    __ubuf__ uint8_t* ubSrcAddr = (__ubuf__ uint8_t*)(antiqInUb.GetPhyAddr());
    __ubuf__ Q_T* ubDstAddr = (__ubuf__ Q_T*)(antiqResUb.GetPhyAddr());
    __ubuf__ Q_T* ubOffsetAddr = (__ubuf__ Q_T*)antiqOffsetUb.GetPhyAddr();
    __ubuf__ Q_T* ubScaleAddr = (__ubuf__ Q_T*)antiqScaleUb.GetPhyAddr();

    MicroAPI::RegTensor<KV_T> vKvData;
    MicroAPI::RegTensor<Q_T> vOffset;
    MicroAPI::RegTensor<Q_T> vScale;
    MicroAPI::RegTensor<Q_T> vRes;
    MicroAPI::RegTensor<half> vCastFp16Res;

    MicroAPI::MaskReg kvTypeMaskAll = MicroAPI::CreateMask<KV_T, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg qTypeMaskAll = MicroAPI::CreateMask<Q_T, MicroAPI::MaskPattern::ALL>(); // Q_T 所有元素（共128个）

    // UB总共dealRowCount行 * baseSize列，每次处理8行 * 16列 = 128个元素
    uint32_t rowBaseSize = 8; // 8行
    uint32_t colBaseSize = 16; // 16列
    uint32_t dealBaseNum = 128; // 128个元素
    uint32_t colDstStride = dealRowCount * colBaseSize;
    uint32_t colSrcStride = (dealRowCount * colBaseSize + 31) / 32 * 32; // 32B对齐
    const uint16_t colLoopCnt = static_cast<uint16_t>(baseSize / colBaseSize);
    const uint16_t rowLoopCnt = static_cast<uint16_t>((dealRowCount + rowBaseSize - 1) / rowBaseSize); // 8行对齐

    for (uint16_t colLoopIdx = 0; colLoopIdx < colLoopCnt; colLoopIdx++) {
      __ubuf__ Q_T* ubDstAddrTmp = ubDstAddr + colDstStride * colLoopIdx;
      __ubuf__ uint8_t* ubSrcTemp = ubSrcAddr + colSrcStride * colLoopIdx;
      if constexpr (hasOffset) {
        MicroAPI::DataCopy<Q_T, MicroAPI::LoadDist::DIST_BLK>(vOffset, ubOffsetAddr + colLoopIdx * colBaseSize);
      }
      MicroAPI::DataCopy<Q_T, MicroAPI::LoadDist::DIST_BLK>(vScale, ubScaleAddr + colLoopIdx * colBaseSize);

      // #pragma unroll(4)
      for (uint16_t rowLoopIdx = 0; rowLoopIdx < rowLoopCnt; rowLoopIdx++) {
        MicroAPI::DataCopy<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_UNPACK_B8>(
            (MicroAPI::RegTensor<uint8_t>&)vKvData, ubSrcTemp, dealBaseNum);
        if constexpr (std::is_same<Q_T, bfloat16_t>::value) {
          MicroAPI::Cast<half, KV_T, castTrait>(vCastFp16Res, vKvData, kvTypeMaskAll);
          MicroAPI::Cast<Q_T, half, castTrait1>(vRes, vCastFp16Res, kvTypeMaskAll);
        } else {
          MicroAPI::Cast<Q_T, KV_T, castTrait>(vRes, vKvData, kvTypeMaskAll);
        }
        if constexpr (hasOffset) {
          MicroAPI::Add<Q_T, MicroAPI::MaskMergeMode::ZEROING>(vRes, vRes, vOffset, qTypeMaskAll);
        }
        MicroAPI::Mul<Q_T, MicroAPI::MaskMergeMode::ZEROING>(vRes, vRes, vScale, qTypeMaskAll);
        MicroAPI::DataCopy<Q_T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(ubDstAddrTmp, vRes, dealBaseNum, qTypeMaskAll);
      }
    }
  }
}

template <typename Q_T, typename KV_T, uint32_t baseSize>
__aicore__ inline void AntiquantVFImplFp8Nz(LocalTensor<KV_T>& antiqInUb, LocalTensor<Q_T>& antiqResUb,
                                          LocalTensor<Q_T>& antiqScaleFp16Ub, uint32_t dealRowCount) {
  static_assert(baseSize % 16 == 0);
  static_assert(IsSameType<KV_T, fp8_e4m3fn_t>::value || IsSameType<KV_T, fp8_e5m2_t>::value,
                "antiquant w8, KV_T must be fp8_e4m3fn_t or fp8_e5m2_t");

  __VEC_SCOPE__ {
    __ubuf__ uint8_t* ubSrcAddr = (__ubuf__ uint8_t*)(antiqInUb.GetPhyAddr());
    __ubuf__ Q_T* ubDstAddr = (__ubuf__ Q_T*)(antiqResUb.GetPhyAddr());
    __ubuf__ Q_T* ubScaleAddr = (__ubuf__ Q_T*)antiqScaleFp16Ub.GetPhyAddr();

    MicroAPI::RegTensor<KV_T> vKvData;
    MicroAPI::RegTensor<float> vCastFp32Res0;
    MicroAPI::RegTensor<float> vCastFp32Res1;
    MicroAPI::RegTensor<Q_T> vCastRes0;
    MicroAPI::RegTensor<Q_T> vCastRes1;
    MicroAPI::RegTensor<Q_T> vScale;
    MicroAPI::RegTensor<Q_T> vRes;

    MicroAPI::MaskReg kvTypeMaskAll = MicroAPI::CreateMask<KV_T, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg qTypeMaskAll = MicroAPI::CreateMask<Q_T, MicroAPI::MaskPattern::ALL>();

    // UB总共dealRowCount行 * baseSize列，每次处理8行 * 16列 = 128个元素
    uint32_t rowBaseSize = 8; // 8行
    uint32_t colBaseSize = 16; // 16列
    uint32_t dealBaseNum = 128; // 128个元素
    uint32_t colDstStride = dealRowCount * colBaseSize;
    uint32_t colSrcStride = (dealRowCount * colBaseSize + 31) / 32 * 32; // 32B对齐
    const uint16_t colLoopCnt = static_cast<uint16_t>(baseSize / colBaseSize);
    const uint16_t rowLoopCnt = static_cast<uint16_t>((dealRowCount + rowBaseSize - 1) / rowBaseSize);

    for (uint16_t colLoopIdx = 0; colLoopIdx < colLoopCnt; colLoopIdx++) {
      __ubuf__ Q_T* ubDstAddrTmp = ubDstAddr + colDstStride * colLoopIdx;
      __ubuf__ uint8_t* ubSrcTemp = ubSrcAddr + colSrcStride * colLoopIdx;
      // 加载 scale
      MicroAPI::DataCopy<Q_T, MicroAPI::LoadDist::DIST_BLK>(vScale, ubScaleAddr + colLoopIdx * colBaseSize);

      for (uint16_t rowLoopIdx = 0; rowLoopIdx < rowLoopCnt; rowLoopIdx++) {
        MicroAPI::DataCopy<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_UNPACK_B16>(
            (MicroAPI::RegTensor<uint8_t>&)vKvData, ubSrcTemp, dealBaseNum);

        // cast操作, Fp8->Fp32
        MicroAPI::Cast<float, KV_T, castTraitFp8_1>(vCastFp32Res0, vKvData, kvTypeMaskAll);
        MicroAPI::Cast<float, KV_T, castTraitFp8_2>(vCastFp32Res1, vKvData, kvTypeMaskAll);
        // cast操作, Fp32->Fp16/Bf16
        MicroAPI::Cast<Q_T, float, castTraitFp8_3>(vCastRes0, vCastFp32Res0, kvTypeMaskAll);
        MicroAPI::Cast<Q_T, float, castTraitFp8_4>(vCastRes1, vCastFp32Res1, kvTypeMaskAll);
        MicroAPI::Or<uint16_t, MicroAPI::MaskMergeMode::ZEROING>((MicroAPI::RegTensor<uint16_t>&)vCastRes0,
                                                                (MicroAPI::RegTensor<uint16_t>&)vCastRes0,
                                                                (MicroAPI::RegTensor<uint16_t>&)vCastRes1, kvTypeMaskAll);
        MicroAPI::Mul<Q_T, MicroAPI::MaskMergeMode::ZEROING>(vRes, vCastRes0, vScale, qTypeMaskAll);
        // 将输出结果copy到UB
        MicroAPI::DataCopy<Q_T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(ubDstAddrTmp, vRes, dealBaseNum, qTypeMaskAll);
      }
    }
  }
}

template <typename Q_T, typename KV_T, typename ANTIQ_PARAMS_T,
  uint32_t baseSize, bool hasOffset = false>
__aicore__ inline void AntiquantVFImplW8D64(LocalTensor<KV_T>& antiqInUb, LocalTensor<Q_T>& antiqResUb,
                                            LocalTensor<ANTIQ_PARAMS_T>& antiqOffsetUb, LocalTensor<ANTIQ_PARAMS_T>& antiqScaleUb,
                                            uint32_t dealRowCount) {
  static_assert(baseSize == 64);
  static_assert(IsSameType<KV_T, int8_t>::value || IsSameType<KV_T, hifloat8_t>::value,
                "antiquant w8, KV_T must be int8_t or hifloat8_t");

  __VEC_SCOPE__ {
    __ubuf__ uint8_t* ubSrcAddr = (__ubuf__ uint8_t*)(antiqInUb.GetPhyAddr());
    __ubuf__ Q_T* ubDstAddr = (__ubuf__ Q_T*)(antiqResUb.GetPhyAddr());
    __ubuf__ Q_T* ubDstAddr_ = ubDstAddr + 16 - (dealRowCount + 1) * 32 * 4 / 2;
    __ubuf__ Q_T* ubOffsetAddr = (__ubuf__ Q_T*)antiqOffsetUb.GetPhyAddr();
    __ubuf__ Q_T* ubScaleAddr = (__ubuf__ Q_T*)antiqScaleUb.GetPhyAddr();

    MicroAPI::RegTensor<KV_T> vKvData;
    MicroAPI::RegTensor<Q_T> vOffset;
    MicroAPI::RegTensor<Q_T> vScale;
    MicroAPI::RegTensor<Q_T> vRes;
    MicroAPI::RegTensor<half> vCastFp16Res;

    MicroAPI::MaskReg kvTypeMaskAll = MicroAPI::CreateMask<KV_T, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg qTypeMaskLower64 = MicroAPI::CreateMask<Q_T, MicroAPI::MaskPattern::VL64>();
    MicroAPI::MaskReg qTypeMaskLower128 = MicroAPI::CreateMask<Q_T, MicroAPI::MaskPattern::VL128>();
    MicroAPI::MaskReg qTypeMaskAll = MicroAPI::CreateMask<Q_T, MicroAPI::MaskPattern::ALL>(); // Q_T 所有元素（共128个）
    MicroAPI::MaskReg qTypeMaskHigher64;
    MicroAPI::MaskXor(qTypeMaskHigher64, qTypeMaskLower64, qTypeMaskAll, qTypeMaskAll); // qTypeMaskAll与qTypeMaskLower64异或得到qTypeMaskHigher64

    uint32_t blockStride = dealRowCount + 1;
    uint32_t repeatStride = 2;
    uint16_t loopCnt = static_cast<uint16_t>((dealRowCount + 1) / 2); // +1是为了兼容处理奇数行

    __ubuf__ uint8_t* ubSrcTemp = ubSrcAddr;
    if constexpr (hasOffset) {
      MicroAPI::DataCopy<Q_T, MicroAPI::LoadDist::DIST_NORM>(vOffset, ubOffsetAddr);
    }
    MicroAPI::DataCopy<Q_T, MicroAPI::LoadDist::DIST_NORM>(vScale, ubScaleAddr);

    // 对D64优化，相邻2行合并计算；
    for (uint16_t i = 0; i < loopCnt; i++) {
      MicroAPI::DataCopy<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_UNPACK_B8>(
          (MicroAPI::RegTensor<uint8_t>&)vKvData, ubSrcTemp, baseSize * 2);

      if constexpr (std::is_same<Q_T, bfloat16_t>::value) {
        MicroAPI::Cast<half, KV_T, castTrait>(vCastFp16Res, vKvData, kvTypeMaskAll);
        MicroAPI::Cast<Q_T, half, castTrait1>(vRes, vCastFp16Res, kvTypeMaskAll);
      } else {
        MicroAPI::Cast<Q_T, KV_T, castTrait>(vRes, vKvData, kvTypeMaskAll);
      }

      if constexpr (hasOffset) {
        MicroAPI::Add<Q_T, MicroAPI::MaskMergeMode::ZEROING>(vRes, vRes, vOffset, qTypeMaskLower128);
      }
      MicroAPI::Mul<Q_T, MicroAPI::MaskMergeMode::ZEROING>(vRes, vRes, vScale, qTypeMaskLower128);

      MicroAPI::DataCopy<Q_T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
          ubDstAddr, vRes, blockStride, repeatStride, qTypeMaskLower64);

      MicroAPI::DataCopy<Q_T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
          ubDstAddr_, vRes, blockStride, repeatStride, qTypeMaskHigher64);
    }
  }
}

template <typename Q_T, typename KV_T, typename ANTIQ_PARAMS_T,
  uint32_t baseSize, bool hasOffset = false>
__aicore__ inline void AntiquantVFImplW8PerTokenD64(LocalTensor<KV_T>& antiqInUb, LocalTensor<Q_T>& antiqResUb,
                                                    LocalTensor<ANTIQ_PARAMS_T>& antiqOffsetUb, LocalTensor<ANTIQ_PARAMS_T>& antiqScaleUb,
                                                    uint32_t dealRowCount) {
  static_assert(baseSize == 64);
  static_assert(IsSameType<KV_T, int8_t>::value, "antiquant perToken w8, KV_T must be int8_t");
  __VEC_SCOPE__ {
    __ubuf__ uint8_t* ubSrcAddr = (__ubuf__ uint8_t*)(antiqInUb.GetPhyAddr());
    __ubuf__ Q_T* ubDstAddr = (__ubuf__ Q_T*)(antiqResUb.GetPhyAddr());
    __ubuf__ Q_T *ubDstAddr_ = ubDstAddr + 16 - (dealRowCount + 1) * 32 * 4 / 2;
    __ubuf__ ANTIQ_PARAMS_T* ubOffsetAddr = (__ubuf__ ANTIQ_PARAMS_T*)antiqOffsetUb.GetPhyAddr();
    __ubuf__ ANTIQ_PARAMS_T* ubScaleAddr = (__ubuf__ ANTIQ_PARAMS_T*)antiqScaleUb.GetPhyAddr();

    MicroAPI::RegTensor<KV_T> vKvData;
    MicroAPI::RegTensor<ANTIQ_PARAMS_T> vOffset;
    MicroAPI::RegTensor<ANTIQ_PARAMS_T> vScale;
    MicroAPI::RegTensor<Q_T> vOffsetFp16;
    MicroAPI::RegTensor<Q_T> vOffsetFp16High;
    MicroAPI::RegTensor<Q_T> vOffsetFp16Low;
    MicroAPI::RegTensor<Q_T> vScaleFp16;
    MicroAPI::RegTensor<Q_T> vScaleFp16High;
    MicroAPI::RegTensor<Q_T> vScaleFp16Low;
    MicroAPI::RegTensor<Q_T> vRes;
    MicroAPI::RegTensor<half> vCastFp16Res;
    MicroAPI::MaskReg maskOne = MicroAPI::CreateMask<ANTIQ_PARAMS_T, MicroAPI::MaskPattern::VL1>();
    MicroAPI::MaskReg kvTypeMaskAll = MicroAPI::CreateMask<KV_T, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg qTypeMaskLower64 = MicroAPI::CreateMask<Q_T, MicroAPI::MaskPattern::VL64>();
    MicroAPI::MaskReg qTypeMaskLower128 = MicroAPI::CreateMask<Q_T, MicroAPI::MaskPattern::VL128>();
    MicroAPI::MaskReg qTypeMaskAll = MicroAPI::CreateMask<Q_T, MicroAPI::MaskPattern::ALL>(); // Q_T 所有元素（共128个）
    MicroAPI::MaskReg qTypeMaskHigher64;
    MicroAPI::MaskXor(qTypeMaskHigher64, qTypeMaskLower64, qTypeMaskAll, qTypeMaskAll); // qTypeMaskAll与qTypeMaskLower64异或得到qTypeMaskHigher64

    MicroAPI::UnalignReg u0;
    MicroAPI::UnalignReg u1;

    uint32_t blockStride = dealRowCount + 1;
    uint32_t repeatStride = 2;

    MicroAPI::DataCopyUnAlignPre(u0, ubScaleAddr);
    if constexpr (hasOffset) {
      MicroAPI::DataCopyUnAlignPre(u1, ubOffsetAddr);
    }
    uint16_t loopCnt = static_cast<uint16_t>((dealRowCount + 1) / 2); // +1是为了兼容处理奇数行
    // 对D64优化，相邻2行合并计算；+1兼容奇数行场景
    for (uint16_t i = 0; i < loopCnt; i++) {
      MicroAPI::DataCopy<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_UNPACK_B8>(
          (MicroAPI::RegTensor<uint8_t>&)vKvData, ubSrcAddr, baseSize * 2);

      if constexpr (std::is_same<Q_T, bfloat16_t>::value) {
        MicroAPI::Cast<half, KV_T, castTrait>(vCastFp16Res, vKvData, kvTypeMaskAll);
        MicroAPI::Cast<Q_T, half, castTrait1>(vRes, vCastFp16Res, kvTypeMaskAll);
      } else {
        MicroAPI::Cast<Q_T, KV_T, castTrait>(vRes, vKvData, kvTypeMaskAll);
      }

      MicroAPI::DataCopyUnAlign<ANTIQ_PARAMS_T>(vScale, u0, ubScaleAddr, 1); // 1表示ub自动往后偏移1个float
      MicroAPI::Cast<Q_T, ANTIQ_PARAMS_T, castTrait0>(vScaleFp16, vScale, maskOne);
      MicroAPI::Duplicate<uint16_t, MicroAPI::HighLowPart::LOWEST, MicroAPI::MaskMergeMode::ZEROING>
        ((MicroAPI::RegTensor<uint16_t>&)vScaleFp16Low, (MicroAPI::RegTensor<uint16_t>&)vScaleFp16, qTypeMaskLower64);

      MicroAPI::DataCopyUnAlign<ANTIQ_PARAMS_T>(vScale, u0, ubScaleAddr, 1); // 1表示ub自动往后偏移1个float
      MicroAPI::Cast<Q_T, ANTIQ_PARAMS_T, castTrait0>(vScaleFp16, vScale, maskOne);
      MicroAPI::Duplicate<uint16_t, MicroAPI::HighLowPart::LOWEST, MicroAPI::MaskMergeMode::ZEROING>
        ((MicroAPI::RegTensor<uint16_t>&)vScaleFp16High, (MicroAPI::RegTensor<uint16_t>&)vScaleFp16, qTypeMaskHigher64);
      MicroAPI::Or<uint16_t, MicroAPI::MaskMergeMode::ZEROING>((MicroAPI::RegTensor<uint16_t>&)vScaleFp16,
                    (MicroAPI::RegTensor<uint16_t>&)vScaleFp16Low,
                    (MicroAPI::RegTensor<uint16_t>&)vScaleFp16High, kvTypeMaskAll);
      if constexpr (hasOffset) {
        MicroAPI::DataCopyUnAlign<ANTIQ_PARAMS_T>(vOffset, u1, ubOffsetAddr, 1); // 1表示ub自动往后偏移1个float
        MicroAPI::Cast<Q_T, ANTIQ_PARAMS_T, castTrait0>(vOffsetFp16, vOffset, maskOne);
        MicroAPI::Duplicate<uint16_t, MicroAPI::HighLowPart::LOWEST, MicroAPI::MaskMergeMode::ZEROING>
          ((MicroAPI::RegTensor<uint16_t>&)vOffsetFp16Low, (MicroAPI::RegTensor<uint16_t>&)vOffsetFp16, qTypeMaskLower64);

        MicroAPI::DataCopyUnAlign<ANTIQ_PARAMS_T>(vOffset, u1, ubOffsetAddr, 1); // 1表示ub自动往后偏移1个float
        MicroAPI::Cast<Q_T, ANTIQ_PARAMS_T, castTrait0>(vOffsetFp16, vOffset, maskOne);
        MicroAPI::Duplicate<uint16_t, MicroAPI::HighLowPart::LOWEST, MicroAPI::MaskMergeMode::ZEROING>
          ((MicroAPI::RegTensor<uint16_t>&)vOffsetFp16High, (MicroAPI::RegTensor<uint16_t>&)vOffsetFp16, qTypeMaskHigher64);
        MicroAPI::Or<uint16_t, MicroAPI::MaskMergeMode::ZEROING>((MicroAPI::RegTensor<uint16_t>&)vOffsetFp16,
                    (MicroAPI::RegTensor<uint16_t>&)vOffsetFp16Low,
                    (MicroAPI::RegTensor<uint16_t>&)vOffsetFp16High, kvTypeMaskAll);
        MicroAPI::Add<Q_T, MicroAPI::MaskMergeMode::ZEROING>(vRes, vRes, vOffsetFp16, qTypeMaskLower128);
      }

      MicroAPI::Mul<Q_T, MicroAPI::MaskMergeMode::ZEROING>(vRes, vRes, vScaleFp16, qTypeMaskLower128);

      MicroAPI::DataCopy<Q_T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
          ubDstAddr, vRes, blockStride, repeatStride, qTypeMaskLower64);

      MicroAPI::DataCopy<Q_T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
          ubDstAddr_, vRes, blockStride, repeatStride, qTypeMaskHigher64);
    }
  }
}

template <typename Q_T, typename KV_T, typename ANTIQ_PARAMS_T,
  uint32_t baseSize, bool hasOffset = false>
__aicore__ inline void AntiquantVFImplW8Norm(LocalTensor<KV_T>& antiqInUb, LocalTensor<Q_T>& antiqResUb,
                                             LocalTensor<ANTIQ_PARAMS_T>& antiqOffsetUb, LocalTensor<ANTIQ_PARAMS_T>& antiqScaleUb,
                                             uint32_t dealRowCount) {
  static_assert(baseSize % 128 == 0);
  static_assert(IsSameType<KV_T, int8_t>::value || IsSameType<KV_T, hifloat8_t>::value,
                "antiquant w8, KV_T must be int8_t or hifloat8_t");

  __VEC_SCOPE__ {
    __ubuf__ uint8_t* ubSrcAddr = (__ubuf__ uint8_t*)(antiqInUb.GetPhyAddr());
    __ubuf__ Q_T* ubDstAddr = (__ubuf__ Q_T*)(antiqResUb.GetPhyAddr());
    __ubuf__ Q_T* ubOffsetAddr = (__ubuf__ Q_T*)antiqOffsetUb.GetPhyAddr();
    __ubuf__ Q_T* ubScaleAddr = (__ubuf__ Q_T*)antiqScaleUb.GetPhyAddr();

    MicroAPI::RegTensor<KV_T> vKvData;
    MicroAPI::RegTensor<Q_T> vOffset;
    MicroAPI::RegTensor<Q_T> vScale;
    MicroAPI::RegTensor<Q_T> vRes;
    MicroAPI::RegTensor<half> vCastFp16Res;

    MicroAPI::MaskReg kvTypeMaskAll = MicroAPI::CreateMask<KV_T, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg qTypeMaskAll = MicroAPI::CreateMask<Q_T, MicroAPI::MaskPattern::ALL>(); // Q_T 所有元素（共128个）

    uint32_t blockStride = dealRowCount + 1;
    uint32_t repeatStride = 1;
    const uint16_t loops = static_cast<uint16_t>(baseSize / 128);

    for (uint16_t j = 0; j < loops; j++) {
      __ubuf__ Q_T* ubDstAddrTmp = ubDstAddr + blockStride * 128 * j;
      __ubuf__ uint8_t* ubSrcTemp = ubSrcAddr + j * 128;

      if constexpr (hasOffset) {
        MicroAPI::DataCopy<Q_T, MicroAPI::LoadDist::DIST_NORM>(vOffset, ubOffsetAddr + j * 128);
      }
      MicroAPI::DataCopy<Q_T, MicroAPI::LoadDist::DIST_NORM>(vScale, ubScaleAddr + j * 128);

      // #pragma unroll(4)
      for (uint16_t i = 0; i < static_cast<uint16_t>(dealRowCount); i++) {
        MicroAPI::DataCopy<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_UNPACK_B8>(
            (MicroAPI::RegTensor<uint8_t>&)vKvData, ubSrcTemp, baseSize);

        if constexpr (std::is_same<Q_T, bfloat16_t>::value) {
          MicroAPI::Cast<half, KV_T, castTrait>(vCastFp16Res, vKvData, kvTypeMaskAll);
          MicroAPI::Cast<Q_T, half, castTrait1>(vRes, vCastFp16Res, kvTypeMaskAll);
        } else {
          MicroAPI::Cast<Q_T, KV_T, castTrait>(vRes, vKvData, kvTypeMaskAll);
        }

        if constexpr (hasOffset) {
          MicroAPI::Add<Q_T, MicroAPI::MaskMergeMode::ZEROING>(vRes, vRes, vOffset, qTypeMaskAll);
        }

        MicroAPI::Mul<Q_T, MicroAPI::MaskMergeMode::ZEROING>(vRes, vRes, vScale, qTypeMaskAll);

        MicroAPI::DataCopy<Q_T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            ubDstAddrTmp, vRes, blockStride, repeatStride, qTypeMaskAll);
      }
    }
  }
}

template <typename Q_T, typename KV_T, typename ANTIQ_PARAMS_T,
  uint32_t baseSize, bool hasOffset = false>
__aicore__ inline void AntiquantVFImplW8PerTokenD128(LocalTensor<KV_T>& antiqInUb, LocalTensor<Q_T>& antiqResUb,
                                             LocalTensor<ANTIQ_PARAMS_T>& antiqOffsetUb, LocalTensor<ANTIQ_PARAMS_T>& antiqScaleUb,
                                             uint32_t dealRowCount) {
  static_assert(baseSize == 128);
  static_assert(IsSameType<KV_T, int8_t>::value, "antiquant perToken w8, KV_T must be int8_t");

  __VEC_SCOPE__ {
    __ubuf__ uint8_t* ubSrcAddr = (__ubuf__ uint8_t*)(antiqInUb.GetPhyAddr());
    __ubuf__ Q_T* ubDstAddr = (__ubuf__ Q_T*)(antiqResUb.GetPhyAddr());
    __ubuf__ ANTIQ_PARAMS_T* ubOffsetAddr = (__ubuf__ ANTIQ_PARAMS_T*)antiqOffsetUb.GetPhyAddr();
    __ubuf__ ANTIQ_PARAMS_T* ubScaleAddr = (__ubuf__ ANTIQ_PARAMS_T*)antiqScaleUb.GetPhyAddr();

    MicroAPI::RegTensor<KV_T> vKvData;
    MicroAPI::RegTensor<ANTIQ_PARAMS_T> vOffset;
    MicroAPI::RegTensor<ANTIQ_PARAMS_T> vScale;
    MicroAPI::RegTensor<Q_T> vOffsetFp16;
    MicroAPI::RegTensor<Q_T> vScaleFp16;

    MicroAPI::RegTensor<Q_T> vRes;
    MicroAPI::RegTensor<half> vCastFp16Res;

    MicroAPI::MaskReg maskOne = MicroAPI::CreateMask<ANTIQ_PARAMS_T, MicroAPI::MaskPattern::VL1>();
    MicroAPI::MaskReg kvTypeMaskAll = MicroAPI::CreateMask<KV_T, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg qTypeMaskAll = MicroAPI::CreateMask<Q_T, MicroAPI::MaskPattern::ALL>(); // Q_T 所有元素（共128个）

    uint32_t blockStride = dealRowCount + 1;
    uint32_t repeatStride = 1;
    const uint16_t loops = baseSize / 128;
    MicroAPI::UnalignReg u0;
    MicroAPI::UnalignReg u1;
    for (uint16_t j = 0; j < static_cast<uint16_t>(loops); j++) {
      __ubuf__ Q_T* ubDstAddrTmp = ubDstAddr + blockStride * 128 * j;
      __ubuf__ uint8_t* ubSrcTemp = ubSrcAddr + j * 128;
      __ubuf__ ANTIQ_PARAMS_T* ubScaleAddrTemp = ubScaleAddr;
      __ubuf__ ANTIQ_PARAMS_T* ubOffsetAddrTemp = ubOffsetAddr;

      MicroAPI::DataCopyUnAlignPre(u0, ubScaleAddrTemp);
      if constexpr (hasOffset) {
        MicroAPI::DataCopyUnAlignPre(u1, ubOffsetAddrTemp);
      }
      for (uint16_t i = 0; i < static_cast<uint16_t>(dealRowCount); i++) {
        MicroAPI::DataCopy<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_UNPACK_B8>(
            (MicroAPI::RegTensor<uint8_t>&)vKvData, ubSrcTemp, baseSize);

        if constexpr (std::is_same<Q_T, bfloat16_t>::value) {
          MicroAPI::Cast<half, KV_T, castTrait>(vCastFp16Res, vKvData, kvTypeMaskAll);
          MicroAPI::Cast<Q_T, half, castTrait1>(vRes, vCastFp16Res, kvTypeMaskAll);
        } else {
          MicroAPI::Cast<Q_T, KV_T, castTrait>(vRes, vKvData, kvTypeMaskAll);
        }
        MicroAPI::DataCopyUnAlign<ANTIQ_PARAMS_T>(vScale, u0, ubScaleAddrTemp, 1); // 1表示ub自动往后偏移1个float
        MicroAPI::Cast<Q_T, ANTIQ_PARAMS_T, castTrait0>(vScaleFp16, vScale, maskOne);
        MicroAPI::Duplicate<uint16_t, MicroAPI::HighLowPart::LOWEST, MicroAPI::MaskMergeMode::ZEROING>
          ((MicroAPI::RegTensor<uint16_t>&)vScaleFp16, (MicroAPI::RegTensor<uint16_t>&)vScaleFp16, qTypeMaskAll);
        if constexpr (hasOffset) {
          MicroAPI::DataCopyUnAlign<ANTIQ_PARAMS_T>(vOffset, u1, ubOffsetAddrTemp, 1); // 1表示ub自动往后偏移1个float
          MicroAPI::Cast<Q_T, ANTIQ_PARAMS_T, castTrait0>(vOffsetFp16, vOffset, maskOne);
          MicroAPI::Duplicate<uint16_t, MicroAPI::HighLowPart::LOWEST, MicroAPI::MaskMergeMode::ZEROING>
            ((MicroAPI::RegTensor<uint16_t>&)vOffsetFp16, (MicroAPI::RegTensor<uint16_t>&)vOffsetFp16, qTypeMaskAll);
          MicroAPI::Add<Q_T, MicroAPI::MaskMergeMode::ZEROING>(vRes, vRes, vOffsetFp16, qTypeMaskAll);
        }

        MicroAPI::Mul<Q_T, MicroAPI::MaskMergeMode::ZEROING>(vRes, vRes, vScaleFp16, qTypeMaskAll);

        MicroAPI::DataCopy<Q_T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            ubDstAddrTmp, vRes, blockStride, repeatStride, qTypeMaskAll);
      }
    }
  }
}

template <typename Q_T, typename KV_T, typename ANTIQ_PARAMS_T,
  uint32_t baseSize, bool hasOffset = false>
__aicore__ inline void AntiquantVFImplW8PerTokenD256(LocalTensor<KV_T>& antiqInUb, LocalTensor<Q_T>& antiqResUb,
                                             LocalTensor<ANTIQ_PARAMS_T>& antiqOffsetUb, LocalTensor<ANTIQ_PARAMS_T>& antiqScaleUb,
                                             uint32_t dealRowCount) {
  static_assert(baseSize == 256);
  static_assert(IsSameType<KV_T, int8_t>::value, "antiquant perToken w8, KV_T must be int8_t");

  __VEC_SCOPE__ {
    __ubuf__ uint8_t* ubSrcAddr = (__ubuf__ uint8_t*)(antiqInUb.GetPhyAddr());
    __ubuf__ uint8_t* ubSrcAddr1 = ubSrcAddr + 128;
    __ubuf__ Q_T* ubDstAddr = (__ubuf__ Q_T*)(antiqResUb.GetPhyAddr());
    __ubuf__ ANTIQ_PARAMS_T* ubOffsetAddr = (__ubuf__ ANTIQ_PARAMS_T*)antiqOffsetUb.GetPhyAddr();
    __ubuf__ ANTIQ_PARAMS_T* ubScaleAddr = (__ubuf__ ANTIQ_PARAMS_T*)antiqScaleUb.GetPhyAddr();

    MicroAPI::RegTensor<KV_T> vKvData;
    MicroAPI::RegTensor<KV_T> vKvData1;
    MicroAPI::RegTensor<ANTIQ_PARAMS_T> vOffset;
    MicroAPI::RegTensor<ANTIQ_PARAMS_T> vScale;
    MicroAPI::RegTensor<Q_T> vOffsetFp16;
    MicroAPI::RegTensor<Q_T> vScaleFp16;

    MicroAPI::RegTensor<Q_T> vRes;
    MicroAPI::RegTensor<Q_T> vRes1;
    MicroAPI::RegTensor<half> vCastFp16Res;
    MicroAPI::RegTensor<half> vCastFp16Res1;

    MicroAPI::MaskReg maskOne = MicroAPI::CreateMask<ANTIQ_PARAMS_T, MicroAPI::MaskPattern::VL1>();
    MicroAPI::MaskReg kvMaskAll = MicroAPI::CreateMask<KV_T, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg qMaskAll = MicroAPI::CreateMask<Q_T, MicroAPI::MaskPattern::ALL>(); // Q_T 所有元素（共128个）

    uint32_t blockStride = dealRowCount + 1;
    uint32_t repeatStride = 1;
    MicroAPI::UnalignReg u0;
    MicroAPI::UnalignReg u1;
    MicroAPI::DataCopyUnAlignPre(u0, ubScaleAddr);
    if constexpr (hasOffset) {
      MicroAPI::DataCopyUnAlignPre(u1, ubOffsetAddr);
    }
    __ubuf__ Q_T* ubDstAddr1 = ubDstAddr + blockStride * 128;
    for (uint16_t j = 0; j < static_cast<uint16_t>(dealRowCount); j++) {
      // 读入每行的伪量化参数
      MicroAPI::DataCopyUnAlign<ANTIQ_PARAMS_T>(vScale, u0, ubScaleAddr, 1); // 1表示ub自动往后偏移1个float
      MicroAPI::Cast<Q_T, ANTIQ_PARAMS_T, castTrait0>(vScaleFp16, vScale, maskOne);
      MicroAPI::Duplicate<uint16_t, MicroAPI::HighLowPart::LOWEST, MicroAPI::MaskMergeMode::ZEROING>
        ((MicroAPI::RegTensor<uint16_t>&)vScaleFp16, (MicroAPI::RegTensor<uint16_t>&)vScaleFp16, qMaskAll);
      if constexpr (hasOffset) {
        MicroAPI::DataCopyUnAlign<ANTIQ_PARAMS_T>(vOffset, u1, ubOffsetAddr, 1); // 1表示ub自动往后偏移1个float
        MicroAPI::Cast<Q_T, ANTIQ_PARAMS_T, castTrait0>(vOffsetFp16, vOffset, maskOne);
        MicroAPI::Duplicate<uint16_t, MicroAPI::HighLowPart::LOWEST, MicroAPI::MaskMergeMode::ZEROING>
          ((MicroAPI::RegTensor<uint16_t>&)vOffsetFp16, (MicroAPI::RegTensor<uint16_t>&)vOffsetFp16, qMaskAll);
      }
      MicroAPI::DataCopy<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_UNPACK_B8>(
        (MicroAPI::RegTensor<uint8_t>&)vKvData, ubSrcAddr, 256); // d=256，自动往后偏移256个数
      MicroAPI::DataCopy<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_UNPACK_B8>(
        (MicroAPI::RegTensor<uint8_t>&)vKvData1, ubSrcAddr1, 256); // d=256
      if constexpr (std::is_same<Q_T, bfloat16_t>::value) {
        MicroAPI::Cast<half, KV_T, castTrait>(vCastFp16Res, vKvData, kvMaskAll);
        MicroAPI::Cast<half, KV_T, castTrait>(vCastFp16Res1, vKvData1, kvMaskAll);
        MicroAPI::Cast<Q_T, half, castTrait1>(vRes, vCastFp16Res, kvMaskAll);
        MicroAPI::Cast<Q_T, half, castTrait1>(vRes1, vCastFp16Res1, kvMaskAll);
      } else {
        MicroAPI::Cast<Q_T, KV_T, castTrait>(vRes, vKvData, kvMaskAll);
        MicroAPI::Cast<Q_T, KV_T, castTrait>(vRes1, vKvData1, kvMaskAll);
      }
      if constexpr (hasOffset) {
        MicroAPI::Add<Q_T, MicroAPI::MaskMergeMode::ZEROING>(vRes, vRes, vOffsetFp16, qMaskAll);
        MicroAPI::Add<Q_T, MicroAPI::MaskMergeMode::ZEROING>(vRes1, vRes1, vOffsetFp16, qMaskAll);
      }

      MicroAPI::Mul<Q_T, MicroAPI::MaskMergeMode::ZEROING>(vRes, vRes, vScaleFp16, qMaskAll);
      MicroAPI::Mul<Q_T, MicroAPI::MaskMergeMode::ZEROING>(vRes1, vRes1, vScaleFp16, qMaskAll);
      MicroAPI::DataCopy<Q_T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
          ubDstAddr, vRes, blockStride, repeatStride, qMaskAll);
      MicroAPI::DataCopy<Q_T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
          ubDstAddr1, vRes1, blockStride, repeatStride, qMaskAll);
    }
  }
}

template <typename Q_T, typename KV_T, typename ANTIQ_PARAMS_T,
  uint32_t baseSize, bool hasOffset = false>
__aicore__ inline void AntiquantVFImplW8PerTokenD512(LocalTensor<KV_T>& antiqInUb, LocalTensor<Q_T>& antiqResUb,
                                             LocalTensor<ANTIQ_PARAMS_T>& antiqOffsetUb, LocalTensor<ANTIQ_PARAMS_T>& antiqScaleUb,
                                             uint32_t dealRowCount) {
  static_assert(baseSize == 512);
  static_assert(IsSameType<KV_T, int8_t>::value, "antiquant perToken w8, KV_T must be int8_t");

  __VEC_SCOPE__ {
    __ubuf__ uint8_t* ubSrcAddr = (__ubuf__ uint8_t*)(antiqInUb.GetPhyAddr());
    __ubuf__ uint8_t* ubSrcAddr1 = ubSrcAddr + 128;
    __ubuf__ uint8_t* ubSrcAddr2 = ubSrcAddr + 128 * 2;
    __ubuf__ uint8_t* ubSrcAddr3 = ubSrcAddr + 128 * 3;
    __ubuf__ Q_T* ubDstAddr = (__ubuf__ Q_T*)(antiqResUb.GetPhyAddr());
    __ubuf__ ANTIQ_PARAMS_T* ubOffsetAddr = (__ubuf__ ANTIQ_PARAMS_T*)antiqOffsetUb.GetPhyAddr();
    __ubuf__ ANTIQ_PARAMS_T* ubScaleAddr = (__ubuf__ ANTIQ_PARAMS_T*)antiqScaleUb.GetPhyAddr();

    MicroAPI::RegTensor<KV_T> vKvData;
    MicroAPI::RegTensor<KV_T> vKvData1;
    MicroAPI::RegTensor<KV_T> vKvData2;
    MicroAPI::RegTensor<KV_T> vKvData3;
    MicroAPI::RegTensor<ANTIQ_PARAMS_T> vOffset;
    MicroAPI::RegTensor<ANTIQ_PARAMS_T> vScale;
    MicroAPI::RegTensor<Q_T> vOffsetFp16;
    MicroAPI::RegTensor<Q_T> vScaleFp16;

    MicroAPI::RegTensor<Q_T> vRes;
    MicroAPI::RegTensor<Q_T> vRes1;
    MicroAPI::RegTensor<Q_T> vRes2;
    MicroAPI::RegTensor<Q_T> vRes3;
    MicroAPI::RegTensor<half> vCastFp16Res;
    MicroAPI::RegTensor<half> vCastFp16Res1;
    MicroAPI::RegTensor<half> vCastFp16Res2;
    MicroAPI::RegTensor<half> vCastFp16Res3;

    MicroAPI::MaskReg maskOne = MicroAPI::CreateMask<ANTIQ_PARAMS_T, MicroAPI::MaskPattern::VL1>();
    MicroAPI::MaskReg kvMaskAll = MicroAPI::CreateMask<KV_T, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg qMaskAll = MicroAPI::CreateMask<Q_T, MicroAPI::MaskPattern::ALL>(); // Q_T 所有元素（共128个）

    uint32_t blockStride = dealRowCount + 1;
    uint32_t repeatStride = 1;
    MicroAPI::UnalignReg u0;
    MicroAPI::UnalignReg u1;
    MicroAPI::DataCopyUnAlignPre(u0, ubScaleAddr);
    if constexpr (hasOffset) {
      MicroAPI::DataCopyUnAlignPre(u1, ubOffsetAddr);
    }
    __ubuf__ Q_T* ubDstAddr1 = ubDstAddr + blockStride * 128;
    __ubuf__ Q_T* ubDstAddr2 = ubDstAddr + blockStride * 128 * 2;
    __ubuf__ Q_T* ubDstAddr3 = ubDstAddr + blockStride * 128 * 3;
    for (uint16_t j = 0; j < static_cast<uint16_t>(dealRowCount); j++) {
      // 读入每行的伪量化参数
      MicroAPI::DataCopyUnAlign<ANTIQ_PARAMS_T>(vScale, u0, ubScaleAddr, 1); // 1表示ub自动往后偏移1个float
      MicroAPI::Cast<Q_T, ANTIQ_PARAMS_T, castTrait0>(vScaleFp16, vScale, maskOne);
      MicroAPI::Duplicate<uint16_t, MicroAPI::HighLowPart::LOWEST, MicroAPI::MaskMergeMode::ZEROING>
        ((MicroAPI::RegTensor<uint16_t>&)vScaleFp16, (MicroAPI::RegTensor<uint16_t>&)vScaleFp16, qMaskAll);
      if constexpr (hasOffset) {
        MicroAPI::DataCopyUnAlign<ANTIQ_PARAMS_T>(vOffset, u1, ubOffsetAddr, 1); // 1表示ub自动往后偏移1个float
        MicroAPI::Cast<Q_T, ANTIQ_PARAMS_T, castTrait0>(vOffsetFp16, vOffset, maskOne);
        MicroAPI::Duplicate<uint16_t, MicroAPI::HighLowPart::LOWEST, MicroAPI::MaskMergeMode::ZEROING>
          ((MicroAPI::RegTensor<uint16_t>&)vOffsetFp16, (MicroAPI::RegTensor<uint16_t>&)vOffsetFp16, qMaskAll);
      }
      MicroAPI::DataCopy<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_UNPACK_B8>(
        (MicroAPI::RegTensor<uint8_t>&)vKvData, ubSrcAddr, 512); // d=512 每次往后偏移512
      MicroAPI::DataCopy<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_UNPACK_B8>(
        (MicroAPI::RegTensor<uint8_t>&)vKvData1, ubSrcAddr1, 512); // d=512
      MicroAPI::DataCopy<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_UNPACK_B8>(
        (MicroAPI::RegTensor<uint8_t>&)vKvData2, ubSrcAddr2, 512); // d=512
      MicroAPI::DataCopy<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_UNPACK_B8>(
        (MicroAPI::RegTensor<uint8_t>&)vKvData3, ubSrcAddr3, 512); // d=512
      if constexpr (std::is_same<Q_T, bfloat16_t>::value) {
        MicroAPI::Cast<half, KV_T, castTrait>(vCastFp16Res, vKvData, kvMaskAll);
        MicroAPI::Cast<half, KV_T, castTrait>(vCastFp16Res1, vKvData1, kvMaskAll);
        MicroAPI::Cast<half, KV_T, castTrait>(vCastFp16Res2, vKvData2, kvMaskAll);
        MicroAPI::Cast<half, KV_T, castTrait>(vCastFp16Res3, vKvData3, kvMaskAll);
        MicroAPI::Cast<Q_T, half, castTrait1>(vRes, vCastFp16Res, kvMaskAll);
        MicroAPI::Cast<Q_T, half, castTrait1>(vRes1, vCastFp16Res1, kvMaskAll);
        MicroAPI::Cast<Q_T, half, castTrait1>(vRes2, vCastFp16Res2, kvMaskAll);
        MicroAPI::Cast<Q_T, half, castTrait1>(vRes3, vCastFp16Res3, kvMaskAll);
      } else {
        MicroAPI::Cast<Q_T, KV_T, castTrait>(vRes, vKvData, kvMaskAll);
        MicroAPI::Cast<Q_T, KV_T, castTrait>(vRes1, vKvData1, kvMaskAll);
        MicroAPI::Cast<Q_T, KV_T, castTrait>(vRes2, vKvData2, kvMaskAll);
        MicroAPI::Cast<Q_T, KV_T, castTrait>(vRes3, vKvData3, kvMaskAll);
      }
      if constexpr (hasOffset) {
        MicroAPI::Add<Q_T, MicroAPI::MaskMergeMode::ZEROING>(vRes, vRes, vOffsetFp16, qMaskAll);
        MicroAPI::Add<Q_T, MicroAPI::MaskMergeMode::ZEROING>(vRes1, vRes1, vOffsetFp16, qMaskAll);
        MicroAPI::Add<Q_T, MicroAPI::MaskMergeMode::ZEROING>(vRes2, vRes2, vOffsetFp16, qMaskAll);
        MicroAPI::Add<Q_T, MicroAPI::MaskMergeMode::ZEROING>(vRes3, vRes3, vOffsetFp16, qMaskAll);
      }

      MicroAPI::Mul<Q_T, MicroAPI::MaskMergeMode::ZEROING>(vRes, vRes, vScaleFp16, qMaskAll);
      MicroAPI::Mul<Q_T, MicroAPI::MaskMergeMode::ZEROING>(vRes1, vRes1, vScaleFp16, qMaskAll);
      MicroAPI::Mul<Q_T, MicroAPI::MaskMergeMode::ZEROING>(vRes2, vRes2, vScaleFp16, qMaskAll);
      MicroAPI::Mul<Q_T, MicroAPI::MaskMergeMode::ZEROING>(vRes3, vRes3, vScaleFp16, qMaskAll);
      MicroAPI::DataCopy<Q_T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
          ubDstAddr, vRes, blockStride, repeatStride, qMaskAll);
      MicroAPI::DataCopy<Q_T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
          ubDstAddr1, vRes1, blockStride, repeatStride, qMaskAll);
      MicroAPI::DataCopy<Q_T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
          ubDstAddr2, vRes2, blockStride, repeatStride, qMaskAll);
      MicroAPI::DataCopy<Q_T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
          ubDstAddr3, vRes3, blockStride, repeatStride, qMaskAll);
    }
  }
}

template <typename Q_T, typename ANTIQ_PARAMS_T, uint32_t baseSize, bool hasOffset = false, bool isPerToken = false, bool isNz = false>
__aicore__ inline void AntiquantVFImpl(LocalTensor<int8_t>& antiqInUb, LocalTensor<Q_T>& antiqResUb,
                                       LocalTensor<ANTIQ_PARAMS_T>& antiqOffsetUb, LocalTensor<ANTIQ_PARAMS_T>& antiqScaleUb,
                                       uint32_t dealRowCount, uint32_t headDim) {
  if constexpr (isNz) {
    if constexpr (!isPerToken) {
      AntiquantVFImplW8Nz<Q_T, int8_t, ANTIQ_PARAMS_T, baseSize, hasOffset>
        (antiqInUb, antiqResUb, antiqOffsetUb, antiqScaleUb, dealRowCount);
    }
  } else {
    if constexpr (isPerToken) {
      if constexpr (baseSize == 64) {
        AntiquantVFImplW8PerTokenD64<Q_T, int8_t, ANTIQ_PARAMS_T, baseSize, hasOffset>
          (antiqInUb, antiqResUb, antiqOffsetUb, antiqScaleUb, dealRowCount);
      } else if constexpr (baseSize == 128) {
        AntiquantVFImplW8PerTokenD128<Q_T, int8_t, ANTIQ_PARAMS_T, baseSize, hasOffset>
          (antiqInUb, antiqResUb, antiqOffsetUb, antiqScaleUb, dealRowCount);
      } else if constexpr (baseSize == 256) {
        AntiquantVFImplW8PerTokenD256<Q_T, int8_t, ANTIQ_PARAMS_T, baseSize, hasOffset>
          (antiqInUb, antiqResUb, antiqOffsetUb, antiqScaleUb, dealRowCount);
      } else {
        AntiquantVFImplW8PerTokenD512<Q_T, int8_t, ANTIQ_PARAMS_T, baseSize, hasOffset>
          (antiqInUb, antiqResUb, antiqOffsetUb, antiqScaleUb, dealRowCount);
      }
    } else {
      if constexpr (baseSize == 64) {
        AntiquantVFImplW8D64<Q_T, int8_t, ANTIQ_PARAMS_T, baseSize, hasOffset>
          (antiqInUb, antiqResUb, antiqOffsetUb, antiqScaleUb, dealRowCount);
      } else {
        AntiquantVFImplW8Norm<Q_T, int8_t, ANTIQ_PARAMS_T, baseSize, hasOffset>
          (antiqInUb, antiqResUb, antiqOffsetUb, antiqScaleUb, dealRowCount);
      }
    }
  }
}

template <typename Q_T, typename ANTIQ_PARAMS_T, uint32_t baseSize, bool hasOffset = false, bool isPerToken = false, bool isNz = false>
__aicore__ inline void AntiquantVFImpl(LocalTensor<hifloat8_t>& antiqInUb, LocalTensor<Q_T>& antiqResUb,
                                       LocalTensor<Q_T>& antiqOffsetUb, LocalTensor<Q_T>& antiqScaleUb,
                                       uint32_t dealRowCount, uint32_t headDim) {
  if constexpr (isNz) {
    AntiquantVFImplW8Nz<Q_T, hifloat8_t, ANTIQ_PARAMS_T, baseSize, hasOffset>
      (antiqInUb, antiqResUb, antiqOffsetUb, antiqScaleUb, dealRowCount);
  } else {
    if constexpr (baseSize == 64) {
      AntiquantVFImplW8D64<Q_T, hifloat8_t, ANTIQ_PARAMS_T, baseSize, hasOffset>(antiqInUb, antiqResUb, antiqOffsetUb, antiqScaleUb,
                                                              dealRowCount);
    } else {
      AntiquantVFImplW8Norm<Q_T, hifloat8_t, ANTIQ_PARAMS_T, baseSize, hasOffset>(antiqInUb, antiqResUb, antiqOffsetUb, antiqScaleUb,
                                                            dealRowCount);
    }
  }
}

template <typename Q_T, typename KV_T, uint32_t baseSize>
__aicore__ inline void AntiquantVFImplFp8D64(LocalTensor<KV_T>& antiqInUb, LocalTensor<Q_T>& antiqResUb,
                                          LocalTensor<Q_T>& antiqScaleFp16Ub, uint32_t dealRowCount) {
  static_assert(baseSize == 64);
  static_assert(IsSameType<KV_T, fp8_e4m3fn_t>::value || IsSameType<KV_T, fp8_e5m2_t>::value,
                "antiquant w8, KV_T must be fp8_e4m3fn_t or fp8_e5m2_t");

  __VEC_SCOPE__ {
    __ubuf__ uint8_t* ubSrcAddr = (__ubuf__ uint8_t*)(antiqInUb.GetPhyAddr());
    __ubuf__ Q_T* ubDstAddr = (__ubuf__ Q_T*)(antiqResUb.GetPhyAddr());
    __ubuf__ Q_T *ubDstAddr_ = ubDstAddr + 16 - (dealRowCount + 1) * 32 * 4 / 2;
    __ubuf__ Q_T* ubScalerSrcAddr = (__ubuf__ Q_T*)antiqScaleFp16Ub.GetPhyAddr();

    MicroAPI::RegTensor<KV_T> vKvData;
    MicroAPI::RegTensor<float> vCastFp32Res0;
    MicroAPI::RegTensor<float> vCastFp32Res1;
    MicroAPI::RegTensor<Q_T> vCastRes0;
    MicroAPI::RegTensor<Q_T> vCastRes1;
    MicroAPI::RegTensor<Q_T> vScale;
    MicroAPI::RegTensor<Q_T> vMulRes;

    MicroAPI::MaskReg kvTypeMaskAll = MicroAPI::CreateMask<KV_T, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg qTypeMaskLower64 = MicroAPI::CreateMask<Q_T, MicroAPI::MaskPattern::VL64>();
    MicroAPI::MaskReg qTypeMaskLower128 = MicroAPI::CreateMask<Q_T, MicroAPI::MaskPattern::VL128>();
    MicroAPI::MaskReg qTypeMaskAll = MicroAPI::CreateMask<Q_T, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg qTypeMaskHigher64;
    // NZ:
    uint32_t blockStride = dealRowCount + 1;
    uint32_t repeatStride = 2;
    uint16_t loopCnt = static_cast<uint16_t>((dealRowCount + 1) / 2); // +1是为了兼容处理奇数行

    __ubuf__ uint8_t* ubSrcTemp = ubSrcAddr;

    // 加载 scale
    MicroAPI::DataCopy<Q_T, MicroAPI::LoadDist::DIST_NORM>(vScale, ubScalerSrcAddr);
    MicroAPI::MaskXor(qTypeMaskHigher64, qTypeMaskLower64, qTypeMaskAll, qTypeMaskAll); // qTypeMaskAll与qTypeMaskLower64异或得到qTypeMaskHigher64

    // D=64时相邻2行合并做伪量化计算，减小循环次数；额外+1是为了处理奇数行时场景
    for (uint16_t i = 0; i < loopCnt; i++) {
      __ubuf__ Q_T* ubDstAddrEven = (__ubuf__ Q_T*)(antiqResUb.GetPhyAddr());
      MicroAPI::DataCopy<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_UNPACK_B16>(
          (MicroAPI::RegTensor<uint8_t>&)vKvData, ubSrcTemp, baseSize * 2);

      // cast操作, Fp8->Fp32
      MicroAPI::Cast<float, KV_T, castTraitFp8_1>(vCastFp32Res0, vKvData, kvTypeMaskAll);
      MicroAPI::Cast<float, KV_T, castTraitFp8_2>(vCastFp32Res1, vKvData, kvTypeMaskAll);

      // cast操作, Fp32->Fp16/Bf16
      MicroAPI::Cast<Q_T, float, castTraitFp8_3>(vCastRes0, vCastFp32Res0, kvTypeMaskAll);
      MicroAPI::Cast<Q_T, float, castTraitFp8_4>(vCastRes1, vCastFp32Res1, kvTypeMaskAll);

      MicroAPI::Or<uint16_t, MicroAPI::MaskMergeMode::ZEROING>((MicroAPI::RegTensor<uint16_t>&)vCastRes0,
                                                              (MicroAPI::RegTensor<uint16_t>&)vCastRes0,
                                                              (MicroAPI::RegTensor<uint16_t>&)vCastRes1, kvTypeMaskAll);

      MicroAPI::Mul<Q_T, MicroAPI::MaskMergeMode::ZEROING>(vMulRes, vCastRes0, vScale, qTypeMaskLower128);

      // 将输出结果copy到UB
      MicroAPI::DataCopy<Q_T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
          ubDstAddr, vMulRes, blockStride, repeatStride, qTypeMaskLower64);

      MicroAPI::DataCopy<Q_T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
          ubDstAddr_, vMulRes, blockStride, repeatStride, qTypeMaskHigher64);
    }
  }
}

template <typename Q_T, typename KV_T, uint32_t baseSize>
__aicore__ inline void AntiquantVFImplFp8Norm(LocalTensor<KV_T>& antiqInUb, LocalTensor<Q_T>& antiqResUb,
                                          LocalTensor<Q_T>& antiqScaleFp16Ub, uint32_t dealRowCount) {
  static_assert(baseSize % 128 == 0);
  static_assert(IsSameType<KV_T, fp8_e4m3fn_t>::value || IsSameType<KV_T, fp8_e5m2_t>::value,
                "antiquant w8, KV_T must be fp8_e4m3fn_t or fp8_e5m2_t");

  __VEC_SCOPE__ {
    __ubuf__ uint8_t* ubSrcAddr = (__ubuf__ uint8_t*)(antiqInUb.GetPhyAddr());
    __ubuf__ Q_T* ubDstAddr = (__ubuf__ Q_T*)(antiqResUb.GetPhyAddr());
    __ubuf__ Q_T* ubScalerSrcAddr = (__ubuf__ Q_T*)antiqScaleFp16Ub.GetPhyAddr();

    MicroAPI::RegTensor<KV_T> vKvData;
    MicroAPI::RegTensor<float> vCastFp32Res0;
    MicroAPI::RegTensor<float> vCastFp32Res1;
    MicroAPI::RegTensor<Q_T> vCastRes0;
    MicroAPI::RegTensor<Q_T> vCastRes1;
    MicroAPI::RegTensor<Q_T> vScale;
    MicroAPI::RegTensor<Q_T> vMulRes;

    MicroAPI::MaskReg kvTypeMaskAll = MicroAPI::CreateMask<KV_T, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg qTypeMaskAll = MicroAPI::CreateMask<Q_T, MicroAPI::MaskPattern::ALL>();

    // 目前不支持D泛化, 仅支持D128对齐场景

    // NZ:
    uint32_t blockStride = dealRowCount + 1;
    uint32_t repeatStride = 1;
    const uint16_t loops = static_cast<uint16_t>(baseSize / 128);
    for (uint16_t j = 0; j < loops; j++) {
      __ubuf__ Q_T* ubDstAddrOdd = ubDstAddr + blockStride * 128 * j;
      __ubuf__ uint8_t* ubSrcTemp = ubSrcAddr + j * 128;
      // 加载 scale
      MicroAPI::DataCopy<Q_T, MicroAPI::LoadDist::DIST_NORM>(vScale, ubScalerSrcAddr + j * 128);

      for (uint16_t i = 0; i < static_cast<uint16_t>(dealRowCount); i++) {  // 共处理dealRowCount * 128个Fp8元素
        MicroAPI::DataCopy<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_UNPACK_B16>(
            (MicroAPI::RegTensor<uint8_t>&)vKvData, ubSrcTemp, baseSize);

        // cast操作, Fp8->Fp32
        MicroAPI::Cast<float, KV_T, castTraitFp8_1>(vCastFp32Res0, vKvData, kvTypeMaskAll);
        MicroAPI::Cast<float, KV_T, castTraitFp8_2>(vCastFp32Res1, vKvData, kvTypeMaskAll);

        // cast操作, Fp32->Fp16/Bf16
        MicroAPI::Cast<Q_T, float, castTraitFp8_3>(vCastRes0, vCastFp32Res0, kvTypeMaskAll);
        MicroAPI::Cast<Q_T, float, castTraitFp8_4>(vCastRes1, vCastFp32Res1, kvTypeMaskAll);

        MicroAPI::Or<uint16_t, MicroAPI::MaskMergeMode::ZEROING>((MicroAPI::RegTensor<uint16_t>&)vCastRes0,
                                                                (MicroAPI::RegTensor<uint16_t>&)vCastRes0,
                                                                (MicroAPI::RegTensor<uint16_t>&)vCastRes1, kvTypeMaskAll);

        MicroAPI::Mul<Q_T, MicroAPI::MaskMergeMode::ZEROING>(vMulRes, vCastRes0, vScale, qTypeMaskAll);

        // 将输出结果copy到UB
        MicroAPI::DataCopy<Q_T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            ubDstAddrOdd, vMulRes, blockStride, repeatStride, qTypeMaskAll);
      }
    }
  }
}

template <typename Q_T, typename ANTIQ_PARAMS_T, uint32_t baseSize, bool hasOffset = false, bool isPerToken = false, bool isNz = false>
__aicore__ inline void AntiquantVFImpl(LocalTensor<fp8_e5m2_t>& antiqInUb, LocalTensor<Q_T>& antiqResUb,
                                       LocalTensor<ANTIQ_PARAMS_T>& antiqOffsetUb, LocalTensor<ANTIQ_PARAMS_T>& antiqScaleUb,
                                       uint32_t dealRowCount, uint32_t headDim) {
  if constexpr (isNz) {
    AntiquantVFImplFp8Nz<Q_T, fp8_e5m2_t, baseSize>(antiqInUb, antiqResUb, antiqScaleUb, dealRowCount);
  } else {
    if constexpr (baseSize == 64) {
      AntiquantVFImplFp8D64<Q_T, fp8_e5m2_t, baseSize>(antiqInUb, antiqResUb, antiqScaleUb, dealRowCount);
    } else {
      AntiquantVFImplFp8Norm<Q_T, fp8_e5m2_t, baseSize>(antiqInUb, antiqResUb, antiqScaleUb, dealRowCount);
    }
  }
}

template <typename Q_T, typename ANTIQ_PARAMS_T, uint32_t baseSize, bool hasOffset = false, bool isPerToken = false, bool isNz = false>
__aicore__ inline void AntiquantVFImpl(LocalTensor<fp8_e4m3fn_t>& antiqInUb, LocalTensor<Q_T>& antiqResUb,
                                       LocalTensor<ANTIQ_PARAMS_T>& antiqOffsetUb, LocalTensor<ANTIQ_PARAMS_T>& antiqScaleUb,
                                       uint32_t dealRowCount, uint32_t headDim) {
  if constexpr (isNz) {
    AntiquantVFImplFp8Nz<Q_T, fp8_e4m3fn_t, baseSize>(antiqInUb, antiqResUb, antiqScaleUb, dealRowCount);
  } else {
    if constexpr (baseSize == 64) {
      AntiquantVFImplFp8D64<Q_T, fp8_e4m3fn_t, baseSize>(antiqInUb, antiqResUb, antiqScaleUb, dealRowCount);
    } else {
      AntiquantVFImplFp8Norm<Q_T, fp8_e4m3fn_t, baseSize>(antiqInUb, antiqResUb, antiqScaleUb, dealRowCount);
    }
  }
}

template <typename Q_T, typename KV_T, typename ANTIQ_PARAMS_T,
  uint32_t baseSize, bool hasOffset = false, bool isPerToken = false, bool isNz = false>
__aicore__ inline void AntiquantVF(LocalTensor<KV_T>& antiqInUb, LocalTensor<Q_T>& antiqResUb,
                                   LocalTensor<ANTIQ_PARAMS_T>& antiqOffsetUb, LocalTensor<ANTIQ_PARAMS_T>& antiqScaleUb,
                                   uint32_t dealRowCount, uint32_t headDim) {
  AntiquantVFImpl<Q_T, ANTIQ_PARAMS_T, baseSize, hasOffset, isPerToken, isNz>
    (antiqInUb, antiqResUb, antiqOffsetUb, antiqScaleUb, dealRowCount, headDim);
}

};  // namespace AscendC

#endif
