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
 * \file basic_block_vf_mx.h
 * \brief
 */
#ifndef GROUPED_MATMUL_WEIGHT_QUANT_BASIC_BLOCK_VF_MX_H
#define GROUPED_MATMUL_WEIGHT_QUANT_BASIC_BLOCK_VF_MX_H

#include "basic_block_config.h"
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#endif

namespace MicroAPI = AscendC::MicroAPI;
using AscendC::BLOCK_CUBE;
using AscendC::VECTOR_REG_WIDTH;
using AscendC::MicroAPI::AddrReg;
using AscendC::MicroAPI::MaskReg;
using AscendC::MicroAPI::RegTensor;

namespace WeightQuantBatchMatmulV2::Arch35 {

__simd_vf__ inline void InitZeroVf(__ubuf__ float * ubAddr, uint16_t loopCount)
{
    MicroAPI::RegTensor<float> zeroReg;
    MicroAPI::Duplicate(zeroVreg, 0);
    MicroAPI::MaskReg preg = MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
    for (uint16_t loopIdx = 0; loopIdx < loopCount; loopIdx++) {
        MicroAPI::AddrReg outAddrReg = MicroAPI::CreateAddrReg<float>(
            loopIdx, QUADRUPLE_BUFFER_NUM * VEC_MAX_ELEM_B32);
        MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_NORM_B32>(
            ubAddr, zeroVreg, outAddrReg, preg);
    }
}

static constexpr MicroAPI::CastTrait CAST_B16_TO_B32_TRAIT = {
    AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN, AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN};;

template <typename sharedInputType>
__simd_vf__ inline void CastAndMulWithSharedWeightVf(__ubuf__ float * dstAddr, __ubuf__ sharedInputType * srcAddr, uint16_t loopCount, float weight)
{
    MicroAPI::RegTensor<sharedInputType> sharedInputB16Vreg, 
    MicroAPI::RegTensor<float> sharedInputB32Vreg;
    MaskReg maskAll = MicroAPI::CreateMask<uint8_t, AscendC::MicroAPI::MaskPattern::ALL>();
    for (uint16_t loopIdx = 0; loopIdx < loopCount; loopIdx ++) {
        MicroAPI::AddrReg sharedInputAreg = MicroAPI::CreateAddrReg<sharedInputType>(loopBiasIdx, VEC_MAX_ELEM_B16);
        MicroAPI::LoadAlign<sharedInputType, MicroAPI::LoadDist::DIST_UNPACK_B16>(sharedInputB16Vreg, srcAddr,
                                                                        sharedInputAreg);
        MicroAPI::Cast<float, sharedInputType, CAST_B16_TO_B32_TRAIT>(
            sharedInputB32Vreg, sharedInputB16Vreg, maskAll);
        MicroAPI::Muls<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(sharedInputB32Vreg, sharedInputB32Vreg, weight,
                                                                                maskAll);
        MicroAPI::AddrReg outAddrReg = MicroAPI::CreateAddrReg<float>(
            loopIdx, QUADRUPLE_BUFFER_NUM * VEC_MAX_ELEM_B32);
        MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_NORM_B32>(
            dstAddr, sharedInputB32Vreg, outAddrReg, preg);

    }
}
}