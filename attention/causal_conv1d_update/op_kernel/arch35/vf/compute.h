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
 * \file compute.h
 * \brief
 */

#ifndef COMPUTE_H
#define COMPUTE_H

#include "kernel_operator.h"

using namespace AscendC;

constexpr MicroAPI::CastTrait castTraitB162B32 = {
    MicroAPI::RegLayout::ZERO,
    MicroAPI::SatMode::UNKNOWN,
    MicroAPI::MaskMergeMode::ZEROING,
    RoundMode::UNKNOWN,
};

constexpr MicroAPI::CastTrait castTraitB322B16 = {
    MicroAPI::RegLayout::ZERO,
    MicroAPI::SatMode::NO_SAT,
    MicroAPI::MaskMergeMode::ZEROING,
    RoundMode::CAST_RINT,
};

constexpr uint32_t REGSIZE = 256;
constexpr uint32_t B16_REP_SIZE = REGSIZE / sizeof(half);
constexpr uint32_t FLOAT_REP_SIZE = REGSIZE / sizeof(float);

//对stateAddr的数据进行原地读写出操作， stateAddr=yAddr
template <typename T>
__simd_vf__ void Conv1dNeedStateVF(__ubuf__ T * xAddr,  __ubuf__ T * weightAddr, __ubuf__ T * stateAddr, __ubuf__ T * yAddr,
    uint8_t stateSLen, uint8_t xSLen, uint8_t dimLen) {
        MicroAPI::RegTensor<float> xB32, mulB32, weightB32, yB32;
        MicroAPI::RegTensor<T> xB16, weightB16, yB16;
        MicroAPI::MaskReg maskB32, maskB16;
        maskB32 = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
        maskB16 = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::ALL>();
        uint8_t dimLoopNum = dimLen / B16_REP_SIZE;
        for(uint8_t dimLoop = 0; dimLoop < dimLoopNum; dimLoop++) {
            MicroAPI::Duplicate(yB32, 0, maskB32);
            MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
            for(uint8_t stateLoop = 0; stateLoop < stateSLen; stateLoop++) {
                MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(weightB16, weightAddr + dimLoop * B16_REP_SIZE + stateLoop * dimLen);
                MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(xB16, stateAddr + dimLoop * B16_REP_SIZE + stateLoop * dimLen);
                MicroAPI::Cast<float, T, castTraitB162B32>(weightB32, weightB16, maskB32);
                MicroAPI::Cast<float, T, castTraitB162B32>(xB32, xB16, maskB32);
                MicroAPI::Mul(mulB32, xB32, weightB32, maskB32);
                MicroAPI::Add(yB32, yB32, mulB32, maskB32); 
            }
            for(uint8_t xLoop = 0; xLoop < xSLen; xLoop++) {
                MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(weightB16, weightAddr + dimLoop * B16_REP_SIZE + (xLoop + stateSLen) * dimLen);
                MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(xB16, xAddr + + dimLoop * B16_REP_SIZE + xLoop * dimLen);
                MicroAPI::Cast<float, T, castTraitB162B32>(weightB32, weightB16, maskB32);
                MicroAPI::Cast<float, T, castTraitB162B32>(xB32, xB16, maskB32);
                MicroAPI::Mul(mulB32, xB32, weightB32, maskB32);
                MicroAPI::Add(yB32, yB32, mulB32, maskB32);
            }
            MicroAPI::Add(yB32, yB32, xB32, maskB32);
            MicroAPI::Cast<T, float, castTraitB322B16>(yB16, yB32, maskB32);
            MicroAPI::StoreAlign<ROPET, MicroAPI::StoreDist::DIST_PACK_B32>(yAddr + dimLoop * B16_REP_SIZE, yB16, maskB16);
        }

}

//对xAddr的数据进行原地读写出操作， xAddr=yAddr
template <typename T>
__simd_vf__ void Conv1dNoNeedStateVF(__ubuf__ T * xAddr,  __ubuf__ T * weightAddr, __ubuf__ T * yAddr, uint8_t xSLen, uint8_t dimLen) 
{
        MicroAPI::RegTensor<float> xB32, mulB32, weightB32, yB32;
        MicroAPI::RegTensor<T> xB16, weightB16, yB16;
        MicroAPI::MaskReg maskB32, maskB16;
        maskB32 = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
        maskB16 = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::ALL>();
        uint8_t dimLoopNum = dimLen / B16_REP_SIZE;
        for(uint8_t dimLoop = 0; dimLoop < dimLoopNum; dimLoop++) {
            MicroAPI::Duplicate(yB32, 0, maskB32);
            MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
            for(uint8_t xLoop = 0; xLoop < xSLen; xLoop++) {
                MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(weightB16, weightAddr + dimLoop * B16_REP_SIZE + xLoop * dimLen);
                MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(xB16, xAddr + dimLoop * B16_REP_SIZE + xLoop * dimLen);
                MicroAPI::Cast<float, T, castTraitB162B32>(weightB32, weightB16, maskB32);
                MicroAPI::Cast<float, T, castTraitB162B32>(xB32, xB16, maskB32);
                MicroAPI::Mul(mulB32, xB32, weightB32, maskB32);
                MicroAPI::Add(yB32, yB32, mulB32, maskB32);
            }
            MicroAPI::Add(yB32, yB32, xB32, maskB32);
            MicroAPI::Cast<T, float, castTraitB322B16>(yB16, yB32, maskB32);
            MicroAPI::StoreAlign<ROPET, MicroAPI::StoreDist::DIST_PACK_B32>(yAddr + dimLoop * B16_REP_SIZE, yB16, maskB16);
        }
}

template<typename T>
__aicore__ inline void Conv1dNeedState(LocalTensor<T> &xUb, LocalTensor<T> &weightUb, LocalTensor<T> &stateUb, LocalTensor<T> &yUb,
    uint8_t startSLen, uint8_t xSLen, uint8_t dimLen) {
    __ubuf__ T * xAddr = (__ubuf__ T *)xUb.GetPhyAddr();
    __ubuf__ T * weightAddr = (__ubuf__ T *)weightUb.GetPhyAddr();
    __ubuf__ T * stateAddr = (__ubuf__ T *)stateUb.GetPhyAddr();
    __ubuf__ T * yAddr = (__ubuf__ T *)yUb.GetPhyAddr();
    Conv1dNeedStateVF(xAddr, weightAddr, stateAddr, yAddr, startSLen, xSLen, dimLen);
}

template<typename T>
__aicore__ inline void Conv1dNoNeedState(LocalTensor<T> &xUb, LocalTensor<T> &weightUb, LocalTensor<T> &yUb, uint8_t xSLen, uint8_t dimLen) {
    __ubuf__ T * xAddr = (__ubuf__ T *)xUb.GetPhyAddr();
    __ubuf__ T * weightAddr = (__ubuf__ T *)weightUb.GetPhyAddr();
    __ubuf__ T * yAddr = (__ubuf__ T *)yUb.GetPhyAddr();
    Conv1dNoNeedStateVF(xAddr, weightAddr, yAddr, xSLen, dimLen);
}
