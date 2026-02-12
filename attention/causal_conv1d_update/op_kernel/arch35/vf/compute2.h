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
constexpr uint32_t B32_REP_SIZE = REGSIZE / sizeof(float);
constexpr uint32_t FLOAT_REP_SIZE = REGSIZE / sizeof(float);

//对stateAddr的数据进行原地读写出操作， stateAddr=yAddr
template <typename T>
__simd_vf__ void Conv1dStateVF(__ubuf__ T * xAddr,  __ubuf__ T * weightAddr, __ubuf__ T * stateAddr, __ubuf__ T * yAddr,
    uint8_t stateSLen, uint8_t xSLen, uint32_t dimLen) {
        MicroAPI::RegTensor<float> xB32, mulB32, weightB32, yB32;
        MicroAPI::RegTensor<T> xB16, weightB16, yB16;
        MicroAPI::MaskReg maskB32 = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
        uint8_t dimLoopNum = dimLen / B32_REP_SIZE;
        int32_t offset = 0;
        for(uint8_t dimLoop = 0; dimLoop < dimLoopNum; dimLoop++) {
            MicroAPI::Duplicate(yB32, 0, maskB32);
            MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
            for(uint8_t stateLoop = 0; stateLoop < stateSLen; stateLoop++) {
                MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(weightB16, weightAddr + offset + stateLoop * dimLen);
                MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(xB16, stateAddr + offset + stateLoop * dimLen);
                MicroAPI::Cast<float, T, castTraitB162B32>(weightB32, weightB16, maskB32);
                MicroAPI::Cast<float, T, castTraitB162B32>(xB32, xB16, maskB32);
                // MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_NORM>(fAddr, weightB32, maskB32);
                MicroAPI::Mul(mulB32, xB32, weightB32, maskB32);
                MicroAPI::Add(yB32, yB32, mulB32, maskB32); 
            }
            for(uint8_t xLoop = 0; xLoop < xSLen; xLoop++) {
                MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(weightB16, weightAddr + offset + (xLoop + stateSLen) * dimLen);
                MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(xB16, xAddr + offset + xLoop * dimLen);
                MicroAPI::Cast<float, T, castTraitB162B32>(weightB32, weightB16, maskB32);
                MicroAPI::Cast<float, T, castTraitB162B32>(xB32, xB16, maskB32);
                MicroAPI::Mul(mulB32, xB32, weightB32, maskB32);
                MicroAPI::Add(yB32, yB32, mulB32, maskB32);
            }
            MicroAPI::Add(yB32, yB32, xB32, maskB32);
            MicroAPI::Cast<T, float, castTraitB322B16>(yB16, yB32, maskB32);
            MicroAPI::StoreAlign<T, MicroAPI::StoreDist::DIST_PACK_B32>(yAddr + offset, yB16, maskB32);
            offset += B32_REP_SIZE;
        }
}

template <typename T>
__simd_vf__ void Conv1dStateNoConVF(__ubuf__ T * xAddr,  __ubuf__ T * weightAddr, __ubuf__ T * stateAddr, __ubuf__ T * yAddr,
    uint8_t stateSLen, uint8_t xSLen, uint32_t dimLen) {
        MicroAPI::RegTensor<float> xB32, mulB32, weightB32, yB32;
        MicroAPI::RegTensor<T> xB16, weightB16, yB16;
        MicroAPI::MaskReg maskB32 = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
        uint8_t dimLoopNum = dimLen / B32_REP_SIZE;
        int32_t offset = 0;
        for(uint8_t dimLoop = 0; dimLoop < dimLoopNum; dimLoop++) {
            MicroAPI::Duplicate(yB32, 0, maskB32);
            MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
            for(uint8_t stateLoop = 0; stateLoop < stateSLen; stateLoop++) {
                MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(weightB16, weightAddr + offset + stateLoop * dimLen);
                MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(xB16, stateAddr + offset + stateLoop * dimLen);
                MicroAPI::Cast<float, T, castTraitB162B32>(weightB32, weightB16, maskB32);
                MicroAPI::Cast<float, T, castTraitB162B32>(xB32, xB16, maskB32);
                // MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_NORM>(fAddr, weightB32, maskB32);
                MicroAPI::Mul(mulB32, xB32, weightB32, maskB32);
                MicroAPI::Add(yB32, yB32, mulB32, maskB32); 
            }
            for(uint8_t xLoop = 0; xLoop < xSLen; xLoop++) {
                MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(weightB16, weightAddr + offset + (xLoop + stateSLen) * dimLen);
                MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(xB16, xAddr + offset + xLoop * dimLen);
                MicroAPI::Cast<float, T, castTraitB162B32>(weightB32, weightB16, maskB32);
                MicroAPI::Cast<float, T, castTraitB162B32>(xB32, xB16, maskB32);
                MicroAPI::Mul(mulB32, xB32, weightB32, maskB32);
                MicroAPI::Add(yB32, yB32, mulB32, maskB32);
            }
            MicroAPI::Cast<T, float, castTraitB322B16>(yB16, yB32, maskB32);
            MicroAPI::StoreAlign<T, MicroAPI::StoreDist::DIST_PACK_B32>(yAddr + offset, yB16, maskB32);
            offset += B32_REP_SIZE;
        }
}

//对xAddr的数据进行原地读写出操作， xAddr=yAddr
template <typename T>
__simd_vf__ void Conv1dNoStateVF(__ubuf__ T * xAddr,  __ubuf__ T * weightAddr, __ubuf__ T * yAddr, uint8_t xSLen, uint32_t dimLen) 
{
    MicroAPI::RegTensor<float> weight1B32, weight2B32, weight3B32, x1B32, x2B32, x3B32;
    MicroAPI::RegTensor<T> weight1B16, weight2B16, weight3B16, x1B16, x2B16, x3B16;

    MicroAPI::RegTensor<float> mulB32, weightB32, yB32;
    MicroAPI::RegTensor<T> xB16, weightB16, yB16;
    MicroAPI::MaskReg maskB32 = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
    uint8_t dimLoopNum = dimLen / B32_REP_SIZE;
    int32_t offset = 0;

    for(uint8_t dimLoop = 0; dimLoop < dimLoopNum; dimLoop++) {
        MicroAPI::Duplicate(yB32, 0, maskB32);
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(weight1B16, weightAddr + offset);
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(weight2B16, weightAddr + offset + dimLen);
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(weight3B16, weightAddr + offset + 2 * dimLen);
        MicroAPI::Cast<float, T, castTraitB162B32>(weight1B32, weight1B16, maskB32);
        MicroAPI::Cast<float, T, castTraitB162B32>(weight2B32, weight2B16, maskB32);
        MicroAPI::Cast<float, T, castTraitB162B32>(weight3B32, weight3B16, maskB32);
        for(uint8_t xLoop = 0; xLoop < xSLen - 2; xLoop++) {
            MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
            MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(x1B16, xAddr + offset + xLoop * dimLen);
            MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(x2B16, xAddr + offset + (xLoop+1) * dimLen);
            MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(x3B16, xAddr + offset + (xLoop+2) * dimLen);
            MicroAPI::Cast<float, T, castTraitB162B32>(x1B32, x1B16, maskB32);
            MicroAPI::Cast<float, T, castTraitB162B32>(x2B32, x2B16, maskB32);
            MicroAPI::Cast<float, T, castTraitB162B32>(x3B32, x3B16, maskB32);
            MicroAPI::Mul(mulB32, x1B32, weight1B32, maskB32);
            MicroAPI::Add(yB32, yB32, mulB32, maskB32);
            MicroAPI::Mul(mulB32, x2B32, weight2B32, maskB32);
            MicroAPI::Add(yB32, yB32, mulB32, maskB32);
            MicroAPI::Mul(mulB32, x3B32, weight3B32, maskB32);
            MicroAPI::Add(yB32, yB32, mulB32, maskB32);
        }
        MicroAPI::Add(yB32, yB32, x1B32, maskB32);
        MicroAPI::Cast<T, float, castTraitB322B16>(yB16, yB32, maskB32);
        MicroAPI::StoreAlign<T, MicroAPI::StoreDist::DIST_PACK_B32>(yAddr + offset + xLoop * dimLen, yB16, maskB32);
        offset += B32_REP_SIZE;
    }
}

//对xAddr的数据进行原地读写出操作， xAddr=yAddr
template <typename T>
__simd_vf__ void Conv1dNoStateNoConVF(__ubuf__ T * xAddr,  __ubuf__ T * weightAddr, __ubuf__ T * yAddr, uint8_t xSLen, uint32_t dimLen) 
{
    MicroAPI::RegTensor<float> weight1B32, weight2B32, weight3B32, x1B32, x2B32, x3B32;
    MicroAPI::RegTensor<T> weight1B16, weight2B16, weight3B16, x1B16, x2B16, x3B16;

    MicroAPI::RegTensor<float> mulB32, weightB32, yB32;
    MicroAPI::RegTensor<T> xB16, weightB16, yB16;
    MicroAPI::MaskReg maskB32 = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
    uint8_t dimLoopNum = dimLen / B32_REP_SIZE;
    int32_t offset = 0;

    for(uint8_t dimLoop = 0; dimLoop < dimLoopNum; dimLoop++) {
        MicroAPI::Duplicate(yB32, 0, maskB32);
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(weight1B16, weightAddr + offset);
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(weight2B16, weightAddr + offset + dimLen);
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(weight3B16, weightAddr + offset + 2 * dimLen);
        MicroAPI::Cast<float, T, castTraitB162B32>(weight1B32, weight1B16, maskB32);
        MicroAPI::Cast<float, T, castTraitB162B32>(weight2B32, weight2B16, maskB32);
        MicroAPI::Cast<float, T, castTraitB162B32>(weight3B32, weight3B16, maskB32);
        for(uint8_t xLoop = 0; xLoop < xSLen - 2; xLoop++) {
            MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
            MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(x1B16, xAddr + offset + xLoop * dimLen);
            MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(x2B16, xAddr + offset + (xLoop+1) * dimLen);
            MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(x3B16, xAddr + offset + (xLoop+2) * dimLen);
            MicroAPI::Cast<float, T, castTraitB162B32>(x1B32, x1B16, maskB32);
            MicroAPI::Cast<float, T, castTraitB162B32>(x2B32, x2B16, maskB32);
            MicroAPI::Cast<float, T, castTraitB162B32>(x3B32, x3B16, maskB32);
            MicroAPI::Mul(mulB32, x1B32, weight1B32, maskB32);
            MicroAPI::Add(yB32, yB32, mulB32, maskB32);
            MicroAPI::Mul(mulB32, x2B32, weight2B32, maskB32);
            MicroAPI::Add(yB32, yB32, mulB32, maskB32);
            MicroAPI::Mul(mulB32, x3B32, weight3B32, maskB32);
            MicroAPI::Add(yB32, yB32, mulB32, maskB32);
        }
        MicroAPI::Cast<T, float, castTraitB322B16>(yB16, yB32, maskB32);
        MicroAPI::StoreAlign<T, MicroAPI::StoreDist::DIST_PACK_B32>(yAddr + offset + xLoop * dimLen, yB16, maskB32);
        offset += B32_REP_SIZE;
    }
}


template<typename T>
__aicore__ inline void Conv1dNeedState(LocalTensor<T> &xUb, LocalTensor<T> &weightUb, LocalTensor<T> &stateUb, LocalTensor<T> &yUb,
    uint8_t stateSLen, uint8_t xSLen, uint32_t dimLen, uint32_t residualConnection) {
    __ubuf__ T * xAddr = (__ubuf__ T *)xUb.GetPhyAddr();
    __ubuf__ T * weightAddr = (__ubuf__ T *)weightUb.GetPhyAddr();
    __ubuf__ T * stateAddr = (__ubuf__ T *)stateUb.GetPhyAddr();
    __ubuf__ T * yAddr = (__ubuf__ T *)yUb.GetPhyAddr();
    // PRINTF("stateSLen %d, xSLen %d, dimLen %d ", stateSLen, xSLen, dimLen);
    if(residualConnection) {
        Conv1dStateVF(xAddr, weightAddr, stateAddr , yAddr, stateSLen, xSLen, dimLen);
    } else {
        Conv1dStateNoConVF(xAddr, weightAddr, stateAddr , yAddr, stateSLen, xSLen, dimLen);
    }
}

template<typename T>
__aicore__ inline void Conv1dNoNeedState(LocalTensor<T> &xUb, LocalTensor<T> &weightUb, LocalTensor<T> &yUb,
    uint8_t xSLen, uint32_t dimLen, uint32_t residualConnection) {
    __ubuf__ T * xAddr = (__ubuf__ T *)xUb.GetPhyAddr();
    __ubuf__ T * weightAddr = (__ubuf__ T *)weightUb.GetPhyAddr();
    __ubuf__ T * yAddr = (__ubuf__ T *)yUb.GetPhyAddr();
    // PRINTF("xSLen %d, dimLen %d ",xSLen, dimLen);
    if(residualConnection) {
        Conv1dNoStateVF(xAddr, weightAddr, yAddr, xSLen, dimLen);
    } else {
        Conv1dNoStateNoConVF(xAddr, weightAddr, yAddr, xSLen, dimLen);
    }
}

#endif
