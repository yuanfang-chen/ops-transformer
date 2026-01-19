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
 * \file vf_add.h
 * \brief
 */

#ifndef VF_ADD_H
#define VF_ADD_H

#include "kernel_operator.h"
using namespace AscendC;
constexpr uint32_t FLOAT_REP_SIZE = 64;
constexpr uint32_t BTYEALIGNSIZE = 32;
constexpr uint32_t REGSIZE = 256;
constexpr uint32_t FLOATSIZE = 4;
constexpr uint32_t HALFCORED = 128;

/*apeOffset —— ape的偏移
  loopCnt —— r行循环次数
  count —— regtensor上的元素数


*/
template<typename T>
__simd_vf__ void AddVFImpl(__ubuf__ T* src0Addr, __ubuf__ T* src1Addr, __ubuf__ T* apeAddr, 
   uint32_t d, uint32_t apeOffset,  uint32_t loopCnt, uint32_t count) 
{
    MicroAPI::RegTensor<T> vreg0;
    MicroAPI::RegTensor<T> vreg1;
    MicroAPI::RegTensor<T> vregape0;
    MicroAPI::RegTensor<T> vregape1;
    MicroAPI::MaskReg mask;
    for(uint32_t loop = 0; loop < loopCnt; loop++) {
        mask = MicroAPI::UpdateMask<T>(count);
        MicroAPI::LoadAlign(vreg0, src0Addr + loop*HALFCORED);
        MicroAPI::LoadAlign(vreg1, src1Addr + loop*HALFCORED);
        MicroAPI::LoadAlign(vregape0, apeAddr + loop*apeOffset);
        MicroAPI::LoadAlign(vregape1, apeAddr + d/2 + loop*apeOffset);
        MicroAPI::Add(vreg0, vreg0, vregape0, mask);
        MicroAPI::Add(vreg1, vreg1, vregape1, mask);
        MicroAPI::StoreAlign(src0Addr + loop*HALFCORED, vreg0, mask);
        MicroAPI::StoreAlign(src1Addr + loop*HALFCORED, vreg1, mask);
        
    }
}

/**
 * @brief AddVF 输入与apt相加
 * @param outputLocal 输出tensor []
 * @param inputLocal 输入tensor [row, col]
 * @param aptLocal apt输入tensor [r]
 * @param srcIdx0 input起始位置
 * @param srcIdx1 apt起始位置
 * @param d d轴总大小
 * @param splitD 核间d轴切分大小
 * @param baseD  核内d轴切分大小
 * @param splitS 核内s轴切分大小
 */
template <typename T>
__aicore__ inline void AddVF(LocalTensor<T> &src0Local, LocalTensor<T> &src1Local, LocalTensor<T> &apeLocal,
    uint32_t apeIdx, uint32_t d, uint32_t splitS) 
{
    uint32_t count = FLOAT_REP_SIZE;
    uint32_t srcIdx = count;
    uint32_t apeOffset = d;
    uint32_t loopCnt = splitS;


    __ubuf__ T * src0Addr = (__ubuf__ T *)src0Local.GetPhyAddr() + srcIdx;
    __ubuf__ T * src1Addr = (__ubuf__ T *)src1Local.GetPhyAddr() + srcIdx;
    __ubuf__ T * apeAddr = (__ubuf__ T *)apeLocal.GetPhyAddr() + apeIdx;
    
    AddVFImpl<T>(src0Addr, src1Addr, apeAddr, d, apeOffset, loopCnt, count);
}

#endif





