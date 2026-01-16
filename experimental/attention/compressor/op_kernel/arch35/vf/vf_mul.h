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
 * \file vf_mul.h
 * \brief
 */

#ifndef VF_MUL_H
#define VF_MUL_H
#include "kernel_operator.h"
using namespace AscendC;
// constexpr uint32_t FLOAT_REP_SIZE = 64;
// constexpr uint32_t BTYEALIGNSIZE = 32;
// constexpr uint32_t REGSIZE = 256;

struct LoadAlignParam{
    uint32_t dataBlockStride; //非连续对齐搬运内的首与首间的间隔，以32B为单位
    uint32_t repeatStride; //非连续对齐搬运时，地址偏移大小，以32B为单位
    uint32_t offset; //搬运结束后偏移的更新大小，以32B为单位
};

/*同时处理多个sc，单寄存器存多个sc某行的元素

  loopCnt —— 多个r需要循环的次数
  loopLeft —— 多个r循环后遗留的尾块
  regLeftNum —— 尾块regtensor上的元素数
  otherScSize —— 单次循环中处理的r的块数-1
  loadAlignParam0、loadAlignParam1
*/
template<typename T>
__simd_vf__ void MulReduceSumbaseVFImpl(__ubuf__ T* dstAddr, __ubuf__ T* src0Addr, __ubuf__ T* src1Addr,
    uint32_t r, uint32_t loopCnt, uint32_t loopLeft, uint32_t regLeftNum, uint32_t otherScSize, LoadAlignParam loadAlignParam0,
    LoadAlignParam loadAlignParam1) 
{
    MicroAPI::RegTensor<T> vreg0;
    MicroAPI::RegTensor<T> vreg1;
    MicroAPI::RegTensor<T> vregSum;
    MicroAPI::MaskReg mask;
    uint32_t count = FLOAT_REP_SIZE;
    for(uint32_t loop1 = 0; loop1 < loopCnt; loop1++) {
        mask = MicroAPI::UpdateMask<T>(count);
        MicroAPI::Duplicate(vregSum, 0, mask);
        for(uint32_t loop2 = 0; loop2 < 2 * r; loop2++) {
            MicroAPI::LoadAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_NORMAL>
                        (vreg0, src0Addr, loadAlignParam0.dataBlockStride, loadAlignParam0.repeatStride, mask);
            MicroAPI::LoadAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_NORMAL>
                        (vreg1, src1Addr, loadAlignParam1.dataBlockStride, loadAlignParam1.repeatStride, mask);
            MicroAPI::Mul(vreg0, vreg0, vreg1, mask);
            loadAlignParam0.repeatStride += loadAlignParam0.offset;
            loadAlignParam1.repeatStride += loadAlignParam1.offset;
        }
        MicroAPI::Add(vregSum, vreg1, vreg0, mask);
        MicroAPI::StoreAlign(dstAddr + loop1 * FLOAT_REP_SIZE, vregSum, mask);
        loadAlignParam0.repeatStride += otherScSize;
        loadAlignParam1.repeatStride += otherScSize;
    }
    if(loopLeft > 0) {
        for(uint32_t loop = 0; loop < 2*r; loop++) {
            mask = MicroAPI::UpdateMask<T>(regLeftNum);
            MicroAPI::LoadAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_NORMAL>
                        (vreg0, src0Addr, loadAlignParam0.dataBlockStride, loadAlignParam0.repeatStride, mask);
            MicroAPI::LoadAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_NORMAL>
                        (vreg1, src1Addr, loadAlignParam1.dataBlockStride, loadAlignParam1.repeatStride, mask);
            MicroAPI::Mul(vreg0, vreg0, vreg1, mask);
            loadAlignParam0.repeatStride += loadAlignParam0.offset;
            loadAlignParam1.repeatStride += loadAlignParam1.offset;

        }
        MicroAPI::Add(vregSum, vreg1, vreg0, mask);
        MicroAPI::StoreAlign(dstAddr + loopCnt * FLOAT_REP_SIZE, vregSum, mask);
    }
}

/**
 * @brief MulReduceSumbaseVF 包含mul和reducesum
 * @param outputLocal 输出tensor []
 * @param input1Local 输入tensor0 [row, col]
 * @param input1Local 输入tensor1 [r]
 * @param srcIdx0 input0起始位置
 * @param srcIdx1 input1起始位置
 * @param r s方向的最小块
 * @param sc r个sc
 * @param baseD  核内d轴切分大小
 */


template<typename T>
__aicore__ inline void MulReduceSumbaseVF(const LocalTensor<T> &outputLocal, const LocalTensor<T> &inputLocal, const LocalTensor<T> &aptLocal,
    uint32_t srcIdx0, uint32_t srcIdx1, uint32_t r, uint32_t sc, uint32_t baseD) 
{
    uint32_t regSplitNum = FLOAT_REP_SIZE / baseD;
    uint32_t loopCnt = sc / regSplitNum;
    uint32_t loopLeft = sc - loopCnt * regSplitNum;
    uint32_t regLeftNum = loopLeft * baseD;
    uint32_t otherScSize = (regSplitNum - 1) * baseD * 2 * r * sizeof(T) / BTYEALIGNSIZE;
    LoadAlignParam loadAlignParam0;
    LoadAlignParam loadAlignParam1;
    loadAlignParam0.dataBlockStride = r * baseD * sizeof(T) / BTYEALIGNSIZE;
    loadAlignParam0.repeatStride = srcIdx0;
    loadAlignParam0.offset = baseD * sizeof(T) / BTYEALIGNSIZE;
    loadAlignParam1.dataBlockStride = r * baseD * sizeof(T) / BTYEALIGNSIZE;
    loadAlignParam1.repeatStride = srcIdx1;
    loadAlignParam1.offset = baseD * sizeof(T) / BTYEALIGNSIZE;

    __ubuf__ T * inputAddr = (__ubuf__ T *)inputLocal.GetPhyAddr();
    __ubuf__ T * aptAddr = (__ubuf__ T *)aptLocal.GetPhyAddr();
    __ubuf__ T * outputAddr = (__ubuf__ T *)outputLocal.GetPhyAddr();
    MulReduceSumbaseVFImpl(outputAddr, inputAddr, aptAddr, r, loopCnt, loopLeft, regLeftNum,
        otherScSize, loadAlignParam0, loadAlignParam1);

}


#endif