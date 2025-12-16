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
 * \file vf_softmax_grad_front_cast.h
 */
#ifndef MY_SOFTMAX_GRAD_CAST_ALIGNED512_F32_INTERFACE_H_
#define MY_SOFTMAX_GRAD_CAST_ALIGNED512_F32_INTERFACE_H_
#include "kernel_tensor.h"

namespace AscendC {
#ifndef __CCE_KT_TEST__
using namespace MicroAPI;
/* **************************************************************************************************

SoftmaxGradFrontCast *
************************************************************************************************* /

@INGROUP SoftmaxGradFrontCast
brief compute :sum = reducesum(cast(grad) * cast(x))
param [out] dstTensor output LocalTensor
param [in] gradTensor input grad LocalTensor
param [in] srcTensor input src LocalTensor
*/
template <typename T1, typename T, uint32_t srcN, uint32_t HEAD_DIM_ALIGN>
__aicore__ inline void MySoftmaxGradFrontCastAligned512F32(const LocalTensor<T> &dstTensor, const LocalTensor<T1> &gradTensor,
                                                const LocalTensor<T1> &srcTensor, uint32_t srcM, uint32_t realN = srcN)
{
    uint64_t srcLocalInt = srcTensor.GetPhyAddr();
    uint64_t dstLocalInt = dstTensor.GetPhyAddr();
    uint64_t gradLocalInt = gradTensor.GetPhyAddr();

    if constexpr (srcN == 320) {
        // D=320 unroll4次
        const uint32_t fullExeSize = 64;
        uint64_t srcLocalInt1 = srcTensor.GetPhyAddr() + fullExeSize * sizeof(T1);
        uint64_t gradLocalInt1 = gradTensor.GetPhyAddr() + fullExeSize * sizeof(T1);

        uint64_t srcLocalInt2 = srcTensor.GetPhyAddr() + fullExeSize * 2 * sizeof(T1);
        uint64_t gradLocalInt2 = gradTensor.GetPhyAddr() + fullExeSize * 2 * sizeof(T1);

        uint64_t srcLocalInt3 = srcTensor.GetPhyAddr() + fullExeSize * 3 * sizeof(T1);
        uint64_t gradLocalInt3 = gradTensor.GetPhyAddr() + fullExeSize * 3 * sizeof(T1);

        uint64_t srcLocalIntTail = srcTensor.GetPhyAddr() + fullExeSize * 4 * sizeof(T1);
        uint64_t gradLocalIntTail = gradTensor.GetPhyAddr() + fullExeSize * 4 * sizeof(T1);

        uint32_t tailSize = realN % fullExeSize;
        uint32_t reduceSize = tailSize == 0 ? fullExeSize : tailSize;

        __VEC_SCOPE__
        {
            RegTensor<float> vregSrc;
            RegTensor<float> vregGrad;
            RegTensor<float> vregSrc1;
            RegTensor<float> vregGrad1;
            RegTensor<float> vregSrc2;
            RegTensor<float> vregGrad2;
            RegTensor<float> vregSrc3;
            RegTensor<float> vregGrad3;
            RegTensor<float> vregMul;
            RegTensor<float> vregMul1;
            RegTensor<float> vregMul2;
            RegTensor<float> vregMul3;
            RegTensor<float> vregSrcTail;
            RegTensor<float> vregGradTail;
            RegTensor<float> vregMulTail;
            RegTensor<float> vregAdd;
            RegTensor<float> vregAdd1;
            RegTensor<float> vregAdd2;
            RegTensor<float> vregAdd3;
            RegTensor<float> vregReduceSum;
            MaskReg pregFullExe = CreateMask<float, MaskPattern::ALL>();
            UnalignReg uregReduceSum;
            MaskReg pregTailExe = UpdateMask<float>(reduceSize);

            for (uint16_t m = 0; m < static_cast<uint16_t>(srcM); m++) {
                // 手动unroll 128个数分64个数做mul和add
                DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrc, ((__ubuf__ T1 *&)srcLocalInt), HEAD_DIM_ALIGN);
                DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGrad, ((__ubuf__ T1 *&)gradLocalInt), HEAD_DIM_ALIGN);
                DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrc1, ((__ubuf__ T1 *&)srcLocalInt1), HEAD_DIM_ALIGN);
                DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGrad1, ((__ubuf__ T1 *&)gradLocalInt1), HEAD_DIM_ALIGN);
                DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrc2, ((__ubuf__ T1 *&)srcLocalInt2), HEAD_DIM_ALIGN);
                DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGrad2, ((__ubuf__ T1 *&)gradLocalInt2), HEAD_DIM_ALIGN);
                DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrc3, ((__ubuf__ T1 *&)srcLocalInt3), HEAD_DIM_ALIGN);
                DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGrad3, ((__ubuf__ T1 *&)gradLocalInt3), HEAD_DIM_ALIGN);
                DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrcTail, ((__ubuf__ T1 *&)srcLocalIntTail), HEAD_DIM_ALIGN);
                DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGradTail, ((__ubuf__ T1 *&)gradLocalIntTail), HEAD_DIM_ALIGN);
                Mul(vregMul, vregGrad, vregSrc, pregFullExe);
                Mul(vregMul1, vregGrad1, vregSrc1, pregFullExe);
                Mul(vregMul2, vregGrad2, vregSrc2, pregFullExe);
                Mul(vregMul3, vregGrad3, vregSrc3, pregFullExe);
                Mul(vregMulTail, vregGradTail, vregSrcTail, pregTailExe);
                Add(vregAdd, vregMul, vregMulTail, pregFullExe);
                Add(vregAdd1, vregMul1, vregMul3, pregFullExe);
                Add(vregAdd2, vregAdd, vregMul2, pregFullExe);
                Add(vregAdd3, vregAdd2, vregAdd1, pregFullExe);
                ReduceSum(vregReduceSum, vregAdd3, pregFullExe);
                DataCopyUnAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ float *&)dstLocalInt), vregReduceSum, uregReduceSum, 1);
            }
            vstas(uregReduceSum, ((__ubuf__ float *&)dstLocalInt), 0, POST_UPDATE);
        }
    } else if constexpr (srcN == 384) {
        // D=384 unroll5次
        const uint32_t fullExeSize = 64;
        uint64_t srcLocalInt1 = srcTensor.GetPhyAddr() + fullExeSize * sizeof(T1);
        uint64_t gradLocalInt1 = gradTensor.GetPhyAddr() + fullExeSize * sizeof(T1);

        uint64_t srcLocalInt2 = srcTensor.GetPhyAddr() + fullExeSize * 2 * sizeof(T1);
        uint64_t gradLocalInt2 = gradTensor.GetPhyAddr() + fullExeSize * 2 * sizeof(T1);

        uint64_t srcLocalInt3 = srcTensor.GetPhyAddr() + fullExeSize * 3 * sizeof(T1);
        uint64_t gradLocalInt3 = gradTensor.GetPhyAddr() + fullExeSize * 3 * sizeof(T1);

        uint64_t srcLocalInt4 = srcTensor.GetPhyAddr() + fullExeSize * 4 * sizeof(T1);
        uint64_t gradLocalInt4 = gradTensor.GetPhyAddr() + fullExeSize * 4 * sizeof(T1);

        uint64_t srcLocalIntTail = srcTensor.GetPhyAddr() + fullExeSize * 5 * sizeof(T1);
        uint64_t gradLocalIntTail = gradTensor.GetPhyAddr() + fullExeSize * 5 * sizeof(T1);

        uint32_t tailSize = realN % fullExeSize;
        uint32_t reduceSize = tailSize == 0 ? fullExeSize : tailSize;

        __VEC_SCOPE__
        {
            RegTensor<float> vregSrc;
            RegTensor<float> vregGrad;
            RegTensor<float> vregSrc1;
            RegTensor<float> vregGrad1;
            RegTensor<float> vregSrc2;
            RegTensor<float> vregGrad2;
            RegTensor<float> vregSrc3;
            RegTensor<float> vregGrad3;
            RegTensor<float> vregSrc4;
            RegTensor<float> vregGrad4;
            RegTensor<float> vregMul;
            RegTensor<float> vregMul1;
            RegTensor<float> vregMul2;
            RegTensor<float> vregMul3;
            RegTensor<float> vregMul4;
            RegTensor<float> vregSrcTail;
            RegTensor<float> vregGradTail;
            RegTensor<float> vregMulTail;
            RegTensor<float> vregAdd;
            RegTensor<float> vregAdd1;
            RegTensor<float> vregAdd2;
            RegTensor<float> vregAdd3;
            RegTensor<float> vregReduceSum;
            MaskReg pregFullExe = CreateMask<float, MaskPattern::ALL>();
            UnalignReg uregReduceSum;
            MaskReg pregTailExe = UpdateMask<float>(reduceSize);

            for (uint16_t m = 0; m < static_cast<uint16_t>(srcM); m++) {
                // 手动unroll 128个数分64个数做mul和add
                DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrc, ((__ubuf__ T1 *&)srcLocalInt), HEAD_DIM_ALIGN);
                DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGrad, ((__ubuf__ T1 *&)gradLocalInt), HEAD_DIM_ALIGN);
                DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrc1, ((__ubuf__ T1 *&)srcLocalInt1), HEAD_DIM_ALIGN);
                DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGrad1, ((__ubuf__ T1 *&)gradLocalInt1), HEAD_DIM_ALIGN);
                DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrc2, ((__ubuf__ T1 *&)srcLocalInt2), HEAD_DIM_ALIGN);
                DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGrad2, ((__ubuf__ T1 *&)gradLocalInt2), HEAD_DIM_ALIGN);
                DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrc3, ((__ubuf__ T1 *&)srcLocalInt3), HEAD_DIM_ALIGN);
                DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGrad3, ((__ubuf__ T1 *&)gradLocalInt3), HEAD_DIM_ALIGN);
                DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrc4, ((__ubuf__ T1 *&)srcLocalInt4), HEAD_DIM_ALIGN);
                DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGrad4, ((__ubuf__ T1 *&)gradLocalInt4), HEAD_DIM_ALIGN);
                DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrcTail, ((__ubuf__ T1 *&)srcLocalIntTail), HEAD_DIM_ALIGN);
                DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGradTail, ((__ubuf__ T1 *&)gradLocalIntTail), HEAD_DIM_ALIGN);
                Mul(vregMul, vregGrad, vregSrc, pregFullExe);
                Mul(vregMul1, vregGrad1, vregSrc1, pregFullExe);
                Mul(vregMul2, vregGrad2, vregSrc2, pregFullExe);
                Mul(vregMul3, vregGrad3, vregSrc3, pregFullExe);
                Mul(vregMul4, vregGrad4, vregSrc4, pregFullExe);
                Mul(vregMulTail, vregGradTail, vregSrcTail, pregTailExe);
                Add(vregAdd, vregMul, vregMulTail, pregFullExe);
                Add(vregAdd1, vregMul1, vregMul4, pregFullExe);
                Add(vregAdd2, vregMul2, vregMul3, pregFullExe);
                Add(vregAdd3, vregAdd, vregAdd1, pregFullExe);
                Add(vregAdd, vregAdd3, vregAdd2, pregFullExe);
                ReduceSum(vregReduceSum, vregAdd, pregFullExe);
                DataCopyUnAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ float *&)dstLocalInt), vregReduceSum, uregReduceSum, 1);
            }
            vstas(uregReduceSum, ((__ubuf__ float *&)dstLocalInt), 0, POST_UPDATE);
        }
    } else if constexpr (srcN == 448) {
        // D=448 unroll6次
        const uint32_t fullExeSize = 64;
        uint64_t srcLocalInt1 = srcTensor.GetPhyAddr() + fullExeSize * sizeof(T1);
        uint64_t gradLocalInt1 = gradTensor.GetPhyAddr() + fullExeSize * sizeof(T1);

        uint64_t srcLocalInt2 = srcTensor.GetPhyAddr() + fullExeSize * 2 * sizeof(T1);
        uint64_t gradLocalInt2 = gradTensor.GetPhyAddr() + fullExeSize * 2 * sizeof(T1);

        uint64_t srcLocalInt3 = srcTensor.GetPhyAddr() + fullExeSize * 3 * sizeof(T1);
        uint64_t gradLocalInt3 = gradTensor.GetPhyAddr() + fullExeSize * 3 * sizeof(T1);

        uint64_t srcLocalInt4 = srcTensor.GetPhyAddr() + fullExeSize * 4 * sizeof(T1);
        uint64_t gradLocalInt4 = gradTensor.GetPhyAddr() + fullExeSize * 4 * sizeof(T1);

        uint64_t srcLocalInt5 = srcTensor.GetPhyAddr() + fullExeSize * 5 * sizeof(T1);
        uint64_t gradLocalInt5 = gradTensor.GetPhyAddr() + fullExeSize * 5 * sizeof(T1);

        uint64_t srcLocalIntTail = srcTensor.GetPhyAddr() + fullExeSize * 6 * sizeof(T1);
        uint64_t gradLocalIntTail = gradTensor.GetPhyAddr() + fullExeSize * 6 * sizeof(T1);

        uint32_t tailSize = realN % fullExeSize;
        uint32_t reduceSize = tailSize == 0 ? fullExeSize : tailSize;

        __VEC_SCOPE__
        {
            RegTensor<float> vregSrc;
            RegTensor<float> vregGrad;
            RegTensor<float> vregSrc1;
            RegTensor<float> vregGrad1;
            RegTensor<float> vregSrc2;
            RegTensor<float> vregGrad2;
            RegTensor<float> vregSrc3;
            RegTensor<float> vregGrad3;
            RegTensor<float> vregSrc4;
            RegTensor<float> vregGrad4;
            RegTensor<float> vregSrc5;
            RegTensor<float> vregGrad5;
            RegTensor<float> vregMul;
            RegTensor<float> vregMul1;
            RegTensor<float> vregMul2;
            RegTensor<float> vregMul3;
            RegTensor<float> vregMul4;
            RegTensor<float> vregMul5;
            RegTensor<float> vregSrcTail;
            RegTensor<float> vregGradTail;
            RegTensor<float> vregMulTail;
            RegTensor<float> vregAdd;
            RegTensor<float> vregAdd1;
            RegTensor<float> vregAdd2;
            RegTensor<float> vregAdd3;
            RegTensor<float> vregAdd4;
            RegTensor<float> vregAdd5;
            RegTensor<float> vregReduceSum;
            MaskReg pregFullExe = CreateMask<float, MaskPattern::ALL>();
            UnalignReg uregReduceSum;
            MaskReg pregTailExe = UpdateMask<float>(reduceSize);

            for (uint16_t m = 0; m < static_cast<uint16_t>(srcM); m++) {
                // 手动unroll 128个数分64个数做mul和add
                DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrc, ((__ubuf__ T1 *&)srcLocalInt), HEAD_DIM_ALIGN);
                DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGrad, ((__ubuf__ T1 *&)gradLocalInt), HEAD_DIM_ALIGN);
                DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrc1, ((__ubuf__ T1 *&)srcLocalInt1), HEAD_DIM_ALIGN);
                DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGrad1, ((__ubuf__ T1 *&)gradLocalInt1), HEAD_DIM_ALIGN);
                DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrc2, ((__ubuf__ T1 *&)srcLocalInt2), HEAD_DIM_ALIGN);
                DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGrad2, ((__ubuf__ T1 *&)gradLocalInt2), HEAD_DIM_ALIGN);
                DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrc3, ((__ubuf__ T1 *&)srcLocalInt3), HEAD_DIM_ALIGN);
                DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGrad3, ((__ubuf__ T1 *&)gradLocalInt3), HEAD_DIM_ALIGN);
                DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrc4, ((__ubuf__ T1 *&)srcLocalInt4), HEAD_DIM_ALIGN);
                DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGrad4, ((__ubuf__ T1 *&)gradLocalInt4), HEAD_DIM_ALIGN);
                DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrc5, ((__ubuf__ T1 *&)srcLocalInt5), HEAD_DIM_ALIGN);
                DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGrad5, ((__ubuf__ T1 *&)gradLocalInt5), HEAD_DIM_ALIGN);
                DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrcTail, ((__ubuf__ T1 *&)srcLocalIntTail), HEAD_DIM_ALIGN);
                DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGradTail, ((__ubuf__ T1 *&)gradLocalIntTail), HEAD_DIM_ALIGN);
                Mul(vregMul, vregGrad, vregSrc, pregFullExe);
                Mul(vregMul1, vregGrad1, vregSrc1, pregFullExe);
                Mul(vregMul2, vregGrad2, vregSrc2, pregFullExe);
                Mul(vregMul3, vregGrad3, vregSrc3, pregFullExe);
                Mul(vregMul4, vregGrad4, vregSrc4, pregFullExe);
                Mul(vregMul5, vregGrad5, vregSrc5, pregFullExe);
                Mul(vregMulTail, vregGradTail, vregSrcTail, pregTailExe);
                Add(vregAdd, vregMul, vregMulTail, pregFullExe);
                Add(vregAdd1, vregMul1, vregMul5, pregFullExe);
                Add(vregAdd2, vregMul2, vregMul4, pregFullExe);
                Add(vregAdd3, vregAdd, vregMul3, pregFullExe);
                Add(vregAdd4, vregAdd3, vregAdd2, pregFullExe);
                Add(vregAdd, vregAdd4, vregAdd1, pregFullExe);
                ReduceSum(vregReduceSum, vregAdd, pregFullExe);
                DataCopyUnAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ float *&)dstLocalInt), vregReduceSum, uregReduceSum, 1);
            }
            vstas(uregReduceSum, ((__ubuf__ float *&)dstLocalInt), 0, POST_UPDATE);
        }
    } else if constexpr (srcN == 512) {
        // D=512 unroll7次
        const uint32_t fullExeSize = 64;
        uint64_t srcLocalInt1 = srcTensor.GetPhyAddr() + fullExeSize * sizeof(T1);
        uint64_t gradLocalInt1 = gradTensor.GetPhyAddr() + fullExeSize * sizeof(T1);

        uint64_t srcLocalInt2 = srcTensor.GetPhyAddr() + fullExeSize * 2 * sizeof(T1);
        uint64_t gradLocalInt2 = gradTensor.GetPhyAddr() + fullExeSize * 2 * sizeof(T1);

        uint64_t srcLocalInt3 = srcTensor.GetPhyAddr() + fullExeSize * 3 * sizeof(T1);
        uint64_t gradLocalInt3 = gradTensor.GetPhyAddr() + fullExeSize * 3 * sizeof(T1);

        uint64_t srcLocalInt4 = srcTensor.GetPhyAddr() + fullExeSize * 4 * sizeof(T1);
        uint64_t gradLocalInt4 = gradTensor.GetPhyAddr() + fullExeSize * 4 * sizeof(T1);

        uint64_t srcLocalInt5 = srcTensor.GetPhyAddr() + fullExeSize * 5 * sizeof(T1);
        uint64_t gradLocalInt5 = gradTensor.GetPhyAddr() + fullExeSize * 5 * sizeof(T1);

        uint64_t srcLocalInt6 = srcTensor.GetPhyAddr() + fullExeSize * 6 * sizeof(T1);
        uint64_t gradLocalInt6 = gradTensor.GetPhyAddr() + fullExeSize * 6 * sizeof(T1);

        uint64_t srcLocalIntTail = srcTensor.GetPhyAddr() + fullExeSize * 7 * sizeof(T1);
        uint64_t gradLocalIntTail = gradTensor.GetPhyAddr() + fullExeSize * 7 * sizeof(T1);

        uint32_t tailSize = realN % fullExeSize;
        uint32_t reduceSize = tailSize == 0 ? fullExeSize : tailSize;

        __VEC_SCOPE__
        {
            RegTensor<float> vregSrc;
            RegTensor<float> vregGrad;
            RegTensor<float> vregSrc1;
            RegTensor<float> vregGrad1;
            RegTensor<float> vregSrc2;
            RegTensor<float> vregGrad2;
            RegTensor<float> vregSrc3;
            RegTensor<float> vregGrad3;
            RegTensor<float> vregSrc4;
            RegTensor<float> vregGrad4;
            RegTensor<float> vregSrc5;
            RegTensor<float> vregGrad5;
            RegTensor<float> vregSrc6;
            RegTensor<float> vregGrad6;
            RegTensor<float> vregMul;
            RegTensor<float> vregMul1;
            RegTensor<float> vregMul2;
            RegTensor<float> vregMul3;
            RegTensor<float> vregMul4;
            RegTensor<float> vregMul5;
            RegTensor<float> vregMul6;
            RegTensor<float> vregSrcTail;
            RegTensor<float> vregGradTail;
            RegTensor<float> vregMulTail;
            RegTensor<float> vregAdd;
            RegTensor<float> vregAdd1;
            RegTensor<float> vregAdd2;
            RegTensor<float> vregAdd3;
            RegTensor<float> vregAdd4;
            RegTensor<float> vregAdd5;
            RegTensor<float> vregReduceSum;
            MaskReg pregFullExe = CreateMask<float, MaskPattern::ALL>();
            UnalignReg uregReduceSum;
            MaskReg pregTailExe = UpdateMask<float>(reduceSize);

            for (uint16_t m = 0; m < static_cast<uint16_t>(srcM); m++) {
                // 手动unroll 128个数分64个数做mul和add
                DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrc, ((__ubuf__ T1 *&)srcLocalInt), HEAD_DIM_ALIGN);
                DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGrad, ((__ubuf__ T1 *&)gradLocalInt), HEAD_DIM_ALIGN);
                DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrc1, ((__ubuf__ T1 *&)srcLocalInt1), HEAD_DIM_ALIGN);
                DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGrad1, ((__ubuf__ T1 *&)gradLocalInt1), HEAD_DIM_ALIGN);
                DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrc2, ((__ubuf__ T1 *&)srcLocalInt2), HEAD_DIM_ALIGN);
                DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGrad2, ((__ubuf__ T1 *&)gradLocalInt2), HEAD_DIM_ALIGN);
                DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrc3, ((__ubuf__ T1 *&)srcLocalInt3), HEAD_DIM_ALIGN);
                DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGrad3, ((__ubuf__ T1 *&)gradLocalInt3), HEAD_DIM_ALIGN);
                DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrc4, ((__ubuf__ T1 *&)srcLocalInt4), HEAD_DIM_ALIGN);
                DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGrad4, ((__ubuf__ T1 *&)gradLocalInt4), HEAD_DIM_ALIGN);
                DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrc5, ((__ubuf__ T1 *&)srcLocalInt5), HEAD_DIM_ALIGN);
                DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGrad5, ((__ubuf__ T1 *&)gradLocalInt5), HEAD_DIM_ALIGN);
                DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrc6, ((__ubuf__ T1 *&)srcLocalInt6), HEAD_DIM_ALIGN);
                DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGrad6, ((__ubuf__ T1 *&)gradLocalInt6), HEAD_DIM_ALIGN);
                DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrcTail, ((__ubuf__ T1 *&)srcLocalIntTail), HEAD_DIM_ALIGN);
                DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGradTail, ((__ubuf__ T1 *&)gradLocalIntTail), HEAD_DIM_ALIGN);
                Mul(vregMul, vregGrad, vregSrc, pregFullExe);
                Mul(vregMul1, vregGrad1, vregSrc1, pregFullExe);
                Mul(vregMul2, vregGrad2, vregSrc2, pregFullExe);
                Mul(vregMul3, vregGrad3, vregSrc3, pregFullExe);
                Mul(vregMul4, vregGrad4, vregSrc4, pregFullExe);
                Mul(vregMul5, vregGrad5, vregSrc5, pregFullExe);
                Mul(vregMul6, vregGrad6, vregSrc6, pregFullExe);
                Mul(vregMulTail, vregGradTail, vregSrcTail, pregTailExe);
                Add(vregAdd, vregMul, vregMulTail, pregFullExe);
                Add(vregAdd1, vregMul1, vregMul6, pregFullExe);
                Add(vregAdd2, vregMul2, vregMul5, pregFullExe);
                Add(vregAdd3, vregMul3, vregMul4, pregFullExe);
                Add(vregAdd4, vregAdd, vregAdd3, pregFullExe);
                Add(vregAdd, vregAdd1, vregAdd2, pregFullExe);
                Add(vregAdd5, vregAdd, vregAdd4, pregFullExe);
                ReduceSum(vregReduceSum, vregAdd5, pregFullExe);
                DataCopyUnAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ float *&)dstLocalInt), vregReduceSum, uregReduceSum, 1);
            }
            vstas(uregReduceSum, ((__ubuf__ float *&)dstLocalInt), 0, POST_UPDATE);
        }
    }
}
#else
template <typename T1, typename T, uint32_t srcN, uint32_t HEAD_DIM_ALIGN>
__aicore__ inline void MySoftmaxGradFrontCastAligned512F32(const LocalTensor<T> &dstTensor, const LocalTensor<T1> &gradTensor,
                                                const LocalTensor<T1> &srcTensor, uint32_t srcM, uint32_t realN = srcN)
{
}
#endif
} // namespace AscendC

#endif // MY_SOFTMAX_GRAD_CAST_ALIGNED512_F32_INTERFACE_H_