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
 * \file fp_matmul.h 
 * \brief
 */

#ifndef MC2_FP_MATMUL_H
#define MC2_FP_MATMUL_H
#include "matmul_factory.h"

namespace MC2KernelTemplate
{
//非量化场景没有额外的数据
struct FpMMAdditionalData {
};

//非量化场景的相关逻辑实现
template <typename MMType>
class FpMMControl {
protected:
    MC2MMBaseGmAddrs* baseDataPtr_;
    FpMMAdditionalData* additionalDataPtr_;
    MMType* MMImplPtr_;
    Mc2MatMulV3TilingData* tilingDataPtr_;
    AscendC::TPipe* tPipePtr_;

public:
    __aicore__ inline void  Init(MC2MMBaseGmAddrs* baseDataPtr, FpMMAdditionalData* additionalDataPtr, Mc2MatMulV3TilingData* tilingDataPtr, MMType* MMImplPtr, AscendC::TPipe *tPipe);
    __aicore__ inline void  UpdateAdditionalData();
    __aicore__ inline void  InitMM();
    __aicore__ inline void  EndMM();
};

template <typename MMType>
__aicore__ inline void FpMMControl<MMType>::Init(MC2MMBaseGmAddrs *baseDataPtr, FpMMAdditionalData *additionalDataPtr,
                                                 Mc2MatMulV3TilingData *tilingDataPtr, MMType *MMImplPtr,
                                                 AscendC::TPipe *tPipe)
{
    if ASCEND_IS_AIV {
        return;
    }
    baseDataPtr_ = baseDataPtr;
    additionalDataPtr_ = additionalDataPtr;
    tilingDataPtr_ = tilingDataPtr;
    MMImplPtr_ = MMImplPtr;
    tPipePtr_ = tPipe;
}

template <typename MMType>
__aicore__ inline void FpMMControl<MMType>::UpdateAdditionalData()
{
    return;
}

template <typename MMType>
__aicore__ inline void FpMMControl<MMType>::InitMM()
{
    if ASCEND_IS_AIV {
        return;
    }
    tPipePtr_->Reset();
    MMImplPtr_->Init(baseDataPtr_->aGM, baseDataPtr_->bGM, baseDataPtr_->cGM, baseDataPtr_->biasGM, nullptr, nullptr, tilingDataPtr_, tPipePtr_);
}

template <typename MMType>
__aicore__ inline void FpMMControl<MMType>::EndMM()
{
    if ASCEND_IS_AIV {
        return;
    }
    MMImplPtr_->End();
}

// 使用matmulv3算子作为计算节点的计算实现,是否转置的参数通过算子的模板参数获取
#ifndef DEFINE_MC2_MATMUL_FOR_MATMUL_COMPUTATION_FP
#define DEFINE_MC2_MATMUL_FOR_MATMUL_COMPUTATION_FP(ComputationType) \
    using ComputationType = MC2MMFactory<\
        MC2MMContext<FpMMAdditionalData, Mc2MatMulV3TilingData>,\
        FpMMControl,\
        Mc2MatmulV3Advanced::Mc2MatmulAswKernel<\
            MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_X1, false>,\
            MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_X2, X2TRANSPOSE>,\
            MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_Y>,\
            MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DtypeBias>,\
            Mc2MatmulV3Advanced::Mc2MatmulAswBlock, MM_CFG_NO_PRELOAD>\
        >
#endif

}; // namespace MC2KernelTemplate
#endif