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
 * \file quant_matmul.h
 * \brief
 */

#ifndef MC2_QUANT_MATMUL_H
#define MC2_QUANT_MATMUL_H
#include "matmul_factory.h"

namespace MC2KernelTemplate {
struct KCQuantMMAdditionalData {
    GM_ADDR x1_scale;
    GM_ADDR x2_scale;
    GM_ADDR x2_offset;
    uint64_t x1_scale_offset;
};

//非量化场景的相关逻辑实现
template <typename MMType>
class KCQuantMMControl {
protected:
    MC2MMBaseGmAddrs* baseDataPtr_;
    KCQuantMMAdditionalData* additionalDataPtr_;
    MMType* MMImplPtr_;
    DequantBmm::Mc2QuantBatchMatmulV3TilingDataParams* tilingDataPtr_;
    AscendC::TPipe* tPipePtr_;

public:
    __aicore__ inline void  Init(MC2MMBaseGmAddrs* baseDataPtr, KCQuantMMAdditionalData* additionalDataPtr, DequantBmm::Mc2QuantBatchMatmulV3TilingDataParams* tilingDataPtr, MMType* MMImplPtr, AscendC::TPipe *tPipe);
    __aicore__ inline void  UpdateAdditionalData();
    __aicore__ inline void  InitMM();
    __aicore__ inline void  EndMM();
};

template <typename MMType>
__aicore__ inline void KCQuantMMControl<MMType>::Init(MC2MMBaseGmAddrs *baseDataPtr, KCQuantMMAdditionalData *additionalDataPtr,
                                                 DequantBmm::Mc2QuantBatchMatmulV3TilingDataParams *tilingDataPtr, MMType *MMImplPtr,
                                                 AscendC::TPipe *tPipe)
{
    baseDataPtr_ = baseDataPtr;
    additionalDataPtr_ = additionalDataPtr;
    tilingDataPtr_ = tilingDataPtr;
    MMImplPtr_ = MMImplPtr;
    tPipePtr_ = tPipe;
}

template <typename MMType>
__aicore__ inline void KCQuantMMControl<MMType>::UpdateAdditionalData()
{
    additionalDataPtr_->x1_scale = additionalDataPtr_->x1_scale + additionalDataPtr_->x1_scale_offset;
}

template <typename MMType>
__aicore__ inline void KCQuantMMControl<MMType>::InitMM()
{
    tPipePtr_->Reset();
    MMImplPtr_->Init(baseDataPtr_->aGM, baseDataPtr_->bGM, additionalDataPtr_->x2_scale, additionalDataPtr_->x2_offset, baseDataPtr_->biasGM, additionalDataPtr_->x1_scale, baseDataPtr_->cGM, nullptr, tilingDataPtr_, tPipePtr_);
}

template <typename MMType>
__aicore__ inline void KCQuantMMControl<MMType>::EndMM(){}

#ifndef DEFINE_AND_IMPL_MC2_MATMUL_FOR_MATMUL_COMPUTATION_QUANT
#define DEFINE_AND_IMPL_MC2_MATMUL_FOR_MATMUL_COMPUTATION_QUANT(ComputationType, MMDtypeX1, MMDtypeX2) \
    using ComputationType = MC2MMFactory<\
        MC2MMContext<KCQuantMMAdditionalData, DequantBmm::Mc2QuantBatchMatmulV3TilingDataParams>,\
        KCQuantMMControl,\
        Mc2QuantBatchMatmulV3::Mc2QuantBmmPertokenRegbaseKernel<MMDtypeX1, MMDtypeX2, float, float, float,\
            DTYPE_Y, CubeFormat::ND, CubeFormat::ND, CubeFormat::ND, false, X2TRANSPOSE, float, Mc2QuantBatchMatmulV3::Mc2QuantBmmAswBlock>\
        >
#endif
};
#endif

