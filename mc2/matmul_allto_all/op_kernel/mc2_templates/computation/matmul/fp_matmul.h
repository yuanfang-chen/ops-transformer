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
#include "matmul_base.h"

namespace MC2KernelTemplate
{
//非量化场景没有额外的数据
struct FpMMAdditionalData {
};

//非量化场景的相关逻辑实现
template <typename MMType>
class MC2FpMMWrapper {
protected:
    MC2MMContext<FpMMAdditionalData, Mc2MatMulV3TilingData> MMcontext_;
    MMType MMImpl_;
    AscendC::TPipe* tPipePtr_;

public:
    __aicore__ inline MC2FpMMWrapper(AscendC::TPipe* tPipe) : tPipePtr_(tPipe) {};
    // 初始化方法
    __aicore__ inline void Init();
    // 获取数据上下文引用
    __aicore__ inline MC2MMContext<FpMMAdditionalData, Mc2MatMulV3TilingData>* GetMMContextPtr();
    // 执行一次计算的方法
    __aicore__ inline void Process(bool isFirst);
    // 结束方法
    __aicore__ inline void End();
};

template <typename MMType>
inline __aicore__ void MC2FpMMWrapper<MMType>::Init() {}

template <typename MMType>
inline __aicore__ MC2MMContext<FpMMAdditionalData, Mc2MatMulV3TilingData> *MC2FpMMWrapper<MMType>::GetMMContextPtr()
{
    return &MMcontext_;
}

template <typename MMType>
inline __aicore__ void MC2FpMMWrapper<MMType>::Process(bool isFirst)
{
    if ASCEND_IS_AIV {
        return;
    }
    if (!isFirst) {
        MMcontext_.baseData.aGM = MMcontext_.baseData.aGM + MMcontext_.baseData.aOffset;
        MMcontext_.baseData.bGM = MMcontext_.baseData.bGM + MMcontext_.baseData.bOffset;
        MMcontext_.baseData.cGM = MMcontext_.baseData.cGM + MMcontext_.baseData.cOffset;
    }
    tPipePtr_->Reset();
    MMImpl_.Init(MMcontext_.baseData.aGM, MMcontext_.baseData.bGM, MMcontext_.baseData.cGM,
        MMcontext_.baseData.biasGM, nullptr, nullptr, MMcontext_.tilingDataPtr, tPipePtr_);
    MMImpl_.Process();
}

template <typename MMType>
inline __aicore__ void MC2FpMMWrapper<MMType>::End()
{
    if ASCEND_IS_AIV {
        return;
    }
    MMImpl_.End();
}

// 计算节点的上下文数据类型声明
#ifndef DEFINE_MC2_MATMUL_CONTEXT_FOR_MATMUL_COMPUTATION_FP
#define DEFINE_MC2_MATMUL_CONTEXT_FOR_MATMUL_COMPUTATION_FP(ContextType) \
    using ContextType = MC2MMContext<FpMMAdditionalData, Mc2MatMulV3TilingData>
#endif

// 使用matmulv3算子作为计算节点的计算实现，是否转置的参数通过算子的模板参数获取
#ifndef DEFINE_MC2_MATMUL_FOR_MATMUL_COMPUTATION_FP
#define DEFINE_MC2_MATMUL_FOR_MATMUL_COMPUTATION_FP(ComputationType) \
    using ComputationType = MC2FpMMWrapper<\
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