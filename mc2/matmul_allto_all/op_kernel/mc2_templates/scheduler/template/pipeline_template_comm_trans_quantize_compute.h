/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file pipeline_template_comm_trans_compute.h
 * \brief
 */

#ifndef MC2_PIPELINE_TEMPLATE_COMM_TRANS_QUANTIZE_COMPUTE_H
#define MC2_PIPELINE_TEMPLATE_COMM_TRANS_QUANTIZE_COMPUTE_H

#include "pipeline_context.h"

// 流水线模板
namespace MC2KernelTemplate {
// 通信转置计算模板
template <typename CommunicationType, typename TransposeType, typename QuantizeType, typename ComputationType,
          typename ContextType>
class MC2KernelPipelineCommTransQuantComputeTemplate {
public:
    __aicore__ inline MC2KernelPipelineCommTransQuantComputeTemplate(CommunicationType *commStage,
                                                                     TransposeType *transStage,
                                                                     QuantizeType *quantStage,
                                                                     ComputationType *computeStage)
        : commStage_(commStage), transStage_(transStage), quantStage_(quantStage), computeStage_(computeStage){};

    __aicore__ inline void Init();

    __aicore__ inline void GetContext(ContextType* context);

    __aicore__ inline void Process(uint32_t taskCnt);

    __aicore__ inline void End();

private:
    CommunicationType *commStage_;  // 通信节点
    TransposeType *transStage_;     // 转置计算的计算节点
    QuantizeType *quantStage_;      // 进行动态量化的节点
    ComputationType *computeStage_; // 矩阵乘的计算节点
    ContextType *context_;          // 相关上下文
};

template <typename CommunicationType, typename TransposeType, typename QuantizeType, typename ComputationType,
          typename ContextType>
__aicore__ inline void MC2KernelPipelineCommTransQuantComputeTemplate<CommunicationType, TransposeType, QuantizeType,
                                                                      ComputationType, ContextType>::Init()
{
    commStage_->Init();
    computeStage_->Init();
}

template <typename CommunicationType, typename TransposeType, typename QuantizeType, typename ComputationType,
          typename ContextType>
__aicore__ inline void
MC2KernelPipelineCommTransQuantComputeTemplate<CommunicationType, TransposeType, QuantizeType, ComputationType,
                                               ContextType>::GetContext(ContextType* context)
{
    context->communicationContext = commStage_->GetContextPtr();
    context->transposeContext = transStage_->GetContextPtr();
    context->quantizationContext = quantStage_->GetContextPtr();
    context->computationContext = computeStage_->GetContextPtr();
}

template <typename CommunicationType, typename TransposeType, typename QuantizeType, typename ComputationType,
          typename ContextType>
__aicore__ inline void
MC2KernelPipelineCommTransQuantComputeTemplate<CommunicationType, TransposeType, QuantizeType, ComputationType,
                                               ContextType>::Process(uint32_t taskCnt)
{
    commStage_->Prepare(taskCnt);
    uint32_t index;
    for (index = 0; index < taskCnt; index++) {
        if ASCEND_IS_AIV {
            commStage_->Process();
            AscendC::SyncAll<true>();

            transStage_->Process(index);
            AscendC::SyncAll<true>();

            quantStage_->Process(index);
        }
        AscendC::SyncAll<false>();
        computeStage_->Process(index);
    }
}

template <typename CommunicationType, typename TransposeType, typename QuantizeType, typename ComputationType,
          typename ContextType>
__aicore__ inline void MC2KernelPipelineCommTransQuantComputeTemplate<CommunicationType, TransposeType, QuantizeType,
                                                                      ComputationType, ContextType>::End()
{
    commStage_->End();
    computeStage_->End();
}
}; // namespace MC2KernelTemplate

#endif