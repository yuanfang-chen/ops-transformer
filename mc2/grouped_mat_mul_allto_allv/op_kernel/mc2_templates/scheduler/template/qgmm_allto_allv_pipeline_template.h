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
 * \file qgmm_allto_allv_pipeline_template.h
 * \brief
 */

#ifndef QGMMATAV_PIPELINE_TEMPLATE_H
#define QGMMATAV_PIPELINE_TEMPLATE_H

#include "pipeline_context.h"

namespace ATAVKernelTemplate {
template <typename ComputationType, typename CommunicationType, typename ContextType>
class QGMMKernelPipelineTemplate {
public:
    __aicore__ inline QGMMKernelPipelineTemplate(ComputationType* computeStage, CommunicationType* commStage) : computeStage_(computeStage), commStage_(commStage){};

    __aicore__ inline void Init();

    __aicore__ inline void ChangeSpecification(void* updateContext);

    __aicore__ inline void Process(uint32_t taskCnt);

    __aicore__ inline void End();

private:
    ComputationType* computeStage_; // 矩阵乘的计算节点
    CommunicationType* commStage_; // 通信节点
    ContextType* context_; //相关上下文
};

// 初始化各节点
template <typename ComputationType, typename CommunicationType, typename ContextType>
__aicore__ inline void QGMMKernelPipelineTemplate<ComputationType, CommunicationType, ContextType>::Init()
{
    commStage_->Init();
}

// 变更各节点在流水线中的运行规格，主要是输入输出和每轮流水的偏移
template <typename ComputationType, typename CommunicationType, typename ContextType>
__aicore__ inline void QGMMKernelPipelineTemplate<ComputationType, CommunicationType, ContextType>::ChangeSpecification(void* updateContext)
{
    context_ = (ContextType*) updateContext;
    computeStage_->Update(context_->xGM, context_->weightGM, context_->yGM, context_->tilingData);
    commStage_->Update(context_->taskCnt, context_->sendBuffer, context_->recvBuffer, 
        context_->sendOffset, context_->recvOffset, context_->sendCount, context_->strideCount, context_->hcclDataType);
}

//执行流水线
template <typename ComputationType, typename CommunicationType, typename ContextType>
__aicore__ inline void QGMMKernelPipelineTemplate<ComputationType, CommunicationType, ContextType>::Process(uint32_t taskCnt)
{ 
    uint32_t index;
    for (index = 0 ; index < taskCnt; index++) {
        computeStage_->Process(index == 0);
        //后续流水需要使用计算节点的结果
        AscendC::SyncAll<false>();
        if ASCEND_IS_AIV {
            //后续通信需要使用转置后的结果
            AscendC::SyncAll<true>();
            commStage_->Process();
        }
        AscendC::SyncAll<false>();
    }
}

//释放流水线各节点资源
template <typename ComputationType, typename CommunicationType, typename ContextType>
__aicore__ inline void QGMMKernelPipelineTemplate<ComputationType, CommunicationType, ContextType>::End()
{
    computeStage_->End();
    commStage_->End();
}
};

#endif