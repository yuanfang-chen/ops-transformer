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
 * \file pipeline_template_qgmm_allto_allv.h
 * \brief
 */

#ifndef PIPELINE_TEMPLATE_QGMMATAV_H
#define PIPELINE_TEMPLATE_QGMMATAV_H

#include "pipeline_context.h"

namespace ATAVKernelTemplate {
template <typename GmmExpertOpType, typename HcclOpType, typename ContextType, typename TilingDataType>
class GmmA2avScheduler {
public:
    __aicore__ inline GmmA2avScheduler(GmmExpertOpType* computeStage, TransposeType* transStage, HcclOpType* commStage) : gmmComputeOp_(computeStage), commStage_(commStage){};

    __aicore__ inline void Init(GM_ADDR gmmxGM, GM_ADDR gmmweightGM, GM_ADDR mmxOptionalGM,
        GM_ADDR mmweightOptionalGM,  GM_ADDR gmmxScaleGM, GM_ADDR gmmWeightScaleGM, GM_ADDR mmxScaleGM,
        GM_ADDR mmWeightScaleGM, GM_ADDR gmmyGM, GM_ADDR mmyOptionalGM, GM_ADDR workspaceGM, 
        TilingDataType tilingData, TPipe *pipe);

    __aicore__ inline void Process(uint32_t taskCnt);

    __aicore__ inline void End();

private:
    GmmExpertOpType* gmmComputeOp_; // 矩阵乘的计算节点
    GmmExpertOpType mmComputeOp_; // 共享专家
    HcclOpType* commStage_; // 通信节点
    TilingDataType tilingData_;
};

// 初始化各节点
template <typename GmmExpertOpType, typename HcclOpType, typename ContextType, typename TilingDataType>
__aicore__ inline void GmmA2avScheduler<GmmExpertOpType, HcclOpType, ContextType>::Init(GM_ADDR gmmxGM, GM_ADDR gmmweightGM, GM_ADDR mmxOptionalGM,
        GM_ADDR mmweightOptionalGM, GM_ADDR gmmxScaleGM, GM_ADDR gmmWeightScaleGM, GM_ADDR mmxScaleGM, GM_ADDR mmWeightScaleGM, GM_ADDR gmmyGM, 
        GM_ADDR mmyOptionalGM, GM_ADDR workspaceGM, TilingDataType tilingData, TPipe* tPipe)
{
    tilingData_ = tilingData;
    commOp.Init();
    if (tilingData_->commonTilingInfo.isNeedMM) {
        mmComputeOp_.Init(mmxOptionalGM, mmweightOptionalGM, mmxScaleGM, mmWeightScaleGM, mmyOptionalGM, tilingData_->mmQuantTilingData, tPipe);
    }
    gmmComputeOp_.Init(gmmxGM, gmmweightGM, gmmxScaleGM, gmmWeightScaleGM, gmmyGM, tilingData_->gmmQuantTilingData, tPipe);
}

//执行流水线
template <typename GmmExpertOpType, typename HcclOpType, typename ContextType, typename TilingDataType>
__aicore__ inline void GmmA2avScheduler<GmmExpertOpType, HcclOpType, ContextType>::Process(uint32_t taskCnt)
{ 
    if (tilingData_->commonTilingInfo.isNeedMM) {
        mmComputeOp_.Process();
    }
    commOp.Prepare();
    uint32_t index;
    for (index = 0 ; index < taskCnt; index++) {
        gmmComputeOp_->Process(index);
        //后续流水需要使用计算节点的结果
        if ASCEND_IS_AIV {
            commOp.Wait();
            commStage_->Process();
        }
        AscendC::SyncAll<false>();
    }
    End();
}

//释放流水线各节点资源
template <typename GmmExpertOpType, typename HcclOpType, typename ContextType, typename TilingDataType>
__aicore__ inline void GmmA2avScheduler<GmmExpertOpType, HcclOpType, ContextType>::End()
{
    gmmComputeOp_->End();
    mmComputeOp_->End();
    commStage_->End();
}
};

#endif