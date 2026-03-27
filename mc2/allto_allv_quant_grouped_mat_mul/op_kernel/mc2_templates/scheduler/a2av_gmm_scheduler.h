/* *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
  */

/* !
 * \file pipeline_template_comm_compute.h
 * \brief
 */

#ifndef MC2_PIPELINE_TEMPLATE_COMM_COMPUTE_H
#define MC2_PIPELINE_TEMPLATE_COMM_COMPUTE_H

#include "kernel_tiling/kernel_tiling.h"
#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

using namespace AscendC;

namespace MC2KernelTemplate {

__aicore__ inline uint64_t Align(uint64_t a, uint64_t b)
{
    if (b == 0) {
        return 0;
    }
    return ((a + b - 1) / b) * b;
}

template <typename CommOpType, typename ComputeOpType, typename LocalComputeOpType, typename TilingDataType,
          typename GmmTilingDataType, typename GmmArrayAddrType, bool IsNeedMM>
class A2avGmmScheduler {
public:
    __aicore__ inline void Init(GM_ADDR gmmxGM, GM_ADDR gmmweightGM, GM_ADDR mmxOptionalGM, GM_ADDR mmweightOptionalGM,
                                GM_ADDR gmmxScaleGM, GM_ADDR gmmWeightScaleGM, GM_ADDR mmxScaleGM,
                                GM_ADDR mmWeightScaleGM, GM_ADDR gmmyGM, GM_ADDR mmyOptionalGM,
                                GM_ADDR permuteOutOptionalGM, GM_ADDR workspaceGM, GM_ADDR tilingGM,
                                GmmArrayAddrType *gmmArrayAddrIn, GmmArrayAddrType *mmArrayAddrIn, TPipe *tPipe,
                                bool isA2avGmmFlag, bool isMxScene)
    {
        GET_TILING_DATA(tilingData, tilingGM);
        tilingData_ = &tilingData;
        e_ = tilingData_->taskTilingInfo.e;
        isMxSceneFlag_ = isMxScene;
        const void *hcclInitTiling = &(tilingData_->hcclA2avTilingInfo.hcclInitTiling);
        uint64_t hcclCcTilingOffset = offsetof(TilingDataType, hcclA2avTilingInfo) +
                                      offsetof(MC2KernelTemplate::HcclA2avTilingInfo, a2avCcTiling);
        commOutGm_ = tilingData_->isPermuteOut ? permuteOutOptionalGM : workspaceGM;
        commOp.Init(hcclInitTiling, hcclCcTilingOffset, &tilingData_->taskTilingInfo, gmmxGM, commOutGm_);

        // 增加scale的通信初始化,fp8通信数据类型长度是相同的，因此可以复用
        if (isMxSceneFlag_) {
            uint64_t commOutLen =
                Align((tilingData_->taskTilingInfo.A) * (tilingData_->taskTilingInfo.H1), TENSOR_LIST_SIZE);
            // permuteOut为true,则将permuteout存放到对应的位置，scale的地址可以从workspaceGM开始
            gmmxScaleCommOutGm_ = tilingData_->isPermuteOut ? workspaceGM : workspaceGM + commOutLen;
            gmmxScaleGm_ = gmmxScaleGM; // 保存参数到成员变量
            // commOp中已经初始化了上下文，这里只更新地址
            commOp.InitScaleBuffer(gmmxScaleGm_, gmmxScaleCommOutGm_);
        }
        if (IsNeedMM) {
            localComputeOp.Init(mmxOptionalGM, mmweightOptionalGM, mmxScaleGM, mmWeightScaleGM, mmyOptionalGM,
                                workspaceGM, tilingData_, &tilingData_->mmQuantTilingData, mmArrayAddrIn, tPipe,
                                isA2avGmmFlag);
        }
        computeScaleGm_ = (isMxSceneFlag_) ? gmmxScaleCommOutGm_ : gmmxScaleGm_;
        computeOp.Init(commOutGm_, gmmweightGM, computeScaleGm_, gmmWeightScaleGM, gmmyGM, workspaceGM, tilingData_,
                       &tilingData_->gmmQuantTilingData, gmmArrayAddrIn, tPipe, isA2avGmmFlag);
    }

    __aicore__ inline void Process()
    {
        if (IsNeedMM) {
            localComputeOp.Process(0);
            SyncAll<false>();
        }
        // mx场景下先启动全量scale通信
        if (isMxSceneFlag_) {
            commOp.LaunchScaleBeforeCompute(0, e_);
        }
        for (uint32_t expertIdx = 0U; expertIdx < e_; expertIdx++) {
            commOp.Launch(expertIdx, 1);
        }
        if (isMxSceneFlag_) {
            commOp.WaitScale(0);
        }
        for (uint32_t expertIdx = 0U; expertIdx < e_; expertIdx++) {
            commOp.Wait(expertIdx);
            SyncAll<false>();
            computeOp.Process(expertIdx);
        }
        this->End();
    }

protected:
    __aicore__ inline void End()
    {
        commOp.End();
        computeOp.End();
        localComputeOp.End();
    }

private:
    CommOpType commOp;
    ComputeOpType computeOp;
    LocalComputeOpType localComputeOp;
    GM_ADDR commOutGm_ = nullptr;
    GM_ADDR gmmxScaleGm_ = nullptr;
    GM_ADDR gmmxScaleCommOutGm_ = nullptr;
    GM_ADDR computeScaleGm_ = nullptr;
    const TilingDataType *tilingData_ = nullptr;
    uint32_t e_ = 0U;
    bool isMxSceneFlag_ = false;
};
}; // namespace MC2KernelTemplate
#endif