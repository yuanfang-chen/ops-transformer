/* *
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
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
#include "basic_api/kernel_basic_intf.h"
#include "scheduler_common.h"

using namespace AscendC;

namespace MC2KernelTemplate {
template <typename CommOpType, typename ComputationOpType, typename ContextType, typename TilingDataType>
class A2avGmmScheduler {
public:
    __aicore__ inline void Init(GM_ADDR gmmxGM, GM_ADDR gmmweightGM, GM_ADDR mmxOptionalGM, GM_ADDR mmweightOptionalGM,
        GM_ADDR gmmxScaleGM, GM_ADDR gmmWeightScaleGM, GM_ADDR mmxScaleGM, GM_ADDR mmWeightScaleGM, GM_ADDR gmmyGM,
        GM_ADDR mmyOptionalGM, GM_ADDR permuteOutOptionalGM, GM_ADDR workspaceGM, const TilingDataType* tilingData,
        TPipe *pipe);

    __aicore__ inline void Process();

    __aicore__ inline void End();

private:
    CommOpType commOp;
    ComputationOpType computeOp;
    ComputationOpType localComputeOp;
    TilingDataType tilingData_;
    uint32_t expertNumInOneRank_ = 0U;
};

template <typename CommOpType, typename ComputationOpType, typename ContextType, typename TilingDataType>
__aicore__ inline void A2avGmmScheduler<CommOpType, ComputationOpType, ContextType, TilingDataType>::Init(
    GM_ADDR gmmxGM, GM_ADDR gmmweightGM, GM_ADDR mmxOptionalGM, GM_ADDR mmweightOptionalGM, GM_ADDR gmmxScaleGM,
    GM_ADDR gmmWeightScaleGM, GM_ADDR mmxScaleGM, GM_ADDR mmWeightScaleGM, GM_ADDR gmmyGM, GM_ADDR mmyOptionalGM,
    GM_ADDR permuteOutOptionalGM, GM_ADDR workspaceGM, const TilingDataType* tilingData, TPipe *tPipe)
{
    tilingData_ = tilingData;
    expertNumInOneRank_ = tilingData_->commonTilingInfo.E_ep;
    commOp.Init();
    if (tilingData_->commonTilingInfo.isNeedMM) {
        localComputeOp.Init(mmxOptionalGM, mmweightOptionalGM, mmxScaleGM, mmWeightScaleGM, mmyOptionalGM,
            tilingData_->mmQuantTilingData, tPipe);
    }
    computeOp.Init(gmmxGM, gmmweightGM, gmmxScaleGM, gmmWeightScaleGM, gmmyGM, tilingData_->gmmQuantTilingData, tPipe);
}

template <typename CommOpType, typename ComputationOpType, typename ContextType, typename TilingDataType>
__aicore__ inline void A2avGmmScheduler<CommOpType, ComputationOpType, ContextType, TilingDataType>::Process()
{
    if (tilingData_->commonTilingInfo.isNeedMM) {
        localComputeOp.Process();
    }
    commOp.Prepare();
    for (uint32_t e = 0U; e < expertNumInOneRank_; e++) {
        commOp.Wait();
        computeOp.Process();
    }
    End();
}

template <typename CommOpType, typename ComputationOpType, typename ContextType, typename TilingDataType>
__aicore__ inline void A2avGmmScheduler<CommOpType, ComputationOpType, ContextType, TilingDataType>::End()
{
    commOp.End();
    computeOp.End();
    localComputeOp.End();
}
};
#endif