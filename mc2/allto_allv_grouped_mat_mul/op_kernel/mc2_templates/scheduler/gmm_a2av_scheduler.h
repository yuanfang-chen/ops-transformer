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

using namespace AscendC;

namespace MC2KernelTemplate {
template <typename CommOpType, typename ComputationOpType, typename TilingDataType, typename GmmTilingDataType,
    typename GmmArrayAddrType>
class GmmA2avScheduler {
public:
    __aicore__ inline void Init(CommOpType hcclOp, ComputationOpType computeOp, ComputationOpType shareComputeOp, GM_ADDR tilingGM)
    {
        hcclOp_ = hcclOp;
        computeOp_ = computeOp;
        shareComputeOp_ = shareComputeOp;
        auto tiling = (__gm__ TilingDataType *)tilingGM;
        GET_TILING_DATA(tilingData_, tilingGM);
        __gm__ void *hcclInitTiling = (__gm__ void *)(&(tiling->hcclA2avTiling.hcclInitTiling));
        __gm__ void *alltoAllvCcTiling = (__gm__ void *)(&(tiling->hcclA2avTiling.alltoAllvCcTiling));
        expertNumInOneRank_ = tilingData_->taskTilingInfo.e;
    }

    __aicore__ inline void Process()
    {
        if ( shareComputeOp_ != nullptr){
            shareComputeOp_.ProcessExpert(0, 1);
        }
        for (uint32_t e = 0U; e < expertNumInOneRank_; e++) {
            computeOp_.ProcessExpert(e, 1); // 每次专家数量设置为1进行调试
            hcclOp_.Launch(e, 1);   // 每次专家数量设置为1进行调试
        }
        End();
    }

    __aicore__ inline void End()
    {
        hcclOp_.End();
        computeOp_.End();
        shareComputeOp_.End();
    }

private:
    CommOpType hcclOp_;
    ComputationOpType computeOp_;
    ComputationOpType shareComputeOp_;
    const TilingDataType *tilingData_;
    uint32_t expertNumInOneRank_ = 0U;
};
};
#endif