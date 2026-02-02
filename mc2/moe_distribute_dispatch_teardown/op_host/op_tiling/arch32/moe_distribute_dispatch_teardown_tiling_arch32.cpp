/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file moe_distribute_dispatch_teardown_tiling_arch32.cpp
 * \brief
 */

#include "mc2_log.h"
#include "moe_distribute_dispatch_teardown_tiling_arch32.h"

namespace optiling {

ge::graphStatus MoeDistributeDispatchTeardownTilingA3::MoeDistributeDispatchTeardownTilingFuncImpl()
{
    OP_LOGD(nodeName_, "Start MoeDistributeDispatchTeardownA3 tiling");
    tilingData_ = context_->GetTilingData<MoeDistributeDispatchTeardownTilingData>();

    // 实现 A3 Tiling 拦截
    if (!((GetRequiredAttrAndSetTilingData() == ge::GRAPH_SUCCESS) &&
          (GetOptionalAttrAndSetTilingData() == ge::GRAPH_SUCCESS) && (CheckTensorShape() == ge::GRAPH_SUCCESS) &&
          (CheckTensorDataType() == ge::GRAPH_SUCCESS))) {
        return ge::GRAPH_FAILED;
    }
    if (CheckHcclBuffSize() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    SetHcommCfg();
    if (SetWorkSpace() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    SetTilingKey();
    SetPlatformInfo();
    PrintTilingDataInfo();
    OP_LOGD(nodeName_, "Finish MoeDistributeDispatchTeardownA3 tiling");
    return ge::GRAPH_SUCCESS;
}

bool MoeDistributeDispatchTeardownTilingA3::IsCapable()
{
    OP_LOGD(nodeName_, "Do MoeDistributeDispatchTeardownTilingA3 tiling.");
    return false;
}

ge::graphStatus MoeDistributeDispatchTeardownTilingA3::DoOpTiling()
{
    return MoeDistributeDispatchTeardownTilingFuncImpl();
}
} // namespace optiling