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
 * \file moe_distribute_dispatch_teardown_tiling_base.h
 * \brief
 */

#ifndef MOE_DISTRIBUTE_DISPATCH_TEARDOWN_TILING_BASE_H_
#define MOE_DISTRIBUTE_DISPATCH_TEARDOWN_TILING_BASE_H_

#include "op_util.h"
#include "runtime/base/mc2/moe_tiling_base.h"
#include "tbe/impl/ascendc/moe_distribute_dispatch_teardown/moe_distribute_dispatch_teardown_tiling.h"

namespace optiling {
class MoeDistributeDispatchTeardownTilingBase : public MoeTilingBase
{
public:
    explicit MoeDistributeDispatchTeardownTilingBase(gert::TilingContext* context)
        : MoeTilingBase(context), nodeName_(context->GetNodeName()){};

protected:
    const char* socTilingName_;
    const char* nodeName_;
    MoeDistributeDispatchTeardownTilingData* tilingData_ = nullptr;
    std::string groupEp_;

    uint64_t GetTilingKey() const override;
};
} // namespace optiling
#endif