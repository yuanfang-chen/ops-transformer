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
 * \file moe_distribute_dispatch_teardown_tiling.cc
 * \brief
 */

#include "tiling/tiling_templates_registry.h"
#include "moe_distribute_dispatch_teardown_tiling_a3.h"

using namespace AscendC;

namespace optiling {

REGISTER_OPS_TILING_TEMPLATE(MoeDistributeDispatchTeardown, MoeDistributeDispatchTeardownTilingA3, 0);

ge::graphStatus MoeDistributeDispatchTeardownTilingFunc(gert::TilingContext* context)
{
    return TilingRegistry::GetInstance().DoTilingImpl(context);
}

ge::graphStatus TilingParseForMoeDistributeDispatchTeardown(gert::TilingParseContext* context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

struct MoeDistributeDispatchTeardownCompileInfo {
};
IMPL_OP_OPTILING(MoeDistributeDispatchTeardown)
    .Tiling(MoeDistributeDispatchTeardownTilingFunc)
    .TilingParse<MoeDistributeDispatchTeardownCompileInfo>(TilingParseForMoeDistributeDispatchTeardown);
} // namespace optiling