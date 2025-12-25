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
 * \file moe_distribute_combine_setup_tiling.cpp
 * \brief
 */

#include "register/op_def_registry.h"
#include "tiling_base/tiling_templates_registry.h"
#include "moe_distribute_combine_setup_tiling_a3.h"

using namespace Ops::Transformer::OpTiling;
using namespace Mc2Tiling;
using namespace AscendC;
using namespace ge;

namespace optiling {

REGISTER_OPS_TILING_TEMPLATE(MoeDistributeCombineSetup, MoeDistributeCombineSetupTilingA3, 0);

ge::graphStatus MoeDistributeCombineSetupTilingFunc(gert::TilingContext* context)
{
    return TilingRegistry::GetInstance().DoTilingImpl(context);
}

ge::graphStatus TilingParseForMoeDistributeCombineSetup(gert::TilingParseContext* context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

struct MoeDistributeCombineSetupCompileInfo {
};
IMPL_OP_OPTILING(MoeDistributeCombineSetup)
    .Tiling(MoeDistributeCombineSetupTilingFunc)
    .TilingParse<MoeDistributeCombineSetupCompileInfo>(TilingParseForMoeDistributeCombineSetup);
} // namespace optiling