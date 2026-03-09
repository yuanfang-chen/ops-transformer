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
 * \file causal_conv1d_fn_tiling.cpp
 * \brief Main tiling entry for CausalConv1dFn operator
 */

#include "log/log.h"
#include "register/op_impl_registry.h"
#include "tiling_base/tiling_util.h"
#include "tiling_base/tiling_templates_registry.h"

namespace optiling {

struct CausalConv1dFnCompileInfo {};

// Main tiling function that dispatches to architecture-specific implementations
static ge::graphStatus TilingCausalConv1dFn(gert::TilingContext* context)
{
    OP_LOGD(context->GetNodeName(), "CausalConv1dFnTiling tiling start");

    // Call the platform-specific tiling implementation via the registry
    // The actual tiling logic is in causal_conv1d_fn_tiling_arch35.cpp
    // which registers itself with REGISTER_OPS_TILING_TEMPLATE with priority 1
    std::vector<int32_t> tilingRegisterList = {1};
    return Ops::Transformer::OpTiling::TilingRegistry::GetInstance().DoTilingImpl(context, tilingRegisterList);
}

static ge::graphStatus TilingPrepareCausalConv1dFn(gert::TilingParseContext* context)
{
    OP_CHECK_IF(context == nullptr,
                OP_LOGE("CausalConv1dFn", "context is null"),
                return ge::GRAPH_FAILED);

    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_IF(platformInfo == nullptr,
                OP_LOGE(context->GetNodeName(), "platformInfo is null"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// Register the main tiling entry point
IMPL_OP_OPTILING(CausalConv1dFn)
    .Tiling(TilingCausalConv1dFn)
    .TilingParse<CausalConv1dFnCompileInfo>(TilingPrepareCausalConv1dFn);
} // namespace optiling
