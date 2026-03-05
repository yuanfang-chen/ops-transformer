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
 * \file fallback_add_rms_norm_dynamic_quant_all_gather_qbmm.cpp
 * \brief 动态shape图回调aclnn
*/
#include "fallback/fallback.h"
#include "op_mc2.h"
#include "mc2_log.h"
 
namespace fallback {

const char* AddRmsNormDynamicQuantAllGatherQbmmInfo = "AddRmsNormDynamicQuantAllGatherQbmmFallback";

static ge::graphStatus AddRmsNormDynamicQuantAllGatherQbmmExecuteFunc(gert::OpExecuteContext* host_api_ctx)
{
    OPS_LOG_D(AddRmsNormDynamicQuantAllGatherQbmmInfo, "Start to fallback for add_rms_norm_dynamic_quant_all_gather_qbmm.");
    OPS_ERR_IF(host_api_ctx == nullptr, OPS_LOG_E(AddRmsNormDynamicQuantAllGatherQbmmInfo, "host_api_ctx is null"), return ge::GRAPH_FAILED);

    // 校验tensor
    // 后面修改掉魔法数字
    const auto x1 = host_api_ctx->GetInputTensor(static_cast<size_t>(0));
    OPS_ERR_IF(x1 == nullptr, OPS_LOG_E(AddRmsNormDynamicQuantAllGatherQbmmInfo, "x1 is null"), return ge::GRAPH_FAILED);
    const auto x2 = host_api_ctx->GetInputTensor(static_cast<size_t>(1));
    OPS_ERR_IF(x2 == nullptr, OPS_LOG_E(AddRmsNormDynamicQuantAllGatherQbmmInfo, "x2 is null"), return ge::GRAPH_FAILED);
    const auto residual = host_api_ctx->GetInputTensor(static_cast<size_t>(2));
    OPS_ERR_IF(residual == nullptr, OPS_LOG_E(AddRmsNormDynamicQuantAllGatherQbmmInfo, "residual is null"), return ge::GRAPH_FAILED);
    const auto y = host_api_ctx->GetInputTensor(static_cast<size_t>(3));
    OPS_ERR_IF(y == nullptr, OPS_LOG_E(AddRmsNormDynamicQuantAllGatherQbmmInfo, "y is null"), return ge::GRAPH_FAILED);
    const auto gamma = host_api_ctx->GetInputTensor(static_cast<size_t>(4));
    OPS_ERR_IF(gamma == nullptr, OPS_LOG_E(AddRmsNormDynamicQuantAllGatherQbmmInfo, "gamma is null"), return ge::GRAPH_FAILED);
    const auto scale = host_api_ctx->GetInputTensor(static_cast<size_t>(5));
    OPS_ERR_IF(scale == nullptr, OPS_LOG_E(AddRmsNormDynamicQuantAllGatherQbmmInfo, "scale  is null"), return ge::GRAPH_FAILED);
    const auto smoothScale = host_api_ctx->GetInputTensor(static_cast<size_t>(6));
    OPS_ERR_IF(smoothScale == nullptr, OPS_LOG_E(AddRmsNormDynamicQuantAllGatherQbmmInfo, "smoothScale is null"), return ge::GRAPH_FAILED);
    const auto bias = host_api_ctx->GetInputTensor(static_cast<size_t>(7));
    OPS_ERR_IF(bias == nullptr, OPS_LOG_E(AddRmsNormDynamicQuantAllGatherQbmmInfo, "bias is null"), return ge::GRAPH_FAILED);
    const auto output = host_api_ctx->GetOutputTensor(static_cast<size_t>(0));
    OPS_ERR_IF(output == nullptr, OPS_LOG_E(AddRmsNormDynamicQuantAllGatherQbmmInfo, "output is null"), return ge::GRAPH_FAILED);
    const auto z = host_api_ctx->GetOutputTensor(static_cast<size_t>(1));
    OPS_ERR_IF(z == nullptr, OPS_LOG_E(AddRmsNormDynamicQuantAllGatherQbmmInfo, "output is null"), return ge::GRAPH_FAILED);

    // 校验attrs
    const auto attrs = host_api_ctx->GetAttrs();
    OPS_ERR_IF(attrs == nullptr, OPS_LOG_E(AddRmsNormDynamicQuantAllGatherQbmmInfo, "attrs is null"), return ge::GRAPH_FAILED);
    const char *group = attrs->GetStr(static_cast<size_t>(0));
    OPS_ERR_IF(group == nullptr, OPS_LOG_E(AddRmsNormDynamicQuantAllGatherQbmmInfo, "group is null"), return ge::GRAPH_FAILED);
    const bool *transposeX2 = attrs->GetBool(static_cast<size_t>(2));
    OPS_ERR_IF(transposeX2 == nullptr, OPS_LOG_E(AddRmsNormDynamicQuantAllGatherQbmmInfo, "reduce_op is null"), return ge::GRAPH_FAILED);
    const int64_t *residualNormMode = attrs->GetInt(static_cast<size_t>(4));
    OPS_ERR_IF(residualNormMode == nullptr, OPS_LOG_E(AddRmsNormDynamicQuantAllGatherQbmmInfo, "reduce_op is null"), return ge::GRAPH_FAILED);

    // 执行回调
    const auto ret = EXEC_OPAPI_CMD(aclnnAddRmsNormDynamicQuantAllGatherQbmm, x1, x2, residual, y, gamma, scale, smoothScale, bias,
        group, *transposeX2, *residualNormMode, output, z);
    OPS_ERR_IF(ret != ge::GRAPH_SUCCESS,
               OPS_LOG_E(AddRmsNormDynamicQuantAllGatherQbmmInfo, "Aclnn api error code %d", ret),
               return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(AddRmsNormDynamicQuantAllGatherQbmm).OpExecuteFunc(AddRmsNormDynamicQuantAllGatherQbmmExecuteFunc);

} // namespace fallback
