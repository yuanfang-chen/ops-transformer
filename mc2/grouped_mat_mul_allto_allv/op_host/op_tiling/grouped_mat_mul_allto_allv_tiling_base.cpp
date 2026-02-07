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
 * \file quant_grouped_mat_mul_allto_allv_tiling.cpp
 * \brief
 */

#include "op_mc2.h"
#include "mc2_log.h"
#include "grouped_mat_mul_allto_allv_tiling_base.h"

using namespace Mc2Log;
using namespace AscendC;
using namespace optiling;

namespace optiling {
// base check required
ge::graphStatus GmmAlltoAllvTilingBase::GetShapeAttrsInfo()
{
    opName_ = context_->GetNodeName();
    auto gmmXTensorDesc = context->GetInputDesc(GMM_X_INDEX);
    auto gmmWeightTensorDesc = context->GetInputDesc(GMM_WEIGHT_INDEX);
    auto gmmYDesc = context->GetOutputDesc(OUTPUT_GMM_Y_INDEX);
    OP_TILING_CHECK((gmmXTensorDesc == nullptr), OP_LOGE(opName_, "the input gmmX tensor is null."),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK((gmmWeightTensorDesc == nullptr), OP_LOGE(opName_, "the input gmmWeight tensor is null."),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK((gmmYDesc == nullptr), OP_LOGE(opName_, "the output gmmY tensor is null."),
                    return ge::GRAPH_FAILED);
    
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE(opName_, "Failed to get attrs."), return ge::GRAPH_FAILED);
    const char *group = attrs->GetAttrPointer<char>(ATTR_GROUP_INDEX);
    // 判断为空或者空字符串
    OP_TILING_CHECK(group == nullptr, OP_LOGE(opName, "The input attr group is null pointer."),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(group[0] == '\0', OP_LOGE(opName, "The input attr group is empty string."),
                    return ge::GRAPH_FAILED);
    // 判断group是否超过127
    size_t groupLen = strlen(group);
    OP_TILING_CHECK(groupLen > MAX_GROUP_NAME_LEN,
                    OP_LOGE(opName, "The input attr group length is %zu, which exceeds the limit of 128.", groupLen),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}
}
