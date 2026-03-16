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
 * \file chunk_gated_delta_rule_recurrence_infershape.cpp
 * \brief
 */
#include <string>
#include "exe_graph/runtime/infer_shape_context.h"
#include "exe_graph/runtime/shape.h"
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "err/ops_err.h"

using namespace gert;

namespace ops {

// Input indices
const size_t IS_INIT_STATE  = 0;
const size_t IS_VALUE       = 2;

// Output indices
const size_t OS_STATE       = 0;
const size_t OS_ATTN_INTER  = 1;
const size_t OS_V_NEW       = 2;

const size_t STATE_NDIM = 4;
const size_t VALUE_NDIM = 4;

static ge::graphStatus InferShapeChunkGatedDeltaRuleRecurrence(InferShapeContext *context)
{
    if (context == nullptr) {
        OP_LOGE("ChunkGatedDeltaRuleRecurrence", "InferShape context is null");
        return ge::GRAPH_FAILED;
    }

    auto opName     = context->GetNodeName();
    auto shapeState = context->GetInputShape(IS_INIT_STATE);
    auto shapeValue = context->GetInputShape(IS_VALUE);
    auto outState   = context->GetOutputShape(OS_STATE);
    auto outAttn    = context->GetOutputShape(OS_ATTN_INTER);
    auto outVNew    = context->GetOutputShape(OS_V_NEW);

    if (shapeState == nullptr || shapeValue == nullptr ||
        outState == nullptr || outAttn == nullptr || outVNew == nullptr) {
        OP_LOGE(opName, "[InferShape] one or more shape pointers are null");
        return ge::GRAPH_FAILED;
    }

    // Output 0: initial_state (same shape as input initial_state [b, hv, dv, dk])
    outState->SetDimNum(STATE_NDIM);
    for (size_t i = 0; i < STATE_NDIM; i++) {
        outState->SetDim(i, shapeState->GetDim(i));
    }

    // Output 1 & 2: attn_inter_out and v_new_out (same shape as value [hv, n_chunks, cs, dv])
    outAttn->SetDimNum(VALUE_NDIM);
    outVNew->SetDimNum(VALUE_NDIM);
    for (size_t i = 0; i < VALUE_NDIM; i++) {
        outAttn->SetDim(i, shapeValue->GetDim(i));
        outVNew->SetDim(i, shapeValue->GetDim(i));
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeChunkGatedDeltaRuleRecurrence(
    gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(OS_STATE,      ge::DT_FLOAT);
    context->SetOutputDataType(OS_ATTN_INTER, ge::DT_FLOAT);
    context->SetOutputDataType(OS_V_NEW,      ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(ChunkGatedDeltaRuleRecurrence)
    .InferShape(InferShapeChunkGatedDeltaRuleRecurrence)
    .InferDataType(InferDataTypeChunkGatedDeltaRuleRecurrence);

} // namespace ops
