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
 * \file attention_worker_combine_graph_infer.cpp
 * \brief
 */

#include "log/log.h"
#include "register/op_impl_registry.h"

using namespace ge;

namespace ops {

constexpr size_t INPUT_IDX_LAYER_ID = 2;
constexpr size_t OUTPUT_IDX_Y = 0;
constexpr size_t OUTPUT_IDX_NEXT_LAYER_ID = 1;
constexpr size_t IDX_ONE = 1;
constexpr int64_t TOKEN_DTYPE_BF16 = 1;


graphStatus InferDtype4AttentionWorkerCombine(gert::InferDataTypeContext *context)
{
    OP_LOGD(context->GetNodeName(), "InferDtype4AttentionWorkerCombine enter");

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    auto tokenDtype = attrs->GetAttrPointer<int64_t>(IDX_ONE);
    OP_CHECK_NULL_WITH_CONTEXT(context, tokenDtype);
    if (*tokenDtype == TOKEN_DTYPE_BF16) {
        context->SetOutputDataType(OUTPUT_IDX_Y, ge::DT_BF16);
    } else {
        context->SetOutputDataType(OUTPUT_IDX_Y, ge::DT_FLOAT16);
    }

    auto layer_id_input_dtype = context->GetInputDataType(INPUT_IDX_LAYER_ID);
    context->SetOutputDataType(OUTPUT_IDX_NEXT_LAYER_ID, layer_id_input_dtype);

    OP_LOGD(context->GetNodeName(), "InferDtype4AttentionWorkerCombine end");
    return GRAPH_SUCCESS;
}

IMPL_OP(AttentionWorkerCombine).InferDataType(InferDtype4AttentionWorkerCombine);

} // namespace ops

