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
 * \file ffn_worker_batching_graph_infer.cpp
 * \brief
 */

#include "log/log.h"
#include "register/op_impl_registry.h"

using namespace ge;

namespace ops {

static constexpr int64_t TOKEN_DTYPE_ATTR = 2;

static constexpr int64_t Y_OUT = 0;
static constexpr int64_t GROUP_LIST_OUT = 1;
static constexpr int64_t SESSION_IDS_OUT = 2;
static constexpr int64_t MICRO_BATCH_IDS_OUT = 3;
static constexpr int64_t TOKEN_IDS_OUT = 4;
static constexpr int64_t EXPERT_OFFSETS_OUT = 5;
static constexpr int64_t DYNAMIC_SCALE_OUT = 6;
static constexpr int64_t ACTUAL_TOKEN_NUM_OUT = 7;

static constexpr int64_t TOKEN_KIND_ZERO = 0;
static constexpr int64_t TOKEN_KIND_ONE = 1;

static graphStatus InferDataType4FfnWorkerBatching(gert::InferDataTypeContext *context)
{
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t *tokenDtypePtr = attrs->GetAttrPointer<int64_t>(TOKEN_DTYPE_ATTR);
    int64_t tokenDtype = TOKEN_KIND_ZERO;
    if (tokenDtypePtr != nullptr) {
        tokenDtype = *tokenDtypePtr;
    }

    if (tokenDtype == TOKEN_KIND_ZERO) {
        context->SetOutputDataType(Y_OUT, ge::DT_FLOAT16);
    } else if (tokenDtype == TOKEN_KIND_ONE) {
        context->SetOutputDataType(Y_OUT, ge::DT_BF16);
    } else {
        context->SetOutputDataType(Y_OUT, ge::DT_INT8);
    }

    context->SetOutputDataType(GROUP_LIST_OUT, ge::DT_INT64);
    context->SetOutputDataType(SESSION_IDS_OUT, ge::DT_INT32);
    context->SetOutputDataType(MICRO_BATCH_IDS_OUT, ge::DT_INT32);
    context->SetOutputDataType(TOKEN_IDS_OUT, ge::DT_INT32);
    context->SetOutputDataType(EXPERT_OFFSETS_OUT, ge::DT_INT32);
    context->SetOutputDataType(DYNAMIC_SCALE_OUT, ge::DT_FLOAT);
    context->SetOutputDataType(ACTUAL_TOKEN_NUM_OUT, ge::DT_INT64);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(FfnWorkerBatching).InferDataType(InferDataType4FfnWorkerBatching);

} // namespace ops
