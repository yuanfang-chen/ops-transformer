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
 * \file sparse_flash_attention_grad_infershape.cpp
 * \brief
 */

#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "err/ops_err.h"

using namespace ge;

namespace ops {
namespace sfag {
static const uint64_t DIM_NUM_0 = 0;
static const uint64_t DIM_NUM_1 = 1;
static const uint64_t DIM_NUM_2 = 2;
static const uint64_t DIM_NUM_3 = 3;

enum class InputIndex : uint32_t {
    QUERY = 0,
    KEY,
    VALUE,
    TOPK_INDICES,
    ATTENTION_OUT_GRAD,
    ATTENTION_OUT,
    SOFTMAX_MAX,
    SOFTMAX_SUM,
    ACTUAL_SEQ_Q_LEN,
    ACTUAL_SEQ_KV_LEN,
    Q_ROPE,
    K_ROPE
};

enum class OutputIndex : uint32_t {
    DQ = 0,
    DK,
    DV,
    DQ_ROPE,
    DK_ROPE
};

enum class AttrIndex : uint32_t {
    SCALE_VALUE = 0,
    SELECTED_BLOCK_SIZE,
    INPUT_LAYOUT
};

ge::graphStatus InferShape4SparseFlashAttentionGrad(gert::InferShapeContext *context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("SparseFlashAttentionGrad", "InferShapeContext is nullptr"),
               return ge::GRAPH_FAILED);
    OP_LOGD(context, "Enter InferShape4SparseFlashAttentionGrad.");

    const gert::Shape *queryShape = context->GetInputShape(static_cast<size_t>(InputIndex::QUERY));
    const gert::Shape *keyShape = context->GetInputShape(static_cast<size_t>(InputIndex::KEY));
    const gert::Shape *valueShape = context->GetInputShape(static_cast<size_t>(InputIndex::VALUE));
    const gert::Shape *queryRopeShape = context->GetOptionalInputShape(static_cast<size_t>(InputIndex::Q_ROPE));
    const gert::Shape *keyRopeShape = context->GetOptionalInputShape(static_cast<size_t>(InputIndex::K_ROPE));
    OP_CHECK_NULL_WITH_CONTEXT(context, queryShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, keyShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, valueShape);

    auto attrs = context->GetAttrs();
    auto scaleValue = attrs->GetInt(static_cast<size_t>(AttrIndex::SCALE_VALUE));
    auto selectedBlockSize = attrs->GetInt(static_cast<size_t>(AttrIndex::SELECTED_BLOCK_SIZE));
    const char *inputLayout = attrs->GetAttrPointer<char>(static_cast<size_t>(AttrIndex::INPUT_LAYOUT));
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    OP_CHECK_NULL_WITH_CONTEXT(context, scaleValue);
    OP_CHECK_NULL_WITH_CONTEXT(context, selectedBlockSize);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputLayout);

    std::string inputLayoutSfag = std::string(inputLayout);
    for (auto &c : inputLayoutSfag) {
        c = toupper(c);
    }
    if (inputLayoutSfag != "BSND" && inputLayoutSfag != "TND") {
        OP_LOGE(context, "The SparseFlashAttentionGrad inputLayout should be BSND/TND, but got %s.",
                  inputLayoutSfag.c_str());
        return GRAPH_FAILED;
    }

    gert::Shape *dqShape = context->GetOutputShape(static_cast<size_t>(OutputIndex::DQ));
    gert::Shape *dkShape = context->GetOutputShape(static_cast<size_t>(OutputIndex::DK));
    gert::Shape *dvShape = context->GetOutputShape(static_cast<size_t>(OutputIndex::DV));
    OP_CHECK_NULL_WITH_CONTEXT(context, dqShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, dkShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, dvShape);
    *dqShape = *queryShape;
    *dkShape = *keyShape;
    *dvShape = *valueShape;

    if (queryRopeShape != nullptr) {
        gert::Shape *dqRopeShape = context->GetOutputShape(static_cast<size_t>(OutputIndex::DQ_ROPE));
        OP_CHECK_NULL_WITH_CONTEXT(context, dqRopeShape);
        *dqRopeShape = *queryRopeShape;
    }
    if (keyRopeShape != nullptr) {
        gert::Shape *dkRopeShape = context->GetOutputShape(static_cast<size_t>(OutputIndex::DK_ROPE));
        OP_CHECK_NULL_WITH_CONTEXT(context, dkRopeShape);
        *dkRopeShape = *keyRopeShape;
    }

    // 比较维度数值与输入是否能够对应
    if (inputLayoutStr == "BSND") {
        if (queryShape->GetDim(DIM_NUM_0) != dqShape->GetDim(DIM_NUM_0) || queryShape->GetDim(DIM_NUM_1) != dqShape->GetDim(DIM_NUM_1) || 
            queryShape->GetDim(DIM_NUM_2) != dqShape->GetDim(DIM_NUM_2) || queryShape->GetDim(DIM_NUM_3) != dqShape->GetDim(DIM_NUM_3)) {
                OP_LOGE(context, "The input query shape is [%ld, %ld, %ld, %ld], but d_query got [%ld, %ld, %ld, %ld].", queryShape->GetDim(DIM_NUM_0),
                queryShape->GetDim(DIM_NUM_1), queryShape->GetDim(DIM_NUM_2), queryShape->GetDim(DIM_NUM_3), 
                dqShape->GetDim(DIM_NUM_0), dqShape->GetDim(DIM_NUM_1), dqShape->GetDim(DIM_NUM_2), dqShape->GetDim(DIM_NUM_3));
                return GRAPH_FAILED;                
        }
        if (keyShape->GetDim(DIM_NUM_0) != dkShape->GetDim(DIM_NUM_0) || keyShape->GetDim(DIM_NUM_1) != dkShape->GetDim(DIM_NUM_1) || 
            keyShape->GetDim(DIM_NUM_2) != dkShape->GetDim(DIM_NUM_2) || keyShape->GetDim(DIM_NUM_3) != dkShape->GetDim(DIM_NUM_3)) {
                OP_LOGE(context, "The input key shape is [%ld, %ld, %ld, %ld], but d_key got [%ld, %ld, %ld, %ld].", keyShape->GetDim(DIM_NUM_0),
                keyShape->GetDim(DIM_NUM_1), keyShape->GetDim(DIM_NUM_2), keyShape->GetDim(DIM_NUM_3), 
                dkShape->GetDim(DIM_NUM_0), dkShape->GetDim(DIM_NUM_1), dkShape->GetDim(DIM_NUM_2), dkShape->GetDim(DIM_NUM_3));
                return GRAPH_FAILED;                
        }
        if (valueShape->GetDim(DIM_NUM_0) != dvShape->GetDim(DIM_NUM_0) || valueShape->GetDim(DIM_NUM_1) != dvShape->GetDim(DIM_NUM_1) || 
            valueShape->GetDim(DIM_NUM_2) != dvShape->GetDim(DIM_NUM_2) || valueShape->GetDim(DIM_NUM_3) != dvShape->GetDim(DIM_NUM_3)) {
                OP_LOGE(context, "The input value shape is [%ld, %ld, %ld, %ld], but d_value got [%ld, %ld, %ld, %ld].", valueShape->GetDim(DIM_NUM_0),
                valueShape->GetDim(DIM_NUM_1), valueShape->GetDim(DIM_NUM_2), valueShape->GetDim(DIM_NUM_3), 
                dvShape->GetDim(DIM_NUM_0), dvShape->GetDim(DIM_NUM_1), dvShape->GetDim(DIM_NUM_2), dvShape->GetDim(DIM_NUM_3));
                return GRAPH_FAILED;         
        }
        if (queryRopeShape->GetDim(DIM_NUM_0) != dqRopeShape->GetDim(DIM_NUM_0) || queryRopeShape->GetDim(DIM_NUM_1) != dqRopeShape->GetDim(DIM_NUM_1) || 
            queryRopeShape->GetDim(DIM_NUM_2) != dqRopeShape->GetDim(DIM_NUM_2) || queryRopeShape->GetDim(DIM_NUM_3) != dqRopeShape->GetDim(DIM_NUM_3)) {
                OP_LOGE(context, "The input query_rope shape is [%ld, %ld, %ld, %ld], but d_query_rope got [%ld, %ld, %ld, %ld].", queryRopeShape->GetDim(DIM_NUM_0),
                queryRopeShape->GetDim(DIM_NUM_1), queryRopeShape->GetDim(DIM_NUM_2), queryRopeShape->GetDim(DIM_NUM_3), 
                dqRopeShape->GetDim(DIM_NUM_0), dqRopeShape->GetDim(DIM_NUM_1), dqRopeShape->GetDim(DIM_NUM_2), dqRopeShape->GetDim(DIM_NUM_3));
                return GRAPH_FAILED;         
        }
        if (keyRopeShape->GetDim(DIM_NUM_0) != dkRopeShape->GetDim(DIM_NUM_0) || keyRopeShape->GetDim(DIM_NUM_1) != dkRopeShape->GetDim(DIM_NUM_1) || 
            keyRopeShape->GetDim(DIM_NUM_2) != dkRopeShape->GetDim(DIM_NUM_2) || keyRopeShape->GetDim(DIM_NUM_3) != dkRopeShape->GetDim(DIM_NUM_3)) {
                OP_LOGE(context, "The input key_rope shape is [%ld, %ld, %ld, %ld], but d_key_rope got [%ld, %ld, %ld, %ld].", keyRopeShape->GetDim(DIM_NUM_0),
                keyRopeShape->GetDim(DIM_NUM_1), keyRopeShape->GetDim(DIM_NUM_2), keyRopeShape->GetDim(DIM_NUM_3), 
                dkRopeShape->GetDim(DIM_NUM_0), dkRopeShape->GetDim(DIM_NUM_1), dkRopeShape->GetDim(DIM_NUM_2), dkRopeShape->GetDim(DIM_NUM_3));
                return GRAPH_FAILED;    
        }
    } else if (inputLayoutStr == "TND") {
        if (queryShape->GetDim(DIM_NUM_0) != dqShape->GetDim(DIM_NUM_0) || queryShape->GetDim(DIM_NUM_1) != dqShape->GetDim(DIM_NUM_1) || 
            queryShape->GetDim(DIM_NUM_2) != dqShape->GetDim(DIM_NUM_2)) {
                OP_LOGE(context, "The input query shape is [%ld, %ld, %ld], but d_query got [%ld, %ld, %ld].", queryShape->GetDim(DIM_NUM_0),
                queryShape->GetDim(DIM_NUM_1), queryShape->GetDim(DIM_NUM_2), 
                dqShape->GetDim(DIM_NUM_0), dqShape->GetDim(DIM_NUM_1), dqShape->GetDim(DIM_NUM_2));
                return GRAPH_FAILED;                
        }
        if (keyShape->GetDim(DIM_NUM_0) != dkShape->GetDim(DIM_NUM_0) || keyShape->GetDim(DIM_NUM_1) != dkShape->GetDim(DIM_NUM_1) || 
            keyShape->GetDim(DIM_NUM_2) != dkShape->GetDim(DIM_NUM_2)) {
                OP_LOGE(context, "The input key shape is [%ld, %ld, %ld], but d_key got [%ld, %ld, %ld].", keyShape->GetDim(DIM_NUM_0),
                keyShape->GetDim(DIM_NUM_1), keyShape->GetDim(DIM_NUM_2), 
                dkShape->GetDim(DIM_NUM_0), dkShape->GetDim(DIM_NUM_1), dkShape->GetDim(DIM_NUM_2));
                return GRAPH_FAILED;            
        }
        if (valueShape->GetDim(DIM_NUM_0) != dvShape->GetDim(DIM_NUM_0) || valueShape->GetDim(DIM_NUM_1) != dvShape->GetDim(DIM_NUM_1) || 
            valueShape->GetDim(DIM_NUM_2) != dvShape->GetDim(DIM_NUM_2)) {
                OP_LOGE(context, "The input value shape is [%ld, %ld, %ld], but d_value got [%ld, %ld, %ld].", valueShape->GetDim(DIM_NUM_0),
                valueShape->GetDim(DIM_NUM_1), valueShape->GetDim(DIM_NUM_2), 
                dvShape->GetDim(DIM_NUM_0), dvShape->GetDim(DIM_NUM_1), dvShape->GetDim(DIM_NUM_2));
                return GRAPH_FAILED;         
        }
        if (queryRopeShape->GetDim(DIM_NUM_0) != dqRopeShape->GetDim(DIM_NUM_0) || queryRopeShape->GetDim(DIM_NUM_1) != dqRopeShape->GetDim(DIM_NUM_1) || 
            queryRopeShape->GetDim(DIM_NUM_2) != dqRopeShape->GetDim(DIM_NUM_2)) {
                OP_LOGE(context, "The input query_rope shape is [%ld, %ld, %ld], but d_query_rope got [%ld, %ld, %ld].", queryRopeShape->GetDim(DIM_NUM_0),
                queryRopeShape->GetDim(DIM_NUM_1), queryRopeShape->GetDim(DIM_NUM_2), 
                dqRopeShape->GetDim(DIM_NUM_0), dqRopeShape->GetDim(DIM_NUM_1), dqRopeShape->GetDim(DIM_NUM_2));
                return GRAPH_FAILED;         
        }
        if (keyRopeShape->GetDim(DIM_NUM_0) != dkRopeShape->GetDim(DIM_NUM_0) || keyRopeShape->GetDim(DIM_NUM_1) != dkRopeShape->GetDim(DIM_NUM_1) || 
            keyRopeShape->GetDim(DIM_NUM_2) != dkRopeShape->GetDim(DIM_NUM_2)) {
                OP_LOGE(context, "The input key_rope shape is [%ld, %ld, %ld], but d_key_rope got [%ld, %ld, %ld].", keyRopeShape->GetDim(DIM_NUM_0),
                keyRopeShape->GetDim(DIM_NUM_1), keyRopeShape->GetDim(DIM_NUM_2),  
                dkRopeShape->GetDim(DIM_NUM_0), dkRopeShape->GetDim(DIM_NUM_1), dkRopeShape->GetDim(DIM_NUM_2));
                return GRAPH_FAILED;   
        }
    }

    return GRAPH_SUCCESS;
}

ge::graphStatus InferDataType4SparseFlashAttentionGrad(gert::InferDataTypeContext *context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("SparseFlashAttentionGrad", "InferDataTypeContext is nullptr"),
               return ge::GRAPH_FAILED);
    OP_LOGD(context, "Enter InferDataType4SparseFlashAttentionGrad.");

    auto dtype = context->GetInputDataType(static_cast<size_t>(InputIndex::QUERY));
    context->SetOutputDataType(static_cast<size_t>(OutputIndex::DQ), dtype);
    context->SetOutputDataType(static_cast<size_t>(OutputIndex::DK), dtype);
    context->SetOutputDataType(static_cast<size_t>(OutputIndex::DV), dtype);
    context->SetOutputDataType(static_cast<size_t>(OutputIndex::DQ_ROPE), dtype);
    context->SetOutputDataType(static_cast<size_t>(OutputIndex::DK_ROPE), dtype);

    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(SparseFlashAttentionGrad)
    .InferShape(InferShape4SparseFlashAttentionGrad)
    .InferDataType(InferDataType4SparseFlashAttentionGrad);
} // namespace sfag
} // namespace ops
