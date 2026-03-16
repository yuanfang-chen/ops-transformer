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
 * \file flash_attention_score_v4_infershape.cpp
 * \brief FlashAttentionScoreV4 InferShape 实现
 *
 * 参考来源：
 *   flash_attention_score/op_host/flash_attention_score_infershape.cpp
 *   — 非量化场景，移除 FP8 相关处理分支。
 *   fused_floyd_attention/op_host/fused_floyd_attention_infershape.cpp
 *   — softmax_max/softmax_sum 的输出 shape 设计参考了推理算子的处理方式。
 */

#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "log/log.h"

using namespace ge;

namespace ops {

/* ---- 属性索引（与 def.cpp 中的顺序一致） ---- */
static const uint64_t INDEX_HEAD_NUM     = 4;
static const uint64_t INDEX_LAYOUT       = 5;

/* ---- 维度常量 ---- */
static const uint64_t DIM_NUM_3 = 3;
static const uint64_t DIM_NUM_4 = 4;
static const uint64_t DIM_IDX_2 = 2;
static const uint64_t DIM_IDX_3 = 3;

/* softmax_max/softmax_sum 最后一维固定为 8（FlashAttention 标准） */
constexpr int FLA_SOFTMAXMAX_LAST_DIM = 8;

ge::graphStatus InferShapeFlashAttentionScoreV4(gert::InferShapeContext *context)
{
    OP_LOGI(context, "Enter FlashAttentionScoreV4 runtime infershape.");

    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const gert::Shape *queryShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, queryShape);
    const auto *queryDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, queryDesc);

    const gert::Shape *keyShape = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, keyShape);
    const gert::Shape *valueShape = context->GetInputShape(DIM_IDX_2);
    OP_CHECK_NULL_WITH_CONTEXT(context, valueShape);

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    auto headNum = attrs->GetInt(INDEX_HEAD_NUM);
    OP_CHECK_NULL_WITH_CONTEXT(context, headNum);
    const char *inputLayout = attrs->GetAttrPointer<char>(INDEX_LAYOUT);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputLayout);

    std::string inputLayoutStr = std::string(inputLayout);
    for (auto &c : inputLayoutStr) {
        c = toupper(c);
    }
    if (inputLayoutStr != "BSH" && inputLayoutStr != "BSND" &&
        inputLayoutStr != "SBH" && inputLayoutStr != "BNSD" &&
        inputLayoutStr != "TND") {
        OP_LOGE(context,
                "The inputLayout should be BSH/SBH/BSND/BNSD/TND (case-insensitive), but got %s.",
                inputLayoutStr.c_str());
        return GRAPH_FAILED;
    }

    /* ---- 解析 B、S（或 T） ---- */
    int64_t shapeB = 1;
    int64_t shapeS = 1;
    int64_t shapeT = 0;
    if (inputLayoutStr == "SBH") {
        shapeB = queryShape->GetDim(1);
        shapeS = queryShape->GetDim(0);
    } else if (inputLayoutStr == "TND") {
        shapeT = queryShape->GetDim(0);
    } else if (inputLayoutStr == "BSND" || inputLayoutStr == "BSH") {
        shapeB = queryShape->GetDim(0);
        shapeS = queryShape->GetDim(1);
    } else {
        /* BNSD */
        shapeB = queryShape->GetDim(0);
        shapeS = queryShape->GetDim(DIM_IDX_2);
    }

    OP_LOGI(context,
            "InferShape: B=%ld, N=%ld, T=%ld, S=%ld, layout=%s, dtype=%s",
            shapeB, *headNum, shapeT, shapeS, inputLayoutStr.c_str(),
            ge::TypeUtils::DataTypeToSerialString(queryDesc->GetDataType()).c_str());

    /* ---- 输出 0: softmax_max ---- */
    gert::Shape *softmaxMaxShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, softmaxMaxShape);

    /* ---- 输出 1: softmax_sum（shape 与 softmax_max 相同） ---- */
    gert::Shape *softmaxSumShape = context->GetOutputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, softmaxSumShape);

    if (inputLayoutStr == "TND") {
        /* TND 格式：softmaxMax shape = [T, N, 8] */
        softmaxMaxShape->SetDimNum(DIM_NUM_3);
        softmaxMaxShape->SetDim(0, shapeT);
        softmaxMaxShape->SetDim(1, *headNum);
        softmaxMaxShape->SetDim(DIM_IDX_2, FLA_SOFTMAXMAX_LAST_DIM);
    } else {
        /* 其他格式：softmaxMax shape = [B, N, S, 8] */
        softmaxMaxShape->SetDimNum(DIM_NUM_4);
        softmaxMaxShape->SetDim(0, shapeB);
        softmaxMaxShape->SetDim(1, *headNum);
        softmaxMaxShape->SetDim(DIM_IDX_2, shapeS);
        softmaxMaxShape->SetDim(DIM_IDX_3, FLA_SOFTMAXMAX_LAST_DIM);
    }
    *softmaxSumShape = *softmaxMaxShape;

    /* ---- 输出 2: softmax_out（保留接口，始终为空 tensor） ---- */
    gert::Shape *softmaxOutShape = context->GetOutputShape(DIM_IDX_2);
    OP_CHECK_NULL_WITH_CONTEXT(context, softmaxOutShape);
    softmaxOutShape->SetDimNum(DIM_NUM_4);
    softmaxOutShape->SetDim(0, 0);
    softmaxOutShape->SetDim(1, 0);
    softmaxOutShape->SetDim(DIM_IDX_2, 0);
    softmaxOutShape->SetDim(DIM_IDX_3, 0);

    /* ---- 输出 3: attention_out ---- */
    gert::Shape *attentionOutShape = context->GetOutputShape(DIM_IDX_3);
    OP_CHECK_NULL_WITH_CONTEXT(context, attentionOutShape);
    *attentionOutShape = *queryShape;

    if (inputLayoutStr == "BSND" || inputLayoutStr == "BNSD") {
        /* value 的 D 维（最后维）可与 query 不同 */
        auto shapeD2 = valueShape->GetDim(3);
        attentionOutShape->SetDim(3, shapeD2);
    } else if (inputLayoutStr == "BSH" || inputLayoutStr == "SBH") {
        auto N1 = *headNum;
        if (N1 == 0) {
            attentionOutShape->SetDim(DIM_IDX_2, 0);
            return GRAPH_SUCCESS;
        }
        auto h1 = queryShape->GetDim(DIM_IDX_2);
        auto D1 = h1 / N1;
        if (D1 == 0) {
            attentionOutShape->SetDim(DIM_IDX_2, 0);
            return GRAPH_SUCCESS;
        }
        auto h2 = keyShape->GetDim(DIM_IDX_2);
        auto N2 = h2 / D1;
        if (N2 == 0) {
            attentionOutShape->SetDim(DIM_IDX_2, N1 * D1);
            return GRAPH_SUCCESS;
        }
        auto h3 = valueShape->GetDim(DIM_IDX_2);
        auto D2 = h3 / N2;
        attentionOutShape->SetDim(DIM_IDX_2, N1 * D2);
    } else {
        /* TND：value 的 D 维 */
        auto shapeD2 = valueShape->GetDim(DIM_IDX_2);
        attentionOutShape->SetDim(DIM_IDX_2, shapeD2);
    }

    return GRAPH_SUCCESS;
}

ge::graphStatus InferDataTypeFlashAttentionScoreV4(gert::InferDataTypeContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto dtype = context->GetInputDataType(0);

    /* softmax_max (idx 0): 固定 FLOAT32 */
    context->SetOutputDataType(0, DT_FLOAT);
    /* softmax_sum (idx 1): 固定 FLOAT32 */
    context->SetOutputDataType(1, DT_FLOAT);
    /* softmax_out (idx 2): 与 query 类型一致（空 tensor，仅保留接口兼容性） */
    context->SetOutputDataType(DIM_IDX_2, dtype);
    /* attention_out (idx 3): 与 query 类型一致 */
    context->SetOutputDataType(DIM_IDX_3, dtype);

    return GRAPH_SUCCESS;
}

IMPL_OP(FlashAttentionScoreV4)
    .InferShape(InferShapeFlashAttentionScoreV4)
    .InferDataType(InferDataTypeFlashAttentionScoreV4);

} // namespace ops
