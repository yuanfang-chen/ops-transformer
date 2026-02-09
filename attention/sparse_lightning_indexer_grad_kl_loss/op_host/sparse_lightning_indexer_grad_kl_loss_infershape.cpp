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
 * \file sparse_lightning_indexer_grad_kl_loss_infershape.cpp
 * \brief
 */

#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "err/ops_err.h"

using namespace ge;

namespace ops {
static const uint64_t INDEX_LAYOUT = 1;
static const uint64_t DIM_NUM_0 = 0;
static const uint64_t DIM_NUM_1 = 1;
static const uint64_t DIM_NUM_2 = 2;
static const uint64_t DIM_NUM_3 = 3;

enum InputIdx {
    queryEnum = 0,
    keyEnum,
    queryIndexEnum,
    keyIndexEnum,
    weightEnum,
    sparseIndexEnum,
    maxEnum,
    sumEnum,
    queryRopeEnum,
    keyRopeEnum,
    actualSeqLengthsQueryEnum,
    actualSeqLengthsKvEnum
};

enum OutputIdx {
    dQueryIndexEnum = 0,
    dKeyIndexEnum,
    dWeightEnum,
    lossEnum
};

ge::graphStatus InferShapeSparseLightningIndexerGradKLLoss(gert::InferShapeContext *context)
{
    OP_LOGI(context, "Start enter ShapeSparseLightningIndexerGradKLLoss runtime infershape impl.");
    if (context == nullptr){
        return ge::GRAPH_FAILED;
    }
    const gert::Shape *queryShape = context->GetInputShape(queryEnum);
    OP_CHECK_NULL_WITH_CONTEXT(context, queryShape);
    const gert::Shape *keyShape = context->GetInputShape(keyEnum);
    OP_CHECK_NULL_WITH_CONTEXT(context, keyShape);
    const gert::Shape *weightShape = context->GetInputShape(weightEnum);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightShape);
    const gert::Shape *queryIndexShape = context->GetInputShape(queryIndexEnum);
    OP_CHECK_NULL_WITH_CONTEXT(context, queryIndexShape);
    const gert::Shape *keyIndexShape = context->GetInputShape(keyIndexEnum);
    OP_CHECK_NULL_WITH_CONTEXT(context, keyIndexShape);

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

    const char *inputLayout = attrs->GetAttrPointer<char>(INDEX_LAYOUT);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputLayout);

    std::string inputLayoutSlig = std::string(inputLayout);
    for (auto &c : inputLayoutSlig) {
        c = toupper(c);
    }
    if (inputLayoutSlig != "BSND" && inputLayoutSlig != "TND") {
        OP_LOGE(context, "The SparseLightningIndexerGradKLLoss inputLayout should be BSND/TND, but got %s.",
                  inputLayoutSlig.c_str());
        return GRAPH_FAILED;
    }

    gert::Shape *lossShape = context->GetOutputShape(lossEnum);
    OP_CHECK_NULL_WITH_CONTEXT(context, lossShape);
    gert::Shape *dWeightShape = context->GetOutputShape(dWeightEnum);
    OP_CHECK_NULL_WITH_CONTEXT(context, dWeightShape);
    gert::Shape *dQIndexShape = context->GetOutputShape(dQueryIndexEnum);
    OP_CHECK_NULL_WITH_CONTEXT(context, dQIndexShape);
    gert::Shape *dKIndexShape = context->GetOutputShape(dKeyIndexEnum);
    OP_CHECK_NULL_WITH_CONTEXT(context, dKIndexShape);

    lossShape->SetDimNum(1); // 1代表设置维度为1
    lossShape->SetDim(DIM_NUM_0, 1); // 设置第0维数值为1
    *dWeightShape = *weightShape;
    *dQIndexShape = *queryIndexShape;
    *dKIndexShape = *keyIndexShape;

    if (lossShape->GetShapeSize() != 1){
        OP_LOGE(context, "The Shape Len of Loss should be 1, but got %ld.", lossShape->GetShapeSize());
        return GRAPH_FAILED;
    } else if (lossShape->GetDim(DIM_NUM_0) != 1) {
        OP_LOGE(context, "The Shape data of Loss should be 1, but got %ld.", lossShape->GetDim(DIM_NUM_0));
        return GRAPH_FAILED;     
    }

    // 比较维度数值与输入是否能够对应
    if (inputLayoutStr == "BSND") {
        if (dWeightShape->GetDim(DIM_NUM_0) != weightShape->GetDim(DIM_NUM_0) || dWeightShape->GetDim(DIM_NUM_1) != weightShape->GetDim(DIM_NUM_1)
            || dWeightShape->GetDim(DIM_NUM_2) != weightShape->GetDim(DIM_NUM_2)) {
                OP_LOGE(context, "The input weights shape is [%ld, %ld, %ld], but d_weights got [%ld, %ld, %ld].", weightShape->GetDim(DIM_NUM_0),
                weightShape->GetDim(DIM_NUM_1), weightShape->GetDim(DIM_NUM_2), dWeightShape->GetDim(DIM_NUM_0),
                dWeightShape->GetDim(DIM_NUM_1), dWeightShape->GetDim(DIM_NUM_2));
                return GRAPH_FAILED;
        }
        if (queryIndexShape->GetDim(DIM_NUM_0) != dQIndexShape->GetDim(DIM_NUM_0) || queryIndexShape->GetDim(DIM_NUM_1) != dQIndexShape->GetDim(DIM_NUM_1) || 
            queryIndexShape->GetDim(DIM_NUM_2) != dQIndexShape->GetDim(DIM_NUM_2) || queryIndexShape->GetDim(DIM_NUM_3) != dQIndexShape->GetDim(DIM_NUM_3)) {
                OP_LOGE(context, "The input query_index shape is [%ld, %ld, %ld, %ld], but d_query_index got [%ld, %ld, %ld, %ld].", queryIndexShape->GetDim(DIM_NUM_0),
                queryIndexShape->GetDim(DIM_NUM_1), queryIndexShape->GetDim(DIM_NUM_2), queryIndexShape->GetDim(DIM_NUM_3), 
                dQIndexShape->GetDim(DIM_NUM_0), dQIndexShape->GetDim(DIM_NUM_1), dQIndexShape->GetDim(DIM_NUM_2), dQIndexShape->GetDim(DIM_NUM_3));
                return GRAPH_FAILED;                
        }
        if (keyIndexShape->GetDim(DIM_NUM_0) != dKIndexShape->GetDim(DIM_NUM_0) || keyIndexShape->GetDim(DIM_NUM_1) != dKIndexShape->GetDim(DIM_NUM_1) || 
            keyIndexShape->GetDim(DIM_NUM_2) != dKIndexShape->GetDim(DIM_NUM_2) || keyIndexShape->GetDim(DIM_NUM_3) != dKIndexShape->GetDim(DIM_NUM_3)) {
                OP_LOGE(context, "The input key_index shape is [%ld, %ld, %ld, %ld], but d_key_index got [%ld, %ld, %ld, %ld].", keyIndexShape->GetDim(DIM_NUM_0),
                keyIndexShape->GetDim(DIM_NUM_1), keyIndexShape->GetDim(DIM_NUM_2), keyIndexShape->GetDim(DIM_NUM_3), 
                dKIndexShape->GetDim(DIM_NUM_0), dKIndexShape->GetDim(DIM_NUM_1), dKIndexShape->GetDim(DIM_NUM_2), dKIndexShape->GetDim(DIM_NUM_3));
                return GRAPH_FAILED;         
        }
    } else if (inputLayoutStr == "TND") {
        if (dWeightShape->GetDim(DIM_NUM_0) != weightShape->GetDim(DIM_NUM_0) || dWeightShape->GetDim(DIM_NUM_1) != weightShape->GetDim(DIM_NUM_1)) {
                OP_LOGE(context, "The input weights shape is [%ld, %ld], but d_weights got [%ld, %ld].", weightShape->GetDim(DIM_NUM_0),
                weightShape->GetDim(DIM_NUM_1), dWeightShape->GetDim(DIM_NUM_0), dWeightShape->GetDim(DIM_NUM_1));
                return GRAPH_FAILED;
        }
        if (queryIndexShape->GetDim(DIM_NUM_0) != dQIndexShape->GetDim(DIM_NUM_0) || queryIndexShape->GetDim(DIM_NUM_1) != dQIndexShape->GetDim(DIM_NUM_1) || 
            queryIndexShape->GetDim(DIM_NUM_2) != dQIndexShape->GetDim(DIM_NUM_2)) {
                OP_LOGE(context, "The input query_index shape is [%ld, %ld, %ld], but d_query_index got [%ld, %ld, %ld].", queryIndexShape->GetDim(DIM_NUM_0),
                queryIndexShape->GetDim(DIM_NUM_1), queryIndexShape->GetDim(DIM_NUM_2), 
                dQIndexShape->GetDim(DIM_NUM_0), dQIndexShape->GetDim(DIM_NUM_1), dQIndexShape->GetDim(DIM_NUM_2));
                return GRAPH_FAILED;                
        }
        if (keyIndexShape->GetDim(DIM_NUM_0) != dKIndexShape->GetDim(DIM_NUM_0) || keyIndexShape->GetDim(DIM_NUM_1) != dKIndexShape->GetDim(DIM_NUM_1) || 
            keyIndexShape->GetDim(DIM_NUM_2) != dKIndexShape->GetDim(DIM_NUM_2)) {
                OP_LOGE(context, "The input key_index shape is [%ld, %ld, %ld], but d_key_index got [%ld, %ld, %ld].", keyIndexShape->GetDim(DIM_NUM_0),
                keyIndexShape->GetDim(DIM_NUM_1), keyIndexShape->GetDim(DIM_NUM_2),  
                dKIndexShape->GetDim(DIM_NUM_0), dKIndexShape->GetDim(DIM_NUM_1), dKIndexShape->GetDim(DIM_NUM_2));
                return GRAPH_FAILED;         
        }
    }

    return GRAPH_SUCCESS;
}

ge::graphStatus InferDataTypeSparseLightningIndexerGradKLLoss(gert::InferDataTypeContext *context)
{
    OP_LOGI(context, "Start enter ShapeSparseLightningIndexerGradKLLoss runtime infershape impl.");
    if (context == nullptr){
        return ge::GRAPH_FAILED;
    } 
    const auto inputDataType = context->GetInputDataType(queryEnum);
    context->SetOutputDataType(dQueryIndexEnum, inputDataType);
    context->SetOutputDataType(dKeyIndexEnum, inputDataType);
    context->SetOutputDataType(dWeightEnum, inputDataType);
    context->SetOutputDataType(lossEnum, DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(SparseLightningIndexerGradKLLoss).InferShape(InferShapeSparseLightningIndexerGradKLLoss).InferDataType(InferDataTypeSparseLightningIndexerGradKLLoss);
} // namespace ops