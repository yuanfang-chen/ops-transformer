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
 * \file fused_floyd_attention_grad_tiling_common.cpp
 * \brief
 */

#include "fused_floyd_attention_grad_tiling_common.h"
#include "log/log.h"
#include "err/ops_err.h"

namespace optiling {

ge::graphStatus CheckSoftmaxMaxShape(gert::TilingContext *context, int64_t b, int64_t n, int64_t s1, int64_t s2)
{
    auto softmaxMaxShape = context->GetOptionalInputShape(SOFTMAX_MAX);
    if (softmaxMaxShape == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    auto softmaxMaxShapeDim = softmaxMaxShape->GetStorageShape().GetDimNum();
    if (softmaxMaxShapeDim != 5) { // softmaxMax only support 5 dimensions
        OP_LOGE(context, "The shape of softmaxMax is invalid, got %lu dimensions", softmaxMaxShapeDim);
        return ge::GRAPH_FAILED;
    }
    auto dim0 = softmaxMaxShape->GetStorageShape().GetDim(0); // 0:b
    auto dim1 = softmaxMaxShape->GetStorageShape().GetDim(1); // 1:n
    auto dim2 = softmaxMaxShape->GetStorageShape().GetDim(2); // 2:s1
    auto dim3 = softmaxMaxShape->GetStorageShape().GetDim(3); // 3:s2
    auto dim4 = softmaxMaxShape->GetStorageShape().GetDim(4); // 4:8

    // softmaxMax pad to 8
    OP_CHECK_IF((dim0 != b || dim1 != n || dim2 != s1 || dim3 != s2 || dim4 != 8),
              OP_LOGE(context, "The shape of softmaxMax is invalid, got (%ld,%ld,%ld,%ld,%ld), should be (%ld,%ld,%ld,%ld,8)",
                        dim0, dim1, dim2, dim3, dim4, b, n, s1, s2),
              return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckSoftmaxSumShape(gert::TilingContext *context, int64_t b, int64_t n, int64_t s1, int64_t s2)
{
    auto softmaxSumShape = context->GetOptionalInputShape(SOFTMAX_SUM);
    if (softmaxSumShape == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    auto softmaxSumShapeDim = softmaxSumShape->GetStorageShape().GetDimNum();
    if (softmaxSumShapeDim != 5) { // softmaxSum only support 5 dimensions
        OP_LOGE(context, "The shape of softmaxSum is invalid, got %lu dimensions", softmaxSumShapeDim);
        return ge::GRAPH_FAILED;
    }
    auto dim0 = softmaxSumShape->GetStorageShape().GetDim(0); // 0:b
    auto dim1 = softmaxSumShape->GetStorageShape().GetDim(1); // 1:n
    auto dim2 = softmaxSumShape->GetStorageShape().GetDim(2); // 2:s1
    auto dim3 = softmaxSumShape->GetStorageShape().GetDim(3); // 3:s2
    auto dim4 = softmaxSumShape->GetStorageShape().GetDim(4); // 4:8

    // softmaxSum pad to 8
    OP_CHECK_IF((dim0 != b || dim1 != n || dim2 != s1 || dim3 != s2 || dim4 != 8),
              OP_LOGE(context, "The shape of softmaxSum is invalid, got (%ld,%ld,%ld,%ld,%d), should be (%ld,%ld,%ld,%d,8)",
              dim0, dim1, dim2, dim3, dim4, b, n, s1, s2),
              return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckAttentionInShape1(gert::TilingContext *context)
{
    auto attentionInShape = context->GetOptionalInputShape(ATTENTION_IN);
    if (attentionInShape == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    auto queryShape = context->GetInputShape(QUERY);
    auto attentionInShapeDim = attentionInShape->GetStorageShape().GetDimNum();
    auto queryShapeDim = queryShape->GetStorageShape().GetDimNum();
    if (attentionInShapeDim != queryShapeDim) {
        OP_LOGE(context, "The dimnum of attentionIn %zu should be equal to query %zu", attentionInShapeDim,
                  queryShapeDim);
        return ge::GRAPH_FAILED;
    }
    for (size_t i = 0; i < queryShapeDim; i++) {
        if (attentionInShape->GetStorageShape().GetDim(i) != queryShape->GetStorageShape().GetDim(i)) {
            OP_LOGE(context, "The dim %zu of attentionIn shape is invalid, got %ld, should be %ld", i,
                      attentionInShape->GetStorageShape().GetDim(i), queryShape->GetStorageShape().GetDim(i));
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckSoftmaxDtype1(gert::TilingContext *context) {
    auto softmaxMax = context->GetOptionalInputDesc(SOFTMAX_MAX);
    auto softmaxSum = context->GetOptionalInputDesc(SOFTMAX_SUM);
    OP_CHECK_IF(softmaxMax == nullptr || softmaxSum == nullptr,
               OP_LOGE(context, "softmax_max or softmax_sum is nullptr."),
               return ge::GRAPH_FAILED);

    auto softmaxMaxType = static_cast<uint32_t>(softmaxMax->GetDataType());
    auto softmaxSumType = static_cast<uint32_t>(softmaxSum->GetDataType());

    bool softmaxTypeCheck = (softmaxMaxType == softmaxSumType) &&
                            (softmaxMaxType == ge::DT_FLOAT);
    OP_CHECK_IF(softmaxTypeCheck != true,
               OP_LOGE(context, "softmaxMaxType should be DT_FLOAT and same with softmaxSumType"),
               return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckAttentionInDtype1(gert::TilingContext *context) {
    auto query = context->GetInputDesc(QUERY);
    auto attentionIn = context->GetOptionalInputDesc(ATTENTION_IN);
    OP_CHECK_IF(query == nullptr || attentionIn == nullptr,
               OP_LOGE(context, "query or attentionIn is nullptr."),
               return ge::GRAPH_FAILED);

    auto queryType = static_cast<uint32_t>(query->GetDataType());
    auto attentionInType = static_cast<uint32_t>(attentionIn->GetDataType());

    OP_CHECK_IF(queryType != attentionInType,
               OP_LOGE(context, "invalid attentionIn dtype should be same with query's dtype"),
               return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckShapeValid1(gert::TilingContext *context, int64_t b, int64_t n, int64_t s1, int64_t s2, int64_t d)
{
    auto isShapeInValid = (b == 0 || n == 0 || s1 == 0 || s2 == 0 || d == 0);
    OP_CHECK_IF(isShapeInValid,
              OP_LOGE(context, "input shape error, got 0 in bns1s2s3d(%ld,%ld,%ld,%ld,%d)", b, n, s1, s2, d),
              return ge::GRAPH_FAILED);

    auto ret = CheckSoftmaxMaxShape(context, b, n, s1, s2);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    ret = CheckSoftmaxSumShape(context, b, n, s1, s2);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    ret = CheckAttentionInShape1(context);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckDtypeValid1(gert::TilingContext *context)
{
    if (context == nullptr) {
        OP_LOGE(context, "context is nullptr");
        return ge::GRAPH_FAILED;
    }

    auto ret = CheckSoftmaxDtype1(context);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    ret = CheckAttentionInDtype1(context);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    return ge::GRAPH_SUCCESS;
}

bool IsSameShape1(const gert::StorageShape *aShape, const gert::StorageShape *bShape) {
    OP_CHECK_IF((aShape == nullptr) || (bShape == nullptr),
               OP_LOGW("fused_floyd_attention_grad_tiling_common", "aShape or bShape is nullptr."),
               return false);
    uint32_t dimSizeA = aShape->GetStorageShape().GetDimNum();
    uint32_t dimSizeB = bShape->GetStorageShape().GetDimNum();
    if (dimSizeA != dimSizeB) {
        return false;
    }

    for (uint32_t i = 0; i < dimSizeA; i++) {
        auto dimA = aShape->GetStorageShape().GetDim(i);
        auto dimB = bShape->GetStorageShape().GetDim(i);
        if (dimA != dimB) {
            return false;
        }
    }
    return true;
}

} // namespace optiling
