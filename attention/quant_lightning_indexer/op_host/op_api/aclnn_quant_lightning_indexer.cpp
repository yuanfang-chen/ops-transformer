/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <algorithm>
#include <unordered_map>
#include "quant_lightning_indexer.h"
#include "aclnn_quant_lightning_indexer.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/pad.h"
#include "aclnn_kernels/reshape.h"
#include "aclnn_kernels/slice.h"
#include "aclnn_kernels/transpose.h"
#include "opdev/common_types.h"
#include "opdev/fast_vector.h"
#include "opdev/op_errno.h"
#include "opdev/op_executor.h"
#include "opdev/tensor_view_utils.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {

static bool quantLightningIndexerCheckDataType(
    const aclTensor *query,
    const aclTensor *key,
    const aclTensor *weights,
    const aclTensor *queryDequantScale,
    const aclTensor *keyDequantScale,
    const aclTensor *blockTableOptional,
    const aclTensor *out)
{
    auto qDtype = query->GetDataType();
    auto kDtype = key->GetDataType();
    auto wtDtype = weights->GetDataType();
    auto qScaleDtype = queryDequantScale->GetDataType();
    auto kScaleDtype = keyDequantScale->GetDataType();
    auto blockTableDtype = blockTableOptional->GetDataType();
    auto outDtype = out->GetDataType();

    static const std::unordered_map<DataType, std::vector<DataType>> dTypeMappingQtoK = {
        {DataType::DT_FLOAT8_E4M3FN, {DataType::DT_FLOAT8_E4M3FN}},
        {DataType::DT_HIFLOAT8, {DataType::DT_HIFLOAT8}},
    };
    if (dTypeMappingQtoK.find(qDtype) == dTypeMappingQtoK.end()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Input query data type is invalid, please check.");
        return false;
    } else {
        auto validKDtypeList = dTypeMappingQtoK.at(qDtype);
        if (std::find(validKDtypeList.begin(), validKDtypeList.end(), kDtype) == validKDtypeList.end()) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Input key data type is invalid, please check.");
            return false;
        }
    }
    if (wtDtype != DataType::DT_BF16) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "weights data type must be bf16, please check.");
        return false;
    }
    if (qScaleDtype != DataType::DT_FLOAT) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "queryDequantScale data type must be fp32, please check.");
        return false;
    }
    if (kScaleDtype != DataType::DT_FLOAT) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "keyDequantScale data type must be fp32, please check.");
        return false;
    }
    if (blockTableDtype != DataType::DT_INT32) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "blockTable data type must be int32, please check.");
        return false;
    }
    if (outDtype != DataType::DT_INT32) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "out data type must be int32, please check.");
        return false;
    }
    return true;
}

aclnnStatus quantLightningIndexerCheckTensorNull(
    const aclTensor *query,
    const aclTensor *key,
    const aclTensor *weights,
    const aclTensor *queryDequantScale,
    const aclTensor *keyDequantScale,
    const aclTensor *blockTableOptional,
    const char *layoutQueryOptional,
    const char *layoutKeyOptional,
    const aclTensor *out,
    const uint64_t *workspaceSize)
{
    // 参数指针判空
    CHECK_RET(query != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(key != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(weights != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(queryDequantScale != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(keyDequantScale != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(blockTableOptional != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(layoutQueryOptional != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(layoutKeyOptional != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(out != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(workspaceSize != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

aclnnStatus quantLightningIndexerContiguous(
    const aclTensor *&query,
    const aclTensor *&weights,
    const aclTensor *&queryDequantScale,
    const aclTensor *&keyDequantScale,
    const aclTensor *&actualSeqLengthsQueryOptional,
    const aclTensor *&actualSeqLengthsKeyOptional,
    const aclTensor *&blockTableOptional,
    aclOpExecutor *executor)
{
    query = l0op::Contiguous(query, executor);
    CHECK_RET(query != nullptr, ACLNN_ERR_INNER_NULLPTR);
    weights = l0op::Contiguous(weights, executor);
    CHECK_RET(weights != nullptr, ACLNN_ERR_INNER_NULLPTR);
    queryDequantScale = l0op::Contiguous(queryDequantScale, executor);
    CHECK_RET(queryDequantScale != nullptr, ACLNN_ERR_INNER_NULLPTR);
    keyDequantScale = l0op::Contiguous(keyDequantScale, executor);
    CHECK_RET(keyDequantScale != nullptr, ACLNN_ERR_INNER_NULLPTR);

    if (actualSeqLengthsQueryOptional) {
        actualSeqLengthsQueryOptional = l0op::Contiguous(actualSeqLengthsQueryOptional, executor);
        CHECK_RET(actualSeqLengthsQueryOptional != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    if (actualSeqLengthsKeyOptional) {
        actualSeqLengthsKeyOptional = l0op::Contiguous(actualSeqLengthsKeyOptional, executor);
        CHECK_RET(actualSeqLengthsKeyOptional != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    if (blockTableOptional) {
        blockTableOptional = l0op::Contiguous(blockTableOptional, executor);
        CHECK_RET(blockTableOptional != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    return ACLNN_SUCCESS;
}

static const aclTensor* calNoContiguous(const aclTensor *self, aclOpExecutor *executor)
{
    aclTensor* newSelf = executor->CreateView(self, self->GetViewShape(), self->GetStorageShape(),
                                                self->GetViewStrides(), self->GetViewOffset());
    CHECK_RET(newSelf != nullptr, nullptr);
    return newSelf;
}

static const aclTensor* tensorContiguous(const aclTensor *tensor, aclOpExecutor *executor, const char *tensorName)
{
    if (tensor == nullptr) {
        return nullptr;
    }
    auto strides = tensor->GetViewStrides();
    if (!IsContiguous(tensor)) {
        printf("QuantLightningIndexer's %s tensor is non-contiguous\n", tensorName);
        return calNoContiguous(tensor, executor);
    }

    printf("QuantLightningIndexer's %s tensor is contiguous\n", tensorName);
    tensor = l0op::Contiguous(tensor, executor);
    CHECK_RET(tensor != nullptr, nullptr);

    return tensor;
}

aclnnStatus aclnnQuantLightningIndexerGetWorkspaceSize(
    const aclTensor *query,
    const aclTensor *key,
    const aclTensor *weights,
    const aclTensor *queryDequantScale,
    const aclTensor *keyDequantScale,
    const aclTensor *actualSeqLengthsQueryOptional,
    const aclTensor *actualSeqLengthsKeyOptional,
    const aclTensor *blockTableOptional,
    int64_t queryQuantMode,
    int64_t keyQuantMode,
    char *layoutQueryOptional,
    char *layoutKeyOptional,
    int64_t sparseCount,
    int64_t sparseMode,
    int64_t preTokens,
    int64_t nextTokens,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    CHECK_RET(quantLightningIndexerCheckTensorNull(query, key, weights, queryDequantScale, keyDequantScale,
        blockTableOptional, layoutQueryOptional, layoutKeyOptional, out, workspaceSize) == ACLNN_SUCCESS,
        ACLNN_ERR_PARAM_NULLPTR);

    CHECK_RET(quantLightningIndexerCheckDataType(query, key, weights, queryDequantScale, keyDequantScale,
        blockTableOptional, out), ACLNN_ERR_PARAM_INVALID);

    L2_DFX_PHASE_1(aclnnQuantLightningIndexer,
                    DFX_IN(query, key, weights, queryDequantScale, keyDequantScale, actualSeqLengthsQueryOptional, 
                    actualSeqLengthsKeyOptional, blockTableOptional, queryQuantMode, keyQuantMode, layoutQueryOptional,
                    layoutKeyOptional, sparseCount, sparseMode, preTokens, nextTokens),
                    DFX_OUT(out));

    // 获取executor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    if (out->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 判断inputLayout
    if (strcmp(layoutQueryOptional, "TND") != 0 && strcmp(layoutQueryOptional, "BSND") != 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Layout %s is not TND and BSND, invalid shape, please check", layoutQueryOptional);
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_ERR_PARAM_INVALID;
    }

    aclOpExecutor *l0Executor = uniqueExecutor.get();

    // 非连续转连续
    CHECK_RET(quantLightningIndexerContiguous(query, weights, queryDequantScale, keyDequantScale, actualSeqLengthsQueryOptional,
            actualSeqLengthsKeyOptional, blockTableOptional, l0Executor) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    const aclTensor *newKey = tensorContiguous(key, l0Executor, "key");

    // 调用L0接口获得输出
    auto l0QuantLightningIndexerOuts = l0op::QuantLightningIndexer(
            query, newKey, weights, queryDequantScale, keyDequantScale, actualSeqLengthsQueryOptional,
            actualSeqLengthsKeyOptional, blockTableOptional, queryQuantMode, keyQuantMode, layoutQueryOptional,
            layoutKeyOptional, sparseCount, sparseMode, preTokens, nextTokens, l0Executor);

    // 检查输出
    if (l0QuantLightningIndexerOuts == nullptr) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_ERR_INNER_NULLPTR;
    }

    auto viewCopyResult = l0op::ViewCopy(l0QuantLightningIndexerOuts, out, l0Executor);
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnQuantLightningIndexer(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnQuantLightningIndexer);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

}  // namespace

#ifdef __cplusplus
}
#endif