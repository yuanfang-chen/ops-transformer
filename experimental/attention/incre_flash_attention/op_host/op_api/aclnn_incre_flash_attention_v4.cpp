/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_incre_flash_attention_v4.h"
#include "opdev/op_log.h"
#include <iostream>
#include "opdev/common_types.h"
using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

extern aclnnStatus aclnnInnerIncreFlashAttentionGetWorkspaceSize(
    const aclTensor *query, const aclTensorList *key, const aclTensorList *value, const aclTensor *pseShift,
    const aclTensor *attenMask, const aclIntArray *actualSeqLengths, const aclTensor *deqScale1,
    const aclTensor *quantScale1, const aclTensor *deqScale2, const aclTensor *quantScale2,
    const aclTensor *quantOffset2, const aclTensor *antiquantScale, const aclTensor *antiquantOffset,
    const aclTensor *blocktable, const aclTensor *kvPaddingSize, int64_t numHeads, double scaleValue, char *inputLayout,
    int64_t numKeyValueHeads, int64_t blockSize, int64_t innerPrecise, const aclTensor *attentionOut,
    uint64_t *workspaceSize, aclOpExecutor **executor);

extern aclnnStatus aclnnInnerIncreFlashAttention(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                                 const aclrtStream stream);

void printf_dim_list(const aclTensorList *tempKey)
{
    for (uint64_t i = 0; i < tempKey->Size(); i++) {
        printf("size[%d] ", i);
        if ((*tempKey)[i] != nullptr) {
            op::Shape viewShape = (*tempKey)[i]->GetViewShape();
            auto viewShapeDim = viewShape.GetDimNum();
            for (uint32_t j = 0; j < viewShapeDim; j++){
                printf("shape[%d] : %d ", j,viewShape[j]);
            }
        }
        printf("\n");
    }
}

void printf_dim(const aclTensor *tempKey)
{
    
    if (tempKey != nullptr) {
        op::Shape viewShape = tempKey->GetViewShape();
        auto viewShapeDim = viewShape.GetDimNum();
        for (uint32_t i = 0; i < viewShapeDim; i++){
            printf("shape[%d] : %d ",i, viewShape[i]);
        }
    }
    printf("\n");
    
}

aclnnStatus aclnnIncreFlashAttentionV4GetWorkspaceSize(
    const aclTensor *query, const aclTensorList *key, const aclTensorList *value, const aclTensor *pseShift,
    const aclTensor *attenMask, const aclIntArray *actualSeqLengths, const aclTensor *deqScale1,
    const aclTensor *quantScale1, const aclTensor *deqScale2, const aclTensor *quantScale2,
    const aclTensor *quantOffset2, const aclTensor *antiquantScale, const aclTensor *antiquantOffset,
    const aclTensor *blocktable, const aclTensor *kvPaddingSize, int64_t numHeads, double scaleValue, char *inputLayout,
    int64_t numKeyValueHeads, int64_t blockSize, int64_t innerPrecise, const aclTensor *attentionOut,
    uint64_t *workspaceSize, aclOpExecutor **executor)
{
    printf("ddddddddddddddddd\n");
    std::cout << "query" << &query << std::endl; printf_dim(query);
    std::cout << "key" << &key << std::endl; printf_dim_list(key);
    std::cout << "value" << &value << std::endl;printf_dim_list(value);
    std::cout << "atten_mask" << &attenMask << std::endl;printf_dim(attenMask);
    std::cout << "pse_shift" << &pseShift << std::endl;printf_dim(pseShift);
    std::cout << "antiquant_scale" << &antiquantScale << std::endl;printf_dim(antiquantScale);
    std::cout << "antiquant_offset" << &antiquantOffset << std::endl;printf_dim(antiquantOffset);
    std::cout << "block_table" << &blocktable << std::endl;printf_dim(blocktable);
    std::cout << "dequant_scale1" << &deqScale1 << std::endl;printf_dim(deqScale1);
    std::cout << "quant_scale1" << &quantScale1 << std::endl;printf_dim(quantScale1);
    std::cout << "dequant_scale2" << &deqScale2 << std::endl;printf_dim(deqScale2);
    std::cout << "quant_scale2" << &quantScale2 << std::endl;printf_dim(quantScale2);
    std::cout << "quant_offset2" << &quantOffset2 << std::endl;printf_dim(quantOffset2);
    std::cout << "kv_padding_size" << &kvPaddingSize << std::endl;printf_dim(kvPaddingSize);
    std::cout << "num_heads" << numHeads << std::endl;
    std::cout << "scale_value" << scaleValue << std::endl;
    std::cout << "input_layout" << inputLayout << std::endl;
    std::cout << "num_key_value_heads" << numKeyValueHeads << std::endl;
    std::cout << "block_size" << blockSize << std::endl;
    std::cout << "inner_precise" << innerPrecise << std::endl;
    aclnnStatus ret = aclnnInnerIncreFlashAttentionGetWorkspaceSize(
        query, key, value, pseShift, attenMask, actualSeqLengths, deqScale1, quantScale1, deqScale2, quantScale2,
        quantOffset2, antiquantScale, antiquantOffset, blocktable, kvPaddingSize, numHeads, scaleValue, inputLayout,
        numKeyValueHeads, blockSize, innerPrecise, attentionOut, workspaceSize, executor);

    return ret;
}

aclnnStatus aclnnIncreFlashAttentionV4(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                       const aclrtStream stream)
{
    aclnnStatus ret = aclnnInnerIncreFlashAttention(workspace, workspaceSize, executor, stream);
    return ret;
}

#ifdef __cplusplus
}
#endif