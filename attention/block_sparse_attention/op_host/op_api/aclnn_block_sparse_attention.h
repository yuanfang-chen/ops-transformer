/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACLNN_BLOCK_SPARSE_ATTENTION_H_
#define ACLNN_BLOCK_SPARSE_ATTENTION_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnBlockSparseAttentionGetWorkspaceSize的第一段接口,根据具体的计算流程,计算workspace大小
 * @domain aclnn_ops_infer
 * 
 * @param [in] query: Query tensor输入
 * @param [in] key: Key tensor输入
 * @param [in] value: Value tensor输入
 * @param [in] blockSparseMaskOptional: block Sparse Mask 适配的参数
 * @param [in] attenMaskOptional: Attention mask (可选)
 * @param [in] blockShapeOptional: 稀疏块形状数组 [blockShapeX, blockShapeY]
 * @param [in] actualSeqLengthsOptional: 实际序列长度Q (可选)
 * @param [in] actualSeqLengthsKvOptional: 实际序列长度KV (可选)
 * @param [in] blockTableOptional: Block表用于PagedAttention (可选)
 * @param [in] qInputLayout: Query输入layout ("TND", "BNSD")
 * @param [in] kvInputLayout: KV输入layout ("TND", "BNSD")
 * @param [in] numKeyValueHeads: KV头数
 * @param [in] maskType: Mask类型
 * @param [in] scaleValue: 缩放因子 (double类型)
 * @param [in] innerPrecise: Softmax精度控制 (0=float32 softmax, 1=fp16 softmax)
 * @param [in] blockSize: Block大小
 * @param [in] preTokens, 新增：滑窗attention场景下，滑窗需要向前包含多少个token
 * @param [in] nextTokens,新增：滑窗attention场景下，滑窗需要向后包含多少个token
 * @param [in] softmaxLseFlag,新增：是否使能softmaxLse输出的标志位。
 * @param [in] attentionOut: Attention输出tensor
 * @param [in] softmaxLseOptional: Softmax log-sum-exp输出tensor (可选)
 * @param [out] workspaceSize: 返回计算所需workspace大小
 * @param [out] executor: 返回op执行器
 * @return aclnnStatus: 返回状态码
 */
__attribute__((visibility("default"))) aclnnStatus aclnnBlockSparseAttentionGetWorkspaceSize(
    const aclTensor *query,
    const aclTensor *key,
    const aclTensor *value,
    const aclTensor *blockSparseMaskOptional,
    const aclTensor *attenMaskOptional,
    const aclIntArray *blockShapeOptional,
    const aclIntArray *actualSeqLengthsOptional,
    const aclIntArray *actualSeqLengthsKvOptional,
    const aclTensor *blockTableOptional,
    char *qInputLayout,
    char *kvInputLayout,
    int64_t numKeyValueHeads,
    int64_t maskType,
    double scaleValue,
    int64_t innerPrecise,
    int64_t blockSize,
    int64_t preTokens, //新增参数
    int64_t nextTokens, //新增参数
    int64_t softmaxLseFlag, //新增参数
    const aclTensor *attentionOut,
    const aclTensor *softmaxLseOptional,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/**
 * @brief aclnnBlockSparseAttention的第二段接口,用于执行计算
 * 
 * @param [in] workspace: 工作空间地址
 * @param [in] workspaceSize: 工作空间大小
 * @param [in] executor: op执行器
 * @param [in] stream: acl stream流
 * @return aclnnStatus: 返回状态码
 */
__attribute__((visibility("default"))) aclnnStatus aclnnBlockSparseAttention(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // ACLNN_BLOCK_SPARSE_ATTENTION_H_

