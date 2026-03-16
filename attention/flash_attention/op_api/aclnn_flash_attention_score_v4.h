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
 * \file aclnn_flash_attention_score_v4.h
 * \brief aclnnFlashAttentionScoreV4 两段式接口声明
 *
 * 训练推理归一 Flash Attention 算子（非量化）对外 C 接口。
 * 接口定义遵循 aclnnFlashAttentionScoreV4 接口文档。
 *
 * 参考来源：
 *   flash_attention_score/op_api/aclnn_flash_attention_score.h
 *   — 第一段 / 第二段接口声明模式完全相同，仅函数名、参数列表按 V4 调整。
 */

#ifndef OP_API_INC_LEVEL2_ACLNN_FLASH_ATTENTION_SCORE_V4_H_
#define OP_API_INC_LEVEL2_ACLNN_FLASH_ATTENTION_SCORE_V4_H_

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnFlashAttentionScoreV4 第一段接口：根据计算流程计算 workspace 大小。
 *
 * @domain aclnn_ops_train
 *
 * @par 接口描述
 * 训练推理归一 FlashAttention 算子（非量化），使用 FlashAttention 算法计算
 * self-attention 正向结果。
 *   - 训练场景：输出 softmaxMaxOut / softmaxSumOut 供反向梯度使用。
 *   - 推理场景：调用方可忽略 softmaxMaxOut / softmaxSumOut；keepProb=1.0 禁用 dropout。
 *
 * @par Dropout 行为（V4 新增）
 *   - keepProb == 1.0：不执行 dropout，seed/offset 不生效。
 *   - keepProb < 1.0 且 dropMaskOptional != nullptr：使用外部传入的 dropMask（优先）。
 *   - keepProb < 1.0 且 dropMaskOptional == nullptr：使用 seed/offset 内部生成 dropMask。
 *
 * @param query              [in]  必选，FLOAT16/BF16/FLOAT32
 * @param key                [in]  必选，类型与 query 一致
 * @param value              [in]  必选，类型与 query 一致
 * @param realShiftOptional  [in]  可选，PSE；FLOAT16/BF16/FLOAT32
 * @param dropMaskOptional   [in]  可选，Dropout mask；UINT8
 * @param paddingMaskOptional [in] 可选，暂未使用
 * @param attenMaskOptional  [in]  可选，注意力掩码；BOOL/UINT8；1 表示不参与计算
 * @param queryRopeOptional  [in]  可选，Q 的 RoPE 分量；FLOAT16/BF16
 * @param keyRopeOptional    [in]  可选，K 的 RoPE 分量；FLOAT16/BF16
 * @param dScaleQOptional    [in]  可选，Q 量化参数（非量化场景传 nullptr）；FLOAT32
 * @param dScaleKOptional    [in]  可选，K 量化参数（非量化场景传 nullptr）；FLOAT32
 * @param dScaleVOptional    [in]  可选，V 量化参数（非量化场景传 nullptr）；FLOAT32
 * @param sinkOptional       [in]  可选，保留参数（传 nullptr）
 * @param prefixOptional     [in]  可选，prefix 稀疏 N 值列表；INT64
 * @param actualSeqQLenOptional [in] 可选，每 Batch Q 实际序列长度；INT64
 * @param actualSeqKvLenOptional [in] 可选，每 Batch KV 实际序列长度；INT64
 * @param qStartIdxOptional  [in]  可选，外切 Q 起始索引；INT64
 * @param kvStartIdxOptional [in]  可选，外切 KV 起始索引；INT64
 * @param scaleValue         [in]  缩放系数（公式中的 scale）
 * @param keepProb           [in]  Dropout 保留率，取值 (0, 1]
 * @param preTokens          [in]  稀疏左边界（sliding window）
 * @param nextTokens         [in]  稀疏右边界
 * @param headNum            [in]  单卡 Q 的 head 数
 * @param inputLayout        [in]  输入排布：BSH / SBH / BSND / BNSD / TND
 * @param innerPrecise       [in]  精度控制；2=使能无效行计算
 * @param sparseMode         [in]  稀疏模式 0-6
 * @param outDtype           [in]  量化输出类型（非量化场景不生效）
 * @param pseType            [in]  PSE 类型：0/1=外部传入，2/3=内部生成
 * @param softmaxOutLayout   [in]  保留参数（传 nullptr 或 ""）
 * @param seed               [in]  Dropout mask 生成 seed（keepProb<1.0 时生效）
 * @param offset             [in]  Dropout mask 生成 offset（keepProb<1.0 时生效）
 * @param softmaxMaxOut      [out] 必选，FLOAT32，训练反向所需（推理可忽略）
 * @param softmaxSumOut      [out] 必选，FLOAT32，训练反向所需（推理可忽略）
 * @param softmaxOutOut      [out] 必选，保留接口（空 tensor）
 * @param attentionOutOut    [out] 必选，attention 计算结果，类型与 query 一致
 * @param workspaceSize      [out] workspace 大小（字节）
 * @param executor           [out] op 执行器
 *
 * @return aclnnStatus 返回状态码
 */
aclnnStatus aclnnFlashAttentionScoreV4GetWorkspaceSize(
    const aclTensor   *query,
    const aclTensor   *key,
    const aclTensor   *value,
    const aclTensor   *realShiftOptional,
    const aclTensor   *dropMaskOptional,
    const aclTensor   *paddingMaskOptional,
    const aclTensor   *attenMaskOptional,
    const aclTensor   *queryRopeOptional,
    const aclTensor   *keyRopeOptional,
    const aclTensor   *dScaleQOptional,
    const aclTensor   *dScaleKOptional,
    const aclTensor   *dScaleVOptional,
    const aclTensor   *sinkOptional,
    const aclIntArray *prefixOptional,
    const aclIntArray *actualSeqQLenOptional,
    const aclIntArray *actualSeqKvLenOptional,
    const aclIntArray *qStartIdxOptional,
    const aclIntArray *kvStartIdxOptional,
    double             scaleValue,
    double             keepProb,
    int64_t            preTokens,
    int64_t            nextTokens,
    int64_t            headNum,
    char              *inputLayout,
    int64_t            innerPrecise,
    int64_t            sparseMode,
    int64_t            outDtype,
    int64_t            pseType,
    char              *softmaxOutLayout,
    int64_t            seed,
    int64_t            offset,
    const aclTensor   *softmaxMaxOut,
    const aclTensor   *softmaxSumOut,
    const aclTensor   *softmaxOutOut,
    const aclTensor   *attentionOutOut,
    uint64_t          *workspaceSize,
    aclOpExecutor    **executor);

/**
 * @brief aclnnFlashAttentionScoreV4 第二段接口：执行计算。
 *
 * @param workspace      [in] Device 侧 workspace 内存地址
 * @param workspaceSize  [in] workspace 大小（由第一段接口返回）
 * @param executor       [in] op 执行器（由第一段接口返回）
 * @param stream         [in] 执行 Stream
 *
 * @return aclnnStatus 返回状态码
 */
aclnnStatus aclnnFlashAttentionScoreV4(
    void             *workspace,
    uint64_t          workspaceSize,
    aclOpExecutor    *executor,
    const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_LEVEL2_ACLNN_FLASH_ATTENTION_SCORE_V4_H_
