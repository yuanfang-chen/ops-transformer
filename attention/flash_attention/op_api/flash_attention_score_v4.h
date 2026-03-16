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
 * \file flash_attention_score_v4.h
 * \brief FlashAttentionScoreV4 L0 (level-0) 算子接口声明
 *
 * 参考来源：
 *   flash_attention_score/op_api/flash_attention_score.h
 *   — 复用 FlashAttentionScore L0 op 声明模式，调整参数列表以匹配 V4 接口
 *     （移除 p_scale，调整可选输入顺序，增加 seed/offset）。
 */

#ifndef OP_API_INC_LEVEL0_OP_FLASH_ATTENTION_SCORE_V4_OP_H_
#define OP_API_INC_LEVEL0_OP_FLASH_ATTENTION_SCORE_V4_OP_H_

#include "opdev/op_executor.h"

namespace l0op {

/**
 * @brief FlashAttentionScoreV4 L0 算子接口
 *
 * 输入参数顺序与 FlashAttentionScoreV4 算子 def.cpp 中的输入/属性定义顺序一致：
 *   - 必选：query / key / value
 *   - 可选 tensor：real_shift(pse) / drop_mask / padding_mask / atten_mask /
 *                  query_rope / key_rope / d_scale_q / d_scale_k / d_scale_v /
 *                  sink
 *   - 可选 IntArray：prefix / actual_seq_qlen / actual_seq_kvlen /
 *                    q_start_idx / kv_start_idx
 *   - 属性：scaleValue / keepProb / preTockens / nextTockens / headNum /
 *           inputLayout / innerPrecise / sparseMode / outDtype / pseType /
 *           softmaxOutLayout / seed / offset
 *
 * 返回值：4 个输出 tensor 的数组 [softmaxMax, softmaxSum, softmaxOut, attentionOut]
 *
 * 非量化场景下 dScaleQ/K/V 以及 pScale 传 nullptr。
 *
 * @return std::array<const aclTensor *, 4>
 *   [0] softmaxMax   — FLOAT32, 训练反向所需，推理忽略
 *   [1] softmaxSum   — FLOAT32, 训练反向所需，推理忽略
 *   [2] softmaxOut   — 与 query 同类型（空 tensor，接口保留）
 *   [3] attentionOut — 与 query 同类型，attention 计算结果
 */
const std::array<const aclTensor *, 4> FlashAttentionScoreV4(
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
    int64_t            preTockens,
    int64_t            nextTockens,
    int64_t            headNum,
    const char        *inputLayout,
    int64_t            innerPrecise,
    int64_t            sparseMode,
    int64_t            outDtype,
    int64_t            pseType,
    const char        *softmaxOutLayout,
    int64_t            seed,
    int64_t            offset,
    aclOpExecutor     *executor);

} // namespace l0op

#endif // OP_API_INC_LEVEL0_OP_FLASH_ATTENTION_SCORE_V4_OP_H_
