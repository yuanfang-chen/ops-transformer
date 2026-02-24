/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to License for details. You may not use this file except in compliance with License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of software repository for the full text of the License.
 */

#ifndef OP_API_INC_ACLNN_MATRIX_TRANSFORM_ROPE_H
#define OP_API_INC_ACLNN_MATRIX_TRANSFORM_ROPE_H

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnMatrixTransformRopeGetWorkspaceSize 的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 * 算子功能：MatrixTransformRope 正向算子
 *
 * @param [in] x: 输入tensor，维度为4维，shape为[B, N, S, D]、[B, S, N, D]、[S, B, N, D]，数据类型支持：BF16/FP16/FP32, 支持ND格式。
 * @param [in] cos: 输入tensor，维度为4维，数据类型支持：BF16/FP16/FP32, 支持ND格式。当x为BNSD时，cos、sin支持11SD、B1SD、BNSD，当x为BSND时，cos、sin支持1S1D、BS1D、BSND，当x为SBND时，cos、sin支持S11D、SB1D、SBND。
 * @param [in] sin: 输入tensor，shape为4维，数据类型支持：BF16/FP16/FP32, 支持ND格式。当x为BNSD时，cos、sin支持11SD、B1SD、BNSD，当x为BSND时，cos、sin支持1S1D、BS1D、BSND，当x为SBND时，cos、sin支持S11D、SB1D、SBND。
 * @param [in] rotate: 输入tensor，维度为4维，shape为[D, D]，数据类型支持：BF16/FP16/FP32, 支持ND格式。
 * @param [out] out: 输入tensor，维度为4维，shape为[B, N, S, D]、[B, S, N, D]、[S, B, N, D]，和x保持一致，数据类型支持：BF16/FP16/FP32, 支持ND格式。
 * @param [out] workspaceSize: 返回需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码
 */
aclnnStatus aclnnMatrixTransformRopeGetWorkspaceSize(
    const aclTensor *x, const aclTensor *cos, const aclTensor *sin, const aclTensor *rotate,
    const aclTensor *out,
    uint64_t *workspaceSize, aclOpExecutor **executor);

/**
 * @brief aclnnMatrixTransformRope 的第二段接口，执行算子计算。
 * @domain aclnn_ops_infer
 *
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口 aclnnMatrixTransformRopeGetWorkspaceSize 获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码
 */
aclnnStatus aclnnMatrixTransformRope(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                   aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_ACLNN_MATRIX_TRANSFORM_ROPE_H
