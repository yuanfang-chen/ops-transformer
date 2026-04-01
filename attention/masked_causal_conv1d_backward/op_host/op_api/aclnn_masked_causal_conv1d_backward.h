/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_ACLNN_MASK_EDCAUSAL_CONV1D_BACKWARD_H_
#define OP_API_INC_ACLNN_MASK_EDCAUSAL_CONV1D_BACKWARD_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief MaskedCausalConv1dBackward 第一段接口：生成执行器，并返回 workspace 大小。
 */
ACLNN_API aclnnStatus aclnnMaskedCausalConv1dBackwardGetWorkspaceSize(const aclTensor *grad_output, const aclTensor *input,
                                                               const aclTensor *weight, const aclTensor *mask,
                                                               aclTensor *grad_input, aclTensor *grad_weight,
                                                               uint64_t *workspaceSize, aclOpExecutor **executor);

/**
 * @brief MaskedCausalConv1dBackward 第二段接口：执行计算。
 */
ACLNN_API aclnnStatus aclnnMaskedCausalConv1dBackward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                               aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
