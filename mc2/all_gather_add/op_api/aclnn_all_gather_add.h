/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_ALL_GATHER_ADD_
#define OP_API_INC_ALL_GATHER_ADD_

#include <string>

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"
#include "hccl/hccl_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @ingroup AscendCANN
 * @brief 计算 all_gather_add 所需 workspace 大小。
 *
 * @param a 输入 Tensor，每张卡上的本地输入矩阵，当前仅支持 shape=[1024,2048]、FLOAT16、ND。
 * @param b 输入 Tensor，加法输入矩阵，当前仅支持 shape=[2048,2048]、FLOAT16、ND。
 * @param group 输入参数，通信域名称。
 * @param rankSize 输入参数，通信域内参与通信的卡数，当前仅支持 2。
 * @param commTurn 输入参数，单次执行内部轮次，当前仅支持 2。
 * @param aGathered 输出 Tensor，AllGather 后结果，当前仅支持 shape=[2048,2048]、FLOAT16、ND。
 * @param c 输出 Tensor，加法结果，当前仅支持 shape=[2048,2048]、FLOAT16、ND。
 * @param workspaceSize 输出参数，返回所需 workspace 大小。
 * @param executor 输出参数，返回执行器。
 * @return ACLNN_SUCCESS: 成功；ACLNN_ERR_PARAM_NULLPTR: 输入参数为空指针；ACLNN_ERR_PARAM_INVALID: 输入参数无效。
 */
aclnnStatus aclnnAllGatherAddGetWorkspaceSize(const aclTensor* a, const aclTensor* b, const char* group,
                                              int64_t rankSize, int64_t commTurn, aclTensor* aGathered,
                                              aclTensor* c, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @ingroup AscendCANN
 * @brief 执行 all_gather_add。
 *
 * @param workspace 输入参数，Device 侧 workspace 地址。
 * @param workspaceSize 输入参数，workspace 大小。
 * @param executor 输入参数，执行器。
 * @param stream 输入参数，执行 stream。
 * @return ACLNN_SUCCESS: 成功；ACLNN_ERR_PARAM_NULLPTR: 输入参数为空指针；ACLNN_ERR_PARAM_INVALID: 输入参数无效。
 */
aclnnStatus aclnnAllGatherAdd(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_ALL_GATHER_ADD_
