/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_ACLNN_ALL_GATHER_MATMUL_INNER_
#define OP_API_ACLNN_ALL_GATHER_MATMUL_INNER_

#include <string>
#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"
#include "hccl/hccl_types.h"

#ifdef __cplusplus
extern "C" {
#endif
/**
 * 算子功能：实现allGather + mm 融合计算
 * @brief aclnnAllGatherMatmulInner的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 * @param [in] x1: matmul左矩阵，数据类型支持：float16, bfloat16, float8_e4m3fn, float8_e5m2, hifloat8, int8。
 * @param [in] x2: matmul右矩阵，数据类型支持：float16, bfloat16, float8_e4m3fn, float8_e5m2, hifloat8, int8。
 * @param [in] context: 上下文矩阵，数据类型支持：uint64。
 * @param [in] bias: 偏置矩阵，数据类型支持：float16, bfloat16, float32。
 * @param [in] group: 通信分组，字符串类型。
 * @param [in] commTurn: 通信轮数，整数类型。
 * @param [in] rankSize: 通信分组内的rank数量，整数类型。
 * @param [out] output: 计算+通信的结果，数据类型支持：float16, bfloat16, float32。
 * @param [out] gatherOut: 通信结果，数据类型支持：float16, bfloat16, float32。
 * @param [out] workspaceSize: 返回需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码
 */
ACLNN_API aclnnStatus aclnnAllGatherMatmulInnerGetWorkspaceSize(const aclTensor* x1,
    const aclTensor *context,
    const char *group, int64_t commTurn, uint32_t rankSize,               
    const aclTensor* output, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnAllGatherMatmulInner的第二段接口，用于执行计算。 
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnAllGatherMatmulInnerGetWorkspaceSize获取。
 * @param [in] exector: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码
 */
ACLNN_API aclnnStatus aclnnAllGatherMatmulInner(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                   aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_ALL_GATHER_MATMUL_V2_