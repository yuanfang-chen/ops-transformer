/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aclnn_matmul_allto_all.cpp
 * \brief
 */
#ifndef OP_API_INC_MATMUL_ALLTO_ALL_
#define OP_API_INC_MATMUL_ALLTO_ALL_

#include <string>

#include "aclnn/aclnn_base.h"
#include "hccl/hccl.h"
#include "hccl/hccl_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * 算子功能：实现 mm + alltoall 融合计算
 * @brief aclnnMatmulAlltoAll的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 * @param [in] x1: matmul左矩阵，数据类型支持：float16, bf16。
 * @param [in] x2: matmul右矩阵，数据类型支持：float16, bf16。
 * @param [in] biasOptional: 偏置，数据类型支持：float16, float32。
 * @param [in] alltoAllAxesOptional: AlltoAll交换数据的方向，默认为 [-1, -2]。
 * @param [in] group: 标识通信域名称的字符串。
 * @param [in] transposeX1: 可选入参，计算输入。表明matmul的左矩阵是否需要转置，默认为false
 * @param [in] transposeX2: 可选入参，计算输入。表明matmul的右矩阵是否需要转置，默认为false
 * @param [out] output: 计算+通信的结果，数据类型：同输入。
 * @param [out] workspaceSize: 返回需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码
 */
__attribute__((visibility("default"))) aclnnStatus aclnnMatmulAlltoAllGetWorkspaceSize(const aclTensor* x1, const aclTensor* x2, const aclTensor* biasOptional,
                                                                                       const aclIntArray* alltoAllAxesOptional, const char* group, 
                                                                                       bool transposeX1, bool transposeX2, const aclTensor* output, 
                                                                                       uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnMatmulAlltoAll的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspace_size: 在npu device侧申请的workspace大小，由第一段接口aclnnMatmulAlltoAllGetWorkspaceSize获取。
 * @param [in] exector: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码
 */
__attribute__((visibility("default"))) aclnnStatus aclnnMatmulAlltoAll(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                           aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_MATMUL_ALLTO_ALL_