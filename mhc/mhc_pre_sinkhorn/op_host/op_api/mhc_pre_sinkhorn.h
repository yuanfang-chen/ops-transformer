/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_MHC_PRE_SINKHORN_H
#define OP_API_INC_MHC_PRE_SINKHORN_H

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnMhcPreSinkhornGetWorkspaceSize 的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 * 算子功能：MhcPreSinkhorn正向算子
 * @param [in] h_res: 输入tensor，mhc_pre算子输出的H^res矩阵，shape为[B,S,n,n]或[T,n,n]，数据类型支持：FP32。
 * @param [in] eps: 归一化防除零参数，建议值：1e-6。
 * @param [in] numIters: Sinkhorn迭代次数，建议值：20，范围：1~100。
 * @param [in] outFlag: 输出标志位，0：仅输出sinkhorn结果，1：输出中间结果（当前版本仅支持0）。
 * @param [out] h_res_sinkhorn: 输出tensor，Sinkhorn变换后的双随机矩阵，shape与h_res一致，数据类型：FP32。
 * @param [out] norm_out: 输出tensor，迭代过程中的归一化中间结果（可选），shape为[2*numIters,n,n,B,S]或[2*numIters,n,n,T]，数据类型：FP32。
 * @param [out] sum_out: 输出tensor，迭代过程中的求和中间结果（可选），shape为[2*numIters,n,B,S]或[2*numIters,n,T]，数据类型：FP32。
 * @param [out] workspaceSize: 返回需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码
 */
ACLNN_API aclnnMhcPreSinkhornGetWorkspaceSize(const aclTensor *h_res, float eps, int64_t numIters,
                                                  int outFlag, aclTensor *h_res_sinkhorn,
                                                  aclTensor *norm_out, aclTensor *sum_out,
                                                  uint64_t *workspaceSize,
                                                  aclOpExecutor **executor);

/**
 * @brief aclnnMhcPreSinkhorn 的第二段接口，执行算子计算。
 * @domain aclnn_ops_infer
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口 aclnnMhcPreSinkhornGetWorkspaceSize 获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码
 */
ACLNN_API aclnnMhcPreSinkhorn(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_MHC_PRE_SINKHORN_H
