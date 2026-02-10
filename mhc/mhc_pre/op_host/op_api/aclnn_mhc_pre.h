/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_MHC_PRE_H
#define OP_API_INC_MHC_PRE_H

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnMhcPreGetWorkspaceSize 的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 * 算子功能：MhcPre正向算子
 * @param [in] x: 输入tensor，shape为[B,S,N,D]或[T,N,D]，数据类型支持：BF16/FP16。
 * @param [in] phi: 输入tensor，shape为[n^2+2n, nD]，数据类型支持：FP32。
 * @param [in] alpha: 输入tensor，shape为[3]，数据类型支持：FP32。
 * @param [in] bias: 输入tensor，shape为[n^2+2n]，数据类型支持：FP32。
 * @param [in] gamma: 可选输入tensor，shape为[n, D]，数据类型支持：FP32。
 * @param [in] norm_eps: 归一化epsilon参数。
 * @param [in] hc_eps: hyper connection epsilon参数。
 * @param [out] out_hin: 输出tensor，shape为[B,S,D]或[T,D]，数据类型：BF16/FP16。
 * @param [out] out_h_post: 输出tensor，shape为[B,S,N]或[T,N]，数据类型：FP32。
 * @param [out] out_h_res: 输出tensor，shape为[B,S,N,N]或[T,N,N]，数据类型：FP32。
 * @param [out] out_inv_rms: 输出tensor，shape为[B,S]或[T]，数据类型：FP32。
 * @param [out] out_mm_res: 输出tensor，shape为[B,S,N^2+2N]或[T,N^2+2N]，数据类型：FP32。
 * @param [out] out_h_pre: 输出tensor，shape为[B,S,N]或[T,N]，数据类型：FP32。
 * @param [out] workspaceSize: 返回需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码
 */
ACLNN_API aclnnStatus aclnnMhcPreGetWorkspaceSize(
    const aclTensor *x, const aclTensor *phi, const aclTensor *alpha, const aclTensor *bias, const aclTensor *gamma,
    int64_t out_flag, double norm_eps, double hc_eps,
    const aclTensor *out_hin, const aclTensor *out_h_post, const aclTensor *out_h_res,
    const aclTensor *out_inv_rms, const aclTensor *out_mm_res, const aclTensor *out_h_pre,
    uint64_t *workspaceSize, aclOpExecutor **executor);

/**
 * @brief aclnnMhcPre 的第二段接口，执行算子计算。
 * @domain aclnn_ops_infer
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口 aclnnMhcPreGetWorkspaceSize 获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码
 */
ACLNN_API aclnnStatus aclnnMhcPre(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                                        aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_MHC_PRE_H