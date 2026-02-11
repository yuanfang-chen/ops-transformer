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
 * \file aclnn_mhc_post.h
 * \brief MhcPost ACLNN API
 */

#ifndef OP_API_INC_MHC_POST_H
#define OP_API_INC_MHC_POST_H

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnMhcPostGetWorkspaceSize的第一段接口，计算workspace大小。
 * 功能描述：该算子实现Manifold-Constraint Hyper-Connection的Post部分，处理残差连接的后处理操作。
 * 计算公式：y = x + alpha * h_res + beta * h_out + gamma * h_post
 * @domain aclnn_ops_infer
 * @param [in] x：必选参数，Device侧的aclTensor，输入张量x，数据类型支持FLOAT16、FLOAT、BFLOAT16，数据格式支持ND。
 * @param [in] h_res：必选参数，Device侧的aclTensor，残差连接h_res，数据类型支持FLOAT16、FLOAT、BFLOAT16，数据格式支持ND。
 * @param [in] h_out：必选参数，Device侧的aclTensor，输出状态h_out，数据类型支持FLOAT16、FLOAT、BFLOAT16，数据格式支持ND。
 * @param [in] h_post：必选参数，Device侧的aclTensor，后处理h_post，数据类型支持FLOAT16、FLOAT、BFLOAT16，数据格式支持ND。
 * @param [in] alpha：可选参数，Host侧的float，h_res的权重系数，默认值1.0。
 * @param [in] beta：可选参数，Host侧的float，h_out的权重系数，默认值1.0。
 * @param [in] gamma：可选参数，Host侧的float，h_post的权重系数，默认值1.0。
 * @param [out] y：输出Tensor，计算结果y，数据类型支持FLOAT16、FLOAT、BFLOAT16，数据格式支持ND，输出形状与x一致。
 * @param [out] workspaceSize：返回用户需要在Device侧申请的workspace大小。
 * @param [out] executor：返回op执行器，包含了算子计算流程。
 * @return      aclnnStatus: 返回状态码
 */
__attribute__((visibility("default"))) aclnnStatus
aclnnMhcPostGetWorkspaceSize(const aclTensor *x, const aclTensor *h_res, const aclTensor *h_out,
                              const aclTensor *h_post, float alpha, float beta, float gamma,
                              const aclTensor *y, uint64_t *workspaceSize, aclOpExecutor **executor);

/**
 * @brief aclnnMhcPost的第二段接口，用于执行计算。
 * @param [in] workspace: 在Device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在Device侧申请的workspace大小，由第一段接口aclnnMhcPostGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: 指定执行任务的AscendCL stream流。
 * @return     aclnnStatus: 返回状态码
 */
__attribute__((visibility("default"))) aclnnStatus aclnnMhcPost(void *workspace, uint64_t workspaceSize,
                                                                 aclOpExecutor *executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_MHC_POST_H