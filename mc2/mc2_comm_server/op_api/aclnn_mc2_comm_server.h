/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aclnn_mc2_comm_server.h
 * \brief
 */

#ifndef ACLNN_MC2_COMM_SERVER_H_
#define ACLNN_MC2_COMM_SERVER_H_

#include <string>
#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"
#include "hccl/hccl_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnMc2CommServer的第一段接口，根据具体的计算流程，计算workspace大小。
 * @param [in] ctx: 输入的上下文tensor，包含了所有的输入输出tensor的信息。
 * @param [out] workspaceSize: 输出的workspace大小，单位为字节。
 * @param [out] executor: 输出的op executor，用于后续的计算。
 * @return aclnnStatus: 返回值，返回状态码
 */
ACLNN_API aclnnStatus aclnnMc2CommServerGetWorkspaceSize(
    const aclTensor* ctx,
    uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * @brief aclnnMc2CommServer的第二段接口。
 * @param [in] ctx: 输入的上下文tensor，包含了所有的输入输出tensor的信息。
 * @param [in] workspaceSize: 输入的workspace大小，单位为字节。
 * @param [out] workspace: 输出的workspace内存指针。
 * @return aclnnStatus: 返回值，返回状态码
 */
ACLNN_API aclnnStatus aclnnMc2CommServer(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    const aclrtStream stream);

#ifdef __cplusplus
}
#endif


struct CommKfcParamDesc {
    uint64_t version : 4;    // 版本号，解耦context方案是2，否则是1
    uint64_t itemNum : 4;    // ctx数量
    uint64_t hasFfts : 1;    // 910下是否是ffts融合算子
    uint64_t tilingOff : 7;  // tilingdata指针所在的参数索引
    uint64_t isDyn : 48;     // 输入参数是否是动态输入
};

constexpr char KFC_SERVER_SO_NAME[] = "libccl_kernel.so";
constexpr char KFC_KERNEL_NAME[] = "RunAicpuKfcSrvLaunch";
char g_hostArgs[512] = {0};

#endif