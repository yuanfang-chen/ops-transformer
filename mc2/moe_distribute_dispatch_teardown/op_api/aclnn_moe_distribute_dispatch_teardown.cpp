/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn_moe_distribute_dispatch_teardown.h"
#include <algorithm>
#include "op_mc2.h"
#include "common/op_host/op_api/matmul_util.h"
#include "op_mc2_def.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/op_log.h"
#include "opdev/common_types.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

static constexpr int32_t DISPATCH_DYNAMIC_QUANT_MODE = 2;
enum NnopbaseHcclServerType {
    NNOPBASE_HCCL_SERVER_TYPE_AICPU = 0,
    NNOPBASE_HCCL_SERVER_TYPE_MTE,
    NNOPBASE_HCCL_SERVER_TYPE_END
};

extern aclnnStatus aclnnInnerMoeDistributeDispatchTeardownGetWorkspaceSize(
    const aclTensor* x, const aclTensor* y, const aclTensor* expertIds, const aclTensor* commCmdInfo,
    const char* groupEp, int64_t epWorldSize, int64_t epRankId,
    int64_t moeExpertNum, int64_t expertShardType, int64_t sharedExpertNum, int64_t sharedExpertRankNum,
    int64_t quantMode, int64_t globalBs, int64_t expertTokenNumsType, int64_t commType, char* commAlg,
    aclTensor* expandXOut, aclTensor* dynamicScalesOut, aclTensor* assistInfoForCombineOut,
    aclTensor* expertTokenNumsOut, uint64_t* workspaceSize, aclOpExecutor** executor);
extern aclnnStatus aclnnInnerMoeDistributeDispatchTeardown(void* workspace, uint64_t workspaceSize,
                                                           aclOpExecutor* executor, aclrtStream stream);
extern "C" void __attribute__((weak)) NnopbaseSetHcclServerType(void* executor, NnopbaseHcclServerType sType);

// check nullptr
static bool CheckNotNull(
    const aclTensor* x, const aclTensor* y, const aclTensor* expertIds, const aclTensor* commCmdInfo,
    const char* groupEp, aclTensor* expandXOut, aclTensor* dynamicScalesOut, aclTensor* assistInfoForCombineOut,
    aclTensor* expertTokenNumsOut)
{
    OP_LOGD("aclnn_moe_distribute_dispatch CheckNotNull start");
    OP_CHECK_NULL(x, return false);
    OP_CHECK_NULL(y, return false);
    OP_CHECK_NULL(expertIds, return false);
    OP_CHECK_NULL(commCmdInfo, return false);
    OP_CHECK_NULL(expandXOut, return false);
    OP_CHECK_NULL(dynamicScalesOut, return false);
    OP_CHECK_NULL(assistInfoForCombineOut, return false);
    OP_CHECK_NULL(expertTokenNumsOut, return false);
    OP_LOGD("aclnn_moe_distribute_dispatch CheckNotNull success");
    if ((groupEp == nullptr) || (strnlen(groupEp, HCCL_GROUP_NAME_MAX) == 0)) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "group gropuEp name is Empty");
        return false;
    }
    return true;
}

// 入参教验
static aclnnStatus CheckParams(
    const aclTensor* x, const aclTensor* y, const aclTensor* expertIds, const aclTensor* commCmdInfo,
    const char* groupEp, int64_t epWorldSize, int64_t epRankId,
    int64_t expertShardType, int64_t sharedExpertRankNum, int64_t moeExpertNum, int64_t quantMode, int64_t globalBs,
    int64_t commType, char* commAlg, aclTensor* expandXOut, aclTensor* dynamicScalesOut,
    aclTensor* assistInfoForCombineOut, aclTensor* expertTokenNumsOut)
{
    OP_LOGD("aclnn_moe_distribute_dispatch_teardown CheckParams start");
    CHECK_RET(CheckNotNull(x, y, expertIds, commCmdInfo, groupEp, expandXOut, dynamicScalesOut,
                           assistInfoForCombineOut, expertTokenNumsOut),
              ACLNN_ERR_PARAM_NULLPTR);
    if (quantMode == DISPATCH_DYNAMIC_QUANT_MODE) {
        OP_LOGD("quantMode = 2, dynamicScalesOut can't be null");
        CHECK_RET(dynamicScalesOut != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    }
    if (strnlen(groupEp, HCCL_GROUP_NAME_MAX) >= HCCL_GROUP_NAME_MAX) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Required groupEp name exceeds %zu", HCCL_GROUP_NAME_MAX);
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    OP_LOGD("aclnn_moe_distribute_dispatch_teardown CheckParams success");
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnMoeDistributeDispatchTeardownGetWorkspaceSize(
    const aclTensor* x, const aclTensor* y, const aclTensor* expertIds, const aclTensor* commCmdInfo,
    const char* groupEp, int64_t epWorldSize, int64_t epRankId,
    int64_t moeExpertNum, int64_t expertShardType, int64_t sharedExpertNum, int64_t sharedExpertRankNum,
    int64_t quantMode, int64_t globalBs, int64_t expertTokenNumsType, int64_t commType, char* commAlg,
    aclTensor* expandXOut, aclTensor* dynamicScalesOut, aclTensor* assistInfoForCombineOut,
    aclTensor* expertTokenNumsOut, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_LOGD("aclnnMoeDistributeDispatchTeardownGetWorkspaceSize start");
    auto ret_param = CheckParams(
        x, y, expertIds, commCmdInfo, groupEp, epWorldSize, epRankId, expertShardType,
        sharedExpertRankNum, moeExpertNum, quantMode, globalBs, commType, commAlg, expandXOut, dynamicScalesOut,
        assistInfoForCombineOut, expertTokenNumsOut);
    CHECK_RET(ret_param == ACLNN_SUCCESS, ret_param);

    aclnnStatus ret = aclnnInnerMoeDistributeDispatchTeardownGetWorkspaceSize(
        x, y, expertIds, commCmdInfo, groupEp, epWorldSize, epRankId, moeExpertNum,
        expertShardType, sharedExpertNum, sharedExpertRankNum, quantMode, globalBs, expertTokenNumsType, commType,
        commAlg, expandXOut, dynamicScalesOut, assistInfoForCombineOut, expertTokenNumsOut, workspaceSize, executor);
    return ret;
}

aclnnStatus aclnnMoeDistributeDispatchTeardown(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                               aclrtStream stream)
{
    if (NnopbaseSetHcclServerType) {
        NnopbaseSetHcclServerType(executor, NNOPBASE_HCCL_SERVER_TYPE_MTE);
    }
    aclnnStatus ret = aclnnInnerMoeDistributeDispatchTeardown(workspace, workspaceSize, executor, stream);
    return ret;
}
#ifdef __cplusplus
}
#endif