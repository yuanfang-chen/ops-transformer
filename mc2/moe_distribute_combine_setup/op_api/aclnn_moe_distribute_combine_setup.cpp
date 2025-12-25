/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn_moe_distribute_combine_setup.h"
#include <algorithm>
#include "op_mc2.h"
#include "matmul_util.h"
#include "op_mc2_def.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/op_log.h"
#include "opdev/common_types.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

static constexpr size_t TWO_DIM = 2;
static constexpr int64_t ALIGN_8 = 8;
static constexpr int64_t ALIGN_32 = 32;
static constexpr int64_t ALIGN_512 = 512;
static constexpr int64_t COMM_CMD_INFO_BASE = 16;

enum NnopbaseHcclServerType {
    NNOPBASE_HCCL_SERVER_TYPE_AICPU = 0,
    NNOPBASE_HCCL_SERVER_TYPE_MTE,
    NNOPBASE_HCCL_SERVER_TYPE_END
};

inline int64_t Align(int64_t x, int64_t base)
{
    return ((x + base - 1) / base) * base;
}

extern aclnnStatus aclnnInnerMoeDistributeCombineSetupGetWorkspaceSize(
    const aclTensor* expandX, const aclTensor* expertIds, const aclTensor* assistInfoForCombine, const char* groupEp,
    int64_t epWorldSize, int64_t epRankId, int64_t moeExpertNum, int64_t expertShardType, int64_t sharedExpertNum,
    int64_t sharedExpertRankNum, int64_t globalBs, int64_t commQuantMode, int64_t commType, const char* commAlg,
    aclTensor* quantExpandXOut, aclTensor* commCmdInfoOut, uint64_t* workspaceSize, aclOpExecutor** executor);
extern aclnnStatus aclnnInnerMoeDistributeCombineSetup(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                                       aclrtStream stream);
extern "C" void __attribute__((weak)) NnopbaseSetHcclServerType(void* executor, NnopbaseHcclServerType sType);

// check nullptr
static bool CheckNotNull(
    const aclTensor* expandX, const aclTensor* expertIds, const aclTensor* assistInfoForCombine, const char* groupEp,
    aclTensor* quantExpandXOut, aclTensor* commCmdInfoOut)
{
    OP_LOGD("aclnn_moe_distribute_combine_setup CheckNotNull start");
    OP_CHECK_NULL(expandX, return false);
    OP_CHECK_NULL(expertIds, return false);
    OP_CHECK_NULL(assistInfoForCombine, return false);
    OP_CHECK_NULL(quantExpandXOut, return false);
    OP_CHECK_NULL(commCmdInfoOut, return false);
    OP_LOGD("aclnn_moe_distribute_combine_setup CheckNotNull success");
    if ((groupEp == nullptr) || (strnlen(groupEp, HCCL_GROUP_NAME_MAX) == 0)) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "group gropuEp name is Empty");
        return false;
    }
    return true;
}

// 入参教验
static aclnnStatus CheckParams(
    const aclTensor* expandX, const aclTensor* expertIds, const aclTensor* assistInfoForCombine, const char* groupEp,
    int64_t epWorldSize, int64_t epRankId, int64_t moeExpertNum, int64_t expertShardType, int64_t sharedExpertNum,
    int64_t sharedExpertRankNum, int64_t globalBs, int64_t commQuantMode, int64_t commType, const char* commAlg,
    aclTensor* quantExpandXOut, aclTensor* commCmdInfoOut)
{
    (void)epWorldSize;
    (void)epRankId;
    (void)moeExpertNum;
    (void)expertShardType;
    (void)sharedExpertNum;
    (void)sharedExpertRankNum;
    (void)globalBs;
    (void)commQuantMode;
    (void)commType;
    (void)commAlg;
    OP_LOGD("aclnn_moe_distribute_combine_setup CheckParams start");
    CHECK_RET(CheckNotNull(expandX, expertIds, assistInfoForCombine, groupEp, quantExpandXOut, commCmdInfoOut),
              ACLNN_ERR_PARAM_NULLPTR);
    if (strnlen(groupEp, HCCL_GROUP_NAME_MAX) >= HCCL_GROUP_NAME_MAX) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Required groupEp name exceeds %zu", HCCL_GROUP_NAME_MAX);
        return ACLNN_ERR_PARAM_INVALID;
    }
    OP_LOGD("aclnn_moe_distribute_combine_setup CheckParams success");
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnMoeDistributeCombineSetupGetWorkspaceSize(
    const aclTensor* expandX, const aclTensor* expertIds, const aclTensor* assistInfoForCombine, const char* groupEp,
    int64_t epWorldSize, int64_t epRankId, int64_t moeExpertNum, int64_t expertShardType, int64_t sharedExpertNum,
    int64_t sharedExpertRankNum, int64_t globalBs, int64_t commQuantMode, int64_t commType, const char* commAlg,
    aclTensor* quantExpandXOut, aclTensor* commCmdInfoOut, uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    OP_LOGD("aclnnMoeDistributeCombineSetupGetWorkspaceSize start.");
    auto retParam = CheckParams(expandX, expertIds, assistInfoForCombine, groupEp, 
                                 epWorldSize, epRankId, moeExpertNum, expertShardType, sharedExpertNum, 
                                 sharedExpertRankNum, globalBs, commQuantMode,
                                 commType, commAlg, quantExpandXOut, commCmdInfoOut);
    CHECK_RET(retParam == ACLNN_SUCCESS, retParam);

    aclnnStatus retStatus = aclnnInnerMoeDistributeCombineSetupGetWorkspaceSize(
        expandX, expertIds, assistInfoForCombine, groupEp, epWorldSize, epRankId, moeExpertNum, 
        expertShardType, sharedExpertNum, sharedExpertRankNum, globalBs, commQuantMode, commType, commAlg, 
        quantExpandXOut, commCmdInfoOut, workspaceSize, executor);
    return retStatus;
}

aclnnStatus aclnnMoeDistributeCombineSetupTeardownCalcOutputSize(
    const aclTensor* expandX, const aclTensor* expertIds, const aclTensor* assistInfoForCombine, const char* groupEp,
    int64_t epWorldSize, int64_t epRankId, int64_t moeExpertNum, int64_t expertShardType, int64_t sharedExpertNum,
    int64_t sharedExpertRankNum, int64_t globalBs, int64_t commQuantMode, int64_t commType, const char* commAlg,
    uint64_t& tokenMsgSize, uint64_t& commCmdInfoOutSize)
{
    (void)expertIds;
    (void)assistInfoForCombine;
    (void)groupEp;
    (void)epRankId;
    (void)moeExpertNum;
    (void)expertShardType;
    (void)sharedExpertNum;
    (void)sharedExpertRankNum;
    (void)globalBs;
    (void)commQuantMode;
    (void)commType;
    (void)commAlg;
    OP_CHECK_NULL(expandX, return ACLNN_ERR_PARAM_NULLPTR);
    if (expandX->GetViewShape().GetDimNum() != TWO_DIM) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "expandX's DimNum shoule be 2, but got %lu.", expandX->GetViewShape().GetDimNum());
        return ACLNN_ERR_PARAM_INVALID;
    }
    int64_t a = expandX->GetViewShape().GetDim(0);
    int64_t h = expandX->GetViewShape().GetDim(1);
    if (a <= 0 || h <= 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "expandX's Dim shoule be greater than zero.");
        return ACLNN_ERR_PARAM_INVALID;
    }

    tokenMsgSize =
        static_cast<uint64_t>(Align(Align(h, ALIGN_32) + Align(h, ALIGN_8) / ALIGN_8 * sizeof(float), ALIGN_512));
    commCmdInfoOutSize = static_cast<uint64_t>((a + epWorldSize) * COMM_CMD_INFO_BASE);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnMoeDistributeCombineSetup(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                           aclrtStream stream)
{
    if (NnopbaseSetHcclServerType) {
        NnopbaseSetHcclServerType(executor, NNOPBASE_HCCL_SERVER_TYPE_AICPU);
    }
    aclnnStatus ret = aclnnInnerMoeDistributeCombineSetup(workspace, workspaceSize, executor, stream);
    return ret;
}
#ifdef __cplusplus
}
#endif