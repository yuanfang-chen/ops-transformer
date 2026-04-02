/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <algorithm>
#include "common/utils/op_mc2.h"
#include "common/utils/op_mc2_def.h"
#include "opdev/op_log.h"
#include "opdev/common_types.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"
#include "common/op_host/op_api/matmul_util.h"
#include "aclnn_kernels/common/op_error_check.h"

using namespace Ops::Transformer;
using namespace op;


#ifdef __cplusplus
extern "C" {
#endif

extern aclnnStatus aclnnInnerMegaMoeGetWorkspaceSize(
    const aclTensor* context, const aclTensor* x, const aclTensor* expertIds,
    const aclTensor* expertScales, int64_t epWorldSize, int64_t epRankId,
    int64_t moeExpertNum, int64_t cclBufferSize, int64_t maxRecvTokenNum,
    int64_t sharedExpertNum, int64_t dispatchQuantMode, int64_t dispatchQuantOutType,
    int64_t combineQuantMode, const char* commAlg, int64_t globalBs,
    aclTensor* y, uint64_t* workspaceSize, aclOpExecutor** executor);

extern aclnnStatus aclnnInnerMegaMoe(void* workspace, uint64_t workspaceSize,
                                     aclOpExecutor* executor, aclrtStream stream);

aclnnStatus aclnnMegaMoeGetWorkspaceSize(
    const aclTensor* context, const aclTensor* x, const aclTensor* expertIds,
    const aclTensor* expertScales, int64_t epWorldSize, int64_t epRankId,
    int64_t moeExpertNum, int64_t cclBufferSize, int64_t maxRecvTokenNum,
    int64_t sharedExpertNum, int64_t dispatchQuantMode, int64_t dispatchQuantOutType,
    int64_t combineQuantMode, const char* commAlg, int64_t globalBs,
    aclTensor* y, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_LOGD("aclnn_mega_moe WorkspaceSize start");

    OP_CHECK_NULL(context, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(x, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(expertIds, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(expertScales, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(y, return ACLNN_ERR_PARAM_NULLPTR);

    aclnnStatus getWorkspaceSizesRes = aclnnInnerMegaMoeGetWorkspaceSize(
        context, x, expertIds, expertScales, epWorldSize, epRankId, moeExpertNum,
        cclBufferSize, maxRecvTokenNum, sharedExpertNum, dispatchQuantMode,
        dispatchQuantOutType, combineQuantMode, commAlg, globalBs,
        y, workspaceSize, executor);

    return getWorkspaceSizesRes;
}

aclnnStatus aclnnMegaMoe(void* workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    OP_LOGD("aclnn_mega_moe start");
    return aclnnInnerMegaMoe(workspace, workspaceSize, executor, stream);
}
#ifdef __cplusplus
}
#endif
