/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_allto_allv_grouped_mat_mul.h"
#include "allto_allv_grouped_mat_mul_checker.h"
#include <algorithm>
#include "op_mc2_def.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/common_types.h"

namespace allto_allv_grouped_mat_mul_checker {

using namespace op;


aclnnStatus CheckSendAndRecv(const aclIntArray *sendCounts, const aclIntArray *recvCounts)
{
    if (sendCounts == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "sendCounts should not be null.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (recvCounts == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "recvCounts should not be null.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    uint64_t recvSize = 0U;
    uint64_t sendSize = 0U;
    aclGetIntArraySize(recvCounts, &recvSize);
    aclGetIntArraySize(sendCounts, &sendSize);
    if (recvSize == 0U) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "recvCounts should not be empty.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (sendSize == 0U) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "sendCounts should not be empty.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

} // namespace allto_allv_grouped_mat_mul_checker