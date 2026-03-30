/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <tuple>
#include "aggregate_hidden_grad.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "aclnn_kernels/common/op_error_check.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(AggregateHiddenGrad);

bool AggregateHiddenGrad(const aclTensor *grad_output, const aclTensor *input, const aclTensor *weight,
                         const aclTensor *mask, aclTensor *grad_input, aclTensor *grad_weight, aclOpExecutor *executor)
{
    L0_DFX(AggregateHiddenGrad, grad_output, input, weight, mask, grad_input, grad_weight);

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(AggregateHiddenGrad, OP_INPUT(grad_output, input, weight, mask),
                                           OP_OUTPUT(grad_input, grad_weight), OP_ATTR());
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "AggregateHiddenGrad ADD_TO_LAUNCHER_LIST_AICORE failed.");
        return false;
    }
    return true;
}

} // namespace l0op
