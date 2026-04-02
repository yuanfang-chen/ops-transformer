/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <tuple>
#include <cstddef>
#include "opdev/make_op_executor.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/tensor_view_utils.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/op_log.h"
#include "aclnn_kernels/cast.h"
#include "opdev/common_types.h"
#include "masked_causal_conv1d_backward.h"
#include "aclnn_masked_causal_conv1d_backward.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {

aclnnStatus MaskedCausalConv1dBackwardCommonProcess(const aclTensor *grad_output, const aclTensor *input,
                                             const aclTensor *weight, const aclTensor *mask, aclTensor *grad_input,
                                             aclTensor *grad_weight, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    auto uniqueExecutor = CREATE_EXECUTOR();

    // grad_output / input 允许视图输入，weight 建议 Contiguous
    const aclTensor *goFinal =
        uniqueExecutor->CreateView(grad_output, grad_output->GetViewShape(), grad_output->GetStorageShape(),
                                   grad_output->GetViewStrides(), grad_output->GetViewOffset());
    CHECK_COND(goFinal != nullptr, ACLNN_ERR_INNER_NULLPTR, "CreateView for grad_output failed.");

    const aclTensor *inFinal = uniqueExecutor->CreateView(input, input->GetViewShape(), input->GetStorageShape(),
                                                          input->GetViewStrides(), input->GetViewOffset());
    CHECK_COND(inFinal != nullptr, ACLNN_ERR_INNER_NULLPTR, "CreateView for input failed.");

    weight = l0op::Contiguous(weight, uniqueExecutor.get());
    CHECK_COND(weight != nullptr, ACLNN_ERR_INNER_NULLPTR, "Contiguous weight failed.");

    if (mask != nullptr) {
        mask = l0op::Contiguous(mask, uniqueExecutor.get());
        CHECK_COND(mask != nullptr, ACLNN_ERR_INNER_NULLPTR, "Contiguous mask failed.");
    }

    bool ok = l0op::MaskedCausalConv1dBackward(goFinal, inFinal, weight, mask, grad_input, grad_weight, uniqueExecutor.get());
    CHECK_RET(ok, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

} // namespace

ACLNN_API aclnnStatus aclnnMaskedCausalConv1dBackwardGetWorkspaceSize(const aclTensor *grad_output, const aclTensor *input,
                                                               const aclTensor *weight, const aclTensor *mask,
                                                               aclTensor *grad_input, aclTensor *grad_weight,
                                                               uint64_t *workspaceSize, aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnMaskedCausalConv1dBackward, DFX_IN(grad_output, input, weight, mask),
                   DFX_OUT(grad_input, grad_weight));
    return MaskedCausalConv1dBackwardCommonProcess(grad_output, input, weight, mask, grad_input, grad_weight, workspaceSize,
                                            executor);
}

ACLNN_API aclnnStatus aclnnMaskedCausalConv1dBackward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                               aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnMaskedCausalConv1dBackward);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
