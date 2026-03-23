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
 * \file moe_finalize_routing_v2_grad.cpp
 * \brief
 */

#include "moe_finalize_routing_v2_grad.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/shape_utils.h"
#include "opdev/make_op_executor.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(MoeFinalizeRoutingV2Grad);

const std::array<const aclTensor *, 2> MoeFinalizeRoutingV2Grad(
        const aclTensor* grad_y, const aclTensor* expanded_row_idx, const aclTensor* expanded_x,
        const aclTensor* scales, const aclTensor* expert_idx, const aclTensor* bias,
        int64_t drop_pad_mode, int64_t active_num, int64_t expert_num, int64_t expert_capacity,
        aclOpExecutor *executor) {
    L0_DFX(MoeFinalizeRoutingV2Grad, grad_y, expanded_row_idx, expanded_x, scales, expert_idx, bias,
            drop_pad_mode, active_num, expert_num, expert_capacity);

    op::Shape gradYShape = grad_y->GetViewShape();
    op::Shape expandedRowIdxShape = expanded_row_idx->GetViewShape();

    op::Shape outShapeGradExpandedX;
    outShapeGradExpandedX.SetDimNum(0);

    if (drop_pad_mode == 0 && active_num > 0 && active_num < expandedRowIdxShape.GetDim(0)) {
        outShapeGradExpandedX.AppendDim(active_num);
    } else if (drop_pad_mode == 1) {
        outShapeGradExpandedX.AppendDim(expert_num);
        outShapeGradExpandedX.AppendDim(expert_capacity);
    } else {
        outShapeGradExpandedX.AppendDim(expandedRowIdxShape.GetDim(0));
    }
    outShapeGradExpandedX.AppendDim(gradYShape.GetDim(1));

    auto grad_expanded_x = executor->AllocTensor(outShapeGradExpandedX, grad_y->GetDataType(), op::Format::FORMAT_ND);

    op::Shape outShapeGradScales;
    outShapeGradScales.SetDimNum(0);
    outShapeGradScales.AppendDim(gradYShape.GetDim(0));

    int64_t scalesDim1 = 1;
    if (scales != nullptr) {
        op::Shape scalesShape = scales->GetViewShape();
        scalesDim1 = scalesShape.GetDim(1);
    }
    outShapeGradScales.AppendDim(scalesDim1);

    auto grad_scales = executor->AllocTensor(outShapeGradScales, grad_y->GetDataType(), op::Format::FORMAT_ND);

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(MoeFinalizeRoutingV2Grad,
                              OP_INPUT(grad_y, expanded_row_idx, expanded_x, scales, expert_idx, bias),
                              OP_OUTPUT(grad_expanded_x, grad_scales),
                              OP_ATTR(drop_pad_mode, active_num, expert_num, expert_capacity));
    if (ret != ACL_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "MoeFinalizeRoutingV2GradAiCore ADD_TO_LAUNCHER_LIST_AICORE failed.");
        return {nullptr, nullptr};
    }
    return std::array<const aclTensor*, 2>{grad_expanded_x, grad_scales};
}
}  // namespace l0op
