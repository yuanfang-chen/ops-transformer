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
 * \file moe_init_routing_v2.cpp
 * \brief
 */

#include "moe_init_routing_v2.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/shape_utils.h"
#include "opdev/make_op_executor.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(MoeInitRoutingV2);

const std::array<const aclTensor *, 4> MoeInitRoutingV2(
        const aclTensor* x, const aclTensor* expert_idx,
        int64_t active_num, int64_t expert_capacity, int64_t expert_num, int64_t drop_pad_mode,
        int64_t expert_tokens_count_or_cumsum_flag, bool expert_tokens_before_capacity_flag,
        aclOpExecutor *executor) {
    L0_DFX(MoeInitRoutingV2, x, expert_idx, active_num, expert_capacity, expert_num, drop_pad_mode,
            expert_tokens_count_or_cumsum_flag, expert_tokens_before_capacity_flag);

    // Get input shapes
    op::Shape xShape = x->GetViewShape();
    op::Shape expertIdxShape = expert_idx->GetViewShape();

    // Get dimensions
    int64_t n = (xShape.GetDimNum() == 1) ? -1 : xShape.GetDim(0);
    int64_t cols = (xShape.GetDimNum() == 1) ? -1 : xShape.GetDim(1);
    int64_t k = (expertIdxShape.GetDimNum() == 1) ? -1 : expertIdxShape.GetDim(1);

    op::Shape outShapeExpandedX;
    outShapeExpandedX.SetDimNum(0);

    int64_t outActiveNum = -1;
    int64_t expandedRowIdxNum = -1;

    if (n > 0 && k > 0) {
        expandedRowIdxNum = n * k;
        outActiveNum = (active_num > 0) ? std::min(n * k, active_num) : n * k;
    }

    if (drop_pad_mode > 0) {
        // 3D shape: [expert_num, expert_capacity, cols]
        outShapeExpandedX.AppendDim(expert_num);
        outShapeExpandedX.AppendDim(expert_capacity);
        outShapeExpandedX.AppendDim((cols < 0) ? -1 : cols));
    } else {
        // 2D shape: [outActiveNum, cols]
        outShapeExpandedX.AppendDim(outActiveNum);
        outShapeExpandedX.AppendDim((cols < 0) ? -1 : cols));
    }

    auto expanded_x = executor->AllocTensor(outShapeExpandedX, x->GetDataType(), op::Format::FORMAT_ND);

    op::Shape outShapeExpandedRowIdx;
    outShapeExpandedRowIdx.SetDimNum(0);
    outShapeExpandedRowIdx.AppendDim(expandedRowIdxNum);

    auto expanded_row_idx = executor->AllocTensor(outShapeExpandedRowIdx, op::DataType::DT_INT32, op::Format::FORMAT_ND);

    // Infer expert_tokens_count_or_cumsum shape (optional output)
    op::Shape outShapeExpertTokensCountOrCumsum;
    const aclTensor* expert_tokens_count_or_cumsum = nullptr;

    if (drop_pad_mode == 0 && expert_tokens_count_or_cumsum_flag > 0) {
        outShapeExpertTokensCountOrCumsum.SetDimNum(0);
        outShapeExpertTokensCountOrCumsum.AppendDim(expert_num);
        expert_tokens_count_or_cumsum = executor->AllocTensor(
            outShapeExpertTokensCountOrCumsum, op::DataType::DT_INT32, op::Format::FORMAT_ND);
    }

    // Infer expert_tokens_before_capacity shape (optional output)
    op::Shape outShapeExpertTokensBeforeCapacity;
    const aclTensor* expert_tokens_before_capacity = nullptr;

    if (drop_pad_mode == 1 && expert_tokens_before_capacity_flag) {
        outShapeExpertTokensBeforeCapacity.SetDimNum(0);
        outShapeExpertTokensBeforeCapacity.AppendDim(expert_num);
        expert_tokens_before_capacity = executor->AllocTensor(
            outShapeExpertTokensBeforeCapacity, op::DataType::DT_INT32, op::Format::FORMAT_ND);
    }

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(MoeInitRoutingV2,
                              OP_INPUT(x, expert_idx),
                              OP_OUTPUT(expanded_x, expanded_row_idx, expert_tokens_count_or_cumsum, expert_tokens_before_capacity),
                              OP_ATTR(active_num, expert_capacity, expert_num, drop_pad_mode,
                                     expert_tokens_count_or_cumsum_flag, expert_tokens_before_capacity_flag));
    if (ret != ACL_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "MoeInitRoutingV2AiCore ADD_TO_LAUNCHER_LIST_AICORE failed.");
        return {nullptr, nullptr, nullptr, nullptr};
    }
    return {expanded_x, expanded_row_idx, expert_tokens_count_or_cumsum, expert_tokens_before_capacity};
}
}  // namespace l0op
