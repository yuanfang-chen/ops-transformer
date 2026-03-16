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
 * \file chunk_gated_delta_rule_recurrence.cpp
 * \brief Level-0 operator bridge implementation for ChunkGatedDeltaRuleRecurrence.
 *
 * OpDef output ordering:
 *   Output 0: initialState  [b, hv, dv, dk]  — in-place update, NOT allocated here
 *   Output 1: attnInterOut  [hv, nC, cs, dv] — allocated via executor->AllocTensor()
 *   Output 2: vNewOut       [hv, nC, cs, dv] — allocated via executor->AllocTensor()
 *
 * Returns {attnInterOut, vNewOut} as std::array<const aclTensor*, 2>.
 */
#include "chunk_gated_delta_rule_recurrence.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(ChunkGatedDeltaRuleRecurrence);

const std::array<const aclTensor*, 2> ChunkGatedDeltaRuleRecurrence(
    aclTensor       *initialState,
    const aclTensor *kgexp,
    const aclTensor *value,
    const aclTensor *kCumdecay,
    const aclTensor *qgexp,
    const aclTensor *gexp,
    const aclTensor *cuSeqlens,
    float            scaleValue,
    aclOpExecutor   *executor)
{
    L0_DFX(ChunkGatedDeltaRuleRecurrence,
            initialState, kgexp, value, kCumdecay, qgexp, gexp, cuSeqlens, scaleValue);

    Format format = Format::FORMAT_ND;

    // Allocate output tensors (initialState is in-place, not allocated here)
    auto attnInterOut = executor->AllocTensor(DataType::DT_FLOAT, format, format);
    auto vNewOut      = executor->AllocTensor(DataType::DT_FLOAT, format, format);

    OP_CHECK(attnInterOut != nullptr,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "ChunkGatedDeltaRuleRecurrence: attnInterOut AllocTensor failed."),
             return {nullptr, nullptr});
    OP_CHECK(vNewOut != nullptr,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "ChunkGatedDeltaRuleRecurrence: vNewOut AllocTensor failed."),
             return {nullptr, nullptr});

    // InferShape: output ordering must match OpDef (initialState=0, attnInterOut=1, vNewOut=2)
    auto ret = INFER_SHAPE(
        ChunkGatedDeltaRuleRecurrence,
        OP_INPUT(initialState, kgexp, value, kCumdecay, qgexp, gexp, cuSeqlens),
        OP_OUTPUT(initialState, attnInterOut, vNewOut),
        OP_ATTR(scaleValue));
    OP_CHECK_INFERSHAPE(ret != ACLNN_SUCCESS, return {nullptr, nullptr},
                        "ChunkGatedDeltaRuleRecurrence InferShape failed.");

    ret = ADD_TO_LAUNCHER_LIST_AICORE(
        ChunkGatedDeltaRuleRecurrence,
        OP_INPUT(initialState, kgexp, value, kCumdecay, qgexp, gexp, cuSeqlens),
        OP_OUTPUT(initialState, attnInterOut, vNewOut),
        OP_ATTR(scaleValue));
    OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(ret != ACLNN_SUCCESS, return {nullptr, nullptr},
                                         "ChunkGatedDeltaRuleRecurrence ADD_TO_LAUNCHER_LIST_AICORE failed.");

    return {attnInterOut, vNewOut};
}

} // namespace l0op
