/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef PTA_NPU_OP_API_COMMON_INC_LEVEL0_OP_CHUNK_GATED_DELTA_RULE_RECURRENCE
#define PTA_NPU_OP_API_COMMON_INC_LEVEL0_OP_CHUNK_GATED_DELTA_RULE_RECURRENCE

#include <array>
#include "opdev/op_executor.h"
#include "opdev/make_op_executor.h"

namespace l0op {

/**
 * @brief Level-0 operator bridge for ChunkGatedDeltaRuleRecurrence.
 *
 * initialState is updated in-place (in OP_OUTPUT but not allocated here).
 * attnInterOut and vNewOut are allocated inside this function via executor->AllocTensor().
 *
 * @return std::array<const aclTensor*, 2>{attnInterOut, vNewOut}
 *         Returns {nullptr, nullptr} on error.
 */
const std::array<const aclTensor*, 2> ChunkGatedDeltaRuleRecurrence(
    aclTensor       *initialState,
    const aclTensor *kgexp,
    const aclTensor *value,
    const aclTensor *kCumdecay,
    const aclTensor *qgexp,
    const aclTensor *gexp,
    const aclTensor *cuSeqlens,
    float            scaleValue,
    aclOpExecutor   *executor);

} // namespace l0op

#endif // PTA_NPU_OP_API_COMMON_INC_LEVEL0_OP_CHUNK_GATED_DELTA_RULE_RECURRENCE
