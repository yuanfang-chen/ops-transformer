/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <tuple>
#include "moe_init_routing_v3_mx_quant.h"
#include "opdev/make_op_executor.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "aclnn_kernels/common/op_error_check.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(MoeInitRoutingV3MxQuant);

std::tuple<aclTensor*, aclTensor*, aclTensor*, aclTensor*, aclTensor*> MoeInitRoutingV3MxQuant(
    const aclTensor *x, const aclTensor *expertIdx, const aclTensor *scale,
    const aclTensor *offset, int64_t activeNum, int64_t expertCapacity,
    int64_t expertNum, int64_t dropPadMode, int64_t expertTokensCountOrCumsumFlag,
    bool expertTokensBeforeCapacityFlag, int64_t axis, char *roundMode,
    int64_t dstType, int64_t blocksize, int64_t scaleAlg,
    const aclTensor *y, const aclTensor *mxscale, const aclTensor *expandedRowIdx,
    const aclTensor *expertTokensCountOrCumsum, const aclTensor *expandedScale,
    aclOpExecutor *executor)
{
    L0_DFX(MoeInitRoutingV3MxQuant, x, expertIdx, scale, offset,
            activeNum, expertCapacity, expertNum, dropPadMode,
            expertTokensCountOrCumsumFlag, expertTokensBeforeCapacityFlag,
            axis, roundMode, dstType, blocksize, scaleAlg,
            y, mxscale, expandedRowIdx, expertTokensCountOrCumsum, expandedScale);

    auto yOut = executor->AllocTensor(y->GetViewShape(), y->GetDataType(), Format::FORMAT_ND);
    auto mxscaleOut = executor->AllocTensor(mxscale->GetViewShape(), mxscale->GetDataType(), Format::FORMAT_ND);
    auto expandedRowIdxOut = executor->AllocTensor(expandedRowIdx->GetViewShape(), expandedRowIdx->GetDataType(), Format::FORMAT_ND);

    aclTensor *expertTokensCountOrCumsumOut = nullptr;
    if (expertTokensCountOrCumsum != nullptr) {
        expertTokensCountOrCumsumOut = executor->AllocTensor(expertTokensCountOrCumsum->GetViewShape(),
                                                             expertTokensCountOrCumsum->GetDataType(), Format::FORMAT_ND);
    }

    aclTensor *expandedScaleOut = nullptr;
    if (expandedScale != nullptr) {
        expandedScaleOut = executor->AllocTensor(expandedScale->GetViewShape(),
                                                 expandedScale->GetDataType(), Format::FORMAT_ND);
    }

    if (yOut == nullptr || mxscaleOut == nullptr || expandedRowIdxOut == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR,
                "alloc yOut or mxscaleOut or expandedRowIdxOut tensor failed.");
        return std::tuple<aclTensor*, aclTensor*, aclTensor*, aclTensor*, aclTensor*>(
            nullptr, nullptr, nullptr, nullptr, nullptr);
    }

    ADD_TO_LAUNCHER_LIST_AICORE(
        MoeInitRoutingV3MxQuant,
        OP_INPUT(x, expertIdx, scale, offset),
        OP_OUTPUT(yOut, mxscaleOut, expandedRowIdxOut, expertTokensCountOrCumsumOut, expandedScaleOut),
        OP_ATTR(activeNum, expertCapacity, expertNum, dropPadMode,
                expertTokensCountOrCumsumFlag, expertTokensBeforeCapacityFlag,
                axis, roundMode, dstType, blocksize, scaleAlg));

    return std::tuple<aclTensor*, aclTensor*, aclTensor*, aclTensor*, aclTensor*>(
        yOut, mxscaleOut, expandedRowIdxOut, expertTokensCountOrCumsumOut, expandedScaleOut);
}

}  // namespace l0op
