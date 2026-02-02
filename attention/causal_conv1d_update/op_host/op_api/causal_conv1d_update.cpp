/**
?* This program is free software, you can redistribute it and/or modify.
?* Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "causal_conv1d_update.h"

#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(CausalConv1dUpdate);

const aclTensor* CausalConv1dUpdate(const aclTensor* x,
                                    const aclTensor* weight,
                                    aclTensor* convState,
                                    const aclTensor* convStateIndicesOptional,
                                    const aclTensor* biasOptional,
                                    const aclTensor* numAcceptedTokensOptional,
                                    const aclTensor* queryStartLocOptional,
                                    int64_t activationMode,
                                    int64_t padSlotId,
                                    aclOpExecutor* executor)
{
    L0_DFX(CausalConv1dUpdate, x, weight, convState, convStateIndicesOptional, biasOptional, numAcceptedTokensOptional, queryStartLocOptional, activationMode, padSlotId);

    if (biasOptional == nullptr) {
        biasOptional = executor->AllocTensor(x->GetDataType(), Format::FORMAT_ND, Format::FORMAT_ND);
    }

    auto y = executor->AllocTensor(x->GetDataType(), Format::FORMAT_ND, Format::FORMAT_ND);

    auto ret = INFER_SHAPE(CausalConv1dUpdate,
                           OP_INPUT(x, weight, convState, convStateIndicesOptional, biasOptional, numAcceptedTokensOptional, queryStartLocOptional),
                           OP_OUTPUT(y),
                           OP_ATTR(activationMode, padSlotId));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "CausalConv1dUpdate InferShape failed.");
        return nullptr;
    }

    ADD_TO_LAUNCHER_LIST_AICORE(CausalConv1dUpdate, OP_INPUT(x, weight, convState, convStateIndicesOptional, biasOptional, numAcceptedTokensOptional, queryStartLocOptional),
                               OP_OUTPUT(y), OP_ATTR(activationMode, padSlotId));
    return y;
}

} // namespace l0op

