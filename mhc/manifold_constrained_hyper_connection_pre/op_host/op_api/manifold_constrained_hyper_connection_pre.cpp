/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "manifold_constrained_hyper_connection_pre.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(ManifoldConstrainedHyperConnectionPre);

const std::tuple<aclTensor *, aclTensor *, aclTensor *, aclTensor *, aclTensor *, aclTensor *>
ManifoldConstrainedHyperConnectionPre(
    const aclTensor *x, const aclTensor *phi, const aclTensor *alpha, const aclTensor *bias, const aclTensor *gamma,
    int64_t out_flag, double norm_eps, double hc_eps, aclOpExecutor *executor)
{
    L0_DFX(ManifoldConstrainedHyperConnectionPre, x, phi, alpha, bias, gamma, out_flag, norm_eps, hc_eps);

    DataType outType = DataType::DT_FLOAT; // 输出类型
    Format format = Format::FORMAT_ND; // 输出分形
    auto outHin = executor->AllocTensor(x->GetDataType(), format, format);
    auto outHpost = executor->AllocTensor(outType, format, format);
    auto outHres = executor->AllocTensor(outType, format, format);

    auto outInvRms = executor->AllocTensor(outType, format, format);
    auto outMmRes = executor->AllocTensor(outType, format, format);
    auto outHpre = executor->AllocTensor(outType, format, format);

    auto ret = INFER_SHAPE(ManifoldConstrainedHyperConnectionPre, OP_INPUT(x, phi, alpha, bias, gamma),
        OP_OUTPUT(outHin, outHpost, outHres, outInvRms, outMmRes, outHpre), OP_ATTR(out_flag, norm_eps, hc_eps));
    OP_CHECK_INFERSHAPE(ret != ACLNN_SUCCESS, return std::tuple(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr), "ManifoldConstrainedHyperConnectionPre InferShape failed.");
    auto ret1 = ADD_TO_LAUNCHER_LIST_AICORE(ManifoldConstrainedHyperConnectionPre, OP_INPUT(x, phi, alpha, bias, gamma),
        OP_OUTPUT(outHin, outHpost, outHres, outInvRms, outMmRes, outHpre),  OP_ATTR(out_flag, norm_eps, hc_eps));
    OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(ret1 != ACLNN_SUCCESS, return std::tuple(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr),
        "ManifoldConstrainedHyperConnectionPre ADD_TO_LAUNCHER_LIST_AICORE failed.");
        
    return std::tuple(outHin, outHpost, outHres, outInvRms, outMmRes, outHpre);
}
}