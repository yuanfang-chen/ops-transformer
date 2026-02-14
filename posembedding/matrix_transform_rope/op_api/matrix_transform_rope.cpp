/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to License for details. You may not use this file except in compliance with License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "matrix_transform_rope.h"

namespace l0op {

const aclTensor *MatrixTransformRope(
    const aclTensor *x, const aclTensor *cos, const aclTensor *sin, const aclTensor *rotate,
    aclOpExecutor *executor) {

    // 实现算子逻辑
    L0_DFX(MatrixTransformRope, x, cos, sin, rotate);

    // 输出类型和格式与x保持一致
    DataType outType = x->GetDataType();
    Format format = Format::FORMAT_ND;

    // 分配输出tensor
    auto out = executor->AllocTensor(outType, format, format);

    auto ret = INFER_SHAPE(MatrixTransformRope,
        OP_INPUT(x, cos, sin, rotate),
        OP_OUTPUT(out));
    OP_CHECK_INFERSHAPE(ret != ACLNN_SUCCESS, return nullptr,
        "MatrixTransformRope InferShape failed.");

    auto ret1 = ADD_TO_LAUNCHER_LIST_AICORE(MatrixTransformRope,
        OP_INPUT(x, cos, sin, rotate),
        OP_OUTPUT(out));
    OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(ret1 != ACLNN_SUCCESS, return nullptr,
        "MatrixTransformRope ADD_TO_LAUNCHER_LIST_AICORE failed.");

    return out;
}

} // namespace l0op
