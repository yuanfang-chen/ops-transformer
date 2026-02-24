/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to License for details. You may not use this file except in compliance with License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_OP_API_COMMON_INC_LEVEL0_OP_MATRIX_TRANSFORM_ROPE_H
#define OP_API_OP_API_COMMON_INC_LEVEL0_OP_MATRIX_TRANSFORM_ROPE_H

#include "opdev/op_executor.h"
#include "opdev/make_op_executor.h"

namespace l0op {
/**
 * @brief MatrixTransformRope 算子函数，返回输出tensor
 * @param [in] x: 输入tensor，4维tensor
 * @param [in] cos: 输入tensor，4维tensor
 * @param [in] sin: 输入tensor，4维tensor
 * @param [in] rotate: 输入tensor，2维tensor，shape为[D, D]
 * @param [in] executor: 执行器
 * @return 输出tensor的tuple
 */
const aclTensor *MatrixTransformRope(
    const aclTensor *x, const aclTensor *cos, const aclTensor *sin, const aclTensor *rotate,
    aclOpExecutor *executor);
}

#endif // OP_API_OP_API_COMMON_INC_LEVEL0_OP_MATRIX_TRANSFORM_ROPE_H
