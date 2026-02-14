/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to License for details. You may not use this file except in compliance with License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file matrix_transform_rope.h
 * \brief MatrixTransformRope 算子 kernel 实现
 */

#ifndef __OP_KERNEL_MATRIX_TRANSFORM_ROPE_H__
#define __OP_KERNEL_MATRIX_TRANSFORM_ROPE_H__

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"

namespace MatrixTransformRope {

using namespace AscendC;
using namespace matmul;

// TODO: 根据算子具体实现定义必要的结构体和常量

extern "C" {
__global__ void matrix_transform_rope(GM_ADDR x, GM_ADDR cos, GM_ADDR sin, GM_ADDR rotate,
                                          GM_ADDR out, GM_ADDR workspaceGM, GM_ADDR tilingGM);
} // namespace MatrixTransformRope

#endif // __OP_KERNEL_MATRIX_TRANSFORM_ROPE_H__
