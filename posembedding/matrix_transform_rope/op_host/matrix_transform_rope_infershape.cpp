/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to License for details. You may not use this file except in compliance with License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file matrix_transform_rope_infershape.cpp
 * \brief 算子形状推导实现
 */
#include <map>
#include <string>
#include <sstream>
#include <initializer_list>

#include "exe_graph/runtime/infer_shape_context.h"
#include "exe_graph/runtime/shape.h"
#include "exe_graph/runtime/storage_shape.h"
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "../../../common/include/error_ops/error.h"

using namespace gert;
using namespace ge;

namespace ops {

const constexpr int64_t X_INDEX = 0;
const constexpr int64_t COS_INDEX = 1;
const constexpr int64_t SIN_INDEX = 2;
const constexpr int64_t ROTATE_INDEX = 3;
const constexpr int64_t OUT_INDEX = 0;

// 实现形状推导函数
static ge::graphStatus InferShapeMatrixTransformRope(InferShapeContext *context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShape MatrixTransformRope");

    // 获取输入tensor的形状
    auto x_shape = context->GetInputShape(X_INDEX);
    auto cos_shape = context->GetInputShape(COS_INDEX);
    auto sin_shape = context->GetInputShape(SIN_INDEX);
    auto rotate_shape = context->GetInputShape(ROTATE_INDEX);

    // 根据输入形状计算输出形状
    // 输出形状与输入x的形状相同
    auto Output1Shape = context->GetOutputShape(OUT_INDEX);

    // 设置输出的形状
    Output1Shape->SetDimNum(x_shape->GetDimNum());
    for (size_t i = 0; i < x_shape->GetDimNum(); ++i) {
        Output1Shape->SetDim(i, x_shape->GetDim(i));
    }

    OP_LOGD(context->GetNodeName(), "End to do InferShape MatrixTransformRope");
    return GRAPH_SUCCESS;
}

static graphStatus InferDataTypeMatrixTransformRope(gert::InferDataTypeContext *context)
{
    // 输出数据类型与输入x的数据类型相同
    auto xType = context->GetInputDataType(X_INDEX);
    context->SetOutputDataType(OUT_INDEX, xType);

    return GRAPH_SUCCESS;
}

// 注册形状推导实现
IMPL_OP_INFER_SHAPE(MatrixTransformRope)
    .InferShape(InferShapeMatrixTransformRope)
    .InferDataType(InferDataTypeMatrixTransformRope);

} // namespace ops
