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
 * \file mhc_post_infershape.cpp
 * \brief
 */

#include "mhc_post_infershape.h"
#include "graph/operator_reg.h"
#include "common/util/error_manager/error_manager.h"

namespace ge {
IMPLEMT_INFERFUNC(MhcPost, MhcPostInferShape) {
    // Output shape is same as input x
    TensorDesc outputTensorDesc = op.GetOutputDescByName("y");
    TensorDesc xTensorDesc = op.GetInputDescByName("x");

    DataType xDtype = xTensorDesc.GetDataType();
    Format xFormat = xTensorDesc.GetFormat();
    Shape xShape = xTensorDesc.GetShape();

    outputTensorDesc.SetDataType(xDtype);
    outputTensorDesc.SetFormat(xFormat);
    outputTensorDesc.SetShape(xShape);

    (void)op.UpdateOutputDesc("y", outputTensorDesc);
    return GRAPH_SUCCESS;
}

INFER_FUNC_REG(MhcPost, MhcPostInferShape);
}