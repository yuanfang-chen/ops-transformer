/**
 * Copyright (c) Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in root of software repository for full text of the License.
 */

/*!
 * \file spatial_transformer_proto.h
 * \brief
 */
#ifndef OPS_NN_INDEX_SPATIAL_TRANSFORMER_PROTO_H_
#define OPS_NN_INDEX_SPATIAL_TRANSFORMER_PROTO_H_

#include "graph/operator_reg.h"
#include "graph/operator.h"

namespace ge {

/**
 *@brief Spatial Transformer Network (STN) operator for affine transformation. \n
 *
 *@par Inputs:
 * Including:
 * @li x: A Tensor. Must be one of following types: float, float16, double, uint8, int8,
 *            uint16, int16, int32, uint32, uint64, int64.
 *            4-D tensor with shape [batch, channels, height, width].
 *            The format must be NCHW or NC1HWC0.
 * @li theta: A Tensor. Must be one of following types: float, float16, double, uint8, int8,
 *              uint16, int16, int32, uint32, uint64, int64.
 *              3-D tensor with shape [batch, 2, 3] or [2, 3] containing
 *              affine transformation parameters. \n
 *
 *@par Outputs:
 * @li y: A Tensor. Must be one of following types: float, float16, double, uint8, int8,
 *         uint16, int16, int32, uint32, uint64, int64.
 *         4-D tensor with shape [batch, channels, height, width].
 *         The format must be NCHW or NC1HWC0. \n
 *
 *@par Attributes:
 * @li output_size: A list of 2 integers. Defaults to [-1, -1].
 *              Specifies the output height and width. If -1, uses input size.
 * @li default_theta: A list of 6 floats. Defaults to [].
 *              Default affine transformation parameters when use_default_theta is true.
 * @li align_corners: A bool. Defaults to false.
 *              If true, centers of the 4 corner pixels of the input and output
 *              tensors are aligned, preserving the values at the corner pixels.
 * @li use_default_theta: A list of 6 integers. Defaults to [].
 *              Specifies which theta parameters to use from default_theta.
 *              1 means use default, 0 means use input theta. \n
 *
 *@par Third-party framework compatibility
 * Compatible with TensorFlow SpatialTransformer operator.
 */
REG_OP(SpatialTransformer)
    .INPUT(x, TensorType({DT_FLOAT,DT_FLOAT16,DT_DOUBLE,DT_UINT8,DT_INT8,DT_UINT16,
                          DT_INT16,DT_INT32,DT_UINT32,DT_UINT64,DT_INT64}))
    .OPTIONAL_INPUT(theta, TensorType({DT_FLOAT,DT_FLOAT16,DT_DOUBLE,DT_UINT8,DT_INT8,
                                       DT_UINT16,DT_INT16,DT_INT32,DT_UINT32,DT_UINT64,DT_INT64}))
    .OUTPUT(y, TensorType({DT_FLOAT,DT_FLOAT16,DT_DOUBLE,DT_UINT8,DT_INT8,DT_UINT16,
                           DT_INT16,DT_INT32,DT_UINT32,DT_UINT64,DT_INT64}))
    .ATTR(output_size, ListInt, {-1, -1})
    .ATTR(default_theta, ListFloat, {})
    .ATTR(align_corners, Bool, false)
    .ATTR(use_default_theta, ListInt, {})
    .OP_END_FACTORY_REG(SpatialTransformer)

} // namespace ge

#endif // OPS_NN_INDEX_SPATIAL_TRANSFORMER_PROTO_H_