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
 * \file add_rms_norm_dynamic_quant_all_gather_qbmm_proto.h
 * \brief 图模式原型定义
 */
#ifndef ADD_RMS_NORM_DYNAMIC_QUANT_ALL_GATHER_QBMM_PROTO_H
#define ADD_RMS_NORM_DYNAMIC_QUANT_ALL_GATHER_QBMM_PROTO_H

#include <graph/operator_reg.h>

namespace ge {
/**
 * @brief Fusion op of quant and reduce scatter.
 * @par Inputs:
 * two inputs, including:
 * @li x1: A matrix Tensor. The type support int8, hifloat8, float8_e4m3fn, float8_e5m2, float4_e1m2, float4_e2m1. The
 * format supports ND.
 * @li scale: A matrix Tensor. The type support float, float8_e8m0. The format supports ND.
 *
 * @par Outputs:
 * out_put: A matrix Tensor. The type support float16, bfloat16, float. The format supports ND.
 *
 * @par Attributes:
 * @li group: A required string identifying the group of ranks participating in the op.
 * @li reduce_op: An optional string identifying the reduction operation to perform. Default: "sum".
 * @li output_dtype: An optional int identifying the data type of output. The type support 0(float), 1(float16),
 * 27(bfloat16). Default: 27(bfloat16).
 * @li world_size: A required int identifying the rank size.
 */
REG_OP(AddRmsNormDynamicQuantAllGatherQbmm)
    .INPUT(x1, TensorType({DT_BF16}))
    .INPUT(x2, TensorType({DT_INT8}))
    .INPUT(residual, TensorType({DT_BF16}))
    .INPUT(y, TensorType({DT_BF16}))
    .INPUT(gamma, TensorType({DT_FLOAT}))
    .INPUT(scale, TensorType({DT_BF16}))
    .INPUT(smooth_scale, TensorType({DT_FLOAT}))
    .INPUT(bias, TensorType({DT_BF16}))
    .OUTPUT(output, TensorType({DT_BF16}))
    .OUTPUT(z, TensorType({DT_BF16}))
    .REQUIRED_ATTR(group, String)
    .ATTR(ranksize, Int, 0)
    .ATTR(transpose_x2, Bool, false)
    .ATTR(dtype, Int, 0)
    .ATTR(residual_norm_mode, Int, 0)
    .OP_END_FACTORY_REG(AddRmsNormDynamicQuantAllGatherQbmm)
} // namespace ge

#endif // ADD_RMS_NORM_DYNAMIC_QUANT_ALL_GATHER_QBMM_PROTO_H
