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
 * @brief Fusion op of AddRmsNormDynamicQuant, AllGather and QuantBatchMatmul.
 * @par Inputs:
 * eight inputs, including:
 * @li x1: A matrix Tensor. The type support float16, bfloat16. The format supports ND.
 * @li x2: A matrix Tensor. The type support int8. The format supports ND and FRACTAL_NZ.
 * @li residual: A matrix Tensor. The type support float16, bfloat16. The format supports ND.
 * @li y: A matrix Tensor. The type support float16, bfloat16. The format supports ND.
 * @li gamma": A Tensor. The type support float. The format supports ND.
 * @li scale: An optional Tensor. The type support float, bfloat16. The format supports ND.
 * @li smooth_scale: An optional Tensor. The type support float. The format supports ND.
 * @li bias: An optional Tensor. The type support int32. The format supports ND.
 *
 * @par Outputs:
 * @li output: A matrix Tensor. The type support float16, bfloat16. The format supports ND.
 * @li z: A matrix Tensor. The type support float16, bfloat16. The format supports ND.
 *
 * @par Attributes:
 * @li group: A required string identifying the group of ranks participating in the op.
 * @li ranksize: A required int identifying the rank size. Default: 0.
 * @li transpose_x2: An optional bool identifying the transpose of x2. Default: "false".
 * @li dtype: An optional int identifying the data type of output. The type support 0(float), 1(float16),
 * 27(bfloat16). Default: 27(bfloat16).
 * @li residual_norm_mode: A required int identifying the norm mode. Default: 0.
 */
REG_OP(AddRmsNormDynamicQuantAllGatherQbmm)
    .INPUT(x1, TensorType({DT_BF16, DT_FLOAT16}))
    .INPUT(x2, TensorType({DT_INT8}))
    .INPUT(residual, TensorType({DT_BF16, DT_FLOAT16}))
    .INPUT(y, TensorType({DT_BF16, DT_FLOAT16}))
    .INPUT(gamma, TensorType({DT_FLOAT}))
    .INPUT(scale, TensorType({DT_BF16, DT_FLOAT}))
    .OPTIONAL_INPUT(smooth_scale, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(bias, TensorType({DT_INT32}))
    .OUTPUT(output, TensorType({DT_BF16, DT_FLOAT}))
    .OUTPUT(z, TensorType({DT_BF16, DT_FLOAT}))
    .OUTPUT(addRmsNormOut, TensorType({DT_BF16, DT_FLOAT}))
    .OUTPUT(dynamicQuantOut, TensorType({DT_INT8}))
    .OUTPUT(allGatherDataOut, TensorType({DT_INT8}))
    .OUTPUT(allGatherScalesOut, TensorType({DT_FLOAT}))
    .REQUIRED_ATTR(group, String)
    .ATTR(ranksize, Int, 0)
    .ATTR(transpose_x2, Bool, false)
    .ATTR(dtype, Int, 0)
    .ATTR(residual_norm_mode, Int, 0)
    .OP_END_FACTORY_REG(AddRmsNormDynamicQuantAllGatherQbmm)
} // namespace ge

#endif // ADD_RMS_NORM_DYNAMIC_QUANT_ALL_GATHER_QBMM_PROTO_H
