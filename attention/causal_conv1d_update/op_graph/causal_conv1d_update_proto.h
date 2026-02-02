/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file causal_conv1d_update_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_CAUSAL_CONV1D_UPDATE_OPS_H_
#define OPS_OP_PROTO_INC_CAUSAL_CONV1D_UPDATE_OPS_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {
/**
* @brief Depthwise causal 1D convolution (decode/update) for Mamba/Mamba2.
* @par Inputs:
* @li x: A 2D/3D Tensor (batch, dim) or (batch, dim, seqlen). Must be one of the following types: float16, bfloat16, float32.
* @li weight: A 2D Tensor (dim, width). Must be one of the following types: float16, bfloat16, float32.
* @li bias: An optional 1D Tensor (dim,). Must be one of the following types: float16, bfloat16, float32.
* @li conv_state: A 3D Tensor (num_cache_lines, dim, state_len). Must be one of the following types: float16, bfloat16, float32.
* @li conv_state_indices: A 1D Tensor (batch,). Must be int32.
* @par Attributes:
* @li activation_mode: An int. 0: none, 1: silu/swish. Default: 0.
* @li pad_slot_id: An int. Padding slot id for continuous batching. Default: -1.
* @par Outputs:
* @li y: A Tensor with the same shape as x. Must be one of the following types: float16, bfloat16, float32.
*/
REG_OP(CausalConv1dUpdate)
    .INPUT(x, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(weight, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(conv_state, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(conv_state_indices, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(num_accpted_tokens, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(query_start_loc, TensorType({DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_BF16}))
    .ATTR(activation_mode, Int, 0)
    .ATTR(pad_slot_id, Int, -1)
    .OP_END_FACTORY_REG(CausalConv1dUpdate)

} // namespace ge

#endif // OPS_OP_PROTO_INC_CAUSAL_CONV1D_OPS_H_
