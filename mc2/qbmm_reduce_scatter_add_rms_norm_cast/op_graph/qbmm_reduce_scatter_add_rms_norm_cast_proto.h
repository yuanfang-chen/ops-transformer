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
 * \file fusion_ops.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_FUSION_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_FUSION_OPS_H_

#include "graph/operator_reg.h"
namespace ge {

/**
* @brief MoeDistributeDispatchV2 operator interface implementation.

* @par Inputs
* Five inputs, including:
* @li x: A tensor. Support dtype: float16,bfloat16, dimension must be 2. Shape supports (BS, H), support format: ND.
* @li expertIds: A tensor. Support dtype: int32, indicates top k experts of each token, dimension must be 2. Shape supports (BS, K), support format: ND.
* @li scales: An optional tensor. Support dtype: float32, dimension must be 2, support format: ND.
* @li x_active_mask: An optional tensor. Support dtype: bool, support format: ND.
* @li expert_scales: An optional tensor. Support dtype: float32. Shape supports (BS, K), support format: ND.
* @li performance_info: An optional tensor. Support dtype: int64, support format: ND.

* @par Attributes
* @li group_ep: Required. Input ep comm group name, ep means experts parallelism, dtype: String.
* @li ep_world_size: Required. Input ep comm world size, dtype: int64.
* @li ep_rank_id: Required. Input ep comm rank Id, dtype: int64.
* @li moe_expert_num: Required. Input moe expert num, dtype: int64.
* @li group_tp: Input tp comm group name, tp means tensor parallelism, dtype: String.
* @li tp_world_size: Input tp comm world size, dtype: int64.
* @li tp_rank_id: Input tp comm rank Id, dtype: int64.
* @li expert_shard_type: Input moe shard type, dtype: int64.
* @li shared_expert_num: Input shared expert num, dtype: int64.
* @li shared_expert_rank_num: Input shared expert rank num, dtype: int64.
* @li quant_mode: Input quant mode. The options are 0 (non-quantization), 1 (static quantization), and 2 (dynamic quantization). dtype: int64.
* @li global_bs: Input global batch size, dtype: int64.
* @li expert_token_nums_type: Input expert token nums type, dtype: int64.
* @li comm_alg: Input comm alg type, dtype: String.

* @par Outputs
* Seven outputs, including:
* @li expand_x: A tensor. Result of each expert after dispatching. Support dtype: float16,bfloat16,int8,float8_e4m3,float8_e5m2，hifloat8. Shape supports (A, H), support format: ND.
* @li dynamic_scales: If quant is enabled, scale value of each token. A tensor. Support dtype: float32,float8_e8m0. Shape supports (A, ), support format: ND.
* @li assist_info_for_combine: A tensor. Support dtype: int32. Shape supports (A * 128), support format: ND.
* @li expert_token_nums: A tensor. Tokens nums of expand_x. Support dtype: int64, support format: ND.
* @li ep_recv_count: A tensor. Received token nums after dispatching. Support dtype: int32, support format: ND.
* @li tp_recv_count: A tensor. Received token nums after allgather. Support dtype: int32, support format: ND.
* @li expand_scales: A tensor. Scales of each token to sum for combine. Support dtype: float32. Shape supports (A, ), support format: ND.
*/
REG_OP(QbmmReduceScatterAddRmsNormCast)
    .INPUT(x1, TensorType({DT_INT8}))
    .INPUT(x2, TensorType({DT_INT8}))
    .INPUT(y, TensorType({DT_BF16, DT_FLOAT16}))
    .INPUT(gamma, TensorType({DT_FLOAT}))
    .INPUT(scale, TensorType({DT_BF16, DT_FLOAT}))
    .OPTIONAL_INPUT(bias, TensorType({DT_BF16, DT_INT32, DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(pertoken_scale, TensorType({DT_BF16, DT_FLOAT}))
    .OUTPUT(y1, TensorType({DT_FLOAT}))
    .OUTPUT(y2, TensorType({DT_BF16, DT_FLOAT16}))
    .OUTPUT(x, TensorType({DT_BF16, DT_FLOAT16}))
    .REQUIRED_ATTR(group, String)
    .ATTR(rank_size, Int, 0)
    .ATTR(transpose_x2, Bool, false)
    .ATTR(dtype, Int, 0)
    .ATTR(epsilon, Float, 1e-6f)
    .OP_END_FACTORY_REG(QbmmReduceScatterAddRmsNormCast)

}  // namespace ge
#endif  // OPS_BUILT_IN_OP_PROTO_INC_FUSION_OPS_H_
