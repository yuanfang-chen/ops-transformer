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
 * \file dispatch_ffn_combine_proto.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_FUSION_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_FUSION_OPS_H_

#include "graph/operator_reg.h"
namespace ge {

/**
* @brief DispatchFFNCombine operator interface implementation.

* @par Inputs
* Ten inputs, including:
* @li context: A tensor. Support dtype: int32, dimension must be 1. Shape supports (xx, ), support format: ND.
* @li x: A tensor. Support dtype: float16,bfloat16,float8_e5m2,float8_e4m3,hifloat8,float4_e2m1,
  float4_e1m2, dimension must be 2. Shape supports (BS, H), support format: ND.
* @li expert_ids: A tensor. Support dtype: int32, indicates top k experts of each token, dimension must be 2.
  Shape supports (BS, K), support format: ND.
* @li expert_scales: A tensor. Support dtype: float32,bfloat16, dimension must be 2. Shape supports (BS, K), support format: ND.
* @li weight1: A dynamic input tensor list. Support dtype: float16,bfloat16,float8_e5m2,float8_e4m3,
  dimension must be 3. Shape supports (expertPerRank, H, N), support format: ND.
* @li weight2: A dynamic input tensor list. Support dtype: float16,bfloat16,float8_e5m2,float8_e4m3,
  dimension must be 3. Shape supports (expertPerRank, N/2, N), support format: ND.
* @li scales: An optional tensor. Support dtype: float32,float8_e8m0, dimension must be 2, support format: ND.
* @li x_active_mask: An optional tensor. Support dtype: bool, support format: ND.
* @li weight_scales1: A dynamic input tensor list. Support dtype: float32,float8_e8m0, support format: ND.
* @li weight_scales2: A dynamic input tensor list. Support dtype: float32,float8_e8m0, support format: ND.

* @par Attributes
* @li ep_world_size: Required. Input ep comm world size, dtype: int64.
* @li ep_rank_id: Required. Input ep comm rank Id, dtype: int64.
* @li moe_expert_num: Required. Input moe expert num, dtype: int64.
* @li ccl_buffer_size: Required. Input ccl buffer size, dtype: int64.
* @li max_recv_token_num: Input max recv token num, dtype: int64. Default: 0
* @li shared_expert_num: Input shared expert num, dtype: int64. Default: 1
* @li dispatch_quant_mode: Input dispatch quant mode, dtype: int64. Default: 0
* @li dispatch_quant_out_type: Input dispatch quant out type, dtype: int64. Default: 0
* @li combine_quant_mode: Input combine quant mode, dtype: int64. Default: 0
* @li comm_alg: Input communication algorithm type, dtype: String. Default: ""
* @li global_bs: Input global batch size, dtype: int64. Default: 0

* @par Outputs
* @li y: A tensor. Support dtype: float16,bfloat16. Shape supports (BS, H), support format: ND.
*/
REG_OP(DispatchFFNCombine)
    .INPUT(context, TensorType({DT_INT32}))
    .INPUT(x, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_HIFLOAT8,
                DT_FLOAT4_E2M1, DT_FLOAT4_E1M2}))
    .INPUT(expert_ids, TensorType({DT_INT32}))
    .INPUT(expert_scales, TensorType({DT_FLOAT}))
    .INPUT(weight1, TensorType({DT_BF16, DT_FLOAT16}))
    .INPUT(weight2, TensorType({DT_BF16, DT_FLOAT16}))
    .OPTIONAL_INPUT(scales, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(x_active_mask, TensorType({DT_BOOL}))
    .OPTIONAL_INPUT(weight_scales1, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(weight_scales2, TensorType({DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_BF16, DT_FLOAT16}))
    .REQUIRED_ATTR(ep_world_size, Int)
    .REQUIRED_ATTR(ep_rank_id, Int)
    .REQUIRED_ATTR(moe_expert_num, Int)
    .REQUIRED_ATTR(ccl_buffer_size, Int)
    .ATTR(max_recv_token_num, Int, 0)
    .ATTR(shared_expert_num, Int, 1)
    .ATTR(dispatch_quant_mode, Int, 0)
    .ATTR(dispatch_quant_out_type, Int, 0)
    .ATTR(combine_quant_mode, Int, 0)
    .ATTR(comm_alg, String, "")
    .ATTR(global_bs, Int, 0)
    .OP_END_FACTORY_REG(DispatchFFNCombine)

}  // namespace ge
#endif  // OPS_BUILT_IN_OP_PROTO_INC_FUSION_OPS_H_
