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
 * \file chunk_gated_delta_rule_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_CHUNKGATEDDELTARULE_H_
#define OPS_OP_PROTO_INC_CHUNKGATEDDELTARULE_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
*@brief chunked GDN.
*@par Inputs:
* seven inputs, including:
* @li query: A matrix Tensor. The type support bf16.
* @li key: A matrix Tensor. The type support bf16.
* @li value: A matrix Tensor. The type support bf16.
* @li beta: A matrix Tensor. The type support bf16.
* @li initial_state: A matrix Tensor. The type support bf16.
* @li actual_seq_lengths: A matrix Tensor. The type support int32.
* @li g: A matrix Tensor. An optional input parameter. The type support float32.
* layout need to be setted TND. ex. If the attn_out seqlen is [2,2,2,2,2], this parameter need be setted [2,4,6,8,10].

* @par Attributes:
* @li scale_value: A float32. A optional attribute.

* @par Outputs:
* @li out: A matrix Tensor. The type support bf16.
* @li final_state: A matrix Tensor. The type support bf16.
*/
REG_OP(ChunkGatedDeltaRule)
    .INPUT(query, TensorType({DT_BF16, DT_BF16}))
    .INPUT(key, TensorType({DT_BF16, DT_BF16}))
    .INPUT(value, TensorType({DT_BF16, DT_BF16}))
    .INPUT(beta, TensorType({DT_BF16}))
    .INPUT(initial_state, TensorType({DT_BF16, DT_BF16, DT_BF16}))
    .INPUT(actual_seq_lengths, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(g, TensorType({DT_FP32}))
    .OPTIONAL_ATTR(scale_value, DT_FP32)
    .OUTPUT(out, TensorType({DT_BF16, DT_BF16}))
    .OUTPUT(final_state, TensorType({DT_BF16, DT_BF16, DT_BF16}))
    .OP_END_FACTORY_REG(ChunkGatedDeltaRule)

} // namespace ge

#endif // OPS_OP_PROTO_INC_CHUNKGATEDDELTARULE_H_
