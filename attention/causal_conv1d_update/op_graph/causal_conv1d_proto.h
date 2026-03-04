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
 * \file causal_conv1d_proto.h
 * \brief
 */

#ifndef OPS_BUILT_IN_OP_PROTO_INC_CAUSAL_CONV1D_H_
#define OPS_BUILT_IN_OP_PROTO_INC_CAUSAL_CONV1D_H_

#include "graph/operator_reg.h"

namespace ge {

/**
* @brief Performs causal 1D convolution on sequences. \n
*
* @par Description:
* This operator performs causal 1D convolution along the sequence dimension. It uses cached data
* (length = kernel_width - 1) to pad the head of each sequence, ensuring that the output depends
* only on the current and historical inputs. After convolution, the tail of the current sequence
* (length = kernel_width - 1) is updated to the cache.
*
* Supported scenarios:
* - prefill: x supports CuSeqLen format with shape [cu_seq_len, dim], where cu_seq_len is the total
*            length of all concatenated variable-length sequences in the batch. Filter shape is [K, dim],
*            K is fixed to 3. Before convolution, each sequence is padded with K-1 cached data at the head.
* - decode:  x supports shape [cu_seq_len, dim] or [batch, m+1, dim], where m is the number of speculative
*            tokens. Filter shape is [K, dim], K is fixed to 3. Before convolution, each sequence is
*            padded with K-1 cached data at the head.
*
* Computation formulas:
* - Cache concatenation:
*   x'[i, dim] = cacheState[i, dim],           for 0 <= i < K-1
*   x'[i, dim] = x[i - (K-1), dim],            for K-1 <= i < L + K - 1
*
* - Causal 1D convolution:
*   y[i, dim] = sum_{k=0}^{K-1} w[k, dim] * x'[i + k, dim]
*
* - Cache update:
*   cacheState[i, dim] = x'[L + i, dim],       for i = 0, 1, ..., K-2
*
* where K is kernel width, L is original sequence length, dim is feature dimension.
*
* @par Inputs:
* Inputs including:
* @li x: Input sequence tensor in CuSeqLen layout. A 2D tensor of shape [cu_seq_len, dim] or
*        [batch, m+1, dim] for decode mode. Must be one of the following types: float16, bfloat16.
* @li weight: Causal 1D convolution kernel. A 2D tensor of shape [K, dim], K is fixed to 3.
*             Must be one of the following types: float16, bfloat16.
* @li conv_states: Cache state tensor storing historical convolution states for each sequence.
*                  A 3D tensor of shape [-1, K-1, dim]. Updated in-place after computation.
*                  Must be one of the following types: float16, bfloat16.
* @li query_start_loc: Sequence start position indices. A 1D tensor of shape [batch+1].
*                      Records the start position of each sequence in the concatenated tensor x.
*                      Must be one of the following types: int32.
* @li cache_indices: Cache indices specifying the cache state index for each sequence.
*                    A 1D tensor of shape [batch]. Must be one of the following types: int32.
* @li has_initial_state: Initial state flag indicating whether each sequence uses cached data.
*                        A 1D tensor of shape [batch]. Must be one of the following types: int32.
* @li bias: Optional bias tensor. A 1D tensor of shape [dim].
*           Must be one of the following types: float16, bfloat16.
* @li num_accepted_tokens: Optional tensor for number of accepted tokens in speculative decoding.
*                          Must be one of the following types: int32.

* @par Attributes:
* @li activation_mode: An optional int attribute. Activation function mode. Defaults to 0.
*                      0: None, 1: silu, 2: swish.
* @li pad_slot_id: An optional int attribute. Pad slot ID. Defaults to -1.
* @li run_mode: An optional int attribute. Running mode. Defaults to 0.
*               0: prefill-fn, 1: decode-update.

* @par Outputs:
* @li y: Output sequence tensor. Same shape as x.
*        Must be one of the following types: float16, bfloat16.
* @li conv_states: Updated cache state tensor. Same shape as input conv_states.
*                  Must be one of the following types: float16, bfloat16.

* @attention Constraints:
* @code{.c}
  - The dtype of x, weight, conv_states, bias (if provided), and y must be the same.
  - K (kernel width) is fixed to 3.
  - For prefill mode: x shape is [cu_seq_len, dim]
  - For decode mode: x shape is [cu_seq_len, dim] or [batch, m+1, dim]
  - conv_states shape is [-1, K-1, dim], first dimension must be >= batch
* @endcode
*/
REG_OP(CausalConv1d)
    .INPUT(x, TensorType({DT_BF16, DT_FLOAT16}))
    .INPUT(weight, TensorType({DT_BF16, DT_FLOAT16}))
    .INPUT(conv_states, TensorType({DT_BF16, DT_FLOAT16}))
    .OPTIONAL_INPUT(query_start_loc, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(cache_indices, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(initial_state_mode, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(bias, TensorType({DT_BF16, DT_FLOAT16}))
    .OPTIONAL_INPUT(num_accepted_tokens, TensorType({DT_INT32}))
    .ATTR(activation_mode, Int, 0)
    .ATTR(pad_slot_id, Int, -1)
    .ATTR(run_mode, Int, 0)
    .OUTPUT(y, TensorType({DT_BF16, DT_FLOAT16}))
    .OUTPUT(conv_states, TensorType({DT_BF16, DT_FLOAT16}))
    .OP_END_FACTORY_REG(CausalConv1d)

} // namespace ge

#endif // OPS_BUILT_IN_OP_PROTO_INC_CAUSAL_CONV1D_H_
