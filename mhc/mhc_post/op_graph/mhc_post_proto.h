

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
 * \file mhc_post_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_MOECOMPUTEEXPERT_H_
#define OPS_OP_PROTO_INC_MOECOMPUTEEXPERT_H_

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Fuse the main branch feature h_out with the residual branch feature x 
* using the gating mechanism h_post and the doubly stochastic matrix  h_res 
* to enable information flow.
* @par Inputs:
* @li x: A Tensor. Type is:BFloat16 or Float16.
* @li h_res: A Tensor. Type is:Float32.
* @li h_out: A Tensor. Type is:BFloat16 or Float16.
* @li h_post: A Tensor. Type is:Float32.
* @par Outputs:
* @li y: A Tensor. Type is:BFloat16 or Float16.
*/
REG_OP(MhcPost)
    .INPUT(x, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(h_res, TensorType({DT_FLOAT}))
    .INPUT(h_out, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(h_post, TensorType({DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_BF16}))
    .OP_END_FACTORY_REG(MhcPost)

} // namespace ge

#endif // OPS_OP_PROTO_INC_MOECOMPUTEEXPERT_H_
