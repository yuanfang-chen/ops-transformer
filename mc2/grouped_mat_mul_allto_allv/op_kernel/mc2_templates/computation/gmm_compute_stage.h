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
 * \file gmm_compute_stage.h
 * \brief
 */

#ifndef GMM_COMPUTE_STAGE_H
#define GMM_COMPUTE_STAGE_H

#include "../../arch35/3rd_head.h"

namespace ATAVKernelTemplate {

// 使用gmmv4算子的gqmm_cube_on_the_fly.h方法作为计算节点的计算实现,后续是否转置的参数通过算子的模板参数获取

using ComputationType = GmmExpertOp<GmmASWKernel<X_T, W_T, BIAS_T, SCALE_T, Y_T, W_FORMAT, A_TRANS, B_TRANS>,QuantExtraData, TilingType>

};

#endif // GMM_COMPUTE_STAGE_H