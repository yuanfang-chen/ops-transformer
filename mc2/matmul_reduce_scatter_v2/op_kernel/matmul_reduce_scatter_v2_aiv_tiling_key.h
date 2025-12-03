/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file matmul_reduce_scatter_v2_aiv_tiling_key.h
 * \brief
 */

#ifndef MATMUL_REDUCE_SCATTER_V2_AIV_TILING_KEY_H
#define MATMUL_REDUCE_SCATTER_V2_AIV_TILING_KEY_H

#include "ascendc/host_api/tiling/template_argument.h"

namespace MatmulReduceScatterv2TilingKey{
// 模板参数
ASCENDC_TPL_ARGS_DECL(
    MatmulReduceScatterV2, // 算子OpType
    ASCENDC_TPL_BOOL_DECL(IS_BIAS, 0, 1),
    ASCENDC_TPL_BOOL_DECL(IS_TRANSPOSE_A, 0, 1),
    ASCENDC_TPL_BOOL_DECL(IS_TRANSPOSE_B, 0, 1)
);

// 模板参数组合
// 用于调用GET_TPL_TILING_KEY获取TilingKey时，接口内部校验TilingKey是否合法
ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_BOOL_SEL(IS_BIAS, 0),
        ASCENDC_TPL_BOOL_SEL(IS_TRANSPOSE_A, 0),
        ASCENDC_TPL_BOOL_SEL(IS_TRANSPOSE_B, 0)),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_BOOL_SEL(IS_BIAS, 0),
        ASCENDC_TPL_BOOL_SEL(IS_TRANSPOSE_A, 0),
        ASCENDC_TPL_BOOL_SEL(IS_TRANSPOSE_B, 1))
);
} // MatmulReduceScatterv2TilingKey
#endif // MATMUL_REDUCE_SCATTER_V2_AIV_TILING_KEY_H