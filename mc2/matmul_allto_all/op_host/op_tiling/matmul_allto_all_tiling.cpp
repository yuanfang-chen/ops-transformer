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
 * \file matmul_allto_all_tiling.cpp
 * \brief host侧tiling实现
 */
#include <register/op_def_registry.h>
#include "../../op_kernel/arch35/matmul_allto_all_tiling_data.h"
#include "../../op_kernel/arch35/matmul_allto_all_tiling_key.h"
#include "mc2_log.h"

using namespace AscendC;
using namespace ge;

namespace optiling {

static ge::graphStatus MatmulAlltoAllTilingFunc(gert::TilingContext *context)
{
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForMatmulAlltoAll(gert::TilingParseContext *context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MatmulAlltoAll)
    .Tiling(MatmulAlltoAllTilingFunc);
} // namespace optiling