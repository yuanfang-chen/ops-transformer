/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file sparse_attn_sharedkv_tiling.h
 * \brief
 */
#ifndef SPARSE_ATTN_SHAREDKV_TILING_H
#define SPARSE_ATTN_SHAREDKV_TILING_H

#include <graph/utils/type_utils.h>
#include <exe_graph/runtime/tiling_context.h>
#include <tiling/platform/platform_ascendc.h>
#include "register/tilingdata_base.h"
#include "exe_graph/runtime/tiling_context.h"

namespace optiling {

// -----------算子TilingData定义---------------
BEGIN_TILING_DATA_DEF(SparseAttnSharedkvTilingData)
TILING_DATA_FIELD_DEF(uint32_t, batchSize)
END_TILING_DATA_DEF

REGISTER_TILING_DATA_CLASS(SparseAttnSharedkv, SparseAttnSharedkvTilingData)

// tiling class
}
#endif