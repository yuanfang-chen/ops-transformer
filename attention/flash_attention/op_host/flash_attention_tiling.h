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
 * \file flash_attention_tiling.h
 * \brief FlashAttention Tiling公共头文件
 *
 * TilingData直接复用common下的FlashAttentionScoreSimplifiedTilingData（包含非量化
 * 场景所需的全部字段：InputParamsRegbase/MultiCoreParamsRegbase/InitOutputParams等）。
 */

#ifndef FLASH_ATTENTION_TILING_H_
#define FLASH_ATTENTION_TILING_H_

#include <cstdint>
#include <register/op_impl_registry.h>
// 复用common下的公共TilingData结构体（非量化场景所需字段均已覆盖）
#include "../../common/op_kernel/arch35/flash_attention_score_tiling_regbase.h"
#include "flash_attention_tiling_common.h"

namespace optiling {

// FlashAttention TilingData直接使用common中的公共结构
// 包含：InputParamsRegbase / MultiCoreParamsRegbase / DropmaskParamsRegbase / InitOutputParams
using FlashAttentionTilingData = FlashAttentionScoreSimplifiedTilingData;

ASCENDC_EXTERN_C ge::graphStatus TilingFlashAttention(gert::TilingContext *context);
ASCENDC_EXTERN_C ge::graphStatus TilingPrepareForFlashAttention(gert::TilingParseContext *context);

}  // namespace optiling

#endif  // FLASH_ATTENTION_TILING_H_
