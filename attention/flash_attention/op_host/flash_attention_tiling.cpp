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
 * \file flash_attention_tiling.cpp
 * \brief FlashAttention Tiling主入口（框架，参照flash_attention_score_tiling.cpp）
 */

#include <cmath>
#include <register/op_impl_registry.h>
#include "log/log.h"
#include "tiling_base/tiling_templates_registry.h"
#include "flash_attention_tiling.h"
#include "flash_attention_tiling_common.h"
#include "../op_kernel/arch35/flash_attention_template_tiling_key.h"
#include "arch35/flash_attention_tiling_regbase.h"

using namespace ge;
using namespace AscendC;
using namespace Ops::Transformer::OpTiling;

namespace optiling {

// 输入索引（与flash_attention_def.cpp对齐）
static constexpr size_t FA_TILING_Q_INDEX          = 0UL;
static constexpr size_t FA_TILING_K_INDEX          = 1UL;
static constexpr size_t FA_TILING_V_INDEX          = 2UL;
static constexpr size_t FA_TILING_ATTN_OUT_INDEX   = 0UL;  // 输出0
static constexpr size_t FA_TILING_SOFTMAX_LSE_IDX  = 1UL;  // 输出1

static constexpr size_t FA_TILING_MIN_COPY_SIZE    = 32UL;
static constexpr size_t FA_TILING_WORKSPACE_RESERVE = 100UL * 1024UL * 1024UL;  // 100MB

struct FAEmptyArgs {
    uint32_t coreNum;
    uint32_t attentionOutFormerNum;
    uint32_t attentionOutTailNum;
    uint32_t softmaxMaxFormerNum;
    uint32_t softmaxMaxTailNum;
    uint64_t attentionOutSingleCoreDataSize;
    uint64_t attentionOutTailCoreDataSize;
    uint64_t softmaxMaxSingleCoreDataSize;
    uint64_t softmaxMaxTailCoreDataSize;
    uint64_t attentionOutLastCoreDataSize = 0ULL;
    uint64_t attentionOutLastCoreIndex    = 0ULL;
    uint64_t attentionOutBlockSize        = 0ULL;
    uint64_t softmaxSumBlockSize          = 0ULL;
    uint32_t aivActualNum                 = 0U;
};

static uint32_t FAEmptyCeil(uint32_t num1, uint32_t num2)
{
    if (num2 == 0U) { return 0U; }
    return (num1 + num2 - 1U) / num2;
}

// 空tensor场景：Q/K/V中有一个为空，但attentionOut或softmaxLse不为空，需要填充0
static bool IsEmptyInput(gert::TilingContext *context)
{
    return false;
}

ASCENDC_EXTERN_C ge::graphStatus TilingFlashAttention(gert::TilingContext *context)
{
    OP_LOGW(context, "FlashAttention TilingFlashAttention start.");

    auto platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_IF(platformInfoPtr == nullptr,
        OP_LOGE(context, "platformInfoPtr is null"),
        return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    if (ascendcPlatform.GetCurNpuArch() == NpuArch::DAV_3510) {
        if (IsEmptyInput(context)) {
            return ge::GRAPH_SUCCESS;
        }
    }

    // 非空tensor场景：通过TilingRegistryArch分发到对应arch的tiling实现类
    return TilingRegistryArch::GetInstance().DoTilingImpl(context);
}

ASCENDC_EXTERN_C ge::graphStatus TilingPrepareForFlashAttention(gert::TilingParseContext *context)
{
    auto platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_IF(platformInfoPtr == nullptr,
        OP_LOGE(context, "platformInfoPtr is null"),
        return ge::GRAPH_FAILED);
    auto compileInfoPtr = context->GetCompiledInfo<FlashAttentionCompileInfo>();
    OP_CHECK_IF(compileInfoPtr == nullptr,
        OP_LOGE(context, "compileInfoPtr is null"),
        return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->aivNum     = ascendcPlatform.GetCoreNumAiv();
    compileInfoPtr->aicNum     = ascendcPlatform.GetCoreNumAic();
    compileInfoPtr->socVersion = ascendcPlatform.GetSocVersion();
    compileInfoPtr->npuArch    = ascendcPlatform.GetCurNpuArch();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB,   compileInfoPtr->ubSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1,   compileInfoPtr->l1Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, compileInfoPtr->l0cSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L2,   compileInfoPtr->l2CacheSize);

    return ge::GRAPH_SUCCESS;
}

// 注册tiling函数：
//   TilingInputsDataDependency：cuSeqlensQ(4)/cuSeqlensKv(5)/sequsedQ(6)/sequsedKv(7)/metadata(9)在tiling时需要数据
IMPL_OP_OPTILING(FlashAttention)
    .Tiling(TilingFlashAttention)
    .TilingInputsDataDependency({4, 5, 6, 7, 9})
    .TilingParse<FlashAttentionCompileInfo>(TilingPrepareForFlashAttention);

} // namespace optiling
