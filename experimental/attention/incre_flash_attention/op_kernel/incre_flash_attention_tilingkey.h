#pragma once
#include "ascendc/host_api/tiling/template_argument.h"

#define ASCENDC_TPL_2_BW 2  // 每个参数占用2个bit位

ASCENDC_TPL_ARGS_DECL(IncreFlashAttention,
    ASCENDC_TPL_UINT_DECL(FLASH_DECODE, ASCENDC_TPL_2_BW, ASCENDC_TPL_UI_LIST, 0, 1),
    // BSH, BNSD, TND
    ASCENDC_TPL_UINT_DECL(LAYOUT_T, ASCENDC_TPL_2_BW, ASCENDC_TPL_UI_LIST, 0, 1, 3),
    // 伪量化: K V per-channel
    // 伪量化: K V per-token
    // 伪量化: K per-channel and V per-token
    ASCENDC_TPL_UINT_DECL(AntiquantMode, ASCENDC_TPL_2_BW, ASCENDC_TPL_UI_LIST, 0, 1, 2),
);

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(FLASH_DECODE, ASCENDC_TPL_UI_LIST, 0, 1),
        ASCENDC_TPL_UINT_SEL(LAYOUT_T, ASCENDC_TPL_UI_LIST, 0, 1),
        ASCENDC_TPL_UINT_SEL(AntiquantMode, ASCENDC_TPL_UI_LIST, 0, 1),
        ASCENDC_TPL_TILING_STRUCT_SEL(optiling::IncreFlashAttentionTilingDataV2),
    ),
);