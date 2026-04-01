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
 * \file moe_init_routing_v3_mx_quant_apt.cpp
 * \brief MoeInitRoutingV3MxQuant kernel entry (regbase mode)
 */

#include "arch35/moe_init_routing_v3_mx_quant_common.h"
#include "arch35/moe_init_routing_v3_mx_quant_sort_one_core.h"
#include "arch35/moe_init_routing_v3_mx_quant_sort_multi_core.h"
#include "arch35/moe_init_routing_v3_mx_quant_expert_tokens_count.h"
#include "arch35/moe_init_routing_v3_mx_quant_row_idx_gather.h"
#include "arch35/moe_init_routing_v3_mx_quant_gather_mxfp8_quant.h"

/*
 * MXFP8 TilingKey definitions
 * Encoding: 1 | sort_mode | quant_mode | row_idx_type | 000
 *   sort_mode: 0=single-core, 1=multi-core
 *   quant_mode: 3=MXFP8
 *   row_idx_type: 0=GATHER, 1=SCATTER
 */
#define MOE_V3_MX_SORTONECORE_GATHER   1030000
#define MOE_V3_MX_SORTONECORE_SCATTER  1031000
#define MOE_V3_MX_SORTMULTICORE_GATHER  1130000
#define MOE_V3_MX_SORTMULTICORE_SCATTER 1131000

using namespace AscendC;
using namespace MoeInitRoutingV3MxQuantNs;

extern "C" __global__ __aicore__ void moe_init_routing_v3_mx_quant(GM_ADDR x, GM_ADDR expertIdx, GM_ADDR scale,
                                                                    GM_ADDR offset, GM_ADDR y, GM_ADDR mxscale,
                                                                    GM_ADDR expandedRowIdx,
                                                                    GM_ADDR expertTokensCountOrCumsum,
                                                                    GM_ADDR expandedScale, GM_ADDR workspace,
                                                                    GM_ADDR tiling)
{
    if (g_coreType == AIC) {
        return;
    }

    if (workspace == nullptr) {
        return;
    }

    REGISTER_TILING_DEFAULT(MoeInitRoutingV3MxQuantArch35TilingData);
    GET_TILING_DATA_WITH_STRUCT(MoeInitRoutingV3MxQuantArch35TilingData, tilingData, tiling);

    GM_ADDR userWS = GetUserWorkspace(workspace);
    if (userWS == nullptr) {
        return;
    }

#if (__NPU_ARCH__ == 3101)
    int64_t oriOverflowMode = GetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>();
    // Disable saturation mode so overflow wraps rather than saturates during MX-FP8 quantization
    SetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>(0);
#endif

    auto t = &tilingData;

    // Stage 1: Sort — compute sorted_expert_idx and sorted_row_idx
    TPipe sortPipe;
    if (TILING_KEY_IS(MOE_V3_MX_SORTONECORE_GATHER) ||
        TILING_KEY_IS(MOE_V3_MX_SORTONECORE_SCATTER)) {
        // Single-core sort
        MoeSortOneCore op;
        op.Init(expertIdx, expandedRowIdx, userWS, t, &sortPipe);
        op.Process();
    } else if (TILING_KEY_IS(MOE_V3_MX_SORTMULTICORE_GATHER) ||
               TILING_KEY_IS(MOE_V3_MX_SORTMULTICORE_SCATTER)) {
        // Multi-core sort
        MoeSortMultiCore op;
        op.Init(expertIdx, expandedRowIdx, userWS, t, &sortPipe);
        op.Process();
    }
    sortPipe.Destroy();

    // Stage 2: ExpertTokensCount — compute expert_tokens_count_or_cumsum
    TPipe histogramPipe;
    ExpertTokensCount countOp;
    countOp.Init(expandedRowIdx, expertTokensCountOrCumsum, userWS, t, &histogramPipe);
    countOp.Process();
    histogramPipe.Destroy();

    // Stage 3: RowIdxGather — only for GATHER index mode (row_idx_type=0)
    if (TILING_KEY_IS(MOE_V3_MX_SORTONECORE_GATHER) ||
        TILING_KEY_IS(MOE_V3_MX_SORTMULTICORE_GATHER)) {
        TPipe rowIdxPipe;
        RowIdxGather rowIdxGatherOp;
        rowIdxGatherOp.Init(expandedRowIdx, userWS, t, &rowIdxPipe);
        rowIdxGatherOp.Process();
        rowIdxPipe.Destroy();
    }

    // Stage 4: GatherMxfp8Quant — gather x rows by sorted_row_idx and quantize to MX-FP8
    if constexpr (IsSameType<DTYPE_X, bfloat16_t>::value || IsSameType<DTYPE_X, half>::value) {
        TPipe gatherPipe;
        MoeGatherOutMxfp8Quant<DTYPE_X, DTYPE_Y> gatherMxfp8QuantOp;
        gatherMxfp8QuantOp.Init(x, scale, userWS, expandedRowIdx, y, mxscale, expandedScale, t, &gatherPipe);
        gatherMxfp8QuantOp.Process();
        gatherPipe.Destroy();
    }

#if (__NPU_ARCH__ == 3101)
    SetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>(oriOverflowMode);
#endif
}
