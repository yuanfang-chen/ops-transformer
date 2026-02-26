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
 * \file moe_distribute_combine_v2.cpp
 * \brief
 */
#include "basic_api/kernel_basic_intf.h"
#include "lib/matmul_intf.h"
#include "moe_distribute_combine_v2_extend_tiling_key.h"

#ifdef __DAV_C310__
#include "../../moe_distribute_combine_v2/arch35/moe_distribute_combine_v2_host_kfc.h"
#endif // __DAV_C310__

#include "../../moe_distribute_combine_v2/moe_distribute_combine_v2_tiling.h"


#ifndef __DAV_C310__
using namespace MoeDistributeCombineV2A5Impl;
#endif // __DAV_C310__

using namespace Mc2Tiling;
using namespace AscendC;

template <bool HasTp, uint8_t QuantMode, uint8_t LayeredMode, uint8_t ArchTag>
__global__ __aicore__ void
moe_distribute_combine_v2_extend(GM_ADDR expandX, GM_ADDR expertIds, GM_ADDR assistInfoForCombine, GM_ADDR epSendCount,
                                 GM_ADDR scales, GM_ADDR mc2_context, GM_ADDR tpSendCount, GM_ADDR xActiveMask, GM_ADDR activationScale,
                                 GM_ADDR weightScale, GM_ADDR groupList, GM_ADDR expandScales, GM_ADDR sharedExpertX,
                                 GM_ADDR elasticInfo, GM_ADDR oriX, GM_ADDR constExpertAlpha1,
                                 GM_ADDR constExpertAlpha2, GM_ADDR constExpertV, GM_ADDR performanceInfo, GM_ADDR XOut,
                                 GM_ADDR workspaceGM, GM_ADDR tilingGM)

{
    REGISTER_TILING_DEFAULT(MoeDistributeCombineV2TilingData);
#ifndef __DAV_C310__
    REGISTER_TILING_FOR_TILINGKEY("ArchTag == TILINGKEY_TPL_A2", MoeDistributeCombineA2TilingData);
#endif
    TPipe pipe;

#if ((ORIG_DTYPE_EXPAND_X == DT_BF16) || (ORIG_DTYPE_EXPAND_X == DT_FLOAT16))
#ifdef __DAV_C310__
    if constexpr (ArchTag == TILINGKEY_TPL_A5) {
        GET_TILING_DATA_WITH_STRUCT(MoeDistributeCombineV2TilingData, tilingData, tilingGM);
        MoeDistributeCombineV2A5Impl::MoeDistributeCombineV2A5LayeredHostcpu<DTYPE_EXPAND_X, DTYPE_X, int32_t, HasTp,
                                                                             QuantMode == TILINGKEY_INT8_QUANT>
            op;
        op.Init(expandX, expertIds, assistInfoForCombine, epSendCount, tpSendCount, scales, expandScales, xActiveMask,
                sharedExpertX, elasticInfo, oriX, constExpertAlpha1, constExpertAlpha2, constExpertV, XOut, workspaceGM,
                &pipe, &tilingData);
        op.Process();
    }
#endif
#endif
}