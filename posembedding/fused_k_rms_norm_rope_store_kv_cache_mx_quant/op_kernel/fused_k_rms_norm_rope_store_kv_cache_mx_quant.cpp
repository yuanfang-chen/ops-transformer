/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fused_k_rms_norm_rope_store_kv_cache_mx_quant.cpp
 * \brief
 */

#include "arch35/fused_k_rms_norm_rope_store_kv_cache_mx_quant_regbase.h"

#define FLOAT_OVERFLOW_MODE_CTRL 60

using namespace FusedKRmsNormRopeStoreKvCacheMxQuant;
extern "C" __global__ __aicore__ void fused_k_rms_norm_rope_store_kv_cache_mx_quant(
    GM_ADDR qkv, GM_ADDR cos, GM_ADDR sin, GM_ADDR gamma, GM_ADDR kv_slot_mapping, GM_ADDR v_scale_slot_mapping,
    GM_ADDR k_cache, GM_ADDR k_scale_cache, GM_ADDR v_cache, GM_ADDR v_scale_cache, GM_ADDR q, GM_ADDR q_scale,
    GM_ADDR k_cache_out, GM_ADDR k_scale_cache_out, GM_ADDR v_cache_out, GM_ADDR v_scale_cache_out, GM_ADDR workspace,
    GM_ADDR tiling)
{
    TPipe pipe;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
#if (__NPU_ARCH__ == 3510)
    int64_t oriOverflowMode = AscendC::GetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>();
#endif

    if (TILING_KEY_IS(0)) {
        GET_TILING_DATA_WITH_STRUCT(FusedKRmsNormRopeStoreKvCacheMxQuantTilingData, tiling_data_in, tiling);
        const FusedKRmsNormRopeStoreKvCacheMxQuantTilingData *__restrict tilingData = &tiling_data_in;
        FusedKRmsNormRopeStoreKvCacheMxQuantRegbase<DTYPE_QKV, DTYPE_Q> op(&pipe, tilingData);
        op.Init(qkv, cos, sin, gamma, kv_slot_mapping, v_scale_slot_mapping, k_cache, k_scale_cache, v_cache,
                v_scale_cache, q, q_scale);
        // 对q做rope，mxquant
        op.DoPhase1();
        // 对k做rms_norm，rope，mxquant，再scatter到cache中
        op.DoPhase2();
        // 对v做mxquant再scatter到cache中
        op.DoPhase3();
    }

#if (__NPU_ARCH__ == 3510)
    AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(oriOverflowMode);
#endif
}