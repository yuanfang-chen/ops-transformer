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
 * \file grouped_matmul_finalize_routing_apt.cpp
 * \brief
 */
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#if defined(ORIG_DTYPE_X) && defined(ORIG_DTYPE_W)
#if (ORIG_DTYPE_X == DT_FLOAT8_E4M3FN && (ORIG_DTYPE_W == DT_FLOAT4_E2M1))
    #define V310_GMM_FR_ANTI_QUANT
#endif
#endif

#if defined (V310_GMM_FR_ANTI_QUANT)
// Weight Quantization scenario (伪量化场景)
// Include all headers directly from source directory to avoid broken paths in copied files
#include "../../grouped_matmul/op_kernel/arch35/weight_quant_basic_block/basic_block_config.h"
#include "../../grouped_matmul/op_kernel/arch35/weight_quant_basic_block/weight_quant_vcv_basic_block_base.h"
#include "../../grouped_matmul/op_kernel/arch35/weight_quant_basic_block/weight_quant_basic_block.h"
#include "arch35/weight_quant_basic_block/grouped_matmul_finalize_routing_weight_quant_tiling_data.h"
#include "arch35/weight_quant_basic_block/grouped_matmul_finalize_routing_weight_quant_tiling_key.h"
#include "arch35/weight_quant_basic_block/grouped_matmul_finalize_routing_weight_quant_resplit_controller.h"
#else
// Full Quantization scenario (全量化场景)
#include "lib/matmul_intf.h"
#include "arch35/grouped_matmul_finalize_routing_tiling_key.h"
#if ORIG_DTYPE_PERTOKEN_SCALE == DT_FLOAT8_E8M0
    #include "arch35/grouped_matmul_finalize_routing.h"
#elif ORIG_DTYPE_PERTOKEN_SCALE == DT_FLOAT
    #include "arch35/grouped_matmul_finalize_routing_pertoken_dequant.h"
#endif

#endif

template <int ATRANS, int BTRANS>
__global__ __aicore__ void
grouped_matmul_finalize_routing(GM_ADDR x, GM_ADDR w, GM_ADDR scale, GM_ADDR bias, GM_ADDR pertoken_scale,
                                GM_ADDR group_list, GM_ADDR share_input, GM_ADDR logit, GM_ADDR row_index,
                                GM_ADDR offset, GM_ADDR y, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    TPipe pipe;
#if defined (V310_GMM_FR_ANTI_QUANT)
    // Weight Quantization scenario - Use GMMFRWeightQuantResplitController
    // REGISTER_TILING_DEFAULT(GMMFinalizeRoutingArch35Tiling::GMMFinalizeRoutingWeightQuantTilingData);
    // GET_TILING_DATA_WITH_STRUCT(GMMFinalizeRoutingArch35Tiling::GMMFinalizeRoutingWeightQuantTilingData, tilingData, tilingGM);
    // const GMMFinalizeRoutingArch35Tiling::GMMFinalizeRoutingWeightQuantTilingData *tiling = &tilingData;
    
    // // Use pre-defined MXA8W4_NZNK config for FP8+FP4 MX (Microscaling) format
    // // Config values: aTrans=false, bTrans=true, antiQuantType=MX, hasAntiQuantOffset=false, quantType=NONE, weightFormat=NZ
    // GROUPED_MATMUL_FINALIZE_ROUTING::GMMFRWeightQuantResplitController<
    //     DTYPE_X, DTYPE_W, DTYPE_SCALE, DTYPE_SCALE, DTYPE_PERTOKEN_SCALE, DTYPE_BIAS, DTYPE_Y, bfloat16_t,
    //     WeightQuantBatchMatmulV2::Arch35::MXA8W4_NZNK, 
    //     WeightQuantBatchMatmulV2::Arch35::VEC_ANTIQUANT_CONFIG_DYNAMIC> controller;
    
    // controller.Init(x, w, scale, scale, x, bias, group_list, pertoken_scale, y, share_input, tiling);
    // controller.Process();
#else
    // Full Quantization scenario
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
#if ORIG_DTYPE_PERTOKEN_SCALE == DT_FLOAT8_E8M0
    if constexpr (ATRANS == 0 && BTRANS == 0) { // transX = false, transW = false
        grouped_matmul_finalize_routing<Cgmct::Gemm::layout::RowMajor, Cgmct::Gemm::layout::RowMajor>(
            x, w, scale, bias, pertoken_scale, group_list, share_input, logit, row_index, offset, y, workspaceGM,
            tilingGM);
    }
    if constexpr (ATRANS == 0 && BTRANS == 1) { // transX = false, transW = true
        grouped_matmul_finalize_routing<Cgmct::Gemm::layout::RowMajor, Cgmct::Gemm::layout::ColumnMajor>(
            x, w, scale, bias, pertoken_scale, group_list, share_input, logit, row_index, offset, y, workspaceGM,
            tilingGM);
    }
#elif ORIG_DTYPE_PERTOKEN_SCALE == DT_FLOAT
    if constexpr (ATRANS == 0 && BTRANS == 0) { // transX = false, transW = false
        grouped_matmul_finalize_routing_pertoken_dequant<Cgmct::Gemm::layout::RowMajor, Cgmct::Gemm::layout::Nz>(
            x, w, scale, bias, pertoken_scale, group_list, share_input, logit, row_index, offset, y, workspaceGM,
            tilingGM);
    }
    if constexpr (ATRANS == 0 && BTRANS == 1) { // transX = false, transW = true
        grouped_matmul_finalize_routing_pertoken_dequant<Cgmct::Gemm::layout::RowMajor, Cgmct::Gemm::layout::Zn>(
            x, w, scale, bias, pertoken_scale, group_list, share_input, logit, row_index, offset, y, workspaceGM,
            tilingGM);
    }
#endif
#endif
}
#endif
