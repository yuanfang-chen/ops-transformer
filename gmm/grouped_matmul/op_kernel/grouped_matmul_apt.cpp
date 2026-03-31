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
 * \file grouped_matmul_apt.cpp
 * \brief
 */

#include "grouped_matmul_utils.h"
#include "arch35/grouped_matmul_tiling_data_apt.h"
using GMMWeightQuantTilingData = GroupedMatmulTilingData::GMMWeightQuantTilingData;
#if defined(V310_GMM_ANTI_QUANT)
#include "arch35/weight_quant_basic_block/block/basic_block_config.h"
#include "arch35/weight_quant_basic_block/kernel/grouped_matmul_mxfp8fp4_kernel_resplit.h"
#include "arch35/weight_quant_basic_block/block/grouped_matmul_mxfp8fp4_block_mmad_resplit.h"
#include "arch35/weight_quant_basic_block/prologue/grouped_matmul_mxfp8fp4_prologue_mx_cast_w.h"
#include "arch35/weight_quant_basic_block/weight_quant_tiling_key.h"
using WeightQuantBatchMatmulV2::Arch35::MXA8W4_NZNK;
using WeightQuantBatchMatmulV2::Arch35::WeightQuantMatmulBasicBlockAic;
using WeightQuantBatchMatmulV2::Arch35::WeightQuantMatmulBasicBlockAiv;
static constexpr VecAntiQuantConfig VEC_ANTIQUANT_CONFIG_DYNAMIC = {4, 0};

__aicore__ inline void LaunchMxA8W4VectorAntiQuantResplit(
    GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR antiquantScale, GM_ADDR groupList, GM_ADDR perTokenScale,
    GM_ADDR y, GM_ADDR tiling)
{
    GET_TILING_DATA_MEMBER(GMMWeightQuantTilingData, gmmWeightQuantParam, gmmBaseParams_, tiling);
    GET_TILING_DATA_MEMBER(GMMWeightQuantTilingData, mmTilingData, mmTilingData_, tiling);

    GROUPED_MATMUL::GMMWeightQuantResplitController<DTYPE_X, DTYPE_WEIGHT, DTYPE_ANTIQUANT_SCALE, DTYPE_SCALE,
                                                    DTYPE_PER_TOKEN_SCALE, DTYPE_BIAS, DTYPE_Y,
                                                    WeightQuantMatmulBasicBlockAic, WeightQuantMatmulBasicBlockAiv,
                                                    MXA8W4_NZNK,
                                                    VEC_ANTIQUANT_CONFIG_DYNAMIC>
        op(x, weight, antiquantScale, bias, groupList, perTokenScale, y, &gmmBaseParams_,
           &mmTilingData_);
    op();
}

template <int8_t W_TYPE, int8_t OFFSET_OR_BIAS_EXIT, int8_t C_QUANT_TYPE, int8_t W_QUANT_TYPE, int8_t WQ_B_TRANS,
          int8_t WQ_A_TRANS, int8_t TEMPLATE_CUSTOM_SC, int8_t ALGORITHM_SUB_CATEGORY, int8_t ALGORITHM_CATEGORY>
__aicore__ inline constexpr bool IsMxA8W4VectorAntiQuantResplit()
{
    return W_TYPE == WQGMM_FRACTAL_NZ &&
           OFFSET_OR_BIAS_EXIT == WQGMM_ANTIQUANT_OFFSET_NOT_EXIST_BIAS_NOT_EXIST &&
           C_QUANT_TYPE == WQGMM_NONE &&
           W_QUANT_TYPE == WQGMM_MX &&
           WQ_B_TRANS == WQGMM_TRANS &&
           WQ_A_TRANS == WQGMM_NO_TRANS &&
           TEMPLATE_CUSTOM_SC == WQGMM_MTE2_INNER_SIZE_DYNAMIC_BUF_NUM_4 &&
           ALGORITHM_SUB_CATEGORY == WQGMM_N_FIRST_TAIL_RESPLIT &&
           ALGORITHM_CATEGORY == WQGMM_VECTOR_ANTIQUANT;
}
#endif

using namespace AscendC;
using namespace GROUPED_MATMUL;


template <int8_t W_TYPE, int8_t OFFSET_OR_BIAS_EXIT, int8_t C_QUANT_TYPE, int8_t W_QUANT_TYPE, int8_t WQ_B_TRANS,
          int8_t WQ_A_TRANS, int8_t TEMPLATE_CUSTOM_SC, int8_t ALGORITHM_SUB_CATEGORY,
          int8_t ALGORITHM_CATEGORY>
__global__ __aicore__ void grouped_matmul(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR scale,
                                                     GM_ADDR offset, GM_ADDR antiquantScale, GM_ADDR antiquantOffset,
                                                     GM_ADDR groupList, GM_ADDR perTokenScale, GM_ADDR y,
                                                     GM_ADDR workspace, GM_ADDR tiling)
{
    AscendCUtils::SetOverflow(1);
#ifndef __CCE_KT_TEST__
#if defined(V310_GMM_ANTI_QUANT)
    REGISTER_TILING_DEFAULT(GMMWeightQuantTilingData);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    #if ORIG_DTYPE_X == DT_FLOAT8_E4M3FN
        if constexpr (IsMxA8W4VectorAntiQuantResplit<W_TYPE, OFFSET_OR_BIAS_EXIT, C_QUANT_TYPE, W_QUANT_TYPE,
                                                     WQ_B_TRANS, WQ_A_TRANS, TEMPLATE_CUSTOM_SC,
                                                     ALGORITHM_SUB_CATEGORY, ALGORITHM_CATEGORY>()) {
            LaunchMxA8W4VectorAntiQuantResplit(x, weight, bias, antiquantScale, groupList, perTokenScale, y, tiling);
        }
    #endif
#endif
#endif
}
