/* *
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
  */

/* !
 * \file allto_allv_grouped_mat_mul.cpp
 * \brief
 */
#include "kernel_operator.h"
#include "allto_allv_gmm_utils.h"
#ifdef ALLTO_ALLV_GMM_NO_QUANT
#include "allto_allv_grouped_mat_mul_coarse_grained.h"
#elif defined(ALLTO_ALLV_GMM_QUANT)
#include "quant_allto_allv_grouped_mat_mul.h"
#endif
#include "allto_allv_grouped_mat_mul_tiling_key.h"

using namespace AscendC;

template <typename X_TYPE, const bool IS_ALLTOALL_OPT_MM, const bool IS_ALLTOALL_GMM_WEIGHT_TRANS,
            const bool IS_ALLTOALL_MM_WEIGHT_TRANS>
struct ATAVGMM { // Grouped_Mat_Mul_All_To_Allv_Type
    using xType = X_TYPE;
    static constexpr bool isOptionalMm = IS_ALLTOALL_OPT_MM;
    static constexpr bool isGmmWeightTrans = IS_ALLTOALL_GMM_WEIGHT_TRANS;
    static constexpr bool isOptWeightTrans = IS_ALLTOALL_MM_WEIGHT_TRANS;
};

template <
    typename X_TYPE, typename W_TYPE, typename BIAS_TYPE, typename SCALE_TYPE, 
    typename Y_TYPE, CubeFormat W_FORMAT, bool A_TRANS, bool B_TRANS,
    bool IS_OPTIONAL_MM, bool IS_GMM_WEIGHT_TRANS, bool IS_OPT_WEIGHT_TRANS>
struct ATAVGMMQuant {
    using xType = X_TYPE;
    using wType = W_TYPE;
    using biasType = BIAS_TYPE;
    using scaleType = SCALE_TYPE;
    using yType = Y_TYPE;
    static constexpr CubeFormat wFormat = W_FORMAT;
    static constexpr bool aTrans = A_TRANS;
    static constexpr bool bTrans = B_TRANS;
    static constexpr bool isOptionalMm = IS_OPTIONAL_MM;
    static constexpr bool isGmmWeightTrans = IS_GMM_WEIGHT_TRANS;
    static constexpr bool isOptWeightTrans = IS_OPT_WEIGHT_TRANS;
};

#define INVOKE_ALLTOALLV_GROUPED_MATMUL_OP_IMPL_PERTENSOR(templateClass, ...)                                     \
    do {                                                                                                          \
        TPipe pipe;                                                                                               \
        templateClass<ATAVGMMQuant<__VA_ARGS__>> op;                                                              \
        op.Init(gmmxGM, gmmweightGM, sendCountsTensorOptionalGM, recvCountsTensorOptionalGM, mmxOptionalGM,       \
            mmweightOptionalGM, biasGM, gmmxScaleGM, gmmWeightScaleGM, mmxScaleGM, mmWeightScaleGM, gmmyGM,       \
            mmyOptionalGM, permuteOutOptionalGM, workspaceGM, contextGM, &tilingData, tilingGM,                    \
            hcclInitTiling, alltoAllvCcTiling, &pipe);                                                            \
        op.Process();                                                                                             \
    } while (0)

#define INVOKE_ALLTOALLV_GROUPED_MATMUL_OP_IMPL(templateClass, ...)                                               \
    do {                                                                                                          \
        TPipe pipe;                                                                                               \
        templateClass<ATAVGMM<__VA_ARGS__>> op;                                                               \
        op.Init(gmmxGM, gmmweightGM, sendCountsTensorOptionalGM, recvCountsTensorOptionalGM, mmxOptionalGM,       \
            mmweightOptionalGM, gmmyGM, mmyOptionalGM, permuteOutOptionalGM, workspaceGM, contextGM, &tilingData, \
            hcclInitTiling, alltoAllvCcTiling, &pipe);                                                            \
        op.Process();                                                                                             \
    } while (0)

template <int D_T_MM, bool TILINGKEY_MM, bool TILINGKEY_GMM_WEIGHT_TRANSPOSE, bool TILINGKEY_MM_WEIGHT_TRANSPOSE>
__global__ __aicore__ void allto_allv_grouped_mat_mul(GM_ADDR gmmxGM, GM_ADDR gmmweightGM, GM_ADDR biasGM,
    GM_ADDR sendCountsTensorOptionalGM, GM_ADDR recvCountsTensorOptionalGM, GM_ADDR mmxOptionalGM,
    GM_ADDR mmweightOptionalGM, GM_ADDR gmmxScaleGM, GM_ADDR gmmWeightScaleGM, GM_ADDR mmxScaleGM,
    GM_ADDR mmWeightScaleGM, GM_ADDR gmmyGM, GM_ADDR mmyOptionalGM, GM_ADDR permuteOutOptionalGM,
    GM_ADDR workspaceGM, GM_ADDR tilingGM) 
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

#ifdef ALLTO_ALLV_GMM_NO_QUANT
    REGISTER_TILING_DEFAULT(AlltoAllvGmmTilingData);
    auto tiling = (__gm__ AlltoAllvGmmTilingData *)tilingGM;
#elif defined(ALLTO_ALLV_GMM_QUANT)
    REGISTER_TILING_DEFAULT(QuantAlltoAllvGroupedMatmulTilingData);
    auto tiling = (__gm__ QuantAlltoAllvGroupedMatmulTilingData *)tilingGM;
#endif
    __gm__ void *hcclInitTiling = (__gm__ void *)(&(tiling->hcclInitTiling));
    __gm__ void *alltoAllvCcTiling = (__gm__ void *)(&(tiling->alltoAllvCcTiling));
    GET_TILING_DATA(tilingData, tilingGM);

    TPipe pipe;
    GM_ADDR contextGM = GetHcclContext<HCCL_GROUP_ID_0>();
#ifdef ALLTO_ALLV_GMM_NO_QUANT
#if (ORIG_DTYPE_GMM_X == DT_BFLOAT16)
    INVOKE_ALLTOALLV_GROUPED_MATMUL_OP_IMP(AlltoAllvGmmCoarseGrained, DTYPE_GMM_X, TILINGKEY_MM,
        TILINGKEY_GMM_WEIGHT_TRANSPOSE, TILINGKEY_MM_WEIGHT_TRANSPOSE);
#elif (ORIG_DTYPE_GMM_X == DT_FLOAT16)
    INVOKE_ALLTOALLV_GROUPED_MATMUL_OP_IMP(AlltoAllvGmmCoarseGrained, DTYPE_GMM_X, TILINGKEY_MM,
        TILINGKEY_GMM_WEIGHT_TRANSPOSE, TILINGKEY_MM_WEIGHT_TRANSPOSE);
#endif
#elif defined(ALLTO_ALLV_GMM_QUANT)
    // using X_TYPE = hifloat8_t;
    // using W_TYPE = hifloat8_t;  // 量化权重
    using BIAS_TYPE = float;
    using SCALE_TYPE = float;
    using Y_TYPE = float;
    constexpr CubeFormat W_FORMAT = CubeFormat::ND;
    constexpr bool A_TRANS = false;
    constexpr bool B_TRANS = false;

    INVOKE_ALLTOALLV_GROUPED_MATMUL_OP_IMPL_PERTENSOR(QuantAlltoAllvGmm,DTYPE_GMM_X, DTYPE_GMM_X, BIAS_TYPE, SCALE_TYPE, 
                                                        Y_TYPE, W_FORMAT, A_TRANS, B_TRANS, TILINGKEY_MM, TILINGKEY_GMM_WEIGHT_TRANSPOSE, 
                                                        TILINGKEY_MM_WEIGHT_TRANSPOSE);
#endif
}