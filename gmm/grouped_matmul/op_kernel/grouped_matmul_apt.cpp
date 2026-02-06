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

#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)
#include "grouped_matmul_utils.h"
#include "arch35/grouped_matmul_tiling_data_apt.h"
using GMMWeightQuantTilingData = GroupedMatmulTilingData::GMMWeightQuantTilingData;
using GMMNoQuantTilingData = GroupedMatmulTilingData::GMMNoQuantTilingData;
using GMMQuantTilingData = GroupedMatmulTilingData::GMMQuantTilingData;
#if defined(V310_GMM_QUANT)
#include "arch35/quant_adaptive_sliding_window_templates/gqmm_tiling_key.h"
#if defined(V310_GMM_QUANT_MX) || defined(V310_GMM_QUANT_CUBE) || defined(V310_GMM_QUANT_PERTENSOR_CUBE)
#include "arch35/quant_adaptive_sliding_window_templates/gqmm_cube_on_the_fly.h"
#endif
#if defined(V310_GMM_QUANT_MX) || defined(V310_GMM_QUANT_PERTENSOR_CUBE)
#include "arch35/quant_adaptive_sliding_window_templates/gqmm_init_output.h"
#endif
#if defined(V310_GMM_QUANT_MX) || defined(V310_GMM_QUANT_PERTENSOR_CUBE)
#include "arch35/quant_adaptive_sliding_window_templates/gqmm_mix_online_dynamic.h"
#endif
#if defined(V310_GMM_QUANT_PERTILE)
#include "arch35/quant_adaptive_sliding_window_templates/gqmm_cgmct_pertile_kernel.h"
#endif
#elif defined(V310_GMM_ANTI_QUANT)
#include "arch35/weight_quant_basic_block/basic_block_config.h"
#include "arch35/weight_quant_basic_block/grouped_matmul_weight_quant_basic_controller.h"
#include "arch35/weight_quant_basic_block/grouped_matmul_weight_quant_resplit_controller.h"
#include "arch35/weight_quant_basic_block/weight_quant_basic_block.h"
#include "arch35/weight_quant_basic_block/weight_quant_vcv_basic_block.h"
#include "arch35/weight_quant_basic_block/weight_quant_tiling_key.h"
using WeightQuantBatchMatmulV2::Arch35::QuantType;
using WeightQuantBatchMatmulV2::Arch35::A16MXF4_NZKN;
using WeightQuantBatchMatmulV2::Arch35::MXA8W4_NZNK;
using WeightQuantBatchMatmulV2::Arch35::S8S4_NZKN_G;
using WeightQuantBatchMatmulV2::Arch35::WeightQuantMatmulBasicBlock;
using WeightQuantBatchMatmulV2::Arch35::WeightQuantVcvMatmulBasicBlock;
static constexpr VecAntiQuantConfig VEC_ANTIQUANT_CONFIG_0 = {2, 512};
static constexpr VecAntiQuantConfig VEC_ANTIQUANT_CONFIG_1 = {4, 512};
static constexpr VecAntiQuantConfig VEC_ANTIQUANT_CONFIG_2 = {2, 1024};
static constexpr VecAntiQuantConfig VEC_ANTIQUANT_CONFIG_3 = {4, 256};
static constexpr VecAntiQuantConfig VEC_ANTIQUANT_CONFIG_4 = {3, 512};
static constexpr VecAntiQuantConfig VEC_ANTIQUANT_CONFIG_5 = {3, 384};
static constexpr VecAntiQuantConfig VEC_ANTIQUANT_CONFIG_DYNAMIC = {4, 0};
#if defined(DT_FLOAT) && defined(ORIG_DTYPE_WEIGHT) && ORIG_DTYPE_WEIGHT == DT_FLOAT
    #undef DTYPE_WEIGHT
    #define DTYPE_WEIGHT fp4x2_e2m1_t
#endif
#if defined(DT_INT32) && defined(ORIG_DTYPE_WEIGHT) && ORIG_DTYPE_WEIGHT == DT_INT32
    #undef DTYPE_WEIGHT
    #define DTYPE_WEIGHT AscendC::int4b_t
    #undef ORIG_DTYPE_WEIGHT
    #define ORIG_DTYPE_WEIGHT DT_INT4
#endif
#else
#include "arch35/non_quant/grouped_matmul_basic_kernel.h"
#include "arch35/non_quant/grouped_matmul_tiling_key.h"
#endif

using namespace AscendC;
using namespace matmul;
using namespace GROUPED_MATMUL;

#ifndef FORMAT_FRACTAL_NZ
    #define FORMAT_FRACTAL_NZ
#endif

namespace {
#if defined(FORMAT_WEIGHT) && FORMAT_WEIGHT == FORMAT_FRACTAL_NZ
constexpr CubeFormat wFormat = CubeFormat::NZ;
#else
constexpr CubeFormat wFormat = CubeFormat::ND;
#endif

}

template <bool trans = false>
using xType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_X, trans>;

template <bool trans = false>
using weightType = MatmulType<AscendC::TPosition::GM, wFormat, DTYPE_X, trans>;

using yType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, MM_DTYPE_Y>;

using biasType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_BIAS>;

#define INVOKE_GMM_WEIGHT_QUANT_BASIC_CONTROLLER_OP_IMPL(templateClass, ...)                                           \
    do {                                                                                                               \
        GET_TILING_DATA_MEMBER(GMMWeightQuantTilingData, gmmWeightQuantParam, gmmBaseParams_, tiling);                 \
        GET_TILING_DATA_MEMBER(GMMWeightQuantTilingData, mmTilingData, mmTilingData_, tiling);                         \
        GET_TILING_DATA_MEMBER_ADDR(GMMWeightQuantTilingData, gmmArray, gmmArrayAddr_, tiling);                        \
        templateClass<DTYPE_X, DTYPE_WEIGHT, DTYPE_BIAS, DTYPE_Y, __VA_ARGS__> op;                                     \
        op.Init(x, weight, antiquantScale, antiquantOffset, bias, groupList, y, &gmmBaseParams_,                       \
                &mmTilingData_, tiling, gmmArrayAddr_, &tPipe);                                                        \
        op.Process();                                                                                                  \
    } while (0)

#define INVOKE_GMM_WEIGHT_QUANT_RESPLIT_CONTROLLER_OP_IMPL(templateClass, ...)                                           \
    do {                                                                                                         \
        GET_TILING_DATA_MEMBER(GMMWeightQuantTilingData, gmmWeightQuantParam, gmmBaseParams_, tiling);           \
        GET_TILING_DATA_MEMBER(GMMWeightQuantTilingData, mmTilingData, mmTilingData_, tiling);                   \
        templateClass<DTYPE_X, DTYPE_WEIGHT, DTYPE_ANTIQUANT_SCALE, DTYPE_SCALE, float,                          \
                      DTYPE_BIAS, DTYPE_Y, WeightQuantMatmulBasicBlock, __VA_ARGS__> op;                         \
        op.Init(x, weight, scale, antiquantScale, antiquantOffset, bias, groupList, perTokenScale, y, &gmmBaseParams_, \
                &mmTilingData_, tiling, &tPipe);                                                                       \
        op.Process();                                                                                                  \
    } while (0)

#define INVOKE_GMM_WEIGHT_QUANT_MXA8W4_CONTROLLER_OP_IMPL(templateClass, ...)                                    \
    do {                                                                                                         \
        GET_TILING_DATA_MEMBER(GMMWeightQuantTilingData, gmmWeightQuantParam, gmmBaseParams_, tiling);           \
        GET_TILING_DATA_MEMBER(GMMWeightQuantTilingData, mmTilingData, mmTilingData_, tiling);                   \
        templateClass<DTYPE_X, DTYPE_WEIGHT, DTYPE_ANTIQUANT_SCALE, DTYPE_SCALE, DTYPE_PER_TOKEN_SCALE,          \
                      DTYPE_BIAS, DTYPE_Y, WeightQuantMatmulBasicBlock, __VA_ARGS__> op;                         \
        op.Init(x, weight, scale, antiquantScale, antiquantOffset, bias, groupList, perTokenScale, y, &gmmBaseParams_, \
                &mmTilingData_, tiling, &tPipe);                                                                       \
        op.Process();                                                                                                  \
    } while (0)

#define INVOKE_GMM_WEIGHT_QUANT_VCV_CONTROLLER_OP_IMPL(templateClass, ...)                                             \
    do {                                                                                                               \
        GET_TILING_DATA_MEMBER(GMMWeightQuantTilingData, gmmWeightQuantParam, gmmBaseParams_, tiling);                 \
        GET_TILING_DATA_MEMBER(GMMWeightQuantTilingData, mmTilingData, mmTilingData_, tiling);                         \
        templateClass<DTYPE_X, DTYPE_WEIGHT, DTYPE_ANTIQUANT_SCALE, DTYPE_SCALE, DTYPE_PER_TOKEN_SCALE,                \
                      DTYPE_BIAS, DTYPE_Y, WeightQuantVcvMatmulBasicBlock, __VA_ARGS__> op;                            \
        op.Init(x, weight, scale, antiquantScale, antiquantOffset, bias, groupList, perTokenScale, y, &gmmBaseParams_, \
                &mmTilingData_, tiling, &tPipe);                                                                       \
        op.Process();                                                                                                  \
    } while (0)

#define GMM_QUANT_IMPL_CLASS(transposeX1, transposeX2, templateClass)                                                  \
    do {                                                                                                               \
        templateClass<DTYPE_X, DTYPE_WEIGHT, DTYPE_BIAS, DTYPE_SCALE, DTYPE_Y, wFormat,                                \
                      transposeX1, transposeX2>                                                                        \
            op;                                                                                                        \
        GET_TILING_DATA_MEMBER(GMMQuantTilingData, gmmQuantParams, gmmQuantParams_, tiling);                           \
        GET_TILING_DATA_MEMBER(GMMQuantTilingData, mmTilingData, mmTilingData_, tiling);                               \
        GET_TILING_DATA_MEMBER_ADDR(GMMQuantTilingData, gmmArray, gmmArrayAddr_, tiling);                              \
        op.Init(x, weight, bias, scale, groupList, perTokenScale, y, user1, &gmmQuantParams_, &mmTilingData_,          \
                gmmArrayAddr_, &tPipe);                                                                                \
        op.Process();                                                                                                  \
    } while (0)

#define GMM_QUANT_MIX_IMPL_CLASS(transposeX1, transposeX2, templateClass)                                              \
    do {                                                                                                               \
        templateClass<DTYPE_X, DTYPE_WEIGHT, DTYPE_BIAS, DTYPE_SCALE, float, DTYPE_Y, wFormat,                         \
                      transposeX1, transposeX2, DTYPE_L0C_LOCAL>                                                       \
            op;                                                                                                        \
        GET_TILING_DATA_MEMBER(GMMQuantTilingData, gmmQuantParams, gmmQuantParams_, tiling);                           \
        GET_TILING_DATA_MEMBER(GMMQuantTilingData, mmTilingData, mmTilingData_, tiling);                               \
        GET_TILING_DATA_MEMBER_ADDR(GMMQuantTilingData, gmmArray, gmmArrayAddr_, tiling);                              \
        op.Init(x, weight, bias, scale, groupList, perTokenScale, y, user1, &gmmQuantParams_, &mmTilingData_,          \
                gmmArrayAddr_, &tPipe);                                                                                \
        op.Process();                                                                                                  \
    } while (0)

#define GMM_QUANT_WITH_EMPTY_TENSOR_IMPL_CLASS(transposeX1, transposeX2, templateClass)                                \
    do {                                                                                                               \
        GET_TILING_DATA_MEMBER(GMMQuantTilingData, gmmQuantParams, gmmQuantParams_, tiling);                           \
        GET_TILING_DATA_MEMBER(GMMQuantTilingData, mmTilingData, mmTilingData_, tiling);                               \
        GET_TILING_DATA_MEMBER_ADDR(GMMQuantTilingData, gmmArray, gmmArrayAddr_, tiling);                              \
        if ASCEND_IS_AIC {                                                                                             \
            templateClass<DTYPE_X, DTYPE_WEIGHT, DTYPE_BIAS, DTYPE_SCALE, DTYPE_Y, wFormat,                            \
                          transposeX1, transposeX2>                                                                    \
                op;                                                                                                    \
            op.Init(x, weight, bias, scale, groupList, perTokenScale, y, user1, &gmmQuantParams_, &mmTilingData_,      \
                    gmmArrayAddr_, &tPipe);                                                                            \
            op.Process();                                                                                              \
        }                                                                                                              \
        if ASCEND_IS_AIV {                                                                                             \
            GQmmEmptyTensor<DTYPE_Y>(groupList, y, &gmmQuantParams_, gmmArrayAddr_, mmTilingData_.usedCoreNum,         \
                                     &tPipe);                                                                          \
        }                                                                                                              \
    } while (0)

#define GMM_QUANT_GB_IMPL_CLASS(xLayout, wLayout, yLayout)                                                             \
    do {                                                                                                               \
        GET_TILING_DATA_MEMBER(GMMQuantTilingData, gmmQuantParams, gmmQuantParams_, tiling);                           \
        GET_TILING_DATA_MEMBER(GMMQuantTilingData, mmTilingData, mmTilingData_, tiling);                               \
        GmmCgmctPerTileKernel<DTYPE_X, DTYPE_WEIGHT, DTYPE_BIAS, DTYPE_SCALE, float, DTYPE_Y, xLayout, wLayout, yLayout, \
                            DTYPE_L0C_LOCAL>(x, weight, bias, scale, groupList, perTokenScale, y, user1,               \
                                             &gmmQuantParams_, &mmTilingData_, &tPipe);                                \
    } while (0)

#if defined(V310_GMM_QUANT)
template <int8_t QUANT_B_TRANS, int8_t QUANT_A_TRANS, int8_t KERNEL_TYPE>
#elif defined(V310_GMM_ANTI_QUANT)
template <int8_t W_TYPE, int8_t OFFSET_OR_BIAS_EXIT, int8_t C_QUANT_TYPE, int8_t W_QUANT_TYPE, int8_t WQ_B_TRANS,
          int8_t WQ_A_TRANS, int8_t TEMPLATE_CUSTOM_SC, int8_t ALGORITHM_SUB_CATEGORY,
          int8_t ALGORITHM_CATEGORY>
#else
template <int8_t NO_QUANT_B_TRANS, int8_t NO_QUANT_A_TRANS>
#endif
__global__ __aicore__ void grouped_matmul(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR scale,
                                                     GM_ADDR offset, GM_ADDR antiquantScale, GM_ADDR antiquantOffset,
                                                     GM_ADDR groupList, GM_ADDR perTokenScale, GM_ADDR y,
                                                     GM_ADDR workspace, GM_ADDR tiling)
{
    TPipe tPipe;
    AscendCUtils::SetOverflow(1);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    GM_ADDR user1 = GetUserWorkspace(workspace);

#ifndef __CCE_KT_TEST__
#if defined(V310_GMM_QUANT) // Quant: A8W8
REGISTER_TILING_DEFAULT(GMMQuantTilingData);
#if defined(V310_GMM_QUANT_MX) // mxfpx
    if constexpr (QUANT_B_TRANS == GMM_NO_TRANS && QUANT_A_TRANS == GMM_NO_TRANS
        && KERNEL_TYPE == GMM_DEQUANT_FIXP) {
        GMM_QUANT_IMPL_CLASS(false, false, GmmASWKernel);
    } else if constexpr (QUANT_B_TRANS == GMM_TRANS && QUANT_A_TRANS == GMM_NO_TRANS
        && KERNEL_TYPE == GMM_DEQUANT_FIXP) {
        GMM_QUANT_IMPL_CLASS(false, true, GmmASWKernel);
    }
#endif
#if defined(V310_GMM_QUANT_CUBE) || defined(V310_GMM_QUANT_PERTENSOR_CUBE) // scale64/perTensor/double perTensor
    if constexpr (QUANT_B_TRANS == GMM_NO_TRANS && QUANT_A_TRANS == GMM_NO_TRANS && KERNEL_TYPE == GMM_DEQUANT_FIXP) {
        do {
            GmmASWKernel<DTYPE_X, DTYPE_WEIGHT, DTYPE_BIAS, DTYPE_SCALE, DTYPE_Y, wFormat, false, false> op;
            GET_TILING_DATA_MEMBER(GMMQuantTilingData, gmmQuantParams, gmmQuantParams_, tiling);
            GET_TILING_DATA_MEMBER(GMMQuantTilingData, mmTilingData, mmTilingData_, tiling);
            GET_TILING_DATA_MEMBER_ADDR(GMMQuantTilingData, gmmArray, gmmArrayAddr_, tiling);
            // Print gmmQuantParams in C++ assignment format
            if (GetBlockIdx() == 0) {
                AscendC::PRINTF(
                    "[GMM_QUANT] Template types: DTYPE_X=%s, DTYPE_WEIGHT=%s, DTYPE_BIAS=%s, DTYPE_SCALE=%s, "
                    "DTYPE_Y=%s, wFormat=%d, transX1=false, transX2=false\n",
                    STR(DTYPE_X), STR(DTYPE_WEIGHT), STR(DTYPE_BIAS), STR(DTYPE_SCALE), STR(DTYPE_Y), wFormat);
                AscendC::PRINTF("// GMMQuantParams\n");
                AscendC::PRINTF("gmmQuantParams.groupNum = %u;        // uint32_t: 组数量\n", gmmQuantParams_.groupNum);
                AscendC::PRINTF("gmmQuantParams.activeType = %u;      // uint32_t: 激活类型\n",
                                gmmQuantParams_.activeType);
                AscendC::PRINTF("gmmQuantParams.aQuantMode = %u;      // uint32_t: A矩阵量化模式\n",
                                gmmQuantParams_.aQuantMode);
                AscendC::PRINTF("gmmQuantParams.bQuantMode = %u;      // uint32_t: B矩阵量化模式\n",
                                gmmQuantParams_.bQuantMode);
                AscendC::PRINTF("gmmQuantParams.singleX = %d;         // uint8_t: 单一X标志\n",
                                gmmQuantParams_.singleX);
                AscendC::PRINTF("gmmQuantParams.singleW = %d;         // uint8_t: 单一W标志\n",
                                gmmQuantParams_.singleW);
                AscendC::PRINTF("gmmQuantParams.singleY = %d;         // uint8_t: 单一Y标志\n",
                                gmmQuantParams_.singleY);
                AscendC::PRINTF("gmmQuantParams.groupType = %d;       // int8_t: 组类型\n", gmmQuantParams_.groupType);
                AscendC::PRINTF("gmmQuantParams.groupListType = %d;   // uint8_t: 组列表类型\n",
                                gmmQuantParams_.groupListType);
                AscendC::PRINTF("gmmQuantParams.hasBias = %d;         // uint8_t: 是否有偏置\n",
                                gmmQuantParams_.hasBias);
                AscendC::PRINTF("gmmQuantParams.reserved = %d;        // uint16_t: 保留字段\n",
                                gmmQuantParams_.reserved);
                // Print gmmArray in C++ assignment format (all groups)
                AscendC::PRINTF("// GMMArray (all %u groups)\n", gmmQuantParams_.groupNum);
                for (int32_t i = 0; i < static_cast<int32_t>(gmmQuantParams_.groupNum); i++) {
                    AscendC::PRINTF("gmmArray.mList[%d] = %d;   // M维度\n", i, *(gmmArrayAddr_ + i));
                    AscendC::PRINTF("gmmArray.kList[%d] = %d;   // K维度\n", i, *(gmmArrayAddr_ + 128 + i));
                    AscendC::PRINTF("gmmArray.nList[%d] = %d;   // N维度\n", i, *(gmmArrayAddr_ + 256 + i));
                }
                // Print mmTilingData all fields
                AscendC::PRINTF("// TCubeTiling\n");
                AscendC::PRINTF("mmTilingData.usedCoreNum = %d;    // 使用的核心数\n", mmTilingData_.usedCoreNum);
                AscendC::PRINTF("mmTilingData.M = %d;              // M维度\n", mmTilingData_.M);
                AscendC::PRINTF("mmTilingData.N = %d;              // N维度\n", mmTilingData_.N);
                AscendC::PRINTF("mmTilingData.Ka = %d;             // Ka维度\n", mmTilingData_.Ka);
                AscendC::PRINTF("mmTilingData.Kb = %d;             // Kb维度\n", mmTilingData_.Kb);
                AscendC::PRINTF("mmTilingData.singleCoreM = %d;    // 单核M维度\n", mmTilingData_.singleCoreM);
                AscendC::PRINTF("mmTilingData.singleCoreN = %d;    // 单核N维度\n", mmTilingData_.singleCoreN);
                AscendC::PRINTF("mmTilingData.singleCoreK = %d;    // 单核K维度\n", mmTilingData_.singleCoreK);
                AscendC::PRINTF("mmTilingData.baseM = %d;          // 基准M\n", mmTilingData_.baseM);
                AscendC::PRINTF("mmTilingData.baseN = %d;          // 基准N\n", mmTilingData_.baseN);
                AscendC::PRINTF("mmTilingData.baseK = %d;          // 基准K\n", mmTilingData_.baseK);
                AscendC::PRINTF("mmTilingData.depthA1 = %d;        // A1深度\n", mmTilingData_.depthA1);
                AscendC::PRINTF("mmTilingData.depthB1 = %d;        // B1深度\n", mmTilingData_.depthB1);
                AscendC::PRINTF("mmTilingData.stepM = %d;          // M步长\n", mmTilingData_.stepM);
                AscendC::PRINTF("mmTilingData.stepN = %d;          // N步长\n", mmTilingData_.stepN);
                AscendC::PRINTF("mmTilingData.isBias = %d;         // 是否有偏置\n", mmTilingData_.isBias);
                AscendC::PRINTF("mmTilingData.transLength = %d;    // 转置长度\n", mmTilingData_.transLength);
                AscendC::PRINTF("mmTilingData.iterateOrder = %d;   // 迭代顺序\n", mmTilingData_.iterateOrder);
                AscendC::PRINTF("mmTilingData.shareMode = %d;      // 共享模式\n", mmTilingData_.shareMode);
                AscendC::PRINTF("mmTilingData.shareL1Size = %d;    // L1共享大小\n", mmTilingData_.shareL1Size);
                AscendC::PRINTF("mmTilingData.shareL0CSize = %d;   // L0C共享大小\n", mmTilingData_.shareL0CSize);
                AscendC::PRINTF("mmTilingData.shareUbSize = %d;    // UB共享大小\n", mmTilingData_.shareUbSize);
                AscendC::PRINTF("mmTilingData.batchM = %d;         // 批处理M\n", mmTilingData_.batchM);
                AscendC::PRINTF("mmTilingData.batchN = %d;         // 批处理N\n", mmTilingData_.batchN);
                AscendC::PRINTF("mmTilingData.singleBatchM = %d;   // 单批M\n", mmTilingData_.singleBatchM);
                AscendC::PRINTF("mmTilingData.singleBatchN = %d;   // 单批N\n", mmTilingData_.singleBatchN);
                AscendC::PRINTF("mmTilingData.stepKa = %d;         // Ka步长\n", mmTilingData_.stepKa);
                AscendC::PRINTF("mmTilingData.stepKb = %d;         // Kb步长\n", mmTilingData_.stepKb);
                AscendC::PRINTF("mmTilingData.depthAL1CacheUB = %d; // AL1缓存UB深度\n", mmTilingData_.depthAL1CacheUB);
                AscendC::PRINTF("mmTilingData.depthBL1CacheUB = %d; // BL1缓存UB深度\n", mmTilingData_.depthBL1CacheUB);
                AscendC::PRINTF("mmTilingData.dbL0A = %d;          // L0A双缓冲\n", mmTilingData_.dbL0A);
                AscendC::PRINTF("mmTilingData.dbL0B = %d;          // L0B双缓冲\n", mmTilingData_.dbL0B);
                AscendC::PRINTF("mmTilingData.dbL0C = %d;          // L0C双缓冲\n", mmTilingData_.dbL0C);
                AscendC::PRINTF("mmTilingData.ALayoutInfoB = %d;   // A布局B\n", mmTilingData_.ALayoutInfoB);
                AscendC::PRINTF("mmTilingData.ALayoutInfoS = %d;   // A布局S\n", mmTilingData_.ALayoutInfoS);
                AscendC::PRINTF("mmTilingData.ALayoutInfoN = %d;   // A布局N\n", mmTilingData_.ALayoutInfoN);
                AscendC::PRINTF("mmTilingData.ALayoutInfoG = %d;   // A布局G\n", mmTilingData_.ALayoutInfoG);
                AscendC::PRINTF("mmTilingData.ALayoutInfoD = %d;   // A布局D\n", mmTilingData_.ALayoutInfoD);
                AscendC::PRINTF("mmTilingData.BLayoutInfoB = %d;   // B布局B\n", mmTilingData_.BLayoutInfoB);
                AscendC::PRINTF("mmTilingData.BLayoutInfoS = %d;   // B布局S\n", mmTilingData_.BLayoutInfoS);
                AscendC::PRINTF("mmTilingData.BLayoutInfoN = %d;   // B布局N\n", mmTilingData_.BLayoutInfoN);
                AscendC::PRINTF("mmTilingData.BLayoutInfoG = %d;   // B布局G\n", mmTilingData_.BLayoutInfoG);
                AscendC::PRINTF("mmTilingData.BLayoutInfoD = %d;   // B布局D\n", mmTilingData_.BLayoutInfoD);
                AscendC::PRINTF("mmTilingData.CLayoutInfoB = %d;   // C布局B\n", mmTilingData_.CLayoutInfoB);
                AscendC::PRINTF("mmTilingData.CLayoutInfoS1 = %d;  // C布局S1\n", mmTilingData_.CLayoutInfoS1);
                AscendC::PRINTF("mmTilingData.CLayoutInfoN = %d;   // C布局N\n", mmTilingData_.CLayoutInfoN);
                AscendC::PRINTF("mmTilingData.CLayoutInfoG = %d;   // C布局G\n", mmTilingData_.CLayoutInfoG);
                AscendC::PRINTF("mmTilingData.CLayoutInfoS2 = %d;  // C布局S2\n", mmTilingData_.CLayoutInfoS2);
                AscendC::PRINTF("mmTilingData.BatchNum = %d;       // 批次数\n", mmTilingData_.BatchNum);
                AscendC::PRINTF("mmTilingData.mxTypePara = %d;     // mx类型参数\n", mmTilingData_.mxTypePara);
            }
            op.Init(x, weight, bias, scale, groupList, perTokenScale, y, user1, &gmmQuantParams_, &mmTilingData_,
                    gmmArrayAddr_, &tPipe);
            op.Process();
        } while (0);
    } else if constexpr (QUANT_B_TRANS == GMM_TRANS && QUANT_A_TRANS == GMM_NO_TRANS &&
                         KERNEL_TYPE == GMM_DEQUANT_FIXP) {
        do {
            // 宏字符串化辅助宏
            GmmASWKernel<DTYPE_X, DTYPE_WEIGHT, DTYPE_BIAS, DTYPE_SCALE, DTYPE_Y, wFormat, false, true> op;
            GET_TILING_DATA_MEMBER(GMMQuantTilingData, gmmQuantParams, gmmQuantParams_, tiling);
            GET_TILING_DATA_MEMBER(GMMQuantTilingData, mmTilingData, mmTilingData_, tiling);
            GET_TILING_DATA_MEMBER_ADDR(GMMQuantTilingData, gmmArray, gmmArrayAddr_, tiling);
            // Print gmmQuantParams in C++ assignment format
            if (GetBlockIdx() == 0) {
                AscendC::PRINTF("[GMM_QUANT] Template types: DTYPE_X=%s, DTYPE_WEIGHT=%s, DTYPE_BIAS=%s, "
                                "DTYPE_SCALE=%s, DTYPE_Y=%s, wFormat=%d, transX1=false, transX2=true\n",
                                STR(DTYPE_X), STR(DTYPE_WEIGHT), STR(DTYPE_BIAS), STR(DTYPE_SCALE), STR(DTYPE_Y),
                                wFormat);
                AscendC::PRINTF("// GMMQuantParams\n");
                AscendC::PRINTF("gmmQuantParams.groupNum = %u;        // uint32_t: 组数量\n", gmmQuantParams_.groupNum);
                AscendC::PRINTF("gmmQuantParams.activeType = %u;      // uint32_t: 激活类型\n",
                                gmmQuantParams_.activeType);
                AscendC::PRINTF("gmmQuantParams.aQuantMode = %u;      // uint32_t: A矩阵量化模式\n",
                                gmmQuantParams_.aQuantMode);
                AscendC::PRINTF("gmmQuantParams.bQuantMode = %u;      // uint32_t: B矩阵量化模式\n",
                                gmmQuantParams_.bQuantMode);
                AscendC::PRINTF("gmmQuantParams.singleX = %d;         // uint8_t: 单一X标志\n",
                                gmmQuantParams_.singleX);
                AscendC::PRINTF("gmmQuantParams.singleW = %d;         // uint8_t: 单一W标志\n",
                                gmmQuantParams_.singleW);
                AscendC::PRINTF("gmmQuantParams.singleY = %d;         // uint8_t: 单一Y标志\n",
                                gmmQuantParams_.singleY);
                AscendC::PRINTF("gmmQuantParams.groupType = %d;       // int8_t: 组类型\n", gmmQuantParams_.groupType);
                AscendC::PRINTF("gmmQuantParams.groupListType = %d;   // uint8_t: 组列表类型\n",
                                gmmQuantParams_.groupListType);
                AscendC::PRINTF("gmmQuantParams.hasBias = %d;         // uint8_t: 是否有偏置\n",
                                gmmQuantParams_.hasBias);
                AscendC::PRINTF("gmmQuantParams.reserved = %d;        // uint16_t: 保留字段\n",
                                gmmQuantParams_.reserved);
                // Print gmmArray in C++ assignment format (all groups)
                AscendC::PRINTF("// GMMArray (all %u groups)\n", gmmQuantParams_.groupNum);
                for (int32_t i = 0; i < static_cast<int32_t>(gmmQuantParams_.groupNum); i++) {
                    AscendC::PRINTF("gmmArray.mList[%d] = %d;   // M维度\n", i, *(gmmArrayAddr_ + i));
                    AscendC::PRINTF("gmmArray.kList[%d] = %d;   // K维度\n", i, *(gmmArrayAddr_ + 128 + i));
                    AscendC::PRINTF("gmmArray.nList[%d] = %d;   // N维度\n", i, *(gmmArrayAddr_ + 256 + i));
                }
                // Print mmTilingData all fields
                AscendC::PRINTF("// TCubeTiling\n");
                AscendC::PRINTF("mmTilingData.usedCoreNum = %d;    // 使用的核心数\n", mmTilingData_.usedCoreNum);
                AscendC::PRINTF("mmTilingData.M = %d;              // M维度\n", mmTilingData_.M);
                AscendC::PRINTF("mmTilingData.N = %d;              // N维度\n", mmTilingData_.N);
                AscendC::PRINTF("mmTilingData.Ka = %d;             // Ka维度\n", mmTilingData_.Ka);
                AscendC::PRINTF("mmTilingData.Kb = %d;             // Kb维度\n", mmTilingData_.Kb);
                AscendC::PRINTF("mmTilingData.singleCoreM = %d;    // 单核M维度\n", mmTilingData_.singleCoreM);
                AscendC::PRINTF("mmTilingData.singleCoreN = %d;    // 单核N维度\n", mmTilingData_.singleCoreN);
                AscendC::PRINTF("mmTilingData.singleCoreK = %d;    // 单核K维度\n", mmTilingData_.singleCoreK);
                AscendC::PRINTF("mmTilingData.baseM = %d;          // 基准M\n", mmTilingData_.baseM);
                AscendC::PRINTF("mmTilingData.baseN = %d;          // 基准N\n", mmTilingData_.baseN);
                AscendC::PRINTF("mmTilingData.baseK = %d;          // 基准K\n", mmTilingData_.baseK);
                AscendC::PRINTF("mmTilingData.depthA1 = %d;        // A1深度\n", mmTilingData_.depthA1);
                AscendC::PRINTF("mmTilingData.depthB1 = %d;        // B1深度\n", mmTilingData_.depthB1);
                AscendC::PRINTF("mmTilingData.stepM = %d;          // M步长\n", mmTilingData_.stepM);
                AscendC::PRINTF("mmTilingData.stepN = %d;          // N步长\n", mmTilingData_.stepN);
                AscendC::PRINTF("mmTilingData.isBias = %d;         // 是否有偏置\n", mmTilingData_.isBias);
                AscendC::PRINTF("mmTilingData.transLength = %d;    // 转置长度\n", mmTilingData_.transLength);
                AscendC::PRINTF("mmTilingData.iterateOrder = %d;   // 迭代顺序\n", mmTilingData_.iterateOrder);
                AscendC::PRINTF("mmTilingData.shareMode = %d;      // 共享模式\n", mmTilingData_.shareMode);
                AscendC::PRINTF("mmTilingData.shareL1Size = %d;    // L1共享大小\n", mmTilingData_.shareL1Size);
                AscendC::PRINTF("mmTilingData.shareL0CSize = %d;   // L0C共享大小\n", mmTilingData_.shareL0CSize);
                AscendC::PRINTF("mmTilingData.shareUbSize = %d;    // UB共享大小\n", mmTilingData_.shareUbSize);
                AscendC::PRINTF("mmTilingData.batchM = %d;         // 批处理M\n", mmTilingData_.batchM);
                AscendC::PRINTF("mmTilingData.batchN = %d;         // 批处理N\n", mmTilingData_.batchN);
                AscendC::PRINTF("mmTilingData.singleBatchM = %d;   // 单批M\n", mmTilingData_.singleBatchM);
                AscendC::PRINTF("mmTilingData.singleBatchN = %d;   // 单批N\n", mmTilingData_.singleBatchN);
                AscendC::PRINTF("mmTilingData.stepKa = %d;         // Ka步长\n", mmTilingData_.stepKa);
                AscendC::PRINTF("mmTilingData.stepKb = %d;         // Kb步长\n", mmTilingData_.stepKb);
                AscendC::PRINTF("mmTilingData.depthAL1CacheUB = %d; // AL1缓存UB深度\n", mmTilingData_.depthAL1CacheUB);
                AscendC::PRINTF("mmTilingData.depthBL1CacheUB = %d; // BL1缓存UB深度\n", mmTilingData_.depthBL1CacheUB);
                AscendC::PRINTF("mmTilingData.dbL0A = %d;          // L0A双缓冲\n", mmTilingData_.dbL0A);
                AscendC::PRINTF("mmTilingData.dbL0B = %d;          // L0B双缓冲\n", mmTilingData_.dbL0B);
                AscendC::PRINTF("mmTilingData.dbL0C = %d;          // L0C双缓冲\n", mmTilingData_.dbL0C);
                AscendC::PRINTF("mmTilingData.ALayoutInfoB = %d;   // A布局B\n", mmTilingData_.ALayoutInfoB);
                AscendC::PRINTF("mmTilingData.ALayoutInfoS = %d;   // A布局S\n", mmTilingData_.ALayoutInfoS);
                AscendC::PRINTF("mmTilingData.ALayoutInfoN = %d;   // A布局N\n", mmTilingData_.ALayoutInfoN);
                AscendC::PRINTF("mmTilingData.ALayoutInfoG = %d;   // A布局G\n", mmTilingData_.ALayoutInfoG);
                AscendC::PRINTF("mmTilingData.ALayoutInfoD = %d;   // A布局D\n", mmTilingData_.ALayoutInfoD);
                AscendC::PRINTF("mmTilingData.BLayoutInfoB = %d;   // B布局B\n", mmTilingData_.BLayoutInfoB);
                AscendC::PRINTF("mmTilingData.BLayoutInfoS = %d;   // B布局S\n", mmTilingData_.BLayoutInfoS);
                AscendC::PRINTF("mmTilingData.BLayoutInfoN = %d;   // B布局N\n", mmTilingData_.BLayoutInfoN);
                AscendC::PRINTF("mmTilingData.BLayoutInfoG = %d;   // B布局G\n", mmTilingData_.BLayoutInfoG);
                AscendC::PRINTF("mmTilingData.BLayoutInfoD = %d;   // B布局D\n", mmTilingData_.BLayoutInfoD);
                AscendC::PRINTF("mmTilingData.CLayoutInfoB = %d;   // C布局B\n", mmTilingData_.CLayoutInfoB);
                AscendC::PRINTF("mmTilingData.CLayoutInfoS1 = %d;  // C布局S1\n", mmTilingData_.CLayoutInfoS1);
                AscendC::PRINTF("mmTilingData.CLayoutInfoN = %d;   // C布局N\n", mmTilingData_.CLayoutInfoN);
                AscendC::PRINTF("mmTilingData.CLayoutInfoG = %d;   // C布局G\n", mmTilingData_.CLayoutInfoG);
                AscendC::PRINTF("mmTilingData.CLayoutInfoS2 = %d;  // C布局S2\n", mmTilingData_.CLayoutInfoS2);
                AscendC::PRINTF("mmTilingData.BatchNum = %d;       // 批次数\n", mmTilingData_.BatchNum);
                AscendC::PRINTF("mmTilingData.mxTypePara = %d;     // mx类型参数\n", mmTilingData_.mxTypePara);
            }
            op.Init(x, weight, bias, scale, groupList, perTokenScale, y, user1, &gmmQuantParams_, &mmTilingData_,
                    gmmArrayAddr_, &tPipe);
            op.Process();
        } while (0);
    }
#endif
#if defined(V310_GMM_QUANT_MX) || defined(V310_GMM_QUANT_PERTENSOR_CUBE) // mx/perTensor/double perTensor
    if constexpr (QUANT_B_TRANS == GMM_NO_TRANS && QUANT_A_TRANS == GMM_TRANS
        && KERNEL_TYPE == GMM_DEQUANT_FIXP) {
        GMM_QUANT_WITH_EMPTY_TENSOR_IMPL_CLASS(true, false, GmmASWKernel);
    }
#endif
#if defined(V310_GMM_QUANT_MIX) // perToken/SPLIT_K/scale bf16/fp32
    if constexpr (QUANT_B_TRANS == GMM_NO_TRANS && QUANT_A_TRANS == GMM_NO_TRANS
        && KERNEL_TYPE == GMM_DEQUANT_VECTOR) {
        GMM_QUANT_MIX_IMPL_CLASS(false, false, GQmmMixRegbaseKernel);
    } else if constexpr (QUANT_B_TRANS == GMM_TRANS && QUANT_A_TRANS == GMM_NO_TRANS
        && KERNEL_TYPE == GMM_DEQUANT_VECTOR) {
        GMM_QUANT_MIX_IMPL_CLASS(false, true, GQmmMixRegbaseKernel);
    } else if constexpr (QUANT_B_TRANS == GMM_NO_TRANS && QUANT_A_TRANS == GMM_TRANS
        && KERNEL_TYPE == GMM_DEQUANT_VECTOR) {
        GMM_QUANT_MIX_IMPL_CLASS(true, false, GQmmMixRegbaseKernel);
    }
#endif
#if defined(V310_GMM_QUANT_PERTILE)
    if constexpr (QUANT_B_TRANS == GMM_NO_TRANS && QUANT_A_TRANS == GMM_NO_TRANS
        && KERNEL_TYPE == GMM_PERGROUP_PERBLOCK) {
        GMM_QUANT_GB_IMPL_CLASS(Cgmct::Gemm::layout::RowMajor, Cgmct::Gemm::layout::RowMajor,
                                Cgmct::Gemm::layout::RowMajorAlign);
    } else if constexpr (QUANT_B_TRANS == GMM_TRANS && QUANT_A_TRANS == GMM_NO_TRANS
        && KERNEL_TYPE == GMM_PERGROUP_PERBLOCK) {
        GMM_QUANT_GB_IMPL_CLASS(Cgmct::Gemm::layout::RowMajor, Cgmct::Gemm::layout::ColumnMajor,
                                Cgmct::Gemm::layout::RowMajorAlign);
    } else if constexpr (QUANT_B_TRANS == GMM_NO_TRANS && QUANT_A_TRANS == GMM_TRANS
        && KERNEL_TYPE == GMM_PERGROUP_PERBLOCK) {
        GMM_QUANT_GB_IMPL_CLASS(Cgmct::Gemm::layout::ColumnMajor, Cgmct::Gemm::layout::RowMajor,
                                Cgmct::Gemm::layout::RowMajorAlign);
    }
#endif
#elif defined(V310_GMM_ANTI_QUANT)
    REGISTER_TILING_DEFAULT(GMMWeightQuantTilingData);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    #if ORIG_DTYPE_X == DT_INT8
        if constexpr (W_TYPE == WQGMM_FRACTAL_NZ && OFFSET_OR_BIAS_EXIT == WQGMM_ANTIQUANT_OFFSET_NOT_EXIST_BIAS_NOT_EXIST
            && C_QUANT_TYPE == WQGMM_NONE && W_QUANT_TYPE == WQGMM_PER_CHANNEL && WQ_B_TRANS == WQGMM_NO_TRANS
            && WQ_A_TRANS == WQGMM_NO_TRANS && TEMPLATE_CUSTOM_SC == WQGMM_MTE2_INNER_SIZE_512_BUF_NUM_DEFAULT
            && ALGORITHM_SUB_CATEGORY == WQGMM_N_FIRST_TAIL_RESPLIT && ALGORITHM_CATEGORY == WQGMM_VECTOR_ANTIQUANT) {
            INVOKE_GMM_WEIGHT_QUANT_VCV_CONTROLLER_OP_IMPL(GMMWeightQuantResplitController, S8S4_NZKN_G,
                                                           VEC_ANTIQUANT_CONFIG_4);
        } else if constexpr (W_TYPE == WQGMM_FRACTAL_NZ &&
            OFFSET_OR_BIAS_EXIT == WQGMM_ANTIQUANT_OFFSET_NOT_EXIST_BIAS_NOT_EXIST &&
            C_QUANT_TYPE == WQGMM_NONE && W_QUANT_TYPE == WQGMM_PER_CHANNEL &&
            WQ_B_TRANS == WQGMM_NO_TRANS && WQ_A_TRANS == WQGMM_NO_TRANS &&
            TEMPLATE_CUSTOM_SC == WQGMM_MTE2_INNER_SIZE_384_BUF_NUM_3 &&
            ALGORITHM_SUB_CATEGORY == WQGMM_N_FIRST_TAIL_RESPLIT &&
            ALGORITHM_CATEGORY == WQGMM_VECTOR_ANTIQUANT) {
            INVOKE_GMM_WEIGHT_QUANT_VCV_CONTROLLER_OP_IMPL(GMMWeightQuantResplitController, S8S4_NZKN_G,
                                                           VEC_ANTIQUANT_CONFIG_5);
        }
    #elif ORIG_DTYPE_X == DT_FLOAT8_E4M3FN
        if constexpr (W_TYPE == WQGMM_FRACTAL_NZ && OFFSET_OR_BIAS_EXIT == WQGMM_ANTIQUANT_OFFSET_NOT_EXIST_BIAS_NOT_EXIST
            && C_QUANT_TYPE == WQGMM_NONE && W_QUANT_TYPE == WQGMM_MX && WQ_B_TRANS == WQGMM_TRANS
            && WQ_A_TRANS == WQGMM_NO_TRANS && TEMPLATE_CUSTOM_SC == WQGMM_MTE2_INNER_SIZE_DYNAMIC_BUF_NUM_4
            && ALGORITHM_SUB_CATEGORY == WQGMM_N_FIRST_TAIL_RESPLIT && ALGORITHM_CATEGORY == WQGMM_VECTOR_ANTIQUANT) {
            INVOKE_GMM_WEIGHT_QUANT_MXA8W4_CONTROLLER_OP_IMPL(GMMWeightQuantResplitController, MXA8W4_NZNK,
                                                              VEC_ANTIQUANT_CONFIG_DYNAMIC);
        }
    #elif ORIG_DTYPE_ANTIQUANT_SCALE == DT_FLOAT8_E8M0
        if constexpr (W_TYPE == WQGMM_FRACTAL_NZ && OFFSET_OR_BIAS_EXIT == WQGMM_ANTIQUANT_OFFSET_NOT_EXIST_BIAS_NOT_EXIST
            && C_QUANT_TYPE == WQGMM_NONE && W_QUANT_TYPE == WQGMM_MX && WQ_B_TRANS == WQGMM_NO_TRANS
            && WQ_A_TRANS == WQGMM_NO_TRANS && TEMPLATE_CUSTOM_SC == WQGMM_MTE2_INNER_SIZE_256_BUF_NUM_4
            && ALGORITHM_SUB_CATEGORY == WQGMM_N_FIRST_TAIL_RESPLIT && ALGORITHM_CATEGORY == WQGMM_VECTOR_ANTIQUANT) {
            INVOKE_GMM_WEIGHT_QUANT_RESPLIT_CONTROLLER_OP_IMPL(GMMWeightQuantResplitController, A16MXF4_NZKN,
                                                               VEC_ANTIQUANT_CONFIG_3);
        }
    #else
        if constexpr (W_TYPE == WQGMM_ND && OFFSET_OR_BIAS_EXIT == WQGMM_ANTIQUANT_OFFSET_NOT_EXIST_BIAS_NOT_EXIST
            && C_QUANT_TYPE == WQGMM_NONE && W_QUANT_TYPE == WQGMM_PER_CHANNEL && WQ_B_TRANS == WQGMM_TRANS
            && WQ_A_TRANS == WQGMM_NO_TRANS && TEMPLATE_CUSTOM_SC == WQGMM_MTE2_INNER_SIZE_256_BUF_NUM_4
            && ALGORITHM_SUB_CATEGORY == WQGMM_N_FIRST_TAIL_RESPLIT && ALGORITHM_CATEGORY == WQGMM_VECTOR_ANTIQUANT) {
            static constexpr WqmmConfig wqmmCfg = {false, true, QuantType::PER_CHANNEL, false,
                                                QuantType::NONE, CubeFormat::ND};
            INVOKE_GMM_WEIGHT_QUANT_RESPLIT_CONTROLLER_OP_IMPL(GMMWeightQuantResplitController, wqmmCfg,
                                                               VEC_ANTIQUANT_CONFIG_3);
        } else if constexpr (W_TYPE == WQGMM_ND && OFFSET_OR_BIAS_EXIT == WQGMM_ANTIQUANT_OFFSET_EXIST_BIAS_NOT_EXIST
            && C_QUANT_TYPE == WQGMM_NONE && W_QUANT_TYPE == WQGMM_PER_CHANNEL && WQ_B_TRANS == WQGMM_TRANS
            && WQ_A_TRANS == WQGMM_NO_TRANS && TEMPLATE_CUSTOM_SC == WQGMM_MTE2_INNER_SIZE_256_BUF_NUM_4
            && ALGORITHM_SUB_CATEGORY == WQGMM_N_FIRST_TAIL_RESPLIT && ALGORITHM_CATEGORY == WQGMM_VECTOR_ANTIQUANT) {
            static constexpr WqmmConfig wqmmCfg = {false, true, QuantType::PER_CHANNEL, true,
                                                   QuantType::NONE, CubeFormat::ND};
            INVOKE_GMM_WEIGHT_QUANT_RESPLIT_CONTROLLER_OP_IMPL(GMMWeightQuantResplitController, wqmmCfg,
                                                               VEC_ANTIQUANT_CONFIG_3);
        } else if constexpr (W_TYPE == WQGMM_ND && OFFSET_OR_BIAS_EXIT == WQGMM_ANTIQUANT_OFFSET_NOT_EXIST_BIAS_NOT_EXIST
            && C_QUANT_TYPE == WQGMM_NONE && W_QUANT_TYPE == WQGMM_PER_CHANNEL && WQ_B_TRANS == WQGMM_TRANS
            && WQ_A_TRANS == WQGMM_NO_TRANS && TEMPLATE_CUSTOM_SC == WQGMM_MTE2_INNER_SIZE_256_BUF_NUM_4
            && ALGORITHM_SUB_CATEGORY == WQGMM_N_FIRST_BASIC_BLOCK && ALGORITHM_CATEGORY == WQGMM_VECTOR_ANTIQUANT) {
            static constexpr WqmmConfig wqmmCfg = {false, true, QuantType::PER_CHANNEL, false,
                                                   QuantType::NONE, CubeFormat::ND};
            INVOKE_GMM_WEIGHT_QUANT_BASIC_CONTROLLER_OP_IMPL(GMMWeightQuantBasicController, wqmmCfg,
                                                             VEC_ANTIQUANT_CONFIG_3);
        } else if constexpr (W_TYPE == WQGMM_ND && OFFSET_OR_BIAS_EXIT == WQGMM_ANTIQUANT_OFFSET_EXIST_BIAS_NOT_EXIST
            && C_QUANT_TYPE == WQGMM_NONE && W_QUANT_TYPE == WQGMM_PER_CHANNEL && WQ_B_TRANS == WQGMM_TRANS
            && WQ_A_TRANS == WQGMM_NO_TRANS && TEMPLATE_CUSTOM_SC == WQGMM_MTE2_INNER_SIZE_256_BUF_NUM_4
            && ALGORITHM_SUB_CATEGORY == WQGMM_N_FIRST_BASIC_BLOCK && ALGORITHM_CATEGORY == WQGMM_VECTOR_ANTIQUANT) {
            static constexpr WqmmConfig wqmmCfg = {false, true, QuantType::PER_CHANNEL, true,
                                                   QuantType::NONE, CubeFormat::ND};
            INVOKE_GMM_WEIGHT_QUANT_BASIC_CONTROLLER_OP_IMPL(GMMWeightQuantBasicController, wqmmCfg,
                                                             VEC_ANTIQUANT_CONFIG_3);
        } else if constexpr (W_TYPE == WQGMM_ND && OFFSET_OR_BIAS_EXIT == WQGMM_ANTIQUANT_OFFSET_NOT_EXIST_BIAS_NOT_EXIST
            && C_QUANT_TYPE == WQGMM_NONE && W_QUANT_TYPE == WQGMM_PER_CHANNEL && WQ_B_TRANS == WQGMM_NO_TRANS
            && WQ_A_TRANS == WQGMM_NO_TRANS && TEMPLATE_CUSTOM_SC == WQGMM_MTE2_INNER_SIZE_256_BUF_NUM_4
            && ALGORITHM_SUB_CATEGORY == WQGMM_N_FIRST_BASIC_BLOCK && ALGORITHM_CATEGORY == WQGMM_VECTOR_ANTIQUANT) {
            static constexpr WqmmConfig wqmmCfg = {false, false, QuantType::PER_CHANNEL, false,
                                                   QuantType::NONE, CubeFormat::ND};
            INVOKE_GMM_WEIGHT_QUANT_BASIC_CONTROLLER_OP_IMPL(GMMWeightQuantBasicController, wqmmCfg,
                                                             VEC_ANTIQUANT_CONFIG_3);
        } else if constexpr (W_TYPE == WQGMM_ND && OFFSET_OR_BIAS_EXIT == WQGMM_ANTIQUANT_OFFSET_EXIST_BIAS_NOT_EXIST
            && C_QUANT_TYPE == WQGMM_NONE && W_QUANT_TYPE == WQGMM_PER_CHANNEL && WQ_B_TRANS == WQGMM_NO_TRANS
            && WQ_A_TRANS == WQGMM_NO_TRANS && TEMPLATE_CUSTOM_SC == WQGMM_MTE2_INNER_SIZE_256_BUF_NUM_4
            && ALGORITHM_SUB_CATEGORY == WQGMM_N_FIRST_BASIC_BLOCK && ALGORITHM_CATEGORY == WQGMM_VECTOR_ANTIQUANT) {
            static constexpr WqmmConfig wqmmCfg = {false, false, QuantType::PER_CHANNEL, true,
                                                   QuantType::NONE, CubeFormat::ND};
            INVOKE_GMM_WEIGHT_QUANT_BASIC_CONTROLLER_OP_IMPL(GMMWeightQuantBasicController, wqmmCfg,
                                                             VEC_ANTIQUANT_CONFIG_3);
        }
    #endif
#else
    REGISTER_TILING_DEFAULT(GMMNoQuantTilingData);
    if constexpr (NO_QUANT_B_TRANS == GMM_NO_TRANS && NO_QUANT_A_TRANS == GMM_NO_TRANS) {
        if constexpr (wFormat == CubeFormat::NZ) {
            GmmNoQuantAswt<layout::RowMajor, layout::Nz>(x, weight, bias, groupList, y, tiling);
        } else {
            GmmNoQuantAswt<layout::RowMajor, layout::RowMajor>(x, weight, bias, groupList, y, tiling);
        }
    } else if constexpr (NO_QUANT_B_TRANS == GMM_NO_TRANS && NO_QUANT_A_TRANS == GMM_TRANS) {    // x transposed
        if ASCEND_IS_AIV {
            EmptyTensor<DTYPE_Y>(x, weight, groupList, y, tiling);
        }
        if ASCEND_IS_AIC {
            if constexpr (wFormat == CubeFormat::NZ) {
                GmmNoQuantAswt<layout::ColumnMajor, layout::Nz>(x, weight, bias, groupList, y, tiling);
            } else {
                GmmNoQuantAswt<layout::ColumnMajor, layout::RowMajor>(x, weight, bias, groupList, y, tiling);
            }
        }
    } else if constexpr (NO_QUANT_B_TRANS == GMM_TRANS && NO_QUANT_A_TRANS == GMM_NO_TRANS) {    // weight transposed
        if constexpr (wFormat == CubeFormat::NZ) {
            GmmNoQuantAswt<layout::RowMajor, layout::Zn>(x, weight, bias, groupList, y, tiling);
        } else {
            GmmNoQuantAswt<layout::RowMajor, layout::ColumnMajor>(x, weight, bias, groupList, y, tiling);
        }
    }
#endif
#endif
}
