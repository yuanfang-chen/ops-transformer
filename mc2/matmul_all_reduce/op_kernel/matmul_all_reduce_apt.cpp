/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file matmul_all_reduce_apt.cpp
 * \brief A5
 */

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "common.h"

// david非量化
#if ((ORIG_DTYPE_X1 == ORIG_DTYPE_X2) && ((ORIG_DTYPE_X1 == DT_FLOAT16) || (ORIG_DTYPE_X1 == DT_BF16)))
#include "arch35/matmul_all_reduce_910_general.h"
#include "arch35/matmul_all_reduce_empty_tensor_k_general.h"
#endif

#if defined(WEIGHT_W4_W8)
#include "./arch35/matmul_all_reduce_weight_quant.h"
#endif

#if defined(WEIGHT_F8) || defined(WEIGHT_W4_W8)
#include "./arch35/matmul_all_reduce_weight_quant_adaptive_split.h"
#include "./arch35/matmul_all_reduce_empty_tensor_k_general.h"
using Mc2WeightQuantBatchMatmulV2::Arch35::WeightQuantBatchMatmulV2BasicBlockController;
namespace{
    static constexpr Mc2WeightQuantBatchMatmulV2::Arch35::VecAntiQuantConfig VEC_ANTIQUANT_CONFIG_0 = {2, 512}; // b 转置场景
    static constexpr Mc2WeightQuantBatchMatmulV2::Arch35::VecAntiQuantConfig VEC_ANTIQUANT_CONFIG_1 = {4, 512};
    static constexpr Mc2WeightQuantBatchMatmulV2::Arch35::VecAntiQuantConfig VEC_ANTIQUANT_CONFIG_2 = {2, 1024};
    static constexpr Mc2WeightQuantBatchMatmulV2::Arch35::VecAntiQuantConfig VEC_ANTIQUANT_CONFIG_3 = {4, 256}; // b 非转置场景
    static constexpr Mc2WeightQuantBatchMatmulV2::Arch35::VecAntiQuantConfig VEC_ANTIQUANT_CONFIG_4 = {2, 256}; // solomon
}
#endif

#if ((ORIG_DTYPE_X1 == ORIG_DTYPE_X2) && (ORIG_DTYPE_X1 == DT_INT8)) ||               \
    (((ORIG_DTYPE_X1 == ORIG_DTYPE_X2) && (ORIG_DTYPE_X1 == DT_HIFLOAT8)) ||          \
     (((ORIG_DTYPE_X1 == DT_FLOAT8_E4M3FN) || (ORIG_DTYPE_X1 == DT_FLOAT8_E5M2)) &&   \
      ((ORIG_DTYPE_X2 == DT_FLOAT8_E4M3FN) || (ORIG_DTYPE_X2 == DT_FLOAT8_E5M2)))) || \
    (((ORIG_DTYPE_X1 == DT_FLOAT4_E1M2) || (ORIG_DTYPE_X1 == DT_FLOAT4_E2M1)) &&      \
     ((ORIG_DTYPE_X2 == DT_FLOAT4_E1M2) || (ORIG_DTYPE_X2 == DT_FLOAT4_E2M1)))
#include "arch35/matmul_all_reduce_quant_pertoken.h"
#include "arch35/matmul_all_reduce_quant_pertoken_comm_int8.h"
#include "arch35/matmul_all_reduce_quant.h"
#include "arch35/matmul_all_reduce_quant_comm_int8.h"
#include "arch35/matmul_all_reduce_quant_perblock.h"
#include "arch35/matmul_all_reduce_quant_pertile_comm_fp8.h"
#include "arch35/matmul_all_reduce_quant_commfp8_mixed_calc.h"
#endif

namespace MatmulAllReduceImpl {}

using namespace AscendC;
using namespace MatmulAllReduceImpl;

extern "C" __global__ __aicore__ void matmul_all_reduce(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR addGM, GM_ADDR antiquantScaleGM, GM_ADDR antiquantOffsetGM,
    GM_ADDR dequantGM, GM_ADDR pertokenGM, GM_ADDR commQuantScale1GM, GM_ADDR commQuantScale2GM, GM_ADDR cGM,
    GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
#ifdef __CCE_KT_TEST__
    REGISTER_TILING_DEFAULT(MatmulAllReduce910TilingData);
#endif
    if (workspaceGM == nullptr) {
        return;
    }

    GM_ADDR userWS = GetUserWorkspace(workspaceGM);
    if (userWS == nullptr) {
        return;
    }

    TPipe tPipe;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    
#if ((ORIG_DTYPE_X1 == ORIG_DTYPE_X2) && ((ORIG_DTYPE_X1 == DT_FLOAT16) || (ORIG_DTYPE_X1 == DT_BF16)))
    // 910非量化
    if (TILING_KEY_IS(11000000000000001100UL)) {
        KERNEL_TASK_TYPE(11000000000000001100UL, KERNEL_TYPE_MIX_AIC_1_0);
        INVOKE_MC2_910_OP_IMPL(Mc2MatmulV3Advanced::Mc2MatmulAswKernel, Mc2CoreType::ON_CUBE);
    } else if (TILING_KEY_IS(11000000000000000001UL)) {
        INVOKE_MC2_910_OP_IMPL(Mc2MatmulV3Advanced::Mc2MatmulAswKernel, Mc2CoreType::ON_CUBE_AND_VECTOR);
    } else if (TILING_KEY_IS(11000000000000000009UL)) {
        KERNEL_TASK_TYPE(11000000000000000009UL, KERNEL_TYPE_MIX_AIV_1_0);
        INVOKE_MC2_EMPTY_TENSOR_OP_IMPL();
    }
#endif
    // 除非量化以外，david的tiling均来源于对应的mmv3算子，在其基础上增加对应偏移，具体参看tiling
#if defined(WEIGHT_W4_W8) || defined(WEIGHT_F8)
#if defined(WEIGHT_W4_W8)
#undef DTYPE_BIAS
#define DTYPE_BIAS DTYPE_X1
    // perchannel&pertensor的key计划删除，走新tilingkey，跟随matmul，暂时保留
    if (TILING_KEY_IS(100200UL)) {
        INVOKE_MC2_WEIGHT_QUANT_KERNEL(false, Mc2QuantType::PER_CHANNEL, false, false);
    } else if (TILING_KEY_IS(101200UL)) {
        INVOKE_MC2_WEIGHT_QUANT_KERNEL(false, Mc2QuantType::PER_CHANNEL, true, false);
    } else if (TILING_KEY_IS(100210UL)) {
        INVOKE_MC2_WEIGHT_QUANT_KERNEL(true, Mc2QuantType::PER_CHANNEL, false, false);
    } else if (TILING_KEY_IS(101210UL)) {
        INVOKE_MC2_WEIGHT_QUANT_KERNEL(true, Mc2QuantType::PER_CHANNEL, true, false);
    } else if (TILING_KEY_IS(100300UL)) {
        INVOKE_MC2_WEIGHT_QUANT_KERNEL(false, Mc2QuantType::PER_GROUP, false, false);
    } else if (TILING_KEY_IS(100310UL)) {
        INVOKE_MC2_WEIGHT_QUANT_KERNEL(true, Mc2QuantType::PER_GROUP, false, false);
    } else if (TILING_KEY_IS(101300UL)) {
        INVOKE_MC2_WEIGHT_QUANT_KERNEL(false, Mc2QuantType::PER_GROUP, true, false);
    } else if (TILING_KEY_IS(101310UL)) {
        INVOKE_MC2_WEIGHT_QUANT_KERNEL(true, Mc2QuantType::PER_GROUP, true, false);
    } else if (TILING_KEY_IS(100100UL)) {
        INVOKE_MC2_WEIGHT_QUANT_KERNEL(false, Mc2QuantType::PER_TENSOR, false, false);
    } else if (TILING_KEY_IS(100110UL)) {
        INVOKE_MC2_WEIGHT_QUANT_KERNEL(true, Mc2QuantType::PER_TENSOR, false, false);
    } else if (TILING_KEY_IS(101100UL)) {
        INVOKE_MC2_WEIGHT_QUANT_KERNEL(false, Mc2QuantType::PER_TENSOR, true, false);
    } else if (TILING_KEY_IS(101110UL)) {
        INVOKE_MC2_WEIGHT_QUANT_KERNEL(true, Mc2QuantType::PER_TENSOR, true, false);
    }
#endif
    if (TILING_KEY_IS(2000030004000012100UL)) {
        INVOKE_MC2_WEIGHT_QUANT_ADAPTIVE_SPLIT_KERNEL(
            true, false, Mc2QuantType::PER_CHANNEL, DTYPE_BIAS, VEC_ANTIQUANT_CONFIG_0);
    } else if (TILING_KEY_IS(2000030004000012120UL)) {
        INVOKE_MC2_WEIGHT_QUANT_ADAPTIVE_SPLIT_KERNEL(
            true, true, Mc2QuantType::PER_CHANNEL, DTYPE_BIAS, VEC_ANTIQUANT_CONFIG_0);
    } else if (TILING_KEY_IS(2000030003000002100UL)) {
        INVOKE_MC2_WEIGHT_QUANT_ADAPTIVE_SPLIT_KERNEL(
            false, false, Mc2QuantType::PER_CHANNEL, DTYPE_BIAS, VEC_ANTIQUANT_CONFIG_3);
    } else if (TILING_KEY_IS(2000030003000002120UL)) {
        INVOKE_MC2_WEIGHT_QUANT_ADAPTIVE_SPLIT_KERNEL(
            false, true, Mc2QuantType::PER_CHANNEL, DTYPE_BIAS, VEC_ANTIQUANT_CONFIG_3);
    } else if (TILING_KEY_IS(2000020000000012100UL)) {
        INVOKE_MC2_WEIGHT_QUANT_ADAPTIVE_SPLIT_KERNEL(
            true, false, Mc2QuantType::PER_CHANNEL, DTYPE_BIAS, VEC_ANTIQUANT_CONFIG_0);
    } else if (TILING_KEY_IS(2000020001000012100UL)) {
        INVOKE_MC2_WEIGHT_QUANT_ADAPTIVE_SPLIT_KERNEL(
            true, false, Mc2QuantType::PER_CHANNEL, DTYPE_BIAS, VEC_ANTIQUANT_CONFIG_0);
    } else if (TILING_KEY_IS(2000020002000012100UL)) {
        INVOKE_MC2_WEIGHT_QUANT_ADAPTIVE_SPLIT_KERNEL(
            true, false, Mc2QuantType::PER_CHANNEL, DTYPE_BIAS, VEC_ANTIQUANT_CONFIG_0);
    } else if (TILING_KEY_IS(2000020003000012100UL)) {
        INVOKE_MC2_WEIGHT_QUANT_ADAPTIVE_SPLIT_KERNEL(
            true, false, Mc2QuantType::PER_CHANNEL, DTYPE_BIAS, VEC_ANTIQUANT_CONFIG_0);
    } else if (TILING_KEY_IS(2000020000000012120UL)) {
        INVOKE_MC2_WEIGHT_QUANT_ADAPTIVE_SPLIT_KERNEL(
            true, true, Mc2QuantType::PER_CHANNEL, DTYPE_BIAS, VEC_ANTIQUANT_CONFIG_0);
    } else if (TILING_KEY_IS(2000020001000012120UL)) {
        INVOKE_MC2_WEIGHT_QUANT_ADAPTIVE_SPLIT_KERNEL(
            true, true, Mc2QuantType::PER_CHANNEL, DTYPE_BIAS, VEC_ANTIQUANT_CONFIG_0);
    } else if (TILING_KEY_IS(2000020002000012120UL)) {
        INVOKE_MC2_WEIGHT_QUANT_ADAPTIVE_SPLIT_KERNEL(
            true, true, Mc2QuantType::PER_CHANNEL, DTYPE_BIAS, VEC_ANTIQUANT_CONFIG_0);
    } else if (TILING_KEY_IS(2000020003000012120UL)) {
        INVOKE_MC2_WEIGHT_QUANT_ADAPTIVE_SPLIT_KERNEL(
            true, true, Mc2QuantType::PER_CHANNEL, DTYPE_BIAS, VEC_ANTIQUANT_CONFIG_0);
    } else if (TILING_KEY_IS(2000030004000011100UL)) {
        INVOKE_MC2_WEIGHT_QUANT_ADAPTIVE_SPLIT_KERNEL(
            true, false, Mc2QuantType::PER_TENSOR, DTYPE_BIAS, VEC_ANTIQUANT_CONFIG_0);
    } else if (TILING_KEY_IS(2000030004000011120UL)) {
        INVOKE_MC2_WEIGHT_QUANT_ADAPTIVE_SPLIT_KERNEL(
            true, true, Mc2QuantType::PER_TENSOR, DTYPE_BIAS, VEC_ANTIQUANT_CONFIG_0);
    } else if (TILING_KEY_IS(2000030003000001100UL)) {
        INVOKE_MC2_WEIGHT_QUANT_ADAPTIVE_SPLIT_KERNEL(
            false, false, Mc2QuantType::PER_TENSOR, DTYPE_BIAS, VEC_ANTIQUANT_CONFIG_3);
    } else if (TILING_KEY_IS(2000030003000001120UL)) {
        INVOKE_MC2_WEIGHT_QUANT_ADAPTIVE_SPLIT_KERNEL(
            false, true, Mc2QuantType::PER_TENSOR, DTYPE_BIAS, VEC_ANTIQUANT_CONFIG_3);
    } else if (TILING_KEY_IS(10000000000000000008UL)) {
        KERNEL_TASK_TYPE(10000000000000000008UL, KERNEL_TYPE_MIX_AIV_1_0);
        INVOKE_MC2_EMPTY_TENSOR_OP_IMPL();
    }
#if defined(ORIG_DTYPE_X1) && defined(DT_BF16) && (ORIG_DTYPE_X1 == DT_BF16)
#undef DTYPE_BIAS
#define DTYPE_BIAS float
    if (TILING_KEY_IS(2000030004000012140UL)) {
        INVOKE_MC2_WEIGHT_QUANT_ADAPTIVE_SPLIT_KERNEL(
            true, false, Mc2QuantType::PER_CHANNEL, float, VEC_ANTIQUANT_CONFIG_0);
    } else if (TILING_KEY_IS(2000030003000002140UL)) {
        INVOKE_MC2_WEIGHT_QUANT_ADAPTIVE_SPLIT_KERNEL(
            false, false, Mc2QuantType::PER_CHANNEL, float, VEC_ANTIQUANT_CONFIG_3);
    } else if (TILING_KEY_IS(2000030004000012160UL)) {
        INVOKE_MC2_WEIGHT_QUANT_ADAPTIVE_SPLIT_KERNEL(
            true, true, Mc2QuantType::PER_CHANNEL, float, VEC_ANTIQUANT_CONFIG_0);
    } else if (TILING_KEY_IS(2000030003000002160UL)) {
        INVOKE_MC2_WEIGHT_QUANT_ADAPTIVE_SPLIT_KERNEL(
            false, true, Mc2QuantType::PER_CHANNEL, float, VEC_ANTIQUANT_CONFIG_3);
    } else if (TILING_KEY_IS(2000020000000012140UL)) {
        INVOKE_MC2_WEIGHT_QUANT_ADAPTIVE_SPLIT_KERNEL(
            true, false, Mc2QuantType::PER_CHANNEL, float, VEC_ANTIQUANT_CONFIG_0);
    } else if (TILING_KEY_IS(2000020001000012140UL)) {
        INVOKE_MC2_WEIGHT_QUANT_ADAPTIVE_SPLIT_KERNEL(
            true, false, Mc2QuantType::PER_CHANNEL, float, VEC_ANTIQUANT_CONFIG_0);
    } else if (TILING_KEY_IS(2000020002000012140UL)) {
        INVOKE_MC2_WEIGHT_QUANT_ADAPTIVE_SPLIT_KERNEL(
            true, false, Mc2QuantType::PER_CHANNEL, float, VEC_ANTIQUANT_CONFIG_0);
    } else if (TILING_KEY_IS(2000020003000012140UL)) {
        INVOKE_MC2_WEIGHT_QUANT_ADAPTIVE_SPLIT_KERNEL(
            true, false, Mc2QuantType::PER_CHANNEL, float, VEC_ANTIQUANT_CONFIG_0);
    } else if (TILING_KEY_IS(2000020000000012160UL)) {
        INVOKE_MC2_WEIGHT_QUANT_ADAPTIVE_SPLIT_KERNEL(
            true, true, Mc2QuantType::PER_CHANNEL, float, VEC_ANTIQUANT_CONFIG_0);
    } else if (TILING_KEY_IS(2000020001000012160UL)) {
        INVOKE_MC2_WEIGHT_QUANT_ADAPTIVE_SPLIT_KERNEL(
            true, true, Mc2QuantType::PER_CHANNEL, float, VEC_ANTIQUANT_CONFIG_0);
    } else if (TILING_KEY_IS(2000020002000012160UL)) {
        INVOKE_MC2_WEIGHT_QUANT_ADAPTIVE_SPLIT_KERNEL(
            true, true, Mc2QuantType::PER_CHANNEL, float, VEC_ANTIQUANT_CONFIG_0);
    } else if (TILING_KEY_IS(2000020003000012160UL)) {
        INVOKE_MC2_WEIGHT_QUANT_ADAPTIVE_SPLIT_KERNEL(
            true, true, Mc2QuantType::PER_CHANNEL, float, VEC_ANTIQUANT_CONFIG_0);
    } else if (TILING_KEY_IS(2000030003000001140UL)) {
        INVOKE_MC2_WEIGHT_QUANT_ADAPTIVE_SPLIT_KERNEL(
            false, false, Mc2QuantType::PER_TENSOR, float, VEC_ANTIQUANT_CONFIG_3);
    } else if (TILING_KEY_IS(2000030003000001160UL)) {
        INVOKE_MC2_WEIGHT_QUANT_ADAPTIVE_SPLIT_KERNEL(
            false, true, Mc2QuantType::PER_TENSOR, float, VEC_ANTIQUANT_CONFIG_3);
    } else if (TILING_KEY_IS(2000030004000011140UL)) {
        INVOKE_MC2_WEIGHT_QUANT_ADAPTIVE_SPLIT_KERNEL(
            true, false, Mc2QuantType::PER_TENSOR, float, VEC_ANTIQUANT_CONFIG_0);
    } else if (TILING_KEY_IS(2000030004000011160UL)) {
        INVOKE_MC2_WEIGHT_QUANT_ADAPTIVE_SPLIT_KERNEL(true, true, Mc2QuantType::PER_TENSOR, float, VEC_ANTIQUANT_CONFIG_0);
    }
#endif
#endif

#if defined(DAVID_QUANT_INT8_OUT_FP16)
#undef DTYPE_BIAS
#define DTYPE_BIAS int32_t
    if (TILING_KEY_IS(1000000000000000001)) {
        INVOKE_MC2_QUANT_910_OP_IMPL(AscendC::MatMulASWKernel, Mc2CoreType::ON_CUBE_AND_VECTOR, uint64_t, false, true);
    } else if (TILING_KEY_IS(1000000000000000000)) {
        INVOKE_MC2_QUANT_910_OP_IMPL(AscendC::MatMulASWKernel, Mc2CoreType::ON_CUBE_AND_VECTOR, uint64_t, false, false);
    } else if (TILING_KEY_IS(1000000000000000011)) {
        INVOKE_MC2_QUANT_COMM_INT8_910_OP_IMPL(AscendC::MatMulASWKernel, Mc2CoreType::ON_CUBE_AND_VECTOR, uint64_t, false, true);
    } else if (TILING_KEY_IS(1000000000000000010)) {
        INVOKE_MC2_QUANT_COMM_INT8_910_OP_IMPL(AscendC::MatMulASWKernel, Mc2CoreType::ON_CUBE_AND_VECTOR, uint64_t, false, false);
    }

    if (TILING_KEY_IS(1000000000000002000)) {
        INVOKE_BATCH_MATMUL_QUANT_PERTOKEN_IMPL(
            Mc2QuantBatchMatmulV3::Mc2QuantBmmPertokenRegbaseKernel, Mc2CoreType::ON_CUBE_AND_VECTOR, float, false, false);
    } else if (TILING_KEY_IS(1000000000000002001)) {
        INVOKE_BATCH_MATMUL_QUANT_PERTOKEN_IMPL(
            Mc2QuantBatchMatmulV3::Mc2QuantBmmPertokenRegbaseKernel, Mc2CoreType::ON_CUBE_AND_VECTOR, float, false, true);
    } else if (TILING_KEY_IS(1000000000000002010)) {
        INVOKE_BATCH_MATMUL_QUANT_PERTOKEN_COMM_INT8_IMPL(
            Mc2QuantBatchMatmulV3::Mc2QuantBmmPertokenRegbaseKernel, Mc2CoreType::ON_CUBE_AND_VECTOR, float, false, false);
    } else if (TILING_KEY_IS(1000000000000002011)) {
        INVOKE_BATCH_MATMUL_QUANT_PERTOKEN_COMM_INT8_IMPL(
            Mc2QuantBatchMatmulV3::Mc2QuantBmmPertokenRegbaseKernel, Mc2CoreType::ON_CUBE_AND_VECTOR, float, false, true);
    }
#elif defined(DAVID_QUANT_INT8_OUT_BF16)
#undef DTYPE_BIAS
#define DTYPE_BIAS int32_t
    if (TILING_KEY_IS(1000000000000000001)) {
        INVOKE_MC2_QUANT_910_OP_IMPL(AscendC::MatMulASWKernel, Mc2CoreType::ON_CUBE_AND_VECTOR, DTYPE_Y, false, true);
    } else if (TILING_KEY_IS(1000000000000000000)) {
        INVOKE_MC2_QUANT_910_OP_IMPL(AscendC::MatMulASWKernel, Mc2CoreType::ON_CUBE_AND_VECTOR, DTYPE_Y, false, false);
    } else if (TILING_KEY_IS(1000000000000000011)) {
        INVOKE_MC2_QUANT_COMM_INT8_910_OP_IMPL(AscendC::MatMulASWKernel, Mc2CoreType::ON_CUBE_AND_VECTOR, DTYPE_Y, false, true);
    } else if (TILING_KEY_IS(1000000000000000010)) {
        INVOKE_MC2_QUANT_COMM_INT8_910_OP_IMPL(AscendC::MatMulASWKernel, Mc2CoreType::ON_CUBE_AND_VECTOR, DTYPE_Y, false, false);
    }

    if (TILING_KEY_IS(1000000000000002000)) {
        INVOKE_BATCH_MATMUL_QUANT_PERTOKEN_IMPL(
            Mc2QuantBatchMatmulV3::Mc2QuantBmmPertokenRegbaseKernel, Mc2CoreType::ON_CUBE_AND_VECTOR, DTYPE_Y, false, false);
    } else if (TILING_KEY_IS(1000000000000002001)) {
        INVOKE_BATCH_MATMUL_QUANT_PERTOKEN_IMPL(
            Mc2QuantBatchMatmulV3::Mc2QuantBmmPertokenRegbaseKernel, Mc2CoreType::ON_CUBE_AND_VECTOR, DTYPE_Y, false, true);
    } else if (TILING_KEY_IS(1000000000000002010)) {
        INVOKE_BATCH_MATMUL_QUANT_PERTOKEN_COMM_INT8_IMPL(
            Mc2QuantBatchMatmulV3::Mc2QuantBmmPertokenRegbaseKernel, Mc2CoreType::ON_CUBE_AND_VECTOR, DTYPE_Y, false, false);
    } else if (TILING_KEY_IS(1000000000000002011)) {
        INVOKE_BATCH_MATMUL_QUANT_PERTOKEN_COMM_INT8_IMPL(
            Mc2QuantBatchMatmulV3::Mc2QuantBmmPertokenRegbaseKernel, Mc2CoreType::ON_CUBE_AND_VECTOR, DTYPE_Y, false, true);
    }
#elif (                                                                         \
    ((ORIG_DTYPE_X1 == DT_FLOAT4_E1M2) || (ORIG_DTYPE_X1 == DT_FLOAT4_E2M1)) && \
    ((ORIG_DTYPE_X2 == DT_FLOAT4_E1M2) || (ORIG_DTYPE_X2 == DT_FLOAT4_E2M1)))
// fp8,hif8和mixfp的场景，bias都是float
#undef DTYPE_BIAS
#define DTYPE_BIAS float
    if (TILING_KEY_IS(1000000000000000001)) {
        INVOKE_MC2_QUANT_MXFP_910_OP_IMPL(AscendC::MatMulASWKernel, Mc2CoreType::ON_CUBE_AND_VECTOR, false, true);
    }
#elif (                                                                            \
    ((ORIG_DTYPE_X1 == ORIG_DTYPE_X2) && (ORIG_DTYPE_X1 == DT_HIFLOAT8)) ||        \
    (((ORIG_DTYPE_X1 == DT_FLOAT8_E4M3FN) || (ORIG_DTYPE_X1 == DT_FLOAT8_E5M2)) && \
     ((ORIG_DTYPE_X2 == DT_FLOAT8_E4M3FN) || (ORIG_DTYPE_X2 == DT_FLOAT8_E5M2))))
// fp8,hif8和mixfp的场景，bias都是float
#undef DTYPE_BIAS
#define DTYPE_BIAS float
#if (ORIG_DTYPE_X1 != DT_HIFLOAT8)
    if (TILING_KEY_IS(1000000000000000011)) {
        INVOKE_MC2_QUANT_MXFP_910_OP_IMPL(AscendC::MatMulASWKernel, Mc2CoreType::ON_CUBE_AND_VECTOR, false, true);
    } else if (TILING_KEY_IS(1000000000050102010)) {
        INVOKE_MC2_COMM_FP8_MIXED_CALC_910_OP_IMPL(Mc2QuantBatchMatmulV3::Mc2QuantBmmPertokenRegbaseKernel,
                                                   Mc2CoreType::ON_CUBE_AND_VECTOR, false, false);
    } else if (TILING_KEY_IS(1000000000050102011)) {
        INVOKE_MC2_COMM_FP8_MIXED_CALC_910_OP_IMPL(Mc2QuantBatchMatmulV3::Mc2QuantBmmPertokenRegbaseKernel,
                                                   Mc2CoreType::ON_CUBE_AND_VECTOR, false, true);
    }
#endif
    if (TILING_KEY_IS(1000000000000000001)) {
        INVOKE_MC2_QUANT_910_OP_IMPL(AscendC::MatMulASWKernel, Mc2CoreType::ON_CUBE_AND_VECTOR, uint64_t, false, true);
    } else if (TILING_KEY_IS(1000000000000000000)) {
        INVOKE_MC2_QUANT_910_OP_IMPL(AscendC::MatMulASWKernel, Mc2CoreType::ON_CUBE_AND_VECTOR, uint64_t, false, false);
    } else if (TILING_KEY_IS(1000000000000002000)) {
        INVOKE_BATCH_MATMUL_QUANT_PERTOKEN_IMPL(
            Mc2QuantBatchMatmulV3::Mc2QuantBmmPertokenRegbaseKernel, Mc2CoreType::ON_CUBE_AND_VECTOR, float, false, false);
    } else if (TILING_KEY_IS(1000000000000002001)) {
        INVOKE_BATCH_MATMUL_QUANT_PERTOKEN_IMPL(
            Mc2QuantBatchMatmulV3::Mc2QuantBmmPertokenRegbaseKernel, Mc2CoreType::ON_CUBE_AND_VECTOR, float, false, true);
    } else if (TILING_KEY_IS(1000000000000004000)) {
        INVOKE_MC2_QUANT_PERBLOCK_910_OP_IMPL(
            Mc2QuantBatchMatmulV3::MatMulPerBlockASW, Mc2CoreType::ON_CUBE_AND_VECTOR, false, false);
    } else if (TILING_KEY_IS(1000000000000004001)) {
        INVOKE_MC2_QUANT_PERBLOCK_910_OP_IMPL(
            Mc2QuantBatchMatmulV3::MatMulPerBlockASW, Mc2CoreType::ON_CUBE_AND_VECTOR, false, true);
    }
#endif
}