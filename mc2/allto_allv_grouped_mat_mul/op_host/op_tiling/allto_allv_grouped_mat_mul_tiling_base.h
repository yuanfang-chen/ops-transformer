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
 * \file allto_allv_grouped_mat_mul_tiling_base.h
 * \brief
 */
#pragma once

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "tiling_base/tiling_base.h"
#include "tiling/mc2_tiling_struct.h"
#include "tiling/matmul_formulaic_tiling.h"
#include "mat_mul_v3/op_host/op_tiling/matmul_v3_tiling.h"
#include "tiling/mc2_tiling_utils.h"
#include "../../op_kernel/allto_allv_grouped_mat_mul_tiling.h"
#include "../../op_kernel/allto_allv_grouped_mat_mul_tiling_key.h"

namespace optiling {
constexpr uint32_t DATA_SIZE_L0C = 4;
constexpr uint64_t CUBE_BLOCK = 16;
constexpr uint64_t CUBE_REDUCE_BLOCK = 32;
constexpr uint32_t BASIC_BLOCK_SIZE_512 = 512;
constexpr uint32_t BASIC_BLOCK_SIZE_256 = 256;
constexpr uint32_t BASIC_BLOCK_SIZE_128 = 128;
constexpr uint32_t PERTENSOR_MODE = 1;
constexpr uint32_t SINGLE_GROUP_NUM = 1;
constexpr uint32_t GMM_ACT_TYPE_NONE = 0;
constexpr uint64_t DB_SIZE = 2UL;

constexpr uint32_t GMM_X_INDEX = 0U;
constexpr uint32_t GMM_WEIGHT_INDEX = 1U;
constexpr uint32_t SEND_COUNTS_TENSOR_INDEX = 2U;
constexpr uint32_t RECV_COUNTS_TENSOR_INDEX = 3U;
constexpr uint32_t MM_X_INDEX = 4U;
constexpr uint32_t MM_WEIGHT_INDEX = 5U;
constexpr uint32_t GMMX_SCALE_INDEX = 6U;
constexpr uint32_t GMMW_SCALE_INDEX = 7U;
constexpr uint32_t MMX_SCALE_INDEX = 8U;
constexpr uint32_t MMW_SCALE_INDEX = 9U;
constexpr uint32_t OUTPUT_Y_INDEX = 0U;
constexpr uint32_t OUTPUT_GMM_Y_INDEX = 0U;
constexpr uint32_t OUTPUT_MM_Y_INDEX = 1U;
constexpr uint32_t OUTPUT_PERMUTE_OUT_INDEX = 2U;

constexpr uint32_t DIM_TWO = 2;
constexpr uint32_t DIM_ONE = 1;
constexpr uint32_t DIM_THREE = 3;

constexpr uint32_t NUM_ZERO = 0;
constexpr uint32_t NUM_ONE = 1;
constexpr uint32_t NUM_TWO = 2;
constexpr uint32_t NUM_THREE = 3;
constexpr uint32_t NUM_EIGHT = 8;
constexpr uint32_t NUM_SIXTEEN = 16;
constexpr uint32_t NUM_THIRTYTWO = 32;
constexpr uint32_t NUM_SIXTYFOUR = 64;
constexpr uint32_t MAX_EXPERT_NUM = 256;
constexpr uint32_t MAX_BSK = 52428800;
constexpr uint32_t MAX_SHAPE_SIZE = 65536;
constexpr uint32_t MAX_SHARED_H_SHAPE_SIZE = 12288;

constexpr uint32_t ATTR_GROUP_INDEX = 0;
constexpr uint32_t ATTR_EP_WORLD_SIZE_INDEX = 1;
constexpr uint32_t ATTR_SEND_COUNTS_INDEX = 2;
constexpr uint32_t ATTR_RECV_COUNTS_INDEX = 3;
constexpr uint32_t ATTR_TRANS_GMM_WEIGHT_INDEX = 4;
constexpr uint32_t ATTR_TRANS_MM_WEIGHT_INDEX = 5;
constexpr uint32_t ATTR_PERMUTE_OUT_FLAG_INDEX = 6;

constexpr int64_t BEST_L1_PARTA = 256 * 1024;
constexpr int64_t BEST_L1_PARTB = 128 * 1024;
constexpr int64_t BEST_BASE_N = 256;
constexpr uint32_t UB_DIVIDE_NUM = 2;
constexpr uint32_t UB_CALSIZE_PER_BLOCK = 16 * 1024;
constexpr uint64_t DOUBLE_BUFFER_L0A_L0B = 2;
constexpr uint64_t DOUBLE_BUFFER_STEPKA_STEPKB = 2;
constexpr uint32_t SYS_WORKSPACE_SIZE = 16 * 1024 * 1024;
constexpr uint32_t MAX_TURN_NUM = 24;
constexpr int32_t MAX_BASE_K = 128;
constexpr uint64_t COMM_TILE = 8;  // Calculation of the number of data allocation times per card

inline const char *A_INNER_DEBUG = "[ERROR] AlltoAllvGroupedMatMul Tiling";

static inline uint32_t SixteenAlign(uint32_t a, bool up = false)
{
    if (up) {
        a += 15;  // 15: 16 bytes up-align
    }
    return a & ~15;  // ~15: 16 bytes down-align
}

static inline uint32_t Ceil(uint32_t a, uint32_t b)
{
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b;
}

static uint64_t GMMGetSizePlatForm(
    const platform_ascendc::CoreMemType memType, platform_ascendc::PlatformAscendC ascendcPlatform)
{
    uint64_t size = 0;
    ascendcPlatform.GetCoreMemSize(memType, size);
    return size;
}

struct PlatFormMemSize {
    uint64_t ubSize;
    uint64_t l1Size;
    uint64_t l0CSize;
    uint64_t l0ASize;
    uint64_t l0BSize;

    explicit PlatFormMemSize(platform_ascendc::PlatformAscendC ascendcPlatform)
        : ubSize(GMMGetSizePlatForm(platform_ascendc::CoreMemType::UB, ascendcPlatform)),
          l1Size(GMMGetSizePlatForm(platform_ascendc::CoreMemType::L1, ascendcPlatform)),
          l0CSize(GMMGetSizePlatForm(platform_ascendc::CoreMemType::L0_C, ascendcPlatform)),
          l0ASize(GMMGetSizePlatForm(platform_ascendc::CoreMemType::L0_A, ascendcPlatform)),
          l0BSize(GMMGetSizePlatForm(platform_ascendc::CoreMemType::L0_B, ascendcPlatform))
    {}
};

struct MMTilingParams {
    int32_t curMaxM;
    int32_t curMaxK;
    int32_t curMaxN;
    int32_t *curBaseM;
    int32_t *curBaseK;
    int32_t *curBaseN;
};

struct SetMMTilingParams {
    matmul_tiling::DataType matmulDtype;
    int32_t curMaxM;
    int32_t curMaxK;
    int32_t curMaxN;
    int32_t curBaseM;
    int32_t curBaseN;
    int32_t type;
};

static void PrintTilingDataGMM(::TCubeTiling msg)
{
    OP_LOGD(A_INNER_DEBUG, " gmmTilingData.usedCoreNum %d.", msg.usedCoreNum);
    OP_LOGD(A_INNER_DEBUG, " gmmTilingData.M %d.", msg.M);
    OP_LOGD(A_INNER_DEBUG, " gmmTilingData.N %d.", msg.N);
    OP_LOGD(A_INNER_DEBUG, " gmmTilingData.Ka %d.", msg.Ka);
    OP_LOGD(A_INNER_DEBUG, " gmmTilingData.Kb %d.", msg.Kb);
    OP_LOGD(A_INNER_DEBUG, " gmmTilingData.singleCoreM %d.", msg.singleCoreM);
    OP_LOGD(A_INNER_DEBUG, " gmmTilingData.singleCoreN %d.", msg.singleCoreN);
    OP_LOGD(A_INNER_DEBUG, " gmmTilingData.singleCoreK %d.", msg.singleCoreK);
    OP_LOGD(A_INNER_DEBUG, " gmmTilingData.baseM %d.", msg.baseM);
    OP_LOGD(A_INNER_DEBUG, " gmmTilingData.baseN %d.", msg.baseN);
    OP_LOGD(A_INNER_DEBUG, " gmmTilingData.baseK %d.", msg.baseK);
    OP_LOGD(A_INNER_DEBUG, " gmmTilingData.stepKa %d.", msg.stepKa);
    OP_LOGD(A_INNER_DEBUG, " gmmTilingData.stepKb %d.", msg.stepKb);
    OP_LOGD(A_INNER_DEBUG, " gmmTilingData.stepM %d.", msg.stepM);
    OP_LOGD(A_INNER_DEBUG, " gmmTilingData.stepN %d.", msg.stepN);
    OP_LOGD(A_INNER_DEBUG, " gmmTilingData.isBias %d.", msg.isBias);
    OP_LOGD(A_INNER_DEBUG, " gmmTilingData.transLength %d.", msg.transLength);
    OP_LOGD(A_INNER_DEBUG, " gmmTilingData.iterateOrder %d.", msg.iterateOrder);
    OP_LOGD(A_INNER_DEBUG, " gmmTilingData.dbL0A %d.", msg.dbL0A);
    OP_LOGD(A_INNER_DEBUG, " gmmTilingData.dbL0B %d.", msg.dbL0B);
    OP_LOGD(A_INNER_DEBUG, " gmmTilingData.dbL0C %d.", msg.dbL0C);
    OP_LOGD(A_INNER_DEBUG, " gmmTilingData.shareMode %d.", msg.shareMode);
    OP_LOGD(A_INNER_DEBUG, " gmmTilingData.shareL1Size %d.", msg.shareL1Size);
    OP_LOGD(A_INNER_DEBUG, " gmmTilingData.shareL0CSize %d.", msg.shareL0CSize);
    OP_LOGD(A_INNER_DEBUG, " gmmTilingData.shareUbSize %d.", msg.shareUbSize);
    OP_LOGD(A_INNER_DEBUG, " gmmTilingData.batchM %d.", msg.batchM);
    OP_LOGD(A_INNER_DEBUG, " gmmTilingData.batchN %d.", msg.batchN);
    OP_LOGD(A_INNER_DEBUG, " gmmTilingData.singleBatchM %d.", msg.singleBatchM);
    OP_LOGD(A_INNER_DEBUG, " gmmTilingData.singleBatchN %d.", msg.singleBatchN);
}

static void PrintTilingDataMM(::TCubeTiling msg)
{
    OP_LOGD(A_INNER_DEBUG, " mmTilingData.usedCoreNum %d.", msg.usedCoreNum);
    OP_LOGD(A_INNER_DEBUG, " mmTilingData.M %d.", msg.M);
    OP_LOGD(A_INNER_DEBUG, " mmTilingData.N %d.", msg.N);
    OP_LOGD(A_INNER_DEBUG, " mmTilingData.Ka %d.", msg.Ka);
    OP_LOGD(A_INNER_DEBUG, " mmTilingData.Kb %d.", msg.Kb);
    OP_LOGD(A_INNER_DEBUG, " mmTilingData.singleCoreM %d.", msg.singleCoreM);
    OP_LOGD(A_INNER_DEBUG, " mmTilingData.singleCoreN %d.", msg.singleCoreN);
    OP_LOGD(A_INNER_DEBUG, " mmTilingData.singleCoreK %d.", msg.singleCoreK);
    OP_LOGD(A_INNER_DEBUG, " mmTilingData.baseM %d.", msg.baseM);
    OP_LOGD(A_INNER_DEBUG, " mmTilingData.baseN %d.", msg.baseN);
    OP_LOGD(A_INNER_DEBUG, " mmTilingData.baseK %d.", msg.baseK);
    OP_LOGD(A_INNER_DEBUG, " mmTilingData.stepKa %d.", msg.stepKa);
    OP_LOGD(A_INNER_DEBUG, " mmTilingData.stepKb %d.", msg.stepKb);
    OP_LOGD(A_INNER_DEBUG, " mmTilingData.stepM %d.", msg.stepM);
    OP_LOGD(A_INNER_DEBUG, " mmTilingData.stepN %d.", msg.stepN);
    OP_LOGD(A_INNER_DEBUG, " mmTilingData.isBias %d.", msg.isBias);
    OP_LOGD(A_INNER_DEBUG, " mmTilingData.transLength %d.", msg.transLength);
    OP_LOGD(A_INNER_DEBUG, " mmTilingData.iterateOrder %d.", msg.iterateOrder);
    OP_LOGD(A_INNER_DEBUG, " mmTilingData.dbL0A %d.", msg.dbL0A);
    OP_LOGD(A_INNER_DEBUG, " mmTilingData.dbL0B %d.", msg.dbL0B);
    OP_LOGD(A_INNER_DEBUG, " mmTilingData.dbL0C %d.", msg.dbL0C);
    OP_LOGD(A_INNER_DEBUG, " mmTilingData.shareMode %d.", msg.shareMode);
    OP_LOGD(A_INNER_DEBUG, " mmTilingData.shareL1Size %d.", msg.shareL1Size);
    OP_LOGD(A_INNER_DEBUG, " mmTilingData.shareL0CSize %d.", msg.shareL0CSize);
    OP_LOGD(A_INNER_DEBUG, " mmTilingData.shareUbSize %d.", msg.shareUbSize);
    OP_LOGD(A_INNER_DEBUG, " mmTilingData.batchM %d.", msg.batchM);
    OP_LOGD(A_INNER_DEBUG, " mmTilingData.batchN %d.", msg.batchN);
    OP_LOGD(A_INNER_DEBUG, " mmTilingData.singleBatchM %d.", msg.singleBatchM);
    OP_LOGD(A_INNER_DEBUG, " mmTilingData.singleBatchN %d.", msg.singleBatchN);
}

static void PrintCommonTilingInfo(AlltoAllvGmmCommonTilingInfo &commonTilingInfo)
{
    OP_LOGD(A_INNER_DEBUG, " commonTilingInfo.BSK %lu.", commonTilingInfo.BSK);
    OP_LOGD(A_INNER_DEBUG, " commonTilingInfo.BS %lu.", commonTilingInfo.BS);
    OP_LOGD(A_INNER_DEBUG, " commonTilingInfo.H1 %lu.", commonTilingInfo.H1);
    OP_LOGD(A_INNER_DEBUG, " commonTilingInfo.H2 %lu.", commonTilingInfo.H2);
    OP_LOGD(A_INNER_DEBUG, " commonTilingInfo.A %lu.", commonTilingInfo.A);
    OP_LOGD(A_INNER_DEBUG, " commonTilingInfo.N1 %lu.", commonTilingInfo.N1);
    OP_LOGD(A_INNER_DEBUG, " commonTilingInfo.N2 %lu.", commonTilingInfo.N2);
    OP_LOGD(A_INNER_DEBUG, " commonTilingInfo.epWorldSize %lu.", commonTilingInfo.epWorldSize);
    OP_LOGD(A_INNER_DEBUG, " commonTilingInfo.E_ep %lu.", commonTilingInfo.E_ep);
    OP_LOGD(A_INNER_DEBUG, " commonTilingInfo.commOut %lu.", commonTilingInfo.commOut);
    OP_LOGD(A_INNER_DEBUG, " commonTilingInfo.aivCoreNum %lu.", commonTilingInfo.aivCoreNum);
    OP_LOGD(A_INNER_DEBUG, " commonTilingInfo.aicCoreNum %lu.", commonTilingInfo.aicCoreNum);
    OP_LOGD(A_INNER_DEBUG, " commonTilingInfo.isGmmWeightTrans %d.", commonTilingInfo.isGmmWeightTrans);
    OP_LOGD(A_INNER_DEBUG, " commonTilingInfo.isMmWeightTrans %d.", commonTilingInfo.isMmWeightTrans);
    OP_LOGD(A_INNER_DEBUG, " commonTilingInfo.isSendCntsTensor %d.", commonTilingInfo.isSendCntsTensor);
    OP_LOGD(A_INNER_DEBUG, " commonTilingInfo.isRecvCntsTensor %d.", commonTilingInfo.isRecvCntsTensor);
    OP_LOGD(A_INNER_DEBUG, " commonTilingInfo.isPermuteOut %d.", commonTilingInfo.isPermuteOut);
    OP_LOGD(A_INNER_DEBUG, " commonTilingInfo.isNeedMM %d.", commonTilingInfo.isNeedMM);
}

class AlltoAllvGmmTilingBase : public Ops::Transformer::OpTiling::TilingBaseClass {
public:
    explicit AlltoAllvGmmTilingBase(gert::TilingContext *context)
        : Ops::Transformer::OpTiling::TilingBaseClass(context){};

protected:
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoLibApiTiling() override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;

    platform_ascendc::SocVersion socVersion_;
};
}  // namespace optiling