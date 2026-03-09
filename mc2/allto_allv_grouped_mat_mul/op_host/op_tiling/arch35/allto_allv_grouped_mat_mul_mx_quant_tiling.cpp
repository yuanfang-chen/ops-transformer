/* *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file allto_allv_grouped_mat_mul_mx_quant_tiling.cc
 * \brief
 */

#include <string>
#include <numeric>
#include <climits>
#include "mc2_hcom_topo_info.h"
#include "mc2_log.h"
#include "context_util.h"
#include "tiling/matmul_formulaic_tiling.h"
#include "tiling/hccl_formulaic_tiling.h"
#include "graph/utils/type_utils.h"
#include "register/op_def_registry.h"
#include "tiling/mc2_tiling_utils.h"
#include "register/op_impl_registry.h"
#include "tiling_base/tiling_templates_registry.h"
#include "../allto_allv_grouped_mat_mul_tiling_base.h"
#include "../../../op_kernel/allto_allv_grouped_mat_mul_tiling.h"
#include "allto_allv_grouped_mat_mul_mx_quant_tiling.h"
#include "allto_allv_grouped_mat_mul_quant_tiling_base.h"

using namespace ge;
using namespace AscendC;
using namespace Ops::Transformer::OpTiling;

namespace optiling {
bool AlltoAllvGmmMXQuantTiling::IsCapable()
{
    // support fp8_e5m2 or fp8_e4m3
    if (gmmXDataType_ != ge::DT_FLOAT8_E5M2 || 
        gmmXDataType_ != ge::DT_FLOAT8_E4M3FN) {
        return false;
    }
    if (gmmXDataType_ != ge::DT_FLOAT8_E5M2 || 
        gmmXDataType_ != ge::DT_FLOAT8_E4M3FN) {
        return false;
    }
    OP_LOGD(context_->GetNodeName(), "AlltoAllvGmmMXQuantTiling is capable.");
    return true;
}

ge::graphStatus AlltoAllvGmmMXQuantTiling::DoLibApiTiling()
{
    OP_LOGD(context_->GetNodeName(), "start DoLibApiTiling.");
    uint64_t maxMSize = 0;
    uint64_t mSize = 0;
    for (uint64_t expertIdx = 0; expertIdx < e_; expertIdx++) {
        mSize = 0;
        for (uint64_t rankIdx = 0; rankIdx < epWorldSize_; rankIdx++) {
            mSize += recvCounts[rankIdx * e_ + expertIdx];
        }
        maxMSize = std::max(mSize, maxMSize);
    }
    if (maxMSize != 0) {
        auto &gmmQuantTilingData = tilingData->gmmQuantTilingData;
        SetGMMQuantParams(gmmQuantTilingData);
        SetTilingArray(gmmQuantTilingData, maxMSize, n1_, h1_);
        SetTilingParams(gmmQuantTilingData, maxMSize, n1_, h1_);
        SetTilingMxTypeParam(gmmQuantTilingData);
        PrintGMMQuantTilingData(gmmQuantTilingData);
    }
    if (bs_ != 0) {
        auto &mmQuantTilingData = tilingData->mmQuantTilingData;
        SetGMMQuantParams(mmQuantTilingData);
        SetTilingArray(mmQuantTilingData, bs_, n2_, h2_);
        SetTilingParams(mmQuantTilingData, bs_, n2_, h2_);
        SetTilingMxTypeParam(gmmQuantTilingData);
        PrintGMMQuantTilingData(mmQuantTilingData);
    }
    OP_LOGD(context_->GetNodeName(), "end DoLibApiTiling.");
    return ge::GRAPH_SUCCESS;
}

uint64_t AlltoAllvGmmMXQuantTiling::GetTilingKey() const
{
    uint64_t tilingKey = GET_TPL_TILING_KEY(ADD_TPL_FP8_E4M3_E5M2, hasSharedExpertFlag_, transGmmWeight_, transMmWeight_);
    return tilingKey;
}

void AlltoAllvGmmMXQuantTiling::SetGMMQuantParams(
    Mc2GroupedMatmulTilingData::GMMQuantTilingData &gmmQuantTilingData) const
{
    gmmQuantTilingData.gmmQuantParams.groupNum = SINGLE_GROUP_NUM;
    gmmQuantTilingData.gmmQuantParams.activeType = GMM_ACT_TYPE_NONE;
    gmmQuantTilingData.gmmQuantParams.aQuantMode = MX_PERGROUP_MODE;
    gmmQuantTilingData.gmmQuantParams.bQuantMode = MX_PERGROUP_MODE;
    gmmQuantTilingData.gmmQuantParams.singleX = 0;
    gmmQuantTilingData.gmmQuantParams.singleW = 0;
    gmmQuantTilingData.gmmQuantParams.singleY = 0;
    gmmQuantTilingData.gmmQuantParams.groupType = 0;
    gmmQuantTilingData.gmmQuantParams.groupListType = 1;
    gmmQuantTilingData.gmmQuantParams.hasBias = 0;
    gmmQuantTilingData.gmmQuantParams.reserved = 0;
}

void AlltoAllvGmmMXQuantTiling::SetTilingParams(Mc2GroupedMatmulTilingData::GMMQuantTilingData &gmmQuantTilingData, uint64_t M, uint64_t N, uint64_t K) const
{
    auto &mm = gmmQuantTilingData.mmTilingData;

    mm.M = M;
    mm.N = N;
    mm.Ka = K;
    mm.Kb = K;
    mm.usedCoreNum = aicCoreNum_;
    mm.isBias = 0;
    mm.dbL0A = DOUBLE_BUFFER;
    mm.dbL0B = DOUBLE_BUFFER;

    mm.baseM = std::min(static_cast<int32_t>(M), static_cast<int32_t>(BASIC_BLOCK_SIZE_256));
    mm.baseM = Ops::Base::CeilAlign(mm.baseM, static_cast<int32_t>(CUBE_BLOCK));
    mm.baseN = std::min(static_cast<int32_t>(N), static_cast<int32_t>(BASIC_BLOCK_SIZE_256));
    mm.baseN = Ops::Base::CeilAlign(mm.baseN. static_cast<uint32_t>(MX_BASIC_FACTOR))
    mm.baseK = std::min(static_cast<int32_t>(K), static_cast<int32_t>(BASIC_BLOCK_SIZE_128));
    mm.baseK = Ops::Base::CeilAlign(mm.baseK, static_cast<int32_t>(MX_BASIC_FACTOR));

    mm.singleCoreM = std::min(static_cast<int32_t>(M), mm.baseM);
    mm.singleCoreN = std::min(static_cast<int32_t>(N), mm.baseN);
    mm.singleCoreK = K;

    uint64_t l0cRequired = static_cast<uint64_t>(mm.baseM) * mm.baseN * DATA_SIZE_L0C * DB_SIZE;
    mm.dbL0C = (l0cRequired <= l0cSize_) ? DB_SIZE : 1;

    mm.iterateOrder = 0U;

    uint64_t baseASize = static_cast<uint64_t>(mm.baseM) * mm.baseK;
    uint64_t baseBSize = static_cast<uint64_t>(mm.baseN) * mm.baseK;
    uint64_t baseScaleASize = Ops::Base::CeilDiv(mm.baseK, MX_BASIC_FACTOR);
    baseScaleASize = Ops::Base::CeilAlign(baseScaleASize, 2UL) * mm.baseM;
    uint64_t baseScaleBSize = Ops::Base::CeilDiv(mm.baseK, MX_BASIC_FACTOR) * mm.baseN;
    baseScaleBSize = Ops::Base::CeilAlign(baseScaleBSize, 2UL) * mm.baseN;
    uint64_t baseL1Size = baseASize + baseBSize + baseScaleASize + baseScaleBSize;

    OP_TILING_CHECK(baseL1Size == 0, OP_LOGW(context_->GetNodeName(), "baseL1Size cannot be zero."), return );

    uint64_t leftL1Size = l1Size_;

    uint64_t depthInit = GetDepthA1B1(leftL1Size, baseL1Size, 1UL);
    uint64_t leftL1SizeByDepthInit = leftL1Size - depthInit * baseL1Size;
    uint64_t depthASec = GetDepthA1B1(leftL1SizeByDepthInit, (baseASize + baseScaleASize) * depthInit, depthInit);
    uint64_t depthBSec = GetDepthA1B1(leftL1SizeByDepthInit, (baseBSize + baseScaleBSize) * depthInit, depthInit);
    mm.depthA1 = std::max(depthASec,  depthBSec);
    mm.depthB1 = mm.depthA1;
    if (mm.depthA1 * baseL1Size > leftL1Size) {
        mm.depthA1 = depthASec >= depthBSec ? depthASec : depthInit;
        mm.depthB1 = depthASec < depthBSec ? depthBSec : depthInit;
    }

    mm.stepKa = (mm.depthA1 > 1) ? (mm.depthA1 / DB_SIZE) : 1;
    mm.stepKb = (mm.depthB1 > 1) ? (mm.depthB1 / DB_SIZE) : 1;

    OP_TILING_CHECK(mm.baseK == 0, OP_LOGW(context_->GetNodeName(), "baseK cannot be zero."), return );

    if (mm.stepKa * mm.baseK > mm.Ka) {
        mm.stepKa = Ops::Base::CeilDiv(mm.Ka, mm.baseK);
    }
    if (mm.stepKb * mm.baseK > mm.Kb) {
        mm.stepKb = Ops::Base::CeilDiv(mm.Kb, mm.baseK);
    }
    // G量化场景下，限制stepK最大为4, 防止issue queue阻塞
    mm.stepKa = std::min(mm.stepKa, static_cast<uint64_t>(4));
    mm.stepKb = std::min(mm.stepKb, static_cast<uint64_t>(4));
    if (mm.stepKa >= mm.stepKb && mm.stepKa * mm.baseK < mm.Ka) {
        mm.stepKa = mm.stepKa / mm.stepKb * mm.stepKb;
    }
    if (mm.stepKb > mm.stepKa && mm.stepKb * mm.baseK < imm.Kb) {
        mm.stepKb = mm.stepKb / mm.stepKa * mm.stepKa;
    }
    mm.depthA1 = mm.stepKa * DB_SIZE;
    mm.depthB1 = mm.stepKb * DB_SIZE;

    mm.stepM = 1;
    mm.stepN = 1;
}

void AlltoAllvGmmMXQuantTiling::SetTilingMxTypeParam(Mc2GroupedMatmulTilingData::GMMQuantTilingData &gmmQuantTilingData)
{
    auto &mm = gmmQuantTilingData.mmTilingData;

    uint64_t baseASize = static_cast<uint64_t>(mm.baseM) * mm.baseK;
    uint64_t baseBSize = static_cast<uint64_t>(mm.baseN) * mm.baseK;
    uint64_t baseScaleASize = Ops::Base::CeilDiv(mm.baseK, MX_BASIC_FACTOR);
    baseScaleASize = Ops::Base::CeilAlign(baseScaleASize, 2UL) * mm.baseM;
    uint64_t baseScaleBSize = Ops::Base::CeilDiv(mm.baseK, MX_BASIC_FACTOR) * mm.baseN;
    baseScaleBSize = Ops::Base::CeilAlign(baseScaleBSize, 2UL) * mm.baseN;

    uint64_t leftL1Size = l1Size_ - (mm.depthA1 * baseASize + mm.depthB1 * baseBSize);
    uint32_t scaleInit = static_cast<uint32_t>(leftL1Size /
        (mm.depthA1 * baseScaleASize + mm.depthB1 * baseScaleBSize));

    // 计算scaleFactorA, scaleFactorB
    // 来自K轴的约束
    if (baseScaleASize == 0) {
        baseScaleASize = 1;
    }
    uint32_t scaleFactorAMax = std::min(static_cast<uint32_t>(MTE2_MIN_LOAD_SIZE / baseScaleASize), SCALER_FACTOR_MAX);
    if (baseScaleBSize == 0) {
        baseScaleBSize = 1;
    }
    uint32_t scaleFactorBMax = std::min(static_cast<uint32_t>(MTE2_MIN_LOAD_SIZE / baseScaleBSize), SCALER_FACTOR_MAX);
    uint32_t scaleFactorA = static_cast<uint32_t>(K / (mm.stepKa * mm.baseK));
    uint32_t scaleFactorB = static_cast<uint32_t>(K / (mm.stepKb * mm.baseK));
    scaleFactorA = std::max(SCALER_FACTOR_MIN, scaleFactorA);
    scaleFactorB = std::max(SCALER_FACTOR_MIN, scaleFactorB);
    scaleFactorA = std::min(scaleFactorAMax, scaleFactorA);
    scaleFactorB = std::min(scaleFactorBMax, scaleFactorB);
    // 来自L1 size 的约束
    if (scaleFactorA <= scaleInit && scaleFactorB > scaleInit) {
        leftL1Size -= (scaleFactorA * mm.depthA1 * baseScaleASize);
        scaleFactorB = std::min(static_cast<uint32_t>(leftL1Size / (mm.depthB1 * baseScaleBSize)), scaleFactorB);
    } else if (scaleFactorB <= scaleInit && scaleFactorA > scaleInit) {
        leftL1Size -= (scaleFactorB * mm.depthB1 * baseScaleBSize);
        scaleFactorA = std::min(static_cast<uint32_t>(leftL1Size / (mm.depthA1 * baseScaleASize)), scaleFactorA);
    } else if (scaleFactorA > scaleInit && scaleFactorB > scaleInit) {
        leftL1Size -= (scaleInit * mm.depthB1 * baseScaleBSize + scaleInit * mm.depthA1 * baseScaleASize);
        uint32_t scaleASec = std::min(static_cast<uint32_t>(leftL1Size / (mm.depthA1 * baseScaleASize)), scaleFactorA - scaleInit);
        uint32_t scaleBSec = std::min(static_cast<uint32_t>(leftL1Size / (mm.depthB1 * baseScaleBSize)), scaleFactorB - scaleInit);
        scaleFactorA = scaleASec >= scaleBSec ? (scaleASec + scaleInit) : scaleInit;
        scaleFactorB = scaleASec < scaleBSec ? (scaleBSec + scaleInit) : scaleInit;
    }
    if (scaleFactorA >= SCALER_FACTOR_MIN && scaleFactorA <= SCALER_FACTOR_MAX &&
        scaleFactorB >= SCALER_FACTOR_MIN && scaleFactorB <= SCALER_FACTOR_MAX) {
        mm.mxTypePara = (SCALER_FACTOR_DEFAULT << SCALER_FACTOR_N_BIT) +
            (SCALER_FACTOR_DEFAULT << SCALER_FACTOR_M_BIT) + (scaleFactorB << SCALER_FACTOR_B_BIT) + scaleFactorA;
    } else {
        mm.mxTypePara = (SCALER_FACTOR_DEFAULT << SCALER_FACTOR_N_BIT) +
            (SCALER_FACTOR_DEFAULT << SCALER_FACTOR_M_BIT) + (SCALER_FACTOR_DEFAULT << SCALER_FACTOR_B_BIT) +
            SCALER_FACTOR_DEFAULT;
    }
}

uint64_t AlltoAllvGmmMXQuantTiling::GetDepthA1B1(uint64_t leftSize, uint64_t perDepthSize, uint64_t baseKSize, uint64_t depthInit)
{
    if (depthInit > 1UL && perDepthSize > DB_SIZE * MTE2_MIN_LOAD_SIZE) {
        return depthInit;
    }
    uint64_t depthScale = leftSize / perDepthSize;
    if (depthInit > 1UL) {
        while (((depthScale * baseKSize) % BASIC_BLOCK_SIZE_512 != 0) && 
                ((depthScale * baseKSize) > BASIC_BLOCK_SIZE_512)) {
            depthScale -= 1;
        }
        if (((depthScale * baseKSize) % BASIC_BLOCK_SIZE_512 != 0) && 
            ((depthScale * baseKSize) >= BASIC_BLOCK_SIZE_256)) {
                if (baseKSize == 0) {
                    baseKSize = 1;
            }
        }
        depthScale = BASIC_BLOCK_SIZE_256 / baseKSize;
    } else {
        constexpr uint64_t index = 2;
        depthScale = 1UL;
        while (depthScale * (perDepthSize) < leftSize) {
            depthScale *= index;
        }
        depthScale == depthScale == 1UL ? depthScale : depthScale / index;
    }
    return depthInit * depthScale;
}
// change
ge::graphStatus AlltoAllvGmmMXQuantTiling::CheckGmmDType() const
{
    OP_LOGD(context_->GetNodeName(), "start CheckGmmDType.");
    OP_TILING_CHECK((context_->GetInputDesc(GMM_X_INDEX) == nullptr) ||
        (context_->GetInputDesc(GMM_WEIGHT_INDEX) == nullptr),
        OP_LOGE(context_->GetNodeName(), "GetInputDesc gmmX or gmmWeight returned null."), return ge::GRAPH_FAILED);
    auto gmmXDataType = context_->GetInputDesc(GMM_X_INDEX)->GetDataType();
    OP_TILING_CHECK((gmmXDataType != ge::DT_FLOAT8_E5M2) ||
        (gmmXDataType != ge::DT_FLOAT8_E4M3FN),
        OP_LOGE(context_->GetNodeName(), "Unsupported dataType, gmmX support fp8_e5m2 or fp8_e4m3."), return ge::GRAPH_FAILED);
    auto gmmWeightDataType = context_->GetInputDesc(GMM_WEIGHT_INDEX)->GetDataType();
    OP_TILING_CHECK((gmmWeightDataType != ge::DT_FLOAT8_E5M2) ||
        (gmmWeightDataType != ge::DT_FLOAT8_E4M3FN),
        OP_LOGE(context_->GetNodeName(), "Unsupported dataType, gmmWeight support fp8_e5m2 or fp8_e4m3."),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(context_->GetOptionalInputDesc(GMM_X_SCALE_INDEX) == nullptr,
        OP_LOGE(context_->GetNodeName(), "GetInputDesc gmmXScale returned null."), return ge::GRAPH_FAILED);
    auto gmmXScaleDataType = context_->GetOptionalInputDesc(GMM_X_SCALE_INDEX)->GetDataType();
    OP_TILING_CHECK(gmmXScaleDataType != ge::DT_FLOAT8_E8M0,
        OP_LOGE(context_->GetNodeName(), "Unsupported dataType, gmmXScale only support fp8_e8m0."),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(context_->GetOptionalInputDesc(GMM_WEIGHT_SCALE_INDEX) == nullptr,
        OP_LOGE(context_->GetNodeName(), "GetInputDesc gmmWeightScale returned null."), return ge::GRAPH_FAILED);
    auto gmmWeightScaleDataType = context_->GetOptionalInputDesc(GMM_WEIGHT_SCALE_INDEX)->GetDataType();
    OP_TILING_CHECK(gmmWeightScaleDataType != ge::DT_FLOAT8_E8M0,
        OP_LOGE(context_->GetNodeName(), "Unsupported dataType, gmmWeightScale only support fp8_e8m0."),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(context_->GetOutputDesc(OUTPUT_GMM_Y_INDEX) == nullptr,
        OP_LOGE(context_->GetNodeName(), "GetOutputDesc y returned null."), return ge::GRAPH_FAILED);
    auto gmmYDataType = context_->GetOutputDesc(OUTPUT_GMM_Y_INDEX)->GetDataType();
    OP_TILING_CHECK(gmmYDataType != ge::DT_FLOAT16 && gmmYDataType != ge::DT_BF16,
        OP_LOGE(context_->GetNodeName(), "Unsupported dataType, gmmY only support float16 and bfloat16."),
        return ge::GRAPH_FAILED);
    if (permuteOutFlag_) {
        // check permuteOut dtype
        tilingData->isPermuteOut = true;
        OP_TILING_CHECK(context_->GetOutputDesc(OUTPUT_PERMUTE_OUT_INDEX) == nullptr,
            OP_LOGE(context_->GetNodeName(), "GetOutputDesc permuteOut returned null."), return ge::GRAPH_FAILED);
        auto permuteOutDataType = context_->GetOutputDesc(OUTPUT_PERMUTE_OUT_INDEX)->GetDataType();
        OP_TILING_CHECK(permuteOutDataType != gmmXDataType,
            OP_LOGE(context_->GetNodeName(), "Unsupported dataType, permuteOut only support hifloat8."),
            return ge::GRAPH_FAILED);
    }
    OP_LOGD(context_->GetNodeName(), "end CheckGmmDType.");
    return ge::GRAPH_SUCCESS;
}
// change
ge::graphStatus AlltoAllvGmmMXQuantTiling::CheckMmDType() const
{
    OP_LOGD(context_->GetNodeName(), "start CheckMmDType.");
    if (!hasSharedExpertFlag_) {
        return ge::GRAPH_SUCCESS;
    }
    OP_TILING_CHECK((context_->GetOptionalInputDesc(MM_X_INDEX) == nullptr) ||
        (context_->GetOptionalInputDesc(MM_WEIGHT_INDEX) == nullptr),
        OP_LOGE(context_->GetNodeName(), "GetOptionalInputDesc mmX or mmWeight returned null."),
        return ge::GRAPH_FAILED);
    auto mmXDataType = context_->GetOptionalInputDesc(MM_X_INDEX)->GetDataType();
    OP_TILING_CHECK((mmXDataType != ge::DT_FLOAT8_E5M2) ||
        (mmXDataType != ge::DT_FLOAT8_E4M3FN),
        OP_LOGE(context_->GetNodeName(), "Unsupported dataType, mmX support fp8_e5m2 or fp8_e4m3."), return ge::GRAPH_FAILED);
    auto mmWeightDataType = context_->GetOptionalInputDesc(MM_WEIGHT_INDEX)->GetDataType();
    OP_TILING_CHECK((mmWeightDataType != ge::DT_FLOAT8_E5M2) ||
        (mmWeightDataType != ge::DT_FLOAT8_E4M3FN),
        OP_LOGE(context_->GetNodeName(), "Unsupported dataType, mmWeight support fp8_e5m2 or fp8_e4m3."),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(context_->GetOptionalInputDesc(MM_X_SCALE_INDEX) == nullptr,
        OP_LOGE(context_->GetNodeName(), "GetOptionalInputDesc mmXScale returned null."), return ge::GRAPH_FAILED);
    auto mmXScaleDataType = context_->GetOptionalInputDesc(MM_X_SCALE_INDEX)->GetDataType();
    OP_TILING_CHECK(mmXScaleDataType != ge::DT_FLOAT8_E8M0,
        OP_LOGE(context_->GetNodeName(), "Unsupported dataType, mmXScale only support fp8_e8m0."),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(context_->GetOptionalInputDesc(MM_WEIGHT_SCALE_INDEX) == nullptr,
        OP_LOGE(context_->GetNodeName(), "GetOptionalInputDesc mmWeightScale returned null."), return ge::GRAPH_FAILED);
    auto mmWeightScaleDataType = context_->GetOptionalInputDesc(MM_WEIGHT_SCALE_INDEX)->GetDataType();
    OP_TILING_CHECK(mmWeightScaleDataType != ge::DT_FLOAT8_E8M0,
        OP_LOGE(context_->GetNodeName(), "Unsupported dataType, mmWeightScale only support fp8_e8m0."),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(context_->GetOutputDesc(OUTPUT_MM_Y_INDEX) == nullptr,
        OP_LOGE(context_->GetNodeName(), "GetOutputDesc y returned null."), return ge::GRAPH_FAILED);
    auto mmYDataType = context_->GetOutputDesc(OUTPUT_MM_Y_INDEX)->GetDataType();
    OP_TILING_CHECK(mmYDataType != ge::DT_FLOAT16 && mmYDataType != ge::DT_BF16,
        OP_LOGE(context_->GetNodeName(), "Unsupported dataType, mmY only support float16 and bfloat16."),
        return ge::GRAPH_FAILED);
    OP_LOGD(context_->GetNodeName(), "end CheckMmDType.");
    return ge::GRAPH_SUCCESS;
}
// change
ge::graphStatus AlltoAllvGmmMXQuantTiling::CheckQuantMode() const
{
    OP_LOGD(context_->GetNodeName(), "start CheckQuantMode.");
    // gmmXQuantMode
    OP_TILING_CHECK(gmmXQuantModePtr_ == nullptr,
        OP_LOGE(context_->GetNodeName(), "gmmXQuantMode attr can not be null."), return ge::GRAPH_FAILED);
    auto gmmXQuantMode = *gmmXQuantModePtr_;
    OP_TILING_CHECK(gmmXQuantMode != MX_PERGROUP_QUANT_MODE,
        OP_LOGE(context_->GetNodeName(), "gmmXQuantMode should be mx pergroup mode."), return ge::GRAPH_FAILED);
    // gmmWeightQuantMode
    OP_TILING_CHECK(gmmWeightQuantModePtr_ == nullptr,
        OP_LOGE(context_->GetNodeName(), "gmmWeightQuantMode attr can not be null."), return ge::GRAPH_FAILED);
    auto gmmWeightQuantMode = *gmmWeightQuantModePtr_;
    OP_TILING_CHECK(gmmWeightQuantMode != MX_PERGROUP_QUANT_MODE,
        OP_LOGE(context_->GetNodeName(), "gmmWeightQuantMode should be mx pergroup mode."),
        return ge::GRAPH_FAILED);
    // check gmmXScale shape
    OP_TILING_CHECK(context_->GetOptionalInputShape(GMM_X_SCALE_INDEX) == nullptr,
        OP_LOGE(context_->GetNodeName(), "gmmXScale input shape can not be null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(context_->GetOptionalInputShape(GMM_X_SCALE_INDEX)->GetStorageShape().GetDimNum() != DIM_THREE,
        OP_LOGE(context_->GetNodeName(), "When mx pergroup mode, gmmXScale input shape should be [3]"), return ge::GRAPH_FAILED);
    auto expectedGHValue = Ops::Base::CeilDiv(h1_, MX_BASIC_FACTOR);
    auto gmmXScaleBSKDim = static_cast<uint64_t>(context_->GetOptionalInputShape(GMM_X_SCALE_INDEX)->GetStorageShape().GetDimNum(DIM_ONE));
    auto gmmXScaleHDim = static_cast<uint64_t>(context_->GetOptionalInputShape(GMM_X_SCALE_INDEX)->GetStorageShape().GetDimNum(DIM_TWO));
    OP_TILING_CHECK((gmmXScaleBSKDim != bsk_) || (gmmXScaleHDim != expectedGHValue), 
        OP_LOGE(context_->GetNodeName(), "When mx pergroup mode, the expected shape of gmmxscale is (%lu, %lu, 2) but the aclual \
        is (%lu, %lu, 2)", bsk_, expectedGHValue, gmmXScaleBSKDim, gmmXScaleHDim), return ge::GRAPH_FAILED);    
    // check gmmWeightScale shape
    OP_TILING_CHECK(context_->GetOptionalInputShape(GMM_WEIGHT_SCALE_INDEX) == nullptr,
        OP_LOGE(context_->GetNodeName(), "gmmWeightScale input shape can not be null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(context_->GetOptionalInputShape(GMM_WEIGHT_SCALE_INDEX)->GetStorageShape().GetDimNum() != DIM_THREE,
        OP_LOGE(context_->GetNodeName(), "When mx pergroup mode, gmmXScale input shape should be [3]."), return ge::GRAPH_FAILED);
    auto gmmWeightScaleDimOne = context_->GetOptionalInputShape(GMM_WEIGHT_SCALE_INDEX)->GetStorageShape().GetDimNum(DIM_ONE);
    auto gmmWeightScaleDimTwo = context_->GetOptionalInputShape(GMM_WEIGHT_SCALE_INDEX)->GetStorageShape().GetDimNum(DIM_TWO);
    auto gmmWeightScaleHDim = static_cast<uint64_t>(transGmmWeight_ ? gmmWeightScaleDimTwo : gmmWeightScaleDimOne);
    auto gmmWeightScaleNDim = static_cast<uint64_t>(transGmmWeight_ ? gmmWeightScaleDimOne : gmmWeightScaleDimTwo);
    OP_TILING_CHECK((gmmWeightScaleNDim != n1_) || (gmmWeightScaleHDim != expectedGHValue), 
        OP_LOGE(context_->GetNodeName(), "When mx pergroup mode, the expected shape of gmmxscale is (%lu, %lu, 2) but the aclual \
        is (%lu, %lu, 2)", expectedGHValue, n1_, gmmWeightScaleHDim, gmmWeightScaleNDim), return ge::GRAPH_FAILED); 
    if (hasSharedExpertFlag_) {
        CheckShareExpertQuantMode();
    }
    OP_LOGD(context_->GetNodeName(), "end CheckQuantMode.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvGmmMXQuantTiling::CheckShareExpertQuantMode() const
{
    OP_TILING_CHECK(mmXQuantModePtr_ == nullptr,
        OP_LOGE(context_->GetNodeName(), "mmXQuantMode attr can not be null."), return ge::GRAPH_FAILED);
    auto mmXQuantMode = *mmXQuantModePtr_;
    OP_TILING_CHECK(mmXQuantMode != MX_PERGROUP_QUANT_MODE,
        OP_LOGE(context_->GetNodeName(), "mmXQuantMode should be mx pergroup mode."), return ge::GRAPH_FAILED);
    // check mmxScale shape
    OP_TILING_CHECK(context_->GetOptionalInputShape(MM_X_SCALE_INDEX) == nullptr,
        OP_LOGE(context_->GetNodeName(), "mmXScale input shape can not be null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(context_->GetOptionalInputShape(MM_X_SCALE_INDEX)->GetStorageShape().GetDimNum() != DIM_THREE,
        OP_LOGE(context_->GetNodeName(), "When mx pergroup mode, mmXScale input shape should be [3]"), return ge::GRAPH_FAILED);
    auto expectedHValue = Ops::Base::CeilDiv(h2_, MX_BASIC_FACTOR);
    auto mmXScaleBSDim = static_cast<uint64_t>(context_->GetOptionalInputShape(MM_X_SCALE_INDEX)->GetStorageShape().GetDimNum(DIM_ONE));
    auto mmXScaleHDim = static_cast<uint64_t>(context_->GetOptionalInputShape(MM_X_SCALE_INDEX)->GetStorageShape().GetDimNum(DIM_TWO));
    OP_TILING_CHECK((mmXScaleBSDim != bs_) || (mmXScaleHDim != expectedHValue), 
        OP_LOGE(context_->GetNodeName(), "When mx pergroup mode, the expected shape of gmmxscale is (%lu, %lu, 2) but the aclual \
        is (%lu, %lu, 2)", bs_, expectedHValue, mmXScaleBSDim, mmXScaleHDim), return ge::GRAPH_FAILED);  
    // mmWeightQuantMode
    OP_TILING_CHECK(mmWeightQuantModePtr_ == nullptr,
        OP_LOGE(context_->GetNodeName(), "mmWeightQuantMode attr can not be null."), return ge::GRAPH_FAILED);
    auto mmWeightQuantMode = *mmWeightQuantModePtr_;
    OP_TILING_CHECK(mmWeightQuantMode != MX_PERGROUP_QUANT_MODE,
        OP_LOGE(context_->GetNodeName(), "mmWeightQuantMode should be mx pergroup mode."),
        return ge::GRAPH_FAILED);
    // check mmWeightScale shape
    OP_TILING_CHECK(context_->GetOptionalInputShape(MM_WEIGHT_SCALE_INDEX) == nullptr,
    OP_LOGE(context_->GetNodeName(), "mmWeightScale input shape can not be null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(context_->GetOptionalInputShape(MM_WEIGHT_SCALE_INDEX)->GetStorageShape().GetDimNum() != DIM_THREE,
        OP_LOGE(context_->GetNodeName(), "When mx pergroup mode, mmXScale input shape should be [3]."), return ge::GRAPH_FAILED);
    auto mmWeightScaleDimOne = context_->GetOptionalInputShape(MM_WEIGHT_SCALE_INDEX)->GetStorageShape().GetDimNum(DIM_ONE);
    auto mmWeightScaleDimTwo = context_->GetOptionalInputShape(MM_WEIGHT_SCALE_INDEX)->GetStorageShape().GetDimNum(DIM_TWO);
    auto mmWeightScaleHDim = static_cast<uint64_t>(transMmWeight_ ? mmWeightScaleDimTwo : mmWeightScaleDimOne);
    auto mmWeightScaleNDim = static_cast<uint64_t>(transMmWeight_ ? mmWeightScaleDimOne : mmWeightScaleDimTwo);
    OP_TILING_CHECK((mmWeightScaleNDim != n2_) || (mmWeightScaleHDim != expectedHValue), 
    OP_LOGE(context_->GetNodeName(), "When mx pergroup mode, the expected shape of gmmxscale is (%lu, %lu, 2) but the aclual \
        is (%lu, %lu, 2)", expectedHValue, n1_, mmWeightScaleHDim, mmWeightScaleNDim), return ge::GRAPH_FAILED); 
    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(AlltoAllvGroupedMatMul, AlltoAllvGmmMXQuantTiling, 2);
} // namespace optiling