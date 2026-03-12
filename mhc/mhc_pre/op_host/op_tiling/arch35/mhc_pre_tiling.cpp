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
 * \file mainifold_constrained_hyper_connection_tiling.cpp
 * \brief
 */

#include "mhc_pre_tiling.h"
#include "tiling_base/tiling_templates_registry.h"
#include "register/op_def_registry.h"
#include "platform/platform_infos_def.h"
#include "err/ops_err.h"

namespace optiling {

using namespace Ops::Transformer::OpTiling;
const constexpr int64_t BSND_DIM_NUM = 4;
const constexpr int64_t TND_DIM_NUM = 3;
const constexpr uint32_t X_INDEX = 0;
const constexpr uint32_t PHI_INDEX = 1;
const constexpr uint32_t ALPHA_INDEX = 2;
const constexpr uint32_t BIAS_INDEX = 3;
const constexpr uint32_t GAMMA_INDEX = 4;

const constexpr uint32_t H_IN_INDEX = 0;
const constexpr uint32_t H_POST_INDEX = 1;
const constexpr uint32_t H_RES_INDEX = 2;
const constexpr uint32_t INV_RMS_INDEX = 3;
const constexpr uint32_t H_MIX_INDEX = 4;
const constexpr uint32_t H_PRE_INDEX = 5;

const constexpr int64_t INDEX_B_BSND = 0;
const constexpr int64_t INDEX_S_BSND = 1;
const constexpr int64_t INDEX_N_BSND = 2;
const constexpr int64_t INDEX_D_BSND = 3;

const constexpr int64_t INDEX_T_TND = 0;
const constexpr int64_t INDEX_N_TND = 1;
const constexpr int64_t INDEX_D_TND = 2;

const constexpr uint32_t N_VALID_VALUES[] = {4, 6, 8};
const constexpr uint32_t D_ALIGNMENT = 16;
const constexpr uint32_t CHUNK_T_MAX = 128;
const constexpr uint32_t V1_CHUNK_D_SIZE = 5120;
const constexpr uint32_t CHUNK_T_CALC_FACTOR = 32;
const constexpr uint32_t L0_B_SIZE = 8 * 1024;
const constexpr uint32_t FLOAT_ELE_SIZE = 4;
const constexpr uint32_t KERNEL_WIDTH = 8;

const constexpr uint32_t DB_L0C = 2;
const constexpr uint32_t STEP_K = 1;
const constexpr uint32_t DEPTH_K = 2;
const constexpr uint32_t STEP_MN = 1;

const constexpr size_t WORKSPACE_MULT_A = 2;
const constexpr size_t WORKSPACE_MULT_B = 192;
const constexpr size_t WORKSPACE_DIM_M = 256;
const constexpr size_t WORKSPACE_ELEMENTS = 8;
const constexpr size_t SYSTEM_WORKSPACE = 20 * 1024 * 1024;

const constexpr uint32_t SCHEDULE_MODE = 1;
const constexpr uint32_t OUT_FLAG_INV_RMS = 1;
const constexpr uint32_t OUT_FLAG_H_MIX = 2;
const constexpr uint32_t OUT_FLAG_H_PRE = 4;

const constexpr float DEFAULT_NORM_EPS = 1e-6f;
const constexpr float DEFAULT_HC_EPS = 1e-6f;

REGISTER_OPS_TILING_TEMPLATE(MhcPre, MhcPreBaseTiling, 1000);

ge::graphStatus MhcPreBaseTiling::GetInputShape()
{
    auto xTensor = context_->GetDynamicInputTensor(X_INDEX, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xTensor);
    auto phiTensor = context_->GetDynamicInputTensor(PHI_INDEX, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, phiTensor);
    auto alphaTensor = context_->GetDynamicInputTensor(ALPHA_INDEX, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, alphaTensor);
    auto biasTensor = context_->GetDynamicInputTensor(BIAS_INDEX, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, biasTensor);

    auto gammaTensor = context_->GetDynamicInputTensor(GAMMA_INDEX, 0);
    hasGamma_ = (gammaTensor == nullptr) ? 0 : 1;

    auto xDims = xTensor->GetStorageShape().GetDimNum();
    auto phiDims = phiTensor->GetStorageShape().GetDimNum();

    if (xDims == BSND_DIM_NUM) {
        return ParseBsndFormat(xTensor);
    } else if (xDims == TND_DIM_NUM) {
        return ParseTndFormat(xTensor);
    }

    OP_LOGE(context_->GetNodeName(), "X dims[%u] is invalid", xDims);
    return ge::GRAPH_FAILED;
}

ge::graphStatus MhcPreBaseTiling::ParseBsndFormat(const gert::Tensor *xTensor)
{
    uint64_t batch = xTensor->GetStorageShape().GetDim(INDEX_B_BSND);
    uint64_t sequence = xTensor->GetStorageShape().GetDim(INDEX_S_BSND);
    uint64_t numsResidual = xTensor->GetStorageShape().GetDim(INDEX_N_BSND);
    uint64_t dimens = xTensor->GetStorageShape().GetDim(INDEX_D_BSND);

    totalLength_ = batch * sequence;
    matK_ = numsResidual * dimens;
    N_ = numsResidual;
    D_ = dimens;

    return ValidateAndSetTilingParams(xTensor);
}

ge::graphStatus MhcPreBaseTiling::ParseTndFormat(const gert::Tensor *xTensor)
{
    totalLength_ = xTensor->GetStorageShape().GetDim(INDEX_T_TND);
    uint64_t numsResidual = xTensor->GetStorageShape().GetDim(INDEX_N_TND);
    uint64_t dimens = xTensor->GetStorageShape().GetDim(INDEX_D_TND);

    matK_ = numsResidual * dimens;
    N_ = numsResidual;
    D_ = dimens;

    return ValidateAndSetTilingParams(xTensor);
}

ge::graphStatus MhcPreBaseTiling::ValidateAndSetTilingParams(const gert::Tensor *xTensor)
{
    auto phiTensor = context_->GetDynamicInputTensor(PHI_INDEX, 0);
    auto phiDims = phiTensor->GetStorageShape().GetDimNum();

    if (phiDims < 2) {
        OP_LOGE(context_->GetNodeName(), "Phi dims[%u] is invalid", phiDims);
        return ge::GRAPH_FAILED;
    }

    bool isValidN = false;
    for (auto validN : N_VALID_VALUES) {
        if (N_ == validN) {
            isValidN = true;
            break;
        }
    }
    if (!isValidN) {
        OP_LOGE(context_->GetNodeName(), "N must be 4/6/8, but got N=%u", N_);
        return ge::GRAPH_FAILED;
    }

    if (D_ % D_ALIGNMENT != 0) {
        OP_LOGE(context_->GetNodeName(),
                "D must be 32 bytes aligned (element count mod 16 == 0 for BF16/FP16), but got D=%u", D_);
        return ge::GRAPH_FAILED;
    }

    matM_ = totalLength_;
    matN_ = phiTensor->GetStorageShape().GetDim(0);
    chunkTSize_ = (((totalLength_ + CHUNK_T_CALC_FACTOR - 1) / CHUNK_T_CALC_FACTOR) + CHUNK_T_CALC_FACTOR - 1)
                  * CHUNK_T_CALC_FACTOR;
    if (chunkTSize_ > CHUNK_T_MAX) {
        chunkTSize_ = CHUNK_T_MAX;
    }
    v1ChunkDSize_ = V1_CHUNK_D_SIZE;

    uint64_t phiSecondDim = phiTensor->GetStorageShape().GetDim(1);
    if (phiSecondDim != matK_) {
        OP_LOGE(context_->GetNodeName(), "Phi[1]=%u and matK_=%u (nD) shape are not compatible", phiSecondDim, matK_);
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPreBaseTiling::ParseInputAndAttr()
{
    if (GetInputShape() != ge::GRAPH_SUCCESS) {
        OP_LOGE(context_->GetNodeName(), "Get input shape failed");
        return ge::GRAPH_FAILED;
    }

    if (InitPlatformMemory() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (ParseOutputFlags() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    return ParseEpsAttributes();
}

ge::graphStatus MhcPreBaseTiling::InitPlatformMemory()
{
    uint64_t ubSize, l1Size, l0CSize;

    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo == nullptr) {
        OP_LOGE(context_->GetNodeName(), "Get platform info failed");
        return ge::GRAPH_FAILED;
    }

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, l0CSize);
    mm_.SetBufferSpace(l1Size, l0CSize, ubSize);
    blockDim_ = ascendcPlatform.GetCoreNumAic();

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPreBaseTiling::ParseOutputFlags()
{
    auto invRmsDesc = context_->GetOutputDesc(INV_RMS_INDEX);
    auto hMixDesc = context_->GetOutputDesc(H_MIX_INDEX);
    auto hPreDesc = context_->GetOutputDesc(H_PRE_INDEX);

    outFlag_ = 0;
    if (invRmsDesc != nullptr) {
        outFlag_ |= OUT_FLAG_INV_RMS;
    }
    if (hMixDesc != nullptr) {
        outFlag_ |= OUT_FLAG_H_MIX;
    }
    if (hPreDesc != nullptr) {
        outFlag_ |= OUT_FLAG_H_PRE;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPreBaseTiling::ParseEpsAttributes()
{
    auto attrs = context_->GetAttrs();

    auto normEpsPtr = attrs->GetAttrPointer<float>(1);
    normEps_ = (normEpsPtr != nullptr) ? *normEpsPtr : DEFAULT_NORM_EPS;

    auto hcEpsPtr = attrs->GetAttrPointer<float>(2);
    hcEps_ = (hcEpsPtr != nullptr) ? *hcEpsPtr : DEFAULT_HC_EPS;

    return ge::GRAPH_SUCCESS;
}


void MhcPreBaseTiling::FillTilingData()
{
    tilingData_.matmulTiling.set_dbL0C(DB_L0C);
    tilingData_.matmulTiling.set_dbL0A(DB_L0C);
    tilingData_.matmulTiling.set_dbL0B(DB_L0C);
    tilingData_.matmulTiling.set_stepKa(STEP_K);
    tilingData_.matmulTiling.set_stepKb(STEP_K);
    tilingData_.matmulTiling.set_depthA1(DEPTH_K);
    tilingData_.matmulTiling.set_depthB1(DEPTH_K);
    tilingData_.matmulTiling.set_stepM(STEP_MN);
    tilingData_.matmulTiling.set_stepN(STEP_MN);

    uint32_t baseM = chunkTSize_;
    uint32_t baseN = baseM;
    uint32_t baseK = L0_B_SIZE / baseN / FLOAT_ELE_SIZE * KERNEL_WIDTH;

    tilingData_.matmulTiling.set_baseM(baseM);
    tilingData_.matmulTiling.set_baseN(baseN);
    tilingData_.matmulTiling.set_baseK(baseK);

    tilingData_.set_coreNum(blockDim_);
    tilingData_.set_totalLength(totalLength_);
    tilingData_.set_nD(matK_);
    tilingData_.set_fusionSize(matN_);
    tilingData_.set_N(N_);
    tilingData_.set_D(D_);
    tilingData_.set_normEps(normEps_);
    tilingData_.set_hcEps(hcEps_);
    tilingData_.set_chunkTSize(chunkTSize_);
    tilingData_.set_v1ChunkDSize(v1ChunkDSize_);
    tilingData_.set_outFlag(outFlag_);
    tilingData_.set_hasGamma(hasGamma_);

    float scaleMean = 1.0f / static_cast<float>(matK_);
    tilingData_.set_scaleMean(scaleMean);
}

ge::graphStatus MhcPreBaseTiling::TilingProcess()
{
    size_t userWorkspaceSize = (WORKSPACE_MULT_A * WORKSPACE_MULT_B * WORKSPACE_DIM_M +
                                WORKSPACE_MULT_A * WORKSPACE_MULT_B * (KERNEL_WIDTH * KERNEL_WIDTH + WORKSPACE_MULT_A * KERNEL_WIDTH))
                                * sizeof(float) * blockDim_;
    size_t systemWorkspaceSize = SYSTEM_WORKSPACE;

    mm_.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT, false);
    mm_.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT, true);
    mm_.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    mm_.SetBias(false);
    mm_.SetDim(1);
    mm_.SetShape(matM_, matN_, matK_);
    mm_.SetOrgShape(matM_, matN_, matK_);
    if (mm_.GetTiling(tilingData_.matmulTiling) == -1) {
        OP_LOGE(context_->GetNodeName(), "MhcPre Tiling get tiling failed, batch: %lu, m: %lu",
                totalLength_, matM_);
        return ge::GRAPH_FAILED;
    }

    tilingKey_ = 0UL;

    workspaceSize_ = userWorkspaceSize + systemWorkspaceSize;

    return ge::GRAPH_SUCCESS;
}


ge::graphStatus MhcPreBaseTiling::DoOpTiling()
{
    auto inputXDesc = context_->GetInputDesc(0);
    if (inputXDesc == nullptr) {
        OP_LOGE(context_->GetNodeName(), "Invalid input pointer: x");
        return ge::GRAPH_FAILED;
    }

    if (ParseInputAndAttr() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (TilingProcess() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    FillTilingData();

    PrintTilingData();

    return ge::GRAPH_SUCCESS;
}

void MhcPreBaseTiling::PrintTilingData()
{
    OP_LOGD(context_->GetNodeName(), "blockDim: [%d]", tilingData_.get_coreNum());
    OP_LOGD(context_->GetNodeName(), "totalLength: [%d]", tilingData_.get_totalLength());
    OP_LOGD(context_->GetNodeName(), "nD: [%d]", tilingData_.get_nD());
    OP_LOGD(context_->GetNodeName(), "fusionSize: [%d]", tilingData_.get_fusionSize());
    OP_LOGD(context_->GetNodeName(), "N: [%d]", tilingData_.get_N());
    OP_LOGD(context_->GetNodeName(), "D: [%d]", tilingData_.get_D());
    OP_LOGD(context_->GetNodeName(), "normEps: [%f]", tilingData_.get_normEps());
    OP_LOGD(context_->GetNodeName(), "hcEps: [%d]", tilingData_.get_hcEps());
    OP_LOGD(context_->GetNodeName(), "outFlag: [%d]", tilingData_.get_outFlag());
    OP_LOGD(context_->GetNodeName(), "hasGamma: [%d]", tilingData_.get_hasGamma());
    OP_LOGD(context_->GetNodeName(), "chunkTSize: [%d]", tilingData_.get_chunkTSize());
    OP_LOGD(context_->GetNodeName(), "v1ChunkDSize: [%d]", tilingData_.get_v1ChunkDSize());
}

uint64_t MhcPreBaseTiling::GetTilingKey() const
{
    return tilingKey_;
}

ge::graphStatus MhcPreBaseTiling::PostTiling()
{
    OP_CHECK_IF(
        tilingData_.GetDataSize() % sizeof(uint64_t) != 0,
        OP_LOGE(context_->GetNodeName(), "Tiling data size[%zu] is not aligned to 8", tilingData_.GetDataSize()),
        return ge::GRAPH_FAILED);
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetRawTilingData());
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    context_->SetBlockDim(tilingData_.get_coreNum());
    context_->SetScheduleMode(SCHEDULE_MODE);

    size_t *workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_IF(workspaces == nullptr, OPS_REPORT_CUBE_INNER_ERR(context_->GetNodeName(), "Workspaces is null"),
                return ge::GRAPH_FAILED);

    workspaces[0] = workspaceSize_;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingFunc4mHCPre(gert::TilingContext *context)
{
    OP_CHECK_IF(context == nullptr, OPS_REPORT_CUBE_INNER_ERR("[mHCPreTilingTilingFunc]", "Context is null"),
                return ge::GRAPH_FAILED);

    return Ops::Transformer::OpTiling::TilingRegistry::GetInstance().DoTilingImpl(context);
}


static ge::graphStatus TilingPrepare4mHCPre(gert::TilingParseContext *context)
{
    OP_CHECK_IF(context == nullptr, OPS_REPORT_CUBE_INNER_ERR("[TilingPrepare4mHC]", "Context is null"),
                return ge::GRAPH_FAILED);
    fe::PlatFormInfos *platformInfo = context->GetPlatformInfo();
    OP_CHECK_IF(platformInfo == nullptr, OPS_REPORT_CUBE_INNER_ERR(context->GetNodeName(), "PlatformInfoPtr is null"),
                return ge::GRAPH_FAILED);

    auto compileInfoPtr = context->GetCompiledInfo<MhcPreCompileInfo>();
    OP_CHECK_IF(compileInfoPtr == nullptr, OPS_REPORT_CUBE_INNER_ERR(context->GetNodeName(), "CompileInfoPtr is null"),
                return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);

    compileInfoPtr->aicNum = ascendcPlatform.GetCoreNumAic();
    compileInfoPtr->aivNum = ascendcPlatform.GetCoreNumAiv();

    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, compileInfoPtr->l1Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, compileInfoPtr->l2Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, compileInfoPtr->l0ASize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, compileInfoPtr->l0BSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, compileInfoPtr->l0CSize);

    OP_LOGI(context->GetNodeName(), "Parse compile info success, l1Size:%lu, l2Size:%lu, coreNum:%lu",
            compileInfoPtr->l1Size, compileInfoPtr->l2Size, compileInfoPtr->aicNum);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MhcPre).Tiling(TilingFunc4mHCPre).TilingParse<MhcPreCompileInfo>(TilingPrepare4mHCPre);
} // namespace optiling
