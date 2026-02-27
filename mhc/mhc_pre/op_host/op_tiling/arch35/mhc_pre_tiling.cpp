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

const constexpr int64_t INDEX_B_BSND = 0;
const constexpr int64_t INDEX_S_BSND = 1;
const constexpr int64_t INDEX_N_BSND = 2;
const constexpr int64_t INDEX_D_BSND = 3;

const constexpr int64_t INDEX_T_TND = 0;
const constexpr int64_t INDEX_N_TND = 1;
const constexpr int64_t INDEX_D_TND = 2;

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
    if (gammaTensor == nullptr) {
        hasGamma_ = 0;
    } else {
        hasGamma_ = 1;
    }
    auto xDims = xTensor->GetStorageShape().GetDimNum();
    auto phiDims = phiTensor->GetStorageShape().GetDimNum();
    if (xDims == BSND_DIM_NUM) {
        // BSND格式: [B, S, N, D]
        uint64_t batch = xTensor->GetStorageShape().GetDim(INDEX_B_BSND);
        uint64_t sequence = xTensor->GetStorageShape().GetDim(INDEX_S_BSND);
        uint64_t numsResidual = xTensor->GetStorageShape().GetDim(INDEX_N_BSND);
        uint64_t dimens = xTensor->GetStorageShape().GetDim(INDEX_D_BSND);
        totalLength_ = batch * sequence;
        matK_ = numsResidual * dimens;
        N_ = numsResidual;
        D_ = dimens;
    } else if (xDims == TND_DIM_NUM) {
        // TND格式: [T, N, D]
        totalLength_ = xTensor->GetStorageShape().GetDim(INDEX_T_TND);
        uint64_t numsResidual = xTensor->GetStorageShape().GetDim(INDEX_N_TND);
        uint64_t dimens = xTensor->GetStorageShape().GetDim(INDEX_D_TND);
        matK_ = numsResidual * dimens;
        N_ = numsResidual;
        D_ = dimens;
    } else {
        OP_LOGE(context_->GetNodeName(), "xDims[%u] is invalid", xDims);
        return ge::GRAPH_FAILED;
    }

    if (phiDims < 2) {
        OP_LOGE(context_->GetNodeName(), "phiDims [%u] is invalid", phiDims);
        return ge::GRAPH_FAILED;
    }

    matM_ = totalLength_;
    matN_ = phiTensor->GetStorageShape().GetDim(0);  // phi的第二个维度是nD
    chunkTSize_ = (((totalLength_ + 24 - 1) / 24) + 32 - 1) / 32 * 32;
    if (chunkTSize_ > 192) {
        chunkTSize_ = 192;
    }
    if (N_ == 4) {
        v1ChunkDSize_ = 5120;
    } else {
        v1ChunkDSize_ = 2560;
    }

    // 检查phi的第二个维度是否等于matK_（即nD）
    uint64_t phiSecondDim = phiTensor->GetStorageShape().GetDim(1);
    if (phiSecondDim != matK_) {
        OP_LOGE(context_->GetNodeName(), "phi[1]=%u and matK_=%u (nD) shape is not compatible", phiSecondDim, matK_);
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPreBaseTiling::ParseInputAndAttr()
{
    uint64_t ubSize, l1Size, l0CSize;

    if (GetInputShape() != ge::GRAPH_SUCCESS) {
        OP_LOGE(context_->GetNodeName(), "get input shape failed");
        return ge::GRAPH_FAILED;
    }

    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo == nullptr) {
        OP_LOGE(context_->GetNodeName(), "get platform info failed");
        return ge::GRAPH_FAILED;
    }

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, l0CSize);
    mm_.SetBufferSpace(l1Size, l0CSize, ubSize);
    blockDim_ = ascendcPlatform.GetCoreNumAic();

    auto attrs = context_->GetAttrs();
    auto outFlagPtr = attrs->GetAttrPointer<uint32_t>(0);
    if (outFlagPtr != nullptr) {
        outFlag_ = static_cast<uint32_t>(*outFlagPtr);
    } else {
        outFlag_ = 0U;
    }
 
    auto normEpsPtr = attrs->GetAttrPointer<double>(1);
    if (normEpsPtr != nullptr) {
        normEps_ = static_cast<float>(*normEpsPtr);
    } else {
        normEps_ = 1e-6;
    }

    auto hcEpsPtr = attrs->GetAttrPointer<double>(2);
    if (hcEpsPtr != nullptr) {
        hcEps_ = static_cast<float>(*hcEpsPtr);
    } else {
        hcEps_ = 1e-6;
    }

    return ge::GRAPH_SUCCESS;
}


void MhcPreBaseTiling::FillTilingData()
{
    // 矩阵计算剩余部分
    tilingData_.matmulTiling.set_dbL0C(2);  // 2: 开启double buffer
    tilingData_.matmulTiling.set_dbL0A(2);  // 2: 开启double buffer
    tilingData_.matmulTiling.set_dbL0B(2);  // 2: 开启double buffer
    tilingData_.matmulTiling.set_stepKa(1); 
    tilingData_.matmulTiling.set_stepKb(1);
    tilingData_.matmulTiling.set_depthA1(2);  // 2: stepKa的两倍，开启double buffer
    tilingData_.matmulTiling.set_depthB1(2);  // 2: stepKb的两倍，开启double buffer
    tilingData_.matmulTiling.set_stepM(1);
    tilingData_.matmulTiling.set_stepN(1);

    uint32_t baseN = (matN_ + 16 - 1) / 16 * 16;  // 16: BaseM，16个元素对齐
    uint32_t baseK = 8 * 1024 / baseN / 8 * 8;  // 8 * 1024: 64k(L0Bsize) / 2(dbL0B) / 4(float), A矩阵不转置且
                                                // B矩阵转置场景下baseK以C0_size对齐，float场景下为8
    uint32_t baseM = 8 * 1024 / baseK / 16 * 16;  // 8 * 1024: 64k(L0Bsize) / 2(dbL0A) / 4(float), 16: BaseM，16个元素对齐

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
    // 当前最大为128*128的matmul计算, 预留3倍空间适配最大矩阵
    size_t userWorkspaceSize = (2 * 192 * 256 +  2 * 192 * (8 * 8 + 2 * 8))* sizeof(float) * blockDim_;
    size_t systemWorkspaceSize = 20 * 1024 * 1024; // 20M

    mm_.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT, false);
    mm_.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT, true);
    mm_.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    mm_.SetBias(false);
    mm_.SetDim(1);
    mm_.SetShape(matM_, matN_, matK_);       
    mm_.SetOrgShape(matM_, matN_, matK_);    // 原始MNK
    if (mm_.GetTiling(tilingData_.matmulTiling) == -1) {
        OP_LOGE(context_->GetNodeName(), "LowerTriangularInverseBaseTiling Get Tiling Failed!, batch, m: %lu, %lu", totalLength_, matM_);
        return ge::GRAPH_FAILED;
    }

    // 后续模板按位处理
    tilingKey_ = 0UL;

    workspaceSize_ = userWorkspaceSize + systemWorkspaceSize;

    return ge::GRAPH_SUCCESS;
}


ge::graphStatus MhcPreBaseTiling::DoOpTiling()
{
    auto inputXDesc = context_->GetInputDesc(0);
    if (inputXDesc == nullptr) {
        OP_LOGE(context_->GetNodeName(), "invalid input pointer: x");
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
    OP_CHECK_IF(tilingData_.GetDataSize() % sizeof(uint64_t) != 0,
        OP_LOGE(context_->GetNodeName(), "tiling data size[%zu] is not aligned to 8", tilingData_.GetDataSize()),
        return ge::GRAPH_FAILED);
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetRawTilingData());
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    context_->SetBlockDim(tilingData_.get_coreNum());
    context_->SetScheduleMode(1);

    size_t *workspaces = context_->GetWorkspaceSizes(1); // set workspace
    OP_CHECK_IF(workspaces == nullptr, OPS_REPORT_CUBE_INNER_ERR(context_->GetNodeName(), "workspaces is null"),
        return ge::GRAPH_FAILED);

    workspaces[0] = workspaceSize_;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingFunc4mHCPre(gert::TilingContext* context)
{
     OP_CHECK_IF(context == nullptr,
        OPS_REPORT_CUBE_INNER_ERR("[mHCPostTilingTilingFunc]", " context is null"),
        return ge::GRAPH_FAILED);

    return Ops::Transformer::OpTiling::TilingRegistry::GetInstance().DoTilingImpl(context);
}


static ge::graphStatus TilingPrepare4mHCPre(gert::TilingParseContext* context)
{
    OP_CHECK_IF(context == nullptr,
                OPS_REPORT_CUBE_INNER_ERR("[TilingPrepare4mHC]", "context is null"),
                return ge::GRAPH_FAILED);
    fe::PlatFormInfos* platformInfo = context->GetPlatformInfo();
    OP_CHECK_IF(platformInfo == nullptr,
                OPS_REPORT_CUBE_INNER_ERR(context->GetNodeName(), "platformInfoPtr is null"),
                return ge::GRAPH_FAILED);

    auto compileInfoPtr = context->GetCompiledInfo<MhcPreCompileInfo>();
    OP_CHECK_IF(compileInfoPtr == nullptr,
                OPS_REPORT_CUBE_INNER_ERR(context->GetNodeName(), "compileInfoPtr is null"),
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

    OP_LOGI(context->GetNodeName(),
            "parse compile info success l1Size:%lu, l2Size:%lu, coreNum:%lu",
            compileInfoPtr->l1Size,
            compileInfoPtr->l2Size,
            compileInfoPtr->aicNum);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MhcPre)
    .Tiling(TilingFunc4mHCPre)
    .TilingParse<MhcPreCompileInfo>(TilingPrepare4mHCPre);
}
