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
 * \file mhc_sinkhorn_tiling.cc
 * \brief
 */

#include <vector>
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "platform/platform_info.h"
#include "error_util.h"
#include "log/log.h"
#include "mhc_sinkhorn_tiling.h"

using namespace AscendC;
namespace optiling
{
constexpr int64_t X_IDX = 0;
constexpr int64_t Y_IDX = 0;
constexpr int64_t ATTR_EPS_IDX = 0;
constexpr int64_t ATTR_NUM_ITERS_IDX = 1;
constexpr int64_t ATTR_OUT_FLAG_IDX = 2;

constexpr int64_t DIM_NUM_3 = 3;
constexpr int64_t DIM_NUM_4 = 4;
constexpr int64_t DIM_ZERO = 0;
constexpr int64_t DIM_ONE = 1;
constexpr int64_t DIM_TWO = 2;

constexpr int64_t SIMD_RESERVED_SIZE = static_cast<uint64_t>(8) * 1024;
constexpr int64_t ASCENDC_TOOLS_WORKSPACE = 16777216;  // 16M
constexpr int64_t MASK_BUFFER = 64;
constexpr int64_t MAX_BUFFER = 256;

static const std::set<ge::DataType> X_DTYPE = {ge::DT_FLOAT};

bool MhcSinkhornTiling::IsCapable()
{
    return true;
}

ge::graphStatus MhcSinkhornTiling::GetPlatformInfo()
{
    auto compileInfo = reinterpret_cast<const MhcSinkhornCompileInfo*>(context_->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context_, compileInfo);
    totalCoreNum_ = compileInfo->coreNum;
    ubSize_ = compileInfo->ubSize;
    OP_CHECK_IF((ubSize_ <= 0), OP_LOGE(opName_, "ub size less than 0"),
                     return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcSinkhornTiling::GetShapeAttrsInfo()
{
    OP_LOGD(opName_, "MhcSinkhorn tiling GetShapeAttrsInfo.");
    auto const attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);

    auto epsPtr = attrs->GetAttrPointer<float>(ATTR_EPS_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, epsPtr);
    eps_ = static_cast<float>(*epsPtr);
    auto numItersPtr = attrs->GetAttrPointer<int64_t>(ATTR_NUM_ITERS_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, numItersPtr);
    num_iters_ = static_cast<int64_t>(*numItersPtr);
    OP_CHECK_IF((num_iters_ <= 0 || num_iters_ > 100), OP_LOGE(opName_, "num_iters_ must be greater than 0 and less than or equal to 100"),
                return ge::GRAPH_FAILED);
    auto outFlagPtr = attrs->GetAttrPointer<int64_t>(ATTR_OUT_FLAG_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outFlagPtr);
    out_flag_ = static_cast<int64_t>(*outFlagPtr);
    OP_CHECK_IF((out_flag_ != 0), OP_LOGE(opName_, "out_flag_ must be 0"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(CheckInputDtype() != ge::GRAPH_SUCCESS, OP_LOGE(opName_, "input dtype check failed."),
                    return ge::GRAPH_FAILED);

    OP_CHECK_IF(CheckInputShape() != ge::GRAPH_SUCCESS, OP_LOGE(opName_, "input shape check failed."),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcSinkhornTiling::CheckInputDtype()
{
    auto xPtr = context_->GetInputDesc(X_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xPtr);
    xDtype_ = xPtr->GetDataType();
    OP_CHECK_IF(
        (X_DTYPE.find(xDtype_) == X_DTYPE.end()),
        OP_LOGE(opName_, "indices dtype only support float32 currently, please check."),
        return ge::GRAPH_FAILED);

    auto outputPtr = context_->GetOutputDesc(Y_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputPtr);
    auto outputDtype = outputPtr->GetDataType();
    OP_CHECK_IF(outputDtype != xDtype_, OP_LOGE(opName_, "expected output dtype to be equal to xdtype, please check."),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcSinkhornTiling::CheckInputShape()
{
    auto xShapePtr = context_->GetInputShape(X_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xShapePtr);
    auto xShape = xShapePtr->GetStorageShape();
    xDimNum_ = static_cast<int64_t>(xShape.GetDimNum());
    OP_CHECK_IF((xDimNum_ != DIM_NUM_3 && xDimNum_ != DIM_NUM_4), OP_LOGE(opName_, "indicesDimNum must be 3 or 4"),
                    return ge::GRAPH_FAILED);
    n_ = xShape.GetDim(DIM_TWO);
    OP_CHECK_IF((n_ != 4 && n_ != 6 && n_ != 8), OP_LOGE(opName_, "&& nDim must be 4 or 6 or 8"),
                    return ge::GRAPH_FAILED);
    if (xDimNum_ == 3) {
        T_ = xShape.GetDim(DIM_ZERO);
    } else {
        T_ = xShape.GetDim(DIM_ZERO) * xShape.GetDim(DIM_ONE);
    }

    auto yShapePtr = context_->GetOutputShape(Y_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, yShapePtr);
    auto yShape = yShapePtr->GetStorageShape();
    yDimNum_ = static_cast<int64_t>(yShape.GetDimNum());
    OP_CHECK_IF((yDimNum_ != xDimNum_), OP_LOGE(opName_, "yDimNum must be equal xDimNum"),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcSinkhornTiling::DoOpTiling()
{
    OP_LOGD(opName_, "MhcSinkhorn tiling DoOpTiling.");
    //普通核和尾核大小计算
    tNormCore_ = Ops::Base::CeilDiv(T_, totalCoreNum_);
    usedCoreNum_ = Ops::Base::CeilDiv(T_, tNormCore_);
    tTailCore_ = T_ - tNormCore_ * (usedCoreNum_ - 1);

    //核内切分
    ubSize_ = ubSize_ - SIMD_RESERVED_SIZE - MASK_BUFFER - MAX_BUFFER;
    int64_t ubBlock = static_cast<int64_t>(Ops::Base::GetUbBlockSize(context_));
    int64_t xDtypeSize = ge::GetSizeByDataType(xDtype_);
    int64_t ubBlockX = ubBlock / xDtypeSize;

    int64_t blockFactorAlignX = Ops::Base::CeilAlign(tNormCore_ * n_ * n_, ubBlockX);
    if (blockFactorAlignX * 4  <= ubSize_ / xDtypeSize) { // 一次全搬入，不需要切分
        tUbFactor_ = blockFactorAlignX;
    } else {
        tUbFactor_ = Ops::Base::FloorAlign(ubSize_ / static_cast<int64_t>(4) / xDtypeSize, ubBlockX);
        tUbFactor_ = Ops::Base::FloorAlign(tUbFactor_, n_ * n_);
    }

    tNormCoreLoop_ = Ops::Base::CeilDiv(blockFactorAlignX, tUbFactor_);
    tUbFactorTail_ = blockFactorAlignX - (tNormCoreLoop_ - 1) * tUbFactor_;  // 正常核的尾循环大小（n*n的个数）

    int64_t blockTailAlignX = Ops::Base::CeilAlign(tTailCore_ * n_ * n_, ubBlockX);
    int64_t tTailUbFactor = 0;
    if (blockTailAlignX * 4 <= ubSize_ / xDtypeSize) {
        tTailUbFactor = blockTailAlignX;
    } else {
        tTailUbFactor = Ops::Base::FloorAlign(ubSize_ / static_cast<int64_t>(4) / xDtypeSize, ubBlockX);
        tTailUbFactor = Ops::Base::FloorAlign(tTailUbFactor, n_ * n_);
    }

    tTailCoreLoop_ = Ops::Base::CeilDiv(blockTailAlignX, tTailUbFactor);
    tUbTailTail_ = blockTailAlignX - (tTailCoreLoop_ - 1) * tTailUbFactor;

    SetTilingData();
    return ge::GRAPH_SUCCESS;
}

void MhcSinkhornTiling::SetTilingData()
{
    MhcSinkhornTilingData* tilingData = context_->GetTilingData<MhcSinkhornTilingData>();
    tilingData->eps = eps_;
    tilingData->num_iters = num_iters_;
    tilingData->out_flag = out_flag_;
    tilingData->T = T_;
    tilingData->n = n_;
    tilingData->usedCoreNum = usedCoreNum_;
    tilingData->tNormCoreLoop = tNormCoreLoop_;
    tilingData->tUbFactor = tUbFactor_;
    tilingData->tUbFactorTail = tUbFactorTail_;
    tilingData->tTailCoreLoop = tTailCoreLoop_;
    tilingData->tUbTailTail = tUbTailTail_;
    tilingData->tNormCore = tNormCore_;
}

ge::graphStatus MhcSinkhornTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t MhcSinkhornTiling::GetTilingKey() const
{
    int64_t tilingKey = 1000000;

    return tilingKey;
}

ge::graphStatus MhcSinkhornTiling::GetWorkspaceSize()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcSinkhornTiling::PostTiling()
{
    auto workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = ASCENDC_TOOLS_WORKSPACE;

    context_->SetBlockDim(usedCoreNum_);
    context_->SetScheduleMode(1);
    auto res = context_->SetLocalMemorySize(ubSize_);
    OP_CHECK_IF(
        (res != ge::GRAPH_SUCCESS), OP_LOGE(opName_, "SetLocalMemorySize ubSize = %ld failed.", ubSize_),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

void MhcSinkhornTiling::DumpTilingInfo()
{
    std::ostringstream info;
    info << "\n eps: " << eps_;
    info << "\n num_iters: " << num_iters_;
    info << "\n out_flag: " << out_flag_;
    info << "\n T: " << T_;
    info << "\n n: " << n_;
    info << "\n tilingKey_: " << GetTilingKey();
    info << "\n usedCoreNum: " << usedCoreNum_;
    info << "\n tNormCoreLoop: " << tNormCoreLoop_;
    info << "\n tUbFactor: " << tUbFactor_;
    info << "\n tUbFactorTail: " << tUbFactorTail_;
    info << "\n tTailCoreLoop: " << tTailCoreLoop_;
    info << "\n tUbTailTail: " << tUbTailTail_;
    info << "\n tNormCore: " << tNormCore_;
    OP_LOGI(opName_, "%s", info.str().c_str());
}

static ge::graphStatus TilingForMhcSinkhorn(gert::TilingContext* context)
{
    MhcSinkhornTiling tiling(context);
    auto ret = tiling.DoTiling();
    return ret;
}

ge::graphStatus TilingPrepareForMhcSinkhorn(gert::TilingParseContext* context)
{
    OP_LOGD(context, "TilingPrepareForMhcSinkhorn entering.");
    auto compileInfo = context->GetCompiledInfo<MhcSinkhornCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF((compileInfo->coreNum <= 0),
        OP_LOGE(context, "Failed to get core num."), return ge::GRAPH_FAILED);
    uint64_t ubSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    compileInfo->ubSize = static_cast<int64_t>(ubSize);
    OP_CHECK_IF((compileInfo->ubSize <= 0),
        OP_LOGE(context, "Failed to get ub size."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MhcSinkhorn)
    .Tiling(TilingForMhcSinkhorn)
    .TilingParse<MhcSinkhornCompileInfo>(TilingPrepareForMhcSinkhorn);
}  // namespace optiling
