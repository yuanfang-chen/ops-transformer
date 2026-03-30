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
 * \file mhc_pre_sinkhorn_tiling.cpp
 * \brief mhc_pre_sinkhorn_tiling
 */

#include <vector>
#include "util/platform_util.h"
#include "util/shape_util.h"
#include "platform/platform_info.h"
#include "log/log.h"
#include "mhc_pre_sinkhorn_tiling.h"

using namespace AscendC;
namespace optiling {
constexpr int64_t H_RES_IDX = 0;
constexpr int64_t H_RES_SINKHORN_IDX = 0;
constexpr int64_t NORM_OUT_IDX = 1;
constexpr int64_t SUM_OUT_IDX = 2;
constexpr int64_t ATTR_EPS_IDX = 0;
constexpr int64_t ATTR_NUM_ITERS_IDX = 1;
constexpr int64_t ATTR_OUT_FLAG_IDX = 2;

constexpr int64_t DIM_NUM_3 = 3;
constexpr int64_t DIM_NUM_4 = 4;
constexpr int64_t DIM_ZERO = 0;
constexpr int64_t DIM_ONE = 1;
constexpr int64_t DIM_TWO = 2;

constexpr int64_t NUM_ZERO = 0;
constexpr int64_t NUM_ONE = 1;
constexpr int64_t NUM_ONE_HUNDRED = 100;

constexpr int64_t N_NUM_4 = 4;
constexpr int64_t N_NUM_6 = 6;
constexpr int64_t N_NUM_8 = 8;

constexpr int64_t SIMD_RESERVED_SIZE = static_cast<uint64_t>(8) * 1024;
constexpr int64_t ASCENDC_TOOLS_WORKSPACE = 0;
constexpr int64_t MASK_BUFFER = 64;
constexpr int64_t MAX_BUFFER = 256;

static const std::set<ge::DataType> X_DTYPE = {ge::DT_FLOAT};

bool MhcPreSinkhornTiling::IsCapable()
{
    return true;
}

ge::graphStatus MhcPreSinkhornTiling::GetPlatformInfo()
{
    auto compileInfo = reinterpret_cast<const MhcPreSinkhornCompileInfo *>(context_->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context_, compileInfo);
    totalCoreNum_ = compileInfo->coreNum;
    ubSize_ = compileInfo->ubSize;
    OP_CHECK_IF((ubSize_ <= 0), OP_LOGE(opName_, "ub size less than 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPreSinkhornTiling::GetShapeAttrsInfo()
{
    OP_LOGD(opName_, "MhcPreSinkhorn tiling GetShapeAttrsInfo.");
    auto const attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);

    auto epsPtr = attrs->GetAttrPointer<float>(ATTR_EPS_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, epsPtr);
    eps_ = static_cast<float>(*epsPtr);
    auto numItersPtr = attrs->GetAttrPointer<int64_t>(ATTR_NUM_ITERS_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, numItersPtr);
    num_iters_ = static_cast<int64_t>(*numItersPtr);
    OP_CHECK_IF(
        (num_iters_ < NUM_ONE || num_iters_ > NUM_ONE_HUNDRED),
        OP_LOGE(opName_, "num_iters_ must be greater than 0 and less than or equal to 100, but got %d .", num_iters_),
        return ge::GRAPH_FAILED);
    auto outFlagPtr = attrs->GetAttrPointer<int64_t>(ATTR_OUT_FLAG_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outFlagPtr);
    out_flag_ = static_cast<int64_t>(*outFlagPtr);
    OP_CHECK_IF((out_flag_ != NUM_ZERO && out_flag_ != NUM_ONE),
                OP_LOGE(opName_, "outFlag value error, outFlag must be 0 or 1, but got outFlag = %d .", out_flag_),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(CheckInputDtype() != ge::GRAPH_SUCCESS, OP_LOGE(opName_, "input dtype check failed."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(CheckInputShape() != ge::GRAPH_SUCCESS, OP_LOGE(opName_, "input shape check failed."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPreSinkhornTiling::CheckInputDtype()
{
    auto hResPtr = context_->GetInputDesc(H_RES_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, hResPtr);
    xDtype_ = hResPtr->GetDataType();
    OP_CHECK_IF((X_DTYPE.find(xDtype_) == X_DTYPE.end()),
                OP_LOGE(opName_, "h_res dtype only support float32 currently, please check."),
                return ge::GRAPH_FAILED);

    auto hResSinkhornPtr = context_->GetOutputDesc(H_RES_SINKHORN_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, hResSinkhornPtr);
    auto hResSinkhornDtype = hResSinkhornPtr->GetDataType();
    OP_CHECK_IF(hResSinkhornDtype != xDtype_, OP_LOGE(opName_, "expected h_res_sinkhorn dtype to be equal to h_res dtype, please check."),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPreSinkhornTiling::CheckInputShape()
{
    auto hResShapePtr = context_->GetInputShape(H_RES_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, hResShapePtr);
    auto hResShape = hResShapePtr->GetStorageShape();
    OP_CHECK_IF((hResShape.GetShapeSize() == 0),
                OP_LOGE(opName_, "Input h_res must not be empty tensor"), return ge::GRAPH_FAILED);
    xDimNum_ = static_cast<int64_t>(hResShape.GetDimNum());
    OP_CHECK_IF((xDimNum_ != DIM_NUM_3 && xDimNum_ != DIM_NUM_4),
                OP_LOGE(opName_, "xDimNum must be 3 or 4, but got %d .", xDimNum_), return ge::GRAPH_FAILED);
    n_ = hResShape.GetDim(DIM_TWO);
    OP_CHECK_IF((n_ != N_NUM_4 && n_ != N_NUM_6 && n_ != N_NUM_8),
                OP_LOGE(opName_, "the nDim of h_res must be 4 or 6 or 8, but got %d .", n_), return ge::GRAPH_FAILED);
    if (xDimNum_ == DIM_NUM_3) {
        T_ = hResShape.GetDim(DIM_ZERO);
    } else {
        T_ = hResShape.GetDim(DIM_ZERO) * hResShape.GetDim(DIM_ONE);
    }

    auto hResSinkhornShapePtr = context_->GetOutputShape(H_RES_SINKHORN_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, hResSinkhornShapePtr);
    auto hResSinkhornShape = hResSinkhornShapePtr->GetStorageShape();
    yDimNum_ = static_cast<int64_t>(hResSinkhornShape.GetDimNum());
    OP_CHECK_IF((yDimNum_ != DIM_NUM_3 && yDimNum_ != DIM_NUM_4),
                OP_LOGE(opName_, "yDimNum must be 3 or 4, but got %d .", yDimNum_), return ge::GRAPH_FAILED);
    int64_t n = hResSinkhornShape.GetDim(DIM_TWO);
    OP_CHECK_IF((n != N_NUM_4 && n != N_NUM_6 && n != N_NUM_8), OP_LOGE(opName_, "the nDim of h_res_sinkhorn must be 4 or 6 or 8, but got %d .", n),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((yDimNum_ != xDimNum_), OP_LOGE(opName_, "yDimNum must be equal xDimNum"), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

void MhcPreSinkhornTiling::SplitByCoreNum(int64_t tCoreNum, int64_t ubBlockX, int64_t xDtypeSize, int64_t &tUbFactor,
                                       int64_t &tCoreLoop, int64_t &tUbFactorTail)
{
    int64_t blockFactorAlignX = Ops::Base::CeilAlign(tCoreNum * n_ * n_, ubBlockX);
    int64_t doubleBufferFactor = 2;
    int64_t maxUbFactor = Ops::Base::FloorDiv(ubSizeUsed_, xDtypeSize);
    
    if (blockFactorAlignX * doubleBufferFactor <= maxUbFactor) {
        tUbFactor = blockFactorAlignX;
    } else {
        tUbFactor = Ops::Base::FloorAlign(maxUbFactor / doubleBufferFactor, ubBlockX);
        tUbFactor = Ops::Base::FloorAlign(tUbFactor, n_ * n_);
        if (tUbFactor == 0) {
            tUbFactor = n_ * n_;
        }
    }
    tCoreLoop = Ops::Base::CeilDiv(blockFactorAlignX, tUbFactor);
    tUbFactorTail = blockFactorAlignX - (tCoreLoop - 1) * tUbFactor;
}

ge::graphStatus MhcPreSinkhornTiling::DoOpTiling()
{
    OP_LOGD(opName_, "MhcPreSinkhorn tiling DoOpTiling.");

    tNormCore_ = Ops::Base::CeilDiv(T_, totalCoreNum_);
    usedCoreNum_ = Ops::Base::CeilDiv(T_, tNormCore_);
    tTailCore_ = T_ - tNormCore_ * (usedCoreNum_ - 1);

    ubSizeUsed_ = ubSize_ - SIMD_RESERVED_SIZE - MASK_BUFFER - MAX_BUFFER;
    int64_t ubBlock = static_cast<int64_t>(Ops::Base::GetUbBlockSize(context_));
    int64_t xDtypeSize = ge::GetSizeByDataType(xDtype_);
    int64_t ubBlockX = Ops::Base::FloorDiv(ubBlock, xDtypeSize);
    int64_t tTailUbFactor = 0;

    SplitByCoreNum(tNormCore_, ubBlockX, xDtypeSize, tUbFactor_, tNormCoreLoop_, tUbFactorTail_);
    SplitByCoreNum(tTailCore_, ubBlockX, xDtypeSize, tTailUbFactor, tTailCoreLoop_, tUbTailTail_);

    SetTilingData();
    return ge::GRAPH_SUCCESS;
}

void MhcPreSinkhornTiling::SetTilingData()
{
    MhcPreSinkhornTilingData *tilingData = context_->GetTilingData<MhcPreSinkhornTilingData>();
    tilingData->eps = eps_;
    tilingData->num_iters = num_iters_;
    tilingData->out_flag = out_flag_;
    tilingData->n = n_;
    tilingData->usedCoreNum = usedCoreNum_;
    tilingData->tNormCoreLoop = tNormCoreLoop_;
    tilingData->tUbFactor = tUbFactor_;
    tilingData->tUbFactorTail = tUbFactorTail_;
    tilingData->tTailCoreLoop = tTailCoreLoop_;
    tilingData->tUbTailTail = tUbTailTail_;
    tilingData->tNormCore = tNormCore_;
}

ge::graphStatus MhcPreSinkhornTiling::DoLibApiTiling()
{
    return DoOpTiling();
}

uint64_t MhcPreSinkhornTiling::GetTilingKey() const
{
    return static_cast<uint64_t>(n_);
}

ge::graphStatus MhcPreSinkhornTiling::GetWorkspaceSize()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPreSinkhornTiling::PostTiling()
{
    return ge::GRAPH_SUCCESS;
}

void MhcPreSinkhornTiling::DumpTilingInfo()
{
    OP_LOGD(opName_, "MhcPreSinkhorn DumpTilingInfo.");
}
