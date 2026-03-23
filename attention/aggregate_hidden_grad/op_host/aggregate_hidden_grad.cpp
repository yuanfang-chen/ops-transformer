/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!\n * \file aggregate_hidden_grad.cpp\n * \brief Host tiling implementation for aggregate_hidden_grad on ascend950\n */
#include <cstring>
#include "aggregate_hidden_grad.h"
#include "log/log.h"
#include "err/ops_err.h"
#include <register/op_impl_registry.h>
#include "tiling_base/tiling_templates_registry.h"
#include "attention/aggregate_hidden_grad/op_kernel/arch35/aggregate_hidden_grad_struct.h"

using namespace Ops::Transformer::OpTiling;
using AggregateHiddenGradArch35Tiling::AggregateHiddenGradTilingDataV35;

namespace optiling {
namespace {
constexpr uint32_t DIM_0 = 0; // S or W
constexpr uint32_t DIM_1 = 1; // B or H
constexpr uint32_t DIM_2 = 2; // H
constexpr uint32_t WORKSPACE_BYTES = 0; // no extra ws
}

bool AggregateHiddenGradTiling::PreparePlatformInfo()
{
    auto platformInfoPtr = context_->GetPlatformInfo();
    if (platformInfoPtr == nullptr) {
        auto compileInfo = context_->GetCompileInfo<AggregateHiddenGradCompileInfo>();
        OP_CHECK_NULL_WITH_CONTEXT(context_, compileInfo);
        aivNum_ = compileInfo->aivNum;
        ubSize_ = compileInfo->ubSize;
    } else {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
        aivNum_ = ascendcPlatform.GetCoreNumAiv();
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize_);
    }
    return true;
}

ge::graphStatus AggregateHiddenGradTiling::GetPlatformInfo()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AggregateHiddenGradTiling::GetShapeAttrsInfo()
{
    // Shapes from inputs
    auto goShape = context_->GetInputShape(0)->GetOriginShape();
    auto inShape = context_->GetInputShape(1)->GetOriginShape();
    auto wShape  = context_->GetInputShape(2)->GetOriginShape();
    S_ = static_cast<int64_t>(goShape.GetDim(DIM_0));
    B_ = static_cast<int64_t>(goShape.GetDim(DIM_1));
    H_ = static_cast<int64_t>(goShape.GetDim(DIM_2));
    W_ = static_cast<int64_t>(wShape.GetDim(DIM_0));

    auto maskTensor = context_->GetOptionalInputTensor(3);
    hasMask_ = (maskTensor != nullptr);

    auto inDtype = context_->GetInputDesc(0)->GetDataType();
    if (inDtype == ge::DT_FLOAT16 || inDtype == ge::DT_BF16) {
        dtypeSize_ = 2;
    } else {
        OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "dtype must be f16/bf16");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

bool AggregateHiddenGradTiling::IsCapable()
{
    // H must be multiple of 64, W==3
    if (H_ <= 0 || (H_ % 64) != 0) {
        return false;
    }
    if (W_ != 3) {
        return false;
    }
    return true;
}

void AggregateHiddenGradTiling::CalcInterCoreSplit()
{
    // Split H tiles (64 each) across aivNum_ cores
    int64_t tiles = H_ / 64;
    int64_t baseTiles = tiles / static_cast<int64_t>(aivNum_);
    int64_t remTiles = tiles % static_cast<int64_t>(aivNum_);
    if (remTiles == 0) {
        hMainCoreCnt_ = static_cast<int64_t>(aivNum_);
        hTailCoreCnt_ = 0;
        hMainSize_ = hTailSize_ = baseTiles * 64;
    } else {
        hMainCoreCnt_ = remTiles;
        hTailCoreCnt_ = static_cast<int64_t>(aivNum_) - remTiles;
        hMainSize_ = (baseTiles + 1) * 64;
        hTailSize_ = baseTiles * 64;
    }
}

void AggregateHiddenGradTiling::CalcUbTiling()
{
    // Start with minimal hUB=64, try full B,S and shrink if needed
    hUB_ = 64; bUB_ = B_; sUB_ = S_;
    // Estimate per-queue sizes
    auto bytesGO = hUB_ * bUB_ * sUB_ * dtypeSize_;
    auto bytesIn = bytesGO;
    auto bytesGI = bytesGO;
    auto bytesW  = hUB_ * 3 * dtypeSize_;
    auto bytesGW = bytesW;
    auto bytesMask = hasMask_ ? (bUB_ * sUB_) : 0;
    auto bytesTmp = 3 * hUB_ * dtypeSize_;

    // Count logical shares: gradOut,input,weight,(mask),gradIn,gradWeight,tmpBuf
    int64_t shares = hasMask_ ? 7 + 1 : 7;
    uint64_t perShare = ubSize_ / static_cast<uint64_t>(shares);
    auto fits = [&](int64_t bTry, int64_t sTry) {
        auto go = hUB_ * bTry * sTry * dtypeSize_;
        auto w  = hUB_ * 3 * dtypeSize_;
        auto mk = hasMask_ ? (bTry * sTry) : 0;
        return go <= perShare && w <= perShare && mk <= perShare && (3 * hUB_ * dtypeSize_) <= perShare;
    };

    if (!fits(bUB_, sUB_)) {
        // reduce B first
        bUB_ = 1;
        while (bUB_ < B_ && !fits(bUB_, sUB_)) {
            // reduce s
            sUB_ = (sUB_ + 1) / 2;
            if (sUB_ <= 0) { sUB_ = 1; break; }
        }
    }

    // Loop counts and tails
    hLoopCnt_ = (hMainSize_ + hUB_ - 1) / hUB_;
    bLoopCnt_ = (B_ + bUB_ - 1) / bUB_;
    sLoopCnt_ = (S_ + sUB_ - 1) / sUB_;
    hUBTail_ = (hMainSize_ % hUB_) == 0 ? hUB_ : (hMainSize_ % hUB_);
    bUBTail_ = (B_ % bUB_) == 0 ? bUB_ : (B_ % bUB_);
    sUBTail_ = (S_ % sUB_) == 0 ? sUB_ : (S_ % sUB_);
}

ge::graphStatus AggregateHiddenGradTiling::DoOpTiling()
{
    OP_CHECK_IF(!PreparePlatformInfo(), OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "platform info missing"),
                return ge::GRAPH_FAILED);
    CalcInterCoreSplit();
    CalcUbTiling();

    // fill tiling struct
    AggregateHiddenGradTilingDataV35 td{};
    td.hasMask = hasMask_ ? 1 : 0;
    td.H = H_; td.S = S_; td.B = B_; td.W = 3; td.dtypeSize = dtypeSize_;
    td.hMainCoreCnt = hMainCoreCnt_; td.hTailCoreCnt = hTailCoreCnt_;
    td.hMainSize = hMainSize_; td.hTailSize = hTailSize_;
    td.hUB = hUB_; td.bUB = bUB_; td.sUB = sUB_;
    td.hLoopCnt = hLoopCnt_; td.bLoopCnt = bLoopCnt_; td.sLoopCnt = sLoopCnt_;
    td.hUBTail = hUBTail_; td.bUBTail = bUBTail_; td.sUBTail = sUBTail_;
    td.hLoopCntTail = 0; td.bLoopCntTail = 0; td.sLoopCntTail = 0;
    td.coreMainRangeStart = 0; td.alignBytes = 32;

    // Save
    auto *raw = context_->GetRawTilingData();
    OP_CHECK_IF(sizeof(td) > raw->GetCapacity(),
                OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "tiling struct too large: %zu > %zu", sizeof(td), raw->GetCapacity()),
                return ge::GRAPH_FAILED);
    std::memcpy(raw->GetData(), &td, sizeof(td));
    raw->SetDataSize(sizeof(td));

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AggregateHiddenGradTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t AggregateHiddenGradTiling::GetTilingKey() const { return 0; }

ge::graphStatus AggregateHiddenGradTiling::GetWorkspaceSize()
{
    size_t *workspaces = context_->GetWorkspaceSizes(1);
    workspaces[0] = WORKSPACE_BYTES;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AggregateHiddenGradTiling::PostTiling()
{
    // set block dim = used vector cores
    context_->SetBlockDim(static_cast<uint32_t>(hMainCoreCnt_ + hTailCoreCnt_));
    context_->SetTilingKey(GetTilingKey());
    return ge::GRAPH_SUCCESS;
}

// Registration
static ge::graphStatus TilingPrepareForAggregateHiddenGrad(gert::TilingParseContext *context)
{
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto compileInfo = context->GetCompiledInfo<AggregateHiddenGradCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->aivNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfo->ubSize);
    return ge::GRAPH_SUCCESS;
}

ASCENDC_EXTERN_C ge::graphStatus TilingAggregateHiddenGrad(gert::TilingContext *context)
{
    AggregateHiddenGradTiling tiling(context);
    return tiling.DoTiling();
}

IMPL_OP_OPTILING(AggregateHiddenGrad)
    .Tiling(TilingAggregateHiddenGrad)
    .TilingParse<AggregateHiddenGradCompileInfo>(TilingPrepareForAggregateHiddenGrad);

} // namespace optiling
