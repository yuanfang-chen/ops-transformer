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
 * \file chunk_gated_delta_rule_recurrence_tiling.cpp
 * \brief
 */
#include "chunk_gated_delta_rule_recurrence_tiling.h"

#include "tiling_base/tiling_templates_registry.h"
#include "register/op_def_registry.h"
#include "platform/platform_infos_def.h"
#include "err/ops_err.h"
#include "log/log.h"
#include "tiling/platform/platform_ascendc.h"
#include "util/math_util.h"

namespace optiling {

REGISTER_OPS_TILING_TEMPLATE(ChunkGatedDeltaRuleRecurrence,
                              ChunkGatedDeltaRuleRecurrenceTiling, 0);

// Input index constants
const size_t INIT_STATE_INDEX  = 0;
const size_t KGEXP_INDEX       = 1;
const size_t VALUE_INDEX       = 2;
const size_t K_CUMDECAY_INDEX  = 3;
const size_t QGEXP_INDEX       = 4;
const size_t GEXP_INDEX        = 5;
const size_t CU_SEQLENS_INDEX  = 6;

const size_t DIM_0 = 0;
const size_t DIM_1 = 1;
const size_t DIM_2 = 2;
const size_t DIM_3 = 3;

const size_t STATE_DIM_NUM   = 4; // [b, hv, dv, dk]
const size_t CHUNK4D_DIM_NUM = 4; // [hv, n_chunks, cs, dk/dv]
const size_t GEXP_DIM_NUM    = 4; // [hv, n_chunks, cs, 1]
const size_t CSEQ_DIM_NUM    = 1; // [b+1]

// System workspace (16 MB)
const int64_t SYS_WORKSPACE_SIZE = 16 * 1024 * 1024;

// ──────────────────────────────────────────────────────────────────────────
void ChunkGatedDeltaRuleRecurrenceTiling::InitCompileInfo()
{
    auto platformInfoPtr = context_->GetPlatformInfo();
    if (platformInfoPtr == nullptr) {
        OP_LOGE(context_->GetNodeName(), "platformInfoPtr is null");
        return;
    }
    const auto &plat = platform_ascendc::PlatformAscendC(platformInfoPtr);
    plat.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfo_.ubSize);
    compileInfo_.aivNum = plat.GetCoreNumAiv();
    if (compileInfo_.aivNum == 0) {
        OP_LOGE(context_->GetNodeName(), "aivNum == 0");
        return;
    }
    tilingData_.coreNum = static_cast<uint32_t>(compileInfo_.aivNum);
}

ge::graphStatus ChunkGatedDeltaRuleRecurrenceTiling::GetPlatformInfo()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ChunkGatedDeltaRuleRecurrenceTiling::GetShapeAttrsInfo()
{
    OP_CHECK_IF(CheckContext() != ge::GRAPH_SUCCESS,
                OP_LOGE(inputParams_.opName, "CheckContext failed"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(AnalyzeDtype() != ge::GRAPH_SUCCESS,
                OP_LOGE(inputParams_.opName, "AnalyzeDtype failed"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(AnalyzeShapes() != ge::GRAPH_SUCCESS,
                OP_LOGE(inputParams_.opName, "AnalyzeShapes failed"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(GetScaleAttr() != ge::GRAPH_SUCCESS,
                OP_LOGE(inputParams_.opName, "GetScaleAttr failed"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ChunkGatedDeltaRuleRecurrenceTiling::DoOpTiling()
{
    OP_CHECK_IF(CalDvTile() != ge::GRAPH_SUCCESS,
                OP_LOGE(inputParams_.opName, "CalDvTile failed"),
                return ge::GRAPH_FAILED);
    PrintTilingData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ChunkGatedDeltaRuleRecurrenceTiling::DoLibApiTiling()
{
    tilingKey_ = 0;
    return ge::GRAPH_SUCCESS;
}

uint64_t ChunkGatedDeltaRuleRecurrenceTiling::GetTilingKey() const
{
    return tilingKey_;
}

ge::graphStatus ChunkGatedDeltaRuleRecurrenceTiling::GetWorkspaceSize()
{
    workspaceSize_ = SYS_WORKSPACE_SIZE;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ChunkGatedDeltaRuleRecurrenceTiling::PostTiling()
{
    context_->SetBlockDim(tilingData_.coreNum);
    auto tilingDataSize = sizeof(ChunkGatedDeltaRuleRecurrenceTilingData);
    errno_t ret = memcpy_s(context_->GetRawTilingData()->GetData(),
                           context_->GetRawTilingData()->GetCapacity(),
                           reinterpret_cast<void *>(&tilingData_),
                           tilingDataSize);
    if (ret != EOK) {
        OP_LOGE(context_->GetNodeName(), "memcpy_s failed, ret=%d", ret);
        return ge::GRAPH_FAILED;
    }
    context_->GetRawTilingData()->SetDataSize(tilingDataSize);

    size_t *workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_IF(workspaces == nullptr,
                OPS_REPORT_CUBE_INNER_ERR(context_->GetNodeName(), "workspaces is null"),
                return ge::GRAPH_FAILED);
    workspaces[0] = workspaceSize_;
    return ge::GRAPH_SUCCESS;
}

// ──────────────────────────────────────────────────────────────────────────
ge::graphStatus ChunkGatedDeltaRuleRecurrenceTiling::CheckContext()
{
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(INIT_STATE_INDEX));
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(INIT_STATE_INDEX));
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(KGEXP_INDEX));
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(KGEXP_INDEX));
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(VALUE_INDEX));
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(VALUE_INDEX));
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(K_CUMDECAY_INDEX));
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(K_CUMDECAY_INDEX));
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(QGEXP_INDEX));
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(QGEXP_INDEX));
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(GEXP_INDEX));
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(GEXP_INDEX));
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(CU_SEQLENS_INDEX));
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(CU_SEQLENS_INDEX));
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ChunkGatedDeltaRuleRecurrenceTiling::AnalyzeDtype()
{
    auto checkFloat = [&](size_t idx, const char *name) -> ge::graphStatus {
        auto dtype = context_->GetInputDesc(idx)->GetDataType();
        OP_CHECK_IF(dtype != ge::DT_FLOAT,
                    OP_LOGE(context_->GetNodeName(), "%s must be float32", name),
                    return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    };
    if (checkFloat(INIT_STATE_INDEX, "initial_state") != ge::GRAPH_SUCCESS) return ge::GRAPH_FAILED;
    if (checkFloat(KGEXP_INDEX,      "kgexp")         != ge::GRAPH_SUCCESS) return ge::GRAPH_FAILED;
    if (checkFloat(VALUE_INDEX,      "value")         != ge::GRAPH_SUCCESS) return ge::GRAPH_FAILED;
    if (checkFloat(K_CUMDECAY_INDEX, "k_cumdecay")    != ge::GRAPH_SUCCESS) return ge::GRAPH_FAILED;
    if (checkFloat(QGEXP_INDEX,      "qgexp")         != ge::GRAPH_SUCCESS) return ge::GRAPH_FAILED;
    if (checkFloat(GEXP_INDEX,       "gexp")          != ge::GRAPH_SUCCESS) return ge::GRAPH_FAILED;

    auto cuSeqDtype = context_->GetInputDesc(CU_SEQLENS_INDEX)->GetDataType();
    OP_CHECK_IF(cuSeqDtype != ge::DT_INT32,
                OP_LOGE(context_->GetNodeName(), "cu_seqlens must be int32"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

bool ChunkGatedDeltaRuleRecurrenceTiling::CheckDim(const gert::Shape &shape,
                                                    size_t expected,
                                                    const std::string &name)
{
    if (shape.GetDimNum() != expected) {
        OP_LOGE(context_->GetNodeName(),
                "%s: expected %zu dims but got %zu",
                name.c_str(), expected, shape.GetDimNum());
        return false;
    }
    return true;
}

ge::graphStatus ChunkGatedDeltaRuleRecurrenceTiling::AnalyzeShapes()
{
    const auto &stateShape   = context_->GetInputShape(INIT_STATE_INDEX)->GetOriginShape();
    const auto &valueShape   = context_->GetInputShape(VALUE_INDEX)->GetOriginShape();
    const auto &kgexpShape   = context_->GetInputShape(KGEXP_INDEX)->GetOriginShape();
    const auto &cuSeqShape   = context_->GetInputShape(CU_SEQLENS_INDEX)->GetOriginShape();

    if (!CheckDim(stateShape, STATE_DIM_NUM,   "initial_state") ||
        !CheckDim(valueShape, CHUNK4D_DIM_NUM, "value")         ||
        !CheckDim(kgexpShape, CHUNK4D_DIM_NUM, "kgexp")         ||
        !CheckDim(cuSeqShape, CSEQ_DIM_NUM,    "cu_seqlens")) {
        return ge::GRAPH_FAILED;
    }

    // state: [b, hv, dv, dk]
    tilingData_.b       = static_cast<uint32_t>(stateShape.GetDim(DIM_0));
    tilingData_.hv      = static_cast<uint32_t>(stateShape.GetDim(DIM_1));
    tilingData_.realDv  = static_cast<uint32_t>(stateShape.GetDim(DIM_2));
    tilingData_.realDk  = static_cast<uint32_t>(stateShape.GetDim(DIM_3));

    // value: [hv, n_chunks, cs, dv]
    tilingData_.nChunks      = static_cast<uint32_t>(valueShape.GetDim(DIM_1));
    tilingData_.realChunkSize = static_cast<uint32_t>(valueShape.GetDim(DIM_2));

    // Align to FP32_PER_BLOCK=8 (32-byte boundary)
    tilingData_.alignDk = Ops::Base::CeilAlign(tilingData_.realDk,
                                               static_cast<uint32_t>(8));
    tilingData_.alignDv = Ops::Base::CeilAlign(tilingData_.realDv,
                                               static_cast<uint32_t>(8));
    tilingData_.alignChunkSize = Ops::Base::CeilAlign(tilingData_.realChunkSize,
                                                      static_cast<uint32_t>(8));

    // b+1 entries in cu_seqlens
    OP_CHECK_IF(static_cast<uint32_t>(cuSeqShape.GetDim(DIM_0)) != tilingData_.b + 1,
                OP_LOGE(context_->GetNodeName(),
                        "cu_seqlens length should be b+1=%u but is %ld",
                        tilingData_.b + 1, cuSeqShape.GetDim(DIM_0)),
                return ge::GRAPH_FAILED);

    tilingData_.totalTasks  = tilingData_.b * tilingData_.hv;
    tilingData_.tasksPerCore = Ops::Base::CeilDiv(tilingData_.totalTasks,
                                                  tilingData_.coreNum);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ChunkGatedDeltaRuleRecurrenceTiling::GetScaleAttr()
{
    auto attrs = context_->GetAttrs();
    if (attrs != nullptr) {
        const float *scalePtr = attrs->GetAttrPointer<float>(0);
        tilingData_.scaleValue = (scalePtr != nullptr) ? *scalePtr : 1.0f;
    } else {
        tilingData_.scaleValue = 1.0f;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ChunkGatedDeltaRuleRecurrenceTiling::CalDvTile()
{
    // UB budget (bytes):
    //   Fixed   = 3 * alignCs * alignDk * 4   (kCumdecay + qgexp + kgexp queues)
    //           + alignDk * 4                  (scratchUb_ in tmpBuf_)
    //   Per-dvTile = (3*alignCs + alignDk) * dvTile * 4
    //                (valueQ + attnOutQ + vNewOutQ queues  +  stateUb_ in tmpBuf_)
    //   Reserve = 256
    uint64_t ubSize  = compileInfo_.ubSize;
    uint64_t alignCs = tilingData_.alignChunkSize;
    uint64_t alignDk = tilingData_.alignDk;
    uint64_t fixed   = 3UL * alignCs * alignDk * 4UL + alignDk * 4UL;
    uint64_t reserve = 256UL;

    if (fixed + reserve >= ubSize) {
        OP_LOGE(context_->GetNodeName(),
                "UB too small for fixed buffers: ubSize=%lu fixed=%lu",
                ubSize, fixed);
        return ge::GRAPH_FAILED;
    }

    uint64_t available = ubSize - fixed - reserve;
    uint64_t perUnit   = (3UL * alignCs + alignDk) * 4UL;
    if (perUnit == 0) {
        OP_LOGE(context_->GetNodeName(),
                "perUnit is 0 (alignCs=%lu alignDk=%lu)", alignCs, alignDk);
        return ge::GRAPH_FAILED;
    }

    // dvTile must be a multiple of 8 (32-byte alignment for float)
    uint64_t dvTile = (available / perUnit / 8UL) * 8UL;
    if (dvTile < 8UL) {
        dvTile = 8UL;
        OP_LOGW(context_->GetNodeName(), "dvTile clamped to minimum 8 — shapes may be large");
    }
    if (dvTile > tilingData_.alignDv) {
        dvTile = tilingData_.alignDv;
    }
    // Balance the number of dvTile rounds to keep each round the same size
    uint64_t nRounds = Ops::Base::CeilDiv(tilingData_.realDv, static_cast<uint32_t>(dvTile));
    dvTile = Ops::Base::CeilAlign(
        Ops::Base::CeilDiv(tilingData_.realDv, static_cast<uint32_t>(nRounds)),
        static_cast<uint32_t>(8));
    if (dvTile > tilingData_.alignDv) {
        dvTile = tilingData_.alignDv;
    }

    tilingData_.dvTile = static_cast<uint32_t>(dvTile);
    return ge::GRAPH_SUCCESS;
}

void ChunkGatedDeltaRuleRecurrenceTiling::PrintTilingData()
{
    OP_LOGD(context_->GetNodeName(), "coreNum=%u b=%u hv=%u realDk=%u alignDk=%u "
            "realDv=%u alignDv=%u nChunks=%u realCs=%u alignCs=%u dvTile=%u "
            "totalTasks=%u tasksPerCore=%u scaleValue=%f",
            tilingData_.coreNum, tilingData_.b, tilingData_.hv,
            tilingData_.realDk, tilingData_.alignDk,
            tilingData_.realDv, tilingData_.alignDv,
            tilingData_.nChunks, tilingData_.realChunkSize, tilingData_.alignChunkSize,
            tilingData_.dvTile, tilingData_.totalTasks, tilingData_.tasksPerCore,
            tilingData_.scaleValue);
}

// ──────────────────────────────────────────────────────────────────────────
static ge::graphStatus ChunkGatedDeltaRuleRecurrenceTilingFunc(gert::TilingContext *context)
{
    OP_CHECK_IF(context == nullptr,
                OPS_REPORT_CUBE_INNER_ERR("ChunkGatedDeltaRuleRecurrence", "context is null"),
                return ge::GRAPH_FAILED);
    return Ops::Transformer::OpTiling::TilingRegistry::GetInstance().DoTilingImpl(context);
}

static ge::graphStatus TilingPrepareForChunkGatedDeltaRuleRecurrence(
    gert::TilingParseContext *context)
{
    OP_CHECK_IF(context == nullptr,
                OPS_REPORT_CUBE_INNER_ERR("ChunkGatedDeltaRuleRecurrence", "context is null"),
                return ge::GRAPH_FAILED);
    auto compileInfoPtr = context->GetCompiledInfo<ChunkGatedDeltaRuleRecurrenceCompileInfo>();
    OP_CHECK_IF(compileInfoPtr == nullptr,
                OPS_REPORT_CUBE_INNER_ERR(context->GetNodeName(), "compileInfoPtr is null"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ChunkGatedDeltaRuleRecurrence)
    .Tiling(ChunkGatedDeltaRuleRecurrenceTilingFunc)
    .TilingParse<ChunkGatedDeltaRuleRecurrenceCompileInfo>(
        TilingPrepareForChunkGatedDeltaRuleRecurrence);

} // namespace optiling
