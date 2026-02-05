/**
* This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file grouped_matmul_finalize_routing.h
 * \brief
 */

#include "chunk_gated_delta_rule_tiling.h"

#include "tiling_base/tiling_templates_registry.h"
#include "register/op_def_registry.h"
#include "platform/platform_infos_def.h"
#include "err/ops_err.h"
#include "log/log.h"
#include "tiling/platform/platform_ascendc.h"
#include "util/math_util.h"

namespace optiling {
    REGISTER_OPS_TILING_TEMPLATE(ChunkGatedDeltaRule, ChunkGatedDeltaRuleTiling, 0);

    const size_t QUERY_INDEX = 0;
    const size_t KEY_INDEX = 1;
    const size_t VALUE_INDEX = 2;
    const size_t BETA_INDEX = 3;
    const size_t STATE_INDEX = 4;
    const size_t CUSEQLENS_INDEX = 5;
    const size_t G_INDEX = 6;

    const size_t QKV_DIM_NUM = 3;
    const size_t BETA_DIM_NUM = 2;
    const size_t STATE_DIM_NUM = 4;
    const size_t CUSEQLENS_DIM_NUM = 1;
    const size_t G_DIM_NUM = 2;

    const size_t DIM_0 = 0;
    const size_t DIM_1 = 1;
    const size_t DIM_2 = 2;
    const size_t DIM_3 = 3;

    const size_t MAX_MTP = 8;

    void ChunkGatedDeltaRuleTiling::InitCompileInfo() {
        auto platformInfoPtr = context_->GetPlatformInfo();
        if (platformInfoPtr == nullptr) {
            OP_LOGE(context_->GetNodeName(), "platformInfoPtr is null");
            return;
        }
        const auto &ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfo_.ubSize);
        compileInfo_.aivNum = ascendcPlatform.GetCoreNumAiv();

        if (compileInfo_.aivNum <= 0) {
            OP_LOGE(context_->GetNodeName(), "aivNum <= 0");
            return;
        }
        tilingData_.vectorCoreNum = compileInfo_.aivNum;
    }

    ge::graphStatus ChunkGatedDeltaRuleTiling::GetPlatformInfo() {
        return ge::GRAPH_SUCCESS;
    };

    ge::graphStatus ChunkGatedDeltaRuleTiling::GetShapeAttrsInfo() {
        OP_CHECK_IF(CheckContext() != ge::GRAPH_SUCCESS, OP_LOGE(inputParams_.opName, "Invalid context."), 
            return ge::GRAPH_FAILED);

        OP_CHECK_IF(AnalyzeDtype() != ge::GRAPH_SUCCESS, OP_LOGE(inputParams_.opName, "Invalid dtypes."), 
            return ge::GRAPH_FAILED);

        OP_CHECK_IF(AnalyzeShapes() != ge::GRAPH_SUCCESS, OP_LOGE(inputParams_.opName, "Invalid shapes."),
            return ge::GRAPH_FAILED);

        OP_CHECK_IF(GetScale() != ge::GRAPH_SUCCESS, OP_LOGE(inputParams_.opName, "Invalid GetScale."),
            return ge::GRAPH_FAILED);
        
        OP_CHECK_IF(GetOptionalInput() != ge::GRAPH_SUCCESS, OP_LOGE(inputParams_.opName, "Invalid GetOptionalInput."),
            return ge::GRAPH_FAILED);

        OP_CHECK_IF(AnalyzeFormat() != ge::GRAPH_SUCCESS, OP_LOGE(inputParams_.opName, "Invalid Format."),
            return ge::GRAPH_FAILED);

        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus ChunkGatedDeltaRuleTiling::DoOpTiling() {
        OP_CHECK_IF(CalUbSize() != ge::GRAPH_SUCCESS, OP_LOGE(inputParams_.opName, "CalUbSize failed."),
            return ge::GRAPH_FAILED);

        PrintTilingData();
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus ChunkGatedDeltaRuleTiling::DoLibApiTiling() {
        tilingKey_ = 0;
        return ge::GRAPH_SUCCESS;
    };

    uint64_t ChunkGatedDeltaRuleTiling::GetTilingKey() const {
        return tilingKey_;
    };

    ge::graphStatus ChunkGatedDeltaRuleTiling::GetWorkspaceSize() {
        // system workspace size is 16 * 1024 * 1024 = 16 M;
        constexpr int64_t sysWorkspaceSize = 16777216;
        workspaceSize_ = sysWorkspaceSize;

        return ge::GRAPH_SUCCESS;
    };

    ge::graphStatus ChunkGatedDeltaRuleTiling::PostTiling() {
        context_->SetBlockDim(tilingData_.vectorCoreNum);
        auto tilingDataSize = sizeof(ChunkGatedDeltaRuleTilingData);
        errno_t ret = memcpy_s(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity(), 
            reinterpret_cast<void *>(&tilingData_), tilingDataSize);
        if (ret != EOK) {
            OP_LOGE(context_->GetNodeName(), "memcpy_s failed, ret=%d", ret);
            return ge::GRAPH_FAILED;
        }
        context_->GetRawTilingData()->SetDataSize(tilingDataSize);

        size_t *workspaces = context_->GetWorkspaceSizes(1); // set workspace
        OP_CHECK_IF(workspaces == nullptr, OPS_REPORT_CUBE_INNER_ERR(context_->GetNodeName(), "workspaces is null"),
            return ge::GRAPH_FAILED);
        workspaces[0] = workspaceSize_;

        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus ChunkGatedDeltaRuleTiling::CheckContext() {
        OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(QUERY_INDEX));
        OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(QUERY_INDEX));

        OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(KEY_INDEX));
        OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(KEY_INDEX));

        OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(VALUE_INDEX));
        OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(VALUE_INDEX));

        OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(BETA_INDEX));
        OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(BETA_INDEX));

        OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(STATE_INDEX));
        OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(STATE_INDEX));

        OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(CUSEQLENS_INDEX));
        OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(CUSEQLENS_INDEX));
    
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus ChunkGatedDeltaRuleTiling::AnalyzeDtype() {
        auto queryDtype = context_->GetInputDesc(QUERY_INDEX)->GetDataType();
        auto keyDtype = context_->GetInputDesc(KEY_INDEX)->GetDataType();
        auto valueDtype = context_->GetInputDesc(VALUE_INDEX)->GetDataType();
        OP_CHECK_IF(queryDtype != ge::DT_BF16 || keyDtype != ge::DT_BF16 || valueDtype != ge::DT_BF16,
            OP_LOGE(context_->GetNodeName(), "query dtype, key dtype and value dtype should be bfloat16"),
            return ge::GRAPH_FAILED);

        auto betaDtype = context_->GetInputDesc(BETA_INDEX)->GetDataType();
        auto stateDtype = context_->GetInputDesc(STATE_INDEX)->GetDataType();
        OP_CHECK_IF(betaDtype != ge::DT_BF16 || stateDtype != ge::DT_BF16,
            OP_LOGE(context_->GetNodeName(), "beta dtype and state dtype should be bfloat16"),
            return ge::GRAPH_FAILED);

        auto cuSeqlensDtype = context_->GetInputDesc(CUSEQLENS_INDEX)->GetDataType();
        OP_CHECK_IF(cuSeqlensDtype != ge::DT_INT32,
            OP_LOGE(context_->GetNodeName(), "cuSeqlens dtype should be int32"),
            return ge::GRAPH_FAILED);

        if (context_->GetOptionalInputDesc(G_INDEX) != nullptr) {
            auto gamaDtype = context_->GetOptionalInputDesc(G_INDEX)->GetDataType();
            OP_CHECK_IF(gamaDtype != ge::DT_FLOAT, OP_LOGE(context_->GetNodeName(), "gama dtype should be float32"),
                return ge::GRAPH_FAILED);
        }

        return ge::GRAPH_SUCCESS;
    }

    bool ChunkGatedDeltaRuleTiling::CheckDimEqual(
        const gert::Shape a, 
        const int64_t dimA, 
        const gert::Shape b, 
        const int64_t dimB,
        const std::string &nameA, 
        const std::string &nameB,
        const std::string &dimDesc) 
    {
        if (a.GetDim(dimA) != b.GetDim(dimB)) {
            OP_LOGE(context_->GetNodeName(), "The %s of %s and %s should be the same, but %s is %ld while %s is %ld",
            dimDesc.c_str(), nameA.c_str(), nameB.c_str(), nameA.c_str(), a.GetDim(dimA), nameB.c_str(), b.GetDim(dimB));
            return false;
        }
        return true;
    }

    bool ChunkGatedDeltaRuleTiling::CheckDim(
        const gert::Shape shape, 
        const size_t dim, 
        const std::string &dimDesc) 
    {
        if (shape.GetDimNum() != dim) {
            OP_LOGE(context_->GetNodeName(), "The number of dimensons of %s should be %zu, but it is %zu",
                dimDesc.c_str(), dim, shape.GetDimNum());
            return false;
        }
        return true;
    }

    ge::graphStatus ChunkGatedDeltaRuleTiling::AnalyzeShapes() {
        const auto &queryShape = context_->GetInputShape(QUERY_INDEX)->GetOriginShape();
        const auto &keyShape = context_->GetInputShape(KEY_INDEX)->GetOriginShape();
        const auto &valueShape = context_->GetInputShape(VALUE_INDEX)->GetOriginShape();
        const auto &betaShape = context_->GetInputShape(BETA_INDEX)->GetOriginShape();
        const auto &stateShape = context_->GetInputShape(STATE_INDEX)->GetOriginShape();
        const auto &cuSeqlensShape = context_->GetInputShape(CUSEQLENS_INDEX)->GetOriginShape();
        
        if (!CheckDim(queryShape, QKV_DIM_NUM, "query") || !CheckDim(keyShape, QKV_DIM_NUM, "key") ||
            !CheckDim(valueShape, QKV_DIM_NUM, "value") || !CheckDim(betaShape, BETA_DIM_NUM, "beta") ||
            !CheckDim(stateShape, STATE_DIM_NUM, "state") ||
            !CheckDim(cuSeqlensShape, CUSEQLENS_DIM_NUM, "actual_seq_lengths")) {
                return ge::GRAPH_FAILED;
            }

        if (!CheckDimEqual(queryShape, DIM_0, keyShape, DIM_0, "query", "key", "T dimension") ||
            !CheckDimEqual(queryShape, DIM_1, keyShape, DIM_1, "query", "key", "Nk dimension") ||
            !CheckDimEqual(queryShape, DIM_2, keyShape, DIM_2, "query", "key", "Dk dimension") ||
            !CheckDimEqual(stateShape, DIM_1, valueShape, DIM_1, "state", "value", "Nv dimension") ||
            !CheckDimEqual(stateShape, DIM_2, valueShape, DIM_2, "state", "value", "Dv dimension") ||
            !CheckDimEqual(valueShape, DIM_0, queryShape, DIM_0, "value", "query", "T dimension") ||
            !CheckDimEqual(betaShape, DIM_0, queryShape, DIM_0, "beta", "query", "T dimension") ||
            !CheckDimEqual(betaShape, DIM_1, valueShape, DIM_1, "beta", "value", "Nv dimension") ||
            !CheckDimEqual(stateShape, DIM_3, queryShape, DIM_2, "state", "query", "Dk dimension")) {
                return ge::GRAPH_FAILED;
            }

        /* */

        return ge::GRAPH_SUCCESS;
    }

    bool ChunkGatedDeltaRuleTiling::CheckFormat(ge::Format format, const std::string &Desc)
    {
        if (format == ge::FORMAT_FRACTAL_NZ) {
            OP_LOGE(context_->GetNodeName(), "%s format not support NZ", Desc.c_str());
            return false;
        }
        return true;
    }

    ge::graphStatus ChunkGatedDeltaRuleTiling::AnalyzeFormat() {
        if (!CheckFormat(context_->GetInputDesc(QUERY_INDEX)->GetStorageFormat(), "query") ||
            !CheckFormat(context_->GetInputDesc(KEY_INDEX)->GetStorageFormat(), "key") ||
            !CheckFormat(context_->GetInputDesc(VALUE_INDEX)->GetStorageFormat(), "value") ||
            !CheckFormat(context_->GetInputDesc(STATE_INDEX)->GetStorageFormat(), "state") ||
            !CheckFormat(context_->GetInputDesc(CUSEQLENS_INDEX)->GetStorageFormat(), "actual_seq_lengths")) {
                return ge::GRAPH_FAILED;
            }

        if (context_->GetOptionalInputDesc(G_INDEX) != nullptr) {
            auto gamaFormat = context_->GetOptionalInputDesc(G_INDEX)->GetStorageFormat();
            OP_CHECK_IF(gamaFormat == ge::FORMAT_FRACTAL_NZ, OP_LOGE(context_->GetNodeName(), "gama format not support NZ"),
                return ge::GRAPH_FAILED);
            }

        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus ChunkGatedDeltaRuleTiling::GetScale() {
        auto attrs = context_->GetAttrs();
        float scaleValue = *attrs->GetAttrPointer<float>(0);
        tilingData_.scale = scaleValue;

        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus ChunkGatedDeltaRuleTiling::GetOptionalInput() {
        if (context_->GetOptionalInputDesc(G_INDEX) == nullptr) {
            tilingData_.hasGamma = 0;
        } else {
            tilingData_.hasGamma = 1;
        }

        return ge::GRAPH_SUCCESS;
    }

    void ChunkGatedDeltaRuleTiling::PrintTilingData() {
        OP_LOGD(context_->GetNodeName(), "vectorCoreNum: [%u]", tilingData_.vectorCoreNum);
        OP_LOGD(context_->GetNodeName(), "hasGamma: [%u]", tilingData_.hasGamma);
        /* */
    }

    ge::graphStatus ChunkGatedDeltaRuleTiling::CalUbSize() {
        /* */
        return ge::GRAPH_SUCCESS;
    }

    static ge::graphStatus ChunkGatedDeltaRuleTilingFunc(gert::TilingContext *context) {
        OP_CHECK_IF(context == nullptr, OPS_REPORT_CUBE_INNER_ERR("ChunkGatedDeltaRule", "context is null"),
            return ge::GRAPH_FAILED);
        return Ops::Transformer::OpTiling::TilingRegistry::GetInstance().DoTilingImpl(context);
    }

    static ge::graphStatus TilingPrepareForChunkGatedDeltaRule(gert::TilingParseContext *context) {
        OP_CHECK_IF(context == nullptr, OPS_REPORT_CUBE_INNER_ERR("ChunkGatedDeltaRule", "context is null"),
            return ge::GRAPH_FAILED);

        fe::PlatFormInfos *platformInfo = context->GetPlatformInfo();
        OP_CHECK_IF(platformInfo == nullptr, OPS_REPORT_CUBE_INNER_ERR(context->GetNodeName(), "platformInfoPtr is null"),
            return ge::GRAPH_FAILED);

        auto compileInfoPtr = context->GetCompiledInfo<ChunkGatedDeltaRuleCompileInfo>();
        OP_CHECK_IF(compileInfoPtr == nullptr, OPS_REPORT_CUBE_INNER_ERR(context->GetNodeName(), "compileInfoPtr is null"),
            return ge::GRAPH_FAILED);

        return ge::GRAPH_SUCCESS;
    }

    IMPL_OP_OPTILING(ChunkGatedDeltaRule)
    .Tiling(ChunkGatedDeltaRuleTilingFunc)
    .TilingParse<ChunkGatedDeltaRuleCompileInfo>(TilingPrepareForChunkGatedDeltaRule);
}  // namespace optiling