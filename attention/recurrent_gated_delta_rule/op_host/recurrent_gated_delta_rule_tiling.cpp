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
 * \file recurrent_gated_delta_rule_tiling.cpp
 * \brief
 */
#include "recurrent_gated_delta_rule_tiling.h"

#include "tiling_base/tiling_templates_registry.h"
#include "register/op_def_registry.h"
#include "platform/platform_infos_def.h"
#include "err/ops_err.h"
#include "log/log.h"
#include "tiling/platform/platform_ascendc.h"
#include "util/math_util.h"

namespace optiling {

REGISTER_OPS_TILING_TEMPLATE(RecurrentGatedDeltaRule, RecurrentGatedDeltaRuleTiling, 0);

const size_t QUERY_INDEX = 0;
const size_t KEY_INDEX = 1;
const size_t VALUE_INDEX = 2;
const size_t BETA_INDEX = 3;
const size_t STATE_INDEX = 4;
const size_t CUSEQLENS_INDEX = 5;
const size_t SSM_STATE_INDICES_INDEX = 6;
const size_t G_INDEX = 7;
const size_t GK_INDEX = 8;
const size_t ACC_TO_INDEX = 9;

const size_t OUT_INDEX = 0;

const size_t QKV_DIM_NUM = 3;
const size_t OUT_NUM = 3;
const size_t BETA_DIM_NUM = 2;
const size_t STATE_DIM_NUM = 4;
const size_t CUSEQLENS_DIM_NUM = 1;
const size_t SSM_STATE_INDICES_DIM_NUM = 1;
const size_t G_DIM_NUM = 2;
const size_t GK_DIM_NUM = 3;
const size_t ACC_DIM_NUM = 1;

const size_t DIM_0 = 0;
const size_t DIM_1 = 1;
const size_t DIM_2 = 2;
const size_t DIM_3 = 3;

const size_t MAX_MTP = 8;

bool gamaFlag = false;
bool gamaKFlag = false;
bool accFlag = false;

void RecurrentGatedDeltaRuleTiling::InitCompileInfo()
{
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

ge::graphStatus RecurrentGatedDeltaRuleTiling::GetPlatformInfo()
{
    return ge::GRAPH_SUCCESS;
};

ge::graphStatus RecurrentGatedDeltaRuleTiling::GetShapeAttrsInfo()
{
    OP_CHECK_IF(CheckContext() != ge::GRAPH_SUCCESS, OP_LOGE(inputParams_.opName, "Invalid context."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(AnalyzeDtype() != ge::GRAPH_SUCCESS, OP_LOGE(inputParams_.opName, "Invalid dtypes."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(AnalyzeShapesParser() != ge::GRAPH_SUCCESS, OP_LOGE(inputParams_.opName, "Invalid shapes parser."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(AnalyzeShapes() != ge::GRAPH_SUCCESS, OP_LOGE(inputParams_.opName, "Invalid shapes."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(AnalyzeEmptyTensor() != ge::GRAPH_SUCCESS, OP_LOGE(inputParams_.opName, "Invalid shapes."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(GetScale() != ge::GRAPH_SUCCESS, OP_LOGE(inputParams_.opName, "Invalid GetScale."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(GetOptionalInput() != ge::GRAPH_SUCCESS, OP_LOGE(inputParams_.opName, "Invalid GetOptionalInput."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(AnalyzeFormat() != ge::GRAPH_SUCCESS, OP_LOGE(inputParams_.opName, "Invalid Format."),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RecurrentGatedDeltaRuleTiling::DoOpTiling()
{
    OP_CHECK_IF(CalUbSize() != ge::GRAPH_SUCCESS, OP_LOGE(inputParams_.opName, "CalUbSize failed."),
                return ge::GRAPH_FAILED);

    PrintTilingData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RecurrentGatedDeltaRuleTiling::DoLibApiTiling()
{
    tilingKey_ = 0;
    return ge::GRAPH_SUCCESS;
};

uint64_t RecurrentGatedDeltaRuleTiling::GetTilingKey() const
{
    return tilingKey_;
};

ge::graphStatus RecurrentGatedDeltaRuleTiling::GetWorkspaceSize()
{
    // system workspace size is 16 * 1024 * 1024 = 16M;
    constexpr int64_t sysWorkspaceSize = 16777216;
    workspaceSize_ = sysWorkspaceSize;

    return ge::GRAPH_SUCCESS;
};

ge::graphStatus RecurrentGatedDeltaRuleTiling::PostTiling()
{
    context_->SetBlockDim(tilingData_.vectorCoreNum);
    auto tilingDataSize = sizeof(RecurrentGatedDeltaRuleTilingData);
    errno_t ret = memcpy_s(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity(),
                           reinterpret_cast<void *>(&tilingData_), tilingDataSize);
    if (ret != EOK) {
        OP_LOGE(context_->GetNodeName(), "memcpy_s failed, ret=%d.", ret);
        return ge::GRAPH_FAILED;
    }
    context_->GetRawTilingData()->SetDataSize(tilingDataSize);

    size_t *workspaces = context_->GetWorkspaceSizes(1); // set workspace
    OP_CHECK_IF(workspaces == nullptr, OPS_REPORT_CUBE_INNER_ERR(context_->GetNodeName(), "workspaces is null."),
                return ge::GRAPH_FAILED);
    workspaces[0] = workspaceSize_;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RecurrentGatedDeltaRuleTiling::CheckContext()
{
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

    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(SSM_STATE_INDICES_INDEX));
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(SSM_STATE_INDICES_INDEX));

    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetOutputShape(OUT_INDEX));
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetOutputDesc(OUT_INDEX));

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RecurrentGatedDeltaRuleTiling::AnalyzeDtype()
{
    auto queryDtype = context_->GetInputDesc(QUERY_INDEX)->GetDataType();
    auto keyDtype = context_->GetInputDesc(KEY_INDEX)->GetDataType();
    auto valueDtype = context_->GetInputDesc(VALUE_INDEX)->GetDataType();
    OP_CHECK_IF(queryDtype != ge::DT_BF16 || keyDtype != ge::DT_BF16 || valueDtype != ge::DT_BF16,
                OP_LOGE(context_->GetNodeName(), "Query dtype, key dtype and value dtype should be bfloat16."),
                return ge::GRAPH_FAILED);

    auto betaDtype = context_->GetInputDesc(BETA_INDEX)->GetDataType();
    auto stateDtype = context_->GetInputDesc(STATE_INDEX)->GetDataType();
    OP_CHECK_IF(betaDtype != ge::DT_BF16 || stateDtype != ge::DT_BF16,
                OP_LOGE(context_->GetNodeName(), "Beta dtype and state dtype should be bfloat16."),
                return ge::GRAPH_FAILED);

    auto cuSeqlensDtype = context_->GetInputDesc(CUSEQLENS_INDEX)->GetDataType();
    auto ssmStateIndicesDtype = context_->GetInputDesc(SSM_STATE_INDICES_INDEX)->GetDataType();
    OP_CHECK_IF(cuSeqlensDtype != ge::DT_INT32 || ssmStateIndicesDtype != ge::DT_INT32,
                OP_LOGE(context_->GetNodeName(), "ActualSeqLengths dtype and ssmStateIndices dtype should be int32."),
                return ge::GRAPH_FAILED);

    if (context_->GetOptionalInputDesc(G_INDEX) != nullptr) {
        gamaFlag = true;
        auto gamaDtype = context_->GetOptionalInputDesc(G_INDEX)->GetDataType();
        OP_CHECK_IF(gamaDtype != ge::DT_FLOAT, OP_LOGE(context_->GetNodeName(), "Gama dtype should be float32."),
                    return ge::GRAPH_FAILED);
    }

    if (context_->GetOptionalInputDesc(GK_INDEX) != nullptr) {
        gamaKFlag = true;
        auto gamaKDtype = context_->GetOptionalInputDesc(GK_INDEX)->GetDataType();
        OP_CHECK_IF(gamaKDtype != ge::DT_FLOAT, OP_LOGE(context_->GetNodeName(), "GamaK dtype should be float32."),
                    return ge::GRAPH_FAILED);
    }

    if (context_->GetOptionalInputDesc(ACC_TO_INDEX) != nullptr) {
        accFlag = true;
        auto numAcceptedTokensDtype = context_->GetOptionalInputDesc(ACC_TO_INDEX)->GetDataType();
        OP_CHECK_IF(numAcceptedTokensDtype != ge::DT_INT32,
                    OP_LOGE(context_->GetNodeName(), "NumAcceptedTokens dtype should be int32."),
                    return ge::GRAPH_FAILED);
    }
    auto outDtype = context_->GetOutputDesc(OUT_INDEX)->GetDataType();
    OP_CHECK_IF(outDtype != ge::DT_BF16,
                OP_LOGE(context_->GetNodeName(), "Out dtype should be bfloat16."),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// 检查各个参数在指定维度上的大小是否相等
bool RecurrentGatedDeltaRuleTiling::CheckDimEqual(const gert::Shape a, const int64_t dimA, gert::Shape b, const int64_t dimB,
                                                  const std::string &nameA, const std::string &nameB,
                                                  const std::string &dimDesc)
{
    if (a.GetDim(dimA) != b.GetDim(dimB)) {
        OP_LOGE(context_->GetNodeName(), "The %s of %s and %s should be the same, but %s is %ld while %s is %ld.",
                dimDesc.c_str(), nameA.c_str(), nameB.c_str(), nameA.c_str(), a.GetDim(dimA), nameB.c_str(),
                b.GetDim(dimB));
        return false;
    }
    return true;
}
// 检查维度数量是否符合预期
bool RecurrentGatedDeltaRuleTiling::CheckDim(const gert::Shape shape, const size_t dim, const std::string &dimDesc)
{
    if (shape.GetDimNum() != dim) {
        OP_LOGE(context_->GetNodeName(), "The number of dimensons of %s should be %zu, but it is %zu.",
                dimDesc.c_str(), dim, shape.GetDimNum());
        return false;
    }
    return true;
}

ge::graphStatus RecurrentGatedDeltaRuleTiling::AnalyzeShapesParser()
{
    const auto &queryShape = context_->GetInputShape(QUERY_INDEX)->GetOriginShape();
    const auto &keyShape = context_->GetInputShape(KEY_INDEX)->GetOriginShape();
    const auto &valueShape = context_->GetInputShape(VALUE_INDEX)->GetOriginShape();
    const auto &stateShape = context_->GetInputShape(STATE_INDEX)->GetOriginShape();
    const auto &cuSeqlensShape = context_->GetInputShape(CUSEQLENS_INDEX)->GetOriginShape();

    // T>0
    OP_CHECK_IF(queryShape.GetDim(DIM_0) <=0,
        OP_LOGE(inputParams_.opName, "T should greater than 0, but T is %ld.", queryShape.GetDim(DIM_0)),
            return ge::GRAPH_FAILED);
    // // B>=0
    OP_CHECK_IF((cuSeqlensShape.GetDim(DIM_0) <= 0U),
        OP_LOGE(inputParams_.opName, "B(actualSeaLengths Dim0) should greater than 0, but B is %ld.", cuSeqlensShape.GetDim(DIM_0)),
            return ge::GRAPH_FAILED);
    // nk>0 nk<=256
    OP_CHECK_IF(queryShape.GetDim(DIM_1) <= 0 || queryShape.GetDim(DIM_1) > 256,
        OP_LOGE(inputParams_.opName, "Nk should be greater than 0 and less than or equal to 256, but Nk is %ld.", queryShape.GetDim(DIM_1)),
            return ge::GRAPH_FAILED);
    // nv>0 nv<=256
    OP_CHECK_IF(valueShape.GetDim(DIM_1) <= 0 || valueShape.GetDim(DIM_1) > 256,
        OP_LOGE(inputParams_.opName, "Nv should be greater than 0 and less than or equal to 256, but Nv is %ld.", valueShape.GetDim(DIM_1)),
            return ge::GRAPH_FAILED);
    // dk>0 dk<=512
    OP_CHECK_IF(queryShape.GetDim(DIM_2) <= 0 || queryShape.GetDim(DIM_2) > 512,
        OP_LOGE(inputParams_.opName, "Dk should be greater than 0 and less than or equal to 512, but Dk is %ld.", queryShape.GetDim(DIM_2)),
            return ge::GRAPH_FAILED);
    // dv>0 dv<=512
    OP_CHECK_IF(valueShape.GetDim(DIM_2) <= 0 || valueShape.GetDim(DIM_2) > 512,
        OP_LOGE(inputParams_.opName, "Dv should be greater than 0 and less than or equal to 512, but Dv is %ld.", valueShape.GetDim(DIM_2)),
            return ge::GRAPH_FAILED);
    // nv>=nk
    OP_CHECK_IF(valueShape.GetDim(DIM_1) % keyShape.GetDim(DIM_1) != 0,
        OP_LOGE(inputParams_.opName, "Nv should be an integer multiple of Nk, but Nv is %ld, Nk is %ld.",
                valueShape.GetDim(DIM_1), keyShape.GetDim(DIM_1)),
            return ge::GRAPH_FAILED);
    // blockNum >= T
    OP_CHECK_IF(stateShape.GetDim(DIM_0) < queryShape.GetDim(DIM_0),
        OP_LOGE(inputParams_.opName, "BlockNum should be greater than or equal to T, Current values: BlockNum=%ld, T=%ld.",
                stateShape.GetDim(DIM_0), queryShape.GetDim(DIM_0)),
            return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RecurrentGatedDeltaRuleTiling::AnalyzeEmptyTensor()
{
    OP_CHECK_IF(context_->GetInputShape(QUERY_INDEX)->GetStorageShape().GetShapeSize() == 0,
        OP_LOGE(inputParams_.opName, "Query not support empty tensor."),
            return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->GetInputShape(KEY_INDEX)->GetStorageShape().GetShapeSize() == 0,
        OP_LOGE(inputParams_.opName, "Key not support empty tensor."),
            return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->GetInputShape(VALUE_INDEX)->GetStorageShape().GetShapeSize() == 0,
        OP_LOGE(inputParams_.opName, "Value not support empty tensor."),
            return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->GetInputShape(BETA_INDEX)->GetStorageShape().GetShapeSize() == 0,
        OP_LOGE(inputParams_.opName, "Beta not support empty tensor."),
            return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->GetInputShape(STATE_INDEX)->GetStorageShape().GetShapeSize() == 0,
        OP_LOGE(inputParams_.opName, "State not support empty tensor."),
            return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->GetInputShape(CUSEQLENS_INDEX)->GetStorageShape().GetShapeSize() == 0,
        OP_LOGE(inputParams_.opName, "ActualSeqLengths not support empty tensor."),
            return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->GetInputShape(SSM_STATE_INDICES_INDEX)->GetStorageShape().GetShapeSize() == 0,
        OP_LOGE(inputParams_.opName, "SsmStateIndices not support empty tensor."),
            return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->GetOutputShape(OUT_INDEX)->GetStorageShape().GetShapeSize() == 0,
        OP_LOGE(inputParams_.opName, "Out not support empty tensor."),
            return ge::GRAPH_FAILED);
    OP_CHECK_IF(gamaFlag && context_->GetOptionalInputShape(G_INDEX)->GetStorageShape().GetShapeSize() == 0,
        OP_LOGE(inputParams_.opName, "G not support empty tensor."),
            return ge::GRAPH_FAILED);
    OP_CHECK_IF(gamaKFlag && context_->GetOptionalInputShape(GK_INDEX)->GetStorageShape().GetShapeSize() == 0,
        OP_LOGE(inputParams_.opName, "Gk not support empty tensor."),
            return ge::GRAPH_FAILED);
    OP_CHECK_IF(accFlag && context_->GetOptionalInputShape(ACC_TO_INDEX)->GetStorageShape().GetShapeSize() == 0,
        OP_LOGE(inputParams_.opName, "NumAcceptedTokens not support empty tensor."),
            return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RecurrentGatedDeltaRuleTiling::AnalyzeShapes()
{
    const auto &queryShape = context_->GetInputShape(QUERY_INDEX)->GetOriginShape();
    const auto &keyShape = context_->GetInputShape(KEY_INDEX)->GetOriginShape();
    const auto &valueShape = context_->GetInputShape(VALUE_INDEX)->GetOriginShape();
    const auto &betaShape = context_->GetInputShape(BETA_INDEX)->GetOriginShape();
    const auto &stateShape = context_->GetInputShape(STATE_INDEX)->GetOriginShape();
    const auto &cuSeqlensShape = context_->GetInputShape(CUSEQLENS_INDEX)->GetOriginShape();
    const auto &ssmStateShape = context_->GetInputShape(SSM_STATE_INDICES_INDEX)->GetOriginShape();
    const auto &outShape = context_->GetOutputShape(OUT_INDEX)->GetOriginShape();

    if (!CheckDim(queryShape, QKV_DIM_NUM, "query") || !CheckDim(keyShape, QKV_DIM_NUM, "key") ||
        !CheckDim(valueShape, QKV_DIM_NUM, "value") || !CheckDim(betaShape, BETA_DIM_NUM, "beta") ||
        !CheckDim(stateShape, STATE_DIM_NUM, "state") ||
        !CheckDim(cuSeqlensShape, CUSEQLENS_DIM_NUM, "actual_seq_lengths") ||
        !CheckDim(ssmStateShape, SSM_STATE_INDICES_DIM_NUM, "ssm_state_indices") ||
        !CheckDim(outShape, OUT_NUM, "out")) {
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
        !CheckDimEqual(stateShape, DIM_3, queryShape, DIM_2, "state", "query", "Dk dimension") ||
        !CheckDimEqual(outShape, DIM_0, queryShape, DIM_0, "out", "query", "T dimension") ||
        !CheckDimEqual(outShape, DIM_1, valueShape, DIM_1, "out", "value", "Nv dimension") ||
        !CheckDimEqual(outShape, DIM_2, valueShape, DIM_2, "out", "value", "Dv dimension")) {
        return ge::GRAPH_FAILED;
    }
    if (gamaFlag &&
        (!CheckDim(context_->GetOptionalInputShape(G_INDEX)->GetOriginShape(), G_DIM_NUM, "G") || 
         !CheckDimEqual(context_->GetOptionalInputShape(G_INDEX)->GetOriginShape(), DIM_0, queryShape, DIM_0, "g", "query", "T dimension") ||
         !CheckDimEqual(context_->GetOptionalInputShape(G_INDEX)->GetOriginShape(), DIM_1, valueShape, DIM_1, "g", "value", "Nv dimension"))){
        return ge::GRAPH_FAILED;
    }
    if (gamaKFlag &&
        (!CheckDim(context_->GetOptionalInputShape(GK_INDEX)->GetOriginShape(), GK_DIM_NUM, "GK") || 
         !CheckDimEqual(context_->GetOptionalInputShape(GK_INDEX)->GetOriginShape(), DIM_0, queryShape, DIM_0, "gk", "query", "T dimension") ||
         !CheckDimEqual(context_->GetOptionalInputShape(GK_INDEX)->GetOriginShape(), DIM_1, valueShape, DIM_1, "gk", "value", "Nv dimension") ||
         !CheckDimEqual(context_->GetOptionalInputShape(GK_INDEX)->GetOriginShape(), DIM_2, keyShape, DIM_2, "gk", "key", "Dk dimension"))){
        return ge::GRAPH_FAILED;
    }
    if (accFlag && !CheckDim(context_->GetOptionalInputShape(ACC_TO_INDEX)->GetOriginShape(), ACC_DIM_NUM, "numAcceptedTokens")){
        return ge::GRAPH_FAILED;
    }

    tilingData_.t = queryShape.GetDim(DIM_0);
    tilingData_.nk = queryShape.GetDim(DIM_1);
    tilingData_.dk = queryShape.GetDim(DIM_2);
    tilingData_.nv = valueShape.GetDim(DIM_1);
    tilingData_.dv = valueShape.GetDim(DIM_2);
    tilingData_.sBlockNum = stateShape.GetDim(DIM_0);
    tilingData_.b = cuSeqlensShape.GetDim(DIM_0);
    return ge::GRAPH_SUCCESS;
}

// 检查是否为FormatND格式
bool RecurrentGatedDeltaRuleTiling::CheckFormat(ge::Format format, const std::string &Desc)
{
    if (format == ge::FORMAT_FRACTAL_NZ) {
        OP_LOGE(context_->GetNodeName(), "%s format not support NZ", Desc.c_str());
        return false;
    }
    return true;
}

ge::graphStatus RecurrentGatedDeltaRuleTiling::AnalyzeFormat()
{
    if (!CheckFormat(context_->GetInputDesc(QUERY_INDEX)->GetStorageFormat(), "Query") ||
        !CheckFormat(context_->GetInputDesc(KEY_INDEX)->GetStorageFormat(), "Key") ||
        !CheckFormat(context_->GetInputDesc(VALUE_INDEX)->GetStorageFormat(), "Value") ||
        !CheckFormat(context_->GetInputDesc(STATE_INDEX)->GetStorageFormat(), "State") ||
        !CheckFormat(context_->GetInputDesc(CUSEQLENS_INDEX)->GetStorageFormat(), "ActualSeqLengths") ||
        !CheckFormat(context_->GetInputDesc(SSM_STATE_INDICES_INDEX)->GetStorageFormat(), "SsmStateIndices")) {
        return ge::GRAPH_FAILED;
    }

    if (context_->GetOptionalInputDesc(G_INDEX) != nullptr) {
        auto gamaFormat = context_->GetOptionalInputDesc(G_INDEX)->GetStorageFormat();
        OP_CHECK_IF(gamaFormat == ge::FORMAT_FRACTAL_NZ, OP_LOGE(context_->GetNodeName(), "G format not support NZ."),
                    return ge::GRAPH_FAILED);
    }
    if (context_->GetOptionalInputDesc(GK_INDEX) != nullptr) {
        auto gamaKFormat = context_->GetOptionalInputDesc(GK_INDEX)->GetStorageFormat();
        OP_CHECK_IF(gamaKFormat == ge::FORMAT_FRACTAL_NZ, OP_LOGE(context_->GetNodeName(), "GK format not support NZ."),
                    return ge::GRAPH_FAILED);
    }
    if (context_->GetOptionalInputDesc(ACC_TO_INDEX) != nullptr) {
        auto numAcceptedTokensFormat = context_->GetOptionalInputDesc(ACC_TO_INDEX)->GetStorageFormat();
        OP_CHECK_IF(numAcceptedTokensFormat == ge::FORMAT_FRACTAL_NZ,
                    OP_LOGE(context_->GetNodeName(), "NumAcceptedTokens format not support NZ."), return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RecurrentGatedDeltaRuleTiling::GetScale()
{
    auto attrs = context_->GetAttrs();
    float scaleValue = *attrs->GetAttrPointer<float>(0);
    tilingData_.scale = scaleValue;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RecurrentGatedDeltaRuleTiling::GetOptionalInput()
{
    if (context_->GetOptionalInputDesc(G_INDEX) == nullptr) {
        tilingData_.hasGama = 0;
    } else {
        tilingData_.hasGama = 1;
    }
    if (context_->GetOptionalInputDesc(GK_INDEX) == nullptr) {
        tilingData_.hasGamaK = 0;
    } else {
        tilingData_.hasGamaK = 1;
    }
    if (context_->GetOptionalInputDesc(ACC_TO_INDEX) == nullptr) {
        tilingData_.hasAcceptedTokens = 0;
    } else {
        tilingData_.hasAcceptedTokens = 1;
    }

    return ge::GRAPH_SUCCESS;
}

void RecurrentGatedDeltaRuleTiling::PrintTilingData()
{
    OP_LOGD(context_->GetNodeName(), "vectorCoreNum: [%u]", tilingData_.vectorCoreNum);
    OP_LOGD(context_->GetNodeName(), "ubCalSize: [%u]", tilingData_.ubCalSize);
    OP_LOGD(context_->GetNodeName(), "ubRestBytes: [%u]", tilingData_.ubRestBytes);
    OP_LOGD(context_->GetNodeName(), "t: [%u]", tilingData_.t);
    OP_LOGD(context_->GetNodeName(), "nk: [%u]", tilingData_.nk);
    OP_LOGD(context_->GetNodeName(), "dk: [%u]", tilingData_.dk);
    OP_LOGD(context_->GetNodeName(), "nv: [%u]", tilingData_.nv);
    OP_LOGD(context_->GetNodeName(), "dv: [%u]", tilingData_.dv);
    OP_LOGD(context_->GetNodeName(), "sBlockNum: [%u]", tilingData_.sBlockNum);
    OP_LOGD(context_->GetNodeName(), "b: [%u]", tilingData_.b);
    OP_LOGD(context_->GetNodeName(), "vStep: [%u]", tilingData_.vStep);
    OP_LOGD(context_->GetNodeName(), "scale: [%f]", tilingData_.scale);
    OP_LOGD(context_->GetNodeName(), "hasGama: [%u]", tilingData_.hasGama);
    OP_LOGD(context_->GetNodeName(), "hasAcceptedTokens: [%u]", tilingData_.hasAcceptedTokens);
}

ge::graphStatus RecurrentGatedDeltaRuleTiling::CalUbSize()
{
    int64_t ubSize = compileInfo_.ubSize;
    int64_t aNv = Ops::Base::CeilAlign(tilingData_.nv, static_cast<uint32_t>(16)); // 16 * 2 = 32B
    int64_t aDv = Ops::Base::CeilAlign(tilingData_.dv, static_cast<uint32_t>(16)); // 16 * 2 = 32B
    int64_t aDk = Ops::Base::CeilAlign(tilingData_.dk, static_cast<uint32_t>(16)); // 16 * 2 = 32B
    int64_t usedUbBytes = MAX_MTP * (4 * aDk + 2 * aDv); // 4 for qInQueue_ & kInQueue_, 2 for vInQueue_
    usedUbBytes += 128;                                  // reserve 128 Bytes
    if (tilingData_.hasGamaK) {
        usedUbBytes += MAX_MTP * 4 * aDk; // 4 for gk gamaInQueue_
    }
    if (tilingData_.hasGama) {
        usedUbBytes += MAX_MTP * 4 * aNv; // 4 for g gamaInQueue_
    }
    usedUbBytes += MAX_MTP * 2 * aNv; // 2 for betaInQueue_
    tilingData_.ubRestBytes = ubSize - usedUbBytes;
    usedUbBytes += MAX_MTP * (8 * aDk + 4 * aDv + 4 * aNv); // 8 for qk in ub, 4 for v in ub, 4 for beta in ub
    int64_t coeff = (2 + 2) * aDk + 4;                      // 2 for stateInQueue_, stateOutQueue_, 4 for attnOutQueue_
    coeff += (4 + 4) * aDk + 4 + 4;                         // 4 for qInUb, kInUb, vInUb, deltaInUb, attnInUb
    int64_t vStep = (ubSize - usedUbBytes) / coeff / 8 * 8; // 8 * sizeof(float) = 32
    if (vStep < 8) {                                        // vStep不小于8
        OP_LOGE(context_->GetNodeName(), "vStep should be bigger than 8, shape is too big.");
        return ge::GRAPH_FAILED;
    }
    int64_t rptime = Ops::Base::CeilDiv(tilingData_.dv, static_cast<uint32_t>(vStep));
    vStep = Ops::Base::CeilAlign(Ops::Base::CeilDiv(tilingData_.dv, static_cast<uint32_t>(rptime)),
                                 static_cast<uint32_t>(8)); // 8 * sizeof(float) = 32
    tilingData_.ubCalSize = compileInfo_.ubSize;
    tilingData_.vStep = vStep;
    tilingData_.ubRestBytes -= ((2 + 2) * aDk + 4) * vStep; // 2 for stateInQueue_, stateOutQueue_, 4 for attnOutQueue_

    return ge::GRAPH_SUCCESS;
}


static ge::graphStatus RecurrentGatedDeltaRuleTilingFunc(gert::TilingContext *context)
{
    OP_CHECK_IF(context == nullptr, OPS_REPORT_CUBE_INNER_ERR("RecurrentGatedDeltaRule", "context is null."),
                return ge::GRAPH_FAILED);
    return Ops::Transformer::OpTiling::TilingRegistry::GetInstance().DoTilingImpl(context);
}

static ge::graphStatus TilingPrepareForRecurrentGatedDeltaRule(gert::TilingParseContext *context)
{
    OP_CHECK_IF(context == nullptr, OPS_REPORT_CUBE_INNER_ERR("RecurrentGatedDeltaRule", "context is null."),
                return ge::GRAPH_FAILED);

    fe::PlatFormInfos *platformInfo = context->GetPlatformInfo();
    OP_CHECK_IF(platformInfo == nullptr, OPS_REPORT_CUBE_INNER_ERR(context->GetNodeName(), "platformInfoPtr is null."),
                return ge::GRAPH_FAILED);

    auto compileInfoPtr = context->GetCompiledInfo<RecurrentGatedDeltaRuleCompileInfo>();
    OP_CHECK_IF(compileInfoPtr == nullptr, OPS_REPORT_CUBE_INNER_ERR(context->GetNodeName(), "compileInfoPtr is null."),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(RecurrentGatedDeltaRule)
    .Tiling(RecurrentGatedDeltaRuleTilingFunc)
    .TilingParse<RecurrentGatedDeltaRuleCompileInfo>(TilingPrepareForRecurrentGatedDeltaRule);
} // namespace optiling
