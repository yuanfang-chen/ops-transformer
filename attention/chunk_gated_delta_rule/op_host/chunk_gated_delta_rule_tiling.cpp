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
 * \file chunk_gated_delta_rule_tiling.cpp
 * \brief
 */

#include "chunk_gated_delta_rule_tiling.h"

#include "tiling_base/tiling_templates_registry.h"
#include "register/op_def_registry.h"
#include "platform/platform_infos_def.h"
#include "err/ops_err.h"
#include "log/log.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
REGISTER_OPS_TILING_TEMPLATE(ChunkGatedDeltaRule, ChunkGatedDeltaRuleTiling, 0);

const size_t QUERY_INDEX = 0;
const size_t KEY_INDEX = 1;
const size_t VALUE_INDEX = 2;
const size_t BETA_INDEX = 3;
const size_t STATE_INDEX = 4;
const size_t ACTUAL_SEQ_LENGTHS_INDEX = 5;
const size_t G_INDEX = 6;

const size_t OUTPUT_OUT_IDX = 0;
const size_t OUTPUT_FINAL_STATE_IDX = 1;

const size_t QKV_DIM_NUM = 3;
const size_t BETA_DIM_NUM = 2;
const size_t STATE_DIM_NUM = 4;
const size_t ACTUAL_SEQ_LENGTHS_DIM_NUM = 1;
const size_t G_DIM_NUM = 2;

const size_t DIM_0 = 0;
const size_t DIM_1 = 1;
const size_t DIM_2 = 2;
const size_t DIM_3 = 3;

// 固定系统 workspace 大小（16 MB）
constexpr int64_t SYS_WORKSPACE_SIZE = 16777216;

// Matmul tiling 相关常量
constexpr uint32_t MATMUL_BASE_M = 128;
constexpr uint32_t MATMUL_BASE_K = 128;
constexpr uint32_t MATMUL_BASE_N = 128;

// 初始化编译信息, 读取平台资源, 并缓存核数到 tilingData_
void ChunkGatedDeltaRuleTiling::InitCompileInfo()
{
    auto platformInfoPtr = context_->GetPlatformInfo();
    if (platformInfoPtr == nullptr) {
        OP_LOGE(context_->GetNodeName(), "platformInfoPtr is null");
        return;
    }
    const auto &ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfo_.ubSize);
    compileInfo_.aivNum = ascendcPlatform.GetCoreNumAiv();
    compileInfo_.aicNum = ascendcPlatform.GetCoreNumAic();

    if (compileInfo_.aivNum <= 0 || compileInfo_.aicNum <= 0) {
        OP_LOGE(context_->GetNodeName(), "aivNum <= 0 or aicNum <= 0");
        return;
    }
    tilingData_.aiCoreNum = compileInfo_.aicNum;
}

ge::graphStatus ChunkGatedDeltaRuleTiling::GetPlatformInfo()
{
    return ge::GRAPH_SUCCESS;
};

// 获取输入输出信息，依次完成上下文、dtype、shape、属性、可选输入和 format 校验
ge::graphStatus ChunkGatedDeltaRuleTiling::GetShapeAttrsInfo()
{
    OP_CHECK_IF(CheckContext() != ge::GRAPH_SUCCESS, OP_LOGE(inputParams_.opName, "Invalid context."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(GetOptionalInput() != ge::GRAPH_SUCCESS, OP_LOGE(inputParams_.opName, "Invalid GetOptionalInput."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(GetScale() != ge::GRAPH_SUCCESS, OP_LOGE(inputParams_.opName, "Invalid GetScale."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(AnalyzeDtype() != ge::GRAPH_SUCCESS, OP_LOGE(inputParams_.opName, "Invalid dtypes."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(AnalyzeShapes() != ge::GRAPH_SUCCESS, OP_LOGE(inputParams_.opName, "Invalid shapes."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(AnalyzeFormat() != ge::GRAPH_SUCCESS, OP_LOGE(inputParams_.opName, "Invalid Format."),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ChunkGatedDeltaRuleTiling::DoOpTiling()
{
    int64_t c = 64; // chunk size 取 64
    int64_t p = 2;  // 一个 chunk 组中，单核最大处理 chunk 数
    tilingData_.chunkSize = c;
    tilingData_.maxGroupLength = p * tilingData_.aiCoreNum * tilingData_.chunkSize;
    tilingData_.stageOneParaNum = 2; // stage1 并行数

    tilingData_.interWorkspaceSz = 0;
    int64_t sizeHigh = ge::GetSizeByDataType(ge::DT_FLOAT);
    int64_t nv = tilingData_.nv;
    int64_t dv = tilingData_.dv;
    int64_t dk = tilingData_.dk;
    int64_t s = tilingData_.maxGroupLength;
    int64_t b = tilingData_.b;
    tilingData_.interWorkspaceSz += sizeHigh * nv * s;                            // gCumExp
    tilingData_.interWorkspaceSz += sizeHigh * nv * s * dk;                       // kCumDecay
    tilingData_.interWorkspaceSz += sizeHigh * nv * s * dv;                       // vInner
    tilingData_.interWorkspaceSz += sizeHigh * nv * s * dk;                       // qPrime
    tilingData_.interWorkspaceSz += sizeHigh * nv * s * dv;                       // attnInter
    tilingData_.interWorkspaceSz += sizeHigh * nv * s * dk;                       // kg
    tilingData_.interWorkspaceSz += sizeHigh * nv * s * c;                        // qkt
    tilingData_.interWorkspaceSz += sizeHigh * b * nv * dv * dk;                  // highState
    tilingData_.interWorkspaceSz += sizeHigh * c * c * tilingData_.aiCoreNum * 4; // mask

    // stage1 临时变量空间
    tilingData_.stageWorkspaceSz = sizeHigh * c * (2 * c + 3 * dk + dv) * tilingData_.stageOneParaNum;
    tilingData_.stageWorkspaceSz *= tilingData_.aiCoreNum;

    PrintTilingData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ChunkGatedDeltaRuleTiling::DoMatmulTiling()
{
    // 计算 baseM、baseN、baseK
    uint32_t baseM = MATMUL_BASE_M;
    uint32_t baseK = MATMUL_BASE_K;
    uint32_t baseN = MATMUL_BASE_N;

    // ========== MT_FP32: FP32 -> FP32 ==========
    matmul_tiling::MultiCoreMatmulTiling mm_;
    const auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
    uint64_t ubSize;
    uint64_t l1Size;
    uint64_t l0CSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, l0CSize);
    mm_.SetBufferSpace(l1Size, l0CSize, ubSize);
    mm_.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT, true);
    mm_.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT, true);
    mm_.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    mm_.SetBias(false);
    mm_.SetDim(1);
    mm_.SetShape(baseM, baseN, baseK);
    mm_.SetOrgShape(baseM, baseN, baseK);
    mm_.SetFixSplit(baseM, baseN, baseK);
    if (mm_.GetTiling(tilingData_.matmulTilingFp32) == -1) {
        OP_LOGE(context_->GetNodeName(), "CGDR: Get Tiling Failed!");
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context_->GetNodeName(), "CGDR: baseM is %d, baseK is %d, baseN is %d.", baseM, baseK, baseN);
    tilingData_.matmulTilingFp32.dbL0C = 1;
    tilingData_.matmulTilingFp32.stepKa = 1;
    tilingData_.matmulTilingFp32.stepKb = 1;
    tilingData_.matmulTilingFp32.depthA1 = 1;
    tilingData_.matmulTilingFp32.depthB1 = 1;
    tilingData_.matmulTilingFp32.stepM = 1;
    tilingData_.matmulTilingFp32.stepN = 1;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ChunkGatedDeltaRuleTiling::DoLibApiTiling()
{
    tilingKey_ = 0;

    // 执行 matmul tiling
    if (DoMatmulTiling() != ge::GRAPH_SUCCESS) {
        OP_LOGE(context_->GetNodeName(), "DoMatmulTiling failed");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
};

// 返回 tiling key
uint64_t ChunkGatedDeltaRuleTiling::GetTilingKey() const
{
    return tilingKey_;
};

// 计算 workspace 大小
ge::graphStatus ChunkGatedDeltaRuleTiling::GetWorkspaceSize()
{
    workspaceSize_ = SYS_WORKSPACE_SIZE;
    workspaceSize_ += tilingData_.interWorkspaceSz;
    workspaceSize_ += tilingData_.stageWorkspaceSz;
    return ge::GRAPH_SUCCESS;
};

// 写回 tilingData 和 workspace 信息
ge::graphStatus ChunkGatedDeltaRuleTiling::PostTiling()
{
    context_->SetBlockDim(tilingData_.aiCoreNum);

    auto tilingDataSize = sizeof(ChunkGatedDeltaRule::ChunkGatedDeltaRuleTilingData);
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

// 校验上下文中必选输入、可选输入和输出是否存在
ge::graphStatus ChunkGatedDeltaRuleTiling::CheckContext()
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

    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(ACTUAL_SEQ_LENGTHS_INDEX));
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(ACTUAL_SEQ_LENGTHS_INDEX));

    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetOutputShape(OUTPUT_OUT_IDX));
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetOutputDesc(OUTPUT_OUT_IDX));

    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetOutputShape(OUTPUT_FINAL_STATE_IDX));
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetOutputDesc(OUTPUT_FINAL_STATE_IDX));

    auto gDesc = context_->GetOptionalInputDesc(G_INDEX);
    auto gShape = context_->GetOptionalInputShape(G_INDEX);
    OP_CHECK_IF((gDesc == nullptr) != (gShape == nullptr),
                OP_LOGE(context_->GetNodeName(), "gamma desc and shape should both exist or both be null"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// 校验输入输出 dtype：q/k/v/beta/state/out/final_state 为 bf16，actual_seq_lengths 为 int32，可选 g 为 float
ge::graphStatus ChunkGatedDeltaRuleTiling::AnalyzeDtype()
{
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

    auto actualSeqLengthsDtype = context_->GetInputDesc(ACTUAL_SEQ_LENGTHS_INDEX)->GetDataType();
    OP_CHECK_IF(actualSeqLengthsDtype != ge::DT_INT32,
                OP_LOGE(context_->GetNodeName(), "actual_seq_lengths dtype should be int32"), return ge::GRAPH_FAILED);

    if (tilingData_.hasGamma != 0) {
        auto gammaDtype = context_->GetOptionalInputDesc(G_INDEX)->GetDataType();
        OP_CHECK_IF(gammaDtype != ge::DT_FLOAT, OP_LOGE(context_->GetNodeName(), "gamma dtype should be float32"),
                    return ge::GRAPH_FAILED);
    }

    auto outDtype = context_->GetOutputDesc(OUTPUT_OUT_IDX)->GetDataType();
    auto finalStateDtype = context_->GetOutputDesc(OUTPUT_FINAL_STATE_IDX)->GetDataType();
    OP_CHECK_IF(outDtype != ge::DT_BF16, OP_LOGE(context_->GetNodeName(), "output dtype should be bfloat16"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(finalStateDtype != ge::DT_BF16,
                OP_LOGE(context_->GetNodeName(), "final_state dtype should be bfloat16"), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// 校验两个 shape 指定维度是否相等，用于跨输入一致性检查
bool ChunkGatedDeltaRuleTiling::CheckDimEqual(const gert::Shape &a, const int64_t dimA, const gert::Shape &b,
                                              const int64_t dimB, const std::string &nameA, const std::string &nameB,
                                              const std::string &dimDesc)
{
    if (a.GetDim(dimA) != b.GetDim(dimB)) {
        OP_LOGE(context_->GetNodeName(), "The %s of %s and %s should be the same, but %s is %ld while %s is %ld",
                dimDesc.c_str(), nameA.c_str(), nameB.c_str(), nameA.c_str(), a.GetDim(dimA), nameB.c_str(),
                b.GetDim(dimB));
        return false;
    }
    return true;
}

// 校验 shape 的维度数是否符合预期
bool ChunkGatedDeltaRuleTiling::CheckDim(const gert::Shape &shape, const size_t dim, const std::string &dimDesc)
{
    if (shape.GetDimNum() != dim) {
        OP_LOGE(context_->GetNodeName(), "The number of dimensions of %s should be %zu, but it is %zu", dimDesc.c_str(),
                dim, shape.GetDimNum());
        return false;
    }
    return true;
}

// 校验输入是否为空，其次检查 shape 各维度是否大于 0
bool ChunkGatedDeltaRuleTiling::CheckShapeNotEmpty(const gert::Shape &shape, const std::string &shapeName)
{
    OP_CHECK_IF(shape.GetDimNum() == 0, OP_LOGE(context_->GetNodeName(), "%s shape is empty", shapeName.c_str()),
                return false);

    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        OP_CHECK_IF(shape.GetDim(i) <= 0,
                    OP_LOGE(context_->GetNodeName(), "%s dim %zu should be greater than 0, but got %ld",
                            shapeName.c_str(), i, shape.GetDim(i)),
                    return false);
    }
    return true;
}

// 统一校验输入 shape/rank 约束和跨输入维度关系
ge::graphStatus ChunkGatedDeltaRuleTiling::CheckInputShapeConstraints(
    const gert::Shape &queryShape, const gert::Shape &keyShape, const gert::Shape &valueShape,
    const gert::Shape &betaShape, const gert::Shape &stateShape, const gert::Shape &actualSeqLengthsShape,
    const gert::Shape *gShape)
{
    if (!CheckShapeNotEmpty(queryShape, "query") || !CheckShapeNotEmpty(keyShape, "key") ||
        !CheckShapeNotEmpty(valueShape, "value") || !CheckShapeNotEmpty(betaShape, "beta") ||
        !CheckShapeNotEmpty(stateShape, "state") || !CheckShapeNotEmpty(actualSeqLengthsShape, "actual_seq_lengths")) {
        return ge::GRAPH_FAILED;
    }

    if (!CheckDim(queryShape, QKV_DIM_NUM, "query") || !CheckDim(keyShape, QKV_DIM_NUM, "key") ||
        !CheckDim(valueShape, QKV_DIM_NUM, "value") || !CheckDim(betaShape, BETA_DIM_NUM, "beta") ||
        !CheckDim(stateShape, STATE_DIM_NUM, "state") ||
        !CheckDim(actualSeqLengthsShape, ACTUAL_SEQ_LENGTHS_DIM_NUM, "actual_seq_lengths")) {
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
        !CheckDimEqual(actualSeqLengthsShape, DIM_0, stateShape, DIM_0, "actual_seq_lengths", "state", "B dimension")) {
        return ge::GRAPH_FAILED;
    }

    if (gShape != nullptr) {
        if (!CheckShapeNotEmpty(*gShape, "g") || !CheckDim(*gShape, G_DIM_NUM, "g") ||
            !CheckDimEqual(*gShape, DIM_0, valueShape, DIM_0, "g", "value", "T dimension") ||
            !CheckDimEqual(*gShape, DIM_1, valueShape, DIM_1, "g", "value", "Nv dimension")) {
            return ge::GRAPH_FAILED;
        }
        return ge::GRAPH_SUCCESS;
    }

    return ge::GRAPH_SUCCESS;
}

// 校验输出 shape 和输出维度是否具有一致性
ge::graphStatus ChunkGatedDeltaRuleTiling::CheckOutputShapeConstraints(const gert::Shape &valueShape,
                                                                       const gert::Shape &stateShape,
                                                                       const gert::Shape &outShape,
                                                                       const gert::Shape &finalStateShape)
{
    if (!CheckDim(outShape, QKV_DIM_NUM, "out") || !CheckDim(finalStateShape, STATE_DIM_NUM, "final_state")) {
        return ge::GRAPH_FAILED;
    }

    if (!CheckDimEqual(outShape, DIM_0, valueShape, DIM_0, "out", "value", "T dimension") ||
        !CheckDimEqual(outShape, DIM_1, valueShape, DIM_1, "out", "value", "Nv dimension") ||
        !CheckDimEqual(outShape, DIM_2, valueShape, DIM_2, "out", "value", "Dv dimension") ||
        !CheckDimEqual(finalStateShape, DIM_0, stateShape, DIM_0, "final_state", "state", "B dimension") ||
        !CheckDimEqual(finalStateShape, DIM_1, stateShape, DIM_1, "final_state", "state", "Nv dimension") ||
        !CheckDimEqual(finalStateShape, DIM_2, stateShape, DIM_2, "final_state", "state", "Dv dimension") ||
        !CheckDimEqual(finalStateShape, DIM_3, stateShape, DIM_3, "final_state", "state", "Dk dimension")) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

// 校验从 shape 推导出的维度约束
ge::graphStatus ChunkGatedDeltaRuleTiling::CheckDerivedDimConstraints()
{
    OP_CHECK_IF(
        tilingData_.t <= 0 || tilingData_.b <= 0 || tilingData_.nk <= 0 || tilingData_.dk <= 0 || tilingData_.nv <= 0 ||
            tilingData_.dv <= 0,
        OP_LOGE(inputParams_.opName,
                "T, B, Nk, Dk, Nv, Dv should be greater than 0, but T=%ld, B=%ld, Nk=%ld, Dk=%ld, Nv=%ld, Dv=%ld",
                tilingData_.t, tilingData_.b, tilingData_.nk, tilingData_.dk, tilingData_.nv, tilingData_.dv),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(tilingData_.nk > 64 || tilingData_.nv > 64,
                OP_LOGE(inputParams_.opName, "nk and nv should no bigger than 64, but nk is %ld, nv is %ld",
                        tilingData_.nk, tilingData_.nv),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(tilingData_.dv > 128 || tilingData_.dk > 128,
                OP_LOGE(inputParams_.opName, "dv and dk should be no bigger than 128, but dv is %ld, dk is %ld",
                        tilingData_.dv, tilingData_.dk),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(tilingData_.nv % tilingData_.nk != 0,
                OP_LOGE(inputParams_.opName, "nv should be an integer multiple of nk, but nv is %ld, nk is %ld",
                        tilingData_.nv, tilingData_.nk),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// 读取输入输出 shape，完成基础校验，并填充 tilingData_ 中的维度信息
ge::graphStatus ChunkGatedDeltaRuleTiling::AnalyzeShapes()
{
    const auto &queryShape = context_->GetInputShape(QUERY_INDEX)->GetOriginShape();
    const auto &keyShape = context_->GetInputShape(KEY_INDEX)->GetOriginShape();
    const auto &valueShape = context_->GetInputShape(VALUE_INDEX)->GetOriginShape();
    const auto &betaShape = context_->GetInputShape(BETA_INDEX)->GetOriginShape();
    const auto &stateShape = context_->GetInputShape(STATE_INDEX)->GetOriginShape();
    const auto &actualSeqLengthsShape = context_->GetInputShape(ACTUAL_SEQ_LENGTHS_INDEX)->GetOriginShape();

    const auto &outShape = context_->GetOutputShape(OUTPUT_OUT_IDX)->GetOriginShape();
    const auto &finalStateShape = context_->GetOutputShape(OUTPUT_FINAL_STATE_IDX)->GetOriginShape();

    const gert::Shape *gShape = nullptr;
    if (tilingData_.hasGamma != 0) {
        gShape = &context_->GetOptionalInputShape(G_INDEX)->GetOriginShape();
    }

    OP_CHECK_IF(CheckInputShapeConstraints(queryShape, keyShape, valueShape, betaShape, stateShape,
                                           actualSeqLengthsShape, gShape) != ge::GRAPH_SUCCESS,
                OP_LOGE(inputParams_.opName, "Invalid input shape constraints."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckOutputShapeConstraints(valueShape, stateShape, outShape, finalStateShape) != ge::GRAPH_SUCCESS,
                OP_LOGE(inputParams_.opName, "Invalid output shape constraints."), return ge::GRAPH_FAILED);

    tilingData_.t = queryShape.GetDim(DIM_0);
    tilingData_.nk = queryShape.GetDim(DIM_1);
    tilingData_.dk = queryShape.GetDim(DIM_2);
    tilingData_.nv = valueShape.GetDim(DIM_1);
    tilingData_.dv = valueShape.GetDim(DIM_2);
    tilingData_.b = actualSeqLengthsShape.GetDim(DIM_0);

    OP_CHECK_IF(CheckDerivedDimConstraints() != ge::GRAPH_SUCCESS,
                OP_LOGE(inputParams_.opName, "Invalid derived dim constraints."), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// tiling 阶段校验 tensor 的主格式。
// 当前链路下，q/k/v 以 NCL 参与 tiling，state 以 NCHW 参与 tiling，
// 其余输入输出保持 ND，因此这里按各 tensor 的允许格式做严格校验。
bool ChunkGatedDeltaRuleTiling::CheckFormat(const gert::CompileTimeTensorDesc *desc, const ge::Format expectFormat0,
                                            const ge::Format expectFormat1, const std::string &name)
{
    auto primaryFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(desc->GetStorageFormat()));
    if (primaryFormat != expectFormat0 && primaryFormat != expectFormat1) {
        if (expectFormat0 == expectFormat1) {
            OP_LOGE(context_->GetNodeName(), "%s format should be %s, but got %s", name.c_str(),
                    ge::TypeUtils::FormatToSerialString(expectFormat0).c_str(),
                    ge::TypeUtils::FormatToSerialString(primaryFormat).c_str());
        } else {
            OP_LOGE(context_->GetNodeName(), "%s format should be %s or %s, but got %s", name.c_str(),
                    ge::TypeUtils::FormatToSerialString(expectFormat0).c_str(),
                    ge::TypeUtils::FormatToSerialString(expectFormat1).c_str(),
                    ge::TypeUtils::FormatToSerialString(primaryFormat).c_str());
        }
        return false;
    }
    return true;
}

// 校验输入 format，可选 gamma 存在时也需要校验
ge::graphStatus ChunkGatedDeltaRuleTiling::AnalyzeFormat()
{
    if (!CheckFormat(context_->GetInputDesc(QUERY_INDEX), ge::FORMAT_ND, ge::FORMAT_NCL, "query") ||
        !CheckFormat(context_->GetInputDesc(KEY_INDEX), ge::FORMAT_ND, ge::FORMAT_NCL, "key") ||
        !CheckFormat(context_->GetInputDesc(VALUE_INDEX), ge::FORMAT_ND, ge::FORMAT_NCL, "value") ||
        !CheckFormat(context_->GetInputDesc(BETA_INDEX), ge::FORMAT_ND, ge::FORMAT_ND, "beta") ||
        !CheckFormat(context_->GetInputDesc(STATE_INDEX), ge::FORMAT_ND, ge::FORMAT_NCHW, "state") ||
        !CheckFormat(context_->GetInputDesc(ACTUAL_SEQ_LENGTHS_INDEX), ge::FORMAT_ND, ge::FORMAT_ND,
                     "actual_seq_lengths") ||
        !CheckFormat(context_->GetOutputDesc(OUTPUT_OUT_IDX), ge::FORMAT_ND, ge::FORMAT_ND, "out") ||
        !CheckFormat(context_->GetOutputDesc(OUTPUT_FINAL_STATE_IDX), ge::FORMAT_ND, ge::FORMAT_ND, "final_state")) {
        return ge::GRAPH_FAILED;
    }

    if (tilingData_.hasGamma != 0) {
        if (!CheckFormat(context_->GetOptionalInputDesc(G_INDEX), ge::FORMAT_ND, ge::FORMAT_ND, "gamma")) {
            return ge::GRAPH_FAILED;
        }
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ChunkGatedDeltaRuleTiling::GetScale()
{
    auto attrs = context_->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OP_LOGE(context_->GetNodeName(), "attrs is null"), return ge::GRAPH_FAILED);
    auto scalePtr = attrs->GetAttrPointer<float>(0);
    OP_CHECK_IF(scalePtr == nullptr, OP_LOGE(context_->GetNodeName(), "scale attr is null"), return ge::GRAPH_FAILED);
    tilingData_.scale = *scalePtr;
    return ge::GRAPH_SUCCESS;
}

// 负责判断 gamma 是否存在，把存在状态写进 tilingData_.hasGamma (0 或 1)
ge::graphStatus ChunkGatedDeltaRuleTiling::GetOptionalInput()
{
    auto gDesc = context_->GetOptionalInputDesc(G_INDEX);
    auto gShapePtr = context_->GetOptionalInputShape(G_INDEX);
    tilingData_.hasGamma =
        (gDesc != nullptr && gShapePtr != nullptr && gShapePtr->GetOriginShape().GetDimNum() != 0) ? 1 : 0;
    return ge::GRAPH_SUCCESS;
}

void ChunkGatedDeltaRuleTiling::PrintTilingData()
{
    OP_LOGD(context_->GetNodeName(), "aiCoreNum: [%ld]", tilingData_.aiCoreNum);
    OP_LOGD(context_->GetNodeName(), "t: [%ld]", tilingData_.t);
    OP_LOGD(context_->GetNodeName(), "nk: [%ld]", tilingData_.nk);
    OP_LOGD(context_->GetNodeName(), "dk: [%ld]", tilingData_.dk);
    OP_LOGD(context_->GetNodeName(), "nv: [%ld]", tilingData_.nv);
    OP_LOGD(context_->GetNodeName(), "dv: [%ld]", tilingData_.dv);
    OP_LOGD(context_->GetNodeName(), "b: [%ld]", tilingData_.b);
    OP_LOGD(context_->GetNodeName(), "hasGamma: [%ld]", tilingData_.hasGamma);
    OP_LOGD(context_->GetNodeName(), "chunkSize: [%ld]", tilingData_.chunkSize);
    OP_LOGD(context_->GetNodeName(), "maxGroupLength: [%ld]", tilingData_.maxGroupLength);
    OP_LOGD(context_->GetNodeName(), "interWorkspaceSz: [%ld]", tilingData_.interWorkspaceSz);
    OP_LOGD(context_->GetNodeName(), "stageWorkspaceSz: [%ld]", tilingData_.stageWorkspaceSz);
    OP_LOGD(context_->GetNodeName(), "stageOneParaNum: [%ld]", tilingData_.stageOneParaNum);
    OP_LOGD(context_->GetNodeName(), "scale: [%f]", tilingData_.scale);
}

// tiling 调度入口
static ge::graphStatus ChunkGatedDeltaRuleTilingFunc(gert::TilingContext *context)
{
    OP_CHECK_IF(context == nullptr, OPS_REPORT_CUBE_INNER_ERR("ChunkGatedDeltaRule", "context is null"),
                return ge::GRAPH_FAILED);
    return Ops::Transformer::OpTiling::TilingRegistry::GetInstance().DoTilingImpl(context);
}

// tiling parse 阶段准备，校验编译信息是否存在
static ge::graphStatus TilingPrepareForChunkGatedDeltaRule(gert::TilingParseContext *context)
{
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
} // namespace optiling
