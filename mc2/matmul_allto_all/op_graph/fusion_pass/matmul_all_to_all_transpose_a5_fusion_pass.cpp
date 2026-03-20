/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "matmul_all_to_all_transpose_a5_fusion_pass.h"
#include "es_MatmulAlltoAll.h" // es autogen header
#include "es_math_ops.h"       // math ops stub
#include "mc2_platform_info.h"
#include "mc2_common_log.h"
#include "ge/ge_utils.h"

namespace ops {
const std::string kPassName = "MatmulAllToAllTransposeA5FusionPass";
const std::string kPatternMC2 = "MatmulAlltoAll";
const std::string kPatternTranspose = "Transpose";
const std::string kPatternTransposeD = "TransposeD";
const int64_t kMatmulAlltoAllCaptureIdx = 0l;
const int64_t kTransposeCaptureIdx = 1l;
const int64_t kMXQuantMode = 6;
const int64_t kTransposePermIdx = 1l;
const int64_t kX2InTransposeNodeIdx = 0l;

static ge::fusion::PatternUniqPtr MakePatternForTranspose()
{
    auto graphBuilder = ge::es::EsGraphBuilder((kPassName + kPatternTranspose).c_str());
    auto [x1, x2, bias, x1Scale, x2Scale, commScale, x1Offset, x2Offset] = graphBuilder.CreateInputs<8>();
    auto transpose = ge::es::Transpose(x2, ge::es::EsTensorLike(std::vector<int64_t>{1, 0}));
    std::string group = std::string(""); // required attr
    int64_t worldSize = 0;               // required attr
    auto mc2 = ge::es::MatmulAlltoAll(x1, transpose, bias, x1Scale, x2Scale, commScale, x1Offset, x2Offset,
                                      group.c_str(), worldSize); // 比对proto原型，带上所有的inputs+requiredAttr
    auto graph = graphBuilder.BuildAndReset({mc2});
    auto pattern = std::make_unique<ge::fusion::Pattern>(std::move(*graph));
    pattern->CaptureTensor({*mc2.GetProducer(), 0}).CaptureTensor({*transpose.GetProducer(), 0});
    return pattern;
}

static ge::fusion::PatternUniqPtr MakePatternForTransposeD()
{
    auto graphBuilder = ge::es::EsGraphBuilder((kPassName + kPatternTransposeD).c_str());
    auto [x1, x2, bias, x1Scale, x2Scale, commScale, x1Offset, x2Offset] = graphBuilder.CreateInputs<8>();
    auto transposeD = ge::es::TransposeD(x2, std::vector<int64_t>{1, 0});
    std::string group = std::string(""); // required attr
    int64_t worldSize = 0;               // required attr
    auto mc2 = ge::es::MatmulAlltoAll(x1, transposeD, bias, x1Scale, x2Scale, commScale, x1Offset, x2Offset,
                                      group.c_str(), worldSize);
    auto graph = graphBuilder.BuildAndReset({mc2});
    auto pattern = std::make_unique<ge::fusion::Pattern>(std::move(*graph));
    pattern->CaptureTensor({*mc2.GetProducer(), 0}).CaptureTensor({*transposeD.GetProducer(), 0});
    return pattern;
}

static bool IsMXQuantMode(const ge::GNode &mc2Node)
{
    int64_t x1QuantMode = 0;
    int64_t x2QuantMode = 0;
    mc2Node.GetAttr("x1_quant_mode", x1QuantMode);
    mc2Node.GetAttr("x2_quant_mode", x2QuantMode);
    return (x1QuantMode == kMXQuantMode && x2QuantMode == kMXQuantMode);
}

static bool GetTransposePerm(const ge::GNode &transposeNode, std::vector<int64_t> &permValue)
{
    ge::AscendString nodeType("");
    transposeNode.GetType(nodeType);
    std::string nodeTypeStr = nodeType.GetString();

    if (nodeTypeStr == kPatternTransposeD) { // return when node is TransposeD
        if (transposeNode.GetAttr("perm", permValue) != ge::GRAPH_SUCCESS) {
            OPS_LOG_W(kPassName.c_str(), "Get perm Attr from TransposeD node failed.");
            return false;
        }
        return !permValue.empty();
    }

    // Transpose 的场景
    ge::TensorDesc permDesc;
    transposeNode.GetInputDesc(kTransposePermIdx, permDesc);
    ge::Shape permShape = permDesc.GetShape();

    if (permShape.GetDimNum() != 1) {
        OPS_LOG_D(kPassName.c_str(), "Transpose permutation dim size must be 1, but got %zu.", permShape.GetDimNum());
        return false;
    }

    ge::Tensor permTensor;
    if (transposeNode.GetInputConstData(kTransposePermIdx, permTensor) != ge::GRAPH_SUCCESS) {
        OPS_LOG_D(kPassName.c_str(), "Failed to get transpose permutation const data.");
        return false;
    }

    ge::DataType permDType = permDesc.GetDataType();
    uint8_t *permData = permTensor.GetData();

    if (permData == nullptr) {
        OPS_LOG_D(kPassName.c_str(), "Transpose permutation data is nullptr.");
        return false;
    }

    size_t size = 0;
    if (permDType == ge::DT_INT32) {
        size = permTensor.GetSize() / sizeof(int32_t);
        for (size_t i = 0; i < size; ++i) {
            permValue.emplace_back(static_cast<int64_t>(*(reinterpret_cast<int32_t *>(permData) + i)));
        }
    } else if (permDType == ge::DT_INT64) {
        size = permTensor.GetSize() / sizeof(int64_t);
        for (size_t i = 0; i < size; ++i) {
            permValue.emplace_back(*(reinterpret_cast<int64_t *>(permData) + i));
        }
    } else {
        OPS_LOG_D(kPassName.c_str(), "Transpose permutation dtype must be int32 or int64.");
        return false;
    }

    return !permValue.empty();
}

static bool IsTransposePermValid(const ge::GNode &transposeNode)
{
    std::vector<int64_t> permValue;
    if (!GetTransposePerm(transposeNode, permValue)) {
        OPS_LOG_D(kPassName.c_str(), "Failed to get transpose permutation.");
        return false;
    }

    const size_t permSize = permValue.size();
    if (permSize != 2UL && permSize != 3UL) {
        OPS_LOG_D(kPassName.c_str(), "Transpose permutation size must be 2 or 3, but got %zu.", permSize);
        return false;
    }

    if ((permValue[permSize - 1UL] != static_cast<int64_t>(permSize - 2UL)) ||
        (permValue[permSize - 2UL] != static_cast<int64_t>(permSize - 1UL))) {
        OPS_LOG_D(kPassName.c_str(), "Transpose permutation last 2 dims must be [1, 0].");
        return false;
    }
    return true;
}

static void GetInputsInfo(const std::vector<ge::fusion::SubgraphInput> &subGraphInputs,
                          std::vector<ge::Shape> &inputShapes, std::vector<ge::DataType> &inputDTypes,
                          std::vector<ge::Format> &inputFormats)
{
    OPS_LOG_D(kPassName.c_str(), "Enter GetInputsInfo for MatmulAllToAllTransposeA5FusionPass");
    for (const auto &subGraphInput : subGraphInputs) {
        auto matchNode = subGraphInput.GetAllInputs().at(0);
        ge::TensorDesc tmpDesc;
        matchNode.node.GetInputDesc(matchNode.index, tmpDesc);
        inputShapes.emplace_back(tmpDesc.GetShape());
        inputDTypes.emplace_back(tmpDesc.GetDataType());
        inputFormats.emplace_back(tmpDesc.GetOriginFormat());
    }
}

static bool InferShapeReplaceGraph(const ge::fusion::GraphUniqPtr &replaceGraph,
                                   const std::vector<ge::fusion::SubgraphInput> &subgraphInputs)
{
    OPS_LOG_D(kPassName.c_str(), "Enter InferShapeReplaceGraph for MatmulAllToAllTransposeA5FusionPass");
    std::vector<ge::Shape> inputShapes;
    for (const auto &subgraphInput : subgraphInputs) {
        auto matchNode = subgraphInput.GetAllInputs().at(0);
        ge::TensorDesc tmpDesc;
        matchNode.node.GetInputDesc(matchNode.index, tmpDesc);
        inputShapes.emplace_back(tmpDesc.GetShape());
    }
    return (ge::GeUtils::InferShape(*replaceGraph, inputShapes) == ge::SUCCESS);
}

std::vector<ge::fusion::PatternUniqPtr> MatmulAllToAllTransposeA5FusionPass::Patterns()
{
    OPS_LOG_I(kPassName.c_str(), "Enter Patterns for MatmulAllToAllTransposeA5FusionPass");
    std::vector<ge::fusion::PatternUniqPtr> patternGraphs;
    patternGraphs.emplace_back(MakePatternForTranspose());
    patternGraphs.emplace_back(MakePatternForTransposeD());
    return patternGraphs;
}

bool MatmulAllToAllTransposeA5FusionPass::MeetRequirements(const std::unique_ptr<ge::fusion::MatchResult> &matchResult)
{
    OPS_LOG_I(kPassName.c_str(), "Enter MeetRequirements for MatmulAllToAllTransposeA5FusionPass");

    // 是否950
    if (!IsTargetPlatformNpuArch(kPassName.c_str(), NPUARCH_A5)) {
        OPS_LOG_D(kPassName.c_str(), "Check target platform fail!");
        return false;
    }

    // 是否mx quant
    ge::fusion::NodeIo mc2NodeIo;
    OP_LOGE_IF(matchResult->GetCapturedTensor(kMatmulAlltoAllCaptureIdx, mc2NodeIo) != ge::SUCCESS, false,
               kPassName.c_str(), "Get MatmulAlltoAll node failed.");
    auto mc2Node = mc2NodeIo.node;
    if (!IsMXQuantMode(mc2Node)) {
        OPS_LOG_D(kPassName.c_str(), "Quant mode is not MX QUANT, fusion will be skipped.");
        return false;
    }

    // 是否transpose 条件吻合
    ge::fusion::NodeIo transposeOutput;
    OP_LOGE_IF(matchResult->GetCapturedTensor(kTransposeCaptureIdx, transposeOutput) != ge::SUCCESS, false,
               kPassName.c_str(), "Get Transpose node failed.");
    auto transposeNode = transposeOutput.node;
    if (!IsTransposePermValid(transposeNode)) {
        OPS_LOG_D(kPassName.c_str(), "Transpose permutation is not valid.");
        return false;
    }

    return true;
}

ge::fusion::GraphUniqPtr
MatmulAllToAllTransposeA5FusionPass::Replacement(const std::unique_ptr<ge::fusion::MatchResult> &matchResult)
{
    OPS_LOG_D(kPassName.c_str(), "Enter Replacement for MatmulAllToAllTransposeA5FusionPass");
    std::vector<ge::fusion::SubgraphInput> subgraphInputs;
    matchResult->ToSubgraphBoundary()->GetAllInputs(subgraphInputs);

    std::vector<ge::Shape> inputShapes;
    std::vector<ge::DataType> inputDTypes;
    std::vector<ge::Format> inputFormats;
    GetInputsInfo(subgraphInputs, inputShapes, inputDTypes, inputFormats);

    auto replaceGraphBuilder = ge::es::EsGraphBuilder("replacement");
    auto rX1 = replaceGraphBuilder.CreateInput(0, "x1", inputDTypes[0], inputFormats[0], inputShapes[0].GetDims());
    auto rX2 = replaceGraphBuilder.CreateInput(1, "x2", inputDTypes[1], inputFormats[1], inputShapes[1].GetDims());
    auto rBias = replaceGraphBuilder.CreateInput(2, "bias", inputDTypes[2], inputFormats[2], inputShapes[2].GetDims());
    auto rX1Scale =
        replaceGraphBuilder.CreateInput(3, "x1_scale", inputDTypes[3], inputFormats[3], inputShapes[3].GetDims());
    auto rX2Scale =
        replaceGraphBuilder.CreateInput(4, "x2_scale", inputDTypes[4], inputFormats[4], inputShapes[4].GetDims());
    auto rCommScale =
        replaceGraphBuilder.CreateInput(5, "comm_scale", inputDTypes[5], inputFormats[5], inputShapes[5].GetDims());
    auto rX1Offset =
        replaceGraphBuilder.CreateInput(6, "x1_offset", inputDTypes[6], inputFormats[6], inputShapes[6].GetDims());
    auto rX2Offset =
        replaceGraphBuilder.CreateInput(7, "x2_offset", inputDTypes[7], inputFormats[7], inputShapes[7].GetDims());

    ge::fusion::NodeIo mc2NodeIo;
    OP_LOGE_IF(matchResult->GetCapturedTensor(kMatmulAlltoAllCaptureIdx, mc2NodeIo) != ge::SUCCESS, nullptr,
               kPassName.c_str(), "Get MatmulAlltoAll node failed in Replacement.");
    auto mc2Node = mc2NodeIo.node; // GNode

    ge::AscendString group = "";
    mc2Node.GetAttr("group", group);

    int64_t worldSize = -1;
    mc2Node.GetAttr("world_size", worldSize);

    std::vector<int64_t> all2allAxes = {-1, -2};
    mc2Node.GetAttr("all2all_axes", all2allAxes);

    int64_t yDType = 28;
    mc2Node.GetAttr("y_dtype", yDType);

    int64_t x1QuantMode = 0;
    mc2Node.GetAttr("x1_quant_mode", x1QuantMode);

    int64_t x2QuantMode = 0;
    mc2Node.GetAttr("x2_quant_mode", x2QuantMode);

    int64_t commQuantMode = 0;
    mc2Node.GetAttr("comm_quant_mode", commQuantMode);

    int64_t commQuantDType = 28;
    mc2Node.GetAttr("comm_quant_dtype", commQuantDType);

    bool transposeX1 = false;
    mc2Node.GetAttr("transpose_x1", transposeX1);

    bool transposeX2 = false;
    mc2Node.GetAttr("transpose_x2", transposeX2);
    transposeX2 = !transposeX2;

    int64_t groupSize = 0;
    mc2Node.GetAttr("group_size", groupSize);

    // 更新x2 shape
    ge::fusion::NodeIo transposeNodeIo;
    OP_LOGE_IF(matchResult->GetCapturedTensor(kMatmulAlltoAllCaptureIdx, transposeNodeIo) != ge::SUCCESS, nullptr,
               kPassName.c_str(), "Get Transpose node failed in Replacement.");
    auto mc2 = ge::es::MatmulAlltoAll(rX1, rX2, rBias, rX1Scale, rX2Scale, rCommScale, rX1Offset, rX2Offset,
                                      group.GetString(), worldSize, all2allAxes, yDType, x1QuantMode, x2QuantMode,
                                      commQuantMode, commQuantDType, transposeX1, transposeX2, groupSize);

    ge::fusion::GraphUniqPtr replaceGraph = replaceGraphBuilder.BuildAndReset({mc2});
    // infershape
    if (!InferShapeReplaceGraph(replaceGraph, subgraphInputs)) {
        OPS_LOG_E(kPassName.c_str(), "InferShapeReplaceGraph failed in Replacement.");
        return nullptr;
    }
    return std::move(replaceGraph);
}

REG_FUSION_PASS(MatmulAllToAllTransposeA5FusionPass).Stage(ge::CustomPassStage::kAfterInferShape); // 待确认stage
} // namespace ops
