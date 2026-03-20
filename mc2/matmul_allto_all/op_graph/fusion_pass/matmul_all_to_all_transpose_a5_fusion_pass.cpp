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
#include "es_MatmulAlltoAll.h"
#include "es_math_ops.h"       // math 仓用到的打桩原型
#include "mc2_platform_info.h"

// using namespace ge;
// using namespace fe;
// using namespace fusion;
// 待改, 编码规范不允许使用using namespace

namespace ge {
namespace ops {
namespace matmulalltoalltransposeA5 {
const std::string kPassName = "MatmulAllToAllTransposeA5FusionPass";
const std::string kPatternMC2 = "MatmulAlltoAll";
const std::string kPatternTranspose = "Transpose";
const std::string kPatternTransposeD = "TransposeD";
const int64_t kMatmulAlltoAllCaptureIdx = 0l;
const int64_t kTransposeCaptureIdx = 1l;
const int64_t kMXQuantMode = 6;
const int64_t kTransposePermIdx = 1l;
const int64_t kX2InTransposeNodeIdx = 0l;


static fusion::PatternUniqPtr MakePatternForTranspose()
{
    auto graphBuilder = es::EsGraphBuilder((kPassName + kPatternTranspose).c_str());
    auto [x1, x2, bias, x1Scale, x2Scale] = graphBuilder.CreateInputs<5>();
    auto transpose = es::Transpose(x2, {1, 0});
    auto mc2 = es::MatmulAlltoAll(x1, transpose, bias, x1Scale, x2Scale);
    auto graph = graphBuilder.BuildAndReset({mc2});
    auto pattern = std::make_unique<fusion::Pattern>(std::move(*graph));
    pattern->CaptureTensor({*mc2.GetProducer(), 0}).CaptureTensor({*transpose.GetProducer(), 0});
    return pattern;
}

static fusion::PatternUniqPtr MakePatternForTransposeD()
{
    auto graphBuilder = es::EsGraphBuilder((kPassName + kPatternTransposeD).c_str());
    auto [x1, x2, x1Scale, x2Scale, bias] = graphBuilder.CreateInputs<5>();
    auto transposeD = es::TransposeD(x2, {1, 0});
    auto mc2 = es::MatmulAlltoAll(x1, transposeD, x1Scale, x2Scale, bias);
    auto graph = graphBuilder.BuildAndReset({mc2});
    auto pattern = std::make_unique<fusion::Pattern>(std::move(*graph));
    pattern->CaptureTensor({*mc2.GetProducer(), 0}).CaptureTensor({*transposeD.GetProducer(), 0});
    return pattern;
}

static bool IsMXQuantMode(const GNode &mc2Node)
{
    int64_t x1QuantMode = 0;
    int64_t x2QuantMode = 0;
    mc2Node.GetAttr("x1_quant_mode", x1QuantMode);
    mc2Node.GetAttr("x2_quant_mode", x2QuantMode);
    return (x1QuantMode == kMXQuantMode && x2QuantMode == kMXQuantMode);
}

static bool GetTransposePerm(const GNode &transposeNode, std::vector<int64_t> &permValue)
{
    AscendString nodeType("");
    transposeNode.GetType(nodeType);
    std::string nodeTypeStr = nodeType.GetString();

    if (nodeTypeStr == kPatternTransposeD) { // return when node is TransposeD
        if (transposeNode.GetAttr("perm", permValue) != GRAPH_SUCCESS) {
            OPS_LOG_W(kPassName.c_str(), "Get perm Attr from TransposeD node failed.");
            return false;
        }
        return !permValue.empty();
    }

    // Transpose 的场景
    TensorDesc permDesc;
    transposeNode.GetInputDesc(kTransposePermIdx, permDesc);
    Shape permShape = permDesc.GetShape();
    size_t permDimNum = permShape.GetDimNum(); // 把GetDim后取size 替换成了GetDimNum

    if (permShape.GetDimNum() != 1) {
        OPS_LOG_D(kPassName.c_str(), "Transpose permutation dim size must be 1, but got %zu.", permShape.GetDimNum());
        return false;
    }

    Tensor permTensor;
    if (transposeNode.GetInputConstData(kTransposePermIdx, permTensor) != GRAPH_SUCCESS) {
        OPS_LOG_D(kPassName.c_str(), "Failed to get transpose permutation const data.");
        return false;
    }

    DataType permDType = permDesc.GetDataType();
    uint8_t *permData = permTensor.GetData();

    if (permData == nullptr) {
        OPS_LOG_D(kPassName.c_str(), "Transpose permutation data is nullptr.");
        return false;
    }

    size_t size = 0;
    if (permDType == DT_INT32) {
        size = permTensor.GetSize() / sizeof(int32_t);
        for (size_t i = 0; i < size; ++i) {
            permValue.emplace_back(static_cast<int64_t>(*(reinterpret_cast<int32_t *>(permData) + i)));
        }
    } else if (permDType == DT_INT64) {
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

static bool IsTransposePermValid(const GNode &transposeNode)
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

static bool InferShapeReplaceGraph(const fusion::GraphUniqPtr &replaceGraph,
                                   const std::vector<fusion::SubgraphInput> &subgraphInputs)
{
    OPS_LOG_D(kPassName.c_str(), "Enter InferShapeReplaceGraph for MatmulAllToAllTransposeA5FusionPass");
    std::vector<Shape> inputShapes;
    for (const auto& subgraphInput : subgraphInputs) {
        auto matchNode = subgraphInput.GetAllInputs().at(0);
        TensorDesc tmpDesc;
        matchNode.node.GetInputDesc(matchNode.index, tmpDesc);
        inputShapes.emplace_back(tmpDesc.GetShape());
    }
    return (GeUtils::InferShape(*replaceGraph, inputShapes) == SUCCESS);
}

} // namespace matmulalltoalltransposeA5

std::vector<fusion::PatternUniqPtr> MatmulAllToAllTransposeA5FusionPass::Patterns()
{
    OPS_LOG_I(kPassName.c_str(), "Enter Patterns for MatmulAllToAllTransposeA5FusionPass");
    std::vector<PatternUniqPtr> patternGraphs;
    patternGraphs.emplace_back(MakePatternForTranspose());
    patternGraphs.emplace_back(MakePatternForTransposeD());
    return patternGraphs;
}

bool MatmulAllToAllTransposeA5FusionPass::MeetRequirements(const std::unique_ptr<fusion::MatchResult> &matchResult)
{
    OPS_LOG_I(kPassName.c_str(), "Enter MeetRequirements for MatmulAllToAllTransposeA5FusionPass");

    // 是否950
    if (!IsTargetPlatformNpuArch(kPassName.c_str(), NPUARCH_A5)) {
        OPS_LOG_D(kPassName.c_str(), "Check target platform fail!");
        return false;
    }

    // 是否mx quant
    fusion::NodeIo mc2NodeIo;
    OP_LOGE_IF(matchResult->GetCapturedTensor(kMatmulAlltoAllCaptureIdx, mc2NodeIo) != SUCCESS, false,
               kPassName.c_str(), "Get MatmulAlltoAll node failed.");
    auto mc2Node = mc2NodeIo.node;
    if (!IsMXQuantMode(mc2Node)) {
        OPS_LOG_D(kPassName.c_str(), "Quant mode is not MX QUANT, fusion will be skipped.");
        return false;
    }

    // 是否transpose 条件吻合
    fusion::NodeIo transposeOutput;
    OP_LOGE_IF(matchResult->GetCapturedTensor(kTransposeCaptureIdx, transposeOutput) != SUCCESS, false,
               kPassName.c_str(), "Get Transpose node failed.");
    auto transposeNode = transposeOutput.node;
    if (!IsTransposePermValid(transposeNode)) {
        OPS_LOG_D(kPassName.c_str(), "Transpose permutation is not valid.");
        return false;
    }

    return true;
}

fusion::GraphUniqPtr
MatmulAllToAllTransposeA5FusionPass::Replacement(const std::unique_ptr<fusion::MatchResult> &matchResult)
{
    OPS_LOG_D(kPassName.c_str(), "Enter Replacement for MatmulAllToAllTransposeA5FusionPass");
    std::vector<fusion::SubgraphInput> subgraphInputs;
    matchResult->ToSubgraphBoundary()->GetAllInputs(subgraphInputs);

    std::vector<Shape> inputShapes;
    std::vector<DataType> inputDTypes;
    std::vector<Format> inputFormats;
    GetInputsInfo(subgraphInputs, inputShapes, inputDTypes, inputFormats);

    auto replaceGraphBuilder = es::EsGraphBuilder("replacement");
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

    fusion::NodeIo mc2NodeIo;
    OP_LOGE_IF(matchResult->GetCapturedTensor(kMatmulAlltoAllCaptureIdx, mc2NodeIo) != SUCCESS, nullptr,
               kPassName.c_str(), "Get MatmulAlltoAll node failed in Replacement.");
    auto mc2Node = mc2NodeIo.node; // GNode

    std::string group = "";
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
    fusion::NodeIo transposeNodeIo;
    OP_LOGE_IF(matchResult->GetCapturedTensor(kMatmulAlltoAllCaptureIdx, transposeNodeIo) != SUCCESS, nullptr,
               kPassName.c_str(), "Get Transpose node failed in Replacement.");
    auto transposeNode = transposeNodeIo.node;
    if (!UpdateInputShape(mc2Node, transposeNode)) {
        OPS_LOG_E(kPassName.c_str(), "Update Input Shape for MatmulAllToAllTransposeA5FusionPass failed.");
        return nullptr;
    }
    auto mc2 = es::MatmulAlltoAll(rX1, rX2, rBias, rX1Scale, rX2Scale, rCommScale, rX1Offset, rX2Offset, group,
                                  worldSize, all2allAxes, yDType, x1QuantMode, x2QuantMode, commQuantMode,
                                  commQuantDType, transposeX1, transposeX2, groupSize);

    fusion::GraphUniqPtr replaceGraph = replaceGraphBuilder.BuildAndReset({mc2});
    // infershape
    if (!InferShapeReplaceGraph(replaceGraph, subgraphInputs)) {
        OPS_LOG_E(kPassName.c_str(), "InferShapeReplaceGraph failed in Replacement.");
        return nullptr;
    }
    return std::move(replaceGraph);
}

REG_FUSION_PASS(MatmulAllToAllTransposeA5FusionPass).Stage(CustomPassStage::kAfterInferShape); // 待确认stage
} // namespace ops
} // namespace ge
