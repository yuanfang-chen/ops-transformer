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
 * \file multi_head_attention_onnx_plugin.cpp
 * \brief
 */

#include "attention/flash_attention_score/op_graph/flash_attention_score_proto.h"
#include "nlohmann/json.hpp"
#include "onnx_common.h"

namespace domi {
using json = nlohmann::json;
namespace {
constexpr int ALIGN_NUM = 128;
constexpr int ONE_BYTE_BITS = 8;
constexpr int EXTRA_LENGTH = 32;
constexpr int ACL_UINT8 = 4;
constexpr int OUTPUT_INDEX = 3;
constexpr int ONE = 1;

static void UpdateFlashAttentionByNode(ge::Operator& op_dest, const ge::Operator& op_src) {
  std::vector<ge::AscendString> input_strings;
  if (op_src.GetAttr("input", input_strings) == ge::GRAPH_SUCCESS) {
    op_dest.DynamicInputRegister("x", input_strings.size());
  }
  std::vector<ge::AscendString> output_strings;
  if (op_src.GetAttr("output", output_strings) == ge::GRAPH_SUCCESS) {
    op_dest.DynamicOutputRegister("y", output_strings.size());
  }
  op_dest.SetAttr("name", op_src.GetName());
  op_dest.SetAttr("original_type", "com.microsoft::11::MultiHeadAttention");
}

static Status GetOriNameFromOperator(const ge::Operator& op, std::string& ori_name) {
  if (op.GetAttr("name", ori_name) != SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "get name from op failed.");
    return FAILED;
  }
  return SUCCESS;
}
static Status GetAttr(const ge::Operator& op, int& head_num, float& scale) {
  if (op.GetAttr("head_num", head_num) != SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "get head_num from op failed");
    return FAILED;
  }
  if (op.GetAttr("scale_value", scale) != SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "get scale from op failed");
    return FAILED;
  }
  return SUCCESS;
}
}

static Status ParseParamsMultiHeadAttention(const ge::Operator& op_src, ge::Operator& op_dest) {
  ge::AscendString attrs_string;
  if (op_src.GetAttr("attribute", attrs_string) == ge::GRAPH_SUCCESS) {
    json attrs = json::parse(attrs_string.GetString());
    for (json& attr : attrs["attribute"]) {
      if (attr["name"] == "num_heads") {
        int head_num = attr["i"];
        op_dest.SetAttr("head_num", head_num);
      }
      if (attr["name"] == "scale") {
        std::string scale_str = attr["f"];
        float scale = std::stof(scale_str);
        op_dest.SetAttr("scale_value", scale);
      }
    }
  }
  UpdateFlashAttentionByNode(op_dest, op_src);
  return SUCCESS;
}

static Status ParseOpToGraphMultiHeadAttention(const ge::Operator& op, ge::Graph& graph) {
  std::string ori_name;
  if (GetOriNameFromOperator(op, ori_name)) {
    return FAILED;
  }
  auto data0 = ge::op::Data((ori_name + "_data0").c_str()).set_attr_index(0);
  auto data1 = ge::op::Data((ori_name + "_data1").c_str()).set_attr_index(1);
  auto data2 = ge::op::Data((ori_name + "_data2").c_str()).set_attr_index(2);
  auto data4 = ge::op::Data((ori_name + "_data4").c_str()).set_attr_index(4);

  int head_num = 0;
  float scale = 1.0f;
  if ((GetAttr(op, head_num, scale) != SUCCESS)) {
    return FAILED;
  }
  std::string input_layout = "BSH";
  int sparse_mode = 1;
  auto attention_score = ge::op::FlashAttentionScore((ori_name + "_FlashAttentionScore").c_str())
      .set_input_query(data0).set_input_key(data1).set_input_value(data2)
      .set_input_atten_mask(data4)
      .set_attr_scale_value(scale)
      .set_attr_input_layout(input_layout)
      .set_attr_sparse_mode(sparse_mode)
      .set_attr_head_num(head_num);
  std::vector<ge::Operator> inputs{data0, data1, data2, data4};
  std::vector<std::pair<ge::Operator, std::vector<size_t>>> outputs;
  outputs.emplace_back(attention_score, std::vector<std::size_t>{OUTPUT_INDEX});
  graph.SetInputs(inputs).SetOutputs(outputs);
  return SUCCESS;
}

// register npu_flash_attention_score op info to GE
REGISTER_CUSTOM_OP("PartitionedCall")
    .FrameworkType(ONNX)
    .OriginOpType({ge::AscendString("com.microsoft::11::MultiHeadAttention"),
                   ge::AscendString("com.microsoft::12::MultiHeadAttention"),
                   ge::AscendString("com.microsoft::13::MultiHeadAttention"),
                   ge::AscendString("com.microsoft::14::MultiHeadAttention"),
                   ge::AscendString("com.microsoft::15::MultiHeadAttention"),
                   ge::AscendString("com.microsoft::16::MultiHeadAttention"),
                   ge::AscendString("com.microsoft::17::MultiHeadAttention"),
                   ge::AscendString("com.microsoft::18::MultiHeadAttention")})
    .ParseParamsByOperatorFn(ParseParamsMultiHeadAttention)
    .ParseOpToGraphFn(ParseOpToGraphMultiHeadAttention)
    .ImplyType(ImplyType::TVM);
}