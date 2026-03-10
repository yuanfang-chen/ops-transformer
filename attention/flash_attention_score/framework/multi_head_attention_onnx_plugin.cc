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

#include "onnx_common.h"
#include "attention/flash_attention_score/op_graph/flash_attention_score_protn

namespace domi {
using NodeProto = ge::onnx::NodeProto;
namespace {
static void UpdateFlashAttentionByNode(ge::Operator& op_dest, const NodeProto* node) {
  int input_size = node->input_size();
  int output_size = node->output_size();
  op_dest.DynamicInputRegister("x", input_size);
  op_dest.DynamicOutputRegister("y", output_size);
  op_dest.SetAttr("name", node->name());
  op_dest.SetAttr("original_type", "npu::1::MultiHeadAttention");
}
static Status GetOriNameFromOperator(const ge::Operator& op, std::string& ori_name) {
  if (op.GetAttr("name", ori_name) != SUCCESS) {
    OP_LOGE(GetOpName(op).c_str(), "get name from op failed.");
    return FAILED;
  }
  return SUCCESS;
}
static Status GetAttr(const ge::Operator& op, int& head_num, float& scale, float &mask_filter_value) {
  if (op.GetAttr("head_num", head_num) != SUCCESS) {
    OP_LOGE(GetOpName(op).c_str(), "get head_num from op failed");
    return FAILED;
  }
  if (op.GetAttr("mask_filter_value", mask_filter_value) != SUCCESS) {
    OP_LOGE(GetOpName(op).c_str(), "get mask_filter_value from op failed");
    return FAILED;
  }
  if (op.GetAttr("scale", scale) != SUCCESS) {
    OP_LOGE(GetOpName(op).c_str(), "get scale from op failed");
    return FAILED;
  }
  return SUCCESS;
}


static Status GetFinalDimsByOperator(const ge::Operator& op, int head_num, vector<int64_t>& final_dims) {
  std::vector<int64_t> dims = op.GetInputDesc(0).GetShape().GetDims();
  int64_t numels = dims[0] * dims[1] * dims[1] * head_num;
  int64_t length = (numels + ALIGN_NUM - 1) / ALIGN_NUM * ALIGN_NUM / ONE_BYTE_BITS;
  length += EXTRA_LENGTH;
  final_dims = {length};
  return SUCCESS;
}
}

static Status ParseParamsMultiHeadAttention(const Message* op_src, ge::Operator& op_dest) {
  const NodeProto* node = dynamic_cast<const NodeProto*>(op_src);
  if (node == nullptr) {
    OP_LOGE("FlashAttention", "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }
  int32_t head_num = 0;
  float scale = 1.0f;
  float mask_filter_value = 1.0f;
  for (const auto& attr : node->attribute()) {
    if (attr.name() == "head_num" && attr.type() == ge::onnx::AttributeProto::INT) {
        head_num = attr.i();
    }
    if (attr.name() == "scale" && attr.type() == ge::onnx::AttributeProto::FLOAT) {
        scale = attr.f();
    }
    // 怎么转换
    if (attr.name() == "mask_filter_value" && attr.type() == ge::onnx::AttributeProto::FLOAT) {
        mask_filter_value = attr.f();
    }
  }
  UpdateFlashAttentionByNode(op_dest, node);
  op_dest.SetAttr("head_num", head_num);
  op_dest.SetAttr("scale", scale);
  op_dest.SetAttr("mask_filter_value", mask_filter_value);
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
  string input_layout = "BSH";
  float scale = 1.0f;
  float mask_filter_value = 1.0f;
  if ((GetAttr(op, head_num, scale, mask_filter_value) != SUCCESS)) {
    return FAILED;
  }
  // create const input tensor "drop_mask" which is filled with the scalar value 1 for inferencing
  // deop_mask.size = [B, N, S, S]
  ge::Tensor saclar_one = CreateScalar(ONE, ge::DT_UINT8);
  auto const_one = ge::op::Const((ori_name + "_Const_one").c_str()).set_attr_value(saclar_one);
  vector<int64_t> final_dims;
  if (GetFinalDimsByOperator(op, head_num, final_dims) != SUCCESS) {
    return FAILED;
  }

  auto tensor_dims = Vec2Tensor(final_dims, {1}, ge::DT_INT64);
  auto const_dims = ge::op::Const((ori_name + "_Const_dims").c_str()).set_attr_value(tensor_dims);
  auto drop_mask = ge::op::Fill((ori_name + "_Fill_ones").c_str()).set_input_dims(const_dims).set_input_value(const_one);
  auto cast_drop_mask = ge::op::Cast((ori_name + "_Cast_drop_mask").c_str()).set_input_x(drop_mask)
                                                                            .set_attr_dst_type(ACL_UINT8);

  auto AttentionScore = ge::op::FlashAttentionScore((ori_name + "_FlashAttentionScore").c_str())
      .set_input_query(data0).set_input_key(data1).set_input_value(data2)
      .set_input_padding_mask(data4).set_input_atten_mask(data5)
      .set_input_drop_mask(cast_drop_mask).set_attr_scale_value(scale)
      .set_attr_head_num(head_num);
  std::vector<ge::Operator> inputs{data0, data1, data2, data4};
  std::vector<std::pair<ge::Operator, std::vector<size_t>>> outputs;
  outputs.emplace_back(AttentionScore, std::vector<std::size_t>{OUTPUT_INDEX});
  graph.SetInputs(inputs).SetOutputs(outputs);
  return SUCCESS;
}

// register npu_flash_attention_score op info to GE
REGISTER_CUSTOM_OP("PartitionedCall")
    .FrameworkType(ONNX)
    .OriginOpType({ge::AscendString("ai.onnx::11::MultiHeadAttention"),
                   ge::AscendString("ai.onnx::12::MultiHeadAttention"),
                   ge::AscendString("ai.onnx::13::MultiHeadAttention"),
                   ge::AscendString("ai.onnx::14::MultiHeadAttention"),
                   ge::AscendString("ai.onnx::15::MultiHeadAttention"),
                   ge::AscendString("ai.onnx::16::MultiHeadAttention"),
                   ge::AscendString("ai.onnx::17::MultiHeadAttention"),
                   ge::AscendString("ai.onnx::18::MultiHeadAttention")})
    .ParseParamsFn(ParseParamsMultiHeadAttention)
    .ParseOpToGraphFn(ParseOpToGraphMultiHeadAttention)
    .ImplyType(ImplyType::TVM);
}