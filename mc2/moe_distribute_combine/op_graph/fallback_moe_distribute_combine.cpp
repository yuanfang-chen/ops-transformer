/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "fallback/fallback_comm.h"
#include "fallback/fallback.h"
#include "mc2_log.h"

namespace fallback {
using namespace ge;
using namespace gert;

const char *MoeDistributeCombineInfo = "MoeDistributeCombineFallback";

namespace {
struct OpInput {
  const gert::Tensor *expand_x;
  const gert::Tensor *expert_ids;
  const gert::Tensor *expand_idx;
  const gert::Tensor *ep_send_counts;
  const gert::Tensor *expert_scales;
};

struct OpOptionalInput {
  const gert::Tensor *tp_send_counts;
  const gert::Tensor *x_active_mask;
  const gert::Tensor *activation_scale;
  const gert::Tensor *weight_scale;
  const gert::Tensor *group_list;
  const gert::Tensor *expand_scales;
};

struct OpOutput {
  const gert::Tensor *y; 
};

struct OpAttrs {
  const char *group_ep;
  const int64_t *ep_word_size;
  const int64_t *ep_rank_id;
  const int64_t *moe_expert_num;
  const char *group_tp;
  const int64_t *tp_word_size;
  const int64_t *tp_rank_id;
  const int64_t *expert_shard_type;
  const int64_t *shared_expert_num;
  const int64_t *shared_expert_rank_num;
  const int64_t *global_bs_ptr;
  const int64_t *out_dtype_ptr;
  const int64_t *comm_quant_mode_ptr;
  const int64_t *group_list_type_ptr;
};

// 获取必选输入
graphStatus MoeDistributeCombineGetOpInput(OpExecuteContext* host_api_ctx, OpInput &opInput)
{
  opInput.expand_x = host_api_ctx->GetInputTensor(static_cast<size_t>(0));
  OP_CHECK_IF(opInput.expand_x == nullptr, OP_LOGE(MoeDistributeCombineInfo,"expand_x is null"), return ge::GRAPH_FAILED);

  opInput.expert_ids = host_api_ctx->GetInputTensor(static_cast<size_t>(1));
  OP_CHECK_IF(opInput.expert_ids == nullptr, OP_LOGE(MoeDistributeCombineInfo,"expert_ids is null"), return ge::GRAPH_FAILED);

  opInput.expand_idx = host_api_ctx->GetInputTensor(static_cast<size_t>(2));
  OP_CHECK_IF(opInput.expand_idx == nullptr, OP_LOGE(MoeDistributeCombineInfo,"expand_idx is null"), return ge::GRAPH_FAILED);

  opInput.ep_send_counts = host_api_ctx->GetInputTensor(static_cast<size_t>(3));
  OP_CHECK_IF(opInput.ep_send_counts == nullptr, OP_LOGE(MoeDistributeCombineInfo,"ep_send_counts is null"), return ge::GRAPH_FAILED);

  opInput.expert_scales = host_api_ctx->GetInputTensor(static_cast<size_t>(4));
  OP_CHECK_IF(opInput.expert_scales == nullptr, OP_LOGE(MoeDistributeCombineInfo,"expert_scales is null"), return ge::GRAPH_FAILED);
  return ge::GRAPH_SUCCESS;
}

// 获取可选输入
graphStatus MoeDistributeCombineGetOpOptionalInput(OpExecuteContext* host_api_ctx, OpOptionalInput &opOptionalInput)
{
  opOptionalInput.tp_send_counts = host_api_ctx->GetOptionalInputTensor(static_cast<size_t>(5));
  OP_CHECK_IF(opOptionalInput.tp_send_counts == nullptr, OP_LOGE(MoeDistributeCombineInfo,"tp_send_counts is null"), return ge::GRAPH_FAILED);

  opOptionalInput.x_active_mask = host_api_ctx->GetOptionalInputTensor(static_cast<size_t>(6));
  opOptionalInput.activation_scale = host_api_ctx->GetOptionalInputTensor(static_cast<size_t>(7));
  opOptionalInput.weight_scale = host_api_ctx->GetOptionalInputTensor(static_cast<size_t>(8));
  opOptionalInput.group_list = host_api_ctx->GetOptionalInputTensor(static_cast<size_t>(9));
  opOptionalInput.expand_scales = host_api_ctx->GetOptionalInputTensor(static_cast<size_t>(10));
  return ge::GRAPH_SUCCESS;
}

// 获取输出
graphStatus MoeDistributeCombineGetOpOutput(OpExecuteContext* host_api_ctx, OpOutput &opOutput)
{
  opOutput.y = host_api_ctx->GetOutputTensor(static_cast<size_t>(0));
  OP_CHECK_IF(opOutput.y == nullptr, OP_LOGE(MoeDistributeCombineInfo,"y is null"), return ge::GRAPH_FAILED);
  return ge::GRAPH_SUCCESS;
}

// 获取属性
graphStatus MoeDistributeCombineGetOpAttrs(OpExecuteContext* host_api_ctx, OpAttrs &opAttrs)
{
  const auto attrs = host_api_ctx->GetAttrs();
  OP_CHECK_IF(attrs == nullptr, OP_LOGE(MoeDistributeCombineInfo,"attrs is null"), return ge::GRAPH_FAILED);

  opAttrs.group_ep = attrs->GetStr(static_cast<size_t>(0));
  OP_CHECK_IF(opAttrs.group_ep == nullptr, OP_LOGE(MoeDistributeCombineInfo,"group_ep is null"), return ge::GRAPH_FAILED);

  opAttrs.ep_word_size = attrs->GetInt(static_cast<size_t>(1));
  OP_CHECK_IF(opAttrs.ep_word_size == nullptr, OP_LOGE(MoeDistributeCombineInfo,"ep_word_size is null"), return ge::GRAPH_FAILED);

  opAttrs.ep_rank_id = attrs->GetInt(static_cast<size_t>(2));
  OP_CHECK_IF(opAttrs.ep_rank_id == nullptr, OP_LOGE(MoeDistributeCombineInfo,"ep_rank_id is null"), return ge::GRAPH_FAILED);

  opAttrs.moe_expert_num = attrs->GetInt(static_cast<size_t>(3));
  OP_CHECK_IF(opAttrs.moe_expert_num == nullptr, OP_LOGE(MoeDistributeCombineInfo,"moe_expert_num is null"), return ge::GRAPH_FAILED);

  opAttrs.group_tp = attrs->GetStr(static_cast<size_t>(4));
  OP_CHECK_IF(opAttrs.group_tp == nullptr, OP_LOGE(MoeDistributeCombineInfo,"group_tp is null"), return ge::GRAPH_FAILED);

  opAttrs.tp_word_size = attrs->GetInt(static_cast<size_t>(5));
  OP_CHECK_IF(opAttrs.tp_word_size == nullptr, OP_LOGE(MoeDistributeCombineInfo,"tp_word_size is null"), return ge::GRAPH_FAILED);

  opAttrs.tp_rank_id = attrs->GetInt(static_cast<size_t>(6));
  OP_CHECK_IF(opAttrs.tp_rank_id == nullptr, OP_LOGE(MoeDistributeCombineInfo,"tp_rank_id is null"), return ge::GRAPH_FAILED);

  opAttrs.expert_shard_type = attrs->GetInt(static_cast<size_t>(7));
  OP_CHECK_IF(opAttrs.expert_shard_type == nullptr, OP_LOGE(MoeDistributeCombineInfo,"expert_shard_type is null"), return ge::GRAPH_FAILED);

  opAttrs.shared_expert_num = attrs->GetInt(static_cast<size_t>(8));
  OP_CHECK_IF(opAttrs.shared_expert_num == nullptr, OP_LOGE(MoeDistributeCombineInfo,"shared_expert_num is null"), return ge::GRAPH_FAILED);

  opAttrs.shared_expert_rank_num = attrs->GetInt(static_cast<size_t>(9));
  OP_CHECK_IF(opAttrs.shared_expert_rank_num == nullptr, OP_LOGE(MoeDistributeCombineInfo,"shared_expert_rank_num is null"), return ge::GRAPH_FAILED);

  opAttrs.global_bs_ptr = attrs->GetInt(static_cast<size_t>(10));
  OP_CHECK_IF(opAttrs.global_bs_ptr == nullptr, OP_LOGE(MoeDistributeCombineInfo,"global_bs_ptr is null"), return ge::GRAPH_FAILED);

  opAttrs.out_dtype_ptr = attrs->GetInt(static_cast<size_t>(11));
  OP_CHECK_IF(opAttrs.out_dtype_ptr == nullptr, OP_LOGE(MoeDistributeCombineInfo,"out_dtype is null"), return ge::GRAPH_FAILED);

  opAttrs.comm_quant_mode_ptr = attrs->GetInt(static_cast<size_t>(12));
  OP_CHECK_IF(opAttrs.comm_quant_mode_ptr == nullptr, OP_LOGE(MoeDistributeCombineInfo,"comm_quant_mode is null"), return ge::GRAPH_FAILED);

  opAttrs.group_list_type_ptr = attrs->GetInt(static_cast<size_t>(13));
  OP_CHECK_IF(opAttrs.group_list_type_ptr == nullptr, OP_LOGE(MoeDistributeCombineInfo,"group_list_type is null"), return ge::GRAPH_FAILED);
  return ge::GRAPH_SUCCESS;
}

} // namespace

static graphStatus MoeDistributeCombineExecuteFunc(OpExecuteContext* host_api_ctx)
{
  OP_LOGD(MoeDistributeCombineInfo, "start to fallback for moeDistributeCombine");
  OP_CHECK_IF(host_api_ctx == nullptr, OP_LOGE(MoeDistributeCombineInfo,"host_api_ctx is null"), return ge::GRAPH_FAILED);

  OpInput opInput;
  OpOptionalInput opOptionalInput;
  OpOutput opOutput;
  OpAttrs opAttrs;

  OP_CHECK_IF(MoeDistributeCombineGetOpInput(host_api_ctx, opInput) != ge::GRAPH_SUCCESS,
              OP_LOGE(MoeDistributeCombineInfo,"get input failed"), return ge::GRAPH_FAILED);
  OP_CHECK_IF(MoeDistributeCombineGetOpOptionalInput(host_api_ctx, opOptionalInput) != ge::GRAPH_SUCCESS,
              OP_LOGE(MoeDistributeCombineInfo,"get optional input failed"), return ge::GRAPH_FAILED);
  OP_CHECK_IF(MoeDistributeCombineGetOpOutput(host_api_ctx, opOutput) != ge::GRAPH_SUCCESS,
              OP_LOGE(MoeDistributeCombineInfo,"get output failed"), return ge::GRAPH_FAILED);
  OP_CHECK_IF(MoeDistributeCombineGetOpAttrs(host_api_ctx, opAttrs) != ge::GRAPH_SUCCESS,
              OP_LOGE(MoeDistributeCombineInfo,"get attrs failed"), return ge::GRAPH_FAILED);

  const auto api_ret = EXEC_OPAPI_CMD(aclnnMoeDistributeCombine,
    opInput.expand_x, opInput.expert_ids, opInput.expand_idx, opInput.ep_send_counts,
    opInput.expert_scales, opOptionalInput.tp_send_counts, opOptionalInput.x_active_mask,
    opOptionalInput.activation_scale, opOptionalInput.weight_scale, opOptionalInput.group_list,
    opOptionalInput.expand_scales, opAttrs.group_ep, *opAttrs.ep_word_size, *opAttrs.ep_rank_id,
    *opAttrs.moe_expert_num, opAttrs.group_tp, *opAttrs.tp_word_size, *opAttrs.tp_rank_id,
    *opAttrs.expert_shard_type, *opAttrs.shared_expert_num, *opAttrs.shared_expert_rank_num,
    *opAttrs.global_bs_ptr, *opAttrs.out_dtype_ptr, *opAttrs.comm_quant_mode_ptr,
    *opAttrs.group_list_type_ptr, opOutput.y);
  OP_CHECK_IF(api_ret != ge::GRAPH_SUCCESS, OP_LOGE(MoeDistributeCombineInfo,"aclnn api error code %u", api_ret), return api_ret);
  return GRAPH_SUCCESS;
}

IMPL_OP(MoeDistributeCombine).OpExecuteFunc(MoeDistributeCombineExecuteFunc);

}  // namespace fallback