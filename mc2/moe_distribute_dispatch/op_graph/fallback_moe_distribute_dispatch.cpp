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

#ifdef __cplusplus
extern "C" {
#endif

namespace fallback {
using namespace ge;
using namespace gert;

const char *MoeDistributeDispatchInfo = "MoeDistributeDispatchFallback";

namespace {
struct OpInput {
  const gert::Tensor *x;
  const gert::Tensor *expand_ids;
};

struct OpOptionalInput {
  const gert::Tensor *scales;
  const gert::Tensor *x_active_mask;
  const gert::Tensor *expert_scales;
};

struct OpOutput {
  const gert::Tensor *expand_x;
  const gert::Tensor *dynamic_scales;
  const gert::Tensor *expand_idx;
  const gert::Tensor *expert_token_nums;
  const gert::Tensor *ep_recv_count;
  const gert::Tensor *tp_recv_count;
  const gert::Tensor *expand_scales;
};

struct OpAttrs {
  const char *group_ep;
  const int64_t *ep_world_size;
  const int64_t *ep_rank_id;
  const int64_t *moe_expert_num;
  const char *group_tp;
  const int64_t *tp_world_size;
  const int64_t *tp_rank_id;
  const int64_t *expert_shard_type;
  const int64_t *shared_expert_num;
  const int64_t *shared_expert_rank_num;
  const int64_t *quant_mode_ptr;
  const int64_t *global_bs_ptr;
  const int64_t *expert_token_nums_type_ptr;
};

// 获取必选输入
graphStatus MoeDistributeDispatchGetOpInput(OpExecuteContext* host_api_ctx, OpInput &opInput)
{
  opInput.x = host_api_ctx->GetInputTensor(static_cast<size_t>(0));
  OP_CHECK_IF(opInput.x == nullptr, OP_LOGE(MoeDistributeDispatchInfo, "x is null"), return ge::GRAPH_FAILED);

  opInput.expand_ids = host_api_ctx->GetInputTensor(static_cast<size_t>(1));
  OP_CHECK_IF(opInput.expand_ids == nullptr, OP_LOGE(MoeDistributeDispatchInfo, "expand_ids is null"), return ge::GRAPH_FAILED);
  return ge::GRAPH_SUCCESS;
}

// 获取可选输入
void MoeDistributeDispatchGetOpOptionalInput(OpExecuteContext* host_api_ctx, OpOptionalInput &opOptionalInput)
{
  opOptionalInput.scales = host_api_ctx->GetOptionalInputTensor(static_cast<size_t>(2));
  opOptionalInput.x_active_mask = host_api_ctx->GetOptionalInputTensor(static_cast<size_t>(3));
  opOptionalInput.expert_scales = host_api_ctx->GetOptionalInputTensor(static_cast<size_t>(4));
}

// 获取输出
graphStatus MoeDistributeDispatchGetOpOutput(OpExecuteContext* host_api_ctx, OpOutput &opOutput)
{
  opOutput.expand_x = host_api_ctx->GetOutputTensor(static_cast<size_t>(0));
  OP_CHECK_IF(opOutput.expand_x == nullptr, OP_LOGE(MoeDistributeDispatchInfo, "expand_x is null"), return ge::GRAPH_FAILED);

  opOutput.dynamic_scales = host_api_ctx->GetOutputTensor(static_cast<size_t>(1));
  OP_CHECK_IF(opOutput.dynamic_scales == nullptr, OP_LOGE(MoeDistributeDispatchInfo, "dynamic_scales is null"), return ge::GRAPH_FAILED);

  opOutput.expand_idx = host_api_ctx->GetOutputTensor(static_cast<size_t>(2));
  OP_CHECK_IF(opOutput.expand_idx == nullptr, OP_LOGE(MoeDistributeDispatchInfo, "expand_idx is null"), return ge::GRAPH_FAILED);

  opOutput.expert_token_nums = host_api_ctx->GetOutputTensor(static_cast<size_t>(3));
  OP_CHECK_IF(opOutput.expert_token_nums == nullptr, OP_LOGE(MoeDistributeDispatchInfo, "expert_token_nums is null"), return ge::GRAPH_FAILED);

  opOutput.ep_recv_count = host_api_ctx->GetOutputTensor(static_cast<size_t>(4));
  OP_CHECK_IF(opOutput.ep_recv_count == nullptr, OP_LOGE(MoeDistributeDispatchInfo, "ep_recv_count is null"), return ge::GRAPH_FAILED);

  opOutput.tp_recv_count = host_api_ctx->GetOutputTensor(static_cast<size_t>(5));
  OP_CHECK_IF(opOutput.tp_recv_count == nullptr, OP_LOGE(MoeDistributeDispatchInfo, "tp_recv_count is null"), return ge::GRAPH_FAILED);

  opOutput.expand_scales = host_api_ctx->GetOutputTensor(static_cast<size_t>(6));
  OP_CHECK_IF(opOutput.expand_scales == nullptr, OP_LOGE(MoeDistributeDispatchInfo, "expand_scales is null"), return ge::GRAPH_FAILED);
  return ge::GRAPH_SUCCESS;
}

// 获取属性
graphStatus MoeDistributeDispatchGetOpAttrs(OpExecuteContext* host_api_ctx, OpAttrs &opAttrs)
{
  const auto attrs = host_api_ctx->GetAttrs();
  OP_CHECK_IF(attrs == nullptr, OP_LOGE(MoeDistributeDispatchInfo, "attrs is null"), return ge::GRAPH_FAILED);

  opAttrs.group_ep = attrs->GetStr(static_cast<size_t>(0));
  OP_CHECK_IF(opAttrs.group_ep == nullptr, OP_LOGE(MoeDistributeDispatchInfo, "group_ep is null"), return ge::GRAPH_FAILED);

  opAttrs.ep_world_size = attrs->GetInt(static_cast<size_t>(1));
  OP_CHECK_IF(opAttrs.ep_world_size == nullptr, OP_LOGE(MoeDistributeDispatchInfo, "ep_world_size is null"), return ge::GRAPH_FAILED);

  opAttrs.ep_rank_id = attrs->GetInt(static_cast<size_t>(2));
  OP_CHECK_IF(opAttrs.ep_rank_id == nullptr, OP_LOGE(MoeDistributeDispatchInfo, "ep_rank_id is null"), return ge::GRAPH_FAILED);

  opAttrs.moe_expert_num = attrs->GetInt(static_cast<size_t>(3));
  OP_CHECK_IF(opAttrs.moe_expert_num == nullptr, OP_LOGE(MoeDistributeDispatchInfo, "moe_expert_num is null"), return ge::GRAPH_FAILED);

  opAttrs.group_tp = attrs->GetStr(static_cast<size_t>(4));
  OP_CHECK_IF(opAttrs.group_tp == nullptr, OP_LOGE(MoeDistributeDispatchInfo, "group_tp is null"), return ge::GRAPH_FAILED);

  opAttrs.tp_world_size = attrs->GetInt(static_cast<size_t>(5));
  OP_CHECK_IF(opAttrs.tp_world_size == nullptr, OP_LOGE(MoeDistributeDispatchInfo, "tp_world_size is null"), return ge::GRAPH_FAILED);

  opAttrs.tp_rank_id = attrs->GetInt(static_cast<size_t>(6));
  OP_CHECK_IF(opAttrs.tp_rank_id == nullptr, OP_LOGE(MoeDistributeDispatchInfo, "tp_rank_id is null"), return ge::GRAPH_FAILED);

  opAttrs.expert_shard_type = attrs->GetInt(static_cast<size_t>(7));
  OP_CHECK_IF(opAttrs.expert_shard_type == nullptr, OP_LOGE(MoeDistributeDispatchInfo, "expert_shard_type is null"), return ge::GRAPH_FAILED);

  opAttrs.shared_expert_num = attrs->GetInt(static_cast<size_t>(8));
  OP_CHECK_IF(opAttrs.shared_expert_num == nullptr, OP_LOGE(MoeDistributeDispatchInfo, "shared_expert_num is null"), return ge::GRAPH_FAILED);

  opAttrs.shared_expert_rank_num = attrs->GetInt(static_cast<size_t>(9));
  OP_CHECK_IF(opAttrs.shared_expert_rank_num == nullptr, OP_LOGE(MoeDistributeDispatchInfo, "shared_expert_rank_num is null"), return ge::GRAPH_FAILED);

  opAttrs.quant_mode_ptr = attrs->GetInt(static_cast<size_t>(10));
  OP_CHECK_IF(opAttrs.quant_mode_ptr == nullptr, OP_LOGE(MoeDistributeDispatchInfo, "quant_mode_ptr is null"), return ge::GRAPH_FAILED);

  opAttrs.global_bs_ptr = attrs->GetInt(static_cast<size_t>(11));
  OP_CHECK_IF(opAttrs.global_bs_ptr == nullptr, OP_LOGE(MoeDistributeDispatchInfo, "global_bs_ptr is null"), return ge::GRAPH_FAILED);

  opAttrs.expert_token_nums_type_ptr = attrs->GetInt(static_cast<size_t>(12));
  OP_CHECK_IF(opAttrs.expert_token_nums_type_ptr == nullptr, OP_LOGE(MoeDistributeDispatchInfo, "expert_token_nums_type_ptr is null"), return ge::GRAPH_FAILED);
  return ge::GRAPH_SUCCESS;
}

} // namespace

static graphStatus MoeDistributeDispatchExecuteFunc(OpExecuteContext* host_api_ctx)
{
  OP_LOGD(MoeDistributeDispatchInfo, "start to fallback for moeDistributeDispatch");
  OP_CHECK_IF(host_api_ctx == nullptr, OP_LOGE(MoeDistributeDispatchInfo, "host_api_ctx is null"), return ge::GRAPH_FAILED);

  OpInput opInput;
  OpOptionalInput opOptionalInput;
  OpOutput opOutput;
  OpAttrs opAttrs;

  OP_CHECK_IF(MoeDistributeDispatchGetOpInput(host_api_ctx, opInput) != ge::GRAPH_SUCCESS,
              OP_LOGE(MoeDistributeDispatchInfo, "get input failed"), return ge::GRAPH_FAILED);
  MoeDistributeDispatchGetOpOptionalInput(host_api_ctx, opOptionalInput);
  OP_CHECK_IF(MoeDistributeDispatchGetOpOutput(host_api_ctx, opOutput) != ge::GRAPH_SUCCESS,
              OP_LOGE(MoeDistributeDispatchInfo, "get output failed"), return ge::GRAPH_FAILED);
  OP_CHECK_IF(MoeDistributeDispatchGetOpAttrs(host_api_ctx, opAttrs) != ge::GRAPH_SUCCESS,
              OP_LOGE(MoeDistributeDispatchInfo, "get attrs failed"), return ge::GRAPH_FAILED);

  const auto api_ret = EXEC_OPAPI_CMD(aclnnMoeDistributeDispatch,
    opInput.x, opInput.expand_ids, opOptionalInput.scales, opOptionalInput.x_active_mask, opOptionalInput.expert_scales,
    opAttrs.group_ep, *opAttrs.ep_world_size, *opAttrs.ep_rank_id, *opAttrs.moe_expert_num, opAttrs.group_tp,
    *opAttrs.tp_world_size, *opAttrs.tp_rank_id, *opAttrs.expert_shard_type, *opAttrs.shared_expert_num,
    *opAttrs.shared_expert_rank_num, *opAttrs.quant_mode_ptr, *opAttrs.global_bs_ptr, *opAttrs.expert_token_nums_type_ptr,
    opOutput.expand_x, opOutput.dynamic_scales, opOutput.expand_idx, opOutput.expert_token_nums, opOutput.ep_recv_count,
    opOutput.tp_recv_count, opOutput.expand_scales);
  OP_CHECK_IF(api_ret != ge::GRAPH_SUCCESS, OP_LOGE(MoeDistributeDispatchInfo, "aclnn api error code %u", api_ret),
           return ge::GRAPH_FAILED);
  return GRAPH_SUCCESS;
}

IMPL_OP(MoeDistributeDispatch).OpExecuteFunc(MoeDistributeDispatchExecuteFunc);

}  // namespace fallback

#ifdef __cplusplus
}
#endif