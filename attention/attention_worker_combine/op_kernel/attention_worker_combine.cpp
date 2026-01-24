/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file attention_worker_combine.cpp
 * \brief
 */

#include "attention_worker_combine_split_bs.h"
#include "attention_worker_combine_split_h.h"
#include "attention_worker_combine_split_k.h"
#include "kernel_operator.h"

#define TILING_KEY_DIVIDE_BS_FP16 10000UL
#define TILING_KEY_DIVIDE_BS_BF16 10001UL
#define TILING_KEY_DIVIDE_H_FP16 10010UL
#define TILING_KEY_DIVIDE_H_BF16 10011UL
#define TILING_KEY_DIVIDE_K_FP16 10020UL
#define TILING_KEY_DIVIDE_K_BF16 10021UL

extern "C" __global__ __aicore__ void attention_worker_combine(GM_ADDR schedule_context, GM_ADDR expert_scales, 
                                                            GM_ADDR layer_id, GM_ADDR y, GM_ADDR next_layer_id,
                                                            GM_ADDR workspace, GM_ADDR tiling)
{
  TPipe pipe;
  GET_TILING_DATA(tiling_data, tiling);
  if (TILING_KEY_IS(TILING_KEY_DIVIDE_BS_FP16)) {
    KernelAttentionWorkerCombineSplitBS<half> op(&pipe, &tiling_data);
    op.Init(schedule_context, expert_scales, layer_id, y, next_layer_id);
    op.Process();
  } else if (TILING_KEY_IS(TILING_KEY_DIVIDE_BS_BF16)) {
    KernelAttentionWorkerCombineSplitBS<bfloat16_t> op(&pipe, &tiling_data);
    op.Init(schedule_context, expert_scales, layer_id, y, next_layer_id);
    op.Process();
  } else if (TILING_KEY_IS(TILING_KEY_DIVIDE_H_FP16)) {
    KernelAttentionWorkerCombineSplitH<half> op(&pipe, &tiling_data);
    op.Init(schedule_context, expert_scales, layer_id, y, next_layer_id);
    op.Process();
  } else if (TILING_KEY_IS(TILING_KEY_DIVIDE_H_BF16)) {
    KernelAttentionWorkerCombineSplitH<bfloat16_t> op(&pipe, &tiling_data);
    op.Init(schedule_context, expert_scales, layer_id, y, next_layer_id);
    op.Process();
  } else if (TILING_KEY_IS(TILING_KEY_DIVIDE_K_FP16)) {
    KernelAttentionWorkerCombineSplitK<half> op(&pipe, &tiling_data);
    op.Init(schedule_context, expert_scales, layer_id, y, next_layer_id);
    op.Process();
  } else if (TILING_KEY_IS(TILING_KEY_DIVIDE_K_BF16)) {
    KernelAttentionWorkerCombineSplitK<bfloat16_t> op(&pipe, &tiling_data);
    op.Init(schedule_context, expert_scales, layer_id, y, next_layer_id);
    op.Process();
  }
}