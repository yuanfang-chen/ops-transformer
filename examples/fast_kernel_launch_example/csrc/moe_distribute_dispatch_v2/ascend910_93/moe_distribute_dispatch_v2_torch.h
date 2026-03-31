/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You can not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_distribute_dispatch_v2_torch.h
 * \brief PyTorch interface for moe_distribute_dispatch_v2
 */

#ifndef MOE_DISTRIBUTE_DISPATCH_V2_TORCH_H
#define MOE_DISTRIBUTE_DISPATCH_V2_TORCH_H

#include <torch/extension.h>
#include <tuple>

namespace ascend_ops {
namespace MoeDistributeDispatchV2 {

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
moe_distribute_dispatch_v2_npu(const at::Tensor &x, const at::Tensor &expert_ids,
                               const c10::optional<at::Tensor> &scales, const c10::optional<at::Tensor> &x_active_mask,
                               const c10::optional<at::Tensor> &expert_scales,
                               const c10::optional<at::Tensor> &elastic_info,
                               const c10::optional<at::Tensor> &performance_info, const at::Tensor &mc2_context,
                               int64_t ep_world_size, int64_t ep_rank_id, int64_t tp_world_size, int64_t tp_rank_id,
                               int64_t moe_expert_num, int64_t shared_expert_num, int64_t shared_expert_rank_num,
                               int64_t global_bs, int64_t k, int64_t quant_mode, int64_t full_mesh_mode);

}
} // namespace ascend_ops

#endif