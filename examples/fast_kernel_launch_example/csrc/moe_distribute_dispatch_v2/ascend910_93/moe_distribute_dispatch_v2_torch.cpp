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
 * \file moe_distribute_dispatch_v2_torch.cpp
 * \brief PyTorch implementation for moe_distribute_dispatch_v2 direct launch
 */

#include "moe_distribute_dispatch_v2_torch.h"
#include "moe_distribute_dispatch_v2_entry.h"
#include "op_kernel/moe_distribute_dispatch_v2_tiling.h"

#include <ATen/Operators.h>
#include <torch/library.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "platform/platform_ascendc.h"

namespace ascend_ops {
namespace MoeDistributeDispatchV2 {

namespace {
constexpr uint32_t AIV_NUM = 48;
constexpr uint32_t MAX_UB_SIZE = 170U * 1024U;
constexpr uint32_t WIN_ADDR_ALIGN = 512UL;
constexpr uint32_t UB_ALIGN = 32U;
} // namespace

TORCH_LIBRARY_FRAGMENT(EXTENSION_MODULE_NAME, m)
{
    m.def("moe_distribute_dispatch_v2(Tensor x, Tensor expert_ids, Tensor? scales, Tensor? x_active_mask, "
          "Tensor? expert_scales, Tensor? elastic_info, Tensor? performance_info, Tensor mc2_context, "
          "int ep_world_size, int ep_rank_id, int tp_world_size, int tp_rank_id, int moe_expert_num, "
          "int shared_expert_num, int shared_expert_rank_num, int global_bs, int k, int quant_mode, "
          "int full_mesh_mode) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
}

static void CalculateTiling(const at::Tensor &x, int64_t ep_world_size, int64_t ep_rank_id, int64_t tp_world_size,
                            int64_t tp_rank_id, int64_t moe_expert_num, int64_t shared_expert_num,
                            int64_t shared_expert_rank_num, int64_t global_bs, int64_t k, int64_t quant_mode,
                            uint64_t win_size_ep, MoeDistributeDispatchV2TilingData &tilingData)
{
    auto x_sizes = x.sizes();
    uint32_t bs = static_cast<uint32_t>(x_sizes[0]);
    uint32_t h = static_cast<uint32_t>(x_sizes[1]);

    tilingData.moeDistributeDispatchV2Info.epWorldSize = static_cast<uint32_t>(ep_world_size);
    tilingData.moeDistributeDispatchV2Info.tpWorldSize = static_cast<uint32_t>(tp_world_size);
    tilingData.moeDistributeDispatchV2Info.epRankId = static_cast<uint32_t>(ep_rank_id);
    tilingData.moeDistributeDispatchV2Info.tpRankId = static_cast<uint32_t>(tp_rank_id);
    tilingData.moeDistributeDispatchV2Info.moeExpertNum = static_cast<uint32_t>(moe_expert_num);
    tilingData.moeDistributeDispatchV2Info.sharedExpertNum = static_cast<uint32_t>(shared_expert_num);
    tilingData.moeDistributeDispatchV2Info.sharedExpertRankNum = static_cast<uint32_t>(shared_expert_rank_num);
    tilingData.moeDistributeDispatchV2Info.globalBs = static_cast<uint32_t>(global_bs);
    tilingData.moeDistributeDispatchV2Info.bs = bs;
    tilingData.moeDistributeDispatchV2Info.k = static_cast<uint32_t>(k);
    tilingData.moeDistributeDispatchV2Info.h = h;
    tilingData.moeDistributeDispatchV2Info.a = 0;
    tilingData.moeDistributeDispatchV2Info.aivNum = AIV_NUM;
    tilingData.moeDistributeDispatchV2Info.quantMode = static_cast<uint32_t>(quant_mode);
    tilingData.moeDistributeDispatchV2Info.isTokenMask = false;
    tilingData.moeDistributeDispatchV2Info.isExpertMask = false;
    tilingData.moeDistributeDispatchV2Info.hasElasticInfo = false;
    tilingData.moeDistributeDispatchV2Info.isPerformance = false;
    tilingData.moeDistributeDispatchV2Info.isQuant = quant_mode > 0;
    tilingData.moeDistributeDispatchV2Info.isMc2Context = true;
    tilingData.moeDistributeDispatchV2Info.totalUbSize = MAX_UB_SIZE;
    tilingData.moeDistributeDispatchV2Info.totalWinSizeEp = win_size_ep;
    tilingData.moeDistributeDispatchV2Info.totalWinSizeTp = 0;
    tilingData.moeDistributeDispatchV2Info.expertTokenNumsType = 1;
    tilingData.moeDistributeDispatchV2Info.zeroComputeExpertNum = 0;
    tilingData.moeDistributeDispatchV2Info.scalesRow = 0;
    tilingData.moeDistributeDispatchV2Info.scalesCol = 0;
    tilingData.moeDistributeDispatchV2Info.scalesTypeSize = 0;
    tilingData.moeDistributeDispatchV2Info.scalesCount = 0;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
moe_distribute_dispatch_v2_npu(const at::Tensor &x, const at::Tensor &expert_ids,
                               const c10::optional<at::Tensor> &scales, const c10::optional<at::Tensor> &x_active_mask,
                               const c10::optional<at::Tensor> &expert_scales,
                               const c10::optional<at::Tensor> &elastic_info,
                               const c10::optional<at::Tensor> &performance_info, const at::Tensor &mc2_context,
                               int64_t ep_world_size, int64_t ep_rank_id, int64_t tp_world_size, int64_t tp_rank_id,
                               int64_t moe_expert_num, int64_t shared_expert_num, int64_t shared_expert_rank_num,
                               int64_t global_bs, int64_t k, int64_t quant_mode, int64_t full_mesh_mode)
{
    const c10::OptionalDeviceGuard guard(x.device());

    TORCH_CHECK(x.device().type() == c10::DeviceType::PrivateUse1, "x must be on NPU");
    TORCH_CHECK(expert_ids.device().type() == c10::DeviceType::PrivateUse1, "expert_ids must be on NPU");
    TORCH_CHECK(mc2_context.device().type() == c10::DeviceType::PrivateUse1, "mc2_context must be on NPU");

    auto x_sizes = x.sizes();
    int64_t bs = x_sizes[0];
    int64_t h = x_sizes[1];

    int64_t expanded_bs = global_bs * k;

    c10::ScalarType output_dtype = at::kBFloat16;
    if (quant_mode > 0) {
        output_dtype = at::kInt8;
    }

    at::Tensor expand_x_out = at::empty({expanded_bs, h}, at::TensorOptions().dtype(output_dtype).device(x.device()));
    at::Tensor dynamic_scales_out = at::empty({0}, at::TensorOptions().dtype(at::kByte).device(x.device()));
    at::Tensor assist_info_out = at::empty({expanded_bs * 3}, at::TensorOptions().dtype(at::kInt).device(x.device()));
    at::Tensor expert_token_nums_out =
        at::empty({moe_expert_num}, at::TensorOptions().dtype(at::kLong).device(x.device()));
    at::Tensor ep_send_counts_out = at::empty({ep_world_size}, at::TensorOptions().dtype(at::kInt).device(x.device()));
    at::Tensor tp_send_counts_out = at::empty({tp_world_size}, at::TensorOptions().dtype(at::kInt).device(x.device()));
    at::Tensor expand_scales_out = at::empty({0}, at::TensorOptions().dtype(at::kFloat).device(x.device()));

    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint64_t ubSizePlatForm;
    ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);

    uint64_t win_size_ep = static_cast<uint64_t>(global_bs / ep_world_size) *
                           ((h * sizeof(bfloat16_t) + WIN_ADDR_ALIGN - 1) / WIN_ADDR_ALIGN * WIN_ADDR_ALIGN) *
                           moe_expert_num;

    MoeDistributeDispatchV2TilingData tilingData;
    CalculateTiling(x, ep_world_size, ep_rank_id, tp_world_size, tp_rank_id, moe_expert_num, shared_expert_num,
                    shared_expert_rank_num, global_bs, k, quant_mode, win_size_ep, tilingData);

    uint32_t blockDim = AIV_NUM;
    auto stream = c10_npu::getCurrentNPUStream().stream(false);

    void *workspace_ptr = nullptr;
    uint64_t workspace_size = 16U * 1024U * 1024U;
    if (workspace_size > 0) {
        auto ret = aclrtMalloc(&workspace_ptr, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
        TORCH_CHECK(ret == ACL_SUCCESS, "allocate workspace failed. ERROR: %d\n", ret);
    }

    int32_t tilingKey = 0;

    auto acl_call = [&]() -> int {
        moe_distribute_dispatch_v2_entry(tilingKey, blockDim, stream, mc2_context.data_ptr(), x.data_ptr(),
                                         expert_ids.data_ptr(), scales.has_value() ? scales->data_ptr() : nullptr,
                                         x_active_mask.has_value() ? x_active_mask->data_ptr() : nullptr,
                                         expert_scales.has_value() ? expert_scales->data_ptr() : nullptr,
                                         elastic_info.has_value() ? elastic_info->data_ptr() : nullptr,
                                         performance_info.has_value() ? performance_info->data_ptr() : nullptr,
                                         expand_x_out.data_ptr(), dynamic_scales_out.data_ptr(),
                                         assist_info_out.data_ptr(), expert_token_nums_out.data_ptr(),
                                         ep_send_counts_out.data_ptr(), tp_send_counts_out.data_ptr(),
                                         expand_scales_out.data_ptr(), workspace_ptr, &tilingData);

        if (workspace_ptr != nullptr) {
            aclrtFree(workspace_ptr);
            workspace_ptr = nullptr;
        }
        return 0;
    };

    at_npu::native::OpCommand::RunOpApi("MoeDistributeDispatchV2", acl_call);

    return std::make_tuple(expand_x_out, dynamic_scales_out, assist_info_out, expert_token_nums_out, ep_send_counts_out,
                           tp_send_counts_out, expand_scales_out);
}

TORCH_LIBRARY_IMPL(EXTENSION_MODULE_NAME, PrivateUse1, m)
{
    m.impl("moe_distribute_dispatch_v2", moe_distribute_dispatch_v2_npu);
}

} // namespace MoeDistributeDispatchV2
} // namespace ascend_ops