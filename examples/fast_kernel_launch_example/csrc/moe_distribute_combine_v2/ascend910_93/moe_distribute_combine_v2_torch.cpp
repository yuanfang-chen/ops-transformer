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
 * \file moe_distribute_combine_v2_torch.cpp
 * \brief PyTorch implementation for moe_distribute_combine_v2 direct launch
 */

#include "moe_distribute_combine_v2_torch.h"
#include "moe_distribute_combine_v2_entry.h"
#include "op_kernel/moe_distribute_combine_v2_tiling.h"

#include <ATen/Operators.h>
#include <torch/library.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "platform/platform_ascendc.h"

namespace ascend_ops {
namespace MoeDistributeCombineV2 {

namespace {
constexpr uint32_t AIV_NUM = 48;
constexpr uint32_t MAX_UB_SIZE = 170U * 1024U;
constexpr uint32_t WIN_ADDR_ALIGN = 512UL;
} // namespace

TORCH_LIBRARY_FRAGMENT(EXTENSION_MODULE_NAME, m)
{
    m.def("moe_distribute_combine_v2(Tensor expand_x, Tensor expert_ids, Tensor assist_info_for_combine, "
          "Tensor ep_send_count, Tensor? tp_send_count, Tensor? scales, Tensor? x_active_mask, "
          "Tensor? shared_expert_x, Tensor? elastic_info, Tensor? ori_x, Tensor? performance_info, "
          "Tensor mc2_context, int ep_world_size, int ep_rank_id, int tp_world_size, int tp_rank_id, "
          "int moe_expert_num, int shared_expert_num, int shared_expert_rank_num, int global_bs, "
          "int k, int quant_mode) -> Tensor");
}

static void CalculateTiling(const at::Tensor &expand_x, int64_t ep_world_size, int64_t ep_rank_id,
                            int64_t tp_world_size, int64_t tp_rank_id, int64_t moe_expert_num,
                            int64_t shared_expert_num, int64_t shared_expert_rank_num, int64_t global_bs, int64_t k,
                            int64_t quant_mode, uint64_t win_size_ep, MoeDistributeCombineV2TilingData &tilingData)
{
    auto expand_x_sizes = expand_x.sizes();
    uint32_t expanded_bs = static_cast<uint32_t>(expand_x_sizes[0]);
    uint32_t h = static_cast<uint32_t>(expand_x_sizes[1]);
    uint32_t bs = expanded_bs / static_cast<uint32_t>(k);

    tilingData.moeDistributeCombineV2Info.epWorldSize = static_cast<uint32_t>(ep_world_size);
    tilingData.moeDistributeCombineV2Info.tpWorldSize = static_cast<uint32_t>(tp_world_size);
    tilingData.moeDistributeCombineV2Info.epRankId = static_cast<uint32_t>(ep_rank_id);
    tilingData.moeDistributeCombineV2Info.tpRankId = static_cast<uint32_t>(tp_rank_id);
    tilingData.moeDistributeCombineV2Info.moeExpertNum = static_cast<uint32_t>(moe_expert_num);
    tilingData.moeDistributeCombineV2Info.sharedExpertNum = static_cast<uint32_t>(shared_expert_num);
    tilingData.moeDistributeCombineV2Info.sharedExpertRankNum = static_cast<uint32_t>(shared_expert_rank_num);
    tilingData.moeDistributeCombineV2Info.moeExpertPerRankNum = static_cast<uint32_t>(moe_expert_num / ep_world_size);
    tilingData.moeDistributeCombineV2Info.zeroExpertNum = 0;
    tilingData.moeDistributeCombineV2Info.copyExpertNum = 0;
    tilingData.moeDistributeCombineV2Info.constExpertNum = 0;
    tilingData.moeDistributeCombineV2Info.globalBs = static_cast<uint32_t>(global_bs);
    tilingData.moeDistributeCombineV2Info.bs = bs;
    tilingData.moeDistributeCombineV2Info.k = static_cast<uint32_t>(k);
    tilingData.moeDistributeCombineV2Info.h = h;
    tilingData.moeDistributeCombineV2Info.a = 0;
    tilingData.moeDistributeCombineV2Info.aivNum = AIV_NUM;
    tilingData.moeDistributeCombineV2Info.isTokenMask = false;
    tilingData.moeDistributeCombineV2Info.isExpertMask = false;
    tilingData.moeDistributeCombineV2Info.hasSharedExpertX = false;
    tilingData.moeDistributeCombineV2Info.hasElasticInfo = false;
    tilingData.moeDistributeCombineV2Info.isPerformance = false;
    tilingData.moeDistributeCombineV2Info.isMc2Context = true;
    tilingData.moeDistributeCombineV2Info.totalUbSize = MAX_UB_SIZE;
    tilingData.moeDistributeCombineV2Info.totalWinSizeEp = win_size_ep;
    tilingData.moeDistributeCombineV2Info.totalWinSizeTp = 0;
    tilingData.moeDistributeCombineV2Info.armAvgFactor = 0.0f;
    tilingData.moeDistributeCombineV2Info.epsilon = 0.0f;
    tilingData.moeDistributeCombineV2Info.bufferNum = 1;
}

at::Tensor moe_distribute_combine_v2_npu(
    const at::Tensor &expand_x, const at::Tensor &expert_ids, const at::Tensor &assist_info_for_combine,
    const at::Tensor &ep_send_count, const c10::optional<at::Tensor> &tp_send_count,
    const c10::optional<at::Tensor> &scales, const c10::optional<at::Tensor> &x_active_mask,
    const c10::optional<at::Tensor> &shared_expert_x, const c10::optional<at::Tensor> &elastic_info,
    const c10::optional<at::Tensor> &ori_x, const c10::optional<at::Tensor> &performance_info,
    const at::Tensor &mc2_context, int64_t ep_world_size, int64_t ep_rank_id, int64_t tp_world_size, int64_t tp_rank_id,
    int64_t moe_expert_num, int64_t shared_expert_num, int64_t shared_expert_rank_num, int64_t global_bs, int64_t k,
    int64_t quant_mode)
{
    const c10::OptionalDeviceGuard guard(expand_x.device());

    TORCH_CHECK(expand_x.device().type() == c10::DeviceType::PrivateUse1, "expand_x must be on NPU");
    TORCH_CHECK(expert_ids.device().type() == c10::DeviceType::PrivateUse1, "expert_ids must be on NPU");
    TORCH_CHECK(mc2_context.device().type() == c10::DeviceType::PrivateUse1, "mc2_context must be on NPU");

    auto expand_x_sizes = expand_x.sizes();
    int64_t expanded_bs = expand_x_sizes[0];
    int64_t h = expand_x_sizes[1];
    int64_t bs = expanded_bs / k;

    c10::ScalarType output_dtype = at::kBFloat16;

    at::Tensor x_out = at::empty({bs, h}, at::TensorOptions().dtype(output_dtype).device(expand_x.device()));

    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint64_t ubSizePlatForm;
    ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);

    uint64_t win_size_ep = static_cast<uint64_t>(global_bs / ep_world_size) *
                           ((h * sizeof(bfloat16_t) + WIN_ADDR_ALIGN - 1) / WIN_ADDR_ALIGN * WIN_ADDR_ALIGN) *
                           moe_expert_num;

    MoeDistributeCombineV2TilingData tilingData;
    CalculateTiling(expand_x, ep_world_size, ep_rank_id, tp_world_size, tp_rank_id, moe_expert_num, shared_expert_num,
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
        moe_distribute_combine_v2_entry(tilingKey, blockDim, stream, mc2_context.data_ptr(), expand_x.data_ptr(),
                                        expert_ids.data_ptr(), assist_info_for_combine.data_ptr(),
                                        ep_send_count.data_ptr(),
                                        tp_send_count.has_value() ? tp_send_count->data_ptr() : nullptr,
                                        scales.has_value() ? scales->data_ptr() : nullptr,
                                        x_active_mask.has_value() ? x_active_mask->data_ptr() : nullptr,
                                        shared_expert_x.has_value() ? shared_expert_x->data_ptr() : nullptr,
                                        elastic_info.has_value() ? elastic_info->data_ptr() : nullptr,
                                        ori_x.has_value() ? ori_x->data_ptr() : nullptr,
                                        performance_info.has_value() ? performance_info->data_ptr() : nullptr,
                                        x_out.data_ptr(), workspace_ptr, &tilingData);

        if (workspace_ptr != nullptr) {
            aclrtFree(workspace_ptr);
            workspace_ptr = nullptr;
        }
        return 0;
    };

    at_npu::native::OpCommand::RunOpApi("MoeDistributeCombineV2", acl_call);

    return x_out;
}

TORCH_LIBRARY_IMPL(EXTENSION_MODULE_NAME, PrivateUse1, m)
{
    m.impl("moe_distribute_combine_v2", moe_distribute_combine_v2_npu);
}

} // namespace MoeDistributeCombineV2
} // namespace ascend_ops