/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_distribute_combine_v2_torch.cpp
 * \brief
 */
#include "acl/acl.h"
#include "tiling/platform/platform_ascendc.h"

#include "kernel_operator.h"

#include "moe_distribute_combine_v2_entry.h"
#include "op_kernel/moe_distribute_combine_v2_tiling.h"
#include "../../moe_distribute_dispatch_v2/ascend910_93/moe_distribute_dispatch_v2_torch.h"

#include <ATen/ATen.h>
#include <vector>
#include <torch/all.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

namespace ascend_ops {

namespace MoeDistributeCombineV2 {

TORCH_LIBRARY_FRAGMENT(EXTENSION_MODULE_NAME, m)
{
    m.def("MoeDistributeCombineV2(Tensor expand_x, Tensor expert_ids, Tensor assist_info_for_combine, Tensor ep_send_counts, " \
            "Tensor expert_scales, Tensor mc2_context, str group_ep, " \
            "int ep_world_size, int ep_rank_id, int moe_expert_num, int total_winsize_ep, *, Tensor? tp_send_counts=None, Tensor? x_active_mask=None, " \
            "Tensor? shared_expert_x=None, Tensor? ori_x=None, " \
            "Tensor? const_expert_alpha_1=None, Tensor? const_expert_alpha_2=None, Tensor? const_expert_v=None, " \
            "Tensor? performance_info=None, str group_tp=\"\", int tp_world_size=0, int tp_rank_id=0, " \
            "int expert_shard_type=0, int shared_expert_num=0, int shared_expert_rank_num=0, " \
            "int global_bs=0, int comm_quant_mode=0, " \
            "str comm_alg=\"\", int zero_expert_num=0, int copy_expert_num=0, int const_expert_num=0) " \
            "-> Tensor");
}

constexpr uint32_t DIM_ONE = 1UL;
constexpr uint32_t DIM_TWO = 2UL;
constexpr uint32_t TILINGKEY_XTYPE = 10;
constexpr uint32_t WORKSPACESIZE = 16 * 1024 *1024;

static void CalTilingKey(int32_t &tilingKey, at::ScalarType xType, const uint32_t quantMode)
{
    tilingKey += static_cast<uint64_t>(quantMode);
    if (xType == at::kBFloat16) {
        tilingKey += static_cast<uint64_t>(TILINGKEY_XTYPE);
    }
    return;
}

void MoeDistributeCombineV2_api(aclrtStream stream,
                                const at::Tensor &expand_x, const at::Tensor &expert_ids,
                                const at::Tensor &assist_info_for_combine,
                                const at::Tensor &ep_send_counts, const at::Tensor &expert_scales,
                                const at::Tensor &new_workspace, const at::Tensor &mc2_context,
                                const c10::optional<at::Tensor> &tp_send_counts,
                                const c10::optional<at::Tensor> &x_active_mask,
                                const c10::optional<at::Tensor> &shared_expert_x,
                                const c10::optional<at::Tensor> &ori_x,
                                const c10::optional<at::Tensor> &const_expert_alpha_1,
                                const c10::optional<at::Tensor> &const_expert_alpha_2,
                                const c10::optional<at::Tensor> &const_expert_v,
                                const c10::optional<at::Tensor> &performance_info,
                                at::Tensor &x_out, int64_t comm_quant_mode,
                                MoeDistributeCombineV2Info tilingData)
{
    auto expandX_ptr = get_first_tensor_address<at::Tensor>(expand_x.scalar_type(), expand_x, false);
    auto expertIds_ptr = get_first_tensor_address<at::Tensor>(expert_ids.scalar_type(), expert_ids, false);
    auto expandIdx_ptr = get_first_tensor_address<at::Tensor>(assist_info_for_combine.scalar_type(), assist_info_for_combine, false);
    auto epSendCount_ptr = get_first_tensor_address<at::Tensor>(ep_send_counts.scalar_type(), ep_send_counts, false);
    auto expertScales_ptr = get_first_tensor_address<at::Tensor>(expert_scales.scalar_type(), expert_scales, false);
    auto workspace_ptr = get_first_tensor_address<at::Tensor>(new_workspace.scalar_type(), new_workspace, false);
    auto mc2Context_ptr = get_first_tensor_address<at::Tensor>(mc2_context.scalar_type(), mc2_context, false);

    void* tpSendCounts_ptr = nullptr;
    if(tp_send_counts.has_value()) {
        tpSendCounts_ptr = get_first_tensor_address<c10::optional<at::Tensor>>(tp_send_counts->scalar_type(), tp_send_counts, false);
    }

    void* xActiveMask_ptr = nullptr;
    if(x_active_mask.has_value()) {
        xActiveMask_ptr = get_first_tensor_address<c10::optional<at::Tensor>>(x_active_mask->scalar_type(), x_active_mask, false);
    }

    void* sharedExpertX_ptr = nullptr;
    if(shared_expert_x.has_value()) {
        sharedExpertX_ptr = get_first_tensor_address<c10::optional<at::Tensor>>(shared_expert_x->scalar_type(), shared_expert_x, false);
    }

    void* oriX_ptr = nullptr;
    if(ori_x.has_value()) {
        oriX_ptr = get_first_tensor_address<c10::optional<at::Tensor>>(ori_x->scalar_type(), ori_x, false);
    }

    void* constExpertAlpha1_ptr = nullptr;
    if(const_expert_alpha_1.has_value()) {
        constExpertAlpha1_ptr = get_first_tensor_address<c10::optional<at::Tensor>>(const_expert_alpha_1->scalar_type(), const_expert_alpha_1, false);
    }

    void* constExpertAlpha2_ptr = nullptr;
    if(const_expert_alpha_2.has_value()) {
        constExpertAlpha2_ptr = get_first_tensor_address<c10::optional<at::Tensor>>(const_expert_alpha_2->scalar_type(), const_expert_alpha_2, false);
    }

    void* constExpertV_ptr = nullptr;
    if(const_expert_v.has_value()) {
        constExpertV_ptr = get_first_tensor_address<c10::optional<at::Tensor>>(const_expert_v->scalar_type(), const_expert_v, false);
    }

    void* performanceInfo_ptr = nullptr;
    if(performance_info.has_value()) {
        performanceInfo_ptr = get_first_tensor_address<c10::optional<at::Tensor>>(performance_info->scalar_type(), performance_info, false);
    }
    void* residualX_ptr = nullptr;
    void* gamma_ptr = nullptr;
    auto XOut_ptr = get_first_tensor_address<at::Tensor>(x_out.scalar_type(), x_out, false);

    
    int32_t tilingKey = 100;
    CalTilingKey(tilingKey, expand_x.scalar_type(), comm_quant_mode);

    moe_distribute_combine_v2_demo(tilingKey, tilingData.aivNum, stream, (GM_ADDR)expandX_ptr, (GM_ADDR)expertIds_ptr,
    (GM_ADDR)expandIdx_ptr, (GM_ADDR)epSendCount_ptr, (GM_ADDR)tpSendCounts_ptr, (GM_ADDR)residualX_ptr,
    (GM_ADDR)gamma_ptr, (GM_ADDR)expertScales_ptr, (GM_ADDR)xActiveMask_ptr, (GM_ADDR)sharedExpertX_ptr,
    (GM_ADDR)oriX_ptr, (GM_ADDR)constExpertAlpha1_ptr, (GM_ADDR)constExpertAlpha2_ptr,
    (GM_ADDR)constExpertV_ptr, (GM_ADDR)performanceInfo_ptr, (GM_ADDR)XOut_ptr,
    (GM_ADDR)workspace_ptr, (GM_ADDR)mc2Context_ptr, tilingData);
}

void calculate_tilingdata(MoeDistributeCombineV2Info &tilingData, int64_t ep_world_size, int64_t ep_rank_id,
    int64_t moe_expert_num, int64_t total_winsize_ep, int64_t tp_world_size, int64_t tp_rank_id,
    int64_t expert_shard_type, int64_t shared_expert_num, int64_t shared_expert_rank_num,
    int64_t global_bs, int64_t bs, int64_t h, int64_t k, int64_t a,
    int64_t zero_expert_num, int64_t copy_expert_num, int64_t const_expert_num,
    const c10::optional<at::Tensor> &x_active_mask,
    const c10::optional<at::Tensor> &shared_expert_x,
    const c10::optional<at::Tensor> &performance_info)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint64_t ubSizePlatFrom;
    ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatFrom);

    tilingData.epWorldSize = ep_world_size;
    tilingData.tpWorldSize = tp_world_size;
    tilingData.epRankId = ep_rank_id;
    tilingData.tpRankId = tp_rank_id;
    tilingData.expertShardType = expert_shard_type;
    tilingData.sharedExpertNum = shared_expert_num;
    tilingData.sharedExpertRankNum = shared_expert_rank_num;
    tilingData.moeExpertNum = moe_expert_num;
    tilingData.moeExpertPerRankNum = moe_expert_num / (ep_world_size - shared_expert_rank_num);
    tilingData.zeroExpertNum = zero_expert_num;
    tilingData.copyExpertNum = copy_expert_num;
    tilingData.constExpertNum = const_expert_num;
    tilingData.globalBs = global_bs;
    tilingData.bs = bs;
    tilingData.k = k;
    tilingData.h = h;
    tilingData.a = a;
    tilingData.aivNum = ascendcPlatform->GetCoreNumAiv();
    tilingData.isTokenMask = (x_active_mask.has_value() && x_active_mask->dim() == DIM_ONE);
    tilingData.isExpertMask = (x_active_mask.has_value() && x_active_mask->dim() == DIM_TWO);
    tilingData.hasSharedExpertX = shared_expert_x.has_value();  // 或根据实际变量名调整
    tilingData.isPerformance = performance_info.has_value();
    tilingData.reserved0 = false;
    tilingData.reserved1 = false;
    tilingData.reserved2 = false;
    tilingData.totalUbSize = 192 * 1024;
    tilingData.totalWinSizeEp = total_winsize_ep;
    tilingData.totalWinSizeTp = 0;
    tilingData.bufferNum = 2;
}

at::Tensor npu_moe_distribute_combine_v2(
    const at::Tensor &expand_x, const at::Tensor &expert_ids,
    const at::Tensor &assist_info_for_combine,
    const at::Tensor &ep_send_counts, const at::Tensor &expert_scales, const at::Tensor &mc2_context,
    c10::string_view group_ep, int64_t ep_world_size, int64_t ep_rank_id,
    int64_t moe_expert_num, int64_t total_winsize_ep, 
    const c10::optional<at::Tensor> &tp_send_counts,
    const c10::optional<at::Tensor> &x_active_mask,
    const c10::optional<at::Tensor> &shared_expert_x,
    const c10::optional<at::Tensor> &ori_x,
    const c10::optional<at::Tensor> &const_expert_alpha_1,
    const c10::optional<at::Tensor> &const_expert_alpha_2,
    const c10::optional<at::Tensor> &const_expert_v,
    const c10::optional<at::Tensor> &performance_info,
    c10::string_view group_tp, int64_t tp_world_size, int64_t tp_rank_id,
    int64_t expert_shard_type, int64_t shared_expert_num, int64_t shared_expert_rank_num,
    int64_t global_bs, int64_t comm_quant_mode,
    c10::string_view comm_alg, int64_t zero_expert_num, int64_t copy_expert_num, int64_t const_expert_num)
{
    auto expand_x_size = expand_x.sizes();
    auto expert_ids_size = expert_ids.sizes();
    
    int64_t bs = expert_ids_size[0];
    int64_t h = expand_x_size[1];
    int64_t k = expert_ids_size[1];

    bool is_shared_default = ((shared_expert_num == 1) && (shared_expert_rank_num == 0));
    bool is_no_shared = ((shared_expert_num == 0) && (shared_expert_rank_num == 0));
    
    bool shared_front = (expert_shard_type == 0);
    int64_t local_moe_expert_num = 1;
    int64_t global_bs_real = (global_bs == 0) ? (bs * ep_world_size) : global_bs;
    int64_t a = 0;
    if (shared_front) {
        if (ep_rank_id < shared_expert_rank_num) {
            local_moe_expert_num = 1;
            int64_t max_bs = global_bs_real / ep_world_size;  // 前面已有拦截，保证ep_world_size > 0
            int64_t rank_num_per_shared_expert = shared_expert_rank_num / shared_expert_num;  // 前面已有拦截, 保证进入该分支时shared_expert_num > 0
            int64_t max_shared_group_num = (ep_world_size + rank_num_per_shared_expert - 1) / rank_num_per_shared_expert;
            a = max_bs * max_shared_group_num;
        } else {
            local_moe_expert_num = moe_expert_num / (ep_world_size - shared_expert_rank_num);
            a = global_bs_real * std::min(local_moe_expert_num, k);
        }
    }
    TORCH_CHECK((expand_x.dim() == DIM_TWO) && (expert_ids.dim() == DIM_TWO), "The x and expert_ids should be 2D");
    TORCH_CHECK((expand_x.scalar_type() == at::kBFloat16) || (expand_x.scalar_type() == at::kHalf) || (expand_x.scalar_type() == at::kInt),
                "dtype of expand_x should be BFloat16, Float16 or Int, but got " + std::string(c10::toString(expand_x.scalar_type())));
    TORCH_CHECK(expert_ids.scalar_type() == at::kInt,
                "dtype of expert_ids should be Int, but got " + std::string(c10::toString(expert_ids.scalar_type())));

    char *group_ep_ptr = const_cast<char *>(group_ep.data());
    std::string group_tp_str = std::string(group_tp);
    char *group_tp_ptr = const_cast<char *>(group_tp_str.c_str());
    at::Tensor output;
    at::Tensor new_workspace = at::empty({WORKSPACESIZE / 4}, expert_ids.options().dtype(at::kInt));
    if (expand_x.scalar_type() != at::kInt) {
        output = at::empty({bs, h}, expert_ids.options().dtype(expand_x.scalar_type()));
    } else {
        output = at::empty({bs, h}, expert_ids.options().dtype(at::kBFloat16));
    }

    c10::optional<at::Tensor> nulltensor = c10::nullopt;
    int64_t out_dtype = 0;
    int64_t group_list_type = 0;

    std::string comm_alg_str = std::string(comm_alg);
    char *comm_alg_ptr = const_cast<char *>(comm_alg_str.c_str());

    MoeDistributeCombineV2Info tilingData;
    calculate_tilingdata(tilingData, ep_world_size, ep_rank_id, moe_expert_num, total_winsize_ep, tp_world_size,
        tp_rank_id, expert_shard_type, shared_expert_num, shared_expert_rank_num, global_bs, bs, h, k, a, zero_expert_num,
        copy_expert_num, const_expert_num, x_active_mask, shared_expert_x, performance_info);

    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    auto acl_call = [=]() mutable -> int {
        MoeDistributeCombineV2_api(stream, expand_x, expert_ids, assist_info_for_combine, ep_send_counts,
            expert_scales, new_workspace, mc2_context, tp_send_counts, x_active_mask,
            shared_expert_x, ori_x, const_expert_alpha_1, const_expert_alpha_2,
            const_expert_v, performance_info, output, comm_quant_mode, tilingData);
        return 0;
    };
    at_npu::native::OpCommand::RunOpApiV2("moeDistributeCombineV2", acl_call);
    
    return output;
}

at::Tensor npu_moe_distribute_combine_v2_meta(
    const at::Tensor &expand_x, const at::Tensor &expert_ids,
    const at::Tensor &assist_info_for_combine,
    const at::Tensor &ep_send_counts, const at::Tensor &expert_scales, const at::Tensor &mc2_context,
    c10::string_view group_ep, int64_t ep_world_size, int64_t ep_rank_id,
    int64_t moe_expert_num, int64_t total_winsize_ep, 
    const c10::optional<at::Tensor> &tp_send_counts,
    const c10::optional<at::Tensor> &x_active_mask,
    const c10::optional<at::Tensor> &shared_expert_x,
    const c10::optional<at::Tensor> &ori_x,
    const c10::optional<at::Tensor> &const_expert_alpha_1,
    const c10::optional<at::Tensor> &const_expert_alpha_2,
    const c10::optional<at::Tensor> &const_expert_v,
    const c10::optional<at::Tensor> &performance_info,
    c10::string_view group_tp, int64_t tp_world_size, int64_t tp_rank_id,
    int64_t expert_shard_type, int64_t shared_expert_num, int64_t shared_expert_rank_num,
    int64_t global_bs, int64_t comm_quant_mode,
    c10::string_view comm_alg, int64_t zero_expert_num, int64_t copy_expert_num, int64_t const_expert_num)
{
    auto expert_ids_size = expert_ids.sizes();
    auto expand_x_size = expand_x.sizes();
    
    int64_t bs = expert_ids_size[0];
    int64_t h = expand_x_size[1];
    
    at::Tensor output;

    if (expand_x.scalar_type() != at::kInt) {
        output = at::empty({bs, h}, expert_ids.options().dtype(expand_x.scalar_type()));
    } else {
        output = at::empty({bs, h}, expert_ids.options().dtype(at::kBFloat16));
    }
    return output;
}

TORCH_LIBRARY_IMPL(ascend_ops, PrivateUse1, m)
{
    m.impl("MoeDistributeCombineV2", TORCH_FN(npu_moe_distribute_combine_v2));
}

TORCH_LIBRARY_IMPL(ascend_ops, Meta, m)
{
    m.impl("MoeDistributeCombineV2", &npu_moe_distribute_combine_v2_meta);
}

} // namespace MoeDistributeCombineV2
} // namespace ascend_ops
