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
 * \file moe_distribute_dispatch_v2_torch.cpp
 * \brief
 */
#include "acl/acl.h"
#include "tiling/platform/platform_ascendc.h"

#include "kernel_operator.h"

#include "update_context.h"
#include "moe_distribute_dispatch_v2_entry.h"
#include "op_kernel/moe_distribute_dispatch_v2_tiling.h"

namespace ascend_ops {

namespace MoeDistributeDispatchV2 {

void MoeDistributeDispatchV2_api(aclrtStream stream, const at::Tensor &x, const at::Tensor &expert_ids,
                                c10::string_view group_ep, int64_t ep_world_size, int64_t ep_rank_id,
                                int64_t moe_expert_num,
                                const c10::optional<at::Tensor> &scales,
                                const c10::optional<at::Tensor> &x_active_mask,
                                const c10::optional<at::Tensor> &expert_scales,
                                const c10::optional<at::Tensor> &elastic_info,
                                const c10::optional<at::Tensor> &performance_info,
                                at::Tensor &expand_x, at::Tensor &dynamic_scales, at::Tensor &assist_info_forcombine, 
                                at::Tensor &expert_token_nums, at::Tensor &ep_recv_counts, at::Tensor &tp_recv_counts, 
                                at::Tensor &expand_scales
                                Mc2ContextStru mc2Context, MoeDistributeDispatchV2TilingData tilingData)
{
    auto x_ptr = get_first_tensor_address<at::TensorList>(x.scalar_type(), x, false);
    auto expertIds_ptr = get_first_tensor_address<at::TensorList>(matchedCombo.scalar_type(), expert_ids, false);
    auto scales_ptr = get_first_tensor_address<at::TensorList>(scales.scalar_type(), scales, false);
    auto xActiveMask_ptr = get_first_tensor_address<c10::optional<at::TensorList>>(xActiveMask.scalar_type(), x_active_mask, true);
    auto expertScales_ptr = get_first_tensor_address<c10::optional<at::TensorList>>(expertScales.scalar_type(), expert_scales, true);
    auto elasticInfo_ptr = get_first_tensor_address<c10::optional<at::TensorList>>(elastic_info.scalar_type(), elastic_info, true);
    auto performanceInfo_ptr = get_first_tensor_address<c10::optional<at::TensorList>>(performance_info.scalar_type(), performance_info, true);
    auto expandXOut_ptr = get_first_tensor_address<c10::optional<at::TensorList>>(expand_x.scalar_type(), expand_x, true);
    auto dynamicScalesOut_ptr =
        get_first_tensor_address<c10::optional<at::TensorList>>(dynamic_scales.scalar_type(), dynamic_scales, true);
    auto assistInfoOut_ptr =
        get_first_tensor_address<c10::optional<at::TensorList>>(assist_info_forcombine.scalar_type(), assist_info_forcombine, true);
    auto expertTokenNumsOut_ptr =
        get_first_tensor_address<c10::optional<torch::Tensor>>(expert_token_nums.scalar_type(), expert_token_nums, true);
    auto epSendCountsOut_ptr =
        get_first_tensor_address<c10::optional<at::TensorList>>(ep_recv_counts.scalar_type(), ep_recv_counts, true);
    auto tpSendCountsOut_ptr =
        get_first_tensor_address<c10::optional<at::TensorList>>(tp_recv_counts.scalar_type(), tp_recv_counts, true);
    auto expandScalesOut_ptr =
        get_first_tensor_address<c10::optional<at::TensorList>>(assist_info_forcombine.scalar_type(), assist_info_forcombine, true);
    uint64_t workspaceSize = 16U * 1024U * 1024U;
    void *workspace_ptr = nullptr;
    if (workspaceSize > 0) {
        auto ret = aclrtMalloc(&workspace_ptr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        TORCH_CHECK(ret == ACL_SUCCESS, "allocate workspace failed. ERROR: %d\n", ret);
    }
    int8_t type = 1;
    moe_distribute_dispatch_v2_demo(type, tilingData.moeDistributeDispatchV2Info.aivNum, (void*)stream, (__gm__ uint8_t *)x,
        (__gm__ uint8_t *)expertIds_ptr, (__gm__ uint8_t *)scales, (__gm__ uint8_t *)xActiveMask_ptr,
        (__gm__ uint8_t *)expertScales_ptr, (__gm__ uint8_t *)elasticInfo_ptr, (__gm__ uint8_t *)performanceInfo_ptr,
        (__gm__ uint8_t *)expandXOut_ptr, (__gm__ uint8_t *)dynamicScalesOut_ptr, (__gm__ uint8_t *)assistInfoOut_ptr,
        (__gm__ uint8_t *)expertTokenNumsOut_ptr, (__gm__ uint8_t *)epSendCountsOut_ptr, (__gm__ uint8_t *)tpSendCountsOut_ptr,
        (__gm__ uint8_t *)expandScalesOut_ptr, (__gm__ uint8_t *)workspace_ptr, (__gm__ uint8_t *)mc2Context, (__gm__ uint8_t *)tilingData);
}

tensor_list npu_moe_distribute_dispatch_v2(const at::Tensor &x, const at::Tensor &expert_ids,
                                           c10::string_view group_ep, int64_t ep_world_size, int64_t ep_rank_id,
                                           int64_t moe_expert_num, int64_t total_winsize_ep, 
                                           const c10::optional<at::Tensor> &scales,
                                           const c10::optional<at::Tensor> &x_active_mask,
                                           const c10::optional<at::Tensor> &expert_scales,
                                           const c10::optional<at::Tensor> &elastic_info,
                                           const c10::optional<at::Tensor> &performance_info,
                                           c10::string_view group_tp, int64_t tp_world_size, int64_t tp_rank_id,
                                           int64_t expert_shard_type, int64_t shared_expert_num, int64_t shared_expert_rank_num,
                                           int64_t quant_mode, int64_t global_bs, int64_t expert_token_nums_type,
                                           c10::string_view comm_alg, int64_t zero_expert_num, int64_t copy_expert_num, int64_t const_expert_num)
{
    TORCH_CHECK((x.dim() == DIM_TWO) && (expert_ids.dim() == DIM_TWO), "The x and expert_ids should be 2D", OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(x.scalar_type() == at::kBFloat16 || x.scalar_type() == at::kHalf,
                "dtype of x should be BFloat16 or Half, but got " + std::string(c10::toString(x.scalar_type())),
                OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(expert_ids.scalar_type() == at::kInt,
                "dtype of expert_ids should be Int, but got " + std::string(c10::toString(expert_ids.scalar_type())),
                OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK((ep_rank_id >= 0) && (ep_rank_id < ep_world_size),
                "ep_rank_id should be in [0, ep_world_size), but got",
                " ep_world_size: ", ep_world_size,
                ", ep_rank_id: ", ep_rank_id,
                ". " + OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK((shared_expert_rank_num >= 0) && (shared_expert_rank_num < ep_world_size),
                "shared_expert_rank_num should be in [0, ep_world_size), but got",
                " ep_world_size: ", ep_world_size,
                ", shared_expert_rank_num: ", shared_expert_rank_num,
                ". " + OPS_ERROR(ErrCode::PARAM));
    bool is_shared_default = ((shared_expert_num == 1) && (shared_expert_rank_num == 0));
    bool is_no_shared = ((shared_expert_num == 0) && (shared_expert_rank_num == 0));
    bool is_valid_shared = ((shared_expert_num > 0)
        && ((shared_expert_rank_num / shared_expert_num) > 0)
        && ((shared_expert_rank_num % shared_expert_num) == 0));
    TORCH_CHECK(is_shared_default || is_no_shared || is_valid_shared,
                "shared_expert_num and shared_expertrank_num have obvious value situations: "
                "1. shared_expert_num is 1, shared_expert_rank_num is 0; 2. shared_expert num is 0, "
                "shared_expert_rank_num is 0; 3. shared_expert_num in (0, shared_expert_rank_num] and "
                "shared_expert_rank_num % shared_expert_num = 0. but the current input value is ",
                " shared_expert_num: ", shared_expert_num,
                ", shared_expert_rank_num: ", shared_expert_rank_num,
                ". " + OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK((expert_token_nums_type == 0) || (expert_token_nums_type == 1),
                "The expert_token_nums_type should be 0 or 1.", OPS_ERROR(ErrCode::PARAM));
    auto x_size = x.sizes();
    auto expert_ids_size = expert_ids.sizes();

    int64_t bs = x_size[0];
    int64_t h = x_size[1];
    int64_t k = expert_ids_size[1];

    // a2 expert_shard_type、shared_expert_rank_num 应为0
    bool shared_front = (expert_shard_type == 0);
    int64_t local_moe_expert_num = 1;
    int64_t global_bs_real = (global_bs == 0) ? (bs * ep_world_size) : global_bs;
    int64_t a = 0;
    int64_t ep_recv_cnt_num = 0;
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
        if (elastic_info.has_value()) {
            if ((is_shared_default) || (is_no_shared)) {
                local_moe_expert_num = std::max(local_moe_expert_num, moe_expert_num / (ep_world_size - shared_expert_rank_num));
                a = global_bs_real * std::min(local_moe_expert_num, k);
            } else {
                int64_t max_bs = global_bs_real / ep_world_size;
                int64_t rank_num_per_shared_expert = shared_expert_rank_num / shared_expert_num;
                int64_t max_shared_group_num = (ep_world_size + rank_num_per_shared_expert - 1) / rank_num_per_shared_expert;
                a = std::max(max_bs * max_shared_group_num, global_bs_real * std::min(moe_expert_num / (ep_world_size - shared_expert_rank_num), k));
                local_moe_expert_num = std::max(local_moe_expert_num, moe_expert_num / (ep_world_size - shared_expert_rank_num));
            }
        }
    }
    if (tp_world_size == DIM_TWO) {
        ep_recv_cnt_num = ep_world_size * local_moe_expert_num * tp_world_size;
    } else {
        ep_recv_cnt_num = ep_world_size * local_moe_expert_num;
    }

    auto output_dtype = (!scales.has_value() && quant_mode == 0) ? x.scalar_type() : at::kChar;
    char *group_ep_ptr = const_cast<char *>(group_ep.data());
    std::string group_tp_str = std::string(group_tp);
    char *group_tp_ptr = const_cast<char *>(group_tp_str.c_str());
    at::Tensor expand_x {nullptr};
    at::Tensor dynamic_scales {nullptr};
    if (tp_world_size == 0) {
        expand_x = npu_preparation::apply_tensor_without_format({a, h}, x.options().dtype(output_dtype));
        dynamic_scales = npu_preparation::apply_tensor_without_format({a}, x.options().dtype(at::kFloat));
    } else {
        expand_x = npu_preparation::apply_tensor_without_format({a * tp_world_size, h}, x.options().dtype(output_dtype));
        dynamic_scales = npu_preparation::apply_tensor_without_format({a * tp_world_size}, x.options().dtype(at::kFloat));
    }
    
    at::Tensor expert_token_nums = npu_preparation::apply_tensor_without_format({local_moe_expert_num}, x.options().dtype(at::kLong));
    at::Tensor ep_recv_counts = npu_preparation::apply_tensor_without_format({ep_recv_cnt_num}, x.options().dtype(at::kInt));
    at::Tensor tp_recv_counts = npu_preparation::apply_tensor_without_format({tp_world_size}, x.options().dtype(at::kInt));
    at::Tensor assist_info_forcombine{nullptr};

    // a2分层方案
    at::Tensor expand_scales = npu_preparation::apply_tensor_without_format({a}, x.options().dtype(at::kFloat));
    if (expert_scales.has_value() && expert_scales.value().defined()) {
        ep_recv_cnt_num = ep_world_size * local_moe_expert_num + 2 * global_bs_real * k * (ep_world_size / 8); // 2: 2 buffer, 8 ranknum per server
        ep_recv_counts = npu_preparation::apply_tensor_without_format({ep_recv_cnt_num}, x.options().dtype(at::kInt));
    }

    std::string comm_alg_str = std::string(comm_alg);
    char *comm_alg_ptr = const_cast<char *>(comm_alg_str.c_str());

    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint64_t ubSizePlatFrom;
    ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatFrom);
    MoeDistributeDispatchV2TilingData tilingData;
    // 初始化TilingData结构体的成员变量
    tilingData.moeDistributeDispatchV2Info.epWorldSize = ep_world_size;                // epWorldSize
    tilingData.moeDistributeDispatchV2Info.tpWorldSize = tp_world_size;                // tpWorldSize
    tilingData.moeDistributeDispatchV2Info.epRankId = ep_rank_id;                   // epRankId
    tilingData.moeDistributeDispatchV2Info.tpRankId = tp_rank_id;                   // tpRankId
    tilingData.moeDistributeDispatchV2Info.expertShardType = expert_shard_type;            // expert type
    tilingData.moeDistributeDispatchV2Info.sharedExpertNum = shared_expert_num;            // shared expert number
    tilingData.moeDistributeDispatchV2Info.sharedExpertRankNum = shared_expert_rank_num;        // shared expert rank number
    tilingData.moeDistributeDispatchV2Info.moeExpertNum = moe_expert_num;               // moe expert number
    tilingData.moeDistributeDispatchV2Info.quantMode = quant_mode;                  // quant mode
    tilingData.moeDistributeDispatchV2Info.globalBs = global_bs;                   // globalBs = gBS * worldSize
    tilingData.moeDistributeDispatchV2Info.bs = bs;                         // bs
    tilingData.moeDistributeDispatchV2Info.k = k;                          // k
    tilingData.moeDistributeDispatchV2Info.h = h;                          // h
    tilingData.moeDistributeDispatchV2Info.a = a;                          // a
    tilingData.moeDistributeDispatchV2Info.aivNum = ascendcPlatform->GetCoreNumAiv();                     // aivNum
    tilingData.moeDistributeDispatchV2Info.isTokenMask = (x_active_mask.has_value() && x_active_mask.dim() == DIM_ONE);                    // input active mask 1dims or not
    tilingData.moeDistributeDispatchV2Info.isExpertMask = (x_active_mask.has_value() && x_active_mask.dim() == DIM_TWO);                   // input active mask 2dims or not
    tilingData.moeDistributeDispatchV2Info.hasElasticInfo = elastic_info.has_value();                 // has elasticinfo or not
    tilingData.moeDistributeDispatchV2Info.isPerformance = performance_info.has_value();                  // whether performance or not
    tilingData.moeDistributeDispatchV2Info.isQuant = (quant_mode != 0);                        // whether quant or not
    tilingData.moeDistributeDispatchV2Info.reserved0 = false;
    tilingData.moeDistributeDispatchV2Info.reserved1 = false;
    tilingData.moeDistributeDispatchV2Info.reserved2 = false;
    tilingData.moeDistributeDispatchV2Info.totalUbSize = 192 * 1024;                // epWorldSize
    tilingData.moeDistributeDispatchV2Info.totalWinSizeEp = total_winsize_ep;
    // tilingData.moeDistributeDispatchV2Info.totalWinSizeTp = totalWinSizeTp;
    tilingData.moeDistributeDispatchV2Info.expertTokenNumsType = expert_token_nums_type;        // expert token nums type, support 0: cumsum mode, 1: count mode
    tilingData.moeDistributeDispatchV2Info.zeroComputeExpertNum = zero_expert_num + copy_expert_num + const_expert_num;       // sum of zero, copy and const expert nums
    tilingData.moeDistributeDispatchV2Info.scalesRow = scales.has_value() ? scales[0] : 0;
    tilingData.moeDistributeDispatchV2Info.scalesCol = scales.has_value() ? scales[1] : 0;
    tilingData.moeDistributeDispatchV2Info.scalesTypeSize = scales.has_value() ? sizeof(scales.scalar_type()) : 0;
    tilingData.moeDistributeDispatchV2Info.scalesCount = scales.has_value() ? (scales[0] * scales[1]) : 0;

    auto context = update_context(group_ep_ptr.c_str(), ep_world_size);
    uint64_t workspaceSize = 16U * 1024U * 1024U;
    void *workspace_ptr = nullptr;
    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    if (workspaceSize > 0) {
        auto ret = aclrtMalloc(&workspace_ptr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        TORCH_CHECK(ret == ACL_SUCCESS, "allocate workspace failed. ERROR: %d\n", ret);
    }
    auto acl_call = [=]() -> int {
        MoeDistributeDispatchV2_api(stream, x, expert_ids, group_ep, ep_world_size, ep_rank_id, moe_expert_num, scales,
                                    x_active_mask, expert_scales, elastic_info, performance_info, expand_x,
                                    dynamic_scales, assist_info_forcombine, expert_token_nums, ep_recv_counts, tp_recv_counts,
                                    expand_scales, context, tilingData);
        return 0;
    };
    at_npu::native::OpCommand::RunOpApiV2("MoeDistributeDispatchV2", acl_call);
    
    return std::tie(expand_x, dynamic_scales, assist_info_forcombine, expert_token_nums, ep_recv_counts, tp_recv_counts,
        expand_scales);
}

// Register Ascend implementations for MoeDistributeDispatchV2
TORCH_LIBRARY_IMPL(ascend_ops, PrivateUse1, m)
{
    m.impl("MoeDistributeDispatchV2", npu_moe_distribute_dispatch_v2);
}

} // namespace MoeDistributeDispatchV2
} // namespace ascend_ops
