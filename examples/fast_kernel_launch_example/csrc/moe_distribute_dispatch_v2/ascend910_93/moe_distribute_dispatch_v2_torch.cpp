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

#include "moe_distribute_dispatch_v2_entry.h"
#include "op_kernel/moe_distribute_dispatch_v2_tiling.h"
#include "moe_distribute_dispatch_v2_torch.h"
// #include <ATen/Operators.h>
// #include <torch/all.h>
// #include <torch/library.h>
// #include "torch_npu/csrc/core/npu/NPUStream.h"
// #include "torch_npu/csrc/framework/OpCommand.h"
// // #include "kernel_operator.h"
// // #include "platform/platform_ascendc.h"
// #include <type_traits>


#include <ATen/ATen.h>
#include <vector>
#include <torch/all.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

namespace ascend_ops {

namespace MoeDistributeDispatchV2 {

TORCH_LIBRARY_FRAGMENT(EXTENSION_MODULE_NAME, m)
{
    m.def("MoeDistributeDispatchV2(Tensor x, Tensor expert_ids, Tensor new_workspace, Tensor mc2_context, str group_ep, int ep_world_size, " \
            "int ep_rank_id, int moe_expert_num, int total_winsize_ep, *, Tensor? scales, Tensor? x_active_mask, " \
            "Tensor? expert_scales, Tensor? elastic_info, Tensor? performance_info, str group_tp, "\
            "int tp_world_size, int tp_rank_id, int expert_shard_type, int shared_expert_num, " \
            "int shared_expert_rank_num, int quant_mode, int global_bs, int expert_token_nums_type, " \
            "str comm_alg, int zero_expert_num, int copy_expert_num, int const_expert_num " \
            ") " \
            "-> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
}
// ,int? y_dtype=None, int? x_dtype=None, int? scales_dtype=None

constexpr uint32_t DIM_ONE = 1UL;
constexpr uint32_t DIM_TWO = 2UL;
constexpr uint32_t TILINGKEY_XTYPE = 10;
constexpr uint32_t TILINGKEY_SCALES = 100;
constexpr uint32_t TILINGKEY_COMM_ALG = 1000;
// using npu_preparation = at_npu::native::OpPreparation;

static void CalTilingKey(int32_t &tilingKey, at::ScalarType xType, const bool isScales, const uint32_t quantMode, const bool isSetCommAlg)
{
    tilingKey += static_cast<uint64_t>(quantMode);
    // 检查是否为 bfloat16 (kBFloat16)
    if (xType == at::kBFloat16) {
        tilingKey += static_cast<uint64_t>(TILINGKEY_XTYPE);
    }
    if (isScales) {
        tilingKey += static_cast<uint64_t>(TILINGKEY_SCALES);
    }
    if (isSetCommAlg) {
        tilingKey += static_cast<uint64_t>(TILINGKEY_COMM_ALG);
    }

    return;
}

void MoeDistributeDispatchV2_api(aclrtStream stream, bool is_fullmesh_v2, const at::Tensor &x, const at::Tensor &expert_ids, const at::Tensor &new_workspace, const at::Tensor &mc2_context,
                                c10::string_view group_ep, int64_t ep_world_size, int64_t ep_rank_id,
                                int64_t moe_expert_num,
                                const c10::optional<at::Tensor> &scales,
                                const c10::optional<at::Tensor> &x_active_mask,
                                const c10::optional<at::Tensor> &expert_scales,
                                const c10::optional<at::Tensor> &elastic_info,
                                const c10::optional<at::Tensor> &performance_info,
                                at::Tensor &expand_x, at::Tensor &dynamic_scales, at::Tensor &assist_info_forcombine, 
                                at::Tensor &expert_token_nums, at::Tensor &ep_recv_counts, at::Tensor &tp_recv_counts, 
                                at::Tensor &expand_scales, MoeDistributeDispatchV2TilingData tilingData)
{
    auto x_ptr = get_first_tensor_address<at::Tensor>(x.scalar_type(), x, false);
    auto expertIds_ptr = get_first_tensor_address<at::Tensor>(expert_ids.scalar_type(), expert_ids, false);
    auto workspace_ptr = get_first_tensor_address<at::Tensor>(new_workspace.scalar_type(), new_workspace, false);
    auto mc2Context_ptr = get_first_tensor_address<at::Tensor>(mc2_context.scalar_type(), mc2_context, false);
    void* scales_ptr = nullptr; 
    if(scales.has_value()) {
        scales_ptr = get_first_tensor_address<c10::optional<at::Tensor>>(scales->scalar_type(), scales, false);
    } 
    void* xActiveMask_ptr = nullptr;
    if(x_active_mask.has_value()) {
        xActiveMask_ptr = get_first_tensor_address<c10::optional<at::Tensor>>(x_active_mask->scalar_type(), x_active_mask, false);
    }
    void* expertScales_ptr = nullptr;
    if(expert_scales.has_value()) {
        expertScales_ptr = get_first_tensor_address<c10::optional<at::Tensor>>(expert_scales->scalar_type(), expert_scales, false);
    }
    void* elasticInfo_ptr = nullptr;
    if(elastic_info.has_value()) {
        elasticInfo_ptr = get_first_tensor_address<c10::optional<at::Tensor>>(elastic_info->scalar_type(), elastic_info, false);
    }
    void* performanceInfo_ptr = nullptr;
    if(performance_info.has_value()) {
        performanceInfo_ptr = get_first_tensor_address<c10::optional<at::Tensor>>(performance_info->scalar_type(), performance_info, false);
    }
    auto expandXOut_ptr = get_first_tensor_address<at::Tensor>(expand_x.scalar_type(), expand_x, false);
    auto dynamicScalesOut_ptr =
        get_first_tensor_address<at::Tensor>(dynamic_scales.scalar_type(), dynamic_scales, false);
    auto assistInfoOut_ptr =
        get_first_tensor_address<at::Tensor>(assist_info_forcombine.scalar_type(), assist_info_forcombine, false);
    auto expertTokenNumsOut_ptr =
        get_first_tensor_address<at::Tensor>(expert_token_nums.scalar_type(), expert_token_nums, false);
    auto epSendCountsOut_ptr =
        get_first_tensor_address<at::Tensor>(ep_recv_counts.scalar_type(), ep_recv_counts, false);
    auto tpSendCountsOut_ptr =
        get_first_tensor_address<at::Tensor>(tp_recv_counts.scalar_type(), tp_recv_counts, false);
    auto expandScalesOut_ptr =
        get_first_tensor_address<at::Tensor>(assist_info_forcombine.scalar_type(), assist_info_forcombine, false);
    int32_t tilingKey = 10000;
    CalTilingKey(tilingKey, x.scalar_type(), scales.has_value(), tilingData.moeDistributeDispatchV2Info.quantMode, is_fullmesh_v2);
    moe_distribute_dispatch_v2_demo(tilingKey, tilingData.moeDistributeDispatchV2Info.aivNum, (void*)stream, (GM_ADDR)x_ptr,
        (GM_ADDR)expertIds_ptr, (GM_ADDR)scales_ptr, (GM_ADDR)xActiveMask_ptr,
        (GM_ADDR)expertScales_ptr, (GM_ADDR)elasticInfo_ptr, (GM_ADDR)performanceInfo_ptr,
        (GM_ADDR)expandXOut_ptr, (GM_ADDR)dynamicScalesOut_ptr, (GM_ADDR)assistInfoOut_ptr,
        (GM_ADDR)expertTokenNumsOut_ptr, (GM_ADDR)epSendCountsOut_ptr, (GM_ADDR)tpSendCountsOut_ptr,
        (GM_ADDR)expandScalesOut_ptr, (GM_ADDR)workspace_ptr, (GM_ADDR)mc2Context_ptr, tilingData.moeDistributeDispatchV2Info);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> npu_moe_distribute_dispatch_v2(
                                            const at::Tensor &x, const at::Tensor &expert_ids, const at::Tensor &new_workspace, const at::Tensor &mc2_context,
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
    TORCH_CHECK((x.dim() == DIM_TWO) && (expert_ids.dim() == DIM_TWO), "The x and expert_ids should be 2D");
    TORCH_CHECK(x.scalar_type() == at::kBFloat16 || x.scalar_type() == at::kHalf,
                "dtype of x should be BFloat16 or Half, but got " + std::string(c10::toString(x.scalar_type())));
    TORCH_CHECK(expert_ids.scalar_type() == at::kInt,
                "dtype of expert_ids should be Int, but got " + std::string(c10::toString(expert_ids.scalar_type())));
    TORCH_CHECK((ep_rank_id >= 0) && (ep_rank_id < ep_world_size),
                "ep_rank_id should be in [0, ep_world_size), but got",
                " ep_world_size: ", ep_world_size,
                ", ep_rank_id: ", ep_rank_id);
    TORCH_CHECK((shared_expert_rank_num >= 0) && (shared_expert_rank_num < ep_world_size),
                "shared_expert_rank_num should be in [0, ep_world_size), but got",
                " ep_world_size: ", ep_world_size,
                ", shared_expert_rank_num: ", shared_expert_rank_num);
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
                ", shared_expert_rank_num: ", shared_expert_rank_num);
    TORCH_CHECK((expert_token_nums_type == 0) || (expert_token_nums_type == 1),
                "The expert_token_nums_type should be 0 or 1.");
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
    at::Tensor expand_x;
    at::Tensor dynamic_scales;
    if (tp_world_size == 0) {
        expand_x = at::empty({a, h}, x.options().dtype(output_dtype));
        dynamic_scales = at::empty({a}, x.options().dtype(at::kFloat));
    } else {
        expand_x = at::empty({a * tp_world_size, h}, x.options().dtype(output_dtype));
        dynamic_scales = at::empty({a * tp_world_size}, x.options().dtype(at::kFloat));
    }
    
    at::Tensor expert_token_nums = at::empty({local_moe_expert_num}, x.options().dtype(at::kLong));
    at::Tensor ep_recv_counts = at::empty({ep_recv_cnt_num}, x.options().dtype(at::kInt));
    at::Tensor tp_recv_counts = at::empty({tp_world_size}, x.options().dtype(at::kInt));
    at::Tensor assist_info_forcombine{nullptr};

    // a2分层方案
    at::Tensor expand_scales = at::empty({a}, x.options().dtype(at::kFloat));
    if (expert_scales.has_value() && expert_scales.value().defined()) {
        ep_recv_cnt_num = ep_world_size * local_moe_expert_num + 2 * global_bs_real * k * (ep_world_size / 8); // 2: 2 buffer, 8 ranknum per server
        ep_recv_counts = at::empty({ep_recv_cnt_num}, x.options().dtype(at::kInt));
    }

    bool is_fullmesh_v2 = (comm_alg == "fullmesh_v2");

    assist_info_forcombine = at::empty({std::max(bs * k, a * 128)}, x.options().dtype(at::kInt));

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
    tilingData.moeDistributeDispatchV2Info.isTokenMask = (x_active_mask.has_value() && x_active_mask->dim() == DIM_ONE);                    // input active mask 1dims or not
    tilingData.moeDistributeDispatchV2Info.isExpertMask = (x_active_mask.has_value() && x_active_mask->dim() == DIM_TWO);                   // input active mask 2dims or not
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
    tilingData.moeDistributeDispatchV2Info.scalesRow = scales.has_value() ? scales.value()[0].item().toLong() : 0;
    tilingData.moeDistributeDispatchV2Info.scalesCol = scales.has_value() ? scales.value()[1].item().toLong() : 0;
    tilingData.moeDistributeDispatchV2Info.scalesTypeSize = scales.has_value() ? at::elementSize(scales.value().scalar_type()) : 0;
    tilingData.moeDistributeDispatchV2Info.scalesCount = scales.has_value() ? (scales.value()[0].item().toLong() * scales.value()[1].item().toLong()) : 0;
    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    auto acl_call = [=]() mutable -> int {
        MoeDistributeDispatchV2_api(stream, is_fullmesh_v2, x, expert_ids, new_workspace, mc2_context, group_ep, ep_world_size, ep_rank_id, moe_expert_num, scales,
                                    x_active_mask, expert_scales, elastic_info, performance_info, expand_x,
                                    dynamic_scales, assist_info_forcombine, expert_token_nums, ep_recv_counts, tp_recv_counts,
                                    expand_scales, tilingData);
        return 0;
    };
    at_npu::native::OpCommand::RunOpApiV2("moeDistributeDispatchV2", acl_call);
    
    return std::tie(expand_x, dynamic_scales, assist_info_forcombine, expert_token_nums, ep_recv_counts, tp_recv_counts,
        expand_scales);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> npu_moe_distribute_dispatch_v2_meta(
                                           const at::Tensor &x, const at::Tensor &expert_ids, const at::Tensor &new_workspace, const at::Tensor &mc2_context,
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
    bool is_shared_default = ((shared_expert_num == 1) && (shared_expert_rank_num == 0));
    bool is_no_shared = ((shared_expert_num == 0) && (shared_expert_rank_num == 0));
    bool is_valid_shared = ((shared_expert_num > 0)        
        && ((shared_expert_rank_num / shared_expert_num) > 0)
        && ((shared_expert_rank_num % shared_expert_num) == 0));
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
    at::Tensor expand_x;
    at::Tensor dynamic_scales;
    if (tp_world_size == 0) {
        expand_x = at::empty({a, h}, x.options().dtype(output_dtype));
        dynamic_scales = at::empty({a}, x.options().dtype(at::kFloat));
    } else {
        expand_x = at::empty({a * tp_world_size, h}, x.options().dtype(output_dtype));
        dynamic_scales = at::empty({a * tp_world_size}, x.options().dtype(at::kFloat));
    }
    
    at::Tensor expert_token_nums = at::empty({local_moe_expert_num}, x.options().dtype(at::kLong));
    at::Tensor ep_recv_counts = at::empty({ep_recv_cnt_num}, x.options().dtype(at::kInt));
    at::Tensor tp_recv_counts = at::empty({tp_world_size}, x.options().dtype(at::kInt));
    at::Tensor assist_info_forcombine{nullptr};

    // a2分层方案
    at::Tensor expand_scales = at::empty({a}, x.options().dtype(at::kFloat));
    if (expert_scales.has_value() && expert_scales.value().defined()) {
        ep_recv_cnt_num = ep_world_size * local_moe_expert_num + 2 * global_bs_real * k * (ep_world_size / 8); // 2: 2 buffer, 8 ranknum per server
        ep_recv_counts = at::empty({ep_recv_cnt_num}, x.options().dtype(at::kInt));
    }

    bool is_fullmesh_v2 = (comm_alg == "fullmesh_v2");

    assist_info_forcombine = at::empty({std::max(bs * k, a * 128)}, x.options().dtype(at::kInt));
    return std::tie(expand_x, dynamic_scales, assist_info_forcombine, expert_token_nums, ep_recv_counts, tp_recv_counts,
        expand_scales);
}

// Register Ascend implementations for MoeDistributeDispatchV2
TORCH_LIBRARY_IMPL(ascend_ops, PrivateUse1, m)
{
    m.impl("MoeDistributeDispatchV2", TORCH_FN(npu_moe_distribute_dispatch_v2));
}

TORCH_LIBRARY_IMPL(ascend_ops, Meta, m)
{
    m.impl("MoeDistributeDispatchV2", &npu_moe_distribute_dispatch_v2_meta);
}

} // namespace MoeDistributeDispatchV2
} // namespace ascend_ops
