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
 * \file moe_distribute_combine.asc
 * \brief MOE Distribute Combine 算子的 pybind + <<<>>> 直调实现（A5 2卡版本）
 * \details 
 *   本文件将原 aclnn 二段式接口调用方式改造为 pybind + <<<>>> 直调方式
 *   适配平台: A5 (dav-3510)
 *   适配场景: 2卡
 */

#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "kernel_operator.h"
#include "adv_api/reduce/sum.h"
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_tiling/kernel_tiling.h"
#include <cstring>
#include "lib/matmul_intf.h"
#include op_kernel\moe_distribute_combine_tiling.h
#include op_kernel\moe_distribute_combine.h

#define TILINGKEY_TPL_A2 0
#define TILINGKEY_TPL_A3 1
#define TILINGKEY_TPL_A5 2
#define TILINGKEY_NO_QUANT 0
#define TILINGKEY_INT8_QUANT 1
#define TILINGKEY_TPL_MTE 0
#define TILINGKEY_TPL_AICPU 1

MoeDistributeCombineTilingData CalculateTilingA5(
    const std::vector<int64_t>& expandXSize,
    const std::vector<int64_t>& expertIdsSize,
    int64_t epRankId)
{
    MoeDistributeCombineTilingData tiling = {};
    const uint32_t epWorldSize = 2;
    const uint32_t tpWorldSize = 1;

    tiling.moeDistributeCombineInfo.bs = static_cast<uint32_t>(expertIdsSize[0]);
    tiling.moeDistributeCombineInfo.h = static_cast<uint32_t>(expandXSize[1]);
    tiling.moeDistributeCombineInfo.k = static_cast<uint32_t>(expertIdsSize[1]);
    tiling.moeDistributeCombineInfo.globalBs = tiling.moeDistributeCombineInfo.bs * epWorldSize;
    
    tiling.moeDistributeCombineInfo.aivNum = 16;
    tiling.moeDistributeCombineInfo.totalUbSize = 1024 * 1024;
    
    tiling.moeDistributeCombineInfo.epWorldSize = epWorldSize;
    tiling.moeDistributeCombineInfo.epRankId = static_cast<uint32_t>(epRankId);
    tiling.moeDistributeCombineInfo.moeExpertNum = 4;
    tiling.moeDistributeCombineInfo.moeExpertPerRankNum = 2;
    
    tiling.moeDistributeCombineInfo.tpWorldSize = tpWorldSize;
    tiling.moeDistributeCombineInfo.tpRankId = 0;
    
    tiling.moeDistributeCombineInfo.expertShardType = 0;
    tiling.moeDistributeCombineInfo.sharedExpertRankNum = 0;
    
    tiling.moeDistributeCombineInfo.totalWinSizeEp = 
        static_cast<uint64_t>(tiling.moeDistributeCombineInfo.globalBs) * 
        tiling.moeDistributeCombineInfo.h * 2;
    tiling.moeDistributeCombineInfo.totalWinSizeTp = 
        static_cast<uint64_t>(expandXSize[0]) * tiling.moeDistributeCombineInfo.h * 2;
    
    tiling.moeDistributeCombineInfo.a = 0;
    
    return tiling;

}


// 核函数实现
template<typename DTYPE_EXPAND_X, bool HasTp, uint8_t QuantMode, uint8_t LayeredMode, uint8_t ArchTag>
__global__ __aicore__ void moe_distribute_combine(
    GM_ADDR expandX, GM_ADDR expertIds, GM_ADDR expandIdx, GM_ADDR epSendCount, 
    GM_ADDR scales, GM_ADDR tpSendCount,GM_ADDR xActiveMask, GM_ADDR activationScale,
    GM_ADDR weightScale, GM_ADDR groupList,GM_ADDR expandScales, GM_ADDR XOut,
    GM_ADDR workspaceGM, MoeDistributeCombineTilingData tilingData)
{
    TPipe pipe;
    if constexpr (ArchTag == TILINGKEY_TPL_A5) {
        MoeDistributeCombine<DTYPE_EXPAND_X, int32_t, HasTp, QuantMode == TILINGKEY_INT8_QUANT> op;
        op.Init(expandX, expertIds, expandIdx, epSendCount, tpSendCount, scales, XOut, workspaceGM, &pipe, tilingData);
        op.Process();
    }
}


namespace ascendc_ops {
at::Tensor ascendc_moe_distribute_combine( 
    const at::Tensor& expandX, const at::Tensor& expertIds, const at::Tensor& expandIdx, 
    const at::Tensor& epSendCount, const at::Tensor& scales, int64_t ep_rank_id
)
    auto aclStream = c10_npu::getCurrentNPUStream().stream(false);

    MoeDistributeCombineTilingData tilingData = CalculateTilingA5(
        expand_x_size.vec(), expert_ids_size.vec(), ep_rank_id);

    uint32_t num_blocks = tiling_data.moeDistributeCombineInfo.aivNum;
    moe_distribute_combine<false, TILINGKEY_NO_QUANT, TILINGKEY_TPL_MTE, TILINGKEY_TPL_A5><<<num_blocks, nullptr, acl_stream>>>(
        (uint8_t*)(expandX.mutable_data_ptr()), (uint8_t*)(expertIds.mutable_data_ptr()), (uint8_t*)(expandIdx.mutable_data_ptr()),
        (uint8_t*)(epSendCount.mutable_data_ptr()), (uint8_t*)(scales.mutable_data_ptr()), 
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        (uint8_t*)(XOut.mutable_data_ptr()), (uint8_t*)(workspaceGM.mutable_data_ptr()), (uint8_t*)(tilingGM.mutable_data_ptr())

    )

    return z;
} // namespace ascendc_ops













// extern "C" __global__ __aicore__ void  moe_distribute_combine_generic(
//     int32_t tilingKey,
//     GM_ADDR expandX, GM_ADDR expertIds, GM_ADDR expandIdx, GM_ADDR epSendCount, 
//     GM_ADDR scales, GM_ADDR tpSendCount,GM_ADDR xActiveMask, GM_ADDR activationScale,
//     GM_ADDR weightScale, GM_ADDR groupList,GM_ADDR expandScales, GM_ADDR XOut,
//     GM_ADDR workspaceGM, MoeDistributeCombineTilingData tilingData)
// {
//     switch (tilingKey) {
//         case 000:
//         moe_distribute_combine<float16_t, false, TILINGKEY_NO_QUANT, TILINGKEY_TPL_MTE, TILINGKEY_TPL_A5>(
//             expandX, expertIds, expandIdx, epSendCount, scales, tpSendCount, xActiveMask, activationScale,
//             weightScale, groupList, expandScales, XOut, workspaceGM,  tilingData);
//         break;
//     }
// }


// // <<<>>>调用函数
// void moe_distribute_combine_v2_entry(int32_t tilingKey, uint32_t blockDim, void* stream,
//     GM_ADDR expandX, GM_ADDR expertIds, GM_ADDR expandIdx, GM_ADDR epSendCount, 
//     GM_ADDR scales, GM_ADDR tpSendCount,GM_ADDR xActiveMask, GM_ADDR activationScale,
//     GM_ADDR weightScale, GM_ADDR groupList,GM_ADDR expandScales, GM_ADDR XOut,
//     GM_ADDR workspaceGM, MoeDistributeCombineTilingData tilingData)
// {
//     moe_distribute_combine_generic<<<blockDim, nullptr, stream>>>(
//         tilingKey,
//         expandX, expertIds, expandIdx, epSendCount, scales, tpSendCount, xActiveMask, activationScale,
//         weightScale, groupList, expandScales, XOut, workspaceGM, tilingData);
// }






PYBIND11_MODULE(ascendc_ops, m)
{
    m.doc() = "moe_distribute_combine_custom pybind11 interfaces";
    m.def("ascendc_moe_distribute_combine", &ascendc_ops::ascendc_moe_distribute_combine, "");
}