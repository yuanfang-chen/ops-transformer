/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under
 * the terms and conditions of CANN Open Software License Agreement Version 2.0
 * (the "License"). Please refer to the License for details. You may not use
 * this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY
 * KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
 * NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the
 * License.
 */


#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "kernel_operator.h"
#include "op_kernel/moe_distribute_dispatch.h"
#include "op_kernel/moe_distribute_dispatch_tiling.h"

// template<uint32_t quantAllReduceCommMode, typename DTYPE_X, typename DTYPE_SCALES, typename DTYPE_OUT_PUT>
// template<typename DTYPE_X, typename DTYPE_EXPAND_X, bool StaticQuant, bool DynamicQuant, bool IsSmoothScaleExist, bool IsNeedAllgather>
// __attribute__((always_inline)) __aicore__ __inline__ void moe_distribute_dispatch_with_mc2context(
//     GM_ADDR x, GM_ADDR expertIds, GM_ADDR scales,
//     GM_ADDR expandXOut, GM_ADDR dynamicScalesOut, GM_ADDR expandIdxOut, 
//     GM_ADDR expertTokenNumsOut, GM_ADDR epSendCountsOut, GM_ADDR tpSendCountsOut,  
//     GM_ADDR workspaceGM, GM_ADDR mc2Context, GM_ADDR tilingGM)
// {
//     KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
//     TPipe pipe;
//     MoeDistributeDispatch<DTYPE_X, DTYPE_EXPAND_X, StaticQuant, DynamicQuant, IsSmoothScaleExist, IsNeedAllgather> op;
//     op.InitWithMc2Context(
//         x, expertIds, scales, 
//         expandXOut, dynamicScalesOut, expandIdxOut, 
//         expertTokenNumsOut, epSendCountsOut, tpSendCountsOut, 
//         workspaceGM, mc2Context, tilingGM, &pipe);
//     op.Process();
    
// }

// // 核函数入口实现
// extern "C" __global__ __aicore__ void moe_distribute_dispatch_generic(
//     uint32_t tilingkey,
//     GM_ADDR x, GM_ADDR expertIds, GM_ADDR scales,
//     GM_ADDR expandXOut, GM_ADDR dynamicScalesOut, GM_ADDR expandIdxOut, 
//     GM_ADDR expertTokenNumsOut, GM_ADDR epSendCountsOut, GM_ADDR tpSendCountsOut, 
//     GM_ADDR workspaceGM, GM_ADDR mc2Context, GM_ADDR tilingGM)
// {
//     // 根据不同的数据类型调用不同的模板
//     switch (tilingkey) {
//         case 0: 
//             moe_distribute_dispatch_with_mc2context<float16_t, float16_t, false, false, false, false>(
//                 x, expertIds, scales, 
//                 expandXOut, dynamicScalesOut, expandIdxOut, 
//                 expertTokenNumsOut, epSendCountsOut, tpSendCountsOut, 
//                 workspaceGM, mc2Context, tilingGM);
//             break;
//         default:
//             AscendC::PRINTF("QuantAllReduce Error: invalid type = %d\n", type);
//             return;
//     }
// }

// // <<<>>>调用函数
// void moe_distribute_dispatch_demo(
//     uint32_t tilingkey, uint32_t blockDim, void* stream, 
//     uint8_t* x, uint8_t* expertIds, uint8_t* scales, 
//     uint8_t* expandXOut, uint8_t* dynamicScalesOut, uint8_t* expandIdxOut, uint8_t* expertTokenNumsOut, 
//     uint8_t* epSendCountsOut, uint8_t* tpSendCountsOut, uint8_t* expandScalesOut,
//     uint8_t* workspaceGM, uint8_t* mc2Context, uint8_t* tilingGM) {
//     moe_distribute_dispatch_generic<<<blockDim, nullptr, stream>>>(
//         tilingkey, 
//         x, expertIds, scales
//         expandXOut, dynamicScalesOut, expandIdxOut, expertTokenNumsOut, epSendCountsOut, tpSendCountsOut,
//         workspaceGM, mc2Context, tilingGM);
// }



constexpr uint64_t MTE_STATE_ZONE_SIZE = 1024UL * 1024UL;
constexpr uint32_t WORKSPACESIZE = 16 * 16 * 1024;
constexpr uint32_t OP_TYPE_ALL_TO_ALL = 8U;

__global__ __aicore__ void moe_distribute_dispatch_custom( 
    GM_ADDR x, GM_ADDR expertIds, GM_ADDR scales, 
    GM_ADDR expandXOut, GM_ADDR dynamicScalesOut, GM_ADDR expandIdxOut, 
    GM_ADDR expertTokenNumsOut, GM_ADDR epSendCountsOut, GM_ADDR tpSendCountsOut,
    GM_ADDR workspaceGM, MoeDistributeDispatchInfo tilingData)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    MoeDistributeDispatch<float16_t, float16_t, false, false, false, false> op;
    op.Init(
        x, expertIds, scales, 
        expandXOut, dynamicScalesOut, expandIdxOut, expertTokenNumsOut, epSendCountsOut, tpSendCountsOut, 
        workspaceGM, &pipe, &tilingData);
    op.Process();
}

namespace ascendc_ops {
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> ascendc_moe_distribute_dispatch(
    const at::Tensor &x, const at::Tensor &expertIds, const c10::optional<at::Tensor> &scales,
    uint32_t epWorldSize, uint32_t tpWorldSize, uint32_t epRankId, uint32_t tpRankId, uint32_t expertShardType,
    uint32_t sharedExpertRankNum, uint32_t moeExpertNum, uint32_t quantMode, uint32_t expertTokenNumsType
    )
{
    
    // 设置tiling结构体信息
    MoeDistributeDispatchInfo tilingData;
    tilingData.epWorldSize = epWorldSize;                
    tilingData.tpWorldSize = tpWorldSize;                
    tilingData.epRankId = epRankId;                   
    tilingData.tpRankId = tpRankId;                   
    tilingData.expertShardType = expertShardType;            
    tilingData.sharedExpertRankNum = sharedExpertRankNum;        
    tilingData.moeExpertNum = moeExpertNum;               
    tilingData.quantMode = quantMode;       
    tilingData.expertTokenNumsType = expertTokenNumsType; 
    uint32_t bs = x.size(0);    
    uint32_t h = x.size(1);    
    uint32_t k = expertIds.size(1);             
    tilingData.globalBs = bs * epWorldSize;
    tilingData.bs = bs;                        
    tilingData.h = h;                          
    tilingData.k = k; 
    uint32_t localMoeExpertNum = moeExpertNum / (epWorldSize - sharedExpertRankNum);
    if (epRankId < sharedExpertRankNum) { // 本卡为共享专家
        uint32_t a = tilingData.globalBs / sharedExpertRankNum;
    } else {     // 本卡为moe专家
        uint32_t a = tilingData.globalBs * std::min(localMoeExpertNum, tilingData.k);
    }  
    tilingData.a = a;                                              
    tilingData.aivNum = 56;                   
    tilingData.isQuant = (quantMode != 0)? 1 : 0;                        
    tilingData.reserved1 = false;                      
    tilingData.reserved2 = false;                      
    tilingData.reserved3 = false;                      
    tilingData.totalUbSize = 253952; // 暂时不知道如何取消context
    tilingData.totalWinSizeEp = mc2tiling::Mc2TilingUtils::GetMaxWindowSize() - MTE_STATE_ZONE_SIZE;              
    tilingData.totalWinSizeTp = 0;

    // uint32_t numBlocks = 1U;
    // auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    // uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    // uint64_t ubSize = 0UL;
    // ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    // numBlocks = ascendcPlatform.CalcTschBlockDim(aivNum, 0, aivNum);
    // context->SetBlockDim(numBlocks);   
    // tilingData->moeDistributeDispatchInfo.totalUbSize = ubSize;
    // tilingData->moeDistributeDispatchInfo.aivNum = aivNum;   

    auto output_dtype = (!scales.has_value() && quantMode == 0) ? x.scalar_type() : at::kChar;
    at::Tensor expandXOut = at::empty({a, h}, x.options().dtype(output_dtype));
    at::Tensor dynamicScalesOut = at::empty({a}, x.options().dtype(at::kFloat));
    at::Tensor expandIdxOut = at::empty({bs * k}, x.options().dtype(at::kInt));
    at::Tensor expertTokenNumsOut = at::empty({moeExpertNum}, x.options().dtype(at::kLong));
    at::Tensor epSendCountsOut = at::empty({epWorldSize}, x.options().dtype(at::kInt));
    at::Tensor tpSendCountsOut = at::empty({tpWorldSize}, x.options().dtype(at::kInt));

    void* workspacePtr = nullptr;
    at::Tensor workspaceGM = at::empty({WORKSPACESIZE / 4}, x.options().dtype(at::kInt));
    aclError ret = aclrtMalloc(&workspacePtr, workspaceGM, ACL_MEM_MALLOC_HUGE_FIRST);

    Mc2InitTiling mc2InitTiling; 
    Mc2CcTiling mc2CcTiling1;
    std::string groupEp = "hccl_comm_ep";
    std::string algConfigAllToAllStr = "AlltoAll=level0:fullmesh;level1:pairwise";
    AscendC::Mc2CcTilingConfig mc2CcTilingConfig("groupEp", OP_TYPE_ALL_TO_ALL, algConfigAllToAllStr);
    mc2CcTilingConfig.SetCommEngine(mc2tiling::AIV_ENGINE);
    int ret = mc2CcTilingConfig.GetTiling(tiling->mc2InitTiling);
    ret = mc2CcTilingConfig.GetTiling(tiling->mc2CcTiling1);
    
    
    auto x_ptr = get_first_tensor_address<at::Tensor>(x.scalar_type(), x, false);
    auto expertIds_ptr = get_first_tensor_address<at::Tensor>(expertIds.scalar_type(), expertIds, false);
    auto workspace_ptr = get_first_tensor_address<at::Tensor>(workspaceGM.scalar_type(), workspaceGM, false);
    void* scales_ptr = nullptr; 
    if (scales.has_value()) {
        scales_ptr = get_first_tensor_address<c10::optional<at::Tensor>>(scales->scalar_type(), scales, true);
    }
    auto expandXOut_ptr = get_first_tensor_address<at::Tensor>(expandXOut.scalar_type(), expandXOut, false);
    auto dynamicScalesOut_ptr = get_first_tensor_address<at::Tensor>(dynamicScalesOut.scalar_type(), dynamicScalesOut, false);
    auto expandIdxOut_ptr = get_first_tensor_address<at::Tensor>(expandIdxOut.scalar_type(), expandIdxOut, false);
    auto expertTokenNumsOut_ptr = get_first_tensor_address<at::Tensor>(expertTokenNumsOut.scalar_type(), expertTokenNumsOut, false);
    auto epSendCountsOut_ptr = get_first_tensor_address<at::Tensor>(epSendCountsOut.scalar_type(), epSendCountsOut, false);
    auto tpSendCountsOut_ptr = get_first_tensor_address<at::Tensor>(tpSendCountsOut.scalar_type(), tpSendCountsOut, false);
    
    auto aclStream = c10_npu::getCurrentNPUStream().stream(false);
    uint32_t numBlocks = tilingData.aivNum;
    moe_distribute_dispatch_custom<<<numBlocks, nullptr, aclStream>>>(
        (GM_ADDR)x_ptr, (GM_ADDR)expertIds_ptr, (GM_ADDR)scales_ptr,
        (GM_ADDR)expandXOut_ptr, (GM_ADDR)dynamicScalesOut_ptr, (GM_ADDR)expandIdxOut_ptr,
        (GM_ADDR)expertTokenNumsOut_ptr, (GM_ADDR)epSendCountsOut_ptr, (GM_ADDR)tpSendCountsOut_ptr,
        (GM_ADDR)workspace_ptr, tilingData);

    return std::make_tuple(expandXOut, dynamicScalesOut, expandIdxOut, expertTokenNumsOut, epSendCountsOut,tpSendCountsOut);
}
} // namespace ascendc_ops

PYBIND11_MODULE(ascendc_ops, m)
{
    m.doc() = "moe_distribute_combine_dispatch pybind11 interfaces";
    m.def("ascendc_moe_distribute_dispatch", &ascendc_ops::ascendc_moe_distribute_dispatch,
        py::arg("x"),
        py::arg("expertIds"),
        py::arg("scales") = py::none(),
        py::arg("epWorldSize"),
        py::arg("tpWorldSize"),
        py::arg("epRankId"),
        py::arg("tpRankId"),
        py::arg("expertShardType"),
        py::arg("sharedExpertRankNum"),
        py::arg("moeExpertNum"),
        py::arg("quantMode"),
        py::arg("expertTokenNumsType")
    );
}