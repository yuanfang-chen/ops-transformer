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

/*!
 * \file moe_distribute_dispatch_torch.cpp
 * \brief
 */



#include "hccl/hccl.h"
#include "acl/acl.h"
#include "securec.h"
#include "tiling/hccl/hccl_tiling.h"

#include <ATen/ATen.h>
#include <vector>
#include <torch/all.h>
#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"
#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "kernel_operator.h"
#include "op_kernel/moe_distribute_dispatch.h"
#include "op_kernel/moe_distribute_dispatch_tiling.h"
#include "moe_distribute_dispatch_torch.h"

using namespace MoeDistributeDispatchImpl;
using namespace AscendC;

constexpr uint64_t MTE_STATE_ZONE_SIZE = 1024UL * 1024UL;
constexpr uint32_t WORKSPACESIZE = 16 * 16 * 1024;
constexpr uint32_t OP_TYPE_ALL_TO_ALL = 8U;

__global__ __aicore__ void moe_distribute_dispatch_custom( 
    GM_ADDR x, GM_ADDR expertIds, GM_ADDR scales, 
    GM_ADDR expandXOut, GM_ADDR dynamicScalesOut, GM_ADDR expandIdxOut, 
    GM_ADDR expertTokenNumsOut, GM_ADDR epSendCountsOut, GM_ADDR tpSendCountsOut,
    GM_ADDR workspaceGM, MoeDistributeDispatchTilingData tilingData)
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
    MoeDistributeDispatchTilingData tilingData;
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
    uint32_t a;
    if (epRankId < sharedExpertRankNum) { // 本卡为共享专家
        a = tilingData.globalBs / sharedExpertRankNum;
    } else {     // 本卡为moe专家
        a = tilingData.globalBs * std::min(localMoeExpertNum, tilingData.k);
    }  
    tilingData.a = a;                                              
    tilingData.aivNum = 56;                   
    tilingData.isQuant = (quantMode != 0)? 1 : 0;                        
    tilingData.reserved1 = false;                      
    tilingData.reserved2 = false;                      
    tilingData.reserved3 = false;                      
    tilingData.totalUbSize = 253952; // 暂时不知道如何取消context
    tilingData.totalWinSizeEp = 11000;
    // tilingData.totalWinSizeEp = mc2tiling::Mc2TilingUtils::GetMaxWindowSize() - MTE_STATE_ZONE_SIZE;              
    tilingData.totalWinSizeTp = 0;

    // uint32_t numBlocks = 1U;
    // auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    // uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    // uint64_t ubSize = 0UL;
    // ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    // numBlocks = ascendcPlatform.CalcTschBlockDim(aivNum, 0, aivNum);
    // context->SetBlockDim(numBlocks);   
    // tilingData->MoeDistributeDispatchTilingData.totalUbSize = ubSize;
    // tilingData->MoeDistributeDispatchTilingData.aivNum = aivNum;   

    auto output_dtype = (!scales.has_value() && quantMode == 0) ? x.scalar_type() : at::kChar;
    at::Tensor expandXOut = at::empty({a, h}, x.options().dtype(output_dtype));
    at::Tensor dynamicScalesOut = at::empty({a}, x.options().dtype(at::kFloat));
    at::Tensor expandIdxOut = at::empty({bs * k}, x.options().dtype(at::kInt));
    at::Tensor expertTokenNumsOut = at::empty({moeExpertNum}, x.options().dtype(at::kLong));
    at::Tensor epSendCountsOut = at::empty({epWorldSize}, x.options().dtype(at::kInt));
    at::Tensor tpSendCountsOut = at::empty({tpWorldSize}, x.options().dtype(at::kInt));

    void* workspacePtr = nullptr;
    at::Tensor workspaceGM = at::empty({WORKSPACESIZE / 4}, x.options().dtype(at::kInt));
    aclError ret = aclrtMalloc(&workspacePtr, WORKSPACESIZE, ACL_MEM_MALLOC_HUGE_FIRST);

    Mc2InitTiling mc2InitTiling; 
    Mc2CcTiling mc2CcTiling1;
    std::string groupEp = "hccl_comm_ep";
    std::string algConfigAllToAllStr = "AlltoAll=level0:fullmesh;level1:pairwise";
    AscendC::Mc2CcTilingConfig mc2CcTilingConfig("groupEp", OP_TYPE_ALL_TO_ALL, algConfigAllToAllStr);
    mc2CcTilingConfig.SetCommEngine(3);
    uint32_t ret1 = mc2CcTilingConfig.GetTiling(mc2InitTiling);
    uint32_t ret2 = mc2CcTilingConfig.GetTiling(mc2CcTiling1);
    
    
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