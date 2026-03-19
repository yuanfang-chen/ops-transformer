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

__global__ __aicore__ void moe_distribute_dispatch( 
    GM_ADDR x, GM_ADDR expertIds, GM_ADDR scales, GM_ADDR xActiveMask, GM_ADDR expertScales, GM_ADDR expandXOut,
    GM_ADDR dynamicScalesOut, GM_ADDR expandIdxOut, GM_ADDR expertTokenNumsOut, GM_ADDR epSendCountsOut,
    GM_ADDR tpSendCountsOut, GM_ADDR expandScalesOut, GM_ADDR workspaceGM, MoeDistributeDispatchInfo tilingData)
{
    TPipe pipe;
    MoeDistributeDispatch<DTYPE_X, DTYPE_EXPAND_X, false, false, false, false> op;
    op.Init(x, expertIds, scales, expandXOut, dynamicScalesOut, expandIdxOut, expertTokenNumsOut, 
            epSendCountsOut, tpSendCountsOut, workspaceGM, &pipe, &tilingData);
    op.Process();
}

namespace ascendc_ops {
std:tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> ascendc_moe_distribute_dispatch(
    const at::Tensor &x, const at::Tensor &expertIds,
    const c10::optional<at::Tensor> &scales, const c10::optional<at::Tensor> &xActiveMask, const c10::optional<at::Tensor> &expertScales)
{
    constexpr uint32_t WORKSPACESIZE = 16 * 16 * 1024;
    MoeDistributeDispatchInfo tilingData;
    auto output_dtype = (!scales.has_value() && tilingData.quantMode == 0) ? x.scalar_type() : at::kChar;
    uint32_t localMoeExpertNum = tilingData.moeExpertNum / tilingData.epWorldSize;
    at::Tensor expandXOut = at::empty({tilingData.a * tilingData.tpWorldSize, tilingData.h}, x.options().dtype(output_dtype));
    at::Tensor dynamicScalesOut = at::empty({0}, x.options().dtype(at::kFloat));
    at::Tensor expandIdxOut = at::empty({tilingData.bs * tilingData.k}, x.options().dtype(at::kInt));
    at::Tensor expertTokenNumsOut = at::empty({localMoeExpertNum}, x.options().dtype(at::kLong));
    at::Tensor epSendCountsOut = at::empty({tilingData.epWorldSize * localMoeExpertNum * tilingData.tpWorldSize}, x.options().dtype(at::kInt));
    at::Tensor tpSendCountsOut = at::empty({tilingData.tpWorldSize}, x.options().dtype(at::kInt));
    at::Tensor expandScalesOut = at::empty({0}, x.options().dtype(at::kFloat));

    at::Tensor workspaceGM = at::empty({WORKSPACESIZE / 4}, x.options().dtype(at::kInt));

    auto aclStream = c10_npu::getCurrentNPUStream().stream(false);
    uint32_t numBlocks = tilingData.aivNum;
    moe_distribute_dispatch<<<numBlocks, nullptr, aclStream>>>(
        (uint8_t*)(x.mutable_data_ptr()), (uint8_t*)(expertIds.mutable_data_ptr()), 
        (uint8_t*)(scales.mutable_data_ptr()),(uint8_t*)(xActiveMask.mutable_data_ptr()), (uint8_t*)(expertScales.mutable_data_ptr()), 
        (uint8_t*)(expandXOut.mutable_data_ptr()),(uint8_t*)(dynamicScalesOut.mutable_data_ptr()), (uint8_t*)(expandIdxOut.mutable_data_ptr()), 
        (uint8_t*)(expertTokenNumsOut.mutable_data_ptr()), (uint8_t*)(epSendCountsOut.mutable_data_ptr()), (uint8_t*)(tpSendCountsOut.mutable_data_ptr()),
        (uint8_t*)(expandScalesOut.mutable_data_ptr()), (uint8_t*)(workspaceGM.mutable_data_ptr()), tilingData);
    return [expandXOut, dynamicScalesOut, expandIdxOut, expertTokenNumsOut, epSendCountsOut, tpSendCountsOut, expandScalesOut];
}
} // ascendc_moe_distribute_dispatch

PYBIND11_MODULE(ascendc_ops, m)
{
    m.doc() = "moe_distribute_combine_dispatch pybind11 interfaces";
    m.def("ascendc_moe_distribute_dispatch", &ascendc_ops::ascendc_moe_distribute_dispatch, "");
}

