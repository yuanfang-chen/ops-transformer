/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/


#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"

#include "acl/acl.h"
#include "tiling/platform/platform_ascendc.h"
#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"

#include <iostream>
#include <cstdlib>
#include <memory>
#include <cstdint>

#define FIA_ENABLE_MLA
#include "common_utils.h"
#include "io_utils.h"
#include "flash_attention_score_tiling_regbase.h"
#include "fia_entry.h"
#include "op_host/abc.h"
#include "op_host/fused_infer_attention_score_tiling.h"
#include "op_host/fused_infer_attention_score_tiling_constants.h"

__global__ __aicore__ void FiaKernelFullQuant(
        GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR keyAntiquantScale,
        GM_ADDR valueAntiquantScale, GM_ADDR dequantScaleQuery, GM_ADDR attentionOut,
        GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    FlashAttentionEntry(
        query, key, value,
        keyAntiquantScale, valueAntiquantScale, dequantScaleQuery,  attentionOut,
        workspace, tiling);
    return;
}

namespace ascendc_ops {
optiling::TilingContext InitContext(std::vector<int64_t> shapeQueryTensor, std::vector<int64_t> shapeKeyTensor, std::vector<int64_t> shapeValueTensor) {
    optiling::TilingContext context;
    context.SetInputDesc(optiling::QUERY_INDEX, optiling::DT_FLOAT16);
    // 使用便捷方法
    context.SetInputShapeFromVector(optiling::QUERY_INDEX, shapeQueryTensor, optiling::DT_FLOAT16);
    context.SetInputShapeFromVector(optiling::KEY_INDEX, shapeKeyTensor, optiling::DT_FLOAT16);
    context.SetInputShapeFromVector(optiling::VALUE_INDEX, shapeValueTensor, optiling::DT_FLOAT16);
    context.SetInputShapeFromVector(optiling::ATTENTION_OUT_INDEX, shapeQueryTensor, optiling::DT_FLOAT16);
    auto& attrs = context.GetAttrs();
    attrs.SetAttr(ATTR_N_INDEX, (uint32_t)1);
    attrs.SetAttr(ATTR_SCALE_INDEX, 3.14f);
    attrs.SetAttr(ATTR_INPUT_LAYOUT_INDEX, "string");
    attrs.SetAttr(ATTR_NUM_KV_HEADS_INDEX, (uint32_t)1);
    attrs.SetAttr(ATTR_BLOCK_SIZE_INDEX, (uint32_t)1);
    attrs.SetAttr(ANTIQUANT_MODE_INDEX, (int64_t)1);
    attrs.SetAttr(SOFTMAX_LSE_FLAG_INDEX, true);
    attrs.SetAttr(KEY_ANTIQUANT_MODE_INDEX, (int64_t)1);
    attrs.SetAttr(VALUE_ANTIQUANT_MODE_INDEX, (int64_t)1);
    attrs.SetAttr(ATTR_INNER_PRECISE_INDEX, (uint32_t)1);
    attrs.SetAttr(ATTR_SPARSE_MODE_INDEX, (uint32_t)1);
    attrs.SetAttr(QUERY_QUANT_MODE_INDEX, (int64_t)1);
    attrs.SetAttr(ATTR_PRE_TOKEN_INDEX, (int64_t)1);
    printf(".isEmpty(): %d\n", attrs.isEmpty());
    return context;
}

static std::vector<int64_t> getTensorShape(const at::Tensor &tensor)
{
    std::vector<int64_t> shape;
    try {
        shape = std::vector<int64_t>(tensor.sizes().begin(), tensor.sizes().end());
    } catch (const std::exception &e) {
        std::cerr << "Error getting tensor shpae: " << e.what() << std::endl;
        shape = {0};
    }
    return shape;
}

at::Tensor ascendc_fia(const at::Tensor& queryTensor, const at::Tensor& keyTensor, const at::Tensor& valueTensor,
    const at::Tensor& keyAntiquantScaleTensor, const at::Tensor& valueAntiquantScaleTensor, const at::Tensor& queryQuantScaleTensor)
{
    std::cerr << "Start fused_infer_attention_score demo." << std::endl;
    std::vector<int64_t> shapeQueryTensor = getTensorShape(queryTensor);
    std::vector<int64_t> shapeKeyTensor = getTensorShape(keyTensor);
    std::vector<int64_t> shapeValueTensor = getTensorShape(valueTensor);
    std::vector<int64_t> shapeKeyAntiquantScale = getTensorShape(keyAntiquantScaleTensor);
    std::vector<int64_t> shapeValueAntiquantScale = getTensorShape(valueAntiquantScaleTensor);
    std::vector<int64_t> shapeQueryQuantScaler = getTensorShape(queryQuantScaleTensor);
    std::cerr << "dtype: " << queryTensor.dtype() << " shape: " << shapeQueryTensor << std::endl;
    at::Tensor outputTensor = at::empty({shapeQueryTensor[0], shapeQueryTensor[1], shapeQueryTensor[2], shapeQueryTensor[3] * 2}, at::dtype(queryTensor.dtype()).device(queryTensor.device()).layout(queryTensor.layout()));
    // -------------------------------------------------------------------------
    // 1. Set the problem shape.
    // -------------------------------------------------------------------------
    uint32_t batchSize = shapeQueryTensor[0];
    uint32_t numHeadsQ = shapeQueryTensor[1];
    uint32_t numHeadsKV = shapeKeyTensor[1];
    uint64_t seqLengthsQ = shapeQueryTensor[2];
    uint64_t seqLengthsKV = shapeKeyTensor[2];
    uint32_t headDim = shapeQueryTensor[3];
    // 拦截
    optiling::TilingContext context = InitContext(shapeQueryTensor, shapeKeyTensor, shapeValueTensor);
    optiling::DoOpTilingFusedInferAttentionScore(&context);

    // -------------------------------------------------------------------------
    // 2. Initialize the ACL runtime and create a stream.
    //
    // The sample assumes one device and one stream for clarity. More advanced
    // applications may use multiple streams or pre-created runtime contexts.
    // -------------------------------------------------------------------------
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    uint32_t deviceCount;
    CHECK_COND(aclrtGetDeviceCount(&deviceCount) == ACL_SUCCESS, "Failed to get ACLRT devices.");
    CHECK_COND(deviceCount > 0U, "No ACLRT devices found.");
    // std::cerr << "error no." << aclInit(nullptr) << std::endl;
    // CHECK_COND(aclInit(nullptr) == ACL_SUCCESS, "aclInit failed.");
    CHECK_COND(aclrtSetDevice(deviceId) == ACL_SUCCESS, "aclrtSetDevice failed.");
    CHECK_COND(aclrtCreateStream(&stream) == ACL_SUCCESS, "aclrtCreateStream failed.");


    // -------------------------------------------------------------------------
    // 3. Declare host-side buffers.
    //
    // Host buffers hold the packed input tensors loaded from disk and receive
    // the output tensor copied back from device memory.
    // -------------------------------------------------------------------------
    uint8_t* outputHost = nullptr;

    // -------------------------------------------------------------------------
    // 4. Declare device-side buffers.
    //
    // Device buffers mirror the host buffers.
    // `GM_ADDR` is the generic "global memory address" type used by the kernel
    // launch interface in Ascend C samples.
    // -------------------------------------------------------------------------
    GM_ADDR outputDevice = nullptr;
    GM_ADDR workspace = nullptr;
    GM_ADDR tilingDataDevice = nullptr;

    // -------------------------------------------------------------------------
    // 4. Compute tensor sizes in bytes.
    // -------------------------------------------------------------------------
    //
    // query:
    //   float8_e4m3fn input tensor of shape [batchSize, numHeadsQ, seqLengthsQ, headDim].
    //
    // key and value:
    //   float8_e4m3fn input tensor of shape [batchSize, numHeadsKV, seqLengthsKV, headDim].
    //
    // dequantScaleQuery:
    //   float32 input tensor shape [batchSize, numHeadsQ, seqLengthsQ / 128, 1].
    //
    // keyAntiquantScale and valueAntiquantScale:
    //   float32 input tensor shape [batchSize, numHeadsKV, seqLengthsKV / 256, 1].
    size_t outputSize = (batchSize * numHeadsQ * seqLengthsQ * headDim) * sizeof(uint16_t);
    size_t workspaceSize = 200 * 2048 * 1024 * sizeof(uint8_t);
    size_t tilingDataSize = sizeof(optiling::FlashAttentionScoreSimplifiedTilingData);

    // -------------------------------------------------------------------------
    // 4. Materialize the default tiling configuration for this problem shape.
    // -------------------------------------------------------------------------

    // Query the platform object to learn how many AIC cores are available.
    // This sample launches one block per AIC core and lets the scheduler assign
    // multiple tiles to each block when the problem is larger than the machine.
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    CHECK_COND(ascendcPlatform != nullptr, "Get ascendcPlatform failed.");
    uint32_t blockDimToBeSet = ascendcPlatform->CalcTschBlockDim(ascendcPlatform->GetCoreNumAiv(),
                    ascendcPlatform->GetCoreNumAic(), ascendcPlatform->GetCoreNumAiv());

    optiling::FlashAttentionScoreSimplifiedTilingData tilingData;
    if (ascendcPlatform->GetCoreNumAic() == 32) {
        std::cerr << "CoreNum is 32." << std::endl;
        SetTilingData(tilingData, context);
    } else if (ascendcPlatform->GetCoreNumAic() == 28) {
        SetTilingDataLess(tilingData);
        std::cerr << "CoreNum is 28." << std::endl;
    } else {
        CHECK_COND(false, "CoreNum only support 28 0r 32");
    }

    // -------------------------------------------------------------------------
    // 5. Allocate pinned host memory.
    //
    // Pinned buffers are used because they are the typical choice for explicit
    // async H2D / D2H copies in standalone performance samples.
    // -------------------------------------------------------------------------
    CHECK_COND(aclrtMallocHost((void**)&outputHost, outputSize) == ACL_SUCCESS, "aclrtMallocHost failed.");
    std::unique_ptr<void, aclError (*)(void*)> HostO(outputHost, aclrtFreeHost);


    // -------------------------------------------------------------------------
    // 6. Allocate global memory on device.
    // -------------------------------------------------------------------------
    CHECK_COND(aclrtMalloc((void**)&outputDevice, outputSize, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS, "aclrtMalloc failed.");
    std::unique_ptr<void, aclError (*)(void*)> DeviceO(outputDevice, aclrtFree);
    CHECK_COND(aclrtMalloc((void**)&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS, "aclrtMalloc failed.");
    std::unique_ptr<void, aclError (*)(void*)> DeviceWS(workspace, aclrtFree);
    CHECK_COND(aclrtMalloc((void**)&tilingDataDevice, tilingDataSize, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS, "aclrtMalloc failed.");
    std::unique_ptr<void, aclError (*)(void*)> DeviceTD(tilingDataDevice, aclrtFree);

    // -------------------------------------------------------------------------
    // 7. Copy host inputs to device memory.
    //
    // These copies are queued on the same stream that will later launch the
    // kernel, which preserves execution order without extra synchronization.
    // -------------------------------------------------------------------------

    CHECK_COND(
        aclrtMemcpyAsync(tilingDataDevice, tilingDataSize, &tilingData, tilingDataSize, ACL_MEMCPY_HOST_TO_DEVICE, stream) == ACL_SUCCESS,
        "aclrtMemcpyAsync failed.");

    // -------------------------------------------------------------------------
    // 8. Launch the kernel.
    //
    // The kernel itself is small because most of the interesting logic is
    // encoded in the template stack and the runtime tiling parameters.
    // -------------------------------------------------------------------------
    constexpr uint8_t inOutLayoutType = 0;
    constexpr bool hasAttenMask = false;
    IncreFlashAttentionContext ifaContext;
    ConvertContextToParamsIFA(context, ifaContext);
    uint64_t tilingKey;
    CalcTilingKey(tilingKey, context);
    if (tilingKey == 2000000012) {
        FiaKernelFullQuant<<<blockDimToBeSet, nullptr, stream>>>(
            (uint8_t*)(queryTensor.mutable_data_ptr()),
            (uint8_t*)(keyTensor.mutable_data_ptr()),
            (uint8_t*)(valueTensor.mutable_data_ptr()),
            (uint8_t*)(keyAntiquantScaleTensor.mutable_data_ptr()),
            (uint8_t*)(valueAntiquantScaleTensor.mutable_data_ptr()),
            (uint8_t*)(queryQuantScaleTensor.mutable_data_ptr()),
            (uint8_t*)(outputTensor.mutable_data_ptr()),
            workspace,
            tilingDataDevice
        );
    }
    
    // Queue the output copy after the kernel launch on the same stream.
    CHECK_COND(
        aclrtMemcpyAsync(outputHost, outputSize, outputDevice, outputSize, ACL_MEMCPY_DEVICE_TO_HOST, stream) == ACL_SUCCESS,
        "aclrtMemcpyAsync failed.");

    // -------------------------------------------------------------------------
    // 9. Synchronize, dump the output, and tear everything down.
    // -------------------------------------------------------------------------
    CHECK_COND(aclrtSynchronizeStream(stream) == ACL_SUCCESS, "aclrtSynchronizeStream failed.");
    WriteFile("./output/npu_out.bin", outputHost, outputSize);

    // `unique_ptr` takes care of freeing host/device buffers.
    // The runtime objects still need explicit destruction/finalization.
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    // aclFinalize();
    return outputTensor;
}
} // namespace ascendc_ops

PYBIND11_MODULE(ascendc_ops, m)
{
    m.doc() = "ascendc_fia pybind11 interfaces";
    m.def("ascendc_fia", &ascendc_ops::ascendc_fia, "");
}
