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
 * \file mhc_post_tiling.cpp
 * \brief MhcPost tiling implementation
 */

#include "mhc_post_tiling.h"

using namespace ge;

namespace optiling {

ge::graphStatus TilingMhcPost(gert::TilingContext* context)
{
    MhcPostTilingParams param;
    return TilingComputeForMhcPost(context, param);
}

static ge::graphStatus InputParamCheck(const gert::TilingContext* context)
{
    const gert::StorageShape* xShape = context->GetInputShape(0);
    const gert::StorageShape* hResShape = context->GetInputShape(1);
    const gert::StorageShape* hOutShape = context->GetInputShape(2);
    const gert::StorageShape* hPostShape = context->GetInputShape(3);

    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, hResShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, hOutShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, hPostShape);

    auto nodeName = context->GetNodeName();

    // Check x dtype is FP16 or BF16
    auto xDtype = context->GetInputTensor(0)->GetDataType();
    OP_CHECK_IF(xDtype != ge::DT_FLOAT16 && xDtype != ge::DT_BF16,
        OP_LOGE(nodeName, "x dtype must be FLOAT16 or BF16, but got %ld", xDtype),
        return ge::GRAPH_FAILED);

    // Check h_res dtype is FP32
    auto hResDtype = context->GetInputTensor(1)->GetDataType();
    OP_CHECK_IF(hResDtype != ge::DT_FLOAT,
        OP_LOGE(nodeName, "h_res dtype must be FLOAT32, but got %ld", hResDtype),
        return ge::GRAPH_FAILED);

    // Check h_out dtype is same as x
    auto hOutDtype = context->GetInputTensor(2)->GetDataType();
    OP_CHECK_IF(hOutDtype != xDtype,
        OP_LOGE(nodeName, "h_out dtype must be same as x (%ld), but got %ld", xDtype, hOutDtype),
        return ge::GRAPH_FAILED);

    // Check h_post dtype is FP32
    auto hPostDtype = context->GetInputTensor(3)->GetDataType();
    OP_CHECK_IF(hPostDtype != ge::DT_FLOAT,
        OP_LOGE(nodeName, "h_post dtype must be FLOAT32, but got %ld", hPostDtype),
        return ge::GRAPH_FAILED);

    return GRAPH_SUCCESS;
}

ge::graphStatus TilingComputeForMhcPost(gert::TilingContext* context, MhcPostTilingParams& param)
{
    if (InputParamCheck(context) == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }

    // Get platform info
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    param.coreNum = static_cast<int64_t>(ascendcPlatform.GetCoreNumAiv());
    param.maxCoreMemery = ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB);

    // Calculate total length
    const gert::StorageShape* xShape = context->GetInputShape(0);
    param.totalLength = 1;
    for (size_t i = 0; i < xShape->GetStorageShape().GetDimNum(); ++i) {
        param.totalLength *= xShape->GetStorageShape().GetDim(i);
    }

    // Calculate core number for element-wise operations
    constexpr int64_t MIN_LENGTH_PER_CORE = 256;
    int64_t usedCoreNum = 1;

    if (param.totalLength >= MIN_LENGTH_PER_CORE * param.coreNum) {
        usedCoreNum = param.coreNum;
    } else if (param.totalLength >= MIN_LENGTH_PER_CORE) {
        usedCoreNum = param.totalLength / MIN_LENGTH_PER_CORE;
        if (usedCoreNum > param.coreNum) {
            usedCoreNum = param.coreNum;
        }
    }
    if (usedCoreNum < 1) {
        usedCoreNum = 1;
    }

    param.coreNum = usedCoreNum;
    param.singleCoreLength = (param.totalLength + param.coreNum - 1) / param.coreNum;
    param.singleCoreLength = (param.singleCoreLength + ALIGN_256 - 1) / ALIGN_256 * ALIGN_256;

    // Calculate tile length based on UB size
    // We have 5 tensors: x, h_out (T), y (T) and h_res, h_post (FP32)
    // Total: 3 * sizeof(T) + 2 * sizeof(float)
    auto xDtype = context->GetInputTensor(0)->GetDataType();
    int64_t dtypeSize = (xDtype == ge::DT_FLOAT16) ? sizeof(int16_t) : sizeof(int16_t); // BF16 also uses 2 bytes

    int64_t bytesPerElement = 3 * dtypeSize + 2 * FLOAT_DATA_SIZE;
    int64_t usableUbSize = param.maxCoreMemery / MIN_BUFFER_NUM;
    param.tileLength = usableUbSize / bytesPerElement / BLOCK_NUM_FP16 * BLOCK_NUM_FP16;

    if (param.tileLength == 0) {
        param.tileLength = BLOCK_NUM_FP16;
    }
    if (param.tileLength > 8192) {
        param.tileLength = 8192;
    }

    // Set tiling data
    MhcPostTilingData tilingData;
    tilingData.set_total_length(param.totalLength);
    tilingData.set_core_num(param.coreNum);
    tilingData.set_single_core_length(param.singleCoreLength);
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());

    // Set block dim
    context->SetBlockDim(param.coreNum);

    // Set tiling key
    int64_t tilingKey = (xDtype == ge::DT_FLOAT16) ? TILING_KEY_FP16 : TILING_KEY_BF16;
    context->SetTilingKey(tilingKey);

    OP_LOGD(context->GetNodeName(), "MhcPost tiling: totalLength=%ld, coreNum=%ld, singleCoreLength=%ld, tileLength=%ld, tilingKey=%ld",
              param.totalLength, param.coreNum, param.singleCoreLength, param.tileLength, tilingKey);

    return GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepareForMhcPost(gert::TilingParseContext* context)
{
    (void)context;
    return GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MhcPost)
    .Tiling(TilingMhcPost)
    .TilingParse<MhcPostCompileInfo>(TilingPrepareForMhcPost);

} // namespace optiling