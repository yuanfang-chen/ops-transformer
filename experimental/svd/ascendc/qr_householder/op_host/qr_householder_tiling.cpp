/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file qr_householder_tiling.cpp
 * \brief
 */

#include "qr_householder_tiling.h"

namespace optiling {
    struct QrHouseholderInfo {};

    ge::graphStatus QrHouseholderTilingFunc(gert::TilingContext* context) {
        QrHouseholderTilingData tiling;

        auto platformInfo = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        uint64_t ubSize;
        platformInfo.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
        const gert::StorageShape* inputShape = context->GetInputShape(0);
        int32_t dimIdx = (int32_t)inputShape->GetOriginShape().GetDimNum() - 1;
        uint32_t kDim = static_cast<uint32_t>(inputShape->GetStorageShape().GetDim(dimIdx--));
        uint32_t mDim = static_cast<uint32_t>(inputShape->GetStorageShape().GetDim(dimIdx--));
        uint32_t batchSize = 1;
        while(dimIdx >= 0) {
            int32_t dim = static_cast<int32_t>(inputShape->GetStorageShape().GetDim(dimIdx--));
            batchSize *= dim;
        }
        tiling.set_ubSize(static_cast<uint32_t>(ubSize));
        tiling.set_batchSize(batchSize);
        tiling.set_mDim(mDim);
        tiling.set_kDim(kDim);
        context->SetBlockDim(1);
        tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
        context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

        size_t* workspaces = context->GetWorkspaceSizes(1);
        workspaces[0] = 16 * 1024 * 1024 + mDim * kDim * sizeof(float);
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus TilingPrepareForQrHouseholder(gert::TilingParseContext* context) {
        return ge::GRAPH_SUCCESS;
    }

IMPL_OP_OPTILING(QrHouseholder).Tiling(QrHouseholderTilingFunc).TilingParse<QrHouseholderInfo>(TilingPrepareForQrHouseholder);
} // namespace optiling

