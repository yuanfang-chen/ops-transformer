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
 * \file gmm_a2a_int8_tiling.cpp
 * \brief INT8 tiling implementation for allto_allv_grouped_mat_mul on ascend950
 */
#include "gmm_a2a_int8_tiling.h"
#include "log/log.h"
#include "tiling_base/tiling_templates_registry.h"

using namespace Ops::Transformer::OpTiling;

namespace optiling {

GmmA2AInt8Tiling::GmmA2AInt8Tiling(gert::TilingContext *context) : GroupedQbmmTiling(context)
{
    isInt8Mode_ = true;
    hcclTilingKey_ = 0;
}

void GmmA2AInt8Tiling::Reset(gert::TilingContext *context)
{
    GroupedQbmmTiling::Reset(context);
    isInt8Mode_ = true;
    hcclTilingKey_ = 0;
}

bool GmmA2AInt8Tiling::IsCapable()
{
    // Check if platform is ascend950 (arch35)
    auto platformInfoPtr = context_->GetPlatformInfo();
    if (platformInfoPtr != nullptr) {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
        auto socVersion = ascendcPlatform.GetSocVersion();
        // Only support ascend950 (arch35) for INT8 mode
        if (socVersion != platform_ascendc::SocVersion::ASCEND950) {
            OP_LOGI(context_, "GmmA2AInt8Tiling only supports ascend950, current platform is not capable.");
            return false;
        }
    }

    // Check if dtype is INT8
    auto xDesc = context_->GetDynamicInputDesc(0, 0); // X_INDEX = 0
    if (xDesc != nullptr) {
        auto dtype = xDesc->GetDataType();
        if (dtype != ge::DT_INT8) {
            OP_LOGI(context_, "GmmA2AInt8Tiling only supports INT8 input, current dtype is %s.",
                    ge::TypeUtils::DataTypeToSerialString(dtype).c_str());
            return false;
        }
    }

    return GroupedQbmmTiling::IsCapable();
}

ge::graphStatus GmmA2AInt8Tiling::DoOpTiling()
{
    // First call parent DoOpTiling for basic GMM tiling
    auto ret = GroupedQbmmTiling::DoOpTiling();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    // Additional gmmA2A-specific tiling logic
    OP_LOGI(context_, "GmmA2AInt8Tiling::DoOpTiling - INT8 mode for ascend950");

    return ge::GRAPH_SUCCESS;
}

uint64_t GmmA2AInt8Tiling::GetTilingKey() const
{
    // Get base tiling key from parent
    uint64_t baseTilingKey = GroupedQbmmTiling::GetTilingKey();

    // Add INT8 flag to tiling key (bit 8)
    uint64_t int8Flag = 1UL << 8;

    return baseTilingKey | int8Flag;
}

bool GmmA2AInt8Tiling::AnalyzeHcclTiling()
{
    // Analyze HCCL AllToAll tiling parameters
    OP_LOGI(context_, "GmmA2AInt8Tiling::AnalyzeHcclTiling");

    // Get HCCL-related attributes if any
    auto attrs = context_->GetAttrs();
    if (attrs != nullptr) {
        // Parse HCCL-specific attributes
        OP_LOGI(context_, "Analyzing HCCL tiling attributes");
    }

    return true;
}

void GmmA2AInt8Tiling::CombineGmmAndHcclTiling()
{
    // Combine GMM tiling with HCCL tiling data
    OP_LOGI(context_, "GmmA2AInt8Tiling::CombineGmmAndHcclTiling");

    // This is where we would combine the tiling data from both GMM and HCCL
    // For now, this is a placeholder for future implementation
}

// Register the tiling template
REGISTER_OPS_TILING_TEMPLATE(AlltoAllvGroupedMatMul, GmmA2AInt8Tiling, 0);

} // namespace optiling
