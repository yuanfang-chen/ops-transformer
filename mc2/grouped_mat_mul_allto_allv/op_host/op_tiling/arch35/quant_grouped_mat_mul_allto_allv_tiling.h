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
 * \file quant_grouped_mat_mul_allto_allv_gmm_tiling.h
 * \brief
 */

#ifndef QUANT_GROUPED_MAT_MUL_ALLTO_ALLV_TILING_H
#define QUANT_GROUPED_MAT_MUL_ALLTO_ALLV_TILING_H

#pragma once
#include "securec.h"
#include "mc2_matmul_tiling_cfg.h"
#include "../grouped_mat_mul_allto_allv_tiling_base.h"
#include "tiling/new_mc2_tiling_utils.h"
#include "grouped_matmul_finalize_routing/op_kernel/arch35/grouped_matmul_finalize_routing_tiling_data.h"
#include "grouped_matmul/op_kernel/arch35/quant_adaptive_sliding_window_templates/gqmm_tiling_key.h"
#include "grouped_matmul/op_kernel/arch35/grouped_matmul_tiling_data_apt.h"
#include "mc2/allto_allv_grouped_mat_mul/op_kernel/mc2_templates/common/a2av_common_tiling.h"
#include "register/tilingdata_base.h"

using MC2KernelTemplate::GmmTilingArray;
using MC2KernelTemplate::GMMQuantTilingData;
using MC2KernelTemplate::GMMArray;
using namespace optiling;
namespace MC2Tiling {

struct TilingInferredInfo {
    uint64_t gmmResultLen = 0UL; // 存储计算GMM的地址大小
    uint64_t mmResultLen = 0UL; // 存储计算MM的地址大小
    uint64_t commLen = 0UL; // 存储通信结果的临时空间，recvCounts
    uint64_t permuteLen = 0UL; // 重排空间大小, 应该与result一致
    uint32_t biasLen = 0UL; // 暂不支持bias
};

class QuantGroupedMatmulAllToAllvTiling : public GmmAlltoAllvTilingBase {
public:
    explicit QuantGroupedMatmulAllToAllvTiling(gert::TilingContext *context);
    ~QuantGroupedMatmulAllToAllvTiling() override = default;
protected:
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus PostTiling() override;
    ge::graphStatus GetWorkspaceSize() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus CheckOpInputInfo();
    ge::graphStatus InitTilingContextParameters(); // set默认值，当前不支持功能参数
    ge::graphStatus DoQuantGMMTiling(); // 按专家为粒度执行
    ge::graphStatus SetHcclTiling();
    
    void SetTilingCommonInfo();
    void PrintQuantGmmA2avTilingData(QuantGmmA2avTilingData &outTilingData);
    
private:
    QuantGmmA2avTilingData localTilingData_;
    void PrintCommonTilingInfo(TaskTilingInfo &tilingInfo);
    void PrintSharedGmmTilingInfo(GroupedMatmulTilingData::GMMQuantTilingData &tiling);
    void PrintGmmQTilingDataInfo(GmmTilingArray &tilingInfo);
    const char *opName_{nullptr};
    uint32_t libApiWorkSpaceSize_{0};
    uint32_t epNum_{2};
    TilingInferredInfo inferredInfo;
};

} // namespace MC2Tiling
#endif
