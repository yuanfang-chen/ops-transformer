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
#include "tiling/tiling_api.h"
#include "tiling/mc2_tiling_utils.h"
#include "tiling_base/tiling_base.h"
#include "tiling_base/tiling_templates_registry.h"
#include "mc2_matmul_tiling_cfg.h"
#include "tiling/new_mc2_tiling_utils.h"
#include "../grouped_mat_mul_allto_allv_tiling_base.h"
#include "../../../op_kernel/arch35/quant_grouped_mat_mul_allto_allv_tiling.h"
#include "../../../op_kernel/arch35/grouped_mat_mul_allto_allv_tiling_key.h"
#include "register/tilingdata_base.h"

// using MC2KernelTemplate::GmmTilingArray;
// using MC2KernelTemplate::GMMQuantTilingData;
// using MC2KernelTemplate::GMMArray;
namespace optiling {
namespace Mc2GroupedMatmul {

struct TilingInferredInfo {
    uint64_t gmmResultLen = 0UL; // 存储计算GMM的地址大小
    uint64_t mmResultLen = 0UL; // 存储计算MM的地址大小
    uint64_t commLen = 0UL; // 存储通信结果的临时空间，recvCounts
    uint64_t permuteLen = 0UL; // 重排空间大小, 应该与result一致
    uint32_t biasLen = 0UL; // 暂不支持bias
};

class QuantGroupedMatmulAllToAllvTiling : public GmmAlltoAllvTilingBase {
public:
    explicit QuantGroupedMatmulAllToAllvTiling(gert::TilingContext *context) : GmmAlltoAllvTilingBase(context) {};
    void Reset(gert::TilingContext *context) override
    {
        TilingBaseClass::Reset(context);
    }
    ~QuantGroupedMatmulAllToAllvTiling() override = default;
protected:
    void Reset();
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus GetPlatformInfo() override;
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus PostTiling() override;
    ge::graphStatus GetWorkspaceSize() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus CheckOpInputInfo();
    ge::graphStatus InitTilingContextParameters(); // set默认值，当前不支持功能参数
    ge::graphStatus SetTilingCommonInfo();
    ge::graphStatus SetGmmA2avWorkspaceInfo();
    ge::graphStatus DoQuantGMMTiling(); // 按专家为粒度执行
    ge::graphStatus SetHcclTiling();

    void PrintQuantGmmA2avTilingData(QuantGmmA2avTilingData &outTilingData);

    QuantGmmA2avTilingData localTilingData_;
    void PrintCommonTilingInfo(TaskTilingInfo &tilingInfo);
    void PrintSharedGmmTilingInfo(Mc2GroupedMatmulTilingData::GMMQuantTilingData &tiling);
    void PrintGmmQTilingDataInfo(GmmTilingArray &tilingInfo);
    const char *opName_{nullptr};
    uint32_t libApiWorkSpaceSize_{0};
    uint32_t workSpaceSize_{0};
    uint32_t epNum_{0};
    uint32_t aicNum_{0};
    TilingInferredInfo inferredInfo;
private:
    ge::graphStatus CalTilingInferredInfo();
};

} // namespace MC2Tiling
}
#endif
