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

struct QuantGmmAlltoAllvParamsInfo {
    uint64_t A;
    uint64_t H1;
    uint64_t ep;
    uint64_t BsK;
    uint64_t N1;
    uint64_t Bs;
    uint64_t H2;
    uint64_t N2;
    uint64_t epWorldSize;
    uint64_t aivCoreNum;
    uint64_t aicCoreNum;
    uint64_t gmmWeightDim1;
    uint64_t gmmWeightDim2;
    uint64_t mmWeightDim0;
    uint64_t mmWeightDim1;
    int64_t gmmXQuantMode;
    int64_t gmmWeightQuantMode;
    int64_t mmXQuantMode;
    int64_t mmWeightQuantMode;
    int64_t commQuantMode;
    int64_t commQuantDtype;
    int64_t gmmYDtype;
    int64_t mmYDtype;
    int64_t groupSize;
    bool hasSharedMm;
    bool isGmmWeightTrans;
    bool isMmWeightTrans;
};

struct TilingInferredInfo {
    uint64_t gmmResultLen = 0UL; // 存储计算GMM的地址大小
    // uint64_t mmResultLen = 0UL; // 存储计算MM的地址大小
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
    ge::graphStatus CheckAndSetInputOutputInfo();
    ge::graphStatus InitTilingContextParameters(); // set默认值，当前不支持功能参数
    ge::graphStatus SetTilingCommonInfo();
    ge::graphStatus SetGmmA2avWorkspaceInfo();
    ge::graphStatus DoQuantGMMTiling(); // 按专家为粒度执行
    ge::graphStatus SetHcclTiling();
    void PrintQuantGmmA2avTilingData(QuantGmmA2avTilingData &outTilingData);
    void PrintCommonTilingInfo(TaskTilingInfo &tilingInfo);
    void PrintSharedGmmTilingInfo(Mc2GroupedMatmulTilingData::GMMQuantTilingData &tiling);
    void PrintGmmQTilingDataInfo(GmmTilingArray &tilingInfo);
    const char *opName_{nullptr};
    uint32_t libApiWorkSpaceSize_{0};
    uint32_t workSpaceSize_{0};
    QuantGmmA2avTilingData localTilingData_ = {0};
    TilingInferredInfo inferredInfo = {0};
    QuantGmmAlltoAllvParamsInfo localParams_ = {0};

private:
    ge::graphStatus CheckOpInputSingleParamsTensorNotSup();
    ge::graphStatus CheckOpInputSingleParamsTensorSup();
    ge::graphStatus CheckOpInputSingleParamsTensorMM();
    ge::graphStatus CheckOpInputSingleParamsTensor();
    ge::graphStatus CheckAndSetLocalParamsGmm();
    ge::graphStatus CheckAndSetLocalParamsMm();
    ge::graphStatus CheckAndSetLocalParamsAttr();
    ge::graphStatus CheckAndSetLocalParams();
    ge::graphStatus CheckParamsRelationGmm();
    ge::graphStatus CheckParamsRelationMm();
    ge::graphStatus CheckParamsAttrEpAndSetLocalParams();
    ge::graphStatus CheckAndSetSendRecvCountsAttr();
    ge::graphStatus CheckLocalParams();
    ge::graphStatus CheckParamsRelationAndSetLocalParams();
    ge::graphStatus CalTilingInferredInfo();
};

} // namespace MC2Tiling
}
#endif
