/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file mx_quant_grouped_mat_mul_allto_allv_gmm_tiling.h
 * \brief
 */

#ifndef MX_QUANT_GROUPED_MAT_MUL_ALLTO_ALLV_TILING_H
#define MX_QUANT_GROUPED_MAT_MUL_ALLTO_ALLV_TILING_H

#pragma once
#include "securec.h"
#include "tiling/tiling_api.h"
#include "op_host/op_tiling/mc2_tiling_utils.h"
#include "tiling_base/tiling_base.h"
#include "tiling_base/tiling_templates_registry.h"
#include "mc2_matmul_tiling_cfg.h"
#include "op_host/op_tiling/new_mc2_tiling_utils.h"
#include "quant_grouped_mat_mul_allto_allv_tiling_base.h"
#include "../../../op_kernel/arch35/quant_grouped_mat_mul_allto_allv_tiling.h"
#include "../../../op_kernel/quant_grouped_mat_mul_allto_allv_tiling_key.h"
#include "register/tilingdata_base.h"

namespace optiling {
namespace Mc2GroupedMatmul {
class MxQuantGroupedMatmulAllToAllvTiling : public QuantGroupedMatmulAllToAllvTilingBase {
public:
    explicit MxQuantGroupedMatmulAllToAllvTiling(gert::TilingContext *context) : QuantGroupedMatmulAllToAllvTilingBase(context) {};
    void Reset(gert::TilingContext *context) override
    {
        TilingBaseClass::Reset(context);
    }
    ~MxQuantGroupedMatmulAllToAllvTiling() override = default;
protected:
    void Reset();
    bool IsCapable() override;
    ge::graphStatus GetWorkspaceSize() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus CheckAndSetInputOutputInfo();
    ge::graphStatus SetGmmA2avWorkspaceInfo();

    // ge::graphStatus CheckOpInputSingleParamsTensorNotSupport();
    // ge::graphStatus CheckOpInputSingleParamsTensorSupport();
    // ge::graphStatus CheckFormat();
    // ge::graphStatus CheckOpInputSingleParamsTensorMM();
    // ge::graphStatus CheckOpInputSingleParamsTensor();
    // ge::graphStatus CheckAndSetLocalParamsGmm();
    // ge::graphStatus CheckAndSetLocalParamsMm();
    // ge::graphStatus CheckAndSetLocalParamsAttr();
    // ge::graphStatus CheckAndSetLocalParams();
    // ge::graphStatus CheckParamsRelationGmm();
    // ge::graphStatus CheckParamsRelationMm();
    // ge::graphStatus CheckParamsAttrEpAndSetLocalParams();
    // ge::graphStatus CheckAndSetSendRecvCountsAttr();
    // ge::graphStatus CheckLocalParams();
    // ge::graphStatus CheckParamsRelationAndSetLocalParams();
};

} // namespace Mc2GroupedMatmul
}
#endif
