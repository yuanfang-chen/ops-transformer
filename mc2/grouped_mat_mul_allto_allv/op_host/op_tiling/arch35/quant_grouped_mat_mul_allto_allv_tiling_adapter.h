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

#ifndef QUANT_GROUPED_MAT_MUL_ALLTO_ALLV_TILING_ADAPTER_H
#define QUANT_GROUPED_MAT_MUL_ALLTO_ALLV_TILING_ADAPTER_H

#pragma once
#include "securec.h"
#include "mc2_matmul_tiling_cfg.h"
#include "quant_grouped_mat_mul_allto_allv_tiling.h"
#include "tiling/new_mc2_tiling_utils.h"
#include "grouped_matmul/op_host/op_tiling/arch35/grouped_quant_matmul_tiling.h"
#include "register/tilingdata_base.h"


using namespace optiling;
namespace MC2Tiling {

class QuantGroupedMatmulAllToAllvAdapter : public Mc2GroupedMatmul::GroupedQbmmTiling {
public:
    explicit QuantGroupedMatmulAllToAllvAdapter(QuantGroupedMatmulAllToAllvTiling& tilingImpl,
        gert::TilingContext *context) : tilingProcesser_(tilingImpl), GroupedQbmmTiling(context) {}
    
    ~QuantGroupedMatmulAllToAllvAdapter() override = default;

    // ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus SetSharedExpertInputParameters();
    ge::graphStatus SetExpertInputParameters(const int64_t* sendCounts, uint64_t worldSize, uint64_t index,
                                             uint32_t epNums);
    const Mc2GroupedMatmulTilingData::GMMQuantTilingData& GetGmmQuantTilingAdapterData() const { return tilingData_; }

    // bool AnalyzeAttrs() override;
    // bool AnalyzeDtype() override;
    // bool AnalyzeInputs() override;
    void PrintMatmulParams();
    ge::graphStatus SetCommonContextParameters();
    ge::graphStatus Process();
    ge::graphStatus SetCommonInputParams();

    QuantGroupedMatmulAllToAllvTiling& tilingProcesser_;
    Mc2GroupedMatmulTilingData::GMMQuantTilingData tilingData_;
    // bool isWeightNz_ = false;

    int32_t mList_[Mc2GroupedMatmul::MAX_TENSOR_CONT] = {0};
    int32_t kList_[Mc2GroupedMatmul::MAX_TENSOR_CONT] = {0};
    int32_t nList_[Mc2GroupedMatmul::MAX_TENSOR_CONT] = {0};
};

} // namespace MC2Tiling
#endif