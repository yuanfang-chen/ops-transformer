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
 * \file allto_allv_mx_quant_grouped_mat_mul_tiling.h
 * \brief
 */
#ifndef ALLTO_ALLV_MX_QUANT_GROUPED_MATMUL_TILING_H
#define ALLTO_ALLV_MX_QUANT_GROUPED_MATMUL_TILING_H

#include "allto_allv_quant_grouped_mat_mul_tiling_common.h"
#include "../../../../3rd/grouped_matmul/op_tiling/gmm_qbmm_tiling.h"

namespace optiling {
class AlltoAllvMXQuantGmmTiling : public AlltoAllvQuantGmmTilingCommon {
    friend class AlltoAllvMXQuantGmmTilingHelper;
public:
    explicit AlltoAllvMXQuantGmmTiling(gert::TilingContext *context) : AlltoAllvQuantGmmTilingCommon(context){};

protected:
    // tiling base
    bool IsCapable() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus DoLibApiTiling() override;
    // quant base
    ge::graphStatus CheckGmmDType() const override;
    ge::graphStatus CheckMmDType() const override;
    ge::graphStatus CheckScaleShape() const override;
    ge::graphStatus CheckQuantMode() const override;
    void GetPermuteScaleOutSize() override;
    ge::graphStatus CheckShareExpScaleShape() const;
    // helper
    bool transB_{false};
    ge::DataType cDtype_;
    uint64_t mSize_{0};
    uint64_t nSize_{0};
    uint64_t kSize_{0};
};

class AlltoAllvMXQuantGmmTilingHelper : public Mc2GroupedMatmulTiling::Mc2GroupedQbmmTiling {
public:
    AlltoAllvMXQuantGmmTilingHelper(
        AlltoAllvMXQuantGmmTiling& AlltoAllvMXQuantGmmTiling, Mc2GroupedMatmulTilingData::GMMQuantTilingData& data)
        : Mc2GroupedQbmmTiling(AlltoAllvMXQuantGmmTiling.context_, &data),
        tilingProcesser_(AlltoAllvMXQuantGmmTiling)
    {}
    bool AnalyzeAttrs() override;
    bool AnalyzeDtype() override;
    bool AnalyzeInputs() override;
private:
    AlltoAllvMXQuantGmmTiling& tilingProcesser_;
};

}  // namespace optiling
#endif  // ALLTO_ALLV_MX_QUANT_GROUPED_MATMUL_TILING_H
