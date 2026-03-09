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
 * \file allto_allv_grouped_mat_mul_mx_quant_tiling.h
 * \brief
 */
#ifndef ALLTO_ALLV_GROUPED_MATMUL_MX_QUANT_TILING_H
#define ALLTO_ALLV_GROUPED_MATMUL_MX_QUANT_TILING_H

#include "allto_allv_grouped_mat_mul_quant_tiling_base.h"

namespace optiling {
class AlltoAllvGmmMXQuantTiling : public AlltoAllvGmmQuantTilingBase {
public:
    explicit AlltoAllvGmmMXQuantTiling(gert::TilingContext *context) : AlltoAllvGmmQuantTilingBase(context){};

protected:
    // tiling base
    bool IsCapable() override;
    uint64_t GetTilingKey() const override;
    // quant base
    ge::graphStatus CheckGmmDType() const override;
    ge::graphStatus CheckMmDType() const override;
    ge::graphStatus CheckQuantMode() const override;
    void SetTilingParams(Mc2GroupedMatmulTilingData::GMMQuantTilingData &gmmQuantTilingData, uint64_t M, uint64_t N, uint64_t K) const override;
private:
    ge::graphStatus CheckShareExpertQuantMode() const;
    void SetTilingMxTypeParam(Mc2GroupedMatmulTilingData::GMMQuantTilingData &gmmQuantTilingData) const;
    uint64_t GetDepthA1B1(uint64_t leftSize, uint64_t perDepthSize, uint64_t baseKSize, uint64_t depthInit) const;
};
}  // namespace optiling
#endif  // ALLTO_ALLV_GROUPED_MATMUL_QUANT_TILING_H