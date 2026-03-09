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
 * \file allto_allv_grouped_mat_mul_quant_tiling_base.h
 * \brief
 */
#ifndef ALLTO_ALLV_GROUPED_MATMUL_QUANT_TILING_BASE_H
#define ALLTO_ALLV_GROUPED_MATMUL_QUANT_TILING_BASE_H

#include "../allto_allv_grouped_mat_mul_tiling_base.h"

namespace optiling {
constexpr uint32_t DATA_SIZE_L0C = 4;
constexpr uint64_t CUBE_REDUCE_BLOCK = 32;
constexpr uint32_t BASIC_BLOCK_SIZE_512 = 512;
constexpr uint32_t BASIC_BLOCK_SIZE_256 = 256;
constexpr uint32_t BASIC_BLOCK_SIZE_128 = 128;
constexpr uint32_t SINGLE_GROUP_NUM = 1;
constexpr uint32_t GMM_ACT_TYPE_NONE = 0;
constexpr uint64_t DB_SIZE = 2UL;
constexpr uint64_t MTE2_MIN_LOAD_SIZE = 64 * 1024UL;
// pertensor
constexpr uint32_t PERTENSOR_MODE = 1;
// pergroup
constexpr uint32_t MX_PERGROUP_MODE = 6;
constexpr uint64_t MX_BASIC_FACTOR = 64;
constexpr uint64_t SCALER_FACTOR_MAX = 127;
constexpr uint64_t SCALER_FACTOR_MIN = 1;
constexpr uint32_t SCALER_FACTOR_DEFAULT = 1;
constexpr uint32_t SCALER_FACTOR_B_BIT = 8;
constexpr uint32_t SCALER_FACTOR_M_BIT = 16;
constexpr uint32_t SCALER_FACTOR_N_BIT = 24;

class AlltoAllvGmmQuantTilingBase : public AlltoAllvGmmTilingBase {
public:
    explicit AlltoAllvGmmQuantTilingBase(gert::TilingContext *context) : AlltoAllvGmmTilingBase(context){
        tilingData = context->GetTilingData<QuantAlltoAllvGroupedMatmulTilingData>();
    };
    QuantAlltoAllvGroupedMatmulTilingData *tilingData;

protected:
    // Tiling base
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    // quant base
    virtual ge::graphStatus CheckGmmDType() const {return ge::GRAPH_SUCCESS;};
    virtual ge::graphStatus CheckMmDType() const {return ge::GRAPH_SUCCESS;};
    virtual ge::graphStatus CheckQuantMode() const {return ge::GRAPH_SUCCESS;};
    ge::graphStatus SetHcclTiling() const;
    void SetGMMQuantParams(Mc2GroupedMatmulTilingData::GMMQuantTilingData &gmmQuantTilingData) const;
    void SetTilingArray(Mc2GroupedMatmulTilingData::GMMQuantTilingData &gmmQuantTilingData, uint64_t M, uint64_t N, uint64_t K) const;
    virtual void SetTilingParams(Mc2GroupedMatmulTilingData::GMMQuantTilingData &gmmQuantTilingData, uint64_t M, uint64_t N, uint64_t K) const {};
    void PrintGMMQuantTilingData(const Mc2GroupedMatmulTilingData::GMMQuantTilingData &data) const;
    void PrintTaskTilingInfo(const MC2KernelTemplate::TaskTilingInfo& taskTilingInfo) const;
    bool isMxfp8_{false};
    bool isHif8_{false};
};
}  // namespace optiling
#endif  // ALLTO_ALLV_GROUPED_MATMUL_QUANT_TILING_H