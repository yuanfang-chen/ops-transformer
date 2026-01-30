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
 * \file allto_allv_grouped_mat_mul_quant_tiling.h
 * \brief
 */
#ifndef ALLTO_ALLV_GROUPED_MATMUL_QUANT_TILING_H
#define ALLTO_ALLV_GROUPED_MATMUL_QUANT_TILING_H

#include "../allto_allv_grouped_mat_mul_tiling_base.h"
#include "../allto_allv_grouped_mat_mul_checker.h"

namespace optiling {
class AlltoAllvGmmQuantTiling : public AlltoAllvGmmTilingBase {
public:
    explicit AlltoAllvGmmQuantTiling(gert::TilingContext *context) : AlltoAllvGmmTilingBase(context),checker(context){};
    QuantAlltoAllvGroupedMatmulTilingData *tilingData;
    ge::graphStatus Init(gert::TilingContext *context);
    ge::graphStatus RunFusionKernelTiling(gert::TilingContext *context);
    virtual ~AlltoAllvGmmQuantTiling() = default;

protected:
    ge::graphStatus GetWorkspaceSize();
    ge::graphStatus PostTiling();
    ge::graphStatus DoOpTiling();
    bool IsCapable();
    ge::graphStatus GetContextAttr(const gert::TilingContext *context);
    ge::graphStatus GetShapeAndFormat(const gert::TilingContext *context);
    ge::graphStatus SetHcclTiling(const gert::TilingContext *context) const;
    void SetGMMQuantParams(Mc2GroupedMatmulTilingData::GMMQuantTilingData &gmmQuantTilingData);
    void SetGMMArray(Mc2GroupedMatmulTilingData::GMMQuantTilingData &gmmQuantTilingData);
    void SetTilingParams(Mc2GroupedMatmulTilingData::GMMQuantTilingData &gmmQuantTilingData);
    ge::graphStatus DoAiCoreTiling(const gert::TilingContext *context);
    uint64_t GetTilingKey(const gert::TilingContext *context) const;
    uint64_t GetTilingKey() const;
    void PrintQuantTilingData(const Mc2GroupedMatmulTilingData::GMMQuantTilingData &data) const;

private:
    int32_t maxM_;
    int32_t maxN_;
    int32_t maxK_;
    int32_t baseM_;
    int32_t baseN_;
    int32_t baseK_;
    uint32_t mmDataTypeSize;

    int32_t maxMForMM_;
    int32_t maxNForMM_;
    int32_t maxKForMM_;
    int32_t baseMForMM_;
    int32_t baseNForMM_;
    int32_t baseKForMM_;

    const char *epGroup_;
    uint32_t rankSize_;
    uint32_t libApiWorkSpaceSize_;
    uint64_t epWorldSize_;
    int32_t mSize_;
    AlltoAllvGmmChecker checker;  // 
    ge::DataType mmDType_ = ge::DT_UNDEFINED;

    bool isGmmWeightTrans;
    bool isMmWeightTrans;
    bool isPermuteOut;
    bool isNeedMM;
};
}  // namespace optiling
#endif  // ALLTO_ALLV_GROUPED_MATMUL_QUANT_TILING_H