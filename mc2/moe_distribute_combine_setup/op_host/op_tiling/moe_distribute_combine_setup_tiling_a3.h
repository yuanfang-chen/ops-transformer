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
 * \file moe_distribute_combine_setup_tiling_a3.h
 * \brief
 */

#ifndef MOE_DISTRIBUTE_COMBINE_SETUP_TILING_A3_H_
#define MOE_DISTRIBUTE_COMBINE_SETUP_TILING_A3_H_

#include "moe_distribute_combine_setup_tiling_base.h"

namespace optiling {
class MoeDistributeCombineSetupTilingA3 : public MoeDistributeCombineSetupTilingBase
{
public:
    explicit MoeDistributeCombineSetupTilingA3(gert::TilingContext* context)
        : MoeDistributeCombineSetupTilingBase(context)
    {
        socTilingName_ = "MoeDistributeCombineSetupA3";
    }

private:
    enum TensorType
    {
        INPUT = 0,
        OUTPUT = 1,
        OPTIONINPUT = 2
    };

    ge::graphStatus DoOpTiling() final;
    bool IsCapable() final;

    ge::graphStatus CheckRequiredAttrValue();
    ge::graphStatus GetRequiredAttrAndSetTilingData();
    ge::graphStatus CheckSharedExpertAttrValue();
    ge::graphStatus CheckOptionalAttrValue();
    ge::graphStatus GetOptionalAttrAndSetTilingData();
    ge::graphStatus GetComplexAttrAndSetTilingData();

    ge::graphStatus CheckTensorDataType();
    ge::graphStatus CheckTensorDim();
    ge::graphStatus CheckTensorShapeRelation();
    ge::graphStatus CheckTensorShapeSizeAInMoeRank();
    ge::graphStatus CheckTensorShapeSizeAInSharedRank();
    ge::graphStatus CheckTensorShapeSizeAndSetTilingData();
    ge::graphStatus CheckCalcTensorShapeSizeAndSetTilingData();

    ge::graphStatus MoeDistributeCombineSetupTilingFuncImpl();
    ge::graphStatus CheckOneTensorDim(std::string name, TensorType tensortype, uint32_t index, uint32_t dims);
    ge::graphStatus CheckInputTensorDim();
    ge::graphStatus CheckOutputTensorDim();
    ge::graphStatus SetWorkspace();
    ge::graphStatus CheckHcclBuffSize();
    void SetTilingKey();
    void SetPlatformInfo();
    void SetHcommCfg();
    void PrintTilingDataInfo();
};
} // namespace optiling
#endif // MOE_DISTRIBUTE_COMBINE_SETUP_TILING_A3_H_