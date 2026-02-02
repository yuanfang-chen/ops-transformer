/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_distribute_dispatch_setup_tiling_arch35.h
 * \brief
 */

#ifndef MOE_DISTRIBUTE_DISPATCH_SETUP_TILING_ARCH35_H_
#define MOE_DISTRIBUTE_DISPATCH_SETUP_TILING_ARCH35_H_

#include "../moe_distribute_dispatch_setup_tiling_base.h"

namespace optiling {
class MoeDistributeDispatchSetupTilingA5 : public MoeDistributeDispatchSetupTilingBase
{
public:
    explicit MoeDistributeDispatchSetupTilingA5(gert::TilingContext* context)
        : MoeDistributeDispatchSetupTilingBase(context)
    {
        socTilingName_ = "MoeDistributeDispatchSetupA5";
    }
private:
    ge::graphStatus DoOpTiling() override final;
    bool IsCapable() override final;
    
    ge::graphStatus CheckRequiredAttrValue() override;
    ge::graphStatus GetRequiredAttrAndSetTilingData() override;
    ge::graphStatus CheckSharedExpertAttrValue() override;
    ge::graphStatus CheckOptionalAttrValue() override;
    ge::graphStatus GetOptionalAttrAndSetTilingData() override;
    ge::graphStatus GetComplexAttrAndSetTilingData() override;

    ge::graphStatus CheckInputTensorDataType() override;
    ge::graphStatus CheckOptionalInputTensorDataType() override;
    ge::graphStatus CheckOutputTensorDataType() override;
    ge::graphStatus CheckTensorDataType() override;
    ge::graphStatus CheckTensorDim() override;
    ge::graphStatus CheckTensorShapeRelation() override;
    ge::graphStatus CheckComplexTensorShapeSize() override;
    ge::graphStatus CheckTensorShapeSizeAndSetTilingData() override;
    ge::graphStatus CheckCalcTensorShapeSizeAndSetTilingData() override;

    ge::graphStatus MoeDistributeDispatchSetupTilingFuncImpl() override;
    ge::graphStatus CheckInputTensorDim() override;
    ge::graphStatus CheckOptionalInputTensorDim() override;
    ge::graphStatus CheckOutputTensorDim() override;
    ge::graphStatus CheckOneTensorDim(std::string name, TensorType tensortype, uint32_t index, uint32_t dims);

    ge::graphStatus SetWorkspace() override;
    ge::graphStatus CheckHcclBuffSize() override;
    void SetTilingKey() override;
    void SetPlatformInfo() override;
    void SetHcommCfg() override;
    void PrintTilingDataInfo() override;
};
} // namespace optiling
#endif // MOE_DISTRIBUTE_DISPATCH_SETUP_TILING_ARCH35_H_