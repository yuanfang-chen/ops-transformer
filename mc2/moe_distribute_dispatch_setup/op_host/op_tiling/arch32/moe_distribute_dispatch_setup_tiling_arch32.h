/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file moe_distribute_dispatch_setup_tiling_arch32.h
 * \brief
 */

#ifndef MOE_DISTRIBUTE_DISPATCH_SETUP_TILING_ARCH32_H_
#define MOE_DISTRIBUTE_DISPATCH_SETUP_TILING_ARCH32_H_

#include "..\moe_distribute_dispatch_setup_tiling_base.h"

namespace optiling {
class MoeDistributeDispatchSetupTilingA3 : public MoeDistributeDispatchSetupTilingBase
{
public:
    explicit MoeDistributeDispatchSetupTilingA3(gert::TilingContext* context)
        : MoeDistributeDispatchSetupTilingBase(context)
    {
        socTilingName_ = "MoeDistributeDispatchSetupA3";
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

    ge::graphStatus SetWorkspace() override;
    ge::graphStatus CheckHcclBuffSize() override;
    void SetTilingKey() override;
    void SetPlatformInfo() override;
    void SetHcommCfg() override;
    void PrintTilingDataInfo() override;
};
} // namespace optiling
#endif // MOE_DISTRIBUTE_DISPATCH_SETUP_TILING_ARCH32_H_