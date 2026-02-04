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
 * \file moe_distribute_dispatch_setup_tiling_a3.h
 * \brief
 */

#ifndef MOE_DISTRIBUTE_DISPATCH_SETUP_TILING_A3_H_
#define MOE_DISTRIBUTE_DISPATCH_SETUP_TILING_A3_H_

#include "moe_distribute_dispatch_setup_tiling_base.h"

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

    ge::graphStatus CheckInputTensorDataType();
    ge::graphStatus CheckOptionalInputTensorDataType();
    ge::graphStatus CheckOutputTensorDataType();
    ge::graphStatus CheckTensorDataType();
    ge::graphStatus CheckTensorDim();
    ge::graphStatus CheckTensorShapeRelation();
    ge::graphStatus CheckComplexTensorShapeSize();
    ge::graphStatus CheckTensorShapeSizeAndSetTilingData();
    ge::graphStatus CheckCalcTensorShapeSizeAndSetTilingData();

    ge::graphStatus MoeDistributeDispatchSetupTilingFuncImpl();
    ge::graphStatus CheckOneTensorDim(std::string name, TensorType tensortype, uint32_t index, uint32_t dims);
    ge::graphStatus CheckInputTensorDim();
    ge::graphStatus CheckOptionalInputTensorDim();
    ge::graphStatus CheckOutputTensorDim();

    ge::graphStatus SetWorkspace();
    ge::graphStatus CheckHcclBuffSize();
    void SetTilingKey();
    void SetPlatformInfo();
    void SetHcommCfg();
    void PrintTilingDataInfo();
};
} // namespace optiling
#endif // MOE_DISTRIBUTE_DISPATCH_SETUP_TILING_A3_H_