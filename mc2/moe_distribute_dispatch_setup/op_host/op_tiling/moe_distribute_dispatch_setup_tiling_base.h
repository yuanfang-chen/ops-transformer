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
 * \file moe_distribute_dispatch_setup_tiling_base.h
 * \brief
 */

#ifndef MOE_DISTRIBUTE_DISPATCH_SETUP_TILING_BASE_H_
#define MOE_DISTRIBUTE_DISPATCH_SETUP_TILING_BASE_H_

#include "tiling/mc2_tiling_utils.h"
#include "tiling/moe_tiling_base.h"
#include "../../op_kernel/moe_distribute_dispatch_setup_tiling.h"

namespace optiling {
class MoeDistributeDispatchSetupTilingBase : public MoeTilingBase
{
public:
    explicit MoeDistributeDispatchSetupTilingBase(gert::TilingContext* context)
        : MoeTilingBase(context), nodeName_(context->GetNodeName()){};

protected:
    const char* socTilingName_;
    std::string nodeName_;
    MoeDistributeDispatchSetupTilingData* tilingData_ = nullptr;
    std::string groupEp_;

    uint64_t GetTilingKey() const override;

    void PrintTilingDataInfo();
    
    enum TensorType
    {
        INPUT = 0,
        OUTPUT = 1,
        OPTIONINPUT = 2
    };
    ge::graphStatus DoOpTiling() override;
    bool IsCapable() override;

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
#endif