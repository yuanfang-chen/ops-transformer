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
    enum TensorType
    {
        INPUT = 0,
        OUTPUT = 1,
        OPTIONINPUT = 2
    };

protected:
    const char* socTilingName_;
    std::string nodeName_;
    MoeDistributeDispatchSetupTilingData* tilingData_ = nullptr;
    std::string groupEp_;

    uint64_t GetTilingKey() const override;
    
    virtual ge::graphStatus CheckRequiredAttrValue();
    virtual ge::graphStatus GetRequiredAttrAndSetTilingData();
    virtual ge::graphStatus CheckSharedExpertAttrValue();
    virtual ge::graphStatus CheckOptionalAttrValue();
    virtual ge::graphStatus GetOptionalAttrAndSetTilingData();
    virtual ge::graphStatus GetComplexAttrAndSetTilingData();

    virtual ge::graphStatus CheckInputTensorDataType();
    virtual ge::graphStatus CheckOptionalInputTensorDataType();
    virtual ge::graphStatus CheckOutputTensorDataType();
    virtual ge::graphStatus CheckTensorDataType();
    virtual ge::graphStatus CheckTensorDim();
    virtual ge::graphStatus CheckTensorShapeRelation();
    virtual ge::graphStatus CheckComplexTensorShapeSize();
    virtual ge::graphStatus CheckTensorShapeSizeAndSetTilingData();
    virtual ge::graphStatus CheckCalcTensorShapeSizeAndSetTilingData();

    virtual ge::graphStatus MoeDistributeDispatchSetupTilingFuncImpl();
    virtual ge::graphStatus CheckInputTensorDim();
    virtual ge::graphStatus CheckOptionalInputTensorDim();
    virtual ge::graphStatus CheckOutputTensorDim();

    virtual ge::graphStatus SetWorkspace();
    virtual ge::graphStatus CheckHcclBuffSize();
    virtual void SetTilingKey();
    virtual void SetPlatformInfo();
    virtual void SetHcommCfg();
    virtual void PrintTilingDataInfo();
};
} // namespace optiling
#endif