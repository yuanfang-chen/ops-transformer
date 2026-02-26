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
 * \file moe_distribute_dispatch_teardown_tiling_arch35.h
 * \brief
 */

#ifndef MOE_DISTRIBUTE_DISPATCH_TEARDOWN_TILING_A5_H_
#define MOE_DISTRIBUTE_DISPATCH_TEARDOWN_TILING_A5_H_

#include "../moe_distribute_dispatch_teardown_tiling_base.h"

namespace optiling {
class MoeDistributeDispatchTeardownTilingA5 : public MoeDistributeDispatchTeardownTilingBase
{
public:
    explicit MoeDistributeDispatchTeardownTilingA5(gert::TilingContext* context)
        : MoeDistributeDispatchTeardownTilingBase(context)
    {
        socTilingName_ = "MoeDistributeDispatchTeardownA3";
    }

private:
    ge::graphStatus DoOpTiling() final;
    bool IsCapable() final;

    ge::graphStatus CheckRequiredAttrValue();
    ge::graphStatus GetRequiredAttrAndSetTilingData();
    ge::graphStatus CheckOptionalAttrValue();
    ge::graphStatus GetOptionalAttrAndSetTilingData();

    ge::graphStatus MoeDistributeDispatchTeardownTilingFuncImpl();
    ge::graphStatus CheckTensorShape();
    ge::graphStatus CheckTensorDataType();
    ge::graphStatus CheckHcclBuffSize();
    ge::graphStatus SetWorkSpace();
    void SetTilingKey();
    void SetHcommCfg();
    void SetPlatformInfo();
    void PrintTilingDataInfo();

    bool CheckInputTensorShapeDim();
    bool CheckOutputTensorShapeDim();
    bool CheckTensorShapeRelation();
    bool CheckTensorShapeSize();
    bool CheckInputTensorDataType();
    bool CheckOutputTensorDataType();
    bool CheckRelationTensorDataType();
};
} // namespace optiling
#endif // MOE_DISTRIBUTE_DISPATCH_TEARDOWN_TILING_A3_H_