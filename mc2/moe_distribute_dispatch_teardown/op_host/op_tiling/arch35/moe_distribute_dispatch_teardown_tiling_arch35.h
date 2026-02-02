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
 * \file moe_distribute_dispatch_teardown_tiling_arch35.h
 * \brief
 */

#ifndef MOE_DISTRIBUTE_DISPATCH_TEARDOWN_TILING_ARCH35_H_
#define MOE_DISTRIBUTE_DISPATCH_TEARDOWN_TILING_ARCH35_H_

#include "../moe_distribute_dispatch_teardown_tiling_base.h"

namespace optiling {
class MoeDistributeDispatchTeardownTilingA5 : public MoeDistributeDispatchTeardownTilingBase
{
public:
    explicit MoeDistributeDispatchTeardownTilingA5(gert::TilingContext* context)
        : MoeDistributeDispatchTeardownTilingBase(context)
    {
        socTilingName_ = "MoeDistributeDispatchTeardownA5";
    }

private:
    ge::graphStatus DoOpTiling() override final;
    bool IsCapable() override final;

    ge::graphStatus CheckRequiredAttrValue() override;
    ge::graphStatus GetRequiredAttrAndSetTilingData() override;
    ge::graphStatus CheckOptionalAttrValue() override;
    ge::graphStatus GetOptionalAttrAndSetTilingData() override;

    ge::graphStatus MoeDistributeDispatchTeardownTilingFuncImpl() override;
    ge::graphStatus CheckTensorShape() override;
    ge::graphStatus CheckTensorDataType() override;
    ge::graphStatus CheckHcclBuffSize() override;
    ge::graphStatus SetWorkSpace() override;
    void SetTilingKey() override;
    void SetHcommCfg() override;
    void SetPlatformInfo() override;
    void PrintTilingDataInfo() override;

    bool CheckInputTensorShapeDim() override;
    bool CheckOutputTensorShapeDim() override;
    bool CheckTensorShapeRelation() override;
    bool CheckTensorShapeSize() override;
    bool CheckInputTensorDataType() override;
    bool CheckOutputTensorDataType() override;
    bool CheckRelationTensorDataType() override;
};
} // namespace optiling
#endif // MOE_DISTRIBUTE_DISPATCH_TEARDOWN_TILING_ARCH35_H_