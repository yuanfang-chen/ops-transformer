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
 * \file moe_distribute_dispatch_teardown_tiling_base.h
 * \brief
 */

#ifndef MOE_DISTRIBUTE_DISPATCH_TEARDOWN_TILING_BASE_H_
#define MOE_DISTRIBUTE_DISPATCH_TEARDOWN_TILING_BASE_H_

#include "tiling/mc2_tiling_utils.h"
#include "tiling/moe_tiling_base.h"
#include "../../op_kernel/moe_distribute_dispatch_teardown_tiling.h"

namespace optiling {
class MoeDistributeDispatchTeardownTilingBase : public MoeTilingBase
{
public:
    explicit MoeDistributeDispatchTeardownTilingBase(gert::TilingContext* context)
        : MoeTilingBase(context), nodeName_(context->GetNodeName()){};

protected:
    const char* socTilingName_;
    const char* nodeName_;
    MoeDistributeDispatchTeardownTilingData* tilingData_ = nullptr;
    std::string groupEp_;

    uint64_t GetTilingKey() const override;
};
} // namespace optiling
#endif