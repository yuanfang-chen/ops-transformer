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
};
} // namespace optiling
#endif // MOE_DISTRIBUTE_DISPATCH_SETUP_TILING_ARCH35_H_