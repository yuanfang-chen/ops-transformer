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
 * \file moe_distribute_dispatch_teardown_tiling_arch32.h
 * \brief
 */

#ifndef MOE_DISTRIBUTE_DISPATCH_TEARDOWN_TILING_ARCH32_H_
#define MOE_DISTRIBUTE_DISPATCH_TEARDOWN_TILING_ARCH32_H_

#include "../moe_distribute_dispatch_teardown_tiling_base.h"

namespace optiling {
class MoeDistributeDispatchTeardownTilingA3 : public MoeDistributeDispatchTeardownTilingBase
{
public:
    explicit MoeDistributeDispatchTeardownTilingA3(gert::TilingContext* context)
        : MoeDistributeDispatchTeardownTilingBase(context)
    {
        socTilingName_ = "MoeDistributeDispatchTeardownA3";
    }

private:
    ge::graphStatus DoOpTiling() override final;
    ge::graphStatus MoeDistributeDispatchTeardownTilingFuncImpl() override;
    bool IsCapable() override final;
};
} // namespace optiling
#endif // MOE_DISTRIBUTE_DISPATCH_TEARDOWN_TILING_ARCH32_H_