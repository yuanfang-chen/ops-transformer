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
 * \file moe_distribute_dispatch_v2_tiling.h
 * \brief
 */

#ifndef MOE_DISTRIBUTE_DISPATCH_TILING_V2
#define MOE_DISTRIBUTE_DISPATCH_TILING_V2

#include "tiling/mc2_tiling_utils.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {

struct DispatchV2Config {
    uint32_t contextIndex = 0U;
    uint32_t xIndex = 0U;
    uint32_t expertIdsIndex = 1U;
    uint32_t scalesIndex = 2U;
    uint32_t xActiveMaskIndex = 3U;
    uint32_t expertScalesIndex = 4U;
    uint32_t elasticInfoIndex = 5U;
    uint32_t performanceInfoIndex = 6U;
    bool isMc2Context = false;
};

ge::graphStatus MoeDistributeDispatchA3TilingFuncImpl(gert::TilingContext* context, const DispatchV2Config& config);

}

#endif