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

#include <cstdint>
#include "tiling/tiling_api.h"
#include "graph/utils/type_utils.h"
#include "register/tilingdata_base.h"
#include "tiling_base/tiling_base.h"
#include "tiling/mc2_opversion_manager.h"
using namespace Ops::Transformer::OpTiling;

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
    uint32_t attrGroupEpIndex = 0;
    uint32_t attrEpWorldSizeIndex = 1;
    uint32_t attrEpRankIdIndex = 2;
    uint32_t attrMoeExpertNumIndex = 3;
    uint32_t attrCclBufferSizeIndex = 3;
    uint32_t attrGroupTpIndex = 4;
    uint32_t attrTpWorldSizeIndex = 5;
    uint32_t attrTpRankIdIndex = 6;
    uint32_t attrExpertSharedTypeIndex = 7;
    uint32_t attrSharedExpertNumIndex = 8;
    uint32_t attrSharedExpertRankNumIndex = 9;
    uint32_t attrQuantModeIndex = 10;
    uint32_t attrGlobalBsIndex = 11;
    uint32_t attrExpertTokenNumsTypeIndex = 12;
    uint32_t attrCommAlgIndex = 13;
    uint32_t attrZeroExpertNumIndex = 14;
    uint32_t attrCopyExpertIndex = 15;
    uint32_t attrConstExpertNumIndex = 16;
    bool isMc2Context = false;
};

ge::graphStatus MoeDistributeDispatchA3TilingFuncImplPublic(gert::TilingContext* context, DispatchV2Config& config);

}

#endif