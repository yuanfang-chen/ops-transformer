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
 * \file moe_distribute_combine_add_rms_norm_tiling.cpp
 * \brief
 */

#include <queue>
#include <vector>
#include <dlfcn.h>
#include <fcntl.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/types.h>
#include <unistd.h>
#include <cmath>
#include <cstdint>
#include <string>
#include <type_traits>

#include "tiling/mc2_tiling_utils.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "mc2_log.h"
#include "graph/utils/type_utils.h"
#include "register/op_def_registry.h"
#include "platform/platform_infos_def.h"
#include "../../../moe_distribute_combine_v2/op_kernel/moe_distribute_combine_v2_tiling.h"
#include "mc2_hcom_topo_info.h"
#include "../../../moe_distribute_combine_v2/op_host/op_tiling/moe_distribute_combine_tiling_helper.h"

using namespace AscendC;
using namespace ge;
using namespace Mc2Tiling;

namespace optiling {
static ge::graphStatus MoeDistributeCombineV3TilingFunc(gert::TilingContext* context)
{
    CombineV2Config config;
    config.contextIndex = 0;
    config.expandXIndex = 1;
    config.expertIdsIndex = 2;
    config.assistInfoIndex = 3;
    config.epSendCountIndex = 4;
    config.expertScalesIndex = 5;
    config.tpSendCountsIndex = 6; // 根据combineV2算子原型标志位设置tpSendCounts索引
    config.xActiveMaskIndex = 7; // 根据combineV2算子原型标志位设置xActiveMask索引
    config.activationScaleIndex = 8; // 根据combineV2算子原型标志位设置activationScale索引
    config.weightScaleIndex = 9; // 根据combineV2算子原型标志位设置weightScale索引
    config.groupListIndex = 10; // 根据combineV2算子原型标志位设置groupList索引
    config.sharedExpertXIndex = 12; // 根据combineV2算子原型标志位设置sharedExpertX索引
    config.elasticInfoIndex = 13; // 根据combineV2算子原型标志位设置elasticInfo索引
    config.oriXIndex = 14; // 根据combineV2算子原型标志位设置oriX索引
    config.constExpertAlpha1Index = 15; // 根据combineV2算子原型标志位设置constExpertAlpha1索引
    config.constExpertAlpha2Index = 16; // 根据combineV2算子原型标志位设置constExpertAlpha2索引
    config.constExpertVIndex = 17; // 根据combineV2算子原型标志位设置constExpertV索引
    config.performanceInfoIndex = 18; // 根据combineV2算子原型标志位设置performanceInfo索引
    config.outputXIndex = 0; // 根据combineV2算子原型标志位设置outputX索引
    config.attrEpWorldSizeIndex = 0;
    config.attrEpRankIdIndex = 1;
    config.attrMoeExpertNumIndex = 2;
    config.attrCclBufferSizeIndex = 3;
    config.attrTpWorldSizeIndex = 4;
    config.attrTpRankIdIndex = 5;
    config.attrExpertSharedTypeIndex = 6;
    config.attrSharedExpertNumIndex = 7;
    config.attrSharedExpertRankNumIndex = 8;
    config.attrGlobalBsIndex  = 9;
    config.attrOutDTypeIndex = 10;
    config.attrCommQuantModeIndex = 11;
    config.attrGroupListTypeIndex = 12;
    config.attrCommAlgIndex = 13;
    config.attrZeroExpertNumIndex = 14; // 根据combineV2算子原型标志位设置attrZeroExpertNum索引
    config.attrCopyExpertNumIndex = 15; // 根据combineV2算子原型标志位设置attrCopyExpertNum索引
    config.attrConstExpertNumIndex = 16; // 根据combineV2算子原型标志位设置attrConstExpertNum索引
    config.hasAddRmsNorm = false;
    config.isMc2Context = true;

    const char *nodeName = context->GetNodeName();
    OP_LOGD(nodeName, "Enter MoeDistributeDispatchV3 tiling");
    ge::graphStatus ret = optiling::MoeDistributeCombineV2TilingFuncNew(context, config);
    return ret;
}

struct MoeDistributeCombineV3CompileInfo {};
ge::graphStatus TilingParseForMoeDistributeCombineV3(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MoeDistributeCombineV3)
    .Tiling(MoeDistributeCombineV3TilingFunc)
    .TilingParse<MoeDistributeCombineV3CompileInfo>(TilingParseForMoeDistributeCombineV3);
} // namespace optiling