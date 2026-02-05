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
 * \file moe_distribute_dispatch_v3_tiling.cpp
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

#include "tiling/mc2_tiling_utils.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "mc2_log.h"
#include "mc2_exception_dump.h"
#include "graph/utils/type_utils.h"
#include "register/op_def_registry.h"
#include "platform/platform_infos_def.h"
#include "../../../moe_distribute_dispatch_v2/op_host/op_tiling/moe_distribute_dispatch_tiling_v2.h"
#include "../../../moe_distribute_dispatch_v2/op_kernel/moe_distribute_dispatch_v2_tiling.h"
#include "mc2_hcom_topo_info.h"

using namespace Mc2Tiling;
using namespace Mc2Exception;
using namespace AscendC;
using namespace ge;


namespace optiling {
static ge::graphStatus MoeDistributeDispatchV3TilingFunc(gert::TilingContext* context)
{
    DispatchV2Config config;
    config.contextIndex = 0U;
    config.xIndex = 1U;
    config.expertIdsIndex = 2U;
    config.scalesIndex = 3U;
    config.xActiveMaskIndex = 4U;
    config.expertScalesIndex = 5U;
    config.elasticInfoIndex = 6U;
    config.performanceInfoIndex = 7U;
    config.attrEpWorldSizeIndex = 0;
    config.attrEpRankIdIndex = 1;
    config.attrMoeExpertNumIndex = 2;
    config.attrCclBufferSizeIndex = 3;
    config.attrTpWorldSizeIndex = 4;
    config.attrTpRankIdIndex = 5;
    config.attrExpertSharedTypeIndex = 6;
    config.attrSharedExpertNumIndex = 7;
    config.attrSharedExpertRankNumIndex = 8;
    config.attrQuantModeIndex = 9;
    config.attrGlobalBsIndex = 10;
    config.attrExpertTokenNumsTypeIndex = 11;
    config.attrCommAlgIndex = 12;
    config.attrZeroExpertNumIndex = 13;
    config.attrCopyExpertNumIndex = 14;
    config.attrConstExpertNumIndex = 15;
    config.isMc2Context = true;
    OP_LOGD("MoeDistributeDispatchV3", "Enter MoeDistributeDispatchV3 tiling");
    ge::graphStatus ret = MoeDistributeDispatchA3TilingFuncImplPublic(context, config);
    return ret;
}

struct MoeDistributeDispatchCompileInfo {};
static ge::graphStatus TilingParseForMoeDistributeDispatchV3(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MoeDistributeDispatchV3)
    .Tiling(MoeDistributeDispatchV3TilingFunc)
    .TilingParse<MoeDistributeDispatchCompileInfo>(TilingParseForMoeDistributeDispatchV3);

// Register exception func
inline void MoeDistributeDispatchV3ExceptionImplWrapper(aclrtExceptionInfo *args, void *userdata)
{
    Mc2ExceptionImpl(args, userdata, "MoeDistributeDispatchV3");
}

IMPL_OP(MoeDistributeDispatchV3)
    .ExceptionDumpParseFunc(MoeDistributeDispatchV3ExceptionImplWrapper);
} // namespace optiling