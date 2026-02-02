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
 * \file moe_distribute_dispatch_v2_tiling_helper.h
 * \brief
 */

#ifndef MOE_DISTRIBUTE_DISPATCH_V2_TILING_HELPER
#define MOE_DISTRIBUTE_DISPATCH_V2_TILING_HELPER

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
#include "../../op_kernel/moe_distribute_dispatch_tiling.h"
#include "arch35/moe_distribute_dispatch_tiling_arch35.h"
#include "../../op_kernel/moe_distribute_dispatch_v2_tiling.h"
#include "../../op_kernel/moe_distribute_dispatch_v2_tiling_key.h"
#include "mc2_hcom_topo_info.h"

using namespace Mc2Tiling;
using namespace Mc2Exception;
using namespace AscendC;
using namespace ge;

namespace optiling {

class MoeDistributeDispatchV2TilingHelper(
public:
    MoeDistributeDispatchV2TilingHelper(bool is_context);
    MoeDistributeDispatchA3TilingImpl(gert::TilingContext *context);

    std::string GetGroupEp()
    {
        return groupEp_;
    }
    std::string GetGrouTp()
    {
        return groupTp_;
    }
    uint32_t GetTpWorldSize()
    {
        return tpWorldSize_;
    }
private:

    ge::graphStatus TilingCheckMoeDistributeDispatch(gert::TilingContext *context, const char *nodeName,
        const bool isActiveMask, const bool isScales, const bool hasElasticInfo, const bool isPerformance,
        const uint32_t quantMode);
    bool CheckTensorDim(const gert::TilingContext *context, const char *nodeName,
        const bool isScales, const uint32_t quantMode, const bool isActiveMask, const bool hasElasticInfo,
        const bool isPerformance);
    ge::graphStatus CheckAttrs(const gert::TilingContext *context, const char *nodeName,
        MoeDistributeDispatchV2TilingData &tilingData, uint32_t &localMoeExpertNum, bool isActiveMask,
        bool isSetCommAlg);
    bool CheckInputTensorDim(const gert::TilingContext *context, const char *nodeName,
        const bool isScales, const uint32_t quantMode);
    bool CheckTensorDataTypeNonQuant(const gert::TilingContext *context,
        const char *nodeName, const bool isScales);
    bool CheckTensorDataTypeStaticOrDynamic(
        const gert::TilingContext *context, const char *nodeName, bool isScales);
    bool CheckTensorDataTypeMxfp8(const gert::TilingContext *context, const char *nodeName);
    bool CheckDistinctTensorDataType(const gert::TilingContext *context, const char *nodeName,
        const bool isScales, const uint32_t quantMode);
    bool CheckTensorDataType(const gert::TilingContext *context, const char *nodeName,
        const bool isScales, const uint32_t quantMode, const bool isActiveMask, const bool hasElasticInfo,
        const bool isPerformance);
    bool CheckTensorFormat(const gert::TilingContext *context, const char *nodeName,
        const bool isScales, const uint32_t quantMode, const bool isActiveMask, const uint32_t hasElasticInfo,
        const bool isPerformance);
    



    bool is_context_{false};
    uint32_t context_offset = 0;
    std::string groupEp_{};
    std::string groupTp_{};
    uint32_t tpWorldSize_{0};
);

ge::graphStatus MoeDistributeDispatchA3TilingFuncImpl(gert::TilingContext *context);

ge::graphStatus MoeDistributeDispatchA2TilingFuncImpl(gert::TilingContext *context);

ge::graphStatus MoeDistributeDispatchA5TilingFuncImpl(gert::TilingContext* context);

} // namespace optiling

#endif