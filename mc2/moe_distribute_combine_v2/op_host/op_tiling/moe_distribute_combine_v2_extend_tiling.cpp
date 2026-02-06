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
 * \file moe_distribute_combine_v2_tiling.cpp
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
#include "moe_distribute_combine_tiling_base.h"
#include "tiling/mc2_tiling_utils.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "mc2_log.h"
#include "graph/utils/type_utils.h"
#include "register/op_def_registry.h"
#include "platform/platform_infos_def.h"
#include "../../op_kernel/moe_distribute_combine_tiling.h"
#include "arch35/moe_distribute_combine_tiling_arch35.h"
#include "../../op_kernel/moe_distribute_combine_v2_tiling.h"
#include "../../op_kernel/moe_distribute_combine_v2_tiling_key.h"
#include "mc2_hcom_topo_info.h"

using namespace Mc2Tiling;
using namespace AscendC;
using namespace ge;


using Idx = common_const::IndexExtend;
// CheckInputTensorDim_1<Idx>(cxt, nodeName);


namespace optiling {

static ge::graphStatus MoeDistributeCombineA5ExtendTilingFuncImpl(gert::TilingContext* context)
{
    auto attrs = context->GetAttrs();
    const char *nodeName = context->GetNodeName();
    auto commAlgPtr = attrs->GetAttrPointer<char>(static_cast<int>(IndexExtend::attr::ATTR_COMM_ALG_INDEX));
    // 检查 commAlg 参数合法性校验
    bool isNullOrEmpty = (commAlgPtr == nullptr) || (std::strlen(commAlgPtr) == 0);
    bool isMte = std::strcmp(commAlgPtr, "mte") == 0;
    
    OP_TILING_CHECK(!(isNullOrEmpty || isMte),
        OP_LOGE(nodeName, "Invalid parameter: 'commAlg'='%s'. Only 'mte' is supported."
            "Nullptr and empty char* are also allowed but will be interpreted as 'mte'.", commAlgPtr),
        return ge::GRAPH_FAILED);

    // 默认空指针和空字符走 MTE 方式
    if (isNullOrEmpty) {
        OP_LOGI(nodeName, "Parameter 'commAlg' is nullptr/empty, defaulting to 'mte'.");
    }
    // MTE 调用 A3 tiling 实现
    return MoeDistributeCombineA3TilingFuncImpl(context);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus MoeDistributeCombineV2ExtendTilingFunc(gert::TilingContext* context)
{
    // 不支持 expandX数据类型为int32 type
    auto expandXDesc = context->GetInputDesc(Idx::input::EXPAND_X_INDEX);
    const char *nodeName = context->GetNodeName();
    OP_TILING_CHECK(expandXDesc == nullptr, OP_LOGE(nodeName, "expandxDesc is null."), return ge::GRAPH_FAILED);
    // 检查expandX数据类型为DT_INT32
    OP_TILING_CHECK((expandXDesc->GetDataType() == ge::DT_INT32),
                    OP_LOGE(nodeName, "expandX dataType is invalid, dataType should be bf16 or float16, but is %d",
                    static_cast<ge::DataType>(expandXDesc->GetDataType())), return ge::GRAPH_FAILED);

    std::string socVersion = mc2tiling::GetSocVersion(context);
    ge::graphStatus ret;
    if (socVersion == "Ascend950") {
        ret = MoeDistributeCombineA5ExtendTilingFuncImpl(context);
    } else {
        return ge::GRAPH_FAILED;
    }

    return ret;
}

struct MoeDistributeCombineCompileInfo {};
ge::graphStatus TilingParseForMoeDistributeCombineV2Extend(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MoeDistributeCombineV2Extend)
    .Tiling(MoeDistributeCombineV2ExtendTilingFunc)
    .TilingParse<MoeDistributeCombineCompileInfo>(TilingParseForMoeDistributeCombineV2Extend);
} // namespace optiling