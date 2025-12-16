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
 * \file moe_distribute_combine_v2_gen_task.cpp
 * \brief
 */
#include <vector>
#include <set>
#include <string>

#include "op_mc2.h"
#include "platform/platform_info.h"

#ifdef BUILD_OPEN_PROJECT
#include "mc2_gen_task_ops_utils.h"
#include "mc2_moe_gen_task_ops_utils.h"
#include "graph/arg_desc_info.h"
#include "graph/kernel_launch_info.h"
#include "register/op_impl_registry.h"
#include "mc2_log.h"
#else
#include "ops_error.h"
#include "mc2_gen_task_moe.h"
#include "mc2_gen_task_utils.h"
#include "mc2_a5_gen_task_utils.h"
#include "register/op_ct_impl_registry.h"
#include "register/op_ext_gentask_registry.h"
#endif

namespace ops {

static bool IsPlatform910B(const char *nodeName)
{
    fe::PlatFormInfos platform_info;
    fe::OptionalInfos optional_info;
    if (fe::PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platform_info, optional_info) !=
        ge::GRAPH_SUCCESS) {
        OPS_LOG_E(nodeName, "Cannot get platform info!");
        return false;
    }
    static std::set<std::string> supported_soc = {"Ascend910B"};
    std::string short_soc_version;
    if (!platform_info.GetPlatformRes("version", "Short_SoC_version", short_soc_version) || short_soc_version.empty()) {
        OPS_LOG_E(nodeName, "Cannot get short soc version!");
        return false;
    }
    OPS_LOG_D(nodeName, "Get soc version: %s", short_soc_version.c_str());
    return supported_soc.count(short_soc_version) > 0;
}

#ifdef BUILD_OPEN_PROJECT
ge::Status MoeDistributeCombineV2CalcParamFunc(gert::ExeResGenerationContext *context)
{
    const ge::AscendString name = "aicpu kfc server";
    const ge::AscendString reuseKey = "kfc_stream";
    return Mc2GenTaskOpsUtils::CommonKFCMc2CalcParamFunc(context, name, reuseKey);
}

ge::Status MoeDistributeCombineV2GenTaskFunc(const gert::ExeResGenerationContext *context,
                                            std::vector<std::vector<uint8_t>> &tasks)
{
    const char *nodeName = context->GetNodeName();
    OPS_LOG_I(nodeName, "MC2 Generate task start.");
    if (IsPlatform910B(nodeName)) {
        return Mc2MoeGenTaskOpsUtils::Mc2MoeGenTaskCallback(context, tasks);
    }
    return Mc2MoeGenTaskOpsUtils::Mc2MoeGenTaskCallbackV2(context, tasks);
}

// new ver
IMPL_OP(MoeDistributeCombineV2)
    .CalcOpParam(MoeDistributeCombineV2CalcParamFunc)
    .GenerateTask(MoeDistributeCombineV2GenTaskFunc);
#else // mc2 gen task utils
static const size_t ATTR_INDEX_COMM_ALG = 14;
ge::Status MoeDistributeCombineV2CalcParamFunc(gert::ExeResGenerationContext *context)
{
    if ((Mc2A5GenTaskUtils::IsTargetPlatform(context->GetNodeName(), PLATFORM_A5)) &&
        (Mc2A5GenTaskUtils::GetCommAlg(context, ATTR_INDEX_COMM_ALG) != COMM_ALG_MTE)) {
        OPS_LOG_D(context->GetNodeName(), "Do A5 ccu calc param.");
        return Mc2GenTaskUtils::CommonKFCMc2CalcParamFunc(context, "ccu server", "ccu_stream");
    }
    const ge::AscendString name = "aicpu kfc server";
    const ge::AscendString reuseKey = "kfc_stream";
    return Mc2GenTaskUtils::CommonKFCMc2CalcParamFunc(context, name, reuseKey);
}

ge::Status MoeDistributeCombineV2GenTaskFunc(const gert::ExeResGenerationContext *context,
                                            std::vector<std::vector<uint8_t>> &tasks)
{
    const char *nodeName = context->GetNodeName();
    if (IsPlatform910B(nodeName)) {
        return Mc2GenTaskUtils::CommonKFCMc2GenTask(context, tasks, Mc2GenTaskMoe::Mc2MoeGenTaskCallback);
    } else if (Mc2A5GenTaskUtils::IsTargetPlatform(context->GetNodeName(), PLATFORM_A5)) {
        const std::string commAlg = Mc2A5GenTaskUtils::GetCommAlg(context, ATTR_INDEX_COMM_ALG);
        if (commAlg == COMM_ALG_MTE) {
            OPS_LOG_D(context->GetNodeName(), "Do A5 mte gen task.");
            return Mc2GenTaskUtils::CommonKFCMc2GenTask(context, tasks, Mc2GenTaskMoe::Mc2MoeGenTaskCallbackV2);
        } else if (commAlg == COMM_ALG_CCU) {
            OPS_LOG_D(context->GetNodeName(), "Do A5 CCU gen task.");
            return Mc2GenTaskUtils::CommonKFCMc2GenTask(context, tasks, Mc2A5GenTaskUtils::Mc2GenTaskCallBack910A5);
        } else {
            OPS_LOG_E(context->GetNodeName(), "Got unsupported commAlg %s.", commAlg.c_str());
            return ge::GRAPH_FAILED;
        }
    }
    OPS_LOG_D(context->GetNodeName(), "Do A3 gen task.");
    return Mc2GenTaskUtils::CommonKFCMc2GenTask(context, tasks, Mc2GenTaskMoe::Mc2MoeGenTaskCallbackV2);
}

IMPL_OP_CT(MoeDistributeCombineV2)
    .CalcOpParam(MoeDistributeCombineV2CalcParamFunc)
    .GenerateTask(MoeDistributeCombineV2GenTaskFunc);
REGISTER_EXT_TASK_TYPE(MoeDistributeCombineV2, fe::ExtTaskType::kAicoreTask);
#endif
} // namespace ops
