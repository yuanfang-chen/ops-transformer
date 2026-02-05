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
 * \file add_rms_norm_dynamic_quant_all_gather_qbmm_gen_task.cpp
 * \brief 静态shape图下沉实现
 */
#include <vector>
#include <platform/platform_info.h>
#include "op_mc2.h"
 
#ifdef BUILD_OPEN_PROJECT
#include "register/op_impl_registry.h"
#include "mc2_gen_task_ops_utils.h"
#include "mc2_gen_task_ops_utils_arch35.h"
#include "mc2_moe_gen_task_ops_utils.h"
#include "mc2_log.h"
#endif

namespace ops {

#ifdef BUILD_OPEN_PROJECT

static ge::Status AddRmsNormDynamicQuantAllGatherQbmmCalcOpParam(gert::ExeResGenerationContext *context)
{
    OPS_LOG_D(context->GetNodeName(), "Do general CalcParam in AddRmsNormDynamicQuantAllGatherQbmm BUILD_OPEN_PROJECT");
    return Mc2GenTaskOpsUtils::CommonKFCMc2CalcParamFunc(context, "aicpu kfc server", "kfc_stream");
}

static ge::Status AddRmsNormDynamicQuantAllGatherQbmmGenTask(const gert::ExeResGenerationContext *context,
                                            std::vector<std::vector<uint8_t>> &tasks)
{
    OPS_LOG_D(context->GetNodeName(), "Do A5 CCU GenTask in AddRmsNormDynamicQuantAllGatherQbmm BUILD_OPEN_PROJECT");
    return Mc2Arch35GenTaskOpsUtils::Mc2Arch35GenTaskCallBack(context, tasks);
}

IMPL_OP(AddRmsNormDynamicQuantAllGatherQbmm)
    .CalcOpParam(AddRmsNormDynamicQuantAllGatherQbmmCalcOpParam)
    .GenerateTask(AddRmsNormDynamicQuantAllGatherQbmmGenTask);

#endif
} // namespace ops
