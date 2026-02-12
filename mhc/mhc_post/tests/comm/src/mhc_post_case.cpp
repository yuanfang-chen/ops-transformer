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
 * \file mhc_post_case.cpp
 * \brief MhcPost 基础测试用例实现.
 */

#include "mhc_post_case.h"
#include "tests/utils/log.h"

using namespace ops::adv::tests::mhc_post;

MhcPostCase::MhcPostCase() : Case() {}

MhcPostCase::MhcPostCase(const char *name, bool enable, const char *dbgInfo,
                         ops::adv::tests::utils::OpInfo opInfo, ops::adv::tests::utils::Param,
                         int32_t tilingTemplatePriority)
    : Case(name, enable, dbgInfo, std::move(opInfo), tilingTemplatePriority)
{
}

bool MhcPostCase::Run()
{
    if (!mEnable) {
        return true;
    }
    if (!mOpInfo.ProcessTiling(mName)) {
        return false;
    }
    if (!mOpInfo.ProcessKernel(mName)) {
        return false;
    }
    return true;
}

bool MhcPostCase::InitParam()
{
    return true;
}

bool MhcPostCase::InitOpInfo()
{
    return true;
}