/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file mc2_win_context_arch35.h
 * \brief
 */

#ifndef MC2_WIN_CONTEXT_ARCH35
#define MC2_WIN_CONTEXT_ARCH35

#include "moe_distribute_comm_ctx.h"
#include "basic_api/kernel_basic_intf.h"

#ifdef __DAV_C310__
namespace Mc2Kernel {
constexpr uint64_t A5_MTE_STATE_WIN_SIZE = 1024UL * 1024UL;

// A5 implmentation
using HcclOpParam = HcclCombinOpParam;

__aicore__ inline uint32_t GetRankId(__gm__ HcclOpParam * winContext)
{
    return winContext->rankId;
}

__aicore__ inline uint32_t GetRankDim(__gm__ HcclOpParam * winContext)
{
    return winContext->rankDim;
}

__aicore__ inline uint64_t GetWinSize(__gm__ HcclOpParam * winContext)
{
    return winContext->winSize;
}

__aicore__ inline GM_ADDR GetStatusDataSpaceGm(__gm__ HcclOpParam * winContext)
{
    return (GM_ADDR)(winContext->windowsIn[winContext->rankId]);
}

__aicore__ inline GM_ADDR GetBaseWindAddrByRankId(__gm__ HcclOpParam * winContext, const int32_t rankId, const int32_t curRankId)
{
    return (GM_ADDR)(winContext->windowsIn[rankId] + A5_MTE_STATE_WIN_SIZE);
}

__aicore__ inline GM_ADDR GetBaseWindStateAddrByRankId(__gm__ HcclOpParam * winContext, const int32_t rankId, const int32_t curRankId)
{
    return (GM_ADDR)(winContext->windowsIn[rankId]);
}
}
#endif
#endif