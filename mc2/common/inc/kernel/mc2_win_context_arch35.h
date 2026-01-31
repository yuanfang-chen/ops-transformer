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

#include "basic_api/kernel_basic_intf.h"

namespace Mc2Kernel {
constexpr uint32_t HCCL_MTE_MAX_RANK_NUM = 64;
constexpr uint64_t A5_MTE_STATE_WIN_SIZE = 1024UL * 1024UL;

struct HcclCombinOpParam {
    uint64_t workSpace; // client和server之间通信的地址
    uint64_t workSpaceSize; // client和server之间通信的空间大小
    uint32_t rankId; // 当前卡rankId
    uint32_t rankDim; // 总卡数
    uint64_t winSize; // ccu不使用
    uint64_t windowsIn[HCCL_MTE_MAX_RANK_NUM]; // ccu不使用, MTE 数据区
    uint64_t windowsOut[HCCL_MTE_MAX_RANK_NUM]; // ccu不使用，MTE 状态区

    // for ccu
    uint64_t xnAddr; // Xn寄存器起始地址
    uint64_t ckeAddr; // CKE寄存器起始地址
    uint64_t msAddr; // MS地址，预留
    uint64_t msSize; // 可写的MS个数，预留
};

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