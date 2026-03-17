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
 * \file mc2_kernel_utils.h
 * \brief
 */

#ifndef MC2_KERNEL_UTILS_H
#define MC2_KERNEL_UTILS_H

namespace AscendC {

static constexpr uint16_t SYNC_AIC_ONLY_ALL_DET_FLAG = 4; // 用于 AIC 核间同步的 flagId
static constexpr uint16_t SYNC_AIC_AIV_DET_FLAG = 8; // 用于 AIC 与 AIV 核间同步的 flagId
static constexpr uint64_t SYNC_MODE0 = 0; // 核间同步模式 0
static constexpr uint64_t SYNC_MODE2 = 2; // 核间同步模式 2

template<AscendC::HardEvent event>
__aicore__ inline void SyncFunc() {
    AscendC::TEventID eventID = GetTPipePtr()->FetchEventID(event);
    AscendC::SetFlag<event>(eventID);
    AscendC::WaitFlag<event>(eventID);
}

__aicore__ inline void CubeNotifyVector()
{
    // 先全 AIC 同步一次
    CrossCoreSetFlag<SYNC_MODE0, PIPE_FIX>(SYNC_AIC_ONLY_ALL_DET_FLAG);
    CrossCoreWaitFlag(SYNC_AIC_ONLY_ALL_DET_FLAG);
    // 通知 AIV
    CrossCoreSetFlag<SYNC_MODE2, PIPE_FIX>(SYNC_AIC_AIV_DET_FLAG);
}

__aicore__ inline void VecWaitCube()
{
    // 等待 AIC 完成
    CrossCoreWaitFlag<SYNC_MODE2, PIPE_MTE2>(SYNC_AIC_AIV_DET_FLAG);
}

}
#endif // MC2_KERNEL_UTILS_H