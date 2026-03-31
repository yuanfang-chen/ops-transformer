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
 * \file all_gather_add.cpp
 * \brief Kernel 入口文件，包含主函数和调度逻辑
 */

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "all_gather_add.h"

enum class AllGatherAddTilingKey : uint32_t {
    TILING_KEY_MODE_0 = 0,
};

template <uint32_t schMode>
__global__ __aicore__ void all_gather_add(GM_ADDR a, GM_ADDR b, GM_ADDR aGathered, GM_ADDR c, GM_ADDR workspace,
                                          GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(AllGatherAddTilingData);
    GET_TILING_DATA_WITH_STRUCT(AllGatherAddTilingData, tilingData, tiling);
    AscendC::TPipe pipe;

    if constexpr (schMode == static_cast<uint32_t>(AllGatherAddTilingKey::TILING_KEY_MODE_0)) {
        NsAllGatherAdd::AllGatherAdd<half> op;
        op.Init(a, b, aGathered, c, &pipe, &tilingData);
        op.Process();
    }
}
