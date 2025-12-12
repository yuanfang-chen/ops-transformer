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
 * \file allto_all_matmul.h
 * \brief kernel内核接口声明
 */
#ifndef ALLTO_ALL_MATMUL_H
#define ALLTO_ALL_MATMUL_H

#include <kernel_operator.h>
#include <kernel_tiling/kernel_tiling.h>
#include "allto_all_matmul_tiling_data.h"

namespace AlltoAllMatmulImpl {

using namespace AscendC;

// 自定义
#define TemplateMC2TypeClass bool Param1, bool Param2, bool Param3
#define TemplateMC2TypeFunc Param1, Param2, Param3

template <AscendC::HardEvent event>
__aicore__ inline void SyncFunc()
{
    int32_t eventID = static_cast<int32_t>(GetTPipePtr()->FetchEventID(event));
    AscendC::SetFlag<event>(eventID);
    AscendC::WaitFlag<event>(eventID);
}

template <TemplateMC2TypeClass>
class AlltoAllMatmul {
public:
    __aicore__ inline AlltoAllMatmul(){};
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR bias, GM_ADDR x1_scale, GM_ADDR x2_scale, GM_ADDR comm_scale, GM_ADDR x1_offset,
                                GM_ADDR x2_offset, GM_ADDR y, GM_ADDR all2all_out, GM_ADDR workspaceGM, TPipe *pipe, const AlltoAllMatmulTilingData *tilingData);
    __aicore__ inline void Process();

private:
};

template <TemplateMC2TypeClass>
__aicore__ inline void AlltoAllMatmul<TemplateMC2TypeFunc>::Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR bias, GM_ADDR x1_scale, GM_ADDR x2_scale, GM_ADDR comm_scale, GM_ADDR x1_offset,
                                                                 GM_ADDR x2_offset, GM_ADDR y, GM_ADDR all2all_out, GM_ADDR workspaceGM, TPipe *pipe,
                                                                 const AlltoAllMatmulTilingData *tilingData)
{
}

template <TemplateMC2TypeClass>
__aicore__ inline void AlltoAllMatmul<TemplateMC2TypeFunc>::Process()
{
}

} // namespace AlltoAllMatmulImpl

#endif // ALLTO_ALL_MATMUL_H