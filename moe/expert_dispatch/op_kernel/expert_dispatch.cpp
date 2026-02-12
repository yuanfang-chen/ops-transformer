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
 * \file expert_dispatch.cpp
 * \brief
 */
#include "expert_mrgsort_out.h"
#include "expert_mrgsort.h"
#include "expert_sort_one_core.h"
#include "expert_sort_multi_core.h"
#include "expert_tokens_count.h"
#include "expert_gather_out.h"
#include "expert_tokens_count_partial_load.h"
#include "expert_gather_out_partial_load.h"

// 单核排序
#define SORTONECORE_HISTFULL_GATHERFULL 1000000        // 直方图全载，GatherOut全载
#define SORTONECORE_HISTFULL_GATHERPARTIAL 1001000     // 直方图全载，GatherOut非全载
#define SORTONECORE_HISTPARTIAL_GATHERFULL 1010000     // 直方图非全载，GatherOut全载
#define SORTONECORE_HISTPARTIAL_GATHERPARTIAL 1011000  // 直方图非全载，GatherOut非全载

// 多核排序
#define SORTMULTICORE_HISTFULL_GATHERFULL 1100000        // 直方图全载，GatherOut全载
#define SORTMULTICORE_HISTFULL_GATHERPARTIAL 1101000     // 直方图全载，GatherOut非全载
#define SORTMULTICORE_HISTPARTIAL_GATHERFULL 1110000     // 直方图非全载，GatherOut全载
#define SORTMULTICORE_HISTPARTIAL_GATHERPARTIAL 1111000  // 直方图非全载，GatherOut非全载

using namespace AscendC;
using namespace ExpertDispatch;
extern "C" __global__ __aicore__ void expert_dispatch(GM_ADDR x, GM_ADDR expertId, GM_ADDR scale, GM_ADDR dispatchedX,
                                                      GM_ADDR dispatchedRowIdx, GM_ADDR expertTokensCount,
                                                      GM_ADDR expertTotalCount, GM_ADDR dispatchedScale,
                                                      GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    if (workspace == nullptr) {
        return;
    }

    GM_ADDR userWS = GetUserWorkspace(workspace);
    if (userWS == nullptr) {
        return;
    }

    auto t = &tilingData;

    // Sort kernel impl
    TPipe sortPipe;
    if (TILING_KEY_IS(SORTONECORE_HISTFULL_GATHERFULL) || TILING_KEY_IS(SORTONECORE_HISTFULL_GATHERPARTIAL) ||
        TILING_KEY_IS(SORTONECORE_HISTPARTIAL_GATHERFULL) || TILING_KEY_IS(SORTONECORE_HISTPARTIAL_GATHERPARTIAL)) {
        ExpertSortOneCore op;
        op.Init(expertId, dispatchedRowIdx, userWS, t, &sortPipe);
        op.Process();
    } else if (TILING_KEY_IS(SORTMULTICORE_HISTFULL_GATHERFULL) ||
               TILING_KEY_IS(SORTMULTICORE_HISTFULL_GATHERPARTIAL) ||
               TILING_KEY_IS(SORTMULTICORE_HISTPARTIAL_GATHERFULL) ||
               TILING_KEY_IS(SORTMULTICORE_HISTPARTIAL_GATHERPARTIAL)) {
        ExpertSortMultiCore op;
        op.Init(expertId, dispatchedRowIdx, userWS, t, &sortPipe);
        op.Process();
    }
    sortPipe.Destroy();

    // Histogram kernel impl
    TPipe histogramPipe;
    if (TILING_KEY_IS(SORTONECORE_HISTFULL_GATHERFULL) || TILING_KEY_IS(SORTONECORE_HISTFULL_GATHERPARTIAL) ||
        TILING_KEY_IS(SORTMULTICORE_HISTFULL_GATHERFULL) || TILING_KEY_IS(SORTMULTICORE_HISTFULL_GATHERPARTIAL)) {
        ExpertTokensCount countOp;
        countOp.Init(expertTokensCount, expertTotalCount, userWS, t, &histogramPipe); // userWS：排好序的expert_id
        countOp.Process();
    } else if (TILING_KEY_IS(SORTONECORE_HISTPARTIAL_GATHERFULL) ||
               TILING_KEY_IS(SORTONECORE_HISTPARTIAL_GATHERPARTIAL) ||
               TILING_KEY_IS(SORTMULTICORE_HISTPARTIAL_GATHERFULL) ||
               TILING_KEY_IS(SORTMULTICORE_HISTPARTIAL_GATHERPARTIAL)) {
        ExpertTokensCountPartialLoad countOp;
        countOp.Init(expertTokensCount, expertTotalCount, userWS, t, &histogramPipe);
        countOp.Process();
    }
    histogramPipe.Destroy();

    // Gather Out kernel impl
    TPipe gatherPipe;
    if (TILING_KEY_IS(SORTONECORE_HISTFULL_GATHERFULL) || TILING_KEY_IS(SORTONECORE_HISTPARTIAL_GATHERFULL) ||
        TILING_KEY_IS(SORTMULTICORE_HISTFULL_GATHERFULL) || TILING_KEY_IS(SORTMULTICORE_HISTPARTIAL_GATHERFULL)) {
        ExpertGatherOut<DTYPE_X> gatherOp;
        gatherOp.Init(x, scale, expertTotalCount, dispatchedRowIdx, dispatchedX, dispatchedScale, t, &gatherPipe);
        gatherOp.Process();
    } else if (TILING_KEY_IS(SORTONECORE_HISTFULL_GATHERPARTIAL) ||
               TILING_KEY_IS(SORTMULTICORE_HISTFULL_GATHERPARTIAL) ||
               TILING_KEY_IS(SORTONECORE_HISTPARTIAL_GATHERPARTIAL) ||
               TILING_KEY_IS(SORTMULTICORE_HISTPARTIAL_GATHERPARTIAL)) {
        ExpertGatherOutPartialLoad<DTYPE_X> gatherOp;
        gatherOp.Init(x, scale, expertTotalCount, dispatchedRowIdx, dispatchedX, dispatchedScale, t, &gatherPipe);
        gatherOp.Process();
    }
    gatherPipe.Destroy();
}
