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
 * \file expert_sort_base.h
 * \brief
 */
#ifndef EXPERT_SORT_BASE_H
#define EXPERT_SORT_BASE_H

#include "kernel_operator.h"

namespace ExpertDispatch
{
using namespace AscendC;

class ExpertSortBase
{
public:
    __aicore__ inline ExpertSortBase(){};
    __aicore__ inline int64_t GetSyncRound();

protected:
    __aicore__ inline void CleanWSCache();
    __aicore__ inline void SyncAll();

protected:
    TPipe* pipe;
    TQue<QuePosition::VECIN, 1> sortDataCopyInQueue;
    TQue<QuePosition::VECOUT, 1> sortDataCopyOutQueue;
    TBuf<TPosition::VECCALC> tempBuffer;
    TBuf<TPosition::VECCALC> sortedBuffer;

    GlobalTensor<int32_t> expertIdGm;
    GlobalTensor<int32_t> expendedRowIdxGm;
    GlobalTensor<int32_t> sortedExpertForSourceRowGm;
    GlobalTensor<int32_t> expandDstToSrcRowGm;
    GlobalTensor<int32_t> sortedExpertIdGm;
    GlobalTensor<int32_t> expertCountTempGm;

    int64_t tileLength;
    int64_t bufferNum = 1;
    int64_t totalLength;
    int64_t coreNum;

    int64_t expertStart_ = 0;
    int64_t expertEnd_ = 0;
    int64_t n;
    int64_t k;

    static constexpr int64_t SYNC_GM_NUM = 2;
    static constexpr int64_t WORK_GM_NUM = 2;
    static constexpr int64_t DST_BLK_STRIDE = 1;
    static constexpr int64_t DST_REP_STRIDE = 8;
};

__aicore__ inline void ExpertSortBase::SyncAll()
{
    if (coreNum == 1) {
        return;
    }
    AscendC::SyncAll();
}

}  // namespace ExpertDispatch
#endif  // EXPERT_SORT_BASE_H
