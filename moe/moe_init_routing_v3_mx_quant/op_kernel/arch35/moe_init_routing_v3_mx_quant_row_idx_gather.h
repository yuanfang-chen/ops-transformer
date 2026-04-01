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
 * \file moe_init_routing_v3_mx_quant_row_idx_gather.h
 * \brief RowIdxGather stage for MoeInitRoutingV3MxQuant (GATHER index mode only)
 */
#ifndef MOE_INIT_ROUTING_V3_MX_QUANT_ROW_IDX_GATHER_H
#define MOE_INIT_ROUTING_V3_MX_QUANT_ROW_IDX_GATHER_H

#include "moe_init_routing_v3_mx_quant_common.h"

namespace MoeInitRoutingV3MxQuantNs {
using namespace AscendC;

class RowIdxGather {
public:
    __aicore__ inline RowIdxGather(){};
    __aicore__ inline void Init(GM_ADDR expandedRowIdx, GM_ADDR workspace,
                                const MoeInitRoutingV3MxQuantArch35TilingData *tilingData, TPipe *tPipe);
    __aicore__ inline void Process();

private:
    GlobalTensor<int32_t> expandedRowIdxGm_;
    GlobalTensor<int32_t> sortedExpertIndicesGm_;
    GlobalTensor<int32_t> expertTotalCountGm_;

    TPipe *pipe_;

    TQue<QuePosition::VECIN, 1> sortedExpertIndicesInQueue_;
    TQue<QuePosition::VECOUT, 1> copyOutQueue_;

    const MoeV3Arch35ExpertTokensCountTilingData *expertTokensCountTilingData_;
    int64_t blockIdx_;
    int64_t needCoreNum_;
    int64_t perCoreElements_;
    int64_t curCoreElements_ = 0;
    int64_t actualExpertNum_ = 0;

    int64_t loops_ = 0;
    int64_t perLoopElements_ = 0;
    int64_t lastLoopElements_ = 0;

    int64_t perCoreElements2_;
    int64_t lastCoreElements_;
};

__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_THREAD_NUM) inline void RowIdxGatherComputeSimt(
    int64_t elements, int64_t indexBase,
    __gm__ int32_t *sortedExpertIndicesGmAddr,
    __gm__ int32_t *expandedRowIdxGmAddr)
{
    for (int32_t index = static_cast<int32_t>(Simt::GetThreadIdx()); index < static_cast<int32_t>(elements);
         index += static_cast<int32_t>(Simt::GetThreadNum())) {
        int64_t outIndices = sortedExpertIndicesGmAddr[index];
        expandedRowIdxGmAddr[outIndices] = indexBase + index;
    }
}

__aicore__ inline void RowIdxGather::Init(GM_ADDR expandedRowIdx, GM_ADDR workspace,
                                          const MoeInitRoutingV3MxQuantArch35TilingData *tilingData, TPipe *tPipe)
{
    pipe_ = tPipe;
    expertTokensCountTilingData_ = &(tilingData->expertTokensCountTilingDataOp);
    blockIdx_ = GetBlockIdx();
    needCoreNum_ = expertTokensCountTilingData_->needCoreNum;
    perCoreElements_ = expertTokensCountTilingData_->perCoreElements;
    actualExpertNum_ = tilingData->actualExpertNum;

    expandedRowIdxGm_.SetGlobalBuffer((__gm__ int32_t *)expandedRowIdx);
    if (blockIdx_ < needCoreNum_ - 1) {
        curCoreElements_ = perCoreElements_;
    } else if (blockIdx_ == needCoreNum_ - 1) {
        curCoreElements_ = expertTokensCountTilingData_->lastCoreElements;
    }

    expertTotalCountGm_.SetGlobalBuffer((__gm__ int32_t *)workspace +
                                            Align(tilingData->n * tilingData->k, sizeof(int32_t)) * 2 +
                                            Align(actualExpertNum_, sizeof(int32_t)));
    int64_t expertTotalCount_ = expertTotalCountGm_.GetValue(0);

    perCoreElements2_ = Ceil(expertTotalCount_, needCoreNum_);
    needCoreNum_ = Ceil(expertTotalCount_, perCoreElements2_);
    lastCoreElements_ = expertTotalCount_ - (needCoreNum_ - 1) * perCoreElements2_;
    int64_t perCoreLoops = Ceil(perCoreElements2_, expertTokensCountTilingData_->perCorePerLoopElements);
    int64_t perCorePerLoopElements = Ceil(perCoreElements2_, perCoreLoops);
    int64_t perCoreLastLoopElements = perCoreElements2_ - (perCoreLoops - 1) * perCorePerLoopElements;

    int64_t lastCoreLoops = Ceil(lastCoreElements_, expertTokensCountTilingData_->perCorePerLoopElements);
    int64_t lastCorePerLoopElements = Ceil(lastCoreElements_, lastCoreLoops);
    int64_t lastCoreLastLoopElements = lastCoreElements_ - (lastCoreLoops - 1) * lastCorePerLoopElements;

    loops_ = perCoreLoops;
    if (blockIdx_ == needCoreNum_ - 1) {
        loops_ = lastCoreLoops;
        perLoopElements_ = lastCorePerLoopElements;
        lastLoopElements_ = lastCoreLastLoopElements;
    } else {
        loops_ = perCoreLoops;
        perLoopElements_ = perCorePerLoopElements;
        lastLoopElements_ = perCoreLastLoopElements;
    }

    expandedRowIdxGm_.SetGlobalBuffer((__gm__ int32_t *)expandedRowIdx, actualExpertNum_);
    sortedExpertIndicesGm_.SetGlobalBuffer((__gm__ int32_t *)workspace +
                                               Align(tilingData->n * tilingData->k, sizeof(int32_t)) +
                                               blockIdx_ * perCoreElements2_,
                                           actualExpertNum_);

    pipe_->InitBuffer(sortedExpertIndicesInQueue_, 1, AlignBytes(perLoopElements_, sizeof(int32_t)));
    pipe_->InitBuffer(copyOutQueue_, 1, AlignBytes(1, sizeof(int32_t)));
}

__aicore__ inline void RowIdxGather::Process()
{
    if (blockIdx_ < needCoreNum_) {
        int64_t elements = (blockIdx_ == needCoreNum_ - 1 ? lastCoreElements_ : perCoreElements2_);
        __gm__ int32_t *sortedExpertIndicesGmAddr = (__gm__ int32_t *)sortedExpertIndicesGm_.GetPhyAddr();
        __gm__ int32_t *expandedRowIdxGmAddr = (__gm__ int32_t *)expandedRowIdxGm_.GetPhyAddr();
        Simt::VF_CALL<RowIdxGatherComputeSimt>(Simt::Dim3{SIMT_THREAD_NUM, 1, 1}, elements,
                                                blockIdx_ * perCoreElements2_,
                                                sortedExpertIndicesGmAddr, expandedRowIdxGmAddr);
    }
}

} // namespace MoeInitRoutingV3MxQuantNs
#endif // MOE_INIT_ROUTING_V3_MX_QUANT_ROW_IDX_GATHER_H
