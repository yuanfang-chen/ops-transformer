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
 * \file expert_tokens_count_partial_load.h
 * \brief
 */
#ifndef EXPERT_TOKENS_COUNT_PARTIAL_LOAD_H
#define EXPERT_TOKENS_COUNT_PARTIAL_LOAD_H

#include "common.h"
#include "kernel_operator.h"

namespace ExpertDispatch
{
using namespace AscendC;

class ExpertTokensCountPartialLoad
{
public:
    __aicore__ inline ExpertTokensCountPartialLoad(){};
    __aicore__ inline void Init(GM_ADDR expertTokensCount, GM_ADDR expertTotalCount, GM_ADDR sortedExpertId,
                                const ExpertDispatchTilingData* tilingData, TPipe* tPipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t loop, int64_t curLoopElements);
    __aicore__ inline void Compute(int64_t curLoopElements);
    __aicore__ inline void CopyOut();

    __aicore__ inline void expertCountCopyIn();
    __aicore__ inline void expertCountCompute();
    __aicore__ inline void expertCountCopyOut();

private:
    GlobalTensor<int32_t> sortedExpertIdGm_;
    GlobalTensor<int32_t> expertTokensCountGm_;
    GlobalTensor<int32_t> expertTotalCountGm_;
    TPipe* pipe_;

    TQue<QuePosition::VECIN, 1> sortedExpertIdInQueue_;
    TQue<QuePosition::VECOUT, 1> expertIdCountOutQueue_;
    TQue<QuePosition::VECIN, 1> expertIdCountInQueue_;
    TQue<QuePosition::VECOUT, 1> expertTotalCountQueue_;

    const ExpertTokensCountTilingData* expertTokensCountTilingData_;
    int64_t blockIdx_;
    int64_t needCoreNum_;
    int64_t perCoreElements_;
    int64_t curCoreElements_ = 0;
    int64_t expertStart_ = 0;
    int64_t expertEnd_ = 0;
    int64_t actualExpertNum_ = 0;
    int64_t coreLoopsNum_ = 0;
    int64_t perCorePerLoopElements_ = 0;
    int64_t perCoreLastLoopElements_ = 0;
    int64_t actualExpertTotalNum_ = 0;
};

__aicore__ inline void ExpertTokensCountPartialLoad::Init(GM_ADDR expertTokensCount, GM_ADDR expertTotalCount,
                                                          GM_ADDR sortedExpertId,
                                                          const ExpertDispatchTilingData* tilingData, TPipe* tPipe)
{
    pipe_ = tPipe;
    expertTokensCountTilingData_ = &(tilingData->expertTokensCountTilingDataOp);
    blockIdx_ = GetBlockIdx();
    needCoreNum_ = expertTokensCountTilingData_->needCoreNum;
    perCoreElements_ = expertTokensCountTilingData_->perCoreElements;
    expertStart_ = tilingData->expertStart;
    expertEnd_ = tilingData->expertEnd;
    actualExpertNum_ = tilingData->actualExpertNum;

    if (blockIdx_ < needCoreNum_ - 1) {
        curCoreElements_ = expertTokensCountTilingData_->perCoreElements;
        coreLoopsNum_ = expertTokensCountTilingData_->perCoreLoops;
        perCorePerLoopElements_ = expertTokensCountTilingData_->perCorePerLoopElements;
        perCoreLastLoopElements_ = expertTokensCountTilingData_->perCoreLastLoopElements;
    } else if (blockIdx_ == needCoreNum_ - 1) {
        curCoreElements_ = expertTokensCountTilingData_->lastCoreElements;
        coreLoopsNum_ = expertTokensCountTilingData_->lastCoreLoops;
        perCorePerLoopElements_ = expertTokensCountTilingData_->lastCorePerLoopElements;
        perCoreLastLoopElements_ = expertTokensCountTilingData_->lastCoreLastLoopElements;
    }

    // 学员补充： GM/UB 初始化
    sortedExpertIdGm_.SetGlobalBuffer((__gm__ int32_t*)sortedExpertId + blockIdx_ * perCoreElements_, curCoreElements_);
    expertTokensCountGm_.SetGlobalBuffer((__gm__ int32_t*)expertTokensCount, actualExpertNum_);
    expertTotalCountGm_.SetGlobalBuffer((__gm__ int32_t*)expertTotalCount, actualExpertNum_);

    int64_t sortedExpertIdInLen = Max(perCorePerLoopElements_, perCoreLastLoopElements_);
    pipe_->InitBuffer(sortedExpertIdInQueue_, 1, AlignBytes(sortedExpertIdInLen, sizeof(int32_t)));
    pipe_->InitBuffer(expertIdCountOutQueue_, 1, AlignBytes(actualExpertNum_, sizeof(int32_t)));
    pipe_->InitBuffer(expertIdCountInQueue_, 1, AlignBytes(actualExpertNum_, sizeof(int32_t)));
    pipe_->InitBuffer(expertTotalCountQueue_, 1, AlignBytes(1, sizeof(int32_t)));
    // 补充结束

}

__aicore__ inline void ExpertTokensCountPartialLoad::Process()
{
    // 计算直方图
    if (blockIdx_ < needCoreNum_) {
        for (int64_t i = 0; i < coreLoopsNum_; i++) {
            int64_t perLoopElements = (i == (coreLoopsNum_ - 1)) ? perCoreLastLoopElements_ : perCorePerLoopElements_;
            CopyIn(i, perLoopElements);
            Compute(perLoopElements);
            CopyOut();
        }
    }

    // 计算有效专家（直方图）总数
    SyncAll();
    /* copy expert tokens count result from worksapce to output GM. */
    if (blockIdx_ == 0) {
        expertCountCopyIn();
        expertCountCompute();
        expertCountCopyOut();
    }
    SyncAll();
}

__aicore__ inline void ExpertTokensCountPartialLoad::CopyIn(int64_t loop, int64_t curLoopElements)
{
    // 学员补充：copyIn排序专家
    LocalTensor<int32_t> sortedExpertIdInLocal = sortedExpertIdInQueue_.AllocTensor<int32_t>();
    DataCopyExtParams dataCopyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(curLoopElements * sizeof(int32_t)),
                                     0, 0, 0};
    DataCopyPadExtParams dataCopyPadParams{false, 0, 0, 0};
    int64_t sortedExpertIdOffset = loop * perCorePerLoopElements_;
    DataCopyPad(sortedExpertIdInLocal, sortedExpertIdGm_[sortedExpertIdOffset], dataCopyParams, dataCopyPadParams);
    sortedExpertIdInQueue_.EnQue(sortedExpertIdInLocal);
    // 补充结束
}

__aicore__ inline void ExpertTokensCountPartialLoad::Compute(int64_t curLoopElements)
{
    LocalTensor<int32_t> sortedExpertIdInLocal = sortedExpertIdInQueue_.DeQue<int32_t>();
    LocalTensor<int32_t> expertCountOutLocal = expertIdCountOutQueue_.AllocTensor<int32_t>();
    Duplicate(expertCountOutLocal.ReinterpretCast<int32_t>(), static_cast<int32_t>(0),
              static_cast<int32_t>(actualExpertNum_));
    SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);
    // 学员补充: 计算单核直方图，写到expertCountOutLocal
    int64_t i = 0;
    int32_t lastExpertId = sortedExpertIdInLocal.GetValue(0);
    int32_t lastIndex = 0;
    for (i = 1; i < curLoopElements; i++) {
        if ((lastExpertId >= expertEnd_) || (lastExpertId < expertStart_)) {
            break;
        }
        int32_t curExpertId = sortedExpertIdInLocal.GetValue(i);
        if (curExpertId != lastExpertId || curExpertId >= expertEnd_) {
            expertCountOutLocal.SetValue(lastExpertId - expertStart_, i - lastIndex);
            lastIndex = i;
            lastExpertId = curExpertId;
        }
    }
    if ((i == curLoopElements) && ((lastExpertId >= expertStart_) && (lastExpertId < expertEnd_))) {
        expertCountOutLocal.SetValue(lastExpertId - expertStart_, i - lastIndex);
    }
    // 补充结束

    SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
    expertIdCountOutQueue_.EnQue<int32_t>(expertCountOutLocal);
    sortedExpertIdInQueue_.FreeTensor(sortedExpertIdInLocal);
}

__aicore__ inline void ExpertTokensCountPartialLoad::CopyOut()
{
    LocalTensor<int32_t> expertCountOutLocal = expertIdCountOutQueue_.DeQue<int32_t>();
    DataCopyExtParams copyParams{static_cast<uint16_t>(1), static_cast<uint32_t>((actualExpertNum_) * sizeof(int32_t)),
                                 0, 0, 0};
    // 计算同时copyOut最终直方图
    SetAtomicAdd<int32_t>();
    DataCopyPad(expertTokensCountGm_, expertCountOutLocal, copyParams);
    SetAtomicNone();
    expertIdCountOutQueue_.FreeTensor(expertCountOutLocal);
}

__aicore__ inline void ExpertTokensCountPartialLoad::expertCountCopyIn()
{
    // 学员补充：copyIn直方图
    LocalTensor<int32_t> expertCountInLocal = expertIdCountInQueue_.AllocTensor<int32_t>();
    DataCopyExtParams dataCopyParams{static_cast<uint16_t>(1),
                                     static_cast<uint32_t>((actualExpertNum_) * sizeof(int32_t)), 0, 0, 0};
    DataCopyPadExtParams dataCopyPadParams{false, 0, 0, 0};
    DataCopyPad(expertCountInLocal, expertTokensCountGm_, dataCopyParams, dataCopyPadParams);
    expertIdCountInQueue_.EnQue(expertCountInLocal);
    // 补充结束
}

__aicore__ inline void ExpertTokensCountPartialLoad::expertCountCompute()
{
    // 学员补充：计算专家总数，推荐使用 ReduceSum
    LocalTensor<int32_t> expertCountInLocal = expertIdCountInQueue_.DeQue<int32_t>();
    LocalTensor<int32_t> expertTotalCountLocal = expertTotalCountQueue_.AllocTensor<int32_t>();
    LocalTensor<float> expertTotalCountLocalFP32 = expertTotalCountLocal.ReinterpretCast<float>();
    LocalTensor<float> expertCountTempInLocalFP32 = expertCountInLocal.ReinterpretCast<float>();
    Cast(expertCountTempInLocalFP32, expertCountInLocal, AscendC::RoundMode::CAST_RINT, actualExpertNum_);
    Cast(expertTotalCountLocalFP32, expertTotalCountLocal, AscendC::RoundMode::CAST_RINT, actualExpertNum_);
    ReduceSum(expertTotalCountLocalFP32, expertCountTempInLocalFP32, expertCountTempInLocalFP32, actualExpertNum_);
    Cast(expertTotalCountLocal, expertTotalCountLocalFP32, AscendC::RoundMode::CAST_RINT, actualExpertNum_);

    expertTotalCountQueue_.EnQue<int32_t>(expertTotalCountLocal);
    expertIdCountInQueue_.FreeTensor(expertCountInLocal);
    // 补充结束
}

__aicore__ inline void ExpertTokensCountPartialLoad::expertCountCopyOut()
{
    // 学员补充：copyOut 专家总数
    LocalTensor<int32_t> expertTotalCountLocal = expertTotalCountQueue_.DeQue<int32_t>();
    DataCopyExtParams copyParams{static_cast<uint16_t>(1), sizeof(int32_t), 0, 0, 0};
    DataCopyPad(expertTotalCountGm_, expertTotalCountLocal, copyParams);
    expertTotalCountQueue_.FreeTensor(expertTotalCountLocal);
    // 补充结束
}

}  // namespace ExpertDispatch
#endif  // EXPERT_TOKENS_COUNT_PARTIAL_LOAD_H
