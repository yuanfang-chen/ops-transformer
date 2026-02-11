/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file expert_tokens_count.h
 * \brief
 */
#ifndef EXPERT_TOKENS_COUNT_H
#define EXPERT_TOKENS_COUNT_H

#include "common.h"
#include "kernel_operator.h"

namespace ExpertDispatch
{
using namespace AscendC;

class ExpertTokensCount
{
public:
    __aicore__ inline ExpertTokensCount(){};
    __aicore__ inline void Init(GM_ADDR expertTokensCount, GM_ADDR expertTotalCount, GM_ADDR sortedExpertId,
                                const ExpertDispatchTilingData* tilingData, TPipe* tPipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn();
    __aicore__ inline void Compute();
    __aicore__ inline void CopyOut();

    __aicore__ inline void expertCountCopyIn();
    __aicore__ inline void expertCountCompute();
    __aicore__ inline void expertCountCopyOut();

private:
    GlobalTensor<int32_t> sortedExpertIdGm_;     // sorted expert id
    GlobalTensor<int32_t> expertTokensCountGm_;  // expert id histogram
    GlobalTensor<int32_t> expertTotalCountGm_;   // expert total number count in range
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
};

__aicore__ inline void ExpertTokensCount::Init(GM_ADDR expertTokensCount, GM_ADDR expertTotalCount,
                                               GM_ADDR sortedExpertId, const ExpertDispatchTilingData* tilingData,
                                               TPipe* tPipe)
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
        curCoreElements_ = perCoreElements_;
    } else if (blockIdx_ == needCoreNum_ - 1) {
        curCoreElements_ = expertTokensCountTilingData_->lastCoreElements;
    }

    sortedExpertIdGm_.SetGlobalBuffer((__gm__ int32_t*)sortedExpertId + blockIdx_ * perCoreElements_, curCoreElements_);
    pipe_->InitBuffer(sortedExpertIdInQueue_, 1, AlignBytes(curCoreElements_, sizeof(int32_t)));
    // 学员补充： 剩余 GM/UB 初始化
    expertTokensCountGm_.SetGlobalBuffer((__gm__ int32_t*)expertTokensCount, actualExpertNum_);
    expertTotalCountGm_.SetGlobalBuffer((__gm__ int32_t*)expertTotalCount, 1);

    pipe_->InitBuffer(expertIdCountOutQueue_, 1, AlignBytes(actualExpertNum_, sizeof(int32_t)));
    pipe_->InitBuffer(expertIdCountInQueue_, 1, AlignBytes(actualExpertNum_, sizeof(int32_t)));
    pipe_->InitBuffer(expertTotalCountQueue_, 1, AlignBytes(1, sizeof(int32_t)));
    // 补充结束
}

__aicore__ inline void ExpertTokensCount::Process()
{
    // 多核计算直方图
    if (blockIdx_ < needCoreNum_) {
        CopyIn();
        Compute();
        CopyOut();
    }
    SyncAll();
    // 单核计算有效专家（直方图）总数
    if (blockIdx_ == 0) {
        expertCountCopyIn();
        expertCountCompute();
        expertCountCopyOut();
    }
    SyncAll();
}

__aicore__ inline void ExpertTokensCount::CopyIn()
{
    // 学员补充：copyIn排序专家，推荐使用DataCopyPad，参考expert_sort_one_core.h ExpertSortOneCore::CopyIn()
    LocalTensor<int32_t> sortedExpertIdInLocal = sortedExpertIdInQueue_.AllocTensor<int32_t>();
    DataCopyExtParams dataCopyParams{static_cast<uint16_t>(1),
                                     static_cast<uint32_t>(curCoreElements_ * sizeof(int32_t)), 0, 0, 0};
    DataCopyPadExtParams dataCopyPadParams{false, 0, 0, 0};
    DataCopyPad(sortedExpertIdInLocal, sortedExpertIdGm_, dataCopyParams, dataCopyPadParams);
    sortedExpertIdInQueue_.EnQue(sortedExpertIdInLocal);
    // 补充结束
}

__aicore__ inline void ExpertTokensCount::Compute()
{
    LocalTensor<int32_t> sortedExpertIdInLocal = sortedExpertIdInQueue_.DeQue<int32_t>();
    LocalTensor<int32_t> expertCountOutLocal = expertIdCountOutQueue_.AllocTensor<int32_t>();
    Duplicate(expertCountOutLocal.ReinterpretCast<int32_t>(), static_cast<int32_t>(0),
              static_cast<int32_t>(actualExpertNum_));
    SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);
    // 学员补充：每个核单独计算直方图，写到expertCountOutLocal
    int64_t i = 0;
    int32_t lastExpertId = sortedExpertIdInLocal.GetValue(0);
    int32_t lastIndex = 0;
    for (i = 1; i < curCoreElements_; i++) {
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
    if ((i == curCoreElements_) && ((lastExpertId >= expertStart_) && (lastExpertId < expertEnd_))) {
        expertCountOutLocal.SetValue(lastExpertId - expertStart_, i - lastIndex);
    }
    // 补充结束

    SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
    expertIdCountOutQueue_.EnQue<int32_t>(expertCountOutLocal);
    sortedExpertIdInQueue_.FreeTensor(sortedExpertIdInLocal);
}

__aicore__ inline void ExpertTokensCount::CopyOut()
{
    LocalTensor<int32_t> expertCountOutLocal = expertIdCountOutQueue_.DeQue<int32_t>();
    DataCopyExtParams copyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(actualExpertNum_ * sizeof(int32_t)),
                                 0, 0, 0};
    // copyOut同时计算最终直方图
    SetAtomicAdd<int32_t>();
    DataCopyPad(expertTokensCountGm_, expertCountOutLocal, copyParams);
    SetAtomicNone();
    expertIdCountOutQueue_.FreeTensor(expertCountOutLocal);
}

__aicore__ inline void ExpertTokensCount::expertCountCopyIn()
{
    // 学员补充：copyIn前序计算的直方图，推荐使用DataCopyPad
    LocalTensor<int32_t> expertCountInLocal = expertIdCountInQueue_.AllocTensor<int32_t>();
    DataCopyExtParams dataCopyParams{static_cast<uint16_t>(1),
                                     static_cast<uint32_t>((actualExpertNum_) * sizeof(int32_t)), 0, 0, 0};
    DataCopyPadExtParams dataCopyPadParams{false, 0, 0, 0};
    DataCopyPad(expertCountInLocal, expertTokensCountGm_, dataCopyParams, dataCopyPadParams);
    expertIdCountInQueue_.EnQue(expertCountInLocal);
    // 补充结束
}

__aicore__ inline void ExpertTokensCount::expertCountCompute()
{
    // 学员补充：计算专家总数，推荐使用ReduceSum，注意类型转换
    LocalTensor<int32_t> expertCountInLocal = expertIdCountInQueue_.DeQue<int32_t>();
    LocalTensor<int32_t> expertTotalCountLocal = expertTotalCountQueue_.AllocTensor<int32_t>();
    LocalTensor<float> expertTotalCountLocalFP32 = expertTotalCountLocal.ReinterpretCast<float>();
    LocalTensor<float> expertCountTempInLocalFP32 = expertCountInLocal.ReinterpretCast<float>();
    Cast(expertCountTempInLocalFP32, expertCountInLocal, AscendC::RoundMode::CAST_RINT, actualExpertNum_);
    Cast(expertTotalCountLocalFP32, expertTotalCountLocal, AscendC::RoundMode::CAST_RINT, 1);
    ReduceSum(expertTotalCountLocalFP32, expertCountTempInLocalFP32, expertCountTempInLocalFP32, actualExpertNum_);
    Cast(expertTotalCountLocal, expertTotalCountLocalFP32, AscendC::RoundMode::CAST_RINT, 1);
    expertTotalCountQueue_.EnQue<int32_t>(expertTotalCountLocal);
    expertIdCountInQueue_.FreeTensor(expertCountInLocal);
    // 补充结束
}

__aicore__ inline void ExpertTokensCount::expertCountCopyOut()
{
    // 学员补充：copyOut 专家总数，推荐使用DataCopyPad
    LocalTensor<int32_t> expertTotalCountLocal = expertTotalCountQueue_.DeQue<int32_t>();
    DataCopyExtParams copyParams{static_cast<uint16_t>(1), sizeof(int32_t), 0, 0, 0};
    DataCopyPad(expertTotalCountGm_, expertTotalCountLocal, copyParams);
    expertTotalCountQueue_.FreeTensor(expertTotalCountLocal);
    // 补充结束
}
}  // namespace ExpertDispatch
#endif  // EXPERT_TOKENS_COUNT_H
