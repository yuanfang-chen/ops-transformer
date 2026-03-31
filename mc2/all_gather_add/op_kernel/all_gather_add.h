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
 * \file all_gather_add.h
 * \brief Kernel 类定义，当前仅保留骨架
 */

#ifndef ALL_GATHER_ADD_H
#define ALL_GATHER_ADD_H

#include <cstddef>

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "kernel_tiling/kernel_tiling.h"
#include "adv_api/hccl/hccl.h"
#include "all_gather_add_tiling_data.h"
#include "all_gather_add_tiling_key.h"

namespace NsAllGatherAdd {

using namespace AscendC;

constexpr uint32_t MAX_COMM_TURN = 2U;
constexpr uint32_t UB_ALIGN_BYTES = 32U;
constexpr uint32_t MAX_RANK_SIZE = 2U;
template <typename T>
class AllGatherAdd {
public:
    __aicore__ inline AllGatherAdd(){};
    __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR aGathered, GM_ADDR c, TPipe* pipe,
                                const AllGatherAddTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline int64_t GetRoundElemCount(int64_t roundIdx) const;
    __aicore__ inline int64_t GetRoundLocalOffset(int64_t roundIdx) const;
    __aicore__ inline HcclHandle LaunchAllGather();
    __aicore__ inline void AddRound(int64_t roundIdx, int64_t roundElemCount);
    __aicore__ inline void AddPairedSegments(int64_t roundGlobalOffset, int64_t elemCount);

private:
    TPipe* pipe_ {nullptr};
    TBuf<> gatheredBuf_;
    TBuf<> bBuf_;
    TBuf<> cBuf_;
    Hccl<HCCL_SERVER_TYPE_AICPU> hccl_;

    GlobalTensor<T> inputGMA_;
    GlobalTensor<T> inputGMB_;
    GlobalTensor<T> outputGMAGathered_;
    GlobalTensor<T> outputGMC_;

    GM_ADDR inputAAddr_ {nullptr};
    GM_ADDR outputAGatheredAddr_ {nullptr};
    int64_t rankCount_ = 0;
    int64_t turnCount_ = 0;
    int64_t inputElementsPerRank_ = 0;
    int64_t elementsPerTurnPerRank_ = 0;
    int64_t lastTurnElementsPerRank_ = 0;
    int64_t outputElements_ = 0;
    int64_t elementsPerCorePerTurn_ = 0;
    int64_t tileElementsPerCoreCalc_ = 0;
    uint32_t blockIdx_ = 0;
    uint32_t blockNum_ = 1;
    AscendC::HcclDataType dataType_ = HcclDataType::HCCL_DATA_TYPE_FP16;
};

template <typename T>
__aicore__ inline void AllGatherAdd<T>::Init(GM_ADDR a, GM_ADDR b, GM_ADDR aGathered, GM_ADDR c, TPipe* pipe,
                                             const AllGatherAddTilingData* tilingData)
{
    pipe_ = pipe;
    blockIdx_ = GetBlockIdx();
    blockNum_ = GetBlockNum();

    inputAAddr_ = a;
    outputAGatheredAddr_ = aGathered;
    inputGMA_.SetGlobalBuffer((__gm__ T*)a);
    inputGMB_.SetGlobalBuffer((__gm__ T*)b);
    outputGMAGathered_.SetGlobalBuffer((__gm__ T*)aGathered);
    outputGMC_.SetGlobalBuffer((__gm__ T*)c);

    rankCount_ = tilingData->allGatherAddTilingInfo.rankCount;
    turnCount_ = tilingData->allGatherAddTilingInfo.turnCount;
    inputElementsPerRank_ = tilingData->allGatherAddTilingInfo.inputElementsPerRank;
    elementsPerTurnPerRank_ = tilingData->allGatherAddTilingInfo.elementsPerTurnPerRank;
    lastTurnElementsPerRank_ = tilingData->allGatherAddTilingInfo.lastTurnElementsPerRank;
    outputElements_ = tilingData->allGatherAddTilingInfo.outputElements;
    elementsPerCorePerTurn_ = tilingData->allGatherAddTilingInfo.elementsPerCorePerTurn;
    tileElementsPerCoreCalc_ = tilingData->allGatherAddTilingInfo.tileElementsPerCoreCalc;

    GM_ADDR contextGM = GetHcclContext<HCCL_GROUP_ID_0>();
    hccl_.InitV2(contextGM, tilingData);
    hccl_.SetCcTilingV2(offsetof(AllGatherAddTilingData, mc2CcTiling));

    pipe_->InitBuffer(gatheredBuf_, tileElementsPerCoreCalc_ * MAX_RANK_SIZE * sizeof(T));
    pipe_->InitBuffer(bBuf_, tileElementsPerCoreCalc_ * MAX_RANK_SIZE * sizeof(T));
    pipe_->InitBuffer(cBuf_, tileElementsPerCoreCalc_ * MAX_RANK_SIZE * sizeof(T));
}

template <typename T>
__aicore__ inline int64_t AllGatherAdd<T>::GetRoundElemCount(int64_t roundIdx) const
{
    if (roundIdx >= turnCount_ - 1) {
        return lastTurnElementsPerRank_;
    }
    return elementsPerTurnPerRank_;
}

template <typename T>
__aicore__ inline int64_t AllGatherAdd<T>::GetRoundLocalOffset(int64_t roundIdx) const
{
    return roundIdx * elementsPerTurnPerRank_;
}

template <typename T>
__aicore__ inline HcclHandle AllGatherAdd<T>::LaunchAllGather()
{
    return hccl_.AllGather<true>(inputAAddr_, outputAGatheredAddr_, elementsPerTurnPerRank_, dataType_, inputElementsPerRank_,
                                 static_cast<uint8_t>(turnCount_));
}

template <typename T>
__aicore__ inline void AllGatherAdd<T>::AddPairedSegments(int64_t roundGlobalOffset, int64_t elemCount)
{
    const uint32_t blockLen = static_cast<uint32_t>(elemCount * sizeof(T));
    const uint32_t gmStrideBytes = static_cast<uint32_t>((inputElementsPerRank_ - elemCount) * sizeof(T));
    DataCopyExtParams copyInParams = {static_cast<uint16_t>(rankCount_), blockLen, gmStrideBytes, 0U, 0U};
    DataCopyExtParams copyOutParams = {static_cast<uint16_t>(rankCount_), blockLen, 0U, gmStrideBytes, 0U};
    DataCopyPadExtParams<T> padParams = {false, 0U, 0U, static_cast<T>(0)};

    LocalTensor<T> gatheredLocal = gatheredBuf_.Get<T>();
    LocalTensor<T> bLocal = bBuf_.Get<T>();
    LocalTensor<T> cLocal = cBuf_.Get<T>();

    DataCopyPad(gatheredLocal, outputGMAGathered_[roundGlobalOffset], copyInParams, padParams);
    DataCopyPad(bLocal, inputGMB_[roundGlobalOffset], copyInParams, padParams);
    Add(cLocal, gatheredLocal, bLocal, elemCount * rankCount_);
    DataCopyPad(outputGMC_[roundGlobalOffset], cLocal, copyOutParams);
}

template <typename T>
__aicore__ inline void AllGatherAdd<T>::AddRound(int64_t roundIdx, int64_t roundElemCount)
{
    const int64_t blockBegin = static_cast<int64_t>(blockIdx_) * elementsPerCorePerTurn_;
    const int64_t workBegin = blockBegin < roundElemCount ? blockBegin : roundElemCount;
    const int64_t blockEnd = workBegin + elementsPerCorePerTurn_;
    const int64_t workEnd = blockEnd < roundElemCount ? blockEnd : roundElemCount;
    if (workBegin >= workEnd) {
        return;
    }

    const int64_t roundLocalOffset = GetRoundLocalOffset(roundIdx);
    const int64_t workElemCount = workEnd - workBegin;
    for (int64_t processed = 0; processed < workElemCount; processed += tileElementsPerCoreCalc_) {
        const int64_t remainElem = workElemCount - processed;
        const int64_t currentTileElem = remainElem < tileElementsPerCoreCalc_ ? remainElem : tileElementsPerCoreCalc_;
        const int64_t roundGlobalOffset = roundLocalOffset + workBegin + processed;
        AddPairedSegments(roundGlobalOffset, currentTileElem);
    }
}

template <typename T>
__aicore__ inline void AllGatherAdd<T>::Process()
{
    HcclHandle handle = LaunchAllGather();
    for (int64_t roundIdx = 0; roundIdx < turnCount_; ++roundIdx) {
        const int64_t roundElemCount = GetRoundElemCount(roundIdx);
        if (roundElemCount <= 0) {
            continue;
        }

        hccl_.Wait(handle);
        AddRound(roundIdx, roundElemCount);
    }

    SyncAll<true>();
    if (blockIdx_ == 0) {
        hccl_.Finalize();
    }
}

} // namespace NsAllGatherAdd

#endif // ALL_GATHER_ADD_H
