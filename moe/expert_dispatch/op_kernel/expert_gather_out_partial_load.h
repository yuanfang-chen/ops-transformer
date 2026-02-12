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
 * \file expert_gather_out_partial_load.h
 * \brief
 */
#ifndef EXPERT_GATHER_OUT_PARTIAL_LOAD_H
#define EXPERT_GATHER_OUT_PARTIAL_LOAD_H

#include "common.h"
#include "kernel_operator.h"

namespace ExpertDispatch
{
using namespace AscendC;

template <typename T>
class ExpertGatherOutPartialLoad
{
public:
    __aicore__ inline ExpertGatherOutPartialLoad(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR scale, GM_ADDR expertTotalCount, GM_ADDR dispatchedRowIdx,
                                GM_ADDR dispatchedX, GM_ADDR dispatchedScale,
                                const ExpertDispatchTilingData* tilingData, TPipe* tPipe);

    __aicore__ inline void Process();
    __aicore__ inline void CopyExpertIn(int64_t curExpertLoopOffset, int64_t curLoopElements);
    __aicore__ inline void CopyXIn(int64_t xSrcOffset, int64_t scaleSrcOffset, int64_t curLoopCols);
    __aicore__ inline void CopyXOut(int64_t xDstOffset, int64_t scaleDstOffset, int64_t curLoopCols);
    __aicore__ inline void CopyScaleIn(int64_t scaleSrcOffset);
    __aicore__ inline void CopyScaleOut(int64_t scaleDstOffset);

private:
    TPipe* pipe_;
    // TQueBind 输入输出使用相同的Buffer
    TQueBind<TPosition::VECIN, TPosition::VECOUT, BUFFER_NUM> xCopyInQueue_;
    TQueBind<TPosition::VECIN, TPosition::VECOUT, BUFFER_NUM> scaleCopyInQueue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> dispatchedRowIdxCopyInQueue_;

    GlobalTensor<T> xGm_;
    GlobalTensor<float> scaleGm_;
    GlobalTensor<T> dispatchedXGm_;
    GlobalTensor<int32_t> dispatchedRowIdxGm_;
    GlobalTensor<float> dispatchedScaleGm_;
    GlobalTensor<int32_t> expertTotalCountGm_;

    int64_t blockIdx_;
    int64_t cols_;
    int64_t n_;
    int64_t k_;

    int64_t colsLoops_;
    int64_t perLoopCols_;
    int64_t lastLoopCols_;

    int64_t indicesLoops_;

    int64_t perCoreIndicesElements_;
    int64_t lastCoreIndicesElements_;
    int64_t perCorePerLoopIndicesElements_;
    int64_t lastCorePerLoopIndicesElements_;
    int64_t curCorePerLoopIndicesElements_;
    int64_t curCoreLastLoopIndicesElements_;
    int64_t needCoreNum_;
    int64_t curCoreIndicesElements_;

    int64_t actualExpertNum_;
    int64_t expertTotalCount_;
};

template <typename T>
__aicore__ inline void ExpertGatherOutPartialLoad<T>::Init(GM_ADDR x, GM_ADDR scale, GM_ADDR expertTotalCount,
                                                           GM_ADDR dispatchedRowIdx, GM_ADDR dispatchedX,
                                                           GM_ADDR dispatchedScale,
                                                           const ExpertDispatchTilingData* tilingData, TPipe* tPipe)
{
    pipe_ = tPipe;

    blockIdx_ = GetBlockIdx();

    cols_ = tilingData->cols;
    n_ = tilingData->n;
    k_ = tilingData->k;

    colsLoops_ = tilingData->gatherOutComputeParamsOp.colsLoops;
    perLoopCols_ = tilingData->gatherOutComputeParamsOp.perLoopCols;
    lastLoopCols_ = tilingData->gatherOutComputeParamsOp.lastLoopCols;

    actualExpertNum_ = tilingData->actualExpertNum;
    expertTotalCountGm_.SetGlobalBuffer((__gm__ int32_t*)expertTotalCount, actualExpertNum_);
    expertTotalCount_ = expertTotalCountGm_.GetValue(0);
    perCorePerLoopIndicesElements_ = tilingData->gatherOutComputeParamsOp.perCorePerLoopIndicesElements;
    lastCorePerLoopIndicesElements_ = tilingData->gatherOutComputeParamsOp.lastCorePerLoopIndicesElements;

    // 核内重新计算Tiling
    perCoreIndicesElements_ = Ceil(expertTotalCount_, tilingData->coreNum);
    needCoreNum_ = Ceil(expertTotalCount_, perCoreIndicesElements_);
    lastCoreIndicesElements_ = expertTotalCount_ - (needCoreNum_ - 1) * perCoreIndicesElements_;
    if (blockIdx_ == needCoreNum_ - 1) {
        curCoreIndicesElements_ = lastCoreIndicesElements_;
        curCorePerLoopIndicesElements_ = Min(lastCorePerLoopIndicesElements_, curCoreIndicesElements_);
    } else {
        curCoreIndicesElements_ = perCoreIndicesElements_;
        curCorePerLoopIndicesElements_ = Min(perCorePerLoopIndicesElements_, curCoreIndicesElements_);
    }
    indicesLoops_ = Ceil(curCoreIndicesElements_, curCorePerLoopIndicesElements_);
    curCoreLastLoopIndicesElements_ = curCoreIndicesElements_ - (indicesLoops_ - 1) * curCorePerLoopIndicesElements_;

    xGm_.SetGlobalBuffer((__gm__ T*)x, n_ * cols_);
    scaleGm_.SetGlobalBuffer((__gm__ float*)scale, n_);

    dispatchedXGm_.SetGlobalBuffer((__gm__ T*)dispatchedX + blockIdx_ * perCoreIndicesElements_ * cols_,
                                   curCoreIndicesElements_ * cols_);
    dispatchedScaleGm_.SetGlobalBuffer((__gm__ float*)dispatchedScale + blockIdx_ * perCoreIndicesElements_,
                                       curCoreIndicesElements_);
    dispatchedRowIdxGm_.SetGlobalBuffer((__gm__ int32_t*)dispatchedRowIdx + blockIdx_ * perCoreIndicesElements_,
                                        Align(curCoreIndicesElements_, sizeof(int32_t)));

    pipe_->InitBuffer(dispatchedRowIdxCopyInQueue_, BUFFER_NUM,
                      AlignBytes(curCorePerLoopIndicesElements_, sizeof(int32_t)));
    pipe_->InitBuffer(xCopyInQueue_, BUFFER_NUM, AlignBytes(perLoopCols_, sizeof(T)));
    pipe_->InitBuffer(scaleCopyInQueue_, BUFFER_NUM, AlignBytes(1, sizeof(float)));
}

template <typename T>
__aicore__ inline void ExpertGatherOutPartialLoad<T>::CopyExpertIn(int64_t curExpertLoopOffset, int64_t curLoopElements)
{
    // copyIn dispatched_row_idx
    LocalTensor<int32_t> subRowIdxLocal = dispatchedRowIdxCopyInQueue_.AllocTensor<int32_t>();
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(curLoopElements * sizeof(int32_t)), 0, 0, 0};
    DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};
    DataCopyPad(subRowIdxLocal, dispatchedRowIdxGm_[curExpertLoopOffset], copyParams, padParams);
    dispatchedRowIdxCopyInQueue_.EnQue(subRowIdxLocal);
}

template <typename T>
__aicore__ inline void ExpertGatherOutPartialLoad<T>::CopyXIn(int64_t xSrcOffset, int64_t scaleSrcOffset,
                                                              int64_t curLoopCols)
{
    LocalTensor<T> xLocal = xCopyInQueue_.AllocTensor<T>();
    DataCopyExtParams copyParams0{static_cast<uint16_t>(1), static_cast<uint32_t>(curLoopCols * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams0{false, 0, 0, 0};
    DataCopyPad(xLocal, xGm_[xSrcOffset], copyParams0, padParams0);
    xCopyInQueue_.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void ExpertGatherOutPartialLoad<T>::CopyXOut(int64_t xDstOffset, int64_t scaleDstOffset,
                                                               int64_t curLoopCols)
{
    LocalTensor<T> xLocal = xCopyInQueue_.DeQue<T>();
    DataCopyExtParams copyParams2{1, static_cast<uint32_t>(curLoopCols * sizeof(T)), 0, 0, 0};
    DataCopyPad(dispatchedXGm_[xDstOffset], xLocal, copyParams2);
    xCopyInQueue_.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void ExpertGatherOutPartialLoad<T>::CopyScaleIn(int64_t scaleSrcOffset)
{
    LocalTensor<float> scaleLocal = scaleCopyInQueue_.AllocTensor<float>();
    DataCopyExtParams copyParams1{static_cast<uint16_t>(1), static_cast<uint32_t>(1 * sizeof(float)), 0, 0, 0};
    DataCopyPadExtParams<float> padParams1{false, 0, 0, 0};
    DataCopyPad(scaleLocal, scaleGm_[scaleSrcOffset], copyParams1, padParams1);
    scaleCopyInQueue_.EnQue(scaleLocal);
}

template <typename T>
__aicore__ inline void ExpertGatherOutPartialLoad<T>::CopyScaleOut(int64_t scaleDstOffset)
{
    LocalTensor<float> scaleLocal = scaleCopyInQueue_.DeQue<float>();
    DataCopyExtParams copyParams3{1, static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
    DataCopyPad(dispatchedScaleGm_[scaleDstOffset], scaleLocal, copyParams3);
    scaleCopyInQueue_.FreeTensor(scaleLocal);
}

template <typename T>
__aicore__ inline void ExpertGatherOutPartialLoad<T>::Process()
{
    if (blockIdx_ < needCoreNum_) {
        int64_t curLoopElements = curCorePerLoopIndicesElements_;
        for (int64_t indicesLoop = 0; indicesLoop < indicesLoops_; indicesLoop++) {
            if (indicesLoop == indicesLoops_ - 1) {
                curLoopElements = curCoreLastLoopIndicesElements_;
            }
            int64_t curExpertLoopOffset = indicesLoop * curCorePerLoopIndicesElements_;
            SetWaitFlag<HardEvent::S_MTE2>(HardEvent::S_MTE2);
            CopyExpertIn(curExpertLoopOffset, curLoopElements);
            LocalTensor<int32_t> subRowIdxLocal = dispatchedRowIdxCopyInQueue_.DeQue<int32_t>();
            SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);
            for (int64_t indicesIndex = 0; indicesIndex < curLoopElements; indicesIndex++) {
                int64_t rowIdx = subRowIdxLocal.GetValue(indicesIndex);
                int64_t xSrcOffset = rowIdx / k_ * cols_;
                int64_t scaleSrcOffset = rowIdx / k_;
                int64_t xDstOffset = (curExpertLoopOffset + indicesIndex) * cols_;
                SetWaitFlag<HardEvent::S_MTE2>(HardEvent::S_MTE2);
                CopyScaleIn(scaleSrcOffset);
                CopyScaleOut(indicesIndex + curExpertLoopOffset);
                int64_t curLoopCols = perLoopCols_;
                for (int64_t colsLoop = 0; colsLoop < colsLoops_; colsLoop++) {
                    if (colsLoop == colsLoops_ - 1) {
                        curLoopCols = lastLoopCols_;
                    }
                    int64_t colsLoopOffset = colsLoop * perLoopCols_;
                    CopyXIn(xSrcOffset + colsLoopOffset, scaleSrcOffset, curLoopCols);
                    CopyXOut(xDstOffset + colsLoopOffset, indicesIndex, curLoopCols);
                }
            }
            dispatchedRowIdxCopyInQueue_.FreeTensor(subRowIdxLocal);
        }
    }
}
}  // namespace ExpertDispatch
#endif  // EXPERT_GATHER_OUT_PARTIAL_LOAD_H
