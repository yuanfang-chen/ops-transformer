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
 * \file expert_gather_out.h
 * \brief
 */
#ifndef EXPERT_GATHER_OUT_H
#define EXPERT_GATHER_OUT_H

#include "common.h"
#include "kernel_operator.h"

namespace ExpertDispatch
{
using namespace AscendC;

constexpr int64_t BUFFER_NUM = 2;

template <typename T>
class ExpertGatherOut
{
public:
    __aicore__ inline ExpertGatherOut(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR scale, GM_ADDR expertTotalCount, GM_ADDR dispatchedRowIdx,
                                GM_ADDR dispatchedX, GM_ADDR dispatchedScale,
                                const ExpertDispatchTilingData* tilingData, TPipe* tPipe);

    __aicore__ inline void Process();
    __aicore__ inline void CopyExpertIn();
    __aicore__ inline void CopyIn(int64_t xSrcOffset, int64_t scaleSrcOffset);
    __aicore__ inline void CopyOut(int64_t xDstOffset, int64_t scaleDstOffset);

private:
    TPipe* pipe_;
    // TQueBind 输入输出使用相同的Buffer
    TQueBind<TPosition::VECIN, TPosition::VECOUT, BUFFER_NUM> xCopyInQueue_;
    TQueBind<TPosition::VECIN, TPosition::VECOUT, BUFFER_NUM> scaleCopyInQueue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> dispatchedRowIdxCopyInQueue_;

    GlobalTensor<T> xGm_;                       // x
    GlobalTensor<float> scaleGm_;               // scale
    GlobalTensor<T> dispatchedXGm_;             // dispatched_x
    GlobalTensor<int32_t> dispatchedRowIdxGm_;  // dispatched_row_idx
    GlobalTensor<float> dispatchedScaleGm_;     // dispatched_scale
    GlobalTensor<int32_t> expertTotalCountGm_;  // expert_total_count

    int64_t blockIdx_;
    int64_t cols_;
    int64_t n_;
    int64_t k_;

    int64_t needCoreNum_;
    int64_t perCoreIndicesElements_;
    int64_t lastCoreIndicesElements_;
    int64_t curCoreIndicesElements_;

    int64_t actualExpertNum_;
    int64_t expertTotalCount_;
};

template <typename T>
__aicore__ inline void ExpertGatherOut<T>::Init(GM_ADDR x, GM_ADDR scale, GM_ADDR expertTotalCount,
                                                GM_ADDR dispatchedRowIdx, GM_ADDR dispatchedX, GM_ADDR dispatchedScale,
                                                const ExpertDispatchTilingData* tilingData, TPipe* tPipe)
{
    pipe_ = tPipe;
    blockIdx_ = GetBlockIdx();

    cols_ = tilingData->cols;
    n_ = tilingData->n;
    k_ = tilingData->k;

    actualExpertNum_ = tilingData->actualExpertNum;
    expertTotalCountGm_.SetGlobalBuffer((__gm__ int32_t*)expertTotalCount, actualExpertNum_);
    expertTotalCount_ = expertTotalCountGm_.GetValue(0);  // 真实的专家数
    needCoreNum_ = tilingData->gatherOutComputeParamsOp.needCoreNum;
    perCoreIndicesElements_ = tilingData->gatherOutComputeParamsOp.perCoreIndicesElements;
    lastCoreIndicesElements_ = tilingData->gatherOutComputeParamsOp.lastCoreIndicesElements;
    if (blockIdx_ == needCoreNum_ - 1) {
        curCoreIndicesElements_ = lastCoreIndicesElements_;
    } else {
        curCoreIndicesElements_ = perCoreIndicesElements_;
    }

    // 学员补充： GM/UB初始化
    perCoreIndicesElements_ = Ceil(expertTotalCount_, tilingData->coreNum);
    needCoreNum_ = Ceil(expertTotalCount_, perCoreIndicesElements_);
    lastCoreIndicesElements_ = expertTotalCount_ - (needCoreNum_ - 1) * perCoreIndicesElements_;
    if (blockIdx_ == needCoreNum_ - 1) {
        curCoreIndicesElements_ = lastCoreIndicesElements_;
    } else {
        curCoreIndicesElements_ = perCoreIndicesElements_;
    }
    
    xGm_.SetGlobalBuffer((__gm__ T*)x, n_ * cols_);
    scaleGm_.SetGlobalBuffer((__gm__ float*)scale, n_);

    dispatchedXGm_.SetGlobalBuffer((__gm__ T*)dispatchedX + blockIdx_ * perCoreIndicesElements_ * cols_,
                                   curCoreIndicesElements_ * cols_);
    dispatchedScaleGm_.SetGlobalBuffer((__gm__ float*)dispatchedScale + blockIdx_ * perCoreIndicesElements_,
                                       curCoreIndicesElements_);
    dispatchedRowIdxGm_.SetGlobalBuffer((__gm__ int32_t*)dispatchedRowIdx + blockIdx_ * perCoreIndicesElements_,
                                        Align(curCoreIndicesElements_, sizeof(int32_t)));

    pipe_->InitBuffer(dispatchedRowIdxCopyInQueue_, BUFFER_NUM, AlignBytes(curCoreIndicesElements_, sizeof(int32_t)));
    pipe_->InitBuffer(xCopyInQueue_, BUFFER_NUM, AlignBytes(cols_, sizeof(T)));
    pipe_->InitBuffer(scaleCopyInQueue_, BUFFER_NUM, AlignBytes(1, sizeof(float)));
    // 补充结束
}

template <typename T>
__aicore__ inline void ExpertGatherOut<T>::CopyExpertIn()
{
    // copyIn dispatched_row_idx
    LocalTensor<int32_t> subRowIdxLocal = dispatchedRowIdxCopyInQueue_.AllocTensor<int32_t>();
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(curCoreIndicesElements_ * sizeof(int32_t)), 0, 0, 0};
    DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};
    DataCopyPad(subRowIdxLocal, dispatchedRowIdxGm_, copyParams, padParams);
    dispatchedRowIdxCopyInQueue_.EnQue(subRowIdxLocal);
}

template <typename T>
__aicore__ inline void ExpertGatherOut<T>::CopyIn(int64_t xSrcOffset, int64_t scaleSrcOffset)
{
    // 学员补充：copyIn x 和 scale
    LocalTensor<T> xLocal = xCopyInQueue_.AllocTensor<T>();
    LocalTensor<float> scaleLocal = scaleCopyInQueue_.AllocTensor<float>();
    DataCopyExtParams copyParams0{static_cast<uint16_t>(1), static_cast<uint32_t>(cols_ * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams0{false, 0, 0, 0};
    DataCopyPad(xLocal, xGm_[xSrcOffset], copyParams0, padParams0);
    DataCopyExtParams copyParams1{static_cast<uint16_t>(1), static_cast<uint32_t>(1 * sizeof(float)), 0, 0, 0};
    DataCopyPadExtParams<float> padParams1{false, 0, 0, 0};
    DataCopyPad(scaleLocal, scaleGm_[scaleSrcOffset], copyParams1, padParams1);
    xCopyInQueue_.EnQue(xLocal);
    scaleCopyInQueue_.EnQue(scaleLocal);
    // 补充结束
}

template <typename T>
__aicore__ inline void ExpertGatherOut<T>::CopyOut(int64_t xDstOffset, int64_t scaleDstOffset)
{
    // 学员补充：copyOut x 和 scale到相应的位置
    LocalTensor<T> xLocal = xCopyInQueue_.DeQue<T>();
    LocalTensor<float> scaleLocal = scaleCopyInQueue_.DeQue<float>();
    DataCopyExtParams copyParams2{1, static_cast<uint32_t>(cols_ * sizeof(T)), 0, 0, 0};
    DataCopyPad(dispatchedXGm_[xDstOffset], xLocal, copyParams2);
    DataCopyExtParams copyParams3{1, static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
    DataCopyPad(dispatchedScaleGm_[scaleDstOffset], scaleLocal, copyParams3);
    xCopyInQueue_.FreeTensor(xLocal);
    scaleCopyInQueue_.FreeTensor(scaleLocal);
    // 补充结束
}

template <typename T>
__aicore__ inline void ExpertGatherOut<T>::Process()
{
    if (blockIdx_ < needCoreNum_) {
        CopyExpertIn();
        LocalTensor<int32_t> subRowIdxLocal = dispatchedRowIdxCopyInQueue_.DeQue<int32_t>();
        SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);
        for (int64_t indicesIndex = 0; indicesIndex < curCoreIndicesElements_; indicesIndex++) {
            int64_t xSrcOffset = 0;
            int64_t scaleSrcOffset = 0;
            int64_t xDstOffset = 0;
            int64_t scaleDstOffset = 0;
            // 学员补充：计算x和scale输入偏移和dispatched_x和dispatched_scale的输出偏移
            int64_t rowIdx = subRowIdxLocal.GetValue(indicesIndex);
            xSrcOffset = rowIdx / k_ * cols_;
            scaleSrcOffset = rowIdx / k_;
            xDstOffset = indicesIndex * cols_;
            scaleDstOffset = indicesIndex;
            // 补充结束

            SetWaitFlag<HardEvent::S_MTE2>(HardEvent::S_MTE2);
            CopyIn(xSrcOffset, scaleSrcOffset);
            CopyOut(xDstOffset, scaleDstOffset);
        }
        dispatchedRowIdxCopyInQueue_.FreeTensor(subRowIdxLocal);
    }
}
}  // namespace ExpertDispatch
#endif  // EXPERT_GATHER_OUT_H
