/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_v3_gather_static_quant.h
 * \brief A5 (Arch35) 静态量化实现，支持Gather和Scatter模式
 */
#ifndef MOE_V3_GATHER_STATIC_QUANT_H_REGBASE
#define MOE_V3_GATHER_STATIC_QUANT_H_REGBASE

#include "moe_v3_common.h"
#include "kernel_operator.h"

namespace MoeInitRoutingV3 {
using namespace AscendC;

constexpr int64_t BUFFER_NUM = 2;
constexpr int64_t DROP_PAD_MODE = 1;
constexpr int64_t DROPLESS_MODE = 0;

#define MIRV3_STATIC_QUANT_DEBUG_PRINT(fmt, ...) AscendC::printf("[MIRV3_SQ] " fmt "\n", ##__VA_ARGS__)

template <typename T>
class MoeV3GatherStaticQuant {
public:
    __aicore__ inline MoeV3GatherStaticQuant(){};
    __aicore__ inline void Init(GM_ADDR inputX, GM_ADDR scale, GM_ADDR workspace, GM_ADDR expandedRowIdx,
                                GM_ADDR expandedX, GM_ADDR offset, const MoeInitRoutingV3Arch35TilingData *tilingData,
                                TPipe *tPipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyInIndices(int64_t progress);
    __aicore__ inline void Compute();
    __aicore__ inline void GatherCopyOut(int64_t progress);
    __aicore__ inline void ScatterCopyOut(int64_t progress);
    __aicore__ inline void CopyXIn(int64_t xSrcOffset, int64_t curLoopCols);
    __aicore__ inline void CopyXOut(int64_t xDstOffset, int64_t curLoopCols);
    __aicore__ inline void CopyInZeroIndices(int64_t progress);
    __aicore__ inline void CopyOutZero(int64_t progress);

private:
    TPipe *pipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputXCopyInQueue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> expandRowIdxCopyInQueue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> inputXCopyOutQueue_;
    TQue<QuePosition::VECOUT, 1> floatQueue_;
    TQue<QuePosition::VECOUT, 1> halfQueue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> expandedRowIdxIndexCopyInQueue_;

    GlobalTensor<T> inputXGm_;
    GlobalTensor<int8_t> expandedXGm_;
    GlobalTensor<int32_t> expandedRowIdxGm_;
    GlobalTensor<float> scaleGm_;
    GlobalTensor<float> offsetGm_;
    GlobalTensor<int32_t> expertTotalCountGm_;
    GlobalTensor<int32_t> expandedRowIdxIndexGm_;

    const MoeV3Arch35GatherOutComputeTilingData *gatherOutTilingData_;

    int64_t needCoreNum_;
    int64_t blockIdx_;
    int64_t cols_;
    int64_t n_;
    int64_t k_;
    int64_t activateRows_;
    int64_t currentLoopRows_;
    int64_t coreRows_;
    int64_t perLoopRows_;
    int64_t lastLoopRows_;
    int64_t rowLoops_;
    int64_t colsTileLength_;
    int64_t perLoopCols_;
    int64_t lastLoopCols_;
    int64_t colLoops_;
    int64_t dropPadMode_;
    float scale_;
    float offset_;
    int64_t expertNum_;
    int64_t actualExpertNum_;
    int64_t expertTotalCount_;
    int64_t totalLength_;
    int64_t rowIdxType_;
    int64_t perCoreRow_;

    int64_t indicesOffset_;
    int64_t inputOffset_;
    int64_t outOffset_;
    int64_t debugInvalidIndexPrintCount_;
};

template <typename T>
__aicore__ inline void MoeV3GatherStaticQuant<T>::CopyInIndices(int64_t progress)
{
    indicesOffset_ = progress * perLoopRows_;
    LocalTensor<int32_t> indicesLocal = expandRowIdxCopyInQueue_.AllocTensor<int32_t>();
    DataCopyExtParams dataCopyParams{1, static_cast<uint32_t>(currentLoopRows_ * sizeof(int32_t)), 0, 0, 0};
    DataCopyPadExtParams<int32_t> dataCopyPadParams{false, 0, 0, 0};
    DataCopyPad(indicesLocal, expandedRowIdxGm_[indicesOffset_], dataCopyParams, dataCopyPadParams);
    expandRowIdxCopyInQueue_.EnQue<int32_t>(indicesLocal);
}

template <typename T>
__aicore__ inline void MoeV3GatherStaticQuant<T>::CopyInZeroIndices(int64_t progress)
{
    indicesOffset_ = progress * perLoopRows_;
    LocalTensor<int32_t> expandedRowIdxIndexLocal = expandedRowIdxIndexCopyInQueue_.AllocTensor<int32_t>();
    DataCopyExtParams dataCopyParams{1, static_cast<uint32_t>((currentLoopRows_ + 1) * sizeof(int32_t)), 0, 0, 0};
    DataCopyPadExtParams<int32_t> dataCopyPadParams{false, 0, 0, 0};
    DataCopyPad(expandedRowIdxIndexLocal, expandedRowIdxIndexGm_[indicesOffset_], dataCopyParams, dataCopyPadParams);
    expandedRowIdxIndexCopyInQueue_.EnQue<int32_t>(expandedRowIdxIndexLocal);
}

template <typename T>
__aicore__ inline void MoeV3GatherStaticQuant<T>::CopyOutZero(int64_t progress)
{
    LocalTensor<int32_t> indicesLocal = expandedRowIdxIndexCopyInQueue_.DeQue<int32_t>();
    if (blockIdx_ == 0 && progress == 0) {
        MIRV3_STATIC_QUANT_DEBUG_PRINT(
            "CopyOutZero begin: activateRows=%ld currentLoopRows=%ld firstBoundary=%d secondBoundary=%d", activateRows_,
            currentLoopRows_, indicesLocal.GetValue(0), (currentLoopRows_ > 0 ? indicesLocal.GetValue(1) : -1));
    }
    if (blockIdx_ == 0) {
        int32_t curIndex = 0;
        int32_t nextIndex = indicesLocal.GetValue(0);
        int32_t count = nextIndex - curIndex;
        if (count > 0) {
            InitOutput(expandedXGm_[curIndex * cols_], count * cols_, static_cast<int8_t>(0));
        }
    }
    for (int i = 0; i < currentLoopRows_; i++) {
        int32_t curIndex = indicesLocal.GetValue(i) + 1;
        if (activateRows_ <= curIndex) {
            break;
        }
        int32_t nextIndex;
        if (blockIdx_ == gatherOutTilingData_->needCoreNum - 1 && progress == rowLoops_ - 1 &&
            i == currentLoopRows_ - 1) {
            nextIndex = activateRows_;
        } else {
            nextIndex = indicesLocal.GetValue(i + 1);
        }
        int32_t count = nextIndex - curIndex;
        if (count > 0) {
            InitOutput(expandedXGm_[curIndex * cols_], count * cols_, static_cast<int8_t>(0));
        }
    }
    expandedRowIdxIndexCopyInQueue_.FreeTensor(indicesLocal);
}

template <typename T>
__aicore__ inline void MoeV3GatherStaticQuant<T>::Compute()
{
    LocalTensor<T> inLocal = inputXCopyInQueue_.DeQue<T>();
    LocalTensor<int8_t> outLocal = inputXCopyOutQueue_.AllocTensor<int8_t>();
    LocalTensor<float> floatLocal = floatQueue_.AllocTensor<float>();
    LocalTensor<half> halfLocal = halfQueue_.AllocTensor<half>();
    uint32_t elements = Align(colsTileLength_, sizeof(T));

    if constexpr (IsSameType<T, bfloat16_t>::value) {
        Cast(floatLocal, inLocal, RoundMode::CAST_NONE, colsTileLength_);
        TQueSync<PIPE_MTE3, PIPE_V> sync;
        sync.SetFlag(0);
        sync.WaitFlag(0);
        Cast(halfLocal, floatLocal, RoundMode::CAST_NONE, elements);
        Muls(halfLocal, halfLocal, static_cast<half>(scale_), elements);
        Adds(halfLocal, halfLocal, static_cast<half>(offset_), elements);
        LocalTensor<int32_t> intLocal = floatLocal.ReinterpretCast<int32_t>();
        Cast(intLocal, halfLocal, RoundMode::CAST_RINT, elements);
        SetDeqScale((half)1.000000e+00f);
        Cast(halfLocal, intLocal, RoundMode::CAST_RINT, elements);
        Cast(outLocal, halfLocal, RoundMode::CAST_RINT, elements);
    } else if constexpr (IsSameType<T, float>::value) {
        Cast(halfLocal, inLocal, RoundMode::CAST_NONE, elements);
        Muls(halfLocal, halfLocal, static_cast<half>(scale_), elements);
        Adds(halfLocal, halfLocal, static_cast<half>(offset_), elements);
        Cast(outLocal, halfLocal, RoundMode::CAST_RINT, elements);
    } else {
        Muls(inLocal, inLocal, static_cast<T>(scale_), elements);
        Adds(inLocal, inLocal, static_cast<T>(offset_), elements);
        Cast(outLocal, inLocal, RoundMode::CAST_RINT, elements);
    }

    inputXCopyOutQueue_.EnQue(outLocal);
    inputXCopyInQueue_.FreeTensor(inLocal);
    floatQueue_.FreeTensor(floatLocal);
    halfQueue_.FreeTensor(halfLocal);
}

template <typename T>
__aicore__ inline void MoeV3GatherStaticQuant<T>::CopyXIn(int64_t xSrcOffset, int64_t curLoopCols)
{
    LocalTensor<T> inLocal = inputXCopyInQueue_.AllocTensor<T>();
    DataCopyExtParams dataCopyParams{1, static_cast<uint32_t>(curLoopCols * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> dataCopyPadParams{false, 0, 0, 0};
    DataCopyPad(inLocal, inputXGm_[xSrcOffset], dataCopyParams, dataCopyPadParams);
    inputXCopyInQueue_.EnQue<T>(inLocal);
}

template <typename T>
__aicore__ inline void MoeV3GatherStaticQuant<T>::CopyXOut(int64_t xDstOffset, int64_t curLoopCols)
{
    LocalTensor<int8_t> outLocal = inputXCopyOutQueue_.DeQue<int8_t>();
    DataCopyExtParams dataCopyParams{1, static_cast<uint32_t>(curLoopCols * sizeof(int8_t)), 0, 0, 0};
    DataCopyPad(expandedXGm_[xDstOffset], outLocal, dataCopyParams);
    inputXCopyOutQueue_.FreeTensor(outLocal);
}

template <typename T>
__aicore__ inline void MoeV3GatherStaticQuant<T>::ScatterCopyOut(int64_t progress)
{
    LocalTensor<int32_t> indicesLocal = expandRowIdxCopyInQueue_.DeQue<int32_t>();
    SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);

    if (blockIdx_ == 0 && progress == 0) {
        MIRV3_STATIC_QUANT_DEBUG_PRINT("ScatterCopyOut begin: coreRows=%ld perLoopRows=%ld rowLoops=%ld firstRowIdx=%d",
                                       coreRows_, perLoopRows_, rowLoops_,
                                       (currentLoopRows_ > 0 ? indicesLocal.GetValue(0) : -1));
    }

    for (int64_t indicesIndex = 0; indicesIndex < currentLoopRows_; indicesIndex++) {
        int64_t rowOffset = perCoreRow_ * blockIdx_ + perLoopRows_ * progress;
        int64_t rowIdx = indicesLocal.GetValue(indicesIndex);
        int64_t xSrcOffset = rowIdx / k_ * cols_;
        int64_t xDstOffset = (rowOffset + indicesIndex) * cols_;
        int64_t curLoopCols = perLoopCols_;

        if (activateRows_ > 0 && dropPadMode_ == DROPLESS_MODE && (rowOffset + indicesIndex) >= activateRows_) {
            break;
        }

        SetWaitFlag<HardEvent::S_MTE2>(HardEvent::S_MTE2);

        for (int64_t colsLoop = 0; colsLoop < colLoops_; colsLoop++) {
            if (colsLoop == colLoops_ - 1) {
                curLoopCols = lastLoopCols_;
            }
            int64_t colsLoopOffset = colsLoop * perLoopCols_;
            colsTileLength_ = curLoopCols;

            CopyXIn(xSrcOffset + colsLoopOffset, curLoopCols);
            Compute();
            CopyXOut(xDstOffset + colsLoopOffset, curLoopCols);
        }
    }
    expandRowIdxCopyInQueue_.FreeTensor(indicesLocal);
}

template <typename T>
__aicore__ inline void MoeV3GatherStaticQuant<T>::GatherCopyOut(int64_t progress)
{
    LocalTensor<int32_t> indicesLocal = expandRowIdxCopyInQueue_.DeQue<int32_t>();
    SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);
    colsTileLength_ = perLoopCols_;

    if (blockIdx_ == 0 && progress == 0) {
        MIRV3_STATIC_QUANT_DEBUG_PRINT(
            "GatherCopyOut begin: coreRows=%ld perLoopRows=%ld rowLoops=%ld firstOutIndex=%d activateRows=%ld",
            coreRows_, perLoopRows_, rowLoops_, (currentLoopRows_ > 0 ? indicesLocal.GetValue(0) : -1), activateRows_);
    }

    for (int64_t colsLoop = 0; colsLoop < colLoops_; colsLoop++) {
        int64_t initialRow = perCoreRow_ * blockIdx_ + perLoopRows_ * progress;
        int64_t curLoopRow = 0;

        if (colsLoop == colLoops_ - 1) {
            colsTileLength_ = lastLoopCols_;
        }

        int64_t currentLoopStartRow = initialRow / k_;
        int64_t currentLoopLastRow = (initialRow + currentLoopRows_ - 1) / k_;

        for (int64_t row = currentLoopStartRow; row <= currentLoopLastRow; row++) {
            inputOffset_ = row * cols_ + colsLoop * perLoopCols_;

            CopyXIn(inputOffset_, colsTileLength_);
            Compute();

            LocalTensor<int8_t> outLocal = inputXCopyOutQueue_.DeQue<int8_t>();
            DataCopyExtParams intriParams{1, static_cast<uint32_t>(colsTileLength_ * sizeof(int8_t)), 0, 0, 0};

            while (curLoopRow < currentLoopRows_ && initialRow / k_ == row) {
                int32_t outIndex = indicesLocal.GetValue(curLoopRow);
                curLoopRow++;
                initialRow++;

                if (outIndex < 0 || outIndex >= totalLength_ ||
                    (dropPadMode_ == DROPLESS_MODE && outIndex >= activateRows_)) {
                    if (blockIdx_ == 0 && debugInvalidIndexPrintCount_ < 8) {
                        MIRV3_STATIC_QUANT_DEBUG_PRINT("Gather invalid outIndex=%d totalLength=%ld activateRows=%ld "
                                                       "progress=%ld colsLoop=%ld row=%ld",
                                                       outIndex, totalLength_, activateRows_, progress, colsLoop, row);
                        debugInvalidIndexPrintCount_++;
                    }
                    continue;
                }

                outOffset_ = outIndex * cols_ + colsLoop * perLoopCols_;
                DataCopyPad(expandedXGm_[outOffset_], outLocal, intriParams);
            }
            inputXCopyOutQueue_.FreeTensor(outLocal);
        }
    }
    expandRowIdxCopyInQueue_.FreeTensor(indicesLocal);
}

template <typename T>
__aicore__ inline void MoeV3GatherStaticQuant<T>::Init(GM_ADDR inputX, GM_ADDR scale, GM_ADDR workspace,
                                                       GM_ADDR expandedRowIdx, GM_ADDR expandedX, GM_ADDR offset,
                                                       const MoeInitRoutingV3Arch35TilingData *tilingData, TPipe *tPipe)
{
    pipe_ = tPipe;
    blockIdx_ = GetBlockIdx();
    gatherOutTilingData_ = &(tilingData->gatherOutComputeParamsOp);

    needCoreNum_ = gatherOutTilingData_->needCoreNum;
    cols_ = tilingData->cols;
    n_ = tilingData->n;
    k_ = tilingData->k;
    dropPadMode_ = tilingData->dropPadMode;
    expertNum_ = tilingData->expertNum;
    actualExpertNum_ = tilingData->actualExpertNum;
    totalLength_ = tilingData->n * tilingData->k;
    activateRows_ = gatherOutTilingData_->activeNum;
    rowIdxType_ = tilingData->rowIdxType;
    debugInvalidIndexPrintCount_ = 0;

    // 初始化每核处理的行数
    perCoreRow_ = Ceil(totalLength_, tilingData->coreNum);

    if (blockIdx_ == gatherOutTilingData_->needCoreNum - 1) {
        coreRows_ = gatherOutTilingData_->lastCoreIndicesElements;
        perLoopRows_ = gatherOutTilingData_->lastCorePerLoopIndicesElements;
        lastLoopRows_ = gatherOutTilingData_->lastCoreLastLoopIndicesElements;
        rowLoops_ = gatherOutTilingData_->lastCoreIndicesLoops;
    } else {
        coreRows_ = gatherOutTilingData_->perCoreIndicesElements;
        perLoopRows_ = gatherOutTilingData_->perCorePerLoopIndicesElements;
        lastLoopRows_ = gatherOutTilingData_->perCoreLastLoopIndicesElements;
        rowLoops_ = gatherOutTilingData_->perCoreIndicesLoops;
    }

    perLoopCols_ = gatherOutTilingData_->perLoopCols;
    lastLoopCols_ = gatherOutTilingData_->lastLoopCols;
    colLoops_ = gatherOutTilingData_->colsLoops;

    inputXGm_.SetGlobalBuffer((__gm__ T *)inputX);
    expandedXGm_.SetGlobalBuffer((__gm__ int8_t *)expandedX);

    if (rowIdxType_ == SCATTER) {
        expandedRowIdxGm_.SetGlobalBuffer((__gm__ int32_t *)expandedRowIdx + blockIdx_ * perCoreRow_,
                                          Align(coreRows_, sizeof(int32_t)));
    } else {
        expandedRowIdxGm_.SetGlobalBuffer((__gm__ int32_t *)expandedRowIdx + blockIdx_ * perCoreRow_,
                                          Align(coreRows_, sizeof(int32_t)));
    }

    scaleGm_.SetGlobalBuffer((__gm__ float *)scale, 1);
    offsetGm_.SetGlobalBuffer((__gm__ float *)offset, 1);
    scale_ = scaleGm_.GetValue(0);
    offset_ = offsetGm_.GetValue(0);

    expertTotalCountGm_.SetGlobalBuffer((__gm__ int32_t *)workspace + Align(totalLength_, sizeof(int32_t)) * 2 +
                                            Align(actualExpertNum_, sizeof(int32_t)),
                                        1);
    AscendC::DataCacheCleanAndInvalid<int32_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(
        expertTotalCountGm_);
    expertTotalCount_ = expertTotalCountGm_.GetValue(0);

    if (blockIdx_ == 0) {
        MIRV3_STATIC_QUANT_DEBUG_PRINT(
            "Init summary: totalLength=%ld cols=%ld n=%ld k=%ld needCoreNum=%ld perCoreRow=%ld coreRows=%ld "
            "perLoopRows=%ld rowLoops=%ld perLoopCols=%ld colLoops=%ld dropPadMode=%ld rowIdxType=%ld "
            "activateRows=%ld expertNum=%ld actualExpertNum=%ld expertTotalCount=%ld",
            totalLength_, cols_, n_, k_, needCoreNum_, perCoreRow_, coreRows_, perLoopRows_, rowLoops_, perLoopCols_,
            colLoops_, dropPadMode_, rowIdxType_, activateRows_, expertNum_, actualExpertNum_, expertTotalCount_);
    }

    int64_t expandedRowIdxIndexBase =
        Align(totalLength_, sizeof(int32_t)) * 2 + Align(actualExpertNum_, sizeof(int32_t)) + blockIdx_ * perCoreRow_;
    expandedRowIdxIndexGm_.SetGlobalBuffer((__gm__ int32_t *)workspace + expandedRowIdxIndexBase, coreRows_ + 1);

    pipe_->InitBuffer(inputXCopyInQueue_, BUFFER_NUM, AlignBytes(perLoopCols_, sizeof(T)));
    pipe_->InitBuffer(inputXCopyOutQueue_, BUFFER_NUM, AlignBytes(perLoopCols_, sizeof(int8_t)));
    pipe_->InitBuffer(expandRowIdxCopyInQueue_, BUFFER_NUM, AlignBytes(perLoopRows_, sizeof(int32_t)));
    pipe_->InitBuffer(floatQueue_, 1, AlignBytes(perLoopCols_, sizeof(float)));
    pipe_->InitBuffer(halfQueue_, 1, AlignBytes(perLoopCols_, sizeof(half)));
    pipe_->InitBuffer(expandedRowIdxIndexCopyInQueue_, BUFFER_NUM, AlignBytes(perLoopRows_ + 1, sizeof(int32_t)));
}

template <typename T>
__aicore__ inline void MoeV3GatherStaticQuant<T>::Process()
{
    if (blockIdx_ == 0) {
        MIRV3_STATIC_QUANT_DEBUG_PRINT("Process begin: dropPadMode=%ld rowIdxType=%ld needCoreNum=%ld rowLoops=%ld "
                                       "expertTotalCount=%ld totalLength=%ld",
                                       dropPadMode_, rowIdxType_, needCoreNum_, rowLoops_, expertTotalCount_,
                                       totalLength_);
    }

    if (dropPadMode_ == DROPLESS_MODE && rowIdxType_ == GATHER) {
        int64_t zeroStartRow = blockIdx_ * perCoreRow_;
        if (zeroStartRow < totalLength_) {
            int64_t zeroRows = Min(perCoreRow_, totalLength_ - zeroStartRow);
            InitOutput(expandedXGm_[zeroStartRow * cols_], zeroRows * cols_, static_cast<int8_t>(0));
        }
        SyncAll();
    }

    if (blockIdx_ < needCoreNum_) {
        currentLoopRows_ = perLoopRows_;
        for (int64_t loop = 0; loop < rowLoops_; loop++) {
            if (loop == rowLoops_ - 1) {
                currentLoopRows_ = lastLoopRows_;
            }

            if (dropPadMode_ == DROP_PAD_MODE) {
                CopyInZeroIndices(loop);
                CopyOutZero(loop);
            }

            CopyInIndices(loop);

            if (rowIdxType_ == SCATTER) {
                ScatterCopyOut(loop);
            } else {
                GatherCopyOut(loop);
            }
        }
    }
}

} // namespace MoeInitRoutingV3
#undef MIRV3_STATIC_QUANT_DEBUG_PRINT
#endif // MOE_V3_GATHER_STATIC_QUANT_H_REGBASE
