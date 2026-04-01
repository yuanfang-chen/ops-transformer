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
 * \file moe_init_routing_v3_mx_quant_sort_one_core.h
 * \brief Single-core sort stage for MoeInitRoutingV3MxQuant
 */
#ifndef MOE_INIT_ROUTING_V3_MX_QUANT_SORT_ONE_CORE_H
#define MOE_INIT_ROUTING_V3_MX_QUANT_SORT_ONE_CORE_H

#include "moe_init_routing_v3_mx_quant_common.h"

namespace MoeInitRoutingV3MxQuantNs {
using namespace AscendC;

class MoeSortOneCore {
public:
    __aicore__ inline MoeSortOneCore(){};
    __aicore__ inline void Init(GM_ADDR expertIdx, GM_ADDR expandedRowIdx, GM_ADDR workspace,
                                const MoeInitRoutingV3MxQuantArch35TilingData *tilingData, TPipe *tPipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn();
    __aicore__ inline void SortCompute();
    __aicore__ inline void CopyOut();

private:
    TPipe *pipe_;
    TQue<QuePosition::VECIN, 1> sortDataCopyInQueue_;
    TQue<QuePosition::VECOUT, 1> sortDataCopyOutQueue_;
    TBuf<TPosition::VECCALC> tempBuffer_;
    TBuf<TPosition::VECCALC> sortedBuffer_;

    GlobalTensor<int32_t> expertIdxGm_;
    GlobalTensor<int32_t> expandedRowIdxGm_;
    GlobalTensor<int32_t> sortedexpertIdxGm_;
    GlobalTensor<int32_t> expertCountTempGm_;

    int64_t tileLength_;
    int64_t totalLength_;
    int64_t coreNum_;
    int64_t sortNum_;

    int64_t expertStart_ = 0;
    int64_t expertEnd_ = 0;
    int64_t rowIdxType_ = 0;

    static constexpr int64_t bufferNum_ = 1;
    static constexpr int64_t DST_BLK_STRIDE = 1;
    static constexpr int64_t DST_REP_STRIDE = 8;
};

__aicore__ inline void MoeSortOneCore::CopyIn()
{
    LocalTensor<int32_t> inLocal = sortDataCopyInQueue_.AllocTensor<int32_t>();
    DataCopyExtParams dataCopyParams{static_cast<uint16_t>(1),
                                     static_cast<uint32_t>(this->totalLength_ * sizeof(int32_t)), 0, 0, 0};
    DataCopyPadExtParams dataCopyPadParams{false, 0, 0, 0};
    DataCopyPad(inLocal[0], expertIdxGm_, dataCopyParams, dataCopyPadParams);
    LocalTensor<int32_t> rowIdxLocal = inLocal[this->sortNum_];
    ArithProgression<int32_t>(rowIdxLocal, 0, 1, this->sortNum_);
    sortDataCopyInQueue_.EnQue(inLocal);
}

__aicore__ inline void MoeSortOneCore::SortCompute()
{
    LocalTensor<int32_t> inLocal = sortDataCopyInQueue_.DeQue<int32_t>();
    LocalTensor<int32_t> expertIdx = inLocal[0];
    LocalTensor<float> expertIdxFp32 = expertIdx.ReinterpretCast<float>();
    Cast(expertIdxFp32, expertIdx, RoundMode::CAST_ROUND, this->tileLength_);

    uint16_t repeatTimes = Ceil(this->tileLength_, FLOAT_REG_TENSOR_LENGTH);
    uint32_t sreg = static_cast<uint32_t>(this->tileLength_);
    __local_mem__ float *inUbAddr = (__local_mem__ float *)expertIdxFp32.GetPhyAddr();
    float cmpScalar = static_cast<float>(expertStart_);
    float negone = static_cast<float>(-1);

    __VEC_SCOPE__
    {
        MicroAPI::MaskReg maskRegLoop, cmpMaskReg;
        MicroAPI::MaskReg pregMain = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();

        MicroAPI::RegTensor<float> inRegToFloat, infFloat, vDstReg0;
        Duplicate(infFloat, static_cast<float>(MIN_FP32), pregMain);

        for (uint16_t i = 0; i < repeatTimes; i++) {
            maskRegLoop = MicroAPI::UpdateMask<float>(sreg);
            MicroAPI::DataCopy(inRegToFloat, inUbAddr + i * FLOAT_REG_TENSOR_LENGTH);
            MicroAPI::CompareScalar<float, CMPMODE::LT>(cmpMaskReg, inRegToFloat, cmpScalar, maskRegLoop);
            MicroAPI::Muls(inRegToFloat, inRegToFloat, negone, maskRegLoop);
            MicroAPI::Select(vDstReg0, infFloat, inRegToFloat, cmpMaskReg);
            MicroAPI::DataCopy(inUbAddr + i * FLOAT_REG_TENSOR_LENGTH, vDstReg0, maskRegLoop);
        }
    }

    int64_t duplicateNum = this->totalLength_ % ONE_REPEAT_SORT_NUM;
    if (duplicateNum > 0) {
        int duplicateIndex = this->totalLength_ - duplicateNum;
        uint64_t mask0 = (UINT64_MAX << duplicateNum) & (UINT64_MAX >> ONE_REPEAT_SORT_NUM);
        uint64_t mask[2] = {mask0, 0};
        Duplicate(expertIdxFp32[duplicateIndex], MIN_FP32, mask, 1, DST_BLK_STRIDE, DST_REP_STRIDE);
    }

    LocalTensor<float> concatLocal;
    LocalTensor<float> tempTensor = tempBuffer_.Get<float>(GetSortLen<float>(this->sortNum_));
    Concat(concatLocal, expertIdxFp32, tempTensor, this->sortNum_ / ONE_REPEAT_SORT_NUM);

    LocalTensor<float> sortedLocal = sortedBuffer_.Get<float>(GetSortLen<float>(this->sortNum_));
    LocalTensor<uint32_t> sourceRowLocal;
    sourceRowLocal = inLocal[this->sortNum_].ReinterpretCast<uint32_t>();
    Sort<float, true>(sortedLocal, concatLocal, sourceRowLocal, tempTensor, this->sortNum_ / ONE_REPEAT_SORT_NUM);

    LocalTensor<float> outLocal = sortDataCopyOutQueue_.AllocTensor<float>();
    LocalTensor<float> sortedExpertForSourceRowLocal = outLocal[0];
    LocalTensor<uint32_t> expandDstToSrcRowLocal = outLocal[this->sortNum_].ReinterpretCast<uint32_t>();
    Extract(sortedExpertForSourceRowLocal, expandDstToSrcRowLocal, sortedLocal, this->sortNum_ / ONE_REPEAT_SORT_NUM);
    Muls(sortedExpertForSourceRowLocal, sortedExpertForSourceRowLocal, (float)-1, this->tileLength_);

    LocalTensor<int32_t> expertForSourceRowLocalInt32 = sortedExpertForSourceRowLocal.ReinterpretCast<int32_t>();
    Cast(expertForSourceRowLocalInt32, sortedExpertForSourceRowLocal, RoundMode::CAST_ROUND, this->tileLength_);
    sortDataCopyOutQueue_.EnQue<float>(outLocal);
    sortDataCopyInQueue_.FreeTensor(inLocal);
}

__aicore__ inline void MoeSortOneCore::CopyOut()
{
    LocalTensor<int32_t> outLocal = sortDataCopyOutQueue_.DeQue<int32_t>();
    DataCopyParams intriParams;
    intriParams.blockCount = 1;
    intriParams.blockLen = this->totalLength_ * sizeof(int32_t);
    DataCopyPad(sortedexpertIdxGm_, outLocal[0], intriParams);
    DataCopyPad(expandedRowIdxGm_, outLocal[this->sortNum_], intriParams);
    sortDataCopyOutQueue_.FreeTensor(outLocal);
}

__aicore__ inline void MoeSortOneCore::Init(GM_ADDR expertIdx, GM_ADDR expandedRowIdx, GM_ADDR workspace,
                                            const MoeInitRoutingV3MxQuantArch35TilingData *tilingData, TPipe *tPipe)
{
    this->pipe_ = tPipe;
    this->tileLength_ = Align(tilingData->vbsComputeParamsOp.lastCorePerLoopElements, sizeof(int32_t));
    this->sortNum_ = Ceil(this->tileLength_, ONE_REPEAT_SORT_NUM) * ONE_REPEAT_SORT_NUM;
    this->totalLength_ = tilingData->n * tilingData->k;
    this->coreNum_ = tilingData->coreNum;
    expertStart_ = tilingData->expertStart;
    expertEnd_ = tilingData->expertEnd;
    rowIdxType_ = tilingData->rowIdxType;

    expertIdxGm_.SetGlobalBuffer((__gm__ int32_t *)expertIdx, this->tileLength_);
    sortedexpertIdxGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(workspace),
                                       Align(this->totalLength_, sizeof(int32_t)));
    if (rowIdxType_ == SCATTER) {
        expandedRowIdxGm_.SetGlobalBuffer((__gm__ int32_t *)expandedRowIdx, this->tileLength_);
    } else {
        expandedRowIdxGm_.SetGlobalBuffer((__gm__ int32_t *)workspace + Align(this->tileLength_, sizeof(int32_t)),
                                          Align(this->tileLength_, sizeof(int32_t)));
    }

    if (GetBlockIdx() == 0) {
        expertCountTempGm_.SetGlobalBuffer((__gm__ int32_t *)workspace +
                                               Align(tilingData->n * tilingData->k, sizeof(int32_t)) * 2,
                                           tilingData->actualExpertNum);
        InitGlobalMemory(expertCountTempGm_, tilingData->actualExpertNum, 0);
        SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
    }

    // key and value
    int64_t kvFactor = 2;
    int64_t buffSize = this->sortNum_ * sizeof(int32_t) * kvFactor;
    pipe_->InitBuffer(sortDataCopyInQueue_, bufferNum_, buffSize);
    pipe_->InitBuffer(sortDataCopyOutQueue_, bufferNum_, buffSize);
    pipe_->InitBuffer(tempBuffer_, buffSize);
    pipe_->InitBuffer(sortedBuffer_, buffSize);
}

__aicore__ inline void MoeSortOneCore::Process()
{
    if (GetBlockIdx() < 1) {
        CopyIn();
        SortCompute();
        CopyOut();
    }
    AscendC::SyncAll();
}
} // namespace MoeInitRoutingV3MxQuantNs
#endif // MOE_INIT_ROUTING_V3_MX_QUANT_SORT_ONE_CORE_H
