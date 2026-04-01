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
 * \file moe_init_routing_v3_mx_quant_sort_multi_core.h
 * \brief Multi-core sort stage for MoeInitRoutingV3MxQuant
 */
#ifndef MOE_INIT_ROUTING_V3_MX_QUANT_SORT_MULTI_CORE_H
#define MOE_INIT_ROUTING_V3_MX_QUANT_SORT_MULTI_CORE_H

#include "moe_init_routing_v3_mx_quant_common.h"
#include "moe_init_routing_v3_mx_quant_mrgsort.h"
#include "moe_init_routing_v3_mx_quant_mrgsort_out.h"

namespace MoeInitRoutingV3MxQuantNs {
using namespace AscendC;

class MoeSortMultiCore {
public:
    __aicore__ inline MoeSortMultiCore(){};
    __aicore__ inline void Init(GM_ADDR expertIdx, GM_ADDR expandedRowIdx, GM_ADDR workspace,
                                const MoeInitRoutingV3MxQuantArch35TilingData *tilingData, TPipe *tPipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void VBSProcess();
    __aicore__ inline void UBSortProcess(int64_t progress, int64_t size, int64_t sortNum);
    __aicore__ inline void OneCoreVMSProcess(int64_t listNum, int64_t perListElements, int64_t lastListElements);
    __aicore__ inline void VMSProcess();
    __aicore__ inline void SortOutProcess();
    __aicore__ inline void VBSCopyIn(int64_t progress, int64_t size, int64_t sortNum);
    __aicore__ inline void UBSortCompute(int64_t progress, int64_t size, int64_t sortNum);
    __aicore__ inline void VBSCopyOut(int64_t progress, int64_t size, int64_t sortNum);
    __aicore__ inline void InitMoeMrgSort(MoeMrgsort *sorter, int64_t listNum, int64_t coreOffset, int64_t loopOffset);
    __aicore__ inline void InitMoeMrgSortOut(MoeMrgsortOut *sorter, int64_t listNum, int64_t coreOffset);

private:
    // Base members (inlined from MoeSortBase)
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

    int64_t expertStart_ = 0;
    int64_t expertEnd_ = 0;
    int64_t n_;
    int64_t k_;
    int64_t rowIdxType_ = 0;

    static constexpr int64_t bufferNum_ = 1;
    static constexpr int64_t DST_BLK_STRIDE = 1;
    static constexpr int64_t DST_REP_STRIDE = 8;
    static constexpr int64_t SYNC_GM_NUM = 2;
    static constexpr int64_t WORK_GM_NUM = 2;

    // Multi-core specific members
    GlobalTensor<float> workspaceGms_[2];

    const MoeV3Arch35VBSComputeTilingData *vbsTilingData_;
    const MoeV3Arch35VMSMiddleComputeTilingData *vmsTilingData_;
    const MoeV3Arch35SortOutComputeTilingData *sortOutTilingData_;

    // for MoeMrgsort
    MoeMrgsort mrgsorter_;
    MoeMrgsortParam mrgsortParam_;

    int64_t blockIdx_;
    int64_t srcWsIndex_ = 0;

    int64_t listNum_;
    int64_t perListElements_;
    int64_t lastListElements_;

    int64_t sortTotalLength_;
    int64_t sortCoreLoops_;
    int64_t sortCoreLoopElements_;
    int64_t sortCoreLastLoopElements_;

    static constexpr int64_t MAX_MRGSORT_LIST = 4;
};

__aicore__ inline void MoeSortMultiCore::VBSCopyIn(int64_t progress, int64_t size, int64_t sortNum)
{
    LocalTensor<int32_t> inLocal = sortDataCopyInQueue_.AllocTensor<int32_t>();
    int64_t inOffset = progress * sortCoreLoopElements_;
    DataCopyExtParams dataCopyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(size * sizeof(int32_t)), 0, 0, 0};
    DataCopyPadExtParams<int32_t> dataCopyPadParams{false, 0, 0, 0};
    DataCopyPad(inLocal[0], expertIdxGm_[inOffset], dataCopyParams, dataCopyPadParams);

    LocalTensor<int32_t> rowIdxLocal = inLocal[sortNum];
    int64_t startValue = this->blockIdx_ * this->vbsTilingData_->perCoreElements + inOffset;
    SetWaitFlag<HardEvent::MTE3_S>(HardEvent::MTE3_S);
    ArithProgression<int32_t>(rowIdxLocal, startValue, 1, size);
    sortDataCopyInQueue_.EnQue(inLocal);
}

__aicore__ inline void MoeSortMultiCore::UBSortCompute(int64_t progress, int64_t size, int64_t sortNum)
{
    LocalTensor<int32_t> inLocal = sortDataCopyInQueue_.DeQue<int32_t>();
    LocalTensor<int32_t> expertForSourceRowLocal = inLocal[0];
    LocalTensor<float> expertForSourceRowLocalFp32;

    expertForSourceRowLocalFp32 = expertForSourceRowLocal.ReinterpretCast<float>();
    Cast(expertForSourceRowLocalFp32, expertForSourceRowLocal, RoundMode::CAST_ROUND, sortNum);

    uint16_t repeatTimes = Ceil(sortNum, FLOAT_REG_TENSOR_LENGTH);
    uint32_t sreg = static_cast<uint32_t>(sortNum);
    __local_mem__ float *inUbAddr = (__local_mem__ float *)expertForSourceRowLocalFp32.GetPhyAddr();
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

    int64_t duplicateNum = size % ONE_REPEAT_SORT_NUM;
    if (duplicateNum > 0) {
        int duplicateIndex = size - duplicateNum;
        uint64_t mask0 = UINT64_MAX;
        mask0 = mask0 << duplicateNum;
        mask0 = mask0 & (UINT64_MAX >> ONE_REPEAT_SORT_NUM);
        uint64_t mask[2] = {mask0, 0};
        Duplicate(expertForSourceRowLocalFp32[duplicateIndex], MIN_FP32, mask, 1, DST_BLK_STRIDE, DST_REP_STRIDE);
    }

    LocalTensor<float> concatLocal = expertForSourceRowLocalFp32;
    LocalTensor<float> sortedLocal = sortedBuffer_.Get<float>(GetSortLen<float>(sortNum));
    LocalTensor<float> outLocal = sortDataCopyOutQueue_.AllocTensor<float>();
    LocalTensor<uint32_t> sourceRowLocal;
    sourceRowLocal = inLocal[sortNum].ReinterpretCast<uint32_t>();
    Sort<float, true>(outLocal, concatLocal, sourceRowLocal, sortedLocal, sortNum / ONE_REPEAT_SORT_NUM);

    sortDataCopyOutQueue_.EnQue<float>(outLocal);
    sortDataCopyInQueue_.FreeTensor(inLocal);
}

__aicore__ inline void MoeSortMultiCore::VBSCopyOut(int64_t progress, int64_t size, int64_t sortNum)
{
    LocalTensor<float> outLocal = sortDataCopyOutQueue_.DeQue<float>();
    DataCopy(workspaceGms_[0][this->blockIdx_ * GetSortLen<float>(this->vbsTilingData_->perCoreElements) +
                              GetSortLen<float>(progress * sortCoreLoopElements_)],
             outLocal, Align(GetSortLen<float>(size), sizeof(float)));
    sortDataCopyOutQueue_.FreeTensor(outLocal);
}

__aicore__ inline void MoeSortMultiCore::InitMoeMrgSort(MoeMrgsort *sorter, int64_t listNum, int64_t coreOffset,
                                                        int64_t loopOffset)
{
    GlobalTensor<float> srcWsGm = workspaceGms_[srcWsIndex_][blockIdx_ * coreOffset + loopOffset];
    LocalTensor<float> inLocal = sortDataCopyInQueue_.AllocTensor<float>();
    LocalTensor<float> outLocal = sortDataCopyOutQueue_.AllocTensor<float>();
    for (int64_t i = 0; i < listNum; i++) {
        LocalTensor<float> inLocalT = inLocal[GetSortLen<float>(this->sortOutTilingData_->oneLoopMaxElements) * i];
        sorter->SetInput(srcWsGm, inLocalT);
    }
    GlobalTensor<float> dstWsGm = workspaceGms_[1 - srcWsIndex_][blockIdx_ * coreOffset + loopOffset];
    sorter->SetOutput(dstWsGm, outLocal);
    sortDataCopyInQueue_.FreeTensor(inLocal);
    sortDataCopyOutQueue_.FreeTensor(outLocal);
}

__aicore__ inline void MoeSortMultiCore::InitMoeMrgSortOut(MoeMrgsortOut *sorter, int64_t listNum, int64_t coreOffset)
{
    GlobalTensor<float> srcWsGm = workspaceGms_[srcWsIndex_];
    LocalTensor<float> inLocal = sortDataCopyInQueue_.AllocTensor<float>();
    LocalTensor<float> outLocal = sortDataCopyOutQueue_.AllocTensor<float>();

    for (int64_t i = 0; i < listNum; i++) {
        LocalTensor<float> inLocalT = inLocal[GetSortLen<float>(this->sortOutTilingData_->oneLoopMaxElements) * i];
        sorter->SetInput(srcWsGm, inLocalT);
    }

    LocalTensor<float> outLocalV = outLocal[this->sortOutTilingData_->oneLoopMaxElements * MAX_MRGSORT_LIST];
    sorter->SetOutput(this->sortedexpertIdxGm_, this->expandedRowIdxGm_, outLocal, outLocalV);

    LocalTensor<float> tempBuf =
        sortedBuffer_.Get<float>(GetSortLen<float>(this->sortOutTilingData_->oneLoopMaxElements) * MAX_MRGSORT_LIST);
    sorter->SetBuffer(tempBuf);
    sortDataCopyInQueue_.FreeTensor(inLocal);
    sortDataCopyOutQueue_.FreeTensor(outLocal);
}

__aicore__ inline void MoeSortMultiCore::OneCoreVMSProcess(int64_t listNum, int64_t perListElements,
                                                           int64_t lastListElements)
{
    int64_t coreOffset = GetSortLen<float>(this->vbsTilingData_->perCoreElements);
    mrgsortParam_.oneLoopMaxElements = this->sortOutTilingData_->oneLoopMaxElements;

    for (int64_t i = 0; listNum >= 1; i++) {
        int64_t loops = (listNum + MAX_MRGSORT_LIST - 1) / MAX_MRGSORT_LIST;
        int64_t remainListNum = listNum - (loops - 1) * MAX_MRGSORT_LIST;

        mrgsortParam_.perListElements = perListElements;
        mrgsortParam_.lastListElements = perListElements;

        int64_t loopOffset = GetSortLen<float>(mrgsortParam_.perListElements * MAX_MRGSORT_LIST);
        for (int64_t loop = 0; loop < loops - 1; loop++) {
            InitMoeMrgSort(&mrgsorter_, MAX_MRGSORT_LIST, coreOffset, loop * loopOffset);
            mrgsorter_.Init(&mrgsortParam_);
            mrgsorter_.Process();
        }

        mrgsortParam_.perListElements = perListElements;
        mrgsortParam_.lastListElements = lastListElements;
        InitMoeMrgSort(&mrgsorter_, remainListNum, coreOffset, (loops - 1) * loopOffset);
        mrgsorter_.Init(&mrgsortParam_);
        mrgsorter_.Process();

        listNum = loops;
        lastListElements = perListElements * (remainListNum - 1) + lastListElements;
        perListElements = perListElements * MAX_MRGSORT_LIST;
        srcWsIndex_ = (srcWsIndex_ + 1) % WORK_GM_NUM;
        if (loops == 1) {
            break;
        }
    }
}

__aicore__ inline void MoeSortMultiCore::UBSortProcess(int64_t progress, int64_t size, int64_t sortNum)
{
    VBSCopyIn(progress, size, sortNum);
    UBSortCompute(progress, size, sortNum);
    VBSCopyOut(progress, size, sortNum);
}

__aicore__ inline void MoeSortMultiCore::VBSProcess()
{
    if (this->blockIdx_ < this->vbsTilingData_->needCoreNum) {
        int64_t sortNum = Ceil(sortCoreLoopElements_, ONE_REPEAT_SORT_NUM) * ONE_REPEAT_SORT_NUM;
        for (int64_t loop = 0; loop < sortCoreLoops_ - 1; loop++) {
            UBSortProcess(loop, sortCoreLoopElements_, sortNum);
        }

        sortNum = Ceil(sortCoreLastLoopElements_, ONE_REPEAT_SORT_NUM) * ONE_REPEAT_SORT_NUM;
        UBSortProcess(sortCoreLoops_ - 1, sortCoreLastLoopElements_, sortNum);

        if (sortCoreLoops_ > 1) {
            OneCoreVMSProcess(sortCoreLoops_, sortCoreLoopElements_, sortCoreLastLoopElements_);
        }
    }
    AscendC::SyncAll();
}

__aicore__ inline void MoeSortMultiCore::VMSProcess()
{
    int64_t currentStageNeedCoreNum = this->vmsTilingData_->needCoreNum;
    perListElements_ = this->vbsTilingData_->perCoreElements;
    lastListElements_ = this->vbsTilingData_->lastCoreElements;
    listNum_ = this->vbsTilingData_->needCoreNum;

    for (; listNum_ > MAX_MRGSORT_LIST;) {
        currentStageNeedCoreNum = Ceil(listNum_, MAX_MRGSORT_LIST);
        int64_t coreOffset = GetSortLen<float>(perListElements_ * MAX_MRGSORT_LIST);
        int64_t remainListNum = listNum_ - (currentStageNeedCoreNum - 1) * MAX_MRGSORT_LIST;

        if (this->blockIdx_ < currentStageNeedCoreNum - 1) {
            mrgsortParam_.perListElements = perListElements_;
            mrgsortParam_.lastListElements = perListElements_;
            mrgsortParam_.oneLoopMaxElements = this->sortOutTilingData_->oneLoopMaxElements;
            InitMoeMrgSort(&mrgsorter_, MAX_MRGSORT_LIST, coreOffset, 0);
            mrgsorter_.Init(&mrgsortParam_);
            mrgsorter_.Process();
        } else if (this->blockIdx_ == currentStageNeedCoreNum - 1) {
            mrgsortParam_.perListElements = perListElements_;
            mrgsortParam_.lastListElements = lastListElements_;
            mrgsortParam_.oneLoopMaxElements = this->sortOutTilingData_->oneLoopMaxElements;
            InitMoeMrgSort(&mrgsorter_, remainListNum, coreOffset, 0);
            mrgsorter_.Init(&mrgsortParam_);
            mrgsorter_.Process();
        }
        listNum_ = currentStageNeedCoreNum;
        currentStageNeedCoreNum = Ceil(listNum_, MAX_MRGSORT_LIST);
        srcWsIndex_ = (srcWsIndex_ + 1) % WORK_GM_NUM;

        lastListElements_ = perListElements_ * (remainListNum - 1) + lastListElements_;
        perListElements_ = perListElements_ * MAX_MRGSORT_LIST;

        AscendC::SyncAll();
    }
}

__aicore__ inline void MoeSortMultiCore::SortOutProcess()
{
    if (this->blockIdx_ < 1) {
        mrgsortParam_.perListElements = perListElements_;
        mrgsortParam_.lastListElements = lastListElements_;
        mrgsortParam_.oneLoopMaxElements = this->sortOutTilingData_->oneLoopMaxElements;

        MoeMrgsortOut sorter;
        InitMoeMrgSortOut(&sorter, listNum_, GetSortLen<float>(perListElements_));
        sorter.Init(&mrgsortParam_, pipe_);
        sorter.Process();
    }
    AscendC::SyncAll();
}

__aicore__ inline void MoeSortMultiCore::Init(GM_ADDR expertIdx, GM_ADDR expandedRowIdx, GM_ADDR workspace,
                                              const MoeInitRoutingV3MxQuantArch35TilingData *tilingData, TPipe *tPipe)
{
    this->totalLength_ = tilingData->n * tilingData->k;
    this->coreNum_ = tilingData->coreNum;
    this->vbsTilingData_ = &(tilingData->vbsComputeParamsOp);
    this->vmsTilingData_ = &(tilingData->vmsMiddleComputeParamsOp);
    this->sortOutTilingData_ = &(tilingData->sortOutComputeParamsOp);

    this->blockIdx_ = GetBlockIdx();
    this->tileLength_ = this->vbsTilingData_->perCorePerLoopElements;
    this->sortTotalLength_ = this->vbsTilingData_->perCoreElements;
    if (this->blockIdx_ == tilingData->vbsComputeParamsOp.needCoreNum - 1) {
        this->tileLength_ = this->vbsTilingData_->lastCorePerLoopElements;
        this->sortTotalLength_ = this->vbsTilingData_->lastCoreElements;
    }
    this->n_ = tilingData->n;
    this->k_ = tilingData->k;

    expertStart_ = tilingData->expertStart;
    expertEnd_ = tilingData->expertEnd;
    rowIdxType_ = tilingData->rowIdxType;

    // VBS param init
    if (this->blockIdx_ == this->vbsTilingData_->needCoreNum - 1) {
        sortCoreLoops_ = this->vbsTilingData_->lastCoreLoops;
        sortCoreLoopElements_ = this->vbsTilingData_->lastCorePerLoopElements;
        sortCoreLastLoopElements_ = this->vbsTilingData_->lastCoreLastLoopElements;
    } else {
        sortCoreLoops_ = this->vbsTilingData_->perCoreLoops;
        sortCoreLoopElements_ = this->vbsTilingData_->perCorePerLoopElements;
        sortCoreLastLoopElements_ = this->vbsTilingData_->perCoreLastLoopElements;
    }

    this->pipe_ = tPipe;
    expertIdxGm_.SetGlobalBuffer((__gm__ int32_t *)expertIdx +
                                     this->blockIdx_ * tilingData->vbsComputeParamsOp.perCoreElements,
                                 this->sortTotalLength_);
    sortedexpertIdxGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(workspace),
                                       Align(this->totalLength_, sizeof(int32_t)));
    if (rowIdxType_ == SCATTER) {
        expandedRowIdxGm_.SetGlobalBuffer((__gm__ int32_t *)expandedRowIdx,
                                          Align(this->totalLength_, sizeof(int32_t)));
    } else {
        expandedRowIdxGm_.SetGlobalBuffer((__gm__ int32_t *)workspace + Align(this->totalLength_, sizeof(int32_t)),
                                          Align(this->totalLength_, sizeof(int32_t)));
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
    workspaceGms_[0].SetGlobalBuffer((__gm__ float *)workspace + Align(this->totalLength_, sizeof(int32_t)) * 2 +
                                         tilingData->actualExpertNum,
                                     Align(this->totalLength_, sizeof(int32_t)) * kvFactor);
    workspaceGms_[1].SetGlobalBuffer((__gm__ float *)workspace +
                                         Align(this->totalLength_, sizeof(int32_t)) * (kvFactor + 2) +
                                         tilingData->actualExpertNum,
                                     Align(this->totalLength_, sizeof(int32_t)) * kvFactor);

    int64_t bufferSize =
        Ceil(Max(this->sortOutTilingData_->oneLoopMaxElements * MAX_MRGSORT_LIST, sortCoreLoopElements_),
             ONE_REPEAT_SORT_NUM) *
        ONE_REPEAT_SORT_NUM * sizeof(int32_t) * kvFactor;
    pipe_->InitBuffer(sortDataCopyInQueue_, bufferNum_, bufferSize);
    pipe_->InitBuffer(sortDataCopyOutQueue_, bufferNum_, bufferSize);
    pipe_->InitBuffer(sortedBuffer_, bufferSize);
    pipe_->InitBuffer(tempBuffer_, bufferSize);
}

__aicore__ inline void MoeSortMultiCore::Process()
{
    VBSProcess();
    VMSProcess();
    SortOutProcess();
}
} // namespace MoeInitRoutingV3MxQuantNs
#endif // MOE_INIT_ROUTING_V3_MX_QUANT_SORT_MULTI_CORE_H
