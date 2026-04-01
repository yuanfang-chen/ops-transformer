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
 * \file moe_init_routing_v3_mx_quant_mrgsort.h
 * \brief Merge sort helper class for MoeInitRoutingV3MxQuant
 */
#ifndef MOE_INIT_ROUTING_V3_MX_QUANT_MRGSORT_H
#define MOE_INIT_ROUTING_V3_MX_QUANT_MRGSORT_H

#include "moe_init_routing_v3_mx_quant_common.h"

namespace MoeInitRoutingV3MxQuantNs {
using namespace AscendC;

struct MoeMrgsortParam {
    int64_t perListElements;
    int64_t lastListElements;
    int64_t oneLoopMaxElements;
};

class MoeMrgsort {
public:
    __aicore__ inline MoeMrgsort(){};
    __aicore__ inline void Init(MoeMrgsortParam *param);
    __aicore__ inline void Process();
    __aicore__ inline void SetInput(GlobalTensor<float> &gmInput, LocalTensor<float> &ubInput);
    __aicore__ inline void SetOutput(GlobalTensor<float> &gmOutput, LocalTensor<float> &ubOutput);

private:
    __aicore__ inline void CopyIn();
    __aicore__ inline void UpdateMrgParam();
    __aicore__ inline void MrgsortCompute();
    __aicore__ inline void UpdateSortInfo();
    __aicore__ inline void CopyOut();
    __aicore__ inline void ClearCache();

private:
    MoeMrgsortParam *param_ = nullptr;

    GlobalTensor<float> gmInputs_[4];
    GlobalTensor<float> gmOutput_;

    LocalTensor<float> ubInputs_[4];
    LocalTensor<float> ubOutput_;

    int64_t listNum_{0};
    int64_t remainListNum_{0};
    int64_t outOffset_{0};
    int64_t offsets_[4];
    int64_t listRemainElements_[4];
    int64_t lengths_[4];
    int64_t allRemainElements_{0};
    int64_t curLoopSortedNum_{0};

    // for MrgSort
    uint16_t validBitTail_{0};
    uint16_t elementCountListTail_[4];
    uint32_t listSortedNums_[4];
    LocalTensor<float> tmpUbInputs_[4];
};

__aicore__ inline void MoeMrgsort::ClearCache()
{
    this->listNum_ = 0;
    this->allRemainElements_ = 0;
    this->outOffset_ = 0;
}

__aicore__ inline void MoeMrgsort::SetInput(GlobalTensor<float> &gmInput, LocalTensor<float> &ubInput)
{
    this->gmInputs_[listNum_] = gmInput;
    this->ubInputs_[listNum_] = ubInput;
    this->listNum_ += 1;
}

__aicore__ inline void MoeMrgsort::SetOutput(GlobalTensor<float> &gmOutput, LocalTensor<float> &ubOutput)
{
    this->gmOutput_ = gmOutput;
    this->ubOutput_ = ubOutput;
}

__aicore__ inline void MoeMrgsort::UpdateMrgParam()
{
    if (this->remainListNum_ == MERGE_LIST_TWO) {
        elementCountListTail_[MERGE_LIST_IDX_TWO] = 0;
        elementCountListTail_[MERGE_LIST_IDX_THREE] = 0;
        validBitTail_ = 0b0011;
    } else if (this->remainListNum_ == MERGE_LIST_THREE) {
        elementCountListTail_[MERGE_LIST_IDX_THREE] = 0;
        validBitTail_ = 0b0111;
    } else if (this->remainListNum_ == MERGE_LIST_FOUR) {
        validBitTail_ = 0b1111;
    } else {
        validBitTail_ = 0b0001;
    }
}

__aicore__ inline void MoeMrgsort::CopyIn()
{
    this->remainListNum_ = 0;
    event_t eventIdMte3ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
    WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
    for (int64_t i = 0, j = 0; i < listNum_; i++) {
        lengths_[i] = Min(param_->oneLoopMaxElements, listRemainElements_[i]);
        if (lengths_[i] > 0) {
            DataCopy(this->ubInputs_[i], this->gmInputs_[i][offsets_[i]],
                     Align(GetSortLen<float>(lengths_[i]), sizeof(float)));
            tmpUbInputs_[j] = this->ubInputs_[i];
            elementCountListTail_[j] = lengths_[i];
            this->remainListNum_ += 1;
            j++;
        }
    }
}

__aicore__ inline void MoeMrgsort::MrgsortCompute()
{
    event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    if (this->remainListNum_ == MERGE_LIST_TWO) {
        MrgSortSrcList sortListTail =
            MrgSortSrcList(tmpUbInputs_[0], tmpUbInputs_[1], tmpUbInputs_[0], tmpUbInputs_[0]);
        MrgSort<float, true>(this->ubOutput_, sortListTail, elementCountListTail_, listSortedNums_, validBitTail_, 1);
    } else if (this->remainListNum_ == MERGE_LIST_THREE) {
        MrgSortSrcList sortListTail =
            MrgSortSrcList(tmpUbInputs_[0], tmpUbInputs_[1], tmpUbInputs_[MERGE_LIST_IDX_TWO], tmpUbInputs_[0]);
        MrgSort<float, true>(this->ubOutput_, sortListTail, elementCountListTail_, listSortedNums_, validBitTail_, 1);
    } else if (this->remainListNum_ == MERGE_LIST_FOUR) {
        MrgSortSrcList sortListTail = MrgSortSrcList(tmpUbInputs_[0], tmpUbInputs_[1],
                                                     tmpUbInputs_[MERGE_LIST_IDX_TWO], tmpUbInputs_[MERGE_LIST_IDX_THREE]);
        MrgSort<float, true>(this->ubOutput_, sortListTail, elementCountListTail_, listSortedNums_, validBitTail_, 1);
    } else {
        DataCopy(this->ubOutput_, this->tmpUbInputs_[0],
                 Align(GetSortLen<float>(elementCountListTail_[0]), sizeof(float)));
        listSortedNums_[0] = elementCountListTail_[0];
    }
}

__aicore__ inline void MoeMrgsort::UpdateSortInfo()
{
    curLoopSortedNum_ = 0;
    for (int64_t i = 0, j = 0; i < listNum_; i++) {
        if (lengths_[i] > 0) {
            listRemainElements_[i] -= listSortedNums_[j];
            allRemainElements_ -= listSortedNums_[j];
            offsets_[i] += GetSortOffset<float>(listSortedNums_[j]);
            curLoopSortedNum_ += listSortedNums_[j];
            j += 1;
        }
    }
}

__aicore__ inline void MoeMrgsort::CopyOut()
{
    DataCopyParams intriParams;
    intriParams.blockCount = 1;
    intriParams.blockLen = GetSortLen<float>(curLoopSortedNum_) * sizeof(float);
    event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
    WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
    DataCopyPad(this->gmOutput_[outOffset_], this->ubOutput_, intriParams);
    outOffset_ += GetSortLen<float>(curLoopSortedNum_);
}

__aicore__ inline void MoeMrgsort::Init(MoeMrgsortParam *param)
{
    this->param_ = param;
    this->remainListNum_ = listNum_;

    for (int64_t i = 0; i < listNum_; i++) {
        offsets_[i] = GetSortOffset<float>(param_->perListElements * i);
        if (i == listNum_ - 1) {
            listRemainElements_[i] = param_->lastListElements;
        } else {
            listRemainElements_[i] = param_->perListElements;
        }
        allRemainElements_ += listRemainElements_[i];
    }
}

__aicore__ inline void MoeMrgsort::Process()
{
    for (; allRemainElements_ > 0;) {
        CopyIn();
        UpdateMrgParam();
        MrgsortCompute();
        UpdateSortInfo();
        CopyOut();
    }

    ClearCache();
}
} // namespace MoeInitRoutingV3MxQuantNs
#endif // MOE_INIT_ROUTING_V3_MX_QUANT_MRGSORT_H
