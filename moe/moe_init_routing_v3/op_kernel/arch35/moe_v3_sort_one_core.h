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
 * \file moe_v3_sort_one_core.h
 * \brief
 */
#ifndef MOE_V3_SORT_ONE_CORE_H_REGBASE
#define MOE_V3_SORT_ONE_CORE_H_REGBASE

#include "moe_v3_sort_base.h"

namespace MoeInitRoutingV3 {
using namespace AscendC;

class MoeSortOneCore : public MoeSortBase {
public:
    __aicore__ inline MoeSortOneCore(){};
    __aicore__ inline void Init(GM_ADDR expertIdx, GM_ADDR expandedRowIdx, GM_ADDR workspace,
                                const MoeInitRoutingV3Arch35TilingData *tilingData, TPipe *tPipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn();
    __aicore__ inline void SortCompute();
    __aicore__ inline void ExpertCountCompute();
    __aicore__ inline void CopyOut();

private:
    int64_t sortNum;
};

__aicore__ inline void MoeSortOneCore::CopyIn()
{
    LocalTensor<int32_t> inLocal = sortDataCopyInQueue.AllocTensor<int32_t>();
    DataCopyExtParams dataCopyParams{static_cast<uint16_t>(1),
                                     static_cast<uint32_t>(this->totalLength * sizeof(int32_t)), 0, 0, 0};
    DataCopyPadExtParams dataCopyPadParams{false, 0, 0, 0};
    DataCopyPad(inLocal[0], expertIdxGm, dataCopyParams, dataCopyPadParams);
    LocalTensor<int32_t> rowIdxLocal = inLocal[this->sortNum];
    ArithProgression<int32_t>(rowIdxLocal, 0, 1, this->sortNum);
    sortDataCopyInQueue.EnQue(inLocal);
}

// 完成专家索引的预处理、拼接、排序、结果转换与输出
__aicore__ inline void MoeSortOneCore::SortCompute()
{
    // 1 数据获取与类型转换
    LocalTensor<int32_t> inLocal = sortDataCopyInQueue.DeQue<int32_t>();
    LocalTensor<int32_t> expertIdx = inLocal[0];
    LocalTensor<float> expertIdxFp32 = expertIdx.ReinterpretCast<float>();
    Cast(expertIdxFp32, expertIdx, RoundMode::CAST_ROUND, this->tileLength);

    uint16_t repeatTimes = Ceil(this->tileLength, FLOAT_REG_TENSOR_LENGTH);
    uint32_t sreg = static_cast<uint32_t>(this->tileLength);
    __local_mem__ float *inUbAddr = (__local_mem__ float *)expertIdxFp32.GetPhyAddr();
    float cmpScalar = static_cast<float>(expertStart_);
    float cmpEndScalar = static_cast<float>(expertEnd_);
    float negone = static_cast<float>(-1);

    // 2 专家索引预处理（小于expertStart的置为-inf）后续排序时这些无效索引会被排到最后，只保留有效专家索引参与排序。
    __VEC_SCOPE__
    {
        MicroAPI::MaskReg maskRegLoop, cmpMaskReg, cmpEndMaskReg;
        MicroAPI::MaskReg pregMain = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();

        MicroAPI::RegTensor<float> inRegToFloat, infFloat, vDstReg0;
        Duplicate(infFloat, static_cast<float>(MIN_FP32), pregMain);

        for (uint16_t i = 0; i < repeatTimes; i++) {
            // 更新循环掩码
            maskRegLoop = MicroAPI::UpdateMask<float>(sreg);
            // 从本地内存加载数据expertIdx到寄存器
            MicroAPI::DataCopy(inRegToFloat, inUbAddr + i * FLOAT_REG_TENSOR_LENGTH);
            // 比较：专家索引 < expertStart_ 时生成掩码
            MicroAPI::CompareScalar<float, CMPMODE::LT>(cmpMaskReg, inRegToFloat, cmpScalar, maskRegLoop);
            MicroAPI::CompareScalar<float, CMPMODE::GE>(cmpEndMaskReg, inRegToFloat, cmpEndScalar, maskRegLoop);
            // 专家索引乘以-1（后续排序后会还原）
            MicroAPI::Muls(inRegToFloat, inRegToFloat, negone, maskRegLoop);
            // 按掩码选择：满足条件的位置置为MIN_FP32（负无穷），否则保留乘-1后的值
            MicroAPI::Select(vDstReg0, infFloat, inRegToFloat, cmpMaskReg);
            MicroAPI::Select(vDstReg0, infFloat, vDstReg0, cmpEndMaskReg);
            // 将处理后的数据写回本地内存
            MicroAPI::DataCopy(inUbAddr + i * FLOAT_REG_TENSOR_LENGTH, vDstReg0, maskRegLoop);
        }
    }
    // 3 补全长度不足的部分（置为 - inf）对不足的部分（duplicateNum 个元素）填充负无穷（MIN_FP32），保证后续排序的长度对齐。
    // ONE_REPEAT_SORT_NUM 是单次排序的固定长度，填充操作确保数据长度符合硬件排序单元的要求。
    int64_t duplicateNum = this->totalLength % ONE_REPEAT_SORT_NUM;
    if (duplicateNum > 0) {
        int duplicateIndex = this->totalLength - duplicateNum;
        uint64_t mask0 = (UINT64_MAX << duplicateNum) & (UINT64_MAX >> ONE_REPEAT_SORT_NUM);
        uint64_t mask[2] = {mask0, 0};
        Duplicate(expertIdxFp32[duplicateIndex], MIN_FP32, mask, 1, DST_BLK_STRIDE, DST_REP_STRIDE);
    }
    // 4 将预处理后的专家索引张量expertIdxFp32与临时张量 tempTensor拼接为concatLocal目的：扩展张量长度，适配后续排序操作的输入要求
    LocalTensor<float> concatLocal;
    LocalTensor<float> tempTensor = tempBuffer.Get<float>(GetSortLen<float>(this->sortNum));
    Concat(concatLocal, expertIdxFp32, tempTensor, this->sortNum / ONE_REPEAT_SORT_NUM);
    // 5 排序：调用 Sort 函数完成排序，参数 true 表示升序排序（因之前将有效索引乘 - 1，升序后有效索引会按原大小降序排列，符合 MoE 选 TopK 专家的需求）
    // sortedLocal 排序后的张量
    LocalTensor<float> sortedLocal = sortedBuffer.Get<float>(GetSortLen<float>(this->sortNum));
    LocalTensor<uint32_t> sourceRowLocal;
    sourceRowLocal = inLocal[this->sortNum].ReinterpretCast<uint32_t>();
    Sort<float, true>(sortedLocal, concatLocal, sourceRowLocal, tempTensor, this->sortNum / ONE_REPEAT_SORT_NUM);
    // 6 结果提取与还原
    LocalTensor<float> outLocal = sortDataCopyOutQueue.AllocTensor<float>();
    LocalTensor<float> sortedExpertForSourceRowLocal = outLocal[0];
    LocalTensor<uint32_t> expandDstToSrcRowLocal = outLocal[this->sortNum].ReinterpretCast<uint32_t>();
    // 提取排序后的专家索引和对应的行索引
    Extract(sortedExpertForSourceRowLocal, expandDstToSrcRowLocal, sortedLocal, this->sortNum / ONE_REPEAT_SORT_NUM);
    // 将之前乘-1的专家索引还原（乘-1）
    Muls(sortedExpertForSourceRowLocal, sortedExpertForSourceRowLocal, (float)-1, this->tileLength);
    // 7 类型转换与数据输出 / 释放
    LocalTensor<int32_t> expertForSourceRowLocalInt32 = sortedExpertForSourceRowLocal.ReinterpretCast<int32_t>();
    Cast(expertForSourceRowLocalInt32, sortedExpertForSourceRowLocal, RoundMode::CAST_ROUND, this->tileLength);
    sortDataCopyOutQueue.EnQue<float>(outLocal);
    sortDataCopyInQueue.FreeTensor(inLocal);
}

__aicore__ inline void MoeSortOneCore::CopyOut()
{
    LocalTensor<int32_t> outLocal = sortDataCopyOutQueue.DeQue<int32_t>();
    DataCopyParams intriParams;
    intriParams.blockCount = 1;
    intriParams.blockLen = this->totalLength * sizeof(int32_t);
    DataCopyPad(sortedexpertIdxGm, outLocal[0], intriParams);
    DataCopyPad(expandedRowIdxGm, outLocal[this->sortNum], intriParams);
    sortDataCopyOutQueue.FreeTensor(outLocal);
}

__aicore__ inline void MoeSortOneCore::Init(GM_ADDR expertIdx, GM_ADDR expandedRowIdx, GM_ADDR workspace,
                                            const MoeInitRoutingV3Arch35TilingData *tilingData, TPipe *tPipe)
{
    this->pipe = tPipe;
    this->tileLength = Align(tilingData->vbsComputeParamsOp.lastCorePerLoopElements, sizeof(int32_t));
    this->sortNum = Ceil(this->tileLength, ONE_REPEAT_SORT_NUM) * ONE_REPEAT_SORT_NUM;
    this->totalLength = tilingData->n * tilingData->k;
    this->coreNum = tilingData->coreNum;
    expertStart_ = tilingData->expertStart;
    expertEnd_ = tilingData->expertEnd;
    rowIdxType_ = tilingData->rowIdxType;

    expertIdxGm.SetGlobalBuffer((__gm__ int32_t *)expertIdx, this->tileLength);
    sortedexpertIdxGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(workspace),
                                      Align(this->totalLength, sizeof(int32_t)));
    if (rowIdxType_ == SCATTER) {
        expandedRowIdxGm.SetGlobalBuffer((__gm__ int32_t *)expandedRowIdx, this->tileLength);
    } else {
        expandedRowIdxGm.SetGlobalBuffer((__gm__ int32_t *)workspace + Align(this->tileLength, sizeof(int32_t)),
                                         Align(this->tileLength, sizeof(int32_t)));
    }

    if (GetBlockIdx() == 0) {
        expertCountTempGm.SetGlobalBuffer((__gm__ int32_t *)workspace +
                                              Align(tilingData->n * tilingData->k, sizeof(int32_t)) * 2,
                                          tilingData->actualExpertNum);
        InitGlobalMemory(expertCountTempGm, tilingData->actualExpertNum, 0);
        SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
    }

    int64_t coreNum = GetBlockNum();

    // key and value
    int64_t kvFactor = 2;
    int64_t buffSize = this->sortNum * sizeof(int32_t) * kvFactor;
    pipe->InitBuffer(sortDataCopyInQueue, bufferNum, buffSize);
    pipe->InitBuffer(sortDataCopyOutQueue, bufferNum, buffSize);
    pipe->InitBuffer(tempBuffer, buffSize);
    pipe->InitBuffer(sortedBuffer, buffSize);
}

__aicore__ inline void MoeSortOneCore::Process()
{
    if (GetBlockIdx() < 1) {
        CopyIn();
        SortCompute();
        CopyOut();
    }
    this->SyncAll();
}
} // namespace MoeInitRoutingV3
#endif // MOE_V3_SORT_ONE_CORE_H_REGBASE