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
 * \file qbmm_reduce_scatter_add_rms_norm_cast_mte.h
 * \brief qbmm_reduce_scatter_add_rms_norm_cast mte通信kernel代码逻辑
 */

#ifndef QBMM_REDUCE_SCATTER_ADD_RMS_NORM_CAST_MTE_H
#define QBMM_REDUCE_SCATTER_ADD_RMS_NORM_CAST_MTE_H

#include "basic_api/kernel_basic_intf.h"
#include "adv_api/hccl/hccl.h"
#include "adv_api/reduce/sum.h"
#include "adv_api/pad/broadcast.h"
#include "kernel_tiling/kernel_tiling.h"
#include "qbmm_reduce_scatter_add_rms_norm_cast_tiling_data.h"
#include "kernel_operator.h"
#include "moe_distribute_base.h"

namespace QbmmReduceScatterAddRmsNormCastImpl {

#define TemplateMC2TypeClass typename XType, typename YType, typename ScaleType, bool bias

using namespace AscendC;
template <TemplateMC2TypeClass>
class QbmmReduceScatterAddRmsNormCastMte {
public:
    __aicore__ inline QbmmReduceScatterAddRmsNormCastMte() {};
    __aicore__ inline void Init();
    __aicore__ inline void Process();


    static constexpr CubeFormat aFormat = DequantBmm::GetFormat(fFormat);
    static constexpr CubeFormat bFormat = DequantBmm::GetFormat(wFormat);

    using AMatmulType = matmul::MatmulType<TPosition::GM, aFormat, xType, aTrans>;
    using BMatmulType = matmul::MatmulType<TPosition::GM, bFormat, wType, bTrans>;
    using BiasMatmulType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, int32_t>;
    // notice: the tpos of ctype must be ub given by mm api when iterate<false>, but actually we can move data to gm
    // then to ub.
    using CMatmulType = matmul::MatmulType<TPosition::VECIN, CubeFormat::ND, int32_t>;
    matmul::Matmul<AMatmulType, BMatmulType, CMatmulType, BiasMatmulType, MM_DEFAULT_MDL_CFG> mm;

private:
    uint32_t blockIdx_;
    uint32_t tileN_;
    GlobalTensor<XType>aGlobalTensor_;
    GlobalTensor<XType>wGlobalTensor_;
    GlobalTensor<YType>yGlobalTensor_;
    __gm__ HcclOpResParam *winContext_{nullptr};
    // define matmul
};

 /** init function for TilingData of mm1
     */
template <TemplateMC2TypeClass>
__aicore__ inline void InitTilingData(const QuantBatchMatmulV3TilingData *tilingData)
{
    hasBias_ = tilingData->matmulTiling.isBias;
    isPerTensor_ = tilingData->params.isPerTensor;
    biasDtype_ = tilingData->params.biasDtype;
    if (biasDtype_ == DT_INT32 || biasDtype_ == DT_FLOAT) {
        biasDtypeSize_ = sizeof(int32_t);
    } else {
        biasDtypeSize_ = sizeof(half);
    }
    singleCoreBatch_ = tilingData->params.singleCoreBatch;
    m_ = tilingData->matmulTiling.M;
    n_ = tilingData->matmulTiling.N;
    k_ = tilingData->matmulTiling.Ka;
    singleTimeM_ = tilingData->matmulTiling.singleCoreM;  // calcM of each mm iterate
    singleTimeN_ = tilingData->matmulTiling.singleCoreN;  // calcN of each mm iterate
    singleCoreK_ = tilingData->matmulTiling.singleCoreK;
    usedCoreNum_ = tilingData->matmulTiling.usedCoreNum;

    baseM_ = tilingData->matmulTiling.baseM;
    baseN_ = tilingData->matmulTiling.baseN;
    isMouter_ = tilingData->matmulTiling.iterateOrder == 0;
    ubCalcM_ = tilingData->qbmmReduceScatterAddRmsNormCastTilingInfo.ubCalcM;
    ubCalcN_ = tilingData->qbmmReduceScatterAddRmsNormCastTilingInfo.ubCalcN;
    ubTmpBuffer_ = tilingData->qbmmReduceScatterAddRmsNormCastTilingInfo.needUbBuffer;
}

template <TemplateMC2TypeClass>
__aicore__ inline GM_ADDR QbmmReduceScatterAddRmsNormCastMte::GetWindAddrByRankId(const int32_t rankId)
{
    if (rankId == tpRankId_) {
        return (GM_ADDR)(winContext_->localWindowsIn);
    }
    return (GM_ADDR)(((HcclRankRelationResV2*)(winContext_->remoteRes[rankId].nextDevicePtr))->windowsIn)
                        + winDataSizeOffset_;
}

template <TemplateMC2TypeClass>
__aicore__ inline GM_ADDR QbmmReduceScatterAddRmsNormCastMte::GetWindStateAddrByRankId(const int32_t rankId)
{
    if (rankId == tpRankId_) {
        return (GM_ADDR)(winContext_->localWindowsExp);
    }
    return (GM_ADDR)(((HcclRankRelationResV2*)(winContext_->remoteRes[rankId].nextDevicePtr))->windowsExp);
}
template <TemplateMC2TypeClass>
__aicore__ inline void QbmmReduceScatterAddRmsNormCastMte::Init(GM_ADDR x, GM_ADDR x2, GM_ADDR y, GM_ADDR gamma, GM_ADDR scale, 
    GM_ADDR bias, GM_ADDR perTokenScale, GM_ADDR y1, GM_ADDR y2, GM_ADDR xOut, TPipe *pipe, GM_ADDR workSpace, QbmmReduceScatterAddRmsNormCastTilingData *tilingData)
{
    PRINTF("kernel init doing.");
    blockIdx_ = GetBlockIdx();  // AIC_AIV_1_1 by default
    blockIdx_ /= GetTaskRation();

    aGlobalTensor_.SetGlobalBuffer((__gm__ XType*)x);
    bGlovalTensor_.SetGlobalBuffer((__gm__ XType*)x2);
    cGlobalTensor_.SetGlobalBuffer((__gm__ bfloat16*)workSpace);
    // init ub local buffer
    pipe_->InitBuffer(vecQueSrc_, BUFFER_NUM, ubCalcM_ * ubCalcN_ * sizeof(int32_t));
    pipe_->InitBuffer(vecQueTmp_, ubTmpBuffer_);
    pipe_->InitBuffer(vecQueOut_, BUFFER_NUM, ubCalcM_ * ubCalcN_ * sizeof(yType));
    pipe_->InitBuffer(vecQueScale_, BUFFER_NUM, ubCalcN_ * sizeof(scaleType));

    uint32_t singleOffset =
            DequantBmm::Max(singleCoreM_, baseM_) * DequantBmm::Max(singleCoreN_, baseN_) * sizeof(int32_t);
    mm.SetWorkspace(workSpace + m_ * n_ * sizeof(float16)  + blockIdx_ * singleOffset, singleOffset);
    auto contextGM0 = AscendC::GetHcclContext<HCCL_GROUP_ID_0>();
    winContext_ = (__gm__ HcclOpResParam*)AscendC::GetHcclContext<HCCL_GROUP_ID_0>();
}

template <TemplateMC2TypeClass>
__aicore__ inline void QbmmReduceScatterAddRmsNormCastMte::DequantCompute(GlobalTensor<int32_t> &curMmOutGm, uint64_t baseMOfffset,
                                          uint64_t baseNOfffset, uint32_t curAicM, uint32_t curAicN)
{
    LocalTensor<float> dstLocalFp32;
    LocalTensor<float> biasFp32;
    LocalTensor<bfloat16_t> oriBiasBf16;
    LocalTensor<half> oriBiasFp16;
    LocalTensor<float> oriBiasFp32;
    uint32_t curAivM = ubCalcM_;
    // calcN in ub is equal to aicN
    uint32_t curAivN = curAicN;
    uint32_t mUbLoops = DequantBmm::CeilDiv(curAicM, ubCalcM_);
    DataCopyParams gm2UbParams{1, 0, 0, 0};
    DataCopyExtParams ub2GmParams{1, 0, 0, 0, 0};
    DataCopyPadParams padParams;
    DequantParams dequantParams;
    DequantBmm::CalcDequantParams(mUbLoops == 1 ? curAicM : ubCalcM_, curAicN, dequantParams);
    for (uint32_t mUbLoopIdx = 0; mUbLoopIdx < mUbLoops; ++mUbLoopIdx) {
        if (mUbLoopIdx == mUbLoops - 1) {
            curAivM = curAicM - ubCalcM_ * (mUbLoops - 1);
            DequantBmm::CalcDequantParams(curAivM, curAicN, dequantParams, mUbLoops != 1 && curAivM != ubCalcM_);
        }
        LocalTensor<int32_t> srcLocal = vecQueSrc_.AllocTensor<int32_t>();
        LocalTensor<yType> dstLocal = vecQueOut_.AllocTensor<yType>();
        LocalTensor<uint8_t> tmpLocal = vecQueTmp_.Get<uint8_t>();
        // datacopypad 32B aligned
        gm2UbParams.blockLen = curAivN * sizeof(int32_t);
        gm2UbParams.blockCount = curAivM;
        gm2UbParams.srcStride = (curAicN - curAivN) * sizeof(int32_t);
        uint32_t curAicAivOffset = mUbLoopIdx * ubCalcM_ * curAicN;
        DataCopyPad(srcLocal, curMmOutGm[curAicAivOffset], gm2UbParams, padParams);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        if (biasDtype_ != DT_INT32) {
            BiasTensorInit(dstLocalFp32, biasFp32, oriBiasBf16, oriBiasFp16, oriBiasFp32);
            BiasGm2Ub(oriBiasBf16, oriBiasFp16, oriBiasFp32, padParams, baseNOfffset, curAicN);
        }
        if (isPerTensor_) {
            SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
            if (biasDtype_ != DT_INT32) {
                AscendDequant(dstLocalFp32, srcLocal, scaleScalar_, tmpLocal, dequantParams);
            } else {
                AscendDequant(dstLocal, srcLocal, scaleScalar_, tmpLocal, dequantParams);
            }
        } else {
            LocalTensor<scaleType> scaleLocal = vecQueScale_.AllocTensor<scaleType>();
            Bf16ScaleGm2Ub(scaleLocal, scaleGm_, padParams, baseNOfffset, curAicN);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
            if (biasDtype_ != DT_INT32) {
                AscendDequant(dstLocalFp32, srcLocal, scaleLocal, tmpLocal, dequantParams);
            } else {
                AscendDequant(dstLocal, srcLocal, scaleLocal, tmpLocal, dequantParams);
            }
            vecQueScale_.FreeTensor(scaleLocal);
        }
        if (biasDtype_ != DT_INT32) {
            CalBiasAdd(dstLocalFp32, biasFp32, oriBiasBf16, oriBiasFp16, oriBiasFp32, dstLocal, curAivN, curAivM);
        }
        SetFlag<HardEvent::V_MTE3>(EVENT_ID2);
        vecQueSrc_.FreeTensor(srcLocal);
        // dst from ub -> gm
        ub2GmParams.blockLen = curAivN * sizeof(yType);
        ub2GmParams.blockCount = curAivM;
        ub2GmParams.dstStride = (n_ - curAivN) * sizeof(yType);
        uint64_t aivOffset = mUbLoopIdx * ubCalcM_ * n_;
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID2);
        // 发送Token
        // TODO 自己改下
        int32_t remoteRankId = curM / (m_ / tpWorldSize);
        GM_ADDR remoteWinAddr = GetWindAddrByRankId(remoteRankId);
        GlobalTensor<bfloat16_t>remoteTensor;
        remoteTensor.SetGlobalBuffer((__gm__ bfloat16_t*)remoteWinAddr);
        DataCopyPad(remoteTensor[remoteRankId * (m_ / tpWorldSize) * N_ + offsetC_ + baseMOfffset + baseNOfffset + aivOffset],
            dstLocal, ub2GmParams);
        vecQueOut_.FreeTensor(dstLocal);
    }
}

template <TemplateMC2TypeClass>
__aicore__ inline void QbmmReduceScatterAddRmsNormCastMte::DequantNOuterSplitAndSendDataToRemote(uint32_t singleM, uint32_t singleN, uint32_t fixpMtimes,
                                              uint32_t fixpNtimes)
{
    uint32_t curAicOuter = baseN_;
    uint32_t curAicInner = baseM_;
    for (uint32_t fixpOuterIdx = 0; fixpOuterIdx < fixpNtimes; ++fixpOuterIdx) {
        if (fixpOuterIdx == fixpNtimes - 1) {
            curAicOuter = singleN - baseN_ * (fixpNtimes - 1);
        }
        for (uint32_t fixpInnerIdx = 0; fixpInnerIdx < fixpMtimes; ++fixpInnerIdx) {
            if (fixpInnerIdx == fixpMtimes - 1) {
                curAicInner = singleM - baseM_ * (fixpMtimes - 1);
            }
            auto mmOutGm = mm.GetTensorC();
            DequantCompute(mmOutGm, static_cast<uint64_t>(fixpInnerIdx) * baseM_ * n_, fixpOuterIdx * baseN_,
                            curAicInner, curAicOuter);
        }
    }
}
template <TemplateMC2TypeClass>

__aicore__ inline void QbmmReduceScatterAddRmsNormCastMte::Process()
{
    uint32_t fixpMtimes = DequantBmm::CeilDiv(singleM, baseM_);
    uint32_t fixpNTimes = DequantBmm::CeilDiv(singleN, baseN_);
    for (uint32_t idx = 0; idx < tileN_; idx++) {
        mm.SetTensorA(aGlobalTensor_[(blockIdx_ / 2) * singleCoreM_ * singleCoreN_ + idx * baseN_]);
        mm.SetTensorB(bGlovalTensor_[(blockIdx_ / 2) / singleCoreM_ * tileN_ * baseN_]);
        mm.template Iterate<false>();
    }
    DequantNOuterSplitAndSendDataToRemote(singleM, singleN, fixpMtimes, fixpNTimes);
    // 后续 按对应 N 发flag对应接受切N，去除全核同步
    SyncAll<true>();
    // 可以先reset一下ubbuffer 重新分配
    SendFlagToRemote();
    // 
    ReduceRes();
    AddRmsNormCast();
    PRINTF("kernel process doing.");
}
} // QbmmReduceScatterAddRmsNormCastImpl
#endif  // QBMM_REDUCE_SCATTER_ADD_RMS_NORM_CAST_MTE_H