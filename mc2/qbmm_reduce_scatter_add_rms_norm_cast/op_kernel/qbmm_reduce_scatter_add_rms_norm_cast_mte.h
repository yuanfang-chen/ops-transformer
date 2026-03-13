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
#include "lib/matmul_intf.h"
#include "adv_api/quantization/ascend_dequant.h"
#include "kernel_type.h"
#include "kernel_tiling/kernel_tiling.h"
#include "quant_batch_matmul_v3_base.h"
#if __has_include("../common/inc/kernel/moe_distribute_base.h")
#include "../common/inc/kernel/moe_distribute_base.h"
#include "../common/inc/kernel/mc2_kernel_utils.h"
#else
#include "../../common/inc/kernel/moe_distribute_base.h"
#include "../../common/inc/kernel/mc2_kernel_utils.h"
#endif

namespace QbmmReduceScatterAddRmsNormCastImpl {

#define TemplateMC2TypeClass typename YType, typename ScaleType, typename Y1Type, typename Y2Type, typename XType, typename GammaType, bool isNz
#define TemplateMC2TypeFunc YType, ScaleType, Y1Type, Y2Type, XType, GammaType, isNz
using namespace AscendC;
using namespace AscendC;

constexpr uint32_t BUFFER_NUM = 1U;
constexpr uint32_t UB_ALIGN_BYTES = 32U;
constexpr uint32_t STATE_ALIGN_BYTES = 32U;
constexpr uint32_t NZ_W0 = 32U;
constexpr uint32_t REDUCE_BUF_BYTES = 256U;
constexpr uint32_t FLOAT_UB_ALIGN_NUM = 8U;
constexpr uint32_t BLOCK_LENGTH = 5120U;
constexpr uint32_t EIGHT_BYTES_ALIGN = 8U;
constexpr uint32_t TP_WORLD_SIZE = 4U;
constexpr uint32_t UB_CALCM = 22U;
constexpr uint32_t M_N_TWO_DIMS = 2U;
constexpr float EPSILON = 1e-6f;
constexpr float ONE = 1;
constexpr static uint64_t SYNC_AIC_TO_AIV = 5;

template <AscendC::HardEvent event>
__aicore__ inline void SyncFunc() {
    int32_t eventID = static_cast<int32_t>(GetTPipePtr()->FetchEventID(event));
    AscendC::SetFlag<event>(eventID);
    AscendC::WaitFlag<event>(eventID);
}

__aicore__ inline uint64_t CeilDiv(uint64_t a, uint64_t b)
{
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b;
}

__aicore__ inline constexpr CubeFormat GetFormat(bool isNz)
{
    if (isNz) {
        return CubeFormat::NZ;
    }
    return CubeFormat::ND;
}

template<TemplateMC2TypeClass>
class QbmmReduceScatterAddRmsNormCastMte {
public:
    __aicore__ inline QbmmReduceScatterAddRmsNormCastMte() {};
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR gamma, GM_ADDR scale, GM_ADDR bias, GM_ADDR perTokenScale, GM_ADDR y1Out,
                GM_ADDR y2Out, GM_ADDR xOut, GM_ADDR workspaceGM, TPipe *pipe, const QbmmReduceScatterAddRmsNormCastTilingData *tilingData);
    static constexpr CubeFormat x2Format = GetFormat(isNz);
    using AMatmulType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, int8_t, false>;
    using BMatmulType = matmul::MatmulType<TPosition::GM, x2Format, int8_t, false>;
    using CMatmulType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, int32_t>;
    using BiasMatmulType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, int32_t>;
    
    matmul::MatmulImpl<AMatmulType, BMatmulType, CMatmulType, BiasMatmulType, MM_DEFAULT_MDL_CFG> mm_;
    __aicore__ inline void Process();

protected:
    __aicore__ inline void InitTilingData(const QbmmReduceScatterAddRmsNormCastTilingData *tilingData);
    __aicore__ inline GM_ADDR GetWindAddrByRankId(const int32_t rankId);
    __aicore__ inline void SplitToCore(const uint32_t curSendCnt, const uint32_t curUseAivNum, const uint32_t coreId, uint32_t &startId, uint32_t &endId, uint32_t &sendNum);
    __aicore__ inline GM_ADDR GetWindStateAddrByRankId(const int32_t rankId);
    __aicore__ inline void DequantCompute(GlobalTensor<int32_t> &curMmOutGm, uint32_t curAicM, uint32_t curAicN, uint64_t row, uint64_t col);
    __aicore__ inline void ReadRemoteDataAdd();
    __aicore__ inline void WriteStatusToWin();
    __aicore__ inline void ReadStatus();
    __aicore__ inline void MatmulProcess();
    __aicore__ inline void AddRmsNormAddCompute(uint32_t tokenIndex, uint32_t numCol,
                                                LocalTensor<float>& x1TmpFloatLocal,
                                                LocalTensor<float>& x2TmpFloatLocal,
                                                LocalTensor<float>& addOutTmpFloatLocal,
                                                const DataCopyExtParams& copyExtParams,
                                                const DataCopyPadExtParams<YType>& copyPadExtParams);
    __aicore__ inline void AddRmsNormRmsNormCompute(uint32_t tokenIndex, uint32_t numCol,
                                                    LocalTensor<float>& xFp32, LocalTensor<float>& sqx,
                                                    LocalTensor<float>& gammaLocal,
                                                    const DataCopyExtParams& copyExtParams);
    __aicore__ inline void MMCompute(uint32_t singleM, uint32_t singleN);
    __aicore__ inline void CalcMAxisOffset(uint32_t mCoreIndex, uint32_t nLoops);
    __aicore__ inline void CalcNAxisOffset(uint32_t loopIdx);

    TPipe *tpipe_{nullptr};
    TBuf<> writeStateBuf_;
    TBuf<> readStateBuf_;
    TBuf<> resetStateBuf_;
    TBuf<> sumFp32Buf_;
    TBuf<> tokenFp32Buf_;

    // addrmsnormcast
    TBuf<> tokenBuf_;
    TBuf<> gammaBuf_;
    TBuf<> gammaFp32Buf_;
    TBuf<> rowTmpFloatBuf_;
    TBuf<> mulBuf_;
    TBuf<> resFp32Buf_;
    TBuf<TPosition::VECCALC> reduceFp32Buf_;

    // define the que
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, BUFFER_NUM> tokenQueue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> vecQueSrc_;
    TQue<QuePosition::VECIN, BUFFER_NUM> vecQueScale_;
    TQue<QuePosition::VECIN, BUFFER_NUM> vecQuePertokenScale_;
    TQue<QuePosition::VECIN, BUFFER_NUM> vecQueBias_;
    TBuf<TPosition::VECCALC> vecQueTmp_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> vecQueOut_;
    TBuf<TPosition::VECCALC> broadcastFp32Tmp_;
    TBuf<TPosition::VECCALC> biasFp32Tmp_;
    TBuf<TPosition::VECCALC> outFp32Tmp_;
    
    GlobalTensor<int8_t> x1GM_;
    GlobalTensor<int8_t> x2GM_;
    // 用来存储通信tensor
    GlobalTensor<YType> yGM_;
    // 用来存储reducescatter输出
    GlobalTensor<YType> xOutGM_;
    GlobalTensor<YType> y2OutGM_;
    GlobalTensor<float> y1OutGM_;
    GlobalTensor<GammaType> gammaGM_;
    GlobalTensor<int32_t> mmOutGm_;
    GlobalTensor<ScaleType> scaleGMTensor_;
    GM_ADDR workspaceAddr_;
    GM_ADDR perTokenScaleAddr_;

    // LocalTensor<bfloat16_t> tmpTensor_;
    LocalTensor<int32_t> srcLocalTensor_;
    LocalTensor<YType> tokenTensor_;
    LocalTensor<float> rowTmpFloatLocal_;
    LocalTensor<float> mulBufLocal_;

    uint32_t singleTpSize_;
    uint32_t singleM_;
    uint32_t singleN_;
    uint32_t tileN_;
    uint32_t tileNRemainder_;
    uint32_t tpWorldSize_;
    uint32_t rankId_{0};
    uint32_t aicNum_{0};
    uint32_t aivNum_{0};

    uint32_t coreVid_{0};
    uint32_t coreCid_{0};

    // matmul tiling data
    uint32_t m_;
    uint32_t n_;
    uint32_t k_;
    uint32_t singleCoreM_;
    uint32_t singleCoreN_;
    uint32_t singleTimeM_;
    uint32_t singleTimeN_;
    uint32_t singleCoreK_;
    uint32_t baseM_;
    uint32_t baseN_;
    uint32_t baseK_;
    uint32_t ubCalcM_;
    uint32_t ubCalcN_;
    uint32_t ubTmpBuffer_;
    uint32_t curAicM;
    uint32_t m_core_num;
    uint32_t n_core_num;

    uint64_t offsetA_{0};
    uint64_t offsetB_{0};
    uint64_t offsetC_{0};
    uint64_t mOffset_{0};
    uint64_t nOffset_{0};
    uint64_t nOffset_fix{0};

    uint16_t int32_gm_to_ub_data_{0};
    uint16_t int32_gm_to_ub_stride_{0};
    uint16_t bf16_ub_to_gm_data_{0};
    uint16_t bf16_ub_to_gm_stride_{0};

    bool aTrans_;
    bool bTrans_;
    bool isGammaBf16_;
    
    float armAvgFactor_;
    float epsilon_;
    __gm__ HcclOpResParam *winContext_{nullptr};
};

template<TemplateMC2TypeClass>
__aicore__ inline void QbmmReduceScatterAddRmsNormCastMte<TemplateMC2TypeFunc>::Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR gamma, GM_ADDR scale, 
    GM_ADDR bias, GM_ADDR perTokenScale, GM_ADDR y1Out, GM_ADDR y2Out, GM_ADDR xOut, GM_ADDR workspaceGM, TPipe *pipe, const QbmmReduceScatterAddRmsNormCastTilingData *tilingData)
{
    tpipe_ = pipe;
    coreVid_ = GetBlockIdx();
    coreCid_ = GetBlockIdx();

    // init global buffer
    yGM_.SetGlobalBuffer((__gm__ YType*) y);
    gammaGM_.SetGlobalBuffer((__gm__ GammaType*) gamma);
    xOutGM_.SetGlobalBuffer((__gm__ YType*) xOut);
    x1GM_.SetGlobalBuffer((__gm__ int8_t*) x1);
    x2GM_.SetGlobalBuffer((__gm__ int8_t*) x2);
    y1OutGM_.SetGlobalBuffer((__gm__ float*) y1Out);
    y2OutGM_.SetGlobalBuffer((__gm__ YType*) y2Out);
    workspaceAddr_ = workspaceGM;
    mmOutGm_.SetGlobalBuffer((__gm__ int32_t *)workspaceAddr_);
    scaleGMTensor_.SetGlobalBuffer((__gm__ ScaleType*)scale);

    mm_.Init(&(tilingData->qbmmReduceScatterAddRmsNormCastTilingInfo.matmulTiling), tpipe_);
    InitTilingData(tilingData);
    singleM_ = singleTimeM_;
    singleN_ = singleTimeN_;
    ubCalcM_ = UB_CALCM;
    ubCalcN_ = baseN_;

    ubTmpBuffer_ = EIGHT_BYTES_ALIGN * ubCalcM_ * ubCalcN_;
    
    tpWorldSize_ = TP_WORLD_SIZE;
    singleTpSize_ = (m_ / TP_WORLD_SIZE) * n_;
    armAvgFactor_ = ONE / n_;
    epsilon_ = EPSILON;

    aTrans_ = false;
    bTrans_ = false;
    perTokenScaleAddr_ = perTokenScale;

    m_core_num = CeilDiv(m_, baseM_);
    n_core_num = aicNum_ / m_core_num;
    
    // init ub local buffer for dequantCompute
    tpipe_->InitBuffer(vecQueSrc_, BUFFER_NUM, ubCalcM_ * ubCalcN_ * sizeof(int32_t));
    tpipe_->InitBuffer(vecQueTmp_, ubTmpBuffer_);
    tpipe_->InitBuffer(vecQueOut_, BUFFER_NUM, ubCalcM_ * ubCalcN_ * sizeof(YType));
    tpipe_->InitBuffer(vecQueScale_, BUFFER_NUM, baseN_ * sizeof(ScaleType));
    tpipe_->InitBuffer(vecQuePertokenScale_, BUFFER_NUM, DequantBmm::Align(ubCalcM_, EIGHT_BYTES_ALIGN) * sizeof(float));
    tpipe_->InitBuffer(broadcastFp32Tmp_, ubCalcM_ * ubCalcN_ * sizeof(float));
    tpipe_->InitBuffer(outFp32Tmp_, ubCalcM_ * ubCalcN_ * sizeof(float));
    tpipe_->InitBuffer(writeStateBuf_, STATE_ALIGN_BYTES);
    tpipe_->InitBuffer(readStateBuf_, STATE_ALIGN_BYTES);
    tpipe_->InitBuffer(resetStateBuf_, STATE_ALIGN_BYTES);
    
    winContext_ = (__gm__ HcclOpResParam*)AscendC::GetHcclContext<HCCL_GROUP_ID_0>();
    rankId_ = winContext_->localUsrRankId;

    int32_gm_to_ub_data_ = baseN_ * sizeof(uint32_t) / UB_ALIGN_BYTES;
    int32_gm_to_ub_stride_ = n_ * sizeof(uint32_t) / UB_ALIGN_BYTES - int32_gm_to_ub_data_;


    bf16_ub_to_gm_data_ = baseN_ * sizeof(bfloat16_t) / UB_ALIGN_BYTES;
    bf16_ub_to_gm_stride_ = n_ * sizeof(bfloat16_t) / UB_ALIGN_BYTES - bf16_ub_to_gm_data_;
}

template<TemplateMC2TypeClass>
__aicore__ inline void QbmmReduceScatterAddRmsNormCastMte<TemplateMC2TypeFunc>::InitTilingData(
    const QbmmReduceScatterAddRmsNormCastTilingData *tilingData)
{
    m_ = tilingData->qbmmReduceScatterAddRmsNormCastTilingInfo.matmulTiling.M;
    n_ = tilingData->qbmmReduceScatterAddRmsNormCastTilingInfo.matmulTiling.N;
    k_ = tilingData->qbmmReduceScatterAddRmsNormCastTilingInfo.matmulTiling.Ka;
    singleTimeM_ = tilingData->qbmmReduceScatterAddRmsNormCastTilingInfo.matmulTiling.singleCoreM;
    singleTimeN_ = tilingData->qbmmReduceScatterAddRmsNormCastTilingInfo.matmulTiling.singleCoreN;
    singleCoreK_ = tilingData->qbmmReduceScatterAddRmsNormCastTilingInfo.matmulTiling.singleCoreK;
    singleCoreM_ = singleTimeM_;
    singleCoreN_ = singleTimeN_;
    baseM_ = singleTimeM_;
    baseN_ = singleTimeN_;
    baseK_ = singleCoreK_;
    aicNum_ = tilingData->qbmmReduceScatterAddRmsNormCastTilingInfo.aicNum;
    aivNum_ = tilingData->qbmmReduceScatterAddRmsNormCastTilingInfo.aivNum;
    isGammaBf16_ = tilingData->qbmmReduceScatterAddRmsNormCastTilingInfo.isGammaBf16;
}

template<TemplateMC2TypeClass>
__aicore__ inline GM_ADDR QbmmReduceScatterAddRmsNormCastMte<TemplateMC2TypeFunc>::GetWindAddrByRankId(const int32_t rankId)
{
    if (rankId == rankId_) {
        return (GM_ADDR)(winContext_->localWindowsIn);
    }
    return (GM_ADDR)(((HcclRankRelationResV2*)(winContext_->remoteRes[rankId].nextDevicePtr))->windowsIn);   // 先找到某个rank的首地址，然后再偏移到具体的处理data的地方
}

template<TemplateMC2TypeClass>
__aicore__ inline GM_ADDR QbmmReduceScatterAddRmsNormCastMte<TemplateMC2TypeFunc>::GetWindStateAddrByRankId(const int32_t rankId)
{
    if (rankId == rankId_) {
        return (GM_ADDR)(winContext_->localWindowsExp);
    }
    return (GM_ADDR)(((HcclRankRelationResV2*)(winContext_->remoteRes[rankId].nextDevicePtr))->windowsExp);
}

template<TemplateMC2TypeClass>
__aicore__ inline void QbmmReduceScatterAddRmsNormCastMte<TemplateMC2TypeFunc>::SplitToCore(const uint32_t curSendCnt, const uint32_t curUseCoreNum, const uint32_t coreId, uint32_t &startId, uint32_t &endId, uint32_t &sendNum)
{
    sendNum = curSendCnt / curUseCoreNum;
    uint32_t remainderNum = curSendCnt % curUseCoreNum;
    startId = sendNum * coreId;
    if (coreId < remainderNum) {
        sendNum += 1;
        startId += coreId;
    } else {
        startId += remainderNum;
    }
    endId = startId + sendNum;
}

template<TemplateMC2TypeClass>
__aicore__ inline void QbmmReduceScatterAddRmsNormCastMte<TemplateMC2TypeFunc>::WriteStatusToWin()
{
    if (coreVid_ >= tpWorldSize_) {
        return;
    }
    uint32_t curOffset = rankId_ * FLOAT_UB_ALIGN_NUM;
    // 写入状态到对端，每个核写一个状态到一个rank
    LocalTensor<float> statusTensor = writeStateBuf_.Get<float>();
    statusTensor(0) = (float)1;
    int32_t curDstId = coreVid_;
    GM_ADDR remoteWinStateGM = GetWindStateAddrByRankId(curDstId);
    GlobalTensor<float> stateGMTensor;
    stateGMTensor.SetGlobalBuffer((__gm__ float*)remoteWinStateGM);
    SyncFunc<AscendC::HardEvent::S_MTE3>();
    DataCopy(stateGMTensor[curOffset], statusTensor, FLOAT_UB_ALIGN_NUM);
}

template<TemplateMC2TypeClass>
__aicore__ inline void QbmmReduceScatterAddRmsNormCastMte<TemplateMC2TypeFunc>::ReadStatus()
{
    // 粗粒度状态区逻辑，只有tpwordsize个状态，所需的v核只需要4个
    if (coreVid_ >= tpWorldSize_) {
        return;
    }
    GM_ADDR stateGM = GetWindStateAddrByRankId(rankId_);
    GlobalTensor<float> selfStateWinTensor;
    // 获取当前核所需读取状态的头地址
    selfStateWinTensor.SetGlobalBuffer((__gm__ float*)(stateGM));
    uint32_t offset = coreVid_ * FLOAT_UB_ALIGN_NUM;
    LocalTensor<float> statusTensor = readStateBuf_.Get<float>();
    LocalTensor<float> resetStatusTensor = resetStateBuf_.Get<float>();
    Duplicate<float>(resetStatusTensor, 0.0, FLOAT_UB_ALIGN_NUM);
    float flag = -1;
    uint32_t statusCnt = FLOAT_UB_ALIGN_NUM;
    float minTarget = (float)0.5;
    float maxTarget = (float)1.5;
    // 读取statusCnt个数据求和
    while ((flag < minTarget) || (flag > maxTarget)) {
        SyncFunc<AscendC::HardEvent::S_MTE2>();
        DataCopy(statusTensor, selfStateWinTensor[offset], statusCnt);
        SyncFunc<AscendC::HardEvent::MTE2_S>();
        flag = statusTensor(0);
    }   
    // reset state
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    SyncFunc<AscendC::HardEvent::MTE2_MTE3>();
    DataCopy(selfStateWinTensor[offset], resetStatusTensor, statusCnt);
}

template<TemplateMC2TypeClass>
__aicore__ inline void QbmmReduceScatterAddRmsNormCastMte<TemplateMC2TypeFunc>::DequantCompute(GlobalTensor<int32_t> &curMmOutGm, uint32_t curAicM, uint32_t curAicN, uint64_t row, uint64_t col)
{
    LocalTensor<float> dstLocalFp32;
    uint32_t curAivM = ubCalcM_;
    uint32_t curAivN = curAicN;

    uint32_t mUbLoops = CeilDiv(curAicM, ubCalcM_);
    DataCopyPadParams padParams;
    DequantParams dequantParams;
    DequantBmm::CalcDequantParams(mUbLoops == 1 ? curAicM : ubCalcM_, curAivN, dequantParams);
    dstLocalFp32 = outFp32Tmp_.Get<float>();
    GlobalTensor<YType> remoteTensor;
    for (uint32_t mUbLoopIdx = 0; mUbLoopIdx < mUbLoops; ++mUbLoopIdx) {
        if (mUbLoopIdx == mUbLoops - 1) {
            curAivM = curAicM - ubCalcM_ * (mUbLoops - 1);
            DequantBmm::CalcDequantParams(curAivM, curAivN, dequantParams, mUbLoops != 1 && curAivM != ubCalcM_);
        }

        uint32_t moffset_begin = mOffset_ + GetSubBlockIdx() * curAicM + mUbLoopIdx * ubCalcM_;
        uint32_t moffset_end = moffset_begin + curAivM - 1;
        uint32_t tp_id = moffset_begin / (m_ / TP_WORLD_SIZE);
        uint32_t tp_start = (m_ * tp_id) / TP_WORLD_SIZE;
        uint32_t tp_end = (((tp_id + 1) * m_) / TP_WORLD_SIZE) -1;
        uint32_t m_diff = moffset_begin - tp_start;

        LocalTensor<int32_t> srcLocal = vecQueSrc_.AllocTensor<int32_t>();
        LocalTensor<YType> dstLocal = vecQueOut_.AllocTensor<YType>();
        LocalTensor<uint8_t> tmpLocal = vecQueTmp_.Get<uint8_t>();

        DataCopyParams gm2UbParams{static_cast<uint16_t>(curAivM), int32_gm_to_ub_data_, int32_gm_to_ub_stride_, 0};
        DataCopyParams ub2GmParams{static_cast<uint16_t>(curAivM), bf16_ub_to_gm_data_, 0, bf16_ub_to_gm_stride_};
        
        uint32_t curAicAivOffset = offsetC_ + GetSubBlockIdx() * curAicM * n_  + mUbLoopIdx * ubCalcM_ * n_;
        DataCopy(srcLocal, mmOutGm_[curAicAivOffset], gm2UbParams);
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        LocalTensor<ScaleType> scaleLocal = vecQueScale_.AllocTensor<ScaleType>();
        DataCopy(scaleLocal, scaleGMTensor_[nOffset_], curAivN);
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        AscendDequant(dstLocalFp32, srcLocal, scaleLocal, tmpLocal, dequantParams);
        vecQueScale_.FreeTensor(scaleLocal);

        DataCopyParams scale2UbParams{1, 0, 0, 0};
        scale2UbParams.blockLen = curAivM * sizeof(float);
        uint64_t pertokenScaleOffset = mOffset_ + GetSubBlockIdx() * curAicM + mUbLoopIdx * ubCalcM_;
        uint32_t computedAivN = DequantBmm::Align(curAivN, 8U);  // 8: 32B aligned for float
        uint32_t ubResAlignedN = DequantBmm::Align(curAivN);     // 16: sizeof(yType) is 2, 32B / 2
        const uint32_t broadCastDst[2] = {curAivM, computedAivN};
        const uint32_t broadCastSrc[2] = {curAivM, 1};

        LocalTensor<float> broadcastFp32 = broadcastFp32Tmp_.Get<float>();
        LocalTensor<float> pertokenScaleLocal = vecQuePertokenScale_.AllocTensor<float>();
        GlobalTensor<float> pertokenScaleGm;
        pertokenScaleGm.SetGlobalBuffer((__gm__ float*)perTokenScaleAddr_);
        DataCopyPad(pertokenScaleLocal, pertokenScaleGm[pertokenScaleOffset], scale2UbParams, padParams);
        vecQuePertokenScale_.EnQue<float>(pertokenScaleLocal);
        pertokenScaleLocal = vecQuePertokenScale_.DeQue<float>();
        BroadCast<float, M_N_TWO_DIMS, 1>(broadcastFp32, pertokenScaleLocal, broadCastDst, broadCastSrc);
        AscendC::PipeBarrier<PIPE_V>();
        LocalTensor<float> tmpdstLocal = vecQueTmp_.Get<float>();
        if (computedAivN == ubResAlignedN) {
            Mul(tmpdstLocal, broadcastFp32, dstLocalFp32, computedAivN * curAivM);
        } else {
            for (auto i = 0; i < curAivM; i++) {
                Mul(tmpdstLocal[ubResAlignedN * i], broadcastFp32[computedAivN * i], dstLocalFp32[computedAivN * i],
                    computedAivN);
            }
        }
        vecQuePertokenScale_.FreeTensor(pertokenScaleLocal);
        PipeBarrier<PIPE_V>();
        Cast(dstLocal, tmpdstLocal, RoundMode::CAST_RINT, curAivM * ubResAlignedN);
        SyncFunc<AscendC::HardEvent::V_MTE3>();
        vecQueSrc_.FreeTensor(srcLocal);

        // 计算当前正处于的TP域
        if (moffset_end <= tp_end){
            int32_t remoteRankId = curAicAivOffset / (m_ * n_ / tpWorldSize_);
            GM_ADDR remoteWinAddr = GetWindAddrByRankId(remoteRankId);
            remoteTensor.SetGlobalBuffer((__gm__ YType*)remoteWinAddr);
            uint64_t tpOffset = (m_ * n_ / tpWorldSize_) * rankId_ + m_diff * n_;
            uint64_t nOffset = nOffset_;
            uint32_t dstOffset = tpOffset + nOffset;
            DataCopy(remoteTensor[dstOffset], dstLocal, ub2GmParams);
            
        } else {
            uint32_t front_count = tp_end - moffset_begin + 1;
            uint32_t end_count = moffset_end - tp_end;
            DataCopyParams ub2GmParams1{static_cast<uint16_t>(front_count), bf16_ub_to_gm_data_, 0, bf16_ub_to_gm_stride_};
            DataCopyParams ub2GmParams2{static_cast<uint16_t>(end_count), bf16_ub_to_gm_data_, 0, bf16_ub_to_gm_stride_};

            GM_ADDR remoteWinAddr = GetWindAddrByRankId(tp_id);
            remoteTensor.SetGlobalBuffer((__gm__ YType*)remoteWinAddr);

            uint64_t tpOffset = (m_ * n_ / tpWorldSize_) * rankId_ + m_diff * n_;
            uint64_t nOffset = nOffset_;
            uint32_t dstOffset = tpOffset + nOffset;
            DataCopy(remoteTensor[dstOffset], dstLocal, ub2GmParams1);
            GM_ADDR remoteWinAddr2 = GetWindAddrByRankId(tp_id + 1);
            remoteTensor.SetGlobalBuffer((__gm__ YType*)remoteWinAddr2);
            tpOffset = (m_ * n_ / tpWorldSize_) * rankId_;
            nOffset = nOffset_;
            dstOffset = tpOffset + nOffset;
            DataCopy(remoteTensor[dstOffset], dstLocal[front_count * curAivN], ub2GmParams2);
        }
        vecQueOut_.FreeTensor(dstLocal);

        
    }
}

template<TemplateMC2TypeClass>
__aicore__ inline void QbmmReduceScatterAddRmsNormCastMte<TemplateMC2TypeFunc>::AddRmsNormAddCompute(uint32_t tokenIndex, uint32_t numCol,
                                                                                LocalTensor<float>& x1TmpFloatLocal,
                                                                                LocalTensor<float>& x2TmpFloatLocal,
                                                                                LocalTensor<float>& addOutTmpFloatLocal,
                                                                                const DataCopyExtParams& copyExtParams,
                                                                                const DataCopyPadExtParams<YType>& copyPadExtParams)
{
    // 计算x + residual_x
    LocalTensor<YType> x2 = tokenBuf_.Get<YType>();
    DataCopyPad(x2, yGM_[tokenIndex * n_], copyExtParams, copyPadExtParams);
    SyncFunc<AscendC::HardEvent::MTE2_V>();
    Cast(x2TmpFloatLocal, x2, AscendC::RoundMode::CAST_NONE, numCol);
    PipeBarrier<PIPE_V>();
    AscendC::Add(addOutTmpFloatLocal, x1TmpFloatLocal, x2TmpFloatLocal, numCol);
}

template<TemplateMC2TypeClass>
__aicore__ inline void QbmmReduceScatterAddRmsNormCastMte<TemplateMC2TypeFunc>::AddRmsNormRmsNormCompute(uint32_t tokenIndex, uint32_t numCol,
                                                    LocalTensor<float>& xFp32, LocalTensor<float>& sqx,
                                                    LocalTensor<float>& gammaLocal,
                                                    const DataCopyExtParams& copyExtParams)
{
    // 计算rstd
    LocalTensor<float> reduceBufLocal = reduceFp32Buf_.Get<float>();
    LocalTensor<float> resFp32Tensor = resFp32Buf_.Get<float>();
    Mul(sqx, xFp32, xFp32, numCol);
    PipeBarrier<PIPE_V>();
    Muls(sqx, sqx, armAvgFactor_, numCol);
    PipeBarrier<PIPE_V>();
    ReduceSum(sqx, sqx, reduceBufLocal, numCol);
    PipeBarrier<PIPE_V>();
    Adds(sqx, sqx, epsilon_, 1);
    PipeBarrier<PIPE_V>();
    Sqrt(sqx, sqx, 1);
    Duplicate(reduceBufLocal, ONE, 1);
    PipeBarrier<PIPE_V>();
    Div(reduceBufLocal, reduceBufLocal, sqx, 1);
    // 计算y
    SyncFunc<AscendC::HardEvent::V_S>();
    float rstdValue = reduceBufLocal.GetValue(0);
    SyncFunc<AscendC::HardEvent::S_V>();
    Muls(xFp32, xFp32, rstdValue, numCol);
    PipeBarrier<PIPE_V>();
    LocalTensor<YType> yLocal = rowTmpFloatBuf_.Get<YType>();
    PipeBarrier<PIPE_V>();
    Mul(xFp32, xFp32, gammaLocal, numCol);
    PipeBarrier<PIPE_V>();
    Cast(yLocal, xFp32, RoundMode::CAST_RINT, numCol);
    // y结果搬出
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopy(y2OutGM_[tokenIndex * n_], yLocal, n_);
    PipeBarrier<PIPE_V>();
    Cast(resFp32Tensor, yLocal, RoundMode::CAST_NONE, numCol);
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopy(y1OutGM_[tokenIndex * n_], resFp32Tensor, n_);
}

template<TemplateMC2TypeClass>
__aicore__ inline void QbmmReduceScatterAddRmsNormCastMte<TemplateMC2TypeFunc>::ReadRemoteDataAdd()
{
    uint32_t startRowId = 0;
    uint32_t endRowId = 0;
    uint32_t rowNum = 0;
    uint32_t tokenIndex = 0;
    SplitToCore(m_ / TP_WORLD_SIZE, aivNum_, coreVid_, startRowId, endRowId, rowNum);

    LocalTensor<float> sumFp32Tensor = sumFp32Buf_.Get<float>();
    LocalTensor<float> tokenFp32Tensor = tokenFp32Buf_.Get<float>();
    mulBufLocal_ = mulBuf_.Get<float>();
    uint32_t curVidOffset = startRowId * n_;
    GlobalTensor<YType> localWinTensor;
    localWinTensor.SetGlobalBuffer((__gm__ YType*)GetWindAddrByRankId(rankId_));

    DataCopyExtParams expandXCopyParams{1U, static_cast<uint32_t>(n_ * sizeof(bfloat16_t)), 0U, 0U, 0U};
    DataCopyExtParams gammaCopyParams{1U, static_cast<uint32_t>(n_ * sizeof(GammaType)), 0U, 0U, 0U};
    const DataCopyPadExtParams<YType> copyPadXTypeParams{false, 0U, 0U, 0U};
    const DataCopyPadExtParams<GammaType> copyPadFloatParams{false, 0U, 0U, 0U};

    for(uint32_t tileIdx = 0; tileIdx < rowNum; ++tileIdx) {
        Duplicate<float>(sumFp32Tensor, (float)0.0, BLOCK_LENGTH);
        tokenIndex = startRowId + tileIdx;
        uint32_t curVidTileOffset = tokenIndex * n_;
        for(int tpIndex = 0; tpIndex < tpWorldSize_; ++tpIndex) {
            uint32_t curOffset = curVidTileOffset + tpIndex * singleTpSize_;
            tokenTensor_ = tokenQueue_.AllocTensor<YType>();
            DataCopy(tokenTensor_, localWinTensor[curOffset], n_);
            tokenQueue_.EnQue(tokenTensor_);
            tokenTensor_ = tokenQueue_.DeQue<YType>();
            SyncFunc<AscendC::HardEvent::MTE2_V>();
            Cast(tokenFp32Tensor, tokenTensor_, RoundMode::CAST_NONE, BLOCK_LENGTH);
            PipeBarrier<PIPE_V>();
            Add(sumFp32Tensor, sumFp32Tensor, tokenFp32Tensor, BLOCK_LENGTH);
            tokenQueue_.FreeTensor<YType>(tokenTensor_);
        }
        rowTmpFloatLocal_ = rowTmpFloatBuf_.Get<float>();
        AddRmsNormAddCompute(tokenIndex, n_, sumFp32Tensor, rowTmpFloatLocal_, sumFp32Tensor, expandXCopyParams, copyPadXTypeParams);
        // 执行AddRmsNorm--Add，输出x
        LocalTensor<YType> sumBufLocal = tokenBuf_.Get<YType>();
        PipeBarrier<PIPE_V>();
        Cast(sumBufLocal, sumFp32Tensor, AscendC::RoundMode::CAST_RINT, n_);
        SyncFunc<AscendC::HardEvent::V_MTE3>();
        DataCopyPad(xOutGM_[tokenIndex * n_], sumBufLocal, expandXCopyParams);

        // 执行AddRmsNorm--RmsNormCast，输出x最终结果的fp16搬出到y2，float32搬出到y1
        LocalTensor<GammaType> gammaLocal = gammaBuf_.Get<GammaType>();
        DataCopyPad(gammaLocal, gammaGM_, gammaCopyParams, copyPadFloatParams);
        SyncFunc<AscendC::HardEvent::MTE2_V>();

        LocalTensor<float> gammaFp32Local;
        if (isGammaBf16_) {
            gammaFp32Local = gammaFp32Buf_.Get<float>();
            Cast(gammaFp32Local, gammaLocal, RoundMode::CAST_NONE, n_);
        } else {
            gammaFp32Local = gammaLocal.template ReinterpretCast<float>();
        }
        AddRmsNormRmsNormCompute(tokenIndex, n_, sumFp32Tensor, mulBufLocal_, gammaFp32Local, expandXCopyParams);
    }
}

template<TemplateMC2TypeClass>
__aicore__ inline void QbmmReduceScatterAddRmsNormCastMte<TemplateMC2TypeFunc>::MMCompute(uint32_t singleM, uint32_t singleN)
{
    mm_.SetSingleShape(singleM, singleN, singleCoreK_);
    mm_.SetTensorA(x1GM_[offsetA_], aTrans_);
    mm_.SetTensorB(x2GM_[offsetB_], bTrans_);
    mm_.template Iterate<false>();
}

template<TemplateMC2TypeClass>
__aicore__ inline void QbmmReduceScatterAddRmsNormCastMte<TemplateMC2TypeFunc>::CalcMAxisOffset(uint32_t mCoreIndex, uint32_t nLoops)
{
    mOffset_ = mCoreIndex * baseM_;
    offsetA_ = mCoreIndex * baseM_ * k_;
    offsetC_ = nOffset_fix + mCoreIndex * baseM_ * n_;
}

template<TemplateMC2TypeClass>
__aicore__ inline void QbmmReduceScatterAddRmsNormCastMte<TemplateMC2TypeFunc>::CalcNAxisOffset(uint32_t loopIdx)
{
    nOffset_ = nOffset_fix + loopIdx * baseN_;
    if constexpr (BMatmulType::format == CubeFormat::ND) {
        offsetB_ = nOffset_;
    } else if constexpr (BMatmulType::format == CubeFormat::NZ) {
        uint64_t temp1 = nOffset_ / NZ_W0;
        uint64_t temp2 = nOffset_ % NZ_W0;
        offsetB_ = temp1 * k_ * NZ_W0 + temp2;
    }
    if (loopIdx == 0) {
        return;
    }
    offsetC_ += baseN_;
}

template<TemplateMC2TypeClass>
__aicore__ inline void QbmmReduceScatterAddRmsNormCastMte<TemplateMC2TypeFunc>::MatmulProcess()
{
    mm_.SetOrgShape(m_, n_, k_);
    uint32_t mDim = CeilDiv(m_, singleCoreM_ * 2);
    uint32_t nDim = CeilDiv(n_, singleCoreN_);
    // cid(0-15)分2个, (16-23)分1个
    uint32_t mCoreIndex = coreCid_ % m_core_num;
    uint32_t nCoreIndex = coreCid_ / m_core_num;
    uint32_t startBlockIdx = 0;
    uint32_t endBlockIdx = 0;
    uint32_t tileNum = 0;
    SplitToCore(n_ / baseN_, n_core_num, nCoreIndex, startBlockIdx, endBlockIdx, tileNum);
    uint32_t remain_num = m_ % baseM_;
    singleM_ = baseM_;
    if (remain_num != 0 && mCoreIndex == m_core_num - 1){
        singleM_ = remain_num;
    }
    curAicM = singleM_ / 2;
    uint32_t mLoops = 1;
    uint32_t nLoops = tileNum;
    // CalcOffset, 默认当前都是ND, 且非转置
    nOffset_fix = static_cast<uint64_t>(startBlockIdx * baseN_);

    for (uint32_t i = 0; i < mLoops; ++i) {
        // uint32_t singleM = singleTimeM_;
        CalcMAxisOffset(mCoreIndex, nLoops);
        for (uint32_t j = 0; j < nLoops; ++j) {
            singleN_ = baseN_;
            CalcNAxisOffset(j);
            MMCompute(singleM_, singleN_);
            mm_.template GetTensorC<false>(mmOutGm_[offsetC_]);
            CrossCoreSetFlag<0x2, PIPE_FIX>(SYNC_AIC_TO_AIV);
        }
    }
    mm_.End();
}

template<TemplateMC2TypeClass>
__aicore__ inline void QbmmReduceScatterAddRmsNormCastMte<TemplateMC2TypeFunc>::Process()
{
    if ASCEND_IS_AIC {
        MatmulProcess();
    }
    if ASCEND_IS_AIV {
        //TODO:整改，代码重复
        uint32_t mDim = CeilDiv(m_, singleCoreM_ * 2);
        uint32_t nDim = CeilDiv(n_, singleCoreN_);
        // cid(0-15)分2个, (16-23)分1个
        uint32_t group_cid = coreVid_ / 2;
        uint32_t mCoreIndex = group_cid % m_core_num;
        uint32_t nCoreIndex = group_cid / m_core_num;
        
        uint32_t startBlockIdx = 0;
        uint32_t endBlockIdx = 0;
        uint32_t tileNum = 0;

        SplitToCore(n_ / baseN_, n_core_num, nCoreIndex, startBlockIdx, endBlockIdx, tileNum);
        uint32_t remain_num = m_ % baseM_;
        singleM_ = baseM_;
        if (remain_num != 0 && mCoreIndex == m_core_num - 1){
            singleM_ = remain_num;
        }
        curAicM = singleM_ / 2;
        uint32_t mLoops = 1;
        uint32_t nLoops = tileNum;
        // CalOffset, 默认当前都是ND, 且非转置
        nOffset_fix = static_cast<uint64_t>(startBlockIdx * baseN_);

        for (uint32_t i = 0; i < mLoops; ++i) {
            // uint32_t singleM = singleTimeM_;
            CalcMAxisOffset(mCoreIndex, nLoops);
            for (uint32_t j = 0; j < nLoops; ++j) {
                // uint32_t singleN = singleTimeN_;
                CalcNAxisOffset(j);
                uint32_t singleN = baseN_;
                CrossCoreWaitFlag(SYNC_AIC_TO_AIV);
                DequantCompute(mmOutGm_, curAicM, baseN_, mOffset_, nOffset_);
            }
        }
        SyncAll<true>();
        // 当前die的数据已经发完
        WriteStatusToWin();
        ReadStatus();
        SyncAll<true>();    //确保前4个核都等到了状态，即数据区完全ready
        tpipe_->Reset();
        tpipe_->InitBuffer(tokenQueue_, BUFFER_NUM, BLOCK_LENGTH * sizeof(float));   //涉及到Cast的src和dst记得，小cast大要分配大的空间
        tpipe_->InitBuffer(sumFp32Buf_, BLOCK_LENGTH * sizeof(float));
        tpipe_->InitBuffer(tokenFp32Buf_, BLOCK_LENGTH * sizeof(float));
        tpipe_->InitBuffer(tokenBuf_, BLOCK_LENGTH * sizeof(bfloat16_t));
        tpipe_->InitBuffer(rowTmpFloatBuf_, BLOCK_LENGTH * sizeof(float));
        tpipe_->InitBuffer(mulBuf_, BLOCK_LENGTH * sizeof(float));
        tpipe_->InitBuffer(reduceFp32Buf_, REDUCE_BUF_BYTES);
        tpipe_->InitBuffer(resFp32Buf_, BLOCK_LENGTH * sizeof(float));
        tpipe_->InitBuffer(gammaBuf_, BLOCK_LENGTH * sizeof(float));
        tpipe_->InitBuffer(gammaFp32Buf_, BLOCK_LENGTH * sizeof(float));
        ReadRemoteDataAdd();
    }
}
}; // QbmmReduceScatterAddRmsNormCastImpl
#endif  // QBMM_REDUCE_SCATTER_ADD_RMS_NORM_CAST_MTE_H

