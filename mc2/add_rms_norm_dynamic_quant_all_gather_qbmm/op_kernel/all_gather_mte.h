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
 * \file all_gather_mte.h
 * \brief all_gather mte通信kernel代码逻辑
 */

#ifndef ALL_GATHER_MTE_H
#define ALL_GATHER_MTE_H

#include "basic_api/kernel_basic_intf.h"
#include "adv_api/hccl/hccl.h"
#include "adv_api/reduce/sum.h"
#include "adv_api/pad/broadcast.h"
#include "kernel_tiling/kernel_tiling.h"
#include "add_rms_norm_dynamic_quant_all_gather_qbmm_tiling_data.h"
#include "all_gather_mte_comm.h"
#include "all_gather_mte_utils.h"

namespace AllGatherImpl {

using namespace QuantMTECommImpl;
using namespace AscendC;

// 之后可修改成从tiling侧获取数据切块大小
constexpr static uint32_t X_PER_BLOCK_NUM = 512U;  // 当前一次搬运一个x数据块，x dtype为 8bit 时对应 512个x数据
constexpr static uint64_t CV_SYNC_START_OFFSET = 100UL * 1024UL; // CV同步状态相对于通信状态向后偏移100K
constexpr static uint64_t CV_STATE_ALIGN = 64UL;    // CV同步的标志位间64B对齐
constexpr static uint64_t ALLOC_UB_SPACE = 180UL * 1024UL;  // 总共192K UB中抽出180K于此处使用

template<AllGatherTemplateTypeClass>
class AllGatherMte {
public:
    __aicore__ inline AllGatherMte() {};
    __aicore__ inline void Init(TPipe *tPipe, const AddRmsNormDynamicQuantAllGatherQbmmInfo *tilingData);

    __aicore__ inline void SetRemoteFlag();
    __aicore__ inline void WaitRemoteFlag();
    __aicore__ inline void ExecuteAllGather(GM_ADDR allGatherDataAddr, GM_ADDR allGatherScalesAddr);
    __aicore__ inline GM_ADDR CalcCvFlagAddr(uint64_t mBlockIdx, uint64_t kBlockIdx);

private:
    __aicore__ inline void ReadDataBlock(uint64_t curXOffset, uint32_t mCnt);
    __aicore__ inline void ReadScales();
    __aicore__ inline void SetCvAtomicFlag(uint32_t mBlockIdx, uint32_t kBlockIdx, uint32_t flagValue);

    uint64_t xSize_{0}; // 单卡上数据大小
    uint64_t xNums_{0}; // 单卡上数据个数
    uint64_t scaleSize_{0}; // 单卡上scale大小
    uint32_t totalBlockNums_{0};
    uint64_t M_{0};
    uint64_t K_{0};
    uint32_t sendCoreNumPerRank_{0};
    uint32_t remoteRankId_{0};
    uint64_t tileK_{0};
    uint64_t xInQueueSize_{0};
    int32_t mStartIndex_{0};
    int32_t kStartIndex_{0};
    int32_t mEndIndex_{0};
    uint32_t mStartFlagCount_{0};
    uint32_t mEndFlagCount_{0};
    uint32_t singleCoreM_{0};
    uint32_t mLoop_{0};
    uint32_t mBlockIdx_{0};
    uint32_t kBlockIdx_{0};
    uint32_t mMteCoreM_{0};
    uint32_t kMteCoreK_{0};
    uint32_t coreInnerMIndex_{0};

    DataCopyExtParams dataCopyParamsIn_;
    DataCopyExtParams dataCopyParamsOut_;
    DataCopyExtParams scalesCopyParams_;
    DataCopyPadExtParams<int8_t> dataCopyPadParams_;
    DataCopyPadExtParams<ScalesType> scalesCopyPadParams_;
    

    MTECommunication<AllGatherTemplateType> mteComm_; // MTE 通信相关实现

    LocalTensor<int32_t> atomicAddTensor_;
    GlobalTensor<int8_t> remoteWinXTensor_;
    GlobalTensor<ScalesType> remoteWinScaleTensor_;
    GlobalTensor<int8_t> localWinXTensor_;
    GlobalTensor<ScalesType> localWinScaleTensor_;
    GlobalTensor<int8_t> allGatherXOutTensor_;
    GlobalTensor<ScalesType> allGatherScaleOutTensor_;
    GlobalTensor<int32_t> remoteCvFlagTensor_;

    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> xInQueue_, scaleInQue; // 用于读数据和反量化求和的通算并行
    TBuf<> atomicAddBuf_;
};

template <AllGatherTemplateTypeClass>
__aicore__ inline void AllGatherMte<AllGatherTemplateType>::Init(
    TPipe *tPipe, const AddRmsNormDynamicQuantAllGatherQbmmInfo *tilingData)
{
    // 初始化HcclContext
    mteComm_.InitHcclContext();

    /* all_gather 自己的数据 */
    M_ = tilingData->addRmsNormDynamicQuantAllGatherTilingData.M;
    K_ = tilingData->addRmsNormDynamicQuantAllGatherTilingData.Ka;
    xNums_ = tilingData->addRmsNormDynamicQuantAllGatherTilingData.xNums; // 总的x数据个数， M * h
    xSize_ = tilingData->addRmsNormDynamicQuantAllGatherTilingData.xSize; // 总的x数据量，B
    scaleSize_ = tilingData->addRmsNormDynamicQuantAllGatherTilingData.scaleSize;
    totalBlockNums_ = CeilDiv(xSize_, X_BLOCK_BYTES); // 按每次搬运x的数据量分块，得到的总块数
    sendCoreNumPerRank_ = tilingData->addRmsNormDynamicQuantAllGatherTilingData.sendCoreNumPerRank;
    tileK_ = tilingData->addRmsNormDynamicQuantAllGatherTilingData.mteTileK;
    mteComm_.round_ = totalBlockNums_ / sendCoreNumPerRank_; // 计算总的数据分核搬运需要的轮次数
    mteComm_.tailBlockNums_ = totalBlockNums_ % sendCoreNumPerRank_; // 搬运的尾块数

    dataCopyPadParams_ = {false, 0, 0, 0};
    scalesCopyParams_ = {1, scaleSize_, 0, 0, 0};
    scalesCopyPadParams_ = {false, 0, 0, 0};

    uint32_t rankSize = tilingData->addRmsNormDynamicQuantAllGatherTilingData.rankSize;
    uint32_t singleCoreM = tilingData->matmulTiling.singleCoreM;
    // 其余tQue或tBuf所需要的空间
    uint64_t usedSpace = scaleSize_ + CV_STATE_ALIGN * 2 + (rankSize * 2 + 1) * UB_ALIGN_BYTES;
    // 剩余的xInQueue可用的空间大小
    uint64_t availableSpaceForXInQueue = (ALLOC_UB_SPACE - usedSpace) / BUFFER_NUM / X_BLOCK_BYTES * X_BLOCK_BYTES;
    // xInQueue需要的最大空间大小
    uint64_t demandSpaceForXInQueue = CeilAlign(xSize_, X_BLOCK_BYTES);
    xInQueueSize_ = availableSpaceForXInQueue < demandSpaceForXInQueue \
                    ? availableSpaceForXInQueue \
                    : demandSpaceForXInQueue;
    tPipe->InitBuffer(xInQueue_, BUFFER_NUM, xInQueueSize_);
    tPipe->InitBuffer(scaleInQue, 1, scaleSize_); // 每次拷贝 63 * 4B scale；
    tPipe->InitBuffer(atomicAddBuf_, CV_STATE_ALIGN); // 用于累加标志位

    atomicAddTensor_ = atomicAddBuf_.Get<int32_t>();
    
    // 公共MTE搬运参数计算
    uint32_t aivNum = tilingData->addRmsNormDynamicQuantAllGatherTilingData.aivNum;
    mteComm_.InitParams(xSize_, aivNum, sendCoreNumPerRank_);

    // 初始化tPipe的各种buffer
    mteComm_.InitBuffer(tPipe);

    uint32_t modCoreIndex = mteComm_.aivId_ % sendCoreNumPerRank_;
    
    // 按k方向奇偶切分成两部分
    uint32_t kDim = tilingData->addRmsNormDynamicQuantAllGatherTilingData.mteKSplitNum;
    uint32_t mDim = tilingData->addRmsNormDynamicQuantAllGatherTilingData.mteMSplitNum;
    remoteRankId_ = mteComm_.aivId_ / sendCoreNumPerRank_;
    // task 先按 mDim 为3份调整，innerLoop 再做循环调整
    mBlockIdx_ = modCoreIndex % mDim;
    kBlockIdx_ = modCoreIndex % kDim;
    mMteCoreM_ = tilingData->addRmsNormDynamicQuantAllGatherTilingData.mteMSplitSize;
    kMteCoreK_ = tilingData->addRmsNormDynamicQuantAllGatherTilingData.mteKSplitSize;
    singleCoreM_ = singleCoreM;
    uint32_t mTailNum = M_ % mDim;
    if (mBlockIdx_ < mTailNum) {
        mMteCoreM_ += 1;
        coreInnerMIndex_ = mBlockIdx_ * mMteCoreM_;
    } else {
        coreInnerMIndex_ = mBlockIdx_ * mMteCoreM_ + mTailNum;
    }
    mStartIndex_ = coreInnerMIndex_ + remoteRankId_ * M_;
    kStartIndex_ = kBlockIdx_ * X_PER_BLOCK_NUM;
    mEndIndex_ = mStartIndex_ + mMteCoreM_;
}

/* 写入状态到状态区 */
template <AllGatherTemplateTypeClass>
__aicore__ inline void AllGatherMte<AllGatherTemplateType>::SetRemoteFlag()
{
    mteComm_.WriteStatusToWin();
}

/* 读状态位，软同步 */
template <AllGatherTemplateTypeClass>
__aicore__ inline void AllGatherMte<AllGatherTemplateType>::WaitRemoteFlag()
{
    mteComm_.ReadStatus(); 
}

/* 读取 x 从 远端Win -> UB -> 本端Win */
template <AllGatherTemplateTypeClass>
__aicore__ inline void AllGatherMte<AllGatherTemplateType>::ReadDataBlock(uint64_t curXOffset, uint32_t mCnt)
{
    if constexpr (!isOptionalOutput) {
        if (remoteRankId_ == mteComm_.hcclContext_->localUsrRankId) {
            return;
        }
    }
    LocalTensor<int8_t> xTmpTensor = xInQueue_.AllocTensor<int8_t>();
    dataCopyParamsIn_ = {static_cast<uint16_t>(mCnt), X_PER_BLOCK_NUM, K_ - X_PER_BLOCK_NUM, 0, 0};
    dataCopyParamsOut_ = {static_cast<uint16_t>(mCnt), X_PER_BLOCK_NUM, 0, K_ - X_PER_BLOCK_NUM, 0};
    DataCopyPad(xTmpTensor, remoteWinXTensor_[curXOffset], dataCopyParamsIn_, dataCopyPadParams_);
    xInQueue_.EnQue(xTmpTensor);
    xTmpTensor = xInQueue_.DeQue<int8_t>();
    DataCopyPad(localWinXTensor_[curXOffset], xTmpTensor, dataCopyParamsOut_);
    // 调试输出
    if constexpr (isOptionalOutput) {
        DataCopyPad(allGatherXOutTensor_[curXOffset], xTmpTensor, dataCopyParamsOut_);
    }
    xInQueue_.FreeTensor(xTmpTensor);
}

/* 读取 scale 从 远端Win -> UB -> 本端Win */
template <AllGatherTemplateTypeClass>
__aicore__ inline void AllGatherMte<AllGatherTemplateType>::ReadScales()
{
    if constexpr (!isOptionalOutput) {
        if (remoteRankId_ == mteComm_.hcclContext_->localUsrRankId) {
            return;
        }
    }
    LocalTensor<ScalesType> scaleTmpTensor = scaleInQue.AllocTensor<ScalesType>();
    DataCopyPad(scaleTmpTensor, remoteWinScaleTensor_, scalesCopyParams_, scalesCopyPadParams_);
    scaleInQue.EnQue(scaleTmpTensor);
    scaleTmpTensor = scaleInQue.DeQue<ScalesType>();
    DataCopyPad(localWinScaleTensor_, scaleTmpTensor, scalesCopyParams_);
    // 调试输出
    if constexpr (isOptionalOutput) {
        DataCopyPad(allGatherScaleOutTensor_, scaleTmpTensor, scalesCopyParams_);
    }
    scaleInQue.FreeTensor(scaleTmpTensor);
}

/* 获取本卡上对应CV状态区的地址 */
template <AllGatherTemplateTypeClass>
__aicore__ inline GM_ADDR AllGatherMte<AllGatherTemplateType>::CalcCvFlagAddr(
    uint64_t mBlockIdx, uint64_t kBlockIdx)
{
    GM_ADDR cvFlagBaseAddr = \
        mteComm_.GetWinStatusAddrGm(mteComm_.hcclContext_->localUsrRankId) + CV_SYNC_START_OFFSET;
    GM_ADDR cvFlagAddr = cvFlagBaseAddr + (mBlockIdx * tileK_ + kBlockIdx) * CV_STATE_ALIGN;
    return cvFlagAddr;
}

template <AllGatherTemplateTypeClass>
__aicore__ inline void AllGatherMte<AllGatherTemplateType>::SetCvAtomicFlag(uint32_t mBlockIdx, uint32_t kBlockIdx, uint32_t flagValue)
{
    GM_ADDR cvFlagAddr = CalcCvFlagAddr(mBlockIdx, kBlockIdx);
    PipeBarrier<PIPE_ALL>();
    // 计算当前CV同步状态的地址
    atomicAddTensor_.SetValue(0, flagValue);
    SyncFunc<AscendC::HardEvent::S_MTE3>();
    remoteCvFlagTensor_.SetGlobalBuffer((__gm__ int32_t*)cvFlagAddr);
    SetAtomicAdd<int32_t>();
    DataCopy(remoteCvFlagTensor_, atomicAddTensor_, CV_STATE_ALIGN / sizeof(int32_t));
    SetAtomicNone();
    PipeBarrier<PIPE_ALL>();
}

template <AllGatherTemplateTypeClass>
__aicore__ inline void AllGatherMte<AllGatherTemplateType>::ExecuteAllGather(GM_ADDR allGatherDataAddr, GM_ADDR allGatherScalesAddr)
{
    // +--------+--------+--------+--------+--------+--------+--------+
    // | Rank0  | Rank1  |  ...   |  512B  | Rank0  | Rank1  |  ...   |
    // |  data  |  data  |  ...   |  align | scales | scales |  ...   |
    // +--------+--------+--------+--------+--------+--------+--------+
    // data整体有512B对齐
    
    // 获取对端Win区中数据区相关的地址
    GM_ADDR remoteDataGm = mteComm_.GetWinDataAddrGm(remoteRankId_) + remoteRankId_ * xSize_;
    remoteWinXTensor_.SetGlobalBuffer((__gm__ int8_t*)remoteDataGm);
    // 本端对应rank win区数据地址
    uint32_t localRankId = mteComm_.hcclContext_->localUsrRankId;
    // TODO: 正确位置如下，调试完毕后需要修改回来
    GM_ADDR localWinGm = mteComm_.GetWinDataAddrGm(localRankId);
    GM_ADDR localDataGm = localWinGm + remoteRankId_ * xSize_;
    localWinXTensor_.SetGlobalBuffer((__gm__ int8_t*)localDataGm);
    GM_ADDR allGatherOutDataGm = allGatherDataAddr + remoteRankId_ * xSize_;
    allGatherXOutTensor_.SetGlobalBuffer((__gm__ int8_t*)allGatherOutDataGm);

    // scales 一次搬运完毕
    if (mteComm_.aivId_ % sendCoreNumPerRank_ == 0) {
        GM_ADDR remoteScaleGm = mteComm_.GetWinDataAddrGm(remoteRankId_) + mteComm_.winDataSize_ + remoteRankId_ * scaleSize_;
        remoteWinScaleTensor_.SetGlobalBuffer((__gm__ ScalesType*)remoteScaleGm);
        // TODO: 正确位置如下，调试完毕后需要修改回来
        GM_ADDR localScaleGm = localWinGm + mteComm_.winDataSize_ + remoteRankId_ * scaleSize_;
        localWinScaleTensor_.SetGlobalBuffer((__gm__ ScalesType*)localScaleGm);
        GM_ADDR allGatherOutScaleGm = allGatherScalesAddr + remoteRankId_ * scaleSize_;
        allGatherScaleOutTensor_.SetGlobalBuffer((__gm__ ScalesType*)allGatherOutScaleGm);
        ReadScales();
    }

    uint32_t mCurOffset = M_ / singleCoreM_;
    uint32_t mTile = 1;
    uint32_t mFlagCount = mMteCoreM_;
    // 先一次拷完，若跨base块则切分拷贝，当前tp M泛化到128，暂不存在跨baseM的情况
    if (((mEndIndex_ - 1) / singleCoreM_ - mStartIndex_ / singleCoreM_) > 0) {
        mTile = 2;
        mFlagCount = Ceil(mStartIndex_, singleCoreM_) * singleCoreM_ - mStartIndex_;
    }
    uint32_t kLoop = kMteCoreK_ / X_PER_BLOCK_NUM;
    uint32_t curXOffset = coreInnerMIndex_ * K_ + kStartIndex_;
    uint32_t baseMIndex = mStartIndex_ / singleCoreM_;
    for (uint64_t curKBlock = 0; curKBlock < kLoop; ++curKBlock) {
        uint64_t innerCurXOffset = curXOffset + curKBlock * X_PER_BLOCK_NUM * 2;
        // 读取对端对应地址的 x 数据
        ReadDataBlock(innerCurXOffset, mFlagCount);
        SetCvAtomicFlag(baseMIndex, kStartIndex_ / X_PER_BLOCK_NUM + curKBlock * 2, mFlagCount);
    }
    // 第二轮
    if (mTile > 1) {
        curXOffset += mFlagCount * K_;
        mFlagCount = mEndIndex_ - Ceil(mStartIndex_, singleCoreM_) * singleCoreM_;
        baseMIndex++;
        for (uint64_t curKBlock = 0; curKBlock < kLoop; ++curKBlock) {
            uint64_t innerCurXOffset = curXOffset + curKBlock * X_PER_BLOCK_NUM * 2;
            // 读取对端对应地址的 x 数据
            ReadDataBlock(innerCurXOffset, mFlagCount);
            SetCvAtomicFlag(baseMIndex, kStartIndex_ / X_PER_BLOCK_NUM + curKBlock * 2, mFlagCount);
        }
    }
    PipeBarrier<PIPE_MTE3>();
}
} // AllGatherImpl
#endif  // ALL_GATHER_MTE_H