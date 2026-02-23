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
constexpr static uint32_t X_PRE_BLOCK_NUM = 512U;  // 当前一次搬运一个x数据块，x dtype为 8bit 时对应 1024个x数据. 对于fp4需要另外算

template<AllGatherTemplateTypeClass>
class AllGatherMte {
public:
    __aicore__ inline AllGatherMte() {};
    __aicore__ inline void Init(TPipe *tPipe, uint32_t M, uint32_t Ka, uint32_t aivNum);

    __aicore__ inline void SetRemoteFlag();
    __aicore__ inline void WaitRemoteFlag();
    __aicore__ inline void ExecuteAllGather(GM_ADDR outputTensor, GM_ADDR zTensor);

private:
    __aicore__ inline void ReadDataBlock(uint64_t curXOffset);
    __aicore__ inline void ReadScales();

    uint64_t xSize_{0}; // 单卡上数据大小
    uint64_t xNums_{0}; // 单卡上数据个数
    uint64_t scaleSize_{0}; // 单卡上scale大小
    uint64_t tailXNums_{0};
    uint32_t totalBlockNums_{0};
    uint64_t mLoopIdx_{0};
    uint64_t kLoopIdx_{0};
    uint64_t M_{0};
    uint64_t K_{0};
    uint32_t sendCoreNumPerRank_{0};

    DataCopyExtParams scalesCopyParams_;
    DataCopyPadExtParams<ScalesType> scalesCopyPadParams_;

    MTECommunication<AllGatherTemplateType> mteComm_; // MTE 通信相关实现

    GlobalTensor<int8_t> remoteWinXTensor_;
    GlobalTensor<ScalesType> remoteWinScaleTensor_;
    GlobalTensor<int8_t> localWinXTensor_;
    GlobalTensor<ScalesType> localWinScaleTensor_;

    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> xInQueue_, scaleInQue; // 用于读数据和反量化求和的通算并行
    TBuf<> sumBuf_; // 用于Reduce_sum 求和
};

template <AllGatherTemplateTypeClass>
__aicore__ inline void AllGatherMte<AllGatherTemplateType>::Init(TPipe *tPipe, uint32_t M, uint32_t Ka, uint32_t aivNum)
{
    // 初始化HcclContext
    mteComm_.InitHcclContext();

    /* all_gather 自己的数据 */
    M_ = M;
    K_ = Ka;
    xNums_ = M * Ka; // 总的x数据个数， M * h
    xSize_ = xNums_ * sizeof(XType); // 总的x数据量，B
    scaleSize_ = M * sizeof(ScalesType);
    tailXNums_ = BlockAlignMod(xNums_, X_PRE_BLOCK_NUM); // 计算最后一个数据块的大小
    totalBlockNums_ = CeilDiv(xSize_, X_BLOCK_BYTES); // 按每次搬运x的数据量分块，得到的总块数
    sendCoreNumPerRank_ = aivNum / mteComm_.hcclContext_->rankSize;
    mteComm_.round_ = totalBlockNums_ / sendCoreNumPerRank_; // 计算总的数据分核搬运需要的轮次数
    mteComm_.tailBlockNums_ = totalBlockNums_ % sendCoreNumPerRank_; // 搬运的尾块数
    scalesCopyParams_ = {1, scaleSize_, 0, 0, 0};
    scalesCopyPadParams_ = {false, 0, 0, 0};

    tPipe->Reset();
    tPipe->InitBuffer(xInQueue_, BUFFER_NUM, X_BLOCK_BYTES); // 每次拷贝 1024B x; 128 * 8
    tPipe->InitBuffer(scaleInQue, BUFFER_NUM, UB_ALIGN_BYTES); // 每次拷贝 32B scale；4 * 8
    tPipe->InitBuffer(sumBuf_, X_PRE_BLOCK_NUM * sizeof(float)); // 用于Reduce_sum 求和，1024 * 4 = 4k

    // 设置切块大小
    mteComm_.SetBlockSize(X_PRE_BLOCK_NUM, aivNum, tailXNums_);

    // 公共MTE搬运参数计算
    mteComm_.InitParams(xSize_);

    // 初始化tPipe的各种buffer
    mteComm_.InitBuffer(tPipe);

    uint32_t modCoreIndex = mteComm_.aivId_ % sendCoreNumPerRank_;
    uint32_t curBlockIndex = modCoreIndex * mteComm_.round_ + \
                             (modCoreIndex < mteComm_.tailBlockNums_ ? modCoreIndex : mteComm_.tailBlockNums_);
    mLoopIdx_ = curBlockIndex % M;
    kLoopIdx_ = curBlockIndex / M;
}

template <AllGatherTemplateTypeClass>
__aicore__ inline void AllGatherMte<AllGatherTemplateType>::SetRemoteFlag()
{
    // 写入状态到状态区
    mteComm_.WriteStatusToWin();
}

template <AllGatherTemplateTypeClass>
__aicore__ inline void AllGatherMte<AllGatherTemplateType>::WaitRemoteFlag()
{
    // 读状态位，软同步
    mteComm_.ReadStatus(); 
}

template <AllGatherTemplateTypeClass>
__aicore__ inline void AllGatherMte<AllGatherTemplateType>::ReadDataBlock(uint64_t curXOffset)
{
    /* 读取 x 从 远端Win -> UB -> 本端Win */
    LocalTensor<int8_t> xTmpTensor = xInQueue_.AllocTensor<int8_t>();
    DataCopy(xTmpTensor, remoteWinXTensor_[curXOffset], X_PRE_BLOCK_NUM);
    xInQueue_.EnQue(xTmpTensor);
    xTmpTensor = xInQueue_.DeQue<int8_t>();
    DataCopy(localWinXTensor_[curXOffset], xTmpTensor, X_PRE_BLOCK_NUM);
    xInQueue_.FreeTensor(xTmpTensor);
}

template <AllGatherTemplateTypeClass>
__aicore__ inline void AllGatherMte<AllGatherTemplateType>::ReadScales()
{
    /* 读取 scale 从 远端Win -> UB -> 本端Win */
    LocalTensor<ScalesType> scaleTmpTensor = scaleInQue.AllocTensor<ScalesType>();
    DataCopyPad(scaleTmpTensor, remoteWinScaleTensor_, scalesCopyParams_, scalesCopyPadParams_);
    scaleInQue.EnQue(scaleTmpTensor);
    scaleTmpTensor = scaleInQue.DeQue<ScalesType>();
    DataCopyPad(localWinScaleTensor_, scaleTmpTensor, scalesCopyParams_);
    scaleInQue.FreeTensor(scaleTmpTensor);
}

template <AllGatherTemplateTypeClass>
__aicore__ inline void AllGatherMte<AllGatherTemplateType>::ExecuteAllGather(GM_ADDR outputTensor, GM_ADDR zTensor)
{
    // +--------+--------+--------+--------+--------+--------+
    // | Rank0  | Rank1  |  ...   | Rank0  | Rank1  |  ...   |
    // |  data  |  data  |  ...   | scales | scales |  ...   |
    // +--------+--------+--------+--------+--------+--------+

    // TODO: 入参为调试用，后续需删除
    // 遍历需要搬运的数据块
    for (uint64_t curBlock = 0; curBlock < mteComm_.assignedBlockNums_; ++curBlock) {
        if (curBlock > 0) {
            mLoopIdx_++;
            if (mLoopIdx_ >= M_) {
                // 切换到下一列搬运
                mLoopIdx_ = 0;
                kLoopIdx_++;
            }
        }
        uint64_t curXOffset = mLoopIdx_ * K_ + kLoopIdx_ * X_PRE_BLOCK_NUM;
        uint32_t remoteRankId = mteComm_.aivId_ / sendCoreNumPerRank_;

        // 获取对端Win区中数据区相关的地址
        GM_ADDR remoteDataGm = mteComm_.GetWinDataAddrGm(remoteRankId) + remoteRankId * xSize_;
        remoteWinXTensor_.SetGlobalBuffer((__gm__ int8_t*)remoteDataGm);
        // 本端对应rank win区数据地址
        uint32_t localRankId = mteComm_.hcclContext_->localUsrRankId;
        // TODO: 正确位置如下，调试完毕后需要修改回来
        // GM_ADDR localDataGm = mteComm_.GetWinDataAddrGm(localRankId) + remoteRankId * xSize_;
        GM_ADDR localDataGm = outputTensor + remoteRankId * xSize_;
        localWinXTensor_.SetGlobalBuffer((__gm__ int8_t*)localDataGm);

        // 读取对端对应地址的 x 数据
        ReadDataBlock(curXOffset);

        // scales 一次搬运完毕
        if (mteComm_.aivId_ % sendCoreNumPerRank_ == 0) {
            GM_ADDR remoteScaleGm = mteComm_.GetWinDataAddrGm(remoteRankId) + mteComm_.winDataSize_ + remoteRankId * scaleSize_;
            remoteWinScaleTensor_.SetGlobalBuffer((__gm__ ScalesType*)remoteScaleGm);
            // TODO: 正确位置如下，调试完毕后需要修改回来
            // GM_ADDR localScaleGm = localDataGm + mteComm_.winDataSize_ + remoteRankId * scaleSize_;
            GM_ADDR localScaleGm = zTensor + remoteRankId * scaleSize_;
            localWinScaleTensor_.SetGlobalBuffer((__gm__ ScalesType*)localScaleGm);
            ReadScales();
        }
    }
}
} // AllGatherImpl
#endif  // ALL_GATHER_MTE_H