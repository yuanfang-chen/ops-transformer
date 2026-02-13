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
#include "all_gather_mte_vec_comp.h"

namespace AllGatherImpl {

using namespace QuantMTECommImpl;
using namespace VectorComputeImpl;
using namespace AscendC;

// 之后可修改成从tiling侧获取数据切块大小
constexpr static uint32_t X_PRE_BLOCK_NUM = 1024U;  // 当前一次搬运一个x数据块，x dtype为 8bit 时对应 1024个x数据. 对于fp4需要另外算

template<AllGatherTemplateTypeClass>
class AllGatherMte {
public:
    __aicore__ inline AllGatherMte() {};
    __aicore__ inline void Init(TPipe *tPipe, uint32_t M, uint32_t Ka, uint32_t aivNum);
    __aicore__ inline void Process();

    __aicore__ inline void SetRemoteFlag();
    __aicore__ inline void WaitRemoteFlag();
    __aicore__ inline void ExecuteAllGather();

private:
    __aicore__ inline void ReadDataBlockDequant(uint64_t curXOffset, uint64_t curScaleOffset);

    uint64_t xSize_{0}; // 单卡上数据大小
    uint64_t xNums_{0}; // 单卡上数据个数
    uint64_t scaleSize_{0}; // 单卡上scale大小
    uint64_t tailXNums_{0};
    uint32_t totalBlockNums_{0};

    MTECommunication<AllGatherTemplateType> mteComm_; // MTE 通信相关实现
    VectorCompute<AllGatherTemplateType> vecComp_; // vector 计算相关实现

    GlobalTensor<int8_t> remoteWinXTensor_;
    GlobalTensor<ScalesType> remoteWinScaleTensor_;
    GlobalTensor<int8_t> localWinXTensor_;
    GlobalTensor<ScalesType> localWinScaleTensor_;

    TQue<QuePosition::VECIN, 1> xInQueue_, scaleInQue; // 用于读数据和反量化求和的通算并行
    TBuf<> sumBuf_; // 用于Reduce_sum 求和
};

template <AllGatherTemplateTypeClass>
__aicore__ inline void AllGatherMte<AllGatherTemplateType>::Init(TPipe *tPipe, uint32_t M, uint32_t Ka, uint32_t aivNum)
{
    // 初始化HcclContext
    mteComm_.InitHcclContext();

    /* all_gather 自己的数据 */
    xNums_ = M * Ka; // 总的x数据个数， M * h
    xSize_ = xNums_ * sizeof(XType); // 总的x数据量，B
    scaleSize_ = M * sizeof(ScalesType);
    tailXNums_ = BlockAlignMod(xNums_, X_PRE_BLOCK_NUM); // 计算最后一个数据块的大小
    totalBlockNums_ = CeilDiv(xSize_, X_BLOCK_BYTES); // 按每次搬运x的数据量分块，得到的总块数
    mteComm_.round_ = totalBlockNums_ / aivNum; // 计算总的数据分核搬运需要的轮次数
    mteComm_.tailBlockNums_ = totalBlockNums_ % aivNum; // 搬运的尾块数
    mteComm_.ComputeTailAivId(aivNum); // 计算最后一个核的id
    tPipe->Reset();
    tPipe->InitBuffer(xInQueue_, BUFFER_NUM, X_BLOCK_BYTES); // 每次拷贝 1024B x; 128 * 8
    tPipe->InitBuffer(scaleInQue, BUFFER_NUM, UB_ALIGN_BYTES); // 每次拷贝 32B scale；4 * 8
    tPipe->InitBuffer(sumBuf_, X_PRE_BLOCK_NUM * sizeof(float)); // 用于Reduce_sum 求和，1024 * 4 = 4k

    // 设置切块大小
    mteComm_.SetBlockSize(X_PRE_BLOCK_NUM, aivNum, tailXNums_);
    vecComp_.SetBlockSize(X_PRE_BLOCK_NUM);  

    // 公共MTE搬运参数计算
    mteComm_.InitParams(xSize_);

    // 初始化tPipe的各种buffer
    mteComm_.InitBuffer(tPipe);
    vecComp_.InitBuffer(tPipe);

    // 初始化GM上的Tensor，包括Win区
    mteComm_.InitGMTensor(xSize_, scaleSize_);
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
__aicore__ inline void AllGatherMte<AllGatherTemplateType>::ReadDataBlockDequant(
    uint64_t curXOffset, uint64_t curScaleOffset)
{
    /* 读取 x 从 远端Win -> UB -> 本端Win */
    LocalTensor<int8_t> xTmpTensor = xInQueue_.AllocTensor<int8_t>();
    DataCopy(xTmpTensor, remoteWinXTensor_[curXOffset], X_PRE_BLOCK_NUM);
    xInQueue_.EnQue(xTmpTensor);
    xTmpTensor = xInQueue_.DeQue<int8_t>();
    DataCopy(localWinXTensor_[curXOffset], xTmpTensor, X_PRE_BLOCK_NUM);
    xInQueue_.FreeTensor(xTmpTensor);

    /* 读取 scale 从 远端Win -> UB -> 本端Win */
    LocalTensor<ScalesType> scaleTmpTensor = scaleInQue.AllocTensor<ScalesType>();
    DataCopy(scaleTmpTensor, remoteWinScaleTensor_[curScaleOffset], mteComm_.scaleNumsPerBlcok_);
    scaleInQue.EnQue(scaleTmpTensor);
    scaleTmpTensor = scaleInQue.DeQue<ScalesType>();
    DataCopy(localWinScaleTensor_[curScaleOffset], scaleTmpTensor, mteComm_.scaleNumsPerBlcok_);
    scaleInQue.FreeTensor(scaleTmpTensor);
}

template <AllGatherTemplateTypeClass>
__aicore__ inline void AllGatherMte<AllGatherTemplateType>::ExecuteAllGather()
{
    // 遍历需要搬运的数据块
    for (uint64_t curBlock = 0; curBlock < mteComm_.assignedBlockNums_; ++curBlock) {
        uint64_t curXOffset = mteComm_.xOffset_ + curBlock * X_PRE_BLOCK_NUM;
        uint64_t curScaleOffset = mteComm_.scaleOffset_ + curBlock * mteComm_.scaleNumsPerBlcok_;
        // 遍历每张卡，读取其Win区的数据，采取错卡序读取，从自己卡上读起
        /* rank0: [0,1,2]; rank1: [1,2,0]; rank2: [2,0,1] */
        uint32_t startRankId = mteComm_.hcclContext_->localUsrRankId;
        for (uint32_t i = 0; i < mteComm_.hcclContext_->rankSize; ++i) {
            uint32_t remoteRankId = (startRankId + i) % mteComm_.hcclContext_->rankSize;

            // 获取对端Win区中数据区相关的地址
            GM_ADDR remoteDataGm = mteComm_.GetWinDataAddrGm(remoteRankId) + remoteRankId * xSize_;
            GM_ADDR remoteScaleGm = remoteDataGm + mteComm_.winDataSize_ + remoteRankId * scaleSize_;
            remoteWinXTensor_.SetGlobalBuffer((__gm__ int8_t*)remoteDataGm);
            remoteWinScaleTensor_.SetGlobalBuffer((__gm__ ScalesType*)remoteScaleGm);
            
            // 本端对应rank win区数据地址
            uint32_t localRankId = mteComm_.hcclContext_->localUsrRankId;
            GM_ADDR localDataGm = mteComm_.GetWinDataAddrGm(localRankId) + remoteRankId * xSize_;
            GM_ADDR localScaleGm = localDataGm + mteComm_.winDataSize_ + remoteRankId * scaleSize_;
            localWinXTensor_.SetGlobalBuffer((__gm__ int8_t*)localDataGm);
            localWinScaleTensor_.SetGlobalBuffer((__gm__ ScalesType*)localScaleGm);

            // 读取对端对应地址的 x 和 scale数据
            ReadDataBlockDequant(curXOffset, curScaleOffset);
        }
    }
}

template <AllGatherTemplateTypeClass>
__aicore__ inline void AllGatherMte<AllGatherTemplateType>::Process()
{
    // 纯AIV过程
    if ASCEND_IS_AIC {
        return;
    }
    // 写入状态到状态区
    mteComm_.WriteStatusToWin();
    // 执行AllGather过程：等待状态区同步，读取数据
    ExecuteAllGather();
}
} // AllGatherImpl
#endif  // ALL_GATHER_MTE_H