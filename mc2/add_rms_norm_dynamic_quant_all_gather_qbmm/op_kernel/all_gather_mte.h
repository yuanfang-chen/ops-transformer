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
#include "all_gather_mte_base.h"
#include "all_gather_mte_utils.h"
#include "all_gather_mte_vec_comp.h"

namespace AllGatherImpl {

using namespace QuantMTECommImpl;
using namespace VectorComputeImpl;
using namespace AscendC;

// 之后可修改成从tiling侧获取数据切块大小
constexpr static uint32_t X_PRE_BLOCK_NUM = 1024U;  // 当前一次搬运一个x数据块，x dtype为 8bit 时对应 1024个x数据. 对于fp4需要另外算

template<TemplateTypeClass>
class AllGatherMte {
public:
    __aicore__ inline AllGatherMte() {};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR scales, GM_ADDR output,
                                TPipe *pipe, const AllGatherTilingData *tilingData);
    __aicore__ inline void Process();
private:
    __aicore__ inline void ClearSumTensor();
    __aicore__ inline void ReadDataBlockDequant(uint64_t curXOffset, uint64_t curScaleOffset);
    __aicore__ inline void ExecuteAllGather();

    uint64_t xSize_{0}; // 单卡上数据大小
    uint64_t xNums_{0}; // 单卡上数据个数
    uint64_t scaleSize_{0}; // 单卡上scale大小
    uint64_t totalWinSize_{0};
    uint64_t tailXNums_{0};
    uint32_t totalBlockNums_{0};

    MTECommunication<TemplateType> mteComm_; // MTE 通信相关实现
    VectorCompute<TemplateType> vecComp_; // vector 计算相关实现

    GlobalTensor<XType> remoteWinXTensor_;
    GlobalTensor<ScalesType> remoteWinScaleTensor_;

    TQue<QuePosition::VECIN, 1> xInQueue_, scaleInQue; // 用于读数据和反量化求和的通算并行
    TBuf<> sumBuf_; // 用于Reduce_sum 求和
};

template <TemplateTypeClass>
__aicore__ inline void AllGatherMte<TemplateType>::Init(GM_ADDR x, GM_ADDR scales,
    GM_ADDR output, TPipe *tPipe, const AllGatherTilingData *tilingData)
{
    // 初始化HcclContext
    mteComm_.InitHcclContext();

    /* all_gather 自己的数据 */
    totalWinSize_ = tilingData->totalWinSize;
    xNums_ = tilingData->M * tilingData->K; // 总的x数据个数， M * h
    xSize_ = xNums_ * sizeof(XType); // 总的x数据量，B
    scaleSize_ = tilingData->M * sizeof(ScalesType);
    tailXNums_ = BlockAlignMod(xNums_, X_PRE_BLOCK_NUM); // 计算最后一个数据块的大小
    totalBlockNums_ = CeilDiv(xSize_, X_BLOCK_BYTES); // 按每次搬运x的数据量分块，得到的总块数
    mteComm_.round_ = totalBlockNums_ / tilingData->aivNum; // 计算总的数据分核搬运需要的轮次数
    mteComm_.tailBlockNums_ = totalBlockNums_ % tilingData->aivNum; // 搬运的尾块数
    mteComm_.ComputeTailAivId(tilingData->aivNum); // 计算最后一个核的id
    tPipe->Reset();
    tPipe->InitBuffer(xInQueue_, BUFFER_NUM, X_BLOCK_BYTES); // 每次拷贝 1024B x; 128 * 8
    tPipe->InitBuffer(scaleInQue, BUFFER_NUM, UB_ALIGN_BYTES); // 每次拷贝 32B scale；4 * 8
    tPipe->InitBuffer(sumBuf_, X_PRE_BLOCK_NUM * sizeof(float)); // 用于Reduce_sum 求和，1024 * 4 = 4k

    // 设置切块大小
    mteComm_.SetBlockSize(X_PRE_BLOCK_NUM, tilingData->aivNum, tailXNums_);
    vecComp_.SetBlockSize(X_PRE_BLOCK_NUM);  

    // 公共MTE搬运参数计算
    mteComm_.InitParams();

    // 初始化tPipe的各种buffer
    mteComm_.InitBuffer(tPipe);
    vecComp_.InitBuffer(tPipe);

    // 初始化GM上的Tensor，包括Win区
    mteComm_.InitGMTensor(x, scales, output, xSize_, scaleSize_, totalWinSize_);
}

template <TemplateTypeClass>
__aicore__ inline void AllGatherMte<TemplateType>::ReadDataBlockDequant(
    uint64_t curXOffset, uint64_t curScaleOffset)
{
    /* 读取 x 从 Win -> UB */
    LocalTensor<XType> xTmpTensor = xInQueue_.AllocTensor<XType>();
    DataCopy(xTmpTensor, remoteWinXTensor_[curXOffset], X_PRE_BLOCK_NUM);
    xInQueue_.EnQue(xTmpTensor);
    xTmpTensor = xInQueue_.DeQue<XType>();

    /* 读取 scale 从 Win -> UB */
    LocalTensor<ScalesType> scaleTmpTensor = scaleInQue.AllocTensor<ScalesType>();
    DataCopy(scaleTmpTensor, remoteWinScaleTensor_[curScaleOffset], mteComm_.scaleNumsPerBlcok_);
    scaleInQue.EnQue(scaleTmpTensor);
    scaleTmpTensor = scaleInQue.DeQue<ScalesType>();

    /* 反量化计算与写回win区 */
    vecComp_.DequantAndCopyBack(
        xTmpTensor, scaleTmpTensor, remoteWinXTensor_[curXOffset]); 
    xInQueue_.FreeTensor(xTmpTensor);
    scaleInQue.FreeTensor(scaleTmpTensor);
}

template <TemplateTypeClass>
__aicore__ inline void AllGatherMte<TemplateType>::ExecuteAllGather()
{   
    // TODO: 修改，最后搬运到 win 区
    // 读状态位，软同步
    mteComm_.ReadStatus(); 
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
            GM_ADDR remoteScaleGm = remoteDataGm + mteComm.hcclContext_->rankSize * xSize_ + remoteRankId * scaleSize_;

            remoteWinXTensor_.SetGlobalBuffer((__gm__ XType*)remoteDataGm);
            remoteWinScaleTensor_.SetGlobalBuffer((__gm__ ScalesType*)remoteScaleGm);

            // 读取对端对应地址的 x 和 scale数据，进行反量化和求和
            //（如果需要反量化，则搬回win区地址需要修改，XOffset和GM地址都需要修改，原空间不够）
            // ReadDataBlockDequant(curXOffset, curScaleOffset);
        }
    }
}

template <TemplateTypeClass>
__aicore__ inline void AllGatherMte<TemplateType>::Process()
{
    // 纯AIV过程
    if ASCEND_IS_AIC {
        return;
    }

    // 这一步不需要，ARN_DQ 之后已经在本地 win 区了
    // 一次性拷贝完所有数据到本地卡win区
    // mteComm_.template CopyDataToWin();
    // 写入状态到状态区
    mteComm_.WriteStatusToWin();
    // 执行AllGather过程：等待状态区同步，读取数据
    ExecuteAllGather();
}
} // AllGatherImpl
#endif  // ALL_GATHER_MTE_H