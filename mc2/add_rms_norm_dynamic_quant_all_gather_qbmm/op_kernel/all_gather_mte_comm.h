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
 * \file all_gather_mte_comm.h
 * \brief MTE通信相关 kernel代码公共部分
 */

#ifndef ALL_GATHER_MTE_COMM_H
#define ALL_GATHER_MTE_COMM_H

#include "adv_api/hccl/hccl.h"
#include "adv_api/reduce/sum.h"
#include "all_gather_mte_utils.h"
#if __has_include("../common/inc/kernel/moe_distribute_base.h")
#include "../common/inc/kernel/moe_distribute_base.h"
#include "../common/inc/kernel/mc2_kernel_utils.h"
#else
#include "../../common/inc/kernel/moe_distribute_base.h"
#include "../../common/inc/kernel/mc2_kernel_utils.h"
#endif

namespace QuantMTECommImpl {

using namespace AscendC;
// 后缀_BYTES 表示单位为字节大小B, _NUM 表示单位为个
constexpr static uint32_t UB_ALIGN_BYTES = 32U;     // UB按32B对齐
constexpr uint32_t FLOAT_UB_ALIGN_NUM = 8U;         // float格式下32B对齐需要 32/4 =8个
constexpr uint32_t BUFFER_NUM = 2U;                 // 用于double buffer
constexpr static uint32_t SCALE_BLCOK_BYTES = 32U;  // 一个scale数据块固定32B，搬运要求32B对齐
constexpr static uint32_t X_BLOCK_BYTES = 1024U;    // 当前一个x数据块固定1024B = 128 * 8(PT) = 32 * 32(MX)，为穿刺取值
constexpr static uint64_t WIN_ADDR_ALIGN = 512UL;   // 每个核的标志位间512B对齐，win区数据部分也512B对齐

#define AllGatherTemplateTypeClass typename XType, typename ScalesType, typename OutputType
#define AllGatherTemplateType XType, ScalesType, OutputType

template<AllGatherTemplateTypeClass>
class MTECommunication {
public:
    __aicore__ inline MTECommunication() {};
    __aicore__ inline void InitHcclContext();
    __aicore__ inline void InitParams(uint64_t xSize);
    __aicore__ inline void InitGMTensor(uint64_t alignedXSize, uint64_t alignedScaleSize);
    __aicore__ inline void InitBuffer(TPipe *tPipe);
    __aicore__ inline void SetBlockSize(uint32_t elementsPerBlock, uint64_t aivNum, uint64_t lastBlockNum);
    __aicore__ inline void WriteStatusToWin();
    __aicore__ inline void ReadStatus();
    __aicore__ inline void CopyResultToOutput(uint64_t outOffsetGM, LocalTensor<float>& localResultTensor, uint32_t count);
    __aicore__ inline void ComputeTailAivId(uint64_t totalAivCount);
    __aicore__ inline GM_ADDR GetWinDataAddrGm(uint32_t rankId);
    __aicore__ inline GM_ADDR GetWinStatusAddrGm(uint32_t rankId);

    __gm__ HcclOpResParam *hcclContext_;
    uint32_t aivId_{0};
    uint64_t aivNum_{0};
    uint32_t round_{0};
    uint32_t tailBlockNums_{0};
    uint32_t assignedBlockNums_{0};
    uint64_t scaleNumsPerBlcok_{0};
    uint64_t xOffset_{0};  
    uint64_t scaleOffset_{0};
    uint64_t lastAivId_{0};
    uint64_t winDataSize_{0};

private:    
    uint32_t xNumPerBlock_{0};
    uint64_t tailXNums_{0};

    GlobalTensor<XType> localWinXGMTensor_;
    GlobalTensor<ScalesType> localWinScaleGMTensor_;
    GlobalTensor<OutputType> outputTensor_;

    LocalTensor<float> stateResetTensor_;
    LocalTensor<OutputType> xOutTensor_;

    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> xQueue_, scaleQueue_;
    TQue<QuePosition::VECOUT, 1> xOutQueue_;
    TBuf<> writeStateBuf_;
    TBuf<> readStateBuf_;
    TBuf<> stateResetBuf_;
};

template <AllGatherTemplateTypeClass>
__aicore__ inline void MTECommunication<AllGatherTemplateType>::InitHcclContext()
{
    hcclContext_ = (__gm__ HcclOpResParam*)GetHcclContext<HCCL_GROUP_ID_0>();
}

template <AllGatherTemplateTypeClass>
__aicore__ inline void MTECommunication<AllGatherTemplateType>::InitParams(uint64_t xSize)
{
    aivId_ = GetBlockIdx(); // 获取当前核Id
    scaleNumsPerBlcok_ = SCALE_BLCOK_BYTES / sizeof(ScalesType); // 一块scale固定32B, 计算包含多少个数据
    assignedBlockNums_ = aivId_ < tailBlockNums_ ? round_ + 1 : round_; // 当前核分配到的数据块数量，顺序分核，序号小的核多搬一轮
    uint64_t blockIdx = aivId_ * round_ + (aivId_ < tailBlockNums_ ? aivId_ : tailBlockNums_); // 计算当前核分派到的首个数据块序列号
    xOffset_ = blockIdx * xNumPerBlock_; // 块数 * 一块有多少数据，得到要搬第几个x数据
    scaleOffset_ = blockIdx * scaleNumsPerBlcok_; // 计算要搬第几个 scale
    winDataSize_ = CeilAlign(hcclContext_->rankSize * xSize, WIN_ADDR_ALIGN);   // win区数据部分大小
}

template <AllGatherTemplateTypeClass>
__aicore__ inline void MTECommunication<AllGatherTemplateType>::InitGMTensor(uint64_t xSize, uint64_t scaleSize)
{
    // 入参相关数据的GMTensor
    // outputTensor_.SetGlobalBuffer((__gm__ OutputType*)output);

    // 获取本卡地址写数据
    // 通过rankId获取本地数据区地址对应卡的数据区域
    // +--------+--------+--------+--------+--------+--------+
    // | Rank0  | Rank1  |  ...   | Rank0  | Rank1  |  ...   |
    // |  data  |  data  |  ...   | scales | scales |  ...   |
    // +--------+--------+--------+--------+--------+--------+
    GM_ADDR localDataGm = GetWinDataAddrGm(hcclContext_->localUsrRankId) + hcclContext_->localUsrRankId * xSize;
    GM_ADDR localScaleGm = localDataGm + winDataSize_ + hcclContext_->localUsrRankId * scaleSize;
    localWinXGMTensor_.SetGlobalBuffer((__gm__ XType*)localDataGm);
    localWinScaleGMTensor_.SetGlobalBuffer((__gm__ ScalesType*)localScaleGm); // sclae数据跟在x后
}

template <AllGatherTemplateTypeClass>
__aicore__ inline void MTECommunication<AllGatherTemplateType>::InitBuffer(TPipe *tPipe)
{
    tPipe->InitBuffer(xQueue_, BUFFER_NUM, X_BLOCK_BYTES);    
    tPipe->InitBuffer(scaleQueue_, BUFFER_NUM, UB_ALIGN_BYTES);     
    tPipe->InitBuffer(xOutQueue_, BUFFER_NUM, xNumPerBlock_ * sizeof(OutputType)); // 用于输出的OutPutTensor
    tPipe->InitBuffer(writeStateBuf_, UB_ALIGN_BYTES); // 状态位每一个按32B对齐
    tPipe->InitBuffer(readStateBuf_, hcclContext_->rankSize * UB_ALIGN_BYTES); // 每次读 rankSize 个状态位
    tPipe->InitBuffer(stateResetBuf_, hcclContext_->rankSize * UB_ALIGN_BYTES); // 用于清理状态区

    stateResetTensor_ = stateResetBuf_.Get<float>();
    Duplicate<float>(stateResetTensor_, (float)0.0, static_cast<uint32_t>(hcclContext_->rankSize * FLOAT_UB_ALIGN_NUM)); // 用于状态区清零
}

/**
 * @brief 配置数据块划分参数，用于多核并行计算
 * 
 * @param elementsPerBlock 每个标准数据块包含的X元素数量
 * @param aivNum AIV核的总数，用于数据分发和负载均衡
 * @param lastBlockNum 最后一个核处理的尾部数据块元素数量
 */
template <AllGatherTemplateTypeClass>
__aicore__ inline void MTECommunication<AllGatherTemplateType>::SetBlockSize(uint32_t elementsPerBlock, uint64_t aivNum, uint64_t lastBlockNum)
{
    xNumPerBlock_ = elementsPerBlock;
    aivNum_ = aivNum;
    tailXNums_ = lastBlockNum;
}

/**
 * @brief 计算负责处理尾部数据的AI核ID
 * 
 * @param totalAivCount AIV核的总数量
 * 
 * @note 当数据量较小时，此时计算仅仅只有一轮（round_ == 0），此时处理尾块的aiv并非最后一个，
 *       需根据当前数据块数量计算。
 */
template <AllGatherTemplateTypeClass>
__aicore__ inline void MTECommunication<AllGatherTemplateType>::ComputeTailAivId(uint64_t totalAivCount)
{
    if (round_ == 0) {
        // 小数据量时，如果只有一轮搬运，负责尾块的aiv由此时计算的总块数决定
        lastAivId_ = tailBlockNums_ - 1;
    } else {
        // 轮次大于一轮时，负责尾块的aiv必定是最后一个
        lastAivId_ = totalAivCount - 1;
    }
}

/**
 * @brief 向Win区状态区写入本核完成数据搬运状态
 * 
 * 该函数负责将当前AI Core（核）的数据搬运完成状态写入到所有Rank的状态Win区中，
 * 通知其他设备当前核的数据准备已完成。这是集群通信软同步机制的一部分，
 * 每个核需要向所有Rank（包括本机和其他设备）的状态窗口写入标识，
 * 确保所有设备都能感知到当前核的完成状态。
 */
template <AllGatherTemplateTypeClass>
__aicore__ inline void MTECommunication<AllGatherTemplateType>::WriteStatusToWin()
{
    uint32_t coreOffset = aivId_ * hcclContext_->rankSize; // Win区大小为 aivNum * rankSize, 此处计算核偏移
    // 遍历每一张卡，给每一张卡都要写入状态
    for (uint32_t curRank = 0; curRank < hcclContext_->rankSize; ++curRank) {
        // 写入状态到对端，每个核写一个状态，表示自己的数据块已经写完
        LocalTensor<float> statusTensor = writeStateBuf_.Get<float>();
        DataCopy<float>(statusTensor, stateResetTensor_, FLOAT_UB_ALIGN_NUM); // 先重置statusTensor数据，后面累加需要Tensor内全部数据，防止脏数据
        SyncFunc<AscendC::HardEvent::MTE2_S>();
        statusTensor(0) = (float)1.0;  // 用1标识
        GM_ADDR remoteWinStateGM = GetWinStatusAddrGm(curRank); // 获取当前要写对端卡的状态区地址
        GlobalTensor<float> stateGMTensor;
        stateGMTensor.SetGlobalBuffer((__gm__ float*)remoteWinStateGM);
        // 不同卡上的核的状态写到相邻位置，读时可以一次读rankSize个状态, 状态区大小设计为 aivNum * ranDim
        uint64_t curOffset = (coreOffset + hcclContext_->localUsrRankId) * FLOAT_UB_ALIGN_NUM; // 当前核偏移 + 卡偏移， 按32B对齐
        SyncFunc<AscendC::HardEvent::S_MTE3>();
        DataCopy(stateGMTensor[curOffset], statusTensor, FLOAT_UB_ALIGN_NUM); // 按32B对齐拷贝
        SyncFunc<AscendC::HardEvent::MTE3_S>();
    }
}

/**
 * @brief 读取Win区状态区并等待同步状态
 * 
 * 该函数从本Rank的状态Win区读取同步标识，并在本地进行忙等待，直到所有参与通信的Rank
 * 都已完成状态设置。这种软同步机制确保所有Rank上当前核（AI Core）所需的数据都已
 * 准备就绪，从而避免跨设备数据不一致性问题。
 */
template <AllGatherTemplateTypeClass>
__aicore__ inline void MTECommunication<AllGatherTemplateType>::ReadStatus()
{
    GM_ADDR stateGM = GetWinStatusAddrGm(hcclContext_->localUsrRankId); // 获取本卡的状态区用于读取
    GlobalTensor<float> selfStatusWinTensor;
    uint32_t offset = aivId_ * hcclContext_->rankSize * FLOAT_UB_ALIGN_NUM; // 获取当前核所需读取状态位的头地址，状态按32B对齐
    selfStatusWinTensor.SetGlobalBuffer((__gm__ float*)(stateGM));
    LocalTensor<float> statusTensor = readStateBuf_.Get<float>();
    float flag = 0; // 用于计算状态和
    uint32_t statusCnt = hcclContext_->rankSize * FLOAT_UB_ALIGN_NUM; // 一次读rankSize个，按32B对齐
    SumParams sumParams{1, statusCnt, statusCnt};
    float minTarget = hcclContext_->rankSize - (float)0.5;
    float maxTarget = hcclContext_->rankSize + (float)0.5;
    // 读取statusCnt个数据求和
    while ((flag < minTarget) || (flag > maxTarget)) {
        SyncFunc<AscendC::HardEvent::S_MTE2>();
        DataCopy<float>(statusTensor, selfStatusWinTensor[offset], statusCnt);
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        Sum(statusTensor, statusTensor, sumParams);
        SyncFunc<AscendC::HardEvent::V_S>();
        flag = statusTensor(0);
    }
    SyncFunc<AscendC::HardEvent::S_MTE3>();
    DataCopy<float>(selfStatusWinTensor[offset], stateResetTensor_, statusCnt); // 相关状态区重新置零
}

/**
 * @brief 将计算得到的UB上的结果Tensor的数据复制到GM上的OutPutTensor，并进行类型转换（如果需要）。
 * 
 * @param outputOffset 输出OutputTensor的GM偏移量，用于指定目标位置。
 * @param sourceTensor 本地UB上计算结果Tensor，包含计算完成的数据。
 * @param count 当前每次处理数据块的元素数量
 */
template <AllGatherTemplateTypeClass>
__aicore__ inline void MTECommunication<AllGatherTemplateType>::CopyResultToOutput(uint64_t outOffsetGM, LocalTensor<float>& localResultTensor, uint32_t count)
{
    // 将计算好的数据拷贝到输出tensor，如果是非float数据类型需要先转换成目标数据类型
    xOutTensor_ = xOutQueue_.AllocTensor<OutputType>();
    if constexpr (AscendC::IsSameType<OutputType, float>::value) {
        DataCopy(xOutTensor_, localResultTensor, count);
    } else {
        Cast(xOutTensor_, localResultTensor, RoundMode::CAST_RINT, count);
    }
    xOutQueue_.EnQue(xOutTensor_);
    xOutTensor_ = xOutQueue_.DeQue<OutputType>();
    DataCopy(outputTensor_[outOffsetGM], xOutTensor_, count);
    xOutQueue_.FreeTensor(xOutTensor_);
}

// 获取对应rank的Win区数据区的地址
template <AllGatherTemplateTypeClass>
__aicore__ inline GM_ADDR MTECommunication<AllGatherTemplateType>::GetWinDataAddrGm(uint32_t rankId)
{
    if (rankId == hcclContext_->localUsrRankId) {
        return (GM_ADDR)(hcclContext_->localWindowsIn);
    }
    return (GM_ADDR)(((HcclRankRelationResV2 *)(hcclContext_->remoteRes[rankId].nextDevicePtr))->windowsIn);
}

// 获取对应rank的Win区状态区的地址
template <AllGatherTemplateTypeClass>
__aicore__ inline GM_ADDR MTECommunication<AllGatherTemplateType>::GetWinStatusAddrGm(uint32_t rankId)
{
    if (rankId == hcclContext_->localUsrRankId) {
        return (GM_ADDR)(hcclContext_->localWindowsExp);
    }
    return (GM_ADDR)(((HcclRankRelationResV2 *)(hcclContext_->remoteRes[rankId].nextDevicePtr))->windowsExp);
}
} // QuantMTECommImpl
#endif  // ALL_GATHER_MTE_COMM_H