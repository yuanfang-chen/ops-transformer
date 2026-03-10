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
constexpr static uint32_t X_BLOCK_BYTES = 512U;    // 当前一个x数据块固定512B = 512 * sizeof(INT8)，为穿刺取值
constexpr static uint32_t SCALES_BLOCK_BYTES = 256U;     // 63*4B并按照32B对齐
constexpr static uint64_t WIN_ADDR_ALIGN = 512UL;   // win区数据部分512B对齐
constexpr static uint64_t CV_SYNC_START_OFFSET = 10UL * 1024UL; // CV同步状态相对于通信状态向后偏移10K
constexpr static uint64_t CV_STATE_ALIGN = 64UL;    // CV同步的标志位间64B对齐
constexpr static uint64_t WIN_DATA_OFFSET = 100UL * 1024UL * 1024UL;   // win区数据区1区起始偏移，100MB
constexpr static uint64_t WIN_STATUS_OFFSET = 500UL * 1024UL;   // win区状态区1区起始偏移，500KB
constexpr static uint64_t WIN_AIV_ZERONE_FLAG_OFFSET = 1000UL * 1024UL;   // V核win区0/1标志位区起始偏移，1000KB
constexpr static uint64_t WIN_AIC_ZERONE_FLAG_OFFSET = 1005UL * 1024UL;   // C核win区0/1标志位区起始偏移，1005KB
constexpr static uint64_t ZERONE_STATUS_ALIGN = 64UL;   // 0/1分区标志位按照64B对齐（刷新DataCache最小单位）
constexpr static uint32_t ZERONE_STATUS_POS = 0U;       // 0/1分区标志位index

/**
 * 状态区划分：
 * ┌─────────────────────────────────────────────────────────────────────────────────────────┐
 * │                                       状态区 (1024K)                                     │
 * ├─────────────┬─────────────┬─────────────┬─────────────┬──────────┬───────────┬──────────┤
 * │  算子1 0区   │  算子2 0区  │  算子1 1区   │  算子2 1区  │ 算子1 V核 │ 算子1 C核 │  算子2   │
 * │             │             │             │             │ 状态位区  │ 状态位区  │ 状态位区  │
 * ├─────────────┼─────────────┼─────────────┼─────────────┼──────────┼───────────┼──────────┤
 * │    0~250K   │  250~500K   │  500~750K   │  750~1000K  │1000~1005 │ 1005~1010 │1010~1024 │
 * │             │             │             │             │    K     │     K     │    K     │
 * └─────────────┴─────────────┴─────────────┴─────────────┴──────────┴───────────┴──────────┘
 * 
 * ┌─────────────────────────────────────────────────────────────┐
 * │                    算子1单区 (250K)                          │
 * ├─────────────┬────────────────────┬──────────────────────────┤
 * │  通信状态区  │      CV状态区       │        保留区域          │
 * ├─────────────┼────────────────────┼──────────────────────────┤
 * │    0~10K    │     10~40K         │        40~250K           │
 * └─────────────┴────────────────────┴──────────────────────────┘
 * 
 * ┌────────────────────────────────────────┐
 * │            算子2单区 (250K)             │
 * ├─────────────┬──────────────────────────┤
 * │  通信状态区  │         保留区域          │
 * ├─────────────┼──────────────────────────┤
 * │    0~10K    │         10~250K          │
 * └─────────────┴──────────────────────────┘
 *
 *
 * 数据区划分：
 * ┌────────────────────────────────────────────────────────────────────────────┐
 * │                             数据区 (200MB)                                  │
 * ├──────────────┬──────────────┬──────────────┬───────────────────────────────┤
 * │   算子1 0区   │  算子2 0区   │   算子2 1区   │        算子2 1区              │
 * ├──────────────┼──────────────┼──────────────┼───────────────────────────────┤
 * │   1~30M      │   30~100M    │   100~130M   │        130~200M               │
 * └──────────────┴──────────────┴──────────────┴───────────────────────────────┘
 */

#define AllGatherTemplateTypeClass typename XType, typename ScalesType, typename OutputType, bool isCVSync, bool isOptionalOutput
#define AllGatherTemplateType XType, ScalesType, OutputType, isCVSync, isOptionalOutput

template<AllGatherTemplateTypeClass>
class MTECommunication {
public:
    __aicore__ inline MTECommunication() {};
    __aicore__ inline void InitHcclContext();
    __aicore__ inline void InitParams(uint64_t xSize, uint64_t aivNum, uint32_t sendCoreNumPerRank);
    __aicore__ inline void InitBuffer(TPipe *tPipe);
    __aicore__ inline void ReadAndFlipWinFlag();
    __aicore__ inline void WriteStatusToWin();
    __aicore__ inline void ReadStatus();
    __aicore__ inline GM_ADDR GetWinDataAddrGm(uint32_t rankId, uint32_t winFlag);
    __aicore__ inline GM_ADDR GetWinStatusAddrGm(uint32_t rankId, uint32_t winFlag);
    __aicore__ inline GM_ADDR GetWinZeroneFlagAddrGm(uint32_t rankId);

    __gm__ HcclOpResParam *hcclContext_;
    uint32_t aivId_{0};
    uint32_t aicId_{0};
    uint64_t aivNum_{0};
    uint64_t winDataSize_{0};
    uint32_t curDstId_{0};
    uint32_t sendCoreNumPerRank_{0};
    uint32_t curRankId_{0};
    uint32_t winFlag_{0};

private:
    GlobalTensor<uint32_t> selfWinFlagGMTensor_;
    LocalTensor<float> stateResetTensor_;

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
__aicore__ inline void MTECommunication<AllGatherTemplateType>::InitParams(
    uint64_t xSize, uint64_t aivNum, uint32_t sendCoreNumPerRank)
{
    aivNum_ = aivNum;
    aivId_ = GetBlockIdx(); // 获取当前核Id
    aicId_ = aivId_ / GetTaskRation();    // C V 1:2
    winDataSize_ = CeilAlignU64(hcclContext_->rankSize * xSize, WIN_ADDR_ALIGN);   // win区数据部分大小
    curRankId_ = hcclContext_->localUsrRankId;
    sendCoreNumPerRank_ = sendCoreNumPerRank;
    curDstId_ = aicId_ / sendCoreNumPerRank_;
}

template <AllGatherTemplateTypeClass>
__aicore__ inline void MTECommunication<AllGatherTemplateType>::ReadAndFlipWinFlag()
{
    // 处理 0/1 分区标志位
    uint64_t curCoreFlagAddr = (uint64_t)GetWinZeroneFlagAddrGm(curRankId_) + aivId_ * ZERONE_STATUS_ALIGN;
    selfWinFlagGMTensor_.SetGlobalBuffer((__gm__ uint32_t*)curCoreFlagAddr);
    // 获取标志位
    DataCacheCleanAndInvalid<uint32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(selfWinFlagGMTensor_);
    winFlag_ = selfWinFlagGMTensor_.GetValue(ZERONE_STATUS_POS);
    // 翻转标志位
    selfWinFlagGMTensor_.SetValue(ZERONE_STATUS_POS, 1 - winFlag_);
    DataCacheCleanAndInvalid<uint32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(selfWinFlagGMTensor_);
}

template <AllGatherTemplateTypeClass>
__aicore__ inline void MTECommunication<AllGatherTemplateType>::InitBuffer(TPipe *tPipe)
{
    tPipe->InitBuffer(writeStateBuf_, UB_ALIGN_BYTES); // 状态位每一个按32B对齐
    tPipe->InitBuffer(readStateBuf_, hcclContext_->rankSize * UB_ALIGN_BYTES); // 每次读 rankSize 个状态位
    tPipe->InitBuffer(stateResetBuf_, hcclContext_->rankSize * UB_ALIGN_BYTES); // 用于清理状态区

    stateResetTensor_ = stateResetBuf_.Get<float>();
    Duplicate<float>(stateResetTensor_, (float)0.0, static_cast<uint32_t>(hcclContext_->rankSize * FLOAT_UB_ALIGN_NUM)); // 用于状态区清零
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
    // Win区大小为aivNum，此处计算核偏移，每个rank有sendCoreNumPerRank_个状态位
    uint32_t curOffset = (aicId_ % sendCoreNumPerRank_ + curRankId_ * sendCoreNumPerRank_) * FLOAT_UB_ALIGN_NUM;
    // 写入状态到对端，每个核写一个状态，sendCoreNumPerRank_个核负责一个对端
    LocalTensor<float> statusTensor = writeStateBuf_.Get<float>();
    DataCopy<float>(statusTensor, stateResetTensor_, FLOAT_UB_ALIGN_NUM); // 先重置statusTensor数据，后面累加需要Tensor内全部数据，防止脏数据
    SyncFunc<AscendC::HardEvent::MTE2_S>();
    statusTensor(0) = (float)1.0;  // 用1标识
    GM_ADDR remoteWinStateGM = GetWinStatusAddrGm(curDstId_, winFlag_); // 获取当前要写对端卡的状态区地址
    GlobalTensor<float> stateGMTensor;
    stateGMTensor.SetGlobalBuffer((__gm__ float*)remoteWinStateGM);
    SyncFunc<AscendC::HardEvent::S_MTE3>();
    DataCopy(stateGMTensor[curOffset], statusTensor, FLOAT_UB_ALIGN_NUM); // 按32B对齐拷贝
    SyncFunc<AscendC::HardEvent::MTE3_S>();
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
    GM_ADDR stateGM = GetWinStatusAddrGm(curRankId_, winFlag_); // 获取本卡的状态区用于读取
    GlobalTensor<float> selfStatusWinTensor;
    // 获取当前核所需读取状态位的头地址，状态按32B对齐
    selfStatusWinTensor.SetGlobalBuffer((__gm__ float*)(stateGM));
    uint32_t offset = aicId_ * FLOAT_UB_ALIGN_NUM;
    LocalTensor<float> statusTensor = readStateBuf_.Get<float>();
    float flag = 0; // 用于计算状态和
    uint32_t statusCnt = FLOAT_UB_ALIGN_NUM; // 一次读一个，按32B对齐
    SumParams sumParams{1, statusCnt, statusCnt};
    float minTarget = (float)0.5;
    float maxTarget = (float)1.5;
    // 读取statusCnt个数据求和
    while ((flag < minTarget) || (flag > maxTarget)) {
        SyncFunc<AscendC::HardEvent::S_MTE2>();
        DataCopy<float>(statusTensor, selfStatusWinTensor[offset], statusCnt);
        SyncFunc<AscendC::HardEvent::MTE2_S>();
        flag = statusTensor(0);
    }
    SyncFunc<AscendC::HardEvent::S_MTE3>();
    DataCopy<float>(selfStatusWinTensor[offset], stateResetTensor_, statusCnt); // 相关状态区重新置零
}

// 获取对应rank的Win区数据区的地址
template <AllGatherTemplateTypeClass>
__aicore__ inline GM_ADDR MTECommunication<AllGatherTemplateType>::GetWinDataAddrGm(
    uint32_t rankId, uint32_t winFlag)
{
    if (rankId == curRankId_) {
        return (GM_ADDR)(hcclContext_->localWindowsIn + winFlag * WIN_DATA_OFFSET);
    }
    return (GM_ADDR)(((HcclRankRelationResV2 *)(hcclContext_->remoteRes[rankId].nextDevicePtr))->windowsIn
        + winFlag * WIN_DATA_OFFSET);
}

// 获取对应rank的Win区状态区的地址
template <AllGatherTemplateTypeClass>
__aicore__ inline GM_ADDR MTECommunication<AllGatherTemplateType>::GetWinStatusAddrGm(
    uint32_t rankId, uint32_t winFlag)
{
    if (rankId == curRankId_) {
        return (GM_ADDR)(hcclContext_->localWindowsExp + winFlag * WIN_STATUS_OFFSET);
    }
    return (GM_ADDR)(((HcclRankRelationResV2 *)(hcclContext_->remoteRes[rankId].nextDevicePtr))->windowsExp
        + winFlag * WIN_STATUS_OFFSET);
}

// 获取对应rank的0/1标志位区的地址
template <AllGatherTemplateTypeClass>
__aicore__ inline GM_ADDR MTECommunication<AllGatherTemplateType>::GetWinZeroneFlagAddrGm(uint32_t rankId)
{
    if ASCEND_IS_AIV {
        return (GM_ADDR)(GetWinStatusAddrGm(rankId, 0) + WIN_AIV_ZERONE_FLAG_OFFSET);
    } else {
        return (GM_ADDR)(GetWinStatusAddrGm(rankId, 0) + WIN_AIC_ZERONE_FLAG_OFFSET);
    }
}
} // QuantMTECommImpl
#endif  // ALL_GATHER_MTE_COMM_H