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
 * \file moe_distribute_a2_base.h
 * \brief Unified hccl buffer management for the A2 hierarchy's Dispatch and Combine operations.
 */

#ifndef MOE_DISTRIBUTE_A2_BASE_H
#define MOE_DISTRIBUTE_A2_BASE_H
#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "kernel_tiling/kernel_tiling.h"

#if __has_include("../../common/inc/kernel/moe_distribute_base.h")
#include "../../common/inc/kernel/moe_distribute_base.h"
#else
#include "../../../common/inc/kernel/moe_distribute_base.h"
#endif
namespace MoeDistributeA2Base {
class MoeDistributeA2Context {
public:
    __aicore__ inline void Init(GM_ADDR hcclContext)
    {
        hcclContext_ = (__gm__ HcclA2CombineOpParam *)hcclContext;
    }
    
    __aicore__ inline GM_ADDR GetWindowsInAddr(uint32_t rankId) const
    {
        if (hcclContext_->multiFlag == 0U) {
            return (GM_ADDR)(hcclContext_->windowsIn[rankId]);
        } else {
            if (rankId == hcclContext_->rankId) {
                return (GM_ADDR)(hcclContext_->data[rankId].localInput.addr);
            } else {
                return (GM_ADDR)(hcclContext_->data[rankId].remoteInput.addr);
            }
        }
    }

    __aicore__ inline GM_ADDR GetWindowsOutAddr(uint32_t rankId) const
    {
        if (hcclContext_->multiFlag == 0U) {
            return (GM_ADDR)(hcclContext_->windowsOut[rankId]);
        } else {
            if (rankId == hcclContext_->rankId) {
                return (GM_ADDR)(hcclContext_->data[rankId].localOutput.addr);
            } else {
                return (GM_ADDR)(hcclContext_->data[rankId].remoteOutput.addr);
            }
        }
    }

private:
    __gm__ HcclA2CombineOpParam *hcclContext_;
};

template <typename T>
inline __aicore__ T RoundUp(const T val, const T align)
{
    static_assert(std::is_arithmetic<T>::value, "T must be an arithmetic type");
    if (align == 0 || val + align - 1 < val) {
        return val;
    }
    return (val + align - 1) / align * align;
}

/* 
HCCL_BUFF结构如下：
W = HCCL_BUFFSIZE
A1 = (epWorldSize / 8) * 512B
TOKEN_SIZE = (H * sizeof(dtype) + align8(K) * sizeof(uint32) * 4) * 1B
MAXBS_TOKEN_SIZE = maxBs * TOKEN_SIZE * 1B
IPC_DATA_SIZE_PER_EXP = epWorldSize * align512(MAXBS_TOKEN_SIZE) * 1B
RDMA_DATA_SIZE = epWorldSize / 8 * align4096(MAXBS_TOKEN_SIZE) * 1B
COMBINE_TOKENFLAG_SIZE = align32((maxBs + (aivNum / (epWorldSize / 8) + 1)) * sizeof(uint64_t)) *1B

# WindowIn
|           |                                              | Start Addr                                   | Size                                      | Function                                                      |
|-----------|----------------------------------------------|----------------------------------------------|-------------------------------------------|---------------------------------------------------------------|
| Ping RDMA | Arrived Flag                                 | 0MB                                          | A1                                        | GetLocalRecvBuffFlagAddr, GetRemoteRecvBuffFlagAddr           |
| Ping RDMA | Inner Flag                                   | A1                                           | A1                                        | GetLocalRecvBuffInnerFlagAddr, GetRemoteRecvBuffInnerFlagAddr |
| Ping RDMA | Inner Data                                   | 2 * A1                                       | 1MB - 2 * A1                              | GetLocalRecvBuffInnerDataAddr, GetRemoteRecvBuffInnerDataAddr |
| Ping RDMA | RDMA Data                                    | 1MB                                          | RDMA_DATA_SIZE                            | GetLocalRecvBuffDataAddr, GetRemoteRecvBuffDataAddr           |
| -         | -                                            | -                                            | -                                         | -                                                             |
| IPC Data  | localExp 0--(srcRankId 0~n/2-1)              | (W/2 - 2)MB - IPC_DATA_SIZE_PER_EXP / 2 * 3B | IPC_DATA_SIZE_PER_EXP / 2                 | GetIpcDataAddr                                                |
| IPC Data  | localExp x--(srcRankId 0~n/2-1)              | (W/2 - 2)MB - IPC_DATA_SIZE_PER_EXP / 2 * 2B | IPC_DATA_SIZE_PER_EXP / 2                 | GetIpcDataAddr                                                |
| IPC Data  | localExp n--(srcRankId 0~n/2-1)              | (W/2 - 2)MB - IPC_DATA_SIZE_PER_EXP / 2 * 1B | IPC_DATA_SIZE_PER_EXP / 2                 | GetIpcDataAddr                                                |
| IPC Flag  | Combine Sync Flag 1: GM2IPC                  | (W/2 - 2)MB                                  | 8 * 32B = 256B                            | GetLocalIpcSyncFlagAddr, GetRemoteIpcSyncFlagAddr             |
| IPC Flag  | Combine Sync Flag 2: SumToWindow--server 0-n | (W/2 - 2)MB + 288B                           | COMBINE_TOKENFLAG_SIZE * epWorldSize / 8B | GetIpcTokenFlagAddr                                           |
| IPC Flag  | Dispatch Sync flag                           | (W/2 - 1)MB                                  | 8 * 32B = 256B                            | GetLocalIpcSyncFlagAddr, GetRemoteIpcSyncFlagAddr             |
| IPC Flag  | Dispatch Magic Value                         | W/2MB - 256 * 32B                            | aivNum * 32B                              | UpdateAndGetMagicValue                                        |
| IPC Flag  | Combine Magic Value                          | W/2MB - 128 * 32B                            | aivNum * 32B                              | UpdateAndGetMagicValue                                        |
| Pong RDMA | Arrived Flag                                 | W/2MB                                        | A1                                        | GetLocalRecvBuffFlagAddr, GetRemoteRecvBuffFlagAddr           |
| Pong RDMA | Inner Flag                                   | W/2MB + A1                                   | A1                                        | GetLocalRecvBuffInnerFlagAddr, GetRemoteRecvBuffInnerFlagAddr |
| Pong RDMA | Inner Data                                   | W/2MB + 2 * A1                               | 1MB - 2 * A1                              | GetLocalRecvBuffInnerDataAddr, GetRemoteRecvBuffInnerDataAddr |
| Pong RDMA | RDMA Data                                    | (W/2 + 1)MB                                  | RDMA_DATA_SIZE                            | GetLocalRecvBuffDataAddr, GetRemoteRecvBuffDataAddr           |
|  -        | -                                            | -                                            | -                                         | -                                                             |
| IPC Data  | localExp 0--(srcRankId n/2~n-1)              | (W - 2)MB - IPC_DATA_SIZE_PER_EXP / 2 * 3B   | IPC_DATA_SIZE_PER_EXP / 2                 | GetIpcDataAddr                                                |
| IPC Data  | localExp x--(srcRankId n/2~n-1)              | (W - 2)MB - IPC_DATA_SIZE_PER_EXP / 2 * 2B   | IPC_DATA_SIZE_PER_EXP / 2                 | GetIpcDataAddr                                                |
| IPC Data  | localExp n--(srcRankId n/2~n-1)              | (W - 2)MB - IPC_DATA_SIZE_PER_EXP / 2 * 1B   | IPC_DATA_SIZE_PER_EXP / 2                 | GetIpcDataAddr                                                |
| IPC Flag  | TokenCnt                                     | (W - 2)MB                                    | moeExpertNum * 32B                        | GetLocalIpcTokenCntAddr, GetRemoteIpcTokenCntAddr             |
# WindowOut
|           |                                              | Start Addr                                   | Size                                      | Function                      |
|-----------|----------------------------------------------|----------------------------------------------|-------------------------------------------|-------------------------------|
| Ping RDMA | Flag                                         | 0MB                                          | 32B                                       | GetLocalSendBuffFlagAddr      |
| Ping RDMA | Inner Data                                   | 2 * A1                                       | 1MB - 2 * A1                              | GetLocalSendBuffInnerDataAddr |
| Ping RDMA | RDMA Data                                    | 1MB                                          | RDMA_DATA_SIZE                            | GetLocalSendBuffDataAddr      |
| -         | -                                            | -                                            | -                                         | -                             |
| Ping RDMA | Flag                                         | 0MB                                          | 32B                                       | GetLocalSendBuffFlagAddr      |
| Pong RDMA | Inner Data                                   | W/2MB + 2 * A1                               | 1M - 2 * A1                               | GetLocalSendBuffInnerDataAddr |
| Pong RDMA | RDMA Data                                    | (W/2 + 1)MB                                  | RDMA_DATA_SIZE                            | GetLocalSendBuffDataAddr      |
| -         | BufferId                                     | W MB - 32B                                   | 32B                                       | UpdateBufferId                |
## WindowOut-Combine--RDMAData
|           |                                              | Start Addr                                   | Size                                      | Function                 |
|-----------|----------------------------------------------|----------------------------------------------|-------------------------------------------|--------------------------|
| RDMA      | RDMA Data-DstServer 0                        | (1 or (W/2 + 1))MB                           | A2 = RDMA_DATA_SIZE / (epWorldSize / 8)   | GetLocalSendBuffDataAddr |
| RDMA      | RDMA Data-DstServer x                        | (1 or (W/2 + 1))MB + A2 * x                  | A2                                        | GetLocalSendBuffDataAddr |
| RDMA      | RDMA Data-DstServer n-1                      | (1 or (W/2 + 1))MB + A2 * (n - 1)            | A2                                        | GetLocalSendBuffDataAddr |
*/
template <typename XType>
class MoeDistributeA2AddrInfo {
protected:
    constexpr static uint32_t BUFFER_NUM = 2U;                     // 多buf
    constexpr static uint64_t STATE_OFFSET = 512UL;                // 状态空间偏移地址
    constexpr static uint64_t STATUS_SIZE_LAYERED = 1024 * 1024UL; // 1M
    constexpr static uint64_t RDMA_BUFFER_ALIGN = 4 * 1024UL;
    constexpr static uint32_t SERVER_RANK_SIZE = 8;
    constexpr static uint32_t UB_32B_ALIGN = 32U;
    constexpr static uint32_t B32_PER_BLOCK = UB_32B_ALIGN / sizeof(int32_t); // 8
    constexpr static uint32_t EXTRA_TOKEN_INFO_NUM = 4U; // 专家信息 权重信息 量化Scale 到达标志位
    constexpr static uint64_t IPC_DISPATCH_MAGIC_OFFSET = 2 * 1024 * 1024UL - 256 * 32UL;
    constexpr static uint64_t IPC_COMBINE_MAGIC_OFFSET = 2 * 1024 * 1024UL - 128 * 32UL;
    constexpr static uint64_t IPC_DISPATCH_FLAG_OFFSET = 1 * 1024 * 1024UL;
    constexpr static uint64_t IPC_COMBINE_FLAG_OFFSET = 0UL;
    constexpr static uint64_t IPC_TOKEN_CNT_OFFSET = 0UL;
    constexpr static uint64_t IPC_NON_DATA_BYTES = 4 * 1024 * 1024UL;
    constexpr static uint64_t IPC_BUFF_ALIGN = 512UL;

public:
    __aicore__ inline void Init(uint32_t rankId, uint32_t maxBs, uint32_t worldSize, uint32_t axisH, uint32_t axisK, uint32_t localMoeExpertNum, uint32_t aivNum)
    {
        curRankId_ = rankId;
        // Get Hccl Buffer Size
        auto hcclContext = AscendC::GetHcclContext<AscendC::HCCL_GROUP_ID_0>();
        context_.Init(hcclContext);
        auto winSize = ((__gm__ HcclA2CombineOpParam *)hcclContext)->winSize;
        // Get BufferId
        bufferChosenGlobal_.SetGlobalBuffer((__gm__ uint32_t *)(context_.GetWindowsOutAddr(curRankId_) + winSize - UB_32B_ALIGN));
        bufferId_ = bufferChosenGlobal_(0);
        aivId_ = AscendC::GetBlockIdx();
        localMoeExpertNum_ = localMoeExpertNum;
        worldSize_ = worldSize;
        serverNum_ = worldSize / SERVER_RANK_SIZE;
        halfWorldSize_ = worldSize / 2U;
        uint64_t maxTokenStructBytes =
            axisH * sizeof(XType) + EXTRA_TOKEN_INFO_NUM * RoundUp(axisK, B32_PER_BLOCK) * sizeof(uint32_t);
        serverSizeOnRdmaData_ = RoundUp(maxBs * maxTokenStructBytes, RDMA_BUFFER_ALIGN);
        rankSizeOnIpcData_ = RoundUp(maxBs * maxTokenStructBytes, IPC_BUFF_ALIGN);
        // rdma addr
        rdmaFlagAddrStart_ = (bufferId_ & 0x1) ? (winSize / 2UL) : 0UL;
        rdmaDataAddrStart_ = rdmaFlagAddrStart_ + STATUS_SIZE_LAYERED;

        // ipc addr
        ipcFlagAddrStart_[0] = winSize / 2UL - IPC_NON_DATA_BYTES / 2UL;
        ipcFlagAddrStart_[1] = winSize - IPC_NON_DATA_BYTES / 2UL;
        ipcDataAddrStart_[0] = (ipcFlagAddrStart_[0] - rankSizeOnIpcData_ * localMoeExpertNum_ * halfWorldSize_) /
                               IPC_BUFF_ALIGN * IPC_BUFF_ALIGN;
        ipcDataAddrStart_[1] = (ipcFlagAddrStart_[1] - rankSizeOnIpcData_ * localMoeExpertNum_ * halfWorldSize_) /
                               IPC_BUFF_ALIGN * IPC_BUFF_ALIGN;
        for (int i = 0; i < SERVER_RANK_SIZE; i++) {
            uint32_t targetRank = curRankId_ / SERVER_RANK_SIZE * SERVER_RANK_SIZE + i;
            shareAddrs[i] = context_.GetWindowsInAddr(targetRank);
        }
        localWindowInGM_ = context_.GetWindowsInAddr(curRankId_);

    }
    __aicore__ inline void UpdateBufferId()
    {
        bufferChosenGlobal_(0) = bufferId_ ^ 1;
        AscendC::DataCacheCleanAndInvalid<uint32_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
                                          AscendC::DcciDst::CACHELINE_OUT>(bufferChosenGlobal_);
        AscendC::PipeBarrier<PIPE_ALL>();
    }
    // ===== Sender =====
    __aicore__ inline GM_ADDR GetLocalSendBuffFlagAddr() const
    {
        return context_.GetWindowsOutAddr(curRankId_) + rdmaFlagAddrStart_;
    }

    __aicore__ inline GM_ADDR GetRemoteRecvBuffFlagAddr(uint32_t dstRankId) const
    {
        return context_.GetWindowsInAddr(dstRankId) + rdmaFlagAddrStart_ + curRankId_ / SERVER_RANK_SIZE * STATE_OFFSET;
    }

    __aicore__ inline GM_ADDR GetRemoteRecvBuffDataAddr(uint32_t dstRankId) const
    {
        return context_.GetWindowsInAddr(dstRankId) + rdmaDataAddrStart_ + curRankId_ / SERVER_RANK_SIZE * serverSizeOnRdmaData_;
    }

    // ===== Receiver =====
    __aicore__ inline GM_ADDR GetLocalRecvBuffFlagAddr(uint32_t srcRankId) const
    {
        return context_.GetWindowsInAddr(curRankId_) + rdmaFlagAddrStart_ + srcRankId / SERVER_RANK_SIZE * STATE_OFFSET;
    }

    __aicore__ inline GM_ADDR GetLocalRecvBuffDataAddr(uint32_t srcRankId) const
    {
        return localWindowInGM_ + rdmaDataAddrStart_ + srcRankId / SERVER_RANK_SIZE * serverSizeOnRdmaData_;
    }

    // ===== Sender And Receiver =====
    __aicore__ inline GM_ADDR GetIpcDataAddr(uint32_t targetRankId, uint32_t localMoeExpertId,
                                               uint32_t srcRankId) const
    {
        return shareAddrs[targetRankId % SERVER_RANK_SIZE] + ipcDataAddrStart_[srcRankId / halfWorldSize_] +
               (localMoeExpertId * halfWorldSize_ + (srcRankId % halfWorldSize_)) * rankSizeOnIpcData_;
    }

protected:
    __aicore__ inline uint64_t UpdateAndGetMagicValue(AscendC::LocalTensor<uint64_t> tempLocal, GM_ADDR magicAddrStart)
    {
        AscendC::GlobalTensor<uint64_t> magicGt;
        magicGt.SetGlobalBuffer((__gm__ uint64_t *)(magicAddrStart));
        AscendC::DataCopy(tempLocal, magicGt, UB_32B_ALIGN / sizeof(uint64_t));
        AscendC::SyncFunc<AscendC::HardEvent::MTE2_S>();
        tempLocal(0) += 1UL;
        AscendC::SyncFunc<AscendC::HardEvent::S_MTE3>();
        AscendC::DataCopy(magicGt, tempLocal, UB_32B_ALIGN / sizeof(uint64_t));
        AscendC::PipeBarrier<PIPE_ALL>();
        return tempLocal(0);
    }

protected:
    AscendC::GlobalTensor<uint32_t> bufferChosenGlobal_;
    uint32_t curRankId_{0U};
    uint32_t aivId_{0};
    uint32_t bufferId_{0U};
    uint32_t serverNum_{0U};
    uint32_t worldSize_{0U};
    uint32_t halfWorldSize_{0U};
    uint32_t localMoeExpertNum_{0U};
    uint64_t serverSizeOnRdmaData_{0UL};
    uint64_t rankSizeOnIpcData_{0UL};

    uint64_t ipcFlagAddrStart_[2]{0UL};
    uint64_t ipcDataAddrStart_[2]{0UL};
    uint64_t rdmaFlagAddrStart_{0UL};
    uint64_t rdmaDataAddrStart_{0UL};

    GM_ADDR shareAddrs[8];
    GM_ADDR localWindowInGM_;

    MoeDistributeA2Context context_;
};

template <typename XType>
class MoeDistributeA2DispatchAddrInfo : public MoeDistributeA2AddrInfo<XType> {

using BaseClass = MoeDistributeA2AddrInfo<XType>;
protected:
    __aicore__ inline void InitInnerAddr()
    {
        auto tokenFlagBytes = BaseClass::STATE_OFFSET * (BaseClass::serverNum_ + 1);
        auto innerTableFlagTotalBytes = BaseClass::STATE_OFFSET * (BaseClass::serverNum_ + 1);
        auto innerTableDataTotalBytes = BaseClass::STATUS_SIZE_LAYERED - tokenFlagBytes - innerTableFlagTotalBytes;
        innerTableSize_ = innerTableDataTotalBytes / BaseClass::serverNum_ / BaseClass::UB_32B_ALIGN * BaseClass::UB_32B_ALIGN;
        rdmaInnerFlagAddrStart_ = BaseClass::rdmaFlagAddrStart_ + tokenFlagBytes;
        rdmaInnerDataAddrStart_ = rdmaInnerFlagAddrStart_ + innerTableFlagTotalBytes;
    }

public:
    __aicore__ inline void Init(uint32_t rankId, uint32_t maxBs, uint32_t worldSize, uint32_t axisH, uint32_t axisK, uint32_t localMoeExpertNum, uint32_t aivNum)
    {
        BaseClass::Init(rankId, maxBs, worldSize, axisH, axisK, localMoeExpertNum, aivNum);
        InitInnerAddr();
        magicAddrStart_ = BaseClass::shareAddrs[BaseClass::curRankId_ % BaseClass::SERVER_RANK_SIZE] + BaseClass::ipcFlagAddrStart_[0] +
            BaseClass::IPC_DISPATCH_MAGIC_OFFSET + BaseClass::aivId_ * BaseClass::UB_32B_ALIGN;
        ipcSyncFlagAddrStart_ = BaseClass::ipcFlagAddrStart_[0] + BaseClass::IPC_DISPATCH_FLAG_OFFSET;
        ipcTokenCntAddrStart_ = BaseClass::ipcFlagAddrStart_[1] + BaseClass::IPC_TOKEN_CNT_OFFSET;
    }

    __aicore__ inline uint64_t UpdateAndGetMagicValue(AscendC::LocalTensor<uint64_t> tempLocal)
    {
        return BaseClass::UpdateAndGetMagicValue(tempLocal, magicAddrStart_);
    }

    // ===== Sender =====
    __aicore__ inline GM_ADDR GetLocalSendBuffDataAddr() const
    {
        return BaseClass::context_.GetWindowsOutAddr(BaseClass::curRankId_) + BaseClass::rdmaDataAddrStart_;
    }
    __aicore__ inline GM_ADDR GetLocalSendBuffInnerDataAddr(uint32_t dstServerId) const
    {
        return BaseClass::context_.GetWindowsOutAddr(BaseClass::curRankId_) + rdmaInnerDataAddrStart_ + dstServerId * innerTableSize_;
    }

    __aicore__ inline GM_ADDR GetRemoteRecvBuffInnerFlagAddr(uint32_t dstServerId) const
    {
        return BaseClass::context_.GetWindowsInAddr(dstServerId * BaseClass::SERVER_RANK_SIZE + BaseClass::curRankId_ % BaseClass::SERVER_RANK_SIZE) +
            rdmaInnerFlagAddrStart_ + BaseClass::curRankId_ / BaseClass::SERVER_RANK_SIZE * BaseClass::STATE_OFFSET;
    }

    __aicore__ inline GM_ADDR GetRemoteRecvBuffInnerDataAddr(uint32_t dstServerId) const
    {
        return BaseClass::context_.GetWindowsInAddr(dstServerId * BaseClass::SERVER_RANK_SIZE + BaseClass::curRankId_ % BaseClass::SERVER_RANK_SIZE) +
            rdmaInnerDataAddrStart_ + BaseClass::curRankId_ / BaseClass::SERVER_RANK_SIZE * innerTableSize_;
    }

    __aicore__ inline GM_ADDR GetRemoteIpcSyncFlagAddr(uint32_t targetRankId) const
    {
        return BaseClass::shareAddrs[targetRankId % BaseClass::SERVER_RANK_SIZE] + ipcSyncFlagAddrStart_ +
               (BaseClass::curRankId_ % BaseClass::SERVER_RANK_SIZE) * BaseClass::UB_32B_ALIGN;
    }

    __aicore__ inline GM_ADDR GetRemoteIpcTokenCntAddr(uint32_t targetRankId, uint32_t targetExpId, uint32_t srcRankId) const
    {
        return BaseClass::shareAddrs[targetRankId % BaseClass::SERVER_RANK_SIZE] + ipcTokenCntAddrStart_ +
               ((targetExpId % BaseClass::localMoeExpertNum_) * BaseClass::worldSize_ + srcRankId) * BaseClass::UB_32B_ALIGN;
    }

    // ===== Receiver =====
    __aicore__ inline GM_ADDR GetLocalRecvBuffInnerFlagAddr(uint32_t srcServerId) const
    {
        return BaseClass::context_.GetWindowsInAddr(BaseClass::curRankId_) + rdmaInnerFlagAddrStart_ + srcServerId * BaseClass::STATE_OFFSET;
    }

    __aicore__ inline GM_ADDR GetLocalRecvBuffInnerDataAddr(uint32_t srcServerId) const
    {
        return BaseClass::context_.GetWindowsInAddr(BaseClass::curRankId_) + rdmaInnerDataAddrStart_ + srcServerId * innerTableSize_;
    }

    __aicore__ inline GM_ADDR GetLocalIpcSyncFlagAddr(uint32_t srcRankId) const
    {
        return BaseClass::shareAddrs[BaseClass::curRankId_ % BaseClass::SERVER_RANK_SIZE] + ipcSyncFlagAddrStart_ +
               (srcRankId % BaseClass::SERVER_RANK_SIZE) * BaseClass::UB_32B_ALIGN;
    }

    __aicore__ inline GM_ADDR GetLocalIpcTokenCntAddr() const
    {
        return BaseClass::shareAddrs[BaseClass::curRankId_ % BaseClass::SERVER_RANK_SIZE] + ipcTokenCntAddrStart_;
    }

private:
    GM_ADDR magicAddrStart_{nullptr};
    uint64_t ipcSyncFlagAddrStart_{0UL};
    uint64_t ipcTokenCntAddrStart_{0UL};
    uint64_t rdmaInnerFlagAddrStart_{0UL};
    uint64_t rdmaInnerDataAddrStart_{0UL};
    uint64_t innerTableSize_{0UL};
};

template <typename XType>
class MoeDistributeA2CombineAddrInfo : public MoeDistributeA2AddrInfo<XType> {

using BaseClass = MoeDistributeA2AddrInfo<XType>;
public:
    __aicore__ inline void Init(uint32_t rankId, uint32_t maxBs, uint32_t worldSize, uint32_t axisH, uint32_t axisK, uint32_t localMoeExpertNum, uint32_t aivNum)
    {
        MoeDistributeA2AddrInfo<XType>::Init(rankId, maxBs, worldSize, axisH, axisK, localMoeExpertNum, aivNum);
        magicAddrStart_ = BaseClass::shareAddrs[BaseClass::curRankId_ % BaseClass::SERVER_RANK_SIZE] + BaseClass::ipcFlagAddrStart_[0] +
            BaseClass::IPC_COMBINE_MAGIC_OFFSET + BaseClass::aivId_ * BaseClass::UB_32B_ALIGN;
        ipcSyncFlagAddrStart_ = BaseClass::ipcFlagAddrStart_[0] + BaseClass::IPC_COMBINE_FLAG_OFFSET;
        shareFlagSize_ =
            RoundUp(static_cast<uint32_t>((maxBs + aivNum / BaseClass::serverNum_ + 1U) * sizeof(uint64_t)), BaseClass::UB_32B_ALIGN);
        shareFlagAddrStart_ = ipcSyncFlagAddrStart_ + (BaseClass::SERVER_RANK_SIZE + 1) * BaseClass::UB_32B_ALIGN;
    }

    __aicore__ inline uint64_t UpdateAndGetMagicValue(AscendC::LocalTensor<uint64_t> tempLocal)
    {
        return BaseClass::UpdateAndGetMagicValue(tempLocal, magicAddrStart_);
    }

    // ===== Sender =====
    __aicore__ inline GM_ADDR GetLocalSendBuffDataAddr(uint32_t dstRankId) const
    {
        return BaseClass::context_.GetWindowsOutAddr(BaseClass::curRankId_) + BaseClass::rdmaDataAddrStart_ + dstRankId / BaseClass::SERVER_RANK_SIZE * BaseClass::serverSizeOnRdmaData_;
    }

    __aicore__ inline GM_ADDR GetRemoteIpcSyncFlagAddr(uint32_t targetRankId) const
    {
        return BaseClass::shareAddrs[targetRankId % BaseClass::SERVER_RANK_SIZE] + ipcSyncFlagAddrStart_ +
               (BaseClass::curRankId_ % BaseClass::SERVER_RANK_SIZE) * BaseClass::UB_32B_ALIGN;
    }

    // ===== Receiver =====
    __aicore__ inline GM_ADDR GetLocalIpcSyncFlagAddr(uint32_t srcRankId) const
    {
        return BaseClass::shareAddrs[BaseClass::curRankId_ % BaseClass::SERVER_RANK_SIZE] + ipcSyncFlagAddrStart_ +
               (srcRankId % BaseClass::SERVER_RANK_SIZE) * BaseClass::UB_32B_ALIGN;
    }


    // ===== Sender And Receiver =====
    __aicore__ inline GM_ADDR GetIpcTokenFlagAddr(uint32_t serverId) const
    {
        return BaseClass::shareAddrs[BaseClass::curRankId_ % BaseClass::SERVER_RANK_SIZE] + shareFlagAddrStart_ +
               serverId * shareFlagSize_;
    }

private:
    GM_ADDR magicAddrStart_{nullptr};
    uint64_t ipcSyncFlagAddrStart_{0UL};
    uint64_t shareFlagSize_{0UL};
    uint64_t shareFlagAddrStart_{0UL};
};

} // namespace MoeDistributeA2Base
#endif // MOE_DISTRIBUTE_A2_BASE_H
