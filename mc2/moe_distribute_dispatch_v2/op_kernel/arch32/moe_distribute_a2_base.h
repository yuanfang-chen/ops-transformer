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
|           |                                              | Start Addr                                   | Size                                      | Function                      |
|-----------|----------------------------------------------|----------------------------------------------|-------------------------------------------|-------------------------------|
| Ping RDMA | Arrived Flag                                 | 0MB                                          | A1                                        | GetRdmaFlagAddrIn             |
| Ping RDMA | Inner Flag                                   | A1                                           | A1                                        | GetInnerFlagAddrIn            |
| Ping RDMA | Inner Data                                   | 2 * A1                                       | 1MB - 2 * A1                              | GetInnerDataAddrIn            |
| Ping RDMA | RDMA Data                                    | 1MB                                          | RDMA_DATA_SIZE                            | GetRdmaDataAddrIn             |
| -         | -                                            | -                                            | -                                         | -                             |
| IPC Data  | localExp 0--(srcRankId 0~n/2-1)              | (W/2 - 2)MB - IPC_DATA_SIZE_PER_EXP / 2 * 3B | IPC_DATA_SIZE_PER_EXP / 2                 | GetIpcDataAddrIn              |
| IPC Data  | localExp x--(srcRankId 0~n/2-1)              | (W/2 - 2)MB - IPC_DATA_SIZE_PER_EXP / 2 * 2B | IPC_DATA_SIZE_PER_EXP / 2                 | GetIpcDataAddrIn              |
| IPC Data  | localExp n--(srcRankId 0~n/2-1)              | (W/2 - 2)MB - IPC_DATA_SIZE_PER_EXP / 2 * 1B | IPC_DATA_SIZE_PER_EXP / 2                 | GetIpcDataAddrIn              |
| IPC Flag  | Combine Sync Flag 1: GM2IPC                  | (W/2 - 2)MB                                  | 8 * 32B = 256B                            | GetIpcSyncFlagAddrForCombine  |
| IPC Flag  | Combine Sync Flag 2: SumToWindow--server 0-n | (w/2 - 2)MB + 288B                           | COMBINE_TOKENFLAG_SIZE * epWorldSize / 8B | GetIpcTokenFlagAddr           |
| IPC Flag  | Dispatch Sync flag                           | (w/2 - 1)MB                                  | 8 * 32B = 256B                            | GetIpcSyncFlagAddrForDispatch |
| IPC Flag  | Dispatch Magic Value                         | W/2MB - 256 * 32B                            | aivNum * 32B                              | GetMagicValue                 |
| IPC Flag  | Combine Magic Value                          | W/2MB - 128 * 32B                            | aivNum * 32B                              | GetMagicValue                 |
| Pong RDMA | Arrived Flag                                 | W/2MB                                        | A1                                        | GetRdmaFlagAddrIn             |
| Pong RDMA | Inner Flag                                   | W/2MB + A1                                   | A1                                        | GetInnerFlagAddrIn            |
| Pong RDMA | Inner Data                                   | W/2MB + 2 * A1                               | 1M - 2 * A1                               | GetInnerDataAddrIn            |
| Pong RDMA | RDMA Data                                    | (W/2 + 1)MB                                  | RDMA_DATA_SIZE                            | GetRdmaDataAddrIn             |
|  -        | -                                            | -                                            | -                                         | -                             |
| IPC Data  | localExp 0--(srcRankId n/2~n-1)              | (W - 2)MB - IPC_DATA_SIZE_PER_EXP / 2 * 3B   | IPC_DATA_SIZE_PER_EXP / 2                 | GetIpcDataAddrIn              |
| IPC Data  | localExp x--(srcRankId n/2~n-1)              | (W - 2)MB - IPC_DATA_SIZE_PER_EXP / 2 * 2B   | IPC_DATA_SIZE_PER_EXP / 2                 | GetIpcDataAddrIn              |
| IPC Data  | localExp n--(srcRankId n/2~n-1)              | (W - 2)MB - IPC_DATA_SIZE_PER_EXP / 2 * 1B   | IPC_DATA_SIZE_PER_EXP / 2                 | GetIpcDataAddrIn              |
| IPC Flag  | TokenCnt                                     | (W - 2)MB                                    | moeExpertNum * 32B                        | GetIpcTokenCntAddr            |
# WindowOut
|           |                                              | Start Addr                                   | Size                                      | Function                      |
|-----------|----------------------------------------------|----------------------------------------------|-------------------------------------------|-------------------------------|
| Ping RDMA | Flag                                         | 0MB                                          | 32B                                       | GetRdmaFlagAddrOut            |
| Ping RDMA | Inner Data                                   | 2 * A1                                       | 1MB - 2 * A1                              | GetInnerDataAddrOut           |
| Ping RDMA | RDMA Data                                    | 1MB                                          | RDMA_DATA_SIZE                            | GetRdmaDataAddrOutForX        |
| -         | -                                            | -                                            | -                                         | -                             |
| Ping RDMA | Flag                                         | 0MB                                          | 32B                                       | GetRdmaFlagAddrOut            |
| Pong RDMA | Inner Data                                   | W/2MB + 2 * A1                               | 1M - 2 * A1                               | GetInnerDataAddrOut           |
| Pong RDMA | RDMA Data                                    | (W/2 + 1)MB                                  | RDMA_DATA_SIZE                            | GetRdmaDataAddrOutForX        |
| -         | BufferId                                     | W MB - 32B                                   | 32B                                       | UpdateBufferId                |
## WindowOut-Combine--RDMAData
|           |                                              | Start Addr                                   | Size                                      | Function                      |
|-----------|----------------------------------------------|----------------------------------------------|-------------------------------------------|-------------------------------|
| RDMA      | RDMA Data-DstServer 0                        | (1 or (W/2 + 1))MB                           | A2 = RDMA_DATA_SIZE / (epWorldSize / 8)   | GetRdmaDataAddrOutForCombine  |
| RDMA      | RDMA Data-DstServer x                        | (1 or (W/2 + 1))MB + A2 * x                  | A2                                        | GetRdmaDataAddrOutForCombine  |
| RDMA      | RDMA Data-DstServer n-1                      | (1 or (W/2 + 1))MB + A2 * (n - 1)            | A2                                        | GetRdmaDataAddrOutForCombine  |
*/
template <typename XType>
class MoeDistributeA2AddrInfo {
private:
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
    constexpr static uint64_t IPC_NON_DATA_BYTES = 4 * 1024 * 1024UL;
    constexpr static uint64_t IPC_BUFF_ALIGN = 512UL;

    __aicore__ inline GM_ADDR GetWindowsInAddr(uint32_t rankId) const
    {
        if (((__gm__ HcclA2CombineOpParam *)hcclContext_)->multiFlag == 0U) {
            return (GM_ADDR)(((__gm__ HcclA2CombineOpParam *)hcclContext_)->windowsIn[rankId]);
        } else {
            if (rankId == curRankId_) {
                return (GM_ADDR)(((__gm__ HcclA2CombineOpParam *)hcclContext_)->data[rankId].localInput.addr);
            } else {
                return (GM_ADDR)(((__gm__ HcclA2CombineOpParam *)hcclContext_)->data[rankId].remoteInput.addr);
            }
        }
    }

    __aicore__ inline GM_ADDR GetWindowsOutAddr(uint32_t rankId) const
    {
        if (((__gm__ HcclA2CombineOpParam *)hcclContext_)->multiFlag == 0U) {
            return (GM_ADDR)(((__gm__ HcclA2CombineOpParam *)hcclContext_)->windowsOut[rankId]);
        } else {
            if (rankId == curRankId_) {
                return (GM_ADDR)(((__gm__ HcclA2CombineOpParam *)hcclContext_)->data[rankId].localOutput.addr);
            } else {
                return (GM_ADDR)(((__gm__ HcclA2CombineOpParam *)hcclContext_)->data[rankId].remoteOutput.addr);
            }
        }
    }

    template <typename T>
    inline __aicore__ T RoundUp(const T val, const T align)
    {
        static_assert(std::is_arithmetic<T>::value, "T must be an arithmetic type");
        if (align == 0 || val + align - 1 < val) {
            return val;
        }
        return (val + align - 1) / align * align;
    }

    __aicore__ inline void InitInnerAddr()
    {
        auto tokenFlagBytes = STATE_OFFSET * (serverNum_ + 1);
        auto innerTableFlagTotalBytes = STATE_OFFSET * (serverNum_ + 1);
        auto innerTableDataTotalBytes = STATUS_SIZE_LAYERED - tokenFlagBytes - innerTableFlagTotalBytes;
        innerTableSize_ = innerTableDataTotalBytes / serverNum_ / UB_32B_ALIGN * UB_32B_ALIGN;
        rdmaInnerFlagAddrStart_ = rdmaFlagAddrStart_ + tokenFlagBytes;
        rdmaInnerDataAddrStart_ = rdmaInnerFlagAddrStart_ + innerTableFlagTotalBytes;
    }

    __aicore__ inline void InitIpcFlagAddr()
    {
        ipcCombineSyncFlagAddrStart_ = ipcFlagAddrStart_[0];
        ipcDispatchSyncFlagAddrStart_ = ipcFlagAddrStart_[0] + IPC_DISPATCH_FLAG_OFFSET;
        ipcCombineMagicAddrStart_ = ipcFlagAddrStart_[0] + IPC_COMBINE_MAGIC_OFFSET;
        ipcDispatchMagicAddrStart_ = ipcFlagAddrStart_[0] + IPC_DISPATCH_MAGIC_OFFSET;
        ipcDispatchTokenCntAddrStart_ = ipcFlagAddrStart_[1];
    }

    __aicore__ inline GM_ADDR GetIpcMagicAddrForDispatch() const
    {
        return shareAddrs[curRankId_ % SERVER_RANK_SIZE] + ipcDispatchMagicAddrStart_ + aivId_ * UB_32B_ALIGN;
    }

    __aicore__ inline GM_ADDR GetIpcMagicAddrForCombine() const
    {
        return shareAddrs[curRankId_ % SERVER_RANK_SIZE] + ipcCombineMagicAddrStart_ + aivId_ * UB_32B_ALIGN;
    }

public:
    __aicore__ inline void Init(uint32_t rankId, uint32_t maxBs, uint32_t worldSize, uint32_t axisH, uint32_t axisK, uint32_t localMoeExpertNum, uint32_t aivNum)
    {
        curRankId_ = rankId;
        // Get Hccl Buffer Size
        hcclContext_ = AscendC::GetHcclContext<AscendC::HCCL_GROUP_ID_0>();
        auto winSize = ((__gm__ HcclA2CombineOpParam *)hcclContext_)->winSize;
        // Get BufferId
        bufferChosenGlobal_.SetGlobalBuffer((__gm__ uint32_t *)(GetWindowsOutAddr(rankId) + winSize - UB_32B_ALIGN));
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
        if (bufferId_ & 0x1) {
            rdmaFlagAddrStart_ = winSize / 2UL;
        }
        rdmaDataAddrStart_ = rdmaFlagAddrStart_ + STATUS_SIZE_LAYERED;
        InitInnerAddr();

        // ipc addr
        ipcFlagAddrStart_[0] = winSize / 2UL - IPC_NON_DATA_BYTES / 2UL;
        ipcFlagAddrStart_[1] = winSize - IPC_NON_DATA_BYTES / 2UL;
        ipcDataAddrStart_[0] = (ipcFlagAddrStart_[0] - rankSizeOnIpcData_ * localMoeExpertNum_ * halfWorldSize_) /
                               IPC_BUFF_ALIGN * IPC_BUFF_ALIGN;
        ipcDataAddrStart_[1] = (ipcFlagAddrStart_[1] - rankSizeOnIpcData_ * localMoeExpertNum_ * halfWorldSize_) /
                               IPC_BUFF_ALIGN * IPC_BUFF_ALIGN;
        InitIpcFlagAddr();
        for (int i = 0; i < SERVER_RANK_SIZE; i++) {
            uint32_t targetRank = curRankId_ / SERVER_RANK_SIZE * SERVER_RANK_SIZE + i;
            shareAddrs[i] = GetWindowsInAddr(targetRank);
        }
        windowInGM_ = GetWindowsInAddr(curRankId_);
        combineShareFlagSize_ =
            RoundUp(static_cast<uint32_t>((maxBs + aivNum / serverNum_ + 1U) * sizeof(uint64_t)), UB_32B_ALIGN);
        combineShareFlagAddrStart_ = ipcCombineSyncFlagAddrStart_ + (SERVER_RANK_SIZE + 1) * UB_32B_ALIGN;
    }
    __aicore__ inline void UpdateBufferId()
    {
        bufferChosenGlobal_(0) = bufferId_ ^ 1;
        AscendC::DataCacheCleanAndInvalid<uint32_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
                                          AscendC::DcciDst::CACHELINE_OUT>(bufferChosenGlobal_);
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline GM_ADDR GetRdmaFlagAddrIn(uint32_t targetRankId, uint32_t serverId) const
    {
        return GetWindowsInAddr(targetRankId) + rdmaFlagAddrStart_ + serverId * STATE_OFFSET;
    }

    __aicore__ inline GM_ADDR GetRdmaDataAddrIn(uint32_t targetRankId, uint32_t serverId) const
    {
        return GetWindowsInAddr(targetRankId) + rdmaDataAddrStart_ + serverId * serverSizeOnRdmaData_;
    }

    __aicore__ inline GM_ADDR GetRdmaFlagAddrOut() const
    {
        return GetWindowsOutAddr(curRankId_) + rdmaFlagAddrStart_;
    }

    __aicore__ inline GM_ADDR GetIpcDataAddrIn(uint32_t targetRankId, uint32_t localMoeExpertId,
                                               uint32_t fromRankId) const
    {
        return shareAddrs[targetRankId % SERVER_RANK_SIZE] + ipcDataAddrStart_[fromRankId / halfWorldSize_] +
               (localMoeExpertId * halfWorldSize_ + (fromRankId % halfWorldSize_)) * rankSizeOnIpcData_;
    }

    __aicore__ inline uint64_t GetMagicValue(AscendC::LocalTensor<uint64_t> tempLocal, bool isDispatch)
    {
        AscendC::GlobalTensor<uint64_t> magicGt;
        if (isDispatch) {
            magicGt.SetGlobalBuffer((__gm__ uint64_t *)(GetIpcMagicAddrForDispatch()));
        } else {
            magicGt.SetGlobalBuffer((__gm__ uint64_t *)(GetIpcMagicAddrForCombine()));
        }
        AscendC::DataCopy(tempLocal, magicGt, UB_32B_ALIGN / sizeof(uint64_t));
        AscendC::SyncFunc<AscendC::HardEvent::MTE2_S>();
        tempLocal(0) += 1UL;
        AscendC::SyncFunc<AscendC::HardEvent::S_MTE3>();
        AscendC::DataCopy(magicGt, tempLocal, UB_32B_ALIGN / sizeof(uint64_t));
        AscendC::PipeBarrier<PIPE_ALL>();
        return tempLocal(0);
    }

    // Combine专用
    __aicore__ inline GM_ADDR GetSelfRdmaDataAddrIn(uint32_t serverId) const
    {
        return windowInGM_ + rdmaDataAddrStart_ + serverId * serverSizeOnRdmaData_;
    }

    __aicore__ inline GM_ADDR GetIpcTokenFlagAddr(uint32_t serverId) const
    {
        return shareAddrs[curRankId_ % SERVER_RANK_SIZE] + combineShareFlagAddrStart_ +
               serverId * combineShareFlagSize_;
    }

    __aicore__ inline GM_ADDR GetRdmaDataAddrOutForCombine(uint32_t serverId) const
    {
        return GetWindowsOutAddr(curRankId_) + rdmaDataAddrStart_ + serverId * serverSizeOnRdmaData_;
    }

    __aicore__ inline GM_ADDR GetIpcSyncFlagAddrForCombine(uint32_t targetRankId, uint32_t fromRankId) const
    {
        return shareAddrs[targetRankId % SERVER_RANK_SIZE] + ipcCombineSyncFlagAddrStart_ +
               (fromRankId % SERVER_RANK_SIZE) * UB_32B_ALIGN;
    }

    // Dispatch专用
    __aicore__ inline GM_ADDR GetRdmaDataAddrOutForDispatch() const
    {
        return GetWindowsOutAddr(curRankId_) + rdmaDataAddrStart_;
    }

    __aicore__ inline GM_ADDR GetIpcSyncFlagAddrForDispatch(uint32_t targetRankId, uint32_t fromRankId) const
    {
        return shareAddrs[targetRankId % SERVER_RANK_SIZE] + ipcDispatchSyncFlagAddrStart_ +
               (fromRankId % SERVER_RANK_SIZE) * UB_32B_ALIGN;
    }

    __aicore__ inline GM_ADDR GetIpcTokenCntAddr(uint32_t targetRankId, uint32_t targetExpId, uint32_t fromRankId) const
    {
        return shareAddrs[targetRankId % SERVER_RANK_SIZE] + ipcDispatchTokenCntAddrStart_ +
               ((targetExpId % localMoeExpertNum_) * worldSize_ + fromRankId) * UB_32B_ALIGN;
    }

    __aicore__ inline GM_ADDR GetInnerFlagAddrIn(uint32_t targetRankId, uint32_t serverId) const
    {
        return GetWindowsInAddr(targetRankId) + rdmaInnerFlagAddrStart_ + serverId * STATE_OFFSET;
    }

    __aicore__ inline GM_ADDR GetInnerDataAddrIn(uint32_t targetRankId, uint32_t serverId) const
    {
        return GetWindowsInAddr(targetRankId) + rdmaInnerDataAddrStart_ + serverId * innerTableSize_;
    }

    __aicore__ inline GM_ADDR GetInnerDataAddrOut(uint32_t serverId) const
    {
        return GetWindowsOutAddr(curRankId_) + rdmaInnerDataAddrStart_ + serverId * innerTableSize_;
    }

private:
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
    uint64_t ipcCombineSyncFlagAddrStart_{0UL};
    uint64_t ipcDispatchSyncFlagAddrStart_{0UL};
    uint64_t ipcCombineMagicAddrStart_{0UL};
    uint64_t ipcDispatchMagicAddrStart_{0UL};
    uint64_t ipcDispatchTokenCntAddrStart_{0UL};
    uint64_t ipcDataAddrStart_[2]{0UL};
    uint64_t rdmaFlagAddrStart_{0UL};
    uint64_t rdmaDataAddrStart_{0UL};
    uint64_t rdmaInnerFlagAddrStart_{0UL};
    uint64_t rdmaInnerDataAddrStart_{0UL};
    uint64_t innerTableSize_{0UL};
    uint64_t combineShareFlagSize_{0UL};
    uint64_t combineShareFlagAddrStart_{0UL};
    GM_ADDR shareAddrs[8];
    GM_ADDR windowInGM_;
    __gm__ uint8_t *hcclContext_;
};
} // namespace MoeDistributeA2Base
#endif // MOE_DISTRIBUTE_A2_BASE_H
