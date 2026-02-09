
#ifndef MOE_DISTRIBUTE_A2_BASE_H
#define MOE_DISTRIBUTE_A2_BASE_H
namespace MoeDistributeA2Base {
struct GetAddrInfo {
    uint64_t rdmaAddrOffset{0UL};
    uint64_t ipcAddrOffset[2]{0UL};
    uint64_t ipcFlagOffset{0UL};
    constexpr static uint32_t IPC_BUFF_ALIGN{512};
    /* WinSize结构如下：
       | Ping RDMA                                 |-| Half IPC                             |-| Pong RDMA                                               |-| Rest IPC                                              |
       | RDMA Flag: 1MB | RDMA Data: rdmaDataSizeB |-| IPC DATA, FromRankId < worldSize / 2 |-| IPC Flag 2M | RDMA Flag: 1MB | RDMA Data: rdmaDataSizeB |-| IPC DATA, FromRankId >= worldSize / 2 |-| IPC Flag 2M |
       Dispatch IPC Flag 结构如下：
       | IPC Flag 2M                                                                         |-| IPC Flag 2M         |
       | Unused | IPC FLAG, Start: 1M, Size: 256B | Unused | Magic, Start: 2M - 4K, Size: 4K |-| Token Cnt, Size: 2M |
    */
    __aicore__ inline void Init(uint64_t winSize, uint64_t bufferId, uint64_t ipcDataSize, uint64_t ipcFlagSize)
    {
        ipcAddrOffset[0] = winSize / 2UL - (ipcDataSize + ipcFlagSize) / 2UL;
        ipcAddrOffset[1] = winSize - (ipcDataSize + ipcFlagSize) / 2UL;
        ipcAddrOffset[0] = (ipcAddrOffset[0] + IPC_BUFF_ALIGN - 1) / IPC_BUFF_ALIGN * IPC_BUFF_ALIGN;
        ipcAddrOffset[1] = (ipcAddrOffset[1] + IPC_BUFF_ALIGN - 1) / IPC_BUFF_ALIGN * IPC_BUFF_ALIGN;
        ipcFlagOffset = winSize / 2UL - ipcFlagSize / 2 - ipcAddrOffset[0];
        if (bufferId == 1UL) {
            rdmaAddrOffset = winSize / 2UL;
        }
    }
    __aicore__ inline uint64_t GetRdmaAddrOffset(uint64_t originAddrOffset)
    {
        return rdmaAddrOffset + originAddrOffset;
    }
    __aicore__ inline uint64_t GetIpcAddrOffset(uint64_t originAddrOffset, uint32_t index) {
        return ipcAddrOffset[index] + originAddrOffset;
    }
};
}
#endif
