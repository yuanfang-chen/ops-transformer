
#ifndef MOE_DISTRIBUTE_A2_BASE_H
#define MOE_DISTRIBUTE_A2_BASE_H
namespace MoeDistributeA2Base {
struct GetAddrInfo {
    uint64_t rdmaAddrOffset{0UL};
    uint64_t ipcAddrOffset{0UL};
    __aicore__ inline void Init(uint64_t winSize, uint64_t bufferId, uint64_t ipcSize)
    {
        rdmaAddrOffset = (bufferId & 0x1) ? winSize / 2UL : 0UL;
        ipcAddrOffset = winSize / (2UL - bufferId) - ipcSize;
    }
    __aicore__ inline uint64_t GetRdmaAddrOffset(uint64_t originAddrOffset)
    {
        return rdmaAddrOffset + originAddrOffset;
    }
    __aicore__ inline uint64_t GetIpcAddrOffset(uint64_t originAddrOffset) {
        return ipcAddrOffset + originAddrOffset;
    }
};
}
#endif
