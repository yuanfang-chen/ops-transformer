#ifndef MC2_MOE_CONTEXT_H
#define MC2_MOE_CONTEXT_H

namespace Mc2Kernel {
constexpr uint32_t HCCL_HOST_KFC_MAX_RANK_NUM = 1024; //AIV
}

struct Mc2MoeContext {
    uint32_t rankId; // 当前卡rankId
    uint32_t rankDim; // 总卡数
    uint64_t winsize;
    uint64_t kfcContextPtr; // kfccontext 指针
    uint64_t windowsIn[Mc2Kernel::HCCL_HOST_KFC_MAX_RANK_NUM]; // ccu不使用, MTE 数据区
}
#endif
