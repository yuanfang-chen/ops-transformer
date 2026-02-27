#ifndef MC2_MOE_CONTEXT_H
#define MC2_MOE_CONTEXT_H
#include <stdint.h>

namespace Mc2Kernel {
constexpr uint32_t HCCL_HOST_KFC_MAX_RANK_NUM = 1024; //AIV
}
namespace Mc2Context {

    struct Mc2MoeContext {
        uint32_t epRankId; // 当前卡rankId
        uint32_t epRankSize; // 总卡数
        uint64_t winSize;
        uint64_t kfcContextPtr; // kfccontext 指针
        uint64_t epHcclBuffer[Mc2Kernel::HCCL_HOST_KFC_MAX_RANK_NUM]; // ccu不使用, MTE 数据区
    };

}
#endif
