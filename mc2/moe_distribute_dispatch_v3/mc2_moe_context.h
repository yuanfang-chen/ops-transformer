#ifndef MC2_MOE_CONTEXT_H
#define MC2_MOE_CONTEXT_H

struct Mc2MoeContext {
    uint64_t epRankId;
    uint64_t kfcContextAddr; // host kfc方案中，需要传递通信API所需的地址
    uint64_t epHcclBuffer_[1024]; // 主要通过该数组传递每卡状态区首地址
};

#endif //MC2_MOE_CONTEXT_H