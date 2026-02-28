#ifndef MC2_MOE_CONTEXT_H
#define MC2_MOE_CONTEXT_H

struct Mc2MoeContext {
    int32_t epRankId;
    uint64_t kfcContextAddr; // host kfc方案中，需要传递通信API所需的地址
    uint64_t epHcclBuffer_[1024];
};

#endif //MC2_MOE_CONTEXT_H