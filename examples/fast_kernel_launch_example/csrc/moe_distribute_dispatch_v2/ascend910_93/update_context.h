#ifndef UPDATE_CONTEXT_H
#define UPDATE_CONTEXT_H

struct Mc2ContextStru {
    uint64_t epRankId;
    uint64_t kfcContextAddr;
    uint64_t epHcclBuffer_[1024];
};

at::Tensor update_context(std::string group_ep, int64_t ep_world_size);

#endif