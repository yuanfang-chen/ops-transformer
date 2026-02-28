// 使用Mc2Context取代hccl context
__gm__ Mc2MoeContext* mc2Context_{nullptr};
mc2Context_ = (__gm__ Mc2MoeContext *)mc2Context;

// 利用mc2Context_获取对应结构体内的内容
rankIdOriginal_ = mc2Context->epRankId;
rankId_ = mc2Context_->epRankId;
statusDataSpaceGm_ = (GM_ADDR)(mc2Context_->epHcclBuffer_[epRankIdHccl]);

__aicore__ inline GM_ADDR GetWindAddrByRankId(uint8_t ctxIdx, const int32_t rankId)
{
    uint64_t winDataSizeOffset = (ctxIdx == COMM_EP_IDX)? winDataSizeOffsetEp_ : winDataSizeOffsetTp_;
    return (GM_ADDR)mc2Context_->epHcclBuffer_[rankId] + WIN_ADDR_OFFSET + winDataSizeOffset;
}

__aicore__ inline GM_ADDR GetWindStateAddrByRankId(const int32_t rankId)
{
    return (GM_ADDR)mc2Context_->epHcclBuffer_[rankId] + dataState_ * WIN_STATE_OFFSET;
}

