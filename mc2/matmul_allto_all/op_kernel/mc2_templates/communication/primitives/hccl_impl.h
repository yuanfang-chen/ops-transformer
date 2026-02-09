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
 * \file hccl_impl.h 
 * \brief
 */

#ifndef MC2_HCCL_IMPL_H
#define MC2_HCCL_IMPL_H

#include "lib/hccl/hccl.h"

namespace MC2KernelTemplate {
using namespace AscendC;

struct MC2AlltoAllContext {
    uint32_t taskCnt;
    GM_ADDR sendBuffer;
    GM_ADDR recvBuffer;
    uint64_t sendOffset;
    uint64_t recvOffset;
    uint64_t sendCount;
    uint64_t strideCount;
    uint64_t hcclDataType;
};
/**
 * ServerType:通信控制方式，ccu/mte/aicpu等
 * SendCnt:每轮发送的次数
 * RecvCnt:每轮等待发送完毕的次数
 * TilingDataType:tilingdata具体数据类型
 * CommunicationPrimitive:通信原语，alltoall等,暂不启用
 */
template <HcclServerType ServerType, typename TilingDataType, uint32_t SendCnt, uint32_t RecvCnt>
class HcclCommunication
{
public:
    __aicore__ inline HcclCommunication(TilingDataType* tiling) : tiling_(tiling){};
    __aicore__ inline void Init();
    __aicore__ inline void Prepare(uint32_t taskCnt);
    __aicore__ inline MC2AlltoAllContext* GetCommContextPtr();
    __aicore__ inline void Process();
    __aicore__ inline void End();

private:
    enum Communicationtype{
        COMMUNICATION_WAIT_ONE,
        COMMUNICATION_SEND_ONE
    };
    TilingDataType* tiling_;
    Hccl<ServerType> hccl_;
    MC2AlltoAllContext context_;
    uint64_t sendIndex_;
    uint64_t recvIndex_;
    bool notifyFlag_ = false;
    AscendC::HcclDataType hcclDataType_;
    Communicationtype communicationType_ = COMMUNICATION_WAIT_ONE;
    AscendC::HcclHandle hTasks_[16]; //hccl只支持最多16个任务并行
};

template <HcclServerType ServerType, typename TilingDataType, uint32_t SendCnt, uint32_t RecvCnt>
__aicore__ inline void HcclCommunication<ServerType, TilingDataType, SendCnt, RecvCnt>::Init()
{
    notifyFlag_ = false;   
    if ASCEND_IS_AIV {
        if (AscendC::GetBlockIdx() == 0) {
            notifyFlag_ = true; 
        }
    }

    hccl_.InitV2(GetHcclContext<0>(), &(tiling_->mc2InitTiling));
    hccl_.SetCcTilingV2(offsetof(TilingDataType, mc2CcTiling));
    if constexpr (SendCnt == 1U && RecvCnt == 0U) {
        communicationType_ = Communicationtype::COMMUNICATION_SEND_ONE;
    } else if constexpr (SendCnt == 0U && RecvCnt == 1U) {
        communicationType_ = Communicationtype::COMMUNICATION_WAIT_ONE;
    }
}

template <HcclServerType ServerType, typename TilingDataType, uint32_t SendCnt, uint32_t RecvCnt>
__aicore__ inline void HcclCommunication<ServerType, TilingDataType, SendCnt, RecvCnt>::Prepare(uint32_t taskCnt)
{
    //只有通信核参与通信
    if (!notifyFlag_) {
        return;
    }
    hcclDataType_ = (AscendC::HcclDataType)(static_cast<uint8_t>(context_.hcclDataType));
    // 如果是先通后算就全量启动通信
    if (communicationType_ == Communicationtype::COMMUNICATION_WAIT_ONE) {
        //alltoall接口职责不单一，这里只使用repeat=1的模式
        uint8_t repeat = 1;
        for (uint32_t i = 0; i < taskCnt; i++) {
            hTasks_[sendIndex_ + i] = hccl_.template AlltoAll<true>(context_.sendBuffer, context_.recvBuffer, context_.sendCount, hcclDataType_, context_.strideCount, repeat);
            context_.sendBuffer += context_.sendOffset;
            context_.recvBuffer += context_.recvOffset;
        }
        //更新全局变量
        sendIndex_ += taskCnt;
    }
}

template <HcclServerType ServerType, typename TilingDataType, uint32_t SendCnt, uint32_t RecvCnt>
__aicore__ inline MC2AlltoAllContext*
HcclCommunication<ServerType, TilingDataType, SendCnt, RecvCnt>::GetCommContextPtr()
{
    return &context_;
}

template <HcclServerType ServerType, typename TilingDataType, uint32_t SendCnt, uint32_t RecvCnt>
__aicore__ inline void HcclCommunication<ServerType, TilingDataType, SendCnt, RecvCnt>::Process()
{
    //只有通信核参与通信
    if (!notifyFlag_) {
        return;
    }
    if (communicationType_ == Communicationtype::COMMUNICATION_WAIT_ONE) {
        hccl_.Wait(hTasks_[recvIndex_]);
        recvIndex_++;
    } else if (communicationType_ == Communicationtype::COMMUNICATION_SEND_ONE) {
        uint8_t repeat = 1;
        hTasks_[sendIndex_] = hccl_.template AlltoAll<true>(context_.sendBuffer, context_.recvBuffer, context_.sendCount, hcclDataType_, context_.strideCount, repeat);
        context_.sendBuffer += context_.sendOffset;
        context_.recvBuffer += context_.recvOffset;
        sendIndex_++;
    }
}

template <HcclServerType ServerType, typename TilingDataType, uint32_t SendCnt, uint32_t RecvCnt>
__aicore__ inline void HcclCommunication<ServerType, TilingDataType, SendCnt, RecvCnt>::End()
{
    // 如果是先算后通就全量等待通信
    if (notifyFlag_ && communicationType_ == Communicationtype::COMMUNICATION_SEND_ONE) {
        for (;recvIndex_ < sendIndex_;++recvIndex_) {
            hccl_.Wait(hTasks_[recvIndex_]);
        }
    }

    // 防止block_idx0执行过快清空RcvCnt,增加全核同步
    SyncAll<false>();
    if (notifyFlag_) {
        hccl_.Finalize();
    }
}
}; // namespace MC2KernelTemplate

#endif