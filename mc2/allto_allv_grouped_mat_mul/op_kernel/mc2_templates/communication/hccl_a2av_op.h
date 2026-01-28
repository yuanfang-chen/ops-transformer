/* *
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
  */

#ifndef MC2_HCCL_IMPL_H
#define MC2_HCCL_IMPL_H

#include "lib/hccl/hccl.h"

using namespace AscendC;

namespace MC2KernelTemplate {
template <typename TilingDataType, uint32_t SendCnt, uint32_t RecvCnt>
class HcclA2avOp {
public:
    __aicore__ inline HcclA2avOp(TilingDataType *tiling) : tiling_(tiling) {}

    __aicore__ inline void Init()
    {
        GM_ADDR hcclContextGm = GetHcclContext<HCCL_GROUP_ID_0>();
        hccl_.Init(hcclContextGm, (__gm__ void *)(&(tiling->hcclInitTiling)));
        hccl_.SetCcTiling((__gm__ void *)(&(tiling->alltoAllvCcTiling)));
        // 获取通信基本信息
        rankId_ = hccl_.GetRankId();
        rankDim_ = hccl_.GetRankDim();
        // 从tiling获取必要参数
        expertNumInOneRank_ = tiling_->commonTilingInfo.E_ep;
        axisH1_ = tiling_->commonTilingInfo.H1;
        axisN1_ = tiling_->commonTilingInfo.N1;
    }

    __aicore__ inline void Prepare(uint8_t hcclDataType)
    {
        if ASCEND_IS_AIC {
            return;
        }
        if ASCEND_IS_AIV {
            if (GetBlockIdx() != 0) {
                return;
            }
        }

        // 设置通信数据类型
        if constexpr (std::is_same_v<hcclDataType, bfloat16_t>) {
            hcclDataType_ = HCCL_DATA_TYPE_BFP16;
        } else if constexpr (std::is_same_v<hcclDataType, hifloat8_t>) {
            hcclDataType_ = HCCL_DATA_TYPE_HIF8;
        } else {
            hcclDataType_ = HCCL_DATA_TYPE_FP16;
        }

        const auto *sendCnt = tiling_->aicpuTiling.sendCnt[0];
        const auto *recvCnt = tiling_->aicpuTiling.recvCnt[0];

        for (uint32_t e = 0U; e < expertNumInOneRank_; e++) {
            for (uint32_t i = 0U; i < rankDim_; i++) {
                alltoAllvSendCnt[i] = static_cast<uint64_t>(sendCnt[i * expertNumInOneRank_ + e]) * axisH1_;
                alltoAllvRecvCnt[i] = static_cast<uint64_t>(recvCnt[i * expertNumInOneRank_ + e]) * axisH1_;
            }
            alltoAllvSendOffset[0] = 0UL;
            for (uint32_t j = 0U; j < e; j++) { // 0sendOffset
                alltoAllvSendOffset[0U] += static_cast<uint64_t>(sendCnt[j]) * axisH1_;
            }
            for (uint32_t i = 1U; i < rankDim_; i++) {
                alltoAllvSendOffset[i] = alltoAllvSendOffset[i - 1U];
                for (uint32_t j = 0U; j < expertNumInOneRank_; j++) {
                    alltoAllvSendOffset[i] +=
                        static_cast<uint64_t>(sendCnt[e + (i - 1U) * expertNumInOneRank_ + j]) * axisH1_;
                }
            }
            for (uint32_t i = 0U; i < rankDim_; i++) {
                if ((e == 0U) && (i == 0U)) {
                    alltoAllvRecvOffset[i] = 0UL;
                    alltoAllvRecvOffsetLastSum += alltoAllvRecvCnt[0];
                } else {
                    alltoAllvRecvOffset[i] = alltoAllvRecvOffsetLastSum;
                    alltoAllvRecvOffsetLastSum += alltoAllvRecvCnt[i];
                }
            }
            alltoAllvHandleId_[e] =
                hccl_.AlltoAllV<true>((__gm__ uint8_t *)sendBuffer, alltoAllvSendCnt, alltoAllvSendOffset,
                hcclDataType_, (__gm__ uint8_t *)recvBuffer, alltoAllvRecvCnt, alltoAllvRecvOffset, hcclDataType_);
        }
    }

    __aicore__ inline void Wait()
    {
        if ASCEND_IS_AIC {
            return;
        }
        if ASCEND_IS_AIV {
            if (GetBlockIdx() != 0) {
                return;
            }
        }
        hccl_.Wait(alltoAllvHandleId_[e]);
        SyncAll<false>();
    }

    __aicore__ inline void End()
    {
        SyncAll<false>();
        hccl_.Finalize();
    }

private:
#if defined(__DAV_C310__)
    Hccl<HcclServerType::HCCL_SERVER_TYPE_CCU> hccl_;
#else
    Hccl<HcclServerType::HCCL_SERVER_TYPE_AICPU> hccl_;
#endif

    static constexpr uint64_t MAX_HANDLE_ID_NUM = 64U;
    static constexpr uint32_t MAX_EP_RANK_SIZE = 8U;

    TilingDataType *tiling_;

    // 通信相关参数
    uint32_t rankId_ = 0U;
    uint32_t rankDim_ = 8U;
    uint32_t expertNumInOneRank_ = 0U;
    uint64_t axisH1_ = 0UL;
    uint64_t axisN1_ = 0UL;

    // 张量地址
    __gm__ uint8_t *inputTensor_ = nullptr;
    __gm__ uint8_t *outputTensor_ = nullptr;

    // 通信数据结构
    HcclHandle alltoAllvHandleId_[MAX_HANDLE_ID_NUM] = {INVALID_HANDLE_ID};
    HcclDataType hcclDataType_ = HCCL_DATA_TYPE_FP16;

    // 计数和偏移缓存
    uint64_t alltoAllvRecvOffsetLastSum = 0UL;
    uint64_t alltoAllvSendCnt[MAX_EP_RANK_SIZE] = {0UL};
    uint64_t alltoAllvSendOffset[MAX_EP_RANK_SIZE] = {0UL};
    uint64_t alltoAllvRecvCnt[MAX_EP_RANK_SIZE] = {0UL};
    uint64_t alltoAllvRecvOffset[MAX_EP_RANK_SIZE] = {0UL};
};
};

#endif