/* *
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file a2av_common_tiling.h
 * \brief
 */

#ifndef A2AV_COMMON_H
#define A2AV_COMMON_H

namespace MC2KernelTemplate {
static constexpr uint32_t MAX_EP_RANK_SIZE = 8U;
static constexpr uint32_t MAX_EXPERT_PER_EP = 32U;
constexpr uint32_t MAX_EXPERT_SIZE = 256U;

struct HcclA2avTilingInfo {
    Mc2InitTiling hcclInitTiling;
    Mc2CcTiling a2avCcTiling;
};

struct TaskTilingInfo {
    uint64_t BSK;
    uint64_t BS;
    uint64_t H1;
    uint64_t H2;
    uint64_t A;
    uint64_t N1;
    uint64_t N2;
    uint64_t epWorldSize;
    uint64_t e;

    uint32_t mainLoopExpertNum;
    uint32_t tailLoopExpertNum;
    uint32_t totalLoopCount;

    int64_t sendCnt[MAX_EXPERT_SIZE];
    int64_t recvCnt[MAX_EXPERT_SIZE];
};
}
#endif // A2AV_COMMON_H
