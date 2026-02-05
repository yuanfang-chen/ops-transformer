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
 * \file mc2_moe_struct.h
 * \brief
 */
#ifndef OPS_COMMON_INC_MC2_MOE_CONTEXT_H
#define OPS_COMMON_INC_MC2_MOE_CONTEXT_H

namespace Mc2Moe {

constexpr uint32_t HCCL_HOST_KFC_MAX_RANK_NUM = 64;

struct Mc2MoeContext {
    int32_t epRankId;
    int32_t tpRankId;
    int32_t winSize;
    uint64_t kfcContextAddr;    // host kfc方案中，需要传递通信API所需的地址
    uint64_t tpHcclBuffer_[2];
    uint64_t epHcclBuffer_[HCCL_HOST_KFC_MAX_RANK_NUM];   // 按最大数设置
}; // A3/A5 moe 算子共用，按需拓展字段

}  // namespace Mc2Moe

#endif  // OPS_COMMON_INC_MC2_MOE_CONTEXT_H
