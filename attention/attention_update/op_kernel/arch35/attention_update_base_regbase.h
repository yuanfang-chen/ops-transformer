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
 * \file attention_update_base_regbase.h
 * \brief
 */
#ifndef ATTENTION_UPDATE_BASE_REGBASE_H_
#define ATTENTION_UPDATE_BASE_REGBASE_H_

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "../../inc/load_store_utils.h"

namespace AttentionUpdateOpt {
using namespace AscendC;

__aicore__ inline constexpr uint32_t GetUbBlockSize()
{
    return 32U;
}

__aicore__ inline constexpr uint32_t GetVRegSize()
{
#if __CCE_AICORE__ == 310
    return AscendC::VECTOR_REG_WIDTH;
#else
    return 256U;
#endif
}

__aicore__ inline uint64_t CeilDiv(uint64_t a, uint64_t b)
{
    using type = typename std::conditional<sizeof(uint64_t) == sizeof(uint8_t) || sizeof(uint64_t) == sizeof(uint16_t),
                                           uint32_t, uint64_t>::type;
    type res = (static_cast<type>(a) + static_cast<type>(b) - 1) / static_cast<type>(b);
    return static_cast<uint64_t>(res);
}

__aicore__ inline uint64_t CeilAlign(uint64_t a, uint64_t b)
{
    using type = typename std::conditional<sizeof(uint64_t) == sizeof(uint8_t) || sizeof(uint64_t) == sizeof(uint16_t),
                                           uint32_t, uint64_t>::type;
    type res = (static_cast<type>(a) + static_cast<type>(b) - 1) / static_cast<type>(b) * static_cast<type>(b);
    return static_cast<uint64_t>(res);
}

static constexpr uint32_t BUFFER_NUM = 2;
static constexpr uint32_t UB_BLOCK_SIZE = GetUbBlockSize();

} // namespace AttentionUpdateOpt
#endif // ATTENTION_UPDATE_BASE_REGBASE_H_
