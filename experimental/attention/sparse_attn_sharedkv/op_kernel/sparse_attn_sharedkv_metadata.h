/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
 
/*!
 * \file sparse_flash_attention_antiquant_metadata.h
 * \brief
 */

#ifndef SPARSE_FLASH_ATTENTION_ANTIQUANT_METADATA_H
#define SPARSE_FLASH_ATTENTION_ANTIQUANT_METADATA_H

#include <cstdint>

namespace optiling {
const uint32_t CORE_NUM = 24;  //TODO 根据编译宏确定 aicpu与kernel的宏保持一致
constexpr uint32_t SCFA_META_SIZE = 1024;
using SCFA_METADATA_T = int32_t;

namespace detail {
    // 分核功能模块输出：FD信息，包含需要归约的数据索引及其分核信息
    struct ScfaMetaData{

    };
};
static_assert(SCFA_META_SIZE * sizeof(SCFA_METADATA_T) >= sizeof(detail::ScfaMetaData));
};

#endif
