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
 * \file kv_quant_sparse_attn_sharedkv_metadata.h
 * \brief
 */

#ifndef KV_QUANT_SPARSE_ATTN_SHAREDKV_METADATA_H
#define KV_QUANT_SPARSE_ATTN_SHAREDKV_METADATA_H

#include <cstdint>

namespace optiling {
static constexpr uint32_t AIC_CORE_NUM = 32;  //TODO 根据编译宏确定 aicpu与kernel的宏保持一致
constexpr uint32_t SCFA_META_SIZE = 1024;
using SCFA_METADATA_T = int32_t;

namespace detail {
    struct CoreMetadata{
        uint32_t cubeMetadata[16]; // C: (hasLoad, bN2Start, mStart, s2Start, bN2End, mEnd, s2End, headFdDataIdx, vectorNumForFd)
        uint32_t vectorMetadata[16]; // V: (fdBN2Idx, fdMIdx, fdMStart, fdMNum, fdWorkSpaceIdx, fdWorkSpaceNum) * 2
    };

    struct SasMetaData {
        struct CoreMetadata coreMetadata[AIC_CORE_NUM];
    };
};
static_assert(SCFA_META_SIZE * sizeof(SCFA_METADATA_T) >= sizeof(detail::SasMetaData));
};

#endif
