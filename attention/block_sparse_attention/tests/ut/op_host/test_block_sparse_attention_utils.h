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
 * \file test_block_sparse_attention_utils.h
 * \brief BlockSparseAttention 测试工具.
 */

#ifndef UTEST_BLOCK_SPARSE_ATTENTION_UTILS_H
#define UTEST_BLOCK_SPARSE_ATTENTION_UTILS_H

#include <cstdint>

namespace bsaTestUtils {
#define BSA_TPL_INVALID 0xFFFFFFFF
#define BSA_TPL_FLOAT 0
#define BSA_TPL_FLOAT16 1
#define BSA_TPL_BF16 27

#define BSA_LAYOUT_TND 0
#define BSA_LAYOUT_BNSD 1

#define BSA_MASK_TYPE_NONE 0
#define BSA_MASK_TYPE_CAUSAL 3

#define BSA_INNER_PRECISE_FLOAT 0
#define BSA_INNER_PRECISE_HALF 1

#define BSA_PAGED_CACHE_FALSE 0
#define BSA_PAGED_CACHE_TRUE 1

#define BSA_LSE_NO_OUT 0
#define BSA_LSE_OUT 1

    inline uint64_t BSAEncodeTilingKey(int qLayout, int kvLayout, int maskType, int innerPrecise,
                                       int pagedCache, int dataType, int lseFlag) {
        uint64_t value = 9000000000000000ULL;
        int shift = 0;

        value |= (uint64_t)(qLayout & 0xF) << shift;
        shift += 4;

        value |= (uint64_t)(maskType & 0x7) << shift;
        shift += 3;

        value |= (uint64_t)(innerPrecise & 0x7) << shift;
        shift += 3;

        value |= (uint64_t)(pagedCache & 0x7) << shift;
        shift += 3;

        value |= (uint64_t)(kvLayout & 0xF) << shift;
        shift += 4;

        value |= (uint64_t)(dataType & 0xFF) << shift;
        shift += 8;

        value |= (uint64_t)(lseFlag & 0x1) << 28;

        return value;
    }
}

#endif
