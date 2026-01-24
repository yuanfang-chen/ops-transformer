/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __MC2_CALC_NUM_BLOCKS_H__
#define __MC2_CALC_NUM_BLOCKS_H__
#include <algorithm>
#include <cstdint>
#include "mc2_log.h"

namespace mc2tiling{
// 根据ascendc提供的aic aiv数量，按照aic:aiv=1:2比例设置逻辑核数量，以aic和aiv/2较小的值为基点
inline uint64_t GetNumBlocks(const uint64_t aic_num, const uint64_t aiv_num, const char* debug_desc)
{
    uint64_t half_aiv_num = aiv_num / 2;                            // aic:aiv按照1：2配比
    uint64_t block_dim = std::min(aic_num, half_aiv_num);
    if(block_dim != aic_num || block_dim != aiv_num * 2) {          // aic:aiv按照1：2配比
        OP_LOGI(debug_desc, 
                "aic_num is %lu, aiv_num is %lu. Since aic_num != 2 * aiv_num, the actual number of enabled AIC cores is %lu, and the number of AIV cores is %lu.",
                aic_num, aiv_num, block_dim, block_dim * 2);        // aic:aiv按照1：2配比
    }
    OP_TILING_CHECK(
        (block_dim == 0U),
        OP_LOGE(
            debug_desc, 
            "platform info is invalid: aic_num is %lu, aiv_num is %lu, the value of block_dim=min(aic_num, aiv_num/2) should not be 0.",
            aic_num, aiv_num),
        return ge::GRAPH_FAILED
    );
    return block_dim;
}
}  // namespace optiling
#endif