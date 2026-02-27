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
 * \file mhc_post_tiling_data.h
 * \brief tiling data struct
 */

#ifndef __MHC_POST_TILLING_DATA_H__	 
#define __MHC_POST_TILLING_DATA_H__
 
struct MhcPostTilingData {
    uint32_t totalItems;
    uint32_t itemsPerCore;
    uint32_t remainderItems;
    uint32_t usedCores;
    uint32_t S;
    uint32_t n;
    uint32_t D;
    uint32_t tileD;
    uint32_t nTilesD;
    uint32_t alignedD;
    uint32_t lastTileD;
    int64_t usedCoreNum;
    int64_t normalCoreProcessNum;
    int64_t tailCoreProcessNum;
    int64_t bsInner;
    int64_t bsOuter;
    int64_t bsTail;
    int64_t dInner;
    int64_t dOuter;
    int64_t dTail;
};

#endif
