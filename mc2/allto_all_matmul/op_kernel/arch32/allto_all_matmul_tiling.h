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
 * \file allto_all_matmul_tiling.h
 * \brief
 */

#ifndef ASCENDC_ALL_TO_ALL_MATMUL_TILING_H
#define ASCENDC_ALL_TO_ALL_MATMUL_TILING_H

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"

struct AlltoAllMatmulInfo {
    uint32_t M;
    uint32_t K;
    uint32_t N;
    uint32_t worldSize;
    uint32_t aivNum;
    uint32_t totalUbSize;
    bool hasBias;
};

struct CoCTiling {
    int32_t m0 = -1;
    int32_t k0 = -1;
    int32_t n0 = -1;
    int32_t swizzlCount = -1;
    int32_t swizzlDirect = -1;
    int32_t pValue = -1;
    int32_t ubMoveNum = -1;
    int32_t first_step_core_num = -1;
    int32_t second_step_core_num = -1;
};


struct AlltoAllMatmulTilingData {
    Mc2InitTiling mc2InitTiling;
    Mc2CcTiling mc2CcTiling;
    AlltoAllMatmulInfo allToAllMatmulInfo;
    CoCTiling cocTiling;
};
#endif