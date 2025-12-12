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
 * \file allto_all_matmul_tiling_data.h
 * \brief 定义tiling_data
 */
#ifndef ALLTO_ALL_MATMUL_TILING_H
#define ALLTO_ALL_MATMUL_TILING_H

#include <cstdint>
#include <kernel_tiling/kernel_tiling.h>

struct AlltoAllMatmulTilingInfo {
};

struct AlltoAllMatmulTilingData {
    Mc2InitTiling mc2InitTiling;  // 初始化通信任务配置
    Mc2CcTiling mc2CcTiling;  // 具体每个通信任务的参数配置
    AlltoAllMatmulTilingInfo alltoAllMatmulTilingInfo;
};

#endif // ALLTO_ALL_MATMUL_TILING_H