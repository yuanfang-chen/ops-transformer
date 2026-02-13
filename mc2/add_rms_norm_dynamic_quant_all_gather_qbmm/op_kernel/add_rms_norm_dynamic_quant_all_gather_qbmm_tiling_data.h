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
 * \file add_rms_norm_dynamic_quant_all_gather_qbmm_tiling_data.h
 * \brief 定义TilingData
 */

#ifndef ADD_RMS_NORM_DYNAMIC_QUANT_ALL_GATHER_QBMM_H
#define ADD_RMS_NORM_DYNAMIC_QUANT_ALL_GATHER_QBMM_H

#include "kernel_tiling/kernel_tiling.h"

struct AddRmsNormDynamicQuantAllGatherTilingData {
    uint64_t useCore = 0;
    uint64_t numFirstDim = 0;
    uint64_t numLastDim = 0;
    uint64_t numLastDimAligned = 0;
    uint64_t firstDimPerCore = 0;
    uint64_t firstDimPerCoreTail = 0;
    uint64_t firstDimPerLoop = 0;
    uint64_t lastDimLoopNum = 0;
    uint64_t lastDimSliceLen = 0;
    uint64_t lastDimSliceLenTail = 0;
    uint32_t smoothNum = 0;
    float epsilon = 1e-6;
    float avgFactor = 1.0 / (float)5120.0;
    uint32_t M = 252;
    uint32_t Ka = 5120;
    uint32_t N = 0;
    uint32_t aivNum = 24;
    uint32_t worldSize = 4;
};

// tiling struct待完善
struct AddRmsNormDynamicQuantAllGatherQbmmTilingData {
    Mc2InitTiling mc2InitTiling;
    Mc2CcTiling mc2CcTiling;
    AddRmsNormDynamicQuantAllGatherTilingData addRmsNormDynamicQuantAllGatherTilingData;
    TCubeTiling matmulTiling;
};

#endif // ADD_RMS_NORM_DYNAMIC_QUANT_ALL_GATHER_QBMM_H