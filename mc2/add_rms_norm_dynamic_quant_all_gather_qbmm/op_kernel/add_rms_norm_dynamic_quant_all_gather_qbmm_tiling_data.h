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

struct AddRmsNormDynamicQuantV2TilingData {
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
    float epsilon = 0;
    float avgFactor = 0;
};

struct AllGatherTilingData {
    uint64_t M;
    uint64_t K;
    uint64_t scaleHiddenSize;
    uint64_t aivNum;
    uint64_t totalWinSize;   // Win区总大小，即HCCL_BUFFER_SIZE
};

// tiling struct待完善
struct AddRmsNormDynamicQuantAllGatherQbmmTilingData {
    Mc2InitTiling mc2InitTiling;
    Mc2CcTiling mc2CcTiling;
    AddRmsNormDynamicQuantV2TilingData addRmsNormDynamicQuantV2TilingData;
    AllGatherTilingData allGatherTilingData;
    TCubeTiling matmulTiling;
};

#endif // ADD_RMS_NORM_DYNAMIC_QUANT_ALL_GATHER_QBMM_H