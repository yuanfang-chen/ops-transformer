/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file masked_causal_conv1d_struct.h
 * \brief TilingData structure for MaskedCausalConv1d
 */

#ifndef MASKED_CAUSAL_CONV1D_STRUCT_H
#define MASKED_CAUSAL_CONV1D_STRUCT_H

struct MaskedCausalConv1dTilingData {
    // ===== Shape =====
    uint32_t S;                       // token sequence length
    uint32_t B;                       // batch size
    uint32_t H;                       // hidden dimension

    // ===== H-dimension inter-core split =====
    uint32_t hCoreCnt;                // number of cores in H direction
    uint32_t hMainCnt;                // first hMainCnt cores are "main" (larger)
    uint32_t hBlockFactor;            // H elements per main core
    uint32_t hBlockTailFactor;        // H elements per tail core

    // ===== B-dimension inter-core split =====
    uint32_t bCoreCnt;                // number of cores in B direction
    uint32_t bMainCnt;                // first bMainCnt cores are "main"
    uint32_t bBlockFactor;            // B elements per main core
    uint32_t bBlockTailFactor;        // B elements per tail core

    // ===== S-dimension inter-core split =====
    uint32_t sCoreCnt;                // number of cores in S direction
    uint32_t sMainCnt;                // first sMainCnt cores are "main"
    uint32_t sBlockFactor;            // S elements per main core
    uint32_t sBlockTailFactor;        // S elements per tail core

    // ===== UB tile sizes =====
    uint32_t hUb;                     // = H_REG = 64 (VF FP32 register width, fixed)
    uint32_t ubFactorB;               // bUb: batch tile size
    uint32_t ubFactorS;               // sUb: sequence tile size

    // ===== Main-core loop params =====
    uint32_t loopNumH;                // H loop count for main H-core
    uint32_t ubTailFactorH;           // H tail tile size for main H-core
    uint32_t loopNumB;                // B loop count for main B-core
    uint32_t ubTailFactorB;           // B tail tile for main B-core
    uint32_t loopNumS;                // S loop count for main S-core
    uint32_t ubTailFactorS;           // S tail tile for main S-core

    // ===== Tail-core loop params =====
    uint32_t tailBlockLoopNumH;
    uint32_t tailBlockUbTailFactorH;
    uint32_t tailBlockLoopNumB;
    uint32_t tailBlockUbTailFactorB;
    uint32_t tailBlockLoopNumS;
    uint32_t tailBlockUbTailFactorS;

    // ===== Non-contiguous input stride =====
    uint32_t xSStride;               // x S-dim stride (elements); contiguous = B*H
    uint32_t xBStride;               // x B-dim stride (elements); contiguous = H

    // ===== Meta =====
    uint32_t realCoreNum;            // = hCoreCnt * bCoreCnt * sCoreCnt
};

#endif // MASKED_CAUSAL_CONV1D_STRUCT_H
