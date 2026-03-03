/**
 * Copyright c 2025 Huawei Technologies Co. Ltd.
 * This program is free software you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 the "License".
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS WITHOUT WARRANTIES OF ANY KIND EITHER EXPRESS OR IMPLIED
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file causal_conv1d_update_struct.h
 * \brief CausalConv1dUpdate tiling struct
 */

#ifndef CAUSAL_CONV1D_UPDATE_STRUCT_H
#define CAUSAL_CONV1D_UPDATE_STRUCT_H


struct CausalConv1dUpdateTilingData{
// Core distribution parameters
int64_t usedCoreNum;              // Total used core number
int64_t dimCoreCnt;               // Number of cores for dim direction
int64_t batchCoreCnt;             // Number of cores for batch direction

// Dim tiling parameters inter-core
int64_t dimChunkSize;             // Dim chunk size per core 256B aligned
int64_t dimTailSize;              // Dim tail size for last core

// Batch tiling parameters inter-core
int64_t batchPerCore;             // Batches per core regular
int64_t batchTailPerCore;         // Batches for tail core
int64_t validBatchStart;          // First valid batch index
int64_t validBatchEnd;            // Last valid batch index inclusive

// Intra-core tiling parameters UB loop
int64_t ubBatchSize;              // Batch size per UB iteration
int64_t ubDimSize;                // Dim size per UB iteration elements
int64_t batchLoopCnt;             // Batch loop count within core
int64_t dimLoopCnt;               // Dim loop count within core

// Shape information for kernel use
int64_t batchSize;                // Batch size
int64_t seqLen;                   // Sequence length for 3D input
int64_t cuSeqLen;                 // Cumulative sequence length for 2D input
int64_t dim;                      // Dimension size
int64_t kernelSize;               // Kernel size K
int64_t xInputMode;               // Input mode: 0 for 3D 1 for 2D
};

#endif