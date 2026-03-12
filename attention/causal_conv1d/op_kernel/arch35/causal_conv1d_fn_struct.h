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
 * \file causal_conv1d_fn_struct.h
 * \brief tiling data struct
 */

#ifndef __CAUSAL_CONV1D_FN_TILING_DATA_H__
#define __CAUSAL_CONV1D_FN_TILING_DATA_H__

// tiling data struct for causal_conv1d_fn
struct CausalConv1dFnTilingData {
    // ===== 核内UB切分参数 =====
    uint64_t loopNumBS;         // 每个核内BS方向的loop循环数
    uint64_t loopNumDim;        // 每个核内Dim方向的loop循环数（2D时固定为1）
    uint64_t ubFactorBS;        // 每个核内BS方向单次循环载入的大小
    uint64_t ubTailFactorBS;    // 每个核内BS方向尾次循环载入的大小
    uint64_t ubFactorDim;       // 每个核内Dim方向单次循环载入的大小
    uint64_t ubTailFactorDim;   // 每个核内Dim方向尾次循环载入的大小
    uint64_t tailBlockloopNumBS;      // 尾核内BS方向的loop循环数
    uint64_t tailBlockloopNumDim;     // 尾核内Dim方向的loop循环数
    uint64_t tailBlockubFactorBS;     // 尾核内BS方向单次循环载入的大小
    uint64_t tailBlockubTailFactorBS; // 尾核内BS方向尾次循环载入的大小
    uint64_t tailBlockubFactorDim;    // 尾核内Dim方向单次循环载入大小
    uint64_t tailBlockubTailFactorDim;// 尾核内Dim方向尾次循环载入大小

    // ===== dim方向核间切分信息 =====
    uint64_t dimCoreNum;              // dim方向总核数
    uint64_t dimRemainderCores;       // 前多少个核是大核（分配base+1个128-块）
    uint64_t dimBlockFactor;          // 大核的dim大小（(base+1) * 128，前dimRemainderCores个核）
    uint64_t dimBlockTailFactor;      // 小核的dim大小（base * 128，后面的核）

    // ===== BS方向核间切分信息 =====
    uint64_t bsCoreNum;               // BS方向总核数
    uint64_t bsRemainderCores;        // BS方向前多少个核是大核（均分策略）
    uint64_t bsBlockFactor;           // BS方向大核处理的长度（含overlap，前bsRemainderCores个核）
    uint64_t bsBlockTailFactor;       // BS方向小核处理的长度（后面的核）

    // ===== 核数信息 =====
    uint64_t realCoreNum;             // 实际使用核数 = dimCoreNum × bsCoreNum
    uint64_t kernelWidth;       // 卷积核宽度 K
    uint64_t cuSeqLen;          // cu_seq_len 大小
    uint64_t dim;               // 特征维度大小
    uint64_t batch;             // batch大小
    uint64_t validBatchStart;   // 有效 batch 的起始索引（在原始 cacheIndices 中）
    uint64_t validBatchCount;   // 有效 batch 的数量
    uint64_t validSeqStart;     // 有效序列的起始位置（在原始 x 中的行偏移）
    uint64_t validSeqLen;       // 有效序列的总长度
    uint64_t xStride;           // x 的 stride（即 dim）
    uint64_t cacheStride;       // convStates 的 stride（即 dim）
    uint64_t residualConnection;// 是否做残差连接
};

#endif