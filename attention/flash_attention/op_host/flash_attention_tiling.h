/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FLASH_ATTENTION_TILING_H_
#define FLASH_ATTENTION_TILING_H_

#include <cstdint>
#include <register/op_impl_registry.h>

namespace optiling {

// ============================================================
// FlashAttention Tiling数据结构
// ============================================================

/**
 * @brief FlashAttention tiling参数，供kernel使用。
 *
 * 该结构体中的参数由TilingFlashAttention函数填充，
 * 当metadata输入不为nullptr时，直接从metadata反序列化得到。
 */
struct FlashAttentionTilingData {
    // 基础维度参数
    uint32_t batchSize    = 0U;  // B（TND时为总token数/head数推断）
    uint32_t numHeadsQ    = 0U;  // N_q
    uint32_t numHeadsKV   = 0U;  // N_kv（GQA时 < N_q）
    uint32_t seqLenQ      = 0U;  // S_q（TND时为T_q）
    uint32_t seqLenKV     = 0U;  // S_kv（TND时为T_kv）
    uint32_t headDim      = 0U;  // D（Q和K共享）
    uint32_t headDimV     = 0U;  // D_v（V的head维度）

    // Tiling切分参数
    uint32_t seqLenQTile  = 0U;  // Q序列方向的切分块大小
    uint32_t seqLenKVTile = 0U;  // KV序列方向的切分块大小
    uint32_t headDimTile  = 0U;  // head维度的切分块大小

    // 核使用参数
    uint32_t coreNum      = 0U;  // 参与计算的AICore数量
    uint32_t tilingKey    = 0U;  // 用于选择kernel template的tiling key

    // 布局标志（与kernel侧保持一致）
    uint32_t layoutQFlag  = 0U;  // 0=BSND, 1=BNSD, 2=TND
    uint32_t layoutKVFlag = 0U;  // 0=BSND, 1=TND, 2=PA_ND, 3=PA_Nz
    uint32_t layoutOutFlag = 0U; // 0=BSND, 1=BNSD, 2=TND

    // 功能标志
    uint32_t returnSoftmaxLse = 0U;  // 是否输出softmax_lse
    uint32_t deterministic    = 0U;  // 是否确定性计算
    uint32_t maskMode         = 0U;  // 掩码模式
    int32_t  winLeft          = 0;   // 左窗口（maskMode=4时有效）
    int32_t  winRight         = 0;   // 右窗口（maskMode=4时有效）

    // softmax缩放系数（存储为float，0.0时kernel侧使用1/sqrt(D)）
    float    softmaxScale     = 0.0f;

    // 分页注意力参数
    uint32_t blockSize        = 0U;  // PA模式下每个KV block的token数
    uint32_t numBlocks        = 0U;  // PA模式下总block数

    // workspace分配参数
    uint64_t workspaceSize    = 0ULL;
};

// ============================================================
// Tiling函数声明
// ============================================================

/**
 * @brief FlashAttention的tiling计算函数。
 *
 * 根据输入shape、属性和硬件资源计算kernel所需的切分参数。
 * 当metadata输入不为nullptr时，直接从metadata读取预计算的tiling方案，
 * 跳过切分计算（用于推理场景的tiling复用）。
 *
 * @param context tiling上下文，包含输入/输出shape和属性信息
 * @return ge::graphStatus GRAPH_SUCCESS表示成功
 */
ge::graphStatus TilingFlashAttention(gert::TilingContext *context);

}  // namespace optiling

#endif  // FLASH_ATTENTION_TILING_H_
