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
 * \file flash_attention.cpp
 * \brief FlashAttention Kernel实现（接口桩，具体实现待补充）
 *
 * Kernel设计要点：
 *
 * 1. 统一训练/推理场景：
 *    - 通过returnSoftmaxLse标志区分：训练时输出softmax_lse供反向传播，推理时不输出。
 *    - 核心计算流程（QK^T → scale → mask → softmax → V加权）完全一致。
 *
 * 2. 支持的layout组合：
 *    - layoutQ: BSND / BNSD / TND
 *    - layoutKV: BSND / TND / PA_ND / PA_Nz
 *    - layoutOut: BSND / BNSD / TND
 *    - 当layoutQ/KV/Out不一致时，kernel侧负责数据重排。
 *
 * 3. 分页KV缓存（PA_ND/PA_Nz）：
 *    - 通过block_table进行KV block的间接寻址。
 *    - 每个sequence在KV cache中有若干不连续的block，通过block_table[batch][block_idx]映射。
 *
 * 4. 变长序列：
 *    - cu_seqlens模式：通过累积长度前缀和定位每个sequence的起止token。
 *    - seqused模式：每个batch提供实际有效序列长度，其余为padding。
 *
 * 5. Tiling驱动：
 *    - kernel通过TilingContext获取FlashAttentionTilingData中的切分参数。
 *    - 支持metadata预计算tiling，消除推理时的tiling计算开销。
 *
 * 6. 掩码模式（maskMode）：
 *    - 0: 全注意力（无掩码）
 *    - 1: 因果掩码（下三角，训练标准场景）
 *    - 2: 非因果掩码（上三角）
 *    - 3: Band/Prefix掩码
 *    - 4: 滑动窗口掩码（winLeft/winRight控制窗口大小）
 *
 * 7. 数据类型：
 *    - 输入q/k/v：FLOAT16 或 BFLOAT16
 *    - 输出attention_out：与输入一致
 *    - 输出softmax_lse：FLOAT32
 *    - sinks：FLOAT32
 *    - 辅助输入（block_table/cu_seqlens/seqused/metadata）：INT32
 */

#include "kernel_operator.h"
#include "../op_host/flash_attention_tiling.h"

using namespace AscendC;
using namespace optiling;

// ============================================================
// 前向声明
// ============================================================

/**
 * @brief FlashAttention核心计算模板
 *
 * @tparam T         q/k/v/attention_out的数据类型（half或bfloat16）
 * @tparam LAYOUT_Q  q的layout（0=BSND, 1=BNSD, 2=TND）
 * @tparam LAYOUT_KV kv的layout（0=BSND, 1=TND, 2=PA_ND, 3=PA_Nz）
 * @tparam LAYOUT_OUT 输出的layout（0=BSND, 1=BNSD, 2=TND）
 * @tparam MASK_MODE 掩码模式（0-4）
 * @tparam RETURN_LSE 是否输出softmax_lse（0或1）
 */
template <typename T,
          int LAYOUT_Q,
          int LAYOUT_KV,
          int LAYOUT_OUT,
          int MASK_MODE,
          int RETURN_LSE>
class FlashAttentionKernel {
public:
    // TODO: 实现FlashAttention核心计算逻辑
    // 参考flash_attention_score/op_kernel/flash_attention_score.cpp中的train kernel实现
    // 和fused_infer_attention_score/op_kernel/中的infer kernel实现

    __aicore__ inline void Init(GM_ADDR q, GM_ADDR k, GM_ADDR v,
                                GM_ADDR blockTable, GM_ADDR cuSeqlensQ, GM_ADDR cuSeqlensKv,
                                GM_ADDR sequsedQ, GM_ADDR sequsedKv, GM_ADDR sinks,
                                GM_ADDR metadata, GM_ADDR attentionOut, GM_ADDR softmaxLse,
                                const FlashAttentionTilingData *tilingData)
    {
        // TODO: 初始化各输入输出GlobalTensor
    }

    __aicore__ inline void Process()
    {
        // TODO: 实现FlashAttention主计算流程：
        // 1. 分块读取Q/K/V到L1/L0 buffer
        // 2. 计算 S = Q * K^T * scale
        // 3. 应用掩码（根据maskMode）
        // 4. 计算 P = softmax(S)，保存lse供训练使用
        // 5. 计算 O = P * V
        // 6. 写回输出
    }

private:
    // TODO: 声明GlobalTensor、LocalTensor及计算所需的中间buffer
};

// ============================================================
// Kernel主入口（由框架根据tilingKey分发到对应template实例）
// ============================================================

/**
 * @brief FlashAttention kernel主入口
 *
 * 框架根据tiling中的tilingKey选择对应的模板特化实例。
 * 当前为接口桩，完整实现请参考arch35目录下的kernel代码。
 *
 * TODO: 根据tilingKey分发到不同的FlashAttentionKernel模板特化
 */
extern "C" __global__ __aicore__ void flash_attention(
    GM_ADDR q, GM_ADDR k, GM_ADDR v,
    GM_ADDR blockTable, GM_ADDR cuSeqlensQ, GM_ADDR cuSeqlensKv,
    GM_ADDR sequsedQ, GM_ADDR sequsedKv, GM_ADDR sinks,
    GM_ADDR metadata, GM_ADDR attentionOut, GM_ADDR softmaxLse,
    GM_ADDR workspace, GM_ADDR tilingGm)
{
    // TODO: 读取tiling数据并根据tilingKey分发到对应kernel template
}
