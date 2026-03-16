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
 * \file flash_attention_tiling.cpp
 * \brief FlashAttention Tiling实现（接口桩，具体实现待补充）
 *
 * Tiling主要职责：
 * 1. 读取输入shape、layout属性和硬件资源信息
 * 2. 当metadata不为空时，从metadata反序列化预计算的tiling方案（推理复用场景）
 * 3. 当metadata为空时，自主计算最优切分方案（训练/在线推理场景）
 * 4. 填充FlashAttentionTilingData并写入TilingContext
 * 5. 设置workspace大小
 */

#include <register/op_impl_registry.h>
#include "log/log.h"
#include "flash_attention_tiling.h"

using namespace ge;
using namespace AscendC;

namespace optiling {

// 属性索引（与flash_attention_def.cpp对齐）
static constexpr size_t ATTR_IDX_SOFTMAX_MODE      = 0;
static constexpr size_t ATTR_IDX_MASK_MODE         = 1;
static constexpr size_t ATTR_IDX_WIN_LEFT          = 2;
static constexpr size_t ATTR_IDX_WIN_RIGHT         = 3;
static constexpr size_t ATTR_IDX_LAYOUT_Q          = 4;
static constexpr size_t ATTR_IDX_LAYOUT_KV         = 5;
static constexpr size_t ATTR_IDX_LAYOUT_OUT        = 6;
static constexpr size_t ATTR_IDX_RETURN_SOFTMAX_LSE = 7;
static constexpr size_t ATTR_IDX_DETERMINISTIC     = 8;

// 输入索引
static constexpr size_t INPUT_IDX_Q                = 0;
static constexpr size_t INPUT_IDX_K                = 1;
static constexpr size_t INPUT_IDX_V                = 2;
static constexpr size_t INPUT_IDX_BLOCK_TABLE      = 3;
static constexpr size_t INPUT_IDX_CU_SEQLENS_Q     = 4;
static constexpr size_t INPUT_IDX_CU_SEQLENS_KV    = 5;
static constexpr size_t INPUT_IDX_SEQUSED_Q        = 6;
static constexpr size_t INPUT_IDX_SEQUSED_KV       = 7;
static constexpr size_t INPUT_IDX_SINKS            = 8;
static constexpr size_t INPUT_IDX_METADATA         = 9;

/**
 * @brief 从metadata tensor反序列化tiling数据（推理场景预计算tiling复用）
 *
 * TODO: 实现从INT32 metadata tensor中读取并填充FlashAttentionTilingData
 *
 * @param context   tiling上下文
 * @param tilingData 待填充的tiling数据
 * @return bool true表示成功从metadata中读取，false表示metadata为空或无效
 */
static bool TryDeserializeFromMetadata(gert::TilingContext *context, FlashAttentionTilingData &tilingData)
{
    // TODO: 检查metadata输入是否有效，若有效则反序列化
    (void)context;
    (void)tilingData;
    return false;
}

/**
 * @brief 根据输入shape和硬件资源计算最优的切分参数
 *
 * TODO: 实现完整的tiling计算逻辑，包括：
 *   - 根据numHeadsQ/seqLenQ/seqLenKV/headDim和硬件AICore数量确定seqLenQTile/seqLenKVTile
 *   - 对于GQA(N_kv < N_q)场景选择合适的template
 *   - 对于PA(PA_ND/PA_Nz)场景设置blockSize等参数
 *   - 选择合适的tilingKey用于kernel template选择
 *   - 计算workspace大小
 *
 * @param context   tiling上下文
 * @param tilingData 待填充的tiling数据
 * @return ge::graphStatus
 */
static ge::graphStatus ComputeTiling(gert::TilingContext *context, FlashAttentionTilingData &tilingData)
{
    // TODO: 实现具体的tiling计算逻辑
    (void)context;
    (void)tilingData;
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief FlashAttention Tiling主入口
 */
ge::graphStatus TilingFlashAttention(gert::TilingContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }

    OP_LOGI(context, "FlashAttention TilingFlashAttention start.");

    FlashAttentionTilingData tilingData;

    // 优先尝试从metadata中反序列化预计算的tiling（推理场景）
    bool metadataUsed = TryDeserializeFromMetadata(context, tilingData);

    if (!metadataUsed) {
        // metadata为空或无效，自主计算tiling（训练场景或在线推理）
        auto ret = ComputeTiling(context, tilingData);
        if (ret != ge::GRAPH_SUCCESS) {
            OP_LOGE(context, "FlashAttention ComputeTiling failed.");
            return ret;
        }
    }

    // 将tilingData写入TilingContext供kernel使用
    // TODO: context->SetTilingData(tilingData);

    // 设置workspace大小
    // TODO: context->SetWorkspaceSize(tilingData.workspaceSize);

    OP_LOGI(context, "FlashAttention TilingFlashAttention done. tilingKey=%u.", tilingData.tilingKey);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_TILING(FlashAttention).Tiling(TilingFlashAttention);

}  // namespace optiling
