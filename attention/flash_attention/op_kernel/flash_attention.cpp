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
 * \brief FlashAttention Kernel主入口（框架，参照flash_attention_score结构）
 *
 * 通过ASCENDC_TPL机制：框架根据tiling侧设置的tilingKey自动选择对应的模板特化。
 * 每个tilingKey组合对应arch35/flash_attention_entry_regbase.h中的flash_attention_regbase函数。
 *
 * 支持场景（非量化）：
 *   - 训练正向传播（isSoftmaxLse=1，输出softmax_lse）
 *   - 推理（isSoftmaxLse=0，不输出softmax_lse）
 *   - 分页注意力PA（isPA=1，通过block_table管理KV cache）
 *   - 变长序列（通过cu_seqlens/seqused指定实际长度）
 *   - 空tensor场景（tilingKey=1，执行填0操作）
 */

#include "kernel_operator.h"
#include "arch35/flash_attention_entry_regbase.h"
#include "arch35/flash_attention_template_tiling_key.h"

using namespace AscendC;

// kernel主入口：框架根据tilingKey自动分发到flash_attention_regbase对应的模板特化
ASCENDC_TPL_KERNEL_EXTERN_C void flash_attention(
    __gm__ uint8_t *query,
    __gm__ uint8_t *key,
    __gm__ uint8_t *value,
    __gm__ uint8_t *blockTable,
    __gm__ uint8_t *actualSeqLengthsQ,
    __gm__ uint8_t *actualSeqLengthsKv,
    __gm__ uint8_t *sequsedQ,
    __gm__ uint8_t *sequsedKv,
    __gm__ uint8_t *sinks,
    __gm__ uint8_t *metadata,
    __gm__ uint8_t *attentionOut,
    __gm__ uint8_t *softmaxLse,
    __gm__ uint8_t *workspace,
    __gm__ uint8_t *tiling)
{
    // 框架根据tilingKey模板参数分发至flash_attention_regbase<...>()
    ASCENDC_TPL_KERNEL_DISPATCH(FlashAttention,
        flash_attention_regbase<
            GET_TPL_KEY_FIELD(ImplMode),
            GET_TPL_KEY_FIELD(Layout),
            GET_TPL_KEY_FIELD(S1TemplateType),
            GET_TPL_KEY_FIELD(S2TemplateType),
            GET_TPL_KEY_FIELD(DTemplateType),
            GET_TPL_KEY_FIELD(DvTemplateType),
            GET_TPL_KEY_FIELD(HasAtten) != 0,
            GET_TPL_KEY_FIELD(IsPA) != 0,
            GET_TPL_KEY_FIELD(IsSoftmaxLse) != 0,
            GET_TPL_KEY_FIELD(Regbase)
        >(
            query, key, value, blockTable,
            actualSeqLengthsQ, actualSeqLengthsKv,
            sequsedQ, sequsedKv, sinks, metadata,
            attentionOut, softmaxLse, workspace, tiling
        )
    );
}
