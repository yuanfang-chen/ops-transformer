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
 * \file incre_flash_attention_arch32.h
 * \brief
 */
#if (__CCE_AICORE__ > 200)
#include "./arch32/incre_flash_attention_preload_dd.h"
#include "incre_flash_attention_tilingkey.h"
#endif

#define NEED_CUBE_TILING (true)
#define NOT_NEED_CUBE_TILING (false)


#define INVOKE_IFA_NO_KFC_DD_OP_IMPL(templateClass, ...)                                                               \
    do {                                                                                                               \
        templateClass<IFAType<__VA_ARGS__>> op;                                                                        \
        COPY_TILING_DATA_ALL(tiling);                                                                                  \
        op.Init(query, key, value, pseShift, attenMask, actualSeqLengthsQ, actualSeqLengths, blocktable, kvPaddingSize,\
                attentionOut, softmaxLse, user, tiling_data, tiling, &tPipe);                                          \
        op.InitQuant(deqScale1, quantScale1, deqScale2, quantScale2, quantOffset2, antiquantScale, antiquantOffset,    \
                     keyAntiquantScale, keyAntiquantOffset, valueAntiquantScale, valueAntiquantOffset, user);          \
        op.Process();                                                                                                  \
    } while (0)


#define COPY_TILING_DATA_ALL(tiling)                                                                                   \
    GET_TILING_DATA_MEMBER(optiling::IncreFlashAttentionTilingDataV2, tilingBase, tiling_data_in, tiling);                       \
    const optiling::IncreFlashAttentionTilingData *__restrict tiling_data = &tiling_data_in;                                     \
    const TCubeTiling *__restrict bmm1tiling = nullptr;                                                                \
    const TCubeTiling *__restrict bmm2tiling = nullptr

template <uint8_t FLASH_DECODE, uint8_t LAYOUT_T, uint8_t ANTIQUANT_MODE>
inline __aicore__ void incre_flash_attention_FIAS_arch32(
    __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
    __gm__ uint8_t *pseShift, __gm__ uint8_t *attenMask, __gm__ uint8_t *actualSeqLengthsQ,
    __gm__ uint8_t *actualSeqLengths, __gm__ uint8_t *deqScale1, __gm__ uint8_t *quantScale1,
    __gm__ uint8_t *deqScale2, __gm__ uint8_t *quantScale2, __gm__ uint8_t *quantOffset2,
    __gm__ uint8_t *antiquantScale, __gm__ uint8_t *antiquantOffset, __gm__ uint8_t *blocktable,
    __gm__ uint8_t *queryPaddingSize, __gm__ uint8_t *kvPaddingSize, __gm__ uint8_t *keyAntiquantScale,
    __gm__ uint8_t *keyAntiquantOffset, __gm__ uint8_t *valueAntiquantScale, __gm__ uint8_t *valueAntiquantOffset,
    __gm__ uint8_t *keySharedPrefix, __gm__ uint8_t *valueSharedPrefix, __gm__ uint8_t *actualSharedPrefixLen,
    __gm__ uint8_t *queryRope, __gm__ uint8_t *keyRope, __gm__ uint8_t *keyRopeAntiquantScale,
    __gm__ uint8_t *dequantScaleQuery, __gm__ uint8_t *attentionOut, __gm__ uint8_t *softmaxLse,
    __gm__ uint8_t *workspace, __gm__ uint8_t *tiling)
{
    REGISTER_TILING_DEFAULT(optiling::IncreFlashAttentionTilingDataV2);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    TPipe tPipe;
    __gm__ uint8_t *user = GetUserWorkspace(workspace);

#if (__CCE_AICORE__ > 200) // new template

#if (ORIG_DTYPE_QUERY == DT_BF16) && (ORIG_DTYPE_ATTENTION_OUT == DT_BF16) && (ORIG_DTYPE_KEY == DT_INT8)
        INVOKE_IFA_NO_KFC_DD_OP_IMPL(IncreFlashAttentionAttenPreloadDD, bfloat16_t, int8_t, bfloat16_t,
                                     bfloat16_t, true, FLASH_DECODE, static_cast<LAYOUT>(LAYOUT_T), ANTIQUANT_MODE, false, LAYOUT::NZ, AMLAMODE::NORMAL);
    
#endif
#endif // new template
}