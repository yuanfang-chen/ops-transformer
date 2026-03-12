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
#include "incre_flash_attention_preload_dd.h"

template <class T>
__inline__ __attribute__((always_inline)) __aicore__ void InitMetaData(const __gm__ uint8_t *p_metadata, T *metadata)
{
    constexpr uint64_t all_bytes = sizeof(T);
#if defined(ASCENDC_CPU_DEBUG) || defined(__DAV_C220_CUBE__) || defined(__DAV_C310_CUBE__) || defined(__DAV_310R6_CUBE__) || defined(__GET_CODE_CHANNEL__)
    copy_data_align64((uint8_t*)metadata, (__gm__ uint8_t *)p_metadata, all_bytes);
#else
    __ubuf__ uint8_t *metadata_in_ub = (__ubuf__ uint8_t *)get_imm(0);
    constexpr uint32_t len_burst = (all_bytes + 31) / 32;
    copy_gm_to_ubuf(((__ubuf__ uint8_t *)metadata_in_ub), p_metadata, 0, 1,len_burst, 0, 0);
    set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
    copy_data_align64((uint8_t*)metadata, (__ubuf__ uint8_t *)metadata_in_ub, all_bytes);
#endif

}

#define INVOKE_IFA_NO_KFC_DD_OP_IMPL(templateClass, ...)                                                               \
    do {                                                                                                               \
        templateClass<IFAType<__VA_ARGS__>> op;                                                                        \
        const optiling::IncreFlashAttentionTilingData *__restrict tiling_data = &tiling;                                 \
        const IncreFlashAttentionMetaData *__restrict meta_data = nullptr;                                        \
        IncreFlashAttentionMetaData metaDataTmp;                                                            \
        if (metaData != nullptr) {                                                          \
            InitMetaData<IncreFlashAttentionMetaData>(metaData, &metaDataTmp);                              \
            meta_data = &metaDataTmp;                                                       \
        }                                                                                   \
        op.Init(query, key, value, pseShift, attenMask, actualSeqLengthsQ, actualSeqLengths, blocktable, kvPaddingSize,\
                meta_data, attentionOut, softmaxLse, user, tiling_data, &tPipe);                                          \
        op.InitQuant(deqScale1, quantScale1, deqScale2, quantScale2, quantOffset2, antiquantScale, antiquantOffset,    \
                     keyAntiquantScale, keyAntiquantOffset, valueAntiquantScale, valueAntiquantOffset, user);          \ 
        op.Process();                                                                                                   \
    } while (0)


template <uint8_t FLASH_DECODE, uint8_t LAYOUT_T, uint8_t ANTIQUANT_MODE>
__global__ __mix__(1, 2) void incre_flash_attention(
    __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
    __gm__ uint8_t *pseShift, __gm__ uint8_t *attenMask, __gm__ uint8_t *actualSeqLengthsQ,
    __gm__ uint8_t *actualSeqLengths, __gm__ uint8_t *deqScale1, __gm__ uint8_t *quantScale1,
    __gm__ uint8_t *deqScale2, __gm__ uint8_t *quantScale2, __gm__ uint8_t *quantOffset2,
    __gm__ uint8_t *antiquantScale, __gm__ uint8_t *antiquantOffset, __gm__ uint8_t *blocktable,
    __gm__ uint8_t *queryPaddingSize, __gm__ uint8_t *kvPaddingSize, __gm__ uint8_t *keyAntiquantScale,
    __gm__ uint8_t *keyAntiquantOffset, __gm__ uint8_t *valueAntiquantScale, __gm__ uint8_t *valueAntiquantOffset,
    __gm__ uint8_t *keySharedPrefix, __gm__ uint8_t *valueSharedPrefix, __gm__ uint8_t *actualSharedPrefixLen,
    __gm__ uint8_t *queryRope, __gm__ uint8_t *keyRope, __gm__ uint8_t *keyRopeAntiquantScale,
    __gm__ uint8_t *dequantScaleQuery, __gm__ uint8_t *metaData, 
    __gm__ uint8_t *attentionOut, __gm__ uint8_t *softmaxLse,
    __gm__ uint8_t *workspace, optiling::IncreFlashAttentionTilingData tiling)
{
    printf("WW FA ENTRY SUCCESSFUL!!!\n");
    TPipe tPipe;
    __gm__ uint8_t *user = workspace; 

    INVOKE_IFA_NO_KFC_DD_OP_IMPL(IncreFlashAttentionAttenPreloadDD, bfloat16_t, int8_t, bfloat16_t,
                                    bfloat16_t, true, FLASH_DECODE, static_cast<LAYOUT>(LAYOUT_T), ANTIQUANT_MODE, false, LAYOUT::NZ, AMLAMODE::NORMAL);
}