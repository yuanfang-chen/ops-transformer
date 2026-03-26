/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fused_infer_attention_score_tiling_index.h
 * \brief
 */

#ifndef FUSED_INFER_ATTENTION_SCORE_TILING_INDEX_H
#define FUSED_INFER_ATTENTION_SCORE_TILING_INDEX_H
// #include "../../prompt_flash_attention/op_host/prompt_flash_attention_tiling.h"
// #include "../../incre_flash_attention/op_host/incre_flash_attention_tiling.h"
// #include "register/tilingdata_base.h"
#include <cstdint>
#include <string>

typedef std::uint8_t uint8_t;
typedef std::int8_t int8_t;
typedef std::uint16_t uint16_t;
typedef std::int16_t int16_t;
typedef std::uint32_t uint32_t;
typedef std::int32_t int32_t;
typedef std::uint64_t uint64_t;
typedef std::int64_t int64_t;

const std::string ACTUAL_SEQ_KV_LEN_NAME = "the key/value's actual sequence lengths";
const std::string ACTUAL_SEQ_Q_LEN_NAME = "the query's actual sequence lengths";
const std::string ATTEN_MASK_NAME = "atten_mask";
const std::string ATTEN_OUT_NAME = "attention_out";
const std::string BLOCK_SIZE_NAME = "block_size";
const std::string BLOCK_TABLE_NAME = "block_table";
const std::string DEQUANT_SCALE_QUERY_NAME = "the query's dequant scale";
const std::string INNER_PRECISE_NAME = "inner_precise";
const std::string KEY_NAME = "key";
const std::string KEY_ANTIQUANT_MODE_NAME = "the key's quant mode";
const std::string KEY_ANTIQUANT_OFFSET_NAME = "the key's quant offset";
const std::string KEY_ANTIQUANT_SCALE_NAME = "the key's quant scale";
const std::string KEY_ROPE_NAME = "key_rope";
const std::string KEY_ROPE_ANTIQUANT_SCALE_NAME = "the key_rope's dequant scale";
const std::string KV_HEADS_NUM_NAME = "the key/value's heads num";
const std::string NEXT_TOKENS_NAME = "next_tokens";
const std::string PRE_TOKENS_NAME = "pre_tokens";
const std::string PSE_SHIFT_NAME = "pse_shift";
const std::string QUANT_OFFSET2_NAME = "the output's dequant offset";
const std::string QUANT_SCALE2_NAME = "the output's dequant scale";
const std::string QUERY_NAME = "query";
const std::string QUERY_HEADS_NUM_NAME = "the query's heads num";
const std::string QUERY_QUANT_MODE_NAME = "the query's quant mode";
const std::string QUERY_ROPE_NAME = "query_rope";
const std::string SOFTMAX_SCALE_NAME = "the softmax's scale";
const std::string SPARSE_MODE_NAME = "sparse_mode";
const std::string VALUE_NAME = "value";
const std::string VALUE_ANTIQUANT_MODE_NAME = "the value's quant mode";
const std::string VALUE_ANTIQUANT_OFFSET_NAME = "the value's dequant offset";
const std::string VALUE_ANTIQUANT_SCALE_NAME = "the value's dequant scale";

const std::string ANTIQUANT_MODE_NAME = "antiquant_mode";
const std::string ANTIQUANT_SCALE_NAME = "antiquant_scale";
const std::string ANTIQUANT_OFFSET_NAME = "antiquant_offset";
const std::string DEQUANT_SCALE1_NAME = "dequant_scale1";
const std::string DEQUANT_SCALE2_NAME = "dequant_scale2";
const std::string KEY_SHARED_PREFIX_NAME = "key_shared_prefix";
const std::string KV_PADDING_SIZE_NAME = "kv_padding_size";
const std::string QUANT_SCALE1_NAME = "quant_scale1";
const std::string QUERY_PADDING_SIZE_NAME = "query_padding_size";
const std::string SOFTMAX_LSE_NAME = "softmax_lse";
const std::string VALUE_SHARED_PREFIX_NAME = "value_shared_prefix";
const std::string ACTUAL_SHARED_PREFIX_LEN_NAME = "actual_shared_prefix_len";
const std::string LEARNABLE_SINK_NAME = "learnable_sink";
namespace optiling {
// Inputs Index
constexpr uint32_t QUERY_INDEX = 0;
constexpr uint32_t KEY_INDEX = 1;
constexpr uint32_t VALUE_INDEX = 2;
constexpr uint32_t PSE_SHIFT_INDEX = 3;
constexpr uint32_t ATTEN_MASK_INDEX = 4;
constexpr uint32_t ACTUAL_SEQ_Q_INDEX = 5;
constexpr uint32_t ACTUAL_SEQ_KV_INDEX = 6;
constexpr uint32_t DEQUANT_SCALE1_INDEX = 7;
constexpr uint32_t QUANT_SCALE1_INDEX = 8;
constexpr uint32_t DEQUANT_SCALE2_INDEX = 9;
constexpr uint32_t QUANT_SCALE2_INDEX = 10;
constexpr uint32_t QUANT_OFFSET2_INDEX = 11;
constexpr uint32_t ANTIQUANT_SCALE_INDEX = 12;
constexpr uint32_t ANTIQUANT_OFFSET_INDEX = 13;
constexpr uint32_t BLOCK_TABLE_INDEX = 14;
constexpr uint32_t QUERY_PADDING_SIZE_INDEX = 15;
constexpr uint32_t KV_PADDING_SIZE_INDEX = 16;
constexpr uint32_t KEY_ANTIQUANT_SCALE_INDEX = 17;
constexpr uint32_t KEY_ANTIQUANT_OFFSET_INDEX = 18;
constexpr uint32_t VALUE_ANTIQUANT_SCALE_INDEX = 19;
constexpr uint32_t VALUE_ANTIQUANT_OFFSET_INDEX = 20;
constexpr uint32_t KEY_SHARED_PREFIX_INDEX = 21;
constexpr uint32_t VALUE_SHARED_PREFIX_INDEX = 22;
constexpr uint32_t ACTUAL_SHARED_PREFIX_LEN_INDEX = 23;
constexpr uint32_t QUERY_ROPE_INDEX = 24;
constexpr uint32_t KEY_ROPE_INDEX = 25;
constexpr uint32_t KEY_ROPE_ANTIQUANT_SCALE_INDEX = 26;
constexpr uint32_t DEQUANT_SCALE_QUERY_INDEX = 27;
constexpr uint32_t LEARNABLE_SINK_INDEX = 28;
constexpr uint32_t Q_START_IDX_INDEX = 29;
constexpr uint32_t KV_START_IDX_INDEX = 30;

// Attributes Index
constexpr uint32_t ATTR_N_INDEX = 0;
constexpr uint32_t ATTR_SCALE_INDEX = 1;
constexpr uint32_t ATTR_PRE_TOKEN_INDEX = 2;
constexpr uint32_t ATTR_NEXT_TOKEN_INDEX = 3;
constexpr uint32_t ATTR_INPUT_LAYOUT_INDEX = 4;
constexpr uint32_t ATTR_NUM_KV_HEADS_INDEX = 5;
constexpr uint32_t ATTR_SPARSE_MODE_INDEX = 6;
constexpr uint32_t ATTR_INNER_PRECISE_INDEX = 7;
constexpr uint32_t ATTR_BLOCK_SIZE_INDEX = 8;
constexpr uint32_t ANTIQUANT_MODE_INDEX = 9;
constexpr uint32_t SOFTMAX_LSE_FLAG_INDEX = 10;
constexpr uint32_t KEY_ANTIQUANT_MODE_INDEX = 11;
constexpr uint32_t VALUE_ANTIQUANT_MODE_INDEX = 12;
constexpr uint32_t QUERY_QUANT_MODE_INDEX = 13;
constexpr uint32_t PSE_TYPE_INDEX = 14;
constexpr uint32_t PSE_SHIFT_S1_INDEX = 2;
constexpr uint32_t PSE_SHIFT_S2_INDEX = 3;

// Output Index
constexpr uint32_t ATTENTION_OUT_INDEX = 0;
constexpr uint32_t SOFTMAX_LSE_INDEX = 1;
} // namespace optiling

#endif // FUSED_INFER_ATTENTION_SCORE_TILING_INDEX_H