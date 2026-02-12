/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <torch/library.h>
#include "ops_common.h"
#include <iostream>

namespace custom {
using namespace at_npu::native;
using npu_preparation = at_npu::native::OpPreparation;

const int LAYOUT_MAX_LENGTH = 20;
// 为NPU设备实现前向接口
at::Tensor npu_incre_flash_attention_npu(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
    const c10::optional<at::Tensor> &padding_mask, const c10::optional<at::Tensor> &atten_mask,
    const c10::optional<at::Tensor> &pse_shift,
    c10::OptionalIntArrayRef actual_seq_lengths, const c10::optional<at::Tensor> &antiquant_scale,
    const c10::optional<at::Tensor> &antiquant_offset, const c10::optional<at::Tensor> &block_table,
    const c10::optional<at::Tensor> &dequant_scale1, const c10::optional<at::Tensor> &quant_scale1,
    const c10::optional<at::Tensor> &dequant_scale2, const c10::optional<at::Tensor> &quant_scale2,
    const c10::optional<at::Tensor> &quant_offset2, const c10::optional<at::Tensor> &kv_padding_size,
    int64_t num_heads, double scale_value, c10::string_view input_layout, int64_t num_key_value_heads,
    int64_t block_size, int64_t inner_precise)
{
    // construct the output tensor of the NPU
    at::Tensor output;
    if (quant_scale2.has_value()) {
        output = at::empty(query.sizes(), c10::dtype(c10::ScalarType::Char));
    } else if (query.dtype() == at::kChar) {
        output = at::empty(query.sizes(), c10::dtype(c10::ScalarType::Half));
    } 
    // else {
    //     output = at::empty(query);
    // }
    at::TensorList keyTensors = key;
    at::TensorList valueTensors = value;

    // auto actual_seq_lengths_ = actual_seq_lengths.value_or(at::IntArrayRef{});

    std::string input_layout_str = std::string(input_layout);
    char input_layout_char[LAYOUT_MAX_LENGTH];
    strncpy(input_layout_char, input_layout_str.c_str(), LAYOUT_MAX_LENGTH - 1);
    printf("aaaaaaaaaaaa\n");
    // std::cout << "query" << &query << std::endl;
    // std::cout << "key" << &key << std::endl;
    // std::cout << "value" << &value << std::endl;
    // std::cout << "padding_mask" << &padding_mask << std::endl;
    // std::cout << "atten_mask" << &atten_mask << std::endl;
    // std::cout << "pse_shift" << &pse_shift << std::endl;
    // std::cout << "antiquant_scale" << &antiquant_scale << std::endl;
    // std::cout << "antiquant_offset" << &antiquant_offset << std::endl;
    // std::cout << "block_table" << &block_table << std::endl;
    // std::cout << "dequant_scale1" << &dequant_scale1 << std::endl;
    // std::cout << "quant_scale1" << &quant_scale1 << std::endl;
    // std::cout << "dequant_scale2" << &dequant_scale2 << std::endl;
    // std::cout << "quant_scale2" << &quant_scale2 << std::endl;
    // std::cout << "quant_offset2" << &quant_offset2 << std::endl;
    // std::cout << "kv_padding_size" << &kv_padding_size << std::endl;
    // std::cout << "num_heads" << num_heads << std::endl;
    // std::cout << "scale_value" << scale_value << std::endl;
    // std::cout << "input_layout" << input_layout << std::endl;
    // std::cout << "num_key_value_heads" << num_key_value_heads << std::endl;
    // std::cout << "block_size" << block_size << std::endl;
    // std::cout << "inner_precise" << inner_precise << std::endl;
    // dispatch hostAPI
    EXEC_NPU_CMD_V1(aclnnIncreFlashAttentionV4, query, keyTensors, valueTensors, pse_shift, atten_mask,
        actual_seq_lengths, dequant_scale1, quant_scale1, dequant_scale2, quant_scale2, quant_offset2, antiquant_scale,
        antiquant_offset, block_table, kv_padding_size, num_heads, scale_value, input_layout_char,
        num_key_value_heads, block_size, inner_precise, output);
    return output;
}

// 为META设备实现前向接口
at::Tensor npu_incre_flash_attention_meta(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
    const c10::optional<at::Tensor> &padding_mask, const c10::optional<at::Tensor> &atten_mask,
    const c10::optional<at::Tensor> &pse_shift,
    c10::OptionalArrayRef<c10::SymInt> actual_seq_lengths, const c10::optional<at::Tensor> &antiquant_scale,
    const c10::optional<at::Tensor> &antiquant_offset, const c10::optional<at::Tensor> &block_table,
    const c10::optional<at::Tensor> &dequant_scale1, const c10::optional<at::Tensor> &quant_scale1,
    const c10::optional<at::Tensor> &dequant_scale2, const c10::optional<at::Tensor> &quant_scale2,
    const c10::optional<at::Tensor> &quant_offset2, const c10::optional<at::Tensor> &kv_padding_size,
    int64_t num_heads, double scale_value, c10::string_view input_layout, int64_t num_key_value_heads,
    int64_t block_size, int64_t inner_precise)
{
    at::Tensor output;
    if (quant_scale2.has_value()) {
        output = at::empty(query.sizes(), c10::dtype(c10::ScalarType::Char));
    } else if (query.dtype() == at::kChar) {
        output = at::empty(query.sizes(), c10::dtype(c10::ScalarType::Half));
    } 
    // else {
    //     output = at::empty(query);
    // }
    return output;
}
}  // namespace custom

// 为NPU设备注册前向实现
TORCH_LIBRARY_IMPL(custom, PrivateUse1, m) {
    m.impl("npu_incre_flash_attention", &custom::npu_incre_flash_attention_npu);
}

// 为META设备注册前向实现
TORCH_LIBRARY_IMPL(custom, Meta, m) {
    m.impl("npu_incre_flash_attention", &custom::npu_incre_flash_attention_meta);
}
