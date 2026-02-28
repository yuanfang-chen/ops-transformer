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
// #include "../../../incre_flash_attention/op_host/incre_flash_attention_tiling.h"

namespace custom {
const static int FLASH_THRESHOLD = 512;
const static int64_t PFA_SPARSE_HIGH_PRECISION_NO_MASK = 10;
const static int64_t PFA_SPARSE_HIGH_PRECISION_BAND = 14;
const static int64_t DIM_0 = 0;
const static int64_t DIM_1 = 1;
const static int64_t DIM_2 = 2;
const static int64_t DIM_3 = 3;
const static int64_t DIM_4 = 4;
const static int64_t PA_BBH_DIMS = 3;
const static int64_t PA_BNBD_DIMS = 4;
const static int64_t PA_NZ_DIMS = 5;
using namespace at_npu::native;
using npu_preparation = at_npu::native::OpPreparation;

std::tuple<at::Tensor, at::Tensor> construct_fia_output_tensor_v2(
    const at::Tensor &query,
    const at::Tensor &value,
    std::string input_layout_str,
    const c10::optional<at::Tensor> &quant_scale_out,
    const c10::optional<at::Tensor> &block_table,
    int64_t num_query_heads,
    int64_t num_key_value_heads, // 增加kvhead用于计算BBH情况下的D
    bool return_softmax_lse,
    const c10::optional<at::Tensor> &query_rope)
{
    at::Tensor output;
    int64_t batchSize = 1;
    int64_t qsSize = 1;
    at::Tensor tmp_output = at::empty_like(query);
    if (input_layout_str == "BNSD_BSND") {
        tmp_output = at::empty({query.size(DIM_0), query.size(DIM_2), query.size(DIM_1), query.size(DIM_3)},
            query.options().dtype(query.dtype()));
        batchSize = query.size(DIM_0);
        qsSize = query.size(DIM_2);
    } else if (input_layout_str == "BNSD_NBSD") {
        tmp_output = at::empty(
            {query.size(DIM_1), query.size(DIM_0), query.size(DIM_2), query.size(DIM_3)},
            query.options().dtype(query.dtype()));
        batchSize = query.size(DIM_0);
        qsSize = query.size(DIM_2);
    } else if (input_layout_str == "BSND_NBSD") {
        tmp_output = at::empty(
            {query.size(DIM_2), query.size(DIM_0), query.size(DIM_1), query.size(DIM_3)},
            query.options().dtype(query.dtype()));
        batchSize = query.size(DIM_0);
        qsSize = query.size(DIM_1);
    } else if (input_layout_str == "BSH_NBSD") {
        // TORCH_CHECK(num_query_heads > 0, "The num_query_heads should be greater than zero, but got ", num_query_heads, OPS_ERROR(ErrCode::PARAM));
        tmp_output = at::empty(
            {num_query_heads, query.size(DIM_0), query.size(DIM_1), query.size(DIM_2) / num_query_heads},
            query.options().dtype(query.dtype()));
        batchSize = query.size(DIM_0);
        qsSize = query.size(DIM_1);
    } else if (input_layout_str == "TND_NTD") {
        tmp_output = at::empty(
            {query.size(DIM_1), query.size(DIM_0), query.size(DIM_2)},
            query.options().dtype(query.dtype()));
    } else if (input_layout_str == "NSD") {
        batchSize = 1;
        qsSize = query.size(DIM_1);
    } else if (input_layout_str == "BSH") {
        batchSize = query.size(DIM_0);
        qsSize = query.size(DIM_1);
    } else if (input_layout_str == "BSND") {
        batchSize = query.size(DIM_0);
        qsSize = query.size(DIM_1);
    } else if (input_layout_str == "BNSD") {
        batchSize = query.size(DIM_0);
        qsSize = query.size(DIM_2);
    } else if (input_layout_str == "TND") {
        int64_t kv_dim = value.dim();
        if (block_table.has_value()) { // IFA目前TND只支持PA场景，PFA目前TND只支持非PA场景
            if (kv_dim == PA_BBH_DIMS) { // BBH的情况下，D = H / N
                tmp_output = at::empty(
                    {query.size(DIM_0), query.size(DIM_1), value.size(DIM_2) / num_key_value_heads},
                    query.options().dtype(query.dtype()));
            } else if (kv_dim == PA_BNBD_DIMS) { // BNBD情况下取D
                tmp_output = at::empty(
                    {query.size(DIM_0), query.size(DIM_1), value.size(DIM_3)},
                    query.options().dtype(query.dtype()));
            } else if (kv_dim == PA_NZ_DIMS) { // blockNum, N, D / 16, blockSize, 16取DIM2*DIM4
                tmp_output = at::empty(
                    {query.size(DIM_0), query.size(DIM_1), value.size(DIM_2) * value.size(DIM_4)},
                    query.options().dtype(query.dtype()));
            } else {
                tmp_output = at::empty(
                    {query.size(DIM_0), query.size(DIM_1), value.size(DIM_2)},
                    query.options().dtype(query.dtype()));
            }
        } else {
            tmp_output = at::empty(
                {query.size(DIM_0), query.size(DIM_1), value.size(DIM_2)},
                query.options().dtype(query.dtype()));
        }
    } else if (input_layout_str == "NTD_TND") {
        int64_t kv_dim = value.dim();
        if (kv_dim == 0) {
            kv_dim = query.dim();
        }
        if (block_table.has_value()) { // pa场景
            if (kv_dim == PA_BBH_DIMS) { // BBH的情况下，D = H / N
                tmp_output = at::empty(
                    {query.size(DIM_1), query.size(DIM_0), value.size(DIM_2) / num_key_value_heads},
                    query.options().dtype(query.dtype()));
            } else if (kv_dim == PA_BNBD_DIMS) { // BNBD情况下取D
                tmp_output = at::empty(
                    {query.size(DIM_1), query.size(DIM_0), value.size(DIM_3)},
                    query.options().dtype(query.dtype()));
            } else if (kv_dim == PA_NZ_DIMS) { // blockNum, N, D / 16, blockSize, 16取DIM2*DIM4
                tmp_output = at::empty(
                    {query.size(DIM_1), query.size(DIM_0), value.size(DIM_2) * value.size(DIM_4)},
                    query.options().dtype(query.dtype()));
            } else {
                tmp_output = at::empty(
                    {query.size(DIM_1), query.size(DIM_0), value.size(DIM_2)},
                    query.options().dtype(query.dtype()));
            }
        } else {
            tmp_output = at::empty(
                {query.size(DIM_1), query.size(DIM_0), value.size(DIM_2)},
                query.options().dtype(query.dtype()));
        }
    }
    if (quant_scale_out.has_value()) {
        output = at::empty(tmp_output.sizes(), c10::dtype(c10::ScalarType::Char));
    } else if (query.dtype() == at::kChar) {
        if (query_rope.has_value()) {
            const at::Tensor &query_rope_tensor = c10::value_or_else(query_rope, [] { return at::Tensor(); });
            output = at::empty(tmp_output.sizes(), c10::dtype(query_rope_tensor.dtype()));
        } else {
            output = at::empty(tmp_output.sizes(), c10::dtype(c10::ScalarType::Half));
        }
    } else {
        output = at::empty_like(tmp_output);
    }

    auto lse_opts = output.options().dtype(c10::ScalarType::Float);
    at::Tensor softmax_lse;
    if (input_layout_str == "TND") {
        if (block_table.has_value()) { // IFA目前TND只支持PA场景，PFA目前TND只支持非PA场景
            if (query.size(DIM_2) == 0) { // 增加softmax lse的情况下，可能存在空tensor的分支
                softmax_lse = at::empty({query.size(DIM_0), num_query_heads, 0,},
                                         lse_opts);
            } else {
                softmax_lse = at::empty({query.size(DIM_0), num_query_heads, 1},
                                         lse_opts);
            }
        } else {
            softmax_lse = at::empty({query.size(DIM_0), query.size(DIM_1), 1},
                                     lse_opts);
        }
    } else if (input_layout_str == "NTD_TND") {
        if (block_table.has_value()) { // pa场景
            if (query.size(DIM_2) == 0) { // 增加softmax lse的情况下，可能存在空tensor的分支
                softmax_lse = at::empty({query.size(DIM_1), query.size(DIM_0), 0},
                                         lse_opts);
            } else {
                softmax_lse = at::empty({query.size(DIM_1), query.size(DIM_0), 1},
                                         lse_opts);
            }
        } else {
            softmax_lse = at::empty({query.size(DIM_1), query.size(DIM_0), 1},
                                     lse_opts);
        }
    } else {
        softmax_lse = at::empty({batchSize, num_query_heads, qsSize, 1},
                                 lse_opts);
    }

    if (!return_softmax_lse) {
        softmax_lse = at::empty({0}, lse_opts);
    }
    return std::tuple<at::Tensor, at::Tensor>(output, softmax_lse);
}

// void ConvertContextToParamsIFA(
//     IFAContext &ifaContext, 
//     const at::Tensor query, const at::Tensor key, const at::Tensor value,
//     const c10::optional<at::Tensor> query_rope,
//     const c10::optional<at::Tensor> key_rope,
//     const c10::optional<at::Tensor> pse_shift,
//     const c10::optional<at::Tensor> atten_mask,
//     c10::OptionalIntArrayRef actual_seq_qlen,
//     c10::OptionalIntArrayRef actual_seq_kvlen,
//     const c10::optional<at::Tensor> block_table,
//     const c10::optional<at::Tensor> dequant_scale_query,
//     const c10::optional<at::Tensor> dequant_scale_key,
//     const c10::optional<at::Tensor> dequant_offset_key,
//     const c10::optional<at::Tensor> dequant_scale_value,
//     const c10::optional<at::Tensor> dequant_offset_value,
//     const c10::optional<at::Tensor> dequant_scale_key_rope,
//     const c10::optional<at::Tensor> quant_scale_out,
//     const c10::optional<at::Tensor> quant_offset_out,
//     const c10::optional<at::Tensor> learnable_sink,
//     int64_t num_query_heads, int64_t num_key_value_heads, double softmax_scale,
//     int64_t pre_tokens, int64_t next_tokens, c10::string_view input_layout,
//     int64_t sparse_mode, int64_t block_size,
//     int64_t query_quant_mode, int64_t key_quant_mode, int64_t value_quant_mode,
//     int64_t inner_precise, bool return_softmax_lse,
//     c10::optional<int64_t> query_dtype, c10::optional<int64_t> key_dtype, c10::optional<int64_t> value_dtype,
//     c10::optional<int64_t> query_rope_dtype, c10::optional<int64_t> key_rope_dtype,
//     c10::optional<int64_t> key_shared_prefix_dtype, c10::optional<int64_t> value_shared_prefix_dtype,
//     c10::optional<int64_t> dequant_scale_query_dtype, c10::optional<int64_t> dequant_scale_key_dtype,
//     c10::optional<int64_t> dequant_scale_value_dtype, c10::optional<int64_t> dequant_scale_key_rope_dtype)
// {
//     ifaContext.opName = "FusedInferAttentionScore"
//     query = query;
//     key = key;
//     value = value;
//     pseShift = pse_shift;
//     attenMask = atten_mask;
//     actualSeqLengthsQ = actual_seq_qlen;
//     actualSeqLengths = actual_seq_kvlen;
//     deqScale1 = c10::nullopt;
//     quantScale1 = c10::nullopt;
//     deqScale2 = c10::nullopt;
//     quantScale2 = c10::nullopt;
//     quantOffset2 = c10::nullopt;
//     antiquantScale = c10::nullopt;
//     antiquantOffset = c10::nullopt;
//     blockTable = block_table;
//     queryPaddingSize = 0;
//     kvPaddingSize = 0;
//     keyAntiquantScale = dequant_scale_key;
//     keyAntiquantOffset = dequant_offset_key;
//     valueAntiquantScale = dequant_scale_value;
//     valueAntiquantOffset = dequant_offset_value;
//     keySharedPrefix = c10::nullopt;
//     valueSharedPrefix = c10::nullopt;
//     actualSharedPrefixLen = c10::nullopt;
//     queryRope = query_rope;
//     keyRope = key_rope;
//     keyRopeAntiquantScale = dequant_scale_key_rope;
//     dequantScaleQuery = dequant_scale_query;
//     qStartIdx = c10::nullopt;
//     kvStartIdx = c10::nullopt;

//     // at::Tensor attenOut;
//     // at::Tensor lseOut;

//     uint32_t numHeads = num_query_heads;
//     int64_t preToken = pre_tokens;
//     int64_t nextToken = next_tokens;
//     float scaleValue= softmax_scale;
//     uint32_t kvHeadNums = num_key_value_heads;
//     char layOut = input_layout;
//     uint32_t blockSize = block_size;
//     uint32_t innerPrecise = inner_precise;
//     int64_t antiquantMode = 0;
//     bool softmaxLseFlag = return_softmax_lse;
//     int64_t keyAntiquantMode = key_quant_mode;
//     int64_t valueAntiquantMode value_quant_mode;
//     uint32_t sparseMode = sparse_mode;
//     int64_t queryQuantMode = query_quant_mode;
//     int64_t pseType = 0;
//     int64_t windowSize = 0;
//     return ge::GRAPH_SUCCESS;
// }

std::tuple<at::Tensor, at::Tensor> npu_fused_infer_attention_score_npu(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
    const c10::optional<at::Tensor> &query_rope,
    const c10::optional<at::Tensor> &key_rope,
    const c10::optional<at::Tensor> &pse_shift,
    const c10::optional<at::Tensor> &atten_mask,
    c10::OptionalIntArrayRef actual_seq_qlen,
    c10::OptionalIntArrayRef actual_seq_kvlen,
    const c10::optional<at::Tensor> &block_table,
    const c10::optional<at::Tensor> &dequant_scale_query,
    const c10::optional<at::Tensor> &dequant_scale_key,
    const c10::optional<at::Tensor> &dequant_offset_key,
    const c10::optional<at::Tensor> &dequant_scale_value,
    const c10::optional<at::Tensor> &dequant_offset_value,
    const c10::optional<at::Tensor> &dequant_scale_key_rope,
    const c10::optional<at::Tensor> &quant_scale_out,
    const c10::optional<at::Tensor> &quant_offset_out,
    const c10::optional<at::Tensor> &learnable_sink,
    int64_t num_query_heads, int64_t num_key_value_heads, double softmax_scale,
    int64_t pre_tokens, int64_t next_tokens, c10::string_view input_layout,
    int64_t sparse_mode, int64_t block_size,
    int64_t query_quant_mode, int64_t key_quant_mode, int64_t value_quant_mode,
    int64_t inner_precise, bool return_softmax_lse,
    c10::optional<int64_t> query_dtype, c10::optional<int64_t> key_dtype, c10::optional<int64_t> value_dtype,
    c10::optional<int64_t> query_rope_dtype, c10::optional<int64_t> key_rope_dtype,
    c10::optional<int64_t> key_shared_prefix_dtype, c10::optional<int64_t> value_shared_prefix_dtype,
    c10::optional<int64_t> dequant_scale_query_dtype, c10::optional<int64_t> dequant_scale_key_dtype,
    c10::optional<int64_t> dequant_scale_value_dtype, c10::optional<int64_t> dequant_scale_key_rope_dtype)
{
    // IFAContext ifaContext;
    // ConvertContextToParamsIFA(ifaContext, query, key, value, query_rope, key_rope,
    //                         pse_shift,atten_mask,actual_seq_qlen,actual_seq_kvlen, block_table,dequant_scale_query,
    //                         dequant_scale_key, dequant_offset_key, dequant_scale_value, dequant_offset_value,
    //                         dequant_scale_key_rope, quant_scale_out, quant_offset_out, learnable_sink,
    //                         num_query_heads, num_key_value_heads, softmax_scale, pre_tokens, next_tokens, input_layout,
    //                         sparse_mode,  block_size, query_quant_mode, key_quant_mode, value_quant_mode,
    //                         inner_precise, return_softmax_lse, query_dtype, key_dtype, value_dtype,
    //                         query_rope_dtype, key_rope_dtype, key_shared_prefix_dtype,  value_shared_prefix_dtype,
    //                         dequant_scale_query_dtype, dequant_scale_key_dtype, dequant_scale_value_dtype, dequant_scale_key_rope_dtype)
    // IFATiling ifaTiling();
    // ifaTiling.DoSubOpTiling(ifaContext);


    //----------------------------------------------------------------------------------------------
    printf("start npu\n");
    // convert str
    std::string input_layout_str = std::string(input_layout);

    // construct the output tensor
    std::tuple<at::Tensor, at::Tensor> fia_output = construct_fia_output_tensor_v2(query, value, input_layout_str,
                                                                                    quant_scale_out, block_table, num_query_heads,
                                                                                    num_key_value_heads,
                                                                                    return_softmax_lse, query_rope);
    at::Tensor output = std::get<0>(fia_output);
    at::Tensor softmax_lse = std::get<1>(fia_output);

    char input_layout_ptr[20];
    strncpy(input_layout_ptr, input_layout_str.c_str(), 20 - 1);

    at::IntArrayRef actual_shared_prefix_len;
    at::Tensor dequant_scale1;
    at::Tensor quant_scale1;
    at::Tensor dequant_scale2;
    at::Tensor antiquant_scale;
    at::Tensor antiquant_offset;
    at::Tensor query_padding_size;
    at::Tensor kv_padding_size;
    at::Tensor key_shared_prefix;
    at::Tensor value_shared_prefix;
    int64_t antiquant_mode = 0;

    at::TensorList valueTensors = value;
    at::TensorList keyTensors = key;

    auto actual_seq_qlen_ = actual_seq_qlen.value_or(at::IntArrayRef{});
    auto actual_seq_kvlen_ = actual_seq_kvlen.value_or(at::IntArrayRef{});

    EXEC_NPU_CMD_V1(aclnnFusedInferAttentionScoreV4, query, keyTensors, valueTensors, pse_shift, atten_mask, actual_seq_qlen_, actual_seq_kvlen_, dequant_scale1, quant_scale1, dequant_scale2,
        quant_scale_out, quant_offset_out, antiquant_scale, antiquant_offset, block_table, query_padding_size, kv_padding_size, dequant_scale_key, dequant_offset_key, dequant_scale_value,
        dequant_offset_value, key_shared_prefix, value_shared_prefix, actual_shared_prefix_len, query_rope, key_rope, dequant_scale_key_rope, 
        dequant_scale_query, learnable_sink, num_query_heads, softmax_scale, pre_tokens, next_tokens, input_layout_ptr,
        num_key_value_heads, sparse_mode, inner_precise, block_size, antiquant_mode, return_softmax_lse, key_quant_mode, value_quant_mode, query_quant_mode, output, softmax_lse);

    return std::tuple<at::Tensor, at::Tensor>(output, softmax_lse);
}

// 为META设备实现前向接口
std::tuple<at::Tensor, at::Tensor> npu_fused_infer_attention_score_meta(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
    const c10::optional<at::Tensor> &query_rope,
    const c10::optional<at::Tensor> &key_rope,
    const c10::optional<at::Tensor> &pse_shift,
    const c10::optional<at::Tensor> &atten_mask,
    c10::OptionalIntArrayRef actual_seq_qlen,
    c10::OptionalIntArrayRef actual_seq_kvlen,
    const c10::optional<at::Tensor> &block_table,
    const c10::optional<at::Tensor> &dequant_scale_query,
    const c10::optional<at::Tensor> &dequant_scale_key,
    const c10::optional<at::Tensor> &dequant_offset_key,
    const c10::optional<at::Tensor> &dequant_scale_value,
    const c10::optional<at::Tensor> &dequant_offset_value,
    const c10::optional<at::Tensor> &dequant_scale_key_rope,
    const c10::optional<at::Tensor> &quant_scale_out,
    const c10::optional<at::Tensor> &quant_offset_out,
    const c10::optional<at::Tensor> &learnable_sink,
    int64_t num_query_heads, int64_t num_key_value_heads, double softmax_scale,
    int64_t pre_tokens, int64_t next_tokens, c10::string_view input_layout,
    int64_t sparse_mode, int64_t block_size,
    int64_t query_quant_mode, int64_t key_quant_mode, int64_t value_quant_mode,
    int64_t inner_precise, bool return_softmax_lse,
    c10::optional<int64_t> query_dtype, c10::optional<int64_t> key_dtype, c10::optional<int64_t> value_dtype,
    c10::optional<int64_t> query_rope_dtype, c10::optional<int64_t> key_rope_dtype,
    c10::optional<int64_t> key_shared_prefix_dtype, c10::optional<int64_t> value_shared_prefix_dtype,
    c10::optional<int64_t> dequant_scale_query_dtype, c10::optional<int64_t> dequant_scale_key_dtype,
    c10::optional<int64_t> dequant_scale_value_dtype, c10::optional<int64_t> dequant_scale_key_rope_dtype)
{
    printf("start meta\n");
    // convert str
    std::string input_layout_str = std::string(input_layout);
    // construct the output tensor
    std::tuple<at::Tensor, at::Tensor> fia_output = construct_fia_output_tensor_v2(query, value, input_layout_str,
                                                                                           quant_scale_out, block_table, num_query_heads,
                                                                                           num_key_value_heads,
                                                                                           return_softmax_lse, query_rope);
    at::Tensor output = std::get<0>(fia_output);
    at::Tensor softmax_lse = std::get<1>(fia_output);
    return std::tuple<at::Tensor, at::Tensor>(output, softmax_lse);
}
}  // namespace custom

// 为NPU设备注册前向实现
TORCH_LIBRARY_IMPL(custom, PrivateUse1, m) {
    m.impl("npu_fused_infer_attention_score", &custom::npu_fused_infer_attention_score_npu);
}

// 为META设备注册前向实现
TORCH_LIBRARY_IMPL(custom, Meta, m) {
    m.impl("npu_fused_infer_attention_score", &custom::npu_fused_infer_attention_score_meta);
}
