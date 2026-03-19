/* *
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <unordered_map>
#include <torch/library.h>
#include <torch/torch.h>
#include <ATen/Operators.h>
#include "torch_npu/csrc/framework/utils/OpPreparation.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "op_kernel/ifa_meta_public_define.h"

extern "C" {
    extern __global__ __aicpu__ uint32_t IncreFlashAttentionMetadataKernel(void *args);
}

namespace custom {
using namespace at_npu::native;

aicpu::kernels::Layout CovertToLayout(const std::string &str)
{
    std::unordered_map<std::string, aicpu::kernels::Layout> layoutDict = {
        {"BSH", aicpu::kernels::Layout::BSH},
        {"BSND", aicpu::kernels::Layout::BSND},
        {"BNSD", aicpu::kernels::Layout::BNSD},
        {"NZ", aicpu::kernels::Layout::NZ},
        {"TND", aicpu::kernels::Layout::TND},
        {"NBSD", aicpu::kernels::Layout::NBSD},
        {"NTD", aicpu::kernels::Layout::NTD}
    };
    auto layoutIter = layoutDict.find(str);
    return (layoutIter == layoutDict.end()) ? aicpu::kernels::Layout::BUTT : layoutIter->second;
}

// step3, 为META设备实现前向接口
at::Tensor npu_fused_infer_attention_score_metadata_meta(
    int64_t batch_size, int64_t query_seq_size, int64_t query_head_num, int64_t key_seq_size, int64_t key_head_num,
    int64_t block_size, int64_t max_block_num_per_batch, bool is_accum_seq_query, bool is_accum_seq_kv,
    at::Tensor &actual_seq_lengths_query, at::Tensor &actual_seq_lengths_kv,
    c10::string_view layout_query, c10::string_view layout_key)
{
    printf("start npu_fused_infer_attention_score_metadata_meta\n");
    at::Tensor output = torch::empty({1024}, torch::dtype(torch::kInt32).device("npu"));
    return output;
}

at::Tensor npu_fused_infer_attention_score_metadata_npu(
    int64_t batch_size, int64_t query_seq_size, int64_t query_head_num, int64_t key_seq_size, int64_t key_head_num,
    int64_t block_size, int64_t max_block_num_per_batch, bool is_accum_seq_query, bool is_accum_seq_kv,
    at::Tensor &actual_seq_lengths_query, at::Tensor &actual_seq_lengths_kv,
    c10::string_view layout_query, c10::string_view layout_key)
{
    printf("start npu_fused_infer_attention_score_metadata_npu\n");
    at::Tensor output = torch::empty({1024}, torch::dtype(torch::kInt32).device("npu"));

    auto aicpu_stream = c10_npu::getCurrentNPUStream().stream(true);

    aicpu::kernels::IncreFlashAttentionMetadataArgs args {};
    args.aicCoreNum = 24;
    args.aivCoreNum = 48;
    args.batchSize = batch_size;
    args.querySeqSize = query_seq_size;
    args.queryHeadNum = query_head_num;
    args.keySeqSize = key_seq_size;
    args.keyHeadNum = key_head_num;
    args.blockSize = block_size;
    args.maxBlockNumPerBatch = max_block_num_per_batch;
    args.isAccumSeqQ = is_accum_seq_query;
    args.actSeqQLenDim = actual_seq_lengths_query.size(0);
    args.actSeqQLen = static_cast<int32_t *>(const_cast<void *>(actual_seq_lengths_query.storage().data()));
    args.isAccumSeqKv = is_accum_seq_kv;
    args.actSeqKvLenDim = actual_seq_lengths_kv.size(0);
    args.actSeqKvLen = static_cast<int32_t *>(const_cast<void *>(actual_seq_lengths_kv.storage().data()));

    // convert str
    args.layoutQuery = CovertToLayout(std::string(layout_query));
    args.layoutKey = CovertToLayout(std::string(layout_key));
    args.metaData = static_cast<int8_t *>(const_cast<void *>(output.storage().data()));

    IncreFlashAttentionMetadataKernel<<<1, nullptr, aicpu_stream>>>(&args, sizeof(aicpu::kernels::IncreFlashAttentionMetadataArgs));
    return output;
}
}

// step4, 为NPU设备注册前向实现
TORCH_LIBRARY_IMPL(custom, PrivateUse1, m) {
    m.impl("npu_fused_infer_attention_score_metadata", &custom::npu_fused_infer_attention_score_metadata_npu);
}


// step5, 为META设备注册前向实现
TORCH_LIBRARY_IMPL(custom, Meta, m) {
    m.impl("npu_fused_infer_attention_score_metadata", &custom::npu_fused_infer_attention_score_metadata_meta);
}
