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
#include <torch/library.h>
#include <ATen/Operators.h>
#include "torch_npu/csrc/framework/utils/OpPreparation.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include <fstream>
#include <sys/stat.h>
#include <dlfcn.h>
#include <vector>
#include <functional>
#include <type_traits>
#include <ATen/Tensor.h>
#include <ATen/NamedTensorUtils.h>
#include <acl/acl_base.h>
#include <acl/acl_rt.h>
#include <c10/util/Exception.h>
#include <torch/extension.h>
#include "torch_npu/csrc/aten/CustomFunctions.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/core/npu/NPUFunctions.h"
#include "torch_npu/csrc/core/npu/NpuVariables.h"
#include "torch_npu/csrc/core/npu/register/OptionsManager.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include <torch_npu/csrc/framework/utils/CalcuOpUtil.h>
#include <torch_npu/csrc/framework/utils/OpAdapter.h>
#include "torch_npu/csrc/framework/utils/OpPreparation.h"
#include "torch_npu/csrc/framework/utils/RandomOpAdapter.h"
#include "torch_npu/csrc/framework/interface/AclOpCompileInterface.h"
#include "torch_npu/csrc/framework/interface/EnvVariables.h"
#include "torch_npu/csrc/flopcount/FlopCount.h"
#include "torch_npu/csrc/flopcount/FlopCounter.h"
#include "op_kernel/split_core.h"

extern "C" {
    extern __global__ __aicpu__ uint32_t IncreFlashAttentionMetadataKernel(void *args);
}

namespace custom {
using namespace at_npu::native;

// step3, 为META设备实现前向接口
at::Tensor npu_fused_infer_attention_score_metadata_meta(
    int64_t batch_size, int64_t query_seq_size, int64_t query_head_num, int64_t key_seq_size, int64_t key_head_num,
    int64_t block_size, int64_t max_block_num_per_batch,
    const c10::optional<at::Tensor> &actual_seq_lengths_query,
    const c10::optional<at::Tensor> &actual_seq_lengths_kv,
    c10::string_view layout_query, c10::string_view layout_key)
{
    printf("start npu_fused_infer_attention_score_metadata_meta\n");
    at::Tensor output = torch::empty({1024}, torch::dtype(torch::kInt32).device("npu"));
    // at::Tensor output = at::empty({1024});
    return output;
}

at::Tensor npu_fused_infer_attention_score_metadata_npu(
    int64_t batch_size, int64_t query_seq_size, int64_t query_head_num, int64_t key_seq_size, int64_t key_head_num,
    int64_t block_size, int64_t max_block_num_per_batch,
    const c10::optional<at::Tensor> &actual_seq_lengths_query,
    const c10::optional<at::Tensor> &actual_seq_lengths_kv,
    c10::string_view layout_query, c10::string_view layout_key)
{
    printf("start npu_fused_infer_attention_score_metadata_npu\n");
    at::Tensor output = torch::empty({1024}, torch::dtype(torch::kInt32).device("npu"));

    // convert str
    std::string layout_query_str = std::string(layout_query);
    std::string layout_kv_str = std::string(layout_key);
    char *layout_query_ptr = const_cast<char *>(layout_query_str.c_str());
    char *layout_kv_ptr = const_cast<char *>(layout_kv_str.c_str());

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
    if (actual_seq_lengths_query.has_value()) {
        args.actSeqQLenDim = actual_seq_lengths_query->size(0);
        args.actSeqQLen = static_cast<int8_t *>(const_cast<void *>(actual_seq_lengths_query->storage().data()));
    } else {
        args.actSeqQLenDim = 0U;
        args.actSeqQLen = nullptr;
    }
    if (actual_seq_lengths_kv.has_value()) {
        args.actSeqKvLenDim = actual_seq_lengths_kv->size(0);
        args.actSeqKvLen = static_cast<int8_t *>(const_cast<void *>(actual_seq_lengths_kv->storage().data()));
    } else {
        args.actSeqKvLenDim = 0U;
        args.actSeqKvLen = nullptr;
    }
    args.layoutQuery = layout_query_str.c_str();
    args.layoutKey = layout_kv_str.c_str();
    args.metaData = static_cast<int8_t *>(const_cast<void *>(output.storage().data()));

    IncreFlashAttentionMetadataKernel<<<1, nullptr, aicpu_stream>>>(&args, sizeof(aicpu::kernels::IncreFlashAttentionMetadataArgs));
    // IncreFlashAttentionMetadataKernel<<<1, nullptr, aicpu_stream>>>(&args);
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
