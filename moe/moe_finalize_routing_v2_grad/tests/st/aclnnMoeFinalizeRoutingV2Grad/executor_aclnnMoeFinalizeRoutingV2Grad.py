#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
import torch
import numpy
from atk.configs.dataset_config import InputDataset

from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi


@register("function_aclnn_moe_finalize_routing_v2_grad")
class FunctionApi(BaseApi):
    def __call__(self, input_data: InputDataset, with_output: bool = False):
        grad_y = input_data.kwargs.get("grad_y").cpu().numpy()
        expanded_row_idx = input_data.kwargs.get("expanded_row_idx").cpu().numpy()
        expanded_x = input_data.kwargs.get("expanded_x").cpu().numpy()
        scales = input_data.kwargs.get("scales").cpu().numpy()
        expert_idx = input_data.kwargs.get("expert_idx").cpu().numpy()
        bias = input_data.kwargs.get("bias").cpu().numpy()
        tmp_dtype = "float16"
        output_dtype = [tmp_dtype]

        drop_pad_mode = input_data.kwargs.get("drop_pad_mode")
        active_num = input_data.kwargs.get("active_num")
        expert_num = input_data.kwargs.get("expert_num")
        expert_capacity = input_data.kwargs.get("expert_capacity")

        grad_y_shape = grad_y.shape
        expanded_row_idx_shape = expanded_row_idx.shape

        row = grad_y_shape[0]
        hidden = grad_y_shape[1]
        row_topk = expanded_row_idx_shape[0]
        topk = 1
        if scales is not None:
            scales_shape = scales.shape
            topk = scales_shape[1]

        expandedX_dim_0 = row_topk
        if drop_pad_mode == 0 and active_num > 0 and active_num < row_topk:
            expandedX_dim_0 = active_num
        elif drop_pad_mode == 1 and expert_num != 0 and expert_capacity != 0:
            expandedX_dim_0 = expert_num * expert_capacity


        if output_dtype[0] == "bfloat16" or output_dtype[0] == "float16":
            grad_y = grad_y.astype("float32")
        grad_y_tensor = torch.from_numpy(grad_y)
        grad_y_dtype = grad_y_tensor.dtype
        grad_y_tensor = grad_y_tensor.unsqueeze(1).expand(row, topk, hidden).reshape(row_topk, hidden)
        expanded_row_idx = expanded_row_idx.astype("int64")
        expanded_row_idx_tensor = torch.from_numpy(expanded_row_idx)
        sorted, indices = torch.sort(expanded_row_idx_tensor, dim=-1)

        zeros = torch.zeros((1, hidden), dtype=grad_y_dtype)

        if scales is None:
            # grad_expanded_x
            if drop_pad_mode == 0:
                if expandedX_dim_0 < row_topk:
                    indices = indices[0:expandedX_dim_0]
                grad_expanded_x_golden_tensor = grad_y_tensor.index_select(0, indices)
            else:
                first_index = torch.nonzero(sorted.ne(-1))[0][0]
                sorted = sorted[first_index:]
                indices = indices[first_index:]
                indices2 = torch.full((expandedX_dim_0,), row_topk)
                indices2.index_put_((sorted,), indices)
                grad_y_tensor = torch.cat((grad_y_tensor, zeros), dim=0)
                grad_expanded_x_golden_tensor = grad_y_tensor.index_select(0, indices2)

            # grad_scales
            grad_scales_golden_tensor = torch.ones((row_topk, 1), dtype=grad_y_dtype)
        else:
            scales = scales.reshape(row_topk)
            if output_dtype[0] == "bfloat16" or output_dtype[0] == "float16":
                scales = scales.astype("float32")
            scales_tensor = torch.from_numpy(scales)
            scales_tensor = scales_tensor.unsqueeze(1).expand(-1, hidden)

            # grad_expanded_x
            if drop_pad_mode == 0:
                if expandedX_dim_0 < row_topk:
                    indices = indices[0:expandedX_dim_0]
                grad_expanded_x_golden_tensor = grad_y_tensor.index_select(0, indices) * scales_tensor.index_select(0,
                                                                                                                    indices)
            else:
                first_index = torch.nonzero(sorted.ne(-1))[0][0]
                sorted = sorted[first_index:]
                indices = indices[first_index:]
                indices2 = torch.full((expandedX_dim_0,), row_topk)
                indices2.index_put_((sorted,), indices)
                grad_y_tensor_2 = torch.cat((grad_y_tensor, zeros), dim=0)
                scales_tensor = torch.cat((scales_tensor, zeros), dim=0)
                grad_expanded_x_golden_tensor = (grad_y_tensor_2.index_select(0, indices2) * scales_tensor.index_select(0, indices2)).reshape(expert_num, expert_capacity, hidden)

            # grad_scales
            if output_dtype[0] == "bfloat16" or output_dtype[0] == "float16":
                expanded_x = expanded_x.astype("float32")
            expanded_x_tensor = torch.from_numpy(expanded_x)

            if drop_pad_mode == 0:
                if expandedX_dim_0 < row_topk:
                    expanded_x_tensor = torch.cat((expanded_x_tensor, zeros), dim=0)
                    expanded_row_idx_tensor = torch.where(expanded_row_idx_tensor >= expandedX_dim_0,
                                                        torch.tensor(expandedX_dim_0), expanded_row_idx_tensor)
            else:
                expanded_x_tensor = expanded_x_tensor.reshape(expandedX_dim_0, hidden)
                expanded_x_tensor = torch.cat((expanded_x_tensor, zeros), dim=0)
                expanded_row_idx_tensor = torch.where(expanded_row_idx_tensor == -1,
                                                    torch.tensor(expandedX_dim_0, dtype=torch.int64),
                                                    expanded_row_idx_tensor)
                expanded_row_idx_tensor = torch.where(expanded_row_idx_tensor >= expandedX_dim_0,
                                                    torch.tensor(expandedX_dim_0, dtype=torch.int64),
                                                    expanded_row_idx_tensor)

            add_result = expanded_x_tensor.index_select(0, expanded_row_idx_tensor)

            if bias is not None:
                if output_dtype[0] == "bfloat16" or output_dtype[0] == "float16":
                    bias = bias.astype("float32")
                bias_tensor = torch.from_numpy(bias)
                expert_idx = expert_idx.reshape(row_topk)
                expert_idx_tensor = torch.from_numpy(expert_idx)
                add_result = add_result + bias_tensor.index_select(0, expert_idx_tensor)

            grad_scales_golden_tensor = torch.sum(add_result * grad_y_tensor, dim=1).reshape(row, topk)

        if output_dtype[0] == "bfloat16":
            grad_expanded_x_golden = grad_expanded_x_golden_tensor.numpy().astype(tf.bfloat16.as_numpy_dtype)
            grad_scales_golden = grad_scales_golden_tensor.numpy().astype(tf.bfloat16.as_numpy_dtype)
        elif output_dtype[0] == "float16":
            grad_expanded_x_golden = grad_expanded_x_golden_tensor.numpy().astype("float16")
            grad_scales_golden = grad_scales_golden_tensor.numpy().astype("float16")
        else:
            grad_expanded_x_golden = grad_expanded_x_golden_tensor.numpy()
            grad_scales_golden = grad_scales_golden_tensor.numpy()

        return torch.from_numpy(grad_expanded_x_golden), torch.from_numpy(grad_scales_golden)


    def get_cpp_func_signature_type(self):
        return "aclnnStatus aclnnMoeFinalizeRoutingV2GradGetWorkspaceSize(const aclTensor *gradY, const aclTensor *expandedRowIdx, const aclTensor *expandedXOptional, const aclTensor *scalesOptional, const aclTensor *expertIdxOptional, const aclTensor *biasOptional, int64_t dropPadMode, int64_t activeNum, int64_t expertNum, int64_t expertCapacity, const aclTensor *gradExpandedXOut, const aclTensor *gradScalesOut, uint64_t *workspaceSize, aclOpExecutor **executor)"
