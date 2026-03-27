#!/usr/bin/python3
# coding=utf-8

# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------


import sys
import os
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
import numpy as np
sys.path.append(os.getcwd())
import ascendc_ops
from typing import NamedTuple
ERROR_TOL = 5e-3
DATA_TYPE = np.float32

class FiaInput(NamedTuple):
    q_tensor: torch.Tensor
    k_tensor: torch.Tensor
    v_tensor: torch.Tensor
    dequant_scale_key: torch.Tensor
    dequant_scale_value: torch.Tensor
    dequant_scale_query: torch.Tensor
    num_query_heads:int
    softmax_scale: float
    input_layout: str
    num_key_value_heads:int
    query_quant_mode: int
    key_quant_mode: int
    value_quant_mode: int
    inner_precise: int
    return_softmax_lse: int
    query_dtype: int
    key_dtype: int
    value_dtype: int


def gen_golden_data_simple(b, n1, n2, s1, s2, d) -> FiaInput:
    input_layout = 'BNSD'
    scale_value = 0.088388

    q_tensor = torch.randint(-127, 127, (b, n1, s1, d), dtype=torch.int8)
    k_tensor = torch.randint(-127, 127, (b, n2, s2, d), dtype=torch.int8)
    v_tensor = torch.randint(-127, 127, (b, n2, s2, d), dtype=torch.int8)
    key_antiquant_scale = torch.rand((1, 1, 32, 1), dtype=torch.float32)
    value_antiquant_scale = torch.rand((1, 1, 32, 1), dtype=torch.float32)
    dequant_scale_query = torch.rand((1, 1, 64, 1), dtype=torch.float32)

    q_tensor_tmp = q_tensor.numpy()
    k_tensor_tmp = k_tensor.numpy()
    v_tensor_tmp = v_tensor.numpy()

    os.makedirs("input", exist_ok=True)
    os.makedirs("output", exist_ok=True)

    q_tensor_tmp.tofile("./input/input_0.bin")
    k_tensor_tmp.tofile("./input/input_1.bin")
    v_tensor_tmp.tofile("./input/input_2.bin")
    key_antiquant_scale.numpy().tofile("./input/input_15.bin")
    value_antiquant_scale.numpy().tofile("./input/input_17.bin")
    dequant_scale_query.numpy().tofile("./input/input_27.bin")

    q_tensor = q_tensor.npu()
    k_tensor = k_tensor.npu()
    v_tensor = v_tensor.npu()
    key_antiquant_scale = key_antiquant_scale.npu()
    value_antiquant_scale = value_antiquant_scale.npu()
    dequant_scale_query = dequant_scale_query.npu()

    npu_out = torch.ops.npu.npu_fused_infer_attention_score_v2(q_tensor, k_tensor, v_tensor,
                                                            dequant_scale_key=key_antiquant_scale,
                                                            dequant_scale_value=value_antiquant_scale,
                                                            dequant_scale_query=dequant_scale_query,
                                                            num_query_heads=n1,
                                                            softmax_scale=scale_value,
                                                            input_layout=input_layout,
                                                            num_key_value_heads=n2, query_quant_mode=7,
                                                            key_quant_mode=7, value_quant_mode=7,
                                                            inner_precise=0, return_softmax_lse=0,
                                                            query_dtype=torch_npu.float8_e4m3fn,
                                                            key_dtype=torch_npu.float8_e4m3fn,
                                                            value_dtype=torch_npu.float8_e4m3fn)
    fia_input = FiaInput(q_tensor, k_tensor, v_tensor, key_antiquant_scale, value_antiquant_scale, dequant_scale_query,
        n1, scale_value, input_layout, n2, 7, 7, 7, 0, 0, torch_npu.float8_e4m3fn, torch_npu.float8_e4m3fn,
        torch_npu.float8_e4m3fn)
    npu_out[0].cpu()
    return fia_input, npu_out[0].cpu()

def load_bf16_bin(file_path):
    # 读取原始字节
    with open(file_path, 'rb') as f:
        data_bytes = f.read()
    
    # 转为 uint16 数组（每个 bf16 占 2 字节）
    uint16_array = np.frombuffer(data_bytes, dtype=np.uint16)

    # 转换为 torch.bfloat16
    # 注意：numpy 不支持 bf16，所以要用 torch.from_numpy + to
    bf16_tensor = torch.from_numpy(uint16_array).to(torch.bfloat16)

    return bf16_tensor

def verify_result(golden, output):
    # output = load_bf16_bin("./output/npu_out.bin")
    # golden = load_bf16_bin("./output/golden_out.bin")
    golden = golden.view(torch.uint16).to(torch.bfloat16).flatten().cpu()
    aa = output
    output = output.view(torch.uint16).to(torch.bfloat16).flatten().cpu()
    print("output:")
    print(output)
    print("golden:")
    print(golden)

    output = output.float()
    golden = golden.float()

    # ------------------------------
    # NaN mask
    # ------------------------------

    output_nan = np.isnan(output)
    golden_nan = np.isnan(golden)
    both_nan = output_nan & golden_nan
    nan_mismatch = output_nan ^ golden_nan

    # ------------------------------
    # 数值误差
    # ------------------------------

    diff = np.abs(output - golden)

    # 误差位置
    diff_mask = diff > 1

    # 合并错误
    error_mask = (diff_mask | nan_mismatch) & (~both_nan)

    diff_indices = np.where(error_mask)[0]

    # ------------------------------
    # 打印前100个错误
    # ------------------------------

    max_print = min(100, diff_indices.size)

    for i in range(max_print):

        idx = diff_indices[i]

        golden_val = golden[idx]
        output_val = output[idx]

        denom = max(abs(golden_val), 1e-12)
 
        rdiff = abs(output_val - golden_val) / denom


    # ------------------------------
    # error ratio
    # ------------------------------

    error_ratio = diff_indices.size / golden.numpy().size

    print("error count:", diff_indices.size)
    print("total count:", golden.numpy().size)

    return error_ratio <= ERROR_TOL

class TestFia(TestCase):
    def test_fia(self):
        fia_input, golden = gen_golden_data_simple(1, 1, 1, 8192, 8192, 128)
        output = ascendc_ops.ascendc_fia(fia_input.q_tensor, fia_input.k_tensor, fia_input.v_tensor,
            fia_input.dequant_scale_key, fia_input.dequant_scale_value, fia_input.dequant_scale_query)

        import pdb;pdb.set_trace()

        try:
            res = verify_result(golden, output)
            if not res:
                raise ValueError("[ERROR] result error")
            print("test pass")
        except Exception as e:
            print(e)
            sys.exit(1)


if __name__ == "__main__":
    run_tests()
