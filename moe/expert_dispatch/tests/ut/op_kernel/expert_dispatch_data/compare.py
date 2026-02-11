#!/usr/bin/python3
# -*- coding:utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import sys
import numpy as np
import glob
import os
import tensorflow as tf

bf16 = tf.bfloat16.as_numpy_dtype

curr_dir = os.path.dirname(os.path.realpath(__file__))


def compare_paras(output_file, golden_file, d_type, precision, name):
    output = np.fromfile(output_file, d_type)
    golden = np.fromfile(golden_file, d_type)

    # Remove padding zeros from output
    output = output[:len(golden)]

    diff_res = np.isclose(output, golden, precision, 0, True)
    diff_idx = np.where(diff_res != True)[0]
    diff_ratio = 0 if golden.size == 0 else len(diff_idx) / golden.size

    if len(diff_idx) == 0 or diff_ratio < precision:
        print(f"{name} PASSED!")
    else:
        print(f"{name} FAILED!")
        print(f"{name} Diff ratio: {diff_ratio}")
        for idx in diff_idx[:5]:
            print(f"index: {idx}, {name} output: {output[idx]}, {name} golden: {golden[idx]}")


def compare_data(  
        dispatched_x_file,
        dispatched_row_idx_file,
        expert_tokens_count_file,
        actual_expert_total_num_file,
        dispatched_scale_file,
        dispatched_x_golden_file,
        dispatched_row_idx_golden_file,
        expert_tokens_count_golden_file,
        actual_expert_total_num_golden_file,
        dispatched_scale_golden_file,
        d_type
):
    if d_type == "float16":
        precision = 1/1000
    else:
        precision = 1/10000

    dispatched_x_type_dict = {
        "int8": np.int8,
        "float16": np.float16,
        "bfloat16": bf16,
        "float32": np.float32
    }

    compare_paras(dispatched_x_file, dispatched_x_golden_file, dispatched_x_type_dict.get(d_type), precision, "dispatched_x")
    compare_paras(dispatched_row_idx_file, dispatched_row_idx_golden_file, np.int32, precision, "dispatched_row_idx")
    compare_paras(expert_tokens_count_file, expert_tokens_count_golden_file, np.int32, precision, "expert_tokens_count")
    compare_paras(actual_expert_total_num_file, actual_expert_total_num_golden_file, np.int32, precision, "actual_expert_total_num")
    compare_paras(dispatched_scale_file, dispatched_scale_golden_file, np.float32, precision, "dispatched_scale")


def process(d_type):
    dispatched_x_file = os.path.join(curr_dir, "dispatched_x.bin")
    dispatched_row_idx_file = os.path.join(curr_dir, "dispatched_row_idx.bin")
    expert_tokens_count_file = os.path.join(curr_dir, "expert_tokens_count.bin")
    actual_expert_total_num_file = os.path.join(curr_dir, "actual_expert_total_num.bin")
    dispatched_scale_file = os.path.join(curr_dir, "dispatched_scale.bin")
    dispatched_x_golden_file = os.path.join(curr_dir, "dispatched_x_golden.bin")
    dispatched_row_idx_golden_file = os.path.join(curr_dir, "dispatched_row_idx_golden.bin")
    expert_tokens_count_golden_file = os.path.join(curr_dir, "expert_tokens_count_golden.bin")
    actual_expert_total_num_golden_file = os.path.join(curr_dir, "actual_expert_total_num_golden.bin")
    dispatched_scale_golden_file = os.path.join(curr_dir, "dispatched_scale_golden.bin")
    
    compare_data(  
        dispatched_x_file,
        dispatched_row_idx_file,
        expert_tokens_count_file,
        actual_expert_total_num_file,
        dispatched_scale_file,
        dispatched_x_golden_file,
        dispatched_row_idx_golden_file,
        expert_tokens_count_golden_file,
        actual_expert_total_num_golden_file,
        dispatched_scale_golden_file,
        d_type)


if __name__ == '__main__':
    process(sys.argv[1])
