#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import sys
import numpy as np
from typing import List


def gen_golden_data_simple(N, H, k, start, end, dtype):
    N = int(N)
    H = int(H)
    k = int(k)

    input_x = np.random.uniform(-2.0, 2.0, [N, H]).astype(dtype)
    input_expertId = np.random.randint(0, 255, size=(N, k)).astype("int32")
    scale = np.random.uniform(-2.0, 2.0, [N]).astype(np.float32)

    input_x.tofile("./input_x.bin")
    input_expertId.tofile("./input_expertId.bin")
    scale.tofile("./scale.bin")

    dispatched_x, dispatched_row_idx, expert_tokens_count, actual_expert_total_num, dispatched_scale = expert_dispatch_np(
        input_x, input_expertId, scale, [int(start), int(end)])
    
    dispatched_x.astype(dtype).tofile("./dispatched_x_golden.bin")
    dispatched_row_idx.astype(np.int32).tofile("./dispatched_row_idx_golden.bin")
    expert_tokens_count.astype(np.int32).tofile("./expert_tokens_count_golden.bin")
    actual_expert_total_num.astype(np.int32).tofile("./actual_expert_total_num_golden.bin")
    dispatched_scale.astype(np.float32).tofile("./dispatched_scale_golden.bin")


def expert_dispatch_np(x: np.ndarray, expert_id: np.ndarray, scale: np.ndarray, expert_range: List):
    expert_start = expert_range[0]
    expert_end = expert_range[1]
    k = expert_id.shape[-1]
    expert_id_in = expert_id.copy().reshape(-1)
    actual_expert_total_num = np.sum(
        (expert_id_in >= expert_start) & (expert_id_in < expert_end))

    # sort
    expert_id_in[(expert_id_in < expert_start)] = np.int32(np.iinfo(np.int32).max)
    sorted_expert_indices = np.argsort(expert_id_in, axis=-1, kind="stable")
    sorted_expert_idx = expert_id_in[sorted_expert_indices]
    dispatched_row_idx = sorted_expert_indices

    # gather
    dispatched_x = x[sorted_expert_indices[:actual_expert_total_num] // k, :]
    dispatched_scale = scale[sorted_expert_indices[:actual_expert_total_num] // k]

    # compute histograms
    expert_tokens_count = np.bincount(
        sorted_expert_idx[:actual_expert_total_num] - expert_start)
    expert_tokens_count = np.concatenate(
        [expert_tokens_count, np.zeros((expert_end - expert_start) - len(expert_tokens_count))])
    
    return dispatched_x, dispatched_row_idx, expert_tokens_count, actual_expert_total_num, dispatched_scale


if __name__ == "__main__":
    gen_golden_data_simple(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6])
