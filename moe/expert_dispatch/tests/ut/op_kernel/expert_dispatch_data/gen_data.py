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


def gen_golden_data_simple(N, H, k, dtype):
    N = int(N)
    H = int(H)
    k = int(k)

    input_x = np.random.uniform(-2, 2, [N, H]).astype(dtype)
    input_expertId = np.random.randint(0, 255, size=(N, k)).astype("int32")
    scale = np.random.uniform(-2, 2, [N]).astype(np.float32)

    input_x.tofile("./input_x.bin")
    input_expertId.tofile("./input_expertId.bin")
    scale.tofile("./scale.bin")


if __name__ == "__main__":
    gen_golden_data_simple(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
