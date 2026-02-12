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

import numpy as np
import sys

case0_params = [
    8, 4, 8, 16, 1, 7, 6,
    1, 64, 1, 64, 64, 64, 1, 64, 64, 8160,
    0,
    1024,
    8, 8, 8, 1, 8, 8, 1, 8, 8,
    8, 8, 8, 1, 8, 8, 1, 8, 8, 1, 8, 8
]


params_info = {
    "case0": case0_params,
}


def main():
    # python gen_tiling.py case0  sys.argv[1]="case0"
    params_list = params_info[sys.argv[1]]

    base_params = np.array(params_list, dtype=np.int64)

    tiling_file = open("tiling.bin", "wb")
    base_params.tofile(tiling_file)


if __name__ == "__main__":
    main()
