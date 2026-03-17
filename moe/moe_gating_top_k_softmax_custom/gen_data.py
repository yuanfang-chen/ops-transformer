#!/usr/bin/python3
# coding=utf-8
#
# Copyright (C) 2023-2024. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================

import numpy as np
import torch

def gen_golden_data_simple():
    x = torch.randn(1200, 64).to(torch.float32) 
    print("#### Compute Finished")
    print(x.shape)
    x = x.to(torch.float16)
    print("before save")
    torch.save(x, "./input/input0.pth")
    print("#### Save End")


if __name__ == "__main__":
    gen_golden_data_simple()
