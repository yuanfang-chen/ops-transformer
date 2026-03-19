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

sys.path.append(os.getcwd())
import ascendc_ops

def get_quant_pre(scale, offset):
    # convert float32 to uint32
    import struct
    scale_binary = struct.pack('f', scale)
    scale_int = int.from_bytes(scale_binary, byteorder='little')

    # round to nearest, tie to even
    offset_round = round(float(offset))
    offset_round_clip = min(max(-256, offset_round), 255)
    offset_binary = struct.pack('i', offset_round_clip)
    offset_int = int.from_bytes(offset_binary, byteorder='little') & 0x1FF  # get complement of int9

    quant_pre_u64 = (1 << 46) + (offset_int << 37) + scale_int

    return quant_pre_u64

def golden(x1, x2, scale, offset, bias):
    y = ((torch.matmul(x1.to(torch.int32), x2.to(torch.int32)) + bias.to(torch.int32)) * scale.to(torch.float32) + offset.to(torch.float32)).to(torch.float16)
    return y

class TestCustomQbmmv3(TestCase):
    def test_qbmmv3_custom_ops(self):
        m, k, n = 5, 2, 3
        x1 = torch.randint(-10, 10, [m, k], device='cpu', dtype=torch.int8)
        x2 = torch.randint(-10, 10, [k, n], device='cpu', dtype=torch.int8)
        scale = torch.rand([n], device='cpu', dtype=torch.float32)
        offset = torch.rand([n], device='cpu', dtype=torch.float32)
        scale_uint64 = torch.zeros([n], device='cpu', dtype=torch.uint64)
        bias = torch.randint(-10, 10, [n], device='cpu', dtype=torch.int32)
        offset *= 0

        for i in range(n):
            scale_uint64[i] = get_quant_pre(scale[i], offset[i])
        output = ascendc_ops.ascendc_qbmmv3(
            x1.npu(),
            x2.npu(),
            bias.npu(),
            scale_uint64.npu(),
        ).cpu()
        cpuout = golden(x1, x2, scale, offset, bias)
        self.assertRtolEqual(output, cpuout)


if __name__ == "__main__":
    run_tests()
