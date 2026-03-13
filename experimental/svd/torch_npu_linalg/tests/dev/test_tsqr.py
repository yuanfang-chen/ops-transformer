# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import torch
import torch_npu
import npu_linalg
import pytest
import math
from scipy.linalg import block_diag
import numpy as np

DEVICE_ID = 0
torch_npu.npu.set_device(int(DEVICE_ID))

TSQR_TEST_CASES = [
(1, 4, 32, 16),
(2, 5, 64, 16),
(3, 11, 128, 64),
(1, 4, 320, 160),
(1, 4, 1024, 160),
(4, 8, 1024, 160),
(48, 64, 1024, 160),
(1, 8*1024, 1024, 160),
]

def tsqr_(A, min_block_size, threshold=0):
    bs, m, n = A.shape
    if m < threshold:
        # use ordinary qr
        return np.linalg.qr(A)
    else:
        total_blocks = m // min_block_size
        levels = int(math.ceil(np.log2(total_blocks)))
        print(f"Generate golden on shape {A.shape} | block size = {min_block_size}, num blocks = {total_blocks}, num levels = {levels}")
        Q_final = []
        R_final = []
        for A_b in A: # iterate by batch size
            # use tsqr
            Q_list = []
            Q, R = tsqr_step(A_b, min_block_size)
            Q_list.append(Q)
            for lvl in range(levels):
                Q, R = tsqr_step(R, 2*n)
                Q_list.append(Q)

            Q = np.linalg.multi_dot(Q_list) # Q = Q_1 @ Q_2 @ ... @ Q_levels
            Q_final.append(Q)
            R_final.append(R)
        return np.array(Q_final), np.array(R_final)


def tsqr_step(A, block_size):
    # Local QR
    m, n = A.shape
    Q_blocks = []
    R_blocks = []
    for start in range(0, m, block_size):
        # parallel steps
        end = min(start + block_size, m)
        Q_local, R_local = np.linalg.qr(A[start:end, :])
        Q_blocks.append(Q_local)
        R_blocks.append(R_local)

    R_stacked = np.vstack(R_blocks)
    Q_stacked = block_diag(*Q_blocks)

    return Q_stacked, R_stacked

def l1(x, y):
    return np.max(np.abs(x - y))
def l2(x, y):
    return np.sqrt(np.sum((x - y)**2))

@pytest.mark.parametrize("test_shapes", TSQR_TEST_CASES)
def test_tsqr(test_shapes):
    np.random.seed(0)
    shape = (test_shapes[0], test_shapes[1]*test_shapes[2], test_shapes[3])
    x = torch.tensor(np.random.uniform(-10, 10, shape)).to(torch.float32)

    torch_npu.npu.set_device(int(DEVICE_ID))
    npu_x = x.to("npu:%s" % DEVICE_ID)

    npu_q, npu_r = torch.ops.npu_linalg.tsqr(npu_x, test_shapes[2])

    output_q = np.asarray(npu_q.cpu())
    output_r = np.asarray(npu_r.cpu())

    a = np.asarray(x)
    a_out = np.matmul(output_q, output_r)
    result = True
    if (len(x.flatten()) <= 2*4*1024*256 ):
        golden_q, golden_r = tsqr_(x, min_block_size=test_shapes[2])
        golden_q = np.asarray(golden_q)
        golden_r = np.asarray(golden_r)

        if not np.allclose(output_q, golden_q, rtol=1e-03, atol=1e-03):
            result = False
            print("output_q != golden_q")
            for batch in range(a.shape[0]):
                if not np.allclose(output_q[batch, :, :], golden_q[batch, :, :], rtol=1e-03, atol=1e-03):
                    print(f"batch={batch}: fail | l1 = {l1(output_q[batch, :, :], golden_q[batch, :, :])}, l2 = {l2(output_q[batch, :, :], golden_q[batch, :, :])}")
                else:
                    print(f"batch={batch}: ok | l1 = {l1(output_q[batch, :, :], golden_q[batch, :, :])}, l2 = {l2(output_q[batch, :, :], golden_q[batch, :, :])}")

        if not np.allclose(output_r, golden_r, rtol=1e-03, atol=1e-03):
            result = False
            print("output_r != golden_r")
            for batch in range(a.shape[0]):
                if not np.allclose(output_r[batch, :, :], golden_r[batch, :, :], rtol=1e-03, atol=1e-03):
                    print(f"batch={batch}: fail | l1 = {l1(output_r[batch, :, :], golden_r[batch, :, :])}, l2 = {l2(output_r[batch, :, :], golden_r[batch, :, :])}")
                else:
                    print(f"batch={batch}: ok | l1 = {l1(output_r[batch, :, :], golden_r[batch, :, :])}, l2 = {l2(output_r[batch, :, :], golden_r[batch, :, :])}")

    else:
        print("No golden generated - shape is too big")

    if not l1(a, a_out) < 1e-4:
        result = False
        print(f"||A - QR||    l1 = {l1(a, a_out)}, l2 = {l2(a, a_out)}")

    assert result == True
