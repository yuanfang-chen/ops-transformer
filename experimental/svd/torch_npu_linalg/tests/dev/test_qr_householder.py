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
import numpy as np

DEVICE_ID = 0
torch_npu.npu.set_device(int(DEVICE_ID))

QR_HOUSEHOLDER_TEST_CASES = [
    ((32, 16), 1e-5),
    ((512, 16), 1e-5),
    ((2048, 16), 1e-5),
    ((320, 160), 1e-5),
    ((1024, 160), 1e-5),
    ((2048, 160), 1e-5),
    ((2048, 320), 1e-5),
    ((640, 320), 1e-5),
]

def l1(x, y):
    return np.max(np.abs(x - y))
def l2(x, y):
    return np.sqrt(np.sum((x - y)**2))


@pytest.mark.parametrize("test_shapes,atol", QR_HOUSEHOLDER_TEST_CASES)
def test_qr_householder(test_shapes, atol):
    np.random.seed(0)
    x = torch.tensor(np.random.uniform(-10, 10, test_shapes)).to(torch.float32)
    golden_q, golden_r = np.linalg.qr(x)

    torch_npu.npu.set_device(int(DEVICE_ID))
    npu_x = x.to("npu:%s" % DEVICE_ID)

    npu_q, npu_r = torch.ops.npu_linalg.qr_householder(npu_x)

    npu_q = np.asarray(npu_q.cpu().T)
    npu_r = np.asarray(npu_r.cpu())
    golden_q = np.asarray(golden_q)
    golden_r = np.asarray(golden_r)

    result = True
    if not np.allclose(np.abs(npu_q), np.abs(golden_q), rtol=1e-03, atol=1e-03):
        result = False
        print("output_q != golden_q")
        print(f"l1 = {l1(np.abs(npu_q), np.abs(golden_q))}, l2 = {l2(np.abs(npu_q), np.abs(golden_q))}")

    if not np.allclose(np.abs(npu_r), np.abs(golden_r), rtol=1e-03, atol=1e-03):
        result = False
        print("output_r != golden_r")
        print(f"l1 = {l1(np.abs(npu_r), np.abs(golden_r))}, l2 = {l2(np.abs(npu_r), np.abs(golden_r))}")

    a = np.asarray(x)
    a_out = np.dot(npu_q, npu_r)
    if not l2(a, a_out) < 3e-3:
        result = False
        print(f"||Q^T * Q - I||    l1 = {l1(npu_q.T@npu_q, np.eye(npu_q.shape[1]))}, l2 = {l2(npu_q.T@npu_q, np.eye(npu_q.shape[1]))}")
        print("R is upper triangle:", np.allclose(npu_r, np.triu(npu_r)))
        print(f"||A - QR||    l1 = {l1(a, a_out)}, l2 = {l2(a, a_out)}")

    assert result == True
