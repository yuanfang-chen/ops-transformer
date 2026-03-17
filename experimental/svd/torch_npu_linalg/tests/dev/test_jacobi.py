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

JACOBI_SVD_TEST_CASES = [
    ((1, 168, 1024), 2e-1, 6),
    ((1, 168, 1024), 2e-1, 7),
    ((2, 168, 512), 2e-1, 7),
    ((2, 1, 3, 168, 512), 1e-3, 10),
    ((48, 168, 512), 2e-1, 10),
]

@pytest.mark.parametrize("test_shapes,atol,num_iter", JACOBI_SVD_TEST_CASES)
def test_jacobi_svd(test_shapes, atol, num_iter):
    np.random.seed(0)
    x = torch.tensor(np.random.uniform(-1, 1, test_shapes)).to(torch.float32)
    u_gold, s_gold, vh_gold = torch.svd(x)
    torch_npu.npu.set_device(int(DEVICE_ID))
    npu_x = x.to("npu:%s" % DEVICE_ID)

    npu_u, npu_s, npu_v = torch.ops.npu_linalg.svd(npu_x, num_iter)
    npu_u = npu_u.cpu().numpy()
    npu_s = npu_s.cpu().numpy()
    npu_v = npu_v.cpu().numpy()
    rank_u = len(npu_u.shape)
    axes = list(range(rank_u))
    axes[rank_u-2]=rank_u-1
    axes[rank_u-1]=rank_u-2
    axes = tuple(axes)
    npu_u = np.transpose(npu_u,axes=axes)
    npu_s= npu_s.reshape(*npu_s.shape[:-1],1, npu_s.shape[-1])
    x= x.numpy()

    s_gold= s_gold.numpy()
    
    x_new = (npu_u*npu_s)@npu_v
    npu_s = npu_s.reshape(s_gold.shape)
    diff = np.sum(np.abs(x-x_new))
    max_diff = np.max(np.abs(x-x_new))
    print("total diff is {0}".format(diff))
    print("max diff is {0}".format(max_diff))
    sv_diff = np.max(np.abs(s_gold-npu_s))
    print("S max diff {0}".format(sv_diff))
    print(s_gold[0,0:16])
    print(npu_s[0,0:16])
    assert (sv_diff < atol) or (max_diff<atol)
