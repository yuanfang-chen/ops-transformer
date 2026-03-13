# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

# based on torch/_lowrank.py
import torch
from qrh import qr_decomposition_householder
from svdj import svd_jacoby_pytorch
from safetensors.torch import load_file

qr_decompose = torch.qr # torch.linalg.qr
svd = torch.svd
matmul = torch.matmul

def get_approximate_basis(A, q, niter=2, omega=None):

    dtype =  A.dtype
    matmul = torch.matmul
    if omega is None:
        omega = torch.randn(A.shape[-1], q, dtype=dtype, device=A.device)
    Y = matmul(A, omega)
    Q, _ = qr_decompose(Y)
    for _ in range(niter):
        Y = matmul(A.T, Q)
        Q, _ = qr_decompose(Y)
        Y = matmul(A, Q)
        Q, _ = qr_decompose(Y)
    return Q

def svd_lowrank(A, q=6, niter=2, omega=None):
    # Algorithm 5.1 in Halko et al., 2009

    q = 6 if q is None else q
    m, n = A.shape[-2:]

    Q = get_approximate_basis(A, q, niter, omega)

    B = matmul(Q.T, A)

    U, S, Vh = svd(B)
    U = matmul(Q, U)

    return U, S, Vh