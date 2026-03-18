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

def qr_decomposition_householder(A, device='cuda'):
    nrow, ncol = A.shape
    Q = torch.zeros(nrow, nrow, dtype=A.dtype)
    Q = Q.fill_diagonal_(1.)
    R = A
    for row in range(min(nrow, ncol)):
        x = R[row:nrow, row]

        y = torch.zeros_like(x, dtype=A.dtype)
        y[0] = -torch.sign(x[0]) * torch.linalg.norm(x)

        u = torch.zeros(nrow, dtype=A.dtype)
        u[row:nrow] = y - x

        #u = u / torch.linalg.norm(u) # torch.sqrt(sum(x**2))
        #H = torch.eye(nrow, dtype=A.dtype) - 2*torch.outer(u, u)
        H = torch.eye(nrow, dtype=A.dtype) - 2/sum(u**2)*torch.outer(u, u)

        Q = Q @ H
        R = H @ R
    return Q[:, :ncol], R
