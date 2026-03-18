# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import numpy as np
import tbetoolkits
from .registry import register_golden
from tbetoolkits.utilities import get_global_storage


def golden_version_for_kernel(A):
    m, n = A.shape
    betas = np.zeros(n)
    v_list = []
    R = A.copy()
    for col in range(n):
        x = R[col:, col]
        norm_x = np.linalg.norm(x)
        if norm_x == 0:
            betas[col] = 0
            continue
        v = x.copy()

        sign_x = 0
        if x[0] >= 0:
            sign_x = 1
        else:
            sign_x = -1
        v[0] = v[0] + sign_x * norm_x

        beta = 2.0 / np.dot(v, v)
        betas[col] = beta
        wT = beta * np.dot(v, R[col:, col:])
        R[col:, col:] -= np.outer(v, wT)

        mulTmp = np.zeros(m - col, dtype=np.float32)
        mulTmp[0] = 1
        R[col:, col] = R[col:, col] * mulTmp

        full_v = np.zeros(m)
        full_v[col:] = v
        v_list.append(full_v)

    R_upper = R[:n, :]

    Q = np.zeros((m, n))
    for i in range(n):
        q = np.zeros(m)
        q[i] = 1.0

        for k in range(i, -1, -1):
            v = v_list[k]
            beta = betas[k]

            q = q - beta * v * np.dot(v, q)
        Q[:, i] = q
    return Q, R_upper


def numpy_version_qr(A):
    batch, m, k = A.shape
    Q, R = np.linalg.qr(A)
    Q = Q[:, :, 0:k]
    R = R[:, 0:k, :]
    return Q, R


@register_golden(["qr_householder"])
def qr_householder(context: "tbetoolkits.UniversalTestcaseStructure"):
    input_matrix = context.input_arrays[0]
    Q, R = numpy_version_qr(input_matrix)
    return Q.transpose((0, 2, 1)), R