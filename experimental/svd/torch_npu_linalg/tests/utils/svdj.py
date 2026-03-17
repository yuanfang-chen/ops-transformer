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
import torch
import time

def svd_jacoby_pytorch(A, num_iteration=6, device='cuda'):
  tol = 1.0e-014
  A = A.to(device)
  m, n = A.shape
  r_c = [[r, c] for r in range(n-1) for c in range (r+1, n)]
  v = torch.eye(n, dtype=A.dtype, device=device)
  on = sum(sum(abs(A)**2))/n

  for _ in range(num_iteration):
    cnt = 0
    for i, j in r_c: # all combinations of columns
      a_i = A[:, i]
      a_j = A[:, j]
      b_rr = sum(abs(a_i)**2)
      b_cc = sum(abs(a_j)**2)
      b_rc = a_i.T @ a_j
      m_b_rc = abs(b_rc)

      if not m_b_rc: # module of scalar product = 0 => ortog vector
        continue

      tau_m = (b_cc - b_rr) / (2 * m_b_rc)

      if not tau_m:
        continue

      cnt += 1
      t = torch.sign(tau_m) / (abs(tau_m) + torch.sqrt(1.0 + tau_m**2))
      c_t = 1 / torch.sqrt(1.0 + t**2)
      s = b_rc * t * c_t / m_b_rc

      g = torch.eye(n, dtype=A.dtype, device=device)
      g[i, i] = c_t
      g[j, j] = c_t
      g[i, j] = s
      g[j, i] = -s

      A = A @ g
      v = v @ g
    if cnt == 0:
       raise Exception('No rotations performed during sweep.')
    b = A.T @ A
    off = sum(sum(abs(torch.triu(b, 1))**2))/n

    if off/on < tol: # TODO another criteria ?
        break

  b = A.T @ A
  t_ = torch.sqrt(abs(torch.diag(b)))
  ind = torch.argsort(t_, descending=True)
  t_ = t_[ind]
  A = A[:,ind]
  v = v[:,ind]
  u = A / torch.tile(t_.T, (m, 1))
  s = t_
  return (u, s, v)


if __name__ == '__main__':
    dtype = torch.float32
    torch.set_printoptions(precision=1)
    A = torch.randn(8, 4,  dtype=torch.float32, device=device)
    device = 'cuda'

    t1 = time.time()
    u_gold, s_gold, vh_gold = torch.svd(A)
    print(time.time() - t1)

    t1 = time.time()
    u, s, v = svd_jacoby_pytorch(A, device=device)
    print(time.time() - t1)