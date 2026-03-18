# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import torch
import torch_npu
import numpy as np
import torch.nn.functional as F


def verify_result(output, golden, tol=1e-3):
    output = output.reshape(-1)
    golden = golden.reshape(-1)

    different_element_results = torch.isclose(output, golden, rtol=tol, atol=tol, equal_nan=True)
    different_element_indexes = torch.where(different_element_results == False)[0]

    for index in range(min(len(different_element_indexes), 10)):
        real_index = different_element_indexes[index]
        golden_data = golden[real_index].detach().numpy()
        output_data = output[real_index].detach().numpy()

        print(
            "data index %06d, expected: %-.9f, actual: %-.9f, rdiff: %-.6f" % (real_index.item(), golden_data.item(),
            output_data.item(), abs(output_data - golden_data) / (abs(golden_data) + 1e-6))
        )
    error_ratio = float(different_element_indexes.size(0)) / float(golden.size(0))
    print("error ratio: %.6f %%, tolerance: %.4f" % (error_ratio * 100.0, tol))
    return error_ratio <= 1e-4


def mhc_pre_golden_tnd(
    x: torch.Tensor, phi: torch.Tensor, alpha: torch.Tensor, bias: torch.Tensor, gamma: torch.Tensor = None,
    norm_eps: float=1e-6, hc_eps: float=1e-6):
    t, n, d = x.shape
    nd = n * d
    x = x.reshape(t, nd).float()
    inv_rms = torch.rsqrt(x.square().mean(-1, keepdim=True) + norm_eps)

    if gamma is not None:
        gamma = gamma.reshape(n * d)
        h_mix = F.linear((x * gamma), phi.float())
        weight = h_mix * inv_rms
    else:
        h_mix = F.linear(x, phi.float())
        weight = h_mix * inv_rms
    
    h_pre, h_post, h_res = weight.split([n, n, n * n], dim=-1)
    h_res = h_res.unflatten(-1, (n, n))
    h_pre = F.sigmoid(h_pre * alpha[0] + bias[:n].unsqueeze(0).unsqueeze(0)) + hc_eps
    h_post = 2 * F.sigmoid(h_post * alpha[1] + bias[n:2 * n].unsqueeze(0).unsqueeze(0))
    h_res = h_res * alpha[2] + bias[2 * n:].view(n, n).unsqueeze(0).unsqueeze(0)
    
    y = torch.sum(h_pre.unsqueeze(-1) * x.unflatten(dim=-1, sizes=(n, -1)), dim=2)

    return (y.bfloat16(), h_post, h_res, inv_rms, h_mix, h_pre)


def test_mhc_pre_case():
    t = 1024
    n = 4
    d = 2560
    x = torch.randn(t, n, d, dtype=torch.bfloat16).npu()
    phi = torch.randn(n * n + 2 * n, n * d, dtype=torch.float32).npu()
    alpha = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32).npu()
    gamma = torch.randn(n, d, dtype=torch.float32).npu()
    bias_pre = torch.full((n,), 0.01, dtype=torch.float32)
    bias_post = torch.full((n,), 0.01, dtype=torch.float32)
    bias_res = torch.full((n, n), 0.01, dtype=torch.float32)
    bias = torch.cat([bias_pre, bias_post, bias_res.reshape(-1)], dim=0).npu()

    out_doc = torch_npu.npu_mhc_pre(x, phi, alpha, bias, gamma = gamma, out_flag = 1)
    out_golden = mhc_pre_golden_tnd(x.cpu(), phi.cpu(), alpha.cpu(), bias.cpu(), gamma = gamma.cpu())

    names = ["h_in", "h_post", "h_res", "inv_rms", "h_mix", "h_pre"]
    res = True
    for name, a, b in zip(names, out_doc, out_golden):
        print(f"{name=}")
        if name == 'h_in':
            resflag = verify_result(a.float().cpu(), b.float().cpu(), 2**-7)
        else:
            resflag = verify_result(a.float().cpu(), b.float().cpu())
        if resflag == False:
            res = False
    if res:
        assert True
    else:
        assert False


if __name__ == "__main__":
    test_mhc_pre_case()