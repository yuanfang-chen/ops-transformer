import torch
import torch_npu
import numpy as np
import omni_training_custom_ops
import pytest
from torch_npu.testing.testcase import TestCase, run_tests

def verify_result(output, golden, tol = 1e-3):
    output = output.reshape(-1)
    golden = golden.reshape(-1)

    different_element_results = torch.isclose(output, golden, rtol=tol, atol=tol, equal_nan=True)
    different_element_indexes = torch.where(different_element_results == False)[0]

    for index in range(min(len(different_element_indexes), 10)):
        real_index = different_element_indexes[index]
        golden_data = golden[real_index]
        output_data = output[real_index]

        print(
            "data index %06d, expected: %-.9f, actual: %-.9f, rdiff: %-.6d" % (real_index.item(), golden_data.item(), output_data.item(),
            abs(output_data - golden_data) / golden_data)
        )
    error_ratio = float(different_element_indexes.size(0)) / float(golden.size(0))
    print("error ratio: %.6f, tolerance: %.6f" % (error_ratio, tol))
    return error_ratio <= 1e-4

def manifold_constrained_hyper_connection_pre_golden_TND(
    x: torch.Tensor, phi: torch.Tensor, alpha: torch.Tensor, bias: torch.Tensor, gamma: torch.Tensor = None,
    norm_eps: float = 1e-6, hc_eps: float = 1e-6):
    T ,N, D = x.shape
    ND = N * D

    phi = phi.transpose(-1, -2)

    alpha_pre = alpha[0]
    alpha_post = alpha[1]
    alpha_res = alpha[2]

    bias_pre = bias[0 : N]
    bias_post = bias[N : 2 * N]
    bias_res = bias[2 * N : 2 * N + N * N].view(N, N)

    if gamma is not None:
        x_rs = x * gamma
    x_rs = x_rs.reshape(T, ND).float()
    inv_rms = torch.rsqrt(x_rs.square().mean(-1, keepdim=True) + norm_eps)

    H_mix = torch.matmul(x_rs, phi.float())
    H_mix_tmp = H_mix * inv_rms

    H_pre_1, H_post_1, H_res_1 = torch.split(H_mix_tmp, [N, N, N * N], dim=-1)
    H_res_2 = H_res_1.reshape(T, N, N)

    H_pre_2 = alpha_pre * H_pre_1 + bias_pre
    H_pre = torch.sigmoid(H_pre_2) + hc_eps

    H_post_2 = alpha_post * H_post_1 + bias_post
    H_post = 2.0 * torch.sigmoid(H_post_2)

    H_comb_before = alpha_res * H_res_2 + bias_res
    h_in_fp = (H_pre.unsqueeze(-1) * x.float()).sum(dim=1)
    h_in = h_in_fp.to(torch.bfloat16)

    return h_in, H_post, H_comb_before, inv_rms, H_mix, H_pre

@pytest.mark.resources(device="npu:910B", npus_per_node=1)
def test_mhc_pre_case():
    T=1024
    n=8
    D=5120
    x = torch.randn(T, n, D, dtype=torch.bfloat16).npu()
    phi = torch.randn(n * n + 2 * n, n * D, dtype=torch.float32).npu()
    alpha = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32).npu()
    bias_pre = torch.full((n,), 0.01, dtype=torch.float32)
    bias_post = torch.full((n,), 0.01, dtype=torch.float32)
    bias_res = torch.full((n, n), 0.01, dtype=torch.float32)
    bias = torch.cat([bias_pre, bias_post, bias_res.reshape(-1)], dim=0).npu()
    gamma = torch.randn(n, D, dtype=torch.float32).npu()

    out_doc = torch.ops.custom.npu_manifold_constrained_hyper_connection_pre(x, phi, alpha, bias, gamma=gamma, out_flag = 1)
    out_golden = manifold_constrained_hyper_connection_pre_golden_TND(x, phi, alpha, bias, gamma=gamma)

    names = ["h_in", "h_post", "h_comb_before", "inv_rms", "h_mix", "h_pre"]
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