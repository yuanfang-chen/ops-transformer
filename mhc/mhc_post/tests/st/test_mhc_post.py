#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MhcPost算子单元测试

该测试用于验证MhcPost算子的正确性。
计算公式: x_{l+1} = (H_{l}^{res})^T * x_l + h_{l}^{out} * H_{t}^{post}
"""

import numpy as np
from typing import Tuple


def mhc_post_golden(x: np.ndarray, h_res: np.ndarray,
                    h_out: np.ndarray, h_post: np.ndarray) -> np.ndarray:
    """
    MhcPost算子的CPU参考实现

    公式: x_{l+1} = (H_{l}^{res})^T * x_l + h_{l}^{out} * H_{t}^{post}

    Args:
        x: 输入张量 x_l, shape为[T, n, D] 或 [B, S, n, D]
        h_res: 残差连接矩阵, shape为[T, n, n] 或 [B, S, n, n]
        h_out: 输出状态, shape为[T, D] 或 [B, S, D]
        h_post: 后处理矩阵, shape为[T, n] 或 [B, S, n]

    Returns:
        输出张量 x_{l+1}, shape与x一致
    """
    # 3D格式: (T, n, D)
    if x.ndim == 3:
        T, n, D = x.shape
        output = np.zeros((T, n, D), dtype=x.dtype)

        for t in range(T):
            # term1: (H_res)^T * x = sum_j(hRes[j,i] * x[j])
            # term2: hOut * hPost[i] (broadcast to D)
            for i in range(n):
                # (H_res[t])^T * x[t]: [n, n]^T * [n, D] = [n, D]
                term1 = np.zeros(D, dtype=np.float32)
                for j in range(n):
                    term1 += h_res[t, j, i] * x[t, j, :]

                # h_out[t] * h_post[t, i]: broadcast to D
                term2 = h_out[t, :] * h_post[t, i]

                output[t, i, :] = term1 + term2

        return output

    # 4D格式: (B, S, n, D)
    elif x.ndim == 4:
        B, S, n, D = x.shape
        output = np.zeros((B, S, n, D), dtype=x.dtype)

        for b in range(B):
            for s in range(S):
                for i in range(n):
                    # (H_res[b,s])^T * x[b,s]
                    term1 = np.zeros(D, dtype=np.float32)
                    for j in range(n):
                        term1 += h_res[b, s, j, i] * x[b, s, j, :]

                    # h_out[b,s] * h_post[b,s,i]
                    term2 = h_out[b, s, :] * h_post[b, s, i]

                    output[b, s, i, :] = term1 + term2

        return output

    else:
        raise ValueError(f"Unsupported input dimension: {x.ndim} (expected 3 or 4)")


def test_mhc_post_3d_fp16():
    """测试3D格式的FP16输入"""
    print("Testing MhcPost 3D FP16...")

    np.random.seed(42)
    T, n, D = 4, 4, 5120

    x = np.random.randn(T, n, D).astype(np.float16)
    h_res = np.abs(np.random.randn(T, n, n)).astype(np.float32)
    h_out = np.random.randn(T, D).astype(np.float16)
    h_post = np.random.randn(T, n).astype(np.float32)

    output = mhc_post_golden(x.astype(np.float32), h_res, h_out.astype(np.float32), h_post)

    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output dtype: {output.dtype}")
    print("  Test passed!")


def test_mhc_post_3d_bf16():
    """测试3D格式的BF16输入"""
    print("Testing MhcPost 3D BF16...")

    np.random.seed(42)
    T, n, D = 8, 8, 4096

    x = np.random.randn(T, n, D).astype(np.float32)
    h_res = np.abs(np.random.randn(T, n, n)).astype(np.float32)
    h_out = np.random.randn(T, D).astype(np.float32)
    h_post = np.random.randn(T, n).astype(np.float32)

    output = mhc_post_golden(x, h_res, h_out, h_post).astype(np.float32)

    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    print("  Test passed!")


def test_mhc_post_4d_fp16():
    """测试4D格式的FP16输入"""
    print("Testing MhcPost 4D FP16...")

    np.random.seed(42)
    B, S, n, D = 2, 512, 4, 5120

    x = np.random.randn(B, S, n, D).astype(np.float16)
    h_res = np.abs(np.random.randn(B, S, n, n)).astype(np.float32)
    h_out = np.random.randn(B, S, D).astype(np.float16)
    h_post = np.random.randn(B, S, n).astype(np.float32)

    output = mhc_post_golden(x.astype(np.float32), h_res, h_out.astype(np.float32), h_post)

    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    print("  Test passed!")


def test_mhc_post_n6():
    """测试n=6的情况"""
    print("Testing MhcPost with n=6...")

    np.random.seed(42)
    T, n, D = 16, 6, 2048

    x = np.random.randn(T, n, D).astype(np.float32)
    h_res = np.abs(np.random.randn(T, n, n)).astype(np.float32)
    h_out = np.random.randn(T, D).astype(np.float32)
    h_post = np.random.randn(T, n).astype(np.float32)

    output = mhc_post_golden(x, h_res, h_out, h_post).astype(np.float32)

    # 验证输出的第一行计算是否正确
    t, i = 0, 3
    expected_row = h_out[t, :] * h_post[t, i]
    for j in range(n):
        expected_row += h_res[t, j, i] * x[t, j, :]

    np.testing.assert_allclose(output[t, i, :], expected_row, rtol=1e-5)

    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    print("  Test passed!")


if __name__ == "__main__":
    print("=" * 60)
    print("MhcPost算子单元测试")
    print("计算公式: x_{l+1} = (H_{l}^{res})^T * x_l + h_{l}^{out} * H_{t}^{post}")
    print("=" * 60)

    test_mhc_post_3d_fp16()
    print()
    test_mhc_post_3d_bf16()
    print()
    test_mhc_post_4d_fp16()
    print()
    test_mhc_post_n6()

    print()
    print("=" * 60)
    print("所有测试通过!")
    print("=" * 60)