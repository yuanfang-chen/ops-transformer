import math
import torch
import torch_npu
import ascendc_ops


def test_basic_decode():
    """Basic decode: BF16, BNSD, B=1, N=32, S1=1, S2=1024, D=128, no mask, no PA."""
    B, N, S1, S2, D = 1, 32, 1, 1024, 128
    scale = 1.0 / math.sqrt(D)

    query = torch.randn(B, N, S1, D, dtype=torch.bfloat16).npu()
    key = torch.randn(B, N, S2, D, dtype=torch.bfloat16).npu()
    value = torch.randn(B, N, S2, D, dtype=torch.bfloat16).npu()

    out = ascendc_ops.fias_decode(
        query, key, value,
        num_heads=N, num_kv_heads=N,
        scale_value=scale,
    )

    assert out.shape == (B, N, S1, D), f"Expected shape {(B, N, S1, D)}, got {out.shape}"
    assert out.dtype == torch.bfloat16, f"Expected bfloat16, got {out.dtype}"
    print(f"PASS test_basic_decode: output shape={out.shape}")


def test_gqa_decode():
    """GQA decode: N=32 query heads, N_kv=8 KV heads (GQA ratio=4)."""
    B, N, N_kv, S1, S2, D = 1, 32, 8, 1, 1024, 128
    scale = 1.0 / math.sqrt(D)

    query = torch.randn(B, N, S1, D, dtype=torch.bfloat16).npu()
    key = torch.randn(B, N_kv, S2, D, dtype=torch.bfloat16).npu()
    value = torch.randn(B, N_kv, S2, D, dtype=torch.bfloat16).npu()

    out = ascendc_ops.fias_decode(
        query, key, value,
        num_heads=N, num_kv_heads=N_kv,
        scale_value=scale,
    )

    assert out.shape == (B, N, S1, D), f"Expected shape {(B, N, S1, D)}, got {out.shape}"
    print(f"PASS test_gqa_decode: output shape={out.shape}")


def test_with_mask():
    """Decode with attention mask."""
    B, N, S1, S2, D = 1, 32, 1, 1024, 128
    scale = 1.0 / math.sqrt(D)

    query = torch.randn(B, N, S1, D, dtype=torch.bfloat16).npu()
    key = torch.randn(B, N, S2, D, dtype=torch.bfloat16).npu()
    value = torch.randn(B, N, S2, D, dtype=torch.bfloat16).npu()
    mask = torch.ones(1, 1, S1, S2, dtype=torch.bool).npu()

    out = ascendc_ops.fias_decode(
        query, key, value,
        atten_mask=mask,
        num_heads=N, num_kv_heads=N,
        scale_value=scale,
    )

    assert out.shape == (B, N, S1, D), f"Expected shape {(B, N, S1, D)}, got {out.shape}"
    print(f"PASS test_with_mask: output shape={out.shape}")


def test_d64():
    """Decode with D=64 (Config 1 path)."""
    B, N, S1, S2, D = 1, 32, 1, 512, 64
    scale = 1.0 / math.sqrt(D)

    query = torch.randn(B, N, S1, D, dtype=torch.bfloat16).npu()
    key = torch.randn(B, N, S2, D, dtype=torch.bfloat16).npu()
    value = torch.randn(B, N, S2, D, dtype=torch.bfloat16).npu()

    out = ascendc_ops.fias_decode(
        query, key, value,
        num_heads=N, num_kv_heads=N,
        scale_value=scale,
    )

    assert out.shape == (B, N, S1, D), f"Expected shape {(B, N, S1, D)}, got {out.shape}"
    print(f"PASS test_d64: output shape={out.shape}")


if __name__ == "__main__":
    test_basic_decode()
    test_gqa_decode()
    test_with_mask()
    test_d64()
    print("\nAll tests passed!")
