#!/usr/bin/env python3
"""
Pytest file for testing npu_prompt_flash_attention correctness
Reuses functions from benchmark.py
"""

import math
import torch
import torch_npu
from torch_pfa import npu_prompt_flash_attention
import pytest

# Import all functions from benchmark.py
from benchmark import (
    DTYPE, INPUT_LAYOUT, 
    gen_pfa_inputs,
    create_attention_mask,
    ref_prompt_flash_attention_fp32,
    BLOCK_SIZE_Q, BLOCK_SIZE_KV, BLOCK_MASK_SEED,
    BAND_PRE_TOKENS, BAND_POST_TOKENS,
    ATTENTION_MATRIX, RUN_REFERENCE, TORCH_REFERENCE
)

# Test configurations from your request
device = "npu:0"

B_VALS = [1]
H_VALS = [1,2,3,4]
S_VALS = [10_000, 20_000]  # S_q = S_kv
D_VALS = [128]   # head dimension

SPARSITY_VALS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

def setup_module(module):
    """Setup function to run before any tests in this module"""
    torch.npu.set_device(device)

@pytest.mark.parametrize("b", B_VALS)
@pytest.mark.parametrize("h", H_VALS)
@pytest.mark.parametrize("s_kv", S_VALS)
@pytest.mark.parametrize("d", D_VALS)
@pytest.mark.parametrize("sparsity", SPARSITY_VALS)
def test_prompt_flash_attention_correctness(b, h, s_kv, d, sparsity):
    """Test correctness of npu_prompt_flash_attention vs reference implementation"""
    
    # Skip test if sparsity is not compatible with current configuration
    if ATTENTION_MATRIX in ["blocks_optimized", "blocks_optimized_batched"] and sparsity > 0.9:
        pytest.skip("Skipping high sparsity for block optimized modes")
    
    s_q = s_kv
    
    # Create attention mask and parameters
    atten_mask, npu_atten_mask, sabi_blocks, sm, scale, pre_tok, post_tok = create_attention_mask(
        b, h, s_q, s_kv, d, sparsity, ATTENTION_MATRIX
    )
    
    # Generate inputs
    q, k, v, actseqlen, actseqlenkv = gen_pfa_inputs(
        b, h, s_q, s_kv, d, device=device, dtype=DTYPE
    )
    
    # Run our implementation
    out_our = npu_prompt_flash_attention(
        q,
        k,
        v,
        sabi_blocks=sabi_blocks,
        actual_seq_lengths=actseqlen,
        actual_seq_lengths_kv=actseqlenkv,
        num_heads=h,
        num_key_value_heads=h,
        input_layout=INPUT_LAYOUT,
        scale_value=scale,
        atten_mask=npu_atten_mask,
        sparse_mode=sm,
        pre_tokens=pre_tok,
        next_tokens=post_tok,
    )
    
    # Run reference implementation
    if TORCH_REFERENCE:
        out_ref = ref_prompt_flash_attention_fp32(q, k, v, scale, atten_mask=atten_mask)
    else:
        out_ref = torch_npu.npu_fusion_attention(
            q, k, v,
            head_num=h,
            input_layout="BNSD",
            scale=scale,
            pre_tockens=0, next_tockens=0,
            sparse_mode=1,
            atten_mask=atten_mask
        )[0]
    
    # Compare results (moved to CPU for comparison)
    out_our_cpu = out_our.cpu()
    out_ref_cpu = out_ref.cpu()
    
    # Assert correctness with reasonable tolerances
    assert torch.allclose(out_our_cpu, out_ref_cpu, rtol=0.05, atol=0.05), (
        f"Outputs don't match for b={b}, h={h}, s_q={s_q}, s_kv={s_kv}, d={d}, sparsity={sparsity}"
    )

def test_dense_attention_correctness():
    """Test dense attention (no mask) correctness"""
    b, h, s_kv, d = 1, 4, 10000, 128
    s_q = s_kv
    
    # Create inputs
    q, k, v, actseqlen, actseqlenkv = gen_pfa_inputs(
        b, h, s_q, s_kv, d, device=device, dtype=DTYPE
    )
    
    scale = 1.0 / math.sqrt(float(d))
    
    # Run our implementation (dense)
    out_our = npu_prompt_flash_attention(
        q,
        k,
        v,
        actual_seq_lengths=actseqlen,
        actual_seq_lengths_kv=actseqlenkv,
        num_heads=h,
        num_key_value_heads=h,
        input_layout=INPUT_LAYOUT,
        scale_value=scale,
        sparse_mode=0,
    )
    
    # Run reference implementation (dense)
    out_ref = ref_prompt_flash_attention_fp32(q, k, v, scale, atten_mask=None)
    
    # Compare results
    out_our_cpu = out_our.cpu()
    out_ref_cpu = out_ref.cpu()
    
    assert torch.allclose(out_our_cpu, out_ref_cpu, rtol=0.05, atol=0.05), (
        "Dense attention outputs don't match"
    )

def test_lower_triangular_attention_correctness():
    """Test lower triangular attention correctness"""
    b, h, s_kv, d = 1, 2, 5000, 128
    s_q = s_kv
    
    # Create inputs
    q, k, v, actseqlen, actseqlenkv = gen_pfa_inputs(
        b, h, s_q, s_kv, d, device=device, dtype=DTYPE
    )
    
    scale = 1.0 / math.sqrt(float(d))
    
    # Import required functions for this test
    from benchmark import make_lower_triangular_mask
    
    # Create lower triangular mask
    atten_mask = make_lower_triangular_mask(s_q, s_kv, device=device)
    npu_atten_mask = make_lower_triangular_mask(2048, 2048, device=device)
    
    # Run our implementation
    out_our = npu_prompt_flash_attention(
        q,
        k,
        v,
        actual_seq_lengths=actseqlen,
        actual_seq_lengths_kv=actseqlenkv,
        num_heads=h,
        num_key_value_heads=h,
        input_layout=INPUT_LAYOUT,
        scale_value=scale,
        atten_mask=npu_atten_mask,
        sparse_mode=2,
    )
    
    # Run reference implementation
    out_ref = ref_prompt_flash_attention_fp32(q, k, v, scale, atten_mask=atten_mask)
    
    # Compare results
    out_our_cpu = out_our.cpu()
    out_ref_cpu = out_ref.cpu()
    
    assert torch.allclose(out_our_cpu, out_ref_cpu, rtol=0.05, atol=0.05), (
        "Lower triangular attention outputs don't match"
    )

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
