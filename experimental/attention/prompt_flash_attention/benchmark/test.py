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
    gen_pfa_inputs,
    create_attention_mask,
    ref_prompt_flash_attention_launcher,
)

# Test configurations from your request
device = "npu:0"
DTYPE = torch.bfloat16
INPUT_LAYOUT = 'BNSD'

TORCH_REF_VALS = [True, False]  # True=our custom reference model;  Fales = torch_npu official kernel (fusion attention)
A_VALS = ["blocks_optimized_batched", "blocks_optimized", "sparse_block_all_same", "lower_triangular", "band", "custom"]
B_VALS = [1]
H_VALS = [1,2,3,4]
S_VALS = [10_000, 20_000, 30_000]  # S_q = S_kv
D_VALS = [128]   # head dimension
SPARSITY_VALS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


@pytest.mark.parametrize("torch_ref", TORCH_REF_VALS)
@pytest.mark.parametrize("a", A_VALS)
@pytest.mark.parametrize("b", B_VALS)
@pytest.mark.parametrize("h", H_VALS)
@pytest.mark.parametrize("s_kv", S_VALS)
@pytest.mark.parametrize("d", D_VALS)
@pytest.mark.parametrize("sparsity", SPARSITY_VALS)
def test_prompt_flash_attention_correctness(torch_ref, a, b, h, s_kv, d, sparsity):
    """Test correctness of npu_prompt_flash_attention vs reference implementation"""
    
    # Skip test if sparsity is not compatible with current configuration
    if a in ["blocks_optimized", "blocks_optimized_batched"] and sparsity > 0.9:
        pytest.skip("Skipping high sparsity for block optimized modes")
    
    s_q = s_kv
    
    # Create attention mask and parameters
    atten_mask, npu_atten_mask, sabi_blocks, sm, scale, pre_tok, post_tok = create_attention_mask(
        b, h, s_q, s_kv, d, sparsity, a
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
    out_ref = ref_prompt_flash_attention_launcher(torch_ref, q, k, v, head_num=h, 
                                                  scale=scale, atten_mask=atten_mask, 
                                                  input_layout=INPUT_LAYOUT, 
                                                  force_dense_sm=(sparsity == 0))   
    
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
    out_ref = ref_prompt_flash_attention_launcher(torch_ref, q, k, v, head_num=h, 
                                                  scale=scale, atten_mask=None, 
                                                  input_layout=INPUT_LAYOUT, 
                                                  force_dense_sm=True)  
    
    # Compare results
    out_our_cpu = out_our.cpu()
    out_ref_cpu = out_ref.cpu()
    
    assert torch.allclose(out_our_cpu, out_ref_cpu, rtol=0.05, atol=0.05), (
        "Dense attention outputs don't match"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
