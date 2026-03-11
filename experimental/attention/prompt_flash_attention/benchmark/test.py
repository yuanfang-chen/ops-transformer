#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""
Pytest file for testing torch_pfa.npu_prompt_flash_attention correctness
Reuses functions from benchmark.py
"""

import math
import pytest
import torch
import torch_npu
import torch_pfa
from benchmark import gen_pfa_inputs, create_attention_mask  # data gen
from benchmark import ref_prompt_flash_attention_launcher  # baseline kernel

# Global test configurations
SEED = 42
DEVICE = "npu:0"
DTYPE = torch.bfloat16
INPUT_LAYOUT = 'BNSD'

# Sweeped test parameters
TORCH_REF_VALS = [True, False]  # True=our custom reference model; False = torch_npu official kernel
A_VALS = ["blocks_optimized_batched", "blocks_optimized", "sparse_block_all_same", "lower_triangular", "band", "custom"]
B_VALS = [1]
H_VALS = [1, 2, 3, 4]
S_VALS = [10_000, 20_000, 30_000]  # s_q = s_kv
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
    """Test correctness of torch_pfa.npu_prompt_flash_attention vs reference implementation"""
    
    # Set random seed for reproducible test inputs
    torch.manual_seed(SEED)
    
    # Skip test if sparsity is not compatible with current configuration
    if a in ["blocks_optimized", "blocks_optimized_batched"] and sparsity > 0.9:
        pytest.skip("Skipping high sparsity for block optimized modes")
    
    s_q = s_kv
    
    # Generate attention mask and parameters
    atten_mask, npu_atten_mask, sabi_blocks, sm, scale, pre_tok, post_tok = create_attention_mask(
        b, h, s_q, s_kv, d, sparsity, a, device=DEVICE, emit_atten_mask=True
    )
    
    # Generate input tensors
    q, k, v, actseqlen, actseqlenkv = gen_pfa_inputs(
        b, h, s_q, s_kv, d, device=DEVICE, dtype=DTYPE
    )
    
    # Run our implementation
    out_our = torch_pfa.npu_prompt_flash_attention(q, k, v,
        sabi_blocks=sabi_blocks,
        actual_seq_lengths=actseqlen, actual_seq_lengths_kv=actseqlenkv, 
        num_heads=h, num_key_value_heads=h, input_layout=INPUT_LAYOUT,
        scale_value=scale, atten_mask=npu_atten_mask, sparse_mode=sm, 
        pre_tokens=pre_tok, next_tokens=post_tok,
    )
    
    # Run reference implementation
    out_ref = ref_prompt_flash_attention_launcher(torch_ref, q, k, v, head_num=h, 
                                                  scale=scale, atten_mask=atten_mask, 
                                                  input_layout=INPUT_LAYOUT, 
                                                  force_dense_sm=(sparsity == 0))   
    
    # Compare results (moved to CPU for comparison)
    out_our_cpu = out_our.cpu()
    out_ref_cpu = out_ref.cpu()
    
    # Check correctness with reasonable tolerances
    if not torch.allclose(out_our_cpu, out_ref_cpu, rtol=0.05, atol=0.05):
        pytest.fail(
            f"Outputs don't match for b={b}, h={h}, s_q={s_q}, s_kv={s_kv}, d={d}, sparsity={sparsity}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
