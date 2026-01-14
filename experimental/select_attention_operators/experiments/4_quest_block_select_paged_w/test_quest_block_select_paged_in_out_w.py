# -*- coding: utf-8 -*-
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

"""
Single input testing (for debugging) -> Run this file with python <filename>
Wide range testing (for validation) -> Run this file with pytest <filename>
"""
import pytest
import torch
import torch_npu
from select_attn_ops import quest_block_select_paged_in_out_w
from ref_quest_block_select_paged_w import ref_quest_block_select_paged_w
from gen_data_quest_block_select_paged_w import gen_quest_paged_w_inputs, compare_indices


DEVICE = "npu:0"
BLOCK_SIZE = 128
HEAD_DIM = 128
SAME_SEQ_LEN_ALL_REQS = False


# --------------------------------------------------------------------------- #
# Central test worker  (assertion crashes <--> test failed)
# --------------------------------------------------------------------------- #
@pytest.mark.skip(reason="Skipping direct invocation of the main test_quest_paged_kernel function")
def test_quest_paged_kernel(dtype: torch.dtype, batch_size: int, num_heads: int, num_kv_heads: int, block_size: int, 
                            head_dim: int, mmbpr: int, k: int, verbose: bool = False) -> None:
    """
    Main test function for paged quest kernel
    """
    # Generate input data
    num_meta_blocks = batch_size * mmbpr
    query, maxblocks, minblocks, metadata_block_tables, seq_lens, tokens_since_metadata_update = (
        gen_quest_paged_w_inputs(
            batch_size, num_heads, num_kv_heads, block_size, head_dim, num_meta_blocks, mmbpr, SAME_SEQ_LEN_ALL_REQS,
            DEVICE, dtype
        )
    )

    # Run reference implementation
    ref_ids = ref_quest_block_select_paged_w(
        query, maxblocks, minblocks, metadata_block_tables, seq_lens, tokens_since_metadata_update, k, fast=True
    )
    
    # Run custom implementation
    custom_ids = torch.zeros_like(ref_ids)
    quest_block_select_paged_in_out_w(
        query, maxblocks, minblocks, metadata_block_tables, seq_lens, tokens_since_metadata_update, custom_ids
    )
    
    # Compare results
    cfg_str = f"{dtype=}, {batch_size=}, {num_heads=}, {num_kv_heads=}, {block_size=}, {head_dim=}, {mmbpr=}, {k=}"
    if verbose:
        print("Input shapes:")
        print(f"  query: {query.shape}")
        print(f"  maxblocks: {maxblocks.shape}")
        print(f"  minblocks: {minblocks.shape}")
        print(f"  metadata_block_tables: {metadata_block_tables.shape}")
        print(f"  seq_lens.shape: {seq_lens.shape}")
        print(f"  seq_lens: {seq_lens.tolist()}")
        print(f"  tokens_since_metadata_update: {tokens_since_metadata_update}")
        print(" ========== Reference torch implementation output ========== ")
        print("ids reference:", ref_ids)
        print(ref_ids.shape)       
        print(" ========== Custom ascendc implementation output =========== ")
        print("ids custom:", custom_ids)
        print(custom_ids.shape)
        print(" ========== DIFF =========== ")
        print("ids diff:", ref_ids - custom_ids)
        print(" ==================== SUMMARY =================== ")
        print(cfg_str)
    indices_match = compare_indices(ref_ids, custom_ids, verbose=verbose)               
    
    # Assert all comparisons pass
    assert indices_match, f"Indices comparison failed for {cfg_str}"


def construct_quest_paged_parameter_sets(dtype_vals, batch_size_vals, num_heads_vals, num_kv_heads_vals, mmbpr_vals, 
                                         k_vals):
    """
    Construct parameter sets for paged Quest kernel testing with constraints:
    - head_dim, block_size are fixed
    - num_kv_heads ≤ num_heads
    - num_heads is a multiple of num_kv_heads (num_heads = num_kv_heads * G for natural G)
    """
    param_set_lst = []
    for dtype in dtype_vals:
        for b in batch_size_vals:
            for h in num_heads_vals:
                for n in num_kv_heads_vals:
                    # Apply constraints: num_kv_heads ≤ num_heads and num_heads is multiple of num_kv_heads
                    if n <= h and h % n == 0:
                        for mmbpr in mmbpr_vals:
                            for k in k_vals:
                                param_set_lst.append((dtype, b, h, n, BLOCK_SIZE, HEAD_DIM, mmbpr, k))
    return param_set_lst

########################### Test 1 - Basic functionality ###########################
parameter_sets = construct_quest_paged_parameter_sets(dtype_vals=[torch.float16, torch.bfloat16], 
                                                      batch_size_vals=[1, 2], 
                                                      num_heads_vals=[8, 16], 
                                                      num_kv_heads_vals=[4, 8], 
                                                      mmbpr_vals=[1, 2, 3, 4], 
                                                      k_vals=[8])


@pytest.mark.parametrize(
    "dtype, batch_size, num_heads, num_kv_heads, block_size, head_dim, mmbpr, k", parameter_sets,
    ids=[f"dtype={dtype}, batch_size={b}, num_heads={h}, num_kv_heads={n}, block_size={block_size}, "
         f"head_dim={d}, mmbpr={mmbpr}, k={k}" 
         for dtype, b, h, n, block_size, d, mmbpr, k in parameter_sets]    
)


@torch.inference_mode()
def test_quest_paged_basic(dtype: torch.dtype, batch_size: int, num_heads: int, num_kv_heads: int, 
                           block_size: int, head_dim: int, mmbpr: int, k: int) -> None:
    test_quest_paged_kernel(dtype, batch_size, num_heads, num_kv_heads, block_size, head_dim, mmbpr, k)

########################### Test 2 - Edge cases ###########################
parameter_sets = construct_quest_paged_parameter_sets(dtype_vals=[torch.float16, torch.bfloat16], 
                                                      batch_size_vals=[1], 
                                                      num_heads_vals=[1, 2, 4, 8, 16, 32], 
                                                      num_kv_heads_vals=[1, 2, 4, 8, 16], 
                                                      mmbpr_vals=[1, 2, 4], 
                                                      k_vals=[8, 16])


@pytest.mark.parametrize(
    "dtype, batch_size, num_heads, num_kv_heads, block_size, head_dim, mmbpr, k", parameter_sets,
    ids=[f"dtype={dtype}, batch_size={b}, num_heads={h}, num_kv_heads={n}, block_size={block_size}, "
         f"head_dim={d}, mmbpr={mmbpr}, k={k}" 
         for dtype, b, h, n, block_size, d, mmbpr, k in parameter_sets]  
)


@torch.inference_mode()
def test_quest_paged_edge_cases(dtype: torch.dtype, batch_size: int, num_heads: int, num_kv_heads: int, 
                                block_size: int, head_dim: int, mmbpr: int, k: int) -> None:
    test_quest_paged_kernel(dtype, batch_size, num_heads, num_kv_heads, block_size, head_dim, mmbpr, k)

########################### Test 3 - Extensive testing ###########################
parameter_sets = construct_quest_paged_parameter_sets(dtype_vals=[torch.float16, torch.bfloat16], 
                                                      batch_size_vals=[1, 2, 8], 
                                                      num_heads_vals=[8, 16, 32, 64], 
                                                      num_kv_heads_vals=[2, 4, 8, 16], 
                                                      mmbpr_vals=[1, 2, 3, 4], 
                                                      k_vals=[8, 16, 32])


@pytest.mark.parametrize(
    "dtype, batch_size, num_heads, num_kv_heads, block_size, head_dim, mmbpr, k", parameter_sets,
    ids=[f"dtype={dtype}, batch_size={b}, num_heads={h}, num_kv_heads={n}, block_size={block_size}, "
         f"head_dim={d}, mmbpr={mmbpr}, k={k}" 
         for dtype, b, h, n, block_size, d, mmbpr, k in parameter_sets]  
)


@torch.inference_mode()
def test_quest_paged_extensive(dtype: torch.dtype, batch_size: int, num_heads: int, num_kv_heads: int, 
                               block_size: int, head_dim: int, mmbpr: int, k: int) -> None:
    test_quest_paged_kernel(dtype, batch_size, num_heads, num_kv_heads, block_size, head_dim, mmbpr, k)

########################### Test 4 - Large scale ###########################
parameter_sets = construct_quest_paged_parameter_sets(dtype_vals=[torch.float16, torch.bfloat16], 
                                                      batch_size_vals=[16, 20, 24, 32], 
                                                      num_heads_vals=[32, 64, 128], 
                                                      num_kv_heads_vals=[4, 8, 16, 32, 64], 
                                                      mmbpr_vals=[1, 2, 4], 
                                                      k_vals=[8, 32, 64])


@pytest.mark.parametrize(
    "dtype, batch_size, num_heads, num_kv_heads, block_size, head_dim, mmbpr, k", parameter_sets,
    ids=[f"dtype={dtype}, batch_size={b}, num_heads={h}, num_kv_heads={n}, block_size={block_size}, head_dim={d}, "
         f"mmbpr={mmbpr}, k={k}" 
         for dtype, b, h, n, block_size, d, mmbpr, k in parameter_sets]  
)


@torch.inference_mode()
def test_quest_paged_large_scale(dtype: torch.dtype, batch_size: int, num_heads: int, num_kv_heads: int, 
                                 block_size: int, head_dim: int, mmbpr: int, k: int) -> None:
    test_quest_paged_kernel(dtype, batch_size, num_heads, num_kv_heads, block_size, head_dim, mmbpr, k)


# --------------------------------------------------------------------------- #
# Quick manual run (kept for copy-paste debugging)
# ---------------------------------------------------------------------------#
if __name__ == "__main__":
    test_quest_paged_kernel(dtype=torch.bfloat16, batch_size=20, num_heads=32, num_kv_heads=8, block_size=128, 
                            head_dim=128, mmbpr=1, k=8, verbose=True)  # fails bfloat16
    print("Manual smoke test PASSED")

