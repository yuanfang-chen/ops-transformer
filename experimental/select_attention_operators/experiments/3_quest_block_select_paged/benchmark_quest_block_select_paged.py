# -*- coding: utf-8 -*-
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

"""

Benchmark driver for quest_block_select_paged kernel.

Measures:
*  latency (usec) - NPU timer
*  effective bandwidth (TB/s) - bytes moved / time
*  correctness comparison with reference implementation
"""
from typing import Callable
import itertools
import torch
import torch_npu

from select_attn_ops import quest_block_select_paged, quest_block_select_paged_in_out
from ref_quest_block_select_paged import ref_quest_block_select_paged
from gen_data_quest_block_select_paged import gen_quest_paged_inputs, ceil_div, compare_indices


torch.npu.set_device("npu:0")
BLOCK_SIZE = 128
HEAD_DIM = 128
SAME_SEQ_LEN_ALL_REQS = True # to set equally long input length and avoid unknown actual size


# --------------------------------------------------------------------------- #
#  bytes-moved calculator
# --------------------------------------------------------------------------- #
def bytes_moved_paged_select(batch_size: int, num_heads: int, num_kv_heads: int, block_size: int, 
                            head_dim: int, mmbpr: int, k: int, seq_lens: torch.Tensor) -> int:
    """
    Global-memory traffic (read + write) for quest_block_select_paged.

    Reads:
    - query: [batch_size, num_heads, head_dim] - batch_size * num_heads * head_dim * 2 bytes (FP16)
    - maxblocks: [?, block_size, num_kv_heads, head_dim] - num_effective_metadata_blocks * block_size * 
                                                            num_kv_heads * head_dim * 2 bytes (FP16)
    - minblocks: [?, block_size, num_kv_heads, head_dim] - num_effective_metadata_blocks * block_size * 
                                                            num_kv_heads * head_dim * 2 bytes (FP16)
    - metadata_block_tables: [batch_size, mmbpr] - batch_size * mmbpr * 4 bytes (INT32)
    - seq_lens: [batch_size] - batch_size * 4 bytes (INT32)

    Writes:
    - selected_indices: [batch_size, num_kv_heads, k] - batch_size * num_kv_heads * k * 4 bytes (INT32)
    """
    toks_per_meta_block = block_size * block_size
    num_effective_metadata_blocks = torch.sum(ceil_div(seq_lens, toks_per_meta_block)).item()

    read_query = batch_size * num_heads * head_dim * 2
    read_maxblocks = num_effective_metadata_blocks * block_size * num_kv_heads * head_dim * 2
    read_minblocks = num_effective_metadata_blocks * block_size * num_kv_heads * head_dim * 2
    read_metadata_tables = batch_size * mmbpr * 4
    read_seq_lens = seq_lens.element_size() * seq_lens.numel()
    
    write_indices = batch_size * num_kv_heads * k * 4
    
    return read_query + read_maxblocks + read_minblocks + read_metadata_tables + read_seq_lens + write_indices


# --------------------------------------------------------------------------- #
#  benchmark body
# --------------------------------------------------------------------------- #
def benchmark_quest_block_select_paged(custom_kernel: Callable, dtype: torch.dtype):
    """
    Benchmark the custom kernel against reference kernel across a range of input shapes. Show latency and bandwidth.
    :param custom_kernel:     a callable kernel: either quest_block_select_paged or quest_block_select_paged_in_out
    """
    run_our = True
    run_ref = True
    n_repeat = 10
    n_warmup = 1
    
    batch_size_vals = [10, 20, 24, 32]
    num_heads_vals = [32]
    num_kv_heads_vals = [8]
    mmbpr_vals = [1, 2, 4, 6] # if custom_kernel == quest_block_select_paged_in_out else [1, 2, 4, 8, 16]

    if custom_kernel == quest_block_select_paged:
        k_vals = [4, 8, 12, 16]
    elif custom_kernel == quest_block_select_paged_in_out:
        k_vals = [8, 16, 24, 32]
    else:
        raise ValueError(f"Unknown custom_kernel: {custom_kernel}")

    if not run_our and not run_ref:
        print("Nothing to run, must set run_our=True or run_ref=True")
        return

    print("=" * 124)
    print(f"  custom_kernel={custom_kernel.__name__} {dtype=}\n  {BLOCK_SIZE=}  {HEAD_DIM=}  {SAME_SEQ_LEN_ALL_REQS=}")
    print("=" * 124)
    print(f"{'H':>3} {'N':>3} {'B':>3} {'MMBPR':>6} {'Max_seq_len':>12} {'k':>4} "
          f"{'Outputs_equal':>15} {'Ref_Latency_[usec]':>18} {'Our_Latency_[usec]':>18} {'Ref_BW_[TB/sec]':>16} "
          f"{'Our_BW_[TB/sec]':>16}")
    print("-" * 124)

    for b, h, n, mmbpr, k in itertools.product(batch_size_vals, num_heads_vals, num_kv_heads_vals, mmbpr_vals, k_vals):
        
        ######## Check correctness #######
        are_equal = "N/A"
        if run_our and run_ref:
            query, maxblocks, minblocks, metadata_block_tables, seq_lens = gen_quest_paged_inputs(
                    b, h, n, BLOCK_SIZE, HEAD_DIM,
                    num_meta_blocks=b * mmbpr,
                    mmbpr=mmbpr,
                    same_seq_len_all_reqs=SAME_SEQ_LEN_ALL_REQS,
                    device="npu:0", 
                    dtype=dtype)
            ref_ids = ref_quest_block_select_paged(query, maxblocks, minblocks, metadata_block_tables, seq_lens, k)
            if custom_kernel == quest_block_select_paged:
                our_ids = quest_block_select_paged(query, maxblocks, minblocks, metadata_block_tables, seq_lens, k)
            elif custom_kernel == quest_block_select_paged_in_out:
                our_ids = torch.zeros((b, n, k), dtype=torch.int32, device=query.device)
                quest_block_select_paged_in_out(query, maxblocks, minblocks, metadata_block_tables, seq_lens, our_ids)
            else:
                raise ValueError(f"Unknown custom_kernel: {custom_kernel}")            
            tol_percentage = 0.02
            are_equal = compare_indices(ref_ids, our_ids, tol_percentage, verbose=False)
            are_equal = "yes" if are_equal else "no"
        
        ########## Our Implementation ##########
        # Our implementation - generate multiple input sets
        if run_our:        
            input_sets = []
            for i in range(n_warmup + n_repeat):
                query, maxblocks, minblocks, metadata_block_tables, seq_lens = \
                    gen_quest_paged_inputs(
                        b, h, n, BLOCK_SIZE, HEAD_DIM,
                        num_meta_blocks=b * mmbpr,
                        mmbpr=mmbpr,
                        same_seq_len_all_reqs=SAME_SEQ_LEN_ALL_REQS,
                        device="npu:0", 
                        dtype=dtype)
                input_sets.append((query, maxblocks, minblocks, metadata_block_tables, seq_lens))
            
            # Our implementation - Warm-up runs
            for i in range(n_warmup):
                query, maxblocks, minblocks, metadata_block_tables, seq_lens = input_sets[i]
                if run_our:
                    if custom_kernel == quest_block_select_paged:
                        quest_block_select_paged(query, maxblocks, minblocks, metadata_block_tables, seq_lens, k)
                    elif custom_kernel == quest_block_select_paged_in_out:
                        quest_block_select_paged_in_out(query, maxblocks, minblocks, metadata_block_tables, seq_lens, 
                                                        our_ids)
                    else:
                        raise ValueError(f"Unknown custom_kernel: {custom_kernel}") 
                if run_ref:
                    ref_quest_block_select_paged(query, maxblocks, minblocks, metadata_block_tables, seq_lens, k)
            torch.npu.synchronize()

            # Our implementation - measurements 
            our_duration = None
            our_bw = None
            start = torch.npu.Event(enable_timing=True)
            end = torch.npu.Event(enable_timing=True)

            start.record()
            for i in range(n_warmup, n_warmup + n_repeat):
                query, maxblocks, minblocks, metadata_block_tables, seq_lens = input_sets[i]
                if custom_kernel == quest_block_select_paged:
                    quest_block_select_paged(query, maxblocks, minblocks, metadata_block_tables, seq_lens, k)
                elif custom_kernel == quest_block_select_paged_in_out:
                    quest_block_select_paged_in_out(query, maxblocks, minblocks, metadata_block_tables, seq_lens, 
                                                    our_ids)
                else:
                    raise ValueError(f"Unknown custom_kernel: {custom_kernel}")                
            end.record()
            torch.npu.synchronize()
            
            our_duration = start.elapsed_time(end) / n_repeat * 1000  # ms to μs
            total_bytes = bytes_moved_paged_select(b, h, n, BLOCK_SIZE, HEAD_DIM, mmbpr, k, seq_lens)
            our_bw = total_bytes / our_duration / 1e6  # TB/s


        ########## Reference ##########
        # Reference implementation - generate multiple input sets
        if run_ref:        
            input_sets = []
            for i in range(n_warmup + n_repeat):
                query, maxblocks, minblocks, metadata_block_tables, seq_lens = \
                    gen_quest_paged_inputs(b, h, n, BLOCK_SIZE, HEAD_DIM,
                        num_meta_blocks=b * mmbpr,
                        mmbpr=mmbpr, same_seq_len_all_reqs=SAME_SEQ_LEN_ALL_REQS,
                        device="npu:0", dtype=dtype)
                input_sets.append((query, maxblocks, minblocks, metadata_block_tables, seq_lens))
            
            # Reference implementation - Warm-up runs
            for i in range(n_warmup):
                query, maxblocks, minblocks, metadata_block_tables, seq_lens = input_sets[i]
                ref_quest_block_select_paged(query, maxblocks, minblocks, metadata_block_tables, seq_lens, k)
            torch.npu.synchronize()
            
            # Reference implementation - measurement
            ref_duration = None
            ref_bw = None
            start = torch.npu.Event(enable_timing=True)
            end = torch.npu.Event(enable_timing=True)
            
            start.record()
            for repetition in range(n_warmup, n_warmup + n_repeat):
                query, maxblocks, minblocks, metadata_block_tables, seq_lens = input_sets[repetition]
                ref_quest_block_select_paged(query, maxblocks, minblocks, metadata_block_tables, seq_lens, k)
            end.record()
            torch.npu.synchronize()
            
            ref_duration = start.elapsed_time(end) / n_repeat * 1000  # ms to μs
            total_bytes_moved = bytes_moved_paged_select(b, h, n, BLOCK_SIZE, HEAD_DIM, mmbpr, k, seq_lens)
            ref_bw = total_bytes_moved / ref_duration / 1e6  # TB/s
        
        ####### Print results row into the table #######
        max_seq_len = mmbpr * BLOCK_SIZE * BLOCK_SIZE
        print(f"{h:>3} {n:>3} {b:>3} {mmbpr:>6} {max_seq_len:>12} {k:>4} {are_equal:>15} ", end='')
        
        if ref_duration and run_ref is not None:
            print(f"{ref_duration:>18.2f} ", end='')
        else:
            print(f"{'N/A':>18} ", end='')

        if our_duration and run_our is not None:
            print(f"{our_duration:>18.2f} ", end='')
        else:
            print(f"{'N/A':>18} ", end='')

        if ref_bw and run_ref is not None:
            print(f"{ref_bw:>16.3f} ", end='')
        else:
            print(f"{'N/A':>16} ", end='')

        if our_bw and run_our is not None:
            print(f"{our_bw:>16.3f}")
        else:
            print(f"{'N/A':>16}")

    print("=" * 124)


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    benchmark_quest_block_select_paged(custom_kernel=quest_block_select_paged, dtype=torch.float16)
    benchmark_quest_block_select_paged(custom_kernel=quest_block_select_paged_in_out, dtype=torch.float16)
    benchmark_quest_block_select_paged(custom_kernel=quest_block_select_paged, dtype=torch.bfloat16)
    benchmark_quest_block_select_paged(custom_kernel=quest_block_select_paged_in_out, dtype=torch.bfloat16)