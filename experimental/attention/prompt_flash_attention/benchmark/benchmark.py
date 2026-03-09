#!/usr/bin/env python3
# benchmark_prompt_flash_attention_pfa.py
"""
Benchmark driver for torch_pfa.npu_prompt_flash_attention.

Measures:
*  latency (usec) - NPU timer
*  effective bandwidth (TB/s) - bytes moved / time
*  correctness comparison with a reference PyTorch implementation
"""

import math
import itertools

import torch
import torch_npu
# from torch_npu import npu_prompt_flash_attention  # UNCOMMENT TO USE OFF-THE-SHELF torch_npu interface
from torch_pfa import npu_prompt_flash_attention

device = "npu:0"

DTYPE = torch.bfloat16
INPUT_LAYOUT = "BNSD"  # [B, num_heads, seq_len, head_dim]

# Parameter sweeps (adjust as needed)
B_VALS = [1]
H_VALS = [3]
S_VALS = [118_806]  # S_q = S_kv
# S_VALS = [10_000]  # S_q = S_kv
D_VALS = [128]   # head dimension

N_REPEATS = 10
N_WARMUP = 2

# Kind of attention matrix
# 'sparse_block' does not pass tests because current kernel does not support different masks for different heads.
# 'sparse_block_all_same' is the same, but with all masks which are the same.
# 'blocks_optimized' is the new optimized version written by us
ATTENTION_MATRIX = "blocks_optimized_batched"   # "dense", "sparse_block", "sparse_block_all_same", "lower_triangular", "band", "custom", "blocks_optimized" "blocks_optimized_batched"

# For block mask and vertical band mask
SPARSITY_VALS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# For the block mask
BLOCK_SIZE_Q = 128
BLOCK_SIZE_KV = 512
BLOCK_MASK_SEED = 1234
USE_FRAME = True

# For the band mask
BAND_PRE_TOKENS = 8
BAND_POST_TOKENS = 2

# Print tensors for manual comparisons
PRINT_OUTPUTS = False
PRINT_MASK = False
# For printing tensor differneces in blocks
PRINT_BLOCK_EQUALITY = False
PRINT_HEIGHT = 128
PRINT_WIDTH = 8

RUN_REFERENCE = False  # True <--> enables accuracy compariston
TORCH_REFERENCE = False  # If False, will instead run the torch_npu reference

torch.set_printoptions(
    threshold=100_000_000,
    linewidth=400,
    edgeitems=3,
    precision=4,
    sci_mode=False
)


def ref_prompt_flash_attention_launcher(torch_reference:bool, q:torch.Tensor, k:torch.Tensor, v:torch.Tensor, head_num:int, scale:float, atten_mask:torch.Tensor, input_layout:str, run_ref_sparsity_0:bool) -> torch.Tensor:
    """
    runs a reference prompt_flash attention, for correctness comparisons and for baseline time measurements.
    torch_reference = True - launch our custom pythonic model "ref_prompt_flash_attention_fp32" 
                      False -> launch the torch_npu.npu_fusion_attention
    run_ref_sparsity_0 relevant only for torch_interface=False:
                      True - apply sparse mode 0 (dense) and don't use the provided atten_mask
                      False - apply sparse mode 1 (sparse token mask) and use the provided atten_mask
     
    """
    if torch_reference: # use our custom pythonic model
        return ref_prompt_flash_attention_fp32(q, k, v, scale, atten_mask=atten_mask)
    else:
        return torch_npu.npu_fusion_attention(q, k, v, head_num=head_num, input_layout=INPUT_LAYOUT, 
                                       scale=scale, pre_tockens=0, next_tockens=0, 
                                       atten_mask=atten_mask,
                                       sparse_mode=0 if run_ref_sparsity_0 else 1)[0]    

# --------------------------------------------------------------------------- #
#  bytes-moved calculator
# --------------------------------------------------------------------------- #
def bytes_moved_pfa(B: int, H: int, S_q: int, S_kv: int, D: int) -> int:
    """
    Approximate global-memory traffic (read + write) for npu_prompt_flash_attention.

    Reads:
    - q: [B, H, S_q, D]    -> B * H * S_q  * D * 2 bytes (FP16)
    - k: [B, H, S_kv, D]   -> B * H * S_kv * D * 2 bytes (FP16)
    - v: [B, H, S_kv, D]   -> B * H * S_kv * D * 2 bytes (FP16)
    - actual_seq_lengths:      B * 4 bytes (INT32, approximated)
    - actual_seq_lengths_kv:   B * 4 bytes (INT32, approximated)

    Writes:
    - out: [B, H, S_q, D]  -> B * H * S_q * D * 2 bytes (FP16)
    """
    elem_size = 2  # FP16
    read_q = B * H * S_q * D * elem_size
    read_k = B * H * S_kv * D * elem_size
    read_v = B * H * S_kv * D * elem_size
    write_out = B * H * S_q * D * elem_size

    seq_lens_bytes = (B + B) * 4  # int32 arrays for q and kv

    return read_q + read_k + read_v + write_out + seq_lens_bytes

# --------------------------------------------------------------------------- #
#  block indices generator (based on sparsity)
# --------------------------------------------------------------------------- #

def generate_sparse_blocks_by_row(
    S_q: int,
    S_kv: int,
    block_size_q: int,
    block_size_kv: int,
    sparsity: float,
    seed: int,
) -> list[list[int]]:
    """
    Returns sparse block columns per block-row.

    Output shape: [n_block_rows][K]
      - n_block_rows = ceil(S_q / block_size)
      - K = round(n_block_cols * (1 - sparsity)), clamped to [1, n_block_cols]
      - values in [0, n_block_cols)
      - each inner list is sorted
    """
    sparsity = max(0.0, min(1.0, float(sparsity)))

    n_block_rows = math.ceil(S_q / block_size_q)
    n_block_cols = math.ceil(S_kv / block_size_kv)
    if n_block_rows == 0 or n_block_cols == 0:
        return []

    K = int(round(n_block_cols * (1.0 - sparsity)))
    K = max(1, min(K, n_block_cols))

    g = torch.Generator(device="cpu")
    g.manual_seed(seed)

    rows: list[list[int]] = []
    for _ in range(n_block_rows):
        cols = torch.randperm(n_block_cols, generator=g)[:K].tolist()
        # cols.sort()
        rows.append(cols)

    return rows


def generate_sparse_blocks_by_row_with_frame(
    S_q: int,
    S_kv: int,
    block_size_q: int,
    block_size_kv: int,
    sparsity: float,
    seed: int,
    pad_value: int = -1,
) -> list[list[int]]:
    """
    Constraints (post-sparsity selection):
      - first 8 block-cols of each row are always included (if they exist)
      - first 29 block-rows contain all blocks
      - last row contains all blocks
      - each row always includes the last block-col
    Output:
      - shape [n_block_rows][n_block_cols]
      - each row: sorted selected cols, then pad_value to the end
    """
    sparsity = max(0.0, min(1.0, float(sparsity)))

    n_block_rows = math.ceil(S_q / block_size_q)
    n_block_cols = math.ceil(S_kv / block_size_kv)
    if n_block_rows == 0 or n_block_cols == 0:
        return []

    last_col = n_block_cols - 1

    # Base target count from sparsity.
    K = int(round(n_block_cols * (1.0 - sparsity)))
    K = max(1, min(K, n_block_cols))

    g = torch.Generator(device="cpu")
    g.manual_seed(seed)

    # Forced columns for EVERY non-fully-dense row:
    # first 8 cols (or fewer if n_block_cols < 8) + last col
    forced_prefix = set(range(min(8, n_block_cols)))
    forced_always = set(forced_prefix)
    forced_always.add(last_col)

    # Ensure K is at least large enough to contain the forced set (unless fully dense anyway)
    K = max(K, len(forced_always))

    rows: list[list[int]] = []
    for r in range(n_block_rows):
        fully_dense = (r < 29) or (r == n_block_rows - 1)

        if fully_dense:
            cols = list(range(n_block_cols))
        else:
            forced = set(forced_always)

            need = K - len(forced)
            if need <= 0:
                cols = sorted(forced)
            else:
                candidates = [c for c in range(n_block_cols) if c not in forced]
                # sample without replacement
                perm = torch.randperm(len(candidates), generator=g).tolist()
                extra = [candidates[i] for i in perm[:need]]
                cols = sorted(list(forced) + extra)

        cols = cols + [pad_value] * (n_block_cols - len(cols))
        rows.append(cols)

    return rows

if USE_FRAME:
    generate_sparse_blocks_by_row = generate_sparse_blocks_by_row_with_frame

# Normal version
def make_block_mask(
    S_q: int,
    S_kv: int,
    block_size_q: int,
    block_size_kv: int,
    sparse_blocks_by_row: list[list[int]],
    device: str = "cpu",
) -> torch.Tensor:
    """
    Builds a dense boolean mask [1, 1, S_q, S_kv] from sparse block columns.

    True  = masked
    False = allowed
    """
    n_block_rows = math.ceil(S_q / block_size_q)
    n_block_cols = math.ceil(S_kv / block_size_kv)

    mask = torch.ones((S_q, S_kv), dtype=torch.bool, device=device)

    rows_to_process = min(n_block_rows, len(sparse_blocks_by_row))
    for r in range(rows_to_process):
        row_start = r * block_size_q
        row_end = min(row_start + block_size_q, S_q)

        for c in sparse_blocks_by_row[r]:
            if 0 <= c < n_block_cols:
                col_start = c * block_size_kv
                col_end = min(col_start + block_size_kv, S_kv)
                mask[row_start:row_end, col_start:col_end] = False

    return mask.unsqueeze(0).unsqueeze(0)


def generate_sparse_blocks_by_row_per_head(
    S_q: int,
    S_kv: int,
    block_size_q: int,
    block_size_kv: int,
    sparsity: float,
    num_heads: int,
    base_seed: int,
) -> list[list[list[int]]]:
    """
    Returns sparse blocks per head.

    Output shape:
      [num_heads][n_block_rows][K]
    """
    return [
        generate_sparse_blocks_by_row(
            S_q=S_q,
            S_kv=S_kv,
            block_size_q=block_size_q,
            block_size_kv=block_size_kv,
            sparsity=sparsity,
            seed=base_seed + h,
        )
        for h in range(num_heads)
    ]


def make_block_mask_per_head(
    S_q: int,
    S_kv: int,
    block_size_q: int,
    block_size_kv: int,
    sparse_blocks_by_row_per_head: list[list[list[int]]],
    device: str = "cpu",
) -> torch.Tensor:
    """
    Builds a dense boolean mask [1, H, S_q, S_kv] from sparse block columns.

    True  = masked
    False = allowed
    """
    head_masks = [
        make_block_mask(
            S_q=S_q,
            S_kv=S_kv,
            block_size_q=block_size_q,
            block_size_kv=block_size_kv,
            sparse_blocks_by_row=sparse_blocks_by_row,
            device=device,
        ).squeeze(0).squeeze(0)  # [S_q, S_kv]
        for sparse_blocks_by_row in sparse_blocks_by_row_per_head
    ]

    return torch.stack(head_masks, dim=0).unsqueeze(0)

# --------------------------------------------------------------------------- #
#  Other masks
# --------------------------------------------------------------------------- #

def make_lower_triangular_mask(
    S_q: int,
    S_kv: int,
    device: str = "npu:0",
) -> torch.Tensor:
    """
    Create an *exact* token-level lower-triangular (causal) mask.

    Mask positions where key index > query index (future attention).
    Works best when S_q == S_kv; for S_q != S_kv this still produces the
    correct "no-looking-forward" relation over indices.

    Returns:
        atten_mask: Bool tensor of shape [1, 1, S_q, S_kv] (BNSD),
                    where True means "masked" (disallowed).
    """
    q = torch.arange(S_q, device=device)[:, None]   # [S_q, 1]
    k = torch.arange(S_kv, device=device)[None, :]  # [1, S_kv]
    mask_2d = k > q                                # [S_q, S_kv]
    return mask_2d.unsqueeze(0).unsqueeze(0)

def make_dense_mask(
    S_q: int,
    S_kv: int,
    device: str = "npu:0",
):
    mask_2d = torch.zeros(S_q, S_kv, dtype=torch.bool, device=device)
    return mask_2d.unsqueeze(0).unsqueeze(0)


def make_band_mask(
    S_q: int,
    S_kv: int,
    pre_tokens: int,
    post_tokens: int = 0,
    device: str = "npu:0",
) -> torch.Tensor:
    """
    Band (sliding-window) mask:
      query i can attend only to keys in [i-pre_tokens, i+post_tokens].

    Returns:
        atten_mask: Bool tensor of shape [1, 1, S_q, S_kv] (BNSD),
                    where True means "masked" (disallowed).
    """
    pre_tokens = int(pre_tokens)
    post_tokens = int(post_tokens)
    if pre_tokens < 0:
        raise ValueError(f"pre_tokens must be >= 0, got {pre_tokens}")
    if post_tokens < 0:
        raise ValueError(f"post_tokens must be >= 0, got {post_tokens}")

    q = torch.arange(S_q, device=device)[:, None]    # [S_q, 1]
    k = torch.arange(S_kv, device=device)[None, :]   # [1, S_kv]

    left_bound = q - pre_tokens
    right_bound = q + post_tokens

    # Mask anything outside the allowed window:
    mask_2d = (k < left_bound) | (k > right_bound)
    return mask_2d.unsqueeze(0).unsqueeze(0)


def make_custom_mask(
    S_q: int,
    S_kv: int,
    device: str = "npu:0",
) -> torch.Tensor:
    # single block mask
    mask_2d = torch.zeros(S_q, S_kv, device=device)
    mask_2d[:BLOCK_SIZE_Q, :BLOCK_SIZE_KV] = 1
    return mask_2d.unsqueeze(0).unsqueeze(0).bool()

# --------------------------------------------------------------------------- #
#  printing utilities
# --------------------------------------------------------------------------- #

def block_allclose_map(
    out: torch.Tensor,
    ref: torch.Tensor,
    block_h: int,
    block_w: int,
    rtol: float = 0.05,
    atol: float = 0.05,
    print_map: bool = False,
) -> torch.Tensor:
    """
    Compare two 4D tensors [B, H, Y, X] in (block_h x block_w) blocks over the last two dims.
    Returns a boolean tensor of shape [B, H, nby, nbx] where each entry indicates whether the
    entire block is allclose. Optionally prints a 0/1 block matrix per (batch, head).

    - First two dims: batch and head (iterated and printed as headers)
    - Last two dims: "drawn" dimensions (split into blocks)

    Blocks at the edges can be smaller if Y or X is not divisible by block sizes.
    """
    if out.shape != ref.shape:
        raise ValueError(f"Shape mismatch: out {tuple(out.shape)} vs ref {tuple(ref.shape)}")
    if out.ndim != 4:
        raise ValueError(f"Expected 4D tensors [B, H, Y, X], got out.ndim={out.ndim}")

    if block_h <= 0 or block_w <= 0:
        raise ValueError("block_h and block_w must be positive integers")

    B, H, Y, X = out.shape
    nby = (Y + block_h - 1) // block_h
    nbx = (X + block_w - 1) // block_w

    # Elementwise closeness map: [B, H, Y, X] boolean
    close = torch.isclose(out, ref, rtol=rtol, atol=atol)

    # Blockwise result: [B, H, nby, nbx]
    block_ok = close.new_empty((B, H, nby, nbx), dtype=torch.bool)

    for b in range(B):
        for h in range(H):
            for by in range(nby):
                y0 = by * block_h
                y1 = min(y0 + block_h, Y)
                for bx in range(nbx):
                    x0 = bx * block_w
                    x1 = min(x0 + block_w, X)
                    block_ok[b, h, by, bx] = close[b, h, y0:y1, x0:x1].all()

    if print_map:
        for b in range(B):
            for h in range(H):
                print(f"batch={b}, head={h}")
                # Print as 0/1 grid
                grid = block_ok[b, h].to(dtype=torch.int32)
                for by in range(nby):
                    row = " ".join(str(int(v)) for v in grid[by].tolist())
                    print(row)
                print()  # blank line between heads

    return block_ok


# --------------------------------------------------------------------------- #
#  reference implementation
# --------------------------------------------------------------------------- #

def ref_prompt_flash_attention_fp32(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale_value: float,
    atten_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Scaled dot-product attention in float32 throughout.

    Assumes:
    - input_layout == "BNSD"
    - atten_mask (if provided) is bool broadcastable
      to [B, H, S_q, S_kv], where True means "masked out".
    """

    # Convert to fp32 for computation
    q_f = q.to(torch.float32)
    k_f = k.to(torch.float32)
    v_f = v.to(torch.float32)

    # [B, H, S_q, D] x [B, H, D, S_kv] -> [B, H, S_q, S_kv]
    attn_scores = torch.matmul(q_f, k_f.transpose(-1, -2))
    attn_scores = attn_scores * scale_value

    # Apply mask: True = disallow
    if atten_mask is not None:
        if atten_mask.dtype == torch.bool:
            attn_scores = attn_scores.masked_fill(
                atten_mask, torch.finfo(attn_scores.dtype).min
            )
        elif atten_mask.dtype in (torch.int8, torch.uint8):
            attn_scores = attn_scores.masked_fill(
                atten_mask == 1, torch.finfo(attn_scores.dtype).min
            )

    # Softmax in fp32
    attn_probs = torch.softmax(attn_scores, dim=-1)

    # [B, H, S_q, S_kv] x [B, H, S_kv, D] -> [B, H, S_q, D]
    out = torch.matmul(attn_probs, v_f)

    # Cast back to original dtype for comparison
    return out.to(dtype=q.dtype)


def ref_prompt_flash_attention_bf16_cpu(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale_value: float,
    atten_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Scaled dot-product attention on CPU.

    - Matmuls in bfloat16
    - Softmax in float32 for stability (then cast back to bf16)
    - input_layout == "BNSD" (interpreted as [B, H, S, D])
    - atten_mask (if provided) is bool broadcastable to [B, H, S_q, S_kv],
      where True means "masked out".
    """

    # Force CPU
    q_cpu = q.detach().to(device="cpu")
    k_cpu = k.detach().to(device="cpu")
    v_cpu = v.detach().to(device="cpu")
    mask_cpu = atten_mask.detach().to(device="cpu") if atten_mask is not None else None

    # Do matmuls in bf16 on CPU
    q_bf = q_cpu.to(torch.bfloat16)
    k_bf = k_cpu.to(torch.bfloat16)
    v_bf = v_cpu.to(torch.bfloat16)

    attn_scores = torch.matmul(q_bf, k_bf.transpose(-1, -2))
    attn_scores = attn_scores * torch.tensor(scale_value, dtype=torch.bfloat16)

    if mask_cpu is not None:
        # True = disallow
        attn_scores = attn_scores.masked_fill(mask_cpu, torch.finfo(torch.bfloat16).min)

    # Softmax in fp32, then back to bf16
    attn_probs = torch.softmax(attn_scores.to(torch.float32), dim=-1).to(torch.bfloat16)

    out = torch.matmul(attn_probs, v_bf)  # bf16

    # Match original behavior: cast back to original q dtype (on CPU)
    return out.to(dtype=q.dtype)


# --------------------------------------------------------------------------- #
#  input generator
# --------------------------------------------------------------------------- #
def gen_pfa_inputs(
    B: int, H: int, S_q: int, S_kv: int, D: int,
    device: str = "npu:0", dtype: torch.dtype = DTYPE
):
    """
    Generate random inputs for npu_prompt_flash_attention.
    """
    q = torch.randn(B, H, S_q, D, dtype=dtype, device=device)
    k = torch.randn(B, H, S_kv, D, dtype=dtype, device=device)
    v = torch.randn(B, H, S_kv, D, dtype=dtype, device=device)

    # For now we assume all sequences are full-length
    actseqlen = [S_q] * B
    actseqlenkv = [S_kv] * B

    return q, k, v, actseqlen, actseqlenkv

# --------------------------------------------------------------------------- #
#  attention mask creator
# --------------------------------------------------------------------------- #
def create_attention_mask(b, h, s_q, s_kv, d, sparsity, attention_matrix):
    """
    Create attention mask and related parameters for benchmarking.
    
    Returns:
        atten_mask, npu_atten_mask, sabi_blocks, sm, scale, pre_tok, post_tok
    """
    # Default parameters
    scale = 1.0 / math.sqrt(float(d))
    pre_tok = 2147483647     # default pre-token value
    post_tok = 0     # default post-token value
    sabi_blocks = None
    atten_mask = None

    # Mask is broadcastable over [B, H, S_q, S_kv].
    if attention_matrix == "sparse_block":
        # Build a block-wise attention mask for this (S_q, S_kv)
        per_head_block_indices = generate_sparse_blocks_by_row_per_head(
            s_q, s_kv, BLOCK_SIZE_Q, BLOCK_SIZE_KV, sparsity, num_heads=h, base_seed=BLOCK_MASK_SEED
        )
        atten_mask = make_block_mask_per_head(
            s_q, s_kv, BLOCK_SIZE_Q, BLOCK_SIZE_KV, per_head_block_indices, device=device
        )
        npu_atten_mask = atten_mask
        sm = 1
    elif attention_matrix == "sparse_block_all_same":
        # Determine which blocks to mask based on sparsity, then build mask.
        block_indices = generate_sparse_blocks_by_row(
            s_q,
            s_kv,
            BLOCK_SIZE_Q,
            BLOCK_SIZE_KV,
            sparsity,
            seed=BLOCK_MASK_SEED,
        )
        # Build a block-wise attention mask for this (S_q, S_kv)
        atten_mask = make_block_mask(
            s_q,
            s_kv,
            BLOCK_SIZE_Q,
            BLOCK_SIZE_KV,
            block_indices,
            device=device,
        )
        npu_atten_mask = atten_mask
        sm = 1
    elif attention_matrix == "lower_triangular":
        # Lower-triangular mask
        atten_mask = make_lower_triangular_mask(
            s_q,
            s_kv,
            device=device,
        )
        npu_atten_mask = make_lower_triangular_mask(
            2048,
            2048,
            device=device,
        )
        sm = 2
    elif attention_matrix == "band":
        atten_mask = make_band_mask(
            s_q,
            s_kv,
            pre_tokens=BAND_PRE_TOKENS,
            post_tokens=BAND_POST_TOKENS,
            device=device,
        )
        npu_atten_mask = make_lower_triangular_mask(
            2048,
            2048,
            device=device,
        )
        sm = 4
        pre_tok = BAND_PRE_TOKENS
        post_tok = BAND_POST_TOKENS
    elif attention_matrix == "blocks_optimized":
        per_head_block_indices = generate_sparse_blocks_by_row_per_head(
            s_q, s_kv, BLOCK_SIZE_Q, BLOCK_SIZE_KV, sparsity, num_heads=h, base_seed=BLOCK_MASK_SEED
        )
        if RUN_REFERENCE:
            atten_mask = make_block_mask_per_head(
                s_q, s_kv, BLOCK_SIZE_Q, BLOCK_SIZE_KV, per_head_block_indices, device=device
            )
            
        sabi_blocks = torch.tensor(per_head_block_indices, dtype=torch.uint16, device=device)

        npu_atten_mask = None
        sm = 0
    elif attention_matrix == "blocks_optimized_batched":
        per_batch_head_block_indices = []
        for bidx in range(b):
            per_head_block_indices = generate_sparse_blocks_by_row_per_head(
                s_q, s_kv,
                BLOCK_SIZE_Q, BLOCK_SIZE_KV,
                sparsity, num_heads=h,
                base_seed=BLOCK_MASK_SEED + bidx
            )
            per_batch_head_block_indices.append(per_head_block_indices)

        sabi_blocks = torch.tensor(per_batch_head_block_indices, dtype=torch.uint16, device=device)
        
        if RUN_REFERENCE:
            atten_masks = []
            for block_indices in per_batch_head_block_indices:
                atten_mask = make_block_mask_per_head(
                    s_q, s_kv, BLOCK_SIZE_Q, BLOCK_SIZE_KV, block_indices, device=device
                )
                atten_masks.append(atten_mask)
        
        npu_atten_mask = None
        sm = 0
    elif attention_matrix == "custom":
        atten_mask = make_custom_mask(
            s_q,
            s_kv,
            device=device,
        )
        npu_atten_mask = atten_mask
        sm = 1
    else:
        assert attention_matrix == "dense", "Attention matrix type not implemented, for dense use 'dense'"
        # No mask
        atten_mask = None
        npu_atten_mask = atten_mask
        sm = 0

    return atten_mask, npu_atten_mask, sabi_blocks, sm, scale, pre_tok, post_tok

# --------------------------------------------------------------------------- #
#  benchmark body
# --------------------------------------------------------------------------- #
def benchmark_prompt_flash_attention():
    run_our = True   # npu_prompt_flash_attention
    run_ref = RUN_REFERENCE   # PyTorch reference
    n_repeat = N_REPEATS
    n_warmup = N_WARMUP

    if not run_our and not run_ref:
        print("Nothing to run, must set run_our=True or run_ref=True")
        return

    print("=" * 140)
    print(f"  {DTYPE=}  {INPUT_LAYOUT=}")
    print("=" * 140)
    print(
        f"{'H':>3} {'B':>3} {'S_q':>6} {'S_kv':>6} {'D':>4} "
        f"{'sparsity':>9} "
        f"{'Outputs_equal':>15} "
        f"{'Ref_Latency_[usec]':>18} {'Our_Latency_[usec]':>18} "
        f"{'Ref_BW_[TB/sec]':>16} {'Our_BW_[TB/sec]':>16}"
    )
    print("-" * 140)

    for b, h, s_kv, d, sparsity in itertools.product(
        B_VALS, H_VALS, S_VALS, D_VALS, SPARSITY_VALS
    ):
        s_q = s_kv
        
        # Extract mask creation to separate function
        atten_mask, npu_atten_mask, sabi_blocks, sm, scale, pre_tok, post_tok = create_attention_mask(
            b, h, s_q, s_kv, d, sparsity, ATTENTION_MATRIX
        )

        if PRINT_MASK and atten_mask is not None:
            print(atten_mask.int())
            print(atten_mask.shape)

        run_ref_sparsity_0 = sparsity == 0

        ######## Check correctness ########
        are_equal_ref = "N/A"

        if run_our and run_ref or run_ref_sparsity_0:
            q, k, v, actseqlen, actseqlenkv = gen_pfa_inputs(
                b, h, s_q, s_kv, d, device=device, dtype=DTYPE
            )

            # Our operator (NPU)
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

            # Reference implementation (matmul+softmax+matmul) with same mask
            out_ref = ref_prompt_flash_attention_launcher(TORCH_REFERENCE, q, k, v, head_num=h, scale=scale, atten_mask=atten_mask, input_layout=INPUT_LAYOUT, run_ref_sparsity_0=run_ref_sparsity_0)               


            # Compare on CPU for convenience
            out_our_cpu = out_our.cpu()
            out_ref_cpu = out_ref.cpu()
            if PRINT_OUTPUTS:
                print("OURS: ", out_our_cpu.shape)
                print(out_our_cpu)
                print("REF: ", out_ref_cpu.shape)
                print(out_ref_cpu)

            # Our vs manual reference
            equal_ref = torch.allclose(out_our_cpu, out_ref_cpu, rtol=0.02, atol=0.02)
            if not equal_ref and PRINT_BLOCK_EQUALITY:
                block_allclose_map(out_our_cpu, out_ref_cpu,
                                   block_h=PRINT_HEIGHT, block_w=PRINT_WIDTH, rtol=0.02, atol=0.02,
                                   print_map=True)

            are_equal_ref = "yes" if equal_ref else "no"

        ########## Our Implementation ##########
        if run_our:
            input_sets = []
            for _ in range(n_warmup + n_repeat):
                q, k, v, actseqlen, actseqlenkv = gen_pfa_inputs(
                    b, h, s_q, s_kv, d, device=device, dtype=DTYPE
                )
                input_sets.append((q, k, v, actseqlen, actseqlenkv))

            # Warm-up
            for i in range(n_warmup):
                q, k, v, actseqlen, actseqlenkv = input_sets[i]
                npu_prompt_flash_attention(
                    q,
                    k,
                    v,
                    sabi_blocks=sabi_blocks,
                    actual_seq_lengths=actseqlen,
                    actual_seq_lengths_kv=actseqlenkv,
                    num_heads=h,
                    input_layout=INPUT_LAYOUT,
                    scale_value=scale,
                    atten_mask=npu_atten_mask,
                    sparse_mode=sm,
                    pre_tokens=pre_tok,
                    next_tokens=post_tok,
                )
            torch.npu.synchronize()

            # Measurements
            start = torch.npu.Event(enable_timing=True)
            end = torch.npu.Event(enable_timing=True)

            start.record()
            for i in range(n_warmup, n_warmup + n_repeat):
                q, k, v, actseqlen, actseqlenkv = input_sets[i]
                npu_prompt_flash_attention(
                    q,
                    k,
                    v,
                    sabi_blocks=sabi_blocks,
                    actual_seq_lengths=actseqlen,
                    actual_seq_lengths_kv=actseqlenkv,
                    num_heads=h,
                    input_layout=INPUT_LAYOUT,
                    scale_value=scale,
                    atten_mask=npu_atten_mask,
                    sparse_mode=sm,
                    pre_tokens=pre_tok,
                    next_tokens=post_tok,
                )
            end.record()
            torch.npu.synchronize()

            our_duration = start.elapsed_time(end) / n_repeat * 1000.0  # ms -> μs
            total_bytes = bytes_moved_pfa(b, h, s_q, s_kv, d)
            our_bw = total_bytes / our_duration / 1e6  # TB/s
        else:
            our_duration = None
            our_bw = None

        ########## Reference Implementation ##########
        if run_ref or run_ref_sparsity_0:
            input_sets = []
            for _ in range(n_warmup + n_repeat):
                q, k, v, actseqlen, actseqlenkv = gen_pfa_inputs(
                    b, h, s_q, s_kv, d, device=device, dtype=DTYPE
                )
                input_sets.append((q, k, v, actseqlen, actseqlenkv))

            # Warm-up
            for i in range(n_warmup):
                q, k, v, _, _ = input_sets[i]
                ref_prompt_flash_attention_launcher(TORCH_REFERENCE, q, k, v, head_num=h, scale=scale, atten_mask=atten_mask, input_layout=INPUT_LAYOUT, run_ref_sparsity_0=run_ref_sparsity_0)               
            
            torch.npu.synchronize()

            # Measurements
            start = torch.npu.Event(enable_timing=True)
            end = torch.npu.Event(enable_timing=True)

            start.record()
            for i in range(n_warmup, n_warmup + n_repeat):
                q, k, v, _, _ = input_sets[i]
                ref_prompt_flash_attention_launcher(TORCH_REFERENCE, q, k, v, head_num=h, scale=scale, atten_mask=atten_mask, input_layout=INPUT_LAYOUT, run_ref_sparsity_0=run_ref_sparsity_0)               
            end.record()
            torch.npu.synchronize()

            ref_duration = start.elapsed_time(end) / n_repeat * 1000.0  # ms -> μs
            total_bytes = bytes_moved_pfa(b, h, s_q, s_kv, d)
            ref_bw = total_bytes / ref_duration / 1e6  # TB/s
        else:
            ref_duration = None
            ref_bw = None

        ######## Print results ########
        print(
            f"{h:>3} {b:>3} {s_q:>6} {s_kv:>6} {d:>4} "
            f"{sparsity:>9.2f} "
            f"{are_equal_ref:>15} ",
            end="",
        )

        if (run_ref or run_ref_sparsity_0) and ref_duration is not None:
            print(f"{ref_duration:>18.2f} ", end="")
        else:
            print(f"{'N/A':>18} ", end="")

        if run_our and our_duration is not None:
            print(f"{our_duration:>18.2f} ", end="")
        else:
            print(f"{'N/A':>18} ", end="")

        if (run_ref or run_ref_sparsity_0) and ref_bw is not None:
            print(f"{ref_bw:>16.3f} ", end="")
        else:
            print(f"{'N/A':>16} ", end="")

        if run_our and our_bw is not None:
            print(f"{our_bw:>16.3f}")
        else:
            print(f"{'N/A':>16}")

    print("=" * 140)


if __name__ == "__main__":
    torch.npu.set_device(device)
    benchmark_prompt_flash_attention()
