## Mental Model for `fused_infer_attention_score` (FIAS)

### What it is

A single mega-operator that fuses the entire attention computation for **inference** on Ascend NPUs:

```
Output = Softmax(Q @ K^T / √d + mask) @ V
```

Instead of launching separate kernels for each step, FIAS fuses QK-matmul → softmax → PV-matmul into one kernel launch, keeping intermediate data in on-chip SRAM (like Flash Attention on CUDA).

---

### The 4-Layer Architecture

```
┌─────────────────────────────────────────────────────┐
│  op_api/    — C API surface (aclnn interface)       │
│  "What the user calls"                              │
│  GetWorkspaceSize() → Execute()                     │
│  29+ params: Q, K[], V[], mask, quant, RoPE, etc.   │
├─────────────────────────────────────────────────────┤
│  op_host/   — CPU-side tiling & validation          │
│  "How work is divided across NPU cores"             │
│  FiaInfoParser → FAInferTiling → tiling key         │
├─────────────────────────────────────────────────────┤
│  op_kernel/ — NPU-side kernel code (AscendC)        │
│  "The actual computation on device"                 │
│  FAInferKernel<QK, PV, Softmax, Rescale, ...>       │
├─────────────────────────────────────────────────────┤
│  op_graph/  — Graph-level registration/fallback     │
│  "How CANN compiler sees this operator"             │
└─────────────────────────────────────────────────────┘
```

---

### The Kernel: 3-Phase Pipeline

The core computation in [flash_attention_regular.h](attention/fused_infer_attention_score/op_kernel/flash_attention_regular.h) runs in a tiled loop over KV sequence chunks:

```
for each KV block (s2 tiles):
  ┌──────────────────┐
  │ 1. BlockMmadQK   │  S = Q @ K^T    (score matrix)
  │    (block_mmad_qk)│
  ├──────────────────┤
  │ 2. OnlineSoftmax  │  P = softmax(S/√d + mask)  (streaming/online)
  │    + Masking       │
  ├──────────────────┤
  │ 3. BlockMmadPV   │  O += P @ V     (accumulate output)
  │    (block_mmad_pv)│
  └──────────────────┘
```

"Online softmax" means it computes softmax incrementally across KV blocks without materializing the full S matrix — the same trick as FlashAttention.

---

### The Tiling Key: How Template Explosion is Managed

This is the **most important architectural decision** to understand. The operator supports a combinatorial explosion of modes:

- **Dtypes**: Q (fp16/bf16/int8) × K/V (fp16/bf16/int8/int4/fp8/fp4) × O (fp16/bf16/int8)
- **Mask types**: no mask, causal, sliding window, full mask, sparse
- **Layouts**: BSND vs TND
- **Features**: paged KV cache, flash decode, quantization mode, MLA, prefix sharing

Instead of compiling all combinations as C++ templates, the system uses **tiling keys** — numeric codes like `5000000000000200100` that encode the specific configuration. The kernel file ([fused_infer_attention_score.cpp](attention/fused_infer_attention_score/op_kernel/fused_infer_attention_score.cpp)) has ~80 `TILING_KEY_IS()` branches, each instantiating one pre-compiled template specialization.

```
Runtime flow:
  Host selects tiling key → Kernel dispatches to matching TILING_KEY_IS() → 
  Pre-compiled FAInferKernel<specific types> runs
```

---

### The Two Modes: Prefill vs Decode

| | **Prefill** (prompt processing) | **Decode** (token generation) |
|---|---|---|
| Q tokens | Many (full sequence) | One (single new token) |
| Kernel | [flash_attention_regular.h](attention/fused_infer_attention_score/op_kernel/flash_attention_regular.h) | [flash_attention_regular_decode.h](attention/fused_infer_attention_score/op_kernel/flash_attention_regular_decode.h) |
| Parallelism | Split across batch × heads | **Flash Decode**: split KV across cores, then reduce |
| Tiling flag | `SPLITFUSE_TILING` | `SPLITFUSE_TILING_FD` |

---

### Key Feature Dimensions

Think of FIAS as parameterized along these axes:

| Axis | Options | Where configured |
|------|---------|-----------------|
| **Dtype** | fp16, bf16, int8, int4, fp8, fp4 | Tiling key + template params |
| **Mask** | None, Causal, SWA, Full, Sparse | `MaskType` enum |
| **Layout** | BSND, TND | `inputLayout` enum |
| **KV Cache** | Contiguous vs Paged (block table) | `PAGED_CACHE_FLAG` bool |
| **Decode mode** | Regular vs Flash Decode | `IS_FD` bool |
| **Quantization** | Per-channel/token/tensor/block | `quantMode` params |
| **Architecture** | Arch32 (older) vs Arch35 (newer) | Separate tiling/kernel paths |

---

### The `attn_infra/` Library

Under [op_kernel/attn_infra/](attention/fused_infer_attention_score/op_kernel/attn_infra/) is a CUTLASS-inspired abstraction hierarchy:

```
attn_infra/
├── arch/         # Hardware: core types, L0/L1 buffers, cross-core sync
├── gemm/         # GEMM blocks: dispatch policy → block_mmad → tile_mmad
├── epilogue/     # Post-GEMM: online softmax, rescale, init output
├── layout/       # Memory layouts: matrix (1207 lines!), vector
└── coord/        # Coordinate types for tiling
```

This mirrors NVIDIA CUTLASS's architecture: **Tile → Block → Epilogue** with configurable policies.

---

### Your Current Work Context

Your `before-HasAtten` branch has been refactoring FIAS's template machinery:
- Extracting `FATypeTraits` to deduplicate type aliases
- Using `if constexpr` for `quantMode` dispatch
- Planning boolean type-erasure (likely to reduce the number of template parameters like `HasAtten`, `PAGED_CACHE_FLAG`, etc. from compile-time bools to runtime dispatch)

This is about **reducing compilation time and binary size** — the ~80 tiling key branches × template instantiations create significant build overhead.