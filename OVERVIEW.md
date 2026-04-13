## Mental Model for ops-transformer

**One sentence:** This repo is a library of ~115 highly-optimized GPU-like operators for Huawei Ascend NPUs, covering the core building blocks of Transformer LLMs.

### The Big Picture

Think of it as **cuDNN/CUTLASS but for Ascend hardware**. Where NVIDIA has CUDA kernels for attention, matmul, etc., this repo provides equivalent optimized kernels in **AscendC** (Huawei's kernel language) running on CANN (their compute stack).

### The 6 Operator Families

Each family maps to a part of a Transformer forward pass:

| Family | What it does | # Ops |
|--------|-------------|-------|
| **attention/** | Flash attention, MLA, sparse attention, paged KV cache | ~48 |
| **moe/** | Mixture-of-Experts routing, permutation, token dispatch | ~19 |
| **mc2/** | Multi-GPU collective ops fused with compute (AllReduce+MatMul, AllToAll) | ~35 |
| **gmm/** | Grouped MatMul (used by MoE experts, with quantization) | ~7 |
| **ffn/** | Feed-forward network fused ops | ~3 |
| **posembedding/** | RoPE variants, sin/cos cache, KV cache manipulation | ~11 |

### How a Single Operator is Structured

Every operator (e.g. [flash_attention_score/](attention/flash_attention_score/)) follows the same 4-layer pattern:

```
op_name/
├── op_host/      # Tiling logic (how to partition work across NPU cores) — C++
├── op_kernel/    # The actual AscendC kernel (runs on device) — C++
├── op_api/       # aclnn API entry point (called from PyTorch/framework) — C++
└── op_graph/     # Graph-level operator registration (for GE compiler) — C++
```

This is the key structural insight: **tiling → kernel → API → graph** is the universal pattern.

### The Shared Layer

[common/](common/) provides reusable tiling utilities, kernel helpers, and framework interfaces that all operators depend on.

### Data Flow Through a Transformer

```
Input → posembedding (RoPE) → attention (FlashAttn/MLA)
      → moe_init_routing → mc2 (distribute across GPUs)
      → gmm (expert compute) → mc2 (combine results)
      → ffn → output
```

### Build System

- [build.sh](build.sh) wraps CMake; target SoC via `--soc=ascend910b`
- Supports JIT (`--jit`) or package (`--pkg`) compilation
- Compiler: BiSheng (Huawei's clang fork for NPU)

### Current Branch Context

Your `before-HasAtten` branch and recent commits focus on **FIAS** (Fused Infer Attention Score) — specifically template metaprogramming refactors (type erasure, `if constexpr` dispatch, deduplicating type aliases) to reduce code bloat and compilation overhead in the attention kernels.