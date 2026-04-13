Here's a complete inventory of every file in `attention/fused_infer_attention_score/`, organized by directory:

---

## Root

| File | Purpose |
|------|---------|
| `CMakeLists.txt` | Top-level build config; pulls in op_api, op_host, op_kernel, op_graph, tests |
| `README.md` | Product documentation (Chinese) — operator params, constraints, supported platforms (Atlas A3/A2) |
| `OVERVIEW.md` | Architectural mental model — 4-layer design, kernel pipeline, tiling key mechanism |

## `docs/` — API Documentation

| File | Purpose |
|------|---------|
| `aclnnFusedInferAttentionScore.md` | V1 API docs (deprecated, removal Dec 2026) |
| `aclnnFusedInferAttentionScoreV2.md` | V2 API docs |
| `aclnnFusedInferAttentionScoreV3.md` | V3 API docs |
| `aclnnFusedInferAttentionScoreV4.md` | V4 API docs |
| `aclnnFusedInferAttentionScoreV5.md` | V5 API docs (current recommended) |
| `TEMPLATE_TYPE_ERASURE_PLAN.md` | Design doc for template type-erasure refactoring to reduce compilation overhead |

## `op_api/` — C API Surface Layer

| File | Purpose |
|------|---------|
| `aclnn_fused_infer_attention_score.h/.cpp` | V1 API (deprecated wrapper) |
| `aclnn_fused_infer_attention_score_v2.h/.cpp` | V2 API |
| `aclnn_fused_infer_attention_score_v3.h/.cpp` | V3 API |
| `aclnn_fused_infer_attention_score_v4.h/.cpp` | V4 API |
| `aclnn_fused_infer_attention_score_v5.h/.cpp` | V5 API (current) |
| `aclnn_fused_infer_attention_score_inner.cpp` | Shared inner implementation used by all versions |
| `fused_infer_attention_score_inner.h` | Header for inner interface |

## `op_host/` — CPU-Side Tiling & Validation

| File | Purpose |
|------|---------|
| `CMakeLists.txt` | Build config for host-side code |
| `fused_infer_attention_score_def.cpp` | Operator definition — registers dtypes and param constraints |
| `fused_infer_attention_score_infershape.cpp` | Output shape inference |
| `fused_infer_attention_score_tiling.cpp` | Main tiling algorithm — partitions work across NPU cores |
| `fused_infer_attention_score_tiling.h` | Tiling data structures (base params, mask params, page attention params) |
| `fused_infer_attention_score_tiling_compile_info.h` | Arch-specific compilation constants |
| `fused_infer_attention_score_tiling_constants.h` | Tiling constants |
| `fused_infer_attention_score_tiling_index.h` | Index definitions for tiling data fields |
| `fused_infer_attention_score_tiling_info_parser.cpp/.h` | Parses and validates tiling info from input tensors |
| `fused_infer_attention_score_tiling_register.cpp` | Registers tiling implementations with CANN framework |
| `fused_infer_attention_score_tiling_utils.h` | Tiling utility functions |
| `flash_attention_infer_tiling.h` | Flash attention inference-specific tiling structures |

### `op_host/arch32/` — Ascend 3.2 (Legacy)

| File | Purpose |
|------|---------|
| `fused_infer_attention_score_tiling_check.cpp/.h` | Parameter validation entry point |
| `fused_infer_attention_score_tiling_check_consistency.cpp` | Cross-parameter consistency checks |
| `fused_infer_attention_score_tiling_check_existence.cpp` | Required parameter presence checks |
| `fused_infer_attention_score_tiling_check_feature.cpp` | Feature-specific validation |
| `fused_infer_attention_score_tiling_check_single_para.cpp` | Single-parameter bounds checks |
| `fused_infer_attention_score_tiling_v3.cpp/.h` | Arch32 V3 tiling algorithm |

### `op_host/arch35/` — Ascend 3.5 (Current)

| File | Purpose |
|------|---------|
| `fused_infer_attention_score_tiling_impl.cpp/.h` | Core arch35 tiling implementation |
| `fused_infer_attention_score_tiling_v4.cpp/.h` | Arch35 V4 tiling algorithm |

### `op_host/checkers/` — Parameter Validators

| File | Purpose |
|------|---------|
| `base_checker.cpp/.h` | Abstract base class for all checkers |
| `fia_checker.cpp/.h` | Main aggregator combining all checkers below |
| `actual_seq_len_checker.cpp/.h` | Validates actual sequence length params |
| `dequant_checker.cpp/.h` | Validates dequantization scale/offset |
| `learnable_sink_checker.cpp/.h` | Validates learnable sink token params |
| `left_padding_checker.cpp/.h` | Validates left padding config |
| `mask_checker.cpp/.h` | Validates attention mask tensor properties |
| `paged_attention_checker.cpp/.h` | Validates paged KV cache block table params |
| `post_quant_checker.cpp/.h` | Validates post-quantization params |
| `pse_checker.cpp/.h` | Validates position-sensitive embeddings |
| `rope_checker.cpp/.h` | Validates RoPE (Rotary Position Embedding) params |
| `shape_checker.cpp/.h` | Validates tensor shape consistency |
| `softmax_lse_checker.cpp/.h` | Validates log-sum-exp for online softmax |
| `system_prefix_checker.cpp/.h` | Validates system prompt prefix params |

## `op_kernel/` — NPU-Side Kernel Code

| File | Purpose |
|------|---------|
| `fused_infer_attention_score.cpp` | Main kernel dispatcher — 80+ `TILING_KEY_IS` branches for configs |
| `fused_infer_attention_score_apt.cpp` | APT (Ascend Performance Tuning) kernel variant |
| `fused_infer_attention_score_template_tiling_key.h` | `ASCENDC_TPL_ARGS_SEL` macros for template type specialization |
| `fused_infer_attention_score_tilingkey.h` | Tiling key definitions encoding dtype/mask/layout/features |
| `fused_infer_attention_score_v3.cpp` | V3 kernel (arch32 conditional compilation) |
| `flash_attention_interface.cpp` | Interface layer between kernel and infra |
| `flash_attention_regular.h` | Prefill mode kernel template (full prompt sequences) |
| `flash_attention_regular_decode.h` | Decode mode kernel template (single token generation) |
| `kernel_common.hpp` | Common kernel utilities and types |

### `op_kernel/attn_infra/` — CUTLASS-Like Attention Infrastructure

**Core types:**
| File | Purpose |
|------|---------|
| `base_defs.hpp` | Enums for mask types, data layouts |
| `coord.hpp` | Tiling index coordinate types |
| `matrix_coord.hpp` | 2D matrix coordinates |
| `gemm_coord.hpp` | GEMM block coordinates |

**`arch/`** — Hardware abstraction:
| File | Purpose |
|------|---------|
| `arch.hpp` | Core types and hardware params per architecture |
| `cross_core_sync.hpp` | Inter-core synchronization primitives |
| `local_tensor_buffer.hpp` | On-chip SRAM buffer management |
| `resource.hpp` | Register/L0/L1 resource allocation |

**`detail/`** — Utilities:
| File | Purpose |
|------|---------|
| `alignment.hpp` | Data alignment for vectorization |
| `dependent_false.hpp` | `static_assert` helper for templates |
| `macros.hpp` | Code generation macros |

**`layout/`** — Memory layout abstractions:
| File | Purpose |
|------|---------|
| `layout.hpp` | Base layout interface |
| `matrix.hpp` | 2D matrix memory layouts (~1200 lines of transformations) |
| `vector.hpp` | 1D vector memory layouts |

**`gemm/`** — Matrix multiplication:
| File | Purpose |
|------|---------|
| `gemm_type.hpp` | GEMM config type definitions |
| `helper.hpp` | GEMM utility functions |
| `dispatch_policy.hpp` | Kernel variant selection policy |
| `block/block_mmad.hpp` | Generic block multiply-accumulate |
| `block/block_mmad_qk.hpp` | Q@K^T score computation |
| `block/block_mmad_qk_decode.hpp` | Decode-optimized Q@K^T |
| `block/block_mmad_pv.hpp` | P@V value computation |
| `block/block_mmad_pv_decode.hpp` | Decode-optimized P@V |
| `tile_common/copy_gm_to_l1.hpp` | Global memory → L1 transfer |
| `tile_common/copy_gm_to_ub.hpp` | Global memory → UB transfer |
| `tile_common/copy_l0c_to_gm.hpp` | L0c → Global memory writeback |
| `tile_common/copy_l1_to_bt.hpp` | L1 → Broadcast Tensor (systolic input) |
| `tile_common/copy_l1_to_l0a.hpp` | L1 → L0a (MMAD input buffer) |
| `tile_common/copy_l1_to_l0b.hpp` | L1 → L0b (MMAD weight buffer) |
| `tile_common/copy_ub_to_gm.hpp` | UB → Global memory writeback |
| `tile_common/tile_copy.hpp` | Generic tile copy utilities |
| `tile_common/tile_copy_tla.hpp` | Tensor Load Accelerator variant |
| `tile_common/tile_mmad.hpp` | Tile-level multiply-accumulate |

**`epilogue/`** — Post-GEMM (softmax, rescaling, output):
| File | Purpose |
|------|---------|
| `dispatch_policy.hpp` | Epilogue variant selection |
| `block/block_epilogue.hpp` | Base epilogue template |
| `block/block_epilogue_init_outputs.hpp` | Initialize output tensors |
| `block/block_epilogue_online_softmax.hpp` | Online softmax (full precision) |
| `block/block_epilogue_online_softmax_low_prec.hpp` | Online softmax (low precision) |
| `block/block_epilogue_rescale_o.hpp` | Output rescaling with normalization |
| `block/block_epilogue_rescale_o_low_prec.hpp` | Low-precision rescale |
| `block/CombineScale.hpp` | Combine multiple scaling factors |
| `tile_common/copy_gm_to_ub.hpp` | GM → UB for output reads |
| `tile_common/copy_ub_to_gm.hpp` | UB → GM for output writes |
| `tile_common/tile_broadcast_inplace_by_column.hpp` | Column-wise broadcast |
| `tile_common/tile_broadcast_inplace_by_row.hpp` | Row-wise broadcast |
| `tile_common/tile_broadcast_mul.hpp` | Broadcast multiply |
| `tile_common/tile_broadcast_one_blk.hpp` | Single-block broadcast |
| `tile_common/tile_cast.hpp` | Dtype casting |
| `tile_common/tile_copy.hpp` | Tile data copy |
| `tile_common/tile_elemwise_add.hpp` | Element-wise add |
| `tile_common/tile_elemwise_mul.hpp` | Element-wise multiply |
| `tile_common/tile_elemwise_muls.hpp` | Multi element-wise multiply |
| `tile_common/tile_swizzle.hpp` | Memory layout swizzling |

## `op_graph/` — Graph-Level Integration

| File | Purpose |
|------|---------|
| `CMakeLists.txt` | Build config |
| `fallback_fused_infer_attention_score.cpp` | Fallback decomposition into primitives when optimized path unavailable |

## `examples/` — Example Usage

| File | Purpose |
|------|---------|
| `test_aclnn_fused_infer_attention_score.cpp` | Standalone V4 API usage example |
| `arch35/test_aclnn_fused_infer_attention_score_v5.cpp` | V5 API example on arch35 |

## `scripts/` — Dev/Analysis Tools

| File | Purpose |
|------|---------|
| `count_template_instantiations.py` | Analyzes `ASCENDC_TPL_ARGS_SEL` blocks to quantify template explosion |
| `find_mergeable_tpl_blocks.py` | Finds mergeable template specializations for consolidation |

## `tests/pytest/` — Python Test Framework

| File | Purpose |
|------|---------|
| `README.md` | Framework architecture and usage docs |
| `test.py` | Main runner — CPU golden generation + NPU result comparison |
| `testcases.py` | Test case parameter configs with randomized shapes/dtypes |
| `check_valid_param.py` | Param validation and precision comparison |
| `gqa_no_quant_bnsd_bsnd.py` | CPU reference implementation (non-quantized BNSD/BSND layouts) |
| `gqa_no_quant_bnsd_bsnd_ge.py` | Graph engine CPU reference |
| `pytest.ini` | Pytest config with CI/graph mode markers |

## `tests/st/` — System Tests (V2/V3/V4)

Each version subfolder contains:
- `all_aclnnFusedInferAttentionScoreVN.json` — test configuration matrix
- `executor_aclnnFusedInferAttentionScoreVN.py` — test executor

## `tests/ut/` — C++ Unit Tests

| File | Purpose |
|------|---------|
| `CMakeLists.txt` | Unit test build config |
| `op_host/CMakeLists.txt` | Host-side unit test build |
| `op_host/arch35/CMakeLists.txt` | Arch35-specific test build |
| `op_host/arch35/test_fused_infer_attention_score_tiling.cpp` | Tiling correctness tests for arch35 |

---

**Summary**: This is a fused flash-attention inference operator for Ascend NPU with a 4-layer architecture (API → Host tiling → NPU kernel → Graph fallback), 5 API versions, 80+ kernel dispatch branches, dual prefill/decode modes, and a CUTLASS-inspired infrastructure library for GEMM and softmax on custom hardware.