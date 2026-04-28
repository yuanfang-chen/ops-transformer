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

---

## Build Infrastructure — How an operator (FIAS) is Built

### Top-level entry point — `build.sh`

A bash wrapper around CMake that handles arg parsing and exposes high-level modes. Flow:

1. Parses flags, sets `CUSTOM_OPTION` string of `-D…` CMake defines (e.g. `-DASCEND_COMPUTE_UNIT=ascend910b`, `-DASCEND_OP_NAME=fused_infer_attention_score`, `-DENABLE_OPS_KERNEL=ON`, `-DENABLE_BUILD_PKG=ON`).
2. `cmake_config` runs `cmake .. ${CUSTOM_OPTION}` in `build/` (or `build_out` path). With `--ninja` it appends `-G Ninja` at `build.sh:1569`.
3. `build <target>` then invokes `cmake --build . --target <target>` (`build.sh:473-488`). Key wrapper targets:
   - `package` → full run package (`build_package`)
   - `ops_transformer_kernel` → device-side kernel bins only (`build_kernel`, invoked by `--opkernel`)
   - `ophost_transformer` / `opapi_transformer` / `opgraph_transformer` / `onnx_plugin_transformer` → individual shared libs
4. On `--pkg` it iterates each SoC in `SOC_ARRAY` and re-configures + builds per SoC, then packages with CPack + makeself to produce `build_out/cann-ops-transformer-*.run`.

### CMake layering (`CMakeLists.txt` → `cmake/*.cmake`)

Top `CMakeLists.txt`:
- Maps `ASCEND_COMPUTE_UNIT` to `ARCH_DIRECTORY` (ascend910b → arch32, ascend950 → arch35, …).
- Includes `cmake/config.cmake` (paths, CANN package discovery at `ASCEND_CANN_PACKAGE_PATH`, `ASCEND_PROJECT_DIR` = `…/tikcpp/ascendc_kernel_cmake`, which is the upstream AscendC cmake machinery), `cmake/custom_build.cmake` (main orchestrator), `cmake/obj_func.cmake` (object-lib macros), `cmake/func.cmake` (kernel-bin macros), `cmake/opbuild.cmake`, `cmake/package.cmake`.
- Walks operator tree via `op_add_subdirectory` → `add_subdirectory(<op>/op_host)` for every op.

`custom_build.cmake` declares the four umbrella shared libs:
- `cust_opapi` (→ `liboptransformer_opapi.so`), `cust_proto` (op_proto), `cust_opmaster` (tiling, installed as `liboptiling.so` via symlink), plus the three build-time-only helper libs `op_host_aclnn{,Inner,Exc}` used solely so `op_build` can introspect op defs.
- All four are populated by object libs `${OPHOST_NAME}_opapi_obj`, `_infer_obj`, `_tiling_obj` created per-op via `add_opapi_modules`/`add_infer_modules`/`add_tiling_modules` (`obj_func.cmake`).

### Per-op layer (the FIAS case)

`attention/fused_infer_attention_score/CMakeLists.txt` just `add_subdirectory`s every subdir that has its own `CMakeLists.txt`. The real work is in `attention/fused_infer_attention_score/op_host/CMakeLists.txt`:

1. `add_op_to_compiled_list()` — pushes `fused_infer_attention_score` into global `COMPILED_OPS`/`COMPILED_OP_DIRS` cache vars.
2. Declares deps: `fused_infer_attention_score_depends = attention/incre_flash_attention;attention/prompt_flash_attention;attention/common`. These are pulled in by `op_add_depend_directory` (top `CMakeLists.txt:330`) so IFA/PFA get compiled alongside.
3. Registers `fused_infer_attention_score_def.cpp` (op def) on `op_host_aclnnInner` — `op_build` later reads that lib to emit aclnn entry stubs.
4. `add_ops_compile_options(OP_NAME FusedInferAttentionScore OPTIONS --cce-auto-sync=off -Werror … --op_relocatable_kernel_binary=true)` (and a different set for `ascend950`). These flags are appended to `ASCEND_CUSTOM_OPTIONS` and passed to the bisheng/ccec compiler when kernel bins are built.
5. `add_modules_sources_with_soc(OP_API_INDEPENDENT ON OP_API_DIR .../op_api OPTYPE fused_infer_attention_score ACLNNTYPE aclnn_inner)` — collects `op_api/*.cpp` into `…_opapi_obj`, `*_infershape*.cpp` into `…_infer_obj`, `*_tiling*.cpp` + `op_graph/fallback_*.cpp` into `…_tiling_obj`.
6. Explicitly adds extra tiling sources that aren't matched by the default globs (arch32/arch35 tiling impls, shared `common/op_host/fia_tiling_*.cpp`, checkers, fallback). These all go into `${OPHOST_NAME}_tiling_obj`.

Result: FIAS's host code lands in `cust_opmaster` (tiling), `cust_proto` (infer), and `cust_opapi` (aclnn C API).

### Device-side kernel build (`--opkernel` / `ops_transformer_kernel`)

Different path — it doesn't go through plain g++:

1. `op_build` (a CANN tool at `${CANN}/tools/opbuild/op_build`) is dlopened against `op_host_aclnn*` to generate aclnn sources, proto, and per-SoC `aic-<soc>-ops-info.ini` files in `build/autogen/` (`custom_build.cmake:800-852`).
2. `add_compile_cmd_target` (`func.cmake:196`) runs `ascendc_bin_param_build.py` on the ops-info ini → writes `build/binary/<soc>/gen/*.sh`, one shell script per (op, tiling-bin-index) pair.
3. `add_bin_compile_target` (`func.cmake:482`) for each script: copies op_kernel sources into `build/binary/<soc>/src/<op>/`, copies the generated `<op>.py` driver, then creates a custom target that runs `bash <bin_script> <py> <bin_out_dir>`. That script ultimately invokes the bisheng `ccec` compiler with the accumulated `--cce-*` flags to emit one `.o`/`.json` per tiling key. Per op all indices hang off `<op>_<soc>` which hangs off `ops_transformer_kernel`.
4. `ascendc_ops_config.py` aggregates all per-key JSON into `binary_info_config.json` (plus `relocatable_kernel_info_config.json`).

For FIAS this is exactly the "template explosion": the ops-info ini enumerates one compile job per `ASCENDC_TPL_ARGS_SEL` combination, so `ops_transformer_kernel` ends up spawning dozens of parallel `ccec` invocations — this is why the two scripts under `attention/fused_infer_attention_score/scripts/` exist and why collapsing instantiations matters for build time.

### Packaging

`cmake/makeself_custom.cmake` + CPack wrap the staged install tree (`packages/vendors/${VENDOR_NAME}_transformer/…`) into `build_out/cann-ops-transformer-<vendor>_linux-<arch>.run`. Install layout for FIAS lands under:
- `op_api/lib/libcust_opapi.so`, `op_api/include/aclnnop/aclnn_fused_infer_attention_score*.h`
- `op_proto/lib/linux/<arch>/libcust_opsproto_rt2.0.so`
- `op_impl/ai_core/tbe/op_tiling/liboptiling.so` (symlink to `cust_opmaster`)
- `op_impl/ai_core/tbe/kernel/<soc>/ops_transformer/fused_infer_attention_score/*.o` + `config/<soc>/*/binary_info_config.json`

### FIAS-specific end-to-end command

```
bash build.sh --pkg --soc=ascend910b --ops=fused_infer_attention_score --ninja -j16 -O3
```
= configure with `-DASCEND_OP_NAME=fused_infer_attention_score -DASCEND_COMPUTE_UNIT=ascend910b -DENABLE_OPS_HOST=ON -DENABLE_OPS_KERNEL=ON -DENABLE_BUILD_PKG=ON -G Ninja`, then `ninja package`, which pulls in `cust_opapi`/`cust_proto`/`cust_opmaster` via their `_obj` libs (FIAS + its three deps: IFA, PFA, attention/common) plus `ops_transformer_kernel` (one ccec invocation per tiling key) and runs CPack+makeself.