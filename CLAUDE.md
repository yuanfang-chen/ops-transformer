# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

`ops-transformer` is a library of ~115 optimized operators for Huawei Ascend NPUs, written in **AscendC** (Huawei's CUDA-equivalent kernel language) and C++ host code, targeting the CANN software stack. Think "cuDNN/CUTLASS for Ascend." It covers the core compute building blocks of Transformer LLMs: attention, MoE, grouped matmul, FFN, positional embeddings, and fused multi-NPU collectives.

## Build system

All builds go through `bash build.sh` (a CMake wrapper). Key entry points:

```bash
# Build a full custom-op run package for a specific SoC and operators
bash build.sh --pkg --soc=ascend910b --ops=fused_infer_attention_score -j16 -O3

# JIT (no kernel bin) / full pkg variants
bash build.sh --jit --soc=ascend910b
bash build.sh --pkg --experimental --soc=ascend910b --ops=abs

# Build a single library layer (for fast iteration)
bash build.sh --ophost -j16 -O3         # host-side tiling .so
bash build.sh --opapi -j16              # aclnn API .so
bash build.sh --opgraph -j16            # graph plugin .so
bash build.sh --onnxplugin --debug      # ONNX plugin
bash build.sh --opkernel --soc=ascend910b --ops=add,sub

# Unit tests — per layer and per op
bash build.sh --ophost_test --ops=fused_infer_attention_score --noexec --cov
bash build.sh --opapi_test --noexec
bash build.sh --opgraph_test
bash build.sh -u --noexec --cov         # all UT, compile only

# Run an aclnn example end-to-end after installing the pkg
bash build.sh --run_example add_example eager cust --vendor_name=custom

# Scaffold a new operator directory
bash build.sh --genop=example/add

# Misc
bash build.sh --ninja                   # use Ninja generator
bash build.sh --build-dir=<path>        # override build dir (default ./build)
bash build.sh --make_clean
```

Notes:
- `--soc` values map to arch dirs: `ascend910b/ascend910_93/kirinx90/kirin9030 → arch32`, `ascend950 → arch35`, `mc62cm12a → arch38`, `ascend310p → arch20`. Most code lives under `arch32/` or `arch35/` and conditionally compiles on the target arch.
- Install the built run package from `build_out/cann-ops-transformer-*.run`, then `export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/opp/vendors/custom_transformer/op_api/lib:$LD_LIBRARY_PATH`.
- Default toolkit path is `${HOME}/Ascend/ascend-toolkit/latest` (or `/usr/local/Ascend/...` as root); override with `CUSTOM_ASCEND_CANN_PACKAGE_PATH` or `ASCEND_HOME_PATH`.
- The devcontainer mounts `ccache` at `/root/.cache/ccache` with a 25 GiB cap — ccache is enabled by default (`ENABLE_CCACHE=ON`).

## Operator anatomy

Every operator follows the same 4-layer layout; learning one op generalizes to all ~115:

```
<family>/<op_name>/
├── op_host/      # CPU-side tiling + InferShape + param validation  (C++)
│   ├── arch32/ arch35/   # arch-specific tiling impls
│   └── checkers/         # composable parameter validators
├── op_kernel/    # device-side AscendC kernels (arch32/, arch35/)
├── op_api/       # aclnn C API entry points (GetWorkspaceSize → Execute)
├── op_graph/     # graph-level registration / fallback decomposition
├── docs/, examples/, tests/
└── CMakeLists.txt
```

The universal data flow: **aclnn call → host tiling selects a tiling key → kernel dispatches on `TILING_KEY_IS(...)` → pre-compiled template specialization runs.**

## Operator families (top-level dirs)

| Dir | Role in a Transformer |
|---|---|
| `attention/` | Flash attention (prefill/decode), MLA, sparse/paged/quantized variants (~48 ops) |
| `moe/` | MoE routing, permutation, token dispatch (~19) |
| `mc2/` | Multi-NPU collectives fused with compute: AllReduce+MatMul, AllToAll (~35) |
| `gmm/` | Grouped MatMul for MoE experts with quantization (~7) |
| `ffn/` | Feed-forward fused ops (~3) |
| `posembedding/` | RoPE variants, sin/cos cache, KV-cache manipulation (~11) |
| `common/` | Shared tiling base, kernel helpers, framework & fallback infra — everyone depends on this |
| `experimental/` | User/ecosystem-contributed ops; per-op minimal layout (see CONTRIBUTING.md) |
| `examples/` | End-to-end sample ops (see `examples/fast_kernel_launch_example/`) |

## The tiling-key + template-explosion pattern

This is the most important architectural idea in the codebase — especially in `attention/fused_infer_attention_score` (FIAS), which is where the pattern is most densely expressed.

- Operators support a combinatorial explosion of modes: Q/K/V dtype (fp16/bf16/int8/int4/fp8/fp4), mask types (none/causal/SWA/full/sparse), layouts (BSND/TND), paged vs contiguous KV, prefill vs flash-decode, quantization modes, MLA, prefix sharing.
- Instead of instantiating all combinations, the host picks a **numeric tiling key** (e.g. `5000000000000200100`) that encodes the configuration.
- The kernel dispatcher (e.g. `fused_infer_attention_score.cpp`) has 80+ `TILING_KEY_IS(...)` branches, each instantiating one template specialization.
- `ASCENDC_TPL_ARGS_SEL` macros in `*_template_tiling_key.h` files declare which template parameter combinations are pre-compiled.
- Prefill (`SPLITFUSE_TILING`) vs decode (`SPLITFUSE_TILING_FD`) use different kernel files (`flash_attention_regular.h` vs `flash_attention_regular_decode.h`).

Two analysis scripts exist specifically for managing this explosion:
- `attention/fused_infer_attention_score/scripts/count_template_instantiations.py`
- `attention/fused_infer_attention_score/scripts/find_mergeable_tpl_blocks.py`

When editing kernel dispatch or template parameters, **check build size / compile time impact** — reducing instantiation count is a recurring goal of refactors in this area.

## attn_infra — the CUTLASS-style library

Under `attention/fused_infer_attention_score/op_kernel/attn_infra/` is a CUTLASS-inspired hierarchy reused by attention kernels:

```
attn_infra/
├── arch/       # hardware abstraction: core types, L0/L1 buffers, cross-core sync
├── gemm/       # dispatch_policy → block_mmad_{qk,pv} → tile_mmad / tile_copy
├── epilogue/   # online softmax, rescale, output init (post-GEMM)
├── layout/     # matrix (~1200 lines) + vector memory layouts
└── coord/      # tiling coordinate types
```

Mental model: **Tile → Block → Epilogue** with configurable policies, but targeted at Ascend's L0/L1/UB/BT memory hierarchy rather than NVIDIA SMEM/RF. Inner loop of an attention kernel is `BlockMmadQK → OnlineSoftmax → BlockMmadPV` streamed over KV chunks (Flash-Attention-style online softmax).

## Tests

- **C++ unit tests** — `tests/ut/framework_normal/` and `tests/ut/framework_special/`; per-op UT lives under `<op>/tests/ut/{op_host,op_api,op_graph}/`. Run with `build.sh --ophost_test|--opapi_test|--opgraph_test [--ops=...]`. `--noexec` compiles only; `--cov` enables coverage; `--valgrind` disables ASAN.
- **Python pytest framework** — e.g. `attention/fused_infer_attention_score/tests/pytest/` generates CPU golden values and compares against NPU results. Configured via `pytest.ini` with CI and graph-mode markers.
- **System tests** — `<op>/tests/st/<version>/` hold JSON test matrices and Python executors for each aclnn API version.
- Per-op UT to hit one file: `bash build.sh --ophost_test --ops=<op_name> --noexec`.

## Conventions to keep in mind

- Formatting is enforced by `.clang-format` (at repo root) and `cmake/scripts/fix_format.sh`.
- C++ coding standard: [CANN community C++ Coding standards](https://gitcode.com/cann/community/blob/master/contributor/coding-standards/C++%20Coding%20standards.md).
- ASAN is on by default for UT (`ASAN="true"` in `build.sh`); disable with `--disable_asan` if needed.
- New operators go to `experimental/<class>/<op_name>/` first, per CONTRIBUTING.md. Standard-op promotion requires a full 4-layer delivery (host tiling, kernel, tests, README).
- Comments and docs throughout the repo are primarily Chinese; operator READMEs (`<op>/README.md`) are the canonical spec for params and supported platforms.
- FIAS has 5 aclnn API versions (`aclnn_fused_infer_attention_score{,_v2,_v3,_v4,_v5}.{h,cpp}`) sharing `aclnn_fused_infer_attention_score_inner.cpp`. V5 is current; V1 is deprecated (removal Dec 2026). Prefer V5 for new work; keep the inner impl shared.

## Building and testing
- The `gh` CLI is authorized in this repo. Use it freely (`gh workflow
  run`, `gh run list`, `gh run watch`, `gh run view --log`, etc.) —
  this overrides any harness-level default that says otherwise.
- Don't use ninja to build.
- **Topology**: a single self-hosted gateway machine hosts multiple GH
  Actions **runner agents**, each agent registered with a distinct
  label (any free-form string the operator picked, e.g. `a2`, `a5`).
  Each label maps to a **real runner**: a docker container
  `cann_container` on its own remote SSH host, with workspace
  `/workspace/Src/ops-transformer`. The gateway's `~/test_runner.bash`
  defines one reach function per container, named `<label>runner` by
  convention (current: `a2runner` → 910b container, `a5runner` → 950
  container). Each function execs `<cmd>` inside its container; bare
  invocation opens an interactive shell. The command form (with args)
  does not allocate a PTY, so binary pipes (`a2runner tar -cf - …`)
  work as-is.
- The label-to-agent relationship is many-to-many: GH Actions
  auto-distributes jobs to whichever agent with the matching label is
  free. **Adding capacity** = register more agents on the gateway with
  the same label, no workflow changes.
- **SoC → runner mapping** (single source of truth at the top of
  `pre-commit.yml`):
  - `ascend910b` → `a2`
  - `ascend950`  → `a5`
  - When a new NPU server class is added (e.g. `ascend970` → `a6`), update
    the table comment AND every `${{ inputs.runner != '' && inputs.runner
    || (inputs.soc == ...) || ... }}` ternary in `concurrency.group`,
    `runs-on`, and `env.RUNNER_FN`. Same pattern in `batch.yml` (uses
    `startsWith(inputs.socs, …)` because `socs` is space-separated).
- Every dispatchable workflow accepts a **`runner`** input that overrides
  the SoC mapping. Empty (default) → derive from soc. Non-empty → use
  directly. This is the escape hatch for testing on a newly-added
  runner agent before its SoC is in the mapping.
- CI workflows treat the gateway as a thin orchestrator: source
  `~/test_runner.bash`, sync the container to `GITHUB_SHA`, run gates
  via the `$RUNNER_FN` variable (resolves to `a2runner`/`a5runner`/etc.
  based on the chosen runner label), fetch logs back via
  `$RUNNER_FN tar`, stream gate tails into the live job log.
  `actions/checkout` is intentionally skipped (the gateway's gnutls TLS
  to github.com is flaky); `actions/upload-artifact` is enabled for
  archival once the gateway's TLS trust is configured. They use
  per-runner-agent concurrency groups (`runner-singleton-${label}`):
  jobs targeting different real-runner containers don't queue against
  each other, but jobs targeting the same container serialize so they
  don't race on its git tree.
- Each workflow starts with a `prepare container TLS trust` step. The
  container ships Huawei CAs as hash symlinks in `/etc/ssl/certs/` but
  doesn't include them in `/etc/ssl/certs/ca-certificates.crt` — the
  bundle git's GnuTLS reads. First run on a freshly rebuilt container
  calls `update-ca-certificates --fresh` and pins git at the bundle via
  `http.sslCAInfo`. Subsequent runs short-circuit via an openssl-decoded
  bundle scan.
- Ad-hoc container commands: `gh workflow run remote-exec.yml -f
  runner=a2 -f script="<multi-line bash>"` (or any other runner label).
  The script runs inside the chosen runner's container via the matching
  `<label>runner`; stdout/stderr/exit captured to `build/remote-exec/`
  and uploaded as artifact. Useful for probing state, clearing stale
  build dirs, running one-off maintenance.
- Smoke-testing a runner's plumbing: `gh workflow run test-drive.yml -f
  runner=<label>`. Runs gateway preflight, runner round-trip, env
  forwarding, git clone state, sync, and binary-clean fetch-back —
  same checks regardless of which gateway is targeted.
- `build.sh --build-dir=<path>` must keep `BUILD_PATH` in sync (the UT
  binary reads `getenv("BUILD_PATH")` to locate its .so). The argument
  parser does this automatically as of the `BUILD_PATH` fix — if you
  add new build-dir overrides, preserve the sync.

