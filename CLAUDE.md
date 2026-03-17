# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ops-transformer is the advanced transformer operator library for CANN (Compute Architecture for Neural Networks) on Ascend NPUs. It provides high-performance operators for transformer-based large models: attention, MoE, FFN, positional embedding, grouped matrix multiplication, and multi-core communication (mc2).

**CANN version**: 8.5.0+ | **Supported SoCs**: ascend910b (default), ascend910_93, ascend950, ascend310p, kirinx90, kirin9030, mc62cm12a

## Build Commands

All builds use `build.sh` from the project root. Run `bash build.sh --help` for full options.

```bash
# Build a custom operator package (most common during development)
bash build.sh --pkg --soc=ascend910b --ops=flash_attention_score -j16

# Build all operators into full ops-transformer package
bash build.sh --pkg --soc=ascend910b

# Build experimental directory operators
bash build.sh --pkg --experimental --soc=ascend910b --ops=<op_name>

# Build static library
bash build.sh --pkg --static --soc=ascend910b

# Build specific component libraries only
bash build.sh --ophost -j16        # Host-side only (OpDef, Tiling, InferShape)
bash build.sh --opapi -j16         # API layer only
bash build.sh --opgraph -j16       # Graph layer only
bash build.sh --opkernel --soc=ascend910b --ops=<op_name>  # Kernel binary only

# Clean build artifacts
bash build.sh --make_clean
```

**Build output**: `build_out/` (run packages), `build/` (intermediate artifacts), `output/` (generated files)

## Running Tests

```bash
# Run all UT (compile + execute)
bash build.sh -u

# Run UT for a specific operator and component
bash build.sh -u --ophost --ops=flash_attention_score
bash build.sh -u --opapi --ops=flash_attention_score
bash build.sh -u --opkernel --ops=flash_attention_score

# Compile UT without executing
bash build.sh -u --noexec

# Run with code coverage
bash build.sh -u --ophost --cov

# Run with specific SoC target
bash build.sh -u --ophost --soc=ascend910b

# Run operator example (after installing the built package)
bash build.sh --run_example <op_name> eager cust --vendor_name=custom
```

UT uses googletest. Test dependencies: `pip3 install -r tests/requirements.txt`

## Install & Run Operator Package

```bash
# Install the built .run package
./build_out/cann-ops-transformer-*linux*.run

# Set library path for custom packages
export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/opp/vendors/custom_transformer/op_api/lib:${LD_LIBRARY_PATH}

# Run an operator example
bash build.sh --run_example <op_name> eager cust --vendor_name=custom
```

## Architecture

### Operator Categories (top-level directories)

| Directory | Content |
|-----------|---------|
| `attention/` | Flash attention, sparse attention, incremental attention, KV cache ops |
| `moe/` | Mixture-of-experts: gating, token dispatch, expert computation |
| `mc2/` | Multi-core communication: all_gather_matmul, matmul_all_reduce |
| `ffn/` | Feed-forward networks, SwiGLU variants |
| `gmm/` | Grouped matrix multiplication |
| `posembedding/` | RoPE, positional encodings |
| `experimental/` | Community-contributed operators (simpler structure) |
| `examples/` | Reference examples (add_example as tutorial) |

### Standard Operator Structure

Each operator (e.g., `attention/flash_attention_score/`) contains:

```
op_host/                    # CPU-side: runs on host
├── *_def.cpp               # OpDef — inputs, outputs, attributes, type/format constraints
├── *_tiling.h/.cpp         # Tiling strategy — partitions work for limited UB memory
├── *_infershape.cpp        # Shape inference
├── arch32/, arch35/        # Architecture-specific tiling variants
└── op_api/                 # aclnn API bindings (aclnn_*.h/.cpp, version variants)

op_kernel/                  # NPU-side: runs on AI Core
├── *.cpp                   # Kernel entry: extern "C" __global__ function
├── *.h                     # Kernel class: Init → CopyIn → Compute → CopyOut
├── *_tiling_data.h         # TilingData struct shared between host and kernel
├── *_tiling_key.h          # Tiling strategy key definitions
└── arch32/, arch35/        # Architecture-specific kernel implementations

tests/ut/                   # Unit tests (op_host/, op_api/, op_kernel/)
examples/                   # Usage examples (test_aclnn_*.cpp)
docs/                       # Operator design documents
```

### Architecture-to-SoC Mapping

| Arch directory | SoC targets |
|---------------|-------------|
| `arch20` | ascend310p |
| `arch32` | ascend910b, ascend910_93, kirinx90, kirin9030 |
| `arch35` | ascend950 |
| `arch38` | mc62cm12a |

### Shared Code

- `common/include/` — Shared headers: tiling_base, kernel utilities, error handling, framework (ONNX) plugins
- `cmake/` — Build functions (`func.cmake`, `ut.cmake`, `config.cmake`, etc.)
- `torch_extension/` — PyTorch NPU extension bindings

### Kernel Programming Pattern

Kernels follow a pipeline pattern with queue-based data flow:

1. **Init**: Parse TilingData, allocate queues/buffers
2. **CopyIn**: `DataCopy` from Global → UB (unified buffer), `EnQue` to input queue
3. **Compute**: `DeQue` from input, compute (Add/Mul/etc.), `EnQue` to output queue
4. **CopyOut**: `DeQue` from output, `DataCopy` UB → Global

Key constraints: 32-byte memory alignment for all NPU API operations. Double buffering (`BUFFER_NUM=2`) overlaps pipeline stages.

## CI Pipeline

Triggered by commenting `compile` on a PR. Checks: code compilation → static analysis → UT tests → smoke tests. After passing, Committer adds `/lgtm`, Maintainer adds `/approve` to merge.

Test configuration in `tests/test_config.yaml` maps file changes to relevant test targets.

## Contribution Structure

- **Standard operators**: Full deliverables — `op_host/` (def + tiling) + `op_kernel/` (kernel + tiling_data) + `tests/ut/` + `README.md`
- **Ecosystem/experimental operators**: Minimal — single kernel `.cpp` + `tests/` + `CMakeLists.txt` + `README.md` under `experimental/`

## Code Style

C++ formatting defined in `.clang-format` (4-space indent, Huawei style). Follow the [C++ Coding Standards](https://gitcode.com/cann/community/blob/master/contributor/coding-standards/C++%20Coding%20standards.md).
