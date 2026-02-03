# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ops-transformer is a CANN (Compute Architecture for Neural Networks) operator library providing optimized transformer-based large model computation operators for Ascend NPUs (Atlas A2/A3, Ascend 950PR/950DT, Kirin X90).

## Build Commands
该仓库所有编译、二进制的执行都依赖cann包，通常cann包放在根目录下，命名为Ascend，你需要在编译、执行二进制之前执行`source Ascend/ascend-toolkit/set_env.sh`

```bash
# Build operator package for specific SoC
bash build.sh --pkg --soc=ascend950 --ops=<op_name> -j16

# Build JIT package (without kernel bin)
bash build.sh --jit --soc=ascend950 --ops=<op_name> -j16

# Run operator example
bash build.sh --run_example <op_name> eager cust --vendor_name=custom
```

## Test Commands

```bash
# Build and run all unit tests
bash build.sh -u

# Compile only (do not execute)
bash build.sh -u --noexec

# Build and run specific UT layer
bash build.sh --ophost_test           # ophost layer tests
bash build.sh --opapi_test            # opapi layer tests
bash build.sh --opgraph_test          # opgraph layer tests
bash build.sh --opkernel_test         # opkernel layer tests

# Enable code coverage
bash build.sh -u --cov

# Specify SoC version
bash build.sh -u --soc=ascend950

# Run with valgrind (disables ASAN and noexec)
bash build.sh -u --valgrind

# O3 optimization compile (no execute)
bash build.sh --ophost_test --noexec -O3
```

## Architecture Overview

### Directory Structure

```
ops-transformer/
├── attention/          # Flash attention, MLA, NSA, sparse attention, etc.
├── mc2/                # Matrix compute: matmul, all2all, reduce, attention_to_ffn
├── moe/                # Mixture-of-Experts operators
├── ffn/                # Feed-Forward Network operators
├── gmm/                # Grouped MatMul operators
├── posembedding/       # Position embedding operators (RoPE, etc.)
├── common/             # Shared headers, utilities, ONNX plugin framework
├── experimental/       # User-contributed/experimental operators
└── tests/              # Test utilities
```

### Operator Directory Structure

Each operator follows this pattern:
```
${op_name}/
├── CMakeLists.txt           # Build configuration
├── op_host/                 # Operator definition, Tiling, InferShape
│   ├── ${op_name}_def.cpp
│   ├── ${op_name}_tiling.cpp
│   ├── ${op_name}_inferred_shape.cpp
│   └── arch32/              # Architecture-specific code
├── op_kernel/               # AICore kernel implementation (AscendC)
│   ├── ${op_name}.cpp
│   ├── ${op_name}.h
│   └── arch32/
├── op_api/                  # ACLNN API bindings
├── op_graph/                # Graph optimization/fusion passes
├── framework/               # ONNX plugin integration
└── tests/
    ├── ut/                  # C++ gtest unit tests
    └── pytest/              # Python integration tests
```

### Key Components

| Component | Purpose |
|-----------|---------|
| **op_host** | Operator metadata, tiling strategy, shape inference |
| **op_kernel** | Core AICore kernel (AscendC programming model) |
| **op_api** | ACLNN (Ascend C Neural Network) bindings |
| **op_graph** | Graph optimization and fusion passes |
| **framework** | ONNX plugins (NPUFlashAttention, NPUMultiHeadAttention) |

### Supported SoC/Architecture Mapping

| SoC | Arch Directory |
|-----|----------------|
| ascend310p | arch20 |
| ascend910b | arch32 |
| ascend950 | arch35 |

## Code Style

- **Standard**: C++17
- **Formatting**: Clang-Format (column limit: 120)
- **Copyright**: All files require CANN Open Software License header

## Key Conventions

- Operators use lowercase with underscores: `flash_attention_score`, `grouped_matmul_allto_allv`
- Grad operators: `${op_name}_grad`
- Versioned operators: `mla_prolog_v2`, `mla_preprocess_v3`

## References

- [QuickStart](QUICKSTART.md): End-to-end development guide
- [Operator Development Guide](docs/zh/develop/aicore_develop_guide.md)
- [CANN Simulator Debug](docs/zh/debug/cann_sim.md)
