# CLAUDE.md                                                                                                                                                                                                              
                                                                                                                                                                                                                           
  This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.                                                                                                                   
                                                                                                                                                                                                                           
  ## Project Overview                                                                                                                                                                                                      

  **ops-transformer** is an advanced operator library for transformer-based LLMs running on Ascend NPU hardware via the CANN framework. It provides optimized implementations for attention mechanisms, MoE, positional
  embeddings, grouped matmul, and cross-device communication operators.

  ## Build Commands

  ```bash
  # Build operator package (with precompiled kernels)
  bash build.sh --pkg --soc=ascend910b --ops=<operator_name> -j16

  # Build without kernel binaries (JIT compilation at runtime)
  bash build.sh --jit --soc=ascend910b --ops=<operator_name> -j16

  # Run operator example
  bash build.sh --run_example <operator_name> eager

  # Build and run C++ unit tests
  bash build.sh --ophost_test --ops=<operator_name>
  bash build.sh --opapi_test --ops=<operator_name>

  # Generate a new operator scaffold
  bash build.sh --genop=<category>/<operator_name>

  Testing

  Each operator has its own pytest suite under <operator>/tests/pytest/.

  # Run all tests for an operator
  cd attention/mla_prolog_v3/tests/pytest/
  pytest test.py -v

  # Run a single test
  pytest test.py::test_mla_prolog_v3 -v

  # Run only CI-marked tests
  pytest test.py -m ci

  # Fuzz tests (opt-in via env vars)
  MLA_PROLOG_V3_ENABLE_FUZZ=1 pytest test.py::test_mla_prolog_v3_fuzz
  MLA_PROLOG_V3_FUZZ_CASES=50 MLA_PROLOG_V3_FUZZ_SEED=42 pytest ...

  Pytest markers: ci (CI tests), graph (graph-mode tests), fuzz (randomized fuzz tests).

  Architecture

  Operator Layout

  Each operator follows a standard structure:

  <category>/<operator_name>/
  ├── op_host/           # Host-side: operator definition, tiling strategy, shape inference, ACLNN API
  ├── op_kernel/         # Device-side: Ascend C kernels for AI Core (arch-specific subdirs)
  ├── op_graph/          # Graph-mode compilation (optional)
  ├── framework/         # ONNX plugin (optional)
  ├── examples/          # C++ usage examples (test_aclnn_*.cpp)
  ├── tests/
  │   ├── pytest/        # Python pytest tests
  │   └── ut/            # C++ unit tests
  └── docs/

  Key Concepts

  - Tiling: Work decomposition that splits large tensors into cache-efficient blocks for AI Core execution. Implemented in *_tiling.cpp with data structures in *_tiling_data.h.
  - InferShape: Output tensor shape computation from input shapes (*_infershape.cpp).
  - Two execution modes: Eager (immediate execution via ACLNN API) and Graph (subgraph optimization via GE compiler).
  - API layers: ACLNN (Python-facing C API) → OpAPI (C++ wrapper) → Host/Kernel (low-level).

  Top-Level Directories

  - attention/, moe/, mc2/, gmm/, posembedding/, ffn/ — production operators by category
  - experimental/ — community-contributed custom operators
  - torch_extension/ — PyTorch extension package (npu_ops_transformer)
  - common/ — shared headers and source code
  - examples/ — tutorial operators (e.g., add_example)

  Multi-SoC Support

  Operators target multiple Ascend chips (910B, 950, 310P, etc.) with architecture-specific kernel directories (arch20/, arch32/, arch35/, etc.).

  Dependencies

  Core: pyyaml, numpy, decorator, sympy, scipy, attrs, protobuf, psutil
  Runtime: Ascend CANN Toolkit 8.5.0+, torch_npu (for PyTorch integration)

  Language Note

  Documentation and commit messages in this repo are primarily in Chinese.

  File write was denied by the current permission mode. You can create the file manually by copying the content above into `CLAUDE.md` at the project root.
