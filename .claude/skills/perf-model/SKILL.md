---
name: perf-model
description: Perform theoretical performance modeling for an Ascend NPU operator by reading its kernel code, tiling logic, and docs, then producing a roofline-based analysis report.
argument-hint: <operator_path> [--chip 950|910b] [--scenario decode|prefill] [--bs <N>] [--heads <N>] [--quant bf16|int8|fp8]
---

# Ascend NPU Operator Performance Modeling

You are performing **theoretical performance modeling** for an operator on Ascend NPU hardware. You must build the model from scratch by reading the operator's source code — do NOT assume any pre-existing analysis scripts exist.

## Input Arguments

Parse the following from `$ARGUMENTS`:
- **operator_path** (required): relative path to the operator directory (e.g., `attention/mla_prolog_v3`)
- **--chip**: target chip, default `950`. Load specs from `chip_specs.md` in this skill directory.
- **--scenario**: `decode` (T=BS) or `prefill` (T=BS×SeqLen), default `decode`
- **--bs**: batch size, default `1`
- **--heads**: number of attention heads, default `128`
- **--quant**: quantization mode (`bf16`, `int8`, `fp8`), default `bf16`
- **--seq-len**: sequence length for prefill, default `128`

## Workflow

### Phase 1: Read Operator Source

1. **Kernel code**: Read all `.h` files in `<operator_path>/op_kernel/` — look for:
   - Matmul calls (Mmad, MatmulSplitK, MatmulFullLoad, etc.)
   - Buffer size constants (L1_A_SIZE, L0A_PP_SIZE, etc.)
   - baseM, baseN, baseK, stepK parameters per matmul
   - Data types and quantization handling
   - Pipeline stages and sync points (AIC/AIV events)

2. **Tiling code**: Read `<operator_path>/op_host/*_tiling.cpp` and `*_tiling.h` — look for:
   - Core allocation per matmul (FillMatmul*Tiling functions)
   - N-axis split vs K-axis split strategy
   - stepBatchSize calculation
   - Workspace size computation

3. **Docs**: Read `<operator_path>/docs/` and `README.md` for:
   - Operator computation graph
   - Input/output tensor dimensions
   - Supported quantization modes

### Phase 2: Extract Matmul Parameters

For each matmul found, record:

| Field | Description |
|-------|-------------|
| Name | e.g., MM1_Cq, MM2_CkvKr |
| M | Batch dimension (typically T = BS × SeqLen) |
| K | Reduction dimension |
| N | Output dimension (total, before core split) |
| dtype_a | Left matrix element size in bytes |
| dtype_b | Right matrix element size in bytes |
| dtype_c | Output element size in bytes |
| split_mode | "split_n", "split_k", or "full_load" |
| active_cores | Number of AIC cores used |
| N_per_core | N / active_cores |
| a_reused | Whether A matrix is reused from a prior matmul (L2 warm) |

### Phase 3: Build Performance Model

Follow the methodology in `methodology.md` (in this skill directory):

1. Load chip specs from `chip_specs.md`
2. For each matmul, compute the inner-loop timing model
3. Apply buffer factor rules (single vs double buffer)
4. Compute roofline metrics (AI, ridge point, efficiency)
5. Build AIC/AIV pipeline DAG and find critical path

### Phase 4: Generate Report

Output a markdown report with these sections:

1. **Chip Specs Summary** — key hardware numbers
2. **Operator Overview** — matmul table with dimensions
3. **Roofline Analysis** — per-matmul AI vs ridge point, bound classification
4. **Detailed Timing** — per-matmul breakdown (MTE1, L0A, L0B, MMAD, FixPipe)
5. **Pipeline Analysis** — AIC/AIV timeline, critical path
6. **Optimization Recommendations** — based on bottleneck analysis

Save the report to `<operator_path>/tests/perf/report_<chip>.md`.

### Phase 5: Summarize

Present key findings to the user:
- Which matmuls dominate execution time
- Whether the operator is compute-bound or memory-bound
- Top optimization opportunities with expected impact
