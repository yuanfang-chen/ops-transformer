# MLA Prolog V3 Performance Modeling Tools

Theoretical performance modeling for the MLA Prolog V3 operator on Ascend NPU. Estimates execution time from chip specs and kernel tiling parameters — no hardware required.

## Quick Start

```bash
cd attention/mla_prolog_v3/tests/perf

# Full analysis on Ascend 950
python perf_analyzer.py --chip 950 --batch-size 1 --seq-len 1 --head-num 128 --mode full

# Report across all test cases
python perf_analyzer.py --chip 950 --mode report

# Ascend 910B (default, legacy chip-total model)
python perf_analyzer.py --batch-size 1 --seq-len 1 --head-num 128 --mode bound
```

## CLI Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--chip` | Target chip (`910b`, `950`) | `910b` |
| `--from-testcase NAME` | Load params from `testcases.py` | — |
| `--batch-size N` | Batch size (mutually exclusive with `--from-testcase`) | — |
| `--seq-len N` | Sequence length | 4 |
| `--head-num N` | Number of attention heads | 8 |
| `--He N` | Hidden size | 7168 |
| `--weight-quant-mode` | 0=BF16, 1=INT8-partial, 2=INT8-full, 3=MXFP8 | 0 |
| `--kv-cache-quant-mode` | KV cache quantization mode | 0 |
| `--query-quant-mode` | 0=normal, 1=dynamic quant | 0 |
| `--mode` | Analysis mode (see below) | `full` |

## Analysis Modes

### `bound` — Per-Stage Resource Breakdown

Shows timing for each matmul (MTE1, L0A, L0B, MMAD, FixPipe) and vector op (MTE2, Vector, MTE3), identifying which resource bounds each stage.

On Ascend 950 (per-cycle specs available), shows inner-loop detail columns: L0A, L0B, MMAD per iteration, and kL1 loop count.

```bash
python perf_analyzer.py --chip 950 --batch-size 1 --head-num 128 --mode bound
```

### `pipeline` — AIC/AIV Timeline Visualization

ASCII art timeline showing how AIC (cube) and AIV (vector) stages overlap. Identifies the critical path through the pipeline DAG.

```bash
python perf_analyzer.py --chip 950 --batch-size 1 --head-num 128 --mode pipeline
```

### `estimate` — Kernel Duration Estimate

Reports estimated total kernel time, accounting for multi-step execution when T > stepBatchSize.

```bash
python perf_analyzer.py --chip 950 --batch-size 32 --head-num 128 --mode estimate
```

### `advice` — Performance Improvement Suggestions

Automated bottleneck detection and optimization advice: MTE1-bound stages, low core utilization, AIC/AIV imbalance, high sync overhead, scatter inefficiency.

```bash
python perf_analyzer.py --chip 950 --batch-size 1 --head-num 128 --mode advice
```

### `search` — Tiling Configuration Search

Explores stepBatchSize candidates (8, 16, 32, 64, 128) and ranks by estimated kernel time.

```bash
python perf_analyzer.py --chip 950 --batch-size 32 --head-num 128 --mode search
```

### `report` — All Test Cases Comparison

Runs all `perf_*` test cases from `testcases.py` and prints a comparison table with per-matmul timing, dominant bottleneck, and quant speedup comparison.

```bash
python perf_analyzer.py --chip 950 --mode report
```

### `roofline` — Roofline Model Analysis

Per-matmul roofline analysis: arithmetic intensity, peak vs achieved GFLOPS, efficiency, and bound classification. Shows ridge points for HBM bandwidth. Requires `--chip 950`.

```bash
python perf_analyzer.py --chip 950 --batch-size 1 --head-num 128 --mode roofline
```

### `block-search` — Optimal baseM/baseN/baseK Search

Exhaustive search over block tile sizes (baseM, baseN, baseK) for each matmul, with stepK auto-derived from L1 constraint. Includes MM2 split-N vs split-K comparison. Requires `--chip 950`.

```bash
python perf_analyzer.py --chip 950 --batch-size 32 --head-num 128 --mode block-search
```

### `split-kn` — 2D Core Split (N×K) Search

Searches 2D split configurations for MM1 and MM2 — splitting both N and K axes across cores. Evaluates each configuration via the **full pipeline** (all 4 matmuls + vector ops) to capture overlap between AIC accumulation and AIV stages.

Reports unified metrics per configuration:
- **Pipeline**: end-to-end kernel duration (critical path)
- **MM1–MM4**: per-matmul cube time
- **Accum**: vector accumulation overhead (may overlap with next AIC stage)
- **CubeUtil**: AIC busy time / pipeline total
- **AIC/AIV**: total busy time per pipe

Requires `--chip 950`.

```bash
python perf_analyzer.py --chip 950 --batch-size 1 --head-num 128 --mode split-kn
```

### `full` — All Modes Combined

Runs bound + pipeline + estimate + advice + search. On Ascend 950, also runs roofline + block-search + split-kn.

## Architecture

```
perf_model.py       Hardware specs, inner-loop timing model, cache validation
tiling_sim.py       Python replica of C++ tiling logic, block/split-KN search
pipeline_model.py   AIC/AIV pipeline DAG, critical path, roofline analysis
perf_analyzer.py    CLI entry point, all analysis modes
```

### Key Concepts

- **Chip-total model** (910B): bandwidth in bytes/sec, compute in TFLOPS. No cache hierarchy detail.
- **Per-cycle model** (950): bandwidth in bytes/cycle + frequency, per-core cache sizes (L0A/L0B/L0C/L1). Models the matmul inner loop (MTE1 → L0 load → MMAD → FixPipe).
- **Buffer strategy**: double buffer (2×) by default. Full-load single buffer (1×) only when the complete matrix fits and it improves performance.
- **Split-KN**: 2D core grid (n_groups × k_groups). K-split cores produce float32 partial sums; vector side accumulates. Pipeline DAG captures overlap between accumulation and next matmul.

## Generated Reports

- `report_ascend950.md` — Comprehensive analysis for Ascend 950 across quantization modes
