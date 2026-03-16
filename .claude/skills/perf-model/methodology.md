# Performance Modeling Methodology

## Overview

This document describes how to compute theoretical execution time for matmul-based operators on Ascend NPU. The model captures the **L1/L0 cache hierarchy** and uses the **roofline model** to classify bottlenecks.

## Architecture: Matmul Inner Loop

An Ascend Cube core executes matmul via a tiled inner loop:

```
Outer loop (kL1): HBM/L2 → L1 (MTE1 engine)
  Inner loop (stepK): L1 → L0A/L0B + MMAD (Cube engine)
  Output: L0C → FixPipe → GM/workspace
```

### Split-K Mode (most matmuls)

```
kL1StepSize = baseK × stepK
kL1Loops = ceil(K / kL1StepSize)

for kL1 in range(kL1Loops):           # outer loop
    MTE1: Load A[baseM, kL1StepSize] from HBM/L2 → L1
    MTE1: Load B[kL1StepSize, N_per_core] from HBM → L1

    for step in range(stepK):          # inner loop
        L1→L0A: A[baseM, baseK]
        L1→L0B: B[baseK, N_per_core]  (transposed)
        MMAD:   C[baseM, N_per_core] += A × B

    FixPipe: C[baseM, N_per_core] → output
```

### Full-Load Mode (small K, e.g., K=128)

```
MTE1: Load A[M, K] from HBM → L1 (once)
nSplits = ceil(N_per_core / nSplitSize)

for split in range(nSplits):
    MTE1: Load B[K, nSplitSize] from HBM → L1
    L1→L0A: A[M, K]
    L1→L0B: B[K, nSplitSize]
    MMAD:   C[M, nSplitSize] = A × B
    FixPipe: C → output
```

## Buffer Factor Rule

For each cache level, determine if the **full matrix** can fit:

```
buffer_factor = 1  if full_matrix_bytes <= cache_size    (single buffer, full load)
buffer_factor = 2  if full_matrix_bytes > cache_size     (double buffer, ping-pong)
```

Apply per-buffer:

| Buffer | Full matrix | Tile | Check |
|--------|------------|------|-------|
| L0A | M × K × dtype_a | baseM × baseK × dtype_a × factor | tile × factor ≤ l0a_size |
| L0B | K × N_pc × dtype_b | baseK × baseN × dtype_b × factor | tile × factor ≤ l0b_size |
| L0C | M × N_pc × dtype_c | baseM × baseN × dtype_c × factor | tile × factor ≤ l0c_size |
| L1 | (M×K×dtype_a) + (K×N_pc×dtype_b) | per_step_A + per_step_B | see below |

**L1 stepK derivation**:
```python
per_step = baseM * baseK * dtype_a + baseK * N_per_core * dtype_b

# Try single buffer (all K in one L1 load):
single_buf_stepK = L1_size // per_step
if single_buf_stepK >= ceil(K / baseK):
    stepK = ceil(K / baseK)       # full load OK
else:
    # Need double buffer:
    stepK = L1_size // 2 // per_step
    stepK = min(stepK, ceil(K / baseK))
```

## Timing Computation

### Per-kL1 Iteration

```python
# 1. MTE1: HBM/L2 → L1
load_A_bytes = baseM * kL1StepSize * dtype_a
load_B_bytes = kL1StepSize * N_per_core * dtype_b

# Apply bandwidth with contention:
# effective_bw = min(per_cycle * freq_hz, chip_total / active_cores)
load_A_time = load_A_bytes / a_bandwidth    # HBM or L2 (if reused)
load_B_time = load_B_bytes / hbm_bandwidth
mte1_time = max(load_A_time, load_B_time)

# 2. Inner loop: L1 → L0 + MMAD (double-buffered overlap)
l0a_time = baseM * baseK * dtype_a / l1_to_l0a_bw
l0b_time = baseK * N_per_core * dtype_b / l1_to_l0b_bw
mmad_time = (baseM * N_per_core * baseK) / mmad_throughput   # MACs / (MACs/sec)
inner_iter = max(l0a_time, l0b_time, mmad_time)
total_inner = stepK * inner_iter

# 3. FixPipe: L0C → output
fixpipe_time = baseM * N_per_core * dtype_c / l0c_to_out_bw

# 4. Pipelined overlap across stages
per_kl1 = max(mte1_time, total_inner, fixpipe_time)
```

### Total Matmul Time

```python
total_time = kL1Loops * per_kl1
```

### Bound Classification

The bottleneck is whichever stage dominates `per_kl1`:
- `mte1_time` → **HBM-bound** (or L2-bound if A is reused)
- `total_inner` → **CUBE-bound** (if mmad dominates) or **L0A/L0B-bound**
- `fixpipe_time` → **FixPipe-bound**

## Roofline Analysis

### Arithmetic Intensity (AI)

```
AI = 2 × M × N × K / (M×K×dtype_a + K×N×dtype_b + M×N×dtype_c)
```

Units: FLOPs / Byte. Higher AI means more compute per byte of data moved.

### Ridge Point

```
ridge_point = peak_compute_per_core / memory_bandwidth_per_core
```

| Chip | BF16 Ridge (HBM) | FP8 Ridge (HBM) |
|------|-------------------|-----------------|
| 950 | 270.3 FLOPS/B | 540.7 FLOPS/B |

### Classification

- AI < ridge_point → **Memory-bound**: optimize data movement
- AI > ridge_point → **Compute-bound**: optimize MMAD utilization

### Efficiency

```
achieved_throughput = 2 × M × N × K / actual_time
efficiency = achieved_throughput / peak_throughput
```

## Pipeline Analysis

### AIC/AIV DAG

Build a DAG of stages with dependencies:
```
AIC: MM1 → signal → MM2 → signal → wait(RmsNorm) → MM3 → signal → wait(Dequant) → MM4
AIV: CopyIn → wait(MM1) → RmsNorm → signal → wait(MM2) → RmsNorm → RoPE → Scatter → ...
```

### Critical Path

1. Topological sort the DAG
2. Forward pass: `start[n] = max(end[dep] for dep in n.depends_on)`
3. `end[n] = start[n] + duration[n]`
4. Total time = max(end[n] for all n)
5. Backtrack from the last node to find the critical path

## Optimization Analysis

### Memory-Bound Optimizations

| Technique | Mechanism | Expected Impact |
|-----------|-----------|----------------|
| INT8/FP8 quantization | Halve weight bytes | ~2x speedup |
| L2 reuse (shared A) | Avoid HBM re-read | Depends on reuse pattern |
| Increase stepK | Fewer kL1 iterations | Reduce MTE1 overhead |
| Cross-core K-split | More cores on small-N matmuls | Up to core_ratio × faster |

### Cross-Core K-Split

When a matmul has small N (low core utilization with split-N):
- Split K across k_cores instead of N
- Each core computes a partial sum → FixPipe outputs float32 (4B)
- Vector side accumulates: load k_cores partial sums, reduce, store
- Trade-off: FixPipe 2x + accumulation overhead vs. more core parallelism

```
cube_time = estimate(K/k_cores, N_full, k_cores)
accum_time = estimate_vector(load=k_cores*M*N*4, compute=M*N*(k_cores-1), store=M*N*out_size)
total = cube_time + accum_time
```

### Compute-Bound Optimizations

| Technique | Mechanism |
|-----------|-----------|
| Increase baseM | More rows per MMAD → better MMAD utilization |
| Use full L0A/L0B | Maximize tile size within cache constraints |
| Reduce kL1 overhead | Larger stepK to amortize L1 load cost |

## Output Format

Present results as a markdown report with tables:

```markdown
## Roofline Analysis

| Matmul | M | K | N | Cores | AI | Ridge | Bound | Time (us) | Efficiency |
|--------|---|---|---|-------|-----|-------|-------|-----------|-----------|
| MM1    | T | K1| N1| C1    | ... | ...   | HBM   | ...       | ...       |

## Timing Breakdown

| Matmul | MTE1 (us) | L0A (us) | L0B (us) | MMAD (us) | FixPipe (us) | Total (us) | kL1 | Bound |
|--------|-----------|----------|----------|-----------|-------------|-----------|-----|-------|

## Pipeline

Total: XX.XX us
Critical path: MM1(XX) → MM2(XX) → MM3(XX) → ...

## Recommendations

1. ...
2. ...
```
