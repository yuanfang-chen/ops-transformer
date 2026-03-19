# Performance Modeling Methodology for MLA Prolog on Ascend NPU

This document describes how to compute theoretical execution time for MLA Prolog (and similar matmul-based operators) on Ascend NPU. Each section includes **worked examples** using MLA Prolog V3 on Ascend 950 with BF16 decode BS=1.

---

## 1. Hardware Memory Hierarchy

### 1.1 Data Flow Architecture

An Ascend AI Core has two compute pipelines sharing HBM access:

```
                        ┌─────────────────────────────────────┐
                        │              HBM (off-chip)          │
                        │         Chip total: 1.6 TB/s         │
                        └──────┬──────────────────┬────────────┘
                               │                  │
                        ┌──────▼──────┐    ┌──────▼──────┐
                        │  L2 Cache   │    │  L2 Cache   │
                        │  5.2 TB/s   │    │  5.2 TB/s   │
                        └──────┬──────┘    └──────┬──────┘
                               │                  │
                ┌──────────────▼──┐          ┌────▼──────────────┐
                │  Cube Pipeline  │          │  Vector Pipeline  │
                │                 │          │                   │
                │   ┌────────┐   │          │   ┌───────────┐   │
          MTE1/ │   │   L1   │   │    MTE2  │   │    UB     │   │
          MTE2  │   │ 512 KB │   │          │   │  256 KB   │   │
                │   └───┬────┘   │          │   └─────┬─────┘   │
                │  MTE1 │        │          │   PIPE_V│         │
                │   ┌───▼──┐ ┌──▼──┐       │   ┌─────▼─────┐   │
                │   │ L0A  │ │ L0B │       │   │  Compute  │   │
                │   │64 KB │ │256KB│       │   │ (in UB)   │   │
                │   └───┬──┘ └──┬──┘       │   └─────┬─────┘   │
                │    MMAD│      │          │         │MTE3     │
                │   ┌───▼──────▼──┐       │         │         │
                │   │    L0C      │       │         ▼         │
                │   │   256 KB    │       │      HBM out      │
                │   └──────┬──────┘       │                   │
                │   FixPipe│              │                   │
                │          ▼              │                   │
                │       HBM out           │                   │
                └─────────────────────────┴───────────────────┘
```

### 1.2 Data Movement Engines

| Engine | Path | Used By | Bandwidth (950) |
|--------|------|---------|-----------------|
| **MTE1/MTE2** | HBM/L2 → L1 | Cube: load A and B matrices | 33.3 B/cycle per core |
| **MTE1** | L1 → L0A | Cube: load A tile for MMAD | 256 B/cycle |
| **MTE1** | L1 → L0B | Cube: load B tile for MMAD (transposed) | 256 B/cycle |
| **MMAD** | L0A × L0B → L0C | Cube: matrix multiply-accumulate | 4096 MACs/cycle (BF16) |
| **FixPipe** | L0C → HBM/UB | Cube: write matmul output | 256 B/cycle |
| **MTE2** | HBM/L2 → UB | Vector: load input data | 33.3 B/cycle per core |
| **PIPE_V** | UB → UB | Vector: compute (RmsNorm, RoPE, etc.) | — |
| **MTE3** | UB → HBM | Vector: store output data | 33.3 B/cycle per core |

### 1.3 Bandwidth Contention

Per-core bandwidth is limited by chip-total caps when all cores are active:

```
effective_per_core_bw = min(per_cycle × freq, chip_total / active_cores)
```

**Example (Ascend 950, 32 cores)**:
- HBM per core: min(33.3 × 1.65e9, 1.6e12 / 32) = min(54.9, 50.0) = **50.0 GB/s**
- L2 per core: min(108.0 × 1.65e9, 5.2e12 / 32) = min(178.2, 162.5) = **162.5 GB/s**
- L1→L0A: 256 × 1.65e9 = **422.4 GB/s** (no cross-core contention)

### 1.4 L2 Cache Reuse

Data loaded by one core from HBM stays in L2, allowing other cores to read it at L2 bandwidth:
- **Split-N reuse**: All cores share the same A matrix → first core loads from HBM, others hit L2
- **Cross-matmul reuse**: MM2 reuses tokenX (A matrix) that MM1 already loaded → L2 warm

---

## 2. Roofline Model

### 2.1 Arithmetic Intensity

```
AI = 2 × M × N × K / (M × K × dtype_a + K × N × dtype_b + M × N × dtype_c)
```

Units: FLOPs per Byte. Higher AI = more compute per byte of data moved.

### 2.2 Ridge Point

The arithmetic intensity where compute time equals memory time:

```
ridge_point = peak_compute / memory_bandwidth    (FLOPs/Byte)
```

**Ascend 950 (per core, 32 cores active)**:

| Dtype | Peak Compute | HBM BW | Ridge Point |
|-------|-------------|--------|-------------|
| BF16 | 4096 × 2 × 1.65e9 = 13,517 GFLOPS | 50 GB/s | **270.3 FLOPs/B** |
| FP8 | 8192 × 2 × 1.65e9 = 27,034 GFLOPS | 50 GB/s | **540.7 FLOPs/B** |

### 2.3 Bound Classification

- AI < ridge → **Memory-bound**: execution limited by HBM bandwidth
- AI > ridge → **Compute-bound**: execution limited by MMAD throughput

### 2.4 Worked Example: MM3 (QcQr), BF16, BS=1

MM3: `Cq[1, 1536] × W_UqQr[1536, 24576]`, 32 cores, N_per_core = 768

```
AI = 2 × 1 × 768 × 1536 / (1 × 1536 × 2 + 1536 × 768 × 2 + 1 × 768 × 2)
   = 2,359,296 / 2,362,368 = 1.0 FLOPs/B

Ridge (BF16) = 270.3 FLOPs/B
→ AI (1.0) << Ridge (270.3) → HBM-bound

Efficiency = AI / Ridge = 1.0 / 270.3 = 0.4%
```

---

## 3. Matmul Modeling

### 3.1 Inner Loop Structure

The Ascend Cube core executes matmul via a tiled two-level loop:

```
Outer loop (kL1 iterations): HBM/L2 → L1 via MTE1/MTE2
  Load A[baseM, kL1StepSize] to L1       ← MTE1/MTE2
  Load B[kL1StepSize, N_per_core] to L1  ← MTE1/MTE2

  Inner loop (stepK iterations): L1 → L0 + MMAD
    Load A[baseM, baseK] from L1 to L0A  ← MTE1
    Load B[baseK, N_per_core] from L1 to L0B (transposed)  ← MTE1
    C[baseM, N_per_core] += A × B        ← MMAD

  Write C[baseM, N_per_core] to HBM      ← FixPipe
```

Where `kL1StepSize = baseK × stepK` and `kL1Loops = ceil(K / kL1StepSize)`.

### 3.2 Timing Model

**Per kL1 iteration** (all three stages overlap via double buffering):

```python
# Stage 1: MTE1 — HBM/L2 → L1
mte1_time = max(A_bytes / a_bw, B_bytes / hbm_bw)

# Stage 2: Inner loop — L1 → L0 + MMAD (double-buffered)
l0a_time = baseM × baseK × dtype_a / l0a_bw
l0b_time = baseK × N_per_core × dtype_b / l0b_bw
mmad_time = baseM × N_per_core × baseK / mmad_throughput
inner_iter = max(l0a_time, l0b_time, mmad_time)
total_inner = stepK × inner_iter

# Stage 3: FixPipe — L0C → HBM
fixpipe_time = baseM × N_per_core × dtype_c / fixpipe_bw

# Pipelined: stages overlap across kL1 iterations
per_kl1 = max(mte1_time, total_inner, fixpipe_time)
total = kL1Loops × per_kl1
```

### 3.3 Mode A: Full-Load (MM4)

When K is small enough that the entire A matrix fits in L1, load A once and iterate over N splits.

**Example: MM4 (Qn)** — `Qc[1, 128] × Uk[128, 512]`, 32 cores, N_per_core=16

```
A = 1 × 128 × 2B = 256 B  → fits in L1 (512 KB), single buffer OK
Load A once from HBM: 256 / 50e9 = 0.005 us

nSplitSize = 128
nSplits = ceil(16 / 128) = 1

Per split:
  Load B[128, 16] = 4096 B from HBM: 4096 / 50e9 = 0.082 us
  L0A = 256 / 422.4e9 = 0.0006 us
  L0B = 4096 / 422.4e9 = 0.0097 us
  MMAD = 1 × 16 × 128 / 6.758e9 = 0.0003 us
  FixPipe = 1 × 16 × 2 / 422.4e9 = 0.00008 us
  per_split = max(0.082, 0.0097, 0.00008) = 0.082 us

Total = 0.005 + 1 × 0.082 = 0.087 us
→ HBM-bound (B matrix load dominates)
```

Actual model result: **0.66 us** (includes overhead and rounding).

### 3.4 Mode B: Split-N (MM3, standard)

N axis is split across cores. Each core loads full K, different N slice.

**Example: MM3 (QcQr)** — `Cq[1, 1536] × W[1536, 24576]`, BF16, 32 cores

```
N_per_core = 24576 / 32 = 768
Block: baseM=32, baseN=256, baseK=64, stepK=2

kL1StepSize = 64 × 2 = 128
kL1Loops = ceil(1536 / 128) = 12

Per kL1 iteration:
  MTE1:
    A = 1 × 128 × 2B = 256 B → 256 / 50e9 = 0.005 us
    B = 128 × 768 × 2B = 196,608 B → 196,608 / 50e9 = 3.932 us
    mte1 = max(0.005, 3.932) = 3.932 us  ← B load dominates

  Inner (per stepK):
    L0A = 1 × 64 × 2 / 422.4e9 = 0.0003 us
    L0B = 64 × 768 × 2 / 422.4e9 = 0.233 us
    MMAD = 1 × 768 × 64 / 6,758.4e9 = 0.007 us
    inner_iter = max(0.0003, 0.233, 0.007) = 0.233 us
  total_inner = 2 × 0.233 = 0.466 us

  FixPipe = 1 × 768 × 2 / 422.4e9 = 0.004 us

  per_kl1 = max(3.932, 0.466, 0.004) = 3.932 us  ← MTE1 (HBM load of B)

Total = 12 × 3.932 = 47.19 us
→ HBM-bound, inner bound = MTE1_B
```

### 3.5 Mode C: Split-KN (2D split for MM1/MM2)

Both N and K axes split across cores. K-split cores produce float32 partial sums; vector side accumulates.

**Core grid**: `n_groups × k_groups` cores. Each computes `M × (K/k_groups) × (N/n_groups)`.

**Example: MM1 (Cq)** — `X[1, 7168] × W[7168, 1536]`, config N8×K4 (32 cores)

```
K_per_core = 7168 / 4 = 1792
N_per_core = 1536 / 8 = 192
dtype_c = 4B (float32, for partial sum)

Cube (per core, 32 cores contending on HBM):
  HBM per core = 50.0 GB/s
  Per kL1 (stepK derived from L1):
    A = 1 × kL1StepSize × 2B → loaded from L2 (shared across N-group)
    B = kL1StepSize × 192 × 2B → from HBM
  kL1Loops = 4, per_kl1 ≈ 0.98 us
  Cube total = 3.93 us

Vector accumulation (per N-group, 8 groups in parallel):
  Load: 4 partial sums × 1 × 192 × 4B = 3,072 B
  Compute: 1 × 192 × 3 add ops = 576 ops
  Store: 1 × 192 × 2B = 384 B
  Accum per group = 0.002 us (negligible)

Total MM1 = 3.93 + 0.002 = 3.93 us (vs 17.89 us with pure split-N)
→ 4.6x faster, still HBM-bound but much better core utilization
```

In the pipeline, the accum (0.002 us) runs on AIV while MM2 starts on AIC — fully hidden.

---

## 4. Vector Operation Modeling

Vector ops run on the AIV pipeline: **MTE2** (HBM→UB) → **PIPE_V** (compute in UB) → **MTE3** (UB→HBM).

### 4.1 General Formula

```python
mte2_time = load_bytes / (mte2_bw / active_cores)    # per-core load
vec_time = compute_elements / (vec_throughput / aiv_num)  # per-core compute
mte3_time = store_bytes / (mte3_bw / active_cores)    # per-core store
total = max(mte2_time, vec_time, mte3_time)           # pipelined overlap
```

For scatter (indexed write): `mte3_bw *= scatter_efficiency (0.4)`.

### 4.2 RmsNorm

Load input, compute element-wise normalize (mul + add + rsqrt ≈ 3 ops/element), store output.

**Example: RmsNormCq** — T=1, Hcq=1536, input INT32 (4B), output BF16 (2B), 1 vector core

```
Load:    1 × 1536 × 4B = 6,144 B
Compute: 1 × 1536 × 3 = 4,608 ops
Store:   1 × 1536 × 2B = 3,072 B

mte2 = 6,144 / (1.6e12 / 1) = 0.004 us
vec  = 4,608 / (20e12 / 64) = 0.015 us  ← dominates
mte3 = 3,072 / (1.6e12 / 1) = 0.002 us

Total = 0.015 us, bound = VECTOR
```

### 4.3 RoPE (Rotary Position Embedding)

Apply rotary embedding: load data + sin/cos tables, compute (mul, sub, mul, add ≈ 4 ops/element), store.

**Example: RopeKr** — T=1, Dr=64, BF16, 1 vector core

```
Load:    1 × 64 × 2B (data) + 1 × 64 × 2B (sin/cos) = 512 B
Compute: 1 × 64 × 4 = 256 ops
Store:   1 × 64 × 2B = 128 B

Total ≈ 0.001 us (negligible — small data)
```

### 4.4 Scatter (KV Cache Write)

Indexed write to KV cache — random access pattern reduces effective MTE3 bandwidth.

**Example: ScatterCkvKr** — T=1, Hckv+Dr=576, BF16, 1 vector core

```
Load:    1 × 576 × 2B = 1,152 B
Compute: 1 × 576 = 576 ops (address compute)
Store:   1 × 576 × 2B = 1,152 B (scattered)

mte3 effective BW = 1.6e12 × 0.4 (scatter efficiency)
Total ≈ 0.002 us
```

### 4.5 Dequant (INT32 → BF16)

Dequantize matmul output: load INT32 values + scale/offset, multiply + add, cast to BF16.

**Example: DequantQc** — T=1, N=128, D=128, input 4B, output 2B, vec_cores vector cores

```
Load:    1 × 128 × 128 × 4B = 65,536 B
Compute: 1 × 128 × 128 × 2 = 32,768 ops (mul + add)
Store:   1 × 128 × 128 × 2B = 32,768 B

Total ≈ 0.1 us
```

### 4.6 DynamicQuant (BF16 → INT8)

Compute per-row scale, quantize to INT8. More expensive: needs min/max reduction + scale + quantize.

**Example: DynamicQuantQn** — T=1, N=128, Hckv=512, input 2B, output 1B

```
Load:    1 × 128 × 512 × 2B = 131,072 B
Compute: 1 × 128 × 512 × 3 = 196,608 ops (reduction + scale + quant)
Store:   1 × 128 × 512 × 1B = 65,536 B
```

### 4.7 RoPE Qr (largest vector op in decode)

**Example: RopeQr** — T=1, N=128, Dr=64, BF16

```
Load:    1 × 128 × 64 × 2B (Qr) + 1 × 64 × 2B (sin/cos) = 16,512 B
Compute: 1 × 128 × 64 × 4 = 32,768 ops
Store:   1 × 128 × 64 × 2B = 16,384 B

vec = 32,768 / (20e12 / 64) = 0.105 us  ← dominates
Total = 0.105 us, bound = VECTOR
```

---

## 5. Buffer Strategy

### 5.1 Default: Double Buffer (2×)

Double buffering overlaps data load with compute via ping-pong. Each buffer level needs 2× tile size:

```
L0A: baseM × baseK × dtype_a × 2 ≤ 64 KB
L0B: baseK × baseN × dtype_b × 2 ≤ 256 KB
L0C: baseM × baseN × dtype_c × 2 ≤ 256 KB
L1:  (A_per_step + B_per_step) × 2 ≤ 512 KB
```

### 5.2 Full-Load Optimization (Single Buffer)

If the **complete matrix** fits in the buffer, no ping-pong needed → single buffer (1×):

```
buffer_factor = 1 if full_matrix_bytes ≤ cache_size else 2
```

Only use full-load when it improves performance (e.g., reduces kL1 loops).

### 5.3 stepK Derivation from L1

```python
per_step = baseM × baseK × dtype_a + baseK × N_per_core × dtype_b

# Try single buffer (all K fits in one L1 load):
single_buf_stepK = L1_size // per_step
if single_buf_stepK >= ceil(K / baseK):
    stepK = ceil(K / baseK)       # full load, single buffer
else:
    # Need multiple loads → double buffer:
    stepK = L1_size // 2 // per_step
```

**Example: MM3 BF16** — baseM=32, baseK=64, N_per_core=768

```
per_step = 32 × 64 × 2 + 64 × 768 × 2 = 4,096 + 98,304 = 102,400 B (100 KB)

Single buffer: 512 KB / 100 KB = 5 → but ceil(1536/64) = 24, need 24 steps
  5 < 24 → single buffer cannot cover all K → need double buffer

Double buffer: 256 KB / 100 KB = 2 → stepK = 2
kL1Loops = ceil(1536 / 128) = 12
```

**Example: MM4 BF16** — M=1, K=128, N_per_core=16

```
Full A = 1 × 128 × 2 = 256 B
Full B = 128 × 16 × 2 = 4,096 B
Full A+B = 4,352 B << 512 KB → single buffer, stepK = 1, kL1 = 1
```

---

## 6. Pipeline DAG (AIC/AIV Overlap)

### 6.1 Pipeline Structure

AIC (cube) and AIV (vector) execute in parallel with synchronization:

```
AIC: MM1 → signal(CQ) → MM2 → signal(CKVKR) → wait(RMSNORM_CQ) → MM3 → signal(QCQR) → [wait(DEQUANT)] → MM4
AIV: CopyIn → wait(CQ) → [AccumCq] → RmsNormCq → signal(RMSNORM_CQ)
     → wait(CKVKR) → [AccumCkvKr] → RmsNormCkvKr → RopeKr → Scatter
     → wait(QCQR) → [DequantQc] → signal(DEQUANT) → RopeQr → [DynQuantQn + MulQr]
```

Stages in `[brackets]` are optional (depend on quant mode or split-KN config). Accumulation stages (AccumCq, AccumCkvKr) from split-KN run on AIV and can overlap with the next AIC matmul.

### 6.2 Critical Path Analysis

Forward-pass longest-path through the DAG:

1. Topological sort all stages
2. `start[n] = max(end[dep] for dep in n.depends_on)`
3. `end[n] = start[n] + duration[n]`
4. Total = max(end[n] for all n)
5. Backtrack to find critical path

### 6.3 Worked Example: BF16 Decode BS=1

```
AIC stages:
  MM1_Cq:     [0.00 — 17.89]  17.89 us  ★ critical
  MM2_CkvKr:  [18.69 — 36.58] 17.89 us  ★ critical
  MM3_QcQr:   [38.18 — 85.37] 47.19 us  ★ critical
  MM4_Qn:     [86.97 — 87.63]  0.66 us  ★ critical

AIV stages:
  RmsNormCq:    [19.49 — 19.50] 0.015 us
  RmsNormCkvKr: [37.38 — 37.39] 0.005 us
  RopeKr:       [37.39 — 37.39] 0.001 us
  ScatterCkvKr: [37.39 — 37.39] 0.002 us
  RopeQr:       [86.17 — 86.28] 0.105 us

Pipeline total: 87.63 us
AIC busy: 83.63 us (95.4%)
AIV busy: 0.63 us (0.7%)
Sync overhead: 3.37 us

Critical path: MM1(17.89) → MM2(17.89) → MM3(47.19) → MM4(0.66)
```

MM3 dominates at 53.9% of pipeline time. AIV is almost entirely idle — AIC is the bottleneck.

### 6.4 With Split-KN Optimization

Applying MM1:N8×K4 + MM2:N3×K8:

```
AIC stages:
  MM1_Cq:      3.93 us (was 17.89)
  MM2_CkvKr:   0.78 us (was 17.89)
  MM3_QcQr:   47.19 us (unchanged)
  MM4_Qn:      0.66 us

AIV stages:
  AccumCq:     0.002 us  (overlaps with MM2 on AIC — hidden)
  AccumCkvKr:  0.001 us  (overlaps with MM3 on AIC — hidden)

Pipeline total: 53.31 us (was 87.63, -39.2%)
```

---

## 7. Optimization Decision Guide

### 7.1 If HBM-Bound (AI << Ridge)

| Technique | When | Impact |
|-----------|------|--------|
| INT8/FP8 quantization | Always for decode | ~2x (halve B matrix) |
| Split-KN (2D core split) | MM has small N (low core util) | Up to 80× for affected MM |
| L2 reuse (shared A) | Split-N with shared input | Already modeled |
| Increase batch (prefill) | T > 512 | Transition to cube-bound |

### 7.2 If Compute-Bound (AI >> Ridge)

| Technique | When | Impact |
|-----------|------|--------|
| Maximize baseM × baseK in L0A | Large T prefill | Fill MMAD pipeline |
| Reduce kL1 loops | Increase stepK | Less MTE1 overhead |
| FP8 (2× MMAD throughput) | If accuracy allows | 2× compute |

### 7.3 Key Insight: Total B Data is Fixed

For a given matmul, total B matrix bytes = `K × N × dtype_b` — constant regardless of core split. Split-KN doesn't reduce total HBM load for B; it only helps when the bottleneck is **per-core K or N** being too large relative to L1/L0 sizes or when more cores improve MTE parallelism on shared A.
