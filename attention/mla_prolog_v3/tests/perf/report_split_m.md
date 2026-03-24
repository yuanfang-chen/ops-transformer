# Split-M 性能建模报告

本报告详细分析 Split-M（M 轴切分）在 Ascend 950 上的理论性能建模方法，并与 Split-N（N 轴切分）进行对比。所有数据基于 MLA Prolog V3 算子在 Ascend 950（32 AIC, 64 AIV, 1.65 GHz）上的建模结果。

---

## 目录

1. [概述与结论](#1-概述与结论)
2. [Split-M 切分原理](#2-split-m-切分原理)
3. [3 级循环建模](#3-3-级循环建模)
4. [B 矩阵 L2 复用模型](#4-b-矩阵-l2-复用模型)
5. [各阶段计算示例](#5-各阶段计算示例)
6. [Vector 操作建模](#6-vector-操作建模)
7. [Pipeline 流水分析](#7-pipeline-流水分析)
8. [baseM 优化分析](#8-basem-优化分析)
9. [场景对比分析](#9-场景对比分析)
10. [工具使用](#10-工具使用)

---

## 1. 概述与结论

### 1.1 动机

当前 Split-N 策略将 N 轴切分到各核，适用于 Decode（M=1）。但对于 Prefill（M=T=BS×SeqLen 很大），每步仅处理 `stepBatchSize` 个 token，需多次内核调用：

```
Split-N 总时间 = ceil(T / stepBatchSize) × 单步 pipeline 时间
```

Split-M 将 M 轴切到各核，每核处理 `ceil(T / m_groups)` 个 token 和完整 N 轴，单次内核调用处理所有 token：

```
Split-M 总时间 = 1 × pipeline 时间（所有 T token 并行处理）
```

### 1.2 关键结论

| 场景 | Split-N | Split-M (最优) | 收益 |
|------|---------|---------------|------|
| N=128, T=512, BF16 | 353.16 us | 783.11 us | **-122%** (Split-N 优) |
| N=8, T=1024, BF16 | 351.40 us | 249.73 us | **+28.9%** (Split-M 优) |
| N=8, T=1024, FP8 | 250.29 us | 131.75 us | **+47.4%** (Split-M 优) |

**结论**: Split-M 在 **N 较小且 T 较大** 时有明显优势。当 N 较大（如 128 heads）时，B 矩阵过大（72 MB），每核从 L2 加载 B 全量的开销远超 Split-N 的分片 HBM 加载。

---

## 2. Split-M 切分原理

### 2.1 核间分工对比

```
Split-N (当前):                       Split-M (新):
┌────────────────────────┐            ┌────────────────────────┐
│  Core 0: M 全量, N/32  │            │  Core 0: M/32, N 全量  │
│  Core 1: M 全量, N/32  │            │  Core 1: M/32, N 全量  │
│  ...                   │            │  ...                   │
│  Core 31: M 全量, N/32 │            │  Core 31: M/32, N 全量 │
└────────────────────────┘            └────────────────────────┘
  ↓ 核间需同步（合并 N 片段）           ↓ 核间无需同步（独立 M 行）
```

### 2.2 数据访问模式

| 数据 | Split-N | Split-M |
|------|---------|---------|
| **A (激活)** | 所有核共享，L2 复用 | 每核独立行，从 HBM 加载 |
| **B (权重)** | 每核加载 N/cores 片段，从 HBM | 所有核共享全量，L2 复用 |
| **输出 C** | 每核写 N/cores 列 | 每核写 M/cores 行 |
| **核间同步** | 需要（合并 N 分片后才能继续下一阶段） | 不需要 |

### 2.3 Split-M 优势条件

Split-M 通过 L2 复用 B 矩阵节省带宽，但每核需加载 B 全量而非 B/cores：

```
Split-N 每核 B 加载量 = K × N / cores × sizeof(B)     → 从 HBM, 50 GB/s
Split-M 每核 B 加载量 = K × N × sizeof(B)             → 从 L2, ~155 GB/s
```

Split-M 的 B 加载量是 Split-N 的 `cores` 倍，但带宽快 ~3.1 倍。因此：

```
Split-M 更快当: cores / 3.1 < 1  →  cores < 3.1
```

这在 32 核场景下**天然不成立**。Split-M 的优势来自其他方面：
1. **消除多步调用开销**: 单次调用处理所有 T token
2. **消除核间同步**: 节省 sync_us × stages 开销
3. **N 较小时 B 矩阵可控**: 当 N 较小，B 全量从 L2 加载时间可接受

---

## 3. 3 级循环建模

### 3.1 循环结构

Split-M 的每核 matmul 为 `[M_per_core, K] × [K, N_full]`，由于 N 轴不切分，需核内分块：

```
Level 1 — M 循环 (m_loops = ceil(M_per_core / baseM)):
  Level 2 — K 循环 (kL1_loops = ceil(K / kL1StepSize)):
    加载 A[baseM, kL1StepSize] → L1              [从 HBM, 每核独立]
    Level 3 — N 循环 (n_blocks = ceil(N / baseN)):
      加载 B[kL1StepSize, baseN] → L1            [从 L2, 共享权重]
      内层 stepK 循环:
        L0A ← A[baseM, baseK]
        L0B ← B[baseK, baseN]    (转置)
        MMAD: C[baseM, baseN] += A × B
      FixPipe: 写回 C[baseM, baseN]
```

对比 Split-N 的 2 级循环（kL1 × stepK），Split-M 多了 **N 循环**——这是因为每核处理完整 N，L0B 无法一次装下全部 N。

### 3.2 时间模型

各级流水重叠后的时间公式：

```python
# --- 第 3 级: 每个 N-block (最内层，双缓冲) ---
load_B_nb = kL1StepSize × baseN × sizeof(B) / effective_b_bw    # B 从 L2
l0a      = baseM × baseK × sizeof(A) / l0a_bw                   # L1→L0A
l0b      = baseK × baseN × sizeof(B) / l0b_bw                   # L1→L0B
mmad     = baseM × baseN × baseK / mmad_tput                     # MMAD 计算
inner    = stepK × max(l0a, l0b, mmad)                           # 内层流水
fixpipe  = baseM × baseN × sizeof(C) / fixpipe_bw               # 输出写回

per_nb = max(load_B_nb, inner, fixpipe)                          # N-block 瓶颈

# --- 第 2 级: 每个 kL1-slab (双缓冲跨 kL1 迭代) ---
load_A = baseM × kL1StepSize × sizeof(A) / hbm_bw               # A 从 HBM
per_kl1 = max(load_A, n_blocks × per_nb)                         # A 加载与 N-blocks 重叠

# --- 第 1 级: M 分块 ---
total = m_loops × kL1_loops × per_kl1
```

### 3.3 与 Split-N 模型的关键差异

| 维度 | Split-N | Split-M |
|------|---------|---------|
| per_core_N | ceil(N / cores) | N (全量) |
| N 循环 | 无 (per_core_N 装入 L0B) | 有 (n_blocks = ceil(N/baseN)) |
| B 带宽 | HBM: 50 GB/s | L2: ~155 GB/s |
| A 带宽 | L2 (共享) / HBM | HBM (每核独立行) |
| M 循环 | 1 (M ≤ baseM for decode) | m_loops = ceil(M_per_core / baseM) |

---

## 4. B 矩阵 L2 复用模型

### 4.1 复用原理

Split-M 中所有核读相同的 B（权重）矩阵。第一个核从 HBM 加载 B 到 L2，其余核从 L2 读取：

```
核 0:  HBM → L2 → L1 → L0B    (冷启动, HBM 带宽)
核 1:        L2 → L1 → L0B    (L2 命中, L2 带宽)
核 2:        L2 → L1 → L0B    (L2 命中, L2 带宽)
...
核 m-1:      L2 → L1 → L0B    (L2 命中, L2 带宽)
```

### 4.2 有效带宽计算

使用调和平均模型（`hw.effective_load_bw()`）：

```
effective_bw = 1 / (1/m × 1/hbm_bw + (m-1)/m × 1/l2_bw)
```

其中 `m = m_groups`（核数），`hbm_bw` 和 `l2_bw` 为 per-core 带宽。

### 4.3 示例 (Ascend 950, BF16)

| m_groups | HBM per core | L2 per core | effective_bw | vs 纯 HBM |
|----------|-------------|-------------|-------------|----------|
| 4 | 54.9 GB/s | 178.2 GB/s | 114.1 GB/s | 2.3× |
| 8 | 54.9 GB/s | 178.2 GB/s | 139.1 GB/s | 2.8× |
| 16 | 50.0 GB/s | 162.5 GB/s | 148.4 GB/s | 3.0× |
| 32 | 50.0 GB/s | 162.5 GB/s | 155.1 GB/s | 3.1× |

> 注: 当 m_groups ≥ 10 时，effective_bw 接近 L2 带宽（~95% 命中率）。

### 4.4 L2 容量验证

L2 Cache 约 48 MB。需验证每个 kL1 slab 的 B 数据可装入 L2：

| Matmul | kL1StepSize | N | sizeof(B) | B slab size | 是否装入 L2 |
|--------|------------|---|----------|------------|-----------|
| MM1 (Hcq=1536) | 128 | 1536 | 2B | 0.4 MB | ✓ |
| MM2 (Hckv+Dr=576) | 128 | 576 | 2B | 0.1 MB | ✓ |
| MM3 (N=128, D+Dr=192) | 128 | 24576 | 2B | 6.3 MB | ✓ |
| MM3 (N=8, D+Dr=192) | 128 | 1536 | 2B | 0.4 MB | ✓ |
| MM4 (Hckv=512) | 128 | 512 | 2B | 0.1 MB | ✓ |

所有 slab 均可装入 L2，L2 复用模型有效。

---

## 5. 各阶段计算示例

### 5.1 场景 A: N=128, T=512, BF16, m_groups=16

每核 M_per_core = ceil(512/16) = 32。

#### MM3 (QcQr): `[32, 1536] × [1536, 24576]`

```
配置: baseM=32, baseN=256, baseK=64

stepK 推导:
  per_step = 32×64×2 + 64×256×2 = 4,096 + 32,768 = 36,864 B
  单缓冲: 512KB / 36,864 = 14 → k_iters = ceil(1536/64) = 24, 14 < 24 → 需双缓冲
  双缓冲: 256KB / 36,864 = 7 → stepK = 7
  kL1StepSize = 64 × 7 = 448
  kL1_loops = ceil(1536 / 448) = 4

维度:
  m_loops = ceil(32 / 32) = 1
  n_blocks = ceil(24576 / 256) = 96

带宽 (16 核):
  hbm_bw = min(54.9, 1600/16) = 50.0 GB/s        (A 加载)
  effective_b_bw = 148.4 GB/s                       (B 从 L2)
  l0a_bw = 422.4 GB/s, l0b_bw = 422.4 GB/s
  mmad_tput = 6,758.4 GOps/s

每 N-block:
  load_B  = 448 × 256 × 2 / 148.4e9 = 1.545 us    ← L2 加载 B
  l0a     = 32 × 64 × 2 / 422.4e9   = 0.010 us
  l0b     = 64 × 256 × 2 / 422.4e9  = 0.078 us
  mmad    = 32 × 256 × 64 / 6758.4e9 = 0.077 us
  inner   = 7 × max(0.010, 0.078, 0.077) = 7 × 0.078 = 0.546 us
  fixpipe = 32 × 256 × 2 / 422.4e9  = 0.039 us
  per_nb  = max(1.545, 0.546, 0.039) = 1.545 us    ← L2 加载 B 主导！

每 kL1 slab:
  load_A  = 32 × 448 × 2 / 50.0e9 = 0.573 us
  per_kl1 = max(0.573, 96 × 1.545) = max(0.573, 148.3) = 148.3 us

总时间 = 1 × 4 × 148.3 = 593.2 us
```

> 实际模型输出: **563.58 us**（差异来自 stepK 边界条件和最后一个 kL1 的部分填充）。
>
> **瓶颈: L2**（B 矩阵 L2 加载占 96.4% 时间）

#### MM1 (Cq): `[32, 7168] × [7168, 1536]`

```
N = Hcq = 1536, n_blocks = ceil(1536/256) = 6
B slab per kL1 = 448 × 256 × 2 = 229,376 B

per_nb = max(229376/148.4e9, inner, fixpipe) = max(1.545, 0.546, 0.039) = 1.545 us
per_kl1 = max(load_A, 6 × 1.545) = max(0.573, 9.27) = 9.27 us
kL1_loops = ceil(7168/448) = 16

总时间 = 1 × 16 × 9.27 = 148.3 us
```

> 实际模型输出: **150.96 us**, 瓶颈: L2

#### MM2 (CkvKr): `[32, 7168] × [7168, 576]`

```
N = Hckv + Dr = 576, n_blocks = ceil(576/256) = 3

per_kl1 = max(0.573, 3 × 1.545) = max(0.573, 4.64) = 4.64 us
总时间 = 1 × 16 × 4.64 = 74.2 us
```

> 实际模型输出: **62.90 us**, 瓶颈: L2（A 从 L2 复用进一步降低）

#### 总 Pipeline

```
AIC: MM1(150.96) + MM2(62.90) + MM3(563.58) + MM4(0.84) = 778.28 us
AIV: RmsNormCq(0.24) + RmsNormCkvKr(0.08) + RopeKr(0.01) + Scatter(0.06) + RopeQr(1.68) = 2.06 us
Sync: 4.83 us

Pipeline 总计: 783.11 us
```

#### 与 Split-N 对比

```
Split-N: 88.29 us/step × 4 steps = 353.16 us
Split-M: 783.11 us (单次调用)

Split-M 慢 122% → 结论: N=128 时 Split-N 更优
原因: MM3 的 B 矩阵 (1536 × 24576 × 2 = 72 MB) 每核从 L2 全量加载太慢
```

---

### 5.2 场景 B: N=8, T=1024, BF16, m_groups=16

每核 M_per_core = ceil(1024/16) = 64。

#### MM3 (QcQr): `[64, 1536] × [1536, 1536]`

N = 8 × (128+64) = 1536（大幅减小）

```
baseM=64, baseN=256, baseK=64, stepK=7
n_blocks = ceil(1536/256) = 6     ← 仅 6 个 N-block（vs N=128 时 96 个）

每 N-block:
  load_B  = 448 × 256 × 2 / 148.4e9 = 1.545 us
  mmad    = 64 × 256 × 64 / 6758.4e9 = 0.154 us
  inner   = 7 × 0.154 = 1.078 us
  per_nb  = max(1.545, 1.078) = 1.545 us

per_kl1 = max(load_A, 6 × 1.545) = max(1.15, 9.27) = 9.27 us
kL1_loops = 4

总时间 = 1 × 4 × 9.27 = 37.1 us
```

> 实际模型输出: **30.19 us**, 瓶颈: L2

#### 总 Pipeline

```
AIC: MM1(150.96) + MM2(62.90) + MM3(30.19) + MM4(1.68) = 245.73 us
AIV: 0.98 us
Sync: 3.02 us

Pipeline 总计: 249.73 us
```

#### 与 Split-N 对比

```
Split-N: 43.92 us/step × 8 steps = 351.40 us
Split-M: 249.73 us (单次调用)

Split-M 快 28.9% → 结论: N=8 时 Split-M 更优
原因: MM3 的 B 矩阵仅 1536×1536×2 = 4.5 MB，L2 加载开销可控
```

---

### 5.3 场景 C: N=8, T=1024, FP8, m_groups=32

每核 M_per_core = ceil(1024/32) = 32。

```
FP8 量化: sizeof(B) = 1B (vs BF16 的 2B)
→ B 矩阵减半，L2 加载时间减半
→ mmad_tput = 13,516.8 GOps/s (2× BF16)

MM3: B = 1536 × 1536 × 1 = 2.3 MB → 仅 BF16 的一半
总 Pipeline: 131.75 us (vs Split-N 250.29 us)

Split-M 快 47.4%
```

---

## 6. Vector 操作建模

### 6.1 核分配差异

| 模式 | 每 AIC 核分配 AIV 核数 | 总 Vector 并行度 |
|------|---------------------|----------------|
| Split-N | 64 (全部) | 64 核处理全量 T×N |
| Split-M | 64/32 = 2 | 2 核处理 M_per_core × N |

### 6.2 各操作时间估算 (场景 A: N=128, M_per_core=32, 2 vec cores)

| 操作 | 加载 (B) | 计算 (ops) | 写回 (B) | 时间 (us) | 瓶颈 |
|------|---------|-----------|---------|----------|------|
| RmsNormCq | 32×1536×4 = 197K | 32×1536×3 = 147K | 32×1536×2 = 98K | 0.236 | VECTOR |
| RmsNormCkvKr | 32×576×4 = 74K | 32×512×3 = 49K | 32×512×2 = 33K | 0.079 | VECTOR |
| RopeKr | 32×64×4 = 8K | 32×64×4 = 8K | 32×64×2 = 4K | 0.013 | VECTOR |
| ScatterCkvKr | 32×576×2 = 37K | 32×576 = 18K | 37K (散列) | 0.058 | MTE3 |
| **RopeQr** | **32×128×64×4 = 1M** | **32×128×64×4 = 1M** | **32×128×64×2 = 524K** | **1.678** | **VECTOR** |

> **RopeQr** 是 Split-M 最大的 Vector 操作（1.678 us），但仍远小于 AIC 的 778.28 us。
>
> Vector 总计 2.06 us / Pipeline 总计 783.11 us = **0.3%**，完全非瓶颈。

### 6.3 Vector 压力上限

Vector 是否可能成为瓶颈？关键在于 DequantQc（量化模式下出现）和 RopeQr：

```
RopeQr 时间 = M_per_core × N × Dr × 4 / (2 × vec_throughput_per_core)

对于 M_per_core = 256, N = 128:
  compute = 256 × 128 × 64 × 4 = 8,388,608 ops
  vec_time = 8,388,608 / 2 / (20e12/64) = 13.4 us
```

即使 M_per_core = 256，RopeQr 仍仅 13.4 us，远小于 Cube 时间（通常 >100 us）。**结论: Vector 在 Split-M 下永远不是瓶颈。**

---

## 7. Pipeline 流水分析

### 7.1 Split-M 无核间同步

Split-N 的 Pipeline 需要核间同步（各核合并 N 分片后才能进入下一阶段）：

```
Split-N Pipeline:
  MM1 → [全核同步] → signal_AIV → MM2 → [全核同步] → signal_AIV → ...
```

Split-M 每核独立处理完整 N，无需等待其他核：

```
Split-M Pipeline (每核独立):
  MM1 → signal_AIV → MM2 → signal_AIV → wait(RmsNorm) → MM3 → signal_AIV → MM4
  AIV: wait(MM1) → RmsNorm → signal → wait(MM2) → RmsNorm+RoPE+Scatter → ...
```

### 7.2 Pipeline DAG

Split-M 复用 Split-N 的 DAG 结构（相同的 AIC/AIV 信号机制），但 sync 开销更小（核内信号 vs 核间屏障）。

```
AIC: MM1(150.96) → sync(0.8) → MM2(62.90) → sync(0.8) → wait(RmsNorm)
     → MM3(563.58) → sync(0.8) → MM4(0.84)

AIV: CopyIn(0.5) → wait(MM1) → RmsNormCq(0.24) → signal → wait(MM2)
     → RmsNormCkvKr+RopeKr+Scatter(0.15) → wait(MM3) → RopeQr(1.68)

关键路径: MM1 → MM2 → MM3 → MM4 (AIC 主导)
Pipeline 总计: 783.11 us
AIC 利用率: 778.28 / 783.11 = 99.4%
```

### 7.3 时间构成

场景 A (N=128, T=512, Split-M m_groups=16):

```
┌────────────────────────────────────────────────────────────────┐
│ MM1     │ MM2   │          MM3                    │MM4│        │
│ 150.96  │ 62.90 │         563.58                  │0.8│ =783us │
│ 19.3%   │ 8.0%  │         72.0%                   │0.1│        │
└────────────────────────────────────────────────────────────────┘
  ↑ L2 bound  ↑ L2 bound    ↑ L2 bound (B 矩阵 72MB 全量加载)
```

场景 B (N=8, T=1024, Split-M m_groups=16):

```
┌──────────────────────────────────────────┐
│ MM1      │ MM2    │ MM3  │MM4│           │
│ 150.96   │ 62.90  │30.19 │1.7│ = 250 us  │
│ 60.4%    │ 25.2%  │12.1% │0.7│           │
└──────────────────────────────────────────┘
  ↑ L2 bound ↑ L2 bound ↑ L2 bound (B 矩阵 4.5MB，可控)
```

---

## 8. baseM 优化分析

### 8.1 为什么需要大 baseM

在每个 N-block 内，时间由三个阶段的最大值决定：

```
per_nb = max(load_B_nb, stepK × max(l0a, l0b, mmad), fixpipe)
```

当 `load_B_nb > inner` 时为 **L2 bound**；当 `inner > load_B_nb` 时为 **Cube bound**。

`inner` 中的 `mmad = baseM × baseN × baseK / mmad_tput` 与 `baseM` 成正比。增大 `baseM` 可使计算时间超过 B 加载时间，实现 **Cube bound**。

### 8.2 Cube bound 的 baseM 阈值

令 `total_inner ≥ load_B_nb`：

```
stepK × baseM × baseN × baseK / mmad_tput ≥ kL1StepSize × baseN × sizeof(B) / b_bw
```

化简（消去 `baseN`, `kL1StepSize = baseK × stepK`）：

```
baseM ≥ sizeof(B) × mmad_tput / b_bw
```

| 数据类型 | sizeof(B) | mmad_tput (GOps/s) | b_bw (GB/s) | baseM 阈值 |
|---------|----------|-------------------|------------|----------|
| BF16 | 2 | 6,758.4 | 155.1 | **87.1** |
| FP8 | 1 | 13,516.8 | 155.1 | **87.1** |
| INT8 | 1 | 6,758.4 | 155.1 | **43.6** |

> BF16 需要 `baseM ≥ 88` 才能达到 Cube bound。

### 8.3 Cache 约束

baseM 受 L0A 和 L0C 大小限制：

```
L0A: baseM × baseK × sizeof(A) × 2 (双缓冲) ≤ 64 KB
L0C: baseM × baseN × sizeof(C) × 2 (双缓冲) ≤ 256 KB
```

| baseM | baseK=64, BF16 L0A | baseN=256, BF16 L0C | 是否合规 |
|-------|-------------------|-------------------|---------|
| 32 | 8 KB | 32 KB | ✓ |
| 64 | 16 KB | 64 KB | ✓ |
| 128 | 32 KB | 128 KB | ✓ |
| 192 | 48 KB | 192 KB | ✓ |
| 256 | 64 KB | 256 KB | ✓ (刚好) |

所有 baseM ≤ 256 均满足 Cache 约束。

### 8.4 baseM 与 M_per_core 的矛盾

`baseM` 不能超过 `M_per_core`（否则产生无效计算）：

```
baseM ≤ M_per_core = ceil(T / m_groups)
```

| m_groups | T=512 M_per_core | T=1024 M_per_core | 最大 baseM |
|----------|-----------------|------------------|----------|
| 32 | 16 | 32 | 32 (< 88, 无法 Cube bound) |
| 16 | 32 | 64 | 64 (< 88, 无法 Cube bound) |
| 8 | 64 | 128 | 128 (**≥ 88, 可 Cube bound**) |
| 4 | 128 | 256 | 256 (≥ 88, Cube bound) |

> **关键限制**: 要达到 BF16 Cube bound (baseM≥88)，需 m_groups ≤ 8 (T=1024) 或 m_groups ≤ 4 (T=512)。
> 但核数越少，总并行度越低，核间 B 数据重复加载反而增加。这是 Split-M 的**核心矛盾**。

### 8.5 搜索结果验证

工具自动搜索 `m_groups × baseM` 网格，选择每个 m_groups 下的最优 baseM：

| 配置 | M_per_core | baseM | 瓶颈 | MM3 (us) | Pipeline (us) |
|------|-----------|-------|------|---------|-------------|
| M16_bm32 | 32 | 32 | L2 | 563.58 | 783.11 |
| M8_bm64 | 64 | 64 | L2 | 542.47 | 789.98 |
| M4_bm128 | 128 | 128 | **CUBE** | 744.73 | 1050.85 |
| M2_bm256 | 256 | 256 | **CUBE** | 1429.88 | 2190.11 |

M4/M2 虽然达到 Cube bound，但核数太少导致总时间反而更长。

---

## 9. 场景对比分析

### 9.1 总吞吐量对比

Split-N 需多步调用处理所有 T token，Split-M 单次处理：

```
Split-N 总时间 = ceil(T / stepBatchSize) × pipeline_per_step
Split-M 总时间 = pipeline_us (单次)
```

### 9.2 N=128 heads (DeepSeek-V3 标准)

| 配置 | T | Split-N 总时间 | Split-M 最优 | 结论 |
|------|---|-------------|------------|------|
| BF16 | 512 | 353.16 us | 783.11 us (M16) | Split-N 快 2.2× |
| BF16 | 1024 | 706.32 us | 783.11 us (M16) | Split-N 快 1.1× |
| BF16 | 4096 | 2825.28 us | 3211.87 us (M32) | Split-N 快 1.1× |

> N=128 时 **Split-N 始终更优**。原因: MM3 的 B 矩阵 72 MB 每核从 L2 全量加载的代价始终大于 Split-N 的分片 HBM 加载。

### 9.3 N=8 heads (小模型或 GQA 场景)

| 配置 | T | Split-N 总时间 | Split-M 最优 | 收益 |
|------|---|-------------|------------|------|
| BF16 | 1024 | 351.40 us | 249.73 us (M16) | **+28.9%** |
| FP8 | 1024 | 250.29 us | 131.75 us (M32) | **+47.4%** |

> N=8 时 **Split-M 明显更优**。原因: MM3 的 B 矩阵仅 4.5 MB，L2 加载开销可控；消除多步调用和核间同步开销。

### 9.4 Split-M 适用条件

```
Split-M 更优当: B_total / L2_bw × m_groups < B_total / cores / HBM_bw × ceil(T/stepBS)
```

简化为:

```
N 较小（B 矩阵可控）且 T 足够大（多步调用开销显著）
```

经验阈值: **N × (D + Dr) < ~2000** 且 **T > 256**。

---

## 10. 工具使用

### 10.1 基本命令

```bash
# BF16, N=128 heads
python perf_analyzer.py --chip 950 --batch-size 128 --seq-len 4 --head-num 128 --mode split-m

# BF16, N=8 heads
python perf_analyzer.py --chip 950 --batch-size 1024 --seq-len 1 --head-num 8 --mode split-m

# FP8 量化
python perf_analyzer.py --chip 950 --batch-size 256 --seq-len 4 --head-num 8 \
    --weight-quant-mode 3 --mode split-m

# 全部分析（包含 split-m）
python perf_analyzer.py --chip 950 --batch-size 512 --head-num 8 --mode full
```

### 10.2 输出解读

```
  Rank Config    M/core |  Total(us)    MM1     MM2     MM3     MM4     Vec | CubeUtil vs SplitN
  ─────────────────────────────────────────────────────────────────────────────────────────────────
     1 M16_bm64      64 |    249.73  150.96   62.90   30.19    1.68   0.98 |   98.4%  +101.67 <--
```

- **Config**: `M{m_groups}_bm{baseM}` — 核数和块高
- **M/core**: 每核处理的 token 数
- **Total(us)**: 单次内核调用总时间（处理所有 T token）
- **MM1-MM4**: 各 Matmul 的 Cube 时间
- **Vec**: 该核上所有 Vector 操作总时间
- **CubeUtil**: AIC 忙碌时间 / Pipeline 总时间
- **vs SplitN**: 正值表示 Split-M 更快（节省的 us 数）
