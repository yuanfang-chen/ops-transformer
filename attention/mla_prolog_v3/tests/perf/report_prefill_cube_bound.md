# Prefill Cube-Bound 优化分析

本报告分析如何通过 **K-outer 循环重排** 使 MLA Prolog V3 在 Prefill 场景下达到 **Cube bound**。所有数据基于 Ascend 950（32 AIC, 64 AIV, 1.65 GHz），BF16 模式，N=128 heads。

---

## 1. 当前瓶颈分析

### 1.1 现有模型的两个问题

**问题 1: 缺少 M 循环**

`estimate_matmul_detailed()` 仅对 `baseM` 行建模，未乘以 `m_loops = ceil(T / baseM)`。对于 T=1024, baseM=32，实际计算量是模型输出的 **32 倍**。

**问题 2: B 矩阵重复加载**

当前循环顺序（M-outer）在每个 M-block 的每个 kL1 迭代都从 HBM 重新加载 B：

```
M-outer (当前):
for m_block in m_loops:                    # M 在外层
  for kL1 in kL1_loops:                    # K 在内层
    load B[kL1StepSize, per_core_N] ← HBM  # ← 每次都从 HBM 加载！
    load A[baseM, kL1StepSize] ← HBM
    compute...
```

但 B 数据对于不同 M-block 是**完全相同**的！B 在首次加载后驻留 L2，后续应从 L2 读取。更优的方案是将 B **固定在 L1** 中。

### 1.2 MM3 的 B 矩阵分析

MM3: `Cq[T, 1536] × W[1536, 24576]`, 32 核, per_core_N = 768

```
B 每核总量 = 1536 × 768 × 2B = 2.36 MB
B 每 kL1 slab = kL1StepSize × per_core_N × 2B

stepK=2, baseK=64: kL1StepSize=128, B slab = 128 × 768 × 2 = 192 KB
L1 = 512 KB → B slab (192 KB) 可以留在 L1！
```

---

## 2. K-outer 循环重排

### 2.1 新循环顺序

将 K 提到外层，M 在内层。B 每个 kL1 slab 只加载一次，跨所有 M-block 复用：

```
K-outer (新):
for kL1 in kL1_loops:                      # K 在外层
  load B[kL1StepSize, per_core_N] ← HBM    # ← 仅加载一次，固定在 L1！
  for m_block in m_loops:                   # M 在内层
    load A[baseM, kL1StepSize] ← HBM       # ← 每 M-block 流式加载
    for stepK:
      L0A ← A[baseM, baseK]
      L0B ← B[baseK, per_core_N]    (B 已在 L1)
      MMAD: C += A × B
    FixPipe: 写回 C[baseM, per_core_N]
```

### 2.2 L1 容量验证

B slab 固定 (1×) + A slab 流式双缓冲 (2×)：

```
L1 需求 = B_slab + A_slab × 2
        = kL1StepSize × per_core_N × dtype_b + baseM × kL1StepSize × dtype_a × 2
```

| 配置 | B slab | A slab × 2 | 总计 | 是否 ≤ 512KB |
|------|--------|-----------|------|-------------|
| baseM=32 | 192 KB | 16 KB | 208 KB | ✓ |
| baseM=64 | 192 KB | 32 KB | 224 KB | ✓ |
| baseM=128 | 192 KB | 64 KB | 256 KB | ✓ |
| baseM=192 | 192 KB | 96 KB | 288 KB | ✓ |
| baseM=256 | 192 KB | 128 KB | 320 KB | ✓ |

所有 baseM ≤ 256 均满足 L1 约束。

### 2.3 时间模型

```python
# B 加载: 仅每 kL1 slab 一次
load_B_once = B_slab / hbm_bw

# 每 M-block:
load_A   = baseM × kL1StepSize × dtype_a / hbm_bw
inner    = stepK × max(l0a, l0b, mmad)
fixpipe  = baseM × per_core_N × dtype_c / fixpipe_bw
per_m    = max(load_A, inner, fixpipe)

# 每 kL1 slab: B 加载与 M-block 序列重叠
per_kl1 = max(load_B_once, m_loops × per_m)

# 总时间
total = kL1_loops × per_kl1
```

---

## 3. baseM 对 Cube Bound 的影响

### 3.1 瓶颈分析

每 M-block 内，瓶颈由 `per_m = max(load_A, inner, fixpipe)` 决定：

```
inner = stepK × max(l0a, l0b, mmad)
```

- `l0b = baseK × per_core_N × dtype_b / l0b_bw` — 与 baseM **无关**
- `mmad = baseM × per_core_N × baseK / mmad_tput` — 与 baseM **成正比**

当 `mmad > l0b` 时，MMAD 主导内层 → **Cube bound**！

### 3.2 Cube bound 阈值

令 `mmad ≥ l0b`：

```
baseM × per_core_N × baseK / mmad_tput ≥ baseK × per_core_N × dtype_b / l0b_bw
baseM ≥ dtype_b × mmad_tput / l0b_bw
baseM ≥ 2 × 6,758.4e9 / 422.4e9 = 32.0
```

> BF16 MM3: **baseM ≥ 32** 即可达到 Cube bound (l0b 与 mmad 非常接近)！

实际上更精确地说，在 baseM=32 时 `l0b=0.233us ≈ mmad=0.232us`，几乎持平。baseM=64 时 mmad 明确主导。

### 3.3 各 baseM 的内层瓶颈

MM3, per_core_N=768, baseK=64, BF16:

| baseM | l0a (us) | l0b (us) | mmad (us) | inner_iter | 瓶颈 |
|-------|---------|---------|----------|-----------|------|
| 32 | 0.010 | 0.233 | 0.232 | 0.233 | L0B (边界) |
| 64 | 0.019 | 0.233 | 0.464 | 0.464 | **CUBE** |
| 128 | 0.039 | 0.233 | 0.929 | 0.929 | **CUBE** |
| 192 | 0.058 | 0.233 | 1.393 | 1.393 | **CUBE** |
| 256 | 0.078 | 0.233 | 1.857 | 1.857 | **CUBE** |

---

## 4. MM3 详细计算 (T=1024, BF16, baseM=128)

```
参数:
  M=1024, K=1536, N=24576, 32 核, per_core_N=768
  baseM=128, baseK=64, stepK=2
  kL1StepSize=128, kL1_loops=12, m_loops=8

B slab = 128 × 768 × 2 = 196,608 B (192 KB) → 固定在 L1

带宽 (32 核):
  hbm_bw = 50.0 GB/s, l0a_bw = l0b_bw = 422.4 GB/s
  mmad_tput = 6,758.4 GOps/s, fixpipe_bw = 422.4 GB/s

B 加载 (每 kL1, 仅一次):
  load_B = 192 KB / 50.0 GB/s = 3.93 us

每 M-block:
  load_A  = 128 × 128 × 2 / 50.0e9 = 0.655 us
  l0a     = 128 × 64 × 2 / 422.4e9  = 0.039 us
  l0b     = 64 × 768 × 2 / 422.4e9  = 0.233 us
  mmad    = 128 × 768 × 64 / 6758.4e9 = 0.929 us  ★ CUBE BOUND
  inner   = 2 × 0.929 = 1.858 us
  fixpipe = 128 × 768 × 2 / 422.4e9 = 0.467 us
  per_m   = max(0.655, 1.858, 0.467) = 1.858 us    ★ MMAD 主导

每 kL1:
  per_kl1 = max(3.93, 8 × 1.858) = max(3.93, 14.86) = 14.86 us
  (B 加载被 M-block 序列完全遮盖)

总时间 = 12 × 14.86 = 178.3 us ★

理论峰值验证:
  有效计算 = 2 × 1024 × 24576 × 1536 = 77.3 TFLOPs
  峰值算力 (32 核) = 32 × 13,517 GFLOPS = 432.5 TFLOPS
  理论最短 = 77.3e12 / 432.5e12 = 178.7 us
  模型/理论比 = 178.3 / 178.7 = 99.8% ★★★
```

> **MM3 达到 99.8% 的理论峰值利用率！**

---

## 5. 全 Pipeline 分析

### 5.1 MM1/MM2: A-load 瓶颈

MM1/MM2 的 per_core_N 很小（64~72），MMAD 计算量不足以遮盖 A 加载。K-outer 对 MM1/MM2 的收益有限。

解决方案: 对 MM1/MM2 继续使用 **Split-KN**（2D N×K 切分），对 MM3 使用 **K-outer**。

### 5.2 最优组合

```
MM1: Split-KN (N1×K28)  → 16.70 us
MM2: Split-KN (N3×K8)   →  4.77 us
MM3: K-outer (baseM=64) → 178.73 us  ★ CUBE bound
MM4: K-outer             →  5.24 us

Vector: RmsNorm(0.47+0.16) + RoPE(0.03+3.35) + Scatter(0.23) + DequantQc(0) + ... = 15.61 us
```

### 5.3 Pipeline 总计

```
当前 (Split-N, 8 步): 88.29 × 8 = 706.32 us
K-outer + Split-KN:    214.77 us

加速比: 706.32 / 214.77 = 3.29×
节省: 491.54 us (69.6%)
Cube 利用率: 95.7%
```

---

## 6. 与当前方案对比

### 6.1 不同 T 的收益

| T | 当前 Split-N (us) | K-outer+KN (us) | 加速比 | MM3 瓶颈 |
|---|-------------------|-----------------|--------|---------|
| 4 | 87.64 × 1 = 87.6 | 71.62 | 1.22× | HBM_B |
| 128 | 88.29 × 1 = 88.3 | 87.65 | 1.01× | HBM_B |
| 256 | 88.29 × 2 = 176.6 | 115.54 | 1.53× | CUBE |
| 512 | 88.29 × 4 = 353.2 | 147.30 | 2.40× | CUBE |
| 1024 | 88.29 × 8 = 706.3 | 214.77 | 3.29× | CUBE |
| 2048 | 88.29 × 16 = 1412.6 | 393.76 | 3.59× | CUBE |

> **T ≥ 256 时 K-outer 显著优于多步 Split-N**。T 越大收益越高（B 加载固定，计算线性增长）。

### 6.2 为什么 K-outer 比 Split-M 好得多

| 方面 | Split-M | K-outer |
|------|---------|---------|
| N 轴切分 | 不切分（每核全量 N） | 照常切分（per_core_N = N/32） |
| B 每核加载量 | 全量 B (72 MB) | B/32 (2.36 MB) |
| B 来源 | L2 (~155 GB/s) | HBM (50 GB/s)，但仅加载一次/kL1 |
| B 总加载时间 | 72 MB / 155e9 = 464 us | 12 × 192KB / 50e9 = 46 us |
| 核间同步 | 无 | 有（与 Split-N 相同） |
| Cube bound | 需 baseM ≥ 88（困难） | 需 baseM ≥ 32（容易） |

K-outer 保留了 Split-N 的 N 轴切分（每核仅处理 768 列），因此 B 每核仅 2.36 MB，远小于 Split-M 的 72 MB。加上 B 在 L1 中固定复用，B 加载成本被彻底摊平。

---

## 7. 实现建议

### 7.1 内核修改

1. **循环重排**: 将 tiling 策略从 M-outer 改为 K-outer
2. **增大 baseM**: 从 32 提升到 64~128
3. **stepBatchSize**: 允许等于 T（单次调用处理所有 token）
4. **L1 分配**: B slab 固定区 + A slab 流式区

### 7.2 tiling 参数建议

MM3, BF16, Ascend 950:

```
baseM = 128    (最优, Cube bound, L0A=32KB ≤ 64KB)
baseN = 768    (= per_core_N, 不再核内分块)
baseK = 64     (标准)
stepK = 2      (由 L1 约束决定)
loop_order = "k_outer"
```

---

## 8. 工具命令

```bash
# Prefill 分析（显示 K-outer vs 多步 Split-N 对比）
python perf_analyzer.py --chip 950 --batch-size 256 --seq-len 4 --head-num 128 --mode prefill

# 不同 T 值
python perf_analyzer.py --chip 950 --batch-size 1024 --seq-len 1 --head-num 128 --mode prefill

# FP8 量化
python perf_analyzer.py --chip 950 --batch-size 256 --seq-len 4 --head-num 128 \
    --weight-quant-mode 3 --mode prefill

# 全部分析（包含 prefill）
python perf_analyzer.py --chip 950 --batch-size 512 --head-num 128 --mode full
```
