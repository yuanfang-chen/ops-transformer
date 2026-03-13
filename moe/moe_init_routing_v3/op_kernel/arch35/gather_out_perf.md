<!-- 1、ub利用率低（其实也是带宽比较低）
x的搬入的srcOffset是不连续的，但是搬出的dstOffset是连续的；可以搬入多行x，再一起搬出
2、提升带宽
开多路buffer，本质上也是提升带宽利用率，需要和1比较哪种方式更好
3、算法层面
scatter当然还是按索引逐个读取；gather模式，按源数据读取，一行源数据搬入后，可以搬出到多行expandedX上，减少了x搬入的数据量，搬入数据量从n*k*h降低为n*h -->

# moe_v3_gather_out.h 小 cols_ 场景性能优化分析

## 一、问题背景

`moe_v3_gather_out.h` 是 arch35 架构下 MoE Gather Out 操作的实现，用于根据排序后的行索引将输入数据 Gather 到输出位置。

**问题现象**：当 `cols_` 值较小时，性能表现不佳。

---

## 二、性能瓶颈分析

### 2.1 核心问题：数据搬运粒度小 + 流水线断裂

#### 问题1：GM 访问效率低（最关键）

```cpp
// Process() 235-241行
for (int64_t colsLoop = 0; colsLoop < colsLoops_; colsLoop++) {
    CopyXIn(xSrcOffset + colsLoopOffset, scaleSrcOffset, curLoopCols);
    CopyXOut(xDstOffset + colsLoopOffset, indicesIndex, curLoopCols);
}
```

`cols_` 小 → `curLoopCols` 小 → 每次搬运字节数少

- GM 访问有固定启动延迟（~几十 cycles）
- 实际数据传输时间可能小于启动延迟
- **带宽利用率极低**

#### 问题2：标量操作开销占比高

```cpp
for (int64_t indicesIndex = 0; indicesIndex < curLoopElements; indicesIndex++) {
    int64_t rowIdx = subRowIdxLocal.GetValue(indicesIndex);  // 标量读取
    int64_t xSrcOffset = rowIdx / k_ * cols_;                // 整数除法+乘法
}
```

每行都要做标量除法和乘法，当 `cols_` 小时，计算/搬运比过低。

#### 问题3：双缓冲失效

```cpp
pipe_->InitBuffer(xCopyInQueue_, GATHER_OUT_BUFFER_NUM, AlignBytes(perLoopCols_, sizeof(T)));
```

- 双缓冲需要 `MTE2`（搬运入）和 `MTE3`（搬出）并行
- 当前实现是**串行**的（先 In 后 Out，循环内无重叠）
- `cols_` 小 → UB 空间未充分利用 → 无法并行搬运多行

#### 问题4：同步开销累积

```cpp
SetWaitFlag<HardEvent::S_MTE2>(HardEvent::S_MTE2);
SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);
```

每个 `indicesIndex` 都有多次同步，小 `cols_` 场景下同步次数/数据量比过高。

### 2.2 性能对比

| 场景 | 单次搬运量 | GM 延迟占比 | 流水线效率 |
|------|-----------|------------|-----------|
| `cols_=4096` | 8KB (FP16) | ~10% | 高 |
| `cols_=64` | 128B (FP16) | ~80% | **极低** |

---

## 三、DMA 概念说明

### 3.1 什么是 DMA

**DMA = Direct Memory Access（直接内存访问）**

硬件自动搬运数据，不占用计算单元。

```
GM (全局内存)  ──DMA搬运──►  UB (片上内存)
   ~32GB                      ~1-2MB
   慢、大容量                  快、小容量
```

### 3.2 DMA 性能特点

| 特性 | 说明 |
|------|------|
| 启动延迟 | ~几十 cycles，固定开销 |
| 带宽 | 高（~1TB/s），但需要足够数据量才能跑满 |
| 粒度影响 | 小数据量时，延迟占比高，效率低 |

### 3.3 小 cols_ 性能差的原因

| 数据量 | DMA 启动延迟 | 实际传输时间 | 启动延迟占比 |
|--------|-------------|-------------|-------------|
| 8KB (cols_=4096) | ~50 cycles | ~200 cycles | 20% |
| 128B (cols_=64) | ~50 cycles | ~3 cycles | **94%** |

**结论**：小数据量时，DMA 启动延迟远大于实际传输时间，带宽利用率极低。

---

## 四、优化方案对比

### 4.1 方案A：多行缓存 UB + 批量写出

**核心思路**：预取多行数据到 UB，然后批量写出

```cpp
constexpr int64_t ROW_BATCH = 16;

void ProcessOptimized() {
    // 1. 批量读取 rowIdx
    LocalTensor<int32_t> rowIdxBatch = ...;
    
    // 2. 预取多行到 UB
    for (int i = 0; i < ROW_BATCH; i++) {
        int64_t rowIdx = rowIdxBatch.GetValue(i);
        CopyXInAsync(rowIdx * cols_, ubBuffer[i]);
    }
    WaitAllAsync();
    
    // 3. 批量写出（一次大 DMA）
    DataCopy(expandedXGm_[dstOffset], combinedBuffer, ROW_BATCH * cols_);
}
```

**核心优化点**：
1. 减少 DMA 次数：从 N 次小 DMA → N/BATCH 次大 DMA + 1 次大写出
2. 利用空闲 UB：小 cols_ 时 UB 空间充足
3. 异步搬运：多行预取可并行

### 4.2 方案B：增大 BufferNum 流水并行

**核心思路**：增加 Buffer 数量，让 In/Out 流水线更好地重叠

```cpp
// 原始：BufferNum = 2
// 优化：BufferNum = 4

constexpr int64_t GATHER_OUT_BUFFER_NUM = 4;  // 从 2 改为 4
```

**流水线效果**：

```
BufferNum = 2：
In:  [====]--------[====]
Out: ------[====]--------[====]
利用率: ~50%

BufferNum = 4：
In:  [====][====][====][====]
Out: ----[====][====][====][====]
利用率: ~75%
```

### 4.3 方案对比总结

| 维度 | 方案A：多行缓存UB | 方案B：增大BufferNum |
|------|------------------|---------------------|
| 核心思路 | 预取多行→UB缓存→批量写出 | 增加并行度→In/Out流水重叠 |
| UB空间需求 | 大（需缓存多行数据） | 小（只需2-4个buffer） |
| 代码改动 | 大（重构数据流） | 小（调整buffer数量） |
| DMA次数 | 大幅减少 | 不变，但流水并行 |
| 最佳场景 | cols_ ≤ 512 | cols_ > 512 |

---

## 五、BufferNum 上限分析

### 5.1 硬件限制

```
UB 总空间 ≈ 128KB（arch35 约 100KB 可用）

BufferNum × 单Buffer大小 ≤ UB可用空间
```

| cols_(FP16) | 单Buffer大小 | BufferNum上限 |
|-------------|-------------|--------------|
| 64 | 128B | ~800 |
| 256 | 512B | ~200 |
| 1024 | 2KB | ~50 |
| 4096 | 8KB | ~12 |
| 8192 | 16KB | ~6 |

### 5.2 收益递减规律

**关键约束**：只有 **2 个 DMA 引擎**

```
MTE2（读）和 MTE3（写）可以并行
但同类操作必须串行
```

| BufferNum | 利用率提升 | 推荐度 |
|-----------|-----------|-------|
| 2 | 基准 | 最小配置 |
| **3** | +16% | ⭐⭐⭐ 性价比最高 |
| **4** | +9% | ⭐⭐⭐⭐ 推荐 |
| 5 | +5% | ⭐⭐ 收益递减明显 |
| 6+ | +3%以内 | ⭐ 不推荐 |

### 5.3 为什么小 cols_ 时方案B收益有限

**流水并行只能让"传输时间"重叠，无法消除"启动延迟"**

| cols_ | 数据量 | 启动延迟 | 传输时间 | 启动占比 |
|-------|-------|---------|---------|---------|
| 64 | 128B | ~50 cycles | ~3 cycles | **94%** |
| 4096 | 8KB | ~50 cycles | ~200 cycles | 20% |

```
小 cols_ 场景（cols_=64）：

Buffer0: [启动50][传3]--------------[启动50][传3]
Buffer1: --------[启动50][传3]--------------[启动50][传3]
                     ↑
              启动延迟仍占94%时间，并行无法优化

大 cols_ 场景（cols_=4096）：

Buffer0: [启动50][====传输200====]--------[启动50][====传输200====]
Buffer1: ----------[启动50][====传输200====]--------[启动50][====传输200====]
                           ↑
                  传输时间长，并行收益明显
```

---

## 六、推荐方案

### 6.1 场景选择

```
┌─────────────────────────────────────────┐
│            cols_ 阈值判断               │
├─────────────────────────────────────────┤
│  cols_ ≤ 512   →  方案A：多行缓存UB     │
│  cols_ > 512   →  方案B：增大BufferNum  │
└─────────────────────────────────────────┘
```

### 6.2 组合方案（最优）

```cpp
if (cols_ * sizeof(T) <= 1024) {
    // 方案A：批量缓存 + 大粒度写出
    ProcessWithBatchCache();
} else {
    // 方案B：流水线并行
    ProcessWithPipeline();
}
```

### 6.3 总结

| 方案 | 最佳场景 | 核心收益 | 风险 |
|------|---------|---------|------|
| **A** | cols_ ≤ 512，UB空闲多 | 减少DMA次数，提升写出效率 | 需重构，UB管理复杂 |
| **B** | cols_ > 512，UB紧张 | 流水隐藏延迟，改动小 | 小cols_收益有限 |

**建议**：
1. 优先实现方案A（针对小 cols_ 场景），这是当前瓶颈的主要解法
2. BufferNum = 4 是最佳实践，超过 4 之后收益递减严重

---

## 七、附录：arch32 与 arch35 实现差异

arch32 版本有两种处理模式：

| 模式 | 策略 | arch35 使用 |
|------|------|------------|
| **GatherCopyOut** | 按源行读取，一次读入多次写出 | ❌ 未使用 |
| **ScatterCopyOut** | 按索引逐个处理，每次读一行写一行 | ✅ 当前实现 |

**GatherCopyOut 优化思路**：如果多个 `rowIdx` 指向同一源行，只读取一次。但源地址不连续的问题仍然存在，小 cols_ 场景下 GM 读取效率依然较低。

---

*文档生成时间：2026-03-13*