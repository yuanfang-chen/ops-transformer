# 案例：quant_lightning_indexer 算子 N1 泛化导致的 Buffer 越界精度问题

## 1. 问题背景

| 项目 | 说明 |
|------|------|
| **算子名称** | quant_lightning_indexer |
| **目标平台** | ascend950 (arch35) |
| **任务类型** | N1 维度泛化（从 N1=24 扩展到 N1=16/32/48/64） |
| **问题现象** | N1=24、48 时精度正确；N1=16 时精度异常 |
| **初步怀疑** | Cube 侧（矩阵乘法）计算问题 |

## 2. 算子架构概述

该算子采用 **MIX_AIC_1_2** 混合执行模式：1 个 AIC（Cube 核）+ 2 个 AIV（Vector 核）。

### 核心流程

```
AIC (Cube核):
  GM → L1 (MTE2) → L0A/L0B (MTE1) → Mmad → L0C → Fixpipe → UB(AIV)

AIV (Vector核):
  UB → MulWeight → ReduceSum → TopK → GM
```

### 关键参数关系

```
N1 = gSize × N2    (N2=1, 所以 gSize = N1)
mBaseSize = S1_BASE_SIZE × gSize = 4 × N1
M_BASIC_BLOCK = 96  (硬编码常量，为 N1=24 时的 mBaseSize)
```

| N1 | gSize | mBaseSize | M_BASIC_BLOCK | 关系 |
|----|-------|-----------|---------------|------|
| 16 | 16 | 64 | 96 | mBaseSize < M_BASIC_BLOCK |
| 24 | 24 | 96 | 96 | mBaseSize = M_BASIC_BLOCK |
| 48 | 48 | 192 | 96 | mBaseSize > M_BASIC_BLOCK |

## 3. 根因分析

### 3.1 问题定位路径

**文件**: `op_kernel/arch35/quant_lightning_indexer_service_cube.h`

该文件中 `QLIMatmul` 类使用了双缓冲（double-buffering）技术管理 L1、L0A、L0C 等 buffer。双缓冲通过 `bufIdx % BUF_NUM` 选择 slot，并用固定偏移量定位每个 slot 的起始地址。

### 3.2 硬编码的偏移量常量

```cpp
static constexpr uint64_t M_BASIC_BLOCK = 96;        // 硬编码，仅适配 N1=24
static constexpr uint64_t D_BASIC_BLOCK = 128;
static constexpr uint64_t S2_BASIC_BLOCK_L0 = 128;

// 以下偏移量均基于 M_BASIC_BLOCK=96 计算
static constexpr uint64_t QUERY_BUFFER_OFFSET = M_BASIC_BLOCK * D_BASIC_BLOCK;           // 96×128 = 12288
static constexpr uint64_t L0AB_BUFFER_OFFSET  = M_BASIC_BLOCK_L0 * D_BASIC_BLOCK_L0;     // 96×128 = 12288
static constexpr uint64_t L0C_BUFFER_OFFSET   = M_BASIC_BLOCK_L0 * S2_BASIC_BLOCK_L0;    // 96×128 = 12288
```

### 3.3 Buffer 分配使用动态 mBaseSize

```cpp
// 分配大小正确使用了动态的 mBaseSize
pipe->InitBuffer(bufQL1_, QUERY_BUF_NUM * constInfo_.mBaseSize * D_BASIC_BLOCK * sizeof(Q_T));
//                        ^^^^^^^^^^^^^^^^^ 动态值
```

### 3.4 Buffer 索引使用硬编码偏移量

```cpp
// 索引时使用了硬编码的 QUERY_BUFFER_OFFSET
DataCopy(queryL1_[(queryL1Mte2BufIdx_ % QUERY_BUF_NUM) * QUERY_BUFFER_OFFSET], ...);
//                                                        ^^^^^^^^^^^^^^^^^^^^ 硬编码值 12288
```

### 3.5 越界分析（N1=16 为例）

```
Query L1 Buffer 内存布局:

分配大小 = 2 × 64 × 128 = 16,384 字节
                                  ^^^^
Slot 0: [0, 8192)      ← bufIdx=0 → offset = 0 × 12288 = 0       ✓ 正确
Slot 1: [12288, 20480)  ← bufIdx=1 → offset = 1 × 12288 = 12288  ✗ 越界!
         ^^^^^^^^^^^^^
         实际有效范围应为 [8192, 16384)

越界量 = 20480 - 16384 = 4096 字节
```

**内存布局示意图**:

```
地址:     0        8192     12288    16384    20480
          |  Slot0  |        | Slot1  |        |
          |  (正确)  |        | (越界!) |        |
          |<-alloc->|<-alloc->|
          |  Query L1 Buffer  | Key L1 Buffer  |
                              ^                ^
                              |  被Slot1覆盖    |
                              |  导致Key数据损坏 |
```

### 3.6 问题传播链

```
1. Query L1 Slot1 越界 → 写入 Key L1 Buffer 区域
2. Key L1 数据被部分覆盖/损坏
3. 损坏的 Key 数据 Load 到 L0B
4. Mmad(L0A × L0B) 使用了错误的 Key 数据
5. L0C 结果错误 → Fixpipe 输出到 UB 的数据错误
6. Vector 侧 TopK 基于错误数据选择 → 最终精度异常
```

### 3.7 为什么 N1=24 和 N1=48 没有问题

| N1 | mBaseSize | Slot 大小 | Slot1 起始 | Slot1 结束 | 分配大小 | 是否越界 |
|----|-----------|----------|-----------|-----------|---------|---------|
| 16 | 64 | 8,192 | 12,288 | 20,480 | 16,384 | **越界 4,096** |
| 24 | 96 | 12,288 | 12,288 | 24,576 | 24,576 | 恰好不越界 |
| 48 | 192 | 24,576 | 12,288 | 36,864 | 49,152 | 不越界（Slot重叠但在范围内） |

- **N1=24**: `mBaseSize=96=M_BASIC_BLOCK`，偏移量恰好等于实际 slot 大小，完美匹配
- **N1=48**: `mBaseSize=192>96`，Slot1 起始地址（12288）小于 Slot0 大小（24576），虽然两个 slot 有重叠，但整体未超出分配范围
- **N1=16**: `mBaseSize=64<96`，Slot1 起始地址（12288）超过了分配范围的一半（8192），导致 Slot1 末尾越界

## 4. 修复方案

### 4.1 修复思路

将硬编码的偏移量常量替换为基于实际 `mBaseSize` 动态计算的偏移量。

### 4.2 具体代码修改

**文件**: `op_kernel/arch35/quant_lightning_indexer_service_cube.h`

#### 修改 1: 新增成员变量

```cpp
// 在 QLIMatmul 类中添加动态偏移量成员变量
uint64_t queryL1Offset_ = 0;   // 替代 QUERY_BUFFER_OFFSET
uint64_t queryL0Offset_ = 0;   // 替代 L0AB_BUFFER_OFFSET (Query L0A用)
uint64_t l0cOffset_ = 0;       // 替代 L0C_BUFFER_OFFSET
```

#### 修改 2: 在 InitBuffers 中初始化

```cpp
__aicore__ inline void InitBuffers(TPipe *pipe)
{
    // 根据实际 mBaseSize 计算动态偏移量
    queryL1Offset_ = constInfo_.mBaseSize * D_BASIC_BLOCK;
    queryL0Offset_ = constInfo_.mBaseSize * D_BASIC_BLOCK_L0;
    l0cOffset_ = constInfo_.mBaseSize * S2_BASIC_BLOCK_L0;

    // ... 其余初始化代码不变
}
```

#### 修改 3: 替换所有引用点（共 6 处）

| 函数 | 原代码 | 修改后 |
|------|--------|--------|
| `QueryNd2Nz` | `(queryL1Mte2BufIdx_ % QUERY_BUF_NUM) * QUERY_BUFFER_OFFSET` | `... * queryL1Offset_` |
| `LoadQueryToL0a` | `(queryL1Mte1BufIdx_ % QUERY_BUF_NUM) * QUERY_BUFFER_OFFSET` | `... * queryL1Offset_` |
| `LoadQueryToL0a` | `(queryL0BufIdx_ % QUERY_BUF_NUM) * L0AB_BUFFER_OFFSET` | `... * queryL0Offset_` |
| `ComputeL0c` | `(cL0BufIdx_ % QUERY_BUF_NUM) * L0C_BUFFER_OFFSET` | `... * l0cOffset_` |
| `Fixp` (2处) | `(fixpBufIdx_ % QUERY_BUF_NUM) * L0C_BUFFER_OFFSET` | `... * l0cOffset_` |

#### 注意：不需要修改的部分

- `KEY_BUFFER_OFFSET`: Key 的维度不依赖 `mBaseSize`，保持不变
- Key L0B 的 `L0AB_BUFFER_OFFSET`: Key 侧的 offset 基于 `kSeqSize` 而非 `mBaseSize`，保持不变

## 5. 修复验证

| N1 | mBaseSize | queryL1Offset_ | Slot1 范围 | 分配大小 | 状态 |
|----|-----------|----------------|-----------|---------|------|
| 16 | 64 | 8,192 | [8192, 16384) | 16,384 | 正确 |
| 24 | 96 | 12,288 | [12288, 24576) | 24,576 | 正确 |
| 32 | 128 | 16,384 | [16384, 32768) | 32,768 | 正确 |
| 48 | 192 | 24,576 | [24576, 49152) | 49,152 | 正确 |

修复后，对任意 N1 值，`Slot1_end = 2 × mBaseSize × D_BASIC_BLOCK = 分配大小`，永远不会越界。

## 6. 经验教训

### 6.1 核心教训：Buffer 分配与索引必须使用同一组参数

> **反模式**: 分配用动态参数，索引用硬编码常量
>
> **正确做法**: 分配和索引都使用同一个动态参数（或由该参数派生的偏移量）

```cpp
// ❌ 错误：分配和索引使用不同的值
pipe->InitBuffer(buf, BUF_NUM * dynamicSize * DIM);
DataCopy(tensor[(idx % BUF_NUM) * HARDCODED_OFFSET], ...);

// ✅ 正确：分配和索引使用相同的值
uint64_t slotOffset = dynamicSize * DIM;
pipe->InitBuffer(buf, BUF_NUM * slotOffset);
DataCopy(tensor[(idx % BUF_NUM) * slotOffset], ...);
```

### 6.2 泛化开发检查清单

在进行维度泛化时，务必检查以下内容：

- [ ] **所有 `constexpr` 常量**: 是否有依赖被泛化维度的硬编码值？
- [ ] **Buffer 偏移量**: 分配大小和 slot 偏移是否一致？
- [ ] **双缓冲索引**: `slot_offset` 是否等于 `allocated_size / BUF_NUM`？
- [ ] **多级存储**: L1、L0A、L0B、L0C 各级 buffer 是否都检查过？
- [ ] **相邻 Buffer**: 越界是否会污染相邻 buffer 的数据？
- [ ] **边界用例**: 新的泛化值是否小于原始设计的硬编码值？

### 6.3 调试技巧

1. **当精度问题仅在特定参数下出现时**，优先检查参数相关的 buffer 大小计算和偏移量
2. **重点关注 "分配" 与 "使用" 的参数一致性**，尤其是存在 `constexpr` 常量的场景
3. **构造越界分析表**：列出各参数组合下的分配大小、slot 偏移、slot 范围，快速定位越界
4. **Buffer 越界通常不会立即崩溃**（NPU 内存是连续分配的），而是表现为数据污染导致的精度异常

### 6.4 预防措施

1. **设计阶段**: 定义 buffer 时，优先使用运行时变量而非编译期常量作为偏移基准
2. **代码审查**: 对所有 `constexpr` 偏移量，追溯其依赖的参数是否可能变化
3. **测试策略**: 泛化测试时，务必包含小于原始设计值的用例（如本例中 N1=16 < 原始 N1=24）
4. **断言保护**: 在 `InitBuffers` 中添加 `ASSERT(slotOffset * BUF_NUM <= allocatedSize)` 类断言

## 7. 影响范围

| 层级 | Buffer | 越界影响 | 严重程度 |
|------|--------|---------|---------|
| L1 | Query L1 → Key L1 | Key 数据被覆盖，Mmad 结果错误 | **严重** |
| L0A | Query L0A | L0A 内部越界，可能影响后续 Mmad | **严重** |
| L0C | 输出 L0C | L0C 内部越界，Fixpipe 输出错误 | **严重** |

三者共同作用，导致最终输出的 TopK 索引完全错误。

## 8. 关联信息

- **算子路径**: `attention/quant_lightning_indexer/`
- **问题文件**: `op_kernel/arch35/quant_lightning_indexer_service_cube.h`
- **平台**: ascend950 (arch35)
- **数据类型**: fp8_e4m3fn / hifloat8
- **执行模式**: MIX_AIC_1_2（1 Cube + 2 Vector）
- **日期**: 2026-03-17
