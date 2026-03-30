# LightningIndexer 算子 ascend950 版本详细设计说明书

<center>**修订记录**</center>

|    日期    | 修订版本 |                  修改描述                   |    作者     |
| :--------: | :------: | :-----------------------------------------: | :---------: |
| 2026-03-25 |   1.0    | LightningIndexer ascend950 版本详细设计初稿 | Claude/工号 |
| 2026-03-27 |   1.1    | 补充 arch35 vf 相关函数详细设计方案         | Claude/工号 |

---

# 1 关联需求

| **需求编号** | 需求标题   | 需求链接 | 概要设计链接             | 修订版本 |
| ------------ | ---------- | -------- | ------------------------ | -------- |
|              | DSA LI算子 |          | lightning_indexer概设.md | 1.0      |

**需求背景**：
LightningIndexer 算子目前仅支持 ascend910b/ascend910_93 平台，需要扩展支持 ascend950 平台。ascend950 与 ascend910b 在硬件架构上存在差异（L0 大小减半），需要基于已有的 quant_lightning_indexer（QLI，ascend950版本）去掉量化逻辑来实现。

**功能定义**：

- 新增 lightning_indexer 算子对 ascend950 平台的支持
- 数据类型从 QLI 的 FP8 量化改为 FP16/BF16 非量化
- 去掉 query_dequant_scale 和 key_dequant_scale 输入
- 计算公式简化为：`Score = W ⊙ ReLU(Q @ K^T)`
- 保持与现有 lightning_indexer（ascend910b版本）相同的功能逻辑

**设计原则**：

- **零框架侵入**：不需要框架侧修改，直接通过算子实现支持
- **最小 Kernel 侵入**：复用 QLI ascend950 的 kernel 框架，去掉量化相关代码
- **格式一致**：输出数据格式与现有 lightning_indexer 保持一致

---

# 2 接口实现设计

## 2.1 PTA接口实现

**变更说明**：无新增 PTA 接口，复用现有的 lightning_indexer PTA 接口定义。

## 2.2 aclnn接口实现

**变更说明**：无新增 aclnn 接口，复用现有的 lightning_indexer aclnn 接口定义。

| 参数名                    | 输入/输出 | 描述               | 使用说明 | 数据类型                      | 数据格式 | 维度                    |
| :------------------------ | :-------- | :----------------- | :------- | :---------------------------- | :------- | :---------------------- |
| query                     | 输入      | Query 数据         | 必选     | **FP16/BF16**（QLI 中为 FP8） | ND       | [B,S1,N1,D] 或 [T,N1,D] |
| key                       | 输入      | Key 数据           | 必选     | **FP16/BF16**（QLI 中为 FP8） | ND       | [B,S2,N2,D] 或 [T,N2,D] |
| weights                   | 输入      | 注意力权重         | 必选     | FP16/BF16/FP32                | ND       | [B,S1,N1] 或 [T,N1]     |
| actual_seq_lengths_query  | 输入      | Query 实际序列长度 | 可选     | INT32                         | ND       | [B]                     |
| actual_seq_lengths_key    | 输入      | Key 实际序列长度   | 可选     | INT32                         | ND       | [B]                     |
| block_table               | 输入      | PageAttention 块表 | 可选     | INT32                         | ND       | [B, maxBlockNum]        |
| **sparse_indices** (输出) | 输出      | Top-K 索引         | 必选     | INT32                         | ND       | [B,S1,N2,sparse_count]  |

**aclnn接口实现变更**：

| 变化点                   | 说明                          |
| ------------------------ | ----------------------------- |
| 去掉 query_dequant_scale | LI 无量化，不需要反量化 scale |
| 去掉 key_dequant_scale   | LI 无量化，不需要反量化 scale |
| query/key 数据类型       | QLI: FP8 → LI: FP16/BF16      |

## 2.3 算子信息库

**OpDef 变更**（`lightning_indexer_def.cpp`）：

| 变更项       | 变更前                    | 变更后                    |
| ------------ | ------------------------- | ------------------------- |
| 输入数量     | 8（包含 2 个 scale 输入） | 6（去掉 2 个 scale 输入） |
| 数据类型校验 | ascend950 必须是 FP8      | ascend950 支持 FP16/BF16  |
| G Size 校验  | ascend950: G=16/24/32/64  | 保持不变                  |

## 2.4 图模式设计

**变更说明**：不涉及图模式变更，复用现有图模式支持。

---

# 3 总体设计

## 3.1 交付方式

| 类型     | 描述                                            | 备注                       |
| -------- | ----------------------------------------------- | -------------------------- |
| 交付内容 | lightning_indexer ascend950 版本 kernel 代码    | 复用 QLI 框架，去掉量化    |
| 代码承载 | `attention/lightning_indexer/op_kernel/arch35/` | 新增 ascend950 专用 kernel |

## 3.2 交付件汇总

| **序号** | **交付件**               | **是否需要** | **涉及变动** | **备注**                      |
| -------- | ------------------------ | ------------ | ------------ | ----------------------------- |
| 01       | pta接口适配              | 否           | 否           | 复用现有接口                  |
| 02       | pta接口文档              | 否           | 否           | 复用现有文档                  |
| 03       | aclnn接口适配            | 否           | 否           | 复用现有接口                  |
| 04       | aclnn接口文档            | 否           | 否           | 复用现有文档                  |
| 05       | GE图模式适配             | 否           | 否           | 复用现有支持                  |
| 06       | AclGraph图模式适配       | 否           | 否           | 复用现有支持                  |
| 07       | 算子原型                 | 是           | 是           | 新增 ascend950 数据类型支持   |
| 08       | OpDef定义                | 是           | 是           | 去掉 scale 输入，更新数据类型 |
| 09       | 算子tiling函数           | 是           | 是           | 更新基本块大小计算            |
| 10       | 算子kernel实现           | 是           | 是           | 核心变更，去掉量化逻辑        |
| 11       | 算子二进制配置           | 否           | 否           | 复用现有配置                  |
| 12       | inferShape/inferDataType | 是           | 是           | 更新数据类型推断              |
| 13       | 图融合pass               | 否           | 否           | 不涉及                        |

## 3.3 TilingKey设计

**变更说明**：通过 TilingData 运行时判定，新增 ascend950 FP16/BF16 数据类型组合。

| TilingKey 字段 | ascend910b (FP16/BF16) | ascend950 (FP16/BF16) |
| -------------- | ---------------------- | --------------------- |
| DT_Q           | FP16/BF16              | FP16/BF16             |
| DT_K           | FP16/BF16              | FP16/BF16             |
| PAGE_ATTENTION | 0/1                    | 0/1                   |
| LAYOUT_Q       | BSND/TND               | BSND/TND              |
| LAYOUT_K       | BSND/TND/PA_BSND       | BSND/TND/PA_BSND      |

## 3.4 模板列表

| 规格                  | 应用场景 | 模板             | 核间切分    | 核内切分   | 备注          |
| --------------------- | -------- | ---------------- | ----------- | ---------- | ------------- |
| ascend950 FP16 主流程 | BSND/TND | arch35/LI kernel | B×N2×gS1×S2 | gS1 内循环 | 无量化        |
| ascend950 FP16 PA     | PA_BSND  | arch35/LI kernel | B×N2×gS1×S2 | gS1 内循环 | PageAttention |
| ascend950 BF16 主流程 | BSND/TND | arch35/LI kernel | B×N2×gS1×S2 | gS1 内循环 | 无量化        |

## 3.5 代码结构设计

| 目录                                                         | 内容             | 变化点                  | 需求编号       | 需求标题       |
| ------------------------------------------------------------ | ---------------- | ----------------------- | -------------- | -------------- |
| `lightning_indexer/op_host/`                                 | Host 侧实现      | 更新 OpDef 和 tiling    | REQ-LI-950-001 | ascend950 支持 |
| `lightning_indexer/op_kernel/arch35/`                        | ascend950 kernel | **新增目录**            | REQ-LI-950-001 | ascend950 支持 |
| `lightning_indexer/op_kernel/arch35/lightning_indexer_kernel.h` | 主 Kernel 类     | 从 QLI 复制并去掉量化   | REQ-LI-950-001 | ascend950 支持 |
| `lightning_indexer/op_kernel/arch35/lightning_indexer_service_cube.h` | Cube 服务        | 从 QLI 复制             | REQ-LI-950-001 | ascend950 支持 |
| `lightning_indexer/op_kernel/arch35/lightning_indexer_service_vector.h` | Vector 服务      | 从 QLI 复制并去掉 scale | REQ-LI-950-001 | ascend950 支持 |
| `lightning_indexer/op_kernel/arch35/vf/`                     | 微核函数         | 从 QLI 复制并去掉 scale | REQ-LI-950-001 | ascend950 支持 |
| `lightning_indexer/op_kernel/lightning_indexer.cpp`          | 入口函数         | 增加 arch35 条件编译    | REQ-LI-950-001 | ascend950 支持 |

---

# 4 模板设计

## 4.1 LightningIndexer ascend950 计算模板

### 4.1.1 计算流程图

#### 正常块计算流程对比

```
QLI (FP8 量化) 流程：                                    LI (FP16/BF16 非量化) 流程：
┌─────────────────┐                                   ┌─────────────────┐
│  Query (FP8)    │                                   │  Query (FP16)    │
│  Key (FP8)      │                                   │  Key (FP16)      │
└────────┬────────┘                                   └────────┬────────┘
         │                                                      │
         ▼                                                      ▼
┌─────────────────┐                                   ┌─────────────────┐
│  Dequant Scale  │           【去掉】                  │  无需 Dequant   │
│  (Q × qScale)   │  ──────────────────────────────▶  │                 │
│  (K × kScale)   │                                    │                 │
└────────┬────────┘                                    └────────┬────────┘
         │                                                      │
         ▼                                                      ▼
┌─────────────────┐                                   ┌─────────────────┐
│  MMAD (FP8)     │                                   │  MMAD (FP16)    │
│  Q @ K^T         │                                   │  Q @ K^T         │
└────────┬────────┘                                   └────────┬────────┘
         │                                                      │
         ▼                                                      ▼
┌─────────────────┐                                   ┌─────────────────┐
│  Fixp (Dequant) │           【去掉量化】              │  Fixp (Direct)  │
│  + ReLU         │  ──────────────────────────────▶  │  + ReLU         │
└────────┬────────┘                                    └────────┬────────┘
         │                                                      │
         ▼                                                      ▼
┌─────────────────┐                                   ┌─────────────────┐
│  × Weight       │                                   │  × Weight       │
│  × kScale       │           【去掉 kScale】          │  (无 scale)     │
└────────┬────────┘                                    └────────┬────────┘
         │                                                      │
         ▼                                                      ▼
┌─────────────────┐                                   ┌─────────────────┐
│  Reduce Sum (G) │                                   │  Reduce Sum (G) │
│  + TopK         │                                   │  + TopK         │
└────────┬────────┘                                   └────────┬────────┘
         │                                                      │
         ▼                                                      ▼
┌─────────────────┐                                   ┌─────────────────┐
│  Indices (INT32)│                                   │  Indices (INT32)│
└─────────────────┘                                   └─────────────────┘
```

#### 三级流水时序图

```
时间 ─────────────────────────────────────────────────────────────────────────▶

AIC Core:
│  LoadKey[0]  │ Compute[0]  │ LoadKey[1]  │ Compute[1]  │ ...
│               │              │              │              │
│               │ Fixp[0] ────┼──────────────┼──────────────┤
│               │              │ Fixp[1] ────┐              │
│               │              │              │ Fixp[2] ────┐
│               ▼              ▼              ▼              ▼
AIV Core:      │ ProcessVec1[0] │ ProcessVec1[1] │ ProcessVec1[2] │
│               │ (Weight×ReLU)  │ (Weight×ReLU)  │ (Weight×ReLU)  │
│               │◀── wait ──────┼───────────────┼───────────────┤
│               │               │ TopK[0] ──────┼───────────────┤
│               │               │◀── wait ───────┤ TopK[1] ──────┤
│               ▼               ▼               ▼               ▼
```

### 4.1.2 Tiling实现设计

#### 4.1.2.1 TilingData设计

**TilingData 结构体定义**（`lightning_indexer_tiling.h`）：

```cpp
BEGIN_TILING_DATA_DEF(LITilingData)
TILING_DATA_FIELD_DEF(uint32_t, bSize)           // Batch 大小
TILING_DATA_FIELD_DEF(uint32_t, n2Size)          // N2 = 1（固定）
TILING_DATA_FIELD_DEF(uint32_t, gSize)            // G = N1/N2
TILING_DATA_FIELD_DEF(uint32_t, s1Size)           // Query 序列长度
TILING_DATA_FIELD_DEF(uint32_t, s2Size)          // Key 序列长度
TILING_DATA_FIELD_DEF(uint32_t, sparseCount)     // TopK 数量
TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum)      // 使用的核数
TILING_DATA_FIELD_DEF(uint32_t, blockSize)        // PA block 大小
TILING_DATA_FIELD_DEF(uint32_t, maxBlockNumPerBatch)  // PA 每 batch 最大 block 数
TILING_DATA_FIELD_DEF(uint32_t, sparseMode)      // 稀疏模式
END_TILING_DATA_DEF
```

**TilingData 字段来源说明**：

| 字段                    | 来源              | 计算公式/取值                                            | 说明                                                 |
| ----------------------- | ----------------- | -------------------------------------------------------- | ---------------------------------------------------- |
| **bSize**               | Query shape 第0维 | `query.shape[0]`                                         | Batch 大小，TND 时取 actual_seq_lengths_query.length |
| **n2Size**              | Key shape 第2维   | `key.shape[2]`                                           | 固定为 1                                             |
| **gSize**               | N1/N2 计算        | `query.shape[2] / n2Size`                                | Group size，ascend950 支持 16/24/32/64               |
| **s1Size**              | Query shape 第1维 | `query.shape[1]`                                         | Query 序列长度                                       |
| **s2Size**              | Key shape         | 非PA: `key.shape[1]`；PA: `block_table.dim1 * blockSize` | Key 序列长度                                         |
| **sparseCount**         | 属性              | `attr.sparse_count`                                      | TopK 数量，支持 [1,2048]                             |
| **usedCoreNum**         | 核数计算          | `CalcTschBlockDim(aivNum, aicNum, aivNum)`               | AIC:AIV = 1:2                                        |
| **blockSize**           | Key shape 第1维   | `key.shape[1]`                                           | PA 场景的 block 大小                                 |
| **maxBlockNumPerBatch** | block_table shape | `block_table.shape[1]`                                   | PA 场景每 batch 最大 block 数                        |
| **sparseMode**          | 属性              | `attr.sparse_mode`                                       | 0: defaultMask, 3: rightDownCausal                   |

**数据类型校验（ascend950 新增）**：

| 数据类型  | ascend950 LI   | ascend950 QLI       | 备注          |
| --------- | -------------- | ------------------- | ------------- |
| query/key | **FP16/BF16**  | FP8_E4M3FN/HIFLOAT8 | **核心差异**  |
| weights   | FP16/BF16/FP32 | BF16/FP16           | 一致          |
| scale     | **不需要**     | 需要                | **LI 无量化** |
| output    | INT32          | INT32               | 一致          |

#### 4.1.2.2 核间Tiling策略（SplitCore）

**核心数据结构 SplitCoreInfo**：

```cpp
struct SplitCoreInfo {
    uint32_t s2Start;    // S2 起始索引（闭区间）
    uint32_t s2End;      // S2 终止索引（闭区间）
    uint32_t bN2Start;   // B×N2 起始索引
    uint32_t bN2End;     // B×N2 终止索引
    uint32_t gS1Start;   // gS1 起始索引
    uint32_t gS1End;     // gS1 终止索引
    bool isLD;           // 是否需要处理 LD（跨核归约）
    bool isCoreEnable;   // 该核是否有任务
};
```

**分核算法详细步骤**：

```
输入：totalBlockNum（总基本块数），coreNum（可用核数）
输出：每个核的 SplitCoreInfo

Step 1: 计算每个核最少处理块数
    minBlockPerCore = totalBlockNum / coreNum
    deal1MoreBlockCoreNum = totalBlockNum % coreNum  // 多分1块的核数

Step 2: 遍历所有 Tile [bN2Idx, gS1Idx, s2Idx]，为每个核分配块
    coreIdx = 0
    coreDealBlockCnt = (coreIdx < deal1MoreBlockCoreNum) ? minBlockPerCore + 1 : minBlockPerCore

    for bN2Idx in [0, batchSize * kHeadNum):
        for gS1Idx in [0, ceil(S1 * G / M_BASE)):
            for s2Idx in [0, ceil(S2 / S2_BASE)):
                coreDealBlockCnt--

                if coreDealBlockCnt == 0:
                    // 当前核分配完毕，记录终止位置
                    splitCoreInfo[coreIdx].bN2End = bN2Idx
                    splitCoreInfo[coreIdx].gS1End = gS1Idx
                    splitCoreInfo[coreIdx].s2End = s2Idx

                    // 判断是否需要 LD（跨核归约）
                    // 条件：s2Idx == 0（第一个S2块）且 s2End + 1 < s2BaseNum（还有后续块）
                    if s2Idx == 0 && (s2End + 1) < s2BaseNum:
                        splitCoreInfo[coreIdx].isLD = true

                    // 切换到下一个核
                    coreIdx++
                    coreDealBlockCnt = (coreIdx < deal1MoreBlockCoreNum) ?
                                       minBlockPerCore + 1 : minBlockPerCore
```

**分核结果示例**：

```
假设：totalBlockNum=100, coreNum=4

计算：
- minBlockPerCore = 100/4 = 25
- deal1MoreBlockCoreNum = 100%4 = 0

分配结果：
- 核0: 块 [0..24]   → bN2[0..], gS1[0..], s2[0..4]
- 核1: 块 [25..49]  → bN2[...], gS1[...], s2[...]
- 核2: 块 [50..74]  → bN2[...], gS1[...], s2[...]
- 核3: 块 [75..99]  → bN2[...], gS1[...], s2[...]
```

#### 4.1.2.3 核内Tiling策略（主循环结构 + sparse_mode mask 处理）

**四维循环空间**：

```
循环维度: [B × N2] × [gS1] × [S2]
         ↓
    [bN2Idx]   ×  [gS1Idx]  ×  [s2Idx]

每个维度范围：
- bN2Idx: 0 ~ batchSize × kHeadNum - 1
- gS1Idx: 0 ~ ceil(S1 × G / M_BASE) - 1
- s2Idx:  0 ~ ceil(S2 / S2_BASE) - 1

每个 Tile 的形状：
- M 方向: M_BASE = 256（最后一个 tile 可能更小）
- S2 方向: S2_BASE = 2048（最后一个 tile 可能更小）
```

**sparse_mode mask 处理逻辑**：

sparse_mode 用于控制 Attention 的 mask 模式，影响 S2 方向的循环范围计算：

| sparse_mode | 模式名称        | mask 逻辑               | 说明                                |
| ----------- | --------------- | ----------------------- | ----------------------------------- |
| 0           | defaultMask     | 无 mask，全 S2 参与计算 | 所有 Key 都可以被任意 Query 关注    |
| 3           | rightDownCausal | 右下三角 causal mask    | 每个 Query 只能关注其位置之前的 Key |

**sparse_mode=3 (rightDownCausal) 计算逻辑**：

```
对于 gS1Idx 对应的 Query 起始位置：s1Offset = s1BaseSize × gS1Idx

有效 S2 范围计算：
- validS2LenBase = actS2Size - actS1Size  // S2 - S1 的差值
- validS2Len = s1Offset + validS2LenBase + s1BaseSize
- validS2Len = min(validS2Len, actS2Size)  // 不超过 S2 总长度
- validS2Len = max(validS2Len, 1)         // 至少为 1

有效 S2 块数：GetS2BaseBlockNum = ceil(validS2Len / s2BaseSize)
```

**示例**（actS1Size=8, actS2Size=16, s1BaseSize=4, s2BaseSize=8）：

- gS1Idx=0: s1Offset=0, validS2Len=1+8+4=13 → 2 个 S2 块
- gS1Idx=1: s1Offset=4, validS2Len=4+8+4=16 → 2 个 S2 块
- gS1Idx=2: s1Offset=8, validS2Len=8+8+4=20 → clamp 到 16 → 2 个 S2 块

**循环边界计算**：

```cpp
// S1 方向基本块数
s1GBaseNum = CeilDiv(actS1Size, s1BaseSize);  // s1BaseSize = M_BASE / G

// S2 方向基本块数
s2BaseNum = CeilDiv(actS2Size, s2BaseSize);

// 尾块大小处理
s1BasicSizeTail = (actS1Size * G) % M_BASE;  // M 方向余数
s1BasicSizeTail = (s1BasicSizeTail == 0) ? M_BASE : s1BasicSizeTail;

s2BasicSizeTail = actS2Size % S2_BASE;  // S2 方向余数
s2BasicSizeTail = (s2BasicSizeTail == 0) ? S2_BASE : s2BasicSizeTail;
```

#### 4.1.2.4 基本块大小详细推导

**ascend950 硬件资源约束**：

| 硬件单元 | 大小 | 说明                  |
| -------- | ---- | --------------------- |
| L0a      | 64KB | AIC Matrix A 输入缓存 |
| L0b      | 64KB | AIC Matrix B 输入缓存 |
| L0C      | 64KB | AIC Matrix C 输出缓存 |
| UB       | ~2MB | AIV Unified Buffer    |

**M_BASE = 256 的推导**：

```
约束条件：
1. L0a 容量: 64KB = 65536 Bytes
2. 数据类型: FP16 (2 Bytes) 或 FP8 (1 Byte)
3. Head Dim: D = 128（固定）
4. MMAD m 参数约束: 16/32/64/128/256

计算：
- L0a 存储 Query: M × D × sizeof(FP16) ≤ 64KB
- M × 128 × 2 ≤ 65536
- M ≤ 256

验证：
- 当 M = 256, FP16: 256 × 128 × 2 = 64KB = L0a 大小 ✓
- 当 M = 512, FP16: 512 × 128 × 2 = 128KB > 64KB ✗

结论：M_BASE = 256
```

**S2_BASE = 2048 的推导**：

```
约束条件：
1. L0C 容量: 64KB = 65536 Bytes（双缓冲，每份 32KB）
2. 数据类型: float (4 Bytes)
3. M_BASE = 256
4. n 参数约束: 16/32/64/128

计算：
- L0C 存储中间结果: M × S2 × sizeof(float) ≤ 64KB
- 256 × S2 × 4 ≤ 65536
- S2 ≤ 64（理论值，考虑双缓冲减半）

但 QLI/LI 实际使用 S2_BASE = 2048，原因：
1. S2_BASE 是外层循环的"基本处理单元"，不是 L0C 直接承载的大小
2. 实际 L0C 承载的是 S2_BASIC_BLOCK_L0 = 128
3. S2_BASE = 2048 意味着外层循环每次处理 2048 个 S2 列
4. 内部通过循环展开 S2_BASIC_BLOCK_L0 = 128 来填充 L0C

结论：S2_BASE = 2048（外层块），S2_BASIC_BLOCK_L0 = 128（L0C 块）
```

**S1_BASE = M_BASE / G 的推导**：

```
定义：S1_BASE = M_BASE / G

含义：每个 S1 基本块包含的 S1 token 数

示例：
- G = 64: S1_BASE = 256/64 = 4
- G = 32: S1_BASE = 256/32 = 8
- G = 24: S1_BASE = 256/24 ≈ 10.67（向上取整）
- G = 16: S1_BASE = 256/16 = 16

AIV 并行度：AIV 核按奇偶分配不同的 S1 行
- 偶数核: 处理前一半 S1_BASE
- 奇数核: 处理后一半 S1_BASE
```

### 4.1.3 Buffer设计

#### 4.1.3.1 UB空间的分配

**UB 空间需求分析**：

| Buffer       | QLI (FP8) | LI (FP16)     | 变化     |
| ------------ | --------- | ------------- | -------- |
| sortOutBuf   | 2048 × 4B | **2048 × 4B** | 不变     |
| tmpBuf       | 2048 × 4B | **2048 × 4B** | 不变     |
| indexBuf     | 2048 × 4B | **2048 × 4B** | 不变     |
| reduceOutBuf | 128 × 4B  | **128 × 4B**  | 不变     |
| kScaleBuf    | 128 × 2B  | **不需要**    | **去掉** |

**变更点**：LI 不需要 kScaleBuf，因为没有 key_dequant_scale 输入。

#### 4.1.3.2 workspace的分配

**变更说明**：workspace 分配保持不变，复用 QLI ascend950 的计算方式。

```
Workspace 内存布局：
|--- mm1ResGm ---|--- vec1ResGm (LD中间) ---|--- vec1ParamGm (LD参数) ---|--- weightWorkspace ---
| Core0_DB0/1   | Core0_Head/Tail          | Core0_Params               | Core0_WorkSpace
| Core1_DB0/1   | Core1_Head/Tail          | Core1_Params               | Core1_WorkSpace
...
```

**大小计算**（无变化）：

- mm1Res: M_BASE × S2_BASE × sizeof(float) × 2 × aicNum = 256 × 2048 × 4 × 2 × aicNum
- LD中间: S1_BASE × 2 × 2 × TOPK × sizeof(float) × aicNum
- LD参数: S1_BASE × 2 × 16 × sizeof(int64) × aicNum

#### 4.1.3.3 L1\L0的分配

| Buffer    | 块大小                                 | BUF_NUM | 数据类型  | 大小              | 备注           |
| --------- | -------------------------------------- | ------- | --------- | ----------------- | -------------- |
| Query L1  | S1G_BASIC_BLOCK_L1 × D × sizeof(T)     | 2       | FP16/BF16 | 256×128×2×2=128KB | 无量化，无变化 |
| Key L1    | S2_BASIC_BLOCK_L0 × D × sizeof(T)      | 2       | FP16/BF16 | 128×128×2×2=64KB  | 无量化，无变化 |
| Weight L1 | S1G_BASIC_BLOCK_L1 × BLOCK_CUBE        | 2       | FP16/BF16 | 256×32×2×2=32KB   | 无量化，无变化 |
| S L1      | S1G_BASIC_BLOCK_L0 × S2_BASIC_BLOCK_L0 | 2       | FP16      | 128×128×2×2=64KB  | 无量化，无变化 |
| L0a       | 64KB                                   | 4       | FP16/BF16 | 64KB × 4          | 复用 QLI       |
| L0b       | 64KB                                   | 4       | FP16/BF16 | 64KB × 4          | 复用 QLI       |
| L0C       | 64KB                                   | 2       | float     | 64KB × 2          | 复用 QLI       |

### 4.1.4 Kernel设计

#### 4.1.4.1 入口函数变更

**文件**：`lightning_indexer.cpp`

```cpp
// ascend950 条件编译判断逻辑
// 通过 TilingKey 区分不同平台和数据类型组合

template <int DT_Q, int DT_K, int DT_OUT, int PAGE_ATTENTION, int LAYOUT_T, int K_LAYOUT_T, int DT_W_FLAG>
__global__ __aicore__ void lightning_indexer(...)
{
    // ascend910b/ascend910_93 使用原有 kernel
    // ascend950 使用 arch35 kernel
    if constexpr (DT_Q == LI_TPL_FP16 && DT_K == LI_TPL_FP16) {
        if constexpr (/* ascend950 平台判断 */) {
            // 使用 arch35 kernel（无量化版本）
            INVOKE_LI_ARCH35_OP_IMPL(LIPreload, half, half, int32_t, ...);
        } else {
            // 使用原有 kernel
            INVOKE_LI_NO_KFC_OP_IMPL(LIPreload, half, half, int32_t, ...);
        }
    } else if constexpr (DT_Q == LI_TPL_BF16 && DT_K == LI_TPL_BF16) {
        if constexpr (/* ascend950 平台判断 */) {
            INVOKE_LI_ARCH35_OP_IMPL(LIPreload, bfloat16_t, bfloat16_t, int32_t, ...);
        } else {
            INVOKE_LI_NO_KFC_OP_IMPL(LIPreload, bfloat16_t, bfloat16_t, int32_t, ...);
        }
    }
}
```

#### 4.1.4.2 主 Kernel 类

**文件**：`arch35/lightning_indexer_kernel.h`

**类成员变量**：

```cpp
template <typename LIT>
class LIPreload {
    // ================================类型定义=================================
    using Q_T = typename LIT::queryType;           // FP16/BF16
    using K_T = typename LIT::keyType;             // FP16/BF16
    using OUT_T = typename LIT::outputType;         // INT32
    static constexpr bool PAGE_ATTENTION = LIT::pageAttention;

    // ================================常量定义=================================
    static constexpr uint32_t M_BASE_SIZE = 256;       // M 方向基本块
    static constexpr uint32_t S2_BASE_SIZE = 2048;    // S2 方向基本块（外层）
    static constexpr uint32_t S2_BASIC_BLOCK_L0 = 128; // S2 方向基本块（L0C 内层）
    static constexpr uint32_t HEAD_DIM = 128;          // 固定 128
    static constexpr uint32_t K_HEAD_NUM = 1;           // Key head 数固定为 1
    static constexpr uint32_t GM_ALIGN_BYTES = 512;     // GM 对齐要求

    // ================================服务类=================================
    LIMatmul<LIT> matmulService;      // AIC Cube 矩阵乘法服务
    LIVector<LIT> vectorService;       // AIV Vector 处理服务

    // ================================全局内存张量=================================
    GlobalTensor<Q_T> queryGm;        // Query: [B, S1, N1, D]
    GlobalTensor<K_T> keyGm;          // Key: [B, S2, N2, D] 或 PA 格式
    GlobalTensor<half> weightsGm;      // Weights: [B, S1, N1]
    GlobalTensor<int32_t> indiceOutGm; // 输出索引: [B, S1, N2, sparse_count]
    GlobalTensor<int32_t> blockTableGm; // PA block 表

    // ================================运行时信息=================================
    LICommon::ConstInfo constInfo{};      // 常量信息
    TempLoopInfo tempLoopInfo{};           // 临时循环信息
    LICommon::SplitCoreInfo splitCoreInfo{}; // 分核信息
};
```

**Init 函数详细流程**：

```cpp
template <typename LIT>
__aicore__ inline void LIPreload<LIT>::Init(
    __gm__ uint8_t *query,
    __gm__ uint8_t *key,
    __gm__ uint8_t *weights,
    // 【变更】去掉: __gm__ uint8_t *queryScale, __gm__ uint8_t *keyScale,
    __gm__ uint8_t *actualSeqLengthsQ,
    __gm__ uint8_t *actualSeqLengthsK,
    __gm__ uint8_t *blockTable,
    __gm__ uint8_t *sparseIndices,
    __gm__ uint8_t *workspace,
    const LITilingData *__restrict tiling,
    TPipe *tPipe)
{
    // Step 1: 获取核索引
    if ASCEND_IS_AIV {
        tmpBlockIdx = GetBlockIdx();  // vec: 0-47
        aiCoreIdx = tmpBlockIdx / 2;   // AIV 与 AIC 配对
    } else {
        tmpBlockIdx = GetBlockIdx();  // cube: 0-23
        aiCoreIdx = tmpBlockIdx;      // AIC 核索引
    }

    // Step 2: 初始化 Tiling 数据
    InitTilingData(tiling);
    InitActualSeqLen(actualSeqLengthsQ, actualSeqLengthsK);

    // Step 3: 计算分核
    SplitCore(aiCoreIdx, usedCoreNum, splitCoreInfo);

    // Step 4: 初始化 TPipe
    pipe = tPipe;

    // Step 5: Workspace 内存布局
    // |--- mm1ResGm ---|--- vec1ResGm ---|--- vec1ParamGm ---|--- weightWorkspace ---|
    uint64_t offset = 0;

    // 5.1 mm1 结果缓冲区（AIC 用，双缓冲）
    GlobalTensor<float> mm1ResGm;
    uint64_t singleCoreMm1ResSize = WS_DOUBLE * s1BaseSize * s2BaseSize * sizeof(float);
    mm1ResGm.SetGlobalBuffer((__gm__ float *)(workspace + aiCoreIdx * singleCoreMm1ResSize));
    offset += GetBlockNum() * singleCoreMm1ResSize;

    // 5.2 LD 中间结果（AIV 用）
    GlobalTensor<float> vec1ResGm;
    vec1ResGm.SetGlobalBuffer((__gm__ float *)(workspace + offset));
    offset += GetBlockNum() * s1BaseSize * WS_DOUBLE * WS_DOUBLE * BASE_TOPK * sizeof(float);

    // 5.3 LD 参数（AIV 用）
    GlobalTensor<int64_t> vec1ParamGm;
    vec1ParamGm.SetGlobalBuffer((__gm__ int64_t *)(workspace + offset));
    offset += GetBlockNum() * s1BaseSize * WS_DOUBLE * LD_PARAM_NUM * sizeof(int64_t);

    // 5.4 Weight Workspace
    GlobalTensor<half> weightWorkspaceGm;
    uint64_t weightMemSize = BLOCK_CUBE * mBaseSize * WS_DOUBLE * sizeof(half);
    weightWorkspaceGm.SetGlobalBuffer((__gm__ half *)(workspace + offset + aiCoreIdx * weightMemSize));

    // Step 6: 初始化各侧张量
    if ASCEND_IS_AIV {
        vectorService.InitParams(constInfo, tiling);
        indiceOutGm.SetGlobalBuffer((__gm__ int32_t *)sparseIndices);
        weightsGm.SetGlobalBuffer((__gm__ half *)weights);
        // 【变更】去掉: qScaleGm, kScaleGm
        // 【变更】InitVecInputTensor 不再传入 scale 参数
        blockTableGm.SetGlobalBuffer((__gm__ int32_t *)blockTable);
        vectorService.InitVecInputTensor(weightsGm, indiceOutGm, blockTableGm);
        vectorService.InitVecWorkspaceTensor(weightWorkspaceGm, mm1ResGm, vec1ResGm, vec1ParamGm);
    } else {
        matmulService.InitParams(constInfo);
        queryGm.SetGlobalBuffer((__gm__ Q_T *)query);
        if constexpr (PAGE_ATTENTION) {
            blockTableGm.SetGlobalBuffer((__gm__ int32_t *)blockTable);
        }
        keyGm.SetGlobalBuffer((__gm__ K_T *)key);
        matmulService.InitMm1GlobalTensor(blockTableGm, keyGm, queryGm, mm1ResGm, weightWorkspaceGm);
    }

    // Step 7: 初始化 Buffer
    InitBuffers();
}
```

#### 4.1.4.3 ProcessMain 主循环详细流程

**主循环伪代码**：

```cpp
template <typename LIT>
__aicore__ inline void LIPreload<LIT>::ProcessMain()
{
    if (aiCoreIdx >= usedCoreNum) {
        return;  // 无任务核直接返回
    }

    // Step 1: 初始化同步事件
    if ASCEND_IS_AIV {
        vectorService.AllocEventID();
        CrossCoreSetFlag<FIA_SYNC_MODE2, PIPE_MTE2>(syncV1C1);
        CrossCoreSetFlag<FIA_SYNC_MODE2, PIPE_MTE2>(syncV1C1);
    } else {
        matmulService.AllocEventID();
        CrossCoreSetFlag<FIA_SYNC_MODE2, PIPE_FIX>(syncC1V0);
        CrossCoreSetFlag<FIA_SYNC_MODE2, PIPE_FIX>(syncC1V0);
    }

    // Step 2: 四维循环遍历
    // for bN2Idx in [bN2Start, bN2End]:
    //     for gS1Idx in [gS1Start, gS1End]:
    //         for s2Idx in [s2Start, s2End]:
    //             ProcessBaseBlock(...)

    RunInfo runInfo[TASK_CACHE_SIZE];  // 双缓冲
    uint32_t gloop = 0;

    for (uint32_t bN2LoopIdx = splitCoreInfo.bN2Start; bN2LoopIdx <= splitCoreInfo.bN2End; bN2LoopIdx++) {
        CalcGS1LoopParams(bN2LoopIdx);  // 计算 S1 方向循环参数

        if (tempLoopInfo.curActSeqLenIsZero) {
            // 处理 actSeqLen == 0 的情况
            DealActSeqLenIsZero(tempLoopInfo.bIdx, tempLoopInfo.n2Idx, 0U);
            continue;
        }

        for (uint32_t gS1LoopIdx = splitCoreInfo.gS1Start; gS1LoopIdx <= tempLoopInfo.gS1LoopEnd; gS1LoopIdx++) {
            CalcS2LoopParams(bN2LoopIdx, gS1LoopIdx);  // 计算 S2 方向循环参数

            bool isEnd = (bN2LoopIdx == splitCoreInfo.bN2End) &&
                        (gS1LoopIdx == splitCoreInfo.gS1End);
            // 最后一个迭代额外预取一次
            uint32_t extraLoop = isEnd ? TASK_CACHE_SIZE - 1 : 0;

            for (int s2LoopIdx = splitCoreInfo.s2Start;
                 s2LoopIdx <= (tempLoopInfo.s2LoopEnd + extraLoop); s2LoopIdx++) {
                ProcessBaseBlock(gloop, s2LoopIdx, runInfo);
                ++gloop;
            }
            splitCoreInfo.s2Start = 0;  // 重置，后续从 0 开始
        }
        splitCoreInfo.gS1Start = 0;  // 重置
    }

    // Step 3: 等待所有计算完成
    if ASCEND_IS_AIV {
        vectorService.FreeEventID();
        CrossCoreWaitFlag(syncC1V0);
        CrossCoreWaitFlag(syncC1V0);
    } else {
        matmulService.FreeEventID();
        CrossCoreWaitFlag(syncV1C1);
        CrossCoreWaitFlag(syncV1C1);
    }
}
```

**ProcessBaseBlock 详细流程**：

```cpp
template <typename LIT>
__aicore__ inline void LIPreload<LIT>::ProcessBaseBlock(
    uint32_t loop,
    uint64_t s2LoopIdx,
    RunInfo runInfo[TASK_CACHE_SIZE])
{
    int32_t curTaskId = loop % TASK_CACHE_SIZE;
    int32_t lastTaskId = 1 - curTaskId;

    // Step 1: 计算当前 RunInfo
    CalcRunInfo(loop, s2LoopIdx, runInfo[curTaskId]);

    // Step 2: AIC 计算（Matmul Q @ K^T）
    if (runInfo[curTaskId].isValid) {
        if ASCEND_IS_AIC {
            if (runInfo[curTaskId].isFirstS2InnerLoop) {
                CrossCoreWaitFlag(syncV0C1);  // 等待上一个 V0 完成
            }
            CrossCoreWaitFlag(syncV1C1);       // 等待上一个 V1 完成

            // 执行 Matmul
            matmulService.ComputeMm1(runInfo[curTaskId]);

            // 通知 V1
            CrossCoreSetFlag<FIA_SYNC_MODE2, PIPE_FIX>(syncC1V1);

            // 最后一个 S2 迭代，通知 V0
            if (runInfo[curTaskId].isLastS2InnerLoop) {
                CrossCoreSetFlag<FIA_SYNC_MODE2, PIPE_FIX>(syncC1V0);
            }
        }
        // AIV 在 else 分支处理 Vec0
        else {
            if (runInfo[curTaskId].isFirstS2InnerLoop) {
                CrossCoreWaitFlag(syncC1V0);  // 等待 C1 完成
                vectorService.ProcessVec0(runInfo[curTaskId]);  // 处理 Vec0
                CrossCoreSetFlag<FIA_SYNC_MODE2, PIPE_MTE3>(syncV0C1);  // 通知 C1
            }
        }
    }

    // Step 3: 处理上一个 RunInfo（流水线）
    if (runInfo[lastTaskId].isValid) {
        if ASCEND_IS_AIV {
            CrossCoreWaitFlag(syncC1V1);       // 等待对应 C1 完成
            vectorService.ProcessVec1(runInfo[lastTaskId]);  // 处理 Vec1（加权归约）
            CrossCoreSetFlag<FIA_SYNC_MODE2, PIPE_MTE3>(syncV1C1);  // 通知 C1
        }
        runInfo[lastTaskId].isValid = false;
    }

    // Step 4: TopK 处理（仅 AIV，最后一个 S2 迭代）
    if ASCEND_IS_AIV && runInfo[curTaskId].isLastS2InnerLoop {
        vectorService.ProcessTopK(runInfo[curTaskId]);
    }
}
```

#### 4.1.4.4 AIC Cube 服务（ComputeMm1 详细流程）

**文件**：`arch35/lightning_indexer_service_cube.h`

**Buffer 初始化配置**：

```cpp
__aicore__ inline void LIMatmul<LIT>::InitBuffers(TPipe *pipe)
{
    // Query L1: 256 × 128 × sizeof(T) × 2 buffer
    pipe->InitBuffer(bufQL1_, DOUBLE_BUF_NUM * S1G_BASIC_BLOCK_L1 * D_BASIC_BLOCK * sizeof(Q_T));
    queryL1_ = bufQL1_.Get<Q_T>();

    // Key L1: 128 × 128 × sizeof(T) × 2 buffer
    pipe->InitBuffer(bufKeyL1_, DOUBLE_BUF_NUM * S2_BASIC_BLOCK_L0 * D_BASIC_BLOCK * sizeof(K_T));
    keyL1_ = bufKeyL1_.Get<K_T>();

    // Weight L1: 256 × 32 × sizeof(half) × 2 buffer
    pipe->InitBuffer(bufWeightL1_, DOUBLE_BUF_NUM * S1G_BASIC_BLOCK_L1 * BLOCK_CUBE * sizeof(half));
    weightL1_ = bufWeightL1_.Get<half>();

    // S L1: 128 × 128 × sizeof(half) × 2 buffer
    pipe->InitBuffer(bufSL1_, DOUBLE_BUF_NUM * S2_BASIC_BLOCK_L0 * S1G_BASIC_BLOCK_L0 * sizeof(half));
    sL1_ = bufSL1_.Get<half>();

    // L0a: 64KB
    pipe->InitBuffer(bufL0A_, 64 * 1024);
    l0a_ = bufL0A_.Get<Q_T>();

    // L0b: 64KB
    pipe->InitBuffer(bufL0B_, 64 * 1024);
    l0b_ = bufL0B_.Get<K_T>();

    // L0C: 128KB（用于存储 float 结果）
    pipe->InitBuffer(bufL0C_, 128 * 1024);
    cL0_ = bufL0C_.Get<int32_t>();
}
```

**ComputeMm1 完整流程**：

```cpp
__aicore__ inline void LIMatmul<LIT>::ComputeMm1(const RunInfo &runInfo)
{
    // ========== 第一阶段：加载 Query 和 Weight（仅首个 S2 内环）==========
    if (runInfo.isFirstS2InnerLoop) {
        WaitFlag<HardEvent::MTE1_MTE2>(QW_MTE1_MTE2_EVENT + qwL1Mte2BufIdx_ % DOUBLE_BUF_NUM);
        // Query: GM → L1 (Nd2Nz)
        QueryNd2Nz(runInfo.actMBaseSize, runInfo);  // 256 × 128
        // Weight: GM → L1
        WeightDmaCopy(runInfo.actMBaseSize, runInfo);
    }

    // ========== 第二阶段：Matmul 主循环 ==========
    // S2 方向循环：ceil(2048/128) = 16 次
    int64_t s2L0LoopCnt = CeilDiv(runInfo.actualSingleProcessSInnerSize, S2_BASIC_BLOCK_L0);
    // S1G 方向循环：ceil(256/128) = 2 次
    int64_t s1L0LoopCnt = CeilDiv(runInfo.actMBaseSize, S1G_BASIC_BLOCK_L0);

    int64_t loopIdx = 0;

    // 2.1 第一次迭代：只做 QK 计算
    MmInfo mmInfo[2];
    CalcMmInfo(mmInfo[loopIdx & 1], loopIdx, s1L0LoopCnt, mmInfo[(loopIdx + 1) & 1], runInfo);
    ProcessQk(s1gL0RealSize[mmInfo[loopIdx & 1].s1gL0LoopId % s1L0LoopCnt], ...);
    loopIdx++;

    // 2.2 中间迭代：交替执行 QK 和 WS
    while (loopIdx < s2L0LoopCnt * s1L0LoopCnt) {
        CalcMmInfo(mmInfo[loopIdx & 1], loopIdx, s1L0LoopCnt, mmInfo[(loopIdx + 1) & 1], runInfo);
        ProcessQk(...);   // QK 计算
        SetFlag<HardEvent::FIX_MTE1>(FIX_MTE1_EVENT + sL1BufIdx_ % DOUBLE_BUF_NUM);
        sL1BufIdx_++;

        WaitFlag<HardEvent::FIX_MTE1>(FIX_MTE1_EVENT + sL1BufIdx_ % DOUBLE_BUF_NUM);
        ProcessWs(...);    // WS 计算
        loopIdx++;
    }

    // 2.3 最后一次迭代：执行 WS
    WaitFlag<HardEvent::FIX_MTE1>(FIX_MTE1_EVENT + (sL1BufIdx_ + 1) % DOUBLE_BUF_NUM);
    ProcessWs(...);

    // ========== 第三阶段：通知下一个 S2 迭代 ==========
    if (runInfo.isLastS2InnerLoop) {
        SetFlag<HardEvent::MTE1_MTE2>(QW_MTE1_MTE2_EVENT + qwL1Mte2BufIdx_ % DOUBLE_BUF_NUM);
        qwL1Mte2BufIdx_++;
    }
}
```

**FixpSToL1 变更（核心量化差异）**：

```cpp
// QLI 版本（有量化，Fixp 做 DEQF16 解量化）：
__aicore__ inline void FixpSToL1(...)
{
    SetFlag<HardEvent::M_FIX>(M_FIX_EVENT);
    WaitFlag<HardEvent::M_FIX>(M_FIX_EVENT);

    DataCopyCO12DstParams params;
    params.mSize = CeilAlign(s1gL0RealSize, BLOCK_CUBE);
    params.nSize = CeilAlign(s2L0RealSize, BLOCK_CUBE);
    params.dstStride = S1G_BASIC_BLOCK_L0;
    params.srcStride = params.mSize;
    params.quantPre = QuantMode_t::DEQF16;  // 【关键】FP8 → FP16 解量化
    params.reluPre = 1;                    // ReLU 激活
    params.channelSplit = 0;
    params.nz2ndEn = 0;
    SetFixpipePreQuantFlag(0x3a800000);

    DataCopy(sL1_[...], cL0_[...], params);
}

// LI 版本（无量化，直接传递 + ReLU）：
__aicore__ inline void FixpSToL1(...)
{
    SetFlag<HardEvent::M_FIX>(M_FIX_EVENT);
    WaitFlag<HardEvent::M_FIX>(M_FIX_EVENT);

    DataCopyCO12DstParams params;
    params.mSize = CeilAlign(s1gL0RealSize, BLOCK_CUBE);
    params.nSize = CeilAlign(s2L0RealSize, BLOCK_CUBE);
    params.dstStride = S1G_BASIC_BLOCK_L0;
    params.srcStride = params.mSize;
    params.quantPre = QuantMode_t::NoQuant;  // 【关键】无量化，直接传递
    params.reluPre = 1;                        // ReLU 激活保持不变
    params.channelSplit = 0;
    params.nz2ndEn = 0;
    SetFixpipePreQuantFlag(0);

    DataCopy(sL1_[...], cL0_[...], params);
}
```

#### 4.1.4.5 AIV Vector 服务详细流程

**文件**：`arch35/lightning_indexer_service_vector.h`

**Buffer 初始化配置**：

```cpp
__aicore__ inline void LIVector<LIT>::InitBuffers(TPipe *pipe)
{
    // sortOutBuf: TopK 排序输出
    pipe->InitBuffer(sortOutBuf_, BASE_TOPK * sizeof(float));

    // tmpBuf: 排序临时缓冲
    pipe->InitBuffer(tmpBuf_, BASE_TOPK * sizeof(float));

    // indexBuf: 索引缓冲
    pipe->InitBuffer(indexBuf_, BASE_TOPK * sizeof(int32_t));

    // reduceOutBuf: 归约输出（128 × float，对应 S2_BASE）
    pipe->InitBuffer(reduceOutBuf_, S2_BASE_SIZE * sizeof(float));

    // brcBuf: Broadcast 缓冲
    pipe->InitBuffer(brcBuf_, S2_BASE_SIZE * sizeof(float));

    // paramBuf: LD 参数缓冲
    pipe->InitBuffer(paramBuf_, LD_PARAM_NUM * sizeof(int64_t));

    // 【变更】QLI 需要 kScaleBuf，但 LI 不需要
}
```

**ProcessVec0 详细流程（权重加载 + 预乘）**：

```cpp
// Vec0 在 AIV 奇数核执行，处理 Weights 加载
__aicore__ inline void LIVector<LIT>::ProcessVec0(const RunInfo &runInfo)
{
    // 计算当前核负责的 S1 范围
    uint32_t curAivS1Idx = (blockId_ % 2 == 0) ?
        0 : CeilDiv(s1BaseSize_, 2);  // 奇偶核分配
    uint32_t curAivS1ProcNum = (blockId_ % 2 == 0) ?
        CeilDiv(s1BaseSize_, 2) : (s1BaseSize_ / 2);

    // Step 1: 加载 Weights 到 UB
    // Weights: [B, S1, N1] -> [curAivS1ProcNum, G]
    DataCopyPad(weightUB_, weightsGm[weightGmOffset], ...);

    // Step 2: 类型转换（如果是 BF16）
    if constexpr (!IsSameType<T, float>::value) {
        Cast(weightBuf_, weightUB_, RoundMode::CAST_NONE, curAivS1ProcNum * gSize_);
    }

    // Step 3: Weight Broadcast
    // [curAivS1ProcNum, G] -> [curAivS1ProcNum, G, 1]
    Brdc(weightBroadcastBuf_, weightBuf_, ...);
}
```

**ProcessVec1 详细流程（核心计算：ReLU × Weight × Sum）**：

```cpp
// Vec1 在 AIV 偶数核执行，执行 ReLU + Weighted Sum
__aicore__ inline void LIVector<LIT>::ProcessVec1(const RunInfo &runInfo)
{
    int pingpong = runInfo.loop % 2;

    // Step 1: 等待 AIC 的 Fixp 完成信号
    CrossCoreWaitFlag<CROSS_CV_EVENT + pingpong>(CROSS_VC_EVENT);

    // Step 2: 获取 AIC 输出的 UB 地址
    // mm1Res UB: [s1BaseSize_, s2BaseSize_] float
    auto qkBase = mm1ResGm_[pingpong * constInfo_.s1BaseSize * constInfo_.s2BaseSize];

    // Step 3: 计算 curAivS1Idx 和 curAivS1ProcNum
    uint32_t curAivS1Idx = (blockId_ % 2 == 0) ? 0 : CeilDiv(s1BaseSize_, 2);
    uint32_t curAivS1ProcNum = (blockId_ % 2 == 0) ?
        CeilDiv(s1BaseSize_, 2) : (s1BaseSize_ / 2);

    // Step 4: 获取当前核负责的 S1 数据
    auto qkCurBase = qkBase[curAivS1Idx * constInfo_.s2BaseSize];

    // Step 5: 执行 ReLU + Weighted Reduce Sum
    // 输入: qkCurBase [curAivS1ProcNum, gSize_, s2BaseSize_]
    // 输出: reduceOutBuf_ [curAivS1ProcNum, s2BaseSize_]
    BatchMulWeightAndReduceSum(
        reduceOutBuf_,                    // 输出
        qkCurBase,                        // 输入 QK
        weightBuf_,                       // Weights
        gSize_,                           // G 大小
        curAivS1ProcNum,                  // batch
        s2BaseSize_);                     // S2 大小

    // Step 6: 写出到 GM（score）
    DataCopyPad(scoreGm_[vec1OutGmOffset], reduceOutBuf_, ...);
}
```

**BatchMulWeightAndReduceSum 详细伪代码（核心变更）**：

```cpp
// LI 版本（无 scale）：计算 sum_g (relu(QK[g]) * W[g])
__aicore__ inline void BatchMulWeightAndReduceSum(
    LocalTensor<float> out,         // [batch, s2Size]
    LocalTensor<float> qk,           // [batch, gSize, s2Size]
    LocalTensor<half> weight,        // [batch, gSize]
    uint32_t gSize,
    uint32_t batch,
    uint32_t s2Size)
{
    // 外层循环：按 gSize/2 分组处理（每次处理 2 个 g）
    for (uint32_t g = 0; g < gSize; g += 2) {
        // 加载 2 个 g 组的 QK 结果
        // QK0 = qk[batch, g, :]   - 第 g 组
        // QK1 = qk[batch, g+1, :] - 第 g+1 组
        auto qk0 = qk[g * s2Size];
        auto qk1 = qk[(g + 1) * s2Size];

        // Broadcast Weights 到 S2 维度
        // w0 = Broadcast(weight[batch, g])
        // w1 = Broadcast(weight[batch, g + 1])
        auto w0 = Brdc(weight[g]);
        auto w1 = Brdc(weight[g + 1]);

        // ReLU(QK) = max(QK, 0)
        auto relu0 = Max(qk0, 0);
        auto relu1 = Max(qk1, 0);

        // 乘以 Weights
        // out += relu0 * w0
        // out += relu1 * w1
        for (uint32_t s = 0; s < s2Size; s++) {
            out[s] += relu0[s] * w0[s];
            out[s] += relu1[s] * w1[s];
        }
    }
    // 【关键差异】QLI 会在乘积后额外乘以 kScale
}

// QLI 版本（有 scale）：计算 sum_g (relu(QK[g]) * W[g] * kScale[g])
__aicore__ inline void BatchMulWeightAndReduceSum_Qli(...)
{
    // 前半部分相同：计算 relu * weight
    ...

    // 【额外步骤】乘以 kScale
    // kScale 是 per-S2 的缩放因子
    for (uint32_t s = 0; s < s2Size; s++) {
        out[s] *= kScale[s];
    }
}
```

#### 4.1.4.6 核间同步详细时序

**同步事件定义**：

| 事件 ID  | 名称  | 方向    | 说明            |
| -------- | ----- | ------- | --------------- |
| syncC1V1 | C1→V1 | AIC→AIV | Matmul 结果通知 |
| syncC1V0 | C1→V0 | AIC→AIV | TopK 开始通知   |
| syncV1C1 | V1→C1 | AIV→AIC | Vec1 完成通知   |
| syncV0C1 | V0→C1 | AIV→AIC | Vec0 完成通知   |

**同步时序图**：

```
时间 ──────────────────────────────────────────────────────────────────────────────▶

AIC Core[0]:
│  Fixp[loop] ───┤
                 │
                 ▼
AIV Core[0] (V0):    │ Vec0[loop] ──┤
                      │              │
                      ▼              ▼
AIV Core[1] (V1):    │              │ Vec1[loop-1] ──┤
                      │              │                 │
                      │              │                 ▼
                      │              │           TopK[loop-1]
                      │              │                 │
                      ▼              ▼                 ▼
                 syncC1V0      syncC1V1          输出
```

#### 4.1.4.7 运行时参数对比

| 参数              | QLI (ascend950) | LI (ascend950) | 变更说明 |
| ----------------- | --------------- | -------------- | -------- |
| queryScale        | half* GM        | **不需要**     | 去掉     |
| keyScale          | half* GM        | **不需要**     | 去掉     |
| kScaleBuf         | UB (128 × 2B)   | **不需要**     | 去掉     |
| M_BASE_SIZE       | 256             | 256            | 不变     |
| S2_BASE_SIZE      | 2048            | 2048           | 不变     |
| S2_BASIC_BLOCK_L0 | 128             | 128            | 不变     |
| S1_BASE_SIZE      | M_BASE/G        | M_BASE/G       | 不变     |
| DOUBLE_BUF_NUM    | 2               | 2              | 不变     |
| L0AB_BUF_NUM      | 4               | 4              | 不变     |

#### 4.1.4.8 VF (Vector Function) 整体架构设计

**【新增】VF 技术背景**：

ascend950 平台支持 **VF (Vector Function)** 微核编程模型，允许在 AIV 上执行细粒度的向量指令。VF 函数使用 `__simd_vf__` 修饰符，可直接操作寄存器 (RegTensor) 和 UB 缓冲区，实现高性能的向量化计算。

**VF 目录结构设计**：

```
op_kernel/arch35/vf/
├── lightning_indexer_topk.h          # TopK 算法封装类 LITopk
├── lightning_indexer_vector1.h       # 加权归约 VF 实现
├── vf_topk.h                         # uint32_t TopK VF 实现
└── vf_topk_16_gather.h               # uint16_t TopK VF 实现 (支持 S2 > 16K)
```

**VF 函数调用层级**：

```
ProcessVec1 (AIV 主流程)
    ├── BatchMulWeightAndReduceSum (vector1 命名空间)
    │       ├── CastWeightType (VF)
    │       ├── MulWeightAndReduceSum (VF)
    │       └── MulWeightAndReduceSum2 (VF)
    └── ProcessTopK
            └── LITopk<T>::operator() (topk 命名空间)
                    ├── vf_topk_16_gather.h 实现  # LI 使用此实现
                    └── vf_topk.h 实现           # 参考实现
```

**详细设计方案**：参见 [lightning_indexer_vf_design.md](./lightning_indexer_vf_design.md)

#### 4.1.4.9 TopK VF 算法设计

**【新增】分层直方图 TopK 算法**：

LightningIndexer 使用基于直方图的分层筛选算法实现 TopK，核心流程：

1. **高 8 位直方图统计**：统计输入数据高 8 位的分布，定位目标桶
2. **低 8 位直方图统计**：在目标高 8 位桶内，统计低 8 位分布
3. **查找第 K 值**：组合高低 8 位，确定第 K 大的完整 16 位值
4. **筛选输出**：遍历输入，输出所有大于 K 的索引，按需补充等于 K 的索引
5. **Gather 归并**（S2 > 16K 场景）：多轮计算，每轮归并历史结果

**Buffer 需求**（uint16_t 版本）：

| Buffer | 大小 | 用途 |
|--------|------|------|
| tmpIdx | topK × 2B | 临时索引输出 |
| histograms | 256 × 4B | 直方图统计 |
| idxHigh | 256 × 4B | 高 8 位目标桶 |
| idxLow | 256 × 4B | 低 8 位目标桶 |
| nkValue | 64 × 4B | next_k 值 |
| hisIndex[2] | 2 × Align(topK, 256) × 4B | 双缓冲历史索引（Gather 模式） |

**详细设计方案**：参见 [lightning_indexer_vf_design.md](./lightning_indexer_vf_design.md) 第 4.1.4.9 节

#### 4.1.4.10 Weighted Accumulation VF 算法设计

**【新增】加权求和计算流程**：

```
输入: QK [batch, G, S2] float, Weight [batch, G] T
输出: Score [batch, S2] uint16_t（排序键）

Step 1: 类型转换（如果需要）
    Weight (FP16/BF16) -> float

Step 2: 对于每个 g in [0, G)，步长 2：
    - 加载 QK[:, g:g+2, :] 到寄存器
    - Broadcast Weight[:, g:g+2] 到 S2 维度
    - ReLU(QK) * Weight
    - 累加到累加器

Step 3: 累加结果合并

Step 4: Cast 到 BF16/FP16

Step 5: FloatToSortableKey 转换为 uint16_t 排序键
```

**与 QLI 的核心差异**：

| 功能点 | QLI 实现 | LI 实现 |
|--------|----------|---------|
| 量化处理 | weight * qScale，最后 * kScale | 无 scale 计算 |
| 数据流 | FP16 -> FP32（带乘法）-> FP16 | FP16/BF16 -> FP32（纯转换）-> BF16 |
| 输出类型 | uint32_t 排序键 | uint16_t 排序键 |

**详细设计方案**：参见 [lightning_indexer_vf_design.md](./lightning_indexer_vf_design.md) 第 4.1.4.10 节

#### 4.1.4.11 代码移植与编译配置关键要点

**【新增】arch35 代码移植注意事项**：

在将 QLI 的 ascend950 代码移植到 LI 时，需要特别注意以下关键点：

| 类别 | 关键点 | 说明 |
|------|--------|------|
| 目录结构 | `op_kernel/arch35/` 目录创建 | 必须与 QLI 保持相同的目录层级 |
| 头文件路径 | 相对路径 `#include "../lightning_indexer_common.h"` | arch35 内文件引用上级目录 |
| 头文件路径 | VF 引用 `#include "../arch35/vf/xxx.h"` | service_vector.h 引用 vf 文件 |
| 命名空间 | `QLIKernel` → `LIKernel` | 全局替换 |
| 命名空间 | `QLICommon` → `LICommon` | 全局替换 |
| 类型名称 | `QLIType` → `LIType` | 模板类型替换 |
| 类型名称 | `QLITilingData` → `LITilingData` | Tiling 数据类型替换 |
| **VF 命名空间** | **保持 QLI 原有风格** | **不强制统一，直接参考 QLI 实现** |
| 编译选项 | `-mllvm -cce-vf-remove-membar=false` | VF 必需编译选项 |
| 条件编译 | `__CCE_AICORE__ == 310` | ascend950 平台判断 |

**【重要】VF 命名空间处理原则**：

移植时应**保持与 QLI 相同的命名空间使用方式**，不要强制统一：

| 文件 | QLI 使用方式 | LI 处理方式 |
|-----|-------------|------------|
| `vf_topk.h` | `MicroAPI::`（非限定） | **保持**，直接使用 MicroAPI:: |
| `vf_topk_16_gather.h` | `MicroAPI::`（非限定） | **保持**，直接使用 MicroAPI:: |
| `quant_lightning_indexer_vector1.h` | `AscendC::MicroAPI::`（完全限定） | **保持**，使用 AscendC::MicroAPI:: |
| `quant_lightning_indexer_topk.h` | 混合使用 | **保持**原有风格 |

**详细移植指南**：参见 [lightning_indexer_arch35_porting_guide.md](./lightning_indexer_arch35_porting_guide.md)

**核心文件清单**：

```
【需要创建的 arch35 文件】
op_kernel/arch35/
├── lightning_indexer_kernel.h           # Kernel 主类
├── lightning_indexer_service_cube.h     # AIC Cube 服务
├── lightning_indexer_service_vector.h   # AIV Vector 服务
└── vf/
    ├── lightning_indexer_topk.h         # TopK 封装类
    ├── lightning_indexer_vector1.h      # 加权归约 VF
    ├── vf_topk.h                        # uint32_t TopK VF
    └── vf_topk_16_gather.h              # uint16_t TopK VF

【需要修改的文件】
op_kernel/lightning_indexer.cpp          # 添加 arch35 条件分支
```

### 4.1.5 异常场景设计

| 异常场景              | 处理方式          | 处理层级 |
| --------------------- | ----------------- | -------- |
| actS1Size == 0        | 跳过该 batch 处理 | Kernel   |
| actS2Size == 0        | 跳过该 batch 处理 | Kernel   |
| S2 被切核时最后一个核 | 执行 LD 归约      | Kernel   |
| 无效输出位置          | 填充 -1           | Kernel   |

### 4.1.6 支持确定性计算设计

**说明**：不涉及，确定性计算与 QLI 保持一致。

### 4.1.7 精度分析及设计

**精度影响分析**：

| 变化点          | 精度影响     | 说明                                             |
| --------------- | ------------ | ------------------------------------------------ |
| FP8 → FP16/BF16 | **精度提升** | 去掉量化/反量化，保留更多有效位数                |
| 去掉 kScale     | **精度变化** | 原始计算本来就需要 kScale，LI 不需要是因为无量化 |
| ReLU 激活       | 无变化       | 保持一致                                         |

**格式一致性风险及应对**：

| 风险         | 描述                            | 应对                           |
| ------------ | ------------------------------- | ------------------------------ |
| 数据类型混合 | LI 使用 FP16/BF16，QLI 使用 FP8 | 明确区分 TilinkKey，运行时判定 |

### 4.1.8 性能分析及设计

**性能影响分析表**：

| 变更点                  | 性能影响       | 说明                     |
| ----------------------- | -------------- | ------------------------ |
| 去掉 dequant scale 加载 | **性能提升**   | 减少 GM → UB 数据搬运    |
| 去掉 kScale 乘累加      | **性能提升**   | 减少 Vector 计算量       |
| FP16/BF16 vs FP8        | **计算量增加** | 数据类型变大，但符合预期 |

**后续优化方向**：

- 如果性能不达标，可考虑进一步优化 UB 空间利用
- 可考虑将 Weights 预加载到 L1 减少重复搬运

### 4.1.9 训推一致性设计

**说明**：LI ascend950 主要用于推理场景，训练场景由 FP16/BF16 版本覆盖。

---

# 5 维测设计

## 5.1 可测试性

### 5.1.1 功能测试

| 测试场景                           | 验证点                               | 优先级 |
| ---------------------------------- | ------------------------------------ | ------ |
| ascend950 BSND FP16 基本功能       | TopK 索引正确性                      | P0     |
| ascend950 BSND BF16 基本功能       | TopK 索引正确性                      | P0     |
| ascend950 TND FP16 基本功能        | TopK 索引正确性                      | P0     |
| ascend950 PA_BSND FP16             | PageAttention 场景                   | P1     |
| ascend950 sparse_mode=0            | defaultMask 模式，无 mask 全 S2 参与 | P0     |
| ascend950 sparse_mode=3            | rightDownCausal 下三角 mask 场景     | P0     |
| ascend950 G=16/24 头数泛化         | 不同 G 值下的正确性                  | P1     |
| ascend950 vs ascend910b 结果一致性 | 相同输入输出一致                     | P0     |
| return_values=true                 | sparse_values 输出正确               | P1     |

### 5.1.2 精度测试

**CPU 参考实现构建方法**：

```python
def lightning_indexer_cpu_reference(query, key, weights, sparse_count=2048):
    # Query: [B, S1, N1, D]
    # Key: [B, S2, N2, D] (N2=1)
    # Weights: [B, S1, N1]

    # Step 1: Q @ K^T -> [B, S1, N1, S2]
    score = np.einsum('bsnd,bssd->bnsf', query, key)

    # Step 2: ReLU(score)
    score = np.maximum(score, 0)

    # Step 3: weights * score -> [B, S1, N1, S2]
    score = score * weights[:, :, :, None]

    # Step 4: Sum over N1 (G direction) -> [B, S1, S2]
    score = np.sum(score, axis=2)

    # Step 5: TopK -> [B, S1, sparse_count]
    indices = np.argsort(score, axis=-1)[:, :, -sparse_count:]

    return indices
```

### 5.1.3 边界条件测试

| 边界场景          | 预期行为 |
| ----------------- | -------- |
| S1 = 1            | 正常处理 |
| S2 = 1            | 正常处理 |
| sparse_count > S2 | 填充 -1  |
| G = 64 (LI 最大)  | 正常处理 |
| G = 48/32         | 正常处理 |
| G = 24            | 正常处理 |
| G = 16 (LI 最小)  | 正常处理 |
| B = 1             | 正常处理 |

## 5.2 可观察性

**cannsim 仿真验证方法**：

```bash
# 编译 ascend950 版本
bash build.sh --pkg --soc=ascend950 --ops=lightning_indexer -j16

# 运行 cannsim 仿真
./build_out/lightning_indexer ascend950 test_case
```

## 5.3 可维护性

### 5.3.1 关键风险与应对

| 风险             | 描述                          | 应对措施                      |
| ---------------- | ----------------------------- | ----------------------------- |
| 基本块大小不匹配 | ascend950 L0 大小约束理解偏差 | 参考 QLI ascend950 的实际配置 |
| 数据类型校验遗漏 | ascend950 错误接受了 FP8 类型 | 严格区分 TilinkKey            |
| 核间同步问题     | 去量化后同步时序变化          | 复用 QLI 验证过的同步机制     |

### 5.3.2 后续演进

- 后续可考虑支持 ascend950 上的 INT8 量化版本（如果需要）
- 后续可考虑进一步优化基本块大小提升性能

---

# 6 资料设计

| 资料类型                      | 是否涉及变更 | 变更内容                                     |
| ----------------------------- | ------------ | -------------------------------------------- |
| README.md                     | 是           | 更新支持平台表格，增加 ascend950             |
| aclnnLightningIndexer.md      | 否           | 无变更                                       |
| lightning_indexer_代码梳理.md | 是           | 新增 lightning_indexer_ascend950_代码梳理.md |
| 详细设计文档                  | 是           | 本文档                                       |