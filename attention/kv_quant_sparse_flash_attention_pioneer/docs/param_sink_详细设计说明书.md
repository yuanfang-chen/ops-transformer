# KvQuantSparseFlashAttentionPioneer 算子 Param Sink 特性详细设计说明书

<center>**修订记录**</center>

|    日期    | 修订版本 |          修改描述          |   作者    |
| :--------: | :------: | :------------------------: | :-------: |
| 2026-03-17 |   1.0    | Param Sink 特性详细设计初稿 | Claude Code |
| 2026-03-17 |   1.1    | 优化为模板方案：FLASH_DECODE→HAS_SINK 编译期分支 | Claude Code |
| 2026-03-17 |   1.2    | Sink KV 搬运内嵌到 IterateBmm1QSFA，消除独立 CopyInSinkKv | Claude Code |

# 1 关联需求

**需求背景**：

在长上下文推理场景中，Sparse Attention 通过选择性计算关键 KV 块来降低计算量。但研究表明，序列开头的若干 token（Attention Sink）对注意力分布具有锚定作用，丢弃这些 token 会导致生成质量显著下降。Param Sink 机制在稀疏注意力基础上引入固定数量的 **Sink Token**（128 个），在每次注意力计算时始终参与，等效于在 KV 序列开头拼接一段关键上下文，确保生成质量不受稀疏选择影响。

**功能定义**：

- 在每层注意力计算中，固定引入 128 个 Sink Token 的 KV 数据
- Sink KV 数据为 **BF16 非量化格式**，通过独立的 `key_sink`/`value_sink` 输入传入
- Sink KV **不经过 Vec0 反量化流程**，由 **Cube 侧（AIC）直接搬运**到 L1
- 仅在 **PA_BSND 布局**下支持
- Sink Token 数量固定为 128（恰好等于 s2BaseSize，即一个完整的 S2 基本块）

**设计原则**：

- **零框架侵入**：框架侧不修改 block_table、sparse_indices 等已有数据结构，仅传入 `key_sink`/`value_sink` 张量
- **最小 Kernel 侵入**：复用现有三级流水架构，在 S2 循环前插入独立 Sink 迭代
- **朴素实现**：优先保证功能正确性，不做极致性能优化
- **格式一致**：Sink KV 写入 L1 后的 NZ 格式与正常反量化 KV 完全一致，Cube 侧 BMM1/BMM2 逻辑无需修改

# 2 接口实现设计

## 2.1 PTA 接口实现

无变更。`key_sink`/`value_sink` 作为 OPTIONAL 输入，框架侧按需传入即可。

## 2.2 aclnn 接口实现

无变更。现有 aclnn 接口已包含 `key_sink`/`value_sink` 参数位。

## 2.3 算子信息库（OpDef）

**无变更**。OpDef 已预留 `key_sink`（输入 9，OPTIONAL，BF16）和 `value_sink`（输入 10，OPTIONAL，BF16）。

当前定义：
```cpp
this->Input("key_sink")
    .ParamType(OPTIONAL)
    .DataType({ge::DT_BF16, ge::DT_BF16})
    .Format({ge::FORMAT_ND, ge::FORMAT_ND})
    .AutoContiguous();
this->Input("value_sink")
    .ParamType(OPTIONAL)
    .DataType({ge::DT_BF16, ge::DT_BF16})
    .Format({ge::FORMAT_ND, ge::FORMAT_ND})
    .AutoContiguous();
```

**Sink KV 张量规格**：

| 参数名 | 输入/输出 | 描述 | 数据类型 | 数据格式 | Shape |
| :----- | :------- | :--- | :------- | :------- | :---- |
| key_sink | 输入（可选） | Sink Key，非量化 | BF16 | ND | [B, 128, 1, 576] |
| value_sink | 输入（可选） | Sink Value，非量化 | BF16 | ND | [B, 128, 1, 512] |

> 说明：MLA-absorb 模式下，key_sink 包含 nope(512)+rope(64) 共 576 维；value_sink 包含 nope(512) 维。由于 MLA 架构中 K 和 V 共享 nope 部分（即 key_sink[0:512] == value_sink），实际 Kernel 侧仅需加载 key_sink 即可同时满足 BMM1（使用全 576 维）和 BMM2（使用前 512 维 nope）的需求。

## 2.4 图模式设计

不涉及。

# 3 总体设计

## 3.1 交付方式

| 类型 | 描述 | 备注 |
| ---- | ---- | ---- |
| 交付内容 | Param Sink 特性代码 | 仅 PA_BSND 布局 |
| 代码承载 | kv_quant_sparse_flash_attention_pioneer 算子目录 | arch35 平台 |

## 3.2 交付件汇总

| **序号** | **交付件** | **是否需要** | **涉及变动** | **备注** |
| -------- | ---------- | ------------ | ------------ | -------- |
| 07 | 算子原型 | 否 | 否 | OpDef 已预留 key_sink/value_sink |
| 08 | OpDef定义 | 否 | 否 | 同上 |
| 09 | 算子tiling函数 | 是 | 是 | GenTilingKey 设置 HAS_SINK、校验 |
| 10 | 算子kernel实现 | 是 | 是 | 主循环插入 Sink 迭代、AIC 搬运 |
| 11 | 算子二进制配置 | 是 | 是 | 新增 HAS_SINK=1 的 TilingKey 组合 |
| 12 | inferShape/inferDataType | 否 | 否 | 输出 shape 不受 sink 影响 |

## 3.3 TilingKey 设计

**有变更**。将现有未使用的 `FLASH_DECODE` 模板参数重命名为 `HAS_SINK`，通过编译期模板特化控制 Sink 功能开关。

**现有 TilingKey 模板参数**（`*_template_tiling_key.h`）：

```cpp
ASCENDC_TPL_ARGS_DECL(KvQuantSparseFlashAttentionPioneer,
ASCENDC_TPL_BOOL_DECL(FLASH_DECODE, 0, 1),    // ← 从未使用，始终为 0
ASCENDC_TPL_UINT_DECL(LAYOUT_T, ...),
ASCENDC_TPL_UINT_DECL(KV_LAYOUT_T, ...),
ASCENDC_TPL_UINT_DECL(TEMPLATE_MODE, ...),
);
```

**变更后**：

```cpp
ASCENDC_TPL_ARGS_DECL(KvQuantSparseFlashAttentionPioneer,
ASCENDC_TPL_BOOL_DECL(HAS_SINK, 0, 1),        // ← 重命名，0=无Sink，1=有Sink
ASCENDC_TPL_UINT_DECL(LAYOUT_T, ...),
ASCENDC_TPL_UINT_DECL(KV_LAYOUT_T, ...),
ASCENDC_TPL_UINT_DECL(TEMPLATE_MODE, ...),
);
```

**新增 TilingKey 组合**（HAS_SINK=1 仅支持 PA_BSND）：

```cpp
ASCENDC_TPL_SEL(
    // 现有组合（HAS_SINK=0）保持不变
    ASCENDC_TPL_ARGS_SEL(
    ASCENDC_TPL_BOOL_SEL(HAS_SINK, 0),
    ASCENDC_TPL_UINT_SEL(LAYOUT_T, ASCENDC_TPL_UI_LIST, QSFA_LAYOUT_BSND),
    ASCENDC_TPL_UINT_SEL(KV_LAYOUT_T, ASCENDC_TPL_UI_LIST, QSFA_LAYOUT_BSND, QSFA_LAYOUT_PA_BSND),
    ASCENDC_TPL_UINT_SEL(TEMPLATE_MODE, ASCENDC_TPL_UI_LIST, V_TEMPLATE),
    ),

    ASCENDC_TPL_ARGS_SEL(
    ASCENDC_TPL_BOOL_SEL(HAS_SINK, 0),
    ASCENDC_TPL_UINT_SEL(LAYOUT_T, ASCENDC_TPL_UI_LIST, QSFA_LAYOUT_TND),
    ASCENDC_TPL_UINT_SEL(KV_LAYOUT_T, ASCENDC_TPL_UI_LIST, QSFA_LAYOUT_TND, QSFA_LAYOUT_PA_BSND),
    ASCENDC_TPL_UINT_SEL(TEMPLATE_MODE, ASCENDC_TPL_UI_LIST, V_TEMPLATE),
    ),

    // 【新增】HAS_SINK=1 组合（仅 PA_BSND）
    ASCENDC_TPL_ARGS_SEL(
    ASCENDC_TPL_BOOL_SEL(HAS_SINK, 1),
    ASCENDC_TPL_UINT_SEL(LAYOUT_T, ASCENDC_TPL_UI_LIST, QSFA_LAYOUT_BSND),
    ASCENDC_TPL_UINT_SEL(KV_LAYOUT_T, ASCENDC_TPL_UI_LIST, QSFA_LAYOUT_PA_BSND),
    ASCENDC_TPL_UINT_SEL(TEMPLATE_MODE, ASCENDC_TPL_UI_LIST, V_TEMPLATE),
    ),

    ASCENDC_TPL_ARGS_SEL(
    ASCENDC_TPL_BOOL_SEL(HAS_SINK, 1),
    ASCENDC_TPL_UINT_SEL(LAYOUT_T, ASCENDC_TPL_UI_LIST, QSFA_LAYOUT_TND),
    ASCENDC_TPL_UINT_SEL(KV_LAYOUT_T, ASCENDC_TPL_UI_LIST, QSFA_LAYOUT_PA_BSND),
    ASCENDC_TPL_UINT_SEL(TEMPLATE_MODE, ASCENDC_TPL_UI_LIST, V_TEMPLATE),
    ),
);
```

> 说明：HAS_SINK=1 时 KV_LAYOUT_T 仅允许 PA_BSND，因为 Sink 功能仅在 PagedAttention 布局下支持。

**模板参数链路变更**（`*_common.h`）：

```cpp
// 现有：
#define TEMPLATE_INTF \
    template <typename Q_T, typename KV_T, typename T, typename OUTPUT_T, bool isFd, bool isPa, ...>

// 变更后：
#define TEMPLATE_INTF \
    template <typename Q_T, typename KV_T, typename T, typename OUTPUT_T, bool hasSink, bool isPa, ...>

// CUBE_BLOCK_TRAITS_CONST_FIELDS 同步变更：
#define CUBE_BLOCK_TRAITS_CONST_FIELDS(X) \
    X(hasSink, bool, false) \    // ← 原 isFd
    X(isPa, bool, true) \
    ...
```

**编译期常量**：当 `hasSink=true` 时，Sink Token 数量固定为 128，作为编译期常量使用：

```cpp
constexpr uint32_t SINK_TOKEN_NUM = 128;  // 在 common.h 中定义
```

## 3.4 模板列表

| 规格 | 应用场景 | HAS_SINK | KV_LAYOUT_T | 备注 |
| ---- | -------- | -------- | ----------- | ---- |
| 现有组合 | 无 Sink 场景 | 0 | BSND/TND/PA_BSND | 不变 |
| 新增组合 | Sink 场景 | 1 | PA_BSND | 仅 PA 布局 |

## 3.5 代码结构设计

| 目录/文件 | 内容 | 变化点 |
| --------- | ---- | ------ |
| `op_kernel/*_template_tiling_key.h` | TilingKey 定义 | `FLASH_DECODE` → `HAS_SINK`；新增 HAS_SINK=1 的组合 |
| `op_kernel/*_common.h` | 模板宏定义 | `isFd` → `hasSink`；新增 `SINK_TOKEN_NUM` 常量 |
| `op_host/*_tiling.cpp` | Tiling 策略 | GenTilingKey 根据 key_sink 输入设置 HAS_SINK；新增校验 |
| `op_kernel/*_kernel_mla.h` | 主 Kernel 类 | Init 存储 keySinkGm 指针；ProcessMainLoop 用 `if constexpr (hasSink)` 插入 Sink 迭代 |
| `op_kernel/*_service_cube_mla.h` | Cube 服务 | 新增 `InitSinkKv()` + `keySinkGm` 成员；`IterateBmm1QSFA` 内嵌 sink KV 搬运；Dummy 类同步 |
| `op_kernel/*_service_vector_mla.h` | Vector 服务 | 新增 `ProcessVec0Sink()` 方法；Dummy 类同步 |
| `op_kernel/*.cpp` | Kernel 入口 | `FLASH_DECODE` → `HAS_SINK`；传递 key_sink 指针到 Init |

# 4 模板设计

## 4.1 Param Sink 计算模板

### 4.1.1 计算流程图

**核心思路**：Sink 迭代作为 S2 循环的第 0 次迭代插入，后续稀疏 KV 迭代从 s2LoopCount=1 开始（isUpdate=true）。

**正常块 vs Sink 块对比**：

```
┌─────────────────────────────────────────────────────────────────┐
│  正常稀疏 KV 块（现有逻辑）                                       │
│                                                                   │
│  AIV Vec0: GM(量化KV) → UB → 反量化 → L1                         │
│  AIC BMM1: L1(K) × L1(Q) → UB(QK^T)                             │
│  AIV Vec1: UB(QK^T) → Softmax → L1(P)                           │
│  AIC BMM2: L1(P) × L1(V) → UB(PV)                               │
│  AIV Vec2: UB(PV) → FlashUpdate → GM(output)                    │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  Sink 块（新增逻辑）                                              │
│                                                                   │
│  AIV Vec0Sink: 空操作（仅做核间同步信号）                          │
│  AIC IterateBmm1QSFA:                                            │
│    ① 加载 Q 到 L1（s2LoopCount==0，与正常路径相同）              │
│    ② WaitCrossCore（收到 AIV dummy 信号）                        │
│    ③ CopyToL1Nd2Nz: GM(BF16 sink KV) → L1（NZ 格式）【新增】   │
│    ④ MatmulK: L1(K_sink) × L1(Q) → L0C → UB(QK^T)              │
│  AIV Vec1: UB(QK^T) → Softmax(NoUpdate)    ← 逻辑不变           │
│  AIC BMM2: L1(P) × L1(V_sink) → UB(PV)    ← 逻辑不变           │
│  AIV Vec2: UB(PV) → DataCopy(首次)          ← 逻辑不变           │
└─────────────────────────────────────────────────────────────────┘
```

**三级流水时序图（含 Sink 迭代）**：

```
时间 →    T0(Sink)    T1(Sparse0)  T2(Sparse1)  T3(Sparse2)  T4           T5
AIC:   SinkBMM1    BMM1[0]      BMM1[1]      BMM1[2]      ...
                   SinkBMM2     BMM2[0]      BMM2[1]      BMM2[2]      ...
AIV:   Vec0Sink    Vec0[0]      Vec0[1]      Vec0[2]      ...
                   SinkVec1     Vec1[0]      Vec1[1]      Vec1[2]      ...
                                SinkVec2     Vec2[0]      Vec2[1]      ...

说明：
- T0: Sink 迭代作为 taskId=0，s2LoopCount=0（NoUpdate Softmax）
- T1+: 稀疏 KV 迭代，s2LoopCount=1,2,...（Update Softmax）
- Sink 迭代自然融入三级流水，无额外排空开销
```

**Softmax 状态管理**：

| 迭代 | s2LoopCount | isUpdate | Softmax 行为 |
| ---- | ----------- | -------- | ------------ |
| Sink | 0 | false | 初始化 max/sum，直接输出 exp(P) |
| Sparse[0] | 1 | true | 合并历史 max/sum，更新统计量 |
| Sparse[1] | 2 | true | 同上 |
| ... | ... | true | 同上 |
| Sparse[last] | s2LoopLimit | true | 最终归一化 ÷ sum |

### 4.1.2 Tiling 实现设计

#### 4.1.2.1 TilingData 设计

**无新增字段**。Sink 功能通过编译期模板参数 `HAS_SINK` 控制，Sink Token 数量（128）为编译期常量 `SINK_TOKEN_NUM`，无需在 TilingData 中传递。

> 设计优势：避免在 BaseParams → CVSharedParams → ConstInfo 的运行时传递链路中新增字段，减少 SSBuf 占用和参数解析开销。

#### 4.1.2.2 核间 tiling 策略

无变更。Sink 迭代不影响核间分配（每个核独立处理自己负责的 batch/S1 块的 sink 数据）。

#### 4.1.2.3 核内 tiling 策略

**GenTilingKey 变更**：

```cpp
void QSFAPMlaTiling::GenTilingKey()
{
    uint32_t layoutQuery = static_cast<uint32_t>(sfaaInfo_->qLayout);
    uint32_t layoutKV = static_cast<uint32_t>(sfaaInfo_->kvLayout);

    // 【变更】第一个参数从固定 0U 改为根据 sink 输入判定
    uint32_t hasSink = (sfaaInfo_->keySink.addr != nullptr) ? 1U : 0U;
    tilingKey_ = GET_TPL_TILING_KEY(hasSink, layoutQuery, layoutKV,
                                     perfMode_ == QSFAPerfMode::V_TEMPLATE_MODE);

    OP_LOGI(sfaaInfo_->opName, "QSFA tilingKey_: %lu, hasSink: %u.", tilingKey_, hasSink);
}
```

**启用判定逻辑**（在 `QSFAPInfoParser::Parse` 中）：

```
if (keySink 输入不为空 && valueSink 输入不为空):
    hasSink = true  // 通过 GenTilingKey 设置 HAS_SINK=1
else:
    hasSink = false  // GenTilingKey 设置 HAS_SINK=0，走原有路径
```

**参数校验**（在 `QSFAPTilingCheck::Process` 中新增）：

```
if (keySink 输入不为空):
    CHECK: kvLayout == PA_BSND          // 仅 PA 布局支持
    CHECK: key_sink.dtype == BF16       // 必须为 BF16
    CHECK: value_sink.dtype == BF16
    CHECK: key_sink.shape[1] == 128     // Sink Token 数量固定 128
    CHECK: key_sink.shape[2] == 1       // KV_N == 1
    CHECK: key_sink.shape[3] == 576     // D = nope + rope
    CHECK: value_sink.shape[1] == 128
    CHECK: value_sink.shape[2] == 1
    CHECK: value_sink.shape[3] == 512   // D = nope
```

**FillTiling**：无新增字段需要填充（Sink 信息完全由模板参数承载）。

### 4.1.3 Buffer 设计

#### 4.1.3.1 UB 空间的分配

无新增。Sink 迭代复用现有 bmm1Buffers / bmm2Buffers / softmax buffers。

#### 4.1.3.2 workspace 的分配

无新增。Sink 迭代不引入额外 workspace 需求。

#### 4.1.3.3 L1/L0 的分配

无新增。Sink KV 加载复用现有 `l1RightBuffers`（3-buffer，每块 s2BaseSize × dBaseSize × sizeof(BF16) = 128 × 576 × 2 = 147,456 字节）。

Sink KV 的 L1 布局与正常反量化 KV 完全一致：

```
l1RightBuffer 布局（NZ 格式）：
┌──────────────────────────────────────────┐
│  128 rows × 576 cols (BF16, NZ format)   │
│                                          │
│  [0, 512): nope 部分 → BMM2 读取为 V    │
│  [512, 576): rope 部分                   │
│  整体 576 维 → BMM1 读取为 K             │
└──────────────────────────────────────────┘
```

### 4.1.4 Kernel 设计

#### 4.1.4.1 入口函数变更（*.cpp）

**变更 1**：模板参数重命名 `FLASH_DECODE` → `HAS_SINK`。

```cpp
// 现有：
template<int FLASH_DECODE, int LAYOUT_T, int KV_LAYOUT_T, int TEMPLATE_MODE>
__global__ __aicore__ void kv_quant_sparse_flash_attention_pioneer(...)

// 变更后：
template<int HAS_SINK, int LAYOUT_T, int KV_LAYOUT_T, int TEMPLATE_MODE>
__global__ __aicore__ void kv_quant_sparse_flash_attention_pioneer(...)
```

**变更 2**：将 `key_sink` 指针从 `nullptr` 改为实际传递。

```cpp
// 现有代码（QSFA_OP_IMPL 宏内）：
op.Init(query, key, value, sparseIndices, keyScale, valueScale, blocktable,
    actualSeqLengthsQuery, actualSeqLengthsKV, nullptr, nullptr,    // ← key_sink, value_sink 传 nullptr
    attentionOut, user, tilingData, &tPipe);

// 变更后：
op.Init(query, key, value, sparseIndices, keyScale, valueScale, blocktable,
    actualSeqLengthsQuery, actualSeqLengthsKV, key_sink, value_sink,  // ← 传递实际指针
    attentionOut, user, tilingData, &tPipe);
```

#### 4.1.4.2 主 Kernel 类变更（*_kernel_mla.h）

**新增成员变量**：

```cpp
class KvQuantSparseFlashAttentionMla {
    // ... 现有成员 ...
    __gm__ uint8_t *keySinkGmPtr = nullptr;   // 【新增】Sink Key GM 指针
};
```

**Init 变更**：

```cpp
void Init(..., __gm__ uint8_t *key_sink, __gm__ uint8_t *value_sink, ...) {
    // ... 现有逻辑 ...
    if constexpr (hasSink) {
        this->keySinkGmPtr = key_sink;  // 【新增】仅 hasSink=true 时存储
    }
    // ... 后续逻辑不变 ...
}
```

**ComputeConstexpr 变更**：

无变更。`hasSink` 为编译期模板参数，无需从 TilingData 解析。

**InitGlobalBuffer 变更**：

```cpp
void InitGlobalBuffer(...) {
    // ... 现有逻辑 ...
    if constexpr (hasSink) {
        cubeBlock.InitSinkKv(keySinkGmPtr, constInfo);  // 【新增】传递 sink GM 指针给 Cube 服务
    }
}
```

**ProcessMainLoop 变更（核心）**：

```
ProcessMainLoop():
  // ... 现有分核信息计算 ...
  for (bnIdx = bN2StartIdx; bnIdx < bN2EndIdx; bnIdx++):
    ComputeParamBatch()
    ComputeS1LoopInfo()

    for (gS1Index = gs1LoopStartIdx; gS1Index < gS1LoopEnd; gS1Index++):
      if (notLastTwoLoop):
        ComputeParamS1()
        ComputeS2LoopInfo()

        // ========== 【新增】编译期 Sink 迭代插入 ==========
        if constexpr (hasSink):
          s2LoopLimit += 1    // S2 循环总次数 +1（Sink 占一次）

      for (s2LoopCount = 0; s2LoopCount <= s2LoopLimit; s2LoopCount++):
        if (notLastTwoLoop):
          RunInfo &runInfo1 = runInfo[taskId % 3]
          SetRunInfo(runInfo1, runParam, taskId, s2LoopCount, s2LoopLimit, ...)

          if ASCEND_IS_AIC:
            // AIC 侧：无需区分 sink/normal，IterateBmm1 内部处理
            cubeBlock.IterateBmm1(bmm1Buffers.Get(), l1RightBuffers.Get(),
                                  runInfo1, constInfo)
          else:
            // AIV 侧：sink 迭代走 dummy sync，正常迭代走 Vec0
            if constexpr (hasSink):
              if (s2LoopCount == 0):
                vecBlock.ProcessVec0Sink(l1RightBuffers.Get(), runInfo1, constInfo)
              else:
                vecBlock.ProcessVec0(l1RightBuffers.Get(), runInfo1, constInfo)
            else:
              vecBlock.ProcessVec0(l1RightBuffers.Get(), runInfo1, constInfo)

        // Vec1 / BMM2 / Vec2 逻辑完全不变
        if (taskId > 0 && notLast):
          // ... 现有 Vec1 + BMM2 逻辑 ...
        if (taskId > 1):
          // ... 现有 Vec2 逻辑 ...

        ++taskId
```

**关键设计点**：
1. `if constexpr (hasSink)`：编译期分支，`hasSink=false` 时 Sink 相关代码完全不编译，零运行时开销
2. **AIC 侧主循环无变化**：Sink KV 搬运内嵌在 `IterateBmm1QSFA` 中，主循环对 AIC 完全透明
3. **AIV 侧仅 Vec0 阶段区分**：`s2LoopCount == 0` 时走 `ProcessVec0Sink`（dummy sync），其余走 `ProcessVec0`
4. `s2LoopLimit += 1`：Sink 占用一次 S2 迭代，总迭代次数加 1
5. Softmax 自然适配：s2LoopCount==0 走 NoUpdate 路径，s2LoopCount>0 走 Update 路径
6. Vec2 的 FlashUpdate 也自然适配：s2LoopCount==0 走 DataCopy，s2LoopCount>0 走 FlashUpdate

**SetRunInfo 变更**：

```cpp
void SetRunInfo(RunInfo &runInfo, RunParamStr &runParam, int64_t taskId,
    int64_t s2LoopCount, int64_t s2LoopLimit, int64_t multiCoreInnerIdx)
{
    // 【新增】Sink 迭代的 s2RealSize 固定为 sinkTokenNum (128)
    // 正常迭代的 s2LoopCount 需要减去 sink 偏移
    // ... 现有逻辑 ...
}
```

**ComputeBmm1Tail 变更**：

```cpp
void ComputeBmm1Tail(RunInfo &runInfo, RunParamStr &runParam)
{
    // ... 现有 S1 相关逻辑不变 ...

    // S2 相关：
    if constexpr (hasSink) {
        if (runInfo.s2LoopCount == 0) {
            // 【新增】Sink 迭代：s2RealSize 固定为 SINK_TOKEN_NUM
            runInfo.s2RealSize = SINK_TOKEN_NUM;  // 128，编译期常量
            runInfo.s2AlignedSize = runInfo.s2RealSize;
        } else {
            // 正常迭代：s2LoopCount 减去 sink 偏移
            int64_t adjustedS2LoopCount = runInfo.s2LoopCount - 1;
            // ... 现有 s2RealSize 计算逻辑，使用 adjustedS2LoopCount ...
        }
    } else {
        // hasSink=false：完全走原有逻辑，无任何开销
        // ... 现有 s2RealSize 计算逻辑 ...
    }
}
```

#### 4.1.4.3 Cube 服务变更（*_service_cube_mla.h）

**新增方法 `InitSinkKv`**（仅传递 GM 指针）：

```cpp
__aicore__ inline void InitSinkKv(__gm__ uint8_t *keySink, const ConstInfo& constInfo) {
    if ASCEND_IS_AIC {
        if (keySink != nullptr) {
            keySinkGm.SetGlobalBuffer((__gm__ Q_T *)keySink);
        }
    }
}
```

**新增成员变量**：

```cpp
GlobalTensor<Q_T> keySinkGm;  // Sink Key GM tensor
```

**核心变更：`IterateBmm1QSFA` 内嵌 Sink KV 搬运**：

Sink KV 搬运不作为独立方法，而是内嵌在 `IterateBmm1QSFA` 的 `s2LoopCount == 0` 分支中。现有代码在该分支已经执行 Q 矩阵加载，Sink KV 搬运紧随其后，在 `WaitCrossCore` 之后、Matmul 之前插入。

```cpp
TEMPLATES_DEF_NO_DEFAULT __aicore__ inline void
QSFAMatmulService<TEMPLATE_ARGS>::IterateBmm1QSFA(
    Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf,
    Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &inputRightBuf,
    RunInfo &runInfo, ConstInfo &constInfo)
{
    Buffer<BufferType::L1> inputLeftBuf;
    // 左矩阵复用，S2的第一次循环加载左矩阵
    if (unlikely(runInfo.s2LoopCount == 0)) {  // 第一个基本块：搬运Q
        inputLeftBuf = l1QBuffers.Get();
        inputLeftBuf.Wait<HardEvent::MTE1_MTE2>();
        LocalTensor<Q_T> inputLeftTensor = inputLeftBuf.GetTensor<Q_T>();
        CopyToL1Nd2Nz<Q_T>(inputLeftTensor, this->queryGm.gmTensor[runInfo.queryOffset],
            runInfo.mRealSize, constInfo.dSize, constInfo.mm1Ka);
        inputLeftBuf.Set<HardEvent::MTE2_MTE1>();
    } else {
        inputLeftBuf = l1QBuffers.GetPre();
        inputLeftBuf.Set<HardEvent::MTE2_MTE1>();
    }

    // 核间同步：等待 AIV 信号（正常迭代=KV数据就绪，Sink迭代=dummy信号）
    inputRightBuf.WaitCrossCore();

    // ========== 【新增】Sink KV 搬运：AIC 直接加载 BF16 sink KV 到 L1 ==========
    if constexpr (hasSink) {
        if (unlikely(runInfo.s2LoopCount == 0)) {
            // key_sink shape: [B, 128, 1, 576]
            int64_t sinkOffset = runInfo.boIdx * SINK_TOKEN_NUM * dBaseSize;
            LocalTensor<Q_T> inputRightTensor = inputRightBuf.GetTensor<Q_T>();
            CopyToL1Nd2Nz<Q_T>(inputRightTensor, keySinkGm[sinkOffset],
                SINK_TOKEN_NUM, dBaseSize, constInfo.mm1Ka);
        }
    }
    // ========== Sink KV 搬运结束 ==========

    inputLeftBuf.Wait<HardEvent::MTE2_MTE1>();
    // ... 后续 Matmul + Fixpipe 逻辑完全不变 ...
}
```

**设计要点**：

1. **位置选择**：Sink KV 搬运插入在 `WaitCrossCore()` 之后、`MatmulK` 之前。此时 AIV 已发出 dummy 信号，L1 buffer 可用
2. **复用 `CopyToL1Nd2Nz`**：与 Q 矩阵加载使用相同的 ND→NZ 转换函数，保证 L1 NZ 格式一致
3. **编译期消除**：`if constexpr (hasSink)` 确保 `hasSink=false` 时无任何额外代码
4. **运行时条件**：`s2LoopCount == 0` 仅在第一个基本块触发，与 Q 加载条件一致
5. **无独立方法**：不引入 `CopyInSinkKv`，Sink 搬运作为 BMM1 流程的自然扩展，减少方法调用开销

> 说明：`CopyToL1Nd2Nz` 将 GM 上的 ND 格式 BF16 数据转换为 L1 上的 NZ 格式。Sink KV 写入 `inputRightBuf` 后，后续 `MatmulK` 读取 `inputRightBuf.GetTensor<Q_T>()` 作为右矩阵 K，与正常反量化 KV 路径完全一致。BMM2 同理，从同一 L1 buffer 的 nope 部分（前 512 维）读取 V。

**Dummy 类同步**：

```cpp
class QSFAMatmulServiceDummy {
    // ... 现有空方法 ...
    __aicore__ inline void InitSinkKv(__gm__ uint8_t *keySink, const ConstInfo& constInfo) {}  // 【新增】
};
```

> 注意：由于 `CopyInSinkKv` 不再作为独立方法存在，Dummy 类无需添加对应空方法。仅需 `InitSinkKv` 的空实现。

#### 4.1.4.4 Vector 服务变更（*_service_vector_mla.h）

**新增方法 `ProcessVec0Sink`**：

```cpp
__aicore__ inline void ProcessVec0Sink(
    Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputL1,
    const RunInfo &runInfo, ConstInfo &constInfo);
```

实现伪代码：

```
ProcessVec0Sink(outputL1, runInfo, constInfo):
  // Sink 迭代中，AIV 不需要加载/反量化 KV 数据（AIC 直接搬运）
  // 仅需维护核间同步协议
  outputL1.WaitCrossCore()    // 等待 AIC 释放 L1 buffer
  outputL1.SetCrossCore()     // 通知 AIC：L1 buffer 可用（AIC 将自行加载 sink KV）
```

> 说明：核间同步协议要求 AIV 和 AIC 的 WaitCrossCore/SetCrossCore 调用必须成对出现。即使 AIV 不写入数据，也必须执行同步操作以避免死锁。

**Dummy 类同步**：

```cpp
class QSFAVectorServiceDummy {
    // ... 现有空方法 ...
    __aicore__ inline void ProcessVec0Sink(
        Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputL1,
        const RunInfo &runInfo, ConstInfo &constInfo) {}  // 【新增】
};
```

#### 4.1.4.5 运行时参数变更（util_regbase.h）

**无变更**。Sink 功能完全由编译期模板参数 `hasSink` 控制：

- `hasSink`：模板 bool 参数，编译期确定，无需运行时传递
- `SINK_TOKEN_NUM`：编译期常量（128），定义在 `*_common.h`
- CVSharedParams：无新增字段（不增加 SSBuf 占用）
- ConstInfo：无新增字段

> 设计优势：相比运行时方案（需在 BaseParams → CVSharedParams → ConstInfo 链路中逐层传递 sinkTokenNum），模板方案零运行时开销，且 `hasSink=false` 时 Sink 相关代码完全不编译。

#### 4.1.4.6 核间同步协议

**正常稀疏块的核间同步时序（不变）**：

```
AIV                              AIC
 │                                │
 ├─ Vec0: 加载量化KV到L1 ────────►│
 │  outputL1.SetCrossCore()       │  inputRightBuf.WaitCrossCore()
 │                                ├─ BMM1: L1(K) × L1(Q) → UB
 │                                │  outputBuf.SetCrossCore()
 │  bmm1ResBuf.WaitCrossCore() ◄──┤
 ├─ Vec1: Softmax                 │
 │  outputBuf.SetCrossCore() ────►│
 │                                ├─ BMM2: L1(P) × L1(V) → UB
 │                                │  outputBuf.SetCrossCore()
 │  bmm2ResBuf.WaitCrossCore() ◄──┤
 ├─ Vec2: FlashUpdate + 写回      │
```

**Sink 块的核间同步时序（新增）**：

```
AIV                              AIC
 │                                │
 ├─ Vec0Sink: 空操作              │
 │  outputL1.WaitCrossCore()      │
 │  outputL1.SetCrossCore() ─────►│  IterateBmm1QSFA:
 │                                │    加载 Q 到 L1（s2LoopCount==0）
 │                                │    inputRightBuf.WaitCrossCore()  ← 收到 AIV dummy 信号
 │                                │    CopyToL1Nd2Nz(sink KV → L1)   ← 【新增】AIC 自行搬运
 │                                │    MatmulK(Q × K_sink)
 │                                │    Fixpipe(L0C → UB)
 │                                │  outputBuf.SetCrossCore()
 │  bmm1ResBuf.WaitCrossCore() ◄──┤
 ├─ Vec1: Softmax (NoUpdate)      │  ← 逻辑完全不变
 │  outputBuf.SetCrossCore() ────►│
 │                                ├─ BMM2: L1(P) × L1(V_sink) → UB  ← V = K[0:512]
 │                                │  outputBuf.SetCrossCore()
 │  bmm2ResBuf.WaitCrossCore() ◄──┤
 ├─ Vec2: DataCopy (首次)         │  ← 逻辑完全不变
```

**关键差异**：仅 Stage 0 的 KV 数据来源不同。正常迭代由 AIV Vec0 加载量化 KV 并反量化写入 L1；Sink 迭代由 AIC 在 `IterateBmm1QSFA` 内部直接从 GM 加载 BF16 sink KV 到 L1。Stage 1（Softmax）和 Stage 2（BMM2+Vec2）逻辑完全复用。

### 4.1.5 异常场景设计

| 异常场景 | 处理方式 | 处理层级 |
| -------- | -------- | -------- |
| key_sink 为空但 value_sink 非空（或反之） | Tiling 校验报错，拒绝执行 | Host Tiling |
| key_sink shape 不符合 [B, 128, 1, 576] | Tiling 校验报错 | Host Tiling |
| value_sink shape 不符合 [B, 128, 1, 512] | Tiling 校验报错 | Host Tiling |
| 非 PA_BSND 布局下传入 sink 输入 | Tiling 校验报错 | Host Tiling |
| key_sink/value_sink 数据类型非 BF16 | Tiling 校验报错 | Host Tiling |
| key_sink/value_sink 均为空 | HAS_SINK=0，编译走无 Sink 模板，原有路径完全不变 | Host Tiling |
| sparseBlockCount=0 且 sink 开启 | 仅计算 sink 块，s2LoopLimit=0（HAS_SINK=1 模板内处理） | Kernel |

### 4.1.6 支持确定性计算设计

不涉及。Sink 迭代使用与正常迭代相同的 Matmul/Softmax 路径，确定性特性不受影响。

### 4.1.7 精度分析及设计

**精度影响分析**：

| 环节 | 影响 | 说明 |
| ---- | ---- | ---- |
| Sink KV 加载 | 无精度损失 | BF16 直接搬运，无量化/反量化 |
| BMM1 (Q @ K_sink^T) | 与正常路径一致 | 相同 Matmul 实现，相同 L1 NZ 格式 |
| Softmax | 与正常路径一致 | 相同 VF 微核函数 |
| BMM2 (P @ V_sink) | 与正常路径一致 | V_sink = key_sink[0:512]，相同 Matmul |
| FlashUpdate | 与正常路径一致 | Sink 作为首次迭代，后续迭代正常合并 |

**格式一致性风险**：

Sink KV 通过 `CopyToL1Nd2Nz` 转换为 NZ 格式，与正常反量化 KV 写入 L1 的 `CopyOutKvUb2L1`（也是 NZ 格式）完全一致。BMM1/BMM2 无法区分数据来源，格式兼容性有保证。

**精度验证标准**：

```
allclose(output_with_sink, reference_output, atol=1e-3, rtol=1e-3)
```

### 4.1.8 性能分析及设计

**性能影响分析**：

| 项目 | 影响 | 说明 |
| ---- | ---- | ---- |
| 额外计算量 | +1 次 S2 迭代 | 128 个 sink token 的 BMM1 + Softmax + BMM2 |
| 额外访存 | +1 次 L1 加载 | 128 × 576 × 2B = 144KB（AIC 直接搬运） |
| 流水效率 | 无损失 | Sink 迭代自然融入三级流水 |
| 核间同步 | +1 次 dummy sync | AIV Vec0Sink 仅做 Wait/Set，开销极小 |

**性能估算**：假设原始稀疏 KV 有 N 个 S2 块，Sink 增加 1 个块，总迭代次数从 N 变为 N+1。性能开销约为 1/(N+1)，当 N 较大时（如 N=32），开销约 3%。

**后续优化方向**（本次不实现）：
- Sink KV 可缓存在 L1 中跨 S1 迭代复用（当前每次 S1 迭代重新加载）
- Sink BMM1 可与首个稀疏块的 Vec0 并行（当前已通过流水实现）

### 4.1.9 训推一致性设计

不涉及。Param Sink 仅用于推理场景。

# 5 维测设计

## 5.1 可测试性

### 5.1.1 功能测试

| 测试场景 | 验证点 | 优先级 |
| -------- | ------ | ------ |
| Sink 开启 + PA_BSND + BF16 Q + FP8 KV | 基本功能正确性 | P0 |
| Sink 开启 + 多 batch (B>1) | 多 batch 下 sink 偏移计算正确 | P0 |
| Sink 开启 + sparseBlockCount=1 | 仅 1 个稀疏块 + 1 个 sink 块 | P0 |
| Sink 开启 + sparseBlockCount=0 | 仅 sink 块，无稀疏块 | P1 |
| Sink 关闭（key_sink=None） | 回归：与修改前结果完全一致 | P0 |
| Sink 开启 + 多核 (usedCoreNum>1) | 多核分配下 sink 计算正确 | P0 |
| Sink 开启 + HiFloat8 KV 类型 | 不同量化类型兼容 | P1 |

### 5.1.2 精度测试

**CPU 参考实现构建方法**：

```python
def reference_attention_with_sink(query, key_quant, value_quant, key_sink, value_sink,
                                   sparse_indices, scale, block_table):
    """
    query:          [B, S1, N, 576] BF16
    key_quant:      [Bn, Bs, 1, 672] FP8 (量化 KV)
    value_quant:    同 key_quant
    key_sink:       [B, 128, 1, 576] BF16
    value_sink:     [B, 128, 1, 512] BF16
    sparse_indices: [B, S1, 1, K] INT32
    """
    for b in range(B):
        for s1 in range(S1):
            # 1. Sink 部分（BF16，无反量化）
            k_sink = key_sink[b]                    # [128, 576]
            v_sink = key_sink[b, :, :, :512]        # [128, 512] (nope part)
            score_sink = query[b, s1] @ k_sink.T * scale  # [N, 128]

            # 2. 稀疏部分（反量化后计算）
            indices = sparse_indices[b, s1, 0]      # [K]
            k_sparse = dequant(gather(key_quant, indices, block_table))  # [K, 576]
            v_sparse = dequant(gather(value_quant, indices, block_table))[:, :512]  # [K, 512]
            score_sparse = query[b, s1] @ k_sparse.T * scale  # [N, K]

            # 3. 拼接后统一 Softmax
            scores = concat(score_sink, score_sparse, dim=-1)  # [N, 128+K]
            weights = softmax(scores, dim=-1)

            # 4. 加权求和
            values = concat(v_sink, v_sparse, dim=0)  # [128+K, 512]
            output[b, s1] = weights @ values           # [N, 512]

    return output
```

### 5.1.3 边界条件测试

| 边界场景 | 预期行为 |
| -------- | -------- |
| S1=1（单 query） | 正常计算，sink + sparse |
| sparseBlockCount=0 | 仅计算 sink 块 |
| actualS2Size < 128 | sink 仍计算完整 128 token（sink 不受 actualS2Size 限制） |
| 多核且某核无工作量 | 该核跳过 sink 和 sparse |
| key_sink 全零 | 正常计算，softmax 后 sink 权重趋近均匀 |

## 5.2 可观察性

通过调试日志（`[INFO][ParamSink]` 前缀）观察：
- Sink 迭代是否被正确触发
- Sink KV 的 GM 偏移是否正确
- s2LoopLimit 是否正确加 1
- Softmax 统计量（max/sum）在 sink 迭代后的值

## 5.3 可维护性

### 5.3.1 关键风险与应对

| 风险 | 描述 | 应对措施 |
| ---- | ---- | -------- |
| L1 NZ 格式不一致 | CopyToL1Nd2Nz 与 CopyOutKvUb2L1 的 NZ 布局可能存在细微差异 | 通过 DumpTensor 对比两种路径的 L1 数据 |
| 核间同步死锁 | Vec0Sink 的 dummy sync 与 AIC 的 CopyInSinkKv 时序不匹配 | 严格遵循 WaitCrossCore/SetCrossCore 配对原则 |
| s2LoopCount 偏移错误 | Sink 占用 s2LoopCount=0 后，稀疏迭代的 s2LoopCount 需要调整 | 在 GetRealCmpS2Idx 等函数中使用 adjustedS2LoopCount |
| MLA nope 假设 | 设计假设 key_sink[0:512] == value_sink | 文档明确约束，Tiling 校验 shape |

### 5.3.2 后续演进

- 支持可变 Sink Token 数量（当前固定 128）
- 支持非 PA 布局（BSND/TND）
- Sink KV L1 缓存复用（跨 S1 迭代不重复加载）
- 支持量化 Sink KV（当前仅 BF16）

# 6 资料设计

| 资料类型 | 是否涉及变更 | 变更内容 |
| -------- | ------------ | -------- |
| README.md | 是 | 新增 Param Sink 功能说明、key_sink/value_sink 参数描述 |
| 算子代码梳理.md | 是 | 更新 Sink 相关数据流和主循环描述 |
| API 文档 | 是 | 补充 key_sink/value_sink 的 shape/dtype 约束 |
