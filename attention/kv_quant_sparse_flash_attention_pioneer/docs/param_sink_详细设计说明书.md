# KvQuantSparseFlashAttentionPioneer 算子 Param Sink 特性详细设计说明书

<center>**修订记录**</center>

|    日期    | 修订版本 |          修改描述          |   作者    |
| :--------: | :------: | :------------------------: | :-------: |
| 2026-03-18 |   1.0    | Param Sink 特性详细设计初稿 | AI Assistant |
| 2026-03-18 |   1.1    | TilingKey 控制开关；Vec0 搬运；MLA 复用+batch 共享 | AI Assistant |
| 2026-03-18 |   1.2    | 修正 key_sink/value_sink shape：(128, N1, 576)/(128, N1, 512)，value_sink nope 复用 key_sink | AI Assistant |
| 2026-03-18 |   1.3    | Sink 搬运改为 AIC BMM1 阶段（复用 CopyToL1Nd2Nz），AIV Vec0 仅 pass-through | AI Assistant |

# 1 关联需求

| **需求编号** | 需求标题 | 需求链接 | 概要设计链接 | 修订版本 |
| ------------ | -------- | -------- | ------------ | -------- |
| - | Param Sink 固定 Sink Token 参与注意力计算 | - | - | 1.3 |

**需求背景**：
在长上下文推理场景中，Sparse Attention 通过选择性计算关键 KV 块来降低计算量。然而，模型的初始 token（通常为系统提示、任务指令等）对所有后续 token 的注意力计算至关重要。Param Sink 机制引入固定数量的 **Sink Token**（128 个），在注意力计算时始终参与，等效于在 KV 序列开头拼接一段关键上下文，确保模型不会"遗忘"关键前文信息。

**功能定义**：
- 在每层注意力计算中，固定引入 128 个 Sink Token 的 KV 数据
- Sink KV 数据为 **BF16 非量化格式**，通过 `key_sink`（shape [128, N1, 576]）和 `value_sink`（shape [128, N1, 512]）输入传入。MLA 架构下 value_sink 的 nope 部分（512 维）物理上复用 key_sink 的 nope 数据
- Sink KV **不经过 AIV Vec0 反量化流程**，由 **AIC 侧在 BMM1 阶段直接搬运 BF16 数据**到 L1（复用 CopyToL1Nd2Nz，跳过 AIV 反量化流程）
- 所有 batch 共用同一份 Sink KV 数据（shape 无 B 维度）
- 仅在 **PA_BSND 布局**下支持
- 功能关闭时（key_sink 为 nullptr），行为与修改前完全等价

**设计原则**：
- **零框架侵入**：框架侧不修改 block_table、sparse_indices 等已有数据结构，仅传入 `key_sink` 张量
- **最小 Kernel 侵入**：复用现有三级流水架构，在 S2 循环前插入独立 Sink 迭代；Sink KV 搬运由 AIC 在 BMM1 阶段完成（复用 CopyToL1Nd2Nz），AIV Vec0 仅做 CrossCore pass-through
- **TilingKey 编译期控制**：复用已有的 `FLASH_DECODE`（当前未使用）模板参数位作为 `HAS_SINK` 标志，通过 TilingKey 区分，**不修改 TilingData 结构**
- **MLA 复用**：value_sink 的 nope 部分（512 维）物理上复用 key_sink 的 nope 数据。Kernel 中只需搬运 key_sink 到 L1，BMM1 取全部 nope+rope=576 维做 Q×K^T，BMM2 从同一份 L1 数据中取 nope=512 维做 P×V
- **格式一致**：Sink KV 写入 L1 后的 NZ 格式与正常反量化 KV 完全一致，Cube 侧 BMM1/BMM2 逻辑无需修改

# 2 接口实现设计

## 2.1 PTA接口实现
无变更。框架侧已有 `key_sink`/`value_sink` 参数传入能力。

## 2.2 aclnn接口实现
无变更。OpDef 中已预注册 `key_sink`（index 9）和 `value_sink`（index 10）为 OPTIONAL 输入。

| 参数名 | 输入/输出 | 描述 | 使用说明 | 数据类型 | 数据格式 | 维度 |
| :----- | :------- | :--- | :------- | :------- | :------- | :--- |
| key_sink | 输入 | Sink Token 的 Key 数据 | 可选，PA_BSND 时有效 | BF16 | ND | [sink_token_num, N1, D] = [128, N1, 576] |
| value_sink | 输入 | Sink Token 的 Value 数据（nope 部分复用 key_sink） | 可选，PA_BSND 时有效 | BF16 | ND | [sink_token_num, N1, D_nope] = [128, N1, 512] |

**说明**：
- **Shape 定义**：
  - `key_sink` shape 为 [128, N1, 576]，其中 N1 为 KV head 数量维度（MLA 下 N1=1，即 KV 在 head 维度共享），D=576（nope 512 + rope 64）
  - `value_sink` shape 为 [128, N1, 512]，其中 D_nope=512（仅 nope 部分）
- **MLA nope 复用**：value_sink 的 nope 部分（512 维）物理上复用 key_sink 的前 512 维数据。即 `value_sink[:, :, :512] == key_sink[:, :, :512]`。因此 Kernel 中只需搬运 key_sink 到 L1，BMM2 自然从同一份 L1 NZ 数据中取 nope 512 维做 P×V
- `key_sink` 的 D 维度为 576（nope 512 + rope 64），与 Q 的 D 维度一致，已是 BF16 格式
- BMM1 使用全部 576 维（nope+rope）做 Q×K^T，BMM2 使用 nope 512 维做 P×V
- **所有 batch 共享**：shape 无 B 维度，所有 batch 共用同一份 Sink KV 数据，GM 偏移与 bIdx 无关
- sink_token_num 固定为 128，与 s2BaseSize 一致
- 当传 None 时等价于功能关闭

## 2.3 算子信息库
无变更。OpDef 中 `key_sink`/`value_sink` 已注册。

## 2.4 图模式设计
不涉及。

# 3 总体设计

## 3.1 交付方式
| 类型 | 描述 | 备注 |
| ---- | ---- | ---- |
| 交付内容 | Param Sink 功能代码 | kernel 侧 + tiling 侧 |
| 代码承载 | kv_quant_sparse_flash_attention_pioneer | 原算子目录 |

## 3.2 交付件汇总
| **序号** | **交付件** | **是否需要** | **涉及变动** | **备注** |
| -------- | ---------- | ------------ | ------------ | -------- |
| 01 | pta接口适配 | 否 | 否 | |
| 02 | pta接口文档 | 否 | 否 | |
| 03 | aclnn接口适配 | 否 | 否 | |
| 04 | aclnn接口文档 | 否 | 否 | |
| 05 | GE图模式适配 | 否 | 否 | |
| 06 | AclGraph图模式适配 | 否 | 否 | |
| 07 | 算子原型 | 否 | 否 | OpDef 已预注册 |
| 08 | OpDef定义 | 否 | 否 | 已有 key_sink/value_sink |
| 09 | 算子tiling函数 | 是 | 是 | GenTilingKey 中设置 HAS_SINK；**TilingData 不修改** |
| 10 | 算子kernel实现 | 是 | 是 | 主循环新增 sink 迭代、AIC BMM1 搬运 sink KV、Vec0 pass-through |
| 11 | 算子二进制配置 | 否 | 否 | |
| 12 | inferShape/inferDataType | 否 | 否 | 输出 shape 不变 |
| 13 | 图融合pass | 否 | 否 | |

## 3.3 TilingKey设计

**核心变更**：复用 `FLASH_DECODE` 模板参数位作为 `HAS_SINK` 标志。

当前 `FLASH_DECODE` 在 kernel 代码中从未被使用（`isFd` 在所有 kernel 文件中无任何引用），且 TilingKey 中恒为 0。本方案将其语义重定义为 `HAS_SINK`：

| 变更前 | 变更后 | 说明 |
| ------ | ------ | ---- |
| `FLASH_DECODE = 0` | `HAS_SINK = 0` | 不启用 Sink（原有行为） |
| `FLASH_DECODE = 1`（未使用） | `HAS_SINK = 1` | 启用 Sink |

**template_tiling_key.h 变更**：
```cpp
// 变更前
ASCENDC_TPL_BOOL_DECL(FLASH_DECODE, 0, 1),

// 变更后
ASCENDC_TPL_BOOL_DECL(HAS_SINK, 0, 1),
```

**TPL_SEL 变更**——新增 HAS_SINK=1 的合法组合（仅 PA_BSND 场景）：
```cpp
ASCENDC_TPL_SEL(
    // 原有组合：HAS_SINK=0
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

    // 新增组合：HAS_SINK=1，仅限 PA_BSND
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

**Tiling 侧 GenTilingKey 变更**：
```cpp
// 变更前
tilingKey_ = GET_TPL_TILING_KEY(0U, layoutQuery, layoutKV, perfMode_ == QSFAPerfMode::V_TEMPLATE_MODE);

// 变更后
uint32_t hasSink = (sfaaInfo_->keySinkExists && sfaaInfo_->kvLayout == QSFALayout::PA_BSND) ? 1U : 0U;
tilingKey_ = GET_TPL_TILING_KEY(hasSink, layoutQuery, layoutKV, perfMode_ == QSFAPerfMode::V_TEMPLATE_MODE);
```

**优势**：
- **编译期分支**：Kernel 中通过 `if constexpr (isFd)` 或模板特化判断是否启用 Sink，零运行时开销
- **不修改 TilingData**：无需新增字段，sink_token_num 固定为 128，编译期可知
- **向后兼容**：HAS_SINK=0 时，编译器完全消除 Sink 相关代码

## 3.4 模板列表

| 变更前 | 变更后 | 说明 |
| ------ | ------ | ---- |
| `template<int FLASH_DECODE, int LAYOUT_T, int KV_LAYOUT_T, int TEMPLATE_MODE>` | `template<int HAS_SINK, int LAYOUT_T, int KV_LAYOUT_T, int TEMPLATE_MODE>` | 入口函数模板参数重命名 |
| `isFd`（_common.h 中模板参数名） | `isFd`（语义变为 hasSink，保留名称以最小化变更量，或可重命名为 `hasSink`） | 模板参数传递 |

**关于 isFd 命名**：`_common.h` 中的 `CUBE_BLOCK_TRAITS_CONST_FIELDS` 定义了 `isFd` 作为模板参数名。有两种选择：
- **方案 A（最小变更）**：保留 `isFd` 名称，仅改变其语义（由"是否 FlashDecode"变为"是否有 Sink"），在注释中说明
- **方案 B（语义清晰）**：将 `isFd` 重命名为 `hasSink`，涉及 `_common.h`、`_service_cube_mla.h`、`_service_vector_mla.h` 中所有引用

建议采用**方案 B**，虽变更文件稍多但语义清晰，避免后续维护困惑。

## 3.5 代码结构设计
| 目录 | 内容 | 变化点 | 需求编号 | 需求标题 |
| ---- | ---- | ------ | -------- | -------- |
| op_kernel/*_template_tiling_key.h | TilingKey 声明 | FLASH_DECODE → HAS_SINK，新增 HAS_SINK=1 合法组合 | - | Param Sink |
| op_host/*_tiling.cpp | Tiling 策略 | GenTilingKey 中根据 key_sink 设置 HAS_SINK | - | Param Sink |
| op_kernel/*_common.h | 模板参数定义 | isFd → hasSink（重命名） | - | Param Sink |
| op_kernel/*.cpp | 入口函数 | FLASH_DECODE → HAS_SINK；传入 key_sink（替换 nullptr） | - | Param Sink |
| op_kernel/*_kernel_mla.h | Kernel 主类 | 新增 sinkKvGm GM 指针、Sink 迭代逻辑、传递 sinkKvGm 给 Cube 服务 | - | Param Sink |
| op_kernel/*_service_cube_mla.h | Cube 服务 | IterateBmm1 新增 sink BF16 GM→L1 搬运（CopyToL1Nd2Nz） | - | Param Sink |
| op_kernel/*_service_vector_mla.h | Vector 服务 | ProcessVec0 新增 sink pass-through 分支（WaitCrossCore + SetCrossCore） | - | Param Sink |

**注意**：**不修改 TilingData（tiling.h）**，**不修改 util_regbase.h**（ConstInfo/CVSharedParams/RunInfo 无需新增字段，hasSink 通过编译期模板参数判定）。

# 4 模板设计

## 4.1 Param Sink 计算模板

### 4.1.1 计算流程图

**正常块 vs Sink 块数据通路对比**：

```
正常 Sparse KV 块（现有流程）：
  AIV Vec0: GM(KV_FP8) → UB → 反量化(VF) → NZ → L1
  AIC BMM1: L1(Q) × L1(K_NZ)^T → UB(scores)
  AIV Vec1: Softmax(scores) → P_NZ → L1
  AIC BMM2: L1(P) × L1(V_NZ) → UB(partial_out)
  AIV Vec2: FlashUpdate → attentionOut

Sink KV 块（新增流程，AIC 搬运）：
  AIV Vec0: L1 pass-through (WaitCrossCore → SetCrossCore)   ← 不搬运数据，仅释放 L1 buffer
  AIC BMM1: GM(sink_BF16) → L1(NZ) → Q×K^T → UB(scores)    ← AIC 负责 GM→L1 搬运 + BMM1
  AIV Vec1: Softmax(scores) → P_NZ → L1                     ← 无变更
  AIC BMM2: L1(P) × L1(V_sink_NZ) → UB(partial_out)         ← 无变更（MLA 复用同一份 L1 数据）
  AIV Vec2: FlashUpdate → attentionOut                       ← 无变更
```

**MLA nope 复用说明**：
- key_sink shape [128, N1, 576]（nope 512 + rope 64），value_sink shape [128, N1, 512]（仅 nope），MLA 下 N1=1
- value_sink 的 nope 512 维物理上复用 key_sink 的前 512 维数据，因此 Kernel 中**只需搬运 key_sink**
- AIC BMM1 阶段搬运 key_sink 数据到 L1（复用 CopyToL1Nd2Nz），L1 中 nope(512)+rope(64) 的 NZ 布局同时服务于 BMM1（取 576 维做 Q×K^T）和 BMM2（取 nope 512 维做 P×V）
- 与正常 KV 的处理方式完全一致（正常 KV 也是 key==value 同一份数据的不同 D 切片）

**所有 batch 共享说明**：
- Sink KV 的 GM 偏移与 bIdx 无关，每个 batch/每个 Q 行都读取同一份 Sink 数据
- 计算 GM 偏移时：`sinkOffset = 0`（固定），不乘 bIdx

**三级流水时序图（含 Sink 迭代）**：

```
时间 →  T_sink0    T_sink1    T0         T1         T2        ...
AIC:  BMM1_sink  BMM2_sink  BMM1[0]    BMM1[1]    BMM1[2]   ...
                            BMM2[0]    BMM2[1]    ...
AIV:  Vec0_sink  Vec1_sink  Vec0[0]    Vec0[1]    Vec0[2]   ...
                            Vec1[0]    Vec1[1]    ...
                 Vec2_sink              Vec2[0]    Vec2[1]   ...
```

**说明**：
- Sink 迭代在正常 S2 循环之前执行，作为 taskId=0 的特殊迭代
- Vec0_sink：AIV 执行 pass-through（WaitCrossCore → SetCrossCore，不搬运数据），AIC 在 BMM1 阶段完成 GM→L1 搬运 + 矩阵乘
- Sink 迭代完成后，正常 S2 循环从 taskId=1 开始（s2LoopCount=0 的 Softmax 需要走 update 路径，因为已有 sink 的 max/sum 状态）
- Sink 的 S2 大小固定为 128（= s2BaseSize），无尾块问题

### 4.1.2 Tiling实现设计

#### 4.1.2.1 TilingData设计

**不修改 TilingData**。Sink 的开启/关闭通过 TilingKey 中的 `HAS_SINK` 位在编译期决定，sink_token_num 固定为 128（= s2BaseSize），无需运行时传递。

#### 4.1.2.2 核间tiling策略
无变更。Sink 数据每个核都需要计算（每个 Q 行都与 Sink Token 做 attention），分核策略不变。

#### 4.1.2.3 核内tiling策略

**启用判定逻辑（Tiling 阶段）**：
```
在 QSFAPInfoParser::Parse() 中：
  sfaaInfo_->hasSink = (opParamInfo_.keySink.tensor != nullptr && kvLayout == PA_BSND)

在 QSFAPMlaTiling::GenTilingKey() 中：
  uint32_t hasSink = sfaaInfo_->hasSink ? 1U : 0U;
  tilingKey_ = GET_TPL_TILING_KEY(hasSink, layoutQuery, layoutKV, ...);
```

**参数校验新增**：
- key_sink 存在时：
  - 校验 kvLayout == PA_BSND
  - 校验 key_sink dtype == BF16（与 Q 一致）
  - 校验 key_sink shape == [128, N1, 576]（无 B 维度，sink_token_num=128, N1=KV head 数量, D=576）
  - 校验 value_sink 也存在，shape == [128, N1, 512]（D_nope=512）
  - MLA 下 N1=1，value_sink nope 部分复用 key_sink 数据

### 4.1.3 Buffer设计

#### 4.1.3.1 UB空间的分配
无新增。Sink 迭代复用现有 bmm1Buffers 和 bmm2Buffers。

Sink 搬运由 AIC 完成（GM→L1 NZ），不经过 UB，因此 AIV 侧无额外 UB 开销。AIC 侧复用 `CopyToL1Nd2Nz<Q_T>` 直接将 BF16 数据从 GM 搬到 L1 NZ 格式。

#### 4.1.3.2 workspace的分配
无新增。

#### 4.1.3.3 L1的分配
无新增 buffer。Sink KV 数据写入现有的 `l1RightBuffers`（3-buf CrossCore），格式与正常反量化后的 NZ 数据一致。

**Sink KV L1 写入格式**：

| Buffer | 块大小 | BUF_NUM | 数据类型 | 大小 | 备注 |
| ------ | ------ | ------- | -------- | ---- | ---- |
| l1RightBuffers | s2Base(128)×N1×576×sizeof(BF16)（MLA 下 N1=1） | 3 | BF16 | 已有 | Sink 复用此 buffer |

Sink KV 写入 L1 后的数据排布（NZ 格式）：
- nope 部分（512 维）：与反量化 KV 一致
- rope 部分（64 维）：紧跟 nope 之后
- BMM1 使用 nope+rope = 576 维做 Q×K^T
- BMM2 使用 nope = 512 维做 P×V
- **MLA 复用**：BMM2 的 V 数据直接从同一份 L1 NZ 数据中取 nope 部分，与正常 KV 的处理方式一致

### 4.1.4 Kernel设计

#### 4.1.4.1 模板参数与入口函数变更

**kv_quant_sparse_flash_attention_pioneer_common.h 变更**：

```cpp
// 变更前
#define CUBE_BLOCK_TRAITS_CONST_FIELDS(X) \
    X(isFd, bool, false) \
    X(isPa, bool, true) \
    ...

// 变更后
#define CUBE_BLOCK_TRAITS_CONST_FIELDS(X) \
    X(hasSink, bool, false) \
    X(isPa, bool, true) \
    ...
```

对应 `TEMPLATE_INTF` 宏中的模板参数也需同步修改：
```cpp
// 变更前
template <typename Q_T, typename KV_T, typename T, typename OUTPUT_T, bool isFd, bool isPa, ...>

// 变更后
template <typename Q_T, typename KV_T, typename T, typename OUTPUT_T, bool hasSink, bool isPa, ...>
```

**kv_quant_sparse_flash_attention_pioneer.cpp 变更**：

1. 模板参数重命名：
```cpp
// 变更前
template<int FLASH_DECODE, int LAYOUT_T, int KV_LAYOUT_T, int TEMPLATE_MODE>

// 变更后
template<int HAS_SINK, int LAYOUT_T, int KV_LAYOUT_T, int TEMPLATE_MODE>
```

2. QSFA_OP_IMPL 宏中传入 key_sink 并替换 nullptr：
```cpp
// 变更前
op.Init(query, key, value, sparseIndices, keyScale, valueScale, blocktable,
    actualSeqLengthsQuery, actualSeqLengthsKV, nullptr, nullptr,
    attentionOut, user, tilingData, &tPipe);

// 变更后
op.Init(query, key, value, sparseIndices, keyScale, valueScale, blocktable,
    actualSeqLengthsQuery, actualSeqLengthsKV, key_sink, value_sink,
    attentionOut, user, tilingData, &tPipe);
```

3. 模板实例化中 FLASH_DECODE → HAS_SINK：
```cpp
// 变更前
QSFA_OP_IMPL(..., bfloat16_t, fp8_e4m3fn_t, float, bfloat16_t, FLASH_DECODE, true, ...);

// 变更后
QSFA_OP_IMPL(..., bfloat16_t, fp8_e4m3fn_t, float, bfloat16_t, HAS_SINK, true, ...);
```

#### 4.1.4.2 主 Kernel 类变更（kernel_mla.h）

**新增成员变量**：
```cpp
// Sink KV 的 GM 指针（指向 key_sink，value 数据取 nope 部分）
GlobalTensor<Q_T> sinkKvGm;
```

**Init 变更**：
在 `InitGlobalBuffer` 中初始化 sink GM 指针，并传递给 Cube 服务：
```cpp
if constexpr (hasSink) {
    sinkKvGm.SetGlobalBuffer((__gm__ Q_T *)key_sink);  // 只需 key_sink，value 从 nope 部分取
    // 传递给 Cube 服务，AIC 在 BMM1 阶段搬运 sink 数据
    this->cubeBlock.InitSinkGm(key_sink);
}
```

注意：使用 `if constexpr (hasSink)` 而非运行时判断，编译器在 `hasSink=false` 时完全消除此分支。

**InitGlobalBuffer 签名变更**：
新增 key_sink 参数传递（将 Init 中收到的 key_sink 传到 InitGlobalBuffer）。

**ProcessMainLoop 变更**——在正常 S2 循环前插入 Sink 迭代：

```cpp
// === 伪代码 ===
for bnIdx in [bN2Start, bN2End):
  ComputeParamBatch()
  ComputeS1LoopInfo()
  for gS1Index in [gs1Start, gs1End+PRELOAD]:
    if (notLastTwoLoop):
      ComputeAxisIdxByBnAndGs1()
      ComputeParamS1()
      ComputeS2LoopInfo()

    // ====== Sink 迭代（新增，编译期控制）======
    if constexpr (hasSink):
      if (notLastTwoLoop):
        RunInfo &sinkRunInfo = runInfo[taskId % 3]
        SetRunInfoForSink(sinkRunInfo, runParam, taskId, multiCoreInnerIdx)
        // s2RealSize=128, s2LoopCount=-1(特殊标记), s2LoopLimit 加 1

        if ASCEND_IS_AIC:
          // AIC: 搬运 sink BF16 到 L1 + BMM1（在 IterateBmm1 中根据 isSinkIter 分支）
          cubeBlock.IterateBmm1(bmm1Buffers.Get(), l1RightBuffers.Get(),
              sinkRunInfo, constInfo)
        else:
          // AIV: Vec0 pass-through（WaitCrossCore → SetCrossCore，不搬运数据）
          vecBlock.ProcessVec0(l1RightBuffers.Get(), sinkRunInfo, constInfo)

      if (taskId > 0 && notLast):
        // Stage1: Vec1 + BMM2 for previous task
        ...
      if (taskId > 1):
        // Stage2: Vec2 for prev-prev task
        ...
      ++taskId

    // ====== 正常 S2 循环 ======
    for s2LoopCount in [0, s2LoopLimit]:
      // 原有三级流水逻辑
      // 变更：s2LoopCount==0 时 ProcessVec1 需要走 update 路径（因为 sink 已经初始化了 max/sum）
```

**SetRunInfoForSink 方法**：

```cpp
__aicore__ inline void SetRunInfoForSink(RunInfo &runInfo, RunParamStr &runParam,
    int64_t taskId, int64_t multiCoreInnerIdx)
{
    // 复用 SetRunInfo 的大部分逻辑，但 s2 相关参数设为 sink 特定值
    runInfo.s2RealSize = 128;   // sink_token_num，固定 128
    runInfo.s2AlignedSize = 128;
    runInfo.s2LoopCount = -1;   // 特殊标记：sink 迭代
    runInfo.s2LoopLimit = /* 原 s2LoopLimit + 1，使正常循环的最后一次仍为 s2LoopLimit */
    runInfo.isSinkIter = true;  // 标记为 sink 迭代

    // 其余字段与正常 SetRunInfo 一致
    runInfo.s1RealSize = runParam.s1RealSize;
    runInfo.mRealSize = runParam.mRealSize;
    runInfo.halfMRealSize = runParam.halfMRealSize;
    // ... 复用 ComputeBmm1Tail 逻辑
}
```

**关键变更点汇总**：
- 使用 `if constexpr (hasSink)` 控制 sink 分支，编译期消除
- Sink 迭代通过 `runInfo.isSinkIter = true` 标记
- Sink 迭代的 s2RealSize 固定为 128
- 正常 S2 循环的 s2LoopCount==0 时，ProcessVec1 需要判断 `hasSink` 决定走 update 还是 no_update 路径

#### 4.1.4.3 Cube 服务变更（service_cube_mla.h）

**IterateBmm1 变更——Sink 迭代时 AIC 负责 GM→L1 搬运**：

正常流程中，AIC IterateBmm1 在 `inputRightBuf.WaitCrossCore()` 后直接使用 L1 中已由 AIV Vec0 搬好的 KV 数据。Sink 迭代时，AIV Vec0 仅做 pass-through（不搬运数据），因此 AIC 需要在 WaitCrossCore 之后、BMM1 之前，自行将 sink BF16 数据从 GM 搬到 L1。

```cpp
TEMPLATES_DEF_NO_DEFAULT __aicore__ inline void QSFAMatmulService<TEMPLATE_ARGS>::IterateBmm1(
    Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf,
    Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &inputRightBuf, RunInfo &runInfo,
    ConstInfo &constInfo)
{
    CalcS1Coord(runInfo, constInfo);

    if constexpr (hasSink) {
        if (runInfo.isSinkIter) {
            // ========= Sink 迭代：AIC 搬运 sink BF16 到 L1 =========
            IterateBmm1Sink(outputBuf, inputRightBuf, runInfo, constInfo);
            return;
        }
    }

    // ========= 正常迭代（不变）=========
    IterateBmm1QSFA(outputBuf, inputRightBuf, runInfo, constInfo);
}
```

**新增 IterateBmm1Sink 方法**：

```cpp
__aicore__ inline void IterateBmm1Sink(
    Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf,
    Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &inputRightBuf,
    RunInfo &runInfo, ConstInfo &constInfo)
{
    // 1. 加载 Q 到 L1（与正常 BMM1 一致，s2LoopCount==0 时搬运 Q）
    Buffer<BufferType::L1> inputLeftBuf = l1QBuffers.Get();
    inputLeftBuf.Wait<HardEvent::MTE1_MTE2>();
    LocalTensor<Q_T> inputLeftTensor = inputLeftBuf.GetTensor<Q_T>();
    CopyToL1Nd2Nz<Q_T>(inputLeftTensor, this->queryGm.gmTensor[runInfo.queryOffset],
        runInfo.mRealSize, constInfo.dSize, constInfo.mm1Ka);
    inputLeftBuf.Set<HardEvent::MTE2_MTE1>();

    // 2. 等待 AIV Vec0 pass-through 释放 L1 buffer
    inputRightBuf.WaitCrossCore();

    // 3. AIC 搬运 sink BF16 数据 GM → L1 NZ
    //    key_sink shape: [128, N1, 576]，MLA 下 N1=1
    //    复用 CopyToL1Nd2Nz<Q_T>，与 Q 搬运方式一致（GM ND → L1 NZ）
    //    所有 batch 共享，GM 偏移与 bIdx 无关
    LocalTensor<Q_T> inputRightTensor = inputRightBuf.GetTensor<Q_T>();
    constexpr int64_t sinkD = 576;  // nope(512) + rope(64)
    CopyToL1Nd2Nz<Q_T>(inputRightTensor, sinkKvGm[0],
        runInfo.s2RealSize, sinkD, sinkD);  // s2RealSize=128, K=576

    // 4. 执行 BMM1: Q × K_sink^T
    inputLeftBuf.Wait<HardEvent::MTE2_MTE1>();
    Buffer<BufferType::L0C> mm1ResL0C = mmL0CBuffers.Get();
    mm1ResL0C.Wait<HardEvent::FIX_M>();
    MMParam param = {static_cast<uint32_t>(runInfo.mRealSize),     // singleM
                     static_cast<uint32_t>(runInfo.s2RealSize),    // singleN = 128
                     static_cast<uint32_t>(constInfo.dSize),       // singleK = 576
                     0,    // isLeftTranspose
                     1     // isRightTranspose
                    };
    MatmulK<Q_T, Q_T, T, s1BaseSize, s2BaseSize, dBaseMatmulSize, ABLayout::MK, ABLayout::KN>(
        inputLeftBuf.GetTensor<Q_T>(), inputRightBuf.GetTensor<Q_T>(),
        mmL0ABuffers, mmL0BBuffers,
        mm1ResL0C.GetTensor<T>(),
        param);

    // 5. Fixpipe L0C → UB（与正常 BMM1 一致）
    mm1ResL0C.Set<HardEvent::M_FIX>();
    mm1ResL0C.Wait<HardEvent::M_FIX>();

    outputBuf.WaitCrossCore();
    FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;
    fixpipeParams.nSize = Align8Func(runInfo.s2RealSize);
    fixpipeParams.mSize = Align2Func(runInfo.mRealSize);
    fixpipeParams.srcStride = Align16Func(fixpipeParams.mSize);
    fixpipeParams.dstStride = s2BaseSize;
    fixpipeParams.dualDstCtl = 1;
    fixpipeParams.params.ndNum = 1;
    fixpipeParams.params.srcNdStride = 0;
    fixpipeParams.params.dstNdStride = 0;

    Fixpipe<T, T, PFA_CFG_ROW_MAJOR_UB>(outputBuf.template GetTensor<T>(),
        mm1ResL0C.GetTensor<T>(), fixpipeParams);
    mm1ResL0C.Set<HardEvent::FIX_M>();
    outputBuf.SetCrossCore();
}
```

**新增成员变量**（Cube 服务类中）：
```cpp
// Sink KV 的 GM 指针，由 Kernel 主类传入
GlobalTensor<Q_T> sinkKvGm;
```

**新增 InitSinkGm 方法**：
```cpp
__aicore__ inline void InitSinkGm(__gm__ uint8_t *key_sink)
{
    if ASCEND_IS_AIC {
        if constexpr (hasSink) {
            sinkKvGm.SetGlobalBuffer((__gm__ Q_T *)key_sink);
        }
    }
}
```

**BMM2 无变更**。Sink 数据在 L1 中的 NZ 排布与正常反量化数据完全一致。BMM2 使用 `inputRightBuf.GetTensor<Q_T>()` 取 nope 部分（前 512 维），`inputRightBuf.GetTensor<Q_T>(s2BaseSize * constInfo.dSizeNope)` 取 P 部分（rope 位置），无需修改。

#### 4.1.4.4 Vector 服务变更（service_vector_mla.h）

**ProcessVec0 变更——Sink 迭代 pass-through**：

Sink 迭代时，AIV Vec0 不搬运数据，仅执行 WaitCrossCore → SetCrossCore 释放 L1 buffer 给 AIC 使用：

```cpp
TEMPLATES_DEF_NO_DEFAULT __aicore__ inline void QSFAVectorService<TEMPLATE_ARGS>::ProcessVec0(
    Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputL1,
    const RunInfo &runInfo, ConstInfo &constInfo)
{
    outputL1.WaitCrossCore();

    if constexpr (hasSink) {
        if (runInfo.isSinkIter) {
            // ========= Sink pass-through =========
            // AIC 负责搬运 sink BF16 到 L1，AIV 仅释放 L1 buffer
            outputL1.SetCrossCore();
            return;
        }
    }

    // ========= 正常 Sparse KV 搬运路径（不变）=========
    blockSize = constInfo.oriBlockSize;
    maxBlockNumPerBatch = constInfo.oriMaxBlockNumPerBatch;
    ProcessSparseKv(outputL1, runInfo, constInfo);
    outputL1.SetCrossCore();
}
```

**不新增 ProcessSinkKv / CopyInSinkBf16ToNz 方法**（搬运逻辑已移至 AIC Cube 服务）。

**ProcessVec1 变更**：

正常 S2 循环的首次迭代（s2LoopCount==0），如果之前有 sink 迭代，需要走 update 路径（而非 no_update），因为 sink 迭代已经初始化了 max/sum 状态：

```cpp
// 变更前
if (runInfo.s2LoopCount == 0) {
    ProcessVec1Vf<..., false, ...>(...);  // false = no update（首次初始化 max/sum）
} else {
    ProcessVec1Vf<..., true, ...>(...);   // true = update
}

// 变更后
// isFirstEver: 整个 S2 维度上的首次（包括 sink），仅 sink 迭代本身为 true
bool isFirstEver;
if constexpr (hasSink) {
    isFirstEver = runInfo.isSinkIter;  // 仅 sink 迭代是真正的首次
} else {
    isFirstEver = (runInfo.s2LoopCount == 0);  // 无 sink 时，s2Loop=0 即首次
}

if (isFirstEver) {
    ProcessVec1Vf<..., false, ...>(...);  // 首次：初始化 max/sum
} else {
    ProcessVec1Vf<..., true, ...>(...);   // 非首次：update max/sum
}
```

**ProcessVec2 无变更**：FlashUpdate 逻辑天然支持 sink 迭代，只要 s2LoopCount 语义正确（sink 迭代的 s2LoopCount 为 0 或 -1，后续迭代正常递增）。

**InitGlobalBuffer 变更**：
新增 sinkKvGm 的 GM 指针设置，并传递给 Cube 服务：
```cpp
if constexpr (hasSink) {
    sinkKvGm.SetGlobalBuffer((__gm__ Q_T *)key_sink);
    this->cubeBlock.InitSinkGm(key_sink);
}
```

#### 4.1.4.5 运行时参数变更

**ConstInfo / CVSharedParams / RunInfo 均不新增字段**。

hasSink 通过编译期模板参数判定，无需运行时传递。

唯一运行时需要的信息：
- `runInfo.isSinkIter`：在 `RunInfo` 中新增此标志位，区分 sink 迭代与正常迭代。或者也可以用 `runInfo.s2LoopCount == -1` 作为特殊标记，不新增字段。

**推荐方案**：在 `RunInfo` 中新增 `bool isSinkIter = false`，语义更清晰。

```cpp
// util_regbase.h RunInfo 中新增
struct RunInfo {
    COMMON_RUN_INFO;
    ...
    bool isSinkIter = false;  // 当前是否为 Sink 迭代
};
```

这是唯一的运行时参数变更，不涉及 CVSharedParams 和 ConstInfo。

#### 4.1.4.6 核间同步协议

**Sink 块的核间同步协议与正常块略有不同（Vec0 阶段）**：

```
正常块：
  AIV Vec0: WaitCrossCore(L1) → 搬运反量化 KV → SetCrossCore(L1)
  AIC BMM1: WaitCrossCore(L1) → Q×K → SetCrossCore(UB)
  AIV Vec1: WaitCrossCore(UB) → Softmax → SetCrossCore(L1)
  AIC BMM2: WaitCrossCore(L1) → P×V → SetCrossCore(UB)
  AIV Vec2: WaitCrossCore(UB) → FlashUpdate → SetCrossCore(UB)

Sink 块（Vec0 阶段变更，其余不变）：
  AIV Vec0: WaitCrossCore(L1) → pass-through → SetCrossCore(L1)   ← 不搬运数据，仅释放 L1
  AIC BMM1: WaitCrossCore(L1) → GM→L1(sink BF16) → Q×K → SetCrossCore(UB)  ← AIC 负责搬运 + BMM1
  AIV Vec1: WaitCrossCore(UB) → Softmax → SetCrossCore(L1)        ← 无变更
  AIC BMM2: WaitCrossCore(L1) → P×V → SetCrossCore(UB)            ← 无变更
  AIV Vec2: WaitCrossCore(UB) → FlashUpdate → SetCrossCore(UB)    ← 无变更
```

**关键设计**：
- AIV Vec0 的 WaitCrossCore → SetCrossCore pass-through 保证了核间同步时序不变（AIC 仍然在 WaitCrossCore 后才访问 L1）
- AIC 在 WaitCrossCore(L1) 之后、BMM1 之前插入 `CopyToL1Nd2Nz` 搬运 sink 数据，不影响后续 Vec1/BMM2/Vec2 的同步协议
- 这种方式更合理：BF16 数据无需反量化，AIC 直接搬运到 L1 NZ 格式后立即执行 BMM1，减少核间通信开销

### 4.1.5 异常场景设计
| 异常场景 | 处理方式 | 处理层级 |
| -------- | -------- | -------- |
| key_sink 为 nullptr | HAS_SINK=0，TilingKey 走无 sink 路径 | Host Tiling + 编译期 |
| key_sink 存在但 kvLayout 非 PA_BSND | Tiling 阶段报错 | Host Tiling Check |
| key_sink 存在但 value_sink 为 nullptr | Tiling 阶段报错（MLA 场景理论上不会发生） | Host Tiling Check |
| key_sink shape != [128, N1, 576] | Tiling 阶段报错 | Host Tiling Check |
| value_sink shape != [128, N1, 512] | Tiling 阶段报错 | Host Tiling Check |
| key_sink dtype 非 BF16 | Tiling 阶段报错 | Host Tiling Check |

### 4.1.6 支持确定性计算设计
不涉及。Sink 迭代的计算顺序确定（始终在 S2 循环之前），不引入非确定性。

### 4.1.7 精度分析及设计

**精度影响分析**：
| 环节 | 正常块 | Sink 块 | 差异 |
|------|--------|---------|------|
| KV 数据精度 | FP8 → BF16（反量化，有精度损失） | BF16（原始精度） | Sink 精度更高 |
| BMM1 计算 | BF16 × BF16 → FP32 | 相同 | 无差异 |
| Softmax | FP32 | 相同 | 无差异 |
| BMM2 计算 | BF16 × BF16 → FP32 | 相同 | 无差异 |
| FlashUpdate | FP32 累加 | 相同 | 无差异 |

**结论**：Sink 块使用 BF16 原始数据，精度不低于正常量化块。Online Softmax 的数值稳定性不受影响（Sink 作为首个块参与 max/sum 初始化）。

### 4.1.8 性能分析及设计

| 指标 | 影响 | 说明 |
|------|------|------|
| 额外计算量 | +128 tokens 的 BMM1/BMM2 | 固定开销，每个 GS1 迭代 +1 次 |
| 额外搬运量 | 128×N1×576×2B（MLA 下 N1=1，= 144KB/每次 GS1 迭代） | AIC MTE2 搬运（CopyToL1Nd2Nz），无反量化开销 |
| 流水线影响 | 在 S2 循环前增加 1 次迭代 | 流水线启动延迟 +1 个 stage |
| 二进制膨胀 | HAS_SINK=1 新增组合 | 增加 2 组 TilingKey（BSND+PA, TND+PA） |

**AIC 搬运方案的优势**：
- BF16 数据无需反量化，AIC 直接 GM→L1 NZ 后立即执行 BMM1，减少核间等待
- 复用 AIC 已有的 `CopyToL1Nd2Nz<Q_T>` 搬运逻辑（与 Q 搬运方式一致），实现简洁
- AIV Vec0 仅做 pass-through，释放 AIV 计算资源
- BMM2 无需修改，L1 NZ 格式与正常反量化数据完全一致

**后续优化方向**：
- Sink KV L1 驻留优化（S1 循环间不重复搬运，仅首次搬运）
- 将 sink BMM1/BMM2 与正常块首次迭代合并（减少流水线启动延迟）

### 4.1.9 训推一致性设计
不涉及。仅支持推理场景。

# 5 维测设计

## 5.1 可测试性

### 5.1.1 功能测试
| 测试场景 | 验证点 | 优先级 |
| -------- | ------ | ------ |
| PA_BSND + key_sink 有效 | Sink token 参与 attention 计算，输出正确 | P0 |
| PA_BSND + key_sink 为 None | 行为与修改前完全等价（回归），HAS_SINK=0 路径 | P0 |
| BSND 布局 + key_sink 为 None | 不影响原有功能（HAS_SINK=0） | P1 |
| TND 布局 + key_sink 为 None | 不影响原有功能（HAS_SINK=0） | P1 |
| 多 batch（B>1）+ sink | 所有 batch 共用同一份 sink KV，各自输出正确 | P0 |
| 不同 sparse_block_count + sink | Sink 与稀疏块正确组合 | P1 |
| sparse_block_count=0 + sink | 仅计算 128 个 sink tokens 的 attention | P1 |

### 5.1.2 精度测试
**CPU 参考实现构建方法**：
1. 将 key_sink 视为 BF16 张量 [128, N1, 576]（MLA 下 N1=1），value_sink 为 [128, N1, 512]（nope 部分复用 key_sink）
2. Sink Key 取 key_sink 全部 576 维（nope+rope），Sink Value 取 key_sink 前 512 维（nope）
3. 在 attention 计算中，将 sink KV 拼接到正常 KV 序列开头
4. 等效于 `K_full = concat(K_sink, K_sparse)`, `V_full = concat(V_sink, V_sparse)`（每个 batch 使用相同的 sink）
5. 计算 `Attention = softmax(Q × K_full^T / √d) × V_full`
6. 对比 NPU 输出与 CPU 输出：atol=1e-3, rtol=1e-2（考虑 FP8 反量化精度损失）

### 5.1.3 边界条件测试
| 边界场景 | 预期行为 |
| -------- | -------- |
| s1=1（单 query token + sink） | 正常计算，128 sink + sparse tokens |
| sparse_block_count=0（仅 sink） | 仅计算 sink tokens 的 attention |
| actualS2=0（无 KV + 有 sink） | 仅计算 sink tokens |
| B=1, s1=128, 大量 sparse blocks + sink | 大规模正确性验证 |
| B=4, 验证所有 batch 共享同一份 sink | 各 batch 输出中 sink 贡献一致 |

## 5.2 可观察性
Cannsim 仿真验证：对比有/无 sink 的 attention 输出差异，确认 sink tokens 的贡献。

## 5.3 可维护性

### 5.3.1 关键风险与应对
| 风险 | 描述 | 应对措施 |
| ---- | ---- | -------- |
| L1 空间不足 | Sink KV 复用 l1RightBuffers，无额外空间需求 | 无风险 |
| AIC sink 搬运正确性 | AIC CopyToL1Nd2Nz 搬运 BF16 数据需与 AIV 反量化路径输出的 NZ 格式一致 | 复用已有 CopyToL1Nd2Nz（与 Q 搬运方式一致），确保 NZ 格式一致 |
| 二进制膨胀 | HAS_SINK=1 新增 TilingKey 组合 | 仅增加 2 组（PA_BSND 场景），影响可控 |
| 向后兼容 | key_sink 为 None 时必须等价 | HAS_SINK=0 时编译器消除所有 sink 代码 |
| FLASH_DECODE 语义丢失 | 原 FLASH_DECODE 功能已不存在 | 确认该字段从未使用，可安全复用 |

### 5.3.2 后续演进
- 支持可变 sink_token_num（非固定 128）
- Sink KV L1 驻留优化（S1 循环间不重复搬运）
- 支持量化 Sink KV（FP8 格式）
- 扩展到非 PA_BSND 布局

# 6 资料设计
| 资料类型 | 是否涉及变更 | 变更内容 |
| -------- | ------------ | -------- |
| README.md | 是 | 新增 key_sink/value_sink 参数说明，标注 nope 复用关系和 batch 共享 |
| API 文档 | 是 | 更新参数表，说明 Param Sink 功能 |
