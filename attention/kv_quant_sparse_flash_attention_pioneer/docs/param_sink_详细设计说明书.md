# KvQuantSparseFlashAttentionPioneer 算子 Param Sink 特性详细设计说明书

<center>**修订记录**</center>

|    日期    | 修订版本 |          修改描述          |   作者    |
| :--------: | :------: | :------------------------: | :-------: |
| 2026-03-18 |   1.0    | Param Sink 特性详细设计初稿 | AI Assistant |
| 2026-03-18 |   1.1    | TilingKey 控制开关；Vec0 搬运；MLA 复用+batch 共享 | AI Assistant |
| 2026-03-18 |   1.2    | 修正 key_sink/value_sink shape：(128, N1, 576)/(128, N1, 512)，value_sink nope 复用 key_sink | AI Assistant |
| 2026-03-18 |   1.3    | Sink 搬运改为 AIC BMM1 阶段（复用 CopyToL1Nd2Nz），AIV Vec0 仅 pass-through | AI Assistant |
| 2026-03-19 |   1.4    | 补全 Tiling 侧校验：反向不对称校验、value_sink desc/dtype 校验；GenTilingKey 防御性双重条件 | AI Assistant |
| 2026-03-19 |   1.5    | Kernel 侧详设重构：Sink 融入 S2 循环（s2LoopLimit+1, effectiveS2-1）；移除独立 Sink 迭代块和 IterateBmm1Sink；sink 搬运在 IterateBmm1QSFA 内实现 | AI Assistant |

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
- **最小 Kernel 侵入**：复用现有三级流水架构，Sink 迭代融入 S2 循环（占用 s2LoopCount=0），不新增独立迭代块或搬运函数；Sink KV 搬运由 AIC 在 IterateBmm1QSFA 内完成（复用 CopyToL1Nd2Nz），AIV Vec0 仅做 CrossCore pass-through
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
| 10 | 算子kernel实现 | 是 | 是 | S2 循环 s2LoopLimit+1、AIC IterateBmm1QSFA 内 sink 搬运、Vec0 pass-through |
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

// 变更后（防御性双重条件：tensor 存在 + kvLayout 为 PA_BSND）
uint32_t hasSink = (sfaaInfo_->opParamInfo.keySink.tensor != nullptr &&
                    sfaaInfo_->kvLayout == QSFALayout::PA_BSND) ? 1U : 0U;
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
| op_kernel/*_kernel_mla.h | Kernel 主类 | S2 循环 s2LoopLimit+1、SetRunInfo 增加 effectiveS2 偏移逻辑、传递 sinkKvGm 给 Cube 服务 | - | Param Sink |
| op_kernel/*_service_cube_mla.h | Cube 服务 | IterateBmm1QSFA 内新增 sink BF16 GM→L1 搬运分支（CopyToL1Nd2Nz）；新增 sinkKvGm 成员和 InitSinkGm | - | Param Sink |
| op_kernel/*_service_vector_mla.h | Vector 服务 | ProcessVec0 新增 sink pass-through 分支（WaitCrossCore + SetCrossCore） | - | Param Sink |

**注意**：**不修改 TilingData（tiling.h）**，**不修改 ConstInfo/CVSharedParams**。`util_regbase.h` 中 `RunInfo` 新增 `bool isSinkIter = false` 字段，hasSink 通过编译期模板参数判定。

# 4 模板设计

## 4.1 Param Sink 计算模板

### 4.1.1 计算流程图

**核心设计思路**：Sink 迭代不作为独立阶段，而是**融入现有 S2 循环**，占用 `s2LoopCount=0` 位置。通过 `s2LoopLimit += 1` 扩展循环次数，正常 KV 索引在使用时 `-1` 偏移。

**正常块 vs Sink 块数据通路对比**：

```
正常 Sparse KV 块（s2LoopCount >= 1，hasSink 时）：
  AIV Vec0: GM(KV_FP8) → UB → 反量化(VF) → NZ → L1
  AIC BMM1: L1(Q) × L1(K_NZ)^T → UB(scores)
  AIV Vec1: Softmax(scores, update=true) → P_NZ → L1
  AIC BMM2: L1(P) × L1(V_NZ) → UB(partial_out)
  AIV Vec2: FlashUpdate → attentionOut

Sink KV 块（s2LoopCount=0，hasSink 时）：
  AIV Vec0: L1 pass-through (WaitCrossCore → SetCrossCore)   ← 不搬运数据，仅释放 L1 buffer
  AIC BMM1: GM(sink_BF16) → L1(NZ) → Q×K^T → UB(scores)    ← AIC 在 IterateBmm1QSFA 内完成 GM→L1 搬运 + BMM1
  AIV Vec1: Softmax(scores, update=false) → P_NZ → L1       ← 首次初始化 max/sum
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

**三级流水时序图（Sink 融入 S2 循环）**：

```
s2LoopCount:  0(sink)    1          2          3         ...
时间 →        T0         T1         T2         T3        ...
AIC:        BMM1_sink  BMM1[0]    BMM1[1]    BMM1[2]   ...
                       BMM2_sink  BMM2[0]    BMM2[1]   ...
AIV:        Vec0_sink  Vec0[0]    Vec0[1]    Vec0[2]   ...
                       Vec1_sink  Vec1[0]    Vec1[1]   ...
                                  Vec2_sink  Vec2[0]   ...
```

**说明**：
- Sink 迭代占用 `s2LoopCount=0`，与正常 S2 迭代共用同一个 for 循环，无独立的 sink 循环块
- `s2LoopLimit += 1`：原始 `s2LoopLimit = s2LoopEndIdx - 1`，hasSink 时加 1，使正常 KV 迭代次数不变
- 正常 KV 索引偏移：`effectiveS2 = s2LoopCount - 1`（hasSink 时），确保 s2LoopCount=1 对应原来的 s2LoopCount=0
- Vec0_sink：AIV 执行 pass-through（WaitCrossCore → SetCrossCore，不搬运数据），AIC 在 IterateBmm1QSFA 内完成 GM→L1 搬运 + 矩阵乘
- Sink 的 S2 大小固定为 128（= s2BaseSize），无尾块问题

### 4.1.2 Tiling实现设计

#### 4.1.2.1 TilingData设计

**不修改 TilingData**。Sink 的开启/关闭通过 TilingKey 中的 `HAS_SINK` 位在编译期决定，sink_token_num 固定为 128（= s2BaseSize），无需运行时传递。

#### 4.1.2.2 核间tiling策略
无变更。Sink 数据每个核都需要计算（每个 Q 行都与 Sink Token 做 attention），分核策略不变。

#### 4.1.2.3 核内tiling策略

**启用判定逻辑（Tiling 阶段）**：
```
在 QSFAPMlaTiling::GenTilingKey() 中：
  // 防御性双重条件：即使 Check 已校验，GenTilingKey 也不依赖 Check 的执行顺序
  uint32_t hasSink = (sfaaInfo_->opParamInfo.keySink.tensor != nullptr &&
                      sfaaInfo_->kvLayout == QSFALayout::PA_BSND) ? 1U : 0U;
  tilingKey_ = GET_TPL_TILING_KEY(hasSink, layoutQuery, layoutKV, ...);
```

**参数校验新增**（`CheckFeatureSinkParams`）：

- key_sink 为 nullptr 时：
  - 校验 value_sink 也为 nullptr（反向不对称校验：key_sink 为空但 value_sink 不为空时报错）
  - 两者均为 nullptr 则直接返回成功（功能关闭）

- key_sink 存在时：
  - 校验 kvLayout == PA_BSND
  - 校验 key_sink desc != nullptr
  - 校验 key_sink dtype == BF16（与 Q 一致）
  - 校验 key_sink shape == [128, N1, 576]（无 B 维度，sink_token_num=128, N1=KV head 数量, D=576）
  - 校验 value_sink tensor != nullptr（key_sink 存在时 value_sink 必须存在）
  - 校验 value_sink desc != nullptr
  - 校验 value_sink dtype == BF16
  - 校验 value_sink shape == [128, N1, 512]（D_nope = qHeadDim - ropeHeadDim = 576 - 64 = 512）
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

**设计原则**：不新增独立的 Sink 迭代块和 `SetRunInfoForSink` 方法。Sink 迭代融入现有 S2 循环，通过 `s2LoopLimit += 1` 扩展循环次数，`SetRunInfo` 内部根据 `s2LoopCount == 0` 设置 `isSinkIter` 标志，正常 KV 索引通过 `effectiveS2 = s2LoopCount - 1` 偏移。

**Init 变更**：
在 `InitGlobalBuffer` 中将 key_sink 传递给 Cube 服务：
```cpp
if constexpr (hasSink) {
    this->cubeBlock.InitSinkGm(key_sink);
}
```

注意：sinkKvGm 成员变量仅在 Cube 服务类中定义，Kernel 主类不持有。使用 `if constexpr (hasSink)` 编译期控制。

**ProcessMainLoop 变更**——Sink 融入 S2 循环：

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
      s2LoopLimit = runParam.s2LoopEndIdx - 1
      if constexpr (hasSink):
        s2LoopLimit += 1   // sink 占用 s2LoopCount=0，总循环次数 +1

    // ====== S2 循环（hasSink 时 s2LoopCount=0 为 sink 迭代）======
    for s2LoopCount in [0, s2LoopLimit]:
      if (notLastTwoLoop):
        RunInfo &runInfo1 = runInfo[taskId % 3]
        SetRunInfo(runInfo1, runParam, taskId, s2LoopCount, s2LoopLimit, multiCoreInnerIdx)
        // SetRunInfo 内部：s2LoopCount==0 时设 isSinkIter=true，否则 effectiveS2=s2LoopCount-1

        if ASCEND_IS_AIC:
          cubeBlock.IterateBmm1(bmm1Buffers.Get(), l1RightBuffers.Get(), runInfo1, constInfo)
        else:
          vecBlock.ProcessVec0(l1RightBuffers.Get(), runInfo1, constInfo)

      if (taskId > 0 && notLast):
        // Stage1: Vec1 + BMM2 for previous task（无变更）
        ...
      if (taskId > 1):
        // Stage2: Vec2 for prev-prev task（无变更）
        ...
      ++taskId
```

**SetRunInfo 变更**（在现有方法中增加 sink 处理，不新增方法）：

```cpp
__aicore__ inline void SetRunInfo(RunInfo &runInfo, RunParamStr &runParam,
    int64_t taskId, int64_t s2LoopCount, int64_t s2LoopLimit, int64_t multiCoreInnerIdx)
{
    // === Sink 索引偏移逻辑 ===
    runInfo.isSinkIter = false;
    int64_t effectiveS2 = s2LoopCount;
    if constexpr (hasSink) {
        if (s2LoopCount == 0) {
            runInfo.isSinkIter = true;   // s2LoopCount=0 为 sink 迭代
        } else {
            effectiveS2 = s2LoopCount - 1;  // 正常 KV 索引 -1 偏移
        }
    }

    // === 使用 effectiveS2 计算 s2 方向参数 ===
    if (effectiveS2 < runParam.oriKvLoopEndIdx) {
        runInfo.s2StartIdx = runParam.s2LineStartIdx;
        runInfo.s2EndIdx = runParam.s2LineEndIdx;
    } else {
        runInfo.s2StartIdx = 0;
        runInfo.s2EndIdx = runParam.s2CmpLineEndIdx;
    }
    runInfo.s2LoopCount = s2LoopCount;  // 保留原始 s2LoopCount（Vec1/Vec2 使用）
    runInfo.s2LoopLimit = s2LoopLimit;

    // ... 其余字段赋值与原有逻辑一致 ...
    this->ComputeBmm1Tail(runInfo, runParam);
}
```

**ComputeBmm1Tail 变更**（尾块计算中使用偏移后的索引）：

```cpp
__aicore__ inline void ComputeBmm1Tail(RunInfo &runInfo, RunParamStr &runParam)
{
    // S1 相关（无变更）
    runInfo.s1RealSize = runParam.s1RealSize;
    runInfo.mRealSize = runParam.mRealSize;
    runInfo.halfMRealSize = runParam.halfMRealSize;
    // ...

    // S2 相关
    runInfo.s2RealSize = constInfo.s2BaseSize;  // 默认 128
    runInfo.s2AlignedSize = runInfo.s2RealSize;

    if constexpr (hasSink) {
        if (runInfo.isSinkIter) {
            return;  // sink 固定 128，无需尾块计算
        }
    }

    // 正常 KV 尾块计算：使用偏移后的 effectiveS2Loop
    int64_t effectiveS2Loop = runInfo.s2LoopCount;
    if constexpr (hasSink) {
        effectiveS2Loop -= 1;  // 正常 KV 索引 -1
    }
    int64_t curS2LoopCnt = (effectiveS2Loop >= runParam.oriKvLoopEndIdx) ?
        (effectiveS2Loop - runParam.oriKvLoopEndIdx) : effectiveS2Loop;
    if (runInfo.s2StartIdx + (curS2LoopCnt + 1) * runInfo.s2RealSize > runInfo.s2EndIdx) {
        runInfo.s2RealSize = runInfo.s2EndIdx - curS2LoopCnt * runInfo.s2RealSize - runInfo.s2StartIdx;
        runInfo.s2AlignedSize = Align(runInfo.s2RealSize);
    }
}
```

**关键变更点汇总**：
- **不新增 `SetRunInfoForSink` 方法**，在现有 `SetRunInfo` 中通过 `s2LoopCount == 0` 判断 sink
- **不新增独立 Sink 迭代块**，Sink 融入 S2 for 循环，`s2LoopLimit += 1`
- 正常 KV 索引通过 `effectiveS2 = s2LoopCount - 1` 偏移，确保 s2LoopCount=1 对应原来的 s2LoopCount=0
- `ComputeBmm1Tail` 中 sink 迭代直接返回（固定 128），正常迭代使用 `effectiveS2Loop -= 1`
- Kernel 主类不持有 `sinkKvGm`，仅通过 `InitSinkGm` 传递给 Cube 服务

#### 4.1.4.3 Cube 服务变更（service_cube_mla.h）

**设计原则**：不新增 `IterateBmm1Sink` 方法，Sink 搬运逻辑直接在现有 `IterateBmm1QSFA` 中通过 `if constexpr (hasSink)` 分支实现。

**IterateBmm1 无变更**（仍直接调用 IterateBmm1QSFA）：
```cpp
__aicore__ inline void IterateBmm1(...) {
    CalcS1Coord(runInfo, constInfo);
    IterateBmm1QSFA(outputBuf, inputRightBuf, runInfo, constInfo);
}
```

**IterateBmm1QSFA 变更**——在 `WaitCrossCore` 之后插入 Sink 搬运分支：

```cpp
__aicore__ inline void IterateBmm1QSFA(
    Buffer<UB, CROSS_CORE_SYNC_BOTH> &outputBuf,
    Buffer<L1, CROSS_CORE_SYNC_FORWARD> &inputRightBuf,
    RunInfo &runInfo, ConstInfo &constInfo)
{
    // 1. 加载 Q 到 L1（s2LoopCount==0 时搬运 Q，sink 迭代也是 s2LoopCount==0）
    if (unlikely(runInfo.s2LoopCount == 0)) {
        inputLeftBuf = l1QBuffers.Get();
        inputLeftBuf.Wait<HardEvent::MTE1_MTE2>();
        CopyToL1Nd2Nz<Q_T>(inputLeftBuf.GetTensor<Q_T>(),
            this->queryGm.gmTensor[runInfo.queryOffset],
            runInfo.mRealSize, constInfo.dSize, constInfo.mm1Ka);
        inputLeftBuf.Set<HardEvent::MTE2_MTE1>();
    } else {
        inputLeftBuf = l1QBuffers.GetPre();  // 复用已加载的 Q
        inputLeftBuf.Set<HardEvent::MTE2_MTE1>();
    }

    // 2. 等待 AIV Vec0 释放 L1 buffer
    inputRightBuf.WaitCrossCore();

    // 3. ========= Sink 搬运（新增，在 WaitCrossCore 之后）=========
    if constexpr (hasSink) {
        if (runInfo.isSinkIter) {
            // AIC 搬运 sink BF16 数据 GM → L1 NZ
            // key_sink shape: [128, N1, 576]，MLA 下 N1=1
            // 复用 CopyToL1Nd2Nz<Q_T>，与 Q 搬运方式一致
            // 所有 batch 共享，GM 偏移固定为 0
            LocalTensor<Q_T> inputRightTensor = inputRightBuf.GetTensor<Q_T>();
            CopyToL1Nd2Nz<Q_T>(inputRightTensor, sinkKvGm[0],
                runInfo.s2RealSize, constInfo.dSize, constInfo.dSize);
        }
    }

    // 4. 执行 BMM1: Q × K^T（sink 和正常迭代共用此路径）
    inputLeftBuf.Wait<HardEvent::MTE2_MTE1>();
    // ... MatmulK + Fixpipe（与原有逻辑完全一致）...
}
```

**关键设计点**：
- Sink 搬运代码仅 5 行，插入在 `WaitCrossCore` 和 `MatmulK` 之间，对原有代码侵入极小
- `CopyToL1Nd2Nz<Q_T>(inputRightTensor, sinkKvGm[0], s2RealSize=128, dSize=576, dSize=576)`：将 sink BF16 数据从 GM 搬到 L1 NZ 格式
- 搬运完成后，后续 MatmulK 和 Fixpipe 路径与正常迭代完全一致，无需任何修改
- `sinkKvGm[0]`：所有 batch 共享，偏移固定为 0

**新增成员变量**（Cube 服务类中）：
```cpp
GlobalTensor<Q_T> sinkKvGm;  // Sink KV 的 GM 指针，由 Kernel 主类通过 InitSinkGm 传入
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
__aicore__ inline void ProcessVec0(
    Buffer<L1, CROSS_CORE_SYNC_FORWARD> &outputL1,
    const RunInfo &runInfo, ConstInfo &constInfo)
{
    outputL1.WaitCrossCore();

    if constexpr (hasSink) {
        if (runInfo.isSinkIter) {
            // Sink pass-through: AIC 负责搬运，AIV 仅释放 L1 buffer
            outputL1.SetCrossCore();
            return;
        }
    }

    // 正常 Sparse KV 搬运路径（不变）
    blockSize = constInfo.oriBlockSize;
    maxBlockNumPerBatch = constInfo.oriMaxBlockNumPerBatch;
    ProcessSparseKv(outputL1, runInfo, constInfo);
    outputL1.SetCrossCore();
}
```

**不新增 ProcessSinkKv / CopyInSinkBf16ToNz 方法**（搬运逻辑已在 AIC Cube 服务的 IterateBmm1QSFA 中实现）。

**ProcessVec1 变更**：

Sink 迭代是整个 S2 维度上的首次迭代，需要走 `update=false` 路径初始化 max/sum。正常 S2 循环的 s2LoopCount=1（hasSink 时对应原来的 s2LoopCount=0）需要走 `update=true` 路径：

```cpp
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

**ProcessVec1 中 QSFAUpdateExpSumAndExpMax 调用**：
```cpp
// 原有逻辑使用 s2LoopCount != 0 判断是否需要 update
// hasSink 时 sink 迭代的 s2LoopCount=0，不执行 update（正确）
// hasSink 时正常迭代的 s2LoopCount>=1，执行 update（正确）
// 因此此处无需修改
if (runInfo.s2LoopCount != 0) {
    QSFAUpdateExpSumAndExpMax<T>(sumUb, maxUb, expUb, sumUb, maxUb, apiTmpBuffer, runInfo.halfMRealSize);
}
```

**ProcessVec2 变更**：

ProcessVec2 中使用 `s2LoopCount` 判断首次/末次迭代：

```cpp
// s2LoopCount == 0：首次迭代（hasSink 时为 sink 迭代），DataCopy 初始化
// s2LoopCount == s2LoopLimit：末次迭代，FlashUpdateLast + CopyOut
// 中间迭代：FlashUpdateNew
```

由于 `s2LoopCount` 保留原始值（sink=0, 正常 KV=1,2,...），且 `s2LoopLimit` 已 +1，ProcessVec2 的首次/末次判断逻辑天然正确，**无需修改**。

#### 4.1.4.5 运行时参数变更

**ConstInfo / CVSharedParams 均不新增字段**。

hasSink 通过编译期模板参数判定，无需运行时传递。

**RunInfo 新增字段**：
```cpp
// util_regbase.h RunInfo 中新增
struct RunInfo {
    COMMON_RUN_INFO;
    ...
    bool isSinkIter = false;  // 当前是否为 Sink 迭代
};
```

**isSinkIter 的设置方式**：由 `SetRunInfo` 方法根据 `s2LoopCount == 0` 自动设置，不需要独立的 `SetRunInfoForSink` 方法：
- `s2LoopCount == 0` 且 `hasSink` → `isSinkIter = true`
- 其他情况 → `isSinkIter = false`

**isSinkIter 的使用位置**：
| 使用位置 | 用途 |
| -------- | ---- |
| ProcessVec0 | 判断是否走 pass-through 路径 |
| IterateBmm1QSFA | 判断是否搬运 sink BF16 到 L1 |
| ProcessVec1 | 判断 isFirstEver（首次初始化 max/sum） |
| ComputeBmm1Tail | sink 迭代直接返回，跳过尾块计算 |

#### 4.1.4.6 核间同步协议

Sink 迭代融入 S2 循环后，核间同步协议与正常迭代完全一致，仅 Vec0 阶段行为不同：

```
所有迭代（统一协议）：
  AIV Vec0: WaitCrossCore(L1) → [搬运/pass-through] → SetCrossCore(L1)
  AIC BMM1: WaitCrossCore(L1) → [可选: sink GM→L1] → Q×K → SetCrossCore(UB)
  AIV Vec1: WaitCrossCore(UB) → Softmax → SetCrossCore(L1)
  AIC BMM2: WaitCrossCore(L1) → P×V → SetCrossCore(UB)
  AIV Vec2: WaitCrossCore(UB) → FlashUpdate → SetCrossCore(UB)
```

**Sink 迭代（s2LoopCount=0）的特殊行为**：
- AIV Vec0：`WaitCrossCore → SetCrossCore`（pass-through，不搬运数据）
- AIC BMM1：`WaitCrossCore` 后先执行 `CopyToL1Nd2Nz` 搬运 sink BF16 到 L1，再执行 MatmulK

**设计优势**：
- 核间同步时序完全不变，Sink 只是 S2 循环中的一个普通迭代
- 不引入额外的同步点或特殊的同步协议
- Vec1/BMM2/Vec2 阶段完全无感知 sink 的存在

### 4.1.5 异常场景设计
| 异常场景 | 处理方式 | 处理层级 |
| -------- | -------- | -------- |
| key_sink 为 nullptr，value_sink 也为 nullptr | HAS_SINK=0，TilingKey 走无 sink 路径 | Host Tiling + 编译期 |
| key_sink 为 nullptr，value_sink 不为 nullptr | Tiling 阶段报错（反向不对称，输入不合法） | Host Tiling Check |
| key_sink 存在但 kvLayout 非 PA_BSND | Tiling 阶段报错 | Host Tiling Check |
| key_sink 存在但 desc 为 nullptr | Tiling 阶段报错 | Host Tiling Check |
| key_sink dtype 非 BF16 | Tiling 阶段报错 | Host Tiling Check |
| key_sink shape != [128, N1, 576] | Tiling 阶段报错 | Host Tiling Check |
| key_sink 存在但 value_sink 为 nullptr | Tiling 阶段报错 | Host Tiling Check |
| value_sink 存在但 desc 为 nullptr | Tiling 阶段报错 | Host Tiling Check |
| value_sink dtype 非 BF16 | Tiling 阶段报错 | Host Tiling Check |
| value_sink shape != [128, N1, 512] | Tiling 阶段报错 | Host Tiling Check |

### 4.1.6 支持确定性计算设计
不涉及。Sink 迭代作为 S2 循环的 s2LoopCount=0 执行，计算顺序确定，不引入非确定性。

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
