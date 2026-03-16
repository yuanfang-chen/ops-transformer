# FlashAttentionScoreV4 — 训练推理归一 Flash Attention 算子

## 一、概述

`FlashAttentionScoreV4` 是在 `flash_attention_score`（训练 FA）和 `fused_floyd_attention`（推理 FA）基础上完成**训练推理归一**的新算子。

**归一目标**：一个算子同时支持：
- **训练正向过程**：输出 `softmax_max` / `softmax_sum` 供反向梯度计算使用。
- **推理过程**：调用方不关注 `softmax_max` / `softmax_sum` 输出，计算流程与训练共享。

**非量化限制**：仅支持 `FLOAT16`、`BFLOAT16`、`FLOAT32`，不包含 FP8 量化路径。

**接口来源**：接口定义参照 `aclnnFlashAttention新接口文档.md`（`aclnnFlashAttentionScoreV4`）。

---

## 二、新接口说明

### 2.1 C++ 算子接口

#### 算子注册名称
```
FlashAttentionScoreV4
```

#### 内核函数名称
```cpp
flash_attention_score_v4
```

#### 两段式 ACLNN 接口（参考文档）
```cpp
aclnnStatus aclnnFlashAttentionScoreV4GetWorkspaceSize(
    const aclTensor   *query,
    const aclTensor   *key,
    const aclTensor   *value,
    const aclTensor   *realShiftOptional,       // PSE，可选
    const aclTensor   *dropMaskOptional,         // 外部 DropMask，可选
    const aclTensor   *paddingMaskOptional,      // 暂未使用，可选
    const aclTensor   *attenMaskOptional,        // 注意力掩码，可选
    const aclTensor   *queryRopeOptional,        // Query RoPE，可选
    const aclTensor   *keyRopeOptional,          // Key RoPE，可选
    const aclTensor   *dScaleQOptional,          // 量化参数（非量化场景传 nullptr）
    const aclTensor   *dScaleKOptional,
    const aclTensor   *dScaleVOptional,
    const aclTensor   *sinkOptional,             // 保留参数
    const aclIntArray *prefixOptional,
    const aclIntArray *actualSeqQLenOptional,
    const aclIntArray *actualSeqKvLenOptional,
    const aclIntArray *qStartIdxOptional,
    const aclIntArray *kvStartIdxOptional,
    double             scaleValue,
    double             keepProb,
    int64_t            preTokens,
    int64_t            nextTokens,
    int64_t            headNum,
    char              *inputLayout,              // "BSH"|"SBH"|"BSND"|"BNSD"|"TND"
    int64_t            innerPrecise,
    int64_t            sparseMode,
    int64_t            outDtype,
    int64_t            pseType,
    char              *softmaxOutLayout,         // 保留参数
    int64_t            seed,                     // V4 新增：Dropout seed
    int64_t            offset,                   // V4 新增：Dropout offset
    const aclTensor   *softmaxMaxOut,
    const aclTensor   *softmaxSumOut,
    const aclTensor   *softmaxOutOut,
    const aclTensor   *attentionOutOut,
    uint64_t          *workspaceSize,
    aclOpExecutor    **executor)

aclnnStatus aclnnFlashAttentionScoreV4(
    void             *workspace,
    uint64_t          workspaceSize,
    aclOpExecutor    *executor,
    const aclrtStream stream)
```

### 2.2 算子属性列表（Attr 索引）

| 索引 | 属性名              | 类型   | 必选/可选 | 默认值         | 说明 |
|------|---------------------|--------|-----------|----------------|------|
| 0    | `scale_value`       | float  | 可选      | 1.0            | 缩放系数（公式中的 scale） |
| 1    | `keep_prob`         | float  | 可选      | 1.0            | Dropout 保留率，取值 (0, 1] |
| 2    | `pre_tockens`       | int64  | 可选      | 2147483647     | 稀疏左边界（sliding window） |
| 3    | `next_tockens`      | int64  | 可选      | 2147483647     | 稀疏右边界 |
| 4    | `head_num`          | int64  | **必选**  | —              | 单卡 Q 的 head 数 |
| 5    | `input_layout`      | string | **必选**  | —              | 支持：BSH / SBH / BSND / BNSD / TND |
| 6    | `inner_precise`     | int64  | 可选      | 0              | 2=使能无效行处理 |
| 7    | `sparse_mode`       | int64  | 可选      | 0              | 稀疏模式 0-6 |
| 8    | `out_dtype`         | int64  | 可选      | 0              | 量化输出类型（非量化场景不生效） |
| 9    | `pse_type`          | int64  | 可选      | 1              | 0=先mul再add, 1=先add再mul, 2/3=内部生成PSE |
| 10   | `softmax_out_layout`| string | 可选      | ""             | 保留参数 |
| 11   | `seed`              | int64  | 可选      | 0              | **V4 新增**：Dropout mask 生成 seed |
| 12   | `offset`            | int64  | 可选      | 0              | **V4 新增**：Dropout mask 生成 offset |

### 2.3 输入/输出列表

#### 输入（按索引）

| 索引 | 名称                | 必选/可选 | 数据类型                      | 说明 |
|------|---------------------|-----------|-------------------------------|------|
| 0    | `query`             | 必选      | FLOAT16 / BF16 / FLOAT32      | Q |
| 1    | `key`               | 必选      | 同 query                      | K |
| 2    | `value`             | 必选      | 同 query                      | V |
| 3    | `real_shift`        | 可选      | FLOAT16 / BF16 / FLOAT32      | PSE |
| 4    | `drop_mask`         | 可选      | UINT8                         | 外部 Dropout mask；keepProb<1.0 且为 nullptr 时由 seed/offset 生成 |
| 5    | `padding_mask`      | 可选      | —                             | 暂未使用 |
| 6    | `atten_mask`        | 可选      | BOOL / UINT8                  | 注意力掩码；1 表示不参与计算 |
| 7    | `query_rope`        | 可选      | FLOAT16 / BF16                | Q 的 RoPE 分量 |
| 8    | `key_rope`          | 可选      | FLOAT16 / BF16                | K 的 RoPE 分量 |
| 9    | `d_scale_q`         | 可选      | FLOAT32                       | 量化参数（非量化场景传 nullptr） |
| 10   | `d_scale_k`         | 可选      | FLOAT32                       | 同上 |
| 11   | `d_scale_v`         | 可选      | FLOAT32                       | 同上 |
| 12   | `sink`              | 可选      | —                             | 保留参数 |
| 13   | `prefix`            | 可选      | INT64                         | prefix 稀疏 N 值 |
| 14   | `actual_seq_qlen`   | 可选      | INT64                         | 每 Batch Q 的实际序列长度 |
| 15   | `actual_seq_kvlen`  | 可选      | INT64                         | 每 Batch KV 的实际序列长度 |
| 16   | `q_start_idx`       | 可选      | INT64                         | 外切 Q 起始索引 |
| 17   | `kv_start_idx`      | 可选      | INT64                         | 外切 KV 起始索引 |

#### 输出

| 索引 | 名称            | 数据类型 | Shape                    | 说明 |
|------|-----------------|----------|--------------------------|------|
| 0    | `softmax_max`   | FLOAT32  | [B,N,S,8] 或 [T,N,8]    | 训练反向所需；推理场景可忽略 |
| 1    | `softmax_sum`   | FLOAT32  | 同 softmax_max           | 训练反向所需；推理场景可忽略 |
| 2    | `softmax_out`   | 同 query | [0,0,0,0]（空）          | 保留接口，始终为空 tensor |
| 3    | `attention_out` | 同 query | 同 query（V 的 D 维可不同） | 最终注意力输出 |

---

## 三、与老接口的对应关系

### 3.1 FlashAttentionScore（训练 FA）→ FlashAttentionScoreV4

| FlashAttentionScore 参数 | FlashAttentionScoreV4 参数 | 变化说明 |
|--------------------------|---------------------------|----------|
| `query` (input 0)        | `query` (input 0)         | 无变化 |
| `key` (input 1)          | `key` (input 1)           | 无变化 |
| `value` (input 2)        | `value` (input 2)         | 无变化 |
| `real_shift` (input 3)   | `real_shift` (input 3)    | 无变化（PSE） |
| `drop_mask` (input 4)    | `drop_mask` (input 4)     | 无变化；V4 新增当为 nullptr 且 keepProb<1.0 时自动用 seed/offset 生成 |
| `padding_mask` (input 5) | `padding_mask` (input 5)  | 无变化（暂未使用） |
| `atten_mask` (input 6)   | `atten_mask` (input 6)    | 无变化 |
| `prefix` (input 7)       | `prefix` (input 13)       | 索引变化（新增 query_rope/key_rope 等输入） |
| `actual_seq_qlen` (input 8) | `actual_seq_qlen` (input 14) | 索引变化 |
| `actual_seq_kvlen` (input 9) | `actual_seq_kvlen` (input 15) | 索引变化 |
| `q_start_idx` (input 10) | `q_start_idx` (input 16)  | 索引变化 |
| `kv_start_idx` (input 11)| `kv_start_idx` (input 17) | 索引变化 |
| `d_scale_q` (input 12)   | `d_scale_q` (input 9)     | 索引变化（非量化传 nullptr） |
| `d_scale_k` (input 13)   | `d_scale_k` (input 10)    | 同上 |
| `d_scale_v` (input 14)   | `d_scale_v` (input 11)    | 同上 |
| `query_rope` (input 15)  | `query_rope` (input 7)    | 索引变化，位置调整至 atten_mask 之后 |
| `key_rope` (input 16)    | `key_rope` (input 8)      | 同上 |
| `sink` (input 17)        | `sink` (input 12)         | 索引变化 |
| `p_scale` (input 18)     | **已移除**                | 仅量化场景使用，V4 非量化不包含 |
| `softmax_max` (output 0) | `softmax_max` (output 0)  | 无变化 |
| `softmax_sum` (output 1) | `softmax_sum` (output 1)  | 无变化 |
| `softmax_out` (output 2) | `softmax_out` (output 2)  | 无变化（空 tensor） |
| `attention_out` (output 3)| `attention_out` (output 3)| 无变化 |
| Attr `scale_value` (idx 0)| Attr `scale_value` (idx 0)| 无变化 |
| Attr `keep_prob` (idx 1) | Attr `keep_prob` (idx 1)  | 无变化 |
| Attr `pre_tockens` (idx 2)| Attr `pre_tockens` (idx 2)| 无变化 |
| Attr `next_tockens` (idx 3)| Attr `next_tockens` (idx 3)| 无变化 |
| Attr `head_num` (idx 4)  | Attr `head_num` (idx 4)   | 无变化 |
| Attr `input_layout` (idx 5)| Attr `input_layout` (idx 5)| 无变化 |
| Attr `inner_precise` (idx 6)| Attr `inner_precise` (idx 6)| 无变化 |
| Attr `sparse_mode` (idx 7)| Attr `sparse_mode` (idx 7)| 无变化 |
| Attr `pse_type` (idx 8)  | Attr `pse_type` (idx 9)   | 索引变化（out_dtype 插入 idx 8） |
| Attr `seed` (idx 9)      | Attr `seed` (idx 11)      | 索引变化；V4 **明确规范**其行为 |
| Attr `offset` (idx 10)   | Attr `offset` (idx 12)    | 索引变化；同上 |
| Attr `out_dtype` (idx 11)| Attr `out_dtype` (idx 8)  | 索引变化（非量化场景不生效） |
| Attr `softmax_out_layout`(idx 12)| Attr `softmax_out_layout` (idx 10)| 索引变化 |
| **无**                   | Attr `seed` (idx 11) **V4 规范** | V4 新增明确语义：keepProb<1.0 且无外部 dropMask 时用 seed/offset 生成 |
| FP8 数据类型             | **已移除**                | 非量化 |
| `p_scale` 量化输入       | **已移除**                | 非量化 |

### 3.2 FusedFloydAttention（推理 FA）→ FlashAttentionScoreV4

| FusedFloydAttention 参数 | FlashAttentionScoreV4 参数 | 变化说明 |
|--------------------------|---------------------------|----------|
| `query` [B,H,N,S,D]     | `query` [B,N,S,D] 等     | 格式从专用 BHNSD 5D 改为通用多格式；H 维（层级）需在框架层处理 |
| `key_1` [B,H,N,S2,D]    | `key` [B,N,S2,D] 等      | 双 K/V 合并为单 K/V；Floyd 双轮次计算在框架层拆分为两次 V4 调用 |
| `value_1`                | `value`                   | 同上 |
| `key_2`                  | **不直接对应**            | 第二组 K/V 对应第二次 V4 调用的 key |
| `value_2`                | **不直接对应**            | 第二次 V4 调用的 value |
| `atten_mask` [B,1,N,1,S]| `atten_mask` [B,N,S,S] 等 | shape 格式变更 |
| `softmax_max` [B,H,N,S,8]| `softmax_max` [B,N,S,8]  | H 维被吸收（单次 V4 调用无层级概念） |
| `softmax_sum`            | `softmax_sum`             | 同上 |
| `attention_out` [B,H,N,S,D]| `attention_out` [B,N,S,D]| H 维被吸收 |
| Attr `scale_value`       | Attr `scale_value` (idx 0)| 无变化 |
| **无 keepProb**          | Attr `keep_prob`          | V4 新增（推理场景保持 1.0） |
| **无 atten_mask 稀疏控制**| Attr `sparse_mode` 等    | V4 新增多种稀疏模式 |
| 硬件：910b / 910_93      | 910b / 910_93 / ascend950 | V4 扩展至 950 系列 |

---

## 四、新算子代码参考来源

### 4.1 来自 `flash_attention_score`

| 文件 | 参考内容 |
|------|---------|
| `flash_attention_score_def.cpp` | Op 注册框架（OpDef / OP_ADD），数据类型组合（非 FP8 部分），属性声明（scale_value / keep_prob / pre_tockens 等），AICore 配置模式，`seed` / `offset` 属性的引入 |
| `flash_attention_score_infershape.cpp` | `InferShapeFlashAttentionScore()` 函数逻辑，BSH / SBH / BSND / BNSD / TND 格式的 B / S / T 解析，softmaxMax/softmaxSum/attentionOut 输出 shape 推导 |
| `flash_attention_score_tiling.cpp` | `CheckParams()` 输入合法性校验，`GetEmptyArgs()` 空输入分核参数计算，`IsEmptyInput()` / `IsEmptyInputRegbase()` 两条空输入处理路径，tiling 主函数 `TilingFlashAttentionScore()`，`TilingPrepare` 编译期信息解析，`IMPL_OP_OPTILING` 注册 |
| `flash_attention_score_tiling_common.h` | `FlashAttentionScoreCompileInfo` 结构体定义（V4 按此复制出 `FlashAttentionScoreV4CompileInfo`） |
| `op_kernel/arch32/flash_attention_score_*.h` | 全部非量化内核模板头文件（`FlashAttentionScoreS1s2Bn2gs1`、`FlashAttentionScoreS1Bn2gs1`、`FlashAttentionScoreBn2gs1s2B`、`FlashAttentionVarLenScore`、`FlashAttentionScoreEmptyTensor`、`FlashAttentionScoreDropMaskAdapter`） |
| `op_kernel/flash_attention_score.cpp` | 内核入口宏定义（`COPY_TILING_DATA`、`INVOKE_FA_GENERAL_OP_IMPL` 系列），非量化（FP16/BF16/FP32）模板分发逻辑 |

### 4.2 来自 `fused_floyd_attention`

| 文件 | 参考内容 |
|------|---------|
| `fused_floyd_attention_def.cpp` | 算子注册最小化接口设计参考（scale_value 单属性），推理场景无 keepProb/PSE/prefix 等参数的默认化处理思路 |
| `fused_floyd_attention_tiling.cpp` | `IsEmptyInput()` 中按数据类型（FP16/FP32/BF16）选择 `tilingKey` 的分支方式（V4 tiling 的数据类型驱动分发参考），`GetInstance().DoTilingImpl(context)` 通用 tiling 委托模式 |
| `fused_floyd_attention_tiling_common.cpp` | `FloydCalcTschBlockDim()` 辅助函数（展示推理优化的 block dim 计算方式） |
| `op_kernel/fused_floyd_attention_s1s2_bn2gs1.h` | `FusedFloydAttentionS1s2Bn2gs1` 模板（BNSD 格式推理优化内核）为 V4 的 B 模板（`FlashAttentionScoreBn2gs1s2B`）BNSD 推理路径提供设计参考；V4 中 `UB0=9,UB1=9,Block=0` 分支直接对应 `LAYOUT_BNSD` 推理优化场景 |
| `fused_floyd_attention_tiling.h` | `FusedFloydAttentionGeneralTilingData` 中 `inputParams` 的 `seed`/`offset` 字段（为 V4 seed/offset 传递路径的设计参考） |

---

## 五、训练与推理的使用区别

### 5.1 训练正向（Training Forward）
```python
# 典型训练场景调用
softmax_max, softmax_sum, _, attention_out = FlashAttentionScoreV4(
    query, key, value,
    real_shift=None,
    drop_mask=None,           # 传 None，由 seed/offset 内部生成
    atten_mask=causal_mask,
    scale_value=1.0/sqrt(D),
    keep_prob=0.9,            # dropout rate = 0.1
    head_num=N,
    input_layout="BNSD",
    seed=42,
    offset=0,
)
# softmax_max / softmax_sum 用于反向传播
```

### 5.2 推理（Inference）
```python
# 典型推理场景调用
_, _, _, attention_out = FlashAttentionScoreV4(
    query, key, value,
    real_shift=None,
    drop_mask=None,
    atten_mask=causal_mask,
    scale_value=1.0/sqrt(D),
    keep_prob=1.0,            # 推理无 dropout
    head_num=N,
    input_layout="BNSD",
    seed=0,
    offset=0,
)
# 推理场景忽略 softmax_max / softmax_sum 输出
```

---

## 六、Dropout 新行为（V4 vs 旧接口）

| 条件 | 旧 FlashAttentionScore | 新 FlashAttentionScoreV4 |
|------|----------------------|--------------------------|
| keepProb == 1.0 | 不执行 dropout | 不执行 dropout，seed/offset 不生效 |
| keepProb < 1.0 且 dropMask != nullptr | 使用外部 dropMask | 使用外部 dropMask（优先） |
| keepProb < 1.0 且 dropMask == nullptr | 行为未定义 | **使用 seed/offset 内部生成 dropMask** |

---

## 七、文件结构

```
flash_attention_score_v4/
├── CMakeLists.txt
├── README.md                                          ← 本文档
├── op_api/
│   ├── flash_attention_score_v4.h                    ← L0 算子接口声明
│   ├── flash_attention_score_v4.cpp                  ← L0 算子实现（OP_TYPE_REGISTER）
│   ├── aclnn_flash_attention_score_v4.h              ← 对外 C 接口声明（两段式）
│   └── aclnn_flash_attention_score_v4.cpp            ← ACLNN 第一/二段接口实现
├── op_host/
│   ├── CMakeLists.txt
│   ├── flash_attention_score_v4_def.cpp              ← Op 定义（基于 FAS def.cpp）
│   ├── flash_attention_score_v4_infershape.cpp       ← InferShape（基于 FAS infershape.cpp）
│   ├── flash_attention_score_v4_tiling.cpp           ← Tiling（基于 FAS tiling.cpp）
│   └── flash_attention_score_v4_tiling_common.h     ← CompileInfo 结构
└── op_kernel/
    └── flash_attention_score_v4.cpp                  ← 内核入口（基于 FAS kernel，无 FP8）
```

---

## 八、op_api 层说明

### 8.1 层次关系

```
调用方（PyTorch / 框架）
    │
    ▼
aclnnFlashAttentionScoreV4GetWorkspaceSize()   ← 第一段接口
aclnnFlashAttentionScoreV4()                   ← 第二段接口
    │  (位于 aclnn_flash_attention_score_v4.cpp)
    │  - 参数校验、非量化 dtype 检查
    │  - Contiguous / Pad / Reshape / Transpose 前处理
    ▼
l0op::FlashAttentionScoreV4()                  ← L0 算子
    │  (位于 flash_attention_score_v4.cpp)
    │  - AllocTensor（可选 nullptr 补空）
    │  - ConvertToTensor（IntArray → Tensor）
    │  - INFER_SHAPE + ADD_TO_LAUNCHER_LIST_AICORE
    ▼
FlashAttentionScoreV4 kernel                   ← AscendC 内核
    (位于 op_kernel/flash_attention_score_v4.cpp)
```

### 8.2 op_api 代码参考来源

| 文件 | 参考来源 | 关键差异 |
|------|---------|---------|
| `flash_attention_score_v4.h` | `flash_attention_score/op_api/flash_attention_score.h` | 函数名改为 `FlashAttentionScoreV4`；移除 `pScaleOptional`；调整参数顺序（queryRope/keyRope/dScale 按 V4 排列） |
| `flash_attention_score_v4.cpp` | `flash_attention_score/op_api/flash_attention_score.cpp` | 注册 `FlashAttentionScoreV4`；非量化不进行 FP8 outputDtype 映射；OP_ATTR 顺序与 V4 def.cpp 一致（out_dtype 在 pse_type 前）；移除 p_scale 处理 |
| `aclnn_flash_attention_score_v4.h` | `flash_attention_score/op_api/aclnn_flash_attention_score.h`（V4 函数声明部分）| 函数名改为 `aclnnFlashAttentionScoreV4`；接口参数严格按文档定义 |
| `aclnn_flash_attention_score_v4.cpp` | `flash_attention_score/op_api/aclnn_flash_attention_score.cpp`（`aclnnFlashAttentionScoreV4GetWorkspaceSize` 实现） | 调用 `l0op::FlashAttentionScoreV4`；`InputDtypeCheckV4` 增加非量化限制（拒绝 FP8）；`CheckFormatV4` 独立封装；移除 p_scale；参数传递顺序与 V4 def 一致 |

---

## 八、硬件支持

| 硬件 | 是否支持 | 说明 |
|------|----------|------|
| Ascend 950PR / Ascend 950DT | ✓ | arch35 路径（apt 内核），参见 aclnnFlashAttentionScoreV4 文档 |
| Atlas A2 训练系列（910b） | ✓ | arch32 路径，继承自 flash_attention_score |
| Atlas A3 训练系列（910_93） | ✓ | arch32 路径，继承自 flash_attention_score |
