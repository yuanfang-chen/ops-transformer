# FlashAttention 新算子介绍文档

## 1. 概述

`FlashAttention`（aclnnFlashAttention）是对原有 `flash_attention_score`（训练FA）和 `fused_infer_attention_score`（推理FA）两个算子的**训练推理归一**实现。

**设计目标**：用一个算子同时支持：
- 训练正向过程（输出 `softmax_lse` 供反向传播）
- 推理过程（支持分页KV缓存、变长序列、预计算tiling）

**约束**：
- 仅支持非量化场景（q/k/v 数据类型为 FLOAT16 或 BFLOAT16）
- 新算子不依赖原有 `flash_attention_score` 或 `fused_infer_attention_score` 的任何文件
- 接口以新接口文档 `aclnnFlashAttention新接口文档.md` 为准

---

## 2. 目录结构

```
attention/flash_attention/
├── CMakeLists.txt
├── FLASH_ATTENTION_介绍文档.md          ← 本文档
├── op_api/
│   ├── aclnn_flash_attention.h          ← aclnn两段式接口声明（公开头文件）
│   ├── aclnn_flash_attention.cpp        ← aclnn两段式接口完整实现（含全量校验）
│   ├── flash_attention.h                ← l0op::FlashAttention 内部接口声明
│   └── flash_attention.cpp              ← l0op::FlashAttention 实现（注册算子、InferShape、Launch）
├── op_host/
│   ├── CMakeLists.txt
│   ├── flash_attention_def.cpp          ← 算子定义（输入/输出/属性声明）
│   ├── flash_attention_infershape.cpp   ← InferShape / InferDataType 实现
│   ├── flash_attention_tiling.h         ← Tiling数据结构声明
│   └── flash_attention_tiling.cpp       ← Tiling计算实现（接口桩）
└── op_kernel/
    └── flash_attention.cpp              ← Kernel实现（接口桩）
```

---

## 3. 新接口说明（aclnnFlashAttention）

### 3.1 第一段接口 GetWorkspaceSize

```cpp
aclnnStatus aclnnFlashAttentionGetWorkspaceSize(
    const aclTensor *q,                    // 必选，query tensor
    const aclTensor *k,                    // 必选，key tensor
    const aclTensor *v,                    // 必选，value tensor
    const aclTensor *blockTableOptional,   // 可选，分页KV块映射表（INT32）
    const aclTensor *cuSeqlensQOptional,   // 可选，query累积序列长度（INT32，1D）
    const aclTensor *cuSeqlensKvOptional,  // 可选，kv累积序列长度（INT32，1D）
    const aclTensor *sequsedQOptional,     // 可选，query各batch实际序列长度（INT32，1D）
    const aclTensor *sequsedKvOptional,    // 可选，kv各batch实际序列长度（INT32，1D）
    const aclTensor *sinksOptional,        // 可选，可学习sink权重（FLOAT32）
    const aclTensor *metadataOptional,     // 可选，预计算tiling方案（INT32）
    float softmaxMode,                     // 属性可选，softmax缩放系数（0.0=1/sqrt(D)）
    int64_t maskMode,                      // 属性可选，掩码模式（0-4）
    int64_t winLeft,                       // 属性可选，左窗口大小（maskMode=4有效）
    int64_t winRight,                      // 属性可选，右窗口大小（maskMode=4有效）
    const char *layoutQ,                   // 属性可选，q布局（"BSND"/"BNSD"/"TND"）
    const char *layoutKv,                  // 属性可选，kv布局（"BSND"/"TND"/"PA_ND"/"PA_Nz"）
    const char *layoutOut,                 // 属性可选，输出布局（"BSND"/"BNSD"/"TND"）
    int64_t returnSoftmaxLse,              // 属性可选，是否输出softmax_lse（0/1）
    int64_t deterministic,                 // 属性可选，是否确定性计算（0/1）
    const aclTensor *attentionOut,         // 必选输出，attention结果
    const aclTensor *softmaxLseOptional,   // 可选输出，softmax_lse（returnSoftmaxLse=1时必须提供）
    uint64_t *workspaceSize,               // 输出，workspace字节数
    aclOpExecutor **executor               // 输出，执行器句柄
);
```

### 3.2 第二段接口 Execute

```cpp
aclnnStatus aclnnFlashAttention(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    const aclrtStream stream
);
```

---

## 4. 新旧接口参数对应关系

### 4.1 与 flash_attention_score（训练FA）的对应

| 新接口参数 | 老接口参数 | 说明 |
|---|---|---|
| `q` | `query` | 同义 |
| `k` | `key` | 同义 |
| `v` | `value` | 同义 |
| `cuSeqlensQOptional` | `actualSeqQLenOptional (aclIntArray*)` | 类型变更：老接口为CPU端IntArray，新接口为设备端INT32 Tensor，支持on-device |
| `cuSeqlensKvOptional` | `actualSeqKvLenOptional (aclIntArray*)` | 同上 |
| `sequsedQOptional` | `actualSeqQLenOptional` | 语义细化：cu_seqlens为前缀和形式，seqused为各batch独立长度 |
| `sequsedKvOptional` | `actualSeqKvLenOptional` | 同上 |
| `sinksOptional` | `sinkOptional` | 同义，可学习sink权重 |
| `softmaxMode` | `scaleValue (double)` | 类型变更：double→float |
| `maskMode` | `sparseMode (int64_t)` | 同义，取值范围0-4一致 |
| `winLeft` | `preTokens (int64_t)` | maskMode=4时生效 |
| `winRight` | `nextTokens (int64_t)` | maskMode=4时生效 |
| `layoutQ`/`layoutKv`/`layoutOut` | `inputLayout (char*)` | **拆分**：老接口用单一字符串（BSH/SBH/BSND/BNSD/TND），新接口按q/kv/out分别指定，且仅支持BSND/BNSD/TND（q和out）及BSND/TND/PA_ND/PA_Nz（kv） |
| `returnSoftmaxLse` | 隐式（通过softmaxMaxOut/softmaxSumOut/softmaxOutOut存在性判断） | **显式化**：新接口通过独立标志位控制 |
| `deterministic` | 不存在 | **新增**：确定性计算控制 |
| `metadataOptional` | 不存在 | **新增**：预计算tiling方案，推理时可传入以减少tiling开销 |
| `blockTableOptional` | 不存在（训练FA不支持PA） | **新增**：分页KV缓存支持 |
| `attentionOut` | `attentionOutOut` | 同义，必选输出 |
| `softmaxLseOptional` | `softmaxMaxOut` + `softmaxSumOut` | **合并**：老接口输出softmax_max和softmax_sum两个Tensor；新接口合并为单一softmax_lse（log-sum-exp） |
| **移除** | `realShiftOptional` (PSE shift) | 新接口不含PSE |
| **移除** | `dropMaskOptional` | 新接口不含dropout |
| **移除** | `paddingMaskOptional` | 由seqused取代 |
| **移除** | `attenMaskOptional` (外部Tensor掩码) | 新接口掩码通过maskMode属性控制 |
| **移除** | `prefixOptional` | 新接口不含prefix |
| **移除** | `qStartIdxOptional`/`kvStartIdxOptional` | 新接口不含起始索引 |
| **移除** | `headNum (int64_t)` | **隐式**：从q的shape和layoutQ中直接解析N维度 |
| **移除** | `keepProb (double)` | 移除dropout支持 |
| **移除** | `innerPrecise (int64_t)` | 新接口不含精度模式控制 |
| **移除** | `pseType (int64_t)` | 移除PSE类型控制 |
| **移除** | `softmaxOutOut` | 新接口的softmax_lse替代了max+sum的方式 |

### 4.2 与 fused_infer_attention_score（推理FA）的对应

| 新接口参数 | 老接口参数 | 说明 |
|---|---|---|
| `q` | `query` | 同义 |
| `k` | `key (aclTensorList*)` | **变更**：老接口支持多路kv（TensorList），新接口为单一Tensor |
| `v` | `value (aclTensorList*)` | **变更**：同上 |
| `blockTableOptional` | `blockTableOptional` | 同义，分页KV块映射表 |
| `cuSeqlensQOptional` | `actualSeqLengthsOptional (aclIntArray*)` | 类型变更：aclIntArray→INT32 Tensor |
| `cuSeqlensKvOptional` | `actualSeqLengthsKvOptional (aclIntArray*)` | 同上 |
| `sequsedQOptional` | `actualSeqLengthsOptional` | 语义细化 |
| `sequsedKvOptional` | `actualSeqLengthsKvOptional` | 语义细化 |
| `sinksOptional` | `learnableSinkOptional` | 同义 |
| `metadataOptional` | 不存在 | **新增**：预计算tiling方案 |
| `softmaxMode` | `scaleValue (double)` | 类型变更 |
| `maskMode` | `sparseMode (int64_t)` | 同义 |
| `winLeft` | `preTokens (int64_t)` | 同义 |
| `winRight` | `nextTokens (int64_t)` | 同义 |
| `layoutQ`/`layoutKv`/`layoutOut` | `inputLayout (char*)` | **拆分**：老接口单一字符串（BSND/TND/BSH等），新接口按q/kv/out分别指定 |
| `returnSoftmaxLse` | `softmaxLseFlag (bool)` | 同义，类型从bool改为int64_t |
| `deterministic` | 不存在 | **新增** |
| `attentionOut` | `attentionOut` | 同义 |
| `softmaxLseOptional` | `softmaxLse` | 同义 |
| **移除** | `deqScale1/quantScale1/deqScale2/...` | 新接口不含量化参数 |
| **移除** | `antiquantScale/antiquantOffset/...` | 同上，仅非量化 |
| **移除** | `keyAntiquantScale/keyAntiquantOffset/...` | 同上 |
| **移除** | `queryRopeOptional`/`keyRopeOptional` | 新接口不含RoPE |
| **移除** | `keySharedPrefixOptional`/`valueSharedPrefixOptional` | 新接口不含shared prefix |
| **移除** | `actualSharedPrefixLenOptional` | 同上 |
| **移除** | `numHeads (int64_t)` | 隐式从shape解析 |
| **移除** | `numKeyValueHeads (int64_t)` | 隐式从kv shape解析 |
| **移除** | `innerPrecise` | 移除 |
| **移除** | `blockSize (int64_t)` | 移入tiling计算 |
| **移除** | `antiquantMode`/`keyAntiquantMode`/... | 移除量化模式 |
| **移除** | `pseType` | 移除 |
| **移除** | `queryQuantMode` | 移除 |
| **移除** | `queryPaddingSizeOptional`/`kvPaddingSizeOptional` | 由seqused替代 |

---

## 5. 新代码参考了老代码的哪些部分

### 5.1 参考 flash_attention_score 的部分

| 新代码文件/内容 | 参考来源 | 说明 |
|---|---|---|
| `aclnn_flash_attention.cpp` 整体框架 | `flash_attention_score/op_api/aclnn_flash_attention_score.cpp` | 参考aclnn两段式接口的整体实现框架：参数校验→CREATE_EXECUTOR→连续化→l0op调用→ViewCopy→ReleaseTo |
| `CheckMandatoryParams` 函数 | `CheckFaParam` 函数 | 参考必选参数非空校验的模式，增加了layoutQ/layoutKv/layoutOut的校验 |
| `CheckInputDtype` 函数 | `InputDtypeCheck` 函数 | 参考数据类型校验逻辑，简化去掉PSE/量化/sink相关检查，增加softmaxLse的FLOAT32检查 |
| `CheckStorageFormat` 函数 | `CheckFormat` 函数 | 参考NZ格式不支持的检查模式 |
| `MakeContiguous` 函数 | `Contiguous` 函数 | 参考各输入tensor逐一调用l0op::Contiguous的模式 |
| `ProcessSinks` 函数 | `aclnnFlashAttentionScoreV3GetWorkspaceSize` 中 sink shape为0时置nullptr | 参考sink空shape处理逻辑 |
| `BuildSoftmaxLsePlaceholder` | V5中 softmaxLseFlag=false时构造placeholder tensor的代码 | 参考通过aclCreateTensor构造shape={0}的占位符 |
| `L2_DFX_PHASE_1/2` 宏使用 | 各GetWorkspaceSize/Execute函数 | 参考DFX打点模式 |
| `CREATE_EXECUTOR` / `ReleaseTo` 模式 | 所有GetWorkspaceSize函数 | 参考executor生命周期管理模式 |
| `IsEmpty()` 快速返回 | 所有GetWorkspaceSize函数中 `softmaxMaxOut->IsEmpty()` 判断 | 参考空tensor时直接返回的优化 |
| `AnalysisAxis*/AnalysisInput` | `AnalysisAxis/AnalysisAxisForBsnd/AnalysisAxisForTnd` 等函数 | 参考从shape和layout解析B/N/S/D维度的设计思路，重构为基于新layout枚举 |
| `aclnnFlashAttentionScore/Execute` 的固定写法 | 各Execute函数 | 参考 `CommonOpExecutorRun` 的固定调用 |
| `flash_attention.cpp` (l0op) 整体框架 | `flash_attention_score.cpp` (l0op) | 参考OP_TYPE_REGISTER、AllocTensor for optional inputs、INFER_SHAPE/ADD_TO_LAUNCHER_LIST_AICORE宏的使用 |
| `flash_attention_def.cpp` 整体框架 | `flash_attention_score_def.cpp` | 参考OpDef子类的定义结构（Input/Output/Attr链式调用）、ParamType/DataType/Format/AutoContiguous等设置方式 |
| `flash_attention_infershape.cpp` 整体框架 | `flash_attention_score_infershape.cpp` | 参考InferShapeFlashAttentionScore的结构：从attrs读取layout/headNum，从input shape推导output shape；使用`IMPL_OP_INFERSHAPE`宏注册 |
| `flash_attention_tiling.h/cpp` | `flash_attention_score_tiling_common.h` / `flash_attention_score_tiling.cpp` | 参考TilingContext参数读取模式和tiling数据结构设计思路 |

### 5.2 参考 fused_infer_attention_score 的部分

| 新代码文件/内容 | 参考来源 | 说明 |
|---|---|---|
| 接口设计（q/k/v为单Tensor而非TensorList） | `aclnn_fused_infer_attention_score_v5.cpp` | 对比学习接口演进方向，新接口采用单Tensor与老训练接口对齐，简化TensorList预处理 |
| `blockTableOptional` 参数设计 | `blockTableOptional` in v5 | 参考分页KV缓存参数的传递方式 |
| `returnSoftmaxLse` / softmaxLse 处理 | `softmaxLseFlag` + placeholder构造 | 参考softmaxLse条件输出的处理模式（通过flag控制是否输出，false时用placeholder） |
| `metadataOptional` 概念 | 无对应（新设计） | 灵感来自fused_infer对tiling预计算的关注（推理性能优化） |
| `cuSeqlens`/`seqused` 两套变长序列参数 | `actualSeqLengthsOptional` / `actualSeqLengthsKvOptional` | 对原有单一IntArray参数语义细化为cu_seqlens（前缀和）和seqused（各batch独立长度）两种模式 |
| `layoutKv` 支持 PA_ND/PA_Nz | `inputLayout` 含PA格式 | 参考分页注意力layout的支持 |
| `CheckAuxTensorDtype` 中 INT32 检查 | INT32 block_table/seqlen 参数 | 参考fused_infer中block_table等辅助参数为INT32的设计规范 |
| `InferShapeFlashAttention` 中 headDimV 处理 | `fused_infer_attention_score_infershape.cpp` | 参考kv head dim与q head dim可能不同（MQA/MLA）的infershape逻辑 |

---

## 6. 关键设计决策说明

### 6.1 headNum 隐式化
老接口要求用户显式传入 `headNum`，新接口通过 `layoutQ` 直接从 shape 解析：
- `BSND`: `numHeadsQ = shape[2]`
- `BNSD`: `numHeadsQ = shape[1]`  
- `TND`: `numHeadsQ = shape[1]`

优势：减少用户参数，避免shape和headNum不一致的错误。

### 6.2 layout 三分法
老接口用单一 `inputLayout` 字符串（如 "BSND"）同时约束 q/k/v/out 的 layout。新接口拆分为 `layoutQ`、`layoutKv`、`layoutOut` 三个独立属性：
- 允许 q 和 out 使用不同 layout（如 q 为 BSND，out 为 BNSD）
- 允许 kv 使用特殊的分页格式（PA_ND/PA_Nz），与 q 的 layout 完全解耦

### 6.3 cu_seqlens vs seqused
两种变长序列表示方式满足不同场景：
- `cu_seqlens`：前缀和形式，与 FlashAttention2 论文接口对齐，适合 TND layout 下高效访问
- `seqused`：各 batch 独立长度，适合 padded batch 模式（与 fused_infer 的 `actualSeqLengths` 对应）

### 6.4 softmax_lse 代替 max+sum
老训练接口输出 `softmax_max`（fp32）和 `softmax_sum`（fp32）两个 Tensor，供反向传播使用。新接口直接输出 `softmax_lse`（log-sum-exp，fp32），语义更清晰，存储也更紧凑。

反向传播时使用 lse 重构的方式：$\text{softmax\_lse} = \log(\text{softmax\_sum}) + \text{softmax\_max}$

### 6.5 metadata 预计算 tiling
推理场景中 tiling 计算（根据 shape/硬件资源切分矩阵块）可在模型编译时完成，运行时通过 `metadata` Tensor 传入，避免重复计算。训练场景中 metadata 为 nullptr，tiling 在算子内部计算。

---

## 7. 使用示例

### 7.1 训练正向传播（返回 softmax_lse）

```cpp
// q: (B, S, N, D), layoutQ = "BSND"
// k/v: (B, S, N, D), layoutKv = "BSND"
aclnnFlashAttentionGetWorkspaceSize(
    q, k, v,
    nullptr,                // blockTable (无分页)
    nullptr, nullptr,       // cu_seqlens
    nullptr, nullptr,       // seqused
    nullptr,                // sinks
    nullptr,                // metadata
    0.0f,                   // softmaxMode (使用1/sqrt(D))
    1,                      // maskMode=1 (因果掩码)
    0, 0,                   // winLeft/winRight (不用)
    "BSND", "BSND", "BSND",// layout
    1,                      // returnSoftmaxLse=1（训练时输出lse）
    0,                      // deterministic=0
    attentionOut,
    softmaxLse,             // 训练时必须提供
    &workspaceSize, &executor);
aclnnFlashAttention(workspace, workspaceSize, executor, stream);
```

### 7.2 推理（分页KV缓存，变长序列）

```cpp
// q: (T_q, N, D), layoutQ = "TND"
// k/v: (NumBlocks, BlockSize, N_kv, D), layoutKv = "PA_ND"
aclnnFlashAttentionGetWorkspaceSize(
    q, k, v,
    blockTable,             // 分页块映射表
    nullptr, nullptr,       // cu_seqlens（不用，用seqused）
    sequsedQ, sequsedKv,    // seqused（各batch实际序列长度）
    nullptr,                // sinks
    metadata,               // 预计算tiling方案
    0.0f,                   // softmaxMode
    0,                      // maskMode=0 (无掩码)
    0, 0,
    "TND", "PA_ND", "TND",  // layout
    0,                      // returnSoftmaxLse=0（推理不需要）
    0,
    attentionOut,
    nullptr,                // 推理时不输出lse
    &workspaceSize, &executor);
aclnnFlashAttention(workspace, workspaceSize, executor, stream);
```

---

## 8. 与新接口文档的对应关系

| 文档参数 | 代码参数 | 文档描述 |
|---|---|---|
| `q` | `q` | query tensor，FLOAT16/BFLOAT16，ND格式，0/3/4维 |
| `k` | `k` | key tensor，与q类型一致 |
| `v` | `v` | value tensor，与q类型一致 |
| `block_table` | `blockTableOptional` | 类似fused_infer的blocktableoptional，INT32 |
| `cu_seqlens_q` | `cuSeqlensQOptional` | 从fused_infer的actualseqlengthsoptional转换而来，INT32 |
| `cu_seqlens_kv` | `cuSeqlensKvOptional` | 从fused_infer的actualseqlengthskvoptional转换而来，INT32 |
| `seqused_q` | `sequsedQOptional` | 从fused_infer的actualseqlengthsoptional转换而来，INT32 |
| `seqused_kv` | `sequsedKvOptional` | 从fused_infer的actualseqlengthskvoptional转换而来，INT32 |
| `sinks` | `sinksOptional` | FLOAT32 |
| `metadata` | `metadataOptional` | 预计算tiling方案，INT32 |
| `softmax_mode` | `softmaxMode` | FLOAT，从fused_infer的scaleValue转换而来 |
| `mask_mode` | `maskMode` | INT，从fused_infer的sparsemode转换而来 |
| `win_left` | `winLeft` | INT，sparsemode=4时的pretoken |
| `win_right` | `winRight` | INT，sparsemode=4时的nexttoken |
| `layout_q` | `layoutQ` | STRING，BSND/TND/BNSD，从inputlayout分离 |
| `layout_kv` | `layoutKv` | STRING，BSND/TND/PA_ND/PA_Nz，从inputlayout分离 |
| `layout_out` | `layoutOut` | STRING，BSND/TND/BNSD，从inputlayout分离 |
| `return_softmax_lse` | `returnSoftmaxLse` | INT，标志位，代表softmax_lse是否需要输出 |
| `deterministic` | `deterministic` | INT，是否进行确定性计算 |
| `attentionOut` | `attentionOut` | FA算子输出，FLOAT16/BFLOAT16，shape与query一致 |
| `softmax_lse` | `softmaxLseOptional` | FLOAT32，2维或3维 |

---

*文档版本：v1.0，2025年3月*
