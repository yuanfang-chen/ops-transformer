# LightningIndexer 算子代码梳理

## 1. 算子概述

### 1.1 功能定位

`LightningIndexer`（简称 LI）是稀疏注意力（Sparse Attention）场景下的索引筛选算子。其核心功能是根据 Query、Key 和 Weights，计算每个 token 对应的 Top-K 个位置索引（`sparse_indices`），可选输出对应值（`sparse_values`）。

该算子服务于 GQA（Grouped Query Attention）架构，通过计算 $QK^T$ 并经权重加权归约后，筛选出注意力分数最高的 K 个位置，用于后续的高效稀疏注意力计算。

### 1.2 计算公式

$$
Indices = \text{Top-}k\left\{ [1]_{1 \times g} @ \left[ (W @ [1]_{1 \times S_k}) \odot \text{ReLU}(Q_{index} @ K_{index}^T) \right] \right\}
$$

其中：
- $Q_{index} \in \mathbb{R}^{gS_1 \times d}$，$g$ 为 GQA group size（通常为 64），$d$ 为 head dim（固定 128）
- $K_{index} \in \mathbb{R}^{S_2 \times d}$
- $W \in \mathbb{R}^{gS_1 \times 1}$，每个位置对应一个 GQA group 的权重
- $[1]_{1 \times g}$ 表示对 GQA group 维度做 ReduceSum
- $\text{ReLU}$ 激活后逐元素乘以权重，再按 group 求和，最后取 Top-K

**计算路径**：
1. Cube 核（AIC）执行矩阵乘法 $QK^T$
2. Vector 核（AIV）执行 ReLU + Weighted ReduceSum
3. Vector 核执行 TopK，输出索引（和可选的值）

### 1.3 支持平台

| SoC | 架构目录 | 数据类型特点 | 核心差异 |
|-----|---------|-------------|---------|
| ascend910b | `op_kernel/` (legacy) | bf16/fp16/fp32 | Cube 结果写 GM workspace，Vector 再读取；Sort32+MrgSort 排序 |
| ascend950 | `op_kernel/arch35/` | bf16/fp16/fp32 | Cube 结果直接写 UB（`dualDstCtl`），Vector 直连消费；直方图 TopK |

---

## 2. 目录结构

```
attention/lightning_indexer/
├── CMakeLists.txt                              # 算子构建入口
├── README.md                                   # API 文档与调用示例
├── docs/
│   ├── aclnnLightningIndexer.md               # aclnn 接口文档
│   └── lightning_indexer_代码梳理.md          # 本文档
│
├── op_host/                                    # ═══ Host 侧（CPU 端）═══
│   ├── CMakeLists.txt                          # Host 构建配置
│   ├── lightning_indexer_def.cpp              # OpDef：算子接口定义
│   ├── lightning_indexer_infershape.cpp       # InferShape：输出 shape 推导
│   ├── lightning_indexer_tiling.h             # Tiling 头文件：LITilingData 结构体
│   └── lightning_indexer_tiling.cpp           # Tiling 实现：切分策略
│
└── op_kernel/                                  # ═══ Kernel 侧（NPU 端）═══
    ├── lightning_indexer.cpp                   # Kernel 入口：__global__ 函数
    ├── lightning_indexer_template_tiling_key.h # 模板参数组合声明（LIType/LI_LAYOUT）
    ├── lightning_indexer_common.h              # 公共定义：ConstInfo/RunInfo/SplitCoreInfo
    │
    ├── arch35/                                 # Ascend 950 (arch35) 实现
    │   ├── lightning_indexer_kernel.h         # 主 Kernel 类：LIPreload（调度中心）
    │   ├── lightning_indexer_service_cube.h   # Cube 服务：LIMatmul（$QK^T$）
    │   ├── lightning_indexer_service_vector.h # Vector 服务：LIVector（ReduceSum+TopK）
    │   └── vf/                                # 微核函数
    │       ├── lightning_indexer_vector1.h    # ReduceSum 微核：BatchMulWeightAndReduceSum
    │       └── lightning_indexer_topk.h       # 直方图 TopK 实现：LITopk
    │
    └── lightning_indexer_kernel.h             # 910 版本（legacy）主控类
        ├── lightning_indexer_service_cube.h   # 910 Cube 服务
        ├── lightning_indexer_service_vector.h # 910 Vector 服务
        └── lightning_indexer_vector.h         # 910 Vector 工具函数集合
```

---

## 3. 输入/输出/属性规格

### 3.1 输入张量

| 序号 | 名称 | 必选 | 数据类型 | Shape | 说明 |
|------|------|------|---------|-------|------|
| 0 | query | 是 | bf16/fp16 | [B, S1, N1, D] 或 [T, N1, D] | Query 张量，N1=N2×G |
| 1 | key | 是 | bf16/fp16 | [B, S2, N2, D] 或 [T, N2, D] | Key 张量 |
| 2 | weights | 是 | bf16/fp16/fp32 | [T, N2, G] 或 [B, S1, N2, G] | 权重张量，G 为 GQA group size |
| 3 | actual_seq_lengths_query | 否 | int32 | [B] | 各 batch 的实际 Query 长度 |
| 4 | actual_seq_lengths_key | 否 | int32 | [B] | 各 batch 的实际 Key 长度 |
| 5 | block_table | 否 | int32 | [B, maxBlockNum] | PageAttention 的 block 映射表 |

### 3.2 输出张量

| 名称 | 数据类型 | Shape | 说明 |
|------|---------|-------|------|
| sparse_indices | int32 | 与 Query 同维度，最后一维替换为 K | Top-K 索引 |
| sparse_values | bf16/fp16 | 同 sparse_indices | Top-K 值（可选，arch35 未实现） |

### 3.3 关键属性

| 属性 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| layout_query | string | "BSND" | Query 布局：BSND/TND |
| layout_key | string | "BSND" | Key 布局：BSND/TND/PA_BSND |
| sparse_count | int | 2048 | Top-K 的 K 值 |
| sparse_mode | int | 3 | 稀疏模式：3 表示 causal mask（下三角） |
| pre_tokens | int64 | max | 前向注意 token 数 |
| next_tokens | int64 | max | 后向注意 token 数 |
| return_value | bool | false | 是否返回 sparse_values（仅训练/非PA支持） |

### 3.4 布局支持

| 布局 | 维度 | 适用张量 | 说明 |
|------|------|---------|------|
| BSND | 4 | Query/Key | Batch-Seq-N-Dim 标准布局 |
| TND | 3 | Query/Key | 累积序列长度布局，适合变长序列 |
| PA_BSND | 4 | Key | PageAttention 物理 block 布局 |

---

## 4. Host 侧逻辑

### 4.1 OpDef（lightning_indexer_def.cpp）

**核心配置**：
- **Query/Key**：支持 `DT_BF16`、`DT_FLOAT16`
- **Weights**：支持 `DT_BF16`、`DT_FLOAT16`、`DT_FLOAT`
- **输出**：`sparse_indices` (int32) + `sparse_values` (bf16/fp16)
- **属性**：`return_value` (bool)、`layout_query/key` (string)、`sparse_count/mode` (int)

**SoC 配置**：
```cpp
aicore_config_95.Input("query").DataType({ge::DT_BF16, ge::DT_FLOAT16})...
aicore_config_95.Input("weights").DataType({ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT})...
this->AICore().AddConfig("ascend950", aicore_config_95);
```

### 4.2 InferShape（lightning_indexer_infershape.cpp）

**输出 Shape 推导**：
```cpp
// BSND 布局
sparseIndicesShape = [B, S1, N2, K]  // K = sparse_count

// TND 布局
sparseIndicesShape = [T, N2, K]
```

**数据类型推导**：
- `sparse_indices` 固定为 `DT_INT32`
- `sparse_values` 与 Query 同类型

### 4.3 Tiling 策略（lightning_indexer_tiling.cpp）

#### 4.3.1 Tiling 主流程

```
DoOpTiling()
  ├── ParseOpParam()           # 解析输入 shape、属性
  ├── CheckAndParseParams()    # 参数校验
  ├── Split()                  # 多核切分
  ├── CalcTilingKey()          # 计算 TilingKey
  ├── FillTilingData()         # 填充 TilingData
  ├── CalcWorkspaceSize()      # 计算 workspace 大小
  └── CalcBlockDim()           # 计算核数（1:2 AIC:AIV）
```

#### 4.3.2 关键切分参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| s1BaseSize | 4 | S1 方向基本块大小 |
| s2BaseSize | 128 | S2 方向基本块大小（arch35）/ 512（legacy） |
| mBaseSize | 256 | gS1 方向基本块（gSize × s1BaseSize = 64×4） |
| usedCoreNum | 24 | 实际使用的核数（24 AIC + 48 AIV） |

#### 4.3.3 TilingData 结构体

```cpp
struct LITilingData {
    // 维度信息
    uint32_t bSize;              // Batch 大小
    uint32_t s1Size;             // Query 序列长度
    uint32_t s2Size;             // Key 序列长度
    uint32_t gSize;              // GQA group size（64）
    uint32_t kHeadNum;           // Key head num（1）
    uint32_t qHeadNum;           // Query head num（=gSize）
    uint32_t headDim;            // Head 维度（128）

    // 切分参数
    uint32_t usedCoreNum;        // 使用的核数
    uint32_t s1BaseSize;         // 4
    uint32_t s2BaseSize;         // 128（arch35）

    // PageAttention
    uint32_t blockSize;          // KV Cache block 大小
    uint32_t maxBlockNumPerBatch; // 每 batch 最大 block 数

    // 功能配置
    uint32_t sparseCount;        // Top-K 值
    uint32_t sparseMode;         // 稀疏模式（3=causal）

    // 同步事件
    uint32_t syncC1V1;           // Cube→Vector 同步事件
    uint32_t syncV1C1;           // Vector→Cube 同步事件
};
```

#### 4.3.4 Workspace 组成

```cpp
// Vector 核的 score workspace（每核独立）
singleCoreScoreSize = s1BaseSize × Align(kSeqSize, s2BaseSize) × sizeof(uint16_t)
totalWorkspaceSize = GetBlockNum() × singleCoreScoreSize
```

#### 4.3.5 参数校验

- `headDim` 必须为 128
- `gSize`（`qHeadNum/kHeadNum`）必须为 64
- `layout_query` 仅支持 BSND/TND
- `layout_key` 仅支持 BSND/TND/PA_BSND
- PageAttention 场景下需满足 blockSize 对齐要求

---

## 5. Kernel 侧逻辑

### 5.1 整体架构：AIC/AIV 异构协同

```
┌─────────────────────────────────────────────────────────────────┐
│                    NPU AI Core (Ascend 950)                      │
│                                                                  │
│   ┌──────────────┐           ┌──────────────┐                   │
│   │   AIC (Cube) │◄──UB───►│  AIV (Vector) │                   │
│   │   LIMatmul   │ dualDst  │    LIVector   │                   │
│   │              │  =1      │               │                   │
│   └──────────────┘           └──────────────┘                   │
│          │                            │                         │
│          ▼                            ▼                         │
│   ┌─────────────────────────────────────────┐                   │
│   │  L1: Query/Key Buf   L0: A/B/C Buffers  │                   │
│   └─────────────────────────────────────────┘                   │
│          │                            │                         │
│          ▼                            ▼                         │
│   ┌─────────────────────────────────────────┐                   │
│   │         Global Memory (HBM)             │                   │
│   │  Query/Key/Weights → Workspace(score)  │                   │
│   └─────────────────────────────────────────────────────────────┘
```

**核间比例**：1 AIC : 2 AIV（`KERNEL_TYPE_MIX_AIC_1_2`）

**跨核同步**：
- `CrossCoreWaitFlag<QLI_SYNC_MODE4>` / `CrossCoreSetFlag<QLI_SYNC_MODE4>`
- Cube→Vector：`CROSS_CV_EVENT`（通知 Cube 结果已写入 UB）
- Vector→Cube：`CROSS_VC_EVENT`（通知 Vector 已消费完 UB 数据）

### 5.2 Kernel 入口（lightning_indexer.cpp）

```cpp
__global__ __aicore__ void lightning_indexer(
    __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *weights,
    __gm__ uint8_t *actualSeqLengthsQ, __gm__ uint8_t *actualSeqLengths,
    __gm__ uint8_t *blocktable, __gm__ uint8_t *sparseIndices,
    __gm__ uint8_t *sparseValues, __gm__ uint8_t *workspace, __gm__ uint8_t *tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    #if (__CCE_AICORE__ == 310)
        // arch35: Ascend 950
        LIPreload<LIType<bfloat16_t, bfloat16_t, int32_t, ...>> op;
        op.Init(...); op.Process();
    #endif
}
```

**模板参数**（`LIType`）：
| 参数 | 含义 | 取值 |
|------|------|------|
| queryType | Query 数据类型 | bfloat16_t / half |
| keyType | Key 数据类型 | bfloat16_t / half |
| outputType | 输出类型 | int32_t |
| pageAttention | 是否启用 PA | true/false |
| layout | Query 布局 | LI_LAYOUT_BSND/TND |
| keyLayout | Key 布局 | LI_LAYOUT_BSND/TND/PA_BSND |
| weightsTypeFlag | Weights 是否为 float | true/false |

### 5.3 主 Kernel 类（arch35/lightning_indexer_kernel.h）

#### 5.3.1 内存层级与 Buffer 布局

**AIC（Cube）侧 Buffer**：
| Buffer | 位置 | 大小 | 说明 |
|--------|------|------|------|
| bufQL1_ | A1 | 2×mBaseSize×D×sizeof(Q_T) | Query L1 Ping-Pong |
| bufKeyL1_ | B1 | 3×S2×D×sizeof(K_T) | Key L1 Ping-Pong |
| bufQL0_ | A2 | 2×mBaseSize×D×sizeof(Q_T) | Query L0A Ping-Pong |
| bufKeyL0_ | B2 | 2×D×S2×sizeof(K_T) | Key L0B Ping-Pong |
| bufL0C_ | CO1 | 2×mBaseSize×S2×sizeof(float) | L0C 输出缓冲 |
| bufUB_ | VECCALC | 2×mBaseSize/2×S2×sizeof(float) | Cube→Vector UB（dualDst） |

**AIV（Vector）侧 Buffer**：
| Buffer | 位置 | 大小 | 说明 |
|--------|------|------|------|
| resMm1Buf_ | VECCALC | 2×mBaseSize/2×S2×sizeof(float) | 读取 Cube 结果（UB） |
| weightBuf_ | VECCALC | 4×s1BaseSize/2×gSize×sizeof(W_T) | Weights Ping-Pong |
| weightFloatBuf_ | VECCALC | 同上，float | Weights cast 后缓存 |
| outBuf_ | VECCALC | 2×s1BaseSize/2×S2×sizeof(uint16_t) | ReduceSum 输出 |
| mrgValueBuf_ | VECCALC | (topkCount+trunkLen)×sizeof(uint16_t) | TopK 归并缓冲 |
| indicesOutBuf_ | VECCALC | (topkCount+64)×sizeof(uint32_t) | TopK 索引输出 |
| scoreOutBuf_ | VECCALC | topkCount×sizeof(uint16_t) | TopK score 输出 |

#### 5.3.2 初始化流程

```cpp
void LIPreload::Init(...) {
    // 1. 解析 TilingData
    InitTilingData(tiling);

    // 2. 多核切分
    SplitCore(aiCoreIdx, usedCoreNum, splitCoreInfo);

    // 3. 绑定 Global Tensor
    if (AIV) {
        scoreGm.SetGlobalBuffer(workspace + aiCoreIdx × singleCoreScoreSize);
        weightsGm.SetGlobalBuffer(weights);
        indiceOutGm.SetGlobalBuffer(sparseIndices);
        vectorService.Init(...);
    } else {
        queryGm.SetGlobalBuffer(query);
        keyGm.SetGlobalBuffer(key);
        matmulService.InitMm1GlobalTensor(...);
    }

    // 4. 初始化 Buffer
    InitBuffers();
}
```

#### 5.3.3 主循环编排

```cpp
void LIPreload::ProcessMain() {
    // 三层循环：bN2LoopIdx → gS1LoopIdx → s2LoopIdx
    for (bN2Idx = bN2Start; bN2Idx <= bN2End; bN2Idx++) {
        for (gS1Idx = gS1Start; gS1Idx <= gS1LoopEnd; gS1Idx++) {
            for (s2Idx = s2Start; s2Idx <= s2LoopEnd; s2Idx++) {
                ProcessBaseBlock(loop++, s2Idx);
            }
        }
    }
}

void ProcessBaseBlock(loop, s2Idx) {
    CalcRunInfo(...);  // 填充 RunInfo
    if (AIC) {
        matmulService.ComputeMm1(runInfo);  // Cube 执行 MM
    } else {
        vectorService.ProcessVec1(runInfo); // Vector 执行 Reduce
        if (isLastS2InnerLoop) {
            vectorService.ProcessTopK(runInfo); // Vector 执行 TopK
        }
    }
}
```

### 5.4 Cube 服务（arch35/lightning_indexer_service_cube.h）

**核心常量**：
| 常量 | 值 | 说明 |
|------|-----|------|
| KEY_BUF_NUM | 3 | Key L1 Ping-Pong 缓冲数 |
| QUERY_BUF_NUM | 2 | Query L1 Ping-Pong 缓冲数 |
| L0_BUF_NUM | 2 | L0A/L0B/L0C Ping-Pong 缓冲数 |
| D_BASIC_BLOCK | 128 | Head dim 固定值 |
| S2_BASIC_BLOCK | 128 | S2 基本块大小（arch35） |

**数据流**：
```
GM(Query/Key)  --Nd2Nz-->  L1 (A1/B1)  --LoadData-->  L0A/L0B  --Mmad-->  L0C  --Fixpipe-->  UB
```

**关键方法**：

```cpp
// 主入口：执行完整的 QK^T 计算
void ComputeMm1(const RunInfo &runInfo) {
    // 1. 等待 Vector 核消费完上一轮 UB 数据
    CrossCoreWaitFlag<CROSS_VC_EVENT>;

    // 2. 循环处理 S2 维度
    for (s2GmOffset = 0; s2GmOffset < s2ProcessSize; s2GmOffset += S2_BASIC_BLOCK) {
        // 2.1 Nd2Nz 加载 Key 到 L1
        KeyNd2Nz(...) / KeyNd2NzForPA(...);

        // 2.2 循环处理 gS1 维度
        for (s1gGmOffset = 0; s1gGmOffset < s1gProcessSize; s1gGmOffset += mBaseSize) {
            // 首次 S2 循环时加载 Query
            if (isFirstS2InnerLoop && s2GmOffset == 0) QueryNd2Nz(...);

            // 2.3 分层加载到 L0 并计算
            for (s2L1Offset) {
                for (s1gL1Offset) {
                    LoadQueryToL0a(...);
                    LoadKeyToL0b(...);
                    ComputeL0c(...);  // Mmad
                    Fixp(...);        // Fixpipe 到 UB
                }
            }
        }
    }

    // 3. 通知 Vector 核 Cube 结果已就绪
    CrossCoreSetFlag<CROSS_CV_EVENT>;
}
```

**Fixpipe 双目标模式**：
```cpp
FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;
fixpipeParams.dualDstCtl = 1;  // 按 M 维拆成两半，分别写入 AIV0/AIV1 的 UB bank
Fixpipe<float, float, QLI_CFG_ROW_MAJOR_UB>(mm1ResUB_, cL0_, fixpipeParams);
```

### 5.5 Vector 服务（arch35/lightning_indexer_service_vector.h）

#### 5.5.1 ProcessVec1 流程

```cpp
void ProcessVec1(const RunInfo &info) {
    // 1. 计算当前 AIV 负责的 S1 范围（AIV0:上半，AIV1:下半）
    curAivS1Idx = curS1Idx + (blockId_ % 2) × CeilDiv(curS1ProcNum, 2);
    curAivS1ProcNum = ...;

    // 2. DataCopyPad：加载 weights 到 UB
    DataCopyPad(weightUB_, weightsGm[weightGmOffset], ...);

    // 3. 等待 Cube 核同步（CV 同步）
    CrossCoreWaitFlag<CROSS_CV_EVENT>;

    // 4. 核心计算：BatchMulWeightAndReduceSum
    vector1::BatchMulWeightAndReduceSum(
        outBase, qkBase, weightBase, weightFloatBase,
        gSize_, curAivS1ProcNum);

    // 5. 通知 Cube 核 Vector 已消费完
    CrossCoreSetFlag<CROSS_VC_EVENT>;

    // 6. DataCopyPad：ReduceSum 结果写回 scoreGm（workspace）
    DataCopyPad(scoreGm[vec1OutGmOffset], outBase, ...);
}
```

#### 5.5.2 ProcessTopK 流程

```cpp
void ProcessTopK(const RunInfo &info) {
    // 处理 Causal Mask：有效 S2 长度逐行递增
    if (attenMaskFlag) validS2Len = actS2Size - actS1Size + curAivS1Idx + 1;

    for (i = 0; i < curAivS1ProcNum; i++) {
        // 1. 读取 scoreGm 中该行的完整 S2 分数
        DataCopyPad(mrgValueLocal_, scoreGm[offset], ...);

        // 2. 直方图 TopK（支持 trunkLen=16K 分块）
        if (validS2Len >= topkCount) {
            topkOp_(mrgValueLocal_, indicesOutLocal_, scoreOutLocal_, validS2Len, loopIdx, s2LoopNum);
        } else {
            // 直接生成索引 0~validS2Len-1，其余补 -1
            CreateVecIndex(indicesOutLocal_, 0, validS2Len);
        }

        // 3. DataCopy：索引写回 sparse_indices
        DataCopyPad(indiceOutGm[indiceOutOffset], indicesOutLocal_, ...);
    }
}
```

### 5.6 微核函数（vf/ 目录）

#### 5.6.1 BatchMulWeightAndReduceSum（vf/lightning_indexer_vector1.h）

**核心功能**：使用 `AscendC::MicroAPI` 手写 SIMD，执行 $ReLU(QK^T) \times W$ 并按 GQA group 求和。

**伪代码**：
```cpp
__simd_callee__ inline void BatchMulWeightAndReduceSum(outUB, qkUB, weightUB, ...) {
    // 初始化常量上下文
    InitFloatSortConstCtx(ctx, maskAll);

    // 主循环：按 gSize 迭代（展开 2）
    for (i = 0; i < gSize; i += 2) {
        // 广播 weight[i], weight[i+1]
        BroadcastLane(regW0, weightFloatUB, i);
        BroadcastLane(regW1, weightFloatUB, i + 1);

        // 加载 qk[i], qk[i+1]
        LoadVecToReg(regQK0, qkUB + i * VL);
        LoadVecToReg(regQK1, qkUB + (i+1) * VL);

        // ReLU + MulAdd
        WeightedAccum<2>(accum, {regQK0, regQK1}, {regW0, regW1}, maskAll);
    }

    // 求和：accum0 + accum1
    Add(regSum, accum[0], accum[1], maskAll);

    // Cast：float → bfloat16
    Cast(regBF16, regSum, maskAll);

    // 转换为 sortable key（uint16）
    FloatToSortableKey(outKey, regBF16, ctx, maskAll);

    // 写回 UB
    StoreRegToVec(outUB, outKey);
}
```

#### 5.6.2 直方图 TopK（vf/lightning_indexer_topk.h）

**实现机制**：
- 基于直方图排序，要求输入长度对齐到 256
- 支持 `trunkLen=16384` 分块归并
- 输入为 `uint16_t` sortable key，输出为索引

**关键特性**：
- `LITopk<uint16_t>` 模板特化
- 使用 `Duplicate` + `mask` 对未对齐尾部补零
- 多轮归并：每轮处理一个 trunk，保留当前 topk，与下一 trunk 归并

---

## 6. 运行时参数结构体

### 6.1 RunInfo

```cpp
struct RunInfo {
    uint32_t loop;              // 当前循环索引
    uint32_t bN2Idx;            // Batch+N2 组合索引
    uint32_t bIdx, n2Idx;       // 拆分后的 batch 和 N2 索引
    uint32_t gS1Idx, s2Idx;     // gS1 和 S2 方向的块索引
    uint32_t actS1Size;         // 当前 batch 的实际 S1 长度
    uint32_t actS2Size;         // 当前 batch 的实际 S2 长度
    uint32_t actMBaseSize;      // 当前块实际的 mBaseSize
    uint32_t actualSingleProcessSInnerSize;  // 当前 S2 块实际大小

    bool isValid;               // 当前块是否有效
    bool isFirstS2InnerLoop;    // 是否是 S2 方向首循环
    bool isLastS2InnerLoop;     // 是否是 S2 方向尾循环（触发 TopK）
    bool isAllLoopEnd;          // 是否所有循环结束

    // GM 偏移
    uint64_t tensorQueryOffset, tensorKeyOffset;
    uint64_t tensorWeightsOffset;
    uint64_t indiceOutOffset;
};
```

### 6.2 ConstInfo

```cpp
struct ConstInfo {
    // 维度常量
    uint32_t batchSize, qSeqSize, kSeqSize;
    uint32_t qHeadNum, kHeadNum, headDim;
    uint32_t gSize;             // GQA group size（64）
    uint32_t sparseCount;       // Top-K 值

    // 切分参数
    uint32_t mBaseSize;         // 256
    uint32_t s1BaseSize;        // 4
    uint32_t s2BaseSize;        // 128
    uint32_t usedCoreNum;       // 24

    // 功能开关
    bool attenMaskFlag;         // 是否启用 causal mask
    bool isAccumSeqS1, isAccumSeqS2;  // TND 布局标记

    // PageAttention
    uint32_t kCacheBlockSize;
    uint32_t maxBlockNumPerBatch;

    // 同步事件
    static constexpr uint32_t CROSS_CV_EVENT = 6;  // Cube→Vector
    static constexpr uint32_t CROSS_VC_EVENT = 0;  // Vector→Cube
    static constexpr uint32_t AIV0_AIV1_OFFSET = 16;
    static constexpr uint32_t QLI_SYNC_MODE4 = 4;  // 跨核同步模式

    // 特殊常量
    int32_t INVALID_IDX = -1;   // 无效索引填充值
};
```

### 6.3 SplitCoreInfo

```cpp
struct SplitCoreInfo {
    bool isCoreEnable;          // 当前核是否启用
    bool isLD;                  // 是否 LD 核（Local Decode，arch35 未使用）
    uint32_t bN2Start, bN2End;  // Batch+N2 方向范围
    uint32_t gS1Start, gS1End;  // gS1 方向范围
    uint32_t s2Start, s2End;    // S2 方向范围
};
```

---

## 7. 端到端数据流

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              Host (CPU)                                  │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  1. OpDef 校验输入数据类型、属性合法性                              │   │
│  │  2. InferShape 推导输出 shape                                       │   │
│  │  3. Tiling 计算：切分参数、workspace 大小、核数分配                   │   │
│  │  4. 下发 TilingData 到 NPU                                          │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         NPU Kernel (Ascend 950)                         │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  Stage 0: Init                                                    │   │
│  │  - 解析 TilingData → ConstInfo                                   │   │
│  │  - SplitCore：确定当前核负责的 (B,N2,gS1,S2) 范围                  │   │
│  │  - InitBuffers：分配 L1/L0/UB Buffer                             │   │
│  │  - 绑定 Global Tensor（Query/Key/Weights/scoreGm/indiceOutGm）    │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                    │                                     │
│                                    ▼                                     │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  Stage 1: 三层循环执行（ProcessMain）                              │   │
│  │                                                                   │   │
│  │  AIC (Cube) 侧：                                                  │   │
│  │  for bN2 in range:                                                │   │
│  │    for gS1 in range:                                              │   │
│  │      for s2 in range:                                             │   │
│  │        ComputeMm1():                                              │   │
│  │          - Nd2Nz 加载 Query/Key 到 L1                             │   │
│  │          - LoadData 到 L0A/L0B                                    │   │
│  │          - Mmad 计算 QK^T → L0C                                   │   │
│  │          - Fixpipe (dualDstCtl=1) → UB（分发给 AIV0/AIV1）         │   │
│  │          - CrossCoreSetFlag 通知 Vector                           │   │
│  │                                                                   │   │
│  │  AIV (Vector) 侧：                                                │   │
│  │  for bN2 in range:                                                │   │
│  │    for gS1 in range:                                              │   │
│  │      for s2 in range:                                             │   │
│  │        ProcessVec1():                                             │   │
│  │          - DataCopyPad 加载 weights → UB                          │   │
│  │          - CrossCoreWaitFlag 等待 Cube 结果                        │   │
│  │          - BatchMulWeightAndReduceSum:                             │   │
│  │            · ReLU(QK^T) × weight → sum                            │   │
│  │            · Cast float → bf16 → sortable key (uint16)            │   │
│  │          - DataCopyPad 写回 scoreGm（workspace）                  │   │
│  │          - CrossCoreSetFlag 通知 Cube                             │   │
│  │        if isLastS2InnerLoop:                                      │   │
│  │          ProcessTopK():                                           │   │
│  │            - DataCopyPad 从 scoreGm 读取完整 S2 分数               │   │
│  │            - 直方图 TopK（支持 trunkLen=16K 分块）                  │   │
│  │            - DataCopyPad 索引写回 sparse_indices                  │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                    │                                     │
│                                    ▼                                     │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  Stage 2: 输出                                                    │   │
│  │  - sparse_indices：每个位置 Top-K 索引                             │   │
│  │  - sparse_values：未实现（arch35）                                 │   │
│  └──────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 8. 关键设计总结

| 设计要点 | 实现方式 |
|---------|---------|
| **Cube-Vector 数据通路** | Cube 结果通过 `Fixpipe` + `dualDstCtl=1` 直接写 UB，按 M 维拆成两半分别给 AIV0/AIV1，省去 GM 中转 |
| **跨核同步** | 使用 `CrossCoreSetFlag/WaitFlag` + `QLI_SYNC_MODE4`，通过 UB 银行级事件 ID（`CROSS_CV_EVENT`/`CROSS_VC_EVENT`）实现 Cube/Vector 同步 |
| **ReduceSum 实现** | `MicroAPI` 手写 SIMD：`BroadcastLane` 广播 weight → `Relu` → `MulAddDst` 累加 → `DeInterleave` 整理 → `Cast` → `FloatToSortableKey` |
| **TopK 算法** | 直方图排序（`LITopk<uint16_t>`），输入为 sortable key（uint16），避免浮点比较，支持 16K trunk 分块归并 |
| **Causal Mask 处理** | Host 侧通过 `GetS2BaseBlockNumOnMask` 计算有效 S2 块数；Kernel 侧在 `ProcessTopK` 中逐行计算 `validS2Len` |
| **PageAttention 支持** | `KeyNd2NzForPA` 通过 `blockTable` 做离散 block 映射，支持非连续的 KV Cache 布局 |
| **多核负载均衡** | `SplitCore` 按 `(B,N2,gS1,S2)` 基本块均匀分配，考虑 causal mask 下有效块数差异 |
| **Ping-Pong 双缓冲** | Query/Key L1（2/3 buf）、L0A/L0B/L0C（2 buf）、UB（2 buf，loop%2）实现计算与搬运重叠 |
| **精度处理** | ReduceSum 全程使用 float；最终结果转 bfloat16 并生成 sortable key；TopK 只比较 key 不保留 value |

