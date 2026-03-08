# BlockSparseAttention算子支持LSE输出 详细设计文档

## 1. 需求概述

### 1.1 背景

BlockSparseAttention算子基于FlashAttention的Online Softmax机制实现稀疏注意力计算。在Transformer模型推理/训练场景中，Softmax的Log-Sum-Exp（LSE）中间结果是反向传播、多卡并行拼接等场景的关键输入。当前算子在前向计算中已经在内部维护了Online Softmax所需的全局最大值（`globalMax`）和全局求和值（`globalSum`），但未将对应的LSE值输出到Global Memory供外部使用。

### 1.2 目标

在BlockSparseAttention算子中新增`softmaxLseFlag`控制开关，当开启时（`softmaxLseFlag=1`），在前向计算完成后，将每个token、每个head对应的LSE值以FLOAT32精度输出到`softmaxLseOptional`张量中。

### 1.3 LSE数学定义

对于标准Scaled Dot-Product Attention：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right) V
$$

其中LSE定义为：

$$
\text{LSE}(x_i) = \ln\left(\sum_{j} e^{x_{ij}}\right)
$$

在Online Softmax的流式计算中，最终的LSE可由全局最大值 $m$ 和全局归一化因子 $l$ 合成：

$$
\text{LSE} = m + \ln(l)
$$

其中 $m = \text{globalMax}$，$l = \text{globalSum}$。

## 2. 涉及模块总览

本次变更涉及以下14个文件，横跨Host侧（API/InferShape/Tiling）和Device侧（Kernel/Epilogue）：

| 模块 | 文件 | 变更说明 |
|------|------|----------|
| API接口文档 | `docs/aclnnBlockSparseAttention.md` | 更新softmaxLseFlag和softmaxLseOptional接口描述 |
| 算子定义 | `op_host/block_sparse_attention_def.cpp` | softmaxLseFlag默认值由1改为0 |
| Shape推导 | `op_host/block_sparse_attention_infershape.cpp` | LSE输出shape推导适配TND/BNSD格式 |
| Tiling策略 | `op_host/block_sparse_attention_tiling.cpp` | 新增ProcessSoftmaxLse、TilingKey编码LSE标志位 |
| Tiling头文件 | `op_host/block_sparse_attention_tiling.h` | 新增softmaxLseFlag\_成员变量 |
| aclnn API | `op_host/op_api/aclnn_block_sparse_attention.cpp` | LSE输出的ViewCopy逻辑 |
| op_api封装 | `op_host/op_api/block_sparse_attention.cpp` | 参数传递格式修正 |
| TilingKey定义 | `op_kernel/block_sparse_attention_tilingkey.h` | 新增6个LSE\_OUT对应的TilingKey宏 |
| Kernel入口 | `op_kernel/block_sparse_attention.cpp` | 新增6个LSE\_OUT分支的模板实例化 |
| Kernel主逻辑 | `op_kernel/block_sparse_attention_kernel.h` | LSE GM偏移计算、LayoutLse传递 |
| Epilogue-RescaleO | `epilogue/block/block_epilogue_rescale_o.hpp` | float精度LSE计算与搬出 |
| Epilogue-RescaleO(LowPrec) | `epilogue/block/block_epilogue_rescale_o_low_prec.hpp` | half精度LSE计算、Cast到float后搬出 |
| Epilogue-OnlineSoftmax | `epilogue/block/block_epilogue_online_softmax.hpp` | LSE模式下硬件event同步 |
| Epilogue-OnlineSoftmax(LowPrec) | `epilogue/block/block_epilogue_online_softmax_low_prec.hpp` | LSE模式下硬件event同步（低精度版） |

## 3. 详细设计

### 3.1 接口层设计

#### 3.1.1 输入参数 `softmaxLseFlag`

| 属性 | 说明 |
|------|------|
| 类型 | Host侧 `int64_t` |
| 语义 | 控制是否输出LSE，1表示输出，0表示不输出 |
| 默认值 | 0（不输出） |
| 取值范围 | {0, 1} |

#### 3.1.2 输出参数 `softmaxLseOptional`

| 属性 | 说明 |
|------|------|
| 类型 | Device侧 `aclTensor` |
| 数据类型 | FLOAT（FP32） |
| format | ND |
| 语义 | 当`softmaxLseFlag=1`时输出Softmax的log-sum-exp中间结果 |

#### 3.1.3 softmaxLseOptional Shape规则

| Q Layout | LSE Shape | 维度说明 |
|----------|-----------|----------|
| TND | `[T, N, 1]` | T=总token数, N=Q Head数, 最后一维固定为1 |
| BNSD | `[B, N, S, 1]` | B=Batch, N=Q Head数, S=Q序列长度, 最后一维固定为1 |

> **设计说明**：LSE的最后一维为1，是因为每个(token, head)对应一个标量LSE值。增加这个维度是为了与框架的Tensor规范保持一致，便于后续ViewCopy和数据对齐。

### 3.2 算子定义层（op_host/def）

**变更**：将`softmaxLseFlag`的默认值从1修改为0。

```cpp
this->Attr("softmaxLseFlag").AttrType(OPTIONAL).Int(0);
```

**设计理由**：默认不输出LSE，避免不需要LSE的场景产生额外的计算和数据搬运开销。仅在显式开启时生效。

### 3.3 Shape推导层（InferShape）

针对softmaxLse输出tensor的shape推导逻辑如下：

**TND格式**：
```cpp
softmaxLseShape->SetDimNum(3);  // TND_DIM_NUM = 3
(*softmaxLseShape)[0] = queryShape->GetDim(0);  // T
(*softmaxLseShape)[1] = queryShape->GetDim(1);  // N
(*softmaxLseShape)[2] = 1;                       // LSE_DIM_D = 1
```

**BNSD格式**：
```cpp
softmaxLseShape->SetDimNum(4);  // BNSD_DIM_NUM = 4
(*softmaxLseShape)[0] = queryShape->GetDim(0);  // B
(*softmaxLseShape)[1] = queryShape->GetDim(1);  // N
(*softmaxLseShape)[2] = queryShape->GetDim(2);  // S
(*softmaxLseShape)[3] = 1;                       // LSE_DIM_D = 1
```

### 3.4 Tiling策略层

#### 3.4.1 属性解析

新增`ProcessSoftmaxLse`方法从算子属性中解析`softmaxLseFlag`：

- 属性索引：`SOFTMAX_LSE_FLAG_INDEX = 9`
- 合法值：0（关闭）、1（开启）
- 非法值直接返回`GRAPH_FAILED`

同时在`CheckAttr`中增加兼容性处理：当属性指针为空时，默认`softmaxLseFlag_ = false`。

#### 3.4.2 TilingKey编码

TilingKey采用64位无符号整数编码，LSE标志位编码在**亿位**（第8-9位）：

```
TilingKey = 9LKKPPSSSMMMQQQ
             ^ LSE标志位（亿位）
```

- `softmaxLseFlag = true`：`tilingKey += 100000000ULL`（亿位置1）
- `softmaxLseFlag = false`：亿位保持0（默认）

#### 3.4.3 新增TilingKey列表

| TilingKey | 数据类型 | Q Layout | KV Layout | Softmax精度 | LSE |
|-----------|----------|----------|-----------|-------------|-----|
| `9000000130000002` | FP16 | TND | TND | Float | OUT |
| `9000000130100002` | FP16 | TND | TND | Half | OUT |
| `9000000130022222` | BF16 | TND | TND | Float | OUT |
| `9000000150000003` | FP16 | BNSD | BNSD | Float | OUT |
| `9000000150100003` | FP16 | BNSD | BNSD | Half | OUT |
| `9000000150022223` | BF16 | BNSD | BNSD | Float | OUT |

### 3.5 aclnn API层

当`softmaxLseFlag == 1`时，在GetWorkspaceSize阶段额外执行ViewCopy，将kernel输出的LSE tensor拷贝到用户提供的`softmaxLseOptional`输出tensor：

```cpp
if (softmaxLseFlag == LSE_OUT) {
    auto viewCopyLseResult = l0op::ViewCopy(outputs[1], softmaxLseOptional, executorImpl);
    CHECK_RET(viewCopyLseResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
}
```

### 3.6 Kernel入口层

#### 3.6.1 模板实例化

Kernel入口函数通过`TILING_KEY_IS`宏增加6个LSE输出分支，每个分支使用`Epilogue::LseMode::OUT_ONLY`模板参数：

```cpp
BlockSparse::BlockSparseAttentionInfer<InputType, SoftmaxType, Epilogue::LseMode::OUT_ONLY, QUERY_LAYOUT, KV_LAYOUT>(
    query, key, value, blockSparseMask, mask, blockTable, attentionOut,
    actualSeqLengths, actualSeqLengthsKv, blockShape, user, softmaxLse, tiling);
```

模板参数`LseMode`的取值：
- `LseMode::NONE`：不输出LSE（原有逻辑）
- `LseMode::OUT_ONLY`：仅输出LSE（本次新增）

#### 3.6.2 硬件Event初始化

新增`EVENT_ID4`的`MTE3_V`事件初始化，用于LSE数据从UB搬出到GM时的同步：

```cpp
AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID4);
```

对应在kernel结束时增加`WaitFlag`确保所有LSE搬出完成：

```cpp
AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID4);
```

### 3.7 Kernel主逻辑层（kernel.h）

#### 3.7.1 LSE GM偏移计算

在batch循环中，为LSE输出计算Global Memory偏移量：

**BNSD格式**：
```cpp
lseBOffset = curBatch * qHeads * maxQSeqlen;
```

**TND格式**：
```cpp
lseBOffset += qSeqlen * qHeads;
```

在每个Q block的处理中，计算当前(head, seq)对应的LSE GM偏移：

**BNSD格式**（LSE排布为 `[B, N, S]`）：
```cpp
gmOffsetLse = lseBOffset + qHeadIdx * maxQSeqlen + qSeqOffset;
```

**TND格式**（LSE排布为 `[T, N]`）：
```cpp
gmOffsetLse = lseBOffset + qSeqOffset * qHeads + qHeadIdx;
```

#### 3.7.2 LayoutLse构造

根据数据格式构造LSE的Layout描述，用于Epilogue阶段的数据搬出：

| Q Layout | LayoutLse(rows, stride) | 说明 |
|----------|------------------------|------|
| BNSD | `LayoutLse(qSeqlen, 1)` | 同一Head内LSE连续存放，stride=1 |
| TND | `LayoutLse(qSeqlen, qHeads)` | 不同Head交织存放，stride=qHeads |

### 3.8 Epilogue层 — LSE计算与搬出

Epilogue阶段是LSE输出的核心计算环节，在RescaleO（输出缩放）的最后阶段完成。分为Float精度版本和Low Precision（Half精度）版本。

#### 3.8.1 Float精度版本（block_epilogue_rescale_o.hpp）

**UB buffer分配**：

```cpp
constexpr uint32_t LSE_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE + 12 * UB_UINT8_VECTOR_SIZE;
lse32_ubuf_tensor = resource.ubBuf.template GetBufferByByte<float>(LSE_UB_TENSOR_OFFSET);
```

> LSE UB buffer与`glUbTensor`共享偏移量，因为两者不会同时活跃使用。

**LSE计算流程**（在`isLastStackTile && isLastRowLoop`时执行）：

```
步骤1: lse = ln(globalSum)    // Ln指令，对glUbTensor取自然对数
步骤2: lse = lse + globalMax  // Add指令，加上全局最大值
步骤3: lse_block = Brcb(lse)  // 将连续数据展开为block对齐格式
步骤4: DataCopyPad → GM       // 按stride搬出到Global Memory
```

对应的AscendC指令序列：

```
Ln(lse32_ubuf_tensor, glUbTensor)        → lse = ln(l)
Add(lse32_ubuf_tensor, lse32_ubuf_tensor, gmUbTensor)  → lse = ln(l) + m
Brcb(tvUbTensor, lse32_ubuf_tensor)      → block对齐
SetFlag<V_MTE3>(EVENT_ID4)               → 同步
WaitFlag<V_MTE3>(EVENT_ID4)
DataCopyPad(gLse, tvUbTensor, ...)       → 搬出到GM
SetFlag<MTE3_V>(EVENT_ID4)               → 标记搬出完成
```

#### 3.8.2 Low Precision版本（block_epilogue_rescale_o_low_prec.hpp）

当Softmax使用Half精度计算时，LSE中间结果也是Half精度，需要额外的Cast操作：

```
步骤1: lse_fp16 = ln(globalSum_fp16)    // Half精度Ln
步骤2: lse_fp16 = lse_fp16 + globalMax_fp16  // Half精度Add
步骤3: lse_fp32 = Cast(lse_fp16)        // Cast half → float
步骤4: lse_block = Brcb(lse_fp32)       // block对齐
步骤5: DataCopyPad → GM                 // 搬出到GM
```

**新增UB buffer**：

```cpp
lse16_ubuf_tensor = resource.ubBuf.GetBufferByByte<half>(LSE16_UB_TENSOR_OFFSET);
lse32_ubuf_tensor = resource.ubBuf.GetBufferByByte<float>(LSE32_UB_TENSOR_OFFSET);
```

> 低精度版本需要两个buffer：`lse16_ubuf_tensor`用于half精度中间计算，`lse32_ubuf_tensor`用于cast后的float精度结果。最终输出始终为FP32精度。

#### 3.8.3 LSE搬出策略（DataCopyPad）

LSE搬出根据`qNThisSubBlock`区分两种情况：

**情况1：qNThisSubBlock == 0**（单Head块，连续token）：
```cpp
DataCopyPad(gLse, tvUbTensor,
    DataCopyExtParams(totalRowNum, sizeof(float), 0, (stride - 1) * sizeof(float), 0));
```
一次性搬出所有token的LSE，使用stride跳过不同head之间的间隔。

**情况2：qNThisSubBlock > 0**（多Head块，需逐Head搬出）：
```cpp
for (uint32_t qNIdx = 0; qNIdx < qNThisSubBlock; qNIdx++) {
    DataCopyPad(gLse[qNIdx], tvUbTensor[qNIdx * qSBlockSize * FLOAT_BLOCK_SIZE],
        DataCopyExtParams(qSBlockSize, sizeof(float), 0, (stride - 1) * sizeof(float), 0));
}
```
按Head遍历，每个Head分别搬出对应qSBlockSize个token的LSE值。

#### 3.8.4 硬件同步设计

LSE搬出使用`EVENT_ID4`进行MTE3（数据搬运引擎）与Vector Core之间的同步：

```
OnlineSoftmax阶段:
  WaitFlag<MTE3_V>(EVENT_ID4)    ← 等待上一次LSE搬出完成，才能重用UB buffer

RescaleO阶段:
  SetFlag<V_MTE3>(EVENT_ID4)     ← Vector计算完成，通知MTE3可以开始搬出
  WaitFlag<V_MTE3>(EVENT_ID4)    ← 等待可搬出信号
  DataCopyPad(gLse, ...)          ← MTE3执行搬出
  SetFlag<MTE3_V>(EVENT_ID4)     ← 搬出完成，通知Vector可以继续
```

这确保了LSE搬出不会与下一轮迭代的Vector计算产生数据竞争。

### 3.9 SubBlock级LSE偏移计算

在RescaleO的`EpilogueRescaleO`入口函数中，根据子块划分计算LSE的GM偏移：

```cpp
uint32_t outLseRowOffsetThisSubBlock = (qNBlockSize == 1U) ? inRowOffsetThisSubBlock : 0;
uint32_t outLseColOffsetThisSubBlock = (qNBlockSize == 1U) ? 0 : subBlockIdx * qNSplitSubBlock;
int64_t offsetLse = layoutLse.GetOffset(MatrixCoord(outLseRowOffsetThisSubBlock, outLseColOffsetThisSubBlock));
```

| 分块策略 | Row偏移 | Col偏移 | 说明 |
|----------|---------|---------|------|
| qNBlockSize == 1 | token偏移 | 0 | 按token方向分块，Head维度不分 |
| qNBlockSize > 1 | 0 | Head偏移 | 按Head方向分块 |

## 4. 数据流总览

```
┌─────────────────────────────────────────────────────────┐
│                    Host 侧                               │
│                                                          │
│  aclnn API  ──→  InferShape  ──→  Tiling                │
│  │ softmaxLseFlag    │ LSE shape     │ TilingKey编码     │
│  │ ViewCopy LSE      │ [T,N,1]       │ 亿位+1            │
│  │                   │ [B,N,S,1]     │ softmaxLseFlag_  │
│  │                   │               │                   │
└──┼───────────────────┼───────────────┼───────────────────┘
   │                   │               │
   ▼                   ▼               ▼
┌─────────────────────────────────────────────────────────┐
│                    Device 侧                             │
│                                                          │
│  Kernel入口 (LseMode::OUT_ONLY模板参数)                   │
│     │                                                    │
│     ├─ gmOffsetLse 计算 (BNSD/TND)                       │
│     ├─ LayoutLse 构造                                    │
│     │                                                    │
│     ▼                                                    │
│  Epilogue: OnlineSoftmax                                 │
│     │ WaitFlag<MTE3_V>(EVENT_ID4) ← 等待上次LSE搬出     │
│     │ 计算 globalMax, globalSum                           │
│     │                                                    │
│     ▼                                                    │
│  Epilogue: RescaleO (isLastStackTile && isLastRowLoop)   │
│     │ LSE = ln(globalSum) + globalMax                    │
│     │ [Low Prec: Cast half→float]                        │
│     │ Brcb → block对齐                                   │
│     │ DataCopyPad → gLse (GM)                            │
│     │ SetFlag<MTE3_V>(EVENT_ID4) → 通知搬出完成          │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

## 5. 支持的配置矩阵

| 数据类型 | Q Layout | KV Layout | Softmax精度 | LSE输出 | 支持状态 |
|----------|----------|-----------|-------------|---------|---------|
| FP16 | TND | TND | Float32 | 是 | ✅ |
| FP16 | TND | TND | Float16 | 是 | ✅ |
| BF16 | TND | TND | Float32 | 是 | ✅ |
| FP16 | BNSD | BNSD | Float32 | 是 | ✅ |
| FP16 | BNSD | BNSD | Float16 | 是 | ✅ |
| BF16 | BNSD | BNSD | Float32 | 是 | ✅ |

> LSE输出的数据类型始终为FLOAT32，无论输入数据类型或Softmax计算精度如何。

## 6. 约束与限制

1. **LSE输出精度**：无论Softmax使用Float或Half精度计算，LSE输出均为FP32，低精度场景通过Cast指令转换。
2. **默认行为**：`softmaxLseFlag`默认值为0（不输出），保持与原有接口的向后兼容。
3. **softmaxLseFlag取值**：仅支持0和1，其他值将返回`GRAPH_FAILED`错误。
4. **UB Buffer复用**：LSE计算复用了Epilogue阶段的临时buffer（与`glUbTensor`/`gmUbTensor`共享偏移），不增加额外的UB内存开销。
5. **执行时机**：LSE的计算和搬出仅在最后一个StackTile的最后一轮RowLoop时执行，确保globalMax和globalSum已经完成Online Softmax的全部累积。

## 7. 关联Commit

| CommitId | 描述 |
|----------|------|
| `340c2f0b` | feat: block sparse attention support lse |
| `e59bb075` | fix（修复softmaxLseFlag默认值及shape推导维度问题） |
