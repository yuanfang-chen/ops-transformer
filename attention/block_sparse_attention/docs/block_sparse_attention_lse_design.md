# BlockSparseAttention 算子 LSE 输出设计文档

## 1. 概述

本设计文档详细描述了 BlockSparseAttention 算子中 LSE (Log Sum Exp) 输出的实现方案。LSE 是 Softmax 计算中的一个重要中间结果，在某些场景下（如模型量化、模型压缩等）需要单独输出以提高计算效率和精度。

## 2. LSE 输出功能说明

### 2.1 功能描述

BlockSparseAttention 算子支持在计算 Softmax 时同时输出 LSE 值，具体包括：

- **LSE 计算**：在 Softmax 计算过程中同步计算 Log Sum Exp 值
- **独立输出**：将 LSE 值作为独立的输出张量返回
- **多精度支持**：支持 FP16 和 BF16 输入精度的 LSE 计算
- **稀疏模式适配**：在块稀疏注意力计算中正确处理 LSE 计算

### 2.2 接口参数

LSE 输出相关的接口参数如下：

| 参数名 | 类型 | 描述 |
|-------|------|------|
| `softmaxLseFlag` | int64_t | 是否输出 LSE，1 表示输出，0 表示不输出 |
| `softmaxLse` | aclTensor* | LSE 输出张量（可选） |

## 3. 实现架构

### 3.1 整体架构

LSE 输出功能在 BlockSparseAttention 算子的计算流程中实现，主要涉及以下组件：

1. **kernel 分发**：根据输入参数和精度选择不同的计算路径
2. **在线 Softmax 计算**：在向量核上执行 Softmax 计算并生成 LSE 中间结果
3. **输出重缩放**：将 LSE 结果从中间缓冲区复制到输出张量

### 3.2 核心组件

| 组件 | 职责 | 文件位置 |
|------|------|----------|
| BlockSparseAttentionInfer | 主计算入口，根据模板参数选择计算路径 | op_kernel/block_sparse_attention.cpp |
| BlockEpilogue (OnlineSoftmax) | 执行在线 Softmax 计算，生成 LSE 中间结果 | op_kernel/attn_infra/epilogue/block/block_epilogue_online_softmax.hpp |
| BlockEpilogue (RescaleO) | 将 LSE 中间结果处理并输出到 gLse 张量 | op_kernel/attn_infra/epilogue/block/block_epilogue_rescale_o.hpp |

## 4. LSE 计算流程

### 4.1 计算步骤

LSE 计算在在线 Softmax 过程中完成，具体步骤如下：

1. **行最大值计算**：计算每个查询向量的注意力分数的最大值
   - 实现函数：`RowmaxSPECTILE512`、`RowmaxSPECTILE256`、`RowmaxTAILTILE`
   - 存储位置：`gmUbTensor`（全局最大值缓冲区）

2. **指数计算**：将注意力分数减去最大值后取指数
   - 实现函数：`CalcExp`
   - 计算公式：`exp(score - max_score)`

3. **行和计算**：计算每个查询向量的指数和
   - 实现函数：`RowsumSPECTILE512`、`RowsumSPECTILE256`、`RowsumTAILTILE`
   - 存储位置：`glUbTensor`（全局和缓冲区）

4. **LSE 生成**：在输出阶段，将行和取自然对数并加上最大值
   - 实现位置：`BlockEpilogue<EpilogueAtlasA2RescaleO>::SubCoreCompute`
   - 计算公式：`ln(row_sum) + max_score`

### 4.2 数据流

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ 注意力分数S │ ──> │ 在线Softmax │ ──> │  LSE计算    │
└─────────────┘     └─────────────┘     └─────────────┘
                           │                   │
                           ▼                   ▼
                    ┌─────────────┐     ┌─────────────┐
                    │  Softmax P  │     │  LSE输出    │
                    └─────────────┘     └─────────────┘
```

## 5. 实现细节

### 5.1 模板参数配置

LSE 输出通过模板参数 `LseMode` 控制，定义如下：

- `LseMode::NONE`：不输出 LSE
- `LseMode::OUT_ONLY`：仅输出 LSE

在 kernel 分发时，根据 `softmaxLseFlag` 参数选择不同的模板实例：

```cpp
// 无LSE输出
BlockSparse::BlockSparseAttentionInfer<half, float, Epilogue::LseMode::NONE, 0, 0>(...)

// 有LSE输出
BlockSparse::BlockSparseAttentionInfer<half, float, Epilogue::LseMode::OUT_ONLY, 0, 0>(...)
```

### 5.2 内存布局

LSE 输出张量的形状与查询输入张量的形状相关，但不包含最后一个维度（head_dim）：

- **TND 格式**：`[T, N]`，其中 T 是总 token 数，N 是头数
- **BNSD 格式**：`[B, N, S]`，其中 B 是 batch 大小，N 是头数，S 是序列长度

### 5.3 关键代码实现

#### 5.3.1 LSE 计算核心代码

```cpp
// 在 BlockEpilogue<EpilogueAtlasA2RescaleO>::SubCoreCompute 中
if constexpr(LSE_MODE == LseMode::OUT_ONLY) {
    if (isLastRowLoop) {
        // 计算 ln(row_sum)
        AscendC::Ln<float, false>(
            lse32_ubuf_tensor,
            glUbTensor,
            (uint64_t)0,
            CeilDiv(totalRowNum, FLOAT_VECTOR_SIZE),
            AscendC::UnaryRepeatParams(1, 1, 8, 8));
        // 加上最大值得到 LSE
        AscendC::Add<float, false>(
            lse32_ubuf_tensor,
            lse32_ubuf_tensor,
            gmUbTensor,
            (uint64_t)0,
            CeilDiv(totalRowNum, FLOAT_VECTOR_SIZE),
            AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
        // 将结果复制到输出张量
        AscendC::DataCopyPad(
            gLse, tvUbTensor,
            AscendC::DataCopyExtParams(totalRowNum, sizeof(float), 0, (stride - 1) * sizeof(float), 0));
    }
}
```

#### 5.3.2 行最大值计算

```cpp
// 在 BlockEpilogue<EpilogueAtlasA2OnlineSoftmax>::RowmaxSPECTILE512 中
AscendC::BlockReduceMax<float, false>(
    tvUbTensor,
    srcUb,
    numRowsRound * numElemsAligned / FLOAT_VECTOR_SIZE,
    0, 1, 1, 8);
AscendC::BlockReduceMax<float, false>(
    tvUbTensor[REDUCE_UB_SIZE],
    tvUbTensor,
    numRowsRound * numElemsAligned / FLOAT_BLOCK_SIZE / FLOAT_VECTOR_SIZE,
    0, 1, 1, 8);
AscendC::BlockReduceMax<float, false>(
    rowmaxUb,
    tvUbTensor[REDUCE_UB_SIZE],
    numRowsRound * numElemsAligned / FLOAT_VECTOR_SIZE / FLOAT_VECTOR_SIZE,
    0, 1, 1, 8);
```

#### 5.3.3 行和计算

```cpp
// 在 BlockEpilogue<EpilogueAtlasA2OnlineSoftmax>::RowsumSPECTILE512 中
AscendC::BlockReduceSum<float, false>(
    tvUbTensor,
    srcUb,
    numRowsRound * numElemsAligned / FLOAT_VECTOR_SIZE,
    0, 1, 1, 8);
AscendC::BlockReduceSum<float, false>(
    tvUbTensor[REDUCE_UB_SIZE],
    tvUbTensor,
    numRowsRound * numElemsAligned / FLOAT_BLOCK_SIZE / FLOAT_VECTOR_SIZE,
    0, 1, 1, 8);
AscendC::BlockReduceSum<float, false>(
    rowsumUb,
    tvUbTensor[REDUCE_UB_SIZE],
    numRowsRound * numElemsAligned / FLOAT_VECTOR_SIZE / FLOAT_VECTOR_SIZE,
    0, 1, 1, 8);
```

## 6. 性能优化

### 6.1 内存访问优化

- **UB 缓冲区复用**：使用统一的 UB 缓冲区存储中间结果，减少内存访问开销
- **数据局部性**：通过分块计算提高数据局部性，减少缓存 miss
- **向量化计算**：使用 AscendC 向量指令进行并行计算

### 6.2 计算优化

- **流水线并行**：通过 ping-pong 缓冲区实现计算与数据传输的流水线并行
- **跨核同步**：使用硬件事件进行跨核同步，减少等待时间
- **条件编译**：根据输入格式（TND/BNSD）进行编译时优化

## 7. 边界情况处理

### 7.1 序列长度非对齐

当序列长度不能被块大小整除时，LSE 计算会自动处理边界情况：

- 使用 `CeilDiv` 计算实际块数
- 对最后一个不完整块进行特殊处理
- 通过掩码机制确保计算正确性

### 7.2 稀疏模式

在块稀疏注意力模式下，LSE 计算仅针对选中的 KV 块进行：

- 根据 `selectIdx` 和 `selectNumIdx` 确定需要计算的 KV 块
- 只对选中的块进行最大值和和的计算
- 正确处理不同数量的 KV 块情况

## 8. 使用示例

### 8.1 接口调用示例

```cpp
// 准备输入数据
int64_t batch = 1;
int64_t qSeqlen = 256;
int64_t kvSeqlen = 512;
int64_t numHeads = 8;
int64_t numKvHeads = 8;
int64_t headDim = 128;
int64_t blockShapeX = 64;
int64_t blockShapeY = 64;
int64_t softmaxLseFlag = 1;  // 启用 LSE 输出

// 创建 LSE 输出张量
aclTensor *softmaxLse = nullptr;
// ... 创建 softmaxLse 张量 ...

// 调用算子获取 workspace 大小
uint64_t workspaceSize = 0;
aclOpExecutor *executor = nullptr;

aclnnStatus ret = aclnnBlockSparseAttentionGetWorkspaceSize(
    query, key, value, nullptr, nullptr, blockShape, nullptr, nullptr,
    selectIdx, selectNumIdx, "BSH", "TND",
    numKvHeads, 0, scaleValue, 0, 0,
    softmaxLseFlag, attentionOut, softmaxLse, &workspaceSize, &executor);

// 分配 workspace 并执行
if (ret == ACLNN_SUCCESS) {
    void *workspace = nullptr;
    aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    
    ret = aclnnBlockSparseAttention(workspace, workspaceSize, executor, stream);
    
    // 此时 softmaxLse 中包含 LSE 输出
    
    aclrtFree(workspace);
}
```

### 8.2 输出格式说明

对于输入形状为 `[T, N, D]` 的 TND 格式：
- 注意力输出形状：`[T, N, D]`
- LSE 输出形状：`[T, N]`

对于输入形状为 `[B, N, S, D]` 的 BNSD 格式：
- 注意力输出形状：`[B, N, S, D]`
- LSE 输出形状：`[B, N, S]`

## 9. 代码优化建议

1. **内存使用优化**：
   - 考虑在不需要 LSE 输出时完全禁用 LSE 相关计算，进一步提高性能
   - 优化 UB 缓冲区分配，减少内存碎片

2. **计算优化**：
   - 探索使用更高效的向量化指令进行 LSE 计算
   - 考虑在某些场景下使用近似计算方法加速 LSE 计算

3. **接口优化**：
   - 提供更灵活的 LSE 输出选项，如支持不同精度的 LSE 输出
   - 考虑添加 LSE 输出的形状检查和验证

## 10. 总结

BlockSparseAttention 算子的 LSE 输出实现通过以下特点确保高效准确：

- **集成式设计**：在 Softmax 计算过程中同步计算 LSE，避免额外的计算开销
- **硬件优化**：充分利用 AscendC 指令和硬件特性，实现高效计算
- **灵活性**：支持不同输入格式和精度的 LSE 计算
- **正确性**：正确处理边界情况和稀疏模式

该实现为需要 LSE 输出的场景提供了高效的解决方案，同时保持了 BlockSparseAttention 算子的整体性能优势。