# Flash Attention Score 空 Tensor 处理逻辑分析

## 概述

当输入 tensor（query/key/value）为空，但输出 tensor（attentionOut/softmaxMax/softmaxSum）不为空时，**直接将输出初始化为 0**，避免无效的 attention 计算。

---

## 类结构

### Arch35 版本

文件：``/Users/kane/projects/issue/transformer-issue/attention/flash_attention_score/op_kernel/arch35/flash_attention_score_empty_tensor_regbase.h`

```cpp
template <typename INPUT_T>
class FlashAttentionScoreEmptyTensorRegbase {
public:
    Init(__gm__ uint8_t *softmaxMax, __gm__ uint8_t *softmaxSum,
         __gm__ uint8_t *attentionOut,
         const FlashAttentionScoreEmptyInputTilingDataRegbase *__restrict tiling);
    Process();

protected:
    ComputeEachCore();
};
```

### Arch32 版本

文件：`/Users/kane/projects/issue/transformer-issue/attention/flash_attention_score/op_kernel/arch32/flash_attention_score_empty_tensor.h`

```cpp
template <typename INPUT_T>
class FlashAttentionScoreEmptyTensor {
public:
    Init(__gm__ uint8_t *softmaxMax, __gm__ uint8_t *softmaxSum,
         __gm__ uint8_t *attentionOut,
         const FlashAttentionScoreTilingData *__restrict tiling);
    Process();

protected:
    ComputeEachCore();
};
```

---

## ComputeEachCore 核心逻辑

### 1. 读取 Tiling 参数

```cpp
uint32_t coreNum = tilingData->coreNum;
uint32_t attentionOutFormerNum = tilingData->attentionOutFormerNum;
uint32_t attentionOutTailNum = tilingData->attentionOutTailNum;
uint32_t softmaxMaxFormerNum = tilingData->softmaxMaxFormerNum;
uint32_t softmaxMaxTailNum = tilingData->softmaxMaxTailNum;
uint64_t attentionOutSingleCoreDataSize = tilingData->attentionOutSingleCoreDataSize;
uint64_t attentionOutTailCoreDataSize = tilingData->attentionOutTailCoreDataSize;
uint64_t softmaxMaxSingleCoreDataSize = tilingData->softmaxMaxSingleCoreDataSize;
uint64_t softmaxMaxTailCoreDataSize = tilingData->softmaxMaxTailCoreDataSize;
uint64_t attentionOutLastCoreDataSize = tilingData->attentionOutLastCoreDataSize;
uint64_t attentionOutLastCoreIndex = tilingData->attentionOutLastCoreIndex;
```

**参数含义**：

| 参数 | 含义 |
|------|------|
| `coreNum` | 总核数 |
| `attentionOutFormerNum` | attentionOut 的主核数 |
| `attentionOutTailNum` | attentionOut 的尾核数 |
| `softmaxMaxFormerNum` | softmaxMax/Sum 的主核数 |
| `softmaxMaxTailNum` | softmaxMax/Sum 的尾核数 |
| `attentionOutSingleCoreDataSize` | 主核处理的数据量 |
| `attentionOutTailCoreDataSize` | 尾核处理的数据量 |
| `attentionOutLastCoreDataSize` | 最后一个核的实际数据量 |
| `attentionOutLastCoreIndex` | 最后一个核的起始地址 |

---

### 2. 初始化 attentionOut

```cpp
// 场景1: 主核数等于总核数，或主核+尾核数小于总核数
if (attentionOutFormerNum == coreNum || (attentionOutFormerNum + attentionOutTailNum) < coreNum) {
    if (tmpBlockIdx < attentionOutFormerNum - 1) {
        // 前 (formerNum-1) 个主核
        InitOutput<INPUT_T>(attentionOutGm[tmpBlockIdx * attentionOutSingleCoreDataSize],
                             attentionOutSingleCoreDataSize, 0.0);
    } else if (tmpBlockIdx == attentionOutFormerNum - 1) {
        // 最后一个主核（处理非对齐数据）
        InitOutput<INPUT_T>(attentionOutGm[attentionOutLastCoreIndex],
                             attentionOutLastCoreDataSize, 0.0);
    }
}
// 场景2: 主核+尾核数等于总核数
else {
    if (tmpBlockIdx < attentionOutFormerNum) {
        // 主核
        InitOutput<INPUT_T>(attentionOutGm[tmpBlockIdx * attentionOutSingleCoreDataSize],
                             attentionOutSingleCoreDataSize, 0.0);
    } else if (tmpBlockIdx >= attentionOutFormerNum && tmpBlockIdx < coreNum - 1) {
        // 尾核（除最后一个外）
        InitOutput<INPUT_T>(
            attentionOutGm[attentionOutFormerNum * attentionOutSingleCoreDataSize +
                           (tmpBlockIdx - attentionOutFormerNum) * attentionOutTailCoreDataSize],
            attentionOutTailCoreDataSize, 0.0);
    } else if (tmpBlockIdx == coreNum - 1) {
        // 最后一个尾核（处理非对齐数据）
        InitOutput<INPUT_T>(attentionOutGm[attentionOutLastCoreIndex],
                             attentionOutLastCoreDataSize, 0.0);
    }
}
```

**多核切分策略**：

| 场景 | 条件 | 主核数 | 尾核数 | 数据分布 |
|------|-------|---------|---------|----------|
| 场景1 | `formerNum == coreNum` | coreNum | 0 | 所有核都是主核 |
| 场景2 | `formerNum + tailNum < coreNum` | formerNum | 0 | 部分核不工作 |
| 场景3 | `formerNum + tailNum == coreNum` | formerNum | tailNum | 主核+尾核 |

---

### 3. 初始化 softmaxMax 和 softmaxSum

```cpp
// 如果当前核不需要处理 softmaxMax/Sum，直接返回
if (tmpBlockIdx >= (softmaxMaxFormerNum + softmaxMaxTailNum)) {
    return;
}
// 主核
else if (tmpBlockIdx < softmaxMaxFormerNum) {
    InitOutput<float>(softmaxMaxGm[tmpBlockIdx * softmaxMaxSingleCoreDataSize],
                       tilingData->softmaxMaxSingleCoreDataSize, 0.0);
    InitOutput<float>(softmaxSumGm[tmpBlockIdx * softmaxMaxSingleCoreDataSize],
                       tilingData->softmaxMaxSingleCoreDataSize, 0.0);
}
// 尾核
else {
    InitOutput<float>(softmaxMaxGm[softmaxMaxFormerNum * softmaxMaxSingleCoreDataSize +
                                   (tmpBlockIdx - softmaxMaxFormerNum) * softmaxMaxTailCoreDataSize],
                       tilingData->softmaxMaxTailCoreDataSize, 0.0);
    InitOutput<float>(softmaxSumGm[softmaxMaxFormerNum * softmaxMaxSingleCoreDataSize +
                                   (tmpBlockIdx - softmaxMaxFormerNum) * softmaxMaxTailCoreDataSize],
                       tilingData->softmaxMaxTailCoreDataSize, 0.0);
}
```

---

## 多核切分示例

### 示例1: 数据量正好被核数整除

```
假设：
- coreNum = 4
- attentionOutSize = 128
- attentionOutSingleCoreDataSize = 32

切分结果：
- attentionOutFormerNum = 4
- attentionOutTailNum = 0

数据分布：
Core 0: [0, 32)
Core 1: [32, 64)
Core 2: [64, 96)
Core 3: [96, 128)
```

### 示例2: 数据量不能被核数整除

```
假设：
- coreNum = 4
- attentionOutSize = 130
- attentionOutSingleCoreDataSize = 33 (主核)
- attentionOutTailCoreDataSize = 32 (尾核)
- attentionOutLastCoreDataSize = 31 (最后一个核)

切分结果：
- attentionOutFormerNum = 2 (blocks % coreNum = 2)
- attentionOutTailNum = 2 (coreNum - formerNum = 2)

数据分布：
Core 0 (主核): [0, 33)
Core 1 (主核): [33, 66)
Core 2 (尾核): [66, 98)
Core 3 (尾核): [98, 130) (实际处理 31)
```

### 示例3: 数据量小于核数

```
假设：
- coreNum = 4
- attentionOutSize = 65
- attentionOutSingleCoreDataSize = 33
- attentionOutLastCoreDataSize = 32

切分结果：
- attentionOutFormerNum = 2 (blocks = 3)
- attentionOutTailNum = 0

数据分布：
Core 0: [0, 33)
Core 1: [33, 65) (实际处理 32)
Core 2: 不工作
Core 3: 不工作
```

---

## Arch35 vs Arch32 的差异

| 特性 | Arch35 | Arch32 |
|------|--------|--------|
| Tiling 数据类型 | `FlashAttentionScoreEmptyInputTilingDataRegbase` | `FlashAttentionScoreTilingData` |
| 访问方式 | `tilingData->attentionOutFormerNum` | `tilingData->emptyInputTilingData.attentionOutFormerNum` |
| `softmaxMaxSingleCoreDataSize` 访问 | `tilingData->softmaxMaxSingleCoreDataSize` | `tilingData->emptyInputTilingData.softmaxMaxSingleCoreDataSize` |

---

## 触发条件

在 Tiling 阶段检测（`flash_attention_score_tiling.cpp`）：

```cpp
// 当 query/key/value 为空，但输出不为空时触发
if ((queryShapeSize == 0 || keyShapeSize == 0 || valueShapeSize == 0) &&
    (attentionOutShapeSize != 0 || softmaxSumShapeSize != 0)) {
    // 进入 empty tensor 处理流程
    SetTilingKey(FA_EMPTY_TILING_KEY);
}
```

---

## 切分算法详解

### 以 MIN_COPY_UINT_SIZE = 32Byte 为例

假设 `blocks` 为数据的块数，`blocks` 与 `coreNum` 存在三种关系：

#### (1) blocks % coreNum == 0

```
主核数量为 coreNum
主核处理块数为 blocks / coreNum
最后一个核处理非 32Byte 对齐的数据
尾核数量为 0

示例：blocks = 8, coreNum = 4
主核数 = 4, 尾核数 = 0
每个主核处理 2 个块
```

#### (2) blocks % coreNum != 0

##### (2.1) blocks < coreNum

```
主核数量为 blocks
主核处理块数为 1
最后一个核处理非 32Byte 对齐的数据
尾核数量为 0

示例：blocks = 3, coreNum = 4
主核数 = 3, 尾核数 = 0
每个主核处理 1 个块
Core 3 不工作
```

##### (2.2) blocks > coreNum

```
主核数量为 blocks % coreNum
尾核数量为 coreNum - (blocks % coreNum)
尾核处理块数为 blocks / coreNum
主核处理块数为 blocks / coreNum + 1
最后一个尾核处理非对齐场景

示例：blocks = 10, coreNum = 4
主核数 = 2 (10 % 4 = 2)
尾核数 = 2 (4 - 2 = 2)
主核处理 3 个块 (10 / 4 + 1 = 3)
尾核处理 2 个块 (10 / 4 = 2)

数据分布：
|-------------主核块-----------------|------------尾核块-----------|非对齐块|
|                                   |                             |       |
|--------2*(blocks/coreNum+1)-------|-----2*(blocks/coreNum)------|<32Byte|
```

---

## 总结

空 tensor 逻辑的核心思想：

1. **避免无效计算**：当输入为空时，直接将输出初始化为 0
2. **多核并行**：使用主核+尾核的策略，将输出数据均匀分配到多个核
3. **非对齐处理**：最后一个核处理非 32 字节对齐的数据
4. **灵活切分**：支持数据量大于、等于、小于核数的场景

这种设计在处理变长序列或稀疏场景时非常有用，可以显著提高效率。

---

## 相关文件

- Arch35: `/Users/kane/projects/issue/transformer-issue/attention/flash_attention_score/op_kernel/arch35/flash_attention_score_empty_tensor_regbase.h`
- Arch32: `/Users/kane/projects/issue/transformer-issue/attention/flash_attention_score/op_kernel/arch32/flash_attention_score_empty_tensor.h`
- Tiling: `/Users/kane/projects/issue/transformer-issue/attention/flash_attention_score/op_host/flash_attention_score_tiling.cpp`
