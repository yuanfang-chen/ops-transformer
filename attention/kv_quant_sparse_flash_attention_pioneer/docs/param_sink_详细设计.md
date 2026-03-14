# Param Sink 功能详细设计

## 1. 概述

### 1.1 需求背景

在长上下文推理场景中，Sparse Attention 通过选择性计算关键 KV 块来降低计算量。Param Sink 机制在此基础上引入固定数量的 **Sink Token**（128 个），在注意力计算时始终参与，等效于在 KV 序列开头拼接一段关键上下文。

### 1.2 功能定义

- 在每层注意力计算中，固定引入 128 个 Sink Token 的 KV 数据
- Sink KV 数据为 **BF16 非量化格式**，通过独立的 `key_sink`/`value_sink` 输入传入
- Sink KV **不经过 Vec0 反量化流程**，由 **Cube 侧（AIC）直接搬运**到 L1
- 仅在 **PA_BSND 布局**下支持

### 1.3 设计原则

- **零框架侵入**：框架侧不修改 block_table、sparse_indices 等已有数据结构，仅传入 `key_sink`/`value_sink` 张量
- **最小 Kernel 侵入**：复用现有三级流水架构，在 S2 循环前插入独立 Sink 迭代
- **朴素实现**：优先保证功能正确性，不做极致性能优化
- **格式一致**：Sink KV 写入 L1 后的 NZ 格式与正常反量化 KV 完全一致，Cube 侧 BMM1/BMM2 逻辑无需修改

---

## 2. 外部接口约定

### 2.1 输入数据约定

| 输入 | Shape | 数据类型 | 说明 |
|------|-------|---------|------|
| key_sink | (128, N2, D) | BF16 | D = Dn + Dr = 512 + 64 = 576，nope+rope 合并 |
| value_sink | (128, N2, D) | BF16 | MLA 场景下复用 key_sink（同一份数据） |

### 2.2 框架侧约定

**框架侧无需任何预处理**。`block_table`、`sparse_indices`、`sparseBlockCount`、`actualS2Size` 等参数均保持原样不变。

框架侧唯一的变化是：当启用 Param Sink 时，传入非空的 `key_sink` 和 `value_sink` 张量。

> **设计理由**：Sink KV 由 AIC 直接从 `key_sink` GM 地址搬运到 L1，完全绕过 block_table 寻址机制（`GetkeyOffset` 函数），因此不需要在 block_table 中为 Sink 分配条目。若强行在 block_table 前插入一个 Block 并将 sparse_indices 整体偏移 +128，会因为 `sinkTokenNum(128) ≠ blockSize(256)` 导致 `s2Idx / blockSize` 和 `s2Idx % blockSize` 计算出错误的物理块号和块内偏移，**破坏正常 KV 的寻址**。

### 2.3 Kernel 侧处理策略

Kernel 在 S2 主循环**开头**插入一次额外的 Sink 迭代（`s2LoopCount == 0`），专门处理 Sink KV 数据。后续正常 KV 迭代从 `s2LoopCount == 1` 开始，使用调整后的有效循环计数 `effectiveS2LoopCount = s2LoopCount - 1` 访问 `sparse_indices`，确保正常 KV 的寻址逻辑**完全不变**。

> 当 `hasSink == 0` 时，不插入额外迭代，行为与原始逻辑完全一致。

---

## 3. 整体方案

### 3.1 数据通路对比

```
┌─────────────────────────────────────────────────────────────────────┐
│                    正常 KV 块（s2LoopCount > 0, 或无 Sink）          │
│                                                                     │
│  [AIV] GM(INT8) ──MTE2──→ UB ──VF反量化──→ UB(BF16) ──MTE3──→ L1(NZ)  │
│  [AIC]                    等待核间同步 → 从L1读K → BMM1              │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                    Sink KV 块（s2LoopCount == 0, hasSink == 1）      │
│                                                                     │
│  [AIV] 空操作（仅维持核间同步握手）                                    │
│  [AIC] GM(BF16) ──MTE2──→ L1(NZ) → 从L1读K → BMM1                  │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 三级流水影响

Sink 作为 S2 循环的**第 0 次迭代**插入，后续正常 KV 从第 1 次迭代开始。三级流水架构完全不变：

```
s2LoopCount:    0(Sink)         1(KV[0])        2(KV[1])        3(KV[2])
AIC:    SinkCopy+BMM1[0]   BMM1[1]         BMM1[2]         BMM1[3]
                            BMM2[0]         BMM2[1]         BMM2[2]
AIV:    Sync Only[0]       Vec0[1]         Vec0[2]         Vec0[3]
                            Vec1[0]         Vec1[1]         Vec1[2]
                                            Vec2[0]         Vec2[1]
```

- **s2LoopCount == 0（Sink）**：AIC 直接搬运 Sink KV 到 L1 + BMM1；AIV 仅做同步握手
- **s2LoopCount >= 1（正常 KV）**：恢复正常流水，AIV 使用 `effectiveS2LoopCount = s2LoopCount - 1` 索引 `sparse_indices`

### 3.3 Softmax 状态管理

Sink 迭代作为 S2 循环的第 0 次迭代，自然走 **NoUpdate 路径**（首次迭代，初始化 softmax 统计量 m/l）。后续正常 KV 迭代（s2LoopCount >= 1）走 **Update 路径**（合并历史统计量）。

由于 Online Softmax 是逐块迭代累计的，不依赖总 S2 长度，因此 `actualS2Size` 不需要包含 Sink Token。无需额外处理。

---

## 4. Host 侧变更

### 4.1 OpDef（*_def.cpp）

**无需修改**。`key_sink`（输入 9）和 `value_sink`（输入 10）已作为可选输入定义，数据类型为 BF16。

### 4.2 TilingData 变更

#### 4.2.1 BaseParams 新增字段

```cpp
BEGIN_TILING_DATA_DEF(KvQuantSparseFlashAttentionPioneerBaseParamsMla)
    // ... 现有字段 ...
    uint32_t hasSink;           // [新增] 是否启用 Param Sink（0/1）
    uint32_t sinkTokenNum;      // [新增] Sink Token 数量（当前固定 128）
END_TILING_DATA_DEF
```

#### 4.2.2 CVSharedParams 新增位域

```cpp
struct CVSharedParams {
    // ... 现有字段 ...
    uint32_t hasSink : 1;       // [新增] 是否启用 Sink
    uint32_t sinkTokenNum : 8;  // [新增] Sink Token 数量（128，用 8 bit 表示）
};
```

### 4.3 Tiling 策略变更

#### 4.3.1 Sink 启用判定

在 `QSFAPInfoParser::Parse()` 中新增：

```cpp
// 当 key_sink 输入不为空且 KV 布局为 PA_BSND 时，启用 Sink
bool hasSink = (opParamInfo.keySink.tensor != nullptr) && (kvLayout == QSFALayout::PA_BSND);
sfaaInfo->hasSink = hasSink;
sfaaInfo->sinkTokenNum = hasSink ? 128 : 0;
```

#### 4.3.2 参数校验（QSFAPTilingCheck）

新增校验规则：

```cpp
// Sink 仅在 PA_BSND 下支持
if (hasSink && kvLayout != QSFALayout::PA_BSND) {
    return ERROR;
}
// Sink Token 数量必须等于 s2BaseSize（128）
if (hasSink && sinkTokenNum != 128) {
    return ERROR;
}
// key_sink 与 value_sink 须同时提供或同时为空
if ((keySink != nullptr) != (valueSink != nullptr)) {
    return ERROR;
}
```

#### 4.3.3 S2 循环边界调整

当启用 Sink 时，Tiling 侧在计算 `s2LoopEndIdx` 后额外加 1，为 Sink 迭代预留位置：

```cpp
// 原有逻辑计算 s2LoopEndIdx（基于 sparseBlockCount 和 s2BaseSize）
// ...

// [新增] Sink 额外迭代
if (sfaaInfo->hasSink) {
    s2LoopEndIdx += 1;  // S2 循环多一次迭代用于 Sink 块
}
```

> **注意**：`sparseBlockCount` 和 `actualS2Size` 不变。额外的迭代仅在 Kernel 侧被识别为 Sink 块，不影响正常 KV 的循环边界计算。

#### 4.3.4 Workspace 调整

Sink 不引入额外 Workspace 需求。Sink KV 数据通过 `key_sink` GM 指针直接访问，AIC 使用 MTE2 搬运到 L1，不占用额外 UB。

#### 4.3.5 FillTiling

在 `FillTilingBaseParamsMla()` 中新增：

```cpp
tilingData_.baseParams.hasSink = sfaaInfo->hasSink ? 1 : 0;
tilingData_.baseParams.sinkTokenNum = sfaaInfo->sinkTokenNum;
```

### 4.4 InferShape

**无需修改**。Sink 不影响输出 Shape。

---

## 5. Kernel 侧变更

### 5.1 入口函数（*.cpp）

**变更点**：将 `QSFA_OP_IMPL` 宏中的 `nullptr, nullptr` 替换为实际的 `key_sink, value_sink` 指针。

```cpp
// 修改前
op.Init(query, key, value, sparseIndices, keyScale, valueScale, blocktable,
    actualSeqLengthsQuery, actualSeqLengthsKV, nullptr, nullptr,
    attentionOut, user, tilingData, &tPipe);

// 修改后
op.Init(query, key, value, sparseIndices, keyScale, valueScale, blocktable,
    actualSeqLengthsQuery, actualSeqLengthsKV, key_sink, value_sink,
    attentionOut, user, tilingData, &tPipe);
```

### 5.2 主 Kernel 类（*_kernel_mla.h）

#### 5.2.1 新增成员变量

```cpp
class KvQuantSparseFlashAttentionMla {
private:
    // ... 现有成员 ...
    __gm__ uint8_t *keySinkAddr = nullptr;   // [新增] Sink KV 的 GM 地址
};
```

#### 5.2.2 Init 修改

在 `Init()` 中保存 `key_sink` 地址，并传递给 Cube 服务：

```cpp
void Init(..., __gm__ uint8_t *key_sink, __gm__ uint8_t *value_sink, ...) {
    // ... 现有逻辑 ...
    this->keySinkAddr = key_sink;  // [新增] 保存 Sink KV 地址

    // [新增] 传递给 Cube 服务
    cubeBlock.SetSinkKvAddr(key_sink);
}
```

#### 5.2.3 ComputeConstexpr 修改

从 CVSharedParams 解析 Sink 参数：

```cpp
void ComputeConstexpr() {
    // ... 现有逻辑 ...
    constInfo.hasSink = sharedParams.hasSink;          // [新增]
    constInfo.sinkTokenNum = sharedParams.sinkTokenNum; // [新增]
}
```

#### 5.2.4 ProcessMainLoop 核心修改

S2 循环的总迭代次数在 hasSink 时增加 1。第 0 次迭代为 Sink 块，后续迭代处理正常 KV。正常 KV 迭代使用 `effectiveS2LoopCount = s2LoopCount - sinkOffset` 访问 sparse_indices，确保寻址逻辑不变。

```cpp
// [新增] Sink 偏移量
int64_t sinkOffset = (constInfo.hasSink == 1) ? 1 : 0;

// S2 循环（s2LoopLimit 已由 Tiling 侧加上 sinkOffset）
for (int64_t s2LoopCount = 0; s2LoopCount <= s2LoopLimit; ++s2LoopCount) {
    if (notLastTwoLoop) {
        bool isSinkBlock = (s2LoopCount == 0) && (sinkOffset == 1);

        // [关键] 正常 KV 使用调整后的循环计数
        int64_t effectiveS2LoopCount = s2LoopCount - sinkOffset;

        RunInfo &runInfo1 = runInfo[taskId % 3];

        if (isSinkBlock) {
            // ============ Sink 块：不调用 SetRunInfo，不访问 sparse_indices ============
            // 仅设置 Sink 块所需的最小 RunInfo 信息
            runInfo1.s2LoopCount = 0;
            runInfo1.s2RealSize = constInfo.sinkTokenNum;  // 128

            if ASCEND_IS_AIC {
                this->cubeBlock.CopySinkKvToL1(
                    this->l1RightBuffers.Get(), runInfo1, this->constInfo);
                this->cubeBlock.IterateBmm1(
                    this->bmm1Buffers.Get(), this->l1RightBuffers.Get(),
                    runInfo1, this->constInfo);
            } else {
                this->vecBlock.ProcessVec0SinkSync(
                    this->l1RightBuffers.Get(), runInfo1, this->constInfo);
            }
        } else {
            // ============ 正常 KV 块：使用 effectiveS2LoopCount ============
            this->SetRunInfo(runInfo1, runParam, taskId,
                             effectiveS2LoopCount,  // [关键] 使用调整后的计数
                             s2LoopLimit - sinkOffset, multiCoreInnerIdx);

            if ASCEND_IS_AIC {
                this->cubeBlock.IterateBmm1(
                    this->bmm1Buffers.Get(), this->l1RightBuffers.Get(),
                    runInfo1, this->constInfo);
            } else {
                this->vecBlock.ProcessVec0(
                    this->l1RightBuffers.Get(), runInfo1, this->constInfo);
            }
        }
    }
    // ... Vec1、BMM2、Vec2 逻辑不变 ...
}
```

**关键设计要点**：

| 要点 | 说明 |
|------|------|
| `sinkOffset` | hasSink 时为 1，否则为 0。统一控制循环偏移 |
| Sink 块不调用 `SetRunInfo` | 避免访问 sparse_indices 和 block_table，防止非法地址计算 |
| `effectiveS2LoopCount` | 正常 KV 的有效循环计数，确保 `GetRealCmpS2Idx` 中 `topkKIdx = s2IdxInBase + effectiveS2LoopCount * s2BaseSize` 从 0 开始索引 sparse_indices |
| `s2LoopLimit - sinkOffset` | 传给 `SetRunInfo` 的 s2LoopLimit 减去 Sink 偏移，保持 tail 计算正确 |

### 5.3 Cube 服务变更（*_service_cube_mla.h）

#### 5.3.1 新增成员与接口

```cpp
class QSFAMatmulService {
public:
    // ... 现有接口 ...
    __aicore__ inline void SetSinkKvAddr(__gm__ uint8_t *keySink);  // [新增]
    __aicore__ inline void CopySinkKvToL1(                           // [新增]
        Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputL1,
        RunInfo &runInfo, ConstInfo &constInfo);
private:
    GlobalTensor<Q_T> keySinkGm;  // [新增] Sink KV GM 视图
};
```

#### 5.3.2 SetSinkKvAddr 实现

```cpp
void SetSinkKvAddr(__gm__ uint8_t *keySink) {
    if ASCEND_IS_AIC {
        if (keySink != nullptr) {
            keySinkGm.SetGlobalBuffer((__gm__ Q_T *)keySink);
        }
    }
}
```

#### 5.3.3 CopySinkKvToL1 实现

核心功能：将 BF16 Sink KV 从 GM 搬运到 L1 并转为 NZ 格式。

```cpp
void CopySinkKvToL1(
    Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputL1,
    RunInfo &runInfo, ConstInfo &constInfo)
{
    // 1. 等待核间同步（AIV 已释放 buffer）
    outputL1.WaitCrossCore();

    // 2. 将 Sink KV (BF16) 从 GM 搬运到 L1，格式转换为 NZ
    //    key_sink shape: (128, N2=1, D=576)
    //    目标 L1 格式需与 Vec0 反量化后的 KV NZ 格式一致
    LocalTensor<Q_T> l1RightTensor = outputL1.GetTensor<Q_T>();
    uint32_t sinkRows = constInfo.sinkTokenNum;  // 128
    uint32_t dSize = constInfo.dSize;             // 576 (nope + rope)

    // 使用 CopyToL1Nd2Nz 将 ND 格式转为 NZ 格式（与 Q 搬运共用的基础能力）
    CopyToL1Nd2Nz<Q_T>(l1RightTensor, keySinkGm, sinkRows, dSize, dSize);

    // 3. 设置核间同步，通知后续消费者（BMM1）数据就绪
    outputL1.SetCrossCore();
}
```

**关键说明**：
- `CopyToL1Nd2Nz` 是 AIC 侧的 MTE2 搬运函数，将 GM 上的 ND 格式数据转为 L1 上的 NZ 格式
- 该函数已被用于 Q 矩阵的 GM → L1 搬运，此处复用同一机制
- Sink KV 只搬运一次（128 行 × 576 列 BF16），恰好填满一个 S2 基本块

#### 5.3.4 IterateBmm1 修改

**无需修改**。Sink 块的 L1 数据格式与正常 KV 一致，BMM1 直接读取即可。唯一区别是 Sink 块的 L1 数据由 `CopySinkKvToL1` 而非 Vec0 写入。

注意：在 `IterateBmm1QSFA` 中，当 `effectiveS2LoopCount == 0` 时会加载 Q 到 L1。对于 Sink 块（物理 s2LoopCount == 0），Q 同样需要加载，因此 IterateBmm1 中的 Q 加载判断需要适配——当 hasSink 时，Q 加载应在 Sink 迭代（物理 s2LoopCount == 0）触发，而非 effectiveS2LoopCount == 0 触发。具体实现可通过传入 `isSinkBlock` 标记或始终在物理 s2LoopCount == 0 时加载 Q。

#### 5.3.5 IterateBmm2 修改

**无需修改**。BMM2 的右矩阵（V）来自 L1，格式与正常 KV 一致。

#### 5.3.6 QSFAMatmulServiceDummy 同步修改

```cpp
class QSFAMatmulServiceDummy {
public:
    // ... 现有空实现 ...
    __aicore__ inline void SetSinkKvAddr(__gm__ uint8_t *keySink) {}    // [新增]
    __aicore__ inline void CopySinkKvToL1(...) {}                        // [新增]
};
```

### 5.4 Vector 服务变更（*_service_vector_mla.h）

#### 5.4.1 新增 ProcessVec0SinkSync 方法

Sink 块时 AIV 不做数据搬运，仅维持核间同步握手：

```cpp
__aicore__ inline void ProcessVec0SinkSync(
    Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputL1,
    const RunInfo &runInfo, ConstInfo &constInfo)
{
    // 维持与正常 ProcessVec0 相同的核间同步协议
    outputL1.WaitCrossCore();   // 等待 buffer 可用
    // 不做任何数据操作（Sink KV 由 AIC 侧搬运）
    outputL1.SetCrossCore();    // 释放 buffer，通知 AIC 可以操作
}
```

#### 5.4.2 QSFAVectorServiceDummy 同步修改

```cpp
class QSFAVectorServiceDummy {
public:
    // ... 现有空实现 ...
    __aicore__ inline void ProcessVec0SinkSync(...) {}  // [新增]
};
```

### 5.5 运行时参数变更（util_regbase.h）

#### 5.5.1 ConstInfo 新增字段

```cpp
struct ConstInfo {
    // ... 现有字段 ...
    uint32_t hasSink;        // [新增] 是否启用 Sink（0/1）
    uint32_t sinkTokenNum;   // [新增] Sink Token 数量（128）
};
```

### 5.6 KV Cache 参数计算变更（*_kvcache.h）

#### 5.6.1 ComputeS2LoopInfo 无需修改

`sparseBlockCount` 和 `actualS2Size` 保持不变（框架侧不修改），因此 `ComputeS2LoopInfo` 计算的正常 KV 循环边界不受影响。Sink 额外迭代由 Tiling 侧在 `s2LoopEndIdx` 上 +1 完成（见 4.3.3），不需要 kvcache 模块参与。

#### 5.6.2 GetRealCmpS2Idx / GetkeyOffset 无需修改

正常 KV 迭代传入的 `runInfo.s2LoopCount` 已被 ProcessMainLoop 替换为 `effectiveS2LoopCount`（从 0 开始），因此 `GetRealCmpS2Idx` 中的索引计算：

```cpp
int64_t topkKIdx = s2IdxInBase + cmpS2LoopCnt * constInfo.s2BaseSize;
```

以及 `GetkeyOffset` 中的 block_table 寻址：

```cpp
int64_t blkTableIdx = s2Idx / blockSize;
int64_t blkTableOffset = s2Idx % blockSize;
realkeyOffset = blockTableGm.GetValue(boIdx * maxBlockNumPerBatch + blkTableIdx) * ...;
```

均与原始逻辑完全一致。**sparse_indices 和 block_table 的值没有任何变化，寻址结果完全正确。**

---

## 6. 核间同步协议

### 6.1 正常块的同步时序（无变化）

```
AIV (Vec0)                              AIC (BMM1)
    │                                       │
    ├─ WaitCrossCore()                      │
    │  （获取 L1 buffer 写权限）              │
    ├─ 搬入 INT8 KV → 反量化 → 写入 L1       │
    ├─ SetCrossCore()  ──────────────────→  ├─ WaitCrossCore()
    │                                       │  （获取 L1 buffer 读权限）
    │                                       ├─ 读 L1 → Matmul → 输出 UB
    │                                       │
```

### 6.2 Sink 块的同步时序（新增）

```
AIV (ProcessVec0SinkSync)               AIC (CopySinkKvToL1 + BMM1)
    │                                       │
    ├─ WaitCrossCore()                      │
    │  （获取 buffer 写权限）                 │
    ├─ （空操作，不做数据处理）                │
    ├─ SetCrossCore()  ──────────────────→  ├─ WaitCrossCore()
    │                                       │  （获取 buffer 操作权限）
    │                                       ├─ GM(BF16) → L1(NZ) 搬运
    │                                       ├─ SetCrossCore()
    │                                       ├─ IterateBmm1() 正常 matmul
    │                                       │
```

**关键点**：
- AIV 侧的 `WaitCrossCore()` → `SetCrossCore()` 保持同步状态机一致
- AIC 侧在获取 buffer 后先写入 Sink KV 数据，再进行 matmul
- 三缓冲轮转机制不受影响

---

## 7. 详细变更文件清单

| 文件 | 变更类型 | 变更内容 |
|------|---------|---------|
| `op_host/*_tiling.h` | **修改** | TilingData 新增 `hasSink`/`sinkTokenNum`；CVSharedParams 新增对应位域 |
| `op_host/*_tiling.cpp` | **修改** | 解析 `key_sink` 输入判定 Sink 启用；新增校验规则；`s2LoopEndIdx += 1`；FillTiling 填充新字段 |
| `op_kernel/*.cpp` | **修改** | QSFA_OP_IMPL 宏传递实际 `key_sink`/`value_sink` 指针 |
| `op_kernel/*_kernel_mla.h` | **修改** | 新增 `keySinkAddr` 成员；Init 传递地址；ProcessMainLoop 新增 `sinkOffset`/`effectiveS2LoopCount` 逻辑 + Sink 分支 |
| `op_kernel/*_service_cube_mla.h` | **修改** | 新增 `SetSinkKvAddr()`、`CopySinkKvToL1()` 方法 |
| `op_kernel/*_service_vector_mla.h` | **修改** | 新增 `ProcessVec0SinkSync()` 方法 |
| `op_kernel/util_regbase.h` | **修改** | ConstInfo 新增 `hasSink`/`sinkTokenNum` 字段 |
| `op_kernel/*_common.h` | **修改** | CVSharedParams 新增 `hasSink`/`sinkTokenNum` 位域 |
| `op_kernel/*_kvcache.h` | 无需修改 | sparse_indices / block_table 寻址逻辑不变 |
| `op_host/*_def.cpp` | 无需修改 | key_sink/value_sink 接口已预留 |
| `op_host/*_infershape.cpp` | 无需修改 | 输出 Shape 不受影响 |

---

## 8. 关键风险与应对

### 8.1 NZ 格式一致性

**风险**：AIC 侧 `CopyToL1Nd2Nz` 产生的 NZ 格式可能与 AIV 侧 `CopyOutKvUb2L1` 产生的格式不一致，导致 matmul 计算错误。

**应对**：
1. 实现后使用 DumpTensor 对比 Sink 块和正常块在 L1 上的数据布局
2. 若格式不一致，改用 AIV 侧搬运（从 `key_sink` GM 加载 BF16 到 UB，跳过反量化，直接以现有 `CopyOutKvUb2L1` 写入 L1），作为备选方案
3. 备选方案的代价是 AIV 参与数据搬运，但格式兼容性有保证

### 8.2 跨核同步状态机

**风险**：Sink 块改变了 AIV/AIC 的角色（AIV 不写数据，AIC 写数据），可能破坏三缓冲同步状态机。

**应对**：
1. AIV 严格保持 `WaitCrossCore → SetCrossCore` 对称调用
2. 三缓冲的 `Get()` 轮转顺序不变
3. 通过单步调试验证 Sink 块前后的 buffer 索引一致性

### 8.3 MLA 场景下 K=V 复用

**风险**：MLA 场景下 value 复用 key。Sink 场景中 BMM2 使用的 V 数据来自 L1，需确认 BMM2 读取的 V 数据区域与 Sink KV 写入的区域一致。

**应对**：
- BMM2 的 V 数据从 L1 右 buffer 的起始位置读取（nope 部分，512 列）
- Sink KV 写入 L1 时覆盖完整的 576 列（nope + rope）
- 因此 BMM2 读取的 V 数据自然包含在 Sink KV 数据中，无需额外处理

### 8.4 Q 加载时机

**风险**：原始代码在 `IterateBmm1` 中通过 `s2LoopCount == 0` 判断是否加载 Q 到 L1。引入 Sink 后，Sink 迭代的 s2LoopCount == 0，正常 KV 第一次迭代的 effectiveS2LoopCount == 0。需确保 Q 仅在 Sink 迭代加载一次。

**应对**：
- 在 Sink 迭代中，由 `CopySinkKvToL1` 完成后再调用 `IterateBmm1`，此时 Q 加载正常触发
- 正常 KV 迭代传入 effectiveS2LoopCount（从 0 开始），IterateBmm1 中判断 effectiveS2LoopCount == 0 时**不应再次加载 Q**
- 解决方案：为 IterateBmm1 传入物理 s2LoopCount 或 `isFirstIteration` 标记，确保 Q 只加载一次

### 8.5 hasSink == 0 时的回归

**风险**：新增的 `sinkOffset` 逻辑可能在 `hasSink == 0` 时引入退化。

**应对**：
- `sinkOffset = 0` 时，`effectiveS2LoopCount = s2LoopCount`，SetRunInfo 传入的 s2LoopLimit 不变
- 逻辑完全退化为原始路径，零开销
- 回归测试必须覆盖 `hasSink == 0` 场景

---

## 9. 验证方案

### 9.1 功能验证

| 测试场景 | 验证点 |
|---------|--------|
| Sink 开启 + PA_BSND + BF16 | 基本功能正确，输出 allclose（atol=1e-3, rtol=1e-3） |
| Sink 关闭 (key_sink=None) | 回归现有功能，输出完全一致 |
| 不同 sparseBlockCount | Sink 块 + 不同数量正常块组合 |
| 多 Batch | 每个 Batch 都有独立的 Sink 块 |
| 不同 actualS2Size | Sink 块（128）+ 不同数量正常 KV 块 |

### 9.2 精度验证

构建 CPU 参考实现：
1. 将 Sink KV (BF16) 拼接到 KV 序列开头
2. 对拼接后的 KV 执行完整 attention 计算
3. 对比 NPU 输出与 CPU 参考输出

### 9.3 边界条件

| 边界场景 | 预期行为 |
|---------|---------|
| sparseBlockCount == 0（仅 Sink 块，无正常 KV） | 只处理 Sink KV，输出正确 |
| S1 = 1（单 token decode） | 最常见推理场景，需重点验证 |
| 多核分配使某核无工作量 | 空闲核正常退出，不 crash |
| hasSink == 0（Sink 关闭） | 完全回归原始逻辑，结果一致 |
| sparseBlockCount 为奇数 | Sink 迭代 + 正常迭代的 tail 处理正确 |
