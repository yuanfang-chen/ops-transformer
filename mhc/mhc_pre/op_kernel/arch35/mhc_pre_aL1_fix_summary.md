# mhc_pre.h aL1_ 修复总结

## 问题背景

在 Ascend 异构芯片上，一个 Cube 核对应 2 个 Vector 核。Vector 核负责将数据从 UB 搬运到 L1 (aL1_)，Cube 核从 L1 读取数据进行矩阵运算。

## 发现的问题及修复

### 1. aL1_ 初始化大小错误 (第282行)

**问题**：`LocalTensor` 构造函数第三个参数是元素个数，不是字节大小。

```cpp
// 修改前（错误）
aL1_ = LocalTensor<P>(TPosition::TSCM, 0, mnConfig_.singleCoreM * mnConfig_.singleCoreK * sizeof(P));

// 修改后（正确）
aL1_ = LocalTensor<P>(TPosition::TSCM, 0, mnConfig_.singleCoreM * mnConfig_.singleCoreK);
```

### 2. DataCopy 偏移计算错误 (第948-950行)

**问题**：offset 计算需要考虑：
- Vector 核在 M 方向的偏移 (subBlockIdx_)
- 当前处理的 N 方向位置 (offsetNd)

```cpp
// 修改前
uint64_t offset = offsetM * curNdLen;

// 修改后
uint64_t l1OffsetM = offsetM + subBlockIdx_ * vectorOffset_.singleCoreM;
uint64_t offset = l1OffsetM * matrixInfo_.nD + offsetNd;
```

### 3. 同步等待逻辑 (第541-543行)

**问题**：每次处理新的 offsetNd 前，需要等待 Cube 处理完上一个 offsetNd 的数据，否则会覆盖。

```cpp
// 修改前
if (vectorCount_ >= 2) {
    AscendC::CrossCoreWaitFlag(SYNC_C2V);
}

// 修改后
if (offsetNd > 0) {
    AscendC::CrossCoreWaitFlag(SYNC_C2V); // 等待cube处理完上一个offsetNd的数据
}
```

### 4. 清理无用代码

- 删除第558行未使用的 `aL1Offset` 变量
- 删除第591行无用的 `vectorCount_++`
- 删除无用的注释代码

## 数据布局说明

### Vector 核任务划分

- **Vector 0** (subBlockIdx_=0): 处理 M 方向 [0, singleCoreM)
- **Vector 1** (subBlockIdx_=1): 处理 M 方向 [singleCoreM, curSingleT)

### aL1_ 写入偏移

| Vector核 | 写入位置 |
|----------|---------|
| Vector 0 | [0, singleCoreM) × nD |
| Vector 1 | [singleCoreM, curSingleT) × nD |

两个 Vector 核写入 aL1_ 的不同区域，不会冲突。

## 同步流程

```
offsetNd = 0:
  Vector: 处理数据 → DataCopy to aL1_ → SetFlag(SYNC_V2C, PIPE_MTE3)
  Cube:   WaitFlag(SYNC_V2C) → 处理数据 → SetFlag(SYNC_C2V, PIPE_MTE1)

offsetNd > 0:
  Vector: WaitFlag(SYNC_C2V) → 处理数据 → DataCopy to aL1_ → SetFlag(SYNC_V2C, PIPE_MTE3)
  Cube:   WaitFlag(SYNC_V2C) → 处理数据 → SetFlag(SYNC_C2V, PIPE_MTE1)
```

### 同步管道说明

| 方向 | 同步管道 |
|------|---------|
| Vector → Cube (UB→L1) | PIPE_MTE3 |
| Cube → Vector | PIPE_MTE1 |

## 参考实现

参考了 `ops-nn/matmul/weight_quant_batch_matmul_v2/op_kernel/arch35/weight_quant_batch_matmul_v2_reg_base_common.h` 中的实现：
- `l1Local_` 作为 UB → L1 的中转缓冲区
- 使用 `PIPE_MTE3` 进行 Vector → Cube 同步
- 使用 `PIPE_MTE1` 进行 Cube → Vector 同步
