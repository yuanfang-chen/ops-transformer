# Block Sparse Attention 支持 LSE 输出详设

## 1. 文档目的

本文档基于最近两次提交的代码实现，说明 `BlockSparseAttention` 算子新增 `LSE(Log-Sum-Exp)` 输出能力的详细设计。对应提交如下：

- `340c2f0b`：`feat: block sparse attention support lse`
- `e59bb075`：`fix`

本文重点描述以下内容：

- 为什么要为 `BlockSparseAttention` 增加 `softmaxLse` 输出
- Host API、算子定义、shape 推导、tiling、kernel 分发和 epilogue 的改动点
- `fix` 提交中对默认行为和输出 shape 的修正

## 2. 背景与目标

在 Attention 类算子中，`LSE` 常用于以下场景：

- 为后续反向计算提供数值稳定的 `logsumexp` 中间量
- 为融合算子或上层框架保留 softmax 归一化相关统计信息
- 在 block sparse 场景下复用在线 softmax 的中间统计结果，避免额外重算

本次设计目标如下：

- 在不破坏原有 `attentionOut` 语义的前提下，为 `BlockSparseAttention` 增加可选输出 `softmaxLse`
- 当 `softmaxLseFlag = 0` 时，保持原有执行路径和调用方式兼容
- 当 `softmaxLseFlag = 1` 时，在在线 softmax 计算结束后输出每个 query 位置对应的 `LSE`
- 同时支持 `TND` 和 `BNSD` 两种输入布局
- 同时支持 `FP16`/`BF16` 输入，其中 `softmaxLse` 固定输出为 `FLOAT`

## 3. 设计范围

本次功能覆盖以下模块：

- Host 接口层
  - `aclnnBlockSparseAttentionGetWorkspaceSize`
  - `l0op::BlockSparseAttention`
  - 算子注册定义 `OpDef`
- 图编译侧
  - `InferShape`
  - `InferDataType`
  - `TilingData`
  - `tilingKey` 生成
- Kernel 侧
  - kernel 模板实例化与分发
  - 在线 softmax epilogue
  - 输出重标定 epilogue
  - `softmaxLse` 写回 GM

不在本次设计范围内的内容：

- `attentionOut` 数学语义变更
- block sparse 掩码语义重定义
- paged cache 新能力扩展

## 4. 总体方案

### 4.1 方案概述

整体设计采用“可选输出 + 编译期分发”的方式实现：

1. Host 侧新增 `softmaxLseFlag` 属性和 `softmaxLseOptional` 输出张量。
2. InferShape/InferDataType 为第二输出设置 shape 和 dtype。
3. Tiling 层读取 `softmaxLseFlag`，将其编码进 `tilingKey`。
4. Kernel 分发时根据 `tilingKey` 选择：
   - `LseMode::NONE`：不输出 `LSE`
   - `LseMode::OUT_ONLY`：输出 `LSE`
5. 在线 softmax 继续维护逐行统计量：
   - `gm`：当前全局行最大值
   - `gl`：当前全局行和
   - `dm`：跨 tile 重标定因子
6. 在最后一个 KV stack tile 完成后，由 `rescale O` epilogue 计算：

```text
LSE = log(gl) + gm
```

并写回 `softmaxLse` 输出张量。

### 4.2 设计原则

- 默认关闭：避免老调用方在未显式申请 `softmaxLse` 时受到影响
- 复用在线 softmax 中间统计量：避免重复遍历或额外 kernel
- 通过 `tilingKey` 做编译期模板分发：避免在 kernel 热路径中引入额外运行时分支
- 输出 dtype 固定为 `FLOAT`：保证数值稳定性和跨精度一致性

## 5. 外部接口设计

### 5.1 ACLNN 接口变更

`aclnnBlockSparseAttentionGetWorkspaceSize` 增加两个与 LSE 能力相关的参数：

- `int64_t softmaxLseFlag`
- `aclTensor *softmaxLseOptional`

含义如下：

- `softmaxLseFlag = 0`：不要求输出 `LSE`
- `softmaxLseFlag = 1`：要求输出 `LSE`
- `softmaxLseOptional`：当 `softmaxLseFlag = 1` 时，由调用方提供的输出 Tensor

### 5.2 OpDef 变更

算子定义新增第二个输出：

- `attentionOut`：主输出
- `softmaxLse`：可选输出，dtype 为 `FLOAT`

同时新增/调整属性：

- `softmaxLseFlag`

`fix` 提交将 `softmaxLseFlag` 默认值由 `1` 改为 `0`，这是一个关键兼容性修正，原因如下：

- 若默认值为 `1`，旧调用流程在未准备第二输出时会产生行为歧义
- 默认关闭更符合“可选输出”的接口语义
- 调用方只有在明确需要 `LSE` 时才打开该能力

## 6. 输出语义与数据规格

### 6.1 attentionOut

保持原语义不变：

```text
attentionOut = Softmax(QK^T * scale + mask) * V
```

### 6.2 softmaxLse

新增输出 `softmaxLse` 表示每一行 softmax 分母对应的对数形式：

```text
softmaxLse = log(sum(exp(score_i))) 
```

在在线 softmax 分块累计实现中，对应为：

```text
softmaxLse = log(gl) + gm
```

其中：

- `gm` 是当前行所有已处理分块中的全局最大值
- `gl` 是以 `gm` 为基准的指数和

### 6.3 输出 shape

`fix` 提交修正了 `softmaxLse` 的 shape 设计，最终规则如下：

| 输入布局 | `attentionOut` shape | `softmaxLse` shape |
| --- | --- | --- |
| `TND` | 与 `query` 相同，即 `[T, N, D]` | `[T, N, 1]` |
| `BNSD` | 与 `query` 相同，即 `[B, N, S, D]` | `[B, N, S, 1]` |

设计原因：

- `softmaxLse` 本质上对应每个 `(token, head)` 的标量统计量
- 保留尾轴 `1` 后，shape 语义与 `attentionOut` 更一致，便于上层框架做广播或后续融合
- 相比早期的 `[T, N]` / `[B, N, S]`，尾轴 `1` 的表达更完整

### 6.4 输出 dtype

`softmaxLse` 固定为 `FLOAT`，即使输入为 `FP16/BF16` 也不改变。

这样设计的原因是：

- `LSE` 是数值敏感统计量
- 使用 `FLOAT` 可以降低累计误差
- 统一上层接口语义，减少调用方针对输入精度做分支处理

## 7. Host 侧详细设计

### 7.1 参数传递链路

LSE 输出控制从 Host 到 Kernel 的传递链路如下：

```text
aclnnBlockSparseAttentionGetWorkspaceSize
  -> l0op::BlockSparseAttention
  -> INFER_SHAPE / ADD_TO_LAUNCHER_LIST_AICORE
  -> Tiling 读取 attr
  -> 生成 tilingKey
  -> kernel 根据 tilingKey 分发到 LSE_MODE
```

### 7.2 `aclnnBlockSparseAttentionGetWorkspaceSize`

该接口中的关键行为如下：

- 接收 `softmaxLseFlag`
- 接收 `softmaxLseOptional`
- 调用 `l0op::BlockSparseAttention(...)` 生成两个内部输出
- 始终对主输出执行 `ViewCopy(outputs[0], attentionOut, ...)`
- 仅当 `softmaxLseFlag == 1` 时，对第二输出执行 `ViewCopy(outputs[1], softmaxLseOptional, ...)`

这意味着：

- 内部图仍然统一构建两个输出
- 是否真正拷贝给用户侧 `softmaxLseOptional`，由 `softmaxLseFlag` 决定

### 7.3 `l0op::BlockSparseAttention`

该层的职责是把 Host 侧的参数组织成图编译输入：

- 分配 `attentionOutTensor`
- 分配 `softmaxLseTensor`
- 将 `softmaxLseFlag` 作为 attr 传给 `INFER_SHAPE`
- 将 `softmaxLseFlag` 继续传给 `ADD_TO_LAUNCHER_LIST_AICORE`

这里的设计重点是：

- `softmaxLseFlag` 不作为单独输入 Tensor，而是作为 attr 控制编译期分支
- `softmaxLse` 输出 Tensor 始终存在于内部图中，但只有在 flag 打开时才对外暴露有效结果

## 8. Shape 推导设计

### 8.1 InferShape

`InferShapeBlockSparseAttention` 的处理逻辑如下：

- `attentionOutShape` 直接继承 `queryShape`
- `softmaxLseShape` 根据 `qInputLayout` 单独构造

具体规则：

- `TND`
  - `attentionOut = [T, N, D]`
  - `softmaxLse = [T, N, 1]`
- `BNSD`
  - `attentionOut = [B, N, S, D]`
  - `softmaxLse = [B, N, S, 1]`

对未知 shape 的处理：

- 若输入存在 `UNKNOWN_DIMS`
- 两个输出均退化为 `[-2]`

### 8.2 InferDataType

数据类型推导规则如下：

- 输出 0：继承输入 `query` 的 dtype
- 输出 1：固定为 `DT_FLOAT`

## 9. Tiling 设计

### 9.1 Tiling 状态

Tiling 层新增内部状态：

- `softmaxLseFlag_`

其作用是：

- 记录本次编译是否需要 LSE 输出
- 参与 `tilingKey` 编码
- 驱动 kernel 实例化分支选择

### 9.2 TilingKey 编码

`GenerateTilingKey()` 在原有编码基础上，新增了 LSE 输出标记：

```text
if (softmaxLseFlag_) {
    tilingKey += 100000000ULL;
}
```

这意味着：

- 不输出 LSE 的 key 保持原编码
- 输出 LSE 的 key 在高位增加一个独立标识位

这种设计的优点：

- 与 dtype/layout/softmax 精度等其他维度正交
- 不破坏原有 key 编码规则
- kernel 可以直接用宏常量做静态分发

### 9.3 已支持的 LSE 分发组合

当前代码已经补齐以下 `LSE_OUT` 版本：

- `FP16 + TND + float softmax`
- `FP16 + TND + half softmax`
- `BF16 + TND + float softmax`
- `FP16 + BNSD + float softmax`
- `FP16 + BNSD + half softmax`
- `BF16 + BNSD + float softmax`

说明：

- `BF16 + half softmax` 本身不在支持范围内，因此也没有对应 LSE 版本

## 10. Kernel 设计

### 10.1 模板扩展

`BlockSparseAttentionInfer` 模板新增 `lseMode` 模板参数：

```cpp
template <
    typename InputDtype,
    typename SoftmaxDtype,
    Epilogue::LseMode lseMode,
    uint32_t QueryLayout,
    uint32_t KvCacheLayout>
```

其中：

- `LseMode::NONE`：不输出 LSE
- `LseMode::OUT_ONLY`：输出 LSE

这样做的目的，是把是否输出 `LSE` 转化为编译期常量，减少运行时判断。

### 10.2 Kernel 分发

`block_sparse_attention.cpp` 中根据 `tilingKey` 分发到不同模板实例：

- 普通路径使用 `Epilogue::LseMode::NONE`
- LSE 输出路径使用 `Epilogue::LseMode::OUT_ONLY`

例如：

- `QF16_KVF16_TND_TND_NOCACHE_FLOATSM_NOMASK_RFA_TILING`
  - 对应 `LseMode::NONE`
- `QF16_KVF16_TND_TND_NOCACHE_FLOATSM_NOMASK_RFA_TILING_LSE_OUT`
  - 对应 `LseMode::OUT_ONLY`

### 10.3 Kernel 参数与 GM 绑定

`BlockSparseAttentionKernel` 中新增/强化了以下绑定：

- `gLse.SetGlobalBuffer((__gm__ ElementLse *)params.lse);`
- 根据布局计算 `gmOffsetLse`
- 根据布局构造 `LayoutLse`

两种布局下的寻址逻辑如下：

- `TND`
  - 本质对应 `[T, N, 1]`
  - `gmOffsetLse = lseBOffset + qSeqOffset * qHeads + qHeadIdx`
- `BNSD`
  - 本质对应 `[B, N, S, 1]`
  - `gmOffsetLse = lseBOffset + qHeadIdx * maxQSeqlen + qSeqOffset`

## 11. 在线 Softmax 与 LSE 计算设计

### 11.1 在线 softmax 累计状态

在 block sparse attention 中，QK 结果按 KV stack tile 分批处理。为了保证数值稳定性，softmax 采用在线归约方式维护以下状态：

- `gm`：行最大值
- `gl`：以 `gm` 为基准的指数和
- `dm`：上一轮到当前轮的重标定系数，通常可理解为 `exp(gm_old - gm_new)`

每处理一个 stack tile，执行：

1. 计算局部 `row max`
2. 更新全局 `gm`
3. 用新的 `gm` 对已有累计量做 rescale
4. 计算局部 `exp`
5. 更新全局 `gl`

### 11.2 为什么 LSE 放在 `rescale O` 阶段输出

`LSE` 的最终值依赖完整的 `gm` 和 `gl`：

```text
LSE = log(gl) + gm
```

只有当最后一个 stack tile 处理完成后，这两个量才是最终值。因此最合适的输出位置是 `rescale O` epilogue 的末尾，而不是中间的 online softmax 阶段。

这样设计的优点：

- 不需要为中间态增加额外外存写回
- 与最终 `O = O_acc / gl` 的归一化时机一致
- 只在最后一次处理时写出一次 `LSE`

## 12. Epilogue 详细设计

### 12.1 `block_epilogue_online_softmax`

在 `LSE_MODE == OUT_ONLY` 时，online softmax 模块本身不直接生成最终 LSE，而是做两件事：

- 继续维护 `gm/gl/dm`
- 通过事件同步为最后阶段的 `LSE` 写回让出/协调 UB 资源

可以理解为：online softmax 阶段负责“准备好最终所需统计量”，真正落盘发生在 `rescale O`。

### 12.2 `block_epilogue_rescale_o`

这是 `LSE` 输出的核心落点。

当满足以下条件时：

- `LSE_MODE == OUT_ONLY`
- 当前处理的是最后一个 stack tile
- 当前处理的是最后一个 row loop

执行：

1. 对 `gl` 做 `ln`
2. 将结果加上 `gm`
3. 得到最终 `LSE`
4. 以 `float` 形式写回 `gLse`

对应数学表达式为：

```text
LSE = log(gl) + gm
```

### 12.3 高精度与低精度分支

代码同时在以下两个 epilogue 中补齐了 LSE 输出逻辑：

- `block_epilogue_rescale_o.hpp`
- `block_epilogue_rescale_o_low_prec.hpp`

差异点：

- `float softmax` 路径直接在 `float` 空间计算 `ln(gl) + gm`
- `half softmax` 路径先在半精度空间得到结果，再显式 `Cast` 到 `float` 后写出

这样保证：

- 内部计算路径与原有 softmax 精度策略一致
- 对外输出 `softmaxLse` 统一为 `float`

## 13. 与原路径的兼容性设计

### 13.1 默认行为兼容

`fix` 提交后，`softmaxLseFlag` 默认值为 `0`，因此旧调用方在不关心 `LSE` 时：

- 接口语义保持不变
- 主输出 `attentionOut` 保持不变
- kernel 仍走 `LseMode::NONE` 分支

### 13.2 主输出不受影响

本次改动不改变以下行为：

- 稀疏块选择方式
- `attentionOut` 的 shape
- `attentionOut` 的 dtype
- Q/K/V layout 解析方式
- block sparse 计算主流程

新增 `softmaxLse` 只是在最终归一化阶段多写出一个统计量。

## 14. `fix` 提交的关键修正

后续 `e59bb075` 提交对首版实现做了两项关键修正：

### 14.1 修正 `softmaxLseFlag` 默认值

从：

```text
softmaxLseFlag = 1
```

修正为：

```text
softmaxLseFlag = 0
```

意义：

- 避免默认打开第二输出引发兼容性问题
- 与“可选输出”语义保持一致

### 14.2 修正 `softmaxLse` 输出 shape

从早期的：

- `TND -> [T, N]`
- `BNSD -> [B, N, S]`

修正为最终版本：

- `TND -> [T, N, 1]`
- `BNSD -> [B, N, S, 1]`

意义：

- 明确 `LSE` 是每个位置的标量输出
- 提高与上层广播/融合逻辑的兼容性

## 15. 约束与注意事项

### 15.1 功能使用约束

- 只有当调用方显式设置 `softmaxLseFlag = 1` 时，才会输出有效的 `softmaxLse`
- `softmaxLse` 输出 Tensor 应按 `FLOAT` 类型创建
- `softmaxLse` 的 shape 必须与布局规则匹配
- `TND` 和 `BNSD` 两种布局都支持 `LSE` 输出

### 15.2 精度约束

- 输入为 `BF16` 时，`innerPrecise` 仍要求走 `float softmax`
- `softmaxLse` 对外统一为 `float`

### 15.3 性能影响

开启 `LSE` 输出后，额外开销主要来自：

- `tilingKey` 分支切换到 `OUT_ONLY`
- `rescale O` 末尾增加 `ln + add + GM copy`

由于该逻辑只在最后一个 stack tile、最后一个 row loop 执行一次，因此额外开销相对可控。

## 16. 测试建议

建议至少覆盖以下测试组合：

- 功能正确性
  - `softmaxLseFlag = 0` 时仅校验 `attentionOut`
  - `softmaxLseFlag = 1` 时同时校验 `attentionOut` 和 `softmaxLse`
- 布局覆盖
  - `TND`
  - `BNSD`
- 精度覆盖
  - `FP16 + float softmax`
  - `FP16 + half softmax`
  - `BF16 + float softmax`
- shape 覆盖
  - 规则 shape
  - 非整块尾块 shape
- 回归验证
  - 对比关闭 `LSE` 前后的 `attentionOut` 一致性
  - 校验 `softmaxLse` shape 是否为 `[T, N, 1]` 或 `[B, N, S, 1]`

## 17. 总结

本次 `BlockSparseAttention` 支持 `LSE` 输出的实现，核心思路是：

- Host 侧新增可选输出和控制开关
- 图编译侧把该开关编码进 `tilingKey`
- Kernel 侧基于模板参数 `LseMode` 做静态分发
- 在线 softmax 继续维护 `gm/gl/dm`
- 在 `rescale O` 最终阶段计算并写回 `LSE = log(gl) + gm`

`fix` 提交进一步保证了该能力的可用性和兼容性：

- 默认关闭 `LSE` 输出
- 明确 `softmaxLse` shape 为带尾轴 `1` 的标量张量表示

因此，当前实现已经形成一条完整且闭环的 `LSE` 输出链路，能够在不改变主输出语义的前提下，为 block sparse attention 提供稳定、可选、可扩展的 `LSE` 输出能力。
