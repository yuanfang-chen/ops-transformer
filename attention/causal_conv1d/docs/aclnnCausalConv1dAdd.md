# aclnnCausalConv1dAdd

[📄 查看源码](https://gitcode.com/cann/ops-transformer/tree/master/attention/causal_conv1d)

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>    |     ×    |
|  <term>Atlas 训练系列产品</term>    |     ×    |

## 功能说明

- 接口功能：对序列执行因果一维卷积，沿序列维度使用缓存数据（长度为卷积核宽减1）对各序列头部进行padding，确保输出依赖当前及历史输入；卷积完成后，将当前序列尾部的数据（长度为卷积核宽减1）更新到缓存。<br>

- 支持以下场景：
  - 场景一（prefill场景）：
    ```
    x: [cu_seq_len, dim]
    weight: [K, dim]，其中K=3
    convStates: [num_slots, K-1, dim]
    queryStartLoc: [batch+1]
    cacheIndices: [batch]
    initialStateMode: [batch]
    bias: [dim]（可选）
    numAcceptedTokens: [batch]（可选）
    y: [cu_seq_len, dim]
    runMode: 0
    ```
    其中cu_seq_len为batch内所有变长序列拼接后的总长度，每个序列卷积前使用长度为K-1的缓存数据对序列头部进行padding，保证因果性。

  - 场景二（decode场景 - 变长序列）：
    ```
    x: [cu_seq_len, dim]
    weight: [K, dim]，其中K=3
    convStates: [num_slots, K-1, dim]
    queryStartLoc: [batch+1]
    cacheIndices: [batch]
    initialStateMode: [batch]
    bias: [dim]（可选）
    numAcceptedTokens: [batch]（用于投机解码）
    y: [cu_seq_len, dim]
    runMode: 1
    ```

  - 场景三（decode场景 - 固定batch）：
    ```
    x: [batch, m+1, dim]
    weight: [K, dim]，其中K=3
    convStates: [num_slots, K-1, dim]
    queryStartLoc: [batch+1]（可选）
    cacheIndices: [batch]
    initialStateMode: [batch]
    bias: [dim]（可选）
    numAcceptedTokens: [batch]（用于投机解码，m为投机token个数）
    y: [batch, m+1, dim]
    runMode: 1
    ```

- 计算公式：

  K是卷积核宽度（固定为3），L是原始序列长度，dim是特征维度。

  1. 缓存拼接：

    $$
    x'[i, dim] =
    \begin{cases}
    cacheState[i, dim], & 0 \leq i < K-1 \\
    x[i - (K-1), dim], & K-1 \leq i < L + K - 1
    \end{cases}
    $$

  2. 因果1维卷积：

    $$
    y[i, dim] = \sum_{k=0}^{K-1} w[k, dim] \cdot x'[i + k, dim]
    $$

  3. 缓存更新：

    $$
    cacheState[i, dim] = x'[L + i, dim], \quad i = 0, 1, \dots, K-2
    $$

  4. 残差连接（可选）：

    $$
    y[i, dim] += x[i, dim]
    $$

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用 "aclnnCausalConv1dAddGetWorkspaceSize"接口获取入参并计算所需workspace大小以及包含了算子计算流程的执行器，再调用"aclnnCausalConv1dAdd"接口执行计算。

```Cpp
aclnnStatus aclnnCausalConv1dAddGetWorkspaceSize(
  const aclTensor *x,
  const aclTensor *weight,
  aclTensor       *convStates,
  const aclTensor *queryStartLoc,
  const aclTensor *cacheIndices,
  const aclTensor *initialStateMode,
  const aclTensor *bias,
  const aclTensor *numAcceptedTokens,
  int64_t          activationMode,
  int64_t          padSlotId,
  int64_t          runMode,
  int64_t          residualConnection,
  const aclTensor *y,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor)