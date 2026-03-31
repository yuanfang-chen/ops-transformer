# aclnnAllGatherAdd

[📄 查看源码](https://gitcode.com/cann/ops-transformer/tree/master/mc2/all_gather_add)

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term> | 开发中 |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | 开发中 |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | 开发中 |
| <term>Atlas 200I/500 A2 推理产品</term> | × |
| <term>Atlas 推理系列产品</term> | × |
| <term>Atlas 训练系列产品</term> | × |

**说明：** 当前文档用于说明 `all_gather_add` 的接口语义与约束，正式产品支持情况以最终发布版本为准。

## 功能说明

- **接口功能**：对输入矩阵 `A` 执行 `AllGather` 通信，并将 Gather 后的结果与矩阵 `B` 做逐元素加法，输出 Gather 结果和加法结果。
- **计算公式**：

    $$
    aGathered = AllGather(A)
    $$

    $$
    C = aGathered + B
    $$

- **轮次处理语义**：单次算子执行内部轮次由输入参数 `commTurn` 决定。
- **当前版本约束**：当前版本仅支持 `commTurn = 2`，因此当前冻结场景下分为 2 轮，每轮处理每个 rank 本地 `A` 的 512 行。
  - Round 0：处理各 rank 本地 `A[0:512, :]`
  - Round 1：处理各 rank 本地 `A[512:1024, :]`

在 2 卡场景下，全局落位区间如下：

```text
Round 0:
  rank0 local A[0:512,    :] -> aGathered[0:512,    :]
  rank1 local A[0:512,    :] -> aGathered[1024:1536, :]

Round 1:
  rank0 local A[512:1024, :] -> aGathered[512:1024,  :]
  rank1 local A[512:1024, :] -> aGathered[1536:2048, :]
```

## 函数原型

每个算子分为两段式接口，必须先调用 `aclnnAllGatherAddGetWorkspaceSize` 接口获取计算所需 workspace 大小以及包含了算子计算流程的执行器，再调用 `aclnnAllGatherAdd` 接口执行计算。

```cpp
aclnnStatus aclnnAllGatherAddGetWorkspaceSize(
    const aclTensor *a,
    const aclTensor *b,
    const char      *group,
    int64_t         rankSize,
    int64_t         commTurn,
    aclTensor       *aGathered,
    aclTensor       *c,
    uint64_t        *workspaceSize,
    aclOpExecutor   **executor)
```

```cpp
aclnnStatus aclnnAllGatherAdd(
    void          *workspace,
    uint64_t      workspaceSize,
    aclOpExecutor *executor,
    aclrtStream   stream)
```

## aclnnAllGatherAddGetWorkspaceSize

- **参数说明：**

    <table style="undefined;table-layout: fixed; width: 1567px"><colgroup>
    <col style="width: 170px">
    <col style="width: 120px">
    <col style="width: 260px">
    <col style="width: 360px">
    <col style="width: 180px">
    <col style="width: 100px">
    <col style="width: 190px">
    <col style="width: 145px">
    </colgroup>
    <thead>
    <tr>
        <th>参数名</th>
        <th>输入/输出</th>
        <th>描述</th>
        <th>使用说明</th>
        <th>数据类型</th>
        <th>数据格式</th>
        <th>维度(shape)</th>
        <th>非连续Tensor</th>
    </tr></thead>
    <tbody>
    <tr>
        <td>a</td>
        <td>输入</td>
        <td>每张卡上的本地输入矩阵。</td>
        <td><ul><li>当前版本仅支持二维输入。</li><li>当前固定 shape 为 `[1024, 2048]`。</li><li>参与 AllGather 通信的输入。</li></ul></td>
        <td>FLOAT16</td>
        <td>ND</td>
        <td>[1024, 2048]</td>
        <td>×</td>
    </tr>
    <tr>
        <td>b</td>
        <td>输入</td>
        <td>与 Gather 后结果做逐元素加法的输入矩阵。</td>
        <td><ul><li>当前版本仅支持二维输入。</li><li>当前固定 shape 为 `[2048, 2048]`。</li><li>与 `aGathered` 按元素对齐相加。</li></ul></td>
        <td>FLOAT16</td>
        <td>ND</td>
        <td>[2048, 2048]</td>
        <td>×</td>
    </tr>
    <tr>
        <td>group</td>
        <td>输入</td>
        <td>通信域名称。</td>
        <td>通过 HCCL 提供的通信域获取接口得到，不能为空字符串。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>rankSize</td>
        <td>输入</td>
        <td>通信域内参与通信的卡数。</td>
        <td>当前版本仅支持输入 `2`。</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>commTurn</td>
        <td>输入</td>
        <td>通信轮次。</td>
        <td><ul><li>单次算子执行内部轮次由该参数决定。</li><li>当前版本仅支持输入 `2`。</li><li>在当前冻结场景下表示内部执行 2 轮，每轮处理每个 rank 本地 `A` 的 512 行。</li></ul></td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>aGathered</td>
        <td>输出</td>
        <td>AllGather 通信后的结果。</td>
        <td><ul><li>当前版本仅支持二维输出。</li><li>当前固定 shape 为 `[2048, 2048]`。</li><li>数据类型、数据格式与 `a` 保持一致。</li></ul></td>
        <td>FLOAT16</td>
        <td>ND</td>
        <td>[2048, 2048]</td>
        <td>×</td>
    </tr>
    <tr>
        <td>c</td>
        <td>输出</td>
        <td>逐元素加法结果。</td>
        <td><ul><li>当前版本仅支持二维输出。</li><li>当前固定 shape 为 `[2048, 2048]`。</li><li>计算公式为 `c = aGathered + b`。</li></ul></td>
        <td>FLOAT16</td>
        <td>ND</td>
        <td>[2048, 2048]</td>
        <td>×</td>
    </tr>
    <tr>
        <td>workspaceSize</td>
        <td>输出</td>
        <td>返回需要在 Device 侧申请的 workspace 大小。</td>
        <td>-</td>
        <td>UINT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>executor</td>
        <td>输出</td>
        <td>返回 op 执行器，包含了算子计算流程。</td>
        <td>-</td>
        <td>aclOpExecutor*</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    </tbody></table>

- **返回值：**

    aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

    第一段接口完成入参校验，出现以下场景时报错：

    <table style="undefined;table-layout: fixed; width: 1149px"><colgroup>
    <col style="width: 282px">
    <col style="width: 120px">
    <col style="width: 747px">
    </colgroup>
    <thead>
    <tr>
        <th>返回值</th>
        <th>错误码</th>
        <th>描述</th>
    </tr></thead>
    <tbody>
    <tr>
        <td>ACLNN_ERR_PARAM_NULLPTR</td>
        <td>161001</td>
        <td>输入或输出的必选参数是空指针。</td>
    </tr>
    <tr>
        <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
        <td rowspan="3">161002</td>
        <td>输入或输出的数据类型、数据格式不在支持范围内。</td>
    </tr>
    <tr>
        <td>输入或输出的 shape 不在支持范围内。</td>
    </tr>
    <tr>
        <td>`rankSize`、`commTurn` 或 `group` 的取值不在支持范围内。</td>
    </tr>
    </tbody></table>

## aclnnAllGatherAdd

- **参数说明：**

    <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
    <col style="width: 168px">
    <col style="width: 128px">
    <col style="width: 854px">
    </colgroup>
    <thead>
    <tr>
        <th>参数名</th>
        <th>输入/输出</th>
        <th>描述</th>
    </tr></thead>
    <tbody>
    <tr>
        <td>workspace</td>
        <td>输入</td>
        <td>在 Device 侧申请的 workspace 内存地址。</td>
    </tr>
    <tr>
        <td>workspaceSize</td>
        <td>输入</td>
        <td>在 Device 侧申请的 workspace 大小，由第一段接口 <code>aclnnAllGatherAddGetWorkspaceSize</code> 获取。</td>
    </tr>
    <tr>
        <td>executor</td>
        <td>输入</td>
        <td>op 执行器，包含了算子计算流程。</td>
    </tr>
    <tr>
        <td>stream</td>
        <td>输入</td>
        <td>指定执行任务的 Stream。</td>
    </tr>
    </tbody></table>

- **返回值：**

    返回 aclnnStatus 状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

- 确定性计算：
  - `aclnnAllGatherAdd` 默认确定性实现。

- 当前版本仅支持固定规格：
  - `a.shape = [1024, 2048]`
  - `b.shape = [2048, 2048]`
  - `aGathered.shape = [2048, 2048]`
  - `c.shape = [2048, 2048]`

- 当前版本仅支持：
  - `dtype = FLOAT16`
  - `format = ND`
  - `rankSize = 2`
  - `commTurn = 2`

- `aGathered` 与 `c` 为双输出，输出顺序固定为：
  1. `aGathered`
  2. `c`

- 输入 `b` 与输出 `aGathered` 在 shape 上必须完全一致，Add 为逐元素加法，不支持 broadcast。

- 当前版本不支持非连续 Tensor。

- 当前版本不支持输出别名或输出地址重叠。

- 在 2 卡场景下，Gather 后输出按通信域内 rank 顺序组织；轮次切分仅改变单次执行内部的数据分段方式，不改变最终输出的全局 rank 排布语义。当前版本因 `commTurn = 2`，对应两轮处理。

## 调用示例

该接口的调用方式与其他 ACLNN 两段式接口一致：

1. 准备输入 `a`、`b` 和输出 `aGathered`、`c`；
2. 获取通信域名称 `group`；
3. 调用 `aclnnAllGatherAddGetWorkspaceSize` 获取 `workspaceSize` 和 `executor`；
4. 申请 workspace；
5. 调用 `aclnnAllGatherAdd` 执行计算；
6. 同步 Stream，并校验双输出结果。

典型 2 卡场景下，可使用如下方式构造验证数据：

- rank0: `A[0:1024, :]` 使用模式值 `1 + (row % 4)`
- rank1: `A[0:1024, :]` 使用模式值 `101 + (row % 4)`
- `B` 的四段区间分别填入不同常量：
  - `B[0:512, :] = 10`
  - `B[512:1024, :] = 20`
  - `B[1024:1536, :] = 30`
  - `B[1536:2048, :] = 40`

此时可以同时验证：

- `aGathered` 的 rank 顺序是否正确；
- 当前 `commTurn = 2` 场景下两轮处理对应的全局落位区间是否正确；
- `c = aGathered + b` 是否正确；
- 当前 `commTurn = 2` 场景下两轮之间是否存在数据污染或错位。
