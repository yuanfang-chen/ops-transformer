# CausalConv1dAdd

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |

## 功能说明

- 算子功能：对序列执行因果一维卷积，沿序列维度使用缓存数据（长度为卷积核宽减1）对各序列头部进行padding，确保输出依赖当前及历史输入；卷积完成后，将当前序列尾部的数据（长度为卷积核宽减1）更新到缓存。

- 本算子支持以下场景：

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

## 参数说明

  <table style="undefined;table-layout: fixed; width: 1576px"><colgroup>
    <col style="width: 170px">
    <col style="width: 170px">
    <col style="width: 312px">
    <col style="width: 213px">
    <col style="width: 100px">
    </colgroup>
    <thead>
      <tr>
        <th>参数名</th>
        <th>输入/输出/属性</th>
        <th>描述</th>
        <th>数据类型</th>
        <th>数据格式</th>
      </tr></thead>
    <tbody>
      <tr>
        <td>x</td>
        <td>输入</td>
        <td>输入序列，对应公式中x。</td>
        <td>FLOAT16、BFLOAT16</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>weight</td>
        <td>输入</td>
        <td>因果1维卷积核，K固定为3，对应公式中w。</td>
        <td>FLOAT16、BFLOAT16</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>convStates</td>
        <td>输入/输出</td>
        <td>缓存状态张量，存储各序列的历史token数据，各序列计算完成后原地更新，对应公式中cacheState。</td>
        <td>FLOAT16、BFLOAT16</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>queryStartLoc</td>
        <td>可选输入</td>
        <td>序列起始位置索引，记录各序列在拼接张量x中的起始位置。queryStartLoc[i]表示第i个序列的起始偏移。</td>
        <td>INT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>cacheIndices</td>
        <td>可选输入</td>
        <td>缓存索引，指定每个序列对应的缓存状态在convStates中的索引。</td>
        <td>INT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>initialStateMode</td>
        <td>可选输入</td>
        <td>初始状态标志，表示各序列是否使用缓存数据：0=零填充，1=使用缓存，2=使用缓存但前K-1个输出置0。</td>
        <td>INT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>bias</td>
        <td>可选输入</td>
        <td>卷积的偏置。</td>
        <td>FLOAT16、BFLOAT16</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>numAcceptedTokens</td>
        <td>可选输入</td>
        <td>decode场景下的投机token个数。</td>
        <td>INT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>activationMode</td>
        <td>属性</td>
        <td>激活函数类型，取值为0、1、2。<br>0：None；<br>1：silu；<br>2：swish。</td>
        <td>INT</td>
        <td>-</td>
      </tr>
      <tr>
        <td>padSlotId</td>
        <td>属性</td>
        <td>用于跳过不需要参与计算的batch，-1表示不跳过。当cacheIndices[i]==padSlotId时跳过该batch。</td>
        <td>INT</td>
        <td>-</td>
      </tr>
      <tr>
        <td>runMode</td>
        <td>属性</td>
        <td>用于判断是prefill场景或decode场景，取值为0、1。<br>0：prefill场景；<br>1：decode场景。</td>
        <td>INT</td>
        <td>-</td>
      </tr>
      <tr>
        <td>residualConnection</td>
        <td>属性</td>
        <td>是否做残差连接，取值为0、1。<br>0：不做残差连接；<br>1：输出y和输入x相加后输出。</td>
        <td>INT</td>
        <td>-</td>
      </tr>
      <tr>
        <td>y</td>
        <td>输出</td>
        <td>输出序列，shape与x一致，对应公式中y。</td>
        <td>FLOAT16、BFLOAT16</td>
        <td>ND</td>
      </tr>
    </tbody>
  </table>

## 约束说明

- 输入值域限制：
  - x支持2维[cu_seq_len, dim]或3维[batch, seqlen, dim]。
  - weight必须是2维[K, dim]，其中K固定为3。
  - convStates必须是3维[..., K-1, dim]，第0维大小不固定。
  - cu_seq_len范围[1, 65536]，dim范围[64, 16384]，batch范围[1, 256]。
  - queryStartLoc是累计偏移量，长度为batch+1，queryStartLoc[i]表示第i个序列的起始偏移。
  - cacheIndices长度为batch，指定每个序列对应的缓存槽索引。
  - initialStateMode长度为batch，每个元素取值0、1或2。
  - padSlotId建议值-1，当cacheIndices[i]==padSlotId时跳过该batch。
  - activationMode建议值0，表示None。
  - runMode建议值0，表示prefill场景。
  - residualConnection建议值1，表示做残差连接。

- 其他限制：
  - x、weight、convStates、bias、y的数据类型必须一致。
  - queryStartLoc、cacheIndices、initialStateMode、numAcceptedTokens的数据类型必须一致。
  - 输入参数支持非连续Tensor。
  - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Ascend 950PR/Ascend 950DT</term>：x/weight/convStates/bias/y数据类型仅支持FLOAT16、BFLOAT16。