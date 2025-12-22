*

# aclnnAlltoAllMatmul

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>昇腾910_95 AI处理器</term>     |    √     |

## 功能说明

- 接口功能：完成AlltoAll通信、Permute(保证通信后地址连续)和Matmul计算的融合，**先通信后计算**。
- 计算公式:
  假设x1输入shape为(BS, H)
  $$
  commOut = AlltoAll(x1.view(rankSize, BS/rankSize, H)) \\
  permutedOut = commOut.permute(1, 0, 2).view(BS/rankSize, rankSize*H) \\
  output = permutedOut @ x2 + bias \\
  $$

## 函数原型

每个算子分为[两段式接口](../../../docs/context/两段式接口.md)，必须先调用 “aclnnAlltoAllMatmulGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnAlltoAllMatmul”接口执行计算。

```cpp
aclnnStatus aclnnAlltoAllMatmulGetWorkspaceSize(
  const aclTensor* x1, 
  const aclTensor* x2,
  const aclTensor* biasOptional,
  const aclIntArray* alltoAllAxesOptional,
  const char* group,
  bool transposeX1,
  bool transposeX2,
  aclTensor* output,
  aclTensor* alltoAllOutOptional,
  uint64_t *workspaceSize,
  aclOpExecutor **executor)
```

```cpp
aclnnStatus aclnnAlltoAllMatmul(
  void *workspace,
  uint64_t workspaceSize,
  aclOpExecutor *executor,
  aclrtStream stream)
```

## aclnnAlltoAllMatmulGetWorkspaceSize

### ​**参数说明**​：

<table style="undefined;table-layout: fixed; width: 1392px"> <colgroup>
 <col style="width: 120px">
 <col style="width: 120px">
 <col style="width: 160px">
 <col style="width: 150px">
 <col style="width: 80px">
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
   <th>非连续tensor</th>
  </tr></thead>
 <tbody>
  <tr>
   <td>x1</td>
   <td>输入</td>
   <td>融合算子的左矩阵输入，对应公式中的x1</td>
   <td>该输入进行AlltoAll通信与Permute操作后结果作为MatMul计算的左矩阵输入</td>
   <td>FLOAT16、BFLOAT16</td>
   <td>ND</td>
   <td>2维, shape为(BS, H)</td>
   <td>x</td>
  </tr>
  <tr>
   <td>x2</td>
   <td>输入</td>
   <td>融合算子的右矩阵输入，也是MatMul计算的右矩阵</td>
   <td>直接作为MatMul计算的右矩阵输入</td>
   <td>FLOAT16、BFLOAT16</td>
   <td>ND</td>
   <td>2维，shape为(H*rankSize, N)</td>
   <td>x</td>
  </tr>
  <tr>
   <td>biasOptional</td>
   <td>可选输入</td>
   <td>阵乘运算后累加的偏置，对应公式中的bias。</td>
   <td></td>
   <td>FLOAT16、BFLOAT16、FLOAT32</td>
   <td>ND</td>
   <td>1维，shape为(N)</td>
   <td>x</td>
  </tr>
  <tr>
   <td>alltoAllAxesOptional</td>
   <td>输入</td>
   <td>AlltoAll和Pemute数据交换的方向</td>
   <td>支持配置空或者[-2,-1]，传入空时默认按[-2,-1]处理，表示将输入由(BS, H)转为(BS/rankSize, rankSize*H)</td>
   <td>aclIntArray*(元素类型INT64)</td>
   <td>ND</td>
   <td>1维，shape为(2)</td>
   <td>x</td>
  </tr>
  <tr>
   <td>group</td>
   <td>输入</td>
   <td>通信域名</td>
   <td>字符串长度要求(0, 128)</td>
   <td>STRING</td>
   <td>ND</td>
   <td>1维</td>
   <td>x</td>
  </tr>
  <tr>
   <td>transposeX1</td>
   <td>输入</td>
   <td>标识左矩阵是否转置过</td>
   <td>配置为True时左矩阵Shape为(H, BS)，暂不支持配为True</td>
   <td>bool</td>
   <td>ND</td>
   <td></td>
   <td></td>
  </tr>
  <tr>
   <td>transposeX2</td>
   <td>输入</td>
   <td>标识右矩阵是否转置过</td>
   <td>配置为True时右矩阵Shape为(N, rankSize*H)</td>
   <td>bool</td>
   <td>ND</td>
   <td></td>
   <td></td>
  </tr>
  <tr>
   <td>output</td>
   <td>输入</td>
   <td>最终的计算结果，</td>
   <td>数据类型与输入x1保持一致</td>
   <td>FLOAT16、BFLOAT16</td>
   <td>ND</td>
   <td>2维，shape为(BS/rankSize, N)</td>
   <td>x</td>
  </tr>
  <tr>
   <td>alltoAllOutOptional</td>
   <td>可选输出</td>
   <td>接收AlltoAll和Pemute后的内容</td>
   <td>传入nullptr时表示不输出通信输出</td>
   <td>FLOAT16、BFLOAT16</td>
   <td>ND</td>
   <td>2维，shape为(BS/rankSize, rankSize*H)</td>
   <td>x</td>
  </tr>
  <tr>
   <td>workspaceSize</td>
   <td>输出</td>
   <td>返回需要在Device侧申请的workspace大小。</td>
   <td></td>
   <td>UINT64</td>
   <td>ND</td>
   <td></td>
   <td></td>
  </tr>
  <tr>
   <td>executor</td>
   <td>输出</td>
   <td>返回op执行器，包含了算子的计算流程。</td>
   <td></td>
   <td>aclOpExecutor*</td>
   <td>ND</td>
   <td></td>
   <td></td>
  </tr>
 </tbody></table>

**返回值**

    aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/context/aclnn返回码.md)。  
    第一段接口完成入参校验，出现以下场景时报错：

  <table style="undefined;table-layout: fixed; width: 1030px"><colgroup>
    <col style="width:250px">
    <col style="width:130px">
    <col style="width:650px">
    </colgroup>
    <thead>
     <tr>
       <th>返回值</th>
       <th>错误码</th>
       <th>描述</th>
     </tr>
    </thead>
    <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>输入和输出的必选参数Tensor是空指针。</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_PARAM_INVALID</td>
      <td>161002</td>
      <td>输入和输出的数据类型不在支持的范围内。</td>
    </tr>
      </tbody>
  </table>

## aclnnAlltoAllMatmul

* **参数说明：**
    * workspace（void*，入参）：在Device侧申请的workspace内存地址。
    * workspaceSize（uint64_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnMoeDistributeCombineV3GetWorkspaceSize获取。
    * executor（aclOpExecutor*，入参）：op执行器，包含了算子计算流程。
    * stream（aclrtStream，入参）：指定执行任务的AscendCL stream流。
* **返回值：**
  返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/context/aclnn返回码.md)。

## 约束说明
* 默认支持确定性计算
* 参数说明中shape使用的变量BS必须整除rankSize
* x1、x2、output、alltoAllOutOptional的数据类型必须一致
* 通算融合算子不支持并发调用，不同的通算融合算子也不支持并发调用。
* 不支持跨超节点通信，只支持超节点内。

## 调用示例
