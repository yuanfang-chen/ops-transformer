*

# aclnnQuantMatmulAlltoAll

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>昇腾910_95 AI处理器</term>     |    √     |

## 功能说明

- 算子功能：完成量化的Matmul计算、Permute(保证通信后地址连续)和AlltoAll通信的融合，**先计算后通信**。支持K-C[量化模式](../../../docs/context/量化介绍.md)
- 计算公式:
  假设x1的shape为(BS, H1), x2的shape为(H1, H2)
    - K-C量化模式：
      $$
      computeOut = (x1 @ x2 + bias) * x1Scale * x2Scale \\
      permutedOut = computeOut.view(BS, rankSize, H2 / rankSize).permute(1, 0, 2) \\
      output = AlltoAll(permutedOut).view(rankSize * BS, H2 / rankSize)
      $$

## 函数原型

每个算子分为[两段式接口](../../../docs/context/两段式接口.md)，必须先调用 “aclnnQuantMatmulAlltoAllGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnQuantMatmulAlltoAll”接口执行计算。

```cpp
aclnnStatus aclnnQuantMatmulAlltoAllGetWorkspaceSize(
const aclTensor* x1,           
const aclTensor* x2,
const aclTensor* biasOptional,
const aclTensor* x1Scale,
const aclTensor* x2Scale,
const aclTensor* commScaleOptional,
const aclTensor* x1OffsetOptional,
const aclTensor* x2OffsetOptional,
const aclIntArray* alltoAllAxesOptional,
const char* group,
int64_t x1QuantMode,
int64_t x2QuantMode,
int64_t commQuantMode,
int64_t commQuantDtype,
int64_t groupSize,
bool transposeX1,
bool transposeX2,
aclTensor* output,
uint64_t *workspaceSize,
aclOpExecutor **executor);
```

```cpp
aclnnStatus aclnnQuantMatmulAlltoAll(
  void *workspace,
  uint64_t workspaceSize,
  aclOpExecutor *executor,
  aclrtStream stream)
```

## aclnnQuantMatmulAlltoAllGetWorkspaceSize

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
   <td>该输入作为MatMul计算的左矩阵输入</td>
   <td>FLOAT8_E4M3FN、FLOAT8_E5M2</td>
   <td>ND</td>
   <td>2维, shape为(BS, H1)</td>
   <td>x</td>
  </tr>
  <tr>
   <td>x2</td>
   <td>输入</td>
   <td>融合算子的右矩阵输入，对应公式中的x2</td>
   <td>直接作为MatMul计算的右矩阵输入</td>
   <td>FLOAT8_E4M3FN、FLOAT8_E5M2</td>
   <td>ND</td>
   <td>2维，shape为(H1, H2)</td>
   <td>x</td>
  </tr>
  <tr>
   <td>biasOptional</td>
   <td>输入</td>
   <td>可选输入, 阵乘运算后累加的偏置，对应公式中的bias</td>
   <td>传入非空时生效</td>
   <td>K-C量化模式且x1/x2为FLOAT8_E4M3FN/FLOAT8_E5M2时，该参数类型为FLOAT32</td>
   <td>ND</td>
   <td>1维，shape为(H2,)</td>
   <td>x</td>
  </tr>
  <tr>
   <td>x1Scale</td>
   <td>输入</td>
   <td>左矩阵的量化系数</td>
   <td>对应公式中的x1Scale</td>
   <td>K-C量化模式且x1为FLOAT8_E4M3FN/FLOAT8_E5M2时，该参数类型为FLOAT32</td>
   <td>ND</td>
   <td>K-C量化模式下是1维, shape为(BS,)</td>
   <td>x</td>
  </tr>
  <tr>
   <td>x2Scale</td>
   <td>输入</td>
   <td>右矩阵的量化系数</td>
   <td>对应公式中的x2Scale</td>
   <td>K-C量化模式且x2为FLOAT8_E4M3FN/FLOAT8_E5M2时，该参数类型为FLOAT32</td>
   <td>ND</td>
   <td>K-C量化模式下是1维, shape为(H2,)</td>
   <td>x</td>
  </tr>
  <tr>
   <td>commScaleOptional</td>
   <td>输入</td>
   <td>可选输入, 低比特通信的量化系数</td>
   <td>预留参数，暂不支持低比特通信</td>
   <td>-</td>
   <td>-</td>
   <td>-</td>
   <td>-</td>
  </tr>
  <tr>
   <td>x1OffsetOptional</td>
   <td>输入</td>
   <td>可选输入，左矩阵的量化偏置</td>
   <td>预留参数，暂不支持</td>
   <td>-</td>
   <td>-</td>
   <td>-</td>
   <td>-</td>
  <tr>
   <td>x2OffsetOptional</td>
   <td>输入</td>
   <td>可选输入，右矩阵的量化偏置</td>
   <td>预留参数，暂不支持</td>
   <td>-</td>
   <td>-</td>
   <td>-</td>
   <td>-</td>
  <tr>
   <td>alltoAllAxesOptional</td>
   <td>输入</td>
   <td>可选输入，AlltoAll和Pemute数据交换的方向</td>
   <td>支持配置空或者[-1,-2]，传入空时默认按[-1,-2]处理，表示将输入由(BS, H2)转为(BS * rankSize, H2 / rankSize)</td>
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
   <td>x1QuantMode</td>
   <td>输入</td>
   <td>左矩阵的量化方式</td>
   <td>当前仅支持配置为3，表示PerToken</td>
   <td>INT</td>
   <td>-</td>
   <td>-</td>
   <td>-</td>
  </tr>
  <tr>
   <td>x2QuantMode</td>
   <td>输入</td>
   <td>左矩阵的量化方式</td>
   <td>当前仅支持配置为2，表示PerChannel</td>
   <td>INT</td>
   <td>-</td>
   <td>-</td>
   <td>-</td>
  </tr>
  <tr>
   <td>commQuantMode</td>
   <td>输入</td>
   <td>低比特通信的量化方式</td>
   <td>预留参数，当前仅支持配置为0，表示不量化</td>
   <td>INT</td>
   <td>-</td>
   <td>-</td>
   <td>-</td>
  </tr>
  <tr>
   <td>commQuantDtype</td>
   <td>输入</td>
   <td>低比特通信的量化类型</td>
   <td>预留参数，当前仅支持配置为-1, 表示ACL_DT_UNDEFINED</td>
   <td>INT</td>
   <td>-</td>
   <td>-</td>
   <td>-</td>
  </tr>
  <tr>
   <td>groupSize</td>
   <td>输入</td>
   <td>用于Matmul计算三个方向上的量化分组大小</td>
   <td>预留参数，K-C量化模式下仅支持配置为0，取值不生效。groupSize输入由3个方向的groupSizeM，groupSizeN，groupSizeK三个值拼接组成，每个值占16位，共占用int64_t类型groupSize的低48位（groupSize中的高16位的数值无效），计算公式为：groupSize = groupSizeK | groupSizeN << 16 | groupSizeM << 32。</td>
   <td>INT</td>
   <td>-</td>
   <td>-</td>
   <td>-</td>
  </tr>
  <tr>
   <td>transposeX1</td>
   <td>输入</td>
   <td>标识左矩阵是否转置过</td>
   <td>配置为True时左矩阵Shape为(H1, BS)，暂不支持配置为True</td>
   <td>bool</td>
   <td>ND</td>
   <td></td>
   <td></td>
  </tr>
  <tr>
   <td>transposeX2</td>
   <td>输入</td>
   <td>标识右矩阵是否转置过</td>
   <td>配置为True时右矩阵Shape为(H2, H1)</td>
   <td>bool</td>
   <td>ND</td>
   <td></td>
   <td></td>
  </tr>
  <tr>
   <td>output</td>
   <td>输入</td>
   <td>最终的计算结果</td>
   <td></td>
   <td>FLOAT16、BFLOAT16、FLOAT32</td>
   <td>ND</td>
   <td>2维，shape为(BS/rankSize, H2/rankSize)</td>
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

x1QuantMode、x2QuantMode、commQuantMode的枚举值跟[量化模式](../../../docs/context/量化介绍.md)关系如下:
* 0: 不量化
* 1: pertensor
* 2: perchanenl
* 3: pertoken
* 4: pergroup
* 5: perblock
* 6: mx量化
* 7: pertoken动态量化

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

## aclnnQuantMatmulAlltoAll

* **参数说明：**
    * workspace（void*，入参）：在Device侧申请的workspace内存地址。
    * workspaceSize（uint64_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnMoeDistributeCombineV3GetWorkspaceSize获取。
    * executor（aclOpExecutor*，入参）：op执行器，包含了算子计算流程。
    * stream（aclrtStream，入参）：指定执行任务的AscendCL stream流。
* **返回值：**
  返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/context/aclnn返回码.md)。

## 约束说明
* 默认支持确定性计算
* 右矩阵和输出矩阵的H2必须整除rankSize
* H1范围仅支持[1, 65535]
* rankSize仅支持2,4,8,16
* 通算融合算子不支持并发调用，不同的通算融合算子也不支持并发调用。
* 不支持跨超节点通信，只支持超节点内。

## 调用示例

## 实现说明
// 方案文档单独说

## 附录

### 1. 缓存机制

跟随AscendC自动生成的缓存机制。（通算融合当前无缓存）

### 2.归属领域

`aclnnop_ops_infer` `aclnnop_ops_train`

### 3. Pytorch AtenIR

无对标

### 4. AtenIR参数描述

无对标

### 5. HostAPI接口约束

| **功能维度** | **已支持**               | **应支持但未支持** |
| -------------------- | -------------------------------- | -------------------------- |
| 数据类型           | FP16/BF16                      | NA                       |
| 数据格式           | ND                             | NA                       |
| 空Tensor           | 仅支持BS为零的空Tensor     | NA                       |
| 非连续Tensor       | 不支持输入非连续、不支持输出非连续 | NA                       |

​**低性能场景**​：

? NA

​**未支持类型说明**​：

? NA

​**边界值场景说明**​：

1. 当输入数据为nan时，输出也为nan
2. 当计算结果超过数据类型的数据范围时：
   浮点类型计算结果为inf，整形计算结果为会出现反转。

### 6. HostAPI异常处理

以下场景会出现参数校验异常：

1. 传入的x1、x2、out是空指针时。
2. 入参的数据类型和shape不符合数学逻辑。

### 7. 兼容性说明

1. 功能兼容性：无Pytorch/TensorFlow/MindSpore/Onnx 原生接口，新增支持pytorch自定义接口；
2. 平台兼容性：已支持的芯片版本，功能无差异；
3. 接口兼容性：新增接口；
4. 行为兼容性：新增接口；
5. 性能兼容性：新增接口；
6. 资源兼容性：新增接口；
7. 错误处理兼容性：新增接口；

### 8. API代码注释

```cpp
/**
 * @brief 计算全连接（All-to-All）矩阵乘法所需的 workspace 大小。
 * 
 * 该接口用于计算分布式训练中通信和计算所需的 workspace 大小。支持多种数据类型和量化模式。
 *
 * @param[in] x1 左矩阵输入张量，对应公式中的x1，数据类型支持FLOAT8_E4M3FN、FLOAT8_E5M2。
 * @param[in] x2 右矩阵输入张量，对应公式中的x2，数据类型支持FLOAT8_E4M3FN、FLOAT8_E5M2。
 * @param[in] biasOptional 可选输入张量，偏置项，仅在传入非空时生效，数据类型为FLOAT32。
 * @param[in] x1Scale 左矩阵的量化系数，对应公式中的x1Scale，数据类型为FLOAT32。
 * @param[in] x2Scale 右矩阵的量化系数，对应公式中的x2Scale，数据类型为FLOAT32。
 * @param[in] commScaleOptional 可选输入，低比特通信的量化系数，暂不支持。
 * @param[in] x1OffsetOptional 可选输入，左矩阵的量化偏置，暂不支持。
 * @param[in] x2OffsetOptional 可选输入，右矩阵的量化偏置，暂不支持。
 * @param[in] alltoAllAxesOptional 可选输入，AlltoAll和Permute数据交换的方向，支持配置空或[-2,-1]，传入空时默认按[-2,-1]处理。
 * @param[in] group 通信域名，字符串长度要求(0, 128)。
 * @param[in] x1QuantMode 左矩阵的量化方式，当前仅支持配置为3，表示PerToken。
 * @param[in] x2QuantMode 右矩阵的量化方式，当前仅支持配置为2，表示PerChannel。
 * @param[in] commQuantMode 低比特通信的量化方式，预留参数，当前仅支持配置为0，表示不量化。
 * @param[in] commQuantDtype 低比特通信的量化类型，预留参数，当前仅支持配置为-1，表示ACL_DT_UNDEFINED。
 * @param[in] groupSize 用于Matmul计算三个方向上的量化分组大小，预留参数，K-C量化模式下仅支持配置为0，取值不生效。
 * @param[in] transposeX1 标识左矩阵是否转置过，配置为True时左矩阵Shape为(H1, BS)，暂不支持配置为True。
 * @param[in] transposeX2 标识右矩阵是否转置过，配置为True时右矩阵Shape为(H2, H1)。
 * @param[out] output 矩阵乘法的输出结果，数据类型与输入x1保持一致，支持FLOAT16、BFLOAT16。
 * @param[out] workspaceSize 用于存储计算所需的 workspace 大小。
 * @param[out] executor 执行器指针，用于后续的计算执行。
 * 
 * @return aclnnStatus 执行状态，返回 0 表示成功，其他值表示错误。
 */
aclnnStatus aclnnQuantMatmulAlltoAllGetWorkspaceSize(
const aclTensor* x1,           
const aclTensor* x2,
const aclTensor* biasOptional,
const aclTensor* x1Scale,
const aclTensor* x2Scale,
const aclTensor* commScaleOptional,
const aclTensor* x1OffsetOptional,
const aclTensor* x2OffsetOptional,
const aclIntArray* alltoAllAxesOptional,
const char* group,
int64_t x1QuantMode,
int64_t x2QuantMode,
int64_t commQuantMode,
int64_t commQuantDtype,
int64_t groupSize,
bool transposeX1,
bool transposeX2,
aclTensor* output,
uint64_t *workspaceSize,
aclOpExecutor **executor);

/**
 * @brief 执行全连接（All-to-All）矩阵乘法计算。
 * 
 * 该接口用于执行分布式训练中的通信和计算流程，需在调用前检查HCCL_BUFFSIZE环境变量配置是否合理。
 *
 * @param[in] workspace 在Device侧申请的workspace内存地址。
 * @param[in] workspaceSize 在Device侧申请的workspace大小，由第一段接口获取。
 * @param[in] executor op执行器，包含了算子计算流程。
 * @param[in] stream 指定执行任务的AscendCL stream流。
 * 
 * @return aclnnStatus 执行状态，返回 0 表示成功，其他值表示错误。
 */
aclnnStatus aclnnQuantMatmulAlltoAll(
  void *workspace,
  uint64_t workspaceSize,
  aclOpExecutor *executor,
  aclrtStream stream);
```
调用本接口前需检查HCCL_BUFFSIZE环境变量取值是否合理，该环境变量表示单个通信域占用内存大小，单位MB，不配置时默认为200MB。