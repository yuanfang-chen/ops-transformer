# QuantAllReduce

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>昇腾910_95 AI处理器</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    ×     |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> |    ×     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |
| <term>Atlas 200/300/500 推理产品</term>                      |    ×     |

## 功能说明

算子功能：实现低比特数据的AllReduce通信，并且在通信的过程中对数据进行反量化，输出半精度或者全精度的通信结果。

- **计算公式**：

    $$
    AllGatherData = AllGather(x)
    $$
    $$
    AllGatherScales = AllGather(scales)
    $$
    $$
    output = Reduce(AllGatherScales * AllGatherData)
    $$
    其中的Reduce计算是将来自不同rank的数据进行reduce计算。

## 参数说明

<table style="undefined;table-layout: fixed; width: 1567px"><colgroup>
    <col style="width: 170px">
    <col style="width: 120px">
    <col style="width: 300px">  
    <col style="width: 330px">  
    <col style="width: 212px">  
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
        <th>连续Tensor</th>
    </tr></thead>
    <tbody>
    <tr>
        <td>x</td>
        <td>输入</td>
        <td>公式中的输入x。</td>
        <td><li>不支持空Tensor。</li><li>支持的shape为：(bs, H)或者(b, s, H)。b为batch size，s为sequence length，H为hidden size。</li></td>
        <td>INT8、HIFLOAT8、FLOAT8_E4M3FN、FLOAT8_E5M2</td>
        <td>ND</td>
        <td>2-3</td>
        <td>√</td>
    </tr>
    <tr>
        <td>scales</td>
        <td>输入</td>
        <td>公式中的输入scales。</td>
        <td><li>不支持空Tensor。</li><li>当scales的数据类型为FLOAT8_E8M0时，x的数据类型必须为FLOAT8_E4M3FN、FLOAT8_E5M2，x的shape为(bs, H)或者(b, s, H)，scales的shape必须对应x的shape为(bs, H/64, 2)或者(b, s, H/64, 2)。</li><li>当scales的数据类型为FLOAT时，x的数据类型必须为INT8、HIFLOAT8、FLOAT8_E4M3FN、FLOAT8_E5M2，x的shape为(bs, H)或者(b, s, H)，scales的shape必须对应x的shape为(bs, H/128)或者(b, s, H/128)。</li></td>
        <td>FLOAT、FLOAT8_E8M0</td>
        <td>ND</td>
        <td>2-4</td>
        <td>√</td>
    </tr>
    <tr>
        <td>group</td>
        <td>输入</td>
        <td>通信域标识。</td>
        <td>通信域标识。</td>
        <td>String</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>reduceOp</td>
        <td>输入</td>
        <td>公式中的reduce操作类型。</td>
        <td>当前仅支持"sum"操作。</td>
        <td>String</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>output</td>
        <td>输出</td>
        <td>公式中的输出output。</td>
        <td><li>不支持空Tensor。</li><li>支持的shape为(bs, H)或者(b, s, H)，output的shape与x保持一致。</li></td>
        <td>FLOAT、FLOAT16、BFLOAT16</td>
        <td>ND</td>
        <td>2-3</td>
        <td>√</td>
    </tr>
    </tbody>
</table>




## 约束说明

- 当x的数据类型为FLOAT8_E4M3FN、FLOAT8_E5M2并且scales的数据类型为FLOAT8_E8M0时，输入数据的量化方式为mx量化。
- 当x的数据类型为INT8、HIFLOAT8、FLOAT8_E4M3FN、FLOAT8_E5M2并且scales的数据类型为FLOAT时，输入数据的量化方式为pertoken-pergroup量化（groupSize=128）。
- 只在Ascend910D系列平台使能。
- 不支持空Tensor输入。
- 通信域大小支持2, 4, 8。
- `HCCL_BUFFSIZE`：调用本算子前需检查`HCCL_BUFFSIZE`环境变量取值是否合理，该环境变量表示单个通信域占用内存大小，单位MB，不配置时默认为200MB。要求满足`HCCL_BUFFSIZE`>= 2 * (`xDataSize` + `scalesDataSize`)，`xDataSize`为输入`x`的数据大小，单位MB，`scalesDataSize`为`scales`的数据大小，单位MB。

## 调用说明

| 调用方式  | 样例代码                                  | 说明                                                     |
| :--------: | :----------------------------------------: | :-------------------------------------------------------: |
| aclnn接口 | [test_aclnn_quant_all_reduce.cpp](./examples/test_aclnn_quant_all_reduce.cpp) | 通过[aclnnQuantAllReduce](./docs/aclnnQuantAllReduce.md)接口方式调用quant_all_reduce算子。 |