# MoeDistributeCombineV3

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas 推理系列产品</term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |

## 功能说明

- 接口功能：当存在TP域通信时，先进行ReduceScatterV通信，再进行AllToAllV通信，最后将接收的数据整合（乘权重再相加）；当不存在TP域通信时，进行AllToAllV通信，最后将接收的数据整合（乘权重再相加）。

    相较于MoeDistributeCombineV2算子，该算子变更如下：
    -   新增`context`入参，存入通信域相关信息；
    -   新增`ccl_buffer_size`入参，指定当前通信域大小；
    -   减少`group_ep`以及`group_tp`通信域名称入参;
    详细说明请参考以下参数说明。
- 计算公式：

    - 不存在TP域通信时：

    $$
    ataOut = AllToAllV(expandX)\\
    xOut = Sum(expertScales * ataOut + expertScales * sharedExpertX)
    $$

    - 存在TP域通信时：

    $$
    rsOut = ReduceScatterV(expandX)\\
    ataOut = AllToAllV(rsOut)\\
    xOut = Sum(expertScales * ataOut + expertScales * sharedExpertX)
    $$

    注意该算子必须与MoeDistributeDispatchV3配套使用，相当于按MoeDistributeDispatchV3算子收集数据的路径原路返还。

## 参数说明

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
   <th>输入/输出/属性</th>
   <th>描述</th>
   <th>数据类型</th>
   <th>数据格式</th>
  </tr>
 </thead>
 <tbody>
  <tr>
   <td>context</td>
   <td>输入</td>
   <td>本卡通信域信息数据。</td>
   <td>FLOAT16、BFLOAT16</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>expandX</td>
   <td>输入</td>
   <td>根据expertIds进行扩展过的token特征。</td>
   <td>FLOAT16、BFLOAT16</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>expertIds</td>
   <td>输入</td>
   <td>每个token的topK个专家索引。</td>
   <td>INT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>assistInfoForCombine</td>
   <td>输入</td>
   <td>表示同一专家收到的token个数，对应MoeDistributeDispatchV3中的assistInfoForCombineOut输出。</td>
   <td>INT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>epSendCounts</td>
   <td>输入</td>
   <td>从EP通信域各卡接收的token数，对应MoeDistributeDispatchV3中的epRecvCounts输出。</td>
   <td>INT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>expertScales</td>
   <td>输入</td>
   <td>每个token的topK个专家的权重。</td>
   <td>FLOAT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>tpSendCountsOptional</td>
   <td>可选输入</td>
   <td>从TP通信域各卡接收的token数，对应MoeDistributeDispatchV3中的tpRecvCounts输出，有TP域通信需传参，无TP域通信传空指针。</td>
   <td>INT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>xActiveMaskOptional</td>
   <td>可选输入</td>
   <td>表示token是否参与通信，可传有效数据或空指针；1D时true需排在false前（例：{true, false, true}非法），2D时token对应K个值全为false则不参与通信；默认所有token参与通信；各卡BS不一致时所有token需有效。</td>
   <td>BOOL</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>activationScaleOptional</td>
   <td>可选输入</td>
   <td>预留参数，当前版本不支持，传空指针即可。</td>
   <td>-</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>weightScaleOptional</td>
   <td>可选输入</td>
   <td>预留参数，当前版本不支持，传空指针即可。</td>
   <td>-</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>groupListOptional</td>
   <td>可选输入</td>
   <td>预留参数，当前版本不支持，传空指针即可。</td>
   <td>-</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>expandScalesOptional</td>
   <td>可选输入</td>
   <td>表示本卡token的权重，对应MoeDistributeDispatchV3中的expandScales输出。</td>
   <td>FLOAT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>sharedExpertXOptional</td>
   <td>可选输入</td>
   <td>表示共享专家计算后的token，数据类型需与expandX一致。</td>
   <td>FLOAT16、BFLOAT16</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>elasticInfoOptional</td>
   <td>可选输入</td>
   <td>EP通信域动态缩容信息。</td>
   <td>INT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>oriXOptional</td>
   <td>可选输入</td>
   <td>表示未经过FFN（Feed-Forward Neural network）的token数据，在使能copyExpert或使能constExpert的场景下需要本输入数据。</td>
   <td>FLOAT16、BFLOAT16</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>constExpertAlpha1Optional</td>
   <td>可选输入</td>
   <td>在使能constExpert的场景下需要输入的计算系数。</td>
   <td>FLOAT16、BFLOAT16</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>constExpertAlpha2Optional</td>
   <td>可选输入</td>
   <td>在使能constExpert的场景下需要输入的计算系数。</td>
   <td>FLOAT16、BFLOAT16</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>constExpertVOptional</td>
   <td>可选输入</td>
   <td>在使能constExpert的场景下需要输入的计算系数。</td>
   <td>FLOAT16、BFLOAT16</td>
   <td>ND</td>
  </tr>
  <tr>
  <tr>
    <td>performanceInfoOptional</td>
    <td>可选输入</td>
    <td>表示本卡等待各卡数据的通信时间，单位为us（微秒）。单次算子调用各卡通信耗时会累加到该Tensor上，算子内部不进行自动清零，因此用户每次启用此Tensor开始记录耗时前需对Tensor清零。</td>
    <td>INT64</td>
    <td>ND</td>
  </tr>
  <tr>
   <td>epWorldSize</td>
   <td>属性</td>
   <td>EP通信域大小。</td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>epRankId</td>
   <td>属性</td>
   <td>EP域本卡Id，取值范围[0, epWorldSize)，同一个EP通信域中各卡的epRankId不重复。</td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>moeExpertNum</td>
   <td>属性</td>
   <td>MoE专家数量，满足moeExpertNum % (epWorldSize - sharedExpertRankNum) = 0。</td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>cclBufferSize</td>
   <td>属性</td>
   <td><li>当前通信域Buffer大小。</li><li>默认值为""。</li></td>
   <td>STRING</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>tpWorldSize</td>
   <td>可选属性</td>
   <td><li>TP通信域大小。</li><li>默认值为0。</li></td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>tpRankId</td>
   <td>可选属性</td>
   <td><li>TP域本卡Id，无TP域通信时传0即可。</li><li>默认值为0。</li></td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>expertShardType</td>
   <td>可选属性</td>
   <td><li>表示共享专家卡分布类型，当前仅支持传0，表示共享专家卡排在MoE专家卡前面。</li><li>默认值为0。</li></td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>sharedExpertNum</td>
   <td>可选属性</td>
   <td><li>表示共享专家数量（一个共享专家可复制部署到多个卡上）。</li><li>默认值为1。</li></td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>sharedExpertRankNum</td>
   <td>可选属性</td>
   <td><li>表示共享专家卡数量，取值范围[0, epWorldSize)；为0时需满足sharedExpertNum为0或1，不为0时需满足sharedExpertRankNum % sharedExpertNum = 0。</li><li>默认值为0。</li></td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>globalBs</td>
   <td>可选属性</td>
   <td><li>EP域全局的batch size大小；各rank Bs一致时，globalBs = Bs * epWorldSize 或 0；各rank Bs不一致时，globalBs = maxBs * epWorldSize（maxBs为单卡Bs最大值）。</li><li>默认值为0。</li></td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>outDtype</td>
   <td>可选属性</td>
   <td><li>用于指定输出x的数据类型，预留参数，当前版本不支持，传0即可。</li><li>默认值为0。</li></td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>commQuantMode</td>
   <td>可选属性</td>
   <td><li>通信量化类型，取值范围0或2；0表示通信不量化，2表示通信int8量化。</li><li>默认值为0。</li></td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>groupListType</td>
   <td>可选属性</td>
   <td><li>groupList格式，预留参数，当前版本不支持，传0即可。</li><li>默认值为0。</li></td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>commAlg</td>
   <td>可选属性</td>
   <td><li>表示通信亲和内存布局算法。</li><li>默认值为""。</li></td>
   <td>STRING</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>zeroExpertNum</td>
   <td>可选属性</td>
   <td><li>零专家数量。</li><li>默认值为0。</li></td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>copyExpertNum</td>
   <td>可选属性</td>
   <td><li>copy专家数量。</li><li>默认值为0。</li></td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>constExpertNum</td>
   <td>可选属性</td>
   <td><li>常量专家数量。</li><li>默认值为0。</li></td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>xOut</td>
   <td>输出</td>
   <td>表示处理后的token，数据类型、数据格式与expandX保持一致。</td>
   <td>FLOAT16、BFLOAT16</td>
   <td>ND</td>
  </tr>
 </tbody>
</table>

- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：
    * 不支持`expandScalesOptional`。
    * 不支持`commAlg`。

## 约束说明

- `MoeDistributeDispatchV3`与`CombineV3`系列算子必须配套使用，具体参考调用示例。

- 在不同产品型号、不同通信算法或不同版本中，`MoeDistributeDispatchV3`的Tensor输出`assistInfoForCombineOut`、`epRecvCountsOut`、`tpRecvCountsOut`、`expandScalesOut`中的元素值可能不同，使用时直接将上述Tensor传给`CombineV3`系列算子对应参数即可，模型其他业务逻辑不应对其存在依赖。

- 调用算子过程中使用的`groupEp`、`epWorldSize`、`moeExpertNum`、`groupTp`、`tpWorldSize`、`expertShardType`、`sharedExpertNum`、`sharedExpertRankNum`、`globalBs`、`commAlg`参数，`HCCL_BUFFSIZE`取值所有卡需保持一致，网络中不同层中也需保持一致，且和`MoeDistributeDispatchV3`算子对应参数也保持一致。

- 参数说明里shape格式说明：
    - `A`：表示本卡可能接收的最大token数量，取值范围如下：
        - 不使能动态缩容场景时：
            - 对于共享专家，要满足`A` = `Bs` * `epWorldSize` * `sharedExpertNum` / `sharedExpertRankNum`。
            - 对于MoE专家，当`globalBs`为0时，要满足`A` >= `Bs` * `epWorldSize` * min(`localExpertNum`, `K`)；当`globalBs`非0时，要满足`A` >= `globalBs` * min(`localExpertNum`, `K`)。
        - 使能动态缩容场景时：
            - 当`globalBs`为0时，`A` >= max(`Bs` * `epWorldSize` * `sharedExpertNum` / `sharedExpertRankNum`, `Bs` * `epWorldSize` * min(`localExpertNum`, `K`))；
            - 当`globalBs`非0时，`A` >= max(`Bs` * `epWorldSize` * `sharedExpertNum` / `sharedExpertRankNum`, `globalBs` * min(`localExpertNum`, `K`))；
    - `K`：表示选取topK个专家，取值范围为0 < `K` ≤ 16同时满足0 < `K` ≤ `moeExpertNum` + `zeroExpertNum` + `copyExpertNum` + `constExpertNum`。
    - `localExpertNum`：表示本卡专家数量。
        - 对于共享专家卡，`localExpertNum` = 1
        - 对于MoE专家卡，`localExpertNum` = `moeExpertNum` / (`epWorldSize` - `sharedExpertRankNum`)，`localExpertNum` > 1时，不支持TP域通信。

- 参数约束：
    - `zeroExpertNum`：取值范围：[0, MAX_INT32)，MAX_INT32 = 2^31 - 1, 合法的零专家的ID的值是[`moeExpertNum`, `moeExpertNum` + `zeroExpertNum`)。
    - `copyExpertNum`：取值范围：[0, MAX_INT32)，MAX_INT32 = 2^31 - 1, 合法的copy专家的ID的值是[`moeExpertNum` + `zeroExpertNum`, `moeExpertNum` + `zeroExpertNum` + `copyExpertNum`)。
    - `constExpertNum`：取值范围：[0, MAX_INT32)，MAX_INT32 = 2^31 - 1, 合法的常量专家的ID的值是[`moeExpertNum` + `zeroExpertNum` + `copyExpertNum`, `moeExpertNum` + `zeroExpertNum` + `copyExpertNum` + `constExpertNum`)。
    - `oriXOptional`：可选择传入有效数据或填空指针，当`copyExpertNum`不为0或`constExpertNum`不为0时必须传入有效输入；当传入有效数据时，要求shape为 (`Bs`, `H`)，数据类型需跟`expandX`保持一致。
    - `constExpertAlpha1Optional`：可选择传入有效数据或填空指针，当`constExpertNum`不为0或`constExpertNum`不为0时必须传入有效输入；当传入有效数据时，要求shape为(`constExpertNum`, )，数据类型需跟`expandX`保持一致。
    - `constExpertAlpha2Optional`：可选择传入有效数据或填空指针，当`constExpertNum`不为0或`constExpertNum`不为0时必须传入有效输入；当传入有效数据时，要求shape为(`constExpertNum`, )，数据类型需跟`expandX`保持一致。
    - `constExpertVOptional`：可选择传入有效数据或填空指针，当`constExpertNum`不为0或`constExpertNum`不为0时必须传入有效输入；当传入有效数据时，要求shape为(`constExpertNum`, `H`)，数据类型需跟`expandX`保持一致。

- 本文公式中的"/"表示整除。
- 通信域使用约束：
    - 一个模型中的`MoeDistributeCombineV3`和`MoeDistributeDispatchV3`仅支持相同EP通信域，且该通信域中不允许有其他算子。
    - 一个模型中的`MoeDistributeCombineV3`和`MoeDistributeDispatchV3`仅支持相同TP通信域或都不支持TP通信域，有TP通信域时该通信域中不允许有其他算子。

- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：
    - 该场景下单卡包含双DIE（简称为“晶粒”或“裸片”），因此参数说明里的“本卡”均表示单DIE。
    - 动态缩容功能不支持在TP并行场景下使能，即仅在 `tpWorldSize` 取值为 1 时生效。
    - 参数说明里shape格式说明：
        - `H`：表示hidden size隐藏层大小，取值范围[1024, 8192]。
        - `Bs`：表示batch sequence size，即本卡最终输出的token数量，取值范围为[1, 512]。
    - 参数约束：
        - `epWorldSize`：取值支持8、16、32、64、128、144、256、288。
        - `moeExpertNum`：取值范围(0, 1024]。
        - `groupTp`：字符串长度范围为[1, 128)，不能和`groupEp`相同。
        - `tpWorldSize`：取值范围[0, 2]，0和1表示无TP域通信，有TP域通信时仅支持2。
        - `tpRankId`：取值范围[0, 1]，同一个TP通信域中各卡的`tpRankId`不重复。无TP域通信时，传0即可。
        - `sharedExpertNum`：当前取值范围[0, 4]。
        - `commQuantMode`：int8量化当且仅当`tpWorldSize` < 2时可使能。
        - `performanceInfoOptional`：预留参数，当前版本不支持，传空指针即可。
    - `HCCL_BUFFSIZE`：调用本算子前需检查`HCCL_BUFFSIZE`环境变量取值是否合理，该环境变量表示单个通信域占用内存大小，单位MB，不配置时默认为200MB。要求 >= 2且满足>= 2 * (`localExpertNum` * `maxBs` * `epWorldSize` * Align512(Align32(2 * `H`) + 64) + (`K` + `sharedExpertNum`) * `maxBs` * Align512(2 * `H`))，`localExpertNum`需使用MoE专家卡的本卡专家数，其中Align512(x) = ((x + 512 - 1) / 512) * 512，Align32(x) = ((x + 32 - 1) / 32) * 32。