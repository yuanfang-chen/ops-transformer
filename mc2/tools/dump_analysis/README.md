# win区dump数据解析脚本

## 脚本支持情况

| 场景                                                         |  是否支持   |
| :----------------------------------------------------------- |:-------:|
| <term>pytorch接口单跑多轮单算子dispatchv2&combinev2算子发生aic error后得到的dump数据</term>                   |    √    |


## 功能说明
dump数据:调用pytorch接口的日志落盘位置通过ASCEND_WORK_PATH环境变量控制，通过查看日志可以得到dump数据的落盘位置，未设置该环境变量时,dump数据落盘在当前目录下extra-info/data-dump/路径

脚本功能:调用dump_analysis.sh脚本并输入对应的入参对dump数据进行解析，并根据解析出的win区数据判断，异常点,目前支持执行序分析以及状态区分析

详细说明请参考以下参数说明。

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
  </tr>
 </thead>
 <tbody>
  <tr>
   <td>TARGET_PATH</td>
   <td>输入</td>
   <td>dump数据所在的文件路径。</td>
   <td>str</td>
  </tr>
  <tr>
   <td>TOOL_PATH</td>
   <td>输入</td>
   <td>装包路径中cann仓所在的文件路径。</td>
   <td>str</td>
  </tr>
  <tr>
   <td>SOC_VERSION</td>
   <td>输入</td>
   <td>910_93 or 910_95。</td>
   <td>str</td>
  </tr>
  <tr>
   <td>BS</td>
   <td>输入</td>
   <td>构造dump数据时的输入bs, bs>0。</td>
   <td>INT</td>
  </tr>
  <tr>
   <td>K</td>
   <td>输入</td>
   <td>构造dump数据时的输入k, k>0。</td>
   <td>INT</td>
  </tr>
  <tr>
   <td>SHARE_EXPERT_CARD_COUNT</td>
   <td>可选输入</td>
   <td>构造dump数据时的输入共享专家卡数,未输入时默认值为0, SHARE_EXPERT_CARD_COUNT <= 卡数。</td>
   <td>INT</td>
  </tr>
  <tr>
    <td>SHARE_EXPERT_NUM</td>
    <td>可选输入</td>
    <td>构造dump数据时的输入共享专家数,未输入时默认值为0, SHARE_EXPERT_NUM>0。</td>
    <td>INT</td>
  </tr>
  <tr>
   <td>打屏日志</td>
   <td>输出</td>
   <td>Info:分析出的win区数据信息，Warning:分析出的异常点。</td>
   <td>str</td>
  </tr>
  <tr>
   <td>win_status_list</td>
   <td>输出</td>
   <td>存放win区对应的0/1区的状态位数据。</td>
   <td>csv表</td>
  </tr>
  <tr>                                                 
   <td>win_all_card_rum_num</td>
   <td>输出</td>
   <td>存放每张卡dispatchv2&combinev2的执行次数。</td>
   <td>csv表</td>
  </tr>
  <tr>
   <td>win_analysis_list</td>
   <td>输出</td>
   <td>存放分析出的异常点及异常位置</td>
   <td>csv表</td>
  </tr>
  <tr>
   <td>win_data</td>
   <td>输出</td>
   <td>解析出的各卡的win区数据。</td>
   <td>csv表</td>
  </tr>
  <tr>
   <td>win_data_list</td>
   <td>输出</td>
   <td>解析出的各卡中每个核的win区数据。。</td>
   <td>csv表</td>
  </tr>
 </tbody>
</table>

## 调用说明

| 调用方式  | 样例代码                                  | 说明                                                     |
| :--------: | :----------------------------------------: | :-------------------------------------------------------: |
| dump_analysis.sh脚本直调 | bash dump_analysis.sh TARGET_PATH=xxx TOOL_PATH=xxx bs=17 k=1 SOC_VERSION=910_95 SHARE_EXPERT_CARD_COUNT=1 SHARE_EXPERT_NUM=1| 通过对sh脚本进行入参并调用进行dummp数据解析。 |
