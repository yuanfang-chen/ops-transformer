# win区dump数据解析脚本

## 脚本支持情况

| 场景                                                         |  是否支持   |
| :----------------------------------------------------------- |:-------:|
| <term>pytorch接口单跑多轮单算子dispatchv2&combinev2算子发生aic error后得到的dump数据</term>                   |    √    |

## 背景说明

- exception dump：算子在出现aic error后，会触发exception dump功能，将算子输入输出tensor dump成bin文件，其中dispatchv2&combinev2算子还额外注册异常处理回调，会将通信中保存状态位的windows内存dump成bin文件。里面包含卡住、aic error等常见问题定位所需数据。

- 在dispatch&combine问题定位中，需要根据多卡数据进行整体分析判断，分析难度大，因此需要开发多卡数据分析工具来快速分析，要求能够根据已有数据，一键式解析数据，按照既定规则分析出可能异常的点，直接打印异常情况描述，部分数据解析完成后需输出csv文件，工具需打印当前正在进行哪一项分析。

## 功能说明

- dump数据解析:对指定的dump数据进行解析，获得该dump数据的输入、输出以及win区数据，判断出每张卡对应的moe专家数、使用核数、执行次数、epworldsize以及0/1标识位，并判断每个核上的这些参数是否一致，并将分析出有异常的点输出,可以用于定位moe专家数、epworldsize输入异常的问题。

- 执行序分析:根据每张卡的dispatchv2&combinev2的执行次数关系，判断该卡挂在dispatchv2/combinev2算子上，可以用于定位卡住、aic error的问题。

- 状态位分析:获取每个卡中每个核的执行位置信息，判断哪些核没有等到状态位，并根据对应的dispatchv2&combinev2的0/1标识位，到对应的0/1状态区找出没等到状态的核里面具体是第几个状态位没有等到，可以用于定位卡住、aic error的问题。

## 调用说明

| 调用方式  | 样例代码                                  | 说明                                                     |
| :--------: | :----------------------------------------: | :-------------------------------------------------------: |
| dump_analysis.sh脚本直调 | bash dump_analysis.sh TARGET_PATH=xxx TOOL_PATH=xxx bs=17 k=1 SOC_VERSION=910_95 SHARE_EXPERT_CARD_COUNT=1 SHARE_EXPERT_NUM=1| 通过对sh脚本进行入参对指定的TARGET_PATH路径下的dump数据进行dummp数据解析。 |

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
   <td>dump数据所在的文件路径，调用pytorch接口的日志落盘位置通过ASCEND_WORK_PATH环境变量控制，通过查看日志可以得到dump数据的落盘位置，未设置该环境变量时,dump数据落盘在当前目录下extra-info/data-dump/路径。</td>
   <td>str</td>
  </tr>
  <tr>
   <td>TOOL_PATH</td>
   <td>输入</td>
   <td>装包路径中cann仓所在的文件路径，如:install_pkg/cann-x.x.x/。</td>
   <td>str</td>
  </tr>
  <tr>
   <td>SOC_VERSION</td>
   <td>输入</td>
   <td>构造dump数据时对应的版本输入，当前仅支持910_93 or 910_95。</td>
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
   <td>构造dump数据时的输入共享专家卡数,未输入时默认值为0, SHARE_EXPERT_CARD_COUNT <= 总卡数。</td>
   <td>INT</td>
  </tr>
  <tr>
    <td>SHARE_EXPERT_NUM</td>
    <td>可选输入</td>
    <td>构造dump数据时的输入共享专家数,未输入时默认值为0, SHARE_EXPERT_NUM > 0。</td>
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
   <td>存放win区中0/1标识区指定的dispatchv2/combinev2的0/1状态区的所有状态位数据，用于查看没等到状态的核里面具体是第几个状态位没有等到。<br>
       如:解析出该卡dispatchv2的状态区储存在0区，则将win区dump数据中的dispatch 0区状态区[0:64 * 1024]的数据存入win_status_list</td>
   <td>csv表</td>
  </tr>
  <tr>                                                 
   <td>win_all_card_rum_num</td>
   <td>输出</td>
   <td>存放每张卡dispatchv2&combinev2的执行次数,用于对比每张卡的dispatchv2&combinev2的执行次数是否一致。<br> 如:d0 dispatch: 1 指第一张卡dispatchv2算子运行了1次</td>
   <td>csv表</td>
  </tr>
  <tr>
   <td>win_analysis_list</td>
   <td>输出</td>
   <td>存放分析出的异常点及异常位置，即把打屏日志中的Warning信息存储在该csv文件中。<br>如：d0卡dispatchv2算子第35个核的第2个状态位没有等到</td>
   <td>csv表</td>
  </tr>
  <tr>
   <td>win_data</td>
   <td>输出</td>
   <td>存放解析出的各卡的moe专家数、使用核数、执行次数、epworldsize以及0/1标识位数据。<br>如:d0_dispatch_moe专家数: 65</td>
   <td>csv表</td>
  </tr>
  <tr>
   <td>win_data_list</td>
   <td>输出</td>
   <td>存放解析出的各卡中每个核的moe专家数、使用核数、执行次数、epworldsize以及0/1标识位数据，<br>如：d0卡dispatchv2的使用核数为72，则将72个核的win区的moe专家数记录为一个长度为72的列表并储存至win_data_list，d0_dispatch_moe专家数:[0:71]。</td>
   <td>csv表</td>
  </tr>
 </tbody>
</table>


