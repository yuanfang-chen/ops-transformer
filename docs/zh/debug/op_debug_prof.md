

# 算子调试调优

## 调试定位（AI Core算子）

算子运行过程中，如果出现算子执行失败、精度异常等问题，可以打印各阶段信息，如Kernel中间结果，进行问题分析和定位。

### 1、Host侧日志获取方式

* **plog获取**

   程序执行结束后，默认可在"$HOME/ascendc/log"下查看，host日志文件存储路径如下：

   ```
   $HOME/ascend/log/debug/plog/plog-pid_*.log
   ```

   开启环境变量ASCEND_SLOG_PRINT_TO_STDOUT可以将log日志直接打屏显示(1:开启打屏，0：关闭打屏)，配置示例如下：

   ```
   export ASCEND_SLOG_PRINT_TO_STDOUT=1
   ```

   日志相关介绍参见[《日志参考》](https://hiascend.com/document/redirect/CannCommunitylogref)，环境变量介绍参见[《环境变量参考》](https://hiascend.com/document/redirect/CannCommunityEnvRef)。

* **aclnn异常错误信息获取**
   
   通过aclGetRecentErrMsg接口（参见[《acl API（C）》](https://hiascend.com/document/redirect/CannCommunityCppApi)）获取aclnn接口调用过程中的异常信息，使用方法如下：

   ```
   printf(aclGetRecentErrMsg());
   ```

   打印错误信息样例如下：

   ```
   [PID:646612] 2026-01-24-11:53:44.671.727 AclNN_Parameter_Error(EZ1001): Expected a proper Tensor but got null for argument addmmTennsor.self.
   ```

### 2、Kernel调试

常见调试方法如下：

* **printf**

  该接口支持打印Scalar类型数据，如整数、字符、布尔型等，详细介绍请参见[《Ascend C API》](https://hiascend.com/document/redirect/CannCommunityAscendCApi)中“算子调测API > printf”。
  
  ```c++
  blockLength_ = tilingData->totalLength / AscendC::GetBlockNum();
  tileNum_ = tilingData->tileNum;
  tileLength_ = blockLength_ / tileNum_ / BUFFER_NUM;
  // 打印当前核计算Block长度
  AscendC::PRINTF("Tiling blockLength is %llu\n", blockLength_);
  ```
* **DumpTensor**

  该接口支持Dump指定Tensor的内容，同时支持打印自定义附加信息，比如当前行号等，详细介绍请参见[《Ascend C API》](https://hiascend.com/document/redirect/CannCommunityAscendCApi)中“算子调测API > DumpTensor”。
  
  ```c++
  AscendC::LocalTensor<T> zLocal = outputQueueZ.DeQue<T>();
  // 打印zLocal Tensor信息
  DumpTensor(zLocal, 0, 128);
  AscendC::DataCopy(outputGMZ[progress * tileLength_], zLocal, tileLength_);
  ```

对于复杂场景的问题定位，比如算子卡死、GM/UB访问越界等场景，可以采取**单步调试**的方式，具体操作请参见[msDebug](https://www.hiascend.com/document/redirect/CannCommunityToolMsdebug)算子调试工具。

## 性能调优


算子开发过程中，如果出现执行精度下降、内存占用异常等问题，可以通过[CANN Simulator](./cann_sim.md)仿真工具分析算子的指令流水情况，从而确定问题根源，并针对性地优化。

约束条件：当前CANN Simulator仅支持Ascend 950PR。针对Atlas A2/A3系列产品，可以借助[msprof](https://www.hiascend.com/document/redirect/CannCommunityToolMsprof)工具查看，具体参考资料[msprof模型调优工具](https://www.hiascend.com/document/detail/zh/mindstudio/82RC1/T&ITools/Profiling/atlasprofiling_16_0110.html#ZH-CN_TOPIC_0000002504160251)。

本章以`AddExample`自定义算子为例，主要介绍仿真工具的使用。通过采集算子上板运行时各项流水指标分析算子Bound场景，了解仿真流水图便于优化算子内部流水。

1. 前提条件

   完成算子开发和编译后，假设采用aclnn接口方式调用，生成的算子可执行文件（test_aclnn_add_example）所在目录为本项目`examples/add_example/examples/build/bin/`。

2. 执行仿真命令，生成仿真数据

   ```
   cannsim record ./test_aclnn_add_example -s Ascend950 --gen-report
   ```

   仿真结果在本项目`examples/add_example/examples/build/bin/cannsim_*`目录，流水相关文件为：

   ```
   trace_core0.json
   ``` 

3. 在Chrome浏览器中输入“chrome://tracing”地址，并将生成的指令流水图文件（trace_core0.json）拖到空白处打开，具体参数介绍参考CANN Simulator中[“仿真结果解析”](./cann_sim.md#仿真结果解析)章节。
