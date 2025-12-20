*

# aclnnMatmulAlltoAll

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>昇腾910_95 AI处理器</term>                             |    ×     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    ×     |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |
| <term>Atlas 200/300/500 推理产品</term>                      |    ×     |

## 功能说明

- 算子功能：完成Matmul计算、Permute(保证通信后地址连续)和AlltoAll通信的融合，**先计算后通信**。
- 计算公式:
  假设x1的shape为(BS, H1), x2的shape为(H1, H2)
  $$
  computeOut = x1 @ x2 + bias \\
  permutedOut = computeOut.view(BS, rankSize, H2/rankSize).permute(1, 0, 2) \\
  output = AlltoAll(permutedOut).view(rankSize*BS, H2/rankSize)
  $$

## 函数原型

每个算子分为[两段式接口](../../../docs/context/两段式接口.md)，必须先调用 “aclnnMatmulAlltoAllGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnMatmulAlltoAll”接口执行计算。

```cpp
aclnnStatus aclnnMatmulAlltoAllGetWorkspaceSize(
  const aclTensor* x1, 
  const aclTensor* x2,
  const aclTensor* biasOptional,
  const aclIntArray* alltoAllAxesOptional,
  const char* group,
  bool transposeX1,
  bool transposeX2,
  aclTensor* output,
  uint64_t *workspaceSize,
  aclOpExecutor **executor)
```

```cpp
aclnnStatus aclnnMatmulAlltoAll(
  void *workspace,
  uint64_t workspaceSize,
  aclOpExecutor *executor,
  aclrtStream stream)
```

## aclnnMatmulAlltoAllGetWorkspaceSize

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
   <td>FLOAT16、BFLOAT16</td>
   <td>ND</td>
   <td>2维, shape为(BS, H1)</td>
   <td>x</td>
  </tr>
  <tr>
   <td>x2</td>
   <td>输入</td>
   <td>融合算子的右矩阵输入，也是MatMul计算的右矩阵</td>
   <td>直接作为MatMul计算的右矩阵输入</td>
   <td>FLOAT16、BFLOAT16</td>
   <td>ND</td>
   <td>2维，shape为(H1, H2)</td>
   <td>x</td>
  </tr>
  <tr>
   <td>biasOptional</td>
   <td>可选输入</td>
   <td>阵乘运算后累加的偏置，对应公式中的bias。</td>
   <td></td>
   <td>FLOAT16、BFLOAT16、FLOAT32</td>
   <td>ND</td>
   <td>1维，shape为(H2)</td>
   <td>x</td>
  </tr>
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
   <td>数据类型与输入x1保持一致</td>
   <td>FLOAT16、BFLOAT16</td>
   <td>ND</td>
   <td>2维，shape为(BS / rankSize, H2 / rankSize)</td>
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

## aclnnMatmulAlltoAll

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
* 仅支持BS为0的空tensor
* H1范围仅支持[1, 65535]
* ranSize仅支持2,4,8,16
* x1、x2、output的数据类型必须一致
* 通算融合算子不支持并发调用，不同的通算融合算子也不支持并发调用。
* 不支持跨超节点通信，只支持超节点内。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考编译与运行样例。

说明：本示例代码调用了部分HCCL集合通信库接口：HcclGetCommName、HcclCommInitAll、HcclCommDestroy, 请参考[ <<HCCL API (C)>>](https://hiascend.com/document/redirect/CannCommunityHcclCppApi)。

- <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>昇腾910_95 AI处理器</term>：
    ```Cpp
    #include <thread>
    #include <iostream>
    #include <string>
    #include <cstring>
    #include <vector>
    #include <acl/acl.h>
    #include <hccl/hccl.h>
    #include "aclnnop/aclnn_matmul_allto_all.h"

    int ndev = 8;

    #define CHECK_RET(cond, return_expr) \
    do {                               \
        if (!(cond)) {                   \
        return_expr;                   \
        }                                \
    } while (0)

    #define LOG_PRINT(message, ...)     \
    do {                              \
        printf(message, ##__VA_ARGS__); \
    } while (0)

    int64_t GetShapeSize(const std::vector<int64_t> &shape) {
        int64_t shapeSize = 1;
        for (auto i: shape) {
            shapeSize *= i;
        }
        return shapeSize;
    }

    template<typename T>
    int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                        aclDataType dataType, aclTensor **tensor) {
        auto size = GetShapeSize(shape) * sizeof(T);
        // 调用aclrtMalloc申请device侧内存
        auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
        // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
        ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
        // 计算连续tensor的strides
        std::vector<int64_t> strides(shape.size(), 1);
        for (int64_t i = shape.size() - 2; i >= 0; i--) {
            strides[i] = shape[i + 1] * strides[i + 1];
        }
        // 调用aclCreateTensor接口创建aclTensor
        *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                                shape.data(), shape.size(), *deviceAddr);
        return 0;
    }

    struct Args {
        uint32_t rankId;
        HcclComm hcclComm;
        aclrtStream stream;
        aclrtContext context;
    };

    int launchOneThreadMatmulAlltoAll(Args &args) {
        int ret;
        ret = aclrtSetCurrentContext(args.context);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetCurrentContext failed. ERROR: %d\n", ret); return ret);
        char hcom_name[128];
        ret = HcclGetCommName(args.hcclComm, hcom_name);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclGetCommName failed. ret = %d \n", ret); return -1);
        LOG_PRINT("[INFO] rank %d hcom: %s stream: %p, context : %p\n", args.rankId, hcom_name, args.stream,
                args.context);

        std::vector<int64_t> x1Shape = {32, 64};
        std::vector<int64_t> x2Shape = {64, 128};
        std::vector<int64_t> biasShape = {128};
        std::vector<int64_t> outShape = {64, 64};
        void *x1DeviceAddr = nullptr;
        void *x2DeviceAddr = nullptr;
        void *biasDeviceAddr = nullptr;
        void *outDeviceAddr = nullptr;
        aclTensor *x1 = nullptr;
        aclTensor *x2 = nullptr;
        aclTensor *bias = nullptr;
        aclTensor *out = nullptr;

        int64_t a2aAxes[2] = {-1, -2};
        aclIntArray* alltoAllAxesOptional = aclCreateIntArray(a2aAxes, static_cast<uint64_t>(2));
        uint64_t workspaceSize = 0;
        aclOpExecutor *executor;
        void *workspaceAddr = nullptr;

        long long x1ShapeSize = GetShapeSize(x1Shape);
        long long x2ShapeSize = GetShapeSize(x2Shape);
        long long biasShapeSize = GetShapeSize(biasShape);
        long long outShapeSize = GetShapeSize(outShape);
        std::vector<int16_t> x1HostData(x1ShapeSize, 1);
        std::vector<int16_t> x2HostData(x2ShapeSize, 1);
        std::vector<int16_t> biasHostData(biasShapeSize, 1);
        std::vector<int16_t> outHostData(outShapeSize, 0);
        // 创建 tensor
        ret = CreateAclTensor(x1HostData, x1Shape, &x1DeviceAddr, aclDataType::ACL_FLOAT16, &x1);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(x2HostData, x2Shape, &x2DeviceAddr, aclDataType::ACL_FLOAT16, &x2);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(biasHostData, biasShape, &biasDeviceAddr, aclDataType::ACL_FLOAT16, &bias);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &out);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        // 调用第一段接口
        ret = aclnnMatmulAlltoAllGetWorkspaceSize(x1, x2, bias, alltoAllAxesOptional, hcom_name, false, false,
                                                out, &workspaceSize, &executor);
        CHECK_RET(ret == ACL_SUCCESS,
                LOG_PRINT("aclnnMatmulAlltoAllGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
        // 根据第一段接口计算出的workspaceSize申请device内存
        if (workspaceSize > 0) {
            ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        }
        // 调用第二段接口
        ret = aclnnMatmulAlltoAll(workspaceAddr, workspaceSize, executor, args.stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMatmulAlltoAll failed. ERROR: %d\n", ret); return ret);
        //（固定写法）同步等待任务执行结束
        ret = aclrtSynchronizeStreamWithTimeout(args.stream, 10000);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
        LOG_PRINT("device%d aclnnMatmulAlltoAll execute success \n", args.rankId);
        // 释放device资源，需要根据具体API的接口定义修改
        if (x1 != nullptr) {
            aclDestroyTensor(x1);
        }
        if (x2 != nullptr) {
            aclDestroyTensor(x2);
        }
        if (bias != nullptr) {
            aclDestroyTensor(bias);
        }
        if (out != nullptr) {
            aclDestroyTensor(out);
        }
        if (x1DeviceAddr != nullptr) {
            aclrtFree(x1DeviceAddr);
        }
        if (x2DeviceAddr != nullptr) {
            aclrtFree(x2DeviceAddr);
        }
        if (biasDeviceAddr != nullptr) {
            aclrtFree(biasDeviceAddr);
        }
        if (outDeviceAddr != nullptr) {
            aclrtFree(outDeviceAddr);
        }
        if (workspaceSize > 0) {
            aclrtFree(workspaceAddr);
        }
        LOG_PRINT("device%d 162 \n", args.rankId);
        aclrtDestroyStream(args.stream);
        LOG_PRINT("device%d 163 \n", args.rankId);
        HcclCommDestroy(args.hcclComm);
        LOG_PRINT("device%d 164 \n", args.rankId);
        aclrtDestroyContext(args.context);
        LOG_PRINT("device%d 165 \n", args.rankId);
        aclrtResetDevice(args.rankId);
        LOG_PRINT("device%d 166 \n", args.rankId);
        return 0;
    }

    int main(int argc, char *argv[])
    {
        // 本样例基于Atlas A2实现，必须在Atlas A2上运行
        int ret;
        int32_t devices[ndev];
        for (int i = 0; i < ndev; i++) {
            devices[i] = i;
        }
        HcclComm comms[128];
        ret = aclInit(nullptr);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
        // 初始化集合通信域
        for (int i = 0; i < ndev; i++) {
            ret = aclrtSetDevice(devices[i]);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
        }
        ret = HcclCommInitAll(ndev, devices, comms);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("HcclCommInitAll failed. ERROR: %d\n", ret); return ret);
        Args args[ndev];
        aclrtStream stream[ndev];
        aclrtContext context[ndev];
        for (uint32_t rankId = 0; rankId < ndev; rankId++) {
            ret = aclrtSetDevice(rankId);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
            ret = aclrtCreateContext(&context[rankId], rankId);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateContext failed. ERROR: %d\n", ret); return ret);
            ret = aclrtCreateStream(&stream[rankId]);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
        }
        // 启动多线程
        std::vector<std::unique_ptr<std::thread>> threads(ndev);
        for (uint32_t rankId = 0; rankId < ndev; rankId++) {
            args[rankId].rankId = rankId;
            args[rankId].hcclComm = comms[rankId];
            args[rankId].stream = stream[rankId];
            args[rankId].context = context[rankId];
            threads[rankId].reset(new(std::nothrow) std::thread(&launchOneThreadMatmulAlltoAll, std::ref(args  [rankId])));
        }
        for (uint32_t rankId = 0; rankId < ndev; rankId++) {
            threads[rankId]->join();
        }
        aclFinalize();
        return 0;
    }
    ```

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
| 空Tensor           | 不支持空Tensor，无现实意义     | NA                       |
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

```php
**
 * @brief 计算全连接（All-to-All）矩阵乘法所需的 workspace 大小。
 * 
 * 该接口用于计算分布式训练中通信和计算所需的 workspace 大小。支持多种数据类型和量化模式。
 *
 * @param[in] x1 左矩阵输入张量
 * @param[in] x2 右矩阵输入张量
 * @param[in] biasOptional 可选输入张量，偏置项
 * @param[in] alltoallAxes all2all做数据交换的方向
 * @param[in] group 通信域标识，用于标识不同的通信组
 * @param[in] transposeX1 是否对 x1 转置
 * @param[in] transposeX2 是否对 x2 转置
 * @param[out] output 矩阵乘法的输出结果
 * @param[out] workspaceSize 用于存储计算所需的 workspace 大小
 * @param[out] executor 执行器指针，用于后续的计算执行
 * 
 * @return aclnnStatus 执行状态，返回 0 表示成功，其他值表示错误
 */
```cpp
aclnnStatus aclnnMatmulAlltoAllGetWorkspaceSize(
  const aclTensor* x1, 
  const aclTensor* x2,
  const aclTensor* biasOptional,
  const aclIntArray* alltoAllAxesOptional,
  const char* group,
  bool transposeX1,
  bool transposeX2,
  aclTensor* output,
  uint64_t *workspaceSize,
  aclOpExecutor **executor)



aclnnStatus aclnnMatmulAlltoAll(
  void *workspace,
  uint64_t workspaceSize,
  aclOpExecutor *executor,
  aclrtStream stream)
```
调用本接口前需检查HCCL_BUFFSIZE环境变量取值是否合理，该环境变量表示单个通信域占用内存大小，单位MB，不配置时默认为200MB。


// axes增加对空输入场景的描述