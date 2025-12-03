# aclnnMoeTokenUnpermuteWithRoutingMapGrad

[📄 查看源码](https://gitcode.com/cann/ops-transformer/tree/master/moe/moe_token_unpermute_with_routing_map_grad)


## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>昇腾910_95 AI处理器</term>                             |    ×     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品 </term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |
| <term>Atlas 200/300/500 推理产品</term>                      |    ×     |

## 功能说明

- **算子功能**：aclnnMoeTokenUnpermuteWithRoutingMap的反向传播。
- **计算公式**：

  (1) probs非None：
  
  $$
  permutedTokensGrad[outIndex[i]] = unpermutedTokensGrad[permuteTokenId[i]]
  $$
  
  $$
  permutedProbsGrad = permutedTokensGrad * permutedTokensOptional
  $$
  
  $$
  probsGradExpertOrder = \sum_{j=0}^{hidden\_size}(permutedProbsGrad_{i,j})
  $$

    - paddedMode为false时
  
  $$
  probsGradOut = masked\_scatter(routingMapOptional.T,probsGradExpertOrder)
  $$
  
  $$
  permutedProbs = probsOptional.T.masked\_select(routingMapOptional.T)
  $$

  $$
  permutedTokensGradOut = permutedProbs.unsqueeze(-1) * permutedTokensGrad
  $$

    - paddedMode为true时
  
  $$
  probsGradOut[permuteTokenId[i], outIndex[i]/capacity] = probsGradExpertOrder[outIndex[i]]
  $$

  $$
  permutedProbs[outIndex[i]] = probsOptional.view(1)[i]
  $$

  $$
  permutedTokensGradOut = permutedProbs * permutedTokensGrad
  $$

    (2) probs为None：
  $$
  permutedTokensGradOut[outIndex[i]] = unpermutedTokensGrad[permuteTokenId[i]]
  $$

  1. hidden_size指unpermutedTokensGrad的第1维大小。
  2. paddedMode等于true时，每个专家固定能够处理capacity个token。输入routingMapOptional的第1维是experts_num，即专家个数，输入outIndex的第0维是experts_num * capacity，根据这两个维度可以算出capacity。
  3. paddedMode等于false时，每个token固定被topK_num个专家处理。输入unpermutedTokensGrad的第0维是tokens_num，即token的个数，输入outIndex的第0维是tokens_num * capacity，根据这两个维度可以算出topK_num。


## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnMoeTokenUnpermuteWithRoutingMapGradGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnMoeTokenUnpermuteWithRoutingMapGrad”接口执行计算。

* `aclnnStatus aclnnMoeTokenUnpermuteWithRoutingMapGradGetWorkspaceSize(const aclTensor* unpermutedTokensGrad, const aclTensor* outIndex, const aclTensor* permuteTokenId, const aclTensor* routingMapOptional, const aclTensor* permutedTokensOptional, const aclTensor* probsOptional, bool dropAndPad, const aclIntArray* restoreShapeOptional, const aclTensor* permutedTokensGradOut, const aclTensor* probsGradOutOptional, uint64_t* workspaceSize, aclOpExecutor** executor)`
* `aclnnStatus aclnnMoeTokenUnpermuteWithRoutingMapGrad(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)`

## aclnnMoeTokenUnpermuteWithRoutingMapGradGetWorkspaceSize

- **参数说明：**
  
  - unpermutedTokensGrad（aclTensor\*，计算输入）：Device侧的aclTensor。计算公式中的unpermutedTokensGrad，代表正向输出unpermutedTokens的梯度，要求为一个维度为2D的Tensor，shape为（tokens_num，hidden_size），数据类型支持BFLOAT16、FLOAT16、FLOAT32，[数据格式](../../../docs/zh/context/数据格式.md)要求为ND。支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)。
  - outIndex（aclTensor\*，计算输入）：Device侧的aclTensor。计算公式中outIndex，代表输出位置索引。当paddedMode为false时，要求shape为一个1D的（tokens_num \* topK_num，），索引取值范围[0，tokens_num \* topK_num - 1]。当paddedMode为true时，要求shape为一个1D的（experts_num\* capacity，）。索引取值范围[0，experts_num\* capacity- 1]。数据类型支持INT32，[数据格式](../../../docs/zh/context/数据格式.md)要求为ND。支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)。
  - permuteTokenId（aclTensor\*，计算输入）：Device侧的aclTensor。计算公式中的permuteTokenId，代表输入permutedTokens每个位置对应的Token序号。shape与outIndex相同。取值范围[0，tokens_num - 1]。数据类型支持INT32，[数据格式](../../../docs/zh/context/数据格式.md)要求为ND。支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)。
  - routingMapOptional（aclTensor\*，计算输入）：Device侧的aclTensor，可选输入，当输入probsOptional为空指针时不需要此输入，应该传入空指针。计算公式中的routingMapOptional，代表对应位置的Token是否被对应专家处理，要求shape为一个2D的（tokens_num，experts_num），数据类型支持INT8、bool。当数据类型为INT8，取值支持0、1，当数据类型为bool，取值支持true、false，[数据格式](../../../docs/zh/context/数据格式.md)要求为ND。支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)。
  - permutedTokensOptional（aclTensor\*，计算输入）：Device侧的aclTensor，可选输入，当输入probsOptional为空指针时不需要此输入，应该传入空指针。当输入probsOptional为nullptr时不需要此输入。计算公式中的permutedTokensOptional，代表将每个专家选中token聚集在一起的结果，要求为一个维度为2D的Tensor，当paddedMode为false时，shape为（tokens_num \* topK_num，hidden_size），其中topK_num <= 512。当paddedMode为true时，shape为（experts_num\* capacity，hidden_size）。数据类型与unpermutedTokensGrad相同，[数据格式](../../../docs/zh/context/数据格式.md)要求为ND。支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)。
  - probsOptional（aclTensor\*，计算输入）：Device侧的aclTensor，可选输入，当不需要时为空指针。计算公式中的probsOptional，代表对应位置的Token被对应专家处理后的结果在最终结果中的权重，shape与routingMapOptional相同，数据类型与unpermutedTokensGrad相同，[数据格式](../../../docs/zh/context/数据格式.md)要求为ND。支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)。
  - paddedMode（bool，计算输入）：host侧的BOOL。true表示开启paddedMode，false表示关闭paddedMode。开启paddedMode时，每个专家固定能够处理capacity个token，输入outIndex、permuteTokenId的shape为（experts_num\* capacity，）。关闭paddedMode时，每个token固定被topK_num个专家处理，输入outIndex、permuteTokenId的shape为（tokens_num \* topK_num，）。
  - restoreShapeOptional（aclIntArray\*，计算输入）：host侧的aclIntArray。可选输入，当不需要时为空指针。支持的数据类型为INT64，size大小为2。当paddedMode为true后生效，否则不会对其进行操作。当paddedMode为true以后，此为unpermutedTokensGrad的shape。
  - permutedTokensGradOut（aclTensor\*，计算输出）：输入permutedTokens的梯度，要求是一个2D的Tensor，当paddedMode为true时，shape为（tokens_num \* capacity，hidden_size），当paddedMode为false时，shape为（tokens_num \* topK_num，hidden_size）。数据类型与unpermutedTokensGrad相同，[数据格式](../../../docs/zh/context/数据格式.md)要求为ND。不支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)。
  - probsGradOutOptional（aclTensor\*，计算输出）：可选输出，当不需要时为空指针。输入probs的梯度，要求是一个2D的Tensor，shape为（tokens_num，experts_num）。数据类型与unpermutedTokensGrad相同，[数据格式](../../../docs/zh/context/数据格式.md)要求为ND。不支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)。
  - workspaceSize（uint64\_t\*，出参）：返回需要在Device侧申请的workspace大小。
  - executor（aclOpExecutor\*\*，出参）：返回op执行器，包含了算子计算流程。
- **返回值：**
  
  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。
  
  ```
  第一段接口完成入参校验，出现以下场景时报错：
  161001(ACLNN_ERR_PARAM_NULLPTR): 1. 必选输入或输出的Tensor是空指针。
  161002(ACLNN_ERR_PARAM_INVALID): 1. 输入或输出的数据类型不在支持的范围内。
  561002(ACLNN_ERR_INNER_TILING_ERROR): 1. topK_num > 512
                                        2. topK_num大于experts_num
                                        3. capacity大于tokens_num
                                        4. 输入或输出的shape不符合要求
  ```

## aclnnMoeTokenUnpermuteWithRoutingMapGrad

- **参数说明：**
  
  -   workspace（void\*，入参）：在Device侧申请的workspace内存地址。
  -   workspaceSize（uint64\_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnMoeTokenUnpermuteWithRoutingMapGradGetWorkspaceSize获取。
  -   executor（aclOpExecutor\*，入参）：op执行器，包含了算子计算流程。
  -   stream（aclrtStream，入参）：指定执行任务的Stream。
- **返回值：**
  
    返回aclnnStatus状态码，具体参见[aclnn返回码](./common/aclnn返回码.md)。

## 约束说明

topkNum <= 512

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_moe_token_unpermute_with_routing_map_grad.h"
#include <iostream>

#define CHECK_RET(cond, return_expr)                                           \
  do {                                                                         \
    if (!(cond)) {                                                             \
      return_expr;                                                             \
    }                                                                          \
  } while (0)

#define LOG_PRINT(message, ...)                                                \
  do {                                                                         \
    printf(message, ##__VA_ARGS__);                                            \
  } while (0)

int64_t GetShapeSize(const std::vector<int64_t> &shape) {
  int64_t shapeSize = 1;
  for (auto i : shape) {
    shapeSize *= i;
  }
  return shapeSize;
}

void PrintOutResult(std::vector<int64_t> &shape, void **deviceAddr) {
  auto size = GetShapeSize(shape);
  std::vector<float> resultData(size, 0);
  auto ret = aclrtMemcpy(
      resultData.data(), resultData.size() * sizeof(resultData[0]), *deviceAddr,
      size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(
      ret == ACL_SUCCESS,
      LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
      return );
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }
}

int Init(int32_t deviceId, aclrtStream *stream) {
  // 固定写法，资源初始化
  auto ret = aclInit(nullptr);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret);
            return ret);
  ret = aclrtSetDevice(deviceId);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret);
            return ret);
  ret = aclrtCreateStream(stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret);
            return ret);
  return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData,
                    const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  // 调用aclrtMalloc申请device侧内存
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret);
            return ret);
  // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size,
                    ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret);
            return ret);

  // 计算连续tensor的strides
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  // 调用aclCreateTensor接口创建aclTensor
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType,
                            strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
  // 1. （固定写法）device/stream初始化，参考acl API手册
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret);
            return ret);

  // 2. 构造输入与输出，需要根据API的接口自定义构造
  bool paddedMode = false;
  int32_t tokenNum = 1;
  int32_t hiddenSize = 2;
  int32_t expertNum = 2;
  int32_t topK = 2;
  int32_t outTokenNum = tokenNum * topK;
  std::vector<int64_t> permutedTokensShape = {outTokenNum, hiddenSize};
  std::vector<int64_t> unpermutedTokensGradShape = {tokenNum, hiddenSize};
  std::vector<int64_t> probsShape = {tokenNum, expertNum};
  std::vector<int64_t> outIndexShape = {outTokenNum};
  std::vector<int64_t> permuteTokenIdShape = {outTokenNum};
  std::vector<int64_t> routingMapShape = {tokenNum, expertNum};
  std::vector<int64_t> permutedTokensGradShape = {outTokenNum, hiddenSize};
  std::vector<int64_t> probsGradShape = {tokenNum, expertNum};
  void* permutedTokensDeviceAddr = nullptr;
  void* unpermutedTokensGradDeviceAddr = nullptr;
  void* probsDeviceAddr = nullptr;
  void* outIndexDeviceAddr = nullptr;
  void* permuteTokenIdDeviceAddr = nullptr;
  void* routingMapDeviceAddr = nullptr;
  void* permutedTokensGradDeviceAddr = nullptr;
  void* probsGradDeviceAddr = nullptr;

  aclTensor* permutedTokens = nullptr;
  aclTensor* unpermutedTokensGrad = nullptr;
  aclTensor* probs = nullptr;
  aclTensor* outIndex = nullptr;
  aclTensor* permuteTokenId = nullptr;
  aclTensor* routingMap = nullptr;
  aclTensor *permutedTokensGrad = nullptr;
  aclTensor *probsGrad = nullptr;

  std::vector<float> permutedTokensHostData = {1, 1, 1, 1};
  std::vector<float> unpermutedTokensGradHostData = {1, 1};
  std::vector<float> probsHostData = {1, 1};
  std::vector<int> outIndexHostData = {0, 1};
  std::vector<int> permuteTokenIdHostData = {0, 0};
  std::vector<int8_t> routingMapHostData = {1, 1};
  std::vector<float> permutedTokensGradHostData = {0, 0, 0, 0};
  std::vector<float> probsGradHostData = {0, 0};

  ret = CreateAclTensor(unpermutedTokensGradHostData, unpermutedTokensGradShape, &unpermutedTokensGradDeviceAddr, aclDataType::ACL_FLOAT, &unpermutedTokensGrad);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(outIndexHostData, outIndexShape, &outIndexDeviceAddr, aclDataType::ACL_INT32, &outIndex);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(permuteTokenIdHostData, permuteTokenIdShape, &permuteTokenIdDeviceAddr, aclDataType::ACL_INT32, &permuteTokenId);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(routingMapHostData, routingMapShape, &routingMapDeviceAddr, aclDataType::ACL_BOOL, &routingMap);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(permutedTokensHostData, permutedTokensShape, &permutedTokensDeviceAddr, aclDataType::ACL_FLOAT, &permutedTokens);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(probsHostData, probsShape, &probsDeviceAddr, aclDataType::ACL_FLOAT, &probs);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(permutedTokensGradHostData, permutedTokensGradShape, &permutedTokensGradDeviceAddr, aclDataType::ACL_FLOAT, &permutedTokensGrad);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(probsGradHostData, probsGradShape, &probsGradDeviceAddr, aclDataType::ACL_FLOAT, &probsGrad);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  uint64_t workspaceSize = 0;
  aclOpExecutor *executor;

  // 调用aclnnMoeTokenUnpermuteWithRoutingMapGrad第一段接口
  ret = aclnnMoeTokenUnpermuteWithRoutingMapGradGetWorkspaceSize(unpermutedTokensGrad, outIndex, permuteTokenId, routingMap, permutedTokens, probs, paddedMode, nullptr, permutedTokensGrad, probsGrad, &workspaceSize, &executor);
  CHECK_RET(
      ret == ACL_SUCCESS,
      LOG_PRINT("aclnnMoeTokenUnpermuteWithRoutingMapGradGetWorkspaceSize failed. ERROR: %d\n", ret);
      return ret);

  // 根据第一段接口计算出的workspaceSize申请device内存
  void *workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
              return ret);
  }

  // 调用aclnnMoeTokenUnpermuteWithRoutingMapGrad第二段接口
  ret = aclnnMoeTokenUnpermuteWithRoutingMapGrad(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclnnMoeTokenUnpermuteWithRoutingMapGrad failed. ERROR: %d\n", ret);
            return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
            return ret);

  // 5.获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  LOG_PRINT("permutedTokensGrad \n");
  PrintOutResult(permutedTokensGradShape, &permutedTokensGradDeviceAddr);
  LOG_PRINT("probsGrad \n");
  PrintOutResult(probsGradShape, &probsGradDeviceAddr);

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(permutedTokens);
  aclDestroyTensor(unpermutedTokensGrad);
  aclDestroyTensor(outIndex);
  aclDestroyTensor(permuteTokenId);
  aclDestroyTensor(routingMap);
  aclDestroyTensor(probs);
  aclDestroyTensor(permutedTokensGrad);
  aclDestroyTensor(probsGrad);

  // 7. 释放device资源
  aclrtFree(permutedTokensDeviceAddr);
  aclrtFree(unpermutedTokensGradDeviceAddr);
  aclrtFree(probsDeviceAddr);
  aclrtFree(outIndexDeviceAddr);
  aclrtFree(permuteTokenIdDeviceAddr);
  aclrtFree(routingMapDeviceAddr);
  aclrtFree(permutedTokensGradDeviceAddr);
  aclrtFree(probsGradDeviceAddr);

  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
```