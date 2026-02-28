# aclnnLightningIndexer

[📄 查看源码](https://gitcode.com/cann/ops-transformer/tree/master/attention/lightning_indexer)

## 产品支持情况
| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A2 推理系列产品</term>   | √  |
|<term>Atlas A3 推理系列产品</term>   | √  |

## 功能说明

-   API功能：`lightning_indexer`基于一系列操作得到每一个token对应的Top-$k$个位置。

-   计算公式：
     $$
     Indices=\text{Top-}k\left\{[1]_{1\times g}@\left[(W@[1]_{1\times S_{k}})\odot\text{ReLU}\left(Q_{index}@K_{index}^T\right)\right]\right\}
     $$
     对于某个token对应的Index Query $Q_{index}\in\R^{g\times d}$，给定上下文Index Key $K_{index}\in\R^{S_{k}\times d},W\in\R^{g\times 1}$，其中$g$为GQA对应的group size，$d$为每一个头的维度，$S_{k}$是上下文的长度。

## 函数原型

```
torch_npu.npu_lightning_indexer(query, key, weights, *, actual_seq_lengths_query=None, actual_seq_lengths_key=None, block_table=None, layout_query="BSND", layout_key="BSND", sparse_count=2048, sparse_mode=3, pre_tokens=2^63-1, next_tokens=2^63-1, return_value=False) -> (Tensor, Tensor)
```

## 参数说明

> [!NOTE]
> - query、key、weights参数维度含义：B（Batch Size）表示输入样本批量大小、S（Sequence Length）表示输入样本序列长度、H（Head Size）表示hidden层的大小、N（Head Num）表示多头数、D（Head Dim）表示hidden层最小的单元尺寸，且满足D=H/N、T表示所有Batch输入样本序列长度的累加和。
> - S1表示query shape中的S，S2表示key shape中的S，T1表示query shape中的T，T2表示key shape中的T，N1表示query shape中的N，N2表示key shape中的N。

-   **query**（`Tensor`）：必选参数，不支持非连续，数据格式支持$ND$，数据类型支持`bfloat16`和`float16`。`layout_query`为BSND时shape为[B,S1,N1,D]，当`layout_query`为TND时shape为[T1,N1,D]，N1仅支持64。
    
-   **key**（`Tensor`）：必选参数，不支持非连续，数据格式支持$ND$，数据类型支持`bfloat16`和`float16`，layout\_key为PA_BSND时shape为[block\_count, block\_size, N2, D]，其中block\_count为PageAttention时block总数，block\_size为一个block的token数，block\_size取值为16的整数倍，最大支持到1024。`layout_key`为BSND时shape为[B, S2, N2, D]，`layout_key`为TND时shape为[T2, N2, D]，N2仅支持1。
    
-   **weights**（`Tensor`）：必选参数，不支持非连续，数据格式支持$ND$，数据类型支持`bfloat16`和`float16`，支持输入shape[B,S1,N1]、[T,N1]。
    
- <strong>*</strong>：必选参数，代表其之前的变量是位置相关的，必须按照顺序输入；之后的变量是可选参数，位置无关，需要使用键值对赋值，不赋值会使用默认值。

-   **actual\_seq\_lengths\_query**（`Tensor`）：可选参数，表示不同Batch中`query`的有效token数，数据类型支持`int32`。如果不指定seqlen可传入None，表示和`query`的shape的S长度相同。
    -   该入参中每个Batch的有效token数不超过`query`中的维度S大小且不小于0。支持长度为B的一维tensor。当`layout_query`为TND时，该入参必须传入，且以该入参元素的数量作为B值，该入参中每个元素的值表示当前batch与之前所有batch的token数总和，即前缀和，因此后一个元素的值必须大于等于前一个元素的值。

-   **actual\_seq\_lengths\_key**（`Tensor`）：可选参数，表示不同Batch中`key`的有效token数，数据类型支持`int32`。如果不指定seqlen可传入None，表示和key的shape的S长度相同。
    -   该参数中每个Batch的有效token数不超过`key/value`中的维度S大小且不小于0。支持长度为B的一维tensor。当`layout_key`为TND或PA_BSND时，该入参必须传入，`layout_key`为TND，该参数中每个元素的值表示当前batch与之前所有batch的token数总和，即前缀和，因此后一个元素的值必须大于等于前一个元素的值。

-   **block\_table**（`Tensor`）：可选参数，表示PageAttention中KV存储使用的block映射表，数据格式支持$ND$，数据类型支持`int32`。
    -   PageAttention场景下，block\_table必须为二维，第一维长度需要等于B，第二维长度不能小于maxBlockNumPerSeq（maxBlockNumPerSeq为每个batch中最大actual\_seq\_lengths\_key对应的block数量）

-   **layout\_query**（`str`）：可选参数，用于标识输入`query`的数据排布格式，当前支持BSND、TND，默认值"BSND"。

-   **layout\_key**（`str`）：可选参数，用于标识输入`key`的数据排布格式，当前支持PA_BSND、BSND、TND，默认值"BSND"，在非PageAttention场景下，该参数值应与**layout\_query**值保持一致。

-   **sparse\_count**（`int`）：可选参数，代表topK阶段需要保留的block数量，支持[1, 2048]，数据类型支持`int32`。

-   **sparse\_mode**（`int`）：可选参数，表示sparse的模式，支持0/3，数据类型支持`int32`。
    
    -   sparse\_mode为0时，代表defaultMask模式。
    -   sparse\_mode为3时，代表rightDownCausal模式的mask，对应以右顶点为划分的下三角场景。

-   **pre\_tokens**（`int`）：可选参数，用于稀疏计算，表示attention需要和前几个Token计算关联。数据类型支持`int64`。仅支持默认值2^63-1。

-   **next\_tokens**（`int`）：可选参数，用于稀疏计算，表示attention需要和后几个Token计算关联。数据类型支持`int64`。仅支持默认值2^63-1。

-   **return\_value**（`bool`）：可选参数，表示是否输出`sparse_values`。True表示输出，但图模式下不支持，False表示不输出；默认值为False。该参数仅在训练且`layout_key`不为PA_BSND场景支持。

## 返回值说明

-   **sparse\_indices**（`Tensor`）：公式中的Indices输出，数据类型支持`int32`,数据格式支持$ND$，当`layout_query`为"BSND"时输出shape为[B, S1, N2, sparse\_count]，当layout\_query为"TND"时输出shape为[T1, N2, sparse\_count]。

-   **sparse\_values**（`Tensor`）：公式中的Indices输出对应的value值，数据类型支持`int32`,数据格式支持$ND$，输出shape与`sparse_indices`保持一致。

## 约束说明

-   该接口支持图模式。
-   参数query中的N支持64，key中的N支持1。
-   参数query中的D和参数key中的D值相等为128。
-   参数query、key、weights的数据类型应保持一致。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```Cpp
/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_incre_flash_attention_v4.cpp
 * \brief
 */
//testci
#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include "securec.h"
#include "acl/acl.h"
#include "aclnnop/aclnn_lightning_indexer.h"
 
using namespace std;

namespace {
 
#define CHECK_RET(cond) ((cond) ? true :(false))
 
#define LOG_PRINT(message, ...)     \
  do {                              \
    (void)printf(message, ##__VA_ARGS__); \
  } while (0)
 
int64_t GetShapeSize(const std::vector<int64_t>& shape) {
  int64_t shapeSize = 1;
  for (auto i : shape) {
    shapeSize *= i;
  }
  return shapeSize;
}
 
int Init(int32_t deviceId, aclrtStream* stream) {
  auto ret = aclInit(nullptr);
  if (!CHECK_RET(ret == ACL_SUCCESS)) {
    LOG_PRINT("aclInit failed. ERROR: %d\n", ret); 
    return ret;
  }
  ret = aclrtSetDevice(deviceId);
  if (!CHECK_RET(ret == ACL_SUCCESS)) {
    LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); 
    return ret;
  }
  ret = aclrtCreateStream(stream);
  if (!CHECK_RET(ret == ACL_SUCCESS)) {
    LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); 
    return ret;
  }
  return 0;
}
 
template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  if (!CHECK_RET(ret == ACL_SUCCESS)) {
    LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); 
    return ret;
  }
  
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  if (!CHECK_RET(ret == ACL_SUCCESS)) { 
    LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); 
    return ret;
  }
 
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }
 
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

struct TensorResources {
    void* queryDeviceAddr = nullptr;
    void* keyDeviceAddr = nullptr;
    void* weightsDeviceAddr = nullptr;
    void* sparseIndicesDeviceAddr = nullptr;
    void* sparseValuesDeviceAddr = nullptr;

    aclTensor* queryTensor = nullptr;
    aclTensor* keyTensor = nullptr;
    aclTensor* weightsTensor = nullptr;
    aclTensor* sparseIndicesTensor = nullptr;
    aclTensor* sparseValuesTensor = nullptr;

};

int InitializeTensors(TensorResources& resources) {
    std::vector<int64_t> queryShape = {1, 2, 1, 128};
    std::vector<int64_t> keyShape = {1, 2, 1, 128};
    std::vector<int64_t> weightsShape = {1, 2, 1};
    std::vector<int64_t> sparseIndicesShape = {1, 2, 1, 2048};
    std::vector<int64_t> sparseValuesShape = {1, 2, 1, 16};

    int64_t queryShapeSize = GetShapeSize(queryShape);
    int64_t keyShapeSize = GetShapeSize(keyShape);
    int64_t weightsShapeSize = GetShapeSize(weightsShape);
    int64_t sparseIndicesShapeSize = GetShapeSize(sparseIndicesShape);
    int64_t sparseValuesShapeSize = GetShapeSize(sparseValuesShape);

    std::vector<float> queryHostData(queryShapeSize, 1);
    std::vector<float> keyHostData(keyShapeSize, 1);
    std::vector<float> weightsHostData(weightsShapeSize, 1);
    std::vector<int32_t> sparseIndicesHostData(sparseIndicesShapeSize, 1);
    std::vector<float> sparseValuesHostData(sparseValuesShapeSize, 1);

    int ret = CreateAclTensor(queryHostData, queryShape, &resources.queryDeviceAddr, 
                              aclDataType::ACL_FLOAT16, &resources.queryTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
      return ret;
    }

    ret = CreateAclTensor(keyHostData, keyShape, &resources.keyDeviceAddr, 
                          aclDataType::ACL_FLOAT16, &resources.keyTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
      return ret;
    }

    ret = CreateAclTensor(weightsHostData, weightsShape, &resources.weightsDeviceAddr, 
                          aclDataType::ACL_FLOAT16, &resources.weightsTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
      return ret;
    }

    ret = CreateAclTensor(sparseIndicesHostData, sparseIndicesShape, &resources.sparseIndicesDeviceAddr, 
                          aclDataType::ACL_INT32, &resources.sparseIndicesTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
      return ret;
    }

    ret = CreateAclTensor(sparseValuesHostData, sparseValuesShape, &resources.sparseValuesDeviceAddr, 
                         aclDataType::ACL_FLOAT16, &resources.sparseValuesTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
      return ret;
    }
    return ACL_SUCCESS;
}

int ExecuteLightningIndexer(TensorResources& resources, aclrtStream stream, 
                              void** workspaceAddr, uint64_t* workspaceSize) {
    int64_t sparseCount = 2048;
    int64_t sparseMode = 3;
    int64_t preTokens = 9223372036854775807;
    int64_t nextTokens = 9223372036854775807;
    bool returnValue = false;
    constexpr const char layerOutStr[] = "BSND";
    constexpr size_t layerOutLen = sizeof(layerOutStr);
    char layoutQuery[layerOutLen];
    char layoutKey[layerOutLen];
    errno_t memcpyRet = memcpy_s(layoutQuery, sizeof(layoutQuery), layerOutStr, layerOutLen);
    if (!CHECK_RET(memcpyRet == 0)) {
        LOG_PRINT("memcpy_s layoutQuery failed. ERROR: %d\n", memcpyRet);
        return -1;
    }
    memcpyRet = memcpy_s(layoutKey, sizeof(layoutKey), layerOutStr, layerOutLen);
    if (!CHECK_RET(memcpyRet == 0)) {
        LOG_PRINT("memcpy_s layoutKey failed. ERROR: %d\n", memcpyRet);
        return -1;
    }
    aclOpExecutor* executor;

    int ret = aclnnLightningIndexerGetWorkspaceSize(resources.queryTensor, resources.keyTensor, resources.weightsTensor, nullptr, nullptr, nullptr,
                                                    layoutQuery, layoutKey, sparseCount, sparseMode, preTokens, nextTokens,returnValue, 
                                                    resources.sparseIndicesTensor, resources.sparseValuesTensor, workspaceSize, &executor);
        
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclnnLightningIndexerGetWorkspaceSize failed. ERROR: %d\n", ret);
        return ret;
    }

    if (*workspaceSize > 0ULL) {
        ret = aclrtMalloc(workspaceAddr, *workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (!CHECK_RET(ret == ACL_SUCCESS)) {
            LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
            return ret;
        }
    }

    ret = aclnnLightningIndexer(*workspaceAddr, *workspaceSize, executor, stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclnnSparseFlashAttention failed. ERROR: %d\n", ret);
        return ret;
    }

    return ACL_SUCCESS;
}

int PrintOutResult(std::vector<int64_t> &shape, void** deviceAddr) {
  auto size = GetShapeSize(shape);
  std::vector<aclFloat16> resultData(size, 0);
  auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                         *deviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
        return ret;
  }
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("mean result[%ld] is: %f\n", i, aclFloat16ToFloat(resultData[i]));
  }
  return ACL_SUCCESS;
}

void CleanupResources(TensorResources& resources, void* workspaceAddr, 
                     aclrtStream stream, int32_t deviceId) {
    if (resources.queryTensor) {
      aclDestroyTensor(resources.queryTensor);
    }
    if (resources.keyTensor) {
      aclDestroyTensor(resources.keyTensor);
    }
    if (resources.weightsTensor) {
      aclDestroyTensor(resources.weightsTensor);
    }
    if (resources.sparseIndicesTensor) {
      aclDestroyTensor(resources.sparseIndicesTensor);
    }
    if (resources.sparseValuesTensor) {
      aclDestroyTensor(resources.sparseValuesTensor);
    }

    if (resources.queryDeviceAddr) {
      aclrtFree(resources.queryDeviceAddr);
    }
    if (resources.keyDeviceAddr) {
      aclrtFree(resources.keyDeviceAddr);
    }
    if (resources.weightsDeviceAddr) {
      aclrtFree(resources.weightsDeviceAddr);
    }
    if (resources.sparseIndicesDeviceAddr) {
      aclrtFree(resources.sparseIndicesDeviceAddr);
    }
    if (resources.sparseValuesDeviceAddr) {
      aclrtFree(resources.sparseValuesDeviceAddr);
    }

    if (workspaceAddr) {
      aclrtFree(workspaceAddr);
    }
    if (stream) {
      aclrtDestroyStream(stream);
    }
    aclrtResetDevice(deviceId);
    aclFinalize();
}

} // namespace
 
int main() {
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    TensorResources resources = {};
    void* workspaceAddr = nullptr;
    uint64_t workspaceSize = 0;
    std::vector<int64_t> sparseIndicesShape = {1, 2, 1, 16};
    std::vector<int64_t> sparseValuesShape = {1, 2, 1, 16};
    int ret = ACL_SUCCESS;

    // 1. Initialize device and stream
    ret = Init(deviceId, &stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("Init acl failed. ERROR: %d\n", ret);
        return ret;
    }

    // 2. Initialize tensors
    ret = InitializeTensors(resources);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        CleanupResources(resources, workspaceAddr, stream, deviceId);
        return ret;
    }

    // 3. Execute the operation
    ret = ExecuteLightningIndexer(resources, stream, &workspaceAddr, &workspaceSize);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        CleanupResources(resources, workspaceAddr, stream, deviceId);
        return ret;
    }

    // 4. Synchronize stream
    ret = aclrtSynchronizeStream(stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
        CleanupResources(resources, workspaceAddr, stream, deviceId);
        return ret;
    }

    // 5. Process results
    PrintOutResult(sparseIndicesShape, &resources.sparseIndicesDeviceAddr);
    PrintOutResult(sparseValuesShape, &resources.sparseValuesDeviceAddr);

    // 6. Cleanup resources
    CleanupResources(resources, workspaceAddr, stream, deviceId);
    return 0;
}
```