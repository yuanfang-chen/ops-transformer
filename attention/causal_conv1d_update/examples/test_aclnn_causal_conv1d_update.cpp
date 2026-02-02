/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_aclnn_causal_conv1d_update.cpp
 * \brief Test example for aclnnCausalConv1dUpdate operator (Decode/Update mode)
 */

#include <iostream>
#include <vector>
#include <cstdint>
#include <cmath>
#include "acl/acl.h"
#include "aclnnop/aclnn_causal_conv1d_update.h"

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

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
  int64_t shapeSize = 1;
  for (auto i : shape) {
    shapeSize *= i;
  }
  return shapeSize;
}

void PrintOutResult(std::vector<int64_t> &shape, void** deviceAddr) {
  auto size = GetShapeSize(shape);
  std::vector<float> resultData(size, 0);
  auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                         *deviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
  for (int64_t i = 0; i < std::min(size, (int64_t)10); i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }
}

int Init(int32_t deviceId, aclrtStream* stream) {
  // 固定写法，AscendCL初始化
  auto ret = aclInit(nullptr);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
  ret = aclrtSetDevice(deviceId);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
  ret = aclrtCreateStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
  return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
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

int main() {
  // 1. （固定写法）device/stream初始化，参考AscendCL对外接口列表
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. 构造输入与输出，需要根据API的接口自定义构造
  // 示例参数：batch_size=4, dim=64, width=4
  // Update模式：每次处理单个token（decode）或少量token
  int64_t batch_size = 4;
  int64_t dim = 64;
  int64_t width = 4;  // 卷积核宽度

  // x: (batch_size, dim) - 单token输入（decode模式）
  std::vector<int64_t> xShape = {batch_size, dim};
  std::vector<float> xHostData(GetShapeSize(xShape), 1.0);

  // weight: (dim, width) - 卷积权重
  std::vector<int64_t> weightShape = {dim, width};
  std::vector<float> weightHostData(GetShapeSize(weightShape), 0.5);

  // bias: (dim,) - 可选偏置
  std::vector<int64_t> biasShape = {dim};
  std::vector<float> biasHostData(GetShapeSize(biasShape), 0.1);

  // conv_state: (batch_size, dim, width-1) - 卷积状态（会被in-place更新）
  std::vector<int64_t> convStateShape = {batch_size, dim, width - 1};
  std::vector<float> convStateHostData(GetShapeSize(convStateShape), 0.2);  // 初始化为非零值模拟历史状态

  // conv_state_indices: (batch_size,) - 状态索引
  std::vector<int64_t> convStateIndicesShape = {batch_size};
  std::vector<int32_t> convStateIndicesHostData = {0, 1, 2, 3};  // 每个batch对应的状态索引

  // y: (batch_size, dim) - 输出
  std::vector<int64_t> yShape = {batch_size, dim};
  std::vector<float> yHostData(GetShapeSize(yShape), 0.0);

  // 创建device侧tensor
  void* xDeviceAddr = nullptr;
  void* weightDeviceAddr = nullptr;
  void* biasDeviceAddr = nullptr;
  void* convStateDeviceAddr = nullptr;
  void* convStateIndicesDeviceAddr = nullptr;
  void* yDeviceAddr = nullptr;

  aclTensor* x = nullptr;
  aclTensor* weight = nullptr;
  aclTensor* bias = nullptr;
  aclTensor* convState = nullptr;
  aclTensor* convStateIndices = nullptr;
  aclTensor* y = nullptr;

  ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT, &x);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_FLOAT, &weight);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(biasHostData, biasShape, &biasDeviceAddr, aclDataType::ACL_FLOAT, &bias);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(convStateHostData, convStateShape, &convStateDeviceAddr, aclDataType::ACL_FLOAT, &convState);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(convStateIndicesHostData, convStateIndicesShape, &convStateIndicesDeviceAddr, aclDataType::ACL_INT32, &convStateIndices);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(yHostData, yShape, &yDeviceAddr, aclDataType::ACL_FLOAT, &y);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 算子参数
  int64_t activationMode = 1;  // 1 = SiLU/Swish, 0 = None
  int64_t padSlotId = -1;       // -1表示无pad slot

  // 3. 调用CANN算子库API
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // 调用aclnnCausalConv1dUpdate第一段接口
  ret = aclnnCausalConv1dUpdateGetWorkspaceSize(
            x, weight, bias, convState, convStateIndices,
            activationMode, padSlotId, y, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCausalConv1dUpdateGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  // 调用aclnnCausalConv1dUpdate第二段接口
  ret = aclnnCausalConv1dUpdate(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCausalConv1dUpdate failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧
  LOG_PRINT("Output y (first 10 elements):\n");
  PrintOutResult(yShape, &yDeviceAddr);
  LOG_PRINT("\nUpdated conv_state (first 10 elements):\n");
  PrintOutResult(convStateShape, &convStateDeviceAddr);

  // 6. 释放aclTensor
  aclDestroyTensor(x);
  aclDestroyTensor(weight);
  aclDestroyTensor(bias);
  aclDestroyTensor(convState);
  aclDestroyTensor(convStateIndices);
  aclDestroyTensor(y);

  // 7. 释放device资源
  aclrtFree(xDeviceAddr);
  aclrtFree(weightDeviceAddr);
  aclrtFree(biasDeviceAddr);
  aclrtFree(convStateDeviceAddr);
  aclrtFree(convStateIndicesDeviceAddr);
  aclrtFree(yDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  LOG_PRINT("\nTest completed successfully!\n");
  return 0;
}
