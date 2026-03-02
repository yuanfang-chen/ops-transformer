/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <vector>
#include <cmath>
#include "acl/acl.h"
#include "aclnnop/aclnn_apply_adamw_v3.h"

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

int Init(int32_t deviceId, aclrtStream* stream) {
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
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int CreateAclScalarTensor(float value, void** deviceAddr, aclTensor** tensor, aclDataType dataType) {
  auto ret = aclrtMalloc(deviceAddr, sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc scalar failed. ERROR: %d\n", ret); return ret);
  ret = aclrtMemcpy(*deviceAddr, sizeof(float), &value, sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy scalar failed. ERROR: %d\n", ret); return ret);

  std::vector<int64_t> shape = {1};
  std::vector<int64_t> strides = {1};
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

void PrintResult(const std::vector<int64_t>& shape, void* deviceAddr, const std::string& name) {
  auto size = GetShapeSize(shape);
  std::vector<float> resultData(size, 0);
  auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(float),
                         deviceAddr, size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
  LOG_PRINT("%s result (first 10 elements):\n", name.c_str());
  int64_t printCount = std::min(size, (int64_t)10);
  for (int64_t i = 0; i < printCount; i++) {
    LOG_PRINT("  %s[%ld] = %f\n", name.c_str(), i, resultData[i]);
  }
}

int main() {
  int32_t deviceId = 0;
  aclrtStream stream = nullptr;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == 0, return ret);

  std::vector<int64_t> shape = {8};
  int64_t elementCount = GetShapeSize(shape);
  aclDataType dataType = ACL_FLOAT;

  std::vector<float> varHost(elementCount, 1.0f);
  std::vector<float> mHost(elementCount, 0.0f);
  std::vector<float> vHost(elementCount, 0.0f);
  std::vector<float> gradHost(elementCount, 0.1f);

  for (int64_t i = 0; i < elementCount; i++) {
    varHost[i] = 1.0f + i * 0.1f;
    gradHost[i] = 0.01f * (i + 1);
  }

  void* varDeviceAddr = nullptr;
  void* mDeviceAddr = nullptr;
  void* vDeviceAddr = nullptr;
  void* gradDeviceAddr = nullptr;
  void* beta1PowerAddr = nullptr;
  void* beta2PowerAddr = nullptr;
  void* lrAddr = nullptr;
  void* weightDecayAddr = nullptr;
  void* beta1Addr = nullptr;
  void* beta2Addr = nullptr;
  void* epsAddr = nullptr;

  aclTensor* varTensor = nullptr;
  aclTensor* mTensor = nullptr;
  aclTensor* vTensor = nullptr;
  aclTensor* gradTensor = nullptr;
  aclTensor* beta1PowerTensor = nullptr;
  aclTensor* beta2PowerTensor = nullptr;
  aclTensor* lrTensor = nullptr;
  aclTensor* weightDecayTensor = nullptr;
  aclTensor* beta1Tensor = nullptr;
  aclTensor* beta2Tensor = nullptr;
  aclTensor* epsTensor = nullptr;

  ret = CreateAclTensor(varHost, shape, &varDeviceAddr, dataType, &varTensor);
  CHECK_RET(ret == 0, LOG_PRINT("Create var tensor failed\n"); return ret);
  ret = CreateAclTensor(mHost, shape, &mDeviceAddr, dataType, &mTensor);
  CHECK_RET(ret == 0, LOG_PRINT("Create m tensor failed\n"); return ret);
  ret = CreateAclTensor(vHost, shape, &vDeviceAddr, dataType, &vTensor);
  CHECK_RET(ret == 0, LOG_PRINT("Create v tensor failed\n"); return ret);
  ret = CreateAclTensor(gradHost, shape, &gradDeviceAddr, dataType, &gradTensor);
  CHECK_RET(ret == 0, LOG_PRINT("Create grad tensor failed\n"); return ret);

  ret = CreateAclScalarTensor(0.9f, &beta1PowerAddr, &beta1PowerTensor, dataType);
  CHECK_RET(ret == 0, LOG_PRINT("Create beta1Power tensor failed\n"); return ret);
  ret = CreateAclScalarTensor(0.999f, &beta2PowerAddr, &beta2PowerTensor, dataType);
  CHECK_RET(ret == 0, LOG_PRINT("Create beta2Power tensor failed\n"); return ret);
  ret = CreateAclScalarTensor(0.001f, &lrAddr, &lrTensor, dataType);
  CHECK_RET(ret == 0, LOG_PRINT("Create lr tensor failed\n"); return ret);
  ret = CreateAclScalarTensor(0.01f, &weightDecayAddr, &weightDecayTensor, dataType);
  CHECK_RET(ret == 0, LOG_PRINT("Create weightDecay tensor failed\n"); return ret);
  ret = CreateAclScalarTensor(0.9f, &beta1Addr, &beta1Tensor, dataType);
  CHECK_RET(ret == 0, LOG_PRINT("Create beta1 tensor failed\n"); return ret);
  ret = CreateAclScalarTensor(0.999f, &beta2Addr, &beta2Tensor, dataType);
  CHECK_RET(ret == 0, LOG_PRINT("Create beta2 tensor failed\n"); return ret);
  ret = CreateAclScalarTensor(1e-8f, &epsAddr, &epsTensor, dataType);
  CHECK_RET(ret == 0, LOG_PRINT("Create eps tensor failed\n"); return ret);

  bool amsgrad = false;
  bool maximize = false;

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;

  ret = aclnnApplyAdamwV3GetWorkspaceSize(varTensor, mTensor, vTensor,
                                           beta1PowerTensor, beta2PowerTensor, lrTensor,
                                           weightDecayTensor, beta1Tensor, beta2Tensor,
                                           epsTensor, gradTensor, nullptr,
                                           amsgrad, maximize,
                                           &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnApplyAdamwV3GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc workspace failed. ERROR: %d\n", ret); return ret);
  }

  ret = aclnnApplyAdamwV3(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnApplyAdamwV3 failed. ERROR: %d\n", ret); return ret);

  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  PrintResult(shape, varDeviceAddr, "var");
  PrintResult(shape, mDeviceAddr, "m");
  PrintResult(shape, vDeviceAddr, "v");

  if (workspaceAddr != nullptr) {
    aclrtFree(workspaceAddr);
  }
  aclrtFree(varDeviceAddr);
  aclrtFree(mDeviceAddr);
  aclrtFree(vDeviceAddr);
  aclrtFree(gradDeviceAddr);
  aclrtFree(beta1PowerAddr);
  aclrtFree(beta2PowerAddr);
  aclrtFree(lrAddr);
  aclrtFree(weightDecayAddr);
  aclrtFree(beta1Addr);
  aclrtFree(beta2Addr);
  aclrtFree(epsAddr);

  aclDestroyTensor(varTensor);
  aclDestroyTensor(mTensor);
  aclDestroyTensor(vTensor);
  aclDestroyTensor(gradTensor);
  aclDestroyTensor(beta1PowerTensor);
  aclDestroyTensor(beta2PowerTensor);
  aclDestroyTensor(lrTensor);
  aclDestroyTensor(weightDecayTensor);
  aclDestroyTensor(beta1Tensor);
  aclDestroyTensor(beta2Tensor);
  aclDestroyTensor(epsTensor);

  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  LOG_PRINT("Test aclnnApplyAdamwV3 success!\n");
  return 0;
}
