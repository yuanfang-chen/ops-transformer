/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_aclnn_quant_flash_attention_score.cpp
 * \brief
 */

#include <iostream>
#include <vector>
#include <cmath>
#include "acl/acl.h"
#include "aclnnop/aclnn_flash_attention_score.h"

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
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("mean result[%ld] is: %f\n", i, resultData[i]);
  }
}

int Init(int32_t deviceId, aclrtContext* context, aclrtStream* stream) {
  // 固定写法，AscendCL初始化
  auto ret = aclInit(nullptr);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
  ret = aclrtSetDevice(deviceId);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
  ret = aclrtCreateContext(context, deviceId);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateContext failed. ERROR: %d\n", ret); return ret);
  ret = aclrtSetCurrentContext(*context);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetCurrentContext failed. ERROR: %d\n", ret); return ret);
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
  // 1. （固定写法）device/context/stream初始化，参考AscendCL对外接口列表
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtContext context;
  aclrtStream stream;
  auto ret = Init(deviceId, &context, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. 构造输入与输出，需要根据API的接口自定义构造
  int64_t B = 1;
  int64_t N1 = 40;
  int64_t N2 = 40;
  int64_t S1 = 7200;
  int64_t S2 = 512;
  int64_t D = 128;

  int64_t q_size = B * S1 * N1 * D;
  int64_t kv_size = B * S2 * N2 * D;
  int64_t softmax_size = B * N1 * S1;
  int64_t d_scale_q_size = B * N1 * (S1 + 127) / 128;
  int64_t d_scale_k_size = B * N1 * S2 / 256;
  int64_t d_scale_v_size = B * N1 * S2 / 512;

  std::vector<int64_t> qShape = {B, S1, N1, D};
  std::vector<int64_t> kShape = {B, S2, N2, D};
  std::vector<int64_t> vShape = {B, S2, N2, D};
  std::vector<int64_t> dScaleQShape = {B, N1, (S1 + 127) / 128, 1};
  std::vector<int64_t> dScaleKShape = {B, N2, S2 / 256, 1};
  std::vector<int64_t> dScaleVShape = {B, N2, S2 / 512, 1};
  std::vector<int64_t> pScaleShape = {1};

  std::vector<int64_t> attentionOutShape = {B, S1, N1, D};
  std::vector<int64_t> softmaxMaxShape = {B, N1, S1, 1};
  std::vector<int64_t> softmaxSumShape = {B, N1, S1, 1};

  void* qDeviceAddr = nullptr;
  void* kDeviceAddr = nullptr;
  void* vDeviceAddr = nullptr;
  void* dScaleQDeviceAddr = nullptr;
  void* dScaleKDeviceAddr = nullptr;
  void* dScaleVDeviceAddr = nullptr;
  void* pScaleDeviceAddr = nullptr;
  void* attentionOutDeviceAddr = nullptr;
  void* softmaxMaxDeviceAddr = nullptr;
  void* softmaxSumDeviceAddr = nullptr;

  aclTensor* q = nullptr;
  aclTensor* k = nullptr;
  aclTensor* v = nullptr;
  aclTensor* attenmask = nullptr;
  aclTensor* dScaleQ = nullptr;
  aclTensor* dScaleK = nullptr;
  aclTensor* dScaleV = nullptr;
  aclTensor* pScale = nullptr;
  aclTensor* attentionOut = nullptr;
  aclTensor* softmaxMax = nullptr;
  aclTensor* softmaxSum = nullptr;
  aclTensor* softmaxOut = nullptr;
 
  std::vector<uint8_t> qHostData(q_size, 1.0);
  std::vector<uint8_t> kHostData(kv_size, 1.0);
  std::vector<uint8_t> vHostData(kv_size, 1.0);
  std::vector<float> dScaleQHostData(d_scale_q_size, 1.0);
  std::vector<float> dScaleKHostData(d_scale_k_size, 1.0);
  std::vector<float> dScaleVHostData(d_scale_v_size, 1.0);
  std::vector<float> pScaleHostData(1, 1.0);
  std::vector<float> attentionOutHostData(q_size, 255);
  std::vector<float> softmaxMaxHostData(softmax_size, 3.0);
  std::vector<float> softmaxSumHostData(softmax_size, 3.0);

  ret = CreateAclTensor(qHostData, qShape, &qDeviceAddr, aclDataType::ACL_HIFLOAT8, &q);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(kHostData, kShape, &kDeviceAddr, aclDataType::ACL_HIFLOAT8, &k);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(vHostData, vShape, &vDeviceAddr, aclDataType::ACL_HIFLOAT8, &v);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(dScaleQHostData, dScaleQShape, &dScaleQDeviceAddr, aclDataType::ACL_FLOAT, &dScaleQ);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(dScaleKHostData, dScaleKShape, &dScaleKDeviceAddr, aclDataType::ACL_FLOAT, &dScaleK);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(dScaleVHostData, dScaleVShape, &dScaleVDeviceAddr, aclDataType::ACL_FLOAT, &dScaleV);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(pScaleHostData, pScaleShape, &pScaleDeviceAddr, aclDataType::ACL_FLOAT, &pScale);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(attentionOutHostData, attentionOutShape, &attentionOutDeviceAddr, aclDataType::ACL_BF16, &attentionOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(softmaxMaxHostData, softmaxMaxShape, &softmaxMaxDeviceAddr, aclDataType::ACL_FLOAT, &softmaxMax);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(softmaxSumHostData, softmaxSumShape, &softmaxSumDeviceAddr, aclDataType::ACL_FLOAT, &softmaxSum);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
 
  double scaleValue = 0.088388;
  int64_t preTokens = 65536;
  int64_t nextTokens = 65536;
  int64_t headNum = 40;
  int64_t sparseMode = 0;
  char layOut[5] = {'B', 'S', 'N', 'D', 0};

  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // 调用aclnnQuantFlashAttentionScore第一段接口
    ret = aclnnQuantFlashAttentionScoreGetWorkspaceSize(
            q, k, v, attenmask, dScaleQ, dScaleK, dScaleV, pScale, scaleValue, preTokens, nextTokens, headNum, layOut, sparseMode,
            softmaxMax, softmaxSum, softmaxOut, attentionOut, &workspaceSize, &executor);

  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantFlashAttentionScoreGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  // 调用aclnnQuantFlashAttentionScore第二段接口
  ret = aclnnQuantFlashAttentionScore(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantFlashAttentionScore failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  PrintOutResult(attentionOutShape, &attentionOutDeviceAddr);
  PrintOutResult(softmaxMaxShape, &softmaxMaxDeviceAddr);
  PrintOutResult(softmaxSumShape, &softmaxSumDeviceAddr);

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(q);
  aclDestroyTensor(k);
  aclDestroyTensor(v);
  aclDestroyTensor(attenmask);
  aclDestroyTensor(attentionOut);
  aclDestroyTensor(softmaxMax);
  aclDestroyTensor(softmaxSum);

  // 7. 释放device资源
  aclrtFree(qDeviceAddr);
  aclrtFree(kDeviceAddr);
  aclrtFree(vDeviceAddr);
  aclrtFree(attentionOutDeviceAddr);
  aclrtFree(softmaxMaxDeviceAddr);
  aclrtFree(softmaxSumDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtDestroyContext(context);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}