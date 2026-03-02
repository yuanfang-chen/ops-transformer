/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_aclnn_flash_attention_score_grad.cpp
 * \brief
 */

#include <iostream>
#include <vector>
#include <cstdint>
#include <cmath>
#include "acl/acl.h"
#include "aclnnop/aclnn_flash_attention_score_grad.h"

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

int Init(int32_t deviceId, aclrtStream* stream) {
  // 固定写法，资源初始化
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
  int64_t B = 1;
  int64_t N1 = 40;
  int64_t N2 = 40;
  int64_t S1 = 7200;
  int64_t S2 = 512;
  int64_t D = 128;
  int64_t H1 = N1 * D;
  int64_t H2 = N2 * D;
  int64_t blockNumQ = (S1 + 511)/ 512;
  int64_t blockNumKV = (S2 + 511)/ 512;

  int64_t q_size = B * N1 * S1 * D;
  int64_t kv_size = B * N2 * S2 * D;
  int64_t softmax_size = B * N1 * S1 * 1;
  int64_t scaleSizeQ = B * N1 * blockNumQ * 1;
  int64_t scaleSizeKV = B * N1 * blockNumKV * 1;

  std::vector<int64_t> qShape = {B, S1, N1, D};
  std::vector<int64_t> kShape = {B, S2, N2, D};
  std::vector<int64_t> vShape = {B, S2, N2, D};
  std::vector<int64_t> dxShape = {B, S1, N1, D};
  std::vector<int64_t> attenmaskShape = {S1, S2};
  std::vector<int64_t> softmaxMaxShape = {B, N1, S1, 1};
  std::vector<int64_t> softmaxSumShape = {B, N1, S1, 1};
  std::vector<int64_t> attentionInShape = {B, S1, N1, D};
  std::vector<int64_t> dScaleQShape = {B, N1, blockNumQ, 1};
  std::vector<int64_t> dScaleKShape = {B, N1, blockNumKV, 1};
  std::vector<int64_t> dScaleVShape = {B, N1, blockNumKV, 1};
  std::vector<int64_t> dScaleDyShape = {B, N1, blockNumQ, 1};
  std::vector<int64_t> dsScaleShape = {1};
  std::vector<int64_t> pScaleShape = {1};

  std::vector<int64_t> dqShape = {B, S1, N1, D};
  std::vector<int64_t> dkShape = {B, S2, N2, D};
  std::vector<int64_t> dvShape = {B, S2, N2, D};
  std::vector<int64_t> printShape = {B, S2, 1, D};

  void* qDeviceAddr = nullptr;
  void* kDeviceAddr = nullptr;
  void* vDeviceAddr = nullptr;
  void* dxDeviceAddr = nullptr;
  void* softmaxMaxDeviceAddr = nullptr;
  void* softmaxSumDeviceAddr = nullptr;
  void* attentionInDeviceAddr = nullptr;
  void* dScaleQDeviceAddr = nullptr;
  void* dScaleKDeviceAddr = nullptr;
  void* dScaleVDeviceAddr = nullptr;
  void* dScaleDyDeviceAddr = nullptr;
  void* dsScaleDeviceAddr = nullptr;
  void* pScaleDeviceAddr = nullptr;
  void* dqDeviceAddr = nullptr;
  void* dkDeviceAddr = nullptr;
  void* dvDeviceAddr = nullptr;

  aclTensor* q = nullptr;
  aclTensor* k = nullptr;
  aclTensor* v = nullptr;
  aclTensor* dx = nullptr;
  aclTensor* attenmask = nullptr;
  aclTensor* softmaxMax = nullptr;
  aclTensor* softmaxSum = nullptr;
  aclTensor* attentionIn = nullptr;
  aclTensor* dScaleQ = nullptr;
  aclTensor* dScaleK = nullptr;
  aclTensor* dScaleV = nullptr;
  aclTensor* dScaleDy = nullptr;
  aclTensor* dsScale = nullptr;
  aclTensor* pScale = nullptr;
  aclTensor* dq = nullptr;
  aclTensor* dk = nullptr;
  aclTensor* dv = nullptr;

  std::vector<uint8_t> qHostData(q_size, 1);
  std::vector<uint8_t> kHostData(kv_size, 1);
  std::vector<uint8_t> vHostData(kv_size, 1);
  std::vector<uint8_t> dxHostData(q_size, 1);
  std::vector<float> softmaxMaxHostData(softmax_size, 3.0);
  std::vector<float> softmaxSumHostData(softmax_size, 3.0);
  std::vector<float> attentionInHostData(q_size, 1.0);
  std::vector<float> dScaleQHostData(scaleSizeQ, 1.0);
  std::vector<float> dScaleKHostData(scaleSizeKV, 1.0);
  std::vector<float> dScaleVHostData(scaleSizeKV, 1.0);
  std::vector<float> dScaleDyHostData(scaleSizeQ, 1.0);
  std::vector<float> dsScaleHostData(1, 1.0);
  std::vector<float> pScaleHostData(1, 1.0);
  std::vector<float> dqHostData(q_size, 0);
  std::vector<float> dkHostData(kv_size, 0);
  std::vector<float> dvHostData(kv_size, 0);

  ret = CreateAclTensor(qHostData, qShape, &qDeviceAddr, aclDataType::ACL_HIFLOAT8, &q);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(kHostData, kShape, &kDeviceAddr, aclDataType::ACL_HIFLOAT8, &k);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(vHostData, vShape, &vDeviceAddr, aclDataType::ACL_HIFLOAT8, &v);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(dxHostData, dxShape, &dxDeviceAddr, aclDataType::ACL_HIFLOAT8, &dx);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(softmaxMaxHostData, softmaxMaxShape, &softmaxMaxDeviceAddr, aclDataType::ACL_FLOAT, &softmaxMax);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(softmaxSumHostData, softmaxSumShape, &softmaxSumDeviceAddr, aclDataType::ACL_FLOAT, &softmaxSum);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(attentionInHostData, attentionInShape, &attentionInDeviceAddr, aclDataType::ACL_BF16, &attentionIn);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(dScaleQHostData, dScaleQShape, &dScaleQDeviceAddr, aclDataType::ACL_FLOAT, &dScaleQ);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(dScaleKHostData, dScaleKShape, &dScaleKDeviceAddr, aclDataType::ACL_FLOAT, &dScaleK);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(dScaleVHostData, dScaleVShape, &dScaleVDeviceAddr, aclDataType::ACL_FLOAT, &dScaleV);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(dScaleDyHostData, dScaleDyShape, &dScaleDyDeviceAddr, aclDataType::ACL_FLOAT, &dScaleDy);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(dsScaleHostData, dsScaleShape, &dsScaleDeviceAddr, aclDataType::ACL_FLOAT, &dsScale);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(pScaleHostData, pScaleShape, &pScaleDeviceAddr, aclDataType::ACL_FLOAT, &pScale);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(dqHostData, dqShape, &dqDeviceAddr, aclDataType::ACL_BF16, &dq);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(dkHostData, dkShape, &dkDeviceAddr, aclDataType::ACL_BF16, &dk);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(dvHostData, dvShape, &dvDeviceAddr, aclDataType::ACL_BF16, &dv);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  double scaleValue = 1.0/sqrt(128);
  int64_t preTokens = INT32_MAX;
  int64_t nextTokens = INT32_MAX;
  int64_t headNum = N1;
  int64_t sparseMode = 0;
  char layOut[6] = {'B', 'S', 'N', 'D', 0};
  int64_t outDtype = 1;

  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // 调用aclnnFlashAttentionScoreGradV2第一段接口
  ret = aclnnQuantFlashAttentionScoreGradGetWorkspaceSize(q, k, v, dx,
            attenmask, softmaxMax, softmaxSum, attentionIn, dScaleQ, dScaleK, dScaleV,dScaleDy, dsScale, pScale,
            scaleValue, preTokens, nextTokens, headNum, layOut, sparseMode, outDtype,
            dq, dk, dv, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantFlashAttentionScoreGradGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  // 调用aclnnFlashAttentionScoreGradV2第二段接口
  ret = aclnnQuantFlashAttentionScoreGrad(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFlashAttentionScoreGradV2 failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  PrintOutResult(printShape, &dqDeviceAddr);
  PrintOutResult(printShape, &dkDeviceAddr);
  PrintOutResult(printShape, &dvDeviceAddr);

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(q);
  aclDestroyTensor(k);
  aclDestroyTensor(v);
  aclDestroyTensor(dx);
  aclDestroyTensor(attenmask);
  aclDestroyTensor(softmaxMax);
  aclDestroyTensor(softmaxSum);
  aclDestroyTensor(attentionIn);
  aclDestroyTensor(dScaleQ);
  aclDestroyTensor(dScaleK);
  aclDestroyTensor(dScaleV);
  aclDestroyTensor(dScaleDy);
  aclDestroyTensor(dsScale);
  aclDestroyTensor(pScale);
  aclDestroyTensor(dq);
  aclDestroyTensor(dk);
  aclDestroyTensor(dv);


  // 7. 释放device资源
  aclrtFree(qDeviceAddr);
  aclrtFree(kDeviceAddr);
  aclrtFree(vDeviceAddr);
  aclrtFree(dxDeviceAddr);
  aclrtFree(softmaxMaxDeviceAddr);
  aclrtFree(softmaxSumDeviceAddr);
  aclrtFree(attentionInDeviceAddr);
  aclrtFree(dScaleQDeviceAddr);
  aclrtFree(dScaleKDeviceAddr);
  aclrtFree(dScaleVDeviceAddr);
  aclrtFree(dScaleDyDeviceAddr);
  aclrtFree(dsScaleDeviceAddr);
  aclrtFree(pScaleDeviceAddr);
  aclrtFree(dqDeviceAddr);
  aclrtFree(dkDeviceAddr);
  aclrtFree(dvDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
