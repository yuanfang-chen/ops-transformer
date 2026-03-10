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
 * \file test_aclnn_mhc_pre.cpp
 * \brief
 */

#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_mhc_pre.h"

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

// 计算Tensor形状对应的总元素数
int64_t GetShapeSize(const std::vector<int64_t>& shape) {
  int64_t size = 1;
  for (int64_t dim : shape) {
    size *= dim;
  }
  return size;
}

// 将Device侧Tensor数据拷贝到Host侧并打印
void PrintTensorData(const std::vector<int64_t>& shape, void* device_addr) {
  int64_t size = GetShapeSize(shape);
  std::vector<float> host_data(size, 0.0f);
  
  // Device -> Host 数据拷贝
  aclError ret = aclrtMemcpy(
      host_data.data(), size * sizeof(float),
      device_addr, size * sizeof(float),
      ACL_MEMCPY_DEVICE_TO_HOST
  );
  CHECK_RET(ret == ACL_SUCCESS, 
            LOG_PRINT("Memcpy device to host failed, error: %d\n", ret); 
            return);

  // 打印前10个元素（示例）
  LOG_PRINT("Tensor data (first 10 elements): ");
  for (int i = 0; i < std::min((int64_t)10, size); ++i) {
    LOG_PRINT("%f ", host_data[i]);
  }
  LOG_PRINT("\n");
}

// 初始化AscendCL环境（Device/Context/Stream）
int InitAcl(int32_t device_id, aclrtContext& context, aclrtStream& stream) {
  // 1. 初始化ACL
  aclError ret = aclInit(nullptr);
  CHECK_RET(ret == ACL_SUCCESS, 
            LOG_PRINT("aclInit failed, error: %d\n", ret); 
            return -1);

  // 2. 设置Device
  ret = aclrtSetDevice(device_id);
  CHECK_RET(ret == ACL_SUCCESS, 
            LOG_PRINT("aclrtSetDevice failed, error: %d\n", ret); 
            return -1);

  // 3. 创建Context
  ret = aclrtCreateContext(&context, device_id);
  CHECK_RET(ret == ACL_SUCCESS, 
            LOG_PRINT("aclrtCreateContext failed, error: %d\n", ret); 
            return -1);

  // 4. 设置当前Context
  ret = aclrtSetCurrentContext(context);
  CHECK_RET(ret == ACL_SUCCESS, 
            LOG_PRINT("aclrtSetCurrentContext failed, error: %d\n", ret); 
            return -1);

  // 5. 创建Stream
  ret = aclrtCreateStream(&stream);
  CHECK_RET(ret == ACL_SUCCESS, 
            LOG_PRINT("aclrtCreateStream failed, error: %d\n", ret); 
            return -1);

  return 0;
}

// 创建Device侧aclTensor（含数据拷贝）
int CreateAclTensor(
    const std::vector<float>& host_data,
    const std::vector<int64_t>& shape,
    void*& device_addr,
    aclTensor*& tensor) {
  // 1. 计算内存大小
  int64_t size = GetShapeSize(shape) * sizeof(float);

  // 2. 申请Device侧内存
  aclError ret = aclrtMalloc(&device_addr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, 
            LOG_PRINT("aclrtMalloc failed, error: %d\n", ret); 
            return -1);

  // 3. Host -> Device 数据拷贝
  ret = aclrtMemcpy(
      device_addr, size,
      host_data.data(), size,
      ACL_MEMCPY_HOST_TO_DEVICE
  );
  CHECK_RET(ret == ACL_SUCCESS, 
            LOG_PRINT("aclrtMemcpy failed, error: %d\n", ret); 
            return -1);

  // 4. 计算Tensor的strides（连续Tensor）
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; --i) {
    strides[i] = strides[i + 1] * shape[i + 1];
  }

  // 5. 创建aclTensor
  tensor = aclCreateTensor(
      shape.data(), shape.size(),
      ACL_FLOAT, strides.data(), 0,
      ACL_FORMAT_ND, shape.data(), shape.size(),
      device_addr
  );
  CHECK_RET(tensor != nullptr, 
            LOG_PRINT("aclCreateTensor failed\n"); 
            return -1);

  return 0;
}

int main() {
  // ========== 1. 初始化环境 ==========
  int32_t device_id = 0;  // 根据实际Device ID调整
  aclrtContext context = nullptr;
  aclrtStream stream = nullptr;
  
  int ret = InitAcl(device_id, context, stream);
  CHECK_RET(ret == 0, 
            LOG_PRINT("InitAcl failed, error: %d\n", ret); 
            return -1);

  // ========== 2. 构造输入/输出参数 ==========
  int B = 1;
  int S = 2048;
  int n = 4;
  int D = 2560;

  std::vector<int64_t> x_shape = {B * S, n, D};
  int64_t x_size = GetShapeSize(x_shape);
  std::vector<float> x_host_data(x_size, 1.0f);  // 初始化输入数据为1.0

  std::vector<int64_t> phi_shape = {n * n + 2 * n, n * D};
  int64_t phi_size = GetShapeSize(phi_shape);
  std::vector<float> phi_host_data(phi_size, 1.0f);  // 初始化输入数据为1.0

  std::vector<int64_t> alpha_shape = {3};
  int64_t alpha_size = GetShapeSize(alpha_shape);
  std::vector<float> alpha_host_data(alpha_size, 1.0f);  // 初始化输入数据为1.0

  std::vector<int64_t> bias_shape = {n * n + 2 * n};
  int64_t bias_size = GetShapeSize(bias_shape);
  std::vector<float> bias_host_data(bias_size, 1.0f);  // 初始化输入数据为1.0

  std::vector<int64_t> gamma_shape = {n, D};
  int64_t gamma_size = GetShapeSize(gamma_shape);
  std::vector<float> gamma_host_data(gamma_size, 1.0f);  // 初始化输入数据为1.0

  // h_in
  std::vector<int64_t> output_hin_shape = {B * S, D};
  void* output_hin_device_addr = nullptr;
  aclTensor* output_hin_tensor = nullptr;
  // h_post
  std::vector<int64_t> output_h_post_shape = {B * S, n};
  void* output_h_post_device_addr = nullptr;
  aclTensor* output_h_post_tensor = nullptr;
  // h_res
  std::vector<int64_t> output_h_res_shape = {B * S, n, n};
  void* output_h_res_device_addr = nullptr;
  aclTensor* output_h_res_tensor = nullptr;
  // inv_rms
  std::vector<int64_t> output_inv_rms_shape = {B * S};
  void* output_inv_rms_device_addr = nullptr;
  aclTensor* output_inv_rms_tensor = nullptr;
  // h_mix
  std::vector<int64_t> output_h_mix_shape = {B * S, n * n + 2 * n};
  void* output_h_mix_device_addr = nullptr;
  aclTensor* output_h_mix_tensor = nullptr;
  // h_pre
  std::vector<int64_t> output_h_pre_shape = {B * S, n};
  void* output_h_pre_device_addr = nullptr;
  aclTensor* output_h_pre_tensor = nullptr;

  // 输入x的Device Tensor
  void* x_device_addr = nullptr;
  aclTensor* x_tensor = nullptr;
  ret = CreateAclTensor(x_host_data, x_shape, x_device_addr, x_tensor);
  CHECK_RET(ret == 0, 
            LOG_PRINT("Create x_tensor failed\n"); 
            return -1);

  // 输入phi的Device Tensor
  void* phi_device_addr = nullptr;
  aclTensor* phi_tensor = nullptr;
  ret = CreateAclTensor(phi_host_data, phi_shape, phi_device_addr, phi_tensor);
  CHECK_RET(ret == 0, 
            LOG_PRINT("Create phi_tensor failed\n"); 
            return -1);

  // 输入alpha的Device Tensor
  void* alpha_device_addr = nullptr;
  aclTensor* alpha_tensor = nullptr;
  ret = CreateAclTensor(alpha_host_data, alpha_shape, alpha_device_addr, alpha_tensor);
  CHECK_RET(ret == 0, 
            LOG_PRINT("Create alpha_tensor failed\n"); 
            return -1);

  // 输入bias的Device Tensor
  void* bias_device_addr = nullptr;
  aclTensor* bias_tensor = nullptr;
  ret = CreateAclTensor(bias_host_data, bias_shape, bias_device_addr, bias_tensor);
  CHECK_RET(ret == 0, 
            LOG_PRINT("Create bias_tensor failed\n"); 
            return -1);

  // 输入gamma的Device Tensor
  void* gamma_device_addr = nullptr;
  aclTensor* gamma_tensor = nullptr;
  ret = CreateAclTensor(gamma_host_data, gamma_shape, gamma_device_addr, gamma_tensor);
  CHECK_RET(ret == 0, 
            LOG_PRINT("Create gamma_tensor failed\n"); 
            return -1);

  // 输出output的Device Tensor（仅申请内存，无初始数据）
  // hin
  ret = aclrtMalloc(&output_hin_device_addr, GetShapeSize(output_hin_shape)*sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, 
            LOG_PRINT("Malloc output_hin failed, error: %d\n", ret); 
            return -1);
  output_hin_tensor = aclCreateTensor(
      output_hin_shape.data(), output_hin_shape.size(),
      ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND,
      output_hin_shape.data(), output_hin_shape.size(),
      output_hin_device_addr
  );

  // h_post
  ret = aclrtMalloc(&output_h_post_device_addr, GetShapeSize(output_h_post_shape)*sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, 
            LOG_PRINT("Malloc output_h_post failed, error: %d\n", ret); 
            return -1);
  output_h_post_tensor = aclCreateTensor(
      output_h_post_shape.data(), output_h_post_shape.size(),
      ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND,
      output_h_post_shape.data(), output_h_post_shape.size(),
      output_h_post_device_addr
  );
    
  // h_res
  ret = aclrtMalloc(&output_h_res_device_addr, GetShapeSize(output_h_res_shape)*sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, 
            LOG_PRINT("Malloc output_res failed, error: %d\n", ret); 
            return -1);
  output_h_res_tensor = aclCreateTensor(
      output_h_res_shape.data(), output_h_res_shape.size(),
      ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND,
      output_h_res_shape.data(), output_h_res_shape.size(),
      output_h_res_device_addr
  );

  // inv_rms
  ret = aclrtMalloc(&output_inv_rms_device_addr, GetShapeSize(output_inv_rms_shape)*sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, 
            LOG_PRINT("Malloc output_inv_rms failed, error: %d\n", ret); 
            return -1);
  output_inv_rms_tensor = aclCreateTensor(
      output_inv_rms_shape.data(), output_inv_rms_shape.size(),
      ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND,
      output_inv_rms_shape.data(), output_inv_rms_shape.size(),
      output_inv_rms_device_addr
  );

  // h_mix
  ret = aclrtMalloc(&output_h_mix_device_addr, GetShapeSize(output_h_mix_shape)*sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, 
            LOG_PRINT("Malloc output_h_mix failed, error: %d\n", ret); 
            return -1);
  output_h_mix_tensor = aclCreateTensor(
      output_h_mix_shape.data(), output_h_mix_shape.size(),
      ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND,
      output_h_mix_shape.data(), output_h_mix_shape.size(),
      output_h_mix_device_addr
  );

  // h_pre
  ret = aclrtMalloc(&output_h_pre_device_addr, GetShapeSize(output_h_pre_shape)*sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, 
            LOG_PRINT("Malloc output_h_pre failed, error: %d\n", ret); 
            return -1);

  output_h_pre_tensor = aclCreateTensor(
      output_h_pre_shape.data(), output_h_pre_shape.size(),
      ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND,
      output_h_pre_shape.data(), output_h_pre_shape.size(),
      output_h_pre_device_addr
  );
  
  // MhcPre算子参数   
  float norm_eps = 1e-6;  // 防除零参数
  float hc_eps = 1e-6;
  int64_t out_flag = 1;   // 输出中间结果

  // ========== 3. 调用第一段接口：获取Workspace大小 ==========
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  
  aclnnStatus aclnn_ret = aclnnMhcPreGetWorkspaceSize(
    x, phi, alpha, bias, gamma,
    out_flag, norm_eps, hc_eps,
    output_hin_tensor, output_h_post_tensor, output_h_res_tensor,
    output_inv_rms_tensor, output_h_mix_tensor, output_h_pre_tensor,
    &workspace_size, &executor);

  CHECK_RET(aclnn_ret == ACLNN_SUCCESS, 
            LOG_PRINT("aclnnMhcPreGetWorkspaceSize failed, error: %d\n", aclnn_ret); 
            return -1);
  
  // ========== 4. 申请Workspace内存 ==========
  void* workspace_addr = nullptr;
  if (workspace_size > 0) {
    ret = aclrtMalloc(&workspace_addr, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, 
              LOG_PRINT("aclrtMalloc workspace failed, error: %d\n", ret); 
              return -1);
  }

  // ========== 5. 调用第二段接口：执行MhcPre计算 ==========
  aclnn_ret = aclnnMhcPre(
      workspace_addr,
      workspace_size,
      executor,
      stream
  );
  CHECK_RET(aclnn_ret == ACLNN_SUCCESS, 
            LOG_PRINT("aclnnMhcPre failed, error: %d\n", aclnn_ret); 
            return -1);

  // ========== 6. 同步Stream并打印结果 ==========
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, 
            LOG_PRINT("aclrtSynchronizeStream failed, error: %d\n", ret); 
            return -1);

  LOG_PRINT("MhcPre compute success!\n");
  LOG_PRINT("Output tensor data: ");

  PrintTensorData(output_hin_shape, output_hin_device_addr);
  PrintTensorData(output_h_post_shape, output_h_post_device_addr);
  PrintTensorData(output_h_res_shape, output_h_res_device_addr);
  PrintTensorData(output_inv_rms_shape, output_inv_rms_device_addr);
  PrintTensorData(output_h_mix_shape, output_h_mix_device_addr);
  PrintTensorData(output_h_pre_shape, output_h_pre_device_addr);

  // ========== 7. 释放资源 ==========
  // 销毁Tensor
  aclDestroyTensor(x_tensor);
  aclDestroyTensor(phi_tensor);
  aclDestroyTensor(alpha_tensor);
  aclDestroyTensor(bias_tensor);
  aclDestroyTensor(gamma_tensor);

  aclDestroyTensor(output_hin_tensor);
  aclDestroyTensor(output_h_post_tensor);
  aclDestroyTensor(output_h_res_tensor);
  aclDestroyTensor(output_inv_rms_tensor);
  aclDestroyTensor(output_h_mix_tensor);
  aclDestroyTensor(output_h_pre_tensor);
  // 释放Device内存

  aclrtFree(x_device_addr);
  aclrtFree(phi_device_addr);
  aclrtFree(alpha_device_addr);
  aclrtFree(bias_device_addr);
  aclrtFree(gamma_device_addr);

  aclrtFree(output_hin_device_addr);
  aclrtFree(output_h_post_device_addr);
  aclrtFree(output_h_res_device_addr);
  aclrtFree(output_inv_rms_device_addr);
  aclrtFree(output_h_mix_device_addr);
  aclrtFree(output_h_pre_device_addr);

  if (workspace_size > 0) {
    aclrtFree(workspace_addr);
  }

  // 销毁Stream/Context，重置Device
  aclrtDestroyStream(stream);
  aclrtDestroyContext(context);
  aclrtResetDevice(device_id);
  aclFinalize();

  LOG_PRINT("All resources released successfully!\n");
  return 0;
}