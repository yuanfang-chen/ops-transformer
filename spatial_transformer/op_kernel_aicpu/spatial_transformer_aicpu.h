/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef OPS_NN_INDEX_SPATIAL_TRANSFORMER_AICPU_H_
#define OPS_NN_INDEX_SPATIAL_TRANSFORMER_AICPU_H_
#include "cpu_kernel.h"
#include "utils/kernel_util.h"

namespace aicpu {
class SpatialTransformerCpuKernel : public CpuKernel {
 public:
  SpatialTransformerCpuKernel() = default;
  ~SpatialTransformerCpuKernel() override = default;
  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  KernelStatus GetInputAndCheckValid(const CpuKernelContext &ctx);
  
  template <typename T, typename T1>
  KernelStatus DoCompute4D();
  
  template <typename T, typename T1>
  KernelStatus DoCompute5D();
  
  template <typename T, typename T1>
  KernelStatus DoCompute5D_C1();
  
  template <typename T>
  uint32_t DoCompute(CpuKernelContext &ctx);

 private:
  const Tensor *input_tensor_ = nullptr;
  const Tensor *input_theta_ = nullptr;
  const Tensor *output_tensor_ = nullptr;
  
  Format date_format_;
  
  int32_t input_n_ = 0;
  int32_t input_c_ = 0;
  int32_t input_h_ = 0;
  int32_t input_w_ = 0;
  int32_t input_c1_ = 0;
  int32_t input_c0_ = 0;
  int32_t output_h_ = 0;
  int32_t output_w_ = 0;
  
  DataType input_data_type_ = DT_FLOAT;
  DataType input_theta_type_ = DT_FLOAT;
  DataType output_data_type_ = DT_FLOAT;
  
  std::vector<float> theta_;
  std::vector<int64_t> theta_valid_;
  int32_t stn_ori_channel_ = 0;
};
} // namespace aicpu
#endif