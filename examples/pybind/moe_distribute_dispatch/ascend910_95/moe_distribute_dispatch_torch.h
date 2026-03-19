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
 * \file moe_distribute_dispatch_torch.h
 * \brief
 */
#ifndef ASCEND_OPS_MOE_DISTRIBUTE_DISPATCH_TORCH_H
#define ASCEND_OPS_MOE_DISTRIBUTE_DISPATCH_TORCH_H

#include <ATen/ATen.h>
#include <vector>
#include <torch/all.h>

template <typename TensorType>
void *get_first_tensor_address(c10::ScalarType dataType, const TensorType &input, bool allow_empty = false)
{
    return get_first_tensor_address_by_type<TensorType, void>(input, allow_empty);
}