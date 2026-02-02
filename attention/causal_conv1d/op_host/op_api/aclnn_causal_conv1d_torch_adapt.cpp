/**
?* This program is free software, you can redistribute it and/or modify.
?* Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// NOTE: This adapter is intentionally guarded to avoid breaking the default CANN build
// (which typically doesn't have PyTorch/torch_npu headers in the include path).

#if defined(__has_include)
#if __has_include(<torch/all.h>) && __has_include(<torch/library.h>) && __has_include("torch_npu/csrc/core/npu/NPUStream.h") && \
    __has_include("torch_npu/csrc/framework/OpCommand.h")
#define OPS_TRANSFORMER_CAUSAL_CONV1D_TORCH_ADAPTER_ENABLED 1
#else
#define OPS_TRANSFORMER_CAUSAL_CONV1D_TORCH_ADAPTER_ENABLED 0
#endif
#else
#define OPS_TRANSFORMER_CAUSAL_CONV1D_TORCH_ADAPTER_ENABLED 0
#endif

#if OPS_TRANSFORMER_CAUSAL_CONV1D_TORCH_ADAPTER_ENABLED

#include <torch/all.h>
#include <torch/library.h>

#include "acl/acl.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

#if __has_include("aclnnop/aclnn_causal_conv1d.h")
#include "aclnnop/aclnn_causal_conv1d.h"
#else
#include "aclnn_causal_conv1d.h"
#endif

namespace op_api {
namespace {

struct AclTensorDeleter final {
    void operator()(aclTensor* tensor) const noexcept
    {
        if (tensor != nullptr) {
            (void)aclDestroyTensor(tensor);
        }
    }
};
using AclTensorPtr = std::unique_ptr<aclTensor, AclTensorDeleter>;

static inline void CheckNpuTensorDefined(const at::Tensor& tensor, const char* name)
{
    TORCH_CHECK(tensor.defined(), name, " must be defined");
    TORCH_CHECK(tensor.device().type() == c10::DeviceType::PrivateUse1, name, " must be on NPU (PrivateUse1)");
}

static inline void CheckContiguous(const at::Tensor& tensor, const char* name)
{
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

static inline aclDataType GetAclDataType(c10::ScalarType scalarType, const char* name)
{
    switch (scalarType) {
        case c10::ScalarType::Half:
            return ACL_FLOAT16;
        case c10::ScalarType::BFloat16:
            return ACL_BF16;
        case c10::ScalarType::Int:
            return ACL_INT32;
        case c10::ScalarType::Bool:
            return ACL_BOOL;
        default:
            TORCH_CHECK(false, name, " has unsupported dtype: ", scalarType);
            return ACL_DT_UNDEFINED;
    }
}

static inline AclTensorPtr WrapTensorAsAclTensor(const at::Tensor& tensor, aclDataType dtype)
{
    std::vector<int64_t> shape(tensor.sizes().begin(), tensor.sizes().end());
    std::vector<int64_t> strides(tensor.strides().begin(), tensor.strides().end());

    aclTensor* aclTensorPtr =
        aclCreateTensor(shape.data(), shape.size(), dtype, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(),
                        shape.size(), const_cast<void*>(tensor.data_ptr()));
    TORCH_CHECK(aclTensorPtr != nullptr, "aclCreateTensor failed");
    return AclTensorPtr(aclTensorPtr);
}

} // namespace

at::Tensor& causal_conv1d_out(const at::Tensor& x,
                              const at::Tensor& weight,
                              const c10::optional<at::Tensor>& biasOptional,
                              at::Tensor& convStates,
                              const at::Tensor& queryStartLoc,
                              const at::Tensor& cacheIndices,
                              const at::Tensor& hasInitialState,
                              int64_t activationMode,
                              int64_t padSlotId,
                              at::Tensor& y)
{
    CheckNpuTensorDefined(x, "x");
    CheckNpuTensorDefined(weight, "weight");
    CheckNpuTensorDefined(convStates, "convStates");
    CheckNpuTensorDefined(queryStartLoc, "queryStartLoc");
    CheckNpuTensorDefined(cacheIndices, "cacheIndices");
    CheckNpuTensorDefined(hasInitialState, "hasInitialState");
    CheckNpuTensorDefined(y, "y");

    CheckContiguous(x, "x");
    CheckContiguous(weight, "weight");
    CheckContiguous(convStates, "convStates");
    CheckContiguous(queryStartLoc, "queryStartLoc");
    CheckContiguous(cacheIndices, "cacheIndices");
    CheckContiguous(hasInitialState, "hasInitialState");
    CheckContiguous(y, "y");

    TORCH_CHECK(x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16, "x must be float16/bfloat16");
    TORCH_CHECK(weight.scalar_type() == x.scalar_type(), "weight dtype must match x dtype");
    TORCH_CHECK(convStates.scalar_type() == x.scalar_type(), "convStates dtype must match x dtype");
    TORCH_CHECK(y.scalar_type() == x.scalar_type(), "y dtype must match x dtype");
    TORCH_CHECK(queryStartLoc.scalar_type() == at::kInt, "queryStartLoc dtype must be int32");
    TORCH_CHECK(cacheIndices.scalar_type() == at::kInt, "cacheIndices dtype must be int32");
    TORCH_CHECK(hasInitialState.scalar_type() == at::kBool, "hasInitialState dtype must be bool");

    if (biasOptional.has_value() && biasOptional.value().defined()) {
        CheckNpuTensorDefined(biasOptional.value(), "bias");
        CheckContiguous(biasOptional.value(), "bias");
        TORCH_CHECK(biasOptional.value().scalar_type() == x.scalar_type(), "bias dtype must match x dtype");
    }

    TORCH_CHECK(x.sizes().equals(y.sizes()), "y shape must match x shape");

    const aclDataType xyType = GetAclDataType(x.scalar_type(), "x");
    auto xAcl = WrapTensorAsAclTensor(x, xyType);
    auto weightAcl = WrapTensorAsAclTensor(weight, xyType);
    auto convStatesAcl = WrapTensorAsAclTensor(convStates, xyType);
    auto queryStartLocAcl = WrapTensorAsAclTensor(queryStartLoc, GetAclDataType(queryStartLoc.scalar_type(), "queryStartLoc"));
    auto cacheIndicesAcl = WrapTensorAsAclTensor(cacheIndices, GetAclDataType(cacheIndices.scalar_type(), "cacheIndices"));
    auto hasInitialStateAcl = WrapTensorAsAclTensor(hasInitialState, GetAclDataType(hasInitialState.scalar_type(), "hasInitialState"));
    auto yAcl = WrapTensorAsAclTensor(y, xyType);

    AclTensorPtr biasAcl;
    const aclTensor* biasAclRaw = nullptr;
    if (biasOptional.has_value() && biasOptional.value().defined()) {
        biasAcl = WrapTensorAsAclTensor(biasOptional.value(), xyType);
        biasAclRaw = biasAcl.get();
    }

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    const aclnnStatus ret1 = aclnnCausalConv1dGetWorkspaceSize(
        xAcl.get(), weightAcl.get(), biasAclRaw, convStatesAcl.get(), queryStartLocAcl.get(), cacheIndicesAcl.get(),
        hasInitialStateAcl.get(), activationMode, padSlotId, yAcl.get(), &workspaceSize, &executor);
    TORCH_CHECK(ret1 == ACLNN_SUCCESS, "aclnnCausalConv1dGetWorkspaceSize failed, error code: ", ret1);

    at::Tensor workspace;
    void* workspacePtr = nullptr;
    if (workspaceSize > 0) {
        TORCH_CHECK(workspaceSize <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
                    "workspaceSize is too large: ", workspaceSize);
        workspace = at::empty({static_cast<int64_t>(workspaceSize)},
                              at::TensorOptions().dtype(at::kByte).device(x.device()));
        workspacePtr = workspace.data_ptr();
    }

    const int deviceIndex = x.device().index();
    const auto stream = c10_npu::getCurrentNPUStream(deviceIndex).stream(false);

    const auto aclCall = [&]() -> int {
        const aclnnStatus ret2 = aclnnCausalConv1d(workspacePtr, workspaceSize, executor, stream);
        TORCH_CHECK(ret2 == ACLNN_SUCCESS, "aclnnCausalConv1d failed, error code: ", ret2);
        return 0;
    };
    at_npu::native::OpCommand::RunOpApi("CausalConv1d", aclCall);

    return y;
}

at::Tensor causal_conv1d(const at::Tensor& x,
                         const at::Tensor& weight,
                         const c10::optional<at::Tensor>& biasOptional,
                         at::Tensor& convStates,
                         const at::Tensor& queryStartLoc,
                         const at::Tensor& cacheIndices,
                         const at::Tensor& hasInitialState,
                         int64_t activationMode,
                         int64_t padSlotId)
{
    at::Tensor y = at::empty_like(x);
    (void)causal_conv1d_out(x, weight, biasOptional, convStates, queryStartLoc, cacheIndices, hasInitialState,
                           activationMode, padSlotId, y);
    return y;
}

} // namespace op_api

#endif // OPS_TRANSFORMER_CAUSAL_CONV1D_TORCH_ADAPTER_ENABLED

