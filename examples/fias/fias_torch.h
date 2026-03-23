/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 */

#ifndef FIAS_TORCH_H
#define FIAS_TORCH_H

#include <ATen/ATen.h>
#include <vector>
#include <map>
#include <string>
#include <torch/all.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "fias_tiling.h"

using namespace fias;

// ---------- error helpers ----------
inline std::string build_error_msg(const std::string &func_name, const std::string &name, const std::string &message)
{
    std::ostringstream oss;
    oss << func_name << " " << name << " " << message;
    return oss.str();
}

// ---------- scalar type → FiasDataType ----------
const std::map<c10::ScalarType, FiasDataType> scalarToFiasDataType = {
    {c10::ScalarType::Float, FDT_FLOAT},
    {c10::ScalarType::Half, FDT_FLOAT16},
    {c10::ScalarType::Char, FDT_INT8},
    {c10::ScalarType::Int, FDT_INT32},
    {c10::ScalarType::BFloat16, FDT_BF16},
    {c10::ScalarType::Bool, FDT_BOOL},
    {c10::ScalarType::Long, FDT_INT64},
};

static FiasDataType ConvertScalarTypeToFiasType(const std::string &func_name, const std::string &name,
                                                 const c10::ScalarType dtype)
{
    const auto iter = scalarToFiasDataType.find(dtype);
    TORCH_CHECK(iter != scalarToFiasDataType.end(), build_error_msg(func_name, name, "dtype not found"));
    return iter->second;
}

// ---------- checkTensorOnNPU ----------
template <typename TensorContainer>
void checkTensorOnNPU(const TensorContainer &container, const std::string &name = "curTensor", bool allow_empty = false)
{
    const std::string &func_name = "[checkTensorOnNPU]";
    if constexpr (std::is_same_v<TensorContainer, c10::optional<torch::Tensor>>) {
        if (!container.has_value()) {
            TORCH_CHECK(allow_empty,
                        build_error_msg(func_name, name, " is not provided (null optional), but empty is not allowed"));
            return;
        }
    }

    auto get_value = [&]() -> auto & {
        if constexpr (std::is_same_v<TensorContainer, c10::optional<torch::Tensor>>) {
            return container.value();
        } else if constexpr (std::is_same_v<TensorContainer, torch::Tensor>) {
            return container;
        }
    };
    const auto &value = get_value();

    if constexpr (std::is_same_v<TensorContainer, torch::Tensor> ||
                  std::is_same_v<TensorContainer, c10::optional<torch::Tensor>>) {
        if (!value.defined()) {
            TORCH_CHECK(allow_empty, build_error_msg(func_name, name, " tensor is undefined"));
            return;
        }
        if (!torch_npu::utils::is_npu(value)) {
            TORCH_CHECK(false, build_error_msg(func_name, name, " tensor must be on NPU device"));
        }
    }
}

// ---------- AddTensorDesc ----------
template <typename TensorType>
static void AddTensorDesc(const TensorType &atTensor, const std::string &name,
                           std::vector<FiasTensorDescTiling> &tensorDescList, bool allow_empty = false)
{
    const std::string &func_name = "[AddTensorDesc]";
    if constexpr (std::is_same_v<TensorType, c10::optional<torch::Tensor>>) {
        if (!atTensor.has_value()) {
            tensorDescList.push_back(FiasTensorDescTiling{.tShape = {}, .tDataType = FDT_MAX, .tIsNull = true});
            return;
        } else {
            if (!atTensor.value().defined()) {
                TORCH_CHECK(allow_empty,
                            build_error_msg(func_name, name, " tensor is undefined, but empty is not allowed"));
                tensorDescList.push_back(FiasTensorDescTiling{.tShape = {}, .tDataType = FDT_MAX, .tIsNull = true});
                return;
            }
            const torch::Tensor &tensor = atTensor.value();
            const auto iter = scalarToFiasDataType.find(tensor.scalar_type());
            TORCH_CHECK(iter != scalarToFiasDataType.end(),
                        build_error_msg(func_name, name, "scalar_type not defined"));
            tensorDescList.push_back(
                FiasTensorDescTiling{.tShape = tensor.sizes().vec(), .tDataType = iter->second, .tIsNull = false});
            return;
        }
    }
    if constexpr (std::is_same_v<TensorType, torch::Tensor>) {
        if (!atTensor.defined()) {
            TORCH_CHECK(allow_empty,
                        build_error_msg(func_name, name, " tensor is undefined, but empty is not allowed"));
            tensorDescList.push_back(FiasTensorDescTiling{.tShape = {}, .tDataType = FDT_MAX, .tIsNull = true});
            return;
        }
        FiasDataType fiasType = ConvertScalarTypeToFiasType(func_name, name, atTensor.scalar_type());
        tensorDescList.push_back(
            FiasTensorDescTiling{.tShape = atTensor.sizes().vec(), .tDataType = fiasType, .tIsNull = false});
        return;
    }

    TORCH_CHECK(false, build_error_msg(func_name, name, "not supported TensorType"));
}

// ---------- get_first_tensor_address ----------
template <typename TensorType>
void *get_first_tensor_address(const TensorType &input, bool allow_empty = false)
{
    if constexpr (std::is_same_v<TensorType, c10::optional<torch::Tensor>>) {
        if (!input.has_value() || !input->defined()) {
            TORCH_CHECK(allow_empty, "optional<Tensor> has no value");
            return nullptr;
        }
        return input->data_ptr();
    } else if constexpr (std::is_same_v<TensorType, torch::Tensor>) {
        if (!input.defined()) {
            TORCH_CHECK(allow_empty, "Tensor is undefined");
            return nullptr;
        }
        return input.data_ptr();
    } else {
        static_assert(std::is_same_v<TensorType, void>, "Unsupported tensor type");
        return nullptr;
    }
}

#endif // FIAS_TORCH_H
