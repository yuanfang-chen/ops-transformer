/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under
 * the terms and conditions of CANN Open Software License Agreement Version 2.0
 * (the "License"). Please refer to the License for details. You may not use
 * this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY
 * KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
 * NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the
 * License.
 */
#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include "dequant_swiglu_quant_tiling.h"
#include "kernel_operator.h"
#include "op_kernel/dequant_swiglu_quant_static_bf16.h"
#include "dequant_swiglu_quant_torch.h"

using namespace ascend_ops_dsq;

const std::vector<c10::ScalarType> SUPPORTED_DTYPE_X = {c10::ScalarType::BFloat16, c10::ScalarType::Half, c10::ScalarType::Int, c10::ScalarType::Int,
    c10::ScalarType::Int, c10::ScalarType::Int};
const std::vector<c10::ScalarType> SUPPORTED_DTYPE_WEIGHT = {c10::ScalarType::Float, c10::ScalarType::Float, c10::ScalarType::Float, c10::ScalarType::Float,
    c10::ScalarType::Float, c10::ScalarType::Float};
const std::vector<c10::ScalarType> SUPPORTED_DTYPE_ACTIV = {c10::ScalarType::Float, c10::ScalarType::Float, c10::ScalarType::Float, c10::ScalarType::Float,
    c10::ScalarType::Float, c10::ScalarType::Float};
const std::vector<c10::ScalarType> SUPPORTED_DTYPE_BIAS = {c10::ScalarType::Float, c10::ScalarType::Float, c10::ScalarType::Half, c10::ScalarType::BFloat16,
    c10::ScalarType::Float, c10::ScalarType::Int};
const std::vector<c10::ScalarType> SUPPORTED_DTYPE_SCALE = {c10::ScalarType::Float, c10::ScalarType::Float, c10::ScalarType::Float, c10::ScalarType::Float,
    c10::ScalarType::Float, c10::ScalarType::Float};
const std::vector<c10::ScalarType> SUPPORTED_DTYPE_OFFSET = {c10::ScalarType::Float, c10::ScalarType::Float, c10::ScalarType::Float, c10::ScalarType::Float,
    c10::ScalarType::Float, c10::ScalarType::Float};
const std::vector<c10::ScalarType> SUPPORTED_DTYPE_GROUP = {c10::ScalarType::Long, c10::ScalarType::Long, c10::ScalarType::Long, c10::ScalarType::Long, 
    c10::ScalarType::Long, c10::ScalarType::Long};

const std::vector<c10::ScalarType> SUPPORTED_DTYPE_OUTY = {c10::ScalarType::Char, c10::ScalarType::Char, c10::ScalarType::Char, c10::ScalarType::Char,
    c10::ScalarType::Char, c10::ScalarType::Char};
const std::vector<c10::ScalarType> SUPPORTED_DTYPE_OUTSCALE = {c10::ScalarType::Float, c10::ScalarType::Float, c10::ScalarType::Float, c10::ScalarType::Float,
    c10::ScalarType::Float, c10::ScalarType::Float};

const std::vector<TypeCombo> &getSupportedCombos()
{
    static const auto combos = TypeComboManager::createCombosFromLists(
        SUPPORTED_DTYPE_X, SUPPORTED_DTYPE_WEIGHT, SUPPORTED_DTYPE_ACTIV, SUPPORTED_DTYPE_BIAS,
        SUPPORTED_DTYPE_SCALE, SUPPORTED_DTYPE_OFFSET, SUPPORTED_DTYPE_GROUP,
        SUPPORTED_DTYPE_OUTY, SUPPORTED_DTYPE_OUTSCALE);
    return combos;
}


__global__ __aicore__ void DequantSwigluQuantKernelLaunch_STATIC_FLOAT16_X(__gm__ uint8_t * xGM, __gm__ uint8_t * weightSscaleGM,
                __gm__ uint8_t * activationScaleGM, __gm__ uint8_t * biasGM, __gm__ uint8_t * quantScaleGM,
                __gm__ uint8_t * quantOffsetGM, __gm__ uint8_t * yGM, __gm__ uint8_t * scaleGM, SwiGluTilingData tilingData)
{
    AscendC::TPipe pipe;
    DequantSwigluQuant::DequantSwigluQuantStaticBF16<half, float, half, int8_t, 1, 1> op;
    op.Init(xGM, weightSscaleGM, activationScaleGM, biasGM, quantScaleGM, quantOffsetGM, yGM, scaleGM, &tilingData,
            &(pipe));
    op.Process();
}

__global__ __aicore__ void DequantSwigluQuantKernelLaunch_STATIC_FLOAT16_XD(__gm__ uint8_t *xGM, __gm__ uint8_t *weightSscaleGM,
                __gm__ uint8_t * activationScaleGM, __gm__ uint8_t * biasGM, __gm__ uint8_t * quantScaleGM,
                __gm__ uint8_t * quantOffsetGM, __gm__ uint8_t * yGM, __gm__ uint8_t * scaleGM, SwiGluTilingData tilingData)
{
    AscendC::TPipe pipe;
    DequantSwigluQuant::DequantSwigluQuantStaticBF16<half, float, half, int8_t, 1, 0> op;
    op.Init(xGM, weightSscaleGM, activationScaleGM, biasGM, quantScaleGM, quantOffsetGM, yGM, scaleGM, &tilingData,
            &(pipe));
    op.Process();
}

// dst_type=2 -> int8
std::tuple<torch::Tensor, torch::Tensor> dequant_swiglu_quant(
    torch::Tensor x, c10::optional<torch::Tensor> weight_scale, c10::optional<torch::Tensor> activation_scale,
    c10::optional<torch::Tensor> bias, c10::optional<torch::Tensor> quant_scale, c10::optional<torch::Tensor> quant_offset,
    c10::optional<torch::Tensor> group_index,
    bool activate_left = false, std::string quant_mode = "static", int64_t dst_type = 2, std::string round_mode = "rint",
    int64_t activate_dim = -1, int64_t swiglu_mode = 0, float clamp_limit = 7.0, float glu_alpha = 1.702, float glu_bias = 1.0)
{
    checkTensorOnNPU(x, "x", false);
    checkTensorOnNPU(weight_scale, "weight_scale", true);
    checkTensorOnNPU(activation_scale, "activation_scale", true);
    checkTensorOnNPU(bias, "bias", true);
    checkTensorOnNPU(quant_scale, "quant_scale", true);
    checkTensorOnNPU(quant_offset, "quant_offset", true);
    checkTensorOnNPU(group_index, "group_index", true);

    const auto &SUPPORTED_COMBOS = getSupportedCombos();
    int matched_index = TypeComboManager::findMatchingCombo(SUPPORTED_COMBOS, x, weight_scale, activation_scale, bias, quant_scale,
                                                            quant_offset, group_index);

    // if (matched_index == -1) {
    //     TORCH_CHECK(false, "no match dtype combo");
    // }
    TORCH_CHECK(matched_index != -1, "no match dtype combo");

    const auto &matched_combo = SUPPORTED_COMBOS[matched_index];
    const c10::ScalarType yDtype = matched_combo.output_y;
    const c10::ScalarType scaleDtype = matched_combo.output_scale;

    // auto shapeX = getTensorListFirstShape(x);
    // int64_t m = shapeX[0];
    // auto shapeWeight = getTensorListFirstShape(weight);
    // int64_t n = shapeWeight[1];

    // std::vector<int64_t> xShape = x.sizes().vec();
    // TORCH_CHECK(xShape.size() > 1, "xShape.size() should be > 1, but got: ", xShape.size());
    // int64_t xCol = xShape.back(); // 或 x.size(-1)
    // TORCH_CHECK(xCol != 0, "xCol cannot be 0");
    // int64_t xRow = x.numel() / xCol;

    std::vector<DsqTensorDescTiling> inTensors;
    AddTensorDesc(x, "x", inTensors, false);
    AddTensorDesc(weight_scale, "weight_scale", inTensors, false);
    AddTensorDesc(activation_scale, "activation_scale", inTensors, false);
    AddTensorDesc(bias, "bias", inTensors, false);
    AddTensorDesc(quant_scale, "quant_scale", inTensors, false);
    AddTensorDesc(quant_offset, "quant_offset", inTensors, false);
    AddTensorDesc(group_index, "group_index", inTensors, false);

    DsqAttrTiling inAttrs = {
        .activate_left = activate_left,
        .quant_mode = quant_mode,
        .dst_type = dst_type,
        .round_mode = round_mode,
        .activate_dim = activate_dim,
        .swiglu_mode = swiglu_mode,
        .clamp_limit = clamp_limit,
        .glu_alpha = glu_alpha,
        .glu_bias = glu_bias
    };


    std::vector<int64_t> xShape = inTensors[0].tShape;
    std::vector<int64_t> yShape(xShape);
    size_t selectDim = (activate_dim >= 0) ? static_cast<size_t>(activate_dim) : static_cast<size_t>(activate_dim + x.dim());
    yShape[selectDim] = xShape[selectDim] / 2;
    std::vector<int64_t> scaleShape(xShape.begin(), xShape.end() - 1);
    if (selectDim < xShape.size() - 1) {
        scaleShape[selectDim] = xShape[selectDim] / 2;
    }
    // 指定目标shape  数据类型  对齐x的设备（CPU/GPU/Ascend） 
    at::Tensor yOutTensor = at::empty(yShape,       // 指定目标shape                
                                at::dtype(yDtype)     // 数据类型
                                .device(x.device()) // 对齐x的设备（CPU/GPU/Ascend）
                                .layout(x.layout()) // 对齐x的内存布局
    );

    at::Tensor scaleOutTensor = at::empty(scaleShape,          // 指定目标shape
                                        at::dtype(scaleDtype)  // 数据类型
                                        .device(x.device()) // 对齐x的设备（CPU/GPU/Ascend）
                                        .layout(x.layout()) // 对齐x的内存布局
    );

    std::vector<DsqTensorDescTiling> outTensors;
    DsqDataType dsqYType = ConvertScalarTypeToDsqType("main", "yOut", yDtype);
    DsqDataType dsqScaleType = ConvertScalarTypeToDsqType("main", "scaleOut", scaleDtype);
    outTensors.push_back(DsqTensorDescTiling{.tShape = yShape, .tDataType = dsqYType, .tIsNull = false});
    outTensors.push_back(DsqTensorDescTiling{.tShape = scaleShape, .tDataType = dsqScaleType, .tIsNull = false});

    auto stream = c10_npu::getCurrentNPUStream().stream(false);

    // DsqDataType xDtype = DT_FLOAT16;
    // if (matched_combo.x == at::kBFloat16) {
    //     xDtype = DT_BF16;
    // } else if (matched_combo.x == c10::ScalarType::Int) {
    //     xDtype = DT_INT32;
    // }
    // DsqDataType biasDataType = DT_FLOAT;
    // bool biasIsEmpty = true;
    // if (bias.has_value()) {
    //     biasIsEmpty = false;
    //     if (matched_combo.bias == at::kBFloat16) {
    //         biasDataType = DT_BF16;
    //     } else if (matched_combo.bias == at::kHalf) {
    //         biasDataType = DT_FLOAT16;
    //     } else if (matched_combo.bias == c10::ScalarType::Int) {
    //         biasDataType = DT_INT32;
    //     }
    // }
    // bool activateScaleIsEmpty = !(activation_scale.has_value());
    // bool quantScaleIsEmpty = !(quant_scale.has_value());
    // uint64_t quantScaleShapeSize = 0;
    // if (quant_scale.has_value()) {
    //     quantScaleShapeSize = quant_scale.value().numel();
    // }
    ////////// torch::Tensor& x, c10::optional<torch::Tensor>& weight_scale, c10::optional<torch::Tensor>& activation_scale,
    //////////// c10::optional<torch::Tensor>& bias, c10::optional<torch::Tensor>& quant_scale, c10::optional<torch::Tensor>& quant_offset,
    /////////// c10::optional<torch::Tensor>& group_index,
    auto x_ptr = get_first_tensor_address<torch::Tensor>(matched_combo.x, x, false);
    auto weight_scale_ptr = get_first_tensor_address<c10::optional<torch::Tensor>>(matched_combo.weight_scale, weight_scale, true);
    auto activation_scale_ptr = get_first_tensor_address<c10::optional<torch::Tensor>>(matched_combo.activation_scale, activation_scale, true);
    auto bias_ptr = get_first_tensor_address<c10::optional<torch::Tensor>>(matched_combo.bias, bias, true);
    auto quant_scale_ptr = get_first_tensor_address<c10::optional<torch::Tensor>>(matched_combo.quant_scale, quant_scale, true);
    auto quant_offset_ptr = get_first_tensor_address<c10::optional<torch::Tensor>>(matched_combo.quant_offset, quant_offset, true);
    auto group_index_ptr =
        get_first_tensor_address<c10::optional<torch::Tensor>>(matched_combo.group_index, group_index, true);

    auto y_out_ptr = get_first_tensor_address<torch::Tensor>(matched_combo.output_y, yOutTensor, false);
    auto scale_out_ptr = get_first_tensor_address<torch::Tensor>(matched_combo.output_scale, scaleOutTensor, false);

    // do tiling
    if (group_index.has_value()) {
        // 非空，走 DskTiling
    } else {
        // 空，走 DequantSwigluQuantTiling
        SwiGluTilingData tilingData;
        DequantSwigluQuantTiling opTiling(&tilingData);
        uint64_t tilingKey = 0;
        uint64_t workspaceSize = 0;
        uint64_t blockDim = 0;
        bool ret = opTiling.DoTiling(inTensors, inAttrs, outTensors, tilingKey, workspaceSize, blockDim);
        TORCH_CHECK(ret == true, "DequantSwigluQuantTiling tiling failed");
        std::cout << "INFO tilingKey:" << tilingKey << " workspaceSize:" << workspaceSize << " blockDim:" << blockDim << std::endl;

        void *workspacePtr = nullptr;
        if (workspaceSize > 0) {
            auto ret = aclrtMalloc(&workspacePtr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            TORCH_CHECK(ret == ACL_SUCCESS, "allocate workspace failed. ERROR: %d\n", ret);
        }

        if (tilingKey == STATIC_FLOAT16_X) {
            DequantSwigluQuantKernelLaunch_STATIC_FLOAT16_X<<<blockDim, nullptr, stream>>>((uint8_t *)x_ptr, (uint8_t *)weight_scale_ptr,
                (uint8_t *)activation_scale_ptr, (uint8_t *)bias_ptr, (uint8_t *)quant_scale_ptr,
                (uint8_t *)quant_offset_ptr, (uint8_t *)y_out_ptr, (uint8_t *)scale_out_ptr, tilingData);
        } else if (tilingKey == STATIC_FLOAT16_XD) {
            DequantSwigluQuantKernelLaunch_STATIC_FLOAT16_XD<<<blockDim, nullptr, stream>>>((uint8_t *)x_ptr, (uint8_t *)weight_scale_ptr,
                (uint8_t *)activation_scale_ptr, (uint8_t *)bias_ptr, (uint8_t *)quant_scale_ptr,
                (uint8_t *)quant_offset_ptr, (uint8_t *)y_out_ptr, (uint8_t *)scale_out_ptr, tilingData);
        } else {
            if (workspaceSize > 0) {
                aclrtFree(workspacePtr);
            }
            TORCH_CHECK(false, "unsupported tilingKey: %lu\n", tilingKey);
        }

        if (workspaceSize > 0) {
            aclrtFree(workspacePtr);
        }
    }

    return std::make_tuple(yOutTensor, scaleOutTensor);
}

PYBIND11_MODULE(ascendc_ops, m) {
    m.def("dequant_swiglu_quant", &dequant_swiglu_quant, "dequant_swiglu_quant",
        py::arg("x"),
        py::arg("weight_scale") = py::none(),
        py::arg("activation_scale") = py::none(),
        py::arg("bias") = py::none(),
        py::arg("quant_scale") = py::none(),
        py::arg("quant_offset") = py::none(),
        py::arg("group_index") = py::none(),
        py::arg("activate_left") = false,
        py::arg("quant_mode") = "static",
        py::arg("dst_type") = 2,
        py::arg("round_mode") = "rint",
        py::arg("activate_dim") = -1,
        py::arg("swiglu_mode") = 0,
        py::arg("clamp_limit") = 7.0,
        py::arg("glu_alpha") = 1.702,
        py::arg("glu_bias") = 1.0
    );
}
