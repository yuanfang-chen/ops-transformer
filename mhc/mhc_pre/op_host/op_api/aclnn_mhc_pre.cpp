/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_mhc_pre.h"
#include <dlfcn.h>
#include <new>
#include <memory>
#include <unordered_map>
#include "securec.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "mhc_pre.h"
#include "aclnn_kernels/transdata.h"
#include "aclnn_kernels/transpose.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/reshape.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {

struct MhcParamsBase {
    const aclTensor *x = nullptr;
    const aclTensor *phi = nullptr;
    const aclTensor *alpha = nullptr;
    const aclTensor *bias = nullptr;
    const aclTensor *gamma = nullptr;
    int64_t out_flag;
    double norm_eps;
    double hc_eps;
    const aclTensor *out_hin = nullptr;
    const aclTensor *out_h_post = nullptr;
    const aclTensor *out_h_res = nullptr;
    const aclTensor *out_inv_rms = nullptr;
    const aclTensor *out_mm_res = nullptr;
    const aclTensor *out_h_pre = nullptr;
    
    // 用于存储转换后的连续tensor（在ConvertDataContiguous中使用）
    const aclTensor *x_contiguous = nullptr;
    const aclTensor *phi_contiguous = nullptr;
    const aclTensor *alpha_contiguous = nullptr;
    const aclTensor *bias_contiguous = nullptr;
    const aclTensor *gamma_contiguous = nullptr;
};

class MhcBuilder {
public:
    static MhcBuilder Create()
    {
        MhcBuilder obj;

        return obj;
    } 

    MhcBuilder &SetInput(const aclTensor *x, const aclTensor *phi, const aclTensor *alpha, const aclTensor *bias, const aclTensor *gamma)
    {
        obj_.x = x;
        obj_.phi = phi;
        obj_.alpha = alpha;
        obj_.bias = bias;
        obj_.gamma = gamma;
        return *this;
    }

    MhcBuilder &SetAttr(int64_t out_flag, double norm_eps, double hc_eps)
    {
        obj_.out_flag = out_flag;
        obj_.norm_eps = norm_eps;
        obj_.hc_eps = hc_eps;
        return *this;
    }

    MhcBuilder &SetOutput(const aclTensor *out_hin, const aclTensor *out_h_post, const aclTensor *out_h_res)
    {
        obj_.out_hin = out_hin;
        obj_.out_h_post = out_h_post;
        obj_.out_h_res = out_h_res;
        return *this;
    }

    MhcBuilder &SetOptionalOutput(const aclTensor *out_inv_rms, const aclTensor *out_mm_res,
        const aclTensor *out_h_pre)
    {
        obj_.out_inv_rms = out_inv_rms;
        obj_.out_mm_res = out_mm_res;
        obj_.out_h_pre = out_h_pre;

        return *this;
    }

    MhcParamsBase Build() const
    {
        return obj_;
    }
private:
    MhcParamsBase obj_;
};

bool CheckNotNull(const MhcParamsBase &params)
{
    if (params.x == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "x tensor is nullptr");
        return false;
    }
    if (params.phi == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "phi tensor is nullptr");
        return false;
    }
    if (params.alpha == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "alpha tensor is nullptr");
        return false;
    }
    if (params.bias == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "bias tensor is nullptr");
        return false;
    }
    if (params.out_hin == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "out_hin tensor is nullptr");
        return false;
    }
    if (params.out_h_post == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "out_h_post tensor is nullptr");
        return false;
    }
    if (params.out_h_res == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "out_h_res tensor is nullptr");
        return false;
    }
    return true;
}

bool CheckEmptyTensor(const MhcParamsBase &params)
{
    if (params.x->IsEmpty()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x tensor is empty");
        return false;
    }
    if (params.phi->IsEmpty()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "phi tensor is empty");
        return false;
    }
    if (params.alpha->IsEmpty()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "alpha tensor is empty");
        return false;
    }
    if (params.bias->IsEmpty()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "bias tensor is empty");
        return false;
    }
    return true;
}

bool CheckInputOutDims(const MhcParamsBase &params)
{
    constexpr size_t DIM_NUM_1 = 1UL;
    constexpr size_t DIM_NUM_2 = 2UL;
    constexpr size_t DIM_NUM_3 = 3UL;
    constexpr size_t DIM_NUM_4 = 4UL;
    
    // x可以是3维或4维
    auto xDimNum = params.x->GetViewShape().GetDimNum();
    if (xDimNum != DIM_NUM_3 && xDimNum != DIM_NUM_4) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x tensor dim num must be 3 or 4, but got %zu", xDimNum);
        return false;
    }
    
    // phi应该是2维: (n^2+2n, nD)
    auto phiDimNum = params.phi->GetViewShape().GetDimNum();
    if (phiDimNum != DIM_NUM_2) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "phi tensor dim num must be 2, but got %zu", phiDimNum);
        return false;
    }

    // alpha应该是1维: (3)
    auto alphaDimNum = params.alpha->GetViewShape().GetDimNum();
    if (alphaDimNum != DIM_NUM_1) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "alpha tensor dim num must be 1, but got %zu", alphaDimNum);
        return false;
    }
    
    // bias应该是1维: (n^2+2n)
    auto biasDimNum = params.bias->GetViewShape().GetDimNum();
    if (biasDimNum != DIM_NUM_1) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "bias tensor dim num must be 1, but got %zu", biasDimNum);
        return false;
    }

    // gamma应该是2维: (n, D)
    if (params.gamma != nullptr) {
        auto gammaDimNum = params.gamma->GetViewShape().GetDimNum();
        if (gammaDimNum != DIM_NUM_2) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "gamma tensor dim num must be 2, but got %zu", gammaDimNum);
            return false;
        }
    }
    
    return true;
}

bool CheckInputOutShape(const MhcParamsBase &params)
{
    auto xShape = params.x->GetViewShape();
    auto phiShape = params.phi->GetViewShape();
    auto alphaShape = params.alpha->GetViewShape();
    auto biasShape = params.bias->GetViewShape();

    auto xDimNum = xShape.GetDimNum();
    
    // alpha的shape必须是(3)
    if (alphaShape.GetDim(0) != 3) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "alpha tensor shape must be (3), but got (%ld)", alphaShape.GetDim(0));
        return false;
    }
    
    // 从x的shape推导n和D
    int64_t n = 0;  // numsResidual
    int64_t d = 0;  // dimen
    int64_t nD = 0; // n * D
    
    if (xDimNum == 4) {
        // BSND格式: (B, S, N, D)
        n = xShape.GetDim(2);  // N维度
        d = xShape.GetDim(3);  // D维度
        nD = n * d;
    } else if (xDimNum == 3) {
        // TND格式: (T, N, D)
        n = xShape.GetDim(1);  // N维度
        d = xShape.GetDim(2);  // D维度
        nD = n * d;
    }
    
    if (n <= 0 || d <= 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Invalid x tensor shape: n=%ld, d=%ld", n, d);
        return false;
    }

    // N只支持4/6/8
    if (n != 4 && n != 6 && n != 8) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "N must be 4/6/8, but got n=%ld", n);
        return false;
    }

    // D只支持32字节对齐（对于BF16/FP16，即元素个数%16==0）
    if (d % 16 != 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "D must be 32 bytes aligned (element count mod 16 == 0 for BF16/FP16), but got d=%ld", d);
        return false;
    }
    
    int64_t n2_plus_2n = n * n + 2 * n;  // n^2 + 2n
    
    // phi的shape应该是(n^2+2n, nD)
    if (phiShape.GetDim(0) != n2_plus_2n) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "phi tensor first dim must be n^2+2n=%ld, but got %ld", n2_plus_2n, phiShape.GetDim(0));
        return false;
    }
    if (phiShape.GetDim(1) != nD) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "phi tensor second dim must be nD=%ld, but got %ld", nD, phiShape.GetDim(1));
        return false;
    }
    // bias的shape应该是(n^2+2n)
    if (biasShape.GetDim(0) != n2_plus_2n) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "bias tensor dim must be n^2+2n=%ld, but got %ld", n2_plus_2n, biasShape.GetDim(0));
        return false;
    }
    // gamma的shape应该是(n, D)
    if (params.gamma != nullptr) {
        auto gammaShape = params.gamma->GetViewShape();
        if (gammaShape.GetDim(0) != n) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "gamma tensor first dim must be n=%ld, but got %ld", n, gammaShape.GetDim(0));
            return false;
        }
        if (gammaShape.GetDim(1) != d) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "gamma tensor second dim must be D=%ld, but got %ld", d, gammaShape.GetDim(1));
            return false;
        }
    }
    
    return true;
}

bool CheckDtypeValid_mhc(const MhcParamsBase &params)
{
    // x支持BF16或FP16
    const std::initializer_list<DataType> X_SUPPORT_DTYPE_LIST = {DataType::DT_BF16, DataType::DT_FLOAT16};
    
    auto xDtype = params.x->GetDataType();
    bool xDtypeValid = false;
    for (const auto &dtype : X_SUPPORT_DTYPE_LIST) {
        if (xDtype == dtype) {
            xDtypeValid = true;
            break;
        }
    }
    if (!xDtypeValid) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x tensor dtype must be BF16 or FP16");
        return false;
    }
    
    // phi, gamma, alpha, bias都必须是FP32
    if (params.phi->GetDataType() != DataType::DT_FLOAT) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "phi tensor dtype must be FP32");
        return false;
    }
    if (params.alpha->GetDataType() != DataType::DT_FLOAT) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "alpha tensor dtype must be FP32");
        return false;
    }
    if (params.bias->GetDataType() != DataType::DT_FLOAT) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "bias tensor dtype must be FP32");
        return false;
    }
    if (params.gamma != nullptr) {
        if (params.gamma->GetDataType() != DataType::DT_FLOAT) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "gamma tensor dtype must be FP32");
            return false;
        }
    }
    
    return true;
}

static bool IsPrivateFormat(ge::Format format)
{
    if (format == ge::FORMAT_NC1HWC0 || format == ge::FORMAT_FRACTAL_Z || format == ge::FORMAT_NDC1HWC0 ||
        format == ge::FORMAT_FRACTAL_Z_3D || format == ge::FORMAT_FRACTAL_NZ || format == ge::FORMAT_NC1HWC0_C04) {
        return true;
    }

    return false;
}

bool CheckFormat(const MhcParamsBase &params)
{
    // 检查所有输入tensor的format必须是ND格式
    if (IsPrivateFormat(params.x->GetViewFormat())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x tensor format must be ND");
        return false;
    }
    
    if (IsPrivateFormat(params.phi->GetViewFormat())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "phi tensor format must be ND");
        return false;
    }

    if (IsPrivateFormat(params.alpha->GetViewFormat())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "alpha tensor format must be ND");
        return false;
    }
    
    if (IsPrivateFormat(params.bias->GetViewFormat())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "bias tensor format must be ND");
        return false;
    }

    if (params.gamma != nullptr) {
        if (IsPrivateFormat(params.gamma->GetViewFormat())) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "gamma tensor format must be ND");
            return false;
        }
    }

    return true;
}

aclnnStatus CheckParams(const MhcParamsBase &params)
{
    // 1. 检查参数是否为空指针、空tensor
    CHECK_RET(CheckNotNull(params), ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckEmptyTensor(params), ACLNN_ERR_PARAM_INVALID);

    // 2. 校验输入、输出参数维度
    CHECK_RET(CheckInputOutDims(params), ACLNN_ERR_PARAM_INVALID);

    // 3. 校验输入、输出shape参数
    CHECK_RET(CheckInputOutShape(params), ACLNN_ERR_PARAM_INVALID);

    // 4. 检查输入的数据类型是否在支持的数据类型范围之内
    CHECK_RET(CheckDtypeValid_mhc(params), ACLNN_ERR_PARAM_INVALID);

    // 5. 检查数据形状是否支持
    CHECK_RET(CheckFormat(params), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

aclnnStatus ConvertDataContiguous(MhcParamsBase &params, aclOpExecutor *executor)
{
    // 将输入tensor转换为连续格式
    params.x_contiguous = l0op::Contiguous(params.x, executor);
    CHECK_RET(params.x_contiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    
    params.phi_contiguous = l0op::Contiguous(params.phi, executor);
    CHECK_RET(params.phi_contiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    params.alpha_contiguous = l0op::Contiguous(params.alpha, executor);
    CHECK_RET(params.alpha_contiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    
    params.bias_contiguous = l0op::Contiguous(params.bias, executor);
    CHECK_RET(params.bias_contiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    if (params.gamma != nullptr) {
        params.gamma_contiguous = l0op::Contiguous(params.gamma, executor);
        CHECK_RET(params.gamma_contiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    
    return ACLNN_SUCCESS;
}

static aclnnStatus mHCPreCommonProcess(MhcParamsBase &params, aclOpExecutor *executor)
{
    auto ret = CheckParams(params);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    ret = ConvertDataContiguous(params, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 使用转换后的连续tensor
    auto outParams = l0op::MhcPre(
        params.x_contiguous, params.phi_contiguous, params.alpha_contiguous, params.bias_contiguous, params.gamma_contiguous,
        params.out_flag, params.norm_eps, params.hc_eps, executor);
    CHECK_RET(outParams != std::tuple(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr), ACLNN_ERR_INNER_NULLPTR);

    auto out0 = std::get<0>(outParams);
    auto ret0 = l0op::ViewCopy(out0, params.out_hin, executor);
    CHECK_RET(ret0 != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto out1 = std::get<1>(outParams);
    auto ret1 = l0op::ViewCopy(out1, params.out_h_post, executor);
    CHECK_RET(ret1 != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto out2 = std::get<2>(outParams);
    auto retView = l0op::ViewCopy(out2, params.out_h_res, executor);
    CHECK_RET(retView != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto out3 = std::get<3>(outParams);
    retView = l0op::ViewCopy(out3, params.out_inv_rms, executor);
    CHECK_RET(retView != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto out4 = std::get<4>(outParams);
    retView = l0op::ViewCopy(out4, params.out_mm_res, executor);
    CHECK_RET(retView != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto out5 = std::get<5>(outParams);
    retView = l0op::ViewCopy(out5, params.out_h_pre, executor);
    CHECK_RET(retView != nullptr, ACLNN_ERR_INNER_NULLPTR);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnMhcPreGetWorkspaceSize(
    const aclTensor *x, const aclTensor *phi, const aclTensor *alpha, const aclTensor *bias, const aclTensor *gamma,
    int64_t out_flag, double norm_eps, double hc_eps,
    const aclTensor *out_hin, const aclTensor *out_h_post, const aclTensor *out_h_res,
    const aclTensor *out_inv_rms, const aclTensor *out_mm_res, const aclTensor *out_h_pre,
    uint64_t *workspaceSize, aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnMhcPre, DFX_IN(x, phi, alpha, bias, gamma, out_flag, norm_eps, hc_eps),
        DFX_OUT(out_hin, out_h_post, out_h_res, out_inv_rms, out_mm_res, out_h_pre));
    auto uniqueExecutor = CREATE_EXECUTOR();

    MhcParamsBase params =
        MhcBuilder::Create()
        .SetInput(x, phi, alpha, bias, gamma)
        .SetAttr(out_flag, norm_eps, hc_eps)
        .SetOutput(out_hin, out_h_post, out_h_res)
        .SetOptionalOutput(out_inv_rms, out_mm_res, out_h_pre)
        .Build();

    auto ret = mHCPreCommonProcess(params, uniqueExecutor.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnMhcPre(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
    aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnMhcPre);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

}
#ifdef __cplusplus
}
#endif