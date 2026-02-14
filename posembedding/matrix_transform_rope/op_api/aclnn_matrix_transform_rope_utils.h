/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to License for details. You may not use this file except in compliance with License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_HOST_OP_API_ACLNN_MATRIX_TRANSFORM_ROPE_UTILS_H
#define OP_HOST_OP_API_ACLNN_MATRIX_TRANSFORM_ROPE_UTILS_H

#include "aclnn_kernels/contiguous.h"
#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/make_op_executor.h"
#include "matrix_transform_rope.h"

namespace matrix_transform_rope {

using namespace op;

/**
 * @brief 算子参数结构体，存储所有输入输出参数
 *
 * 设计说明：
 * 1. 原始输入输出 tensor：x, cos, sin, rotate（用户传入）
 * 2. 连续格式 tensor：x_contiguous, cos_contiguous, sin_contiguous, rotate_contiguous（内部使用）
 * 3. 输出 tensor：out（用户传入，用于拷贝结果）
 */
struct MatrixTransformRopeParamsBase {
    // ========== 原始输入 tensor（用户传入，不可修改） ==========
    const aclTensor *x = nullptr;
    const aclTensor *cos = nullptr;
    const aclTensor *sin = nullptr;
    const aclTensor *rotate = nullptr;

    // ========== 连续格式 tensor（内部使用，指向转换后的 tensor） ==========
    const aclTensor *x_contiguous = nullptr;
    const aclTensor *cos_contiguous = nullptr;
    const aclTensor *sin_contiguous = nullptr;
    const aclTensor *rotate_contiguous = nullptr;

    // ========== 输出 tensor（用户传入，用于拷贝结果） ==========
    const aclTensor *out = nullptr;
};

/**
 * @brief Builder 类，用于构建 MatrixTransformRopeParamsBase 参数结构体
 * 使用 Builder 模式，提供链式调用的方式设置参数
 */
class MatrixTransformRopeParamsBuilder {
public:
    /**
     * @brief 创建 Builder 实例
     * @param x 第一个输入 tensor
     * @param cos 第二个输入 tensor
     * @param sin 第三个输入 tensor
     * @param rotate 第四个输入 tensor
     * @param out 输出 tensor
     * @return MatrixTransformRopeParamsBuilder 实例
     */
    static MatrixTransformRopeParamsBuilder Create(const aclTensor *x, const aclTensor *cos,
                                           const aclTensor *sin, const aclTensor *rotate,
                                           const aclTensor *out)
    {
        MatrixTransformRopeParamsBuilder b;
        b.p_.x = x;
        b.p_.cos = cos;
        b.p_.sin = sin;
        b.p_.rotate = rotate;
        b.p_.out = out;
        return b;
    }

    /**
     * @brief 构建参数结构体
     * @return MatrixTransformRopeParamsBase 参数结构体
     */
    MatrixTransformRopeParamsBase Build() const
    {
        return p_;
    }

private:
    MatrixTransformRopeParamsBase p_;
};

/**
 * @brief Handler 基类，定义多 SOC 架构的通用接口
 *
 * 设计说明：
 * 1. 基类定义通用的校验和处理流程框架
 * 2. 不同 SOC 版本通过继承基类实现具体的校验逻辑
 * 3. 使用工厂模式根据当前 NPU 架构选择对应的 Handler
 */
class MatrixTransformRopeHandler {
public:
    virtual ~MatrixTransformRopeHandler() = default;

protected:
    /**
     * @brief 检查参数是否为 nullptr
     * @return true 表示所有必需参数都不为 nullptr，false 表示有参数为 nullptr
     */
    virtual bool CheckNotNull(void)
    {
        OP_CHECK_NULL(params_.x, return false);
        OP_CHECK_NULL(params_.cos, return false);
        OP_CHECK_NULL(params_.sin, return false);
        OP_CHECK_NULL(params_.rotate, return false);
        OP_CHECK_NULL(params_.out, return false);
        return true;
    }

    /**
     * @brief 检查 tensor 是否为空
     * @return true 表示所有 tensor 都不为空，false 表示有 tensor 为空
     */
    virtual bool CheckEmptyTensor(void)
    {
        if (params_.x->IsEmpty()) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x tensor is empty");
            return false;
        }
        if (params_.cos->IsEmpty()) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "cos tensor is empty");
            return false;
        }
        if (params_.sin->IsEmpty()) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "sin tensor is empty");
            return false;
        }
        if (params_.rotate->IsEmpty()) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "rotate tensor is empty");
            return false;
        }
        return true;
    }

    /**
     * @brief 检查输入输出 tensor 的维度
     * 子类必须实现此方法，根据不同 SOC 的校验规则进行检查
     * @return true 表示维度符合要求，false 表示维度不符合要求
     */
    virtual bool CheckInputOutDims() = 0;

    /**
     * @brief 检查输入输出 tensor 的 shape
     * 子类必须实现此方法，根据不同 SOC 的校验规则进行检查
     * @return true 表示 shape 符合要求，false 表示 shape 不符合要求
     */
    virtual bool CheckInputOutShape() = 0;

    /**
     * @brief 检查数据类型是否有效
     * 子类必须实现此方法，根据不同 SOC 支持的数据类型进行检查
     * @return true 表示数据类型有效，false 表示数据类型无效
     */
    virtual bool CheckDtypeValid() = 0;

    /**
     * @brief 检查 format 是否支持
     * 子类必须实现此方法，根据不同 SOC 支持的 format 进行检查
     * @return true 表示 format 支持，false 表示 format 不支持
     */
    virtual bool CheckFormat() = 0;

    /**
     * @brief 综合参数校验
     * 按照固定的校验流程依次调用各个校验方法
     * @return aclnnStatus 状态码
     */
    virtual aclnnStatus CheckParams()
    {
        // 1. 检查参数是否为空指针、空 tensor
        CHECK_RET(CheckNotNull(), ACLNN_ERR_PARAM_NULLPTR);
        CHECK_RET(CheckEmptyTensor(), ACLNN_ERR_PARAM_INVALID);

        // 2. 校验输入、输出参数维度
        CHECK_RET(CheckInputOutDims(), ACLNN_ERR_PARAM_INVALID);

        // 3. 校验输入、输出 shape 参数
        CHECK_RET(CheckInputOutShape(), ACLNN_ERR_PARAM_INVALID);

        // 4. 检查输入的数据类型是否在支持的数据类型范围之内
        CHECK_RET(CheckDtypeValid(), ACLNN_ERR_PARAM_INVALID);

        // 5. 检查数据形状是否支持
        CHECK_RET(CheckFormat(), ACLNN_ERR_PARAM_INVALID);

        return ACLNN_SUCCESS;
    }

    /**
     * @brief 将输入 tensor 转换为连续格式
     *
     * 设计说明：
     * 1. 原始 tensor（params_.x/cos/sin/rotate）保持不变，用于错误信息输出
     * 2. 连续格式 tensor（params_.x_contiguous/cos_contiguous 等）用于实际计算
     * 3. 如果原始 tensor 已经是连续格式，可直接赋值，无需调用 l0op::Contiguous
     *
     * 子类可以重写此方法以实现特定的数据转换逻辑
     * @return aclnnStatus 状态码
     */
    virtual aclnnStatus ConvertDataContiguous()
    {
        // 转换 x 为连续格式
        if (params_.x->IsContiguous()) {
            params_.x_contiguous = params_.x;
        } else {
            params_.x_contiguous = l0op::Contiguous(params_.x, l0Executor_);
            CHECK_RET(params_.x_contiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
        }

        // 转换 cos 为连续格式
        if (params_.cos->IsContiguous()) {
            params_.cos_contiguous = params_.cos;
        } else {
            params_.cos_contiguous = l0op::Contiguous(params_.cos, l0Executor_);
            CHECK_RET(params_.cos_contiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
        }

        // 转换 sin 为连续格式
        if (params_.sin->IsContiguous()) {
            params_.sin_contiguous = params_.sin;
        } else {
            params_.sin_contiguous = l0op::Contiguous(params_.sin, l0Executor_);
            CHECK_RET(params_.sin_contiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
        }

        // 转换 rotate 为连续格式
        if (params_.rotate->IsContiguous()) {
            params_.rotate_contiguous = params_.rotate;
        } else {
            params_.rotate_contiguous = l0op::Contiguous(params_.rotate, l0Executor_);
            CHECK_RET(params_.rotate_contiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
        }

        return ACLNN_SUCCESS;
    }

public:
    /**
     * @brief 初始化 Handler
     * @param interfaceName 接口名称
     * @param params 参数结构体
     * @param workspaceSize 工作空间大小指针
     * @param executor 执行器指针的指针
     */
    void Initialize(const char *interfaceName, MatrixTransformRopeParamsBase &params,
                  uint64_t *workspaceSize, aclOpExecutor **executor)
    {
        interfaceName_ = interfaceName;
        params_ = params;
        workspaceSize_ = workspaceSize;
        executor_ = executor;
    }

    /**
     * @brief 处理流程入口
     * 执行完整的处理流程：参数校验 -> 数据转换 -> 执行算子 -> 拷贝输出
     * @return aclnnStatus 状态码
     */
    aclnnStatus Process()
    {
        // 固定写法，创建 OpExecutor
        auto uniqueExecutor = CREATE_EXECUTOR();
        CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
        l0Executor_ = uniqueExecutor.get();

        // 参数校验
        auto ret = CheckParams();
        CHECK_RET(ret == ACLNN_SUCCESS, ret);

        // 空 Tensor 场景处理
        if (params_.out->IsEmpty()) {
            *workspaceSize_ = 0ULL;
            uniqueExecutor.ReleaseTo(executor_);
            return ACLNN_SUCCESS;
        }

        // 数据转换为连续格式
        ret = ConvertDataContiguous();
        CHECK_RET(ret == ACLNN_SUCCESS, ret);

        // 执行算子（使用转换后的连续 tensor）
        auto outTensor = l0op::MatrixTransformRope(
            params_.x_contiguous, params_.cos_contiguous, params_.sin_contiguous,
            params_.rotate_contiguous,
            l0Executor_);
        CHECK_RET(outTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);

        // 拷贝输出到用户输出 tensor
        auto retView = l0op::ViewCopy(outTensor, params_.out, l0Executor_);
        CHECK_RET(retView != nullptr, ACLNN_ERR_INNER_NULLPTR);

        *workspaceSize_ = uniqueExecutor->GetWorkspaceSize();
        uniqueExecutor.ReleaseTo(executor_);
        return ACLNN_SUCCESS;
    }

protected:
    std::string interfaceName_;
    MatrixTransformRopeParamsBase params_;
    uint64_t *workspaceSize_;
    aclOpExecutor **executor_;
    aclOpExecutor *l0Executor_;
};

} // namespace matrix_transform_rope

#endif // OP_HOST_OP_API_ACLNN_MATRIX_TRANSFORM_ROPE_UTILS_H
