/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to License for details. You may not use this file except in compliance with License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_HOST_OP_API_ACLNN_MATRIX_TRANSFORM_ROPE_BASE_HANDLER_H
#define OP_HOST_OP_API_ACLNN_MATRIX_TRANSFORM_ROPE_BASE_HANDLER_H

#include "aclnn_matrix_transform_rope_utils.h"
#include "util/math_util.h"

namespace matrix_transform_rope_base {

using namespace matrix_transform_rope;
using namespace op;

/**
 * @brief SOC 特定常量定义
 */
constexpr int64_t DIM_IDX_0 = 0L;
constexpr int64_t DIM_IDX_1 = 1L;
constexpr int64_t DIM_IDX_2 = 2L;
constexpr int64_t DIM_IDX_3 = 3L;
constexpr size_t INPUT_DIM_LIMIT_X = 4UL;
constexpr size_t INPUT_DIM_LIMIT_ROTATE = 2UL;
constexpr size_t OUTPUT_DIM_LIMIT = 4UL;

/**
 * @brief 支持的数据类型列表
 */
const std::initializer_list<DataType> DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT16, DataType::DT_FLOAT, DataType::DT_BF16
};

/**
 * @brief 判断 format 是否为私有 format
 * @param format 待判断的 format
 * @return true 表示是私有 format，false 表示不是私有 format
 */
static bool IsPrivateFormat(ge::Format format)
{
    if (format == ge::FORMAT_NC1HWC0 || format == ge::FORMAT_FRACTAL_Z ||
        format == ge::FORMAT_NDC1HWC0 || format == ge::FORMAT_FRACTAL_Z_3D ||
        format == ge::FORMAT_FRACTAL_NZ || format == ge::FORMAT_NC1HWC0_C04) {
        return true;
    }
    return false;
}

/**
 * @brief 检查 cos/sin 的 shape 是否符合 x 的 shape 要求
 *
 * 规则：
 * - cos/sin 必须是 4 维
 * - cos/sin 的最后一维必须等于 x 的最后一维
 * - cos/sin 的倒数第二维必须等于 x 的对应维度
 * - cos/sin 的第一维可以是 1、或者等于 x 的对应维度
 *
 * @param xShape x 的 shape
 * @param csShape cos/sin 的 shape
 * @param tensorName tensor 名称（用于错误信息）
 * @return true 表示 shape 符合要求，false 表示不符合要求
 */
static bool CheckCosSinShape(const Shape &xShape, const Shape &csShape, const char *tensorName)
{
    auto xDimNum = xShape.GetDimNum();
    auto csDimNum = csShape.GetDimNum();

    // cos/sin 必须是 4 维
    if (csDimNum != INPUT_DIM_LIMIT_X) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "%s tensor dim num must be 4, but got %zu", tensorName, csDimNum);
        return false;
    }

    // 检查最后一维（D）必须一致
    int64_t xD = xShape.GetDim(DIM_IDX_3);
    int64_t csD = csShape.GetDim(DIM_IDX_3);
    if (csD != xD) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "%s tensor last dim must equal x tensor last dim: csD=%ld, xD=%ld",
            tensorName, csD, xD);
        return false;
    }

    // 检查第一维：可以是1，或者等于x的对应维度
    int64_t xFirstDim = xShape.GetDim(DIM_IDX_0);
    int64_t csFirstDim = csShape.GetDim(DIM_IDX_0);

    if (csFirstDim != 1 && csFirstDim != xFirstDim) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "%s tensor first dim must be 1 or equal to x first dim: csDim=%ld, xDim=%ld",
            tensorName, csFirstDim, xFirstDim);
        return false;
    }

    return true;
}

/**
 * @brief 基础 Handler 实现类
 *
 * 设计说明：
 * 1. 继承自 MatrixTransformRopeHandler 基类
 * 2. 实现具体的校验逻辑
 * 3. 针对 DAV_2201 版本进行定制
 */
class MatrixTransformRopeBaseHandler : public MatrixTransformRopeHandler {
protected:
    bool CheckInputOutDims() override
    {
        // x 必须是 4 维
        auto xDimNum = params_.x->GetViewShape().GetDimNum();
        if (xDimNum != INPUT_DIM_LIMIT_X) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x tensor dim num must be 4, but got %zu", xDimNum);
            return false;
        }

        // cos 必须是 4 维
        auto cosDimNum = params_.cos->GetViewShape().GetDimNum();
        if (cosDimNum != INPUT_DIM_LIMIT_X) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "cos tensor dim num must be 4, but got %zu", cosDimNum);
            return false;
        }

        // sin 必须是 4 维
        auto sinDimNum = params_.sin->GetViewShape().GetDimNum();
        if (sinDimNum != INPUT_DIM_LIMIT_X) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "sin tensor dim num must be 4, but got %zu", sinDimNum);
            return false;
        }

        // rotate 必须是 2 维，shape 为 [D, D]
        auto rotateDimNum = params_.rotate->GetViewShape().GetDimNum();
        if (rotateDimNum != INPUT_DIM_LIMIT_ROTATE) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "rotate tensor dim num must be 2, but got %zu", rotateDimNum);
            return false;
        }

        // out 必须是 4 维，与 x 保持一致
        auto outDimNum = params_.out->GetViewShape().GetDimNum();
        if (outDimNum != OUTPUT_DIM_LIMIT) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "out tensor dim num must be 4, but got %zu", outDimNum);
            return false;
        }

        return true;
    }

    bool CheckInputOutShape() override
    {
        auto xShape = params_.x->GetViewShape();
        auto cosShape = params_.cos->GetViewShape();
        auto sinShape = params_.sin->GetViewShape();
        auto rotateShape = params_.rotate->GetViewShape();
        auto outShape = params_.out->GetViewShape();

        // 检查 rotate 的 shape 必须是 [D, D]（2维）
        if (rotateShape.GetDim(DIM_IDX_0) != rotateShape.GetDim(DIM_IDX_1)) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "rotate tensor shape must be [D, D], but got [%ld, %ld]",
                rotateShape.GetDim(DIM_IDX_0), rotateShape.GetDim(DIM_IDX_1));
            return false;
        }

        // 检查 cos 的 shape 是否符合 x 的要求
        if (!CheckCosSinShape(xShape, cosShape, "cos")) {
            return false;
        }

        // 检查 sin 的 shape 是否符合 x 的要求
        if (!CheckCosSinShape(xShape, sinShape, "sin")) {
            return false;
        }

        // 检查 out 的 shape 是否与 x 一致
        if (xShape.GetDimNum() == outShape.GetDimNum()) {
            for (size_t i = 0; i < xShape.GetDimNum(); ++i) {
                if (xShape.GetDim(i) != outShape.GetDim(i)) {
                    OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                        "out tensor shape must be same as x tensor shape, but x dim[%zu]=%ld, out dim[%zu]=%ld",
                        i, xShape.GetDim(i), i, outShape.GetDim(i));
                    return false;
                }
            }
        }

        // 检查 x、cos、sin、rotate 的最后一维（D）必须一致
        int64_t xD = xShape.GetDim(DIM_IDX_3);
        int64_t cosD = cosShape.GetDim(DIM_IDX_3);
        int64_t sinD = sinShape.GetDim(DIM_IDX_3);
        int64_t rotateD = rotateShape.GetDim(DIM_IDX_2);
        int64_t outD = outShape.GetDim(DIM_IDX_3);

        if (cosD != xD) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "cos tensor last dim must equal x tensor last dim: cosD=%ld, xD=%ld", cosD, xD);
            return false;
        }
        if (sinD != xD) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "sin tensor last dim must equal x tensor last dim: sinD=%ld, xD=%ld", sinD, xD);
            return false;
        }
        if (rotateD != xD) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "rotate tensor dims must be [D, D], rotateD=%ld, xD=%ld", rotateD, xD);
            return false;
        }
        if (outD != xD) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "out tensor last dim must equal x tensor last dim: outD=%ld, xD=%ld", outD, xD);
            return false;
        }

        return true;
    }

    bool CheckDtypeValid() override
    {
        // 检查 x 的数据类型
        OP_CHECK_DTYPE_NOT_SUPPORT(params_.x, DTYPE_SUPPORT_LIST, return false);

        // 检查 cos 的数据类型
        OP_CHECK_DTYPE_NOT_SUPPORT(params_.cos, DTYPE_SUPPORT_LIST, return false);

        // 检查 sin 的数据类型
        OP_CHECK_DTYPE_NOT_SUPPORT(params_.sin, DTYPE_SUPPORT_LIST, return false);

        // 检查 rotate 的数据类型
        OP_CHECK_DTYPE_NOT_SUPPORT(params_.rotate, DTYPE_SUPPORT_LIST, return false);

        // 检查 out 的数据类型
        OP_CHECK_DTYPE_NOT_SUPPORT(params_.out, DTYPE_SUPPORT_LIST, return false);

        return true;
    }

    bool CheckFormat() override
    {
        // 检查 x 的 format 必须是 ND 格式
        if (IsPrivateFormat(params_.x->GetViewFormat())) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x tensor format must be ND");
            return false;
        }

        // 检查 cos 的 format 必须是 ND 格式
        if (IsPrivateFormat(params_.cos->GetViewFormat())) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "cos tensor format must be ND");
            return false;
        }

        // 检查 sin 的 format 必须是 ND 格式
        if (IsPrivateFormat(params_.sin->GetViewFormat())) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "sin tensor format must be ND");
            return false;
        }

        // 检查 rotate 的 format 必须是 ND 格式
        if (IsPrivateFormat(params_.rotate->GetViewFormat())) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "rotate tensor format must be ND");
            return false;
        }

        return true;
    }
};

} // namespace matrix_transform_rope_base

#endif // OP_HOST_OP_API_ACLNN_MATRIX_TRANSFORM_ROPE_BASE_HANDLER_H
