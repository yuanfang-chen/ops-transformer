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
 * \file matmul_allto_all_util_tiling.cpp
 * \brief
 */
#include "matmul_allto_all_util_tiling.h"

namespace MC2Tiling {

/**
 * @brief 工具函数：判断指定value是否存在于list中
 *
 * @param list: 有效值列表
 * @param value: 给定值
 * @return
 */
static bool IsContains(const std::vector<uint32_t> &list, uint32_t value)
{
    return std::count(list.begin(), list.end(), value) > 0;
}

/**
 * @brief 校验attrs信息
 *
 * @param context 框架根据input，output，attrs等信息生成tiling需要的context
 * @param opName  算子名称
 * @return ge::graphStatus
 */
ge::graphStatus MatmulAlltoAllTilingUtil::CheckAttrsInfo(const gert::TilingContext *context, const char *opName)
{
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE(opName, "Failed to get attrs."), return ge::GRAPH_FAILED);

    const char *group = attrs->GetAttrPointer<char>(ATTR_GROUP);
    const int *worldSize = attrs->GetAttrPointer<int>(ATTR_WORLD_SIZE);
    // 判断为空或者空字符串
    OP_TILING_CHECK(group == nullptr, OP_LOGE(opName, "The input attr group is null pointer."),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(group[0] == '\0', OP_LOGE(opName, "The input attr group is empty string."),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK((worldSize == nullptr), OP_LOGE(opName, "The input attr worldSize is null pointer."),
                    return ge::GRAPH_FAILED);
    int64_t rankDim = *worldSize;
    OP_TILING_CHECK(
        SUPPORT_RANK_SIZE.find(rankDim) == SUPPORT_RANK_SIZE.end(),
        OP_LOGE(opName, "World_size should be 2 or 4 or 8 or 16 or 32, but the actual value is %ld.", rankDim),
        return ge::GRAPH_FAILED);

    const bool *isTransX1 = attrs->GetAttrPointer<bool>(ATTR_X1_TRANSPOSE);
    bool x1TransposeFlag = (isTransX1 != nullptr) ? *isTransX1 : false;
    OP_TILING_CHECK(x1TransposeFlag, OP_LOGE(opName, "X1 transpose is not supported, should be false."),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 非量化场景校验参数的DType
 *
 * @param context 框架根据input，output，attrs等信息生成tiling需要的context
 * @param opName  算子名称
 * @return ge::graphStatus
 */
ge::graphStatus MatmulAlltoAllTilingUtil::CheckNonQuantTensorDataType(const gert::TilingContext *context,
                                                                      const char *opName)
{
    // 获取并校验输入张量描述符
    auto x1TensorDesc = context->GetInputDesc(INPUT_X1);
    OP_TILING_CHECK((x1TensorDesc == nullptr), OP_LOGE(opName, "The input tensor x1 is invalid."),
                    return ge::GRAPH_FAILED);
    auto x2TensorDesc = context->GetInputDesc(INPUT_X2);
    OP_TILING_CHECK((x2TensorDesc == nullptr), OP_LOGE(opName, "The input tensor x2 is invalid."),
                    return ge::GRAPH_FAILED);

    // 获取数据类型并校验一致性与范围
    ge::DataType x1Dtype = x1TensorDesc->GetDataType();
    ge::DataType x2Dtype = x2TensorDesc->GetDataType();
    OP_TILING_CHECK((x1Dtype != x2Dtype),
                    OP_LOGE(opName, "The Input x1 and x2 Dtype should be same, but x1 is %s, x2 is %s.",
                            Ops::Base::ToString(x1Dtype).c_str(), Ops::Base::ToString(x2Dtype).c_str()),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(!IsContains(NON_QUANT_X_DTYPE_LIST, x1Dtype),
                    OP_LOGE(opName,
                            "The Input x Dtype should be in non-quant range (float16/bf16), but x1 is %s, x2 is %s.",
                            Ops::Base::ToString(x1Dtype).c_str(), Ops::Base::ToString(x2Dtype).c_str()),
                    return ge::GRAPH_FAILED);

    // 校验 bias 数据类型（如果存在）
    auto biasTensorDesc = context->GetOptionalInputDesc(INPUT_BIAS);
    if (biasTensorDesc != nullptr) {
        ge::DataType biasDtype = biasTensorDesc->GetDataType();
        OP_TILING_CHECK((x1Dtype != biasDtype),
                        OP_LOGE(opName, "Bias Dtype should be same as x Dtype, but bias is %s.",
                                Ops::Base::ToString(biasDtype).c_str()),
                        return ge::GRAPH_FAILED);
    }

    // 校验 scale 张量为空（非量化场景）
    auto x1ScaleTensorDesc = context->GetOptionalInputDesc(INPUT_X1_SCALE);
    auto x2ScaleTensorDesc = context->GetOptionalInputDesc(INPUT_X2_SCALE);
    OP_TILING_CHECK((x1ScaleTensorDesc != nullptr || x2ScaleTensorDesc != nullptr),
                    OP_LOGE(opName, "Scale tensors should be null in non-quant mode."), return ge::GRAPH_FAILED);

    // 校验输出张量数据类型
    auto yDesc = context->GetOutputDesc(OUTPUT_Y);
    OP_TILING_CHECK((yDesc == nullptr), OP_LOGE(opName, "Output tensor y is nullptr."), return ge::GRAPH_FAILED);
    ge::DataType yDtype = yDesc->GetDataType();
    OP_TILING_CHECK((yDtype != x1Dtype),
                    OP_LOGE(opName, "Output y Dtype should be same as input x Dtype, but y is %s.",
                            Ops::Base::ToString(yDtype).c_str()),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 校验tiling inputshape非空
 *
 * @param context 框架根据input，output，attrs等信息生成tiling需要的context
 * @param opName 算子名称
 * @return ge::graphStatus
 */
static ge::graphStatus CheckInputShapesValid(const gert::TilingContext *context, const char *opName)
{
    const gert::StorageShape *x1Shape = context->GetInputShape(INPUT_X1);
    const gert::StorageShape *x2Shape = context->GetInputShape(INPUT_X2);
    OP_TILING_CHECK((x1Shape == nullptr) || (x2Shape == nullptr), OP_LOGE(opName, "The input shape is invalid"),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 校验tiling shape的Dim数量信息
 *
 * @param context 框架根据input，output，attrs等信息生成tiling需要的context
 * @param opName 算子名称
 * @return ge::graphStatus
 */
static ge::graphStatus CheckShapeDimensions(const gert::StorageShape *shape, const char *shapeName, const char *opName)
{
    uint64_t dimNum = shape->GetStorageShape().GetDimNum();
    OP_TILING_CHECK((dimNum != 2), OP_LOGE(opName, "The %s dimNum should be two.", shapeName), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 校验tiling输入的shape的合法性
 *
 * @param context 框架根据input，output，attrs等信息生成tiling需要的context
 * @param opName 算子名称
 * @return ge::graphStatus
 */
static ge::graphStatus CheckMatrixMulShapes(const gert::TilingContext *context, const char *opName)
{
    const gert::StorageShape *x1Shape = context->GetInputShape(INPUT_X1);
    const gert::StorageShape *x2Shape = context->GetInputShape(INPUT_X2);
    const gert::StorageShape *yShape = context->GetOutputShape(OUTPUT_Y);

    uint64_t x1Dim0 = x1Shape->GetStorageShape().GetDim(0);
    uint64_t x1Dim1 = x1Shape->GetStorageShape().GetDim(1);
    uint64_t x2Dim0 = x2Shape->GetStorageShape().GetDim(0);
    uint64_t x2Dim1 = x2Shape->GetStorageShape().GetDim(1);
    uint64_t yDim0 = yShape->GetStorageShape().GetDim(0);
    uint64_t yDim1 = yShape->GetStorageShape().GetDim(1);

    bool x2TransFlag = false;
    const bool *isTransX2 = context->GetAttrs()->GetAttrPointer<bool>(ATTR_X2_TRANSPOSE);
    if (isTransX2) {
        x2TransFlag = *isTransX2;
    }

    if (x2TransFlag) {
        OP_TILING_CHECK((x1Dim1 != x2Dim1),
                        OP_LOGE(opName,
                                "When x2 is transposed, the x1 second dim should be the same with the "
                                "second dim of x2, the x1 second dim is %lu, the x2 second dim is %lu.",
                                x1Dim1, x2Dim1),
                        return ge::GRAPH_FAILED);
        OP_TILING_CHECK(((x1Dim0 != yDim0) || (x2Dim0 != yDim1)),
                        OP_LOGE(opName,
                                "When x2 is transposed, the x1 first dim should be the same with the "
                                "first dim of y, the x2 first dim should be the same with the second dim of y. "
                                "x1Dim0: %lu, yDim0: %lu, x2Dim0: %lu, yDim1: %lu.",
                                x1Dim0, yDim0, x2Dim0, yDim1),
                        return ge::GRAPH_FAILED);
    } else {
        OP_TILING_CHECK((x1Dim1 != x2Dim0),
                        OP_LOGE(opName,
                                "The x1 second dim should be the same with the "
                                "first dim of x2, the x1 second dim is %lu, the x2 first dim is %lu.",
                                x1Dim1, x2Dim0),
                        return ge::GRAPH_FAILED);
        OP_TILING_CHECK(((x1Dim0 != yDim0) || (x2Dim1 != yDim1)),
                        OP_LOGE(opName,
                                "The x1 first dim should be the same with the "
                                "first dim of y, the x2 second dim should be the same with the second dim of y. "
                                "x1Dim0: %lu, yDim0: %lu, x2Dim1: %lu, yDim1: %lu.",
                                x1Dim0, yDim0, x2Dim1, yDim1),
                        return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 校验tiling输入的bias的shape信息
 *
 * @param context 框架根据input，output，attrs等信息生成tiling需要的context
 * @param opName 算子名称
 * @return ge::graphStatus
 */
static ge::graphStatus CheckBiasShape(const gert::TilingContext *context, const char *opName)
{
    const gert::StorageShape *biasShape = context->GetOptionalInputShape(INPUT_BIAS);
    if (biasShape != nullptr) {
        uint64_t biasShapeDimNum = biasShape->GetStorageShape().GetDimNum();
        OP_TILING_CHECK((biasShapeDimNum != 1), OP_LOGE(opName, "The input bias dimNum should be one."),
                        return ge::GRAPH_FAILED);
        uint64_t biasDim0 = biasShape->GetStorageShape().GetDim(0);
        const gert::StorageShape *x1Shape = context->GetInputShape(INPUT_X1);
        const gert::StorageShape *x2Shape = context->GetInputShape(INPUT_X2);
        uint64_t x1Dim1 = x1Shape->GetStorageShape().GetDim(1);
        uint64_t x2Dim0 = x2Shape->GetStorageShape().GetDim(0);
        uint64_t x2Dim1 = x2Shape->GetStorageShape().GetDim(1);
        uint64_t nAxis = (x1Dim1 == x2Dim0) ? x2Dim1 : x2Dim0;
        OP_TILING_CHECK((biasDim0 != nAxis),
                        OP_LOGE(opName, "The bias dimNum0 should be %lu, but actual value is %lu.", nAxis, biasDim0),
                        return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 校验tiling输入Input的Dim范围信息
 *
 * @param context 框架根据input，output，attrs等信息生成tiling需要的context
 * @param opName 算子名称
 * @return ge::graphStatus
 */
static ge::graphStatus CheckShapeDimRange(const gert::TilingContext *context, const char *opName)
{
    // attr非空在CheckAttrsInfo校验过
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    const int *rankSize = attrs->GetAttrPointer<int>(ATTR_WORLD_SIZE);

    const gert::StorageShape *x1Shape = context->GetInputShape(INPUT_X1);
    const gert::StorageShape *x2Shape = context->GetInputShape(INPUT_X2);
    uint64_t x1Dim0 = x1Shape->GetStorageShape().GetDim(0);
    uint64_t x1Dim1 = x1Shape->GetStorageShape().GetDim(1);
    uint64_t x2Dim0 = x2Shape->GetStorageShape().GetDim(0);
    uint64_t x2Dim1 = x2Shape->GetStorageShape().GetDim(1);
    uint64_t nAxis = (x1Dim1 == x2Dim0) ? x2Dim1 : x2Dim0;
    // 校验M,当前M为0的话，走公式化tiling切分实际是不支持的,后面可去除
    OP_TILING_CHECK(x1Dim0 == 0, OP_LOGE(opName, "Invalid x1 shape: dim 0(m) cannot be 0."), return ge::GRAPH_FAILED);
    // 校验K,K的范围应该在[1, 65535],且要整除rankSize
    OP_TILING_CHECK(x1Dim1 > K_MAX_VALUE, OP_LOGE(opName, "X1 dim 1(k) exceeds max value 65535, got %lu.", x1Dim1),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(x1Dim1 % static_cast<uint64_t>(*rankSize) != 0,
                    OP_LOGE(opName, "X1 dim 1 (%lu) is not divisible by rankSize (%ld).", x1Dim1, rankSize),
                    return ge::GRAPH_FAILED);
    // 校验N, N不为空，且要整除rankSize
    OP_TILING_CHECK(nAxis == 0, OP_LOGE(opName, "Invalid x2 shape: N cannot be 0."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(nAxis % static_cast<uint64_t>(*rankSize) != 0,
                    OP_LOGE(opName, "N (%lu) is not divisible by rankSize (%ld).", nAxis, rankSize),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 校验tiling输入的shape信息
 *
 * @param context 框架根据input，output，attrs等信息生成tiling需要的context
 * @param opName 算子名称
 * @return ge::graphStatus
 */
ge::graphStatus MatmulAlltoAllTilingUtil::CheckShapeInfo(const gert::TilingContext *context, const char *opName)
{
    ge::graphStatus status;

    // 校验输入Input Shape是否为空
    status = CheckInputShapesValid(context, opName);
    if (status != ge::GRAPH_SUCCESS)
        return status;

    // 校验维度数目是否合法
    const gert::StorageShape *x1Shape = context->GetInputShape(INPUT_X1);
    const gert::StorageShape *x2Shape = context->GetInputShape(INPUT_X2);
    status = CheckShapeDimensions(x1Shape, "input x1", opName);
    if (status != ge::GRAPH_SUCCESS)
        return status;
    status = CheckShapeDimensions(x2Shape, "input x2", opName);
    if (status != ge::GRAPH_SUCCESS)
        return status;

    // 校验输出
    const gert::StorageShape *yShape = context->GetOutputShape(OUTPUT_Y);
    OP_TILING_CHECK((yShape == nullptr), OP_LOGE(opName, "The yShape is nullptr."), return ge::GRAPH_FAILED);
    status = CheckShapeDimensions(yShape, "output y", opName);
    if (status != ge::GRAPH_SUCCESS)
        return status;

    // 校验shape的维度矩阵是否合法
    status = CheckMatrixMulShapes(context, opName);
    if (status != ge::GRAPH_SUCCESS)
        return status;

    // 校验bias的shape信息
    status = CheckBiasShape(context, opName);
    if (status != ge::GRAPH_SUCCESS)
        return status;

    // 校验shape的dim范围
    status = CheckShapeDimRange(context, opName);
    if (status != ge::GRAPH_SUCCESS)
        return status;
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 设置attrs信息，group， rank等
 *
 * @param context 框架根据input，output，attrs等信息生成tiling需要的context
 * @param opName  算子名称
 * @param contextInfo 过程信息
 * @return ge::graphStatus
 */
ge::graphStatus MatmulAlltoAllTilingUtil::SetAttrsInfo(const gert::TilingContext *context, const char *opName,
                                                       TilingContextInfo &contextInfo)
{
    // 在前面的check位置校验过
    const gert::RuntimeAttrs *attrs = context->GetAttrs();

    const char *group = attrs->GetAttrPointer<char>(ATTR_GROUP);
    const int *worldSize = attrs->GetAttrPointer<int>(ATTR_WORLD_SIZE);
    int64_t rankDim = 0;
    if ((worldSize == nullptr) || (*worldSize == 0)) {
        OP_TILING_CHECK(!mc2tiling::GetRankSize(opName, group, rankDim), OP_LOGE(opName, "GetRankSize failed."),
                        return ge::GRAPH_FAILED);
    } else {
        rankDim = *worldSize;
    }
    contextInfo.group = group;
    contextInfo.args_.rankDim = rankDim;
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 设置算子的shape信息
 *
 * @param context 框架根据input，output，attrs等信息生成tiling需要的context
 * @param contextInfo 存储了tiling的过程信息
 * @return ge::graphStatus
 */
ge::graphStatus MatmulAlltoAllTilingUtil::SetShapeInfo(const gert::TilingContext *context,
                                                       TilingContextInfo &contextInfo)
{
    const gert::StorageShape *x1Shape = context->GetInputShape(INPUT_X1);
    const gert::StorageShape *x2Shape = context->GetInputShape(INPUT_X2);
    uint64_t x1Dim0 = x1Shape->GetStorageShape().GetDim(0);
    uint64_t x1Dim1 = x1Shape->GetStorageShape().GetDim(1);
    uint64_t x2Dim0 = x2Shape->GetStorageShape().GetDim(0);
    uint64_t x2Dim1 = x2Shape->GetStorageShape().GetDim(1);

    contextInfo.args_.orgMValue = x1Dim0;
    contextInfo.args_.orgNValue = (x1Dim1 == x2Dim0) ? x2Dim1 : x2Dim0;
    contextInfo.args_.orgKValue = x1Dim1;

    // rank上的值
    contextInfo.args_.mValue = x1Dim0;
    contextInfo.args_.nValue = (x1Dim1 == x2Dim0) ? x2Dim1 : x2Dim0;
    contextInfo.args_.kValue = x1Dim1;
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 设置算子的数据类型信息
 *
 * @param context 框架根据input，output，attrs等信息生成tiling需要的context
 * @param opName 算子名称
 * @param contextInfo 存储了tiling的过程信息
 * @return ge::graphStatus
 */
ge::graphStatus MatmulAlltoAllTilingUtil::SetDataTypeInfo(const gert::TilingContext *context, const char *opName,
                                                          TilingContextInfo &contextInfo)
{
    const gert::StorageShape *matrixBias = context->GetOptionalInputShape(INPUT_BIAS);
    ge::DataType aType = context->GetInputDesc(INPUT_X1)->GetDataType();
    ge::DataType bType = context->GetInputDesc(INPUT_X2)->GetDataType();
    ge::DataType biasType;
    bool isBias = true;
    ge::DataType cType = aType;

    if (matrixBias == nullptr) {
        isBias = false;
        biasType = cType;
    } else {
        biasType = context->GetOptionalInputDesc(INPUT_BIAS)->GetDataType();
    }

    contextInfo.args_.inputDtypeSize = mc2tiling::GetDataTypeSize(opName, aType);
    contextInfo.args_.outputDtypeSize = mc2tiling::GetDataTypeSize(opName, cType);
    contextInfo.args_.isBias = isBias;
    contextInfo.args_.geAType = aType;
    contextInfo.args_.geBType = bType;
    contextInfo.args_.geCType = cType;
    contextInfo.args_.geBiasType = biasType;
    contextInfo.args_.aType = mc2tiling::ConvertGeTypeToMmType(opName, aType);
    contextInfo.args_.bType = mc2tiling::ConvertGeTypeToMmType(opName, bType);
    contextInfo.args_.cType = mc2tiling::ConvertGeTypeToMmType(opName, cType);
    contextInfo.args_.biasType = mc2tiling::ConvertGeTypeToMmType(opName, biasType);
    return ge::GRAPH_SUCCESS;
}


/**
 * @brief 功能函数：获取算子对应的QUANT类型
 *
 * @param context 框架根据input，output，attrs等信息生成tiling需要的context
 * @param opName 算子名称
 * @return QuantMode
 */
QuantMode MatmulAlltoAllTilingUtil::GetQuantMode(const gert::TilingContext *context, const char *opName)
{
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    if (attrs == nullptr) {
        OP_LOGE(opName, "Failed to get attrs.");
        return QuantMode::ERROR;
    }
    // 获取量化模式属性（默认为0，表示非量化）
    int x1QuantMode = 0;
    int x2QuantMode = 0;
    if (const int *ptr = attrs->GetAttrPointer<int>(ATTR_X1_QUANTMODE)) {
        x1QuantMode = *ptr;
    }
    if (const int *ptr = attrs->GetAttrPointer<int>(ATTR_X2_QUANTMODE)) {
        x2QuantMode = *ptr;
    }
    // 获取输入的x1,x2的数据类型
    auto x1TensorDesc = context->GetInputDesc(INPUT_X1);
    auto x2TensorDesc = context->GetInputDesc(INPUT_X2);
    if (x1TensorDesc == nullptr) {
        OP_LOGE(opName, "Input x1 tensor descriptor is invalid.");
        return QuantMode::ERROR; // 返回一个异常值，代表没有落入量化组合模式范围内
    }
    if (x2TensorDesc == nullptr) {
        OP_LOGE(opName, "Input x2 tensor descriptor is invalid.");
        return QuantMode::ERROR;
    }
    ge::DataType aType = x1TensorDesc->GetDataType();
    ge::DataType bType = x2TensorDesc->GetDataType();
    if (x1QuantMode == 0 && x2QuantMode == 0 && aType == bType && (aType == ge::DT_BF16 || aType == ge::DT_FLOAT16)) {
        return QuantMode::NON_QUANT;
    }
    // 当前只有两种场景，K-C量化的准入条件可以在这里添加
    return QuantMode::ERROR;
}

Mc2CcTilingConfigBuilder::Mc2CcTilingConfigBuilder(const std::string &groupName, uint32_t opType,
                                                   const std::string &algConfig)
    : mc2CcTilingConfig(groupName, opType, algConfig)
{
}

Mc2CcTilingConfigBuilder &Mc2CcTilingConfigBuilder::withReduceType(const char *opName, AscendC::HcclReduceOp reduceType,
                                                                   ge::DataType dstDataType, ge::DataType srcDataType)
{
    this->reduceType = static_cast<uint32_t>(reduceType);
    this->dstDataType = static_cast<uint8_t>(mc2tiling::ConvertGeTypeToHcclType(opName, dstDataType));
    this->srcDataType = static_cast<uint8_t>(mc2tiling::ConvertGeTypeToHcclType(opName, srcDataType));
    this->hasReduceType = true;
    return *this;
}

Mc2CcTilingConfigBuilder &Mc2CcTilingConfigBuilder::withStepSize(uint8_t stepSize)
{
    this->stepSize = stepSize;
    this->hasStepSize = true;
    return *this;
}

Mc2CcTilingConfigBuilder &Mc2CcTilingConfigBuilder::isLocalRankDataToLocalDst(bool flag)
{
    this->localRankDataToLocalDst = flag ? CONSTANTS_ONE : CONSTANTS_ZERO;
    this->hasLocalRankDataToLocalDst = true;
    return *this;
}

Mc2CcTilingConfigBuilder &Mc2CcTilingConfigBuilder::isSrcDataFromWindow(bool fromWindow)
{
    this->srcDataFromWindow = fromWindow ? CONSTANTS_TWO : CONSTANTS_ONE;
    this->hasSrcDataFromWindow = true;
    return *this;
}

Mc2CcTilingConfigBuilder &Mc2CcTilingConfigBuilder::withDebugMode(uint8_t debugMode)
{
    this->debugMode = debugMode;
    this->hasDebugMode = true;
    return *this;
}

Mc2CcTilingConfigBuilder &Mc2CcTilingConfigBuilder::withCommBlockNum(uint16_t commBlockNum)
{
    this->commBlockNum = commBlockNum;
    this->hasCommBlockNum = true;
    return *this;
}

Mc2CcTilingConfigBuilder &Mc2CcTilingConfigBuilder::withQueueNum(uint16_t queueNum)
{
    this->queueNum = queueNum;
    this->hasQueueNum = true;
    return *this;
}

Mc2CcTilingConfigBuilder &Mc2CcTilingConfigBuilder::withCommEngine(uint8_t commEngine)
{
    this->commEngine = commEngine;
    this->hasCommEngine = true;
    return *this;
}

AscendC::Mc2CcTilingConfig Mc2CcTilingConfigBuilder::build()
{
    if (this->hasReduceType) {
        if (this->SUCCESS != mc2CcTilingConfig.SetReduceType(reduceType, dstDataType, srcDataType)) {
            this->errorSet.insert(ConfigFile::REDUCE_TYPE);
        } else {
            this->errorSet.erase(ConfigFile::REDUCE_TYPE);
        }
    }
    //  setConfig
    auto setConfig = [this](const char *fieldName, bool has, const std::function<int()> &setter) {
        if (has) {
            if (this->SUCCESS != setter()) {
                this->errorSet.insert(fieldName);
            } else {
                this->errorSet.erase(fieldName);
            }
        }
    };

    setConfig(ConfigFile::STEP_SIZE, this->hasStepSize,
              [this]() { return this->mc2CcTilingConfig.SetStepSize(this->stepSize); });
    setConfig(ConfigFile::SKIP_LOCAL_RANK_COPY, this->hasLocalRankDataToLocalDst,
              [this]() { return this->mc2CcTilingConfig.SetSkipLocalRankCopy(this->localRankDataToLocalDst); });
    setConfig(ConfigFile::SKIP_BUFFER_WINDOW_COPY, this->hasSrcDataFromWindow,
              [this]() { return this->mc2CcTilingConfig.SetSkipBufferWindowCopy(this->srcDataFromWindow); });
    setConfig(ConfigFile::DEBUG_MODE, this->hasDebugMode,
              [this]() { return this->mc2CcTilingConfig.SetDebugMode(this->debugMode); });
    setConfig(ConfigFile::COMM_BLOCK_NUM, this->hasCommBlockNum,
              [this]() { return this->mc2CcTilingConfig.SetCommBlockNum(this->commBlockNum); });
    setConfig(ConfigFile::QUEUE_NUM, this->hasQueueNum,
              [this]() { return this->mc2CcTilingConfig.SetQueueNum(this->queueNum); });
    setConfig(ConfigFile::COMM_ENGINE, this->hasCommEngine,
              [this]() { return this->mc2CcTilingConfig.SetCommEngine(this->commEngine); });
    return this->mc2CcTilingConfig;
}

bool Mc2CcTilingConfigBuilder::isSuccess() const
{
    return this->errorSet.empty();
}

std::string Mc2CcTilingConfigBuilder::errorMsg() const
{
    std::string msg;
    for (auto &error : this->errorSet) {
        msg += error + ",";
    }
    return msg;
}

void Mc2CcTilingConfigBuilder::setInitTilingData(::Mc2InitTiling initTiling)
{
    if (!this->hasGotInit) {
        this->hasGotInit = true;
        mc2CcTilingConfig.GetTiling(initTiling);
    }
}

Mc2CcTilingConfigBuilder Mc2CcTilingConfigBuilder::create(const std::string &groupName, mc2tiling::AicpuComType opType,
                                                          const std::string &algConfig)
{
    return Mc2CcTilingConfigBuilder(groupName, static_cast<uint32_t>(opType), algConfig);
}


} // namespace MC2Tiling