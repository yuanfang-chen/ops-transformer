/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quant_reduce_scatter_util_tiling.cpp
 * \brief
 */
#include "quant_reduce_scatter_util_tiling.h"

namespace MC2Tiling {

/**
 * @brief 工具函数：判断指定value是否存在于list中
 * @param list: 有效值列表
 * @param value: 给定值
 * @return
 */
static bool IsContains(const std::vector<uint32_t> &list, uint32_t value)
{
    return std::find(list.begin(), list.end(), value) != list.end();
}

/**
 * @brief 校验attrs，并设置group
 * @param context: 框架根据input，output，attrs等信息生成tiling需要的context
 * @return
 */
static ge::graphStatus CheckAttrsInfo(const gert::TilingContext *context, TilingRunInfo &runInfo)
{
    const char *nodeName = context->GetNodeName();
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE(nodeName, "attrs is nullptr."), return ge::GRAPH_FAILED);
    // 校验group是否为空
    const char *groupPtr = attrs->GetAttrPointer<char>(GROUP_INDEX);
    OP_TILING_CHECK(groupPtr == nullptr, OP_LOGE(nodeName, "groupPtr is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(std::string(groupPtr).empty(), OP_LOGE(nodeName, "group should not be empty."),
                    return ge::GRAPH_FAILED);
    runInfo.group = std::string(groupPtr);
    // 校验reduce_op的类型是否为sum
    const char *reduceOpPtr = attrs->GetAttrPointer<char>(REDUCE_OP_INDEX);
    OP_TILING_CHECK(reduceOpPtr == nullptr, OP_LOGE(nodeName, "reduceOpPtr is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        std::strcmp(reduceOpPtr, REDUCE_OP_TYPE.c_str()) != 0,
        OP_LOGE(nodeName, "reduce_op type should be %s, but actual value is %s.", REDUCE_OP_TYPE.c_str(), reduceOpPtr),
        return ge::GRAPH_FAILED);
    // 校验output_dtype
    const int64_t *outputTypePtr = attrs->GetAttrPointer<int64_t>(OUTPUT_DTYPE_INDEX);
    OP_TILING_CHECK(outputTypePtr == nullptr, OP_LOGE(nodeName, "outputTypePtr is nullptr."), return ge::GRAPH_FAILED);
    ge::DataType outputType = static_cast<ge::DataType>(*outputTypePtr);
    OP_TILING_CHECK(!IsContains(OUTPUT_DTYPE_LIST, outputType),
                    OP_LOGE(nodeName, "outPutType should be bfloat16/float/float16, but actual value is %s.",
                            Ops::Base::ToString(outputType).c_str()),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 设置rankSize
 * @param context: 框架根据input，output，attrs等信息生成tiling需要的context
 * @param runInfo: 封装的doTiling所需要的参数
 * @return
 */
static ge::graphStatus SetRankSize(const gert::TilingContext *context, TilingRunInfo &runInfo)
{
    const char *nodeName = context->GetNodeName();
    uint32_t rankSize = mc2tiling::MatmulFormulaicTiling::GetRankSize(runInfo.group.c_str());
    OP_TILING_CHECK(!IsContains(RANK_SIZE_LIST, rankSize),
                    OP_LOGE(nodeName, "The rankSize should be in [2, 4, 8], but actual value is %u.", rankSize),
                    return ge::GRAPH_FAILED);
    // 设置rankSize
    runInfo.rankSize = rankSize;
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 设置量化模式
 * @param context: 框架根据input，output，attrs等信息生成tiling需要的context
 * @param runInfo: 封装的doTiling所需要的参数
 * @return
 */
static bool SetQuantMode(const gert::TilingContext *context, TilingRunInfo &runInfo)
{
    const char *nodeName = context->GetNodeName();
    // context->GetInputDesc在函数CheckTensorDataType中已经校验
    ge::DataType xDtype = context->GetInputDesc(X_INDEX)->GetDataType();
    ge::DataType scalesDtype = context->GetInputDesc(SCALES_INDEX)->GetDataType();
    // 0: 无量化模式; 1: TG量化; 2: MX量化
    uint32_t quantMode = 0;
    if (IsContains(X_DTYPE_LIST, xDtype) && scalesDtype == ge::DT_FLOAT) {
        quantMode = TG_QUANT_MOD;
    } else if ((xDtype == ge::DT_FLOAT8_E4M3FN || xDtype == ge::DT_FLOAT8_E5M2) && scalesDtype == ge::DT_FLOAT8_E8M0) {
        quantMode = MX_QUANT_MOD;
    }
    OP_TILING_CHECK(!static_cast<bool>(quantMode),
                    OP_LOGE(nodeName, "x dataType is %s and scale dataType is %s do not match any quantMode.",
                            Ops::Base::ToString(xDtype).c_str(), Ops::Base::ToString(scalesDtype).c_str()),
                    return false);
    // 设置quantMode
    runInfo.quantMode = quantMode;
    return true;
}

/**
 * @brief 校验所有参数的dtype，并设置量化模式
 * @param context: 框架根据input，output，attrs等信息生成tiling需要的context
 * @param runInfo: 封装的doTiling所需要的参数
 * @return
 */
static bool CheckTensorDataType(const gert::TilingContext *context, TilingRunInfo &runInfo)
{
    const char *nodeName = context->GetNodeName();
    // 校验x的dtype
    auto xDesc = context->GetInputDesc(X_INDEX);
    OP_TILING_CHECK(xDesc == nullptr, OP_LOGE(nodeName, "xDesc is null."), return false);
    ge::DataType xDtype = context->GetInputDesc(X_INDEX)->GetDataType();
    OP_TILING_CHECK(!IsContains(X_DTYPE_LIST, xDtype),
                    OP_LOGE(nodeName,
                            "x dataType should be int8/hifloat8/float8_e4m3fn/float8_e5m2, but actual value is %s.",
                            Ops::Base::ToString(xDtype).c_str()),
                    return false);
    // 校验scales的dtype
    auto scalesDesc = context->GetInputDesc(SCALES_INDEX);
    OP_TILING_CHECK(scalesDesc == nullptr, OP_LOGE(nodeName, "scalesDesc is null"), return false);
    ge::DataType scalesDtype = context->GetInputDesc(SCALES_INDEX)->GetDataType();
    OP_TILING_CHECK(!IsContains(SCALES_DTYPE_LIST, scalesDtype),
                    OP_LOGE(nodeName, "scale dataType should be float/float8_e8m0, but actual value is %s.",
                            Ops::Base::ToString(scalesDtype).c_str()),
                    return false);
    // 校验output的dtype
    auto outputDesc = context->GetOutputDesc(OUTPUT_INDEX);
    OP_TILING_CHECK(outputDesc == nullptr, OP_LOGE(nodeName, "OutputDesc is null."), return false);
    ge::DataType outputType = outputDesc->GetDataType();
    OP_TILING_CHECK(!IsContains(OUTPUT_DTYPE_LIST, outputType),
                    OP_LOGE(nodeName, "output dataType should be float16/bfloat16/float, but actual value is %s.",
                            Ops::Base::ToString(outputType).c_str()),
                    return false);
    // 设置量化模式
    OP_TILING_CHECK(!SetQuantMode(context, runInfo), OP_LOGE(nodeName, "get quantMode error."), return false);
    return true;
}

/**
 * @brief 校验x的维度
 * @param context: 框架根据input，output，attrs等信息生成tiling需要的context
 * @param runInfo: 封装的doTiling所需要的参数
 * @param opType: 当前op类型
 * @return
 */
static bool CheckXDim(const gert::TilingContext *context, TilingRunInfo &runInfo, const OpType opType)
{
    const char *nodeName = context->GetNodeName();
    // context->GetInputShape在函数CheckInputTensorDim中已经校验
    size_t xDimNum = context->GetInputShape(X_INDEX)->GetStorageShape().GetDimNum();
    // quant_reduce_scatter算子的x一定是2维，即x.shape(bs, h)
    bool inValidDimNum = xDimNum != TWO_DIMS;
    if (opType == OpType::OP_QUANT_ALL_REDUCE) {
        // quant_all_reduce算子的x可能是2维或者3维，即x.shape(b, s, h)
        inValidDimNum = inValidDimNum && (xDimNum != THREE_DIMS);
    }
    OP_TILING_CHECK(inValidDimNum, OP_LOGE(nodeName, "xDimNum is invalid, the actual input xDimNum is %lu.", xDimNum),
                    return false);
    // context->GetInputShape在CheckInputTensorDim函数中已经校验
    uint64_t xValueOne = context->GetInputShape(X_INDEX)->GetStorageShape().GetDim(DIM_ZERO);
    uint64_t xValueTwo = context->GetInputShape(X_INDEX)->GetStorageShape().GetDim(DIM_ONE);
    uint64_t scalesValueOne = context->GetInputShape(SCALES_INDEX)->GetStorageShape().GetDim(DIM_ZERO);
    uint64_t scalesValueTwo = context->GetInputShape(SCALES_INDEX)->GetStorageShape().GetDim(DIM_ONE);
    // 校验x第1维
    OP_TILING_CHECK(xValueOne != scalesValueOne,
                    OP_LOGE(nodeName, "dim1 of scales %lu is not equal to x %lu.", scalesValueOne, xValueOne),
                    return false);
    // 校验是否为空tensor
    bool emptyTensor = xValueOne == 0 || xValueTwo == 0 || scalesValueOne == 0 || scalesValueTwo == 0;
    // bs需要整除worldSize。只有x是2维时，当前轴才是b*s，当x是3维时，当前轴是b
    uint64_t xValueBS = xValueOne;
    // 泛化场景下h必须是128的倍数。只有x是2维时，当前轴才是h。当x是3维时，当前轴是s，后一个轴才是h
    uint64_t xValueH = xValueTwo;
    if (opType == OpType::OP_QUANT_ALL_REDUCE && xDimNum == THREE_DIMS) {
        uint64_t xValueThree = context->GetInputShape(X_INDEX)->GetStorageShape().GetDim(DIM_TWO);
        uint64_t scalesValueThree = context->GetInputShape(SCALES_INDEX)->GetStorageShape().GetDim(DIM_TWO);
        emptyTensor = emptyTensor || (xValueThree == 0 || scalesValueThree == 0);
        xValueBS = xValueOne * xValueTwo;
        xValueH = context->GetInputShape(X_INDEX)->GetStorageShape().GetDim(DIM_TWO);
        OP_TILING_CHECK(xValueTwo != scalesValueTwo,
                        OP_LOGE(nodeName, "dim2 of scales %lu is not equal to x %lu.", scalesValueTwo, xValueTwo),
                        return false);
    }
    OP_TILING_CHECK(emptyTensor, OP_LOGE(nodeName, "x and scale should not be empty tensor."), return false);
    OP_TILING_CHECK(xValueBS % runInfo.rankSize != 0,
                    OP_LOGE(nodeName,
                            "x b*s dim should be multiple of ranksize, but actual x b*s dim is %lu, ranksize is %u.",
                            xValueBS, runInfo.rankSize),
                    return false);
    if (opType == OpType::OP_QUANT_ALL_REDUCE) {
        // quant_all_reduce算子的h必须在[1024, 8192]之间，且能被128整除
        OP_TILING_CHECK(
            xValueH < H_VALUE_LOWER_LIMIT || xValueH > H_VALUE_UPPER_LIMIT || xValueH % TG_QUANT_NUMBER != 0,
            OP_LOGE(nodeName,
                    "x h dim is invalid, which should be in [1024, 8192] and 128 multiple, but actual value is %lu.",
                    xValueH),
            return false);
    } else {
        // quant_reduce_scatter算子的h必须为5120和7168
        OP_TILING_CHECK(
            xValueH != H_VALUE_5120 && xValueH != H_VALUE_7168,
            OP_LOGE(nodeName, "x h dim is invalid, which should be 5120 or 7168, but actual value is %lu.", xValueH),
            return false);
    }
    return true;
}

/**
 * @brief 根据量化模式校验scales的维度
 * @param context: 框架根据input，output，attrs等信息生成tiling需要的context
 * @param runInfo: 封装的doTiling所需要的参数
 * @return
 */
static bool CheckScalesDim(const gert::TilingContext *context, TilingRunInfo &runInfo, const OpType opType)
{
    const char *nodeName = context->GetNodeName();
    // x和scales的context->GetInputShape在CheckInputTensorDim函数中校验过
    uint64_t xValueH = context->GetInputShape(X_INDEX)->GetStorageShape().GetDim(DIM_ONE);
    uint64_t scalesValueH = context->GetInputShape(SCALES_INDEX)->GetStorageShape().GetDim(DIM_ONE);
    size_t scalesDim = context->GetInputShape(SCALES_INDEX)->GetStorageShape().GetDimNum();
    if (runInfo.quantMode == TG_QUANT_MOD) {
        // TG量化: scales.shape(bs, h/128)或(b, s, h/128)
        bool invalidScalesDim = scalesDim != TWO_DIMS;
        if (opType == OpType::OP_QUANT_ALL_REDUCE) {
            invalidScalesDim = invalidScalesDim && (scalesDim != THREE_DIMS);
        }
        OP_TILING_CHECK(invalidScalesDim,
                        OP_LOGE(nodeName, "In TG quantmode, scalesDim should be 2 or 3, but actual value is %lu.", scalesDim),
                        return false);
        if (opType == OpType::OP_QUANT_ALL_REDUCE && scalesDim == THREE_DIMS) {
            xValueH = context->GetInputShape(X_INDEX)->GetStorageShape().GetDim(DIM_TWO);
            scalesValueH = context->GetInputShape(SCALES_INDEX)->GetStorageShape().GetDim(DIM_TWO);
        }
        OP_TILING_CHECK(ops::CeilDiv(xValueH, TG_QUANT_NUMBER) != scalesValueH,
                        OP_LOGE(nodeName,
                                "In TG quantmode, scales last dim should be equal to x divided by 128, but actual x "
                                "last dim is %lu, scales last dim is %lu.",
                                xValueH, scalesValueH), return false);
    } else if (runInfo.quantMode == MX_QUANT_MOD) {
        // MX量化: scales.shape(bs, h/64, 2)或(b, s, h/64, 2)
        bool invalidScalesDim = scalesDim != THREE_DIMS;
        if (opType == OpType::OP_QUANT_ALL_REDUCE) {
            invalidScalesDim = invalidScalesDim && (scalesDim != FOUR_DIMS);
        }
        OP_TILING_CHECK(
            invalidScalesDim, OP_LOGE(nodeName, "In MX quantmode, scaleDim should be 3 or 4, but actual value is %lu.", scalesDim),
            return false);
        uint64_t scalesValueLast = context->GetInputShape(SCALES_INDEX)->GetStorageShape().GetDim(DIM_TWO);
        if (opType == OpType::OP_QUANT_ALL_REDUCE && scalesDim == FOUR_DIMS) {
            xValueH = context->GetInputShape(X_INDEX)->GetStorageShape().GetDim(DIM_TWO);
            scalesValueH = context->GetInputShape(SCALES_INDEX)->GetStorageShape().GetDim(DIM_TWO);
            scalesValueLast = context->GetInputShape(SCALES_INDEX)->GetStorageShape().GetDim(DIM_THREE);
        }
        // 校验scales最后一维一定为2
        OP_TILING_CHECK(scalesValueLast != MX_SCALE_LAST_DIM,
                        OP_LOGE(nodeName, "In MX quantmode, scales last dim should be 2, but actual value is %lu.",
                                scalesValueLast), return false);
        OP_TILING_CHECK(ops::CeilDiv(xValueH, MX_QUANT_NUMBER) != scalesValueH,
                        OP_LOGE(nodeName,
                                "In MX quantmode, scales h dim should be equal to x divided by 64, "
                                "but actual x h dim is %lu, scales h dim is %lu.",
                                xValueH, scalesValueH), return false);
    }
    return true;
}

/**
 * @brief 校验所有入参的维度
 * @param context: 框架根据input，output，attrs等信息生成tiling需要的context
 * @param runInfo: 封装的doTiling所需要的参数
 * @param opType:当前op类型
 * @return
 */
static bool CheckInputTensorDim(const gert::TilingContext *context, TilingRunInfo &runInfo, const OpType opType)
{
    const char *nodeName = context->GetNodeName();
    // 校验xDim
    const gert::StorageShape *xShape = context->GetInputShape(X_INDEX);
    OP_TILING_CHECK(xShape == nullptr, OP_LOGE(nodeName, "xShape is null."), return false);
    OP_TILING_CHECK(!CheckXDim(context, runInfo, opType), OP_LOGE(nodeName, "x dimensions is invalid."), return false);
    // 校验scalesDim
    const gert::StorageShape *scalesShape = context->GetInputShape(SCALES_INDEX);
    OP_TILING_CHECK(scalesShape == nullptr, OP_LOGE(nodeName, "scaleShape is null."), return false);
    // scalesDim根据量化模式判断
    OP_TILING_CHECK(!CheckScalesDim(context, runInfo, opType),
                    OP_LOGE(nodeName, "scales dimensions is invalid in the quantmode."), return false);
    return true;
}

/**
 * @brief 校验output的维度
 * @param context: 框架根据input，output，attrs等信息生成tiling需要的context
 * @param runInfo: 封装的doTiling所需要的参数
 * @return
 */
static bool CheckOutputDim(const gert::TilingContext *context, TilingRunInfo &runInfo, const OpType opType)
{
    const char *nodeName = context->GetNodeName();
    // context->GetOutputShape在函数CheckOutputTensorDim中已经校验
    const gert::StorageShape *outputShape = context->GetOutputShape(OUTPUT_INDEX);
    size_t outputDim = outputShape->GetStorageShape().GetDimNum();
    // output的shape与x对应，quant_reduce_scatter算子的output一定是2维
    bool invalidOutputDim = outputDim != TWO_DIMS;
    if (opType == OpType::OP_QUANT_ALL_REDUCE) {
        // quant_all_reduce算子的output可能是2维或者3维
        invalidOutputDim = invalidOutputDim && (outputDim != THREE_DIMS);
    }
    OP_TILING_CHECK(invalidOutputDim,
                    OP_LOGE(nodeName, "outputShape is invalid, the actual input outputDim is %lu", outputDim),
                    return false);
    // 检查x和output维度关系，output.shape = x.shape
    uint64_t outputValueOne = outputShape->GetStorageShape().GetDim(DIM_ZERO);
    uint64_t outputValueTwo = outputShape->GetStorageShape().GetDim(DIM_ONE);
    // context->GetInputShape在CheckInputTensorDim函数中校验过
    uint64_t xValueOne = context->GetInputShape(X_INDEX)->GetStorageShape().GetDim(DIM_ZERO);
    uint64_t xValueTwo = context->GetInputShape(X_INDEX)->GetStorageShape().GetDim(DIM_ONE);
    bool invalidShape = xValueTwo != outputValueTwo;
    if (opType == OpType::OP_QUANT_ALL_REDUCE) {
        invalidShape = invalidShape || (xValueOne != outputValueOne);
    } else {
        invalidShape = invalidShape || (xValueOne / runInfo.rankSize != outputValueOne);
    }
    // quant_all_reduce算子的output可能需要校验第3维
    if (opType == OpType::OP_QUANT_ALL_REDUCE && outputDim == THREE_DIMS) {
        uint64_t outputValueThree = outputShape->GetStorageShape().GetDim(DIM_TWO);
        uint64_t xValueThree = context->GetInputShape(X_INDEX)->GetStorageShape().GetDim(DIM_TWO);
        OP_LOGI(nodeName, "output dim2 is %lu, x dim2 is %lu", outputValueThree, xValueThree);
        invalidShape = invalidShape || (xValueThree != outputValueThree);
    }
    OP_TILING_CHECK(invalidShape,
                    OP_LOGE(nodeName,
                            "output shape is invalid, which was caculated with x shape,"
                            "actual output shape is (%lu, %lu), x shape is (%lu, %lu), rankSize is %u.",
                            outputValueOne, outputValueTwo, xValueOne, xValueTwo, runInfo.rankSize),
                    return false);
    return true;
}

/**
 * @brief 校验所有出参的维度
 * @param context: 框架根据input，output，attrs等信息生成tiling需要的context
 * @param runInfo: 封装的doTiling所需要的参数
 * @param opType: 当前op类型
 * @return
 */
static bool CheckOutputTensorDim(const gert::TilingContext *context, TilingRunInfo &runInfo, const OpType opType)
{
    const char *nodeName = context->GetNodeName();
    // 红线校验
    const gert::StorageShape *outputShape = context->GetOutputShape(OUTPUT_INDEX);
    OP_TILING_CHECK(outputShape == nullptr, OP_LOGE(nodeName, "The outputShape is null."), return false);
    return CheckOutputDim(context, runInfo, opType);
}

/**
 * @brief 校验所有参数的format
 * @param context: 框架根据input，output，attrs等信息生成tiling需要的context
 * @return
 */
static bool CheckTensorFormat(const gert::TilingContext *context)
{
    const char *nodeName = context->GetNodeName();
    // context->GetInputDesc在CheckTensorDataType函数中已经校验
    auto xDesc = context->GetInputDesc(X_INDEX);
    ge::Format xFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(xDesc->GetStorageFormat()));
    OP_TILING_CHECK(
        xFormat != ge::FORMAT_ND,
        OP_LOGE(nodeName, "x format should be ND, but actual value is %s.", Ops::Base::ToString(xFormat).c_str()),
        return false);
    auto scalesDesc = context->GetInputDesc(SCALES_INDEX);
    ge::Format scalesFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(scalesDesc->GetStorageFormat()));
    OP_TILING_CHECK(scalesFormat != ge::FORMAT_ND,
                    OP_LOGE(nodeName, "scale format should be ND, but actual value is %s.",
                            Ops::Base::ToString(scalesFormat).c_str()),
                    return false);
    // context->GetOutputDesc在CheckTensorDataType函数中已经校验
    auto outputDesc = context->GetOutputDesc(OUTPUT_INDEX);
    ge::Format outPutFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(outputDesc->GetStorageFormat()));
    OP_TILING_CHECK(outPutFormat != ge::FORMAT_ND,
                    OP_LOGE(nodeName, "output format should be ND, but actual value is %s.",
                            Ops::Base::ToString(outPutFormat).c_str()),
                    return false);
    return true;
}

/**
 * @brief 校验win区大小
 * @param context: 框架根据input，output，attrs等信息生成tiling需要的context
 * @param runInfo: 封装的doTiling所需要的参数
 * @return
 */
static bool CheckWindowSize(const gert::TilingContext *context, const TilingRunInfo &runInfo)
{
    const char *nodeName = context->GetNodeName();
    // 获取量化模式，数据类型
    uint64_t xValueOne = context->GetInputShape(X_INDEX)->GetStorageShape().GetDim(DIM_ZERO);
    uint64_t xValueTwo = context->GetInputShape(X_INDEX)->GetStorageShape().GetDim(DIM_ONE);
    uint64_t scalesValueOne = context->GetInputShape(SCALES_INDEX)->GetStorageShape().GetDim(DIM_ZERO);
    uint64_t scalesValueTwo = context->GetInputShape(SCALES_INDEX)->GetStorageShape().GetDim(DIM_ONE);
    uint64_t xDataSize =
        ((xValueOne * xValueTwo * X_DTYPE_SIZE_ONE + WIN_ADDR_ALIGN - 1UL) / WIN_ADDR_ALIGN) * WIN_ADDR_ALIGN;
    uint64_t actualWinSize = 0UL;
    if (runInfo.quantMode == TG_QUANT_MOD) {
        uint64_t scalesSize = scalesValueOne * scalesValueTwo * SCALE_DTYPE_SIZE_FOUR;
        uint64_t scalesDataSize = ((scalesSize + WIN_ADDR_ALIGN - 1UL) / WIN_ADDR_ALIGN) * WIN_ADDR_ALIGN;
        // 数据区（x和scales）+ 状态区（1Mb）
        actualWinSize = xDataSize + scalesDataSize + MB_SIZE;
    } else if (runInfo.quantMode == MX_QUANT_MOD) {
        uint64_t scalesValueThree = context->GetInputShape(SCALES_INDEX)->GetStorageShape().GetDim(DIM_TWO);
        uint64_t scalesSize = scalesValueOne * scalesValueTwo * scalesValueThree * SCALE_DTYPE_SIZE_ONE;
        uint64_t scalesDataSize = ((scalesSize + WIN_ADDR_ALIGN - 1UL) / WIN_ADDR_ALIGN) * WIN_ADDR_ALIGN;
        actualWinSize = xDataSize + scalesDataSize + MB_SIZE;
    }
    uint64_t maxWinSize = mc2tiling::Mc2TilingUtils::GetMaxWindowSize();
    OP_TILING_CHECK(actualWinSize > maxWinSize,
                    OP_LOGE(nodeName, "actual winSize %lu is bigger than max winSize %lu.", actualWinSize, maxWinSize),
                    return false);
    return true;
}

/**
 * @brief 设置workspace
 * @param context: 框架根据input，output，attrs等信息生成tiling需要的context
 * @return
 */
static ge::graphStatus SetWorkSpace(gert::TilingContext *context)
{
    const char *nodeName = context->GetNodeName();
    size_t *workSpaces = context->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workSpaces == nullptr, OP_LOGE(nodeName, "workSpaces is nullptr."), return ge::GRAPH_FAILED);
    workSpaces[0] = SYSTEM_NEED_WORKSPACE;
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 校验socVersion
 * @param context: 框架根据input，output，attrs等信息生成tiling需要的context
 * @return
 */
ge::graphStatus QuantReduceScatterUtilTiling::CheckSocVersion(const gert::TilingContext *context)
{
    const char *nodeName = context->GetNodeName();
    // 校验socVersion
    fe::PlatFormInfos *platformInfoPtr = context->GetPlatformInfo();
    OP_TILING_CHECK(platformInfoPtr == nullptr, OP_LOGE(nodeName, "platformInfoPtr is null."), return ge::GRAPH_FAILED);
    platform_ascendc::PlatformAscendC ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    platform_ascendc::SocVersion socVersion = ascendcPlatform.GetSocVersion();
    OP_TILING_CHECK(socVersion != platform_ascendc::SocVersion::ASCEND910_95,
                    OP_LOGE(nodeName, "socVersion needed to be 910_95."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 功能函数：quant_all_reduce tiling部分总的校验函数
 * @param context: 框架根据input，output，attrs等信息生成tiling需要的context
 * @param runInfo: 封装的doTiling所需要的参数
 * @param opType: 当前op类型
 * @return
 */
ge::graphStatus QuantReduceScatterUtilTiling::CheckTilingFunc(gert::TilingContext *context, TilingRunInfo &runInfo,
                                                              const OpType opType)
{
    const char *nodeName = context->GetNodeName();
    // set group
    OP_TILING_CHECK(CheckAttrsInfo(context, runInfo) != ge::GRAPH_SUCCESS, OP_LOGE(nodeName, "attrs are invalied."),
                    return ge::GRAPH_FAILED);
    // set rankSize
    OP_TILING_CHECK(SetRankSize(context, runInfo) != ge::GRAPH_SUCCESS, OP_LOGE(nodeName, "set rankSize failed."),
                    return ge::GRAPH_FAILED);
    // set quantMode
    OP_TILING_CHECK(!CheckTensorDataType(context, runInfo), OP_LOGE(nodeName, "tensor datatype is invalid."),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(!CheckInputTensorDim(context, runInfo, opType), OP_LOGE(nodeName, "input tensor dim is invalid."),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(!CheckOutputTensorDim(context, runInfo, opType), OP_LOGE(nodeName, "output tensor dim is invalid."),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(!CheckTensorFormat(context), OP_LOGE(nodeName, "tensor format is invalid."),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(!CheckWindowSize(context, runInfo), OP_LOGE(nodeName, "HCCL_BUFFSIZE is too small."),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(SetWorkSpace(context) != ge::GRAPH_SUCCESS, OP_LOGE(nodeName, "set workspace failed."),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

} // namespace MC2Tiling
