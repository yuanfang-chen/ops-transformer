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
 * \file moe_distribute_dispatch_infer.cpp
 * \brief
 */
#include "runtime_util.h"
#include "mc2_log.h"
#include "platform/platform_info.h"
#include "runtime/rt_external_base.h"
#include "platform/soc_spec.h"

using namespace ge;
namespace ops {
static constexpr size_t DIM_ONE = 1UL;
static constexpr size_t DIM_TWO = 2UL;
static constexpr int64_t NEG_ONE = -1;
static constexpr int64_t RANK_NUM_PER_NODE = 8;

static constexpr size_t DISPATCH_INPUT_X_INDEX = 0;
static constexpr size_t DISPATCH_INPUT_EXPERT_IDX_INDEX = 1;
static constexpr size_t DISPATCH_INPUT_SCALES_IDX_INDEX = 2;
static constexpr size_t DISPATCH_INPUT_EXPERT_SCALES_IDX_INDEX = 4;
static constexpr size_t DISPATCH_OUTPUT_EXPAND_X_INDEX = 0;
static constexpr size_t DISPATCH_OUTPUT_DYNAMIC_SCALES_INDEX = 1;
static constexpr size_t DISPATCH_OUTPUT_EXPAND_IDX_INDEX = 2;
static constexpr size_t DISPATCH_OUTPUT_EXPERT_TOKEN_NUMS_INDEX = 3;
static constexpr size_t DISPATCH_OUTPUT_EP_RECV_COUNTS_INDEX = 4;
static constexpr size_t DISPATCH_OUTPUT_TP_RECV_COUNTS_INDEX = 5;
static constexpr size_t DISPATCH_OUTPUT_EXPAND_SCALES = 6;
static constexpr size_t DISPATCH_INPUT_ATTR_EP_WORLD_SIZE_INDEX = 1;
static constexpr size_t DISPATCH_INPUT_ATTR_EP_RANK_ID_INDEX = 2;
static constexpr size_t DISPATCH_INPUT_ATTR_MOE_EXPERT_NUM_INDEX = 3;
static constexpr size_t DISPATCH_INPUT_ATTR_TP_WORLD_SIZE_INDEX = 5;
static constexpr size_t DISPATCH_INPUT_ATTR_TP_RANK_ID_INDEX = 6;
static constexpr size_t DISPATCH_INPUT_ATTR_EXPERT_SHARD_TYPE_INDEX = 7;
static constexpr size_t DISPATCH_INPUT_ATTR_SHARED_EXPERT_RANK_NUM_INDEX = 9;
static constexpr size_t DISPATCH_INPUT_ATTR_QUANT_MODE_INDEX = 10;
static constexpr size_t DISPATCH_INPUT_ATTR_GLOBAL_BS_INDEX = 11;

static constexpr uint32_t SOC_VERSION_SIZE = 32;

// 原有函数，完全保留
bool IsPlatform910B(const char *nodeName)
{
    char versionValVersion[SOC_VERSION_SIZE];
    // rtGetSocSpec获取成功返回值是0，获取失败返回非0
    if (rtGetSocSpec("version", "Short_SoC_version", versionValVersion, SOC_VERSION_SIZE) != RT_ERROR_NONE) {
        OPS_LOG_E(nodeName, "Cannot get Short_SoC_version info in infershape!");
        return false;
    }
    static std::set<std::string> supportedSoc = {"Ascend910B"};
    OPS_LOG_D(nodeName, "Get Short_SoC_version %s", versionValVersion);
    return (supportedSoc.count(versionValVersion) > 0);
}

// 子函数1：处理专家ID相关Shape + 基础参数计算（~40行，<50）
static ge::graphStatus InferExpertIdsShape(gert::InferShapeContext *context, 
    int64_t& bs, int64_t& h, int64_t& k, int64_t& globalBsReal,
    const int64_t *epWorldSize, const int64_t *epRankId, const int64_t *sharedExpertRankNum,
    const int64_t *globalBs)
{
    // 获取输入shape
    const gert::Shape *xShape = context->GetInputShape(DISPATCH_INPUT_X_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, xShape);
    const gert::Shape *expertIdsShape = context->GetInputShape(DISPATCH_INPUT_EXPERT_IDX_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, expertIdsShape);

    // 获取expandIdxShape
    gert::Shape *expandIdxShape = context->GetOutputShape(DISPATCH_OUTPUT_EXPAND_IDX_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, expandIdxShape);

    // 校验epRankId和sharedExpertRankNum
    OP_CHECK_IF((*epRankId < 0) || (*epRankId >= *epWorldSize),
        OP_LOGE(context->GetNodeName(), "epRankId shoule be in [0, epWorldSize), but got"
        " epWorldSize: %ld, epRankId: %ld.", *epWorldSize, *epRankId), return ge::GRAPH_FAILED);
    OP_CHECK_IF((*sharedExpertRankNum < 0) || (*sharedExpertRankNum >= *epWorldSize),
        OP_LOGE(context->GetNodeName(), "sharedExpertRankNum shoule be in [0, epWorldSize), but got"
        " epWorldSize: %ld, sharedExpertRankNum: %ld.", *epWorldSize, *sharedExpertRankNum), return ge::GRAPH_FAILED);

    // 计算bs/h/bsTmp/k并校验
    bs = xShape->GetDimNum() == 1U ? NEG_ONE : xShape->GetDim(0);
    h = xShape->GetDimNum() == 1U ? NEG_ONE : xShape->GetDim(1);
    int64_t bsTmp = expertIdsShape->GetDimNum() == 1U ? NEG_ONE : expertIdsShape->GetDim(0);
    k = expertIdsShape->GetDimNum() == 1U ? NEG_ONE : expertIdsShape->GetDim(1);
    OP_CHECK_IF((bs <= 0) || (h <= 0) || (bsTmp <= 0) || (k <= 0),
        OP_LOGE(context->GetNodeName(), "Input shape of xShape or input shape of expertIdsShape is incorrect, "
        "xShape [%ld, %ld], expertIdsShape [%ld, %ld]", bs, h, bsTmp, k),
        return ge::GRAPH_FAILED);

    // 计算globalBsReal并校验
    globalBsReal = (*globalBs == 0) ? (bs * *epWorldSize) : *globalBs;
    OP_CHECK_IF(globalBsReal < 0, OP_LOGE(context->GetNodeName(), "real global bs should be larger than 0"
        " but got %ld.", globalBsReal), return ge::GRAPH_FAILED);

    // 设置expandIdxShape
    expandIdxShape->SetDimNum(DIM_ONE);
    expandIdxShape->SetDim(0U, bs * k);
    OP_LOGD(context->GetNodeName(), "expandIdxShape shape is :%s after infershape.",
        Ops::Base::ToString(*expandIdxShape).c_str());

    return ge::GRAPH_SUCCESS;
}

// 子函数2：处理ExpandX/DynamicScales/ExpandScales Shape + 本地专家数计算（~40行，<50）
static ge::graphStatus InferExpandXAndScalesShape(gert::InferShapeContext *context, 
    int64_t& localExpertNum, int64_t& a, int64_t bs, int64_t h, int64_t k,
    int64_t globalBsReal, const int64_t *epWorldSize, const int64_t *epRankId,
    const int64_t *moeExpertNum, const int64_t *expertShardType, const int64_t *sharedExpertRankNum,
    const int64_t *tpWorldSize)
{
    // 获取输出shape
    gert::Shape *expandXShape = context->GetOutputShape(DISPATCH_OUTPUT_EXPAND_X_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, expandXShape);
    gert::Shape *dynamicScalesShape = context->GetOutputShape(DISPATCH_OUTPUT_DYNAMIC_SCALES_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, dynamicScalesShape);
    gert::Shape *expandScalesShape = context->GetOutputShape(DISPATCH_OUTPUT_EXPAND_SCALES);
    OPS_CHECK_NULL_WITH_CONTEXT(context, expandScalesShape);
    const gert::Shape *expertScalesShape = context->GetOptionalInputShape(DISPATCH_INPUT_EXPERT_SCALES_IDX_INDEX);

    // 计算moeRankNum/localMoeExpertNum/localExpertNum/a
    int64_t moeRankNum = *epWorldSize - *sharedExpertRankNum;
    OP_CHECK_IF(moeRankNum <= 0, OP_LOGE(context->GetNodeName(), "moeRankNum(epWorldSize - sharedExpertRankNum)"
        " should be larger than 0, but got %ld.", moeRankNum), return ge::GRAPH_FAILED);
    int64_t localMoeExpertNum = *moeExpertNum / moeRankNum;
    if (*expertShardType == 0) {
        if (*epRankId < *sharedExpertRankNum) {
            localExpertNum = 1;
            a = globalBsReal / *sharedExpertRankNum;
        } else {
            localExpertNum = localMoeExpertNum;
            a = globalBsReal * std::min(localExpertNum, k);
        }
    } else {
        if (*epRankId >= (*epWorldSize - *sharedExpertRankNum)) {
            localExpertNum = 1;
            a = globalBsReal / *sharedExpertRankNum;
        } else {
            localExpertNum = localMoeExpertNum;
            a = globalBsReal * std::min(localExpertNum, k);
        }
    }

    // 设置expandXShape
    auto realA = (*tpWorldSize == 0) ? a : a * *tpWorldSize;
    expandXShape->SetDimNum(DIM_TWO);
    expandXShape->SetDim(0U, realA);
    expandXShape->SetDim(1U, h);
    OP_LOGD(context->GetNodeName(), "expandx shape is :%s after infershape.",
        Ops::Base::ToString(*expandXShape).c_str());

    // 设置dynamicScalesShape
    dynamicScalesShape->SetDimNum(DIM_ONE);
    dynamicScalesShape->SetDim(0U, realA);
    OP_LOGD(context->GetNodeName(), "dynamicScalesShape shape is :%s after infershape.",
        Ops::Base::ToString(*dynamicScalesShape).c_str());

    // 设置expandScalesShape
    expandScalesShape->SetDimNum(DIM_ONE);
    expandScalesShape->SetDim(0U, 0);
    if (expertScalesShape != nullptr) {
        expandScalesShape->SetDim(0U, a);
    }
    OP_LOGD(context->GetNodeName(), "expandScalesShape shape is :%s after infershape.",
        Ops::Base::ToString(*expandScalesShape).c_str());

    return ge::GRAPH_SUCCESS;
}

// 子函数3：处理剩余输出Shape（ExpertTokenNums/EpRecvCount/TpRecvCount）（~30行，<50）
static ge::graphStatus InferRemainingOutputShapes(gert::InferShapeContext *context, 
    int64_t localExpertNum, int64_t k, int64_t globalBsReal,
    const int64_t *epWorldSize, const int64_t *tpWorldSize)
{
    // 获取输出shape
    gert::Shape *expertTokenNumsShape = context->GetOutputShape(DISPATCH_OUTPUT_EXPERT_TOKEN_NUMS_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, expertTokenNumsShape);
    gert::Shape *epRecvCountShape = context->GetOutputShape(DISPATCH_OUTPUT_EP_RECV_COUNTS_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, epRecvCountShape);
    gert::Shape *tpRecvCountShape = context->GetOutputShape(DISPATCH_OUTPUT_TP_RECV_COUNTS_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, tpRecvCountShape);

    // 设置expertTokenNumsShape
    expertTokenNumsShape->SetDimNum(DIM_ONE);
    expertTokenNumsShape->SetDim(0U, localExpertNum);
    OP_LOGD(context->GetNodeName(), "expertTokenNumsShape shape is :%s after infershape.",
        Ops::Base::ToString(*expertTokenNumsShape).c_str());

    // 设置epRecvCountShape
    epRecvCountShape->SetDimNum(DIM_ONE);
    if (IsPlatform910B(context->GetNodeName())) {
        epRecvCountShape->SetDim(0U, *epWorldSize * localExpertNum + globalBsReal * 2 * k * (*epWorldSize) / RANK_NUM_PER_NODE);
    } else {
        if (*tpWorldSize == DIM_TWO)  {
            epRecvCountShape->SetDim(0U, (*epWorldSize) * localExpertNum * (*tpWorldSize));
        } else {
            epRecvCountShape->SetDim(0U, (*epWorldSize) * localExpertNum);
        }
    }
    OP_LOGD(context->GetNodeName(), "epRecvCountShape shape is :%s after infershape.",
        Ops::Base::ToString(*epRecvCountShape).c_str());

    // 设置tpRecvCountShape
    tpRecvCountShape->SetDimNum(DIM_ONE);
    tpRecvCountShape->SetDim(0U, *tpWorldSize);
    OP_LOGD(context->GetNodeName(), "tpRecvCountShape shape is :%s after infershape.",
        Ops::Base::ToString(*tpRecvCountShape).c_str());

    return ge::GRAPH_SUCCESS;
}

// 重构后的主函数（~40行，<50）
static ge::graphStatus InferShapeMoeDistributeDispatch(gert::InferShapeContext *context)
{
    if (context == nullptr){
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context->GetNodeName(), "Begin to do InferShapeMoeDistributeDispatch.");

    // 1. 获取所有属性指针
    const auto attrs = context->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const auto epWorldSize = attrs->GetAttrPointer<int64_t>(DISPATCH_INPUT_ATTR_EP_WORLD_SIZE_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, epWorldSize);
    const auto epRankId = attrs->GetAttrPointer<int64_t>(DISPATCH_INPUT_ATTR_EP_RANK_ID_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, epRankId);
    const auto moeExpertNum = attrs->GetAttrPointer<int64_t>(DISPATCH_INPUT_ATTR_MOE_EXPERT_NUM_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, moeExpertNum);
    const auto tpWorldSize = attrs->GetAttrPointer<int64_t>(DISPATCH_INPUT_ATTR_TP_WORLD_SIZE_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, tpWorldSize);
    const auto tpRankId = attrs->GetAttrPointer<int64_t>(DISPATCH_INPUT_ATTR_TP_RANK_ID_INDEX); // 保留但未使用，与原代码一致
    OPS_CHECK_NULL_WITH_CONTEXT(context, tpRankId);
    const auto expertShardType = attrs->GetAttrPointer<int64_t>(DISPATCH_INPUT_ATTR_EXPERT_SHARD_TYPE_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, expertShardType);
    const auto sharedExpertRankNum = attrs->GetAttrPointer<int64_t>(DISPATCH_INPUT_ATTR_SHARED_EXPERT_RANK_NUM_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, sharedExpertRankNum);
    const auto quantMode = attrs->GetAttrPointer<int64_t>(DISPATCH_INPUT_ATTR_QUANT_MODE_INDEX); // 保留但未使用，与原代码一致
    OPS_CHECK_NULL_WITH_CONTEXT(context, quantMode);
    const auto globalBs = attrs->GetAttrPointer<int64_t>(DISPATCH_INPUT_ATTR_GLOBAL_BS_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, globalBs);

    // 2. 声明中间变量
    int64_t bs = 0, h = 0, k = 0, globalBsReal = 0, localExpertNum = 0, a = 0;

    // 3. 调用子函数1：处理专家ID Shape
    OP_CHECK_IF(InferExpertIdsShape(context, bs, h, k, globalBsReal, epWorldSize, epRankId, sharedExpertRankNum, globalBs) != ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "InferExpertIdsShape failed."), return ge::GRAPH_FAILED);

    // 4. 调用子函数2：处理ExpandX/Scales Shape
    OP_CHECK_IF(InferExpandXAndScalesShape(context, localExpertNum, a, bs, h, k, globalBsReal,
        epWorldSize, epRankId, moeExpertNum, expertShardType, sharedExpertRankNum, tpWorldSize) != ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "InferExpandXAndScalesShape failed."), return ge::GRAPH_FAILED);

    // 5. 调用子函数3：处理剩余输出Shape
    OP_CHECK_IF(InferRemainingOutputShapes(context, localExpertNum, k, globalBsReal, epWorldSize, tpWorldSize) != ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "InferRemainingOutputShapes failed."), return ge::GRAPH_FAILED);

    OP_LOGD(context->GetNodeName(), "End to do InferShapeMoeDistributeDispatch.");
    return ge::GRAPH_SUCCESS;
}

// 原有函数，完全保留（~25行，<50）
static ge::graphStatus InferDataTypeMoeDistributeDispatch(gert::InferDataTypeContext *context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferDataTypeMoeDistributeDispatch.");
    auto xDtype = context->GetInputDataType(DISPATCH_INPUT_X_INDEX);
    const auto attrs = context->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const auto quantMode = attrs->GetAttrPointer<int64_t>(DISPATCH_INPUT_ATTR_QUANT_MODE_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, quantMode);
    const auto scalesType = context->GetOptionalInputDataType(DISPATCH_INPUT_SCALES_IDX_INDEX);
    bool quantFlag = (scalesType != ge::DT_UNDEFINED) ? true : false;
    OP_LOGD(context->GetNodeName(), "quantFlag id %d.", quantFlag);
    if (quantFlag || (*quantMode != 0)) {
        context->SetOutputDataType(DISPATCH_OUTPUT_EXPAND_X_INDEX, ge::DT_INT8);
    } else {
        context->SetOutputDataType(DISPATCH_OUTPUT_EXPAND_X_INDEX, xDtype);
    }
    context->SetOutputDataType(DISPATCH_OUTPUT_DYNAMIC_SCALES_INDEX, ge::DT_FLOAT);
    context->SetOutputDataType(DISPATCH_OUTPUT_EXPAND_IDX_INDEX, ge::DT_INT32);
    context->SetOutputDataType(DISPATCH_OUTPUT_EXPERT_TOKEN_NUMS_INDEX, ge::DT_INT64);
    context->SetOutputDataType(DISPATCH_OUTPUT_EP_RECV_COUNTS_INDEX, ge::DT_INT32);
    context->SetOutputDataType(DISPATCH_OUTPUT_TP_RECV_COUNTS_INDEX, ge::DT_INT32);
    OP_LOGD(context->GetNodeName(), "End to do InferDataTypeMoeDistributeDispatch.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MoeDistributeDispatch)
    .InferShape(InferShapeMoeDistributeDispatch)
    .InferDataType(InferDataTypeMoeDistributeDispatch);
}  // namespace ops