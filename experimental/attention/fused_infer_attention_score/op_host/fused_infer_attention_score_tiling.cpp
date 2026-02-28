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
 * \file fused_infer_attention_score_tiling.cpp
 * \brief
 */

#include "fused_infer_attention_score_tiling.h"
#include "../../incre_flash_attention/op_host/incre_flash_attention_tiling_impl.h"
#include "log/log.h"
#include "log/error_code.h"
#include "err/ops_err.h"
#include "tiling/tiling_api.h"
#include "platform/platform_info.h"
#include "arch32/fused_infer_attention_score_tiling_v3.h"
#include "fused_infer_attention_score_const.h"

using namespace ge;
using namespace AscendC;
namespace optiling {
    
bool IsGqaIfa(gert::TilingContext &context, const string inputLayoutStr, const int64_t queryS, const int64_t queryD)
{
    if (context.GetOptionalInputTensor(QUERY_ROPE_INDEX) != nullptr) {
        return false;
    }
    if ((inputLayoutStr == "TND_NTD") || (inputLayoutStr == "TND")) {
        if (queryD == 512) { // 512: qD need 512
            return true;
        }
    }
    else if ((inputLayoutStr == "BSH") || (inputLayoutStr == "BNSD") || (inputLayoutStr == "BSND") ||
            (inputLayoutStr == "BNSD_NBSD") || (inputLayoutStr == "BSND_NBSD") || (inputLayoutStr == "BSH_NBSD")) {
        if (queryS == 1) {
            return true;
        }
    }
    return false;
}

int64_t GetTndQueryS(gert::TilingContext &context)
{
    auto queryShape = context.GetInputShape(QUERY_INDEX);
    auto actualSeqlenthsQ = context.GetOptionalInputTensor(ACTUAL_SEQ_Q_INDEX);
    auto actualSeqlenthsKv = context.GetOptionalInputTensor(ACTUAL_SEQ_KV_INDEX);
    auto blockTable = context.GetOptionalInputTensor(BLOCK_TABLE_INDEX);
    int64_t batchSize = blockTable->GetStorageShape().GetDim(DIM_0);
    const int64_t *actualSeqQ = actualSeqlenthsQ->GetData<int64_t>();
    const int64_t *actualSeqKv = actualSeqlenthsKv->GetData<int64_t>();

    if (batchSize == 0) {
        return 0;
    }
    if (actualSeqQ == nullptr || actualSeqKv == nullptr) { // tiling下沉场景
        int64_t queryT4Tnd = queryShape->GetStorageShape().GetDim(DIM_0);
        return (queryT4Tnd + batchSize - 1) / batchSize;
    }
    int64_t qActualSeqMax = 0;
    for (int64_t i = 0; i < batchSize; i++) {
        int64_t tmpS1 = (i == 0) ? actualSeqQ[0] : (actualSeqQ[i] - actualSeqQ[i - 1U]);
        if (tmpS1 > qActualSeqMax) {
            qActualSeqMax = tmpS1;
        }
    }
    return qActualSeqMax;
}

bool IsGqaMtp(gert::TilingContext &context, const string inputLayoutStr, const int64_t queryS, const int64_t queryD)
{
    if (context.GetOptionalInputTensor(QUERY_ROPE_INDEX) != nullptr) {
        return false;
    }
    bool isIFALayout = (inputLayoutStr == "BSH") || (inputLayoutStr == "BNSD") || (inputLayoutStr == "BSND") ||
            (inputLayoutStr == "BNSD_NBSD") || (inputLayoutStr == "BSND_NBSD") || (inputLayoutStr == "BSH_NBSD")
            || (inputLayoutStr == "TND");
    if (!isIFALayout) {
        return false;
    }
    auto tempK = context.GetInputShape(KEY_INDEX);
    bool isNz = (tempK->GetStorageShape().GetDimNum() == 5);
    if (!isNz) {
        return false;
    }

    int64_t actualQueryS = queryS;
    if (inputLayoutStr == "TND") {
        actualQueryS = GetTndQueryS(context);
    }
    if (!(actualQueryS >= 1 && actualQueryS <= 16)) { // 16: mtp
        return false;
    }

    if (inputLayoutStr == "TND") {
        if (queryD != 128) { // 128: queryD need 128 when gqa kv_nz
            return false;
        }
        if (context.GetInputDesc(QUERY_INDEX)->GetDataType() != ge::DT_BF16 ||
            context.GetInputDesc(KEY_INDEX)->GetDataType() != ge::DT_INT8 ||
            context.GetInputDesc(VALUE_INDEX)->GetDataType() != ge::DT_INT8) {
            return false;
        }
    }
    return true;
}

bool IsAtbIfa(gert::TilingContext &context, const string inputLayoutStr, const int64_t queryD)
{
    bool isIsAtbIfaLayout = (inputLayoutStr == "TND_NTD") || (inputLayoutStr == "TND");
    if (!isIsAtbIfaLayout) {
        return false;
    }
    if (!((context.GetInputDesc(KEY_INDEX)->GetDataType() == ge::DT_INT8) &&
        (context.GetInputDesc(QUERY_INDEX)->GetDataType() != ge::DT_INT8))) { // KV antiquant
        return false;
    }
    if (context.GetOptionalInputShape(BLOCK_TABLE_INDEX) == nullptr) { // PA
        return false;
    }
    if (queryD == 512) { // 512: max D limix
        return false;
    }
    bool isAntiquantParamFloat = (context.GetOptionalInputTensor(ANTIQUANT_SCALE_INDEX) != nullptr) &&
                                 (context.GetOptionalInputTensor(ANTIQUANT_OFFSET_INDEX) != nullptr) &&
                                 (context.GetOptionalInputDesc(ANTIQUANT_SCALE_INDEX)->GetDataType() == ge::DT_FLOAT) &&
                                 (context.GetOptionalInputDesc(ANTIQUANT_OFFSET_INDEX)->GetDataType() == ge::DT_FLOAT);
    if (!isAntiquantParamFloat) {
        return false;
    }
    return true;
}

bool IsMlaIfaOrMtp(gert::TilingContext &context, const string inputLayoutStr, const int64_t queryS, const int64_t queryD)
{
    if (context.GetOptionalInputTensor(QUERY_ROPE_INDEX) == nullptr) { // mla
        return false;
    }
    if ((inputLayoutStr == "TND_NTD") || (inputLayoutStr == "TND")) {
        if (queryD == 512) { // 512: qD need 512
            return true;
        }
    } else if ((inputLayoutStr == "BSH") || (inputLayoutStr == "BNSD") || (inputLayoutStr == "BSND") ||
            (inputLayoutStr == "BNSD_NBSD") || (inputLayoutStr == "BSND_NBSD") || (inputLayoutStr == "BSH_NBSD")) {
        if (queryS == 1)  {
            return true;
        }
        if ((queryS > 1 && queryS <= 16) && (queryD == 512)) { // 16: mtp; 512: qD need 512
            return true;
        }
    }
    return false;
}

bool IsSlidingAttention(gert::TilingContext &context, const string inputLayoutStr, const int64_t queryD)
{
    if (context.GetOptionalInputTensor(QUERY_ROPE_INDEX) == nullptr) { // mla
        return false;
    }
    if ((inputLayoutStr == "BSH") || (inputLayoutStr == "BNSD") || (inputLayoutStr == "BSND") ||
            (inputLayoutStr == "BNSD_NBSD") || (inputLayoutStr == "BSND_NBSD") || (inputLayoutStr == "BSH_NBSD")) {
        if (queryD != 512) { // 512: qD need 512
            return false;
        }
        if (*context.GetAttrs()->GetAttrPointer<uint32_t>(ATTR_SPARSE_MODE_INDEX) == 4) { // 4: sparseMode =4
            return true;
        }
    }
    return false;
}

static bool IsUsingIFA(gert::TilingContext &context, const string inputLayoutStr, const uint32_t queryD, 
    const int64_t queryS)
{
    if (IsGqaIfa(context, inputLayoutStr, queryS, queryD) || 
        IsGqaMtp(context, inputLayoutStr, queryS, queryD) || 
        IsAtbIfa(context, inputLayoutStr, queryD) || 
        IsMlaIfaOrMtp(context, inputLayoutStr, queryS, queryD) ||
        IsSlidingAttention(context, inputLayoutStr, queryD)) {
        return true;
    }
    return false;
}

static void ConvertOptionalInputsIFA(gert::TilingContext &context, IncreFlashAttentionContext &ifaContext)
{
    ifaContext.pseShift.desc = context.GetOptionalInputDesc(PSE_SHIFT_INDEX);
    ifaContext.pseShift.tensor = context.GetOptionalInputTensor(PSE_SHIFT_INDEX);

    ifaContext.attenMask.desc = context.GetOptionalInputDesc(ATTEN_MASK_INDEX);
    ifaContext.attenMask.tensor = context.GetOptionalInputTensor(ATTEN_MASK_INDEX);

    ifaContext.actualSeqLengthsQ.tensor = context.GetOptionalInputTensor(ACTUAL_SEQ_Q_INDEX);
    ifaContext.actualSeqLengths.tensor = context.GetOptionalInputTensor(ACTUAL_SEQ_KV_INDEX);
    ifaContext.deqScale1.tensor = context.GetOptionalInputTensor(DEQUANT_SCALE1_INDEX);
    ifaContext.quantScale1.tensor = context.GetOptionalInputTensor(QUANT_SCALE1_INDEX);
    ifaContext.deqScale2.tensor = context.GetOptionalInputTensor(DEQUANT_SCALE2_INDEX);
    ifaContext.quantScale2.tensor = context.GetOptionalInputTensor(QUANT_SCALE2_INDEX);
    ifaContext.quantOffset2.tensor = context.GetOptionalInputTensor(QUANT_OFFSET2_INDEX);
    ifaContext.quantScale2.desc = context.GetOptionalInputDesc(QUANT_SCALE2_INDEX);
    ifaContext.quantOffset2.desc = context.GetOptionalInputDesc(QUANT_OFFSET2_INDEX);
    ifaContext.antiquantScale.tensor = context.GetOptionalInputTensor(ANTIQUANT_SCALE_INDEX);
    ifaContext.antiquantScale.desc = context.GetOptionalInputDesc(ANTIQUANT_SCALE_INDEX);
    ifaContext.antiquantOffset.tensor = context.GetOptionalInputTensor(ANTIQUANT_OFFSET_INDEX);
    ifaContext.antiquantOffset.desc = context.GetOptionalInputDesc(ANTIQUANT_OFFSET_INDEX);
    ifaContext.blockTable.tensor = context.GetOptionalInputTensor(BLOCK_TABLE_INDEX);
    ifaContext.queryPaddingSize.tensor = context.GetOptionalInputTensor(QUERY_PADDING_SIZE_INDEX);
    ifaContext.kvPaddingSize.tensor = context.GetOptionalInputTensor(KV_PADDING_SIZE_INDEX);
    ifaContext.keyAntiquantScale.tensor = context.GetOptionalInputTensor(KEY_ANTIQUANT_SCALE_INDEX);
    ifaContext.keyAntiquantScale.desc = context.GetOptionalInputDesc(KEY_ANTIQUANT_SCALE_INDEX);
    ifaContext.keyAntiquantOffset.tensor = context.GetOptionalInputTensor(KEY_ANTIQUANT_OFFSET_INDEX);
    ifaContext.keyAntiquantOffset.desc = context.GetOptionalInputDesc(KEY_ANTIQUANT_OFFSET_INDEX);
    ifaContext.valueAntiquantScale.tensor = context.GetOptionalInputTensor(VALUE_ANTIQUANT_SCALE_INDEX);
    ifaContext.valueAntiquantScale.desc = context.GetOptionalInputDesc(VALUE_ANTIQUANT_SCALE_INDEX);
    ifaContext.valueAntiquantOffset.tensor = context.GetOptionalInputTensor(VALUE_ANTIQUANT_OFFSET_INDEX);
    ifaContext.valueAntiquantOffset.desc = context.GetOptionalInputDesc(VALUE_ANTIQUANT_OFFSET_INDEX);
    ifaContext.keySharedPrefix.tensor = context.GetOptionalInputTensor(KEY_SHARED_PREFIX_INDEX);
    ifaContext.keySharedPrefix.desc = context.GetOptionalInputDesc(KEY_SHARED_PREFIX_INDEX);
    ifaContext.valueSharedPrefix.tensor = context.GetOptionalInputTensor(VALUE_SHARED_PREFIX_INDEX);
    ifaContext.valueSharedPrefix.desc = context.GetOptionalInputDesc(VALUE_SHARED_PREFIX_INDEX);
    ifaContext.actualSharedPrefixLen.tensor = context.GetOptionalInputTensor(ACTUAL_SHARED_PREFIX_LEN_INDEX);

    ifaContext.queryRope.tensor = context.GetOptionalInputTensor(QUERY_ROPE_INDEX);
    ifaContext.queryRope.desc = context.GetOptionalInputDesc(QUERY_ROPE_INDEX);
    ifaContext.keyRope.tensor = context.GetOptionalInputTensor(KEY_ROPE_INDEX);
    ifaContext.keyRope.desc = context.GetOptionalInputDesc(KEY_ROPE_INDEX);
    ifaContext.keyRopeAntiquantScale.tensor = context.GetOptionalInputTensor(KEY_ROPE_ANTIQUANT_SCALE_INDEX);
    ifaContext.keyRopeAntiquantScale.desc = context.GetOptionalInputDesc(KEY_ROPE_ANTIQUANT_SCALE_INDEX);
    ifaContext.dequantScaleQuery.tensor = context.GetOptionalInputTensor(DEQUANT_SCALE_QUERY_INDEX);
    ifaContext.dequantScaleQuery.desc = context.GetOptionalInputDesc(DEQUANT_SCALE_QUERY_INDEX);
}

static ge::graphStatus ConvertAttrsIFA(gert::TilingContext &context, IncreFlashAttentionContext &ifaContext)
{
    auto attrs = context.GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(), "attrs got from ge is nullptr"),
        return ge::GRAPH_FAILED);

    ifaContext.numHeads = attrs->GetAttrPointer<uint32_t>(ATTR_N_INDEX);
    ifaContext.scaleValue = attrs->GetAttrPointer<float>(ATTR_SCALE_INDEX);
    ifaContext.layOut = attrs->GetStr(ATTR_INPUT_LAYOUT_INDEX);
    ifaContext.kvHeadNums = attrs->GetAttrPointer<uint32_t>(ATTR_NUM_KV_HEADS_INDEX);
    ifaContext.blockSize = attrs->GetAttrPointer<uint32_t>(ATTR_BLOCK_SIZE_INDEX);
    ifaContext.antiquantMode = attrs->GetAttrPointer<int64_t>(ANTIQUANT_MODE_INDEX);
    ifaContext.softmaxLseFlag = attrs->GetAttrPointer<bool>(SOFTMAX_LSE_FLAG_INDEX);
    ifaContext.keyAntiquantMode = attrs->GetAttrPointer<int64_t>(KEY_ANTIQUANT_MODE_INDEX);
    ifaContext.valueAntiquantMode = attrs->GetAttrPointer<int64_t>(VALUE_ANTIQUANT_MODE_INDEX);
    ifaContext.innerPrecise = attrs->GetAttrPointer<uint32_t>(ATTR_INNER_PRECISE_INDEX);
    ifaContext.sparseMode = attrs->GetAttrPointer<uint32_t>(ATTR_SPARSE_MODE_INDEX);
    ifaContext.queryQuantMode = attrs->GetAttrPointer<int64_t>(QUERY_QUANT_MODE_INDEX);
    ifaContext.windowSize = attrs->GetAttrPointer<int64_t>(ATTR_PRE_TOKEN_INDEX);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ConvertContextToParamsIFA(gert::TilingContext &context, IncreFlashAttentionContext &ifaContext)
{
    if (context.GetNodeName() == nullptr) {
        OP_LOGE("FusedInferAttentionScore", "opName got from TilingContext is nullptr");
        return ge::GRAPH_FAILED;
    }
    ifaContext.opName = context.GetNodeName();
    ifaContext.platformInfo = context.GetPlatformInfo();
    ifaContext.query.desc = context.GetInputDesc(QUERY_INDEX);
    ifaContext.query.shape = context.GetInputShape(QUERY_INDEX);
    ifaContext.key.desc = context.GetInputDesc(KEY_INDEX);
    ifaContext.key.shape = context.GetInputShape(KEY_INDEX);
    OP_CHECK_IF((ifaContext.query.shape == nullptr) || (ifaContext.key.shape == nullptr),
        OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(), "shape of query of shape of key is null."),
        return ge::GRAPH_FAILED);
    auto batchOfQuery = ifaContext.query.shape->GetStorageShape().GetDim(0);
    auto batchOfKey = ifaContext.key.shape->GetStorageShape().GetDim(0);
    if (batchOfQuery != batchOfKey) {
        ifaContext.kCache.resize(batchOfQuery);
        ifaContext.vCache.resize(batchOfQuery);
        for (int64_t size = 0; size < batchOfQuery; ++size) {
            ifaContext.kCache[size] = const_cast<gert::StorageShape *>(context.GetDynamicInputShape(KEY_INDEX, size));
            ifaContext.vCache[size] = const_cast<gert::StorageShape *>(context.GetDynamicInputShape(VALUE_INDEX, size));
        }
    } else {
        ifaContext.kCache.resize(1);
        ifaContext.vCache.resize(1);
        ifaContext.kCache[0] = const_cast<gert::StorageShape *>(context.GetDynamicInputShape(KEY_INDEX, 0));
        ifaContext.vCache[0] = const_cast<gert::StorageShape *>(context.GetDynamicInputShape(VALUE_INDEX, 0));
    }

    ifaContext.value.desc = context.GetInputDesc(VALUE_INDEX);
    ifaContext.value.shape = context.GetInputShape(VALUE_INDEX);
    ifaContext.attenOut.desc = context.GetOutputDesc(ATTENTION_OUT_INDEX);
    ifaContext.attenOut.shape = context.GetOutputShape(ATTENTION_OUT_INDEX);

    ConvertOptionalInputsIFA(context, ifaContext);

    OP_CHECK_IF(ConvertAttrsIFA(context, ifaContext) != ge::GRAPH_SUCCESS,
        OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(), "convert attrs failed"), return ge::GRAPH_FAILED);

    OP_CHECK_IF(context.GetWorkspaceSizes(1) == nullptr,
        OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(), "workSpaceSize got from ge is nullptr"),
        return ge::GRAPH_FAILED);
    ifaContext.workSpaces = context.GetWorkspaceSizes(1);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingProcess4IFA(gert::TilingContext *context)
{
    // IFA tiling path
    IncreFlashAttentionContext ifaContext {};
    auto ret = ConvertContextToParamsIFA(*context, ifaContext);
    if (ret != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "Error occored while convert tilingContext to ifa context");
        return ret;
    }
    IFATiling ifaTiling(context);
    return ifaTiling.DoSubOpTiling(ifaContext);
}

static ge::graphStatus CheckQKV(gert::TilingContext &context)
{
    auto tempQ = context.GetInputShape(QUERY_INDEX);
    auto tempK = context.GetInputShape(KEY_INDEX);
    auto tempV = context.GetInputShape(VALUE_INDEX);
    auto tempOut = context.GetOutputShape(ATTENTION_OUT_INDEX);
    OP_CHECK_IF((tempQ == nullptr), OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(), "Query input is null pointer!"),
               return ge::GRAPH_FAILED);
    OP_CHECK_IF((tempK == nullptr), OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(), "Key input is null pointer!"),
               return ge::GRAPH_FAILED);
    OP_CHECK_IF((tempV == nullptr), OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(), "Value input is null pointer!"),
               return ge::GRAPH_FAILED);
    OP_CHECK_IF((tempOut == nullptr),
               OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(), "Attention_Out is null pointer!"),
               return ge::GRAPH_FAILED);
    OP_CHECK_IF((tempQ->GetStorageShape().GetShapeSize() == 0) && (tempOut->GetStorageShape().GetShapeSize() != 0),
               OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(),
               "Query head should not be 0, or when attentionOut is not empty tensor, query input shoud not be empty tensor!"),
               return ge::GRAPH_FAILED);
    OP_CHECK_IF((tempQ->GetStorageShape().GetShapeSize() == gert::Shape::kInvalidDimValue),
               OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(), "Query input dims are invalid!"),
               return ge::GRAPH_FAILED);
    OP_CHECK_IF((tempK->GetStorageShape().GetShapeSize() == gert::Shape::kInvalidDimValue),
               OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(), "Key input dims are invalid!"),
               return ge::GRAPH_FAILED);
    OP_CHECK_IF((tempV->GetStorageShape().GetShapeSize() == gert::Shape::kInvalidDimValue),
               OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(), "Value input dims are invalid!"),
               return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckOutShapeInDim3(const gert::TilingContext *context, const string &outputLayoutStr, const gert::Shape outShape,
    const gert::Shape exceptOutShape)
{
    OP_CHECK_IF((outShape.GetDimNum() != DIM_NUM_3),
            OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
            "OutputLayout is %s, Attention out shape dim should be 3, but got %zu!",
            outputLayoutStr.c_str(), outShape.GetDimNum()), return ge::GRAPH_FAILED);
    OP_CHECK_IF((exceptOutShape != outShape),
        OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
                                    "Expect outputLayout is %s and Out shape size[%ld, %ld, %ld] does NOT match "
                                    "Attention Out shape size[%ld, %ld, %ld]!",
                                    outputLayoutStr.c_str(),
                                    exceptOutShape.GetDim(DIM_0),
                                    exceptOutShape.GetDim(DIM_1),
                                    exceptOutShape.GetDim(DIM_2),
                                    outShape.GetDim(DIM_0),
                                    outShape.GetDim(DIM_1),
                                    outShape.GetDim(DIM_2)),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckOutShapeInDim4(const gert::TilingContext *context, const string &outputLayoutStr, const gert::Shape outShape,
    const gert::Shape exceptOutShape)
{
    OP_CHECK_IF((outShape.GetDimNum() != DIM_NUM_4),
            OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "OutputLayout is %s, Attention out shape dim should be 4, but got %zu!",
            outputLayoutStr.c_str(), outShape.GetDimNum()), return ge::GRAPH_FAILED);
    OP_CHECK_IF((exceptOutShape != outShape),
        OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
                                    "Expect outputLayout is %s and Out shape size[%ld, %ld, %ld, %ld] does NOT match "
                                    "Attention Out shape size[%ld, %ld, %ld, %ld]!",
                                    outputLayoutStr.c_str(),
                                    exceptOutShape.GetDim(DIM_0),
                                    exceptOutShape.GetDim(DIM_1),
                                    exceptOutShape.GetDim(DIM_2),
                                    exceptOutShape.GetDim(DIM_3),
                                    outShape.GetDim(DIM_0),
                                    outShape.GetDim(DIM_1),
                                    outShape.GetDim(DIM_2),
                                    outShape.GetDim(DIM_3)),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}
static ge::graphStatus GetB(const gert::TilingContext *context, const string inputLayoutStr, int64_t &b)
{
    auto tempQ = context->GetInputShape(QUERY_INDEX);
    if (inputLayoutStr == "NSD") {
        b = 1;
    } else {
        b = tempQ->GetStorageShape().GetDim(DIM_0);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetQueryN(const gert::TilingContext *context, const string inputLayoutStr, int64_t &queryN)
{
    auto tempQ = context->GetInputShape(QUERY_INDEX);
    if (inputLayoutStr == "NSD" || 
        inputLayoutStr == "NTD_TND") {
        queryN = tempQ->GetStorageShape().GetDim(DIM_0);
    } else if (inputLayoutStr == "BSND_NBSD" || 
        inputLayoutStr == "BSND") {
        queryN = tempQ->GetStorageShape().GetDim(DIM_2);
    } else if (inputLayoutStr == "BSH" || 
        inputLayoutStr == "BSH_NBSD") {
        auto attrs = context->GetAttrs();
        int64_t numHeads = static_cast<int64_t>(*attrs->GetAttrPointer<uint32_t>(ATTR_N_INDEX));
        queryN = numHeads;
    } else {
        queryN = tempQ->GetStorageShape().GetDim(DIM_1);
    }
    OP_CHECK_IF(queryN == 0, 
            OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "Q numhead is 0!"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetQueryS(const gert::TilingContext *context, const string inputLayoutStr, int64_t &queryS)
{
    auto tempQ = context->GetInputShape(QUERY_INDEX);
    if (inputLayoutStr == "NSD" || 
        inputLayoutStr == "BSH" || 
        inputLayoutStr == "BSH_NBSD" || 
        inputLayoutStr == "BSND_NBSD" || 
        inputLayoutStr == "BSND") {
        queryS = tempQ->GetStorageShape().GetDim(DIM_1);
    } else if (inputLayoutStr == "BNSD_BSND" || 
        inputLayoutStr == "BNSD_NBSD" || 
        inputLayoutStr == "BNSD") {
        queryS = tempQ->GetStorageShape().GetDim(DIM_2);
    } else {
        int64_t queryT = 0;
        if (inputLayoutStr == "NTD_TND") {
            queryT = tempQ->GetStorageShape().GetDim(DIM_1);
        } else {
            queryT = tempQ->GetStorageShape().GetDim(DIM_0);
        }
        queryS = queryT;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetQueryT(const gert::TilingContext *context, const string inputLayoutStr, int64_t &queryT)
{
    auto tempQ = context->GetInputShape(QUERY_INDEX);
    if (inputLayoutStr == "TND" || 
        inputLayoutStr == "TND_NTD") {
        queryT = tempQ->GetStorageShape().GetDim(DIM_0);
    } else if (inputLayoutStr == "NTD_TND") {
        queryT = tempQ->GetStorageShape().GetDim(DIM_1);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetQueryD(const gert::TilingContext *context, const string inputLayoutStr, int64_t &queryD)
{
    auto tempQ = context->GetInputShape(QUERY_INDEX);
    if (inputLayoutStr == "NSD" || 
        inputLayoutStr == "TND" || 
        inputLayoutStr == "TND_NTD" || 
        inputLayoutStr == "NTD_TND") {
        queryD = tempQ->GetStorageShape().GetDim(DIM_2);
    } else if (inputLayoutStr == "BNSD_BSND" || 
            inputLayoutStr == "BNSD_NBSD"  || 
            inputLayoutStr == "BNSD"  || 
            inputLayoutStr == "BSND_NBSD" || 
            inputLayoutStr == "BSND") {
        queryD = tempQ->GetStorageShape().GetDim(DIM_3);
    } else {
        int64_t queryH = tempQ->GetStorageShape().GetDim(DIM_2);
        auto attrs = context->GetAttrs();
        int64_t numHeads = static_cast<int64_t>(*attrs->GetAttrPointer<uint32_t>(ATTR_N_INDEX));
        queryD = queryH / numHeads;
    }
    const int64_t maxDlimit = 512;
    OP_CHECK_IF((queryD > maxDlimit), OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
        "D should be less than or equal to 512 of Q shape! but now D = %ld. "
        "When layout is BNSD, D is the last dimension of Q shape, and layout is BSH, D = h / n", queryD),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetPAValueD(const gert::TilingContext *context, int64_t &valueD)
{
    auto tempV = context->GetInputShape(VALUE_INDEX);
    if (tempV->GetStorageShape().GetDimNum() == DIM_BSH) { // BnBsH
        auto attrs = context->GetAttrs();
        int64_t numKvHeads = static_cast<int64_t>(*attrs->GetAttrPointer<uint32_t>(ATTR_NUM_KV_HEADS_INDEX));
        if (numKvHeads == 0) {
            numKvHeads = static_cast<int64_t>(*attrs->GetAttrPointer<uint32_t>(ATTR_N_INDEX));
        }
        valueD = tempV->GetStorageShape().GetDim(DIM_2) / numKvHeads;
    } else if (tempV->GetStorageShape().GetDimNum() == DIM_BNSD_OR_BSND) { // BnNBsD
        valueD = tempV->GetStorageShape().GetDim(DIM_3);
    } else if (tempV->GetStorageShape().GetDimNum() == NUM5) { // NZ
        valueD = tempV->GetStorageShape().GetDim(DIM_2) * tempV->GetStorageShape().GetDim(DIM_4);
    } else {
        OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),"PagedAttention not support Value DimNum is %zu.\n",
            tempV->GetStorageShape().GetDimNum());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetValueD(gert::TilingContext *context, const string inputLayoutStr, int64_t &valueD, bool isPageAttention)
{
    if (isPageAttention) {
        return GetPAValueD(context, valueD);
    }
    auto tempV = context->GetInputShape(VALUE_INDEX);
    if (inputLayoutStr == "NSD" || 
        inputLayoutStr == "TND" || 
        inputLayoutStr == "TND_NTD" || 
        inputLayoutStr == "NTD_TND") {
        OP_CHECK_IF((tempV->GetStorageShape().GetDimNum() != DIM_NUM_3),
                    OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
                    "When block_table is null and input_layout is %s, dim number of key/value should be 3, but it is %zu.\n",
                    inputLayoutStr.c_str(), tempV->GetStorageShape().GetDimNum()), return ge::GRAPH_FAILED);
        valueD = tempV->GetStorageShape().GetDim(DIM_2);
    } else if (inputLayoutStr == "BNSD_BSND" || 
        inputLayoutStr == "BNSD_NBSD"  || 
        inputLayoutStr == "BNSD"  || 
        inputLayoutStr == "BSND_NBSD" || 
        inputLayoutStr == "BSND") {
        OP_CHECK_IF((tempV->GetStorageShape().GetDimNum() != DIM_NUM_4),
                    OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
                    "When block_table is null and input_layout is %s, dim number of key/value should be 4, but it is %zu.\n",
                    inputLayoutStr.c_str(), tempV->GetStorageShape().GetDimNum()), return ge::GRAPH_FAILED);
        valueD = tempV->GetStorageShape().GetDim(DIM_3);
    } else {
        int64_t valueH = tempV->GetStorageShape().GetDim(DIM_2);
        auto attrs = context->GetAttrs();
        int64_t numKvHeads = static_cast<int64_t>(*attrs->GetAttrPointer<uint32_t>(ATTR_NUM_KV_HEADS_INDEX));
        if (numKvHeads == 0) {
            numKvHeads = static_cast<int64_t>(*attrs->GetAttrPointer<uint32_t>(ATTR_N_INDEX));
        }
        valueD = valueH / numKvHeads;
    }
    return ge::GRAPH_SUCCESS;
}

string GetOutputLayoutStr(const string &inputLayoutStr)
{
    size_t underLinePos = inputLayoutStr.find_last_of('_');
    if (underLinePos == std::string::npos) {
        return inputLayoutStr;
    }
    return inputLayoutStr.substr(underLinePos + 1);
}


static ge::graphStatus CheckInputLayoutMla(const gert::TilingContext *context, const string inputLayoutStr, const int64_t queryD, const bool isPageAttention)
{
    // MLA support layout
    bool isIfaMlaLayout = (inputLayoutStr == "BSH") || (inputLayoutStr == "BNSD") ||
        (inputLayoutStr == "BSND") || (inputLayoutStr == "BNSD_NBSD") || (inputLayoutStr == "BSND_NBSD") ||
        (inputLayoutStr == "BSH_NBSD") || (inputLayoutStr == "TND_NTD") || (inputLayoutStr == "TND");
    OP_CHECK_IF((!isIfaMlaLayout && (inputLayoutStr != "NTD_TND")),
        OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "inputlayout(%s) only support {BSH, BNSD, BSND, TND, BNSD_NBSD, "
                                                            "BSND_NBSD, BSH_NBSD, TND_NTD} in mla.", inputLayoutStr.c_str()),
        return ge::GRAPH_FAILED);

    if (inputLayoutStr == "TND") {
        bool ifaWithoutPA = ((queryD == 512)&& (!isPageAttention));
        OP_CHECK_IF(ifaWithoutPA,
            OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
            "Layout is %s, MLA enabled, PA must be enabled when query's D dimension is %ld.",
            inputLayoutStr.c_str(), queryD),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus IsMla(const gert::TilingContext *context)
{
    auto qRope = context->GetOptionalInputTensor(QUERY_ROPE_INDEX);
    auto kRope = context->GetOptionalInputTensor(KEY_ROPE_INDEX);
    OP_CHECK_IF((qRope != nullptr && kRope == nullptr),
        OP_LOGE(context->GetNodeName(), "keyRope is null, but queryRope exists, they should be both null or exist."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF((qRope == nullptr && kRope != nullptr),
        OP_LOGE(context->GetNodeName(), "queryRope is null, but keyRope exists, they should be both null or exist."),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckInputLayout(gert::TilingContext *context, const string inputLayoutStr,const int64_t queryS, const int64_t queryD, const bool isPageAttention)
{
    auto tempQ = context->GetInputShape(QUERY_INDEX);
    auto tempK = context->GetInputShape(KEY_INDEX);
    auto tempV = context->GetInputShape(VALUE_INDEX);
    // mla check
    auto qRope = context->GetOptionalInputTensor(QUERY_ROPE_INDEX);
    if (IsMla(context) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (qRope != nullptr) {
        if (CheckInputLayoutMla(context, inputLayoutStr, queryD, isPageAttention) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }
    if (inputLayoutStr == "BNSD_BSND") {
        OP_CHECK_IF((queryS == 1), 
            OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),"BNSD_BSND layout is not supported when S is 1!"),
            return ge::GRAPH_FAILED);
    }
    if (inputLayoutStr == "SH") {
        OP_LOGE(context->GetNodeName(), "SH layout is not supported!");
        return ge::GRAPH_FAILED;
    }
    // check Q DimNum
    if (inputLayoutStr == "BSH" || inputLayoutStr == "BSH_NBSD" || inputLayoutStr == "TND" || 
        inputLayoutStr == "TND_NTD" || inputLayoutStr == "NTD_TND" || inputLayoutStr == "NSD") {
        OP_CHECK_IF((tempQ->GetStorageShape().GetDimNum() != DIM_NUM_3),
            OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "Layout is %s, queryDims must be 3! but actual value is %zu.\n",
            inputLayoutStr.c_str(), tempQ->GetStorageShape().GetDimNum()), return ge::GRAPH_FAILED);
    }
    else {
        OP_CHECK_IF((tempQ->GetStorageShape().GetDimNum() != DIM_NUM_4),
            OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "Layout is %s, queryDims must be 4! but actual value is %zu.\n",
            inputLayoutStr.c_str(), tempQ->GetStorageShape().GetDimNum()), return ge::GRAPH_FAILED);
    }
    // check KV DimNum when PA
    if (isPageAttention == false) {
        if (inputLayoutStr == "TND" || inputLayoutStr == "NTD_TND") {
            OP_CHECK_IF(((tempK->GetStorageShape().GetDimNum() != DIM_NUM_3) || (tempV->GetStorageShape().GetDimNum() != DIM_NUM_3)),
                OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "Layout is %s, key and value dims must be %zu! but actual value is keydim(%zu) valuedim(%zu).\n",
                inputLayoutStr.c_str(), DIM_NUM_3, tempK->GetStorageShape().GetDimNum(), tempV->GetStorageShape().GetDimNum()), return ge::GRAPH_FAILED);
        }
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckOutShape(gert::TilingContext *context, const string &outputLayoutStr, const gert::Shape outShape, const gert::Shape qkvShapeInfo)
{
    int64_t b = qkvShapeInfo.GetDim(DIM_0);
    int64_t queryN = qkvShapeInfo.GetDim(DIM_1);
    int64_t queryS = qkvShapeInfo.GetDim(DIM_2);
    int64_t queryT = qkvShapeInfo.GetDim(DIM_3);
    int64_t valueD = qkvShapeInfo.GetDim(DIM_4);
    ge::graphStatus ret = ge::GRAPH_SUCCESS;
    if (outputLayoutStr == "NSD") {
        ret = CheckOutShapeInDim3(context, outputLayoutStr, outShape, gert::Shape{queryN, queryS, valueD});
    } else if (outputLayoutStr == "BSH") {
        ret = CheckOutShapeInDim3(context, outputLayoutStr, outShape, gert::Shape{b, queryS, valueD * queryN});
    }else if (outputLayoutStr == "NBSD") {
        ret = CheckOutShapeInDim4(context, outputLayoutStr, outShape, gert::Shape{queryN, b, queryS, valueD});
    } else if (outputLayoutStr == "BSND") {
        ret = CheckOutShapeInDim4(context, outputLayoutStr, outShape, gert::Shape{b, queryS, queryN, valueD});
    } else if (outputLayoutStr == "BNSD") {
        ret = CheckOutShapeInDim4(context, outputLayoutStr, outShape, gert::Shape{b, queryN, queryS, valueD});
    } else if (outputLayoutStr == "NTD") {
        ret = CheckOutShapeInDim3(context, outputLayoutStr, outShape, gert::Shape{queryN, queryT, valueD});
    } else if (outputLayoutStr == "TND") {
        ret = CheckOutShapeInDim3(context, outputLayoutStr, outShape, gert::Shape{queryT, queryN, valueD});
    } else {
        OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
        "Not support outputLayout(%s)",
        outputLayoutStr.c_str());
        ret = ge::GRAPH_FAILED;
    }
    return ret;
}

ge::graphStatus TilingFusedInferAttentionScore(gert::TilingContext *context)
{
    // if (context == nullptr) {
    //     OP_LOGE("FusedInferAttentionScore", "tiling context is nullptr!");
    //     return ge::GRAPH_FAILED;
    // }
    // printf("1111111111111111\n");
    // if (RouteToFia(context)) {
    //     return TilingFusedInferAttentionScoreV3(context);
    // }
    // printf("2111111111111111\n");
    // OP_CHECK_IF(CheckQKV(*context) != ge::GRAPH_SUCCESS,
    //     OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "check query/key/value failed"), return ge::GRAPH_FAILED);
    // auto attrs = context->GetAttrs();
    // OP_CHECK_IF(attrs == nullptr, OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
    //     "Attributes returned from GetAttrs() is a nullptr"), return ge::GRAPH_FAILED);
    // const string inputLayoutStr = string(attrs->GetAttrPointer<char>(ATTR_INPUT_LAYOUT_INDEX));
    // // 获取关键轴信息
    // int64_t b = 0;
    // int64_t queryN = 0;
    // int64_t queryS = 0;
    // int64_t queryT = 0;
    // int64_t queryD = 1;
    // int64_t valueD = 0;
    // bool isPageAttention = context->GetOptionalInputShape(BLOCK_TABLE_INDEX) != nullptr ? true : false;
    // if ((GetB(context, inputLayoutStr, b) != ge::GRAPH_SUCCESS) ||
    //     (GetQueryN(context, inputLayoutStr, queryN) != ge::GRAPH_SUCCESS) ||
    //     (GetQueryS(context, inputLayoutStr, queryS) != ge::GRAPH_SUCCESS) ||
    //     (GetQueryT(context, inputLayoutStr, queryT) != ge::GRAPH_SUCCESS) ||
    //     (GetValueD(context, inputLayoutStr, valueD, isPageAttention) != ge::GRAPH_SUCCESS) ||
    //     (GetQueryD(context, inputLayoutStr, queryD) != ge::GRAPH_SUCCESS)) {
    //     return ge::GRAPH_FAILED;
    // }
    // // 校验intput
    // OP_CHECK_IF(CheckInputLayout(context, inputLayoutStr, queryS, queryD, isPageAttention) != ge::GRAPH_SUCCESS,
    //     OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "check InputLayout failed"), return ge::GRAPH_FAILED);
    // // 校验OutShape
    // string outputLayoutStr = GetOutputLayoutStr(inputLayoutStr);
    // auto outShape = context->GetOutputShape(ATTENTION_OUT_INDEX)->GetStorageShape();
    // gert::Shape qkvShapeInfo{b, queryN, queryS, queryT, valueD};
    // OP_CHECK_IF(CheckOutShape(context, outputLayoutStr, outShape, qkvShapeInfo) != ge::GRAPH_SUCCESS,
    //     OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "check output shape failed"), return ge::GRAPH_FAILED);
    // // 是否路由到IFA
    // bool usingIFA = IsUsingIFA(*context, inputLayoutStr, queryD, queryS);
    // if (usingIFA) { // IFA tiling process
        OP_CHECK_IF(TilingProcess4IFA(context) != ge::GRAPH_SUCCESS,
                    OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "tiling process for ifa failed"),
                    return ge::GRAPH_FAILED);
    // }
    return ge::GRAPH_SUCCESS;
}

FIA_EXTERN_C ge::graphStatus DoOpTilingFusedInferAttentionScore(gert::TilingContext *context)
{
    printf("start fia tiling\n");
    OP_CHECK_IF(context == nullptr,
        OPS_REPORT_VECTOR_INNER_ERR("FusedInferAttentionScore", "Tiling context is null."),
        return ge::GRAPH_FAILED);
    auto platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_IF(platformInfoPtr == nullptr,
        OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "platformInfoPtr is null"),
        return ge::GRAPH_FAILED);
    
    return TilingFusedInferAttentionScore(context);
}

extern "C" {
__attribute__((visibility("default"))) ge::graphStatus DeviceDoOpTilingIncreFlashAttention(gert::TilingContext *context)
{

}
__attribute__((visibility("default"))) ge::graphStatus DeviceDoOpTilingFusedInferAttentionScore(
    gert::TilingContext *context)
{
    return DoOpTilingFusedInferAttentionScore(context);
}
}
} // namespace optiling