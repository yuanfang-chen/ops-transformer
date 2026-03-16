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
 * \file fused_infer_attention_score_tiling_v3.cpp
 * \brief
 */

#include "fused_infer_attention_score_tiling_v3.h"
#include "fused_infer_attention_score_tiling_check.h"
#include "fused_infer_attention_score_tiling_info_parser.h"
#include "../../../common/op_host/arch32/fia_tiling_nonquant_mla.h"
#include "../../../common/op_host/arch32/fia_tiling_nonquant.h"
#include "../../../common/op_host/arch32/fia_tiling_empty_tensor.h"
#include "../../../common/op_host/fia_tiling_templates_registry.h"

using namespace AscendC;
namespace optiling {
// FIA新TilingKey, 18位编码, IFA原有TilingKey是17位, 新的TilingKey只是把最高位从1X->10X


REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_104000000000000000, FusedInferAttentionScoreTilingData)

constexpr size_t DIM_NZ = 5;
constexpr uint32_t NZ_D1_IDX = 2;
constexpr uint32_t NZ_D0_IDX = 4;
constexpr uint32_t TND_NTD_D_IDX = 2;
constexpr int64_t HEAD_DIM_192 = 192;


FIA_EXTERN_C ge::graphStatus TilingFusedInferAttentionScoreV3(gert::TilingContext *context)
{
    FiaTilingInfo fiaInfo;
    FiaInfoParser fiaInfoParser(context);
    if (fiaInfoParser.Parse(fiaInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // Check函数只做校验，不能修改fiaInfo中的信息
    if (TilingCheck::Check(fiaInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    return FiaTilingRegistry::GetInstance().DoTilingImpl(context, &fiaInfo);
}

bool GetPaValueD(const gert::TilingContext *context, int64_t &valueD)
{
    auto attrs = context->GetAttrs();
    int64_t numHeads = static_cast<int64_t>(*attrs->GetAttrPointer<uint32_t>(ATTR_N_INDEX));
    int64_t numKvHeads = static_cast<int64_t>(*attrs->GetAttrPointer<uint32_t>(ATTR_NUM_KV_HEADS_INDEX));
    if (numKvHeads == 0) {
        numKvHeads = numHeads;
    }
    auto vStorageShape = context->GetInputShape(VALUE_INDEX)->GetStorageShape();
    if (vStorageShape.GetDimNum() == DIM_BSH) {
        valueD = vStorageShape.GetDim(BSH_H_IDX) / numKvHeads; // BnBsH
    } else if (vStorageShape.GetDimNum() == DIM_BNSD_OR_BSND) {
        valueD = vStorageShape.GetDim(BNSD_D_IDX); // BnNBsD
    } else if (vStorageShape.GetDimNum() == DIM_NZ) {
        valueD = vStorageShape.GetDim(NZ_D1_IDX) * vStorageShape.GetDim(NZ_D0_IDX); // NZ: BnND1BsD0
    } else {
        return false;
    }
    return true;
}

bool GetValueD(gert::TilingContext *context, int64_t &valueD)
{
    auto vShape = context->GetInputShape(VALUE_INDEX);
    if (vShape == nullptr) {
        return false;
    }
    auto vStorageShape = vShape->GetStorageShape();

    bool isPageAttention = context->GetOptionalInputShape(BLOCK_TABLE_INDEX) != nullptr;
    if (isPageAttention) {
        return GetPaValueD(context, valueD);
    }

    auto attrs = context->GetAttrs();
    int64_t numHeads = static_cast<int64_t>(*attrs->GetAttrPointer<uint32_t>(ATTR_N_INDEX));
    int64_t numKvHeads = static_cast<int64_t>(*attrs->GetAttrPointer<uint32_t>(ATTR_NUM_KV_HEADS_INDEX));
    if (numKvHeads == 0) {
        numKvHeads = numHeads;
    }
    const std::string inputLayoutStr = std::string(context->GetAttrs()->GetAttrPointer<char>(ATTR_INPUT_LAYOUT_INDEX));
    if (inputLayoutStr == "BNSD_BSND" ||
        inputLayoutStr == "BSND_BNSD" ||
        inputLayoutStr == "BNSD_NBSD" ||
        inputLayoutStr == "BSND_NBSD" ||
        inputLayoutStr == "BNSD" ||
        inputLayoutStr == "BSND") {
        if (vStorageShape.GetDimNum() != DIM_BNSD_OR_BSND) {
            return false;
        }
        valueD = vStorageShape.GetDim(BNSD_D_IDX);
    } else if (inputLayoutStr == "BSH" ||
        inputLayoutStr == "BSH_NBSD" ||
        inputLayoutStr == "BSH_BNSD") {
        if (vStorageShape.GetDimNum() != DIM_BSH) {
            return false;
        }
        valueD = vStorageShape.GetDim(BSH_H_IDX) / numKvHeads;
    } else if (inputLayoutStr == "TND" ||
        inputLayoutStr == "NTD" ||
        inputLayoutStr == "TND_NTD" ||
        inputLayoutStr == "NTD_TND") {
        if (vStorageShape.GetDimNum() != DIM_TND) {
            return false;
        }
        valueD = vStorageShape.GetDim(TND_NTD_D_IDX);
    } else {
        return false;
    }

    return true;
}

bool GetQS(const gert::TilingContext *context, int64_t &queryS) {
    const std::string inputLayoutStr = std::string(context->GetAttrs()->GetAttrPointer<char>(ATTR_INPUT_LAYOUT_INDEX));
    auto qShape = context->GetInputShape(QUERY_INDEX);
    if (qShape == nullptr) {
        return false;
    }
    auto qStorageShape = qShape->GetStorageShape();

    if (inputLayoutStr == "BNSD") {
        if (qStorageShape.GetDimNum() != DIM_BNSD_OR_BSND) {
            return false;
        }
        queryS = qStorageShape.GetDim(BNSD_S_IDX);
    } else if (inputLayoutStr == "BSND") {
        if (qStorageShape.GetDimNum() != DIM_BNSD_OR_BSND) {
            return false;
        }
        queryS = qStorageShape.GetDim(BSND_S_IDX);
    } else if (inputLayoutStr == "BSH") {
        if (qStorageShape.GetDimNum() != DIM_BSH) {
            return false;
        }
        queryS = qStorageShape.GetDim(BSH_S_IDX);
    }

    return true;
}

bool GetQkvD(gert::TilingContext *context, int64_t &queryD, int64_t &queryRopeD, int64_t &valueD)
{
    auto qShape = context->GetInputShape(QUERY_INDEX);
    auto qRopeShape = context->GetOptionalInputShape(QUERY_ROPE_INDEX);
    if (qShape == nullptr) {
        return false;
    }
    auto qStorageShape = qShape->GetStorageShape();
    
    auto attrs = context->GetAttrs();
    if (attrs == nullptr) {
        return false;
    }

    int64_t numHeads = static_cast<int64_t>(*attrs->GetAttrPointer<uint32_t>(ATTR_N_INDEX));
    const std::string inputLayoutStr = std::string(context->GetAttrs()->GetAttrPointer<char>(ATTR_INPUT_LAYOUT_INDEX));
    if (inputLayoutStr == "BNSD_BSND" ||
        inputLayoutStr == "BSND_BNSD" ||
        inputLayoutStr == "BNSD_NBSD" ||
        inputLayoutStr == "BSND_NBSD" ||
        inputLayoutStr == "BNSD" ||
        inputLayoutStr == "BSND") {
        if (qStorageShape.GetDimNum() != DIM_BNSD_OR_BSND) {
            return false;
        }
        queryD = qStorageShape.GetDim(BNSD_D_IDX);
        if (qRopeShape != nullptr) {
            queryRopeD = qRopeShape->GetStorageShape().GetDim(BNSD_D_IDX);
        }
    } else if (inputLayoutStr == "BSH" ||
        inputLayoutStr == "BSH_NBSD" ||
        inputLayoutStr == "BSH_BNSD") {
        if (qStorageShape.GetDimNum() != DIM_BSH) {
            return false;
        }
        queryD = qStorageShape.GetDim(BSH_H_IDX) / numHeads;
        if (qRopeShape != nullptr) {
            queryRopeD = qRopeShape->GetStorageShape().GetDim(BSH_H_IDX) / numHeads;
        }
    } else if (inputLayoutStr == "TND" ||
        inputLayoutStr == "NTD" ||
        inputLayoutStr == "TND_NTD" ||
        inputLayoutStr == "NTD_TND") {
        if (qStorageShape.GetDimNum() != DIM_TND) {
            return false;
        }
        queryD = qStorageShape.GetDim(TND_NTD_D_IDX);
        if (qRopeShape != nullptr) {
            queryRopeD = qRopeShape->GetStorageShape().GetDim(TND_NTD_D_IDX);
        }
    } else {
        return false;
    }

    return GetValueD(context, valueD);
}

bool CheckGqaDSupport(gert::TilingContext *context)
{
    int64_t queryD = 0;
    int64_t queryRopeD = 0;
    int64_t valueD = 0;
    if (GetQkvD(context, queryD, queryRopeD, valueD) != true) {
        return false;
    }
    if ((queryD  == 128 && queryRopeD  == 0 && valueD == 128) || // 128: gqa qkvD
        (queryD  == 64 && queryRopeD  == 0 && valueD == 64) || // 64: gqa qkvD
        (queryD  == 128 && queryRopeD  == 64 && valueD == 128) || // 128: gqa qkvD, 64: mla ropeD
        (queryD  == 192 && queryRopeD  == 0 && valueD == 128)) { // 192: gqa qkD, 128: gqa valueD
        return true;
    }
    return false;
}

bool CheckGqaInputLayoutSupport(const gert::TilingContext *context)
{
    const std::string inputLayoutStr = std::string(context->GetAttrs()->GetAttrPointer<char>(ATTR_INPUT_LAYOUT_INDEX));
    if (inputLayoutStr == "BNSD_BSND" ||
        inputLayoutStr == "BSND_BNSD" ||
        inputLayoutStr == "BNSD" ||
        inputLayoutStr == "BSND" ||
        inputLayoutStr == "BSH_BNSD" ||
        inputLayoutStr == "BSH" ||
        inputLayoutStr == "TND" ||
        inputLayoutStr == "NTD" ||
        inputLayoutStr == "NTD_TND") {
        return true;
    }

    return false;
}

bool IsEmptyTensor(const gert::TilingContext *context)
{
    auto qShape = context->GetInputShape(QUERY_INDEX);
    if ((qShape != nullptr) && (qShape->GetStorageShape().GetShapeSize() == 0)) {
        return true;
    }

    auto attenoutShape = context->GetInputShape(ATTENTION_OUT_INDEX);
    if ((attenoutShape != nullptr) && (attenoutShape->GetStorageShape().GetShapeSize() == 0)) {
        return true;
    }


    uint32_t keyBIdx = 0;
    while ((context->GetDynamicInputShape(KEY_INDEX, keyBIdx)) != nullptr) {
        const gert::StorageShape *keyShape = context->GetDynamicInputShape(KEY_INDEX, keyBIdx);
        if (keyShape->GetStorageShape().GetShapeSize() == 0) {
            return true;
        }
        keyBIdx++;
    }

    uint32_t valueBIdx = 0;
    while ((context->GetDynamicInputShape(VALUE_INDEX, valueBIdx)) != nullptr) {
        const gert::StorageShape *valueShape = context->GetDynamicInputShape(VALUE_INDEX, valueBIdx);
        if (valueShape->GetStorageShape().GetShapeSize() == 0) {
            return true;
        }
        valueBIdx++;
    }

    return false;
}

bool CheckGqaFeatureSupport(const gert::TilingContext *context)
{
    auto quantScale2 = context->GetOptionalInputTensor(QUANT_SCALE2_INDEX);
    auto quantOffset2 = context->GetOptionalInputTensor(QUANT_OFFSET2_INDEX);
    if (quantScale2 != nullptr ||
        quantOffset2 != nullptr) {
        return false;
    }

    return true;
}

bool CheckSpecConditions(const gert::TilingContext *context)
{
    constexpr int64_t BLOCKSIZE_ALIGN_16 = 16;
    constexpr int64_t MAX_BLOCKSIZE = 512;
    auto tempQ = context->GetInputShape(QUERY_INDEX);
    auto tempK = context->GetInputShape(KEY_INDEX);
    auto tempV = context->GetInputShape(VALUE_INDEX);
    auto kvDimNum = tempK->GetStorageShape().GetDimNum();
    auto qRope = context->GetOptionalInputTensor(QUERY_ROPE_INDEX);
    auto kRope = context->GetOptionalInputTensor(KEY_ROPE_INDEX);
    auto tempAttnMaskShape = context->GetOptionalInputShape(ATTEN_MASK_INDEX);
    auto qDataType = context->GetInputDesc(QUERY_INDEX)->GetDataType();

    auto attrs = context->GetAttrs();
    string inputLayoutStr = string(attrs->GetAttrPointer<char>(ATTR_INPUT_LAYOUT_INDEX));
    int32_t headNum = *(attrs->GetAttrPointer<int32_t>(ATTR_N_INDEX));
    int32_t kvHeadNum = *(attrs->GetAttrPointer<int32_t>(ATTR_NUM_KV_HEADS_INDEX));
    int32_t innerPrecise = *(attrs->GetAttrPointer<int32_t>(ATTR_INNER_PRECISE_INDEX));
    int32_t sparseMode = *(attrs->GetAttrPointer<int32_t>(ATTR_SPARSE_MODE_INDEX));
    
    bool isLayoutSupported = (inputLayoutStr == "TND");
    bool isPageAttention = (context->GetOptionalInputShape(BLOCK_TABLE_INDEX) != nullptr);
    bool isLearnableSink = (context->GetOptionalInputTensor(LEARNABLE_SINK_INDEX) != nullptr);
    bool isLearnableSinkFlag = true;
    if (isLearnableSink && isLayoutSupported) {
        int64_t tempQHeadDim = tempQ->GetStorageShape().GetDim(DIM_2);
        auto sinkDataType = context->GetOptionalInputDesc(LEARNABLE_SINK_INDEX)->GetDataType();
        if (tempQHeadDim == 64 && sinkDataType == ge::DT_BF16) { // 64: qD need 64, condition to set sinkflag to disable
            isLearnableSinkFlag = false;
        }
    }

    bool sparseModeSupported = (sparseMode == 0) || (sparseMode == 3) || (sparseMode == 4);
    bool isRopeSplitMla = (qRope != nullptr) && (kRope != nullptr);
    
    bool isMha = (kvHeadNum == 0) || (headNum == kvHeadNum);
    bool mhaConditions = isMha && !((qDataType == ge::DT_BF16) && (innerPrecise == 1)) && 
        !((sparseMode == 0) && (tempAttnMaskShape != nullptr));
    bool nonMhaConditions = !isMha && (innerPrecise == 0);
    bool specConditionFlag = false;
    bool quantScale2Flag = context->GetOptionalInputTensor(QUANT_SCALE2_INDEX) != nullptr ? true : false;
    if (isLayoutSupported && isLearnableSinkFlag && !isRopeSplitMla && sparseModeSupported &&
        (nonMhaConditions || mhaConditions) && !quantScale2Flag) {
        int64_t tempQD = tempQ->GetStorageShape().GetDim(DIM_2);
        if (!isPageAttention) {
            int64_t tempKD = tempK->GetStorageShape().GetDim(DIM_2);
            int64_t tempVD = tempV->GetStorageShape().GetDim(DIM_2);
            bool isFAIDSize = (tempQD <= 256 && tempKD <= 256 && tempVD <= 256) &&
                    (tempQD == tempKD && tempQD == tempVD);
            if (isFAIDSize) {
                specConditionFlag = true;
            }
        } else if (kvDimNum == 3U) {
            int64_t tempKD = (tempK->GetStorageShape().GetDim(DIM_2)) / kvHeadNum;
            int64_t tempVD = (tempV->GetStorageShape().GetDim(DIM_2)) / kvHeadNum;
            int64_t blockSize = tempK->GetStorageShape().GetDim(DIM_1);
            bool isFAIDSize = (tempQD <= 256 && tempKD <= 256 && tempVD <= 256) &&
                    (tempQD == tempKD && tempQD == tempVD);
            bool blockSizeSupported = (blockSize % BLOCKSIZE_ALIGN_16 == 0) && 
                    (blockSize <= MAX_BLOCKSIZE);
            if (isFAIDSize && blockSizeSupported) {
                specConditionFlag = true;
            }
        }
    }
    return specConditionFlag;
}

bool isNotLegacyGQA(gert::TilingContext *context)
{
    const std::string inputLayoutStr = std::string(context->GetAttrs()->GetAttrPointer<char>(ATTR_INPUT_LAYOUT_INDEX));
    if (inputLayoutStr != "BSH" &&
        inputLayoutStr != "BSND" &&
        inputLayoutStr != "BNSD" &&
        inputLayoutStr != "BNSD_BSND" &&
        inputLayoutStr != "TND" &&
        inputLayoutStr != "NSD") {
        return true;
    }

    auto queryRope = context->GetOptionalInputTensor(QUERY_ROPE_INDEX);
    auto keyRope = context->GetOptionalInputTensor(KEY_ROPE_INDEX);
    int64_t queryD = 0;
    int64_t queryRopeD = 0;
    int64_t valueD = 0;
    if (GetQkvD(context, queryD, queryRopeD, valueD) != true) {
        return false;
    }

    if (queryD != valueD || queryRope != nullptr || keyRope != nullptr) {
        return true;
    }

    if (inputLayoutStr == "TND" && valueD == HEAD_DIM_192) {
        return false;
    }

    uint32_t keyDimNum = context->GetInputShape(KEY_INDEX)->GetStorageShape().GetDimNum();
    uint32_t valueDimNum = context->GetInputShape(VALUE_INDEX)->GetStorageShape().GetDimNum();
    if ((keyDimNum != DIM_NUM_3 && keyDimNum != DIM_NUM_4) ||
        (valueDimNum != DIM_NUM_3 && valueDimNum != DIM_NUM_4)) {
        return true;
    }

    return false;
}

bool CheckGqaConstrain(gert::TilingContext *context)
{
    if (isNotLegacyGQA(context)) {
        return true;
    }

    if (CheckGqaInputLayoutSupport(context) &&
        CheckGqaDSupport(context) && 
        CheckGqaFeatureSupport(context)) { 
            return true;
    }

    return false;
}


bool CheckMlaInputLayoutSupport(const gert::TilingContext *context)
{
    const std::string inputLayoutStr = std::string(context->GetAttrs()->GetAttrPointer<char>(ATTR_INPUT_LAYOUT_INDEX));
    if (inputLayoutStr == "BSH" ||
        inputLayoutStr == "BNSD" ||
        inputLayoutStr == "BSND" ||
        inputLayoutStr == "BNSD_NBSD" ||
        inputLayoutStr == "BSND_NBSD" ||
        inputLayoutStr == "BSH_NBSD" ||
        inputLayoutStr == "TND" ||
        inputLayoutStr == "TND_NTD") {
        return true;
    }

    return false;
}

bool CheckMlaDSupport(gert::TilingContext *context)
{
    int64_t queryD = 0;
    int64_t queryRopeD = 0;
    int64_t valueD = 0;
    if (GetQkvD(context, queryD, queryRopeD, valueD) != true) {
        return false;
    }

    if ((queryD  == 512 && queryRopeD  == 64 && valueD == 512)) { // 512: mla qkvD, 64: mla ropeD
        return true;
    }

    return false;
}

bool CheckMlaConstrain(gert::TilingContext *context)
{
    if (isNotLegacyGQA(context)) {
        return true;
    }

    if (CheckMlaInputLayoutSupport(context) &&
        CheckMlaDSupport(context)) {
        return true;
    }

    return false;
}

bool RouteToFia(gert::TilingContext *context)
{
    if ((context == nullptr) || context->GetAttrs() == nullptr ||
        (context->GetInputDesc(QUERY_INDEX) == nullptr) ||
        (context->GetInputDesc(KEY_INDEX) == nullptr)) {
        return false;
    }
    auto platformInfoPtr = context->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    if (ascendcPlatform.GetSocVersion() == platform_ascendc::SocVersion::ASCEND310P) {
        return false;
    }

    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    if((aicNum != aivNum) && (aicNum * 2U != aivNum)) {
        OP_LOGI(context->GetNodeName(), "aicNum(%u):aivNum(%u) only support 1:1 or 1:2.", aicNum, aivNum);
        return false;
    }

    ge::DataType qDataType = context->GetInputDesc(QUERY_INDEX)->GetDataType();
    ge::DataType kDataType = context->GetInputDesc(KEY_INDEX)->GetDataType();
    bool isRopeSplit = (context->GetOptionalInputTensor(QUERY_ROPE_INDEX) != nullptr &&
        context->GetOptionalInputTensor(KEY_ROPE_INDEX) != nullptr);

    if ((qDataType == ge::DT_FLOAT16 || qDataType == ge::DT_BF16) && (qDataType == kDataType)) {
        auto attrs = context->GetAttrs();
        int32_t headNum = *(attrs->GetAttrPointer<int32_t>(ATTR_N_INDEX));
        int32_t kvHeadNum = *(attrs->GetAttrPointer<int32_t>(ATTR_NUM_KV_HEADS_INDEX));
        bool isMha = (kvHeadNum == 0) || (headNum == kvHeadNum);
        bool isPageAttention = (context->GetOptionalInputShape(BLOCK_TABLE_INDEX) != nullptr);
        bool isPrefix = (context->GetOptionalInputShape(KEY_SHARED_PREFIX_INDEX) != nullptr) ||
                        (context->GetOptionalInputShape(VALUE_SHARED_PREFIX_INDEX) != nullptr);
    
        int64_t queryD = 0;
        int64_t queryRopeD = 0;
        int64_t valueD = 0;
        int64_t queryS = 0;
        if (!GetQkvD(context, queryD, queryRopeD, valueD)) {
            return false;
        }
        if (!GetQS(context, queryS)) {
            return false;
        }
    
        const std::string inputLayoutStr = std::string(context->GetAttrs()->GetAttrPointer<char>(ATTR_INPUT_LAYOUT_INDEX));
        // 部分场景性能在重构前的模板性能更好，路由到老模板处理
        if (queryD == valueD && queryRopeD == 0 && queryS == 1 &&
            (inputLayoutStr == "BNSD" || inputLayoutStr == "BSND" || inputLayoutStr == "BSH") &&
            ((queryD == 256U && !isPrefix) || (queryD == 80U && isPrefix)) &&
            isMha && !isPageAttention) {
            return false;
        }
        if (!isRopeSplit) {
            if (CheckSpecConditions(context)) {
                return false;
            }
        }
        if (inputLayoutStr == "NSD") {
            return false;
        }
        return true;
    }
    return false;
}
} // namespace optiling
