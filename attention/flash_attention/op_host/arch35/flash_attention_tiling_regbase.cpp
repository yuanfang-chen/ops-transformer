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
 * \file flash_attention_tiling_regbase.cpp
 * \brief FlashAttention arch35 tiling基类实现（非量化场景框架，具体计算待补充）
 */

#include "flash_attention_tiling_regbase.h"
#include "../flash_attention_tiling_common.h"

namespace optiling {
namespace FA {

void FlashAttentionTilingRegbase::Reset()
{
    inputDtype = ge::DT_FLOAT16;
    inputDtypeBytes = ge::GetSizeByDataType(ge::DT_FLOAT16);
    calcTypeSize = ge::GetSizeByDataType(ge::DT_FLOAT);
    isHighPrecision = true;
    tilingKeyDType = DtypeEnum::FLOAT16;
    bmmDtype = matmul_tiling::DataType::DT_FLOAT16;
    bmm1OutDtype = matmul_tiling::DataType::DT_FLOAT;
    bmm2OutDtype = matmul_tiling::DataType::DT_FLOAT;

    tilingKeyLayout = FALayoutType::NONE;
    tilingKeyKVLayout = FAKVLayoutType::BSND;
    implMode = FAImplMode::HIGH_PRECISION;
    inputLayoutQ = nullptr;
    inputLayoutKv = nullptr;
    inputLayoutOut = nullptr;

    bSize = gSize = dSize = dSizeV = n1Size = n2Size = s1Size = s2Size = 0LL;
    accumS1 = accumS2 = realT1Size = 0LL;

    softmaxScale = 0.0f;
    maskMode = 0LL;
    winLeft = 0LL;
    winRight = 0LL;
    returnSoftmaxLse = 0LL;
    deterministic = 0LL;

    hasAttenMask = false;
    isPA = false;
    isTND = false;
    hasVarLen = false;
    hasMetadata = false;

    blockSize = 0;
    blockTableDim2 = 0;
    paBlockNumSum = 0;
    paLayoutType = 0U;

    s1BasicBlock = FA_NUM_128;
    s2BasicBlock = FA_NUM_128;
    dBasicBlock = FA_NUM_128;
    dVBasicBlock = FA_NUM_128;
    dTemplateType = FADTemplateType::BOTTOM;
    dVTemplateType = FADTemplateType::BOTTOM;

    opName = nullptr;
}

ge::graphStatus FlashAttentionTilingRegbase::CheckContext()
{
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);

    auto queryShape = context_->GetInputShape(FA_INPUT_Q_INDEX);
    auto keyShape = context_->GetInputShape(FA_INPUT_K_INDEX);
    auto valueShape = context_->GetInputShape(FA_INPUT_V_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, queryShape);
    OP_CHECK_NULL_WITH_CONTEXT(context_, keyShape);
    OP_CHECK_NULL_WITH_CONTEXT(context_, valueShape);
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetRawTilingData());
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetRawTilingData()->GetData());

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FlashAttentionTilingRegbase::GetPlatformInfo()
{
    auto platformInfoPtr = context_->GetPlatformInfo();
    if (platformInfoPtr == nullptr) {
        auto compileInfoPtr = reinterpret_cast<const FlashAttentionCompileInfo *>(context_->GetCompileInfo());
        OP_CHECK_IF(compileInfoPtr == nullptr, OPS_REPORT_VECTOR_INNER_ERR(opName, "compileInfoPtr is null."),
                   return ge::GRAPH_FAILED);
        aivNum = compileInfoPtr->aivNum;
        aicNum = compileInfoPtr->aicNum;
        npuArch = compileInfoPtr->npuArch;
        socVersion = compileInfoPtr->socVersion;
        aicoreParams_.ubSize = compileInfoPtr->ubSize;
        aicoreParams_.l1Size = compileInfoPtr->l1Size;
        aicoreParams_.l0cSize = compileInfoPtr->l0cSize;
    } else {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
        aivNum = ascendcPlatform.GetCoreNumAiv();
        aicNum = ascendcPlatform.GetCoreNumAic();
        socVersion = ascendcPlatform.GetSocVersion();
        npuArch = ascendcPlatform.GetCurNpuArch();
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, aicoreParams_.ubSize);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1,  aicoreParams_.l1Size);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, aicoreParams_.l0cSize);
    }
    OP_LOGI(context_, "FlashAttention platform: aivNum(%u) aicNum(%u) ubSize(%lu) l1Size(%lu) l0cSize(%lu).",
            aivNum, aicNum, aicoreParams_.ubSize, aicoreParams_.l1Size, aicoreParams_.l0cSize);
    return ge::GRAPH_SUCCESS;
}

bool FlashAttentionTilingRegbase::AnalyzeDtype()
{
    inputDtype = context_->GetInputDesc(FA_INPUT_Q_INDEX)->GetDataType();
    inputDtypeBytes = ge::GetSizeByDataType(inputDtype);

    switch (inputDtype) {
        case ge::DT_FLOAT16:
            bmmDtype = matmul_tiling::DataType::DT_FLOAT16;
            bmm1OutDtype = isHighPrecision ? matmul_tiling::DataType::DT_FLOAT
                                            : matmul_tiling::DataType::DT_FLOAT16;
            tilingKeyDType = isHighPrecision ? DtypeEnum::FLOAT16_PRECISION : DtypeEnum::FLOAT16;
            calcTypeSize = isHighPrecision ? ge::GetSizeByDataType(ge::DT_FLOAT)
                                             : ge::GetSizeByDataType(inputDtype);
            break;
        case ge::DT_BF16:
            bmmDtype = matmul_tiling::DataType::DT_BF16;
            bmm1OutDtype = matmul_tiling::DataType::DT_FLOAT;
            tilingKeyDType = DtypeEnum::BFLOAT16;
            calcTypeSize = ge::GetSizeByDataType(ge::DT_FLOAT);
            isHighPrecision = false;
            break;
        default:
            OPS_REPORT_VECTOR_INNER_ERR(opName, "FlashAttention only supports FP16/BF16, got: %d.",
                                        static_cast<int>(inputDtype));
            return false;
    }
    bmm2OutDtype = bmm1OutDtype;
    return true;
}

bool FlashAttentionTilingRegbase::AnalyzeAttrs()
{
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);

    auto softmaxModePtr = attrs->GetAttrPointer<float>(FA_ATTR_SOFTMAX_MODE_INDEX);
    auto maskModePtr = attrs->GetAttrPointer<int64_t>(FA_ATTR_MASK_MODE_INDEX);
    auto winLeftPtr = attrs->GetAttrPointer<int64_t>(FA_ATTR_WIN_LEFT_INDEX);
    auto winRightPtr = attrs->GetAttrPointer<int64_t>(FA_ATTR_WIN_RIGHT_INDEX);
    inputLayoutQ = attrs->GetAttrPointer<char>(FA_ATTR_LAYOUT_Q_INDEX);
    inputLayoutKv = attrs->GetAttrPointer<char>(FA_ATTR_LAYOUT_KV_INDEX);
    inputLayoutOut = attrs->GetAttrPointer<char>(FA_ATTR_LAYOUT_OUT_INDEX);
    auto returnLsePtr = attrs->GetAttrPointer<int64_t>(FA_ATTR_RETURN_SOFTMAX_LSE);
    auto deterministicPtr = attrs->GetAttrPointer<int64_t>(FA_ATTR_DETERMINISTIC);

    OP_CHECK_NULL_WITH_CONTEXT(context_, softmaxModePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context_, maskModePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context_, winLeftPtr);
    OP_CHECK_NULL_WITH_CONTEXT(context_, winRightPtr);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputLayoutQ);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputLayoutKv);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputLayoutOut);
    OP_CHECK_NULL_WITH_CONTEXT(context_, returnLsePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context_, deterministicPtr);

    softmaxScale = *softmaxModePtr;
    maskMode = *maskModePtr;
    winLeft = *winLeftPtr;
    winRight = *winRightPtr;
    returnSoftmaxLse = *returnLsePtr;
    deterministic = *deterministicPtr;

    hasAttenMask = (maskMode != 0LL);

    implMode = FAImplMode::HIGH_PRECISION;

    OP_LOGD(context_, "FlashAttention attrs: softmaxScale[%f] maskMode[%ld] winLeft[%ld] winRight[%ld] "
            "layoutQ[%s] layoutKv[%s] layoutOut[%s] returnLse[%ld] deterministic[%ld].",
            softmaxScale, maskMode, winLeft, winRight,
            inputLayoutQ, inputLayoutKv, inputLayoutOut, returnSoftmaxLse, deterministic);
    return true;
}

bool FlashAttentionTilingRegbase::AnalyzeLayout()
{
    return true;
}

bool FlashAttentionTilingRegbase::AnalyzeVarLenInput()
{
    // TODO: 解析cuSeqlens或seqused
    return true;
}

bool FlashAttentionTilingRegbase::AnalyzePAInput()
{
    return true;
}

bool FlashAttentionTilingRegbase::AnalyzeMetadataInput()
{
    //TODO:metada 解析 
    auto metadataShape = context_->GetOptionalInputShape(FA_INPUT_METADATA_INDEX);
    if (metadataShape != nullptr && metadataShape->GetStorageShape().GetShapeSize() > 0) {
        hasMetadata = true;
        OP_LOGD(context_, "FlashAttention: metadata input is present, may use pre-computed tiling.");
    }
    return true;
}

ge::graphStatus FlashAttentionTilingRegbase::GetShapeAttrsInfo()
{
    opName = context_->GetNodeName();
    OP_LOGD(opName, "FlashAttention TilingContext: %s.", GetTilingContextDebugStr().c_str());

    OP_CHECK_IF(CheckContext() != ge::GRAPH_SUCCESS,
               OPS_REPORT_VECTOR_INNER_ERR(opName, "invalid context."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(!AnalyzeAttrs() || !AnalyzeDtype() || !AnalyzeLayout(),
               OPS_REPORT_VECTOR_INNER_ERR(opName, "fail to analyze attrs/dtype/layout."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(!AnalyzeVarLenInput() || !AnalyzePAInput() || !AnalyzeMetadataInput(),
               OPS_REPORT_VECTOR_INNER_ERR(opName, "fail to analyze optional inputs."), return ge::GRAPH_FAILED);

    // 填充基础参数到InputParamsRegbase
    inputParamsRegbase_->set_bSize(bSize);
    inputParamsRegbase_->set_n2Size(n2Size);
    inputParamsRegbase_->set_gSize(gSize);
    inputParamsRegbase_->set_s1Size(s1Size);
    inputParamsRegbase_->set_s2Size(s2Size);
    inputParamsRegbase_->set_dSize(dSize);
    inputParamsRegbase_->set_dSizeV(dSizeV);
    inputParamsRegbase_->set_scaleValue(softmaxScale);
    inputParamsRegbase_->set_isGqa(static_cast<uint8_t>(n1Size != n2Size));
    inputParamsRegbase_->set_isSoftMaxLseEnable(static_cast<uint8_t>(returnSoftmaxLse != 0));
    inputParamsRegbase_->set_needDropMaskOp(0U);  // 无dropout
    inputParamsRegbase_->set_isKvContinuous(static_cast<uint8_t>(!isPA));

    OP_LOGD(context_, "FlashAttention shape: B=%ld N_q=%ld N_kv=%ld S_q=%ld S_kv=%ld D=%ld Dv=%ld isPA=%d.",
            bSize, n1Size, n2Size, s1Size, s2Size, dSize, dSizeV, static_cast<int>(isPA));
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FlashAttentionTilingRegbase::DoOpTiling()
{
    // TODO: dotiling填写
    // 参照flash_attention_score_tiling_regbase.cpp中DoOpTiling的框架
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FlashAttentionTilingRegbase::DoLibApiTiling()
{
    // TODO: 
    return ge::GRAPH_SUCCESS;
}

void FlashAttentionTilingRegbase::CalcDVBasicBlock()
{
    // TODO:
}

int64_t FlashAttentionTilingRegbase::CalcTotalSize()
{
    // bSize * n2Size * gSize * multiCoreParamsRegbase_->get_s1OuterSize()
    return 0;
}

ge::graphStatus FlashAttentionTilingRegbase::PostTiling()
{
    // 设置TilingKey和BlockDim，由子类GetTilingKey()提供key
    uint64_t tilingKey = GetTilingKey();
    context_->SetTilingKey(tilingKey);
    OP_LOGD(context_, "FlashAttention PostTiling: tilingKey=0x%lx.", tilingKey);

    // 设置TilingData大小
    // context_->GetRawTilingData()->SetDataSize(sizeof(FlashAttentionScoreSimplifiedTilingData));
    return ge::GRAPH_SUCCESS;
}

} // namespace FA
} // namespace optiling