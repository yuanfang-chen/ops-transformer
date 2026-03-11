/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file moe_finalize_routing_v2_tiling_arch35.cpp
 * \brief
 */
#include "tiling_base/tiling_util.h"
#include "moe_finalize_routing_v2_tiling.h"
#include "moe_finalize_routing_v2_tiling_arch35.h"

namespace optiling {
static constexpr int64_t TWO = 2;
static constexpr int64_t EXPANDED_X_IDX = 0;
static constexpr int64_t EXPANDED_ROW_IDX_IDX = 1;
static constexpr int64_t X1_IDX = 2;
static constexpr int64_t X2_IDX = 3;
static constexpr int64_t BIAS_IDX = 4;
static constexpr int64_t SCALES_IDX = 5;
static constexpr int64_t EXPERTIDX_IDX = 6;
static constexpr int64_t X_IDX = 7;
static constexpr int64_t CONST_EXPERT_ALPHA1_IDX = 8;
static constexpr int64_t CONST_EXPERT_ALPHA2_IDX = 9;
static constexpr int64_t CONST_EXPERT_V_IDX = 10;
static constexpr int64_t BIAS_DIM_NUM = 2;
static constexpr int64_t SCALES_DIM_NUM = 2;
static constexpr int64_t DROPLESS_EXPANDED_X_DIM_NUM = 2;
static constexpr int64_t DROPPAD_EXPANDED_X_DIM_NUM = 3;
static constexpr int64_t DOUBLE_BUFFER = 2;
static constexpr int64_t DROP_LESS_COL = 0;
static constexpr int64_t DROP_PAD_COL = 1;
static constexpr int64_t DROP_LESS_ROW = 2;
static constexpr int64_t DROP_PAD_ROW = 3;
static constexpr int64_t INPUT_BUFFER_NUM = 8;  // expaned_x bias x1 x2
static constexpr int64_t CONST_EXPERT_BUFFER_NUM = 3;  // expaned_x bias x1 x2
static constexpr int64_t OUTPUT_BUFFER_NUM = 1; // y
static constexpr uint64_t FULL_LOAD_H_BASE_TILING_KEY = 10000;
static constexpr uint64_t SPLIT_H_BASE_TILING_KEY = 20000;
static constexpr uint64_t FULL_LOAD_ROW_K_H_BASE_TILING_KEY = 30000;
static constexpr uint64_t FULL_LOAD_K_H_BASE_TILING_KEY = 40000;
static constexpr uint64_t DROPLESS_COL_TILING_LEY = 0;
static constexpr uint64_t DROPPAD_COL_TILING_KEY = 10;
static constexpr uint64_t DROPLESS_ROW_TILING_LEY = 20;
static constexpr uint64_t DROPPAD_ROW_TILING_KEY = 30;
static constexpr uint64_t FLOAT16_TILING_KEY = 1;
static constexpr uint64_t BFLOAT16_TILING_KEY = 2;
static constexpr size_t WORKSPACE_RESERVED = 16 * 1024 * 1024;

const static int64_t ATTR_DROP_PAD_MODE = 0LL;
const static int64_t ATTR_ZERO_EXPERT_RANGE = 1LL;
const static int64_t ATTR_COPY_EXPERT_RANGE = 2LL;
const static int64_t ATTR_CONSTANT_EXPERT_RANGE = 3LL;

class MoeFinalizeRoutingV2Regbase : public MoeFinalizeRoutingTilingV2
{
public:
    explicit MoeFinalizeRoutingV2Regbase(gert::TilingContext* context) : MoeFinalizeRoutingTilingV2(context)
    {}
    ~MoeFinalizeRoutingV2Regbase() override = default;

    void Reset(gert::TilingContext* context) override
    {
        MoeFinalizeRoutingTilingV2::Reset(context);
    }

protected:
    // 不用修改
    ge::graphStatus DoGetPlatformInfo() override;
    // 此处已修改
    ge::graphStatus DoGetShapeAttrsInfo() override;
    ge::graphStatus CalcOpTiling() override;
    // 不用修改
    ge::graphStatus CalcTilingKey() override;
    // 不用修改
    void DoPostTiling() override;
    // 不用修改
    void PrintTilingData() override;
    // 此处已修改
    bool IsCapable() override;

    bool IsRowKHFullLoad();
    bool IsKHFullLoad();
    bool IsHFullLoad();
    ge::graphStatus CheckShapeAndDtypeIsValid();
    ge::graphStatus CheckPartShapeAndDtypeIsValid();
    ge::graphStatus FinalCheckShapeAndDtypeIsValid();
    ge::graphStatus GetRow(const gert::StorageShape* expandedRowIdxShape);
    ge::graphStatus CheckBiasShape(const gert::StorageShape* biasShape);
    ge::graphStatus GetECH(const gert::StorageShape* expandedXShape);
    ge::graphStatus GetK(const gert::StorageShape* scalesShape);
    int64_t RowsHSize(int64_t rowFactor, bool scalesInUb);
    int64_t RowsHSizeForKHFullLoad(int64_t rowFactor, bool scalesInUb);
    int64_t CalcRowFactor(int64_t ubSizeRemained, bool scalesInUb);
    int64_t CalcRowFactorForKHFullLoad(int64_t ubSizeRemained, bool scalesInUb);
    void SetFullLoadTilingData(int64_t rowOfFormerBlock, int64_t rowOfTailBlock, int64_t rowFactor);
    ge::graphStatus DoOpTilingRowKHFullLoad(int64_t rowOfFormerBlock, int64_t rowOfTailBlock);
    ge::graphStatus DoOpTilingKHFullLoad(int64_t rowOfFormerBlock, int64_t rowOfTailBlock);
    ge::graphStatus DoOpTilingHFullLoad(int64_t rowOfFormerBlock, int64_t rowOfTailBlock);
    ge::graphStatus DoOpTilingSplitH(int64_t rowOfFormerBlock, int64_t rowOfTailBlock);

    bool hasX1_{false};
    bool hasX2_{false};
    bool hasScales_{false};
    bool hasBias_{false};
    bool hasX_{false};   // 有x输入，且constantEnd - constantStart >= 0
    bool hasConstantExpert_{false};
    bool rowKHFullLoad_{false};
    bool kHFullLoad_{false};
    bool hFullLoad_{false};
    uint32_t coreNum_{0};
    uint64_t blockSize_{0};
    uint64_t vlFp32_{0};
    uint64_t scaleDtypeKey{0};

    ge::DataType dtype{ge::DataType::DT_FLOAT};
    int64_t dtypeSize{0};
    int64_t scaleDtypeSize{0};
    int64_t dropPadMode{0};
    int64_t row{0};
    int64_t e{0};
    int64_t c{0};
    int64_t k{0};
    int64_t h{0};
    int64_t hAligned{0};
    int64_t dim0OfExpandedX{0};
    // 新增判断零专家、拷贝专家以及常量专家的范围
    int64_t zeroExpertStart{-1};
    int64_t zeroExpertEnd{-1};
    int64_t copyExpertStart{-1};
    int64_t copyExpertEnd{-1};
    int64_t constantExpertStart{-1};
    int64_t constantExpertEnd{-1};
    int64_t constExpertRangeNum{-1};
    MoeFinalizeRoutingV2RegbaseTilingData* tilingData{nullptr};
};

ge::graphStatus MoeFinalizeRoutingV2Regbase::DoGetPlatformInfo()
{
    OP_LOGI(context_->GetNodeName(), "MoeFinalizeRoutingV2, DoGetPlatformInfo");
    printf("MoeFinalizeRoutingV2, DoGetPlatformInfo\n");
    auto platformInfo = context_->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context_, platformInfo);

    blockSize_ = Ops::Base::GetUbBlockSize(context_);
    OP_CHECK_IF(
        blockSize_ == 0,
        OP_LOGE(context_->GetNodeName(), "Get blockSize failed, blockSize: %lu", blockSize_),
        return ge::GRAPH_FAILED);
    vlFp32_ = Ops::Base::GetVRegSize(context_) / sizeof(float);
    OP_CHECK_IF(
        vlFp32_ == 0, OP_LOGE(context_->GetNodeName(), "Get VL32 failed, VL32: %lu", vlFp32_),
        return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    coreNum_ = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(
        coreNum_ == 0,
        OP_LOGE(context_->GetNodeName(), "Get core num failed, core num: %u", coreNum_),
        return ge::GRAPH_FAILED);
    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(
        ubSize == 0, OP_LOGE(context_->GetNodeName(), "Get ubSize failed, ubSize: %lu", ubSize),
        return ge::GRAPH_FAILED);
    ubSize_ = static_cast<int64_t>(ubSize);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeFinalizeRoutingV2Regbase::GetK(const gert::StorageShape* scalesShape)
{
    OP_LOGI(context_->GetNodeName(), "MoeFinalizeRoutingV2, GetK");
    printf("MoeFinalizeRoutingV2, GetK\n");
    if (!scalesShape) {
        hasScales_ = false;
        k = 1;
        return ge::GRAPH_SUCCESS;
    }

    hasScales_ = true;
    OP_CHECK_IF(
        scalesShape->GetStorageShape().GetDimNum() != SCALES_DIM_NUM,
        OP_LOGE(context_->GetNodeName(), "dim num of scales should be 2."),
        return ge::GRAPH_FAILED);
    k = scalesShape->GetStorageShape().GetDim(1);
    OP_CHECK_IF(
        k <= 0, OP_LOGE(context_->GetNodeName(), "k[%ld] must be greater than 0.", k),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeFinalizeRoutingV2Regbase::GetECH(const gert::StorageShape* expandedXShape)
{
    OP_LOGI(context_->GetNodeName(), "MoeFinalizeRoutingV2, GetECH");
    printf("MoeFinalizeRoutingV2, GetECH\n");
    if (dropPadMode == DROP_LESS_ROW || dropPadMode == DROP_LESS_COL) {
        OP_CHECK_IF(
            expandedXShape->GetStorageShape().GetDimNum() != DROPLESS_EXPANDED_X_DIM_NUM,
            OP_LOGE(
                context_->GetNodeName(), "dim num of expanded_x should be 2 in dropless mode."),
            return ge::GRAPH_FAILED);
        dim0OfExpandedX = expandedXShape->GetStorageShape().GetDim(0);
        h = expandedXShape->GetStorageShape().GetDim(1);
        OP_CHECK_IF(
            h <= 0, OP_LOGE(context_->GetNodeName(), "h[%ld] must be greater than 0.", h),
            return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }

    OP_CHECK_IF(
        expandedXShape->GetStorageShape().GetDimNum() != DROPPAD_EXPANDED_X_DIM_NUM,
        OP_LOGE(context_->GetNodeName(), "dim num of expanded_x should be 3 in drop pad mode."),
        return ge::GRAPH_FAILED);
    e = expandedXShape->GetStorageShape().GetDim(0);
    c = expandedXShape->GetStorageShape().GetDim(1);
    h = expandedXShape->GetStorageShape().GetDim(TWO);
    OP_CHECK_IF(
        e <= 0 || c <= 0 || h <= 0,
        OP_LOGE(
            context_->GetNodeName(), "e[%ld],c[%ld],h[%ld] must be greater than 0.", e, c, h),
        return ge::GRAPH_FAILED);
    dim0OfExpandedX = e;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeFinalizeRoutingV2Regbase::GetRow(const gert::StorageShape* expandedRowIdxShape)
{
    OP_LOGI(context_->GetNodeName(), "MoeFinalizeRoutingV2, GetRow");
    printf("MoeFinalizeRoutingV2, GetRow\n");
    OP_CHECK_IF(
        expandedRowIdxShape->GetStorageShape().GetDimNum() != 1,
        OP_LOGE(context_->GetNodeName(), "dim num of expanded_row_idx should be 1."),
        return ge::GRAPH_FAILED);
    row = expandedRowIdxShape->GetStorageShape().GetDim(0) / k;
    OP_CHECK_IF(
        row <= 0, OP_LOGE(context_->GetNodeName(), "row[%ld] must be greater than 0.", row),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeFinalizeRoutingV2Regbase::CheckBiasShape(const gert::StorageShape* biasShape)
{
    OP_LOGI(context_->GetNodeName(), "MoeFinalizeRoutingV2, CheckBiasShape");
    printf("MoeFinalizeRoutingV2, CheckBiasShape\n");
    OP_CHECK_IF(
        biasShape->GetStorageShape().GetDimNum() != BIAS_DIM_NUM,
        OP_LOGE(context_->GetNodeName(), "dim num of bias should be 2."),
        return ge::GRAPH_FAILED);
    e = biasShape->GetStorageShape().GetDim(0);
    if (dropPadMode == DROP_PAD_ROW || dropPadMode == DROP_PAD_COL) {
        OP_CHECK_IF(
            dim0OfExpandedX != e,
            OP_LOGE(
                context_->GetNodeName(), "dim 0 of expanded_x should be equal to dim 0 of bias in droppad mode."),
            return ge::GRAPH_FAILED);
    }
    OP_CHECK_IF(
        e < k,
        OP_LOGE(
            context_->GetNodeName(), "e[%ld] must be greater than or equal to k[%ld].", e, k),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeFinalizeRoutingV2Regbase::CheckShapeAndDtypeIsValid()
{
    OP_LOGI(context_->GetNodeName(), "MoeFinalizeRoutingV2, CheckShapeAndDtypeIsValid");
    printf("MoeFinalizeRoutingV2, CheckShapeAndDtypeIsValid\n");
    gert::Shape rowIdxShape = {row * k};
    gert::Shape bsh = {row, h};

    auto expandedRowIdxDesc = context_->GetInputDesc(EXPANDED_ROW_IDX_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, expandedRowIdxDesc);
    auto expandedRowIdxShape = context_->GetInputShape(EXPANDED_ROW_IDX_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, expandedRowIdxShape);
    OP_CHECK_IF(
        expandedRowIdxDesc->GetDataType() != ge::DataType::DT_INT32,
        OP_LOGE(context_->GetNodeName(), "dtype of expanded_row_idx must be int32."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        expandedRowIdxShape->GetStorageShape() != rowIdxShape,
        OP_LOGE(context_->GetNodeName(), "shape of expanded_row_idx must be (bs,k)."),
        return ge::GRAPH_FAILED);

    auto x1Desc = context_->GetOptionalInputDesc(X1_IDX);
    if (x1Desc) {
        OP_CHECK_IF(
            x1Desc->GetDataType() != dtype,
            OP_LOGE(context_->GetNodeName(), "dtype of x1 is invalid."),
            return ge::GRAPH_FAILED);
    }
    auto x1Shape = context_->GetOptionalInputShape(X1_IDX);
    if (x1Shape) {
        OP_CHECK_IF(
            x1Shape->GetStorageShape() != bsh,
            OP_LOGE(context_->GetNodeName(), "shape of x1 must be (bs,h)."),
            return ge::GRAPH_FAILED);
    }

    auto x2Desc = context_->GetOptionalInputDesc(X2_IDX);
    if (x2Desc) {
        OP_CHECK_IF(
            x2Desc->GetDataType() != dtype,
            OP_LOGE(context_->GetNodeName(), "dtype of x2 is invalid."),
            return ge::GRAPH_FAILED);
    }

    OP_CHECK_IF(
            CheckPartShapeAndDtypeIsValid() != ge::GRAPH_SUCCESS,
            OP_LOGE(context_->GetNodeName(), "CheckPartShapeAndDtypeIsValid failed."),
            return ge::GRAPH_FAILED);
    
    OP_CHECK_IF(
            FinalCheckShapeAndDtypeIsValid() != ge::GRAPH_SUCCESS,
            OP_LOGE(context_->GetNodeName(), "FinalCheckShapeAndDtypeIsValid failed."),
            return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeFinalizeRoutingV2Regbase::CheckPartShapeAndDtypeIsValid()
{
    OP_LOGI(context_->GetNodeName(), "MoeFinalizeRoutingV2, CheckPartShapeAndDtypeIsValid");
    printf("MoeFinalizeRoutingV2, CheckPartShapeAndDtypeIsValid\n");
    gert::Shape bsh = {row, h};

    auto x2Shape = context_->GetOptionalInputShape(X2_IDX);
    if (x2Shape) {
        OP_CHECK_IF(
            x2Shape->GetStorageShape() != bsh,
            OP_LOGE(context_->GetNodeName(), "shape of x2 must be (bs,h)."),
            return ge::GRAPH_FAILED);
    }

    auto biasDesc = context_->GetOptionalInputDesc(BIAS_IDX);
    if (biasDesc) {
        OP_CHECK_IF(
            biasDesc->GetDataType() != dtype,
            OP_LOGE(context_->GetNodeName(), "dtype of bias is invalid."),
            return ge::GRAPH_FAILED);
    }
    auto biasShape = context_->GetOptionalInputShape(BIAS_IDX);
    if (biasShape) {
        OP_CHECK_IF(
            CheckBiasShape(biasShape) != ge::GRAPH_SUCCESS,
            OP_LOGE(context_->GetNodeName(), "failed to get e."), return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            biasShape->GetStorageShape().GetDim(1) != h,
            OP_LOGE(context_->GetNodeName(), "dim 1 of of bias should be h."),
            return ge::GRAPH_FAILED);
    }

    auto scalesDesc = context_->GetOptionalInputDesc(SCALES_IDX);
    scaleDtypeSize = dtypeSize;
    if (scalesDesc) {
        auto scaleDtype = scalesDesc->GetDataType();
        OP_CHECK_IF(
            scaleDtype != ge::DataType::DT_FLOAT && scaleDtype != ge::DataType::DT_FLOAT16 &&
                scaleDtype != ge::DataType::DT_BF16,
            OP_LOGE(
                context_->GetNodeName(), "scale data type only supports float32,half,bf16."),
            return ge::GRAPH_FAILED);
        scaleDtypeSize = (scaleDtype == ge::DataType::DT_FLOAT) ? sizeof(float) : sizeof(short);
        if (scaleDtype == ge::DataType::DT_FLOAT16) {
            scaleDtypeKey = FLOAT16_TILING_KEY;
        } else if (scaleDtype == ge::DataType::DT_BF16) {
            scaleDtypeKey = BFLOAT16_TILING_KEY;
        }
    }

    // const_expert_alpha等的shape信息要与const_expert_range_num一致
    constExpertRangeNum = constantExpertEnd - constantExpertStart;
    auto constExpertAlpha1Desc = context_->GetOptionalInputDesc(CONST_EXPERT_ALPHA1_IDX);
    if (constExpertAlpha1Desc) {
        OP_CHECK_IF(
            constExpertAlpha1Desc->GetDataType() != dtype,
            OP_LOGE(context_->GetNodeName(), "dtype of const expert alpha1 is invalid."),
            return ge::GRAPH_FAILED);
    }
    auto constExpertAlpha1Shape = context_->GetOptionalInputShape(CONST_EXPERT_ALPHA1_IDX);
    if (constExpertAlpha1Shape) {
        OP_CHECK_IF(
            constExpertAlpha1Shape->GetStorageShape().GetDim(1) != h,
            OP_LOGE(context_->GetNodeName(), "dim 1 of of const expert alpha1 should be h."),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            constExpertAlpha1Shape->GetStorageShape().GetDim(0) != constExpertRangeNum,
            OP_LOGE(context_->GetNodeName(), "dim 0 of of  expert alpha1 should be constExpertRangeNum."),
            return ge::GRAPH_FAILED);
    }
    
    auto constExpertAlpha2Desc = context_->GetOptionalInputDesc(CONST_EXPERT_ALPHA2_IDX);
    if (constExpertAlpha2Desc) {
        OP_CHECK_IF(
            constExpertAlpha2Desc->GetDataType() != dtype,
            OP_LOGE(context_->GetNodeName(), "dtype of const expert alpha2 is invalid."),
            return ge::GRAPH_FAILED);
    }
    auto constExpertAlpha2Shape = context_->GetOptionalInputShape(CONST_EXPERT_ALPHA2_IDX);
    if (constExpertAlpha2Shape) {
        OP_CHECK_IF(
            constExpertAlpha2Shape->GetStorageShape().GetDim(1) != h,
            OP_LOGE(context_->GetNodeName(), "dim 1 of of const expert alpha2 should be h."),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            constExpertAlpha2Shape->GetStorageShape().GetDim(0) != constExpertRangeNum,
            OP_LOGE(context_->GetNodeName(), "dim 0 of of  expert alpha2 should be constExpertRangeNum."),
            return ge::GRAPH_FAILED);
    }

    auto vDesc = context_->GetOptionalInputDesc(CONST_EXPERT_V_IDX);
    if (vDesc) {
        OP_CHECK_IF(
            vDesc->GetDataType() != dtype,
            OP_LOGE(context_->GetNodeName(), "dtype of v is invalid."),
            return ge::GRAPH_FAILED);
    }

    auto vShape = context_->GetOptionalInputShape(CONST_EXPERT_V_IDX);
    if (vShape) {
        OP_CHECK_IF(
            vShape->GetStorageShape().GetDim(1) != h,
            OP_LOGE(context_->GetNodeName(), "dim 1 of of v should be h."),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            vShape->GetStorageShape().GetDim(0) != constExpertRangeNum,
            OP_LOGE(context_->GetNodeName(), "dim 0 of of v should be constExpertRangeNum."),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeFinalizeRoutingV2Regbase::FinalCheckShapeAndDtypeIsValid()
{
    OP_LOGI(context_->GetNodeName(), "MoeFinalizeRoutingV2, FinalCheckShapeAndDtypeIsValid");
    printf("MoeFinalizeRoutingV2, FinalCheckShapeAndDtypeIsValid\n");
    gert::Shape bsk = {row, k};
    gert::Shape bsh = {row, h};

    auto scalesShape = context_->GetOptionalInputShape(SCALES_IDX);
    if (scalesShape) {
        OP_CHECK_IF(
            scalesShape->GetStorageShape() != bsk,
            OP_LOGE(context_->GetNodeName(), "shape of scales must be (bs,k)."),
            return ge::GRAPH_FAILED);
    }

    auto expertIdxDesc = context_->GetOptionalInputDesc(EXPERTIDX_IDX);
    if (expertIdxDesc) {
        OP_CHECK_IF(
            expertIdxDesc->GetDataType() != ge::DataType::DT_INT32,
            OP_LOGE(context_->GetNodeName(), "dtype of expert_idx must be int32."),
            return ge::GRAPH_FAILED);
    }
    auto expertIdxsShape = context_->GetOptionalInputShape(EXPERTIDX_IDX);
    if (expertIdxsShape) {
        OP_CHECK_IF(
            expertIdxsShape->GetStorageShape() != bsk,
            OP_LOGE(context_->GetNodeName(), "shape of expert_idx must be (bs,k)."),
            return ge::GRAPH_FAILED);
    }

    // 判断x的shape信息要与（ROW_NUM, H）一致
    auto xDesc = context_->GetOptionalInputDesc(X_IDX);
    if (xDesc) {
        auto xShape = context_->GetOptionalInputShape(X_IDX);
        OP_CHECK_IF(
            xDesc->GetDataType() != dtype,
            OP_LOGE(context_->GetNodeName(), "dtype of x is invalid."),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            xShape->GetStorageShape() != bsh,
            OP_LOGE(context_->GetNodeName(), "shape of x must be (bs,h)."),
            return ge::GRAPH_FAILED);
    }


    auto yDesc = context_->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, yDesc);
    auto yShape = context_->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, yShape);
    OP_CHECK_IF(
        yDesc->GetDataType() != dtype,
        OP_LOGE(context_->GetNodeName(), "dtype of y is invalid."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        yShape->GetStorageShape() != bsh,
        OP_LOGE(context_->GetNodeName(), "shape of y must be (bs,h)."),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeFinalizeRoutingV2Regbase::DoGetShapeAttrsInfo()
{
    OP_LOGI(context_->GetNodeName(), "MoeFinalizeRoutingV2, DoGetShapeAttrsInfo");
    printf("MoeFinalizeRoutingV2, DoGetShapeAttrsInfo\n");
    // attr的实现
    auto attrsPtr = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrsPtr);
    auto dropPadModePtr = attrsPtr->GetAttrPointer<int64_t>(ATTR_DROP_PAD_MODE);
    OP_CHECK_NULL_WITH_CONTEXT(context_, dropPadModePtr);
    dropPadMode = *dropPadModePtr;
    OP_CHECK_IF(
        dropPadMode != DROP_LESS_ROW && dropPadMode != DROP_LESS_COL && dropPadMode != DROP_PAD_COL &&
            dropPadMode != DROP_PAD_ROW,
        OP_LOGE(context_->GetNodeName(), "drop pad mode only supports 0 or 1 or 2."),
        return ge::GRAPH_FAILED);

    const auto *zeroPtr = attrsPtr->GetAttrPointer<gert::ContinuousVector>(ATTR_ZERO_EXPERT_RANGE);
    if (zeroPtr != nullptr && zeroPtr->GetSize() == 2) {
        const int64_t *zeroList = reinterpret_cast<const int64_t *>(zeroPtr->GetData());
        zeroExpertStart = zeroList[0];
        zeroExpertEnd = zeroList[1];
        OP_LOGD(context_, "Extracted input attrs zeroExpertStart = %ld, zeroExpertEnd = %ld.",
            zeroExpertStart, zeroExpertEnd);
    }

    const auto *copyPtr = attrsPtr->GetAttrPointer<gert::ContinuousVector>(ATTR_COPY_EXPERT_RANGE);
    if (copyPtr != nullptr && copyPtr->GetSize() == 2) {
        const int64_t *copyList = reinterpret_cast<const int64_t *>(copyPtr->GetData());
        copyExpertStart = copyList[0];
        copyExpertEnd = copyList[1];
        OP_LOGD(context_, "Extracted input attrs copyExpertStart = %ld, copyExpertEnd = %ld.",
            copyExpertStart, copyExpertEnd);
    }

    const auto *constantPtr = attrsPtr->GetAttrPointer<gert::ContinuousVector>(ATTR_CONSTANT_EXPERT_RANGE);
    if (constantPtr != nullptr && constantPtr->GetSize() == 2) {
        const int64_t *constList = reinterpret_cast<const int64_t *>(constantPtr->GetData());
        constantExpertStart = constList[0];
        constantExpertEnd = constList[1];
        OP_LOGD(context_, "Extracted input attrs constantExpertStart = %ld, constantExpertEnd = %ld.",
            constantExpertStart, constantExpertEnd);
    }

    // 输入数据
    auto expandedXDesc = context_->GetInputDesc(EXPANDED_X_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, expandedXDesc);
    dtype = expandedXDesc->GetDataType();
    OP_CHECK_IF(
        dtype != ge::DataType::DT_FLOAT && dtype != ge::DataType::DT_FLOAT16 && dtype != ge::DataType::DT_BF16,
        OP_LOGE(context_->GetNodeName(), "data type only supports float32,half,bf16."),
        return ge::GRAPH_FAILED);
    dtypeSize = (dtype == ge::DataType::DT_FLOAT) ? sizeof(float) : sizeof(short);
    auto expandedXShape = context_->GetInputShape(EXPANDED_X_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, expandedXShape);
    OP_CHECK_IF(
        GetECH(expandedXShape) != ge::GRAPH_SUCCESS,
        OP_LOGE(context_->GetNodeName(), "failed to get H."), return ge::GRAPH_FAILED);

    auto scalesShape = context_->GetOptionalInputShape(SCALES_IDX);
    OP_CHECK_IF(
        GetK(scalesShape) != ge::GRAPH_SUCCESS,
        OP_LOGE(context_->GetNodeName(), "failed to get K."), return ge::GRAPH_FAILED);

    auto expandedRowIdxShape = context_->GetInputShape(EXPANDED_ROW_IDX_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, expandedRowIdxShape);
    OP_CHECK_IF(
        GetRow(expandedRowIdxShape) != ge::GRAPH_SUCCESS,
        OP_LOGE(context_->GetNodeName(), "failed to get row."), return ge::GRAPH_FAILED);

    auto x1Desc = context_->GetOptionalInputDesc(X1_IDX);
    hasX1_ = x1Desc != nullptr;
    auto x2Desc = context_->GetOptionalInputDesc(X2_IDX);
    hasX2_ = x2Desc != nullptr;
    auto biasDesc = context_->GetOptionalInputDesc(BIAS_IDX);
    hasBias_ = biasDesc != nullptr;
    OP_CHECK_IF(
        hasX2_ && !hasX1_, OP_LOGE(context_->GetNodeName(), "has x2 but x1 not exist."),
        return ge::GRAPH_FAILED);
    
    // 新增x， shape与bias一致
    auto xDesc = context_->GetOptionalInputDesc(X_IDX);
    hasX_ = xDesc != nullptr;
    auto constExpertAlpha1Desc = context_->GetOptionalInputDesc(CONST_EXPERT_ALPHA1_IDX);
    auto constExpertAlpha2Desc = context_->GetOptionalInputDesc(CONST_EXPERT_ALPHA2_IDX);
    auto constExpertVDesc = context_->GetOptionalInputDesc(CONST_EXPERT_V_IDX);
    hasConstantExpert_ =  constExpertVDesc != nullptr && constExpertAlpha1Desc != nullptr && constExpertAlpha2Desc != nullptr;
    OP_CHECK_IF(
        CheckShapeAndDtypeIsValid() != ge::GRAPH_SUCCESS,
        OP_LOGE(context_->GetNodeName(), "check shapes and dtype are invalid."),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

int64_t MoeFinalizeRoutingV2Regbase::RowsHSize(int64_t rowFactor, bool scalesInUb)
{
    OP_LOGI(context_->GetNodeName(), "MoeFinalizeRoutingV2, RowsHSize");
    printf("MoeFinalizeRoutingV2, RowsHSize\n");
    return (static_cast<int64_t>(hasX1_) + static_cast<int64_t>(hasX2_) + static_cast<int64_t>(hasX_)) *
               Ops::Base::CeilDiv(static_cast<uint64_t>(rowFactor * h * dtypeSize), blockSize_) * blockSize_ +
           (hasConstantExpert_ ? 1 : 0) * constExpertRangeNum * hAligned * dtypeSize * 3 +
           (scalesInUb && hasScales_ ? 1 : 0) *
               Ops::Base::CeilDiv(static_cast<uint64_t>(rowFactor * k * scaleDtypeSize), blockSize_) * blockSize_ +
           /* y */ Ops::Base::CeilDiv(static_cast<uint64_t>(rowFactor * h * sizeof(float)), blockSize_) * blockSize_;
}

int64_t MoeFinalizeRoutingV2Regbase::RowsHSizeForKHFullLoad(int64_t rowFactor, bool scalesInUb)
{
    OP_LOGI(context_->GetNodeName(), "MoeFinalizeRoutingV2, RowsHSizeForKHFullLoad");
    printf("MoeFinalizeRoutingV2, RowsHSizeForKHFullLoad\n");
    return (static_cast<int64_t>(hasX1_) + static_cast<int64_t>(hasX2_) + static_cast<int64_t>(hasX_)) *
               Ops::Base::CeilDiv(static_cast<uint64_t>(rowFactor * h * dtypeSize), blockSize_) * blockSize_ +
           (scalesInUb && hasScales_ ? 1 : 0) *
               Ops::Base::CeilDiv(static_cast<uint64_t>(rowFactor * k * scaleDtypeSize), blockSize_) * blockSize_ +
           (hasBias_ ? 1 : 0) * rowFactor * k * hAligned * dtypeSize + 
           (hasConstantExpert_ ? 1 : 0) * constExpertRangeNum * hAligned * dtypeSize * 3 +
               rowFactor * k * hAligned * dtypeSize +
               Ops::Base::CeilDiv(static_cast<uint64_t>(rowFactor * k * sizeof(int32_t)), blockSize_) * blockSize_ +
               Ops::Base::CeilDiv(static_cast<uint64_t>(rowFactor * k * sizeof(int32_t)), blockSize_) * blockSize_ +
               Ops::Base::CeilDiv(static_cast<uint64_t>(rowFactor * h * sizeof(float)), blockSize_) * blockSize_;
}

int64_t MoeFinalizeRoutingV2Regbase::CalcRowFactorForKHFullLoad(int64_t ubSizeRemained, bool scalesInUb)
{
    OP_LOGI(context_->GetNodeName(), "MoeFinalizeRoutingV2, CalcRowFactorForKHFullLoad");
    printf("MoeFinalizeRoutingV2, CalcRowFactorForKHFullLoad\n");
    int64_t rowFactor = 1;
    int64_t factor = 1;
    while (RowsHSizeForKHFullLoad(rowFactor, scalesInUb) <= ubSizeRemained) {
        factor *= TWO;
        rowFactor *= factor;
    }
    int64_t upper = rowFactor;
    int64_t lower = rowFactor / factor;
    while (upper - lower > 1) {
        int64_t mid = (lower + upper) / TWO;
        int64_t size = RowsHSizeForKHFullLoad(mid, scalesInUb);
        if (size <= ubSizeRemained) {
            lower = mid;
        } else {
            upper = mid;
        }
    }
    rowFactor = lower;
    return rowFactor;
}

int64_t MoeFinalizeRoutingV2Regbase::CalcRowFactor(int64_t ubSizeRemained, bool scalesInUb)
{
    OP_LOGI(context_->GetNodeName(), "MoeFinalizeRoutingV2, CalcRowFactor");
    printf("MoeFinalizeRoutingV2, CalcRowFactor\n");
    int64_t rowFactor = 1;
    int64_t factor = 1;
    while (RowsHSize(rowFactor, scalesInUb) <= ubSizeRemained) {
        factor *= TWO;
        rowFactor *= factor;
    }
    int64_t upper = rowFactor;
    int64_t lower = rowFactor / factor;
    while (upper - lower > 1) {
        int64_t mid = (lower + upper) / TWO;
        int64_t size = RowsHSize(mid, scalesInUb);
        if (size <= ubSizeRemained) {
            lower = mid;
        } else {
            upper = mid;
        }
    }
    rowFactor = lower;
    return rowFactor;
}

void MoeFinalizeRoutingV2Regbase::SetFullLoadTilingData(
    int64_t rowOfFormerBlock, int64_t rowOfTailBlock, int64_t rowFactor)
{
    OP_LOGI(context_->GetNodeName(), "MoeFinalizeRoutingV2, SetFullLoadTilingData");
    printf("MoeFinalizeRoutingV2, SetFullLoadTilingData\n");
    int64_t rowLoopOfFormerBlock = Ops::Base::CeilDiv(rowOfFormerBlock, rowFactor);
    rowFactor = Ops::Base::CeilDiv(rowOfFormerBlock, rowLoopOfFormerBlock);
    int64_t rowLoopOfTailBlock = Ops::Base::CeilDiv(rowOfTailBlock, rowFactor);
    int64_t tailRowFactorOfFormerBlock = rowOfFormerBlock - (rowLoopOfFormerBlock - 1) * rowFactor;
    int64_t tailRowFactorOfTailBlock = rowOfTailBlock - (rowLoopOfTailBlock - 1) * rowFactor;
    tilingData->rowLoopOfFormerBlock = rowLoopOfFormerBlock;
    tilingData->rowLoopOfTailBlock = rowLoopOfTailBlock;
    tilingData->rowFactor = rowFactor;
    tilingData->constExpertRangeFactor = std::min(constExpertRangeNum, rowFactor);
    tilingData->tailRowFactorOfFormerBlock = tailRowFactorOfFormerBlock;
    tilingData->tailRowFactorOfTailBlock = tailRowFactorOfTailBlock;
    tilingData->hLoop = 1;
    tilingData->hFactor = h;
    tilingData->tailHFactor = h;
    tilingData->kLoop = k;
    tilingData->kFactor = 1;
    tilingData->tailKFactor = 1;
    tilingData->activeNum = dim0OfExpandedX;

    tilingData->zeroExpertStart = zeroExpertStart;
    tilingData->zeroExpertEnd = zeroExpertEnd;
    tilingData->copyExpertStart = copyExpertStart;
    tilingData->copyExpertEnd = copyExpertEnd;
    tilingData->constantExpertStart = constantExpertStart;
    tilingData->constantExpertEnd = constantExpertEnd;
    tilingData->constExpertRangeNum = constExpertRangeNum;
}

// 切分的地方需要修改
ge::graphStatus MoeFinalizeRoutingV2Regbase::DoOpTilingRowKHFullLoad(int64_t rowOfFormerBlock, int64_t rowOfTailBlock)
{
    OP_LOGI(context_->GetNodeName(), "MoeFinalizeRoutingV2, DoOpTilingRowKHFullLoad");
    printf("MoeFinalizeRoutingV2, DoOpTilingRowKHFullLoad\n");
    int64_t expandedXAlignedByte;
    if (dropPadMode == DROP_LESS_COL || dropPadMode == DROP_LESS_ROW) {
        expandedXAlignedByte = Ops::Base::CeilDiv(static_cast<uint64_t>(row * k * h * dtypeSize), blockSize_) * blockSize_;
    } else {
        expandedXAlignedByte = Ops::Base::CeilDiv(static_cast<uint64_t>(e * c * h * dtypeSize), blockSize_) * blockSize_;
    }
    int64_t eHAlignedByte = Ops::Base::CeilDiv(static_cast<uint64_t>(e * h * dtypeSize), blockSize_) * blockSize_;
    int64_t hasBiasvalue = hasBias_ ? eHAlignedByte : 0;
    int64_t ubSizeRemained = (ubSize_ - expandedXAlignedByte - hasBiasvalue) / DOUBLE_BUFFER;
    int64_t rowFactor = CalcRowFactor(ubSizeRemained, true);
    SetFullLoadTilingData(rowOfFormerBlock, rowOfTailBlock, rowFactor);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeFinalizeRoutingV2Regbase::DoOpTilingKHFullLoad(int64_t rowOfFormerBlock, int64_t rowOfTailBlock)
{
    OP_LOGI(context_->GetNodeName(), "MoeFinalizeRoutingV2, DoOpTilingKHFullLoad");
    printf("MoeFinalizeRoutingV2, DoOpTilingKHFullLoad\n");
    int64_t rowFactor = 1;
    if (k == 1) {
        int64_t ubSizeRemained = ubSize_ / DOUBLE_BUFFER;
        rowFactor = CalcRowFactorForKHFullLoad(ubSizeRemained, true);
    } else {
        int64_t kHAlignedByte = k * hAligned * dtypeSize;	
        int64_t hasBiasvalue = hasBias_ ? kHAlignedByte : 0;
        int64_t ubSizeRemained = ubSize_ / DOUBLE_BUFFER - kHAlignedByte - hasBiasvalue;
        rowFactor = CalcRowFactor(ubSizeRemained, true);
    }
    SetFullLoadTilingData(rowOfFormerBlock, rowOfTailBlock, rowFactor);	
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeFinalizeRoutingV2Regbase::DoOpTilingHFullLoad(int64_t rowOfFormerBlock, int64_t rowOfTailBlock)
{
    OP_LOGI(context_->GetNodeName(), "MoeFinalizeRoutingV2, DoOpTilingHFullLoad");
    printf("MoeFinalizeRoutingV2, DoOpTilingHFullLoad\n");
    int64_t expandedXAndBiasSize = (1 /* expanded_x */ + static_cast<int64_t>(hasBias_)) * hAligned * dtypeSize;
    int64_t kFactor = ubSize_ / DOUBLE_BUFFER / expandedXAndBiasSize;
    int64_t upper = kFactor;
    int64_t lower = 1;
    int64_t blockSize = static_cast<int64_t>(blockSize_);
    while (upper - lower > 1) {
        int64_t mid = (lower + upper) / TWO;
        int64_t useUbSize =
            mid * expandedXAndBiasSize + Ops::Base::CeilDiv(mid * dtypeSize, blockSize) * blockSize + RowsHSize(1, false);
        if (useUbSize > ubSize_ / DOUBLE_BUFFER) {
            upper = mid;
        } else {
            lower = mid;
        }
    }
    kFactor = lower;
    int64_t kLoop = Ops::Base::CeilDiv(k, kFactor);
    kFactor = Ops::Base::CeilDiv(k, kLoop);
    int64_t tailKFactor = k - (kLoop - 1) * kFactor;

    int64_t ubSizeRemained = ubSize_ / DOUBLE_BUFFER - kFactor * expandedXAndBiasSize -
                             Ops::Base::CeilDiv(kFactor * dtypeSize, blockSize) * blockSize;
    int64_t rowFactor = CalcRowFactor(ubSizeRemained, false);
    SetFullLoadTilingData(rowOfFormerBlock, rowOfTailBlock, rowFactor);
    tilingData->kLoop = kLoop;
    tilingData->kFactor = kFactor;
    tilingData->tailKFactor = tailKFactor;
    OP_LOGI(context_->GetNodeName(), "MoeFinalizeRoutingV2, end DoOpTilingHFullLoad");
    printf("MoeFinalizeRoutingV2, end DoOpTilingHFullLoad\n");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeFinalizeRoutingV2Regbase::DoOpTilingSplitH(int64_t rowOfFormerBlock, int64_t rowOfTailBlock)
{
    OP_LOGI(context_->GetNodeName(), "MoeFinalizeRoutingV2, DoOpTilingSplitH");
    printf("MoeFinalizeRoutingV2, DoOpTilingSplitH\n");
    int64_t actualInputNum = INPUT_BUFFER_NUM - static_cast<int64_t>(!hasX1_) - static_cast<int64_t>(!hasX2_) -
                             static_cast<int64_t>(!hasBias_) - static_cast<int64_t>(!hasX_) - static_cast<int64_t>(!hasConstantExpert_) * CONST_EXPERT_BUFFER_NUM;
    int64_t totalBufferNum = actualInputNum + OUTPUT_BUFFER_NUM + (dtype != ge::DataType::DT_FLOAT ? 1 : 0);
    int64_t hFactor = ubSize_ / DOUBLE_BUFFER / dtypeSize / totalBufferNum;
    int64_t hLoop = Ops::Base::CeilDiv(h, hFactor);
    hFactor = Ops::Base::CeilDiv(h, hLoop);
    int64_t tailHFactor = h - (hLoop - 1) * hFactor;
    tilingData->rowLoopOfFormerBlock = rowOfFormerBlock;
    tilingData->rowLoopOfTailBlock = rowOfTailBlock;
    tilingData->rowFactor = 1;
    tilingData->tailRowFactorOfFormerBlock = rowOfFormerBlock;
    tilingData->tailRowFactorOfTailBlock = rowOfTailBlock;
    tilingData->hLoop = hLoop;
    tilingData->hFactor = hFactor;
    tilingData->tailHFactor = tailHFactor;
    tilingData->activeNum = dim0OfExpandedX;
        
    OP_LOGI(context_->GetNodeName(), "MoeFinalizeRoutingV2, end DoOpTilingSplitH");
    printf("MoeFinalizeRoutingV2, end DoOpTilingSplitH\n");
    return ge::GRAPH_SUCCESS;
}

void MoeFinalizeRoutingV2Regbase::PrintTilingData()
{

    OP_LOGI(
        context_->GetNodeName(),
        "MoeFinalizeRoutingV2 tiling data: numBlocks[%ld] row[%ld] e[%ld] c[%ld] h[%ld] hAligned[%ld] k[%ld]"
        "rowOfFormerBlock[%ld] rowOfTailBlock[%ld] rowLoopOfFormerBlock[%ld] rowLoopOfTailBlock[%ld] "
        "rowFactor[%ld] tailRowFactorOfFormerBlock[%ld] tailRowFactorOfTailBlock[%ld] "
        "hLoop[%ld] hFactor[%ld] tailHFactor[%ld] zeroExpertStart[%ld] zeroExpertEnd[%ld] copyExpertStart[%ld] "
        "copyExpertEnd[%ld] constantExpertStart[%ld] constantExpertEnd[%ld] constExpertRangeNum[%ld]",
        usedCoreNum_, tilingData->row, tilingData->e, tilingData->c, tilingData->h, tilingData->hAligned, tilingData->k,
        tilingData->rowOfFormerBlock, tilingData->rowOfTailBlock, tilingData->rowLoopOfFormerBlock,
        tilingData->rowLoopOfTailBlock, tilingData->rowFactor, tilingData->tailRowFactorOfFormerBlock,
        tilingData->tailRowFactorOfTailBlock, tilingData->hLoop, tilingData->hFactor, tilingData->tailHFactor,
        tilingData->zeroExpertStart, tilingData->zeroExpertEnd, tilingData->copyExpertStart, tilingData->copyExpertEnd,
        tilingData->constantExpertStart, tilingData->constantExpertEnd, tilingData->constExpertRangeNum);
}

ge::graphStatus MoeFinalizeRoutingV2Regbase::CalcOpTiling()
{
    OP_LOGI(context_->GetNodeName(), "MoeFinalizeRoutingV2, CalcOpTiling");
    printf("MoeFinalizeRoutingV2, CalcOpTiling\n");
    int64_t rowPerCore = Ops::Base::CeilDiv(row, static_cast<int64_t>(coreNum_));
    usedCoreNum_ = std::min(Ops::Base::CeilDiv(row, rowPerCore), static_cast<int64_t>(coreNum_));
    int64_t rowOfTailBlock = row - (usedCoreNum_ - 1) * rowPerCore;

    tilingData = context_->GetTilingData<MoeFinalizeRoutingV2RegbaseTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context_, tilingData);

    tilingData->row = row;
    tilingData->e = e;
    tilingData->c = c;
    tilingData->h = h;
    tilingData->hAligned = hAligned;
    tilingData->k = k;
    tilingData->rowOfFormerBlock = rowPerCore;
    tilingData->rowOfTailBlock = rowOfTailBlock;

    tilingData->zeroExpertStart = zeroExpertStart;
    tilingData->zeroExpertEnd = zeroExpertEnd;
    tilingData->copyExpertStart = copyExpertStart;
    tilingData->copyExpertEnd = copyExpertEnd;
    tilingData->constantExpertStart = constantExpertStart;
    tilingData->constantExpertEnd = constantExpertEnd;

    // 切分
    ge::graphStatus ret;
    if (rowKHFullLoad_) {
        ret = DoOpTilingRowKHFullLoad(rowPerCore, rowOfTailBlock);
    } else if (kHFullLoad_) {
        ret = DoOpTilingKHFullLoad(rowPerCore, rowOfTailBlock);
    } else if (hFullLoad_) {
        ret = DoOpTilingHFullLoad(rowPerCore, rowOfTailBlock);
    } else {
        ret = DoOpTilingSplitH(rowPerCore, rowOfTailBlock);
    }

    OP_CHECK_IF(
        ret != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "failed to do tiling"),
        return ge::GRAPH_FAILED);
    PrintTilingData();
    OP_LOGI(context_->GetNodeName(), "MoeFinalizeRoutingV2, end CalcOpTiling");
    printf("MoeFinalizeRoutingV2, end CalcOpTiling\n");
    return ge::GRAPH_SUCCESS;
}
// 校验 Row-K-H 维度数据是否能全量加载到 UB
bool MoeFinalizeRoutingV2Regbase::IsRowKHFullLoad()
{
    OP_LOGI(context_->GetNodeName(), "MoeFinalizeRoutingV2, IsRowKHFullLoad");
    printf("MoeFinalizeRoutingV2, IsRowKHFullLoad\n");
    int64_t expandedXAlignedByte;
    // 1. 计算expandedX数据的对齐字节数（区分不同的drop/pad模式）
    if (dropPadMode == DROP_LESS_COL || dropPadMode == DROP_LESS_ROW) {
        expandedXAlignedByte = Ops::Base::CeilDiv(static_cast<uint64_t>(row * k * h * dtypeSize), blockSize_) * blockSize_;
    } else {
        expandedXAlignedByte = Ops::Base::CeilDiv(static_cast<uint64_t>(e * c * h * dtypeSize), blockSize_) * blockSize_;
    }
    // 2. 计算e*h维度数据的对齐字节数（专家数-高度）
    int64_t eHAlignedByte = Ops::Base::CeilDiv(static_cast<uint64_t>(e * h * dtypeSize), blockSize_) * blockSize_;
    // 3. 计算scales（缩放系数）数据的对齐字节数（专家数-缩放系数类型长度）
    int64_t scalesAlignedByte = Ops::Base::CeilDiv(static_cast<uint64_t>(k * scaleDtypeSize), blockSize_) * blockSize_;
    // 4. 计算h维度数据的对齐字节数（高度），并记录对齐后的h维度长度
    int64_t hAlignedByte = Ops::Base::CeilDiv(static_cast<uint64_t>(h * dtypeSize), blockSize_) * blockSize_;
    hAligned = hAlignedByte / dtypeSize;
    int64_t hAligned32Byte = Ops::Base::CeilDiv(static_cast<uint64_t>(h * sizeof(float)), blockSize_) * blockSize_;
    // 6. 条件性计算：有bias（偏置）则加eHAlignedByte，否则为0
    int64_t hasBiasvalue = hasBias_ ? eHAlignedByte : 0;
    int64_t hasScalevalue = hasScales_ ? scalesAlignedByte : 0;
    // 8. 计算所有数据的总字节数（核心：DOUBLE_BUFFER是双缓冲机制，提升流水线效率）
    int64_t totalSize =
        expandedXAlignedByte + hasBiasvalue +
        DOUBLE_BUFFER * (hasScalevalue + (static_cast<int64_t>(hasX1_) + static_cast<int64_t>(hasX2_) +
        static_cast<int64_t>(hasX_) + static_cast<int64_t>(hasConstantExpert_) * CONST_EXPERT_BUFFER_NUM) * hAlignedByte +
                         hAligned32Byte * OUTPUT_BUFFER_NUM);
                        
    OP_LOGI(context_->GetNodeName(), "MoeFinalizeRoutingV2, IsRowKHFullLoad, totalSize[%ld], ubSize_[%ld]", totalSize, ubSize_);
    printf("MoeFinalizeRoutingV2, IsRowKHFullLoad, totalSize[%ld], ubSize_[%ld]\n", totalSize, ubSize_);
    return totalSize <= ubSize_;
}

// 校验 K-H 维度数据是否能全量加载到 UB
// 聚焦 K-H（专家数 - 高度）维度，而非 Row-K-H 新增了行索引（expandedRowIdx） 和专家索引（expertIdx） 的字节数计算，且仅在 k=1（单专家）时生效；
// 适用于不需要考虑 Row 维度、仅处理 K-H 维度的 MoE 子任务（如专家内数据处理）
bool MoeFinalizeRoutingV2Regbase::IsKHFullLoad()
{
    OP_LOGI(context_->GetNodeName(), "MoeFinalizeRoutingV2, IsKHFullLoad");
    printf("MoeFinalizeRoutingV2, IsKHFullLoad\n");
    int64_t scalesAlignedByte = Ops::Base::CeilDiv(static_cast<uint64_t>(k * scaleDtypeSize), blockSize_) * blockSize_;
    int64_t hAlignedByte = Ops::Base::CeilDiv(static_cast<uint64_t>(h * dtypeSize), blockSize_) * blockSize_;
    hAligned = hAlignedByte / dtypeSize;
    int64_t kHAlignedByte = k * hAligned * dtypeSize;
    int64_t hAligned32Byte = Ops::Base::CeilDiv(static_cast<uint64_t>(h * sizeof(float)), blockSize_) * blockSize_;
    int64_t hasBiasvalue = hasBias_ ? kHAlignedByte : 0;
    int64_t hasScalevalue = hasScales_ ? scalesAlignedByte : 0;
    int64_t expandedRowIdxAlignedByte = 
        Ops::Base::CeilDiv(static_cast<uint64_t>(k * sizeof(int32_t)), blockSize_) * blockSize_;
    int64_t hasExpandedRowIdxValue = k == 1 ? expandedRowIdxAlignedByte : 0;
    int64_t expertIdxAlignedByte = 
        Ops::Base::CeilDiv(static_cast<uint64_t>(k * sizeof(int32_t)), blockSize_) * blockSize_;
    int64_t hasExpertIdxValue = (hasBias_ && k == 1) ? expertIdxAlignedByte : 0;
    int64_t totalSize = DOUBLE_BUFFER * (kHAlignedByte + hasBiasvalue + hasScalevalue + hasExpandedRowIdxValue + hasExpertIdxValue +
                                        (static_cast<int64_t>(hasX1_) + static_cast<int64_t>(hasX2_) + static_cast<int64_t>(hasX_) + 
                                        static_cast<int64_t>(hasConstantExpert_) * 3) * hAlignedByte +
                                        hAligned32Byte * OUTPUT_BUFFER_NUM);
    
    OP_LOGI(context_->GetNodeName(), "MoeFinalizeRoutingV2, IsKHFullLoad, totalSize[%ld], ubSize_[%ld]", totalSize, ubSize_);
    printf("MoeFinalizeRoutingV2, IsKHFullLoad, totalSize[%ld], ubSize_[%ld]\n", totalSize, ubSize_);
    return totalSize <= ubSize_;
}
// 校验 H 维度数据是否能全量加载到 UB
// 适用于仅处理 H 维度的极小粒度 MoE 任务（如单维度数据转换、偏置叠加）
bool MoeFinalizeRoutingV2Regbase::IsHFullLoad()
{
    OP_LOGI(context_->GetNodeName(), "MoeFinalizeRoutingV2, IsHFullLoad");
    printf("MoeFinalizeRoutingV2, IsHFullLoad\n");
    int64_t oneKAlignedByte = static_cast<int64_t>(blockSize_);
    int64_t hAlignedByte = Ops::Base::CeilDiv(static_cast<uint64_t>(h * dtypeSize), blockSize_) * blockSize_;
    hAligned = hAlignedByte / dtypeSize;
    int64_t hAligned32Byte = Ops::Base::CeilDiv(static_cast<uint64_t>(h * sizeof(float)), blockSize_) * blockSize_;
    int64_t actualInputNum = INPUT_BUFFER_NUM - static_cast<int64_t>(!hasX1_) - static_cast<int64_t>(!hasX2_) -
                             static_cast<int64_t>(!hasBias_) - static_cast<int64_t>(!hasX_) - static_cast<int64_t>(!hasConstantExpert_) * 3;
    int64_t hasScalevalue = hasScales_ ? oneKAlignedByte : 0;
    int64_t totalSize =
        DOUBLE_BUFFER * (hAlignedByte * actualInputNum + hasScalevalue + hAligned32Byte * OUTPUT_BUFFER_NUM);

    OP_LOGI(context_->GetNodeName(), "MoeFinalizeRoutingV2, IsHFullLoad, totalSize[%ld], ubSize_[%ld]", totalSize, ubSize_);
    printf("MoeFinalizeRoutingV2, IsHFullLoad, totalSize[%ld], ubSize_[%ld]\n", totalSize, ubSize_);
    return totalSize <= ubSize_;
}
// 此处计算的时候，没有考虑常量专家的 a1 、 a2 和 v 的空间占用
bool MoeFinalizeRoutingV2Regbase::IsCapable()
{
    OP_LOGI(context_->GetNodeName(), "MoeFinalizeRoutingV2, IsCapable");
    printf("MoeFinalizeRoutingV2, IsCapable\n");
    if (!Ops::Transformer::OpTiling::IsRegbaseSocVersion(context_)) {
        return false;
    }
    // IsRowKHFullLoad（全维度）→ IsKHFullLoad（K-H 维度）→ IsHFullLoad（仅 H 维度），粒度由粗到细，适配不同 MoE 子任务；
    rowKHFullLoad_ = IsRowKHFullLoad();
    if (rowKHFullLoad_) {
        return true;
    }

    kHFullLoad_ = IsKHFullLoad();
    if (kHFullLoad_) {
        return true;
    }
    hFullLoad_ = IsHFullLoad();

    return true;
}

ge::graphStatus MoeFinalizeRoutingV2Regbase::CalcTilingKey()
{
    OP_LOGI(context_->GetNodeName(), "MoeFinalizeRoutingV2, CalcTilingKey");
    printf("MoeFinalizeRoutingV2, CalcTilingKey\n");
    if (rowKHFullLoad_) {
        tilingKey_ = FULL_LOAD_ROW_K_H_BASE_TILING_KEY;
    } else if (kHFullLoad_) {
        tilingKey_ = FULL_LOAD_K_H_BASE_TILING_KEY;
    } else if (hFullLoad_) {
        tilingKey_ = FULL_LOAD_H_BASE_TILING_KEY;
    } else {
        tilingKey_ = SPLIT_H_BASE_TILING_KEY;
    }

    if (dropPadMode == DROP_LESS_COL) {
        tilingKey_ += DROPLESS_COL_TILING_LEY;
    } else if (dropPadMode == DROP_LESS_ROW) {
        tilingKey_ += DROPLESS_ROW_TILING_LEY;
    } else if (dropPadMode == DROP_PAD_COL) {
        tilingKey_ += DROPPAD_COL_TILING_KEY;
    } else {
        tilingKey_ += DROPPAD_ROW_TILING_KEY;
    }

    tilingKey_ += scaleDtypeKey;

    OP_LOGI(context_->GetNodeName(), "MoeFinalizeRoutingV2 get tiling key: %lx", tilingKey_);
    printf("MoeFinalizeRoutingV2 get tiling key: %lx\n", tilingKey_);
    return ge::GRAPH_SUCCESS;
}

void MoeFinalizeRoutingV2Regbase::DoPostTiling()
{
    OP_LOGI(context_->GetNodeName(), "MoeFinalizeRoutingV2, DoPostTiling");
    printf("MoeFinalizeRoutingV2, DoPostTiling\n");
    return;
}

REGISTER_OPS_TILING_TEMPLATE(MoeFinalizeRoutingV2, MoeFinalizeRoutingV2Regbase, 30000);

} // namespace optiling