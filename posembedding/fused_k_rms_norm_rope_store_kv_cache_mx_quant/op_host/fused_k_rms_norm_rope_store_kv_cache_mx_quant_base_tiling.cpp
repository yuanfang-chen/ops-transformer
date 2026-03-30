/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fused_k_rms_norm_rope_store_kv_cache_mx_quant_base_tiling.cpp
 * \brief
 */

#include "fused_k_rms_norm_rope_store_kv_cache_mx_quant_tiling.h"
#include "log/log.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_impl_registry.h"
#include "util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "tiling_base/tiling_util.h"

namespace optiling {

ge::graphStatus FusedKRmsNormRopeStoreKvCacheMxQuantTilingBase::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo == nullptr) {
        auto compileInfoPtr = context_->GetCompileInfo<FusedKRmsNormRopeStoreKvCacheMxQuantCompileInfo>();
        OP_CHECK_IF(
            compileInfoPtr == nullptr, OP_LOGE(context_->GetNodeName(), "CompileInfo is nullptr."),
            return ge::GRAPH_FAILED);
        coreNum_ = compileInfoPtr->coreNum;
        ubSize_ = compileInfoPtr->ubSize;
    } else {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        coreNum_ = ascendcPlatform.GetCoreNumAiv();
        uint64_t ubSize = 0;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
        ubSize_ = ubSize;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedKRmsNormRopeStoreKvCacheMxQuantTilingBase::GetShapeAttrsInfo()
{
    OP_CHECK_IF(
        context_ == nullptr, OP_LOGE(context_->GetNodeName(), "context_ can not be nullptr."), return ge::GRAPH_FAILED);

    auto qkvInputShapePtr = context_->GetInputShape(QKV_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, qkvInputShapePtr);
    auto& qkvInputShape = qkvInputShapePtr->GetStorageShape();
    auto kCacheInputShapePtr = context_->GetInputShape(K_CACHE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, kCacheInputShapePtr);
    auto& kCacheInputShape = kCacheInputShapePtr->GetStorageShape();
    auto vCacheInputShapePtr = context_->GetInputShape(V_CACHE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, vCacheInputShapePtr);
    auto& vCacheInputShape = vCacheInputShapePtr->GetStorageShape();

    seqLengthSum_ = qkvInputShape.GetDim(DIM_ZERO);
    numHead_ = qkvInputShape.GetDim(DIM_ONE);
    headDim_ = qkvInputShape.GetDim(DIM_TWO);

    blockNum_ = kCacheInputShape.GetDim(DIM_ZERO);
    numHeadK_ = kCacheInputShape.GetDim(DIM_ONE);
    blockSize_ = kCacheInputShape.GetDim(DIM_TWO);

    numHeadV_ = vCacheInputShape.GetDim(DIM_ONE);
    numHeadQ_ = numHead_ - numHeadK_ - numHeadV_;
    
    OP_CHECK_IF(numHeadQ_ <= 0,
        OP_LOGE(context_->GetNodeName(), "numHeadQ must be positive, numHead=%ld, numHeadK=%ld, numHeadV=%ld.",
                 numHead_, numHeadK_, numHeadV_), return ge::GRAPH_FAILED);
    
    auto qkvDesc = context_->GetInputDesc(QKV_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, qkvDesc);
    qkvDtype_ = qkvDesc->GetDataType();
    if (qkvDtype_ == ge::DT_FLOAT16 || qkvDtype_ == ge::DT_BF16) {
        qkvDtypeSize_ = FLOAT16_BYTES;
    } else {
        OP_LOGE(context_->GetNodeName(), "qkv dtype only support FP16/BF16.");
        return ge::GRAPH_FAILED;
    }
    
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    
    const float* epsilon = attrs->GetFloat(EPSILON_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, epsilon);
    epsilon_ = *epsilon;
    reciprocal_ = 1.0 / static_cast<float>(headDim_);

    OP_CHECK_IF(CheckQkvValid() != ge::GRAPH_SUCCESS,
        OP_LOGE(context_->GetNodeName(), "CheckQkvValid failed."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckCosSinValid() != ge::GRAPH_SUCCESS,
        OP_LOGE(context_->GetNodeName(), "CheckCosSinValid failed."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckGammaValid() != ge::GRAPH_SUCCESS,
        OP_LOGE(context_->GetNodeName(), "CheckGammaValid failed."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckKvSlotMappingValid() != ge::GRAPH_SUCCESS,
        OP_LOGE(context_->GetNodeName(), "CheckKvSlotMappingValid failed."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckVScaleSlotMappingValid() != ge::GRAPH_SUCCESS,
        OP_LOGE(context_->GetNodeName(), "CheckVScaleSlotMappingValid failed."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckKCacheValid() != ge::GRAPH_SUCCESS,
        OP_LOGE(context_->GetNodeName(), "CheckKCacheValid failed."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckKScaleCacheValid() != ge::GRAPH_SUCCESS,
        OP_LOGE(context_->GetNodeName(), "CheckKScaleCacheValid failed."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckVCacheValid() != ge::GRAPH_SUCCESS,
        OP_LOGE(context_->GetNodeName(), "CheckVCacheValid failed."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckVScaleCacheValid() != ge::GRAPH_SUCCESS,
        OP_LOGE(context_->GetNodeName(), "CheckVScaleCacheValid failed."), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedKRmsNormRopeStoreKvCacheMxQuantTilingBase::CheckQkvValid()
{
    auto& qkvShape = context_->GetInputShape(QKV_INDEX)->GetStorageShape();
    if (qkvShape.GetDimNum() != DIM_THREE) {
        OP_LOGE(context_->GetNodeName(), "qkv must be 3D tensor [T, N, D].");
        return ge::GRAPH_FAILED;
    }
    if (qkvShape.GetDim(DIM_TWO) != headDim_) {
        OP_LOGE(context_->GetNodeName(), "qkv D dimension must be %ld, got %ld.", headDim_, qkvShape.GetDim(DIM_TWO));
        return ge::GRAPH_FAILED;
    }
    if (qkvShape.GetDim(DIM_ZERO) <= 0 || qkvShape.GetDim(DIM_ONE) <= 0) {
        OP_LOGE(context_->GetNodeName(), "qkv T and N dimensions must be positive.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedKRmsNormRopeStoreKvCacheMxQuantTilingBase::CheckCosSinValid()
{
    auto& cosShape = context_->GetInputShape(COS_INDEX)->GetStorageShape();
    auto& sinShape = context_->GetInputShape(SIN_INDEX)->GetStorageShape();

    if (cosShape.GetDimNum() != DIM_THREE) {
        OP_LOGE(context_->GetNodeName(), "cos must be 3D tensor [T, 1, D].");
        return ge::GRAPH_FAILED;
    }
    if (cosShape.GetDim(DIM_ZERO) != seqLengthSum_) {
        OP_LOGE(context_->GetNodeName(), "cos T dimension must be %ld, got %ld.", seqLengthSum_,
                cosShape.GetDim(DIM_ZERO));
        return ge::GRAPH_FAILED;
    }
    if (cosShape.GetDim(DIM_ONE) != DIM_ONE) {
        OP_LOGE(context_->GetNodeName(), "cos second dimension must be 1.");
        return ge::GRAPH_FAILED;
    }
    if (cosShape.GetDim(DIM_TWO) != headDim_) {
        OP_LOGE(context_->GetNodeName(), "cos D dimension must be %ld, got %ld.", headDim_,
                cosShape.GetDim(DIM_TWO));
        return ge::GRAPH_FAILED;
    }

    if (sinShape.GetDimNum() != DIM_THREE) {
        OP_LOGE(context_->GetNodeName(), "sin must be 3D tensor [T, 1, D].");
        return ge::GRAPH_FAILED;
    }
    if (sinShape.GetDim(DIM_ZERO) != seqLengthSum_) {
        OP_LOGE(context_->GetNodeName(), "sin T dimension must be %ld, got %ld.", seqLengthSum_,
                sinShape.GetDim(DIM_ZERO));
        return ge::GRAPH_FAILED;
    }
    if (sinShape.GetDim(DIM_ONE) != DIM_ONE) {
        OP_LOGE(context_->GetNodeName(), "sin second dimension must be 1.");
        return ge::GRAPH_FAILED;
    }
    if (sinShape.GetDim(DIM_TWO) != headDim_) {
        OP_LOGE(context_->GetNodeName(), "sin D dimension must be %ld, got %ld.", headDim_,
                sinShape.GetDim(DIM_TWO));
        return ge::GRAPH_FAILED;
    }

    auto cosDesc = context_->GetInputDesc(COS_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, cosDesc);
    if (cosDesc->GetDataType() != qkvDtype_) {
        OP_LOGE(context_->GetNodeName(), "cos dtype must match qkv dtype, expected %d, got %d.",
                qkvDtype_, cosDesc->GetDataType());
        return ge::GRAPH_FAILED;
    }
    auto sinDesc = context_->GetInputDesc(SIN_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, sinDesc);
    if (sinDesc->GetDataType() != qkvDtype_) {
        OP_LOGE(context_->GetNodeName(), "sin dtype must match qkv dtype, expected %d, got %d.",
                qkvDtype_, sinDesc->GetDataType());
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedKRmsNormRopeStoreKvCacheMxQuantTilingBase::CheckGammaValid()
{
    auto& gammaShape = context_->GetInputShape(GAMMA_INDEX)->GetStorageShape();
    if (gammaShape.GetDimNum() != DIM_ONE) {
        OP_LOGE(context_->GetNodeName(), "gamma must be 1D tensor [D].");
        return ge::GRAPH_FAILED;
    }
    if (gammaShape.GetDim(DIM_ZERO) != headDim_) {
        OP_LOGE(context_->GetNodeName(), "gamma dimension must be %ld, got %ld.", headDim_,
                gammaShape.GetDim(DIM_ZERO));
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedKRmsNormRopeStoreKvCacheMxQuantTilingBase::CheckKvSlotMappingValid()
{
    auto& kvSlotMappingShape = context_->GetInputShape(KV_SLOT_MAPPING_INDEX)->GetStorageShape();
    if (kvSlotMappingShape.GetDimNum() != DIM_ONE) {
        OP_LOGE(context_->GetNodeName(), "kv_slot_mapping must be 1D tensor [T].");
        return ge::GRAPH_FAILED;
    }
    if (kvSlotMappingShape.GetDim(DIM_ZERO) != seqLengthSum_) {
        OP_LOGE(context_->GetNodeName(), "kv_slot_mapping length must be %ld, got %ld.", seqLengthSum_,
                kvSlotMappingShape.GetDim(DIM_ZERO));
        return ge::GRAPH_FAILED;
    }
    auto kvSlotMappingDesc = context_->GetInputDesc(KV_SLOT_MAPPING_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, kvSlotMappingDesc);
    if (kvSlotMappingDesc->GetDataType() != ge::DT_INT64) {
        OP_LOGE(context_->GetNodeName(), "kv_slot_mapping dtype must be INT64, got %d.",
                kvSlotMappingDesc->GetDataType());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedKRmsNormRopeStoreKvCacheMxQuantTilingBase::CheckVScaleSlotMappingValid()
{
    auto& vScaleSlotMappingShape = context_->GetInputShape(V_SCALE_SLOT_MAPPING_INDEX)->GetStorageShape();
    if (vScaleSlotMappingShape.GetDimNum() != DIM_ONE) {
        OP_LOGE(context_->GetNodeName(), "v_scale_slot_mapping must be 1D tensor [T/32/2].");
        return ge::GRAPH_FAILED;
    }
    int64_t expectedLen = seqLengthSum_ / QUANT_BLOCK_SIZE / DIGIT_TWO;
    if (vScaleSlotMappingShape.GetDim(DIM_ZERO) != expectedLen) {
        OP_LOGE(context_->GetNodeName(), "v_scale_slot_mapping length must be %ld (T/32/2), got %ld.", expectedLen,
                vScaleSlotMappingShape.GetDim(DIM_ZERO));
        return ge::GRAPH_FAILED;
    }
    auto vScaleSlotMappingDesc = context_->GetInputDesc(V_SCALE_SLOT_MAPPING_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, vScaleSlotMappingDesc);
    if (vScaleSlotMappingDesc->GetDataType() != ge::DT_INT64) {
        OP_LOGE(context_->GetNodeName(), "v_scale_slot_mapping dtype must be INT64, got %d.",
                vScaleSlotMappingDesc->GetDataType());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedKRmsNormRopeStoreKvCacheMxQuantTilingBase::CheckKCacheValid()
{
    auto& kCacheShape = context_->GetInputShape(K_CACHE_INDEX)->GetStorageShape();
    if (kCacheShape.GetDimNum() != DIM_SIZE) {
        OP_LOGE(context_->GetNodeName(), "k_cache must be 4D tensor [Bn, Nk, Bs, D].");
        return ge::GRAPH_FAILED;
    }
    if (kCacheShape.GetDim(DIM_ZERO) <= 0 || kCacheShape.GetDim(DIM_ONE) <= 0) {
        OP_LOGE(context_->GetNodeName(), "k_cache Bn and Nk dimensions must be positive.");
        return ge::GRAPH_FAILED;
    }
    if (kCacheShape.GetDim(DIM_TWO) <= 0) {
        OP_LOGE(context_->GetNodeName(), "k_cache Bs dimension must be positive.");
        return ge::GRAPH_FAILED;
    }
    if (kCacheShape.GetDim(DIM_THREE) != headDim_) {
        OP_LOGE(context_->GetNodeName(), "k_cache D dimension must be %ld, got %ld.", headDim_,
                kCacheShape.GetDim(DIM_THREE));
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedKRmsNormRopeStoreKvCacheMxQuantTilingBase::CheckKScaleCacheValid()
{
    auto& kScaleCacheShape = context_->GetInputShape(K_SCALE_CACHE_INDEX)->GetStorageShape();
    if (kScaleCacheShape.GetDimNum() != DIM_SIZE + DIM_ONE) {
        OP_LOGE(context_->GetNodeName(), "k_scale_cache must be 5D tensor [Bn, Nk, Bs, D/32/2, 2].");
        return ge::GRAPH_FAILED;
    }
    if (kScaleCacheShape.GetDim(DIM_ZERO) != blockNum_) {
        OP_LOGE(context_->GetNodeName(), "k_scale_cache Bn must be %ld, got %ld.", blockNum_,
                kScaleCacheShape.GetDim(DIM_ZERO));
        return ge::GRAPH_FAILED;
    }
    if (kScaleCacheShape.GetDim(DIM_ONE) != numHeadK_) {
        OP_LOGE(context_->GetNodeName(), "k_scale_cache Nk must be %ld, got %ld.", numHeadK_,
                kScaleCacheShape.GetDim(DIM_ONE));
        return ge::GRAPH_FAILED;
    }
    if (kScaleCacheShape.GetDim(DIM_TWO) != blockSize_) {
        OP_LOGE(context_->GetNodeName(), "k_scale_cache Bs must be %ld, got %ld.", blockSize_,
                kScaleCacheShape.GetDim(DIM_TWO));
        return ge::GRAPH_FAILED;
    }
    int64_t expectedScaleDim = headDim_ / QUANT_BLOCK_SIZE / DIGIT_TWO;
    if (kScaleCacheShape.GetDim(DIM_THREE) != expectedScaleDim) {
        OP_LOGE(context_->GetNodeName(), "k_scale_cache D/32/2 dimension must be %ld, got %ld.", expectedScaleDim,
                kScaleCacheShape.GetDim(DIM_THREE));
        return ge::GRAPH_FAILED;
    }
    if (kScaleCacheShape.GetDim(DIM_SIZE) != DIGIT_TWO) {
        OP_LOGE(context_->GetNodeName(), "k_scale_cache last dimension must be 2.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedKRmsNormRopeStoreKvCacheMxQuantTilingBase::CheckVCacheValid()
{
    auto& vCacheShape = context_->GetInputShape(V_CACHE_INDEX)->GetStorageShape();
    if (vCacheShape.GetDimNum() != DIM_SIZE) {
        OP_LOGE(context_->GetNodeName(), "v_cache must be 4D tensor [Bn, Nv, Bs, D].");
        return ge::GRAPH_FAILED;
    }
    if (vCacheShape.GetDim(DIM_ZERO) != blockNum_) {
        OP_LOGE(context_->GetNodeName(), "v_cache Bn must be %ld, got %ld.", blockNum_,
                vCacheShape.GetDim(DIM_ZERO));
        return ge::GRAPH_FAILED;
    }
    if (vCacheShape.GetDim(DIM_ONE) <= 0) {
        OP_LOGE(context_->GetNodeName(), "v_cache Nv dimension must be positive.");
        return ge::GRAPH_FAILED;
    }
    if (vCacheShape.GetDim(DIM_TWO) != blockSize_) {
        OP_LOGE(context_->GetNodeName(), "v_cache Bs must be %ld, got %ld.", blockSize_,
                vCacheShape.GetDim(DIM_TWO));
        return ge::GRAPH_FAILED;
    }
    if (vCacheShape.GetDim(DIM_THREE) != headDim_) {
        OP_LOGE(context_->GetNodeName(), "v_cache D dimension must be %ld, got %ld.", headDim_,
                vCacheShape.GetDim(DIM_THREE));
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedKRmsNormRopeStoreKvCacheMxQuantTilingBase::CheckVScaleCacheValid()
{
    auto& vScaleCacheShape = context_->GetInputShape(V_SCALE_CACHE_INDEX)->GetStorageShape();
    if (vScaleCacheShape.GetDimNum() != DIM_SIZE + DIM_ONE) {
        OP_LOGE(context_->GetNodeName(), "v_scale_cache must be 5D tensor [Bn, Nv, Bs/32/2, D, 2].");
        return ge::GRAPH_FAILED;
    }
    if (vScaleCacheShape.GetDim(DIM_ZERO) != blockNum_) {
        OP_LOGE(context_->GetNodeName(), "v_scale_cache Bn must be %ld, got %ld.", blockNum_,
                vScaleCacheShape.GetDim(DIM_ZERO));
        return ge::GRAPH_FAILED;
    }
    if (vScaleCacheShape.GetDim(DIM_ONE) != numHeadV_) {
        OP_LOGE(context_->GetNodeName(), "v_scale_cache Nv must be %ld, got %ld.", numHeadV_,
                vScaleCacheShape.GetDim(DIM_ONE));
        return ge::GRAPH_FAILED;
    }
    int64_t expectedBsScale = blockSize_ / QUANT_BLOCK_SIZE / DIGIT_TWO;
    if (vScaleCacheShape.GetDim(DIM_TWO) != expectedBsScale) {
        OP_LOGE(context_->GetNodeName(), "v_scale_cache Bs/32/2 dimension must be %ld, got %ld.", expectedBsScale,
                vScaleCacheShape.GetDim(DIM_TWO));
        return ge::GRAPH_FAILED;
    }
    if (vScaleCacheShape.GetDim(DIM_THREE) != headDim_) {
        OP_LOGE(context_->GetNodeName(), "v_scale_cache D dimension must be %ld, got %ld.", headDim_,
                vScaleCacheShape.GetDim(DIM_THREE));
        return ge::GRAPH_FAILED;
    }
    if (vScaleCacheShape.GetDim(DIM_SIZE) != DIGIT_TWO) {
        OP_LOGE(context_->GetNodeName(), "v_scale_cache last dimension must be 2.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

uint64_t FusedKRmsNormRopeStoreKvCacheMxQuantTilingBase::GetTilingKey() const
{
    return tilingKey_;
}

ge::graphStatus Tiling4FusedKRmsNormRopeStoreKvCacheMxQuant(gert::TilingContext* context)
{
    OP_LOGD(context, "TilingForFusedKRmsNormRopeStoreKvCacheMxQuant running.");
    return Ops::Transformer::OpTiling::TilingRegistry::GetInstance().DoTilingImpl(context);
}

ge::graphStatus TilingPrepare4FusedKRmsNormRopeStoreKvCacheMxQuant(gert::TilingParseContext* context)
{
    OP_LOGD(context, "TilingPrepare4FusedKRmsNormRopeStoreKvCacheMxQuant running.");
    auto compileInfo = context->GetCompiledInfo<FusedKRmsNormRopeStoreKvCacheMxQuantCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(
        compileInfo->coreNum <= 0, OP_LOGE(context, "coreNum must be greater than 0."), return ge::GRAPH_FAILED);
    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    compileInfo->ubSize = ubSize;
    OP_CHECK_IF(compileInfo->ubSize <= 0, OP_LOGE(context, "ubSize must be greater than 0."), return ge::GRAPH_FAILED);
    OP_LOGD(context, "coreNum: %ld, ubSize: %ld", compileInfo->coreNum, compileInfo->ubSize);
    OP_LOGD(context, "TilingPrepare4FusedKRmsNormRopeStoreKvCacheMxQuant success.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(FusedKRmsNormRopeStoreKvCacheMxQuant)
    .Tiling(Tiling4FusedKRmsNormRopeStoreKvCacheMxQuant)
    .TilingParse<FusedKRmsNormRopeStoreKvCacheMxQuantCompileInfo>(TilingPrepare4FusedKRmsNormRopeStoreKvCacheMxQuant);

} // namespace optiling
