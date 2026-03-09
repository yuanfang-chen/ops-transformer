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

    const gert::Shape* qkvInputShape = context_->GetInputShape(QKV_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, qkvInputShape);
    const gert::Shape* kCacheInputShape = context_->GetInputShape(K_CACHE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, kCacheInputShape);
    const gert::Shape* vCacheInputShape = context_->GetInputShape(V_CACHE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, vCacheInputShape);
    
    seqLengthSum_ = qkvInputShape->GetDim(DIM_ZERO);
    numHead_ = qkvInputShape->GetDim(DIM_ONE);
    qkvDim_ = qkvInputShape->GetDim(DIM_TWO);
    
    blockNum_ = kCacheInputShape->GetDim(DIM_ZERO);
    numHeadK_ = kCacheInputShape->GetDim(DIM_ONE);
    blockSize_ = kCacheInputShape->GetDim(DIM_TWO);
    
    numHeadV_ = vCacheInputShape->GetDim(DIM_ONE);
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
    reciprocal_ = 1.0 / static_cast<float>(qkvDim_);
    
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedKRmsNormRopeStoreKvCacheMxQuantTilingBase::CheckQkvValid()
{
    const gert::Shape* qkvShape = context_->GetInputShape(QKV_INDEX);
    if (qkvShape->GetDimNum() != DIM_SIZE) {
        OP_LOGE(context_->GetNodeName(), "qkv must be 3D tensor [T, N, D].");
        return ge::GRAPH_FAILED;
    }
    if (qkvShape->GetDim(DIM_THREE) != qkvDim_) {
        OP_LOGE(context_->GetNodeName(), "qkv D dimension must be %ld, got %ld.", qkvDim_, qkvShape->GetDim(DIM_THREE));
        return ge::GRAPH_FAILED;
    }
    if (qkvShape->GetDim(DIM_ZERO) <= 0 || qkvShape->GetDim(DIM_ONE) <= 0) {
        OP_LOGE(context_->GetNodeName(), "qkv T and N dimensions must be positive.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedKRmsNormRopeStoreKvCacheMxQuantTilingBase::CheckCosSinValid()
{
    const gert::Shape* cosShape = context_->GetInputShape(COS_INDEX);
    const gert::Shape* sinShape = context_->GetInputShape(SIN_INDEX);

    if (cosShape->GetDimNum() != DIM_SIZE) {
        OP_LOGE(context_->GetNodeName(), "cos must be 3D tensor [T, 1, D].");
        return ge::GRAPH_FAILED;
    }
    if (cosShape->GetDim(DIM_ONE) != DIM_ONE) {
        OP_LOGE(context_->GetNodeName(), "cos second dimension must be 1.");
        return ge::GRAPH_FAILED;
    }

    if (sinShape->GetDim() != DIM_SIZE) {
        OP_LOGE(context_->GetNodeName(), "sin must be 3D tensor [T, 1, D].");
        return ge::GRAPH_FAILED;
    }
    if (sinShape->GetDim(DIM_ONE) != DIM_ONE) {
        OP_LOGE(context_->GetNodeName(), "sin second dimension must be 1.");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedKRmsNormRopeStoreKvCacheMxQuantTilingBase::CheckGammaValid()
{
    const gert::Shape* gammaShape = context_->GetInputShape(GAMMA_INDEX);
    if (gammaShape->GetDimNum() != DIM_ONE) {
        OP_LOGE(context_->GetNodeName(), "gamma must be 1D tensor [D].");
        return ge::GRAPH_FAILED;
    }
    if (gammaShape->GetDim(DIM_ZERO) <= 0) {
        OP_LOGE(context_->GetNodeName(), "gamma dimension must be positive.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedKRmsNormRopeStoreKvCacheMxQuantTilingBase::CheckKvSlotMappingValid()
{
    const gert::::Shape* kvSlotMappingShape = context_->GetInputShape(KV_SLOT_MAPPING_INDEX);
    if (kvSlotMappingShape->GetDimNum() != DIM_ONE) {
        OP_LOGE(context_->GetNodeName(), "kv_slot_mapping must be 1D tensor [T].");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedKRmsNormRopeStoreKvCacheMxQuantTilingBase::CheckVScaleSlotMappingValid()
{
    const gert::Shape* vScaleSlotMappingShape = context_->GetInputShape(V_SCALE_SLOT_MAPPING_INDEX);
    if (vScaleSlotMappingShape->GetDimNum() != DIM_ONE) {
        OP_LOGE(context_->GetNodeName(), "v_scale_slot_mapping must be 1D tensor [T//32].");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedKRmsNormRopeStoreKvCacheMxQuantTilingBase::CheckKCacheValid()
{
    const gert::Shape* kCacheShape = context_->GetInputShape(K_CACHE_INDEX);
    if (kCacheShape->GetDimNum() != DIM_SIZE + DIM_ONE) {
        OP_LOGE(context_->GetNodeName(), "k_cache must be 4D tensor [Bn, Nk, Bs, D].");
        return ge::GRAPH_FAILED;
    }
    if (kCacheShape->GetDim(DIM_ZERO) <= 0 || kCacheShape->GetDim(DIM_ONE) <= 0) {
        OP_LOGE(context_->GetNodeName(), "k_cache Bn and Nk dimensions must be positive.");
        return ge::GRAPH_FAILED;
    }
    if (kCacheShape->GetDim(DIM_TWO) <= 0) {
        OP_LOGE(context_->GetNodeName(), "k_cache Bs dimension must be positive.");
        return ge::GRAPH_FAILED;
    }
    if (kCacheShape->GetDim(DIM_THREE) <= 0) {
        OP_LOGE(context_->GetNodeName(), "k_cache D dimension must be positive.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedKRmsNormRopeStoreKvCacheMxQuantTilingBase::CheckKScaleCacheValid()
{
    const gert::Shape* kScaleCacheShape = context_->GetInputShape(K_SCALE_CACHE_INDEX);
    if (kScaleCacheShape->GetDimNum() != DIM_SIZE + DIM_ONE) {
        OP_LOGE(context_->GetNodeName(), "k_scale_cache must be 4D tensor [Bn, Nk, Bs, D//32].");
        return ge::GRAPH_FAILED;
    }
    if (kScaleCacheShape->GetDim(DIM_ZERO) <= 0 || kScaleCacheShape->GetDim(DIM_ONE) <= 0) {
        OP_LOGE(context_->GetNodeName(), "k_scale_cache Bn and Nk dimensions must be positive.");
        return ge::GRAPH_FAILED;
    }
    if (kScaleCacheShape->GetDim(DIM_TWO) <= 0) {
        OP_LOGE(context_->GetNodeName(), "k_scale_cache Bs dimension must be positive.");
        return ge::GRAPH_FAILED;
    }
    if (kScaleCacheShape->GetDim(DIM_THREE) <= 0) {
        OP_LOGE(context_->GetNodeName(), "k_scale_cache D dimension must be positive.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedKRmsNormRopeStoreKvCacheMxQuantTilingBase::CheckVCacheValid()
{
    const gert::Shape* vCacheShape = context_->GetInputShape(V_CACHE_INDEX);
    if (vCacheShape->GetDimNum() != DIM_SIZE + DIM_ONE) {
        OP_LOGE(context_->GetNodeName(), "v_cache must be 4D tensor [Bn, Nv, Bs, D].");
        return ge::GRAPH_FAILED;
    }
    if (vCacheShape->GetDim(DIM_ZERO) <= 0 || vCacheShape->GetDim(DIM_ONE) <= 0) {
        OP_LOGE(context_->GetNodeName(), "v_cache Bn and Nv dimensions must be positive.");
        return ge::GRAPH_FAILED;
    }
    if (vCacheShape->GetDim(DIM_TWO) <= 0) {
        OP_LOGE(context_->GetNodeName(), "v_cache Bs dimension must be positive.");
        return ge::GRAPH_FAILED;
    }
    if (vCacheShape->GetDim(DIM_THREE) <= 0) {
        OP_LOGE(context_->GetNodeName(), "v_cache D dimension must be positive.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedKRmsNormRopeStoreKvCacheMxQuantTilingBase::CheckVScaleCacheValid()
{
    const gert::Shape* vScaleCacheShape = context_->GetInputShape(V_SCALE_CACHE_INDEX);
    if (vScaleCacheShape->GetDimNum() != DIM_SIZE + DIM_ONE) {
        OP_LOGE(context_->GetNodeName(), "v_scale_cache must be 4D tensor [Bn, Nv, Bs//32, D].");
        return ge::GRAPH_FAILED;
    }
    if (vScaleCacheShape->GetDim(DIM_ZERO) <= 0 || vScaleCacheShape->GetDim(DIM_ONE) <= 0) {
        OP_LOGE(context_->GetNodeName(), "v_scale_cache Bn and Nv dimensions must be positive.");
        return ge::GRAPH_FAILED;
    }
    if (vScaleCacheShape->GetDim(DIM_TWO) <= 0) {
        OP_LOGE(context_->GetNodeName(), "v_scale_cache Bs dimension must be positive.");
        return ge::GRAPH_FAILED;
    }
    if (vScaleCacheShape->GetDim(DIM_THREE) <= 0) {
        OP_LOGE(context_->GetNodeName(), "v_scale_cache D dimension must be positive.");
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
