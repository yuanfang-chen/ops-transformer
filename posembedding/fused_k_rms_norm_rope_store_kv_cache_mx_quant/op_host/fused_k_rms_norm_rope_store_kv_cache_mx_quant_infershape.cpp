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
 * \file fused_k_rms_norm_rope_store_kv_cache_mx_quant_infershape.cpp
 * \brief
 */

#include "register/op_impl_registry.h"
#include "log/log.h"
#include "util/shape_util.h"
using namespace ge;
namespace ops {

constexpr int64_t QKV_INDEX = 0;
constexpr int64_t COS_INDEX = 1;
constexpr int64_t SIN_INDEX = 2;
constexpr int64_t GAMMA_INDEX = 3;
constexpr int64_t KV_SLOT_MAPPING_INDEX = 4;
constexpr int64_t V_SCALE_SLOT_MAPPING_INDEX = 5;
constexpr int64_t K_CACHE_INDEX = 6;
constexpr int64_t K_SCALE_CACHE_INDEX = 7;
constexpr int64_t V_CACHE_INDEX = 8;
constexpr int64_t V_SCALE_CACHE_INDEX = 9;

constexpr size_t OUTPUT_IDX_Q = 0;
constexpr size_t OUTPUT_IDX_Q_SCALE = 1;
constexpr size_t OUTPUT_IDX_K_CACHE = 2;
constexpr size_t OUTPUT_IDX_K_SCALE_CACHE = 3;
constexpr size_t OUTPUT_IDX_V_CACHE = 4;
constexpr size_t OUTPUT_IDX_V_SCALE_CACHE = 5;

constexpr int64_t DIM_ZERO = 0;
constexpr int64_t DIM_ONE = 1;
constexpr int64_t DIM_TWO = 2;
constexpr int64_t DIM_THREE = 3;
constexpr int64_t DIM_FOUR = 4;
constexpr int64_t SUPPORTED_D = 128;
constexpr int64_t QUANT_BLOCK_SIZE = 32;
constexpr int64_t DIGIT_TWO = 2;

graphStatus InferShape4FusedKRmsNormRopeStoreKvCacheMxQuant(gert::InferShapeContext* context)
{
    OP_LOGI(context, "Begin to do InferShape4FusedKRmsNormRopeStoreKvCacheMxQuant.");

    const gert::Shape* qkvInputShape = context->GetInputShape(QKV_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, qkvInputShape);
    const gert::Shape* cosInputShape = context->GetInputShape(COS_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, cosInputShape);
    const gert::Shape* sinInputShape = context->GetInputShape(SIN_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, sinInputShape);
    const gert::Shape* gammaInputShape = context->GetInputShape(GAMMA_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, gammaInputShape);
    const gert::Shape* kvSlotMappingInputShape = context->GetInputShape(KV_SLOT_MAPPING_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, kvSlotMappingInputShape);
    const gert::Shape* vScaleSlotMappingInputShape = context->GetInputShape(V_SCALE_SLOT_MAPPING_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, vScaleSlotMappingInputShape);
    const gert::Shape* kCacheInputShape = context->GetInputShape(K_CACHE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, kCacheInputShape);
    const gert::Shape* kScaleCacheInputShape = context->GetInputShape(K_SCALE_CACHE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, kScaleCacheInputShape);
    const gert::Shape* vCacheInputShape = context->GetInputShape(V_CACHE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, vCacheInputShape);
    const gert::Shape* vScaleCacheInputShape = context->GetInputShape(V_SCALE_CACHE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, vScaleCacheInputShape);

    gert::Shape* qOutputShape = context->GetOutputShape(OUTPUT_IDX_Q);
    OP_CHECK_NULL_WITH_CONTEXT(context, qOutputShape);
    gert::Shape* qScaleOutputShape = context->GetOutputShape(OUTPUT_IDX_Q_SCALE);
    OP_CHECK_NULL_WITH_CONTEXT(context, qScaleOutputShape);
    gert::Shape* kCacheOutputShape = context->GetOutputShape(OUTPUT_IDX_K_CACHE);
    OP_CHECK_NULL_WITH_CONTEXT(context, kCacheOutputShape);
    gert::Shape* kScaleCacheOutputShape = context->GetOutputShape(OUTPUT_IDX_K_SCALE_CACHE);
    OP_CHECK_NULL_WITH_CONTEXT(context, kScaleCacheOutputShape);
    gert::Shape* vCacheOutputShape = context->GetOutputShape(OUTPUT_IDX_V_CACHE);
    OP_CHECK_NULL_WITH_CONTEXT(context, vCacheOutputShape);
    gert::Shape* vScaleCacheOutputShape = context->GetOutputShape(OUTPUT_IDX_V_SCALE_CACHE);
    OP_CHECK_NULL_WITH_CONTEXT(context, vScaleCacheOutputShape);

    OP_CHECK_IF(qkvInputShape->GetDimNum() != DIM_TWO + 1,
            OP_LOGE(context, "qkv must be 3D tensor [T, N, D]."), return ge::GRAPH_FAILED);
    int64_t DQKV = qkvInputShape->GetDim(DIM_TWO);
    OP_CHECK_IF(DQKV != SUPPORTED_D,
            OP_LOGE(context, "qkv D dimension must be %ld, got %ld.", SUPPORTED_D, DQKV), return ge::GRAPH_FAILED);
    OP_CHECK_IF(qkvInputShape->GetDim(DIM_ZERO) <= 0 || qkvInputShape->GetDim(DIM_ONE) <= 0,
            OP_LOGE(context, "qkv T and N dimensions must be positive."), return ge::GRAPH_FAILED);

    OP_CHECK_IF(cosInputShape->GetDimNum() != DIM_TWO + 1,
            OP_LOGE(context, "cos must be 3D tensor [T, 1, D]."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(cosInputShape->GetDim(DIM_ONE) != 1,
            OP_LOGE(context, "cos second dimension must be 1."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(sinInputShape->GetDimNum() != DIM_TWO + 1,
            OP_LOGE(context, "sin must be 3D tensor [T, 1, D]."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(sinInputShape->GetDim(DIM_ONE) != 1,
            OP_LOGE(context, "sin second dimension must be 1."), return ge::GRAPH_FAILED);

    OP_CHECK_IF(gammaInputShape->GetDimNum() != DIM_ONE,
            OP_LOGE(context, "gamma must be 1D tensor [D]."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(gammaInputShape->GetDim(DIM_ZERO) <= 0,
            OP_LOGE(context, "gamma dimension must be positive."), return ge::GRAPH_FAILED);

    OP_CHECK_IF(kvSlotMappingInputShape->GetDimNum() != DIM_ONE,
            OP_LOGE(context, "kv_slot_mapping must be 1D tensor [T]."), return ge::GRAPH_FAILED);

    OP_CHECK_IF(vScaleSlotMappingInputShape->GetDimNum() != DIM_ONE,
            OP_LOGE(context, "v_scale_slot_mapping must be 1D tensor [T/32/2]."), return ge::GRAPH_FAILED);

    OP_CHECK_IF(kCacheInputShape->GetDimNum() != DIM_THREE + 1,
            OP_LOGE(context, "k_cache must be 4D tensor [Bn, Nk, Bs, D]."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(kCacheInputShape->GetDim(DIM_ZERO) <= 0 || kCacheInputShape->GetDim(DIM_ONE) <= 0,
            OP_LOGE(context, "k_cache Bn and Nk dimensions must be positive."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(kCacheInputShape->GetDim(DIM_TWO) <= 0,
            OP_LOGE(context, "k_cache Bs dimension must be positive."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(kCacheInputShape->GetDim(DIM_THREE) <= 0,
            OP_LOGE(context, "k_cache D dimension must be positive."), return ge::GRAPH_FAILED);

    OP_CHECK_IF(kScaleCacheInputShape->GetDimNum() != DIM_FOUR + 1,
            OP_LOGE(context, "k_scale_cache must be 5D tensor [Bn, Nk, Bs, D/32/2, 2]."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(kScaleCacheInputShape->GetDim(DIM_ZERO) <= 0 || kScaleCacheInputShape->GetDim(DIM_ONE) <= 0,
            OP_LOGE(context, "k_scale_cache Bn and Nk dimensions must be positive."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(kScaleCacheInputShape->GetDim(DIM_TWO) <= 0,
            OP_LOGE(context, "k_scale_cache Bs dimension must be positive."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(kScaleCacheInputShape->GetDim(DIM_THREE) <= 0,
            OP_LOGE(context, "k_scale_cache D/32/2 dimension must be positive."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(kScaleCacheInputShape->GetDim(DIM_FOUR) != DIGIT_TWO,
            OP_LOGE(context, "k_scale_cache last dimension must be 2."), return ge::GRAPH_FAILED);

    OP_CHECK_IF(vCacheInputShape->GetDimNum() != DIM_THREE + 1,
            OP_LOGE(context, "v_cache must be 4D tensor [Bn, Nv, Bs, D]."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(vCacheInputShape->GetDim(DIM_ZERO) <= 0 || vCacheInputShape->GetDim(DIM_ONE) <= 0,
            OP_LOGE(context, "v_cache Bn and Nv dimensions must be positive."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(vCacheInputShape->GetDim(DIM_TWO) <= 0,
            OP_LOGE(context, "v_cache Bs dimension must be positive."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(vCacheInputShape->GetDim(DIM_THREE) <= 0,
            OP_LOGE(context, "v_cache D dimension must be positive."), return ge::GRAPH_FAILED);

    OP_CHECK_IF(vScaleCacheInputShape->GetDimNum() != DIM_FOUR + 1,
            OP_LOGE(context, "v_scale_cache must be 5D tensor [Bn, Nv, Bs/32/2, D, 2]."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(vScaleCacheInputShape->GetDim(DIM_ZERO) <= 0 || vScaleCacheInputShape->GetDim(DIM_ONE) <= 0,
            OP_LOGE(context, "v_scale_cache Bn and Nv dimensions must be positive."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(vScaleCacheInputShape->GetDim(DIM_TWO) <= 0,
            OP_LOGE(context, "v_scale_cache Bs/32/2 dimension must be positive."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(vScaleCacheInputShape->GetDim(DIM_THREE) <= 0,
            OP_LOGE(context, "v_scale_cache D dimension must be positive."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(vScaleCacheInputShape->GetDim(DIM_FOUR) != DIGIT_TWO,
            OP_LOGE(context, "v_scale_cache last dimension must be 2."), return ge::GRAPH_FAILED);

    qOutputShape->SetDimNum(DIM_TWO + 1);
    qOutputShape->SetDim(DIM_ZERO, qkvInputShape->GetDim(DIM_ZERO));
    int64_t N = qkvInputShape->GetDim(DIM_ONE);
    int64_t Nk = kCacheInputShape->GetDim(DIM_ONE);
    int64_t Nv = vCacheInputShape->GetDim(DIM_ONE);
    int64_t Nq = N - Nk - Nv;
    OP_CHECK_IF(Nq <= 0,
            OP_LOGE(context, "Nq must be positive, N=%ld, Nk=%ld, Nv=%ld.", Nq, Nk, Nv), return ge::GRAPH_FAILED);
    qOutputShape->SetDim(DIM_ONE, Nq);
    qOutputShape->SetDim(DIM_TWO, qkvInputShape->GetDim(DIM_TWO));

    qScaleOutputShape->SetDimNum(DIM_TWO + 1 + 1);
    qScaleOutputShape->SetDim(DIM_ZERO, qkvInputShape->GetDim(DIM_ZERO));
    qScaleOutputShape->SetDim(DIM_ONE, Nq);
    qScaleOutputShape->SetDim(DIM_TWO, qkvInputShape->GetDim(DIM_TWO) / QUANT_BLOCK_SIZE / DIGIT_TWO);
    qScaleOutputShape->SetDim(DIM_THREE, DIGIT_TWO);

    *kCacheOutputShape = *kCacheInputShape;
    *kScaleCacheOutputShape = *kScaleCacheInputShape;
    *vCacheOutputShape = *vCacheInputShape;
    *vScaleCacheOutputShape = *vScaleCacheInputShape;

    OP_LOGD(context, "End to do InferShape4FusedKRmsNormRopeStoreKvCacheMxQuant.");
    return GRAPH_SUCCESS;
}

graphStatus InferDtype4FusedKRmsNormRopeStoreKvCacheMxQuant(gert::InferDataTypeContext* context)
{
    OP_LOGD(context, "InferDtype4FusedKRmsNormRopeStoreKvCacheMxQuant enter");

    auto k_cache_dtype = context->GetInputDataType(K_CACHE_INDEX);
    auto k_scale_cache_dtype = context->GetInputDataType(K_SCALE_CACHE_INDEX);
    auto v_cache_dtype = context->GetInputDataType(V_CACHE_INDEX);
    auto v_scale_cache_dtype = context->GetInputDataType(V_SCALE_CACHE_INDEX);

    context->SetOutputDataType(OUTPUT_IDX_Q, k_cache_dtype);
    context->SetOutputDataType(OUTPUT_IDX_Q_SCALE, k_scale_cache_dtype);
    context->SetOutputDataType(OUTPUT_IDX_K_CACHE, k_cache_dtype);
    context->SetOutputDataType(OUTPUT_IDX_K_SCALE_CACHE, k_scale_cache_dtype);
    context->SetOutputDataType(OUTPUT_IDX_V_CACHE, v_cache_dtype);
    context->SetOutputDataType(OUTPUT_IDX_V_SCALE_CACHE, v_scale_cache_dtype);

    OP_LOGD(context, "InferDtype4FusedKRmsNormRopeStoreKvCacheMxQuant end");

    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(FusedKRmsNormRopeStoreKvCacheMxQuant)
    .InferShape(InferShape4FusedKRmsNormRopeStoreKvCacheMxQuant)
    .InferDataType(InferDtype4FusedKRmsNormRopeStoreKvCacheMxQuant);
} // namespace ops
