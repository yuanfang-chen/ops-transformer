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
 * \file fused_k_rms_norm_rope_store_kv_cache_mx_quant_tiling.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_FUSED_K_RMS_NORM_ROPE_STORE_KV_CACHE_MX_QUANT_H_
#define OPS_BUILT_IN_OP_TILING_RUNTIME_FUSED_K_RMS_NORM_ROPE_STORE_KV_CACHE_MX_QUANT_H_

#include "register/tilingdata_base.h"
#include "tiling_base/tiling_base.h"
#include "tiling_base/tiling_templates_registry.h"
#include <unordered_map>
namespace optiling {
BEGIN_TILING_DATA_DEF(FusedKRmsNormRopeStoreKvCacheMxQuantTilingData)
TILING_DATA_FIELD_DEF(int64_t, seqLengthSum);
TILING_DATA_FIELD_DEF(int64_t, qkvNumHead);
TILING_DATA_FIELD_DEF(int64_t, qNumHead);
TILING_DATA_FIELD_DEF(int64_t, kNumHead);
TILING_DATA_FIELD_DEF(int64_t, vNumHead);
TILING_DATA_FIELD_DEF(int64_t, headDim);
TILING_DATA_FIELD_DEF(int64_t, blockNum);
TILING_DATA_FIELD_DEF(int64_t, blockSize);
TILING_DATA_FIELD_DEF(int64_t, qUsedCoreNum);
TILING_DATA_FIELD_DEF(int64_t, qBlockFactor);
TILING_DATA_FIELD_DEF(int64_t, qUbFactor);
TILING_DATA_FIELD_DEF(int64_t, kUsedCoreNum);
TILING_DATA_FIELD_DEF(int64_t, kBlockFactor);
TILING_DATA_FIELD_DEF(int64_t, kUbFactor);
TILING_DATA_FIELD_DEF(int64_t, vUsedCoreNum);
TILING_DATA_FIELD_DEF(int64_t, vBlockFactor);
TILING_DATA_FIELD_DEF(int64_t, vTUbFactor);            // T轴上的ub切分值
TILING_DATA_FIELD_DEF(int64_t, vNumHeadUbFactor);      // N轴上的ub切分值
TILING_DATA_FIELD_DEF(float, epsilon);
TILING_DATA_FIELD_DEF(float, reciprocal);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(FusedKRmsNormRopeStoreKvCacheMxQuant, FusedKRmsNormRopeStoreKvCacheMxQuantTilingData)

struct FusedKRmsNormRopeStoreKvCacheMxQuantCompileInfo {
    int64_t coreNum = 0;
    int64_t ubSize = 0;
};

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

constexpr int64_t EPSILON_IDX = 0;

constexpr int64_t FLOAT32_BYTES = 4;
constexpr int64_t FLOAT16_BYTES = 2;
constexpr int64_t INT8_BYTES = 1;
constexpr int64_t FP32_BLOCK_ALIGN_NUM = 8;
constexpr int64_t FP16_BLOCK_ALIGN_NUM = 16;
constexpr int64_t INT8_BLOCK_ALIGN_NUM = 32;
constexpr int64_t QUANT_BLOCK_SIZE = 32;
constexpr int64_t DIGIT_TWO = 2;
constexpr int64_t DOUBLE_BUFFER = 2;

constexpr int64_t DIM_SIZE = 4;
constexpr int64_t DIM_ZERO = 0;
constexpr int64_t DIM_ONE = 1;
constexpr int64_t DIM_TWO = 2;
constexpr int64_t DIM_THREE = 3;

class FusedKRmsNormRopeStoreKvCacheMxQuantTilingBase : public Ops::Transformer::OpTiling::TilingBaseClass {
public:
    explicit FusedKRmsNormRopeStoreKvCacheMxQuantTilingBase(gert::TilingContext* tillingContext) : TilingBaseClass(tillingContext)
    {}
    ~FusedKRmsNormRopeStoreKvCacheMxQuantTilingBase() override
    {}
    uint64_t tilingKey_{0};
    int64_t coreNum_ = 0;
    int64_t ubSize_ = 0;
    int64_t seqLengthSum_ = 0;
    int64_t numHead_ = 0;
    int64_t headDim_ = 0;
    int64_t numHeadQ_ = 0;
    int64_t numHeadK_ = 0;
    int64_t numHeadV_ = 0;
    int64_t blockNum_ = 0;
    int64_t blockSize_ = 0;
    float epsilon_ = 0.0;
    float reciprocal_ = 0.0;
    ge::DataType qkvDtype_{ge::DataType::DT_BF16};
    int64_t qkvDtypeSize_{0};

protected:
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus GetPlatformInfo() override;
    bool IsCapable() override
    {
        return false;
    }
    ge::graphStatus DoOpTiling() override
    {
        return ge::GRAPH_SUCCESS;
    }
    ge::graphStatus DoLibApiTiling() override
    {
        return ge::GRAPH_SUCCESS;
    }
    ge::graphStatus GetWorkspaceSize() override
    {
        return ge::GRAPH_SUCCESS;
    }
    uint64_t GetTilingKey() const override;

    ge::graphStatus CheckQkvValid();
    ge::graphStatus CheckCosSinValid();
    ge::graphStatus CheckGammaValid();
    ge::graphStatus CheckKvSlotMappingValid();
    ge::graphStatus CheckVScaleSlotMappingValid();
    ge::graphStatus CheckKCacheValid();
    ge::graphStatus CheckKScaleCacheValid();
    ge::graphStatus CheckVCacheValid();
    ge::graphStatus CheckVScaleCacheValid();
};

class FusedKRmsNormRopeStoreKvCacheMxQuantRegbaseTiling : virtual public FusedKRmsNormRopeStoreKvCacheMxQuantTilingBase {
public:
    explicit FusedKRmsNormRopeStoreKvCacheMxQuantRegbaseTiling(gert::TilingContext* tillingContext) : FusedKRmsNormRopeStoreKvCacheMxQuantTilingBase(tillingContext)
    {}
    ~FusedKRmsNormRopeStoreKvCacheMxQuantRegbaseTiling()
    {}
 
protected:
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus PostTiling() override;
 
protected:
    ge::graphStatus GetShapeAttrsInfoInner();
    ge::graphStatus CalUbTiling();
    void PrintTilingData();

private:
    FusedKRmsNormRopeStoreKvCacheMxQuantTilingData tilingData_;
};
} // namespace optiling

#endif // OPS_BUILT_IN_OP_TILING_RUNTIME_FUSED_K_RMS_NORM_ROPE_STORE_KV_CACHE_MX_QUANT_H_
