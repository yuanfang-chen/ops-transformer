/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file matmul_reduce_scatter_v2_aiv_mode_tiling.cpp
 * \brief
 */
#include "matmul_reduce_scatter_v2_tiling.h"
#include "platform/platform_infos_def.h"
#include "vector"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "mc2_log.h"
#include "register/op_def_registry.h"
#include "tiling/mc2_tiling_utils.h"
#include "../../op_kernel/matmul_reduce_scatter_v2_aiv_mode_tiling.h"
#include "../../op_kernel/matmul_reduce_scatter_v2_aiv_tiling_key.h"


using namespace AscendC;
using namespace ge;
using namespace MatmulReduceScatterv2TilingKey;
using namespace matmulReduceScatterV2_aivmode_tiling;
namespace{
    const char *K_INNER_DEBUG = "MatmulReduceScatterV2AivMode Tiling Debug";
    constexpr uint32_t ATTR_GROUP_INDEX = 0;
    constexpr uint32_t ATTR_IS_TRANS_A = 2;
    constexpr uint32_t ATTR_IS_TRANS_B = 3;
    constexpr uint32_t A_INDEX = 0;
    constexpr uint32_t B_INDEX = 1;
    constexpr uint32_t BIAS_INDEX = 2;
    constexpr uint32_t X1_SCALE_INDEX = 3;
    constexpr uint32_t X2_SCALE_INDEX = 4;
    constexpr uint32_t C_INDEX = 0;
    constexpr uint32_t SYSTEM_NEED_WORKSPACE = 16 * 1024 * 1024;
    constexpr uint32_t USER_WORKSPACE_A2 = 1 * 1024 * 1024; // moeExpertNum_ * sizeof(uint32_t) + epWorldSize_ * 2 * 32
    constexpr uint32_t OP_TYPE_REDUCE_SCATTER = 7U;
}

namespace optiling {
static ge::graphStatus MatmulReduceScatterV2CheckAttrAndSetTiling(gert::TilingContext *context, MatmulReduceScatterV2AivModeInfo& info)
{
    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, VECTOR_INNER_ERR_REPORT_TILING(context->GetNodeName(), "AivMode attrs is null."), return ge::GRAPH_FAILED);

    // todo：Attr相关tilingdata的设置、校验、打印
    auto groupPtr = attrs->GetAttrPointer<char>(static_cast<int>(ATTR_GROUP_INDEX));
    auto is_trans_a = attrs->GetAttrPointer<bool>(ATTR_IS_TRANS_A);
    auto is_trans_b = attrs->GetAttrPointer<bool>(ATTR_IS_TRANS_B);
    OP_TILING_CHECK(groupPtr == nullptr || strlen(groupPtr) == 0,
        VECTOR_INNER_ERR_REPORT_TILING(context->GetNodeName(), "AivMode group is invalid."), return GRAPH_FAILED);
    OP_TILING_CHECK(is_trans_a == nullptr || is_trans_b == nullptr,
        VECTOR_INNER_ERR_REPORT_TILING(context->GetNodeName(), "AivMode, is_trans_a or is_trans_b is invalid."), return GRAPH_FAILED);
    info.isTransposeA = false; // 当前默认a矩阵不转置
    info.isTransposeB = *is_trans_b ? *is_trans_b : false;

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus MatmulReduceScatterV2CheckShapeAndSetTiling(gert::TilingContext *context, MatmulReduceScatterV2AivModeInfo &info)
{
    const char *nodeName = context->GetNodeName();
    OP_LOGI("MatmulReduceScatterV2AivMode MatmulReduceScatterV2CheckShapeAndSetTiling.");

    const gert::StorageShape *aStorageShape = context->GetInputShape(A_INDEX);
    const gert::StorageShape *bStorageShape = context->GetInputShape(B_INDEX);
    uint32_t M = aStorageShape->GetStorageShape().GetDim(0);
    uint32_t K = aStorageShape->GetStorageShape().GetDim(1);
    uint32_t N = bStorageShape->GetStorageShape().GetDim(1);

    if (aStorageShape->GetStorageShape().GetDim(1) != bStorageShape->GetStorageShape().GetDim(0)) {
        OP_LOGD(nodeName, "A.shape(1) %lu B.shape(0) %lu, istransB = %d",
                  aStorageShape->GetStorageShape().GetDim(1), bStorageShape->GetStorageShape().GetDim(0), info.isTransposeB);
        N = bStorageShape->GetStorageShape().GetDim(0);
    }

    info.M = M;
    info.N = N;
    info.K = K;
    OP_LOGD(K_INNER_DEBUG, "M=%d", info.M);
    OP_LOGD(K_INNER_DEBUG, "K=%d", info.K);
    OP_LOGD(K_INNER_DEBUG, "N=%d", info.N);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus MatmulReduceScatterV2GetPlatformInfoAndSetTiling(gert::TilingContext *context, MatmulReduceScatterV2AivModeInfo& info)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSize = 0U;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    info.aivNum = aivNum;
    info.totalUbSize = ubSize;

    OP_LOGD(K_INNER_DEBUG, "aivNum=%d", info.aivNum);
    OP_LOGD(K_INNER_DEBUG, "ubSize=%d", info.totalUbSize);

    return ge::GRAPH_SUCCESS;
}

const std::map<ge::DataType, int64_t> D_TYPE_SIZE_MAP =
{
    {ge::DT_BF16, 2},
    {ge::DT_FLOAT16, 2},
    {ge::DT_FLOAT, 4},
    {ge::DT_INT8, 1},
};

static uint32_t AlignUp(uint32_t len, uint32_t size)
{
    return static_cast<uint32_t>((static_cast<int64_t>(len) + size - 1) & ~(size - 1));
}

static bool IsMatrixAligned(const uint32_t &m, const uint32_t &n, const bool &transpose, const uint32_t &nElemAlign)
{
    return (transpose ? m : n) % nElemAlign == 0;
}

static void GetTilingKey(uint64_t& tilingKey, MatmulReduceScatterV2AivModeInfo& info, gert::TilingContext* context)
{
    const gert::StorageShape *matrix_bias = context->GetOptionalInputShape(BIAS_INDEX);
    bool isBias = (matrix_bias == nullptr) ? false : true;
    tilingKey = GET_TPL_TILING_KEY(isBias, info.isTransposeA, info.isTransposeB);
    return;
}

int32_t GetValueFromMKNConditionMap(int32_t m, int32_t k, int32_t n, int32_t defaultValue, 
                                    std::map<int, std::vector<std::vector<int>>> conditionMap)
{
    int32_t value = defaultValue;
    for (auto &item : conditionMap) {
        for (auto &condition : item.second) {
            bool inRange =
                m > condition[CONDITION_M_ST] && m <= condition[CONDITION_M_END] &&
                k > condition[CONDITION_K_ST] && k <= condition[CONDITION_K_END] &&
                n > condition[CONDITION_N_ST] && n <= condition[CONDITION_N_END];
            if (inRange) {
                return item.first;
            }
        }
    }
    return value;
}

int32_t CeilDev(int32_t num, int32_t div)
{
    if (div == 0) {
        return 0;
    }
    return (num + div - 1) / div;
}

void CalTilingParam(CoCTiling &cocTilingData, const std::map<int*, MatmulReduceScatterV2AivModeTilingValue>& TilingParamMap, MatmulReduceScatterV2AivModeInfo &info)
{
    int32_t m = static_cast<int32_t>(info.M);
    int32_t k = static_cast<int32_t>(info.K);
    int32_t n = static_cast<int32_t>(info.N);

    for (auto &item : TilingParamMap) {
        auto value = item.second.value;
        auto conditionMap = item.second.conditionMap;
        if (!conditionMap.empty()) {
            *item.first = GetValueFromMKNConditionMap(m, k, n, value, conditionMap);
        } else if (value != -1) {
            *item.first = value;
        }
    }

    cocTilingData.ubMoveNum = cocTilingData.ubMoveNum * HALF_KBYTE;
    if (cocTilingData.m0 >= DEFAULT_ROW) {
        cocTilingData.k0 = DEFAULT_COL;
        cocTilingData.n0 = cocTilingData.m0 == DEFAULT_ROW ? DEFAULT_COL : DEFAULT_ROW;
    }
}

void ReduceScatterV2DecodeTilingData(int32_t code, CoCTiling &tilingData, MatmulReduceScatterV2AivModeInfo &info)
{
    int32_t m = static_cast<int32_t>(info.M);
    int32_t k = static_cast<int32_t>(info.K);
    int32_t n = static_cast<int32_t>(info.N);
    tilingData.commDataSplit = code & COMMDATASPLIT_MASK;
    code >>= COMMDATASPLIT_BNUM;
    tilingData.commNpuSplit = code & COMMNPUSPLIT_MASK;
    code >>= COMMNPUSPLIT_BNUM;
    tilingData.commDirect = code & COMMDIRECT_MASK;
    code >>= COMMDIRECT_BNUM;
    tilingData.ubMoveNum = (code & UBMOVENUM_MASK) * HALF_KBYTE;
    code >>= UBMOVENUM_BNUM;
    tilingData.pValue = code & PVALUE_MASK;
    code >>= PVALUE_BNUM;
    tilingData.swizzlCount = code & SWIZZLCOUNT_MASK;
    code >>= SWIZZLCOUNT_BNUM;
    tilingData.swizzlDirect = code & SWIZZLDIRECT_MASK;
    code >>= SWIZZLDIRECT_BNUM;
    tilingData.m0 = (code & M0_MASK) * DEFAULT_ROW + DEFAULT_ROW;
    tilingData.k0 = DEFAULT_COL;
    tilingData.n0 = tilingData.m0 == DEFAULT_ROW ? DEFAULT_COL : DEFAULT_ROW;
    tilingData.mLoop = CeilDev(m, tilingData.m0);
    tilingData.nLoop = CeilDev(n, tilingData.n0);
    tilingData.kLoop = CeilDev(k, tilingData.k0);
}

void MatmulReduceScatterA2FourRankINT8Tiling(CoCTiling &cocTilingData, MatmulReduceScatterV2AivModeInfo &info)
{
    int32_t code = MATMUL_REDUCESCATTER_A2_FOUR_RANK_INT8_CODE_DEFAULT;
    std::map<int*, MatmulReduceScatterV2AivModeTilingValue> TilingParamMap;
    TilingParamMap[&code] = MatmulReduceScatterV2AivModeTilingValue(
            MATMUL_REDUCESCATTER_A2_FOUR_RANK_INT8_CODE_DEFAULT,
            g_matmulReduceScatterA2FourRankINT8CodeMap);

    CalTilingParam(cocTilingData, TilingParamMap, info);

    ReduceScatterV2DecodeTilingData(code, cocTilingData, info);
    cocTilingData.lenPerLoop = cocTilingData.ubMoveNum;
}

void MatmulReduceScatterA2FourRankFP16Tiling(CoCTiling &cocTilingData, MatmulReduceScatterV2AivModeInfo &info)
{
    int32_t code = MATMUL_REDUCESCATTER_A2_FOUR_RANK_FP16_CODE_DEFAULT;
    std::map<int*, MatmulReduceScatterV2AivModeTilingValue> TilingParamMap;
    TilingParamMap[&code] = MatmulReduceScatterV2AivModeTilingValue(
            MATMUL_REDUCESCATTER_A2_FOUR_RANK_FP16_CODE_DEFAULT,
            g_matmulReduceScatterA2FourRankFP16CodeMap);

    CalTilingParam(cocTilingData, TilingParamMap, info);

    ReduceScatterV2DecodeTilingData(code, cocTilingData, info);
    cocTilingData.lenPerLoop = cocTilingData.ubMoveNum;
}

void MatmulReduceScatterA2EightRankINT8Tiling(CoCTiling &cocTilingData, MatmulReduceScatterV2AivModeInfo &info)
{
    int32_t code = MATMUL_REDUCESCATTER_A2_EIGHT_RANK_INT8_CODE_DEFAULT;
    std::map<int*, MatmulReduceScatterV2AivModeTilingValue> TilingParamMap;
    TilingParamMap[&code] = MatmulReduceScatterV2AivModeTilingValue(
            MATMUL_REDUCESCATTER_A2_EIGHT_RANK_INT8_CODE_DEFAULT,
            g_matmulReduceScatterA2EightRankINT8CodeMap);

    CalTilingParam(cocTilingData, TilingParamMap, info);

    ReduceScatterV2DecodeTilingData(code, cocTilingData, info);
    cocTilingData.lenPerLoop = cocTilingData.ubMoveNum;
}

void MatmulReduceScatterA2EightRankFP16Tiling(CoCTiling &cocTilingData, MatmulReduceScatterV2AivModeInfo &info)
{
    int32_t code = MATMUL_REDUCESCATTER_A2_EIGHT_RANK_FP16_CODE_DEFAULT;
    std::map<int*, MatmulReduceScatterV2AivModeTilingValue> TilingParamMap;
    TilingParamMap[&code] = MatmulReduceScatterV2AivModeTilingValue(
            MATMUL_REDUCESCATTER_A2_EIGHT_RANK_FP16_CODE_DEFAULT,
            g_matmulReduceScatterA2EightRankFP16CodeMap);

    CalTilingParam(cocTilingData, TilingParamMap, info);

    ReduceScatterV2DecodeTilingData(code, cocTilingData, info);
    cocTilingData.lenPerLoop = cocTilingData.ubMoveNum;
}

void MatmulReduceScatterA3EightRankINT8Tiling(CoCTiling &cocTilingData, MatmulReduceScatterV2AivModeInfo &info)
{
    int32_t code = MATMUL_REDUCESCATTER_A3_EIGHT_RANK_INT8_CODE_DEFAULT;
    std::map<int*, MatmulReduceScatterV2AivModeTilingValue> TilingParamMap;
    TilingParamMap[&code] = MatmulReduceScatterV2AivModeTilingValue(
            MATMUL_REDUCESCATTER_A3_EIGHT_RANK_INT8_CODE_DEFAULT,
            g_matmulReduceScatterA3EightRankINT8CodeMap);

    CalTilingParam(cocTilingData, TilingParamMap, info);

    ReduceScatterV2DecodeTilingData(code, cocTilingData, info);
    cocTilingData.lenPerLoop = cocTilingData.ubMoveNum;
}

void MatmulReduceScatterA3EightRankFP16Tiling(CoCTiling &cocTilingData, MatmulReduceScatterV2AivModeInfo &info)
{
    int32_t code = MATMUL_REDUCESCATTER_A3_EIGHT_RANK_FP16_CODE_DEFAULT;
    std::map<int*, MatmulReduceScatterV2AivModeTilingValue> TilingParamMap;
    TilingParamMap[&code] = MatmulReduceScatterV2AivModeTilingValue(
            MATMUL_REDUCESCATTER_A3_EIGHT_RANK_FP16_CODE_DEFAULT,
            g_matmulReduceScatterA3EightRankFP16CodeMap);

    CalTilingParam(cocTilingData, TilingParamMap, info);

    ReduceScatterV2DecodeTilingData(code, cocTilingData, info);
    cocTilingData.lenPerLoop = cocTilingData.ubMoveNum;
}

void MatmulReduceScatterA3FourRankINT8Tiling(CoCTiling &cocTilingData, MatmulReduceScatterV2AivModeInfo &info)
{
    int32_t code = MATMUL_REDUCESCATTER_A3_FOUR_RANK_INT8_CODE_DEFAULT;
    std::map<int*, MatmulReduceScatterV2AivModeTilingValue> TilingParamMap;
    TilingParamMap[&code] = MatmulReduceScatterV2AivModeTilingValue(
            MATMUL_REDUCESCATTER_A3_FOUR_RANK_INT8_CODE_DEFAULT,
            g_matmulReduceScatterA3FourRankINT8CodeMap);

    CalTilingParam(cocTilingData, TilingParamMap, info);

    ReduceScatterV2DecodeTilingData(code, cocTilingData, info);
    cocTilingData.lenPerLoop = cocTilingData.ubMoveNum;
}

void MatmulReduceScatterA3FourRankFP16Tiling(CoCTiling &cocTilingData, MatmulReduceScatterV2AivModeInfo &info)
{
    int32_t code = MATMUL_REDUCESCATTER_A3_FOUR_RANK_FP16_CODE_DEFAULT;
    std::map<int*, MatmulReduceScatterV2AivModeTilingValue> TilingParamMap;
    TilingParamMap[&code] = MatmulReduceScatterV2AivModeTilingValue(
            MATMUL_REDUCESCATTER_A3_FOUR_RANK_FP16_CODE_DEFAULT,
            g_matmulReduceScatterA3FourRankFP16CodeMap);

    CalTilingParam(cocTilingData, TilingParamMap, info);

    ReduceScatterV2DecodeTilingData(code, cocTilingData, info);
    cocTilingData.lenPerLoop = cocTilingData.ubMoveNum;
}

void SetTilingData(CoCTiling &cocTilingData, MatmulReduceScatterV2AivModeInfo &info, int64_t rankSize)
{
    if (info.is910C) {
        if (rankSize == RANKSIZE_FOUR && info.quantFlag) {
            MatmulReduceScatterA3FourRankINT8Tiling(cocTilingData, info);
            return;
        } else if (rankSize == RANKSIZE_FOUR && !info.quantFlag) {
            MatmulReduceScatterA3FourRankFP16Tiling(cocTilingData, info);
            return;
        } else if (rankSize == RANKSIZE_EIGHT && info.quantFlag) {
            MatmulReduceScatterA3EightRankINT8Tiling(cocTilingData, info);
            return;
        } else if (rankSize == RANKSIZE_EIGHT && !info.quantFlag) {
            MatmulReduceScatterA3EightRankFP16Tiling(cocTilingData, info);
            return;
        }
    } else {
        if (rankSize == RANKSIZE_FOUR && info.quantFlag) {
            MatmulReduceScatterA2FourRankINT8Tiling(cocTilingData, info);
            return;
        } else if (rankSize == RANKSIZE_FOUR && !info.quantFlag) {
            MatmulReduceScatterA2FourRankFP16Tiling(cocTilingData, info);
            return;
        } else if (rankSize == RANKSIZE_EIGHT && info.quantFlag) {
            MatmulReduceScatterA2EightRankINT8Tiling(cocTilingData, info);
            return;
        } else if (rankSize == RANKSIZE_EIGHT && !info.quantFlag) {
            MatmulReduceScatterA2EightRankFP16Tiling(cocTilingData, info);
            return;
        }
    }
    MatmulReduceScatterA2EightRankFP16Tiling(cocTilingData, info);
}

void GetUsrWorkSpaceSize(uint32_t elementSize, uint32_t blockDim, uint64_t &userWorkSpaceSize,
    MatmulReduceScatterV2AivModeInfo &info, CoCTiling &cocTilingData)
{
    constexpr int32_t TWO = 2;
    constexpr uint32_t NUMSIZE_ONE = 1;
    uint32_t nElemAlign = HALF_KBYTE / elementSize;
    bool hasAAlign = (!IsMatrixAligned(info.M, info.K, info.isTransposeA, nElemAlign) && info.M != NUMSIZE_ONE);
    bool hasBAlign = !IsMatrixAligned(info.K, info.N, info.isTransposeB, nElemAlign);
    uint32_t mAlign = AlignUp(info.M, nElemAlign);
    uint32_t kAlign = AlignUp(info.K, nElemAlign);
    uint32_t nAlign = AlignUp(info.N, nElemAlign);
    
    info.aAlignSize = 0;
    info.bAlignSize = 0;
    info.hasAAlign = hasAAlign;
    info.hasBAlign = hasBAlign;
    if (info.hasAAlign) {
        info.aAlignSize = static_cast<uint64_t>((info.isTransposeA ? info.K * mAlign : info.M * kAlign) * elementSize);
        userWorkSpaceSize += info.aAlignSize;
    }
    if (info.hasBAlign) {
        info.bAlignSize = static_cast<uint64_t>((info.isTransposeB ? info.N * kAlign : info.K * nAlign) * elementSize);
        userWorkSpaceSize += info.bAlignSize;
    }
    if (info.quantFlag) {
        userWorkSpaceSize += static_cast<uint64_t>(cocTilingData.pValue * blockDim * cocTilingData.m0 * cocTilingData.n0 * TWO * sizeof(int32_t)); //当前输出为BF16时，量化参数最大为32位
    }
}

static bool CheckDtype_X1(gert::TilingContext *context)
{
    const gert::Tensor* x1Scale = context->GetInputTensor(X1_SCALE_INDEX);
    if (x1Scale == nullptr) {
        return false;
    }
    auto x1Type = x1Scale->GetDataType();
    if (x1Type != ge::DT_FLOAT) {
        return false;
    }
    return true;
}

static bool CheckDtype_X2(gert::TilingContext *context, MatmulReduceScatterV2AivModeInfo &info, ge::DataType cType)
{
    const gert::Tensor* x2Scale = context->GetInputTensor(X2_SCALE_INDEX);
    if (x2Scale == nullptr) {
        return false;
    }
    auto x2ScaleType = x2Scale->GetDataType();
    info.isX2ScaleTypeInt64 = false;
    /* x2ScaleType支持float类型 */
    if (x2ScaleType == ge::DT_FLOAT) {
        return true;
    }
    /* 在输出类型为fp16时，x2ScaleType支持int64类型 */
    if (cType == ge::DT_FLOAT16 && x2ScaleType == ge::DT_INT64) {
        info.isX2ScaleTypeInt64 = true;
        return true;
    }
    return false;
}

static void PrintTilingDataInfo(MatmulReduceScatterV2AivModeInfo& info, CoCTiling& cocTilingInfo)
{
    OP_LOGD("MatmulReduceScatterV2AivModeTiling", " tiling.M %u", info.M);
    OP_LOGD("MatmulReduceScatterV2AivModeTiling", " tiling.k %u", info.K);
    OP_LOGD("MatmulReduceScatterV2AivModeTiling", " tiling.N %u", info.N);
    OP_LOGD("MatmulReduceScatterV2AivModeTiling", " tiling.aivNum %u", info.aivNum);
    OP_LOGD("MatmulReduceScatterV2AivModeTiling", " tiling.totalUbSize %u", info.totalUbSize);
    OP_LOGD("MatmulReduceScatterV2AivModeTiling", " tiling.isTransposeA %d", info.isTransposeA);
    OP_LOGD("MatmulReduceScatterV2AivModeTiling", " tiling.isTransposeB %d", info.isTransposeB);
    OP_LOGD("MatmulReduceScatterV2AivModeTiling", " tiling.aAlignSize %lu", info.aAlignSize);
    OP_LOGD("MatmulReduceScatterV2AivModeTiling", " tiling.bAlignSize %lu", info.bAlignSize);
    OP_LOGD("MatmulReduceScatterV2AivModeTiling", " tiling.quantFlag %d", info.quantFlag);
    OP_LOGD("MatmulReduceScatterV2AivModeTiling", " tiling.is910C %d", info.is910C);
    OP_LOGD("MatmulReduceScatterV2AivModeTiling", " tiling.isX2ScaleTypeInt64 %d", info.isX2ScaleTypeInt64);    

    OP_LOGD("MatmulReduceScatterV2AivModeTiling", " tiling.m0 %d", cocTilingInfo.m0); 
    OP_LOGD("MatmulReduceScatterV2AivModeTiling", " tiling.k0 %d", cocTilingInfo.k0); 
    OP_LOGD("MatmulReduceScatterV2AivModeTiling", " tiling.n0 %d", cocTilingInfo.n0); 
    OP_LOGD("MatmulReduceScatterV2AivModeTiling", " tiling.swizzlCount %d", cocTilingInfo.swizzlCount); 
    OP_LOGD("MatmulReduceScatterV2AivModeTiling", " tiling.swizzlDirect %d", cocTilingInfo.swizzlDirect); 
    OP_LOGD("MatmulReduceScatterV2AivModeTiling", " tiling.pValue %d", cocTilingInfo.pValue);
    OP_LOGD("MatmulReduceScatterV2AivModeTiling", " tiling.ubMoveNum %d", cocTilingInfo.ubMoveNum); 
    OP_LOGD("MatmulReduceScatterV2AivModeTiling", " tiling.commNpuSplit %d", cocTilingInfo.commNpuSplit); 
    OP_LOGD("MatmulReduceScatterV2AivModeTiling", " tiling.commDataSplit %d", cocTilingInfo.commDataSplit); 
    OP_LOGD("MatmulReduceScatterV2AivModeTiling", " tiling.lenPerLoop %d", cocTilingInfo.lenPerLoop); 
}

ge::graphStatus MatmulReduceScatterTilingV2AivModeFunc(gert::TilingContext *context)
{
    OP_LOGI("Enter MatmulReduceScatterV2 aivMode tiling func.");

    // 1. tilingData
    MatmulReduceScatterV2AivModeTilingData *tilingData = context->GetTilingData<MatmulReduceScatterV2AivModeTilingData>();
    OP_TILING_CHECK(tilingData == nullptr,
        VECTOR_INNER_ERR_REPORT_TILING(context->GetNodeName(), "MatmulReduceScatterV2 aivMode tilingData is nullptr."), return ge::GRAPH_FAILED);
    MatmulReduceScatterV2AivModeInfo& info = tilingData->matmulReduceScatterV2AivModeInfo;
    OP_TILING_CHECK(MatmulReduceScatterV2CheckAttrAndSetTiling(context, info) != ge::GRAPH_SUCCESS,
        VECTOR_INNER_ERR_REPORT_TILING(context->GetNodeName(), "MatmulReduceScatterV2 aivMode CheckAttrAndSetTiling Failed"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(MatmulReduceScatterV2CheckShapeAndSetTiling(context, info) != ge::GRAPH_SUCCESS,
        VECTOR_INNER_ERR_REPORT_TILING(context->GetNodeName(), "MatmulReduceScatterV2 aivMode CheckShapeAndSetTiling Failed"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(MatmulReduceScatterV2GetPlatformInfoAndSetTiling(context, info) != ge::GRAPH_SUCCESS,
        VECTOR_INNER_ERR_REPORT_TILING(context->GetNodeName(), "MatmulReduceScatterV2 aivMode GetPlatformInfoAndSetTiling Failed"),
        return ge::GRAPH_FAILED);
    auto attrs = context->GetAttrs();
    auto group = attrs->GetAttrPointer<char>(static_cast<int>(ATTR_GROUP_INDEX));
    const char* opName = context->GetNodeName();

    int64_t rankSize = 0;
    mc2tiling::GetRankSize(opName, group, rankSize);

    // 2. set blockDim
    uint32_t blockDim = 1U;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto aicNum = ascendcPlatform.GetCoreNumAic();
    auto aivNum = ascendcPlatform.GetCoreNumAiv();
    blockDim = ascendcPlatform.CalcTschBlockDim(aivNum, aicNum, aivNum);
    context->SetBlockDim(blockDim);

    // 3. set tilingKey
    uint64_t tilingKey = 0;
    GetTilingKey(tilingKey, info, context);
    context->SetTilingKey(tilingKey);

    // 4. workspace
    size_t *workSpaces = context->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workSpaces == nullptr, VECTOR_INNER_ERR_REPORT_TILING(context->GetNodeName(), "MatmulReduceScatterV2 aivMode workSpaces is nullptr."),
        return ge::GRAPH_FAILED);
    auto aType = context->GetInputTensor(A_INDEX)->GetDataType();
    auto bType = context->GetInputTensor(B_INDEX)->GetDataType();
    auto cType = context->GetOutputDesc(0)->GetDataType();
    info.quantFlag = (aType == ge::DT_INT8) && (bType == ge::DT_INT8) && (cType == ge::DT_BF16 || cType == ge::DT_FLOAT16);
    if (info.quantFlag) {
        OP_TILING_CHECK(!CheckDtype_X2(context, info, cType), VECTOR_INNER_ERR_REPORT_TILING(context->GetNodeName(), "MatmulReduceScatterV2 aivMode Invalid x2Scale."),
        return ge::GRAPH_FAILED);
        info.dequant_type = DequantType::PER_CHANNEL;
        if (CheckDtype_X1(context)) {
            info.dequant_type = DequantType::PER_TOKEN;
        }
    }

    info.is910C = false;
    fe::PlatFormInfos *platformInfoPtr = context->GetPlatformInfo();
    fe::PlatFormInfos &platformInfo = *platformInfoPtr;

    std::string socVersion;
    (void)platformInfo.GetPlatformResWithLock("version", "Short_SoC_version", socVersion);
    if (socVersion == "Ascend910_93") {
        info.is910C = true;
    }

    // Tiling
    SetTilingData(tilingData->cocTiling, info, rankSize);

    uint32_t elementSize = D_TYPE_SIZE_MAP.at(aType);
    uint64_t userWorkSpaceSize = 0;
    GetUsrWorkSpaceSize(elementSize, blockDim, userWorkSpaceSize, info, tilingData->cocTiling);
    workSpaces[0] = SYSTEM_NEED_WORKSPACE + userWorkSpaceSize;

    PrintTilingDataInfo(info, tilingData->cocTiling);

    // 5. communication
    if (info.is910C){
        uint32_t opType = OP_TYPE_REDUCE_SCATTER;
        std::string algConfig = "ReduceScatter=level0:fullmesh";
        AscendC::Mc2CcTilingConfig mc2CcTilingConfig(group, opType, algConfig);
        mc2CcTilingConfig.GetTiling(tilingData->mc2InitTiling);
        mc2CcTilingConfig.GetTiling(tilingData->mc2CcTiling);
    } else {
        uint32_t opType = 18;
        std::string algConfig = "MultiPut=level0:fullmesh";
        AscendC::Mc2CcTilingConfig mc2CcTilingConfig(group, opType, algConfig);
        mc2CcTilingConfig.GetTiling(tilingData->mc2InitTiling);
        mc2CcTilingConfig.GetTiling(tilingData->mc2CcTiling);
    }

    OP_LOGI("Leave MatmulReduceScatterV2AivMode tiling func.");
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling