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
 * \file all_gather_matmul_inner_tiling.cpp
 * \brief
 */
#include "vector"
#include "mc2_log.h"
#include "ops_utils.h"
#include "graph/utils/type_utils.h"
#include "register/op_def_registry.h"

#include "../../op_kernel/all_gather_matmul_inner_tiling.h"

#define INFO_LOG(fmt, args...) fprintf(stdout, "[INFO]  " fmt "\n", ##args)
#define WARN_LOG(fmt, args...) fprintf(stdout, "[WARN]  " fmt "\n", ##args)
#define ERROR_LOG(fmt, args...) fprintf(stderr, "[ERROR]  " fmt "\n", ##args)

// tiling
constexpr uint32_t WORKSPACE_MSG_SIZE = 16 * 1024 * 1024; // 16M for communication messages
constexpr uint32_t CUSTOM_TILING_KEY_100UL = 100UL; // strideCount zero
constexpr uint32_t CUSTOM_TILING_KEY_101UL = 101UL; // strideCount not zero

namespace optiling {
namespace {
    enum class HcclDataType {
        HCCL_DATA_TYPE_INT8 = 0,   /* *< int8 */
        HCCL_DATA_TYPE_INT16 = 1,  /* *< int16 */
        HCCL_DATA_TYPE_INT32 = 2,  /* *< int32 */
        HCCL_DATA_TYPE_FP16 = 3,   /* *< fp16 */
        HCCL_DATA_TYPE_FP32 = 4,   /* *< fp32 */
        HCCL_DATA_TYPE_INT64 = 5,  /* *< int64 */
        HCCL_DATA_TYPE_UINT64 = 6, /* *< uint64 */
        HCCL_DATA_TYPE_UINT8 = 7,  /* *< uint8 */
        HCCL_DATA_TYPE_UINT16 = 8, /* *< uint16 */
        HCCL_DATA_TYPE_UINT32 = 9, /* *< uint32 */
        HCCL_DATA_TYPE_FP64 = 10,  /* *< fp64 */
        HCCL_DATA_TYPE_BFP16 = 11, /* *< bfp16 */
        HCCL_DATA_TYPE_INT128 = 12, /* *< int128 */
        HCCL_DATA_TYPE_HIF8 = 14,  /* *< hif8 */
        HCCL_DATA_TYPE_FP8E4M3 = 15,  /* *< fp8e4m3 */
        HCCL_DATA_TYPE_FP8E5M2 = 16,  /* *< fp8e5m2 */
        HCCL_DATA_TYPE_FP8E8M0 = 17,  /* *< fp8e8m0 */
        HCCL_DATA_TYPE_RESERVED    /* *< reserved */
    };

    const std::map <ge::DataType, HcclDataType> HCCL_DATA_TYPE_MAP = {
            {ge::DataType::DT_INT8, HcclDataType::HCCL_DATA_TYPE_INT8},
            {ge::DataType::DT_INT16, HcclDataType::HCCL_DATA_TYPE_INT16},
            {ge::DataType::DT_INT32, HcclDataType::HCCL_DATA_TYPE_INT32},
            {ge::DataType::DT_FLOAT, HcclDataType::HCCL_DATA_TYPE_FP32},
            {ge::DataType::DT_INT64, HcclDataType::HCCL_DATA_TYPE_INT64},
            {ge::DataType::DT_UINT64, HcclDataType::HCCL_DATA_TYPE_UINT64},
            {ge::DataType::DT_UINT8, HcclDataType::HCCL_DATA_TYPE_UINT8},
            {ge::DataType::DT_UINT16, HcclDataType::HCCL_DATA_TYPE_UINT16},
            {ge::DataType::DT_UINT32, HcclDataType::HCCL_DATA_TYPE_UINT32},
            {ge::DataType::DT_DOUBLE, HcclDataType::HCCL_DATA_TYPE_FP64},
            {ge::DataType::DT_FLOAT16, HcclDataType::HCCL_DATA_TYPE_FP16},
            {ge::DataType::DT_BF16, HcclDataType::HCCL_DATA_TYPE_BFP16},
            {ge::DataType::DT_FLOAT8_E8M0, HcclDataType::HCCL_DATA_TYPE_FP8E8M0}

    };

    const std::map <ge::DataType, int> DATA_TYPE_SIZE_MAP = {
            {ge::DataType::DT_INT8, 1},
            {ge::DataType::DT_INT16, 2},
            {ge::DataType::DT_INT32, 4},
            {ge::DataType::DT_FLOAT, 4},
            {ge::DataType::DT_INT64, 8},
            {ge::DataType::DT_UINT64, 8},
            {ge::DataType::DT_UINT8, 1},
            {ge::DataType::DT_UINT16, 2},
            {ge::DataType::DT_UINT32, 4},
            {ge::DataType::DT_DOUBLE, 8},
            {ge::DataType::DT_FLOAT16, 2},
            {ge::DataType::DT_BF16, 2},
            {ge::DataType::DT_FLOAT8_E8M0, 1}
    };

    const std::set<ge::Format> SUPPORTED_FORMAT = {ge::FORMAT_NCL, ge::FORMAT_NCDHW, ge::FORMAT_DHWCN, ge::FORMAT_NHWC,
                                                   ge::FORMAT_NCHW,  ge::FORMAT_ND};
}


static ge::graphStatus ParamsCheck(gert::TilingContext* context)
{
    // 2维校验
    const gert::StorageShape* aShape = context->GetInputShape(0);
    if (aShape->GetStorageShape().GetDimNum() != 2) {
        ERROR_LOG("Input dim num must be 2.");
        return ge::GRAPH_FAILED;
    }
    // 空tensor校验
    if (aShape->GetStorageShape().GetDim(0) == 0 || aShape->GetStorageShape().GetDim(1) == 0) {
        ERROR_LOG("Input tensor shape is 0");
        return ge::GRAPH_FAILED;
    }
    // format校验
    auto aTensor = context->GetInputDesc(0);
    auto output = context->GetOutputDesc(0);
    auto aShapeFormat = aTensor->GetStorageFormat();
    auto outputFormat = output->GetStorageFormat();
    if (aShapeFormat != outputFormat) {
        ERROR_LOG("X1 format and output format must be same.");
        return ge::GRAPH_FAILED;
    }
    if ((SUPPORTED_FORMAT.count(aShapeFormat) == 0)) {
        ERROR_LOG("X1 format only support NCL, NCDHW, DHWCN, NHWC, NCHW, ND");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus AllGatherMatmulTilingFunc(gert::TilingContext *context) {
  // 对参数进行校验
    if (ParamsCheck(context) != ge::GRAPH_SUCCESS) {
        ERROR_LOG("Param check failed");
        return ge::GRAPH_FAILED;
    }
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto aivCoreNum = ascendcPlatform.GetCoreNumAiv();
    // get attrs
    uint32_t index = 0U;
    const char *group = context->GetAttrs()->GetAttrPointer<char>(index++);
    int rankSize = *(context->GetAttrs()->GetAttrPointer<int>(index++));
    uint64_t tileM = static_cast<uint64_t>(static_cast<uint32_t>(*(context->GetAttrs()->GetAttrPointer<int>(index++))));
    uint64_t strideCount = static_cast<uint64_t>(static_cast<uint32_t>(*(context->GetAttrs()->GetAttrPointer<int>(index++))));
    int commTurn = *(context->GetAttrs()->GetAttrPointer<int>(index++));

    // get shape
    uint64_t M = context->GetInputShape(0)->GetStorageShape().GetDim(0);
    uint64_t K = context->GetInputShape(0)->GetStorageShape().GetDim(1);
    // get dtype
    auto aType = context->GetInputDesc(0)->GetDataType();
    auto bType = context->GetOutputDesc(0)->GetDataType();
    // set block dim & tiling key
    context->SetBlockDim(aivCoreNum);
    uint32_t tiling_key = (strideCount == 0) ? CUSTOM_TILING_KEY_100UL : CUSTOM_TILING_KEY_101UL;
    context->SetTilingKey(tiling_key);
    // set work space size
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = WORKSPACE_MSG_SIZE;

    uint8_t tileNum = M / tileM;
    uint8_t tailNum = (M % tileM == 0) ? 0 : 1;
    uint64_t tailM = M % tileM;

    AllGatherCustomV3TilingData *tiling = context->GetTilingData<AllGatherCustomV3TilingData>();

    tiling->param.rankDim = rankSize;
    tiling->param.M = M;
    tiling->param.K = K;
    tiling->param.tileM = tileM;
    tiling->param.tileNum = tileNum;
    tiling->param.tailM = tailM;
    tiling->param.tailNum = tailNum; 
    tiling->param.strideCount = strideCount;
    tiling->param.dataType = static_cast<uint8_t>(HCCL_DATA_TYPE_MAP.at(aType));
    tiling->param.dataTypeSize = DATA_TYPE_SIZE_MAP.at(aType);
    INFO_LOG("M is %llu, K is %llu, tileM is %llu, tileNum is %u, tailM is %llu, tailNum is %u, strideCount is %llu",
             M, K, tileM, tileNum, tailM, tailNum, strideCount);
    INFO_LOG("init aType is %u, bType is %u", static_cast<uint8_t>(HCCL_DATA_TYPE_MAP.at(aType)), static_cast<uint8_t>(HCCL_DATA_TYPE_MAP.at(bType)));

    return ge::GRAPH_SUCCESS;
}

struct AllGatherMatmulCompileInfo {};
static ge::graphStatus TilingParseForAllGatherMatmul([[maybe_unused]] gert::TilingParseContext *context) { return ge::GRAPH_SUCCESS; }

IMPL_OP_OPTILING(AllGatherMatmulInner)
    .Tiling(AllGatherMatmulTilingFunc)
    .TilingParse<AllGatherMatmulCompileInfo>(TilingParseForAllGatherMatmul);
}  // namespace optiling

