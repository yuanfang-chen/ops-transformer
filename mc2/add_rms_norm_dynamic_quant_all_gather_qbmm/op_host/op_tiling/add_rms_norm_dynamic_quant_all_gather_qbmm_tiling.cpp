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
 * \file add_rms_norm_dynamic_quant_all_gather_qbmm_tiling.cpp
 * \brief host侧tiling实现
 */

#include <register/op_def_registry.h>
// #include "../../op_kernel/add_rms_norm_dynamic_quant_all_gather_qbmm_tiling_key.h"
#include "../../op_kernel/add_rms_norm_dynamic_quant_all_gather_qbmm_tiling_data.h"
#include "add_rms_norm_dynamic_quant_all_gather_qbmm_tiling_helper.h"
#include "add_rms_norm_dynamic_quant_v2_tiling.h"
#include "mc2_log.h"

namespace MC2Tiling {

using namespace AscendC;
using namespace ge;

constexpr uint32_t AIV_TYPE = 3;
constexpr uint32_t OP_TYPE_ALL_TO_ALL = 8;

/**
 * @brief 打印tilingData, addrms  and mamtul tcubetiling
 * @param context: 框架根据input，output，attrs等信息生成tiling需要的context
 * @param tilingData: 框架根据context的opName匹配tiling模板，计算产生的tilingData
 * @return
 */
static void PrintTilingDataInfo(gert::TilingContext *context, AddRmsNormDynamicQuantAllGatherQbmmTilingData &tilingData)
{
    OP_LOGD("AddRmsNormDynamicQuantAllGatherQbmm", "Tiling end");
}

// addrmsNormdynamicquantallgatherqbmm tiling
static ge::graphStatus GetAddRmsNormDynamicQuantAllGatherQbmm(
    gert::TilingContext *context, AddRmsNormDynamicQuantAllGatherQbmmTilingData &tilingData)
{   
    AddRmsNormDynamicQuantV2TilingHelper instanceNormV3TilingHelper(context);
    bool status = instanceNormV3TilingHelper.DoTiling();
    OP_CHECK_IF(!status, OP_LOGE(context, "DoTiling Failed, return Failed."), return ge::GRAPH_FAILED);

    instanceNormV3TilingHelper.SetTilingData(&tilingData.addRmsNormDynamicQuantAllGatherTilingData);
    return ge::GRAPH_SUCCESS;
}

// matmul切分
static ge::graphStatus GetMatmultiling(
    gert::TilingContext *context, AddRmsNormDynamicQuantAllGatherQbmmTilingData &tilingData)
{
    MmTilingHelper mmTilingHelper(context);
    OP_CHECK_IF(!mmTilingHelper.getMamtulArgs(),
        OP_LOGE(context, "MmTilingHelper get matmulArgs failed, return Failed."),
        return ge::GRAPH_FAILED);
    mmTilingHelper.InitCompileInfo();
    OP_CHECK_IF(!mmTilingHelper.InitTCubeTilingData(tilingData.matmulTiling),
        OP_LOGE(context, "MmTilingHelper InitTCubeTilingData failed, return Failed."),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 设置hcomm参数
 * @param context: 框架根据input，output，attrs等信息生成tiling需要的context
 * @param tilingData: 框架根据context的opName匹配tiling模板，计算产生的tilingData
 * @param runInfo: 封装的doTiling所需要的参数
 * @return
 */
static ge::graphStatus SetHcommCfg(const gert::TilingContext *context,
    AddRmsNormDynamicQuantAllGatherQbmmTilingData *tilingData, const TilingRunInfo &runInfo)
{
    const char *nodeName = context->GetNodeName();
    OP_LOGD(nodeName, "group is %s in add_rms_norm_dynamic_quant_all_gather_qbmm.", runInfo.group.c_str());
    AscendC::Mc2CcTilingConfig mc2CcTilingConfig(runInfo.group, OP_TYPE_ALL_TO_ALL,
                                                 "AlltoAll=level0:fullmesh;level1:pairwise");
    // MTE方式必要适配
    mc2CcTilingConfig.SetCommEngine(AIV_TYPE);
    OP_TILING_CHECK(mc2CcTilingConfig.GetTiling(tilingData->mc2InitTiling) != 0,
        OP_LOGE(nodeName, "mc2CcTilingConfig mc2InitTiling GetTiling failed"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(mc2CcTilingConfig.GetTiling(tilingData->mc2CcTiling) != 0,
        OP_LOGE(nodeName, "mc2CcTilingConfig mc2CcTiling GetTiling failed"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 设置tilingData
 * @param context: 框架根据input，output，attrs等信息生成tiling需要的context
 * @param tilingData: 框架根据context的opName匹配tiling模板，计算产生的tilingData
 * @return
 */
//  需要修改核数设置
static void SetTilingData(gert::TilingContext *context, AddRmsNormDynamicQuantAllGatherQbmmTilingData &tilingData)
{
    fe::PlatFormInfos *platformInfoPtr = context->GetPlatformInfo();
    platform_ascendc::PlatformAscendC ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    // set corenum
    uint32_t aivNum = 24;
    context->SetBlockDim(ascendcPlatform.CalcTschBlockDim(aivNum, aivNum, aivNum));
    tilingData.addRmsNormDynamicQuantAllGatherTilingData.aivNum = aivNum;
}

/**
 * @brief 设置tilingKey, 先使用老的方式生成tilingkey
 * @param context: 框架根据input，output，attrs等信息生成tiling需要的context
 * @return
 */
static void SetTilingKey(gert::TilingContext *context)
{
    const char *nodeName = context->GetNodeName();
    // 设置tilingKey模板参数
    // const uint64_t tilingKey = GET_TPL_TILING_KEY(MTE_COMM);
    context->SetTilingKey(1);
    // OP_LOGD(nodeName, "tilingKey is [%lu] in add_rms_norm_dynamic_quant_all_gather_qbmm.", tilingKey);
}

/**
 * @brief add_rms_norm_dynamic_quant_all_gather_qbmm算子的tiling函数
 * @param context: 框架根据input，output，attrs等信息生成tiling需要的context
 * @return
 */
static ge::graphStatus AddRmsNormDynamicQuantAllGatherQbmmTilingFunc(gert::TilingContext *context)
{
    OP_LOGD("AddRmsNormDynamicQuantAllGatherQbmm", "Tiling start");
    OP_TILING_CHECK(context == nullptr,
                    OP_LOGE("add_rms_norm_dynamic_quant_all_gather_qbmm",
                        "failed to get tiling context in add_rms_norm_dynamic_quant_all_gather_qbmm."),
                    return ge::GRAPH_FAILED);
    const char *nodeName = context->GetNodeName();
    OP_TILING_CHECK(nodeName == nullptr,
                    OP_LOGE("add_rms_norm_dynamic_quant_all_gather_qbmm",
                        "failed to get nodeName in add_rms_norm_dynamic_quant_all_gather_qbmm."),
                    return ge::GRAPH_FAILED);

    AddRmsNormDynamicQuantAllGatherQbmmTilingData *tilingData \
        = context->GetTilingData<AddRmsNormDynamicQuantAllGatherQbmmTilingData>();
    OP_TILING_CHECK(tilingData == nullptr,
        OP_LOGE(nodeName, "tilingData is nullptr in add_rms_norm_dynamic_quant_all_gather_qbmm."),
        return ge::GRAPH_FAILED);

    // addrmsnormdynamicquantallgatherqbmm 部分tiling切分
    // OP_TILING_CHECK(GetAddRmsNormDynamicQuantAllGatherQbmm(context, *tilingData) != ge::GRAPH_SUCCESS,
    //     OP_LOGE(nodeName, "GetAddRmsNormDynamicQuantAllGatherQbmm failed."),
    //     return ge::GRAPH_FAILED);
    // 先切k= 2560，直接调用matmul做tiling切分
    // OP_TILING_CHECK(GetMatmultiling(context, *tilingData) != ge::GRAPH_SUCCESS,
    //     OP_LOGE(nodeName, "GetMatmultiling failed."),
    //     return ge::GRAPH_FAILED);
    TilingRunInfo runInfo = {};
    OP_TILING_CHECK(SetHcommCfg(context, tilingData, runInfo) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "SetHCommCfg failed."), return ge::GRAPH_FAILED);
    SetTilingData(context, *tilingData);
    SetTilingKey(context);
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 16 * 1024 * 1024;

    PrintTilingDataInfo(context, *tilingData);
    return ge::GRAPH_SUCCESS;
}

struct AddRmsNormDynamicQuantAllGatherQbmmCompileInfo {};

ge::graphStatus TilingParseForAddRmsNormDynamicQuantAllGatherQbmm(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}
 
IMPL_OP_OPTILING(AddRmsNormDynamicQuantAllGatherQbmm)
    .Tiling(AddRmsNormDynamicQuantAllGatherQbmmTilingFunc)
    .TilingParse<AddRmsNormDynamicQuantAllGatherQbmmCompileInfo>(TilingParseForAddRmsNormDynamicQuantAllGatherQbmm);

} // namespace MC2Tiling
