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
 * \file qbmm_reduce_scatter_add_rms_norm_cast_tiling.cpp
 * \brief host侧tiling实现
 */

#include <register/op_def_registry.h>
#include "tiling/mc2_tiling_utils.h"
#include "util/math_util.h"
#include "../../op_kernel/qbmm_reduce_scatter_add_rms_norm_cast_tiling_key.h"
#include "../../op_kernel/qbmm_reduce_scatter_add_rms_norm_cast_tiling_data.h"
using namespace AscendC;
using namespace ge;
namespace MC2Tiling {
constexpr size_t GROUP_INDEX = 0;
constexpr uint32_t OP_TYPE_ALL_TO_ALL = 8;
constexpr uint32_t AIV_TYPE = 2;

using namespace AscendC;
using namespace ge;

/**
 * @brief 打印tilingData
 * @param context: 框架根据input，output，attrs等信息生成tiling需要的context
 * @param tilingData: 框架根据context的opName匹配tiling模板，计算产生的tilingData
 * @return
 */
static void PrintTilingDataInfo(gert::TilingContext *context, QbmmReduceScatterAddRmsNormCastTilingData &tilingData)
{

}

/**
 * @brief 设置hcomm参数
 * @param context: 框架根据input，output，attrs等信息生成tiling需要的context
 * @param tilingData: 框架根据context的opName匹配tiling模板，计算产生的tilingData
 * @return
 */
static ge::graphStatus SetHcommCfg(const gert::TilingContext *context, QbmmReduceScatterAddRmsNormCastTilingData *tilingData,
                        const std::string &group)
{
    const char *nodeName = context->GetNodeName();
    OP_LOGD(nodeName, "group is %s in qbmm_reduce_scatter_add_rms_norm_cast.", group.c_str());
    // TODO:待完善
    AscendC::Mc2CcTilingConfig mc2CcTilingConfig(group, OP_TYPE_ALL_TO_ALL,
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
static void SetTilingData(gert::TilingContext *context, QbmmReduceScatterAddRmsNormCastTilingData &tilingData)
{
    fe::PlatFormInfos *platformInfoPtr = context->GetPlatformInfo();
    platform_ascendc::PlatformAscendC ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    // set corenum
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    context->SetBlockDim(ascendcPlatform.CalcTschBlockDim(aivNum, 0, aivNum));
    tilingData.qbmmReduceScatterAddRmsNormCastTilingInfo.aivNum = aivNum;
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
    const uint64_t tilingKey = GET_TPL_TILING_KEY(MTE_COMM);
    context->SetTilingKey(tilingKey);
    OP_LOGD(nodeName, "tilingKey is [%lu] in qbmm_reduce_scatter_add_rms_norm_cast.", tilingKey);
}

/**
 * @brief qbmm_reduce_scatter_add_rms_norm_cast算子的tiling函数
 * @param context: 框架根据input，output，attrs等信息生成tiling需要的context
 * @return
 */
static ge::graphStatus QbmmReduceScatterAddRmsNormCastTilingFunc(gert::TilingContext *context)
{
    OP_TILING_CHECK(context == nullptr,
                    OP_LOGE("qbmm_reduce_scatter_add_rms_norm_cast", "failed to get tiling context in qbmm_reduce_scatter_add_rms_norm_cast."),
                    return ge::GRAPH_FAILED);
    const char *nodeName = context->GetNodeName();
    OP_TILING_CHECK(nodeName == nullptr,
                    OP_LOGE("qbmm_reduce_scatter_add_rms_norm_cast", "failed to get nodeName in qbmm_reduce_scatter_add_rms_norm_cast."),
                    return ge::GRAPH_FAILED);

    QbmmReduceScatterAddRmsNormCastTilingData *tilingData = context->GetTilingData<QbmmReduceScatterAddRmsNormCastTilingData>();
    OP_TILING_CHECK(tilingData == nullptr, OP_LOGE(nodeName, "tilingData is nullptr in qbmm_reduce_scatter_add_rms_norm_cast."),
                    return ge::GRAPH_FAILED);
    // 获取group以便ranksize设置与通信设置
    std::string group = "";
    const char *groupPtr = context->GetAttrs()->GetAttrPointer<char>(GROUP_INDEX);
    group = std::string(groupPtr);
    OP_TILING_CHECK(SetHcommCfg(context, tilingData, group) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "SetHCommCfg failed."), return ge::GRAPH_FAILED);
    SetTilingData(context, *tilingData);
    SetTilingKey(context);
    PrintTilingDataInfo(context, *tilingData);
    return ge::GRAPH_SUCCESS;
}

struct QbmmReduceScatterAddRmsNormCastCompileInfo {};
 
ge::graphStatus TilingParseForQbmmReduceScatterAddRmsNormCast(gert::TilingParseContext *context) {
    (void)context;
  	return ge::GRAPH_SUCCESS;
}
 
IMPL_OP_OPTILING(QbmmReduceScatterAddRmsNormCast)
    .Tiling(QbmmReduceScatterAddRmsNormCastTilingFunc)
    .TilingParse<QbmmReduceScatterAddRmsNormCastCompileInfo>(TilingParseForQbmmReduceScatterAddRmsNormCast);

} // namespace MC2Tiling
