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
#include "qbmm_reduce_scatter_add_rms_norm_cast_tiling_check.h"
// #include "../../op_kernel/qbmm_reduce_scatter_add_rms_norm_cast_tiling_key.h"
#include "../../op_kernel/qbmm_reduce_scatter_add_rms_norm_cast_tiling_data.h"
using namespace AscendC;
using namespace ge;
namespace MC2Tiling {

using AscendC::BLOCK_CUBE;    // uint32_t 16
using AscendC::ONE_BLK_SIZE;  // uint32_t 32

constexpr size_t GROUP_INDEX = 0;
constexpr size_t RANK_SIZE_INDEX = 1;
constexpr uint32_t OP_TYPE_ALL_TO_ALL = 8;
constexpr uint32_t AIV_TYPE = 3;
constexpr uint32_t SYSTEM_NEED_WORKSPACE = 16U * 1024 * 1024;
constexpr uint64_t UB_EXTRE_BYTE = 8;

using namespace AscendC;
using namespace ge;

// /**
//  * @brief 打印tilingData
//  * @param context: 框架根据input，output，attrs等信息生成tiling需要的context
//  * @param tilingData: 框架根据context的opName匹配tiling模板，计算产生的tilingData
//  * @return
//  */
// static void PrintTilingDataInfo(gert::TilingContext *context, QbmmReduceScatterAddRmsNormCastTilingData &tilingData)
// {

// }

static ge::graphStatus CheckSocVersion(const gert::TilingContext *context)
{
    const char *nodeName = context->GetNodeName();
    // 校验socVersion
    fe::PlatFormInfos *platformInfoPtr = context->GetPlatformInfo();
    OP_TILING_CHECK(platformInfoPtr == nullptr, OP_LOGE(nodeName, "platformInfoPtr is null."), return ge::GRAPH_FAILED);
    platform_ascendc::PlatformAscendC ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    platform_ascendc::SocVersion socVersion = ascendcPlatform.GetSocVersion();
    OP_TILING_CHECK(socVersion != platform_ascendc::SocVersion::ASCEND910_93,
        OP_LOGE(nodeName, "SocVersion needed to be 910_93."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
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
    AscendC::Mc2CcTilingConfig mc2CcTilingConfig(group, OP_TYPE_ALL_TO_ALL,
                                                 "AlltoAll=level0:fullmesh;level1:pairwise");
    // MTE方式必要适配
    mc2CcTilingConfig.SetCommEngine(AIV_TYPE);
    OP_TILING_CHECK(mc2CcTilingConfig.GetTiling(tilingData->mc2InitTiling) != 0,
        OP_LOGE(nodeName, "mc2CcTilingConfig mc2InitTiling GetTiling failed"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(mc2CcTilingConfig.GetTiling(tilingData->mc2CcTiling) != 0,
        OP_LOGE(nodeName, "mc2CcTilingConfig mc2CcTiling GetTiling failed"), return ge::GRAPH_FAILED);
    OP_LOGD("QbmmReduceScatterAddRmsNormCast set HCCL success.");
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
    context->SetTilingKey(0);
}

static ge::graphStatus SetWorkSpace(gert::TilingContext *context)
{
    const char *nodeName = context->GetNodeName();
    size_t *workSpaces = context->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workSpaces == nullptr, OP_LOGE(nodeName, "workSpaces is nullptr."), return ge::GRAPH_FAILED);
    workSpaces[0] = SYSTEM_NEED_WORKSPACE;
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief qbmm_reduce_scatter_add_rms_norm_cast算子的tiling函数
 * @param context: 框架根据input，output，attrs等信息生成tiling需要的context
 * @return
 */
static ge::graphStatus QbmmReduceScatterAddRmsNormCastTilingFunc(gert::TilingContext *context)
{
    // 1. tiling参数校验
    OP_LOGD("QbmmReduceScatterAddRmsNormCast tiling start.");
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

    // 校验socVersion与Attr属性
    OP_TILING_CHECK(CheckSocVersion(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "socVersion is invalid."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(TilingCheckQbmmReduceScatterAddRmsNormCast::CheckAttrs(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "Attrs are invalied."), return ge::GRAPH_FAILED);

    // 校验输入输出tensor的dim/dtype/format
    OP_TILING_CHECK(TilingCheckQbmmReduceScatterAddRmsNormCast::TilingCheckQbmmReduceScatterAddRmsNormCast(context, params) !=
        ge::GRAPH_SUCCESS, OP_LOGE(nodeName, "Tiling check param failed."), return ge::GRAPH_FAILED);
    
    // 2. tiling参数设置
    // 获取group以便ranksize设置与通信设置
    std::string group = "";
    const char *groupPtr = context->GetAttrs()->GetAttrPointer<char>(GROUP_INDEX);
    tilingData->tpWorldSize = context->GetAttrs()->GetAttr<int32_t>(RANK_SIZE_INDEX);
    group = std::string(groupPtr);
    SetTCubeTiling(context, tilingData);
    OP_TILING_CHECK(SetHcommCfg(context, tilingData, group) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "SetHCommCfg failed."), return ge::GRAPH_FAILED);
    SetWorkSpace(context);
    SetTilingData(context, *tilingData);
    SetTilingKey(context);
    // PrintTilingDataInfo(context, *tilingData);
    OP_LOGD("QbmmReduceScatterAddRmsNormCast tiling end.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QbmmReduceScatterAddRmsNormCastCheckTiling::SetTCubeTiling(const gert::TilingContext *context, QbmmReduceScatterAddRmsNormCastTilingData *tilingData)
{

    matmul_tiling::PlatformInfo platformInfo;
    InitPlatformInfo(&compileInfo_, platformInfo);
    matmul_tiling::MultiCoreMatmulTiling mm(platformInfo);
    mm.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_INT8, false);
    mm.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::NZ, matmul_tiling::DataType::DT_INT8, false);
    mm.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_BF16);
    mm.SetBias(false);
    int32_t baseM = tilingData->M / tilingData->tpWorldSize * 2; // 63 * 2
    int32_t baseN = 128; // 128
    int32_t baseK = 256; // int8 可以baseM * baseK < 64K
    mm.SetOrgShape(tilingData.M, tilingData.N, tilingData.K);
    mm.SetShape(baseM, baseN, inputParams_.kSize);
    mm.SetFixSplit(baseM, baseN, baseK);
    if (mm.GetTiling(tilingData->matmulTiling) ==-1) {
        return false;
    }
}

ge::graphStatus QuantBatchMatmulV3Tiling::CalcUbTiling(uint32_t baseN, uint32_t baseM, QbmmReduceScatterAddRmsNormCastTilingData *tilingData)
{
    uint64_t ubSize = aicoreParams_.ubSize;
    uint64_t needUbSize = 0;
    uint32_t ubCalcN = baseN;
    // src(int32) + scale(fp32/bf16) + pertoken(fp32) + out(fp16/bf16) + veccalc, in and out need double buffer
    // int16_t reprersent bf16, input src + output dst + veccalc dequant api
    uint64_t ubCalc = (NUM_DB * (sizeof(int32_t) + sizeof(int16_t)) + UB_EXTRE_BYTE) * ubCalcN;
    // input: scale perchannel
    ubCalc += NUM_DB * ge::GetSizeByDataType(inputParams_.scaleDtype)* ubCalcN;
    // veccalc: dequant api dst fp32
    ubCalc += sizeof(float) * ubCalcN;
    // veccalc: BroadCast需要的临时空间，最小为256b，最大为align(ubM, 8) * 32b, 按照baseM先算
    // baseM不会超过2048，不需要乘法溢出校验
    needUbSize += baseM * ONE_BLK_SIZE;
    // input: pertokenScale fp32
    ubCalc += NUM_DB * sizeof(float);
    // 7: to comfirm that pertokenScale 32B(8, fp32) aligned, up to 7, eg: 1->8
    needUbSize += NUM_DB * sizeof(float) * 7;
    // veccalc: mul(* pertokenScale) fp32 m * n, res of broadcast
    ubCalc += sizeof(float) * ubCalcN;
    if (inputParams_.biasDtype != ge::DT_INT32) {
    // veccalc: fp32 out muls fp32 bias
    ubCalc += sizeof(float) * ubCalcN;
    // input: bias bf16/fp16/fp32, veccalc: bias fp32
    needUbSize += NUM_DB * ge::GetSizeByDataType(inputParams_.biasDtype) * ubCalcN + sizeof(float) * ubCalcN;
    OP_TILING_CHECK(needUbSize >= ubSize,
                    CUBE_INNER_ERR_REPORT(inputParams_.opName,
                                          "there is no proper ub tiling when m(%lu) n(%lu) baseM(%u) baseN(%u)",
                                          inputParams_.mSize, inputParams_.nSize, baseM, baseN),
                    return ge::GRAPH_FAILED);
    ubSize -= needUbSize;
    uint32_t ubCalcM = std::min(std::min(ubSize / ubCalc, static_cast<uint64_t>(baseM)), inputParams_.mSize);
    OP_TILING_CHECK(ubCalcM == 0,
                    CUBE_INNER_ERR_REPORT(inputParams_.opName, "failed to calc ubCalcM(0) with ubCalcN(%u)", ubCalcN),
                    return ge::GRAPH_FAILED);
    tilingData->ubCalcN = ubCalcN;
    tilingData->ubCalcM = ubCalcM;
    tilingData->needUbBuffer = ubCalcN * ubCalcM * UB_EXTRE_BYTE;
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
