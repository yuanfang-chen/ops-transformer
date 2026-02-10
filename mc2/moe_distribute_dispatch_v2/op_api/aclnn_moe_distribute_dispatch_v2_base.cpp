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
 * \file aclnn_moe_distribute_dispatch_v2_base.cpp
 * \brief
 */

#include <algorithm>
#include "mc2_moe_context.h"
#include "op_mc2.h"
#include "op_mc2_def.h"
#include "opdev/op_log.h"
#include "opdev/common_types.h"
#include "opdev/platform.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_moe_distribute_dispatch_v2_base.h"
#include "hccl/hcom.h"
// #include "hccl/hccl_comm.h"
#include "hccl/hccl_rank_graph.h"
// #include "hccl/hccl_res.h"
#include "hccl/hccl.h"
// #include "hccl/hccn_rping.h"
using namespace Ops::Transformer;
using namespace op;
using namespace Mc2Context;
#ifdef __cplusplus
extern "C" {
#endif
extern aclnnStatus aclnnInnerMoeDistributeDispatchV2GetWorkspaceSize(
    const aclTensor* x, const aclTensor* expertIds, const aclTensor* scales,
    const aclTensor* xActiveMask, const aclTensor* expertScales,  const aclTensor* elasticInfo,
    const aclTensor* performanceInfo, const char* groupEp, int64_t epWorldSize,
    int64_t epRankId, int64_t moeExpertNum, const char* groupTp, int64_t tpWorldSize,
    int64_t tpRankId, int64_t expertShardType, int64_t sharedExpertNum, int64_t shareExpertRankNum,
    int64_t quantMode, int64_t globalBs, int64_t expertTokenNumsType, const char* commAlg,
    int64_t zeroExpertNum, int64_t copyExpertNum, int64_t constExpertNum, int64_t ydtype, aclTensor* expandX,
    aclTensor* dynamicScales, aclTensor* assist_info_for_combine, aclTensor* expertTokensNums, aclTensor* epRecvCounts,
    aclTensor* tpRecvCounts, aclTensor* expandScales,
    uint64_t* workspaceSize, aclOpExecutor** executor);
extern aclnnStatus aclnnInnerMoeDistributeDispatchV2ExtendGetWorkspaceSize(
    const aclTensor* x, const aclTensor* expertIds, const aclTensor* mc2Context,const aclTensor* scales,
    const aclTensor* xActiveMask, const aclTensor* expertScales,  const aclTensor* elasticInfo,
    const aclTensor* performanceInfo, const char* groupEp, int64_t epWorldSize,
    int64_t epRankId, int64_t moeExpertNum, int64_t hcclBuffSize, const char * hcclTopoType, const char* groupTp, int64_t tpWorldSize,
    int64_t tpRankId, int64_t expertShardType, int64_t sharedExpertNum, int64_t shareExpertRankNum,
    int64_t quantMode, int64_t globalBs, int64_t expertTokenNumsType, const char* commAlg,
    int64_t zeroExpertNum, int64_t copyExpertNum, int64_t constExpertNum, int64_t ydtype, aclTensor* expandX,
    aclTensor* dynamicScales, aclTensor* assist_info_for_combine, aclTensor* expertTokensNums, aclTensor* epRecvCounts,
    aclTensor* tpRecvCounts, aclTensor* expandScales,
    uint64_t* workspaceSize, aclOpExecutor** executor);
extern aclnnStatus aclnnInnerMoeDistributeDispatchV2(void* workspace, uint64_t workspaceSize,
                                                     aclOpExecutor* executor, aclrtStream stream);
extern aclnnStatus aclnnInnerMoeDistributeDispatchV2Extend(void* workspace, uint64_t workspaceSize,
                                                     aclOpExecutor* executor, aclrtStream stream);
extern "C" void __attribute__((weak)) NnopbaseSetHcclServerType(void *executor, NnopbaseHcclServerType sType);
extern "C" void NnopbaseSetUserHandle(void *executor, void *handle);
extern "C" void* NnopbaseGetUserHandle(void *executor);

bool DispatchCheckNotNull(const aclTensor* x, const aclTensor* expertIds, const char* groupEp,
                          [[maybe_unused]] const char* groupTp, aclTensor* expandX, [[maybe_unused]] aclTensor* dynamicScales,
                          aclTensor* assistInfoForCombine, aclTensor* expertTokensNums, aclTensor* epRecvCounts,
                          aclTensor* tpRecvCounts)
{
    OP_CHECK_NULL(x, return false);
    OP_CHECK_NULL(expertIds, return false);
    OP_CHECK_NULL(expandX, return false);
    OP_CHECK_NULL(assistInfoForCombine, return false);
    OP_CHECK_NULL(expertTokensNums, return false);
    OP_CHECK_NULL(tpRecvCounts, return false);
    OP_CHECK_NULL(epRecvCounts, return false);
    if ((groupEp == nullptr)||(strnlen(groupEp, HCCL_GROUP_NAME_MAX) == 0)) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "group groupEp name is Empty");
        return false;
    }
    return true;
}

aclnnStatus DispatchCheckParams(const aclTensor* x, const aclTensor* expertIds, const char* groupEp, const char* groupTp,
                                int64_t quantMode, aclTensor* expandX, aclTensor* dynamicScales, aclTensor* assistInfoForCombine,
                                aclTensor* expertTokensNums, aclTensor* epRecvCounts, aclTensor* tpRecvCounts)
{
    CHECK_RET(DispatchCheckNotNull(x, expertIds, groupEp, groupTp, expandX, dynamicScales, assistInfoForCombine,
        expertTokensNums, epRecvCounts, tpRecvCounts), ACLNN_ERR_PARAM_NULLPTR);

    if (quantMode == DISPATCH_DYNAMIC_QUANT_MODE) {
        OP_LOGD("quantMode = 2, dynamicScales can't be null");
        CHECK_RET(dynamicScales != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    }
    if (strnlen(groupEp, HCCL_GROUP_NAME_MAX) >= HCCL_GROUP_NAME_MAX) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Required groupEp name exceeds %zu", HCCL_GROUP_NAME_MAX);
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (strnlen(groupTp, HCCL_GROUP_NAME_MAX) >= HCCL_GROUP_NAME_MAX) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Required groupTp name exceeds %zu", HCCL_GROUP_NAME_MAX);
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    return ACLNN_SUCCESS;
}


aclnnStatus GetCommMode(const char* groupEp, HcclComm& hcclHandle, uint32_t& netLayerNum)
{
    OP_LOGD("PRINT GetCommMode start");
    HcclResult ret;
    uint32_t* netLayers = nullptr;
    ret = HcomGetCommHandleByGroup(groupEp, &hcclHandle);
    if(ret != HCCL_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER, "Hccl Get Ep Handle failed.");
        return ACLNN_ERR_INNER;
    }
    OP_LOGD("PRINT HcomGetCommHandleByGroup success");

    ret = HcclRankGraphGetLayers(hcclHandle, &netLayers, &netLayerNum);
    if(ret != HCCL_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER, "Hccl Get NetLayers failed.");
        return ACLNN_ERR_INNER;
    }

    return ACLNN_SUCCESS;
}

aclnnStatus CreatMc2Context(HcclComm hcclHandle, std::string mc2Ctxtag, CommEngine engine, void * ctx, Mc2MoeContext*  mc2_context)
{
    OP_LOGD("PRINT inter to the CreatMc2Context");
    uint64_t ctxSize = sizeof(Mc2MoeContext);
    void * tempBuffer = nullptr;
    uint64_t buffersize = 0;
    uint64_t dstCtxOffset = 0; // 全部拷贝，偏移为0
    HcclResult ret;
    std::vector<HcclChannelDesc> channelDesc;
    std::vector<ChannelHandle> channeles;

    ret = HcclEngineCtxCreate(hcclHandle, mc2Ctxtag.c_str(), engine, ctxSize, &ctx);
    if(ret != HCCL_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER, "Creat MC2 Context failed.");
        return ACLNN_ERR_INNER;
    }

    //获取对应的资源
    ret = HcclGetRankId(hcclHandle, &mc2_context->rankId);
    if(ret != HCCL_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER, "Hccl Get Rank Id failed.");
        return ACLNN_ERR_INNER;
    }

    ret = HcclGetRankSize(hcclHandle, &mc2_context->rankDim);
    if(ret != HCCL_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER, "Hccl Get Rank Size failed.");
        return ACLNN_ERR_INNER;
    }

    channelDesc.resize(mc2_context->rankDim);
    channeles.resize(mc2_context->rankDim);
    HcclChannelDescInit(channelDesc.data(), mc2_context->rankDim);

    for (uint64_t index = 0; index < mc2_context->rankDim; index++) {
        if(index != mc2_context->rankId) {
            channelDesc[index].remoteRank = index;
            channelDesc[index].channelProtocol = CommProtocol::COMM_PROTOCOL_UB_MEM;
            channelDesc[index].notifyNum =3;
        }
    }

    HcclChannelAcquire(hcclHandle, engine, channelDesc.data(), mc2_context->rankDim, channeles.data());

    for(uint64_t index = 0; index < mc2_context->rankDim; index++) {
        if(index == mc2_context->rankId) {
            ret = HcclGetHcclBuffer(hcclHandle, &tempBuffer, &mc2_context->winsize);
        } else {
            ret = HcclChannelGetHcclBuffer(hcclHandle, channeles[index], &tempBuffer, &buffersize);
        }
        if(ret != HCCL_SUCCESS) {
            OP_LOGE(ACLNN_ERR_INNER, "Hccl Get hccl buffer failed.");
            return ACLNN_ERR_INNER;
        }
        mc2_context->windowsIn[index] = reinterpret_cast<uint64_t>(tempBuffer);
    }
    
    //把数据拷贝到device侧
    ret = HcclEngineCtxCopy(hcclHandle, engine, mc2Ctxtag.c_str(), mc2_context, ctxSize, dstCtxOffset);
    if(ret != HCCL_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER, "Copy data from host to device failed.");
        return ACLNN_ERR_INNER;
    }
    OP_LOGD("PRINT end to the CreatMc2Context");
    return ACLNN_SUCCESS;
}

void CreatMc2ContextTensor(void * ctx, const aclTensor* mc2Context)
{
    OP_LOGD("PRINT inter to the CreatMc2ContextTensor");
    uint64_t mc2ContextLength = sizeof(Mc2MoeContext);
    int64_t shap[1] = {mc2ContextLength / sizeof(uint32_t)}; // 默认1维
    int64_t strides[1] = {1};
    mc2Context = aclCreateTensor(
        shap, 1, aclDataType::ACL_UINT32, strides, 0, 
        aclFormat::ACL_FORMAT_ND, shap, 1, ctx);
    OP_LOGD("PRINT end to the CreatMc2ContextTensor");
}



aclnnStatus GetMc2Context(HcclComm hcclHandle, const char* groupEp, const aclTensor* mc2Context, int64_t& hcclBuffSize,
                         std::string& hcclTopoType) 
{
    OP_LOGD("PRINT inter to the GetMc2Context");
    Mc2MoeContext mc2_context;
    HcclResult ret;
    CommEngine engine = CommEngine::COMM_ENGINE_AIV; //默认AIV引擎
    std::string mc2Ctxtag = std::string(groupEp) + "_moe_distribute_dispatch_v2"; // 最长255
    void * ctx = nullptr;
    uint64_t ctxSize = sizeof(Mc2MoeContext);

    ret = HcclEngineCtxGet(hcclHandle, mc2Ctxtag.c_str(), engine, &ctx, &ctxSize);
    if(ret != HCCL_SUCCESS) { 
        //如果资源不存在则进行context结构体创建
        auto retParam = CreatMc2Context(hcclHandle, mc2Ctxtag, engine, ctx, &mc2_context);
        CHECK_RET(retParam == ACLNN_SUCCESS, retParam);
    }
    OP_LOGD("PRINT HcclEngineCtxGet success");
    hcclBuffSize = mc2_context.winsize;
    hcclTopoType = "MTE"; //TODO:目前未找到对应的通讯方式。
    CreatMc2ContextTensor(ctx, mc2Context);
    OP_LOGD("PRINT end to the GetMc2Context");
    return ACLNN_SUCCESS;
}

void SetCommArgs(const bool is950, const bool is910B, const char* commAlg, aclOpExecutor** executor)
{
    if(is950) {
        void *arg = reinterpret_cast<void *>(static_cast<uintptr_t>(0)); // 默认MTE为0
        if(commAlg != nullptr && std::strcmp(commAlg, "ccu") == 0) {
            arg = reinterpret_cast<void *>(static_cast<uintptr_t>(1)); //ccu为1
        }
        NnopbaseSetUserHandle(*executor, arg);
    }
    
    if (NnopbaseSetHcclServerType) {
        if (is910B) {
            NnopbaseSetHcclServerType(*executor, NNOPBASE_HCCL_SERVER_TYPE_AICPU);
        } else if (is950 && commAlg != nullptr && std::strcmp(commAlg, "ccu") == 0) {
            NnopbaseSetHcclServerType(*executor, NNOPBASE_HCCL_SERVER_TYPE_CCU);
        } else {
            NnopbaseSetHcclServerType(*executor, NNOPBASE_HCCL_SERVER_TYPE_MTE);
        }
    }
}

aclnnStatus aclnnMoeDistributeDispatchGetWorkspaceSizeBase(
    const aclTensor* x, const aclTensor* expertIds, const aclTensor* scalesOptional,
    const aclTensor* xActiveMaskOptional, const aclTensor* expertScalesOptional,
    const aclTensor* elasticInfoOptional, const aclTensor* performanceInfoOptional, const char* groupEp,
    int64_t epWorldSize, int64_t epRankId, int64_t moeExpertNum, const char* groupTp, int64_t tpWorldSize,
    int64_t tpRankId, int64_t expertShardType, int64_t sharedExpertNum, int64_t sharedExpertRankNum,
    int64_t quantMode, int64_t globalBs, int64_t expertTokenNumsType, const char* commAlg,
    int64_t zeroExpertNum, int64_t copyExpertNum, int64_t constExpertNum, aclTensor* expandXOut,
    aclTensor* dynamicScalesOut, aclTensor* assistInfoForCombineOut, aclTensor* expertTokenNumsOut, aclTensor* epRecvCountsOut,
    aclTensor* tpRecvCountsOut, aclTensor* expandScalesOut,
    uint64_t* workspaceSize, aclOpExecutor** executor)
{
    const static bool is910B = GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B;
    const static bool is950 = GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510;
    aclnnStatus ret;
    ret = DispatchCheckParams(x, expertIds, groupEp, groupTp, quantMode, expandXOut, dynamicScalesOut,
                                         assistInfoForCombineOut, expertTokenNumsOut, epRecvCountsOut, tpRecvCountsOut);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    const aclTensor* performanceInfoOptionalDispatchV2Temp = performanceInfoOptional;
    const char* groupTpDispatchV2Temp = groupTp;
    const aclTensor* mc2Context = nullptr;
    HcclComm hcclHandle;
    uint32_t netLayerNum;
    aclnnStatus getWorkspaceSizesRes;
    if (is910B) {
        groupTpDispatchV2Temp = "";
    } else if (is950) {
        performanceInfoOptionalDispatchV2Temp = nullptr;
    }
    int64_t ydtype = expandXOut->GetDataType();

    ret = GetCommMode(groupEp, hcclHandle, netLayerNum);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    OP_LOGD("PRINT commAlg:%s",commAlg);
    if(!is950 || (commAlg != nullptr && std::strcmp(commAlg, "ccu") != 0)) { //ccu暂时不支持新方案
        getWorkspaceSizesRes = aclnnInnerMoeDistributeDispatchV2GetWorkspaceSize(
            x, expertIds, scalesOptional, xActiveMaskOptional, expertScalesOptional,
            elasticInfoOptional, performanceInfoOptionalDispatchV2Temp, groupEp, epWorldSize, epRankId, moeExpertNum,
            groupTpDispatchV2Temp, tpWorldSize, tpRankId, expertShardType, sharedExpertNum,
            sharedExpertRankNum, quantMode, globalBs, expertTokenNumsType, commAlg, zeroExpertNum, copyExpertNum,
            constExpertNum, ydtype, expandXOut, dynamicScalesOut, assistInfoForCombineOut, expertTokenNumsOut,
            epRecvCountsOut, tpRecvCountsOut, expandScalesOut, workspaceSize, executor);
    } else {
        OP_LOGD("PRINT inter to the 950");
        int64_t hcclBuffSize = 0;
        std::string hcclTopoType;
        ret =GetMc2Context(hcclHandle, mc2Context, hcclBuffSize, hcclTopoType);
        CHECK_RET(ret == ACLNN_SUCCESS, ret);
        getWorkspaceSizesRes = aclnnInnerMoeDistributeDispatchV2ExtendGetWorkspaceSize(
            x, expertIds, mc2Context,scalesOptional, xActiveMaskOptional, expertScalesOptional,
            elasticInfoOptional, performanceInfoOptionalDispatchV2Temp, groupEp, epWorldSize, epRankId, moeExpertNum,
            hcclBuffSize, hcclTopoType.c_str(), groupTpDispatchV2Temp, tpWorldSize, tpRankId, expertShardType, sharedExpertNum,
            sharedExpertRankNum, quantMode, globalBs, expertTokenNumsType, commAlg, zeroExpertNum, copyExpertNum,
            constExpertNum, ydtype, expandXOut, dynamicScalesOut, assistInfoForCombineOut, expertTokenNumsOut,
            epRecvCountsOut, tpRecvCountsOut, expandScalesOut, workspaceSize, executor);
    }
    SetCommArgs(is950, is910B, commAlg, executor);
    return getWorkspaceSizesRes;
}

aclnnStatus  aclnnMoeDistributeDispatchBase(void* workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream) 
{
    OP_LOGD("PRINT inter to the 2 aclnnMoeDistributeDispatchBase");
    const static bool is950 = GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510;
    if(is950) {
        OP_LOGD("PRINT is950");
        void *arg = NnopbaseGetUserHandle(executor);
        uintptr_t handleVal = reinterpret_cast<uintptr_t>(arg);
        if(handleVal == 0) {
            OP_LOGD("PRINT inter to the  aclnnInnerMoeDistributeDispatchV2Extend");
            return aclnnInnerMoeDistributeDispatchV2Extend(workspace, workspaceSize, executor, stream); //mte走新模版
        }
    }
    OP_LOGD("PRINT inter to the  aclnnInnerMoeDistributeDispatchV2");
    return aclnnInnerMoeDistributeDispatchV2(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif