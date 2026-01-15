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
 * \file aclnn_distribute_barrier_base.cpp
 * \brief
 */
#include <algorithm>

#include "aclnn_distribute_barrier_v2.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "op_mc2_def.h"
#include "opdev/common_types.h"
#include "opdev/op_log.h"
#include "aclnn_distribute_barrier_base.h"
#include "mc2_moe_context.h"
#include <vector>
#include "acl/acl.h"
#include "hccl_util.h"
#include "hccl/hcom.h"
#include "hccl/hccl_comm.h"
#include "hccl/hccl_rank_graph.h"
#include "hccl/hccl_res.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif
extern aclnnStatus aclnnInnerDistributeBarrier(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                               aclrtStream stream);

extern "C" void __attribute__((weak)) NnopbaseSetHcclServerType(void* executor, NnopbaseHcclServerType sType);

extern aclnnStatus aclnnInnerDistributeBarrierGetWorkspaceSize(const aclTensor* xRef, const aclTensor* timeOut,
                                                               const aclTensor* elasticInfo, const char* group,
                                                               int64_t worldSize, uint64_t* workspaceSize,
                                                               aclOpExecutor** executor);

extern aclnnStatus aclnnInnerDistributeBarrierExtend(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                               aclrtStream stream);


extern aclnnStatus aclnnInnerDistributeBarrierExtendGetWorkspaceSize(const aclTensor* mc2ContextTensor,
                                                                     const aclTensor* xRef, const aclTensor* timeOut,
                                                                     const aclTensor* elasticInfo, const char* group,
                                                                     int64_t worldSize, int64_t hccl_topo_type, 
                                                                     uint64_t* workspaceSize, aclOpExecutor** executor);


// check nullptr
bool BarrierCheckNullStatus(const aclTensor* xRef, const char* group)
{
    // 检查必选入参出参为非空
    OP_CHECK_NULL(xRef, return false);
    if (group == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Required group name is Empty.");
        return false;
    }
    return true;
}

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shape_size = 1;
    for (auto i : shape) {
        shape_size *= i;
    }
    return shape_size;
}

int CreateMc2ContextTensor(Mc2MoeContext &mc2Context, const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * sizeof(mc2Context);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    OP_API_CHECK(ret != ACLNN_SUCCESS,
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "aclrtMalloc failed. ret: %d", ret);
                return ACLNN_ERR_PARAM_INVALID;);
    ret = aclrtMemcpy(*deviceAddr, size, &mc2Context, size, ACL_MEMCPY_HOST_TO_DEVICE);
    OP_API_CHECK(ret != ACLNN_SUCCESS,
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "aclrtMemcpy failed. ret: %d", ret);
                return ACLNN_ERR_PARAM_INVALID;);
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
    return 0;
}


aclnnStatus CreatMc2Context(HcclComm hcclHandle, CommEngine engine,
                            Mc2MoeContext* mc2Context, int64_t worldSize, void* ctx)
{
    uint32_t hcclEpRankId = 0;
    const char* groupEp = "group_ep";
    std::string mc2Ctxtag = std::string(groupEp) + "distribue_barrier"; // 最长255
    uint64_t ctxSize = sizeof(Mc2MoeContext);

    HcclResult ret = HcclEngineCtxCreate(hcclHandle, mc2Ctxtag.c_str(), engine, ctxSize, &ctx);
    if(ret != HCCL_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "HcclEngineCtxCreate failed.");
        return ACLNN_ERR_PARAM_NULLPTR;
    }

    ret = HcclGetRankId(hcclHandle, &hcclEpRankId);
    if (ret != HCCL_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "HcckGetRankId failed.");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    mc2Context->epRankId = hcclEpRankId;

    void* hcclBufferAddr = nullptr;
    uint64_t hcclBufferSize = 0;
    ret = HcclGetHcclBuffer(hcclHandle, &hcclBufferAddr, &hcclBufferSize);
    if (ret != HCCL_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "HcckGetHcclBuffer failed.");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    mc2Context->epHcclBuffer_[mc2Context->epRankId] = (uint64_t)hcclBufferAddr;

    uint32_t channelNum = 1;
    std::vector<HcclChannelDesc> channelDesc(channelNum);
    ret = HcclChannelDescInit(channelDesc.data(), channelNum);
    if (ret != HCCL_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "HcclChannelAcquire failed.");
        return ACLNN_ERR_PARAM_NULLPTR;
    }

    for (uint64_t rankId = 0; rankId < worldSize; rankId++) {
        if (rankId == mc2Context->epRankId) {
            continue;
        }

        channelDesc[0].remoteRank = rankId;
        channelDesc[0].channelProtocol = CommProtocol::COMM_PROTOCOL_UB_MEM;
        channelDesc[0].notifyNum = 3;
        std::vector<ChannelHandle> channels(channelNum);
        ret = HcclChannelAcquire(hcclHandle, engine, channelDesc.data(), channelNum, channels.data());
        if (ret != HCCL_SUCCESS) {
            OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "HcclChannelAcquire failed.");
            return ACLNN_ERR_PARAM_NULLPTR;
        }
        void* remoteHcclBufferAddr = nullptr;
        uint64_t remoteHcclBufferSize = 0;
        ret = HcclChannelGetHcclBuffer(hcclHandle, channels[0], &remoteHcclBufferAddr, &remoteHcclBufferSize);
        if (ret != HCCL_SUCCESS) {
            OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "HcclChannelGetHcclBuffer failed.");
            return ACLNN_ERR_PARAM_NULLPTR;
        }
        mc2Context->epHcclBuffer_[rankId] = (uint64_t)remoteHcclBufferAddr;
    }

    return ACLNN_SUCCESS;
}

aclnnStatus GetMc2Context(aclTensor* mc2Context, int64_t worldSize) 
{
    Mc2MoeContext mc2_context;
    HcclComm hcclHandle;
    HcclResult ret;
    CommEngine engine = CommEngine::COMM_ENGINE_AIV; //默认AIV引擎
    const char* groupEp = "group_ep";
    std::string mc2Ctxtag = std::string(groupEp) + "_distribue_barrier"; // 最长255
    void * ctx = nullptr;
    uint64_t ctxSize = sizeof(Mc2MoeContext);
    ret = HcomGetCommHandleByGroup(groupEp, &hcclHandle);
    if(ret != HCCL_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER, "Get Hccl Ep Handle failed.");
        return ACLNN_ERR_INNER;
    }

    ret = HcclEngineCtxGet(hcclHandle, mc2Ctxtag.c_str(), engine, &ctx, &ctxSize);
    if (ret != HCCL_SUCCESS) { 
        //如果资源不存在则进行context结构体创建
        auto retParam = CreatMc2Context(hcclHandle, engine, &mc2_context, worldSize, ctx);
        CHECK_RET(retParam == ACLNN_SUCCESS, retParam);
        std::vector<int64_t> mc2ContextShape = {sizeof(Mc2MoeContext) / sizeof(int64_t)};
        int creatTensorRet = CreateMc2ContextTensor(mc2_context, mc2ContextShape, &ctx,
                                                    ACL_INT8, &mc2Context);
        if (!creatTensorRet) {
            OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "CreateMc2ContextTensor failed.");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }

    return ACLNN_SUCCESS;
}

// 入参校验
aclnnStatus BarrierCheckParams(const aclTensor* xRef, const char* group)
{
    CHECK_RET(BarrierCheckNullStatus(xRef, group), ACLNN_ERR_PARAM_NULLPTR);
    auto groupStrnLen = strnlen(group, HCCL_GROUP_NAME_MAX);
    if ((groupStrnLen >= HCCL_GROUP_NAME_MAX) || (groupStrnLen == 0)) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Required group name length in range (0, HCCL_GROUP_NAME_MAX), but it's %zu.",
                groupStrnLen);
        return false;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnDistributeBarrierGetWorkspaceSizeBase(const aclTensor* xRef, const aclTensor* timeOut,
                                                       const aclTensor* elasticInfo, const char* group,
                                                       int64_t worldSize, uint64_t* workspaceSize,
                                                       aclOpExecutor** executor)
{
    auto retParam = BarrierCheckParams(xRef, group);
    CHECK_RET(retParam == ACLNN_SUCCESS, retParam);
    aclTensor* mc2Context = nullptr;
    retParam = GetMc2Context(mc2Context, worldSize);
 	CHECK_RET(retParam == ACLNN_SUCCESS, retParam);
    int64_t hccl_topo_type = 0;


    return aclnnInnerDistributeBarrierExtendGetWorkspaceSize(mc2Context, xRef, timeOut, elasticInfo, group,
                                                             worldSize, hccl_topo_type, workspaceSize, executor);
}

aclnnStatus aclnnDistributeBarrierBase(void* workspace, uint64_t workspaceSize,
                                       aclOpExecutor* executor,
                                       aclrtStream stream)
{
    if (NnopbaseSetHcclServerType) {
        NnopbaseSetHcclServerType(executor, NNOPBASE_HCCL_SERVER_TYPE_MTE);
    }
    return aclnnInnerDistributeBarrierExtend(workspace, workspaceSize, executor,
                                             stream);
}

#ifdef __cplusplus
}
#endif