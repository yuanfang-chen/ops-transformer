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
 * \file aclnn_moe_distribute_combine_v2_base.cpp
 * \brief
 */


#include <algorithm>
#include "op_mc2.h"
#include "op_mc2_def.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/op_log.h"
#include "opdev/common_types.h"
#include "common/op_host/op_api/matmul_util.h"
#include "aclnn_moe_distribute_combine_v2_base.h"
#include "hccl/hcom.h"
#include "hccl/hccl.h"
#include "hccl/hccl_types.h"
#include "hccl/hccl_comm.h"
#include "hccl/hccl_rank_graph.h"
#include "hccl/hccl_res.h"
#include "hccl/hccn_rping.h"
#include "mc2_moe_struct.h"

using namespace Ops::Transformer;
using namespace op;
using namespace Mc2Moe;
#ifdef __cplusplus
extern "C" {
#endif

extern "C" void __attribute__((weak)) NnopbaseSetHcclServerType(void *executor, NnopbaseHcclServerType sType);
extern "C" void NnopbaseSetUserHandle(void *executor, void *handle);
extern "C" void* NnopbaseGetUserHandle(void *executor);

// host侧通信资源准备
extern uint32_t AscCommResPrepare(const char *group, const std::string &opName, void *ascCommArgs,
                                  void **ascCommContext);
// host侧通信资源下发
extern uint32_t AscCommGetArgs(void **ascCommArgs);
extern uint32_t AscCommSetCommEngine(void *ascCommArgs, uint8_t commEngine);
extern uint32_t AscCommSetHcclAlgo(void *ascCommArgs, const std::string &hcclAlgo);
extern uint32_t AscCommFreeArgs(void **ascCommArgs);

extern aclnnStatus aclnnInnerMoeDistributeCombineV2GetWorkspaceSize(
    const aclTensor *expandX, const aclTensor *expertIds, const aclTensor *assistInfoForCombine,
    const aclTensor *epSendCounts, const aclTensor *expertScales, const aclTensor *tpSendCounts,
    const aclTensor *xActiveMask, const aclTensor *activationScale, const aclTensor *weightScale,
    const aclTensor *groupList, const aclTensor *expandScales, const aclTensor *sharedExpertX,
    const aclTensor *elasticInfo, const aclTensor *oriX, const aclTensor *constExpertAlpha1,
    const aclTensor *constExpertAlpha2, const aclTensor *constExpertV, const aclTensor *performanceInfo,
    const char *groupEp, int64_t epWorldSize, int64_t epRankId, int64_t moeExpertNum, const char *groupTp,
    int64_t tpWorldSize, int64_t tpRankId, int64_t expertShardType, int64_t sharedExpertNum,
    int64_t sharedExpertRankNum, int64_t globalBs, int64_t outDtype, int64_t commQuantMode, int64_t groupListType,
    const char *commAlg, int64_t zeroExpertNum, int64_t copyExpertNum, int64_t constExpertNum, aclTensor *x,
    uint64_t *workspaceSize, aclOpExecutor **executor);

extern aclnnStatus aclnnInnerMoeDistributeCombineV2ExtendGetWorkspaceSize(
    const aclTensor *expandX, const aclTensor *expertIds, const aclTensor *assistInfoForCombine,
    const aclTensor *epSendCounts, const aclTensor *expertScales, const aclTensor *mc2Context,
    const aclTensor *tpSendCounts, const aclTensor *xActiveMask, const aclTensor *activationScale,
    const aclTensor *weightScale, const aclTensor *groupList, const aclTensor *expandScales,
    const aclTensor *sharedExpertX, const aclTensor *elasticInfo, const aclTensor *oriX,
    const aclTensor *constExpertAlpha1, const aclTensor *constExpertAlpha2, const aclTensor *constExpertV,
    const aclTensor *performanceInfo, const char *groupEp, int64_t epWorldSize, int64_t epRankId, int64_t moeExpertNum,
    int64_t hcclBuffSize, const char *hcclTopoType, const char *groupTp, int64_t tpWorldSize, int64_t tpRankId,
    int64_t expertShardType, int64_t sharedExpertNum, int64_t sharedExpertRankNum, int64_t globalBs, int64_t outDtype,
    int64_t commQuantMode, int64_t groupListType, const char *commAlg, int64_t zeroExpertNum, int64_t copyExpertNum,
    int64_t constExpertNum, aclTensor *x, uint64_t *workspaceSize, aclOpExecutor **executor);

extern aclnnStatus aclnnInnerMoeDistributeCombineV2(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                                    aclrtStream stream);

extern aclnnStatus aclnnInnerMoeDistributeCombineV2Extend(void *workspace, uint64_t workspaceSize,
                                                          aclOpExecutor *executor, aclrtStream stream);
namespace {
inline aclnnStatus CheckHccl(HcclResult res, aclnnStatus err, const char *msg)
{
    if (res != HCCL_SUCCESS) {
        OP_LOGE(err, "%s", msg);
        return res;
    }
    return ACLNN_SUCCESS;
}
#define CHECK_HCCL(res, err, msg)                                                                                      \
    do {                                                                                                               \
        auto _st = CheckHccl(res, err, msg);                                                                           \
        if (_st != ACLNN_SUCCESS)                                                                                      \
            return _st;                                                                                                \
    } while (0)

} // namespace

namespace {
constexpr CommEngine commEngine = CommEngine::COMM_ENGINE_AIV; // 默认AIV引擎
std::string opName = "moe_distribute_combine_v2";
bool isCcu = false;
enum Mc2TopoType : uint32_t {
    MC2_TOPO_AIV_DPU = 0,
    MC2_TOPO_HOST_KFC = 1,
};
} // namespace


bool CombineCheckNotNull(const aclTensor *expandX, const aclTensor *expertIds, const aclTensor *assistInfoForCombine,
                         const aclTensor *epSendCounts, const aclTensor *expertScales, const char *groupEp,
                         aclTensor *x)
{
    OP_CHECK_NULL(epSendCounts, return false);
    OP_CHECK_NULL(expertScales, return false);
    OP_CHECK_NULL(x, return false);
    OP_CHECK_NULL(expandX, return false);
    OP_CHECK_NULL(expertIds, return false);
    OP_CHECK_NULL(assistInfoForCombine, return false);

    if ((groupEp == nullptr) || (strnlen(groupEp, HCCL_GROUP_NAME_MAX) == 0)) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "groupEp Name is Empty.");
        return false;
    }
    return true;
}

aclnnStatus CombineCheckParams(const aclTensor *expandX, const aclTensor *expertIds, const aclTensor *expandIdx,
                               const aclTensor *epSendCounts, const aclTensor *expertScales, const char *groupEp,
                               const char *groupTp, aclTensor *x)
{
    CHECK_RET(CombineCheckNotNull(expandX, expertIds, expandIdx, epSendCounts, expertScales, groupEp, x),
              ACLNN_ERR_PARAM_NULLPTR);

    if (strnlen(groupEp, HCCL_GROUP_NAME_MAX) >= HCCL_GROUP_NAME_MAX) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Required groupEp name exceeds %zu.", HCCL_GROUP_NAME_MAX);
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (strnlen(groupTp, HCCL_GROUP_NAME_MAX) >= HCCL_GROUP_NAME_MAX) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Required groupTp name exceeds %zu.", HCCL_GROUP_NAME_MAX);
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus GetNetAndTopo(const char *groupEp, int64_t epRankId, HcclComm &hcclHandle, uint32_t &rank, uint32_t &world,
                          uint32_t &netLayerNum, Mc2TopoType &topoTypeOut)
{
    HcclResult res = HcomGetCommHandleByGroup(groupEp, &hcclHandle);
    CHECK_HCCL(res, ACLNN_ERR_INNER, "Get CommHandle Failed.");

    res = HcclGetRankId(hcclHandle, &rank);
    CHECK_HCCL(res, ACLNN_ERR_INNER, "Hccl Get epRankId Failed.");
    res = HcclGetRankSize(hcclHandle, &world);
    CHECK_HCCL(res, ACLNN_ERR_INNER, "Hccl Get epRankSize Failed.");

    if (epRankId >= 0 && (uint32_t)epRankId != rank) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "epRankId mismatch.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    uint32_t *netLayers = nullptr;
    netLayerNum = 0;
    res = HcclRankGraphGetLayers(hcclHandle, &netLayers, &netLayerNum);
    CHECK_HCCL(res, ACLNN_ERR_INNER, "Hccl Get Net Layers Failed.");
    topoTypeOut = Mc2TopoType::MC2_TOPO_AIV_DPU;
    if (netLayerNum <= 1) { // 第一层: MTE/CCU
        return ACLNN_SUCCESS;
    }
    // 第二层
    CommLink *commLink = nullptr;
    uint32_t linkNum = 0;
    uint32_t srcRank = rank;
    uint32_t dstRank = (srcRank + 1) % world;
    const uint32_t netLayer = 1;
    res = HcclRankGraphGetLinks(hcclHandle, netLayer, srcRank, dstRank, &commLink, &linkNum); // 获取第二层组网的links
    CHECK_HCCL(res, ACLNN_ERR_INNER, "Hccl Get Layer2 Links Failed.");
    bool isHost = false;
    for (uint32_t i = 0; i < linkNum && commLink; ++i) {
        if (commLink[i].linkAttr.hop > 0) {
            isHost = true;
            break;
        }
    }
    topoTypeOut = isHost ? Mc2TopoType::MC2_TOPO_HOST_KFC : Mc2TopoType::MC2_TOPO_AIV_DPU;
    return ACLNN_SUCCESS;
}

// aclnnStatus BuildKfcContext()
// {
//     void* ascCommArgs;
//     AscCommGetArgs(&ascCommArgs);
//     AscCommSetCommEngine(ascCommArgs, commEngine);  // 选MTE,AICPU等方式
//     AscCommSetCommEngine(ascCommArgs, hcclAlgo);
//     void * kfcContextAddr;
//     AscCommResPrepare("group_name", "AllToAll", ascCommArgs, &kfcContextAddr);
//     AscCommResPrepare("group_name", "AllGather", ascCommArgs2, &kfcContextAddr); //
//     多个通信域，传同一个kfcContextAddr即可
// }

aclnnStatus BuildMc2Context(HcclComm hcclHandle, const char *groupEp, int64_t epRankId, void *&devCtx,
                            const aclTensor *&mc2TensorOut, Mc2TopoType &topoTypeOut, uint64_t &hcclBuffSize)
{
    std::string mc2CtxTag = std::string(groupEp) + opName;
    OP_LOGD("[BuildMc2Context] mc2CtxTag:%s", mc2CtxTag);
    uint64_t ctxSize = 0;
    HcclResult res = HcclEngineCtxGet(hcclHandle, mc2CtxTag.c_str(), commEngine, &devCtx, &ctxSize);
    if (res != HCCL_SUCCESS && devCtx != nullptr && ctxSize >= sizeof(Mc2MoeContext)) {
        res = HcclEngineCtxCreate(hcclHandle, mc2CtxTag.c_str(), commEngine, sizeof(Mc2MoeContext), &devCtx);
        CHECK_HCCL(res, ACLNN_ERR_INNER, "Hccl Mc2Context Create Failed.");

        Mc2MoeContext mc2Context{};
        res = HcclGetRankId(hcclHandle, &mc2Context.epRankId);
        CHECK_HCCL(res, ACLNN_ERR_INNER, "Hccl Get epRankId Failed.");

        res = HcclGetRankSize(hcclHandle, &mc2Context.epRankSize);
        CHECK_HCCL(res, ACLNN_ERR_INNER, "Hccl Get epRankSize Failed.");

        if (epRankId >= 0 && (uint32_t)epRankId != mc2Context.epRankId) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "epRankId mismatch.");
            return ACLNN_ERR_PARAM_INVALID;
        }
        void *hcclBuffer = nullptr;
        res = HcclGetHcclBuffer(hcclHandle, &hcclBuffer, &hcclBuffSize);
        CHECK_HCCL(res, ACLNN_ERR_INNER, "Get HcclBuffer Failed.");
        if (mc2Context.epRankId < HCCL_HOST_KFC_MAX_RANK_NUM) {
            mc2Context.epHcclBuffer_[mc2Context.epRankId] = (uint64_t)hcclBuffer;
        }
        if (mc2Context.epRankSize > 1) {
            const uint32_t channelNum = mc2Context.epRankSize - 1;
            HcclChannelDesc *descs = new (std::nothrow) HcclChannelDesc[channelNum];
            ChannelHandle *ch = new (std::nothrow) ChannelHandle[channelNum];
            if (descs == nullptr || ch == nullptr) {
                delete[] descs;
                delete[] ch;
                OP_LOGE(ACLNN_ERR_INNER, "Alloc Channel Desc Failed.");
                return ACLNN_ERR_INNER;
            }
            res = HcclChannelDescInit(descs, channelNum);
            if (res != ACLNN_SUCCESS) {
                delete[] descs;
                delete[] ch;
                CHECK_HCCL(res, ACLNN_ERR_INNER, "Hccl Channel Desc Init Failed.");
            }
            uint32_t idx = 0;
            for (uint32_t r = 0; r < mc2Context.epRankSize; ++r) {
                if (r != (uint32_t)mc2Context.epRankId) {
                    descs[idx++].remoteRank = r;
                }
            }
            res = HcclChannelAcquire(hcclHandle, commEngine, descs, channelNum, ch);
            if (res != ACLNN_SUCCESS) {
                delete[] descs;
                delete[] ch;
                CHECK_HCCL(res, ACLNN_ERR_INNER, "Hccl Channel Acquire Failed.");
            }
            for (uint32_t i = 0; i < channelNum; ++i) {
                void *buf = nullptr;
                uint64_t bufSize = 0;
                res = HcclChannelGetHcclBuffer(hcclHandle, ch[i], &buf, &bufSize);
                if (res != ACLNN_SUCCESS) {
                    delete[] descs;
                    delete[] ch;
                    CHECK_HCCL(res, ACLNN_ERR_INNER, "Hccl Channel Get HcclBuffer Failed.");
                }
                uint32_t remoteRank = descs[i].remoteRank;
                if (remoteRank < HCCL_HOST_KFC_MAX_RANK_NUM) {
                    mc2Context.epHcclBuffer_[remoteRank] = (uint64_t)buf;
                }
            }
            delete[] descs;
            delete[] ch;
        }
        // BuildKfcContext();
        const uint64_t dstCtxOffset = 0; // 全部拷贝，偏移为0
        res = HcclEngineCtxCopy(hcclHandle, commEngine, mc2CtxTag.c_str(), &mc2Context, sizeof(Mc2MoeContext),
                                dstCtxOffset);
        CHECK_HCCL(res, ACLNN_ERR_INNER, "Copy Mc2Context from Host to Device Failed.");
    }
    uint64_t bytes = sizeof(Mc2MoeContext);
    int64_t shape[1] = {(int64_t)(bytes / sizeof(uint32_t))};
    int64_t strides[1] = {1};
    mc2TensorOut =
        aclCreateTensor(shape, 1, aclDataType::ACL_UINT32, strides, 0, aclFormat::ACL_FORMAT_ND, shape, 1, devCtx);
    if (mc2TensorOut == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER, " Create mc2Context Tensor Failed.");
        return ACLNN_ERR_INNER;
    }
    return ACLNN_SUCCESS;
}

inline void SetCommArgs(aclOpExecutor **executor, const bool is910B, const bool is950, const char *commAlg)
{
    if (is950) {
        const uint64_t aiv_comm = 0;
        const uint64_t ccu_comm = 1;
        void *args = (void *)aiv_comm;
        if (isCcu) {
            args = (void *)ccu_comm;
        }
        NnopbaseSetUserHandle(executor, args);
    }
    if (NnopbaseSetHcclServerType) {
        if (is910B) {
            NnopbaseSetHcclServerType(*executor, NNOPBASE_HCCL_SERVER_TYPE_AICPU);
        } else if (is950 && isCcu) {
            NnopbaseSetHcclServerType(*executor, NNOPBASE_HCCL_SERVER_TYPE_CCU);
        } else {
            NnopbaseSetHcclServerType(*executor, NNOPBASE_HCCL_SERVER_TYPE_MTE);
        }
    }
}

// aclnn一段式接口
aclnnStatus aclnnMoeDistributeCombineBaseGetWorkspaceSize(
    const aclTensor *expandX, const aclTensor *expertIds, const aclTensor *assistInfoForCombine,
    const aclTensor *epSendCounts, const aclTensor *expertScales, const aclTensor *tpSendCountsOptional,
    const aclTensor *xActiveMaskOptional, const aclTensor *activationScaleOptional,
    const aclTensor *weightScaleOptional, const aclTensor *groupListOptional, const aclTensor *expandScalesOptional,
    const aclTensor *sharedExpertXOptional, const aclTensor *elasticInfoOptional, const aclTensor *oriXOptional,
    const aclTensor *constExpertAlpha1Optional, const aclTensor *constExpertAlpha2Optional,
    const aclTensor *constExpertVOptional, const aclTensor *performanceInfoOptional, const char *groupEp,
    int64_t epWorldSize, int64_t epRankId, int64_t moeExpertNum, const char *groupTp, int64_t tpWorldSize,
    int64_t tpRankId, int64_t expertShardType, int64_t sharedExpertNum, int64_t sharedExpertRankNum, int64_t globalBs,
    int64_t outDtype, int64_t commQuantMode, int64_t groupListType, const char *commAlg, int64_t zeroExpertNum,
    int64_t copyExpertNum, int64_t constExpertNum, aclTensor *xOut, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    const static bool is910B = GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B;
    const static bool is950 = GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510;
    auto retParam = CombineCheckParams(expandX, expertIds, assistInfoForCombine, epSendCounts, expertScales, groupEp,
                                       groupTp, xOut);
    CHECK_RET(retParam == ACLNN_SUCCESS, retParam);

    const aclTensor *performanceInfoOptionalCombineV2Temp = performanceInfoOptional;
    const aclTensor *mc2Context = nullptr;
    const char *groupTpCombineV2Temp = groupTp;
    if (is910B) {
        groupTpCombineV2Temp = "";
    } else if (is950) {
        performanceInfoOptionalCombineV2Temp = nullptr;
    }
    isCcu = (commAlg != nullptr && std::strcmp(commAlg, "ccu") == 0);
    HcclComm hcclHandle;
    uint32_t rank;
    uint32_t world;
    uint32_t netLayerNum = 0;
    Mc2TopoType topoType;
    aclnnStatus res = GetNetAndTopo(groupEp, epRankId, &hcclHandle, rank, world, netLayerNum, topoType);
    CHECK_RET(res == ACLNN_SUCCESS, res);
    OP_LOGD("[aclnn-1] commAlg: %s", commAlg);
    aclnnStatus getWorkspaceSizesRes;

    if (!is950 || (is950 && isCcu)) {
        OP_LOGD("[aclnn-1] Enter to the 910B | CCU");
        getWorkspaceSizesRes = aclnnInnerMoeDistributeCombineV2GetWorkspaceSize(
            expandX, expertIds, assistInfoForCombine, epSendCounts, expertScales, tpSendCountsOptional,
            xActiveMaskOptional, activationScaleOptional, weightScaleOptional, groupListOptional, expandScalesOptional,
            sharedExpertXOptional, elasticInfoOptional, oriXOptional, constExpertAlpha1Optional,
            constExpertAlpha2Optional, constExpertVOptional, performanceInfoOptionalCombineV2Temp, groupEp, epWorldSize,
            epRankId, moeExpertNum, groupTpCombineV2Temp, tpWorldSize, tpRankId, expertShardType, sharedExpertNum,
            sharedExpertRankNum, globalBs, outDtype, commQuantMode, groupListType, commAlg, zeroExpertNum,
            copyExpertNum, constExpertNum, xOut, workspaceSize, executor);
    } else {
        OP_LOGD("[aclnn-1] Enter to the 950");
        void *devCtx = nullptr;
        uint64_t hcclBuffSize = 0;
        res = BuildMc2Context(hcclHandle, groupEp, epRankId, devCtx, mc2Context, topoType, hcclBuffSize);
        CHECK_RET(res == ACLNN_SUCCESS, res);
        const char *hcclTopoType = (topoType == Mc2TopoType::MC2_TOPO_AIV_DPU) ? "AIV_DPU" : "HOST_KFC";
        getWorkspaceSizesRes = aclnnInnerMoeDistributeCombineV2ExtendGetWorkspaceSize(
            expandX, expertIds, assistInfoForCombine, epSendCounts, expertScales, mc2Context, tpSendCountsOptional,
            xActiveMaskOptional, activationScaleOptional, weightScaleOptional, groupListOptional, expandScalesOptional,
            sharedExpertXOptional, elasticInfoOptional, oriXOptional, constExpertAlpha1Optional,
            constExpertAlpha2Optional, constExpertVOptional, performanceInfoOptionalCombineV2Temp, groupEp, epWorldSize,
            epRankId, moeExpertNum, hcclBuffSize, hcclTopoType, groupTpCombineV2Temp, tpWorldSize, tpRankId,
            expertShardType, sharedExpertNum, sharedExpertRankNum, globalBs, outDtype, commQuantMode, groupListType,
            commAlg, zeroExpertNum, copyExpertNum, constExpertNum, xOut, workspaceSize, executor);
    }
    SetCommArgs(executor, is910B, is950, commAlg);
    return getWorkspaceSizesRes;
}

// aclnn二段式接口
aclnnStatus aclnnMoeDistributeCombineBase(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                          aclrtStream stream)
{
    const static bool is950 = GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510;
    if (is950) {
        OP_LOGD("[aclnn-2] Enter to the 950");
        void *args = NnopbaseGetUserHandle(executor);
        uint64_t handleVal = reinterpret_cast<uint64_t>(args);
        if (handleVal == 0) {
            OP_LOGD("[aclnn-2] aclnnInnerMoeDistributeCombineV2Extend");
            return aclnnInnerMoeDistributeCombineV2Extend(workspace, workspaceSize, executor, stream);
        }
    }
    OP_LOGD("[aclnn-2] aclnnInnerMoeDistributeCombineV2");
    return aclnnInnerMoeDistributeCombineV2(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif