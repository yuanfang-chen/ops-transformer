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
 * \file moe_distribute_base.h
 * \brief
 */

#ifndef MOE_DISTRIBUTE_BASE_H
#define MOE_DISTRIBUTE_BASE_H

#include "kernel_operator.h"

constexpr uint32_t LOCAL_NOTIFY_MAX_NUM = 64;
constexpr uint32_t LOCAL_STREAM_MAX_NUM = 19U;
constexpr uint32_t AICPU_OP_NOTIFY_MAX_NUM = 2;
constexpr uint32_t AICPU_MAX_RANK_NUM = 128 * 1024;

constexpr uint32_t MAX_RANK_NUM = 64U; // 最大卡数
constexpr uint32_t WRITE_SQE_SIZE = 64U;
constexpr uint32_t WRITE_WITH_NOTIFY_SQE_SIZE = 96U;
constexpr uint32_t WIN_PICI_OFFSET = 1024U * 1024U;
constexpr uint64_t WIN_ADDR_ALIGN = 512UL;
constexpr uint32_t NORMAL_CQE_SIZE = 64U;
constexpr uint32_t CQ_DEPTH_256 =
    256U; // 为cqeBuf申请256*32B空间，初始化HGM上的CQ空间时，如果cqDepth>256，则循环多次DataCopy
constexpr uint32_t UB_ALIGN = 32U; // UB按32字节对齐

struct HcclSignalInfo {
    uint64_t resId; // 在代表event时为eventid，notify时为notifyid
    uint64_t addr;
    uint32_t devId;
    uint32_t tsId;
    uint32_t rankId;
    uint32_t flag;
};

struct ListCommon {
    uint64_t nextHost;
    uint64_t preHost;
    uint64_t nextDevice;
    uint64_t preDevice;
};

struct HcclStreamInfo {
    int32_t streamIds;
    uint32_t sqIds;
    uint32_t cqIds;      // 记录物理cqId
    uint32_t logicCqids; // 记录逻辑cqId
};

struct LocalResInfoV2 {
    uint32_t streamNum;
    uint32_t signalNum;
    HcclSignalInfo localSignals[LOCAL_NOTIFY_MAX_NUM];
    HcclStreamInfo streamInfo[LOCAL_STREAM_MAX_NUM];
    HcclStreamInfo mainStreamInfo;
    HcclSignalInfo aicpuOpNotify[AICPU_OP_NOTIFY_MAX_NUM]; // 集合通信AICPU展开资源
    ListCommon nextTagRes;                                 // HccltagLocalResV2
};

enum class rtFloatOverflowMode_t {
    RT_OVERFLOW_MODE_SATURATION = 0,
    RT_OVERFLOW_MODE_INFNAN,
    RT_OVERFLOW_MODE_UNDEF,
};

struct AlgoTopoInfo {
    uint32_t userRank;     // 通信域 RankID
    uint32_t userRankSize; // 通信域的Rank数量
    int32_t deviceLogicId;
    bool isSingleMeshAggregation;
    uint32_t deviceNumPerAggregation; // 每个Module中的Device数量
    uint32_t superPodNum;             // 集群中总的超节点数
    uint32_t devicePhyId;
    uint32_t topoType; // TopoType
    uint32_t deviceType;
    uint32_t serverNum;
    uint32_t meshAggregationRankSize;
    uint32_t multiModuleDiffDeviceNumMode;
    uint32_t multiSuperPodDiffServerNumMode;
    uint32_t realUserRank;
    bool isDiffDeviceModule;
    bool isDiffDeviceType;
    uint32_t gcdDeviceNumPerAggregation;
    uint32_t moduleNum;
    uint32_t isUsedRdmaRankPairNum;
    uint64_t isUsedRdmaRankPair;
    uint32_t pairLinkCounterNum;
    uint64_t pairLinkCounter;
    uint32_t nicNum;
    uint64_t nicList;                     // niclist数组指针
    uint64_t complanRankLength;           // complanRank占用的字节数
    uint64_t complanRank;                 // 指针
    uint64_t bridgeRankNum;               // bridgeRank占用的个数
    uint64_t bridgeRank;                  // 指针
    uint64_t serverAndsuperPodRankLength; // serverAndsuperPodRank占用的字节数
    uint64_t serverAndsuperPodRank;       // 指针
};

struct HcclOpConfig {
    uint8_t deterministic; // 确定性计算开关
    uint8_t retryEnable;   // 是否重执行
    uint8_t highPerfEnable;
    uint8_t padding[5];      // 大小需要64By对齐，未来添加参数时减小padding
    uint8_t linkTimeOut[8];  // 发送超时时长
    uint64_t notifyWaitTime; // 超时时长，同HCCL_EXEC_TIMEOUT
    uint32_t retryHoldTime;
    uint32_t retryIntervalTime;
    bool interHccsDisable = false; // 使能rdma开关
    rtFloatOverflowMode_t floatOverflowMode = rtFloatOverflowMode_t::RT_OVERFLOW_MODE_UNDEF;
    uint32_t multiQpThreshold = 512; // 多QP每个QP分担数据量最小阈值
};

struct HcclMC2WorkSpace {
    uint64_t workSpace;
    uint64_t workSpaceSize;
};

struct RemoteResPtr {
    uint64_t nextHostPtr;
    uint64_t nextDevicePtr;
};

struct HDCommunicateParams {
    uint64_t hostAddr{0};
    uint64_t deviceAddr{0};
    uint64_t readCacheAddr{0};
    uint32_t devMemSize{0};
    uint32_t buffLen{0};
    uint32_t flag{0};
};

struct HcclRankRelationResV2 {
    uint32_t remoteUsrRankId;
    uint32_t remoteWorldRank;
    uint64_t windowsIn;
    uint64_t windowsOut;
    uint64_t windowsExp;
    ListCommon nextTagRes;
};

struct HcclOpResParam {
    // 本地资源
    HcclMC2WorkSpace mc2WorkSpace;
    uint32_t localUsrRankId;  // usrrankid
    uint32_t rankSize;        // 通信域内total rank个数
    uint64_t winSize;         // 每个win大小，静态图时，可能是0，如果通信域内也有动态图，则可能为非0
    uint64_t localWindowsIn;  // 全F为无效值
    uint64_t localWindowsOut; // 全F为无效值
    char hcomId[128];
    // aicore识别remote window
    uint64_t winExpSize;
    uint64_t localWindowsExp;
    uint32_t rWinStart;  // 为HcclRankRelationRes起始位置
    uint32_t rWinOffset; // 为HcclRemoteRes的大小
    uint64_t version;
    LocalResInfoV2 localRes;
    AlgoTopoInfo topoInfo;

    // 外部配置参数
    HcclOpConfig config;
    uint64_t hostStateInfo;
    uint64_t aicpuStateInfo;
    uint64_t lockAddr;
    uint32_t rsv[16];
    uint32_t notifysize;                        // RDMA场景使用，910B/910_93为4B，其余芯片为8B
    uint32_t remoteResNum;                      // 有效的remoteResNum
    RemoteResPtr remoteRes[AICPU_MAX_RANK_NUM]; // 数组指针，指向HcclRankRelationResV2，下标为remoteUserRankId

    // communicate retry
    HDCommunicateParams kfcControlTransferH2DParams;
    HDCommunicateParams kfcStatusTransferD2HParams;
    uint64_t tinyMem; // for all2all
    uint64_t tinyMemSize;
    // 零拷贝场景使用
    uint64_t zeroCopyHeadPtr;
    uint64_t zeroCopyTailPtr;
    uint64_t zeroCopyRingBuffer;
    uint64_t zeroCopyIpcPtrs[16];     // 保存集合通信时每个对端的输入输出内存地址
    uint32_t zeroCopyDevicePhyId[16]; // 保存每个rank对应的物理卡Id

    bool utraceStatusFlag;
};

// Transport 内存类型
enum class HcclAiRMAMemType : uint32_t {
    LOCAL_INPUT = 0,
    REMOTE_INPUT,

    LOCAL_OUTPUT,
    REMOTE_OUTPUT,

    // 可透传更多的内存，可在MAX_NUM之前追加，例如：
    // LOCAL_EXP,
    // REMOTE_EXP,
    MAX_NUM
};

// Transport 内存信息
struct HcclAiRMAMemInfo {
    uint32_t memMaxNum{0};        // 最大内存数量，等于 HcclAiRMAMemType::MAX_NUM
    uint32_t sizeOfMemDetails{0}; // sizeof(MemDetails)，用于内存校验和偏移计算
    uint64_t memDetailPtr{0};     // MemDetails数组首地址, 个数: HcclAiRMAMemType::MAX_NUM
    // 可往后追加字段
};

// 全部 Transport QP/Mem 信息
struct HcclAiRMAInfo {
    uint32_t curRankId{0}; // 当前rankId
    uint32_t rankNum{0};   // rank数量
    uint32_t qpNum{0};     // 单个Transport的QP数量

    uint32_t sizeOfAiRMAWQ{0};  // sizeof(HcclAiRMAWQ)
    uint32_t sizeOfAiRMACQ{0};  // sizeof(HcclAiRMACQ)
    uint32_t sizeOfAiRMAMem{0}; // sizeof(HcclAiRMAMemInfo)

    // HcclAiRMAWQ二维数组首地址
    // QP个数: rankNum * qpNum
    // 计算偏移获取SQ指针：sqPtr + (dstRankId * qpNum + qpIndex) * sizeOfAiRMAWQ
    // 0 <= qpIndex < qpNum
    uint64_t sqPtr{0};

    // HcclAiRMACQ二维数组首地址
    // QP个数: rankNum * qpNum
    // 计算偏移获取SCQ指针：scqPtr + (dstRankId * qpNum + qpIndex) * sizeOfAiRMACQ
    // 0 <= qpIndex < qpNum
    uint64_t scqPtr{0};

    // HcclAiRMAWQ二维数组首地址
    // QP个数: rankNum * qpNum
    // 计算偏移获取RQ指针：rqPtr + (dstRankId * qpNum + qpIndex) * sizeOfAiRMAWQ
    // 0 <= qpIndex < qpNum
    uint64_t rqPtr{0};

    // HcclAiRMACQ二维数组首地址
    // QP个数: rankNum * qpNum
    // 计算偏移获取RCQ指针: rcqPtr + (dstRankId * qpNum + qpIndex) * sizeOfAiRMACQ
    // 0 <= qpIndex < qpNum
    uint64_t rcqPtr{0};

    // HcclAivMemInfo一维数组
    // 内存信息个数: rankNum
    // 计算偏移获取内存信息指针: memPtr + rankId * sizeOfAiRMAMem
    // srcRankId 获取自身内存信息，dstRankId 获取 Transport 内存信息
    uint64_t memPtr{0};
    // 可往后追加字段
};
struct CombinedCapability {
    uint64_t dataplaneModeBitmap;
};

struct HcclA2CombineOpParam {
    uint64_t workSpace;                              // Address for communication between client and server,
                                                     // hccl requests and clears
    uint64_t workSpaceSize;                          // Space for communication between client and server
    uint32_t rankId;                                 // id of this rank
    uint32_t rankNum;                                // num of ranks in this comm group
    uint64_t winSize;                                // size of each windows memory
    uint64_t windowsIn[AscendC::HCCL_MAX_RANK_NUM];  // windows address for input, windowsIn[rankId] corresponds
                                                     // to the local card address,
                                                     // and others are cross-card mapping addresses.
    uint64_t windowsOut[AscendC::HCCL_MAX_RANK_NUM]; // windows address for output, windowsOut[rankId] corresponds
                                                     // to the local card address,
                                                     // and others are cross-card mapping addresses.
    uint8_t res[8328];
    uint8_t multiFlag;
    __gm__ AscendC::IbVerbsData *data;
    uint64_t dataSize;
    // 追加字段
    uint64_t sizeOfAiRMAInfo; // sizeof(HcclAiRMAInfo)
    uint64_t aiRMAInfo;       // HcclAiRMAInfo* 单个结构体指针

    CombinedCapability *capability; // address of the communication capability information structure on the Device
    uint64_t capabilitySize;        // size of the communication capability information structure
};
enum class DataplaneMode : uint32_t {
    HOST = 0,
    AICPU = 1,
    AIV = 2,
};

enum class DBMode : int32_t {
    INVALID_DB = -1,
    HW_DB = 0,
    SW_DB
};

struct HcclAiRMAWQ {
    // uint32_t wqn{0};
    // uint64_t bufAddr{0};
    // uint32_t wqeSize{0};
    // uint32_t depth{0};
    // uint64_t headAddr{0};
    // uint64_t tailAddr{0};
    // DBMode dbMode{DBMode::INVALID_DB}; // 0-hw/1-sw
    // uint64_t dbAddr{0};
    // uint32_t sl{0};
    uint32_t jettyId;
    uint64_t sqVA;
    uint32_t wqeSize;
    uint32_t sqDepth;
    uint64_t headAddr;
    uint64_t tailAddr;
    uint64_t dbAddr;
    uint32_t tpId;
    uint8_t rmtEid[16];
    uint32_t rmtObjId; // rmtTokenId
    uint32_t rmtTokenValue;
    uint32_t localTokenId;
};

struct HcclAiRMACQ {
    // uint32_t cqn{0};
    // uint64_t bufAddr{0};
    // uint32_t cqeSize{0};
    // uint32_t depth{0};
    // uint64_t headAddr{0};
    // uint64_t tailAddr{0};
    // DBMode dbMode{DBMode::INVALID_DB}; // 0-hw/1-sw
    // uint64_t dbAddr{0};
    uint32_t jfcId;
    uint64_t cqVA;
    uint32_t cqeSize;
    uint32_t cqDepth;
    uint64_t headAddr;
    uint64_t tailAddr;
    uint64_t dbAddr;
};

struct hns_roce_rc_sq_wqe {
    uint32_t byte_4;
    uint32_t msg_len;
    uint32_t immtdata;
    uint32_t byte_16;
    uint32_t byte_20;
    uint32_t rkey;
    uint64_t remoteVA;
};


struct hns_roce_lite_wqe_data_seg {
    uint32_t len;
    uint32_t lkey;
    uint64_t localVA;
};

struct HcclCombinOpParam {
    uint64_t workSpace;                // client和server之间通信的地址
    uint64_t workSpaceSize;            // client和server之间通信的空间大小
    uint32_t rankId;                   // 当前卡rankId
    uint32_t rankDim;                  // 总卡数
    uint64_t winSize;                  // ccu不使用
    uint64_t windowsIn[MAX_RANK_NUM];  // ccu不使用
    uint64_t windowsOut[MAX_RANK_NUM]; // ccu不使用

    // for ccu
    uint64_t xnAddr;  // Xn寄存器其实地址
    uint64_t ckeAddr; // CKE寄存器其实地址
    uint64_t msAddr;  // MS地址，预留
    uint64_t msSize;  // 可写的MS个数，预留

    HcclAiRMAWQ sqs[MAX_RANK_NUM];
    HcclAiRMACQ cqs[MAX_RANK_NUM];
};

__aicore__ inline void cacheWriteThrough(__gm__ uint8_t *sourceAddr, uint64_t length)
{
    __gm__ uint8_t *start =
        (__gm__ uint8_t *)((uint64_t)sourceAddr / AscendC::CACHE_LINE_SIZE * AscendC::CACHE_LINE_SIZE);
    __gm__ uint8_t *end =
        (__gm__ uint8_t *)(((uint64_t)sourceAddr + length) / AscendC::CACHE_LINE_SIZE * AscendC::CACHE_LINE_SIZE);
    AscendC::GlobalTensor<uint8_t> global;
    global.SetGlobalBuffer(start);
    for (uint32_t i = 0; i <= end - start; i += AscendC::CACHE_LINE_SIZE) {
        AscendC::DataCacheCleanAndInvalid<uint8_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
                                          AscendC::DcciDst::CACHELINE_OUT>(global[i]);
    }
}
__aicore__ inline DataplaneMode GetDataplaneMode(GM_ADDR contextGM0)
{
    __gm__ HcclA2CombineOpParam *winContext_ = (__gm__ HcclA2CombineOpParam *)contextGM0;
    CombinedCapability *capability = winContext_->capability;
    uint64_t capabilitySize = winContext_->capabilitySize;
    DataplaneMode dataplaneMode = DataplaneMode::AICPU;
    if (capability == 0) {
        return dataplaneMode;
    }
    uint64_t dataplaneModeBitmap = capability->dataplaneModeBitmap;
    if ((dataplaneModeBitmap & 0x04) == 0x04) {
        dataplaneMode = DataplaneMode::AIV;
    }
    return dataplaneMode;
}

template <AscendC::HardEvent event>
__aicore__ inline void SyncFunc()
{
    // TODO URMA提取到公共类后该函数如何处理？
    int32_t eventID = static_cast<int32_t>(GetTPipePtr()->FetchEventID(event));
    AscendC::SetFlag<event>(eventID);
    AscendC::WaitFlag<event>(eventID);
}

// URMA通信接口
__aicore__ inline GM_ADDR GetURMASqInfoGMAddr(GM_ADDR hcclContext, uint32_t dstRankId)
{
    ascendc_assert((hcclContext != nullptr && dstRankId < MAX_RANK_NUM),
                   "hcclContext is nullptr or dstRankId >= MAX_RANK_NUM");

    return (GM_ADDR)((__gm__ HcclAiRMAWQ *)(((__gm__ HcclCombinOpParam *)hcclContext)->sqs) + dstRankId);
}

__aicore__ inline GM_ADDR GetURMACqInfoGMAddr(GM_ADDR hcclContext, uint32_t dstRankId)
{
    ascendc_assert((hcclContext != nullptr && dstRankId < MAX_RANK_NUM),
                   "hcclContext is nullptr or dstRankId >= MAX_RANK_NUM");

    return (GM_ADDR)((__gm__ HcclAiRMACQ *)(((__gm__ HcclCombinOpParam *)hcclContext)->cqs) + dstRankId);
}

__aicore__ inline void GetURMASqInfoTensor(const AscendC::LocalTensor<uint8_t> &sqInfoTensor, GM_ADDR hcclContext,
                                           uint32_t dstRankId)
{
    GM_ADDR sqInfoGMAddr = GetURMASqInfoGMAddr(hcclContext, dstRankId);
    ascendc_assert((sqInfoGMAddr != nullptr), "GetURMASqInfoGMAddr failed");

    AscendC::GlobalTensor<uint8_t> sqInfoGlobalTensor;
    sqInfoGlobalTensor.SetGlobalBuffer((__gm__ uint8_t *)(sqInfoGMAddr));
    AscendC::DataCopyExtParams sqInfoParams = {1U, static_cast<uint32_t>(sizeof(HcclAiRMAWQ)), 0U, 0U, 0U};
    AscendC::DataCopyPadExtParams<uint8_t> sqInfoPadParams{false, 0U, 0U, 0U};
    AscendC::DataCopyPad(sqInfoTensor, sqInfoGlobalTensor, sqInfoParams, sqInfoPadParams);
}

__aicore__ inline void GetURMACqInfoTensor(const AscendC::LocalTensor<uint8_t> &cqInfoTensor, GM_ADDR hcclContext,
                                           uint32_t dstRankId)
{
    GM_ADDR cqInfoGMAddr = GetURMACqInfoGMAddr(hcclContext, dstRankId);
    ascendc_assert((cqInfoGMAddr != nullptr), "GetURMACqInfoGMAddr failed");

    AscendC::GlobalTensor<uint8_t> cqInfoGlobalTensor;
    cqInfoGlobalTensor.SetGlobalBuffer((__gm__ uint8_t *)(cqInfoGMAddr));
    AscendC::DataCopyExtParams cqInfoParams = {1U, static_cast<uint32_t>(sizeof(HcclAiRMACQ)), 0U, 0U, 0U};
    AscendC::DataCopyPadExtParams<uint8_t> cqInfoPadParams{false, 0U, 0U, 0U};
    AscendC::DataCopyPad(cqInfoTensor, cqInfoGlobalTensor, cqInfoParams, cqInfoPadParams);
}

__aicore__ inline void GenerateCommWriteSQE(const AscendC::LocalTensor<uint8_t> &sqeTensor, uint32_t sqeCount = 1)
{
    AscendC::Duplicate<uint8_t>(sqeTensor, 0, WRITE_SQE_SIZE); // 初始化为全0
    SyncFunc<AscendC::HardEvent::V_S>();
    sqeTensor(2) = 0b001;      // odr=0b001
    sqeTensor(3) = 0b10110000; // owner=1, rmt_jetty_type=0b01, token_en=1, nf=0
    sqeTensor(5) = 0x3;        // opcode=0x3
    sqeTensor(11) = 1;         // sge_num=1
}

__aicore__ inline void GenerateCommWriteWithNotifySQE(const AscendC::LocalTensor<uint8_t> &sqeTensor,
                                                      uint32_t sqeCount = 1)
{
    AscendC::Duplicate<uint8_t>(sqeTensor, 0, WRITE_WITH_NOTIFY_SQE_SIZE); // 初始化为全0
    SyncFunc<AscendC::HardEvent::V_S>();
    sqeTensor(2) = 0b001;      // odr=0b001
    sqeTensor(3) = 0b10110000; // owner=1, rmt_jetty_type=0b01, token_en=1, nf=0
    sqeTensor(5) = 0x5;        // opcode=0x5，使用write with notify SQE
    sqeTensor(11) = 1;         // sge_num=1
}

__aicore__ inline void UpdateCommWriteSQE(const AscendC::LocalTensor<uint8_t> &sqeTensor,
                                          const AscendC::LocalTensor<uint8_t> &sqInfoTensor)
{
    // 更新tp_id(24b), jetty_id(20b), rmt_eid(128b), rmt_token_value(32b), rmt_token_id(20b)

    AscendC::LocalTensor<uint32_t> sqInfoU32 = sqInfoTensor.ReinterpretCast<uint32_t>();
    AscendC::LocalTensor<uint32_t> templateSqeU32 = sqeTensor.ReinterpretCast<uint32_t>();
    templateSqeU32(2) = (1U << 24) + (sqInfoU32(12) & 0x00ffffff); // sge_num=1 tp_id
    templateSqeU32(3) = sqInfoU32(17);                             // rmt_jetty_or_seg_id
    templateSqeU32(4) = sqInfoU32(13);                             // rmt_eid
    templateSqeU32(5) = sqInfoU32(14);
    templateSqeU32(6) = sqInfoU32(15);
    templateSqeU32(7) = sqInfoU32(16);
    templateSqeU32(8) = sqInfoU32(18);  // rmt_token_value
    templateSqeU32(13) = sqInfoU32(17); // rmt_token_id
}

__aicore__ inline void UpdateCommWriteWithNotifySQE(const AscendC::LocalTensor<uint8_t> &sqeTensor,
                                                    const AscendC::LocalTensor<uint8_t> &sqInfoTensor)
{
    // 更新tp_id(24b), jetty_id(20b), rmt_eid(128b), rmt_token_value(32b), rmt_token_id(20b),
    // notify_token_value(32b), notify_token_id(20b)

    AscendC::LocalTensor<uint32_t> sqInfoU32 = sqInfoTensor.ReinterpretCast<uint32_t>();
    AscendC::LocalTensor<uint32_t> templateSqeU32 = sqeTensor.ReinterpretCast<uint32_t>();
    templateSqeU32(2) = (1U << 24) + (sqInfoU32(12) & 0x00ffffff); // sge_num=1 tp_id
    templateSqeU32(3) = sqInfoU32(17);                             // rmt_jetty_or_seg_id
    templateSqeU32(4) = sqInfoU32(13);                             // rmt_eid
    templateSqeU32(5) = sqInfoU32(14);
    templateSqeU32(6) = sqInfoU32(15);
    templateSqeU32(7) = sqInfoU32(16);
    templateSqeU32(8) = sqInfoU32(18);  // rmt_token_value
    templateSqeU32(21) = sqInfoU32(17); // rmt_token_id
    templateSqeU32(13) = sqInfoU32(18); // notify_token_value
    templateSqeU32(12) = sqInfoU32(17); // notify_token_id
}

__aicore__ inline void SetCommWriteSQE(const AscendC::LocalTensor<uint8_t> &sqeTensor, uint64_t dataAddr,
                                       uint64_t rmtAddr, uint32_t length, uint8_t cqe)
{
    // 设置cqe(1 bit), odr(3 bits), data_addr(64 bits), rmt_addr(64 bits), length(32 bits)

    AscendC::LocalTensor<uint32_t> sqeLocalU32 = sqeTensor.ReinterpretCast<uint32_t>();
    AscendC::LocalTensor<uint64_t> sqeLocalU64 = sqeTensor.ReinterpretCast<uint64_t>();

    sqeLocalU64(5) = rmtAddr;
    sqeLocalU64(7) = dataAddr;
    sqeLocalU32(12) = length;
    if (cqe > 0U) {
        // cqe=1, odr=0b010
        sqeTensor(2) |= 0b00100010;
        sqeTensor(2) &= 0b11111010;
    } else {
        // cqe=0, odr=0b001
        sqeTensor(2) &= 0b11011001;
        sqeTensor(2) |= 0b00000001;
    }
}

__aicore__ inline void SetCommWriteWithNotifySQE(const AscendC::LocalTensor<uint8_t> &sqeTensor, uint64_t dataAddr,
                                                 uint64_t rmtAddr, uint32_t length, uint64_t notifyAddr,
                                                 uint64_t notifyData, uint8_t cqe)
{
    // 设置cqe(1 bit), odr(3 bits), data_addr(64 bits), rmt_addr(64 bits), length(32 bits), notify_addr(64 bits),
    // notify_data(64 bits)

    AscendC::LocalTensor<uint32_t> sqeLocalU32 = sqeTensor.ReinterpretCast<uint32_t>();
    AscendC::LocalTensor<uint64_t> sqeLocalU64 = sqeTensor.ReinterpretCast<uint64_t>();

    sqeLocalU64(5) = rmtAddr;
    sqeLocalU64(11) = dataAddr;
    sqeLocalU32(20) = length;
    sqeLocalU64(7) = notifyAddr; // notify_addr
    sqeLocalU64(8) = notifyData; // notify_data

    if (cqe > 0U) {
        // cqe=1, odr=0b010
        sqeTensor(2) |= 0b00100010;
        sqeTensor(2) &= 0b11111010;
    } else {
        // cqe=0, odr=0b001
        sqeTensor(2) &= 0b11011001;
        sqeTensor(2) |= 0b00000001;
    }
}

__aicore__ inline void SendJFSDoorBell(const AscendC::LocalTensor<uint8_t> &jfsDoorBellTensor,
                                       const AscendC::LocalTensor<uint8_t> &sqInfoTensor, uint32_t sqPi)
{
    // // // 敲JFS DoorBell，更新硬件的sqPi
    // // // AscendC::LocalTensor<uint32_t> jfsDoorBellU32 = jfsDoorBellBuf_.Get<uint32_t>(1); // 1*sizeof(uint32_t)
    // // AscendC::LocalTensor<uint32_t> jfsDoorBellU32 = jfsDoorBellTensor.ReinterpretCast<uint32_t>(); //
    // // 1*sizeof(uint32_t) jfsDoorBellU32(0) = sqPi;

    // AscendC::GlobalTensor<uint32_t> jfsDoorBellGlobalTensor;
    // // // AscendC::LocalTensor<uint64_t> sqInfoU64 = urmaSqInfoBuf_.Get<uint64_t>();
    AscendC::LocalTensor<uint64_t> sqInfoU64 = sqInfoTensor.ReinterpretCast<uint64_t>();
    // jfsDoorBellGlobalTensor.SetGlobalBuffer((__gm__ uint32_t *)(sqInfoU64(5)));
    // // AscendC::DataCopyExtParams jfsDbParams = {1U, 4U, 0U, 0U, 0U}; // 4=1*sizeof(uint32_t)

    // // SyncFunc<AscendC::HardEvent::S_MTE3>(); // 等jfsDoorBellU32标量写
    // // AscendC::DataCopyPad(jfsDoorBellGlobalTensor, jfsDoorBellU32, jfsDbParams);
    // // // 缺jfsDoorBellGlobalTensor同步？
    // jfsDoorBellGlobalTensor(0) = sqPi;
    // AscendC::DataCacheCleanAndInvalid<uint32_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
    // AscendC::DcciDst::CACHELINE_OUT>(
    //     jfsDoorBellGlobalTensor);

    st_dev(sqPi, (__gm__ uint32_t *)(sqInfoU64(5)), 0);
}

__aicore__ inline void SendJFCDoorBell(const AscendC::LocalTensor<uint8_t> &jfcDoorBellTensor,
                                       const AscendC::LocalTensor<uint8_t> &cqInfoTensor, uint32_t cqCi)
{
    // // 敲JFC DoorBell，更新硬件的cqCi
    // AscendC::LocalTensor<uint32_t> jfcDoorBellU32 = jfcDoorBellTensor.ReinterpretCast<uint32_t>(); //
    // 2*sizeof(uint32_t)
    AscendC::LocalTensor<uint32_t> cqInfoU32 = cqInfoTensor.ReinterpretCast<uint32_t>();
    AscendC::LocalTensor<uint64_t> cqInfoU64 = cqInfoTensor.ReinterpretCast<uint64_t>();

    // jfcDoorBellU32(0) = cqCi;
    // jfcDoorBellU32(1) = cqInfoU32(0);

    // AscendC::GlobalTensor<uint32_t> jfcDoorBellGlobalTensor;
    // jfcDoorBellGlobalTensor.SetGlobalBuffer((__gm__ uint32_t *)(cqInfoU64(5)));
    // // AscendC::DataCopyExtParams jfcDbParams = {1U, 8U, 0U, 0U, 0U}; // 8=2*sizeof(uint32_t)

    // // SyncFunc<AscendC::HardEvent::S_MTE3>(); // 等jfsDoorBellU32标量写入完成
    // // AscendC::DataCopyPad(jfcDoorBellGlobalTensor, jfcDoorBellU32, jfcDbParams);
    // // // TODO 缺jfcDoorBellGlobalTensor同步？
    // jfcDoorBellGlobalTensor(0) = cqCi;
    // jfcDoorBellGlobalTensor(1) = cqInfoU32(0);
    // AscendC::DataCacheCleanAndInvalid<uint32_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
    // AscendC::DcciDst::CACHELINE_OUT>(
    //     jfcDoorBellGlobalTensor);

    uint64_t jfcDbValue = (static_cast<uint64_t>(cqInfoU32(0)) << 32) + static_cast<uint64_t>(cqCi);
    st_dev(jfcDbValue, (__gm__ uint64_t *)(cqInfoU64(5)), 0);
}

__aicore__ inline void PollCommCQUpdateSQCI(const AscendC::LocalTensor<uint8_t> &sqInfoTensor,
                                            const AscendC::LocalTensor<uint8_t> &cqInfoTensor,
                                            const AscendC::LocalTensor<uint8_t> &cqeTensor,
                                            const AscendC::LocalTensor<uint8_t> &jfcDoorBellTensor, uint32_t &outSqCi,
                                            uint32_t &outCqCi)
{
    // TODO 联调不带CQ
    // return;

    AscendC::LocalTensor<uint32_t> sqInfoU32 = sqInfoTensor.ReinterpretCast<uint32_t>();
    AscendC::LocalTensor<uint32_t> cqInfoU32 = cqInfoTensor.ReinterpretCast<uint32_t>();
    AscendC::LocalTensor<uint64_t> cqInfoU64 = cqInfoTensor.ReinterpretCast<uint64_t>();
    // AscendC::LocalTensor<uint8_t> cqeTensorU8 = cqeBuf_.Get<uint8_t>();
    uint32_t sqDepth = sqInfoU32(5);
    uint32_t cqeSize = cqInfoU32(4);
    uint32_t cqDepth = cqInfoU32(5);
    AscendC::GlobalTensor<uint8_t> cqGlobalTensor;
    cqGlobalTensor.SetGlobalBuffer((__gm__ uint8_t *)(cqInfoU64(1)));
    AscendC::DataCopyExtParams cqeParams = {1U, 8U, 0U, 0U, 0U};
    AscendC::DataCopyPadExtParams<uint8_t> cqePadParams{false, 0U, 0U, 0U};

    uint32_t pollTimes = 0;
    uint32_t newestCompletedPi = 0;
    while (pollTimes < cqDepth - 1) {
        AscendC::DataCopyPad(cqeTensor, cqGlobalTensor[cqeSize * outCqCi], cqeParams, cqePadParams);
        SyncFunc<AscendC::HardEvent::MTE2_S>();
        if (cqeTensor(3) == 0xff) {
            break;
        } else if (cqeTensor(3) != 0) {
            // 退出kernel并报错
            ascendc_assert(false, "CQE status is abnormal! status is %d, substatus is %d.\n", cqeTensor(3),
                           cqeTensor(2));
        }

        // status==0，处理当前CQE，从entry_idx获取对应WQE的sqPi
        newestCompletedPi = (static_cast<uint32_t>(cqeTensor(5)) << 8) + static_cast<uint32_t>(cqeTensor(4));
        // 把CQE的status设置为无效值，并写回CQ
        cqeTensor(3) = 0xff;
        SyncFunc<AscendC::HardEvent::S_MTE3>();
        AscendC::DataCopyPad(cqGlobalTensor[cqeSize * outCqCi], cqeTensor, cqeParams);
        // 防止cqeTensor被下一轮加载覆盖
        SyncFunc<AscendC::HardEvent::MTE3_MTE2>();

        // 递增本地的outCqCi
        outCqCi = (outCqCi + 1) % cqDepth;
        // ++cqCiLinear_;
        ++pollTimes;
    }

    // 更新本地的outSqCi
    if (pollTimes > 0) {
        outSqCi = (newestCompletedPi + 1) % sqDepth;
    }
    AscendC::PipeBarrier<PIPE_MTE3>();
    // 通过敲JFC DoorBell，更新硬件的cqCi
    SendJFCDoorBell(jfcDoorBellTensor, cqInfoTensor, outCqCi);
}

__aicore__ inline void PollNotifyCommCQUpdateSQCI(const AscendC::LocalTensor<uint8_t> &sqInfoTensor,
                                                  const AscendC::LocalTensor<uint8_t> &cqInfoTensor,
                                                  const AscendC::LocalTensor<uint8_t> &cqeTensor,
                                                  const AscendC::LocalTensor<uint8_t> &jfcDoorBellTensor,
                                                  uint32_t &outSqCi, uint32_t &outCqCi)
{
    // TODO 联调不带CQ
    // return;

    AscendC::LocalTensor<uint32_t> sqInfoU32 = sqInfoTensor.ReinterpretCast<uint32_t>();
    AscendC::LocalTensor<uint32_t> cqInfoU32 = cqInfoTensor.ReinterpretCast<uint32_t>();
    AscendC::LocalTensor<uint64_t> cqInfoU64 = cqInfoTensor.ReinterpretCast<uint64_t>();
    // AscendC::LocalTensor<uint8_t> cqeTensorU8 = cqeBuf_.Get<uint8_t>();
    uint32_t sqDepth = sqInfoU32(5);
    uint32_t cqeSize = cqInfoU32(4);
    uint32_t cqDepth = cqInfoU32(5);
    AscendC::GlobalTensor<uint8_t> cqGlobalTensor;
    cqGlobalTensor.SetGlobalBuffer((__gm__ uint8_t *)(cqInfoU64(1)));
    AscendC::DataCopyExtParams cqeParams = {1U, 8U, 0U, 0U, 0U};
    AscendC::DataCopyPadExtParams<uint8_t> cqePadParams{false, 0U, 0U, 0U};

    uint32_t pollTimes = 0;
    uint32_t newestCompletedPi = 0;
    while (pollTimes < cqDepth - 1) {
        AscendC::DataCopyPad(cqeTensor, cqGlobalTensor[cqeSize * outCqCi], cqeParams, cqePadParams);
        SyncFunc<AscendC::HardEvent::MTE2_S>();
        if (cqeTensor(3) == 0xff) {
            break;
        } else if (cqeTensor(3) != 0) {
            // 退出kernel并报错
            ascendc_assert(false, "CQE status is abnormal! status is %d, substatus is %d.\n", cqeTensor(3),
                           cqeTensor(2));
        }

        // status==0，处理当前CQE，从entry_idx获取对应WQE的sqPi
        newestCompletedPi = (static_cast<uint32_t>(cqeTensor(5)) << 8) + static_cast<uint32_t>(cqeTensor(4));
        // 把CQE的status设置为无效值，并写回CQ
        cqeTensor(3) = 0xff;
        SyncFunc<AscendC::HardEvent::S_MTE3>();
        AscendC::DataCopyPad(cqGlobalTensor[cqeSize * outCqCi], cqeTensor, cqeParams);
        // 防止cqeTensor被下一轮加载覆盖
        SyncFunc<AscendC::HardEvent::MTE3_MTE2>();

        // 递增本地的outCqCi
        outCqCi = (outCqCi + 1) % cqDepth;
        // ++cqCiLinear_;
        ++pollTimes;
    }

    // 更新本地的outSqCi
    if (pollTimes > 0) {
        outSqCi = (newestCompletedPi + 1) % sqDepth;
    }
    AscendC::PipeBarrier<PIPE_MTE3>(); // TODO 这个在等谁
    // 通过敲JFC DoorBell，更新硬件的cqCi
    SendJFCDoorBell(jfcDoorBellTensor, cqInfoTensor, outCqCi);
}

__aicore__ inline void InvalidateCqeStatus(const AscendC::LocalTensor<uint8_t> &cqeInfoTensor,
                                           const AscendC::LocalTensor<uint8_t> &cqeTensor)
{
    // 联调版本，HCCP无法提供CQ相关的信息，MC2算子先不处理CQ（cqCi_和sqCi_保持为0），kernel假设SQ的队列深度足够大
    // 在脚本侧，每隔几个MC2算子，需要主动销毁HCCL通信域并重新建链，避免SQ队列满，导致出错
    // return;

    // 为cqeBuf_申请256*32B空间，初始化HGM上的CQ空间时，如果cqDepth>256，则循环多次DataCopy
    AscendC::LocalTensor<uint64_t> cqInfoU64 = cqeInfoTensor.ReinterpretCast<uint64_t>();
    AscendC::LocalTensor<uint32_t> cqInfoU32 = cqeInfoTensor.ReinterpretCast<uint32_t>();

    AscendC::GlobalTensor<uint8_t> cqGlobalTensor;
    cqGlobalTensor.SetGlobalBuffer((__gm__ uint8_t *)(cqInfoU64(1)));
    uint32_t cqDepth = cqInfoU32(5);
    uint32_t cqeNum = (cqDepth > CQ_DEPTH_256) ? CQ_DEPTH_256 : cqDepth;

    AscendC::Duplicate<uint8_t>(cqeTensor, 0xff, UB_ALIGN * cqeNum); // 初始化为全1
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    // AscendC::Duplicate<uint8_t>(cqeTensor, 0, UB_ALIGN); // 初始化为全0
    // SyncFunc<AscendC::HardEvent::V_S>();
    // // for (uint32_t cqeIdx = 0; cqeIdx < cqeNum; ++cqeIdx) {
    // //     cqeTensor(UB_ALIGN * cqeIdx + 3) = 0xff; // 3: status在Normal CQE中的偏移
    // // }
    // cqeTensor(3) = 0xff;                    // 3: status在Normal CQE中的偏移
    // SyncFunc<AscendC::HardEvent::S_MTE3>(); // 等cqeTensor标量写入，后续Local向GM拷贝

    AscendC::DataCopyExtParams dataCopyParams = {static_cast<uint16_t>(cqeNum), UB_ALIGN, 0U,
                                                 NORMAL_CQE_SIZE - UB_ALIGN, 0U};
    uint32_t loopTimes = AscendC::Ceil(cqDepth, CQ_DEPTH_256);
    if (loopTimes > 1) {
        for (uint32_t i = 0; i < loopTimes - 1; ++i) {
            AscendC::DataCopyPad(cqGlobalTensor[NORMAL_CQE_SIZE * CQ_DEPTH_256 * i], cqeTensor, dataCopyParams);
        }
        uint32_t tailBlockCqeNum = cqDepth - CQ_DEPTH_256 * (loopTimes - 1);
        if (tailBlockCqeNum != CQ_DEPTH_256) {
            dataCopyParams = {static_cast<uint16_t>(tailBlockCqeNum), UB_ALIGN, 0U, NORMAL_CQE_SIZE - UB_ALIGN, 0U};
        }
    }
    AscendC::DataCopyPad(cqGlobalTensor[NORMAL_CQE_SIZE * CQ_DEPTH_256 * (loopTimes - 1)], cqeTensor, dataCopyParams);

    // AscendC::DataCopyExtParams dataCopyParams = {1U, 8U, 0U, 0U, 0U};
    // for (uint32_t i = 0; i < cqDepth; ++i) {
    //     AscendC::DataCopyPad(cqGlobalTensor[NORMAL_CQE_SIZE * i], cqeTensor, dataCopyParams);
    // }

    // TODO 缺cqGlobalTensor同步?
}

__aicore__ inline uint32_t GetAvailableSpace(const AscendC::LocalTensor<uint8_t> &sqInfoTensor, uint32_t &outSqPi,
                                             uint32_t &outSqCi)
{
    AscendC::LocalTensor<uint32_t> sqInfoU32 = sqInfoTensor.ReinterpretCast<uint32_t>();

    // 根据本地的outSqPi outSqCi 计算SQ可用空间大小（SQ不会用满，会预留一个WQE）
    uint32_t sqDepth = sqInfoU32(5);

    if (outSqPi == outSqCi) {
        return (sqDepth - 1);
    } else if ((outSqPi + 1) % sqDepth == outSqCi) {
        return 0;
    } else if (outSqPi > outSqCi) {
        return (sqDepth - (outSqPi - outSqCi) - 1);
    }
    return (outSqCi - outSqPi - 1);
}

__aicore__ inline void PutCommSQE(const AscendC::LocalTensor<uint8_t> &sqInfoTensor,
                                  const AscendC::LocalTensor<uint8_t> &cqInfoTensor,
                                  const AscendC::LocalTensor<uint8_t> &sqeTensor,
                                  const AscendC::LocalTensor<uint8_t> &cqeTensor,
                                  const AscendC::LocalTensor<uint8_t> &jfcDoorBellTensor, uint32_t sqeCount,
                                  uint32_t &outSqPi, uint32_t &outSqPiLinear, uint32_t &outSqCi, uint32_t &outCqCi)
{
    AscendC::LocalTensor<uint32_t> sqInfoU32 = sqInfoTensor.ReinterpretCast<uint32_t>();
    AscendC::LocalTensor<uint64_t> sqInfoU64 = sqInfoTensor.ReinterpretCast<uint64_t>();
    uint32_t sqeSize = sqInfoU32(4);
    uint32_t sqDepth = sqInfoU32(5);
    ascendc_assert(sqeCount < sqDepth, "too many SQE! SQE num[%d] should less than sqDepth[%d].", sqeCount, sqDepth);

    // 计算SQ可用空间大小
    uint32_t availableSpace = GetAvailableSpace(sqInfoTensor, outSqPi, outSqCi);
    while (availableSpace < sqeCount) {
        // 可用空间不足时，轮询CQ，更新本地的cqCi_ sqCi_和硬件的cqCi
        PollCommCQUpdateSQCI(sqInfoTensor, cqInfoTensor, cqeTensor, jfcDoorBellTensor, outSqCi, outCqCi);
        availableSpace = GetAvailableSpace(sqInfoTensor, outSqPi, outSqCi);
    }

    // 可用空间足够时，把WQE拷贝到HBM上的SQ中
    AscendC::GlobalTensor<uint8_t> sqGlobalTensor;
    sqGlobalTensor.SetGlobalBuffer((__gm__ uint8_t *)(sqInfoU64(1)));

    if (likely(sqDepth - outSqPi >= sqeCount)) {
        AscendC::DataCopy(sqGlobalTensor[sqeSize * outSqPi], sqeTensor, WRITE_SQE_SIZE * sqeCount);
    } else {
        uint32_t firstPartSize = sqDepth - outSqPi;
        uint32_t secondPartSize = sqeCount - firstPartSize;

        AscendC::DataCopy(sqGlobalTensor[sqeSize * outSqPi], sqeTensor, WRITE_SQE_SIZE * firstPartSize);
        AscendC::DataCopy(sqGlobalTensor, sqeTensor[WRITE_SQE_SIZE * firstPartSize], WRITE_SQE_SIZE * secondPartSize);
    }

    // 更新本地的outSqPi
    outSqPi = (outSqPi + sqeCount) % sqDepth;
    outSqPiLinear += sqeCount;
}

__aicore__ inline void
PutCommNotifySQE(const AscendC::LocalTensor<uint8_t> &sqInfoTensor, const AscendC::LocalTensor<uint8_t> &cqInfoTensor,
                 const AscendC::LocalTensor<uint8_t> &sqeTensor, const AscendC::LocalTensor<uint8_t> &cqeTensor,
                 const AscendC::LocalTensor<uint8_t> &jfcDoorBellTensor, uint32_t sqeCount, uint32_t &outSqPi,
                 uint32_t &outSqPiLinear, uint32_t &outSqCi, uint32_t &outCqCi)
{
    AscendC::LocalTensor<uint32_t> sqInfoU32 = sqInfoTensor.ReinterpretCast<uint32_t>();
    AscendC::LocalTensor<uint64_t> sqInfoU64 = sqInfoTensor.ReinterpretCast<uint64_t>();
    uint32_t sqeSize = sqInfoU32(4);
    uint32_t sqDepth = sqInfoU32(5);
    ascendc_assert((sqeCount << 1) < sqDepth, "too many SQE! SQE num[%d] should less than sqDepth[%d].",
                   (sqeCount << 1), sqDepth);

    // 计算SQ可用空间大小
    uint32_t availableSpace = GetAvailableSpace(sqInfoTensor, outSqPi, outSqCi);
    while (availableSpace < (sqeCount << 1)) {
        // 可用空间不足时，轮询CQ，更新本地的cqCi_ sqCi_和硬件的cqCi
        PollNotifyCommCQUpdateSQCI(sqInfoTensor, cqInfoTensor, cqeTensor, jfcDoorBellTensor, outSqCi, outCqCi);
        availableSpace = GetAvailableSpace(sqInfoTensor, outSqPi, outSqCi);
    }

    // 可用空间足够时，把WQE拷贝到HBM上的SQ中
    AscendC::GlobalTensor<uint8_t> sqGlobalTensor;
    sqGlobalTensor.SetGlobalBuffer((__gm__ uint8_t *)(sqInfoU64(1)));

    if (likely((sqDepth - outSqPi) >= (sqeCount << 1))) {
        AscendC::DataCopyParams intriParams{static_cast<uint16_t>(sqeCount), WRITE_WITH_NOTIFY_SQE_SIZE >> 5, 0, 1};
        AscendC::DataCopy(sqGlobalTensor[sqeSize * outSqPi], sqeTensor, intriParams);
    } else {
        uint32_t firstPartSize = (sqDepth - outSqPi) >> 1;
        uint32_t secondPartSize = sqeCount - firstPartSize;
        AscendC::DataCopyParams firstPartParams{static_cast<uint16_t>(firstPartSize), WRITE_WITH_NOTIFY_SQE_SIZE >> 5,
                                                0, 1};
        AscendC::DataCopy(sqGlobalTensor[sqeSize * outSqPi], sqeTensor, firstPartParams);

        if (((sqDepth - outSqPi) & 1) == 0) {
            AscendC::DataCopyParams secondPartParams{static_cast<uint16_t>(secondPartSize),
                                                     WRITE_WITH_NOTIFY_SQE_SIZE >> 5, 0, 1};
            AscendC::DataCopy(sqGlobalTensor, sqeTensor[WRITE_WITH_NOTIFY_SQE_SIZE * firstPartSize], secondPartParams);
        } else {
            // 如果剩余奇数个WQEBB，其中一个WQE会被切块分在尾和头
            AscendC::DataCopy(sqGlobalTensor[sqeSize * (sqDepth - 1)],
                              sqeTensor[WRITE_WITH_NOTIFY_SQE_SIZE * firstPartSize], sqeSize);
            AscendC::DataCopy(sqGlobalTensor, sqeTensor[WRITE_WITH_NOTIFY_SQE_SIZE * firstPartSize + sqeSize],
                              WRITE_WITH_NOTIFY_SQE_SIZE - sqeSize);

            AscendC::DataCopyParams secondPartParams{static_cast<uint16_t>(secondPartSize - 1),
                                                     WRITE_WITH_NOTIFY_SQE_SIZE >> 5, 0, 1};
            AscendC::DataCopy(sqGlobalTensor[sqeSize], sqeTensor[WRITE_WITH_NOTIFY_SQE_SIZE * (firstPartSize + 1)],
                              secondPartParams);
        }
    }

    // 更新本地的outSqPi
    outSqPi = (outSqPi + (sqeCount << 1)) % sqDepth;
    outSqPiLinear += (sqeCount << 1);
}

__aicore__ inline void GetPICI(GM_ADDR hcclContext, uint32_t curRankId, uint32_t dstRankId, uint32_t &outSqPi,
                               uint32_t &outSqCi, uint32_t &outCqPi, uint32_t &outCqCi, uint32_t &outSqPiLinear/*,
                               uint32_t &outCqCiLinear*/)
{
    ascendc_assert((hcclContext != nullptr && curRankId < MAX_RANK_NUM && dstRankId < MAX_RANK_NUM),
                   "hcclContext is nullptr or curRankId >= MAX_RANK_NUM or dstRankId >= MAX_RANK_NUM");

    GM_ADDR piCiSpaceGM = (GM_ADDR)(((__gm__ HcclCombinOpParam *)hcclContext)->windowsOut[curRankId] + WIN_PICI_OFFSET +
                                    WIN_ADDR_ALIGN * dstRankId);
    AscendC::GlobalTensor<uint32_t> piCiGlobalTensor;
    piCiGlobalTensor.SetGlobalBuffer((__gm__ uint32_t *)piCiSpaceGM);
    AscendC::DataCacheCleanAndInvalid<uint32_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(
        piCiGlobalTensor);

    outSqPi = piCiGlobalTensor(0);
    outSqCi = piCiGlobalTensor(1);
    outCqPi = piCiGlobalTensor(2);
    outCqCi = piCiGlobalTensor(3);
    outSqPiLinear = piCiGlobalTensor(4);
    // outIsFirst = (piCiGlobalTensor(4) > 0);
    // outSqPiLinear = outSqPi;
    // outCqCiLinear = outCqCi;
}

__aicore__ inline void GetIsFirstInComm(GM_ADDR hcclContext, uint32_t curRankId, uint32_t dstRankId, bool &outIsFirst)
{
    ascendc_assert((hcclContext != nullptr && curRankId < MAX_RANK_NUM && dstRankId < MAX_RANK_NUM),
                   "hcclContext is nullptr or curRankId >= MAX_RANK_NUM or dstRankId >= MAX_RANK_NUM");

    GM_ADDR piCiSpaceGM = (GM_ADDR)(((__gm__ HcclCombinOpParam *)hcclContext)->windowsOut[curRankId] + WIN_PICI_OFFSET +
                                    WIN_ADDR_ALIGN * dstRankId);
    AscendC::GlobalTensor<uint32_t> piCiGlobalTensor;
    piCiGlobalTensor.SetGlobalBuffer((__gm__ uint32_t *)piCiSpaceGM);
    AscendC::DataCacheCleanAndInvalid<uint32_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(
        piCiGlobalTensor);

    // outSqPi = piCiGlobalTensor(0);
    // outSqCi = piCiGlobalTensor(1);
    // outCqPi = piCiGlobalTensor(2);
    // outCqCi = piCiGlobalTensor(3);
    outIsFirst = (piCiGlobalTensor(5) > 0);
    // outSqPiLinear = outSqPi;
    // outCqCiLinear = outCqCi;
}

__aicore__ inline void UpdatePICI(GM_ADDR hcclContext, uint32_t curRankId, uint32_t dstRankId, uint32_t sqPi,
                                  uint32_t sqCi, uint32_t cqPi, uint32_t cqCi, uint32_t sqPiLinear)
{
    ascendc_assert((hcclContext != nullptr && curRankId < MAX_RANK_NUM && dstRankId < MAX_RANK_NUM),
                   "hcclContext is nullptr or curRankId >= MAX_RANK_NUM or dstRankId >= MAX_RANK_NUM");

    GM_ADDR piCiSpaceGM = (GM_ADDR)(((__gm__ HcclCombinOpParam *)hcclContext)->windowsOut[curRankId] + WIN_PICI_OFFSET +
                                    WIN_ADDR_ALIGN * dstRankId);
    AscendC::GlobalTensor<uint32_t> piCiGlobalTensor;
    piCiGlobalTensor.SetGlobalBuffer((__gm__ uint32_t *)piCiSpaceGM);

    piCiGlobalTensor(0) = sqPi;
    piCiGlobalTensor(1) = sqCi;
    piCiGlobalTensor(2) = cqPi;
    piCiGlobalTensor(3) = cqCi;
    piCiGlobalTensor(4) = sqPiLinear;
    // piCiGlobalTensor(4) = isFirst ? 1 : 0;
    AscendC::DataCacheCleanAndInvalid<uint32_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(
        piCiGlobalTensor);
}

__aicore__ inline void UpdateIsFirstInComm(GM_ADDR hcclContext, uint32_t curRankId, uint32_t dstRankId, bool isFirst)
{
    ascendc_assert((hcclContext != nullptr && curRankId < MAX_RANK_NUM && dstRankId < MAX_RANK_NUM),
                   "hcclContext is nullptr or curRankId >= MAX_RANK_NUM or dstRankId >= MAX_RANK_NUM");

    GM_ADDR piCiSpaceGM = (GM_ADDR)(((__gm__ HcclCombinOpParam *)hcclContext)->windowsOut[curRankId] + WIN_PICI_OFFSET +
                                    WIN_ADDR_ALIGN * dstRankId);
    AscendC::GlobalTensor<uint32_t> piCiGlobalTensor;
    piCiGlobalTensor.SetGlobalBuffer((__gm__ uint32_t *)piCiSpaceGM);

    // piCiGlobalTensor(0) = sqPi;
    // piCiGlobalTensor(1) = sqCi;
    // piCiGlobalTensor(2) = cqPi;
    // piCiGlobalTensor(3) = cqCi;
    piCiGlobalTensor(5) = isFirst ? 1 : 0;
    AscendC::DataCacheCleanAndInvalid<uint32_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(
        piCiGlobalTensor);
}

#endif // MOE_DISTRIBUTE_BASE_H
