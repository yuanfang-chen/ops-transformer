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

#if __has_include("../common/inc/kernel/mc2_kernel_utils.h")
#include "../common/inc/kernel/mc2_kernel_utils.h"
#else
#include "../../common/inc/kernel/mc2_kernel_utils.h"
#endif

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

struct HcclAiRMAWQ {
    uint32_t jettyId;
    uint64_t sqVA;
    uint32_t wqeSize;
    uint32_t sqDepth;
    uint64_t headAddr; // AIV无依赖
    uint64_t tailAddr; // AIV无依赖
    uint64_t dbAddr;
    uint32_t tp_id;
    uint8_t rmtEid[16];
    uint32_t rmtObjId; // rmtTokenID
    uint32_t rmtTokenValue;
    uint32_t localTokenId;
};

struct HcclAiRMACQ {
    uint32_t jfcId;
    uint64_t cqVA;
    uint32_t cqeSize;
    uint32_t cqDepth;
    uint64_t headAddr;
    uint64_t tailAddr;
    uint64_t dbAddr;
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
    AscendC::SyncFunc<AscendC::HardEvent::V_S>();
    sqeTensor(2) = 0b001;      // odr=0b001
    sqeTensor(3) = 0b10110000; // owner=1, rmt_jetty_type=0b01, token_en=1, nf=0
    sqeTensor(5) = 0x3;        // opcode=0x3
    sqeTensor(11) = 1;         // sge_num=1
}

__aicore__ inline void GenerateCommWriteWithNotifySQE(const AscendC::LocalTensor<uint8_t> &sqeTensor,
                                                      uint32_t sqeCount = 1)
{
    AscendC::Duplicate<uint8_t>(sqeTensor, 0, WRITE_WITH_NOTIFY_SQE_SIZE); // 初始化为全0
    AscendC::SyncFunc<AscendC::HardEvent::V_S>();
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

    // // AscendC::SyncFunc<AscendC::HardEvent::S_MTE3>(); // 等jfsDoorBellU32标量写
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

    // // AscendC::SyncFunc<AscendC::HardEvent::S_MTE3>(); // 等jfsDoorBellU32标量写入完成
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
                                            uint32_t &outCqCi, uint32_t &outCqCiLinear)
{
    // TODO 联调不带CQ
    // return;

    AscendC::LocalTensor<uint32_t> sqInfoU32 = sqInfoTensor.ReinterpretCast<uint32_t>();
    AscendC::LocalTensor<uint32_t> cqInfoU32 = cqInfoTensor.ReinterpretCast<uint32_t>();
    AscendC::LocalTensor<uint64_t> cqInfoU64 = cqInfoTensor.ReinterpretCast<uint64_t>();
    // AscendC::LocalTensor<uint8_t> cqeTensorU8 = cqeBuf_.Get<uint8_t>();
    uint32_t sqDepth = sqInfoU32(5) << 2;
    uint32_t cqeSize = cqInfoU32(4);
    uint32_t cqDepth = cqInfoU32(5) << 2;
    AscendC::GlobalTensor<uint8_t> cqGlobalTensor;
    cqGlobalTensor.SetGlobalBuffer((__gm__ uint8_t *)(cqInfoU64(1)));
    AscendC::DataCopyExtParams cqeParams = {1U, 8U, 0U, 0U, 0U};
    AscendC::DataCopyPadExtParams<uint8_t> cqePadParams{false, 0U, 0U, 0U};

    uint32_t pollTimes = 0;
    uint32_t newestCompletedPi = 0;
    while (pollTimes < cqDepth - 1) {
        AscendC::DataCopyPad(cqeTensor, cqGlobalTensor[cqeSize * outCqCi], cqeParams, cqePadParams);
        AscendC::SyncFunc<AscendC::HardEvent::MTE2_S>();
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
        AscendC::SyncFunc<AscendC::HardEvent::S_MTE3>();
        AscendC::DataCopyPad(cqGlobalTensor[cqeSize * outCqCi], cqeTensor, cqeParams);
        // 防止cqeTensor被下一轮加载覆盖
        AscendC::SyncFunc<AscendC::HardEvent::MTE3_MTE2>();

        // 递增本地的outCqCi
        outCqCi = (outCqCi + 1) % cqDepth;
        ++outCqCiLinear;
        ++pollTimes;
    }

    // 更新本地的outSqCi
    if (pollTimes > 0) {
        outSqCi = (newestCompletedPi + 1) % sqDepth;
    }
    // AscendC::PipeBarrier<PIPE_MTE3>();
    AscendC::SyncFunc<AscendC::HardEvent::MTE3_S>(); // TODO 这个在等谁
    // 通过敲JFC DoorBell，更新硬件的cqCi
    SendJFCDoorBell(jfcDoorBellTensor, cqInfoTensor, outCqCiLinear);
}

__aicore__ inline void PollNotifyCommCQUpdateSQCI(const AscendC::LocalTensor<uint8_t> &sqInfoTensor,
                                                  const AscendC::LocalTensor<uint8_t> &cqInfoTensor,
                                                  const AscendC::LocalTensor<uint8_t> &cqeTensor,
                                                  const AscendC::LocalTensor<uint8_t> &jfcDoorBellTensor,
                                                  uint32_t &outSqCi, uint32_t &outCqCi, uint32_t &outCqCiLinear)
{
    // TODO 联调不带CQ
    // return;

    AscendC::LocalTensor<uint32_t> sqInfoU32 = sqInfoTensor.ReinterpretCast<uint32_t>();
    AscendC::LocalTensor<uint32_t> cqInfoU32 = cqInfoTensor.ReinterpretCast<uint32_t>();
    AscendC::LocalTensor<uint64_t> cqInfoU64 = cqInfoTensor.ReinterpretCast<uint64_t>();
    // AscendC::LocalTensor<uint8_t> cqeTensorU8 = cqeBuf_.Get<uint8_t>();
    uint32_t sqDepth = sqInfoU32(5) << 2;
    uint32_t cqeSize = cqInfoU32(4);
    uint32_t cqDepth = cqInfoU32(5) << 2;
    AscendC::GlobalTensor<uint8_t> cqGlobalTensor;
    cqGlobalTensor.SetGlobalBuffer((__gm__ uint8_t *)(cqInfoU64(1)));
    AscendC::DataCopyExtParams cqeParams = {1U, 8U, 0U, 0U, 0U};
    AscendC::DataCopyPadExtParams<uint8_t> cqePadParams{false, 0U, 0U, 0U};

    uint32_t pollTimes = 0;
    uint32_t newestCompletedPi = 0;
    while (pollTimes < cqDepth - 1) {
        AscendC::DataCopyPad(cqeTensor, cqGlobalTensor[cqeSize * outCqCi], cqeParams, cqePadParams);
        AscendC::SyncFunc<AscendC::HardEvent::MTE2_S>();
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
        AscendC::SyncFunc<AscendC::HardEvent::S_MTE3>();
        AscendC::DataCopyPad(cqGlobalTensor[cqeSize * outCqCi], cqeTensor, cqeParams);
        // 防止cqeTensor被下一轮加载覆盖
        AscendC::SyncFunc<AscendC::HardEvent::MTE3_MTE2>();

        // 递增本地的outCqCi
        outCqCi = (outCqCi + 1) % cqDepth;
        ++outCqCiLinear;
        ++pollTimes;
    }

    // 更新本地的outSqCi
    if (pollTimes > 0) {
        outSqCi = (newestCompletedPi + 1) % sqDepth;
    }
    // AscendC::PipeBarrier<PIPE_MTE3>(); // TODO 这个在等谁
    AscendC::SyncFunc<AscendC::HardEvent::MTE3_S>();
    // 通过敲JFC DoorBell，更新硬件的cqCi
    SendJFCDoorBell(jfcDoorBellTensor, cqInfoTensor, outCqCiLinear);
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
    uint32_t cqDepth = cqInfoU32(5) << 2;
    uint32_t cqeNum = (cqDepth > CQ_DEPTH_256) ? CQ_DEPTH_256 : cqDepth;

    AscendC::Duplicate<uint8_t>(cqeTensor, 0xff, UB_ALIGN * cqeNum); // 初始化为全1
    AscendC::SyncFunc<AscendC::HardEvent::V_MTE3>();
    // AscendC::Duplicate<uint8_t>(cqeTensor, 0, UB_ALIGN); // 初始化为全0
    // AscendC::SyncFunc<AscendC::HardEvent::V_S>();
    // // for (uint32_t cqeIdx = 0; cqeIdx < cqeNum; ++cqeIdx) {
    // //     cqeTensor(UB_ALIGN * cqeIdx + 3) = 0xff; // 3: status在Normal CQE中的偏移
    // // }
    // cqeTensor(3) = 0xff;                    // 3: status在Normal CQE中的偏移
    // AscendC::SyncFunc<AscendC::HardEvent::S_MTE3>(); // 等cqeTensor标量写入，后续Local向GM拷贝

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
    uint32_t sqDepth = sqInfoU32(5) << 2;

    if (outSqPi == outSqCi) {
        return (sqDepth - 1);
    } else if ((outSqPi + 1) % sqDepth == outSqCi) {
        return 0;
    } else if (outSqPi > outSqCi) {
        return (sqDepth - (outSqPi - outSqCi) - 1);
    }
    return (outSqCi - outSqPi - 1);
}

__aicore__ inline void
PutCommSQE(const AscendC::LocalTensor<uint8_t> &sqInfoTensor, const AscendC::LocalTensor<uint8_t> &cqInfoTensor,
           const AscendC::LocalTensor<uint8_t> &sqeTensor, const AscendC::LocalTensor<uint8_t> &cqeTensor,
           const AscendC::LocalTensor<uint8_t> &jfcDoorBellTensor, uint32_t sqeCount, uint32_t &outSqPi,
           uint32_t &outSqPiLinear, uint32_t &outSqCi, uint32_t &outCqCi, uint32_t &outCqCiLinear)
{
    AscendC::LocalTensor<uint32_t> sqInfoU32 = sqInfoTensor.ReinterpretCast<uint32_t>();
    AscendC::LocalTensor<uint64_t> sqInfoU64 = sqInfoTensor.ReinterpretCast<uint64_t>();
    uint32_t sqeSize = sqInfoU32(4);
    uint32_t sqDepth = sqInfoU32(5) << 2;
    ascendc_assert(sqeCount < sqDepth, "too many SQE! SQE num[%d] should less than sqDepth[%d].", sqeCount, sqDepth);

    // 计算SQ可用空间大小
    uint32_t availableSpace = GetAvailableSpace(sqInfoTensor, outSqPi, outSqCi);
    while (availableSpace < sqeCount) {
        // 可用空间不足时，轮询CQ，更新本地的cqCi_ sqCi_和硬件的cqCi
        PollCommCQUpdateSQCI(sqInfoTensor, cqInfoTensor, cqeTensor, jfcDoorBellTensor, outSqCi, outCqCi, outCqCiLinear);
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
    outSqPiLinear += sqeCount;
    outSqPi = outSqPiLinear % sqDepth;
}

__aicore__ inline void
PutCommNotifySQE(const AscendC::LocalTensor<uint8_t> &sqInfoTensor, const AscendC::LocalTensor<uint8_t> &cqInfoTensor,
                 const AscendC::LocalTensor<uint8_t> &sqeTensor, const AscendC::LocalTensor<uint8_t> &cqeTensor,
                 const AscendC::LocalTensor<uint8_t> &jfcDoorBellTensor, uint32_t sqeCount, uint32_t &outSqPi,
                 uint32_t &outSqPiLinear, uint32_t &outSqCi, uint32_t &outCqCi, uint32_t &outCqCiLinear)
{
    AscendC::LocalTensor<uint32_t> sqInfoU32 = sqInfoTensor.ReinterpretCast<uint32_t>();
    AscendC::LocalTensor<uint64_t> sqInfoU64 = sqInfoTensor.ReinterpretCast<uint64_t>();
    uint32_t sqeSize = sqInfoU32(4);
    uint32_t sqDepth = sqInfoU32(5) << 2;
    ascendc_assert((sqeCount << 1) < sqDepth, "too many SQE! SQE num[%d] should less than sqDepth[%d].",
                   (sqeCount << 1), sqDepth);

    // 计算SQ可用空间大小
    uint32_t availableSpace = GetAvailableSpace(sqInfoTensor, outSqPi, outSqCi);
    while (availableSpace < (sqeCount << 1)) {
        // 可用空间不足时，轮询CQ，更新本地的cqCi_ sqCi_和硬件的cqCi
        PollNotifyCommCQUpdateSQCI(sqInfoTensor, cqInfoTensor, cqeTensor, jfcDoorBellTensor, outSqCi, outCqCi,
                                   outCqCiLinear);
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
    outSqPiLinear += (sqeCount << 1);
    outSqPi = outSqPiLinear % sqDepth;
}

__aicore__ inline void GetPICI(GM_ADDR hcclContext, uint32_t curRankId, uint32_t dstRankId, uint32_t &outSqPi,
                               uint32_t &outSqCi, uint32_t &outCqPi, uint32_t &outCqCi, uint32_t &outSqPiLinear,
                               uint32_t &outCqCiLinear)
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
    outCqCiLinear = piCiGlobalTensor(5);
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
    outIsFirst = (piCiGlobalTensor(6) > 0);
    // outSqPiLinear = outSqPi;
    // outCqCiLinear = outCqCi;
}

__aicore__ inline void UpdatePICI(GM_ADDR hcclContext, uint32_t curRankId, uint32_t dstRankId, uint32_t sqPi,
                                  uint32_t sqCi, uint32_t cqPi, uint32_t cqCi, uint32_t sqPiLinear, uint32_t cqCiLinear)
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
    piCiGlobalTensor(5) = cqCiLinear;
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
    piCiGlobalTensor(6) = isFirst ? 1 : 0;
    AscendC::DataCacheCleanAndInvalid<uint32_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(
        piCiGlobalTensor);
}

#endif // MOE_DISTRIBUTE_BASE_H
