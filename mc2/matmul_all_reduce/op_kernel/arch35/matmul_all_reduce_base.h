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
 * \file matmul_all_reduce_base.h
 * \brief
 */
#ifndef MATMUL_ALL_REDUCE_BASE_H
#define MATMUL_ALL_REDUCE_BASE_H

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "lib/matmul_intf.h"
#include "../common.h"
#include "matmul_all_reduce_add_x3.h"
#include "matmul_all_reduce_tiling_struct_ar35.h"
#include "../../common/inc/kernel/reduce_sum.h"

namespace MatmulAllReduceImpl {
using namespace AscendC;
using namespace AiVReduceSumImpl;
constexpr uint32_t PERTILE_MIXED_MAX_HANDLE_ID_NUM = 16;
template <typename XType, typename YType, Mc2CoreType CoreType>
class MatmulAllReduceBase
{
public:
    __aicore__ inline MatmulAllReduceBase(
        MC2GmAddrs* addrs, QuantGmAddrs* quantAddrs, ArnGmAddrs* arnAddrs, MC2TilingHeader* tilingData, TPipe* tPipe)
        : addrs_(addrs), quantAddrs_(quantAddrs), arnAddrs_(arnAddrs), tilingData_(tilingData), tPipe_(tPipe)
    {
        if constexpr (CoreType == Mc2CoreType::ON_CUBE) {
            notifyFlag_ = (GetBlockIdx() == 0);
        } else {
            notifyFlag_ = (g_coreType == AscendC::AIV && GetBlockIdx() == 0);
        }
        if (tilingData->allReduceBasedAtaSumAg){
            notifyFlag_ = (g_coreType == AscendC::AIV && GetBlockIdx() == 0);
        }
        paramInTiling_ = &tilingData->param;
        rankNum_ = paramInTiling_->rankDim;
    }

    __aicore__ inline void Init()
    {
        hccl_.InitV2(GetHcclContext<0>(), tilingData_);
        hccl_.SetCcTilingV2(offsetof(MC2TilingHeader, mc2CcTiling));
        if (tilingData->allReduceBasedAtaSumAg){
            hccl_.SetCcTilingV2(offsetof(MC2TilingHeader, mc2CcTilingComm));
        }
        __gm__ HcclCombinOpParam* context = (__gm__ HcclCombinOpParam*)(GetHcclContext<0>());
        OOMInit(context);
#if defined(__DAV_C310__)
        addrs_->cGM = addrs_->workspaceGM + paramInTiling_->nd2NzWorkLen + paramInTiling_->biasLen;
#else
        if (msgInTiling_->useBufferType == MC2_BUFFER_TYPE::MC2_BUFFER_TYPE_WINDOW_IN &&
            context->config.determinism != 1) {
            addrs_->cGM = hccl_.GetWindowsInAddr(hccl_.GetRankId());
        }
#endif
        addFlag_ = (paramInTiling_->isAdd != 0U);
        tailFlag_ = (paramInTiling_->tailCnt != 0U);
        isOneTileFlag_ = (paramInTiling_->tileCnt == 1U) && (paramInTiling_->tailCnt == 0U);
        const uint64_t mVal = isOneTileFlag_ ? ((uint64_t)paramInTiling_->rankM) : (uint64_t)tileInfo_.mmTiling->M;

        tileInfo_.aOffset = mVal * (uint64_t)tileInfo_.mmTiling->Ka;
        if (AscendC::IsSameType<XType, fp4x2_e2m1_t>::value) {
            // In 4-bits scenario, the data length is 0.5, the size of Xtype is 1, it should be divided by 2.
            tileInfo_.aAddrOffset = tileInfo_.aOffset * sizeof(XType) / 2;
        } else {
            tileInfo_.aAddrOffset = tileInfo_.aOffset * sizeof(XType);
        }
        tileInfo_.cOffset = mVal * (uint64_t)tileInfo_.mmTiling->N;
        tileInfo_.cAddrOffset = tileInfo_.cOffset * sizeof(YType);
        if (tailFlag_) {
            tailInfo_.aOffset = (uint64_t)tailInfo_.mmTiling->M * (uint64_t)tailInfo_.mmTiling->Ka;
            if (AscendC::IsSameType<XType, fp4x2_e2m1_t>::value || AscendC::IsSameType<XType, fp4x2_e1m2_t>::value) {
                // In 4-bits scenario, the data length is 0.5, the size of Xtype is 1, it should be divided by 2.
                tailInfo_.aAddrOffset = tailInfo_.aOffset * sizeof(XType) / 2;
            } else {
                tailInfo_.aAddrOffset = tailInfo_.aOffset * sizeof(XType);
            }
            tailInfo_.cOffset = (uint64_t)tailInfo_.mmTiling->M * (uint64_t)tailInfo_.mmTiling->N;
            tailInfo_.cAddrOffset = tailInfo_.cOffset * sizeof(YType);
        }
        if (!tilingData->allReduceBasedAtaSumAg){
            if (notifyFlag_) {
                tileInfo_.hcclHandleId = hccl_.AllReduce(
                    addrs_->cGM, addrs_->outputGM, tileInfo_.cOffset, HCCL_DATA_TYPE, AscendC::HCCL_REDUCE_SUM,
                    paramInTiling_->tileCnt);
                if (tailFlag_) {
                    const uint64_t offset = tileInfo_.cAddrOffset * paramInTiling_->tileCnt;
                    tailInfo_.hcclHandleId = hccl_.AllReduce(
                        addrs_->cGM + offset, addrs_->outputGM + offset, tailInfo_.cOffset, HCCL_DATA_TYPE,
                        AscendC::HCCL_REDUCE_SUM, paramInTiling_->tailCnt);
                }
            }
        } else {
            cgmAddr_ = tileInfo_.cAddrOffset * paramInTiling_->tileCnt + tailInfo_.cAddrOffset * paramInTiling_->tailCnt;
            cgmLen_ = tileInfo_.cOffset * paramInTiling_->tileCnt + tailInfo_.cOffset * paramInTiling_->tailCnt;
            
            all2allInGM_ = addrs_->cGM;
            all2allOutGM_ = all2allInGM_ + cgmAddr_;
            reduceSumInGM_ = all2allOutGM_;
            reduceSumOutGM_ = reduceSumInGM_ + cgmAddr_;
            allGatherInGM_ = reduceSumOutGM_;
            allGatherOutGM_ = addrs->outputGM;
            PrePareHCCL();
        }
    }

    __aicore__ inline void PrePareHCCL(){
        for (uint32_t i = 0U; i < paramInTiling_->tileCnt; i++){
            const uint64_t alltoallIndexOffsetTile = tileInfo_.cAddrOffset * i;
            const uint64_t allgatherIndexOffsetTile = alltoallIndexOffsetTile;
            all2allSendGM_[i] = all2allInGM_ + alltoallIndexOffsetTile;
            all2allRecvGM_[i] = all2allOutGM_ + alltoallIndexOffsetTile;
            allgatherSendGM_[i] = allGatherInGM_ + allgatherIndexOffsetTile / rankNum_;
            allgatherRecvGM_[i] = allgatherOutGM_ + allgatherIndexOffsetTile;

            all2allHandleId_[i] = hccl_.AlltoAll<false>(
                all2allSendGM_[i], all2allRecvGM_[i], tileInfo_.cOffset / rankNum_, HCCL_DATA_TYPE);
        }

        for (uint32_t i = 0U; i < paramInTiling_->tailCnt; i++){
            const uint64_t alltoallIndexOffsetTile = tileInfo_.cAddrOffset * paramInTiling_->tileCnt + tailInfo_.cAddrOffset * i;
            const uint64_t allgatherIndexOffsetTile = alltoallIndexOffsetTile;
            const uint64_t index = paramInTiling_->tileCnt + i;
            all2allSendGM_[index] = all2allInGM_ + alltoallIndexOffsetTile;
            all2allRecvGM_[index] = all2allOutGM_ + alltoallIndexOffsetTile;
            allgatherSendGM_[index] = allGatherInGM_ + allgatherIndexOffsetTile / rankNum_;
            allgatherRecvGM_[index] = allgatherOutGM_ + allgatherIndexOffsetTile;

            all2allHandleId_[index] = hccl_.AlltoAll<false>(
                all2allSendGM_[index], all2allRecvGM_[index], tailInfo_.cOffset / rankNum_, HCCL_DATA_TYPE);
        }

        for (uint32_t i = 0U; i < paramInTiling_->tileCnt; i++){
            allgatherHandleId_[i] = hccl_.AllGather<false>(
                allgatherSendGM_[i], allgatherRecvGM_[i], tileInfo_.cOffset / rankNum_, HCCL_DATA_TYPE, 0, 1);
        }

        for (uint32_t i = 0U; i < paramInTiling_->tailCnt; i++){
            const uint64_t index = paramInTiling_->tileCnt + i;
            allgatherHandleId_[index] = hccl_.AllGather<false>(
                allgatherSendGM_[index], allgatherRecvGM_[index], tailInfo_.cOffset / rankNum_, HCCL_DATA_TYPE, 0, 1);
        }
    }

protected:
#if (ORIG_DTYPE_X1 == DT_BF16)
    __aicore__ inline void PreProcForBiasOnVector()
    {
        if (paramInTiling_->biasLen == 0U) {
            return;
        }

        TBuf<TPosition::VECCALC> tmpBuf;
        tPipe_->InitBuffer(tmpBuf, TOTAL_UB_SIZE);
        CastBFtoFloatOnAiv0(addrs_->workspaceGM, addrs_->biasGM, paramInTiling_->rankN, tmpBuf);
        SyncAll<false>();
        addrs_->biasGM = addrs_->workspaceGM;
        addrs_->workspaceGM += paramInTiling_->biasLen;
    }
#endif

    __aicore__ inline void PostProcEachTurn(AscendC::HcclHandle handleId, uint64_t aOffset, uint64_t cOffset, const uint64_t index)
    {
        if (addFlag_ && addrs_->cGM != addrs_->addGM) {
            if (tilingData->allReduceBasedAtaSumAg){
                SyncAll<false>();
            } else {
                Mc2SyncAll<CoreType>();
            }
            MatmulAllReduceAddX3Kernel<YType>(
                addrs_->cGM, addrs_->addGM, cOffset / sizeof(YType), paramInTiling_->addX3UbCnt, tPipe_);
            addrs_->addGM += cOffset;
        }

        addrs_->aGM += aOffset;
        addrs_->cGM += cOffset;
        if (tilingData->allReduceBasedAtaSumAg){
            SyncAll<false>();
            if (notifyFlag_) {
                hccl_.Commit(all2allHandleId_[index]);
            }
        } else {
            Mc2SyncAll<CoreType>();
            if (notifyFlag_) {
                hccl_.Commit(handleId);
            }
        }
        
    }
    __aicore__ inline void WaitAlltoAllEachTurn(bool tailFlag, uint32_t turnCnt){
        if (notifyFlag_) {
            for (uint32_t i = 0U; i < turnCnt; ++i) {
                const uint64_t index = tailFlag ? i + paramInTiling_->tileCnt : i;
                hccl_.Wait(all2allHandleId_[index]);
            }
        }
        SyncAll();
    }

    __aicore__ inline void ReduceSumAndAllGather()
    {
        if ASCEND_IS_AIV {
            for (int i = 0; i < paramInTiling_->tileCnt; i++){
                tPipe_->Reset();
                reduceSum_.Init(tileInfo_.cOffset, rankNum_, aivNum, reduceSumInGM_, reduceSumOutGM_, tPipe_);
                reduceSum_.ExecuteReduceSum();
                reduceSumInGM_ += tileInfo_.cAddrOffset;
                reduceSumOutGM_ += tileInfo_.cAddrOffset / rankNum_;
                SyncAll();
            }

            for (int i = 0; i < paramInTiling_->tailCnt; i++){
                tPipe_->Reset();
                reduceSum_.Init(tailInfo_.cOffset, rankNum_, aivNum, reduceSumInGM_, reduceSumOutGM_, tPipe_);
                reduceSum_.ExecuteReduceSum();
                reduceSumInGM_ += tailInfo_.cAddrOffset;
                reduceSumOutGM_ += tailInfo_.cAddrOffset / rankNum_;
                SyncAll();
            }
        }
        SyncAll();
        if (notifyFlag_) {
            for (int i = 0; i < paramInTiling_->tileCnt + paramInTiling_->tailCnt; i++) {
                hccl_.Commit(allgatherHandleId_[i]);
            }
        }
    }

    __aicore__ inline void HcclFinalize()
    {

        if (notifyFlag_) {
            if (tilingData->allReduceBasedAtaSumAg){
                for (int i = 0; i < paramInTiling_->tileCnt + paramInTiling_->tailCnt; i++) {
                    hccl_.Wait(allgatherHandleId_[i]);
                }
            } else {
                for (uint32_t i = 0; i < paramInTiling_->tileCnt; ++i) {
                    hccl_.Wait(tileInfo_.hcclHandleId);
                }
                if (tailFlag_) {
                    for (uint32_t i = 0; i < paramInTiling_->tailCnt; ++i) {
                        hccl_.Wait(tailInfo_.hcclHandleId);
                    }
                }
            }
        }

        if (tilingData->allReduceBasedAtaSumAg){
            SyncAll();
        } else {
            Mc2SyncAll<CoreType>();
        }

        if (notifyFlag_) {
            hccl_.Finalize();
        }
    }
    uint32_t rankNum_ = 0UL;
    uint32_t cgmLen_ = 0UL;
    uint32_t cgmAddr_ = 0UL;
    MC2GmAddrs* addrs_;
    QuantGmAddrs* quantAddrs_;
    ArnGmAddrs* arnAddrs_;
    Mc2Tiling::Mc2Msg* msgInTiling_;
    Mc2Tiling::RCSTiling* paramInTiling_;
    MC2TileInfo tileInfo_, tailInfo_;
    MC2TilingHeader* tilingData_;
    TPipe* tPipe_;
    Hccl<HcclServerType::HCCL_SERVER_TYPE_CCU> hccl_;
    bool notifyFlag_;
    bool addFlag_;
    bool tailFlag_;
    bool isOneTileFlag_;
    
    AscendC::HcclHandle all2allHandleId_[PERTILE_MIXED_MAX_HANDLE_ID_NUM] = {0};
    AscendC::HcclHandle allgatherHandleId_[PERTILE_MIXED_MAX_HANDLE_ID_NUM] = {0};
    GM_ADDR all2allSendGM_[PERTILE_MIXED_MAX_HANDLE_ID_NUM] = {0};
    GM_ADDR all2allRecvGM_[PERTILE_MIXED_MAX_HANDLE_ID_NUM] = {0};
    GM_ADDR allgatherSendGM_[PERTILE_MIXED_MAX_HANDLE_ID_NUM] = {0};
    GM_ADDR allgatherRecvGM_[PERTILE_MIXED_MAX_HANDLE_ID_NUM] = {0};
    GM_ADDR all2allInGM_;
    GM_ADDR all2allOutGM_;
    GM_ADDR reduceSumInGM_;
    GM_ADDR reduceSumOutGM_;
    GM_ADDR allGatherInGM_;
    GM_ADDR allGatherOutGM_;
};
} // namespace MatmulAllReduceImpl
#endif // MATMUL_ALL_REDUCE_BASE_H