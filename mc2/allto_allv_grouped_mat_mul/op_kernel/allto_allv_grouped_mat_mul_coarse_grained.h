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
 * \file allto_allv_grouped_mat_mul_coarse_grained.h
 * \brief
 */
#ifndef ALL_TO_ALL_V_GROUPED_MAT_MUL_H
#define ALL_TO_ALL_V_GROUPED_MAT_MUL_H

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "adv_api/hccl/hccl.h"
#include "allto_allv_gmm.h"
#include "lib/matmul_intf.h"
#include "allto_allv_grouped_mat_mul_tiling.h"
namespace AscendC {
using namespace ALLTO_ALLV_GMM;

template <AscendC::HardEvent event>
__aicore__ static inline void SyncFunc()
{
    int32_t eventID = static_cast<int32_t>(GetTPipePtr()->FetchEventID(event));
    AscendC::SetFlag<event>(eventID);
    AscendC::WaitFlag<event>(eventID);
}

template <typename DataType, bool IsNeedMM, bool IsTranGmmW, bool IsTranMmW>
class AlltoAllvGmmCoarseGrained {
public:
    __aicore__ inline AlltoAllvGmmCoarseGrained()
    {
    }
    __aicore__ inline void Init(GM_ADDR gmmxGM, GM_ADDR gmmweightGM, GM_ADDR sendCountsTensorOptionalGM,
                                GM_ADDR recvCountsTensorOptionalGM, GM_ADDR mmxOptionalGM, GM_ADDR mmweightOptionalGM,
                                GM_ADDR gmmyGM, GM_ADDR mmyOptionalGM, GM_ADDR permuteOutOptionalGM,
                                GM_ADDR workspaceGM, GM_ADDR contextGM, const AlltoAllvGmmTilingData *tilingData,
                                __gm__ void *hcclInitTiling, __gm__ void *alltoAllvCcTiling, TPipe *tPipe);
    __aicore__ inline void Process();

private:
    using aType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DataType, false>;
    using gmmBType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DataType, IsTranGmmW>;
    using mmBType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DataType, IsTranMmW>;
    using cType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DataType, false>;
    using biasType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DataType>;
    using gmmType = MMImplType<aType, gmmBType, cType, biasType, CFG_MDL>;
    using mmType = MMImplType<aType, mmBType, cType, biasType, CFG_MDL>;

    __aicore__ inline void CalcMatmul();
    __aicore__ inline void HcclAlltoAllvPrepare();
    __aicore__ inline void HcclAlltoAllvExec();
    __aicore__ inline void HcclFinalize();
    __aicore__ inline void DoPermuted(uint32_t eGroup, uint64_t *recvSumCntBefore, uint64_t *recvSumCntAfter);
    __aicore__ inline void SplitToCore(uint32_t curSendCnt, uint32_t &startTokenId,
                                        uint32_t &endTokenId, uint32_t &sendTokenNum);

    GM_ADDR gmmxGM_ = nullptr;
    GM_ADDR gmmwGM_ = nullptr;
    GM_ADDR sendCntsGM_ = nullptr;
    GM_ADDR recvCntsGM_ = nullptr;
    GM_ADDR mmxGM_ = nullptr;
    GM_ADDR mmwGM_ = nullptr;
    GM_ADDR gmmyGM_ = nullptr;
    GM_ADDR mmyGM_ = nullptr;
    GM_ADDR permuteOutGM_ = nullptr;
    const AlltoAllvGmmTilingData *tilingData_ = nullptr;
    uint32_t rankId_ = 0U;             // 当前卡ID
    uint32_t rankDim_ = 8U;            // 通信域内卡的数量
    uint32_t expertNumInOneRank_ = 0U; // 单卡上面的专家个数
    uint32_t expertNumAll_ = 0U;       // 通信域内专家个数
    uint32_t expertNumInGroup_ = 0U;
    uint32_t groupNum_ = 0U;
    uint32_t totalUbSize_ = 0U;
    uint32_t totalUbElm_ = 0U;
    uint32_t aivNum_ = 0U;
    uint32_t aivId_ = 0U;

    // alltoall 流程数据结构
    static constexpr uint64_t MAX_HANDLE_ID_NUM = 64U;
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
    Hccl<HcclServerType::HCCL_SERVER_TYPE_CCU> hccl_;
#else
    Hccl<HCCL_SERVER_TYPE_AICPU> hccl_;
#endif
    HcclDataType hcclDataType_ = HCCL_DATA_TYPE_FP16;
    HcclHandle alltoAllvHandleId_[MAX_HANDLE_ID_NUM] = {INVALID_HANDLE_ID};

    GlobalTensor<DataType> gmmxGMTensor_;
    GlobalTensor<DataType> permutedGMTensor_;
    GlobalTensor<DataType> workspaceGMTensor_;

    TBuf<> tBuf_;

    uint64_t axisH1_;
    uint64_t axisN1_;

    typename gmmType::MT gmm_;
    typename mmType::MT mm_;
};

template <typename DataType, bool IsNeedMM, bool IsTranGmmW, bool IsTranMmW>
__aicore__ inline void AlltoAllvGmmCoarseGrained<DataType, IsNeedMM, IsTranGmmW, IsTranMmW>::Init(
    GM_ADDR gmmxGM, GM_ADDR gmmweightGM, GM_ADDR sendCountsTensorOptionalGM, GM_ADDR recvCountsTensorOptionalGM,
    GM_ADDR mmxOptionalGM, GM_ADDR mmweightOptionalGM, GM_ADDR gmmyGM, GM_ADDR mmyOptionalGM,
    GM_ADDR permuteOutOptionalGM, GM_ADDR workspaceGM, GM_ADDR contextGM, const AlltoAllvGmmTilingData *tilingData,
    __gm__ void *hcclInitTiling, __gm__ void *alltoAllvCcTiling, TPipe *tPipe)
{
    printf("[PRINT] in kernel\n");
    gmmxGM_ = gmmxGM;
    gmmwGM_ = gmmweightGM;
    sendCntsGM_ = sendCountsTensorOptionalGM;
    recvCntsGM_ = recvCountsTensorOptionalGM;
    mmxGM_ = mmxOptionalGM;
    mmwGM_ = mmweightOptionalGM;
    gmmyGM_ = gmmyGM;
    mmyGM_ = mmyOptionalGM;
    tilingData_ = tilingData;
    permuteOutGM_ =
        tilingData_->commonTilingInfo.isPermuteOut ? permuteOutOptionalGM : workspaceGM + 1024UL * 1024UL * 1024UL; // todo

    const void *hcclInitTilingV2 = &(tilingData_->hcclInitTiling);
    uint64_t hcclCcTilingOffset = offsetof(AlltoAllvGmmTilingData, alltoAllvCcTiling);
    hccl_.InitV2(contextGM, hcclInitTilingV2);
    hccl_.SetCcTilingV2(hcclCcTilingOffset);
    rankId_ = hccl_.GetRankId();
    rankDim_ = hccl_.GetRankDim();
    aivNum_ = tilingData_->commonTilingInfo.aivCoreNum;
    aivId_ = GetBlockIdx();
    totalUbSize_ = tilingData_->commonTilingInfo.totalUbSize;
    totalUbElm_ = totalUbSize_ / sizeof(DataType);
    expertNumInOneRank_ = tilingData_->commonTilingInfo.E_ep;
    expertNumAll_ = expertNumInOneRank_ * rankDim_;

    axisH1_ = tilingData_->commonTilingInfo.H1;
    axisN1_ = tilingData_->commonTilingInfo.N1;

    if constexpr (AscendC::IsSameType<DataType, bfloat16_t>::value) {
        hcclDataType_ = HCCL_DATA_TYPE_BFP16;
    }

    expertNumInGroup_ = 4; // todo
    groupNum_ = Ceil(expertNumInOneRank_, expertNumInGroup_);
    printf("[PRINT] expertNumInGroup_: %d\n", expertNumInGroup_);
    printf("[PRINT] groupNum_: %d\n", groupNum_);

    gmmxGMTensor_.SetGlobalBuffer((__gm__ DataType *)this->gmmxGM_);
    permutedGMTensor_.SetGlobalBuffer((__gm__ DataType *)this->permuteOutGM_);
    workspaceGMTensor_.SetGlobalBuffer((__gm__ DataType *)workspaceGM);

    tPipe->InitBuffer(tBuf_, totalUbSize_);
}

template <typename DataType, bool IsNeedMM, bool IsTranGmmW, bool IsTranMmW>
__aicore__ inline void AlltoAllvGmmCoarseGrained<DataType, IsNeedMM, IsTranGmmW, IsTranMmW>::Process()
{
    HcclAlltoAllvPrepare();
    if (tilingData_->commonTilingInfo.isNeedMM) {
        CalcMatmul();
        SyncAll<false>();
    }
    HcclAlltoAllvExec();
    SyncAll<false>();
    HcclFinalize();
}

template <typename DataType, bool IsNeedMM, bool IsTranGmmW, bool IsTranMmW>
__aicore__ inline void
AlltoAllvGmmCoarseGrained<DataType, IsNeedMM, IsTranGmmW, IsTranMmW>::SplitToCore(uint32_t curSendCnt, uint32_t &startTokenId,
                                                                 uint32_t &endTokenId, uint32_t &sendTokenNum)
{
    sendTokenNum = curSendCnt / aivNum_;               // 每个aiv需要发送的token数
    uint32_t remainderTokenNum = curSendCnt % aivNum_; // 余数
    startTokenId = sendTokenNum * aivId_;              // 每个aiv发送时的起始rankid
    if (aivId_ < remainderTokenNum) {                  // 前remainderRankNum个aiv需要多发1个卡的数据
        sendTokenNum += 1;
        startTokenId += aivId_;
    } else {
        startTokenId += remainderTokenNum;
    }
    endTokenId = startTokenId + sendTokenNum;
}


template <typename DataType, bool IsNeedMM, bool IsTranGmmW, bool IsTranMmW>
__aicore__ inline void AlltoAllvGmmCoarseGrained<DataType, IsNeedMM, IsTranGmmW, IsTranMmW>::CalcMatmul()
{
    mm_.Init(&(tilingData_->mmTilingData));
    GMMCompute<mmType> computeOp(mm_);
    computeOp.Init(mmxGM_, mmwGM_, mmyGM_);
    GMMProcess<decltype(computeOp)> mmOp(computeOp);
    mmOp.Init(tilingData_->mmTilingData.baseM, tilingData_->mmTilingData.baseN,
              tilingData_->commonTilingInfo.aicCoreNum);

    uint64_t mmInOffset[1] = {0};
    uint64_t mmOutOffset[1] = {0};
    uint64_t mmWeightOffset[1] = {0};
    uint32_t tokenNum[1] = {(uint32_t)(tilingData_->commonTilingInfo.BS)};
    if ASCEND_IS_AIC {
        mmOp.Process(this->rankId_, tilingData_->commonTilingInfo.H2, tilingData_->commonTilingInfo.N2, mmInOffset,
                     mmOutOffset, tokenNum, 1, 0);
    }
}

template <typename DataType, bool IsNeedMM, bool IsTranGmmW, bool IsTranMmW>
__aicore__ inline void AlltoAllvGmmCoarseGrained<DataType, IsNeedMM, IsTranGmmW, IsTranMmW>::HcclAlltoAllvPrepare()
{
    if ASCEND_IS_AIV {
        if (GetBlockIdx() != 0) {
            return;
        }
        const auto *sendCnt = &tilingData_->aicpuTiling.sendCnt[0];
        const auto *recvCnt = &tilingData_->aicpuTiling.recvCnt[0];
        uint64_t alltoAllvRecvOffsetLastSum = 0UL;
        uint64_t sendGroupCnt[MAX_EXPERT_SIZE] = {0UL};
        uint64_t recvGroupCnt[MAX_EXPERT_SIZE] = {0UL};
        uint64_t alltoAllvSendCnt[MAX_EP_RANK_SIZE] = {0UL};
        uint64_t alltoAllvSendOffset[MAX_EP_RANK_SIZE] = {0UL};
        uint64_t alltoAllvRecvCnt[MAX_EP_RANK_SIZE] = {0UL};
        uint64_t alltoAllvRecvOffset[MAX_EP_RANK_SIZE] = {0UL};

        for (uint32_t eGroup = 0; eGroup < groupNum_; eGroup++) {
            uint32_t eStart = eGroup * expertNumInGroup_;
            uint32_t eEnd = min(eStart + expertNumInGroup_, expertNumInOneRank_);
            for (uint32_t i = 0; i < rankDim_; i++) {
                for (uint32_t e = eStart; e < eEnd; e++) {
                    sendGroupCnt[i * groupNum_ + eGroup] += sendCnt[i * expertNumInOneRank_ + e];
                    recvGroupCnt[i * groupNum_ + eGroup] += recvCnt[i * expertNumInOneRank_ + e];
                }
            }
        }

        printf("[PRINT][HcclAlltoAllvPrepare] sendCnt\n");
        for (uint32_t i = 0; i < rankDim_; i++) {
            for (uint32_t e = 0; e < expertNumInOneRank_; e++) {
                printf("%d ", sendCnt[i * expertNumInOneRank_ + e]);
            }
            printf("\n");
        }
        printf("\n");
        printf("[PRINT][HcclAlltoAllvPrepare] recvCnt\n");
        for (uint32_t i = 0; i < rankDim_; i++) {
            for (uint32_t e = 0; e < expertNumInOneRank_; e++) {
                printf("%d ", recvCnt[i * expertNumInOneRank_ + e]);
            }
            printf("\n");
        }
        printf("\n");

        printf("[PRINT][HcclAlltoAllvPrepare] sendGroupCnt\n");
        for (uint32_t i = 0; i < rankDim_; i++) {
            for (uint32_t eGroup = 0; eGroup < groupNum_; eGroup++) {
                printf("%d ", sendGroupCnt[i * groupNum_ + eGroup]);
            }
            printf("\n");
        }
        printf("\n");
        printf("[PRINT][HcclAlltoAllvPrepare] recvGroupCnt\n");
        for (uint32_t i = 0; i < rankDim_; i++) {
            for (uint32_t eGroup = 0; eGroup < groupNum_; eGroup++) {
                printf("%d ", recvGroupCnt[i * groupNum_ + eGroup]);
            }
            printf("\n");
        }
        printf("\n");

        for (uint32_t e = 0U; e < groupNum_; e++) {
            printf("[PRINT][HcclAlltoAllvPrepare] e: %d\n", e);
            // 计算sendcnts/recvcnts
            for (uint32_t i = 0U; i < rankDim_; i++) {
                alltoAllvSendCnt[i] = static_cast<uint64_t>(sendGroupCnt[i * groupNum_ + e]) * axisH1_;
                alltoAllvRecvCnt[i] = static_cast<uint64_t>(recvGroupCnt[i * groupNum_ + e]) * axisH1_;
            }

            // 计算sendoffs: 本卡上专家数偏移+前面卡token数的和 -- 维护每张卡每个专家的总长度
            alltoAllvSendOffset[0] = 0UL;
            for (uint32_t j = 0U; j < e; j++) { // 0卡上的sendOffset
                alltoAllvSendOffset[0U] += static_cast<uint64_t>(sendGroupCnt[j]) * axisH1_;
            }
            for (uint32_t i = 1U; i < rankDim_; i++) {
                alltoAllvSendOffset[i] = alltoAllvSendOffset[i - 1U];
                for (uint32_t j = 0U; j < groupNum_; j++) {
                    // 后面的每张卡的sendOffset/等同于前一张卡的sendOffset+后面expertNumInOneRank_个sendCnt之和
                    alltoAllvSendOffset[i] +=
                        static_cast<uint64_t>(sendGroupCnt[e + (i - 1U) * groupNum_ + j]) * axisH1_;
                }
            }
            for (uint32_t i = 0U; i < rankDim_; i++) {
                if ((e == 0U) && (i == 0U)) {
                    alltoAllvRecvOffset[i] = 0UL;
                    alltoAllvRecvOffsetLastSum += alltoAllvRecvCnt[0];
                } else {
                    alltoAllvRecvOffset[i] = alltoAllvRecvOffsetLastSum;
                    alltoAllvRecvOffsetLastSum += alltoAllvRecvCnt[i];
                }
            }
            alltoAllvHandleId_[e] = hccl_.AlltoAllV<true>((__gm__ uint8_t *)this->gmmxGMTensor_.GetPhyAddr(),
                                                          alltoAllvSendCnt, alltoAllvSendOffset, hcclDataType_,
                                                          (__gm__ uint8_t *)this->workspaceGMTensor_.GetPhyAddr(),
                                                          alltoAllvRecvCnt, alltoAllvRecvOffset, hcclDataType_);
        }
    }
}

template <typename DataType, bool IsNeedMM, bool IsTranGmmW, bool IsTranMmW>
__aicore__ inline void AlltoAllvGmmCoarseGrained<DataType, IsNeedMM, IsTranGmmW, IsTranMmW>::DoPermuted(
    uint32_t eGroup, uint64_t *recvSumCntBefore, uint64_t *recvSumCntAfter)
{
    printf("[PRINT][DoPermuted] in DoPermuted\n");
    auto *recvCnt = &tilingData_->aicpuTiling.recvCnt[0];

    uint32_t eStart = eGroup * expertNumInGroup_;
    uint32_t eEnd = min(eStart + expertNumInGroup_, expertNumInOneRank_);
    LocalTensor<DataType> copyTensor = tBuf_.Get<DataType>();

    for(uint32_t rIndex = 0; rIndex < rankDim_; rIndex++) {
        for(uint32_t e = eStart; e < eEnd; e++) {
            uint32_t posStart = rankDim_ * eGroup * expertNumInGroup_;
            uint32_t posBefore = 0;
            uint32_t posAfter = 0;

            if(posStart + rIndex * (eEnd - eStart) + e - eStart != 0) {
                posBefore = recvSumCntBefore[posStart + rIndex * (eEnd - eStart) + e - eStart - 1];
            }
            if(posStart + (e - eStart) * rankDim_ + rIndex != 0) {
                posAfter = recvSumCntAfter[posStart + (e - eStart) * rankDim_ + rIndex - 1];
            }
            uint32_t totalCnt = recvCnt[rIndex * expertNumInOneRank_ + e] * axisH1_;
            uint32_t startCnt, endCnt, sendCnt;
            SplitToCore(totalCnt, startCnt, endCnt, sendCnt);
            if(startCnt > totalCnt) {
                continue;
            }

            uint32_t tileNum = Ceil(sendCnt, totalUbElm_);
            for(uint32_t tile = 0; tile < tileNum; tile ++) {
                uint32_t realLength = totalUbElm_;
                if(tile == tileNum - 1) {
                    realLength = min(sendCnt - tile * totalUbElm_, totalUbElm_);
                }
                DataCopyExtParams dataCopyParams{1U, static_cast<uint32_t>(realLength * sizeof(DataType)), 0U, 0U, 0U};
                DataCopyPadExtParams<DataType> dataCopyPadParams{false, 0U, 0U, 0U};
                DataCopyPad(copyTensor, workspaceGMTensor_[posBefore * axisH1_ + startCnt + tile * totalUbElm_], dataCopyParams, dataCopyPadParams);
                SyncFunc<AscendC::HardEvent::MTE2_MTE3>();
                DataCopyPad(permutedGMTensor_[posAfter * axisH1_ + startCnt + tile * totalUbElm_], copyTensor, dataCopyParams);
                SyncFunc<AscendC::HardEvent::MTE3_MTE2>();
            }
        }
    }
}

template <typename DataType, bool IsNeedMM, bool IsTranGmmW, bool IsTranMmW>
__aicore__ inline void AlltoAllvGmmCoarseGrained<DataType, IsNeedMM, IsTranGmmW, IsTranMmW>::HcclAlltoAllvExec()
{
    printf("[PRINT][HcclAlltoAllvExec] in HcclAlltoAllvExec\n");
    gmm_.Init(&(tilingData_->gmmTilingData));
    GMMCompute<gmmType> computeOp(gmm_);
    computeOp.Init(permuteOutGM_, gmmwGM_, gmmyGM_);
    GMMProcess<decltype(computeOp)> gmmOp(computeOp);
    gmmOp.Init(tilingData_->gmmTilingData.baseM, tilingData_->gmmTilingData.baseN,
               tilingData_->commonTilingInfo.aicCoreNum);

    auto *recvCnt = &tilingData_->aicpuTiling.recvCnt[0];

    uint64_t mmInOffset[10] = {0}; // todo
    uint64_t mmOutOffset[10] = {0};
    uint64_t mmWeightOffset[10] = {0};
    uint32_t tokenNum[10] = {0};

    uint64_t recvSumCntBefore[MAX_EXPERT_SIZE] = {0UL};
    uint64_t recvSumCntAfter[MAX_EXPERT_SIZE] = {0UL};

    for (uint32_t groupId = 0; groupId < groupNum_; groupId++) {
        for (uint32_t rIndex = 0; rIndex < rankDim_; rIndex++) {
            uint32_t eStart = groupId * expertNumInGroup_;
            uint32_t eEnd = min(eStart + expertNumInGroup_, expertNumInOneRank_);
            for (uint32_t e = eStart; e < eEnd; e++) {
                uint32_t pos = groupId * rankDim_ * expertNumInGroup_ + rIndex * (eEnd - eStart) + e - eStart;
                recvSumCntBefore[pos] = recvCnt[rIndex * expertNumInOneRank_ + e];
                if(pos != 0) {
                    recvSumCntBefore[pos] += recvSumCntBefore[pos - 1];
                }
            }
        }
    }
    
    for(uint32_t e = 0; e < expertNumInOneRank_; e++) {
        for (uint32_t rIndex = 0; rIndex < rankDim_; rIndex++) {
            uint32_t pos = e * rankDim_ + rIndex;
            recvSumCntAfter[pos] = recvCnt[rIndex * expertNumInOneRank_ + e];
            if(pos != 0) {
                recvSumCntAfter[pos] += recvSumCntAfter[pos - 1];
            }
        }
    }

    printf("[PRINT][HcclAlltoAllvExec] recvSumCntBefore\n");
    for (uint32_t i = 0; i < rankDim_; i++) {
        for (uint32_t e = 0; e < expertNumInOneRank_; e++) {
            printf("%d ", recvSumCntBefore[i * expertNumInOneRank_ + e]);
        }
    }
    printf("\n");
    printf("[PRINT][HcclAlltoAllvExec] recvSumCntAfter\n");
    for (uint32_t i = 0; i < rankDim_; i++) {
        for (uint32_t e = 0; e < expertNumInOneRank_; e++) {
            printf("%d ", recvSumCntAfter[i * expertNumInOneRank_ + e]);
        }
    }
    printf("\n");

    for (uint32_t e = 0U; e < groupNum_; e++) {
        printf("[PRINT][HcclAlltoAllvExec] e: %d\n", e);
        if ASCEND_IS_AIV {
            if (GetBlockIdx() == 0) {
                hccl_.Wait(alltoAllvHandleId_[e]);
                printf("[PRINT][HcclAlltoAllvExec] after wait\n");
            }
        }

        SyncAll<false>();
        if ASCEND_IS_AIV {
            DoPermuted(e, recvSumCntBefore, recvSumCntAfter);
        }
        SyncAll<false>();
        printf("[PRINT][HcclAlltoAllvExec] before gmm\n");

        int32_t calNum = min(expertNumInGroup_, expertNumInOneRank_ - e * expertNumInGroup_);
        for (uint32_t i = 0; i < calNum; i++) {
            if(i == 0 && e == 0) {
                mmInOffset[i] = 0;
                mmOutOffset[i] = 0;
                tokenNum[i] = recvSumCntAfter[rankDim_ - 1];
            } else {
                mmInOffset[i] = recvSumCntAfter[expertNumInGroup_ * rankDim_ * e + i * rankDim_ - 1] * axisH1_;
                mmOutOffset[i] = recvSumCntAfter[expertNumInGroup_ * rankDim_ * e + i * rankDim_ - 1] * axisN1_;
                tokenNum[i] = recvSumCntAfter[expertNumInGroup_ * rankDim_ * e + (i + 1) * rankDim_ - 1] -
                              recvSumCntAfter[expertNumInGroup_ * rankDim_ * e + i * rankDim_ - 1];
            }
            printf("[PRINT][HcclAlltoAllvExec] mmInOffset[%d]: %d\n", i, mmInOffset[i]);
            printf("[PRINT][HcclAlltoAllvExec] mmOutOffset[%d]: %d\n", i, mmOutOffset[i]);
            printf("[PRINT][HcclAlltoAllvExec] tokenNum[%d]: %d\n", i, tokenNum[i]);
        }
        
        if ASCEND_IS_AIC {
            if (tokenNum[0] != 0) {
                gmmOp.Process(this->rankId_, axisH1_, axisN1_, mmInOffset, mmOutOffset, tokenNum, calNum, e * expertNumInGroup_);
            }
        }
    }
}

template <typename DataType, bool IsNeedMM, bool IsTranGmmW, bool IsTranMmW>
__aicore__ inline void AlltoAllvGmmCoarseGrained<DataType, IsNeedMM, IsTranGmmW, IsTranMmW>::HcclFinalize()
{
    if ASCEND_IS_AIV {
        hccl_.Finalize();
    }
}

} // namespace AscendC
#endif
