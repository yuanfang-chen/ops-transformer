/* *
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

#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#include "allto_allv_gmm.h"
#include "allto_allv_grouped_mat_mul_tiling.h"
#include "../grouped_matmul_apt/op_kernel/arch35/quant_adaptive_sliding_window_templates/gqmm_cube_on_the_fly.h"
#include "../grouped_matmul_apt/op_kernel/arch35/non_quant/grouped_matmul_basic_kernel.h"

#if defined(CONST_TILING)
  #define GET_TILING_ARRAY_MEMBER_ADDR(tilingType, member, subMember, var, tiling)     \
      GET_TILING_DATA_MEMBER(tilingType, member, arrayObj, tiling);                             \
      const int32_t* (var) = (const int32_t*)((const uint8_t*)&arrayObj.subMember);
#else
  #define GET_TILING_ARRAY_MEMBER_ADDR(tilingType, member, subMember, var, tiling)       \
    size_t arrayOffset##var = (size_t)(&((tilingType*)0)->member);                              \
    size_t elementOffset##var = (size_t)(&(((tilingType*)0)->member.subMember));             \
    __gm__ int32_t* (var) = (__gm__ int32_t*)((tiling) + arrayOffset##var +                     \
                                               elementOffset##var);
#endif

namespace AscendC {
using namespace ALLTO_ALLV_GMM;
template <typename ATAVGMMQuant>
class QuantAlltoAllvGmm {
public:
    __aicore__ inline QuantAlltoAllvGmm() {}
    __aicore__ inline void Init(GM_ADDR gmmxGM, GM_ADDR gmmweightGM, GM_ADDR sendCountsTensorOptionalGM,
        GM_ADDR recvCountsTensorOptionalGM, GM_ADDR mmxOptionalGM, GM_ADDR mmweightOptionalGM, GM_ADDR biasGM, 
        GM_ADDR gmmxScaleGM, GM_ADDR gmmWeightScaleGM, GM_ADDR mmxScaleGM, GM_ADDR mmWeightScaleGM, GM_ADDR gmmyGM,
        GM_ADDR mmyOptionalGM, GM_ADDR permuteOutOptionalGM, GM_ADDR workspaceGM, GM_ADDR contextGM,
        const QuantAlltoAllvGroupedMatmulTilingData *tilingData, GM_ADDR tilingGM, __gm__ void *hcclInitTiling, __gm__ void *alltoAllvCcTiling,
        TPipe *tPipe); // TODO ADD scale
    __aicore__ inline void Process();

    using X_T = typename ATAVGMMQuant::xType;
    using W_T = typename ATAVGMMQuant::wType;
    using BIAS_T = typename ATAVGMMQuant::biasType;
    using SCALE_T = typename ATAVGMMQuant::scaleType;
    using Y_T = typename ATAVGMMQuant::yType;
    static constexpr CubeFormat W_FORMAT = ATAVGMMQuant::wFormat;
    static constexpr bool A_TRANS = ATAVGMMQuant::aTrans;
    static constexpr bool B_TRANS = ATAVGMMQuant::bTrans;
    static constexpr bool NEED_MM = ATAVGMMQuant::isOptionalMm;
    static constexpr bool NEED_GMMW_TRANS = ATAVGMMQuant::isGmmWeightTrans;
    static constexpr bool NEED_MMW_TRANS = ATAVGMMQuant::isOptWeightTrans;
    TILING_TYPE* gmmArrayAddrIn;
private:
    using aType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, X_T, false>;
    using gmmBType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, X_T, NEED_GMMW_TRANS>;
    using mmBType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, X_T, NEED_MMW_TRANS>;
    using cType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, X_T, false>;
    using biasType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, X_T>;
    using gmmType = MMImplType<aType, gmmBType, cType, biasType, CFG_MDL>;
    using mmType = MMImplType<aType, mmBType, cType, biasType, CFG_MDL>;

    __aicore__ inline void CalcMatmul();
    __aicore__ inline void HcclAlltoAllvPrepare();
    __aicore__ inline void HcclAlltoAllvExec();
    __aicore__ inline void HcclFinalize();

    GM_ADDR gmmxGM_ = nullptr;
    GM_ADDR gmmwGM_ = nullptr;
    GM_ADDR sendCntsGM_ = nullptr;
    GM_ADDR recvCntsGM_ = nullptr;
    GM_ADDR mmxGM_ = nullptr;
    GM_ADDR mmwGM_ = nullptr;
    GM_ADDR biasGM_ = nullptr;
    GM_ADDR gmmyGM_ = nullptr;
    GM_ADDR mmyGM_ = nullptr;
    GM_ADDR permuteOutGM_ = nullptr;
    GM_ADDR workspaceGM_ = nullptr;
    GM_ADDR gmmxScaleGM_ = nullptr;
    GM_ADDR gmmWeightScaleGM_ = nullptr;
    GM_ADDR mmxScaleGM_ = nullptr;
    GM_ADDR mmWeightScaleGM_ = nullptr;
    GM_ADDR tilingGM_ = nullptr;

    TPipe *tPipe_;

    const QuantAlltoAllvGroupedMatmulTilingData *tilingData_ = nullptr;
    uint32_t rankId_ = 0U;             // 当前卡ID
    uint32_t rankDim_ = 8U;            // 通信域内卡的数量
    uint32_t expertNumInOneRank_ = 0U; // 单卡上面的专家个数
    uint32_t expertNumAll_ = 0U;       // 通信域内专家个数

    // alltoall 流程数据结构
    static constexpr uint64_t MAX_HANDLE_ID_NUM = 64U;
#if defined(__DAV_C310__)
    Hccl<HcclServerType::HCCL_SERVER_TYPE_CCU> hccl_;
#else
    Hccl<HCCL_SERVER_TYPE_AICPU> hccl_;
#endif
    HcclDataType hcclDataType_ = HCCL_DATA_TYPE_FP16;
    HcclHandle alltoAllvHandleId_[MAX_HANDLE_ID_NUM] = {INVALID_HANDLE_ID};

    GlobalTensor<X_T> gmmxGMTensor_;
    GlobalTensor<X_T> permutedGMTensor_;
    GlobalTensor<W_T> gmmWeightGMTensor_;
    GlobalTensor<SCALE_T> gmmxScaleGMTensor_;
    GlobalTensor<SCALE_T> gmmWeightScaleGMTensor_;
    GlobalTensor<int64_t> groupListGMTensor_;

    uint64_t axisH1_;
    uint64_t axisN1_;

    GmmASWKernel<X_T, W_T, BIAS_T, SCALE_T, Y_T, W_FORMAT, A_TRANS, B_TRANS> gmmASWKernel;

};

template <typename ATAVGMMQuant>
__aicore__ inline void QuantAlltoAllvGmm<ATAVGMMQuant>::Init(GM_ADDR gmmxGM, GM_ADDR gmmweightGM, GM_ADDR sendCountsTensorOptionalGM,
        GM_ADDR recvCountsTensorOptionalGM, GM_ADDR mmxOptionalGM, GM_ADDR mmweightOptionalGM, GM_ADDR biasGM, GM_ADDR gmmxScaleGM,
        GM_ADDR gmmWeightScaleGM, GM_ADDR mmxScaleGM, GM_ADDR mmWeightScaleGM, GM_ADDR gmmyGM,
        GM_ADDR mmyOptionalGM, GM_ADDR permuteOutOptionalGM, GM_ADDR workspaceGM, GM_ADDR contextGM,
        const QuantAlltoAllvGroupedMatmulTilingData *tilingData, GM_ADDR tilingGM, __gm__ void *hcclInitTiling, __gm__ void *alltoAllvCcTiling,
        TPipe *tPipe) // TODO ADD scale
{
    GM_ADDR userWorkspace = AscendC::GetUserWorkspace(workspaceGM);
    gmmxGM_ = gmmxGM;
    gmmwGM_ = gmmweightGM;
    sendCntsGM_ = sendCountsTensorOptionalGM;
    recvCntsGM_ = recvCountsTensorOptionalGM;
    mmxGM_ = mmxOptionalGM;
    mmwGM_ = mmweightOptionalGM;
    gmmxScaleGM_ = gmmxScaleGM;
    gmmWeightScaleGM_ = gmmWeightScaleGM;
    mmxScaleGM_ = mmxScaleGM;
    mmWeightScaleGM_ = mmWeightScaleGM;
    gmmyGM_ = gmmyGM;
    mmyGM_ = mmyOptionalGM;
    workspaceGM_ = workspaceGM;
    tilingData_ = tilingData;
    tilingGM_ = tilingGM;
    tPipe_ = tPipe;
    permuteOutGM_ = tilingData_->commonTilingInfo.isPermuteOut ? permuteOutOptionalGM : workspaceGM;
    // TODO set scale gm addr

    hccl_.Init(contextGM, hcclInitTiling);
    hccl_.SetCcTiling(alltoAllvCcTiling);
    rankId_ = hccl_.GetRankId();
    rankDim_ = hccl_.GetRankDim();

    expertNumInOneRank_ = tilingData_->commonTilingInfo.E_ep;
    expertNumAll_ = expertNumInOneRank_ * rankDim_;

    axisH1_ = tilingData_->commonTilingInfo.H1;
    axisN1_ = tilingData_->commonTilingInfo.N1;

    if constexpr (AscendC::IsSameType<X_T, bfloat16_t>::value) {
        hcclDataType_ = HCCL_DATA_TYPE_BFP16;
    }

    gmmxGMTensor_.SetGlobalBuffer((__gm__ X_T *)this->gmmxGM_);
    permutedGMTensor_.SetGlobalBuffer((__gm__ X_T *)this->permuteOutGM_);
    gmmWeightGMTensor_.SetGlobalBuffer((__gm__ W_T *)this->gmmwGM_);
    gmmxScaleGMTensor_.SetGlobalBuffer((__gm__ SCALE_T *)this->gmmxScaleGM_);
    gmmWeightScaleGMTensor_.SetGlobalBuffer((__gm__ SCALE_T *)this->gmmWeightScaleGM_);
    groupListGMTensor_.SetGlobalBuffer((__gm__ int64_t *)userWorkspace);
}

template <typename ATAVGMMQuant> __aicore__ inline void QuantAlltoAllvGmm<ATAVGMMQuant>::Process()
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

template <typename ATAVGMMQuant> __aicore__ inline void QuantAlltoAllvGmm<ATAVGMMQuant>::CalcMatmul()
{
    uint64_t mmInOffset[1] = {0};
    uint64_t mmOutOffset[1] = {0};
    uint64_t mmWeightOffset[1] = {0};
    uint32_t tokenNum[1] = {(uint32_t)(tilingData_->commonTilingInfo.BS)};
    if ASCEND_IS_AIC {
        // 共享专家计算
        const GroupedMatmulTilingData::GMMQuantTilingData &gmmQuantTilingData = tilingData_->gmmQuantTilingDataList[0];

        // 使用类型名而不是变量名
        GET_TILING_DATA_MEMBER(GroupedMatmulTilingData::GMMQuantTilingData, gmmQuantParams, gmmQuantParams_, tilingGM_);
        GET_TILING_DATA_MEMBER(GroupedMatmulTilingData::GMMQuantTilingData, mmTilingData, mmTilingData_, tilingGM_);
        GET_TILING_DATA_MEMBER_ADDR(GroupedMatmulTilingData::GMMQuantTilingData, gmmArray, gmmArrayAddr_, tilingGM_);

        gmmASWKernel.Init(permuteOutGM_, // TODO compute offset
            gmmwGM_, biasGM_, gmmxScaleGM_, 0, gmmWeightScaleGM_, gmmyGM_, workspaceGM_,
            &gmmQuantParams_, &mmTilingData_, gmmArrayAddr_,  // 添加&取地址符
            tPipe_);
        gmmASWKernel.Process();
    }
}

template <typename ATAVGMMQuant> __aicore__ inline void QuantAlltoAllvGmm<ATAVGMMQuant>::HcclAlltoAllvPrepare()
{
    if ASCEND_IS_AIV {
        if (GetBlockIdx() != 0) {
            return;
        }
        const auto *sendCnt = &tilingData_->aicpuTiling.sendCnt[0];
        const auto *recvCnt = &tilingData_->aicpuTiling.recvCnt[0];
        uint64_t alltoAllvRecvOffsetLastSum = 0UL;
        uint64_t alltoAllvSendCnt[MAX_EP_RANK_SIZE] = {0UL};
        uint64_t alltoAllvSendOffset[MAX_EP_RANK_SIZE] = {0UL};
        uint64_t alltoAllvRecvCnt[MAX_EP_RANK_SIZE] = {0UL};
        uint64_t alltoAllvRecvOffset[MAX_EP_RANK_SIZE] = {0UL};
        for (uint32_t e = 0U; e < expertNumInOneRank_; e++) {
            // 计算sendcnts/recvcnts
            for (uint32_t i = 0U; i < rankDim_; i++) {
                alltoAllvSendCnt[i] = static_cast<uint64_t>(sendCnt[i * expertNumInOneRank_ + e]) * axisH1_;
                alltoAllvRecvCnt[i] = static_cast<uint64_t>(recvCnt[i * expertNumInOneRank_ + e]) * axisH1_;
            }

            // 计算sendoffs: 本卡上专家数偏移+前面卡token数的和 -- 维护每张卡每个专家的总长度
            alltoAllvSendOffset[0] = 0UL;
            for (uint32_t j = 0U; j < e; j++) { // 0卡上的sendOffset
                alltoAllvSendOffset[0U] += static_cast<uint64_t>(sendCnt[j]) * axisH1_;
            }
            for (uint32_t i = 1U; i < rankDim_; i++) {
                alltoAllvSendOffset[i] = alltoAllvSendOffset[i - 1U];
                for (uint32_t j = 0U; j < expertNumInOneRank_; j++) {
                    // 后面的每张卡的sendOffset/等同于前一张卡的sendOffset+后面expertNumInOneRank_个sendCnt之和
                    alltoAllvSendOffset[i] +=
                        static_cast<uint64_t>(sendCnt[e + (i - 1U) * expertNumInOneRank_ + j]) * axisH1_;
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
            alltoAllvHandleId_[e] =
                hccl_.AlltoAllV<true>((__gm__ uint8_t *)this->gmmxGMTensor_.GetPhyAddr(), alltoAllvSendCnt,
                alltoAllvSendOffset, hcclDataType_, (__gm__ uint8_t *)this->permutedGMTensor_.GetPhyAddr(),
                alltoAllvRecvCnt, alltoAllvRecvOffset, hcclDataType_);
        }
    }
}

template <typename ATAVGMMQuant> __aicore__ inline void QuantAlltoAllvGmm<ATAVGMMQuant>::HcclAlltoAllvExec()
{
    AscendC::printf("tilingData_->gmmQuantTilingData.mmTilingData.M = %ld\n", tilingData_->gmmQuantTilingData.mmTilingData.M);

    // for (uint32_t e = 0U; e < expertNumInOneRank_; e++) {
    //     groupListGMTensor_.SetValue(e, tilingData_->gmmQuantTilingData[e + 1].mmTilingData.M);
    // }
    // AscendC::DataCacheCleanAndInvalid<int64_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(groupListGMTensor_);
    // SyncAll<false>();

    auto *recvCnt = &tilingData_->aicpuTiling.recvCnt[0];

    uint64_t mmInOffset[2] = {0};
    uint64_t mmOutOffset[2] = {0};
    uint64_t mmWeightOffset[2] = {0};
    uint32_t tokenNum[2] = {0};
    uint32_t processNum = 1;

    uint64_t expertoffset[MAX_HANDLE_ID_NUM] = {0UL};
    uint64_t gmmTokennum[MAX_HANDLE_ID_NUM] = {0UL};
    for (uint32_t e = 0U; e < expertNumInOneRank_; e++) {
        uint64_t curTokenNum = 0;
        for (uint32_t i = 0U; i < rankDim_; i++) {
            curTokenNum += static_cast<uint64_t>(recvCnt[e + i * expertNumInOneRank_]);
        }
        gmmTokennum[e] = curTokenNum; // 本专家的tokenNum
    }
    for (uint64_t i = 0; i < expertNumInOneRank_; i++) {
        if (i >= 1) {
            expertoffset[i] = expertoffset[i - 1] + gmmTokennum[i - 1] * axisH1_;
        }
    }
    for (uint32_t e = 0U; e < this->expertNumInOneRank_; e++) {
        if ASCEND_IS_AIV {
            if (GetBlockIdx() == 0) {
                hccl_.Wait(alltoAllvHandleId_[e]);
            }
        }
        SyncAll<false>();
        mmInOffset[0] = expertoffset[e];
        mmOutOffset[0] = expertoffset[e] / axisH1_ * axisN1_;
        mmWeightOffset[0] = e * axisH1_ * axisN1_;
        tokenNum[0] = gmmTokennum[e];

        if ASCEND_IS_AIC {
            GET_TILING_ARRAY_MEMBER_ADDR(QuantAlltoAllvGroupedMatmulTilingData, gmmQuantTilingData, gmmArray, gmmArrayAddr_, tilingGM_);
            gmmASWKernel.Init(permuteOutGM_,
                gmmwGM_, biasGM_, gmmxScaleGM_, 0, gmmWeightScaleGM_, gmmyGM_, workspaceGM_,
                &tilingData_->gmmQuantTilingData.gmmQuantParams, &tilingData_->gmmQuantTilingData.mmTilingData, gmmArrayAddr_,
                tPipe_);
            gmmASWKernel.Process();
        }
    }
}

template <typename ATAVGMMQuant> __aicore__ inline void QuantAlltoAllvGmm<ATAVGMMQuant>::HcclFinalize()
{
    if ASCEND_IS_AIV {
        hccl_.Finalize();
    }
}
} // namespace AscendC
#endif