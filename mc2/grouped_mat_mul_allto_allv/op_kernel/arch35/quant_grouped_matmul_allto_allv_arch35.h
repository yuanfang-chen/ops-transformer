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
 * \file quant_grouped_matmul_allto_allv_arch35.h
 * \brief
 */

#ifndef QUANT_GROUPED_MATMUL_ALLTO_ALLV_ARCH35_H
#define QUANT_GROUPED_MATMUL_ALLTO_ALLV_ARCH35_H


#include "../grouped_mat_mul_allto_allv_tiling.h"

namespace GroupedMatmulAlltoAllv {
using namespace AscendC;
template <typename SchedulerType, typename SchedulerContextType, typename QuantGroupedMatMulAlltoAllvTilingData>
class QuantGmmA2avKernel
{
public:
    __aicore__ inline QuantGmmA2avKernel(SchedulerType* pipeLine) : pipeLine_(pipeLine){};
    __aicore__ inline void Init(GM_ADDR gmmxGM, GM_ADDR gmmweightGM, GM_ADDR sendCountsTensorGM, GM_ADDR recvCountsTensorGM, GM_ADDR mmGM,
        GM_ADDR mmweightGM, GM_ADDR yGM, GM_ADDR mmyGM, GM_ADDR workspaceGM, GM_ADDR contextGM,
        const QuantGroupedMatMulAlltoAllvTilingData* tilingData, __gm__ void* hcclInitTiling, __gm__ void* alltoAllvCcTiling,
        TPipe* tPipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ProcessExpert(uint32_t taskCnt);
    // 执行流水线
    __aicore__ inline void ProcessPipeLine(uint32_t taskCnt);

    SchedulerType* pipeLine_;
    SchedulerContextType pipeLineContext_;
    MatmulAlltoAllTilingDataType* tilingData_;
    TCubeTiling matmulTiling_;
    TCubeTiling sharedMatmulTiling_;
    TPipe* tPipe_;

    GM_ADDR gmmxGM_ = nullptr;
    GM_ADDR gmmweightGM_ = nullptr;
    GM_ADDR sendCntsGM_ = nullptr;
    GM_ADDR recvCntsGM_ = nullptr;
    GM_ADDR mmxGM_ = nullptr;
    GM_ADDR mmweightGM_ = nullptr;
    GM_ADDR gmmbiasGM_ = nullptr;
    GM_ADDR mmbiasGM_ = nullptr;
    GM_ADDR gmmyGM_ = nullptr;
    GM_ADDR gmmOutGM_ = nullptr;
    GM_ADDR mmyGM_ = nullptr;
    GM_ADDR workspaceGM_ = nullptr;
    GM_ADDR gmmxScaleGM_ = nullptr;
    GM_ADDR gmmWeightScaleGM_ = nullptr;
    GM_ADDR mmxScaleGM_ = nullptr;
    GM_ADDR mmWeightScaleGM_ = nullptr;
    GM_ADDR tilingGM_ = nullptr;

    uint64_t rankId_{0};
    uint64_t rankDim_{8};

    uint64_t expertNumInOneRank_ = 0U; // 单卡专家数
    uint64_t expertNumAll_ = 0U;       // 通信域内专家数

    uint64_t axisBsK_ = 0U;
    uint64_t axisH_ = 0U;
    uint64_t axisA_ = 0U;
    uint64_t axisN1_ = 0U;
    uint64_t workSpaceOffset = 0U;

    // alltoallv 流程数据结构
    static constexpr uint64_t MAX_EP_RANK_SIZE = 128U;
    static constexpr uint64_t TOTAL_UBSIZE = static_cast<uint64_t>(190U * 1024U / 2U);
    static constexpr uint64_t MAX_AIV_NUM = 48U;
    static constexpr uint64_t MAX_HANDLE_ID_NUM = 64U;
};

template <typename SchedulerType, typename SchedulerContextType, typename QuantGroupedMatMulAlltoAllvTilingData>
__aicore__ inline void QuantGmmA2avKernel<SchedulerType, SchedulerContextType, QuantGroupedMatMulAlltoAllvTilingData>::Init(
        GM_ADDR gmmxGM, GM_ADDR gmmweightGM, GM_ADDR sendCountsTensorOptionalGM,
        GM_ADDR recvCountsTensorOptionalGM, GM_ADDR mmxOptionalGM, GM_ADDR mmweightOptionalGM, GM_ADDR biasGM, 
        GM_ADDR gmmxScaleGM, GM_ADDR gmmWeightScaleGM, GM_ADDR mmxScaleGM, GM_ADDR mmWeightScaleGM, GM_ADDR gmmyGM,
        GM_ADDR mmyOptionalGM, GM_ADDR workspaceGM, GM_ADDR contextGM,
        const QuantGroupedMatMulAlltoAllvTilingData* tilingData, GM_ADDR tilingGM, __gm__ void* hcclInitTiling, __gm__ void* alltoAllvCcTiling,
        TPipe* tPipe)
{
    gmmxGM_ = gmmxGM;
    gmmweightGM_ = gmmweightGM;
    mmxGM_ = mmxOptionalGM;
    mmweightGM_ = mmweightOptionalGM;
    gmmxScaleGM_ = gmmxScaleGM;
    gmmWeightScaleGM_ = gmmWeightScaleGM;
    mmxScaleGM_ = mmxScaleGM;
    mmWeightScaleGM_ = mmWeightScaleGM;
    gmmyGM_ = gmmyGM;
    mmyGM_ = mmyOptionalGM;
    gmmOutGM_ = workspaceGM;
    workspaceGM_ = workspaceGM;
    tilingData_ = tilingData;
    tilingGM_ = tilingGM;
    tPipe_ = tPipe;

    axisBsK_ = tilingData_->commonTilingInfo.BsK;
    axisH_ = tilingData_->commonTilingInfo.H;
    axisA_ = tilingData_->commonTilingInfo.A;
    axisN1_ = tilingData_->commonTilingInfo.N1;

    expertNumInOneRank_ = tilingData_->commonTilingInfo.E_ep;

}

template <typename SchedulerType, typename SchedulerContextType, typename QuantGroupedMatMulAlltoAllvTilingData>
__aicore__ inline void QuantGmmA2avKernel<SchedulerType, SchedulerContextType, QuantGroupedMatMulAlltoAllvTilingData>::Process()
{
    // 启动流水
    ProcessPipeLine(expertNumInOneRank_);

    // 结束流水线
    pipeLine_->End();
}

template <typename SchedulerType, typename SchedulerContextType, typename QuantGroupedMatMulAlltoAllvTilingData>
__aicore__ inline void QuantGmmA2avKernel<SchedulerType, SchedulerContextType, QuantGroupedMatMulAlltoAllvTilingData>::ProcessPipeLine(uint32_t taskCnt)
{
    pipeLine_->Init(gmmxGM_, gmmweightGM_, mmxGM_, mmweightGM_, gmmxScaleGM_, gmmWeightScaleGM_, mmxScaleGM_, mmWeightScaleGM_, gmmyGM_, 
        mmyGM_, workspaceGM_, tilingData_, tPipe_)

    pipeLine_->Process(taskCnt);
}

} // GroupedMatmulAlltoAllv
#endif // QUANT_GROUPED_MATMUL_ALLTO_ALLV_ARCH35_H