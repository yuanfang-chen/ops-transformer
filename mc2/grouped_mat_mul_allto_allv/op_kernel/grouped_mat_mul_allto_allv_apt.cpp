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
 * \file grouped_mat_mul_allto_allv_apt.cpp
 * \brief
 */
#include <cstring>
#include <lib/matmul_intf.h>
#include "basic_api/kernel_basic_intf.h"
#include "common.h"
#include "./grouped_mat_mul_allto_allv_tiling.h"
#include "./grouped_mat_mul_allto_allv_tiling_key.h"
#include "./arch35/template_head.h"
#include "./arch35/quant_grouped_matmul_allto_allv_arch35.h"

using namespace AscendC;
using namespace ATAVKernelTemplate;
using namespace GroupedMatmulAlltoAllv;

template <uint32_t QUANTMODE, bool X2TRANSPOSE, uint32_t DTYPEBIAS>
__global__ __aicore__ void grouped_mat_mul_allto_allv(
    GM_ADDR gmmxGM, GM_ADDR gmmweightGM,
    GM_ADDR sendCountsTensorOptionalGM, GM_ADDR recvCountsTensorOptionalGM, GM_ADDR mmxOptionalGM,
    GM_ADDR mmweightOptionalGM, GM_ADDR biasGM, GM_ADDR gmmxScaleGM, GM_ADDR gmmWeightScaleGM, GM_ADDR mmxScaleGM, 
    GM_ADDR mmWeightScaleGM, GM_ADDR gmmyGM, GM_ADDR mmyOptionalGM, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    //kernel的使用类型，这里是cube和vic混用，cube是主核，cube:vec=1:2
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    TPipe pipe;

    //注册默认的tilingdata，需要保证有且只有一个默认tilingdata被注册
    REGISTER_TILING_DEFAULT(QuantGroupedMatMulAlltoAllvTilingData);
    GET_TILING_DATA_WITH_STRUCT(QuantGroupedMatMulAlltoAllvTilingData, tilingData, tilingGM);

    DEFINE_AND_IMPL_MC2_MATMUL_FOR_MATMUL_COMPUTATION_QUANT(QuantGroupedMatMulAlltoAllvTilingData::gmmQuantTilingData::GMMQuantTilingData, ComputationType);
    ComputationType matmulImplName(&pipe);
    DEFINE_MC2_HCCL_FOR_COMMUNICATION(HcclServerType::HCCL_SERVER_TYPE_CCU, 1, 0, QuantGroupedMatMulAlltoAllvTilingData, CommunicationType);
    CommunicationType commImplName(&tilingData);
    using SchedulerContextType = PipelineContext<QuantGroupedMatMulAlltoAllvTilingData::gmmQuantTilingData::GMMQuantTilingData>;
    using SchedulerType = QGMMKernelPipelineTemplate<ComputationType, CommunicationType, SchedulerContextType>;
    SchedulerType SchedulerImpl(&matmulImplName, &commImplName);

    QuantGmmA2avKernel<SchedulerType, SchedulerContextType, QuantGroupedMatMulAlltoAllvTilingData> op(&SchedulerImpl);
    op.Init(gmmxGM, gmmweightGM, sendCountsTensorOptionalGM, recvCountsTensorOptionalGM, mmxOptionalGM,
            mmweightOptionalGM, biasGM, gmmxScaleGM, gmmWeightScaleGM, mmxScaleGM, mmWeightScaleGM, gmmyGM,
            mmyOptionalGM, workspaceGM, contextGM, &tilingData, tilingGM, hcclInitTiling, alltoAllvCcTiling,
            &pipe);
            
    op.Process();
}