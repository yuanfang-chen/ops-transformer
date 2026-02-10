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
 * \file quant_grouped_mat_mul_allto_allv_tiling_adapter.cpp
 * \brief
 */

#include "op_mc2.h"
#include "mc2_log.h"
#include "quant_grouped_mat_mul_allto_allv_tiling_adapter.h"

using namespace Mc2Log;
using namespace AscendC;
using namespace optiling;
using namespace Mc2GroupedMatmulTiling;
using namespace Mc2GroupedMatmulTiling::GmmConstant;
using namespace Mc2GroupedMatmul;

ge::graphStatus QuantGroupedMatmulAllToAllvAdapter::SetGroupExpertInputParameters(const int32_t* sendCounts,
    uint64_t worldSize, uint64_t index, uint32_t epNums)
{
    uint64_t mSizePerLoop = 0;
    // index sendcounts起始   epNums 当前loop专家数 -- 每轮专家 与 尾轮专家
    if (epNums == 1) {
        for (uint32_t i = 0; i < worldSize; i++) {
            mSizePerLoop += sendCounts[index + i];
        }

        inputParams_.mSize = mSizePerLoop;
        inputParams_.isSingleX = true;
        inputParams_.isSingleW = true;
        inputParams_.isSingleY = true;
        mList_[0] = static_cast<int32_t>(mSizePerLoop);
        kList_[0] = static_cast<int32_t>(inputParams_.kSize);
        nList_[0] = static_cast<int32_t>(inputParams_.nSize);
        return ge::GRAPH_SUCCESS; 
    }

    // 每轮多专家
    for (uint32_t i = 0; i < epNums; i++) {
        mSizePerLoop = 0;
        for (uint32_t j = 0; j < worldSize; j++) {
            mSizePerLoop += sendCounts[index + i * worldSize + j];
        }
        mList_[i] = static_cast<int32_t>(mSizePerLoop);
        kList_[i] = static_cast<int32_t>(inputParams_.kSize);
        nList_[i] = static_cast<int32_t>(inputParams_.nSize);
    }

    inputParams_.isSingleX = false;
    inputParams_.isSingleW = false;
    inputParams_.isSingleY = false;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantGroupedMatmulAllToAllvAdapter::SetSharedExpertInputParameters(const QuantGmmAlltoAllvParamsInfo& params)
{
    inputParams_.mSize = params.Bs;
    inputParams_.kSize = params.H2;
    inputParams_.nSize = params.N2;
    // quantMode bit position: 1 << mode
    inputParams_.aQuantMode = static_cast<QuantMode>(1U << params.mmXQuantMode);
    inputParams_.bQuantMode = static_cast<QuantMode>(1U << params.mmWeightQuantMode);
    // 是否做切分
    inputParams_.groupType = optiling::Mc2GroupedMatmul::NO_SPLIT;
    // 非负递增为0，非负数列为1
    inputParams_.groupListType = 1;
    // 输出是否切分，0/1代表输出多tensor， 2/3代表输出单tensor
    inputParams_.splitItem = 2;
    inputParams_.actType = 0;

    inputParams_.aDtype = params.mmXDtype;
    inputParams_.bDtype = params.mmWeightDtype;
    inputParams_.cDtype = params.mmYDtype;
    inputParams_.biasDtype = ge::DT_INT32;
    inputParams_.scaleDtype = params.mmXScaleDtype;
    inputParams_.perTokenScaleDtype = params.mmXScaleDtype;
    inputParams_.aFormat = ge::FORMAT_ND;
    inputParams_.bFormat = ge::FORMAT_ND;
    inputParams_.cFormat = ge::FORMAT_ND;
    inputParams_.transA = false;
    inputParams_.transB = params.isMmWeightTrans;
    inputParams_.hasBias = false;
    inputParams_.isSingleX = true;
    inputParams_.isSingleW = true;
    inputParams_.isSingleY = true;
    SetMKNList();
    SetKernelType();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantGroupedMatmulAllToAllvAdapter::SetCommonInputParams(const QuantGmmAlltoAllvParamsInfo& params)
{
    inputParams_.opName = params.opName;
    inputParams_.mSize = params.A;
    inputParams_.kSize = params.H1;
    inputParams_.nSize = params.N1;
    inputParams_.groupNum = params.ep;
    // gmm未使用
    // inputParams_.outDtype = params.gmmYDtype;
    // inputParams_.outDataDtype = ge::DT_FLOAT16;
    // inputParams_.outScaleDtype = ge::DT_FLOAT;
    // need set
    inputParams_.kernelType = 0UL;
    // quantMode bit position: 1 << mode
    inputParams_.aQuantMode = static_cast<QuantMode>(1U << params.gmmXQuantMode);
    inputParams_.bQuantMode = static_cast<QuantMode>(1U << params.gmmWeightQuantMode);
    // 是否做切分
    inputParams_.groupType = optiling::Mc2GroupedMatmul::SPLIT_M;
    inputParams_.groupListType = 1;
    // 输出是否切分，0/1代表输出多tensor， 2/3代表输出单tensor
    inputParams_.splitItem = 0;
    inputParams_.actType = 0;
    inputParams_.aDtype = params.gmmXDtype;
    inputParams_.bDtype = params.gmmWeightDtype;
    inputParams_.cDtype = params.gmmYDtype;
    inputParams_.biasDtype = ge::DT_INT32;
    inputParams_.scaleDtype = params.gmmXScaleDtype;
    inputParams_.perTokenScaleDtype = params.gmmXScaleDtype;
    inputParams_.aFormat = ge::FORMAT_ND;
    inputParams_.bFormat = ge::FORMAT_ND;
    inputParams_.cFormat = ge::FORMAT_ND;
    inputParams_.transA = false;
    inputParams_.transB = params.isGmmWeightTrans;
    inputParams_.hasBias = false;
    inputParams_.isSingleX = true;
    inputParams_.isSingleW = true;
    inputParams_.isSingleY = true;
    SetMKNList();
    SetKernelType();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantGroupedMatmulAllToAllvAdapter::Process()
{
    GE_ASSERT_GRAPH_SUCCESS(DoOpTiling());
    GE_ASSERT_GRAPH_SUCCESS(DoLibApiTiling());
    return ge::GRAPH_SUCCESS;
}
