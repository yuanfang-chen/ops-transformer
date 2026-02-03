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
 * \file quant_grouped_mat_mul_allto_allv_tiling_split_strategy.cpp
 * \brief
 */

#include "op_mc2.h"
#include "mc2_log.h"
#include "quant_grouped_mat_mul_allto_allv_tiling_split_strategy.h"

using namespace Mc2Log;
using namespace AscendC;
using namespace optiling;
using namespace Mc2GroupedMatmul;

// struct GQmmInputInfo {
//     uint64_t mSize = 0UL;
//     uint64_t kSize = 0UL;
//     uint64_t nSize = 0UL;
//     uint64_t groupNum = 0UL;
//     int64_t outDtype = 0L;
//     uint64_t kernelType = 0UL;
//     QuantMode aQuantMode = QuantMode::DEFAULT;
//     QuantMode bQuantMode = QuantMode::DEFAULT;
//     int8_t groupType = Mc2GroupedMatmul::NO_SPLIT;
//     int8_t groupListType = 0;
//     int8_t splitItem = 0;
//     int8_t actType = 0;
//     const char *opName = nullptr;
//     ge::DataType aDtype = ge::DT_INT8;
//     ge::DataType bDtype = ge::DT_INT8;
//     ge::DataType cDtype = ge::DT_FLOAT16;
//     ge::DataType biasDtype = ge::DT_INT32;
//     ge::DataType scaleDtype = ge::DT_UINT64;
//     ge::DataType perTokenScaleDtype = ge::DT_FLOAT;
//     ge::DataType outDataDtype = ge::DT_FLOAT16;
//     ge::DataType outScaleDtype = ge::DT_FLOAT;

//     ge::Format aFormat = ge::FORMAT_ND;
//     ge::Format bFormat = ge::FORMAT_ND;
//     ge::Format cFormat = ge::FORMAT_ND;
//     bool transA = false;
//     bool transB = false;
//     bool hasBias = false;
//     bool isSingleX = false;
//     bool isSingleW = false;
//     bool isSingleY = false;
// } inputParams_

namespace MC2Tiling {

}