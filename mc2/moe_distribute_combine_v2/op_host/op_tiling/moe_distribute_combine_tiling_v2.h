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
 * \file moe_distribute_combine_tiling_v2.h
 * \brief
 */

#ifndef MOE_DISTRIBUTE_COMBINE_TILING_V2
#define MOE_DISTRIBUTE_COMBINE_TILING_V2

#include "tiling/mc2_tiling_utils.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {

struct CombineV2Config {
    uint32_t contextIndex = 0;
    uint32_t expandXIndex = 0;
    uint32_t expertIdsIndex = 1;
    uint32_t assistInfoIndex = 2;
    uint32_t epSendCountIndex = 3;
    uint32_t expertScalesIndex = 4;
    uint32_t residualXIndex = 5; // 根据combineARN算子原型标志位初始化residualX索引
    uint32_t gammaIndex = 6; // 根据combineARN算子原型标志位初始化gamma索引
    uint32_t tpSendCountsIndex = 5; // 根据combineV2算子原型标志位初始化tpSendCounts索引
    uint32_t xActiveMaskIndex = 6; // 根据combineV2算子原型标志位初始化xActiveMask索引
    uint32_t activationScaleIndex = 7; // 根据combineV2算子原型标志位初始化activationScale索引
    uint32_t weightScaleIndex = 8; // 根据combineV2算子原型标志位初始化weightScale索引
    uint32_t groupListIndex = 9; // 根据combineV2算子原型标志位初始化groupList索引
    uint32_t sharedExpertXIndex = 10; // 根据combineV2算子原型标志位初始化sharedExpertX索引
    uint32_t elasticInfoIndex = 11; // 根据combineV2算子原型标志位初始化elasticInfo索引
    uint32_t oriXIndex = 13; // 根据combineV2算子原型标志位初始化oriX索引
    uint32_t constExpertAlpha1Index = 14; // 根据combineV2算子原型标志位初始化constExpertAlpha1索引
    uint32_t constExpertAlpha2Index = 15; // 根据combineV2算子原型标志位初始化constExpertAlpha2索引
    uint32_t constExpertVIndex = 16; // 根据combineV2算子原型标志位初始化constExpertV索引
    uint32_t performanceInfoIndex = 17; // 根据combineV2算子原型标志位初始化 performanceInfo索引
    uint32_t outputYIndex = 0; // 根据combineARN算子原型标志位初始化outputY索引
    uint32_t outputRstdIndex = 1; // 根据combineARN算子原型标志位初始化outputRstd索引
    uint32_t outputXIndex = 0; // 根据combineV2算子原型标志位初始化outputX索引
    uint32_t attrNormEpsIndex = 15; // 根据combineARN算子原型标志位初始化attrNormEps索引
    uint32_t attrZeroExpertNumIndex = 15; // 根据combineV2算子原型标志位初始化attrZeroExpertNum索引
    uint32_t attrCopyExpertNumIndex = 16; // 根据combineV2算子原型标志位初始化attrCopyExpertNum索引
    uint32_t attrConstExpertNumIndex = 17; // 根据combineV2算子原型标志位初始化attrConstExpertNum索引
    bool hasAddRmsNorm = false;
    bool isMc2Context = false;
};

ge::graphStatus MoeDistributeCombineV2TilingFuncNew(gert::TilingContext* context, const CombineV2Config& config);

}

#endif