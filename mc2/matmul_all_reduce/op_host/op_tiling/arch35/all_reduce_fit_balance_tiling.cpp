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
 * \file all_reduce_fit_balance_tiling.cpp
 * \brief
 */
#include "all_reduce_fit_balance_tiling.h"
#include <cmath>
#include "mc2_log.h"

constexpr static double LARGE_BACKTILE_CALC_COMM_RATIO_BAR = 1.75;
constexpr static double MM_EXPANSION_TIME = 30;
constexpr static double COMM_EXPANSION_TIME = 40;
constexpr static uint64_t L2_CACHE_SIZE = 128 * ONE_MBYTE;

void MMAllReduceFitBalanceTiling::EstimateMMCommTime()
{
    matmulPerf_.FindCubeUtil(rankTileNum_);
    matmulPerf_.GetMatmulGradient();

    // Find total matmul time and comm time
    double totalMatmulTime = matmulPerf_.MatmulTime(mmInfo_.mValue, 1);
    double totalTpTime = commPerf_.CommTime(mmInfo_.mValue);
    if (totalMatmulTime >= totalTpTime) {
        tilingM_.cutRes.shortTileAtBack = true;
    }

    ratioCalcComm_ = (std::max(totalTpTime, totalMatmulTime) / std::min(totalTpTime, totalMatmulTime));

    uint64_t sizeOfComm = mmInfo_.mValue * mmInfo_.nValue * commPerf_.GetCommDTypeSize() / ONE_MBYTE;
    OP_LOGD("MatmulAllReduce", "Input shape {M, N, K} = {%lu, %lu, %lu}, cubeUtil_ %f, sizeOfComm %lu, "
        "totalMatmulTime %f, totalCommTime %f, minTileSize %lu, mAlignLen %lu, commTimeFactor_ %f, "
        "rankDim_ %lu, rankTile %lu",
        mmInfo_.mValue, mmInfo_.nValue, mmInfo_.kValue, matmulPerf_.cubeUtil_, sizeOfComm,
        totalMatmulTime, totalTpTime, tilingM_.GetMinLen(), tilingM_.GetAlignLength(), commPerf_.commTimeFactor_,
        rankDim_, rankTileNum_);
}

void MMAllReduceFitBalanceTiling::SetShortTileLen()
{
    if (tilingM_.cutRes.shortTileAtBack) {
        tilingM_.SetMinLenByMax(commPerf_.InverseCommTime(COMM_EXPANSION_TIME));
    } else {
        tilingM_.SetMinLenByMax(matmulPerf_.InverseMatmulTime(MM_EXPANSION_TIME, rankTileNum_));
    }
    // Encourage split more if the comm and calc is balanced and the cost of cutLen is sufficiently high
    bool isCalcCommBalance = ratioCalcComm_ < CALC_COMM_RATIO;
    uint64_t cutLen = tilingM_.GetAlignLength() / TWO;
    double mmCost = matmulPerf_.MatmulTime(cutLen, 1);
    double commCost = commPerf_.CommTime(cutLen);
    uint64_t l2UseSize = mmInfo_.mValue * mmInfo_.kValue * mmInfo_.inMatrixADtypeSize +
        mmInfo_.kValue * mmInfo_.nValue * mmInfo_.inMatrixBDtypeSize;
    if (l2UseSize > L2_CACHE_SIZE) {
        tilingM_.SetMinLenByMax(tilingM_.GetMinLen() * TWO);
    } else if (isCalcCommBalance && (mmCost > MM_EXPANSION_TIME) && (commCost > COMM_EXPANSION_TIME) && (l2UseSize < L2_CACHE_SIZE)) {
        tilingM_.SetAlignLength(cutLen);
        tilingM_.SetMinLenByMin(cutLen);
    }
    tilingM_.cutRes.shortTileLen = tilingM_.GetMinLen();
    tilingM_.cutRes.numShortTile = 1U;
}

void MMAllReduceFitBalanceTiling::AdjustLongShortTileLen()
{
    // Adjusting the tiling based on constraints such as the size of the first and last blocks and the number of rounds.
    tilingM_.GenerateInitialPartition();
    bool largeCalcCommRatio = ratioCalcComm_ > LARGE_BACKTILE_CALC_COMM_RATIO_BAR;
    bool kGreaterThanN = mmInfo_.kValue > mmInfo_.nValue;
    tilingM_.FitTileLengthContinuous(kGreaterThanN, largeCalcCommRatio, false);
    // When the long and short tiles are equal, the long and short pieces become one.
    if (tilingM_.cutRes.shortTileLen == tilingM_.cutRes.longTileLen) {
        tilingM_.cutRes.shortTileLen = 0U;
        tilingM_.cutRes.numShortTile = 0U;
        tilingM_.cutRes.numLongTile++;
    }
}