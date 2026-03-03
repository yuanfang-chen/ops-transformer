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
 * \file reduce_scatter_formulaic_tiling.cpp
 * \brief
 */
#include <iostream>
#include "mc2_log.h"
#include "reduce_scatter_fit_balance_tiling.h"

constexpr static double L2_CACHE_SIZE = 128 * ONE_MBYTE;

void MMReduceScatterFitBalanceTiling::EstimateMMCommTime()
{
    matmulPerf_.FindCubeUtil(rankTileNum_);
    matmulPerf_.GetMatmulGradient();

    // Find total matmul time and comm time
    double totalMatmulTime = matmulPerf_.MatmulTime(mmInfo_.mValue, rankDim_);
    double totalTpTime = commPerf_.CommTime(mmInfo_.mValue);
    ratioCalcComm_ = (std::max(totalTpTime, totalMatmulTime) / std::min(totalTpTime, totalMatmulTime));
    if (totalMatmulTime >= totalTpTime) {
        tilingM_.cutRes.shortTileAtBack = true;
    }

    uint64_t sizeOfComm = mmInfo_.mValue * mmInfo_.nValue * commPerf_.GetCommDTypeSize() / ONE_MBYTE;
    OP_LOGD("MatmulReduceScatter", "Input shape {M, N, K} = {%lu, %lu, %lu}, cubeUtil_ %f, sizeOfComm %lu, "
        "totalMatmulTime %f, totalCommTime %f, minTileSize %lu, mAlignLen %lu, commTimeFactor_ %f, "
        "rankDim_ %lu, rankTile %lu",
        mmInfo_.mValue, mmInfo_.nValue, mmInfo_.kValue, matmulPerf_.cubeUtil_, sizeOfComm,
        totalMatmulTime, totalTpTime, tilingM_.GetMinLen(), tilingM_.GetAlignLength(), commPerf_.commTimeFactor_,
        rankDim_, rankTileNum_);
}

void MMReduceScatterFitBalanceTiling::SetShortTileLen()
{
    uint64_t l2UseSize = mmInfo_.mValue * mmInfo_.kValue * mmInfo_.inMatrixADtypeSize +
        mmInfo_.kValue * mmInfo_.nValue * mmInfo_.inMatrixBDtypeSize;
    if (l2UseSize > L2_CACHE_SIZE && mmInfo_.mValue > 512) {
        tilingM_.SetMinLenByMax(tilingM_.GetMinLen() * TWO);
        isLargerThanL2_ = true;
    }

    tilingM_.cutRes.shortTileLen = tilingM_.GetMinLen();
    tilingM_.cutRes.numShortTile = 1U;
}


void MMReduceScatterFitBalanceTiling::AdjustLongShortTileLen()
{
    bool goodLinearityShape = (mmInfo_.kValue * mmInfo_.nValue >= LARGE_NK_BAR_BASE * ONE_MBYTE);
    tilingM_.FitTileLengthDiscrete(false, goodLinearityShape);
    // When the long and short tiles are equal, the long and short pieces become one.
    if (tilingM_.cutRes.shortTileLen == tilingM_.cutRes.longTileLen) {
        tilingM_.cutRes.shortTileLen = 0U;
        tilingM_.cutRes.numShortTile = 0U;
        tilingM_.cutRes.numLongTile++;
    }
}
