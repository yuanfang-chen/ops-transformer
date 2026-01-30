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
 * \file all_gather_matmul_inner.cpp
 * \brief
 */
#include "basic_api/kernel_basic_intf.h"
#include "lib/matmul_intf.h"
#include "all_gather_matmul_tiling.h"

using namespace AscendC;

__global__ __aicore__ void all_gather_matmul_inner(GM_ADDR aGM, GM_ADDR ctxGM, 
    GM_ADDR cGM, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    // v核调用init和prepare接口
    if (workspaceGM == nullptr) {
        return;
    }
    if (g_coreType == AIC) {
        return;
    }
    GM_ADDR userWS = GetUserWorkspace(workspaceGM);
    if (userWS == nullptr) {
        return;
    }

    REGISTER_TILING_DEFAULT(AllGatherCustomV3TilingData);
    GET_TILING_DATA_WITH_STRUCT(AllGatherCustomV3TilingData, tilingData, tilingGM);

    auto &&cfg = tilingData.param;
    uint8_t tileNum = cfg.tileNum;
    uint8_t tailNum = cfg.tailNum;
    uint64_t tileM = cfg.tileM;
    uint64_t tailM = cfg.tailM;
    uint64_t strideCount = cfg.strideCount;
    const uint64_t M = cfg.M;
    const uint64_t K = cfg.K;

    uint64_t TileCnt = tileM * K;
    uint64_t TailCnt = tailM * K;
    uint64_t TileOffset = TileCnt * cfg.dataTypeSize;

    TPipe pipe;
    Hccl<HcclServerType::HCCL_SERVER_TYPE_AICPU> hccl;
    uint64_t ctx = *(__gm__ uint64_t *)ctxGM;
    hccl.InitV2((GM_ADDR)ctx, nullptr);
    if (TILING_KEY_IS(100UL)) {
        tileM = M;
        tailM = 0;
        tileNum = 1;
        tailNum = 0;
        strideCount = 0;
        TileCnt = tileM * K;
        TailCnt = tailM * K;

        auto handleId = hccl.AllGather<true>(aGM, cGM, TileCnt, HcclDataType(cfg.dataType), strideCount, tileNum);
        hccl.Wait(handleId);
    } else if (TILING_KEY_IS(101UL)) {
        // 下发allgather任务
        // 首块
        auto handleId = hccl.AllGather<true>(aGM, cGM, TileCnt, HcclDataType(cfg.dataType), strideCount, tileNum);
        // 尾块
        auto tailHandleId = hccl.AllGather<true>(aGM + tileNum * TileOffset, cGM + tileNum * TileOffset, TailCnt,
                                                 HcclDataType(cfg.dataType), strideCount, tailNum);
        for (uint8_t i=0; i<tileNum; i++) {
            hccl.Wait(handleId);
        }
        hccl.Wait(tailHandleId);
    }
    SyncAll(); // 纯v核同步
    hccl.Finalize();

}