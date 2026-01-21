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
 * \file elastic_recievable_info_collect.cpp
 * \brief
 */

#include "basic_api/kernel_basic_intf.h"
#include "elastic_receivable_info_collect_tiling.h"
#include "elastic_receivable_info_collect.h"

using namespace AscendC;
using namespace ElasticReceivableInfoCollectImpl;

extern "C" __global__ __aicore__ void elastic_receivable_info_collect(GM_ADDR y, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    REGISTER_TILING_DEFAULT(ElasticReceivableInfoCollectTilingData);
    TPipe pipe;

    if (TILING_KEY_IS(10000)) {
        GET_TILING_DATA_WITH_STRUCT(ElasticReceivableInfoCollectTilingData, tilingData, tilingGM);
        ElasticReceivableInfoCollect<DTYPE_Y> op;
        op.Init(y, workspaceGM, &pipe, &tilingData);
        op.Process();
    }
}