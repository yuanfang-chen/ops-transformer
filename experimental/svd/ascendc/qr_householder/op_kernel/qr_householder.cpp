/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file qr_householder.cpp
 * \brief
 */

#include "qr_householder_single_vec.h"
using namespace AscendC;


extern "C" __global__ __aicore__ void qr_householder(GM_ADDR input_x,
                                                     GM_ADDR output_q, GM_ADDR output_r,
                                                     GM_ADDR workspace, GM_ADDR tiling) {
    TPipe pipe;
    GET_TILING_DATA(tiling_data, tiling);
    GM_ADDR user = GetUserWorkspace(workspace);
    QRHouseholderSingleVec op;
    op.Init(input_x, output_q, output_r, user, &tiling_data, &pipe);
    op.Process();
    pipe.Destroy();
}