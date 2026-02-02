/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file moe_distribute_dispatch_setup.cpp
 * \brief
 */

 #include "kernel_operator.h"
 #include "moe_distribute_dispatch_setup.h"
 #include "../moe_distribute_dispatch_setup/moe_distribute_dispatch_setup_tiling.h"
 
 using namespace AscendC;
 using namespace MoeDistributeDispatchSetupImpl;
 extern "C" __global__ __aicore__ void moe_distribute_dispatch_setup(
     GM_ADDR x, GM_ADDR expertIds, GM_ADDR scales, GM_ADDR xActiveMask, GM_ADDR YOut, GM_ADDR expandIdxOut, GM_ADDR commCmdInfoOut,
     GM_ADDR workspaceGM, GM_ADDR tilingGM)
 {
     REGISTER_TILING_DEFAULT(MoeDistributeDispatchSetupTilingData);
     TPipe pipe;
     auto tiling = (__gm__ MoeDistributeDispatchSetupTilingData*)tilingGM;
     __gm__ void* mc2InitTiling = (__gm__ void*)(&(tiling->mc2InitTiling));
     __gm__ void* mc2CcTiling = (__gm__ void*)(&(tiling->mc2CcTiling));
 #if (ORIG_DTYPE_Y == DT_BF16 || ORIG_DTYPE_Y == DT_FLOAT16)
     if (TILING_KEY_IS(1000)) {
         GET_TILING_DATA_WITH_STRUCT(MoeDistributeDispatchSetupTilingData, tilingData, tilingGM);
         MoeDistributeDispatchSetup<DTYPE_X, DTYPE_Y, false, false, false> op;
         op.Init(x, expertIds, scales, xActiveMask, YOut, expandIdxOut, commCmdInfoOut,
                 workspaceGM, &pipe, &tilingData, mc2InitTiling, mc2CcTiling);
         op.Process();
     } else if (TILING_KEY_IS(1100)) {
         GET_TILING_DATA_WITH_STRUCT(MoeDistributeDispatchSetupTilingData, tilingData, tilingGM);
         MoeDistributeDispatchSetup<DTYPE_X, DTYPE_Y, false, false, false> op;
         op.Init(x, expertIds, scales, xActiveMask, YOut, expandIdxOut, commCmdInfoOut,
            workspaceGM, &pipe, &tilingData, mc2InitTiling, mc2CcTiling);
         op.Process();
     }
 #elif (ORIG_DTYPE_Y == DT_INT8)
     if (TILING_KEY_IS(1011)) {
         GET_TILING_DATA_WITH_STRUCT(MoeDistributeDispatchSetupTilingData, tilingData, tilingGM);
         MoeDistributeDispatchSetup<DTYPE_X, DTYPE_Y, true, false, false> op;
         op.Init(x, expertIds, scales, xActiveMask, YOut, expandIdxOut, commCmdInfoOut,
            workspaceGM, &pipe, &tilingData, mc2InitTiling, mc2CcTiling);
         op.Process();
     } else if (TILING_KEY_IS(1002)) {
         GET_TILING_DATA_WITH_STRUCT(MoeDistributeDispatchSetupTilingData, tilingData, tilingGM);
         MoeDistributeDispatchSetup<DTYPE_X, DTYPE_Y, false, true, false> op;
         op.Init(x, expertIds, scales, xActiveMask, YOut, expandIdxOut, commCmdInfoOut,
            workspaceGM, &pipe, &tilingData, mc2InitTiling, mc2CcTiling);
         op.Process();
     } else if (TILING_KEY_IS(1012)) {
         GET_TILING_DATA_WITH_STRUCT(MoeDistributeDispatchSetupTilingData, tilingData, tilingGM);
         MoeDistributeDispatchSetup<DTYPE_X, DTYPE_Y, false, true, true> op;
         op.Init(x, expertIds, scales, xActiveMask, YOut, expandIdxOut, commCmdInfoOut,
            workspaceGM, &pipe, &tilingData, mc2InitTiling, mc2CcTiling);
         op.Process();
     }
 #endif
 }