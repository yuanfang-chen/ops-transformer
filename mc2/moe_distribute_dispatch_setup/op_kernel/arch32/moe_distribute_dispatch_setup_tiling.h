/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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
 * \file moe_distribute_dispatch_setup_tiling.h
 * \brief
 */
#ifndef ASCENDC_MOE_DISTRIBUTE_DISPATCH_SETUP_TILING_H
#define ASCENDC_MOE_DISTRIBUTE_DISPATCH_SETUP_TILING_H

struct MoeDistributeDispatchSetupInfo {
    uint32_t epWorldSize;
    uint32_t epRankId;
    uint32_t expertShardType;
    uint32_t sharedExpertNum;
    uint32_t sharedExpertRankNum;
    uint32_t moeExpertNum;
    uint32_t moeExpertPerRankNum;
    uint32_t quantMode;
    uint32_t globalBs;
    uint32_t bs;
    uint32_t k;
    uint32_t h;
    uint32_t aivNum;
    uint32_t sdmaUsedStreamPerCore;
    bool isQuant;
    bool isActiveMask;
    bool reserved2;
    bool reserved3;
    uint64_t totalUbSize;
    uint64_t totalWinSize;
};

struct MoeDistributeDispatchSetupTilingData {
    Mc2InitTiling mc2InitTiling;
    Mc2CcTiling mc2CcTiling;
    MoeDistributeDispatchSetupInfo moeDistributeDispatchSetupInfo;
};

#endif