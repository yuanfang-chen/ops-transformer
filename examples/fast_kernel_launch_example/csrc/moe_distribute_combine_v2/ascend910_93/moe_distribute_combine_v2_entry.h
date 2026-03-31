/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You can not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_distribute_combine_v2_entry.h
 * \brief Kernel entry declarations for direct launch
 */

#ifndef MOE_DISTRIBUTE_COMBINE_V2_ENTRY_H
#define MOE_DISTRIBUTE_COMBINE_V2_ENTRY_H

#include <cstdint>
#include "acl/acl.h"

void moe_distribute_combine_v2_entry(int32_t tilingKey, uint32_t blockDim, aclrtStream stream, void *mc2Context,
                                     void *expandX, void *expertIds, void *assistInfoForCombine, void *epSendCount,
                                     void *tpSendCount, void *scales, void *xActiveMask, void *sharedExpertX,
                                     void *elasticInfo, void *oriX, void *performanceInfo, void *xOut, void *workspace,
                                     const void *tilingData);

#endif