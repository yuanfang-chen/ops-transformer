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
 * \file expert_dispatch_tiling.h
 * \brief
 */
#ifndef EXPERT_DISPATCH_TILING_H
#define EXPERT_DISPATCH_TILING_H
#include <cmath>
#include <cstdint>
#include <algorithm>
#include "tiling/tiling_api.h"
#include "tiling_base/tiling_base.h"
#include "tiling_base/tiling_templates_registry.h"
#include "register/op_def_registry.h"
#include "log/log.h"
#include "platform/platform_infos_def.h"
#include "util/math_util.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling
{
BEGIN_TILING_DATA_DEF(ExpertVBSComputeTilingData)
TILING_DATA_FIELD_DEF(int64_t, needCoreNum);
TILING_DATA_FIELD_DEF(int64_t, perCoreElements);
TILING_DATA_FIELD_DEF(int64_t, perCoreLoops);
TILING_DATA_FIELD_DEF(int64_t, perCorePerLoopElements);
TILING_DATA_FIELD_DEF(int64_t, perCoreLastLoopElements);
TILING_DATA_FIELD_DEF(int64_t, lastCoreElements);
TILING_DATA_FIELD_DEF(int64_t, lastCoreLoops);
TILING_DATA_FIELD_DEF(int64_t, lastCorePerLoopElements);
TILING_DATA_FIELD_DEF(int64_t, lastCoreLastLoopElements);
TILING_DATA_FIELD_DEF(int64_t, oneLoopMaxElements);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(ExpertVBSComputeTilingDataOp, ExpertVBSComputeTilingData)

BEGIN_TILING_DATA_DEF(ExpertVMSMiddleComputeTilingData)
TILING_DATA_FIELD_DEF(int64_t, needCoreNum);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(ExpertVMSMiddleComputeTilingDataOp, ExpertVMSMiddleComputeTilingData)

BEGIN_TILING_DATA_DEF(ExpertSortOutComputeTilingData)
TILING_DATA_FIELD_DEF(int64_t, oneLoopMaxElements);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(ExpertSortOutComputeTilingDataOp, ExpertSortOutComputeTilingData)

BEGIN_TILING_DATA_DEF(ExpertTokensCountTilingData)
TILING_DATA_FIELD_DEF(int64_t, needCoreNum);               // AIV needed actual
TILING_DATA_FIELD_DEF(int64_t, perCoreElements);           // element number per AIV
TILING_DATA_FIELD_DEF(int64_t, lastCoreElements);          // element number for last AIV
TILING_DATA_FIELD_DEF(int64_t, perCoreLoops);              // loop times in per AIV
TILING_DATA_FIELD_DEF(int64_t, perCorePerLoopElements);    // element number in per loop of per AIV
TILING_DATA_FIELD_DEF(int64_t, perCoreLastLoopElements);   // element number in last loop of per AIV
TILING_DATA_FIELD_DEF(int64_t, lastCoreLoops);             // loop times in last AIV
TILING_DATA_FIELD_DEF(int64_t, lastCorePerLoopElements);   // element number in per loop of last AIV
TILING_DATA_FIELD_DEF(int64_t, lastCoreLastLoopElements);  // element number in last loop of last AIV
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(ExpertTokensCountTilingDataOp, ExpertTokensCountTilingData)

BEGIN_TILING_DATA_DEF(ExpertGatherOutComputeTilingData)
TILING_DATA_FIELD_DEF(int64_t, needCoreNum);                      // AIV needed actual
TILING_DATA_FIELD_DEF(int64_t, perCoreIndicesElements);           // element number per AIV
TILING_DATA_FIELD_DEF(int64_t, lastCoreIndicesElements);          // element number for last AIV
TILING_DATA_FIELD_DEF(int64_t, perCoreIndicesLoops);              // loop times in per AIV
TILING_DATA_FIELD_DEF(int64_t, perCorePerLoopIndicesElements);    // element number in per loop of per AIV
TILING_DATA_FIELD_DEF(int64_t, perCoreLastLoopIndicesElements);   // element number in last loop of per AIV
TILING_DATA_FIELD_DEF(int64_t, lastCoreIndicesLoops);             // loop times in last AIV
TILING_DATA_FIELD_DEF(int64_t, lastCorePerLoopIndicesElements);   // element number in per loop of last AIV
TILING_DATA_FIELD_DEF(int64_t, lastCoreLastLoopIndicesElements);  // element number in last loop of last AIV
TILING_DATA_FIELD_DEF(int64_t, colsLoops);                        // loop times for one cols
TILING_DATA_FIELD_DEF(int64_t, perLoopCols);                      // cols elements for per cols loop
TILING_DATA_FIELD_DEF(int64_t, lastLoopCols);                     // cols elements for last cols loop
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(ExpertGatherOutComputeTilingDataOp, ExpertGatherOutComputeTilingData)

BEGIN_TILING_DATA_DEF(ExpertDispatchTilingData)
TILING_DATA_FIELD_DEF(int64_t, coreNum);                                               // AIV Core number total
TILING_DATA_FIELD_DEF(int64_t, n);                                                     // batch * sequence
TILING_DATA_FIELD_DEF(int64_t, cols);                                                  // hidden size
TILING_DATA_FIELD_DEF(int64_t, k);                                                     // top k
TILING_DATA_FIELD_DEF(int64_t, expertStart);                                           // expert range start
TILING_DATA_FIELD_DEF(int64_t, expertEnd);                                             // expert range end
TILING_DATA_FIELD_DEF(int64_t, actualExpertNum);                                       // expert range stride
TILING_DATA_FIELD_DEF_STRUCT(ExpertVBSComputeTilingData, vbsComputeParamsOp);              // vbs sort Tiling Data
TILING_DATA_FIELD_DEF_STRUCT(ExpertVMSMiddleComputeTilingData, vmsMiddleComputeParamsOp);  // vms sort Tiling Data
TILING_DATA_FIELD_DEF_STRUCT(ExpertSortOutComputeTilingData, sortOutComputeParamsOp);      // sort out Tiling Data
TILING_DATA_FIELD_DEF_STRUCT(ExpertTokensCountTilingData, expertTokensCountTilingDataOp);  // Histogram Tiling Data
TILING_DATA_FIELD_DEF_STRUCT(ExpertGatherOutComputeTilingData, gatherOutComputeParamsOp);  // Gather Out Tiling Data
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(ExpertDispatch, ExpertDispatchTilingData)
struct ExpertDispatchCompileInfo {
};
}  // namespace optiling
#endif  // EXPERT_DISPATCH_H
