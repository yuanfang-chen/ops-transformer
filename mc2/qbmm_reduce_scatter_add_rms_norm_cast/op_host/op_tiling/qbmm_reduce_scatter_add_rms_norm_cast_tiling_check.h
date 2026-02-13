/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef QBMM_REDECE_SCATTER_ADD_RMS_NORM_CAST_TILING_CHECK_H
#define QBMM_REDECE_SCATTER_ADD_RMS_NORM_CAST_TILING_CHECK_H
#include "tiling/mc2_tiling_utils.h"
namespace MC2Tiling {
class QbmmReduceScatterAddRmsNormCastCheckTiling {
public:
    static ge::graphStatus CheckAttrs(const gert::TilingContext *context);
    static bool CheckTensorDataType(const gert::TilingContext *context);
    static bool CheckTensorDim(const gert::TilingContext *context);
    static bool CheckTensorFormat(const gert::TilingContext *context);
    static bool CheckWindowSize(const gert::TilingContext *context);
    static ge::graphStatus TilingCheckQbmmReduceScatterAddRmsNormCast(const gert::TilingContext *context);
};
}; // namespace MC2Tiling
#endif //__QBMM_REDECE_SCATTER_ADD_RMS_NORM_CAST_TILING_CHECK_H__