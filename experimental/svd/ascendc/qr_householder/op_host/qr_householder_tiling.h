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
 * \file qr_householder_tiling.h
 * \brief
 */

#ifndef QR_HOUSEHOLDER_TILING_H_
#define QR_HOUSEHOLDER_TILING_H_

#include "exe_graph/runtime/tiling_context.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "error/ops_error.h"
#include "platform/platform_info.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(QrHouseholderTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, batchSize);
    TILING_DATA_FIELD_DEF(uint32_t, mDim);
    TILING_DATA_FIELD_DEF(uint32_t, kDim);
    TILING_DATA_FIELD_DEF(uint32_t, ubSize);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(QrHouseholder, QrHouseholderTilingData)
} // namespace optiling

#endif // QR_HOUSEHOLDER_TILING_H_