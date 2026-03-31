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
 * \file all_gather_add_tiling_key.h
 * \brief TilingKey 宏参数模板声明
 */

#ifndef ALL_GATHER_ADD_TILING_KEY_H
#define ALL_GATHER_ADD_TILING_KEY_H

#include "ascendc/host_api/tiling/template_argument.h"

#define ALL_GATHER_ADD_SCH_MODE_BASIC 0

ASCENDC_TPL_ARGS_DECL(AllGatherAdd,
    ASCENDC_TPL_UINT_DECL(SCH_MODE, 1, ASCENDC_TPL_UI_LIST, ALL_GATHER_ADD_SCH_MODE_BASIC)
);

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(SCH_MODE, ASCENDC_TPL_UI_LIST, ALL_GATHER_ADD_SCH_MODE_BASIC)
    )
);

#endif // ALL_GATHER_ADD_TILING_KEY_H
