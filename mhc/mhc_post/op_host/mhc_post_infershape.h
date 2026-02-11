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
 * \file mhc_post_infershape.h
 * \brief
 */

#ifndef OPS_BUILT_MHC_POST_INFERSHAPE_H
#define OPS_BUILT_MHC_POST_INFERSHAPE_H

#include "graph/operator.h"

namespace ge {
graphStatus MhcPostInferShape(Operator &op);
}
#endif // OPS_BUILT_MHC_POST_INFERSHAPE_H