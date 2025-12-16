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
 * \file ts_ffag_tc.cpp
 * \brief FusedFloydAttentionGrad用例.
 */

#include "ts_ffag.h"

TEST_F(Ts_Ffag_Ascend910B2, case_empty_query)
{
    FfagCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 4;
    cs.mParam.s1 = 128;
    cs.mParam.s2 = 256;
    cs.mParam.s3 = 256;
    cs.mParam.d = 64;
    cs.mParam.scaleValue = 1.0f;
    cs.mOpInfo.mExp.mSuccess = true;
    ASSERT_TRUE(cs.Init());
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}
