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
 * \file ts_aclnn_mhc_post.h
 * \brief MhcPost UTest 相关基类定义.
 */

#ifndef UTEST_TS_ACLNN_MHC_POST_H
#define UTEST_TS_ACLNN_MHC_POST_H

#include "tests/utest/ts.h"
#include "comm/inc/mhc_post_case.h"
#include "comm/inc/aclnn_mhc_post_case.h"

using ops::adv::tests::mhc_post::AclnnMhcPostCase;
using ops::adv::tests::mhc_post::GenTensor;
using AclnnMhcPostParam = ops::adv::tests::mhc_post::AclnnMhcPostParam;

class Ts_Aclnn_Mhc_Post_WithParam : public Ts_WithParam<AclnnMhcPostCase> {};
class Ts_Aclnn_Mhc_Post_WithParam_Ascend950 : public Ts_WithParam_Ascend950<AclnnMhcPostCase> {};

#endif // UTEST_TS_ACLNN_MHC_POST_H