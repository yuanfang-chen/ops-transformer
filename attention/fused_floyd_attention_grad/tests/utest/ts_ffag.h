/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*!
 * \file ts_ffag.h
 * \brief FusedFloydAttentionGrad UTest 相关基类定义.
 */

#include "tests/utest/ts.h"
#include "ffag_case.h"

using FfagCase = ops::adv::tests::ffag::FfagCase;
using AttenMaskShapeType = FfagCase::AttenMaskShapeType;

class Ts_Ffag : public Ts<FfagCase> {};
class Ts_Ffag_Ascend910B2 : public Ts_Ascend910B2<FfagCase> {};

class Ts_Ffag_WithParam : public Ts_WithParam<FfagCase> {};
class Ts_Ffag_WithParam_Ascend910B2 : public Ts_WithParam_Ascend910B2<FfagCase> {};
