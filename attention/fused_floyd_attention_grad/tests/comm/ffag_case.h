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
 * \file ffag_case.h
 * \brief FusedFloydAttentionGrad 测试用例.
 */

#pragma once

#include <vector>
#include <cstdint>
#include "graph/types.h"
#include "tests/utils/case.h"
#include "tests/utils/op_info.h"
#include "tests/utils/context.h"
#include "tests/utils/tensor.h"
#include <exe_graph/runtime/tiling_context.h>
#include <register/op_impl_registry.h>

namespace ops::adv::tests::ffag {
class FfagCase : public ops::adv::tests::utils::Case {
    using OpInfo = ops::adv::tests::utils::OpInfo;
    using Context = ops::adv::tests::utils::Context;
    using Tensor = ops::adv::tests::utils::Tensor;

public:
    enum class AttenMaskShapeType {
        NONE,
        B_1_S1_1_S3,
    };

    class Param {
    public:
        int64_t b = 0;
        int64_t n = 0;
        int64_t s1 = 0;
        int64_t s2 = 0;
        int64_t s3 = 0;
        int64_t d = 0;
        float scaleValue = 1.0f;
        ge::DataType qkvDataType = ge::DataType::DT_FLOAT16;
        ge::DataType outDataType = ge::DataType::DT_FLOAT16;
        AttenMaskShapeType attenMaskType = AttenMaskShapeType::B_1_S1_1_S3;
        Param();
        Param(int64_t pB, int64_t pN, int64_t pS1, int64_t pS2, int64_t pS3, int64_t pD,
              float pScaleValue);
    };

    class DoTilingParam {
        public:
        gert::TilingContext* ctx = nullptr;
        ge::graphStatus ret = ge::GRAPH_SUCCESS;
    };
 
    Tensor query, key1, key2, value1, value2, dy, attenMask, softmaxMax, softmaxSum, attentionIn, dq, dk1, dk2, dv1, dv2;
    OpInfo mOpInfo;
    Context mCtx;
    Param mParam;
    gert::OpImplRegisterV2::TilingKernelFunc ffagTilingFunc = nullptr;
    FfagCase();
    FfagCase(const char *name, bool enable, const char *dbgInfo, OpInfo mOpInfo, Param param);
    bool Run() override;
    bool InitParam() override;
    bool InitOpInfo() override;
    bool InitCurrentCasePtr() override;
    bool DoOpTiling(DoTilingParam& tilingParam);
};

} // namespace ops::adv::tests::ffag
