/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file chunk_gated_delta_rule_inverse.cc
 * \brief
 */
#include <map>
#include <string>
#include <sstream>
#include <initializer_list>

#include "exe_graph/runtime/infer_shape_context.h"
#include "exe_graph/runtime/shape.h"
#include "exe_graph/runtime/storage_shape.h"
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "../../../common/include/err/ops_err.h"

using namespace gert;
using namespace ge;

namespace ops {

const constexpr int64_t X_INDEX = 0;
const constexpr int64_t PHI_INDEX = 1;

const constexpr int64_t OUT_H_IN_INDEX = 0;
const constexpr int64_t OUT_H_POST_INDEX = 1;
const constexpr int64_t OUT_H_RES_INDEX = 2;
const constexpr int64_t OUT_INV_RMS_INDEX = 3;
const constexpr int64_t OUT_MM_RES_INDEX = 4;
const constexpr int64_t OUT_H_PRE_INDEX = 5;

const constexpr int64_t BSND_DIM_NUM = 4;
const constexpr int64_t TND_DIM_NUM = 3;

const constexpr int64_t INDEX_B_BSND = 0;
const constexpr int64_t INDEX_S_BSND = 1;
const constexpr int64_t INDEX_N_BSND = 2;
const constexpr int64_t IEDEX_D_BSND = 3;

const constexpr int64_t INDEX_T_TND = 0;
const constexpr int64_t INDEX_N_TND = 1;
const constexpr int64_t INDEX_D_TND = 2;

static ge::graphStatus InferShape4mHCPre(InferShapeContext *context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShape ManifoldConstrainedHyperConnectionPre");
    const gert::Shape *xShape = context->GetDynamicInputShape(X_INDEX, 0);
        OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    const gert::Shape *phiShape = context->GetDynamicInputShape(PHI_INDEX, 0);
        OP_CHECK_NULL_WITH_CONTEXT(context, phiShape);

    int64_t xShapeDim = xShape->GetDimNum();
    int64_t phiShapeDim = xShape->GetDimNum();
    auto outHinShape = context->GetOutputShape(OUT_H_IN_INDEX);
    auto outHpostShape = context->GetOutputShape(OUT_H_POST_INDEX);
    auto outHresShape = context->GetOutputShape(OUT_H_RES_INDEX);

    auto outInvRmsShape = context->GetOutputShape(OUT_INV_RMS_INDEX);
    auto outMmresShape = context->GetOutputShape(OUT_MM_RES_INDEX);
    auto outHpreShape = context->GetOutputShape(OUT_H_PRE_INDEX);

    if (phiShapeDim < 2) {
        OP_LOGD(context->GetNodeName(), "phiShapeDim dims is invalid");
        return GRAPH_FAILED;
    }

    uint64_t matK = phiShape->GetDim(0);
    if (xShapeDim == BSND_DIM_NUM) {
        uint64_t batch = xShape->GetDim(INDEX_B_BSND);
        uint64_t sequence = xShape->GetDim(INDEX_S_BSND);
        uint64_t numsResidual = xShape->GetDim(INDEX_N_BSND);
        uint64_t dimen = xShape->GetDim(IEDEX_D_BSND);

        outHinShape->SetDimNum(BSND_DIM_NUM - 1);
        outHinShape->SetDim(0, batch);
        outHinShape->SetDim(1, sequence);
        outHinShape->SetDim(2, dimen);

        outHpostShape->SetDimNum(BSND_DIM_NUM - 1);
        outHpostShape->SetDim(0, batch);
        outHpostShape->SetDim(1, sequence);
        outHpostShape->SetDim(2, numsResidual);

        outHresShape->SetDimNum(BSND_DIM_NUM);
        outHresShape->SetDim(0, batch);
        outHresShape->SetDim(1, sequence);
        outHresShape->SetDim(2, numsResidual);
        outHresShape->SetDim(3, numsResidual);

        outInvRmsShape->SetDimNum(BSND_DIM_NUM - 2);
        outInvRmsShape->SetDim(0, batch);
        outInvRmsShape->SetDim(1, sequence);

        outMmresShape->SetDimNum(BSND_DIM_NUM - 1);
        outMmresShape->SetDim(0, batch);
        outMmresShape->SetDim(1, sequence);
        outMmresShape->SetDim(2, matK);

        outHpreShape->SetDimNum(BSND_DIM_NUM - 1);
        outHpreShape->SetDim(0, batch);
        outHpreShape->SetDim(1, sequence);
        outHpreShape->SetDim(2, numsResidual);

    } else if (xShapeDim == TND_DIM_NUM) {
        uint64_t t = xShape->GetDim(INDEX_T_TND);
        uint64_t numsResidual = xShape->GetDim(INDEX_N_TND);
        uint64_t dimen = xShape->GetDim(INDEX_D_TND);

        outHinShape->SetDimNum(TND_DIM_NUM - 1);
        outHinShape->SetDim(0, t);
        outHinShape->SetDim(1, dimen);

        outHpostShape->SetDimNum(TND_DIM_NUM - 1);
        outHpostShape->SetDim(0, t);
        outHpostShape->SetDim(1, numsResidual);

        outHresShape->SetDimNum(TND_DIM_NUM);
        outHresShape->SetDim(0, t);
        outHresShape->SetDim(1, numsResidual);
        outHresShape->SetDim(2, numsResidual);

        outInvRmsShape->SetDimNum(TND_DIM_NUM - 2);
        outInvRmsShape->SetDim(0, t);

        outMmresShape->SetDimNum(TND_DIM_NUM - 1);
        outMmresShape->SetDim(0, t);
        outMmresShape->SetDim(1, matK);

        outHpreShape->SetDimNum(TND_DIM_NUM - 1);
        outHpreShape->SetDim(0, t);
        outHpreShape->SetDim(1, numsResidual);
    }

    OP_LOGD(context->GetNodeName(), "End to do InferShape ManifoldConstrainedHyperConnectionPre");
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType4mHCPre(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, DataType::DT_FLOAT);
    context->SetOutputDataType(0, DataType::DT_FLOAT);
    context->SetOutputDataType(0, DataType::DT_FLOAT);
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(ManifoldConstrainedHyperConnectionPre)
    .InferShape(InferShape4mHCPre)
    .InferDataType(InferDataType4mHCPre);
} // namespace ops