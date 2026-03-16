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
 * \file aclnn_chunk_gated_delta_rule_recurrence.cpp
 * \brief
 */
#include "aclnn_chunk_gated_delta_rule_recurrence.h"
#include "chunk_gated_delta_rule_recurrence.h"

#include "securec.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "aclnn_kernels/contiguous.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {

static const std::initializer_list<op::DataType> FLOAT_TYPE_LIST  = {op::DataType::DT_FLOAT};
static const std::initializer_list<op::DataType> INT32_TYPE_LIST   = {op::DataType::DT_INT32};

struct CGDRParams {
    aclTensor       *initialState {nullptr};
    const aclTensor *kgexp        {nullptr};
    const aclTensor *value        {nullptr};
    const aclTensor *kCumdecay    {nullptr};
    const aclTensor *qgexp        {nullptr};
    const aclTensor *gexp         {nullptr};
    const aclTensor *cuSeqlens    {nullptr};
    float            scaleValue   {1.0f};
    aclTensor       *attnInterOut {nullptr};
    aclTensor       *vNewOut      {nullptr};
};

static inline bool CheckNotNull(const CGDRParams &p)
{
    OP_CHECK_NULL(p.initialState,  return false);
    OP_CHECK_NULL(p.kgexp,         return false);
    OP_CHECK_NULL(p.value,         return false);
    OP_CHECK_NULL(p.kCumdecay,     return false);
    OP_CHECK_NULL(p.qgexp,         return false);
    OP_CHECK_NULL(p.gexp,          return false);
    OP_CHECK_NULL(p.cuSeqlens,     return false);
    OP_CHECK_NULL(p.attnInterOut,  return false);
    OP_CHECK_NULL(p.vNewOut,       return false);
    return true;
}

static inline bool CheckDtype(const CGDRParams &p)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(p.initialState,  FLOAT_TYPE_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(p.kgexp,         FLOAT_TYPE_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(p.value,         FLOAT_TYPE_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(p.kCumdecay,     FLOAT_TYPE_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(p.qgexp,         FLOAT_TYPE_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(p.gexp,          FLOAT_TYPE_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(p.cuSeqlens,     INT32_TYPE_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(p.attnInterOut,  FLOAT_TYPE_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(p.vNewOut,       FLOAT_TYPE_LIST, return false);
    return true;
}

static aclnnStatus PreProcess(CGDRParams &p)
{
    p.initialState->SetOriginalShape(p.initialState->GetViewShape());
    p.kgexp->SetOriginalShape(p.kgexp->GetViewShape());
    p.value->SetOriginalShape(p.value->GetViewShape());
    p.kCumdecay->SetOriginalShape(p.kCumdecay->GetViewShape());
    p.qgexp->SetOriginalShape(p.qgexp->GetViewShape());
    p.gexp->SetOriginalShape(p.gexp->GetViewShape());
    p.cuSeqlens->SetOriginalShape(p.cuSeqlens->GetViewShape());
    p.attnInterOut->SetOriginalShape(p.attnInterOut->GetViewShape());
    p.vNewOut->SetOriginalShape(p.vNewOut->GetViewShape());
    return ACLNN_SUCCESS;
}

} // namespace

aclnnStatus aclnnChunkGatedDeltaRuleRecurrenceGetWorkspaceSize(
    aclTensor       *initialState,
    const aclTensor *kgexp,
    const aclTensor *value,
    const aclTensor *kCumdecay,
    const aclTensor *qgexp,
    const aclTensor *gexp,
    const aclTensor *cuSeqlens,
    float            scaleValue,
    aclTensor       *attnInterOut,
    aclTensor       *vNewOut,
    uint64_t        *workspaceSize,
    aclOpExecutor  **executor)
{
    L2_DFX_PHASE_1(aclnnChunkGatedDeltaRuleRecurrence,
                   DFX_IN(initialState, kgexp, value, kCumdecay, qgexp, gexp, cuSeqlens, scaleValue),
                   DFX_OUT(attnInterOut, vNewOut, initialState));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    CGDRParams params{initialState, kgexp, value, kCumdecay, qgexp, gexp,
                      cuSeqlens, scaleValue, attnInterOut, vNewOut};

    CHECK_RET(CheckNotNull(params), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckDtype(params),   ACLNN_ERR_PARAM_INVALID);
    auto ret = PreProcess(params);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // Make contiguous
    auto kgexp_     = l0op::Contiguous(kgexp,     uniqueExecutor.get());
    auto value_     = l0op::Contiguous(value,     uniqueExecutor.get());
    auto kCumdecay_ = l0op::Contiguous(kCumdecay, uniqueExecutor.get());
    auto qgexp_     = l0op::Contiguous(qgexp,     uniqueExecutor.get());
    auto gexp_      = l0op::Contiguous(gexp,      uniqueExecutor.get());
    auto cuSeqlens_ = l0op::Contiguous(cuSeqlens, uniqueExecutor.get());

    // Call l0 operator
    auto outRet = l0op::ChunkGatedDeltaRuleRecurrence(
        initialState, kgexp_, value_, kCumdecay_, qgexp_, gexp_,
        cuSeqlens_, scaleValue, uniqueExecutor.get());
    if (outRet[0] == nullptr || outRet[1] == nullptr) {
        return ACLNN_ERR_INNER_NULLPTR;
    }

    // ViewCopy results to user-provided output tensors
    // outRet[0] = attn_inter_out, outRet[1] = v_new_out
    auto attnInterOut_ = l0op::Contiguous(attnInterOut, uniqueExecutor.get());
    auto vNewOut_      = l0op::Contiguous(vNewOut,      uniqueExecutor.get());

    auto copyAttn = l0op::ViewCopy(outRet[0], attnInterOut_, uniqueExecutor.get());
    if (copyAttn == nullptr) {
        return ACLNN_ERR_INNER_NULLPTR;
    }
    auto copyVNew = l0op::ViewCopy(outRet[1], vNewOut_, uniqueExecutor.get());
    if (copyVNew == nullptr) {
        return ACLNN_ERR_INNER_NULLPTR;
    }

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnChunkGatedDeltaRuleRecurrence(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream)
{
    L2_DFX_PHASE_2(aclnnChunkGatedDeltaRuleRecurrence);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
