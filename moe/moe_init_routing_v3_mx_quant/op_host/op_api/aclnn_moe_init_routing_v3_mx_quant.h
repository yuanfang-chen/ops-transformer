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
 * \file aclnn_moe_init_routing_v3_mx_quant.h
 * \brief
 */
#ifndef OP_API_INC_MOE_INIT_ROUTING_V3_MX_QUANT_H_
#define OP_API_INC_MOE_INIT_ROUTING_V3_MX_QUANT_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnMoeInitRoutingV3MxQuant first-stage interface: computes workspace size.
 * @domain aclnn_ops_infer
 */
ACLNN_API aclnnStatus aclnnMoeInitRoutingV3MxQuantGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *expertIdx,
    const aclTensor *scaleOptional,
    const aclTensor *offsetOptional,
    int64_t activeNum,
    int64_t expertCapacity,
    int64_t expertNum,
    int64_t dropPadMode,
    int64_t expertTokensCountOrCumsumFlag,
    bool expertTokensBeforeCapacityFlag,
    int64_t axis,
    char *roundModeOptional,
    int64_t dstType,
    int64_t blocksize,
    int64_t scaleAlg,
    const aclTensor *yOut,
    const aclTensor *mxscaleOut,
    const aclTensor *expandedRowIdxOut,
    const aclTensor *expertTokensCountOrCumsumOutOptional,
    const aclTensor *expandedScaleOutOptional,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* @brief aclnnMoeInitRoutingV3MxQuant second-stage interface: executes computation. */
ACLNN_API aclnnStatus aclnnMoeInitRoutingV3MxQuant(void *workspace, uint64_t workspaceSize,
                                                    aclOpExecutor *executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
