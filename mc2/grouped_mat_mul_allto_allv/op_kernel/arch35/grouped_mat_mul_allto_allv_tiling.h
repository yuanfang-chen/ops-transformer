/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file grouped_mat_mul_allto_allv_tiling.h
 * \brief Quant Grouped MatMul AlltoAllV TilingData 定义
 */
#ifndef QUANT_GROUPED_MAT_MUL_ALLTO_ALLV_TILING_H__
#define QUANT_GROUPED_MAT_MUL_ALLTO_ALLV_TILING_H__

#include "kernel_operator.h"
#include "mc2/allto_allv_grouped_mat_mul/op_kernel/mc2_templates/common/a2av_common_tiling.h"

#pragma once

// 使用公共命名空间中的类型
using MC2KernelTemplate::GmmTilingArray;
using MC2KernelTemplate::GMMQuantTilingData;
using MC2KernelTemplate::GMMArray;

/**
 * GMM A2AV Workspace 信息
 * 专门管理 workspace 相关配置
 */
struct GmmA2avWorkspaceInfo {
    uint64_t wsGmmSize;           // GMM workspace 大小
};

struct QuantGmmA2avTilingData {
    // ============ HCCL AlltoAllV Tiling ============
    MC2KernelTemplate::HcclA2avTilingInfo hcclA2avTiling;

    // ============ Workspace 配置信息 ============
    GmmA2avWorkspaceInfo workspaceInfo;

    // ============ 核心配置信息 (Task/Loop/Comm) ============
    TaskTilingInfo taskTilingInfo;

    // ============ 共享专家 GMM Tiling（放在前面）============
    GMMQuantTilingData sharedGmmTiling; // 共享专家 GMM Tiling 数据

    // ============ 普通专家 GMM Tiling 数组 ============
    GmmTilingArray gmmTiling; // 普通专家 GMM Tiling 数组
};