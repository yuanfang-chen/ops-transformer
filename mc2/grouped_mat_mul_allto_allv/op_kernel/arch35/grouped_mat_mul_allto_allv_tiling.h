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
#include "mc2/3rd/grouped_matmul/op_kernel/arch35/grouped_matmul_tiling_data_apt.h"

#pragma once

/**
 * 常量定义
 */
constexpr uint32_t MAX_EXPERT_PER_EP = 32U;
constexpr uint32_t MAX_EP_RANK_SIZE = 256U;

/**
 * 类型复用声明
 * 复用 Mc2GroupedMatmulTilingData 中的类型定义
 */
using GMMQuantTilingData = Mc2GroupedMatmulTilingData::GMMQuantTilingData;
using GMMArray = Mc2GroupedMatmulTilingData::GMMArray;

/**
 * GMM Tiling 数组封装
 */
struct GmmTilingArray {
    uint32_t count;                                  // 实际使用的 tiling 数量
    GMMQuantTilingData array[MAX_EXPERT_PER_EP];     // GMM Tiling 数组
};

/**
 * QuantGmmA2avTilingInfo 核心配置信息
 * 合并了任务维度、流水线切分、Workspace 大小以及通信计数
 */
struct QuantGmmA2avTilingInfo {
    // --- Task Info (任务维度与专家信息) ---
    uint64_t taskM;              // 总 M 维度
    uint64_t taskK;              // K 维度
    uint64_t taskN;              // N 维度
    uint32_t taskLocalExpertNum; // 本 EP 专家数
    uint32_t taskEpWorldSize;    // EP 通信域大小

    // --- Loop Info (通算融合流水线切分信息) ---
    uint32_t loopMainExpertNum; // 主块：每次 loop 处理几个专家
    uint32_t loopTailExpertNum; // 尾块：最后一次 loop 处理几个专家
    uint32_t loopTotalCount;    // 总 loop 次数

    // --- Workspace Info (Workspace 大小) ---
    uint64_t wsGmmSize; // GMM workspace 大小

    // --- Comm Info (通信计数数组) ---
    // 每专家发送到各 rank 的 token 数
    uint16_t commSendCnt[MAX_EXPERT_PER_EP * MAX_EP_RANK_SIZE];
    // 从各 rank 接收每专家的 token 数
    uint16_t commRecvCnt[MAX_EXPERT_PER_EP * MAX_EP_RANK_SIZE];
};

/**
 * HCCL AlltoAllV Tiling 封装
 */
struct HcclA2avTilingInfo {
    Mc2InitTiling initTiling; // HCCL 初始化配置
    Mc2CcTiling a2avCcTiling; // AlltoAllV CC 配置
};

/**
 * 完整 TilingData 结构
 *
 * 设计要点：
 *   1. 共享专家放在普通专家之前
 *   2. QuantGmmA2avTilingInfo 包含扁平化的核心配置（Task/Loop/Workspace/Comm）
 */
struct QuantGmmA2avTilingData {
    // ============ HCCL AlltoAllV Tiling ============
    HcclA2avTilingInfo hcclA2avTiling;

    // ============ 核心配置信息 (Task/Loop/Workspace/Comm) ============
    QuantGmmA2avTilingInfo tilingInfo;

    // ============ 共享专家 GMM Tiling（放在前面）============
    GMMQuantTilingData sharedGmmTiling; // 共享专家 GMM Tiling 数据

    // ============ 普通专家 GMM Tiling 数组 ============
    GmmTilingArray gmmTiling; // 普通专家 GMM Tiling 数组
};