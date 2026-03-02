/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. 
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel_operator.h"
#include "arch32/apply_adamw_v3_dag.h"

using namespace AscendC;
using namespace ApplyAdamwV3Ns;

// TilingKey 常量定义（符合设计文档 3.1 节）
constexpr uint32_t FP16_TILING_KEY = 1;
constexpr uint32_t BF16_TILING_KEY = 2;
constexpr uint32_t FP32_TILING_KEY = 3;
constexpr uint32_t AMSGRAD_FP16_TILING_KEY = 11;
constexpr uint32_t AMSGRAD_BF16_TILING_KEY = 12;
constexpr uint32_t AMSGRAD_FP32_TILING_KEY = 13;

template <typename T>
__aicore__ inline void ProcessAdamW(GM_ADDR var, GM_ADDR m, GM_ADDR v, 
                                      GM_ADDR grad, GM_ADDR var_out, GM_ADDR m_out, GM_ADDR v_out,
                                      GM_ADDR tiling) {
    GET_TILING_DATA_WITH_STRUCT(ApplyAdamwV3TilingData, tilingData, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    
    if (GetBlockIdx() >= tilingData.usedCoreNum) {
        return;
    }
    
    // 计算当前核的起始偏移和处理的元素数
    uint32_t totalLength = tilingData.totalLength;
    uint32_t usedCoreNum = tilingData.usedCoreNum;
    uint32_t tileNumPerCore = tilingData.tileNumPerCore;
    uint32_t tileLength = tilingData.tileLength;
    
    uint32_t blockIdx = GetBlockIdx();
    uint32_t avgElementsPerCore = totalLength / usedCoreNum;
    uint32_t coreOffset = blockIdx * avgElementsPerCore;
    
    // 处理最后一个核的边界情况
    uint32_t elementsThisCore = avgElementsPerCore;
    if (blockIdx == usedCoreNum - 1) {
        elementsThisCore = totalLength - coreOffset;
    }
    
    uint32_t tileNum = (elementsThisCore + tileLength - 1) / tileLength;
    
    ApplyAdamwV3Kernel<T> op;
    op.Init(var, m, v, grad, var_out, m_out, v_out,
            totalLength, tileLength, tileNum,
            tilingData.beta1Power, tilingData.beta2Power, tilingData.lr,
            tilingData.weightDecay, tilingData.beta1, tilingData.beta2,
            tilingData.epsilon, tilingData.maximizeFactor,
            coreOffset, elementsThisCore);
    op.Process();
}

extern "C" __global__ __aicore__ void apply_adamw_v3(GM_ADDR var, GM_ADDR m, GM_ADDR v,
                                                        GM_ADDR beta1_power, GM_ADDR beta2_power,
                                                        GM_ADDR lr, GM_ADDR weight_decay,
                                                        GM_ADDR beta1, GM_ADDR beta2,
                                                        GM_ADDR epsilon, GM_ADDR grad,
                                                        GM_ADDR max_grad_norm,
                                                        GM_ADDR var_out, GM_ADDR m_out, GM_ADDR v_out,
                                                        GM_ADDR workspace, GM_ADDR tiling) {
    if (g_coreType == AscendC::AIC) {
        return;
    }
    
    // 根据 TilingKey 选择对应的模板类型（符合设计文档 3.1 节）
    GET_TILING_DATA_WITH_STRUCT(ApplyAdamwV3TilingData, tilingData, tiling);
    
    switch (tilingData.tilingKey) {
        case FP16_TILING_KEY:
            ProcessAdamW<half>(var, m, v, grad, var_out, m_out, v_out, tiling);
            break;
        case BF16_TILING_KEY:
            ProcessAdamW<bfloat16_t>(var, m, v, grad, var_out, m_out, v_out, tiling);
            break;
        case FP32_TILING_KEY:
            ProcessAdamW<float>(var, m, v, grad, var_out, m_out, v_out, tiling);
            break;
        case AMSGRAD_FP16_TILING_KEY:
            // AMSGrad 模式（Phase 2 实现）
            // ProcessAdamWAMSGrad<half>(var, m, v, max_grad_norm, grad, var_out, m_out, v_out, tiling);
            break;
        case AMSGRAD_BF16_TILING_KEY:
            // AMSGrad 模式（Phase 2 实现）
            break;
        case AMSGRAD_FP32_TILING_KEY:
            // AMSGrad 模式（Phase 2 实现）
            break;
        default:
            return;
    }
}
