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
 * \file quant_grouped_mat_mul_allto_allv_gmm_tiling.h
 * \brief
 */

#ifndef QUANT_GROUPED_MAT_MUL_ALLTO_ALLV_TILING_H
#define QUANT_GROUPED_MAT_MUL_ALLTO_ALLV_TILING_H

#pragma once
#include "securec.h"
#include "mc2_matmul_tiling_cfg.h"
#include "../grouped_mat_mul_allto_allv_tiling_base.h"
#include "tiling/new_mc2_tiling_utils.h"
#include "grouped_matmul_finalize_routing/op_kernel/arch35/grouped_matmul_finalize_routing_tiling_data.h"
#include "grouped_matmul/op_kernel/arch35/quant_adaptive_sliding_window_templates/gqmm_tiling_key.h"
#include "grouped_matmul/op_kernel/arch35/grouped_matmul_tiling_data_apt.h"
#include "register/tilingdata_base.h"


using namespace optiling;
namespace MC2Tiling {

/**
 * HCCL AlltoAllV Tiling 封装
 */
struct HcclA2avTilingInfo {
    Mc2InitTiling initTiling;          // HCCL 初始化配置
    Mc2CcTiling a2avCcTiling;          // AlltoAllV CC 配置
};

// gmm/grouped_matmul/op_host/op_tiling/grouped_matmul_tiling.h

struct QuantGmmA2avTilingInfo {
    // --- Task Info (任务维度与专家信息) ---
    uint64_t taskM;                                 // 总 M 维度
    uint64_t taskK;                                 // K 维度
    uint64_t taskN;                                 // N 维度
    uint32_t taskLocalExpertNum;                    // 本 EP 专家数
    uint32_t taskEpWorldSize;                       // EP 通信域大小
    
    // --- Loop Info (通算融合流水线切分信息) ---
    uint32_t loopMainExpertNum;                     // 主块：每次 loop 处理几个专家
    uint32_t loopTailExpertNum;                     // 尾块：最后一次 loop 处理几个专家
    uint32_t loopTotalCount;                        // 总 loop 次数

    // --- Workspace Info (Workspace 大小) ---
    uint64_t wsGmmSize;                             // GMM workspace 大小

    // --- Comm Info (通信计数数组) ---
    // 每专家发送到各 rank 的 token 数
    uint16_t commSendCnt[MAX_EXPERT_NUM];
    // 从各 rank 接收每专家的 token 数
    uint16_t commRecvCnt[MAX_EXPERT_NUM];
};

struct QuantGMMTilingDataArray {
    uint32_t counts;
    GroupedMatmulTilingData::GMMQuantTilingData gmmQuantTilingDataList[MAX_EXPERT_NUM_PER_RANK];
}

struct QuantGroupedMatMulAlltoAllvTilingData {
    HcclA2avTilingInfo hcclA2avTiling;
    QuantGmmA2avTilingInfo commonTilingInfo;
    GMMQuantTilingData sharedGmmTiling;
    QuantGMMTilingDataArray gmmQTilingDataInfo;
};

struct TilingInferredInfo {
    uint64_t gmmResultLen = 0UL; // 存储计算GMM的地址大小
    uint64_t mmResultLen = 0UL; // 存储计算MM的地址大小
    uint64_t commLen = 0UL; // 存储通信结果的临时空间，recvCounts
    uint64_t permuteLen = 0UL; // 重排空间大小, 应该与result一致
    uint32_t biasLen = 0UL; // 暂不支持bias
};

class QuantGroupedMatmulAllToAllvTiling : public GmmAlltoAllvTilingBase {
public:
    explicit QuantGroupedMatmulAllToAllvTiling(gert::TilingContext *context);
    ~QuantGroupedMatmulAllToAllvTiling() override = default;
protected:
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus PostTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus CheckOpInputInfo();
    ge::graphStatus InitTilingContextParameters(); // set默认值，当前不支持功能参数
    ge::graphStatus DoQuantGMMTiling(); // 按专家为粒度执行
    ge::graphStatus SetHcclTiling();
    
    void SetTilingCommonInfo();
    void PrintQuantGroupedMatMulAlltoAllvTilingData(QuantGroupedMatMulAlltoAllvTilingData &outTilingData);
    
private:
    QuantGroupedMatMulAlltoAllvTilingData localTilingData_;
    void PrintQuantGMMAlltoAllTilingInfo(const std::string &opName, QuantGmmA2avTilingInfo &tilingInfo);
    void PrintQuantGMMTilingData(const std::string &opName, GroupedMatmulTilingData::GMMQuantTilingData &tiling);
    void PrintSharedMMTilingData(const std::string &opName, GroupedMatmulTilingData::GMMQuantTilingData &tiling);
    const char *opName_{nullptr};
    uint32_t libApiWorkSpaceSize_{0};
    uint32_t epNum_{2};
    TilingInferredInfo inferredInfo;
};

} // namespace MC2Tiling
#endif
