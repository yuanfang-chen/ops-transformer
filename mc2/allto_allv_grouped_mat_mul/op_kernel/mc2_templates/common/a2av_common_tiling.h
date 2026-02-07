/* *
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file a2av_common_tiling.h
 * \brief
 */

#ifndef A2AV_COMMON_H
#define A2AV_COMMON_H

#if __has_include("../../../3rd/grouped_matmul/op_kernel/arch35/grouped_matmul_tiling_data_apt.h")
#include "../../../3rd/grouped_matmul/op_kernel/arch35/grouped_matmul_tiling_data_apt.h"
#else
#include "../../../../3rd/grouped_matmul/op_kernel/arch35/grouped_matmul_tiling_data_apt.h"
#endif

namespace MC2KernelTemplate {
static constexpr uint32_t MAX_EP_RANK_SIZE = 8U;
static constexpr uint32_t MAX_EXPERT_PER_EP = 32U;
static constexpr uint32_t MAX_EXPERT_SIZE = 256U;

// 类型复用声明
using GMMQuantTilingData = Mc2GroupedMatmulTilingData::GMMQuantTilingData;
using GMMArray = Mc2GroupedMatmulTilingData::GMMArray;

/**
 * GMM Tiling 数组封装
 * 供 GMM All2AllV 和 All2AllV GMM 两个算子共用
 */
 //TODO 删除
struct GmmTilingArray {
    uint32_t count;                              // 实际使用的 tiling 数量
    GMMQuantTilingData array[MAX_EXPERT_PER_EP]; // GMM Tiling 数组
};

/**
 * 每次 GMM 迭代中随专家切分而变化的 Tiling 参数
 *
 * 在 GMM A2Av / A2Av GMM 融合算子中，调度器按专家颗粒度切分，
 * 每次迭代可能处理不同数量的专家、不同的 token 总数(M)，
 * 导致以下 5 个字段随每次 GMM 迭代变化，其余字段恒定。
 */
#pragma pack(push, 8)
struct GmmExpertDiffTiling {
    // --- GMMQuantParams 中的变化字段 ---
    uint32_t groupNum;        // 本次迭代处理的专家(组)数量

    // --- TCubeTiling 中的 M 相关变化字段 ---
    uint32_t M;               // 本次迭代的 token 总数
    uint32_t singleCoreM;     // 单核分配的 M
    uint32_t baseM;           // M 方向基础分块大小
    uint32_t dbL0C;           // L0C 双缓冲开关 (1 或 2)
};
#pragma pack(pop)

/**
 * GMM 专家差异 Tiling 数组
 * 记录每次 GMM 迭代（按专家颗粒度切分）的变化 tiling
 */
#pragma pack(push, 8)
struct GmmExpertDiffTilingArray {
    uint32_t count;                                 // 实际使用的条目数
    GmmExpertDiffTiling array[MAX_EXPERT_PER_EP];   // 每次迭代的差异 tiling
};
#pragma pack(pop)

struct HcclA2avTilingInfo {
    Mc2InitTiling hcclInitTiling;
    Mc2CcTiling a2avCcTiling;
};

struct TaskTilingInfo {
    // Tensor维度参数（对应aclnn接口的输入输出shape）
    uint64_t BSK;           // 参考各个算子aclnn接口
    uint64_t BS;            // mmXOptional的第一维: (BS, H)，batch*seq
    uint64_t H1;            // gmmX和gmmWeight的隐藏层维度H
    uint64_t H2;            // mmWeightOptional的隐藏层维度H
    uint64_t A;             // 参考各个算子aclnn接口
    uint64_t N1;            // gmmWeight/gmmY的输出维度N1
    uint64_t N2;            // mmWeightOptional/mmYOptional的输出维度N2
    
    // Expert并行参数
    uint64_t epWorldSize;   // expert parallel world size (EP并行域大小)
    uint64_t e;             // 单卡上的专家数量
    
    // 循环调度参数
    uint32_t mainLoopExpertNum;  // 主循环每次处理的expert数量
    uint32_t tailLoopExpertNum;  // 尾循环处理的expert数量
    uint32_t totalLoopCount;     // 总循环次数
    
    // 通信参数（对应sendCounts和recvCounts）
    int64_t sendCnt[MAX_EXPERT_SIZE];  // 每个expert的发送计数
    int64_t recvCnt[MAX_EXPERT_SIZE];  // 每个expert的接收计数
};

}
#endif // A2AV_COMMON_H
