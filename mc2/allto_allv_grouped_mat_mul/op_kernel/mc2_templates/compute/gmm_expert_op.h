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
 * \file quant_grouped_matmul.h
 * \brief GmmExpertOp 类：GMM 专家计算操作封装
 */

#ifndef MC2_QUANT_GROUPED_MATMUL_H
#define MC2_QUANT_GROUPED_MATMUL_H

#include "kernel_operator.h"
#include "../common/a2av_common_tiling.h"
#if __has_include(                                                                                                     \
    "../../../3rd/grouped_matmul/op_kernel/arch35/quant_adaptive_sliding_window_templates/gqmm_cube_on_the_fly.h")
#include "../../../3rd/grouped_matmul/op_kernel/arch35/quant_adaptive_sliding_window_templates/gqmm_cube_on_the_fly.h"
#else
#include "../../../../3rd/grouped_matmul/op_kernel/arch35/grouped_matmul_tiling_data_apt.h"
#endif

namespace MC2KernelTemplate {

/**
 * GMM 专家计算操作封装
 * 内部基于 GmmASWKernel 实现，按专家拆解计算逻辑
 *
 * @tparam xType              输入 X 数据类型
 * @tparam wType              权重数据类型
 * @tparam biasType           偏置数据类型
 * @tparam yType              输出 Y 数据类型
 * @tparam scaleType          量化 scale 数据类型
 * @tparam wFormat            权重格式（默认 ND）
 * @tparam aTrans             输入 A 是否转置
 * @tparam bTrans             权重 B 是否转置
 * @tparam USE_SEND_COUNTS    true=使用 sendCnt（GMM A2AV），false=使用 recvCnt（A2AV GMM）
 * @tparam IS_SHARED_EXPERT   true=共享专家（单 tiling），false=路由专家（多 tiling）
 */
template <typename xType,
          typename wType,
          typename biasType,
          typename yType,
          typename scaleType,
          AscendC::CubeFormat wFormat = AscendC::CubeFormat::ND,
          bool aTrans = false,
          bool bTrans = false,
          bool USE_SEND_COUNTS = true,
          bool IS_SHARED_EXPERT = false>
class GmmExpertOp {
public:
    // 内部使用的 GMM Kernel 类型
    using GmmKernelType = AscendC::GmmASWKernel<xType, wType, biasType,
                                                scaleType, yType,
                                                wFormat, aTrans, bTrans>;
    using GMMQuantParams = Mc2GroupedMatmulTilingData::GMMQuantParams;

    __aicore__ inline GmmExpertOp() = default;

    /**
     * 初始化（路由专家版本）
     * 当 IS_SHARED_EXPERT == false 时使用
     *
     * @param taskTilingInfo  任务 Tiling 信息（包含 sendCnt/recvCnt）
     * @param gmmTilingArray  GMM Tiling 数组
     * @param tPipe           Pipe 指针
     */
    template <bool Shared = IS_SHARED_EXPERT>
    __aicore__ inline typename AscendC::EnableIf<!Shared, void>::type
    Init(const TaskTilingInfo* taskTilingInfo,
         const GmmTilingArray* gmmTilingArray,
         AscendC::TPipe* tPipe)
    {
        taskTilingInfo_ = taskTilingInfo;
        gmmTilingArray_ = gmmTilingArray;
        sharedGmmTiling_ = nullptr;
        tPipe_ = tPipe;
        expertNum_ = static_cast<uint32_t>(taskTilingInfo_->e);
    }

    /**
     * 初始化（共享专家版本）
     * 当 IS_SHARED_EXPERT == true 时使用
     *
     * @param taskTilingInfo  任务 Tiling 信息（包含 sendCnt/recvCnt）
     * @param sharedGmmTiling 单个共享专家 GMM Tiling 数据
     * @param tPipe           Pipe 指针
     */
    template <bool Shared = IS_SHARED_EXPERT>
    __aicore__ inline typename AscendC::EnableIf<Shared, void>::type
    Init(const TaskTilingInfo* taskTilingInfo,
         const GMMQuantTilingData* sharedGmmTiling,
         AscendC::TPipe* tPipe)
    {
        taskTilingInfo_ = taskTilingInfo;
        gmmTilingArray_ = nullptr;
        sharedGmmTiling_ = sharedGmmTiling;
        tPipe_ = tPipe;
        expertNum_ = 1;  // 共享专家只有一个
    }

    /**
     * 初始化所有输入输出的基地址
     * groupList 将在内部根据 USE_SEND_COUNTS 自动计算
     *
     * @param x         输入 X 基地址
     * @param weight    权重基地址
     * @param bias      偏置基地址（可为 nullptr）
     * @param scaleA    量化 scale A 基地址
     * @param scaleB    量化 scale B 基地址
     * @param y         输出 Y 基地址
     * @param workspace 工作空间基地址
     */
    __aicore__ inline void InitAddr(
        GM_ADDR x,
        GM_ADDR weight,
        GM_ADDR bias,
        GM_ADDR scaleA,
        GM_ADDR scaleB,
        GM_ADDR y,
        GM_ADDR workspace)
    {
        xBase_ = x;
        weightBase_ = weight;
        biasBase_ = bias;
        scaleABase_ = scaleA;
        scaleBBase_ = scaleB;
        yBase_ = y;
        workspaceBase_ = workspace;
        
        // groupList 缓存在 workspace 中，预留空间给 groupList
        // groupList 是累积和数组，用于确定每个 group 的边界
        groupListCache_ = workspace;
    }

    /**
     * 处理专家计算
     * 内部调用 GmmASWKernel 执行实际计算
     *
     * @param startExpertIdx  起始专家索引
     * @param expertNum       专家数量
     */
    __aicore__ inline void ProcessExpert(uint32_t startExpertIdx, uint32_t expertNum);

private:
    GmmKernelType gmmKernel_;

    const TaskTilingInfo* taskTilingInfo_ = nullptr;
    const GmmTilingArray* gmmTilingArray_ = nullptr;      // 路由专家使用
    const GMMQuantTilingData* sharedGmmTiling_ = nullptr; // 共享专家使用
    AscendC::TPipe* tPipe_ = nullptr;
    uint32_t expertNum_ = 0;

    // 输入输出基地址
    GM_ADDR xBase_ = nullptr;
    GM_ADDR weightBase_ = nullptr;
    GM_ADDR biasBase_ = nullptr;
    GM_ADDR scaleABase_ = nullptr;
    GM_ADDR scaleBBase_ = nullptr;
    GM_ADDR yBase_ = nullptr;
    GM_ADDR workspaceBase_ = nullptr;
    GM_ADDR groupListCache_ = nullptr;

    /**
     * 内部：获取 groupList 来源数组
     * 根据 USE_SEND_COUNTS 模板参数选择 sendCnt 或 recvCnt
     */
    __aicore__ inline const int64_t* GetGroupCounts() const
    {
        if constexpr (USE_SEND_COUNTS) {
            return taskTilingInfo_->sendCnt;
        } else {
            return taskTilingInfo_->recvCnt;
        }
    }

    /**
     * 内部：获取专家索引对应的 tiling 数据
     * 根据 IS_SHARED_EXPERT 模板参数选择返回 sharedGmmTiling_ 或 gmmTilingArray_->array[expertIdx]
     */
    __aicore__ inline const GMMQuantTilingData* GetTilingData(uint32_t expertIdx) const
    {
        if constexpr (IS_SHARED_EXPERT) {
            // 共享专家始终返回同一个 tiling
            return sharedGmmTiling_;
        } else {
            // 路由专家根据索引返回对应的 tiling
            return &gmmTilingArray_->array[expertIdx];
        }
    }

    /**
     * 内部：计算 X 在 M 轴上的偏移量
     * 根据 sendCnt/recvCnt 累加计算偏移
     *
     * @param expertIdx 专家索引
     * @return M 轴偏移量（token 数）
     */
    __aicore__ inline int64_t CalcXOffset(uint32_t expertIdx) const
    {
        const int64_t* counts = GetGroupCounts();
        int64_t offset = 0;
        for (uint32_t i = 0; i < expertIdx; ++i) {
            offset += counts[i];
        }
        return offset;
    }

    /**
     * 内部：计算 Y 在 M 轴上的偏移量
     *
     * @param expertIdx 专家索引
     * @return M 轴偏移量（token 数）
     */
    __aicore__ inline int64_t CalcYOffset(uint32_t expertIdx) const
    {
        // Y 的偏移计算与 X 相同
        return CalcXOffset(expertIdx);
    }

    /**
     * 内部：计算权重在 expert 维度上的偏移量
     *
     * @param expertIdx 专家索引
     * @return 权重偏移量
     */
    __aicore__ inline int64_t CalcWeightOffset(uint32_t expertIdx) const
    {
        // 权重是按 expert 索引直接偏移，K * N per expert
        int64_t K = static_cast<int64_t>(taskTilingInfo_->H1);
        int64_t N = static_cast<int64_t>(taskTilingInfo_->N1);
        return expertIdx * K * N;
    }

    /**
     * 内部：更新地址并准备 groupList
     *
     * @param startExpertIdx  起始专家索引
     * @param expertNum       本次处理的专家数量
     * @param groupListOut    输出 groupList 的 GM 地址
     */
    __aicore__ inline void PrepareGroupList(uint32_t startExpertIdx, uint32_t expertNum);
};

// ========== 方法实现 ==========

template <typename xType, typename wType, typename biasType, typename yType, typename scaleType,
          AscendC::CubeFormat wFormat, bool aTrans, bool bTrans,
          bool USE_SEND_COUNTS, bool IS_SHARED_EXPERT>
__aicore__ inline void GmmExpertOp<xType, wType, biasType, yType, scaleType,
                                    wFormat, aTrans, bTrans,
                                    USE_SEND_COUNTS, IS_SHARED_EXPERT>::PrepareGroupList(
    uint32_t startExpertIdx, uint32_t expertNum)
{
    // 将 counts 转换为累积和形式的 groupList 写入 groupListCache_
    const int64_t* counts = GetGroupCounts();
    __gm__ int64_t* groupList = reinterpret_cast<__gm__ int64_t*>(groupListCache_);
    
    int64_t cumSum = 0;
    for (uint32_t i = 0; i < expertNum; ++i) {
        uint32_t expertIdx = startExpertIdx + i;
        cumSum += counts[expertIdx];
        groupList[i] = cumSum;
    }
}

template <typename xType, typename wType, typename biasType, typename yType, typename scaleType,
          AscendC::CubeFormat wFormat, bool aTrans, bool bTrans,
          bool USE_SEND_COUNTS, bool IS_SHARED_EXPERT>
__aicore__ inline void GmmExpertOp<xType, wType, biasType, yType, scaleType,
                                    wFormat, aTrans, bTrans,
                                    USE_SEND_COUNTS, IS_SHARED_EXPERT>::ProcessExpert(
    uint32_t startExpertIdx, uint32_t expertNum)
{
    // 1. 准备 groupList（累积和形式，包含 expertNum 个专家的 counts）
    //    GmmASWKernel 一次调用会处理 groupNum 个 group
    PrepareGroupList(startExpertIdx, expertNum);
    
    // 2. 计算 X 和 Y 的基地址偏移（基于 startExpertIdx）
    int64_t xMOffset = CalcXOffset(startExpertIdx);
    int64_t yMOffset = CalcYOffset(startExpertIdx);
    int64_t weightOffset = CalcWeightOffset(startExpertIdx);
    
    // 3. 计算偏移后的地址
    int64_t K = static_cast<int64_t>(taskTilingInfo_->H1);
    int64_t N = static_cast<int64_t>(taskTilingInfo_->N1);
    
    GM_ADDR xCur = xBase_ + xMOffset * K * sizeof(xType);
    GM_ADDR yCur = yBase_ + yMOffset * N * sizeof(yType);
    GM_ADDR weightCur = weightBase_ + weightOffset * sizeof(wType);
    
    // bias 和 scale 按 expert 索引偏移
    GM_ADDR biasCur = biasBase_;
    if (biasBase_ != nullptr) {
        biasCur = biasBase_ + startExpertIdx * N * sizeof(biasType);
    }
    
    GM_ADDR scaleACur = scaleABase_;
    if (scaleABase_ != nullptr) {
        scaleACur = scaleABase_ + startExpertIdx * sizeof(scaleType);
    }
    
    GM_ADDR scaleBCur = scaleBBase_;
    if (scaleBBase_ != nullptr) {
        scaleBCur = scaleBBase_ + startExpertIdx * sizeof(scaleType);
    }
    
    // 4. 获取本次循环对应的 tiling 数据
    //    路由专家：从 gmmTilingArray_->array[startExpertIdx] 获取
    //    共享专家：使用 sharedGmmTiling_
    const GMMQuantTilingData* tilingData = GetTilingData(startExpertIdx);
    
    // 获取 tiling 参数
    const GMMQuantParams* gmmQuantParams = &tilingData->gmmQuantParams;
    const TCubeTiling* mmTilingData = &tilingData->mmTilingData;
    
    // gmmArray 包含 mList, kList, nList
    TILING_TYPE* gmmArrayAddr = const_cast<TILING_TYPE*>(
        reinterpret_cast<const TILING_TYPE*>(&tilingData->gmmArray.mList[0]));
    
    // 5. 一次调用 GmmASWKernel 处理所有 expertNum 个专家
    //    GmmASWKernel.Process() 内部会遍历 groupNum 个 group
    gmmKernel_.Init(
        xCur,                    // x
        weightCur,               // weight
        biasCur,                 // bias
        scaleBCur,               // scale (weight scale)
        groupListCache_,         // groupList (累积和形式)
        scaleACur,               // perTokenScale (activation scale)
        yCur,                    // y
        workspaceBase_,          // workspace
        gmmQuantParams,          // gmmQuantParams (包含 groupNum)
        mmTilingData,            // mmTilingData
        gmmArrayAddr,            // gmmArrayAddr (mList, kList, nList)
        tPipe_                   // TPipe
    );
    
    gmmKernel_.Process();
}

} // namespace MC2KernelTemplate

#endif // MC2_QUANT_GROUPED_MATMUL_H
