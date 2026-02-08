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
 * \file gmm_compute_op.h
 * \brief GmmComputeOp 类：GMM 计算操作封装，替代 GmmExpertOp 和 QuantGroupedMatmul
 */

#ifndef MC2_GMM_COMPUTE_OP_H
#define MC2_GMM_COMPUTE_OP_H

#include "../common/a2av_common_tiling.h"
#include "kernel_operator.h"
// GmmComputeOp 依赖 op_kernel/3rd/ 下的 gqmm_cube_on_the_fly.h
// 在 pkg 联编时，mc2 版本可能已被 quant_grouped_matmul.h 提前 include，
// 两者接口一致（仅内部命名空间不同），通过双 guard 避免 GmmASWKernel 重复定义
#if !defined(GQMM_CUBE_ON_THE_FLY_H) && !defined(MC2_GQMM_CUBE_ON_THE_FLY_H)
#include "../../3rd/gqmm_cube_on_the_fly.h"
#endif

using namespace AscendC;

namespace MC2KernelTemplate {

// ascend950 L0C 大小：256KB, 依赖的GmmASWKernel
// tilingdata需要动态刷新dbL0C,因此需要这个LOC_SIZE变量以动态计算是否使用double buffer
static constexpr uint64_t GMM_COMPUTE_L0C_SIZE = 256ULL * 1024ULL;

/**
 * GMM 计算操作
 * 在 MC2 融合算子中负责 Grouped MatMul 计算，支持路由专家和共享专家
 *
 * 核心流程：
 * 1. Init 接收 gmmBaseTiling（K/N 相关字段由 host 预计算）及所有 tensor 基地址
 * 2. ProcessExperts 逐专家迭代：
 *    a. 从 sendCnt/recvCnt 获取本专家 token 数量 M
 *    b. 构建 group_list（单条累积和，值为 M）
 *    c. 动态刷新 TilingData：groupNum=1, M, singleCoreM, baseM, dbL0C
 *    d. 计算 x/y 地址偏移（累积 token 偏移 * K/N）
 *    e. 计算 weight 地址偏移（expertIdx * K * N）
 *    f. 包装偏移后地址为 GetTensorAddr 的指针表格式
 *    g. 调用 GmmASWKernel.Init() + Process()
 *
 * @tparam xType, wType, biasType, scaleType, yType  数据类型
 * @tparam wFormat       权重格式（如 CubeFormat::ND）
 * @tparam aTrans        输入 A 是否转置
 * @tparam bTrans        输入 B（权重）是否转置
 * @tparam USE_SEND_COUNTS  true=使用 sendCnt（GMM A2AV），false=使用 recvCnt（A2AV GMM）
 */
template <class xType, class wType, class biasType, class scaleType, class yType,
          CubeFormat wFormat, bool aTrans, bool bTrans, bool USE_SEND_COUNTS>
class GmmComputeOp {
public:
    using GMMQuantParams = Mc2GroupedMatmulTilingData::GMMQuantParams;

    __aicore__ inline GmmComputeOp() = default;

    /**
     * 初始化
     *
     * @param x              输入 X 基地址（直接数据地址，所有专家 token 连续排列）
     * @param weight         权重基地址（直接数据地址，(E, K, N) 或 (E, N, K) 布局）
     * @param bias           偏置基地址（可为 nullptr）
     * @param scaleA         X 的量化 scale 基地址（perTokenScale，直接透传给 GmmASWKernel）
     * @param scaleB         Weight 的量化 scale 基地址（直接透传给 GmmASWKernel）
     * @param y              输出 Y 基地址（直接数据地址，输出 token 连续排列）
     * @param workspace      工作空间基地址
     * @param taskTilingInfo 任务调度 Tiling（含 sendCnt/recvCnt、维度参数、循环参数）
     * @param gmmBaseTiling  GMM 基础 Tiling（K/N 相关字段已计算，M 相关字段待动态刷新）
     * @param tPipe          TPipe 指针
     */
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR bias,
                                GM_ADDR scaleA, GM_ADDR scaleB,
                                GM_ADDR y, GM_ADDR workspace,
                                const TaskTilingInfo *taskTilingInfo,
                                const GMMQuantTilingData *gmmBaseTiling,
                                TILING_TYPE *gmmArrayAddr,
                                TPipe *tPipe);

    /**
     * 处理指定范围的专家计算
     *
     * 内部逐专家迭代，每个专家独立调用一次 GmmASWKernel：
     * - 跳过 tokenCount=0 的专家
     * - 自动累积 x/y 的 token 偏移
     * - 调用间自动 Reset TPipe
     *
     * @param startExpertIdx 起始专家索引（对应 sendCnt/recvCnt 中的索引）
     * @param expertNum      本次处理的专家数量
     */
    __aicore__ inline void ProcessExperts(uint32_t startExpertIdx, uint32_t expertNum);

private:
    const TaskTilingInfo *taskTilingInfo_ = nullptr;
    const GMMQuantTilingData *gmmBaseTiling_ = nullptr;
    TILING_TYPE *gmmArrayAddr_ = nullptr;
    TPipe *tPipe_ = nullptr;

    // 可变 tiling 副本
    GMMQuantTilingData currentTiling_;

    // 基地址
    GM_ADDR xBase_ = nullptr;
    GM_ADDR weightBase_ = nullptr;
    GM_ADDR biasBase_ = nullptr;
    GM_ADDR scaleABase_ = nullptr;
    GM_ADDR scaleBBase_ = nullptr;
    GM_ADDR yBase_ = nullptr;
    GM_ADDR workspaceBase_ = nullptr;

    // 维度缓存
    uint64_t K_ = 0;
    uint64_t N_ = 0;
    uint64_t epWorldSize_ = 0;
    uint64_t expertNumPerRank_ = 0;

    // 偏移状态
    uint64_t tokenOffset_ = 0;

    // workspace 中的指针表区域（4 个 tensor * 16 bytes = 64 bytes）
    GM_ADDR ptrTableBase_ = nullptr;

    // workspace 中 groupList 区域（紧接指针表之后）
    GM_ADDR groupListBase_ = nullptr;

    /**
     * 获取 count 数组（根据 USE_SEND_COUNTS 选择 sendCnt 或 recvCnt）
     */
    __aicore__ inline const int32_t *GetGroupCounts() const
    {
        if constexpr (USE_SEND_COUNTS) {
            return taskTilingInfo_->sendCnt;
        } else {
            return taskTilingInfo_->recvCnt;
        }
    }

    /**
     * 计算指定专家的 token 总数
     * token 总数 = sum_{rank=0}^{epWorldSize-1} counts[expertIdx + rank * expertNumPerRank]
     */
    __aicore__ inline uint64_t GetExpertTokenCount(uint32_t expertIdx) const
    {
        const int32_t *counts = GetGroupCounts();
        uint64_t total = 0;
        for (uint64_t rank = 0; rank < epWorldSize_; ++rank) {
            total += static_cast<uint64_t>(counts[expertIdx + rank * expertNumPerRank_]);
        }
        return total;
    }

    /**
     * 根据 M 动态刷新 currentTiling_ 中的 M 相关字段
     */
    __aicore__ inline void RefreshTilingForM(uint64_t M)
    {
        currentTiling_.gmmQuantParams.groupNum = 1;
        currentTiling_.mmTilingData.M = static_cast<uint32_t>(M);

        // baseM: CeilAlign(min(M, 256), alignUnit)
        // 内联 Min / Align 逻辑，避免对 QuantUtils / Mc2QuantUtils 命名空间的编译期依赖
        constexpr uint64_t MAX_BASE_M = 256;
        constexpr uint64_t CUBE_ALIGN = 16;
        constexpr uint64_t INNER_AXIS_ALIGN = 128;
        uint64_t baseMCandidate = (M < MAX_BASE_M) ? M : MAX_BASE_M;
        if constexpr (!aTrans) {
            currentTiling_.mmTilingData.baseM = static_cast<uint32_t>(
                ((baseMCandidate + CUBE_ALIGN - 1) / CUBE_ALIGN) * CUBE_ALIGN);
        } else {
            currentTiling_.mmTilingData.baseM = static_cast<uint32_t>(
                ((baseMCandidate + INNER_AXIS_ALIGN - 1) / INNER_AXIS_ALIGN) * INNER_AXIS_ALIGN);
        }

        // singleCoreM = min(M, baseM)
        uint64_t baseMVal = currentTiling_.mmTilingData.baseM;
        currentTiling_.mmTilingData.singleCoreM = static_cast<uint32_t>(
            (M < baseMVal) ? M : baseMVal);

        // dbL0C: 如果 baseM * baseN * 4（float 占 4 字节）* 2（double buffer）能装入 L0C 则为 2，否则为 1
        uint64_t baseM = currentTiling_.mmTilingData.baseM;
        uint64_t baseN = currentTiling_.mmTilingData.baseN;
        currentTiling_.mmTilingData.dbL0C =
            (baseM * baseN * sizeof(float) * 2 <= GMM_COMPUTE_L0C_SIZE) ? 2 : 1;
    }

    /**
     * 在 workspace 指针表区域构建 GetTensorAddr 所需的双重间接指针
     * slotIdx: 0=x, 1=weight, 2=scaleB, 3=y
     * 返回指向指针表条目的地址
     */
    __aicore__ inline GM_ADDR BuildPtrTable(GM_ADDR dataAddr, uint32_t slotIdx)
    {
        // 每个 slot 占 16 bytes (2 * uint64_t)
        __gm__ uint64_t *slot = reinterpret_cast<__gm__ uint64_t *>(
            reinterpret_cast<__gm__ uint8_t *>(ptrTableBase_) + slotIdx * 16);
        slot[0] = sizeof(uint64_t);  // byteOffset
        slot[1] = reinterpret_cast<uint64_t>(dataAddr);  // 实际数据地址
        return reinterpret_cast<GM_ADDR>(slot);
    }

    /**
     * 将单个 groupList 值（累积和形式）写入 workspace 的 groupList 区域
     */
    __aicore__ inline void WriteSingleGroupList(uint64_t tokenCount)
    {
        __gm__ int64_t *groupList = reinterpret_cast<__gm__ int64_t *>(groupListBase_);
        groupList[0] = static_cast<int64_t>(tokenCount);
    }
};

// ========== 方法实现 ==========

template <class xType, class wType, class biasType, class scaleType, class yType,
          CubeFormat wFormat, bool aTrans, bool bTrans, bool USE_SEND_COUNTS>
__aicore__ inline void
GmmComputeOp<xType, wType, biasType, scaleType, yType, wFormat, aTrans, bTrans, USE_SEND_COUNTS>::Init(
    GM_ADDR x, GM_ADDR weight, GM_ADDR bias,
    GM_ADDR scaleA, GM_ADDR scaleB,
    GM_ADDR y, GM_ADDR workspace,
    const TaskTilingInfo *taskTilingInfo,
    const GMMQuantTilingData *gmmBaseTiling,
    TILING_TYPE *gmmArrayAddr,
    TPipe *tPipe)
{
    taskTilingInfo_ = taskTilingInfo;
    gmmBaseTiling_ = gmmBaseTiling;
    gmmArrayAddr_ = gmmArrayAddr;
    tPipe_ = tPipe;

    xBase_ = x;
    weightBase_ = weight;
    biasBase_ = bias;
    scaleABase_ = scaleA;
    scaleBBase_ = scaleB;
    yBase_ = y;
    workspaceBase_ = workspace;

    K_ = taskTilingInfo_->H1;
    N_ = taskTilingInfo_->N1;
    epWorldSize_ = taskTilingInfo_->epWorldSize;
    expertNumPerRank_ = taskTilingInfo_->e;

    tokenOffset_ = 0;

    // 拷贝 gmmBaseTiling 到可变副本
    currentTiling_ = *gmmBaseTiling_;

    // workspace 布局：[ptrTable: 64 bytes][groupList: 8 bytes][GmmASWKernel workspace ...]
    ptrTableBase_ = workspace;
    groupListBase_ = reinterpret_cast<GM_ADDR>(
        reinterpret_cast<__gm__ uint8_t *>(workspace) + 64);
}

template <class xType, class wType, class biasType, class scaleType, class yType,
          CubeFormat wFormat, bool aTrans, bool bTrans, bool USE_SEND_COUNTS>
__aicore__ inline void
GmmComputeOp<xType, wType, biasType, scaleType, yType, wFormat, aTrans, bTrans, USE_SEND_COUNTS>::ProcessExperts(
    uint32_t startExpertIdx, uint32_t expertNum)
{
    for (uint32_t i = 0; i < expertNum; ++i) {
        uint32_t expertIdx = startExpertIdx + i;
        uint64_t tokenCount = GetExpertTokenCount(expertIdx);

        // 跳过 token 数为 0 的专家
        if (tokenCount == 0) {
            continue;
        }

        // 1. 动态刷新 M 相关 tiling 字段
        RefreshTilingForM(tokenCount);

        // 2. 写入 groupList（单条，值为本专家 token 数）
        WriteSingleGroupList(tokenCount);

        // 3. 计算偏移后地址
        __gm__ uint8_t *xAddr = reinterpret_cast<__gm__ uint8_t *>(xBase_) +
                                tokenOffset_ * K_ * sizeof(xType);
        __gm__ uint8_t *yAddr = reinterpret_cast<__gm__ uint8_t *>(yBase_) +
                                tokenOffset_ * N_ * sizeof(yType);
        __gm__ uint8_t *wAddr = reinterpret_cast<__gm__ uint8_t *>(weightBase_) +
                                static_cast<uint64_t>(expertIdx) * K_ * N_ * sizeof(wType);

        // 4. 构建 GetTensorAddr 指针表
        GM_ADDR xPtr = BuildPtrTable(reinterpret_cast<GM_ADDR>(xAddr), 0);
        GM_ADDR wPtr = BuildPtrTable(reinterpret_cast<GM_ADDR>(wAddr), 1);
        GM_ADDR scaleBPtr = BuildPtrTable(scaleBBase_, 2);
        GM_ADDR yPtr = BuildPtrTable(reinterpret_cast<GM_ADDR>(yAddr), 3);

        // 5. GmmASWKernel workspace 起始于 ptrTable + groupList 之后（72 bytes offset）
        GM_ADDR kernelWorkspace = reinterpret_cast<GM_ADDR>(
            reinterpret_cast<__gm__ uint8_t *>(workspaceBase_) + 72);

        // 6. 每次迭代创建新的 GmmASWKernel 实例
        //    MatmulImpl 内部的 buffer 状态在 Process() 后无法通过 TPipe::Reset() 完全清理，
        //    因此需要在栈上重新构造，确保 mm_.Init() 在干净状态下初始化
        tPipe_->Reset();
        GmmASWKernel<xType, wType, biasType, scaleType, yType, wFormat, aTrans, bTrans> gmmKernel;
        gmmKernel.Init(xPtr, wPtr, biasBase_, scaleBPtr,
                        groupListBase_, scaleABase_, yPtr,
                        kernelWorkspace,
                        reinterpret_cast<const ::GMMQuantParams *>(&currentTiling_.gmmQuantParams),
                        &currentTiling_.mmTilingData,
                        gmmArrayAddr_,
                        tPipe_);
        gmmKernel.Process();

        // 7. 累积 token 偏移
        tokenOffset_ += tokenCount;
    }
}

} // namespace MC2KernelTemplate

#endif // MC2_GMM_COMPUTE_OP_H
