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
 * \file shared_gmm_compute_op.h
 * \brief SharedGmmComputeOp 类：共享专家 GMM 计算操作封装
 */

#ifndef MC2_SHARED_GMM_COMPUTE_OP_H
#define MC2_SHARED_GMM_COMPUTE_OP_H

#include "../common/a2av_common_tiling.h"
#include "kernel_operator.h"
#if !defined(GQMM_CUBE_ON_THE_FLY_H) && !defined(MC2_GQMM_CUBE_ON_THE_FLY_H)
#include "../../3rd/gqmm_cube_on_the_fly.h"
#endif

using namespace AscendC;

namespace MC2KernelTemplate {

/**
 * 共享专家 GMM 计算操作
 * 封装单次 x @ weight + quant 计算，内部调用 GmmASWKernel
 *
 * 与 GmmComputeOp 的区别：
 * - 无多专家迭代，固定 groupNum=1
 * - 无 sendCnt/recvCnt 依赖，m/n/k 由调用方直接传入
 * - 无动态 tiling 刷新（host 侧已预计算好 sharedGmmTiling，kernel 直接使用）
 * - 不修改传入的 TilingData，直接透传给 GmmASWKernel
 * - 地址不需要 per-expert 偏移（只有一组 weight）
 * - Init 和 Process 均有 AIV 拦截，仅在 AIC 核上执行
 *
 * @tparam xType, wType, biasType, scaleType, yType  数据类型
 * @tparam wFormat       权重格式（如 CubeFormat::ND）
 * @tparam aTrans        输入 A 是否转置
 * @tparam bTrans        输入 B（权重）是否转置
 */
template <class xType, class wType, class biasType, class scaleType, class yType,
          CubeFormat wFormat, bool aTrans, bool bTrans>
class SharedGmmComputeOp {
public:
    using GMMQuantParams = Mc2GroupedMatmulTilingData::GMMQuantParams;

    __aicore__ inline SharedGmmComputeOp() = default;

    /**
     * 初始化
     *
     * @param x              输入 X 基地址，shape (m, k)
     * @param weight         权重基地址，shape (k, n) 或 (n, k) 取决于 bTrans
     * @param bias           偏置基地址（可为 nullptr）
     * @param scaleA         X 的量化 scale（perTokenScale，直接透传给 GmmASWKernel）
     * @param scaleB         Weight 的量化 scale（透传给 GmmASWKernel）
     * @param y              输出 Y 基地址，shape (m, n)，由调用方指定最终输出目标
     * @param tempAddr       临时空间基地址（内含 ptrTable / groupList / kernel workspace）
     * @param tempAddrSize   临时空间总大小（字节）
     * @param m              token 数量（即共享专家的 M 维度，通常为 BS）
     * @param n              输出维度 N
     * @param k              输入维度 K
     * @param sharedGmmTiling 共享专家 GMM Tiling（host 预计算，包含 quantParams 和 mmTilingData，直接透传不修改）
     * @param tPipe          TPipe 指针
     */
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR bias,
                                GM_ADDR scaleA, GM_ADDR scaleB,
                                GM_ADDR y, GM_ADDR tempAddr, uint64_t tempAddrSize,
                                uint64_t m, uint64_t n, uint64_t k,
                                const GMMQuantTilingData *sharedGmmTiling,
                                TPipe *tPipe);

    /**
     * 执行共享专家计算
     * 内部调用一次 GmmASWKernel.Init() + Process()
     */
    __aicore__ inline void Process();

private:
    // 直接保存 host 提供的 tiling 指针，不拷贝不修改
    const GMMQuantTilingData *sharedGmmTiling_ = nullptr;
    TILING_TYPE *gmmArrayAddr_ = nullptr;
    TPipe *tPipe_ = nullptr;

    // 基地址
    GM_ADDR xBase_ = nullptr;
    GM_ADDR weightBase_ = nullptr;
    GM_ADDR biasBase_ = nullptr;
    GM_ADDR scaleABase_ = nullptr;
    GM_ADDR scaleBBase_ = nullptr;
    GM_ADDR yBase_ = nullptr;
    GM_ADDR tempAddr_ = nullptr;
    uint64_t tempAddrSize_ = 0;

    // 维度（由调用方显式传入）
    uint64_t m_ = 0;
    uint64_t n_ = 0;
    uint64_t k_ = 0;

    // tempAddr 子区域指针
    GM_ADDR ptrTableBase_ = nullptr;
    GM_ADDR groupListBase_ = nullptr;

    // tempAddr 内存布局常量
    static constexpr uint64_t PTR_TABLE_SIZE = 64;   // 4 x 16B
    static constexpr uint64_t GROUP_LIST_SIZE = 8;    // 1 x int64_t
    static constexpr uint64_t KERNEL_WS_OFFSET = PTR_TABLE_SIZE + GROUP_LIST_SIZE; // 72

    /** 在 ptrTable 区域构建 GetTensorAddr 双重间接指针 */
    __aicore__ inline GM_ADDR BuildPtrTable(GM_ADDR dataAddr, uint32_t slotIdx);

    /** 将 groupList[0] = tokenCount 写入 groupList 区域 */
    __aicore__ inline void WriteSingleGroupList(uint64_t tokenCount);
};

// ========== 方法实现 ==========

template <class xType, class wType, class biasType, class scaleType, class yType,
          CubeFormat wFormat, bool aTrans, bool bTrans>
__aicore__ inline void
SharedGmmComputeOp<xType, wType, biasType, scaleType, yType, wFormat, aTrans, bTrans>::Init(
    GM_ADDR x, GM_ADDR weight, GM_ADDR bias,
    GM_ADDR scaleA, GM_ADDR scaleB,
    GM_ADDR y, GM_ADDR tempAddr, uint64_t tempAddrSize,
    uint64_t m, uint64_t n, uint64_t k,
    const GMMQuantTilingData *sharedGmmTiling,
    TPipe *tPipe)
{
    if ASCEND_IS_AIV {
        return;
    }

    // 保存基地址
    xBase_ = x;
    weightBase_ = weight;
    biasBase_ = bias;
    scaleABase_ = scaleA;
    scaleBBase_ = scaleB;
    yBase_ = y;
    tempAddr_ = tempAddr;
    tempAddrSize_ = tempAddrSize;

    // 保存维度
    m_ = m;
    n_ = n;
    k_ = k;

    // 保存 tiling 指针（不拷贝、不修改，直接透传给 GmmASWKernel）
    sharedGmmTiling_ = sharedGmmTiling;
    tPipe_ = tPipe;

    // tempAddr 布局：
    //   [0, 64)       ptrTable: 4 x 16B GetTensorAddr 双重间接指针
    //   [64, 72)      groupList: 1 x int64_t
    //   [72, ...)     GmmASWKernel workspace
    ptrTableBase_ = tempAddr;
    groupListBase_ = reinterpret_cast<GM_ADDR>(
        reinterpret_cast<__gm__ uint8_t *>(tempAddr) + PTR_TABLE_SIZE);

    // GmmASWKernel 只读 gmmArray，直接引用 tiling 中的连续地址
    gmmArrayAddr_ = reinterpret_cast<TILING_TYPE *>(
        const_cast<GMMArray *>(&sharedGmmTiling->gmmArray));
}

template <class xType, class wType, class biasType, class scaleType, class yType,
          CubeFormat wFormat, bool aTrans, bool bTrans>
__aicore__ inline void
SharedGmmComputeOp<xType, wType, biasType, scaleType, yType, wFormat, aTrans, bTrans>::Process()
{
    if ASCEND_IS_AIV {
        return;
    }

    // 写入 groupList[0] = m（单组，值为全部 token 数）
    WriteSingleGroupList(m_);

    // 构建 GetTensorAddr 指针表
    // slotIdx: 0=x, 1=weight, 2=scaleB, 3=y
    GM_ADDR xPtr = BuildPtrTable(xBase_, 0);
    GM_ADDR wPtr = BuildPtrTable(weightBase_, 1);
    GM_ADDR scaleBPtr = BuildPtrTable(scaleBBase_, 2);
    GM_ADDR yPtr = BuildPtrTable(yBase_, 3);

    // GmmASWKernel workspace 起始于 tempAddr 的 KERNEL_WS_OFFSET 处
    GM_ADDR kernelWorkspace = reinterpret_cast<GM_ADDR>(
        reinterpret_cast<__gm__ uint8_t *>(tempAddr_) + KERNEL_WS_OFFSET);

    // 创建 GmmASWKernel 实例，调用 Init + Process
    tPipe_->Reset();
    GmmASWKernel<xType, wType, biasType, scaleType, yType, wFormat, aTrans, bTrans> gmmKernel;
    gmmKernel.Init(xPtr, wPtr, biasBase_, scaleBPtr,
                    groupListBase_, scaleABase_, yPtr,
                    kernelWorkspace,
                    reinterpret_cast<const ::GMMQuantParams *>(&sharedGmmTiling_->gmmQuantParams),
                    &sharedGmmTiling_->mmTilingData,
                    gmmArrayAddr_,
                    tPipe_);
    gmmKernel.Process();
}

template <class xType, class wType, class biasType, class scaleType, class yType,
          CubeFormat wFormat, bool aTrans, bool bTrans>
__aicore__ inline GM_ADDR
SharedGmmComputeOp<xType, wType, biasType, scaleType, yType, wFormat, aTrans, bTrans>::BuildPtrTable(
    GM_ADDR dataAddr, uint32_t slotIdx)
{
    // 每个 slot 占 16 bytes (2 * uint64_t)
    __gm__ uint64_t *slot = reinterpret_cast<__gm__ uint64_t *>(
        reinterpret_cast<__gm__ uint8_t *>(ptrTableBase_) + slotIdx * 16);
    slot[0] = sizeof(uint64_t);  // byteOffset
    slot[1] = reinterpret_cast<uint64_t>(dataAddr);  // 实际数据地址
    return reinterpret_cast<GM_ADDR>(slot);
}

template <class xType, class wType, class biasType, class scaleType, class yType,
          CubeFormat wFormat, bool aTrans, bool bTrans>
__aicore__ inline void
SharedGmmComputeOp<xType, wType, biasType, scaleType, yType, wFormat, aTrans, bTrans>::WriteSingleGroupList(
    uint64_t tokenCount)
{
    __gm__ int64_t *groupList = reinterpret_cast<__gm__ int64_t *>(groupListBase_);
    groupList[0] = static_cast<int64_t>(tokenCount);
}

} // namespace MC2KernelTemplate

#endif // MC2_SHARED_GMM_COMPUTE_OP_H
