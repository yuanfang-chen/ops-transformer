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
 * \file flash_attention_score_kernel_infer_mla_fullquant.h
 * \brief MLA Fullquant Flash Attention Kernel层实现 (推理版本)
 * 
 * ============================================================================
 * 文件概述 / File Overview
 * ============================================================================
 * 本文件是华为昇腾CANN架构下MLA (Multi-head Latent Attention) 全量化Flash Attention
 * 算子的Kernel层核心实现，专为推理(Inference)场景优化。
 * 
 * 主要功能:
 * - 管理Cube和Vector计算单元的协同工作
 * - 实现多级流水线调度 (BMM1 → Softmax → BMM2)
 * - 支持Flash Decode (变长序列解码)
 * - 支持Split-KV并行计算
 * 
 * ============================================================================
 * 架构层级 / Architecture Layers
 * ============================================================================
 * ┌─────────────────────────────────────────────────────────────────────┐
 * │  Kernel层 (本文件 - flash_attention_score_kernel_infer_mla_...)     │
 * │  职责:                                                             │
 * │    - 多核任务分配与调度                                             │
 * │    - Cube/Vector流水线编排                                          │
 * │    - 主循环控制 (ProcessMainLoop)                                   │
 * ├─────────────────────────────────────────────────────────────────────┤
 * │  CubeBlock层 (flash_attention_score_block_cube_mla_...)             │
 * │  职责:                                                             │
 * │    - BMM1: Q×K^T 矩阵乘法 (计算attention scores)                    │
 * │    - BMM2: P×V 矩阵乘法 (计算最终输出)                              │
 * │    - 使用Cube计算单元进行矩阵运算                                   │
 * ├─────────────────────────────────────────────────────────────────────┤
 * │  VecBlock层 (flash_attention_score_block_vec_infer_...)             │
 * │  职责:                                                             │
 * │    - Softmax计算 (exp/sum/max)                                     │
 * │    - Post-Quantization (输出量化)                                    │
 * │    - Flash Decode多批次合并                                         │
 * │    - 使用Vector计算单元进行向量运算                                 │
 * └─────────────────────────────────────────────────────────────────────┘
 * 
 * ============================================================================
 * 核心数据流 / Data Flow
 * ============================================================================
 * 输入: Q(Query), K(Key), V(Value)
 *   ↓
 * [Cube] BMM1: Q × K^T → S (attention scores)
 *   ↓
 * [Vec] Softmax(S) → P (attention weights)  计算: P = exp(S - max(S)) / sum(exp(S - max(S)))
 *   ↓
 * [Cube] BMM2: P × V → O (output)
 *   ↓
 * [Vec] Post-Quant(O) → 最终输出
 * 
 * ============================================================================
 * 关键特性 / Key Features
 * ============================================================================
 * 1. MLA (Multi-head Latent Attention)
 *    - 将K/V分解为 Nope(压缩部分) + RoPE(旋转位置编码)
 *    - 支持FP8输入量化
 *    - 降低KV cache内存占用
 * 
 * 2. Flash Decode (FD)
 *    - 支持变长序列解码
 *    - 多批次结果合并机制
 *    - Split-KV分片计算
 * 
 * 3. Full Quantization
 *    - FP8输入量化
 *    - Post-Quantization输出
 *    - 支持Tensor级和Channel级量化
 * 
 * 4. 流水线并行
 *    - 4级流水线: BMM1 → Vec1(Softmax) → BMM2 → Vec2(Post-Quant)
 *    - 使用Ping-Pong缓冲减少等待
 *    - taskId模3/4调度确保数据依赖
 * 
 * ============================================================================
 * 模板参数说明 / Template Parameters
 * ============================================================================
 * - CubeBlockType: Cube计算块类型
 * - VecBlockType: Vector计算块类型
 * 
 * 以下参数通过CubeBlockType继承:
 * - INPUT_T: 输入数据类型 (fp8_e5m2_t, fp8_e4m3fn_t, hifloat8_t, half, bfloat16_t)
 * - OUTPUT_T: 输出数据类型
 * - T: 计算数据类型 (float或bfloat16_t)
 * - isInfer: 推理模式标志 (true)
 * - isFd: Flash Decode标志
 * - hasRope: 是否包含RoPE
 * - isPa: Page Attention标志
 * - hasAtten: Attention Mask标志
 * - hasDrop: Dropout标志 (推理时为false)
 * - enableKVPrefix: KV Prefix标志
 * - layout: 数据布局 (BNSD, BSHD, TND, NTD等)
 * - s1TemplateType, s2TemplateType, dTemplateType, dVTemplateType: 分块大小
 * 
 * ============================================================================
 * 关键数据结构 / Key Data Structures
 * ============================================================================
 * - RunParamStr<isInfer>: 运行参数结构，包含批次、头、序列维度索引
 * - RunInfo<isInfer>: 运行信息结构，包含计算偏移量、循环计数等
 * - ConstInfo<isInfer, hasRope>: 常量信息，维度大小、分块参数等
 * 
 * ============================================================================
 * 使用说明 / Usage Notes
 * ============================================================================
 * 1. 初始化阶段:
 *    - 调用InitUniqueConstInfo()初始化常量信息
 *    - 调用ProcessMainLoop()执行主计算循环
 * 
 * 2. Flash Decode场景:
 *    - Process()中会调用FlashDecodeCompute()合并多批次结果
 *    - 使用32KB UB缓冲区存储中间结果
 * 
 * 3. 多核调度:
 *    - 根据coreNum分配BN (Batch × N2) 任务
 *    - BN内部分配GS1 (Group × S1 Outer) 任务
 *    - 支持多核负载均衡
 */

#ifndef FLASH_ATTENTION_SCORE_KERNEL_INFER_MLA_FULLQUANT_H_
#define FLASH_ATTENTION_SCORE_KERNEL_INFER_MLA_FULLQUANT_H_
#include "flash_attention_score_kernel_base_fullquant.h"
#include "vf/vf_flash_decode.h"
#include "infer_flash_attention_comm.h"
#include "infer_flash_attention_kvcache.h"
#include "infer_flash_attention_sparse.h"

namespace BaseApi {

/**
 * @brief MLA Fullquant Flash Attention Kernel主类 (推理版本)
 * 
 * 继承自FlashAttentionScoreKernelBaseFullquant，负责:
 * - 多核任务分配与调度
 * - Cube/Vector流水线编排
 * - Flash Decode支持
 * 
 * @tparam CubeBlockType Cube计算块类型
 * @tparam VecBlockType Vector计算块类型
 */
template <typename CubeBlockType, typename VecBlockType>
class FlashAttentionScoreKernelInferMlaFullquant : public FlashAttentionScoreKernelBaseFullquant<FlashAttentionScoreKernelInferMlaFullquant<CubeBlockType, VecBlockType>, CubeBlockType, VecBlockType> {
public:
    ARGS_TRAITS;
    
    /**
     * @brief 是否执行Post-Quantization
     * 当输出类型不是half/bfloat16_t/float时启用
     * 用于将计算结果量化到目标数据类型 (如INT8/FP8等)
     */
    static constexpr bool POST_QUANT = !IsSameType<OUTPUT_T, half>::value && !IsSameType<OUTPUT_T, bfloat16_t>::value && !IsSameType<OUTPUT_T, float>::value;
    
    using BaseClass = FlashAttentionScoreKernelBaseFullquant<FlashAttentionScoreKernelInferMlaFullquant<CubeBlockType, VecBlockType>, CubeBlockType, VecBlockType>;
    
    /* =====================公开接口==================== */
    
    /**
     * @brief 初始化唯一常量信息
     * 
     * 从sharedParams初始化constInfo，包括:
     * - Split-KV参数 (isFd模式)
     * - Post-Quantization参数 (POST_QUANT模式)
     * - GQA/PA等特殊模式标志
     * - 维度大小和步长计算
     */
    __aicore__ inline void InitUniqueConstInfo();
    
    /**
     * @brief 初始化唯一运行信息
     * 
     * 根据运行参数初始化task相关的信息:
     * - 调用InitTaskParamByRun初始化任务参数
     * - 调用ComputeOffset计算数据偏移量
     * 
     * @param runParam 运行参数
     * @param runInfo 运行信息输出
     */
    __aicore__ inline void InitUniqueRunInfo(const RunParamStr<isInfer> &runParam, 
        RunInfo<isInfer> &runInfo);
    
    /**
     * @brief 主处理函数
     * 
     * Kernel层入口函数，流程:
     * 1. 如果needInit，执行SyncAll同步
     * 2. 调用ProcessMainLoop()执行主计算循环
     * 3. Flash Decode场景: 同步后调用FlashDecodeCompute()
     */
    __aicore__ inline void Process();
    
    /**
     * @brief 主计算循环
     * 
     * 实现4级流水线调度:
     * ┌──────────────────────────────────────────────────────────────┐
     * │  Pipeline Stage  │  AIC (Cube)      │  AIV (Vec)            │
     * ├──────────────────────────────────────────────────────────────┤
     * │  taskId % 4 = 0  │  IterateBmm1     │  -                    │
     * │  taskId % 4 = 1  │  -               │  ProcessVec1          │
     * │  taskId % 4 = 2  │  IterateBmm2     │  -                    │
     * │  taskId % 4 = 3  │  -               │  ProcessVec2          │
     * └──────────────────────────────────────────────────────────────┘
     * 
     * 多级循环结构:
     * - 外层: BN循环 (Batch × N2)
     * - 中层: GS1循环 (Group × S1 Outer)
     * - 内层: S2循环 (Sequence2)
     */
    __aicore__ inline void ProcessMainLoop();

private:
    /**
     * @brief 根据BN和GS1索引计算轴索引
     * 
     * 将全局的gS1Index分解为:
     * - goIdx: Group索引 (goIdx = gS1Index / s1OuterSize)
     * - s1oIdx: S1 Outer索引 (s1oIdx = gS1Index % s1OuterSize)
     * 
     * @param bnIndex Batch×N2索引
     * @param gS1Index Group×S1索引
     * @param runParam 运行参数输出
     */
    __aicore__ inline void ComputeAxisIdxByBnAndGs1(int64_t bnIndex, int64_t gS1Index,
                                                    RunParamStr<isInfer> &runParam);
};

/**
 * @brief 初始化唯一常量信息
 * 
 * 从sharedParams初始化constInfo，构建常量信息结构体。
 * 这些常量在计算过程中不会改变，用于:
 * - 维度大小计算
 * - 偏移量计算
 * - 内存布局控制
 * 
 * 初始化内容:
 * 1. Flash Decode参数 (isFd模式):
 *    - splitKVNum: KV分片数量
 *    - sInnerLoopSize: 每个分片的S2大小
 * 
 * 2. Post-Quantization参数 (POST_QUANT模式):
 *    - isPostQuantPerChnl: 是否为Channel级量化
 *    - isPostQuantBF16: 量化参数是否使用BF16存储
 * 
 * 3. 特殊模式标志:
 *    - isRowInvalid: 是否有无效行
 *    - headNumRatio: GQA头数比例
 *    - isGqa: 是否为GQA模式
 *    - isPfaGS1Merge: PFA的GS1合轴标志
 *    - isKvContinuous: KV是否连续存储
 * 
 * 4. 序列长度信息:
 *    - actualSeqLenSize: Q实际长度数组大小
 *    - actualSeqLenKVSize: KV实际长度数组大小
 *    - isActualLenDimsNull: 长度维度是否为空
 *    - isQHasLeftPadding: Q是否有左padding
 *    - isKVHasLeftPadding: KV是否有左padding
 * 
 * 5. Page Attention参数 (isPa模式):
 *    - blockTableDim2: Block Table第二维大小
 *    - blockSize: 每个Block的大小
 *    - paLayoutType: PA布局类型
 *    - paBlockNumSum: PA Block数量总和
 * 
 * 6. 布局和维度计算:
 *    - transposeLayout: 是否转置输出布局
 *    - attentionOutStride: 输出步长
 *    - n2GD, gDR, n2GDR: 组合维度
 *    - mm1RopeKa, mm1Ka: 矩阵乘法K维度
 */
template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void
FlashAttentionScoreKernelInferMlaFullquant<CubeBlockType, VecBlockType>::InitUniqueConstInfo()
{
    if constexpr (isFd) {
        this->constInfo.splitKVNum = this->sharedParams.splitKVNum;
        this->constInfo.sInnerLoopSize = CeilDiv(this->constInfo.s2Size, this->constInfo.splitKVNum);
    }
    if constexpr (POST_QUANT) {
        this->constInfo.isPostQuantPerChnl = this->sharedParams.isPostQuantPerChnl;
        this->constInfo.isPostQuantBF16 = this->sharedParams.isPostQuantBF16;
    }
    this->constInfo.isRowInvalid = this->sharedParams.isRowInvalid;
    this->constInfo.headNumRatio = this->sharedParams.headNumRatio;
    this->constInfo.isGqa = this->sharedParams.isGqa;
    this->constInfo.isPfaGS1Merge = this->sharedParams.isPfaGS1Merge;
    this->constInfo.isKvContinuous = this->sharedParams.isKvContinuous;
    this->constInfo.actualSeqLenSize = this->sharedParams.actualSeqLengthsSize;
    this->constInfo.actualSeqLenKVSize = this->sharedParams.actualSeqLengthsKVSize;
    this->constInfo.isActualLenDimsNull = static_cast<bool>(this->sharedParams.isActualSeqLengthsNull);
    this->constInfo.isActualLenDimsKVNull = static_cast<bool>(this->sharedParams.isActualSeqLengthsKVNull);
    this->constInfo.isQHasLeftPadding = static_cast<bool>(this->sharedParams.isQHasLeftPadding);
    this->constInfo.isKVHasLeftPadding = static_cast<bool>(this->sharedParams.isKVHasLeftPadding);
    // pageAttention
    if constexpr (isPa) {
        this->constInfo.blockTableDim2 = this->sharedParams.blockTableDim2;
        this->constInfo.blockSize = this->sharedParams.blockSize;
        this->constInfo.paLayoutType = this->sharedParams.paLayoutType;
        this->constInfo.paBlockNumSum = this->sharedParams.paBlockNumSum;
    }

    this->constInfo.transposeLayout = this->sharedParams.transposeLayout;
    if (this->constInfo.transposeLayout == static_cast<uint32_t>(TransposeLayoutEnum::BNSD_BSND)) {
        this->constInfo.attentionOutStride =
            (this->constInfo.n2GDv - this->constInfo.dSizeV) * sizeof(OUTPUT_T);
    }

    this->constInfo.n2GD = this->constInfo.n2G * this->constInfo.dSize;
    this->constInfo.gDR = this->constInfo.gSize * this->constInfo.dSizeRope;
    this->constInfo.n2GDR = this->constInfo.n2Size * this->constInfo.gDR;
    this->constInfo.mm1RopeKa = this->constInfo.dSizeRope;
    this->constInfo.mm1Ka = this->constInfo.dSize;
    this->constInfo.attentionOutStride = 0;
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void
FlashAttentionScoreKernelInferMlaFullquant<CubeBlockType, VecBlockType>::InitUniqueRunInfo(
    const RunParamStr<isInfer> &runParam, RunInfo<isInfer> &runInfo)
{
    InitTaskParamByRun<CHILD_SPEC_TEMPLATE_ARGS, BaseClass::useDn, BaseClass::enableKVPrefix>(runParam, runInfo);
    ComputeOffset<CHILD_SPEC_TEMPLATE_ARGS, BaseClass::useDn, BaseClass::enableKVPrefix>(runParam, this->constInfo, runInfo.s2LoopCount + runInfo.s2StartIdx / this->constInfo.s2BaseSize, runInfo);
}

/**
 * @brief 主计算循环 - 实现4级流水线调度
 * 
 * 这是Kernel层的核心函数，实现了多核并行和多级流水线调度。
 * 
 * ============================================================================
 * 多核任务分配
 * ============================================================================
 * 总核数计算:
 * - 非FD模式: actualCoreNums = coreNum (来自tiling)
 * - FD模式:   actualCoreNums = bSize * n2Size * splitKVNum
 * 
 * 超出范围的核直接返回，不参与计算。
 * 
 * ============================================================================
 * 任务层级结构
 * ============================================================================
 * ┌─────────────────────────────────────────────────────────────────────┐
 * │                        BN Loop (Batch × N2)                          │
 * │  ├─ bnIdx: 遍历该核负责的BN范围                                      │
 * │  │                                                                     │
 * │  └─ GS1 Loop (Group × S1 Outer)                                     │
 * │     ├─ gS1Index: 遍历该BN内的GS1任务                                │
 * │     │                                                                     │
 * │     └─ S2 Loop (Sequence2 Inner)                                    │
 * │        └─ s2LoopCount: 遍历S2分块                                    │
 * │           执行4级流水线计算                                          │
 * └─────────────────────────────────────────────────────────────────────┘
 * 
 * ============================================================================
 * 4级流水线调度
 * ============================================================================
 * 使用3个RunInfo缓冲实现Ping-Pong调度，taskId模4确定当前阶段:
 * 
 * ┌─────────────┬─────────────────────────────┬─────────────────────────────┐
 * │   taskId    │         AIC (Cube)         │         AIV (Vector)        │
 * ├─────────────┼─────────────────────────────┼─────────────────────────────┤
 * │   taskId    │  IterateBmm1()              │  -                          │
 * │   % 4 = 0   │  Q × K^T → S               │                             │
 * ├─────────────┼─────────────────────────────┼─────────────────────────────┤
 * │   taskId    │  -                          │  ProcessVec1()              │
 * │   % 4 = 1   │                             │  Softmax(S) → P            │
 * ├─────────────┼─────────────────────────────┼─────────────────────────────┤
 * │   taskId    │  IterateBmm2()              │  -                          │
 * │   % 4 = 2   │  P × V → O                 │                             │
 * ├─────────────┼─────────────────────────────┼─────────────────────────────┤
 * │   taskId    │  -                          │  ProcessVec2()              │
 * │   % 4 = 3   │                             │  Post-Quant → Output        │
 * └─────────────┴─────────────────────────────┴─────────────────────────────┘
 * 
 * ============================================================================
 * 流水线控制逻辑
 * ============================================================================
 * 使用3个标志位控制流水线收尾:
 * - notLast: 是否为最后一次GS1迭代 (用于Vec2)
 * - notLastTwoLoop: 是否为倒数第二次GS1迭代 (用于Vec1)
 * - notLastThreeLoop: 是否为倒数第三次GS1迭代 (用于Bmm2)
 * - isLastBmm1: 是否为倒数第四次GS1迭代 (用于Bmm1收尾)
 * 
 * 当s1NoNeedCalc或s2NoNeedCalc为true时，跳过当前S2循环。
 * 
 * ============================================================================
 * Flash Decode特殊处理
 * ============================================================================
 * FD模式下:
 * - bnIdx固定为0，只处理当前核的单一BN
 * - s1LoopTimes固定为1 (每个token单独处理)
 * - 使用splitKVNum并行处理KV的不同分片
 * - 最后需要调用FlashDecodeCompute合并结果
 */
template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreKernelInferMlaFullquant<CubeBlockType, VecBlockType>::ProcessMainLoop()
{
    int32_t actualCoreNums = this->sharedParams.coreNum;
    if constexpr (isFd) {
        actualCoreNums = this->sharedParams.bSize * this->constInfo.n2Size *
                         this->constInfo.splitKVNum; // b * n2 * splitkv
    }

    if (this->aicIdx >= actualCoreNums) {
        return;
    }
    // 确定核内切分起点
    int64_t gS1StartIdx;
    uint32_t bnStartIdx;
    uint32_t bnEndIdx;
    int64_t s2LoopLimit;
    int64_t nextGs1Idx = this->sharedParams.multiCoreInnerLimit;
    if constexpr (!isFd) {
        bnStartIdx = this->sharedParams.bnStartIdx;
        gS1StartIdx = this->sharedParams.multiCoreInnerOffset;
        if (likely((this->sharedParams.coreNum - 1) > this->aicIdx)) {
            bnEndIdx = this->sharedParams.bnEndIdx;
            if (nextGs1Idx != 0) {
                bnEndIdx++;
            }
        } else {
            bnEndIdx = this->sharedParams.bSize * this->constInfo.n2Size *
                this->constInfo.headNumRatio;
        }
    } else {
        gS1StartIdx = 0;
        bnStartIdx = 0;
        bnEndIdx = 1;
        s2LoopLimit = 0;
    }
    int64_t taskId = 0;
    bool notLast = true;
    bool isLastBmm1 = false;
    RunInfo<isInfer> runInfo[4];
    RunParamStr<isInfer> runParam;
    if constexpr (isFd) {
        runParam.boIdx = this->aicIdx / (this->constInfo.n2Size * this->constInfo.splitKVNum);
        runParam.n2oIdx = (this->aicIdx / this->constInfo.splitKVNum) % this->constInfo.n2Size;
        bnStartIdx = runParam.boIdx * this->constInfo.n2Size + runParam.n2oIdx;
        bnEndIdx = bnStartIdx + 1;
    }
    // 注意这里不等于0是因为，推理的在SetRunInfo中第一次也需要赋值runInfo.s1oIdx，boIdx，n2oIdx，goIdx
    // 训练这些值在multiCoreInnerIdx = 0的时候都是0，两边不统一
    int64_t multiCoreInnerIdx = 1;
    for (uint32_t bnIdx = bnStartIdx; bnIdx < bnEndIdx; ++bnIdx) {
        bool lastBN = (bnIdx == bnEndIdx - 1);
        if constexpr (!isFd) {
            runParam.boIdx = bnIdx / (this->constInfo.n2Size * this->constInfo.headNumRatio);
            runParam.n2oIdx = (bnIdx / this->constInfo.headNumRatio) % this->constInfo.n2Size;
        }
        ComputeParamBatch<CHILD_SPEC_TEMPLATE_ARGS, BaseClass::useDn, BaseClass::enableKVPrefix>(runParam, this->constInfo, this->attenMaskInfo,
            this->keyGm, this->actualSeqQlenAddr, this->actualSeqKvlenAddr);
        ComputeS1LoopInfo<CHILD_SPEC_TEMPLATE_ARGS, BaseClass::useDn, BaseClass::enableKVPrefix>(runParam, this->constInfo, lastBN,
                                                                      nextGs1Idx);
        if constexpr (isFd) {
            if (this->constInfo.sInnerLoopSize * (this->aicIdx % this->constInfo.splitKVNum) >
                runParam.actualS2Size) {
                runParam.s2LineEndIdx = 0;
            } else {
                int64_t tailSInnerLoopSize =
                    runParam.actualS2Size -
                    this->constInfo.sInnerLoopSize * (this->aicIdx % this->constInfo.splitKVNum);
                runParam.s2LineEndIdx = tailSInnerLoopSize > this->constInfo.sInnerLoopSize ?
                                        this->constInfo.sInnerLoopSize :
                                        tailSInnerLoopSize;
            }
            runParam.s1LoopTimes = 1;
        }
        int64_t tempGS1End = lastBN ? (runParam.s1LoopTimes + 3) : runParam.s1LoopTimes;
        for (int64_t gS1Index = gS1StartIdx; gS1Index < tempGS1End; ++gS1Index) {
            bool notLastThreeLoop = true;
            bool notLastTwoLoop = true;
            if (lastBN) {
                int32_t extraGS1 = gS1Index - runParam.s1LoopTimes;
                switch (extraGS1) {
                    case -1:
                        isLastBmm1 = true;
                        break;
                    case 0:
                        notLastThreeLoop = false;
                        break;
                    case 1:
                        notLastThreeLoop = false;
                        notLastTwoLoop = false;
                        break;
                    case 2:
                        notLast = false;
                        notLastThreeLoop = false;
                        notLastTwoLoop = false;
                        break;
                    default:
                        break;
                }
            }
            if (notLastThreeLoop) {
                this->ComputeAxisIdxByBnAndGs1(bnIdx, gS1Index, runParam);
                bool s1NoNeedCalc = ComputeParamS1<CHILD_SPEC_TEMPLATE_ARGS, BaseClass::useDn, BaseClass::enableKVPrefix>(
                    runParam, this->constInfo, gS1Index, this->actualSeqQlenAddr, this->pseInfo);
                bool s2NoNeedCalc =
                    ComputeS2LoopInfo<CHILD_SPEC_TEMPLATE_ARGS, BaseClass::useDn, BaseClass::enableKVPrefix>(runParam, this->constInfo);
                // s1和s2有任意一个不需要算, 则continue, 如果是当前核最后一次循环，则补充计算taskIdx+2的部分
                if (s1NoNeedCalc || s2NoNeedCalc) {
                    continue;
                }
                s2LoopLimit = runParam.s2LoopEndIdx - 1;
            } else {
                s2LoopLimit = 0;
            }

            for (int64_t s2LoopCount = 0; s2LoopCount <= s2LoopLimit; ++s2LoopCount) {
                if (notLastThreeLoop) {
                    RunInfo<isInfer> &runInfo1 = runInfo[taskId & 3];
                    this->SetRunInfo(runInfo1, runParam, taskId, s2LoopCount, s2LoopLimit, multiCoreInnerIdx);
                    if ASCEND_IS_AIC {
                        this->cubeBlock.IterateBmm1(this->bmm1Buffers.Get(), runInfo1, this->constInfo);
                    }
                }
                if (taskId > 0 && notLastTwoLoop) {
                    if ASCEND_IS_AIV {
                        auto &runInfo3 = runInfo[(taskId + 3) & 3];
                        this->vecBlock.ProcessVec1(this->l1PBuffers.Get(), this->bmm1Buffers.Get(), runInfo3,
                            this->constInfo);
                    }
                }
                if (taskId > 1 && notLast) {
                    if ASCEND_IS_AIC {
                        RunInfo<isInfer> &runInfo2 = runInfo[(taskId + 2) & 3];
                        if constexpr (BaseClass::bmm2Write2Ub) {
                            this->cubeBlock.IterateBmm2(this->bmm2Buffers.Get(), this->l1PBuffers, runInfo2,
                                this->constInfo);
                        } else {
                            this->cubeBlock.IterateBmm2(this->bmm2ResGmBuffers.Get(), this->l1PBuffers, runInfo2,
                                this->constInfo);
                        }
                    }
                }
                if (taskId > 2) {
                    if ASCEND_IS_AIV {
                        RunInfo<isInfer> &runInfo3 = runInfo[(taskId + 1) & 3];
                        if constexpr (BaseClass::bmm2Write2Ub) {
                            this->vecBlock.ProcessVec2(this->bmm2Buffers.Get(), runInfo3, this->constInfo);
                        } else {
                            this->vecBlock.ProcessVec2(this->bmm2ResGmBuffers.Get(), runInfo3, this->constInfo);
                        }
                    }
                }
                ++taskId;
            }
            ++multiCoreInnerIdx;
        }
        gS1StartIdx = 0;
    }
}

/**
 * @brief 主处理函数 - Kernel层入口
 * 
 * 完整的处理流程:
 * 1. 初始化同步 (needInit模式):
 *    - 如果needInit=1，调用SyncAll<false>()进行初次同步
 *    - 确保所有核的缓冲初始化完成
 * 
 * 2. 主计算循环:
 *    - 调用ProcessMainLoop()执行4级流水线计算
 *    - 在AIC核上执行BMM1和BMM2
 *    - 在AIV核上执行Softmax和Post-Quantization
 * 
 * 3. Flash Decode后处理 (仅FD模式 + AIV核):
 *    - 调用SyncAll()确保主循环完成
 *    - 调用InitFDBuffers()重新初始化32KB UB缓冲
 *    - 调用FlashDecodeCompute()合并多批次解码结果
 * 
 * @note SyncAll的调用时机:
 *       - 初始化时: SyncAll<false> (不等待自身)
 *       - FD后处理前: SyncAll() (完整同步)
 */
template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreKernelInferMlaFullquant<CubeBlockType, VecBlockType>::Process()
{
    // SyncAll Cube和Vector都需要调用
    if (this->sharedParams.needInit) {
        SyncAll<false>();
    }
    ProcessMainLoop();
    if constexpr (isFd) {
        if ASCEND_IS_AIV {
            SyncAll();
            this->vecBlock.InitFDBuffers(this->constInfo);
            this->vecBlock.FlashDecodeCompute(this->constInfo, this->keyGm, this->actualSeqKvlenAddr);
        }
    }
}

// =========================================== 私有辅助函数 ===========================================

/**
 * @brief 根据BN和GS1索引计算轴索引
 * 
 * 将一维的gS1Index (Group × S1 Outer索引) 分解为二维索引:
 * - goIdx: Group索引，用于访问不同的attention head组
 * - s1oIdx: S1 Outer索引，用于访问sequence outer维度
 * 
 * 计算公式:
 * - goIdx = gS1Index / s1OuterSize
 * - s1oIdx = gS1Index % s1OuterSize
 * 
 * 例如:
 * - s1OuterSize = 4, gS1Index = 9
 * - goIdx = 9 / 4 = 2
 * - s1oIdx = 9 % 4 = 1
 * 
 * 这些索引用于:
 * - 计算数据偏移量
 * - 确定当前处理的任务块
 * - 控制循环边界
 * 
 * @param bnIndex Batch × N2索引 (在ProcessMainLoop的BN循环中确定)
 * @param gS1Index Group × S1索引 (在ProcessMainLoop的GS1循环中确定)
 * @param runParam 运行参数输出，包含goIdx和s1oIdx
 */
template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreKernelInferMlaFullquant<CubeBlockType, VecBlockType>::ComputeAxisIdxByBnAndGs1(
    int64_t bnIndex, int64_t gS1Index, RunParamStr<isInfer> &runParam)
{
    runParam.goIdx = gS1Index / this->constInfo.s1OuterSize;
    runParam.s1oIdx = gS1Index % this->constInfo.s1OuterSize;
}
}
#endif