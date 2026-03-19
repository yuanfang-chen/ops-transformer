/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file flash_attention_score_block_cube_mla_fullquant.h
 * \brief MLA Fullquant Flash Attention CubeBlock层实现
 * 
 * ============================================================================
 * 文件概述 / File Overview
 * ============================================================================
 * 本文件实现MLA (Multi-head Latent Attention) 全量化Flash Attention算子的
 * Cube计算块层，负责矩阵乘法运算:
 * - BMM1: Q × K^T (计算attention scores)
 * - BMM2: P × V (计算最终输出)
 * 
 * ============================================================================
 * MLA核心概念 / MLA Core Concepts
 * ============================================================================
 * MLA (Multi-head Latent Attention) 是一种内存高效的注意力机制:
 * 
 * 传统MHA (Multi-head Attention):
 * - 每个head有完整的K和V向量
 * - KV Cache占用: batch_size × seq_len × num_heads × head_dim
 * 
 * MLA (Multi-head Latent Attention):
 * ┌─────────────────────────────────────────────────────────────────────┐
 * │                        MLA结构图                                      │
 * │                                                                     │
 * │    Q ──┬─> Q Nope ──────────────────> Q_compressed                 │
 * │        │                                                          │
 * │        └─> Q RoPE ──────────────────> Q_RoPE                      │
 * │                                                                     │
 * │    K ──┬─> K Nope ──> Compress ──> K_compressed ──┬─> K_Lora     │
 * │        │                                          │               │
 * │        └─> K RoPE ────────────────────────────────┴─> K_RoPE     │
 * │                                                                     │
 * │    V ────────────────────────────────────────────────> V_Lora      │
 * └─────────────────────────────────────────────────────────────────────┘
 * 
 * 关键特点:
 * - K/V被压缩为潜在向量 (Latent Vector)
 * - K分解为 K_compressed + K_RoPE
 * - Q分解为 Q_compressed + Q_RoPE
 * - 大幅降低KV Cache内存占用
 * 
 * ============================================================================
 * 矩阵乘法计算流程 / Matrix Multiplication Flow
 * ============================================================================
 * 
 * BMM1: Q × K^T → Attention Scores
 * ┌─────────────────────────────────────────────────────────────────────┐
 * │                                                                     │
 * │   Q_compressed [S1, Dv]     K_compressed^T [Dk, S2]               │
 * │         │                              │                            │
 * │         ▼                              │                            │
 * │   QNope [S1, Dv] ────────────────────│───●                        │
 * │                                       │   │                        │
 * │   Q_RoPE [S1, Dr]                    ▼   ▼                        │
 * │                                       KNope [S2, Dk]               │
 * │                                          │                           │
 * │   Nope × Nope^T ────────────────────────│───▶ ScoreNope            │
 * │                                          │                           │
 * │   Q_RoPE [S1, Dr] ◀────────────────────│───▶ ScoreRoPE            │
 * │                                          │                           │
 * │   K_RoPE [S2, Dr] ─────────────────────▶ Add                       │
 * │                                          │                           │
 * │                                          ▼                           │
 * │                                  Score [S1, S2]                     │
 * │                                                                     │
 * └─────────────────────────────────────────────────────────────────────┘
 * 
 * 关键操作:
 * 1. QNope × KNope^T → ScoreNope
 * 2. QRoPE × KRoPE^T → ScoreRoPE  
 * 3. ScoreNope + ScoreRoPE → Final Score
 * 
 * BMM2: P × V → Output
 * ┌─────────────────────────────────────────────────────────────────────┐
 * │                                                                     │
 * │   P [S1, S2] (Softmax后的attention weights)                        │
 * │        │                                                            │
 * │        ▼                                                            │
 * │   P × V_Lora ─────────────────────────────────────────▶ Output     │
 * │            [S1, Dv]                                               │
 * │                                                                     │
 * └─────────────────────────────────────────────────────────────────────┘
 * 
 * ============================================================================
 * 数据存储层级 / Memory Hierarchy
 * ============================================================================
 * 
 *        ┌─────────────────────────────────────┐
 *        │           GM (Global Memory)        │
 *        │  - Query, Key, Value (原始数据)     │
 *        │  - 输出结果                         │
 *        │  - workspace                        │
 *        └─────────────────────────────────────┘
 *                    ▲           │
 *                    │   DataCopy │
 *                    │           ▼
 *        ┌─────────────────────────────────────┐
 *        │           L1 (On-Chip Buffer)        │
 *        │  - Q Nope, Q RoPE                  │
 *        │  - K Nope, K RoPE                   │
 *        │  - V (复用K的buffer)                │
 *        │  - Softmax后的P                     │
 *        └─────────────────────────────────────┘
 *                    ▲           │
 *                    │  Matmul   │
 *                    │           ▼
 *        ┌─────────────────────────────────────┐
 *        │           L0 (Matrix Multiply Unit) │
 *        │  - L0A: 左矩阵A                     │
 *        │  - L0B: 右矩阵B                     │
 *        │  - L0C: 结果矩阵C                   │
 *        └─────────────────────────────────────┘
 *                    ▲           │
 *                    │  Fixpipe  │
 *                    │           ▼
 *        ┌─────────────────────────────────────┐
 *        │           UB (Unified Buffer)        │
 *        │  - BMM1结果 (待Softmax)            │
 *        │  - BMM2结果 (待Post-Quant)          │
 *        └─────────────────────────────────────┘
 * 
 * ============================================================================
 * Buffer管理策略 / Buffer Management Strategy
 * ============================================================================
 * 
 * L1 Buffer策略选择:
 * ┌─────────────────────────────────────────────────────────────────────────┐
 * │ 条件                                            │ Buffer策略            │
 * ├─────────────────────────────────────────────────────────────────────────┤
 * │ float类型输入                                    │ SingleBuffer          │
 * │ 非FP8 + s2BaseSize=256 + dBaseSize>128         │ SingleBuffer          │
 * │ FP8 + s2BaseSize=128 + dBaseSize=576           │ 4Buffer               │
 * │ 其他情况                                         │ DoubleBuffer          │
 * └─────────────────────────────────────────────────────────────────────────┘
 * 
 * L0 Buffer策略选择:
 * ┌─────────────────────────────────────────────────────────────────────────┐
 * │ Buffer │ 条件                                           │ 策略            │
 * ├─────────────────────────────────────────────────────────────────────────┤
 * │ L0A     │ float类型                                      │ SingleBuffer    │
 * │         │ 其他                                           │ DoubleBuffer    │
 * ├─────────────────────────────────────────────────────────────────────────┤
 * │ L0B     │ float类型或(非FP8+s2=256+d>128)               │ SingleBuffer    │
 * │         │ 其他                                           │ DoubleBuffer    │
 * ├─────────────────────────────────────────────────────────────────────────┤
 * │ L0C     │ s1*s2和s1*dV都小于阈值                        │ 4Buffer         │
 * │         │ 其他                                           │ DoubleBuffer    │
 * └─────────────────────────────────────────────────────────────────────────┘
 * 
 * DoubleBuffer: 允许同时预加载下一块数据到备用buffer
 * 4Buffer: 允许同时预加载3块数据，提高并行度
 * SingleBuffer: 简单但可能效率较低
 * 
 * ============================================================================
 * 核心优化技术 / Core Optimization Techniques
 * ============================================================================
 * 
 * 1. 左矩阵复用 (Q Matrix Reuse):
 *    - 在GS1循环内，Q矩阵在多个S2迭代中复用
 *    - s2LoopCount==0时加载Q到L1
 *    - 后续迭代使用l1QBuffers.GetPre()复用
 * 
 * 2. Ping-Pong调度:
 *    - Q使用Ping-Pong在GS1循环间切换
 *    - 提高L1利用率
 * 
 * 3. MTE同步:
 *    - MTE1: GM → L1的数据搬运事件
 *    - MTE2: L1 → L0的数据搬运事件
 *    - MTE3: L0 → GM/UB的数据搬运事件
 *    - FIX: Matrix Multiply Unit完成事件
 *    - 使用Wait/Set管理依赖关系
 * 
 * 4. Fixpipe:
 *    - L0C → UB的数据搬运
 *    - 使用ROW_MAJOR布局
 *    - 支持对齐和分块传输
 */

/*!
 * \file flash_attention_score_block_cube_mla_fullquant.h
 * \brief
 */
#ifndef FLASH_ATTENTION_SCORE_BLOCK_CUBE_MLA_FULLQUANT_H_
#define FLASH_ATTENTION_SCORE_BLOCK_CUBE_MLA_FULLQUANT_H_
#include "util_regbase.h"
#include "../offset_calculator.h"
#include "../matmul.h"
#include "../FixpipeOut.h"
#include "../CopyInL1.h"

#include "infer_flash_attention_comm.h"
#include "flash_attention_score_common_regbase.h"
#include "kernel_operator_list_tensor_intf.h"
using namespace AscendC;
using namespace AscendC::Impl::Detail;
using namespace regbaseutil;
using namespace fa_base_matmul;
namespace BaseApi {

/**
 * @brief CubeBlock辅助函数命名空间
 * 
 * 包含:
 * - GetQueryGmFormat: 根据布局获取Query的GM格式
 * - GetKVGmFormat: 根据布局获取KV的GM格式
 * - QL1BuffSel: Query的L1 Buffer选择策略
 * - KVL1BuffSel: KV的L1 Buffer选择策略
 * - L0ABuffSel: L0A Buffer选择策略
 * - L0BBuffSel: L0B Buffer选择策略
 * - L0CBuffSel: L0C Buffer选择策略
 */
namespace BlockCubeMlaFullquant {

/**
 * @brief 根据布局类型获取Query的GM数据格式
 * 
 * 不同布局下的Query存储格式:
 * - BSHD布局 → BSNGD格式
 * - SBHD布局 → SBNGD格式
 * - BNSD布局 → BNGSD格式
 * - TND布局  → TNGD格式
 * - NTD布局  → NGTD格式
 * 
 * @tparam LAYOUT 布局类型
 * @return GmFormat GM格式枚举值
 */
template <LayOutTypeEnum LAYOUT>
__aicore__ inline constexpr GmFormat GetQueryGmFormat()
{
    if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_BSH) {
        return GmFormat::BSNGD;
    } else if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_SBH) {
        return GmFormat::SBNGD;
    } else if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_BNSD) {
        return GmFormat::BNGSD;
    } else if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_TND) {
        return GmFormat::TNGD;
    } else {
        return GmFormat::NGTD;
    }
}

/**
 * @brief 根据布局类型获取KV的GM数据格式
 * 
 * 不同布局下的Key/Value存储格式:
 * - BSHD布局 → BSND格式
 * - SBHD布局 → SBND格式
 * - BNSD布局 → BNSD格式
 * - TND布局  → TND格式
 * - NTD布局  → NTD格式
 * 
 * @tparam LAYOUT 布局类型
 * @return GmFormat GM格式枚举值
 */
template <LayOutTypeEnum LAYOUT>
__aicore__ inline constexpr GmFormat GetKVGmFormat()
{
    if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_BSH) {
        return GmFormat::BSND;
    } else if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_SBH) {
        return GmFormat::SBND;
    } else if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_BNSD) {
        return GmFormat::BNSD;
    } else if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_TND) {
        return GmFormat::TND;
    } else {
        return GmFormat::NTD;
    }
}

/* ============确定Query的L1类型============= */
/**
 * @brief Query L1 Buffer选择策略
 * 
 * 选择依据:
 * - float类型: 使用SingleBuffer
 * - 非FP8且dBaseSize > 256: 使用SingleBuffer
 * - 其他情况: 使用DoubleBuffer
 * 
 * 原因: 大d维度或高精度类型需要更大buffer
 */
template <typename INPUT_T, uint32_t dBaseSize>
struct QL1BuffSel {
    using Type = std::conditional_t<
        std::is_same_v<INPUT_T, float> ||
        (!(std::is_same_v<INPUT_T, fp8_e4m3fn_t> ||
           std::is_same_v<INPUT_T, fp8_e5m2_t> ||
           std::is_same_v<INPUT_T, hifloat8_t>) && dBaseSize > 256),
        BuffersPolicySingleBuffer<BufferType::L1>,
        BuffersPolicyDB<BufferType::L1>>;
};

/* ============确定Key的L1类型============= */
/**
 * @brief KV L1 Buffer选择策略
 * 
 * 选择依据:
 * - FP8 + s2BaseSize=128 + dBaseSize=576: 使用4Buffer (MLA特殊优化)
 * - 非FP8 + s2BaseSize=256 + dBaseSize>128: 使用SingleBuffer
 * - 其他情况: 使用DoubleBuffer
 */
template <typename INPUT_T, uint32_t s2BaseSize, uint32_t dBaseSize>
struct KVL1BuffSel {
    constexpr static bool isFP8DType =  
            std::is_same_v<INPUT_T, fp8_e4m3fn_t> ||
           std::is_same_v<INPUT_T, fp8_e5m2_t> ||
           std::is_same_v<INPUT_T, hifloat8_t>;
    using Type = std::conditional_t<
            (isFP8DType && s2BaseSize == 128 && dBaseSize == 576),
            BuffersPolicy4buff<BufferType::L1>,
            std::conditional_t<
                (!(isFP8DType) && s2BaseSize == 256 && dBaseSize > 128),
                BuffersPolicySingleBuffer<BufferType::L1>,
                BuffersPolicyDB<BufferType::L1>
            >
        >;
};

/* ============确定L0A的类型============= */
/**
 * @brief L0A Buffer选择策略
 * 
 * float类型使用SingleBuffer，其他使用DoubleBuffer
 */
template <typename INPUT_T>
struct L0ABuffSel {
    using Type = std::conditional_t<
        std::is_same_v<INPUT_T, float>,
        BuffersPolicySingleBuffer<BufferType::L0A>,
        BuffersPolicyDB<BufferType::L0A>>;
};

/* ============确定L0B的类型============= */
/**
 * @brief L0B Buffer选择策略
 * 
 * 选择依据:
 * - float类型: SingleBuffer
 * - 非FP8 + s2BaseSize=256 + dBaseSize>128: SingleBuffer
 * - 其他情况: DoubleBuffer
 */
template <typename INPUT_T, uint32_t s2BaseSize, uint32_t dBaseSize>
struct L0BBuffSel {
    using Type = std::conditional_t<
        std::is_same_v<INPUT_T, float> || (s2BaseSize == 256 && dBaseSize > 128 && 
        !(std::is_same_v<INPUT_T, fp8_e4m3fn_t> ||
           std::is_same_v<INPUT_T, fp8_e5m2_t> ||
           std::is_same_v<INPUT_T, hifloat8_t>)),
        BuffersPolicySingleBuffer<BufferType::L0B>,
        BuffersPolicyDB<BufferType::L0B>>;
};

/* ============确定L0C的类型============= */
/**
 * @brief L0C Buffer选择策略
 * 
 * 当s1BaseSize * s2BaseSize和s1BaseSize * dVBaseSize都小于L0C可用空间的1/4时，
 * 使用4Buffer以支持更大的并行度
 */
template <typename INPUT_T, uint32_t s1BaseSize, uint32_t s2BaseSize, uint32_t dVBaseSize>
struct L0CBuffSel {
    using Type = std::conditional_t<
        (s1BaseSize * s2BaseSize * FLOAT_BYTES <= (L0C_SIZE * KB_TO_BYTES) / NUM_4 && s1BaseSize * dVBaseSize * FLOAT_BYTES <= (L0C_SIZE * KB_TO_BYTES) / NUM_4),
        BuffersPolicy4buff<BufferType::L0C>,
        BuffersPolicyDB<BufferType::L0C>>;
};
}


TEMPLATES_DEF

/**
 * @brief MLA Fullquant Cube计算块主类
 * 
 * 继承自基类，实现MLA特有的矩阵乘法逻辑。
 * 
 * 主要职责:
 * 1. 管理GM到L1的数据搬运 (Q, K, V, RoPE)
 * 2. 调用Cube矩阵乘法单元执行BMM1和BMM2
 * 3. 管理L1/L0 buffer的分配和复用
 * 4. 处理Page Attention场景
 * 
 * @tparam TEMPLATE_ARGS 模板参数包
 */
class FABlockCubeMlaFullquant {
public:
    /* =================编译期常量的基本块信息================= */
    
    /**
     * @brief 基本块大小 - S1维度
     * 从s1TemplateType模板参数获取
     */
    static constexpr uint32_t s1BaseSize = (uint32_t)s1TemplateType;
    
    /**
     * @brief 基本块大小 - S2维度
     * 从s2TemplateType模板参数获取
     */
    static constexpr uint32_t s2BaseSize = (uint32_t)s2TemplateType;
    
    /**
     * @brief 基本块大小 - D维度
     * 从dTemplateType模板参数获取
     */
    static constexpr uint32_t dBaseSize = (uint32_t)dTemplateType;
    
    /**
     * @brief 基本块大小 - DV维度 (Value的head维度)
     * 从dVTemplateType模板参数获取
     */
    static constexpr uint32_t dVBaseSize = (uint32_t)dVTemplateType;
    
    /**
     * @brief S2分片大小
     * 固定为256，用于分片处理长序列
     */
    static constexpr uint32_t s2SplitSize = (uint32_t)256;
    
    /**
     * @brief 是否为FP8输入
     * 检测输入类型是否为FP8格式 (fp8_e5m2_t, fp8_e4m3fn_t, hifloat8_t)
     */
    static constexpr bool isFp8 = IsSameType<INPUT_T, fp8_e5m2_t>::value ||
                                IsSameType<INPUT_T, fp8_e4m3fn_t>::value ||
                                IsSameType<INPUT_T, hifloat8_t>::value;
    
    /**
     * @brief 是否启用MLA Fullquant模式
     * 条件: FP8输入 + RoPE
     * MLA将K/V分解为: K = K_compressed + RoPE
     */
    static constexpr bool isMlaFullQuant = isFp8 && hasRope;
    
    /**
     * @brief 是否使用DN (Destination Normalization)
     * MLA Fullquant不使用DN
     */
    static constexpr bool useDn = false;
    
    /**
     * @brief 是否使用NZ格式
     * MLA Fullquant不使用NZ
     */
    static constexpr bool useNz = false;
    
    /**
     * @brief RoPE数据类型选择
     * MLA模式使用bfloat16_t存储RoPE，否则使用INPUT_T
     */
    using ROPE_T = std::conditional_t<isMlaFullQuant, bfloat16_t, INPUT_T>;
    
    /**
     * @brief BMM2输出位置
     * 决定结果写入UB还是GM
     */
    static constexpr TPosition bmm2OutPos = GetC2Position(dVTemplateType,
                                                          UbOutCondition<INPUT_T>(IsSameType<INPUT_T, float>::value, pseMode, hasAtten, hasDrop, hasRope,
                                                                                s1BaseSize == 64), (s2BaseSize == 256 && s1BaseSize == 64), isMlaFullQuant);
    
    /**
     * @brief BMM2是否写入UB
     * 当bmm2OutPos为VECCALC时为true
     */
    static constexpr bool bmm2Write2Ub = bmm2OutPos == TPosition::VECCALC;
    
    /**
     * @brief BMM2 Fixpipe配置
     * 配置输出布局和目标位置
     */
    static constexpr FixpipeConfig BMM2_FIXPIPE_CONFIG = {CO2Layout::ROW_MAJOR, bmm2Write2Ub};
    
    /**
     * @brief BMM2结果存储位置类型
     * 根据bmm2Write2Ub选择UB或GM
     */
    using mm2ResPos = typename std::conditional<bmm2Write2Ub, Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>,
        Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_FORWARD>>::type;

    /* =================公开接口================= */
    
    /**
     * @brief 构造函数
     */
    __aicore__ inline FABlockCubeMlaFullquant() {};
    
    /**
     * @brief 初始化Cube Block
     * 
     * 在AIC核上执行，初始化:
     * - TPipe指针
     * - L1 Buffer管理器
     * - GM张量 (Query, QueryRoPE, KeyRoPE)
     * - 局部Buffer
     * 
     * @param pipe TPipe指针
     * @param l1BufferManagerPtr L1 Buffer管理器指针
     * @param query Query数据指针
     * @param key Key数据指针
     * @param value Value数据指针
     * @param blockTable Block Table指针 (Page Attention)
     * @param queryRope Query RoPE数据指针
     * @param keyRope Key RoPE数据指针
     */
    __aicore__ inline void InitCubeBlock(TPipe *pipe, BufferManager<BufferType::L1> *l1BufferManagerPtr,
        __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *blockTable, 
        __gm__ uint8_t *queryRope, __gm__ uint8_t *keyRope);
    
    /**
     * @brief 初始化Cube输入
     * 
     * 初始化Key/Value GM张量和offset calculator
     * 
     * @param key Key数据指针
     * @param value Value数据指针
     * @param sharedParams 共享参数
     * @param attenMaskInfo Attention Mask信息
     * @param actualSeqQlenAddr Q实际长度地址
     * @param actualSeqKvlenAddr KV实际长度地址
     * @param keySharedPrefix Key Prefix指针
     * @param valueSharedPrefix Value Prefix指针
     * @param actualSharedPrefixLen Prefix实际长度地址
     */
    __aicore__ inline void InitCubeInput(__gm__ uint8_t *key, __gm__ uint8_t *value,
                                         CVSharedParams<isInfer, isPa> *sharedParams, AttenMaskInfo *attenMaskInfo,
                                         __gm__ int64_t *actualSeqQlenAddr, __gm__ int64_t *actualSeqKvlenAddr,
                                         __gm__ uint8_t *keySharedPrefix, __gm__ uint8_t *valueSharedPrefix,
                                         __gm__ uint8_t *actualSharedPrefixLen);
    
    /**
     * @brief BMM1迭代
     * 
     * MLA Fullquant的BMM1计算流程:
     * 1. 计算S1和S2坐标
     * 2. 加载Q到L1 (Ping-Pong)
     * 3. 加载K到L1
     * 4. 执行QNope × KNope^T
     * 5. 执行QRoPE × KRoPE^T
     * 6. Fixpipe结果到UB
     * 
     * @param output BMM1结果UB缓冲
     * @param runInfo 运行信息
     * @param constInfo 常量信息
     */
    __aicore__ inline void IterateBmm1(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &output,
        RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo);

    /**
     * @brief BMM2迭代
     * 
     * MLA Fullquant的BMM2计算:
     * P × V → Output
     * 
     * @param outputBuf BMM2结果缓冲 (UB或GM)
     * @param inputBuf BMM2输入缓冲 (L1上的P)
     * @param runInfo 运行信息
     * @param constInfo 常量信息
     */
    __aicore__ inline void IterateBmm2(mm2ResPos &outputBuf,
        BuffersPolicy3buff<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &inputBuf, RunInfo<isInfer> &runInfo,
        ConstInfo<isInfer, hasRope> &constInfo);
    
    /**
     * @brief 初始化反量化参数
     * MLA Fullquant不使用单独的dequant，这里为空实现
     */
    __aicore__ inline void InitDequantParams(__gm__ uint8_t *deqScaleQ,
        __gm__ uint8_t *deqScaleK, __gm__ uint8_t *deqScaleV) {};

private:
    /* =================私有成员函数================= */
    
    /**
     * @brief 初始化局部Buffer
     * 
     * 分配L1, L0A, L0B, L0C Buffer
     * 根据编译期条件选择Single/Double/4 Buffer策略
     */
    __aicore__ inline void InitLocalBuffer();
    
    /**
     * @brief 初始化GM张量
     * 
     * 设置Query, Key, Value, RoPE的shape和strides
     * 根据布局类型选择不同的初始化方式
     */
    __aicore__ inline void InitGmTensor(CVSharedParams<isInfer, isPa> *sharedParams, __gm__ int64_t *actualSeqQlenAddr,
        __gm__ int64_t *actualSeqKvlenAddr);
    
    /**
     * @brief 计算S1坐标
     * 
     * 计算当前任务的S1起始坐标:
     * - base = s1oIdx * s1BaseSize
     * - padding调整 (推理时)
     */
    __aicore__ inline void CalcS1Coord(RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo);
    
    /**
     * @brief 计算S2坐标
     * 
     * 计算当前任务的S2起始坐标:
     * - base = s2StartIdx + s2LoopCount * s2BaseSize
     * - padding调整
     * - Flash Decode分片调整
     */
    __aicore__ inline void CalcS2Coord(RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo);
    
    /**
     * @brief 从Tensor List获取KV
     * 
     * 处理变长序列场景:
     * 当KV不连续时，使用ListTensorDesc获取实际数据指针
     */
    __aicore__ inline void GetKvByTensorList(RunInfo<isInfer> &runInfo, const ConstInfo<isInfer, hasRope> &constInfo,
        GlobalTensor<INPUT_T> &keyValueGm, GlobalTensor<INPUT_T> &tempKeyValueGm);
    
    /**
     * @brief 获取Key GM张量
     * 
     * @return Key GlobalTensor
     */
    __aicore__ inline GlobalTensor<INPUT_T> GetKeyGm(RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo);
    
    /**
     * @brief 获取Value GM张量
     * 
     * @return Value GlobalTensor
     */
    __aicore__ inline GlobalTensor<INPUT_T> GetValueGm(RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo);
    
    /**
     * @brief MLA Fullquant BMM1核心实现
     * 
     * 完整的MLA BMM1计算:
     * 1. 加载QNope和QRoPE到L1
     * 2. 加载KNope和KRoPE到L1
     * 3. 执行QNope × KNope^T → ScoreNope
     * 4. 执行QRoPE × KRoPE^T → ScoreRoPE (累加到ScoreNope)
     * 5. Fixpipe到UB
     */
    __aicore__ inline void IterateBmm1MLAFullQuant(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf,
        RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo);

    /* =================Bmm2相关================= */
    
    /**
     * @brief MLA Fullquant BMM2核心实现
     * 
     * P × V计算:
     * 1. 从L1获取P (Softmax结果)
     * 2. 复用K的Buffer获取V
     * 3. 执行MatmulN
     * 4. Fixpipe到UB或GM
     */
    __aicore__ inline void IterateBmm2MLAFullQuant(mm2ResPos &outputBuf,
        BuffersPolicy3buff<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &inputBuf, RunInfo<isInfer> &runInfo,
        ConstInfo<isInfer, hasRope> &constInfo);
    
    /**
     * @brief 判断是否GS1合轴
     * 
     * 用于PFA (Partial Flash Attention)优化
     * 条件: BSNGD/TNGD格式 + isPfaGS1Merge
     */
    __aicore__ inline bool IsGS1Merge(ConstInfo<isInfer, hasRope> &constInfo);
}

/* ============================================================================
 * Cube Block虚拟基类 - 用于类型特化
 * ============================================================================
 * 
 * FABlockCubeMlaFullquantDummy是一个空实现类，用于:
 * 1. 模板特化: 某些配置下可能不需要MLA Fullquant
 * 2. 类型统一: 提供统一的接口定义
 * 3. Traits提取: 用于生成编译期类型信息
 * 
 * 所有方法都是空实现或返回默认值
 */
TEMPLATES_DEF
class FABlockCubeMlaFullquantDummy {
public:
    static constexpr bool isFp8 = FABlockCubeMlaFullquant<TEMPLATE_ARGS>::isFp8;
    static constexpr bool useDn = FABlockCubeMlaFullquant<TEMPLATE_ARGS>::useDn;
    static constexpr bool useNz = FABlockCubeMlaFullquant<TEMPLATE_ARGS>::useNz;
    static constexpr TPosition bmm2OutPos = FABlockCubeMlaFullquant<TEMPLATE_ARGS>::bmm2OutPos;
    static constexpr bool bmm2Write2Ub = FABlockCubeMlaFullquant<TEMPLATE_ARGS>::bmm2Write2Ub;
    __aicore__ inline FABlockCubeMlaFullquantDummy() {};
    __aicore__ inline void InitCubeBlock(TPipe *pipe, BufferManager<BufferType::L1> *l1BufferManagerPtr,
        __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *blockTable, 
        __gm__ uint8_t *queryRope, __gm__ uint8_t *keyRope) {}
    __aicore__ inline void InitCubeInput(__gm__ uint8_t *key, __gm__ uint8_t *value,
                                         CVSharedParams<isInfer, isPa> *sharedParams, AttenMaskInfo *attenMaskInfo,
                                         __gm__ int64_t *actualSeqQlenAddr, __gm__ int64_t *actualSeqKvlenAddr,
                                         __gm__ uint8_t *keySharedPrefix, __gm__ uint8_t *valueSharedPrefix,
                                         __gm__ uint8_t *actualSharedPrefixLen) {}
    __aicore__ inline void IterateBmm1(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf,
        RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo) {}
    __aicore__ inline void IterateBmm1Nz(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf,
        RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo) {}

    using mm2ResPos = typename std::conditional<bmm2Write2Ub, Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>,
        Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_FORWARD>>::type;
    __aicore__ inline void IterateBmm2(mm2ResPos &outputBuf,
        BuffersPolicy3buff<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &inputBuf,RunInfo<isInfer> &runInfo,
        ConstInfo<isInfer, hasRope> &constInfo) {}
    __aicore__ inline void InitDequantParams(__gm__ uint8_t *deqScaleQ,
        __gm__ uint8_t *deqScaleK, __gm__ uint8_t *deqScaleV) {}
};

namespace BaseApi {

/* ============================================================================
 * Cube Block Traits系统
 * ============================================================================
 * 
 * Traits系统用于在编译期提取和传递CubeBlock的模板参数。
 * 这允许Kernel层无需了解CubeBlock的具体实现细节，
 * 只需通过Traits获取所需信息。
 * 
 * ┌─────────────────────────────────────────────────────────────────────────┐
 * │                         Traits生成流程                                  │
 * ├─────────────────────────────────────────────────────────────────────────┤
 * │ DEFINE_CUBE_BLOCK_TRAITS(FABlockCubeMlaFullquant)                     │
 * │   ↓                                                                     │
 * │ 生成CubeBlockTraits<FABlockCubeMlaFullquant<TEMPLATE_ARGS>>           │
 * │   ↓                                                                     │
 * │ 通过宏展开:                                                             │
 * │   - CUBE_BLOCK_TRAITS_TYPE_FIELDS: 提取类型别名                        │
 * │   - CUBE_BLOCK_TRAITS_CONST_FIELDS: 提取常量值                        │
 * └─────────────────────────────────────────────────────────────────────────┘
 * 
 * ┌─────────────────────────────────────────────────────────────────────────┐
 * │                         ARGS_TRAITS使用                                 │
 * ├─────────────────────────────────────────────────────────────────────────┤
 * │ 在Kernel类中使用:                                                       │
 * │   class FlashAttentionScoreKernelInferMlaFullquant                    │
 * │       : public FlashAttentionScoreKernelBaseFullquant<...> {          │
 * │       public:                                                          │
 * │           ARGS_TRAITS;  // 展开后包含所有CubeBlock参数                  │
 * │       };                                                               │
 * └─────────────────────────────────────────────────────────────────────────┘
 */

/**
 * @brief CubeBlockTraits前向声明
 */
template <typename T>
struct CubeBlockTraits;  

/**
 * @brief 生成类型Traits宏
 * 用于提取CubeBlock的类型别名
 */
#define GEN_TRAIT_TYPE(name, ...) using name##_TRAITS = name;

/**
 * @brief 生成常量Traits宏
 * 用于提取CubeBlock的编译期常量
 */
#define GEN_TRAIT_CONST(name, type, ...) static constexpr type name##Traits = name;

/**
 * @brief 定义CubeBlockTraits结构体
 * 
 * 遍历CUBE_BLOCK_TRAITS_TYPE_FIELDS生成类型别名
 * 遍历CUBE_BLOCK_TRAITS_CONST_FIELDS生成常量值
 * 
 * @param CUBE_BLOCK_CLASS CubeBlock类名
 */
#define DEFINE_CUBE_BLOCK_TRAITS(CUBE_BLOCK_CLASS) \
    TEMPLATES_DEF_NO_DEFAULT \
    struct CubeBlockTraits<CUBE_BLOCK_CLASS<TEMPLATE_ARGS>> { \
        CUBE_BLOCK_TRAITS_TYPE_FIELDS(GEN_TRAIT_TYPE) \
        CUBE_BLOCK_TRAITS_CONST_FIELDS(GEN_TRAIT_CONST) \
    };

/**
 * @brief FABlockCubeMlaFullquant的Traits定义
 */
DEFINE_CUBE_BLOCK_TRAITS(FABlockCubeMlaFullquant);

/**
 * @brief FABlockCubeMlaFullquantDummy的Traits定义
 * 用于不需要实际计算的场景
 */
DEFINE_CUBE_BLOCK_TRAITS(FABlockCubeMlaFullquantDummy);

// /* 生成Arg Traits, kernel中只需要调用ARGS_TRAITS就可以获取所有CubeBlock中的模板参数 */

/**
 * @brief 从CubeBlockTraits生成参数类型别名
 */
#define GEN_ARGS_TYPE(name, ...) using name = typename CubeBlockTraits<CubeBlockType>::name##_TRAITS;

/**
 * @brief 从CubeBlockTraits生成参数常量
 */
#define GEN_ARGS_CONST(name, type, ...) static constexpr type name = CubeBlockTraits<CubeBlockType>::name##Traits;

/**
 * @brief ARGS_TRAITS宏
 * 
 * 在Kernel类中使用此宏，可以自动获取CubeBlock中的所有模板参数。
 * 展开后包含:
 * - 类型别名: INPUT_T, OUTPUT_T, T, ROPE_T等
 * - 常量值: isFp8, isMlaFullQuant, s1BaseSize等
 */
#define ARGS_TRAITS \
    CUBE_BLOCK_TRAITS_TYPE_FIELDS(GEN_ARGS_TYPE)\
    CUBE_BLOCK_TRAITS_CONST_FIELDS(GEN_ARGS_CONST)
}
#endif // FLASH_ATTENTION_SCORE_BLOCK_CUBE_MLA_FULLQUANT_H_
