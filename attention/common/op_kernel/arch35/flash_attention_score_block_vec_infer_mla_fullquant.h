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
 * \file flash_attention_score_block_vec_infer_mla_fullquant.h
 * \brief MLA Fullquant Flash Attention Vec层实现 (推理版本)
 * 
 * 功能概述:
 * 本文件实现MLA (Multi-head Latent Attention) 全量化Flash Attention的Vec层计算逻辑。
 * 主要负责Softmax计算、BMM2结果处理、Post-Quantization以及Flash Decode支持。
 * 
 * 架构层级:
 * ┌─────────────────────────────────────────────────────┐
 * │  Kernel (flash_attention_score_kernel_infer_mla_...) │
 * ├─────────────────────────────────────────────────────┤
 * │  CubeBlock (flash_attention_score_block_cube_mla_...) │
 * │  - Q×K^T (BMM1) Matmul                              │
 * │  - P×V (BMM2) Matmul                                │
 * ├─────────────────────────────────────────────────────┤
 * │  VecBlock (本文件 - flash_attention_score_block_vec_)│
 * │  - Vec1: Softmax计算 (exp/sum/max)                  │
 * │  - Vec2: Post-Quantization (输出量化)               │
 * └─────────────────────────────────────────────────────┘
 * 
 * 核心数据流:
 * 1. BMM1输出(Q×K^T) → Softmax计算 → BMM2输入
 * 2. BMM2输出(P×V) → Post-Quantization → 最终输出
 * 
 * 关键特性:
 * - Flash Decode (FD): 支持变长序列解码场景
 * - Post-Quantization: 支持Tensor级和Channel级量化
 * - Split-KV: 支持KV分片以提高内存效率
 * - Learnable Sink: 支持可学习的注意力偏置
 * - Softmax LSE: 计算log-sum-exp用于attention权重归一化
 * 
 * 模板参数:
 * - INPUT_T: 输入数据类型 (fp8_e5m2_t, fp8_e4m3fn_t, hifloat8_t, half, bfloat16_t, float)
 * - OUTPUT_T: 输出数据类型 (支持多种量化类型)
 * - T: 计算数据类型 (通常为float或bfloat16_t)
 * - isInfer: 是否为推理模式
 * - isFd: 是否支持Flash Decode (变长序列)
 * - hasRope: 是否包含RoPE (Rotary Position Embedding)
 * - isPa: 是否支持Page Attention
 * - hasAtten: 是否包含attention mask
 * - hasDrop: 是否包含dropout
 * - enableKVPrefix: 是否支持KV Prefix
 * - layout: 数据布局 (BNSD, BSHD, TND, NTD等)
 * - hasSoftmax: 是否执行softmax计算
 * - hasSink: 是否包含learnable sink
 * - pseMode: Pse (Position-Sensitive Encoding) 模式
 */
#ifndef FLASH_ATTENTION_SCORE_BLOCK_VEC_INFER_MLA_FULLQUANT_H_
#define FLASH_ATTENTION_SCORE_BLOCK_VEC_INFER_MLA_FULLQUANT_H_
#include "flash_attention_score_block_vec_base_fullquant.h"
#include "util_regbase.h"
#include "flash_attention_score_common_regbase.h"
#include "kernel_operator_list_tensor_intf.h"
#include "vf/vf_flash_decode.h"
#include "vf/vf_post_quant.h"
using namespace AscendC;
using namespace AscendC::Impl::Detail;
using namespace regbaseutil;

/**
 * @brief MLA Fullquant Flash Attention Vec层命名空间
 * 
 * 包含Vec层的所有实现逻辑，负责:
 * - Softmax计算 (Vec1)
 * - BMM2结果后处理
 * - Post-Quantization (Vec2)
 * - Flash Decode多批次合并
 */
namespace BaseApi {
TEMPLATES_DEF

/**
 * @brief MLA Fullquant Vec层主类 (推理版本)
 * 
 * 继承自FABlockVecBaseFullquant，添加了:
 * - Flash Decode (FD) 支持
 * - Post-Quantization 支持
 * - Split-KV 合并逻辑
 * - Softmax LSE 计算和输出
 * 
 * @tparam TEMPLATE_ARGS 模板参数包
 */
class FABlockVecInferMlaFullquant
    : public FABlockVecBaseFullquant<FABlockVecInferMlaFullquant<TEMPLATE_ARGS>, TEMPLATE_ARGS> {
public:
    using BaseClass = FABlockVecBaseFullquant<FABlockVecInferMlaFullquant<TEMPLATE_ARGS>, TEMPLATE_ARGS>;

public:
    /* ================编译期常量信息======================= */
    
    /**
     * @brief UB缓冲区大小 (32KB)
     * 用于Flash Decode场景下的中间数据存储
     */
    static constexpr uint32_t bufferSizeByte32K = 32768;
    
    /**
     * @brief G维度最大分片数
     * 用于控制Split-KV合并时的内存使用
     */
    static constexpr uint32_t gSplitMax = 16;
    
    /**
     * @brief 预加载次数
     * Flash Decode场景下的pipeline深度
     */
    static constexpr uint32_t preloadTimes = 3;
    
    /**
     * @brief 是否执行Post-Quantization
     * 当输出类型不是half/bfloat16_t/float时启用
     * 用于将计算结果量化到目标数据类型
     */
    static constexpr bool POST_QUANT = !IsSameType<OUTPUT_T, half>::value && !IsSameType<OUTPUT_T, bfloat16_t>::value &&
                                       !IsSameType<OUTPUT_T, float>::value;
    
    /**
     * @brief 是否使用FP8输入
     * FP8输入会触发MLA Fullquant特殊处理路径
     */
    static constexpr bool isFp8 = IsSameType<INPUT_T, fp8_e5m2_t>::value ||
                                   IsSameType<INPUT_T, fp8_e4m3fn_t>::value ||
                                   IsSameType<INPUT_T, hifloat8_t>::value;
    
    /**
     * @brief 是否启用MLA Fullquant模式
     * MLA Fullquant = FP8输入 + RoPE
     * MLA将K/V分解为: K = Nope + RoPE
     */
    static constexpr bool isMlaFullQuant = isFp8 && hasRope;

    /* =====================GM变量========================== */
    
    /**
     * @brief Softmax Log-Sum-Exp全局张量
     * 存储每个attention头的log-sum-exp值
     * 用于输出层的归一化和梯度计算
     */
    GlobalTensor<float> softmaxLseGm;

    /**
     * @brief Flash Decode累加输出GM张量
     * 存储多批次解码的BMM2中间结果
     * 仅在Flash Decode (isFd) 模式下使用
     */
    using FDGmType = typename std::conditional<isFd, GlobalTensor<float>, int8_t>::type;
    FDGmType accumOutGm;
    
    /**
     * @brief Flash Decode Softmax Max GM张量
     * 存储各批次的softmax max值，用于后续合并
     */
    FDGmType softmaxFDMaxGm;
    
    /**
     * @brief Flash Decode Softmax Sum GM张量
     * 存储各批次的softmax sum值，用于后续合并
     */
    FDGmType softmaxFDSumGm;

    /**
     * @brief Post-Quantization Scale GM张量
     * 量化参数：每个channel的缩放因子
     */
    using postQuantGmType = typename std::conditional<POST_QUANT, GlobalTensor<float>, int8_t>::type;
    postQuantGmType postQuantScaleGm;
    
    /**
     * @brief Post-Quantization Offset GM张量
     * 量化参数：每个channel的偏置值
     */
    postQuantGmType postQuantOffsetGm;
    
    /**
     * @brief Post-Quantization Scale (BF16) GM张量
     * 当使用BF16存储量化参数时使用
     */
    using postQuantBf16GmType = typename std::conditional<POST_QUANT, GlobalTensor<bfloat16_t>, int8_t>::type;
    postQuantBf16GmType postQuantScaleBf16Gm;
    
    /**
     * @brief Post-Quantization Offset (BF16) GM张量
     * 当使用BF16存储量化参数时使用
     */
    postQuantBf16GmType postQuantOffsetBf16Gm;

    /* =====================UB变量========================== */
    
    /**
     * @brief LSE临时缓冲区
     * 用于Flash Decode场景下的中间计算
     */
    TBuf<> lseTmpBuff;
    
    /**
     * @brief Softmax LSE输出队列
     * 用于Vec1层的LSE计算结果输出
     */
    TQue<QuePosition::VECOUT, 1> softmaxLseQueue;
    
    /**
     * @brief Flash Decode结果输出队列
     * 存储Post-Quantization后的中间结果
     */
    TQue<QuePosition::VECOUT, 1> FDResOutputQue;
    
    /**
     * @brief 累加输出输入队列
     * 用于接收需要合并的BMM2结果
     */
    TQue<QuePosition::VECIN, 1> accumOutInputQue;
    
    /**
     * @brief Softmax Max输入队列
     * 存储Flash Decode各批次的max值
     */
    TQue<QuePosition::VECIN, 1> softmaxMaxInputQue;
    
    /**
     * @brief Softmax Sum输入队列
     * 存储Flash Decode各批次的sum值
     */
    TQue<QuePosition::VECIN, 1> softmaxSumInputQue;
    
    /**
     * @brief Post-Quantization Scale队列
     * 存储量化参数用于后续计算
     */
    TQue<QuePosition::VECIN, 1> postQuantScaleQue;
    
    /**
     @brief Post-Quantization Offset队列
     * 存储量化偏置用于后续计算
     */
    TQue<QuePosition::VECIN, 1> postQuantOffsetQue;
    
    /**
     * @brief Attention Sink输入队列
     * 用于learnable sink机制的数据传输
     */
    TQue<QuePosition::VECIN, 1> sinkQue;

    /* =====================公开接口======================== */

    __aicore__ inline FABlockVecInferMlaFullquant() {};

    /**
     * @brief 初始化Cube和Vec共享参数
     * 
     * 从tiling数据中提取共享参数，包括:
     * - Batch/Head维度信息
     * - Sequence长度信息
     * - 多核切分偏移
     * - 特殊模式标志 (GQA, PageAttention等)
     * 
     * @param sharedParams 输出参数结构体
     * @param aicIdx 核索引
     * @param subBlockIdx 子块索引
     */
    __aicore__ inline void InitCubeVecSharedParams(CVSharedParams<isInfer, isPa> &sharedParams, int32_t aicIdx, uint8_t subBlockIdx);
    
    /**
     * @brief 清理和初始化输出缓冲区
     * 
     * 初始化attention输出和softmax LSE输出缓冲区
     * 仅在AIV核上执行，且需要needInit标志
     * 
     * @param softmaxLse LSE输出缓冲区指针
     * @param attentionOut Attention输出缓冲区指针
     * @param constInfo 常量信息
     */
    __aicore__ inline void CleanOutput(__gm__ uint8_t *softmaxLse, __gm__ uint8_t *attentionOut,
        ConstInfo<isInfer, hasRope> &constInfo);
    
    /**
     * @brief 初始化Dropout相关缓冲区
     * MLA Fullquant版本不包含Dropout，此处为空实现
     */
    __aicore__ inline void InitDropOut(__gm__ uint8_t *dropMask, __gm__ uint8_t *workspace) {}
    
    /**
     * @brief 初始化全局缓冲区
     * 
     * 设置所有GM张量的全局缓冲区指针，包括:
     * - Q/K/V的量化参数
     * - Post-Quantization参数
     * - Prefix/KV相关缓冲区
     * 
     * @param pse Pse缓冲区
     * @param deqScaleQ/K/V Q/K/V的反量化参数
     * @param postQuantScale/Offset 量化参数
     * @param prefix KV Prefix缓冲区
     * @param attenMask Attention Mask缓冲区
     * @param workspace 工作空间
     * @param singleCoreOffset 单核偏移量
     * @param aicIdx 核索引
     * @param constInfo 常量信息
     */
    __aicore__ inline void InitGlobalBuffer(
        __gm__ uint8_t *pse, __gm__ uint8_t *deqScaleQ, __gm__ uint8_t *deqScaleK, __gm__ uint8_t *deqScaleV,
        __gm__ uint8_t *pScale, __gm__ uint8_t *postQuantScale, __gm__ uint8_t *postQuantOffset,
        __gm__ uint8_t *prefix, __gm__ uint8_t *attenMask,
        __gm__ uint8_t *queryPaddingSize, __gm__ uint8_t *kvPaddingSize, __gm__ uint8_t *learnableSink, __gm__ uint8_t *softmaxMax,
        __gm__ uint8_t *softmaxSum, __gm__ uint8_t *&workspace, uint64_t singleCoreOffset, uint32_t aicIdx,
        ConstInfo<isInfer, hasRope> &constInfo);
    
    /**
     * @brief 初始化Vec层独有的局部缓冲区
     * 
     * 分配UB上的队列缓冲区，包括:
     * - Softmax LSE队列
     * - Post-Quantization参数队列
     * - Query Scale缓冲区 (MLA Fullquant)
     * - Sink队列 (learnable sink机制)
     */
    __aicore__ inline void InitUniqueLocalBuffer(ConstInfo<isInfer, hasRope> &constInfo);
    
    /**
     * @brief 初始化Post-Quantization
     * 
     * 设置量化参数的GM张量，支持:
     * - Tensor级量化 (单一scale/offset)
     * - Channel级量化 (每通道独立参数)
     * - 多种存储精度 (float/bfloat16_t)
     */
    __aicore__ inline void InitPostQuant(ConstInfo<isInfer, hasRope> &constInfo, __gm__ uint8_t *postQuantScale, __gm__ uint8_t *postQuantOffset);
    
    /**
     * @brief 生成Dropout Mask
     * MLA Fullquant版本不包含Dropout，此处为空实现
     */
    __aicore__ inline void GenerateDropoutMask(RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo, LocalTensor<uint8_t> &dropMaskUb) {}
    
    /**
     * @brief Softmax数据拷贝输出
     * 
     * Vec1的核心函数，执行softmax计算并输出结果:
     * 1. 调用ComputeLogSumExpAndCopyToGm (Flash Decode场景)
     * 2. 执行Sink计算 (learnable sink机制)
     * 3. 拷贝Softmax LSE到GM
     * 
     * @param runInfo 运行信息
     * @param constInfo 常量信息
     * @param sumUb Softmax Sum UB张量
     * @param maxUb Softmax Max UB张量
     */
    __aicore__ inline void SoftmaxDataCopyOut(RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo, LocalTensor<float> &sumUb,
                                              LocalTensor<float> &maxUb);
    
    /**
     * @brief Softmax数据拷贝输出 (FP8版本)
     * MLA Fullquant的FP8输入场景使用
     */
    __aicore__ inline void SoftmaxDataCopyOutFp8(RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo,
                                                 LocalTensor<half> &sumUb, LocalTensor<half> &maxUb) {}
    
    /**
     * @brief 拷贝Attention输出
     * 
     * Vec2层的核心函数，处理BMM2结果:
     * - Flash Decode场景: 调用Bmm2FDOut进行中间结果存储
     * - 非FD场景: 调用Bmm2DataCopyOut直接输出
     * 
     * @param runInfo 运行信息
     * @param constInfo 常量信息
     * @param vec2ResUb BMM2结果UB张量
     * @param vec2S1Idx S1维度索引
     * @param vec2CalcSize 计算大小
     */
    template <typename VEC2_RES_T>
    __aicore__ inline void CopyOutAttentionOut(RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo, LocalTensor<VEC2_RES_T> &vec2ResUb,
                                               int64_t vec2S1Idx, int64_t vec2CalcSize);
    
    /**
     * @brief 初始化Flash Decode缓冲区
     * 
     * 为Flash Decode场景分配32KB UB缓冲区:
     * - LSE临时缓冲区
     * - Softmax Max/Sum输入队列
     * - 累加输出队列
     * - Post-Quantization队列
     */
    __aicore__ inline void InitFDBuffers(ConstInfo<isInfer, hasRope> &constInfo);
    
    /**
     * @brief Flash Decode主计算函数
     * 
     * 处理变长序列解码场景:
     * 1. 获取当前batch的实际序列长度
     * 2. 计算循环次数
     * 3. 调用CombineSplitKVRes合并多批次结果
     * 
     * @param constInfo 常量信息
     * @param keyGm Key全局张量
     * @param actualSeqKvlenAddr 实际KV长度地址
     */
    __aicore__ inline void FlashDecodeCompute(ConstInfo<isInfer, hasRope> &constInfo, GlobalTensor<INPUT_T> &keyGm, __gm__ int64_t *actualSeqKvlenAddr);

    /**
     * @brief Post-Quantization处理 (通用版本)
     * 
     * 对BMM2结果执行量化操作:
     * - Tensor级量化: 使用单一scale/offset
     * - Channel级量化: 使用per-channel参数
     * 
     * @param constInfo 常量信息
     * @param runInfo 运行信息
     * @param attenOut 输出张量
     * @param vec2ResUb BMM2结果
     * @param vec2S1Idx S1索引
     * @param dSizeAligned64 对齐后的维度
     */
    template <typename VEC2_RES_T>
    __aicore__ inline void PostQuant(ConstInfo<isInfer, hasRope> &constInfo, RunInfo<isInfer> &runInfo, LocalTensor<OUTPUT_T> &attenOut, LocalTensor<VEC2_RES_T> &vec2ResUb, int64_t vec2S1Idx, int64_t dSizeAligned64);

    /**
     * @brief Flash Decode Post-Quantization
     * 
     * 针对Flash Decode场景的量化处理
     * 
     * @param constInfo 常量信息
     * @param attenOut 输出张量
     * @param accumOutLocal 累加结果
     * @param perChannelQuantOffset 通道量化偏移
     * @param dealRowCount 处理行数
     * @param dSizeAligned64 对齐后的维度
     */
    __aicore__ inline void FDPostQuant(ConstInfo<isInfer, hasRope> &constInfo, LocalTensor<OUTPUT_T> &attenOut, LocalTensor<T> &accumOutLocal, uint64_t perChannelQuantOffset, uint32_t dealRowCount, uint32_t dSizeAligned64);

    /**
     * @brief Channel级Post-Quantization
     * 
     * 支持per-channel量化参数的量化实现:
     * 1. 从GM拷贝scale/offset到UB
     * 2. 调用PostQuantPerChnlImpl执行量化
     * 
     * @tparam POSTQUANT_PARAMS_T 量化参数类型
     * @tparam VEC2_RES_T BMM2结果类型
     */
    template <typename POSTQUANT_PARAMS_T, typename VEC2_RES_T>
    __aicore__ inline void PostQuantPerChnl(ConstInfo<isInfer, hasRope> &constInfo, LocalTensor<OUTPUT_T> &attenOut,
    LocalTensor<VEC2_RES_T> &vec2ResUb, uint64_t perChannelQuantOffset, uint32_t gSplitSize, uint32_t s1RowCount, uint32_t splitOffset, int64_t dSizeAligned64,
    GlobalTensor<POSTQUANT_PARAMS_T> postQuantScaleGm, GlobalTensor<POSTQUANT_PARAMS_T> postQuantOffsetGm);

private:
    /* =====================私有辅助函数======================== */
    
    /**
     * @brief 初始化单核输出
     * 
     * 将attention输出缓冲区初始化为0
     * 支持POST_QUANT模式的half初始化
     */
    __aicore__ inline void InitOutputSingleCore(ConstInfo<isInfer, hasRope> &constInfo);
    
    /**
     * @brief 初始化LSE单核输出
     * 
     * 将softmax LSE初始化为3e+99 (极大值，表示无效)
     * 按核数平均分配LSE空间
     */
    __aicore__ inline void InitLseOutputSingleCore(ConstInfo<isInfer, hasRope> &constInfo);
    
    /**
     * @brief 获取KV的实际序列长度
     * 
     * 处理变长序列场景:
     * 1. 检查KV是否连续存储
     * 2. 处理padding情况
     * 3. 从actualSeqKvlenAddr获取实际长度
     * 
     * @param constInfo 常量信息
     * @param keyGm Key GM张量
     * @param actualSeqKvlenAddr 实际长度地址
     * @param boIdx Batch索引
     * @param actualSeqLen 输出：实际长度
     */
    __aicore__ inline void GetActualSeqLenKV(ConstInfo<isInfer, hasRope> &constInfo, GlobalTensor<INPUT_T> &keyGm, 
        __gm__ int64_t *actualSeqKvlenAddr, int64_t boIdx, int64_t &actualSeqLen);
    
    /**
     * @brief 拷贝Softmax LSE到GM
     * 
     * 计算log-sum-exp并写入GM:
     * LSE = log(sum(exp(x - max)))
     * 
     * @param softmaxSumTmp Softmax Sum UB张量
     * @param softmaxMaxTmp Softmax Max UB张量
     * @param runInfo 运行信息
     * @param constInfo 常量信息
     */
    __aicore__ inline void SoftmaxLseCopyOut(LocalTensor<float> &softmaxSumTmp, LocalTensor<float> &softmaxMaxTmp,
                                             RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo);
    
    /**
     * @brief 合并Split-KV结果
     * 
     * Flash Decode核心函数:
     * 1. 计算G维度分片大小 (32KB UB限制)
     * 2. 按分片循环处理非尾块和尾块
     * 3. 每次分片: 拷贝LSE → 计算Scale → 合并结果 → 写出
     * 
     * @param constInfo 常量信息
     * @param attenOutOffset 输出偏移
     * @param bIdx Batch索引
     * @param n2Idx Head索引
     */
    __aicore__ inline void CombineSplitKVRes(ConstInfo<isInfer, hasRope> &constInfo, uint64_t attenOutOffset, uint32_t bIdx, uint32_t n2Idx);

    /**
     * @brief 计算Scale值
     * 
     * 计算softmax的归一化因子用于结果合并:
     * - 拷贝Sink值到UB (可选)
     * - 调用ComputeScaleValue_VF计算
     * - 拷贝LSE到GM (如果启用)
     * 
     * @param lseMaxUb LSE Max UB张量
     * @param lseSumUb LSE Sum UB张量
     * @param constInfo 常量信息
     * @param splitSize 分片大小
     * @param lseOffset LSE输出偏移
     * @param sinkOffset Sink数据偏移
     */
    __aicore__ inline void ComputeScaleValue(LocalTensor<T> lseMaxUb, LocalTensor<T> lseSumUb, 
        ConstInfo<isInfer, hasRope> &constInfo, uint32_t splitSize, uint64_t lseOffset, uint64_t sinkOffset);

    /**
     * @brief BMM2 Flash Decode输出
     * 
     * 将BMM2结果存储到累加缓冲区:
     * 1. 等待V向量计算完成
     * 2. 使用DataCopyPad将结果写入accumOutGm
     * 
     * @param vec2ResUb BMM2结果UB张量
     * @param runInfo 运行信息
     * @param constInfo 常量信息
     * @param vec2S1Idx S1索引
     * @param vec2CalcSize 计算大小
     */
    __aicore__ inline void Bmm2FDOut(LocalTensor<T> &vec2ResUb, RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo,
                                     int64_t vec2S1Idx, int64_t vec2CalcSize);

    /**
     * @brief 拷贝LSE到输入队列
     * 
     * 从GM拷贝softmax max/sum到UB输入队列
     * 用于后续的合并计算
     * 
     * @param constInfo 常量信息
     * @param bIdx Batch索引
     * @param n2Idx Head索引
     * @param startRow 起始行
     * @param dealRowCount 处理行数
     */
    __aicore__ inline void CopyLseIn(ConstInfo<isInfer, hasRope> &constInfo, uint32_t bIdx, uint32_t n2Idx, uint32_t startRow, uint32_t dealRowCount);

    /**
     * @brief 拷贝最终结果输出
     * 
     * 将合并后的结果写出到GM:
     * 1. 分配临时UB张量
     * 2. 执行Post-Quantization (可选)
     * 3. 调用ReduceFDDataCopyOut写入GM
     * 
     * @param constInfo 常量信息
     * @param attenOutOffset 输出偏移
     * @param accumOutLocal 累加结果
     * @param startRow 起始行
     * @param dealRowCount 处理行数
     * @param perChannelQuantOffset 通道量化偏移
     */
    __aicore__ inline void CopyFinalResOut(ConstInfo<isInfer, hasRope> &constInfo, uint64_t attenOutOffset, LocalTensor<T> &accumOutLocal, uint32_t startRow,
                                           uint32_t dealRowCount, uint64_t perChannelQuantOffset);

    /**
     * @brief 拷贝累加输出到输入队列
     * 
     * 从accumOutGm拷贝数据到UB输入队列
     * 准备进行Split-KV合并
     * 
     * @param constInfo 常量信息
     * @param bIdx Batch索引
     * @param n2Idx Head索引
     * @param splitKVIndex Split-KV索引
     * @param startRow 起始行
     * @param dealRowCount 处理行数
     */
    __aicore__ inline void CopyAccumOutIn(ConstInfo<isInfer, hasRope> &constInfo, uint32_t bIdx, uint32_t n2Idx, uint32_t splitKVIndex, uint32_t startRow,
                                          uint32_t dealRowCount);

    /**
     * @brief 计算LogSumExp并拷贝到GM
     * 
     * Flash Decode场景下的softmax后处理:
     * 1. 计算offset参数
     * 2. 调用BroadCastAndCopyOut进行广播和拷贝
     * 
     * @param runInfo 运行信息
     * @param constInfo 常量信息
     */
    __aicore__ inline void ComputeLogSumExpAndCopyToGm(RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo);

    /**
     * @brief 拷贝Sink输入
     * 
     * 将learnable sink数据从GM拷贝到UB队列
     * 
     * @param runInfo 运行信息
     * @param constInfo 常量信息
     */
    __aicore__ inline void CopySinkIn(RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo);

    /**
     * @brief 拷贝Sink输入 (Flash Decode版本)
     * 
     * @param splitSize 分片大小
     * @param sinkOffset Sink偏移
     */
    __aicore__ inline void CopySinkFDIn(uint32_t splitSize, uint64_t sinkOffset);

    /**
     * @brief Vec1 Sink计算
     * 
     * 执行learnable sink的计算:
     * - 获取sink值
     * - 调用SinkSubExpAddVF进行计算
     * 
     * @param runInfo 运行信息
     * @param constInfo 常量信息
     * @param sumUb Softmax Sum UB张量
     * @param maxUb Softmax Max UB张量
     */
    __aicore__ inline void Vec1SinkCompute(RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo, LocalTensor<float> &sumUb, LocalTensor<float> &maxUb);

    /**
     * @brief Vec1 Sink计算 (GQA Fused版本)
     * 
     * 针对GQA优化的sink计算
     * 
     * @param runInfo 运行信息
     * @param constInfo 常量信息
     * @param sumUb Softmax Sum UB张量
     * @param maxUb Softmax Max UB张量
     */
    __aicore__ inline void Vec1SinkComputeGSFused(RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo, LocalTensor<float> &sumUb, LocalTensor<float> &maxUb);

    /**
     * @brief 合并最终结果
     * 
     * 将多个Split-KV分片的结果合并:
     * 1. 循环actualCombineLoopSize次
     * 2. 每次: 拷贝累加输出 → 调用ReduceFinalRes_VF合并
     * 
     * @param constInfo 常量信息
     * @param bIdx Batch索引
     * @param n2Idx Head索引
     * @param dst 目标张量
     * @param lseLocal LSE本地张量
     * @param startRow 起始行
     * @param dealRowCount 处理行数
     */
    __aicore__ inline void ReduceFinalRes(ConstInfo<isInfer, hasRope> &constInfo, uint32_t bIdx, uint32_t n2Idx, LocalTensor<T> &dst, LocalTensor<T> &lseLocal,
                                          uint32_t startRow, uint32_t dealRowCount);

    /**
     * @brief 合并FD数据拷贝输出
     * 
     * 将合并后的数据写入GM
     * 
     * @param constInfo 常量信息
     * @param attenOutOffset 输出偏移
     * @param attenOutUb 输出UB张量
     * @param startRow 起始行
     * @param dealRowCount 处理行数
     * @param columnCount 列数
     * @param actualColumnCount 实际列数
     */
    __aicore__ inline void ReduceFDDataCopyOut(ConstInfo<isInfer, hasRope> &constInfo, uint64_t attenOutOffset, LocalTensor<OUTPUT_T> &attenOutUb,
                                               uint32_t startRow, uint32_t dealRowCount, uint32_t columnCount,
                                               uint32_t actualColumnCount);

};

/* ==================== InitCubeVecSharedParams 实现 ==================== */
/**
 * @brief 初始化Cube和Vec共享参数
 * 
 * 从tiling数据中提取所有共享参数，包括:
 * - 维度大小 (bSize, s1Size, s2Size, dSize等)
 * - 多核切分信息 (coreNum, bnStartIdx等)
 * - 特殊模式标志 (GQA, PageAttention等)
 * - 序列长度信息
 * 
 * 在AIV核上会将参数拷贝到SSBUF以供AIC核使用
 */
TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockVecInferMlaFullquant<TEMPLATE_ARGS>::InitCubeVecSharedParams(
    CVSharedParams<isInfer, isPa> &sharedParams, int32_t aicIdx, uint8_t subBlockIdx)
{
    auto &inputParamsRegbase = this->tilingData->inputParamsRegbase;
    sharedParams.bSize = inputParamsRegbase.bSize;
    sharedParams.t1Size = inputParamsRegbase.t1Size;
    sharedParams.t2Size = inputParamsRegbase.t2Size;
    sharedParams.n2Size = inputParamsRegbase.n2Size;
    sharedParams.gSize = inputParamsRegbase.gSize;
    sharedParams.s1Size = inputParamsRegbase.s1Size;
    sharedParams.s2Size = inputParamsRegbase.s2Size;
    sharedParams.dSize = inputParamsRegbase.dSize;
    sharedParams.dSizeV = inputParamsRegbase.d.dSizeV;
    if constexpr (hasRope) {
        sharedParams.dSizeRope = inputParamsRegbase.dSizeRope;
    } else {
        sharedParams.dSizeRope = 0;
    }
    sharedParams.preTokens = inputParamsRegbase.preTokens;
    sharedParams.nextTokens = inputParamsRegbase.nextTokens;
    sharedParams.s1SparseValidSize = inputParamsRegbase.s1SparseValidSize;
    sharedParams.s2SparseValidSize = inputParamsRegbase.s2SparseValidSize;
    sharedParams.bandIndex = inputParamsRegbase.bandIndex;
    sharedParams.implMode = inputParamsRegbase.implMode;
    sharedParams.layoutType = inputParamsRegbase.layoutType;
    sharedParams.sparseType = inputParamsRegbase.sparseType;
    sharedParams.compressMode = inputParamsRegbase.attenMaskCompressMode;
    sharedParams.attenMaskS1Size = inputParamsRegbase.attenMaskS1Size;
    sharedParams.attenMaskS2Size = inputParamsRegbase.attenMaskS2Size;
 
    if constexpr (isFd) {
        sharedParams.splitKVNum = inputParamsRegbase.kvSplitPart;
    }
    if constexpr (POST_QUANT) {
        sharedParams.isPostQuantPerChnl = inputParamsRegbase.isPostQuantPerChnl;
        sharedParams.isPostQuantBF16 = inputParamsRegbase.isPostQuantBF16;
    }
    sharedParams.transposeLayout = inputParamsRegbase.transposeLayout;
    sharedParams.fromFused = inputParamsRegbase.fromFused;
    sharedParams.isRowInvalid = inputParamsRegbase.isRowInvalid;
    sharedParams.headNumRatio = inputParamsRegbase.headNumRatio;
    sharedParams.isGqa = inputParamsRegbase.isGqa;
    sharedParams.isPfaGS1Merge = (inputParamsRegbase.isGqa && sharedParams.s1Size > 1);
    sharedParams.isKvContinuous = inputParamsRegbase.isKvContinuous;
    sharedParams.actualSeqLengthsSize = inputParamsRegbase.actualSeqLengthsSize;
    sharedParams.actualSeqLengthsKVSize = inputParamsRegbase.actualSeqLengthsKVSize;
    sharedParams.isActualSeqLengthsNull = inputParamsRegbase.isActualSeqLengthsNull;
    sharedParams.isActualSeqLengthsKVNull = inputParamsRegbase.isActualSeqLengthsKVNull;
    sharedParams.isQHasLeftPadding = inputParamsRegbase.isQHasLeftPadding;
    sharedParams.isKVHasLeftPadding = inputParamsRegbase.isKVHasLeftPadding;
    
    // PageAttention参数
    if constexpr (isPa) {
        sharedParams.blockTableDim2 = inputParamsRegbase.blockTableDim2;
        sharedParams.blockSize = inputParamsRegbase.blockSize;
        sharedParams.paLayoutType = inputParamsRegbase.paLayoutType;
        sharedParams.paBlockNumSum = inputParamsRegbase.paBlockNumSum;
    }
    
    // KV Prefix参数
    if constexpr (enableKVPrefix) {
        sharedParams.isActualSharedPrefixLenNull = inputParamsRegbase.isActualSharedPrefixLenNull;
        sharedParams.kvPrefixSize = inputParamsRegbase.prefixSeqInnerSize;
    }
    
    auto &multiCoreParamsRegbase = this->tilingData->multiCoreParamsRegbase;
    sharedParams.s1OuterSize = multiCoreParamsRegbase.s1OuterSize;
    sharedParams.coreNum = multiCoreParamsRegbase.coreNum;
    
    /* 多核切分偏移计算 */
    sharedParams.multiCoreInnerOffset = multiCoreParamsRegbase.sparseStartIdx[aicIdx];
    sharedParams.multiCoreInnerLimit = multiCoreParamsRegbase.sparseStartIdx[aicIdx + 1];
    sharedParams.bnStartIdx = multiCoreParamsRegbase.bnStartIdx[aicIdx];
    sharedParams.bnEndIdx = multiCoreParamsRegbase.bnStartIdx[aicIdx + 1];
    sharedParams.needInit = this->tilingData->initOutputParams.needInit;

    // AIV核: 将参数拷贝到SSBUF供AIC核使用
    if ASCEND_IS_AIV {
        if (subBlockIdx == 0) {
            auto tempTilingSSbuf = reinterpret_cast<__ssbuf__ uint32_t*>(0);
            auto tempTiling = reinterpret_cast<uint32_t *>(&sharedParams);
            #pragma unroll
            for (int i = 0; i < sizeof(CVSharedParams<isInfer, isPa>) / sizeof(uint32_t); ++i, ++tempTilingSSbuf, ++tempTiling) {
                *tempTilingSSbuf = *tempTiling;
            }
            CrossCoreSetFlag<SYNC_MODE, PIPE_S>(15);
        }
    }
}

/* ==================== CleanOutput 实现 ==================== */
/**
 * @brief 清理和初始化输出缓冲区
 * 
 * 在AIV核上执行:
 * 1. 设置attention输出GM缓冲区
 * 2. 设置softmax LSE GM缓冲区
 * 3. 如果needInit=1，初始化输出为0
 * 4. 同步后初始化LSE输出
 */
TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockVecInferMlaFullquant<TEMPLATE_ARGS>::CleanOutput(__gm__ uint8_t *softmaxLse,
    __gm__ uint8_t *attentionOut, ConstInfo<isInfer, hasRope> &constInfo) 
{
    if ASCEND_IS_AIV {
        this->attentionOutGm.SetGlobalBuffer((__gm__ OUTPUT_T *)attentionOut);
        if constexpr (POST_QUANT) {
            this->attentionOutInitGm.SetGlobalBuffer((__gm__ half *)attentionOut);
        }
        softmaxLseGm.SetGlobalBuffer((__gm__ float *)softmaxLse);
        constInfo.isSoftmaxLseEnable = this->tilingData->inputParamsRegbase.isSoftMaxLseEnable;
        if (this->tilingData->initOutputParams.needInit == 1) {
            InitOutputSingleCore(constInfo);
            // LSE输出
            if (constInfo.isSoftmaxLseEnable) {
                SyncAll();
                InitLseOutputSingleCore(constInfo);
            }
        }
    }
}

/* ==================== InitGlobalBuffer 实现 ==================== */
/**
 * @brief 初始化全局缓冲区
 * 
 * 流程:
 * 1. 调用基类初始化公共GM缓冲区
 * 2. 如果是Flash Decode模式:
 *    - 调整workspace指针
 *    - 设置accumOutGm、softmaxFDMaxGm、softmaxFDSumGm
 * 3. 如果启用Post-Quantization，初始化量化参数
 */
TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockVecInferMlaFullquant<TEMPLATE_ARGS>::InitGlobalBuffer(
    __gm__ uint8_t *pse, __gm__ uint8_t *deqScaleQ, __gm__ uint8_t *deqScaleK, __gm__ uint8_t *deqScaleV, __gm__ uint8_t *pScale,
    __gm__ uint8_t *postQuantScale, __gm__ uint8_t *postQuantOffset, __gm__ uint8_t *prefix, __gm__ uint8_t *attenMask,
    __gm__ uint8_t *queryPaddingSize, __gm__ uint8_t *kvPaddingSize, __gm__ uint8_t *learnableSink, __gm__ uint8_t *softmaxMax,
    __gm__ uint8_t *softmaxSum, __gm__ uint8_t *&workspace, uint64_t singleCoreOffset, uint32_t aicIdx,
    ConstInfo<isInfer, hasRope> &constInfo)
{
    BaseClass::InitCommonGlobalBuffer(pse, deqScaleQ, deqScaleK, deqScaleV, pScale, postQuantScale, prefix, attenMask, learnableSink, workspace, constInfo);
    if constexpr (isFd) {
        // 调整workspace回到基地址
        workspace -= singleCoreOffset * preloadTimes * (aicIdx + 1);
        auto &inputParamsRegbase = this->tilingData->inputParamsRegbase;
        int32_t actualCoreNums = inputParamsRegbase.bSize * constInfo.n2Size * constInfo.splitKVNum;
        workspace += actualCoreNums * singleCoreOffset * preloadTimes;

        uint64_t accumOutSize = this->tilingData->inputParamsRegbase.accumOutSize;
        uint64_t logSumExpSize = this->tilingData->inputParamsRegbase.logSumExpSize;
        accumOutGm.SetGlobalBuffer((__gm__ T *)(workspace));
        workspace += accumOutSize * sizeof(float);
        softmaxFDMaxGm.SetGlobalBuffer((__gm__ float *)(workspace));
        workspace += logSumExpSize * sizeof(float);
        softmaxFDSumGm.SetGlobalBuffer((__gm__ float *)(workspace));
        workspace += logSumExpSize * sizeof(float);
    }
    if constexpr (POST_QUANT) {
        this->InitPostQuant(constInfo, postQuantScale, postQuantOffset);
    }
}

/* ==================== InitPostQuant 实现 ==================== */
/**
 * @brief 初始化Post-Quantization参数
 * 
 * 支持多种配置组合:
 * - Tensor级量化 vs Channel级量化
 * - Float存储 vs BF16存储
 * 
 * 量化公式: output = round(input * scale + offset)
 */
TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockVecInferMlaFullquant<TEMPLATE_ARGS>::InitPostQuant(ConstInfo<isInfer, hasRope> &constInfo, __gm__ uint8_t *postQuantScale, __gm__ uint8_t *postQuantOffset)
{
    if constexpr (POST_QUANT) {
        constInfo.isPostQuantOffsetExist = false;
        
        // Tensor级量化 + Float存储
        if (!constInfo.isPostQuantPerChnl && !constInfo.isPostQuantBF16) {
            if (postQuantScale != nullptr) {
                postQuantScaleGm.SetGlobalBuffer((__gm__ float *)postQuantScale);
                constInfo.postQuantScaleValue = postQuantScaleGm.GetValue(0);
            }
            if (postQuantOffset != nullptr) {
                postQuantOffsetGm.SetGlobalBuffer((__gm__ float *)postQuantOffset);
                constInfo.postQuantOffsetValue = postQuantOffsetGm.GetValue(0);
            } else {
                constInfo.postQuantOffsetValue = 0.0;
            }
        }
        
        // Tensor级量化 + BF16存储
        if (!constInfo.isPostQuantPerChnl && constInfo.isPostQuantBF16) {
            if (postQuantScale != nullptr) {
                postQuantScaleBf16Gm.SetGlobalBuffer((__gm__ bfloat16_t *)postQuantScale);
                constInfo.postQuantScaleValue = ToFloat(postQuantScaleBf16Gm.GetValue(0));
            }
            if (postQuantOffset != nullptr) {
                postQuantOffsetBf16Gm.SetGlobalBuffer((__gm__ bfloat16_t *)postQuantOffset);
                constInfo.postQuantOffsetValue = ToFloat(postQuantOffsetBf16Gm.GetValue(0));
            } else {
                constInfo.postQuantOffsetValue = 0.0;
            }
        }

        // Channel级量化 + Float存储
        if (constInfo.isPostQuantPerChnl && !constInfo.isPostQuantBF16) {
            if (postQuantScale != nullptr) {
                this->postQuantScaleGm.SetGlobalBuffer((__gm__ float *)postQuantScale);
            }
            if (postQuantOffset != nullptr) {
                constInfo.isPostQuantOffsetExist = true;
                postQuantOffsetGm.SetGlobalBuffer((__gm__ float *)postQuantOffset);
            }
        }

        // Channel级量化 + BF16存储
        if (constInfo.isPostQuantPerChnl && constInfo.isPostQuantBF16) {
            if (postQuantScale != nullptr) {
                postQuantScaleBf16Gm.SetGlobalBuffer((__gm__ bfloat16_t *)postQuantScale);
            }
            if (postQuantOffset != nullptr) {
                constInfo.isPostQuantOffsetExist = true;
                postQuantOffsetBf16Gm.SetGlobalBuffer((__gm__ bfloat16_t *)postQuantOffset);
            }
        }
    }
}

/* ==================== InitUniqueLocalBuffer 实现 ==================== */
/**
 * @brief 初始化Vec层独有局部缓冲区
 * 
 * 根据不同模式分配UB队列:
 * - Softmax LSE: 如果启用，分配(s1BaseSize/2)*sizeof(float)*8
 * - Post-Quant: 分配2KB队列
 * - MLA FullQuant: 分配Query Scale和pScale缓冲区
 * - Learnable Sink: 分配256B队列
 */
TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockVecInferMlaFullquant<TEMPLATE_ARGS>::InitUniqueLocalBuffer(ConstInfo<isInfer, hasRope> &constInfo)
{
    if (constInfo.isSoftmaxLseEnable) {
        // 8: 适配TND，每行的结果存为8个重复lse元素（32B对齐）
        this->tPipe->InitBuffer(softmaxLseQueue, 1, (BaseClass::s1BaseSize >> 1U) * sizeof(float) * 8);
    }
    if constexpr (POST_QUANT) {
        this->tPipe->InitBuffer(postQuantScaleQue, 1, 2048); // 2K
        if (constInfo.isPostQuantOffsetExist) {
            this->tPipe->InitBuffer(postQuantOffsetQue, 1, 2048); // 2K
        }
    }
    if constexpr (isMlaFullQuant) {
        constexpr uint32_t softmaxRowmaxBufSize = 256; // s1 baseSize * 4b(fp32)
        this->tPipe->InitBuffer(BaseClass::queryScaleQue[0], 1, BaseClass::s1BaseSize / CV_RATIO * sizeof(float));
        this->tPipe->InitBuffer(BaseClass::queryScaleQue[1], 1, BaseClass::s1BaseSize / CV_RATIO * sizeof(float));
        this->tPipe->InitBuffer(BaseClass::pScaleBuf[0], softmaxRowmaxBufSize);
        this->tPipe->InitBuffer(BaseClass::pScaleBuf[1], softmaxRowmaxBufSize);
        this->tPipe->InitBuffer(BaseClass::pScaleBuf[2], softmaxRowmaxBufSize);
    }
    if (constInfo.learnableSinkFlag) {
        this->tPipe->InitBuffer(sinkQue, 1, 256); // buffer size = 256 bytes
    }
}

/* ==================== InitFDBuffers 实现 ==================== */
/**
 * @brief 初始化Flash Decode缓冲区
 * 
 * 分配32KB UB缓冲区用于Flash Decode场景:
 * - lseTmpBuff: LSE临时缓冲区
 * - softmaxMaxInputQue: Max值输入队列
 * - softmaxSumInputQue: Sum值输入队列
 * - FDResOutputQue: 结果输出队列
 * - accumOutInputQue: 累加输入队列
 * - Post-Quant队列
 * - Softmax LSE队列
 */
TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockVecInferMlaFullquant<TEMPLATE_ARGS>::InitFDBuffers(ConstInfo<isInfer, hasRope> &constInfo)
{
    this->tPipe->Reset();
    this->tPipe->InitBuffer(lseTmpBuff, bufferSizeByte32K);
    this->tPipe->InitBuffer(softmaxMaxInputQue, 1, bufferSizeByte32K);
    this->tPipe->InitBuffer(softmaxSumInputQue, 1, bufferSizeByte32K);
    this->tPipe->InitBuffer(FDResOutputQue, 1, bufferSizeByte32K);
    this->tPipe->InitBuffer(accumOutInputQue, 1, bufferSizeByte32K);
    if constexpr (POST_QUANT) {
        this->tPipe->InitBuffer(postQuantScaleQue, 1, bufferSizeByte32K);
        if (constInfo.isPostQuantOffsetExist) {
            this->tPipe->InitBuffer(postQuantOffsetQue, 1, bufferSizeByte32K);
        }
    }
    if (constInfo.isSoftmaxLseEnable) {
        // 8: 适配TND, 每行结果存为8个重复lse元素(32B对齐)
        this->tPipe->InitBuffer(softmaxLseQueue, 1, (BaseClass::s1BaseSize >> 1U) * sizeof(float) * 8);
    }
}

/* ==================== FlashDecodeCompute 实现 ==================== */
/**
 * @brief Flash Decode主计算函数
 * 
 * 处理变长序列解码:
 * 1. 根据aivIdx计算batch和head索引
 * 2. 获取当前batch的实际序列长度
 * 3. 如果序列长度为0，直接返回
 * 4. 计算循环次数 (s2Size / sInnerLoopSize)
 * 5. 调用CombineSplitKVRes合并多批次结果
 */
TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockVecInferMlaFullquant<TEMPLATE_ARGS>::FlashDecodeCompute(ConstInfo<isInfer, hasRope> &constInfo,
    GlobalTensor<INPUT_T> &keyGm, __gm__ int64_t *actualSeqKvlenAddr)
{
    int64_t bIdx = constInfo.aivIdx / constInfo.n2Size;
    int64_t n2Idx = constInfo.aivIdx % constInfo.n2Size;
    int64_t batchSize = this->tilingData->inputParamsRegbase.bSize;
    if (constInfo.aivIdx >= batchSize * constInfo.n2Size) {
        return;
    }
    int64_t actualSeqLen;
    GetActualSeqLenKV(constInfo, keyGm, actualSeqKvlenAddr, bIdx, actualSeqLen);
    if (actualSeqLen == 0) {
        return;
    }
    uint64_t attenOutOffset = (uint64_t)bIdx * constInfo.n2GDv + n2Idx * constInfo.gDv;
    constInfo.actualCombineLoopSize = (actualSeqLen + constInfo.sInnerLoopSize - 1) / constInfo.sInnerLoopSize;
    CombineSplitKVRes(constInfo, attenOutOffset, bIdx, n2Idx);
}

/* ==================== SoftmaxDataCopyOut 实现 ==================== */
/**
 * @brief Softmax数据拷贝输出
 * 
 * Vec1的核心入口函数:
 * 
 * Flash Decode模式:
 *   → ComputeLogSumExpAndCopyToGm
 * 
 * 非Flash Decode模式 + Learnable Sink:
 *   → Vec1SinkComputeGSFused (GQA)
 *   → Vec1SinkCompute (非GQA)
 *   → SoftmaxLseCopyOut
 * 
 * @param runInfo 运行信息
 * @param constInfo 常量信息
 * @param sumUb Softmax Sum UB张量
 * @param maxUb Softmax Max UB张量
 */
TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockVecInferMlaFullquant<TEMPLATE_ARGS>::SoftmaxDataCopyOut(
    RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo, LocalTensor<float> &sumUb,
    LocalTensor<float> &maxUb)
{
    if constexpr (isFd) {
        ComputeLogSumExpAndCopyToGm(runInfo, constInfo);
        return;
    }
    if (constInfo.learnableSinkFlag) {
        if (constInfo.isGqa) {
            this->Vec1SinkComputeGSFused(runInfo, constInfo, sumUb, maxUb);
        } else {
            this->Vec1SinkCompute(runInfo, constInfo, sumUb, maxUb);
        }
    }
    SoftmaxLseCopyOut(sumUb, maxUb, runInfo, constInfo);
}

/* ==================== Vec1SinkCompute 实现 ==================== */
/**
 * @brief Vec1 Sink计算
 * 
 * 对softmax结果添加learnable sink偏置:
 * 1. 获取sink值 (half或bfloat16_t)
 * 2. 调用SinkSubExpAddVF进行计算
 *    - 计算 exp(sink - max)
 *    - 更新 sum = sum * exp(sink - max) + exp(x - max)
 *    - 更新 max = max(sink, max)
 */
TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockVecInferMlaFullquant<TEMPLATE_ARGS>::Vec1SinkCompute(RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo, LocalTensor<float> &sumUb, LocalTensor<float> &maxUb) 
{
    int64_t sinkOffset = runInfo.n2oIdx * constInfo.gSize + runInfo.goIdx;
    auto sinkRaw = this->sinkGm.GetValue(sinkOffset);
    float sinkValue;
    if constexpr (IsSameType<decltype(sinkRaw), half>::value) {
        sinkValue = static_cast<float>(sinkRaw);
    } else {
        sinkValue = ToFloat(sinkRaw);
    }
    SinkSubExpAddVF<float>(sumUb, maxUb, sinkValue, runInfo.halfS1RealSize);
}

/* ==================== Vec1SinkComputeGSFused 实现 ==================== */
/**
 * @brief Vec1 Sink计算 (GQA Fused版本)
 * 
 * 优化版本:
 * 1. 先将sink数据拷贝到UB (CopySinkIn)
 * 2. 调用SinkSubExpAddGSFusedVF一次性完成GS合轴融合的sink计算
 */
TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockVecInferMlaFullquant<TEMPLATE_ARGS>::Vec1SinkComputeGSFused(RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo, LocalTensor<float> &sumUb, LocalTensor<float> &maxUb) 
{
    CopySinkIn(runInfo, constInfo);
    LocalTensor<INPUT_T> sinkUb = sinkQue.DeQue<INPUT_T>();
    SinkSubExpAddGSFusedVF<float, INPUT_T>(sinkUb, sumUb, maxUb, runInfo.halfS1RealSize);
    sinkQue.FreeTensor(sinkUb);
}

/* ==================== CopySinkIn 实现 ==================== */
/**
 * @brief 拷贝Sink输入到UB
 * 
 * DataCopy参数:
 * - blockCount = 1: 一次连续拷贝
 * - blockLen = halfS1RealSize * sizeof(INPUT_T)
 * - srcStride = 0: 源地址连续
 * - dstStride = 0: 目的地址连续
 */
TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockVecInferMlaFullquant<TEMPLATE_ARGS>::CopySinkIn(RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo)
{
    LocalTensor<INPUT_T> sinkUbBf16 = sinkQue.AllocTensor<INPUT_T>();
    int64_t sinkOffset = runInfo.n2oIdx * constInfo.gSize + constInfo.subBlockIdx * runInfo.halfS1RealSize;
    DataCopyExtParams sinkCopyParams;
    sinkCopyParams.blockCount = 1; // 进行一次连续拷贝
    sinkCopyParams.blockLen = runInfo.halfS1RealSize * sizeof(INPUT_T);
    sinkCopyParams.srcStride = 0;
    sinkCopyParams.dstStride = 0;

    DataCopyPadExtParams<INPUT_T> sinkCopyPadParams{};
    DataCopyPad(sinkUbBf16, this->sinkGm[sinkOffset], sinkCopyParams, sinkCopyPadParams);
    sinkQue.EnQue(sinkUbBf16);
}

/* ==================== CopyOutAttentionOut 实现 ==================== */
/**
 * @brief 拷贝Attention输出
 * 
 * Vec2层入口函数:
 * 
 * Flash Decode模式:
 *   → Bmm2FDOut: 存储到累加缓冲区
 * 
 * 非Flash Decode模式:
 *   → Bmm2DataCopyOut: 直接输出到GM
 * 
 * @param runInfo 运行信息
 * @param constInfo 常量信息
 * @param vec2ResUb BMM2结果UB张量
 * @param vec2S1Idx S1维度索引
 * @param vec2CalcSize 计算大小
 */
TEMPLATES_DEF_NO_DEFAULT
template <typename VEC2_RES_T>
__aicore__ inline void FABlockVecInferMlaFullquant<TEMPLATE_ARGS>::CopyOutAttentionOut(
    RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo, LocalTensor<VEC2_RES_T> &vec2ResUb, int64_t vec2S1Idx, int64_t vec2CalcSize)
{
    if constexpr (isFd) {
        Bmm2FDOut(vec2ResUb, runInfo, constInfo, vec2S1Idx, vec2CalcSize);
    } else {
        this->Bmm2DataCopyOut(runInfo, constInfo, vec2ResUb, vec2S1Idx, vec2CalcSize);
    }
}

/* ==================== GetActualSeqLenKV 实现 ==================== */
/**
 * @brief 获取KV的实际序列长度
 * 
 * 处理变长序列场景:
 * 
 * 1. 获取基础s2Size
 * 
 * 2. 如果KV不连续 (isKvContinuous == 0):
 *    - 使用ListTensorDesc获取当前batch的shape
 *    - 根据layout提取实际的s2维度
 * 
 * 3. 获取实际长度:
 *    - 如果isActualLenDimsKVNull: 直接使用s2Size
 *    - 否则: 从actualSeqKvlenAddr读取
 * 
 * 4. 处理左padding:
 *    - 如果有左padding，调整actualSeqLen
 *    - 如果padding导致负数，返回0
 */
TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockVecInferMlaFullquant<TEMPLATE_ARGS>::GetActualSeqLenKV(ConstInfo<isInfer, hasRope> &constInfo, 
    GlobalTensor<INPUT_T> &keyGm, __gm__ int64_t *actualSeqKvlenAddr, int64_t boIdx, int64_t &actualSeqLen)
{
    int64_t s2InCurrentBatch = constInfo.s2Size;
    if (constInfo.isKvContinuous == 0) {
        ListTensorDesc keyListTensorDesc((__gm__ void *)keyGm.GetPhyAddr());
        AscendC::TensorDesc<__gm__ uint8_t> kvTensorDesc;
        uint64_t dimInfo[4];
        kvTensorDesc.SetShapeAddr(&dimInfo[0]);
        keyListTensorDesc.GetDesc(kvTensorDesc, boIdx);
        if constexpr (layout == LayOutTypeEnum::LAYOUT_BNSD) {
            s2InCurrentBatch = kvTensorDesc.GetShape(2);
        } else {
            s2InCurrentBatch = kvTensorDesc.GetShape(1);
        }
    }
    if (constInfo.isActualLenDimsKVNull) {
        actualSeqLen = s2InCurrentBatch;
    } else {
        actualSeqLen = (constInfo.actualSeqLenKVSize == 1) ? actualSeqKvlenAddr[0] :
                                                             actualSeqKvlenAddr[boIdx];
    }
    if (constInfo.isKVHasLeftPadding) {
        int64_t kvLeftPaddingSize = constInfo.s2Size - actualSeqLen - constInfo.kvRightPaddingSize;
        if (kvLeftPaddingSize < 0) {
            actualSeqLen = 0;
        }
    }
}

/* ==================== SoftmaxLseCopyOut 实现 ==================== */
/**
 * @brief 拷贝Softmax LSE到GM
 * 
 * 计算log-sum-exp: LSE = log(sum(exp(x - max)))
 * 
 * 步骤:
 * 1. 如果realSize为0，直接返回
 * 2. 分配LSE UB张量
 * 3. 调用ComputeLseOutputVF计算LSE
 * 4. 拷贝到GM，根据layout设置不同的stride:
 *    - TND/NTD: GQA模式需要特殊stride
 *    - BSH: 使用(s1Size - 1)的stride
 *    - 其他: 连续存储
 * 
 * MLA Fullquant + BSH布局的特殊处理:
 * - 处理n2G边界情况
 * - 逐行拷贝并正确设置offset
 */
TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockVecInferMlaFullquant<TEMPLATE_ARGS>::SoftmaxLseCopyOut(
    LocalTensor<float> &softmaxSumTmp, LocalTensor<float> &softmaxMaxTmp, RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo)
{
    if (unlikely(runInfo.halfS1RealSize == 0)) {
        return;
    }

    if (!constInfo.isSoftmaxLseEnable) {
        return;
    }
    LocalTensor<float> lseUb = this->softmaxLseQueue.template AllocTensor<float>();
    ComputeLseOutputVF(lseUb, softmaxSumTmp, softmaxMaxTmp, runInfo.halfS1RealSize);
    softmaxLseQueue.template EnQue(lseUb);
    softmaxLseQueue.DeQue<float>();
    DataCopyExtParams intriParams1;
    intriParams1.blockLen = sizeof(float);
    intriParams1.blockCount = runInfo.halfS1RealSize;
    intriParams1.srcStride = 0;
    if (layout == LayOutTypeEnum::LAYOUT_TND || layout == LayOutTypeEnum::LAYOUT_NTD) {
        intriParams1.dstStride = constInfo.isGqa ? 0 : sizeof(float) * (constInfo.n2G - 1);
    } else {
        intriParams1.dstStride = 0;
    }
    if constexpr (isMlaFullQuant) {
        intriParams1.dstStride = (layout == LayOutTypeEnum::LAYOUT_BSH) ? sizeof(float) * (constInfo.s1Size - 1) : 0;
    }
    if (isMlaFullQuant && layout == LayOutTypeEnum::LAYOUT_BSH && constInfo.gSize < 32) {
        int64_t currRowOffset = runInfo.sOuterOffset % constInfo.n2G;
        int64_t remainDataLen = runInfo.halfS1RealSize;
        int64_t dealDataLen = 0;
        int64_t ubLseOffset = 0;
        int64_t tmpSoftmaxLseOffset = runInfo.softmaxLseOffset;
        int64_t oSoftmaxLseOffset = tmpSoftmaxLseOffset - constInfo.s1Size * (runInfo.sOuterOffset % constInfo.gSize);
        while (remainDataLen > 0) {
            dealDataLen = currRowOffset + remainDataLen < constInfo.n2G ? remainDataLen : constInfo.n2G - currRowOffset;
            intriParams1.blockCount = dealDataLen;
            DataCopyPad(this->softmaxLseGm[tmpSoftmaxLseOffset], lseUb[ubLseOffset], intriParams1);
            remainDataLen -= dealDataLen;
            ubLseOffset += (dealDataLen * 8); // 8: fp32对齐
            currRowOffset = (currRowOffset + dealDataLen) % constInfo.n2G;
            tmpSoftmaxLseOffset = ++oSoftmaxLseOffset;
        }
    } else {
        DataCopyPad(this->softmaxLseGm[runInfo.softmaxLseOffset], lseUb, intriParams1);
    }
    softmaxLseQueue.FreeTensor(lseUb);
}

/* ==================== InitOutputSingleCore 实现 ==================== */
/**
 * @brief 初始化单核输出缓冲区
 * 
 * 将attention输出初始化为0:
 * - POST_QUANT模式: 初始化为half类型的0
 * - 其他模式: 初始化为OUTPUT_T类型的0
 * 
 * 处理尾核情况 (tailSize < singleCoreSize)
 */
TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockVecInferMlaFullquant<TEMPLATE_ARGS>::InitOutputSingleCore(ConstInfo<isInfer, hasRope> &constInfo)
{
    auto &initParams = this->tilingData->initOutputParams;
    uint32_t tailSize = (initParams.totalOutputSize - constInfo.aivIdx * initParams.singleCoreSize) > 0 ?
        (initParams.totalOutputSize - constInfo.aivIdx * initParams.singleCoreSize) : 0;
    uint32_t singleInitOutputSize = tailSize < initParams.singleCoreSize ? tailSize : initParams.singleCoreSize;
    if constexpr (POST_QUANT) {
        InitOutput<half>(this->attentionOutInitGm[constInfo.aivIdx * initParams.singleCoreSize / 2], singleInitOutputSize / 2, 0.0);
    } else {
        InitOutput<OUTPUT_T>(this->attentionOutGm[constInfo.aivIdx * initParams.singleCoreSize], singleInitOutputSize, 0.0);
    }
}

/* ==================== InitLseOutputSingleCore 实现 ==================== */
/**
 * @brief 初始化LSE单核输出
 * 
 * 将softmax LSE初始化为3e+99 (极大值，表示无效批次)
 * 按核数平均分配空间，尾核处理余数
 */
TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockVecInferMlaFullquant<TEMPLATE_ARGS>::InitLseOutputSingleCore(ConstInfo<isInfer, hasRope> &constInfo)
{
    int64_t coreNum = GetBlockNum() * GetTaskRation();
    auto &initParams = this->tilingData->initOutputParams;
    if (coreNum != 0 && constInfo.aivIdx < coreNum) {
        int64_t singleCoreLseSize = initParams.totalSoftMaxLseOutputSize / coreNum;
        if (constInfo.aivIdx == coreNum - 1) {
            singleCoreLseSize += initParams.totalSoftMaxLseOutputSize % coreNum;
        }
        InitOutput<float>(softmaxLseGm[constInfo.aivIdx * (initParams.totalSoftMaxLseOutputSize / coreNum)], 
            singleCoreLseSize, 3e+99); // 3e+99: set the value of invalid batch to inf
    }
}

/* ==================== CombineSplitKVRes 实现 ==================== */
/**
 * @brief 合并Split-KV结果
 * 
 * Flash Decode核心函数，将多批次解码结果合并:
 * 
 * 1. 计算G维度分片大小:
 *    - gSplitSizeLse = 32KB / (32B * splitKVNum)
 *    - gSplitSizeAccumOut = 32KB / sizeof(float) / dVTemplateType
 *    - 取两者较小值以确保UB足够
 *    - 限制最大为gSplitMax(16)或gSize
 * 
 * 2. 计算循环次数:
 *    - loopCount = CeilDiv(gSize, gSplitSize)
 *    - tailSplitSize = gSize - (loopCount-1) * gSplitSize
 * 
 * 3. 循环处理每个分片:
 *    - CopyLseIn: 拷贝max/sum到UB
 *    - ComputeScaleValue: 计算归一化scale
 *    - ReduceFinalRes: 合并attention结果
 *    - CopyFinalResOut: 写出到GM
 */
TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockVecInferMlaFullquant<TEMPLATE_ARGS>::CombineSplitKVRes(
    ConstInfo<isInfer, hasRope> &constInfo, uint64_t attenOutOffset, uint32_t bIdx, uint32_t n2Idx)
{
    uint32_t gSplitSizeLse =
        bufferSizeByte32K / (FA_BYTE_BLOCK * constInfo.splitKVNum); // 32K / (splitKVNum * 32B)
    uint32_t gSplitSizeAccumOut = bufferSizeByte32K / sizeof(float) / (uint32_t)dVTemplateType;
    // 取两者较小的，用来切g，保证ub够用
    uint32_t gSplitSize = (gSplitSizeLse < gSplitSizeAccumOut) ? gSplitSizeLse : gSplitSizeAccumOut;
    if (constInfo.gSize > gSplitMax) {
        gSplitSize = (gSplitSize > gSplitMax) ? gSplitMax : gSplitSize;
    } else {
        gSplitSize = (gSplitSize > constInfo.gSize) ? constInfo.gSize : gSplitSize;
    }
    uint32_t loopCount = CeilDiv(constInfo.gSize, gSplitSize);
    uint32_t tailSplitSize = constInfo.gSize - (loopCount - 1) * gSplitSize;
    uint64_t lseOffset = 0;
    uint64_t sinkOffset = 0;

    // 复用lseTmpBuff作为临时UB
    LocalTensor<T> lseMaxUb = lseTmpBuff.Get<T>();
    uint32_t shapeArray[] = {(uint32_t)gSplitSize, fp32BaseSize};
    lseMaxUb.SetShapeInfo(ShapeInfo(2, shapeArray, DataFormat::ND));

    uint64_t perChannelQuantOffset = n2Idx * constInfo.dSizeV * constInfo.gSize;
    
    // 非尾块处理
    for (uint32_t i = 0; i < loopCount - 1; i++) {
        uint32_t startRow = i * gSplitSize;
        CopyLseIn(constInfo, bIdx, n2Idx, startRow, gSplitSize);
        LocalTensor<T> softmaxMaxLocal = softmaxMaxInputQue.DeQue<T>();
        LocalTensor<T> softmaxSumLocal = softmaxSumInputQue.DeQue<T>();

        lseOffset = (bIdx * constInfo.n2Size + n2Idx) * constInfo.gSize + i * gSplitSize;
        sinkOffset = n2Idx * constInfo.gSize + i * gSplitSize;
        ComputeScaleValue(softmaxMaxLocal, softmaxSumLocal, constInfo, gSplitSize, lseOffset, sinkOffset);

        LocalTensor<T> tmp1 = lseMaxUb;
        ReduceFinalRes(constInfo, bIdx, n2Idx, tmp1, softmaxSumLocal, startRow, gSplitSize);

        softmaxMaxInputQue.FreeTensor(softmaxMaxLocal);
        softmaxSumInputQue.FreeTensor(softmaxSumLocal);
        CopyFinalResOut(constInfo, attenOutOffset, tmp1, startRow, gSplitSize, perChannelQuantOffset);
    }
    
    // 尾块处理
    if (tailSplitSize > 0) {
        uint32_t startRow = (loopCount - 1) * gSplitSize;
        CopyLseIn(constInfo, bIdx, n2Idx, startRow, tailSplitSize);
        LocalTensor<T> softmaxMaxLocal = softmaxMaxInputQue.DeQue<T>();
        LocalTensor<T> softmaxSumLocal = softmaxSumInputQue.DeQue<T>();

        lseOffset = (bIdx * constInfo.n2Size + n2Idx) * constInfo.gSize + (loopCount - 1) * gSplitSize;
        sinkOffset = n2Idx * constInfo.gSize + (loopCount - 1) * gSplitSize;
        ComputeScaleValue(softmaxMaxLocal, softmaxSumLocal, constInfo, tailSplitSize, lseOffset, sinkOffset);

        LocalTensor<T> tmp1 = lseMaxUb;
        ReduceFinalRes(constInfo, bIdx, n2Idx, tmp1, softmaxSumLocal, startRow, tailSplitSize);

        softmaxMaxInputQue.FreeTensor(softmaxMaxLocal);
        softmaxSumInputQue.FreeTensor(softmaxSumLocal);
        CopyFinalResOut(constInfo, attenOutOffset, tmp1, startRow, tailSplitSize, perChannelQuantOffset);
    }
}

/* ==================== ComputeScaleValue 实现 ==================== */
/**
 * @brief 计算Scale值
 * 
 * 计算softmax的归一化因子:
 * 1. 分配LSE输出UB (如果启用)
 * 2. 拷贝Sink数据到UB (如果启用learnable sink)
 * 3. 调用ComputeScaleValue_VF计算scale
 * 4. 如果启用LSE，拷贝到GM
 */
TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockVecInferMlaFullquant<TEMPLATE_ARGS>::ComputeScaleValue(
    LocalTensor<T> lseMaxUb, LocalTensor<T> lseSumUb, ConstInfo<isInfer, hasRope> &constInfo, 
    uint32_t splitSize, uint64_t lseOffset, uint64_t sinkOffset)
{
    LocalTensor<T> lseOutputUb;
    if (constInfo.isSoftmaxLseEnable) {
        lseOutputUb = softmaxLseQueue.template AllocTensor<T>();
    }
    LocalTensor<INPUT_T> tmpSinkUb;
    if (constInfo.learnableSinkFlag) {
        CopySinkFDIn(splitSize, sinkOffset);
        tmpSinkUb = sinkQue.DeQue<INPUT_T>();
    }
    ComputeScaleValue_VF(tmpSinkUb, lseMaxUb, lseSumUb, lseOutputUb, splitSize, constInfo.actualCombineLoopSize,
                         constInfo.isSoftmaxLseEnable, constInfo.learnableSinkFlag);
    if (constInfo.isSoftmaxLseEnable) {
        softmaxLseQueue.template EnQue<T>(lseOutputUb);
        softmaxLseQueue.DeQue<T>();
        DataCopyExtParams intriParams1;
        intriParams1.blockLen = sizeof(float);
        intriParams1.blockCount = splitSize;
        intriParams1.srcStride = 0;
        intriParams1.dstStride = 0;
        DataCopyPad(softmaxLseGm[lseOffset], lseOutputUb, intriParams1);
        softmaxLseQueue.FreeTensor(lseOutputUb);
    }
    if (constInfo.learnableSinkFlag) {
        sinkQue.FreeTensor(tmpSinkUb);
    }
}

/* ==================== CopySinkFDIn 实现 ==================== */
/**
 * @brief 拷贝Sink输入 (Flash Decode版本)
 * 
 * 简单的连续DataCopyPad操作
 */
TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockVecInferMlaFullquant<TEMPLATE_ARGS>::CopySinkFDIn(uint32_t splitSize, uint64_t sinkOffset)
{
    LocalTensor<INPUT_T> sinkUbBf16 = sinkQue.AllocTensor<INPUT_T>();
    DataCopyExtParams sinkCopyParams;
    sinkCopyParams.blockCount = 1;
    sinkCopyParams.blockLen = splitSize * sizeof(INPUT_T);
    sinkCopyParams.srcStride = 0;
    sinkCopyParams.dstStride = 0;

    DataCopyPadExtParams<INPUT_T> sinkCopyPadParams{};
    DataCopyPad(sinkUbBf16, this->sinkGm[sinkOffset], sinkCopyParams, sinkCopyPadParams);
    sinkQue.EnQue(sinkUbBf16);
}

/* ==================== Bmm2FDOut 实现 ==================== */
/**
 * @brief BMM2 Flash Decode输出
 * 
 * 将BMM2结果存储到累加缓冲区:
 * 1. 等待V向量计算完成 (V_MTE3)
 * 2. 设置dataCopyParams:
 *    - blockCount = vec2S1RealSize
 *    - blockLen = dSizeV * sizeof(T)
 *    - srcStride = 对齐填充
 * 3. 计算输出地址:
 *    base = (boIdx * n2Size * gSize * dSizeV + n2oIdx * gSize * dSizeV) * splitKVNum + ...
 * 4. 调用DataCopyPad写入accumOutGm
 */
TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockVecInferMlaFullquant<TEMPLATE_ARGS>::Bmm2FDOut(LocalTensor<T> &vec2ResUb,
    RunInfo<isInfer> &runInfo,  ConstInfo<isInfer, hasRope> &constInfo, int64_t vec2S1Idx, int64_t vec2CalcSize)
{
    LocalTensor<T> attenOut;
    int64_t dSizeAligned64 = (int64_t)dVTemplateType;
    if constexpr (BaseClass::splitD){
        dSizeAligned64 = constInfo.dBasicBlock;
    }
    SetFlag<HardEvent::V_MTE3>(this->vToMte3Id[runInfo.taskIdMod2]);
    WaitFlag<HardEvent::V_MTE3>(this->vToMte3Id[runInfo.taskIdMod2]);
    attenOut = vec2ResUb;

    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = runInfo.vec2S1RealSize;
    dataCopyParams.blockLen = constInfo.dSizeV * sizeof(T);
    dataCopyParams.srcStride = (dSizeAligned64 - constInfo.dSizeV) / (FA_BYTE_BLOCK / sizeof(T));
    dataCopyParams.dstStride = 0;

    uint32_t mStart = constInfo.subBlockIdx * runInfo.firstHalfS1RealSize;
    uint64_t base = (runInfo.boIdx * constInfo.n2Size * constInfo.gSize * constInfo.dSizeV +
                   runInfo.n2oIdx * constInfo.gSize * constInfo.dSizeV) *
                      constInfo.splitKVNum +
                  mStart * constInfo.dSizeV + vec2S1Idx * runInfo.vec2S1BaseSize * constInfo.dSizeV;

    DataCopyPad(this->accumOutGm[base + runInfo.flashDecodeS2Idx * constInfo.gSize * constInfo.dSizeV],
                attenOut, dataCopyParams);
}

/* ==================== CopyLseIn 实现 ==================== */
/**
 * @brief 拷贝LSE到输入队列
 * 
 * 从GM拷贝softmax max/sum到UB:
 * - blockCount = splitKVNum (多批次并行)
 * - blockLen = dealRowCount * fp32BaseSize * sizeof(T)
 * - srcStride = (gSize - dealRowCount) * fp32BaseSize * sizeof(T)
 * - 使用SetShapeInfo设置2D shape
 */
TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockVecInferMlaFullquant<TEMPLATE_ARGS>::CopyLseIn(ConstInfo<isInfer, hasRope> &constInfo,
    uint32_t bIdx, uint32_t n2Idx, uint32_t startRow, uint32_t dealRowCount)
{
    LocalTensor<T> softmaxMaxLocal = softmaxMaxInputQue.AllocTensor<T>();
    LocalTensor<T> softmaxSumLocal = softmaxSumInputQue.AllocTensor<T>();

    DataCopyExtParams copyInParams;
    DataCopyPadExtParams<T> copyInPadParams;
    copyInParams.blockCount = constInfo.splitKVNum;
    copyInParams.blockLen = dealRowCount * fp32BaseSize * sizeof(T);
    copyInParams.srcStride = (constInfo.gSize - dealRowCount) * fp32BaseSize * sizeof(T);
    copyInParams.dstStride = 0;

    copyInPadParams.isPad = false;
    copyInPadParams.leftPadding = 0;
    copyInPadParams.rightPadding = 0;
    copyInPadParams.paddingValue = 0;

    uint64_t combineLseOffset =
        ((uint64_t)bIdx * constInfo.n2Size * constInfo.splitKVNum + n2Idx * constInfo.splitKVNum) *
            constInfo.gSize * fp32BaseSize + startRow * fp32BaseSize;

    DataCopyPad(softmaxMaxLocal, softmaxFDMaxGm[combineLseOffset], copyInParams, copyInPadParams);
    DataCopyPad(softmaxSumLocal, softmaxFDSumGm[combineLseOffset], copyInParams, copyInPadParams);
    softmaxMaxInputQue.EnQue(softmaxMaxLocal);
    softmaxSumInputQue.EnQue(softmaxSumLocal);
}

/* ==================== CopyFinalResOut 实现 ==================== */
/**
 * @brief 拷贝最终结果输出
 * 
 * 将合并后的结果写出:
 * 1. 分配临时UB张量
 * 2. 设置2D shape
 * 3. Cast到OUTPUT_T (非POST_QUANT) 或执行Post-Quantization
 * 4. 调用ReduceFDDataCopyOut写入GM
 */
TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockVecInferMlaFullquant<TEMPLATE_ARGS>::CopyFinalResOut(ConstInfo<isInfer, hasRope> &constInfo, uint64_t attenOutOffset, 
    LocalTensor<T> &accumOutLocal, uint32_t startRow, uint32_t dealRowCount, uint64_t perChannelQuantOffset)
{
    LocalTensor<OUTPUT_T> tmpBmm2ResCastTensor = FDResOutputQue.AllocTensor<OUTPUT_T>();
    uint32_t dSizeAligned64 = (uint32_t)dVTemplateType;
    if constexpr (BaseClass::splitD){
        dSizeAligned64 = constInfo.dBasicBlock;
    }
    uint32_t shapeArray[] = {(uint32_t)dealRowCount, dSizeAligned64};
    tmpBmm2ResCastTensor.SetShapeInfo(ShapeInfo(2, shapeArray, DataFormat::ND));
    if constexpr (!POST_QUANT) {
        Cast(tmpBmm2ResCastTensor, accumOutLocal, AscendC::RoundMode::CAST_ROUND, dealRowCount * dSizeAligned64);
    } else {
        FDPostQuant(constInfo, tmpBmm2ResCastTensor, accumOutLocal, perChannelQuantOffset + startRow * constInfo.dSizeV, dealRowCount, dSizeAligned64);
    }

    FDResOutputQue.EnQue(tmpBmm2ResCastTensor);
    FDResOutputQue.DeQue<OUTPUT_T>();
    ReduceFDDataCopyOut(constInfo, attenOutOffset, tmpBmm2ResCastTensor, startRow, dealRowCount, dSizeAligned64,
                        constInfo.dSizeV);
    FDResOutputQue.FreeTensor(tmpBmm2ResCastTensor);
}

/* ==================== ReduceFinalRes 实现 ==================== */
/**
 * @brief 合并最终结果
 * 
 * 将多个Split-KV分片的结果合并:
 * 循环actualCombineLoopSize次:
 * 1. CopyAccumOutIn: 从GM拷贝累加结果到UB
 * 2. ReduceFinalRes_VF: 执行融合的归一化计算
 *    dst = (dst * sum1 + accum * sum2) / (sum1 + sum2)
 *    max = max(max1, max2)
 */
TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockVecInferMlaFullquant<TEMPLATE_ARGS>::ReduceFinalRes(ConstInfo<isInfer, hasRope> &constInfo, 
    uint32_t bIdx, uint32_t n2Idx, LocalTensor<T> &dst, LocalTensor<T> &lseLocal, uint32_t startRow, uint32_t dealRowCount)
{
    int64_t dSizeAligned64 = (int64_t)dVTemplateType;
    if constexpr (BaseClass::splitD){
        dSizeAligned64 = constInfo.dBasicBlock;
    }
    for (uint32_t j = 0; j < constInfo.actualCombineLoopSize; ++j) {
        CopyAccumOutIn(constInfo, bIdx, n2Idx, j, startRow, dealRowCount);
        LocalTensor<T> accumOutLocal = accumOutInputQue.DeQue<T>();
        ReduceFinalRes_VF<T>(dst, lseLocal, accumOutLocal, dealRowCount, dSizeAligned64, j);
        accumOutInputQue.FreeTensor(accumOutLocal);
    }
}

/* ==================== CopyAccumOutIn 实现 ==================== */
/**
 * @brief 拷贝累加输出到输入队列
 * 
 * DataCopy参数:
 * - blockCount = dealRowCount
 * - blockLen = dSizeV * sizeof(T)
 * - srcStride = 0 (连续)
 * - dstStride = 对齐填充
 * - padding: 右侧填充0
 */
TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockVecInferMlaFullquant<TEMPLATE_ARGS>::CopyAccumOutIn(ConstInfo<isInfer, hasRope> &constInfo, 
    uint32_t bIdx, uint32_t n2Idx, uint32_t splitKVIndex, uint32_t startRow, uint32_t dealRowCount)
{
    LocalTensor<T> accumOutLocal = accumOutInputQue.AllocTensor<T>();
    int64_t dSizeAligned64 = (int64_t)dVTemplateType;
    if constexpr (BaseClass::splitD){
        dSizeAligned64 = constInfo.dBasicBlock;
    }
    DataCopyExtParams copyInParams;
    DataCopyPadExtParams<T> copyInPadParams;
    copyInParams.blockCount = dealRowCount;
    copyInParams.blockLen = constInfo.dSizeV * sizeof(T);
    copyInParams.srcStride = 0;
    copyInParams.dstStride = (dSizeAligned64 - constInfo.dSizeV) / 8; // 8 for align factor

    copyInPadParams.isPad = true;
    copyInPadParams.leftPadding = 0;
    copyInPadParams.rightPadding = (dSizeAligned64 - constInfo.dSizeV) % 8; // 8 for align factor
    copyInPadParams.paddingValue = 0;

    uint64_t combineAccumOutOffset = ((uint64_t)bIdx * constInfo.n2Size * constInfo.splitKVNum +
                                      n2Idx * constInfo.splitKVNum + splitKVIndex) *
                                         constInfo.gSize * constInfo.dSizeV +
                                     startRow * constInfo.dSizeV;
    DataCopyPad(accumOutLocal, this->accumOutGm[combineAccumOutOffset], copyInParams, copyInPadParams);
    accumOutInputQue.EnQue(accumOutLocal);
}

/* ==================== ComputeLogSumExpAndCopyToGm 实现 ==================== */
/**
 * @brief 计算LogSumExp并拷贝到GM
 * 
 * Flash Decode场景下的softmax后处理:
 * 1. 计算不同layout下的offset
 * 2. 调用BroadCastAndCopyOut进行广播和拷贝
 */
TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void
FABlockVecInferMlaFullquant<TEMPLATE_ARGS>::ComputeLogSumExpAndCopyToGm(RunInfo<isInfer> &runInfo,
    ConstInfo<isInfer, hasRope> &constInfo)
{
    if (unlikely(runInfo.halfS1RealSize == 0)) {
        return;
    }
    int64_t bOffset;
    int64_t n2Offset;
    int64_t gOffset;
    if constexpr (layout == LayOutTypeEnum::LAYOUT_TND) {
        bOffset = constInfo.n2G * runInfo.s1SizeAcc;
        n2Offset = runInfo.n2oIdx * constInfo.gSize * runInfo.actualS1Size;
        gOffset = runInfo.goIdx * runInfo.actualS1Size;
    } else {
        bOffset = runInfo.boIdx * constInfo.n2Size * constInfo.gS1;
        n2Offset = runInfo.n2oIdx * constInfo.gS1;
        gOffset = runInfo.goIdx * constInfo.s1Size;
    }
    int64_t s1Offset = runInfo.s1oIdx * this->s1BaseSize + constInfo.subBlockIdx * runInfo.firstHalfS1RealSize;
    int64_t calculateSize = runInfo.halfS1RealSize * fp32BaseSize;
    uint32_t mStart = constInfo.subBlockIdx * runInfo.halfS1RealSize;
    size_t gmOffset =
        runInfo.boIdx * constInfo.n2Size * constInfo.splitKVNum * constInfo.gSize * fp32BaseSize +
        runInfo.n2oIdx * constInfo.splitKVNum * constInfo.gSize * fp32BaseSize +
        runInfo.flashDecodeS2Idx * constInfo.gSize * fp32BaseSize + mStart * fp32BaseSize;
    this->BroadCastAndCopyOut(runInfo, softmaxFDSumGm, softmaxFDMaxGm, gmOffset, calculateSize);
}

/* ==================== ReduceFDDataCopyOut 实现 ==================== */
/**
 * @brief 合并FD数据拷贝输出
 * 
 * 将合并后的数据写入GM:
 * - blockCount = dealRowCount
 * - blockLen = actualColumnCount * sizeof(OUTPUT_T)
 * - srcStride = 对齐填充
 * - dstStride = 0 (连续)
 */
TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockVecInferMlaFullquant<TEMPLATE_ARGS>::ReduceFDDataCopyOut(ConstInfo<isInfer, hasRope> &constInfo,
    uint64_t attenOutOffset, LocalTensor<OUTPUT_T> &attenOutUb, uint32_t startRow, uint32_t dealRowCount,
    uint32_t columnCount, uint32_t actualColumnCount)
{
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = dealRowCount;
    dataCopyParams.blockLen = actualColumnCount * sizeof(OUTPUT_T);
    dataCopyParams.srcStride = (columnCount - actualColumnCount) / (FA_BYTE_BLOCK / sizeof(OUTPUT_T));
    dataCopyParams.dstStride = 0;
    DataCopyPad(this->attentionOutGm[attenOutOffset + startRow * actualColumnCount], attenOutUb, dataCopyParams);
}

/* ==================== PostQuantPerChnl 实现 ==================== */
/**
 * @brief Channel级Post-Quantization
 * 
 * 支持per-channel量化:
 * 1. 分配scale UB张量
 * 2. 从GM拷贝scale参数
 * 3. 如果存在offset，拷贝offset并调用带offset的量化实现
 * 4. 否则调用不带offset的量化实现
 */
TEMPLATES_DEF_NO_DEFAULT
template <typename POSTQUANT_PARAMS_T, typename VEC2_RES_T>
__aicore__ inline void FABlockVecInferMlaFullquant<TEMPLATE_ARGS>::PostQuantPerChnl(
    ConstInfo<isInfer, hasRope> &constInfo, LocalTensor<OUTPUT_T> &attenOut, LocalTensor<VEC2_RES_T> &vec2ResUb,
    uint64_t perChannelQuantOffset, uint32_t gSplitSize, uint32_t s1RowCount, uint32_t splitOffset, int64_t dSizeAligned64,
    GlobalTensor<POSTQUANT_PARAMS_T> postQuantScaleGm, GlobalTensor<POSTQUANT_PARAMS_T> postQuantOffsetGm)
{
    DataCopyExtParams copyInParams;
    DataCopyPadExtParams<POSTQUANT_PARAMS_T> copyInPadParams;
    copyInParams.blockCount = gSplitSize;
    copyInParams.blockLen = constInfo.dSizeV * sizeof(POSTQUANT_PARAMS_T);
    copyInParams.srcStride = 0;
    copyInParams.dstStride = (dSizeAligned64 - constInfo.dSizeV) / (32 / sizeof(POSTQUANT_PARAMS_T));  // 32: datablock size

    LocalTensor<POSTQUANT_PARAMS_T> postQuantScaleUb =
        this->postQuantScaleQue.template AllocTensor<POSTQUANT_PARAMS_T>();
    DataCopyPad(postQuantScaleUb, postQuantScaleGm[perChannelQuantOffset], copyInParams, copyInPadParams);
    this->postQuantScaleQue.template EnQue(postQuantScaleUb);
    this->postQuantScaleQue.template DeQue<POSTQUANT_PARAMS_T>();
    if (constInfo.isPostQuantOffsetExist) {
        LocalTensor<POSTQUANT_PARAMS_T> postQuantOffsetUb =
            this->postQuantOffsetQue.template AllocTensor<POSTQUANT_PARAMS_T>();
        DataCopyPad(postQuantOffsetUb, postQuantOffsetGm[perChannelQuantOffset], copyInParams, copyInPadParams);
        this->postQuantOffsetQue.template EnQue(postQuantOffsetUb);
        this->postQuantOffsetQue.template DeQue<POSTQUANT_PARAMS_T>();
        PostQuantPerChnlImpl<T, OUTPUT_T, POSTQUANT_PARAMS_T>(
            attenOut[splitOffset], vec2ResUb[splitOffset], postQuantScaleUb, postQuantOffsetUb, gSplitSize, s1RowCount,
            constInfo.dSizeV, dSizeAligned64);
        this->postQuantOffsetQue.FreeTensor(postQuantOffsetUb);
    } else {
        PostQuantPerChnlImpl<T, OUTPUT_T, POSTQUANT_PARAMS_T>(
            attenOut[splitOffset], vec2ResUb[splitOffset], postQuantScaleUb, gSplitSize, s1RowCount, constInfo.dSizeV, dSizeAligned64);
    }
    this->postQuantScaleQue.FreeTensor(postQuantScaleUb);
}

/* ==================== PostQuant 实现 ==================== */
/**
 * @brief Post-Quantization (通用版本)
 * 
 * 对BMM2结果执行量化:
 * 
 * Channel级量化:
 * 1. 计算gSplitSize (2KB UB限制)
 * 2. 循环处理每个分片
 * 3. 根据存储精度选择float或bfloat16_t参数
 * 
 * Tensor级量化:
 * 直接调用PostQuantPerTensorImpl
 */
TEMPLATES_DEF_NO_DEFAULT
template <typename VEC2_RES_T>
__aicore__ inline void FABlockVecInferMlaFullquant<TEMPLATE_ARGS>::PostQuant(ConstInfo<isInfer, hasRope> &constInfo,
                                                                      RunInfo<isInfer> &runInfo, LocalTensor<OUTPUT_T> &attenOut,
                                                                      LocalTensor<VEC2_RES_T> &vec2ResUb,
                                                                      int64_t vec2S1Idx, int64_t dSizeAligned64)
{
    uint32_t s1RowCount = constInfo.isGqa ? 1U : runInfo.vec2S1RealSize;
    uint32_t gRowCount = constInfo.isGqa ? runInfo.vec2S1RealSize : 1U;
    if (constInfo.isPostQuantPerChnl) {
        uint64_t perChannelQuantGQAOffset = runInfo.n2oIdx * constInfo.gDv + vec2S1Idx * runInfo.vec2S1BaseSize * constInfo.dSizeV +
                                            runInfo.sOuterOffset * constInfo.dSizeV;
        uint64_t perChannelQuantOffset = constInfo.isGqa ?
                                                perChannelQuantGQAOffset :
                                                runInfo.n2oIdx * constInfo.gDv + runInfo.goIdx * constInfo.dSizeV;
        uint32_t gSplitSize = constInfo.isPostQuantBF16 ? (2048U / ((uint32_t)dSizeAligned64 * sizeof(bfloat16_t))) :
                                                            (2048U / ((uint32_t)dSizeAligned64 * sizeof(float)));
        gSplitSize = gSplitSize > gRowCount ? gRowCount : gSplitSize;
        uint32_t loopCount = (gRowCount + gSplitSize - 1) / gSplitSize;
        uint32_t tailSplitSize = gRowCount - (loopCount - 1) * gSplitSize;
        for (uint32_t i = 0; i < loopCount; i++) {
            uint32_t startRow = i * gSplitSize;
            if (i + 1 == loopCount) {
                gSplitSize = tailSplitSize;
            }
            uint32_t splitOffset = startRow * dSizeAligned64;
            if (constInfo.isPostQuantBF16) {
                PostQuantPerChnl(constInfo, attenOut, vec2ResUb, perChannelQuantOffset + startRow * constInfo.dSizeV,
                                    gSplitSize, s1RowCount, splitOffset, dSizeAligned64, postQuantScaleBf16Gm, postQuantOffsetBf16Gm);
            } else {
                PostQuantPerChnl(constInfo, attenOut, vec2ResUb, perChannelQuantOffset + startRow * constInfo.dSizeV,
                                    gSplitSize, s1RowCount, splitOffset, dSizeAligned64, postQuantScaleGm, postQuantOffsetGm);
            }
        }
    } else {
        PostQuantPerTensorImpl<T, OUTPUT_T, true>(attenOut, vec2ResUb, constInfo.postQuantScaleValue, constInfo.postQuantOffsetValue, 
                                                 runInfo.vec2S1RealSize, constInfo.dSizeV, dSizeAligned64);
    }
}

/* ==================== FDPostQuant 实现 ==================== */
/**
 * @brief Flash Decode Post-Quantization
 * 
 * 针对Flash Decode场景的量化:
 * - Channel级量化: 调用PostQuantPerChnl
 * - Tensor级量化: 调用PostQuantPerTensorImpl
 */
TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockVecInferMlaFullquant<TEMPLATE_ARGS>::FDPostQuant(ConstInfo<isInfer, hasRope> &constInfo,
                                                                   LocalTensor<OUTPUT_T> &attenOut,
                                                                   LocalTensor<T> &accumOutLocal,
                                                                   uint64_t perChannelQuantOffset,
                                                                   uint32_t dealRowCount, uint32_t dSizeAligned64)
{
    if (constInfo.isPostQuantPerChnl) {
        if (constInfo.isPostQuantBF16) {
            PostQuantPerChnl(constInfo, attenOut, accumOutLocal, perChannelQuantOffset, dealRowCount, 1U, 0U, dSizeAligned64,
                             postQuantScaleBf16Gm, postQuantOffsetBf16Gm);
        } else {
            PostQuantPerChnl(constInfo, attenOut, accumOutLocal, perChannelQuantOffset, dealRowCount, 1U, 0U, dSizeAligned64,
                             postQuantScaleGm, postQuantOffsetGm);
        }
    } else {
        PostQuantPerTensorImpl<T, OUTPUT_T, true>(attenOut, accumOutLocal, constInfo.postQuantScaleValue, constInfo.postQuantOffsetValue, 
                                                   dealRowCount, constInfo.dSizeV, dSizeAligned64);
    }
}

#endif // FLASH_ATTENTION_SCORE_BLOCK_VEC_INFER_MLA_FULLQUANT_H_
