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
 * \file vf_basic_block_aligned128_no_update.h
 * \brief 针对 128 字节对齐、无更新（no update）场景的基础块向量处理函数。
 *        用于 Attention 机制中 softmax 的前向计算，包含 mask、pse（位置编码）、dropout 等融合操作。
 */
#ifndef VF_BASIC_BLOCK_ALIGNED128_NO_UPDATE_H
#define VF_BASIC_BLOCK_ALIGNED128_NO_UPDATE_H

#include "vf_basic_block_utils.h"
#include "../pse.h"

using namespace regbaseutil;

namespace FaVectorApi {

/**
 * @brief 向量处理核心实现（向量函数版本，使用 __simd_vf__ 指令）
 * 
 * @tparam T 输入/中间计算的数据类型（通常为 float）
 * @tparam T2 输出数据类型（float / bfloat16_t / half / fp8 等）
 * @tparam pseShiftType PSE（位置编码）数据的类型
 * @tparam s1BaseSize S1 方向基础大小（默认128）
 * @tparam s2BaseSize S2 方向基础大小（默认128）
 * @tparam hasAtten 是否应用 attention mask
 * @tparam pseMode PSE 类型枚举（无、内联乘加、外联乘加等）
 * @tparam hasDrop 是否应用 dropout
 * @tparam isMlaSgd 是否为 MLA（多头潜在注意力）SGD 模式
 * @tparam isMlaFullQuant 是否为 MLA 全量化模式
 * @param expUb         输出缓冲区：存储指数结果（exp）的 UB 指针
 * @param x_expUb       辅助输出缓冲区（当 T2=float 时使用，存储另一半指数）
 * @param pseUb         PSE 数据 UB 指针
 * @param expSumUb      指数和输出 UB 指针
 * @param maxUb         最大值输出 UB 指针（每行的最大值）
 * @param maxUbStart    最大值起始位置（用于广播加载）
 * @param srcUb         输入源数据 UB 指针（QK^T 结果）
 * @param qScaleUb      量化缩放因子 UB 指针（MLA 全量化时使用）
 * @param indexesUb     索引数据 UB 指针（用于 fp8 数据重排）
 * @param maskUb        attention mask 数据 UB 指针（前一半）
 * @param maskUbUnroll  attention mask 数据 UB 指针（后一半，unroll 部分）
 * @param dropMaskUb    dropout mask 数据 UB 指针
 * @param divValue      1 / keepProb，用于 dropout 缩放
 * @param blockStride   块 stride（高16位为 blockStride，低16位为 repeatStride）
 * @param repeatStride  repeat stride
 * @param dScale        缩放因子（scale * dScaleQK）
 * @param m             循环次数（行数）
 * @param pseStride     PSE 数据 stride
 * @param slopes        ALiBi 斜率
 * @param posShift      位置偏移（用于 ALiBi 计算）
 * @param scale         缩放因子
 * @param dScaleQK      QK 缩放因子
 * @param minValue      最小值（用于 mask 填充）
 * @param deSCaleKValue 反量化缩放值（用于 MLA 全量化）
 */
template <typename T, typename T2, typename pseShiftType, uint32_t s1BaseSize = 128, uint32_t s2BaseSize = 128,
    bool hasAtten = 0, PseTypeEnum pseMode = PseTypeEnum::PSE_NONE_TYPE, bool hasDrop = 0, bool isMlaSgd = false, bool isMlaFullQuant = false>
__simd_vf__ void ProcessVec1NoUpdateImpl128VF(
    __ubuf__ T2 * expUb, __ubuf__ T2 * x_expUb, __ubuf__ pseShiftType * pseUb, __ubuf__ T * expSumUb, 
    __ubuf__ T * maxUb, __ubuf__ T * maxUbStart, __ubuf__ T * srcUb, __ubuf__ T * qScaleUb, 
    __ubuf__ uint8_t * indexesUb, __ubuf__ uint32_t * maskUb, __ubuf__ uint32_t * maskUbUnroll, __ubuf__ uint32_t * dropMaskUb,
    float divValue, const uint32_t blockStride, const uint32_t repeatStride, const float dScale, 
    const uint16_t m, const uint32_t pseStride, const float slopes, const float posShift, const T scale, 
    const float dScaleQK, const T minValue, const float deSCaleKValue = 1.0f)
{
    // ---------- 声明向量寄存器变量 ----------
    // 使用 RegTensor 模板封装向量寄存器，提供类型安全和操作重载
    RegTensor<float> vreg_min;                // 存储最小值常量（用于 mask）
    RegTensor<float> vreg_sel;                 // 选择结果（mask 后的值）
    RegTensor<float> vreg_sel_unroll;           // 选择结果（unroll 部分）
    RegTensor<float> vreg_input_x;              // 输入数据（前半部分）
    RegTensor<float> vreg_input_x_unroll;        // 输入数据（后半部分，unroll）
    RegTensor<float> vreg_max_tmp;               // 临时最大值（用于 reduce）
    RegTensor<float> vreg_input_max;              // 当前行的最大值
    RegTensor<float> vreg_max_brc;                // 广播后的最大值（用于 exp 减法）
    RegTensor<float> vreg_exp_sum;                // 指数和（用于 reduce）
    RegTensor<float> vreg_exp_even;                // exp 结果（偶数部分）
    RegTensor<float> vreg_exp_odd;                 // exp 结果（奇数部分）
    RegTensor<float> vreg_zero;                    // 零向量（用于 dropout）
    RegTensor<float> vreg_pse;                      // PSE 数据（前半）
    RegTensor<float> vreg_pse_unroll;                // PSE 数据（后半）
    RegTensor<float> vreg_alibi;                     // ALiBi 位置编码值
    RegTensor<float> vreg_alibi_unroll;               // ALiBi unroll 部分
    RegTensor<float> vreg_sel_drop;                   // dropout 选择结果（前半）
    RegTensor<float> vreg_sel_drop2;                  // dropout 选择结果（后半）
    RegTensor<float> vreg_rowmax_p;                   // 行最大值（暂未使用，可能保留）
    RegTensor<float> vreg_scale_qk;                    // QK 缩放因子（量化时使用）

    // bfloat16_t 类型寄存器
    RegTensor<bfloat16_t> vreg_exp_even_bf16;
    RegTensor<bfloat16_t> vreg_exp_odd_bf16;
    RegTensor<bfloat16_t> vreg_exp_bf16;
    RegTensor<bfloat16_t> vreg_pse_bf16_src;
    RegTensor<bfloat16_t> vreg_pse_bf16;
    RegTensor<bfloat16_t> vreg_pse_bf16_unroll;

    // half 类型寄存器
    RegTensor<half> vreg_exp_even_f16;
    RegTensor<half> vreg_exp_odd_f16;
    RegTensor<half> vreg_exp_f16;
    RegTensor<half> vreg_pse_f16_src;
    RegTensor<half> vreg_pse_f16;
    RegTensor<half> vreg_pse_f16_unroll;

    // 用于非对齐存储的辅助结构（内部管理寄存器与地址更新）
    UnalignRegForStore ureg_max;
    UnalignRegForStore ureg_exp_sum;

    // 创建掩码寄存器：全1掩码（根据数据类型和模式）
    MaskReg preg_all = CreateMask<T, MaskPattern::ALL>();
    MaskReg preg_all_b16 = CreateMask<uint16_t, MaskPattern::ALL>();
    MaskReg preg_compare;          // 比较掩码（用于 mask 选择）
    MaskReg preg_compare_unroll;    // unroll 部分比较掩码

    // 临时掩码变量（用于 dropout 等）
    MaskReg preg1;
    MaskReg preg2 = CreateMask<int8_t, MaskPattern::ALLF>();  // 全0掩码？实际 ALLF 可能表示全 false
    MaskReg preg3;
    MaskReg preg4;
    MaskReg preg5;
    MaskReg preg6;

    // ---------- 预处理：初始化常量和掩码 ----------
    if constexpr (hasAtten == 1) {
        Duplicate(vreg_min, minValue);      // 将 minValue 复制到向量寄存器所有通道
        if constexpr (isMlaSgd) {
            // MLA SGD 模式：直接加载 mask 数据到掩码寄存器（DIST_DS 表示数据按双字分布）
            MicroAPI::LoadAlign<uint32_t, MicroAPI::MaskDist::DIST_DS>
                (preg_compare, ((__ubuf__ uint32_t*)(maskUb)));
            MicroAPI::LoadAlign<uint32_t, MicroAPI::MaskDist::DIST_DS>
                (preg_compare_unroll, ((__ubuf__ uint32_t*)(maskUbUnroll)));
        }
    }
    if constexpr (pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_TYPE ||
                    pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_SQRT_TYPE) {
        // ALiBi 位置编码：生成从 posShift 开始的等差数列
        Arange(vreg_alibi, posShift);
        Arange(vreg_alibi_unroll, posShift + 64); // 后半部分偏移 64
    }

    // ---------- 主循环：按行处理，每行 s2BaseSize 个元素（通常128） ----------
    for (uint16_t i = 0; i < m; ++i) {
        // 加载当前行的源数据（前半和后半）
        LoadAlign(vreg_input_x, srcUb + i * s2BaseSize);
        LoadAlign(vreg_input_x_unroll, srcUb + floatRepSize + i * s2BaseSize); // floatRepSize 为 64 字节（32个float？需根据上下文）

        // ---------- 量化缩放（MLA 全量化） ----------
        if constexpr (isMlaFullQuant) {
            // 加载量化缩放因子（每个通道一个标量，广播到向量）
            LoadAlign<T, MicroAPI::LoadDist::DIST_BRC_B32>(vreg_scale_qk, qScaleUb + i);
            Muls(vreg_scale_qk, vreg_scale_qk, scale, preg_all);          // 乘以 scale
            Muls(vreg_scale_qk, vreg_scale_qk, deSCaleKValue, preg_all);  // 乘以反量化值
            Mul(vreg_input_x, vreg_input_x, vreg_scale_qk, preg_all);     // 应用缩放
            Mul(vreg_input_x_unroll, vreg_input_x_unroll, vreg_scale_qk, preg_all);
        } else {
            // 非量化模式，根据 pseMode 决定是否缩放
            if constexpr (pseMode != PseTypeEnum::PSE_OUTER_ADD_MUL_TYPE) {
                Muls(vreg_input_x, vreg_input_x, dScale, preg_all);
                Muls(vreg_input_x_unroll, vreg_input_x_unroll, dScale, preg_all);
            } else {
                // 对于外联乘加类型，仅当输出为 fp8 时才缩放 dScaleQK（后续还会用 scale 再乘一次）
                if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
                            IsSameType<T2, hifloat8_t>::value) {
                    Muls(vreg_input_x, vreg_input_x, dScaleQK, preg_all);
                    Muls(vreg_input_x_unroll, vreg_input_x_unroll, dScaleQK, preg_all);
                }
            }
        }

        // ---------- 位置编码（PSE）融合 ----------
        if constexpr (pseMode != PseTypeEnum::PSE_NONE_TYPE) {
            if constexpr (pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_TYPE ||
                            pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_SQRT_TYPE) {
                // ALiBi 模式：计算 slopes * |posShift + offset|
                Abs(vreg_pse, vreg_alibi, preg_all);
                Abs(vreg_pse_unroll, vreg_alibi_unroll, preg_all);
                if constexpr (pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_SQRT_TYPE) {
                    Sqrt(vreg_pse, vreg_pse, preg_all);
                    Sqrt(vreg_pse_unroll, vreg_pse_unroll, preg_all);
                }
                Muls(vreg_pse, vreg_pse, slopes, preg_all);
                Muls(vreg_pse_unroll, vreg_pse_unroll, slopes, preg_all);
                // 更新 alibi 向量（递减，用于下一行？可能用于累积位置）
                Adds(vreg_alibi, vreg_alibi, -1.0f, preg_all);
                Adds(vreg_alibi_unroll, vreg_alibi_unroll, -1.0f, preg_all);
            } else {
                // 其他 PSE 模式：从外部缓冲区加载
                if constexpr (IsSameType<pseShiftType, float>::value) {
                    LoadAlign(vreg_pse, pseUb + i * pseStride);
                    LoadAlign(vreg_pse_unroll, pseUb + i * pseStride + (s2BaseSize >> 1));
                } else if constexpr (IsSameType<pseShiftType, bfloat16_t>::value) {
                    // 加载 bf16 类型，解交织后转换为 float
                    LoadAlign(vreg_pse_bf16_src, pseUb + i * pseStride);
                    Interleave(vreg_pse_bf16, vreg_pse_bf16_unroll, vreg_pse_bf16_src, vreg_pse_bf16_src);
                    Cast<T, pseShiftType, castTraitZero>(vreg_pse, vreg_pse_bf16, preg_all_b16);
                    Cast<T, pseShiftType, castTraitZero>(vreg_pse_unroll, vreg_pse_bf16_unroll, preg_all_b16);
                } else {
                    // 默认 half 类型
                    LoadAlign(vreg_pse_f16_src, pseUb + i * pseStride);
                    Interleave(vreg_pse_f16, vreg_pse_f16_unroll, vreg_pse_f16_src, vreg_pse_f16_src);
                    Cast<T, pseShiftType, castTraitZero>(vreg_pse, vreg_pse_f16, preg_all_b16);
                    Cast<T, pseShiftType, castTraitZero>(vreg_pse_unroll, vreg_pse_f16_unroll, preg_all_b16);
                }
            }
            // 将 PSE 加到输入上
            Add(vreg_input_x, vreg_input_x, vreg_pse, preg_all);
            Add(vreg_input_x_unroll, vreg_input_x_unroll, vreg_pse_unroll, preg_all);
        }

        // ---------- 外联乘加 PSE 的额外缩放 ----------
        if constexpr (pseMode == PseTypeEnum::PSE_OUTER_ADD_MUL_TYPE) {
            Muls(vreg_input_x, vreg_input_x, scale, preg_all);
            Muls(vreg_input_x_unroll, vreg_input_x_unroll, scale, preg_all);
        }

        // ---------- Attention Mask 处理 ----------
        if constexpr (hasAtten == 1) {
            if constexpr (!isMlaSgd) {
                // 非 MLA SGD 模式：每次循环加载 mask（后更新指针）
                LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                    preg_compare, (__ubuf__ uint32_t *&)maskUb, s2BaseSize);
                LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                    preg_compare_unroll, (__ubuf__ uint32_t *&)maskUbUnroll, s2BaseSize); 
            }
            // 根据 mask 选择：mask 为 1 的位置保留原值，否则填充 minValue
            Select(vreg_sel, vreg_min, vreg_input_x, preg_compare);
            Select(vreg_sel_unroll, vreg_min, vreg_input_x_unroll, preg_compare_unroll);
            // 将选择结果写回 srcUb（覆盖原值？可能用于后续计算，但函数名 no_update 似乎矛盾，需结合上下文）
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                (__ubuf__ T *&)srcUb + i * s2BaseSize, vreg_sel, preg_all);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                (__ubuf__ T *&)srcUb + floatRepSize + i * s2BaseSize, vreg_sel_unroll, preg_all);
            // 取前后两部分的最大值（用于后续行最大值）
            Max(vreg_max_tmp, vreg_sel, vreg_sel_unroll, preg_all);
        } else {
            // 无 mask：直接存储输入值，并取最大值
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                (__ubuf__ T *&)srcUb + i * s2BaseSize, vreg_input_x, preg_all);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                (__ubuf__ T *&)srcUb + floatRepSize + i * s2BaseSize, vreg_input_x_unroll, preg_all);
            Max(vreg_max_tmp, vreg_input_x, vreg_input_x_unroll, preg_all);
        }

        // ---------- 行内规约求最大值 ----------
        Reduce<MicroAPI::ReduceType::MAX, float, float, MicroAPI::MaskMergeMode::ZEROING>(
            vreg_input_max, vreg_max_tmp, preg_all);
        // 将当前行的最大值存储到 maxUb 中（非对齐，更新指针）
        StoreUnAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            ((__ubuf__ T *&)maxUb), vreg_input_max, ureg_max, 1);
    }
    // 完成所有行最大值存储后，清空剩余数据（处理部分写）
    StoreUnAlignPost<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
        ((__ubuf__ T *&)maxUb), ureg_max, 0);

    // 如果需要 dropout，初始化零向量
    if constexpr (hasDrop == 1) {
        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, float>(vreg_zero, 0.0f, preg_all);
    }

    // 内存屏障：确保之前的存储对后续加载可见
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();

    // ---------- 第二阶段：计算 exp(x - max) 并求和 ----------
    for (uint16_t i = 0; i < m; ++i) {
        // 加载之前保存的该行最大值（广播到所有通道）
        LoadAlign<T, MicroAPI::LoadDist::DIST_BRC_B32>(
            vreg_max_brc, maxUbStart + i);

        // 加载源数据（经过 mask 处理后的值）
        if constexpr (IsSameType<T2, float>::value) {
            // 输出为 float 时，直接按 float 加载（前后部分连续）
            LoadAlign(vreg_input_x, srcUb + i * s2BaseSize);
            LoadAlign(vreg_input_x_unroll, srcUb + i * s2BaseSize + (s2BaseSize >> 1));
        } else {
            // 其他类型（如 half/bf16），可能按交错方式加载，得到两个 float 向量
            LoadAlign<T, MicroAPI::LoadDist::DIST_DINTLV_B32>(
                vreg_input_x, vreg_input_x_unroll, srcUb + i * s2BaseSize);
        }

        // 计算 exp(x - max)
        ExpSub(vreg_exp_even, vreg_input_x, vreg_max_brc, preg_all);
        ExpSub(vreg_exp_odd, vreg_input_x_unroll, vreg_max_brc, preg_all);

        // 计算当前行的指数和
        Add(vreg_exp_sum, vreg_exp_even, vreg_exp_odd, preg_all);
        Reduce<MicroAPI::ReduceType::SUM, float, float, MicroAPI::MaskMergeMode::ZEROING>(
            vreg_exp_sum, vreg_exp_sum, preg_all);
        // 存储指数和（非对齐）
        StoreUnAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            ((__ubuf__ T *&)expSumUb), vreg_exp_sum, ureg_exp_sum, 1);

        // MLA 全量化：将指数结果缩放到 fp8 范围
        if constexpr (isMlaFullQuant) {
            Muls(vreg_exp_even, vreg_exp_even, fp8e4m3MaxValue, preg_all);
            Muls(vreg_exp_odd, vreg_exp_odd, fp8e4m3MaxValue, preg_all);
        }

        // ---------- Dropout 处理 ----------
        if constexpr (hasDrop == 1) {
            // 加载 dropout mask（每 8 个元素一个 uint32_t，因此 stride 为 s2BaseSize >> 3）
            LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_US>(
                preg1, (__ubuf__ uint32_t *&)dropMaskUb, s2BaseSize >> 3);
            // 将 mask 转换为适合前后两部分的掩码（根据输出类型进行交织/解交织）
            if constexpr (IsSameType<T2, float>::value) {
                MaskInterleave<half>(preg5, preg6, preg1, preg2);
            } else {
                MaskInterleave<half>(preg3, preg4, preg1, preg2);
                MaskDeInterleave<T>(preg5, preg6, preg3, preg4);
            }
            // 根据 mask 选择保留原值或置零
            Select(vreg_sel_drop, vreg_exp_even, vreg_zero, preg5);
            Muls(vreg_exp_even, vreg_sel_drop, divValue, preg_all);  // 缩放 1/keepProb
            Select(vreg_sel_drop2, vreg_exp_odd, vreg_zero, preg6);
            Muls(vreg_exp_odd, vreg_sel_drop2, divValue, preg_all);
        }

        // ---------- 存储指数结果到输出缓冲区（根据 T2 类型进行格式转换和重排）----------
        if constexpr (IsSameType<T2, float>::value) {
            // 输出 float：直接存储，使用块拷贝模式（blockStride, repeatStride 控制 S1 方向步长）
            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2 *&)expUb), vreg_exp_even, blockStride, repeatStride, preg_all);
            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2 *&)x_expUb), vreg_exp_odd, blockStride, repeatStride, preg_all);
        } else if constexpr (IsSameType<T2, bfloat16_t>::value) {
            // 输出 bf16：将 float 转换为 bf16，然后合并偶数/奇数部分（按位或）后存储
            Cast<T2, T, castTraitZero>(vreg_exp_even_bf16, vreg_exp_even, preg_all);
            Cast<T2, T, castTraitOne>(vreg_exp_odd_bf16, vreg_exp_odd, preg_all);
            Or((RegTensor<uint16_t>&)vreg_exp_bf16, (RegTensor<uint16_t>&)vreg_exp_even_bf16,
                (RegTensor<uint16_t>&)vreg_exp_odd_bf16, preg_all_b16);
            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2 *&)expUb), vreg_exp_bf16, blockStride, repeatStride, preg_all_b16);
        } else if constexpr (IsSameType<T2, fp8_e5m2_t>::value) {
            // 输出 fp8_e5m2：转换为 fp8，然后根据 indexesUb 进行 Gather 重排（因为 fp8 可能需特殊布局）
            RegTensor<fp8_e5m2_t> vreg_exp_even_f8e5m2;
            RegTensor<fp8_e5m2_t> vreg_exp_odd_f8e5m2;
            RegTensor<fp8_e5m2_t> vreg_exp_merge_tmp_f8e5m2;
            RegTensor<fp8_e5m2_t> vreg_exp_merge_f8e5m2;
            RegTensor<uint8_t> vreg_exp_merge_f8e5m2_indexes;
            MaskReg preg_all_b8 = CreateMask<T2, MaskPattern::ALL>();
            uint32_t maskLen = 128;  // 输出长度为 128
            MaskReg preg_all_b8_128 = UpdateMask<T2>(maskLen);
            // 转换并合并（偶数部分和奇数部分分别转换，然后按位或合并为临时向量）
            Cast<T2, T, castTraitRintZero>(vreg_exp_even_f8e5m2, vreg_exp_even, preg_all);
            Cast<T2, T, castTraitRintTwo>(vreg_exp_odd_f8e5m2, vreg_exp_odd, preg_all);
            Or((RegTensor<uint8_t>&)vreg_exp_merge_tmp_f8e5m2, (RegTensor<uint8_t>&)vreg_exp_even_f8e5m2, (RegTensor<uint8_t>&)vreg_exp_odd_f8e5m2, preg_all_b8);
            // 加载索引，根据索引重排数据
            LoadAlign(vreg_exp_merge_f8e5m2_indexes, indexesUb);
            Gather(vreg_exp_merge_f8e5m2, vreg_exp_merge_tmp_f8e5m2, vreg_exp_merge_f8e5m2_indexes);
            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2 *&)expUb), vreg_exp_merge_f8e5m2, blockStride, repeatStride, preg_all_b8_128);
        } else if constexpr (IsSameType<T2, fp8_e4m3fn_t>::value) {
            // 输出 fp8_e4m3fn：类似 fp8_e5m2 处理
            RegTensor<fp8_e4m3fn_t> vreg_exp_even_f8e4m3;
            RegTensor<fp8_e4m3fn_t> vreg_exp_odd_f8e4m3;
            RegTensor<fp8_e4m3fn_t> vreg_exp_merge_tmp_f8e4m3;
            RegTensor<fp8_e4m3fn_t> vreg_exp_merge_f8e4m3;
            RegTensor<uint8_t> vreg_exp_merge_f8e4m3_indexes;
            MaskReg preg_all_b8 = CreateMask<T2, MaskPattern::ALL>();
            uint32_t maskLen = 128;
            MaskReg preg_all_b8_128 = UpdateMask<T2>(maskLen);
            Cast<T2, T, castTraitRintZero>(vreg_exp_even_f8e4m3, vreg_exp_even, preg_all);
            Cast<T2, T, castTraitRintTwo>(vreg_exp_odd_f8e4m3, vreg_exp_odd, preg_all);
            Or((RegTensor<uint8_t>&)vreg_exp_merge_tmp_f8e4m3, (RegTensor<uint8_t>&)vreg_exp_even_f8e4m3, (RegTensor<uint8_t>&)vreg_exp_odd_f8e4m3, preg_all_b8);
            LoadAlign(vreg_exp_merge_f8e4m3_indexes, indexesUb);
            Gather(vreg_exp_merge_f8e4m3, vreg_exp_merge_tmp_f8e4m3, vreg_exp_merge_f8e4m3_indexes);
            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2 *&)expUb), vreg_exp_merge_f8e4m3, blockStride, repeatStride, preg_all_b8_128);
        } else if constexpr (IsSameType<T2, hifloat8_t>::value) {
            // 输出 hifloat8：类似 fp8 处理
            RegTensor<hifloat8_t> vreg_exp_even_hif8;
            RegTensor<hifloat8_t> vreg_exp_odd_hif8;
            RegTensor<hifloat8_t> vreg_exp_merge_tmp_hif8;
            RegTensor<hifloat8_t> vreg_exp_merge_hif8;
            RegTensor<uint8_t> vreg_exp_merge_hif8_indexes;
            MaskReg preg_all_b8 = CreateMask<T2, MaskPattern::ALL>();
            uint32_t maskLen = 128;
            MaskReg preg_all_b8_128 = UpdateMask<T2>(maskLen);
            Cast<T2, T, castTraitZero>(vreg_exp_even_hif8, vreg_exp_even, preg_all);
            Cast<T2, T, castTraitTwo>(vreg_exp_odd_hif8, vreg_exp_odd, preg_all);
            Or((RegTensor<uint8_t>&)vreg_exp_merge_tmp_hif8, (RegTensor<uint8_t>&)vreg_exp_even_hif8, (RegTensor<uint8_t>&)vreg_exp_odd_hif8, preg_all_b8);
            LoadAlign(vreg_exp_merge_hif8_indexes, indexesUb);
            Gather(vreg_exp_merge_hif8, vreg_exp_merge_tmp_hif8, vreg_exp_merge_hif8_indexes);
            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2 *&)expUb), vreg_exp_merge_hif8, blockStride, repeatStride, preg_all_b8_128);
        } else {
            // 默认 half 类型：转换为 half，合并后存储
            Cast<T2, T, castTraitZero>(vreg_exp_even_f16, vreg_exp_even, preg_all);
            Cast<T2, T, castTraitOne>(vreg_exp_odd_f16, vreg_exp_odd, preg_all);
            Or((RegTensor<uint16_t>&)vreg_exp_f16, (RegTensor<uint16_t>&)vreg_exp_even_f16, (RegTensor<uint16_t>&)vreg_exp_odd_f16, preg_all_b16);
            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2 *&)expUb), vreg_exp_f16, blockStride, repeatStride, preg_all_b16);
        }
    }
    // 完成所有指数和存储
    StoreUnAlignPost<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            ((__ubuf__ T *&)expSumUb), ureg_exp_sum, 0);
}

/**
 * @brief 向量处理入口（AI Core 函数，封装 UB 指针并调用向量函数）
 * 
 * @tparam T 输入数据类型（通常为 float）
 * @tparam T2 输出数据类型
 * @tparam pseShiftType PSE 数据类型
 * @tparam s1BaseSize S1 基础大小（默认128）
 * @tparam s2BaseSize S2 基础大小（默认128）
 * @tparam hasAtten 是否使用 attention mask
 * @tparam pseMode PSE 类型
 * @tparam hasDrop 是否使用 dropout
 * @tparam isMlaSgd 是否为 MLA SGD
 * @tparam isMlaFullQuant 是否为 MLA 全量化
 * @param dstTensor         输出 Tensor（存储指数结果）
 * @param indexesTensor     索引 Tensor（用于 fp8 重排）
 * @param expSumTensor      指数和 Tensor
 * @param maxTensor         最大值 Tensor（每行的最大值）
 * @param srcTensor         源 Tensor（QK^T 结果，会被 mask 修改）
 * @param expMaxTensor      指数最大值 Tensor（未使用，可能保留）
 * @param inExpSumTensor    输入指数和 Tensor（未使用）
 * @param inMaxTensor       输入最大值 Tensor（未使用）
 * @param maskTensor        attention mask Tensor（uint8_t 类型，但被 reinterpret 为 uint32_t 使用）
 * @param pseTensor         PSE Tensor
 * @param dropTensor        dropout mask Tensor
 * @param sharedTmpBuffer   共享临时缓冲区（未使用）
 * @param m                 行数
 * @param originN           原始 N 大小（固定 128，未使用）
 * @param pseStride         PSE 步长
 * @param slopes            ALiBi 斜率
 * @param posShift          位置偏移
 * @param scale             缩放因子
 * @param dScaleQK          QK 缩放因子
 * @param minValue          最小值（用于 mask）
 * @param keepProb          dropout 保留概率
 * @param queryScaleUb      查询缩放 Tensor（MLA 全量化使用）
 * @param deSCaleKValue     反量化缩放值
 */
template <typename T, typename T2, typename pseShiftType, uint32_t s1BaseSize = 128, uint32_t s2BaseSize = 128,
    bool hasAtten = 0, PseTypeEnum pseMode = PseTypeEnum::PSE_NONE_TYPE, bool hasDrop = 0, bool isMlaSgd = false, bool isMlaFullQuant = false>
__aicore__ inline void ProcessVec1NoUpdateImpl128(
    const LocalTensor<T2>& dstTensor, const LocalTensor<uint8_t>& indexesTensor, const LocalTensor<T>& expSumTensor, const LocalTensor<T>& maxTensor,
    const LocalTensor<T>& srcTensor, const LocalTensor<T>& expMaxTensor, const LocalTensor<T>& inExpSumTensor,
    const LocalTensor<T>& inMaxTensor, const LocalTensor<uint8_t>& maskTensor, const LocalTensor<pseShiftType>& pseTensor,
    const LocalTensor<uint8_t>& dropTensor, const LocalTensor<uint8_t>& sharedTmpBuffer, const uint16_t m,
    const uint32_t originN, const uint32_t pseStride, const float slopes, const float posShift, const T scale, const float dScaleQK,
    const T minValue, float keepProb, const LocalTensor<T>& queryScaleUb = LocalTensor<T>(), const float deSCaleKValue = 1.0f)
{
    float divValue = 1.0f / keepProb;
    // 块拷贝参数：高16位为 blockStride，低16位为 repeatStride。
    // 这里 blockStride = s1BaseSize >> 1 | 0x1，即 (64 | 1) = 65，表示每次写块后跳 65 个元素（可能对应 S1 方向步长）
    // repeatStride = 1 表示每次重复后跳 1 个元素。
    const uint32_t blockStride = s1BaseSize >> 1 | 0x1;
    const uint32_t repeatStride = 1;

    // 将 LocalTensor 转换为统一缓冲区指针（__ubuf__），供向量函数使用
    __ubuf__ T2 * expUb = (__ubuf__ T2*)dstTensor.GetPhyAddr();
    __ubuf__ T2 * x_expUb = nullptr;
    if constexpr (IsSameType<T2, float>::value) {
        // 输出为 float 时，需要两个指针分别存储偶数/奇数部分，偏移量为 (s1BaseSize/2+1) * (s2BaseSize/2)
        x_expUb = expUb + ((s1BaseSize >> 1) + 1) * (s2BaseSize >> 1);
    }
    __ubuf__ pseShiftType * pseUb = (__ubuf__ pseShiftType*)pseTensor.GetPhyAddr();
    __ubuf__ T * expSumUb = (__ubuf__ T*)expSumTensor.GetPhyAddr();
    __ubuf__ T * maxUb = (__ubuf__ T*)maxTensor.GetPhyAddr();
    __ubuf__ T * maxUbStart = (__ubuf__ T*)maxTensor.GetPhyAddr(); // 保留起始位置用于后续广播加载
    __ubuf__ T * srcUb = (__ubuf__ T*)srcTensor.GetPhyAddr();
    __ubuf__ T * qScaleUb = (__ubuf__ T*)queryScaleUb.GetPhyAddr();
    __ubuf__ uint8_t * indexesUb = (__ubuf__ uint8_t*)indexesTensor.GetPhyAddr();
    __ubuf__ uint32_t * maskUb = (__ubuf__ uint32_t*)maskTensor.GetPhyAddr();           // mask 前半部分（按 uint32_t 解释）
    __ubuf__ uint32_t * maskUbUnroll = (__ubuf__ uint32_t*)(maskTensor.GetPhyAddr() + floatRepSize); // mask 后半部分
    __ubuf__ uint32_t *  dropMaskUb = (__ubuf__ uint32_t*)dropTensor.GetPhyAddr();

    const float dScale = scale * dScaleQK;  // 综合缩放因子

    // 调用向量函数实现具体计算
    ProcessVec1NoUpdateImpl128VF<T, T2, pseShiftType, s1BaseSize, s2BaseSize, hasAtten, pseMode, hasDrop, isMlaSgd, isMlaFullQuant>(
        expUb, x_expUb, pseUb, expSumUb, maxUb, maxUbStart, srcUb, qScaleUb, indexesUb, maskUb, maskUbUnroll, dropMaskUb, 
        divValue, blockStride, repeatStride, dScale, m, pseStride, slopes, posShift, scale, dScaleQK, minValue, deSCaleKValue);
}
} // namespace

#endif // VF_BASIC_BLOCK_ALIGNED128_NO_UPDATE_H