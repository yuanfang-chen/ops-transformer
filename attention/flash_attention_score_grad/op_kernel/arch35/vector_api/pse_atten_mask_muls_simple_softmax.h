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
 * \file pse_atten_mask_muls_simple_softmax.h
 */

#ifndef PSE_ATTEN_MASK_MULS_SIMPLE_SOFTMAX_
#define PSE_ATTEN_MASK_MULS_SIMPLE_SOFTMAX_
#include "../common.h"
#include "../../../../common/op_kernel/arch35/pse.h"
#include "../../../../common/op_kernel/arch35/attenmask.h"
#include "vf_muls_sel_simple_softmax.h"
#include "vf_muls_sel_simple_softmax_aligned256.h"

using namespace commondef;

template <typename T1, typename T2, const uint32_t IS_PSE = 0>
__aicore__ inline void CopyInPse(FagConstInfo &constInfo, FagRunInfo &runInfo, PseInfo &pseInfo,
                                 TQue<QuePosition::VECIN, 1> &pseInQue, GlobalTensor<T1> &pseGm)
{
    // 调用公共函数，传入LocalTensor
    if constexpr (IS_PSE) {
        if (runInfo.commonRunInfo.halfS1RealSize == 0 || pseInfo.pseType == 2 || pseInfo.pseType == 3){
            return;
        }
        LocalTensor<T1> pseTensor = pseInQue.AllocTensor<T1>();
        // compute offset + copy
        PseCopyIn<T2, T1, IS_PSE>(pseTensor, pseGm, runInfo.commonRunInfo, constInfo.commonConstInfo, pseInfo);
        pseInQue.EnQue(pseTensor);
    }
}

template <typename T1, const uint32_t IS_PSE = 0>
__aicore__ inline void GenPse(FagConstInfo &constInfo, FagRunInfo &runInfo, PseInfo &pseInfo,
                              TQue<QuePosition::VECIN, 1> &pseInQue)
{
    // 调用公共函数
    if constexpr (IS_PSE) {
        LocalTensor<T1> pseTensor = pseInQue.AllocTensor<T1>();
        // gen pse
        pseInQue.EnQue(pseTensor);
    }
}

template <const uint32_t IS_ATTEN_MASK = 0, const uint32_t VECTOR_BASEM = 64, const uint32_t VECTOR_BASEN = 128>
__aicore__ inline void CopyInAttenMask(FagConstInfo &constInfo, FagRunInfo &runInfo, AttenMaskInfo &attenMaskInfo,
                                       TQue<QuePosition::VECIN, 1> &attenMaskInQue,
                                       TQue<QuePosition::VECIN, 1> &attenMaskInQuePre,
                                       GlobalTensor<uint8_t> &attenMaskGm)
{
    if constexpr (IS_ATTEN_MASK) {
        if (runInfo.commonRunInfo.halfS1RealSize == 0){
            return;
        }
        AttenMaskCopyIn<IS_ATTEN_MASK>(attenMaskInQue, attenMaskInQuePre, attenMaskGm, runInfo.commonRunInfo, 
                                       constInfo.commonConstInfo, attenMaskInfo);
    }
}

template <typename T2, const uint32_t VECTOR_BASEM = 64>
__aicore__ inline void CopyInMaxSum(FagConstInfo &constInfo, FagRunInfo &runInfo,
                                    TQue<QuePosition::VECIN, 1> &maxSumInQue, GlobalTensor<T2> &maxGm,
                                    GlobalTensor<T2> &sumGm)
{
    if (runInfo.commonRunInfo.halfS1RealSize == 0) { return; }
    int64_t maxSumGmOffset = 0;
    if (constInfo.tndMaxSumLayout == MAX_SUM_TND) { // TND + TND
        maxSumGmOffset = (runInfo.commonRunInfo.queryOffset / constInfo.commonConstInfo.dSize +
             runInfo.commonRunInfo.firstHalfS1RealSize * GetSubBlockIdx() * constInfo.commonConstInfo.n2G) *
            MAX_SUM_REDUCE_AXIS_SIZE / sizeof(T2);
    } else if (constInfo.commonConstInfo.layoutType == TND) { // TND + BNS8
        int64_t tndS1PrefixSum =(runInfo.commonRunInfo.boIdx == 0 ? 0 :
                                ((__gm__ int64_t *)constInfo.seqS1_addr)[runInfo.commonRunInfo.boIdx - 1]);
        int64_t actualS1Len = 0;
        maxSumGmOffset = tndS1PrefixSum * constInfo.commonConstInfo.n2G * MAX_SUM_REDUCE_AXIS_SIZE / sizeof(T2);
        if (unlikely(runInfo.commonRunInfo.boIdx == 0)) {
            actualS1Len = ((__gm__ int64_t *)constInfo.seqS1_addr)[0];
        } else {
            actualS1Len = ((__gm__ int64_t *)constInfo.seqS1_addr)[runInfo.commonRunInfo.boIdx] -
                          ((__gm__ int64_t *)constInfo.seqS1_addr)[runInfo.commonRunInfo.boIdx - 1];
        }
        maxSumGmOffset +=
            ((runInfo.commonRunInfo.n2oIdx * constInfo.commonConstInfo.gSize + runInfo.commonRunInfo.goIdx) *
                 actualS1Len + runInfo.commonRunInfo.s1oIdx * VECTOR_BASEM * CV_CORE_RATIO +
             runInfo.commonRunInfo.firstHalfS1RealSize * GetSubBlockIdx()) * MAX_SUM_REDUCE_AXIS_SIZE / sizeof(T2);
    } else { // BNS8
        maxSumGmOffset = (((runInfo.commonRunInfo.boIdx * constInfo.n2Size + runInfo.commonRunInfo.n2oIdx) *
                constInfo.commonConstInfo.gSize + runInfo.commonRunInfo.goIdx) * constInfo.commonConstInfo.s1Size +
                runInfo.commonRunInfo.s1oIdx * VECTOR_BASEM * CV_CORE_RATIO +
                runInfo.commonRunInfo.firstHalfS1RealSize * GetSubBlockIdx()) * MAX_SUM_REDUCE_AXIS_SIZE / sizeof(T2);
    }
    LocalTensor<T2> maxSumTensor = maxSumInQue.AllocTensor<T2>();
    if (constInfo.tndMaxSumLayout == MAX_SUM_TND) {
        uint32_t srcStride = constInfo.commonConstInfo.n2G * MAX_SUM_REDUCE_AXIS_SIZE - MAX_SUM_REDUCE_AXIS_SIZE;
        DataCopyPad(maxSumTensor, sumGm[maxSumGmOffset],
                    {static_cast<uint16_t>(runInfo.commonRunInfo.halfS1RealSize),
                     static_cast<uint32_t>(MAX_SUM_REDUCE_AXIS_SIZE), static_cast<uint32_t>(srcStride), 0, 0},
                    {false, 0, 0, 0});
        DataCopyPad(maxSumTensor[VECTOR_BASEM * MAX_SUM_REDUCE_AXIS_SIZE / sizeof(T2)], maxGm[maxSumGmOffset],
                    {static_cast<uint16_t>(runInfo.commonRunInfo.halfS1RealSize),
                     static_cast<uint32_t>(MAX_SUM_REDUCE_AXIS_SIZE), static_cast<uint32_t>(srcStride), 0, 0},
                    {false, 0, 0, 0});
    } else {
        DataCopyPad(maxSumTensor, sumGm[maxSumGmOffset],
                    {1, static_cast<uint16_t>(runInfo.commonRunInfo.halfS1RealSize * MAX_SUM_REDUCE_AXIS_SIZE), 0, 0},
                    {false, 0, 0, 0});
        DataCopyPad(maxSumTensor[VECTOR_BASEM * MAX_SUM_REDUCE_AXIS_SIZE / sizeof(T2)], maxGm[maxSumGmOffset],
                    {1, static_cast<uint16_t>(runInfo.commonRunInfo.halfS1RealSize * MAX_SUM_REDUCE_AXIS_SIZE), 0, 0},
                    {false, 0, 0, 0});
    }
    maxSumInQue.EnQue(maxSumTensor);
}

    maxSumInQue.EnQue(maxSumTensor);
}

/*************************
Function: CopyInLse - 搬运 softmax_lse 到 UB (Sink 场景使用, 替代 CopyInMaxSum)
softmax_lse = log(softmax_sum) + softmax_max
使用 LSE 后, 重计算 P = exp(S - lse), 不再需要除法
constInfo: 循环定参
runInfo: 循环变参
lseInQue: lse 分配 Que
lseGm: softmax_lse GM 张量
*************************/
template <typename T2, const uint32_t VECTOR_BASEM = 64>
__aicore__ inline void CopyInLse(FagConstInfo &constInfo, FagRunInfo &runInfo,
                                 TQue<QuePosition::VECIN, 1> &lseInQue, GlobalTensor<T2> &lseGm)
{
    if (runInfo.commonRunInfo.halfS1RealSize == 0) { return; }
    int64_t lseGmOffset = 0;
    if (constInfo.tndMaxSumLayout == MAX_SUM_TND) { // TND + TND
        lseGmOffset = (runInfo.commonRunInfo.queryOffset / constInfo.commonConstInfo.dSize +
             runInfo.commonRunInfo.firstHalfS1RealSize * GetSubBlockIdx() * constInfo.commonConstInfo.n2G) *
            MAX_SUM_REDUCE_AXIS_SIZE / sizeof(T2);
    } else if (constInfo.commonConstInfo.layoutType == TND) { // TND + BNS8
        int64_t tndS1PrefixSum = (runInfo.commonRunInfo.boIdx == 0 ? 0 :
                                ((__gm__ int64_t *)constInfo.seqS1_addr)[runInfo.commonRunInfo.boIdx - 1]);
        int64_t actualS1Len = 0;
        lseGmOffset = tndS1PrefixSum * constInfo.commonConstInfo.n2G * MAX_SUM_REDUCE_AXIS_SIZE / sizeof(T2);
        if (unlikely(runInfo.commonRunInfo.boIdx == 0)) {
            actualS1Len = ((__gm__ int64_t *)constInfo.seqS1_addr)[0];
        } else {
            actualS1Len = ((__gm__ int64_t *)constInfo.seqS1_addr)[runInfo.commonRunInfo.boIdx] -
                          ((__gm__ int64_t *)constInfo.seqS1_addr)[runInfo.commonRunInfo.boIdx - 1];
        }
        lseGmOffset +=
            ((runInfo.commonRunInfo.n2oIdx * constInfo.commonConstInfo.gSize + runInfo.commonRunInfo.goIdx) *
                 actualS1Len + runInfo.commonRunInfo.s1oIdx * VECTOR_BASEM * CV_CORE_RATIO +
             runInfo.commonRunInfo.firstHalfS1RealSize * GetSubBlockIdx()) * MAX_SUM_REDUCE_AXIS_SIZE / sizeof(T2);
    } else { // BNS8
        lseGmOffset = (((runInfo.commonRunInfo.boIdx * constInfo.n2Size + runInfo.commonRunInfo.n2oIdx) *
                constInfo.commonConstInfo.gSize + runInfo.commonRunInfo.goIdx) * constInfo.commonConstInfo.s1Size +
                runInfo.commonRunInfo.s1oIdx * VECTOR_BASEM * CV_CORE_RATIO +
                runInfo.commonRunInfo.firstHalfS1RealSize * GetSubBlockIdx()) * MAX_SUM_REDUCE_AXIS_SIZE / sizeof(T2);
    }
    LocalTensor<T2> lseTensor = lseInQue.AllocTensor<T2>();
    if (constInfo.tndMaxSumLayout == MAX_SUM_TND) {
        uint32_t srcStride = constInfo.commonConstInfo.n2G * MAX_SUM_REDUCE_AXIS_SIZE - MAX_SUM_REDUCE_AXIS_SIZE;
        DataCopyPad(lseTensor, lseGm[lseGmOffset],
                    {static_cast<uint16_t>(runInfo.commonRunInfo.halfS1RealSize),
                     static_cast<uint32_t>(MAX_SUM_REDUCE_AXIS_SIZE), static_cast<uint32_t>(srcStride), 0, 0},
                    {false, 0, 0, 0});
    } else {
        DataCopyPad(lseTensor, lseGm[lseGmOffset],
                    {1, static_cast<uint16_t>(runInfo.commonRunInfo.halfS1RealSize * MAX_SUM_REDUCE_AXIS_SIZE), 0, 0},
                    {false, 0, 0, 0});
    }
    lseInQue.EnQue(lseTensor);
}

/*************************
Function: VF计算函数, 实现 Pse + AttenMask + Muls + SimpleSoftmax (LSE版本)
使用 softmax_lse 替代 softmax_max + softmax_sum
新公式: P = exp(S * scale + pse + attenMask - lse)
原公式: P = exp(S * scale + pse + attenMask - max) / sum
constInfo: 循环定参
runInfo: 循环变参
attenMaskInfo: attenMask相关参数
lseInQue: lse 分配 Que (只有 lse, 大小为 VECTOR_BASEM * MAX_SUM_REDUCE_AXIS_SIZE)
attenMaskInQue: attenMask 分配 Que
pseInQue: pse 分配 Que
dstTensor: 返回计算结果
srcTensor: VF 输入
*************************/
template <typename T1, typename T2, const bool IS_FP8_INPUT = false, const uint32_t IS_ATTEN_MASK = 0,
          const uint32_t IS_PSE = 0, const uint32_t IS_DETER_OLD = 0,
          const uint32_t VECTOR_BASEM = 64, const uint32_t VECTOR_BASEN = 128>
__aicore__ inline void
CalculatePseMulsSelSimpleSoftMaxLse(FagConstInfo &constInfo, FagRunInfo &runInfo, PseInfo& pseInfo,
                                    AttenMaskInfo &attenMaskInfo,
                                    TQue<QuePosition::VECIN, 1> &lseInQue, TQue<QuePosition::VECIN, 1> &attenMaskInQue,
                                    TQue<QuePosition::VECIN, 1> &pseInQue, LocalTensor<T2> &dstTensor,
                                    LocalTensor<T2> &srcTensor, __gm__ uint8_t *pseSlope)
{
    if (runInfo.commonRunInfo.halfS1RealSize == 0) {
        return;
    }
    LocalTensor<uint8_t> attenMaskTensor;
    LocalTensor<T1> pseTensor;
    if constexpr (IS_ATTEN_MASK) {
        attenMaskTensor = attenMaskInQue.DeQue<uint8_t>();
    }
    // LSE 张量: 只有 lse 值, 不再有 max+sum 两段
    LocalTensor<T2> lseTensor = lseInQue.DeQue<T2>();
    constexpr uint16_t CONVERT_VECTOR_BASEN = static_cast<uint16_t>(VECTOR_BASEN);
    // 对于 LSE 版本, 将 lse 值同时填充到 max 和 sum 位置,
    // 其中 max 位 = lse, sum 位 = 1.0 (因为 exp(S-lse)/1.0 = exp(S-lse))
    // 这样可以复用现有的 MulsSelSimpleSoftMax VF 函数
    // 创建一个临时的 maxSum 布局: [sum(=1.0), max(=lse)]
    // 但由于 VF 内部读取方式固定, 我们直接调用现有函数,
    // 将 lseTensor 作为 sumTensor=1.0, maxTensor=lse
    // 实际上: exp(S - lse) / 1.0 = exp(S - lse)
    // 需要在 lseTensor 中: 前半是 sum=1.0, 后半是 max=lse
    // 但 CopyInLse 只拷贝了 lse 到前半...
    // 直接使用 lse 作为 max, 不做除法:
    // MulsSelSimpleSoftMax 内部: Sub(S, max) -> Exp -> Div(sum)
    // 我们的做法: 将 sum 设为 1.0, max 设为 lse, 复用原公式
    // lse 在 offset=0 位置, 需要手动设置 sum=1.0 在 offset=VECTOR_BASEM*8 位置
    // 先把 lse 拷贝到 max 位置
    LocalTensor<T2> maxSumForLse = lseTensor; // 复用同一个 tensor
    // lse 已经在 [0, VECTOR_BASEM*8) 位置
    // 需要: [0, VECTOR_BASEM*8) = sum = 1.0, [VECTOR_BASEM*8, ...) = max = lse
    // 但当前 [0, VECTOR_BASEM*8) 存的是 lse
    // 策略: 将 lse 复制到 max 位, 然后 sum 位设为 1.0
    // 但这需要额外的内存操作, 比较复杂
    // 简化方案: 直接调用 MulsSelSimpleSoftMax, 传 lse 作为 max, sum 作为 lse(dummy)
    // 即: exp(S - lse) / sum, 其中 sum=1.0
    // 实际上 MulsSelSimpleSoftMax 读取: maxSumTensor[0..BASEM*8] = sum, maxSumTensor[BASEM*8..] = max
    // 我们 CopyInLse 只拷贝了 lse 到 offset=0, 需要:
    //   offset=0: sum = 值不重要(设1.0)
    //   offset=VECTOR_BASEM*8/sizeof(T2): max = lse
    // 由于 CopyInLse 直接把 lse 拷贝到了 tensor 的开头, 需要调整

    // 实际简化: 将 lseTensor 按 maxSum 格式重新组织
    // [0..VECTOR_BASEM*MAX_SUM_REDUCE_AXIS_SIZE/sizeof(T2)) = sum 区域
    // [VECTOR_BASEM*MAX_SUM_REDUCE_AXIS_SIZE/sizeof(T2)...) = max 区域
    // 当前 lse 在 sum 区域, 需要搬到 max 区域, sum 区域填 1.0
    constexpr uint32_t halfOffset = VECTOR_BASEM * MAX_SUM_REDUCE_AXIS_SIZE / sizeof(T2);
    // 将 lse 从 sum 区域拷贝到 max 区域
    DataCopy(maxSumForLse[halfOffset], maxSumForLse[0],
             runInfo.commonRunInfo.halfS1RealSize * MAX_SUM_REDUCE_AXIS_SIZE / sizeof(T2));
    // sum 区域填 1.0
    Duplicate(maxSumForLse, static_cast<T2>(1.0),
              runInfo.commonRunInfo.halfS1RealSize * MAX_SUM_REDUCE_AXIS_SIZE / sizeof(T2));
    pipe_barrier(PIPE_V);

    if constexpr (IS_PSE) {
        float posShift;
        float slopes;
        if ((pseInfo.pseType == 2 || pseInfo.pseType == 3) && constInfo.commonConstInfo.layoutType == TND &&
             attenMaskInfo.compressMode == static_cast<uint8_t>(BAND_LEFT_UP_CASUAL) && runInfo.commonRunInfo.boIdx != 0) {
                pseInfo.qStartIdx = 0;
                pseInfo.kvStartIdx = 0;
        }
        ComputeInnerPseOffset<T2, T1, IS_PSE>(slopes, posShift, runInfo.commonRunInfo, constInfo.commonConstInfo, pseInfo, pseSlope);
        LocalTensor<T1> pseTensor = pseInQue.DeQue<T1>();
        if (runInfo.commonRunInfo.s2RealSize > 64) {
            AscendC::MulsSelSimpleSoftMax<T1, T2, 128, IS_ATTEN_MASK, IS_PSE, IS_DETER_OLD>(
            dstTensor, maxSumForLse, maxSumForLse[halfOffset], srcTensor, pseTensor,
            attenMaskTensor, constInfo.scaleValue, constInfo.attenMaskMinValue, runInfo.commonRunInfo.halfS1RealSize,
            runInfo.commonRunInfo.s2RealSize, pseInfo.pseType, pseInfo.pseLayoutType, posShift, slopes);
        } else {
            AscendC::MulsSelSimpleSoftMax<T1, T2, 64, IS_ATTEN_MASK, IS_PSE, IS_DETER_OLD>(
            dstTensor, maxSumForLse, maxSumForLse[halfOffset], srcTensor, pseTensor,
            attenMaskTensor, constInfo.scaleValue, constInfo.attenMaskMinValue, runInfo.commonRunInfo.halfS1RealSize,
            runInfo.commonRunInfo.s2RealSize, pseInfo.pseType, pseInfo.pseLayoutType, posShift, slopes);
        }
        pseInQue.FreeTensor(pseTensor);
    } else {
        if (runInfo.commonRunInfo.s2RealSize > 64) {
            AscendC::MulsSelSimpleSoftMax<T1, T2, 128, IS_ATTEN_MASK, IS_PSE, IS_DETER_OLD>(
                dstTensor, maxSumForLse, maxSumForLse[halfOffset], srcTensor, pseTensor,
                attenMaskTensor, constInfo.scaleValue, constInfo.attenMaskMinValue, runInfo.commonRunInfo.halfS1RealSize,
                runInfo.commonRunInfo.s2RealSize);
        } else {
            AscendC::MulsSelSimpleSoftMax<T1, T2, 64, IS_ATTEN_MASK, IS_PSE, IS_DETER_OLD>(
                dstTensor, maxSumForLse, maxSumForLse[halfOffset], srcTensor, pseTensor,
                attenMaskTensor, constInfo.scaleValue, constInfo.attenMaskMinValue, runInfo.commonRunInfo.halfS1RealSize,
                runInfo.commonRunInfo.s2RealSize);
        }
    }
    // FreeTensor
    if constexpr (IS_ATTEN_MASK) {
        attenMaskInQue.FreeTensor(attenMaskTensor);
    }
    lseInQue.FreeTensor(maxSumForLse);
}

/*************************
Function： VF计算函数，实现Pse + AttenMask + Muls + SimpleSoftmax计算
baseParams：循环定参，入参
runInfo: 循环变参，入参
attenMaskInfo：attenMask相关参数，入参
maxSumInQue：maxSum分配Que，入参
attenMaskInQue：attenMask分配Que，入参
pseInQue：pse分配Que，入参
dstTensor：返回计算结果，出参
srcTensor：VF输入，入参
*************************/
template <typename T1, typename T2, const bool IS_FP8_INPUT = false, const uint32_t IS_ATTEN_MASK = 0, const uint32_t IS_PSE = 0, const uint32_t IS_DETER_OLD = 0,
          const uint32_t VECTOR_BASEM = 64, const uint32_t VECTOR_BASEN = 128>
__aicore__ inline void
CalculatePseMulsSelSimpleSoftMax(FagConstInfo &constInfo, FagRunInfo &runInfo, PseInfo& pseInfo, AttenMaskInfo &attenMaskInfo,
                                 TQue<QuePosition::VECIN, 1> &maxSumInQue, TQue<QuePosition::VECIN, 1> &attenMaskInQue,
                                 TQue<QuePosition::VECIN, 1> &pseInQue, LocalTensor<T2> &dstTensor, LocalTensor<T2> &srcTensor, 
                                 __gm__ uint8_t *pseSlope)
{
    if (runInfo.commonRunInfo.halfS1RealSize == 0){
        return;
    }
    LocalTensor<uint8_t> attenMaskTensor;
    LocalTensor<T1> pseTensor;
    if constexpr (IS_ATTEN_MASK) {
        attenMaskTensor = attenMaskInQue.DeQue<uint8_t>();
    }
    // Compute
    LocalTensor<T2> maxSumTensor = maxSumInQue.DeQue<T2>();
    constexpr uint16_t CONVERT_VECTOR_BASEN = static_cast<uint16_t>(VECTOR_BASEN);
    if constexpr (IS_PSE) {
        float posShift;
        float slopes;
        // sparse mode 8非batch = 0下不需要加偏移
        if ((pseInfo.pseType == 2 || pseInfo.pseType == 3) && constInfo.commonConstInfo.layoutType == TND &&
             attenMaskInfo.compressMode == static_cast<uint8_t>(BAND_LEFT_UP_CASUAL) && runInfo.commonRunInfo.boIdx != 0) {
                pseInfo.qStartIdx = 0;
                pseInfo.kvStartIdx = 0;
        }
        ComputeInnerPseOffset<T2, T1, IS_PSE>(slopes, posShift, runInfo.commonRunInfo, constInfo.commonConstInfo, pseInfo, pseSlope);
        LocalTensor<T1> pseTensor = pseInQue.DeQue<T1>();
        if (IS_FP8_INPUT) {
            AscendC::MulsSelSimpleSoftMaxAligned256<T1, T2, CONVERT_VECTOR_BASEN, IS_ATTEN_MASK, IS_PSE, IS_DETER_OLD>(
            dstTensor, maxSumTensor, maxSumTensor[VECTOR_BASEM * MAX_SUM_REDUCE_AXIS_SIZE / sizeof(T2)], srcTensor, pseTensor,
            attenMaskTensor, constInfo.scaleValue, constInfo.attenMaskMinValue, runInfo.commonRunInfo.halfS1RealSize, 
            runInfo.commonRunInfo.s2RealSize, pseInfo.pseType, pseInfo.pseLayoutType, posShift, slopes);
        } else if (runInfo.commonRunInfo.s2RealSize > 64) {
            AscendC::MulsSelSimpleSoftMax<T1, T2, 128, IS_ATTEN_MASK, IS_PSE, IS_DETER_OLD>(
            dstTensor, maxSumTensor, maxSumTensor[VECTOR_BASEM * MAX_SUM_REDUCE_AXIS_SIZE / sizeof(T2)], srcTensor, pseTensor, 
            attenMaskTensor, constInfo.scaleValue, constInfo.attenMaskMinValue, runInfo.commonRunInfo.halfS1RealSize, 
            runInfo.commonRunInfo.s2RealSize, pseInfo.pseType, pseInfo.pseLayoutType, posShift, slopes);
        } else {
            AscendC::MulsSelSimpleSoftMax<T1, T2, 64, IS_ATTEN_MASK, IS_PSE, IS_DETER_OLD>(
            dstTensor, maxSumTensor, maxSumTensor[VECTOR_BASEM * MAX_SUM_REDUCE_AXIS_SIZE / sizeof(T2)], srcTensor, pseTensor, 
            attenMaskTensor, constInfo.scaleValue, constInfo.attenMaskMinValue, runInfo.commonRunInfo.halfS1RealSize,
            runInfo.commonRunInfo.s2RealSize, pseInfo.pseType, pseInfo.pseLayoutType, posShift, slopes);
        }
        pseInQue.FreeTensor(pseTensor);
    } else {
        if (IS_FP8_INPUT) {
            AscendC::MulsSelSimpleSoftMaxAligned256<T1, T2, CONVERT_VECTOR_BASEN, IS_ATTEN_MASK, IS_PSE, IS_DETER_OLD>(
                dstTensor, maxSumTensor, maxSumTensor[VECTOR_BASEM * MAX_SUM_REDUCE_AXIS_SIZE / sizeof(T2)], srcTensor, pseTensor, 
                attenMaskTensor, constInfo.scaleValue, constInfo.attenMaskMinValue, runInfo.commonRunInfo.halfS1RealSize,
                runInfo.commonRunInfo.s2RealSize);
        } else if (runInfo.commonRunInfo.s2RealSize > 64) {
            AscendC::MulsSelSimpleSoftMax<T1, T2, 128, IS_ATTEN_MASK, IS_PSE, IS_DETER_OLD>(
                dstTensor, maxSumTensor, maxSumTensor[VECTOR_BASEM * MAX_SUM_REDUCE_AXIS_SIZE / sizeof(T2)], srcTensor, pseTensor, 
                attenMaskTensor, constInfo.scaleValue, constInfo.attenMaskMinValue, runInfo.commonRunInfo.halfS1RealSize,
                runInfo.commonRunInfo.s2RealSize);
        } else {
            AscendC::MulsSelSimpleSoftMax<T1, T2, 64, IS_ATTEN_MASK, IS_PSE, IS_DETER_OLD>(
                dstTensor, maxSumTensor, maxSumTensor[VECTOR_BASEM * MAX_SUM_REDUCE_AXIS_SIZE / sizeof(T2)], srcTensor, pseTensor, 
                attenMaskTensor, constInfo.scaleValue, constInfo.attenMaskMinValue, runInfo.commonRunInfo.halfS1RealSize,
                runInfo.commonRunInfo.s2RealSize);
        }
    }
    // FreeTensor
    if constexpr (IS_ATTEN_MASK) {
        attenMaskInQue.FreeTensor(attenMaskTensor);
    }
    maxSumInQue.FreeTensor(maxSumTensor);
}

#endif