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
 * \file weight_quant_vec_compute.h
 * \brief
 */
#ifndef GROUPED_MATMUL_WEIGHT_QUANT_VEC_COMPUTE_H
#define GROUPED_MATMUL_WEIGHT_QUANT_VEC_COMPUTE_H

#include "../block/basic_block_config.h"
#include "../tile/basic_block_vf_mx.h"
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#endif
#include "tool.h"

using AscendC::BLOCK_CUBE;
using AscendC::CacheMode;
using AscendC::DataCopyParams;
using AscendC::GlobalTensor;
using AscendC::HardEvent;
using AscendC::LocalTensor;
using AscendC::ONE_BLK_SIZE;
using AscendC::SetFlag;
using AscendC::TEventID;
using AscendC::VECTOR_REG_WIDTH;
using AscendC::WaitFlag;
namespace MicroAPI = AscendC::MicroAPI;
using AscendC::MicroAPI::AddrReg;
using AscendC::MicroAPI::GetRound;
using AscendC::MicroAPI::MaskReg;
using AscendC::MicroAPI::RegTensor;
using AscendC::MicroAPI::TypeGet;

namespace WeightQuantBatchMatmulV2::Arch35 {

#define GMM_WQ_VEC_ANTIQUANT_COMPUTE_BASIC_BLOCK_TEMPLATE_PARAM                                                        \
    template <typename xType, typename wType, typename antiQuantScaleType, typename biasType, typename yType,          \
              const WqmmConfig &wqmmConfig, const VecAntiQuantConfig &vecConfig>

#define GMM_WQ_VEC_ANTIQUANT_COMPUTE_BASIC_BLOCK_CLASS                                                                 \
    BasicBlockLibVectorAntiQuantCompute<xType, wType, antiQuantScaleType, biasType, yType, wqmmConfig, vecConfig>

GMM_WQ_VEC_ANTIQUANT_COMPUTE_BASIC_BLOCK_TEMPLATE_PARAM
class BasicBlockLibVectorAntiQuantCompute {
public:
    __aicore__ inline BasicBlockLibVectorAntiQuantCompute(){};
    __aicore__ inline BasicBlockLibVectorAntiQuantCompute(bool hasBias);

    __aicore__ inline void UpdateGlobalAddr(uint64_t mSize, uint64_t kSize, uint64_t nSize, __gm__ wType *weight, __gm__ biasType *bias,
                                            const bool weightL2Cacheable);
    __aicore__ inline void WaitVToMTE2();
    __aicore__ inline void SetVToMTE2();
    __aicore__ inline void CopyGmToUb(uint64_t ubMte2NSize, uint64_t ubMte2KSize, uint64_t ubMte2NOffset,
                                      uint64_t ubMte2KOffset, const BasicBlockOffsetParam &offsetParam);
    __aicore__ inline void CopyMxBiasGmToUb(uint64_t ubMte2MxBiasNSize, uint64_t ubMte2MxBiasNOffset);
    __aicore__ inline void WeightAntiQuantComputeNzNk(const UbConsumeConfig &ubConsumeConfig,
                                                      const LocalTensor<xType> &weightHighBitL1,
                                                      const L1ConsumeConfig &l1ConsumeConfig,
                                                      const LocalTensor<biasType> &biasL1);
    __aicore__ inline void End();

private:
    __aicore__ inline void InitMx();
    __aicore__ inline void CopyAntiQuantParamsGmToUb(uint64_t ubMte2NSize, uint64_t ubMte2KSize, uint64_t ubMte2NOffset,
                                                     uint64_t ubMte2KOffset, const BasicBlockOffsetParam &offsetParam);
    __aicore__ inline void AntiQuantProcess(uint64_t vfExternalRealLen, uint64_t vfInnerRealLen,
                                            uint64_t nWeightLowBitUbOffset, uint64_t kWeightLowBitUbOffset,
                                            const UbConsumeConfig &ubConsumeConfig);
    __aicore__ inline void WeightHighBitUbToL1(uint64_t weightHighBitL1Offset, uint64_t antiQuantRealN,
                                               uint64_t antiQuantRealK, const LocalTensor<xType> &weightHighBitL1,
                                               uint64_t l1RealExternalLen);
    __aicore__ inline uint64_t ComputeWeightHighBitL1Offset(uint64_t antiQuantNOffset, uint64_t antiQuantKOffset,
                                                            uint64_t nRealLen, uint64_t kRealLen,
                                                            const L1ConsumeConfig &l1ConsumeConfig);
    __aicore__ inline void AntiQuantProcessNzMxA8W4(const UbConsumeConfig &ubConsumeConfig);
    __aicore__ inline void CopyWeightHighBitForAligned(uint64_t weightHighBitL1Offset, uint64_t antiQuantRealN,
                                                       const uint64_t antiQuantRealK,
                                                       const LocalTensor<xType> &weightHighBitL1);
    __aicore__ inline void CopyWeightHighBitForUnaligned(uint64_t weightHighBitL1Offset, uint64_t antiQuantRealN,
                                                         uint64_t antiQuantRealK,
                                                         const LocalTensor<xType> &weightHighBitL1);

    // mte2搬运计数，用于控制weight输入的buffer和 mte2&&V间同步控制
    uint64_t ubMte2LoopIdx_ = 0;
    // mte2搬运计数，用于控制antiquantY输入的buffer和 mte2&&V间同步控制
    uint64_t ubMte2AntiquantYLoopIdx_ = 0;
    // vf中标准计算单元(vfNStandardLen, vfKStandardLen)的计数，用于控制weight反量化后输出的buffer和V&&mte3间同步控制
    uint64_t ubComputeLoopIdx_ = 0;
    constexpr static TEventID vecEventIdVToMte2_[QUADRUPLE_BUFFER_NUM] = {0, 1, 2, 3};
    constexpr static TEventID vecEventIdMte3ToV_[QUADRUPLE_BUFFER_NUM] = {0, 1, 2, 3};

    GlobalTensor<wType> wGlobal_;
    GlobalTensor<biasType> biasGlobal_;

    LocalTensor<int8_t> ubWeightInputLowBitTotalBuffer_;
    LocalTensor<xType> ubHighBitTotalBuffer_;
    LocalTensor<antiQuantScaleType> ubAntiQuantScaleTotalBuffer_;
    LocalTensor<xType> ubAntiQuantScaleAfterCastTotalBuffer_;
    LocalTensor<xType> ubAntiQuantOffsetTotalBuffer_;
    LocalTensor<biasType> ubBiasTotalBuffer_;
    LocalTensor<biasType> ubBiasOutTotalBuffer_;

    bool hasBias_;

    constexpr static uint32_t C0_SIZE = C0_SIZE_B8;
    constexpr static uint64_t VEC_REG_ELEM = VECTOR_REG_WIDTH;

    constexpr static uint64_t UB_ANTI_QUANT_Y_BUFFER_NUM = DOUBLE_BUFFER_NUM;

    TEventID vecEventIdAntiQuantYVToMte2_[UB_ANTI_QUANT_Y_BUFFER_NUM];

    constexpr static UbBufferInfo UB_BUFFER_INFO = GetMxA8W4NzBufferInfo(vecConfig);
};

GMM_WQ_VEC_ANTIQUANT_COMPUTE_BASIC_BLOCK_TEMPLATE_PARAM
__aicore__ inline void GMM_WQ_VEC_ANTIQUANT_COMPUTE_BASIC_BLOCK_CLASS::UpdateGlobalAddr(
    __gm__ wType *weight, __gm__ biasType *bias, const bool weightL2Cacheable)
{
    wGlobal_.SetGlobalBuffer(weight);

    if (!weightL2Cacheable) {
        wGlobal_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
    }
    if (hasBias_) {
        biasGlobal_.SetGlobalBuffer(bias);
    }
}

/*
 * 初始化buffer和同步所需的EventID
 */
GMM_WQ_VEC_ANTIQUANT_COMPUTE_BASIC_BLOCK_TEMPLATE_PARAM
__aicore__ inline GMM_WQ_VEC_ANTIQUANT_COMPUTE_BASIC_BLOCK_CLASS::BasicBlockLibVectorAntiQuantCompute(bool hasBias)
{
    hasBias_ = hasBias;
    InitMx();
}

/*
 * 初始化mx场景buffer 封装防止Init超长
 */
GMM_WQ_VEC_ANTIQUANT_COMPUTE_BASIC_BLOCK_TEMPLATE_PARAM
__aicore__ inline void GMM_WQ_VEC_ANTIQUANT_COMPUTE_BASIC_BLOCK_CLASS::InitMx()
{
    ubWeightInputLowBitTotalBuffer_ = LocalTensor<int8_t>(TPosition::LCM, 0, UB_BUFFER_INFO.weightInputLowbitUbTotalSize); // 16KB * 4 = 64KB
    ubHighBitTotalBuffer_ = LocalTensor<xType>(TPosition::LCM, 64 * GetKBUnit<int8_t>(), UB_BUFFER_INFO.highBitDataUbTotalSize);  // 32KB * 4 = 128KB
    if (hasBias_) {
        ubBiasTotalBuffer_ = LocalTensor<biasType>(TPosition::LCM, 192 * GetKBUnit<int8_t>(), UB_BUFFER_INFO.biasUbTotalSize);  // 2KB
        ubBiasOutTotalBuffer_ = LocalTensor<biasType>(TPosition::LCM, 194 * GetKBUnit<int8_t>(), UB_BUFFER_INFO.biasReducedUbTotalSize); // 2KB
    }
}

GMM_WQ_VEC_ANTIQUANT_COMPUTE_BASIC_BLOCK_TEMPLATE_PARAM
__aicore__ inline void GMM_WQ_VEC_ANTIQUANT_COMPUTE_BASIC_BLOCK_CLASS::WaitVToMTE2()
{
    if (likely(ubMte2LoopIdx_ > vecConfig.ubMte2BufferNum - 1)) {
        if constexpr (vecConfig.ubMte2BufferNum == 2 || vecConfig.ubMte2BufferNum == 4) {
            WaitFlag<HardEvent::V_MTE2>(vecEventIdVToMte2_[ubMte2LoopIdx_ & (vecConfig.ubMte2BufferNum - 1)]);
        } else {
            WaitFlag<HardEvent::V_MTE2>(vecEventIdVToMte2_[ubMte2LoopIdx_ % vecConfig.ubMte2BufferNum]);
        }
    }
}

GMM_WQ_VEC_ANTIQUANT_COMPUTE_BASIC_BLOCK_TEMPLATE_PARAM
__aicore__ inline void GMM_WQ_VEC_ANTIQUANT_COMPUTE_BASIC_BLOCK_CLASS::SetVToMTE2()
{
    if constexpr (vecConfig.ubMte2BufferNum == 2 || vecConfig.ubMte2BufferNum == 4) {
        SetFlag<HardEvent::V_MTE2>(vecEventIdVToMte2_[(ubMte2LoopIdx_ - 1) & (vecConfig.ubMte2BufferNum - 1)]);
    } else {
        SetFlag<HardEvent::V_MTE2>(vecEventIdVToMte2_[(ubMte2LoopIdx_ - 1) % vecConfig.ubMte2BufferNum]);
    }
}

GMM_WQ_VEC_ANTIQUANT_COMPUTE_BASIC_BLOCK_TEMPLATE_PARAM
__aicore__ inline void GMM_WQ_VEC_ANTIQUANT_COMPUTE_BASIC_BLOCK_CLASS::CopyGmToUb(
    uint64_t ubMte2NSize, uint64_t ubMte2KSize, uint64_t ubMte2NOffset, uint64_t ubMte2KOffset,
    const BasicBlockOffsetParam &offsetParam)
{
    // ubMte2NSize和ubMte2KSize为实际MTE2搬运到UB的有效数据，
    // 其按照ubMte2InnerSize进行跳写，垃圾数据无需操作，搬出的时搬运有效数据即可。
    if (ubMte2NSize == 0 || ubMte2KSize == 0) {
        ubMte2LoopIdx_++;  // 避免当前核无任务时，SetVToMTE2()对同一个flagID重复SetFlag的问题
        return;
    }

    DataCopyPad2D(ubWeightInputLowBitTotalBuffer_[(ubMte2LoopIdx_ % vecConfig.ubMte2BufferNum) *
                                                    UB_BUFFER_INFO.weightInputLowBitUbSingleBufferSize]
                        .template ReinterpretCast<wType>(),
                    wGlobal_[ubMte2KOffset * offsetParam.nAlign + ubMte2NOffset * static_cast<uint64_t>(C0_SIZE)],
                    CeilDivide(ubMte2KSize, static_cast<uint64_t>(C0_SIZE)),
                    CeilAlign(ubMte2NSize, static_cast<uint64_t>(BLOCK_CUBE)) * C0_SIZE,
                    CeilAlign(ubMte2NSize, static_cast<uint64_t>(BLOCK_CUBE)) * C0_SIZE,
                    offsetParam.nAlign * C0_SIZE);

    SetFlag<HardEvent::MTE2_V>(0);
    WaitFlag<HardEvent::MTE2_V>(0);
    ubMte2LoopIdx_++;
}

GMM_WQ_VEC_ANTIQUANT_COMPUTE_BASIC_BLOCK_TEMPLATE_PARAM
__aicore__ inline void GMM_WQ_VEC_ANTIQUANT_COMPUTE_BASIC_BLOCK_CLASS::CopyMxBiasGmToUb(uint64_t ubMte2MxBiasNSize,
                                                                                        uint64_t ubMte2MxBiasNOffset)
{
    if (hasBias_) {
        DataCopyPad2D(
            ubBiasTotalBuffer_[(ubMte2LoopIdx_ % QUADRUPLE_BUFFER_NUM) * UB_BUFFER_INFO.biasUbSingleBufferSize],
            biasGlobal_[ubMte2MxBiasNOffset], 1, ubMte2MxBiasNSize, ubMte2MxBiasNSize, ubMte2MxBiasNSize);
        SetFlag<HardEvent::MTE2_V>(1);
        WaitFlag<HardEvent::MTE2_V>(1);
    }
}

GMM_WQ_VEC_ANTIQUANT_COMPUTE_BASIC_BLOCK_TEMPLATE_PARAM
__aicore__ inline void GMM_WQ_VEC_ANTIQUANT_COMPUTE_BASIC_BLOCK_CLASS::WeightAntiQuantComputeNzNk(
    const UbConsumeConfig &ubConsumeConfig, const LocalTensor<xType> &weightHighBitL1,
    const L1ConsumeConfig &l1ConsumeConfig, const LocalTensor<biasType> &biasL1)
{
    if (likely(ubComputeLoopIdx_ > UB_BUFFER_INFO.ubWeightOutputHighBitBufferNum - 1)) {
        WaitFlag<HardEvent::MTE3_V>(
            vecEventIdMte3ToV_[ubComputeLoopIdx_ & (UB_BUFFER_INFO.ubWeightOutputHighBitBufferNum - 1)]);
    }

    AntiQuantProcessNzMxA8W4(ubConsumeConfig);

    uint64_t weightHighBitL1Offset = ComputeWeightHighBitL1Offset(
        0, 0, ubConsumeConfig.l1RequireVfComputeRealN, ubConsumeConfig.l1RequireVfComputeRealK, l1ConsumeConfig);

    SetFlag<HardEvent::V_MTE3>(0);
    WaitFlag<HardEvent::V_MTE3>(0);

    if (likely(ubConsumeConfig.l1RequireVfComputeRealK > 0)) {
        CopyWeightHighBitForAligned(weightHighBitL1Offset, ubConsumeConfig.l1RequireVfComputeRealN,
                                    ubConsumeConfig.l1RequireVfComputeRealK, weightHighBitL1);
    }
    if (ubConsumeConfig.calcMxBias) {
        DataCopy(biasL1[l1ConsumeConfig.l1MxBiasSplitNOffset],
                 ubBiasOutTotalBuffer_[((ubMte2LoopIdx_ - 1) & (vecConfig.ubMte2BufferNum - 1)) *
                                       UB_BUFFER_INFO.biasReducedSingleBufferSize],
                 ubConsumeConfig.ubMxBiasNsize);
    }
    SetFlag<HardEvent::MTE3_V>(
        vecEventIdMte3ToV_[ubComputeLoopIdx_ & (UB_BUFFER_INFO.ubWeightOutputHighBitBufferNum - 1)]);
    ubComputeLoopIdx_++;
}

GMM_WQ_VEC_ANTIQUANT_COMPUTE_BASIC_BLOCK_TEMPLATE_PARAM
__aicore__ inline uint64_t GMM_WQ_VEC_ANTIQUANT_COMPUTE_BASIC_BLOCK_CLASS::ComputeWeightHighBitL1Offset(
    uint64_t antiQuantNOffset, uint64_t antiQuantKOffset, uint64_t nRealLen, uint64_t kRealLen,
    const L1ConsumeConfig &l1ConsumeConfig)
{
    if constexpr (wqmmConfig.weightFormat != CubeFormat::NZ) {
        uint64_t l1RealExternalLenAlign =
            CeilAlign(l1ConsumeConfig.l1RealExternalLen, static_cast<uint64_t>(BLOCK_CUBE));
        if constexpr (!wqmmConfig.bTrans) {
            return l1RealExternalLenAlign * antiQuantNOffset +
                   (antiQuantKOffset + l1ConsumeConfig.l1SplitTwoVecExternalOffset) * static_cast<uint64_t>(BLOCK_CUBE);
        } else {
            return l1RealExternalLenAlign * antiQuantKOffset +
                   (antiQuantNOffset + l1ConsumeConfig.l1SplitTwoVecExternalOffset) * static_cast<uint64_t>(BLOCK_CUBE);
        }
    } else {
        if constexpr (!wqmmConfig.bTrans) {
            uint64_t kRealLenAlign = CeilAlign(kRealLen, static_cast<uint64_t>(BLOCK_CUBE));
            return kRealLenAlign * (antiQuantNOffset + l1ConsumeConfig.l1SplitTwoVecExternalOffset) +
                   antiQuantKOffset * static_cast<uint64_t>(C0_SIZE);
        } else {
            uint64_t nRealLenAlign = CeilAlign(nRealLen, static_cast<uint64_t>(BLOCK_CUBE));
            return nRealLenAlign * (antiQuantKOffset + l1ConsumeConfig.l1SplitTwoVecExternalOffset) +
                   antiQuantNOffset * static_cast<uint64_t>(C0_SIZE);
        }
    }
}

GMM_WQ_VEC_ANTIQUANT_COMPUTE_BASIC_BLOCK_TEMPLATE_PARAM
__aicore__ inline void GMM_WQ_VEC_ANTIQUANT_COMPUTE_BASIC_BLOCK_CLASS::AntiQuantProcessNzMxA8W4(
    const UbConsumeConfig &ubConsumeConfig)
{
    MxA8W4NzParams<xType, wType, biasType> mxA8W4NzParams;
    uint64_t ubMte2BufferIdx = (ubMte2LoopIdx_ - 1) & (vecConfig.ubMte2BufferNum - 1);
    mxA8W4NzParams.nRealSizeAlign = CeilAlign(ubConsumeConfig.l1RequireVfComputeRealN, static_cast<uint64_t>(BLOCK_CUBE));
    mxA8W4NzParams.weightLowBitPhyAddr =
        (__ubuf__ wType *)
            ubWeightInputLowBitTotalBuffer_[ubMte2BufferIdx * UB_BUFFER_INFO.weightInputLowBitUbSingleBufferSize]
                .GetPhyAddr();

    mxA8W4NzParams.weightHighBitPhyAddr =
        (__ubuf__ xType *)
            ubHighBitTotalBuffer_[(ubComputeLoopIdx_ & (UB_BUFFER_INFO.ubWeightOutputHighBitBufferNum - 1)) *
                                  VECTOR_REG_WIDTH]
                .GetPhyAddr();
    mxA8W4NzParams.loopKNum = CeilDivide(ubConsumeConfig.l1RequireVfComputeRealK, static_cast<uint64_t>(C0_SIZE));
    mxA8W4NzParams.innerLoopNum = CeilDivide(CeilAlign(ubConsumeConfig.l1RequireVfComputeRealN, static_cast<uint64_t>(BLOCK_CUBE)) * C0_SIZE,
                                          static_cast<uint64_t>(VECTOR_REG_WIDTH));
    // 跳写UB避免bank冲突
    mxA8W4NzParams.innerDstStride = VECTOR_REG_WIDTH * UB_BUFFER_INFO.ubWeightOutputHighBitBufferNum;
    mxA8W4NzParams.loopKDstStride = mxA8W4NzParams.innerLoopNum * mxA8W4NzParams.innerDstStride;
    if (ubConsumeConfig.calcMxBias) {
        mxA8W4NzParams.biasInUbAddr =
            (__ubuf__ biasType *)ubBiasTotalBuffer_[ubMte2BufferIdx * UB_BUFFER_INFO.biasUbSingleBufferSize]
                .GetPhyAddr();
        mxA8W4NzParams.biasOutUbAddr =
            (__ubuf__ biasType *)ubBiasOutTotalBuffer_[ubMte2BufferIdx * UB_BUFFER_INFO.biasReducedSingleBufferSize]
                .GetPhyAddr();
        if (ubConsumeConfig.isBiasSingleVector) {
            mxA8W4NzParams.biasLoopNum = CeilDivide(ubConsumeConfig.ubMxBiasNsize, VEC_MAX_ELEM_B16);
            AntiQuantMxA8W4NzNkVf<xType, wType, biasType, true, true>(mxA8W4NzParams);
        } else {
            AntiQuantMxA8W4NzNkVf<xType, wType, biasType, true, false>(mxA8W4NzParams);
        }
    } else {
        AntiQuantMxA8W4NzNkVf<xType, wType, biasType, false, false>(mxA8W4NzParams);
    }
}

GMM_WQ_VEC_ANTIQUANT_COMPUTE_BASIC_BLOCK_TEMPLATE_PARAM
__aicore__ inline void GMM_WQ_VEC_ANTIQUANT_COMPUTE_BASIC_BLOCK_CLASS::WeightHighBitUbToL1(
    uint64_t weightHighBitL1Offset, uint64_t antiQuantRealN, uint64_t antiQuantRealK,
    const LocalTensor<xType> &weightHighBitL1, uint64_t l1RealExternalLen)
{
    // 适用场景: s8s4 nz kn
    CopyWeightHighBitForAligned(weightHighBitL1Offset, antiQuantRealN, antiQuantRealK, weightHighBitL1);
}

GMM_WQ_VEC_ANTIQUANT_COMPUTE_BASIC_BLOCK_TEMPLATE_PARAM
__aicore__ inline void GMM_WQ_VEC_ANTIQUANT_COMPUTE_BASIC_BLOCK_CLASS::CopyWeightHighBitForAligned(
    uint64_t weightHighBitL1Offset, uint64_t antiQuantRealN, uint64_t antiQuantRealK,
    const LocalTensor<xType> &weightHighBitL1)
{
    DataCopyParams params;
    if constexpr (!wqmmConfig.bTrans) {
        params.blockCount = CeilAlign(antiQuantRealK, static_cast<uint64_t>(BLOCK_CUBE)) *
                            CeilAlign(antiQuantRealN, static_cast<uint64_t>(C0_SIZE)) / VEC_REG_ELEM;
    } else {
        params.blockCount = CeilAlign(antiQuantRealK, static_cast<uint64_t>(C0_SIZE)) *
                            CeilAlign(antiQuantRealN, static_cast<uint64_t>(BLOCK_CUBE)) / VEC_REG_ELEM;
    }

    params.blockLen = VEC_REG_ELEM / ONE_BLK_SIZE;
    params.srcStride = (UB_BUFFER_INFO.ubWeightOutputHighBitBufferNum - 1) * params.blockLen;
    params.dstStride = 0;  // dst地址连续
    DataCopy(
        weightHighBitL1[weightHighBitL1Offset],
        ubHighBitTotalBuffer_[(ubComputeLoopIdx_ & (UB_BUFFER_INFO.ubWeightOutputHighBitBufferNum - 1)) * VEC_REG_ELEM],
        params);
}

GMM_WQ_VEC_ANTIQUANT_COMPUTE_BASIC_BLOCK_TEMPLATE_PARAM
__aicore__ inline void GMM_WQ_VEC_ANTIQUANT_COMPUTE_BASIC_BLOCK_CLASS::CopyWeightHighBitForUnaligned(
    uint64_t weightHighBitL1Offset, uint64_t antiQuantRealN, uint64_t antiQuantRealK,
    const LocalTensor<xType> &weightHighBitL1)
{
    DataCopyParams params;
    // 跳写UB避免bank冲突，A16跳1024B; MTE3对应跳读
    uint64_t innerDstStride = VEC_MAX_ELEM_B16 * UB_BUFFER_INFO.ubWeightOutputHighBitBufferNum;
    uint64_t blockCubeAlignOfk = CeilAlign(antiQuantRealK, static_cast<uint64_t>(BLOCK_CUBE));
    uint64_t mxGroupSizeAlignOfk = CeilAlign(antiQuantRealK, static_cast<uint64_t>(MX_GROUPSIZE));
    for (int i = 0; i < CeilDivide(antiQuantRealN, static_cast<uint64_t>(C0_SIZE)); i++) {
        params.blockCount =
            CeilAlign(antiQuantRealK, static_cast<uint64_t>(BLOCK_CUBE)) / (VEC_MAX_ELEM_B16 / BLOCK_CUBE);
        params.blockLen = VEC_MAX_ELEM_B16 / BLOCK_CUBE;
        params.srcStride = (UB_BUFFER_INFO.ubWeightOutputHighBitBufferNum - 1) * params.blockLen;
        params.dstStride = 0; // dst地址连续
        weightHighBitL1Offset += i * blockCubeAlignOfk * BLOCK_CUBE;
        // dst需要再加i * 32对齐后的blockCount * innerDstStride作为地址偏置
        DataCopy(weightHighBitL1[weightHighBitL1Offset],
                 ubHighBitTotalBuffer_[(ubComputeLoopIdx_ & (UB_BUFFER_INFO.ubWeightOutputHighBitBufferNum - 1)) *
                                           VEC_MAX_ELEM_B16 +
                                       i * (mxGroupSizeAlignOfk / (VEC_MAX_ELEM_B16 / BLOCK_CUBE) * innerDstStride)],
                 params);
    }
}

GMM_WQ_VEC_ANTIQUANT_COMPUTE_BASIC_BLOCK_TEMPLATE_PARAM
__aicore__ inline void GMM_WQ_VEC_ANTIQUANT_COMPUTE_BASIC_BLOCK_CLASS::End()
{
    for (uint16_t idx = 0; idx < ubComputeLoopIdx_ && idx < UB_BUFFER_INFO.ubWeightOutputHighBitBufferNum; idx++) {
        WaitFlag<HardEvent::MTE3_V>(vecEventIdMte3ToV_[idx]);
    }

    for (uint16_t idx = 0; idx < ubMte2LoopIdx_ && idx < vecConfig.ubMte2BufferNum; idx++) {
        WaitFlag<HardEvent::V_MTE2>(vecEventIdVToMte2_[idx]);
    }
}
} // namespace WeightQuantBatchMatmulV2::Arch35

#endif // GROUPED_MATMUL_WEIGHT_QUANT_VEC_COMPUTE_H
