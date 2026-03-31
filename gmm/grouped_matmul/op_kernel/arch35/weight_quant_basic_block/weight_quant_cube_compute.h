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
 * \file weight_quant_cube_compute.h
 * \brief
 */
#ifndef GROUPED_MATMUL_WEIGHT_QUANT_CUBE_COMPUTE_H
#define GROUPED_MATMUL_WEIGHT_QUANT_CUBE_COMPUTE_H

#include "include/experimental/tensor_api/tensor.h"
#include "tile/tile_mmad_mx.h"
#include "tile/copy_scale_gm_to_l1.h"
#include "tile/copy_scale_l1_to_l0a.h"
#include "tile/copy_scale_l1_to_l0b.h"

using AscendC::Dn2NzParams;
using AscendC::GetBlockIdx;
using AscendC::GlobalTensor;
using AscendC::HardEvent;
using AscendC::IsSameType;
using AscendC::LocalTensor;
using AscendC::PipeBarrier;
using AscendC::SetFlag;
using AscendC::TPosition;
using AscendC::WaitFlag;
using AscendC::BLOCK_CUBE;
using AscendC::TEventID;

namespace WeightQuantBatchMatmulV2::Arch35 {

#define WQBMM_CUBE_COMPUTE_TEMPLATE_PARAM                                                                 \
    template <typename xType, typename biasType, typename antiQuantScaleType, typename perTokenScaleType, \
              typename yType, const WqmmConfig &wqmmConfig, typename MatmulImplType>

#define WQBMM_CUBE_COMPUTE_CLASS                                                                                   \
    WeightQuantBatchMatmulV2CubeCompute<xType, biasType, antiQuantScaleType, perTokenScaleType, yType, wqmmConfig, \
                                        MatmulImplType>

WQBMM_CUBE_COMPUTE_TEMPLATE_PARAM
class WeightQuantBatchMatmulV2CubeCompute {
public:
    __aicore__ inline WeightQuantBatchMatmulV2CubeCompute(){};
    __aicore__ inline void UpdateGlobalAddr(uint64_t mSize, uint64_t kSize, uint64_t nSize, __gm__ xType *x, __gm__ yType *y, __gm__ biasType *bias,
                                            __gm__ antiQuantScaleType *antiquantScale, __gm__ uint64_t *quantScale,
                                            __gm__ perTokenScaleType *perTokenScale, const bool isBias);
    __aicore__ inline void MxA8W4Init(uint64_t aPrefetchSize, uint64_t l1RemainSize, uint64_t l1StartSize,
                                      uint64_t mxBiasL1DbOffset, const TCubeTiling *__restrict matmulTiling,
                                      uint64_t biasL1Offset);
    __aicore__ inline void LaunchMatmul(uint64_t bL1Offset, int64_t kbOffset, uint64_t kbL1RealSize,
                                        uint64_t cvLoopIdx, const BasicBlockOffsetParam &param);
    __aicore__ inline void WaitMTE1ToMTE2(uint64_t kaGmOffset, const BasicBlockOffsetParam &offsetParam);
    __aicore__ inline void SetMTE1ToMTE2(uint64_t kaGmOffset, const BasicBlockOffsetParam &offsetParam);
    __aicore__ inline void WaitScaleMTE1ToMTE2(uint64_t kbGmOffset);
    __aicore__ inline void SetScaleMTE1ToMTE2(uint64_t kbGmOffset, const BasicBlockOffsetParam &offsetParam);
    __aicore__ inline void CopyAAndBiasGmToL1(const BasicBlockOffsetParam &param, int64_t kaGmOffset,
                                              uint64_t cvLoopIdx);
    __aicore__ inline void CopyMxScaleGmToL1(const BasicBlockOffsetParam &param, uint64_t kbL1Offset);
    __aicore__ inline void GetTensorC(const BasicBlockOffsetParam &param);
    __aicore__ inline void EndSync();
    __aicore__ inline void ClearAFullLoadFlag();
    __aicore__ inline void PrefetchA(uint64_t aPrefetchSize, uint64_t xSizeLimit);

private:
    using MakeLayoutA = typename AscendC::Te::NDLayoutFormat<xType>;
    using MakeLayoutScaleA = typename AscendC::Te::ScaleANDLayoutFormat<fp8_e8m0_t>;
    using MakeLayoutScaleB = typename AscendC::Te::ScaleBDNLayoutFormat<fp8_e8m0_t>;
    using MakeLayoutBias = typename AscendC::Te::NDLayoutFormat<biasType>;
    using MakeLayoutC = typename AscendC::Te::NDLayoutFormat<yType>;

    using MakeLayoutAL1 = AscendC::Te::NzLayoutFormat<xType>;
    using MakeLayoutBL1 = AscendC::Te::ZnLayoutFormat<xType>;
    using MakeLayoutScaleAL1 = typename AscendC::Te::ZzLayoutFormat<fp8_e8m0_t>;
    using MakeLayoutScaleBL1 = typename AscendC::Te::NnLayoutFormat<fp8_e8m0_t>;
    using MakeLayoutBiasL1 = typename AscendC::Te::NDLayoutFormat<biasType>;

    using MakeLayoutAL0 = AscendC::Te::NzLayoutFormat<xType>;
    using MakeLayoutBL0 = AscendC::Te::ZnLayoutFormat<xType>;
    using MakeLayoutScaleAL0 = typename AscendC::Te::ZzLayoutFormat<fp8_e8m0_t>;
    using MakeLayoutScaleBL0 = typename AscendC::Te::NnLayoutFormat<fp8_e8m0_t>;
    using MakeLayoutBt = typename AscendC::Te::NDLayoutFormat<biasType>;
    static constexpr uint64_t L0_BUF_NUM = 2;
    static constexpr uint64_t MXFP_DIVISOR_SIZE = 64;

    // PS 将AscendC::LocalTensor换成地址偏移
    __aicore__ inline void PrefetchA(uint64_t aPrefetchSize, uint64_t aGmSize, uint64_t aL1Offset);
    __aicore__ inline void InitSync();
    __aicore__ inline uint64_t CheckMaxSpace(const BasicBlockOffsetParam &param);
    __aicore__ inline void CopyAGmToL1SingleBuffer(const BasicBlockOffsetParam &param, int64_t kaGmOffset,
                                                   int64_t kbL1RealSize, int64_t biasRealN);
    __aicore__ inline void ConfigScaleDn2NzParams(uint64_t rowNum, uint64_t scaleKGmSize, uint64_t scaleKL1Stride,
                                                  uint64_t scaleKL1RealSize, Dn2NzParams &dn2NzParams);

    int8_t aL1DbNum_;
    bool isBias_;
    uint64_t quantScaleValue_;
    static constexpr uint32_t KB_UNIT = GetKBUnit<xType>();
    static constexpr uint64_t MX_SCALE_L1_SIZE = 32 * GetKBUnit<xType>() * sizeof(xType); // scaleA/B单块分配空间
    static constexpr uint64_t BIAS_TABLE_OFFSET_B32 = 2 * 256;
    static constexpr uint64_t MX_GROUP_SIZE = 32;

    uint64_t aL1Count_;
    uint64_t aL1MaxHalfCount_;
    uint64_t mxScaleBufIdx_ = 0;
    uint64_t aL1BufIdx_ = 0;

    AscendC::TEventID cubeEventIdsMxScaleMte1ToMte2_[DOUBLE_BUFFER_NUM];
    AscendC::TEventID cubeEventIdsMte1ToMte2_[DOUBLE_BUFFER_NUM];
    AscendC::TEventID cubeEventIdMte2ToMte1_;

    uint64_t aL1Offset_;
    uint64_t biasL1Offset_;
    uint64_t mxScaleAL1Offset_;
    uint64_t mxScaleBL1Offset_;

    uint64_t aL1DbOffset_;
    uint64_t biasL1DbOffset_;
    uint64_t mxScaleAL1DbOffset_;
    uint64_t mxScaleBL1DbOffset_;

    uint64_t l0LoopIdx_ = 0;

    __gm__ uint8_t *aPrefetchAddr_;

    // TODO 如何简化
    decltype(AscendC::Te::MakeTensor(AscendC::Te::MakeGMmemPtr(reinterpret_cast<__gm__ xType*>(0)), MakeLayoutA{}(16UL, 16UL))) gmA_;
    decltype(AscendC::Te::MakeTensor(AscendC::Te::MakeGMmemPtr(reinterpret_cast<__gm__ yType*>(0)), MakeLayoutC{}(16UL, 16UL))) gmC_;
    decltype(AscendC::Te::MakeTensor(AscendC::Te::MakeGMmemPtr(reinterpret_cast<__gm__ biasType*>(0)), MakeLayoutBias{}(1UL, 16UL))) gmBias_;
    decltype(AscendC::Te::MakeTensor(AscendC::Te::MakeGMmemPtr(reinterpret_cast<__gm__ fp8_e8m0_t*>(0)), MakeLayoutScaleA{}(16UL, 16UL))) gmScaleA_;
    decltype(AscendC::Te::MakeTensor(AscendC::Te::MakeGMmemPtr(reinterpret_cast<__gm__ fp8_e8m0_t*>(0)), MakeLayoutScaleB{}(16UL, 16UL))) gmScaleB_;

    decltype(AscendC::Te::MakeTensor(AscendC::Te::MakeL1memPtr(reinterpret_cast<__cbuf__ xType*>(0)), MakeLayoutAL1{}(16UL, 16UL))) tensorAL1_;
    decltype(AscendC::Te::MakeTensor(AscendC::Te::MakeL1memPtr(reinterpret_cast<__cbuf__ xType*>(0)), MakeLayoutBL1{}(16UL, 16UL))) tensorBL1_;
    decltype(AscendC::Te::MakeTensor(AscendC::Te::MakeL1memPtr(reinterpret_cast<__cbuf__ fp8_e8m0_t*>(0)), MakeLayoutScaleAL1{}(16UL, 16UL))) tensorScaleAL1_;
    decltype(AscendC::Te::MakeTensor(AscendC::Te::MakeL1memPtr(reinterpret_cast<__cbuf__ fp8_e8m0_t*>(0)), MakeLayoutScaleBL1{}(16UL, 16UL))) tensorScaleBL1_;
};

WQBMM_CUBE_COMPUTE_TEMPLATE_PARAM
__aicore__ inline uint64_t WQBMM_CUBE_COMPUTE_CLASS::CheckMaxSpace(const BasicBlockOffsetParam &param)
{
    uint64_t maxSpace = aL1MaxHalfCount_ * param.kbL1Size * CeilAlign(param.mL1Size, static_cast<uint64_t>(BLOCK_CUBE));
    if (param.kbL1Size > 0 && param.kSize % param.kbL1Size == 0 && !wqmmConfig.aTrans && maxSpace <= aL1DbOffset_) {
        return maxSpace;
    }
    return 0;
}

static constexpr uint64_t L0_BUF_OFFSET_B8 = 32 * 1024;
// unitFlag: 2表示累加中, 3表示累加结束
static constexpr uint16_t UNIT_FLAG_ENABLE = 2;
static constexpr uint16_t UNIT_FLAG_ENABLE_AUTO_CLOSE = 3;
static constexpr TEventID eventIdMToMte1_ = 3;
static constexpr TEventID eventIdMte1ToM_ = 3;

WQBMM_CUBE_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void WQBMM_CUBE_COMPUTE_CLASS::LaunchMatmul(uint64_t bL1Offset, int64_t kbOffset,
                                                              uint64_t kbL1RealSize, uint64_t cvLoopIdx,
                                                              const BasicBlockOffsetParam &param)
{
    SetFlag<HardEvent::MTE2_MTE1>(cubeEventIdMte2ToMte1_);
    WaitFlag<HardEvent::MTE2_MTE1>(cubeEventIdMte2ToMte1_);
    uint64_t aL1Offset = 0;
    if (aL1DbNum_ == SINGLE_BUFFER_NUM) {
        uint64_t maxSpace = CheckMaxSpace(param);
        if (maxSpace > 0) {
            // block = kbOffset / param.kbL1Size 计算是在第几块
            // blockOffset = block / 2 确定从A0还是A1读取数据后，在块内的偏移，单位是块
            // k = blockOffset * param.kbL1Size 当前块内的偏移量kOffset，单位是元素
            // 将block和blockOffset带入，计算k
            // k = (kbOffset / param.kbL1Size) / 2 * param.kbL1Size
            // 块内偏移量 = m * k
            // 举例：
            // L1A: |0|2|4|      |1|3|
            //      |A0:0~128KB  |A1:128KB~256KB|
            // 第5块（block = 5），在A1中偏移为3（blockOffset = 3），块内偏移量为m * (3 * k)
            aL1Offset = (aL1BufIdx_ & 1) * aL1DbOffset_ +
                        CeilAlign(param.mL1Size, static_cast<uint64_t>(BLOCK_CUBE)) *
                            (static_cast<uint64_t>(kbOffset) / (param.kbL1Size * 2) * param.kbL1Size);
        } else {
            aL1Offset = CeilAlign(param.mL1Size, static_cast<uint64_t>(BLOCK_CUBE)) * kbOffset;
        }
    } else {
        aL1Offset = (aL1BufIdx_ & 1) * aL1DbOffset_;
    }
    // auto coordScaleKL1 = Cgmct::Gemm::CeilDiv(static_cast<uint64_t>(kbOffset), MXFP_DIVISOR_SIZE) * 2;
    // auto tensorBlockScaleAL1 = tensorScaleAL1_(
    //     AscendC::Te::MakeCoord(0, coordScaleKL1),
    //     AscendC::Te::MakeShape(param.mL1Size, Cgmct::Gemm::CeilDiv(kbL1RealSize, MXFP_DIVISOR_SIZE) * 2));
    // auto tensorBlockScaleBL1 = tensorScaleBL1_(
    //     AscendC::Te::MakeCoord(coordScaleKL1, 0),
    //     AscendC::Te::MakeShape(Cgmct::Gemm::CeilDiv(kbL1RealSize, MXFP_DIVISOR_SIZE) * 2, param.nL1Size));
    // AscendC::printf("====== mL1 %d nL1 %d ========", param.mL1Size, param.nL1Size);
    auto layoutBL1 = MakeLayoutBL1{}(kbL1RealSize, param.nL1Size);
    tensorBL1_ = AscendC::Te::MakeTensor(AscendC::Te::MakeL1memPtr<xType>(bL1Offset), layoutBL1);
    bool isLastGmK = kbOffset + kbL1RealSize >= param.kSize;
    bool isFirstGmK = kbOffset == 0;
    uint64_t l0KSize = (param.mL1Size <= 128 && param.nL1Size <= 128) ? 256 : 128;
    for (uint64_t l1KOffset = 0; l1KOffset < kbL1RealSize; l1KOffset += l0KSize) {
        bool isLastL1K = l1KOffset + l0KSize >= kbL1RealSize;
        uint64_t realL0k = isLastL1K ? kbL1RealSize - l1KOffset : l0KSize;
        uint64_t loopId = l0LoopIdx_ % L0_BUF_NUM; // 等价于 l0LoopIdx_ % L0_BUF_NUM，减少scalar
        WaitFlag<HardEvent::M_MTE1>(eventIdMToMte1_ + loopId);
        
        // LoadAL1ToL0A(realL0k, loopId, l1KOffset, basicApiParams, aL1, aMxScaleL1);
        auto CopyL12L0 = AscendC::Te::MakeCopy(AscendC::Te::CopyL12L0{});
        auto layoutAL0 = MakeLayoutAL0{}(param.mL1Size, realL0k);
        auto tensorAL0 = AscendC::Te::MakeTensor(AscendC::Te::MakeL0AmemPtr<xType>(loopId * L0_BUF_OFFSET_B8), layoutAL0);
        // PS 考虑kaL1Size >= kbL1Size的场景
        auto tensorBlockAL1 = tensorAL1_(AscendC::Te::MakeCoord(0, (l1KOffset+kbOffset)%param.kaL1Size), AscendC::Te::MakeShape(param.mL1Size, realL0k));
        // PS 同旧的差异是外部地址偏移好，旧方法是通过startposition处理
        // AscendC::printf("(k1,m1,m0,k0) (%d,%d,%d,%d) offset %d startposition %d", Cgmct::Gemm::CeilDiv(realL0k, 32), Cgmct::Gemm::CeilDiv(param.mL1Size, 16), 16, 32, Cgmct::Gemm::CeilDiv((l1KOffset+kbOffset)%param.kaL1Size, 32)*32*Cgmct::Gemm::CeilDiv(param.mL1Size, 16)*16, Cgmct::Gemm::CeilDiv((l1KOffset+kbOffset)%param.kaL1Size, 32));
        AscendC::Te::Copy(CopyL12L0, tensorAL0, tensorBlockAL1);

        // scaleA, scaleB L1->L0
        // TODO scaleA 在L0A?
        auto layoutScaleAL0 = MakeLayoutScaleAL0{}(
            param.mL1Size, Cgmct::Gemm::CeilDiv(realL0k, MXFP_DIVISOR_SIZE) * 2);
        auto tensorScaleAL0 =
            AscendC::Te::MakeTensor(AscendC::Te::MakeL0AmemPtr<fp8_e8m0_t>(loopId * L0_BUF_OFFSET_B8), layoutScaleAL0);
        auto CopyL12L0MxScaleA3510 = AscendC::Te::MakeCopy(Cgmct::Gemm::Tile::CopyL12L0MxScaleA3510{});
        // AscendC::printf("(k1,n1,n0,k0) (%d,%d,%d,%d) offset %d startposition %d", Cgmct::Gemm::CeilDiv(realL0k, 32), Cgmct::Gemm::CeilDiv(param.nL1Size, 16), 16, 32, Cgmct::Gemm::CeilDiv(l1KOffset, 32)*32*Cgmct::Gemm::CeilDiv(param.nL1Size, 16)*16, Cgmct::Gemm::CeilDiv(l1KOffset, 32));
        // PS scale多倍载入L1
        AscendC::Te::Copy(
            CopyL12L0MxScaleA3510, tensorScaleAL0, tensorScaleAL1_,
            AscendC::Te::MakeCoord(0, ((l1KOffset + kbOffset) % MX_SCALE_K_L1_SIZE) / MX_GROUP_SIZE));

        auto layoutBL0 = MakeLayoutBL0{}(realL0k, param.nL1Size);
        auto tensorBL0 = AscendC::Te::MakeTensor(AscendC::Te::MakeL0BmemPtr<xType>(loopId * L0_BUF_OFFSET_B8), layoutBL0);
        auto tensorBlockBL1 = tensorBL1_(
            AscendC::Te::MakeCoord(l1KOffset, 0), AscendC::Te::MakeShape(realL0k, param.nL1Size));
        AscendC::Te::Copy(CopyL12L0, tensorBL0, tensorBlockBL1);

        auto layoutScaleBL0 = MakeLayoutScaleBL0{}(
            Cgmct::Gemm::CeilDiv(realL0k, MXFP_DIVISOR_SIZE) * 2, param.nL1Size);
        auto tensorScaleBL0 =
            AscendC::Te::MakeTensor(AscendC::Te::MakeL0BmemPtr<fp8_e8m0_t>(loopId * L0_BUF_OFFSET_B8), layoutScaleBL0);
        auto CopyL12L0MxScaleB3510 = AscendC::Te::MakeCopy(Cgmct::Gemm::Tile::CopyL12L0MxScaleB3510{});
        AscendC::Te::Copy(
            CopyL12L0MxScaleB3510, tensorScaleBL0, tensorScaleBL1_,
            AscendC::Te::MakeCoord(((l1KOffset + kbOffset) % MX_SCALE_K_L1_SIZE) / MX_GROUP_SIZE, 0));

        if (isBias_ && isFirstGmK && l1KOffset == 0) {            
            auto CopyL12BT = AscendC::Te::MakeCopy(AscendC::Te::CopyL12BT{});
            auto layoutBt = MakeLayoutBt{}(1L, Cgmct::Gemm::Align(param.nL1Size, AscendC::BLOCK_CUBE));
            auto tensorBt = AscendC::Te::MakeTensor(
                AscendC::Te::MakeBiasmemPtr<float>(loopId * BIAS_TABLE_OFFSET_B32 * sizeof(float)), layoutBt);
            auto layoutBiasL1 = MakeLayoutBias{}(1L, Cgmct::Gemm::Align(param.nL1Size, AscendC::BLOCK_CUBE));
            auto tensorBiasL1 = AscendC::Te::MakeTensor(
                AscendC::Te::MakeL1memPtr<biasType>(biasL1Offset_ + (cvLoopIdx & 1) * biasL1DbOffset_ * sizeof(biasType)), layoutBiasL1);
            AscendC::Te::Copy(CopyL12BT, tensorBt, tensorBiasL1);
        }

        bool isFirstK = isFirstGmK && l1KOffset == 0;
        // Mmad(isLastGmK && isLastL1K, isFirstGmK && l1KOffset == 0, realL0k, loopId, basicApiParams);
        SetFlag<HardEvent::MTE1_M>(eventIdMte1ToM_);
        WaitFlag<HardEvent::MTE1_M>(eventIdMte1ToM_);
        uint8_t mmadUnitFlag = (isLastGmK && isLastL1K) ? UNIT_FLAG_ENABLE_AUTO_CLOSE : UNIT_FLAG_ENABLE;

        auto layoutL0C = AscendC::Te::MakeL0CLayout(param.mL1Size, param.nL1Size);
        auto tensorL0C = AscendC::Te::MakeTensor(AscendC::Te::MakeL0CmemPtr<float>(0), layoutL0C);
        if (isBias_ && isFirstK) {
            auto layoutBt =
                MakeLayoutBt{}(1L, Cgmct::Gemm::Align(param.nL1Size, AscendC::BLOCK_CUBE));
            auto tensorBt = AscendC::Te::MakeTensor(
                AscendC::Te::MakeBiasmemPtr<float>(loopId * BIAS_TABLE_OFFSET_B32 * sizeof(float)), layoutBt);
            AscendC::Te::Mad(
                        AscendC::Te::MmadAtom<AscendC::Te::MmadTraits<Cgmct::Gemm::Tile::MmadMxWithBias>>{}.with(
                            static_cast<uint16_t>(param.mL1Size),
                            static_cast<uint16_t>(realL0k),
                            static_cast<uint16_t>(param.nL1Size), mmadUnitFlag, /*btBuffCtrl*/true, /*initCMatrixCtrl*/false),
                        tensorL0C, tensorAL0, tensorBL0, tensorBt);
        } else {
            AscendC::Te::Mad(
                        AscendC::Te::MmadAtom<AscendC::Te::MmadTraits<Cgmct::Gemm::Tile::MmadMx>>{}.with(
                            static_cast<uint16_t>(param.mL1Size),
                            static_cast<uint16_t>(realL0k),
                            static_cast<uint16_t>(param.nL1Size), mmadUnitFlag, /*btBuffCtrl*/false, /*initCMatrixCtrl*/isFirstK),
                        tensorL0C, tensorAL0, tensorBL0);
        }

        SetFlag<HardEvent::M_MTE1>(eventIdMToMte1_ + loopId);
        l0LoopIdx_++;
    }
}

WQBMM_CUBE_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void WQBMM_CUBE_COMPUTE_CLASS::WaitMTE1ToMTE2(uint64_t kaGmOffset,
                                                                const BasicBlockOffsetParam &offsetParam)
{
    if (aL1DbNum_ > SINGLE_BUFFER_NUM && kaGmOffset % offsetParam.kaL1Size == 0) {
        WaitFlag<HardEvent::MTE1_MTE2>(cubeEventIdsMte1ToMte2_[aL1BufIdx_ & 1]);
    }
}

WQBMM_CUBE_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void WQBMM_CUBE_COMPUTE_CLASS::WaitScaleMTE1ToMTE2(uint64_t kbGmOffset)
{
    if (kbGmOffset % MX_SCALE_K_L1_SIZE == 0) {
        WaitFlag<HardEvent::MTE1_MTE2>(cubeEventIdsMxScaleMte1ToMte2_[mxScaleBufIdx_ & 1]);
    }
}

WQBMM_CUBE_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void WQBMM_CUBE_COMPUTE_CLASS::SetScaleMTE1ToMTE2(uint64_t kbGmOffset,
                                                                    const BasicBlockOffsetParam &offsetParam)
{
    if ((kbGmOffset + offsetParam.kbL1Size) % MX_SCALE_K_L1_SIZE == 0 ||
        kbGmOffset + offsetParam.kbL1Size >= offsetParam.kSize) {
        SetFlag<HardEvent::MTE1_MTE2>(cubeEventIdsMxScaleMte1ToMte2_[mxScaleBufIdx_ & 1]);
        mxScaleBufIdx_++;
    }
}

WQBMM_CUBE_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void WQBMM_CUBE_COMPUTE_CLASS::SetMTE1ToMTE2(uint64_t kaGmOffset,
                                                               const BasicBlockOffsetParam &offsetParam)
{
    if (aL1DbNum_ > SINGLE_BUFFER_NUM && ((kaGmOffset + offsetParam.kbL1Size) % offsetParam.kaL1Size == 0 ||
                                          kaGmOffset + offsetParam.kbL1Size >= offsetParam.kSize)) {
        SetFlag<HardEvent::MTE1_MTE2>(cubeEventIdsMte1ToMte2_[aL1BufIdx_ & 1]);
        aL1BufIdx_++;
    }
}

WQBMM_CUBE_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void WQBMM_CUBE_COMPUTE_CLASS::CopyAGmToL1SingleBuffer(const BasicBlockOffsetParam &param,
                                                                         int64_t kaGmOffset, int64_t kbL1RealSize,
                                                                         int64_t biasRealN)
{
    AscendC::Nd2NzParams nd2nzParams;
    uint64_t maxSpace = CheckMaxSpace(param);
    if (maxSpace > 0) {
        // TODO 不支持
        // nd2nzParams.ndNum = aL1MaxHalfCount_;
        // nd2nzParams.nValue = param.mL1Size;
        // nd2nzParams.dValue = param.kbL1Size;
        // nd2nzParams.srcDValue = param.kSize;
        // nd2nzParams.srcNdMatrixStride = 2 * nd2nzParams.dValue;
        // nd2nzParams.dstNzC0Stride = CeilAlign(nd2nzParams.nValue, static_cast<uint16_t>(BLOCK_CUBE));
        // nd2nzParams.dstNzNStride = 1;
        // nd2nzParams.dstNzMatrixStride =
        //     nd2nzParams.dstNzC0Stride * CeilAlign(nd2nzParams.dValue, static_cast<uint32_t>(BLOCK_CUBE));
        // DataCopy(aL1_[(aL1BufIdx_ & 1) * aL1DbOffset_], xGlobal_[aGmOffset], nd2nzParams);

        // nd2nzParams.ndNum = aL1Count_ - aL1MaxHalfCount_;
        // DataCopy(aL1_[((aL1BufIdx_ + 1) & 1) * aL1DbOffset_], xGlobal_[aGmOffset + nd2nzParams.dValue], nd2nzParams);
    } else {
        auto copyGM2L1 = AscendC::Te::MakeCopy(AscendC::Te::CopyGM2L1{});
        auto layoutAL1 = MakeLayoutAL1{}(param.mL1Size, param.kSize);
        tensorAL1_ = AscendC::Te::MakeTensor(AscendC::Te::MakeL1memPtr<xType>(0), layoutAL1);
        auto gmBlockA =
            gmA_(AscendC::Te::MakeCoord(param.mOffset, kaGmOffset),
                AscendC::Te::MakeShape(param.mL1Size, param.kSize));
        AscendC::Te::Copy(copyGM2L1, tensorAL1_, gmBlockA);
    }
}

WQBMM_CUBE_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void WQBMM_CUBE_COMPUTE_CLASS::CopyAAndBiasGmToL1(const BasicBlockOffsetParam &param,
                                                                    int64_t kaGmOffset, uint64_t cvLoopIdx)
{
    if (kaGmOffset % param.kaL1Size != 0) {
        return;
    }
    int64_t kaL1RealSize = (kaGmOffset + param.kaL1Size) >= param.kSize ? param.kSize - kaGmOffset : param.kaL1Size;
    auto copyGM2L1 = AscendC::Te::MakeCopy(AscendC::Te::CopyGM2L1{});
    if (aL1DbNum_ > SINGLE_BUFFER_NUM) {
        auto layoutAL1 = MakeLayoutAL1{}(param.mL1Size, kaL1RealSize);
        auto gmBlockA =
            gmA_(AscendC::Te::MakeCoord(param.mOffset, kaGmOffset),
                AscendC::Te::MakeShape(param.mL1Size, kaL1RealSize));
        tensorAL1_ = AscendC::Te::MakeTensor(AscendC::Te::MakeL1memPtr<xType>(aL1Offset_ + (aL1BufIdx_ & 1) * aL1DbOffset_), layoutAL1);
        AscendC::Te::Copy(copyGM2L1, tensorAL1_, gmBlockA);
    } else if (aL1DbNum_ == SINGLE_BUFFER_NUM && kaGmOffset == 0) {
        CopyAGmToL1SingleBuffer(param, kaGmOffset, kaL1RealSize, param.nL1Size);
    }
}

WQBMM_CUBE_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void WQBMM_CUBE_COMPUTE_CLASS::CopyMxScaleGmToL1(const BasicBlockOffsetParam &param,
                                                                   uint64_t kbL1Offset)
{
    if (kbL1Offset % MX_SCALE_K_L1_SIZE != 0) {
        return;
    }
    uint64_t scaleKGmSize = param.kSize / MX_GROUPSIZE;
    // 当前scaleFactor为1，暂不考虑scaleFactor相关计算
    uint64_t scaleKL1StandardLen = MX_SCALE_K_L1_SIZE / MX_GROUPSIZE;
    uint64_t scaleKL1RealSize = (kbL1Offset + MX_SCALE_K_L1_SIZE) > param.kSize ?
                                    (param.kSize - kbL1Offset) / MX_GROUPSIZE :
                                    scaleKL1StandardLen;
    // L1上的K需要填完整的K（scaleKL1_），不能是尾块，GM上的填实际大小（可能是尾块）
    auto CopyScaleGM2L1 = AscendC::Te::MakeCopy(Cgmct::Gemm::Tile::CopyScaleGM2L1{});
    auto layoutScaleAL1 = MakeLayoutScaleAL1{}(param.mL1Size, scaleKL1RealSize);
    // PS 旧方式固定dst stride（dstNzC0Stride）为64
    tensorScaleAL1_ = AscendC::Te::MakeTensor(
        AscendC::Te::MakeL1memPtr<fp8_e8m0_t>(mxScaleAL1Offset_ + (mxScaleBufIdx_ & 1) * mxScaleAL1DbOffset_), layoutScaleAL1);
    auto gmBlockScaleA = gmScaleA_(
        AscendC::Te::MakeCoord(param.mOffset, kbL1Offset / MX_GROUP_SIZE),
        AscendC::Te::MakeShape(
            param.mL1Size,
            scaleKL1RealSize));
    AscendC::Te::Copy(CopyScaleGM2L1, tensorScaleAL1_, gmBlockScaleA);

    auto layoutScaleBL1 = MakeLayoutScaleBL1{}(scaleKL1RealSize, param.nL1Size);
    tensorScaleBL1_ = AscendC::Te::MakeTensor(
        AscendC::Te::MakeL1memPtr<fp8_e8m0_t>(mxScaleBL1Offset_ + (mxScaleBufIdx_ & 1) * mxScaleBL1DbOffset_), layoutScaleBL1);
    auto gmBlockScaleB = gmScaleB_(
        AscendC::Te::MakeCoord(kbL1Offset / MX_GROUP_SIZE, param.nOffset),
        AscendC::Te::MakeShape(
            scaleKL1RealSize,
            param.nL1Size));
    AscendC::Te::Copy(CopyScaleGM2L1, tensorScaleBL1_, gmBlockScaleB);
}

WQBMM_CUBE_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void WQBMM_CUBE_COMPUTE_CLASS::ConfigScaleDn2NzParams(uint64_t rowNum, uint64_t scaleKGmSize,
                                                                        uint64_t scaleKL1Stride,
                                                                        uint64_t scaleKL1RealSize,
                                                                        Dn2NzParams &dn2NzParams)
{
    dn2NzParams.dnNum = 1;
    dn2NzParams.dValue = rowNum;  // 矩阵的行数，即待搬运的mxScaleA的m或mxScaleB的n
    dn2NzParams.nValue = CeilDivide(scaleKL1RealSize, SCALE_COPY_GROUP_SIZE);  // 矩阵的列数，使用B16搬B8需要除以2向上取整
    dn2NzParams.srcDnMatrixStride = SCALE_COPY_DEFAULT_STRIDE;
    dn2NzParams.srcDValue = CeilDivide(scaleKGmSize, SCALE_COPY_GROUP_SIZE);  // 源矩阵一行所含B16元素个数
    // 目标矩阵行方向两个相邻分形起始地址之间的间隔，单位32B
    dn2NzParams.dstNzC0Stride = CeilDivide(MX_SCALE_K_L1_SIZE, MX_GROUPSIZE * SCALE_COPY_GROUP_SIZE);
    // 目标矩阵列方向两个相邻分形起始地址之间的间隔，单位32B
    dn2NzParams.dstNzNStride = SCALE_COPY_DEFAULT_N_STRIDE;
    dn2NzParams.dstNzMatrixStride = SCALE_COPY_DEFAULT_STRIDE;
}

WQBMM_CUBE_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void WQBMM_CUBE_COMPUTE_CLASS::EndSync()
{
    for (uint64_t i = 0; i < L0_BUF_NUM; i++) {
        WaitFlag<HardEvent::M_MTE1>(eventIdMToMte1_ + i);
    }

    for (uint64_t i = 0; i < DOUBLE_BUFFER_NUM; i++) {
        WaitFlag<HardEvent::MTE1_MTE2>(cubeEventIdsMxScaleMte1ToMte2_[i]);
        if (aL1DbNum_ > SINGLE_BUFFER_NUM) {
            WaitFlag<HardEvent::MTE1_MTE2>(cubeEventIdsMte1ToMte2_[i]);
        }
    }
}

WQBMM_CUBE_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void WQBMM_CUBE_COMPUTE_CLASS::ClearAFullLoadFlag()
{
    if (aL1DbNum_ == SINGLE_BUFFER_NUM) {
        SetFlag<HardEvent::MTE1_MTE2>(cubeEventIdsMte1ToMte2_[0]);
        WaitFlag<HardEvent::MTE1_MTE2>(cubeEventIdsMte1ToMte2_[0]);
    }
}

WQBMM_CUBE_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void WQBMM_CUBE_COMPUTE_CLASS::InitSync()
{
    for (uint64_t i = 0; i < DOUBLE_BUFFER_NUM; i++) {
        cubeEventIdsMte1ToMte2_[i] = GetTPipePtr()->AllocEventID<HardEvent::MTE1_MTE2>();
        cubeEventIdsMxScaleMte1ToMte2_[i] = GetTPipePtr()->AllocEventID<HardEvent::MTE1_MTE2>();
        SetFlag<HardEvent::MTE1_MTE2>(cubeEventIdsMxScaleMte1ToMte2_[i]);
        if (aL1DbNum_ > SINGLE_BUFFER_NUM) {
            SetFlag<HardEvent::MTE1_MTE2>(cubeEventIdsMte1ToMte2_[i]);
        }
    }
    cubeEventIdMte2ToMte1_ = GetTPipePtr()->AllocEventID<HardEvent::MTE2_MTE1>();
}



WQBMM_CUBE_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void WQBMM_CUBE_COMPUTE_CLASS::UpdateGlobalAddr(uint64_t mSize, uint64_t kSize, uint64_t nSize,
    __gm__ xType *x, __gm__ yType *y, __gm__ biasType *bias, __gm__ antiQuantScaleType *antiquantScale,
    __gm__ uint64_t *quantScale, __gm__ perTokenScaleType *perTokenScale, const bool isBias)
{
    isBias_ = isBias;
    aPrefetchAddr_ = reinterpret_cast<__gm__ uint8_t *>(x);
    auto layoutA = MakeLayoutA{}(mSize, kSize);
    gmA_ = AscendC::Te::MakeTensor(AscendC::Te::MakeGMmemPtr(x), layoutA);
    auto layoutC = MakeLayoutC{}(mSize, nSize);
    gmC_ = AscendC::Te::MakeTensor(AscendC::Te::MakeGMmemPtr(y), layoutC);
    if (isBias_) {
        auto layoutBias = MakeLayoutBias{}(1UL, nSize);
        gmBias_ = AscendC::Te::MakeTensor(AscendC::Te::MakeGMmemPtr(bias), layoutBias);
    }

    auto layoutScaleA = MakeLayoutScaleA{}(mSize, Cgmct::Gemm::CeilDiv(kSize, 64) * 2);
    gmScaleA_ = AscendC::Te::MakeTensor(AscendC::Te::MakeGMmemPtr(perTokenScale), layoutScaleA);
    auto layoutScaleB = MakeLayoutScaleB{}(Cgmct::Gemm::CeilDiv(kSize, 64) * 2, nSize);
    gmScaleB_ = AscendC::Te::MakeTensor(AscendC::Te::MakeGMmemPtr(antiquantScale), layoutScaleB);
}

WQBMM_CUBE_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void WQBMM_CUBE_COMPUTE_CLASS::PrefetchA(uint64_t aPrefetchSize, uint64_t xSizeLimit,
                                                           uint64_t perloadBufferOffset)
{
    uint64_t xOffset = GetBlockIdx() * aPrefetchSize;
    if (aPrefetchSize == 0 || xOffset >= xSizeLimit) {
        return;
    }
	#ifdef __XBL_PRINT__
    AscendC::printf("[ERROR] need prefetchA 3");
    #endif
    // TODO
    // DataCopyPadExtParams<xType> extParams;
    // DataCopyExtParams param;
    // param.blockCount = 1;
    // param.blockLen = (xOffset + aPrefetchSize > xSizeLimit ? xSizeLimit - xOffset : aPrefetchSize) * sizeof(xType);
    // param.srcStride = 0;
    // param.dstStride = 0;
    // DataCopyPad(perloadBuffer, xGlobal_[xOffset], param, extParams);
    // PipeBarrier<PIPE_MTE2>();
}

WQBMM_CUBE_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void WQBMM_CUBE_COMPUTE_CLASS::PrefetchA(uint64_t aPrefetchSize, uint64_t xSizeLimit)
{
    uint64_t xOffset = GetBlockIdx() * aPrefetchSize;
    if (aPrefetchSize == 0 || xOffset >= xSizeLimit) {
        return;
    }
    event_t eventIdMTE1ToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID<HardEvent::MTE1_MTE2>());
    SetFlag<HardEvent::MTE1_MTE2>(eventIdMTE1ToMTE2);
    WaitFlag<HardEvent::MTE1_MTE2>(eventIdMTE1ToMTE2);

    // 不支持直接搬运fp8，转成uint8搬
    uint64_t blockLen = xOffset + aPrefetchSize > xSizeLimit ? xSizeLimit - xOffset : aPrefetchSize;
    #ifdef __XBL_PRINT__
    AscendC::printf("[INFO] prefetchA 2 base_addr %p offset %d size %d", aPrefetchAddr_, xOffset, blockLen);
    #endif
    auto aPrefetchGm = AscendC::Te::MakeTensor(AscendC::Te::MakeGMmemPtr(reinterpret_cast<__gm__ uint8_t*>(aPrefetchAddr_ + xOffset)), AscendC::Te::MakeNDLayout<uint8_t>(1UL, blockLen));
    auto aPrefetchL1 = AscendC::Te::MakeTensor(AscendC::Te::MakeL1memPtr(reinterpret_cast<__cbuf__ uint8_t*>(aL1Offset_)), AscendC::Te::MakeNDLayout<uint8_t>(1UL, blockLen));
    auto copyGM2L1 = AscendC::Te::MakeCopy(AscendC::Te::CopyGM2L1{});
    AscendC::Te::Copy(copyGM2L1, aPrefetchL1, aPrefetchGm);
    PipeBarrier<PIPE_MTE2>();
}

WQBMM_CUBE_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void WQBMM_CUBE_COMPUTE_CLASS::MxA8W4Init(uint64_t aPrefetchSize, uint64_t l1RemainSize,
                                                            uint64_t l1StartSize, uint64_t mxBiasL1DbOffset,
                                                            const TCubeTiling *__restrict matmulTiling,
                                                            uint64_t biasL1Offset)
{
    //  MxA8W4场景空间分配: 其中 weight\Bias为全局分配，此处不感知
    //  (1) 有bias场景
    //  L1 (0~512KB): WeightL1_P0(64KB) |    Bias_P0(4KB)   | ScaleAL1_P0(32KB) | ScaleBL1_P0(32KB) | AL1_P0(124KB) |
    //              | AL1_P1(124KB)     | ScaleBL1_P1(32KB) | ScaleAL1_P1(32KB) | Bias_P1(4KB)      | WeightL1_P1(64KB)
    //  (2) 无bias场景
    //  L1 (0~512KB): WeightL1_P0(64KB) | ScaleAL1_P0(32KB) | ScaleBL1_P0(32KB) | AL1_P0(128KB) |
    //              | AL1_P1(128KB)     | ScaleBL1_P1(32KB) | ScaleAL1_P1(32KB) | WeightL1_P1(64KB)
    biasL1Offset_ = biasL1Offset; // 预计大小4Kb
    biasL1DbOffset_ = mxBiasL1DbOffset;
    aL1Count_ = 0;        // GMM 动态场景暂时关闭A分段全载策略
    aL1MaxHalfCount_ = 0; // GMM 动态场景暂时关闭A分段全载策略
    aL1DbNum_ = DOUBLE_BUFFER_NUM;

    // PS 使用offset替代创建AscendC::LocalTensor
    mxScaleAL1Offset_ = l1StartSize;
    mxScaleAL1DbOffset_ = l1RemainSize - MX_SCALE_L1_SIZE;
    l1RemainSize -= DOUBLE_BUFFER_NUM * MX_SCALE_L1_SIZE;
    l1StartSize += MX_SCALE_L1_SIZE;

    mxScaleBL1Offset_ = l1StartSize;
    mxScaleBL1DbOffset_ = l1RemainSize - MX_SCALE_L1_SIZE;
    l1RemainSize -= DOUBLE_BUFFER_NUM * MX_SCALE_L1_SIZE;
    l1StartSize += MX_SCALE_L1_SIZE;

    aL1Offset_ = l1StartSize;
    aL1DbOffset_ = l1RemainSize >> 1;                                      // 最后剩余空间全部给AL1开DB

    PrefetchA(aPrefetchSize, matmulTiling->M * matmulTiling->Ka, aL1Offset_);
    // 设置反向同步
    for (uint64_t i = 0; i < L0_BUF_NUM; i++) {
        SetFlag<HardEvent::M_MTE1>(eventIdMToMte1_ + i);
    }
    InitSync();
}

WQBMM_CUBE_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void WQBMM_CUBE_COMPUTE_CLASS::GetTensorC(const BasicBlockOffsetParam &param)
{
    constexpr uint64_t FP32_64_AS_UINT64 = 0x42800000;
    auto layoutL0C = AscendC::Te::MakeL0CLayout(param.mL1Size, param.nL1Size);
    auto tensorL0C = AscendC::Te::MakeTensor(AscendC::Te::MakeL0CmemPtr<float>(0), layoutL0C);
    auto CopyL0C2GM = AscendC::Te::MakeCopy(AscendC::Te::CopyL0C2GM{});
    auto tensorBlockGmC = gmC_(
        AscendC::Te::MakeCoord(param.mOffset, param.nOffset),
        AscendC::Te::MakeShape(param.mL1Size, param.nL1Size));
    AscendC::Te::Copy(CopyL0C2GM, tensorBlockGmC, tensorL0C, FP32_64_AS_UINT64, AscendC::Te::FixpipeParams{/*unitflag*/3});
}
}  // namespace WeightQuantBatchMatmulV2::Arch35

#endif
