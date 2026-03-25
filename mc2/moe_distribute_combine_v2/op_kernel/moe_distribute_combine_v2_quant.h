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
 * \file moe_distribute_v2_quant.h
 * \brief
 */

#ifndef MOE_DISTRIBUTE_COMBINE_V2_QUANT_H
#define MOE_DISTRIBUTE_COMBINE_V2_QUANT_H

#if __has_include("../moe_distribute_dispatch_v2/check_winsize.h")
#include "../moe_distribute_dispatch_v2/moe_distribute_v2_constant.h"
#include "../moe_distribute_dispatch_v2/moe_distribute_v2_base.h"
#else
#include "../../moe_distribute_dispatch_v2/op_kernel/moe_distribute_v2_constant.h"
#include "../../moe_distribute_dispatch_v2/op_kernel/moe_distribute_v2_base.h"
#endif

#if __has_include("../moe_distribute_dispatch_v2/quantize_functions.h")
#include "../moe_distribute_dispatch_v2/quantize_functions.h"
#else
#include "../../moe_distribute_dispatch_v2/op_kernel/quantize_functions.h"
#endif

namespace Mc2Kernel {
using namespace AscendC;
using namespace MoeDistributeV2Base;

template <typename ExpandXType, typename XType, typename ExpandIdxType, bool IsNeedReduceScatter, int32_t QuantMode, bool HasAddRmsNorm>
class MoeDistributeCombineQuant{
public:
    float scaleValFloat_;
    LocalTensor<half> fp16CastTensor_;
    LocalTensor<float> absFloatTensor_;
    LocalTensor<float> reduceMaxFloatTensor_;
    LocalTensor<float> scaleDivFloatTensor_;
    LocalTensor<float> scaleDupLocalTensor_;
    LocalTensor<float> winTpSendCountFloatTensor_;
    LocalTensor<float> floatLocalTemp_;
    uint32_t axisH_{0};
    uint32_t mask_{0};
    uint32_t repeatNum_{0};
    uint32_t hAlign32Size_{0};
    uint32_t quantScaleNum_{0};
    LocalTensor<int8_t> castLocalTensor_;
    LocalTensor<XType> scaleDivTensor_;

    __aicore__ inline MoeDistributeCombineQuant() = default;

    __aicore__ inline void SetQuantInitParams(LocalTensor<float> winTpSendCountFloatTensor,
        LocalTensor<half> fp16CastTensor, LocalTensor<float> absFloatTensor,
        LocalTensor<float> reduceMaxFloatTensor, LocalTensor<float> scaleDupLocalTensor) 
    {
        winTpSendCountFloatTensor_ = winTpSendCountFloatTensor;
        floatLocalTemp_ = winTpSendCountFloatTensor;
        fp16CastTensor_ = fp16CastTensor;
        absFloatTensor_ = absFloatTensor;
        reduceMaxFloatTensor_ = reduceMaxFloatTensor;
        scaleDupLocalTensor_ = scaleDupLocalTensor;
    }

    __aicore__ inline void SetDeQuantInitParams(LocalTensor<half> fp16CastTensor, LocalTensor<float> absFloatTensor,
        LocalTensor<float> scaleDupLocalTensor, LocalTensor<float> scaleDivFloatTensor) 
    {
        fp16CastTensor_ = fp16CastTensor;
        absFloatTensor_ = absFloatTensor;
        scaleDupLocalTensor_ = scaleDupLocalTensor;
        scaleDivFloatTensor_ = scaleDivFloatTensor;
    }

     __aicore__ inline void QuantInit(uint32_t &scaleNum_, uint32_t &hExpandXAlign32Size_, uint32_t &hExpandXAlignSize_,
                                     uint32_t &hFloatAlign256Size_, uint32_t &tokenScaleCnt_, uint32_t axisH)
    {
        axisH_ = axisH;
        if constexpr (QuantMode == INT8_COMM_QUANT) {
            hAlign32Size_ = Ceil(axisH_, UB_ALIGN) * UB_ALIGN;
            scaleValFloat_ = static_cast<float>(1.0f / SCALE_PARAM);
            uint32_t scaleGranu = static_cast<uint32_t>(UB_ALIGN / sizeof(float)); // 计算每个block得到的reducemax结果数量
            quantScaleNum_ = (hExpandXAlign32Size_ / sizeof(ExpandXType)) / scaleGranu; // 得到有效scale的个数
            scaleNum_ = quantScaleNum_;
            hExpandXAlignSize_ = hExpandXAlign32Size_;
            repeatNum_ = static_cast<uint32_t>(hFloatAlign256Size_ / ALIGNED_LEN); // BlockReduceMax 与 Brcb的重复迭代次数，每次256b参与计算
            mask_ = static_cast<uint32_t>(ALIGNED_LEN / sizeof(float));
            tokenScaleCnt_ = hAlign32Size_ / sizeof(ExpandXType) + quantScaleNum_; // int8_align + scale有效个数
        } else if constexpr(QuantMode == MXFP8_E5M2_COMM_QUANT || QuantMode == MXFP8_E4M3_COMM_QUANT) {
            hAlign32Size_ = Ceil(axisH_, UB_ALIGN) * UB_ALIGN;
            hExpandXAlignSize_ = Align128(axisH) * sizeof(ExpandXType);
            scaleNum_ = Align2(Ceil32(axisH));
            tokenScaleCnt_ = Align256(axisH) + scaleNum_; // int8_align + scale有效个数
        }
    }

    __aicore__ inline void Int8QuantProcess(LocalTensor<ExpandXType> &outLocal, LocalTensor<ExpandXType> &inLocal)
    {
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        castLocalTensor_ = outLocal.template ReinterpretCast<int8_t>(); // 长度为int8H_Align + scaleNum
        scaleDivTensor_ = castLocalTensor_[hAlign32Size_].template ReinterpretCast<ExpandXType>(); // 偏移前面的int8

        Cast(winTpSendCountFloatTensor_, inLocal, RoundMode::CAST_NONE, axisH_);
        PipeBarrier<PIPE_V>();
        Abs(absFloatTensor_, winTpSendCountFloatTensor_, axisH_); // absFloatTensor_ align到256并写0，支持ReduceMax与Brcb
        PipeBarrier<PIPE_V>();
        BlockReduceMax(reduceMaxFloatTensor_, absFloatTensor_, repeatNum_, mask_, 1, 1, BLOCK_NUM); // 32->1 256->8
        PipeBarrier<PIPE_V>();
        Muls(reduceMaxFloatTensor_, reduceMaxFloatTensor_, scaleValFloat_, quantScaleNum_); // 有效个数
        PipeBarrier<PIPE_V>();
        Cast(scaleDivTensor_, reduceMaxFloatTensor_, RoundMode::CAST_RINT, quantScaleNum_); // 有效个数
        PipeBarrier<PIPE_V>();
        Brcb(scaleDupLocalTensor_, reduceMaxFloatTensor_, repeatNum_, {1, BLOCK_NUM}); // 一次256
        PipeBarrier<PIPE_V>();
        Div(winTpSendCountFloatTensor_, winTpSendCountFloatTensor_, scaleDupLocalTensor_, axisH_); // 有效个数
        PipeBarrier<PIPE_V>();
        Cast(fp16CastTensor_, winTpSendCountFloatTensor_, RoundMode::CAST_RINT, axisH_);
        PipeBarrier<PIPE_V>();
        Cast(castLocalTensor_, fp16CastTensor_, RoundMode::CAST_RINT, axisH_);
        SyncFunc<AscendC::HardEvent::V_MTE3>();
    }

    __aicore__ inline void Int8DequantProcess(LocalTensor<XType>& inLocal, LocalTensor<XType> &outLocal)
    {
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        castLocalTensor_ = inLocal.template ReinterpretCast<int8_t>();
        scaleDivTensor_ = inLocal[hAlign32Size_ / INT8_DIVIVE];

        SyncFunc<AscendC::HardEvent::S_V>();
        Cast(scaleDivFloatTensor_, scaleDivTensor_, RoundMode::CAST_NONE, quantScaleNum_);
        Cast(fp16CastTensor_, castLocalTensor_, RoundMode::CAST_NONE, axisH_);
        PipeBarrier<PIPE_V>();
        Cast(absFloatTensor_, fp16CastTensor_, RoundMode::CAST_NONE, axisH_);
        Brcb(scaleDupLocalTensor_, scaleDivFloatTensor_, repeatNum_, {1, BLOCK_NUM});
        PipeBarrier<PIPE_V>();
        Mul(absFloatTensor_, absFloatTensor_, scaleDupLocalTensor_, axisH_);
        PipeBarrier<PIPE_V>();
        Cast(outLocal, absFloatTensor_, RoundMode::CAST_RINT, axisH_);
        PipeBarrier<PIPE_V>();
    }

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
    __aicore__ inline void QuantMxFp8(LocalTensor<ExpandXType>& outLocal, LocalTensor<ExpandXType>& inLocal)
    {
        uint32_t mxScaleNum = Align2(Ceil32(axisH_));
        __ubuf__ ExpandXType* srcAddr = (__ubuf__ ExpandXType*)inLocal.GetPhyAddr();
        __ubuf__ uint16_t* maxExpAddr = (__ubuf__ uint16_t*)floatLocalTemp_.GetPhyAddr();
        __ubuf__ uint16_t* halfScaleLocalAddr = (__ubuf__ uint16_t*)floatLocalTemp_[Align32(mxScaleNum)].GetPhyAddr();
        __ubuf__ int8_t* outLocalAddr = (__ubuf__ int8_t*)outLocal.GetPhyAddr();
        __ubuf__ uint16_t* mxScaleLocalAddr = (__ubuf__ uint16_t*)outLocal[Align256<uint32_t>(axisH_)].GetPhyAddr();
        if constexpr (QuantMode == MXFP8_E5M2_COMM_QUANT) {
            using fp8Type = fp8_e5m2_t;
            quant::ComputeMaxExp(srcAddr, maxExpAddr, axisH_); // 计算最大Exp
            quant::ComputeScale<fp8Type>(maxExpAddr, mxScaleLocalAddr, halfScaleLocalAddr, mxScaleNum); // 计算scales并填充
            quant::ComputeData<ExpandXType, fp8Type, AscendC::RoundMode::CAST_TRUNC, AscendC::RoundMode::CAST_RINT>(
                srcAddr, halfScaleLocalAddr, outLocalAddr, axisH_); // 计算量化后的expandx并填充
        } else if constexpr (QuantMode == MXFP8_E4M3_COMM_QUANT) {
            using fp8Type = fp8_e4m3fn_t;
            quant::ComputeMaxExp(srcAddr, maxExpAddr, axisH_); // 计算最大Exp
            quant::ComputeScale<fp8Type>(maxExpAddr, mxScaleLocalAddr, halfScaleLocalAddr, mxScaleNum); // 计算scales并填充
            quant::ComputeData<ExpandXType, fp8Type, AscendC::RoundMode::CAST_TRUNC, AscendC::RoundMode::CAST_RINT>(
                srcAddr, halfScaleLocalAddr, outLocalAddr, axisH_); // 计算量化后的expandx并填充
        }
    }

    __aicore__ inline void DeQuantMxFp8(LocalTensor<XType>& inLocal, LocalTensor<XType>& outLocal)
    {
        uint32_t mxScaleNum = Align2(Ceil32(axisH_));
        __ubuf__ XType* srcAddr = (__ubuf__ XType*)inLocal.GetPhyAddr();
        __ubuf__ uint16_t* mxScaleLocalAddr = (__ubuf__ uint16_t*)inLocal[Align256<uint32_t>(axisH_)].GetPhyAddr();
        __ubuf__ XType* outLocalAddr = (__ubuf__ XType*)outLocal.GetPhyAddr();
        quant::DequantizeData<XType, AscendC::RoundMode::CAST_TRUNC>(srcAddr, mxScaleLocalAddr, outLocalAddr, axisH_); 
    }
#endif

    __aicore__ inline void QuantProcess(LocalTensor<ExpandXType>& outLocal, LocalTensor<ExpandXType>& inLocal)
    {
        if constexpr (QuantMode == INT8_COMM_QUANT) {
            Int8QuantProcess(outLocal, inLocal);
        }
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
        else if constexpr (QuantMode == MXFP8_E5M2_COMM_QUANT || QuantMode == MXFP8_E4M3_COMM_QUANT) {
            QuantMxFp8(outLocal, inLocal);
        }
#endif
    }
    __aicore__ inline void DeQuantProcess(LocalTensor<XType>& inLocal, LocalTensor<XType>& outLocal)
    {
        if constexpr (QuantMode == INT8_COMM_QUANT) {
            Int8DequantProcess(inLocal, outLocal);
        }
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
        else if constexpr (QuantMode == MXFP8_E5M2_COMM_QUANT || QuantMode == MXFP8_E4M3_COMM_QUANT) {
            DeQuantMxFp8(inLocal, outLocal);
        }
#endif
    }

};
}
#endif // MOE_DISTRIBUTE_V2_QUANT_H