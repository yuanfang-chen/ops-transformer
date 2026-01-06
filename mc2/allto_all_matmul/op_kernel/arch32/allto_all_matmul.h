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
 * \file allto_all_matmul.h
 * \brief
 */

#ifndef ALL_TO_ALL_MATMUL_H
#define ALL_TO_ALL_MATMUL_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "allto_all_matmul_tiling.h"
#include "allto_all_matmul_util.h"
#include "block_epilogue_dequant.hpp"
#include "tile_broadcast_add.hpp"

#include "../../3rd/template_linear_algebra/include/template_linear_algebra/catlass.hpp"
#include "../../3rd/template_linear_algebra/include/template_linear_algebra/arch/arch.hpp"
#include "../../3rd/template_linear_algebra/include/template_linear_algebra/arch/resource.hpp"
#include "../../3rd/template_linear_algebra/include/template_linear_algebra/layout/layout.hpp"
#include "../../3rd/template_linear_algebra/include/template_linear_algebra/gemm/block/block_mmad.hpp"
#include "../../3rd/template_linear_algebra/include/template_linear_algebra/gemm/block/block_swizzle.hpp"
#include "../../3rd/template_linear_algebra/include/template_linear_algebra/gemm/dispatch_policy.hpp"
#include "../../3rd/template_linear_algebra/include/template_linear_algebra/gemm/gemm_type.hpp"
#include "../../3rd/template_linear_algebra/include/template_linear_algebra/gemm_coord.hpp"
#include "../../3rd/template_linear_algebra/include/template_linear_algebra/epilogue/tile/copy_gm_to_ub.hpp"
#include "../../3rd/template_linear_algebra/include/template_linear_algebra/epilogue/dispatch_policy.hpp"
#include "../../3rd/template_linear_algebra/include/template_linear_algebra/epilogue/tile/tile_broadcast_mul.hpp"
#include "../../3rd/template_linear_algebra/include/template_linear_algebra/epilogue/tile/tile_broadcast_one_blk.hpp"
#include "../../3rd/template_linear_algebra/include/template_linear_algebra/epilogue/tile/tile_swizzle.hpp"
#include "../../3rd/template_linear_algebra/include/template_linear_algebra/epilogue/tile/tile_copy.hpp"

#include "matmul.hpp"

using namespace AscendC;
using namespace Catlass;

namespace AlltoAllMatmulImpl {

template<AscendC::HardEvent event>
__aicore__ inline void SyncFunc() {
    int32_t eventID = static_cast<int32_t>(GetTPipePtr()->FetchEventID(event));
    AscendC::SetFlag<event>(eventID);
    AscendC::WaitFlag<event>(eventID);
}
// A2AMM : AlltoAllMatmul
#define TemplateA2AMMClass typename AType, typename BType, typename BiasType, typename PerTokenScaleType, typename ScaleType, typename CType, typename AllToAllResultType, bool hasBias, bool transB
#define TemplateA2AMMFunc AType, BType, BiasType, PerTokenScaleType, ScaleType, CType, AllToAllResultType, hasBias, transB

using namespace AscendC;
template <TemplateA2AMMClass>
class AlltoAllMatmul : public CommBase{
public:
    __aicore__ inline AlltoAllMatmul() {};
    __aicore__ inline void Init(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM,
                                GM_ADDR perTokenScaleGM, GM_ADDR scaleGM,
                                GM_ADDR cGM, GM_ADDR allToAllResult,
                                GM_ADDR workspaceGM, GM_ADDR tilingGM);
    __aicore__ inline void Process();

private:
    __aicore__ inline void AIVInit();
    __aicore__ inline void AICInit();
    __aicore__ inline void CatlassMatmul();
    __aicore__ inline void AlltoAll();
    __aicore__ inline void Dequant();
    __aicore__ inline void Quant(uint64_t flag_idx, int32_t cal_idx);
    __aicore__ inline void QuantPerToken(LocalTensor<float> copyTensor, LocalTensor<float> absTensor, LocalTensor<float> reduceMaxTensor, 
                                        LocalTensor<float> quantScaleTensor, int32_t actual_move_size, int32_t actual_move_token,
                                        int32_t token_per_move, int32_t move_idx, event_t event_id);
    __aicore__ inline void QuantToken(__gm__ AType *data_src, int32_t data_offset, int32_t token_per_core,
                                        int32_t data_len, int32_t tokenSize, int32_t cal_idx);
    __aicore__ inline void QuantTokenSegment(__gm__ AType *data_src, int32_t data_offset, int32_t token_per_core,
                                        int32_t data_len, int32_t tokenSize, int32_t cal_idx);

private:
    GM_ADDR aGM_;
    GM_ADDR bGM_;
    GM_ADDR cGM_;
    GM_ADDR biasGM_;
    GM_ADDR scaleGM_;
    GM_ADDR perTokenScaleGM_;
    GM_ADDR allToAllResultGM_;
    GM_ADDR workspaceGM_;

    int32_t aligned_a;
    int32_t aligned_b;
    int32_t gm_a_pingpong_size;

    __gm__ AType* gm_peer_mem;
    __gm__ int8_t* quant_aGM_;
    __gm__ int32_t* dequant_cGM_;
    GM_ADDR quant_scale_gm;

    Catlass::Arch::Resource<Catlass::Arch::AtlasA2> resource;
};


template <TemplateA2AMMClass>
__aicore__ inline void AlltoAllMatmul<TemplateA2AMMFunc>::Init(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM,
                                                               GM_ADDR perTokenScaleGM, GM_ADDR scaleGM,
                                                               GM_ADDR cGM, GM_ADDR allToAllResultGM,
                                                               GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    REGISTER_TILING_DEFAULT(AlltoAllMatmulTilingData);
    auto tiling = (__gm__ AlltoAllMatmulTilingData*)tilingGM;
    GET_TILING_DATA(tilingData, tilingGM);

    auto contextGM = AscendC::GetHcclContext<HCCL_GROUP_ID_0>();
    winContext_ = (__gm__ HcclCombineOpParam *)contextGM;
    rank = winContext_ -> rankId;
    rank_size = tilingData.allToAllMatmulInfo.worldSize;

    aGM_ = aGM;
    bGM_ = bGM;
    cGM_ = cGM;
    biasGM_ = biasGM;
    scaleGM_ = scaleGM;
    allToAllResultGM_ = allToAllResultGM;
    workspaceGM_ = GetUserWorkspace(workspaceGM);

    CommBase::SetArgs<AType>(rank, rank_size, tilingData);
    this->ub_offset = UB_OFFSET / sizeof(AType);

    quant_aGM_ = reinterpret_cast<__gm__ int8_t *>(std::is_same_v<BType, int8_t> ? workspaceGM_ : nullptr);
    dequant_cGM_ = reinterpret_cast<__gm__ int32_t *>(std::is_same_v<BType, int8_t> ? workspaceGM_ + quantSize : nullptr);
    quant_scale_gm = reinterpret_cast<GM_ADDR>(std::is_same_v<BType, int8_t>? workspaceGM_ + quantSize + dequantSize : nullptr);

    AlltoAllMatmul<TemplateA2AMMFunc>::AICInit();
    AlltoAllMatmul<TemplateA2AMMFunc>::AIVInit();
}

template <TemplateA2AMMClass>
__aicore__ inline void AlltoAllMatmul<TemplateA2AMMFunc>::AICInit()
{
    if ASCEND_IS_AIC {
        SetLoadDataPaddingValue(0);
        SetAtomicNone();
        SetFixpipeNz2ndFlag(1, 0, 0);
        gm_peer_mem = reinterpret_cast<__gm__ AType*>(buff[rank]);
    }
}

template <TemplateA2AMMClass>
__aicore__ inline void AlltoAllMatmul<TemplateA2AMMFunc>::AIVInit()
{
    if ASCEND_IS_AIV {
        SetAtomicNone();
        SetMaskNorm();
        SetVectorMask<int32_t>((uint64_t)-1, (uint64_t)-1);
    }
}

template <TemplateA2AMMClass>
__aicore__ inline void AlltoAllMatmul<TemplateA2AMMFunc>::CatlassMatmul()
{
    if ASCEND_IS_AIC {
        using ArchTag = Arch::AtlasA2;

        constexpr bool ENABLE_UNIT_FLAG = false;
        constexpr bool ENABLE_SHUFFLE_K = false;
        constexpr bool aicCalBias = !std::is_same_v<BType, int8_t> && hasBias;

        using ElementA = std::conditional_t<std::is_same_v<BType, int8_t>, int8_t, AType>;
        using ElementB = BType;
        using ElementC = std::conditional_t<std::is_same_v<BType, int8_t>, int32_t, CType>;
        using ElementBias = BiasType;
        using LayoutA = layout::RowMajor;
        // B转置
        using LayoutB = std::conditional_t<transB, layout::ColumnMajor, layout::RowMajor>;
        using LayoutC = layout::RowMajor;
        using LayoutBias = layout::VectorLayout;

        uint32_t realM = m / rank_size;
        uint32_t realK = k * rank_size;
        LayoutA layoutA{static_cast<uint32_t>(realM), static_cast<uint32_t>(realK)};
        LayoutB layoutB{static_cast<uint32_t>(realK), static_cast<uint32_t>(n)};
        LayoutC layoutC{static_cast<uint32_t>(realM), static_cast<uint32_t>(n)};
        LayoutBias layoutBias{static_cast<uint32_t>(n)};

        using DispatchPolicy = std::conditional_t<aicCalBias, Gemm::MmadAtlasA2PingpongBias<ENABLE_UNIT_FLAG>,
                                                           Gemm::MmadAtlasA2Preload<ENABLE_UNIT_FLAG, ENABLE_SHUFFLE_K>>;

        using AType_ = Gemm::GemmType<ElementA, LayoutA>;
        using BType_ = Gemm::GemmType<ElementB, LayoutB>;
        using CType_ = Gemm::GemmType<ElementC, LayoutC>;
        using BiasType_ = std::conditional_t<aicCalBias, Gemm::GemmType<ElementBias, LayoutBias>, void>;

        struct TileCopyOpt : public Catlass::Gemm::Tile::TileCopy<ArchTag, AType_, BType_, CType_, BiasType_> {
            using Base = Catlass::Gemm::Tile::TileCopy<ArchTag, AType_, BType_, CType_, BiasType_>;
            using ElementA = typename Base::ElementA;
            using ElementB = typename Base::ElementB;
            using ElementAccumulator = typename Base::ElementAccumulator;

            using CopyGmToL1A = typename Base::CopyGmToL1A;
            using CopyGmToL1B = typename Base::CopyGmToL1B;

            using CopyL1ToL0A = typename Base::CopyL1ToL0A;
            using CopyL1ToL0B = typename Base::CopyL1ToL0B;

            using CopyL0CToGm = typename Base::CopyL0CToGm;
            using BiasTypeSelector = typename Base::BiasTypeSelector;
            using CopyGmToL1Bias = typename Base::CopyGmToL1Bias;
            using CopyL1ToBT = typename Base::CopyL1ToBT;
        };

        using TileCopy = TileCopyOpt;
        GemmCoord processSize{static_cast<uint32_t>(realM), static_cast<uint32_t>(n), static_cast<uint32_t>(realK)};
        using BlockScheduler30 = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;

        GM_ADDR srcGM = std::is_same_v<BType, int8_t> ? reinterpret_cast<GM_ADDR>(quant_aGM_) : reinterpret_cast<GM_ADDR>(gm_peer_mem);
        GM_ADDR matmulResultGM = std::is_same_v<BType, int8_t> ? reinterpret_cast<GM_ADDR>(dequant_cGM_) : cGM_;
        if (m0 == 128) {
            using L1TileShape = GemmShape<128, 256, 256>;
            using L0TileShape = GemmShape<128, 256, 64>;
            using BlockMmadOpt = Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, AType_, BType_, CType_, BiasType_, TileCopy>;
            using MatmulKernel = Gemm::Kernel::AlltoAllMatmulKernel<void, void, BlockMmadOpt, void, BlockScheduler30, aicCalBias>;
            MatmulKernel matmul_op;
            typename MatmulKernel::Params params{processSize,
                                    reinterpret_cast<GM_ADDR>(srcGM), layoutA,
                                    reinterpret_cast<GM_ADDR>(bGM_), layoutB,
                                    reinterpret_cast<GM_ADDR>(biasGM_),
                                    reinterpret_cast<GM_ADDR>(matmulResultGM), layoutC,
                                    p_value, 3, 0, static_cast<int32_t>(rank_size), MAX_BLOCK_COUNT};
            matmul_op(params);
        } else {
            using L1TileShape = GemmShape<256, 128, 256>;
            using L0TileShape = GemmShape<256, 128, 64>;
            using BlockMmadOpt = Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, AType_, BType_, CType_, BiasType_, TileCopy>;
            using MatmulKernel = Gemm::Kernel::AlltoAllMatmulKernel<void, void, BlockMmadOpt, void, BlockScheduler30, aicCalBias>;
            MatmulKernel matmul_op;
            typename MatmulKernel::Params params{processSize,
                                    reinterpret_cast<GM_ADDR>(srcGM), layoutA,
                                    reinterpret_cast<GM_ADDR>(bGM_), layoutB,
                                    reinterpret_cast<GM_ADDR>(biasGM_),
                                    reinterpret_cast<GM_ADDR>(matmulResultGM), layoutC,
                                    p_value, 3, 0, static_cast<int32_t>(rank_size), MAX_BLOCK_COUNT};
            matmul_op(params);
        }
    }
}

template <TemplateA2AMMClass>
__aicore__ inline void AlltoAllMatmul<TemplateA2AMMFunc>::QuantPerToken(LocalTensor<float> copyTensor,
    LocalTensor<float> absTensor, LocalTensor<float> reduceMaxTensor, LocalTensor<float> quantScaleTensor,
    int32_t actual_move_size, int32_t actual_move_token, int32_t token_per_move, int32_t move_idx,
    event_t event_id)

{
    PipeBarrier<PIPE_V>();
    Abs(absTensor, copyTensor, actual_move_size);
    int32_t tokenSize = k * rank_size;
    for (int32_t tokenIdx = 0; tokenIdx < actual_move_token; tokenIdx++) {
        uint32_t tokenOffset = tokenIdx * tokenSize;
        PipeBarrier<PIPE_V>();
        ReduceMax<float>(reduceMaxTensor, absTensor[tokenOffset], absTensor[tokenOffset], tokenSize);
        SetFlag<HardEvent::V_S>(event_id);
        WaitFlag<HardEvent::V_S>(event_id);
        float maxValue = reduceMaxTensor.GetValue(0);
        float quantScale = maxValue / MAX_INT8;
        float quant_scale_reciproal = MAX_INT8 / maxValue;
        quantScaleTensor.SetValue(move_idx * token_per_move + tokenIdx, quantScale);
        SetFlag<HardEvent::S_V>(event_id);
        WaitFlag<HardEvent::S_V>(event_id);

        Muls(copyTensor[tokenOffset], copyTensor[tokenOffset], quant_scale_reciproal, tokenSize);
        PipeBarrier<PIPE_V>();

        Cast(copyTensor.ReinterpretCast<int32_t>()[tokenOffset], copyTensor[tokenOffset], RoundMode::CAST_RINT, tokenSize);
        PipeBarrier<PIPE_V>();
        SetDeqScale((half)1.000000e+00f);
        PipeBarrier<PIPE_V>();
        Cast(copyTensor.ReinterpretCast<half>()[tokenOffset], copyTensor.ReinterpretCast<int32_t>()[tokenIdx * tokenSize], RoundMode::CAST_ROUND, tokenSize);
    }
    PipeBarrier<PIPE_V>();

    Cast(copyTensor.ReinterpretCast<int8_t>(), copyTensor.ReinterpretCast<half>(), RoundMode::CAST_TRUNC, actual_move_size);

    PipeBarrier<PIPE_V>();
}

template <TemplateA2AMMClass>
__aicore__ inline void AlltoAllMatmul<TemplateA2AMMFunc>::QuantToken(__gm__ AType *data_src, int32_t data_offset,
    int32_t token_per_core, int32_t data_len, int32_t tokenSize, int32_t cal_idx)
{
    int32_t ub_ping_pong_size = max_ub_ping_pong_size / tokenSize * tokenSize;
    int32_t ping_pong_move_count = (data_len + ub_ping_pong_size - 1) / ub_ping_pong_size;
    int32_t actual_move_size = ub_ping_pong_size;
    int32_t token_per_move = actual_move_size / tokenSize;
    int32_t actual_move_token = token_per_move; /* ub_ping_pong_size已经与tokenSize对齐，因此必然每次搬运整数倍token */
    int32_t total_token_num = data_len / tokenSize;

    uint32_t midElementCnt = UB_OFFSET / sizeof(float);
    LocalTensor<float> ubTensor = uBuf_.Get<float>();
    LocalTensor<float> copyTensor0 = ubTensor;
    LocalTensor<float> copyTensor1 = ubTensor[midElementCnt];

    /* 用于存储计算quantScale的token取abs的结果 */
    int32_t abs_offset = Block32B<float>::AlignUp(ub_ping_pong_size); /* 从GM拷贝的数据用abs_offset_a大小空间，case为float后用abs_offset大小的空间 */
    int32_t cast_offset = Block32B<AType>::AlignUp(abs_offset);   // 换成AType可能不一定32B对齐
    LocalTensor<float> absTensor0 = copyTensor0[abs_offset];
    LocalTensor<float> absTensor1 = copyTensor1[abs_offset];

    /* 用于存储取abs后，取出token中最大元素的值 */
    int32_t reduce_max_offset = Block32B<float>::AlignUp(ub_ping_pong_size);
    LocalTensor<float> reduceMaxTensor0 = absTensor0[reduce_max_offset];
    LocalTensor<float> reduceMaxTensor1 = absTensor1[reduce_max_offset];

    /* 用于存储计算完成的量化系数 */
    int32_t quant_scale_offset = BLOCK_ALIGN_BYTES / sizeof(float);
    LocalTensor<float> quantScaleTensor = reduceMaxTensor0[quant_scale_offset];

    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
    for (int32_t move_idx = 0; move_idx < ping_pong_move_count; ++move_idx) {
        if (move_idx == ping_pong_move_count - 1) {
            actual_move_size = data_len - move_idx * ub_ping_pong_size;
            actual_move_token = actual_move_size / tokenSize;
        }
        auto event_id = (move_idx & 1) ? EVENT_ID0 : EVENT_ID1;
        LocalTensor<float> copyTensor = (move_idx & 1) ? copyTensor0 : copyTensor1;
        LocalTensor<float> absTensor = (move_idx & 1) ? absTensor0 : absTensor1;
        LocalTensor<float> reduceMaxTensor = (move_idx & 1) ? reduceMaxTensor0 : reduceMaxTensor1;

        WaitFlag<HardEvent::MTE3_MTE2>(event_id);
        CopyGmToUbufAlignB16(copyTensor.ReinterpretCast<AType>()[cast_offset], reinterpret_cast<__gm__ AType *>(data_src) +
            data_offset, 1, actual_move_size * sizeof(AType), 0, 0);
        SetFlag<HardEvent::MTE2_V>(event_id);
        WaitFlag<HardEvent::MTE2_V>(event_id);
        Cast(copyTensor, copyTensor.ReinterpretCast<AType>()[cast_offset], RoundMode::CAST_NONE, actual_move_size);
        QuantPerToken(copyTensor, absTensor, reduceMaxTensor, quantScaleTensor,
            actual_move_size, actual_move_token, token_per_move, move_idx, event_id);
        SetFlag<HardEvent::V_MTE3>(event_id);
        WaitFlag<HardEvent::V_MTE3>(event_id);
        /* 搬运到GM上时，与peerMem上的相对位置保持不变，后续数据offset可以复用 */
        CopyUbufToGmAlignB16(reinterpret_cast<__gm__ int8_t *>(quant_aGM_) + data_offset, copyTensor.ReinterpretCast<int8_t>(),
            1, actual_move_size * sizeof(int8_t), 0, 0);
        data_offset += actual_move_size;
        SetFlag<HardEvent::MTE3_MTE2>(event_id);
    }
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
    CopyUbufToGmAlignB16(reinterpret_cast<__gm__ float *>(quant_scale_gm) + core_idx * token_per_core + cal_idx * num_per_rank_m,
        quantScaleTensor, 1, total_token_num * sizeof(float), 0, 0);
}

template <TemplateA2AMMClass>
__aicore__ inline void AlltoAllMatmul<TemplateA2AMMFunc>::QuantTokenSegment(__gm__ AType *data_src, int32_t data_offset,
    int32_t token_per_core, int32_t data_len, int32_t tokenSize, int32_t cal_idx)
{
    uint32_t midElementCnt = UB_OFFSET / sizeof(float);
    LocalTensor<float> ubTensor = uBuf_.Get<float>();
    LocalTensor<float> copyTensor0 = ubTensor;
    LocalTensor<float> copyTensor1 = ubTensor[midElementCnt];

    int32_t abs_offset = Block32B<float>::AlignUp(copyTensorSize); /* 从GM拷贝的数据用abs_offset_a大小空间，case为float后用abs_offset大小的空间 */
    LocalTensor<float> absTensor0 = copyTensor0[abs_offset];
    LocalTensor<float> absTensor1 = copyTensor1[abs_offset];

    int32_t reduce_max_offset = Block32B<float>::AlignUp(copyTensorSize);
    LocalTensor<float> reduceMaxTensor = absTensor0[reduce_max_offset];

    int32_t quant_scale_offset = BLOCK_SIZE / sizeof(float);
    LocalTensor<float> quantScaleTensor = reduceMaxTensor[quant_scale_offset];

    int32_t actual_move_size = copyTensorSize;
    int32_t total_token_num = data_len / tokenSize; /* 当前核实际处理的token数 */

    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
    for (int32_t token_loop = 0; token_loop < total_token_num; token_loop++) {
        int32_t data_token_offset = data_offset + token_loop * tokenSize;
        int32_t data_segment_offset = 0;
        Duplicate<float>(reduceMaxTensor, static_cast<float>(0), 1);
        PipeBarrier<PIPE_V>();
        /* 获取当前token的max_abs_value */
        for (int32_t token_segment_loop = 0; token_segment_loop < copyTimes; token_segment_loop++) {
            if (token_segment_loop == copyTimes - 1) {
                actual_move_size = tokenSize - token_segment_loop * copyTensorSize;
            }
            auto event_id = (token_segment_loop & 1) ? EVENT_ID0 : EVENT_ID1;
            LocalTensor<float> copyTensor = (token_segment_loop & 1) ? copyTensor0 : copyTensor1;
            LocalTensor<float> absTensor = (token_segment_loop & 1) ? absTensor0 : absTensor1;
            WaitFlag<HardEvent::MTE3_MTE2>(event_id);
            /* 下一步需要将copyTensor转换成float类型，目标地址复用copyTensor。为防止踩踏，将AType类型数据内存放在后半段 */
            CopyGmToUbufAlignB16(copyTensor.ReinterpretCast<AType>()[abs_offset], reinterpret_cast<__gm__ AType *>(data_src) +
                data_token_offset + data_segment_offset, 1, actual_move_size * sizeof(AType), 0, 0);
            SetFlag<HardEvent::MTE2_V>(event_id);
            WaitFlag<HardEvent::MTE2_V>(event_id);
            Cast(copyTensor, copyTensor.ReinterpretCast<AType>()[abs_offset], RoundMode::CAST_NONE, actual_move_size);
            PipeBarrier<PIPE_V>();
            Abs(absTensor, copyTensor, actual_move_size);
            PipeBarrier<PIPE_V>();
            ReduceMax<float>(copyTensor, absTensor, absTensor, actual_move_size);
            SetFlag<HardEvent::V_S>(event_id);
            WaitFlag<HardEvent::V_S>(event_id);
            float current_max_value = copyTensor.GetValue(0);
            float last_max_value = reduceMaxTensor.GetValue(0);
            float max_value = current_max_value > last_max_value ? current_max_value : last_max_value;
            quantScaleTensor.SetValue(token_loop, max_value);
            SetFlag<HardEvent::MTE3_MTE2>(event_id);
            data_segment_offset += actual_move_size;
        }
        float token_max_value = quantScaleTensor.GetValue(token_loop);
        float quantScale = token_max_value / MAX_INT8;
        float quant_scale_reciproal = MAX_INT8 / token_max_value;
        quantScaleTensor.SetValue(token_loop, quantScale);
        data_segment_offset = 0;
        actual_move_size = copyTensorSize;
        /* 量化当前token */
        for (int32_t token_segment_loop = 0; token_segment_loop < copyTimes; token_segment_loop++) {
            if (token_segment_loop == copyTimes - 1) {
                actual_move_size = tokenSize - token_segment_loop * copyTensorSize;
            }
            auto event_id = (token_segment_loop & 1) ? EVENT_ID0 : EVENT_ID1;
            LocalTensor<float> copyTensor = (token_segment_loop & 1) ? copyTensor0 : copyTensor1;
            LocalTensor<float> absTensor = (token_segment_loop & 1) ? absTensor0 : absTensor1;
            WaitFlag<HardEvent::MTE3_MTE2>(event_id);
            /* 下一步需要将copyTensor转换成float类型，目标地址复用copyTensor。为防止踩踏，将AType类型数据内存放在后半段 */
            CopyGmToUbufAlignB16(copyTensor.ReinterpretCast<AType>()[abs_offset], reinterpret_cast<__gm__ AType *>(data_src) +
                data_token_offset + data_segment_offset, 1, actual_move_size * sizeof(AType), 0, 0);
            SetFlag<HardEvent::MTE2_V>(event_id);
            WaitFlag<HardEvent::MTE2_V>(event_id);
            Cast(copyTensor, copyTensor.ReinterpretCast<AType>()[abs_offset], RoundMode::CAST_NONE, actual_move_size);
            PipeBarrier<PIPE_V>();
            Muls(copyTensor, copyTensor, quant_scale_reciproal, actual_move_size);
            PipeBarrier<PIPE_V>();
            Cast(copyTensor.ReinterpretCast<half>(), copyTensor, RoundMode::CAST_ROUND, actual_move_size);
            PipeBarrier<PIPE_V>();
            Cast(copyTensor.ReinterpretCast<int8_t>(), copyTensor.ReinterpretCast<half>(), RoundMode::CAST_ROUND, actual_move_size);
            SetFlag<HardEvent::V_MTE3>(event_id);
            WaitFlag<HardEvent::V_MTE3>(event_id);

            CopyUbufToGmAlignB16(reinterpret_cast<__gm__ int8_t *>(quant_aGM_) + data_token_offset + data_segment_offset,
                copyTensor.ReinterpretCast<int8_t>(), 1, actual_move_size * sizeof(int8_t), 0, 0);  

            SetFlag<HardEvent::MTE3_MTE2>(event_id);
            data_segment_offset += actual_move_size;
        }
    }

    CopyUbufToGmAlignB16(reinterpret_cast<__gm__ float *>(quant_scale_gm) + core_idx * token_per_core, quantScaleTensor, 1,
        total_token_num * sizeof(float), 0, 0);
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
}

template <TemplateA2AMMClass>
__aicore__ inline void AlltoAllMatmul<TemplateA2AMMFunc>::Quant(uint64_t flag_idx, int32_t cal_idx) {
    __gm__ AType* data_src = (__gm__ AType *)buff[rank];
    
    int32_t data_src_core_offset = (core_idx % first_step_core_num) * data_per_core;  // 一共first_stem_core_num个核，每个核分段处理一部分数据
    int32_t total_move = num_per_rank_move * rank_size;
    // data_per_core是按照m0的粒度，均分给每个核处理的数据量。在尾块处理时，部分核会分不到数据
    int32_t data_len = data_src_core_offset + data_per_core > total_move ? total_move - data_src_core_offset : data_per_core;
    if (data_len < 0) {
        return;
    }
    int64_t data_src_offset = flag_idx * peer_mem_block_size;
    int32_t data_offset = data_src_offset + data_src_core_offset;
    int32_t token_per_core = num_per_rank_m / first_step_core_num;

    if (copyTokenNum == 0) {
        // token过大，分段量化
        QuantTokenSegment(data_src, data_offset, token_per_core, data_len, mid_output_k_size, cal_idx);
    } else {
        // token较小，一次量化多个
        QuantToken(data_src, data_offset, token_per_core, data_len, mid_output_k_size, cal_idx);
    }
    
}

template <TemplateA2AMMClass>
__aicore__ inline void AlltoAllMatmul<TemplateA2AMMFunc>::Dequant()
{
    using ArchTag = Arch::AtlasA2;

    using ElementC = int32_t;
    using ElementBias = BiasType;
    using LayoutD = layout::RowMajor;

    uint32_t realM = m / rank_size;
    uint32_t realK = k * rank_size;
    LayoutD layoutD{static_cast<uint32_t>(realM), static_cast<uint32_t>(n)};

    using CType_ = Gemm::GemmType<ElementC, layout::RowMajor>;

    constexpr uint32_t ubStages = 2;
    using EpilogueDispatchPolicy = Epilogue::EpilogueAtlasA2PerTokenDequant<ubStages>;
    using ScaleGType = Gemm::GemmType<ScaleType, layout::VectorLayout>;
    using PerTokenScaleGType = Gemm::GemmType<PerTokenScaleType, layout::VectorLayout>;
    using BiasGType = Gemm::GemmType<BiasType, layout::VectorLayout>;
    using DType = Gemm::GemmType<CType, layout::RowMajor>;
    layout::VectorLayout layoutScale{static_cast<uint32_t>(n)};
    layout::VectorLayout layoutPerTokenScale{static_cast<uint32_t>(realM)};
    layout::VectorLayout layoutBias{static_cast<uint32_t>(n)};
    using LayoutScale = layout::VectorLayout;
    using LayoutPerTokenScale = layout::VectorLayout;
    using ElementD = CType;

    using RowBroadcastMulType = Gemm::GemmType<float, layout::RowMajor>;
    using RowBroadcastAddType = Gemm::GemmType<float, layout::RowMajor>;
    using BroadcastOneBlkType = Gemm::GemmType<float, layout::RowMajor>;
    using OneBlkColumnBroadcastMulType = Gemm::GemmType<float, layout::RowMajor>;

    struct TileCopyDequant : public Catlass::Epilogue::Tile::TileCopy<ArchTag, CType_, ScaleGType, PerTokenScaleGType, DType> {
        using Base = Catlass::Epilogue::Tile::TileCopy<ArchTag, CType_, ScaleGType, PerTokenScaleGType, DType>;
        using ElementC = typename Base::ElementC;
        using ElementScale = typename Base::ElementX;
        using ElementPerTokenScale = typename Base::ElementY;
        using ElementBias = typename BiasGType::Element;
        using ElementD = typename Base::ElementD;

        using CopyGmToUbC = typename Base::CopyGmToUbC;
        using CopyGmToUbScale = typename Base::CopyGmToUbX;
        using CopyGmToUbPerTokenScale = typename Base::CopyGmToUbY;
        using CopyGmToUbBias = Catlass::Epilogue::Tile::CopyGm2Ub<ArchTag, BiasGType>;
        using CopyUbToGmD = typename Base::CopyUbToGmD;
    };

    using EpilogueTileScheduler = Epilogue::Tile::EpilogueHorizontalTileSwizzle;
    GemmCoord problemShape{static_cast<uint32_t>(realM), static_cast<uint32_t>(n), static_cast<uint32_t>(realK)};

    AscendC::GlobalTensor<ElementD> gmD;
    gmD.SetGlobalBuffer((__gm__ ElementD *)cGM_);
    AscendC::GlobalTensor<ElementC> gmC;
    gmC.SetGlobalBuffer((__gm__ ElementC *)dequant_cGM_);

    AscendC::GlobalTensor<ScaleType> gmScale;
    gmScale.SetGlobalBuffer((__gm__ ScaleType *)scaleGM_);
    AscendC::GlobalTensor<PerTokenScaleType> gmPerTokenScale;
    gmPerTokenScale.SetGlobalBuffer((__gm__ PerTokenScaleType *)quant_scale_gm);
    AscendC::GlobalTensor<ElementBias> gmBias;
    gmBias.SetGlobalBuffer((__gm__ ElementBias *)biasGM_);

    uint32_t rows_per_core = DivCeil(problemShape.m(), core_num);
    uint32_t rows_this_core = rows_per_core;
    uint32_t st_row_per_core = core_idx * rows_per_core;
    if (st_row_per_core < problemShape.m()) {
        if (rows_this_core + st_row_per_core > problemShape.m()) {
            rows_this_core = problemShape.m() - st_row_per_core;
        }
    } else {
        rows_this_core = 0;
    }
    MatrixCoord coreOffset(st_row_per_core, 0u);
    auto layoutC = layout::RowMajor{problemShape.m(), n};
    int64_t gmOffsetC = layoutC.GetOffset(coreOffset);
    GemmCoord actualBlockShape{rows_this_core, n, 1};

    if (m0 == 128) {
        using EpilogueTileShape = MatrixShape<32, 256>;
        using TileRowBroadcastMul = Epilogue::Tile::TileRowBroadcastMul<ArchTag, RowBroadcastMulType, EpilogueTileShape>;
        using TileRowBroadcastAdd = Epilogue::Tile::TileRowBroadcastAdd<ArchTag, RowBroadcastAddType, EpilogueTileShape>;
        using TileBroadcastOneBlk = Epilogue::Tile::TileBroadcastOneBlk<ArchTag, BroadcastOneBlkType,
            EpilogueTileShape::ROW>;
        using TileOneBlkColumnBroadcastMul = Epilogue::Tile::TileOneBlkColumnBroadcastMul<ArchTag,
            OneBlkColumnBroadcastMulType, EpilogueTileShape>;
        using QuantBlockEpilogue = Epilogue::Block::BlockEpilogue<EpilogueDispatchPolicy, CType_, ScaleGType, PerTokenScaleGType, BiasGType, DType,
            TileRowBroadcastMul, TileRowBroadcastAdd, TileBroadcastOneBlk, TileOneBlkColumnBroadcastMul, TileCopyDequant, EpilogueTileScheduler>;
        QuantBlockEpilogue blockEpilogue(resource);

        using EpilogueParams = typename QuantBlockEpilogue::Params;
        EpilogueParams epilogueParams {
            scaleGM_, layoutScale,
            quant_scale_gm, layoutPerTokenScale.GetTileLayout(problemShape.template GetCoordByAxis<0>()),
            biasGM_, layoutBias
        };
        blockEpilogue.UpdateParams(epilogueParams);

        blockEpilogue(coreOffset, actualBlockShape, gmC[gmOffsetC], layoutC, gmD[gmOffsetC], layoutC);
    } else {
        using EpilogueTileShape = MatrixShape<64, 128>;
        using TileRowBroadcastMul = Epilogue::Tile::TileRowBroadcastMul<ArchTag, RowBroadcastMulType, EpilogueTileShape>;

        using TileRowBroadcastAdd = Epilogue::Tile::TileRowBroadcastAdd<ArchTag, RowBroadcastAddType, EpilogueTileShape>;

        using TileBroadcastOneBlk = Epilogue::Tile::TileBroadcastOneBlk<ArchTag, BroadcastOneBlkType,
            EpilogueTileShape::ROW>;
        using TileOneBlkColumnBroadcastMul = Epilogue::Tile::TileOneBlkColumnBroadcastMul<ArchTag,
            OneBlkColumnBroadcastMulType, EpilogueTileShape>;
        using QuantBlockEpilogue = Epilogue::Block::BlockEpilogue<EpilogueDispatchPolicy, CType_, ScaleGType, PerTokenScaleGType, BiasGType, DType,
            TileRowBroadcastMul, TileRowBroadcastAdd, TileBroadcastOneBlk, TileOneBlkColumnBroadcastMul, TileCopyDequant, EpilogueTileScheduler>;
        QuantBlockEpilogue blockEpilogue(resource);

        using EpilogueParams = typename QuantBlockEpilogue::Params;
        EpilogueParams epilogueParams {
            scaleGM_, layoutScale,
            quant_scale_gm, layoutPerTokenScale.GetTileLayout(problemShape.template GetCoordByAxis<0>()),
            biasGM_, layoutBias
        };
        blockEpilogue.UpdateParams(epilogueParams);

        blockEpilogue(coreOffset, actualBlockShape, gmC[gmOffsetC], layoutC, gmD[gmOffsetC], layoutC);
    }
}

template <TemplateA2AMMClass>
__aicore__ inline void AlltoAllMatmul<TemplateA2AMMFunc>::AlltoAll()
{
    if ASCEND_IS_AIV {
        ResetIpcFlags(BUFFER_NUM);
        PipeBarrier<PIPE_ALL>();

        int64_t src_offset = 0;
        for (int32_t cal_idx = 0; cal_idx <= cal_count; ++cal_idx) {
            uint64_t flag_idx = cal_idx % peer_mem_block_count;

            if (cal_idx == cal_count - 1) {
                num_per_rank_move = data_size_per_rank - src_offset;
            }

            if (cal_idx >= peer_mem_block_count && cal_idx < cal_count) {
                WaitEvent(flag_idx);
            }

            SetAndWaitAivSync(flag_idx);
            if (cal_idx < cal_count) {
                CrossRankSyncV1(FLAG_ZERO_IDX, cal_idx + 1);
            }
            SetAndWaitAivSync(flag_idx);

            if (aiv_idx == 0 && cal_idx < cal_count && core_idx < first_step_core_num) {
                int32_t dst_rank = core_idx / core_num_per_rank;
                int32_t dst_loc = core_idx % core_num_per_rank;
                int32_t data_src_in_move = dst_loc * data_per_core;
                int32_t data_len = data_src_in_move + data_per_core > num_per_rank_move ?
                        num_per_rank_move - data_src_in_move : data_per_core;
                int64_t data_src = dst_rank * data_size_per_rank + src_offset + data_src_in_move;
                int64_t data_dst = flag_idx * peer_mem_block_size + data_src_in_move * rank_size + rank * k;

                if (data_len > 0) {
                    MoveResultFromSrcToPeerMem(reinterpret_cast<__gm__ AType*>(aGM_) + data_src, (__gm__ AType *)buff[dst_rank] + data_dst, data_len / k);
                }
                src_offset += num_per_rank_move;
            }
            else if (aiv_idx == 1 && cal_idx > 0 && core_idx >= first_step_core_num && core_idx < core_count) {
                int32_t block_dst = ((cal_idx - 1) % peer_mem_block_count) * peer_mem_block_size;
                int32_t total_m = cal_idx == cal_count ? m / rank_size - (cal_idx - 1) * num_per_rank_m : num_per_rank_m;
                int32_t m_per_core = DivCeil(total_m, second_step_core_num);
                int32_t m_st = (core_idx - first_step_core_num) * m_per_core;
                int32_t m_len = m_st + m_per_core > total_m ? total_m - m_st : m_per_core;
                int64_t src_st = block_dst + m_st * mid_output_k_size;
                int64_t dst_st = ((cal_idx - 1) * num_per_rank_m + m_st) * mid_output_k_size;
                if (m_len > 0) {
                    MoveResultFromPeerMemToOutput((__gm__ AType *)buff[rank] + src_st, reinterpret_cast<__gm__ AllToAllResultType*>(allToAllResultGM_) + dst_st, m_len);
                }
            }

            SetAndWaitAivSync(flag_idx);
            if (cal_idx < cal_count) {
                CrossRankSyncV1(FLAG_ONE_IDX, cal_idx + 1);
            }
            SetAndWaitAivSync(flag_idx);

            if (AscendC::IsSameType<BType, int8_t>::value && cal_idx < cal_count) {  // 拷贝完成后，对左矩阵进行quant
                Quant(flag_idx, cal_idx);
                SetAndWaitAivSync(flag_idx);
            }

            if (cal_idx < cal_count) {
                SetAicSync(flag_idx);
            }
        }

        WaitEvent(FLAG_ZERO_IDX);
        if (cal_count % 2 == 0) {  // 若AIC计算次数为偶数，则多等一次
            WaitEvent(FLAG_ONE_IDX);
        }

        if constexpr (AscendC::IsSameType<BType, int8_t>::value) {
            SetAndWaitAivSync(FLAG_ONE_IDX);
            Dequant();
        }

        PipeBarrier<PIPE_ALL>();
        ResetIpcFlags(1);
    }
}

template <TemplateA2AMMClass>
__aicore__ inline void AlltoAllMatmul<TemplateA2AMMFunc>::Process()
{
    AlltoAll();
    CatlassMatmul();
    SyncAll<false>();
}

} // AlltoAllMatmulImpl
#endif // ALL_TO_ALL_MATMUL_H