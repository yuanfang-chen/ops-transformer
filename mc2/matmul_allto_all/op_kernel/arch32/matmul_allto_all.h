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
 * \file matmul_allto_all.h
 * \brief
 */

#ifndef MATMUL_ALLTO_ALL_H
#define MATMUL_ALLTO_ALL_H

using namespace AscendC;

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "matmul_allto_all_tiling.h"
#include "moe_distribute_base.h"
#include "matmul_allto_all_util.h"
#include "../../3rd/template_linear_algebra/include/template_linear_algebra/catlass.hpp"
#include "../../3rd/template_linear_algebra/include/template_linear_algebra/arch/arch.hpp"
#include "../../3rd/template_linear_algebra/include/template_linear_algebra/layout/layout.hpp"
#include "../../3rd/template_linear_algebra/include/template_linear_algebra/gemm/block/block_mmad.hpp"
#include "../../3rd/template_linear_algebra/include/template_linear_algebra/gemm/block/block_swizzle.hpp"
#include "../../3rd/template_linear_algebra/include/template_linear_algebra/gemm/dispatch_policy.hpp"
#include "../../3rd/template_linear_algebra/include/template_linear_algebra/gemm/gemm_type.hpp"
#include "../../3rd/template_linear_algebra/include/template_linear_algebra/gemm_coord.hpp"
#include "../../3rd/template_linear_algebra/include/template_linear_algebra/status.hpp"
#include "matmul.hpp"

using namespace Catlass;

namespace MatmulAlltoAllImpl {

template<AscendC::HardEvent event>
__aicore__ inline void SyncFunc() {
    int32_t eventID = static_cast<int32_t>(GetTPipePtr()->FetchEventID(event));
    AscendC::SetFlag<event>(eventID);
    AscendC::WaitFlag<event>(eventID);
}
// MMA2A : MatmulAlltoAll
#define TemplateMMA2AClass typename AType, typename BType, typename BiasType, typename cType, bool TB, bool hasBias
#define TemplateMMA2AFunc AType, BType, BiasType, cType, TB, hasBias

using namespace AscendC;
template <TemplateMMA2AClass>
class MatmulAlltoAll : public CommBase{
public:
    __aicore__ inline MatmulAlltoAll() {};
    __aicore__ inline void Init(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR cGM,
                                GM_ADDR workspaceGM, GM_ADDR tilingGM);
    __aicore__ inline void Process();

private:
    __aicore__ inline void AIVInit();
    __aicore__ inline void AICInit();
    __aicore__ inline void CatlassMatmul();

private:
    GM_ADDR aGM_;
    GM_ADDR bGM_;
    GM_ADDR cGM_;
    GM_ADDR biasGM_;

    int32_t cal_count;
    int32_t gm_a_pingpong_size;
    uint32_t rank_size{0};
    uint32_t rank{0};
    static constexpr bool has_bias = hasBias;

    __gm__ cType* gm_peer_mem;
};


template <TemplateMMA2AClass>
__aicore__ inline void MatmulAlltoAll<TemplateMMA2AFunc>::Init(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR cGM,
                                                               GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    REGISTER_TILING_DEFAULT(MatmulAlltoAllTilingData);
    auto tiling = (__gm__ MatmulAlltoAllTilingData*)tilingGM;
    GET_TILING_DATA(tilingData, tilingGM);

    rank_size = tilingData.matmulAlltoAllInfo.worldSize;
    aGM_ = aGM;
    bGM_ = bGM;
    cGM_ = cGM;
    biasGM_ = biasGM;

    CommBase::SetArgs(&rank, rank_size, tilingData);

    MatmulAlltoAll<TemplateMMA2AFunc>::AICInit();

    MatmulAlltoAll<TemplateMMA2AFunc>::AIVInit();
}

template <TemplateMMA2AClass>
__aicore__ inline void MatmulAlltoAll<TemplateMMA2AFunc>::AICInit()
{
    if ASCEND_IS_AIC {
        SetLoadDataPaddingValue(0);
        SetAtomicNone();
        SetFixpipeNz2ndFlag(1, 0, 0);
        gm_peer_mem = reinterpret_cast<__gm__ cType*>(buff[rank]);
    }
}

template <TemplateMMA2AClass>
__aicore__ inline void MatmulAlltoAll<TemplateMMA2AFunc>::AIVInit()
{
    if ASCEND_IS_AIV {
        SetAtomicNone();
        SetMaskNormImpl();
        SetVectorMask<int32_t>((uint64_t)-1, (uint64_t)-1);

        cal_count = DivCeil(m_loop, p_value);
        gm_a_pingpong_size = m0 * n * p_value;
    }
}

template <TemplateMMA2AClass>
__aicore__ inline void MatmulAlltoAll<TemplateMMA2AFunc>::CatlassMatmul()
{
    if ASCEND_IS_AIC {
        using ArchTag = Arch::AtlasA2;

        constexpr bool ENABLE_UNIT_FLAG = false;
        constexpr bool ENABLE_SHUFFLE_K = false;
        using ElementA = cType;
        using ElementB = cType;
        using ElementC = cType;
        using ElementBias = BiasType;
        using LayoutA = layout::RowMajor;
        // 支持B转置场景
        using LayoutB = typename std::conditional<TB, layout::ColumnMajor, layout::RowMajor>::type;
        using LayoutBias = layout::VectorLayout;

        using LayoutC = layout::RowMajor;
        LayoutA layoutA{static_cast<uint32_t>(m), static_cast<uint32_t>(k)};
        LayoutB layoutB{static_cast<uint32_t>(k), static_cast<uint32_t>(n)};
        LayoutC layoutC{static_cast<uint32_t>(m * rank_size), static_cast<uint32_t>(n / rank_size)};
        LayoutBias layoutBias{static_cast<uint32_t>(n)};

        using DispatchPolicy = std::conditional_t<has_bias, Gemm::MmadAtlasA2PingpongBias<ENABLE_UNIT_FLAG>,
                                                           Gemm::MmadAtlasA2Preload<ENABLE_UNIT_FLAG, ENABLE_SHUFFLE_K>>;
        
        using AType_ = Gemm::GemmType<ElementA, LayoutA>;
        using BType_ = Gemm::GemmType<ElementB, LayoutB>;
        using CType_ = Gemm::GemmType<ElementC, LayoutC>;
        using BiasType_ = std::conditional_t<std::is_same_v<BiasType, void>, void, Gemm::GemmType<ElementBias, LayoutBias>>;

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

        using BlockEpilogue = void;
        using BlockScheduler30 = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;
        GemmCoord processSize{static_cast<uint32_t>(m), static_cast<uint32_t>(n), static_cast<uint32_t>(k)};
        if (m0 == 128) {
            using L1TileShape = GemmShape<128, 256, 256>;
            using L0TileShape = GemmShape<128, 256, 64>;
            using BlockMmadOpt = Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, AType_, BType_, CType_, BiasType_, TileCopy>;
            using MatmulKernel = Gemm::Kernel::MatmulAlltoAllKernel<void, void, BlockMmadOpt, BlockEpilogue, BlockScheduler30, has_bias>;
            MatmulKernel matmul_op;
            typename MatmulKernel::Params params{processSize,
                                                    reinterpret_cast<GM_ADDR>(aGM_), layoutA,
                                                    reinterpret_cast<GM_ADDR>(bGM_), layoutB,
                                                    reinterpret_cast<GM_ADDR>(biasGM_),
                                                    reinterpret_cast<GM_ADDR>(gm_peer_mem), layoutC,
                                                    p_value, static_cast<int32_t>(rank_size), MAX_BLOCK_COUNT, TB};
            matmul_op(params);
        } else {
            using L1TileShape = GemmShape<256, 128, 256>;
            using L0TileShape = GemmShape<256, 128, 64>;
            using BlockMmadOpt = Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, AType_, BType_, CType_, BiasType_, TileCopy>;
            using MatmulKernel = Gemm::Kernel::MatmulAlltoAllKernel<void, void, BlockMmadOpt, BlockEpilogue, BlockScheduler30, has_bias>;
            MatmulKernel matmul_op;
            typename MatmulKernel::Params params{processSize,
                                                    reinterpret_cast<GM_ADDR>(aGM_), layoutA,
                                                    reinterpret_cast<GM_ADDR>(bGM_), layoutB,
                                                    reinterpret_cast<GM_ADDR>(biasGM_),
                                                    reinterpret_cast<GM_ADDR>(gm_peer_mem), layoutC,
                                                    p_value, static_cast<int32_t>(rank_size), MAX_BLOCK_COUNT, TB};
            matmul_op(params);
        }
    }
}

template <TemplateMMA2AClass>
__aicore__ inline void MatmulAlltoAll<TemplateMMA2AFunc>::Process()
{
    CatlassMatmul();
    if ASCEND_IS_AIV {
        ResetIpcFlags(2);
        PipeBarrier<PIPE_ALL>();

        for (int32_t cal_idx = 0; cal_idx < cal_count; ++cal_idx) {
            int32_t actual_p_value = p_value;

            int32_t token_total = p_value * m0 * n;
            if (cal_idx == cal_count - 1) {
                token_total = (m - (cal_idx * m0 * p_value)) * n;
            }
            int32_t token_per_rank = token_total / rank_size;

            uint64_t flag_idx = cal_idx % MAX_BLOCK_COUNT;
            WaitEvent(flag_idx);

            SetAndWaitAivSync(flag_idx);
            CrossRankSyncV1(FLAG_ZERO_IDX, cal_idx + 1);
            SetAndWaitAivSync(flag_idx);

            int32_t rank_offset = m * n / rank_size;
            if (aiv_idx == 0 && core_idx < rank_size) {
                int64_t src_offset = flag_idx * gm_a_pingpong_size + gm_a_pingpong_size / rank_size * rank;
                int64_t dst_offset = core_idx * rank_offset + cal_idx * m0 * p_value * (n / rank_size);
                SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
                SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
                CopyGMToGM((__gm__ cType *)buff[core_idx] + src_offset, reinterpret_cast<__gm__ cType*>(cGM_) + dst_offset, token_per_rank);
                WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
                WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
            }

            SetAndWaitAivSync(flag_idx);
            CrossRankSyncV1(FLAG_ONE_IDX, cal_idx + 1);
            SetAndWaitAivSync(flag_idx);

            SetAicSync(flag_idx);
        }
        PipeBarrier<PIPE_ALL>();
        ResetIpcFlags(1);
    }
    SyncAll<false>();
}
} // MatmulAlltoAllImpl
#endif // MATMUL_ALLTO_ALL_H