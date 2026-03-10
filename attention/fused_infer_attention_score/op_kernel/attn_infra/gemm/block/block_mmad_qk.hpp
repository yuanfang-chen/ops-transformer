/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GEMM_BLOCK_MMAD_QK_HPP
#define GEMM_BLOCK_MMAD_QK_HPP

#include "../../../attn_infra/base_defs.hpp"
#include "../../../attn_infra/arch/resource.hpp"
#include "../../../attn_infra/coord.hpp"
#include "../../../attn_infra/gemm/dispatch_policy.hpp"
#include "../../../attn_infra/gemm/helper.hpp"
#include "../../../attn_infra/gemm_coord.hpp"
#include "../../../attn_infra/gemm/tile_common/tile_copy.hpp"
#include "../../../attn_infra/gemm/tile_common/tile_mmad.hpp"
////////////////////////////////////////////////////////////////////
namespace NpuArch::Gemm::Block {
////////////////////////////////////////////////////////////////////

template <
    bool PAGED_CACHE_FLAG_,
    bool ENABLE_UNIT_FLAG_,
    class L1TileShape_,
    class L0TileShape_,
    class AType_,
    class BType_,
    class CType_,
    class BiasType_,
    class TileCopy_,
    class TileMmad_>
struct BlockMmad<
    MmadAtlasA2FAIQK<PAGED_CACHE_FLAG_, ENABLE_UNIT_FLAG_>,
    L1TileShape_,
    L0TileShape_,
    AType_,
    BType_,
    CType_,
    BiasType_,
    TileCopy_,
    TileMmad_> {
public:
    // Type Aliases
    using DispatchPolicy = MmadAtlasA2FAIQK<PAGED_CACHE_FLAG_, ENABLE_UNIT_FLAG_>;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using L1TileShape = L1TileShape_;
    using L0TileShape = L0TileShape_;
    using ElementA = typename AType_::Element;
    using LayoutA = typename AType_::Layout;
    using ElementB = typename BType_::Element;
    using LayoutB = typename BType_::Layout;
    using ElementC = typename CType_::Element;
    using LayoutC = typename CType_::Layout;
    using TileMmad = TileMmad_;
    using CopyGmToL1A = typename TileCopy_::CopyGmToL1A;
    using CopyGmToL1B = typename TileCopy_::CopyGmToL1B;
    using CopyL1ToL0A = typename TileCopy_::CopyL1ToL0A;
    using CopyL1ToL0B = typename TileCopy_::CopyL1ToL0B;
    using CopyL0CToGm = typename TileCopy_::CopyL0CToGm;
    using ElementAccumulator =
        typename Gemm::helper::ElementAccumulatorSelector<ElementA, ElementB>::ElementAccumulator;
    using LayoutAInL1 = typename CopyL1ToL0A::LayoutSrc;
    using LayoutBInL1 = typename CopyL1ToL0B::LayoutSrc;
    using LayoutAInL0 = typename CopyL1ToL0A::LayoutDst;
    using LayoutBInL0 = typename CopyL1ToL0B::LayoutDst;
    using LayoutCInL0 = layout::zN;

    using L1AAlignHelper = Gemm::helper::L1AlignHelper<ElementA, LayoutA>;
    using L1BAlignHelper = Gemm::helper::L1AlignHelper<ElementB, LayoutB>;

    static constexpr uint32_t STAGES = DispatchPolicy::STAGES;
    static constexpr uint32_t L1A_SIZE = L1TileShape::M * L1TileShape::K * sizeof(ElementA);
    static constexpr uint32_t L1B_SIZE = L1TileShape::N * L1TileShape::K * sizeof(ElementB);
    static constexpr uint32_t L0A_SIZE = ArchTag::L0A_SIZE;
    static constexpr uint32_t L0B_SIZE = ArchTag::L0B_SIZE;
    static constexpr uint32_t L0C_SIZE = ArchTag::L0C_SIZE;
    static constexpr uint32_t L0A_PINGPONG_BUF_SIZE = L0A_SIZE / STAGES;
    static constexpr uint32_t L0B_PINGPONG_BUF_SIZE = L0B_SIZE / STAGES;
    static constexpr uint32_t L0C_PINGPONG_BUF_SIZE = L0C_SIZE / STAGES;
    static constexpr uint32_t BLOCK_SIZE = 16;
    static constexpr uint32_t EMBED_SPLIT_SIZE = 128;
    static constexpr uint32_t UNIT_BLOCK_STACK_NUM = 4;
    static constexpr uint32_t KV_BASE_BLOCK = 512;
    static constexpr uint32_t KV_SPLIT_SIZE = 128;
    static constexpr uint32_t COORD_DIM0 = 0;
    static constexpr uint32_t COORD_DIM1 = 1;
    static constexpr uint32_t COORD_DIM2 = 2;

    static_assert(std::is_same_v<LayoutC, layout::RowMajor>, "LayoutC only support RowMajor yet!");

    __aicore__ inline
    BlockMmad() {}

    __aicore__ inline
    void init(Arch::Resource<ArchTag> &resource, uint32_t nDyn, uint32_t kDyn, uint32_t KVStackLen = 512, uint32_t l1BufAddrStart = 0)
    {   
        maxKVStackLen = KVStackLen;
        // Allocate L1 memory space
        l1ATensor = resource.l1Buf.template GetBufferByByte<ElementA>(l1BufAddrStart);
        for (uint32_t i = 0; i < STAGES; i++) {
            l1BTensor[i] = resource.l1Buf.template GetBufferByByte<ElementB>(l1BufAddrStart +
                L1TileShape::M * kDyn * sizeof(ElementA) + nDyn * kDyn * sizeof(ElementB) * i);
            l0ATensor[i] = resource.l0ABuf.template GetBufferByByte<ElementA>(L0A_PINGPONG_BUF_SIZE * i);
            l0BTensor[i] = resource.l0BBuf.template GetBufferByByte<ElementB>(L0B_PINGPONG_BUF_SIZE * i);
            l0CTensor[i] = resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(L0C_PINGPONG_BUF_SIZE * i);
        }
        l1NDynamic = nDyn;
        l1KDynamic = kDyn;
    }

    __aicore__ inline
    ~BlockMmad() {}

    __aicore__ inline
    void loadQGM(
        AscendC::GlobalTensor<ElementA> gA,
        LayoutA layoutA,
        uint32_t rowNum, uint32_t &singleGroupHeads, uint32_t &qHeads)
    {
        uint32_t embed = layoutA.shape(1);
        uint32_t rowNumRound = NpuArch::Detail::Alignment::RoundUp(rowNum, L1AAlignHelper::M_ALIGNED);
        uint32_t tokenNumPerGroup = rowNum / singleGroupHeads;
        auto layoutSingleANd = layoutA.GetTileLayout(MakeCoord(singleGroupHeads, embed));
        LayoutAInL1 layoutAInL1 = LayoutAInL1::template MakeLayout<ElementA>(rowNum, embed);
        copyGmToL1A(
            l1ATensor, gA,
            layoutAInL1, layoutSingleANd,
            tokenNumPerGroup, qHeads * embed, tokenNumPerGroup, BLOCK_SIZE, rowNumRound);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID3);
    }

    __aicore__ inline
    void loadQGM(
        AscendC::GlobalTensor<ElementA> gA,
        LayoutA layoutA,
        uint32_t rowNum, uint32_t &singleGroupHeads, uint32_t &qHeads,
        uint32_t kvNBlockSizeParam = 1)
    {
        uint32_t embed = layoutA.shape(1);
        // 对齐到16的倍数，表示ND转NZ之后，源操作数的一行转化为NZ的多行的行数，一行长度是16
        uint32_t rowNumRound = NpuArch::Detail::Alignment::RoundUp(rowNum, L1AAlignHelper::M_ALIGNED);
        // 当前场景下，应该表示的是qsblocksize * kvnblocksize。？是改成qs * g * kvn，将Q一行行的搬运到L1上呢，还是将Q一把全部搬移进去呢。
        uint32_t tokenNumPerGroup = rowNum / singleGroupHeads;
        auto layoutSingleANd = layoutA.GetTileLayout(MakeCoord(singleGroupHeads, embed));
        // layoutA: shape:(rowNum, embed); stride:(embed, 1)
        // layoutSingleANd: shape:(singleGroupHeads, embed); stride:(embed, 1)
        LayoutAInL1 layoutAInL1 = LayoutAInL1::template MakeLayout<ElementA>(rowNum, embed);
        // AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID7);
        copyGmToL1A(
            l1ATensor, gA,
            layoutAInL1, layoutSingleANd,
            tokenNumPerGroup, qHeads * embed, tokenNumPerGroup, BLOCK_SIZE, rowNumRound);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID3);
        // AscendC::DumpTensor(l1ATensor, 137137, 32);
        
        // 保存L1上Q矩阵的总行数和每个kvhead对应的行数，用于后续跳着取数据
        l1ATotalRowNum = rowNum;
        l1ARowNumPerKvHead = rowNum / kvNBlockSizeParam;
        l1AEmbedDim = embed;
        l1AKvNBlockSize = kvNBlockSizeParam;
    }
    
    __aicore__ inline
    void setBlockParam(uint32_t stackSeqTile, uint32_t &blockStart, uint32_t &blockEnd, uint32_t &curBlockTotalNum, uint32_t blockSize){
        if(stackSeqTile >= blockStart && blockSize != 0) {
            blockEnd = ((stackSeqTile - blockStart) % blockSize == 0) ? blockSize : (stackSeqTile - blockStart) % blockSize;
            curBlockTotalNum = (((stackSeqTile - blockStart) + blockSize - 1) / blockSize) + 1;
        } else {
            curBlockTotalNum = 1;
            blockStart = stackSeqTile;
            blockEnd = stackSeqTile + blockStartOffset;
        }
    }

    __aicore__ inline
    void getBlockShape(GemmCoord &actualShape, uint32_t nL1Idx, uint32_t nL1Loop, uint32_t stackSeqTile)
    {
        uint32_t nSplitSize = l1NDynamic;
        if (nL1Idx == nL1Loop - 1U) {
            nSplitSize = stackSeqTile - nL1Idx * l1NDynamic;
        }
        actualShape[COORD_DIM1] = nSplitSize;
    }

     __aicore__ inline
    void getBlockShape(GemmCoord &actualShape, uint32_t& blockStartOffset, uint32_t& l1NResDynamic, uint32_t& kvL1Len, uint32_t& nowLen, uint32_t& blockSize)
    {
        nowLen = (blockSize - blockStartOffset < l1NResDynamic - kvL1Len) ?
                 blockSize - blockStartOffset :
                 l1NResDynamic - kvL1Len;
        actualShape[COORD_DIM1] = nowLen;
    }

    __aicore__ inline
    void getKVOffset(uint32_t &kOffset, uint32_t nIdx, uint32_t nowNIdx, uint32_t strideKV)
    {
        kOffset = nIdx * maxKVStackLen * strideKV + nowNIdx * l1NDynamic * strideKV;
    }

    __aicore__ inline
    void getKVOffset(AscendC::GlobalTensor<int32_t> &gBlockTable, uint32_t &kOffset, uint32_t nowNIdx, 
         uint32_t startOffset, uint32_t strideKV, uint32_t blockSize)
    {
        uint32_t blockTableId = gBlockTable.GetValue(nowNIdx);
        kOffset = blockTableId * blockSize * strideKV + startOffset * strideKV;
    }

    __aicore__ inline
    void resetBlockStart(){
        blockStartOffset = 0;
    }

    __aicore__ inline
    void updateBlockOffset(uint32_t nowLen, uint32_t &curBlockIdx, uint32_t blockSize){
        if(blockStartOffset + nowLen == blockSize){
            blockStartOffset = 0;
            curBlockIdx++;
        } else{
            blockStartOffset += nowLen;
        }
    }

    __aicore__ inline
    void operator()(AscendC::GlobalTensor<ElementA> gA,
                    AscendC::GlobalTensor<ElementB> gB,
                    AscendC::GlobalTensor<ElementC> gC,
                    AscendC::GlobalTensor<int32_t> gBlockTable,
                    LayoutA layoutA, LayoutB layoutB, LayoutC layoutC, GemmCoord actualOriShape,
                    uint32_t nIdx, uint32_t nLoop, uint32_t blockSize, uint32_t strideKV,
                    uint32_t kvNIncreIdx = 0)
    {
        uint32_t rowNum = actualOriShape[COORD_DIM0];
        uint32_t stackSeqTile = actualOriShape[COORD_DIM1];
        uint32_t embed = actualOriShape[COORD_DIM2];

        GemmCoord actualShape{rowNum, 0, embed};
        uint32_t gBOffset = 0;

        // 使用loadQGM时保存的总行数创建L1布局，以便正确计算跨kvhead的偏移
        // l1ATotalRowNum是所有kvhead的Q数据在L1上的总行数
        // l1ARowNumPerKvHead是单个kvhead对应的行数
        LayoutAInL1 layoutAInL1 = LayoutAInL1::template MakeLayout<ElementA>(l1ATotalRowNum, l1AEmbedDim);
        
        // 计算当前kvhead在L1上的行偏移
        // 当kvNIncreIdx = 0时，取N0, N1（第一个kvhead对应的qhead）
        // 当kvNIncreIdx = 1时，取N2, N3（第二个kvhead对应的qhead）
        uint32_t kvNRowOffset = kvNIncreIdx * l1ARowNumPerKvHead;
        AscendC::printf("kvNIncreIdx: %d, kvNRowOffset: %d, l1ARowNumPerKvHead: %d\n", kvNIncreIdx, kvNRowOffset, l1ARowNumPerKvHead);

        uint32_t tileNNumPerBaseBlock = blockSize / l1NDynamic;
        uint32_t nL1Loop = NpuArch::Detail::Alignment::CeilDiv(stackSeqTile, l1NDynamic);
        for (uint32_t nL1Idx = 0; nL1Idx < nL1Loop; ++nL1Idx) {
            uint32_t nowNIdx = nIdx + nL1Idx / tileNNumPerBaseBlock;
            getBlockShape(actualShape, nL1Idx, nL1Loop, stackSeqTile);
            getKVOffset(gBlockTable, gBOffset, nowNIdx, nL1Idx % tileNNumPerBaseBlock, strideKV, blockSize);
            uint32_t mActual = actualShape.m();
            uint32_t kActual = actualShape.k();
            uint32_t nActual = actualShape.n();
            LayoutBInL1 layoutBInL1 = LayoutBInL1::template MakeLayout<ElementB>(kActual, nActual);

            auto layoutBTile = layoutB.GetTileLayout(MakeCoord(kActual, nActual));
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1KvPingPongFlag);
            copyGmToL1B(l1BTensor[l1KvPingPongFlag], gB[gBOffset], layoutBInL1, layoutBTile);
            // if(nIdx != 0){
            //     AscendC::printf("---- nL1Idx %d nL1Loop %d l1NDynamic %d\n", nL1Idx, nL1Loop, l1NDynamic);
            //     AscendC::DumpTensor(l1BTensor[l1KvPingPongFlag], 10000 + __LINE__, 4);
            // }
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1KvPingPongFlag);

            uint32_t mL0Loop = NpuArch::Detail::Alignment::CeilDiv(mActual, L0TileShape::M);
            uint32_t kL0Loop = NpuArch::Detail::Alignment::CeilDiv(kActual, L0TileShape::K);
            for (uint32_t mL0Idx = 0; mL0Idx < mL0Loop; mL0Idx++) {
                uint32_t mL0Actual = (mL0Idx < mL0Loop - 1U) ? L0TileShape::M : (mActual - mL0Idx * L0TileShape::M);
                AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0CPingPongFlag);
                for (uint32_t kL0Idx = 0; kL0Idx < kL0Loop; kL0Idx++) {
                    uint32_t kL0Actual = (kL0Idx < kL0Loop - 1U) ? L0TileShape::K : (kActual - kL0Idx * L0TileShape::K);

                    LayoutAInL0 layoutAInL0 = LayoutAInL0::template MakeLayout<ElementA>(mL0Actual, kL0Actual);
                    // 加入kvNRowOffset，实现跳着取数据：
                    // 当kvNIncreIdx = 0时，取L1上的N0, N1行（第一个kvhead对应的qhead）
                    // 当kvNIncreIdx = 1时，取L1上的N2, N3行（第二个kvhead对应的qhead）
                    MatrixCoord l1ATileCoord{mL0Idx * L0TileShape::M + kvNRowOffset, kL0Idx * L0TileShape::K};
                    auto l1ATile = l1ATensor[layoutAInL1.GetOffset(l1ATileCoord)];
                    auto offset = layoutAInL1.GetOffset(l1ATileCoord);
                    AscendC::printf("kvNIncreIdx: %d, kvNRowOffset: %d, l1ATile offset: %d\n", kvNIncreIdx, kvNRowOffset, offset);
                    AscendC::DumpTensor(l1ATile, 206206, 256);

                    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0ABPingPongFlag);
                    copyL1ToL0A(l0ATensor[l0ABPingPongFlag], l1ATile, layoutAInL0, layoutAInL1);
                    // if ((nIdx == 0U) && (nL1Idx == nL1Loop - 1U) && (mL0Idx == mL0Loop - 1U) && (kL0Idx == kL0Loop - 1U)) {
                    //     AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID7);
                    // }

                    LayoutBInL0 layoutBInL0 = LayoutBInL0::template MakeLayout<ElementB>(kL0Actual, nActual);
                    MatrixCoord l1BTileCoord{kL0Idx * L0TileShape::K, 0};
                    auto l1BTile = l1BTensor[l1KvPingPongFlag][layoutBInL1.GetOffset(l1BTileCoord)];
                    if ((mL0Idx == 0U) && (kL0Idx == 0U)) {
                        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1KvPingPongFlag);
                    }
                    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0ABPingPongFlag + 2U);
                    copyL1ToL0B(l0BTensor[l0ABPingPongFlag], l1BTile, layoutBInL0, layoutBInL1);
                    if ((mL0Idx == mL0Loop - 1U) && (kL0Idx == kL0Loop - 1U)) {
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1KvPingPongFlag);
                    }

                    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);
                    bool initMmad = (kL0Idx == 0U);
                    uint32_t mL0Align = (mL0Actual + BLOCK_SIZE - 1U) / BLOCK_SIZE * BLOCK_SIZE;
                    AscendC::printf("mL0Align: %d, nActual: %d, kL0Actual: %d, initMmad: %d\n", mL0Align, nActual, kL0Actual, initMmad);
                    tileMmad(l0CTensor[l0CPingPongFlag],
                        l0ATensor[l0ABPingPongFlag],
                        l0BTensor[l0ABPingPongFlag],
                        mL0Align,
                        nActual,
                        kL0Actual,
                        initMmad);
                    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0ABPingPongFlag);
                    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0ABPingPongFlag + 2U);
                    // if(nIdx != 0){
                    //     AscendC::DumpTensor(l1ATile, 10000 + __LINE__, 32);
                    //     AscendC::DumpTensor(l1BTile, 10000 + __LINE__, 32);
                    //     AscendC::DumpTensor(l0CTensor[l0CPingPongFlag], 10000 + __LINE__, 36);
                    // }
                    l0ABPingPongFlag = 1U - l0ABPingPongFlag;
                }
                AscendC::SetFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);
                MatrixCoord gmCTileCoord{mL0Idx * L0TileShape::M, nL1Idx * l1NDynamic};
                LayoutC layoutCTile = layoutC.GetTileLayout(MakeCoord(mL0Actual, nActual));
                auto layoutInL0C = LayoutCInL0::MakeLayoutInL0C(MakeCoord(mL0Actual, nActual));
                copyL0CToGm(gC[layoutC.GetOffset(gmCTileCoord)], l0CTensor[l0CPingPongFlag], layoutCTile, layoutInL0C);
                // AscendC::DumpTensor(l0CTensor[l0CPingPongFlag], 777888, 16);
                // AscendC::DumpTensor(gC[layoutC.GetOffset(gmCTileCoord)], 777888, 16);
                AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0CPingPongFlag);
                l0CPingPongFlag = 1U - l0CPingPongFlag;
            }
            l1KvPingPongFlag = 1U - l1KvPingPongFlag;
        }
    }

    __aicore__ inline
    void operator()(AscendC::GlobalTensor<ElementA> gA,
                    AscendC::GlobalTensor<ElementB> gB,
                    AscendC::GlobalTensor<ElementC> gC,
                    AscendC::GlobalTensor<int32_t> gBlockTable,
                    LayoutA layoutA, LayoutB layoutB, LayoutC layoutC, GemmCoord actualOriShape,
                    uint32_t nIdx, uint32_t nLoop, uint32_t blockSize, uint32_t strideKV)
    {
        uint32_t rowNum = actualOriShape[COORD_DIM0];
        uint32_t stackSeqTile = actualOriShape[COORD_DIM1];
        uint32_t embed = actualOriShape[COORD_DIM2];
        GemmCoord actualShape{rowNum, 0, embed};
        uint32_t gBOffset = 0;
        LayoutAInL1 layoutAInL1 = LayoutAInL1::template MakeLayout<ElementA>(rowNum, embed);
        uint32_t tileNNumPerBaseBlock = blockSize / l1NDynamic;
        uint32_t nL1Loop = NpuArch::Detail::Alignment::CeilDiv(stackSeqTile, l1NDynamic);
        uint32_t curBlockIdx =  0;
        uint32_t blockStart = 0;
        uint32_t blockEnd = 0;
        uint32_t curBlockTotalNum = 0;
        if constexpr (PAGED_CACHE_FLAG_){
            blockStart = blockSize - blockStartOffset;
            setBlockParam(stackSeqTile, blockStart, blockEnd, curBlockTotalNum, blockSize);
        }
        for (uint32_t nL1Idx = 0; nL1Idx < nL1Loop; ++nL1Idx) {
            uint32_t mActual = actualShape.m();
            uint32_t kActual = actualShape.k();
            uint32_t nActual = actualShape.n();
            LayoutBInL1 layoutBInL1 = LayoutBInL1::template MakeLayout<ElementB>(kActual, nActual);
            if constexpr (PAGED_CACHE_FLAG_){
                uint32_t l1NResDynamic = (nL1Idx < (nL1Loop-1)) ? l1NDynamic : (stackSeqTile - nL1Idx * l1NDynamic);
                layoutBInL1 = LayoutBInL1::template MakeLayout<ElementB>(embed, l1NResDynamic);
                uint32_t kvL1Len = 0;
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1KvPingPongFlag);
                while(kvL1Len < l1NResDynamic){
                    uint32_t nowLen = 0;
                    uint32_t curBlockSize = (curBlockIdx < (curBlockTotalNum-1)) ? blockSize : blockEnd;
                    uint32_t nowNIdx = nIdx * maxKVStackLen / blockSize + curBlockIdx;
                    getBlockShape(actualShape, blockStartOffset, l1NResDynamic, kvL1Len, nowLen, curBlockSize);
                    getKVOffset(gBlockTable, gBOffset, nowNIdx, blockStartOffset, strideKV, blockSize);
                    auto layoutBTile = layoutB.GetTileLayout(MakeCoord(embed, nowLen));
                    MatrixCoord l1BTileCoord{0, kvL1Len};
                    auto l1BTile = l1BTensor[l1KvPingPongFlag][layoutBInL1.GetOffset(l1BTileCoord)];
                    copyGmToL1B(l1BTile, gB[gBOffset], layoutBInL1, layoutBTile);
                    kvL1Len += nowLen;
                    updateBlockOffset(nowLen, curBlockIdx, blockSize);
                }
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1KvPingPongFlag);
                mActual = actualShape.m();
                kActual = actualShape.k();
                nActual = l1NResDynamic;
            } else {
                getBlockShape(actualShape, nL1Idx, nL1Loop, stackSeqTile);
                getKVOffset(gBOffset, nIdx, nL1Idx, strideKV);
                mActual = actualShape.m();
                kActual = actualShape.k();
                nActual = actualShape.n();
                layoutBInL1 = LayoutBInL1::template MakeLayout<ElementB>(kActual, nActual);

                auto layoutBTile = layoutB.GetTileLayout(MakeCoord(kActual, nActual));
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1KvPingPongFlag);
                copyGmToL1B(l1BTensor[l1KvPingPongFlag], gB[gBOffset], layoutBInL1, layoutBTile);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1KvPingPongFlag);
            }
            uint32_t mL0Loop = NpuArch::Detail::Alignment::CeilDiv(mActual, L0TileShape::M);
            uint32_t kL0Loop = NpuArch::Detail::Alignment::CeilDiv(kActual, L0TileShape::K);
            for (uint32_t mL0Idx = 0; mL0Idx < mL0Loop; mL0Idx++) {
                uint32_t mL0Actual = (mL0Idx < mL0Loop - 1U) ? L0TileShape::M : (mActual - mL0Idx * L0TileShape::M);
                AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0CPingPongFlag);
                for (uint32_t kL0Idx = 0; kL0Idx < kL0Loop; kL0Idx++) {
                    uint32_t kL0Actual = (kL0Idx < kL0Loop - 1U) ? L0TileShape::K : (kActual - kL0Idx * L0TileShape::K);

                    LayoutAInL0 layoutAInL0 = LayoutAInL0::template MakeLayout<ElementA>(mL0Actual, kL0Actual);
                    MatrixCoord l1ATileCoord{mL0Idx * L0TileShape::M, kL0Idx * L0TileShape::K};
                    auto l1ATile = l1ATensor[layoutAInL1.GetOffset(l1ATileCoord)];

                    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0ABPingPongFlag);
                    copyL1ToL0A(l0ATensor[l0ABPingPongFlag], l1ATile, layoutAInL0, layoutAInL1);

                    LayoutBInL0 layoutBInL0 = LayoutBInL0::template MakeLayout<ElementB>(kL0Actual, nActual);
                    MatrixCoord l1BTileCoord{kL0Idx * L0TileShape::K, 0};
                    auto l1BTile = l1BTensor[l1KvPingPongFlag][layoutBInL1.GetOffset(l1BTileCoord)];
                    if ((mL0Idx == 0U) && (kL0Idx == 0U)) {
                        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1KvPingPongFlag);
                    }
                    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0ABPingPongFlag + 2U);
                    copyL1ToL0B(l0BTensor[l0ABPingPongFlag], l1BTile, layoutBInL0, layoutBInL1);
                    if ((mL0Idx == mL0Loop - 1U) && (kL0Idx == kL0Loop - 1U)) {
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1KvPingPongFlag);
                    }

                    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);
                    bool initMmad = (kL0Idx == 0U);
                    uint32_t mL0Align = (mL0Actual + BLOCK_SIZE - 1U) / BLOCK_SIZE * BLOCK_SIZE;
                    tileMmad(l0CTensor[l0CPingPongFlag],
                        l0ATensor[l0ABPingPongFlag],
                        l0BTensor[l0ABPingPongFlag],
                        mL0Align,
                        nActual,
                        kL0Actual,
                        initMmad);
                    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0ABPingPongFlag);
                    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0ABPingPongFlag + 2U);
                    l0ABPingPongFlag = 1U - l0ABPingPongFlag;
                }
                AscendC::SetFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);
                MatrixCoord gmCTileCoord{mL0Idx * L0TileShape::M, nL1Idx * l1NDynamic};
                LayoutC layoutCTile = layoutC.GetTileLayout(MakeCoord(mL0Actual, nActual));
                auto layoutInL0C = LayoutCInL0::MakeLayoutInL0C(MakeCoord(mL0Actual, nActual));
                copyL0CToGm(gC[layoutC.GetOffset(gmCTileCoord)], l0CTensor[l0CPingPongFlag], layoutCTile, layoutInL0C);
                AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0CPingPongFlag);
                l0CPingPongFlag = 1U - l0CPingPongFlag;
            }
            l1KvPingPongFlag = 1U - l1KvPingPongFlag;
        }
    }
protected:
    /// Data members
    AscendC::LocalTensor<ElementA> l1ATensor;
    AscendC::LocalTensor<ElementB> l1BTensor[STAGES];
    AscendC::LocalTensor<ElementA> l0ATensor[STAGES];
    AscendC::LocalTensor<ElementB> l0BTensor[STAGES];
    AscendC::LocalTensor<ElementAccumulator> l0CTensor[STAGES];

    TileMmad tileMmad;
    CopyGmToL1A copyGmToL1A;
    CopyGmToL1B copyGmToL1B;
    CopyL1ToL0A copyL1ToL0A;
    CopyL1ToL0B copyL1ToL0B;
    CopyL0CToGm copyL0CToGm;

    uint32_t l1KvPingPongFlag = 0;
    uint32_t l0CPingPongFlag = 0;
    uint32_t l0ABPingPongFlag = 0;

    uint32_t l1MDynamic = 0;
    uint32_t l1NDynamic = 0;
    uint32_t l1KDynamic = 0;

    uint32_t blockStartOffset = 0;
    uint32_t maxKVStackLen = 0;

    // L1上Q矩阵的布局信息，用于跨kvhead跳着取数据
    uint32_t l1ATotalRowNum = 0;      // L1上Q矩阵的总行数（所有kvhead）
    uint32_t l1ARowNumPerKvHead = 0;  // 单个kvhead对应的行数
    uint32_t l1AEmbedDim = 0;         // embed维度
    uint32_t l1AKvNBlockSize = 1;     // kvhead合轴数量
};

////////////////////////////////////////////////////////////////////

} // namespace NpuArch::Gemm::Block

#endif // GEMM_BLOCK_MMAD_QK_HPP