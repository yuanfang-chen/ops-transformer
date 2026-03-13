/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_SIMPLY_SOFTMAX_HPP
#define CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_SIMPLY_SOFTMAX_HPP

#include "../../../attn_infra/arch/resource.hpp"
#include "../../../attn_infra/epilogue/dispatch_policy.hpp"
#include "kernel_operator.h"


using namespace AscendC;

struct SimplySoftMaxInfo {
    LocalTensor<float> sTensor;
    LocalTensor<float> lseTensor;
    LocalTensor<float> lseBrocTensor;
    LocalTensor<float> pTensor;

    GlobalTensor<float> sGm;
    GlobalTensor<float> lseGm;
    GlobalTensor<float> pGm;
};

struct CalDsInfo {
    LocalTensor<float> dpTensor;
    LocalTensor<float> softmaxGradTensor;
    LocalTensor<float> pTensor;

    GlobalTensor<float> dpGm;
    GlobalTensor<float> softmaxGradGm;
    GlobalTensor<float> dsGm;
};


namespace NpuArch::Epilogue::Block {
template <
    typename InputDType,
    typename OutputDtype,
    // class ArchTag,
    uint32_t INPUT_LAYOUT>
class SimpltSoftmax
{
public:
    using DispatchPolicy = EpilogueAtlasA2FAGPre;
    using ArchTag = typename DispatchPolicy::ArchTag;

    struct Params {
        // Data members
        GM_ADDR s; // 连续
        GM_ADDR softmaxLse; //需要跳着搬运
        GM_ADDR dp; // 连续
        GM_ADDR blockSparseMask;
        // GM_ADDR blockShape;
        GM_ADDR actualQSeqlen;
        GM_ADDR actualKvSeqlen;
        GM_ADDR softGradworkspace; // 需要跳着搬运
        GM_ADDR pWorkspace; // 连续
        GM_ADDR dsWorkspace; // 连续
        GM_ADDR tilingData;
        uint64_t actualRow = 0;
        uint64_t actualCol = 0;
        uint64_t processNums = 0;
        uint64_t curCoreBatch = 0;
        uint64_t curCoreN1Idx = 0;
        uint64_t curCoreS1Idx = 0;
        uint64_t curT1Idx = 0;

        // Methods
        __aicore__ inline
        Params() {}

        __aicore__ inline
        Params(
            GM_ADDR s_, GM_ADDR softmaxLse_,  GM_ADDR dp_, GM_ADDR blockSparseMask_,
            // GM_ADDR blockShape_,
            GM_ADDR actualQSeqlen_, GM_ADDR actualKvSeqlen_, GM_ADDR softGradworkspace_, GM_ADDR pWorkspace_, GM_ADDR dsWorkspace_, GM_ADDR tilingData_,
            uint64_t acutualRow_, uint64_t actualCol_, uint64_t processNums_, uint64_t curBatch_, uint64_t curN1_, uint64_t curS1_, uint64_t curT1_
        ) : s(s_), softmaxLse(softmaxLse_), dp(dp_), blockSparseMask(blockSparseMask_),
            // blockShape(blockShape_),
            actualQSeqlen(actualQSeqlen_), actualKvSeqlen(actualKvSeqlen_),
            softGradworkspace(softGradworkspace_), pWorkspace(pWorkspace_), dsWorkspace(dsWorkspace_), tilingData(tilingData_),
            actualRow(acutualRow_), actualCol(actualCol_), processNums(processNums_),
            curCoreBatch(curBatch_), curCoreN1Idx(curN1_), curCoreS1Idx(curS1_), curT1Idx(curT1_)
        {
            
        }    
    };

    NpuArch::Arch::Resource<ArchTag> resource;

    constexpr static uint64_t STAGES=1;
    constexpr static uint64_t INPUT_NUM = 2;
    constexpr static uint64_t DOUBLE_BUFFER = 2;
    constexpr static uint64_t BNSD = 1;
    constexpr static uint64_t TND = 0;
    constexpr static uint64_t proceeM = 128;
    constexpr static uint64_t proceeK = 128;
    constexpr static uint64_t BRCB_BASE_NUM = 8;
    constexpr static uint64_t REAPTE_BYTE = 256;

    constexpr static uint64_t BLOCK_BYTE_SIZE = 32;
    constexpr static uint64_t BLOCK_SIZE = 8;
    constexpr static uint64_t SFMG_HIGH_PERF_N_FACTOR = 8;
    constexpr static uint64_t SFMG_HIGH_PERF_D_FACTOR = 64;

    uint64_t cBlockIdx = 0;
    uint64_t cubeCoreIdx = 0;
    uint64_t vecCoreIdx = 0;
    uint64_t row = 0; // 当前core需要处理q方向的s数
    uint64_t col = 0; // 当前core需要处理kv方向的s数
    uint64_t curCoreBatch = 0;
    uint64_t curCoreN1Idx = 0;      // q_n
    uint64_t curT1Idx = 0;          // q_t
    uint64_t curCoreS1Idx = 0;     // q_s 
    uint64_t maxQSeqlen = 0;
    uint64_t maxKvSeqlen = 0;
    uint64_t n1 = 0; // q_n
    uint64_t transpseStride = 0;

    // uint64_t s2 = 0;     // q_s 
    uint64_t processNums = 0;
    uint64_t usedVecCoreNums = 0;
    uint64_t baseBufLen = 0;

    // const BlockSparseAttentionGradTilingData *tilingData;

    GlobalTensor<float> sGm;  // (N s1 s2)
    GlobalTensor<float> softmaxLseGm; // (N s1 1)
    GlobalTensor<float> dpGm; // (N s1 s2)
    // GlobalTensor<uint8_t> blockSparseMaskGm; // (b n block_s1 block_s2)
    GlobalTensor<float> pWorkspaceGm; // (N s1 s2)
    GlobalTensor<float> softGradworkspaceGm; // (N s1 8)
    GlobalTensor<float> dsWorkspaceGm; // (N s1 s2)

    GM_ADDR actualQSeqlen;
    GM_ADDR actualKvSeqlen;

    LocalTensor<float> sTensor[STAGES];
    LocalTensor<float> lseTensor[STAGES];
    LocalTensor<float> lseBrocTensor[STAGES];
    LocalTensor<float> pTensor[STAGES];
    LocalTensor<float> dpTensor[STAGES];
    // LocalTensor<float> mask[STAGES];
    LocalTensor<float> softmaxGradTensor[STAGES];
    LocalTensor<float> dsTensor[STAGES];

    __aicore__ inline
    SimpltSoftmax(Params const &params)
    {
        vecCoreIdx = GetBlockIdx();
        cBlockIdx = vecCoreIdx;
        __gm__ BlockSparseAttentionGradTilingData *tilingData = reinterpret_cast<__gm__ BlockSparseAttentionGradTilingData *>(params.tilingData);
        usedVecCoreNums = tilingData->usedVecCoreNum;
        printf("cBlockIdx: %lu \n", cBlockIdx);


        if (cBlockIdx >= usedVecCoreNums) {
            return;
        }

        maxQSeqlen = tilingData -> maxQSeqlen;
        maxKvSeqlen = tilingData -> maxKvSeqlen;
        n1 = tilingData -> numHeads; // q_n
        col = params.actualCol;
        curCoreBatch = params.curCoreBatch;
        curCoreN1Idx = params.curCoreN1Idx;
        curT1Idx = params.curT1Idx;
        curCoreS1Idx = params.curCoreS1Idx;
        actualQSeqlen = params.actualQSeqlen;
        actualKvSeqlen = params.actualKvSeqlen;

        uint64_t s2 = ((__gm__ uint64_t *)actualKvSeqlen)[curCoreBatch];
            printf("s2: %lu \n", s2);
        uint64_t curCoreProcessNum = params.processNums;
        uint64_t ubSize = tilingData->ubSize;
        uint64_t ubSizeEeachStage = ubSize  / STAGES / BLOCK_BYTE_SIZE * BLOCK_BYTE_SIZE; // 32字节对齐

        if (vecCoreIdx % 2 == 0) {
            row = params.actualRow / 2 + params.actualRow % 2;
        } else {
            row = params.actualRow / 2;
            curCoreS1Idx +=  params.actualRow / 2 + params.actualRow % 2;
        }
        processNums = row * col;
        if constexpr(INPUT_LAYOUT == TND) {
            transpseStride = (n1 * 1 - 1) * sizeof(float);
        } else if constexpr(INPUT_LAYOUT == BNSD){
            transpseStride = 0;
        }
  
        // 默认 s2 > 8
        // 计算 simply_softmax p = (exp(S - L)) buffer 大小 记得广播
        // 设S buffer x, l broc buffer 8 * x / s2, 则x + 8 * x / s2 = 192*1024 / 2
        // x 需保持32字节对齐
        // 计算 ds = p * (dp - D)  buffer 大小
        // 设D buffer y * 8 / s2, dp 为  y, p 为 y, 则y *2 +  8 * y / s2 = 192*1024 / 2
        // y 需要保持32字节对齐
        //  x > y ? y : x 理论上要保持x = y 
        // uint64_t xBufferLen = static_cast<uint64_t>(ubSizeEeachStage / (1 + (float)8.0 / s2));
        uint64_t xBufferLen = static_cast<uint64_t>(ubSizeEeachStage / (1 + 8 / s2));
        // 字节对齐
        xBufferLen = (xBufferLen * BRCB_BASE_NUM) / s2 / BLOCK_BYTE_SIZE * BLOCK_BYTE_SIZE * s2 / BRCB_BASE_NUM;
        xBufferLen = xBufferLen / BLOCK_BYTE_SIZE * BLOCK_BYTE_SIZE;
        // uint64_t yBufferLen = static_cast<uint64_t>(ubSizeEeachStage / (2 + (float)8.0 / s2));
        uint64_t yBufferLen = static_cast<uint64_t>(ubSizeEeachStage / (2 + 8 / s2));
        // 字节对齐
        yBufferLen =  (BRCB_BASE_NUM * yBufferLen) / s2 / BLOCK_BYTE_SIZE * BLOCK_BYTE_SIZE * s2 / BRCB_BASE_NUM;
        xBufferLen = yBufferLen / BLOCK_BYTE_SIZE * BLOCK_BYTE_SIZE;
        baseBufLen = xBufferLen > yBufferLen ? yBufferLen : xBufferLen;
        // 保持8元素对齐
        baseBufLen = (baseBufLen / s2 / sizeof(float))  / BRCB_BASE_NUM * BRCB_BASE_NUM * s2 * sizeof(float);

        // 空间大小计算
        uint64_t sBufferLen = baseBufLen;
        uint64_t lBufferLen = baseBufLen / s2;
        uint64_t lBrobBufferLen = BRCB_BASE_NUM * baseBufLen / s2; 
        uint64_t pBufferLen = baseBufLen;
        uint64_t dpBufLen = baseBufLen;
        uint64_t pBufLen = baseBufLen;
        uint64_t dBufLen = BRCB_BASE_NUM * baseBufLen / s2;
  
        for (uint64_t i = 0; i < STAGES; i++) {
            uint64_t stageOffset = ubSizeEeachStage * i;
            // 第一轮 softmax 计算空间划分
            sTensor[i] = resource.ubBuf.template GetBufferByByte<float>(stageOffset);
            pTensor[i] = sTensor[i]; // 复用s
            lseTensor[i] = sTensor[i]; // 复用s
            lseBrocTensor[i] = resource.ubBuf.template GetBufferByByte<float>(sBufferLen + stageOffset);

            // 第二轮 ds 计算空间划分
            dpTensor[i] = 
                resource.ubBuf.template GetBufferByByte<float>(pBufferLen + stageOffset);
            softmaxGradTensor[i] = 
                resource.ubBuf.template GetBufferByByte<float>(pBufferLen + stageOffset + dpBufLen);
            dsTensor[i] = pTensor[i]; // 复用s
        }

        uint64_t coreOffset = 0; // ai core 的每个vectore 的偏移
        if (vecCoreIdx % 2 != 0) {
            coreOffset += (params.actualRow / 2 + params.actualRow % 2) * params.actualCol;
        }

        // 初始化 GM
        sGm.SetGlobalBuffer((__gm__ float *)params.s + coreOffset);
        softmaxLseGm.SetGlobalBuffer((__gm__ float *)params.softmaxLse);
        dpGm.SetGlobalBuffer((__gm__ float *)params.dp + coreOffset);
        pWorkspaceGm.SetGlobalBuffer((__gm__ float *)params.pWorkspace + coreOffset);
        softGradworkspaceGm.SetGlobalBuffer((__gm__ float *)params.softGradworkspace);
        dsWorkspaceGm.SetGlobalBuffer((__gm__ float *)params.dsWorkspace + coreOffset);
    }
        
    __aicore__ inline
    ~SimpltSoftmax()
    {
    }

    template <int32_t CORE_TYPE = g_coreType>
    __aicore__ inline
    void operator()();

    template <>
    __aicore__ inline
    void operator()<AscendC::AIC>()
    {

    }

    template <>
    __aicore__ inline
    void operator()<AscendC::AIV>()
    {
        PipeBarrier<PIPE_ALL>(); // 去掉pre和sfmg之间的SyncALL，这里需要增加pipeALL ??
        if (cBlockIdx >= usedVecCoreNums) {
            return;
        }
        
        // col < 128
        // 计算单loop的计算量及loop次数
        uint64_t eleBaseBuffNum = baseBufLen / sizeof(float); // 基本buffer块的元素数量
        uint64_t bufferRows = eleBaseBuffNum / col == 0 ? 1 :  eleBaseBuffNum / col; // 一次lopp可以执行的row行数
        uint64_t rowLoopTimes = row / bufferRows;
        uint64_t tailRowNum = row - rowLoopTimes * bufferRows;


        uint64_t ping = 0;
        // 不包含尾行处理
        for (uint64_t i = 0; i < rowLoopTimes; i++) {
            auto eventId = ping ? EVENT_ID1 : EVENT_ID0;

            uint64_t curS1 = curCoreS1Idx + i * bufferRows;
            int32_t gmRowOffset = i * bufferRows * col;
            compute(gmRowOffset, bufferRows, col, curS1, ping);
            gmRowOffset += bufferRows * col;
      
            if (STAGES == DOUBLE_BUFFER) {
                ping = 1 - ping;
            }
        }

        if (tailRowNum > 0) {
            auto eventId = ping ? EVENT_ID0 : EVENT_ID1;

            uint64_t curS1 = curCoreS1Idx + rowLoopTimes * bufferRows;
            int32_t gmOffset =  rowLoopTimes * bufferRows * col;
            uint64_t tempRow = tailRowNum;
            uint64_t tempCol = col;
            compute(gmOffset, tempRow, tempCol, curS1, ping);
            gmOffset += tempRow * tempCol;
            if (STAGES == DOUBLE_BUFFER) {
                ping = 1 - ping;
            }
        }
    }

    __aicore__ inline
    void compute(int32_t gmOffset, uint64_t row, uint64_t col, uint64_t curS1, uint64_t ping)
    {
        struct SimplySoftMaxInfo runSftInfo = {sTensor[ping], lseTensor[ping], lseBrocTensor[ping], pTensor[ping], sGm[gmOffset], softmaxLseGm, pWorkspaceGm[gmOffset]};
        struct CalDsInfo runDsInfo = {dpTensor[ping], softmaxGradTensor[ping], pTensor[ping], dpGm[gmOffset], softGradworkspaceGm, dsWorkspaceGm[gmOffset]};
        SimplySoftmax(runSftInfo, row, col, curS1);
        CalDs(runDsInfo, row, col, curS1 );
    }


   /*
    * lse copy and brocast
    * lse input shape (b n s 1) or (t n 1)
    * out shape (b n s 8) or (n s 8)
    * dtype float
    */
    __aicore__ inline
    void LseBrocast(GlobalTensor<float> &LseGm, LocalTensor<float> &lse, LocalTensor<float> &lseFp32Brc, uint64_t count, uint64_t curS1)
    {
        uint64_t startOffset = 0;
        if constexpr (INPUT_LAYOUT == TND) {
            uint64_t bOffset = n1 * ((__gm__ int64_t *)actualQSeqlen)[curCoreBatch];
            startOffset = bOffset + curS1 * n1 + curCoreN1Idx;               
        } else {
            startOffset = curCoreBatch * (n1 * maxQSeqlen) + curCoreN1Idx * maxQSeqlen + curS1;
        }

        // 对于TND 格式来说， 会进行leis (s n) -> (n s) 的transpose转换
        DataCopyPad(lse, LseGm[startOffset],
                    {static_cast<uint16_t>(count), static_cast<uint32_t>(1 * sizeof(float)),
                    static_cast<uint32_t>(transpseStride), 0, 0},
                    {false, 0, 0, 0});
        AscendC::PipeBarrier<PIPE_ALL>();


        uint8_t repeatimes = CeilDiv(count, BRCB_BASE_NUM);
        Brcb(lseFp32Brc, lse, repeatimes, {1, 8});

    }

    /*
        * brief: Compute the elementwise multiplication of a tensor of shape (m, n) and a tensor of shape
        * ubIn0:[m, n], ubIn1[m, 8]
    */
    __aicore__ inline
    void SubBrcb(LocalTensor<float> const &ubOut, LocalTensor<float> const &ubIn0, LocalTensor<float> const &ubIn1, uint64_t row, uint64_t col)
    {
        uint32_t maxRepeatNum = 255;
        uint32_t eleNumPerBlk = static_cast<uint32_t>(BLOCK_BYTE_SIZE) / static_cast<uint32_t>(sizeof(float));

        uint32_t blkNumPerColumn = col / eleNumPerBlk;
        AscendC::BinaryRepeatParams repeatParams;
        repeatParams.dstBlkStride = blkNumPerColumn;
        repeatParams.src0BlkStride = blkNumPerColumn;
        repeatParams.src1BlkStride = 1;
        repeatParams.dstRepStride = 1;
        repeatParams.src0RepStride = 1;
        repeatParams.src1RepStride = 0;

        // 执行一次sub，迭代次数 col / oneblock ， 一次迭代计算 row * oneblock 元素， 
        uint32_t rowNumPerCompute = BLK_NUM_PER_VECTOR_FRACTAL; // 256 / 32 = 8 即per_repeat / per_block = 8
        uint32_t colNumPerCompute = eleNumPerBlk * maxRepeatNum; // 255 * 8

        for (uint32_t rowOffset = 0; rowOffset < row; rowOffset += rowNumPerCompute) {
            uint32_t residueM = row - rowOffset;
            uint32_t currentRowNum = (residueM > rowNumPerCompute) ? rowNumPerCompute : residueM;
            uint64_t mask = static_cast<uint64_t>(currentRowNum) * static_cast<uint64_t>(eleNumPerBlk);

            for (uint32_t colOffset = 0; colOffset < col; colOffset += colNumPerCompute) {
                uint32_t residueN = col - colOffset;
                uint32_t currentColNum = (residueN > colNumPerCompute) ? colNumPerCompute : residueN;
                uint8_t repeatTimes = static_cast<uint8_t>(currentColNum / eleNumPerBlk);

                AscendC::Sub(
                    ubOut[rowOffset * col + colOffset],
                    ubIn0[rowOffset * col + colOffset],
                    ubIn1[rowOffset * eleNumPerBlk],
                    mask, repeatTimes, repeatParams
                );
            }
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    /*
        * p = exp(s - lse_brc)
        * lse  shape (n s1 8) fp32 需非连续搬运
        * s shape (n s1 s2) fp32 连续
    */
    __aicore__ inline
    void SimplySoftmax(struct SimplySoftMaxInfo runInfo, uint64_t row, uint64_t col, uint64_t curS1)
    {
        LocalTensor<float> &sLocal = runInfo.sTensor;
        LocalTensor<float> &lse = runInfo.lseTensor;
        LocalTensor<float> &lseFp32Brc = runInfo.lseBrocTensor;
        LocalTensor<float> &pLocal = runInfo.pTensor;

        GlobalTensor<float> s = runInfo.sGm;
        GlobalTensor<float> lseGm = runInfo.lseGm;
        GlobalTensor<float> pGm = runInfo.pGm;

        uint64_t count = row * col;
        uint64_t countAlign = (count + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;


        LseBrocast(lseGm, lse, lseFp32Brc, row, curS1);
        AscendC::PipeBarrier<PIPE_ALL>();

        if (count * sizeof(float) % BLOCK_SIZE == 0) {
            DataCopy(sLocal, s, count);
        } else {
            DataCopyPad(sLocal, s, {static_cast<uint16_t>(1), static_cast<uint32_t>(count * sizeof(float)), 0, 0, 0}, 
                        {true, 0, static_cast<uint8_t>(countAlign - count), 0});
        }

        SubBrcb(pLocal, sLocal, lseFp32Brc, row, col);

        AscendC::PipeBarrier<PIPE_ALL>();

        Exp(pLocal, pLocal, count);

        AscendC::PipeBarrier<PIPE_ALL>();


        if (count * sizeof(float) % BLOCK_SIZE == 0) {
            DataCopy(pGm, pLocal, count);
        } else {
            DataCopyPad(pGm, pLocal, {static_cast<uint16_t>(1), static_cast<uint32_t>(count * sizeof(float)), 0, 0, 0});
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    /*
        * ds = p * (dp - D)
        * dp  shape (n s1 s2) fp32 连续
        * D shape (n s1 8) fp32 需非连续搬运
        * p shape (n s1 s2) fp32 连续
    */
    __aicore__ inline
    void CalDs(struct CalDsInfo runInfo,  uint64_t row, uint64_t col, uint64_t curS1)
    {
        LocalTensor<float> dpLocal = runInfo.dpTensor;
        LocalTensor<float> dLocal = runInfo.softmaxGradTensor;
        LocalTensor<float> pLocal = runInfo.pTensor;

        GlobalTensor<float> dp = runInfo.dpGm;
        GlobalTensor<float> d = runInfo.softmaxGradGm;
        GlobalTensor<float> ds = runInfo.dsGm;

        uint64_t count = row * col;


        AscendC::PipeBarrier<PIPE_ALL>();

        DataCopyPad(dpLocal, dp, {static_cast<uint16_t>(1), static_cast<uint32_t>(count * sizeof(float)), 0, 0, 0}, {false, 0, 0, 0});
        CopyDIn(d, dLocal, row, curS1);

        AscendC::PipeBarrier<PIPE_ALL>();
        SubBrcb(dpLocal, dpLocal, dLocal, row, col);
        AscendC::PipeBarrier<PIPE_ALL>();

        AscendC::PipeBarrier<PIPE_ALL>();
        Mul(dpLocal, pLocal, dpLocal, count);
        AscendC::PipeBarrier<PIPE_ALL>();

        DataCopyPad(ds, dpLocal, {static_cast<uint16_t>(1), static_cast<uint32_t>(count * sizeof(float)), 0, 0, 0});
    }

    /*
        * D shape ((n s1 8) or (b n s1 8) fp32 
    */
    __aicore__ inline
    void CopyDIn(GlobalTensor<float> d, LocalTensor<float> dLocal, uint64_t count, uint64_t curS1)
    {
        uint64_t startOffset = 0;
        if constexpr (INPUT_LAYOUT == TND) {
            uint64_t bOffset = n1 * ((__gm__ int64_t *)actualQSeqlen)[curCoreBatch] * BRCB_BASE_NUM;
            startOffset = bOffset + curS1 * n1 * BRCB_BASE_NUM + curCoreN1Idx * BRCB_BASE_NUM;               
        } else {
            startOffset = curCoreBatch * (n1 * maxQSeqlen * BRCB_BASE_NUM) + curCoreN1Idx * maxQSeqlen * BRCB_BASE_NUM + curS1 * BRCB_BASE_NUM;
        }
        startOffset = curCoreBatch * (n1 * maxQSeqlen * BRCB_BASE_NUM) + curCoreN1Idx * maxQSeqlen * BRCB_BASE_NUM + curS1 * BRCB_BASE_NUM;
        DataCopyPad(dLocal, d[startOffset],
                    {static_cast<uint16_t>(count), static_cast<uint32_t>(BRCB_BASE_NUM * sizeof(float)), 0, 0, 0}, {false, 0, 0, 0});
    }
};

}

#endif // CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_SIMPLY_SOFTMAX_HPP