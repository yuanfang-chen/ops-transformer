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
 * \file cube4.h
 * \brief Formula: dk = s^T * q
 * l0_a： dimG * selectedBlockSize * sizof(T1) (26-28k/28-30k)
 * l0_b： dimG * 512 * sizof(T1) (0k-16k)
 * l0_c： dimG * dimDqk * sizof(flot) (0-12k / 64k-76k)
 */

template <typename T1>
__aicore__ inline __attribute__((always_inline)) void
CubeOp<T1>::cube4LoadQuery(LocalTensor<T1>& l1Query, const int64_t queryGmOffset, const int64_t queryRopeGmOffset, 
                           const uint32_t dIdx, const uint32_t perLoopDSize, const uint32_t tailLoopDSize, 
                           const bool isTail, const bool reloadQuery)
{
    if (reloadQuery) {
        WaitFlag<HardEvent::MTE1_MTE2>(MM_L1_COMMON_EVENTS[ping_pong_flag_l1_common_]);
        l1Query = l1_common_tensors[ping_pong_flag_l1_common_];
        int64_t currentQueryOffset = isTail ? (HAS_ROPE ? queryRopeGmOffset : queryGmOffset + (dIdx - 1) * perLoopDSize) 
                                            : queryGmOffset + dIdx * perLoopDSize;
        GlobalTensor<T1> srcGm = isTail && HAS_ROPE ? queryRopeGm[currentQueryOffset] : queryGm[currentQueryOffset];
        int64_t destStride = isTail && HAS_ROPE ? dimRope : dimDqk;
        uint32_t copySize = isTail ? tailLoopDSize : perLoopDSize;
        CopyGmToL1(l1Query, srcGm, dimG, copySize, destStride);
    } else {
        uint32_t l1Offset = isTail ? (dIdx - 1) * dimGAlign * perLoopDSize : dIdx * dimGAlign * perLoopDSize;
        l1Query = l1_query_tensor[l1Offset];
    }
}

template <typename T1>
__aicore__ inline __attribute__((always_inline)) void
CubeOp<T1>::cube4ProcessDLoop(LocalTensor<T1>& l1Ds, const int64_t queryGmOffset, const int64_t queryRopeGmOffset,
                               const uint32_t dLoopTimes, const uint32_t perLoopDSize, const uint32_t tailLoopDSize,
                               const uint32_t mmParamM, const int64_t mm4ResOutOffset, const bool reloadQuery)
{
    for (uint32_t dIdx = 0; dIdx < dLoopTimes; dIdx++) {
        bool isTail = (dIdx == dLoopTimes - 1);
        LocalTensor<float> l0cTensor = cL0TensorPingPong[ping_pong_flag_l0c_];
        LocalTensor<T1> l1Query;
        
        cube4LoadQuery(l1Query, queryGmOffset, queryRopeGmOffset, dIdx, perLoopDSize, tailLoopDSize, isTail, reloadQuery);
        
        MMParam mmParam;
        mmParam.singleM = mmParamM;
        mmParam.singleN = isTail ? tailLoopDSize : perLoopDSize;
        mmParam.singleK = dimG;
        mmParam.isFixOut = true;
        mmParam.isLeftTranspose = true;
        mmParam.isRightTranspose = true;
        mmParam.dstStride = dimDTotal * dimN2;
        
        int64_t currentOutGmOffset = mm4ResOutOffset + dIdx * perLoopDSize;
        uint32_t l0a_ping_pong_flag = ping_pong_flag_l0a_;
        
        MmadInnerWithSync<T1>(l0cTensor, l1Ds, l1Query,
                aL0TensorPingPong, bL0TensorPingPong,
                mmParam, l0a_ping_pong_flag, ping_pong_flag_l0b_, ping_pong_flag_l0c_, dIdx == 0, mm4ResWorkspaceGm[currentOutGmOffset]);
        
        if (reloadQuery) {
            SetFlag<HardEvent::MTE1_MTE2>(MM_L1_COMMON_EVENTS[ping_pong_flag_l1_common_]);
            UpdatePingPongFlag(ping_pong_flag_l1_common_);
        }
        UpdatePingPongFlag(ping_pong_flag_l0c_);
    }
}

template <typename T1>
__aicore__ inline __attribute__((always_inline)) void
CubeOp<T1>::cube4ProcessMLoop(const int64_t dsGmOffset, const int64_t queryGmOffset, const int64_t queryRopeGmOffset,
                               const int64_t indicesGmOffset, const int32_t blkCntOffset, const int64_t mm4ResOutBaseOffset,
                               const bool reloadQuery)
{
    uint32_t dLoopTimes = (dimDTotal + 127) / N_SPLIT_SIZE;
    uint32_t perLoopDSize = N_SPLIT_SIZE;
    uint32_t tailLoopDSize = dimDTotal - (dLoopTimes - 1) * perLoopDSize;
    uint32_t blockOffset = M_SPLIT_SIZE / selectedBlockSize;

    LocalTensor<T1> l1DsTensor = l1_ds_tensors[ping_pong_flag_l1_ds_];
    CopyGmToL1(l1DsTensor, dsWorkspaceGm[dsGmOffset], dimG, selectedCntOffset * selectedBlockSize, TOTAL_BLOCK_SIZE);

    for (int32_t mIdx = blkCntOffset; mIdx < blkCntOffset + selectedCntOffset; mIdx += blockOffset) {
        int32_t l1Offset = (mIdx - blkCntOffset) * selectedBlockSize * dimGAlign;
        LocalTensor<T1> l1Ds = l1DsTensor[l1Offset];
        uint32_t mmParamM = min(selectedBlockSize * blockOffset, selectedCntOffset * selectedBlockSize - (mIdx - blkCntOffset) * selectedBlockSize);
        int64_t mm4ResOutOffset = mm4ResOutBaseOffset + mIdx * selectedBlockSizeDtotal;
        
        cube4ProcessDLoop(l1Ds, queryGmOffset, queryRopeGmOffset, dLoopTimes, perLoopDSize, tailLoopDSize, mmParamM, mm4ResOutOffset, reloadQuery);
    }
}

template <typename T1>
__aicore__ inline __attribute__((always_inline)) void
CubeOp<T1>::cube4ProcessSparse(const int64_t dsGmOffset, const int64_t queryGmOffset, const int64_t queryRopeGmOffset, const int64_t indicesGmOffset,
                         const int64_t outGmOffset, const int32_t blkCntOffset, const int32_t mmPingPongIdx, const RunInfo &runInfo)
{
    int64_t mm4ResOutBaseOffset = runInfo.scatterTaskId * MAX_CORE_NUM * selectedBlockCount * selectedBlockSizeDtotal + cBlockIdx * selectedBlockCount * selectedBlockSizeDtotal;
    const bool reloadQuery = !runInfo.noReload && runInfo.isLastBasicBlock;
    
    cube4ProcessMLoop(dsGmOffset, queryGmOffset, queryRopeGmOffset, indicesGmOffset, blkCntOffset, mm4ResOutBaseOffset, reloadQuery);
}

template <typename T1>
__aicore__ inline __attribute__((always_inline)) void
CubeOp<T1>::cube4ProcessDense(const int32_t blkCntOffset, const int32_t mmPingPongIdx, const RunInfo &runInfo)
{
    const int64_t dsGmOffset = runInfo.mm345GmOffset;
    const int64_t queryGmOffset = runInfo.queryGmOffset; 
    const int64_t queryRopeGmOffset = runInfo.queryRopeGmOffset;
    const int64_t indicesGmOffset = runInfo.indicesGmOffset;
    const int64_t outGmOffset = runInfo.mm5OutGmOffset;

    int64_t mm4ResOutBaseOffset = runInfo.scatterTaskId * MAX_CORE_NUM * selectedBlockCount * selectedBlockSizeDtotal + cBlockIdx * selectedBlockCount * selectedBlockSizeDtotal;
    const bool reloadQuery = !runInfo.noReload && runInfo.isLastBasicBlock;
    
    cube4ProcessMLoop(dsGmOffset, queryGmOffset, queryRopeGmOffset, indicesGmOffset, blkCntOffset, mm4ResOutBaseOffset, reloadQuery);
}

template <typename T1>
__aicore__ inline __attribute__((always_inline)) void
CubeOp<T1>::cube4Process(const int64_t dsGmOffset, const int64_t queryGmOffset, const int64_t queryRopeGmOffset, const int64_t indicesGmOffset,
                         const int64_t outGmOffset, const int32_t blkCntOffset, const int32_t mmPingPongIdx, const RunInfo &runInfo)
{
    if (!runInfo.isSmallS2) {
        cube4ProcessSparse(dsGmOffset, queryGmOffset, queryRopeGmOffset, indicesGmOffset, outGmOffset, blkCntOffset, mmPingPongIdx, runInfo);
    } else {
        cube4ProcessDense(blkCntOffset, mmPingPongIdx, runInfo);
    }
}
