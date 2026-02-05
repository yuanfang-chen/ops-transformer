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
 * \file cube1.h
 * \brief Formula: s = q * k^T
 * l0_a： dimG * dimDqk * sizof(T1) (0-6k)
 * l0_b： selectedBlockSize * dimDqk * sizof(T1) (0-24k / 24k-48k)
 * l0_c： dimG * selectedBlockSize * sizof(T1) (0-4k / 64k-68k)
 */

template <typename T1>
__aicore__ inline __attribute__((always_inline)) void
CubeOp<T1>::cube1CopyKey(LocalTensor<T1>& l1Key, const int64_t keyGmOffset, const int64_t keyRopeGmOffset,
                         const int32_t blkCntOffset, const int32_t nIdx, const int32_t dIdx,
                         const uint32_t mmParamN, const uint32_t mmParamK, const bool isDense)
{
    WaitFlag<HardEvent::MTE1_MTE2>(MM_L1_COMMON_EVENTS[ping_pong_flag_l1_common_]);
    l1Key = l1_common_tensors[ping_pong_flag_l1_common_];
    
    if (isDense) {
        if (dIdx != -1) {
            int64_t currentKeyOffset = keyGmOffset + (blkCntOffset * dimN2 + nIdx) * selectedBlockSizeDqk + dIdx * K_SPLIT_SIZE;
            CopyGmToL1(l1Key, keyGm[currentKeyOffset], mmParamN, mmParamK, dimDqk);
        } else {
            int64_t currentKeyOffset = HAS_ROPE ? keyRopeGmOffset + blkCntOffset * dimN2 * selectedBlockSizeDrope + nIdx * selectedBlockSize * dimRope
                                                 : keyGmOffset + (blkCntOffset * dimN2 + nIdx) * selectedBlockSizeDqk;
            GlobalTensor<T1> kSrcGm = HAS_ROPE ? keyRopeGm[currentKeyOffset] : keyGm[currentKeyOffset];
            int64_t kSrcDstride = HAS_ROPE ? mmParamK : dimDqk;
            CopyGmToL1(l1Key, kSrcGm, mmParamN, mmParamK, kSrcDstride);
        }
    } else {
        int64_t currentKeyOffset = keyGmOffset + nIdx * selectedBlockSizeDtotal + (dIdx == -1 ? 0 : dIdx * K_SPLIT_SIZE);
        CopyGmToL1(l1Key, selectedKWorkspaceGm[currentKeyOffset], mmParamN, mmParamK, dimDTotal);
    }
}

template <typename T1>
__aicore__ inline __attribute__((always_inline)) void
CubeOp<T1>::cube1ProcessDLoop(LocalTensor<float>& l0cTensor, const int64_t keyGmOffset, const int64_t keyRopeGmOffset,
                               const int64_t outGmOffset, const int32_t blkCntOffset, const int32_t nIdx,
                               const bool isDense)
{
    uint32_t dLoopTimes = (dimDTotal + 127) / K_SPLIT_SIZE;
    uint32_t perLoopDSize = K_SPLIT_SIZE;
    uint32_t tailLoopDSize = dimDTotal - (dLoopTimes - 1) * perLoopDSize;
    uint32_t blockOffset = N_SPLIT_SIZE / selectedBlockSize;

    MMParam mmParam;
    mmParam.singleM = dimG;
    mmParam.singleN = selectedBlockSize * blockOffset;
    mmParam.dstStride = PER_LOOP_BLOCK_SIZE;

    for (int32_t dIdx = 0; dIdx < dLoopTimes; dIdx++) {
        bool isTail = (dIdx == dLoopTimes - 1);
        int32_t actualDIdx = isTail ? -1 : dIdx;
        mmParam.isFixOut = isTail;
        mmParam.isOutKFisrt = (dIdx == 0);
        mmParam.singleK = isTail ? tailLoopDSize : perLoopDSize;

        LocalTensor<T1> l1Query = l1_query_tensor[dIdx * dimGAlign * perLoopDSize];
        LocalTensor<T1> l1Key;
        
        cube1CopyKey(l1Key, keyGmOffset, keyRopeGmOffset, blkCntOffset, nIdx, actualDIdx, 
                     mmParam.singleN, mmParam.singleK, isDense);

        int64_t mm1WorkspaceGmOffset = outGmOffset + nIdx * selectedBlockSize;
        MmadInnerWithSync<T1>(l0cTensor, l1Query, l1Key,
                    aL0TensorPingPong, bL0TensorPingPong,
                    mmParam, ping_pong_flag_l0a_, ping_pong_flag_l0b_, ping_pong_flag_l0c_, true, mm1WorkspaceGm[mm1WorkspaceGmOffset]);
        SetFlag<HardEvent::MTE1_MTE2>(MM_L1_COMMON_EVENTS[ping_pong_flag_l1_common_]);
        UpdatePingPongFlag(ping_pong_flag_l1_common_);
    }
}

template <typename T1>
__aicore__ inline __attribute__((always_inline)) void
CubeOp<T1>::cube1ProcessNLoop(const int64_t keyGmOffset, const int64_t keyRopeGmOffset, const int64_t outGmOffset,
                               const int32_t blkCntOffset, const bool isDense)
{
    uint32_t blockOffset = N_SPLIT_SIZE / selectedBlockSize;

    for (int32_t nIdx = 0; nIdx < selectedCntOffset; nIdx += blockOffset) {
        LocalTensor<float> l0cTensor = cL0TensorPingPong[ping_pong_flag_l0c_ & 1];
        int64_t adjustedKeyGmOffset = isDense ? keyGmOffset : keyGmOffset + nIdx * selectedBlockSizeDtotal;
        int64_t adjustedOutGmOffset = outGmOffset + nIdx * selectedBlockSize;
        
        cube1ProcessDLoop(l0cTensor, adjustedKeyGmOffset, keyRopeGmOffset, adjustedOutGmOffset, blkCntOffset, nIdx, isDense);
        UpdatePingPongFlag(ping_pong_flag_l0c_);
    }
}

template <typename T1>
__aicore__ inline __attribute__((always_inline)) void
CubeOp<T1>::cube1ProcessSparse(const int64_t queryGmOffset, const int64_t queryRopeGmOffset, const int64_t keyGmOffset, const int64_t indicesGmOffset,
                         const int64_t outGmOffset, const int32_t blkCntOffset, const int32_t mmPingPongIdx)
{
    cube1ProcessNLoop(keyGmOffset, 0, outGmOffset, blkCntOffset, false);
}

template <typename T1>
__aicore__ inline __attribute__((always_inline)) void
CubeOp<T1>::cube1ProcessDense(const int32_t blkCntOffset, const int32_t mmPingPongIdx, const RunInfo &runInfo)
{
    const int64_t keyGmOffset = runInfo.keyGmOffset;
    const int64_t keyRopeGmOffset = runInfo.keyRopeGmOffset;
    const int64_t outGmOffset = runInfo.mm12GmOffset;

    cube1ProcessNLoop(keyGmOffset, keyRopeGmOffset, outGmOffset, blkCntOffset, true);
}

template <typename T1>
__aicore__ inline __attribute__((always_inline)) void
CubeOp<T1>::cube1Process(const int64_t queryGmOffset, const int64_t queryRopeGmOffset, const int64_t selectedKGmOffset, const int64_t indicesGmOffset,
                         const int64_t outGmOffset, const int32_t blkCntOffset, const int32_t mmPingPongIdx, const RunInfo &runInfo)
{
    if (!runInfo.isSmallS2) {
        cube1ProcessSparse(queryGmOffset, queryRopeGmOffset, selectedKGmOffset, indicesGmOffset, outGmOffset, blkCntOffset, mmPingPongIdx);
    } else {
        cube1ProcessDense(blkCntOffset, mmPingPongIdx, runInfo);
    }
}
