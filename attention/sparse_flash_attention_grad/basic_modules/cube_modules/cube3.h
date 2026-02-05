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
 * \file cube3.h
 * \brief Formula: dq = ds * k
 * l0_a： dimG * 512 * sizof(T1) (10k-26k)
 * l0_b： selectedBlockSize * dimDv * sizof(T1) (0-24k / 24k-48k)
 * l0_c： dimG * selectedBlockSize * sizof(T1) (0-4k / 48k-52k)
 */

template <typename T1>
__aicore__ inline __attribute__((always_inline)) void
CubeOp<T1>::cube3CopyKey(LocalTensor<T1>& l1Key, const int64_t keyGmOffset, const int64_t keyRopeGmOffset,
                         const int32_t blkCntOffset, const int32_t nIdx, const int32_t dIdx,
                         const uint32_t mmParamK, const uint32_t mmParamN, const bool isDense)
{
    WaitFlag<HardEvent::MTE1_MTE2>(MM_L1_COMMON_EVENTS[ping_pong_flag_l1_common_]);
    l1Key = l1_common_tensors[ping_pong_flag_l1_common_];
    
    if (isDense) {
        if (dIdx != -1) {
            int64_t currentKeyOffset = keyGmOffset + (blkCntOffset * dimN2 + nIdx) * selectedBlockSizeDqk + dIdx * mmParamN;
            CopyGmToL1(l1Key, keyGm[currentKeyOffset], mmParamK, mmParamN, dimDqk);
        } else {
            if constexpr (HAS_ROPE) {
                int64_t currentKeyOffset = keyRopeGmOffset + (blkCntOffset * dimN2 + nIdx) * selectedBlockSizeDrope;
                CopyGmToL1(l1Key, keyRopeGm[currentKeyOffset], mmParamK, mmParamN, dimRope);
            } else {
                int64_t currentKeyOffset = keyGmOffset + (blkCntOffset * dimN2 + nIdx) * selectedBlockSizeDqk;
                CopyGmToL1(l1Key, keyGm[currentKeyOffset], mmParamK, mmParamN, dimDqk);
            }
        }
    } else {
        int64_t currentKeyOffset = keyGmOffset + nIdx * selectedBlockSizeDtotal + (dIdx == -1 ? 0 : dIdx * mmParamN);
        CopyGmToL1(l1Key, selectedKWorkspaceGm[currentKeyOffset], mmParamK, mmParamN, dimDTotal);
    }
}

template <typename T1>
__aicore__ inline __attribute__((always_inline)) void
CubeOp<T1>::cube3ProcessNLoop(LocalTensor<float>& l0cTensor, const int64_t keyGmOffset, const int64_t keyRopeGmOffset,
                               const int64_t outGmOffset, const int32_t blkCntOffset, const int32_t dIdx,
                               const uint32_t perLoopDSize, const uint32_t tailLoopDSize, const bool isDense,
                               const int64_t lastBlockSize, const bool isLastBasicBlock)
{
    uint32_t blockOffset = K_SPLIT_SIZE / selectedBlockSize;
    uint32_t mmParamN = (dIdx == -1) ? tailLoopDSize : perLoopDSize;
    int64_t currentOutGmOffset = outGmOffset + (dIdx == -1 ? (perLoopDSize - tailLoopDSize) : dIdx * perLoopDSize);

    for (int32_t nIdx = 0; nIdx < selectedCntOffset; nIdx += blockOffset) {
        int32_t l1Offset = nIdx * selectedBlockSize * dimGAlign;
        bool isFirstLoop = (nIdx == 0);
        bool isLastLoop = (nIdx + blockOffset >= selectedCntOffset);

        uint32_t totalSel = selectedCntOffset * selectedBlockSize;
        if (isLastBasicBlock && isLastLoop) {
            totalSel = totalSel - selectedBlockSize + lastBlockSize;
        }
        uint32_t mmParamK = min(selectedBlockSize * blockOffset, totalSel - nIdx * selectedBlockSize);

        LocalTensor<T1> l1Ds = l1_ds_tensors[ping_pong_flag_l1_ds_][l1Offset];
        LocalTensor<T1> l1Key;
        
        cube3CopyKey(l1Key, keyGmOffset, keyRopeGmOffset, blkCntOffset, nIdx, dIdx, mmParamK, mmParamN, isDense);

        MMParam mmParam;
        mmParam.singleM = dimG;
        mmParam.singleN = mmParamN;
        mmParam.singleK = mmParamK;
        mmParam.isRightTranspose = true;
        mmParam.isOutKFisrt = isFirstLoop;
        mmParam.isFixOut = isLastLoop;
        mmParam.dstStride = dimDTotal;

        MmadInnerWithSync<T1, true>(l0cTensor, l1Ds, l1Key,
                    aL0TensorPingPong, bL0TensorPingPong,
                    mmParam, ping_pong_flag_l0a_, ping_pong_flag_l0b_, ping_pong_flag_l0c_, true, dqWorkspaceGm[currentOutGmOffset]);
        SetFlag<HardEvent::MTE1_MTE2>(MM_L1_COMMON_EVENTS[ping_pong_flag_l1_common_]);
        UpdatePingPongFlag(ping_pong_flag_l1_common_);
    }
}

template <typename T1>
__aicore__ inline __attribute__((always_inline)) void
CubeOp<T1>::cube3ProcessDLoop(const int64_t keyGmOffset, const int64_t keyRopeGmOffset, const int64_t outGmOffset,
                               const int32_t blkCntOffset, const bool isDense, const int64_t lastBlockSize,
                               const bool isLastBasicBlock)
{
    uint32_t dLoopTimes = (dimDTotal + 127) / N_SPLIT_SIZE;
    uint32_t perLoopDSize = N_SPLIT_SIZE;
    uint32_t tailLoopDSize = dimDTotal - (dLoopTimes - 1) * perLoopDSize;

    for (int32_t dIdx = 0; dIdx < dLoopTimes; dIdx++) {
        LocalTensor<float> l0cTensor = cL0TensorPingPong[ping_pong_flag_l0c_ & 1];
        bool isTail = (dIdx == dLoopTimes - 1);
        int32_t actualDIdx = isTail ? -1 : dIdx;
        
        cube3ProcessNLoop(l0cTensor, keyGmOffset, keyRopeGmOffset, outGmOffset, blkCntOffset, actualDIdx,
                          perLoopDSize, tailLoopDSize, isDense, lastBlockSize, isLastBasicBlock);
        UpdatePingPongFlag(ping_pong_flag_l0c_);
    }
}

template <typename T1>
__aicore__ inline __attribute__((always_inline)) void
CubeOp<T1>::cube3ProcessSparse(const int64_t dsGmOffset, const int64_t keyGmOffset, const int64_t indicesGmOffset,
                         const int64_t outGmOffset, const int32_t blkCntOffset, const int32_t mmPingPongIdx, 
                         const int64_t lastBlockSize, const bool isLastBasicBlock)
{
    cube3ProcessDLoop(keyGmOffset, 0, outGmOffset, blkCntOffset, false, lastBlockSize, isLastBasicBlock);
}

template <typename T1>
__aicore__ inline __attribute__((always_inline)) void
CubeOp<T1>::cube3ProcessDense(const int32_t blkCntOffset, const int32_t mmPingPongIdx, const int64_t lastBlockSize,
                              const bool isLastBasicBlock, const RunInfo &runInfo)
{
    const int64_t dsGmOffset = runInfo.mm345GmOffset;
    const int64_t keyGmOffset = runInfo.keyGmOffset;
    const int64_t keyRopeGmOffset = runInfo.keyRopeGmOffset;
    const int64_t indicesGmOffset = runInfo.indicesGmOffset;
    const int64_t outGmOffset = runInfo.mm3OutGmOffset;

    cube3ProcessDLoop(keyGmOffset, keyRopeGmOffset, outGmOffset, blkCntOffset, true, lastBlockSize, isLastBasicBlock);
}

template <typename T1>
__aicore__ inline __attribute__((always_inline)) void
CubeOp<T1>::cube3Process(const int64_t dsGmOffset, const int64_t keyGmOffset, const int64_t indicesGmOffset,
                         const int64_t outGmOffset, const int32_t blkCntOffset, const int32_t mmPingPongIdx,
                         const int64_t lastBlockSize, const bool isLastBasicBlock, const RunInfo &runInfo)
{
    if (!runInfo.isSmallS2) {
        cube3ProcessSparse(dsGmOffset, keyGmOffset, indicesGmOffset, outGmOffset, blkCntOffset, mmPingPongIdx, lastBlockSize, isLastBasicBlock);
    } else {
        cube3ProcessDense(blkCntOffset, mmPingPongIdx, lastBlockSize, isLastBasicBlock, runInfo);
    }
}
