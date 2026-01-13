/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file compressor_block_cube.h
 * \brief
 */

#ifndef COMPRESSOR_BLOCK_CUBE_H
#define COMPRESSOR_BLOCK_CUBE_H

#include "compressor_comm.h"

using namespace AscendC;

namespace Compressor {

template<typename COMP> class CompressorBlockCube {
public:
    __aicore__ inline CompressorBlockCube(){};
    __aicore__ inline void InitParams(const ConstInfo &constInfo);
    __aicore__ inline void Init(
        __gm__ uint8_t *x,
        __gm__ uint8_t *wKv,
        __gm__ uint8_t *wGate,
        __gm__ uint8_t *kvState,
        __gm__ uint8_t *scoreState,
        __gm__ uint8_t *ape,
        __gm__ uint8_t *normWeight,
        __gm__ uint8_t *ropeSin,
        __gm__ uint8_t *ropeCos,
        __gm__ uint8_t *blockTable,
        __gm__ uint8_t *cuSeqlens,
        __gm__ uint8_t *seqUsed,
        __gm__ uint8_t *startPos,
        __gm__ uint8_t *cmpKvOut,
        __gm__ uint8_t *kvStateOut,
        __gm__ uint8_t *scoreStateOut);
    __aicore__ inline void InitBuffers(TPipe *pipe);
    __aicore__ inline void AllocEventID(TPipe *pipe);
    __aicore__ inline void FreeEventID(TPipe *pipe);
    __aicore__ inline void ComputeMm1(const RunInfo &info);

private:
    using T = float;
    using X_T = typename AscendC::Conditional<COMP::xDtype == X_DTYPE::BF16, bfloat16_t, half>::type;

    __aicore__ inline void CopyWeightGmToL1(const RunInfo &info, LocalTensor<X_T> wL1Tensor,
        uint32_t hIdx, uint32_t kBase, uint32_t nLoopIdx);

    ConstInfo constInfo_ = {};

    // GM
    GlobalTensor<X_T> xGm_;
    GlobalTensor<X_T> wkvGm_;
    GlobalTensor<X_T> wgateGm_;
    GlobalTensor<int32_t> cuSeqlensGm_;
    GlobalTensor<int32_t> sequsedGm_;
    GlobalTensor<int32_t> startPosGm_;

    // =================================L1 Buffer=================================
    static constexpr uint32_t L1_X_SIZE = 128 * 1024;
    static constexpr uint32_t L1_W_SIZE = 64 * 1024;
    // L1 Buffer
    TBuf<TPosition::A1> xBufL1;
    TBuf<TPosition::A1> wBufL1;
    // =================================L0 Buffer=================================
    // L0 buffer size
    static constexpr uint32_t L0A_PP_SIZE = 32 * 1024;
    static constexpr uint32_t L0B_PP_SIZE = 32 * 1024;
    static constexpr uint32_t L0C_PP_SIZE = 64 * 1024;
    // L0_A
    TBuf<TPosition::A2> tmpBufL0A;
    // L0_B
    TBuf<TPosition::B2> tmpBufL0B;
    // L0_C
    TBuf<TPosition::CO1> tmpBufL0C;
    // =================================Event&Buffer ID===========================
    // mte2 <> mte1 EventID
    static constexpr uint32_t X_EVENT0 = EVENT_ID0;
    static constexpr uint32_t X_EVENT1 = EVENT_ID1;
    uint32_t xBufId = 0;
    static constexpr uint32_t W_EVENT0 = EVENT_ID2;
    static constexpr uint32_t W_EVENT1 = EVENT_ID3;
    uint32_t wBufId = 0;
    static constexpr uint32_t M_LOCK_EVENT0 = EVENT_ID4;
    static constexpr uint32_t M_LOCK_EVENT1 = EVENT_ID5;
    uint32_t mLockId = 0;
    static constexpr uint32_t N_LOCK_EVENT0 = EVENT_ID6;
    static constexpr uint32_t N_LOCK_EVENT1 = EVENT_ID7;
    uint32_t nLockId = 0;
    // mte1 <> mmad EventID
    static constexpr uint32_t L0AB_EVENT0 = EVENT_ID3;
    static constexpr uint32_t L0AB_EVENT1 = EVENT_ID4;
    uint32_t l0abBufId = 0;
    // mmad <> fixpipe EventID
    static constexpr uint32_t L0C_EVENT0 = EVENT_ID0;
    static constexpr uint32_t L0C_EVENT1 = EVENT_ID1;
    static constexpr uint32_t L0C_EVENT2 = EVENT_ID2;
    static constexpr uint32_t L0C_EVENT3 = EVENT_ID3;
    uint32_t l0cBufId = 0;

};

template <typename COMP>
__aicore__ inline void CompressorBlockCube<COMP>::InitParams(const ConstInfo &constInfo)
{
    this->constInfo_ = constInfo;
}

template <typename COMP> __aicore__ inline void CompressorBlockCube<COMP>::Init(
        __gm__ uint8_t *x,
        __gm__ uint8_t *wKv,
        __gm__ uint8_t *wGate,
        __gm__ uint8_t *kvState,
        __gm__ uint8_t *scoreState,
        __gm__ uint8_t *ape,
        __gm__ uint8_t *normWeight,
        __gm__ uint8_t *ropeSin,
        __gm__ uint8_t *ropeCos,
        __gm__ uint8_t *blockTable,
        __gm__ uint8_t *cuSeqlens,
        __gm__ uint8_t *seqUsed,
        __gm__ uint8_t *startPos,
        __gm__ uint8_t *cmpKvOut,
        __gm__ uint8_t *kvStateOut,
        __gm__ uint8_t *scoreStateOut)
{
    xGm_.SetGlobalBuffer((__gm__ X_T *)x);
    wkvGm_.SetGlobalBuffer((__gm__ X_T *)wKv);
    wgateGm_.SetGlobalBuffer((__gm__ X_T *)wGate);
    startPosGm_.SetGlobalBuffer((__gm__ int32_t *)startPos);
    if (seqUsed != nullptr) {
        sequsedGm_.SetGlobalBuffer((__gm__ int32_t *)seqUsed);
    }
    if constexpr (COMP::xLayout == X_LAYOUT::TH) {
        cuSeqlensGm_.SetGlobalBuffer((__gm__ int32_t *)cuSeqlens);
    }
}

template <typename COMP>
__aicore__ inline void CompressorBlockCube<COMP>::InitBuffers(TPipe *pipe)
{
    // L1
    pipe->InitBuffer(xBufL1, L1_X_SIZE * 2);
    pipe->InitBuffer(wBufL1, L1_W_SIZE * 2);

    // L0
    pipe->InitBuffer(tmpBufL0A, L0A_PP_SIZE * 2);
    pipe->InitBuffer(tmpBufL0B, L0B_PP_SIZE * 2);
    pipe->InitBuffer(tmpBufL0C, L0C_PP_SIZE * 4);
}

template <typename COMP>
__aicore__ inline void CompressorBlockCube<COMP>::AllocEventID(TPipe *pipe)
{
    SetFlag<HardEvent::MTE1_MTE2>(X_EVENT0);
    SetFlag<HardEvent::MTE1_MTE2>(X_EVENT1);
    SetFlag<HardEvent::MTE1_MTE2>(W_EVENT0);
    SetFlag<HardEvent::MTE1_MTE2>(W_EVENT1);
    SetFlag<HardEvent::MTE1_MTE2>(M_LOCK_EVENT0);
    SetFlag<HardEvent::MTE1_MTE2>(M_LOCK_EVENT1);
    SetFlag<HardEvent::MTE1_MTE2>(N_LOCK_EVENT0);
    SetFlag<HardEvent::MTE1_MTE2>(N_LOCK_EVENT1);

    SetFlag<HardEvent::M_MTE1>(L0AB_EVENT0);
    SetFlag<HardEvent::M_MTE1>(L0AB_EVENT1);

    SetFlag<HardEvent::FIX_M>(L0C_EVENT0);
    SetFlag<HardEvent::FIX_M>(L0C_EVENT1);
    SetFlag<HardEvent::FIX_M>(L0C_EVENT2);
    SetFlag<HardEvent::FIX_M>(L0C_EVENT3);
}

template <typename COMP>
__aicore__ inline void CompressorBlockCube<COMP>::FreeEventID(TPipe *pipe)
{
    WaitFlag<HardEvent::MTE1_MTE2>(X_EVENT0);
    WaitFlag<HardEvent::MTE1_MTE2>(X_EVENT1);
    WaitFlag<HardEvent::MTE1_MTE2>(W_EVENT0);
    WaitFlag<HardEvent::MTE1_MTE2>(W_EVENT1);
    WaitFlag<HardEvent::MTE1_MTE2>(M_LOCK_EVENT0);
    WaitFlag<HardEvent::MTE1_MTE2>(M_LOCK_EVENT1);
    WaitFlag<HardEvent::MTE1_MTE2>(N_LOCK_EVENT0);
    WaitFlag<HardEvent::MTE1_MTE2>(N_LOCK_EVENT1);

    WaitFlag<HardEvent::M_MTE1>(L0AB_EVENT0);
    WaitFlag<HardEvent::M_MTE1>(L0AB_EVENT1);

    WaitFlag<HardEvent::FIX_M>(L0C_EVENT0);
    WaitFlag<HardEvent::FIX_M>(L0C_EVENT1);
    WaitFlag<HardEvent::FIX_M>(L0C_EVENT2);
    WaitFlag<HardEvent::FIX_M>(L0C_EVENT3);
}

template <typename COMP>
__aicore__ inline void CompressorBlockCube<COMP>::CopyWeightGmToL1(const RunInfo &info, LocalTensor<X_T> wL1Tensor,
    uint32_t hIdx, uint32_t kBase, uint32_t nLoopIdx)
{
    // hIdx: hidden_size轴的索引ID
    // kBase: hidden_size轴单次往L1搬运的长度, 128
    // nLoopIdx: D方向L1搬运的循环ID, 0/1
    //      
    //      coff=1时, nLoopIdx=0, 搬运wkv nBase, nLoopIdx=1, 搬运wgate nBase
    if constexpr (COMP::coff == COFF::OVERLAP) {
        // coff=2时, constInfo_.dBaseSize = 64, wkv和wgate在D轴各占一半
        // nLoopIdx=0, 搬运coffIdx=0的数据; nLoopIdx=1, 搬运coffIdx=1的数据
        uint64_t coffIdx = nLoopIdx;
        uint64_t dIdx = constInfo_.aiCoreIdx % constInfo_.dBasicBlockNum * constInfo_.dBaseSize;
        uint64_t gmOffset = coffIdx * constInfo_.headDim * constInfo_.hSize + dIdx * constInfo_.hSize + hIdx;
        uint32_t wkvUbOffset = nLoopIdx * (kBase * 2 * constInfo_.dBaseSize);
        uint32_t wgateUbOffset = wkvUbOffset + constInfo_.dBaseSize * (32 / sizeof(X_T));
        CopySingleMatrixNDToNZ(wL1Tensor[wkvUbOffset], wkvGm_[gmOffset], constInfo_.dBaseSize, kBase, constInfo_.hSize, 2 * constInfo_.dBaseSize);
        CopySingleMatrixNDToNZ(wL1Tensor[wgateUbOffset], wgateGm_[gmOffset], constInfo_.dBaseSize, kBase, constInfo_.hSize, 2 * constInfo_.dBaseSize);
    } else {
        // coff=1时, constInfo_.dBaseSize = 128, nLoopIdx=0, 搬运wkv nBase, nLoopIdx=1, 搬运wgate nBase
        uint64_t dIdx = constInfo_.aiCoreIdx % constInfo_.dBasicBlockNum * constInfo_.dBaseSize;
        uint64_t gmOffset = dIdx * constInfo_.hSize + hIdx;
        uint32_t ubOffset = nLoopIdx * kBase * constInfo_.dBaseSize;
        if (nLoopIdx == 0) {
            CopySingleMatrixNDToNZ(wL1Tensor[ubOffset], wkvGm_[gmOffset], constInfo_.dBaseSize, kBase, constInfo_.hSize, constInfo_.dBaseSize);
        } else {
            CopySingleMatrixNDToNZ(wL1Tensor[ubOffset], wgateGm_[gmOffset], constInfo_.dBaseSize, kBase, constInfo_.hSize, constInfo_.dBaseSize);
        }
    }
}

template <typename COMP>
__aicore__ inline void CompressorBlockCube<COMP>::ComputeMm1(const RunInfo &info)
{
    static constexpr uint32_t K_SIZE = 256;
    static constexpr uint32_t K_L1_BASE = 128;
    static constexpr uint32_t N_L1_BASE = 128;
    static constexpr uint32_t M_L1_BASE = 128;
    static constexpr uint32_t K_L1_LOOP = K_SIZE / K_L1_BASE;

    uint32_t nSize = constInfo_.dBaseSize * 2 * ((uint32_t)COMP::coff);
    uint32_t mSize = info.dealTcNum * constInfo_.cmpRatio;
    bool needMLock = (mSize > M_L1_BASE);

    uint32_t hSize = constInfo_.hSize;
    for (uint32_t h = 0; h < hSize; h += K_SIZE) {
        for (uint32_t kL1Idx = 0; kL1Idx < K_L1_LOOP; kL1Idx++) {
            WaitFlag<HardEvent::MTE1_MTE2>(X_EVENT0 + xBufId);
            WaitFlag<HardEvent::MTE1_MTE2>(W_EVENT0 + wBufId);

            LocalTensor<X_T> xL1Tensor = xBufL1.GetWithOffset<X_T>(xBufId * (L1_X_SIZE / sizeof(X_T)), L1_X_SIZE);
            LocalTensor<X_T> wL1Tensor = wBufL1.GetWithOffset<X_T>(wBufId * (L1_W_SIZE / sizeof(X_T)), L1_W_SIZE);
            for (uint32_t nL1 = 0; nL1 < nSize; nL1 += N_L1_BASE) {
                // nLockId = nL1 / N_L1_BASE
                WaitFlag<HardEvent::MTE1_MTE2>(N_LOCK_EVENT0 + nLockId);
                CopyWeightGmToL1(info, wL1Tensor, h + kL1Idx * K_L1_BASE, K_L1_BASE, nL1 / N_L1_BASE);
                SetFlag<HardEvent::MTE2_MTE1>(N_LOCK_EVENT0 + nLockId);
                WaitFlag<HardEvent::MTE2_MTE1>(N_LOCK_EVENT0 + nLockId);
                for (uint32_t mL1 = 0; mL1 < mSize; mL1 += M_L1_BASE) {
                    if (nL1 == 0) {
                        mLockId = mL1 / M_L1_BASE;
                        if (needMLock) {
                            WaitFlag<HardEvent::MTE1_MTE2>(M_LOCK_EVENT0 + mLockId);
                        }
                        // 当Coff=2时，mL1为最后一块时，需要多拷贝一个cmpRatio
                        // CopyXGmToL1(xL1Tensor, mL1, kL1Idx);

                        // m轴可以一次拷贝时, 不需要m轴的拷贝锁；同步替换成大锁
                        if (needMLock) {
                            SetFlag<HardEvent::MTE2_MTE1>(M_LOCK_EVENT0 + mLockId);
                            WaitFlag<HardEvent::MTE2_MTE1>(M_LOCK_EVENT0 + mLockId);
                        } else {
                            SetFlag<HardEvent::MTE2_MTE1>(X_EVENT0 + xBufId);
                            WaitFlag<HardEvent::MTE2_MTE1>(X_EVENT0 + xBufId);
                        }
                    }

                    {
                        // 获取L0C
                        l0cBufId = (nL1 / N_L1_BASE) * 2 + (mL1 / M_L1_BASE); // 2: m方向最多两块, 两块用不满时, 第二块L0C空着
                        WaitFlag<HardEvent::FIX_M>(L0C_EVENT0 + l0cBufId);
                        {
                            WaitFlag<HardEvent::M_MTE1>(L0AB_EVENT0 + l0abBufId);
                            // 当Coff=2时，nL1=0时计算的是pre数据，nL1=N_L1_BASE时计算的是cur数据
                            // LoadAToL0(mL1, nL1);
                            // LoadBToL0(nL1);
                            SetFlag<HardEvent::MTE1_M>(L0AB_EVENT0 + l0abBufId);
                            WaitFlag<HardEvent::MTE1_M>(L0AB_EVENT0 + l0abBufId);
                            // Mmad();
                            SetFlag<HardEvent::M_MTE1>(L0AB_EVENT0 + l0abBufId);
                            l0abBufId = (l0abBufId + 1) % 2;
                        }
                        if ((h + K_SIZE >= hSize) && (kL1Idx + 1 == K_L1_LOOP)) {
                            SetFlag<HardEvent::M_FIX>(L0C_EVENT0 + l0cBufId);
                            WaitFlag<HardEvent::M_FIX>(L0C_EVENT0 + l0cBufId);
                            // FixPipe();
                        }
                        SetFlag<HardEvent::FIX_M>(L0C_EVENT0 + l0cBufId);
                    }

                    if (nL1 + N_L1_BASE >= nSize) {
                        if (needMLock) {
                            mLockId = mL1 / M_L1_BASE;
                            SetFlag<HardEvent::MTE1_MTE2>(M_LOCK_EVENT0 + mLockId);
                        }
                    }
                }
                SetFlag<HardEvent::MTE1_MTE2>(N_LOCK_EVENT0 + nLockId);
                nLockId = (nLockId + 1) % 2;
            }

            SetFlag<HardEvent::MTE1_MTE2>(X_EVENT0 + xBufId);
            xBufId = (xBufId + 1) % 2;
            SetFlag<HardEvent::MTE1_MTE2>(W_EVENT0 + wBufId);
            wBufId = (wBufId + 1) % 2;
        }
    }

}

} // namespace Compressor

#endif // COMPRESSOR_BLOCK_VECTOR_H