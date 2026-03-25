/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file tsqr.cpp
 * \brief
 */
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
class QrHouseholderTilingData {
public:
    int batchSize, mDim, kDim, ubSize;
};
#include "qr_householder_single_vec.h"

using namespace AscendC;

template <typename T> class TsqrKernel {
public:

    __aicore__ inline void Init(GM_ADDR a, GM_ADDR q, GM_ADDR r, GM_ADDR workspace,
        const TsqrTilingData tiling, TPipe* pipe);
    __aicore__ inline void Process();
    __aicore__ inline void ProcessBatch(const GlobalTensor<T>& aGm, const GlobalTensor<T>& qGm, const GlobalTensor<T>& rGm, const GlobalTensor<T>& tmpQGm);
    __aicore__ inline void CallQR(const GlobalTensor<T>& aGm, const GlobalTensor<T>& qGm, const GlobalTensor<T>& rGm, int32_t localM, int32_t localN);
    __aicore__ inline void CopyVec(const GlobalTensor<T>& dst, const GlobalTensor<T>& src, int32_t rows, int32_t columns);
    __aicore__ inline void ForwardZeroStep(const GlobalTensor<T>& aGm, const GlobalTensor<T>& qGm,
        int rId, int32_t localM, int32_t blockSize);
    __aicore__ inline void ForwardStep(int aId, const GlobalTensor<T>& qGm, int rId, int32_t localM, int32_t blockSize);
    __aicore__ inline void Forward(const GlobalTensor<T>& aGm, const GlobalTensor<T>& qGm,
        const GlobalTensor<T>& rGm, const GlobalTensor<T>& tmpQGm, int32_t blockSize);
    __aicore__ inline void Reorder(const GlobalTensor<T>& qGm, bool onCube);
    __aicore__ inline void RunMatmul(int M, int N, bool isTransposeB,
        const GlobalTensor<T>& aGm, const GlobalTensor<T>& bGm, const GlobalTensor<T>& cGm);
    __aicore__ inline void BackwardStep(const GlobalTensor<T>& resultQGm, const GlobalTensor<T>& leftQGm,
        const GlobalTensor<T>& rightQGm, int32_t bsLeft, int32_t numBlocksLeft, bool hasTail);
    __aicore__ inline void Backward(const GlobalTensor<T>& qGm, const GlobalTensor<T>& tmpQGm,
        int32_t blockSize);
    __aicore__ inline GlobalTensor<T> getRBlock(int id);

    GlobalTensor<T> aGlobal_; // Input
    GlobalTensor<T> qGlobal_, rGlobal_; // Output
    GlobalTensor<T> tmpQGlobal_, tmpRGlobal_, bufferQGlobal_; // temporary storage in workspace

    QRHouseholderSingleVec qrSubkernel;
    QrHouseholderTilingData qrTilingData;

    TBuf<TPosition::VECOUT> vecBuf;
    TBuf<TPosition::B1> cubeBuf;
    LocalTensor<T> localTensor;

    int32_t M_, N_, batchSize_;
    int32_t blockSize_, numLevels_, batchFactor_;
    int64_t tmpQSize_, tmpRSize_, bufferQSize_, maxQrWorkspace_;
    __gm__ uint8_t* qrWorkspace;
    TsqrTilingData tiling;
    TPipe* pipe;

    bool firstTail = true;

    MatmulImpl <MatmulType<TPosition::GM, CubeFormat::ND, T, true>, MatmulType<TPosition::GM, CubeFormat::ND, T, true>,
        MatmulType<TPosition::GM, CubeFormat::ND, T>> mm;
    MatmulImpl <MatmulType<TPosition::GM, CubeFormat::ND, T, true>, MatmulType<TPosition::GM, CubeFormat::ND, T, false>,
        MatmulType<TPosition::GM, CubeFormat::ND, T>> mmF;
};


template <typename T>
__aicore__ inline void TsqrKernel<T>::Init(GM_ADDR a, GM_ADDR q, GM_ADDR r, GM_ADDR workspace,
    const TsqrTilingData tiling, TPipe* tpipe) {

    M_ = tiling.m;
    N_ = tiling.n;
    batchSize_ = tiling.batchSize;
    blockSize_ = tiling.blockSize;
    numLevels_ = tiling.numLevels;
    batchFactor = tiling.batchFactor;
    tmpQSize_ = tiling.tmpQSize;
    tmpRSize_ = tiling.tmpRSize;
    bufferQSize_ = tiling.bufferQSize;
    maxQrWorkspace_ = tiling.maxQrWorkspace;

    this->tiling = tiling;
    this->pipe = tpipe;

    aGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(a), batchSize_ * M_ * N_);
    qGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(q), batchSize_ * M_ * N_);
    rGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(r), batchSize_ * N_ * N_);

    __gm__ uint8_t* tmpQPtr = workspace;
    __gm__ uint8_t* tmpRPtr = tmpQPtr + tmpQSize_ * batchFactor_ * sizeof(T);
    __gm__ uint8_t* bufferQPtr = tmpRPtr + tmpRSize_ * batchFactor_ * sizeof(T);
    qrWorkspace = bufferQPtr + bufferQSize_ * batchFactor_ * sizeof(T);
    tmpQGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(tmpQPtr), tmpQSize_ * batchFactor);
    tmpRGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(tmpRPtr), tmpRSize_ * batchFactor);
    bufferQGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(bufferQPtr), bufferQSize_ * batchFactor);

}

template <typename T>
__aicore__ inline GlobalTensor<T> TsqrKernel<T>::getRBlock(int id) {
    int64_t offset = (id * N_ * N_);
    return tmpRGlobal_[offset];
}

template <typename T>
__aicore__ inline void TsqrKernel<T>::Process() {
    for (int i = 0; i < batchSize_; i += batchFactor_) {
        ProcessBatch(aGlobal_[M_ * N_ * i], qGlobal_[M_ * N_ * i], rGlobal_[N_ * N_ * i], tmpQGlobal_);
    }
}

template <typename T>
__aicore__ inline void TsqrKernel<T>::ProcessBatch(const GlobalTensor<T>& aGm, const GlobalTensor<T>& qGm, const GlobalTensor<T>& rGm, const GlobalTensor<T>& tmpQGm) {
    if ASCEND_IS_AIV {
        Forward(aGm, qGm, rGm, tmpQGm, blockSize_);
    }
    SyncAll<false>();
    #if defined(__DAV_C310__)
    if ASCEND_IS_AIV {
        if (GetBlockIdx() == 0) {
            Reorder(tmpQGm, false);
        }
    }
    SyncAll<false>();
    #else
    if ASCEND_IS_AIC {
        if (GetBlockIdx() == 0) {
            Reorder(tmpQGm, true);
        }
    }
    #endif
    SyncAll<false>();
    Backward(qGm, tmpQGm, blockSize_);
    SyncAll<false>();
}

template <typename T>
__aicore__ inline void TsqrKernel<T>::CallQR(const GlobalTensor<T>& aGm, const GlobalTensor<T>& qGm,
    const GlobalTensor<T>& rGm, int32_t localM, int32_t localN) {

    PipeBarrier<PIPE_ALL>();
    pipe->Reset();
    PipeBarrier<PIPE_ALL>();
    qrTilingData.batchSize = 1;
    qrTilingData.mDim = localM;
    qrTilingData.kDim = localN;
    qrTilingData.ubSize = tiling.ubSize;
    __gm__ uint8_t* aGmPtr = reinterpret_cast<__gm__ uint8_t*>(const_cast<__gm__ float*>(aGm.GetPhyAddr()));
    __gm__ uint8_t* qGmPtr = reinterpret_cast<__gm__ uint8_t*>(const_cast<__gm__ float*>(qGm.GetPhyAddr()));
    __gm__ uint8_t* rGmPtr = reinterpret_cast<__gm__ uint8_t*>(const_cast<__gm__ float*>(rGm.GetPhyAddr()));
    __gm__ uint8_t* localWorkspace = qrWorkspace + GetBlockIdx() * maxQrWorkspace_ * sizeof(T);
    qrSubkernel.Init(aGmPtr, qGmPtr, rGmPtr, localWorkspace, &qrTilingData, pipe);
    qrSubkernel.Process();
}

template <typename T>
__aicore__ inline void TsqrKernel<T>::CopyVec(const GlobalTensor<T>& dst, const GlobalTensor<T>& src, int32_t rows, int32_t columns) {
    PipeBarrier<PIPE_ALL>();
    pipe->Reset();
    PipeBarrier<PIPE_ALL>();

    pipe->InitBuffer(vecBuf, rows * columns * sizeof(T));
    LocalTensor localTensor = vecBuf.Get<T>();
    DataCopyParams params = { static_cast<uint16_t>(columns), static_cast<uint16_t>(rows / 8), 0, 0 };
    DataCopy(localTensor, src, params);
    int32_t eventIDMTE2ToMTE3 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
    SetFlag<HardEvent::MTE2_MTE3>(eventIDMTE2ToMTE3);
    WaitFlag<HardEvent::MTE2_MTE3>(eventIDMTE2ToMTE3);
    DataCopy(dst, localTensor, params);
}

template <typename T>
__aicore__ inline void TsqrKernel<T>::ForwardZeroStep(const GlobalTensor<T>& aGm, const GlobalTensor<T>& qGm,
    int rId, int32_t localM, int32_t blockSize) {
    int numBlocksPerBatch = localM / blockSize;
    int numBlocks = numBlocksPerBatch * batchFactor_;
    int coreIdx = AscendC::GetBlockIdx();
    int numCores = GetBlockNum() * 2; // vector cores
    #if defined(__DAV_C220_VEC__)
    if (N_ % 32 != 0 && N_ > 128) {
        numCores /= 2;
        if (coreIdx % 2 == 0) {
            coreIdx /= 2;
        } else {
            coreIdx = numCores;
        }
    }
    #endif
    if (numBlocks < numCores) {
        numCores = numBlocks; // don't use other cores
    }
    if (coreIdx < numCores) {
        int perCore = numBlocks / numCores;
        int tail = numBlocks % numCores;
        int start, end;
        if (coreIdx < tail) {
            start = (perCore + 1) * coreIdx;
            end = start + perCore + 1;
        } else {
            start = perCore * coreIdx + tail;
            end = start + perCore;
        }
        for (int idx = start; idx < end; idx++) {
            int64_t qaOffset = idx * blockSize * N_;
            CallQR(aGm[qaOffset], qGm[qaOffset], getRBlock(rId + idx), blockSize, N_);
        }
    }
    SyncAll<true>();
}

template <typename T>
__aicore__ inline void TsqrKernel<T>::ForwardStep(int aId, const GlobalTensor<T>& qGm,
    int rId, int32_t localM, int32_t blockSize) {
    int numBlocks = localM / blockSize;
    int hasTail = numBlocks % 2;
    int processedBlocksPerBatch = numBlocks / 2 + hasTail;
    blockSize *= 2;
    int processedBlocks = processedBlocksPerBatch * batchFactor_;
    int coreIdx = AscendC::GetBlockIdx();
    int numCores = GetBlockNum() * 2; // vector cores
    if (processedBlocks < numCores) {
        numCores = processedBlocks; // don't use other cores
    }
    if (coreIdx < numCores) {
        int perCore = processedBlocks / numCores;
        int tail = processedBlocks % numCores;
        int start, end;
        if (coreIdx < tail) {
            start = (perCore + 1) * coreIdx;
            end = start + perCore + 1;
        } else {
            start = perCore * coreIdx + tail;
            end = start + perCore;
        }
        for (int idx = start; idx < end; idx++) {
            int64_t qOffset = idx * blockSize * N_;
            int correction = (idx / processedBlocksPerBatch) * hasTail;
            if (hasTail && idx > 0 && ((idx + 1) % processedBlocksPerBatch == 0)) {
                // unpaired block
                CopyVec(getRBlock(rId + idx), getRBlock(aId + 2 * idx - correction), N_, N_);
            } else {
                CallQR(getRBlock(aId + 2 * idx - correction), qGm[qOffset - (int64_t)correction], getRBlock(rId + idx), blockSize, N_);
            }
        }
    }
    SyncAll<true>();
}

template <typename T>
__aicore__ inline void TsqrKernel<T>::Forward(const GlobalTensor<T>& aGm, const GlobalTensor<T>& qGm,
    const GlobalTensor<T>& rGm, const GlobalTensor<T>& tmpQGm, int32_t blockSize) {
    // zero step
    // aGm -> qGm, tmpRGlobal_
    ForwardZeroStep(aGm, tmpQGm, 0, M_, blockSize);
    // blocks are stacked in paires
    int numBlocks = M_ / blockSize;
    int numPairs = numBlocks / 2;
    int tail = (numBlocks % 2 > 0);
    int64_t aOffset = 0;
    int64_t qOffset = M_ * N_;
    int64_t rOffset = numBlocks;
    int localM = numBlocks * N_;
    for (int lvl = 0; lvl < numLevels_ - 1; lvl++) {
        // tmpRGlobal_ -> tmpQGlobal_, tmpRGlobal_
        ForwardStep(aOffset * batchFactor_, tmpQGm[qOffset * batchFactor_], rOffset * batchFactor_, localM, N_);

        aOffset += numBlocks;
        qOffset += (numPairs * 2 + tail) * N_ * N_;
        rOffset += (numPairs + tail);
        numBlocks = numPairs + tail;
        numPairs = numBlocks / 2;
        tail = (numBlocks % 2 > 0);
        localM = numBlocks * N_;

    }
    // last iteration
    // tmpRGlobal_ -> tmpQGlobal_, rGm
    if (GetBlockIdx() == 0) {
        for (int batch = 0; batch < batchFactor_; batch++) {
            CallQR(getRBlock(aOffset * batchFactor_ + batch * 2), tmpQGm[qOffset * batchFactor_ + batch * 2 * N_ * N_], rGm[batch * N_ * N_], 2 * N_, N_);
        }        
    }
}

template <typename T>
__aicore__ inline void TsqrKernel<T>::Reorder(const GlobalTensor<T>& qGm, bool onCube) {
    PipeBarrier<PIPE_ALL>();
    pipe->Reset();
    PipeBarrier<PIPE_ALL>();

    int numBlocks = M_ / blockSize_;
    int numPairs = numBlocks / 2;
    int tail = (numBlocks % 2 > 0);
    int64_t qOffset = M_ * N_;
    for (int lvl = 0; lvl < numLevels_ - 1; lvl++) {
        qOffset += (numPairs * 2 + tail) * N_ * N_;
        numBlocks = numPairs + tail;
        numPairs = numBlocks / 2;
        tail = (numBlocks % 2 > 0);
    }

    LocalTensor<T> localTensor;
    if (onCube) {
        pipe->InitBuffer(cubeBuf, 2 * N_ * N_ * sizeof(T));
        localTensor = cubeBuf.Get<float>();
    } else {
        pipe->InitBuffer(vecBuf, 2 * N_ * N_ * sizeof(T));
        localTensor = vecBuf.Get<float>();
    }
    uint16_t n = static_cast<uint16_t>(N_);
    uint16_t blocks = static_cast<uint16_t>(N_ / 8);

    DataCopyParams inParams = { static_cast<uint16_t>(N_), blocks, blocks, 0 };
    DataCopyParams outParams = { n, static_cast<uint16_t>(2 * n / 8), 0, 0 };

    for (int batch = 0; batch < batchFactor_; batch++) {
        GlobalTensor<T> global = qGm[qOffset * batchFactor_ + batch * 2 * N_ * N_];

        DataCopy(localTensor, global, inParams);
        DataCopy(localTensor[N_ * N_], global[N_], inParams);

        int32_t eventIDMTE2ToMTE3 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
        SetFlag<HardEvent::MTE2_MTE3>(eventIDMTE2ToMTE3);
        WaitFlag<HardEvent::MTE2_MTE3>(eventIDMTE2ToMTE3);

        DataCopy(global, localTensor, outParams);

        int32_t eventIDMTE3ToMTE2 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
        WaitFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
    }
}

template <typename T>
__aicore__ inline void TsqrKernel<T>::RunMatmul(int M, int N, bool isTransposeB,
    const GlobalTensor<T>& aGm, const GlobalTensor<T>& bGm, const GlobalTensor<T>& cGm) {
    PipeBarrier<PIPE_ALL>();
    pipe->Reset();
    PipeBarrier<PIPE_ALL>();
    if (isTransposeB) {
        mm.SetSubBlockIdx(0);
        mm.Init(&tiling.mmTilingData, pipe);
        mm.SetOrgShape(M, N, N);
        mm.SetSingleShape(M, N, N);
        mm.SetTensorA(aGm, true);
        mm.SetTensorB(bGm, true);
        mm.IterateAll(cGm);
        mm.End();
    } else {
        mmF.SetSubBlockIdx(0);
        mmF.Init(&tiling.mmTilingDataF, pipe);
        mmF.SetOrgShape(M, N, N);
        mmF.SetSingleShape(M, N, N);
        mmF.SetTensorA(aGm, true);
        mmF.SetTensorB(bGm, false);
        mmF.IterateAll(cGm);
        mmF.End();
    }
}

template <typename T>
__aicore__ inline void TsqrKernel<T>::BackwardStep(const GlobalTensor<T>& outQGm, const GlobalTensor<T>& leftQGm,
    const GlobalTensor<T>& rightQGm, int32_t bsLeft, int32_t numBlocksLeft, bool hasTail) {

    int coreIdx = AscendC::GetBlockIdx();
    int numCores = GetBlockNum(); // cube cores
    int processedBlocks = numBlocksLeft * batchFactor_;
    if (processedBlocks < numCores) {
        numCores = processedBlocks; // don't use other cores
    }
    if (coreIdx < numCores) {
        int perCore = processedBlocks / numCores;
        int start, end;
        if (coreIdx < processedBlocks % numCores) {
            start = (perCore + 1) * coreIdx;
            end = start + perCore + 1;
        } else {
            start = perCore * coreIdx + processedBlocks % numCores;
            end = start + perCore;
        }
        int64_t leftOffset = bsLeft * N_;
        int64_t rightOffset = N_ * N_;
        int64_t outQOffset = bsLeft * N_;
        for (int idx = start; idx < end; idx++) {
            PipeBarrier<PIPE_ALL>();
            bool isFirstIter = (numBlocksLeft == 2);
            bool isTransfered = ((idx + 1) % numBlocksLeft == 0) && (numBlocksLeft % 2 == 1) && (!hasTail) && firstTail;
            int64_t correction = (idx / numBlocksLeft) * hasTail * (bsLeft - N_) * N_;
            if (hasTail && idx > 0 && ((idx + 1) % numBlocksLeft == 0)) {
                // unpaired block
                if ASCEND_IS_AIV {
                    CopyVec(outQGm[idx * outQOffset - correction], rightQGm[idx * rightOffset], N_, N_);
                }
            } else {
                if ASCEND_IS_AIC {
                    PipeBarrier<PIPE_ALL>();
                    pipe->Reset();
                    PipeBarrier<PIPE_ALL>();
                    RunMatmul(bsLeft, N_, (isFirstIter || isTransfered),
                        leftQGm[idx * leftOffset - correction], rightQGm[idx * rightOffset], outQGm[idx * outQOffset - correction]);
                }
            }
        }
    }
    PipeBarrier<PIPE_ALL>();
    if ((numBlocksLeft % 2 == 1) && (!hasTail) && firstTail) {
        firstTail = false;
    }
}

template <typename T>
__aicore__ inline void TsqrKernel<T>::Backward(const GlobalTensor<T>& qGm, const GlobalTensor<T>& tmpQGm, int32_t blockSize) {
    int numBlocks, numPairs, tail, lvl;
    int64_t rightOffset, leftOffset;
    bool hasTail;
    for (int lvl = numLevels_ - 1; lvl > 0; lvl--) {
        numBlocks = M_ / blockSize;
        numPairs = numBlocks / 2;
        tail = (numBlocks % 2 > 0);
        rightOffset = M_ * N_;
        leftOffset = 0;
        for (int __lvl = 0; __lvl < lvl; __lvl++) {
            leftOffset = rightOffset;
            rightOffset += (numPairs * 2 + tail) * N_ * N_;
            numBlocks = numPairs + tail;
            numPairs = numBlocks / 2;
            hasTail = tail;
            tail = (numBlocks % 2 > 0);
        }
        int64_t rrightOffset = rightOffset + (numPairs * 2 + tail) * N_ * N_;
        if (lvl == numLevels_ - 1) {
            firstTail = hasTail;
        }
        if ((numLevels_ - 1 - lvl) % 2 == 0) {
            if (lvl == numLevels_ - 1) {
                BackwardStep(bufferQGlobal_, tmpQGm[leftOffset * batchFactor_], tmpQGm[rightOffset * batchFactor_], 2 * N_, numBlocks, hasTail);
            } else {
                BackwardStep(bufferQGlobal_, tmpQGm[leftOffset * batchFactor_], tmpQGm[rrightOffset * batchFactor_], 2 * N_, numBlocks, hasTail);
            }
        } else {
            BackwardStep(tmpQGm[rightOffset * batchFactor_], tmpQGm[leftOffset * batchFactor_], bufferQGlobal_, 2 * N_, numBlocks, hasTail);
        }
        SyncAll<false>();
    }
    numBlocks = M_ / blockSize_;
    tail = (numBlocks % 2 > 0);
    numPairs = numBlocks / 2;
    rightOffset = M_ * N_;
    for (int __lvl = 0; __lvl < 1; __lvl++) {
        rightOffset += (numPairs * 2 + tail) * N_ * N_;
        numBlocks = numPairs + tail;
        numPairs = numBlocks / 2;
        hasTail = tail;
        tail = (numBlocks % 2 > 0);
    }
    if ((numLevels_ - 1) % 2 == 0) {
        BackwardStep(qGm, tmpQGm, tmpQGm[rightOffset * batchFactor_], blockSize_, M_ / blockSize_, false);
    } else {
        BackwardStep(qGm, tmpQGm, bufferQGlobal_, blockSize_, M_ / blockSize_, false);
    }
}

extern "C" __global__ __aicore__ void tsqr(GM_ADDR a, GM_ADDR q, GM_ADDR r,
    GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    TsqrKernel<float> tsqrKernel;
    AscendC::TPipe pipe;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    GM_ADDR user = GetUserWorkspace(workspace);

    tsqrKernel.Init(a, q, r, user, tilingData, &pipe);
    tsqrKernel.Process();
}
