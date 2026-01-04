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
 * \file rotate_matrix.h
 * \brief
 */
#ifndef ROTATE_MATRIX_H
#define ROTATE_MATRIX_H

#include "lib/matmul_intf.h"

struct MMConfig {
    // BNSD
    int32_t bn_ = 1;    // B*N
    int32_t m_ = 32;    // S
    int32_t n_ = 128;   // D
    int32_t k_ = 128;   // D
    int32_t baseM_;
    int32_t baseN_;
    int32_t baseK_;
    int32_t curBaseM_;
    int32_t curBaseN_;
    int32_t curBaseK_;
    int32_t baseMN_;
    int32_t sdSize_;
    int32_t blockNum_;
    int32_t blockNumM_;
    int32_t blockNumN_;
};

struct CVConfig {
    int32_t blockIdx;
    int32_t cvParall;       // 当前cv缓存块序号
    int32_t cvParallNum;
};

namespace RotateMatrix {
using namespace AscendC;
using namespace matmul;

constexpr int32_t SINGLE_BUFFER = 1;
constexpr int32_t DOUBLE_BUFFER = 2;
constexpr int32_t HALF = 2;
constexpr int32_t ALIGN_32 = 32;

// Matmul类型定义 - 用于Cube核矩阵乘法
using aT = MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t>;
using bT = MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t>;
using cT = MatmulType<TPosition::GM, CubeFormat::ND, float>;
using MT = matmul::MatmulImpl<aT, bT, cT>;

template <typename T>
class RotateMatrixBNSD {
public:
    __aicore__ inline RotateMatrixBNSD(){}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR cos, GM_ADDR sin, GM_ADDR rotate, GM_ADDR y, GM_ADDR workSpace,
                                const RotaryPositionEmbeddingTilingData &tilingData, TPipe *pipe);
    __aicore__ inline void InitData(const RotaryPositionEmbeddingTilingData &tiling);
    __aicore__ inline void Process();
    __aicore__ inline void AICProcess(int64_t offset);
    __aicore__ inline void innerProcess(int64_t bnIdx);
    __aicore__ inline void XCosProcess(int64_t offset, int64_t baseMNOffset, int64_t relativeOffset);
    __aicore__ inline void XRSinProcess(int64_t offset, int64_t baseMNOffset, int64_t relativeOffset);
    template <typename U>
    __aicore__ inline void AIVCopyIn(GlobalTensor<U> xGM);
    __aicore__ inline void AIVCopyOut(GlobalTensor<T> xGM);

protected:
    MT mm_;  // Matmul对象
    TQue<QuePosition::VECIN, SINGLE_BUFFER> inQueue_;
    TQue<QuePosition::VECOUT, SINGLE_BUFFER> outQueue_;
    GlobalTensor<T> xGm_;
    GlobalTensor<T> cosGm_;
    GlobalTensor<T> sinGm_;
    GlobalTensor<T> rotateGm_;
    GlobalTensor<T> yGm_;
    GlobalTensor<float> xRotatedGm_; // 存放矩阵乘法结果
    TBuf<TPosition::VECCALC> calcBuf_;
    LocalTensor<float> xCosLocal_;
    int32_t blockIdx_;
    int32_t subBlockIdx_;
    int32_t coreNum_;
    MMConfig mmConfig_;
    CVConfig cvConfig_;
    TCubeTiling cubeTiling_;  // Matmul Tiling数据
};

template <typename T>
__aicore__ inline void RotateMatrixBNSD<T>::InitData(const RotaryPositionEmbeddingTilingData &tiling)
{
    const RotateMatrixParams &rotateTiling = tiling.rotateMatrixParams;
    // BNSD
    mmConfig_.bn_ = rotateTiling.bn;                            // B * N
    mmConfig_.m_ = rotateTiling.totalSLines;                    // S
    mmConfig_.n_ = rotateTiling.dLength;                        // D
    mmConfig_.k_ = mmConfig_.n_;                                // D
    mmConfig_.baseM_ = rotateTiling.baseM;
    mmConfig_.baseN_ = rotateTiling.baseN;
    mmConfig_.baseK_ = rotateTiling.baseK;                      // D
    mmConfig_.baseMN_ = mmConfig_.baseM_ * mmConfig_.baseN_;
    mmConfig_.blockNumM_ = rotateTiling.blockNumM;              // SD块内的baseMN块个数
    mmConfig_.blockNumN_ = rotateTiling.blockNumN;              // SD块内的baseMN块个数
    mmConfig_.blockNum_ = rotateTiling.blockNum;                // SD块内的baseMN块个数
    cvConfig_.cvParallNum = rotateTiling.cvParallNum;
    mmConfig_.sdSize_ = mmConfig_.m_ * mmConfig_.k_;            // BNSD内的SD块大小
    coreNum_ = rotateTiling.coreNum;
}

template <typename T>
__aicore__ inline void RotateMatrixBNSD<T>::Init(GM_ADDR x, GM_ADDR cos, GM_ADDR sin, GM_ADDR rotate,
    GM_ADDR y, GM_ADDR workSpace, const RotaryPositionEmbeddingTilingData &tilingData, TPipe *pipe)
{
    InitData(tilingData);
    blockIdx_ = GetBlockIdx();
    subBlockIdx_ = GetSubBlockIdx();
    xGm_.SetGlobalBuffer((__gm__ T *) x);
    yGm_.SetGlobalBuffer((__gm__ T *) y);
    cosGm_.SetGlobalBuffer((__gm__ T *) cos);
    sinGm_.SetGlobalBuffer((__gm__ T *) sin);
    rotateGm_.SetGlobalBuffer((__gm__ T *) rotate);
    xRotatedGm_.SetGlobalBuffer((__gm__ float *) workSpace); // 暂存矩阵输出

    pipe->InitBuffer(inQueue_, SINGLE_BUFFER, Ceil(mmConfig_.baseMN_ / HALF, ALIGN_32) * HALF * ALIGN_32 * sizeof(T));
    pipe->InitBuffer(outQueue_, SINGLE_BUFFER, Ceil(mmConfig_.baseMN_ / HALF, ALIGN_32) * HALF * ALIGN_32 * sizeof(T));
    pipe->InitBuffer(calcBuf_, Ceil(mmConfig_.baseMN_ / HALF, ALIGN_32) * HALF * HALF * ALIGN_32 * sizeof(float));

    // 初始化Matmul Tiling
    cubeTiling_ = tilingData.rotateMatrixParams.matmulTiling;
    if ASCEND_IS_AIC {
        mm_.Init(&cubeTiling_, pipe);
    }
}

template <typename T>
__aicore__ inline void RotateMatrixBNSD<T>::Process()
{
    for (int i = 0; i < mmConfig_.bn_; ++i) {
        // 重置block块和cv并行序号
        cvConfig_.blockIdx = blockIdx_;
        cvConfig_.cvParall = 0;
        if ASCEND_IS_AIV {
            cvConfig_.blockIdx /= 2;
        }

        innerProcess(i);
        
        // 防止BN间SD内容互相影响
        if ASCEND_IS_AIV {
            AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(0x1);
            AscendC::CrossCoreWaitFlag(0x3);
        }
        if ASCEND_IS_AIC {
            AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(0x3);
            AscendC::CrossCoreWaitFlag(0x1);
        }
    }
}

template <typename T>
__aicore__ inline void RotateMatrixBNSD<T>::innerProcess(int64_t bnIdx)
{
    while (cvConfig_.blockIdx < mmConfig_.blockNum_) {
        // 本轮起始位置
        auto idxM = cvConfig_.blockIdx / mmConfig_.blockNumN_;
        auto idxN = cvConfig_.blockIdx % mmConfig_.blockNumN_;

        // 计算当前baseMN块的偏移
        int64_t mOffset = idxM * mmConfig_.baseM_ * mmConfig_.k_;
        int64_t nOffset = idxN * mmConfig_.baseN_;
        int64_t baseOffset = bnIdx * mmConfig_.sdSize_ + mOffset + nOffset;
        
        mmConfig_.curBaseM_ = (idxM + 1) * mmConfig_.baseM_ < mmConfig_.m_ ?
                              mmConfig_.baseM_ : mmConfig_.m_ - idxM * mmConfig_.baseM_;
        if ((idxN + 1) * mmConfig_.baseN_ > mmConfig_.n_) {
            baseOffset -= mmConfig_.baseN_ - (mmConfig_.n_ - nOffset);
        }
        mmConfig_.curBaseN_ = mmConfig_.baseN_;
        if ASCEND_IS_AIV {
            // 两个AIV, 各处理一半baseM
            if (unlikely(mmConfig_.curBaseM_ < 2 && subBlockIdx_ == 0)) {
                cvConfig_.blockIdx += coreNum_;
                AscendC::CrossCoreWaitFlag(0x5);
                if (cvConfig_.cvParall > cvConfig_.cvParallNum - 2) {
                    CrossCoreSetFlag<2, PIPE_MTE3>(0x2);
                }
                cvConfig_.cvParall++;
                continue;
            }
            // 重写baseM偏移大小及起始地址
            baseOffset = subBlockIdx_ == 0 ? baseOffset : baseOffset + (mmConfig_.curBaseM_ / 2 * mmConfig_.k_);
            int64_t relativeOffset = mmConfig_.curBaseM_ / 2 * mmConfig_.curBaseN_; // base块内偏移
            int64_t sinCosOffset = subBlockIdx_ == 0 ? (mOffset + nOffset)
                                   : (mOffset + nOffset + mmConfig_.curBaseM_ / 2 * mmConfig_.k_);
            mmConfig_.curBaseM_= subBlockIdx_ == 0 ? 
                                 (mmConfig_.curBaseM_ / 2) : (mmConfig_.curBaseM_ - mmConfig_.curBaseM_ / 2);
            XCosProcess(baseOffset, sinCosOffset, relativeOffset);
            XRSinProcess(baseOffset, sinCosOffset, relativeOffset);
        }

        if ASCEND_IS_AIC {
            AICProcess(baseOffset);
            AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(0x5);
            if (cvConfig_.cvParall > cvConfig_.cvParallNum - 2) {
                AscendC::CrossCoreWaitFlag(0x2);
            }
        }
        cvConfig_.blockIdx += coreNum_;
        cvConfig_.cvParall++;
    }
}

template <typename T>
__aicore__ inline void RotateMatrixBNSD<T>::XCosProcess(int64_t offset, int64_t baseMNOffset, int64_t relativeOffset)
{
    int64_t aivOffset = subBlockIdx_ * relativeOffset;
    AIVCopyIn<T>(xGm_[offset]); // baseMN, GM->UB
    LocalTensor<T> xLocal = inQueue_.DeQue<T>();

    auto buff = calcBuf_.Get<float>();
    Cast(buff[aivOffset], xLocal, RoundMode::CAST_NONE, mmConfig_.curBaseM_ * mmConfig_.curBaseN_); // x: bf16 -> fp32
    inQueue_.FreeTensor(xLocal);

    AIVCopyIn<T>(cosGm_[baseMNOffset]); // baseMN, GM->UB
    LocalTensor<T> cosLocal = inQueue_.DeQue<T>();
    Cast(buff[mmConfig_.baseMN_ + aivOffset], cosLocal,
         RoundMode::CAST_NONE, mmConfig_.curBaseM_ * mmConfig_.curBaseN_); // cos: bf16 -> fp32
    inQueue_.FreeTensor(cosLocal);

    xCosLocal_ = buff[mmConfig_.baseMN_];
    Mul(xCosLocal_[aivOffset], buff[mmConfig_.baseMN_ + aivOffset],
        buff[aivOffset], mmConfig_.curBaseM_ * mmConfig_.curBaseN_);
}

template <typename T>
__aicore__ inline void RotateMatrixBNSD<T>::XRSinProcess(int64_t offset, int64_t baseMNOffset, int64_t relativeOffset)
{
    int64_t aivOffset = subBlockIdx_ * relativeOffset;
    AIVCopyIn<T>(sinGm_[baseMNOffset]); // baseMN, GM->UB
    LocalTensor<T> sinLocal = inQueue_.DeQue<T>();
    auto buff = calcBuf_.Get<float>();
    Cast(buff[aivOffset], sinLocal, RoundMode::CAST_NONE, mmConfig_.curBaseM_ * mmConfig_.curBaseN_); // sin: bf16 -> fp32
    inQueue_.FreeTensor(sinLocal);

    AscendC::CrossCoreWaitFlag(0x5);

    LocalTensor<float> xRLocal = inQueue_.AllocTensor<float>();
    DataCopy(xRLocal, xRotatedGm_[blockIdx_ / 2 * cvConfig_.cvParallNum * mmConfig_.baseMN_
             + (cvConfig_.cvParall % cvConfig_.cvParallNum) * mmConfig_.baseMN_ + aivOffset],
             static_cast<uint32_t>(mmConfig_.curBaseM_ * mmConfig_.curBaseN_));

    SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);

    if (cvConfig_.cvParall > cvConfig_.cvParallNum - 2) {
        CrossCoreSetFlag<2, PIPE_MTE2>(0x2);
    }
    
    Mul(buff[aivOffset], xRLocal, buff[aivOffset], mmConfig_.curBaseM_ * mmConfig_.curBaseN_); // x_r * sin
    PipeBarrier<PIPE_V>();
    
    // x * cos + x_r * sin
    Add(xCosLocal_[aivOffset], buff[aivOffset], xCosLocal_[aivOffset], mmConfig_.curBaseM_ * mmConfig_.curBaseN_);
    PipeBarrier<PIPE_V>();

    LocalTensor<T> yLocal = outQueue_.AllocTensor<T>();
    Cast(yLocal, xCosLocal_[aivOffset], RoundMode::CAST_RINT, mmConfig_.curBaseM_ * mmConfig_.curBaseN_);

    outQueue_.EnQue(yLocal);
    AIVCopyOut(yGm_[offset]);
    inQueue_.FreeTensor(xRLocal);
}

template <typename T>
template <typename U>
__aicore__ inline void RotateMatrixBNSD<T>::AIVCopyIn(GlobalTensor<U> xGM)
{
    // 需要32B对齐, 否则会出现padding补齐
    LocalTensor<U> xLocal = inQueue_.AllocTensor<U>();
    // 拷入当前基本块大小的矩阵
    DataCopyExtParams copyParams;
    copyParams.blockCount = static_cast<uint16_t>(mmConfig_.curBaseM_);
    copyParams.blockLen = static_cast<uint32_t>(mmConfig_.curBaseN_ * sizeof(U));
    copyParams.srcStride = static_cast<uint32_t>((mmConfig_.k_ - mmConfig_.curBaseN_) * sizeof(U));
    copyParams.dstStride = static_cast<uint32_t>(0);
    DataCopyPadExtParams<U> copyPadParams{true, 0, 0, 0};
    DataCopyPad(xLocal, xGM, copyParams, copyPadParams);
    inQueue_.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void RotateMatrixBNSD<T>::AIVCopyOut(GlobalTensor<T> yGM)
{
    LocalTensor<T> yLocal = outQueue_.DeQue<T>();
    // 拷出当前基本块大小的矩阵 复用
    DataCopyExtParams copyParams;
    copyParams.blockCount = static_cast<uint16_t>(mmConfig_.curBaseM_);                                 // 行数
    copyParams.blockLen = static_cast<uint32_t>(mmConfig_.curBaseN_ * sizeof(T));                       // 每个连续数据块长度，单位长度 32B
    copyParams.srcStride = static_cast<uint32_t>(0);                                                    // 相邻块的间隔
    copyParams.dstStride = static_cast<uint32_t>((mmConfig_.k_ - mmConfig_.curBaseN_) * sizeof(T));     // 相邻块的间隔
    DataCopyPad(yGM, yLocal, copyParams);
    outQueue_.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void RotateMatrixBNSD<T>::AICProcess(int64_t offset)
{
    int64_t xOffset = offset / mmConfig_.k_ * mmConfig_.k_;
    int64_t rotateOffset = offset - xOffset;

    mm_.SetOrgShape(mmConfig_.m_, mmConfig_.n_, mmConfig_.k_);
    mm_.SetSingleShape(mmConfig_.curBaseM_, mmConfig_.curBaseN_, mmConfig_.k_);
    mm_.SetTensorA(xGm_[xOffset]);
    mm_.SetTensorB(rotateGm_[rotateOffset]);
    while (mm_.Iterate()) {
        mm_.GetTensorC(xRotatedGm_[blockIdx_ * cvConfig_.cvParallNum * mmConfig_.baseMN_
                       + (cvConfig_.cvParall % cvConfig_.cvParallNum) * mmConfig_.baseMN_], 0, true);    // true: 开启连续写
    }
    mm_.End();
}

}
#endif // ROTATE_MATRIX_H