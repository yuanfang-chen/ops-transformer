/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. 
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef APPLY_ADAM_W_V3_DAG_H
#define APPLY_ADAM_W_V3_DAG_H

#include "kernel_operator.h"

namespace ApplyAdamwV3Ns {
using namespace AscendC;

constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t TMP_BUFFER_NUM = 12;

template <typename T>
class ApplyAdamwV3Kernel {
public:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue;
    TBuf<QuePosition::VECCALC> tmpBuf;
    
    __aicore__ inline ApplyAdamwV3Kernel() {}
    
    __aicore__ inline void Init(GM_ADDR var, GM_ADDR m, GM_ADDR v, 
                                 GM_ADDR grad, GM_ADDR var_out, GM_ADDR m_out, GM_ADDR v_out,
                                 uint32_t totalLen, uint32_t tileLen, uint32_t loopNum,
                                 float b1p, float b2p, float l, float wd, float b1, float b2, float eps, float mf,
                                 uint32_t coreOff = 0, uint32_t elemThisCore = 0);
    __aicore__ inline void Process();
    
private:
    __aicore__ inline void CopyIn(uint32_t loopIdx);
    __aicore__ inline void Compute(uint32_t loopIdx);
    __aicore__ inline void CopyOut(uint32_t loopIdx);
    
    GlobalTensor<T> varGm;
    GlobalTensor<T> mGm;
    GlobalTensor<T> vGm;
    GlobalTensor<T> gradGm;
    GlobalTensor<T> varOutGm;
    GlobalTensor<T> mOutGm;
    GlobalTensor<T> vOutGm;
    
    uint32_t blockIdx;
    uint32_t totalLength;
    uint32_t tileLength;
    uint32_t tileNum;
    uint32_t coreOffset;
    uint32_t elementsThisCore;
    
    float beta1Power;
    float beta2Power;
    float lr;
    float weightDecay;
    float beta1;
    float beta2;
    float epsilon;
    float maximizeFactor;
};

template <typename T>
__aicore__ inline void ApplyAdamwV3Kernel<T>::Init(
    GM_ADDR var, GM_ADDR m, GM_ADDR v, GM_ADDR grad,
    GM_ADDR var_out, GM_ADDR m_out, GM_ADDR v_out,
    uint32_t totalLen, uint32_t tileLen, uint32_t loopNum,
    float b1p, float b2p, float l, float wd, float b1, float b2, float eps, float mf,
    uint32_t coreOff, uint32_t elemThisCore)
{
    blockIdx = GetBlockIdx();
    totalLength = totalLen;
    tileLength = tileLen;
    tileNum = loopNum;
    coreOffset = coreOff;
    elementsThisCore = elemThisCore;
    
    beta1Power = b1p;
    beta2Power = b2p;
    lr = l;
    weightDecay = wd;
    beta1 = b1;
    beta2 = b2;
    epsilon = eps;
    maximizeFactor = mf;
    
    varGm.SetGlobalBuffer((__gm__ T*)var, totalLength);
    mGm.SetGlobalBuffer((__gm__ T*)m, totalLength);
    vGm.SetGlobalBuffer((__gm__ T*)v, totalLength);
    gradGm.SetGlobalBuffer((__gm__ T*)grad, totalLength);
    varOutGm.SetGlobalBuffer((__gm__ T*)var_out, totalLength);
    mOutGm.SetGlobalBuffer((__gm__ T*)m_out, totalLength);
    vOutGm.SetGlobalBuffer((__gm__ T*)v_out, totalLength);
    
    uint32_t typeSize = sizeof(T);
    uint32_t totalBytes = tileLength * typeSize;
    
    pipe.InitBuffer(inQueue, BUFFER_NUM, totalBytes * 4);
    pipe.InitBuffer(outQueue, BUFFER_NUM, totalBytes * 3);
    pipe.InitBuffer(tmpBuf, tileLength * TMP_BUFFER_NUM * sizeof(float));
}

template <typename T>
__aicore__ inline void ApplyAdamwV3Kernel<T>::CopyIn(uint32_t loopIdx)
{
    uint32_t offset = coreOffset + loopIdx * tileLength;
    uint32_t currentLength = tileLength;
    
    if (offset + tileLength > coreOffset + elementsThisCore) {
        currentLength = coreOffset + elementsThisCore - offset;
    }
    
    LocalTensor<T> varLocal = inQueue.AllocTensor<T>();
    LocalTensor<T> mLocal = varLocal[tileLength];
    LocalTensor<T> vLocal = mLocal[tileLength];
    LocalTensor<T> gradLocal = vLocal[tileLength];
    
    DataCopy(varLocal, varGm[offset], currentLength);
    DataCopy(mLocal, mGm[offset], currentLength);
    DataCopy(vLocal, vGm[offset], currentLength);
    DataCopy(gradLocal, gradGm[offset], currentLength);
    
    inQueue.EnQue(varLocal);
}

template <typename T>
__aicore__ inline void ApplyAdamwV3Kernel<T>::Compute(uint32_t loopIdx)
{
    LocalTensor<T> inLocal = inQueue.DeQue<T>();
    LocalTensor<T> outLocal = outQueue.AllocTensor<T>();
    
    LocalTensor<T> varLocal = inLocal;
    LocalTensor<T> mLocal = inLocal[tileLength];
    LocalTensor<T> vLocal = inLocal[tileLength * 2];
    LocalTensor<T> gradLocal = inLocal[tileLength * 3];
    
    LocalTensor<T> varOutLocal = outLocal;
    LocalTensor<T> mOutLocal = outLocal[tileLength];
    LocalTensor<T> vOutLocal = outLocal[tileLength * 2];
    
    uint32_t currentLength = tileLength;
    uint32_t offset = coreOffset + loopIdx * tileLength;
    if (offset + tileLength > coreOffset + elementsThisCore) {
        currentLength = coreOffset + elementsThisCore - offset;
    }
    
    LocalTensor<float> tmpFloat = tmpBuf.Get<float>();
    LocalTensor<float> gtLocal = tmpFloat;
    LocalTensor<float> mOutFloat = tmpFloat[tileLength];
    LocalTensor<float> vOutFloat = tmpFloat[tileLength * 2];
    LocalTensor<float> varTFloat = tmpFloat[tileLength * 3];
    LocalTensor<float> denomFloat = tmpFloat[tileLength * 4];
    LocalTensor<float> varOutFloat = tmpFloat[tileLength * 5];
    LocalTensor<float> varFloat = tmpFloat[tileLength * 6];
    LocalTensor<float> mFloat = tmpFloat[tileLength * 7];
    LocalTensor<float> vFloat = tmpFloat[tileLength * 8];
    LocalTensor<float> gradFloat = tmpFloat[tileLength * 9];
    LocalTensor<float> tmp1 = tmpFloat[tileLength * 10];
    LocalTensor<float> tmp2 = tmpFloat[tileLength * 11];
    
    Cast(gradFloat, gradLocal, RoundMode::CAST_NONE, currentLength);
    Cast(mFloat, mLocal, RoundMode::CAST_NONE, currentLength);
    Cast(vFloat, vLocal, RoundMode::CAST_NONE, currentLength);
    Cast(varFloat, varLocal, RoundMode::CAST_NONE, currentLength);
    
    Muls(gtLocal, gradFloat, maximizeFactor, currentLength);
    
    Muls(tmp1, gtLocal, (beta1 - 1.0f), currentLength);
    Muls(tmp2, mFloat, beta1, currentLength);
    Sub(mOutFloat, tmp2, tmp1, currentLength);
    
    Mul(tmp1, gtLocal, gtLocal, currentLength);
    Muls(tmp1, tmp1, (beta2 - 1.0f), currentLength);
    Muls(tmp2, vFloat, beta2, currentLength);
    Sub(vOutFloat, tmp2, tmp1, currentLength);
    
    float varT_factor = 1.0f - lr * weightDecay;
    Muls(varTFloat, varFloat, varT_factor, currentLength);
    
    float beta2PowerOut = beta2Power * beta2;
    float denomDivisor = 1.0f / (beta2PowerOut - 1.0f);
    Muls(tmp1, vOutFloat, -1.0f * denomDivisor, currentLength);
    Sqrt(tmp2, tmp1, currentLength);
    Adds(denomFloat, tmp2, epsilon, currentLength);
    
    float beta1PowerOut = beta1Power * beta1;
    float stepSize = lr / (beta1PowerOut - 1.0f);
    Div(tmp1, mOutFloat, denomFloat, currentLength);
    Muls(tmp1, tmp1, stepSize, currentLength);
    Add(varOutFloat, varTFloat, tmp1, currentLength);
    
    Cast(varOutLocal, varOutFloat, RoundMode::CAST_RINT, currentLength);
    Cast(mOutLocal, mOutFloat, RoundMode::CAST_RINT, currentLength);
    Cast(vOutLocal, vOutFloat, RoundMode::CAST_RINT, currentLength);
    
    inQueue.FreeTensor(inLocal);
    outQueue.EnQue<T>(outLocal);
}

template <typename T>
__aicore__ inline void ApplyAdamwV3Kernel<T>::CopyOut(uint32_t loopIdx)
{
    LocalTensor<T> outLocal = outQueue.DeQue<T>();
    
    LocalTensor<T> varOutLocal = outLocal;
    LocalTensor<T> mOutLocal = outLocal[tileLength];
    LocalTensor<T> vOutLocal = outLocal[tileLength * 2];
    
    uint32_t offset = coreOffset + loopIdx * tileLength;
    uint32_t currentLength = tileLength;
    if (offset + tileLength > coreOffset + elementsThisCore) {
        currentLength = coreOffset + elementsThisCore - offset;
    }
    
    DataCopy(varOutGm[offset], varOutLocal, currentLength);
    DataCopy(mOutGm[offset], mOutLocal, currentLength);
    DataCopy(vOutGm[offset], vOutLocal, currentLength);
    
    outQueue.FreeTensor(outLocal);
}

template <typename T>
__aicore__ inline void ApplyAdamwV3Kernel<T>::Process()
{
    for (uint32_t i = 0; i < tileNum; i++) {
        CopyIn(i);
        Compute(i);
        CopyOut(i);
    }
}

}

#endif
