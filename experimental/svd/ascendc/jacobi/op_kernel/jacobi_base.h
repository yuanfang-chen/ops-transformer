/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file jacobi_base.h
 * \brief
 */

#ifndef JACOBI_BASE_H
#define JACOBI_BASE_H

#define DEBUG_MODE
#ifdef DEBUG_MODE

#define PRINT_VAL(X)   printf("Variable %s has value %d\n", #X, (int)X)
#else
#define PRINT_VAL(X)
#endif

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"

namespace SVD{

    template<typename T>
    constexpr __aicore__ T GET_MIN_BLOCK_SIZE()
    {
        constexpr uint32_t MIN_BLOCK_SIZE = 32;
        return static_cast<T>(MIN_BLOCK_SIZE);
    }

    template<typename SourceType, typename ResultType = uint8_t>
    constexpr __aicore__ uint8_t SMALL_MASK_NUM_ELEMENTS()
    {
        return static_cast<ResultType>(GET_MIN_BLOCK_SIZE<size_t>() / sizeof(SourceType));
    }


    template<typename T>
    constexpr __aicore__ T GET_MAX_BLOCK_SIZE()
    {
        constexpr uint32_t MIN_BLOCK_SIZE = 256;
        return static_cast<T>(MIN_BLOCK_SIZE);
    }

    template<typename SourceType, typename ResultType = uint8_t>
    constexpr __aicore__ uint8_t BIG_MASK_NUM_ELEMENTS()
    {
        return static_cast<ResultType>(GET_MAX_BLOCK_SIZE<size_t>() / sizeof(SourceType));
    }


using namespace AscendC;
template<typename AType, typename SType, typename UType, typename VType>
class JacobiBase {
public:

__aicore__ inline JacobiBase(TPipe* pipe)
{
    pipe_ = pipe;
}

__aicore__ inline void Init(__gm__ uint8_t* a, __gm__ uint8_t* s,
    __gm__ uint8_t* u, __gm__ uint8_t* v, __gm__ uint8_t* workspace,
    const JMTilingData* tilingData)
{
    coreIdx = GetBlockIdxImpl();
    needSort = true;
    numIterations = tilingData->nIterations;
    nSize = tilingData->nSize;
    mSize = tilingData->mSize;
    batchSize = tilingData->batchSize;
    nSizeAlignedUp32 = ((nSize + 31) / 32) * 32;

    aGm.SetGlobalBuffer((__gm__ AType*)a);
    aTmpGm.SetGlobalBuffer((__gm__ float*)workspace);
    sGm.SetGlobalBuffer((__gm__ SType*)s);
    uGm.SetGlobalBuffer((__gm__ UType*)u);
    vGm.SetGlobalBuffer((__gm__ VType*)v);
    pipe_->InitBuffer(ubBuf, 192 * 1024 * sizeof(uint8_t));
    ubMemory = ubBuf.Get<uint8_t>();
}

__aicore__ inline void Process()
{
    // printf("Num Cores is %i", GetBlockNum());
    if (coreIdx > 0) {
        printf("Core %i is dropped!!!", coreIdx);
        return;
    }

    printf("Core %i is working!!!!!", coreIdx);
    // PRINT_VAL(numIterations);
    // PRINT_VAL(batchSize);
    // PRINT_VAL(mSize);
    // PRINT_VAL(nSize);
    ProcessImpl();

}
private:
    __aicore__ inline void AllocMem() {
        uint32_t offsetPtr = 0;
        iSRow = ubBuf.GetWithOffset<float>(nSize, 0);
        iVRow = ubBuf.GetWithOffset<float>(nSize, nSize * sizeof(float));
        iSVRow = ubBuf.GetWithOffset<float>(2 * nSize, 0);
        offsetPtr += 2 * nSize * sizeof(float);

        jSRow = ubBuf.GetWithOffset<float>(nSize, offsetPtr);
        jVRow = ubBuf.GetWithOffset<float>(nSize, offsetPtr + nSize * sizeof(float));
        jSVRow = ubBuf.GetWithOffset<float>(2 * nSize, offsetPtr);
        offsetPtr += 2 * nSize * sizeof(float);

        tmpRow1 = ubBuf.GetWithOffset<float>(nSizeAlignedUp32, offsetPtr);
        tmpRow12 = ubBuf.GetWithOffset<float>(2 * nSizeAlignedUp32, offsetPtr);
        offsetPtr += nSizeAlignedUp32 * sizeof(float);

        tmpRow2 = ubBuf.GetWithOffset<float>(nSizeAlignedUp32, offsetPtr);
        offsetPtr += nSizeAlignedUp32 * sizeof(float);

        tmpRow3 = ubBuf.GetWithOffset<float>(nSizeAlignedUp32, offsetPtr);
        tmpRow34 = ubBuf.GetWithOffset<float>(2 * nSizeAlignedUp32, offsetPtr);
        offsetPtr += nSizeAlignedUp32 * sizeof(float);

        tmpRow4 = ubBuf.GetWithOffset<float>(nSizeAlignedUp32, offsetPtr);
        offsetPtr += nSizeAlignedUp32 * sizeof(float);

        //Small blocks for cos/sin vectorisation
        {
            constexpr uint8_t MIN_BLOCK_ELEMENTS_NUM = SMALL_MASK_NUM_ELEMENTS<float>();
            constexpr uint32_t MIN_BLOCK_SIZE = GET_MIN_BLOCK_SIZE<uint32_t>();
            tmpBlockSpace = ubBuf.GetWithOffset<float>(16 * MIN_BLOCK_ELEMENTS_NUM, offsetPtr);

            tmpBlock1 = ubBuf.GetWithOffset<float>(MIN_BLOCK_ELEMENTS_NUM, offsetPtr);
            offsetPtr += MIN_BLOCK_SIZE;

            tmpBlock2 = ubBuf.GetWithOffset<float>(MIN_BLOCK_ELEMENTS_NUM, offsetPtr);
            offsetPtr += MIN_BLOCK_SIZE;

            tmpBlock3 = ubBuf.GetWithOffset<float>(MIN_BLOCK_ELEMENTS_NUM, offsetPtr);
            offsetPtr += MIN_BLOCK_SIZE;

            tmpBlock4 = ubBuf.GetWithOffset<float>(MIN_BLOCK_ELEMENTS_NUM, offsetPtr);
            offsetPtr += MIN_BLOCK_SIZE;

            tmpBlock5 = ubBuf.GetWithOffset<float>(MIN_BLOCK_ELEMENTS_NUM, offsetPtr);
            offsetPtr += MIN_BLOCK_SIZE;

            tmpBlock6 = ubBuf.GetWithOffset<float>(MIN_BLOCK_ELEMENTS_NUM, offsetPtr);
            offsetPtr += MIN_BLOCK_SIZE;

            tmpBlock7 = ubBuf.GetWithOffset<float>(MIN_BLOCK_ELEMENTS_NUM, offsetPtr);
            offsetPtr += MIN_BLOCK_SIZE;

            tmpBlock8 = ubBuf.GetWithOffset<float>(MIN_BLOCK_ELEMENTS_NUM, offsetPtr);
            offsetPtr += MIN_BLOCK_SIZE;

            tmpBlock9 = ubBuf.GetWithOffset<float>(MIN_BLOCK_ELEMENTS_NUM, offsetPtr);
            offsetPtr += MIN_BLOCK_SIZE;

            tmpBlock10 = ubBuf.GetWithOffset<float>(MIN_BLOCK_ELEMENTS_NUM, offsetPtr);
            offsetPtr += MIN_BLOCK_SIZE;

            tmpBlock11 = ubBuf.GetWithOffset<float>(MIN_BLOCK_ELEMENTS_NUM, offsetPtr);
            offsetPtr += MIN_BLOCK_SIZE;

            tmpBlock12 = ubBuf.GetWithOffset<float>(MIN_BLOCK_ELEMENTS_NUM, offsetPtr);
            offsetPtr += MIN_BLOCK_SIZE;

            tmpBlock13 = ubBuf.GetWithOffset<float>(MIN_BLOCK_ELEMENTS_NUM, offsetPtr);
            offsetPtr += MIN_BLOCK_SIZE;

            tmpBlock14 = ubBuf.GetWithOffset<float>(MIN_BLOCK_ELEMENTS_NUM, offsetPtr);
            offsetPtr += MIN_BLOCK_SIZE;

            tmpBlock15 = ubBuf.GetWithOffset<float>(MIN_BLOCK_ELEMENTS_NUM, offsetPtr);
            offsetPtr += MIN_BLOCK_SIZE;

            tmpBlock16 = ubBuf.GetWithOffset<float>(MIN_BLOCK_ELEMENTS_NUM, offsetPtr);
            offsetPtr += MIN_BLOCK_SIZE;
        }
    }

    __aicore__ inline void ProcessImpl()
    {
        // printf("old_version");
        AllocMem();
        bool iIsFirstRotatateMat = true;
        bool jIsFirstRotatateMat = true;
        for (uint16_t iter = 0; iter < numIterations; ++iter) {
            printVal = iter;
            // printf("************Iteration:%i *******************\n", int(iter));
            for (uint16_t i = 0; i < mSize - 1; ++i) {
                for (uint16_t j = i + 1; j < mSize; ++j) {
                    RotateMat(i, j, iIsFirstRotatateMat, jIsFirstRotatateMat);
                    iIsFirstRotatateMat = false;
                    // PipeBarrier<PIPE_ALL>();
                }
                jIsFirstRotatateMat = false;
                // PipeBarrier<PIPE_ALL>();
            }
            // break;
            // PipeBarrier<PIPE_ALL>();
        }
        PipeBarrier<PIPE_ALL>();
        GetDiagonal(aTmpGm);

    }

    __aicore__ inline void RotateMat(uint16_t i, uint16_t j, bool iIsFirstRotate, bool jIsFirstRotate)
    {
        constexpr uint8_t SMNE = SMALL_MASK_NUM_ELEMENTS<float>();
        constexpr uint8_t BMNE = BIG_MASK_NUM_ELEMENTS<float>();
        // LocalTensor<float>& normSquareDiffIJ = tmpBlock1;
        // LocalTensor<float>& dotProductIJ = tmpBlock2;
        // LocalTensor<float>& tauTensor = tmpBlock3;
        // LocalTensor<uint8_t> normSquareDiffIsPositive = tmpBlock4.ReinterpretCast<uint8_t>();
        // LocalTensor<uint8_t> isNonRotateCase = tmpBlock5.ReinterpretCast<uint8_t>();
        uint16_t oneOffsetBlockI = 0;
        uint16_t oneOffsetBlockJ = 0;
        uint16_t localBlockSize = nSize;
        if (nSize > BMNE) {
            oneOffsetBlockI = (i / BMNE) * BMNE;
            oneOffsetBlockJ = (j / BMNE) * BMNE;
            localBlockSize = BMNE;
        }


        if (!iIsFirstRotate) {
            DataCopy(iSVRow, aTmpGm[i * 2 * nSize], { 1, static_cast<uint16_t>(2 * nSize * sizeof(AType) / 32), 0,0 });
        } else {
            uint64_t localMaskI = 1;
            localMaskI <<= (i % localBlockSize);
            uint64_t mask[2] = { localMaskI, 0 };
            DataCopy(iSRow, aGm[i * nSize], { 1, static_cast<uint16_t>(nSize * sizeof(AType) / 32), 0,0 });
            event_t evtMte2ToV2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::MTE2_V>(evtMte2ToV2);
            WaitFlag<HardEvent::MTE2_V>(evtMte2ToV2);
            Duplicate(iVRow, 0.f, nSize);
            PipeBarrier<PIPE_V>();
            Duplicate(iVRow[oneOffsetBlockI], 1.f, mask, 1, 1, 8);

        }
        if (!jIsFirstRotate) {
            DataCopy(jSVRow, aTmpGm[j * 2 * nSize], { 1, static_cast<uint16_t>(2 * nSize * sizeof(AType) / 32), 0,0 });

        } else {
            uint64_t localMaskJ = 1;
            localMaskJ <<= (j % localBlockSize);

            uint64_t mask[2] = { localMaskJ, 0 };
            DataCopy(jSRow, aGm[j * nSize], { 1, static_cast<uint16_t>(nSize * sizeof(AType) / 32), 0,0 });
            event_t evtMte2ToV2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::MTE2_V>(evtMte2ToV2);
            WaitFlag<HardEvent::MTE2_V>(evtMte2ToV2);
            Duplicate(jVRow, 0.f, nSize);
            PipeBarrier<PIPE_V>();
            Duplicate(jVRow[oneOffsetBlockJ], 1.f, mask, 1, 1, 8);
        }

        LocalTensor<float>& leftSquareNorms = tmpBlock9;
        LocalTensor<float>& rightSquareNorms = tmpBlock10;
        LocalTensor<float>& dotProducts = tmpBlock11;
        event_t evtMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(evtMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(evtMte2ToV);
        // PipeBarrier<PIPE_ALL>();
        // if(printVal==3){
        //     printf("iSRow(%i) before {%f, %f, %f, %f, ...}\n", i, iSRow.GetValue(0), iSRow.GetValue(1), iSRow.GetValue(2),iSRow.GetValue(3));
        //     printf("jSRow(%i) before {%f, %f, %f, %f, ...}\n", j, jSRow.GetValue(0), jSRow.GetValue(1), jSRow.GetValue(2),jSRow.GetValue(3));
        // }
        // PipeBarrier<PIPE_ALL>();
        PipeBarrier<PIPE_V>();
        Mul(tmpRow1, iSRow, iSRow, nSize);
        Mul(tmpRow2, jSRow, jSRow, nSize);
        Mul(tmpRow3, iSRow, jSRow, nSize);
        Duplicate(tmpRow4, 0.f, nSize);
        PipeBarrier<PIPE_V>();
        // Sub(tmpRow1, tmpRow2, tmpRow1, nSize);
        ReduceSum(leftSquareNorms, tmpRow1, tmpRow4, nSize);
        PipeBarrier<PIPE_V>();
        ReduceSum(rightSquareNorms, tmpRow2, tmpRow4, nSize);
        PipeBarrier<PIPE_V>();
        ReduceSum(dotProducts, tmpRow3, tmpRow4, nSize);
        PipeBarrier<PIPE_V>();


        LocalTensor<float>& leftNorms = tmpBlock9;// after calculation sin and cos leftSquareNorms and
        //rightSquareNorms is useles, so its will be update to typical norms
        LocalTensor<float>& rightNorms = tmpBlock10;


        LocalTensor<float>& sinTensor = tmpBlock12;
        LocalTensor<float>& cosTensor = tmpBlock13;
        LocalTensor<float>& negSinTensor = tmpBlock14;

        __ubuf__ float* sinTensorPtr = (__ubuf__ float*)sinTensor.GetPhyAddr();
        __ubuf__ float* cosTensorPtr = (__ubuf__ float*)cosTensor.GetPhyAddr();
        __ubuf__ float* negSinTensorPtr = (__ubuf__ float*)negSinTensor.GetPhyAddr();
        // if(printVal==3){
        //     PipeBarrier<PIPE_ALL>();
        //     printf("(%i, %i) leftSquareNorms is %f, rightSquareNorms is %f, dotProduct is %f\n", i, j, leftSquareNorms.GetValue(0),
        //                             rightSquareNorms.GetValue(0), dotProducts.GetValue(0));
        //     PipeBarrier<PIPE_ALL>();
        // }
        // badCase=(printVal==3)&&(i==6&&j==7);
        CalculateSinCos(leftSquareNorms, rightSquareNorms, dotProducts, sinTensor, cosTensor, negSinTensor);
        event_t evtVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(evtVToS);
        WaitFlag<HardEvent::V_S>(evtVToS);
        float sin = sinTensor.GetValue(0);
        float cos = cosTensor.GetValue(0);
        uint32_t sini32 = *((int32_t*)&sin);
        uint32_t cosi32 = *((int32_t*)&cos);
        event_t evtSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(evtSToV);
        WaitFlag<HardEvent::S_V>(evtSToV);
        PipeBarrier<PIPE_V>();
        Muls(tmpRow1, iSRow, cos, nSize);
        Muls(tmpRow2, jSRow, -sin, nSize);
        Muls(tmpRow3, iSRow, sin, nSize);
        Muls(tmpRow4, jSRow, cos, nSize);
        PipeBarrier<PIPE_V>();
        Add(iSRow, tmpRow1, tmpRow2, nSize);
        Add(jSRow, tmpRow3, tmpRow4, nSize);
        PipeBarrier<PIPE_V>();
        Muls(tmpRow1, iVRow, cos, nSize);
        Muls(tmpRow2, jVRow, -sin, nSize);
        Muls(tmpRow3, iVRow, sin, nSize);
        Muls(tmpRow4, jVRow, cos, nSize);
        PipeBarrier<PIPE_V>();
        Add(iVRow, tmpRow1, tmpRow2, nSize);
        Add(jVRow, tmpRow3, tmpRow4, nSize);
        PipeBarrier<PIPE_V>();
        event_t evtVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(evtVToMte3);
        WaitFlag<HardEvent::V_MTE3>(evtVToMte3);
        DataCopy(aTmpGm[i * 2 * nSize], iSVRow, { 1, static_cast<uint16_t>(2 * nSize * sizeof(AType) / 32), 0,0 });
        DataCopy(aTmpGm[j * 2 * nSize], jSVRow, { 1, static_cast<uint16_t>(2 * nSize * sizeof(AType) / 32), 0,0 });

        event_t evtMte3ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(evtMte3ToMte2);
        WaitFlag<HardEvent::MTE3_MTE2>(evtMte3ToMte2);
        event_t evtMte3ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(evtMte3ToV);
        WaitFlag<HardEvent::MTE3_V>(evtMte3ToV);
    }

    __aicore__ inline void GetDiagonal(const GlobalTensor<AType>& source)
    {
        //ToDo: In future delete S_ and V_S synchronisation

        LocalTensor<float>& squareNorms = tmpRow1;
        LocalTensor<float>& singularValues = tmpRow2;
        for (int i = 0; i < nSize; ++i) {
            DataCopy(iSVRow, source[2 * i * nSize], { 1, static_cast<uint16_t>(2 * nSize * sizeof(AType) / 32), 0,0 });

            event_t evtMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::MTE2_V>(evtMte2ToV);
            WaitFlag<HardEvent::MTE2_V>(evtMte2ToV);
            Mul(jSRow, iSRow, iSRow, nSize);
            PipeBarrier<PIPE_V>();
            ReduceSum(squareNorms, jSRow, tmpRow3, nSize);
            PipeBarrier<PIPE_V>();
            Muls(jSRow, iSRow, 1.f / sqrt(squareNorms.GetValue(0)), nSize);
            event_t evtVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
            SetFlag<HardEvent::V_S>(evtVToS);
            WaitFlag<HardEvent::V_S>(evtVToS);
            singularValues.SetValue(i, squareNorms.GetValue(0));
            event_t evtSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
            SetFlag<HardEvent::S_V>(evtSToV);
            WaitFlag<HardEvent::S_V>(evtSToV);

            event_t evtSToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
            SetFlag<HardEvent::S_MTE3>(evtSToMte3);
            WaitFlag<HardEvent::S_MTE3>(evtSToMte3);
            DataCopy(uGm[i * nSize], iVRow, { 1, static_cast<uint16_t>(nSize * sizeof(AType) / 32), 0,0 });
            DataCopy(vGm[i * nSize], jSRow, { 1, static_cast<uint16_t>(nSize * sizeof(AType) / 32), 0,0 });
            event_t evtMte3ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
            SetFlag<HardEvent::MTE3_MTE2>(evtMte3ToMte2);
            WaitFlag<HardEvent::MTE3_MTE2>(evtMte3ToMte2);

            // event_t evtVToMte2= static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
            // SetFlag<HardEvent::V_MTE2>(evtVToMte2);
            // WaitFlag<HardEvent::V_MTE2>(evtVToMte2);
        }
        Sqrt(singularValues, singularValues, nSize);
        event_t evtVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(evtVToMte3);
        WaitFlag<HardEvent::V_MTE3>(evtVToMte3);

        DataCopy(sGm, singularValues, { 1, static_cast<uint16_t>(nSize * sizeof(AType) / 32), 0,0 });
    }

    __aicore__ inline void CalculateSinCos(const LocalTensor<float>& leftSquareNorms, const LocalTensor<float>& rightSquareNorms,
        const LocalTensor<float>& dotProducts, const LocalTensor<float>& sinTensor,
        const LocalTensor<float>& cosTensor, const LocalTensor<float>& negSinTensor)
    {
        constexpr uint8_t SMNE = SMALL_MASK_NUM_ELEMENTS<float>();
        //Generating coeffs for rotating
        // tmpBlock1-8 used for tmp calculatins

        //tauTensor used for calculation tTensor
        LocalTensor<float>& tauTensor = tmpBlock1;
        //normSquareDiffIsPositive ensures  that sin and cos are selected so that the singular
        // values are sorted in descending order. this trick need to avoid  to sort outputs.
        LocalTensor<uint8_t> normSquareDiffIsPositive = tmpBlock2.ReinterpretCast<uint8_t>();
        // The vaules of  isFakeRotateCase indicate whether the vactors are orthogonal or not.
        // If vectors are orthogonal a fake rotation is performed
        LocalTensor<uint8_t> isFakeRotateCase = tmpBlock3.ReinterpretCast<uint8_t>();
        //Free tmpBlocks are {4,5,6,7,8}
        {
            LocalTensor<float>& normSquareDiff = tmpBlock4;
            LocalTensor<float>& constValues = tmpBlock5;
            LocalTensor<float>& absDotProducts = tmpBlock6;
            Duplicate(constValues, 0.f, SMNE);

            Sub(normSquareDiff, rightSquareNorms, leftSquareNorms, SMNE);
            PipeBarrier<PIPE_V>();
            // Compare(normSquareDiffIsPositive, normSquareDiff, constValues, CMPMODE::GE, SMNE);
            VcmpvImpl((__ubuf__ uint8_t*)normSquareDiffIsPositive.GetPhyAddr(), (__ubuf__ float*)normSquareDiff.GetPhyAddr(), (__ubuf__ float*)constValues.GetPhyAddr(),
                CMPMODE::GE, SMNE, 1, { 1,1,1, 1,1,1 });
            Div(tauTensor, normSquareDiff, dotProducts, SMNE);
            PipeBarrier<PIPE_V>();

            PipeBarrier<PIPE_V>();
            Muls(tauTensor, tauTensor, 0.5f, SMNE);
            PipeBarrier<PIPE_V>();
            Abs(normSquareDiff, normSquareDiff, SMNE);
            PipeBarrier<PIPE_V>();
            Abs(absDotProducts, dotProducts, SMNE);
            PipeBarrier<PIPE_V>();
            Duplicate(constValues, NEAR_ZERO_VALUE, SMNE);
            PipeBarrier<PIPE_V>();
            LocalTensor<uint8_t>& normSquareDiffIsZero = isFakeRotateCase;
            LocalTensor<uint8_t> dotProductIsZero = tmpBlock7.ReinterpretCast<uint8_t>();
            VcmpvImpl((__ubuf__ uint8_t*)normSquareDiffIsZero.GetPhyAddr(), (__ubuf__ float*)normSquareDiff.GetPhyAddr(), (__ubuf__ float*)constValues.GetPhyAddr(),
                CMPMODE::LT, SMNE, 1, { 1,1,1, 1,1,1 });
            PipeBarrier<PIPE_V>();
            VcmpvImpl((__ubuf__ uint8_t*)dotProductIsZero.GetPhyAddr(), (__ubuf__ float*)absDotProducts.GetPhyAddr(), (__ubuf__ float*)constValues.GetPhyAddr(),
                CMPMODE::LT, SMNE, 1, { 1,1,1, 1,1,1 });
            PipeBarrier<PIPE_V>();
            Or(isFakeRotateCase, normSquareDiffIsZero, dotProductIsZero, SMNE);
            PipeBarrier<PIPE_V>();
        }
        LocalTensor<float>& tTensor = tmpBlock4;
        //Free tmpBlocks are {5,6,7,8}
        //Calculating t = sign(tau)/(absTau+sqrt(1.f+tau*tau));
        {
            PipeBarrier<PIPE_V>();
            LocalTensor<float>& absTaoTensor = tmpBlock5;
            //absTau = abs(tau)
            Abs(absTaoTensor, tauTensor, SMNE);
            LocalTensor<float>& tmp1 = tmpBlock6;
            LocalTensor<float>& tmp2 = tmpBlock7;
            PipeBarrier<PIPE_V>();
            // tmp1=tau*tau
            Mul(tmp1, tauTensor, tauTensor, SMNE);
            PipeBarrier<PIPE_V>();
            //tmp1 = 1.f+tau*tau
            Adds(tmp1, tmp1, 1.f, SMNE);
            PipeBarrier<PIPE_V>();
            //tmp2 = sign(tau)=tau/absTau
            Div(tmp2, tauTensor, absTaoTensor, SMNE);
            PipeBarrier<PIPE_V>();
            //tmp1 = sqrt(1.f+tau*tau)
            Sqrt(tmp1, tmp1, SMNE);
            PipeBarrier<PIPE_V>();
            //tmp1 = absTau+sqrt(1.f+tau*tau)
            Add(tmp1, tmp1, absTaoTensor, SMNE);
            PipeBarrier<PIPE_V>();
            // t = sign(tau)/(absTau+sqrt(1.f+tau*tau))
            Div(tTensor, tmp2, tmp1, SMNE);
            PipeBarrier<PIPE_V>();
        }
        //taoTensor is free and can be reused
        LocalTensor<float>& s1Tensor = tmpBlock1;
        LocalTensor<float>& s2Tensor = tmpBlock5;
        //Free tmpBlocks are {6,7,8}
        {
            LocalTensor<float>& negativeOneTensor = tmpBlock6;
            Duplicate(negativeOneTensor, -1.f, SMNE);
            //s1 = 1 / sqrt(1.f + t*t)
            //s1 = t*t
            PipeBarrier<PIPE_V>();
            Mul(s1Tensor, tTensor, tTensor, SMNE);
            PipeBarrier<PIPE_V>();
            Duplicate(s2Tensor, 1.f, SMNE);
            PipeBarrier<PIPE_V>();
            //s1 = 1.f + t*t
            Add(s1Tensor, s2Tensor, s1Tensor, SMNE);
            PipeBarrier<PIPE_V>();
            //s1 = sqrt(1.f + t*t)
            Sqrt(s1Tensor, s1Tensor, SMNE);
            PipeBarrier<PIPE_V>();
            //s1 = 1 / sqrt(1.f + t*t)
            Div(s1Tensor, s2Tensor, s1Tensor, SMNE);
            PipeBarrier<PIPE_V>();
            //s2 = t*s1
            Mul(s2Tensor, s1Tensor, tTensor, SMNE);
            PipeBarrier<PIPE_V>();
            //tTensor is free
            //  if(normSquareDiff>=0.f) s2=-s2;
            LocalTensor<float>& coeff = tmpBlock4;
            Select(coeff, normSquareDiffIsPositive, negativeOneTensor, 1.f, SELMODE::VSEL_TENSOR_SCALAR_MODE, SMNE);
            PipeBarrier<PIPE_V>();
            Mul(s2Tensor, s2Tensor, coeff, SMNE);
            PipeBarrier<PIPE_V>();
        }
        // Calculate sin, cos, -sin
        // Free tmpBlocks are {8}
        {
            LocalTensor<float>& onesTensor = tmpBlock8;

            Duplicate(onesTensor, 1.f, SMNE);
            Select(sinTensor, normSquareDiffIsPositive, s1Tensor, s2Tensor, SELMODE::VSEL_CMPMASK_SPR, SMNE);
            PipeBarrier<PIPE_V>();
            Select(cosTensor, normSquareDiffIsPositive, s2Tensor, s1Tensor, SELMODE::VSEL_CMPMASK_SPR, SMNE);
            PipeBarrier<PIPE_V>();
            // Free s1Tensor(tmpBlock1), s2Tensor(tmpBlock5), normSquareDiffIsPositive(tmpBlock2),
            LocalTensor<float>& zerosTensor = tmpBlock1;
            Duplicate(zerosTensor, 0.f, SMNE);
            Select(cosTensor, isFakeRotateCase, onesTensor, cosTensor, SELMODE::VSEL_CMPMASK_SPR, SMNE);
            PipeBarrier<PIPE_V>();
            Select(sinTensor, isFakeRotateCase, zerosTensor, sinTensor, SELMODE::VSEL_CMPMASK_SPR, SMNE);
            PipeBarrier<PIPE_V>();
            Muls(negSinTensor, sinTensor, -1.f, SMNE);
            PipeBarrier<PIPE_V>();
        }
    }

private:
    static constexpr float NEAR_ZERO_VALUE = 0.00001f;
private:
    TPipe* pipe_;
    GlobalTensor<AType> aGm;
    GlobalTensor<AType> aTmpGm;
    GlobalTensor<AType> sGm;
    GlobalTensor<AType> uGm;
    GlobalTensor<AType> vGm;
    uint32_t coreIdx;
    uint32_t numIterations;
    uint32_t mSize, nSize, batchSize, nSizeAlignedUp32;
    uint32_t rowSize, printVal;
    bool badCase = false;
    TBuf<TPosition::VECCALC> ubBuf;
    LocalTensor<uint8_t> ubMemory;
    LocalTensor<float> iSVRow, jSVRow, tmpRow1, tmpRow2, tmpRow3, tmpRow4, iSRow, jSRow, iVRow, jVRow, tmpRow12, tmpRow34,  testedSorting1024Tmp1, testedSorting1024Tmp2, testedSorting1024Tmp3;
    //Constant tensors
    LocalTensor<float> tmpBlock1, tmpBlock2, tmpBlock3, tmpBlock4, tmpBlock5,
                       tmpBlock6, tmpBlock7, tmpBlock8, tmpBlock9, tmpBlock10, tmpBlock11, tmpBlock12, tmpBlock13,
                       tmpBlock14, tmpBlock15, tmpBlock16, tmpBlockSpace;
    bool needSort;
};
}
#endif //JACOBI_BASE_H
