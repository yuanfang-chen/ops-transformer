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
 * \file nsa_selected_attention_infer_impl.h
 * \brief
 */

#pragma once

#include "nsa_selected_attention_infer.h"

template <typename NSAT>
__aicore__ inline void
NsaSelectAttentionInfer<NSAT>::Bmm2DataCopyOut(uint64_t attenOutOffset, LocalTensor<OUT_T> &attenOutUb,
    uint32_t startRow, uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = dealRowCount;
    dataCopyParams.blockLen = actualColumnCount * sizeof(OUT_T);
    dataCopyParams.srcStride = (columnCount - actualColumnCount) / (BYTE_BLOCK / sizeof(OUT_T));
    dataCopyParams.dstStride = 0;
    DataCopyPad(attentionOutGm[attenOutOffset + (gSizeStart + startRow) * actualColumnCount], attenOutUb, dataCopyParams);
}

template <typename NSAT>
__aicore__ inline void
NsaSelectAttentionInfer<NSAT>::Bmm2CastAndCopyOut(ExtraInfo& info, LocalTensor<T> &bmm2ResUb, uint32_t startRow,
                                                                uint32_t dealRowCount, uint32_t columnCount,
                                                                uint32_t actualColumnCount)
{
    LocalTensor<OUT_T> tmpBmm2ResCastTensor = outputQue1.AllocTensor<OUT_T>();
    if constexpr (IsSameType<OUT_T, bfloat16_t>::value) { // bf16 采取四舍六入五成双模式
        Cast(tmpBmm2ResCastTensor, bmm2ResUb, AscendC::RoundMode::CAST_RINT, dealRowCount * columnCount);
    } else {
        Cast(tmpBmm2ResCastTensor, bmm2ResUb, AscendC::RoundMode::CAST_ROUND, dealRowCount * columnCount);
    }
    outputQue1.EnQue(tmpBmm2ResCastTensor);
    outputQue1.DeQue<OUT_T>();
    Bmm2DataCopyOut(info.attenOutOffset, tmpBmm2ResCastTensor, startRow, dealRowCount, columnCount, actualColumnCount);
    outputQue1.FreeTensor(tmpBmm2ResCastTensor);
}

template <typename NSAT>
__aicore__ inline void NsaSelectAttentionInfer<NSAT>::CopyInMm1AToL1(LocalTensor<KV_T> &l1Tensor, ExtraInfo& info)
{
    uint32_t mmRowCount = msdIterNum * gSize;
    uint32_t copyIterNum = (mmRowCount + 15) / 16;
    uint32_t copyStride = 16 * headDimAlign;
    for(int i = 0; i < copyIterNum; i++){
        Nd2NzParams mm1Nd2NzParamsForA;
        mm1Nd2NzParamsForA.ndNum = 1; // ND矩阵的个数
        if(i == copyIterNum - 1) {
            mm1Nd2NzParamsForA.nValue = msdIterNum * gSize - i * 16;
        }
        else {
            mm1Nd2NzParamsForA.nValue = 16; // 单个ND矩阵的行数, 单位为元素个数  16
        }
        mm1Nd2NzParamsForA.dValue = headDimAlign; // 单个ND矩阵的列数, 单位为元素个数   128
        mm1Nd2NzParamsForA.srcDValue = headDimAlign; // 同一个ND矩阵中相邻行起始地址之间的偏移, 单位为元素个数  128
        mm1Nd2NzParamsForA.srcNdMatrixStride = 0; // 相邻ND矩阵起始地址之间的偏移, 单位为元素个数
        mm1Nd2NzParamsForA.dstNzC0Stride = 16; // 转换为NZ矩阵后，相邻Block起始地址之间的偏移, 单位为Block个数  16
        mm1Nd2NzParamsForA.dstNzNStride = 1; // 转换为NZ矩阵后，ND中之前相邻两行在NZ矩阵中起始地址之间的偏移
        mm1Nd2NzParamsForA.dstNzMatrixStride = 0; // 两个NZ矩阵，起始地址之间的偏移   
        DataCopy(l1Tensor[i * copyStride], queryGm[info.tensorAOffset + i * copyStride], mm1Nd2NzParamsForA);
    }
}

template <typename NSAT>
__aicore__ inline void NsaSelectAttentionInfer<NSAT>::CopyInMm1BToL1ForPA(
    LocalTensor<KV_T>& bL1Tensor, uint64_t keyGmBaseOffset, uint32_t copyTotalRowCnt, uint32_t copyStartRowCnt, uint32_t nActCopyRowCount)
{
    uint32_t baseN = 256 / sizeof(KV_T);
    uint64_t step = headDim;
    step = headDim * kvHeadNum;
    uint32_t nHead = 0;
    uint32_t nTail = 0;
    uint32_t midNdNum = 0;
    uint32_t copyStartRowCntInBaseN = copyStartRowCnt % baseN;
    uint32_t copyStartRowCntInBaseNTail = 0;
    if (copyStartRowCntInBaseN + nActCopyRowCount <= baseN) {
        nHead = 0;
        midNdNum = 0;
        nTail = nActCopyRowCount;
        copyStartRowCntInBaseNTail = copyStartRowCntInBaseN;
    } else {
        if (copyStartRowCntInBaseN == 0) {
            nHead = 0;
        } else {
            nHead = baseN - copyStartRowCntInBaseN;
        }
        midNdNum = (nActCopyRowCount - nHead) / baseN;
        nTail = (nActCopyRowCount - nHead) % baseN;
        copyStartRowCntInBaseNTail = 0;
    }

    uint32_t dstNzC0StrideTail = baseN;
    if (copyTotalRowCnt % baseN != 0) {
        if ((copyStartRowCnt + nActCopyRowCount) > (copyTotalRowCnt / baseN * baseN)) {
            dstNzC0StrideTail = copyTotalRowCnt % baseN;
        }
    }
    Nd2NzParams mm1Nd2NzParamsForB;
    mm1Nd2NzParamsForB.dValue = headDim;
    mm1Nd2NzParamsForB.srcDValue = step; // 同一个ND矩阵中相邻行起始地址之间的偏移, 单位为元素个数
    mm1Nd2NzParamsForB.dstNzC0Stride = baseN; // 转换为NZ矩阵后，相邻Block起始地址之间的偏移, 单位为Block个数 
    mm1Nd2NzParamsForB.dstNzNStride = 1; // 转换为NZ矩阵后，ND中之前相邻两行在NZ矩阵中起始地址之间的偏移, 单位为Block个数
    if (nHead != 0) {
        mm1Nd2NzParamsForB.ndNum = 1;
        mm1Nd2NzParamsForB.nValue = nHead; // 单个ND矩阵的行数, 单位为元素个数
        mm1Nd2NzParamsForB.srcNdMatrixStride = 0; // 相邻ND矩阵起始地址之间的偏移, 单位为元素个数
        mm1Nd2NzParamsForB.dstNzMatrixStride = 0; // 两个NZ矩阵，起始地址之间的偏移, 单位为元素个数

        uint32_t ndNumFinish = copyStartRowCnt / baseN;
        DataCopy(bL1Tensor[ndNumFinish * baseN * headDimAlign + copyStartRowCntInBaseN * (32 / sizeof(KV_T))], keyGm[keyGmBaseOffset],
                mm1Nd2NzParamsForB);
    }

    if (midNdNum != 0) {
        mm1Nd2NzParamsForB.nValue = baseN; // 单个ND矩阵的行数, 单位为元素个数   256
        mm1Nd2NzParamsForB.dValue = headDim; // 单个ND矩阵的列数, 单位为元素个数   128
        mm1Nd2NzParamsForB.srcDValue = step; // 同一个ND矩阵中相邻行起始地址之间的偏移, 单位为元素个数  // 128
        mm1Nd2NzParamsForB.dstNzC0Stride = baseN;// 转换为NZ矩阵后，相邻Block起始地址之间的偏移, 单位为Block个数   256
        mm1Nd2NzParamsForB.dstNzNStride = 1; // 转换为NZ矩阵后，ND中之前相邻两行在NZ矩阵中起始地址之间的偏移, 单位为Block个数
        int32_t ndNumFinish = (copyStartRowCnt + nHead) / baseN;
        if ((LAYOUT_T == LAYOUT::BSH || LAYOUT_T == LAYOUT::BSND || LAYOUT_T == LAYOUT::TND) && (baseN * step > 65535)) {
            mm1Nd2NzParamsForB.ndNum = 1;
            mm1Nd2NzParamsForB.srcNdMatrixStride = 0; // 相邻ND矩阵起始地址之间的偏移, 单位为元素个数 256*128
            mm1Nd2NzParamsForB.dstNzMatrixStride = 0; // 两个NZ矩阵，起始地址之间的偏移, 单位为元素个数 256*128
            for (uint32_t i = 0; i < midNdNum; i++) {
                DataCopy(bL1Tensor[(ndNumFinish + i) * baseN * headDimAlign], keyGm[keyGmBaseOffset + nHead * step + i * baseN * step], mm1Nd2NzParamsForB);
            }
        } else {
            mm1Nd2NzParamsForB.ndNum = midNdNum;
            mm1Nd2NzParamsForB.srcNdMatrixStride = baseN * step; // 相邻ND矩阵起始地址之间的偏移, 单位为元素个数 256*128
            mm1Nd2NzParamsForB.dstNzMatrixStride = baseN * headDimAlign; // 两个NZ矩阵，起始地址之间的偏移, 单位为元素个数 256*128
            DataCopy(bL1Tensor[ndNumFinish * baseN * headDimAlign], keyGm[keyGmBaseOffset + nHead * step], mm1Nd2NzParamsForB);
        }
    }
    if (nTail != 0) {
        mm1Nd2NzParamsForB.ndNum = 1;
        mm1Nd2NzParamsForB.nValue = nTail; // 单个ND矩阵的行数, 单位为元素个数   actualSingleProcessSInnerSizeAlign为32对齐，nTail已经是32对齐
        mm1Nd2NzParamsForB.dValue = headDim; // 单个ND矩阵的列数, 单位为元素个数   128
        mm1Nd2NzParamsForB.srcNdMatrixStride = 0; // 相邻ND矩阵起始地址之间的偏移, 单位为元素个数 0
        mm1Nd2NzParamsForB.srcDValue = step; // 同一个ND矩阵中相邻行起始地址之间的偏移, 单位为元素个数  // 128
        mm1Nd2NzParamsForB.dstNzC0Stride = dstNzC0StrideTail;// 转换为NZ矩阵后，相邻Block起始地址之间的偏移, 单位为Block个数 
        mm1Nd2NzParamsForB.dstNzNStride = 1; // 转换为NZ矩阵后，ND中之前相邻两行在NZ矩阵中起始地址之间的偏移, 单位为Block个数
        mm1Nd2NzParamsForB.dstNzMatrixStride = 0; // 两个NZ矩阵，起始地址之间的偏移, 单位为元素个数  0

        int32_t ndNumFinish = (copyStartRowCnt + nHead) / baseN + midNdNum;
        DataCopy(bL1Tensor[ndNumFinish * baseN * headDimAlign + copyStartRowCntInBaseNTail * (32 / sizeof(KV_T))], keyGm[keyGmBaseOffset + (nHead + midNdNum * baseN) * step],
                mm1Nd2NzParamsForB);  //需要调整偏移地址，bL1Tensor的偏移， keyGm的偏移
    }
}

template <typename NSAT>
__aicore__ inline void NsaSelectAttentionInfer<NSAT>::LoadDataMm1A(LocalTensor<KV_T>& aL0Tensor,
                        LocalTensor<KV_T>& aL1Tensor, uint32_t k, uint32_t copyColCnt)
{
    uint32_t mmRowCount = msdIterNum * gSize;
    uint32_t copyIterNum = (mmRowCount + 15) / 16;
    for(int i = 0; i < copyIterNum; i++) {
        LoadData2DParams loadData2DParams;
        loadData2DParams.startIndex = 0;
        loadData2DParams.repeatTimes = 16 / 16 * copyColCnt / (32 / sizeof(KV_T)); // copyColCnt:128/64
        loadData2DParams.srcStride = 1;
        loadData2DParams.dstGap = 0;
        loadData2DParams.ifTranspose = false;
        LoadData(aL0Tensor[i * 16 * copyColCnt], aL1Tensor[16 * i * 192 + 16 * 128 * k], loadData2DParams);
    }
}

template <typename NSAT>
__aicore__ inline void NsaSelectAttentionInfer<NSAT>::LoadDataMm1B(LocalTensor<KV_T> &bl0Tensor,
    LocalTensor<KV_T> &bl1Tensor,  uint32_t k, uint32_t copyColCnt, uint32_t i, uint32_t copyRowCnt)
{
    uint32_t blockElementCnt = 32 / sizeof(KV_T);
    LoadData2DParams loadData2DParamsForB;
    loadData2DParamsForB.startIndex = 0;
    loadData2DParamsForB.srcStride = 1;
    loadData2DParamsForB.dstGap = 0;
    loadData2DParamsForB.ifTranspose = false;
    loadData2DParamsForB.repeatTimes = (copyRowCnt / 16) * (copyColCnt / blockElementCnt);
    LoadData(bl0Tensor, bl1Tensor[copyRowCnt * 128 * k], loadData2DParamsForB);
}

template <typename NSAT>
__aicore__ inline void NsaSelectAttentionInfer<NSAT>::CopyInMm2AToL1(LocalTensor<KV_T>& aL1Tensor, ExtraInfo& info, uint32_t kCopyIdx,
        uint32_t kCopyRowCount, uint32_t kActCopyRowCountAlign, uint32_t kActCopyRowCount) {
    uint32_t mmRowCount = msdIterNum * gSize;
    uint32_t copyStrideL1 = 16 * kActCopyRowCountAlign;
    uint32_t copyStrideGm = 16 * info.actualSingleProcessSInnerSizeAlign;
    uint32_t copyIterNum = (mmRowCount + 15) / 16;
    for(int i = 0; i < copyIterNum; i++){
        Nd2NzParams mm1Nd2NzParamsForA;
        mm1Nd2NzParamsForA.ndNum = 1; // ND矩阵的个数
        if(i == copyIterNum - 1) {
            mm1Nd2NzParamsForA.nValue = msdIterNum * gSize - i * 16;
        }
        else {
            mm1Nd2NzParamsForA.nValue = 16; // 单个ND矩阵的行数, 单位为元素个数  16
        }
        mm1Nd2NzParamsForA.dValue = kActCopyRowCount; // 单个ND矩阵的列数, 单位为元素个数
        mm1Nd2NzParamsForA.srcNdMatrixStride = 0; // 相邻ND矩阵起始地址之间的偏移, 单位为元素个数
        mm1Nd2NzParamsForA.srcDValue = info.actualSingleProcessSInnerSizeAlign; // 同一个ND矩阵中相邻行起始地址之间的偏移, 单位为元素个数
        mm1Nd2NzParamsForA.dstNzC0Stride = 16; // 转换为NZ矩阵后，相邻Block起始地址之间的偏移, 单位为Block个数
        mm1Nd2NzParamsForA.dstNzNStride = 1; // 转换为NZ矩阵后，ND中之前相邻两行在NZ矩阵中起始地址之间的偏移
        mm1Nd2NzParamsForA.dstNzMatrixStride = 0; // 两个NZ矩阵，起始地址之间的偏移
        DataCopy(aL1Tensor[i * copyStrideL1], vec1ResGm[(info.loop % (PRE_LOAD_NUM)) * mmResUbSize + kCopyIdx * kCopyRowCount + i * copyStrideGm], mm1Nd2NzParamsForA);
    }
}


template <typename NSAT>
__aicore__ inline void NsaSelectAttentionInfer<NSAT>::CopyInMm2BToL1ForPA(
    LocalTensor<KV_T>& bL1Tensor, uint64_t valueGmBaseOffset, uint32_t copyTotalRowCnt, uint32_t copyStartRowCnt, uint32_t kActCopyRowCount)
{
    uint32_t blockK = 32 / sizeof(KV_T); // 单个ND矩阵的行数
    uint64_t step = headDimV;
    if constexpr (LAYOUT_T == LAYOUT::BSH || LAYOUT_T == LAYOUT::BSND || LAYOUT_T == LAYOUT::TND) {
        step = headDimV * kvHeadNum;
    }

    uint32_t nHead = 0;
    uint32_t nTail = 0;
    uint32_t midNdNum = 0;
    uint32_t copyStartRowCntInBaseN = copyStartRowCnt % blockK;
    uint32_t copyStartRowCntInBaseNTail = 0;
    if (copyStartRowCntInBaseN + kActCopyRowCount <= blockK) {
        nHead = 0;
        midNdNum = 0;
        nTail = kActCopyRowCount;
        copyStartRowCntInBaseNTail = copyStartRowCntInBaseN;
    } else {
        if (copyStartRowCntInBaseN == 0) {
            nHead = 0;
        } else {
            nHead = blockK - copyStartRowCntInBaseN;
        }
        midNdNum = (kActCopyRowCount - nHead) / blockK;
        nTail = (kActCopyRowCount - nHead) % blockK;
        copyStartRowCntInBaseNTail = 0;
    }

    Nd2NzParams mm1Nd2NzParamsForB;
    mm1Nd2NzParamsForB.dValue = headDimV;
    mm1Nd2NzParamsForB.srcDValue = step; // 同一个ND矩阵中相邻行起始地址之间的偏移, 单位为元素个数
    mm1Nd2NzParamsForB.dstNzC0Stride = blockK;
    mm1Nd2NzParamsForB.dstNzNStride = 1;
    if (nHead != 0) {
        mm1Nd2NzParamsForB.ndNum = 1;
        mm1Nd2NzParamsForB.nValue = nHead; // 单个ND矩阵的行数, 单位为元素个数
        mm1Nd2NzParamsForB.srcNdMatrixStride = 0; // 相邻ND矩阵起始地址之间的偏移, 单位为元素个数
        mm1Nd2NzParamsForB.dstNzMatrixStride = 0; // 两个NZ矩阵，起始地址之间的偏移, 单位为元素个数

        uint32_t ndNumFinish = copyStartRowCnt / blockK;
        DataCopy(bL1Tensor[ndNumFinish * blockK * headDimVAlign + copyStartRowCntInBaseN * (32 / sizeof(KV_T))], valueGm[valueGmBaseOffset],
                mm1Nd2NzParamsForB);
    }

    if (midNdNum != 0) {
        int32_t ndNumFinish = (copyStartRowCnt + nHead) / blockK;
        if ((LAYOUT_T == LAYOUT::BSH || LAYOUT_T == LAYOUT::BSND || LAYOUT_T == LAYOUT::TND) && (blockK * step > 65535)) {
            mm1Nd2NzParamsForB.ndNum = 1;
            mm1Nd2NzParamsForB.nValue = blockK; // 单个ND矩阵的行数, 单位为元素个数
            mm1Nd2NzParamsForB.srcNdMatrixStride = 0; // 相邻ND矩阵起始地址之间的偏移, 单位为元素个数
            mm1Nd2NzParamsForB.dstNzMatrixStride = 0; // 两个NZ矩阵，起始地址之间的偏移, 单位为元素个数

            for (uint32_t i = 0; i < midNdNum; i++) {
                DataCopy(bL1Tensor[(ndNumFinish + i) * blockK * headDimVAlign], valueGm[valueGmBaseOffset + nHead * step + i * blockK * step], mm1Nd2NzParamsForB);
            }
        } else {
            mm1Nd2NzParamsForB.ndNum = midNdNum;
            mm1Nd2NzParamsForB.nValue = blockK; // 单个ND矩阵的行数, 单位为元素个数
            mm1Nd2NzParamsForB.srcNdMatrixStride = blockK * step; // 相邻ND矩阵起始地址之间的偏移, 单位为元素个数
            mm1Nd2NzParamsForB.dstNzMatrixStride = blockK * headDimVAlign; // 两个NZ矩阵，起始地址之间的偏移, 单位为元素个数

            DataCopy(bL1Tensor[ndNumFinish * blockK * headDimVAlign], valueGm[valueGmBaseOffset + nHead * step], mm1Nd2NzParamsForB);
        }
    }

    if (nTail != 0) {
        mm1Nd2NzParamsForB.ndNum = 1;
        mm1Nd2NzParamsForB.nValue = nTail; // 单个ND矩阵的行数, 单位为元素个数
        mm1Nd2NzParamsForB.srcNdMatrixStride = 0; // 相邻ND矩阵起始地址之间的偏移, 单位为元素个数
        mm1Nd2NzParamsForB.dstNzMatrixStride = 0; // 两个NZ矩阵，起始地址之间的偏移, 单位为元素个数

        int32_t ndNumFinish = (copyStartRowCnt + nHead) / blockK + midNdNum;
        DataCopy(bL1Tensor[ndNumFinish * blockK * headDimVAlign + copyStartRowCntInBaseNTail * (32 / sizeof(KV_T))], valueGm[valueGmBaseOffset + (nHead + midNdNum * blockK) * step],
                 mm1Nd2NzParamsForB);
    }
}

template <typename NSAT>
__aicore__ inline void NsaSelectAttentionInfer<NSAT>::LoadDataMm2A(LocalTensor<KV_T> aL0Tensor, LocalTensor<KV_T> aL1Tensor, uint32_t kSize) {
    LoadData2DParams loadData2DParams;
    loadData2DParams.startIndex = 0;
    loadData2DParams.repeatTimes = kSize / (32 / sizeof(KV_T));
    loadData2DParams.srcStride = 1;
    loadData2DParams.dstGap = 0;
    loadData2DParams.ifTranspose = false;
    LoadData(aL0Tensor, aL1Tensor, loadData2DParams);
}