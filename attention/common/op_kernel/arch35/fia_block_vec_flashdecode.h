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
 * \file fia_block_vec_flashdecode.h
 * \brief
 */
#ifndef FIA_BLOCK_VEC_FLASHDECODE_H
#define FIA_BLOCK_VEC_FLASHDECODE_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "../fia_public_define.h"
#include "../vector_common_35.h"
// #include "../offset_calculator.h"
#include "infer_flash_attention_comm.h"
// #include "../post_quant.h"
#include "../memory_copy_fd.h"
#include "infer_flash_attention_comm.h"
// using namespace AttentionCommon;
namespace BaseApi{


struct TaskInfo {
    uint32_t bIdx;
    uint32_t n2Idx;
    uint32_t gS1Idx;
    uint32_t actualCombineLoopSize;
};

//SBH目前不支持
__aicore__ inline constexpr AttentionCommon::FIA_LAYOUT ConvertLayoutEnum(LayOutTypeEnum layoutType, bool isOutput = false) {
    switch (layoutType) {
        case LayOutTypeEnum::LAYOUT_BSH:
            return AttentionCommon::FIA_LAYOUT::BSH;

        case LayOutTypeEnum::LAYOUT_BNSD:
            return AttentionCommon::FIA_LAYOUT::BNSD;

        case LayOutTypeEnum::LAYOUT_TND:
            return AttentionCommon::FIA_LAYOUT::TND;

        case LayOutTypeEnum::LAYOUT_NTD_TND:
            if(isOutput){
                return AttentionCommon::FIA_LAYOUT::TND;
            }else{
                return AttentionCommon::FIA_LAYOUT::NTD;
            }
        default: //LAYOUT_SBH
            ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "Unhandled LayOutTypeEnum value in ConvertLayoutEnum, current value is %d", layoutType); });
            return AttentionCommon::FIA_LAYOUT::BSH; // 不会执行，默认值
    }
}

template <typename T>
__aicore__ inline T Align(T num, T rnd)
{
    return (((rnd) == 0) ? 0 : (((num) + (rnd)-1) / (rnd) * (rnd)));
}

TEMPLATES_DEF 
class FiaBlockVecFlashDecode {
public:
    // =================================类型定义区=================================
    // 中间计算数据类型为float，高精度模式
    using Type = float;
    using FDGmType = typename std::conditional<isFd, GlobalTensor<float>, int8_t>::type;
    using OUT_T = OUTPUT_T;   //问

    static constexpr AttentionCommon::FIA_LAYOUT LAYOUT_T = ConvertLayoutEnum(layout);

    __aicore__ inline void InitGlobalTensor(FDGmType lseMaxFdGm, FDGmType lseSumFdGm, FDGmType accumOutGm, 
        GlobalTensor<OUT_T> attentionOutGm, GlobalTensor<uint64_t> actualSeqLengthsGmQ, GlobalTensor<uint64_t> actualSeqLengthsGm);

    __aicore__ inline void InitParams(const ConstInfo<isInfer, hasRope> &constInfo);
    __aicore__ inline void InitDecodeParams();
    __aicore__ inline void InitBuffers(TPipe *pipe);
    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();   
    __aicore__ inline void FlashDecode(FDparams &fd);
    template <typename U> //避免重名用U
    __aicore__ inline U Align(U num, U rnd)
    {
        return (((rnd) == 0) ? 0 : (((num) + (rnd)-1) / (rnd) * (rnd)));
    }
protected:
    __aicore__ inline void CopyAccumOutIn(LocalTensor<Type> &accumOutLocal, uint32_t splitKVIndex, uint32_t startRow,
                                          uint32_t dealRowCount);                            
    __aicore__ inline void CopyLseIn(uint32_t startRow, uint32_t dealRowCount, uint64_t baseOffset, uint32_t cntM);
    __aicore__ inline void ComputeScaleValue(LocalTensor<Type> &lseExp, uint32_t startRow, uint32_t dealRowCount,
                                             uint32_t cntM);
    __aicore__ inline void Bmm2DataCopyOutTrans(LocalTensor<OUT_T> &attenOutUb, uint32_t startRow,
                                                 uint32_t dealRowCount, uint32_t columnCount);
    __aicore__ inline void Bmm2DataCopyOut(uint64_t attenOutOffset, LocalTensor<OUT_T> &attenOutUb, uint32_t startRow,
                                           uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount);
    __aicore__ inline void ReduceFinalRes(LocalTensor<Type> &reduceOut, LocalTensor<Type> &mm2Res, LocalTensor<Type> &lseLocal, 
                                          uint32_t cntKV, uint32_t dealRowCount);
    __aicore__ inline void CopyFinalResOut(LocalTensor<Type> &accumOutLocal, uint32_t startRow, uint32_t dealRowCount,
                                           uint32_t cntM);
    __aicore__ inline void CalcPreNextTokens();

private:
// =================================常量区=================================
    static constexpr uint64_t SYNC_LSE_MAX_SUM_BUF1_FLAG = 8;
    static constexpr uint64_t SYNC_LSE_MAX_SUM_BUF2_FLAG = 9;
    static constexpr uint64_t SYNC_MM2RES_BUF1_FLAG = 10;
    static constexpr uint64_t SYNC_MM2RES_BUF2_FLAG = 11;
    static constexpr uint64_t SYNC_FDOUTPUT_BUF_FLAG = 6;
    static constexpr uint64_t SYNC_LSEOUTPUT_BUF_FLAG = 7;

    static constexpr uint32_t BLOCK_ELEMENT_NUM = fa_base_vector::BYTE_BLOCK / sizeof(Type); // 32/4=8

    // uint64_t constInfo.queryLeftPaddingSize = runInfo.constInfo.queryLeftPaddingSize; //只有右padding？
    //mbasesize  s1basesize ?
    AttentionCommon::FIA_LAYOUT outputLayout = ConvertLayoutEnum(layout, true);
    uint32_t preLoadNum = 2U;
    uint32_t dSizeV_Align;  // 声明为成员变量
    
protected:
    FDGmType lseSumFdGm;
    FDGmType lseMaxFdGm;
    FDGmType accumOutGm;
    GlobalTensor<OUT_T> attentionOutGm;
    GlobalTensor<uint64_t> actualSeqLengthsGmQ;
    GlobalTensor<uint64_t> actualSeqLengthsGm;


    // =======================获取实际Act_S，用于行无效处理===========================
    static constexpr bool PAGE_ATTENTION = isPa; //问
    static constexpr ActualSeqLensMode Q_MODE = fa_memory_copy_fd::GetQActSeqMode<LAYOUT_T>();
    static constexpr ActualSeqLensMode KV_MODE = fa_memory_copy_fd::GetKvActSeqMode<LAYOUT_T, PAGE_ATTENTION>();
    // tensorlist
    __gm__ uint8_t *keyPtr = nullptr;
    fa_memory_copy_fd::ActualSeqLensParser<Q_MODE> qActSeqLensParser;
    fa_memory_copy_fd::ActualSeqLensParser<KV_MODE> kvActSeqLensParser;
    uint64_t actSeqLensKv = 0;
    uint64_t actSeqLensQ = 0;
    
    int64_t preTokensPerBatch = 0;
    int64_t nextTokensPerBatch = 0;
    
    static constexpr Type BOOL_ATTEN_MASK_SCALAR_VALUE = -1000000000000.0; // 用于mask为bool类型
    uint32_t negativeIntScalar = *((uint32_t *)&BOOL_ATTEN_MASK_SCALAR_VALUE);
    // static constexpr bool POST_QUANT = IsSameType<OUT_T, int8_t>::value;
    Type scale2Value = 0;
    Type offset2Value = 0;
    // ================================类成员变量====================================
    // aic、aiv核信息
    uint32_t blockIdx = 0U;
    ConstInfo<isInfer, hasRope> constInfo{};
    TaskInfo taskInfo{};
private:    
 // ================================FD Local Buffer区====================================
    TBuf<> fdSumBuf1;    // 1.5k: 16*24*4
    TBuf<> fdSumBuf2;    // 1.5k: 16*24*4
    TBuf<> fdMaxBuf1;    // 1.5k: 16*24*4
    TBuf<> fdMaxBuf2;    // 1.5k: 16*24*4
    TBuf<> fdLseExpBuf;  // 1.5k: 16*24*4
    TBuf<> fdMm2ResBuf1; // 32k: 16*512*4
    TBuf<> fdMm2ResBuf2; // 32k: 16*512*4
    TBuf<> fdReduceBuf;  // 32k: 16*512*4
    TBuf<> fdOutputBuf;  // 32k: 16*512*4

    TBuf<> fdLseMaxUbBuf1; // 64B: 16*4
    TBuf<> fdLseMaxUbBuf2; // 64B: 16*4
    TBuf<> fdLseSumUbBuf1; // 64B: 16*4
    TBuf<> fdLseSumUbBuf2; // 64B: 16*4
    // TBuf<> quant2TmpBuf1;
    // TBuf<> quant2TmpBuf2;
    TBuf<> fdLseMaxUbBuf; // 64B: 16*4
    TBuf<> fdLseSumUbBuf; // 64B: 16*4
    TBuf<> fdLseUbBuf; // 64B: 16*4
};

TEMPLATES_DEF_NO_DEFAULT __aicore__ inline 
void FiaBlockVecFlashDecode<TEMPLATE_ARGS>::InitGlobalTensor(FDGmType lseMaxFdGm, 
                                                        FDGmType lseSumFdGm, 
                                                        FDGmType accumOutGm,
                                                        GlobalTensor<OUT_T> attentionOutGm,
                                                        GlobalTensor<uint64_t> actualSeqLengthsGmQ,
                                                        GlobalTensor<uint64_t> actualSeqLengthsGm
                                                        )
                                                        // __gm__ uint8_t *key
                                                        // __gm__ uint8_t *quantScale2,
                                                        // __gm__ uint8_t *quantOffset2)
{
   this->lseMaxFdGm = lseMaxFdGm;
   this->lseSumFdGm = lseSumFdGm;
   this->accumOutGm = accumOutGm;
   this->attentionOutGm = attentionOutGm;
   this->actualSeqLengthsGmQ = actualSeqLengthsGmQ;
   this->actualSeqLengthsGm = actualSeqLengthsGm;
//    this->keyPtr = key;

   qActSeqLensParser.Init(this->actualSeqLengthsGmQ, constInfo.actualSeqLenSize, constInfo.s1Size);
   kvActSeqLensParser.Init(this->actualSeqLengthsGm, constInfo.actualSeqLenKVSize, constInfo.s2Size);

}


TEMPLATES_DEF_NO_DEFAULT __aicore__ inline 
void FiaBlockVecFlashDecode<TEMPLATE_ARGS>::InitParams(const ConstInfo<isInfer, hasRope> &constInfo)
{
   this->constInfo = constInfo;
   // constexpr uint32_t byte_block = static_cast<uint32_t>(fa_base_vector::BYTE_BLOCK);
   this->dSizeV_Align = Align(constInfo.dSizeV, fa_base_vector::BYTE_BLOCK);
    
}


TEMPLATES_DEF_NO_DEFAULT  __aicore__ inline 
void FiaBlockVecFlashDecode<TEMPLATE_ARGS>::InitDecodeParams()
{
    this->blockIdx = GetBlockIdx();
}

TEMPLATES_DEF_NO_DEFAULT __aicore__ inline 
void FiaBlockVecFlashDecode<TEMPLATE_ARGS>::InitBuffers(TPipe *pipe)
{
    if ASCEND_IS_AIV {
        pipe->Reset();
        // InQue, DB, SYNC_LSE_MAX_SUM_BUF1_FLAG SYNC_LSE_MAX_SUM_BUF2_FLAG
        pipe->InitBuffer(fdSumBuf1, ConstInfo<isInfer, hasRope>::BUFFER_SIZE_BYTE_4K + ConstInfo<isInfer, hasRope>::BUFFER_SIZE_BYTE_2K); //考虑调大点 在关注一下

        pipe->InitBuffer(fdSumBuf2, ConstInfo<isInfer, hasRope>::BUFFER_SIZE_BYTE_4K + ConstInfo<isInfer, hasRope>::BUFFER_SIZE_BYTE_2K);
        pipe->InitBuffer(fdMaxBuf1, ConstInfo<isInfer, hasRope>::BUFFER_SIZE_BYTE_4K + ConstInfo<isInfer, hasRope>::BUFFER_SIZE_BYTE_2K);
        pipe->InitBuffer(fdMaxBuf2, ConstInfo<isInfer, hasRope>::BUFFER_SIZE_BYTE_4K + ConstInfo<isInfer, hasRope>::BUFFER_SIZE_BYTE_2K);
        // TmpBuf
        pipe->InitBuffer(fdLseExpBuf, ConstInfo<isInfer, hasRope>::BUFFER_SIZE_BYTE_4K + ConstInfo<isInfer, hasRope>::BUFFER_SIZE_BYTE_2K);
        // InQue, DB, SYNC_MM2RES_BUF1_FLAG SYNC_MM2RES_BUF2_FLAG
        pipe->InitBuffer(fdMm2ResBuf1, ConstInfo<isInfer, hasRope>::BUFFER_SIZE_BYTE_16K);
        pipe->InitBuffer(fdMm2ResBuf2, ConstInfo<isInfer, hasRope>::BUFFER_SIZE_BYTE_16K);
        // TmpBuf
        pipe->InitBuffer(fdReduceBuf, ConstInfo<isInfer, hasRope>::BUFFER_SIZE_BYTE_16K);
        // OutQue, SYNC_FDOUTPUT_BUF_FLAG
        pipe->InitBuffer(fdOutputBuf, ConstInfo<isInfer, hasRope>::BUFFER_SIZE_BYTE_16K);
        // TmpBuf, UB开DB
        pipe->InitBuffer(fdLseMaxUbBuf1, ConstInfo<isInfer, hasRope>::BUFFER_SIZE_BYTE_256B);
        pipe->InitBuffer(fdLseSumUbBuf1, ConstInfo<isInfer, hasRope>::BUFFER_SIZE_BYTE_256B);
        pipe->InitBuffer(fdLseMaxUbBuf2, ConstInfo<isInfer, hasRope>::BUFFER_SIZE_BYTE_256B);
        pipe->InitBuffer(fdLseSumUbBuf2, ConstInfo<isInfer, hasRope>::BUFFER_SIZE_BYTE_256B);
        // OutQue, SYNC_LSEOUTPUT_BUF_FLAG
        pipe->InitBuffer(fdLseUbBuf, ConstInfo<isInfer, hasRope>::BUFFER_SIZE_BYTE_256B);


     }
}

TEMPLATES_DEF_NO_DEFAULT __aicore__ inline 
void FiaBlockVecFlashDecode<TEMPLATE_ARGS>::AllocEventID()
{
    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_LSE_MAX_SUM_BUF1_FLAG);
    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_LSE_MAX_SUM_BUF2_FLAG);
    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_MM2RES_BUF1_FLAG);
    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_MM2RES_BUF2_FLAG);
    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_FDOUTPUT_BUF_FLAG);
    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_LSEOUTPUT_BUF_FLAG);
}

TEMPLATES_DEF_NO_DEFAULT __aicore__ inline 
void FiaBlockVecFlashDecode<TEMPLATE_ARGS>::FreeEventID()
{
    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_LSE_MAX_SUM_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_LSE_MAX_SUM_BUF2_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_MM2RES_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_MM2RES_BUF2_FLAG);
    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_FDOUTPUT_BUF_FLAG);
    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_LSEOUTPUT_BUF_FLAG);
}

TEMPLATES_DEF_NO_DEFAULT __aicore__ inline 
void FiaBlockVecFlashDecode<TEMPLATE_ARGS>::CopyAccumOutIn(LocalTensor<Type> &accumOutLocal, uint32_t splitKVIndex,
    uint32_t startRow, uint32_t dealRowCount)
{
    DataCopyExtParams copyInParams;
    DataCopyPadExtParams<Type> copyInPadParams;
    copyInParams.blockCount = dealRowCount;
    copyInParams.blockLen = constInfo.dSizeV * sizeof(Type);
    copyInParams.srcStride = 0;
    copyInParams.dstStride = (this->dSizeV_Align - constInfo.dSizeV) / BLOCK_ELEMENT_NUM;

    copyInPadParams.isPad = true;
    copyInPadParams.leftPadding = 0;
    copyInPadParams.rightPadding = (this->dSizeV_Align - constInfo.dSizeV) % BLOCK_ELEMENT_NUM;
    copyInPadParams.paddingValue = 0;
    uint64_t combineAccumOutOffset = startRow * constInfo.dSizeV +                          // taskoffset + g轴offset
                                      splitKVIndex * constInfo.s1BaseSize * constInfo.dSizeV; // 份数offset

    DataCopyPad(accumOutLocal, accumOutGm[combineAccumOutOffset], copyInParams, copyInPadParams);
}

TEMPLATES_DEF_NO_DEFAULT __aicore__ inline 
void FiaBlockVecFlashDecode<TEMPLATE_ARGS>::CopyLseIn(uint32_t startRow,
    uint32_t dealRowCount, uint64_t baseOffset, uint32_t cntM)
{
    LocalTensor<Type> lseSum = cntM % 2 == 0 ? fdSumBuf1.Get<Type>() : fdSumBuf2.Get<Type>();
    LocalTensor<Type> lseMax = cntM % 2 == 0 ? fdMaxBuf1.Get<Type>() : fdMaxBuf2.Get<Type>();

    uint64_t combineLseOffset = (baseOffset + startRow) * fa_base_vector::FP32_BLOCK_ELEMENT_NUM;
    uint64_t combineLoopOffset = constInfo.s1BaseSize * fa_base_vector::FP32_BLOCK_ELEMENT_NUM;
    uint64_t dealRowCountAlign = dealRowCount * fa_base_vector::FP32_BLOCK_ELEMENT_NUM;
    for (uint32_t i = 0; i < taskInfo.actualCombineLoopSize; i++) {
        DataCopy(lseSum[i * dealRowCountAlign], lseSumFdGm[combineLseOffset + i * combineLoopOffset],
                 dealRowCountAlign); // 份数offset
        DataCopy(lseMax[i * dealRowCountAlign], lseMaxFdGm[combineLseOffset + i * combineLoopOffset],
                 dealRowCountAlign);
    }
}


TEMPLATES_DEF_NO_DEFAULT __aicore__ inline void
FiaBlockVecFlashDecode<TEMPLATE_ARGS>::ComputeScaleValue(LocalTensor<Type> &lseExp, 
                                                    uint32_t startRow,
                                                    uint32_t dealRowCount, 
                                                    uint32_t cntM)
{
    LocalTensor<Type> lseSum = cntM % 2 == 0 ? fdSumBuf1.Get<Type>() : fdSumBuf2.Get<Type>();
    LocalTensor<Type> lseMax = cntM % 2 == 0 ? fdMaxBuf1.Get<Type>() : fdMaxBuf2.Get<Type>();

    // 开双buff
    LocalTensor<Type> lseMaxUb = cntM % 2 == 0 ? fdLseMaxUbBuf1.Get<Type>() : fdLseMaxUbBuf2.Get<Type>();
    LocalTensor<Type> lseSumUb = cntM % 2 == 0 ? fdLseSumUbBuf1.Get<Type>() : fdLseSumUbBuf2.Get<Type>();
    uint64_t dealRowCountAlign = dealRowCount * fa_base_vector::FP32_BLOCK_ELEMENT_NUM;

    Duplicate(lseMaxUb, -AttentionCommon::ConstInfo::, dealRowCountAlign); //attention/incre_flash_attention/op_kernel/arch35/incre_flash_attention_pub.h 定义FLOAT_MAX

    Duplicate(lseSumUb, -AttentionCommon::ConstInfo::FLOAT_ZERO, dealRowCountAlign); //attention/incre_flash_attention/op_kernel/arch35/incre_flash_attention_pub.h 定义FLOAT_ZERO
    AscendC::PipeBarrier<PIPE_V>();

    fa_base_vector::ColMax(lseMaxUb, lseMax, lseMaxUb, taskInfo.actualCombineLoopSize, dealRowCountAlign, dealRowCountAlign);
    AscendC::PipeBarrier<PIPE_V>();

    fa_base_vector::RowSub(lseExp, lseMax, lseMaxUb, taskInfo.actualCombineLoopSize, dealRowCountAlign, dealRowCountAlign);
    AscendC::PipeBarrier<PIPE_V>();

    Exp(lseExp, lseExp, taskInfo.actualCombineLoopSize * dealRowCountAlign);
    AscendC::PipeBarrier<PIPE_V>();

    Mul(lseExp, lseSum, lseExp, taskInfo.actualCombineLoopSize * dealRowCountAlign);
    AscendC::PipeBarrier<PIPE_V>();

    fa_base_vector::ColAdd(lseSumUb, lseExp, lseSumUb, taskInfo.actualCombineLoopSize, dealRowCountAlign, dealRowCountAlign);
    AscendC::PipeBarrier<PIPE_V>();


    fa_base_vector::MatDivsVec(lseExp, lseExp, lseSumUb, taskInfo.actualCombineLoopSize, dealRowCountAlign, dealRowCountAlign);
    AscendC::PipeBarrier<PIPE_V>();
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FiaBlockVecFlashDecode<TEMPLATE_ARGS>::Bmm2DataCopyOutTrans(LocalTensor<OUT_T> &attenOutUb, uint32_t startRow,
                                                                      uint32_t dealRowCount, uint32_t columnCount)
{
    fa_memory_copy_fd::FaUbTensor<OUT_T> ubTensor {
        .tensor = attenOutUb,
        .rowCount = dealRowCount,
        .colCount = columnCount,
    };
    GmCoord gmCoord{.bIdx = taskInfo.bIdx,
                    .n2Idx = taskInfo.n2Idx,
                    .gS1Idx = taskInfo.gS1Idx + startRow,
                    .dIdx = 0,
                    .gS1DealSize = dealRowCount,
                    .dDealSize = (uint32_t)constInfo.dSizeV};

    if (outputLayout == AttentionCommon::FIA_LAYOUT::BSH) {
        constexpr GmFormat OUT_FORMAT = GmFormat::BSNGD;
        fa_memory_copy_fd::FaGmTensor<OUT_T, OUT_FORMAT> outGmTensor;
        outGmTensor.gmTensor = attentionOutGm;
        outGmTensor.offsetCalculator.Init(constInfo.bSize, constInfo.n2Size, constInfo.gSize, constInfo.s1Size,
                                          constInfo.dSizeV, actualSeqLengthsGmQ, constInfo.actualSeqLenSize, constInfo.isQHasLeftPadding, constInfo.queryLeftPaddingSize);
        fa_memory_copy_fd::CopyAttenOutUbToGm<OUT_T, OUT_FORMAT, fa_memory_copy_fd::GetOutUbFormat<LAYOUT_T>()> copyAttenOutUbToGm;
        copyAttenOutUbToGm(outGmTensor, ubTensor, gmCoord);
    } else if (outputLayout == AttentionCommon::FIA_LAYOUT::BNSD) {
        constexpr GmFormat OUT_FORMAT = GmFormat::BNGSD;
        fa_memory_copy_fd::FaGmTensor<OUT_T, OUT_FORMAT> outGmTensor;
        outGmTensor.gmTensor = attentionOutGm;
        outGmTensor.offsetCalculator.Init(constInfo.bSize, constInfo.n2Size, constInfo.gSize, constInfo.s1Size,
                                          constInfo.dSizeV, actualSeqLengthsGmQ, constInfo.actualSeqLenSize, constInfo.isQHasLeftPadding, constInfo.queryLeftPaddingSize);
        fa_memory_copy_fd::CopyAttenOutUbToGm<OUT_T, OUT_FORMAT, fa_memory_copy_fd::GetOutUbFormat<LAYOUT_T>()> copyAttenOutUbToGm;
        copyAttenOutUbToGm(outGmTensor, ubTensor, gmCoord);
    } else if (outputLayout == AttentionCommon::FIA_LAYOUT::NBSD) {
        constexpr GmFormat OUT_FORMAT = GmFormat::NGBSD;
        fa_memory_copy_fd::FaGmTensor<OUT_T, OUT_FORMAT> outGmTensor;
        outGmTensor.gmTensor = attentionOutGm;
        outGmTensor.offsetCalculator.Init(constInfo.bSize, constInfo.n2Size, constInfo.gSize, constInfo.s1Size,
                                          constInfo.dSizeV, actualSeqLengthsGmQ, constInfo.actualSeqLenSize);
        fa_memory_copy_fd::CopyAttenOutUbToGm<OUT_T, OUT_FORMAT, fa_memory_copy_fd::GetOutUbFormat<LAYOUT_T>()> copyAttenOutUbToGm;
        copyAttenOutUbToGm(outGmTensor, ubTensor, gmCoord);
    } else if (outputLayout == AttentionCommon::FIA_LAYOUT::TND) {
        constexpr GmFormat OUT_FORMAT = GmFormat::TNGD;
        fa_memory_copy_fd::FaGmTensor<OUT_T, OUT_FORMAT> outGmTensor;
        outGmTensor.gmTensor = attentionOutGm;
        outGmTensor.offsetCalculator.Init(constInfo.n2Size, constInfo.gSize, constInfo.dSizeV, actualSeqLengthsGmQ,
                                          constInfo.actualSeqLenSize);
        fa_memory_copy_fd::CopyAttenOutUbToGm<OUT_T, OUT_FORMAT, fa_memory_copy_fd::GetOutUbFormat<LAYOUT_T>()> copyAttenOutUbToGm;
        copyAttenOutUbToGm(outGmTensor, ubTensor, gmCoord);
    } else if (outputLayout == AttentionCommon::FIA_LAYOUT::NTD) {
        constexpr GmFormat OUT_FORMAT = GmFormat::NGTD;
        fa_memory_copy_fd::FaGmTensor<OUT_T, OUT_FORMAT> outGmTensor;
        outGmTensor.gmTensor = attentionOutGm;
        outGmTensor.offsetCalculator.Init(constInfo.n2Size, constInfo.gSize, constInfo.dSizeV, actualSeqLengthsGmQ,
                                          constInfo.actualSeqLenSize);
        fa_memory_copy_fd::CopyAttenOutUbToGm<OUT_T, OUT_FORMAT, fa_memory_copy_fd::GetOutUbFormat<LAYOUT_T>()> copyAttenOutUbToGm;
        copyAttenOutUbToGm(outGmTensor, ubTensor, gmCoord);
    }
}

TEMPLATES_DEF_NO_DEFAULT __aicore__ inline 
void FiaBlockVecFlashDecode<TEMPLATE_ARGS>::Bmm2DataCopyOut(uint64_t attenOutOffset, LocalTensor<OUT_T> &attenOutUb,
                                                               uint32_t startRow, uint32_t dealRowCount,
                                                               uint32_t columnCount, uint32_t actualColumnCount)
{
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = dealRowCount;
    dataCopyParams.blockLen = actualColumnCount * sizeof(OUT_T);
    dataCopyParams.srcStride = (columnCount - actualColumnCount) / (fa_base_vector::BYTE_BLOCK / sizeof(OUT_T));
    dataCopyParams.dstStride = 0;
    DataCopyPad(attentionOutGm[attenOutOffset + startRow * actualColumnCount], attenOutUb,
                dataCopyParams);
}

TEMPLATES_DEF_NO_DEFAULT __aicore__ inline 
void FiaBlockVecFlashDecode<TEMPLATE_ARGS>::ReduceFinalRes(LocalTensor<Type> &reduceOut, 
                                                      LocalTensor<Type> &mm2Res, 
                                                      LocalTensor<Type> &lseLocal, 
                                                      uint32_t cntKV, 
                                                      uint32_t dealRowCount)
{
    uint32_t dealRowCountAlign = dealRowCount * fa_base_vector::FP32_BLOCK_ELEMENT_NUM;
    LocalTensor<Type> tmpRst =
        cntKV == 0 ? reduceOut : mm2Res; // 第一次mul结果直接写入reduceOut，否则在mm2Res原地进行mul，再加到reduceOut

    fa_base_vector::RowMuls(tmpRst, mm2Res, lseLocal[cntKV * dealRowCountAlign], dealRowCount, this->dSizeV_Align, constInfo.dSizeV);

    if (cntKV != 0) {
        AscendC::PipeBarrier<PIPE_V>();
        Add(reduceOut, reduceOut, tmpRst, dealRowCount * this->dSizeV_Align);
        AscendC::PipeBarrier<PIPE_V>();
    }
}

TEMPLATES_DEF_NO_DEFAULT __aicore__ inline 
void FiaBlockVecFlashDecode<TEMPLATE_ARGS>::CopyFinalResOut(LocalTensor<Type> &accumOutLocal, 
                                                       uint32_t startRow,
                                                       uint32_t dealRowCount,
                                                       uint32_t cntM)
{
    // DealInvalidRows(accumOutLocal, startRow, dealRowCount, this->dSizeV_Align); //刷sum正无穷，保证算出来结果为0
    // DealInvalidMaskRows(accumOutLocal, startRow, dealRowCount, this->dSizeV_Align, cntM);
    AscendC::PipeBarrier<PIPE_V>();
    LocalTensor<OUT_T> tmpBmm2ResCastTensor = fdOutputBuf.Get<OUT_T>();
    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_FDOUTPUT_BUF_FLAG);
    uint32_t shapeArray[] = {dealRowCount, (uint32_t)constInfo.dSizeV};
    tmpBmm2ResCastTensor.SetShapeInfo(ShapeInfo(2, shapeArray, DataFormat::ND));
    if constexpr (IsSameType<OUT_T, bfloat16_t>::value) { // bf16 采取四舍六入五成双模式
        Cast(tmpBmm2ResCastTensor, accumOutLocal, AscendC::RoundMode::CAST_RINT, dealRowCount * this->dSizeV_Align);
    } else {
        Cast(tmpBmm2ResCastTensor, accumOutLocal, AscendC::RoundMode::CAST_ROUND, dealRowCount * this->dSizeV_Align);
    }

    SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_FDOUTPUT_BUF_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_FDOUTPUT_BUF_FLAG);
    Bmm2DataCopyOutTrans(tmpBmm2ResCastTensor, startRow, dealRowCount, this->dSizeV_Align);
    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_FDOUTPUT_BUF_FLAG);
}

TEMPLATES_DEF_NO_DEFAULT __aicore__ inline void
FiaBlockVecFlashDecode<TEMPLATE_ARGS>::FlashDecode(FDparams &fd)
{
    if (blockIdx >= fd.fdUsedVecNum ) {
        return;
    }
    uint32_t fdTaskPrevEnd = (blockIdx > 0) ? fd.fdBalanceEndIdx1[blockIdx - 1] : 0; // 上一个核末尾是第几个规约
    uint32_t fdS1gOuterMPrevEnd =
        (blockIdx > 0) ? fd.fdBalanceEndIdx2[blockIdx - 1] : 0; //上一个核末尾是该规约的第几个base行
    uint32_t fdTaskEnd = fd.fdBalanceEndIdx1[blockIdx];                 // 当前核的末尾是第几个规约任务
    uint32_t fdS1gOuterMEnd = fd.fdBalanceEndIdx2[blockIdx]; // 当前核的末尾是该规约的第几个base行
    uint32_t tmpFdS1gOuterMStart = (blockIdx > 0) ? fdS1gOuterMPrevEnd + 1 : 0; // 当前核从第几个base行开始
    uint32_t tmpFdS1gOuterMEnd = 0;
    uint32_t reduceGlobaLoop = 0;
    uint32_t reduceMLoop = 0;

    for (uint32_t fdTaskId = fdTaskPrevEnd; fdTaskId <= fdTaskEnd; fdTaskId++) {
        tmpFdS1gOuterMEnd = (fdTaskId == fdTaskEnd) ? fdS1gOuterMEnd : (fd.fdBalanceMSplitNum[fdTaskId] - 1);
        taskInfo.bIdx = fd.fdBN2Idx[fdTaskId] / constInfo.n2Size;
        taskInfo.n2Idx = fd.fdBN2Idx[fdTaskId] % constInfo.n2Size;
        taskInfo.gS1Idx = fd.fdMIdx[fdTaskId] * constInfo.s1BaseSize;
        taskInfo.actualCombineLoopSize = fd.fdS2SplitNum[fdTaskId]; // 当前规约任务kv方向有几份

        uint64_t combineTaskPrefixSum = 0;
        for (int i = 0; i < fdTaskId; i++) {
            // 计算此前规约数据的累计份数，每一份的数据大小为 kvHeadNum * constInfo.tndSgBasicSize
            // |Task0-0|Task0-1|Task0-3|Task1-0|Task1-2|...|
            combineTaskPrefixSum += fd.fdS2SplitNum[i];
        }

        uint64_t taskOffset = combineTaskPrefixSum * constInfo.s1BaseSize;

        for (uint32_t fdS1gOuterMIdx = tmpFdS1gOuterMStart; fdS1gOuterMIdx <= tmpFdS1gOuterMEnd;
             fdS1gOuterMIdx++) { // 左闭右闭

            uint32_t actualGSplitSize = fd.fdBalanceMBaseSize;
            if (fdS1gOuterMIdx == fd.fdBalanceMSplitNum[fdTaskId] - 1) {
                actualGSplitSize = fd.fdBalanceMTailSize[fdTaskId];
            }
            uint32_t startRow = fdS1gOuterMIdx * fd.fdBalanceMBaseSize;

            LocalTensor<Type> lseExp = fdLseExpBuf.Get<Type>();
            LocalTensor<Type> reduceOut = fdReduceBuf.Get<Type>();

            WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_LSE_MAX_SUM_BUF1_FLAG + reduceMLoop % 2);
            CopyLseIn(startRow, actualGSplitSize, taskOffset, reduceMLoop);
            SetFlag<AscendC::HardEvent::MTE2_V>(SYNC_LSE_MAX_SUM_BUF1_FLAG + reduceMLoop % 2);
            WaitFlag<AscendC::HardEvent::MTE2_V>(SYNC_LSE_MAX_SUM_BUF1_FLAG + reduceMLoop % 2);

            LocalTensor<Type> mm2Res;
            for (uint32_t preLoadIdx = 0; preLoadIdx < preLoadNum; preLoadIdx++) {
                mm2Res = (reduceGlobaLoop + preLoadIdx) % 2 == 0 ? fdMm2ResBuf1.Get<Type>() : fdMm2ResBuf2.Get<Type>();
                WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_MM2RES_BUF1_FLAG + (reduceGlobaLoop + preLoadIdx) % 2);
                CopyAccumOutIn(mm2Res, preLoadIdx, taskOffset + startRow, actualGSplitSize);
                SetFlag<AscendC::HardEvent::MTE2_V>(SYNC_MM2RES_BUF1_FLAG + (reduceGlobaLoop + preLoadIdx) % 2);
            }

            ComputeScaleValue(lseExp, startRow, actualGSplitSize, reduceMLoop);
            // CalcPreNextTokens();
            SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_LSE_MAX_SUM_BUF1_FLAG + reduceMLoop % 2);

            for (uint32_t i = 0; i < taskInfo.actualCombineLoopSize; i++) {
                mm2Res = reduceGlobaLoop % 2 == 0 ? fdMm2ResBuf1.Get<Type>() : fdMm2ResBuf2.Get<Type>();
                if (i >= preLoadNum) {
                    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_MM2RES_BUF1_FLAG + reduceGlobaLoop % 2);
                    CopyAccumOutIn(mm2Res, i, taskOffset + startRow, actualGSplitSize);
                    SetFlag<AscendC::HardEvent::MTE2_V>(SYNC_MM2RES_BUF1_FLAG + reduceGlobaLoop % 2);
                }

                WaitFlag<AscendC::HardEvent::MTE2_V>(SYNC_MM2RES_BUF1_FLAG + reduceGlobaLoop % 2);
                ReduceFinalRes(reduceOut, mm2Res, lseExp, i, actualGSplitSize);
                SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_MM2RES_BUF1_FLAG + reduceGlobaLoop % 2);
                reduceGlobaLoop += 1;
            }
            CopyFinalResOut(reduceOut, startRow, actualGSplitSize, reduceMLoop);
            reduceMLoop += 1;
        }
        tmpFdS1gOuterMStart = 0;
    }
}

TEMPLATES_DEF 
class FiaBlockVecFlashDecodeDummy {
public:
    // =================================类型定义区=================================
    // 中间计算数据类型为float，高精度模式
    using Type = float;
    using OUT_T = bfloat16_t;   //问
    using FDGmType = typename std::conditional<isFd, GlobalTensor<float>, int8_t>::type;

    static constexpr AttentionCommon::FIA_LAYOUT LAYOUT_T = AttentionCommon::FIA_LAYOUT::BSND; //问
    // static constexpr GmFormat PostQuant_FORMAT = GmFormat::NGD;
    __aicore__ inline FiaBlockVecFlashDecodeDummy(){};
    __aicore__ inline void InitGlobalTensor(FDGmType lseMaxFdGm, FDGmType lseSumFdGm, FDGmType accumOutGm, 
        GlobalTensor<OUT_T> attentionOutGm, GlobalTensor<uint64_t> actualSeqLengthsGmQ, GlobalTensor<uint64_t> actualSeqLengthsGm){};
    // __aicore__ inline void InitSoftmaxLseGm(GlobalTensor<float> softmaxLseGm){};
    __aicore__ inline void InitParams(const ConstInfo<isInfer, hasRope> &constInfo){};
    __aicore__ inline void InitDecodeParams(){};
    __aicore__ inline void InitBuffers(TPipe *pipe){};
    __aicore__ inline void AllocEventID(){};
    __aicore__ inline void FreeEventID(){};   
    __aicore__ inline void FlashDecode(FDparams &fd){};
protected:
    __aicore__ inline void CopyAccumOutIn(LocalTensor<Type> &accumOutLocal, uint32_t splitKVIndex, uint32_t startRow,
                                          uint32_t dealRowCount){};                            
    __aicore__ inline void CopyLseIn(uint32_t startRow, uint32_t dealRowCount, uint64_t baseOffset, uint32_t cntM){};
    __aicore__ inline void ComputeScaleValue(LocalTensor<Type> &lseExp, uint32_t startRow, uint32_t dealRowCount,
                                             uint32_t cntM){};
    __aicore__ inline void Bmm2DataCopyOutTrans(LocalTensor<OUT_T> &attenOutUb, uint32_t startRow,
                                                 uint32_t dealRowCount, uint32_t columnCount){};
    __aicore__ inline void Bmm2DataCopyOut(uint64_t attenOutOffset, LocalTensor<OUT_T> &attenOutUb, uint32_t startRow,
                                           uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount){};
    __aicore__ inline void ReduceFinalRes(LocalTensor<Type> &reduceOut, LocalTensor<Type> &mm2Res, LocalTensor<Type> &lseLocal, 
                                          uint32_t cntKV, uint32_t dealRowCount){};
    __aicore__ inline void CopyFinalResOut(LocalTensor<Type> &accumOutLocal, uint32_t startRow, uint32_t dealRowCount,
                                           uint32_t cntM){};
    __aicore__ inline void CalcPreNextTokens(){};
    // __aicore__ inline void DealInvalidRows(LocalTensor<Type> &attenOutUb, uint32_t startRow, uint32_t dealRowCount,
    //                                        uint32_t columnCount){};
    // __aicore__ inline void DealInvalidMaskRows(LocalTensor<Type> &attenOutUb, uint32_t startRow, uint32_t dealRowCount,
    //                                            uint32_t columnCount, uint32_t cntM){};


};
}
#endif