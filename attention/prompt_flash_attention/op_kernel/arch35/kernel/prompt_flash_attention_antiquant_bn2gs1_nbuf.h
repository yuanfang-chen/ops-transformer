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
 * \file prompt_flash_attention_antiquant_bn2gs1_nbuf.h
 * \brief
 */
#ifndef PROMPT_FLASH_ATTENTION_ANTIQUANT_BN2GS1_NBUF_H
#define PROMPT_FLASH_ATTENTION_ANTIQUANT_BN2GS1_NBUF_H

#include "../comm/prompt_flash_attention_comm.h"
#include "../comm/prompt_flash_attention_sparse.h"
#include "../comm/prompt_flash_attention_kvcache.h"
#include "../service/prompt_flash_attention_mm1.h"
#include "../service/prompt_flash_attention_mm2.h"
#include "../service/prompt_flash_attention_vector1.h"
#include "../service/prompt_flash_attention_vector2.h"
#include "../service/prompt_flash_attention_antiquant_kv.h"
#include "../service/prompt_flash_attention_antiquant_preproc_q.h"
#include "../matmul_modules/pfa_matmul_policy.h"

using namespace matmul;

namespace {
    constexpr uint32_t BUF_SIZE_BYTE_1K = 1024;
    constexpr uint32_t BUF_SIZE_BYTE_4K = 4096;
    constexpr uint32_t BUF_SIZE_BYTE_8K = 8192;
    constexpr uint32_t BUF_SIZE_BYTE_16K = 16384;
    constexpr uint32_t BUF_SIZE_BYTE_32K = 32768;

    constexpr int32_t PFA_PRELOAD_TASK_CACHE_SIZE_ANTIQUANT = 3;
    constexpr int32_t ANTIQ_PRE_LOAD_NUM = 2;
    constexpr int32_t TASK_CACHE = 4;
    constexpr int32_t ANTIQ_INPUT_BUF_SIZE = 16384;

    typedef struct {
    uint32_t SOUTER;
    uint32_t D;
    uint32_t SINNER;   // sinner
    uint32_t M1;  // baseM
    uint32_t N1;
    uint32_t K1;
    uint32_t M2;
    uint32_t N2;
    uint32_t K2;
    } PFAProfile;

    static constexpr PFAProfile PFA_PROFILE_DEFAULT = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    static constexpr PFAProfile PFA_PROFILE_D64 = {32, 128, 128, 32, 64, 128, 32, 64, 128};
    static constexpr PFAProfile PFA_PROFILE_D128 = {32, 128, 128, 32, 128, 128, 32, 128, 128};

    // 生成matmul配置
    static constexpr MatmulConfig PfaGenConfMM1(const PFAProfile& pfa) {
        return {
            .doNorm = true,
            .doBasicBlock = false,
            .doMultiDataLoad = false,
            .basicM = pfa.M1,
            .basicN = pfa.N1,
            .basicK = pfa.K1,
            .intrinsicsCheck = false, 
            .isNBatch = false,
            .enVecND2NZ = false,
            .doSpecialBasicBlock = false,
            .doMTE2Preload = 0,
            .singleCoreM = pfa.M1,
            .singleCoreN = pfa.N1,
            .singleCoreK = pfa.K1,
            .stepM = 0,
            .stepN = 0,
            .baseMN = 0,
            .singleCoreMN = 0,
            .enUnitFlag = true,
            .isPerTensor = false,
            .hasAntiQuantOffset = false,
            .doIBShareNorm = false,
            .doSpecialMDL = false,
            .enableInit = true,
            .batchMode = BatchMode::BATCH_LESS_THAN_L1,
            .enableEnd = true,
            .enableGetTensorC = true,
            .enableSetOrgShape = true,
            .enableSetBias = true,
            .enableSetTail = true,
            .enableQuantVector = true,
            .enableSetDefineData = true,
            .iterateMode = IterateMode::ITERATE_MODE_DEFAULT,
            .enableReuse = true,
            .enableUBReuse = true,
            .enableL1CacheUB = false,
            .intraBlockPartSum = false,
            .iterateOrder = IterateOrder::UNDEF,
            .scheduleType = ScheduleType::INNER_PRODUCT,
            .enableDoubleCache = false,
            .isBiasBatch = true,
            .enableStaticPadZeros = false,
            .isA2B2Shared = false,
            .isC1Shared = false,
            .sharedC1BufferSize = 64 * 1024, // buffer size of matmul
        };
    }

    // 生成matmul配置
    static constexpr MatmulConfig PfaGenConfMM2(const PFAProfile& pfa) {
        return {
            .doNorm = true,
            .doBasicBlock = false,
            .doMultiDataLoad = false,
            .basicM = pfa.M2,
            .basicN = pfa.N2,
            .basicK = pfa.K2,
            .intrinsicsCheck = false,
            .isNBatch = false,
            .enVecND2NZ = false,
            .doSpecialBasicBlock = false,
            .doMTE2Preload = 0,
            .singleCoreM = pfa.M2,
            .singleCoreN = pfa.N2,
            .singleCoreK = pfa.K2,
            .stepM = 0,
            .stepN = 0,
            .baseMN = 0,
            .singleCoreMN = 0,
            .enUnitFlag = true,
            .isPerTensor = false,
            .hasAntiQuantOffset = false,
            .doIBShareNorm = false,
            .doSpecialMDL = false,
            .enableInit = true,
            .batchMode = BatchMode::BATCH_LESS_THAN_L1,
            .enableEnd = true,
            .enableGetTensorC = true,
            .enableSetOrgShape = true,
            .enableSetBias = true,
            .enableSetTail = true,
            .enableQuantVector = true,
            .enableSetDefineData = true,
            .iterateMode = IterateMode::ITERATE_MODE_DEFAULT,
            .enableReuse = true,
            .enableUBReuse = true,
            .enableL1CacheUB = false,
            .intraBlockPartSum = false,
            .iterateOrder = IterateOrder::UNDEF,
            .scheduleType = ScheduleType::INNER_PRODUCT,
            .enableDoubleCache = false,
            .isBiasBatch = true,
            .enableStaticPadZeros = false,
            .isA2B2Shared = false,
            .isC1Shared = false,
            .sharedC1BufferSize = 64 * 1024,  // buffer size of matmul
        };
    }
}

template <typename PFAT>
class PromptFlashAttentionAntiQuantKVBN2GS1 {
public:
    using T = typename PFAT::inputType;
    using KV_T = typename PFAT::kvInputType;
    using U = typename PFAT::maskType;
    using O = typename PFAT::outputType;
    using mmBiasType = typename PromptFlashAttentionTypeTraits<T,PFAT::calcMode>::mmBiasType;
    using mmOutputType = typename PromptFlashAttentionTypeTraits<T,PFAT::calcMode>::mmOutputType;
    using computeType = typename PromptFlashAttentionTypeTraits<T,PFAT::calcMode>::softmaxType;
    using pseShiftType = typename PromptFlashAttentionTypeTraits<T,PFAT::calcMode>::pseShiftType;
    using pseShiftCastType = typename PromptFlashAttentionTypeTraits<T,PFAT::calcMode>::pseShiftCastType;

    __aicore__ inline PromptFlashAttentionAntiQuantKVBN2GS1() {};
    __aicore__ inline void Init(__gm__ uint8_t* query, __gm__ uint8_t* key, __gm__ uint8_t* value,
                                __gm__ uint8_t* pseShift, __gm__ uint8_t* attenMask,
                                __gm__ uint8_t* actualSeqLengths, __gm__ uint8_t* actualSeqLengthsKV,
                                __gm__ uint8_t* blocktable, __gm__ uint8_t* queryPaddingSize,
                                __gm__ uint8_t* kvPaddingSize, __gm__ uint8_t* attentionOut,
                                __gm__ uint8_t* softmaxLse, __gm__ uint8_t* workspace,
                                const PromptFlashAttentionTilingData* __restrict tiling, TPipe* tPipe);
    __aicore__ inline void InitQuant(__gm__ uint8_t* deq_scale1, __gm__ uint8_t* scale1, __gm__ uint8_t* deq_scale2,
                                     __gm__ uint8_t* scale2, __gm__ uint8_t* offset2);
    __aicore__ inline void Process();
    __aicore__ inline void InitKvAntiquantSeparate(__gm__ uint8_t* key_antiquant_scale, __gm__ uint8_t* key_antiquant_offset, __gm__ uint8_t* value_antiquant_scale, __gm__ uint8_t* value_antiquant_offset);

    using a1Type = MatmulType<TPosition::TSCM, CubeFormat::NZ, T, false, LayoutMode::NONE, false, TPosition::VECOUT>;
    using b1Type = MatmulType<TPosition::TSCM, CubeFormat::NZ, T, true, LayoutMode::NONE, false, TPosition::VECOUT>;
    using bias1Type = MatmulType<TPosition::GM, CubeFormat::ND, mmBiasType>;
    using c1Type = MatmulType<TPosition::VECCALC, CubeFormat::ND_ALIGN, mmOutputType>;

    static constexpr auto mm1TilingPfa = GetMatmulApiTiling<a1Type, b1Type, c1Type, bias1Type>(PfaGenConfMM1(PFA_PROFILE_D128));

    using mm1Type = Matmul<a1Type, b1Type, c1Type, bias1Type, mm1TilingPfa>;

    using a2Type = MatmulType<TPosition::VECCALC, CubeFormat::NZ, T, false, LayoutMode::NONE, false>;
    using b2Type = MatmulType<TPosition::TSCM, CubeFormat::NZ, T, false, LayoutMode::NONE, false, TPosition::VECOUT>;
    using bias2Type = MatmulType<TPosition::GM, CubeFormat::ND, mmBiasType>;

    using c2Type = MatmulType<TPosition::VECCALC, CubeFormat::ND, mmOutputType>;

    static constexpr auto mm2TilingPfa = GetMatmulApiTiling<a2Type, b2Type, c2Type, bias2Type>(PfaGenConfMM2(PFA_PROFILE_D128));
    using mm2Type = Matmul<a2Type, b2Type, c2Type, bias2Type, mm2TilingPfa>;

    PromptFlashAttentionNormalMM1<PFAT, mm1Type> mm1;
    PromptFlashAttentionNormalMM2<PFAT, mm2Type> mm2;
    PromptFlashAttentionNormalVector1<PFAT> vector1;
    PromptFlashAttentionNormalVector2<PFAT> vector2;

    PromptFlashAttentionAntiQuantKV<PFAT> antiquant;
    PromptFlashAttentionAntiQuantPreProcessQ<PFAT> processQuery;

protected:
    __aicore__ inline void InitOutputSingleCore();
    __aicore__ inline void AllocGlobalResources();
    __aicore__ inline void FreeGlobalResources();
    __aicore__ inline void ComputeEachCore(uint32_t coreIdx);
    __aicore__ inline void IterateAllMm1(TaskParam& taskParam);
    __aicore__ inline void WaitIterateAllMm1(TaskParam& taskParam);
    __aicore__ inline void IterateAllMm2(TaskParam& taskParam);
    __aicore__ inline void WaitIterateAllMm2(TaskParam& taskParam);
    __aicore__ inline void ProcVector1(TaskParam& taskParam);
    __aicore__ inline void ProcVector2(TaskParam& taskParam);
    __aicore__ inline void AntiquantKey(TaskParam& task, int32_t loop);
    __aicore__ inline void AntiquantValue(TaskParam& task, int32_t loop);
    __aicore__ inline void CopyQueryToL1(TaskParam& task, int32_t loop);
    __aicore__ inline void LaunchTaskQuantPipe(uint64_t taskIdx, int32_t endIndex);

protected:
    TPipe* pipe;
    const PromptFlashAttentionTilingData* __restrict tilingData;

    // GM
    __gm__ uint8_t* currentKey;
    __gm__ uint8_t* currentValue;
    __gm__ uint8_t* blocktablePtr;

    GlobalTensor<T> queryGm;
    GlobalTensor<KV_T> keyGm;
    GlobalTensor<KV_T> valueGm;
    GlobalTensor<O> attentionOutGm;

    GlobalTensor<int64_t> actualSeqLengthsGm;
    GlobalTensor<int64_t> actualSeqLengthsKVGm;

    GlobalTensor<uint32_t> deqScale1Fp32Gm;
    GlobalTensor<uint32_t> deqScale2Fp32Gm;
    
    GlobalTensor<bfloat16_t> quantScale2BF16Gm;
    GlobalTensor<bfloat16_t> quantOffset2BF16Gm;

    // define the que
    TQue<QuePosition::VECIN, 1> tempBmm2Queue;
    TQue<QuePosition::VECIN, 1> Bmm2Queue;
    TQue<QuePosition::VECOUT, 1> Bmm1Queue;
    TQue<QuePosition::VECIN, 1> maskQueue;
    TQue<QuePosition::VECIN, 1> pseQueue;
    TQue<QuePosition::VECOUT, 2> Bmm1CastUb; //2: block of ub

    TSCM<QuePosition::VECIN, 1, 0x4> bmm2Scm[2]; //2: block of L1

    TBuf<> softmaxApiBuffer;
    TBuf<> softmaxMaxBuffer;
    TBuf<> softmaxSumBuffer[2]; //2: block of ub
    TBuf<> softmaxExpBuffer;
    TBuf<> tempBmm2Ub;
    TBuf<> quantScale2BF16Ub;
    TBuf<> quantOffset2BF16Ub;
    TBuf<> quantScale2FloatUb;
    TBuf<> quantOffset2FloatUb;

    // antiquant GM buffer
    GlobalTensor<T> keyAntiquantScaleGm;
    GlobalTensor<T> keyAntiquantOffsetGm;
    GlobalTensor<T> valueAntiquantScaleGm;
    GlobalTensor<T> valueAntiquantOffsetGm;

    // antiquant L1 buffer
    TSCM<QuePosition::VECIN, 1, 0x4> queryScmQueue;
    TSCM<QuePosition::VECIN, 1, 0x4> keyValueScmQueue;

    // antiquant ub buffer
    TQue<QuePosition::VECIN, 1> queryInputQueue;
    TQue<QuePosition::VECOUT, 1> queryOutputQueue;
    TQue<QuePosition::VECIN, 1> antiquantInputQueue;
    TQue<QuePosition::VECOUT, 1> antiquantOutputQueue;
    TQue<QuePosition::VECIN, 1> antiquantParamQueue;

    LocalTensor<computeType> mmResUb[2]; //2: block of ub
    LocalTensor<T> tmpSoftmaxResUb[2]; //2: block of ub
    LocalTensor<computeType> mm2ResPQUb[2]; //2: block of ub
    LocalTensor<uint8_t> softmaxApiUb;
    LocalTensor<float> softmaxMaxUb;
    LocalTensor<float> softmaxSumUb[2]; //2: block of ub
    LocalTensor<computeType> softmaxExpUb;

    uint64_t dequantScale1;
    float quantScale1;
    uint64_t dequantScale2;
    float quantScale2;
    float quantOffset2;
    bool isQuant2PerChn = false;
    bool isQuantOffset2Exit = false;
    uint32_t perChannelQuantUBSize = 0;

    ConstParam constParam;
    TaskManager<PFA_PRELOAD_TASK_CACHE_SIZE_ANTIQUANT> taskManager;
    __gm__ uint8_t* key_ptr;
    __gm__ uint8_t* value_ptr;
};

template<typename PFAT>
__aicore__ inline void PromptFlashAttentionAntiQuantKVBN2GS1<PFAT>::Init(__gm__ uint8_t* query, __gm__ uint8_t* key,
    __gm__ uint8_t* value, __gm__ uint8_t* pseShift, __gm__ uint8_t* attenMask, __gm__ uint8_t* actualSeqLengths,
    __gm__ uint8_t* actualSeqLengthsKV, __gm__ uint8_t* blocktable, __gm__ uint8_t* queryPaddingSize,
    __gm__ uint8_t* kvPaddingSize, __gm__ uint8_t* attentionOut, __gm__ uint8_t* softmaxLse, __gm__ uint8_t* workspace,
    const PromptFlashAttentionTilingData* __restrict tiling, TPipe* tPipe)
{
    pipe = tPipe;

    // init const param
    tilingData = tiling;
    InitConstParam<PFAT>(constParam, tilingData);
    key_ptr = key;
    value_ptr = value;

    // init GM
    queryGm.SetGlobalBuffer((__gm__ T*)query);
    attentionOutGm.SetGlobalBuffer((__gm__ O*)attentionOut);

    if (tilingData->promptAttentionBaseParams.fromFused) {
        ListTensorDesc keyListTensorDescInit((__gm__ void*)key);
        ListTensorDesc valueListTensorDescInit((__gm__ void*)value);
        currentKey = (__gm__ uint8_t*)keyListTensorDescInit.GetDataPtr<__gm__ uint8_t>(0);
        currentValue = (__gm__ uint8_t*)valueListTensorDescInit.GetDataPtr<__gm__ uint8_t>(0);
        if constexpr (PFAT::MM_TYPE == PFAMatMulType::MM_PA) {
            blocktablePtr = blocktable;
        }
        keyGm.SetGlobalBuffer((__gm__ KV_T*)currentKey);
        valueGm.SetGlobalBuffer((__gm__ KV_T*)currentValue);
    } else {
        keyGm.SetGlobalBuffer((__gm__ KV_T*)key);
        valueGm.SetGlobalBuffer((__gm__ KV_T*)value);
    }

    if (!constParam.isActualLenDimsNull) {
        actualSeqLengthsGm.SetGlobalBuffer((__gm__ int64_t*)actualSeqLengths,
                                           tilingData->promptAttentionBaseParams.batchSize);
    }
    if (!constParam.isActualLenDimsKVNull) {
        actualSeqLengthsKVGm.SetGlobalBuffer((__gm__ int64_t*)actualSeqLengthsKV,
                                             tilingData->promptAttentionBaseParams.batchSize);
    }

    // init UB
    pipe->InitBuffer(tempBmm2Ub, ANTIQUANT_SOUTER_BASE_SIZE * ANTIQUANT_SINNER_BASE_SIZE * sizeof(computeType));
    pipe->InitBuffer(Bmm1CastUb, 2, (ANTIQUANT_SOUTER_BASE_SIZE + 1) * ANTIQUANT_SINNER_BASE_SIZE * sizeof(T)); // ping pong形式避免读写冲突
    pipe->InitBuffer(softmaxApiBuffer, 2 * ANTIQUANT_SOUTER_BASE_SIZE * ANTIQUANT_SOFTMAX_COLUMN_SIZE * sizeof(computeType));
    pipe->InitBuffer(softmaxMaxBuffer, ANTIQUANT_SOUTER_BASE_SIZE * ANTIQUANT_SOFTMAX_COLUMN_SIZE * sizeof(float));
    pipe->InitBuffer(softmaxSumBuffer[0], ANTIQUANT_SOUTER_BASE_SIZE * ANTIQUANT_SOFTMAX_COLUMN_SIZE * sizeof(float));
    pipe->InitBuffer(softmaxSumBuffer[1], ANTIQUANT_SOUTER_BASE_SIZE * ANTIQUANT_SOFTMAX_COLUMN_SIZE * sizeof(float));
    pipe->InitBuffer(softmaxExpBuffer, ANTIQUANT_SOUTER_BASE_SIZE * ANTIQUANT_SOFTMAX_COLUMN_SIZE * sizeof(computeType));
    pipe->InitBuffer(Bmm2Queue, 2, ANTIQUANT_SOUTER_BASE_SIZE * ANTIQUANT_SINNER_BASE_SIZE * sizeof(computeType));
    pipe->InitBuffer(Bmm1Queue, 2, ANTIQUANT_SOUTER_BASE_SIZE * ANTIQUANT_SINNER_BASE_SIZE * sizeof(computeType));
    if constexpr (PFAT::isBand) {
        pipe->InitBuffer(maskQueue, 1, ANTIQUANT_SOUTER_BASE_SIZE * ANTIQUANT_SINNER_BASE_SIZE * sizeof(U) * 2); // 2: band mode needs 2 mask
    } else {
        pipe->InitBuffer(maskQueue, 1, ANTIQUANT_SOUTER_BASE_SIZE * ANTIQUANT_SINNER_BASE_SIZE * sizeof(U));
    }
    if constexpr (PFAT::isHasPse) {
        pipe->InitBuffer(pseQueue, 1, ANTIQUANT_SOUTER_BASE_SIZE * ANTIQUANT_SINNER_BASE_SIZE * sizeof(T));
    }

    softmaxApiUb = softmaxApiBuffer.Get<uint8_t>();
    softmaxMaxUb = softmaxMaxBuffer.Get<float>(ANTIQUANT_SOUTER_BASE_SIZE * ANTIQUANT_SOFTMAX_COLUMN_SIZE);
    softmaxSumUb[0] = softmaxSumBuffer[0].Get<float>(ANTIQUANT_SOUTER_BASE_SIZE * ANTIQUANT_SOFTMAX_COLUMN_SIZE);
    softmaxSumUb[1] = softmaxSumBuffer[1].Get<float>(ANTIQUANT_SOUTER_BASE_SIZE * ANTIQUANT_SOFTMAX_COLUMN_SIZE);
    softmaxExpUb = softmaxExpBuffer.Get<computeType>(ANTIQUANT_SOUTER_BASE_SIZE * ANTIQUANT_SOFTMAX_COLUMN_SIZE);

    if (tilingData->promptAttentionInitOutputParams.needInit == 1) {
        InitOutputSingleCore();
    }

    // init service
    mm2.Init();
    vector1.Init(attenMask, pseShift);
    vector2.Init(attentionOut);
}

template<typename PFAT>
__aicore__ inline void PromptFlashAttentionAntiQuantKVBN2GS1<PFAT>::InitKvAntiquantSeparate(__gm__ uint8_t* key_antiquant_scale, __gm__ uint8_t* key_antiquant_offset, __gm__ uint8_t* value_antiquant_scale, __gm__ uint8_t* value_antiquant_offset)
{
    // 初始化L1 buffer
    pipe->InitBuffer(queryScmQueue, 2, BUF_SIZE_BYTE_8K);
    pipe->InitBuffer(keyValueScmQueue, 2, BUF_SIZE_BYTE_32K);

    // 初始化UB buffer
    pipe->InitBuffer(queryInputQueue, 1, BUF_SIZE_BYTE_8K);
    pipe->InitBuffer(queryOutputQueue, 1, BUF_SIZE_BYTE_8K);
    pipe->InitBuffer(antiquantInputQueue, 4, ANTIQ_INPUT_BUF_SIZE); // 4:block of ub
    pipe->InitBuffer(antiquantOutputQueue, 2, BUF_SIZE_BYTE_32K); // 2:block of ub
    pipe->InitBuffer(antiquantParamQueue, 1, BUF_SIZE_BYTE_1K);

    constParam.isAntiquantSymmetric = true;

    if (key_antiquant_scale != nullptr) {
        keyAntiquantScaleGm.SetGlobalBuffer((__gm__ T*)key_antiquant_scale);
    }

    if (key_antiquant_offset != nullptr) {
        constParam.isAntiquantSymmetric = false;
        keyAntiquantOffsetGm.SetGlobalBuffer((__gm__ T*)key_antiquant_offset);
    }

    if (value_antiquant_scale != nullptr) {
        valueAntiquantScaleGm.SetGlobalBuffer((__gm__ T*)value_antiquant_scale);
    }

    if (value_antiquant_offset != nullptr) {
        constParam.isAntiquantSymmetric = false;
        valueAntiquantOffsetGm.SetGlobalBuffer((__gm__ T*)value_antiquant_offset);
    }
}

template<typename PFAT>
__aicore__ inline void PromptFlashAttentionAntiQuantKVBN2GS1<PFAT>::AntiquantKey(TaskParam& task, int32_t loop)
{
    antiquant.Process(keyValueScmQueue,
                      keyGm,
                      keyAntiquantScaleGm,
                      keyAntiquantOffsetGm,
                      antiquantInputQueue,
                      antiquantOutputQueue,
                      antiquantParamQueue,
                      task, constParam,
                      tilingData);
}

template<typename PFAT>
__aicore__ inline void PromptFlashAttentionAntiQuantKVBN2GS1<PFAT>::AntiquantValue(TaskParam& task, int32_t loop)
{
    antiquant.Process(keyValueScmQueue,
                      valueGm,
                      valueAntiquantScaleGm,
                      valueAntiquantOffsetGm,
                      antiquantInputQueue,
                      antiquantOutputQueue,
                      antiquantParamQueue,
                      task, constParam,
                      tilingData);
}

template<typename PFAT>
__aicore__ inline void PromptFlashAttentionAntiQuantKVBN2GS1<PFAT>::CopyQueryToL1(TaskParam& task, int32_t loop)
{
    processQuery.Process(queryScmQueue, queryGm, queryInputQueue, queryOutputQueue, task, constParam, tilingData);
}

template<typename PFAT>
__aicore__ inline void PromptFlashAttentionAntiQuantKVBN2GS1<PFAT>::InitOutputSingleCore()
{
    auto &initParams = tilingData->promptAttentionInitOutputParams;
    uint32_t tailSize = initParams.totalOutputSize - constParam.tmpBlockIdx * initParams.singleCoreSize;
    uint32_t singleInitOutputSize = tailSize < initParams.singleCoreSize ? tailSize : initParams.singleCoreSize;
    InitOutput<O>(attentionOutGm[constParam.tmpBlockIdx * static_cast<int64_t>(initParams.singleCoreSize)], singleInitOutputSize, 0);
    SyncAll();
}

template<typename PFAT>
__aicore__ inline void PromptFlashAttentionAntiQuantKVBN2GS1<PFAT>::Process()
{
    AllocGlobalResources();
    ComputeEachCore(constParam.tmpBlockIdx);
    FreeGlobalResources();
}

template<typename PFAT>
__aicore__ inline void PromptFlashAttentionAntiQuantKVBN2GS1<PFAT>::ComputeEachCore(uint32_t coreIdx)
{
    int actualCoreNums = this->tilingData->promptAttentionSingleCoreParams.actualCoreNums;

    if (g_coreType == AIV && coreIdx >= actualCoreNums) {
        return;
    }
    bool splitPingPong = false;          // 用于两轮sinner之间的ping pong
    bool taskPingPong = false;           // 用于sinner内每个task之间的ping pong
    uint64_t taskIdx = 0;
    RunParam runParam;
    ComputeParamCore<PFAT>(runParam, constParam, tilingData, coreIdx);

    for (int32_t sIdx = constParam.sIdStart; sIdx < constParam.sIdEnd; sIdx++) {
        ComputeParamN<PFAT>(runParam, constParam, sIdx);
        for (int32_t loopNIdx = constParam.nLoopStart; loopNIdx < runParam.tmpNLoopEnd; loopNIdx++) {
            runParam.batchNOffset = loopNIdx;
            ComputeParamBatch<PFAT>(runParam, constParam, sIdx, keyGm, actualSeqLengthsGm, actualSeqLengthsKVGm);
            ComputeS1LoopInfo<PFAT>(runParam, constParam, loopNIdx);

            for (int32_t sOuterLoopIdx = constParam.outerLoopStart;
                 sOuterLoopIdx < runParam.tmpOuterLoopEnd;
                 sOuterLoopIdx++) {
                bool s1NeedCalc = ComputeParamS1<PFAT>(runParam, constParam, sIdx, sOuterLoopIdx);
                bool s2NeedCalc = ComputeS2LoopInfo<PFAT>(runParam, constParam);
                bool lastLoopThisCore = (loopNIdx == runParam.tmpNLoopEnd - 1) && (sIdx == constParam.sIdEnd - 1) && \
                    (sOuterLoopIdx == runParam.tmpOuterLoopEnd - 1);
                // s1和s2有任意一个不需要算, 则continue, 如果时当前核最后一次循环，则补充计算taskIdx+2的部分
                if ((!s1NeedCalc || !s2NeedCalc) && !lastLoopThisCore) {
                    continue;
                }

                int32_t sInnerEndIndex = runParam.endIndex;
                if (lastLoopThisCore) {
                    sInnerEndIndex += 4; // 4:for the tail of the inner loop
                }

                splitPingPong ^= 1;
                for (int32_t sInnerLoopIdx = runParam.startIndex;
                     sInnerLoopIdx < sInnerEndIndex;
                     sInnerLoopIdx++) {
                    taskIdx++;
                    taskPingPong ^= 1;

                    TaskParam &taskParam = taskManager.GetTaskRef(taskIdx);
                    taskParam.isValid = (sInnerLoopIdx < runParam.endIndex);
                    taskParam.splitPingPong = splitPingPong;
                    taskParam.taskPingPong = taskPingPong;
                    InitTaskParamByRun<PFAT>(taskParam, runParam);
                    ComputeParamS2<PFAT>(taskParam, runParam, constParam, sInnerLoopIdx);
                    LaunchTaskQuantPipe(taskIdx, runParam.endIndex);
                }
            }
            constParam.outerLoopStart = 0;
        }
        constParam.nLoopStart = 0;
    }
}

template<typename PFAT>
__aicore__ inline void PromptFlashAttentionAntiQuantKVBN2GS1<PFAT>::LaunchTaskQuantPipe(uint64_t taskIdx, int32_t endIndex)
{
    TaskParam &taskQuant = taskManager.GetTaskRef(taskIdx);

    if (taskQuant.singleProcessSInnerSizeNow == 0 || taskQuant.cubeSOuterSize == 0) {
        taskQuant.isValid = false;
    }

    if (taskIdx % ANTIQ_PRE_LOAD_NUM == 0) {
        for (uint32_t i = taskIdx - ANTIQ_PRE_LOAD_NUM; i < taskIdx; i++) {
            if (i - ANTIQ_PRE_LOAD_NUM >= 0) {
                TaskParam &task = taskManager.GetTaskRef(i - ANTIQ_PRE_LOAD_NUM);
                if (task.isValid) {
                    WaitIterateAllMm2(task);
                    ProcVector2(task);
                    task.isValid = false;
                }
            }
            TaskParam &task = taskManager.GetTaskRef(i);
            if (task.isValid) {
                AntiquantKey(task, i);
                CopyQueryToL1(task, i);
                IterateAllMm1(task);
            }
        }

        for (uint32_t i = taskIdx - ANTIQ_PRE_LOAD_NUM; i < taskIdx; i++) {
            TaskParam &task = taskManager.GetTaskRef(i);
            if (task.isValid) {
                WaitIterateAllMm1(task);
                ProcVector1(task);
                AntiquantValue(task, i);
                IterateAllMm2(task);
            }
        }

        if ((endIndex <= ANTIQ_PRE_LOAD_NUM) && (taskIdx == ANTIQ_PRE_LOAD_NUM)) {
            for (uint32_t i = 0; i < ANTIQ_PRE_LOAD_NUM; i++) {
                TaskParam &task = taskManager.GetTaskRef(i);
                if (task.isValid) {
                    WaitIterateAllMm2(task);
                    ProcVector2(task);
                    task.isValid = false;
                }
            }
        }
    }
}

template<typename PFAT>
__aicore__ inline void PromptFlashAttentionAntiQuantKVBN2GS1<PFAT>::IterateAllMm1(TaskParam& taskParam)
{
    mm1.IterateAll(mmResUb[taskParam.taskPingPong], queryScmQueue, keyValueScmQueue, taskParam, constParam, antiquantOutputQueue);
}

template<typename PFAT>
__aicore__ inline void PromptFlashAttentionAntiQuantKVBN2GS1<PFAT>::WaitIterateAllMm1(TaskParam& taskParam)
{
    mm1.WaitIterateAll();
}

template<typename PFAT>
__aicore__ inline void PromptFlashAttentionAntiQuantKVBN2GS1<PFAT>::IterateAllMm2(TaskParam& taskParam)
{
    mm2.IterateAll(mm2ResPQUb[taskParam.taskPingPong], tmpSoftmaxResUb[taskParam.taskPingPong], bmm2Scm[taskParam.taskPingPong],
                   keyValueScmQueue, taskParam, constParam);
}

template<typename PFAT>
__aicore__ inline void PromptFlashAttentionAntiQuantKVBN2GS1<PFAT>::WaitIterateAllMm2(TaskParam& taskParam)
{
    mm2.WaitIterateAll();
}

template<typename PFAT>
__aicore__ inline void PromptFlashAttentionAntiQuantKVBN2GS1<PFAT>::ProcVector1(TaskParam& taskParam)
{
    vector1.ProVector(tmpSoftmaxResUb[taskParam.taskPingPong], maskQueue, pseQueue, mmResUb[taskParam.taskPingPong],
        softmaxMaxUb, softmaxSumUb[taskParam.splitPingPong], softmaxExpUb, softmaxApiUb, taskParam, constParam, 0, 0);
}

template<typename PFAT>
__aicore__ inline void PromptFlashAttentionAntiQuantKVBN2GS1<PFAT>::ProcVector2(TaskParam& taskParam)
{
    LocalTensor<computeType> bmm2ResPreUb = tempBmm2Ub.Get<computeType>(
        constParam.singleProcessSOuterSizeWhole * constParam.headSize);

    vector2.ProVector(bmm2ResPreUb, softmaxSumUb[taskParam.splitPingPong], mm2ResPQUb[taskParam.taskPingPong],
        softmaxExpUb, taskParam, constParam, 0);
}

template<typename PFAT>
__aicore__ inline void PromptFlashAttentionAntiQuantKVBN2GS1<PFAT>::AllocGlobalResources()
{
    for (int i = 0; i < 2; ++i) { // 2:block of ub
        this->mmResUb[i] = this->Bmm1Queue.template AllocTensor<computeType>();
        this->mm2ResPQUb[i] = this->Bmm2Queue.template AllocTensor<computeType>();
        this->tmpSoftmaxResUb[i] = this->Bmm1CastUb.template AllocTensor<T>();
    }
}

template<typename PFAT>
__aicore__ inline void PromptFlashAttentionAntiQuantKVBN2GS1<PFAT>::FreeGlobalResources()
{
    for (int i = 0; i < 2; ++i) { // 2:block of ub
        this->Bmm1Queue.FreeTensor(this->mmResUb[i]);
        this->Bmm2Queue.FreeTensor(this->mm2ResPQUb[i]);
        this->Bmm1CastUb.FreeTensor(this->tmpSoftmaxResUb[i]);
    }
}

#endif  // PROMPT_FLASH_ATTENTION_ANTIQUANT_BN2GS1_NBUF_H