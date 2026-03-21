/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_CUSTOM_SAB_VEC_HPP
#define CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_CUSTOM_SAB_VEC_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "kernel_operator.h"

using AscendC::CopyRepeatParams;
using AscendC::DataCopyExtParams;
using AscendC::DataCopyParams;
using AscendC::GetBlockIdx;
using AscendC::GlobalTensor;
using AscendC::LocalTensor;
using AscendC::QuePosition;
using AscendC::RoundMode;
using AscendC::TBuf;
using AscendC::TQue;

namespace Catlass::Epilogue::Block {

struct DBParams {
  int64_t blockId;
  int64_t taskId;
  int64_t bIdx;
  int64_t n2Idx;
  int64_t s2oIdx;
  int64_t gIdx;
  int64_t s1oIdx;
  int32_t s1CvExtend;
  int32_t s2CvExtend;
  int32_t s1CvExtendAlign;
  int32_t s2CvExtendAlign;
  int64_t aTensorOffsetCv{0};
  int64_t aTensorOffsetCv_rope{0};
  int64_t bTensorOffsetCv{0};
  int64_t bTensorOffsetCv_rope{0};
  int64_t actualS1Len{0};
  int64_t actualS2Len{0};
  int64_t s1Stride;
  int64_t s2Stride;
  int64_t blockIdArr[24];
  int32_t s1CvExtendArr[24];
  int32_t s2CvExtendArr[24];
  int8_t dqGroupId[24];
  int8_t kvGroupId[24];
};

template <
    class OutputType_,
    class UpdateType_,
    class InputType_,
    const uint32_t IS_ATTEN_MASK = 0,
    const uint32_t IS_PSE = 1,
    const uint32_t IS_DROP = 1,
    const AscendC::CubeFormat MM_OUT_FORMAT = AscendC::CubeFormat::ND,
    const uint32_t INPUT_LAYOUT = 0,
    const AscendC::CubeFormat MM2_OUT_FORMAT = AscendC::CubeFormat::NZ,
    const uint32_t IS_DTM = 0,
    const AscendC::STemplateType S1TEMPLATETYPE = AscendC::STemplateType::NotAligned,
    const AscendC::STemplateType S2TEMPLATETYPE = AscendC::STemplateType::NotAligned,
    const AscendC::DTemplateType DTEMPLATETYPE = AscendC::DTemplateType::NotAligned,
    const uint32_t HAS_ROPE = 0>
class BlockEpilogue<
    EpilogueAtlasA2CustomSabVec,
    OutputType_,
    UpdateType_,
    InputType_,
    IS_ATTEN_MASK,
    IS_PSE,
    IS_DROP,
    MM_OUT_FORMAT,
    INPUT_LAYOUT,
    MM2_OUT_FORMAT,
    IS_DTM,
    S1TEMPLATETYPE,
    S2TEMPLATETYPE,
    DTEMPLATETYPE,
    HAS_ROPE>
{
public:
    using DispatchPolicy = EpilogueAtlasA2CustomSabVec;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using T1 = OutputType_;
    using T2 = UpdateType_;
    static constexpr uint32_t isAttenMask = IS_ATTEN_MASK;
    static constexpr uint32_t isPse = IS_PSE;
    static constexpr uint32_t isDrop = IS_DROP;
    static constexpr AscendC::CubeFormat mmOutFormat = MM_OUT_FORMAT;
    static constexpr uint32_t inputLayout = INPUT_LAYOUT;
    static constexpr AscendC::CubeFormat mm2OutFormat = MM2_OUT_FORMAT;
    static constexpr uint32_t isDtm = IS_DTM;
    static constexpr AscendC::STemplateType s1TemplateType = S1TEMPLATETYPE;
    static constexpr AscendC::STemplateType s2TemplateType = S2TEMPLATETYPE;
    static constexpr AscendC::DTemplateType dTemplateType = DTEMPLATETYPE;
    static constexpr uint32_t hasRope = HAS_ROPE;

    AscendC::TPipe *pipe;
    TBuf<> unifiedBuffer;

    uint32_t coreNum;
    uint32_t cubeCoreNum;
    uint32_t cBlockIdx;
    uint32_t cCubeBlockIdx;
    uint32_t cSubIdx;

    uint32_t vecBlockNum;

    GlobalTensor<T1> keyGm, valueGm, dxGm, queryGm, forwardResGm, pseGm;
    GlobalTensor<T1> keyRopeGm;
    GlobalTensor<T1> queryRopeGm;
    GlobalTensor<uint8_t> maskWorkSpaceGm, attenMaskU8Gm, dropMaskGm;
    GlobalTensor<float> softmaxMaxGm, softmaxSumGm, sinkGm;

    GlobalTensor<float> dqWorkSpaceGm, dkWorkSpaceGm, dvWorkSpaceGm, sfmgWorkspaceGm, dqDtmWsGm, dkDtmWsGm, dvDtmWsGm;
    GlobalTensor<float> dqRopeWorkSpaceGm;
    GlobalTensor<float> dkRopeWorkSpaceGm;
    GlobalTensor<float> dqRopeDtmWsGm;
    GlobalTensor<float> dkRopeDtmWsGm;

    GlobalTensor<T1> dropWorkSpaceGm, mulWorkSpaceGm;

    GlobalTensor<T2> mm1WorkspaceGm;
    GlobalTensor<T2> mm2WorkspaceGm;
    GlobalTensor<float> dsinksumWorkSpaceGm;
    GlobalTensor<uint32_t> dsinksumDataSizeGm;

    TBuf<AscendC::TPosition::A1> queryBufL1;
    LocalTensor<T1> qL1Tensor;
    TBuf<AscendC::TPosition::A1> keyBufL1;
    LocalTensor<T1> vL1Tensor;
    LocalTensor<T1> kL1Tensor;
    TBuf<AscendC::TPosition::A1> dsBufL1;
    LocalTensor<T1> dsL1Tensor;
    LocalTensor<T1> dxL1Tensor;

    __gm__ uint8_t *prefixN_addr;
    __gm__ uint8_t *actual_seq_qlen_addr;
    __gm__ uint8_t *actual_seq_kvlen_addr;

    GM_ADDR workspaceAddr;

    GlobalTensor<int32_t> syncGlobal;

    GlobalTensor<half> pseAlibiGm;
    GlobalTensor<float> dvGm;
    __gm__ uint8_t *pseSlope;

    AscendC::PseInfo pseInfo = {0};

    constexpr static uint32_t BNGSD = 0;
    constexpr static uint32_t SBNGD = 1;
    constexpr static uint32_t BSNGD = 2;
    constexpr static uint32_t TND = 3;
    constexpr static uint32_t ENABLE = 1;

    constexpr static uint64_t SYNC_MODE2 = 2;
    static constexpr uint64_t SYNC_V1_C2_FLAG[3] = {4, 5, 6};
    static constexpr uint64_t SYNC_C1_V1_FLAG[3] = {1, 2, 3};
    static constexpr uint64_t SYNC_C2_V1_FLAG[3] = {7, 8, 9};

    float keepProb;
    int64_t s1Token;
    int64_t s2Token;
    int64_t actualCalcS1Token;
    int64_t actualCalcS2Token;
    uint32_t sparseMode;
    bool dropBitMode;

    int64_t b;
    int64_t n2;
    int64_t g;
    int64_t s1;
    int64_t s2;
    int64_t d;
    int64_t rope_d = 0;
    int64_t value_d;
    int64_t dAlign;
    int64_t rope_dAlign = 0;
    int64_t value_dAlign;
    int64_t attenMaskDimS2;

    uint32_t baseMN;
    uint32_t cubeBaseMN;

    int64_t s1Outer;
    uint32_t s1CvInner;
    uint32_t s1CvTail;
    int64_t s2Outer;
    uint32_t s2CvInner;
    uint32_t s2CvTail;

    int64_t sfmgOffset = 0;
    uint32_t preS1Idx = -1;

    int64_t baseIdx{0};
    int64_t bDimIdx{0};
    int64_t n2DimIdx{0};
    int64_t gDimIdx{0};
    int64_t s1oDimIdx{0};
    int64_t s2oCvDimIdx{0};

    int32_t isStart = 1;
    uint32_t pingpongIdx = 1;
    int32_t vecLoopStart;
    int32_t vecLoopEnd;

    uint32_t s1VecLoop = 0;
    uint32_t s1VecSize = 0;
    uint32_t s1ExtendSubGraph = 0;
    uint32_t s2Extend = 0;
    uint32_t s2ExtendAlign = 0;
    uint32_t s2VecLoop = 0;
    uint32_t s2VecSize = 0;

    int64_t dqOutBase{0};
    int64_t kvOutBase{0};
    int64_t dqOutIdx{0};
    int64_t kvOutIdx{0};
    int64_t dqOutArr[24];
    int64_t kvOutArr[24];

    DBParams dbParams[3];
    int64_t blockStartIdx = 0;

    int64_t bandIdx = 0;

    AscendC::DropMaskInfo dropMaskInfo = {0};

    constexpr static uint32_t T2Begin = 0;
    constexpr static uint32_t T1Begin = 33 * 1024;
    constexpr static uint32_t BoolBegin = 50 * 1024;
    constexpr static uint32_t T2BlockBegin = 58 * 1024;
    constexpr static uint32_t U8Begin = 66 * 1024;
    constexpr static uint32_t DbBegin = 74 * 1024;

    constexpr static uint32_t DTYPE_FACTOR = sizeof(T2) / sizeof(T1);
    constexpr static uint32_t cal_block_num = 32 / sizeof(T2);
    constexpr static uint32_t cal_repeat_num = 256 / sizeof(T2);
    constexpr static uint32_t input_block_num = 32 / sizeof(T1);
    constexpr static uint32_t ADDR_ALIGN_SIZE = 512;
    constexpr static uint32_t INPUT_NUMS = 2;
    constexpr static uint32_t BLOCK_SIZE = 32;
    constexpr static int64_t C0_SIZE = 16;
    constexpr static int64_t VEC_REPEAT = 8;
    constexpr static uint32_t PREFIX_COMPRESS_CAUSAL_S_SIZE = 2048;
    constexpr static uint32_t PREFIX_COMPRESS_ALL_MASK_S1_SIZE = 1024;
    constexpr static int64_t GM_DOUBLE_BUFFER = 2;
    constexpr static int64_t TMP_UB_OFFSET = 148 * 1024;
    constexpr static int64_t SFMG_UB_OFFSET = (148 + 33) * 1024;
    constexpr static int64_t TMP_UB_SIZE = 33 * 1024;
    constexpr static int64_t SFMG_UB_SIZE = 8 * 1024;
    constexpr static int64_t TOTAL_SIZE = 189 * 1024;

    constexpr static uint32_t MMAD_BASE_SIZE = 128;
    constexpr static uint32_t S_BASE_SIZE = 512;
    constexpr static uint32_t S_SPLITT_SIZE = 256;
    constexpr static uint32_t L1_CACHE_CAPACITY_LIMIT = 14;
    constexpr static uint32_t DIM_64 = 64;
    constexpr static uint32_t VEC_S2_LEN = 256;
    constexpr static int8_t OUTIDX= -1;
    bool tndSoftmaxIn;
    enum class AttenMaskCompress {
        Empty = 0,
        PreOnly = 1,
        NextOnly = 2,
        All = 3
    };
    AttenMaskCompress AttenBandMode = AttenMaskCompress::All;

    CATLASS_DEVICE
    BlockEpilogue(Arch::Resource<ArchTag> &resource, AscendC::TPipe *pipe_in,
                  __gm__ uint8_t *key, __gm__ uint8_t *keyRope, __gm__ uint8_t *value, __gm__ uint8_t *dx,
                  __gm__ uint8_t *query, __gm__ uint8_t *queryRope, __gm__ uint8_t *pse_shift,
                  __gm__ uint8_t *drop_mask, __gm__ uint8_t *atten_mask, __gm__ uint8_t *forward_res,
                  __gm__ uint8_t *softmax_max, __gm__ uint8_t *softmax_sum, __gm__ uint8_t *sink,
                  __gm__ uint8_t *prefixN, __gm__ uint8_t *actual_seq_qlen, __gm__ uint8_t *actual_seq_kvlen,
                  __gm__ uint8_t *dq, __gm__ uint8_t *dqRope, __gm__ uint8_t *dk, __gm__ uint8_t *dkRope,
                  __gm__ uint8_t *dv, __gm__ uint8_t *dpse, __gm__ uint8_t *dsink,
                  __gm__ uint8_t *workspace, __gm__ uint8_t *tiling_in)
    {
        InitBuffer(pipe_in);
        AscendC::PRINTF("za::BlockEpilogue init begain");
        keyGm.SetGlobalBuffer((__gm__ T1 *)key);
        valueGm.SetGlobalBuffer((__gm__ T1 *)value);
        dxGm.SetGlobalBuffer((__gm__ T1 *)dx);
        queryGm.SetGlobalBuffer((__gm__ T1 *)query);
        forwardResGm.SetGlobalBuffer((__gm__ T1 *)forward_res);
        pseGm.SetGlobalBuffer((__gm__ T1 *)pse_shift);
        sinkGm.SetGlobalBuffer((__gm__ float *)sink);
        if constexpr (HAS_ROPE == ENABLE) {
            keyRopeGm.SetGlobalBuffer((__gm__ T1 *)keyRope);
            queryRopeGm.SetGlobalBuffer((__gm__ T1 *)queryRope);
        }

        pseSlope = pse_shift;

        dropMaskGm.SetGlobalBuffer((__gm__ uint8_t *)drop_mask);
        attenMaskU8Gm.SetGlobalBuffer((__gm__ uint8_t *)atten_mask);
        softmaxMaxGm.SetGlobalBuffer((__gm__ float *)softmax_max);
        softmaxSumGm.SetGlobalBuffer((__gm__ float *)softmax_sum);
        dvGm.SetGlobalBuffer((__gm__ float *)dv);

        if ASCEND_IS_AIV {
            cBlockIdx = GetBlockIdx();
            cCubeBlockIdx = cBlockIdx / 2;
            cSubIdx = cBlockIdx % 2;
        } else {
            cCubeBlockIdx = GetBlockIdx();
            cBlockIdx = cCubeBlockIdx * 2;
        }

        AscendC::GlobalTensor<uint64_t> tilingData;
        tilingData.SetGlobalBuffer((__gm__ uint64_t *)tiling_in);

        coreNum = tilingData.GetValue(0);
        cubeCoreNum = coreNum / 2;
        vecBlockNum = coreNum / 3;

        b = tilingData.GetValue(1);
        n2 = tilingData.GetValue(2);
        g = tilingData.GetValue(3);
        s1 = tilingData.GetValue(4);
        s2 = tilingData.GetValue(5);
        d = tilingData.GetValue(6);
        value_d = tilingData.GetValue(7);
        tndSoftmaxIn = tilingData.GetValue(8) == 1 ? true : false;
        dAlign = (d + 15) / 16 * 16;
        value_dAlign = (value_d + 15) / 16 * 16;

        attenMaskDimS2 = tilingData.GetValue(9);

        s1Token = tilingData.GetValue(10);
        s2Token = tilingData.GetValue(11);
        actualCalcS1Token = s1Token;
        actualCalcS2Token = s2Token;
        sparseMode = tilingData.GetValue(12);
        bandIdx = tilingData.GetValue(13);

        s1Outer = tilingData.GetValue(14);
        s1CvInner = tilingData.GetValue(15);
        s1CvTail = tilingData.GetValue(16);
        s2Outer = tilingData.GetValue(17);
        s2CvInner = tilingData.GetValue(18);
        s2CvTail = s2 - (s2Outer - 1) * s2CvInner;

        baseMN = tilingData.GetValue(19);
        cubeBaseMN = s1CvInner * s2CvInner;

        prefixN_addr = prefixN;
        actual_seq_qlen_addr = actual_seq_qlen;
        actual_seq_kvlen_addr = actual_seq_kvlen;

        dropBitMode = s2 % 8 == 0;

        AscendC::GlobalTensor<float> tilingDataFp;
        tilingDataFp.SetGlobalBuffer((__gm__ float *)tiling_in);
        keepProb = tilingDataFp.GetValue(20 * 2);

        int64_t sfmgOutputSize = b * n2 * g * s1 * 8;

        int64_t maskPreBlockTotal = tilingData.GetValue(21);
        int64_t qPostBlockTotal = tilingData.GetValue(22);
        int64_t kvPostBlockTotal = tilingData.GetValue(23);
        int64_t vPostBlockTotal = tilingData.GetValue(24);
        int64_t qRopePostBlockTotal = 0;
        int64_t kRopePostBlockTotal = 0;

        workspaceAddr = workspace;

        syncGlobal.SetGlobalBuffer((__gm__ int32_t *)workspace);
        AscendC::InitOutput<int32_t>(syncGlobal[GetBlockIdx() * 256], 256, 0);

        int64_t dqWorkSpaceOffset = tilingData.GetValue(25);
        int64_t dkWorkSpaceOffset = tilingData.GetValue(26);
        int64_t dvWorkSpaceOffset = tilingData.GetValue(27);
        int64_t sfmgPreBeginAddr = tilingData.GetValue(28);

        dqWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace + dqWorkSpaceOffset / sizeof(float));
        dkWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace + dkWorkSpaceOffset / sizeof(float));
        dvWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace + dvWorkSpaceOffset / sizeof(float));

        sfmgWorkspaceGm.SetGlobalBuffer((__gm__ T2 *)workspace + sfmgPreBeginAddr / sizeof(T2));
        int64_t workspaceOffsets =
            (sfmgPreBeginAddr + sfmgOutputSize * sizeof(float) + ADDR_ALIGN_SIZE) /
            ADDR_ALIGN_SIZE * ADDR_ALIGN_SIZE;

        AscendC::PRINTF("za::BlockEpilogue init end");

        int64_t pseInnerAlibiSize = tilingData.GetValue(29) * tilingData.GetValue(30) * sizeof(half);
        int64_t pseAlibiOffset = CeilDiv(pseInnerAlibiSize, 512) * 512;

        uint32_t matmulWorkspaceSize = cubeBaseMN * sizeof(float);
        mm1WorkspaceGm.SetGlobalBuffer((__gm__ T2 *)(workspace + workspaceOffsets +
                                                     cCubeBlockIdx * matmulWorkspaceSize * GM_DOUBLE_BUFFER));
        mm2WorkspaceGm.SetGlobalBuffer(
            (__gm__ T2 *)(workspace + workspaceOffsets + cubeCoreNum * matmulWorkspaceSize * GM_DOUBLE_BUFFER +
                          cCubeBlockIdx * matmulWorkspaceSize * GM_DOUBLE_BUFFER));

        dropWorkSpaceGm.SetGlobalBuffer(
            (__gm__ T1 *)(workspace + workspaceOffsets + cubeCoreNum * matmulWorkspaceSize * GM_DOUBLE_BUFFER +
                          cCubeBlockIdx * matmulWorkspaceSize * GM_DOUBLE_BUFFER));

        mulWorkSpaceGm.SetGlobalBuffer((__gm__ T1 *)(workspace + workspaceOffsets +
                                                     cCubeBlockIdx * matmulWorkspaceSize * GM_DOUBLE_BUFFER));

        int64_t dsinksumWorkSpaceOffset = tilingData.GetValue(31);
        int64_t dsinksumDataSizeOffset = tilingData.GetValue(32);

        dsinksumWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace +
                                dsinksumWorkSpaceOffset / sizeof(float));

        dsinksumDataSizeGm.SetGlobalBuffer((__gm__ uint32_t *)workspace +
                                dsinksumDataSizeOffset / sizeof(uint32_t));

        int64_t sinkDataSize = tilingData.GetValue(33);
        uint64_t pseAlibiAddr = (workspaceOffsets + cubeCoreNum * matmulWorkspaceSize * INPUT_NUMS *
                                                        GM_DOUBLE_BUFFER + ADDR_ALIGN_SIZE + sinkDataSize) / ADDR_ALIGN_SIZE * ADDR_ALIGN_SIZE;
        this->pseAlibiGm.SetGlobalBuffer((__gm__ half*)(workspace + pseAlibiAddr + cBlockIdx * pseAlibiOffset));

        if constexpr (IS_DTM == ENABLE) {
            workspaceOffsets = (pseAlibiAddr + coreNum * pseAlibiOffset + ADDR_ALIGN_SIZE - 1) / ADDR_ALIGN_SIZE *
                           ADDR_ALIGN_SIZE;

            dqDtmWsGm.SetGlobalBuffer((__gm__ float *)(workspace + workspaceOffsets));
            workspaceOffsets = (workspaceOffsets + s1CvInner * dAlign * sizeof(float) * cubeCoreNum * GM_DOUBLE_BUFFER +
                                ADDR_ALIGN_SIZE - 1) / ADDR_ALIGN_SIZE * ADDR_ALIGN_SIZE;
            if constexpr (HAS_ROPE == ENABLE) {
                dqRopeDtmWsGm.SetGlobalBuffer((__gm__ float *)(workspace + workspaceOffsets));
                workspaceOffsets = (workspaceOffsets + s1CvInner * rope_dAlign * sizeof(float) * cubeCoreNum * GM_DOUBLE_BUFFER +
                                    ADDR_ALIGN_SIZE - 1) / ADDR_ALIGN_SIZE * ADDR_ALIGN_SIZE;
            }

            dkDtmWsGm.SetGlobalBuffer((__gm__ float *)(workspace + workspaceOffsets));
            workspaceOffsets = (workspaceOffsets + s2CvInner * dAlign * sizeof(float) * cubeCoreNum * GM_DOUBLE_BUFFER +
                                ADDR_ALIGN_SIZE - 1) / ADDR_ALIGN_SIZE * ADDR_ALIGN_SIZE;
            if constexpr (HAS_ROPE == ENABLE) {
                dkRopeDtmWsGm.SetGlobalBuffer((__gm__ float *)(workspace + workspaceOffsets));
                workspaceOffsets = (workspaceOffsets + s2CvInner * rope_dAlign * sizeof(float) * cubeCoreNum * GM_DOUBLE_BUFFER +
                                    ADDR_ALIGN_SIZE - 1) / ADDR_ALIGN_SIZE * ADDR_ALIGN_SIZE;
            }

            dvDtmWsGm.SetGlobalBuffer((__gm__ float *)(workspace + workspaceOffsets));
        }
    }

    CATLASS_DEVICE
    ~BlockEpilogue()
    {
    }

    CATLASS_DEVICE
    void InitBuffer(AscendC::TPipe *pipe_in)
    {
        AscendC::PRINTF("za::BlockEpilogue InitBuffer begain");
        pipe = pipe_in;
        if ASCEND_IS_AIV {
            pipe->InitBuffer(unifiedBuffer, TOTAL_SIZE);
        }
        AscendC::SyncAll();
        AscendC::PRINTF("za::BlockEpilogue InitBuffer end");
    }

    CATLASS_DEVICE
    void GetSeqQlenKvlenByBidx(int64_t bIdx, int64_t &actualSeqQlen, int64_t &actualSeqKvlen)
    {
        if (unlikely(bIdx == 0)) {
            actualSeqQlen = ((__gm__ int64_t *)actual_seq_qlen_addr)[0];
            actualSeqKvlen = ((__gm__ int64_t *)actual_seq_kvlen_addr)[0];
        } else {
            actualSeqQlen =
                ((__gm__ int64_t *)actual_seq_qlen_addr)[bIdx] - ((__gm__ int64_t *)actual_seq_qlen_addr)[bIdx - 1];
            actualSeqKvlen =
                ((__gm__ int64_t *)actual_seq_kvlen_addr)[bIdx] - ((__gm__ int64_t *)actual_seq_kvlen_addr)[bIdx - 1];
        }
        return;
    }

    CATLASS_DEVICE
    void CopyInAttenMaskBool(LocalTensor<uint8_t> &dstTensor, int64_t attenMaskOffset,
                                             uint32_t s1Extend, uint32_t s2Extend)
    {
        AscendC::DataCopyExtParams intriParams;
        intriParams.blockCount = s1Extend;
        intriParams.blockLen = s2Extend * sizeof(uint8_t);
        intriParams.srcStride = (attenMaskDimS2 - s2Extend) * sizeof(uint8_t);
        intriParams.dstStride = 0;
        intriParams.rsv = 0;
        AscendC::DataCopyPad(dstTensor, attenMaskU8Gm[attenMaskOffset], intriParams, {false, 0, 0, 0});
    }

    CATLASS_DEVICE
    void CalcAttenMaskBool(LocalTensor<T2> &dstTensor, LocalTensor<uint8_t> srcTensor,
                                             uint32_t s1Extend, uint32_t s2Extend, uint8_t maskType = 0)
    {
        LocalTensor<uint8_t> tmpUbBuffer = unifiedBuffer.GetWithOffset<uint8_t>(TMP_UB_SIZE / sizeof(uint8_t), TMP_UB_OFFSET);

        T2 scalar;
        if constexpr (AscendC::IsSameType<T2, float>::value) {
            uint32_t tmp = 0xFF7FFFFF;
            scalar = *((float *)&tmp);
        } else {
            uint16_t tmp = 0xFBFF;
            scalar = *((half *)&tmp);
        }

        AscendC::SelectWithBytesMaskShapeInfo info;
        info.firstAxis = s1Extend;
        info.srcLastAxis = s2Extend;
        info.maskLastAxis = (s2Extend * sizeof(uint8_t) + 31) / 32 * 32 / sizeof(uint8_t);
        dstTensor.SetSize(info.firstAxis * info.srcLastAxis);
        srcTensor.SetSize(info.firstAxis * info.maskLastAxis);
        if (maskType == 0) {
            AscendC::SelectWithBytesMask(dstTensor, dstTensor, scalar, srcTensor, tmpUbBuffer, info);
        } else {
            AscendC::SelectWithBytesMask(dstTensor, scalar, dstTensor, srcTensor, tmpUbBuffer, info);
        }
    }

    CATLASS_DEVICE
    void CalcAttenMaskOffset(int64_t &attenMaskOffset, const int64_t delta,
                                                         uint32_t s1VSize, uint32_t s2VSize)
    {
        if (delta == 0) {
            attenMaskOffset = 0;
        } else if (delta < 0) {
            if (-delta > s1VSize) {
                attenMaskOffset = s1VSize;
            } else {
                attenMaskOffset = -delta;
            }
        } else {
            if (delta > s2VSize) {
                attenMaskOffset = s2VSize * attenMaskDimS2;
            } else {
                attenMaskOffset = delta * attenMaskDimS2;
            }
        }
    }

    CATLASS_DEVICE
    void CalcAttenBandMode(int64_t compressMode, int64_t causal_delta, DBParams &dbParam)
    {
        int64_t actualS1Len;
        int64_t actualS2Len;
        if (compressMode == 1 || compressMode == 2 || compressMode == 3 || sparseMode == 7 || sparseMode == 8) {
            int64_t next_delta = causal_delta;
            int64_t pre_delta = causal_delta - INT32_MAX - 1;
            if (compressMode == 1 || (sparseMode == 8 && dbParam.bIdx != bandIdx)) {
            } else if (compressMode == 2) {
                if constexpr (INPUT_LAYOUT == TND) {
                    next_delta = causal_delta - dbParam.actualS1Len + dbParam.actualS2Len;
                } else {
                    next_delta = causal_delta - s1 + s2;
                }
            } else if (sparseMode == 7 && dbParam.bIdx != bandIdx) {
                next_delta = causal_delta - dbParam.actualS1Len + dbParam.actualS2Len;
            } else {
                next_delta = causal_delta + actualCalcS2Token;
                pre_delta = causal_delta - actualCalcS1Token - 1;
            }

            bool NoNext = (next_delta - s2Extend >= 0);
            bool NoPre = (pre_delta + 1 + s1ExtendSubGraph <= 0);

            if (NoNext && NoPre) {
                AttenBandMode = AttenMaskCompress::Empty;
            } else if (NoNext && !NoPre) {
                AttenBandMode = AttenMaskCompress::PreOnly;
            } else if (!NoNext && NoPre) {
                AttenBandMode = AttenMaskCompress::NextOnly;
            } else {
                AttenBandMode = AttenMaskCompress::All;
            }
        }
    }

    CATLASS_DEVICE
    void CalcAttenMaskOffsetForPrefixCompressMode(int64_t &attenMaskOffset,
        int64_t &attenMaskOffset2, const int64_t delta, uint32_t s1VSize, uint32_t s2VSize, uint32_t s2VBegin,
        bool &canSimplify, DBParams& dbParam)
    {
        canSimplify = false;

        int64_t S1 = static_cast<int64_t>(s1);
        int64_t S2 = static_cast<int64_t>(s2);
        uint32_t curBatchDimIdx = dbParam.bIdx;
        if constexpr (INPUT_LAYOUT == TND) {
            S1 = dbParam.actualS1Len;
            S2 = dbParam.actualS2Len;
        }

        int64_t N = ((__gm__ int64_t *)prefixN_addr)[curBatchDimIdx];

        if (S1 + N <= S2) {
            canSimplify = true;
            int64_t causal_delta = delta - S1 + S2;
            CalcAttenMaskOffset(attenMaskOffset, causal_delta, s1VSize, s2VSize);
            return;
        }

        int64_t delta1 = delta - S1 + S2;
        int64_t delta2 = N + 1 - static_cast<int64_t>(s2VBegin);

        if (delta2 > static_cast<int64_t>(s2VSize)) {
            canSimplify = true;
            attenMaskOffset = PREFIX_COMPRESS_CAUSAL_S_SIZE * attenMaskDimS2;
            return;
        }

        if (delta1 >= 0) {
            attenMaskOffset = (delta1 <= s2VSize) ? delta1 * static_cast<int64_t>(attenMaskDimS2) :
                                                    s2VSize * static_cast<int64_t>(attenMaskDimS2);
        } else {
            attenMaskOffset = (-delta1 <= s1VSize) ? -delta1 : s1VSize;
        }

        int64_t offsetStartPos =
            (int64_t)PREFIX_COMPRESS_CAUSAL_S_SIZE * (int64_t)attenMaskDimS2 + (int64_t)PREFIX_COMPRESS_ALL_MASK_S1_SIZE;
        attenMaskOffset2 = (delta2 > 0) ? (offsetStartPos - delta2 + 1) : offsetStartPos;
    }

    CATLASS_DEVICE
    void CalcAttenMaskOffsetWithSparseMode(int64_t &attenMaskOffset,
        int64_t &attenMaskOffset2, uint32_t s1VSize, uint32_t s2VSize, int64_t curS1Idx, uint32_t s2VBegin,
        bool &canSimplify, DBParams &dbParam)
    {
        uint64_t compressMode = 0;
        int64_t causal_delta =
            static_cast<int64_t>(dbParam.s1oIdx * s1CvInner + curS1Idx * s1VecSize) - static_cast<int64_t>(s2VBegin);
        CalcAttenBandMode(compressMode, causal_delta, dbParam);
        if (compressMode == 1) {
            CalcAttenMaskOffset(attenMaskOffset, causal_delta, s1VSize, s2VSize);
            return;
        }

        if (compressMode == 2) {
            causal_delta = causal_delta - s1 + s2;
            CalcAttenMaskOffset(attenMaskOffset, causal_delta, s1VSize, s2VSize);
            return;
        }

        if (compressMode == 3) {
            int64_t pre_delta = causal_delta - actualCalcS1Token - 1;
            CalcAttenMaskOffset(attenMaskOffset2, pre_delta, s1VSize, s2VSize);
            int64_t next_delta = causal_delta + actualCalcS2Token;
            CalcAttenMaskOffset(attenMaskOffset, next_delta, s1VSize, s2VSize);
            return;
        }

        if (compressMode == 4) {
            CalcAttenMaskOffsetForPrefixCompressMode(attenMaskOffset, attenMaskOffset2, causal_delta, s1VSize, s2VSize,
                                                     s2VBegin, canSimplify, dbParam);
            return;
        }

        int64_t attenMaskShapeType = 0;
        if (attenMaskShapeType == 0) {
            attenMaskOffset = (static_cast<int64_t>(dbParam.s1oIdx) * s1CvInner + curS1Idx * s1VecSize) * s2 + s2VBegin;
        } else if (attenMaskShapeType == 1) {
            attenMaskOffset =
                (dbParam.bIdx * s1 + dbParam.s1oIdx * s1CvInner + curS1Idx * s1VecSize) * s2 + s2VBegin;
        } else {
            attenMaskOffset = (((dbParam.bIdx * n2 + dbParam.n2Idx) * g + dbParam.gIdx) * s1 +
                               dbParam.s1oIdx * s1CvInner + curS1Idx * s1VecSize) * s2 + s2VBegin;
        }
    }

    CATLASS_DEVICE
    void CalcAttenMaskOffsetWithSparseModeForUnpad(int64_t &attenMaskOffset,
                                                                         int64_t &attenMaskOffset2, uint32_t s1VSize,
                                                                         uint32_t s2VSize, int64_t curS1Idx,
                                                                         uint32_t s2VBegin, bool unpadUseBand,
                                                                         bool &canSimplify, DBParams &dbParam)
    {
        CalcAttenMaskOffsetWithSparseMode(attenMaskOffset, attenMaskOffset2, s1VSize, s2VSize, curS1Idx,
                                        s2VBegin, canSimplify, dbParam);
    }

    CATLASS_DEVICE
    void CopyInSoftMax(LocalTensor<float> &dstTensor, uint32_t s1Extend, uint32_t softMaxOffset)
    {
        AscendC::DataCopyPad(dstTensor, softmaxSumGm[softMaxOffset], {1, static_cast<uint16_t>(s1Extend * 32), 0, 0},
                    {false, 0, 0, 0});
        AscendC::DataCopyPad(dstTensor[s1Extend * 32 / sizeof(float)], softmaxMaxGm[softMaxOffset],
                    {1, static_cast<uint16_t>(s1Extend * 32), 0, 0}, {false, 0, 0, 0});
    }

    CATLASS_DEVICE
    void CalcSoftMax(LocalTensor<T2>& dstTensor, LocalTensor<float>& src0Tensor, LocalTensor<float>& src1Tensor,
                     uint32_t s1Extend, uint32_t s2Extend, uint32_t s2ExtendAlign, const AscendC::SoftMaxTiling& tiling)
    {
        bool isBasicBlock = (s1Extend % 8 == 0) && (s2Extend % 64 == 0);

        if (isBasicBlock) {
            LocalTensor<uint8_t> vecOutBuffer = unifiedBuffer.GetWithOffset<uint8_t>(TMP_UB_SIZE / sizeof(uint8_t), TMP_UB_OFFSET);
            uint32_t shapeArray1[2];
            shapeArray1[0] = s1Extend;
            shapeArray1[1] = s2Extend;
            dstTensor.SetShapeInfo(AscendC::ShapeInfo(2, shapeArray1, AscendC::DataFormat::ND));
            src0Tensor.SetShapeInfo(AscendC::ShapeInfo(2, shapeArray1, AscendC::DataFormat::ND));
            AscendC::SimpleSoftMax<T2, false, true>(dstTensor, src1Tensor, src1Tensor[s1Extend * 32 / sizeof(float)], src0Tensor,
                                      vecOutBuffer, tiling);
        } else {
            LocalTensor<T2> vecOutBuffer = unifiedBuffer.GetWithOffset<T2>(TMP_UB_SIZE / sizeof(T2), TMP_UB_OFFSET);
            uint32_t sub_block_count = (s2Extend + cal_repeat_num - 1) / cal_repeat_num;

            for(uint32_t subIdx = 0; subIdx < sub_block_count; subIdx++) {
                uint32_t subMaskCount = (subIdx == sub_block_count - 1) ? (s2Extend - subIdx * cal_repeat_num) : cal_repeat_num;
                AscendC::Sub(dstTensor[subIdx * cal_repeat_num], src0Tensor[subIdx * cal_repeat_num], src1Tensor[s1Extend * 8],
                        subMaskCount, s1Extend,
                        {static_cast<uint8_t>(1), static_cast<uint8_t>(1), 0,
                        static_cast<uint8_t>(s2ExtendAlign / 8), static_cast<uint8_t>(s2ExtendAlign / 8), 1});
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Exp(vecOutBuffer[subIdx * cal_repeat_num], dstTensor[subIdx * cal_repeat_num],
                    subMaskCount, s1Extend,
                        {static_cast<uint8_t>(1), static_cast<uint8_t>(1),
                        static_cast<uint8_t>(s2ExtendAlign / 8), static_cast<uint8_t>(s2ExtendAlign / 8)});
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Div(dstTensor[subIdx * cal_repeat_num], vecOutBuffer[subIdx * cal_repeat_num], src1Tensor,
                        subMaskCount, s1Extend,
                        {static_cast<uint8_t>(1), static_cast<uint8_t>(1), 0,
                        static_cast<uint8_t>(s2ExtendAlign / 8), static_cast<uint8_t>(s2ExtendAlign / 8), 1});
                AscendC::PipeBarrier<PIPE_V>();
            }
        }
    }

    CATLASS_DEVICE
    void DropOutCopy(LocalTensor<uint8_t> &vecInDropBuffer, int64_t curS1Idx, int64_t s2VBegin)
    {
    }

    CATLASS_DEVICE
    void NZCopyIn(int64_t mmAddr, GlobalTensor<T2> &mmWspGm, LocalTensor<T2> &mmTensorCurr,
                                uint32_t s1VecSize, uint32_t s2VecSize, uint32_t s1CvInner)
    {
    }

    CATLASS_DEVICE
    void NZ2ND(LocalTensor<T2> &mmTensorCurr, LocalTensor<T2> &tmpTensor, uint32_t s1VecSize,
                             uint32_t s2VecSize)
    {
    }

    CATLASS_DEVICE
    void ND2NZ(LocalTensor<T1> &mmTensorCurr, LocalTensor<T1> &tmpTensor, uint32_t s1VecSize,
                             uint32_t s2VecSize)
    {
    }

    CATLASS_DEVICE
    void SubGrapA(int64_t curIdx, int64_t curS1Idx, int64_t curS2Idx, DBParams& dbParam,
                                event_t mte2WaitMte3A)
    {
        pingpongIdx = dbParam.taskId % 2;
        s2Extend = (curS2Idx == s2VecLoop - 1) ? (dbParam.s2CvExtend - (s2VecLoop - 1) * s2VecSize) : s2VecSize;
        s2ExtendAlign = (s2Extend + 15) / 16 * 16;
        uint32_t s2VBegin = dbParam.s2oIdx * s2CvInner + curS2Idx * s2VecSize;

        uint32_t ubBufferOffset = 0;
        uint32_t ubTmpBufferOffset = 0;

        if (curIdx > 0) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(mte2WaitMte3A);
        }

        LocalTensor<float> vecInBuffer3 =
            unifiedBuffer.GetWithOffset<float>(8 * 1024 / sizeof(float), ubBufferOffset + T2BlockBegin);
        int64_t softMaxOffset = 0;
        if constexpr (INPUT_LAYOUT == TND) {
            if(tndSoftmaxIn){
                int64_t innerRowOffsetLeft = unlikely(dbParam.bIdx == 0) ? 0 : ((__gm__ int64_t *)actual_seq_qlen_addr)[dbParam.bIdx - 1] * 32 / sizeof(float);
                int64_t originInnerBatchOffset = ((dbParam.n2Idx * g + dbParam.gIdx) * dbParam.actualS1Len +
                                dbParam.s1oIdx * s1CvInner + curS1Idx * s1VecSize) * 32 / sizeof(float);
                softMaxOffset = ((((__gm__ int64_t *)actual_seq_qlen_addr)[b - 1] * 32 / sizeof(float)) * (dbParam.n2Idx * g + dbParam.gIdx) + innerRowOffsetLeft + originInnerBatchOffset % (dbParam.actualS1Len * 32 / sizeof(float)));
            }else {
                if (dbParam.bIdx > 0) {
                    softMaxOffset = ((__gm__ int64_t *)actual_seq_qlen_addr)[dbParam.bIdx - 1] * n2 * g * 32 / sizeof(float);
                }
                softMaxOffset += ((dbParam.n2Idx * g + dbParam.gIdx) * dbParam.actualS1Len +
                                dbParam.s1oIdx * s1CvInner + curS1Idx * s1VecSize) * 32 / sizeof(float);
            }
        } else {
            softMaxOffset = (((dbParam.bIdx * n2 + dbParam.n2Idx) * g + dbParam.gIdx) * s1 + dbParam.s1oIdx * s1CvInner +
                             curS1Idx * s1VecSize) * 32 / sizeof(float);
        }
        CopyInSoftMax(vecInBuffer3, s1ExtendSubGraph, softMaxOffset);

        LocalTensor<T1> pseUbT1 = unifiedBuffer.GetWithOffset<T1>(16 * 1024 / sizeof(T1), ubBufferOffset + T1Begin);
        LocalTensor<half> pseUb = pseUbT1.template ReinterpretCast<half>();

        LocalTensor<uint8_t> attenMaskUbuint8 =
            unifiedBuffer.GetWithOffset<uint8_t>(8 * 1024 / sizeof(uint8_t), ubBufferOffset + BoolBegin);
        bool unpadUseBand = (sparseMode == 7 && dbParam.bIdx == bandIdx) || (sparseMode == 8 && dbParam.bIdx == bandIdx);
        int64_t attenMaskOffsetPre = 0;
        bool prefixCompressCanSimplify = false;
        if constexpr (IS_ATTEN_MASK == ENABLE) {
            int64_t attenMaskOffset = 0;
            if constexpr(INPUT_LAYOUT == TND) {
                CalcAttenMaskOffsetWithSparseModeForUnpad(attenMaskOffset, attenMaskOffsetPre, s1ExtendSubGraph, s2Extend,
                                                        curS1Idx, s2VBegin, unpadUseBand, prefixCompressCanSimplify, dbParam);
            } else {
                CalcAttenMaskOffsetWithSparseMode(attenMaskOffset, attenMaskOffsetPre, s1ExtendSubGraph, s2Extend, curS1Idx,
                                                s2VBegin, prefixCompressCanSimplify, dbParam);
            }
            if (AttenBandMode == AttenMaskCompress::All || AttenBandMode == AttenMaskCompress::NextOnly) {
                CopyInAttenMaskBool(attenMaskUbuint8, attenMaskOffset, s1ExtendSubGraph, s2Extend);
            } else if (AttenBandMode == AttenMaskCompress::PreOnly) {
                CopyInAttenMaskBool(attenMaskUbuint8, attenMaskOffsetPre, s1ExtendSubGraph, s2Extend);
            }
        }

        LocalTensor<uint8_t> vecInDropBuffer =
            unifiedBuffer.GetWithOffset<uint8_t>(8 * 1024 / sizeof(uint8_t), ubBufferOffset + U8Begin);
        if constexpr (IS_DROP == ENABLE) {
            if constexpr (AscendC::IsSameType<T1, float>::value) {
                AscendC::PipeBarrier<PIPE_ALL>();
            }
            DropOutCopy(vecInDropBuffer, curS1Idx, s2VBegin);
        }

        LocalTensor<float> vecClc2Buffer =
            unifiedBuffer.GetWithOffset<float>(32 * 1024 / sizeof(float), ubBufferOffset + T2Begin);
        if constexpr (MM_OUT_FORMAT == AscendC::CubeFormat::ND) {
            if (s2VecLoop == 1) {
                AscendC::DataCopy(vecClc2Buffer, mm2WorkspaceGm[pingpongIdx * cubeBaseMN + curS1Idx * s1VecSize * s2ExtendAlign],
                        s1ExtendSubGraph * s2ExtendAlign);
            } else {
                AscendC::DataCopyPad(vecClc2Buffer, mm2WorkspaceGm[pingpongIdx * cubeBaseMN + curS1Idx * s1VecSize * dbParam.s2CvExtendAlign + curS2Idx * s2VecSize],
                            {static_cast<uint16_t>(s1ExtendSubGraph), static_cast<uint16_t>(s2ExtendAlign * sizeof(float)),
                             static_cast<uint16_t>((dbParam.s2CvExtendAlign - s2ExtendAlign) * sizeof(float)), 0},
                            {false, 0, 0, 0});
            }
            event_t vWaitMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(vWaitMte2);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(vWaitMte2);
        } else {
            int64_t mmAddr = pingpongIdx * cubeBaseMN + curS1Idx * s1VecSize * C0_SIZE + curS2Idx * dbParam.s1CvExtendAlign * s2VecSize;
            NZCopyIn(mmAddr, mm2WorkspaceGm, vecClc2Buffer, s1VecSize, s2ExtendAlign, dbParam.s1CvExtendAlign);
            event_t vWaitMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(vWaitMte2);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(vWaitMte2);
            auto tmpTensor = unifiedBuffer.GetWithOffset<T2>(TMP_UB_SIZE / sizeof(T2), TMP_UB_OFFSET);
            AscendC::DataCopy(tmpTensor, vecClc2Buffer, s1VecSize * s2ExtendAlign + s2ExtendAlign / C0_SIZE * VEC_REPEAT);
            AscendC::PipeBarrier<PIPE_V>();
            NZ2ND(vecClc2Buffer, tmpTensor, s1VecSize, s2ExtendAlign);
        }

        float scaleValue = 1.0f;
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(vecClc2Buffer, vecClc2Buffer, scaleValue, s1ExtendSubGraph * s2ExtendAlign);

        if constexpr (IS_ATTEN_MASK == ENABLE) {
            int64_t compressMode = 0;
            AscendC::PipeBarrier<PIPE_V>();

            if (compressMode == 4) {
                if (prefixCompressCanSimplify == false) {
                    LocalTensor<uint8_t> attenMaskUbPreuint8 =
                        unifiedBuffer.GetWithOffset<uint8_t>(8 * 1024 / sizeof(uint8_t), TMP_UB_OFFSET + ubTmpBufferOffset);
                    uint32_t s2ExtendPadAlign = (s2Extend + 31) / 32 * 32;
                    int32_t maskNum = s1ExtendSubGraph * s2ExtendPadAlign / 2;

                    event_t mte2WaitV = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE2));
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(mte2WaitV);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(mte2WaitV);
                    CopyInAttenMaskBool(attenMaskUbPreuint8, attenMaskOffsetPre, s1ExtendSubGraph, s2Extend);

                    event_t vWaitMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(vWaitMte2);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(vWaitMte2);
                    auto attenMaskUbuint8Tmp = attenMaskUbuint8.ReinterpretCast<uint16_t>();
                    auto attenMaskUbPreuint8Tmp = attenMaskUbPreuint8.ReinterpretCast<uint16_t>();
                    AscendC::And(attenMaskUbuint8Tmp, attenMaskUbPreuint8Tmp, attenMaskUbuint8Tmp, maskNum);
                    AscendC::PipeBarrier<PIPE_V>();
                    attenMaskUbuint8 = attenMaskUbuint8Tmp.ReinterpretCast<uint8_t>();
                }
            }

            if (AttenBandMode == AttenMaskCompress::All || AttenBandMode == AttenMaskCompress::NextOnly) {
                CalcAttenMaskBool(vecClc2Buffer, attenMaskUbuint8, s1ExtendSubGraph, s2ExtendAlign);
            } else if (AttenBandMode == AttenMaskCompress::PreOnly) {
                CalcAttenMaskBool(vecClc2Buffer, attenMaskUbuint8, s1ExtendSubGraph, s2ExtendAlign, 1);
            }

            if ((compressMode == 3 || unpadUseBand) && AttenBandMode == AttenMaskCompress::All) {
                event_t mte2WaitV = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE2));
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(mte2WaitV);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(mte2WaitV);
                CopyInAttenMaskBool(attenMaskUbuint8, attenMaskOffsetPre, s1ExtendSubGraph, s2Extend);
                event_t vWaitMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(vWaitMte2);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(vWaitMte2);
                CalcAttenMaskBool(vecClc2Buffer, attenMaskUbuint8, s1ExtendSubGraph, s2ExtendAlign, 1);
            }
        }

        AscendC::PipeBarrier<PIPE_V>();
        LocalTensor<float> simpleSoftmaxResBuf = unifiedBuffer.GetWithOffset<float>(33 * 1024 / sizeof(T2), DbBegin);
        AscendC::SoftMaxTiling softmaxTilingData;
        CalcSoftMax(simpleSoftmaxResBuf, vecClc2Buffer, vecInBuffer3, s1ExtendSubGraph, s2Extend, s2ExtendAlign, softmaxTilingData);

        LocalTensor<T2> vecDropBuffer = simpleSoftmaxResBuf;
        if constexpr (IS_DROP == ENABLE) {
            vecDropBuffer = unifiedBuffer.GetWithOffset<T2>(33 * 1024 / sizeof(T2), TMP_UB_OFFSET);
            AscendC::PipeBarrier<PIPE_V>();
            LocalTensor<uint8_t> tmpDropBuffer =
                unifiedBuffer.GetWithOffset<uint8_t>(32 * 1024 / sizeof(uint8_t), ubBufferOffset + T1Begin);

            dropMaskInfo.lstAxis = s2ExtendAlign;
            dropMaskInfo.maskLstAxis = s2ExtendAlign;
            AscendC::ComputeDropMask<T2, true>(vecDropBuffer, simpleSoftmaxResBuf, vecInDropBuffer, tmpDropBuffer, this->dropMaskInfo);
            if constexpr (AscendC::IsSameType<T1, float>::value) {
                AscendC::PipeBarrier<PIPE_ALL>();
            }
        }

        LocalTensor<T1> vecCopyOutBuffer = vecDropBuffer.template ReinterpretCast<T1>();
        if constexpr (!AscendC::IsSameType<T1, float>::value) {
            vecCopyOutBuffer = unifiedBuffer.GetWithOffset<T1>(17 * 1024 / sizeof(T1), ubBufferOffset + T1Begin);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(vecCopyOutBuffer, vecDropBuffer, RoundMode::CAST_ROUND, s1ExtendSubGraph * s2ExtendAlign);
        }
        int64_t copyOutOffset = 0;
        DataCopyParams copyOutParam;
        if constexpr (MM_OUT_FORMAT == AscendC::CubeFormat::NZ) {
            AscendC::PipeBarrier<PIPE_V>();
            LocalTensor<T1> tmpTensor = unifiedBuffer.GetWithOffset<T1>(TMP_UB_SIZE / sizeof(T1), TMP_UB_OFFSET);
            copyOutOffset = pingpongIdx * cubeBaseMN * DTYPE_FACTOR + curS1Idx * s1VecSize * C0_SIZE +
                            curS2Idx * dbParam.s1CvExtendAlign * DTYPE_FACTOR * s2VecSize;
            copyOutParam = {
                static_cast<uint16_t>(s2ExtendAlign / C0_SIZE),
                static_cast<uint16_t>(s1ExtendSubGraph * C0_SIZE * sizeof(T1)),
                1,
                static_cast<uint16_t>((dbParam.s1CvExtendAlign * DTYPE_FACTOR - s1ExtendSubGraph) * C0_SIZE * sizeof(T1))
            };
            AscendC::DataCopy(tmpTensor, vecCopyOutBuffer, s1ExtendSubGraph * s2ExtendAlign);
            AscendC::PipeBarrier<PIPE_V>();
            ND2NZ(vecCopyOutBuffer, tmpTensor, s1ExtendSubGraph, s2ExtendAlign);
        } else {
            copyOutOffset = pingpongIdx * cubeBaseMN * DTYPE_FACTOR +
                            curS1Idx * s1VecSize * dbParam.s2CvExtendAlign * DTYPE_FACTOR + curS2Idx * s2VecSize;
            copyOutParam = {
                static_cast<uint16_t>(s1ExtendSubGraph),
                static_cast<uint16_t>(s2ExtendAlign * sizeof(T1)),
                0,
                static_cast<uint16_t>((dbParam.s2CvExtendAlign * DTYPE_FACTOR - s2ExtendAlign) * sizeof(T1))
            };
        }
        event_t mte3WaitV = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(mte3WaitV);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(mte3WaitV);
        AscendC::DataCopyPad(dropWorkSpaceGm[copyOutOffset], vecCopyOutBuffer, copyOutParam);

        if (curIdx < vecLoopEnd - vecLoopStart - 1) {
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(mte2WaitMte3A);
        }
    }

    CATLASS_DEVICE
    void SubGrapB(int64_t curIdx, int64_t s1VecLoop, int64_t s2VecLoop,
                                 int64_t curS1Idx, int64_t curS2Idx, DBParams& dbParam, event_t mte2WaitMte3B, float* dsinkSumLocal)
    {
        pingpongIdx = dbParam.taskId % 2;
        uint32_t ubBufferOffset = DbBegin;
        s2Extend = (curS2Idx == s2VecLoop -1) ? (dbParam.s2CvExtend - (s2VecLoop - 1) * s2VecSize) : s2VecSize;
        s2ExtendAlign = (s2Extend + 15) / 16 * 16;

        if (curIdx > 0) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(mte2WaitMte3B);
        }

        if (preS1Idx != curS1Idx) {
            preS1Idx = curS1Idx;
            LocalTensor<float> sfmgClc3 = unifiedBuffer.GetWithOffset<float>(SFMG_UB_SIZE / sizeof(float), SFMG_UB_OFFSET);
            AscendC::DataCopy(sfmgClc3, sfmgWorkspaceGm[sfmgOffset + curS1Idx * s1VecSize * 8], s1ExtendSubGraph * 8);
        }

        LocalTensor<uint8_t> vecInDropBuffer =
            unifiedBuffer.GetWithOffset<uint8_t>(8 * 1024 / sizeof(uint8_t), ubBufferOffset + U8Begin);
        if constexpr (IS_DROP == ENABLE) {
            int64_t s2VBegin = dbParam.s2oIdx * s2CvInner + curS2Idx * s2VecSize;
            DropOutCopy(vecInDropBuffer, curS1Idx, s2VBegin);
            if constexpr (AscendC::IsSameType<T1, float>::value) {
                AscendC::PipeBarrier<PIPE_ALL>();
            }
        }

        LocalTensor<T2> vecClc1Buffer = unifiedBuffer.GetWithOffset<T2>(33 * 1024 / sizeof(T2), ubBufferOffset + T1Begin);
        LocalTensor<T2> dyvBuffer = unifiedBuffer.GetWithOffset<T2>(33 * 1024 / sizeof(T2), TMP_UB_OFFSET);
        if constexpr (MM_OUT_FORMAT == AscendC::CubeFormat::ND) {
            if (s2VecLoop == 1) {
                AscendC::DataCopy(vecClc1Buffer, mm1WorkspaceGm[pingpongIdx * cubeBaseMN + curS1Idx * s1VecSize * s2ExtendAlign],
                         s1ExtendSubGraph * s2ExtendAlign);
            } else {
                AscendC::DataCopyPad(vecClc1Buffer, mm1WorkspaceGm[pingpongIdx * cubeBaseMN + curS1Idx * s1VecSize * dbParam.s2CvExtendAlign + curS2Idx * s2VecSize],
                            {static_cast<uint16_t>(s1ExtendSubGraph), static_cast<uint16_t>(s2ExtendAlign * sizeof(float)),
                             static_cast<uint16_t>((dbParam.s2CvExtendAlign - s2ExtendAlign) * sizeof(float)), 0},
                            {false, 0, 0, 0});
            }
            event_t vWaitMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(vWaitMte2);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(vWaitMte2);
        } else {
            int64_t mmAddr = pingpongIdx * cubeBaseMN + curS1Idx * s1VecSize * C0_SIZE + curS2Idx * dbParam.s1CvExtendAlign * s2VecSize;
            NZCopyIn(mmAddr, mm1WorkspaceGm, vecClc1Buffer, s1VecSize, s2ExtendAlign, dbParam.s1CvExtendAlign);
            event_t vWaitMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(vWaitMte2);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(vWaitMte2);
            auto tmpTensor = unifiedBuffer.GetWithOffset<T2>(TMP_UB_SIZE / sizeof(T2), TMP_UB_OFFSET);
            AscendC::DataCopy(tmpTensor, vecClc1Buffer, s1VecSize * s2ExtendAlign + s2ExtendAlign / C0_SIZE * VEC_REPEAT);
            AscendC::PipeBarrier<PIPE_V>();
            NZ2ND(vecClc1Buffer, tmpTensor, s1VecSize, s2ExtendAlign);
        }

        AscendC::PipeBarrier<PIPE_V>();

        uint32_t sub_block_cout = (s2ExtendAlign + cal_repeat_num - 1) / cal_repeat_num;
        LocalTensor<float> sfmgClc3 = unifiedBuffer.GetWithOffset<float>(SFMG_UB_SIZE / sizeof(float), SFMG_UB_OFFSET);
        AscendC::PipeBarrier<PIPE_V>();
        for (uint32_t subIdx = 0; subIdx < sub_block_cout; subIdx++) {
            uint32_t subMaskCout =
                (subIdx == sub_block_cout - 1) ? (s2ExtendAlign - subIdx * cal_repeat_num) : cal_repeat_num;
            AscendC::Sub(vecClc1Buffer[subIdx * cal_repeat_num], vecClc1Buffer[subIdx * cal_repeat_num], sfmgClc3,
                subMaskCout, s1ExtendSubGraph,
                {static_cast<uint8_t>(1), static_cast<uint8_t>(1), 0, static_cast<uint8_t>(s2ExtendAlign / 8),
                 static_cast<uint8_t>(s2ExtendAlign / 8), 1});
        }

        AscendC::PipeBarrier<PIPE_V>();
        LocalTensor<float> simpleSoftmaxResBuf = unifiedBuffer.GetWithOffset<float>(32 * 1024 / sizeof(float), DbBegin);
        AscendC::Mul(vecClc1Buffer, vecClc1Buffer, simpleSoftmaxResBuf, s1ExtendSubGraph * s2ExtendAlign);
        LocalTensor<T1> vecCopyOutBuffer = vecClc1Buffer.template ReinterpretCast<T1>();
        if constexpr (!AscendC::IsSameType<T1, float>::value) {
            vecCopyOutBuffer = unifiedBuffer.GetWithOffset<T1>(17 * 1024 / sizeof(T1), ubBufferOffset + T1Begin);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(vecCopyOutBuffer, vecClc1Buffer, RoundMode::CAST_ROUND, s1ExtendSubGraph * s2ExtendAlign);
        }

        int64_t copyOutOffset = 0;
        DataCopyParams copyOutParam;
        if constexpr (MM_OUT_FORMAT == AscendC::CubeFormat::NZ) {
            auto tmpTensor1 = unifiedBuffer.GetWithOffset<T1>(TMP_UB_SIZE / sizeof(T1), TMP_UB_OFFSET);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::DataCopy(tmpTensor1, vecCopyOutBuffer, s1ExtendSubGraph * s2ExtendAlign);
            AscendC::PipeBarrier<PIPE_V>();
            ND2NZ(vecCopyOutBuffer, tmpTensor1, s1ExtendSubGraph, s2ExtendAlign);

            copyOutOffset = pingpongIdx * cubeBaseMN * DTYPE_FACTOR + curS1Idx * s1VecSize * C0_SIZE +
                            curS2Idx * dbParam.s1CvExtendAlign * DTYPE_FACTOR * s2VecSize;
            copyOutParam = {
                static_cast<uint16_t>(s2ExtendAlign / C0_SIZE),
                static_cast<uint16_t>(s1ExtendSubGraph * C0_SIZE * sizeof(T1)),
                1,
                static_cast<uint16_t>((dbParam.s1CvExtendAlign * DTYPE_FACTOR - s1ExtendSubGraph) * C0_SIZE * sizeof(T1))
            };
        } else {
            copyOutOffset = pingpongIdx * cubeBaseMN * DTYPE_FACTOR +
                            curS1Idx * s1VecSize * dbParam.s2CvExtendAlign * DTYPE_FACTOR + curS2Idx * s2VecSize;
            copyOutParam = {
                static_cast<uint16_t>(s1ExtendSubGraph),
                static_cast<uint16_t>(s2ExtendAlign * sizeof(T1)),
                0,
                static_cast<uint16_t>((dbParam.s2CvExtendAlign * DTYPE_FACTOR - s2ExtendAlign) * sizeof(T1))
            };
        }
        event_t mte3WaitV = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(mte3WaitV);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(mte3WaitV);

        AscendC::DataCopyPad(mulWorkSpaceGm[copyOutOffset], vecCopyOutBuffer, copyOutParam);

        if (curIdx < vecLoopEnd - vecLoopStart - 1) {
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(mte2WaitMte3B);
        }
    }

    CATLASS_DEVICE
    void operator(DBParams& dbParam)
    {
        int64_t actualS1Len;
        int64_t actualS2Len;

        s2VecSize = dbParam.s2CvExtend > VEC_S2_LEN ? VEC_S2_LEN : dbParam.s2CvExtend;
        s2VecLoop = s2VecSize == 0 ? 0 : CeilDiv(dbParam.s2CvExtend, s2VecSize);

        uint32_t s2AlignFactor = BLOCK_SIZE / 2;
        if constexpr (IS_ATTEN_MASK == ENABLE) {
            s2AlignFactor = BLOCK_SIZE / sizeof(uint8_t);
        }

        s1VecSize = baseMN / AlignUp(s2VecSize, s2AlignFactor);
        s1VecSize = s1VecSize > dbParam.s1CvExtend ? dbParam.s1CvExtend : s1VecSize;
        s1VecSize = s1VecSize > 128 ? 128 : s1VecSize;
        s1VecLoop = s1VecSize == 0 ? 0 : CeilDiv(dbParam.s1CvExtend, s1VecSize);
        dropMaskInfo.splitS1BaseSize = s1VecSize;

        dropMaskInfo.s2TotalSize = s2;
        dropMaskInfo.bSSOffset = dbParam.bIdx * s1 * s2;
        pseInfo.s2SizeAcc = dbParam.bIdx * s2;
        pseInfo.bSSOffset = dropMaskInfo.bSSOffset;

        dropMaskInfo.gOutIdx = dbParam.gIdx;
        dropMaskInfo.n2OutIdx = dbParam.n2Idx;
        dropMaskInfo.s1OutIdx = dbParam.s1oIdx;

        sfmgOffset = 0;

        sfmgOffset = (((dbParam.bIdx * n2 + dbParam.n2Idx) * g + dbParam.gIdx) * s1 + dbParam.s1oIdx * s1CvInner) * 8;

        int32_t loopSize = s1VecLoop * s2VecLoop;
        int32_t halfLoop = 0;

        halfLoop = (s1VecLoop / 2) * s2VecLoop;

        vecLoopStart = cSubIdx ? halfLoop : 0;
        vecLoopEnd = cSubIdx ? loopSize : halfLoop;
        preS1Idx = -1;
        event_t mte2WaitMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_MTE2));
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(mte2WaitMte3);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(mte2WaitMte3);
        float dsinkSumLocal = 0.0f;
        for (int32_t i = vecLoopStart, loopCnt = 0; i < vecLoopEnd; i++, loopCnt++) {
            int32_t curS1Idx;
            int32_t curS2Idx;
            curS1Idx = i / s2VecLoop;
            curS2Idx = i % s2VecLoop;

            s1ExtendSubGraph = (curS1Idx == s1VecLoop - 1) ? (dbParam.s1CvExtend - (s1VecLoop - 1) * s1VecSize) : s1VecSize;
            dropMaskInfo.s1CopySize = s1ExtendSubGraph;
            dropMaskInfo.s1InnerIdx = curS1Idx;
            dropMaskInfo.firstAxis = s1ExtendSubGraph;

            event_t mte2WaitMte3A = static_cast<event_t>(GetTPipePtr()->AllocEventID<AscendC::HardEvent::MTE3_MTE2>());
            event_t mte2WaitMte3B = static_cast<event_t>(GetTPipePtr()->AllocEventID<AscendC::HardEvent::MTE3_MTE2>());
            SubGrapA(loopCnt, curS1Idx, curS2Idx, dbParam, mte2WaitMte3A);
            SubGrapB(loopCnt, s1VecLoop, s2VecLoop, curS1Idx, curS2Idx, dbParam, mte2WaitMte3B, &dsinkSumLocal);

            GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::MTE3_MTE2>(mte2WaitMte3A);
            GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::MTE3_MTE2>(mte2WaitMte3B);
        }
    }
};

}

#endif
