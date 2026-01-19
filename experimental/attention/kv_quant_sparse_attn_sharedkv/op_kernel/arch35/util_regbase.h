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
 * \file util_regbase.h
 * \brief
 */

#ifndef FLASH_ATTENTION_UTIL_REGBASE_H
#define FLASH_ATTENTION_UTIL_REGBASE_H

#include "util.h"

using AscendC::TQue;
using AscendC::QuePosition;

namespace regbaseutil {
constexpr uint16_t regBytes = 256;
constexpr int64_t MAX_PRE_NEXT_TOKENS = 0x7FFFFFFF;
enum class VselrIndexEnum {GT_64_AND_LTE_128_INDEX = 0, GT_0_AND_LTE_64_INDEX = 1, DN_INDEX = 2};
enum class DTemplateType {
    Aligned16 = 16,
    Aligned32 = 32,
    Aligned48 = 48,
    Aligned64 = 64,
    Aligned80 = 80,
    Aligned96 = 96,
    Aligned128 = 128,
    Aligned160 = 160,
    Aligned192 = 192,
    Aligned256 = 256,
    Aligned512 = 512,
    Aligned576 = 576,
    Aligned768 = 768,
    NotAligned,
};

enum class S1TemplateType {
    Aligned16 = 16,
    Aligned64 = 64,
    Aligned128 = 128,
    Aligned256 = 256,
    NotAligned,
};

enum class S2TemplateType {
    Aligned16 = 16,
    Aligned32 = 32,
    Aligned64 = 64,
    Aligned128 = 128,
    Aligned256 = 256,
    Aligned512 = 512,
    Aligned1024 = 1024,
    NotAligned,
};

enum class SparseType : uint8_t {
    DENSE = 0,
    CASUAL = 1,
    BAND = 2,
    UNSUPPORTED = 3    // 超L2优化暂不支持sparse的场景
};

#define COMMON_RUN_PARAM \
    int64_t boIdx; \
    int64_t s1oIdx; \
    int64_t n2oIdx; \
    int64_t goIdx; \
    int64_t s2LoopEndIdx;          /* S2方向的循环控制信息 souter层确定 */ \
    int64_t s2LineStartIdx = 0;    /* S2方向按行的起始位置 */ \
    int64_t s2LineEndIdx;          /* S2方向按行的结束位置 */ \
    int64_t s2CmpLineEndIdx; \
    /* cube视角的sOuter，在SAMEAB场景中cubeSOuterSize为两倍的 halfS1RealSize souter层确定 */ \
    uint32_t s1RealSize; \
    uint32_t halfS1RealSize; \
    uint32_t firstHalfS1RealSize; \
    uint32_t mRealSize; \
    uint32_t halfMRealSize; \
    uint32_t firstHalfMRealSize; \
    int64_t tensorQOffset;         /* query的offset souter层确定 */ \
    int64_t attentionOutOffset;    /* attentionOut的offset souter层确定 */ \
    int32_t actualS1Size;      /* Q的actualSeqLength */ \
    int32_t actualS2Size;    /* KV的actualSeqLength */ \
    uint64_t b1SSOffset; \
    uint64_t b1SSAttenMaskOffset; \
    uint64_t b1SSOffsetAlign16; \

struct RunParamStr {  // 分核与切块需要使用到参数
    COMMON_RUN_PARAM;
    /* 推理新增 */
    int64_t s1LoopTimes;
    // BN循环生产的数据
    int64_t preTokensPerBatch = MAX_PRE_NEXT_TOKENS; // 左上顶点的pretoken
    int64_t nextTokensPerBatch = MAX_PRE_NEXT_TOKENS; // 左上顶点的nexttoken

    // NBS1循环生产的数据
    int64_t sOuterOffset;               // 单个S内 souter的 souterIdx * halfS1RealSize souter层确定
    int64_t cubeSOuterOffset;           // 单个S内 souter的 souterIdx * halfS1RealSize souter层确定
    int64_t mOuterOffset;
    int64_t cubeMOuterOffset;
    int64_t keyCoreOffset;              // BN方向上，不同BN的Key的offset batch层确定
    int64_t valueCoreOffset;            // BN方向上，不同BN的value的offset batch层确定
    int64_t keyOffset;              // mm1 Key 的offset,后续更名为KFinalOffset

    // q k v attenMask不同轴的offset
    // B轴offset 
    int64_t qBOffset;             // bIdx * seqSize * multiHeadQ，后续更名为qBOffset batch层确定

    // lse 输出offset
    int64_t softmaxLseOffset;       // souter层确定

    int64_t qSNumInOneBlock;
    int64_t oriKvLoopEndIdx;
    int64_t cmpKvLoopEndIdx;
};

#define COMMON_RUN_INFO \
    int64_t s2StartIdx; /* s2的起始位置，sparse场景下可能不是0 */ \
    int64_t s2EndIdx; \
    int64_t s2LoopCount; /* s2循环当前的循环index */ \
    int64_t s2LoopLimit; \
    int64_t s1oIdx = 0; /* s1轴的index */ \
    int64_t loop = 0; /* for v0 perload loop */ \
    int64_t boIdx = 0; /* b轴的index */ \
    int64_t n2oIdx = 0; /* n2轴的index */ \
    int64_t goIdx = 0; /* g轴的index */ \
    int32_t s1RealSize; \
    int32_t halfS1RealSize; /* vector侧实际的s1基本块大小，如果Cube基本块=128，那么halfS1RealSize=64 */ \
    int32_t firstHalfS1RealSize; /* 当s1RealSize不是2的整数倍时，v0比v1少计算一行，计算subblock偏移的时候需要使用v0的s1 size */ \
    int32_t mRealSize; \
    int32_t halfMRealSize; \
    int32_t firstHalfMRealSize; \
    int32_t s2RealSize; /* s2方向基本块的真实长度 */ \
    int64_t s2AlignedSize; /* s2方向基本块对齐到16之后的长度 */ \
    int32_t vec2S1BaseSize; /* vector2侧开循环之后，经过切分的S1大小，例如把64切分成两份32 */ \
    int32_t vec2S1RealSize; /* vector2侧开循环之后，经过切分的S1的尾块大小，例如把63切分成两份32和31，第二份的实际大小是31 */ \
    int32_t vec2MBaseSize; \
    int32_t vec2MRealSize; \
    int64_t vecCoreOffset; /* vec核基于cube核起始处s1方向偏移 */ \
    int64_t queryOffset; /* mm1 Query的offset*/\
    int64_t keyOffset; /* mm1 Key的offset */ \
    int64_t valueOffset; /* mm2 Value的offset*/ \
    int64_t taskId; \
    int64_t multiCoreInnerIdx = 0; \
    int64_t attentionOutOffset; \
    int64_t s1SizeAcc; /* 对于非TND场景 = boIdx * pseInfo.s2Size; TND场景等于前面boIdx个batch的s2之和（每个batch的s2不同）*/ \
    int64_t s2SizeAcc; /* 对于非TND场景 = boIdx * pseInfo.s2Size; TND场景等于前面boIdx个batch的s2之和（每个batch的s2不同）*/ \
    int32_t actualS1Size; /* 非TND场景=总s1Size, Tnd场景下当前batch对应的s1 */ \
    int32_t actualS2Size; /* 非TND场景=总s2Size, Tnd场景下当前batch对应的s2 */ \
    int64_t preTokensPerBatch; /* vector2 左上顶点的pretoken */ \
    int64_t nextTokensPerBatch; /* vector2 左上顶点的nexttoken */ \
    uint8_t taskIdMod2; \
    uint8_t taskIdMod3; \
    uint8_t multiCoreIdxMod2 = 0; \
    uint8_t multiCoreIdxMod3 = 0; \
    int64_t sOuterOffset; \
    int64_t mOuterOffset;

struct RunInfo {
    COMMON_RUN_INFO;
    // 推理新增
    // lse 输出offset
    int64_t softmaxLseOffset;

    // FD相关
    int64_t flashDecodeS2Idx;

    int64_t qSNumInOneBlock;
    int64_t oriKvLoopEndIdx;
    int64_t cmpKvLoopEndIdx;
};

#define COMMON_CONST_INFO \
    /* 全局的基本块信息 */ \
    uint32_t bSize; \
    uint32_t needInit; \
    uint32_t s1BaseSize; \
    uint32_t s2BaseSize; \
    int64_t dSize; /* query d 512 */ \
    int64_t dSizeV; /* key d 512 */ \
    int64_t dSizeVInput; /* key inpue d 640 = rope + nope + scale + pad */ \
    int64_t dBasicBlock; \
    int64_t dSizeNope; /* key nope d 448 */ \
    int64_t dSizeRope; /* key rope d 64 */ \
    int64_t tileSize; /* 64 */ \
    int64_t sparseMode = 3; \
    int64_t gSize; /* g轴的大小 */ \
    int64_t n2Size; \
    int64_t s1Size; /* s1总大小 */ \
    int64_t s2Size; /* s2总大小 */ \
    /* 轴的乘积 */ \
    int64_t s1D; \
    int64_t gS1D; \
    int64_t n2GS1D; \
    int64_t s2D; \
    int64_t n2S2D; \
    int64_t s1Dv; \
    int64_t gS1Dv; \
    int64_t n2GS1Dv; \
    int64_t s2Dv; \
    int64_t n2S2Dv; \
    int64_t s1S2; \
    int64_t gS1; \
    int64_t gD; \
    int64_t n2D; \
    int64_t bN2D; \
    int64_t gDv; \
    int64_t n2Dv; \
    int64_t bN2Dv; \
    int64_t n2G; \
    int64_t n2GD; \
    int64_t bN2GD; \
    int64_t n2GDv; \
    int64_t bN2GDv; \
    int64_t gS2; \
    int64_t s1Dr; \
    int64_t gS1Dr; \
    int64_t n2GS1Dr; \
    int64_t s2Dr; \
    int64_t n2S2Dr; \
    int64_t gDr; \
    int64_t n2Dr; \
    int64_t bN2Dr; \
    int64_t n2GDr; \
    int64_t bN2GDr; \
    int32_t s2BaseN2D; \
    int32_t s1BaseN2GD; \
    int64_t s2BaseBN2D; \
    int64_t s1BaseBN2GD; \
    int32_t s1BaseD; \
    int32_t s2BaseD; \
    int64_t s2BaseN2Dv; \
    int64_t s2BaseBN2Dv; \
    int64_t s1BaseN2GDv; \
    int64_t s1BaseBN2GDv; \
    int32_t s1BaseDv; \
    int32_t s2BaseDv; \
    /* matmul跳读参数 */ \
    int64_t mm1Ka; \
    int64_t mm1Kb; \
    int64_t mm2Kb; \
    /* dq 或者attentionOut的Stride */ \
    int64_t attentionOutStride; \
    uint32_t aivIdx; \
    uint8_t layoutType; \
    uint8_t subBlockIdx;\
    bool softMaxCheckRes; \
    int64_t matmulMSize; /* 在matmul运算中，左矩阵的M轴大小需要区分GS1合轴与不合轴的情况 */ 

#define INFER_CONST_INFO \
    /* 推理 */ \
    bool isActualLenDimsNull; /* 判断是否有actualseq */ \
    bool isActualLenDimsKVNull; /* 判断是否有actualseq_kv */ \
    bool isGqa; \
    bool isPfaGS1Merge; /* 判断是否为PFA GS1合轴 */\
    \
    uint32_t actualSeqLenSize; /* 用户输入的actualseq的长度 */ \
    uint32_t actualSeqLenKVSize; /* 用户输入的actualseq_kv的长度 */ \
    /* service mm1 mm2 pageAttention */ \
    uint32_t blockSize; \
    uint32_t paLayoutType; \
    uint32_t maxBlockNumPerBatch; /* for v0 */ \
    uint32_t oriMaxBlockNumPerBatch; \
    uint32_t cmpMaxBlockNumPerBatch; \
    bool rsvd1; \
    bool isSoftmaxLseEnable; \
    /* FD */ \
    int64_t sInnerLoopSize; /* FD s2总大小 */ \
    int64_t actualCombineLoopSize; /* 实际规约块数 */ \
    int64_t splitKVNum; \
    int64_t oriWinLeft; \
    int64_t oriWinRight; \
    uint32_t sparseBlockCount; \
    int64_t sparseBlockSize; \
    float softmaxScale; \
    int64_t cmpRatio; \

#define CV_SHARED_PARAMS \
    /* base params */ \
    uint32_t s1BaseSize; \
    uint32_t s2BaseSize; \
    uint32_t bSize;  \
    uint32_t n2Size;  \
    uint32_t gSize;  \
    uint32_t s1Size;  \
    uint32_t s2Size;  \
    uint32_t dSize : 10;  \
    uint32_t dSizeV : 10;  \
    uint32_t dSizeVInput : 12;  \
    uint32_t sparseBlockCount; \
    int64_t sparseBlockSize; \
    float softmaxScale; \
    int64_t cmpRatio; \
    /* special params */  \
    int64_t preTokens;  \
    int64_t nextTokens;  \
    uint64_t oriMaskMode; \
    uint64_t cmpMaskMode; \
    int64_t oriWinLeft; \
    int64_t oriWinRight; \
    int64_t kvQuantMode; \
    int64_t tileSize; \
    int64_t ropeHeadDim; \
    uint32_t coreNum;  \
    uint32_t needInit : 1; \
    uint32_t layoutType : 4;  \
    uint32_t isActualSeqLengthsNull : 10; \
    uint32_t isActualSeqLengthsKVNull : 16; \
    uint32_t actualSeqLengthsSize; \
    uint32_t actualSeqLengthsKVSize; \
    uint32_t splitKVNum; /* FD */ \
    /* pa params */  \
    uint32_t blockSize; \
    uint32_t oriMaxBlockNumPerBatch; \
    uint32_t cmpMaxBlockNumPerBatch; 


struct ConstInfo{
    // BUFFER的字节数
    static constexpr uint32_t BUFFER_SIZE_BYTE_32B = 32;
    static constexpr uint32_t BUFFER_SIZE_BYTE_64B = 64;
    static constexpr uint32_t BUFFER_SIZE_BYTE_256B = 256;
    static constexpr uint32_t BUFFER_SIZE_BYTE_512B = 512;
    static constexpr uint32_t BUFFER_SIZE_BYTE_1K = 1024;
    static constexpr uint32_t BUFFER_SIZE_BYTE_2K = 2048;
    static constexpr uint32_t BUFFER_SIZE_BYTE_4K = 4096;
    static constexpr uint32_t BUFFER_SIZE_BYTE_8K = 8192;
    static constexpr uint32_t BUFFER_SIZE_BYTE_16K = 16384;
    static constexpr uint32_t BUFFER_SIZE_BYTE_32K = 32768;
    COMMON_CONST_INFO;
    INFER_CONST_INFO;
};

/* only support b32 or b64 */
struct CVSharedParams {
    CV_SHARED_PARAMS;
};
}

#endif // FLASH_ATTENTION_UTIL_REGBASE_H
