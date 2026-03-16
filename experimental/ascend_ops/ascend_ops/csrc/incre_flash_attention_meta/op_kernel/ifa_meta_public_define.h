#ifndef IFA_META_PUBLIC_DEFINE_H
#define IFA_META_PUBLIC_DEFINE_H

#include <string>
#include <vector>
#include <array>
#include <cstdio>
#include <math.h>
#include <numeric>
#include <algorithm>

namespace aicpu::kernels {

inline constexpr uint32_t MAX_CORE_NUM = 50;

enum class Layout {
    BSH = 0,
    BSND = 0,
    BNSD = 1,
    NZ = 2,
    TND = 3,
    NBSD = 4,
    NTD = 5,
    BUTT
};

struct IncreFlashAttentionMetadata {
    // FA
    uint32_t usedCoreNum = 0U;
    uint32_t formerCoreNum = 0U;
    uint32_t sInnerLoopTimes = 0U;
    uint32_t singleProcessSInnerSize = 0U;
    uint32_t singleProcessSInnerSizeTail = 0U;
    uint32_t blockSplitBn2Range = 0U;
    uint32_t tailSplitedBatchRange = 0U;
    uint32_t groupSplitSize = 0U;
    uint32_t s1SplitSize = 0U;
    uint32_t startIdxEachCore[MAX_CORE_NUM];
    // FD
    uint32_t s2 = 0U;
    uint32_t sInnerLoopSize = 0U;
    uint32_t accumOutSize = 0U;
    uint32_t logSumExpSize = 0U;
};

struct IncreFlashAttentionMetadataArgs {
    uint32_t aicCoreNum;
    uint32_t aivCoreNum;
    uint32_t batchSize;
    uint32_t querySeqSize;
    uint32_t queryHeadNum;
    uint32_t keySeqSize;
    uint32_t keyHeadNum;
    uint32_t headDim;
    uint32_t blockSize;
    uint32_t maxBlockNumPerBatch;
    bool isAccumSeqQ = false;
    int32_t *actSeqQLen = nullptr;
    uint64_t actSeqQLenDim;
    bool isAccumSeqKv = false;
    int32_t *actSeqKvLen = nullptr;
    uint64_t actSeqKvLenDim;
    Layout layoutQuery = Layout::BUTT;
    Layout layoutKey = Layout::BUTT;
    int8_t* metaData = nullptr;
};

}

#endif