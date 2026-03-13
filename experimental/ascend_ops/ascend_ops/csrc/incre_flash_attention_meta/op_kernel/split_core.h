
#ifndef SPLIT_CORE_H
#define SPLIT_CORE_H

#include <string>
#include <vector>
#include <array>

namespace aicpu::kernels {

inline constexpr uint32_t MAX_CORE_NUM = 50;

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
    int8_t *actSeqQLen = nullptr;
    uint64_t actSeqQLenDim;
    int8_t *actSeqKvLen = nullptr;
    uint64_t actSeqKvLenDim;
    const char *layoutQuery = nullptr;
    const char *layoutKey = nullptr;
    int8_t* metaData = nullptr;
};

template <typename T> 
inline auto Align(T num, T rnd) -> T
{
    return (((rnd) == 0) ? 0 : (((num) + (rnd) - 1) / (rnd) * (rnd)));
}

static int64_t CeilDivision(int64_t num1, int64_t num2)
{
    if (num2 == 0) {
        return 0;
    }
    return (num1 + num2 - 1) / num2;
}

class SplitCore {
public:
    SplitCore() = default;
    ~SplitCore() = default;
    uint32_t Compute(IncreFlashAttentionMetadataArgs *args);
private:
    // main
    void AcquireParam(IncreFlashAttentionMetadataArgs *args);
    void ParamsInit();
    bool BalanceSchedule();
    bool GenMetaData(IncreFlashAttentionMetadataArgs *args);

    // util
    bool CheckCoreOkFlag(uint32_t coreNum) const;
    bool IsFlashDecode(uint32_t coreNum) const;
    void CalcInnerSize(uint32_t seqSize);
    bool SplitBNS();
    bool SplitBN();
    bool SplitBN_V0();
    void GetEstimatedLoad(int64_t &estimatedLoad) const;
    std::vector<int64_t> InitSparseValidArray(int64_t actualLensDim, const int64_t *actualLens) const;
    bool BalanceLoad(const std::vector<int64_t> &sparseValidArray, int64_t totalSize, int64_t validAivNum,
        std::vector<int64_t> &localValue, std::vector<int64_t> &sparseStartIdx) const;
    void SetSparseStartIdx(const std::vector<int64_t> &sparseValidArray, int64_t totalSize, int64_t validAivNum,
        uint32_t *sparseStartIdx, int64_t splitFactorSize) const;
    void InitLoadValue(const std::vector<int64_t> &sparseValidArray, int64_t totalSize, int64_t validAivNum,
        const std::vector<int64_t> &sparseStartIdx, std::vector<int64_t> &localValue) const;
private:
    static constexpr uint32_t MAX_SPLIT_SIZE = 8192;

    // input
    uint32_t aicCoreNum_ = 24U;
    uint32_t aivCoreNum_ = 48U;
    uint32_t batchSize_ = 0U;
    uint32_t qSeqSize_ = 0U;
    uint32_t qHeadNum_ = 0U;
    uint32_t kvSeqSize_ = 0U;
    uint32_t kvHeadNum_ = 0U;
    uint32_t headDim_ = 0U;
    uint32_t blockSize_ = 0U;
    uint32_t maxBlockNumPerBatch_ = 0U;
    std::string layoutQuery_ = "BSND";
    std::string layoutKV_ = "BSND";
    std::string socVersion_ = "";
    int64_t *actSeqQLen_ = nullptr;
    int64_t actSeqQLenDim_ = 0;
    int64_t *actSeqKvLen_ = nullptr;
    int64_t actSeqKvLenDim_ = 0;

    // init param
    bool isSameActualseq_ = true;
    bool gqaMtpFlag_ = false;
    uint32_t groupNum_ = 0U;
    uint32_t sMax_ = 0U;
    uint32_t seqSize_ = 0U;
    int64_t maxActualseq_ = 0;

    uint32_t kvSplit_ = 0U;
    uint32_t s1Outer_ = 1U;
    uint32_t gOuter_ = 1U;

    // output
    uint32_t usedCoreNum_ = 0U;
    bool splitKVFlag_ = false;
    uint32_t groupSplitSize_ = 0U;
    uint32_t s1SplitSize_ = 0U;
    uint32_t sInnerSize_ = 0U;
    uint32_t sInnerLoopTimes_ = 0U;
    uint32_t sInnerSizeTail_ = 0U;
    uint32_t sInnerSizeAlign_ = 0U;
    uint32_t formerCoreNum_ = 0U;
    uint32_t blockSplitBn2Range_ = 0U;
    uint32_t tailSplitedBatchRange_ = 0U;
    uint32_t kvSplitPart_ = 0U;
    uint32_t startIdxEachCore_[MAX_CORE_NUM] = {};
};

}



#endif