#include <string>
#include <vector>
#include <array>
#include <cstdio>
#include <math.h>
#include <numeric>
#include <algorithm>
#include "./acl/acl.h"
#include "aicpu_api.h"
#include "ifa_meta_public_define.h"

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
    uint32_t Compute(aicpu::kernels::IncreFlashAttentionMetadataArgs *args);
    void AcquireParam(aicpu::kernels::IncreFlashAttentionMetadataArgs *args);
    void ParamsInit();
    bool BalanceSchedule();
private:
    // main
    bool GenMetaData(aicpu::kernels::IncreFlashAttentionMetadataArgs *args);

    // util
    bool CheckCoreOkFlag(uint32_t coreNum) const;
    bool IsFlashDecode(uint32_t coreNum) const;
    void CalcInnerSize(uint32_t seqSize);
    bool SplitBNS();
    bool SplitBN();
    bool SplitBN_V0();
    void GetEstimatedLoad(int64_t &estimatedLoad) const;
    std::vector<int64_t> InitSparseValidArray(int64_t actualLensDim, const int32_t *actualLens) const;
    bool BalanceLoad(const std::vector<int64_t> &sparseValidArray, int64_t totalSize, int64_t validAivNum,
        std::vector<int64_t> &localValue, std::vector<int64_t> &sparseStartIdx) const;
    void SetSparseStartIdx(const std::vector<int64_t> &sparseValidArray, int64_t totalSize, int64_t validAivNum,
        uint32_t *sparseStartIdx, int64_t splitFactorSize) const;
    void InitLoadValue(const std::vector<int64_t> &sparseValidArray, int64_t totalSize, int64_t validAivNum,
        const std::vector<int64_t> &sparseStartIdx, std::vector<int64_t> &localValue) const;
public:
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
    aicpu::kernels::Layout layoutQuery_ = aicpu::kernels::Layout::BUTT;
    aicpu::kernels::Layout layoutKV_ = aicpu::kernels::Layout::BUTT;
    std::string socVersion_ = "";
    bool isAccumSeqQ_ = false;
    int32_t *actSeqQLen_ = nullptr;
    int64_t actSeqQLenDim_ = 0;
    bool isAccumSeqKv_ = false;
    int32_t *actSeqKvLen_ = nullptr;
    int64_t actSeqKvLenDim_ = 0;

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
    uint32_t startIdxEachCore_[aicpu::kernels::MAX_CORE_NUM] = {};
private:
    static constexpr uint32_t MAX_SPLIT_SIZE = 8192;

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
};

uint32_t SplitCore::Compute(aicpu::kernels::IncreFlashAttentionMetadataArgs *args)
{
    AscendC::printf("AcquireParam\n");
    printf("AcquireParam\n");
    AcquireParam(args);
    AscendC::printf("ParamsInit\n");
    ParamsInit();
    AscendC::printf("BalanceSchedule\n");
    BalanceSchedule();
    AscendC::printf("GenMetaData\n");
    GenMetaData(args);
    AscendC::printf("Compute Over\n");
    return 0;
}

void SplitCore::AcquireParam(aicpu::kernels::IncreFlashAttentionMetadataArgs *args)
{
    aicCoreNum_ = args->aicCoreNum;
    aivCoreNum_ = args->aivCoreNum;
    batchSize_ = args->batchSize;
    qSeqSize_ = args->querySeqSize;
    qHeadNum_ = args->queryHeadNum;
    kvSeqSize_ = args->keySeqSize;
    kvHeadNum_ = args->keyHeadNum;
    blockSize_ = args->blockSize;
    maxBlockNumPerBatch_ = args->maxBlockNumPerBatch;

    isAccumSeqQ_ = args->isAccumSeqQ;
    actSeqQLen_ = (int32_t *)args->actSeqQLen;
    actSeqQLenDim_ = args->actSeqQLenDim;
    isAccumSeqKv_ = args->isAccumSeqQ;
    actSeqKvLen_ = (int32_t *)args->actSeqKvLen;
    actSeqKvLenDim_ = args->actSeqKvLenDim;

    layoutQuery_ = args->layoutQuery;
    layoutKV_ = args->layoutKey;
}

void SplitCore::ParamsInit()
{
    if (qSeqSize_ > 1U && qSeqSize_ <= 16U) {
        gqaMtpFlag_ = true;
    }

    maxActualseq_ = qSeqSize_;
    if (actSeqKvLenDim_ > 0) {
        maxActualseq_ = actSeqQLen_[0];

        for (int64_t i = 1; i < actSeqKvLenDim_; ++i) {
            int32_t tmp = (isAccumSeqKv_) ? actSeqKvLen_[i] - actSeqKvLen_[i - 1] : actSeqKvLen_[i];
            if (maxActualseq_ < tmp) {
                maxActualseq_ = tmp;
            }
            if (tmp != actSeqKvLen_[0]) {
                isSameActualseq_ = false;
            }
        }
    }

    groupNum_ = qHeadNum_ / kvHeadNum_;
    groupSplitSize_ = groupNum_;
    s1SplitSize_ = qSeqSize_;
    sMax_ = maxBlockNumPerBatch_ * blockSize_;
    seqSize_ = sMax_;
}

bool SplitCore::BalanceSchedule() 
{
    constexpr uint32_t gMax = 128U;

    if (gqaMtpFlag_) {
        if (layoutQuery_ == aicpu::kernels::Layout::BSH || layoutQuery_ == aicpu::kernels::Layout::BSND ||
            layoutQuery_ == aicpu::kernels::Layout::TND) {
            s1SplitSize_ = gMax / groupNum_;
            if (s1SplitSize_ > qSeqSize_) {
                s1SplitSize_ = qSeqSize_;
            }
            s1Outer_ = (qSeqSize_ + s1SplitSize_ - 1U) / s1SplitSize_;
        } else {
            // mla 模式支持G切分
            groupSplitSize_ = gMax / qSeqSize_;
            groupSplitSize_ -= (groupSplitSize_ & 1U);
            if (groupSplitSize_ > groupNum_) {
                groupSplitSize_ = groupNum_;
            }
            gOuter_ = (groupNum_ + groupSplitSize_ - 1U) / groupSplitSize_;
        }
    }

    if (IsFlashDecode(aicCoreNum_)) {
        splitKVFlag_ = true;
        kvSplit_++;
        return SplitBNS();
    }

    CalcInnerSize(seqSize_);
    return SplitBN();
}

bool SplitCore::IsFlashDecode(uint32_t coreNum) const
{
    if (socVersion_ == "910B" && maxActualseq_ <= 1024U) { // 1024, 经验值
        return false;
    }

    if (gqaMtpFlag_) {
        return false; // 投机推理FD待实现
    }

    if (CheckCoreOkFlag(coreNum)) {
        return true;
    }

    return false;
}

bool SplitCore::CheckCoreOkFlag(uint32_t coreNum) const
{
    float flashDecodeBNRatio = 0.5f; // 0.5, 经验值
    bool coreOkFlag = (static_cast<float>(batchSize_ * kvHeadNum_) <= flashDecodeBNRatio * coreNum);

    if (coreOkFlag && (groupNum_ == 1U)) {
        return true;
    }

    if (coreOkFlag && (maxActualseq_ >= 2048U)) { // 2048, 在flash decode + gqa时的经验值
        return true;
    }
    return false;
}

void SplitCore::CalcInnerSize(uint32_t seqSize)
{
    /**
     * sInnerSize：s2的切分大小，直接决定了MM的singleN/K和vector的切块大小，但当前切分也并非适用所有case。
     * 1、非GQA场景：按照vector的最大基本块8*1024进行切分，sInnerSize=8192
     * 2、GQA场景：(1) 非伪量化：vector计算比较少，cube MTE2bound，
     *                          因此，cube发射大块，减少通信次数。sInnerSize=8192
     *            (2) 伪量化：vector比较重，尽量较少vector的循环次数,
     *                          因此，cube发小块，期望vector尽量被cube的mte2掩盖。sInnerSize=1024
     */
    sInnerSize_ = 1024U;

    // PA特性泛化场景，blockSize_可能为112等值，无法被sInnerSize_整除，当step*base跨block时，搬运处理复杂，通过向下对齐避免
    if (blockSize_ != 0U) {
        if (sInnerSize_ % blockSize_ != 0U) {
            sInnerSize_ = (sInnerSize_ / blockSize_) * blockSize_;
        }
    }

    sInnerLoopTimes_ = (seqSize + sInnerSize_ - 1U) / sInnerSize_;
    sInnerSize_ = Align(sInnerSize_, 2U);
    sInnerSizeTail_ = seqSize - (sInnerLoopTimes_ - 1U) * sInnerSize_;
    sInnerSizeAlign_ = Align(sInnerSize_, 64U); // 元素个数按照基本块大小对齐
}

bool SplitCore::SplitBNS()
{
    formerCoreNum_ = 0U;
    blockSplitBn2Range_ = 1U;
    tailSplitedBatchRange_ = 1U;

    uint32_t bn = batchSize_ * kvHeadNum_;
    kvSplitPart_ = aicCoreNum_ / bn;
    while (((maxActualseq_ / kvSplitPart_) < 512U) && (kvSplitPart_ > 1U)) { // 512, 经验值
        kvSplitPart_--;
    }

    usedCoreNum_ = bn * kvSplitPart_;
    uint32_t computeSeqSize = (seqSize_ + (kvSplitPart_ - 1U)) / kvSplitPart_;
    computeSeqSize = Align(computeSeqSize, blockSize_);
    computeSeqSize = Align(computeSeqSize, 2U);
    CalcInnerSize(computeSeqSize);
    return true;
}

bool SplitCore::SplitBN()
{
    uint32_t bn;
    if (layoutQuery_ == aicpu::kernels::Layout::BSH || layoutQuery_ == aicpu::kernels::Layout::BSND ||
        layoutQuery_ == aicpu::kernels::Layout::TND) {
        bn = batchSize_ * kvHeadNum_ * s1Outer_;
    } else {
        bn = batchSize_ * kvHeadNum_ * gOuter_;
    }

    for (auto& elem : startIdxEachCore_) {
        elem = bn;
    }

    if (actSeqKvLenDim_ == 1U || bn <= aicCoreNum_ || isSameActualseq_) {
        return SplitBN_V0();
    }

    std::vector<int64_t> validArray = InitSparseValidArray(actSeqKvLenDim_, actSeqKvLen_);
    SetSparseStartIdx(validArray, bn, aicCoreNum_, startIdxEachCore_, CeilDivision(bn, aicCoreNum_));

    usedCoreNum_ = aicCoreNum_;
    return true;
}

bool SplitCore::SplitBN_V0()
{
    uint32_t bn;
    if (layoutQuery_ == aicpu::kernels::Layout::BSH || layoutQuery_ == aicpu::kernels::Layout::BSND ||
        layoutQuery_ == aicpu::kernels::Layout::TND) {
        bn = batchSize_ * kvHeadNum_ * s1Outer_;
    } else {
        bn = batchSize_ * kvHeadNum_ * gOuter_;
    }

    formerCoreNum_ = bn % aicCoreNum_;
    if (formerCoreNum_ == 0U) {
        blockSplitBn2Range_ = bn / aicCoreNum_;
        tailSplitedBatchRange_ = blockSplitBn2Range_;
    } else {
        blockSplitBn2Range_ = bn / aicCoreNum_ + 1U;
        tailSplitedBatchRange_ = blockSplitBn2Range_ - 1U;
    }

    usedCoreNum_ = bn > aicCoreNum_ ? aicCoreNum_ : bn;

    for (uint32_t i = 0U; i < formerCoreNum_; i++) {
        startIdxEachCore_[i] = blockSplitBn2Range_ * i;
    }

    uint32_t formerBase = formerCoreNum_ * blockSplitBn2Range_;
    for (uint32_t i = formerCoreNum_; i < usedCoreNum_; i++) {
        startIdxEachCore_[i] = formerBase + tailSplitedBatchRange_ * (i - formerCoreNum_);
    }
    return true;
}

std::vector<int64_t> SplitCore::InitSparseValidArray(int64_t actualLensDim, const int32_t *actualLens) const
{
    uint32_t outer;
    if (layoutQuery_ == aicpu::kernels::Layout::BSH || layoutQuery_ == aicpu::kernels::Layout::BSND ||
        layoutQuery_ == aicpu::kernels::Layout::TND) {
        outer = s1Outer_;
    } else {
        outer = gOuter_;
    }
    std::vector<int64_t> res((batchSize_ * kvHeadNum_ * outer));
    for (uint32_t b = 0U; b < batchSize_; b++) {
        for (uint32_t n = 0U; n < kvHeadNum_ * outer; n++) {
            int64_t estimatedLoad = static_cast<int64_t>(seqSize_);
            if (actualLensDim > 0) {
                estimatedLoad = actualLens[b];
                GetEstimatedLoad(estimatedLoad);
            }
            res[b * kvHeadNum_ * outer + n] = estimatedLoad;
        }
    }
    return res;
}

void SplitCore::GetEstimatedLoad(int64_t &estimatedLoad) const
{
    constexpr int64_t MSD_VEC_LOAD = 1024; 
    if (estimatedLoad < MSD_VEC_LOAD) { // 1024
        estimatedLoad = MSD_VEC_LOAD;
    } else if (estimatedLoad == 0) {
        // 空tensor输出也计入负载，否则当最后一个batch为空tensor时，分核算法会将该batch优化掉
        estimatedLoad = 1;
    }
}

void SplitCore::InitLoadValue(const std::vector<int64_t> &sparseValidArray, int64_t totalSize, int64_t validAivNum,
                              const std::vector<int64_t> &sparseStartIdx, std::vector<int64_t> &localValue) const
{
    for (int64_t idx = 0; idx < validAivNum; ++idx) {
        int64_t start = sparseStartIdx[idx];
        int64_t end = ((idx + 1) < validAivNum) ? sparseStartIdx[idx + 1] : totalSize;
        if (start < totalSize) {
            localValue[idx] = std::accumulate(sparseValidArray.begin() + start, sparseValidArray.begin() + end, 0);
        } else {
            break;
        }
    }
}

void SplitCore::SetSparseStartIdx(const std::vector<int64_t> &sparseValidArray, int64_t totalSize, int64_t validAivNum,
                                  uint32_t *sparseStartIdx, int64_t splitFactorSize) const
{
    // initLoad: 使用均分策略, 保证后续不会比均分差
    std::vector<int64_t> localSparseStartIdx(aicpu::kernels::MAX_CORE_NUM, totalSize);
    for (uint32_t idx = 0; idx < aicpu::kernels::MAX_CORE_NUM; ++idx) {
        localSparseStartIdx[idx] = std::min((idx * splitFactorSize), totalSize);
    }
    std::vector<int64_t> localValue(validAivNum, 0);
    InitLoadValue(sparseValidArray, totalSize, validAivNum, localSparseStartIdx, localValue);

    // 负载均衡粗调
    std::vector<int64_t> tmpLocalValue(validAivNum, 0);
    std::vector<int64_t> tmpsparseStartIdx(aicpu::kernels::MAX_CORE_NUM, totalSize);
    int64_t sparseArraySum = std::accumulate(sparseValidArray.begin(), sparseValidArray.end(), 0);
    int64_t avgVal = CeilDivision(sparseArraySum, validAivNum);

    tmpsparseStartIdx[0] = 0;
    for (uint32_t idx = 1; idx < aicpu::kernels::MAX_CORE_NUM; ++idx) {
        int64_t start = tmpsparseStartIdx[idx - 1];
        int64_t singleLoadValue = 0;
        tmpsparseStartIdx[idx] = start;
        while (singleLoadValue < avgVal && tmpsparseStartIdx[idx] < totalSize) {
            singleLoadValue += sparseValidArray[tmpsparseStartIdx[idx]];
            tmpsparseStartIdx[idx] += 1;
        }

        if ((start + 1) < tmpsparseStartIdx[idx]) {
            int64_t redoSingleLoadValue = singleLoadValue - sparseValidArray[tmpsparseStartIdx[idx] - 1];
            if ((singleLoadValue - avgVal) > (avgVal - redoSingleLoadValue)) {
                tmpsparseStartIdx[idx] = tmpsparseStartIdx[idx] - 1;
                singleLoadValue = redoSingleLoadValue;
            }
            sparseArraySum -= singleLoadValue;
            avgVal = CeilDivision(sparseArraySum, (validAivNum - static_cast<int64_t>(idx)));
        }
    }

    InitLoadValue(sparseValidArray, totalSize, validAivNum, tmpsparseStartIdx, tmpLocalValue);

    // 负载均衡精调
    while (BalanceLoad(sparseValidArray, totalSize, validAivNum, tmpLocalValue, tmpsparseStartIdx)) {
        // 根据负载均衡是否能得到更好预测结果决定是否结束循环
    }

    // exchange initLoad and 负载均衡
    if ((*std::max_element(localValue.begin(), localValue.end())) >
        (*std::max_element(tmpLocalValue.begin(), tmpLocalValue.end()))) {
        localSparseStartIdx.swap(tmpsparseStartIdx);
        localValue.swap(tmpLocalValue);
    }
    for (uint32_t idx = 0; idx < aicpu::kernels::MAX_CORE_NUM; ++idx) {
        sparseStartIdx[idx] = localSparseStartIdx[idx];
    }
}

bool SplitCore::BalanceLoad(const std::vector<int64_t> &sparseValidArray, int64_t totalSize, int64_t validAivNum,
                            std::vector<int64_t> &localValue, std::vector<int64_t> &sparseStartIdx) const
{
    // to avoid buffer overflow, or maybe sometimes we want to only verify single core
    int64_t maxVal = *std::max_element(localValue.begin(), localValue.end());
    int64_t tmpMaxVal = maxVal;

    // 从前往后遍历
    for (int64_t idx = 1; idx < validAivNum; ++idx) {
        int64_t start = sparseStartIdx[idx];
        if (start < totalSize && start > 0 && ((localValue[idx - 1] + sparseValidArray[start]) < maxVal)) {
            localValue[idx - 1] += sparseValidArray[start];
            localValue[idx] -= sparseValidArray[start];
            sparseStartIdx[idx] += 1;
        } else if (start == totalSize) {
            break;
        }
    }
    tmpMaxVal = *std::max_element(localValue.begin(), localValue.end());

    // 从后往前遍历
    for (int64_t idx = validAivNum - 1; idx > 0; --idx) {
        int64_t start = sparseStartIdx[idx];
        if (start == totalSize) {
            if (sparseStartIdx[idx - 1] == totalSize) {
                continue;
            }
            localValue[idx - 1] -= sparseValidArray[start - 1];
            localValue[idx] = sparseValidArray[start - 1];
            sparseStartIdx[idx] -= 1;
        } else if (start > 0) {
            if ((localValue[idx] + sparseValidArray[start - 1]) >= tmpMaxVal) {
                continue;
            }
            localValue[idx - 1] -= sparseValidArray[start - 1];
            localValue[idx] += sparseValidArray[start - 1];
            sparseStartIdx[idx] -= 1;
        } else {
            break;
        }
    }
    tmpMaxVal = *std::max_element(localValue.begin(), localValue.end());

    return (tmpMaxVal >= maxVal) ? false : true;
}

bool SplitCore::GenMetaData(aicpu::kernels::IncreFlashAttentionMetadataArgs *args)
{
    // constexpr uint32_t BYTE_BLOCK = 32UL;
    // aicpu::kernels::IncreFlashAttentionMetadata *metaData = (aicpu::kernels::IncreFlashAttentionMetadata *)args->metaData;
    // metaData->usedCoreNum = usedCoreNum_;
    // metaData->formerCoreNum = formerCoreNum_;
    // metaData->sInnerLoopTimes = sInnerLoopTimes_;
    // metaData->singleProcessSInnerSize = sInnerSize_;
    // metaData->singleProcessSInnerSizeTail = sInnerSizeTail_;
    // metaData->blockSplitBn2Range = blockSplitBn2Range_;
    // metaData->tailSplitedBatchRange = tailSplitedBatchRange_;
    // metaData->groupSplitSize = groupSplitSize_;
    // metaData->s1SplitSize = s1SplitSize_;
    // for (size_t i = 0; i < aicpu::kernels::MAX_CORE_NUM; ++i) {
    //     metaData->startIdxEachCore[i] = startIdxEachCore_[i];
    // }

    // // fd
    // uint32_t sInnerLoopSize = (maxActualseq_ + (kvSplitPart_ - 1U)) / kvSplitPart_;
    // sInnerLoopSize = Align(sInnerLoopSize, blockSize_);

    // uint32_t headDimAlign = Align(headDim_, BYTE_BLOCK);
    // metaData->s2 = kvSplitPart_;
    // metaData->sInnerLoopSize = sInnerLoopSize;
    // if (layoutQuery_ == aicpu::kernels::Layout::TND) {
    //     metaData->accumOutSize = batchSize_ * qSeqSize_ * qHeadNum_ * kvSplitPart_ * headDimAlign;
    //     metaData->logSumExpSize = 2U * batchSize_ * qHeadNum_ * kvSplitPart_ * qSeqSize_ * (BYTE_BLOCK / sizeof(float));
    // }  else {
    //     metaData->accumOutSize = batchSize_ * qHeadNum_ * kvSplitPart_ * headDimAlign;
    //     metaData->logSumExpSize = 2U * batchSize_ * qHeadNum_ * kvSplitPart_ * (BYTE_BLOCK / sizeof(float));
    // }

    // if (!splitKVFlag_) {
    //     metaData->s2 = 0U;
    // }
    constexpr uint32_t BYTE_BLOCK = 32UL;
    aicpu::kernels::IncreFlashAttentionMetadata *metaData = (aicpu::kernels::IncreFlashAttentionMetadata *)args->metaData;
    metaData->usedCoreNum = 18;
    metaData->formerCoreNum = 18;
    metaData->sInnerLoopTimes = 9;
    metaData->singleProcessSInnerSize = 1024;
    metaData->singleProcessSInnerSizeTail = 128;
    metaData->blockSplitBn2Range = 1;
    metaData->tailSplitedBatchRange = 0;
    metaData->groupSplitSize = 64;
    metaData->s1SplitSize = 1;
    metaData->startIdxEachCore[0] = 0;
    metaData->startIdxEachCore[1] = 1;
    metaData->startIdxEachCore[2] = 2;
    metaData->startIdxEachCore[3] = 3;
    metaData->startIdxEachCore[4] = 4;
    metaData->startIdxEachCore[5] = 5;
    metaData->startIdxEachCore[6] = 6;
    metaData->startIdxEachCore[7] = 7;
    metaData->startIdxEachCore[8] = 8;
    metaData->startIdxEachCore[9] = 9;
    metaData->startIdxEachCore[10] = 10;
    metaData->startIdxEachCore[11] = 11;
    metaData->startIdxEachCore[12] = 12;
    metaData->startIdxEachCore[13] = 13;
    metaData->startIdxEachCore[14] = 14;
    metaData->startIdxEachCore[15] = 15;
    metaData->startIdxEachCore[16] = 16;
    metaData->startIdxEachCore[17] = 17;
    metaData->startIdxEachCore[18] = 18;
    metaData->startIdxEachCore[19] = 18;
    metaData->startIdxEachCore[20] = 18;
    metaData->startIdxEachCore[21] = 27;
    metaData->startIdxEachCore[22] = 28;
    metaData->startIdxEachCore[23] = 29;
    metaData->startIdxEachCore[24] = 30;
    metaData->startIdxEachCore[25] = 30;
    metaData->startIdxEachCore[26] = 30;
    metaData->startIdxEachCore[27] = 30;
    metaData->startIdxEachCore[28] = 30;
    metaData->startIdxEachCore[29] = 30;
    metaData->startIdxEachCore[30] = 30;
    metaData->startIdxEachCore[31] = 30;
    metaData->startIdxEachCore[32] = 30;
    metaData->startIdxEachCore[33] = 30;
    metaData->startIdxEachCore[34] = 30;
    metaData->startIdxEachCore[35] = 30;
    metaData->startIdxEachCore[36] = 30;
    metaData->startIdxEachCore[37] = 30;
    metaData->startIdxEachCore[38] = 30;
    metaData->startIdxEachCore[39] = 30;
    metaData->startIdxEachCore[40] = 30;
    metaData->startIdxEachCore[41] = 30;
    metaData->startIdxEachCore[42] = 30;
    metaData->startIdxEachCore[43] = 30;
    metaData->startIdxEachCore[44] = 30;
    metaData->startIdxEachCore[45] = 30;
    metaData->startIdxEachCore[46] = 30;
    metaData->startIdxEachCore[47] = 30;
    metaData->startIdxEachCore[48] = 30;
    metaData->startIdxEachCore[49] = 30;

    // fd
    uint32_t sInnerLoopSize = (maxActualseq_ + (kvSplitPart_ - 1U)) / kvSplitPart_;
    sInnerLoopSize = Align(sInnerLoopSize, blockSize_);

    uint32_t headDimAlign = Align(headDim_, BYTE_BLOCK);
    metaData->s2 = 0;
    metaData->sInnerLoopSize = 65536;
    if (layoutQuery_ == aicpu::kernels::Layout::TND) {
        metaData->accumOutSize = 147456;
        metaData->logSumExpSize = 18432;
    }  else {
        metaData->accumOutSize = 147456;
        metaData->logSumExpSize = 18432;
    }

    if (!splitKVFlag_) {
        metaData->s2 = 0U;
    }
    return true;
}

void PrintInput(aicpu::kernels::IncreFlashAttentionMetadataArgs* arg_ptr)
{
    AscendC::printf("================== PrintInput!! ==================\n");
    AscendC::printf("aicCoreNum %u!!\n", arg_ptr->aicCoreNum);
    AscendC::printf("aivCoreNum %u!!\n", arg_ptr->aivCoreNum);
    AscendC::printf("batchSize %u!!\n", arg_ptr->batchSize);
    AscendC::printf("querySeqSize %u!!\n", arg_ptr->querySeqSize);
    AscendC::printf("queryHeadNum %u!!\n", arg_ptr->queryHeadNum);
    AscendC::printf("keySeqSize %u!!\n", arg_ptr->keySeqSize);
    AscendC::printf("keyHeadNum %u!!\n", arg_ptr->keyHeadNum);
    AscendC::printf("headDim %u!!\n", arg_ptr->headDim);
    AscendC::printf("blockSize %u!!\n", arg_ptr->blockSize);
    AscendC::printf("maxBlockNumPerBatch %u!!\n", arg_ptr->maxBlockNumPerBatch);
    AscendC::printf("isAccumSeqQ %u!!\n", arg_ptr->isAccumSeqQ);
    AscendC::printf("actSeqQLenDim %u!!\n", arg_ptr->actSeqQLenDim);
    AscendC::printf("actSeqQLen: [");
    for (size_t i = 0; i < arg_ptr->actSeqQLenDim; ++i) {
        AscendC::printf("%u, ", arg_ptr->actSeqQLen[i]);
    }
    AscendC::printf("]\n");
    AscendC::printf("isAccumSeqKv %u!!\n", arg_ptr->isAccumSeqKv);
    AscendC::printf("actSeqKvLenDim %u!!\n", arg_ptr->actSeqKvLenDim);
    AscendC::printf("actSeqKvLen: [");
    for (size_t i = 0; i < arg_ptr->actSeqQLenDim; ++i) {
        AscendC::printf("%u, ", arg_ptr->actSeqKvLen[i]);
    }
    AscendC::printf("]\n");
    AscendC::printf("layoutQuery %u!!\n", arg_ptr->layoutQuery);
    AscendC::printf("layoutKey %u!!\n", arg_ptr->layoutKey);
}

void PrintOutput(aicpu::kernels::IncreFlashAttentionMetadataArgs* arg_ptr)
{
    AscendC::printf("================== PrintOutput!! ==================\n");

    aicpu::kernels::IncreFlashAttentionMetadata *metaData = (aicpu::kernels::IncreFlashAttentionMetadata *)arg_ptr->metaData;
    // FA
    AscendC::printf("usedCoreNum %u!!\n", metaData->usedCoreNum);
    AscendC::printf("formerCoreNum %u!!\n", metaData->formerCoreNum);
    AscendC::printf("sInnerLoopTimes %u!!\n", metaData->sInnerLoopTimes);
    AscendC::printf("singleProcessSInnerSize %u!!\n", metaData->singleProcessSInnerSize);
    AscendC::printf("singleProcessSInnerSizeTail %u!!\n", metaData->singleProcessSInnerSizeTail);
    AscendC::printf("blockSplitBn2Range %u!!\n", metaData->blockSplitBn2Range);
    AscendC::printf("tailSplitedBatchRange %u!!\n", metaData->tailSplitedBatchRange);
    AscendC::printf("groupSplitSize %u!!\n", metaData->groupSplitSize);
    AscendC::printf("s1SplitSize %u!!\n", metaData->s1SplitSize);
    AscendC::printf("startIdxEachCore [");
    for (size_t i = 0; i < aicpu::kernels::MAX_CORE_NUM; ++i) {
        AscendC::printf("%u, ", metaData->startIdxEachCore[i]);
    }
    AscendC::printf("]\n");

    // FD
    AscendC::printf("s2 %u!!\n", metaData->s2);
    AscendC::printf("sInnerLoopSize %u!!\n", metaData->sInnerLoopSize);
    AscendC::printf("accumOutSize %u!!\n", metaData->accumOutSize);
    AscendC::printf("logSumExpSize %u!!\n", metaData->logSumExpSize);
}

extern "C" __global__ __aicpu__ uint32_t IncreFlashAttentionMetadataKernel(void *args)
{
    aicpu::kernels::IncreFlashAttentionMetadataArgs* arg_ptr = (aicpu::kernels::IncreFlashAttentionMetadataArgs *)args;
    AscendC::printf("Inside aicpu kernel!!\n");

    PrintInput(arg_ptr);
    AscendC::printf("aicpu::kernels::SplitCore balancer !!\n");
    SplitCore balancer {};
    AscendC::printf("aicpu::kernels::SplitCore balancer balance !!\n");
    balancer.Compute(arg_ptr);
    AscendC::printf("IncreFlashAttentionMetadataKernel over !!\n");
    PrintOutput(arg_ptr);

    return 0;
}
