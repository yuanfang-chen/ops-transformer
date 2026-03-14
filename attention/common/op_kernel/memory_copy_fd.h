#ifndef MEMORY_COPY_FD_H
#define MEMORY_COPY_FD_H
#include "fia_public_define.h"
#include "offset_calculator.h"
namespace fa_memory_copy_fd{

// BLOCK和REPEAT的字节数
constexpr uint64_t BYTE_BLOCK = 32UL;

enum class UbFormat
{
    GS1 = 0,
    S1G = 1
};

template <AttentionCommon::FIA_LAYOUT LAYOUT_T>
__aicore__ inline constexpr UbFormat GetOutUbFormat() {
    static_assert((LAYOUT_T == AttentionCommon::FIA_LAYOUT::BSH) ||
                  (LAYOUT_T == AttentionCommon::FIA_LAYOUT::BNSD) ||
                  (LAYOUT_T == AttentionCommon::FIA_LAYOUT::TND) ||
                  (LAYOUT_T == AttentionCommon::FIA_LAYOUT::NTD),
                  "Get OutAttention UB GmFormat fail, LAYOUT_T is incorrect");
    if constexpr (LAYOUT_T == AttentionCommon::FIA_LAYOUT::BSH || LAYOUT_T == AttentionCommon::FIA_LAYOUT::TND) {
        return UbFormat::S1G;
    } else if constexpr (LAYOUT_T == AttentionCommon::FIA_LAYOUT::BNSD || LAYOUT_T == AttentionCommon::FIA_LAYOUT::NTD) {
        return UbFormat::GS1;
    }
}


template <AttentionCommon::FIA_LAYOUT LAYOUT_T>
__aicore__ inline constexpr ActualSeqLensMode GetQActSeqMode() {
    if constexpr (LAYOUT_T == AttentionCommon::FIA_LAYOUT::TND || LAYOUT_T == AttentionCommon::FIA_LAYOUT::NTD) {
        return ActualSeqLensMode::ACCUM;
    } else {
        return ActualSeqLensMode::BY_BATCH;
    }
}
template <AttentionCommon::FIA_LAYOUT LAYOUT_T, const bool PAGE_ATTENTION>
__aicore__ inline constexpr ActualSeqLensMode GetKvActSeqMode() {
    if constexpr (PAGE_ATTENTION) {
        return ActualSeqLensMode::BY_BATCH;
    }
    if constexpr (LAYOUT_T == AttentionCommon::FIA_LAYOUT::TND || LAYOUT_T == AttentionCommon::FIA_LAYOUT::NTD) {
        return ActualSeqLensMode::ACCUM;
    } else {
        return ActualSeqLensMode::BY_BATCH;
    }
}


template <typename OUT_T>
struct FaUbTensor {
    LocalTensor<OUT_T> tensor;
    uint32_t rowCount;
    uint32_t colCount;
};


template <ActualSeqLensMode MODE>
class ActualSeqLensParser {
};

template <>
class ActualSeqLensParser<ActualSeqLensMode::ACCUM> {
public:
    __aicore__ inline ActualSeqLensParser() = default;

    __aicore__ inline void Init(GlobalTensor<uint64_t> actualSeqLengthsGm, uint32_t actualLenDims, uint64_t defaultVal = 0)
    {
        this->actualSeqLengthsGm = actualSeqLengthsGm;
        this->actualLenDims = actualLenDims;
    }

    __aicore__ inline uint64_t GetTBase(uint32_t bIdx) const
    {
        if (bIdx == 0) {
            return 0;
        }
        return actualSeqLengthsGm.GetValue(bIdx - 1);
    }

    __aicore__ inline uint64_t GetActualSeqLength(uint32_t bIdx) const
    {
        if (bIdx == 0) {
            return actualSeqLengthsGm.GetValue(0);
        }
        return (actualSeqLengthsGm.GetValue(bIdx) - actualSeqLengthsGm.GetValue(bIdx - 1));
    }

    __aicore__ inline uint64_t GetTSize() const
    {
        return actualSeqLengthsGm.GetValue(actualLenDims - 1);
    }
private:
    GlobalTensor<uint64_t> actualSeqLengthsGm;
    uint32_t actualLenDims = 0U;
};

template <>
class ActualSeqLensParser<ActualSeqLensMode::BY_BATCH> {
public:
    __aicore__ inline ActualSeqLensParser() = default;

    __aicore__ inline void Init(GlobalTensor<uint64_t> actualSeqLengthsGm, uint32_t actualLenDims, uint64_t defaultVal)
    {
        this->actualSeqLengthsGm = actualSeqLengthsGm;
        this->actualLenDims = actualLenDims;
        this->defaultVal = defaultVal;
    }

    __aicore__ inline uint64_t GetActualSeqLength(uint32_t bIdx) const
    {
        if (actualLenDims == 0) {
            return defaultVal;
        }
        if (actualLenDims == 1) {
            return actualSeqLengthsGm.GetValue(0);
        }
        return actualSeqLengthsGm.GetValue(bIdx);
    }

    __aicore__ inline uint32_t GetActualLenDims() const 
    {
        return actualLenDims;
    }
private:
    GlobalTensor<uint64_t> actualSeqLengthsGm;
    uint32_t actualLenDims = 0U;
    uint64_t defaultVal = 0U;
};

template <GmFormat FORMAT, FormatCategory CATEGORY>
struct OffsetCalculatorImpl {};

template <GmFormat FORMAT>
struct OffsetCalculatorImpl<FORMAT, FormatCategory::GM_Q_OUT_BNGSD> {
    GmLayout<FORMAT> gmLayout;
    ActualSeqLensParser<ActualSeqLensMode::BY_BATCH> actualSeqLensQParser;
    bool isQPaddingFlag = false;
    uint64_t qPaddingSize = 0;
    __aicore__ inline OffsetCalculatorImpl() = default;

    __aicore__ inline void Init(uint32_t b, uint32_t n2, uint32_t g, uint32_t s1, uint32_t d,
                                GlobalTensor<uint64_t> actualSeqLengthsGmQ, uint32_t actualLenQDims,
                                bool isQPaddingFlag = false, uint64_t qPaddingSize = 0)
    {
        this->isQPaddingFlag = isQPaddingFlag;
        this->qPaddingSize = qPaddingSize;
        if(actualLenQDims != 0) {
            actualSeqLensQParser.Init(actualSeqLengthsGmQ, actualLenQDims, 0);
        }
        gmLayout.MakeLayout(b, n2, g, s1, d);
    }

    __aicore__ inline uint64_t GetOffset(uint32_t bIdx, uint32_t n2Idx, uint32_t gIdx, uint32_t s1Idx, uint32_t dIdx)
    {
        if (isQPaddingFlag) {
            s1Idx += GetDimS1() - qPaddingSize - actualSeqLensQParser.GetActualSeqLength(bIdx);
        }
        uint64_t offset = bIdx * GetStrideB() + n2Idx * GetStrideN2() + gIdx * GetStrideG() + s1Idx * GetStrideS1() +
                          dIdx * GetStrideD();
        return offset;
    }

    // Get Stride
    __aicore__ inline uint64_t GetStrideB()
    {
        return AscendC::Std::get<0>(gmLayout.stride);
    }

    __aicore__ inline uint64_t GetStrideN2()
    {
        return AscendC::Std::get<1>(gmLayout.stride);
    }

    __aicore__ inline uint64_t GetStrideG()
    {
        return AscendC::Std::get<2>(gmLayout.stride); // 2:代表第3个维度，索引从0开始
    }

    __aicore__ inline uint64_t GetStrideS1()
    {
        return AscendC::Std::get<3>(gmLayout.stride); // 3:代表第4个维度，索引从0开始
    }

    __aicore__ inline uint64_t GetStrideD()
    {
        return AscendC::Std::get<4>(gmLayout.stride); // 4:代表第5个维度，索引从0开始
    }

    // Get Dim
    __aicore__ inline uint64_t GetDimB()
    {
        return AscendC::Std::get<0>(gmLayout.shape);
    }

    __aicore__ inline uint64_t GetDimN2()
    {
        return AscendC::Std::get<1>(gmLayout.shape);
    }

    __aicore__ inline uint64_t GetDimG()
    {
        return AscendC::Std::get<2>(gmLayout.shape); // 2:代表第3个维度，索引从0开始
    }

    __aicore__ inline uint64_t GetDimS1()
    {
        return AscendC::Std::get<3>(gmLayout.shape); // 3:代表第4个维度，索引从0开始
    }

    __aicore__ inline uint64_t GetDimD()
    {
        return AscendC::Std::get<4>(gmLayout.shape); // 4:代表第5个维度，索引从0开始
    }
};

template <GmFormat FORMAT>
struct OffsetCalculatorImpl<FORMAT, FormatCategory::GM_Q_OUT_TND> {
    GmLayout<FORMAT> gmLayout;
    ActualSeqLensParser<ActualSeqLensMode::ACCUM> actualSeqLensQParser;

    __aicore__ inline OffsetCalculatorImpl() = default;

    __aicore__ inline void Init(uint32_t n2, uint32_t g, uint32_t d, GlobalTensor<uint64_t> actualSeqLengthsGmQ,
                                uint32_t actualLenQDims)
    {
        actualSeqLensQParser.Init(actualSeqLengthsGmQ, actualLenQDims);
        gmLayout.MakeLayout(actualSeqLensQParser.GetTSize(), n2, g, d);
    }

    __aicore__ inline uint64_t GetOffset(uint32_t bIdx, uint32_t n2Idx, uint32_t gIdx, uint32_t s1Idx, uint32_t dIdx)
    {
        uint64_t tIdx = actualSeqLensQParser.GetTBase(bIdx) + s1Idx;
        uint64_t offset = tIdx * GetStrideT() + n2Idx * GetStrideN2() + gIdx * GetStrideG() + dIdx * GetStrideD();
        return offset;
    }

    // Get Stride
    __aicore__ inline uint64_t GetStrideT()
    {
        return AscendC::Std::get<0>(gmLayout.stride);
    }

    __aicore__ inline uint64_t GetStrideN2()
    {
        return AscendC::Std::get<1>(gmLayout.stride);
    }

    __aicore__ inline uint64_t GetStrideG()
    {
        return AscendC::Std::get<2>(gmLayout.stride); // 2:代表第3个维度，索引从0开始
    }

    __aicore__ inline uint64_t GetStrideD()
    {
        return AscendC::Std::get<3>(gmLayout.stride); // 3:代表第4个维度，索引从0开始
    }

    __aicore__ inline uint64_t GetStrideS1()
    {
        return GetStrideT();
    }

    // Get Dim
    __aicore__ inline uint64_t GetDimT()
    {
        return AscendC::Std::get<0>(gmLayout.shape);
    }

    __aicore__ inline uint64_t GetDimN2()
    {
        return AscendC::Std::get<1>(gmLayout.shape);
    }

    __aicore__ inline uint64_t GetDimG()
    {
        return AscendC::Std::get<2>(gmLayout.shape); // 2:代表第3个维度，索引从0开始
    }

    __aicore__ inline uint64_t GetDimD()
    {
        return AscendC::Std::get<3>(gmLayout.shape); // 3:代表第4个维度，索引从0开始
    }
};

template <GmFormat FORMAT>
struct OffsetCalculatorImpl<FORMAT, FormatCategory::GM_KV_BNSD> {
    GmLayout<FORMAT> gmLayout;
    ActualSeqLensParser<ActualSeqLensMode::BY_BATCH> actualSeqLensKVParser;
    bool isKvPaddingFlag = false;
    uint64_t kvPaddingSize = 0;

    __aicore__ inline OffsetCalculatorImpl() = default;

    __aicore__ inline void Init(uint32_t b, uint32_t n2, uint32_t s2, uint32_t d)
    {
        gmLayout.MakeLayout(b, n2, s2, d);
    }

    __aicore__ inline void Init(uint32_t b, uint32_t n2, uint32_t s2, uint32_t d, GlobalTensor<uint64_t> actualSeqLengthsGm,
                                uint32_t actualLenKvDims, bool isKvPaddingFlag = false, uint64_t kvPaddingSize = 0)
    {
        this->isKvPaddingFlag = isKvPaddingFlag;
        this->kvPaddingSize = kvPaddingSize;
        if(actualLenKvDims != 0) {
            actualSeqLensKVParser.Init(actualSeqLengthsGm, actualLenKvDims, 0);
        }
        gmLayout.MakeLayout(b, n2, s2, d);
    }

    __aicore__ inline uint64_t GetOffset(uint32_t bIdx, uint32_t n2Idx, uint32_t s2Idx, uint32_t dIdx)
    {
        if (isKvPaddingFlag) {
            s2Idx += GetDimS2() - kvPaddingSize - actualSeqLensKVParser.GetActualSeqLength(bIdx);
        }
        
        uint64_t offset = bIdx * GetStrideB() + n2Idx * GetStrideN2() + s2Idx * GetStrideS2() + dIdx * GetStrideD();
        return offset;
    }

    // Get Stride
    __aicore__ inline uint64_t GetStrideB()
    {
        return AscendC::Std::get<0>(gmLayout.stride);
    }

    __aicore__ inline uint64_t GetStrideN2()
    {
        return AscendC::Std::get<1>(gmLayout.stride);
    }

    __aicore__ inline uint64_t GetStrideS2()
    {
        return AscendC::Std::get<2>(gmLayout.stride); // 2:代表第3个维度，索引从0开始
    }

    __aicore__ inline uint64_t GetStrideD()
    {
        return AscendC::Std::get<3>(gmLayout.stride); // 3:代表第4个维度，索引从0开始
    }

    // Get Dim
    __aicore__ inline uint64_t GetDimB()
    {
        return AscendC::Std::get<0>(gmLayout.shape);
    }

    __aicore__ inline uint64_t GetDimN2()
    {
        return AscendC::Std::get<1>(gmLayout.shape);
    }

    __aicore__ inline uint64_t GetDimS2()
    {
        return AscendC::Std::get<2>(gmLayout.shape); // 2:代表第3个维度，索引从0开始
    }

    __aicore__ inline uint64_t GetDimD()
    {
        return AscendC::Std::get<3>(gmLayout.shape); // 3:代表第4个维度，索引从0开始
    }
};

template <GmFormat FORMAT>
struct OffsetCalculatorImpl<FORMAT, FormatCategory::GM_KV_TND> {
    GmLayout<FORMAT> gmLayout;
    ActualSeqLensParser<ActualSeqLensMode::ACCUM> actualSeqLensKVParser;

    __aicore__ inline OffsetCalculatorImpl() = default;

    __aicore__ inline void Init(uint32_t n2, uint32_t d, GlobalTensor<uint64_t> actualSeqLengthsGmKV,
                                uint32_t actualLenKVDims)
    {
        actualSeqLensKVParser.Init(actualSeqLengthsGmKV, actualLenKVDims);
        gmLayout.MakeLayout(actualSeqLensKVParser.GetTSize(), n2, d);
    }

    __aicore__ inline uint64_t GetOffset(uint32_t bIdx, uint32_t n2Idx, uint32_t s2Idx, uint32_t dIdx)
    {
        uint64_t tIdx = actualSeqLensKVParser.GetTBase(bIdx) + s2Idx;
        uint64_t offset = tIdx * GetStrideT() + n2Idx * GetStrideN2() + dIdx * GetStrideD();
        return offset;
    }

    // Get Stride
    __aicore__ inline uint64_t GetStrideT()
    {
        return AscendC::Std::get<0>(gmLayout.stride);
    }

    __aicore__ inline uint64_t GetStrideN2()
    {
        return AscendC::Std::get<1>(gmLayout.stride);
    }

    __aicore__ inline uint64_t GetStrideD()
    {
        return AscendC::Std::get<2>(gmLayout.stride); // 2:代表第3个维度，索引从0开始
    }

    __aicore__ inline uint64_t GetStrideS2()
    {
        return GetStrideT();
    }

    // Get Dim
    __aicore__ inline uint64_t GetDimT()
    {
        return AscendC::Std::get<0>(gmLayout.shape);
    }

    __aicore__ inline uint64_t GetDimN2()
    {
        return AscendC::Std::get<1>(gmLayout.shape);
    }

    __aicore__ inline uint64_t GetDimD()
    {
        return AscendC::Std::get<2>(gmLayout.shape); // 2:代表第3个维度，索引从0开始
    }
};

template <GmFormat FORMAT>
struct OffsetCalculatorImpl<FORMAT, FormatCategory::GM_KV_PA_BNBD> {
    GmLayout<FORMAT> gmLayout;
    BlockTableParser blockTableParser;

    __aicore__ inline OffsetCalculatorImpl() = default;

    __aicore__ inline void Init(uint32_t n2, uint32_t blockSize, uint32_t d, GlobalTensor<int32_t> blockTableGm,
                                uint32_t maxblockNumPerBatch)
    {
        blockTableParser.Init(blockTableGm, maxblockNumPerBatch);
        gmLayout.MakeLayout(n2, blockSize, d);
    }

    __aicore__ inline uint64_t GetOffset(uint32_t bIdx, uint32_t n2Idx, uint32_t s2Idx, uint32_t dIdx)
    {
        uint64_t blockIdxInBatch = s2Idx / GetBlockSize(); // 获取block table上的索引
        uint64_t bsIdx = s2Idx % GetBlockSize();           // 获取在单个块上超出的行数
        int32_t blockIdx = blockTableParser.GetBlockIdx(bIdx, blockIdxInBatch);
        uint64_t offset =
            blockIdx * GetStrideBlockNum() + n2Idx * GetStrideN2() + bsIdx * GetStrideBlockSize() + dIdx * GetStrideD();
        return offset;
    }

    // Get Stride
    __aicore__ inline uint64_t GetStrideBlockNum()
    {
        return AscendC::Std::get<0>(gmLayout.stride);
    }

    __aicore__ inline uint64_t GetStrideN2()
    {
        return AscendC::Std::get<1>(gmLayout.stride);
    }

    __aicore__ inline uint64_t GetStrideBlockSize()
    {
        return AscendC::Std::get<2>(gmLayout.stride); // 2:代表第3个维度，索引从0开始
    }

    __aicore__ inline uint64_t GetStrideD()
    {
        return AscendC::Std::get<3>(gmLayout.stride); // 3:代表第4个维度，索引从0开始
    }

    // Get Dim
    __aicore__ inline uint64_t GetN2()
    {
        return AscendC::Std::get<0>(gmLayout.shape);
    }

    __aicore__ inline uint64_t GetBlockSize()
    {
        return AscendC::Std::get<1>(gmLayout.shape);
    }

    __aicore__ inline uint64_t GetD()
    {
        return AscendC::Std::get<2>(gmLayout.shape); // 2:代表第3个维度，索引从0开始
    }
};

template <GmFormat FORMAT>
struct OffsetCalculatorImpl<FORMAT, FormatCategory::GM_KV_PA_NZ> {
    GmLayout<FORMAT> gmLayout;
    BlockTableParser blockTableParser;

    __aicore__ inline OffsetCalculatorImpl() = default;

    __aicore__ inline void Init(uint32_t n2, uint32_t blockSize, uint32_t d1, uint32_t d0,
                                GlobalTensor<int32_t> blockTableGm, uint32_t maxblockNumPerBatch)
    {
        blockTableParser.Init(blockTableGm, maxblockNumPerBatch);
        gmLayout.MakeLayout(n2, blockSize, d1, d0);
    }

    __aicore__ inline uint64_t GetOffset(uint32_t bIdx, uint32_t n2Idx, uint32_t s2Idx, uint32_t dIdx)
    {
        uint64_t blockIdxInBatch = s2Idx / GetBlockSize(); // 获取block table上的索引
        uint64_t bsIdx = s2Idx % GetBlockSize();           // 获取在单个块上超出的行数
        int32_t blockIdx = blockTableParser.GetBlockIdx(bIdx, blockIdxInBatch);

        uint32_t d1Idx = dIdx / GetD0();
        uint32_t d0Idx = dIdx % GetD0();
        uint64_t offset = blockIdx * GetStrideBlockNum() + n2Idx * GetStrideN2() +
                          d1Idx * GetStrideD1() + bsIdx * GetStrideBlockSize() + d0Idx * GetStrideD0();
        return offset;
    }

    // Get Stride
    __aicore__ inline uint64_t GetStrideBlockNum()
    {
        return AscendC::Std::get<0>(gmLayout.stride);
    }

    __aicore__ inline uint64_t GetStrideN2()
    {
        return AscendC::Std::get<1>(gmLayout.stride);
    }

    __aicore__ inline uint64_t GetStrideD1()
    {
        return AscendC::Std::get<2>(gmLayout.stride); // 2:代表第3个维度，索引从0开始
    }

    __aicore__ inline uint64_t GetStrideBlockSize()
    {
        return AscendC::Std::get<3>(gmLayout.stride); // 3:代表第4个维度，索引从0开始
    }

    __aicore__ inline uint64_t GetStrideD0()
    {
        return AscendC::Std::get<4>(gmLayout.stride); // 4:代表第5个维度，索引从0开始
    }

    // Get Dim
    __aicore__ inline uint64_t GetN2()
    {
        return AscendC::Std::get<0>(gmLayout.shape);
    }

    __aicore__ inline uint64_t GetD1()
    {
        return AscendC::Std::get<1>(gmLayout.shape);
    }

    __aicore__ inline uint64_t GetBlockSize()
    {
        return AscendC::Std::get<2>(gmLayout.shape); // 2:代表第3个维度，索引从0开始
    }

    __aicore__ inline uint64_t GetD0()
    {
        return AscendC::Std::get<3>(gmLayout.shape); // 3:代表第4个维度，索引从0开始
    }
};

template <GmFormat FORMAT>
struct OffsetCalculator : public OffsetCalculatorImpl<FORMAT, GmLayoutParams<FORMAT>::CATEGORY> {
};

// ----------------------------------------------CopyQueryGmToL1--------------------------------
template <typename Q_T, GmFormat FORMAT>
struct FaGmTensor {
    GlobalTensor<Q_T> gmTensor;
    OffsetCalculator<FORMAT> offsetCalculator;
};
template <typename OUT_T, GmFormat GM_FORMAT, UbFormat UB_FORMAT>
class CopyAttenOutUbToGm
{
public:
    __aicore__ inline void SafeStrideCopy(GlobalTensor<OUT_T> gmTensor, const LocalTensor<OUT_T> ubTensor,
                                            uint32_t blockCount, uint32_t blockLen, uint32_t srcStride, uint64_t dstStride)
    {
        DataCopyExtParams dataCopyParams;
        // B*S过大时，跳写参数dataCopyParams.dstStride(uint32_t)计算结果将溢出，使用for循环拷贝代替
        if (dstStride > UINT32_MAX) {
            uint64_t gmSingleStride = (dstStride + blockLen) / sizeof(OUT_T);
            uint64_t ubSingleStride = (srcStride * BYTE_BLOCK + blockLen) / sizeof(OUT_T);
            dataCopyParams.blockCount = 1;
            dataCopyParams.blockLen = blockLen;
            dataCopyParams.srcStride = 0;
            dataCopyParams.dstStride = 0; // 单位为Byte
            for (uint32_t i = 0; i < blockCount; i++) {
                DataCopyPad(gmTensor[i * gmSingleStride], ubTensor[i * ubSingleStride], dataCopyParams);
            }
        } else {
            // dataCopyParams.dstStride(uint32_t)没有溢出时，进行跳写
            dataCopyParams.blockCount = blockCount;
            dataCopyParams.blockLen = blockLen;
            dataCopyParams.srcStride = srcStride;
            dataCopyParams.dstStride = dstStride; // 单位为Byte
            DataCopyPad(gmTensor, ubTensor, dataCopyParams);
        }
    }
    __aicore__ inline void operator()(FaGmTensor<OUT_T, GM_FORMAT> &dstTensor,
                                      FaUbTensor<OUT_T> &srcTensor,
                                      GmCoord &gmCoord)
    {
        if constexpr (UB_FORMAT == UbFormat::GS1) {
            OffsetCalculator<GM_FORMAT> &offsetCalculator = dstTensor.offsetCalculator;
            uint32_t s1Size = 0;
            if constexpr (GmLayoutParams<GM_FORMAT>::CATEGORY == FormatCategory::GM_Q_OUT_TND) {
                s1Size = offsetCalculator.actualSeqLensQParser.GetActualSeqLength(gmCoord.bIdx);
            } else {
                if( offsetCalculator.actualSeqLensQParser.GetActualLenDims() != 0 ) {
                    s1Size = offsetCalculator.actualSeqLensQParser.GetActualSeqLength(gmCoord.bIdx);
                } else {
                    s1Size = offsetCalculator.GetDimS1();
                }
            }
            uint32_t gIdxStart = gmCoord.gS1Idx / s1Size;
            uint32_t s1IdxStart = gmCoord.gS1Idx % s1Size;
            uint32_t gIdxEnd = (gmCoord.gS1Idx + gmCoord.gS1DealSize) / s1Size;
            uint32_t s1IdxEnd = (gmCoord.gS1Idx + gmCoord.gS1DealSize) % s1Size;

            uint64_t attenOutGmbaseOffset = offsetCalculator.GetOffset(gmCoord.bIdx, gmCoord.n2Idx, gIdxStart, 0, 0);

            // 处理第一个S
            uint32_t headS1 = 0;
            if (gIdxStart == gIdxEnd) {
                headS1 = s1IdxEnd - s1IdxStart;
            } else {
                headS1 = s1Size - s1IdxStart;
            }
            uint64_t gmOffset = attenOutGmbaseOffset + s1IdxStart * offsetCalculator.GetStrideS1();
            uint64_t ubOffset = 0;
            uint32_t blockCount = headS1;
            uint32_t blockLen = gmCoord.dDealSize * sizeof(OUT_T);
            uint32_t srcStride = (srcTensor.colCount - gmCoord.dDealSize) / (BYTE_BLOCK / sizeof(OUT_T));
            uint64_t dstStride = (offsetCalculator.GetStrideS1() - gmCoord.dDealSize) * sizeof(OUT_T); // 单位为Byte
            SafeStrideCopy(dstTensor.gmTensor[gmOffset], srcTensor.tensor[ubOffset], blockCount, blockLen, srcStride,
                            dstStride);

            if (gIdxEnd - gIdxStart >= 1) {
                // 处理中间块
                gmOffset = attenOutGmbaseOffset + offsetCalculator.GetStrideG();
                ubOffset = headS1 * srcTensor.colCount;
                for (uint32_t i = gIdxStart + 1; i < gIdxEnd; i++) {
                    blockCount = s1Size;
                    SafeStrideCopy(dstTensor.gmTensor[gmOffset], srcTensor.tensor[ubOffset], blockCount, blockLen,
                                    srcStride, dstStride);
                    gmOffset += offsetCalculator.GetStrideG();
                    ubOffset += s1Size * srcTensor.colCount;
                }

                // 处理尾块
                if (s1IdxEnd > 0) {
                    blockCount = s1IdxEnd;
                    SafeStrideCopy(dstTensor.gmTensor[gmOffset], srcTensor.tensor[ubOffset], blockCount, blockLen,
                                    srcStride, dstStride);
                }
            }
        } else if constexpr (UB_FORMAT == UbFormat::S1G) {
            OffsetCalculator<GM_FORMAT> &offsetCalculator = dstTensor.offsetCalculator;
            uint32_t s1IdxStart = gmCoord.gS1Idx / offsetCalculator.GetDimG();
            uint32_t gIdxStart = gmCoord.gS1Idx % offsetCalculator.GetDimG();
            uint32_t s1IdxEnd = (gmCoord.gS1Idx + gmCoord.gS1DealSize) / offsetCalculator.GetDimG();
            uint32_t gIdxEnd = (gmCoord.gS1Idx + gmCoord.gS1DealSize) % offsetCalculator.GetDimG();

            uint64_t attenOutGmbaseOffset = offsetCalculator.GetOffset(gmCoord.bIdx, gmCoord.n2Idx, 0, s1IdxStart, 0);

            // 处理第一个S
            uint32_t headSize = 0;
            if (s1IdxStart == s1IdxEnd) {
                headSize = gIdxEnd - gIdxStart;
            } else {
                headSize = offsetCalculator.GetDimG() - gIdxStart;
            }
            uint64_t gmOffset = attenOutGmbaseOffset + gIdxStart * offsetCalculator.GetStrideG();
            uint64_t ubOffset = 0;
            uint32_t blockCount = headSize;
            uint32_t blockLen = gmCoord.dDealSize * sizeof(OUT_T);
            uint32_t srcStride = (srcTensor.colCount - gmCoord.dDealSize) / (BYTE_BLOCK / sizeof(OUT_T));
            uint64_t dstStride = (offsetCalculator.GetStrideG() - gmCoord.dDealSize) * sizeof(OUT_T); // 单位为Byte
            SafeStrideCopy(dstTensor.gmTensor[gmOffset], srcTensor.tensor[ubOffset], blockCount, blockLen, srcStride,
                            dstStride);

            if (s1IdxEnd - s1IdxStart >= 1) {
                // 处理中间块
                gmOffset = attenOutGmbaseOffset + offsetCalculator.GetStrideS1();
                ubOffset = ((uint64_t)headSize) * ((uint64_t)srcTensor.colCount);
                for (uint32_t i = s1IdxStart + 1; i < s1IdxEnd; i++) {
                    blockCount = offsetCalculator.GetDimG();
                    SafeStrideCopy(dstTensor.gmTensor[gmOffset], srcTensor.tensor[ubOffset], blockCount, blockLen,
                                    srcStride, dstStride);
                    gmOffset += offsetCalculator.GetStrideS1();
                    ubOffset += offsetCalculator.GetDimG() * srcTensor.colCount;
                }

                // 处理尾块
                if (gIdxEnd > 0) {
                    blockCount = gIdxEnd;
                    SafeStrideCopy(dstTensor.gmTensor[gmOffset], srcTensor.tensor[ubOffset], blockCount, blockLen,
                                    srcStride, dstStride);
                }
            }
        }
    }
};
};
#endif