
#ifndef COPYQTOL1_H
#define COPYQTOL1_H

#include "offset_calculator.h"

constexpr uint32_t HALF_SIZE_DIVISOR = 2;
constexpr uint32_t ND_MATRIX_STRIDE_LIMIT = 65536; // Mutil ND2NZ搬运时，Nd2NzParams支持的srcNdMatrixStride的取值范围为[0, 65536]，单位为元素

template <typename T>
__aicore__ inline void CopySingleMatrixNDToNZ(LocalTensor<T> l1Tensor, const GlobalTensor<T> gmTensor,
    uint32_t nValue, uint32_t dValue, uint32_t srcDValue, uint32_t dstNzC0Stride)
{
    Nd2NzParams nd2nzPara;
    nd2nzPara.ndNum = 1;
    nd2nzPara.nValue = nValue; //nd矩阵的行数
    if constexpr (IsSameType<T, int4b_t>::value) {
        nd2nzPara.dValue = dValue / HALF_SIZE_DIVISOR;
        nd2nzPara.srcDValue = srcDValue / HALF_SIZE_DIVISOR;
    } else {
        nd2nzPara.dValue = dValue; //nd矩阵的列数
        nd2nzPara.srcDValue = srcDValue; //同一nd矩阵相邻行起始地址间的偏移
    }
    nd2nzPara.dstNzC0Stride = dstNzC0Stride;
    nd2nzPara.dstNzNStride = 1;
    nd2nzPara.srcNdMatrixStride = 0;
    nd2nzPara.dstNzMatrixStride = 0;
    DataCopy(l1Tensor, gmTensor, nd2nzPara);
}

template <typename T>
__aicore__ inline void CopyMultiMatrixNDToNZ(LocalTensor<T> l1Tensor, const GlobalTensor<T> gmTensor,
    uint32_t srcNdMatrixNum, uint32_t srcNdMatrixStride, uint32_t dstNzMatrixStride, uint32_t nValue, uint32_t dValue, uint32_t srcDValue, uint32_t dstNzC0Stride)
{
    if (unlikely(srcNdMatrixStride > ND_MATRIX_STRIDE_LIMIT)) {
        uint64_t l1Offset = 0;
        uint64_t gmOffset = 0;
        for (uint32_t i = 0; i < srcNdMatrixNum; i++) {
            CopySingleMatrixNDToNZ(l1Tensor[l1Offset], gmTensor[gmOffset], nValue, dValue, srcDValue, dstNzC0Stride);
            gmOffset += srcNdMatrixStride;
            l1Offset += dstNzMatrixStride;
        }
    } else {
        Nd2NzParams nd2nzPara;
        nd2nzPara.ndNum = srcNdMatrixNum;
        nd2nzPara.nValue = nValue; //nd矩阵的行数
        if constexpr (IsSameType<T, int4b_t>::value) {
            nd2nzPara.dValue = dValue / HALF_SIZE_DIVISOR;
            nd2nzPara.srcDValue = srcDValue / HALF_SIZE_DIVISOR;
        } else {
            nd2nzPara.dValue = dValue; //nd矩阵的列数
            nd2nzPara.srcDValue = srcDValue; //同一nd矩阵相邻行起始地址间的偏移
        }
        nd2nzPara.dstNzC0Stride = dstNzC0Stride;
        nd2nzPara.dstNzNStride = 1;
        nd2nzPara.srcNdMatrixStride = srcNdMatrixStride;
        nd2nzPara.dstNzMatrixStride = dstNzMatrixStride;
        DataCopy(l1Tensor, gmTensor, nd2nzPara);
    }
}

template <typename Q_T, GmFormat GM_FORMAT, L1Format L1_FORMAT = L1Format::NZ>
class CopyQueryGmToL1 {
public:
    __aicore__ inline void operator()(FaL1Tensor<Q_T, L1_FORMAT> &dstTensor,
                                      FaGmTensor<Q_T, GM_FORMAT> &srcTensor,
                                      GmCoord &gmCoord)
    {
        if constexpr ((GM_FORMAT == GmFormat::BSNGD) || (GM_FORMAT == GmFormat::TNGD)) {
            ProcessS1G(dstTensor, srcTensor, gmCoord);
        } else if constexpr (GM_FORMAT == GmFormat::NGTD) {
            ProcessGS1(dstTensor, srcTensor, gmCoord);
        } else {
            ProcessContinuous(dstTensor, srcTensor, gmCoord);
        }
    }

private:
    __aicore__ inline void ProcessS1G(FaL1Tensor<Q_T, L1_FORMAT> &dstTensor, FaGmTensor<Q_T, GM_FORMAT> &srcTensor,
                                      GmCoord &gmCoord)
    {
        OffsetCalculator<GM_FORMAT> &offsetCalculator = srcTensor.offsetCalculator;
        uint32_t s1IdxStart = gmCoord.gS1Idx / offsetCalculator.GetDimG();
        uint32_t gIdxStart = gmCoord.gS1Idx % offsetCalculator.GetDimG();
        uint32_t s1IdxEnd = (gmCoord.gS1Idx + gmCoord.gS1DealSize) / offsetCalculator.GetDimG();
        uint32_t gIdxEnd = (gmCoord.gS1Idx + gmCoord.gS1DealSize) % offsetCalculator.GetDimG();

        uint64_t queryGmbaseOffset =
            offsetCalculator.GetOffset(gmCoord.bIdx, gmCoord.n2Idx, 0, s1IdxStart, gmCoord.dIdx);

        if (offsetCalculator.GetDimG() == 1) {
            CopySingleMatrixNDToNZ(dstTensor.tensor, srcTensor.gmTensor[queryGmbaseOffset], s1IdxEnd - s1IdxStart, gmCoord.dDealSize,
                                    offsetCalculator.GetStrideS1(), dstTensor.rowCount);
            return;
        }

        // 处理第一个S
        uint32_t headSize = 0;
        if (s1IdxStart == s1IdxEnd) {
            headSize = gIdxEnd - gIdxStart;
        } else {
            headSize = offsetCalculator.GetDimG() - gIdxStart;
        }

        uint64_t offset = queryGmbaseOffset + gIdxStart * offsetCalculator.GetDimD();
        CopySingleMatrixNDToNZ(dstTensor.tensor, srcTensor.gmTensor[offset], headSize, gmCoord.dDealSize,
                               offsetCalculator.GetStrideG(), dstTensor.rowCount);

        if (s1IdxEnd - s1IdxStart >= 1) {
            // 处理中间块
            uint64_t gmOffset = queryGmbaseOffset + offsetCalculator.GetStrideS1();
            uint64_t l1Offset = headSize * 16U;
            if (s1IdxEnd - s1IdxStart > 1) {
                CopyMultiMatrixNDToNZ(dstTensor.tensor[l1Offset], srcTensor.gmTensor[gmOffset],
                        s1IdxEnd - s1IdxStart - 1, offsetCalculator.GetStrideS1(), offsetCalculator.GetDimG() * 16U,
                        offsetCalculator.GetDimG(), gmCoord.dDealSize,
                        offsetCalculator.GetStrideG(), dstTensor.rowCount);
                gmOffset += (s1IdxEnd - s1IdxStart - 1) * offsetCalculator.GetStrideS1();
                l1Offset += (s1IdxEnd - s1IdxStart - 1) * offsetCalculator.GetDimG() * 16U;
            }

            // 处理尾块
            if (gIdxEnd > 0) {
                CopySingleMatrixNDToNZ(dstTensor.tensor[l1Offset], srcTensor.gmTensor[gmOffset], gIdxEnd,
                                       gmCoord.dDealSize, offsetCalculator.GetStrideG(), dstTensor.rowCount);
            }
        }
    }

    __aicore__ inline void ProcessContinuous(FaL1Tensor<Q_T, L1_FORMAT> &dstTensor,
                                             FaGmTensor<Q_T, GM_FORMAT> &srcTensor, GmCoord &gmCoord)
    {
        // B*N2*GS1*D
        OffsetCalculator<GM_FORMAT> &offsetCalculator = srcTensor.offsetCalculator;
        uint32_t gIdxStart = gmCoord.gS1Idx / offsetCalculator.GetDimS1();
        uint32_t s1IdxStart = gmCoord.gS1Idx % offsetCalculator.GetDimS1();

        uint64_t offset =
            offsetCalculator.GetOffset(gmCoord.bIdx, gmCoord.n2Idx, gIdxStart, s1IdxStart, gmCoord.dIdx);
        CopySingleMatrixNDToNZ(dstTensor.tensor, srcTensor.gmTensor[offset], gmCoord.gS1DealSize, gmCoord.dDealSize,
                               offsetCalculator.GetDimD(), dstTensor.rowCount);
    }

    __aicore__ inline void ProcessGS1(FaL1Tensor<Q_T, L1_FORMAT> &dstTensor, FaGmTensor<Q_T, GM_FORMAT> &srcTensor,
                                      GmCoord &gmCoord)
    {
        // N2*G*T(BS1)*D
        OffsetCalculator<GM_FORMAT> &offsetCalculator = srcTensor.offsetCalculator;
        uint64_t s1Size = 0;
        if constexpr (GmLayoutParams<GM_FORMAT>::CATEGORY == FormatCategory::GM_Q_OUT_TND) {
            s1Size = offsetCalculator.actualSeqLensQParser.GetActualSeqLength(gmCoord.bIdx);
        } else {
            s1Size = offsetCalculator.GetDimS1();
        }

        uint32_t gIdxStart = gmCoord.gS1Idx / s1Size;
        uint32_t s1IdxStart = gmCoord.gS1Idx % s1Size;
        uint32_t gIdxEnd = (gmCoord.gS1Idx + gmCoord.gS1DealSize) / s1Size;
        uint32_t s1IdxEnd = (gmCoord.gS1Idx + gmCoord.gS1DealSize) % s1Size;

        uint64_t queryGmbaseOffset =
            offsetCalculator.GetOffset(gmCoord.bIdx, gmCoord.n2Idx, gIdxStart, 0, gmCoord.dIdx);

        // 处理第一个S
        uint32_t headSize = 0;
        if (gIdxStart == gIdxEnd) {
            headSize = s1IdxEnd - s1IdxStart;
        } else {
            headSize = s1Size - s1IdxStart;
        }

        uint64_t offset = queryGmbaseOffset + s1IdxStart * offsetCalculator.GetDimD();
        CopySingleMatrixNDToNZ(dstTensor.tensor, srcTensor.gmTensor[offset], headSize, gmCoord.dDealSize,
                               offsetCalculator.GetStrideS1(), dstTensor.rowCount);

        if (gIdxEnd - gIdxStart >= 1) {
            // 处理中间块
            uint64_t gmOffset = queryGmbaseOffset + offsetCalculator.GetStrideG();
            uint64_t l1Offset = headSize * 16U;

            if (gIdxEnd - gIdxStart > 1) {
                CopyMultiMatrixNDToNZ(dstTensor.tensor[l1Offset], srcTensor.gmTensor[gmOffset],
                        gIdxEnd - gIdxStart - 1, offsetCalculator.GetStrideG(), s1Size * 16U,
                        s1Size, gmCoord.dDealSize, offsetCalculator.GetStrideS1(), dstTensor.rowCount);
                gmOffset += (gIdxEnd - gIdxStart - 1) * offsetCalculator.GetStrideG();
                l1Offset += (gIdxEnd - gIdxStart - 1) * s1Size * 16U;
            }

            // 处理尾块
            if (s1IdxEnd > 0) {
                CopySingleMatrixNDToNZ(dstTensor.tensor[l1Offset], srcTensor.gmTensor[gmOffset], s1IdxEnd,
                                       gmCoord.dDealSize, offsetCalculator.GetStrideS1(), dstTensor.rowCount);
            }
        }
    }
};

#endif