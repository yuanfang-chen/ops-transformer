#include "kernel_operator.h"
#include "fa2_tiling.h"

using namespace AscendC;

constexpr int32_t ALIGN_FP32 = 8;

class FlashAttention2Kernel {
public:
    __aicore__ inline FlashAttention2Kernel() {}
    __aicore__ inline void Init(
        GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR output,
        GM_ADDR tilingGM);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ProcessOneHead(int32_t batchIdx, int32_t headIdx);
    __aicore__ inline void ComputeQKDot(
        int64_t qHeadOff, int32_t mStart, int32_t mSize,
        int64_t kHeadOff, int32_t nStart, int32_t nSize);
    __aicore__ inline void ApplyCausalMask(
        int32_t mStart, int32_t mSize, int32_t nStart, int32_t nSize);
    __aicore__ inline void OnlineSoftmax(int32_t mSize, int32_t nSize);
    __aicore__ inline void AccumulatePV(
        int64_t vHeadOff, int32_t mSize, int32_t nStart, int32_t nSize,
        bool isFirst);
    __aicore__ inline void NormalizeAndWrite(
        int64_t outHeadOff, int32_t mStart, int32_t mSize);

    __aicore__ inline int32_t AlignUp(int32_t n, int32_t a) {
        return ((n + a - 1) / a) * a;
    }

    GlobalTensor<half> gmQ, gmK, gmV, gmOut;
    FA2TilingData tiling;

    TBuf<TPosition::VECCALC> qkBuf;
    TBuf<TPosition::VECCALC> accBuf;
    TBuf<TPosition::VECCALC> rowMaxBuf;
    TBuf<TPosition::VECCALC> rowSumBuf;
    TBuf<TPosition::VECCALC> corrBuf;
    TBuf<TPosition::VECCALC> tmpExpBuf;

    TPipe pipe;
};

__aicore__ inline void FlashAttention2Kernel::Init(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR output,
    GM_ADDR tilingGM)
{
    auto* dst = reinterpret_cast<uint8_t*>(&tiling);
    auto* src = reinterpret_cast<__gm__ uint8_t*>(tilingGM);
    for (uint32_t i = 0; i < sizeof(FA2TilingData); i++) {
        dst[i] = src[i];
    }

    int64_t totalElem = (int64_t)tiling.batchSize * tiling.numHeads *
                        tiling.seqLenQ * tiling.headDim;
    gmQ.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(q), totalElem);
    gmK.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(k), totalElem);
    gmV.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(v), totalElem);
    gmOut.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(output), totalElem);

    int32_t bm = tiling.blockM;
    int32_t bnAligned = AlignUp(tiling.blockN, ALIGN_FP32);
    int32_t da = tiling.headDimAligned;
    int32_t rowAligned = AlignUp(bm, ALIGN_FP32);

    pipe.InitBuffer(qkBuf,  bm * bnAligned * (int32_t)sizeof(float));
    pipe.InitBuffer(accBuf, bm * da * (int32_t)sizeof(float));
    pipe.InitBuffer(rowMaxBuf, rowAligned * (int32_t)sizeof(float));
    pipe.InitBuffer(rowSumBuf, rowAligned * (int32_t)sizeof(float));
    pipe.InitBuffer(corrBuf,   rowAligned * (int32_t)sizeof(float));
    pipe.InitBuffer(tmpExpBuf, ALIGN_FP32 * (int32_t)sizeof(float));
}

__aicore__ inline void FlashAttention2Kernel::Process()
{
    int32_t blockIdx = GetBlockIdx();
    int32_t totalHeads = tiling.batchSize * tiling.numHeads;
    if (blockIdx >= totalHeads) return;

    int32_t headIdx  = blockIdx % tiling.numHeads;
    int32_t batchIdx = blockIdx / tiling.numHeads;
    ProcessOneHead(batchIdx, headIdx);
}

__aicore__ inline void FlashAttention2Kernel::ProcessOneHead(
    int32_t batchIdx, int32_t headIdx)
{
    int32_t seqQ  = tiling.seqLenQ;
    int32_t seqKV = tiling.seqLenKV;
    int32_t bm = tiling.blockM;
    int32_t bn = tiling.blockN;
    int32_t d  = tiling.headDim;
    int32_t da = tiling.headDimAligned;

    int64_t headOff = ((int64_t)batchIdx * tiling.numHeads + headIdx) *
                      seqQ * d;

    for (int32_t mb = 0; mb < tiling.numBlocksM; mb++) {
        int32_t mStart = mb * bm;
        int32_t mSize  = bm;
        if (mStart + mSize > seqQ) mSize = seqQ - mStart;

        LocalTensor<float> acc = accBuf.Get<float>();
        LocalTensor<float> rowMax = rowMaxBuf.Get<float>();
        LocalTensor<float> rowSum = rowSumBuf.Get<float>();

        int32_t accCount = AlignUp(mSize * da, ALIGN_FP32);
        int32_t rowCount = AlignUp(mSize, ALIGN_FP32);
        Duplicate(acc, 0.0f, accCount);
        Duplicate(rowMax, -1e30f, rowCount);
        Duplicate(rowSum, 0.0f, rowCount);

        int32_t nLimit = seqKV;
        if (tiling.isCausal) {
            nLimit = mStart + mSize;
            if (nLimit > seqKV) nLimit = seqKV;
        }

        bool isFirst = true;
        for (int32_t nb = 0; nb < tiling.numBlocksN; nb++) {
            int32_t nStart = nb * bn;
            if (nStart >= nLimit) break;
            int32_t nSize = bn;
            if (nStart + nSize > nLimit) nSize = nLimit - nStart;

            ComputeQKDot(headOff, mStart, mSize, headOff, nStart, nSize);

            if (tiling.isCausal) {
                ApplyCausalMask(mStart, mSize, nStart, nSize);
            }

            OnlineSoftmax(mSize, nSize);
            AccumulatePV(headOff, mSize, nStart, nSize, isFirst);
            isFirst = false;
        }

        NormalizeAndWrite(headOff, mStart, mSize);
    }
}

__aicore__ inline void FlashAttention2Kernel::ComputeQKDot(
    int64_t qHeadOff, int32_t mStart, int32_t mSize,
    int64_t kHeadOff, int32_t nStart, int32_t nSize)
{
    int32_t d = tiling.headDim;
    float scale = tiling.softmaxScale;
    int32_t bnAligned = AlignUp(nSize, ALIGN_FP32);
    LocalTensor<float> qk = qkBuf.Get<float>();

    Duplicate(qk, -1e30f, mSize * bnAligned);

    for (int32_t mi = 0; mi < mSize; mi++) {
        int64_t qOff = qHeadOff + (int64_t)(mStart + mi) * d;
        for (int32_t ni = 0; ni < nSize; ni++) {
            int64_t kOff = kHeadOff + (int64_t)(nStart + ni) * d;
            float dot = 0.0f;
            for (int32_t di = 0; di < d; di++) {
                dot += (float)gmQ.GetValue(qOff + di) *
                       (float)gmK.GetValue(kOff + di);
            }
            qk.SetValue(mi * bnAligned + ni, dot * scale);
        }
    }
}

__aicore__ inline void FlashAttention2Kernel::ApplyCausalMask(
    int32_t mStart, int32_t mSize, int32_t nStart, int32_t nSize)
{
    int32_t bnAligned = AlignUp(nSize, ALIGN_FP32);
    LocalTensor<float> qk = qkBuf.Get<float>();
    for (int32_t mi = 0; mi < mSize; mi++) {
        int32_t gRow = mStart + mi;
        for (int32_t ni = 0; ni < nSize; ni++) {
            if (nStart + ni > gRow) {
                qk.SetValue(mi * bnAligned + ni, -1e30f);
            }
        }
    }
}

__aicore__ inline void FlashAttention2Kernel::OnlineSoftmax(
    int32_t mSize, int32_t nSize)
{
    int32_t bnAligned = AlignUp(nSize, ALIGN_FP32);
    LocalTensor<float> qk   = qkBuf.Get<float>();
    LocalTensor<float> rMax = rowMaxBuf.Get<float>();
    LocalTensor<float> rSum = rowSumBuf.Get<float>();
    LocalTensor<float> corr = corrBuf.Get<float>();
    LocalTensor<float> tmpE = tmpExpBuf.Get<float>();

    for (int32_t mi = 0; mi < mSize; mi++) {
        float oldMax = rMax.GetValue(mi);

        float newMax = oldMax;
        for (int32_t ni = 0; ni < bnAligned; ni++) {
            float v = qk.GetValue(mi * bnAligned + ni);
            if (v > newMax) newMax = v;
        }
        rMax.SetValue(mi, newMax);

        LocalTensor<float> row = qk[mi * bnAligned];
        Adds(row, row, -newMax, bnAligned);
        pipe_barrier(PIPE_V);

        Exp(row, row, bnAligned);
        pipe_barrier(PIPE_V);

        for (int32_t ni = nSize; ni < bnAligned; ni++) {
            row.SetValue(ni, 0.0f);
        }

        float eSum = 0.0f;
        for (int32_t ni = 0; ni < bnAligned; ni++) {
            eSum += row.GetValue(ni);
        }

        float correction;
        float oldMaxShift = oldMax - newMax;
        if (oldMaxShift < -80.0f) {
            correction = 0.0f;
        } else {
            Duplicate(tmpE, 0.0f, ALIGN_FP32);
            tmpE.SetValue(0, oldMaxShift);
            Exp(tmpE, tmpE, ALIGN_FP32);
            pipe_barrier(PIPE_V);
            correction = tmpE.GetValue(0);
        }

        corr.SetValue(mi, correction);
        rSum.SetValue(mi, rSum.GetValue(mi) * correction + eSum);
    }
}

__aicore__ inline void FlashAttention2Kernel::AccumulatePV(
    int64_t vHeadOff, int32_t mSize, int32_t nStart, int32_t nSize,
    bool isFirst)
{
    int32_t d  = tiling.headDim;
    int32_t da = tiling.headDimAligned;
    int32_t bnAligned = AlignUp(nSize, ALIGN_FP32);
    LocalTensor<float> acc  = accBuf.Get<float>();
    LocalTensor<float> qk   = qkBuf.Get<float>();
    LocalTensor<float> corr = corrBuf.Get<float>();

    if (!isFirst) {
        for (int32_t mi = 0; mi < mSize; mi++) {
            float c = corr.GetValue(mi);
            LocalTensor<float> accRow = acc[mi * da];
            Muls(accRow, accRow, c, da);
            pipe_barrier(PIPE_V);
        }
    }

    for (int32_t mi = 0; mi < mSize; mi++) {
        for (int32_t di = 0; di < d; di++) {
            float s = 0.0f;
            for (int32_t ni = 0; ni < nSize; ni++) {
                float w = qk.GetValue(mi * bnAligned + ni);
                float vv = (float)gmV.GetValue(
                    vHeadOff + (int64_t)(nStart + ni) * d + di);
                s += w * vv;
            }
            int32_t idx = mi * da + di;
            acc.SetValue(idx, acc.GetValue(idx) + s);
        }
    }
}

__aicore__ inline void FlashAttention2Kernel::NormalizeAndWrite(
    int64_t outHeadOff, int32_t mStart, int32_t mSize)
{
    int32_t d  = tiling.headDim;
    int32_t da = tiling.headDimAligned;
    LocalTensor<float> acc  = accBuf.Get<float>();
    LocalTensor<float> rSum = rowSumBuf.Get<float>();

    for (int32_t mi = 0; mi < mSize; mi++) {
        float inv = 1.0f / rSum.GetValue(mi);
        LocalTensor<float> accRow = acc[mi * da];
        Muls(accRow, accRow, inv, da);
        pipe_barrier(PIPE_V);

        for (int32_t di = 0; di < d; di++) {
            float val = accRow.GetValue(di);
            gmOut.SetValue(outHeadOff + (int64_t)(mStart + mi) * d + di,
                           (half)val);
        }
    }
}

extern "C" __global__ __aicore__
void flash_attention2_do(GM_ADDR q, GM_ADDR k, GM_ADDR v,
                         GM_ADDR output, GM_ADDR workspace,
                         GM_ADDR tilingGM)
{
    FlashAttention2Kernel op;
    op.Init(q, k, v, output, tilingGM);
    op.Process();
}
