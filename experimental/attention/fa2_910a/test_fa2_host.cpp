#include <acl/acl.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <chrono>
#include "fa2_tiling.h"

#define CHECK_ACL(call)                                              \
    do {                                                             \
        aclError _e = (call);                                        \
        if (_e != ACL_SUCCESS) {                                     \
            fprintf(stderr, "ACL error %d at %s:%d\n",               \
                    (int)_e, __FILE__, __LINE__);                    \
            exit(1);                                                 \
        }                                                            \
    } while (0)

extern "C" aclError aclrtlaunch_flash_attention2_do(
    uint32_t blockDim, aclrtStream stream,
    void* q, void* k, void* v, void* output,
    void* workspace, void* tilingGM);

static void fillRandom(std::vector<uint16_t>& buf, int seed) {
    srand(seed);
    for (auto& x : buf) {
        float val = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
        uint32_t bits;
        memcpy(&bits, &val, 4);
        uint16_t sign = (bits >> 31) & 1;
        int32_t  exp  = ((bits >> 23) & 0xFF) - 127 + 15;
        uint32_t frac = (bits >> 13) & 0x3FF;
        if (exp <= 0) { exp = 0; frac = 0; }
        if (exp >= 31) { exp = 31; frac = 0; }
        x = (sign << 15) | (exp << 10) | frac;
    }
}

static float fp16ToFloat(uint16_t h) {
    uint32_t sign = (h >> 15) & 1;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t frac = h & 0x3FF;
    if (exp == 0) return sign ? -0.0f : 0.0f;
    if (exp == 31) return sign ? -INFINITY : INFINITY;
    float f = ldexpf((float)(1024 + frac), (int)exp - 25);
    return sign ? -f : f;
}

static uint16_t floatToFp16(float val) {
    uint32_t bits;
    memcpy(&bits, &val, 4);
    uint16_t sign = (bits >> 31) & 1;
    int32_t  exp  = ((bits >> 23) & 0xFF) - 127 + 15;
    uint32_t frac = (bits >> 13) & 0x3FF;
    if (exp <= 0) { exp = 0; frac = 0; }
    if (exp >= 31) { exp = 31; frac = 0; }
    return (sign << 15) | (exp << 10) | frac;
}

static void refAttention(
    const std::vector<uint16_t>& q,
    const std::vector<uint16_t>& k,
    const std::vector<uint16_t>& v,
    std::vector<uint16_t>& out,
    int B, int H, int S, int D, bool causal)
{
    float scale = 1.0f / sqrtf((float)D);
    for (int b = 0; b < B; b++) {
      for (int h = 0; h < H; h++) {
        int64_t headOff = ((int64_t)b * H + h) * S * D;
        for (int i = 0; i < S; i++) {
            std::vector<float> scores(S);
            float maxS = -1e30f;
            int jLimit = causal ? (i + 1) : S;
            for (int j = 0; j < S; j++) {
                if (j >= jLimit) { scores[j] = -1e30f; continue; }
                float dot = 0.0f;
                for (int d = 0; d < D; d++)
                    dot += fp16ToFloat(q[headOff + (int64_t)i*D + d]) *
                           fp16ToFloat(k[headOff + (int64_t)j*D + d]);
                scores[j] = dot * scale;
                if (scores[j] > maxS) maxS = scores[j];
            }
            float sumE = 0.0f;
            for (int j = 0; j < S; j++) {
                scores[j] = expf(scores[j] - maxS);
                sumE += scores[j];
            }
            for (int j = 0; j < S; j++) scores[j] /= sumE;
            for (int d = 0; d < D; d++) {
                float acc = 0.0f;
                for (int j = 0; j < S; j++)
                    acc += scores[j] *
                           fp16ToFloat(v[headOff + (int64_t)j*D + d]);
                out[headOff + (int64_t)i*D + d] = floatToFp16(acc);
            }
        }
      }
    }
}

int main(int argc, char** argv) {
    int deviceId = 4;
    int B = 1, H = 2, S = 64, D = 64;
    bool causal = true;

    if (argc > 1) deviceId = atoi(argv[1]);
    if (argc > 2) S = atoi(argv[2]);
    if (argc > 3) causal = (atoi(argv[3]) != 0);

    printf("=== AscendC FA2 Test ===\n");
    printf("Device=%d  B=%d H=%d S=%d D=%d causal=%d\n",
           deviceId, B, H, S, D, causal);

    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(deviceId));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    int64_t totalElem = (int64_t)B * H * S * D;
    size_t dataBytes  = totalElem * sizeof(uint16_t);

    std::vector<uint16_t> hQ(totalElem), hK(totalElem), hV(totalElem);
    fillRandom(hQ, 42);
    fillRandom(hK, 123);
    fillRandom(hV, 456);

    void *dQ, *dK, *dV, *dOut, *dWs, *dTiling;
    CHECK_ACL(aclrtMalloc(&dQ, dataBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&dK, dataBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&dV, dataBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&dOut, dataBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&dWs, 1024, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&dTiling, sizeof(FA2TilingData),
                          ACL_MEM_MALLOC_HUGE_FIRST));

    CHECK_ACL(aclrtMemcpy(dQ, dataBytes, hQ.data(), dataBytes,
                          ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(dK, dataBytes, hK.data(), dataBytes,
                          ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(dV, dataBytes, hV.data(), dataBytes,
                          ACL_MEMCPY_HOST_TO_DEVICE));

    int32_t blockM = 32, blockN = 32;
    int32_t dAligned = ((D + 7) / 8) * 8;
    FA2TilingData tiling;
    tiling.batchSize = B;
    tiling.numHeads  = H;
    tiling.seqLenQ   = S;
    tiling.seqLenKV  = S;
    tiling.headDim   = D;
    tiling.headDimAligned = dAligned;
    tiling.blockM    = blockM;
    tiling.blockN    = blockN;
    tiling.softmaxScale = 1.0f / sqrtf((float)D);
    tiling.isCausal  = causal ? 1 : 0;
    tiling.numBlocksM = (S + blockM - 1) / blockM;
    tiling.numBlocksN = (S + blockN - 1) / blockN;
    tiling.totalCores = B * H;

    CHECK_ACL(aclrtMemcpy(dTiling, sizeof(FA2TilingData), &tiling,
                          sizeof(FA2TilingData), ACL_MEMCPY_HOST_TO_DEVICE));

    printf("Launching kernel with blockDim=%d ...\n", tiling.totalCores);
    aclError launchRet = aclrtlaunch_flash_attention2_do(
        tiling.totalCores, stream,
        dQ, dK, dV, dOut, dWs, dTiling);
    if (launchRet != ACL_SUCCESS) {
        printf("ERROR: kernel launch failed: %d\n", (int)launchRet);
    }
    CHECK_ACL(aclrtSynchronizeStream(stream));
    printf("Kernel finished.\n");

    std::vector<uint16_t> hOut(totalElem);
    CHECK_ACL(aclrtMemcpy(hOut.data(), dataBytes, dOut, dataBytes,
                          ACL_MEMCPY_DEVICE_TO_HOST));

    printf("Computing reference ...\n");
    std::vector<uint16_t> hRef(totalElem);
    refAttention(hQ, hK, hV, hRef, B, H, S, D, causal);

    float maxDiff = 0.0f, sumDiff = 0.0f;
    int   diffCount = 0;
    for (int64_t i = 0; i < totalElem; i++) {
        float a = fp16ToFloat(hOut[i]);
        float r = fp16ToFloat(hRef[i]);
        float d = fabsf(a - r);
        if (d > maxDiff) maxDiff = d;
        sumDiff += d;
        if (d > 0.05f) diffCount++;
    }
    float avgDiff = sumDiff / totalElem;

    printf("\n=== Results ===\n");
    printf("Max diff : %.6f\n", maxDiff);
    printf("Avg diff : %.6f\n", avgDiff);
    printf("Outliers (>0.05): %d / %ld\n", diffCount, (long)totalElem);

    if (maxDiff < 0.02f) {
        printf("PASSED\n");
    } else if (maxDiff < 0.1f) {
        printf("ACCEPTABLE (fp16 precision)\n");
    } else {
        printf("FAILED\n");
    }

    printf("\nSample output [0..7]: ");
    for (int i = 0; i < 8 && i < (int)totalElem; i++)
        printf("%.4f ", fp16ToFloat(hOut[i]));
    printf("\nSample ref   [0..7]: ");
    for (int i = 0; i < 8 && i < (int)totalElem; i++)
        printf("%.4f ", fp16ToFloat(hRef[i]));
    printf("\n");

    CHECK_ACL(aclrtFree(dQ));
    CHECK_ACL(aclrtFree(dK));
    CHECK_ACL(aclrtFree(dV));
    CHECK_ACL(aclrtFree(dOut));
    CHECK_ACL(aclrtFree(dWs));
    CHECK_ACL(aclrtFree(dTiling));
    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtResetDevice(deviceId));
    CHECK_ACL(aclFinalize());

    return (maxDiff < 0.1f) ? 0 : 1;
}
