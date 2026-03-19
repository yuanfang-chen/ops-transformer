#ifndef FA2_TILING_H
#define FA2_TILING_H

#include <cstdint>

#pragma pack(push, 8)
struct FA2TilingData {
    int32_t batchSize;
    int32_t numHeads;
    int32_t seqLenQ;
    int32_t seqLenKV;
    int32_t headDim;
    int32_t headDimAligned;
    int32_t blockM;
    int32_t blockN;
    float softmaxScale;
    int32_t isCausal;
    int32_t numBlocksM;
    int32_t numBlocksN;
    int32_t totalCores;
};
#pragma pack(pop)

#endif
