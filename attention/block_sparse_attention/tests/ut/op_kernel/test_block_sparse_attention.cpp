/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "block_sparse_attention_tiling.h"
#include "data_utils.h"
#include "../../../op_kernel/block_sparse_attention.cpp"

using namespace std;

#define PARAM_LIST_DEF GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR blockSparseMask,\
                       GM_ADDR attenMask, GM_ADDR blockShape, GM_ADDR actualSeqLengths,\
                       GM_ADDR actualSeqLengthsKv, GM_ADDR blockTable, GM_ADDR attentionOut,\
                       GM_ADDR softmaxLse, GM_ADDR workspace, GM_ADDR tiling

#define PARAM_LIST query, key, value, blockSparseMask, attenMask, blockShape, actualSeqLengths,\
                   actualSeqLengthsKv, blockTable, attentionOut, softmaxLse, workspace, tiling

class block_sparse_attention_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "block_sparse_attention_test SetUp\n" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "block_sparse_attention_test TearDown\n" << std::endl;
    }
};

TEST_F(block_sparse_attention_test, test_case_tnd_fp16)
{
    size_t batch = 2;
    size_t numHeads = 32;
    size_t kvHeads = 8;
    size_t embeddingSize = 128;
    size_t qSeqlen = 128;
    size_t kvSeqlen = 128;
    size_t blockSize = 128;
    size_t totalTokens = qSeqlen * batch;
    size_t totalTokensKv = kvSeqlen * batch;
    
    AscendC::SetKernelMode(KernelMode::AIC_MODE);
    
    size_t querySize = totalTokens * numHeads * embeddingSize * sizeof(half);
    size_t keySize = totalTokensKv * kvHeads * embeddingSize * sizeof(half);
    size_t valueSize = totalTokensKv * kvHeads * embeddingSize * sizeof(half);
    size_t blockSizeMask = batch * numHeads * (qSeqlen / blockSize) * (kvSeqlen / blockSize) * sizeof(uint8_t);
    size_t attentionOutSize = totalTokens * numHeads * embeddingSize * sizeof(half);
    size_t softmaxLseSize = totalTokens * numHeads * sizeof(float);
    size_t tilingSize = sizeof(BlockSparseAttentionTilingData);
    
    uint8_t* query = (uint8_t*)AscendC::GmAlloc(querySize);
    uint8_t* key = (uint8_t*)AscendC::GmAlloc(keySize);
    uint8_t* value = (uint8_t*)AscendC::GmAlloc(valueSize);
    uint8_t* blockSparseMask = (uint8_t*)AscendC::GmAlloc(blockSizeMask);
    uint8_t* attentionOut = (uint8_t*)AscendC::GmAlloc(attentionOutSize);
    uint8_t* softmaxLse = (uint8_t*)AscendC::GmAlloc(softmaxLseSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);
    
    uint32_t NumBlocks = 20;
    
    BlockSparseAttentionTilingData* tilingData = reinterpret_cast<BlockSparseAttentionTilingData*>(tiling);
    tilingData->batch = batch;
    tilingData->numHeads = numHeads;
    tilingData->kvHeads = kvHeads;
    tilingData->embeddingSize = embeddingSize;
    tilingData->blockSize = blockSize;
    tilingData->maxNumBlocksPerBatch = 0;
    tilingData->firstBatchTaskNum = numHeads * (qSeqlen / blockSize);
    tilingData->totalTaskNum = batch * numHeads * (qSeqlen / blockSize);
    tilingData->maskType = 0;
    tilingData->scaleValue = 1.0f / std::sqrt(embeddingSize);
    tilingData->totalQBlocks = batch * numHeads * (qSeqlen / blockSize);
    tilingData->firstQBlockNum = numHeads * (qSeqlen / blockSize);
    tilingData->blockShapeX = blockSize;
    tilingData->blockShapeY = blockSize;
    tilingData->maxKvBlockNum = kvSeqlen / blockSize;
    tilingData->maxQBlockNum = qSeqlen / blockSize;
    tilingData->queryLayout = 0;
    tilingData->kvCacheLayout = 0;
    tilingData->maxQSeqlen = qSeqlen;
    tilingData->maxKvSeqlen = kvSeqlen;
    tilingData->useUniformQSeqlen = 1;
    tilingData->useUniformKvSeqlen = 1;
    tilingData->tilingKey = 9000000030100002ULL;
    tilingData->selectNumIdxSize = 0;
    tilingData->selectIdxSize = 0;
    tilingData->mm1OutSize = 0;
    tilingData->smOnlineOutSize = 0;
    tilingData->mm2OutSize = 0;
    tilingData->updateSize = 0;
    tilingData->workSpaceSize = 1024 * 1024 * 1024;
    
    ICPU_RUN_KF(block_sparse_attention, NumBlocks, query, key, value, blockSparseMask, nullptr, nullptr,
                nullptr, nullptr, nullptr, attentionOut, softmaxLse, workspace, tiling);
    
    AscendC::GmFree(query);
    AscendC::GmFree(key);
    AscendC::GmFree(value);
    AscendC::GmFree(blockSparseMask);
    AscendC::GmFree(attentionOut);
    AscendC::GmFree(softmaxLse);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(block_sparse_attention_test, test_case_bnsd_fp16)
{
    size_t batch = 2;
    size_t numHeads = 32;
    size_t kvHeads = 8;
    size_t embeddingSize = 128;
    size_t qSeqlen = 128;
    size_t kvSeqlen = 128;
    size_t blockSize = 128;
    
    AscendC::SetKernelMode(KernelMode::AIC_MODE);
    
    size_t querySize = batch * numHeads * qSeqlen * embeddingSize * sizeof(half);
    size_t keySize = batch * kvHeads * kvSeqlen * embeddingSize * sizeof(half);
    size_t valueSize = batch * kvHeads * kvSeqlen * embeddingSize * sizeof(half);
    size_t blockSizeMask = batch * numHeads * (qSeqlen / blockSize) * (kvSeqlen / blockSize) * sizeof(uint8_t);
    size_t attentionOutSize = batch * numHeads * qSeqlen * embeddingSize * sizeof(half);
    size_t softmaxLseSize = batch * numHeads * qSeqlen * sizeof(float);
    size_t tilingSize = sizeof(BlockSparseAttentionTilingData);
    
    uint8_t* query = (uint8_t*)AscendC::GmAlloc(querySize);
    uint8_t* key = (uint8_t*)AscendC::GmAlloc(keySize);
    uint8_t* value = (uint8_t*)AscendC::GmAlloc(valueSize);
    uint8_t* blockSparseMask = (uint8_t*)AscendC::GmAlloc(blockSizeMask);
    uint8_t* attentionOut = (uint8_t*)AscendC::GmAlloc(attentionOutSize);
    uint8_t* softmaxLse = (uint8_t*)AscendC::GmAlloc(softmaxLseSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);
    
    uint32_t NumBlocks = 20;
    
    BlockSparseAttentionTilingData* tilingData = reinterpret_cast<BlockSparseAttentionTilingData*>(tiling);
    tilingData->batch = batch;
    tilingData->numHeads = numHeads;
    tilingData->kvHeads = kvHeads;
    tilingData->embeddingSize = embeddingSize;
    tilingData->blockSize = blockSize;
    tilingData->maxNumBlocksPerBatch = 0;
    tilingData->firstBatchTaskNum = numHeads * (qSeqlen / blockSize);
    tilingData->totalTaskNum = batch * numHeads * (qSeqlen / blockSize);
    tilingData->maskType = 0;
    tilingData->scaleValue = 1.0f / std::sqrt(embeddingSize);
    tilingData->totalQBlocks = batch * numHeads * (qSeqlen / blockSize);
    tilingData->firstQBlockNum = numHeads * (qSeqlen / blockSize);
    tilingData->blockShapeX = blockSize;
    tilingData->blockShapeY = blockSize;
    tilingData->maxKvBlockNum = kvSeqlen / blockSize;
    tilingData->maxQBlockNum = qSeqlen / blockSize;
    tilingData->queryLayout = 1;
    tilingData->kvCacheLayout = 1;
    tilingData->maxQSeqlen = qSeqlen;
    tilingData->maxKvSeqlen = kvSeqlen;
    tilingData->useUniformQSeqlen = 1;
    tilingData->useUniformKvSeqlen = 1;
    tilingData->tilingKey = 9000000050100003ULL;
    tilingData->selectNumIdxSize = 0;
    tilingData->selectIdxSize = 0;
    tilingData->mm1OutSize = 0;
    tilingData->smOnlineOutSize = 0;
    tilingData->mm2OutSize = 0;
    tilingData->updateSize = 0;
    tilingData->workSpaceSize = 1024 * 1024 * 1024;
    
    ICPU_RUN_KF(block_sparse_attention, NumBlocks, query, key, value, blockSparseMask, nullptr, nullptr,
                nullptr, nullptr, nullptr, attentionOut, softmaxLse, workspace, tiling);
    
    AscendC::GmFree(query);
    AscendC::GmFree(key);
    AscendC::GmFree(value);
    AscendC::GmFree(blockSparseMask);
    AscendC::GmFree(attentionOut);
    AscendC::GmFree(softmaxLse);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
