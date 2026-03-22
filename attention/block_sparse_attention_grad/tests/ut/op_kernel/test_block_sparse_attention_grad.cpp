/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 */

#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "data_utils.h"

#include "block_sparse_attention_grad_tiling.h" 

using namespace std;

// 顺序：dout, query, key, value, out, softmaxLse, blockSparseMask, attentionMask, 
//       blockShape, actualSeqLengths, actualSeqLengthsKv, dq, dk, dv, workspace, tiling
extern "C" __global__ __aicore__ void block_sparse_attention_grad(
    GM_ADDR dout, GM_ADDR query, GM_ADDR key, GM_ADDR value,
    GM_ADDR out, GM_ADDR softmaxLse, GM_ADDR blockSparseMask,
    GM_ADDR attentionMask, GM_ADDR blockShape, 
    GM_ADDR actualSeqLengths, GM_ADDR actualSeqLengthsKv,
    GM_ADDR dq, GM_ADDR dk, GM_ADDR dv,
    GM_ADDR workspace, GM_ADDR tiling);

class block_sparse_attention_grad_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "--- block_sparse_attention_grad_test SetUp ---\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "--- block_sparse_attention_grad_test TearDown ---\n" << endl;
    }
};

TEST_F(block_sparse_attention_grad_test, test_case_0)
{
    // 启用混合模式以支持模拟计算
    AscendC::SetKernelMode(KernelMode::MIX_MODE);

    // 1. 定义 BNSD 形状 (128x128 序列，64x64 切块)
    size_t b = 1;
    size_t n = 4;
    size_t n_kv = 2; // GQA 测试
    size_t s = 128;
    size_t s_kv = 128;
    size_t d = 128; // 标准 128 维
    size_t blockX = 64;
    size_t blockY = 64;
    
    size_t ceilQ = (s + blockX - 1) / blockX;
    size_t ceilKv = (s_kv + blockY - 1) / blockY;

    // 2. 计算各 Tensor 所需内存大小
    size_t qkvSize = b * n * s * d * sizeof(half);      // Q, dout, out, dq
    size_t kvSize = b * n_kv * s_kv * d * sizeof(half); // K, V, dk, dv
    size_t lseSize = b * n * s * sizeof(float);         // SoftmaxLse (FP32)
    size_t maskSize = b * n * ceilQ * ceilKv * sizeof(uint8_t); // BlockSparseMask
    
    // 假定申请 20MB 的 Workspace 供测试，防止越界
    size_t workspaceSize = 20 * 1024 * 1024; 
    size_t tilingSize = sizeof(BlockSparseAttentionGradTilingData);

    // 3. 申请 CPU 模拟全局内存 (GM_ADDR)
    uint8_t *doutGm = (uint8_t *)AscendC::GmAlloc(qkvSize);
    uint8_t *queryGm = (uint8_t *)AscendC::GmAlloc(qkvSize);
    uint8_t *keyGm = (uint8_t *)AscendC::GmAlloc(kvSize);
    uint8_t *valueGm = (uint8_t *)AscendC::GmAlloc(kvSize);
    uint8_t *outGm = (uint8_t *)AscendC::GmAlloc(qkvSize);
    uint8_t *softmaxLseGm = (uint8_t *)AscendC::GmAlloc(lseSize);
    uint8_t *blockSparseMaskGm = (uint8_t *)AscendC::GmAlloc(maskSize);
    
    uint8_t *attenMaskGm = (uint8_t *)AscendC::GmAlloc(64);
    uint8_t *blockShapeGm = (uint8_t *)AscendC::GmAlloc(64);
    uint8_t *actualSeqLenGm = (uint8_t *)AscendC::GmAlloc(64);
    uint8_t *actualSeqLenKvGm = (uint8_t *)AscendC::GmAlloc(64);

    uint8_t *dqGm = (uint8_t *)AscendC::GmAlloc(qkvSize);
    uint8_t *dkGm = (uint8_t *)AscendC::GmAlloc(kvSize);
    uint8_t *dvGm = (uint8_t *)AscendC::GmAlloc(kvSize);

    // 模拟器下的 workspace 也需要足够大以满足该机制
    uint8_t *workspaceGm = (uint8_t *)AscendC::GmAlloc(workspaceSize);
    uint8_t *tilingGm = (uint8_t *)AscendC::GmAlloc(tilingSize);

    memset(tilingGm, 0, tilingSize);

    // 4. 填充核心 Tiling 数据以支撑内核初始化逻辑
    // 这些数据是防止内核指针访问越界的最底线保障
    BlockSparseAttentionGradTilingData *tilingData = reinterpret_cast<BlockSparseAttentionGradTilingData *>(tilingGm);
    tilingData->set_batch(b);
    tilingData->set_numHeads(n);
    tilingData->set_kvHeads(n_kv);
    tilingData->set_headDim(d);
    tilingData->set_totalTaskNum(4); 
    tilingData->set_blockShapeX(blockX);
    tilingData->set_blockShapeY(blockY);
    tilingData->set_inputLayout(1); // 1 = BNSD
    tilingData->set_maxQSeqlen(s);
    tilingData->set_maxKvSeqlen(s_kv);
    tilingData->set_taskNumPerCore(1); // 让每个核分配 1 个任务
    tilingData->set_tailTaskNum(0);
    tilingData->set_scaleValue(0.088388f); // 1/sqrt(128)
    tilingData->set_basicQBlockSize(128); 
    tilingData->set_basicKVBlockSize(128);
    
    // 设置 Workspace 各个块的大小划分 
    tilingData->set_sOutSize(1024 * 1024);
    tilingData->set_dPOutSize(1024 * 1024);
    tilingData->set_dQOutSize(qkvSize);
    tilingData->set_dKOutSize(kvSize);
    tilingData->set_dVOutSize(kvSize);
    tilingData->set_gradSize(1024 * 1024);

    // 设置分配给各核的起始偏移
    for(int i = 0; i < 4; i++) {
            tilingData->get_beginBatch()[i] = 0;
            tilingData->get_beginHead()[i] = i; 
            tilingData->get_beginQSeqOffset()[i] = 0;
            tilingData->get_preQSeqLengths()[i] = i * s; 
            tilingData->get_preKVSeqLengths()[i] = (i / (n / n_kv)) * s_kv; 
        }

    // 5. 启动内核模拟执行
    // 模拟启动 4 个 AI Core 实例
    uint32_t blockDim = 4;
    ICPU_SET_TILING_KEY(0); 
    
    // 调用 Kernel 函数（严格对齐源码里的 16 个参数）
    ICPU_RUN_KF(block_sparse_attention_grad, blockDim, 
                doutGm, queryGm, keyGm, valueGm, outGm, 
                softmaxLseGm, blockSparseMaskGm, attenMaskGm, 
                blockShapeGm, actualSeqLenGm, actualSeqLenKvGm, 
                dqGm, dkGm, dvGm, workspaceGm, tilingGm);

    // 6. 清理内存
    AscendC::GmFree(doutGm);
    AscendC::GmFree(queryGm);
    AscendC::GmFree(keyGm);
    AscendC::GmFree(valueGm);
    AscendC::GmFree(outGm);
    AscendC::GmFree(softmaxLseGm);
    AscendC::GmFree(blockSparseMaskGm);
    AscendC::GmFree(attenMaskGm);
    AscendC::GmFree(blockShapeGm);
    AscendC::GmFree(actualSeqLenGm);
    AscendC::GmFree(actualSeqLenKvGm);
    AscendC::GmFree(dqGm);
    AscendC::GmFree(dkGm);
    AscendC::GmFree(dvGm);
    AscendC::GmFree(workspaceGm);
    AscendC::GmFree(tilingGm);
}