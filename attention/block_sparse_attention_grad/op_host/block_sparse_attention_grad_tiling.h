/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BLOCK_SPARSE_ATTENTION_GRAD_TILING_H
#define BLOCK_SPARSE_ATTENTION_GRAD_TILING_H

#include <cstdint>
#include "register/tilingdata_base.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"
#include "register/op_def_registry.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(BlockSparseAttentionGradTilingData)
// 基础参数
TILING_DATA_FIELD_DEF(uint32_t, batch);
TILING_DATA_FIELD_DEF(uint32_t, numHeads);
TILING_DATA_FIELD_DEF(uint32_t, kvHeads);
TILING_DATA_FIELD_DEF(uint32_t, headDim);
TILING_DATA_FIELD_DEF(uint32_t, blockSize);
TILING_DATA_FIELD_DEF(uint32_t, maxNumBlocksPerBatch);
TILING_DATA_FIELD_DEF(uint32_t, firstBatchTaskNum);
TILING_DATA_FIELD_DEF(uint32_t, totalTaskNum);
TILING_DATA_FIELD_DEF(uint32_t, maskType);
TILING_DATA_FIELD_DEF(float, scaleValue);
TILING_DATA_FIELD_DEF(uint32_t, totalQBlocks);       // T: 所有batch中Q方向切块的总数
TILING_DATA_FIELD_DEF(uint32_t, firstQBlockNum);       // T: 所有batch中Q方向切块的总数

// 稀疏分块参数 (blockShapeOptional)
TILING_DATA_FIELD_DEF(uint64_t, blockShapeX);  // block的x维度(Q方向)
TILING_DATA_FIELD_DEF(uint64_t, blockShapeY);  // block的y维度(KV方向)

// selectIdx相关参数
TILING_DATA_FIELD_DEF(uint32_t, maxKvBlockNum);      // 最大KV块数量（selectIdx的最后一维）

// Layout: 0=TND, 1=BNSD
TILING_DATA_FIELD_DEF(uint32_t, inputLayout);

// BNSD格式的最大序列长度（用于计算stride）
// 当actualSeqLengths为nullptr时，maxQSeqlen也用作统一的qseqlen值
TILING_DATA_FIELD_DEF(uint32_t, maxQSeqlen);  // BNSD格式Q的第三维（S维度），或统一的qseqlen值
// 当actualSeqLengthsKv为nullptr时，maxKvSeqlen也用作统一的kvseqlen值
TILING_DATA_FIELD_DEF(uint32_t, maxKvSeqlen);  // BNSD格式KV的第三维（S维度），或统一的kvseqlen值

// TilingKey for kernel dispatch (生成在tiling层)
TILING_DATA_FIELD_DEF(uint64_t, tilingKey);

// Workspace大小
TILING_DATA_FIELD_DEF(uint64_t, mm1OutSize);
TILING_DATA_FIELD_DEF(uint64_t, smOnlineOutSize);
TILING_DATA_FIELD_DEF(uint64_t, mm2OutSize);
TILING_DATA_FIELD_DEF(uint64_t, updateSize);
TILING_DATA_FIELD_DEF(uint64_t, workSpaceSize);

END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(BlockSparseAttentionGrad, BlockSparseAttentionGradTilingData)

// BlockSparseAttentionGrad编译信息
struct BlockSparseAttentionGradCompileInfo {
    uint32_t inputDataByte = 2;
    ge::DataType inputDataType;
    
    uint32_t coreNum = 0;
    uint32_t aivNum = 0;
    uint32_t aicNum = 0;
    uint64_t ubSize = 0;
    uint64_t l1Size = 0;
    uint64_t sysWorkspaceSize = 0;
    platform_ascendc::SocVersion socVersion;
};

// Input Layout枚举
enum InputLayout : uint32_t {
    TND = 0,  // [T, N, D] format
    BNSD = 1  // [B, N, S, D] format
};

// Tiling类
class BSAGradTiling {
public:
    BSAGradTiling() = default;
    ~BSAGradTiling() = default;
    
    ge::graphStatus GetBSAGradTiling(gert::TilingContext *context,
                                     BlockSparseAttentionGradTilingData &tilingData);
    ge::graphStatus SetTilingData(gert::TilingContext *context,
                                  BlockSparseAttentionGradTilingData &tilingData);
private:
    ge::graphStatus GetNpuInfo(gert::TilingContext *context);
    ge::graphStatus ProcessInput(gert::TilingContext *context);
    ge::graphStatus CalculateTaskSplit(gert::TilingContext *context);
    ge::graphStatus CalculateWorkSpace(gert::TilingContext *context);
    ge::graphStatus FillTilingData(gert::TilingContext *context);
    
    uint64_t GenerateTilingKey();

private:
    uint32_t batch_ = 0;
    // uint32_t qSeqlen_ = 0;
    // uint32_t kvSeqlen_ = 0;
    uint32_t numHeads_ = 0;
    uint32_t kvHeads_ = 0;
    uint32_t headDim_ = 0;
    int64_t blockShapeX_ = 0;  // block的x维度
    int64_t blockShapeY_ = 0;  // block的y维度
    float scaleValue_ = 0.0f;
    uint32_t maskType_ = 0;
    
    uint32_t totalQBlocks_ = 0;
    uint32_t maxKvBlockNum_ = 0;
    uint32_t firstQBlockNum_ = 0;
    uint32_t firstBatchTaskNum_ = 0;
    uint32_t totalTaskNum_ = 0;
    uint32_t maxNumBlocksPerBatch_ = 0;
    const int64_t *qSeqLenList = nullptr;
    const int64_t *kvSeqLenList = nullptr;
    const int64_t *blockShapeList = nullptr;
    bool useUniformQSeqlen_ = false;  // 是否使用统一的qseqlen值（使用maxQSeqlen_）
    bool useUniformKvSeqlen_ = false;  // 是否使用统一的kvseqlen值（使用maxKvSeqlen_）

    uint64_t mm1OutSize_ = 0;
    uint64_t smOnlineOutSize_ = 0;
    uint64_t mm2OutSize_ = 0;
    uint64_t updateSize_ = 0;
    
    InputLayout layout_ = InputLayout::TND;
    
    uint32_t blockDim_ = 20;
    uint32_t aivNum_ = 0;
    uint32_t aicNum_ = 0;
    uint64_t ubSize_ = 0;
    uint64_t workSpaceSize_ = 0;
    uint64_t libapiSize_ = 0;
    
    uint32_t maxQSeqlen_ = 0;  // BNSD格式Q的第三维（S维度）
    uint32_t maxKvSeqlen_ = 0;  // BNSD格式KV的第三维（S维度）
    int64_t totalTokensT_ = 0;  // TND格式Q的第一维（T维度，总token数）
    
    ge::DataType dataType_ = ge::DT_FLOAT16;

    BlockSparseAttentionGradTilingData *tilingData_ = nullptr;
};

}  // namespace optiling

#endif  // BLOCK_SPARSE_ATTENTION_GRAD_TILING_H

