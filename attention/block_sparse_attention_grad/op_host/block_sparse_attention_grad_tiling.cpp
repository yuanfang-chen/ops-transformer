/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "block_sparse_attention_grad_tiling.h"
#include <cmath>
#include <cstring>
#include "log/log.h"

#include <cstdint>
#include <string>
#include "err/ops_err.h"
#include "graph/types.h"
#include "graph/tensor.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling_base/tiling_base.h"

using namespace ge;
using namespace std;

constexpr int TND_DIM_T = 0;
constexpr int TND_DIM_N = 1;
constexpr int TND_DIM_D = 2;
constexpr int TND_DIM_NUM = 3;

constexpr int BNSD_DIM_B = 0;
constexpr int BNSD_DIM_N = 1;
constexpr int BNSD_DIM_S = 2;
constexpr int BNSD_DIM_D = 3;
constexpr int BNSD_DIM_NUM = 4;

constexpr int BSH_DIM_B = 0;
constexpr int BSH_DIM_S = 1;
constexpr int BSH_DIM_H = 2;

constexpr int DOUT_INDEX = 0;
constexpr int QUERY_INDEX = 1;
constexpr int KEY_INDEX = 2;
constexpr int VALUE_INDEX = 3;
constexpr int OUT_INDEX = 4;
constexpr int SOFTMAX_LSE_INDEX = 5;
constexpr int BLOCK_SPARSE_MASK_INDEX = 6;
constexpr int BLOCK_SHAPE_INDEX = 8;

constexpr int ATTENTION_MASK_INDEX = 7;
constexpr int ACTUAL_SEQ_LENGTHS_INDEX = 9;
constexpr int ACTUAL_SEQ_LENGTHS_KV_INDEX = 10;

constexpr int Q_INPUT_LAYOUT_INDEX = 0;
constexpr int KV_INPUT_LAYOUT_INDEX = 1;
constexpr int NUM_KEY_VALUE_HEADS_INDEX = 2;
constexpr int MASK_TYPE_INDEX = 3;
constexpr int SCALE_VALUE_INDEX = 4;

constexpr int VALID_HEAD_DIM_128 = 128;

namespace optiling {

constexpr uint32_t BASIC_BLOCK_SIZE = 128;
constexpr uint32_t WORKSPACE_BLOCK_SIZE_DB = 131072;
constexpr uint32_t NUM3 = 3;

static inline uint32_t CeilDiv(uint32_t n1, uint32_t n2)
{
    if (n1 == 0) {
        return 0;
    }
    return (n2 != 0) ? ((n1 + n2 - 1) / n2) : n1;
}

static inline uint32_t GetQBlocks(int32_t qseqlen, int32_t x)
{
    uint32_t qBlocksInX = CeilDiv(x, BASIC_BLOCK_SIZE);
    uint32_t completeXBlocks = x != 0 ? qseqlen / x : qseqlen / BASIC_BLOCK_SIZE;
    uint32_t remainingSeqlen = x != 0 ? qseqlen - completeXBlocks * x : qseqlen % BASIC_BLOCK_SIZE;
    uint32_t remainingBlocks = CeilDiv(remainingSeqlen, BASIC_BLOCK_SIZE);
    return qBlocksInX * completeXBlocks + remainingBlocks;
}

ge::graphStatus BSAGradTiling::GetNpuInfo(gert::TilingContext *context)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize_);
    libapiSize_ = ascendcPlatform.GetLibApiWorkSpaceSize();
    aivNum_ = ascendcPlatform.GetCoreNumAiv();
    aicNum_ = ascendcPlatform.GetCoreNumAic();
    
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BSAGradTiling::ProcessInput(gert::TilingContext *context)
{
    ge::graphStatus ret;
    
    if (context->GetAttrs()->GetAttrPointer<char>(Q_INPUT_LAYOUT_INDEX) == nullptr) {
        return ge::GRAPH_FAILED;
    }
    
    std::string qLayout(context->GetAttrs()->GetAttrPointer<char>(Q_INPUT_LAYOUT_INDEX));
    if (qLayout == "TND") {
        layout_ = InputLayout::TND;
    } else if (qLayout == "BNSD") {
        layout_ = InputLayout::BNSD;
    } else {
        OP_LOGE(context->GetNodeName(), "Unsupported layout: %s. Supported formats: TND, BNSD", 
                context->GetAttrs()->GetAttrPointer<char>(Q_INPUT_LAYOUT_INDEX));
        return ge::GRAPH_FAILED;
    }

    const auto *queryShape = context->GetInputShape(QUERY_INDEX);
    if (queryShape == nullptr) {
        OP_LOGE(context->GetNodeName(), "Query shape is null");
        return ge::GRAPH_FAILED;
    }
    const auto *kvShape = context->GetInputShape(KEY_INDEX);
    if (kvShape == nullptr) {
        OP_LOGE(context->GetNodeName(), "KV shape is null");
        return ge::GRAPH_FAILED;
    }
    auto blockSparseMaskShape = context->GetInputShape(BLOCK_SPARSE_MASK_INDEX);
    if (blockSparseMaskShape == nullptr) {
        OP_LOGE(context->GetNodeName(), "Block sparse mask is null");
        return ge::GRAPH_FAILED;
    }

    if (layout_ == InputLayout::TND) {
        if (queryShape->GetStorageShape().GetDimNum() != TND_DIM_NUM ||
            kvShape->GetStorageShape().GetDimNum() != TND_DIM_NUM ||
            blockSparseMaskShape->GetStorageShape().GetDimNum() != TND_DIM_NUM) {
            OP_LOGE(context->GetNodeName(), "TND format must have 3 dimensions");
            return ge::GRAPH_FAILED;
        }
        totalTokensT_ = queryShape->GetStorageShape().GetDim(TND_DIM_T);
        numHeads_ = static_cast<uint32_t>(queryShape->GetStorageShape().GetDim(TND_DIM_N));
        headDim_ = static_cast<uint32_t>(queryShape->GetStorageShape().GetDim(TND_DIM_D));
        kvHeads_ = static_cast<uint32_t>(queryShape->GetStorageShape().GetDim(TND_DIM_N));
        auto actualSeqLengths = context->GetOptionalInputTensor(ACTUAL_SEQ_LENGTHS_INDEX);
        if (actualSeqLengths == nullptr) {
            OP_LOGE(context->GetNodeName(), "TND format must have is actualSeqLengthsOptional");
            return ge::GRAPH_FAILED;
        } else {
            batch_ = static_cast<uint32_t>(actualSeqLengths->GetShapeSize());
            qSeqLenList = actualSeqLengths->GetData<int64_t>();
            if (qSeqLenList == nullptr) {
                OP_LOGE(context->GetNodeName(), "Actual seq lengths GetData is nullptr");
                return ge::GRAPH_FAILED;
            }
        }
    } else if (layout_ == InputLayout::BNSD) {
        if (queryShape->GetStorageShape().GetDimNum() != BNSD_DIM_NUM ||
            kvShape->GetStorageShape().GetDimNum() != BNSD_DIM_NUM ||
            blockSparseMaskShape->GetStorageShape().GetDimNum() != BNSD_DIM_NUM) {
            OP_LOGE(context->GetNodeName(), "BNSD format must have 4 dimensions");
            return ge::GRAPH_FAILED;
        }
        // 保存BNSD格式的batch和S维度
        batch_ = static_cast<uint32_t>(queryShape->GetStorageShape().GetDim(BNSD_DIM_B));
        numHeads_ = static_cast<uint32_t>(queryShape->GetStorageShape().GetDim(BNSD_DIM_N));
        maxQSeqlen_ = static_cast<uint32_t>(queryShape->GetStorageShape().GetDim(BNSD_DIM_S));
        headDim_ = static_cast<uint32_t>(queryShape->GetStorageShape().GetDim(BNSD_DIM_D));
        maxKvSeqlen_ = static_cast<uint32_t>(queryShape->GetStorageShape().GetDim(BNSD_DIM_S));
    }

    auto blockShapeOptional = context->GetInputTensor(BLOCK_SHAPE_INDEX);
    if (blockShapeOptional == nullptr) {
        OP_LOGE(context->GetNodeName(), "Block shape tensor is null");
        return ge::GRAPH_FAILED;
    }
    blockShapeList = blockShapeOptional->GetData<int64_t>();
    if (blockShapeList == nullptr) {
        OP_LOGE(context->GetNodeName(), "Block shape GetData is nullptr");
        return ge::GRAPH_FAILED;
    }
    blockShapeX_ = blockShapeList[0];
    blockShapeY_ = blockShapeList[1];

    if (context->GetAttrs()->GetAttrPointer<float>(SCALE_VALUE_INDEX) == nullptr) {
        scaleValue_ = 1.0f / std::sqrt(static_cast<float>(headDim_));
    } else {
        scaleValue_ = *context->GetAttrs()->GetAttrPointer<float>(SCALE_VALUE_INDEX);
    }

    auto qInputDesc = context->GetInputDesc(QUERY_INDEX);
    if (blockShapeList == nullptr) {
        OP_LOGE(context->GetNodeName(), "Query inputDesc is null");
        return ge::GRAPH_FAILED;
    } else {
        dataType_ = qInputDesc->GetDataType();
    }


    // maxKvBlockNum_ = static_cast<uint32_t>(selectIdxShape->GetStorageShape().GetDim(MAX_BLOCK_NUM_INDEX));
    // maxNumBlocksPerBatch_ = maxKvBlockNum_ * numHeads_;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BSAGradTiling::CalculateTaskSplit(gert::TilingContext *context)
{
    // 计算总的Q块数量和最大KV块数量
    totalQBlocks_ = 0;
    if (batch_ == 0) {
        OP_LOGE(context->GetNodeName(), "batch_ is 0 in CalculateTaskSplit");
        return ge::GRAPH_FAILED;
    }
    // 遍历每个batch进行分核计算
    for (auto i = 0; i < batch_; i++) {
        int64_t qSeqlen;
        if (layout_ == InputLayout::TND) {
            qSeqlen = static_cast<int64_t>(maxQSeqlen_);
        } else {
            qSeqlen = qSeqLenList[i];
        }
        uint32_t curTaskNum = GetQBlocks(qSeqlen, blockShapeX_) * numHeads_;
        uint32_t curQBlockNum = CeilDiv(qSeqlen, blockShapeX_) * numHeads_;
        if (i == 0) {
            firstBatchTaskNum_ = curTaskNum;
            firstQBlockNum_ = curQBlockNum;
        }
        totalTaskNum_ += curTaskNum;
        totalQBlocks_ += curQBlockNum;
    }
    blockDim_ = std::min(aicNum_, totalTaskNum_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BSAGradTiling::CalculateWorkSpace(gert::TilingContext *context)
{
    if (blockDim_ == 0) {
        OP_LOGE(context->GetNodeName(), "blockDim is 0");
        return ge::GRAPH_FAILED;
    }
    
    mm1OutSize_ = blockDim_ * WORKSPACE_BLOCK_SIZE_DB * sizeof(float) * NUM3;
    smOnlineOutSize_ = blockDim_ * WORKSPACE_BLOCK_SIZE_DB * sizeof(uint16_t) * NUM3;
    mm2OutSize_ = blockDim_ * WORKSPACE_BLOCK_SIZE_DB * sizeof(float) * NUM3;
    updateSize_ = blockDim_ * WORKSPACE_BLOCK_SIZE_DB * sizeof(float) * NUM3;
    
    workSpaceSize_ = libapiSize_ + mm1OutSize_ + smOnlineOutSize_ + mm2OutSize_ + updateSize_;
    context->GetWorkspaceSizes(1)[0] = workSpaceSize_;
    
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BSAGradTiling::FillTilingData(gert::TilingContext *context)
{
    if (tilingData_ == nullptr) {
        return ge::GRAPH_FAILED;
    }

    tilingData_->set_batch(batch_);
    tilingData_->set_numHeads(numHeads_);
    tilingData_->set_kvHeads(kvHeads_);
    tilingData_->set_headDim(headDim_);
    tilingData_->set_maxNumBlocksPerBatch(maxNumBlocksPerBatch_);
    tilingData_->set_firstBatchTaskNum(firstBatchTaskNum_);
    tilingData_->set_totalTaskNum(totalTaskNum_);
    tilingData_->set_maskType(maskType_);

    tilingData_->set_blockShapeX(blockShapeX_);
    tilingData_->set_blockShapeY(blockShapeY_);

    tilingData_->set_firstQBlockNum(firstQBlockNum_);
    tilingData_->set_totalQBlocks(totalQBlocks_);
    tilingData_->set_maxKvBlockNum(maxKvBlockNum_);

    tilingData_->set_inputLayout(static_cast<uint32_t>(layout_));
    tilingData_->set_maxQSeqlen(maxQSeqlen_);
    tilingData_->set_maxKvSeqlen(maxKvSeqlen_);

    // 生成tilingKey（按照开发规范：在tiling层生成）
    uint64_t tilingKey = GenerateTilingKey();
    tilingData_->set_tilingKey(tilingKey);
    context->SetTilingKey(tilingKey);
    context->SetBlockDim(blockDim_);

    tilingData_->set_mm1OutSize(mm1OutSize_);
    tilingData_->set_smOnlineOutSize(smOnlineOutSize_);
    tilingData_->set_mm2OutSize(mm2OutSize_);
    tilingData_->set_updateSize(updateSize_);
    tilingData_->set_workSpaceSize(workSpaceSize_);
    tilingData_->set_scaleValue(scaleValue_);

    return ge::GRAPH_SUCCESS;
}

uint64_t BSAGradTiling::GenerateTilingKey()
{
    return 0;
}

ge::graphStatus BSAGradTiling::GetBSAGradTiling(gert::TilingContext *context,
                                                BlockSparseAttentionGradTilingData &tilingData)
{
    tilingData_ = &tilingData;
    ge::graphStatus ret = GetNpuInfo(context);
    if (ret != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "GetNpuInfo failed");
        return ret;
    }
    
    ret = ProcessInput(context);
    if (ret != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "ProcessInput failed");
        return ret;
    }
    
    ret = CalculateTaskSplit(context);
    if (ret != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "CalculateTaskSplit failed");
        return ret;
    }
    
    ret = CalculateWorkSpace(context);
    if (ret != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "CalculateWorkSpace failed");
        return ret;
    }
    
    ret = FillTilingData(context);
    if (ret != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "FillTilingData failed");
        return ret;
    }
    
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BSAGradTiling::SetTilingData(gert::TilingContext *context,
                                             BlockSparseAttentionGradTilingData &tilingData)
{
    OP_CHECK_IF(context->GetRawTilingData() == nullptr,
        OPS_REPORT_VECTOR_INNER_ERR("BlockSparseAttentionGrad",
        "RawTilingData got from GE context is nullptr."), return ge::GRAPH_FAILED);
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

ASCENDC_EXTERN_C ge::graphStatus TilingBlockSparseAttentionGrad(gert::TilingContext* context)
{
    OP_CHECK_IF(context == nullptr, OPS_REPORT_VECTOR_INNER_ERR("RainFusionAttention",
        "Context is nullptr."), return ge::GRAPH_FAILED);

    BlockSparseAttentionGradTilingData tilingData;
    BSAGradTiling tiling;
    if (tiling.GetBSAGradTiling(context, tilingData) == ge::GRAPH_SUCCESS) {
        tiling.SetTilingData(context, tilingData);
        return ge::GRAPH_SUCCESS;
    } else {
        OP_LOGE(context->GetNodeName(), "GetBSAGradTiling failed");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ASCENDC_EXTERN_C ge::graphStatus TilingPrepareBlockSparseAttentionGrad(gert::TilingParseContext* context)
{
    (void) context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(BlockSparseAttentionGrad)
    .Tiling(TilingBlockSparseAttentionGrad)
    .TilingParse<BlockSparseAttentionGradCompileInfo>(TilingPrepareBlockSparseAttentionGrad); // 向框架注册入口函数;

}  // namespace optiling
