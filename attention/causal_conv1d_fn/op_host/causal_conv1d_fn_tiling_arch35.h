/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file causal_conv1d_tiling_arch35.h
 * \brief
 */
#ifndef ASCEND_OPS_CAUSAL_CONV1D_TILING_H
#define ASCEND_OPS_CAUSAL_CONV1D_TILING_H
#include "log/log.h"
#include "platform/platform_info.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "tiling_base/tiling_templates_registry.h"
#include "util/math_util.h"
#include "util/platform_util.h"
#include "util/shape_util.h"

namespace optiling {

struct CausalConv1dFnCompileInfo {
    uint32_t aivNum;
    uint32_t aicNum;
    uint64_t ubSize;
    uint64_t l1Size;
    uint64_t l0cSize;
    uint64_t l2CacheSize;
    platform_ascendc::SocVersion socVersion;
    NpuArch npuArch;
};

BEGIN_TILING_DATA_DEF(CausalConv1dFnTilingData)
TILING_DATA_FIELD_DEF(uint32_t, loopNumBS);         // 每个核内BS方向的loop循环数
TILING_DATA_FIELD_DEF(uint32_t, loopNumDim);        // 每个核内Dim方向的loop循环数
TILING_DATA_FIELD_DEF(uint32_t, ubFactorBS);        // 每个核内BS方向单次循环载入的大小
TILING_DATA_FIELD_DEF(uint32_t, ubTailFactorBS);    // 每个核内BS方向尾次循环载入的大小
TILING_DATA_FIELD_DEF(uint32_t, ubFactorDim);       // 每个核内Dim方向单次循环载入的大小
TILING_DATA_FIELD_DEF(uint32_t, ubTailFactorDim);   // 每个核内Dim方向尾次循环载入的大小
TILING_DATA_FIELD_DEF(uint64_t, blockFactor);       // 切核的切分因子
TILING_DATA_FIELD_DEF(uint64_t, blockIndex);        // 切核的切分轴 (0: cu_seq_len, 1: dim)
TILING_DATA_FIELD_DEF(uint64_t, blockTailFactor);   // 切核的尾核切分因子
TILING_DATA_FIELD_DEF(uint32_t, tailBlockloopNumBS);      // 尾核内BS方向的loop循环数
TILING_DATA_FIELD_DEF(uint32_t, tailBlockloopNumDim);     // 尾核内Dim方向的loop循环数
TILING_DATA_FIELD_DEF(uint32_t, tailBlockubFactorBS);     // 尾核内BS方向单次循环载入的大小
TILING_DATA_FIELD_DEF(uint32_t, tailBlockubTailFactorBS); // 尾核内BS方向尾次循环载入的大小
TILING_DATA_FIELD_DEF(uint32_t, tailBlockubFactorDim);    // 尾核内Dim方向单次循环载入大小
TILING_DATA_FIELD_DEF(uint32_t, tailBlockubTailFactorDim);// 尾核内Dim方向尾次循环载入大小
TILING_DATA_FIELD_DEF(uint32_t, realCoreNum);       // 实际使用核数
TILING_DATA_FIELD_DEF(uint32_t, kernelWidth);       // 卷积核宽度 K
TILING_DATA_FIELD_DEF(uint64_t, cuSeqLen);          // cu_seq_len 大小
TILING_DATA_FIELD_DEF(uint64_t, dim);               // 特征维度大小
TILING_DATA_FIELD_DEF(uint32_t, batch);             // batch大小
TILING_DATA_FIELD_DEF(uint32_t, validBatchStart);   // 有效 batch 的起始索引（在原始 cacheIndices 中）
TILING_DATA_FIELD_DEF(uint32_t, validBatchCount);   // 有效 batch 的数量
TILING_DATA_FIELD_DEF(uint64_t, validSeqStart);     // 有效序列的起始位置（在原始 x 中的行偏移）
TILING_DATA_FIELD_DEF(uint64_t, validSeqLen);       // 有效序列的总长度

END_TILING_DATA_DEF

REGISTER_TILING_DATA_CLASS(CausalConv1dFn, CausalConv1dFnTilingData)


class CausalConv1dFnTiling : public Ops::Transformer::OpTiling::TilingBaseClass {
public:
    explicit CausalConv1dFnTiling(gert::TilingContext *context) : TilingBaseClass(context)
    {
    }

protected:
    bool IsCapable() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    void DumpTilingInfo() override;

    ge::graphStatus CheckInputParams();
    ge::graphStatus CheckInputDim();
    ge::graphStatus CheckInputDtype();
    ge::graphStatus CheckOutputParams();
    ge::graphStatus CalculateCuSeqLenTiling();
    ge::graphStatus CalculateDimTiling();

private:
    // 辅助结构体：切cu_seq_len时的核间切分信息
    struct CuSeqLenSplitInfo {
        uint64_t realCoreNum;
        uint64_t effectiveTotal;
        uint64_t baseLen;
        uint64_t remainder;
        uint64_t blockFactor;
        uint64_t blockTailFactor;
    };

    // 辅助函数：计算切cu_seq_len时实际需要的核数（考虑因果卷积重叠）
    // 同时返回中间计算结果，避免重复计算
    CuSeqLenSplitInfo CalculateCuSeqLenSplitInfo(uint64_t cuSeqLen, uint64_t bsOverlap) const;

    // 硬件信息
    uint64_t ubSize_ = 0;
    uint64_t totalCoreNum_ = 0;
    uint64_t workspaceSize_ = 0;
    uint64_t ubBlockSize_ = 0;

    // 输入参数信息
    uint64_t cuSeqLen_ = 0;
    uint64_t dim_ = 0;
    uint32_t kernelWidth_ = 0;  // K
    uint32_t batch_ = 0;
    uint64_t xDtypeSize_ = 0;

    // padding相关信息
    int64_t padSlotId_ = -1;
    uint32_t validBatchStart_ = 0;
    uint32_t validBatchCount_ = 0;
    uint64_t validSeqStart_ = 0;
    uint64_t validSeqLen_ = 0;

    // 缓存的核间切分信息（用于避免重复计算）
    bool hasCachedSplitInfo_ = false;
    CuSeqLenSplitInfo cachedSplitInfo_;

    // 分核信息
    uint64_t blockIndex_ = 0;        // 切分轴: 0-cu_seq_len, 1-dim
    uint64_t blockFactor_ = 0;       // 切分因子
    uint64_t blockTailFactor_ = 0;   // 尾核切分因子
    uint64_t realCoreNum_ = 0;       // 实际使用核数

    // 核内切分信息 - 整核
    uint32_t loopNumBS_ = 0;         // BS方向循环次数
    uint32_t loopNumDim_ = 0;        // Dim方向循环次数
    uint32_t ubFactorBS_ = 0;        // BS方向单次循环载入大小
    uint32_t ubTailFactorBS_ = 0;    // BS方向尾次循环载入大小
    uint32_t ubFactorDim_ = 0;       // Dim方向单次循环载入大小
    uint32_t ubTailFactorDim_ = 0;   // Dim方向尾次循环载入大小

    // 核内切分信息 - 尾核
    uint32_t tailBlockloopNumBS_ = 0;
    uint32_t tailBlockloopNumDim_ = 0;
    uint32_t tailBlockubFactorBS_ = 0;
    uint32_t tailBlockubTailFactorBS_ = 0;
    uint32_t tailBlockubFactorDim_ = 0;
    uint32_t tailBlockubTailFactorDim_ = 0;

    gert::Shape xShape_;
    gert::Shape weightShape_;
    gert::Shape cacheStatesShape_;
    gert::Shape seqStartIndexShape_;
    ge::DataType xType_;
    ge::DataType weightType_;
    CausalConv1dFnTilingData tilingData_;
};

} // namespace optiling
#endif // ASCEND_OPS_CAUSAL_CONV1D_TILING_H
