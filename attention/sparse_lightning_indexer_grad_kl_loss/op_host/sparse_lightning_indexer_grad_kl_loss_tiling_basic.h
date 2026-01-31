/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file sparse_lightning_indexer_grad_kl_loss_tiling_basic.h
 * \brief
 */

#ifndef SPARSE_LIGHTNING_INDEXER_GRAD_KL_LOSS_TILING_BASIC_H
#define SPARSE_LIGHTNING_INDEXER_GRAD_KL_LOSS_TILING_BASIC_H

#include <numeric>
#include <algorithm>
#include <tiling/tiling_api.h>
#include "tiling_base/tiling_base.h"
#include "tiling_base/tiling_type.h"
#include "sparse_lightning_indexer_grad_kl_loss_tiling_common.h"
#include "err/ops_err.h"

using namespace ge;
using namespace AscendC;

namespace optiling {

static const int64_t PING_PONG_VALUE = 2L;
static const int64_t GM_ALIGN = 512;
static const int32_t FRACTAL_NUM = 16L;
static constexpr size_t WORK_SPACE_RESERVE_SIZE = 16 * 1024 * 1024;

static constexpr uint32_t BUFFER_SIZE_BYTE_1K = 1024;
static constexpr uint32_t BUFFER_SIZE_BYTE_2K = 2 * 1024;
static constexpr uint32_t BUFFER_SIZE_BYTE_8K = 8 * 1024;
static constexpr uint32_t BUFFER_SIZE_BYTE_32K = 32 * 1024;
static constexpr uint32_t BUFFER_SIZE_BYTE_33K = 33 * 1024;

static constexpr uint32_t NQUERY_SIZE_8   = 8;
static constexpr uint32_t NQUERY_SIZE_16  = 16;
static constexpr uint32_t NQUERY_SIZE_32  = 32;
static constexpr uint32_t NQUERY_SIZE_64  = 64;
static constexpr uint32_t NQUERY_SIZE_128 = 128;

static constexpr uint32_t NQUERYINDEX_SIZE_8  = 8;
static constexpr uint32_t NQUERYINDEX_SIZE_16 = 16;
static constexpr uint32_t NQUERYINDEX_SIZE_32 = 32;
static constexpr uint32_t NQUERYINDEX_SIZE_64 = 64;

static constexpr uint32_t N2_SIZE_1 = 1;
static constexpr uint32_t D_SIZE_512 = 512;
static constexpr uint32_t DINDEX_SIZE_128 = 128;
static constexpr uint32_t DROPE_SIZE_64 = 64;
static constexpr uint32_t TOPK_SIZE_2048 = 2048;
static constexpr int64_t  SPARSE_MODE_SIZE_3 = 3;

template <typename T>
static auto AlignUp(T num1, T num2) -> T
{
    if (num2 == 0) {
        return 0;
    }
    if (num1 < 0) {
        return -(-num1 / num2) * num2;
    }
    return (num1 + num2 - 1) / num2 * num2;
}

template <typename T>
static auto AlignDown(T num1, T num2) -> T
{
    if (num2 == 0) {
        return 0;
    }
    return num1 / num2 * num2;
}

template <typename T>
static auto CeilDivision(T num1, T num2) -> T
{
    if (num2 == 0) {
        return 0;
    }
    return (num1 + num2 - 1) / num2;
}

template <typename T>
static auto CeilDiv(const T n1, const T n2) -> T
{
    if (n1 == 0) {
        return 0;
    }
    return (n2 != 0) ? (((n1 - 1) / n2) + 1) : n1;
}

template <typename T>
static auto CalcTailSize(T num1, T num2) -> T
{
    if (num2 == 0) {
        return 0;
    }
    T mod = num1 % num2;
    return mod != 0 ? mod : num2;
}

class SparseLightningIndexerGradKLLossTilingGeneral : public Ops::Transformer::OpTiling::TilingBaseClass {
public:
    explicit SparseLightningIndexerGradKLLossTilingGeneral(gert::TilingContext *context) : TilingBaseClass(context)
    {
        Reset();
    }
    ~SparseLightningIndexerGradKLLossTilingGeneral() override = default;

    void Reset(gert::TilingContext *context) override
    {
        TilingBaseClass::Reset(context);
        Reset();
    }

protected:
    void Reset() {
        bSize = 0;
        gSizeQuery = 0;
        gSizeQueryIndex = 0;
        dSizeQuery = 0;
        dSizeQueryIndex = 0;
        kSize = 2048;
        n2Size = 0;
        s1Size = 0;
        s2Size = 0;
        accumS1 = 0;
        accumS2 = 0;
        sparseMode = 3;
        scaleValue = 1.0f;
        dQueryRopeSize = 0;
        dKeyRopeSize = 0;

        deterministic = false;
        opName = nullptr;
        inputLayout = nullptr;
    }

    [[nodiscard]] gert::TilingContext *GetContext()
    {
        return context_;
    }
    bool IsCapable()
    {
        return true;
    }
    // 1、获取平台信息比如CoreNum、UB/L1/L0C资源大小
    ge::graphStatus GetPlatformInfo() override;
    // 2、获取INPUT/OUTPUT/ATTR信息
    ge::graphStatus GetShapeAttrsInfo() override {};
    // 3、计算数据切分TilingData
    ge::graphStatus DoOpTiling() override {};
    // 4、计算高阶API的TilingData
    ge::graphStatus DoLibApiTiling() override 
    {
        return ge::GRAPH_SUCCESS;
    }
    ge::graphStatus PostTiling() override 
    {
        return ge::GRAPH_SUCCESS;
    }
    // 5、计算TilingKey
    uint64_t GetTilingKey() const override = 0;
    // 6、计算Workspace 大小
    ge::graphStatus GetWorkspaceSize() override {};
    // 7、保存Tiling数据
    void GetActualSeqLenData(int64_t inputIdx, std::vector<int64_t> &res, int64_t &actualLen) const;
    ge::graphStatus CheckContext();
    bool AnalyzeAttrs();
    bool CrossShapeVerify(const gert::Shape &queryRopeShape, const gert::Shape &keyRopeShap);
    bool AnalyzeDimLayout(const gert::Shape &queryShape, const gert::Shape &keyShape, const gert::Shape &queryIndexShape,
                         const gert::Shape &topKShape, size_t layoutLen, const gert::Shape &queryRopeShape, const gert::Shape &keyRopeShape);
    bool AnalyzeDtype();
    bool AnalyzeLayout();
    int64_t GetS2RealSize(int32_t sparseMode, int32_t s1Size, int32_t s2Size, int32_t s1Idx);
    bool InitSparseValidArray(std::vector<int64_t> &sparseValidArray, uint32_t sparseMode);
    bool InitLoadValue(const std::vector<int64_t> &sparseValidArray, int64_t validAicNum, int64_t totalSize,
                            const std::vector<int64_t> &sparseStartIdx, std::vector<int64_t> &localValue);
    bool BalanceLoad(const std::vector<int64_t> &sparseValidArray,std::vector<int64_t> &localValue, 
                    std::vector<int64_t> &sparseStartIdx, int64_t validAicNum, int64_t totalSize);
    bool Balance4DLoad(std::vector<int64_t> &tmpSparseValue, const std::vector<int64_t> sparseValidArray, 
                    const int64_t balanceNum);
    bool SetSparseStartIdx(const std::vector<int64_t> &sparseValidArray, int64_t validAicNum, int64_t totalSize, 
                            int64_t *sparseStartIdx, int64_t splitFactorSize, int64_t maxCoreNum);
    void SetSparseParamsRegbase();
    int64_t CalcTotalSize();
    void SetMultiCoreParamsRegbase(int64_t totalSize, int64_t coreNum);
    int64_t InitOutputSplit();

    // 基础输入参数
    int32_t bSize;
    int32_t n2Size;
    int32_t gSizeQuery;
    int32_t gSizeQueryIndex;
    int32_t s1Size;
    int32_t s2Size;
    int32_t dSizeQuery;
    int32_t dSizeQueryIndex;
    int32_t kSize;
    int32_t sparseMode;
    int32_t rsvd;
    float scaleValue;

    const char *templateName = "sligklloss_general";
    uint32_t aivNum;
    uint32_t aicNum;
    uint64_t l2CacheSize;

    int64_t realT1Size;
    int64_t dQueryRopeSize;
    int64_t dKeyRopeSize;
    int64_t accumS1;
    int64_t accumS2;
    int64_t maxS1Val;
    int64_t maxS2Val;
    LayoutType tilingKeyLayout;
    platform_ascendc::SocVersion socVersion;
    NpuArch npuArch = NpuArch::DAV_RESV;
    bool deterministic;
    bool hasRope = false;

    const char *opName;
    const char *inputLayout;

    std::vector<int64_t> actualSeqLenData;
    std::vector<int64_t> actualSeqLenKData;
};

} // optiling
#endif