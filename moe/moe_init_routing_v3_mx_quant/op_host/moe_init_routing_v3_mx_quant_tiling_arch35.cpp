/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "moe_init_routing_v3_mx_quant_tiling_arch35.h"
#include <cstdint>
#include <cstring>
#include <string>
#include <graph/utils/type_utils.h>
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_def_registry.h"
#include "log/log.h"

using namespace ge;

namespace optiling {

// ==================== Constants ====================

constexpr int64_t BLOCK_BYTES = 32;
constexpr int64_t INT32_SIZE = 4;
constexpr int64_t INT32_ONE_BLOCK_NUM = BLOCK_BYTES / INT32_SIZE;  // 8
constexpr int64_t MX_BLOCK_SIZE = 32;
constexpr int64_t SORT_ONE_CORE = 0;
constexpr int64_t SORT_MULTI_CORE = 1;
constexpr int64_t IDX_GATHER = 0;
constexpr int64_t IDX_SCATTER = 1;

// Sort mode constants
constexpr int64_t ONE_REPEAT_SORT_NUM = 32;
constexpr int64_t MRGSORT_LIST_MAX_ELEMENT = 2040;
constexpr int64_t FP32_ONE_BLOCK_NUM = 8;
constexpr int64_t FP32_ONE_REPEAT_NUM = 64;
constexpr int64_t ASSIST_NUM = 256;

// ==================== Helper Functions ====================

static inline int64_t CeilDiv(int64_t a, int64_t b)
{
    if (b == 0) {
        return 0;
    }
    return (a + b - 1) / b;
}

static inline int64_t AlignUp(int64_t value, int64_t align)
{
    if (align == 0) {
        return value;
    }
    return (value + align - 1) / align * align;
}

static inline int64_t AlignDown(int64_t value, int64_t align)
{
    if (align == 0) {
        return value;
    }
    return value / align * align;
}

// ==================== Attribute Index (must match _def.cpp order) ====================
// 0: active_num (Int)
// 1: expert_capacity (Int)
// 2: expert_num (Int)
// 3: drop_pad_mode (Int)
// 4: expert_tokens_count_or_cumsum_flag (Int)
// 5: expert_tokens_before_capacity_flag (Bool)
// 6: axis (Int)
// 7: round_mode (String)
// 8: dst_type (Int)
// 9: blocksize (Int)
// 10: scale_alg (Int)

constexpr int ATTR_IDX_ACTIVE_NUM = 0;
constexpr int ATTR_IDX_EXPERT_CAPACITY = 1;
constexpr int ATTR_IDX_EXPERT_NUM = 2;
constexpr int ATTR_IDX_DROP_PAD_MODE = 3;
constexpr int ATTR_IDX_EXPERT_TOKENS_COUNT_FLAG = 4;
constexpr int ATTR_IDX_EXPERT_TOKENS_BEFORE_CAP = 5;
constexpr int ATTR_IDX_AXIS = 6;
constexpr int ATTR_IDX_ROUND_MODE = 7;
constexpr int ATTR_IDX_DST_TYPE = 8;
constexpr int ATTR_IDX_BLOCKSIZE = 9;
constexpr int ATTR_IDX_SCALE_ALG = 10;

// ==================== CompileInfo ====================

struct MoeInitRoutingV3MxQuantCompileInfo {
    uint32_t coreNum;
    uint64_t ubSize;
};

// ==================== VBS Tiling Calculation ====================

static void CalcVBSTiling(MoeV3Arch35VBSComputeTilingData &vbs,
                          int64_t totalElements, int64_t coreNum, int64_t ubSize)
{
    // VBS: Vector Bitonic Sort stage
    // Each sort operation handles ONE_REPEAT_SORT_NUM=32 elements at a time
    // UB needs: sort input (int32) + sort temp + rowIdx (int32) + assist data
    // Reference: MoeSortOneCore and MoeSortMultiCore
    int64_t sortElemBytes = INT32_SIZE;  // int32
    // Max elements per loop is limited by UB: need 2 copies of data (value + rowIdx) + temp
    // Approximate: ubSize / (3 * sortElemBytes) -> align down to sort boundary
    int64_t maxSortElements = ubSize / (3 * sortElemBytes + ASSIST_NUM * sortElemBytes / ONE_REPEAT_SORT_NUM);
    maxSortElements = AlignDown(maxSortElements, ONE_REPEAT_SORT_NUM);
    if (maxSortElements < ONE_REPEAT_SORT_NUM) {
        maxSortElements = ONE_REPEAT_SORT_NUM;
    }

    vbs.oneLoopMaxElements = maxSortElements;
    vbs.needCoreNum = coreNum;

    int64_t perCoreElements = CeilDiv(totalElements, coreNum);
    perCoreElements = AlignUp(perCoreElements, INT32_ONE_BLOCK_NUM);
    int64_t lastCoreElements = totalElements - perCoreElements * (coreNum - 1);
    if (lastCoreElements <= 0) {
        // Adjust: reduce cores or recalculate
        perCoreElements = CeilDiv(totalElements, coreNum);
        lastCoreElements = totalElements - perCoreElements * (coreNum - 1);
    }

    vbs.perCoreElements = perCoreElements;
    vbs.lastCoreElements = lastCoreElements;

    // Per-core loop calculation
    int64_t perLoopElements = (maxSortElements < perCoreElements) ? maxSortElements : perCoreElements;
    vbs.perCoreLoops = CeilDiv(perCoreElements, perLoopElements);
    vbs.perCorePerLoopElements = perLoopElements;
    vbs.perCoreLastLoopElements = perCoreElements - perLoopElements * (vbs.perCoreLoops - 1);
    if (vbs.perCoreLastLoopElements <= 0) {
        vbs.perCoreLastLoopElements = perLoopElements;
    }

    // Last core loop calculation
    int64_t lastPerLoopElements = (maxSortElements < lastCoreElements) ? maxSortElements : lastCoreElements;
    if (lastCoreElements <= 0) {
        vbs.lastCoreLoops = 0;
        vbs.lastCorePerLoopElements = 0;
        vbs.lastCoreLastLoopElements = 0;
    } else {
        vbs.lastCoreLoops = CeilDiv(lastCoreElements, lastPerLoopElements);
        vbs.lastCorePerLoopElements = lastPerLoopElements;
        vbs.lastCoreLastLoopElements = lastCoreElements - lastPerLoopElements * (vbs.lastCoreLoops - 1);
        if (vbs.lastCoreLastLoopElements <= 0) {
            vbs.lastCoreLastLoopElements = lastPerLoopElements;
        }
    }
}

// ==================== VMS Tiling Calculation ====================

static void CalcVMSTiling(MoeV3Arch35VMSMiddleComputeTilingData &vms, int64_t coreNum)
{
    // VMS: Vector Merge Sort multi-core middle stage
    vms.needCoreNum = coreNum;
}

// ==================== SortOut Tiling Calculation ====================

static void CalcSortOutTiling(MoeV3Arch35SortOutComputeTilingData &sortOut, int64_t ubSize)
{
    // SortOut: merge sorted sub-arrays to produce final sorted output
    // Max elements limited by UB for merge operation
    int64_t maxElements = ubSize / (INT32_SIZE * 4);  // need multiple copies for merge
    maxElements = AlignDown(maxElements, INT32_ONE_BLOCK_NUM);
    if (maxElements > MRGSORT_LIST_MAX_ELEMENT) {
        maxElements = MRGSORT_LIST_MAX_ELEMENT;
    }
    if (maxElements < INT32_ONE_BLOCK_NUM) {
        maxElements = INT32_ONE_BLOCK_NUM;
    }
    sortOut.oneLoopMaxElements = maxElements;
}

// ==================== ExpertTokensCount Tiling ====================

static void CalcExpertTokensCountTiling(MoeV3Arch35ExpertTokensCountTilingData &etc,
                                         int64_t totalElements, int64_t coreNum, int64_t ubSize)
{
    // ExpertTokensCount: histogram counting over sorted expert indices
    int64_t maxPerLoop = ubSize / (INT32_SIZE * 2);  // need input + output buffers
    maxPerLoop = AlignDown(maxPerLoop, INT32_ONE_BLOCK_NUM);
    if (maxPerLoop < INT32_ONE_BLOCK_NUM) {
        maxPerLoop = INT32_ONE_BLOCK_NUM;
    }

    etc.needCoreNum = coreNum;
    int64_t perCoreElements = CeilDiv(totalElements, coreNum);
    perCoreElements = AlignUp(perCoreElements, INT32_ONE_BLOCK_NUM);
    int64_t lastCoreElements = totalElements - perCoreElements * (coreNum - 1);
    if (lastCoreElements <= 0) {
        perCoreElements = CeilDiv(totalElements, coreNum);
        lastCoreElements = totalElements - perCoreElements * (coreNum - 1);
    }

    etc.perCoreElements = perCoreElements;
    etc.lastCoreElements = lastCoreElements;

    int64_t perLoopElements = (maxPerLoop < perCoreElements) ? maxPerLoop : perCoreElements;
    etc.perCoreLoops = CeilDiv(perCoreElements, perLoopElements);
    etc.perCorePerLoopElements = perLoopElements;
    etc.perCoreLastLoopElements = perCoreElements - perLoopElements * (etc.perCoreLoops - 1);
    if (etc.perCoreLastLoopElements <= 0) {
        etc.perCoreLastLoopElements = perLoopElements;
    }

    int64_t lastPerLoop = (maxPerLoop < lastCoreElements) ? maxPerLoop : lastCoreElements;
    if (lastCoreElements <= 0) {
        etc.lastCoreLoops = 0;
        etc.lastCorePerLoopElements = 0;
        etc.lastCoreLastLoopElements = 0;
    } else {
        etc.lastCoreLoops = CeilDiv(lastCoreElements, lastPerLoop);
        etc.lastCorePerLoopElements = lastPerLoop;
        etc.lastCoreLastLoopElements = lastCoreElements - lastPerLoop * (etc.lastCoreLoops - 1);
        if (etc.lastCoreLastLoopElements <= 0) {
            etc.lastCoreLastLoopElements = lastPerLoop;
        }
    }
}

// ==================== GatherOut Tiling Calculation ====================

static void CalcGatherOutTiling(MoeV3Arch35GatherOutComputeTilingData &gather,
                                int64_t totalExpandedTokens, int64_t cols, int64_t coreNum,
                                int64_t ubSize, int64_t blocksize, int64_t activeNum)
{
    gather.activeNum = activeNum;
    gather.needCoreNum = coreNum;

    // === Column dimension tiling ===
    // perLoopCols must be aligned to MX_BLOCK_SIZE (=blocksize, default 32) for complete MX blocks
    // UB budget for one column slice of one row:
    //   xIn:       perLoopCols * 2 (bf16)
    //   xQuantOut: perLoopCols * 1 (fp8) -- actually need ceil(perLoopCols/4)*4 for pack4
    //   mxScaleOut: ceil(perLoopCols / blocksize) * 1 (fp8_e8m0)
    //   maxExpBuf:  ceil(perLoopCols / blocksize) * 2 (uint16)
    //   invScaleBuf: ceil(perLoopCols / blocksize) * 2 (uint16)
    //   sortedRowIdxIn: at least 1 * 4 bytes (int32), but we batch perLoopRows indices
    // Reserve some UB for sortedRowIdx buffer and overhead
    constexpr int64_t UB_RESERVE = 4096;  // reserve for sortedRowIdx + pipe overhead
    int64_t availUb = static_cast<int64_t>(ubSize) - UB_RESERVE;
    if (availUb < BLOCK_BYTES) {
        availUb = BLOCK_BYTES;
    }

    // Per column element cost: 2 (bf16 xIn) + 1 (fp8 out) + 5/blocksize (scale+maxExp+invScale)
    // = 3 + 5/32 ~ 3.16 bytes per element
    // Compute max perLoopCols from UB budget
    int64_t bytesPerCol = 2 + 1;  // bf16 input + fp8 output
    // Scale overhead per blocksize columns: 1 (mxscale) + 2 (maxExp) + 2 (invScale) = 5 bytes
    // So total = perLoopCols * 3 + ceil(perLoopCols/blocksize) * 5
    // Approximate: perLoopCols * (3 + 5.0/blocksize)
    // Solve: perLoopCols = availUb / (3 + 5.0 / blocksize)
    int64_t perLoopCols = availUb * blocksize / (3 * blocksize + 5);
    // Align down to blocksize
    perLoopCols = AlignDown(perLoopCols, blocksize);
    if (perLoopCols < blocksize) {
        perLoopCols = blocksize;
    }
    // Cap to actual cols (do NOT AlignUp here — kernel uses DataCopyPad for tail padding)
    if (perLoopCols > cols) {
        perLoopCols = cols;
    }

    gather.perLoopCols = perLoopCols;
    gather.colsLoops = CeilDiv(cols, perLoopCols);
    int64_t lastLoopCols = cols - perLoopCols * (gather.colsLoops - 1);
    if (lastLoopCols <= 0) {
        lastLoopCols = perLoopCols;
    }
    // lastLoopCols might not fill a full perLoopCols, but must be padded to blocksize in kernel
    gather.lastLoopCols = lastLoopCols;

    // === Row dimension tiling (indices distribution across cores) ===
    int64_t perCoreIndices = CeilDiv(totalExpandedTokens, coreNum);
    int64_t lastCoreIndices = totalExpandedTokens - perCoreIndices * (coreNum - 1);
    if (lastCoreIndices <= 0) {
        perCoreIndices = CeilDiv(totalExpandedTokens, coreNum);
        lastCoreIndices = totalExpandedTokens - perCoreIndices * (coreNum - 1);
    }

    gather.perCoreIndicesElements = perCoreIndices;
    gather.lastCoreIndicesElements = lastCoreIndices;

    // Determine how many row indices we can process per loop
    // We need UB space for: perLoopRows * sizeof(int32) for sortedRowIdx + per-row column buffers
    // Since column buffers are reused per row, the constraint is mainly on sortedRowIdx batch size
    int64_t ubForCols = perLoopCols * 2 + perLoopCols +
                        CeilDiv(perLoopCols, blocksize) * 5;  // per-row column buffers
    int64_t ubForIndices = static_cast<int64_t>(ubSize) - ubForCols - 1024;  // extra overhead
    if (ubForIndices < BLOCK_BYTES) {
        ubForIndices = BLOCK_BYTES;
    }
    int64_t maxPerLoopIndices = ubForIndices / INT32_SIZE;
    maxPerLoopIndices = AlignDown(maxPerLoopIndices, INT32_ONE_BLOCK_NUM);
    if (maxPerLoopIndices < 1) {
        maxPerLoopIndices = 1;
    }

    int64_t perLoopIndices = (maxPerLoopIndices < perCoreIndices) ? maxPerLoopIndices : perCoreIndices;
    gather.perCoreIndicesLoops = CeilDiv(perCoreIndices, perLoopIndices);
    gather.perCorePerLoopIndicesElements = perLoopIndices;
    gather.perCoreLastLoopIndicesElements = perCoreIndices - perLoopIndices * (gather.perCoreIndicesLoops - 1);
    if (gather.perCoreLastLoopIndicesElements <= 0) {
        gather.perCoreLastLoopIndicesElements = perLoopIndices;
    }

    int64_t lastPerLoopIndices = (maxPerLoopIndices < lastCoreIndices) ? maxPerLoopIndices : lastCoreIndices;
    if (lastCoreIndices <= 0) {
        gather.lastCoreIndicesLoops = 0;
        gather.lastCorePerLoopIndicesElements = 0;
        gather.lastCoreLastLoopIndicesElements = 0;
    } else {
        gather.lastCoreIndicesLoops = CeilDiv(lastCoreIndices, lastPerLoopIndices);
        gather.lastCorePerLoopIndicesElements = lastPerLoopIndices;
        gather.lastCoreLastLoopIndicesElements = lastCoreIndices -
            lastPerLoopIndices * (gather.lastCoreIndicesLoops - 1);
        if (gather.lastCoreLastLoopIndicesElements <= 0) {
            gather.lastCoreLastLoopIndicesElements = lastPerLoopIndices;
        }
    }
}

// ==================== Workspace Calculation ====================

static int64_t CalcWorkspaceSize(int64_t n, int64_t k, int64_t actualExpertNum, int64_t coreNum)
{
    int64_t totalElements = n * k;

    // ws_sorted_expert_idx: aligned n*k int32
    int64_t wsSortedExpertIdx = AlignUp(totalElements * INT32_SIZE, BLOCK_BYTES);
    // ws_sorted_row_idx: aligned n*k int32
    int64_t wsSortedRowIdx = AlignUp(totalElements * INT32_SIZE, BLOCK_BYTES);
    // ws_expert_count_temp: per-expert count array (actualExpertNum int32)
    int64_t wsExpertCountTemp = AlignUp(actualExpertNum * INT32_SIZE, BLOCK_BYTES);
    // ws_expert_total_count: scalar written at offset 0 of a separate region after expert_count_temp
    // (kernel reads expertTotalCountGm_ at expert_count_temp + Align(actualExpertNum, sizeof(int32_t)))
    int64_t wsExpertTotalCount = AlignUp(actualExpertNum * INT32_SIZE, BLOCK_BYTES);

    // ws_sort_middle: multi-core merge sort intermediate buffer
    // Each core produces a sorted segment; merge requires temp space
    // Reference: coreNum sorted arrays, each up to perCoreElements
    // Merge buffer: at least 2 * totalElements * INT32_SIZE (key + value pairs)
    int64_t wsSortMiddle = 0;
    if (coreNum > 1) {
        // For multi-core merge: need buffer for merged key-value pairs
        // plus sync counters. Conservative estimate.
        wsSortMiddle = AlignUp(totalElements * INT32_SIZE * 2, BLOCK_BYTES) +
                       AlignUp(coreNum * INT32_SIZE, BLOCK_BYTES);
    }

    return wsSortedExpertIdx + wsSortedRowIdx + wsExpertCountTemp + wsExpertTotalCount + wsSortMiddle;
}

// ==================== Sort Threshold ====================

static int64_t CalcSortThreshold(int64_t ubSize)
{
    // When n*k fits in single-core VBS sort (within UB capacity), use ONECORE
    // Sort needs: sortData(int32) + rowIdx(int32) + assistData + tempBuffer
    // ~ ubSize / (3 * sizeof(int32)) aligned to sort boundary
    int64_t threshold = ubSize / (3 * INT32_SIZE);
    threshold = AlignDown(threshold, ONE_REPEAT_SORT_NUM);
    return threshold;
}

// ==================== TilingPrepare ====================

static ge::graphStatus TilingPrepareForMoeInitRoutingV3MxQuant(gert::TilingParseContext* context)
{
    auto compileInfo = context->GetCompiledInfo<MoeInitRoutingV3MxQuantCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);

    auto* platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto platform = platform_ascendc::PlatformAscendC(platformInfo);

    compileInfo->coreNum = platform.GetCoreNumAiv();
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfo->ubSize);

    return ge::GRAPH_SUCCESS;
}

// ==================== Main Tiling Function ====================

static ge::graphStatus TilingForMoeInitRoutingV3MxQuant(gert::TilingContext* context)
{
    uint32_t coreNum = 0;
    uint64_t ubSize = 0;

    auto compileInfo = context->GetCompileInfo<MoeInitRoutingV3MxQuantCompileInfo>();
    if (compileInfo != nullptr) {
        coreNum = compileInfo->coreNum;
        ubSize = compileInfo->ubSize;
    } else {
        // Fallback: TilingParse may not be called in cannsim environment
        auto* platformInfo = context->GetPlatformInfo();
        OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
        auto platform = platform_ascendc::PlatformAscendC(platformInfo);
        coreNum = platform.GetCoreNumAiv();
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    }

    // ==================== Read input shapes ====================
    // x: [num_tokens, hidden_size]
    auto xShapePtr = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShapePtr);
    const gert::Shape xShape = xShapePtr->GetStorageShape();
    int64_t n = xShape.GetDim(0);       // num_tokens
    int64_t cols = xShape.GetDim(1);     // hidden_size

    // expert_idx: [num_tokens, topk]
    auto expertIdxShapePtr = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, expertIdxShapePtr);
    const gert::Shape expertIdxShape = expertIdxShapePtr->GetStorageShape();
    int64_t k = expertIdxShape.GetDim(1);  // topk

    // ==================== Read attributes ====================
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

    // active_num (index 0)
    int64_t activeNum = -1;
    const int64_t* activeNumPtr = attrs->GetAttrPointer<int64_t>(ATTR_IDX_ACTIVE_NUM);
    if (activeNumPtr != nullptr) {
        activeNum = *activeNumPtr;
    }

    // expert_capacity (index 1)
    int64_t expertCapacity = -1;
    const int64_t* expertCapacityPtr = attrs->GetAttrPointer<int64_t>(ATTR_IDX_EXPERT_CAPACITY);
    if (expertCapacityPtr != nullptr) {
        expertCapacity = *expertCapacityPtr;
    }

    // expert_num (index 2)
    int64_t expertNum = -1;
    const int64_t* expertNumPtr = attrs->GetAttrPointer<int64_t>(ATTR_IDX_EXPERT_NUM);
    if (expertNumPtr != nullptr) {
        expertNum = *expertNumPtr;
    }

    // drop_pad_mode (index 3)
    int64_t dropPadMode = 0;
    const int64_t* dropPadModePtr = attrs->GetAttrPointer<int64_t>(ATTR_IDX_DROP_PAD_MODE);
    if (dropPadModePtr != nullptr) {
        dropPadMode = *dropPadModePtr;
    }

    // expert_tokens_count_or_cumsum_flag (index 4)
    int64_t expertTokensCountFlag = 0;
    const int64_t* expertTokensCountFlagPtr = attrs->GetAttrPointer<int64_t>(ATTR_IDX_EXPERT_TOKENS_COUNT_FLAG);
    if (expertTokensCountFlagPtr != nullptr) {
        expertTokensCountFlag = *expertTokensCountFlagPtr;
    }

    // expert_tokens_before_capacity_flag (index 5)
    int64_t expertTokensBeforeCapFlag = 0;
    const bool* expertTokensBeforeCapFlagPtr = attrs->GetAttrPointer<bool>(ATTR_IDX_EXPERT_TOKENS_BEFORE_CAP);
    if (expertTokensBeforeCapFlagPtr != nullptr) {
        expertTokensBeforeCapFlag = (*expertTokensBeforeCapFlagPtr) ? 1 : 0;
    }

    // axis (index 6) -- must be -1
    int64_t axis = -1;
    const int64_t* axisPtr = attrs->GetAttrPointer<int64_t>(ATTR_IDX_AXIS);
    if (axisPtr != nullptr) {
        axis = *axisPtr;
    }
    if (axis != -1) {
        return ge::GRAPH_PARAM_INVALID;
    }

    // round_mode (index 7) -- must be "rint"
    const char* roundModePtr = attrs->GetAttrPointer<char>(ATTR_IDX_ROUND_MODE);
    if (roundModePtr != nullptr) {
        std::string roundMode(roundModePtr);
        if (roundMode != "rint") {
            return ge::GRAPH_PARAM_INVALID;
        }
    }

    // dst_type (index 8)
    int64_t dstType = 0;
    const int64_t* dstTypePtr = attrs->GetAttrPointer<int64_t>(ATTR_IDX_DST_TYPE);
    if (dstTypePtr != nullptr) {
        dstType = *dstTypePtr;
    }

    // blocksize (index 9)
    int64_t blocksize = MX_BLOCK_SIZE;
    const int64_t* blocksizePtr = attrs->GetAttrPointer<int64_t>(ATTR_IDX_BLOCKSIZE);
    if (blocksizePtr != nullptr) {
        blocksize = *blocksizePtr;
    }

    // scale_alg (index 10) -- must be 0
    int64_t scaleAlg = 0;
    const int64_t* scaleAlgPtr = attrs->GetAttrPointer<int64_t>(ATTR_IDX_SCALE_ALG);
    if (scaleAlgPtr != nullptr) {
        scaleAlg = *scaleAlgPtr;
    }
    if (scaleAlg != 0) {
        return ge::GRAPH_FAILED;
    }

    // ==================== Determine optional inputs ====================
    // scale (input index 2): OPTIONAL
    int64_t isInputScale = 0;
    auto scaleShapePtr = context->GetInputShape(2);
    if (scaleShapePtr != nullptr && scaleShapePtr->GetStorageShape().GetDimNum() > 0) {
        isInputScale = 1;
    }

    // offset (input index 3): OPTIONAL
    int64_t isInputOffset = 0;
    auto offsetShapePtr = context->GetInputShape(3);
    if (offsetShapePtr != nullptr && offsetShapePtr->GetStorageShape().GetDimNum() > 0) {
        isInputOffset = 1;
    }

    // ==================== Compute derived parameters ====================
    int64_t effectiveN = (activeNum > 0 && activeNum < n) ? activeNum : n;
    int64_t totalElements = effectiveN * k;  // total expert assignments

    // Expert range
    int64_t actualExpertNum = expertNum;
    if (actualExpertNum <= 0) {
        // expertNum not provided or invalid; use conservative default
        actualExpertNum = 64;
    }
    int64_t expertStart = 0;
    int64_t expertEnd = actualExpertNum;

    // Row index type: determine based on output expanded_row_idx semantics
    // For this fused op, default to GATHER (0) since we gather x by sorted_row_idx
    int64_t rowIdxType = IDX_GATHER;

    // ==================== Sort mode selection ====================
    int64_t sortThreshold = CalcSortThreshold(static_cast<int64_t>(ubSize));
    int64_t sortMode = (totalElements <= sortThreshold) ? SORT_ONE_CORE : SORT_MULTI_CORE;

    // ==================== TilingKey ====================
    // tilingKey = 1000000 + sortMode*100000 + 30000 + idxMode*1000
    int64_t tilingKey = 1000000 + sortMode * 100000 + 30000 + rowIdxType * 1000;
    context->SetTilingKey(static_cast<uint64_t>(tilingKey));

    // ==================== BlockDim ====================
    context->SetBlockDim(coreNum);

    // ==================== Calculate sub-stage tiling ====================
    MoeInitRoutingV3MxQuantArch35TilingData tilingData;
    std::memset(&tilingData, 0, sizeof(tilingData));

    // Fill main fields
    tilingData.coreNum = coreNum;
    tilingData.n = n;
    tilingData.cols = cols;
    tilingData.k = k;
    tilingData.expertStart = expertStart;
    tilingData.expertEnd = expertEnd;
    tilingData.actualExpertNum = actualExpertNum;
    tilingData.rowIdxType = rowIdxType;
    tilingData.isInputScale = isInputScale;
    tilingData.isInputOffset = isInputOffset;
    tilingData.expertNum = expertNum;
    tilingData.expertTokensNumType = expertTokensCountFlag;
    tilingData.expertTokensNumFlag = expertTokensBeforeCapFlag;
    tilingData.activeNum = activeNum;
    tilingData.dropPadMode = dropPadMode;

    // MX quantization fields
    tilingData.dstType = dstType;
    tilingData.blocksize = blocksize;
    tilingData.scaleAlg = scaleAlg;

    // VBS sort tiling
    CalcVBSTiling(tilingData.vbsComputeParamsOp, totalElements, coreNum,
                  static_cast<int64_t>(ubSize));

    // VMS merge sort tiling
    CalcVMSTiling(tilingData.vmsMiddleComputeParamsOp, coreNum);

    // SortOut tiling
    CalcSortOutTiling(tilingData.sortOutComputeParamsOp, static_cast<int64_t>(ubSize));

    // ExpertTokensCount tiling
    CalcExpertTokensCountTiling(tilingData.expertTokensCountTilingDataOp,
                                 totalElements, coreNum, static_cast<int64_t>(ubSize));

    // GatherOut tiling (includes column tiling for MX quantization)
    CalcGatherOutTiling(tilingData.gatherOutComputeParamsOp,
                        totalElements, cols, coreNum,
                        static_cast<int64_t>(ubSize), blocksize, activeNum);

    // ==================== Workspace ====================
    int64_t workspaceSize = CalcWorkspaceSize(n, k, actualExpertNum, coreNum);
    size_t* workspaceSizes = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspaceSizes);
    workspaceSizes[0] = static_cast<size_t>(workspaceSize);

    // ==================== Write TilingData ====================
    // Serialize the tiling data struct to the tiling buffer
    auto tilingBuf = context->GetRawTilingData();
    OP_CHECK_NULL_WITH_CONTEXT(context, tilingBuf);
    tilingBuf->Append(reinterpret_cast<const uint8_t*>(&tilingData), sizeof(tilingData));
    tilingBuf->SetDataSize(sizeof(tilingData));

    return ge::GRAPH_SUCCESS;
}

// ==================== Registration ====================

IMPL_OP_OPTILING(MoeInitRoutingV3MxQuant)
    .Tiling(TilingForMoeInitRoutingV3MxQuant)
    .TilingParse<MoeInitRoutingV3MxQuantCompileInfo>(TilingPrepareForMoeInitRoutingV3MxQuant);

}  // namespace optiling
