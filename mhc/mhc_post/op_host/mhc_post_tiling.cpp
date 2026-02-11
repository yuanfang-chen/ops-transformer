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
 * \file mhc_post_tiling.cpp
 * \brief MhcPost tiling implementation
 */

#include <cmath>
#include <algorithm>
#include <vector>
#include "register/op_def_registry.h"
#include "tiling_base/tiling_templates_registry.h"
#include "tiling_base/tiling_base.h"
#include "platform/platform_info.h"
#include "log/log.h"
#include "mhc_post_tiling.h"

namespace optiling {

// Constants for memory alignment and buffer configuration
constexpr int64_t DEFAULT_WORKSPACE_SIZE = 0;
constexpr uint64_t TILING_KEY_GENERALIZED = 0;

// Memory alignment constants (in elements)
constexpr uint32_t BF16_FP16_ALIGN_SIZE = 16;  // 16 elements = 32 bytes for bf16/fp16
constexpr uint32_t FLOAT32_ALIGN_SIZE = 8;     // 8 elements = 32 bytes for float32

// Double Buffer configuration
constexpr uint32_t DOUBLE_BUFFER_DEPTH = 2;    // Double Buffer depth for data tiles
constexpr uint32_t SINGLE_BUFFER_DEPTH = 1;    // Single Buffer depth for weights

// Spec constraints
constexpr uint32_t MAX_TOTAL_ITEMS = 512 * 1024;  // BS max 512K
constexpr uint32_t MIN_D = 384;                    // D min = 192 * 2
constexpr uint32_t MAX_D = 24576;                  // D max = 192 * 128

// Ceiling division
inline int64_t CeilDiv(int64_t a, int64_t b)
{
    return (b == 0) ? 0 : (a + b - 1) / b;
}

// Align value up to the nearest multiple of align
inline uint32_t AlignUp(uint32_t value, uint32_t align)
{
    return (align == 0) ? 0 : ((value + align - 1) / align) * align;
}

// Align value down to the nearest multiple of align
inline uint32_t AlignDown(uint32_t value, uint32_t align)
{
    return (align == 0) ? 0 : (value / align) * align;
}

// Input indices - 按照OpDef定义的顺序
// Input: x, h_res, h_out, h_post
const static int64_t X_INPUT_INDEX = 0;           // residual (B, S, n, D)
const static int64_t H_RES_INPUT_INDEX = 1;       // comb (B, S, n, n)
const static int64_t H_OUT_INPUT_INDEX = 2;       // x (B, S, D)
const static int64_t H_POST_INPUT_INDEX = 3;      // post (B, S, n)

// Output indices
const static int64_t OUTPUT_INDEX = 0;            // output (B, S, n, D)

class MhcPostTilingBase : public Ops::Transformer::OpTiling::TilingBaseClass {
public:
    explicit MhcPostTilingBase(gert::TilingContext *context)
        : Ops::Transformer::OpTiling::TilingBaseClass(context)
    {
        Reset();
    }
    ~MhcPostTilingBase() override = default;

    void Reset(gert::TilingContext *context) override
    {
        TilingBaseClass::Reset(context);
        Reset();
    }

protected:
    bool IsCapable() override
    {
        return true;
    }

    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    void Reset();

private:
    // Check functions
    ge::graphStatus CheckNullptr();
    ge::graphStatus CheckShapeAllPositive(int64_t idx) const;
    ge::graphStatus CheckShapeAllPositive();
    ge::graphStatus CheckDataType();
    ge::graphStatus CheckShapeConsistency();
    ge::graphStatus CheckSpecConstraints();
    ge::graphStatus CheckParam();

    void ComputeTiling();

    const gert::Shape *xShape_ = nullptr;

    uint32_t B_ = 0;
    uint32_t S_ = 0;
    uint32_t n_ = 0;
    uint32_t D_ = 0;
    uint32_t totalItems_ = 0;
    uint32_t usedCores_ = 0;
    uint32_t itemsPerCore_ = 0;
    uint32_t remainderItems_ = 0;
    uint32_t tileD_ = 0;
    uint32_t nTilesD_ = 0;

    const char *opName_ = "";
    ge::DataType dtype_ = ge::DT_UNDEFINED;

    MhcPostTilingData* tilingData_ = context_->GetTilingData<MhcPostTilingData>();;
};

ge::graphStatus MhcPostTilingBase::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo == nullptr) {
        OP_LOGE(context_, "fail to get platform info");
        return ge::GRAPH_FAILED;
    }

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    aicoreParams_.numBlocks = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    aicoreParams_.ubSize = ubSizePlatForm;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPostTilingBase::GetShapeAttrsInfo()
{
    opName_ = context_->GetNodeName();

    // 获取x shape信息: (B, S, n, D)
    auto xShapePtr = context_->GetInputShape(X_INPUT_INDEX);
    if (xShapePtr == nullptr) {
        OP_LOGE(context_, "x shape is null");
        return ge::GRAPH_FAILED;
    }
    xShape_ = &xShapePtr->GetStorageShape();

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPostTilingBase::CheckNullptr()
{
    // Check all input desc and shape
    for (int64_t i = X_INPUT_INDEX; i <= H_POST_INPUT_INDEX; i++) {
        auto desc = context_->GetInputDesc(i);
        OP_CHECK_IF(desc == nullptr,
                    OP_LOGE(context_, "input %ld desc is nullptr", i),
                    return ge::GRAPH_FAILED);
        auto shape = context_->GetInputShape(i);
        OP_CHECK_IF(shape == nullptr,
                    OP_LOGE(context_, "input %ld shape is nullptr", i),
                    return ge::GRAPH_FAILED);
    }

    // Check output desc and shape
    auto desc = context_->GetOutputDesc(OUTPUT_INDEX);
    OP_CHECK_IF(desc == nullptr,
                OP_LOGE(context_, "output desc is nullptr"),
                return ge::GRAPH_FAILED);
    auto shape = context_->GetOutputShape(OUTPUT_INDEX);
    OP_CHECK_IF(shape == nullptr,
                OP_LOGE(context_, "output shape is nullptr"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPostTilingBase::CheckShapeAllPositive(int64_t idx) const
{
    auto shape = context_->GetInputShape(idx)->GetStorageShape();
    for (size_t i = 0; i < shape.GetDimNum(); i++) {
        OP_CHECK_IF(shape.GetDim(i) <= 0,
                    OP_LOGE(context_, "input %ld has non-positive shape, dim %lu actual %ld",
                            idx, i, shape.GetDim(i)),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPostTilingBase::CheckShapeAllPositive()
{
    // Check all inputs
    for (int64_t i = X_INPUT_INDEX; i <= H_POST_INPUT_INDEX; i++) {
        OP_CHECK_IF(CheckShapeAllPositive(i) != ge::GRAPH_SUCCESS,
                    OP_LOGE(context_, "input %ld has non-positive shape", i),
                    return ge::GRAPH_FAILED);
    }

    // Check output
    auto shape = context_->GetOutputShape(OUTPUT_INDEX)->GetStorageShape();
    for (size_t i = 0; i < shape.GetDimNum(); i++) {
        OP_CHECK_IF(shape.GetDim(i) <= 0,
                    OP_LOGE(context_, "output has non-positive shape, dim %lu actual %ld",
                            i, shape.GetDim(i)),
                    return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPostTilingBase::CheckDataType()
{
    // Get x dtype as reference
    dtype_ = context_->GetInputDesc(X_INPUT_INDEX)->GetDataType();

    // Check supported dtype
    const std::vector<ge::DataType> supportedDtype = {ge::DT_BF16, ge::DT_FLOAT16};
    OP_CHECK_IF(std::find(supportedDtype.begin(), supportedDtype.end(), dtype_) == supportedDtype.end(),
                OP_LOGE(context_, "Only support BF16 and FP16 dtype, actual %s",
                        ge::TypeUtils::DataTypeToSerialString(dtype_).c_str()),
                return ge::GRAPH_FAILED);

    // Check x and h_out have same dtype (bf16/fp16)
    auto hOutType = context_->GetInputDesc(H_OUT_INPUT_INDEX)->GetDataType();
    OP_CHECK_IF(hOutType != dtype_,
                OP_LOGE(context_, "h_out datatype expect %s, actual %s",
                        ge::TypeUtils::DataTypeToSerialString(dtype_).c_str(),
                        ge::TypeUtils::DataTypeToSerialString(hOutType).c_str()),
                return ge::GRAPH_FAILED);

    // Check h_res and h_post are float32
    auto hResType = context_->GetInputDesc(H_RES_INPUT_INDEX)->GetDataType();
    OP_CHECK_IF(hResType != ge::DT_FLOAT,
                OP_LOGE(context_, "h_res datatype must be float32, actual %s",
                        ge::TypeUtils::DataTypeToSerialString(hResType).c_str()),
                return ge::GRAPH_FAILED);

    auto hPostType = context_->GetInputDesc(H_POST_INPUT_INDEX)->GetDataType();
    OP_CHECK_IF(hPostType != ge::DT_FLOAT,
                OP_LOGE(context_, "h_post datatype must be float32, actual %s",
                        ge::TypeUtils::DataTypeToSerialString(hPostType).c_str()),
                return ge::GRAPH_FAILED);

    // Check output dtype matches x dtype
    auto outputType = context_->GetOutputDesc(OUTPUT_INDEX)->GetDataType();
    OP_CHECK_IF(outputType != dtype_,
                OP_LOGE(context_, "output datatype expect %s, actual %s",
                        ge::TypeUtils::DataTypeToSerialString(dtype_).c_str(),
                        ge::TypeUtils::DataTypeToSerialString(outputType).c_str()),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPostTilingBase::CheckShapeConsistency()
{
    OP_CHECK_IF(xShape_ == nullptr,
                OP_LOGE(context_, "x shape is null"),
                return ge::GRAPH_FAILED);

    // Support both BSND (4D) and TND (3D) formats
    // BSND: (B, S, n, D) -> totalItems = B * S
    // TND:  (T, n, D)    -> totalItems = T
    uint32_t dimNum = xShape_->GetDimNum();

    if (dimNum == 4) {
        // BSND format: (B, S, n, D)
        int64_t B_int = xShape_->GetDim(0);
        int64_t S_int = xShape_->GetDim(1);
        int64_t n_int = xShape_->GetDim(2);
        int64_t D_int = xShape_->GetDim(3);

        B_ = static_cast<uint32_t>(B_int);
        S_ = static_cast<uint32_t>(S_int);
        n_ = static_cast<uint32_t>(n_int);
        D_ = static_cast<uint32_t>(D_int);
        totalItems_ = B_ * S_;
        OP_LOGI(context_, "BSND format: B=%u, S=%u, n=%u, D=%u, totalItems=%u", B_, S_, n_, D_, totalItems_);
    } else if (dimNum == 3) {
        // TND format: (T, n, D)
        int64_t T_int = xShape_->GetDim(0);
        int64_t n_int = xShape_->GetDim(1);
        int64_t D_int = xShape_->GetDim(2);

        B_ = 1;  // Not used in TND format
        S_ = 1;  // Not used in TND format
        totalItems_ = static_cast<uint32_t>(T_int);
        n_ = static_cast<uint32_t>(n_int);
        D_ = static_cast<uint32_t>(D_int);
        OP_LOGI(context_, "TND format: T=%u, n=%u, D=%u", totalItems_, n_, D_);
    } else {
        OP_LOGE(context_, "Unsupported input dimension: %u (expected 3 for TND or 4 for BSND)", dimNum);
        return ge::GRAPH_FAILED;
    }

    // Cross-validate all input shapes to ensure consistency
    // Input: x(0), h_res(1), h_out(2), h_post(3)
    // Expected shapes:
    //   BSND (4D): x(B,S,n,D), h_res(B,S,n,n), h_out(B,S,D), h_post(B,S,n)
    //   TND (3D):  x(T,n,D),   h_res(T,n,n),   h_out(T,D),   h_post(T,n)

    if (dimNum == 4) {
        // BSND format validation
        int64_t B = static_cast<int64_t>(B_);
        int64_t S = static_cast<int64_t>(S_);
        int64_t n = static_cast<int64_t>(n_);
        int64_t D = static_cast<int64_t>(D_);

        // Validate h_res: (B, S, n, n)
        auto hResShapePtr = context_->GetInputShape(H_RES_INPUT_INDEX);
        const gert::Shape* hResShape = &hResShapePtr->GetStorageShape();
        OP_CHECK_IF(hResShape->GetDimNum() != dimNum,
                    OP_LOGE(context_, "h_res has %u dimensions, expected %u (format mismatch)",
                            hResShape->GetDimNum(), dimNum),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(hResShape->GetDim(0) != B || hResShape->GetDim(1) != S ||
                    hResShape->GetDim(2) != n || hResShape->GetDim(3) != n,
                    OP_LOGE(context_, "h_res shape (%ld,%ld,%ld,%ld) != expected (%ld,%ld,%ld,%ld)",
                            hResShape->GetDim(0), hResShape->GetDim(1), hResShape->GetDim(2), hResShape->GetDim(3),
                            B, S, n, n),
                    return ge::GRAPH_FAILED);

        // Validate h_out: (B, S, D)
        auto hOutShapePtr = context_->GetInputShape(H_OUT_INPUT_INDEX);
        const gert::Shape* hOutShape = &hOutShapePtr->GetStorageShape();
        OP_CHECK_IF(hOutShape->GetDimNum() != 3,
                    OP_LOGE(context_, "h_out has %u dimensions, expected 3",
                            hOutShape->GetDimNum()),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(hOutShape->GetDim(0) != B || hOutShape->GetDim(1) != S || hOutShape->GetDim(2) != D,
                    OP_LOGE(context_, "h_out shape (%ld,%ld,%ld) != expected (%ld,%ld,%ld)",
                            hOutShape->GetDim(0), hOutShape->GetDim(1), hOutShape->GetDim(2),
                            B, S, D),
                    return ge::GRAPH_FAILED);

        // Validate h_post: (B, S, n)
        auto hPostShapePtr = context_->GetInputShape(H_POST_INPUT_INDEX);
        const gert::Shape* hPostShape = &hPostShapePtr->GetStorageShape();
        OP_CHECK_IF(hPostShape->GetDimNum() != 3,
                    OP_LOGE(context_, "h_post has %u dimensions, expected 3",
                            hPostShape->GetDimNum()),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(hPostShape->GetDim(0) != B || hPostShape->GetDim(1) != S || hPostShape->GetDim(2) != n,
                    OP_LOGE(context_, "h_post shape (%ld,%ld,%ld) != expected (%ld,%ld,%ld)",
                            hPostShape->GetDim(0), hPostShape->GetDim(1), hPostShape->GetDim(2),
                            B, S, n),
                    return ge::GRAPH_FAILED);

        // Validate output: (B, S, n, D)
        auto outputShapePtr = context_->GetOutputShape(OUTPUT_INDEX);
        const gert::Shape* outputShape = &outputShapePtr->GetStorageShape();
        OP_CHECK_IF(outputShape->GetDimNum() != dimNum,
                    OP_LOGE(context_, "output has %u dimensions, expected %u (format mismatch)",
                            outputShape->GetDimNum(), dimNum),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(outputShape->GetDim(0) != B || outputShape->GetDim(1) != S ||
                    outputShape->GetDim(2) != n || outputShape->GetDim(3) != D,
                    OP_LOGE(context_, "output shape (%ld,%ld,%ld,%ld) != expected (%ld,%ld,%ld,%ld)",
                            outputShape->GetDim(0), outputShape->GetDim(1), outputShape->GetDim(2), outputShape->GetDim(3),
                            B, S, n, D),
                    return ge::GRAPH_FAILED);

    } else {  // dimNum == 3
        // TND format validation
        int64_t T = static_cast<int64_t>(totalItems_);
        int64_t n = static_cast<int64_t>(n_);
        int64_t D = static_cast<int64_t>(D_);

        // Validate h_res: (T, n, n)
        auto hResShapePtr = context_->GetInputShape(H_RES_INPUT_INDEX);
        const gert::Shape* hResShape = &hResShapePtr->GetStorageShape();
        OP_CHECK_IF(hResShape->GetDimNum() != dimNum,
                    OP_LOGE(context_, "h_res has %u dimensions, expected %u (format mismatch)",
                            hResShape->GetDimNum(), dimNum),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(hResShape->GetDim(0) != T || hResShape->GetDim(1) != n || hResShape->GetDim(2) != n,
                    OP_LOGE(context_, "h_res shape (%ld,%ld,%ld) != expected (%ld,%ld,%ld)",
                            hResShape->GetDim(0), hResShape->GetDim(1), hResShape->GetDim(2),
                            T, n, n),
                    return ge::GRAPH_FAILED);

        // Validate h_out: (T, D)
        auto hOutShapePtr = context_->GetInputShape(H_OUT_INPUT_INDEX);
        const gert::Shape* hOutShape = &hOutShapePtr->GetStorageShape();
        OP_CHECK_IF(hOutShape->GetDimNum() != 2,
                    OP_LOGE(context_, "h_out has %u dimensions, expected 2",
                            hOutShape->GetDimNum()),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(hOutShape->GetDim(0) != T || hOutShape->GetDim(1) != D,
                    OP_LOGE(context_, "h_out shape (%ld,%ld) != expected (%ld,%ld)",
                            hOutShape->GetDim(0), hOutShape->GetDim(1),
                            T, D),
                    return ge::GRAPH_FAILED);

        // Validate h_post: (T, n)
        auto hPostShapePtr = context_->GetInputShape(H_POST_INPUT_INDEX);
        const gert::Shape* hPostShape = &hPostShapePtr->GetStorageShape();
        OP_CHECK_IF(hPostShape->GetDimNum() != 2,
                    OP_LOGE(context_, "h_post has %u dimensions, expected 2",
                            hPostShape->GetDimNum()),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(hPostShape->GetDim(0) != T || hPostShape->GetDim(1) != n,
                    OP_LOGE(context_, "h_post shape (%ld,%ld) != expected (%ld,%ld)",
                            hPostShape->GetDim(0), hPostShape->GetDim(1),
                            T, n),
                    return ge::GRAPH_FAILED);

        // Validate output: (T, n, D)
        auto outputShapePtr = context_->GetOutputShape(OUTPUT_INDEX);
        const gert::Shape* outputShape = &outputShapePtr->GetStorageShape();
        OP_CHECK_IF(outputShape->GetDimNum() != dimNum,
                    OP_LOGE(context_, "output has %u dimensions, expected %u (format mismatch)",
                            outputShape->GetDimNum(), dimNum),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(outputShape->GetDim(0) != T || outputShape->GetDim(1) != n || outputShape->GetDim(2) != D,
                    OP_LOGE(context_, "output shape (%ld,%ld,%ld) != expected (%ld,%ld,%ld)",
                            outputShape->GetDim(0), outputShape->GetDim(1), outputShape->GetDim(2),
                            T, n, D),
                    return ge::GRAPH_FAILED);
    }

    OP_LOGI(context_, "All input and output shapes validated successfully");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPostTilingBase::CheckSpecConstraints()
{
    // 1. Check totalItems (BS or T) <= 512K
    OP_CHECK_IF(totalItems_ == 0,
                OP_LOGE(context_, "totalItems cannot be zero"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(totalItems_ > MAX_TOTAL_ITEMS,
                OP_LOGE(context_, "totalItems (%u) exceeds max allowed (%u)", totalItems_, MAX_TOTAL_ITEMS),
                return ge::GRAPH_FAILED);

    // 2. Check n in {4, 6, 8}
    OP_CHECK_IF(n_ != 4 && n_ != 6 && n_ != 8,
                OP_LOGE(context_, "n (%u) must be 4, 6, or 8", n_),
                return ge::GRAPH_FAILED);

    // 3. Check D in [384, 24576]
    OP_CHECK_IF(D_ < MIN_D || D_ > MAX_D,
                OP_LOGE(context_, "D (%u) must be in range [%u, %u]", D_, MIN_D, MAX_D),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPostTilingBase::CheckParam()
{
    OP_CHECK_IF(CheckNullptr() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_, "CheckNullptr failed"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(CheckDataType() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_, "CheckDataType failed"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(CheckShapeConsistency() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_, "CheckShapeConsistency failed"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(CheckSpecConstraints() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_, "CheckSpecConstraints failed"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(CheckShapeAllPositive() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_, "CheckShapeAllPositive failed"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

void MhcPostTilingBase::ComputeTiling()
{
    // Core Partitioning - handle remainder properly
    uint32_t coreNum = static_cast<uint32_t>(aicoreParams_.numBlocks);
    usedCores_ = (totalItems_ < coreNum) ? totalItems_ : coreNum;
    itemsPerCore_ = totalItems_ / usedCores_;
    remainderItems_ = totalItems_ % usedCores_;

    // Dynamic Tiling Strategy - maximize tileD based on UB size
    // UB usage formula (generalized for any n):
    // TQue (bf16): (2n+2) * tileD * 2 bytes
    //   - hOutTile: tileD, xTile: n*tileD
    //   - outputTile: n*tileD
    // TQue (f32): (2n+2) * sizeof(float) bytes (small, n-related buffers)
    //   - hPost: n, hRes: n*n
    // TBuf (f32): (2n+4) * tileD * 4 bytes
    //   - hOutF32: tileD, xF32: n*tileD
    //   - outF32: tileD, temp: tileD
    // Simplified formula: bytesPerTileD = (2n+2)*2 + (2n+4)*4 = 4n+4 + 8n+16 = 12n+20
    // More accurate: include n-related small buffers separately
    const uint32_t UB_SIZE = static_cast<uint32_t>(aicoreParams_.ubSize);

    // Calculate aligned n and n*n for float32 vector operations
    uint32_t alignedN = AlignUp(n_, FLOAT32_ALIGN_SIZE);
    uint32_t alignedNN = AlignUp(n_ * n_, FLOAT32_ALIGN_SIZE);

    // Calculate bytes per tileD element based on actual n
    // TQue bf16: (2n+2) * 2 bytes (hOut:1, x:n, output:n)
    // TBuf f32:  (2n+4) * 4 bytes (hOutF32:1, xF32:n, outF32:1, temp:1)
    // Small buffers (fixed, not per-tileD):
    //   - TQue f32: (alignedN + alignedNN) * 4 for hPost, hRes
    uint32_t bytesPerTileD = (2 * n_ + 2) * 2 + (2 * n_ + 4) * 4;  // = 12n + 20
    uint32_t smallBufferBytes = (alignedN + alignedNN) * 4;  // aligned sizes

    // Reserve space for small buffers, then calculate max tileD
    uint32_t availableUB = UB_SIZE - smallBufferBytes;
    uint32_t maxTileD = availableUB / bytesPerTileD;
    maxTileD = AlignDown(maxTileD, BF16_FP16_ALIGN_SIZE);  // Align to 16 elements

    // Align D to BF16_FP16_ALIGN_SIZE for proper memory access
    uint32_t alignedD = AlignUp(D_, BF16_FP16_ALIGN_SIZE);

    // Determine optimal tileD:
    if (alignedD <= maxTileD) {
        tileD_ = alignedD;
        nTilesD_ = 1;
    } else {
        // Find largest aligned value that fits in UB
        tileD_ = maxTileD;
        // Try to find a divisor of alignedD for even splitting
        while (tileD_ >= BF16_FP16_ALIGN_SIZE) {
            if (alignedD % tileD_ == 0) {
                break;  // 找到可以整除的tileD
            }
            tileD_ -= BF16_FP16_ALIGN_SIZE;
        }
        // Fallback: if no exact divisor found, use maxTileD
        if (tileD_ < BF16_FP16_ALIGN_SIZE) {
            tileD_ = maxTileD;
        }
        nTilesD_ = CeilDiv(alignedD, tileD_);
    }

    // Calculate last tile size
    uint32_t lastTileD = alignedD - (nTilesD_ - 1) * tileD_;
    // Ensure lastTileD doesn't exceed tileD
    if (lastTileD > tileD_) {
        lastTileD = tileD_;
    }

    // Set tiling data
    tilingData_->totalItems = totalItems_;
    tilingData_->itemsPerCore = itemsPerCore_;
    tilingData_->remainderItems = remainderItems_;
    tilingData_->usedCores = usedCores_;
    tilingData_->S = S_;
    tilingData_->n = n_;
    tilingData_->D = D_;
    tilingData_->tileD = tileD_;
    tilingData_->nTilesD = nTilesD_;
    tilingData_->alignedD = alignedD;
    tilingData_->lastTileD = lastTileD;
    tilingData_->alignedN = alignedN;
    tilingData_->alignedNN = alignedNN;

    OP_LOGI(context_,
        "Tiling: n=%u, D=%u, alignedD=%u, tileD=%u, lastTileD=%u, nTilesD=%u",
        n_, D_, alignedD, tileD_, lastTileD, nTilesD_);
    OP_LOGI(context_, "Tiling: alignedN=%u, alignedNN=%u", alignedN, alignedNN);
    OP_LOGI(context_,
        "Tiling: usedCores=%u, itemsPerCore=%u, remainderItems=%u, UB=%u, bytesPerTileD=%u",
        usedCores_, itemsPerCore_, remainderItems_, UB_SIZE, bytesPerTileD);
}

ge::graphStatus MhcPostTilingBase::DoOpTiling()
{
    auto ret = CheckParam();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    ComputeTiling();

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPostTilingBase::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPostTilingBase::GetWorkspaceSize()
{
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo == nullptr) {
        OP_LOGE(context_, "fail to get platform info");
        return ge::GRAPH_FAILED;
    }

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    workspaceSize_ = ascendcPlatform.GetLibApiWorkSpaceSize();

    OP_LOGI(context_, "Workspace size: %ld bytes (%.2f MB)",
            workspaceSize_, workspaceSize_ / (1024.0 * 1024.0));

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPostTilingBase::PostTiling()
{
    context_->SetBlockDim(usedCores_);
    size_t *currentWorkspace = context_->GetWorkspaceSizes(1);
    currentWorkspace[0] = workspaceSize_;
    return ge::GRAPH_SUCCESS;
}

uint64_t MhcPostTilingBase::GetTilingKey() const
{
    // Calculate fast path flags
    // For n: n=4 (16 bytes) and n=8 (32 bytes) are 8-aligned for float32
    // n=6 (24 bytes) is not 8-aligned
    uint16_t isNAligned = (n_ % FLOAT32_ALIGN_SIZE == 0) ? 1 : 0;
    // For n*n: 16 (n=4) and 64 (n=8) are 8-aligned, 36 (n=6) is not
    uint16_t isNNAligned = ((n_ * n_) % FLOAT32_ALIGN_SIZE == 0) ? 1 : 0;
    // D is aligned if D % 16 == 0 and single tile
    uint16_t isDAligned = ((D_ % BF16_FP16_ALIGN_SIZE == 0) && (nTilesD_ == 1)) ? 1 : 0;
    OP_LOGI(context_,
        "Tiling:isNAligned=%u, isNNAligned=%u, isDAligned=%u", isNAligned, isNNAligned, isDAligned);

    return GET_TPL_TILING_KEY(isNAligned, isNNAligned, isDAligned);
}

void MhcPostTilingBase::Reset()
{
    opName_ = nullptr;
    xShape_ = nullptr;
    B_ = 0;
    S_ = 0;
    n_ = 0;
    D_ = 0;
    totalItems_ = 0;
    usedCores_ = 0;
    itemsPerCore_ = 0;
    remainderItems_ = 0;
    tileD_ = 0;
    nTilesD_ = 0;
}

REGISTER_OPS_TILING_TEMPLATE(MhcPost, MhcPostTilingBase, 2000);
} // namespace optiling