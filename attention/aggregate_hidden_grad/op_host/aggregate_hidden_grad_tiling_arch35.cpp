/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aggregate_hidden_grad_arch35.cpp
 * \brief AggregateHiddenGrad tiling implementation
 */

#include "aggregate_hidden_grad_tiling_arch35.h"
#include <algorithm>
#include "securec.h"

namespace optiling {


using AggregateHiddenGradArch35Tiling::AggregateHiddenGradTilingDataV35;

bool AggregateHiddenGradTiling::IsCapable()
{
    // 详细校验在 CheckInputParams 中完成，这里返回 true
    return true;
}

ge::graphStatus AggregateHiddenGradTiling::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo == nullptr) {
        auto compileInfoPtr =
            reinterpret_cast<const AggregateHiddenGradArch35CompileInfo *>(context_->GetCompileInfo());
        OP_CHECK_IF(compileInfoPtr == nullptr, OP_LOGE(context_, "compile info is null"), return ge::GRAPH_FAILED);
        totalCoreNum_ = compileInfoPtr->coreNum;
        ubSize_ = compileInfoPtr->ubSize;
    } else {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        totalCoreNum_ = static_cast<uint64_t>(ascendcPlatform.GetCoreNumAiv());
        OP_CHECK_IF(totalCoreNum_ == 0UL, OP_LOGE(context_->GetNodeName(), "coreNum is 0"), return ge::GRAPH_FAILED);
        uint64_t ubSize = 0;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
        OP_CHECK_IF(ubSize == static_cast<uint64_t>(0), OP_LOGE(context_->GetNodeName(), "ubSize is 0"),
                    return ge::GRAPH_FAILED);
        ubSize_ = static_cast<uint64_t>(ubSize);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AggregateHiddenGradTiling::ValidateGradOutputShape()
{
    auto shape = context_->GetInputShape(GRAD_OUTPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, shape);
    auto origin = shape->GetOriginShape();

    OP_CHECK_IF(origin.GetDimNum() != 3,
                OP_LOGE(context_->GetNodeName(), "grad_output dim num must be 3, but got %lu", origin.GetDimNum()),
                return ge::GRAPH_FAILED);

    S_ = static_cast<int64_t>(origin.GetDim(DIM_0));
    B_ = static_cast<int64_t>(origin.GetDim(DIM_1));
    H_ = static_cast<int64_t>(origin.GetDim(DIM_2));

    // 约束范围（来自需求文档）：S∈[1, 32K], B∈[1, 8], H∈[384, 24576]
    OP_CHECK_IF(S_ <= 0 || S_ > (1 << 15), OP_LOGE(context_->GetNodeName(), "invalid S=%ld", S_),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(B_ <= 0 || B_ > 8, OP_LOGE(context_->GetNodeName(), "invalid B=%ld", B_), return ge::GRAPH_FAILED);
    OP_CHECK_IF(H_ < 384 || H_ > (192 * 128), OP_LOGE(context_->GetNodeName(), "invalid H=%ld", H_),
                return ge::GRAPH_FAILED);

    // H 必须按 64 对齐以便 H 方向切分
    OP_CHECK_IF((H_ % DIM_ALIGN_ELEMENT) != 0,
                OP_LOGE(context_->GetNodeName(), "H must be multiple of %ld, got %ld", DIM_ALIGN_ELEMENT, H_),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AggregateHiddenGradTiling::ValidateInputShape()
{
    auto goShape = context_->GetInputShape(GRAD_OUTPUT_INDEX)->GetOriginShape();

    auto inShape = context_->GetInputShape(INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inShape);
    auto inOrigin = inShape->GetOriginShape();

    OP_CHECK_IF(inOrigin.GetDimNum() != 3,
                OP_LOGE(context_->GetNodeName(), "input dim num must be 3, but got %lu", inOrigin.GetDimNum()),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(goShape.GetDim(DIM_0) != inOrigin.GetDim(DIM_0) || goShape.GetDim(DIM_1) != inOrigin.GetDim(DIM_1) ||
                    goShape.GetDim(DIM_2) != inOrigin.GetDim(DIM_2),
                OP_LOGE(context_->GetNodeName(), "input shape mismatch with grad_output"), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AggregateHiddenGradTiling::ValidateWeightShape()
{
    auto wShape = context_->GetInputShape(WEIGHT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, wShape);
    auto wOrigin = wShape->GetOriginShape();

    OP_CHECK_IF(wOrigin.GetDimNum() != 2,
                OP_LOGE(context_->GetNodeName(), "weight dim num must be 2, but got %lu", wOrigin.GetDimNum()),
                return ge::GRAPH_FAILED);

    W_ = static_cast<int64_t>(wOrigin.GetDim(DIM_0));
    auto wH = static_cast<int64_t>(wOrigin.GetDim(DIM_1));

    OP_CHECK_IF(W_ != 3, OP_LOGE(context_->GetNodeName(), "weight[0]=W must be 3, but got %ld", W_),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(wH != H_, OP_LOGE(context_->GetNodeName(), "weight dim1 %ld must match H %ld", wH, H_),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AggregateHiddenGradTiling::ValidateMaskShape()
{
    auto maskShape = context_->GetOptionalInputShape(MASK_INDEX);
    if (maskShape == nullptr) {
        hasMask_ = 0;
        return ge::GRAPH_SUCCESS;
    }
    hasMask_ = 1;
    auto mOrigin = maskShape->GetOriginShape();

    OP_CHECK_IF(mOrigin.GetDimNum() != 2,
                OP_LOGE(context_->GetNodeName(), "mask dim num must be 2, but got %lu", mOrigin.GetDimNum()),
                return ge::GRAPH_FAILED);

    auto mB = static_cast<int64_t>(mOrigin.GetDim(DIM_0));
    auto mS = static_cast<int64_t>(mOrigin.GetDim(DIM_1));

    OP_CHECK_IF(mB != B_ || mS != S_,
                OP_LOGE(context_->GetNodeName(), "mask shape must be [B,S]=[%ld,%ld], got [%ld,%ld]", B_, S_, mB, mS),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AggregateHiddenGradTiling::ValidateGradOutputType()
{
    dataType_ = context_->GetInputDesc(GRAD_OUTPUT_INDEX)->GetDataType();
    OP_CHECK_IF(dataType_ != ge::DataType::DT_FLOAT16 && dataType_ != ge::DataType::DT_BF16,
                OP_LOGE(context_->GetNodeName(), "grad_output dtype must be FLOAT16 or BF16, but got %s",
                        Ops::Base::ToString(dataType_).c_str()),
                return ge::GRAPH_FAILED);
    dtypeSize_ = DTYPE_SIZE;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AggregateHiddenGradTiling::ValidateInputType()
{
    auto t = context_->GetInputDesc(INPUT_INDEX)->GetDataType();
    OP_CHECK_IF(t != dataType_,
                OP_LOGE(context_->GetNodeName(), "input dtype %s must match grad_output dtype %s",
                        Ops::Base::ToString(t).c_str(), Ops::Base::ToString(dataType_).c_str()),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AggregateHiddenGradTiling::ValidateWeightType()
{
    auto t = context_->GetInputDesc(WEIGHT_INDEX)->GetDataType();
    OP_CHECK_IF(t != dataType_,
                OP_LOGE(context_->GetNodeName(), "weight dtype %s must match grad_output dtype %s",
                        Ops::Base::ToString(t).c_str(), Ops::Base::ToString(dataType_).c_str()),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AggregateHiddenGradTiling::ValidateMaskType()
{
    auto desc = context_->GetOptionalInputDesc(MASK_INDEX);
    if (desc == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    auto t = desc->GetDataType();
    OP_CHECK_IF(
        t != ge::DataType::DT_UINT8,
        OP_LOGE(context_->GetNodeName(), "mask dtype must be UINT8, but got %s", Ops::Base::ToString(t).c_str()),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AggregateHiddenGradTiling::CheckInputParams()
{
    // Shapes
    OP_CHECK_IF(ValidateGradOutputShape() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "grad_output shape validation failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ValidateInputShape() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "input shape validation failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ValidateWeightShape() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "weight shape validation failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ValidateMaskShape() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "mask shape validation failed"), return ge::GRAPH_FAILED);

    // Types
    OP_CHECK_IF(ValidateGradOutputType() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "grad_output type validation failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ValidateInputType() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "input type validation failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ValidateWeightType() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "weight type validation failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ValidateMaskType() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "mask type validation failed"), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AggregateHiddenGradTiling::GetShapeAttrsInfo()
{
    OP_CHECK_IF(context_ == nullptr, OP_LOGE("AggregateHiddenGrad", "context is null"), return ge::GRAPH_FAILED);

    // 收集形状和类型，并做完整校验
    OP_CHECK_IF(CheckInputParams() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "AggregateHiddenGrad CheckInputParams FAILED."),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AggregateHiddenGradTiling::ComputeInterCoreSplit()
{
    // 仅沿 H 方向按 64 切分
    // 根据文档：主核处理的H会比尾核多64(hMainSize = hTailSize + 64)
    int64_t tiles = H_ / DIM_ALIGN_ELEMENT;
    OP_CHECK_IF(tiles <= 0, OP_LOGE(context_->GetNodeName(), "H(%ld) < %ld", H_, DIM_ALIGN_ELEMENT),
                return ge::GRAPH_FAILED);

    int64_t maxCores = static_cast<int64_t>(totalCoreNum_);
    usedCoreNum_ = std::min<int64_t>(tiles, std::max<int64_t>(1, maxCores));

    if (H_ % usedCoreNum_ == 0) {
        // H能均分，则认为没有尾核
        hMainCoreCnt_ = usedCoreNum_;
        hTailCoreCnt_ = 0;
        hMainSize_ = H_ / usedCoreNum_;
        hTailSize_ = hMainSize_;  // 文档要求：hTailSize仍等于hMainSize
    } else {
        // 计算主核和尾核的分配
        int64_t baseTiles = H_ / (usedCoreNum_ * DIM_ALIGN_ELEMENT);
        hTailSize_ = baseTiles * DIM_ALIGN_ELEMENT;
        hMainSize_ = hTailSize_ + DIM_ALIGN_ELEMENT;  // 主核比尾核多64

        // 计算需要多少主核
        int64_t totalWithMain = hMainSize_ * usedCoreNum_;
        int64_t excess = totalWithMain - H_;
        hTailCoreCnt_ = excess / DIM_ALIGN_ELEMENT;
        hMainCoreCnt_ = usedCoreNum_ - hTailCoreCnt_;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AggregateHiddenGradTiling::ComputeIntraCoreUbTiling()
{
    // UB 预留系统空间
    int64_t availableUbSize = static_cast<int64_t>(ubSize_) - SYSTEM_RESERVED_UB_SIZE;
    OP_CHECK_IF(availableUbSize <= 0, OP_LOGE(context_->GetNodeName(), "available UB size <= 0, ubSize=%lu", ubSize_),
                return ge::GRAPH_FAILED);

    // 根据文档：保证H不小于64元素情况下，多载BS
    // UB先按H=64(128B)，全载核内BS
    int64_t minH = DIM_ALIGN_ELEMENT;  // 64
    int64_t perCoreH = (hMainCoreCnt_ > 0) ? hMainSize_ : hTailSize_;

    // Buffer占用估算函数
    auto calculateBufferSize = [&](int64_t h, int64_t b, int64_t s) -> int64_t {
        // 根据文档第4节Buffer设计
        int64_t gradOutputSize = h * b * s * static_cast<int64_t>(dtypeSize_);
        int64_t inputSize = h * b * s * static_cast<int64_t>(dtypeSize_);
        int64_t weightSize = h * W_ * static_cast<int64_t>(dtypeSize_);
        int64_t maskSize = (hasMask_ != 0) ? ((b * s + 7) / 8) : 0;            // BOOL类型，8bit对齐
        int64_t gradInputSize = h * b * s * static_cast<int64_t>(dtypeSize_);
        int64_t gradWeightSize = h * W_ * static_cast<int64_t>(dtypeSize_);

        // 所有Buffer都是double buffer，所以乘2
        int64_t totalSize = 2 * (gradOutputSize + inputSize + weightSize + maskSize +
                                 gradInputSize + gradWeightSize);
        return totalSize;
    };

    // 首先尝试H=64，全载BS
    hUB_ = minH;
    bUB_ = B_;
    sUB_ = S_;

    if (calculateBufferSize(hUB_, bUB_, sUB_) <= availableUbSize) {
        // 能够全载，则尝试增加H（保证H*DTypeSize为128B的倍数）
        int64_t h_increment = 128 / static_cast<int64_t>(dtypeSize_);  // FP16/BF16: 64
        int64_t maxH = std::min(perCoreH, (int64_t)(availableUbSize / (2 * B_ * S_ * dtypeSize_ + 2 * W_ * dtypeSize_)));

        while (hUB_ + h_increment <= maxH) {
            if (calculateBufferSize(hUB_ + h_increment, bUB_, sUB_) <= availableUbSize) {
                hUB_ += h_increment;
            } else {
                break;
            }
        }
    } else {
        // 不能全载，先压缩B
        bUB_ = 1;
        if (calculateBufferSize(hUB_, bUB_, sUB_) > availableUbSize) {
            // B=1仍不能全载，再压缩S
            while (sUB_ > 1) {
                sUB_ = (sUB_ + 1) / 2;  // 二分压缩
                if (calculateBufferSize(hUB_, bUB_, sUB_) <= availableUbSize) {
                    break;
                }
            }
            // 最小保证S=1
            if (calculateBufferSize(hUB_, bUB_, sUB_) > availableUbSize) {
                sUB_ = 1;
            }
        }
    }

    // 计算主核的循环次数和尾块大小
    hLoopCnt_ = (perCoreH + hUB_ - 1) / hUB_;
    bLoopCnt_ = (B_ + bUB_ - 1) / bUB_;
    sLoopCnt_ = (S_ + sUB_ - 1) / sUB_;

    // 计算主核的主块和尾块大小
    ubMainFactorH_ = hUB_;
    ubTailFactorH_ = (hLoopCnt_ == 1) ? hUB_ : (perCoreH % hUB_ == 0 ? hUB_ : perCoreH % hUB_);

    ubMainFactorB_ = bUB_;
    ubTailFactorB_ = (bLoopCnt_ == 1) ? bUB_ : (B_ % bUB_ == 0 ? bUB_ : B_ % bUB_);

    ubMainFactorS_ = sUB_;
    ubTailFactorS_ = (sLoopCnt_ == 1) ? sUB_ : (S_ % sUB_ == 0 ? sUB_ : S_ % sUB_);

    // 计算尾核的参数（如果有尾核）
    if (hTailCoreCnt_ > 0 && hTailSize_ > 0) {
        // 尾核可能需要不同的H循环次数
        int64_t tailCoreH = hTailSize_;
        tailHloopCnt_ = (tailCoreH + hUB_ - 1) / hUB_;
        tailBLoopCnt_ = bLoopCnt_;  // B和S的循环次数保持一致
        tailSLoopCnt_ = sLoopCnt_;

        tailCoreUbMainFactorH_ = hUB_;
        tailCoreUbTailFactorH_ = (tailHloopCnt_ == 1) ? hUB_ :
                                 (tailCoreH % hUB_ == 0 ? hUB_ : tailCoreH % hUB_);

        tailCoreUbMainFactorB_ = ubMainFactorB_;
        tailCoreUbTailFactorB_ = ubTailFactorB_;

        tailCoreUbMainFactorS_ = ubMainFactorS_;
        tailCoreUbTailFactorS_ = ubTailFactorS_;
    } else {
        // 没有尾核，或尾核参数与主核相同
        tailHloopCnt_ = hLoopCnt_;
        tailBLoopCnt_ = bLoopCnt_;
        tailSLoopCnt_ = sLoopCnt_;

        tailCoreUbMainFactorH_ = ubMainFactorH_;
        tailCoreUbTailFactorH_ = ubTailFactorH_;
        tailCoreUbMainFactorB_ = ubMainFactorB_;
        tailCoreUbTailFactorB_ = ubTailFactorB_;
        tailCoreUbMainFactorS_ = ubMainFactorS_;
        tailCoreUbTailFactorS_ = ubTailFactorS_;
    }

    // 保留原有的兼容性变量（用于日志输出）
    hUBTail_ = ubTailFactorH_;
    bUBTail_ = ubTailFactorB_;
    sUBTail_ = ubTailFactorS_;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AggregateHiddenGradTiling::DoOpTiling()
{
    OP_CHECK_IF(ComputeInterCoreSplit() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "ComputeInterCoreSplit failed"), return ge::GRAPH_FAILED);

    OP_CHECK_IF(ComputeIntraCoreUbTiling() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "ComputeIntraCoreUbTiling failed"), return ge::GRAPH_FAILED);

    // 填充 tilingData_ - 按照文档3.3节的结构体定义
    // 核间切分参数
    tilingData_.hMainCoreCnt = hMainCoreCnt_;
    tilingData_.hTailCoreCnt = hTailCoreCnt_;
    tilingData_.hMainSize = hMainSize_;
    tilingData_.hTailSize = hTailSize_;

    // 主核循环参数
    tilingData_.hloopCnt = hLoopCnt_;
    tilingData_.bLoopCnt = bLoopCnt_;
    tilingData_.sLoopCnt = sLoopCnt_;

    // 主核UB切块参数
    tilingData_.ubMainFactorH = ubMainFactorH_;
    tilingData_.ubTailFactorH = ubTailFactorH_;
    tilingData_.ubMainFactorB = ubMainFactorB_;
    tilingData_.ubTailFactorB = ubTailFactorB_;
    tilingData_.ubMainFactorS = ubMainFactorS_;
    tilingData_.ubTailFactorS = ubTailFactorS_;

    // 尾核循环参数
    tilingData_.tailHloopCnt = tailHloopCnt_;
    tilingData_.tailBLoopCnt = tailBLoopCnt_;
    tilingData_.tailSLoopCnt = tailSLoopCnt_;

    // 尾核UB切块参数
    tilingData_.tailCoreUbMainFactorH = tailCoreUbMainFactorH_;
    tilingData_.tailCoreUbTailFactorH = tailCoreUbTailFactorH_;
    tilingData_.tailCoreUbMainFactorB = tailCoreUbMainFactorB_;
    tilingData_.tailCoreUbTailFactorB = tailCoreUbTailFactorB_;
    tilingData_.tailCoreUbMainFactorS = tailCoreUbMainFactorS_;
    tilingData_.tailCoreUbTailFactorS = tailCoreUbTailFactorS_;

    // 全局参数
    tilingData_.hasMask = hasMask_;
    tilingData_.S = S_;
    tilingData_.B = B_;
    tilingData_.H = H_;
    tilingData_.W = W_;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AggregateHiddenGradTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t AggregateHiddenGradTiling::GetTilingKey() const
{
    // 可按 dtype 区分，当前返回 0
    if (dataType_ == ge::DataType::DT_BF16) {
        return TILING_KEY_AGGREGATE_HIDDEN_GRAD_BF16;
    } else if (dataType_ == ge::DataType::DT_FLOAT16) {
        return TILING_KEY_AGGREGATE_HIDDEN_GRAD_FP16;
    }
    
}

ge::graphStatus AggregateHiddenGradTiling::GetWorkspaceSize()
{
    auto platformInfo = context_->GetPlatformInfo();
    uint32_t sysWorkspaceSize = 0;
    if (platformInfo != nullptr) {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    }
    size_t *currentWorkspace = context_->GetWorkspaceSizes(1);
    currentWorkspace[0] = static_cast<size_t>(0UL + sysWorkspaceSize);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AggregateHiddenGradTiling::PostTiling()
{
    // 设置核数
    context_->SetBlockDim(static_cast<uint32_t>(usedCoreNum_));

    // 保存 tiling 数据
    auto *raw = context_->GetRawTilingData();
    auto tilingDataSize = sizeof(AggregateHiddenGradTilingDataV35);
    errno_t ret = memcpy_s(raw->GetData(), raw->GetCapacity(), reinterpret_cast<void *>(&tilingData_), tilingDataSize);
    if (ret != EOK) {
        OP_LOGE(context_->GetNodeName(), "memcpy_s failed, ret=%d", ret);
        return ge::GRAPH_FAILED;
    }
    raw->SetDataSize(tilingDataSize);

    context_->SetTilingKey(GetTilingKey());
    return ge::GRAPH_SUCCESS;
}

void AggregateHiddenGradTiling::DumpTilingInfo()
{
    OP_LOGI(context_->GetNodeName(), "=== AggregateHiddenGrad DumpTilingInfo ===");
    OP_LOGI(context_->GetNodeName(), "S=%ld B=%ld H=%ld W=%ld", S_, B_, H_, W_);
    OP_LOGI(context_->GetNodeName(), "hasMask=%ld dtypeSize=%zu", hasMask_, dtypeSize_);
    OP_LOGI(context_->GetNodeName(), "totalCoreNum=%lu usedCoreNum=%ld", totalCoreNum_, usedCoreNum_);
    OP_LOGI(context_->GetNodeName(), "hMainCoreCnt=%ld hTailCoreCnt=%ld", hMainCoreCnt_, hTailCoreCnt_);
    OP_LOGI(context_->GetNodeName(), "hMainSize=%ld hTailSize=%ld", hMainSize_, hTailSize_);
    OP_LOGI(context_->GetNodeName(), "hUB=%ld bUB=%ld sUB=%ld", hUB_, bUB_, sUB_);
    OP_LOGI(context_->GetNodeName(), "hLoopCnt=%ld bLoopCnt=%ld sLoopCnt=%ld", hLoopCnt_, bLoopCnt_, sLoopCnt_);
    OP_LOGI(context_->GetNodeName(), "hUBTail=%ld bUBTail=%ld sUBTail=%ld", hUBTail_, bUBTail_, sUBTail_);
}


} // namespace optiling
