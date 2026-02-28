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
 * \file causal_conv1d_fn_tiling_arch35.cpp
 * \brief
 */

#include "causal_conv1d_fn_tiling_arch35.h"

namespace optiling {
constexpr uint64_t DIM_0 = 0;
constexpr uint64_t DIM_1 = 1;
constexpr uint64_t DIM_2 = 2;

constexpr uint64_t INPUT_X_INDEX = 0;
constexpr uint64_t INPUT_WEIGHT_INDEX = 1;
constexpr uint64_t INPUT_CACHE_STATES_INDEX = 2;
constexpr uint64_t INPUT_CACHE_INDICES_INDEX = 3;
constexpr uint64_t INPUT_SEQ_START_INDEX = 4;
constexpr uint64_t INPUT_HAS_INITIAL_STATE_INDEX = 5;

constexpr uint64_t OUTPUT_Y_INDEX = 0;
constexpr uint64_t OUTPUT_CACHE_STATES_INDEX = 1;

constexpr uint64_t X_DIM_NUM = 2;
constexpr uint64_t WEIGHT_DIM_NUM = 2;
constexpr uint64_t CACHE_STATES_DIM_NUM = 3;
constexpr uint64_t SEQ_START_INDEX_DIM_NUM = 1;

constexpr uint64_t DIM_MIN = 64;
constexpr uint64_t DIM_MAX = 16384;
constexpr uint64_t DIM_ALIGN = 16;
constexpr uint64_t CU_SEQ_LEN_MIN = 1;
constexpr uint64_t CU_SEQ_LEN_MAX = 65536;
constexpr uint64_t BATCH_MIN = 1;
constexpr uint64_t BATCH_MAX = 256;
constexpr uint64_t KERNEL_WIDTH_MAX = 6;

constexpr uint64_t ALIGN_BYTES = 256;
constexpr uint64_t MIN_CORE_NUM_FOR_DIM_SPLIT = 32;
constexpr uint64_t MIN_DIM_PER_CORE = 256;
constexpr uint64_t DOUBLE_BUFFER_NUM = 2;
constexpr uint64_t TILING_KEY_VALUE = 10000UL;
constexpr uint64_t SYS_WORKSPACE_SIZE = static_cast<uint64_t>(16 * 1024 * 1024);

bool CausalConv1dFnTiling::IsCapable()
{
    return true;
}

ge::graphStatus CausalConv1dFnTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CausalConv1dFnTiling::GetPlatformInfo()
{
    ubBlockSize_ = Ops::Base::GetUbBlockSize(context_);
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo == nullptr) {
        OP_LOGE(context_->GetNodeName(), "platform info is null");
        return ge::GRAPH_FAILED;
    }

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    totalCoreNum_ = static_cast<uint64_t>(ascendcPlatform.GetCoreNumAiv());
    if (totalCoreNum_ == 0UL) {
        OP_LOGE(context_->GetNodeName(), "coreNum is 0");
        return ge::GRAPH_FAILED;
    }

    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    if (ubSize == static_cast<uint64_t>(0)) {
        OP_LOGE(context_->GetNodeName(), "ubSize is 0");
        return ge::GRAPH_FAILED;
    }
    ubSize_ = static_cast<uint64_t>(ubSize);

    return ge::GRAPH_SUCCESS;
}

// 检查输入数据类型
ge::graphStatus CausalConv1dFnTiling::CheckInputDtype()
{
    if (xType_ != ge::DataType::DT_FLOAT16 && xType_ != ge::DataType::DT_BF16) {
        OP_LOGE(context_->GetNodeName(), "X dtype must be fp16 or bf16, but got: %s",
                Ops::Base::ToString(xType_).c_str());
        return ge::GRAPH_FAILED;
    }

    if (weightType_ != xType_) {
        OP_LOGE(context_->GetNodeName(), "Weight dtype must equal to X dtype. X dtype: %s, Weight dtype: %s",
                Ops::Base::ToString(xType_).c_str(), Ops::Base::ToString(weightType_).c_str());
        return ge::GRAPH_FAILED;
    }

    auto cacheStatesType = context_->GetInputDesc(INPUT_CACHE_STATES_INDEX)->GetDataType();
    if (cacheStatesType != xType_) {
        OP_LOGE(context_->GetNodeName(), "CacheStates dtype must equal to X dtype. X dtype: %s, CacheStates dtype: %s",
                Ops::Base::ToString(xType_).c_str(), Ops::Base::ToString(cacheStatesType).c_str());
        return ge::GRAPH_FAILED;
    }

    auto cacheIndicesType = context_->GetInputDesc(INPUT_CACHE_INDICES_INDEX)->GetDataType();
    if (cacheIndicesType != ge::DataType::DT_INT32) {
        OP_LOGE(context_->GetNodeName(), "CacheIndices dtype must be INT32, but got: %s",
                Ops::Base::ToString(cacheIndicesType).c_str());
        return ge::GRAPH_FAILED;
    }

    auto seqStartIndexType = context_->GetInputDesc(INPUT_SEQ_START_INDEX)->GetDataType();
    if (seqStartIndexType != ge::DataType::DT_INT32) {
        OP_LOGE(context_->GetNodeName(), "SeqStartIndex dtype must be INT32, but got: %s",
                Ops::Base::ToString(seqStartIndexType).c_str());
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

// 检查输入数据维度
ge::graphStatus CausalConv1dFnTiling::CheckInputDim()
{
    // 检查X的维度
    uint64_t xDimNum = xShape_.GetDimNum();
    OP_CHECK_IF(xDimNum != X_DIM_NUM,
                OP_LOGE(context_->GetNodeName(), "X dim must be 2, but got: %lu", xDimNum),
                return ge::GRAPH_FAILED);

    // 检查cu_seq_len范围
    OP_CHECK_IF(!(cuSeqLen_ >= CU_SEQ_LEN_MIN && cuSeqLen_ <= CU_SEQ_LEN_MAX),
                OP_LOGE(context_->GetNodeName(), "cu_seq_len must in [%lu, %lu], but got: %lu",
                        CU_SEQ_LEN_MIN, CU_SEQ_LEN_MAX, cuSeqLen_),
                return ge::GRAPH_FAILED);

    // 检查dim范围和对齐
    OP_CHECK_IF(!(dim_ >= DIM_MIN && dim_ <= DIM_MAX && dim_ % DIM_ALIGN == 0),
                OP_LOGE(context_->GetNodeName(), "dim must in [%lu, %lu] and be multiple of %lu, but got: %lu",
                        DIM_MIN, DIM_MAX, DIM_ALIGN, dim_),
                return ge::GRAPH_FAILED);

    // 检查weight的维度
    uint64_t weightDimNum = weightShape_.GetDimNum();
    OP_CHECK_IF(weightDimNum != WEIGHT_DIM_NUM,
                OP_LOGE(context_->GetNodeName(), "Weight dim must be 2, but got: %lu", weightDimNum),
                return ge::GRAPH_FAILED);

    uint64_t weightDim = weightShape_.GetDim(DIM_1);
    OP_CHECK_IF(weightDim != dim_,
                OP_LOGE(context_->GetNodeName(), "Weight dim[1] must equal to X dim[1], X dim: %lu, Weight dim: %lu",
                        dim_, weightDim),
                return ge::GRAPH_FAILED);

    // 检查kernel width
    OP_CHECK_IF(kernelWidth_ > KERNEL_WIDTH_MAX,
                OP_LOGE(context_->GetNodeName(), "Kernel width must <= %lu, but got: %u",
                        KERNEL_WIDTH_MAX, kernelWidth_),
                return ge::GRAPH_FAILED);

    // 检查cacheStates的维度
    uint64_t cacheStatesDimNum = cacheStatesShape_.GetDimNum();
    OP_CHECK_IF(cacheStatesDimNum != CACHE_STATES_DIM_NUM,
                OP_LOGE(context_->GetNodeName(), "CacheStates dim must be 3, but got: %lu", cacheStatesDimNum),
                return ge::GRAPH_FAILED);

    uint64_t cacheStatesDim1 = cacheStatesShape_.GetDim(DIM_1);
    OP_CHECK_IF(cacheStatesDim1 != (kernelWidth_ - 1),
                OP_LOGE(context_->GetNodeName(), "CacheStates dim[1] must equal to K-1=%u, but got: %lu",
                        kernelWidth_ - 1, cacheStatesDim1),
                return ge::GRAPH_FAILED);

    uint64_t cacheStatesDim2 = cacheStatesShape_.GetDim(DIM_2);
    OP_CHECK_IF(cacheStatesDim2 != dim_,
                OP_LOGE(context_->GetNodeName(), "CacheStates dim[2] must equal to dim=%lu, but got: %lu",
                        dim_, cacheStatesDim2),
                return ge::GRAPH_FAILED);

    // 检查seqStartIndex的维度
    uint64_t seqStartIndexDimNum = seqStartIndexShape_.GetDimNum();
    OP_CHECK_IF(seqStartIndexDimNum != SEQ_START_INDEX_DIM_NUM,
                OP_LOGE(context_->GetNodeName(), "SeqStartIndex dim must be 1, but got: %lu", seqStartIndexDimNum),
                return ge::GRAPH_FAILED);

    uint64_t seqStartIndexDim0 = seqStartIndexShape_.GetDim(DIM_0);
    OP_CHECK_IF(seqStartIndexDim0 != (batch_ + 1),
                OP_LOGE(context_->GetNodeName(), "SeqStartIndex dim[0] must equal to batch+1=%u, but got: %lu",
                        batch_ + 1, seqStartIndexDim0),
                return ge::GRAPH_FAILED);

    // 检查batch范围
    OP_CHECK_IF(!(batch_ >= BATCH_MIN && batch_ <= BATCH_MAX),
                OP_LOGE(context_->GetNodeName(), "batch must in [%lu, %lu], but got: %u",
                        BATCH_MIN, BATCH_MAX, batch_),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// 检查输入参数
ge::graphStatus CausalConv1dFnTiling::CheckInputParams()
{
    if (CheckInputDtype() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckInputDim() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CausalConv1dFnTiling::CheckOutputParams()
{
    auto outputYDesc = context_->GetOutputDesc(OUTPUT_Y_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputYDesc);
    auto outputYType = outputYDesc->GetDataType();
    OP_CHECK_IF(xType_ != outputYType,
                OP_LOGE(context_->GetNodeName(), "Output Y dtype must equal to X dtype. X dtype: %s, Y dtype: %s",
                        Ops::Base::ToString(xType_).c_str(), Ops::Base::ToString(outputYType).c_str()),
                return ge::GRAPH_FAILED);

    auto outputCacheStatesDesc = context_->GetOutputDesc(OUTPUT_CACHE_STATES_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputCacheStatesDesc);
    auto outputCacheStatesType = outputCacheStatesDesc->GetDataType();
    OP_CHECK_IF(xType_ != outputCacheStatesType,
                OP_LOGE(context_->GetNodeName(),
                        "Output CacheStates dtype must equal to X dtype. X dtype: %s, CacheStates dtype: %s",
                        Ops::Base::ToString(xType_).c_str(), Ops::Base::ToString(outputCacheStatesType).c_str()),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CausalConv1dFnTiling::GetShapeAttrsInfo()
{
    OP_CHECK_IF(context_ == nullptr, OP_LOGE("CausalConv1dFn", "context is null"), return ge::GRAPH_FAILED);

    // 获取输入shape
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(INPUT_X_INDEX));
    xShape_ = context_->GetInputShape(INPUT_X_INDEX)->GetOriginShape();
    cuSeqLen_ = xShape_.GetDim(DIM_0);
    dim_ = xShape_.GetDim(DIM_1);

    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(INPUT_WEIGHT_INDEX));
    weightShape_ = context_->GetInputShape(INPUT_WEIGHT_INDEX)->GetOriginShape();
    kernelWidth_ = static_cast<uint32_t>(weightShape_.GetDim(DIM_0));

    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(INPUT_CACHE_STATES_INDEX));
    cacheStatesShape_ = context_->GetInputShape(INPUT_CACHE_STATES_INDEX)->GetOriginShape();

    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(INPUT_SEQ_START_INDEX));
    seqStartIndexShape_ = context_->GetInputShape(INPUT_SEQ_START_INDEX)->GetOriginShape();
    batch_ = static_cast<uint32_t>(seqStartIndexShape_.GetDim(DIM_0) - 1);

    // 获取输入数据类型
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(INPUT_X_INDEX));
    xType_ = context_->GetInputDesc(INPUT_X_INDEX)->GetDataType();

    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(INPUT_WEIGHT_INDEX));
    weightType_ = context_->GetInputDesc(INPUT_WEIGHT_INDEX)->GetDataType();

    xDtypeSize_ = GetSizeByDataType(xType_);
    OP_CHECK_IF(xDtypeSize_ == 0,
                OP_LOGE(context_->GetNodeName(), "CausalConv1dFn get X dtype[%s] size is 0.",
                        Ops::Base::ToString(xType_).c_str()),
                return ge::GRAPH_FAILED);

    // 读取 padSlotId attribute
    padSlotId_ = -1;
    if (context_->GetAttrs() != nullptr && context_->GetAttrs()->GetInt(0) != nullptr) {
        padSlotId_ = *(context_->GetAttrs()->GetInt(0));
    }

    // 初始化有效 batch 范围（默认为全部 batch）
    validBatchStart_ = 0;
    validBatchCount_ = batch_;
    validSeqStart_ = 0;
    validSeqLen_ = cuSeqLen_;

    // 如果有 padSlotId，需要读取 cacheIndices 来确定有效 batch 范围
    if (padSlotId_ >= 0) {
        // 尝试读取 cacheIndices 数据
        const gert::Tensor* cacheIndicesTensor = context_->GetInputTensor(INPUT_CACHE_INDICES_INDEX);
        if (cacheIndicesTensor != nullptr && cacheIndicesTensor->GetData<int32_t>() != nullptr) {
            // 获取 cacheIndices tensor (batch 个 int32 元素)
            const int32_t* cacheIndices = cacheIndicesTensor->GetData<int32_t>();

            // 从前往后找第一个不等于 padSlotId 的位置
            uint32_t validStart = batch_;  // 默认全是padding
            for (uint32_t i = 0; i < batch_; i++) {
                if (static_cast<int64_t>(cacheIndices[i]) != padSlotId_) {
                    validStart = i;
                    break;
                }
            }

            // 从后往前找最后一个不等于 padSlotId 的位置
            uint32_t validEnd = 0;
            for (int32_t i = static_cast<int32_t>(batch_) - 1; i >= 0; i--) {
                if (static_cast<int64_t>(cacheIndices[i]) != padSlotId_) {
                    validEnd = static_cast<uint32_t>(i);
                    break;
                }
            }

            // 计算有效 batch 数量
            if (validStart <= validEnd && validStart < batch_) {
                validBatchStart_ = validStart;
                validBatchCount_ = validEnd - validStart + 1;

                // 读取 queryStartLoc 计算有效序列范围
                const gert::Tensor* queryStartLocTensor = context_->GetInputTensor(INPUT_SEQ_START_INDEX);
                if (queryStartLocTensor != nullptr && queryStartLocTensor->GetData<int32_t>() != nullptr) {
                    const int32_t* queryStartLoc = queryStartLocTensor->GetData<int32_t>();
                    validSeqStart_ = static_cast<uint64_t>(queryStartLoc[validBatchStart_]);
                    uint64_t validSeqEnd = static_cast<uint64_t>(queryStartLoc[validEnd + 1]);
                    validSeqLen_ = validSeqEnd - validSeqStart_;
                }
            } else {
                // 所有 batch 都是 padding，设置为 0
                validBatchCount_ = 0;
                validSeqLen_ = 0;
            }
        }
    }

    // 检查输入和输出参数
    OP_CHECK_IF(CheckInputParams() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "CausalConv1dFn CheckInputParams FAILED."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckOutputParams() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "CausalConv1dFn CheckOutputParams FAILED."),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// 辅助函数：计算切cu_seq_len时的核间切分信息（考虑因果卷积重叠）
CausalConv1dFnTiling::CuSeqLenSplitInfo CausalConv1dFnTiling::CalculateCuSeqLenSplitInfo(
    uint64_t cuSeqLen, uint64_t bsOverlap) const
{
    CuSeqLenSplitInfo info;

    info.effectiveTotal = cuSeqLen + (totalCoreNum_ - 1) * bsOverlap;
    info.baseLen = info.effectiveTotal / totalCoreNum_;
    info.remainder = info.effectiveTotal % totalCoreNum_;

    // blockFactor: 整核的处理长度（前 remainder 个核）
    info.blockFactor = info.baseLen + (info.remainder > 0 ? 1 : 0);

    // 通过累积输出计算实际核数和尾核长度
    uint64_t accumulated_output = 0;
    info.realCoreNum = 0;
    info.blockTailFactor = 0;

    for (uint64_t core_id = 0; core_id < totalCoreNum_; ++core_id) {
        uint64_t core_load_length = info.baseLen + (core_id < info.remainder ? 1 : 0);

        // 这个核能产生的实际输出（去除与前面核的重叠）
        uint64_t core_output;
        if (core_id == 0) {
            core_output = core_load_length;
        } else {
            // 后续核：载入量 - 重叠部分 = 输出
            core_output = core_load_length - bsOverlap;
        }

        accumulated_output += core_output;

        if (accumulated_output >= cuSeqLen) {
            // 这个核已经能覆盖所有数据
            info.realCoreNum = core_id + 1;

            // 计算尾核实际需要的载入长度
            uint64_t excess = accumulated_output - cuSeqLen;
            info.blockTailFactor = core_load_length - excess;
            break;
        }
    }

    // 如果循环结束还没break，说明所有核都需要
    if (info.realCoreNum == 0) {
        info.realCoreNum = totalCoreNum_;
        info.blockTailFactor = info.baseLen + (totalCoreNum_ - 1 < info.remainder ? 1 : 0);
    }

    return info;
}

// 计算切cu_seq_len时的tiling
ge::graphStatus CausalConv1dFnTiling::CalculateCuSeqLenTiling()
{
    blockIndex_ = 0; // 切cu_seq_len
    uint64_t bsOverlap = kernelWidth_ - 1;

    // 步骤1: 核间切分策略（均分带重叠的总长度）
    //
    // 核心思想：
    // 1. 计算带重叠的总长度：effective_total = total_data + (num_cores - 1) * overlap
    // 2. 均分这个总长度到所有核
    // 3. 每个核的起始位置 = 前一个核的结束位置 - overlap
    //
    // 示例（total_data=13, overlap=2, num_cores=3）：
    //   effective_total = 13 + (3-1)*2 = 17
    //   base_len = 17/3 = 5, remainder = 17%3 = 2
    //   核0: start=0, length=6(5+1), end=6
    //   核1: start=4(6-2), length=6(5+1), end=10
    //   核2: start=8(10-2), length=5, end=13(强制)
    //
    // 注意：使用 validBatchCount_ 而不是 batch_ 进行核间切分

    // 检查是否已经在 DoOpTiling 中计算过
    CuSeqLenSplitInfo splitInfo;
    if (hasCachedSplitInfo_) {
        // 使用缓存的结果
        splitInfo = cachedSplitInfo_;
    } else {
        // 第一次计算，使用有效序列长度
        splitInfo = CalculateCuSeqLenSplitInfo(validSeqLen_, bsOverlap);
    }

    // 使用计算结果
    blockFactor_ = splitInfo.blockFactor;
    blockTailFactor_ = static_cast<uint32_t>(splitInfo.blockTailFactor);
    realCoreNum_ = splitInfo.realCoreNum;

    // 步骤2: 核内UB切分参数
    // 根据需求文档6节Buffer设计,计算UB使用
    // weightInQueue: k * 128 * xDtypeSize_, BUF_NUM=1
    // cacheQueue: (k-1) * 128 * xDtypeSize_, BUF_NUM=1
    // startLocInQueue: 256 * sizeof(int32), BUF_NUM=1
    // indicesInQueue: 256 * sizeof(int32), BUF_NUM=1
    // hasInitialInQueue: 256 * sizeof(int32), BUF_NUM=1
    // xQueue(y复用): 剩余UB, BUF_NUM=2

    uint64_t fixedUbUsage = kernelWidth_ * 128 * xDtypeSize_ +           // weightInQueue
                            (kernelWidth_ - 1) * 128 * xDtypeSize_ +      // cacheQueue
                            256 * sizeof(int32_t) +                       // startLocInQueue
                            256 * sizeof(int32_t) +                       // indicesInQueue
                            256 * sizeof(int32_t);                        // hasInitialInQueue

    uint64_t availableUb = ubSize_ - fixedUbUsage;

    // 计算AlignElement: 256B对齐的元素个数
    uint64_t alignElement = ALIGN_BYTES / xDtypeSize_;

    // xQueue使用双buffer,每个buffer可以存储的元素个数
    // 每次处理 n * alignElement 个元素
    // ubFactor: n * alignElement * xDtypeSize * DOUBLE_BUFFER_NUM <= availableUb
    uint64_t maxN = availableUb / (alignElement * xDtypeSize_ * DOUBLE_BUFFER_NUM);

    if (maxN == 0) {
        OP_LOGE(context_->GetNodeName(), "UB size is not enough for tiling");
        return ge::GRAPH_FAILED;
    }

    // 优先保证dim=256B,尽可能往BS方向切
    // 但如果blockFactor_小于maxN，说明单个核不需要那么大的buffer
    if (blockFactor_ <= maxN) {
        ubFactorBS_ = static_cast<uint32_t>(blockFactor_);
    } else {
        ubFactorBS_ = static_cast<uint32_t>(maxN);
    }
    ubFactorDim_ = static_cast<uint32_t>(alignElement);

    // 如果ubFactorBS * alignElement占不满UB,扩展dim
    uint64_t currentUbUsage = ubFactorBS_ * alignElement * xDtypeSize_ * DOUBLE_BUFFER_NUM;
    if (currentUbUsage < availableUb && dim_ > alignElement) {
        // 以256B为粒度扩展dim
        uint64_t remainingUb = availableUb - currentUbUsage;
        uint64_t additionalDimBlocks = remainingUb / (ubFactorBS_ * xDtypeSize_ * DOUBLE_BUFFER_NUM * alignElement);
        ubFactorDim_ += static_cast<uint32_t>(additionalDimBlocks * alignElement);
        ubFactorDim_ = std::min(static_cast<uint64_t>(ubFactorDim_), dim_);
        // 向下对齐到alignElement
        ubFactorDim_ = (ubFactorDim_ / alignElement) * alignElement;
    }

    // 步骤3: 计算整核和尾核的循环次数及最后一次循环载入大小
    //
    // 注意：blockFactor_ 和 blockTailFactor_ 表示每个核的处理长度（包含重叠部分）
    // 这个长度是该核实际需要载入的数据量

    // 计算整核需要多少次循环
    uint64_t coreLoopsNeeded;
    if (blockFactor_ <= ubFactorBS_) {
        coreLoopsNeeded = 1;
    } else {
        uint64_t remaining = blockFactor_ - ubFactorBS_;
        uint64_t subsequentLoops = Ops::Base::CeilDiv(remaining, ubFactorBS_ - bsOverlap);
        coreLoopsNeeded = 1 + subsequentLoops;
    }
    loopNumBS_ = static_cast<uint32_t>(coreLoopsNeeded);

    // 计算整核最后一次循环载入大小
    uint64_t coreLastLoopInput = blockFactor_ - (coreLoopsNeeded - 1) * (ubFactorBS_ - bsOverlap);
    ubTailFactorBS_ = static_cast<uint32_t>(std::min(coreLastLoopInput, static_cast<uint64_t>(ubFactorBS_)));

    // 计算尾核需要多少次循环
    uint64_t tailLoopsNeeded;
    if (blockTailFactor_ <= ubFactorBS_) {
        tailLoopsNeeded = 1;
    } else {
        uint64_t remaining = blockTailFactor_ - ubFactorBS_;
        uint64_t subsequentLoops = Ops::Base::CeilDiv(remaining, ubFactorBS_ - bsOverlap);
        tailLoopsNeeded = 1 + subsequentLoops;
    }
    tailBlockloopNumBS_ = static_cast<uint32_t>(tailLoopsNeeded);

    // 计算尾核最后一次循环载入大小
    uint64_t tailLastLoopInput = blockTailFactor_ - (tailLoopsNeeded - 1) * (ubFactorBS_ - bsOverlap);
    tailBlockubTailFactorBS_ = static_cast<uint32_t>(std::min(tailLastLoopInput, static_cast<uint64_t>(ubFactorBS_)));

    // 步骤4: Dim方向参数
    loopNumDim_ = static_cast<uint32_t>(Ops::Base::CeilDiv(dim_, static_cast<uint64_t>(ubFactorDim_)));
    ubTailFactorDim_ = static_cast<uint32_t>(dim_ - (loopNumDim_ - 1) * ubFactorDim_);

    // 步骤5: 尾核其他参数
    tailBlockubFactorBS_ = ubFactorBS_;
    tailBlockloopNumDim_ = loopNumDim_;
    tailBlockubFactorDim_ = ubFactorDim_;
    tailBlockubTailFactorDim_ = ubTailFactorDim_;

    return ge::GRAPH_SUCCESS;
}

// 计算切dim时的tiling
ge::graphStatus CausalConv1dFnTiling::CalculateDimTiling()
{
    blockIndex_ = 1; // 切dim
    uint64_t bsOverlap = kernelWidth_ - 1;

    // 步骤1: 核间切分 - dim 维度均分到所有核
    // 注意：dim 切分时，核间没有重叠（不像 cu_seq_len 有因果重叠）

    // 计算每个核分到的 dim 大小（向上取整）
    uint64_t dim_per_core = Ops::Base::CeilDiv(dim_, totalCoreNum_);

    // 计算实际需要的核数
    realCoreNum_ = Ops::Base::CeilDiv(dim_, dim_per_core);

    // blockFactor_: 整核处理的 dim 大小
    blockFactor_ = dim_per_core;

    // blockTailFactor_: 尾核处理的 dim 大小
    blockTailFactor_ = dim_ - (realCoreNum_ - 1) * blockFactor_;

    // 步骤2: 核内UB切分参数
    // 根据需求文档6节Buffer设计,计算UB使用
    // weightInQueue: k * 128 * xDtypeSize_, BUF_NUM=1
    // cacheQueue: (k-1) * 128 * xDtypeSize_, BUF_NUM=1
    // startLocInQueue: 256 * sizeof(int32), BUF_NUM=1
    // indicesInQueue: 256 * sizeof(int32), BUF_NUM=1
    // hasInitialInQueue: 256 * sizeof(int32), BUF_NUM=1
    // xQueue(y复用): 剩余UB, BUF_NUM=2

    uint64_t fixedUbUsage = kernelWidth_ * 128 * xDtypeSize_ +           // weightInQueue
                            (kernelWidth_ - 1) * 128 * xDtypeSize_ +      // cacheQueue
                            256 * sizeof(int32_t) +                       // startLocInQueue
                            256 * sizeof(int32_t) +                       // indicesInQueue
                            256 * sizeof(int32_t);                        // hasInitialInQueue

    uint64_t availableUb = ubSize_ - fixedUbUsage;

    // 计算AlignElement: 256B对齐的元素个数
    uint64_t alignElement = ALIGN_BYTES / xDtypeSize_;

    // xQueue使用双buffer,每个buffer可以存储的元素个数
    // 切dim时，优先保证能容纳尽可能多的 BS，然后再切 dim
    // 数据布局：[BS, dim]
    // 每次处理 ubFactorBS * ubFactorDim 个元素
    // 注意：使用 validSeqLen_ 而不是 cuSeqLen_

    // 计算能容纳的最大元素数（考虑双buffer）
    uint64_t maxElements = availableUb / (xDtypeSize_ * DOUBLE_BUFFER_NUM);

    // 优先尝试处理完整的 validSeqLen_
    if (maxElements >= validSeqLen_ * alignElement) {
        // UB 能容纳至少 validSeqLen_ * alignElement 的数据
        ubFactorBS_ = static_cast<uint32_t>(validSeqLen_);

        // 计算能容纳的 dim 大小（必须是 alignElement 的倍数）
        uint64_t maxDim = maxElements / validSeqLen_;
        maxDim = (maxDim / alignElement) * alignElement;  // 向下对齐

        // ubFactorDim_ 不超过每个核分到的 dim 大小
        ubFactorDim_ = static_cast<uint32_t>(std::min(maxDim, blockFactor_));

    } else {
        // UB 不够容纳完整的 validSeqLen_，需要在 BS 方向切分
        // 优先保证 dim = alignElement
        ubFactorDim_ = static_cast<uint32_t>(alignElement);

        // 计算能容纳的 BS 大小
        uint64_t maxBS = maxElements / alignElement;
        ubFactorBS_ = static_cast<uint32_t>(maxBS);

        if (ubFactorBS_ == 0) {
            OP_LOGE(context_->GetNodeName(), "UB size is not enough for tiling in dim mode");
            return ge::GRAPH_FAILED;
        }

        // 检查 ubFactorBS_ 是否大于重叠长度
        if (ubFactorBS_ <= bsOverlap) {
            OP_LOGE(context_->GetNodeName(),
                    "ubFactorBS_=%u is too small in dim mode, must be > kernelWidth-1=%lu",
                    ubFactorBS_, bsOverlap);
            return ge::GRAPH_FAILED;
        }
    }

    // 步骤3: 计算 BS 方向循环次数（考虑重叠）
    // BS 方向有因果卷积重叠，需要按照 CalculateCuSeqLenTiling 的方式计算

    uint64_t bsLoopsNeeded;
    if (validSeqLen_ <= ubFactorBS_) {
        // 一次循环就能处理完
        bsLoopsNeeded = 1;
        ubTailFactorBS_ = static_cast<uint32_t>(validSeqLen_);
    } else {
        // 计算需要多少次循环
        uint64_t remaining = validSeqLen_ - ubFactorBS_;
        uint64_t subsequentLoops = Ops::Base::CeilDiv(remaining, ubFactorBS_ - bsOverlap);
        bsLoopsNeeded = 1 + subsequentLoops;

        // 计算最后一次循环载入大小
        uint64_t lastLoopInput = validSeqLen_ - (bsLoopsNeeded - 1) * (ubFactorBS_ - bsOverlap);
        ubTailFactorBS_ = static_cast<uint32_t>(std::min(lastLoopInput, static_cast<uint64_t>(ubFactorBS_)));
    }
    loopNumBS_ = static_cast<uint32_t>(bsLoopsNeeded);

    // 步骤4: 计算 Dim 方向循环次数（整核）
    loopNumDim_ = static_cast<uint32_t>(Ops::Base::CeilDiv(blockFactor_, static_cast<uint64_t>(ubFactorDim_)));
    ubTailFactorDim_ = static_cast<uint32_t>(blockFactor_ - (loopNumDim_ - 1) * ubFactorDim_);

    // 步骤5: 计算尾核的 dim 方向循环次数
    tailBlockloopNumDim_ = static_cast<uint32_t>(Ops::Base::CeilDiv(blockTailFactor_, static_cast<uint64_t>(ubFactorDim_)));
    tailBlockubTailFactorDim_ = static_cast<uint32_t>(blockTailFactor_ - (tailBlockloopNumDim_ - 1) * ubFactorDim_);

    // 步骤6: 尾核其他参数
    tailBlockloopNumBS_ = loopNumBS_;
    tailBlockubFactorBS_ = ubFactorBS_;
    tailBlockubTailFactorBS_ = ubTailFactorBS_;
    tailBlockubFactorDim_ = ubFactorDim_;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CausalConv1dFnTiling::DoOpTiling()
{
    // 根据需求文档5.1节,决定是切cu_seq_len还是切dim
    // 切dim的条件:
    // 1. 内存容量约束: cu_seq_len * dim * xDtypeSize > ubSize / 2
    // 2. 核数限制: 切cu_seq_len时开核数 < 32
    // 3. 每个核处理的dim长度 > 256B
    //
    // 注意：使用 validSeqLen_ 而不是 cuSeqLen_ 进行判断

    bool shouldSplitDim = false;
    uint64_t bsOverlap = kernelWidth_ - 1;

    // 条件1: 内存容量约束（使用有效序列长度）
    uint64_t totalMemory = validSeqLen_ * dim_ * xDtypeSize_;
    bool memoryCondition = (totalMemory > ubSize_ / 2);

    // 条件2: 核数限制（考虑因果卷积重叠，使用有效序列长度）
    // 计算并缓存切分信息，避免在 CalculateCuSeqLenTiling 中重复计算
    cachedSplitInfo_ = CalculateCuSeqLenSplitInfo(validSeqLen_, bsOverlap);
    hasCachedSplitInfo_ = true;

    uint64_t coreNumForCuSeqLen = cachedSplitInfo_.realCoreNum;
    bool coreNumCondition = (coreNumForCuSeqLen < MIN_CORE_NUM_FOR_DIM_SPLIT);

    // 条件3: 每个核处理的dim长度
    uint64_t dimPerCore = Ops::Base::CeilDiv(dim_, totalCoreNum_);
    bool dimPerCoreCondition = (dimPerCore * xDtypeSize_ > MIN_DIM_PER_CORE);

    shouldSplitDim = memoryCondition && coreNumCondition && dimPerCoreCondition;

    if (shouldSplitDim) {
        return CalculateDimTiling();
    } else {
        return CalculateCuSeqLenTiling();
    }
}

uint64_t CausalConv1dFnTiling::GetTilingKey() const
{
    return TILING_KEY_VALUE;
}

ge::graphStatus CausalConv1dFnTiling::GetWorkspaceSize()
{
    workspaceSize_ = SYS_WORKSPACE_SIZE;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CausalConv1dFnTiling::PostTiling()
{
    auto workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = workspaceSize_;

    context_->SetBlockDim(realCoreNum_);

    // 设置tiling数据
    tilingData_.set_loopNumBS(loopNumBS_);
    tilingData_.set_loopNumDim(loopNumDim_);
    tilingData_.set_ubFactorBS(ubFactorBS_);
    tilingData_.set_ubTailFactorBS(ubTailFactorBS_);
    tilingData_.set_ubFactorDim(ubFactorDim_);
    tilingData_.set_ubTailFactorDim(ubTailFactorDim_);
    tilingData_.set_blockFactor(blockFactor_);
    tilingData_.set_blockIndex(blockIndex_);
    tilingData_.set_blockTailFactor(blockTailFactor_);
    tilingData_.set_tailBlockloopNumBS(tailBlockloopNumBS_);
    tilingData_.set_tailBlockloopNumDim(tailBlockloopNumDim_);
    tilingData_.set_tailBlockubFactorBS(tailBlockubFactorBS_);
    tilingData_.set_tailBlockubTailFactorBS(tailBlockubTailFactorBS_);
    tilingData_.set_tailBlockubFactorDim(tailBlockubFactorDim_);
    tilingData_.set_tailBlockubTailFactorDim(tailBlockubTailFactorDim_);
    tilingData_.set_realCoreNum(static_cast<uint32_t>(realCoreNum_));
    tilingData_.set_kernelWidth(kernelWidth_);
    tilingData_.set_cuSeqLen(cuSeqLen_);
    tilingData_.set_dim(dim_);
    tilingData_.set_batch(batch_);
    tilingData_.set_validBatchStart(validBatchStart_);
    tilingData_.set_validBatchCount(validBatchCount_);
    tilingData_.set_validSeqStart(validSeqStart_);
    tilingData_.set_validSeqLen(validSeqLen_);

    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());

    return ge::GRAPH_SUCCESS;
}

void CausalConv1dFnTiling::DumpTilingInfo()
{
    std::ostringstream info;
    info << "cuSeqLen: " << cuSeqLen_ << std::endl;
    info << "dim: " << dim_ << std::endl;
    info << "kernelWidth: " << kernelWidth_ << std::endl;
    info << "batch: " << batch_ << std::endl;
    info << "padSlotId: " << padSlotId_ << std::endl;
    info << "validBatchStart: " << validBatchStart_ << std::endl;
    info << "validBatchCount: " << validBatchCount_ << std::endl;
    info << "validSeqStart: " << validSeqStart_ << std::endl;
    info << "validSeqLen: " << validSeqLen_ << std::endl;
    info << "blockIndex: " << blockIndex_ << std::endl;
    info << "blockFactor: " << blockFactor_ << std::endl;
    info << "blockTailFactor: " << blockTailFactor_ << std::endl;
    info << "realCoreNum: " << realCoreNum_ << std::endl;
    info << "loopNumBS: " << loopNumBS_ << std::endl;
    info << "loopNumDim: " << loopNumDim_ << std::endl;
    info << "ubFactorBS: " << ubFactorBS_ << std::endl;
    info << "ubTailFactorBS: " << ubTailFactorBS_ << std::endl;
    info << "ubFactorDim: " << ubFactorDim_ << std::endl;
    info << "ubTailFactorDim: " << ubTailFactorDim_ << std::endl;
    info << "tailBlockloopNumBS: " << tailBlockloopNumBS_ << std::endl;
    info << "tailBlockloopNumDim: " << tailBlockloopNumDim_ << std::endl;
    info << "tailBlockubFactorBS: " << tailBlockubFactorBS_ << std::endl;
    info << "tailBlockubTailFactorBS: " << tailBlockubTailFactorBS_ << std::endl;
    info << "tailBlockubFactorDim: " << tailBlockubFactorDim_ << std::endl;
    info << "tailBlockubTailFactorDim: " << tailBlockubTailFactorDim_ << std::endl;

    OP_LOGI(context_->GetNodeName(), "%s", info.str().c_str());
}

REGISTER_OPS_TILING_TEMPLATE(CausalConv1dFn, CausalConv1dFnTiling, 1);
} // namespace optiling
