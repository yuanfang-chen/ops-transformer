#include "causal_conv1d_update_tiling.h"
#include <cmath>
#include "register/op_def_registry.h"

namespace optiling {

constexpr int64_t ALIGN_SIZE = 256;
constexpr int64_t MIN_DIM = 64;
constexpr int64_t MAX_DIM = 16384;
constexpr int64_t MIN_BATCH = 1;
constexpr int64_t MAX_BATCH = 256;
constexpr int64_t KERNEL_SIZE = 3;

static ge::graphStatus TilingFunc(gert::TilingContext* context) {
    CausalConv1dUpdateTiling tiling(context);
    return tiling.Tiling();
}

ge::graphStatus CausalConv1dUpdateTiling::Tiling() {
    // Get input shapes
    auto xShape = context_->GetInputShape(X_INDEX);
    if (xShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto xOriginShape = xShape->GetOriginShape();
    if (xOriginShape.GetDimNum() != 3) {
        return ge::GRAPH_FAILED;
    }

    batchSize_ = xOriginShape.GetDim(0);
    seqLen_ = xOriginShape.GetDim(1);
    dim_ = xOriginShape.GetDim(2);

    // Get filter shape
    auto filterShape = context_->GetInputShape(FILTER_INDEX);
    if (filterShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto filterOriginShape = filterShape->GetOriginShape();
    if (filterOriginShape.GetDimNum() != 2) {
        return ge::GRAPH_FAILED;
    }
    kernelSize_ = filterOriginShape.GetDim(0);

    // Get cache state shape for validation
    auto cacheStateShape = context_->GetInputShape(CONV_STATE_INDEX);
    if (cacheStateShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto cacheStateOriginShape = cacheStateShape->GetOriginShape();
    if (cacheStateOriginShape.GetDimNum() != 3) {
        return ge::GRAPH_FAILED;
    }

    // Get data types
    xDtype_ = context_->GetInputDesc(X_INDEX)->GetDataType();
    filterDtype_ = context_->GetInputDesc(FILTER_INDEX)->GetDataType();
    cacheStateDtype_ = context_->GetInputDesc(CONV_STATE_INDEX)->GetDataType();
    indicesDtype_ = context_->GetInputDesc(CONV_STATE_INDICES_INDEX)->GetDataType();

    // Get acceptTokenNum dtype if available
    if (context_->GetInputTensor(ACCEPT_TOKEN_NUM_INDEX) != nullptr) {
        acceptTokenDtype_ = context_->GetInputDesc(ACCEPT_TOKEN_NUM_INDEX)->GetDataType();
    }

    // Get dtype size
    if (xDtype_ == ge::DT_FLOAT16 || xDtype_ == ge::DT_BF16) {
        xDtypeSize_ = 2;
    } else {
        return ge::GRAPH_FAILED;
    }

    // Validate shapes and types
    if (!ValidateShapes() || !ValidateDataTypes()) {
        return ge::GRAPH_FAILED;
    }

    // Get pad_slot_id attribute
    pad_slot_id_ = 0;
    if (context_->GetAttrs() != nullptr) {
        context_->GetAttrs()->GetInt(0, pad_slot_id_);
    }

    // Calculate invalid batch number
    inValidBatchNum_ = 0;
    auto convStateIndicesTensor = context_->GetInputTensor(CONV_STATE_INDICES_INDEX);
    if (convStateIndicesTensor != nullptr) {
        const int64_t* dataPtr = convStateIndicesTensor->GetData<int64_t>();
        if (dataPtr != nullptr) {
            for (int64_t i = batchSize_ - 1; i >= 0; i--) {
                if (pad_slot_id_ == dataPtr[i]) {
                    inValidBatchNum_++;
                } else {
                    break;
                }
            }
        }
    }

    // Calculate tiling parameters
    CausalConv1dUpdateTilingData tilingData;
    CalculateTilingParams(tilingData);

    // Set tiling data
    context_->SetTilingData(tilingData);

    return ge::GRAPH_SUCCESS;
}

bool CausalConv1dUpdateTiling::ValidateShapes() {
    // Validate batch size
    if (batchSize_ < MIN_BATCH || batchSize_ > MAX_BATCH) {
        return false;
    }

    // Validate sequence length (m+1, where m in [0,5])
    if (seqLen_ < 1 || seqLen_ > 6) {
        return false;
    }

    // Validate dimension
    if (dim_ < MIN_DIM || dim_ > MAX_DIM) {
        return false;
    }

    // Validate kernel size
    if (kernelSize_ != KERNEL_SIZE) {
        return false;
    }

    // Validate filter shape
    auto filterShape = context_->GetInputShape(FILTER_INDEX);
    auto filterOriginShape = filterShape->GetOriginShape();
    if (filterOriginShape.GetDim(1) != dim_) {
        return false;
    }

    // Validate cache state shape
    auto cacheStateShape = context_->GetInputShape(CONV_STATE_INDEX);
    auto cacheStateOriginShape = cacheStateShape->GetOriginShape();
    int64_t expectedCacheLen = kernelSize_ - 1 + (seqLen_ - 1);
    if (cacheStateOriginShape.GetDim(1) != expectedCacheLen) {
        return false;
    }
    if (cacheStateOriginShape.GetDim(2) != dim_) {
        return false;
    }

    // Validate conv state indices shape
    auto indicesShape = context_->GetInputShape(CONV_STATE_INDICES_INDEX);
    if (indicesShape == nullptr) {
        return false;
    }
    auto indicesOriginShape = indicesShape->GetOriginShape();
    if (indicesOriginShape.GetDimNum() != 1) {
        return false;
    }
    if (indicesOriginShape.GetDim(0) != batchSize_) {
        return false;
    }

    // Validate accept token num shape if available
    if (context_->GetInputTensor(ACCEPT_TOKEN_NUM_INDEX) != nullptr) {
        auto acceptShape = context_->GetInputShape(ACCEPT_TOKEN_NUM_INDEX);
        if (acceptShape == nullptr) {
            return false;
        }
        auto acceptOriginShape = acceptShape->GetOriginShape();
        if (acceptOriginShape.GetDimNum() != 1) {
            return false;
        }
        if (acceptOriginShape.GetDim(0) != batchSize_) {
            return false;
        }
    }

    return true;
}

bool CausalConv1dUpdateTiling::ValidateDataTypes() {
    // Validate x data type
    if (xDtype_ != ge::DT_FLOAT16 && xDtype_ != ge::DT_BF16) {
        return false;
    }

    // Validate filter data type matches x
    if (filterDtype_ != xDtype_) {
        return false;
    }

    // Validate cache state data type matches x
    if (cacheStateDtype_ != xDtype_) {
        return false;
    }

    // Validate indices data type
    if (indicesDtype_ != ge::DT_INT64) {
        return false;
    }

    // Validate accept token num data type if available
    if (context_->GetInputTensor(ACCEPT_TOKEN_NUM_INDEX) != nullptr) {
        if (acceptTokenDtype_ != ge::DT_INT64) {
            return false;
        }
    }

    return true;
}

void CausalConv1dUpdateTiling::CalculateTilingParams(CausalConv1dUpdateTilingData& tilingData) {
    // Calculate valid batch (excluding invalid batches at the end)
    int64_t validBatch = batchSize_ - inValidBatchNum_;

    // Calculate core parameters
    CalculateCoreParams(tilingData, validBatch);

    // Calculate loop parameters for regular blocks
    CalculateLoopParams(tilingData, false);

    // Calculate loop parameters for tail block
    CalculateLoopParams(tilingData, true);

    // Set block dimension
    int64_t TOTAL_CORE_NUM = GetCoreNumAiv();
    int64_t usedCoreNum = (validBatch + tilingData.blockFactor - 1) / tilingData.blockFactor;
    context_->SetBlockDim(usedCoreNum);
}

void CausalConv1dUpdateTiling::CalculateCoreParams(CausalConv1dUpdateTilingData& tilingData, int64_t validBatch) {
    int64_t TOTAL_CORE_NUM = GetCoreNumAiv();

    // Calculate block factor and used core number
    int64_t blockFactor = (validBatch + TOTAL_CORE_NUM - 1) / TOTAL_CORE_NUM;
    int64_t usedCoreNum = (validBatch + blockFactor - 1) / blockFactor;
    int64_t blockTailFactor = validBatch - (usedCoreNum - 1) * blockFactor;

    tilingData.blockFactor = blockFactor;
    tilingData.blockTailFactor = blockTailFactor;
}

void CausalConv1dUpdateTiling::CalculateLoopParams(CausalConv1dUpdateTilingData& tilingData, bool isTailBlock) {
    int64_t UB_SIZE = GetUbBlockSize();
    int64_t currentBlockFactor = isTailBlock ? tilingData.blockTailFactor : tilingData.blockFactor;

    // Calculate fixed buffer sizes
    int64_t filterBufferSize = kernelSize_ * ALIGN_SIZE;
    int64_t cacheBufferSize = (kernelSize_ - 1 + (seqLen_ - 1)) * ALIGN_SIZE;
    int64_t indicesBufferSize = ALIGN_SIZE * sizeof(int64_t);
    int64_t acceptTokenBufferSize = ALIGN_SIZE * sizeof(int64_t);

    // Total fixed buffer size
    int64_t fixedBufferSize = filterBufferSize + cacheBufferSize + indicesBufferSize + acceptTokenBufferSize;

    // Remaining UB size for xQueue (with 2 buffers for double buffering)
    int64_t remainingUBSize = UB_SIZE - fixedBufferSize;
    int64_t xQueueSizePerBuffer = remainingUBSize / 2;

    // Calculate AlignElement (256B alignment for dim)
    int64_t AlignElement = ALIGN_SIZE / xDtypeSize_;

    // Calculate BS (batch * sequence) for current block
    int64_t totalBS = currentBlockFactor * seqLen_;

    // Try to maximize n (number of BS items per loop) while keeping dim at AlignElement
    int64_t elementPerBSItem = AlignElement;
    int64_t bytesPerBSItem = elementPerBSItem * xDtypeSize_;
    int64_t maxBSPerLoop = xQueueSizePerBuffer / bytesPerBSItem;

    if (maxBSPerLoop >= totalBS) {
        // Can fit all BS in one loop
        int64_t ubFactorBS = totalBS;
        int64_t loopNumBS = 1;
        int64_t ubTailFactorBS = totalBS;

        // Try to expand dim if there's remaining space
        int64_t remainingSpace = xQueueSizePerBuffer - totalBS * bytesPerBSItem;
        int64_t additionalElements = remainingSpace / (totalBS * xDtypeSize_);
        additionalElements = (additionalElements / AlignElement) * AlignElement;
        int64_t expandedDim = AlignElement + additionalElements;
        if (expandedDim > dim_) {
            expandedDim = dim_;
        }

        int64_t ubFactorDim = expandedDim;
        int64_t loopNumDim = (dim_ + ubFactorDim - 1) / ubFactorDim;
        int64_t ubTailFactorDim = dim_ - (loopNumDim - 1) * ubFactorDim;

        if (isTailBlock) {
            tilingData.tailBlockloopNumBS = loopNumBS;
            tilingData.tailBlockubFactorBS = ubFactorBS;
            tilingData.tailBlockubTailFactorBS = ubTailFactorBS;
            tilingData.tailBlockloopNumDim = loopNumDim;
            tilingData.tailBlockubFactorDim = ubFactorDim;
            tilingData.tailBlockubTailFactorDim = ubTailFactorDim;
        } else {
            tilingData.loopNumBS = loopNumBS;
            tilingData.ubFactorBS = ubFactorBS;
            tilingData.ubTailFactorBS = ubTailFactorBS;
            tilingData.loopNumDim = loopNumDim;
            tilingData.ubFactorDim = ubFactorDim;
            tilingData.ubTailFactorDim = ubTailFactorDim;
        }
    } else {
        // Need multiple loops for BS dimension
        int64_t ubFactorBS = maxBSPerLoop;
        int64_t loopNumBS = (totalBS + ubFactorBS - 1) / ubFactorBS;
        int64_t ubTailFactorBS = totalBS - (loopNumBS - 1) * ubFactorBS;

        // Dim uses AlignElement size per loop
        int64_t ubFactorDim = AlignElement;
        int64_t loopNumDim = (dim_ + ubFactorDim - 1) / ubFactorDim;
        int64_t ubTailFactorDim = dim_ - (loopNumDim - 1) * ubFactorDim;

        if (isTailBlock) {
            tilingData.tailBlockloopNumBS = loopNumBS;
            tilingData.tailBlockubFactorBS = ubFactorBS;
            tilingData.tailBlockubTailFactorBS = ubTailFactorBS;
            tilingData.tailBlockloopNumDim = loopNumDim;
            tilingData.tailBlockubFactorDim = ubFactorDim;
            tilingData.tailBlockubTailFactorDim = ubTailFactorDim;
        } else {
            tilingData.loopNumBS = loopNumBS;
            tilingData.ubFactorBS = ubFactorBS;
            tilingData.ubTailFactorBS = ubTailFactorBS;
            tilingData.loopNumDim = loopNumDim;
            tilingData.ubFactorDim = ubFactorDim;
            tilingData.ubTailFactorDim = ubTailFactorDim;
        }
    }
}

REGISTER_TILING_DATA_CLASS(CausalConv1dUpdate, CausalConv1dUpdateTiling)

} // namespace optiling
