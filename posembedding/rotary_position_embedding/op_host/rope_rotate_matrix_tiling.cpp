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
 * \file rope_rotate_tiling.cc
 * \brief
 */
#include "rope_rotate_matrix_tiling.h"
#include "rotary_position_embedding_tiling.h"
#include "log/log.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"

namespace {
const uint64_t INDEX_INPUT_X = 0;
const uint64_t INDEX_INPUT_COS = 1;
const uint64_t INDEX_INPUT_SIN = 2;
const uint64_t INDEX_INPUT_ROTATE = 3;
const uint32_t INDEX_ATTR_MODE = 0;
const uint64_t BYTE_PER_DATA_4 = 4;
const uint64_t BYTE_PER_DATA_2 = 2;
const uint64_t BYTE_OF_BLOCK = 32;
const uint64_t MODE_ROTATE_INTERLEAVED = 1;
const uint64_t DIM_NUM = 4;
const uint64_t DIM_NUM_TWO = 2;
const uint64_t DIM_FIRST = 0;
const uint64_t DIM_SECOND = 1;
const uint64_t DIM_THIRD = 2;
const uint64_t DIM_FOURTH = 3;
const uint64_t D_LENGTH_LIMIT = 896;               // keep the same  support D length as the grad
const uint64_t BNSD_ALIGNED_BLOCK_S_SCALE = 8;     // empiric value
const uint64_t BNSD_UNALIGNED_BLOCK_BN_SCALE = 2;  // empiric value
const uint64_t BNSD_UNALIGNED_BLOCK_D_LENGTH = 80; // empiric value
const uint64_t TWO = 2;
const uint64_t TILING_KEY_PREFIX = 3000;
const uint64_t TILING_MODE_WEIGHT = 10;

const uint64_t BASE_M = 128;
const uint64_t BASE_N = 128;
const uint64_t BASE_K = 128;
const uint64_t CV_PARALL_NUM = 4;

const uint64_t TILING_MODE_UNKNOWN = 0;
const uint64_t TILING_MODE_BNSD = 1;
const uint64_t TILING_DTYPE_UNKNOWN = 0;
const uint64_t TILING_DTYPE_FP32 = 1;
const uint64_t TILING_DTYPE_FP16 = 2;
const uint64_t TILING_DTYPE_BF16 = 3;

__attribute__((always_inline)) inline uint64_t GetCeilDiv(uint64_t value1, uint64_t value2)
{
    if (value2 == 0) {
        return value2;
    }
    return (value1 + value2 - 1) / value2;
}

__attribute__((always_inline)) inline uint64_t GetDiv(uint64_t value1, uint64_t value2)
{
    if (value2 == 0) {
        return value2;
    }
    return value1 / value2;
}

__attribute__((always_inline)) inline uint64_t GetRem(uint64_t value1, uint64_t value2)
{
    if (value2 == 0) {
        return value2;
    }
    return value1 % value2;
}

__attribute__((always_inline)) inline uint64_t GetBytePerData(uint64_t dtype)
{
    if (dtype == TILING_DTYPE_FP32) {
        return BYTE_PER_DATA_4;
    }
    return BYTE_PER_DATA_2;
}

__attribute__((always_inline)) inline uint64_t GetTilingDtype(const ge::DataType &dtype)
{
    if (dtype == ge::DT_FLOAT) {
        return TILING_DTYPE_FP32;
    }
    if (dtype == ge::DT_FLOAT16) {
        return TILING_DTYPE_FP16;
    }
    if (dtype == ge::DT_BF16) {
        return TILING_DTYPE_BF16;
    }
    return TILING_DTYPE_UNKNOWN;
}

__attribute__((always_inline)) inline uint64_t GetTilingKey(uint64_t tilingMode, uint64_t tilingDtype)
{
    return TILING_KEY_PREFIX + tilingMode * TILING_MODE_WEIGHT + tilingDtype;
}
} // namespace

namespace optiling {
class RotateMatrixTiling {
public:
    explicit RotateMatrixTiling(gert::TilingContext *tilingContext) : context(tilingContext) {};
    uint64_t coreNum = 0;
    uint64_t tilingDtype = TILING_DTYPE_UNKNOWN;
    uint64_t tilingKey = TILING_KEY_PREFIX;
    RotaryPositionEmbeddingTilingData tiling;
    RotateMatrixParams &tilingData_ = tiling.rotateMatrixParams;
    ge::graphStatus DoRotateMatrixTiling();

private:
    uint64_t ubSize;
    uint64_t l1Size;
    uint64_t l0CSize;
    gert::TilingContext *context = nullptr;
    matmul_tiling::MultiCoreMatmulTiling mm_;
    inline void PrintTilingParams();
    inline void GetAlignedInfo(const ge::DataType inputDtype, uint64_t dLength);
    inline void ChooseTilingMode(const gert::Shape &xShape);
    ge::graphStatus CheckShape();
    ge::graphStatus CheckDtype();
    ge::graphStatus MatmulTilingProcess();
    ge::graphStatus CheckHalfShape();
    ge::graphStatus CheckShapeSupport(const gert::Shape &xShape, const gert::Shape &cosShape,
                                      const gert::Shape &sinShape, const gert::Shape &rotateShape, uint64_t dLength);
};

inline void RotateMatrixTiling::PrintTilingParams()
{
    auto nodeName = context->GetNodeName();
    OP_LOGD(nodeName, ">>>>>>>>>>>>>>> Start to print Rotate tiling data <<<<<<<<<<<<<<<<");
    OP_LOGD(nodeName, ">>> tilingKey:               %ld", tilingKey);
    OP_LOGD(nodeName, ">>> coreNum:                 %ld", coreNum);
    OP_LOGD(nodeName, ">>> tilingMode:              %ld", tilingData_.get_tilingMode());
    OP_LOGD(nodeName, ">>> tilingDtype:             %ld", tilingDtype);
    OP_LOGD(nodeName, ">>> gmLength:                %ld", tilingData_.get_gmLength());
    OP_LOGD(nodeName, ">>> broadcastFirstDim:       %ld", tilingData_.get_broadcastFirstDim());
    OP_LOGD(nodeName, ">>> broadcastSecondDim:      %ld", tilingData_.get_broadcastSecondDim());
    OP_LOGD(nodeName, ">>> dLength:                 %ld", tilingData_.get_dLength());
    OP_LOGD(nodeName, ">>> totalSLines:             %ld", tilingData_.get_totalSLines());
    OP_LOGD(nodeName, ">>> bn:                      %ld", tilingData_.get_bn());
}

inline void RotateMatrixTiling::ChooseTilingMode(const gert::Shape &xShape)
{   
    uint64_t xFirstDim = 0;
    uint64_t xSecondDim = 0;
    uint64_t xThirdDim = 0;
        
    xFirstDim = static_cast<uint64_t>(xShape.GetDim(DIM_FIRST));
    xSecondDim = static_cast<uint64_t>(xShape.GetDim(DIM_SECOND));
    xThirdDim = static_cast<uint64_t>(xShape.GetDim(DIM_THIRD));

    OP_LOGD(context, "Rotate layout: BNSD");
    tilingData_.set_tilingMode(TILING_MODE_BNSD);
    tilingData_.set_broadcastFirstDim(xFirstDim);
    tilingData_.set_broadcastSecondDim(xSecondDim);
    tilingData_.set_totalSLines(xThirdDim);
}

/* Check input shape */
ge::graphStatus RotateMatrixTiling::CheckShapeSupport(const gert::Shape &xShape, const gert::Shape &cosShape, const gert::Shape &sinShape,
                                                const gert::Shape &rotateShape, uint64_t dLength)
{   
    OP_CHECK_IF(xShape.GetDimNum() != DIM_NUM || cosShape.GetDimNum() != DIM_NUM || sinShape.GetDimNum() != DIM_NUM,
                OP_LOGE(context, "the x, cos, and sin shape must be 4-dimensional."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(rotateShape.GetDimNum() != DIM_NUM_TWO,
                OP_LOGE(context, "the rotate shape must be 2-dimensional."), return ge::GRAPH_FAILED);

    OP_CHECK_IF(dLength > D_LENGTH_LIMIT,
                OP_LOGE(context, "input last dim (head_dim) should be less than %lu.", D_LENGTH_LIMIT),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(GetRem(dLength, TWO) != 0, OP_LOGE(context, "input last dim (head_dim) must be an even number."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(cosShape != sinShape, OP_LOGE(context, "cos shape and sin shape should be equal."),
                return ge::GRAPH_FAILED);
    
    uint64_t cosDLength = static_cast<uint64_t>(cosShape.GetDim(DIM_FOURTH));
    uint64_t rotateFirstLength =  static_cast<uint64_t>(rotateShape.GetDim(DIM_FIRST));
    uint64_t rotateSecondLength =  static_cast<uint64_t>(rotateShape.GetDim(DIM_SECOND));
    OP_CHECK_IF(rotateFirstLength != rotateSecondLength,
                OP_LOGE(context, "the rotate shape 0 should be equal to dim 1, but get [%lu] [%lu].", 
                rotateFirstLength, rotateSecondLength), return ge::GRAPH_FAILED);
    OP_CHECK_IF(cosDLength != dLength || rotateSecondLength != dLength,
                OP_LOGE(context,
                        "input last dim (head_dim) should be equal, but get x [%lu], cos [%lu] and rotate [%lu].",
                        dLength, cosDLength, rotateSecondLength), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

/** 
 * Check do not use Rope:
 * 1. mode=half, soc=ASCEND910, and D is unaligned
 * 2. mode=half and layout=BNSD
 *  2.1. if D is aligned: check B * N > 8 * S --> return true
 *  2.2. if D is unaligned: check 2 * B * N > S / coreNum && D < 80 --> return true
 * else false
 *
 * **Note: The above judgment conditions are empirical values obtained through testing**
 */
ge::graphStatus RotateMatrixTiling::CheckHalfShape(){
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    if (*(attrs->GetAttrPointer<uint32_t>(INDEX_ATTR_MODE)) == MODE_ROTATE_INTERLEAVED){
        return ge::GRAPH_SUCCESS;
    }

    uint64_t dLength = tilingData_.get_dLength();
    uint64_t halfDLength = GetDiv(dLength, TWO);
    uint64_t bytePerData = GetBytePerData(tilingDtype);
    uint64_t dataEachBlock = GetDiv(BYTE_OF_BLOCK, bytePerData);
    bool alignFlag = GetRem(halfDLength, dataEachBlock) == 0;
    const auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    OP_CHECK_IF(!alignFlag && ascendcPlatform.GetSocVersion() == platform_ascendc::SocVersion::ASCEND910,
                OP_LOGE(context, "current soc does not support non aligned calculations."), return ge::GRAPH_FAILED);
    
    if (tilingData_.get_tilingMode() != TILING_MODE_BNSD){
        return ge::GRAPH_SUCCESS;
    }
    int64_t broadcastDim = tilingData_.get_broadcastFirstDim() * tilingData_.get_broadcastSecondDim();
    int64_t unalignedBroadcastTime = broadcastDim * BNSD_UNALIGNED_BLOCK_BN_SCALE;
    int64_t alignedSThreshold = tilingData_.get_totalSLines() * BNSD_ALIGNED_BLOCK_S_SCALE;
    int64_t eachCoreSLines = GetCeilDiv(tilingData_.get_totalSLines(), coreNum);

    bool alignShapeFlag = alignFlag && broadcastDim > alignedSThreshold;
    bool unAlignShapeFlag = !alignFlag && unalignedBroadcastTime > eachCoreSLines &&
                           dLength < BNSD_UNALIGNED_BLOCK_D_LENGTH;
    if (alignShapeFlag || unAlignShapeFlag) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RotateMatrixTiling::CheckShape(){
    // get x and r shape. layout: [..., head_dim]
    auto inputXShapePtr = context->GetInputShape(INDEX_INPUT_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputXShapePtr);
    const gert::Shape &xShape = inputXShapePtr->GetStorageShape();
    
    auto inputCosShapePtr = context->GetInputShape(INDEX_INPUT_COS);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputCosShapePtr);
    const gert::Shape &cosShape = inputCosShapePtr->GetStorageShape();
    
    auto inputSinShapePtr = context->GetInputShape(INDEX_INPUT_SIN);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputSinShapePtr);
    const gert::Shape &sinShape = inputSinShapePtr->GetStorageShape();

    auto inputRotateShapePtr = context->GetInputShape(INDEX_INPUT_ROTATE);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputRotateShapePtr);
    const gert::Shape &rotateShape = inputRotateShapePtr->GetStorageShape();
    
    uint64_t gmLength = static_cast<uint64_t>(xShape.GetShapeSize());
    uint64_t dLength = static_cast<uint64_t>(xShape.GetDim(DIM_FOURTH));
    tilingData_.set_gmLength(gmLength);
    tilingData_.set_dLength(dLength);
    auto shapeCheckRes = CheckShapeSupport(xShape, cosShape, sinShape, rotateShape, dLength);
    ChooseTilingMode(xShape);
    return shapeCheckRes;
}

ge::graphStatus RotateMatrixTiling::CheckDtype(){
    // check input data type
    auto inputInfoPtr = context->GetInputDesc(INDEX_INPUT_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputInfoPtr);
    auto cosInfoPtr = context->GetInputDesc(INDEX_INPUT_COS);
    OP_CHECK_NULL_WITH_CONTEXT(context, cosInfoPtr);
    auto sinInfoPtr = context->GetInputDesc(INDEX_INPUT_SIN);
    OP_CHECK_NULL_WITH_CONTEXT(context, sinInfoPtr);
    auto rotateInfoPtr = context->GetInputDesc(INDEX_INPUT_ROTATE);
    OP_CHECK_NULL_WITH_CONTEXT(context, rotateInfoPtr);
    const ge::DataType inputDtype = inputInfoPtr->GetDataType();
    const ge::DataType cosDtype = cosInfoPtr->GetDataType();
    const ge::DataType sinDtype = sinInfoPtr->GetDataType();
    const ge::DataType rotateDtype = rotateInfoPtr->GetDataType();

    OP_CHECK_IF(inputDtype != cosDtype || inputDtype != sinDtype || inputDtype != rotateDtype,
                OP_LOGE(context, "the dtype of input x, cos, sin and rotate must be the same."), return ge::GRAPH_FAILED);

    tilingDtype = GetTilingDtype(inputDtype);
    OP_CHECK_IF((tilingDtype != TILING_DTYPE_BF16),
                OP_LOGE(context, "only supports bfloat16 data type."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RotateMatrixTiling::MatmulTilingProcess(){
    // compute baseMNK
    uint64_t baseM = tilingData_.get_totalSLines() < BASE_M ? tilingData_.get_totalSLines() : BASE_M;
    uint64_t baseN = tilingData_.get_dLength() < BASE_N ? tilingData_.get_dLength() : BASE_N;
    uint64_t baseK = tilingData_.get_dLength() < BASE_K ? tilingData_.get_dLength() : BASE_K;
    // compute loop
    uint64_t bn = tilingData_.get_broadcastFirstDim() * tilingData_.get_broadcastSecondDim();
    uint64_t blockNumM = GetCeilDiv(tilingData_.get_totalSLines(), baseM);
    uint64_t blockNumN = GetCeilDiv(tilingData_.get_dLength(), baseN);
    uint64_t blockNum = blockNumM * blockNumN;
    tilingData_.set_blockNum(blockNum);
    tilingData_.set_blockNumM(blockNumM);
    tilingData_.set_blockNumN(blockNumN);
    tilingData_.set_bn(bn);
    tilingData_.set_baseM(baseM);
    tilingData_.set_baseN(baseN);
    tilingData_.set_baseK(baseK);
    tilingData_.set_cvParallNum(CV_PARALL_NUM);
    // init matmul
    mm_.SetBufferSpace(l1Size, l0CSize, ubSize);
    mm_.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_BFLOAT16, false);
    mm_.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_BFLOAT16, false);
    mm_.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    mm_.SetBias(false);
    mm_.SetDim(1);
    mm_.SetShape(tilingData_.get_totalSLines(), tilingData_.get_dLength(), tilingData_.get_dLength());
    mm_.SetOrgShape(tilingData_.get_totalSLines(), tilingData_.get_dLength(), tilingData_.get_dLength());
    mm_.SetFixSplit(baseM, baseN, baseK);
    if (mm_.GetTiling(tilingData_.matmulTiling) == -1) {
        OP_LOGE(context->GetNodeName(), "RotaryPositionEmbedding Get Tiling Failed!");
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context->GetNodeName(), "ROPE_tiling: baseM is %d, baseK is %d, baseN is %d.", baseM, baseK, baseN);
    tilingData_.matmulTiling.set_dbL0C(TWO);
    tilingData_.matmulTiling.set_stepKa(TWO);
    tilingData_.matmulTiling.set_stepKb(TWO);
    tilingData_.matmulTiling.set_depthA1(1);
    tilingData_.matmulTiling.set_depthB1(1);
    tilingData_.matmulTiling.set_stepM(1);
    tilingData_.matmulTiling.set_stepN(1);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RotateMatrixTiling::DoRotateMatrixTiling()
{
    // get chip core num and ub size 
    const auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const auto aicNum = ascendcPlatform.GetCoreNumAic();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, l0CSize);
    coreNum = aicNum;
    tilingData_.set_coreNum(coreNum);
    if (ge::GRAPH_SUCCESS != CheckShape()) {
        OP_LOGE(context, "input shape does not meet the requirements.");
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != CheckDtype()) {
        OP_LOGE(context, "input Dtype does not meet the requirements.");
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != CheckHalfShape()) {
        OP_LOGE(context, "when input mode is half, layout is BNSD layout and  B * N is large or D is not aligned, "
                "please do not use RotaryPositionEmbedding fusion operator.");
        return ge::GRAPH_FAILED;
    }

    MatmulTilingProcess();

    tilingKey = GetTilingKey(tilingData_.get_tilingMode(), tilingDtype);
    PrintTilingParams();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RopeRotateMatrixTilingClass::DoOpTiling()
{
    OP_LOGD(context_->GetNodeName(), "Rotate tiling start.");
    RotateMatrixTiling rotateTiling(context_);

    auto tilingRes = rotateTiling.DoRotateMatrixTiling();
    if (ge::GRAPH_SUCCESS != tilingRes) {
        OP_LOGE(context_->GetNodeName(), "DoRotateMatrixTiling failed.");
        return tilingRes;
    }

    context_->SetTilingKey(rotateTiling.tilingKey);
    context_->SetBlockDim(rotateTiling.coreNum);
    rotateTiling.tiling.SaveToBuffer(context_->GetRawTilingData()->GetData(),
                                         context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(rotateTiling.tiling.GetDataSize());
    
    size_t usrWorkspaceSize = rotateTiling.coreNum * BASE_M * BASE_N * CV_PARALL_NUM * BYTE_PER_DATA_4;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t *currentWorkspace = context_->GetWorkspaceSizes(1);
    currentWorkspace[0] = usrWorkspaceSize + sysWorkspaceSize;

    OP_LOGD(context_->GetNodeName(), "Rotate tiling end.");
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling