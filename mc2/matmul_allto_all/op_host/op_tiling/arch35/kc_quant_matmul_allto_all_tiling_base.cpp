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
 * \file kc_quant_matmul_allto_all_tiling_base.cpp
 * \brief
 */
#include "op_mc2.h"
#include "mc2_log.h"
#include "kc_quant_matmul_allto_all_tiling_base.h"

using namespace Mc2Log;
using namespace AscendC;
using namespace Mc2Tiling;

namespace MC2Tiling {

/**
 * @brief 当前量化过程的准入条件
 * @return true
 */
bool KcQuantMatmulAllToAllTilingBase::IsCapable()
{
    QuantMode mode = MatmulAlltoAllTilingUtil::GetQuantMode(context_, opName_);
    if (mode == QuantMode::KC_QUANT) {
        OP_LOGI(opName_, "Start with KcQuantMatmulAllToAll tiling.");
        return true;
    }
    OP_LOGI(opName_, "Skip KcQuantMatmulAllToAll tiling when not KC_QUANT.");
    return false;
}

/**
 * @brief 校验输入信息是否合规:attr,Dtype,shape等，使用通用校验util中的check方法
 *
 * @return ge::graphStatus
 */
ge::graphStatus KcQuantMatmulAllToAllTilingBase::CheckOpInputInfo()
{
    OP_TILING_CHECK(MatmulAlltoAllTilingUtil::CheckAttrsInfo(context_, opName_, MATMUL_ALLTOALL_INDEX_SCHEMA) !=
                        ge::GRAPH_SUCCESS,
                    OP_LOGE(opName_, "Tiling check Attrs failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(MatmulAlltoAllTilingUtil::CheckKcQuantTensorDataType(context_, opName_) != ge::GRAPH_SUCCESS,
                    OP_LOGE(opName_, "tiling check Dtype failed in kc quant matmul all to all."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(MatmulAlltoAllTilingUtil::CheckKcQuantShapeInfo(context_, opName_, MATMUL_ALLTOALL_INDEX_SCHEMA) !=
                        ge::GRAPH_SUCCESS,
                    OP_LOGE(opName_, "Tiling check kc quant shape info failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckKcQuantMatrixMulShapes(context_, opName_) != ge::GRAPH_SUCCESS,
                    OP_LOGE(opName_, "Tiling check kc quant matrix shape failed."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 根据输入设置tiling参数
 *
 * @return ge::graphStatus
 */
ge::graphStatus KcQuantMatmulAllToAllTilingBase::InitTilingContextParameters()
{
    GE_ASSERT_GRAPH_SUCCESS(
        MatmulAlltoAllTilingUtil::SetAttrsInfo(context_, opName_, contextInfo, MATMUL_ALLTOALL_INDEX_SCHEMA));
    GE_ASSERT_GRAPH_SUCCESS(MatmulAlltoAllTilingUtil::SetDataTypeInfo(context_, opName_, contextInfo));
    GE_ASSERT_GRAPH_SUCCESS(MatmulAlltoAllTilingUtil::SetShapeInfo(context_, contextInfo));
    contextInfo.quantMode = QuantMode::KC_QUANT;
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 设置hccl参数；进行通算切分, 获取mm tiling等
 *
 * @return ge::graphStatus
 */
ge::graphStatus KcQuantMatmulAllToAllTilingBase::DoOpTiling()
{
    // 输入参数的校验:Attrs,Dtype,Shape等
    GE_ASSERT_GRAPH_SUCCESS(CheckOpInputInfo());
    // 参数校验通过后赋值给全局上下文变量
    GE_ASSERT_GRAPH_SUCCESS(InitTilingContextParameters());
    // 进行通算切分
    GE_ASSERT_GRAPH_SUCCESS(TileCommAndCompute());
    // 调用量化Matmul的tiling方法进行切分
    GE_ASSERT_GRAPH_SUCCESS(DoKcQuantMMTiling());
    // hccl的tiling参数赋值处理
    GE_ASSERT_GRAPH_SUCCESS(SetHcclTiling());
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 设置hccl的config,进行hccl对应的通信任务设置
 *
 * @return ge::graphStatus
 */
ge::graphStatus KcQuantMatmulAllToAllTilingBase::SetHcclTiling()
{
    OP_TILING_CHECK(mc2tiling::ConvertGeTypeToHcclType(opName_, contextInfo.args_.geCType) ==
                        mc2tiling::HcclDataType::HCCL_DATA_TYPE_RESERVED,
                    VECTOR_INNER_ERR_REPORT_TILING(opName_, "Cannot find HcclDataType according to ge datatype = %d.",
                                                   static_cast<int32_t>(contextInfo.args_.geCType)),
                    return ge::GRAPH_FAILED;);
    Mc2CcTilingConfigBuilder allToAllBuilder =
        Mc2CcTilingConfigBuilder::create(contextInfo.group, mc2tiling::AicpuComType::HCCL_CMD_ALLTOALL,
                                         Mc2CcTilingConfigBuilder::AlgConfigType::ALL_TO_ALL);
    AscendC::Mc2CcTilingConfig allToAllTilingConfig = allToAllBuilder.withCommEngine(mc2tiling::A5_CCU_ENGINE).build();
    if (!allToAllBuilder.isSuccess()) {
        OP_LOGE(opName_, "Build hccl tiling config failed: %s", allToAllBuilder.errorMsg().c_str());
        return ge::GRAPH_FAILED;
    }
    allToAllTilingConfig.GetTiling(localTilingData_.mc2InitTiling);
    allToAllTilingConfig.GetTiling(localTilingData_.mc2CcTiling);
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 进行通算切分之后单个块的MM Tiling
 *
 * @return ge::graphStatus
 */
ge::graphStatus KcQuantMatmulAllToAllTilingBase::DoKcQuantMMTiling()
{
    // 设置MM切前信息
    SetTilingInfo(localTilingData_.kcQuantMatmulAlltoAllTilingInfo);
    mmMvalueLen = inferredInfo.tileM;
    KcQuantMatmulAlltoAllHelper mmTile(*this, localTilingData_.mc2KcQuantMmTileTilingData, mmMvalueLen);
    GE_ASSERT_GRAPH_SUCCESS(mmTile.DoTiling());
    if (inferredInfo.tailCnt == 0) {
        return ge::GRAPH_SUCCESS;
    }
    mmMvalueLen = inferredInfo.tailM;
    KcQuantMatmulAlltoAllHelper mmTail(*this, localTilingData_.mc2KcQuantMmTailTilingData, mmMvalueLen);
    GE_ASSERT_GRAPH_SUCCESS(mmTail.DoTiling());
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 重写获取MM index的信息
 * 由于本算子的context和MM不一样，需要重写获取MM index的一些信息，把我们的context传给Matmul，来达到可以调用MM策略的目的。
 * @return ge::graphStatus
 */
const gert::Shape KcQuantMatmulAlltoAllHelper::GetX1Shape(const size_t index)
{
    (void)index;
    return gert::Shape(
        {static_cast<int64_t>(mmLen), static_cast<int64_t>(tilingProcesser_.contextInfo.args_.kValue)});
}
const gert::Shape KcQuantMatmulAlltoAllHelper::GetX2Shape(const size_t index)
{
    (void)index;
    if (tilingProcesser_.contextInfo.args_.isBTrans) {
        return gert::Shape(
            {static_cast<int64_t>(tilingProcesser_.contextInfo.args_.nValue), static_cast<int64_t>(tilingProcesser_.contextInfo.args_.kValue)});
    }
    return gert::Shape(
        {static_cast<int64_t>(tilingProcesser_.contextInfo.args_.kValue), static_cast<int64_t>(tilingProcesser_.contextInfo.args_.nValue)});
}

const gert::Shape& KcQuantMatmulAlltoAllHelper::GetScaleShape(const size_t index)
{
    (void)index;
    return context_->GetOptionalInputShape(static_cast<size_t>(INPUT_X2_SCALE_INDEX))->GetStorageShape();
}

const gert::StorageShape* KcQuantMatmulAlltoAllHelper::GetPertokenShape(const size_t index)
{
    (void)index;
    return context_->GetOptionalInputShape(static_cast<size_t>(INPUT_X1_SCALE_INDEX));
}

const gert::StorageShape* KcQuantMatmulAlltoAllHelper::GetBiasShape(const size_t index)
{
    (void)index;
    return context_->GetOptionalInputShape(static_cast<size_t>(INPUT_BIAS_INDEX));
}

ge::graphStatus KcQuantMatmulAlltoAllHelper::GetShapeAttrsInfo()
{   
    OP_LOGD(tilingProcesser_.opName_, "Start assemble input params for matmul tiling");
    auto&& tilingArgs = tilingProcesser_.contextInfo.args_;
    inputParams_.opName = tilingProcesser_.opName_;
    inputParams_.transB = tilingArgs.isBTrans;
    inputParams_.hasBias = tilingArgs.isBias;
    inputParams_.libApiWorkSpaceSize = tilingProcesser_.libApiWorkSpaceSize_;
    inputParams_.mSize = tilingArgs.mValue;
    inputParams_.kSize = tilingArgs.kValue;
    inputParams_.nSize = tilingArgs.nValue;
    inputParams_.aDtype = tilingArgs.geAType;
    inputParams_.bDtype = tilingArgs.geBType;
    int yDType = *context_->GetAttrs()->GetAttrPointer<uint64_t>(ATTR_Y_DTYPE_INDEX);
    auto scaleTensorDesc = context_->GetOptionalInputDesc(INPUT_X2_SCALE_INDEX);
    auto perTokenScaleTensorDesc = context_->GetOptionalInputDesc(INPUT_X1_SCALE_INDEX);
    OP_TILING_CHECK((scaleTensorDesc == nullptr),
                    VECTOR_INNER_ERR_REPORT_TILING(tilingProcesser_.opName_, "the scale tensor is invalid"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK((perTokenScaleTensorDesc == nullptr),
                    VECTOR_INNER_ERR_REPORT_TILING(tilingProcesser_.opName_, "the perToken scale tensor is invalid"),
                    return ge::GRAPH_FAILED);
    inputParams_.scaleDtype = scaleTensorDesc->GetDataType();
    inputParams_.perTokenScaleDtype = perTokenScaleTensorDesc->GetDataType();
    inputParams_.cDtype = static_cast<ge::DataType>(yDType);
    inputParams_.outDtype = static_cast<int64_t>(yDType);
    OP_LOGD(tilingProcesser_.opName_, "yDType is %ld", inputParams_.outDtype);
    inputParams_.biasDtype = tilingArgs.isBias ? tilingArgs.geBiasType : ge::DT_INT32;
    inputParams_.isPerChannel = true;
    inputParams_.isDoubleScale = true;
    if (inputParams_.isPerChannel) {
        inputParams_.groupSizeM = 1;
        inputParams_.groupSizeN = 1;
    }
    GE_ASSERT_TRUE(AnalyzeInputs());
    PrintTilingInputParam(inputParams_);
    return ge::GRAPH_SUCCESS;
}

void KcQuantMatmulAlltoAllHelper::PrintTilingInputParam(Mc2QuantBatchMatmulInfo& quantMatmulInfo)
{
    OP_LOGD(tilingProcesser_.opName_, "mSize_ %ld kSize_ %ld nSize_ %ld libApiWorkSpaceSize %u",
            quantMatmulInfo.mSize, quantMatmulInfo.kSize, quantMatmulInfo.nSize,
            quantMatmulInfo.libApiWorkSpaceSize);
    OP_LOGD(tilingProcesser_.opName_,
            "aDtype_ %d bDtype_ %d cDtype_ %d biasDtype_ %d outDtype %ld"
            " scaleDtype %d perTokenScaleDtype %d",
            static_cast<int32_t>(quantMatmulInfo.aDtype), static_cast<int32_t>(quantMatmulInfo.bDtype),
            static_cast<int32_t>(quantMatmulInfo.cDtype), static_cast<int32_t>(quantMatmulInfo.biasDtype),
            quantMatmulInfo.outDtype, static_cast<int32_t>(quantMatmulInfo.scaleDtype),
            static_cast<int32_t>(quantMatmulInfo.perTokenScaleDtype));
    OP_LOGD(tilingProcesser_.opName_, "Check isPertoken=%d.", static_cast<int32_t>(quantMatmulInfo.isPerChannel));
}

ge::graphStatus KcQuantMatmulAlltoAllHelper::DoLibApiTiling()
{
    GE_ASSERT_GRAPH_SUCCESS(Mc2AdaptiveSlidingWindowTiling::DoLibApiTiling());
    isBf16Opt_ = false;
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 重写友元类PostTiling方法
 * PostTiling主要做的是拷贝tilingdata的活，但是本算子拷贝tilingdata是在大结构体中拷贝，不需要在此处拷贝。
 * @return ge::graphStatus
 */
ge::graphStatus KcQuantMatmulAlltoAllHelper::PostTiling()
{
    tilingProcesser_.workspaceSize_ = std::max(tilingProcesser_.workspaceSize_, workspaceSize_);
    OP_LOGD(tilingProcesser_.opName_, "set mm workspace size %lu to mc2", tilingProcesser_.workspaceSize_);
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 构造函数，创建一个KcQuantMatmulAlltoAllHelper对象
 *
 * @param context
 */
KcQuantMatmulAlltoAllHelper::KcQuantMatmulAlltoAllHelper(KcQuantMatmulAllToAllTilingBase& kcQuantMatmulAllToAllTilingBase, 
                                                     DequantBmm::Mc2QuantBatchMatmulV3TilingDataParams& data, uint64_t& mmMvalueLen)
    : Mc2AdaptiveSlidingWindowTiling(kcQuantMatmulAllToAllTilingBase.context_, &data), tilingProcesser_(kcQuantMatmulAllToAllTilingBase),
    mmLen(mmMvalueLen)
{
}

/**
 * @brief 打印量化matmul tiling的信息
 *
 * @param opName
 * @param tiling
 */
void KcQuantMatmulAllToAllTilingBase::PrintKcQuantMMV3TilingData(const std::string &opName, DequantBmm::Mc2QuantBatchMatmulV3TilingDataParams &tiling)
{
    OP_LOGD(opName, " tiling.matmulTiling.usedCoreNum %d", tiling.matmulTiling.usedCoreNum);
    OP_LOGD(opName, " tiling.matmulTiling.M %d", tiling.matmulTiling.M);
    OP_LOGD(opName, " tiling.matmulTiling.N %d", tiling.matmulTiling.N);
    OP_LOGD(opName, " tiling.matmulTiling.Ka %d", tiling.matmulTiling.Ka);
    OP_LOGD(opName, " tiling.matmulTiling.Kb %d", tiling.matmulTiling.Kb);
    OP_LOGD(opName, " tiling.matmulTiling.singleCoreM %d", tiling.matmulTiling.singleCoreM);
    OP_LOGD(opName, " tiling.matmulTiling.singleCoreK %d", tiling.matmulTiling.singleCoreK);
    OP_LOGD(opName, " tiling.matmulTiling.singleCoreN %d", tiling.matmulTiling.singleCoreN);
    OP_LOGD(opName, " tiling.matmulTiling.baseM %d", tiling.matmulTiling.baseM);
    OP_LOGD(opName, " tiling.matmulTiling.baseN %d", tiling.matmulTiling.baseN);
    OP_LOGD(opName, " tiling.matmulTiling.baseK %d", tiling.matmulTiling.baseK);
    OP_LOGD(opName, " tiling.matmulTiling.depthA1 %d", tiling.matmulTiling.depthA1);
    OP_LOGD(opName, " tiling.matmulTiling.depthB1 %d", tiling.matmulTiling.depthB1);
    OP_LOGD(opName, " tiling.matmulTiling.stepM %d", tiling.matmulTiling.stepM);
    OP_LOGD(opName, " tiling.matmulTiling.stepN %d", tiling.matmulTiling.stepN);
    OP_LOGD(opName, " tiling.matmulTiling.isBias %d", tiling.matmulTiling.isBias);
    OP_LOGD(opName, " tiling.matmulTiling.transLength %d", tiling.matmulTiling.transLength);
    OP_LOGD(opName, " tiling.matmulTiling.iterateOrder %d", tiling.matmulTiling.iterateOrder);
    OP_LOGD(opName, " tiling.matmulTiling.dbL0A %d", tiling.matmulTiling.dbL0A);
    OP_LOGD(opName, " tiling.matmulTiling.dbL0B %d", tiling.matmulTiling.dbL0B);
    OP_LOGD(opName, " tiling.matmulTiling.dbL0C %d", tiling.matmulTiling.dbL0C);
    OP_LOGD(opName, " tiling.matmulTiling.shareMode %d", tiling.matmulTiling.shareMode);
    OP_LOGD(opName, " tiling.matmulTiling.shareL0CSize %d", tiling.matmulTiling.shareL0CSize);
    OP_LOGD(opName, " tiling.matmulTiling.shareL1Size %d", tiling.matmulTiling.shareL1Size);
    OP_LOGD(opName, " tiling.matmulTiling.shareUbSize %d", tiling.matmulTiling.shareUbSize);
    OP_LOGD(opName, " tiling.matmulTiling.batchM %d", tiling.matmulTiling.batchM);
    OP_LOGD(opName, " tiling.matmulTiling.batchN %d", tiling.matmulTiling.batchN);
    OP_LOGD(opName, " tiling.matmulTiling.singleBatchM %d", tiling.matmulTiling.singleBatchM);
    OP_LOGD(opName, " tiling.matmulTiling.singleBatchN %d", tiling.matmulTiling.singleBatchN);
    OP_LOGD(opName, " tiling.tileL2cacheTiling.mTileCntL2 %d", tiling.tileL2cacheTiling.mTileCntL2);
    OP_LOGD(opName, " tiling.tileL2cacheTiling.nTileCntL2 %d", tiling.tileL2cacheTiling.nTileCntL2);
    OP_LOGD(opName, " tiling.tileL2cacheTiling.mTileBlock %d", tiling.tileL2cacheTiling.mTileBlock);
    OP_LOGD(opName, " tiling.tileL2cacheTiling.nTileBlock %d", tiling.tileL2cacheTiling.nTileBlock);
    OP_LOGD(opName, " tiling.tileL2cacheTiling.calOrder %d", tiling.tileL2cacheTiling.calOrder);
    OP_LOGD(opName, " tiling.tileL2cacheTiling.isBasicTiling %d", tiling.tileL2cacheTiling.isBasicTiling);
    OP_LOGD(opName, " tiling.adaptiveSlidingWin.mTailTile %d", tiling.adaptiveSlidingWin.mTailTile);
    OP_LOGD(opName, " tiling.adaptiveSlidingWin.nTailTile %d", tiling.adaptiveSlidingWin.nTailTile);
}

/**
 * @brief 打印tilingInfo信息
 *
 * @param opName
 * @param tilingInfo
 */
void KcQuantMatmulAllToAllTilingBase::PrintKcQuantMatmulAlltoAllTilingInfo(const std::string &opName,
                                                               MatmulAlltoAllTilingInfo &tilingInfo)
{
    OP_LOGD(opName, "tilingInfo.rankDim: %u", tilingInfo.rankDim);
    OP_LOGD(opName, "tilingInfo.tileM: %u", tilingInfo.tileM);
    OP_LOGD(opName, "tilingInfo.tileCnt: %u", tilingInfo.tileCnt);
    OP_LOGD(opName, "tilingInfo.tailM: %u", tilingInfo.tailM);
    OP_LOGD(opName, "tilingInfo.tailCnt: %u", tilingInfo.tailCnt);
    OP_LOGD(opName, "tilingInfo.biasLen: %u", tilingInfo.biasLen);
    OP_LOGD(opName, "tilingInfo.rankM: %u", tilingInfo.rankM);
    OP_LOGD(opName, "tilingInfo.rankN: %u", tilingInfo.rankN);
    OP_LOGD(opName, "tilingInfo.rankK: %u", tilingInfo.rankK);
    OP_LOGD(opName, "tilingInfo.mmResultLen: %u", tilingInfo.mmResultLen);
    OP_LOGD(opName, "tilingInfo.permuteLen: %u", tilingInfo.permuteLen);
    OP_LOGD(opName, "tilingInfo.biasLen: %u", tilingInfo.biasLen);
    OP_LOGD(opName, "tilingInfo.aicCoreNum: %u", tilingInfo.aicCoreNum);
    OP_LOGD(opName, "tilingInfo.hcclDataType: %u", tilingInfo.hcclDataType);
}

/**
 * @brief 打印传递给kernel的tilingData
 *
 * @param outTilingData tilingData参数
 */
void KcQuantMatmulAllToAllTilingBase::PrintKcQuantMatmulAlltoAllTilingData(KcQuantMatmulAlltoAllTilingData &outTilingData)
{
    PrintKcQuantMatmulAlltoAllTilingInfo(opName_, outTilingData.kcQuantMatmulAlltoAllTilingInfo);
    PrintKcQuantMMV3TilingData(opName_, outTilingData.mc2KcQuantMmTileTilingData);
    if (outTilingData.kcQuantMatmulAlltoAllTilingInfo.tailCnt == 0) {
        return;
    }
    OP_LOGD(opName_, "KcQuantMatmulAlltoall has tail");
    PrintKcQuantMMV3TilingData(opName_, outTilingData.mc2KcQuantMmTailTilingData);
}

/**
 * @brief 保存量化tiling数据到context
 *
 * @return ge::graphStatus
 */
ge::graphStatus KcQuantMatmulAllToAllTilingBase::PostTiling()
{
    KcQuantMatmulAlltoAllTilingData *outTilingData = context_->GetTilingData<KcQuantMatmulAlltoAllTilingData>();
    size_t tilingBufCap = context_->GetRawTilingData()->GetCapacity();
    OP_TILING_CHECK((outTilingData == nullptr), OP_LOGE(opName_, "failed to get tiling data from context"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK((tilingBufCap < sizeof(localTilingData_)),
                    OP_LOGE(opName_, "TilingBuffer capacity too small, capacity = %zu, need = %zu.", tilingBufCap,
                            sizeof(localTilingData_)),
                    return ge::GRAPH_FAILED);
    errno_t ret = memcpy_s(outTilingData, tilingBufCap, &localTilingData_, sizeof(localTilingData_));
    if (ret != EOK) {
        OP_LOGE(opName_, "MatmulAlltoAll postTiling: memcpy_s tiling data failed, ret=%d.", ret);
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(opName_, "Final tiling data size=%zu and context capacity size=%zu.", sizeof(KcQuantMatmulAlltoAllTilingData),
            context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(sizeof(KcQuantMatmulAlltoAllTilingData));
    context_->SetBlockDim(contextInfo.args_.aicCoreNum);
    PrintKcQuantMatmulAlltoAllTilingData(*outTilingData);
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 设置tilingInfo结构体
 *
 * @param tilingInfo 目标结构体
 */
void KcQuantMatmulAllToAllTilingBase::SetTilingInfo(MatmulAlltoAllTilingInfo &tilingInfo) const
{
    // 基本字段拷贝
    tilingInfo.tileM = inferredInfo.tileM;
    tilingInfo.tileCnt = inferredInfo.tileCnt;
    tilingInfo.tailM = inferredInfo.tailM;
    tilingInfo.tailCnt = inferredInfo.tailCnt;
    tilingInfo.rankM = contextInfo.args_.mValue;
    tilingInfo.rankN = contextInfo.args_.nValue;
    tilingInfo.rankK = contextInfo.args_.kValue;
    tilingInfo.mmResultLen = inferredInfo.mmResultLen;
    tilingInfo.permuteLen = inferredInfo.permuteLen;
    tilingInfo.biasLen = inferredInfo.biasLen;
    tilingInfo.aicCoreNum = contextInfo.args_.aicCoreNum;
    tilingInfo.rankDim = contextInfo.args_.rankDim;
    tilingInfo.hcclDataType =
        (static_cast<uint8_t>(mc2tiling::ConvertGeTypeToHcclType(opName_, contextInfo.args_.geAType))); // hccl数据类型
}

/**
 * @brief 获取对应的tilingKey
 * 使用QUANT_MODE来区分tilingKey,此处的QUANT_MODE指的是x1,x2的QUANT模式组合，以x1为pertoken量化(K)，x2为perchannel量化(C)
 * 为例子，K-C量化就代表一种组合
 *
 * @return uint64_t tilingKey结果
 */
uint64_t KcQuantMatmulAllToAllTilingBase::GetTilingKey() const
{
    // 按照量化组合模式，是否转置，bias数据类型进行展开
    bool x2TransposeFlag = contextInfo.args_.isBTrans ? true : false;
    uint32_t biasDType = DTYPE_BIAS_FP32;
    const uint64_t tilingKey = GET_TPL_TILING_KEY(KC_QUANT_MODE, x2TransposeFlag, biasDType);
    OP_LOGD(opName_, "KCQUANTMODE,X2TRANSPOSE,DTYPEBIAS: [%d,%d,%d], TilingKey is [%lu].", KC_QUANT_MODE,
            x2TransposeFlag, biasDType, tilingKey);
    return tilingKey;
}

/**
 * @brief 构造函数，创建一个KcQuantMatmulAllToAllTilingBase对象
 *
 * @param context
 */
KcQuantMatmulAllToAllTilingBase::KcQuantMatmulAllToAllTilingBase(gert::TilingContext *context) : MatmulAllToAllTilingBase(context)
{
}

// 注册tiling类
REGISTER_TILING_TEMPLATE_WITH_SOCVERSION(MatmulAlltoAll, KcQuantMatmulAllToAllTilingBase,
                                         static_cast<int32_t>(platform_ascendc::SocVersion::ASCEND910_95), 1);

} // namespace optiling