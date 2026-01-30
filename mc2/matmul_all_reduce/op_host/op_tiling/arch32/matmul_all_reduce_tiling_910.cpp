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
 * \file matmul_all_reduce_tiling_910.cc
 * \brief
 */
#include "matmul_all_reduce_tiling_910.h"
#include <string>
#include <vector>
#include "tiling/matmul_formulaic_tiling.h"
#include "platform/platform_infos_def.h"
#include "hccl/hccl_types.h"
#include "op_mc2.h"

namespace optiling {
namespace {
constexpr uint32_t ATTR_GROUP_INDEX = 0;
static const std::vector<int32_t> soc_version = {
    static_cast<int32_t>(platform_ascendc::SocVersion::ASCEND910B),
    static_cast<int32_t>(platform_ascendc::SocVersion::ASCEND910_93)};
} // namespace

using namespace Mc2Tiling;
bool MatmulAllReduceTiling910::IsCapable()
{
    OP_LOGI(opName_, "start with MatmulAllReduceTiling910 tiling.");
    return true;
}

void MatmulAllReduceTiling910::DoEmptyTensorTiling()
{
    MutableTCubeTileTilingData().M = args_.orgMValue;
    MutableTCubeTileTilingData().N = args_.orgNValue;
    MutableTCubeTileTilingData().isBias = args_.isBias;
    MutableTCubeTileTilingData().usedCoreNum = 1;
}

ge::graphStatus MatmulAllReduceTiling910::DoOpTiling()
{
    GE_ASSERT_GRAPH_SUCCESS(CheckA16W16());
    GE_ASSERT_GRAPH_SUCCESS(CheckInput());
    DoRCSTiling();
    DoSplitMTiling();
    if (!isKZero_) {
        GE_ASSERT_GRAPH_SUCCESS(Do910Tiling());
    } else {
        DoEmptyTensorTiling();
    }
    SetHcclTiling();
    return ge::GRAPH_SUCCESS;
}

uint64_t MatmulAllReduceTiling910::GetTilingKey() const
{
    if (unlikely(isKZero_)) {
        uint64_t tilingKey = GET_TPL_TILING_KEY(
            static_cast<uint64_t>(ASCEND_910B),
            static_cast<uint64_t>(MATMUL_ALLREDUCE_MM_TYPE_FP_MM),
            isKZero_,
            MATMUL_ALLREDUCE_INT8_COMM_F,
            0UL,    // ENABLE_L2_CACHE
            0UL,    // SHARE_MM
            SET_NOT_USE_FM_MM_TPL_TILING,
            SET_NOT_USE_QUANT_MM_TPL_TILING,
            SET_NOT_USE_WEIGHT_QUANT_MM_TPL_TILING);
        OP_LOGI(opName_, "Get tiling key %lu for empty tensor.", tilingKey);

        return tilingKey;
    }

    uint64_t MM_type = (matmulTPLParam_.disableMixNd2nz == MAT_MUL_V3_MIXND2NZ_FALSE &&
        !enableBiasConvert_ && !matmulAllReduce910TilingData_.param.isAdd) ?
        MATMUL_ALLREDUCE_MM_TYPE_FP_MM_CUBE_ONLY :
        MATMUL_ALLREDUCE_MM_TYPE_FP_MM;

    uint64_t tilingKey = GET_TPL_TILING_KEY(
        static_cast<uint64_t>(ASCEND_910B),
        MM_type,
        MATMUL_ALLREDUCE_EMPTY_INPUT_F,
        MATMUL_ALLREDUCE_INT8_COMM_F,
        0UL,    // ENABLE_L2_CACHE
        0UL,    // SHARE_MM
        matmulTPLParam_.disableMixNd2nz,
        SET_NOT_USE_QUANT_MM_TPL_TILING,
        SET_NOT_USE_WEIGHT_QUANT_MM_TPL_TILING);
    OP_LOGI(opName_, "Get tiling key %lu. Cube only flag is %lu", tilingKey, MM_type);

    return tilingKey;
}

ge::graphStatus MatmulAllReduceTiling910::GetWorkspaceSize()
{
    GE_ASSERT_GRAPH_SUCCESS(MatmulAllReduceTilingBase::GetWorkspaceSize());
    OP_LOGI(
        opName_, "select max workspace size to context, myWorkSpaceSize_:%lu, workspaceSize_:%lu", myWorkSpaceSize_,
        workspaceSize_);
    myWorkSpaceSize_ = std::max(myWorkSpaceSize_, workspaceSize_);
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    workspaces[0] = myWorkSpaceSize_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MatmulAllReduceTiling910::PostTiling()
{
    size_t tilingDataSize = sizeof(MatmulAllReduce910TilingData);
    OP_LOGD(
        opName_, "final tiling data size: %zu and context capacity size: %zu ",
        tilingDataSize, context_->GetRawTilingData()->GetCapacity());
    OP_TILING_CHECK(
        tilingDataSize % sizeof(uint64_t) != 0,
        VECTOR_INNER_ERR_REPORT_TILING(
            opName_, "tiling data size[%zu] not aligned to 8", tilingDataSize),
        return ge::GRAPH_FAILED);
    context_->GetRawTilingData()->SetDataSize(tilingDataSize);

    errno_t ret = memcpy_s(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity(),
        reinterpret_cast<void *>(&matmulAllReduce910TilingData_), tilingDataSize);
    if (ret != EOK){
        OP_LOGE(context_->GetNodeName(), "memcpy_s failed, ret=%d", ret);
        return ge::GRAPH_FAILED;
    }
    PrintTilingData();

    context_->SetBlockDim(args_.aicCoreNum);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MatmulAllReduceTiling910::SetHcclTiling()
{
    // A2和A3芯片设置hccltiling
    fe::PlatFormInfos *platformInfoPtr = context_->GetPlatformInfo();
    OP_TILING_CHECK(platformInfoPtr == nullptr,                         \
        OP_LOGE(context_->GetNodeName(), "fail to get platfoem info"),  \
        return ge::GRAPH_FAILED);
    fe::PlatFormInfos &platformInfo = *platformInfoPtr;
    std::string socVersionStr;
    (void)platformInfo.GetPlatformResWithLock("version", "Short_SoC_version", socVersionStr);
    auto group = context_->GetAttrs()->GetAttrPointer<char>(ATTR_GROUP_INDEX);
    OP_TILING_CHECK(group == nullptr,                         \
        OP_LOGE(context_->GetNodeName(), "GetAttrPointer for ATTR_GROUP_INDEX failed"),  \
        return ge::GRAPH_FAILED);
    uint32_t optype = HcclCMDType::HCCL_CMD_ALLREDUCE;
    std::string algConfig = (socVersionStr == "Ascend910_93") ? \
                        "AllReduce=level0:doublering" : "AllReduce=level0:fullmesh";
    OP_LOGD(context_->GetNodeName(), "MatmulAllReduceTiling910, SetHcclTiling algConfig is: %s", \
            algConfig.c_str());
    uint32_t reduceType = HcclReduceOp::HCCL_REDUCE_SUM;
    AscendC::Mc2CcTilingConfig mc2CcTilingConfig(group, optype, algConfig, reduceType);

    OP_TILING_CHECK(mc2CcTilingConfig.SetSkipBufferWindowCopy(                                              \
                        static_cast<uint8_t>(mc2tiling::MC2_BUFFER_TYPE::MC2_BUFFER_TYPE_DEFAULT)) != 0,  \
        OP_LOGE(context_->GetNodeName(), "mc2CcTilingConfig setSkipBufferWindowCopy failed"),               \
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(mc2CcTilingConfig.GetTiling(matmulAllReduce910TilingData_.mc2InitTiling) != 0,      \
        OP_LOGE(context_->GetNodeName(), "mc2CcTilingConfig mc2tiling GetTiling mc2InitTiling failed"), \
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(mc2CcTilingConfig.GetTiling(matmulAllReduce910TilingData_.mc2CcTilingV1) != 0,        \
        OP_LOGE(context_->GetNodeName(), "mc2CcTilingConfig mc2tiling GetTiling mc2CcTilingV1 failed"),   \
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MatmulAllReduceTiling910::Do910Tiling()
{
    args_.mValue = tileMValue_;
    TilingTransferHelper mmTile(*this, matmulAllReduce910TilingData_.tilematmulTiling);
    if (args_.enableSplitK) {
        OP_LOGD(opName_, "Enable SplitK Tiling.");
        auto res = mmTile.DoTiling();
        matmulTPLParam_ = mmTile.GetMatmulTPLParam();
        return res;
    } else {
        GE_ASSERT_GRAPH_SUCCESS(mmTile.DoTiling());
        if (MutableRCSTilingData().tailCnt == 0) {
            matmulTPLParam_ = mmTile.GetMatmulTPLParam();
            return ge::GRAPH_SUCCESS;
        }
        args_.mValue = tailMValue_;
        TilingTransferHelper mmTail(*this, matmulAllReduce910TilingData_.tailmatmulTiling);
        auto res = mmTail.DoTiling();
        matmulTPLParam_ = mmTail.GetMatmulTPLParam();
        return res;
    }
}

Mc2Tiling::RCSTiling& MatmulAllReduceTiling910::MutableRCSTilingData()
{
    return matmulAllReduce910TilingData_.param;
}

AscendC::tiling::TCubeTiling& MatmulAllReduceTiling910::MutableTCubeTileTilingData()
{
    return matmulAllReduce910TilingData_.tilematmulTiling.matmulTiling;
}

AscendC::tiling::TCubeTiling& MatmulAllReduceTiling910::MutableTCubeTailTilingData()
{
    return matmulAllReduce910TilingData_.tailmatmulTiling.matmulTiling;
}

ge::graphStatus MatmulAllReduceTiling910::CheckAxisSize()
{
    const uint64_t m = MatmulAllReduceTilingBase::GetMValue();
    OP_TILING_CHECK(
        m > static_cast<uint64_t>(INT32_MAX),
        VECTOR_INNER_ERR_REPORT_TILING(context_->GetNodeName(), "The size of m-axis(%lu) exceeds the upper limit.", m),
        return ge::GRAPH_FAILED);
    const uint64_t k = MatmulAllReduceTilingBase::GetKValue();
    OP_TILING_CHECK(
        k > static_cast<uint64_t>(INT32_MAX),
        VECTOR_INNER_ERR_REPORT_TILING(context_->GetNodeName(), "The size of k-axis(%lu) exceeds the upper limit.", k),
        return ge::GRAPH_FAILED);
    const uint64_t n = MatmulAllReduceTilingBase::GetNValue();
    OP_TILING_CHECK(
        n > static_cast<uint64_t>(INT32_MAX),
        VECTOR_INNER_ERR_REPORT_TILING(context_->GetNodeName(), "The size of n-axis(%lu) exceeds the upper limit.", n),
        return ge::GRAPH_FAILED);

    return CheckEmptyTensor();
}

ge::graphStatus MatmulAllReduceTiling910::CheckInputDtype()
{
    // x2 shape 为 2 维
    size_t x2DimNum = mmrCtxInfo_.x2_shape->GetStorageShape().GetDimNum();
    OP_TILING_CHECK(
        x2DimNum != DIM_NUM_TWO,
        VECTOR_INNER_ERR_REPORT_TILING(
            context_->GetNodeName(),
            "In the not quant scenario, Expect x2 dim to be 2, but "
            " got x2 dim:[%lu].",
            x2DimNum),
        return ge::GRAPH_FAILED);
    auto x1Type = mmrCtxInfo_.x1->GetDataType();
    //  x1 为fp16 或者bf16
    OP_TILING_CHECK(
        !((x1Type == ge::DT_FLOAT16) || (x1Type == ge::DT_BF16)),
        VECTOR_INNER_ERR_REPORT_TILING(
            context_->GetNodeName(),
            "In the not quant scenario, type of x1 should be"
            " fp16 or bf16."),
        return ge::GRAPH_FAILED);
    // x1，x2数据类型相同
    auto x2Type = mmrCtxInfo_.x2->GetDataType();
    OP_TILING_CHECK(
        x1Type != x2Type,
        VECTOR_INNER_ERR_REPORT_TILING(
            context_->GetNodeName(),
            "In the not quant scenario, type of x1 and x2"
            " should be same"),
        return ge::GRAPH_FAILED);
    // x1,bias数据类型相同
    if (mmrCtxInfo_.bias_shape != nullptr) {
        auto biasType = mmrCtxInfo_.bias->GetDataType();
        OP_TILING_CHECK(
            x1Type != biasType,
            VECTOR_INNER_ERR_REPORT_TILING(
                context_->GetNodeName(),
                "In the not quant scenario, type of x1 and bias should be"
                " same."),
            return ge::GRAPH_FAILED);
    }
    OP_LOGD(opName_, "Check Input Dtype Success.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MatmulAllReduceTiling910::CheckInputFormat()
{
    // 非量化场景B矩阵不支持NZ
    OP_TILING_CHECK(
        static_cast<ge::Format>(ge::GetPrimaryFormat(mmrCtxInfo_.x2->GetStorageFormat())) ==
            ge::Format::FORMAT_FRACTAL_NZ,
        VECTOR_INNER_ERR_REPORT_TILING(
            context_->GetNodeName(),
            "In the not quant scenario, weigth dont support NZ"),
        return ge::GRAPH_FAILED);
    OP_LOGD(opName_, "Check Input Format Success.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MatmulAllReduceTiling910::CheckInputShape()
{
    auto outputDimNum = mmrCtxInfo_.y_shape->GetStorageShape().GetDimNum();
    if (mmrCtxInfo_.x3_shape != nullptr) {
        auto x3DimNum = mmrCtxInfo_.x3_shape->GetStorageShape().GetDimNum();
        OP_TILING_CHECK(outputDimNum != x3DimNum,
            VECTOR_INNER_ERR_REPORT_TILING(
                context_->GetNodeName(),
                "In the not quant scenario, shape of x3 and output should be"
                " same."),
            return ge::GRAPH_FAILED);
        for (size_t i = 0U; i < outputDimNum; i++) {
            OP_TILING_CHECK(mmrCtxInfo_.y_shape->GetStorageShape().GetDim(i) !=
                mmrCtxInfo_.x3_shape->GetStorageShape().GetDim(i),
                VECTOR_INNER_ERR_REPORT_TILING(
                    context_->GetNodeName(),
                    "In the not quant scenario, shape of x3 and output should be"
                    " same."),
                return ge::GRAPH_FAILED);
        }
    }
    OP_LOGD(opName_, "Check Input Shape Success.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MatmulAllReduceTiling910::CheckInput()
{
    OP_LOGD(opName_, "Begin Check Input.");
    GE_ASSERT_GRAPH_SUCCESS(MatmulAllReduceTilingBase::CheckInput());
    GE_ASSERT_GRAPH_SUCCESS(CheckInputDtype());

    //  非量化场景不支持B矩阵Nz格式 除了310P
    if (socVersion_ != platform_ascendc::SocVersion::ASCEND310P) {
        GE_ASSERT_GRAPH_SUCCESS(CheckInputFormat());
    }

    GE_ASSERT_GRAPH_SUCCESS(CheckInputShape());

    return CheckAxisSize();
}

MatmulAllReduceTiling910::MatmulAllReduceTiling910(gert::TilingContext* context)
    : MatmulAllReduceTilingBase(context), matmulAllReduce910TilingData_(matmulAllReduce910TilingDataSelf_) {}

MatmulAllReduceTiling910::MatmulAllReduceTiling910(
    gert::TilingContext* context, MMRCtxInfo* mmrCtxInfo, MatmulAllReduce910TilingData* out)
    : MatmulAllReduceTilingBase(context, mmrCtxInfo), matmulAllReduce910TilingData_(*out)
{}

ge::graphStatus TilingTransferHelper::GetShapeAttrsInfo()
{
    auto&& tilingArgs = tilingProcesser_.args_;
    args_.opName = tilingProcesser_.opName_;
    args_.isATrans = tilingArgs.isATrans;
    args_.isBTrans = tilingArgs.isBTrans;
    args_.hasBias = tilingArgs.isBias;
    args_.aType = tilingArgs.geAType;
    args_.bType = tilingArgs.geBType;
    args_.cType = tilingArgs.geCType;
    args_.biasType = tilingArgs.isBias ? tilingArgs.geBiasType : ge::DT_INT32;

    args_.aFormat = ge::FORMAT_ND;
    args_.bFormat = ge::FORMAT_ND;
    args_.outFormat = ge::FORMAT_ND;

    args_.mValue = tilingArgs.mValue;
    args_.kValue = tilingArgs.kValue;
    args_.nValue = tilingArgs.nValue;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingTransferHelper::PostTiling()
{
    tilingProcesser_.myWorkSpaceSize_ = std::max(tilingProcesser_.myWorkSpaceSize_, workspaceSize_);
    OP_LOGI(tilingProcesser_.opName_, " set mm workspace size %lu to mc2", tilingProcesser_.myWorkSpaceSize_);
    return ge::GRAPH_SUCCESS;
}

MatmulTPLParam TilingTransferHelper::GetMatmulTPLParam()
{
    MatmulTPLParam param;
    param.disableMixNd2nz = static_cast<uint64_t>(GetMixNd2nzType());
    // 1: disable mix nd2nz 0: enable mix nd2nz
    return param;
}

TilingTransferHelper::TilingTransferHelper(MatmulAllReduceTiling910& matmulAllReduceTiling910, Mc2MatmulV3TilingData& data)
    : Mc2MatmulV3BaseTiling(matmulAllReduceTiling910.context_, &data), tilingProcesser_(matmulAllReduceTiling910)
{}

//注册Tiling类
REGISTER_TILING_TEMPLATE_WITH_SOCVERSION(MatmulAllReduce,MatmulAllReduceTiling910,soc_version,2);
} // namespace optiling
