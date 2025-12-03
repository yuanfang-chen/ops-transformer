/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file matmul_all_reduce_tiling_910_95.cc
 * \brief
 */
#include "matmul_all_reduce_tiling_910_95.h"
#include "tiling/new_mc2_tiling_utils.h"
#include "op_mc2.h"

namespace optiling {
bool MatmulAllReduceTilingA5::IsCapable()
{
    OP_LOGI(opName_, "Start with MatmulAllReduceTilingA5 tiling.");
    return true;
}

void MatmulAllReduceTilingA5::SetMc2Hcomm()
{
    OP_TILING_CHECK(
        mc2tiling::ConvertGeTypeToHcclType(opName_, args_.geCType) == mc2tiling::HcclDataType::HCCL_DATA_TYPE_RESERVED,
        VECTOR_INNER_ERR_REPORT_TILING(
            opName_, "cannot find HcclDataType according to ge datatype = %d.", static_cast<int32_t>(args_.geCType)),
        return );
    matmulAllReduce910TilingData_.hcommCfg.set_opType(
        static_cast<uint32_t>(mc2tiling::AicpuComType::HCCL_CMD_ALLREDUCE));
    matmulAllReduce910TilingData_.hcommCfg.set_srcDataType(
        static_cast<uint32_t>(mc2tiling::ConvertGeTypeToHcclType(opName_, args_.geCType)));
    matmulAllReduce910TilingData_.hcommCfg.set_dstDataType(
        static_cast<uint32_t>(mc2tiling::ConvertGeTypeToHcclType(opName_, args_.geCType)));

    matmulAllReduce910TilingData_.set_version(mc2tiling::COMM_VERSION3); // 新版本
    matmulAllReduce910TilingData_.set_hcommCnt(1);                       // allreduce 通信域数量为1
}

void MatmulAllReduceTilingA5::DoEmptyTensorTiling()
{
    MutableTCubeTileTilingData().set_M(args_.orgMValue);
    MutableTCubeTileTilingData().set_N(args_.orgNValue);
    MutableTCubeTileTilingData().set_isBias(args_.isBias);
    MutableTCubeTileTilingData().set_usedCoreNum(1);
}

ge::graphStatus MatmulAllReduceTilingA5::DoOpTiling()
{
    GE_ASSERT_GRAPH_SUCCESS(CheckA16W16());
    GE_ASSERT_GRAPH_SUCCESS(CheckInput());
    SetMc2Hcomm();
    DoRCSTiling();
    DoSplitMTiling();
    if (!isKZero_) {
        GE_ASSERT_GRAPH_SUCCESS(Do910Tiling());
    } else {
        DoEmptyTensorTiling();
    }
    DoAllReduceTiling(true);
    return ge::GRAPH_SUCCESS;
}

uint64_t MatmulAllReduceTilingA5::GetTilingKey() const
{
    if (unlikely(isKZero_)) {
        OP_LOGI(opName_, "Get tilingKey=%lu for empty tensor.", EMPTY_TENSOR_KEY);
        return EMPTY_TENSOR_KEY;
    }

    uint64_t tilingKey = 0;
    if (!matmulAllReduce910TilingData_.param.get_isAdd()) {
        tilingKey = CUBE_ONLY_KEY;
    } else {
        tilingKey = MM_ALINGNED_TILING_KEY;
    }
    // 为了不影响A2，910_95的tilingKey额外增加1）10^18
    tilingKey += mc2tiling::MC2_TILINGKEY_OFFSET;
    OP_LOGI(opName_, "Get tilingKey=%lu.", tilingKey);
    return tilingKey;
}

void MatmulAllReduceTilingA5::PrintExtendMatmulTiling(bool isTail)
{
    auto& tiling = matmulAllReduce910TilingData_.mC2Mmv3TailTilingData;
    if (isTail) {
        tiling = matmulAllReduce910TilingData_.mC2Mmv3TailTilingData;
    }

    OP_LOGD(opName_, "Matmul tiling mTailCnt=%u", tiling.get_mTailCnt());
    OP_LOGD(opName_, "Matmul tiling nTailCnt=%u", tiling.get_nTailCnt());
    OP_LOGD(opName_, "Matmul tiling kTailCnt=%u", tiling.get_kTailCnt());
    OP_LOGD(opName_, "Matmul tiling mBaseTailSplitCnt=%u", tiling.get_mBaseTailSplitCnt());
    OP_LOGD(opName_, "Matmul tiling nBaseTailSplitCnt=%u", tiling.get_nBaseTailSplitCnt());
    OP_LOGD(opName_, "Matmul tiling mTailMain=%u", tiling.get_mTailMain());
    OP_LOGD(opName_, "Matmul tiling nTailMain=%u", tiling.get_nTailMain());
    OP_LOGD(opName_, "Matmul tiling isHf32=%u", tiling.get_isHf32());
    OP_LOGD(opName_, "Matmul tiling aswWindowLen=%u", tiling.get_aswWindowLen());
}

ge::graphStatus MatmulAllReduceTilingA5::GetWorkspaceSize()
{
    GE_ASSERT_GRAPH_SUCCESS(MatmulAllReduceTilingBase::GetWorkspaceSize());
    OP_LOGI(
        opName_, "Select max workspace size to context, myWorkSpaceSize_=%lu, workspaceSize_=%lu.", myWorkSpaceSize_,
        workspaceSize_);
    myWorkSpaceSize_ = std::max(myWorkSpaceSize_, workspaceSize_);
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    workspaces[0] = myWorkSpaceSize_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MatmulAllReduceTilingA5::PostTiling()
{
    OP_LOGD(
        opName_, "Final tiling data size=%zu and context capacity size=%zu.",
        matmulAllReduce910TilingData_.GetDataSize(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(matmulAllReduce910TilingData_.GetDataSize());

    OP_TILING_CHECK(
        matmulAllReduce910TilingData_.GetDataSize() % sizeof(uint64_t) != 0,
        VECTOR_INNER_ERR_REPORT_TILING(
            opName_, "Tiling data size=%zu not aligned to 8.", matmulAllReduce910TilingData_.GetDataSize()),
        return ge::GRAPH_FAILED);
    PrintTilingData();
    context_->SetBlockDim(args_.aicCoreNum);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MatmulAllReduceTilingA5::Do910Tiling()
{
    OP_LOGD(opName_, "Start to excute DoMatmulV3Tiling!");
    // 获取芯片平台信息
    auto platformInfo = context_->GetPlatformInfo();
    OP_TILING_CHECK(platformInfo == nullptr, VECTOR_INNER_ERR_REPORT_TILING(opName_, "get platform info failed"),
                    return ge::GRAPH_FAILED);
    // 获取compileInfo
    OP_TILING_CHECK(mc2_matmul_v3_advanced::InitCompileInfo(platformInfo, &compileInfo_) != ge::GRAPH_SUCCESS,
                    VECTOR_INNER_ERR_REPORT_TILING(opName_, "init compile info failed"), return ge::GRAPH_FAILED);

    // 根据芯片型号获取策略模板
    std::vector<int32_t> priorities;
    OP_TILING_CHECK(mc2tiling::NewGetMatmulV3PriorityPolicy(socVersion_, priorities, opName_) != ge::GRAPH_SUCCESS,
                    VECTOR_INNER_ERR_REPORT_TILING(opName_, "get mmv3 priority policy failed"),
                    return ge::GRAPH_FAILED);
    Mc2MMRegisterCfg registerCfg {"Mc2MatMulV3", socVersion_, priorities};
    mc2tiling::NewUpdateMatmulV3Args(mmV3Args_, args_, opName_);

    // 获取tileTiling
    mmV3Args_.mValue = tileMValue_;
    OP_LOGD(opName_, "Do Mc2MatmulV3 tile tiling!");
    Mc2MatmulHelper::NewMc2MatmulTilingCfg tileTilingCfg(reinterpret_cast<const void*>(&compileInfo_),
                                                      reinterpret_cast<const void*>(&mmV3Args_));
    GE_ASSERT_GRAPH_SUCCESS(DoMatmulV3Tiling(tileTilingCfg, registerCfg, MutableMC2MmV3TileTilingData()));
    if (tailMValue_ != 0UL) {
        mmV3Args_.mValue = tailMValue_;
        OP_LOGD(opName_, "Do Mc2MatmulV3 tail tiling!");
        Mc2MatmulHelper::NewMc2MatmulTilingCfg tailTilingCfg(reinterpret_cast<const void*>(&compileInfo_),
                                                          reinterpret_cast<const void*>(&mmV3Args_));
        GE_ASSERT_GRAPH_SUCCESS(DoMatmulV3Tiling(tileTilingCfg, registerCfg, MutableMC2MmV3TailTilingData()));
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MatmulAllReduceTilingA5::DoMatmulV3Tiling(Mc2MatmulHelper::NewMc2MatmulTilingCfg& tilingCfg,
    Mc2MMRegisterCfg& registerCfg, optiling::MC2MatmulV3TilingData& tilingData)
{
    tilingCfg.SetRankDim(args_.rankDim);
    tilingCfg.SetMatMulV3TilingData(tilingData);
    if (Mc2MMTilingRegistry::GetInstance().DoTilingImpl(context_, tilingCfg, registerCfg) != ge::GRAPH_SUCCESS) {
        OP_LOGE(opName_, "Failed to do MatmulV3Tiling.");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

Mc2Msg& MatmulAllReduceTilingA5::MutableMc2MsgData()
{
    return matmulAllReduce910TilingData_.msg;
}

RCSTiling& MatmulAllReduceTilingA5::MutableRCSTilingData()
{
    return matmulAllReduce910TilingData_.param;
}

ge::graphStatus MatmulAllReduceTilingA5::CheckAxisSize()
{
    const uint64_t m = MatmulAllReduceTilingBase::GetMValue();
    OP_TILING_CHECK(
        m > static_cast<uint64_t>(INT32_MAX),
        VECTOR_INNER_ERR_REPORT_TILING(context_->GetNodeName(), "The size of m-axis=%lu exceeds the upper limit.", m),
        return ge::GRAPH_FAILED);
    const uint64_t k = MatmulAllReduceTilingBase::GetKValue();
    OP_TILING_CHECK(
        k > static_cast<uint64_t>(INT32_MAX),
        VECTOR_INNER_ERR_REPORT_TILING(context_->GetNodeName(), "The size of k-axis=%lu exceeds the upper limit.", k),
        return ge::GRAPH_FAILED);
    const uint64_t n = MatmulAllReduceTilingBase::GetNValue();
    OP_TILING_CHECK(
        n > static_cast<uint64_t>(INT32_MAX),
        VECTOR_INNER_ERR_REPORT_TILING(context_->GetNodeName(), "The size of n-axis=%lu exceeds the upper limit.", n),
        return ge::GRAPH_FAILED);

    return CheckEmptyTensor();
}

ge::graphStatus MatmulAllReduceTilingA5::CheckX1X2()
{
    // x2 shape 为 2 维
    size_t x2DimNum = mmrCtxInfo_.x2_shape->GetStorageShape().GetDimNum();
    OP_TILING_CHECK(
        x2DimNum != DIM_NUM_TWO,
        VECTOR_INNER_ERR_REPORT_TILING(
            context_->GetNodeName(),
            "In the not quant scenario, Expect x2 dim to be 2, "
            "but got x2 dim=%lu.",
            x2DimNum),
        return ge::GRAPH_FAILED);
    auto x1Type = mmrCtxInfo_.x1->GetDataType();
    //  x1 为fp16 或者bf16
    OP_TILING_CHECK(
        !((x1Type == ge::DT_FLOAT16) || (x1Type == ge::DT_BF16)),
        VECTOR_INNER_ERR_REPORT_TILING(
            context_->GetNodeName(),
            "In the not quant scenario, type of x1 should be fp16 or bf16, "
            "but got type of x1 type=%d.",
            x1Type),
        return ge::GRAPH_FAILED);
    // x1，x2数据类型相同
    auto x2Type = mmrCtxInfo_.x2->GetDataType();
    OP_TILING_CHECK(
        x1Type != x2Type,
        VECTOR_INNER_ERR_REPORT_TILING(
            context_->GetNodeName(),
            "In the not quant scenario, type of x1 and x2 should be same, "
            "but got type of x1=%d, type of x2=%d.",
            x1Type, x2Type),
        return ge::GRAPH_FAILED);
    // x1,bias数据类型相同
    if (mmrCtxInfo_.bias_shape != nullptr) {
        auto biasType = mmrCtxInfo_.bias->GetDataType();
        OP_TILING_CHECK(
            x1Type != biasType,
            VECTOR_INNER_ERR_REPORT_TILING(
                context_->GetNodeName(),
                "In the not quant scenario, type of x1 and bias should be same, "
                "but got type of x1=%d, type of bias=%d.",
                x1Type, biasType),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MatmulAllReduceTilingA5::CheckInput()
{
    GE_ASSERT_GRAPH_SUCCESS(MatmulAllReduceTilingBase::CheckInput());
    OP_TILING_CHECK(
        CheckX1X2() != ge::GRAPH_SUCCESS,
        VECTOR_INNER_ERR_REPORT_TILING(context_->GetNodeName(), "Check input_X failed."), return ge::GRAPH_FAILED);
    auto outputDimNum = mmrCtxInfo_.y_shape->GetStorageShape().GetDimNum();
    if (mmrCtxInfo_.x3_shape != nullptr) {
        auto x3DimNum = mmrCtxInfo_.x3_shape->GetStorageShape().GetDimNum();
        OP_TILING_CHECK(
            outputDimNum != x3DimNum,
            VECTOR_INNER_ERR_REPORT_TILING(
                context_->GetNodeName(),
                "In the not quant scenario, shape of x3 and output should be same, "
                "but got dim of x3=%lu, dim of output=%lu.",
                outputDimNum, x3DimNum),
            return ge::GRAPH_FAILED);
        for (size_t i = 0U; i < outputDimNum; i++) {
            auto outputDimValue = mmrCtxInfo_.y_shape->GetStorageShape().GetDim(i);
            auto x3DimValue = mmrCtxInfo_.x3_shape->GetStorageShape().GetDim(i);
            OP_TILING_CHECK(
                outputDimValue != x3DimValue,
                VECTOR_INNER_ERR_REPORT_TILING(
                    context_->GetNodeName(),
                    "In the not quant scenario, shape of x3 and output should be same, "
                    "but when dim=%lu, value of x3=%ld, value of output=%ld.",
                    i, outputDimValue, x3DimValue),
                return ge::GRAPH_FAILED);
        }
    }

    return CheckAxisSize();
}

MatmulAllReduceTilingA5::MatmulAllReduceTilingA5(gert::TilingContext* context)
    : MatmulAllReduceTilingBase(context), matmulAllReduce910TilingData_(matmulAllReduce910TilingDataSelf_)
{
    matmulAllReduce910TilingData_.SetDataPtr(context_->GetRawTilingData()->GetData());
}

MatmulAllReduceTilingA5::MatmulAllReduceTilingA5(
    gert::TilingContext* context, MMRCtxInfo* mmrCtxInfo, MatmulAllReduce910TilingDataA5* out)
    : MatmulAllReduceTilingBase(context, mmrCtxInfo), matmulAllReduce910TilingData_(*out)
{}

//注册tiling类
REGISTER_TILING_TEMPLATE_WITH_SOCVERSION(MatmulAllReduce,MatmulAllReduceTilingA5,static_cast<int32_t>(platform_ascendc::SocVersion::ASCEND910_95),2);
} // namespace optiling
