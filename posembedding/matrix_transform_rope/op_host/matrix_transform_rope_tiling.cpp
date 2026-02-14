/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to License for details. You may not use this file except in compliance with License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "matrix_transform_rope_tiling.h"
#include "../../../common/include/tiling_base/tiling_templates_registry.h"
#include "register/op_def_registry.h"
#include "platform/platform_infos_def.h"
#include "../../../common/include/error_ops/error.h"

namespace optiling {

// 使用REGISTER_TILING_TEMPLATE注册Tiling模板类
// 参数: "MatrixTransformRope" - 算子类型名称
//       MatrixTransformRopeBaseTiling - Tiling实现类
//       1000 - 优先级
REGISTER_TILING_TEMPLATE("MatrixTransformRope", MatrixTransformRopeBaseTiling, 1000);

// 获取输入tensor的shape信息
ge::graphStatus MatrixTransformRopeBaseTiling::GetInputShape()
{
    auto xTensor = context_->GetInputTensor(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xTensor);
    // 获取shape信息
    auto shape = xTensor->GetStorageShape();
    auto dimNum = shape.GetDimNum();

    if (dimNum != 4) {
        OP_LOGE(context_->GetNodeName(), "x tensor dim must be 4, but got %zu", dimNum);
        return ge::GRAPH_FAILED;
    }

    xB_ = shape.GetDim(0);
    xN_ = shape.GetDim(1);
    xS_ = shape.GetDim(2);
    xD_ = shape.GetDim(3);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MatrixTransformRopeBaseTiling::ParseInputAndAttr()
{
    // 获取输入tensor信息
    auto xTensor = context_->GetInputTensor(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xTensor);

    auto totalElements = xTensor->GetStorageShape().GetShapeSize();

    // 获取平台信息
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo == nullptr) {
        OP_LOGE(context_->GetNodeName(), "get platform info failed");
        return ge::GRAPH_FAILED;
    }

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    uint64_t ubSize, l1Size, l0CSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, l0CSize);

    // 根据实际需要设置buffer space
    // TODO: 根据算子特性调整buffer大小
    mm_.SetBufferSpace(l1Size, l0CSize, ubSize);
    blockDim_ = ascendcPlatform.GetCoreNumAic();

    return ge::GRAPH_SUCCESS;
}

void MatrixTransformRopeBaseTiling::FillTilingData()
{
    // 设置tiling数据
    tilingData_.set_coreNum(blockDim_);
    tilingData_.set_totalElements(xB_ * xN_ * xS_ * xD_);
}

ge::graphStatus MatrixTransformRopeBaseTiling::TilingProcess()
{
    // 计算workspace大小
    // TODO: 根据算子特性计算workspace大小
    size_t userWorkspaceSize = 0;
    size_t systemWorkspaceSize = 10 * 1024 * 1024; // 10M

    workspaceSize_ = userWorkspaceSize + systemWorkspaceSize;

    // TODO: 根据算子特性设置tiling key
    tilingKey_ = 0UL;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MatrixTransformRopeBaseTiling::DoOpTiling()
{
    auto inputDesc = context_->GetInputDesc(0);
    if (inputDesc == nullptr) {
        OP_LOGE(context_->GetNodeName(), "invalid input pointer");
        return ge::GRAPH_FAILED;
    }

    if (ParseInputAndAttr() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (TilingProcess() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    FillTilingData();

    PrintTilingData();

    return ge::GRAPH_SUCCESS;
}

void MatrixTransformRopeBaseTiling::PrintTilingData()
{
    OP_LOGD(context_->GetNodeName(), "coreNum: [%d]", tilingData_.get_coreNum());
    OP_LOGD(context_->GetNodeName(), "totalElements: [%lu]", tilingData_.get_totalElements());
    OP_LOGD(context_->GetNodeName(), "x shape: [%ld, %ld, %ld, %ld]", xB_, xN_, xS_, xD_);
}

uint64_t MatrixTransformRopeBaseTiling::GetTilingKey() const
{
    return tilingKey_;
}

ge::graphStatus MatrixTransformRopeBaseTiling::PostTiling()
{
    OP_CHECK_IF(tilingData_.GetDataSize() % sizeof(uint64_t) != 0,
        OP_LOGE(context_->GetNodeName(), "tiling data size[%zu] is not aligned to 8", tilingData_.GetDataSize()),
        return ge::GRAPH_FAILED);
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetRawTilingData());
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    context_->SetBlockDim(tilingData_.get_coreNum());
    context_->SetScheduleMode(1);

    // 设置workspace
    size_t *workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_IF(workspaces == nullptr, OPS_REPORT_CUBE_INNER_ERR(context_->GetNodeName(), "workspaces is null"),
        return ge::GRAPH_FAILED);
    workspaces[0] = workspaceSize_;
    return ge::GRAPH_SUCCESS;
}

// Tiling函数入口
static ge::graphStatus TilingFunc4MatrixTransformRope(gert::TilingContext* context)
{
    OP_CHECK_IF(context == nullptr,
        OPS_REPORT_CUBE_INNER_ERR("[MatrixTransformRopeTilingFunc]", "context is null"),
        return ge::GRAPH_FAILED);

    return Ops::Transformer::OpTiling::TilingRegistry::GetInstance().DoTilingImpl(context);
}

// TilingPrepare函数，用于获取编译时平台信息
static ge::graphStatus TilingPrepare4MatrixTransformRope(gert::TilingParseContext* context)
{
    OP_CHECK_IF(context == nullptr,
                OPS_REPORT_CUBE_INNER_ERR("[TilingPrepare4MatrixTransformRope]", "context is null"),
                return ge::GRAPH_FAILED);
    fe::PlatformInfos* platformInfo = context->GetPlatformInfo();
    OP_CHECK_IF(platformInfo == nullptr,
                OPS_REPORT_CUBE_INNER_ERR(context->GetNodeName(), "platformInfoPtr is null"),
                return ge::GRAPH_FAILED);

    auto compileInfoPtr = context->GetCompileInfo<MatrixTransformRopeCompileInfo>();
    OP_CHECK_IF(compileInfoPtr == nullptr,
                OPS_REPORT_CUBE_INNER_ERR(context->GetNodeName(), "compileInfoPtr is null"),
                return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);

    compileInfoPtr->aicNum = ascendcPlatform.GetCoreNumAic();
    compileInfoPtr->aivNum = ascendcPlatform.GetCoreNumAiv();

    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, compileInfoPtr->l1Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, compileInfoPtr->l0ASize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, compileInfoPtr->l0BSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, compileInfoPtr->l0CSize);

    OP_LOGI(context->GetNodeName(),
            "parse compile info success l1Size:%lu, l0CSize:%lu, coreNum:%lu",
            compileInfoPtr->l1Size,
            compileInfoPtr->l0CSize,
            compileInfoPtr->aicNum);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_TILING(MatrixTransformRope)
    .Tiling(TilingFunc4MatrixTransformRope)
    .TilingParse<MatrixTransformRopeCompileInfo>(TilingPrepare4MatrixTransformRope);

} // namespace optiling
