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
 * \file matmul_allto_all_tiling_base.cpp
 * \brief
 */
#include "matmul_allto_all_tiling_base.h"
#include "mc2_log.h"

using namespace AscendC;
using namespace ge;
using namespace Mc2Tiling;

namespace MC2Tiling {

/**
 * @brief 基类private私有方法，仅用于算子名称初始化
 *
 * @return ge::graphStatus
 */
ge::graphStatus MatmulAllToAllTilingBase::GetShapeAttrsInfo()
{
    opName_ = context_->GetNodeName();
    return ge::GRAPH_SUCCESS;
};

/**
 * @brief 获取平台相关信息
 *
 * @return ge::graphStatus
 */
ge::graphStatus MatmulAllToAllTilingBase::GetPlatformInfo()
{
    fe::PlatFormInfos *platformInfo = context_->GetPlatformInfo();
    OP_TILING_CHECK(platformInfo == nullptr, OP_LOGE(opName_, "Fail to get platform info."), return ge::GRAPH_FAILED);
    platform_ascendc::PlatformAscendC ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    socVersion_ = ascendcPlatform.GetSocVersion();
    libApiWorkSpaceSize_ = ascendcPlatform.GetLibApiWorkSpaceSize();
    contextInfo.args_.aicCoreNum = ascendcPlatform.GetCoreNumAic();
    return ge::GRAPH_SUCCESS;
};

/**
 * @brief 基类private私有方法，原本是用于高阶api，但在业务中实际没有使用到
 *
 * @return ge::graphStatus
 */
ge::graphStatus MatmulAllToAllTilingBase::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 获取额外申请的空间
 *
 * @return ge::graphStatus
 */
ge::graphStatus MatmulAllToAllTilingBase::GetWorkspaceSize()
{
    size_t *workspaces = context_->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workspaces == nullptr, OP_LOGE(opName_, "Get workspace failed"), return ge::GRAPH_FAILED);
    SetUserWorkSpace();
    uint64_t workspaceSize_ =
        libApiWorkSpaceSize_ + inferredInfo.mmResultLen + inferredInfo.permuteLen + inferredInfo.biasLen;
    workspaces[0] = workspaceSize_;
    OP_LOGD(opName_, "Workspaces[0] size=%ld, biasLen=%d, mmResultLen=%d", workspaces[0], inferredInfo.biasLen,
            inferredInfo.mmResultLen);
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 获取对应的tilingKey
 * 使用QUANT_MODE来区分tilingKey,此处的QUANT_MODE指的是x1,x2的QUANT模式组合，以x1为pertoken量化(K)，x2为perchannel量化(C)
 * 为例子，K-C量化就代表一种组合
 *
 * @return uint64_t tilingKey结果
 */
uint64_t MatmulAllToAllTilingBase::GetTilingKey() const
{
    // 按照量化组合模式，是否转置，bias数据类型进行展开
    bool x2TransposeFlag = contextInfo.args_.isBTrans ? true : false;
    // 0代表数据类型和x一致(FP16 OR BF16)，1代表FP32
    uint32_t biasDType = DTYPE_BIAS_SAME_WITH_X;
    if (contextInfo.args_.geBiasType != contextInfo.args_.geAType) {
        biasDType = DTYPE_BIAS_FP32;
    }
    const uint64_t tilingKey = GET_TPL_TILING_KEY(NON_QUANT_MODE, x2TransposeFlag, biasDType);
    OP_LOGD(opName_, "QUANTMODE,X2TRANSPOSE,DTYPEBIAS: [%d,%d,%d], TilingKey is [%lu].", NON_QUANT_MODE,
            x2TransposeFlag, biasDType, tilingKey);
    return tilingKey;
}

/**
 * @brief 进行通算切分:使用公式化tiling的方式，当前阶段公式化tiling只是个预估，需要针对alltoall的场景进行细化分析
 *
 * @return ge::graphStatus
 */
ge::graphStatus MatmulAllToAllTilingBase::TileCommAndCompute()
{
    OP_LOGD(opName_, "Start to find proper tile by formulaic tiling.");
    AlltoAllMM tileFormulate(contextInfo.args_, contextInfo.args_.rankDim, KernelType::ALL_TO_ALL,
                             SocVersion::SOC910_95);
    tileFormulate.GetTiling();
    CutResult mCutMMAlltoAll = tileFormulate.tilingM_.cutRes;
    inferredInfo.tileCnt = mCutMMAlltoAll.numLongTile;
    inferredInfo.tileM = mCutMMAlltoAll.longTileLen;
    inferredInfo.tailCnt = 0;
    inferredInfo.tailM = 0;
    if (mCutMMAlltoAll.numShortTile > 0) {
        inferredInfo.tailM = mCutMMAlltoAll.shortTileLen;
        inferredInfo.tailCnt = mCutMMAlltoAll.numShortTile;
    }
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 设置额外需要的空间，包括计算结果地址，重排地址，偏移地址等
 *
 */
void MatmulAllToAllTilingBase::SetUserWorkSpace()
{
    constexpr uint64_t alignAddrLen = 512;
    // MatmulAlltoAll先进行计算，需要有对应的空间先存放结果，假设x1(m,k),
    // x2(k,n),那么计算结果大小为m*n,这里申请的是一块总的空间，通算切分的头尾块偏移由kernel侧自行计算
    inferredInfo.mmResultLen = mc2tiling::AlignUp(
        contextInfo.args_.mValue * contextInfo.args_.nValue * contextInfo.args_.inputDtypeSize, alignAddrLen);
    // 重排空间等于mm计算结果空间
    inferredInfo.permuteLen = inferredInfo.mmResultLen;
    if (contextInfo.args_.isBias) {
        inferredInfo.biasLen =
            mc2tiling::AlignUp(contextInfo.args_.nValue, mc2tiling::SHAPE_ALIGN_SIZE) * sizeof(float);
    }
}

/**
 * @brief 校验MatmulAlltoAll在不同转置情况下的x1,x2,output的shape关系,以及需要满足n/rankSize的整除关系
 * 需要满足 x1(BS,H1), x2(H2, H1) if trans else x2(H1, H2)
 * output(BS*rankSize, H2/rankSize)
 *
 * @return ge::graphStatus
 */
ge::graphStatus MatmulAllToAllTilingBase::Check2DMatrixMulShapes(const gert::TilingContext *context, const char *opName)
{
    bool x2TransFlag = false;
    // attr及其元素的非空校验在前置的Check方法里都校验过，所以这里不需要额外判断
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    const bool *isTransX2 = attrs->GetAttrPointer<bool>(ATTR_X2_TRANSPOSE_INDEX);
    if (isTransX2) {
        x2TransFlag = *isTransX2;
    }
    const char *group = attrs->GetAttrPointer<char>(ATTR_GROUP_INDEX);

    int64_t rankDim = 0;
    if (MatmulAlltoAllTilingUtil::GetAndValidateRankSize(context, opName, group, rankDim) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    Matrix2DShapes shapeInfo;
    MatmulAlltoAllTilingUtil::GetMatrix2DShapes(context, shapeInfo);
    uint64_t kAxis = x2TransFlag ? shapeInfo.x2Dim1 : shapeInfo.x2Dim0;
    uint64_t nAxis = x2TransFlag ? shapeInfo.x2Dim0 : shapeInfo.x2Dim1;
    // MatmulAlltoAll, n要整除rankSize
    OP_TILING_CHECK(nAxis % static_cast<uint64_t>(rankDim) != 0,
                    OP_LOGE(opName, "N (%lu) is not divisible by rankSize (%ld).", nAxis, rankDim),
                    return ge::GRAPH_FAILED);
    // MatmulAlltoAll: x1Dim1 = x2 K-axis
    OP_TILING_CHECK((shapeInfo.x1Dim1 != kAxis),
                    OP_LOGE(opName,
                            "The x1 second dim should be the same with the %s dim of x2, "
                            "the x1 second dim is %lu, the x2 %s dim is %lu.",
                            x2TransFlag ? "second" : "first", shapeInfo.x1Dim1, x2TransFlag ? "second" : "first",
                            kAxis),
                    return ge::GRAPH_FAILED);
    // MatmulAlltoAll: yDim0 = x1Dim0 * rankDim and x2 N-axis = yDim1 * rankDim
    OP_TILING_CHECK((((shapeInfo.x1Dim0 * rankDim) != shapeInfo.yDim0) || (nAxis != (shapeInfo.yDim1 * rankDim))),
                    OP_LOGE(opName,
                            "The y first dim should be %lu times of the first dim of x1, "
                            "the x2 %s dim should be %lu times of the second dim of y. "
                            "rankDim: %lu, x1Dim0: %lu, yDim0: %lu, x2Dim%d: %lu, yDim1: %lu.",
                            rankDim, x2TransFlag ? "first" : "second", rankDim, rankDim, shapeInfo.x1Dim0,
                            shapeInfo.yDim0, x2TransFlag ? 0 : 1, nAxis, shapeInfo.yDim1),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

} // namespace MC2Tiling
