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
 * \file flash_attention_score_v4_tiling.cpp
 * \brief FlashAttentionScoreV4 Tiling 实现
 *
 * 参考来源：
 *   flash_attention_score/op_host/flash_attention_score_tiling.cpp
 *   — 复用 FlashAttentionScore 的 tiling 注册框架（TilingRegistryArch）、
 *     空输入处理逻辑（IsEmptyInput/IsEmptyInputRegbase）以及 CompileInfo 结构。
 *   fused_floyd_attention/op_host/fused_floyd_attention_tiling.cpp
 *   — 参考推理算子对空输入场景的 tilingKey 处理方式（数据类型驱动的分支选择）。
 *
 * V4 与 FlashAttentionScore 的 Tiling 差异：
 *   1. 操作符名称为 FlashAttentionScoreV4，通过 IMPL_OP_OPTILING 注册。
 *   2. seed / offset 属性（索引 11/12）已由 def.cpp 声明，tiling 层面
 *      无需额外处理，内核侧直接通过 tilingData 中 inputParams.seed /
 *      inputParams.offset 字段获取（字段由底层 FlashAttentionScoreGeneralTilingData
 *      中已有的对应字段携带）。
 *   3. 非量化场景：移除 FLOAT8 类型检查分支。
 *
 * 输入索引：query=0, key=1, value=2
 * 输出索引：softmaxMax=0, softmaxSum=1, softmaxOut=2, attentionOut=3
 * 属性索引：inputLayout=5（与 infershape 一致）
 */

#include <queue>
#include <cmath>
#include <cfloat>
#include <register/op_impl_registry.h>
#include "log/log.h"
/* 复用 flash_attention_score 的 arch35 tiling regbase 框架 */
#include "../../common/op_kernel/arch35/flash_attention_score_tiling_regbase.h"
#include "tiling_base/data_copy_transpose_tiling.h"
#include "tiling_base/tiling_templates_registry.h"
#include "flash_attention_score_v4_tiling_common.h"
/* 复用 arch32 / arch35 的 tiling key 与 tiling 数据结构 */
#include "../../flash_attention_score/op_kernel/arch32/flash_attention_score_tiling.h"
#include "../../flash_attention_score/op_kernel/arch35/flash_attention_score_template_tiling_key.h"

using namespace ge;
using namespace AscendC;
using namespace Ops::Transformer::OpTiling;

namespace optiling {

/* ---- 输入/输出索引常量 ---- */
constexpr size_t FAV4_QUERY_IDX          = 0;
constexpr size_t FAV4_KEY_IDX            = 1;
constexpr size_t FAV4_VALUE_IDX          = 2;
constexpr size_t FAV4_SOFTMAXSUM_OUT_IDX = 1;
constexpr size_t FAV4_ATTENTIONOUT_IDX   = 3;
constexpr size_t FAV4_INPUTLAYOUT_ATTR   = 5;

constexpr size_t MIN_COPY_UNIT = 32;
constexpr size_t WS_100M       = 100UL * 1024UL * 1024UL;

constexpr uint32_t TILING_KEY_EMPTY = 1U;

/* ---- 工具函数 ---- */
static uint32_t CeilDiv(uint32_t a, uint32_t b)
{
    if (b == 0U) { return 0U; }
    return (a + b - 1U) / b;
}

/* ---- 空输入 Tiling 参数 ---- */
struct EmptyTilingArgs {
    uint32_t coreNum;
    uint32_t attentionOutFormerNum;
    uint32_t attentionOutTailNum;
    uint32_t softmaxMaxFormerNum;
    uint32_t softmaxMaxTailNum;
    uint64_t attentionOutSingleCoreDataSize;
    uint64_t attentionOutTailCoreDataSize;
    uint64_t softmaxMaxSingleCoreDataSize;
    uint64_t softmaxMaxTailCoreDataSize;
    uint64_t attentionOutLastCoreDataSize = 0ULL;
    uint64_t attentionOutLastCoreIndex    = 0ULL;
    uint64_t attentionOutBlockSize        = 0ULL;
    uint64_t softmaxSumBlockSize          = 0ULL;
    uint32_t aivActualNum                 = 0U;
};

/**
 * 计算空输入场景下各核的分配参数，逻辑与 flash_attention_score_tiling.cpp
 * 中的 GetEmptyArgs() 完全相同，此处按 V4 命名空间独立实现以保持解耦。
 */
static bool GetEmptyTilingArgs(EmptyTilingArgs &args, gert::TilingContext *context,
                                uint32_t coreNum, uint64_t attnOutSize, int64_t softmaxSumSize)
{
    args.coreNum = coreNum;
    if (coreNum == 0U) {
        OP_LOGE(context, "coreNum is 0");
        return false;
    }

    auto kType = context->GetInputDesc(FAV4_KEY_IDX)->GetDataType();
    if (kType != ge::DT_FLOAT16 && kType != ge::DT_FLOAT && kType != ge::DT_BF16) {
        OP_LOGE(context, "unsupported kernelType %d for V4 (non-quantized only)", kType);
        return false;
    }
    uint32_t kTypeBytes = ge::GetSizeByDataType(kType);
    if (kTypeBytes == 0U) {
        OP_LOGE(context, "kTypeBytes is 0");
        return false;
    }

    /* attentionOut 分块 */
    args.attentionOutBlockSize = CeilDiv(static_cast<uint32_t>(attnOutSize) * kTypeBytes, MIN_COPY_UNIT);
    if (attnOutSize != 0ULL) {
        if (args.attentionOutBlockSize % coreNum == 0ULL) {
            args.attentionOutTailCoreDataSize    = 0ULL;
            args.attentionOutFormerNum           = coreNum;
            args.attentionOutTailNum             = 0U;
            args.attentionOutSingleCoreDataSize  = args.attentionOutBlockSize / coreNum * MIN_COPY_UNIT / kTypeBytes;
            args.attentionOutLastCoreDataSize    = args.attentionOutSingleCoreDataSize -
                (args.attentionOutBlockSize * MIN_COPY_UNIT / kTypeBytes - attnOutSize);
            args.attentionOutLastCoreIndex       = static_cast<uint64_t>(args.attentionOutFormerNum - 1U) *
                                                   args.attentionOutSingleCoreDataSize;
        } else {
            args.attentionOutTailCoreDataSize   = args.attentionOutBlockSize / coreNum * MIN_COPY_UNIT / kTypeBytes;
            args.attentionOutSingleCoreDataSize = args.attentionOutTailCoreDataSize + MIN_COPY_UNIT / kTypeBytes;
            if (args.attentionOutBlockSize > coreNum) {
                args.attentionOutFormerNum        = static_cast<uint32_t>(args.attentionOutBlockSize % coreNum);
                args.attentionOutTailNum          = coreNum - args.attentionOutFormerNum;
                args.attentionOutLastCoreIndex    =
                    static_cast<uint64_t>(args.attentionOutFormerNum) * args.attentionOutSingleCoreDataSize +
                    static_cast<uint64_t>(args.attentionOutTailNum - 1U) * args.attentionOutTailCoreDataSize;
                args.attentionOutLastCoreDataSize =
                    args.attentionOutTailCoreDataSize -
                    (args.attentionOutSingleCoreDataSize * args.attentionOutFormerNum +
                     args.attentionOutTailCoreDataSize  * args.attentionOutTailNum - attnOutSize);
            } else {
                args.attentionOutFormerNum        = args.attentionOutBlockSize;
                args.attentionOutTailNum          = 0U;
                args.attentionOutLastCoreIndex    =
                    static_cast<uint64_t>(args.attentionOutFormerNum - 1U) * args.attentionOutSingleCoreDataSize;
                args.attentionOutLastCoreDataSize = args.attentionOutSingleCoreDataSize -
                    (args.attentionOutFormerNum * args.attentionOutSingleCoreDataSize - attnOutSize);
            }
        }
    } else {
        args.attentionOutFormerNum          = 0U;
        args.attentionOutTailNum            = 0U;
        args.attentionOutSingleCoreDataSize = 0ULL;
        args.attentionOutTailCoreDataSize   = 0ULL;
        args.attentionOutLastCoreDataSize   = 0ULL;
        args.attentionOutLastCoreIndex      = 0ULL;
    }

    /* softmaxSum/softmaxMax 分块（fp32） */
    uint32_t fp32Bytes = ge::GetSizeByDataType(ge::DT_FLOAT);
    args.softmaxSumBlockSize = CeilDiv(static_cast<uint32_t>(softmaxSumSize) * fp32Bytes, MIN_COPY_UNIT);
    if (args.softmaxSumBlockSize % coreNum == 0ULL) {
        args.softmaxMaxSingleCoreDataSize = args.softmaxSumBlockSize / coreNum * MIN_COPY_UNIT / fp32Bytes;
        args.softmaxMaxTailCoreDataSize   = 0ULL;
        args.softmaxMaxFormerNum          = coreNum;
        args.softmaxMaxTailNum            = 0U;
    } else {
        if (args.softmaxSumBlockSize > coreNum) {
            args.softmaxMaxFormerNum = static_cast<uint32_t>(args.softmaxSumBlockSize % coreNum);
            args.softmaxMaxTailNum   = coreNum - args.softmaxMaxFormerNum;
        } else {
            args.softmaxMaxFormerNum = args.softmaxSumBlockSize;
            args.softmaxMaxTailNum   = 0U;
        }
        args.softmaxMaxTailCoreDataSize   = args.softmaxSumBlockSize / coreNum * MIN_COPY_UNIT / fp32Bytes;
        args.softmaxMaxSingleCoreDataSize = args.softmaxMaxTailCoreDataSize + MIN_COPY_UNIT / fp32Bytes;
    }

    args.aivActualNum = std::max(
        (args.attentionOutFormerNum + args.attentionOutTailNum),
        (args.softmaxMaxFormerNum   + args.softmaxMaxTailNum));
    return true;
}

/**
 * 空输入处理（arch35 / regbase 路径）。
 * 与 flash_attention_score_tiling.cpp::IsEmptyInputRegbase() 逻辑相同。
 */
static bool IsEmptyInputRegbase(gert::TilingContext *context)
{
    auto attnOutShape   = context->GetOutputShape(FAV4_ATTENTIONOUT_IDX);
    auto queryShape     = context->GetInputShape(FAV4_QUERY_IDX);
    auto keyShape       = context->GetInputShape(FAV4_KEY_IDX);
    auto valueShape     = context->GetInputShape(FAV4_VALUE_IDX);
    auto softmaxSumShp  = context->GetOutputShape(FAV4_SOFTMAXSUM_OUT_IDX);

    OP_CHECK_NULL_WITH_CONTEXT(context, attnOutShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, queryShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, keyShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, valueShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, softmaxSumShp);

    uint64_t attnOutSize    = attnOutShape->GetStorageShape().GetShapeSize();
    int64_t  querySize      = queryShape->GetStorageShape().GetShapeSize();
    int64_t  keySize        = keyShape->GetStorageShape().GetShapeSize();
    int64_t  valueSize      = valueShape->GetStorageShape().GetShapeSize();
    int64_t  softmaxSumSize = softmaxSumShp->GetStorageShape().GetShapeSize();

    if ((querySize == 0 || keySize == 0 || valueSize == 0) &&
        (attnOutSize != 0 || softmaxSumSize != 0)) {
        auto *emptyTiling   = context->GetTilingData<FlashAttentionScoreEmptyInputTilingDataRegbase>();
        auto *compileInfo   = reinterpret_cast<const FlashAttentionScoreV4CompileInfo *>(context->GetCompileInfo());
        OP_CHECK_IF(compileInfo == nullptr,
                    OP_LOGE(context, "compileInfo is null"), return false);

        EmptyTilingArgs args;
        if (!GetEmptyTilingArgs(args, context, compileInfo->aivNum, attnOutSize, softmaxSumSize)) {
            return false;
        }

        emptyTiling->set_coreNum(args.coreNum);
        emptyTiling->set_attentionOutFormerNum(args.attentionOutFormerNum);
        emptyTiling->set_attentionOutTailNum(args.attentionOutTailNum);
        emptyTiling->set_softmaxMaxFormerNum(args.softmaxMaxFormerNum);
        emptyTiling->set_softmaxMaxTailNum(args.softmaxMaxTailNum);
        emptyTiling->set_attentionOutSingleCoreDataSize(args.attentionOutSingleCoreDataSize);
        emptyTiling->set_attentionOutTailCoreDataSize(args.attentionOutTailCoreDataSize);
        emptyTiling->set_softmaxMaxSingleCoreDataSize(args.softmaxMaxSingleCoreDataSize);
        emptyTiling->set_softmaxMaxTailCoreDataSize(args.softmaxMaxTailCoreDataSize);
        emptyTiling->set_attentionOutLastCoreDataSize(args.attentionOutLastCoreDataSize);
        emptyTiling->set_attentionOutLastCoreIndex(args.attentionOutLastCoreIndex);

        context->SetTilingKey(GET_TPL_TILING_KEY(1U, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1));
        context->SetBlockDim(compileInfo->aicNum);
        size_t *workspaces = context->GetWorkspaceSizes(1);
        workspaces[0] = WS_100M;
        return true;
    }
    return false;
}

/**
 * 空输入处理（arch32 路径）。
 * 与 flash_attention_score_tiling.cpp::IsEmptyInput() 逻辑相同。
 */
static bool IsEmptyInput(gert::TilingContext *context)
{
    auto attnOutShape  = context->GetOutputShape(FAV4_ATTENTIONOUT_IDX);
    auto queryShape    = context->GetInputShape(FAV4_QUERY_IDX);
    auto keyShape      = context->GetInputShape(FAV4_KEY_IDX);
    auto softmaxSumShp = context->GetOutputShape(FAV4_SOFTMAXSUM_OUT_IDX);

    OP_CHECK_NULL_WITH_CONTEXT(context, attnOutShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, queryShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, keyShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, softmaxSumShp);

    int64_t attnOutSize    = attnOutShape->GetStorageShape().GetShapeSize();
    int64_t querySize      = queryShape->GetStorageShape().GetShapeSize();
    int64_t keySize        = keyShape->GetStorageShape().GetShapeSize();
    int64_t softmaxSumSize = softmaxSumShp->GetStorageShape().GetShapeSize();

    if ((querySize == 0 || keySize == 0) &&
        (attnOutSize != 0 || softmaxSumSize != 0)) {
        auto *emptyTiling  = context->GetTilingData<FlashAttentionScoreTilingData>();
        emptyTiling->reset();
        auto *compileInfo  = reinterpret_cast<const FlashAttentionScoreV4CompileInfo *>(context->GetCompileInfo());
        OP_CHECK_IF(compileInfo == nullptr,
                    OP_LOGE(context, "compileInfo is null"), return false);

        EmptyTilingArgs args;
        if (!GetEmptyTilingArgs(args, context, compileInfo->aivNum, attnOutSize, softmaxSumSize)) {
            return false;
        }

        emptyTiling->emptyInputTilingData.set_coreNum(args.coreNum);
        emptyTiling->emptyInputTilingData.set_attentionOutFormerNum(args.attentionOutFormerNum);
        emptyTiling->emptyInputTilingData.set_attentionOutTailNum(args.attentionOutTailNum);
        emptyTiling->emptyInputTilingData.set_softmaxMaxFormerNum(args.softmaxMaxFormerNum);
        emptyTiling->emptyInputTilingData.set_softmaxMaxTailNum(args.softmaxMaxTailNum);
        emptyTiling->emptyInputTilingData.set_attentionOutSingleCoreDataSize(args.attentionOutSingleCoreDataSize);
        emptyTiling->emptyInputTilingData.set_attentionOutTailCoreDataSize(args.attentionOutTailCoreDataSize);
        emptyTiling->emptyInputTilingData.set_softmaxMaxSingleCoreDataSize(args.softmaxMaxSingleCoreDataSize);
        emptyTiling->emptyInputTilingData.set_softmaxMaxTailCoreDataSize(args.softmaxMaxTailCoreDataSize);
        emptyTiling->emptyInputTilingData.set_attentionOutLastCoreDataSize(args.attentionOutLastCoreDataSize);
        emptyTiling->emptyInputTilingData.set_attentionOutLastCoreIndex(args.attentionOutLastCoreIndex);

        context->SetTilingKey(TILING_KEY_EMPTY);

        auto platformInfo = context->GetPlatformInfo();
        OP_CHECK_IF(platformInfo == nullptr,
                    OP_LOGE(context, "platformInfo is null"), return false);
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        context->SetBlockDim(
            ascendcPlatform.CalcTschBlockDim(args.aivActualNum, 0, compileInfo->aivNum));

        size_t *workspaces = context->GetWorkspaceSizes(1);
        workspaces[0] = WS_100M;
        return true;
    }
    return false;
}

/**
 * 参数合法性校验（与 flash_attention_score_tiling::CheckParams 逻辑一致）。
 */
static ge::graphStatus CheckParams(const gert::TilingContext *context)
{
    if (context->GetAttrs() == nullptr ||
        context->GetInputShape(FAV4_QUERY_IDX)  == nullptr ||
        context->GetInputShape(FAV4_KEY_IDX)    == nullptr ||
        context->GetInputShape(FAV4_VALUE_IDX)  == nullptr) {
        OP_LOGW(context, "fail to get shape or attr from context");
        return ge::GRAPH_FAILED;
    }

    auto &valueShape = context->GetInputShape(FAV4_VALUE_IDX)->GetStorageShape();
    auto &keyShape   = context->GetInputShape(FAV4_KEY_IDX)->GetStorageShape();
    auto &queryShape = context->GetInputShape(FAV4_QUERY_IDX)->GetStorageShape();

    const char *inputLayout = context->GetAttrs()->GetAttrPointer<char>(FAV4_INPUTLAYOUT_ATTR);

    if (strlen(inputLayout) == 3) {
        if (inputLayout[0] == 'B') {
            OP_CHECK_IF((queryShape.GetDim(0) != keyShape.GetDim(0)),
                        OP_LOGE(context, "query or key batch dim mismatch"),
                        return ge::GRAPH_FAILED);
        } else if (inputLayout[0] == 'T') {
            /* TND：D 维须相同 */
            OP_CHECK_IF((queryShape.GetDim(2) != keyShape.GetDim(2)),
                        OP_LOGE(context, "query or key D dim mismatch in TND"),
                        return ge::GRAPH_FAILED);
            return ge::SUCCESS;
        } else {
            /* SBH */
            OP_CHECK_IF((queryShape.GetDim(1) != keyShape.GetDim(1)),
                        OP_LOGE(context, "query or key batch dim mismatch in SBH"),
                        return ge::GRAPH_FAILED);
        }
        OP_CHECK_IF((keyShape.GetDim(2) < valueShape.GetDim(2)),
                    OP_LOGE(context, "key D < value D"),
                    return ge::GRAPH_FAILED);
    } else if (strlen(inputLayout) == 4) {
        /* BNSD / BSND */
        OP_CHECK_IF((queryShape.GetDim(0) != keyShape.GetDim(0)),
                    OP_LOGE(context, "query or key batch dim mismatch"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF((queryShape.GetDim(3) != keyShape.GetDim(3)),
                    OP_LOGE(context, "query or key D dim mismatch"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF((keyShape.GetDim(3) < valueShape.GetDim(3)),
                    OP_LOGE(context, "key D < value D"),
                    return ge::GRAPH_FAILED);
    } else {
        OP_LOGW(context, "invalid input_layout[%s]", inputLayout);
        return ge::GRAPH_FAILED;
    }
    return ge::SUCCESS;
}

/* ---- Tiling 主入口 ---- */
ASCENDC_EXTERN_C ge::graphStatus TilingFlashAttentionScoreV4(gert::TilingContext *context)
{
    OP_LOGW(context, "FlashAttentionScoreV4 tiling start.");
    if (CheckParams(context) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_IF(platformInfo == nullptr,
                OP_LOGE(context, "platformInfo is null"),
                return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    if (ascendcPlatform.GetCurNpuArch() == NpuArch::DAV_3510) {
        /* arch35 路径（ascend950） */
        if (IsEmptyInputRegbase(context)) {
            return ge::GRAPH_SUCCESS;
        }
    } else {
        /* arch32 路径（ascend910b / ascend910_93） */
        if (IsEmptyInput(context)) {
            return ge::GRAPH_SUCCESS;
        }
    }

    /* 委托给与 FlashAttentionScore 共享的通用 tiling 注册器 */
    return TilingRegistryArch::GetInstance().DoTilingImpl(context);
}

/* ---- TilingPrepare（编译期信息解析） ---- */
ASCENDC_EXTERN_C ge::graphStatus TilingPrepareForFlashAttentionScoreV4(gert::TilingParseContext *context)
{
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_IF(platformInfo == nullptr,
                OP_LOGE(context, "platformInfo is null"),
                return ge::GRAPH_FAILED);

    auto *compileInfo = context->GetCompiledInfo<FlashAttentionScoreV4CompileInfo>();
    OP_CHECK_IF(compileInfo == nullptr,
                OP_LOGE(context, "compileInfo is null"),
                return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->aivNum    = ascendcPlatform.GetCoreNumAiv();
    compileInfo->aicNum    = ascendcPlatform.GetCoreNumAic();
    compileInfo->socVersion = ascendcPlatform.GetSocVersion();
    compileInfo->npuArch   = ascendcPlatform.GetCurNpuArch();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB,    compileInfo->ubSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1,    compileInfo->l1Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C,  compileInfo->l0cSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L2,    compileInfo->l2CacheSize);

    return ge::GRAPH_SUCCESS;
}

/*
 * 注册 tiling 入口。
 * TilingInputsDataDependency 中的索引对应 ValueDepend 的输入：
 *   13=prefix, 14=actual_seq_qlen, 15=actual_seq_kvlen, 16=q_start_idx, 17=kv_start_idx
 */
IMPL_OP_OPTILING(FlashAttentionScoreV4)
    .Tiling(TilingFlashAttentionScoreV4)
    .TilingInputsDataDependency({13, 14, 15, 16, 17})
    .TilingParse<FlashAttentionScoreV4CompileInfo>(TilingPrepareForFlashAttentionScoreV4);

} // namespace optiling
