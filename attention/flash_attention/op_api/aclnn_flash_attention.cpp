/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_flash_attention.h"
#include "flash_attention.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/common_types.h"
#include "opdev/fast_vector.h"
#include "opdev/op_errno.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {

// ============================================================
// 常量定义
// ============================================================
static const int64_t HEAD_DIM_MAX = 768;
static const uint64_t DIM_NUM_2 = 2U;
static const uint64_t DIM_NUM_3 = 3U;
static const uint64_t DIM_NUM_4 = 4U;

// maskMode 枚举值
static const int64_t MASK_MODE_NONE       = 0;
static const int64_t MASK_MODE_CAUSAL     = 1;
static const int64_t MASK_MODE_ANTI_CAUSAL = 2;
static const int64_t MASK_MODE_BAND       = 3;
static const int64_t MASK_MODE_SLIDING    = 4;

// ============================================================
// 布局枚举
// ============================================================
enum class QLayout {
    BSND,
    BNSD,
    TND,
    INVALID
};

enum class KVLayout {
    BSND,
    TND,
    PA_ND,
    PA_Nz,
    INVALID
};

enum class OutLayout {
    BSND,
    BNSD,
    TND,
    INVALID
};

// ============================================================
// shape信息结构体
// ============================================================
struct FaUnifiedShapeInfo {
    int64_t batchSize  = 0;  // Batch维度 B（TND layout时虚拟为1）
    int64_t numHeadsQ  = 0;  // query的head数量 N_q
    int64_t numHeadsKV = 0;  // kv的head数量 N_kv（GQA时 N_kv <= N_q）
    int64_t seqLenQ    = 0;  // query序列长度（TND时为总token数 T_q）
    int64_t seqLenKV   = 0;  // kv序列长度（TND时为总token数 T_kv，PA时为块数*块大小）
    int64_t headDim    = 0;  // query和key共享的head维度 D
    int64_t headDimV   = 0;  // value的head维度 D_v（MQA/MLA时可能与D不同）

    QLayout  layoutQ   = QLayout::INVALID;
    KVLayout layoutKV  = KVLayout::INVALID;
    OutLayout layoutOut = OutLayout::INVALID;

    bool isTND   = false;  // 是否使用TND变长布局
    bool isPA    = false;  // 是否使用分页注意力（PA_ND/PA_Nz）
};

// ============================================================
// 布局字符串解析
// ============================================================
static QLayout ParseQLayout(const char *layout)
{
    if (layout == nullptr) { return QLayout::INVALID; }
    std::string s = op::ToString(layout).GetString();
    for (auto &c : s) { c = static_cast<char>(toupper(static_cast<unsigned char>(c))); }
    if (s == "BSND") { return QLayout::BSND; }
    if (s == "BNSD") { return QLayout::BNSD; }
    if (s == "TND")  { return QLayout::TND;  }
    return QLayout::INVALID;
}

static KVLayout ParseKVLayout(const char *layout)
{
    if (layout == nullptr) { return KVLayout::INVALID; }
    std::string s = op::ToString(layout).GetString();
    // PA_ND / PA_Nz 大小写保持原样比较（含下划线）
    if (s == "BSND" || s == "bsnd") { return KVLayout::BSND; }
    if (s == "TND"  || s == "tnd")  { return KVLayout::TND;  }
    if (s == "PA_ND" || s == "pa_nd") { return KVLayout::PA_ND; }
    if (s == "PA_Nz" || s == "pa_nz" || s == "PA_NZ") { return KVLayout::PA_Nz; }
    return KVLayout::INVALID;
}

static OutLayout ParseOutLayout(const char *layout)
{
    if (layout == nullptr) { return OutLayout::INVALID; }
    std::string s = op::ToString(layout).GetString();
    for (auto &c : s) { c = static_cast<char>(toupper(static_cast<unsigned char>(c))); }
    if (s == "BSND") { return OutLayout::BSND; }
    if (s == "BNSD") { return OutLayout::BNSD; }
    if (s == "TND")  { return OutLayout::TND;  }
    return OutLayout::INVALID;
}

// ============================================================
// 必要参数非空校验
// ============================================================
static aclnnStatus CheckMandatoryParams(
    const aclTensor *q,
    const aclTensor *k,
    const aclTensor *v,
    const char *layoutQ,
    const char *layoutKv,
    const char *layoutOut,
    const aclTensor *attentionOut,
    const uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    OP_CHECK(q != nullptr,
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Param q cannot be nullptr"),
        return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK(k != nullptr,
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Param k cannot be nullptr"),
        return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK(v != nullptr,
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Param v cannot be nullptr"),
        return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK(layoutQ != nullptr,
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Param layoutQ cannot be nullptr"),
        return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK(layoutKv != nullptr,
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Param layoutKv cannot be nullptr"),
        return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK(layoutOut != nullptr,
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Param layoutOut cannot be nullptr"),
        return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK(attentionOut != nullptr,
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Param attentionOut cannot be nullptr"),
        return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK(workspaceSize != nullptr,
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Param workspaceSize cannot be nullptr"),
        return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK(executor != nullptr,
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Param executor cannot be nullptr"),
        return ACLNN_ERR_PARAM_NULLPTR);
    return ACLNN_SUCCESS;
}

// ============================================================
// 数据类型校验（仅非量化：FLOAT16 / BFLOAT16）
// ============================================================
static aclnnStatus CheckInputDtype(
    const aclTensor *q,
    const aclTensor *k,
    const aclTensor *v,
    const aclTensor *attentionOut,
    const aclTensor *sinksOptional,
    const aclTensor *softmaxLseOptional,
    int64_t returnSoftmaxLse)
{
    auto qDtype   = q->GetDataType();
    auto kDtype   = k->GetDataType();
    auto vDtype   = v->GetDataType();
    auto outDtype = attentionOut->GetDataType();

    // q/k/v必须为FLOAT16或BFLOAT16
    if (qDtype != op::DataType::DT_FLOAT16 && qDtype != op::DataType::DT_BF16) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "q dtype must be FLOAT16 or BFLOAT16, but got [%s].",
                op::ToString(DataType(qDtype)).GetString());
        return ACLNN_ERR_PARAM_INVALID;
    }

    // q/k/v数据类型必须一致
    if (qDtype != kDtype || kDtype != vDtype) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "q[%s], k[%s], v[%s] dtypes must be identical.",
                op::ToString(DataType(qDtype)).GetString(),
                op::ToString(DataType(kDtype)).GetString(),
                op::ToString(DataType(vDtype)).GetString());
        return ACLNN_ERR_PARAM_INVALID;
    }

    // attentionOut数据类型必须与q一致
    if (outDtype != qDtype) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "attentionOut dtype [%s] must match q dtype [%s].",
                op::ToString(DataType(outDtype)).GetString(),
                op::ToString(DataType(qDtype)).GetString());
        return ACLNN_ERR_PARAM_INVALID;
    }

    // sinks必须为FLOAT32
    if (sinksOptional != nullptr) {
        auto sinksDtype = sinksOptional->GetDataType();
        if (sinksDtype != op::DataType::DT_FLOAT) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "sinks dtype must be FLOAT32, but got [%s].",
                    op::ToString(DataType(sinksDtype)).GetString());
            return ACLNN_ERR_PARAM_INVALID;
        }
    }

    // softmaxLse必须为FLOAT32
    if (returnSoftmaxLse != 0 && softmaxLseOptional != nullptr) {
        auto lseDtype = softmaxLseOptional->GetDataType();
        if (lseDtype != op::DataType::DT_FLOAT) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "softmaxLse dtype must be FLOAT32, but got [%s].",
                    op::ToString(DataType(lseDtype)).GetString());
            return ACLNN_ERR_PARAM_INVALID;
        }
    }

    return ACLNN_SUCCESS;
}

// ============================================================
// 辅助输入Tensor数据类型校验（block_table/cu_seqlens/seqused/metadata均为INT32）
// ============================================================
static aclnnStatus CheckAuxTensorDtype(const aclTensor *tensor, const char *name)
{
    if (tensor == nullptr) { return ACLNN_SUCCESS; }
    auto dtype = tensor->GetDataType();
    if (dtype != op::DataType::DT_INT32) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "%s dtype must be INT32, but got [%s].", name,
                op::ToString(DataType(dtype)).GetString());
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

// ============================================================
// layout字符串合法性校验
// ============================================================
static aclnnStatus CheckLayouts(
    const char *layoutQ,
    const char *layoutKv,
    const char *layoutOut,
    FaUnifiedShapeInfo &shapeInfo)
{
    shapeInfo.layoutQ = ParseQLayout(layoutQ);
    if (shapeInfo.layoutQ == QLayout::INVALID) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "layoutQ must be BSND/BNSD/TND (case-insensitive), but got [%s].", layoutQ);
        return ACLNN_ERR_PARAM_INVALID;
    }

    shapeInfo.layoutKV = ParseKVLayout(layoutKv);
    if (shapeInfo.layoutKV == KVLayout::INVALID) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "layoutKv must be BSND/TND/PA_ND/PA_Nz, but got [%s].", layoutKv);
        return ACLNN_ERR_PARAM_INVALID;
    }

    shapeInfo.layoutOut = ParseOutLayout(layoutOut);
    if (shapeInfo.layoutOut == OutLayout::INVALID) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "layoutOut must be BSND/BNSD/TND (case-insensitive), but got [%s].", layoutOut);
        return ACLNN_ERR_PARAM_INVALID;
    }

    shapeInfo.isTND = (shapeInfo.layoutQ == QLayout::TND);
    shapeInfo.isPA  = (shapeInfo.layoutKV == KVLayout::PA_ND || shapeInfo.layoutKV == KVLayout::PA_Nz);

    return ACLNN_SUCCESS;
}

// ============================================================
// storage格式检查（不支持NZ格式）
// ============================================================
static aclnnStatus CheckStorageFormat(
    const aclTensor *q,
    const aclTensor *k,
    const aclTensor *v,
    const aclTensor *blockTableOptional,
    const aclTensor *sinksOptional,
    const aclTensor *attentionOut)
{
    auto CheckNZ = [](const aclTensor *t, const char *name) -> bool {
        if (t == nullptr) { return true; }
        if (t->GetStorageFormat() == op::Format::FORMAT_FRACTAL_NZ) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "Tensor [%s] storage format NZ is not supported.", name);
            return false;
        }
        return true;
    };

    if (!CheckNZ(q, "q") || !CheckNZ(k, "k") || !CheckNZ(v, "v") ||
        !CheckNZ(attentionOut, "attentionOut") ||
        !CheckNZ(blockTableOptional, "blockTable") ||
        !CheckNZ(sinksOptional, "sinks")) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

// ============================================================
// 从shape和layout中解析B/N/S/D维度
// ============================================================
static aclnnStatus AnalyzeQShape(const aclTensor *q, FaUnifiedShapeInfo &info)
{
    Shape qShape = q->GetViewShape();
    switch (info.layoutQ) {
        case QLayout::BSND:
            if (qShape.GetDimNum() != DIM_NUM_4) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                        "q layoutQ=BSND requires 4-dim shape, got %zu.", qShape.GetDimNum());
                return ACLNN_ERR_PARAM_INVALID;
            }
            info.batchSize = qShape[0];
            info.seqLenQ   = qShape[1];
            info.numHeadsQ = qShape[2];
            info.headDim   = qShape[3];
            break;
        case QLayout::BNSD:
            if (qShape.GetDimNum() != DIM_NUM_4) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                        "q layoutQ=BNSD requires 4-dim shape, got %zu.", qShape.GetDimNum());
                return ACLNN_ERR_PARAM_INVALID;
            }
            info.batchSize = qShape[0];
            info.numHeadsQ = qShape[1];
            info.seqLenQ   = qShape[2];
            info.headDim   = qShape[3];
            break;
        case QLayout::TND:
            if (qShape.GetDimNum() != DIM_NUM_3) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                        "q layoutQ=TND requires 3-dim shape, got %zu.", qShape.GetDimNum());
                return ACLNN_ERR_PARAM_INVALID;
            }
            info.batchSize = 1;           // TND模式batch维虚拟为1
            info.seqLenQ   = qShape[0];   // T总token数
            info.numHeadsQ = qShape[1];
            info.headDim   = qShape[2];
            break;
        default:
            return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus AnalyzeKVShape(const aclTensor *k, const aclTensor *v, FaUnifiedShapeInfo &info)
{
    Shape kShape = k->GetViewShape();
    Shape vShape = v->GetViewShape();

    switch (info.layoutKV) {
        case KVLayout::BSND:
            if (kShape.GetDimNum() != DIM_NUM_4) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                        "k layoutKv=BSND requires 4-dim shape, got %zu.", kShape.GetDimNum());
                return ACLNN_ERR_PARAM_INVALID;
            }
            if (kShape[0] != info.batchSize) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                        "k batch(%ld) != q batch(%ld) under BSND layout.",
                        kShape[0], info.batchSize);
                return ACLNN_ERR_PARAM_INVALID;
            }
            info.seqLenKV   = kShape[1];
            info.numHeadsKV = kShape[2];
            info.headDimV   = (vShape.GetDimNum() >= DIM_NUM_4) ? vShape[3] : kShape[3];
            if (kShape[3] != info.headDim) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                        "k headDim(%ld) != q headDim(%ld).", kShape[3], info.headDim);
                return ACLNN_ERR_PARAM_INVALID;
            }
            break;
        case KVLayout::TND:
            if (kShape.GetDimNum() != DIM_NUM_3) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                        "k layoutKv=TND requires 3-dim shape, got %zu.", kShape.GetDimNum());
                return ACLNN_ERR_PARAM_INVALID;
            }
            info.seqLenKV   = kShape[0];   // T总token数
            info.numHeadsKV = kShape[1];
            info.headDimV   = (vShape.GetDimNum() >= DIM_NUM_3) ? vShape[2] : kShape[2];
            if (kShape[2] != info.headDim) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                        "k headDim(%ld) != q headDim(%ld).", kShape[2], info.headDim);
                return ACLNN_ERR_PARAM_INVALID;
            }
            break;
        case KVLayout::PA_ND:
            // PA_ND: shape=(NumBlocks, BlockSize, N_kv, D_k)
            if (kShape.GetDimNum() != DIM_NUM_4) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                        "k layoutKv=PA_ND requires 4-dim shape, got %zu.", kShape.GetDimNum());
                return ACLNN_ERR_PARAM_INVALID;
            }
            // NumBlocks * BlockSize为最大kv容量，运行时根据block_table确定实际长度
            info.seqLenKV   = kShape[0] * kShape[1];
            info.numHeadsKV = kShape[2];
            info.headDimV   = (vShape.GetDimNum() >= DIM_NUM_4) ? vShape[3] : kShape[3];
            if (kShape[3] != info.headDim) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                        "k headDim(%ld) != q headDim(%ld) under PA_ND layout.", kShape[3], info.headDim);
                return ACLNN_ERR_PARAM_INVALID;
            }
            break;
        case KVLayout::PA_Nz:
            // PA_Nz: shape=(NumBlocks, N_kv, BlockSize/16, D_k, 16)，NZ分页格式
            if (kShape.GetDimNum() < DIM_NUM_4) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                        "k layoutKv=PA_Nz requires at least 4-dim shape, got %zu.", kShape.GetDimNum());
                return ACLNN_ERR_PARAM_INVALID;
            }
            info.numHeadsKV = kShape[1];
            // headDimV沿用Q的headDim，PA_Nz时v shape含NZ排布
            info.headDimV   = info.headDim;
            info.seqLenKV   = 0;  // PA_Nz时实际kv长度由block_table/seqused在运行时决定
            break;
        default:
            return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

// ============================================================
// 综合shape合法性校验
// ============================================================
static aclnnStatus AnalyzeAndValidateShapes(
    const aclTensor *q,
    const aclTensor *k,
    const aclTensor *v,
    const aclTensor *blockTableOptional,
    FaUnifiedShapeInfo &info)
{
    CHECK_RET(AnalyzeQShape(q, info) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(AnalyzeKVShape(k, v, info) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    // GQA校验：N_kv必须是N_q的因数
    if (info.numHeadsQ > 0 && info.numHeadsKV > 0 && info.numHeadsQ % info.numHeadsKV != 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "numHeadsQ(%ld) must be divisible by numHeadsKV(%ld) for GQA.",
                info.numHeadsQ, info.numHeadsKV);
        return ACLNN_ERR_PARAM_INVALID;
    }

    // headDim上限校验
    if (info.headDim > HEAD_DIM_MAX) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "headDim(%ld) exceeds maximum allowed value %ld.", info.headDim, HEAD_DIM_MAX);
        return ACLNN_ERR_PARAM_INVALID;
    }

    // 分页注意力时block_table必须提供
    if (info.isPA && blockTableOptional == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "blockTable is required when layoutKv is PA_ND or PA_Nz.");
        return ACLNN_ERR_PARAM_INVALID;
    }

    // 非PA模式时block_table应为空
    if (!info.isPA && blockTableOptional != nullptr) {
        // 允许传入但给出警告，非PA模式block_table不参与计算
        OP_LOGW("blockTable is provided but layoutKv is not PA_ND/PA_Nz, blockTable will be ignored.");
    }

    // block_table shape校验：应为2维 (B, MaxBlocksPerSeq)
    if (blockTableOptional != nullptr) {
        Shape btShape = blockTableOptional->GetViewShape();
        if (btShape.GetDimNum() != DIM_NUM_2) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "blockTable must be 2-dim (B, MaxBlocksPerSeq), got %zu dims.", btShape.GetDimNum());
            return ACLNN_ERR_PARAM_INVALID;
        }
    }

    return ACLNN_SUCCESS;
}

// ============================================================
// 变长序列参数校验
// ============================================================
static aclnnStatus CheckVarLenParams(
    const aclTensor *cuSeqlensQOptional,
    const aclTensor *cuSeqlensKvOptional,
    const aclTensor *sequsedQOptional,
    const aclTensor *sequsedKvOptional,
    const FaUnifiedShapeInfo &info)
{
    // cu_seqlens和seqused不能同时提供
    bool hasCuSeqlens = (cuSeqlensQOptional != nullptr || cuSeqlensKvOptional != nullptr);
    bool hasSeqused   = (sequsedQOptional  != nullptr || sequsedKvOptional  != nullptr);
    if (hasCuSeqlens && hasSeqused) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "cu_seqlens and seqused cannot be provided at the same time.");
        return ACLNN_ERR_PARAM_INVALID;
    }

    // cu_seqlens_q和cu_seqlens_kv须同时提供或同时不提供
    if ((cuSeqlensQOptional == nullptr) != (cuSeqlensKvOptional == nullptr)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "cuSeqlensQ and cuSeqlensKv must be both provided or both nullptr.");
        return ACLNN_ERR_PARAM_INVALID;
    }

    // seqused_q和seqused_kv须同时提供或同时不提供
    if ((sequsedQOptional == nullptr) != (sequsedKvOptional == nullptr)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "sequsedQ and sequsedKv must be both provided or both nullptr.");
        return ACLNN_ERR_PARAM_INVALID;
    }

    // cu_seqlens只在TND layout下有意义
    if (cuSeqlensQOptional != nullptr && !info.isTND) {
        OP_LOGW("cuSeqlensQ is provided but layoutQ is not TND. "
                "cuSeqlens is typically used with TND layout.");
    }

    // cu_seqlens的shape校验：应为1维
    if (cuSeqlensQOptional != nullptr) {
        if (cuSeqlensQOptional->GetViewShape().GetDimNum() != 1U) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "cuSeqlensQ must be 1-dim tensor, got %zu dims.",
                    cuSeqlensQOptional->GetViewShape().GetDimNum());
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    if (cuSeqlensKvOptional != nullptr) {
        if (cuSeqlensKvOptional->GetViewShape().GetDimNum() != 1U) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "cuSeqlensKv must be 1-dim tensor, got %zu dims.",
                    cuSeqlensKvOptional->GetViewShape().GetDimNum());
            return ACLNN_ERR_PARAM_INVALID;
        }
    }

    // seqused的shape校验：应为1维 (B,)
    if (sequsedQOptional != nullptr) {
        if (sequsedQOptional->GetViewShape().GetDimNum() != 1U) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "sequsedQ must be 1-dim tensor, got %zu dims.",
                    sequsedQOptional->GetViewShape().GetDimNum());
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    if (sequsedKvOptional != nullptr) {
        if (sequsedKvOptional->GetViewShape().GetDimNum() != 1U) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "sequsedKv must be 1-dim tensor, got %zu dims.",
                    sequsedKvOptional->GetViewShape().GetDimNum());
            return ACLNN_ERR_PARAM_INVALID;
        }
    }

    return ACLNN_SUCCESS;
}

// ============================================================
// attention属性校验
// ============================================================
static aclnnStatus CheckAttentionAttrs(
    float softmaxMode,
    int64_t maskMode,
    int64_t winLeft,
    int64_t winRight,
    int64_t returnSoftmaxLse,
    int64_t deterministic,
    const aclTensor *softmaxLseOptional)
{
    // maskMode合法值
    if (maskMode < MASK_MODE_NONE || maskMode > MASK_MODE_SLIDING) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "maskMode must be in [0, 4], but got %ld.", maskMode);
        return ACLNN_ERR_PARAM_INVALID;
    }

    // 滑动窗口模式下 winLeft/winRight 须 >= 0
    if (maskMode == MASK_MODE_SLIDING) {
        if (winLeft < 0) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "winLeft must be >= 0 when maskMode=4, but got %ld.", winLeft);
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (winRight < 0) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "winRight must be >= 0 when maskMode=4, but got %ld.", winRight);
            return ACLNN_ERR_PARAM_INVALID;
        }
    }

    // returnSoftmaxLse为0或1
    if (returnSoftmaxLse != 0 && returnSoftmaxLse != 1) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "returnSoftmaxLse must be 0 or 1, but got %ld.", returnSoftmaxLse);
        return ACLNN_ERR_PARAM_INVALID;
    }

    // returnSoftmaxLse=1时softmaxLseOptional不应为nullptr
    if (returnSoftmaxLse == 1 && softmaxLseOptional == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR,
                "softmaxLseOptional must not be nullptr when returnSoftmaxLse=1.");
        return ACLNN_ERR_PARAM_NULLPTR;
    }

    // deterministic为0或1
    if (deterministic != 0 && deterministic != 1) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "deterministic must be 0 or 1, but got %ld.", deterministic);
        return ACLNN_ERR_PARAM_INVALID;
    }

    return ACLNN_SUCCESS;
}

// ============================================================
// 连续化：将非连续tensor转为连续tensor
// ============================================================
static aclnnStatus MakeContiguous(
    const aclTensor *&q,
    const aclTensor *&k,
    const aclTensor *&v,
    const aclTensor *&blockTableOptional,
    const aclTensor *&cuSeqlensQOptional,
    const aclTensor *&cuSeqlensKvOptional,
    const aclTensor *&sequsedQOptional,
    const aclTensor *&sequsedKvOptional,
    const aclTensor *&sinksOptional,
    aclOpExecutor *executor)
{
    q = l0op::Contiguous(q, executor);
    OP_CHECK(q != nullptr,
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Failed to make q contiguous"),
        return ACLNN_ERR_PARAM_NULLPTR);

    k = l0op::Contiguous(k, executor);
    OP_CHECK(k != nullptr,
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Failed to make k contiguous"),
        return ACLNN_ERR_PARAM_NULLPTR);

    v = l0op::Contiguous(v, executor);
    OP_CHECK(v != nullptr,
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Failed to make v contiguous"),
        return ACLNN_ERR_PARAM_NULLPTR);

    if (blockTableOptional != nullptr) {
        blockTableOptional = l0op::Contiguous(blockTableOptional, executor);
        OP_CHECK(blockTableOptional != nullptr,
            OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Failed to make blockTable contiguous"),
            return ACLNN_ERR_PARAM_NULLPTR);
    }

    if (cuSeqlensQOptional != nullptr) {
        cuSeqlensQOptional = l0op::Contiguous(cuSeqlensQOptional, executor);
        OP_CHECK(cuSeqlensQOptional != nullptr,
            OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Failed to make cuSeqlensQ contiguous"),
            return ACLNN_ERR_PARAM_NULLPTR);
    }

    if (cuSeqlensKvOptional != nullptr) {
        cuSeqlensKvOptional = l0op::Contiguous(cuSeqlensKvOptional, executor);
        OP_CHECK(cuSeqlensKvOptional != nullptr,
            OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Failed to make cuSeqlensKv contiguous"),
            return ACLNN_ERR_PARAM_NULLPTR);
    }

    if (sequsedQOptional != nullptr) {
        sequsedQOptional = l0op::Contiguous(sequsedQOptional, executor);
        OP_CHECK(sequsedQOptional != nullptr,
            OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Failed to make sequsedQ contiguous"),
            return ACLNN_ERR_PARAM_NULLPTR);
    }

    if (sequsedKvOptional != nullptr) {
        sequsedKvOptional = l0op::Contiguous(sequsedKvOptional, executor);
        OP_CHECK(sequsedKvOptional != nullptr,
            OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Failed to make sequsedKv contiguous"),
            return ACLNN_ERR_PARAM_NULLPTR);
    }

    if (sinksOptional != nullptr) {
        sinksOptional = l0op::Contiguous(sinksOptional, executor);
        OP_CHECK(sinksOptional != nullptr,
            OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Failed to make sinks contiguous"),
            return ACLNN_ERR_PARAM_NULLPTR);
    }

    return ACLNN_SUCCESS;
}

// ============================================================
// sinks shape为空时置nullptr（与FlashAttentionScore行为对齐）
// ============================================================
static void ProcessSinks(const aclTensor *&sinksOptional)
{
    if (sinksOptional != nullptr) {
        const auto &shape = sinksOptional->GetViewShape();
        if (shape.GetDimNum() == 1U && shape[0] == 0) {
            sinksOptional = nullptr;
        }
    }
}

// ============================================================
// 构造用于softmaxLse输出的占位符Tensor（returnSoftmaxLse=0时使用）
// ============================================================
static const aclTensor *BuildSoftmaxLsePlaceholder(aclOpExecutor *executor)
{
    std::vector<int64_t> shape = {0};
    int64_t addr = 0xff;
    return aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT,
                           shape.data(), 0, ACL_FORMAT_ND,
                           shape.data(), shape.size(),
                           static_cast<void *>(&addr));
}

// ============================================================
// 全量校验函数（对所有输入参数做完整校验，供GetWorkspaceSize调用）
// ============================================================
static aclnnStatus ValidateAllParams(
    const aclTensor *q,
    const aclTensor *k,
    const aclTensor *v,
    const aclTensor *blockTableOptional,
    const aclTensor *cuSeqlensQOptional,
    const aclTensor *cuSeqlensKvOptional,
    const aclTensor *sequsedQOptional,
    const aclTensor *sequsedKvOptional,
    const aclTensor *sinksOptional,
    float softmaxMode,
    int64_t maskMode,
    int64_t winLeft,
    int64_t winRight,
    const char *layoutQ,
    const char *layoutKv,
    const char *layoutOut,
    int64_t returnSoftmaxLse,
    int64_t deterministic,
    const aclTensor *attentionOut,
    const aclTensor *softmaxLseOptional,
    const uint64_t *workspaceSize,
    aclOpExecutor **executor,
    FaUnifiedShapeInfo &shapeInfo)
{
    // 1. 必须参数非空
    CHECK_RET(CheckMandatoryParams(q, k, v, layoutQ, layoutKv, layoutOut,
                                  attentionOut, workspaceSize, executor) == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_NULLPTR);

    // 2. 存储格式检查
    CHECK_RET(CheckStorageFormat(q, k, v, blockTableOptional, sinksOptional, attentionOut) == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);

    // 3. 数据类型检查
    CHECK_RET(CheckInputDtype(q, k, v, attentionOut, sinksOptional, softmaxLseOptional, returnSoftmaxLse) == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);

    // 4. 辅助tensor数据类型检查（均须为INT32）
    CHECK_RET(CheckAuxTensorDtype(blockTableOptional,   "blockTable") == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckAuxTensorDtype(cuSeqlensQOptional,  "cuSeqlensQ") == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckAuxTensorDtype(cuSeqlensKvOptional, "cuSeqlensKv") == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckAuxTensorDtype(sequsedQOptional,    "sequsedQ") == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckAuxTensorDtype(sequsedKvOptional,   "sequsedKv") == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckAuxTensorDtype(metadataOptional,    "metadata") == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    // 5. layout字符串解析与校验
    CHECK_RET(CheckLayouts(layoutQ, layoutKv, layoutOut, shapeInfo) == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);

    // 6. shape分析与校验
    CHECK_RET(AnalyzeAndValidateShapes(q, k, v, blockTableOptional, shapeInfo) == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);

    // 7. 变长序列参数校验
    CHECK_RET(CheckVarLenParams(cuSeqlensQOptional, cuSeqlensKvOptional,
                                sequsedQOptional, sequsedKvOptional, shapeInfo) == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);

    // 8. attention属性校验
    CHECK_RET(CheckAttentionAttrs(softmaxMode, maskMode, winLeft, winRight,
                                  returnSoftmaxLse, deterministic, softmaxLseOptional) == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);

    OP_LOGD("FlashAttention validation passed. B=%ld, N_q=%ld, N_kv=%ld, S_q=%ld, S_kv=%ld, D=%ld, D_v=%ld, "
            "isTND=%d, isPA=%d, maskMode=%ld, returnLSE=%ld.",
            shapeInfo.batchSize, shapeInfo.numHeadsQ, shapeInfo.numHeadsKV,
            shapeInfo.seqLenQ, shapeInfo.seqLenKV, shapeInfo.headDim, shapeInfo.headDimV,
            static_cast<int>(shapeInfo.isTND), static_cast<int>(shapeInfo.isPA),
            maskMode, returnSoftmaxLse);

    return ACLNN_SUCCESS;
}

}  // namespace

// ============================================================
// aclnnFlashAttentionGetWorkspaceSize：第一段接口
// ============================================================
aclnnStatus aclnnFlashAttentionGetWorkspaceSize(
    const aclTensor *q,
    const aclTensor *k,
    const aclTensor *v,
    const aclTensor *blockTableOptional,
    const aclTensor *cuSeqlensQOptional,
    const aclTensor *cuSeqlensKvOptional,
    const aclTensor *sequsedQOptional,
    const aclTensor *sequsedKvOptional,
    const aclTensor *sinksOptional,
    const aclTensor *metadataOptional,
    float softmaxMode,
    int64_t maskMode,
    int64_t winLeft,
    int64_t winRight,
    const char *layoutQ,
    const char *layoutKv,
    const char *layoutOut,
    int64_t returnSoftmaxLse,
    int64_t deterministic,
    const aclTensor *attentionOut,
    const aclTensor *softmaxLseOptional,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    // ---- 全量参数校验 ----
    FaUnifiedShapeInfo shapeInfo;
    CHECK_RET(ValidateAllParams(q, k, v, blockTableOptional, cuSeqlensQOptional, cuSeqlensKvOptional,
                                sequsedQOptional, sequsedKvOptional, sinksOptional,
                                softmaxMode, maskMode, winLeft, winRight,
                                layoutQ, layoutKv, layoutOut, returnSoftmaxLse, deterministic,
                                attentionOut, softmaxLseOptional, workspaceSize, executor, shapeInfo) == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);

    L2_DFX_PHASE_1(aclnnFlashAttention,
                   DFX_IN(q, k, v, blockTableOptional, cuSeqlensQOptional, cuSeqlensKvOptional,
                          sequsedQOptional, sequsedKvOptional, sinksOptional, metadataOptional,
                          softmaxMode, maskMode, winLeft, winRight,
                          layoutQ, layoutKv, layoutOut, returnSoftmaxLse, deterministic),
                   DFX_OUT(attentionOut, softmaxLseOptional));

    // ---- 创建执行器 ----
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // ---- 空tensor快速返回（B/S/N维度为0时不执行计算）----
    if (attentionOut->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    aclOpExecutor *l0Executor = uniqueExecutor.get();

    // ---- 连续化 ----
    CHECK_RET(MakeContiguous(q, k, v, blockTableOptional, cuSeqlensQOptional, cuSeqlensKvOptional,
                             sequsedQOptional, sequsedKvOptional, sinksOptional, l0Executor) == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_NULLPTR);

    // ---- sinks空shape处理 ----
    ProcessSinks(sinksOptional);

    // ---- 构造softmaxLse输出（returnSoftmaxLse=0时使用占位符）----
    const aclTensor *lseOutput = nullptr;
    const aclTensor *lsePlaceholder = nullptr;
    if (returnSoftmaxLse != 0) {
        lseOutput = softmaxLseOptional;
    } else {
        lsePlaceholder = BuildSoftmaxLsePlaceholder(l0Executor);
        lseOutput = lsePlaceholder;
    }

    // ---- 调用l0op::FlashAttention ----
    auto l0Outs = l0op::FlashAttention(
        q, k, v, blockTableOptional,
        cuSeqlensQOptional, cuSeqlensKvOptional,
        sequsedQOptional, sequsedKvOptional,
        sinksOptional, metadataOptional,
        softmaxMode, maskMode, winLeft, winRight,
        layoutQ, layoutKv, layoutOut,
        returnSoftmaxLse, deterministic,
        l0Executor);

    // ---- 销毁占位符 ----
    if (lsePlaceholder != nullptr) {
        aclDestroyTensor(lsePlaceholder);
    }

    auto l0AttentionOut = l0Outs[0];
    auto l0SoftmaxLse   = l0Outs[1];

    if (l0AttentionOut == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "l0op::FlashAttention returned nullptr for attentionOut.");
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_ERR_PARAM_NULLPTR;
    }

    // ---- 拷贝attentionOut到用户tensor ----
    auto vcAttentionOut = l0op::ViewCopy(l0AttentionOut, attentionOut, l0Executor);
    OP_CHECK(vcAttentionOut != nullptr,
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "ViewCopy for attentionOut failed"),
        return ACLNN_ERR_PARAM_NULLPTR);

    // ---- returnSoftmaxLse=1时拷贝softmaxLse到用户tensor ----
    if (returnSoftmaxLse != 0 && softmaxLseOptional != nullptr && l0SoftmaxLse != nullptr) {
        auto vcLse = l0op::ViewCopy(l0SoftmaxLse, softmaxLseOptional, l0Executor);
        OP_CHECK(vcLse != nullptr,
            OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "ViewCopy for softmaxLse failed"),
            return ACLNN_ERR_PARAM_NULLPTR);
    }

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

// ============================================================
// aclnnFlashAttention：第二段接口
// ============================================================
aclnnStatus aclnnFlashAttention(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    const aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnFlashAttention);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
