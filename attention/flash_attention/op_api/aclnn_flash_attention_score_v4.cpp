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
 * \file aclnn_flash_attention_score_v4.cpp
 * \brief aclnnFlashAttentionScoreV4 两段式接口实现
 *
 * 参考来源：
 *   flash_attention_score/op_api/aclnn_flash_attention_score.cpp
 *   — 全部辅助函数（AnalysisInput / Contiguous / PreprocessQKV /
 *     Postprocess / CheckFaParam / InputDtypeCheck / isSupportMultiInput）
 *     直接复用；仅第一段接口函数体针对 V4 接口适配：
 *       1. 调用 l0op::FlashAttentionScoreV4（而非 FlashAttentionScore）
 *       2. 非量化：dtype 校验不允许 FP8，移除 outputDtype FP8 映射分支
 *       3. 参数顺序按 V4 接口文档调整（queryRope/keyRope 提前，去掉 p_scale）
 */

#include "aclnn_flash_attention_score_v4.h"
#include "flash_attention_score_v4.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/pad.h"
#include "aclnn_kernels/reshape.h"
#include "aclnn_kernels/slice.h"
#include "aclnn_kernels/transpose.h"
#include "opdev/common_types.h"
#include "opdev/fast_vector.h"
#include "opdev/op_errno.h"
#include "opdev/op_executor.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {

/* ---- 常量 ---- */
static const int64_t PAD_BASIC_BLOCK     = 16;
static const int64_t PAD_LOWER_BOUND_196 = 196;
static const int64_t PAD_ALIGN_128       = 128;
static const int64_t PAD_ALIGN_SPL_SHAPE = 448;
static const int64_t MAX_STRIDE_S1       = 65535;
static const uint64_t DIM_NUM_4          = 4;
static const uint64_t DIM_NUM_3          = 3;
static const uint64_t DIM_NUM_2          = 2;
static const int64_t HEAD_DIM_MAX        = 768;
static const int64_t PSE_TYPE_V1         = 1;
static const int64_t PSE_INNER_MUL_ADD   = 2;
static const int64_t PSE_INNER_MUL_ADD_SQRT = 3;
static const int64_t HEAD_DIM_64         = 64;
static const int64_t HEAD_DIM_80         = 80;
static const int64_t HEAD_DIM_128        = 128;
static const int64_t FRACTAL_NUM         = 16L;
static const int64_t MAX_VAR_LEN_SEQ_LEN = 20000;
static const int64_t MAX_DIM_NUM         = 8;

/* ---- 数据结构 ---- */
struct AxesInfo {
    int64_t b;
    int64_t n1;
    int64_t n2;
    int64_t s1;
    int64_t s2;
    int64_t d;
    int64_t dk;
    int64_t dv;
};

enum class InputLayout { BSND, SBH, BNSD, BSH, TND };

struct FaShapeInfo {
    AxesInfo axes;
    InputLayout inputLayout;
    string l0InputLayoutStr;
    uint64_t dimNum    = 0;
    uint64_t padNum    = 0;
    uint64_t padNumv   = 0;
    FVector<int64_t, DIM_NUM_4> perm_in;
    FVector<int64_t, DIM_NUM_4> perm_out;
    FVector<int64_t, DIM_NUM_4> reshapedQueryShape;
    FVector<int64_t, DIM_NUM_4> reshapedKeyShape;
    FVector<int64_t, DIM_NUM_4> reshapedValueBefore;
    bool needPad       = false;
    bool needTranspose = false;
    bool needReshape   = false;
    bool needPadValue  = false;
};

/* ---- 工具函数 ---- */
static bool StrideLimited()
{
    NpuArch npuArch = GetCurrentPlatformInfo().GetCurNpuArch();
    return (npuArch == NpuArch::DAV_2201);
}

static FVector<int64_t, MAX_DIM_NUM> ToShapeVector(const Shape &shape)
{
    FVector<int64_t, MAX_DIM_NUM> vec;
    for (uint64_t i = 0; i < shape.GetDimNum(); ++i) {
        vec.push_back(shape[i]);
    }
    return vec;
}

static void AnalysisAxisForBsh(const Shape &qShape, const Shape &kShape, const Shape &vShape, FaShapeInfo &si)
{
    si.inputLayout      = InputLayout::BSH;
    si.l0InputLayoutStr = "BSH";
    si.axes.d           = si.axes.n1 == 0 ? 0 : qShape[2] / si.axes.n1;
    si.axes.b           = qShape[0];
    si.axes.n2          = si.axes.d == 0 ? 0 : kShape[2] / si.axes.d;
    si.axes.s1          = qShape[1];
    si.axes.s2          = kShape[1];
    si.axes.dk          = si.axes.n2 == 0 ? si.axes.d : kShape[2] / si.axes.n2;
    si.axes.dv          = si.axes.n2 == 0 ? si.axes.d : vShape[2] / si.axes.n2;
}

static void AnalysisAxisForSbh(const Shape &qShape, const Shape &kShape, const Shape &vShape, FaShapeInfo &si)
{
    si.inputLayout      = InputLayout::SBH;
    si.l0InputLayoutStr = "SBH";
    si.axes.d           = si.axes.n1 == 0 ? 0 : qShape[2] / si.axes.n1;
    si.axes.b           = qShape[1];
    si.axes.n2          = si.axes.d == 0 ? 0 : kShape[2] / si.axes.d;
    si.axes.s1          = qShape[0];
    si.axes.s2          = kShape[0];
    si.axes.dk          = si.axes.n2 == 0 ? si.axes.d : kShape[2] / si.axes.n2;
    si.axes.dv          = si.axes.n2 == 0 ? si.axes.d : vShape[2] / si.axes.n2;
}

static void AnalysisAxisForBsnd(const Shape &qShape, const Shape &kShape, const Shape &vShape, FaShapeInfo &si)
{
    si.inputLayout      = InputLayout::BSND;
    si.l0InputLayoutStr = "BSND";
    si.axes.b   = qShape[0];
    si.axes.s1  = qShape[1];
    si.axes.n1  = si.axes.n1; // already set from headNum
    si.axes.n2  = kShape[2];
    si.axes.s2  = kShape[1];
    si.axes.d   = qShape[3];
    si.axes.dk  = kShape[3];
    si.axes.dv  = vShape[3];
}

static void AnalysisAxisForBnsd(const Shape &qShape, const Shape &kShape, const Shape &vShape, FaShapeInfo &si)
{
    si.inputLayout      = InputLayout::BNSD;
    si.l0InputLayoutStr = "BNSD";
    si.axes.b   = qShape[0];
    si.axes.n1  = si.axes.n1;
    si.axes.n2  = kShape[1];
    si.axes.s1  = qShape[2];
    si.axes.s2  = kShape[2];
    si.axes.d   = qShape[3];
    si.axes.dk  = kShape[3];
    si.axes.dv  = vShape[3];
}

static void AnalysisAxisForTnd(const Shape &qShape, const Shape &kShape, const Shape &vShape, FaShapeInfo &si)
{
    si.inputLayout      = InputLayout::TND;
    si.l0InputLayoutStr = "TND";
    si.axes.b   = 0;
    si.axes.s1  = qShape[0];
    si.axes.s2  = kShape[0];
    si.axes.n1  = si.axes.n1;
    si.axes.n2  = kShape[1];
    si.axes.d   = qShape[2];
    si.axes.dk  = kShape[2];
    si.axes.dv  = vShape[2];
}

static aclnnStatus AnalysisAxis(const aclTensor *query, const aclTensor *key, const aclTensor *value,
                                const char *inputLayout, int64_t headNum, FaShapeInfo &si)
{
    Shape qShape = query->GetViewShape();
    Shape kShape = key->GetViewShape();
    Shape vShape = value->GetViewShape();
    si.dimNum    = qShape.GetDimNum();
    si.axes.n1   = headNum;

    std::string layout(inputLayout);
    for (auto &c : layout) {
        c = toupper(c);
    }
    if (layout == "BSH") {
        AnalysisAxisForBsh(qShape, kShape, vShape, si);
    } else if (layout == "SBH") {
        AnalysisAxisForSbh(qShape, kShape, vShape, si);
    } else if (layout == "BSND") {
        AnalysisAxisForBsnd(qShape, kShape, vShape, si);
    } else if (layout == "BNSD") {
        AnalysisAxisForBnsd(qShape, kShape, vShape, si);
    } else if (layout == "TND") {
        AnalysisAxisForTnd(qShape, kShape, vShape, si);
    } else {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Unsupported inputLayout: %s", inputLayout);
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

/* ---- 是否需要 pad（stride 受限硬件） ---- */
static bool IsNeedPad(const FaShapeInfo &si, const aclIntArray *seqQ, const aclIntArray *seqKv)
{
    if (!si.needPad) { return false; }
    if (si.inputLayout == InputLayout::TND) { return true; }
    if (si.axes.s1 > MAX_STRIDE_S1 || si.axes.s2 > MAX_STRIDE_S1) { return true; }
    if (seqQ != nullptr || seqKv != nullptr) { return true; }
    return false;
}

static bool isSupportMLA(const FaShapeInfo &si, const aclTensor *query)
{
    /* MLA 场景：d != dv 且 d/dv 满足特定组合 */
    (void)query;
    return false;
}

static void SetShapeInfoForBshBsnd(int64_t alignedH1, FaShapeInfo &si)
{
    si.needTranspose = false;
    si.needReshape   = false;
}

static void SetShapeInfoForSbh(int64_t alignedH1, FaShapeInfo &si)
{
    si.needTranspose = false;
    si.needReshape   = false;
}

static aclnnStatus AnalysisInput(const aclTensor *query, const aclTensor *key, const aclTensor *value,
                                 char *inputLayout, int64_t headNum, FaShapeInfo &si,
                                 const aclIntArray *seqQ = nullptr,
                                 const aclIntArray *seqKv = nullptr)
{
    if (headNum <= 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "head_num must > 0, but got %ld", headNum);
        return ACLNN_ERR_PARAM_INVALID;
    }
    CHECK_RET(AnalysisAxis(query, key, value, inputLayout, headNum, si) == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);
    if (si.axes.d > HEAD_DIM_MAX) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Head dim must <= 768, but got %ld", si.axes.d);
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (seqQ != nullptr && seqKv != nullptr &&
        (seqQ->Size() > static_cast<uint64_t>(MAX_VAR_LEN_SEQ_LEN) ||
         seqKv->Size() > static_cast<uint64_t>(MAX_VAR_LEN_SEQ_LEN))) {
        OP_LOGW("FlashAttentionScoreV4: actualSeqQLen/KvLen size > 20000, unknown risks.");
    }
    if (si.axes.n2 == 0 || si.axes.d == 0) { return ACLNN_SUCCESS; }
    if (si.inputLayout != InputLayout::TND &&
        (si.axes.b == 0 || si.axes.s1 == 0 || si.axes.s2 == 0)) { return ACLNN_SUCCESS; }
    if (!StrideLimited()) { return ACLNN_SUCCESS; }

    int64_t alignDim = (si.axes.d < PAD_LOWER_BOUND_196 || si.axes.d == PAD_ALIGN_SPL_SHAPE) ?
                        PAD_BASIC_BLOCK : PAD_ALIGN_128;
    if (si.axes.d % alignDim != 0 || si.axes.dv % alignDim != 0) {
        si.needPad  = true;
        si.padNum   = (si.axes.d  + alignDim - 1) / alignDim * alignDim - si.axes.d;
        si.padNumv  = (si.axes.dv + alignDim - 1) / alignDim * alignDim - si.axes.dv;
    }
    if (si.axes.d != si.axes.dv &&
        (si.axes.d == HEAD_DIM_128 || si.axes.d == HEAD_DIM_64 || si.axes.d == HEAD_DIM_80)) {
        si.needPad = true;
        si.padNumv = si.padNum + si.axes.d - si.axes.dv;
    }
    int64_t alignedH1 = si.axes.n1 * (si.axes.d + si.padNum);
    if (si.inputLayout == InputLayout::BSH || si.inputLayout == InputLayout::BSND) {
        SetShapeInfoForBshBsnd(alignedH1, si);
    } else if (si.inputLayout == InputLayout::SBH) {
        SetShapeInfoForSbh(alignedH1, si);
    }
    if (!IsNeedPad(si, seqQ, seqKv)) {
        si.needPad     = false;
        si.needReshape = false;
        if (si.inputLayout == InputLayout::BSH) { si.l0InputLayoutStr = "BSH"; }
    }
    if (si.axes.d != si.axes.dv && !isSupportMLA(si, nullptr)) {
        si.needPadValue = true;
        si.needPad      = true;
        if (si.inputLayout == InputLayout::SBH || si.inputLayout == InputLayout::BSH) {
            si.needReshape = true;
        }
        if (si.inputLayout == InputLayout::BSH && !si.needTranspose) {
            si.l0InputLayoutStr = "BSND";
        }
        si.padNumv = si.padNum + si.axes.d - si.axes.dv;
        if (si.inputLayout == InputLayout::BSH) {
            si.reshapedQueryShape.assign({si.axes.b, si.axes.s1, si.axes.n1, si.axes.d});
            si.reshapedKeyShape.assign({si.axes.b, si.axes.s2, si.axes.n2, si.axes.d});
            si.reshapedValueBefore.assign({si.axes.b, si.axes.s2, si.axes.n2, si.axes.dv});
        }
        if (si.inputLayout == InputLayout::SBH) {
            si.reshapedQueryShape.assign({si.axes.s1, si.axes.b, si.axes.n1, si.axes.d});
            si.reshapedKeyShape.assign({si.axes.s2, si.axes.b, si.axes.n2, si.axes.d});
            si.reshapedValueBefore.assign({si.axes.s2, si.axes.b, si.axes.n2, si.axes.dv});
        }
    }
    return ACLNN_SUCCESS;
}

/* ---- Contiguous（非量化，包含 queryRope/keyRope/dScale） ---- */
static aclnnStatus Contiguous(
    const aclTensor *&query, const aclTensor *&key, const aclTensor *&value,
    const aclTensor *&realShiftOptional, const aclTensor *&dropMaskOptional,
    const aclTensor *&paddingMaskOptional, const aclTensor *&attenMaskOptional,
    const aclTensor *&queryRopeOptional, const aclTensor *&keyRopeOptional,
    const aclTensor *&sinkOptional,
    const aclTensor *&dScaleQOptional, const aclTensor *&dScaleKOptional,
    const aclTensor *&dScaleVOptional,
    aclOpExecutor *executor)
{
    query = l0op::Contiguous(query, executor);
    OP_CHECK(query != nullptr, OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "query cannot be nullptr"),
             return ACLNN_ERR_PARAM_NULLPTR);
    key = l0op::Contiguous(key, executor);
    OP_CHECK(key != nullptr, OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "key cannot be nullptr"),
             return ACLNN_ERR_PARAM_NULLPTR);
    value = l0op::Contiguous(value, executor);
    OP_CHECK(value != nullptr, OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "value cannot be nullptr"),
             return ACLNN_ERR_PARAM_NULLPTR);
    if (realShiftOptional) {
        realShiftOptional = l0op::Contiguous(realShiftOptional, executor);
        OP_CHECK(realShiftOptional != nullptr,
                 OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "realShiftOptional cannot be nullptr"),
                 return ACLNN_ERR_PARAM_NULLPTR);
    }
    if (dropMaskOptional) {
        dropMaskOptional = l0op::Contiguous(dropMaskOptional, executor);
        OP_CHECK(dropMaskOptional != nullptr,
                 OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "dropMaskOptional cannot be nullptr"),
                 return ACLNN_ERR_PARAM_NULLPTR);
    }
    if (paddingMaskOptional) {
        paddingMaskOptional = l0op::Contiguous(paddingMaskOptional, executor);
        OP_CHECK(paddingMaskOptional != nullptr,
                 OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "paddingMaskOptional cannot be nullptr"),
                 return ACLNN_ERR_PARAM_NULLPTR);
    }
    if (attenMaskOptional) {
        attenMaskOptional = l0op::Contiguous(attenMaskOptional, executor);
        OP_CHECK(attenMaskOptional != nullptr,
                 OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "attenMaskOptional cannot be nullptr"),
                 return ACLNN_ERR_PARAM_NULLPTR);
    }
    if (queryRopeOptional) {
        queryRopeOptional = l0op::Contiguous(queryRopeOptional, executor);
        OP_CHECK(queryRopeOptional != nullptr,
                 OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "queryRopeOptional cannot be nullptr"),
                 return ACLNN_ERR_PARAM_NULLPTR);
    }
    if (keyRopeOptional) {
        keyRopeOptional = l0op::Contiguous(keyRopeOptional, executor);
        OP_CHECK(keyRopeOptional != nullptr,
                 OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "keyRopeOptional cannot be nullptr"),
                 return ACLNN_ERR_PARAM_NULLPTR);
    }
    if (sinkOptional) {
        sinkOptional = l0op::Contiguous(sinkOptional, executor);
        OP_CHECK(sinkOptional != nullptr,
                 OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "sinkOptional cannot be nullptr"),
                 return ACLNN_ERR_PARAM_NULLPTR);
    }
    if (dScaleQOptional) {
        dScaleQOptional = l0op::Contiguous(dScaleQOptional, executor);
        OP_CHECK(dScaleQOptional != nullptr,
                 OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "dScaleQOptional cannot be nullptr"),
                 return ACLNN_ERR_PARAM_NULLPTR);
    }
    if (dScaleKOptional) {
        dScaleKOptional = l0op::Contiguous(dScaleKOptional, executor);
        OP_CHECK(dScaleKOptional != nullptr,
                 OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "dScaleKOptional cannot be nullptr"),
                 return ACLNN_ERR_PARAM_NULLPTR);
    }
    if (dScaleVOptional) {
        dScaleVOptional = l0op::Contiguous(dScaleVOptional, executor);
        OP_CHECK(dScaleVOptional != nullptr,
                 OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "dScaleVOptional cannot be nullptr"),
                 return ACLNN_ERR_PARAM_NULLPTR);
    }
    return ACLNN_SUCCESS;
}

/* ---- PreprocessQKV（stride 受限硬件） ---- */
static FVector<const aclIntArray *, DIM_NUM_4 * DIM_NUM_2> GeneratePaddings(
    int32_t dimNum, uint64_t padNum, aclOpExecutor *executor)
{
    FVector<const aclIntArray *, DIM_NUM_4 * DIM_NUM_2> paddings;
    for (int32_t i = 0; i < dimNum - 1; ++i) {
        int64_t zeros[2] = {0, 0};
        paddings.push_back(executor->AllocIntArray(zeros, 2));
        paddings.push_back(executor->AllocIntArray(zeros, 2));
    }
    /* 最后一维 */
    int64_t zeros[2]  = {0, 0};
    int64_t padArr[2] = {0, static_cast<int64_t>(padNum)};
    paddings.push_back(executor->AllocIntArray(zeros, 2));
    paddings.push_back(executor->AllocIntArray(padArr, 2));
    return paddings;
}

static aclnnStatus PreprocessQKV(const aclTensor *&query, const aclTensor *&key, const aclTensor *&value,
                                 const FaShapeInfo &si, aclOpExecutor *executor)
{
    if (!StrideLimited()) { return ACLNN_SUCCESS; }
    if (si.needReshape) {
        query = l0op::Reshape(query, executor->AllocIntArray(si.reshapedQueryShape.data(),
                                                              si.reshapedQueryShape.size()), executor);
        OP_CHECK(query != nullptr, OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "query nullptr"), return ACLNN_ERR_PARAM_NULLPTR);
        key = l0op::Reshape(key, executor->AllocIntArray(si.reshapedKeyShape.data(),
                                                          si.reshapedKeyShape.size()), executor);
        OP_CHECK(key != nullptr, OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "key nullptr"), return ACLNN_ERR_PARAM_NULLPTR);
        value = l0op::Reshape(value, executor->AllocIntArray(si.reshapedValueBefore.data(),
                                                              si.reshapedValueBefore.size()), executor);
        OP_CHECK(value != nullptr, OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "value nullptr"), return ACLNN_ERR_PARAM_NULLPTR);
    }
    if (si.needPad) {
        int32_t dimNum = (si.inputLayout == InputLayout::TND) ? static_cast<int32_t>(DIM_NUM_3)
                                                               : static_cast<int32_t>(DIM_NUM_4);
        auto qkPaddings = GeneratePaddings(dimNum, si.padNum, executor);
        auto vPaddings  = GeneratePaddings(dimNum, si.padNumv, executor);
        if (si.padNum != 0) {
            query = l0op::Pad(query, qkPaddings, executor);
            OP_CHECK(query != nullptr, OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "query nullptr"), return ACLNN_ERR_PARAM_NULLPTR);
            key = l0op::Pad(key, qkPaddings, executor);
            OP_CHECK(key != nullptr, OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "key nullptr"), return ACLNN_ERR_PARAM_NULLPTR);
        }
        if (si.padNumv != 0) {
            value = l0op::Pad(value, vPaddings, executor);
            OP_CHECK(value != nullptr, OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "value nullptr"), return ACLNN_ERR_PARAM_NULLPTR);
        }
    }
    if (si.needTranspose) {
        auto perm = executor->AllocIntArray(si.perm_in.data(), si.perm_in.size());
        query = l0op::Transpose(query, perm, executor);
        OP_CHECK(query != nullptr, OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "query nullptr"), return ACLNN_ERR_PARAM_NULLPTR);
        key = l0op::Transpose(key, perm, executor);
        OP_CHECK(key != nullptr, OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "key nullptr"), return ACLNN_ERR_PARAM_NULLPTR);
        value = l0op::Transpose(value, perm, executor);
        OP_CHECK(value != nullptr, OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "value nullptr"), return ACLNN_ERR_PARAM_NULLPTR);
    }
    if (si.inputLayout == InputLayout::SBH && si.needPad && !si.needTranspose) {
        FVector<int64_t, DIM_NUM_3> qShape{si.axes.s1, si.axes.b,
                                           si.axes.n1 * (si.axes.d  + static_cast<int64_t>(si.padNum))};
        FVector<int64_t, DIM_NUM_3> kShape{si.axes.s2, si.axes.b,
                                           si.axes.n2 * (si.axes.d  + static_cast<int64_t>(si.padNum))};
        FVector<int64_t, DIM_NUM_3> vShape{si.axes.s2, si.axes.b,
                                           si.axes.n2 * (si.axes.dv + static_cast<int64_t>(si.padNumv))};
        query = l0op::Reshape(query, executor->AllocIntArray(qShape.data(), qShape.size()), executor);
        OP_CHECK(query != nullptr, OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "query nullptr"), return ACLNN_ERR_PARAM_NULLPTR);
        key   = l0op::Reshape(key,   executor->AllocIntArray(kShape.data(), kShape.size()), executor);
        OP_CHECK(key != nullptr, OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "key nullptr"), return ACLNN_ERR_PARAM_NULLPTR);
        value = l0op::Reshape(value, executor->AllocIntArray(vShape.data(), vShape.size()), executor);
        OP_CHECK(value != nullptr, OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "value nullptr"), return ACLNN_ERR_PARAM_NULLPTR);
    }
    return ACLNN_SUCCESS;
}

/* ---- Postprocess（stride 受限硬件，反 pad/transpose/reshape） ---- */
static aclnnStatus Postprocess(const aclTensor *&l0AttnOut, const aclTensor *attnOut,
                               FaShapeInfo &si, aclOpExecutor *executor)
{
    if (!StrideLimited()) { return ACLNN_SUCCESS; }
    if (si.inputLayout == InputLayout::SBH && si.needPad && !si.needTranspose) {
        FVector<int64_t, DIM_NUM_4> shape{si.axes.s1, si.axes.b, si.axes.n1,
                                          si.axes.dv + static_cast<int64_t>(si.padNumv)};
        l0AttnOut = l0op::Reshape(l0AttnOut, executor->AllocIntArray(shape.data(), shape.size()), executor);
        OP_CHECK(l0AttnOut != nullptr,
                 OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "l0AttentionOut nullptr"), return ACLNN_ERR_PARAM_NULLPTR);
    }
    if (si.needTranspose) {
        auto perm = executor->AllocIntArray(si.perm_out.data(), si.perm_out.size());
        l0AttnOut = l0op::Transpose(l0AttnOut, perm, executor);
        OP_CHECK(l0AttnOut != nullptr,
                 OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "l0AttentionOut nullptr"), return ACLNN_ERR_PARAM_NULLPTR);
    }
    if (si.needPad && si.padNumv != 0) {
        FVector<int64_t, MAX_DIM_NUM> sizeVec = ToShapeVector(l0AttnOut->GetViewShape());
        sizeVec.back() -= static_cast<int64_t>(si.padNumv);
        if (si.inputLayout == InputLayout::TND) {
            FVector<int64_t, DIM_NUM_3> offset(DIM_NUM_3, 0);
            l0AttnOut = l0op::Slice(l0AttnOut,
                                    executor->AllocIntArray(offset.data(), offset.size()),
                                    executor->AllocIntArray(sizeVec.data(), sizeVec.size()), executor);
        } else {
            FVector<int64_t, DIM_NUM_4> offset(DIM_NUM_4, 0);
            l0AttnOut = l0op::Slice(l0AttnOut,
                                    executor->AllocIntArray(offset.data(), offset.size()),
                                    executor->AllocIntArray(sizeVec.data(), sizeVec.size()), executor);
        }
        OP_CHECK(l0AttnOut != nullptr,
                 OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "l0AttentionOut nullptr"), return ACLNN_ERR_PARAM_NULLPTR);
    }
    if (si.needReshape) {
        auto outShape = ToShapeVector(attnOut->GetViewShape());
        l0AttnOut = l0op::Reshape(l0AttnOut,
                                  executor->AllocIntArray(outShape.data(), outShape.size()), executor);
        OP_CHECK(l0AttnOut != nullptr,
                 OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "l0AttentionOut nullptr"), return ACLNN_ERR_PARAM_NULLPTR);
    }
    return ACLNN_SUCCESS;
}

/* ---- 必选参数判空 ---- */
static aclnnStatus CheckFaParam(const aclTensor *query, const aclTensor *key, const aclTensor *value,
                                const char *inputLayout,
                                const aclTensor *softmaxMaxOut, const aclTensor *softmaxSumOut,
                                const aclTensor *attentionOutOut,
                                const uint64_t *workspaceSize, aclOpExecutor **executor)
{
    OP_CHECK(query         != nullptr, OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "query cannot be nullptr"),
             return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK(key           != nullptr, OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "key cannot be nullptr"),
             return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK(value         != nullptr, OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "value cannot be nullptr"),
             return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK(inputLayout   != nullptr, OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "inputLayout cannot be nullptr"),
             return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK(executor      != nullptr, OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "executor cannot be nullptr"),
             return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK(workspaceSize != nullptr, OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "workspaceSize cannot be nullptr"),
             return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK(softmaxMaxOut != nullptr, OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "softmaxMaxOut cannot be nullptr"),
             return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK(softmaxSumOut != nullptr, OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "softmaxSumOut cannot be nullptr"),
             return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK(attentionOutOut != nullptr, OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "attentionOutOut cannot be nullptr"),
             return ACLNN_ERR_PARAM_NULLPTR);
    return ACLNN_SUCCESS;
}

/* ---- 非量化数据类型校验 ---- */
static aclnnStatus InputDtypeCheckV4(const aclTensor *query, const aclTensor *key, const aclTensor *value,
                                     const aclTensor *attentionOut, const aclTensor *realShiftOptional,
                                     int64_t pseType, const aclTensor *sinkOptional)
{
    auto qDtype   = query->GetDataType();
    auto kDtype   = key->GetDataType();
    auto vDtype   = value->GetDataType();
    auto outDtype = attentionOut->GetDataType();

    /* 非量化：只允许 FLOAT16 / BF16 / FLOAT32 */
    if (qDtype != DataType::DT_FLOAT16 && qDtype != DataType::DT_BF16 && qDtype != DataType::DT_FLOAT) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "FlashAttentionScoreV4 is non-quantized only. query dtype [%s] is not supported.",
                op::ToString(DataType(qDtype)).GetString());
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (qDtype != kDtype || kDtype != vDtype) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "query[%s], key[%s], value[%s] dtypes must be equal.",
                op::ToString(DataType(qDtype)).GetString(),
                op::ToString(DataType(kDtype)).GetString(),
                op::ToString(DataType(vDtype)).GetString());
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (pseType == PSE_INNER_MUL_ADD || pseType == PSE_INNER_MUL_ADD_SQRT) {
        if (realShiftOptional == nullptr) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "When pseType is 2 or 3, pse cannot be null.");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (realShiftOptional->GetDataType() != DataType::DT_FLOAT) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "pse dtype must be FLOAT32 when pseType is 2 or 3.");
            return ACLNN_ERR_PARAM_INVALID;
        }
        return ACLNN_SUCCESS;
    }
    if (realShiftOptional != nullptr) {
        auto pseDtype = realShiftOptional->GetDataType();
        if (pseDtype != outDtype) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "pse dtype [%s] must equal attentionOut dtype [%s].",
                    op::ToString(DataType(pseDtype)).GetString(),
                    op::ToString(DataType(outDtype)).GetString());
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    if (sinkOptional != nullptr && sinkOptional->GetDataType() != DataType::DT_FLOAT) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "sink dtype must be FLOAT32.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

/* ---- 格式校验（不支持 FRACTAL_NZ） ---- */
static bool CheckFormatV4(const aclTensor *query, const aclTensor *queryRope,
                          const aclTensor *key, const aclTensor *keyRope, const aclTensor *value,
                          const aclTensor *realShiftOptional, const aclTensor *dropMaskOptional,
                          const aclTensor *paddingMaskOptional, const aclTensor *attenMaskOptional,
                          const aclTensor *sinkOptional,
                          const aclTensor *softmaxMaxOut, const aclTensor *softmaxSumOut,
                          const aclTensor *attentionOutOut)
{
    bool ok = query->GetStorageFormat()         != op::Format::FORMAT_FRACTAL_NZ &&
              key->GetStorageFormat()           != op::Format::FORMAT_FRACTAL_NZ &&
              value->GetStorageFormat()         != op::Format::FORMAT_FRACTAL_NZ &&
              softmaxMaxOut->GetStorageFormat() != op::Format::FORMAT_FRACTAL_NZ &&
              softmaxSumOut->GetStorageFormat() != op::Format::FORMAT_FRACTAL_NZ &&
              attentionOutOut->GetStorageFormat() != op::Format::FORMAT_FRACTAL_NZ;
    if (!ok) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "Input/output format does not support FRACTAL_NZ.");
        return false;
    }
    if (queryRope != nullptr)  { ok = ok && (queryRope->GetStorageFormat()  != op::Format::FORMAT_FRACTAL_NZ); }
    if (keyRope   != nullptr)  { ok = ok && (keyRope->GetStorageFormat()    != op::Format::FORMAT_FRACTAL_NZ); }
    if (realShiftOptional)     { ok = ok && (realShiftOptional->GetStorageFormat() != op::Format::FORMAT_FRACTAL_NZ); }
    if (dropMaskOptional)      { ok = ok && (dropMaskOptional->GetStorageFormat()  != op::Format::FORMAT_FRACTAL_NZ); }
    if (paddingMaskOptional)   { ok = ok && (paddingMaskOptional->GetStorageFormat() != op::Format::FORMAT_FRACTAL_NZ); }
    if (attenMaskOptional)     { ok = ok && (attenMaskOptional->GetStorageFormat()   != op::Format::FORMAT_FRACTAL_NZ); }
    if (sinkOptional)          { ok = ok && (sinkOptional->GetStorageFormat() != op::Format::FORMAT_FRACTAL_NZ); }
    if (!ok) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Optional input format does not support FRACTAL_NZ.");
    }
    return ok;
}

} // anonymous namespace

/* ======================================================================
 * 第一段接口实现
 * ====================================================================== */
aclnnStatus aclnnFlashAttentionScoreV4GetWorkspaceSize(
    const aclTensor   *query,
    const aclTensor   *key,
    const aclTensor   *value,
    const aclTensor   *realShiftOptional,
    const aclTensor   *dropMaskOptional,
    const aclTensor   *paddingMaskOptional,
    const aclTensor   *attenMaskOptional,
    const aclTensor   *queryRopeOptional,
    const aclTensor   *keyRopeOptional,
    const aclTensor   *dScaleQOptional,
    const aclTensor   *dScaleKOptional,
    const aclTensor   *dScaleVOptional,
    const aclTensor   *sinkOptional,
    const aclIntArray *prefixOptional,
    const aclIntArray *actualSeqQLenOptional,
    const aclIntArray *actualSeqKvLenOptional,
    const aclIntArray *qStartIdxOptional,
    const aclIntArray *kvStartIdxOptional,
    double             scaleValue,
    double             keepProb,
    int64_t            preTokens,
    int64_t            nextTokens,
    int64_t            headNum,
    char              *inputLayout,
    int64_t            innerPrecise,
    int64_t            sparseMode,
    int64_t            outDtype,
    int64_t            pseType,
    char              *softmaxOutLayout,
    int64_t            seed,
    int64_t            offset,
    const aclTensor   *softmaxMaxOut,
    const aclTensor   *softmaxSumOut,
    const aclTensor   *softmaxOutOut,
    const aclTensor   *attentionOutOut,
    uint64_t          *workspaceSize,
    aclOpExecutor    **executor)
{
    /* ---- 必选参数判空 ---- */
    CHECK_RET(CheckFaParam(query, key, value, inputLayout, softmaxMaxOut, softmaxSumOut, attentionOutOut,
                           workspaceSize, executor) == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_NULLPTR);

    L2_DFX_PHASE_1(aclnnFlashAttentionScoreV4,
                   DFX_IN(query, key, value, realShiftOptional, dropMaskOptional, paddingMaskOptional,
                          attenMaskOptional, queryRopeOptional, keyRopeOptional,
                          dScaleQOptional, dScaleKOptional, dScaleVOptional, sinkOptional,
                          prefixOptional, actualSeqQLenOptional, actualSeqKvLenOptional,
                          qStartIdxOptional, kvStartIdxOptional,
                          scaleValue, keepProb, preTokens, nextTokens, headNum,
                          inputLayout, innerPrecise, sparseMode, outDtype, pseType, seed, offset),
                   DFX_OUT(softmaxMaxOut, softmaxSumOut, softmaxOutOut, attentionOutOut));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    /* ---- 全空输出时快速返回 ---- */
    if (softmaxMaxOut->IsEmpty() && softmaxSumOut->IsEmpty() && attentionOutOut->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    /* ---- 格式校验 ---- */
    CHECK_RET(CheckFormatV4(query, queryRopeOptional, key, keyRopeOptional, value,
                            realShiftOptional, dropMaskOptional, paddingMaskOptional,
                            attenMaskOptional, sinkOptional,
                            softmaxMaxOut, softmaxSumOut, attentionOutOut),
              ACLNN_ERR_PARAM_INVALID);

    /* ---- 非量化数据类型校验 ---- */
    CHECK_RET(InputDtypeCheckV4(query, key, value, attentionOutOut, realShiftOptional, pseType, sinkOptional)
                  == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);

    /* ---- 解析输入 shape ---- */
    FaShapeInfo shapeInfo;
    CHECK_RET(AnalysisInput(query, key, value, inputLayout, headNum, shapeInfo,
                            actualSeqQLenOptional, actualSeqKvLenOptional) == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);

    aclOpExecutor *l0Executor = uniqueExecutor.get();

    /* ---- Contiguous ---- */
    CHECK_RET(Contiguous(query, key, value, realShiftOptional, dropMaskOptional,
                         paddingMaskOptional, attenMaskOptional,
                         queryRopeOptional, keyRopeOptional, sinkOptional,
                         dScaleQOptional, dScaleKOptional, dScaleVOptional,
                         l0Executor) == ACLNN_SUCCESS,
              ACLNN_ERR_INNER_NULLPTR);

    /* ---- Pad / reshape / transpose（stride 受限硬件） ---- */
    CHECK_RET(PreprocessQKV(query, key, value, shapeInfo, l0Executor) == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_NULLPTR);

    /* ---- sink shape == [0] 时置空 ---- */
    if (sinkOptional != nullptr &&
        sinkOptional->GetViewShape().GetDimNum() == 1 &&
        sinkOptional->GetViewShape()[0] == 0) {
        sinkOptional = nullptr;
    }

    /* ---- 调用 FlashAttentionScoreV4 L0 算子 ---- */
    auto l0Outs = l0op::FlashAttentionScoreV4(
        query, key, value,
        realShiftOptional, dropMaskOptional, paddingMaskOptional, attenMaskOptional,
        queryRopeOptional, keyRopeOptional,
        dScaleQOptional, dScaleKOptional, dScaleVOptional, sinkOptional,
        prefixOptional, actualSeqQLenOptional, actualSeqKvLenOptional,
        qStartIdxOptional, kvStartIdxOptional,
        scaleValue, keepProb, preTokens, nextTokens, headNum,
        shapeInfo.l0InputLayoutStr.c_str(),
        innerPrecise, sparseMode, outDtype, pseType, softmaxOutLayout, seed, offset,
        l0Executor);

    auto l0SoftmaxMax = l0Outs[0];
    auto l0SoftmaxSum = l0Outs[1];
    /* l0Outs[2] = softmaxOut（空，不使用） */
    auto l0AttnOut    = l0Outs[3];

    if (l0SoftmaxMax == nullptr || l0SoftmaxSum == nullptr || l0AttnOut == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR,
                "l0SoftmaxMax / l0SoftmaxSum / l0AttnOut is null, check input validity.");
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_ERR_PARAM_NULLPTR;
    }

    /* ---- Postprocess（反变换） ---- */
    CHECK_RET(Postprocess(l0AttnOut, attentionOutOut, shapeInfo, l0Executor) == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_NULLPTR);

    /* ---- 将 L0 输出拷贝至调用方传入的 tensor ---- */
    auto vc0 = l0op::ViewCopy(l0SoftmaxMax, softmaxMaxOut, l0Executor);
    OP_CHECK(vc0 != nullptr,
             OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "ViewCopy softmaxMax failed"),
             return ACLNN_ERR_PARAM_NULLPTR);
    auto vc1 = l0op::ViewCopy(l0SoftmaxSum, softmaxSumOut, l0Executor);
    OP_CHECK(vc1 != nullptr,
             OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "ViewCopy softmaxSum failed"),
             return ACLNN_ERR_PARAM_NULLPTR);
    auto vc3 = l0op::ViewCopy(l0AttnOut, attentionOutOut, l0Executor);
    OP_CHECK(vc3 != nullptr,
             OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "ViewCopy attentionOut failed"),
             return ACLNN_ERR_PARAM_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

/* ======================================================================
 * 第二段接口实现（固定写法）
 * ====================================================================== */
aclnnStatus aclnnFlashAttentionScoreV4(void *workspace, uint64_t workspaceSize,
                                       aclOpExecutor *executor, const aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnFlashAttentionScoreV4);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
