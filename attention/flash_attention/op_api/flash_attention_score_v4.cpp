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
 * \file flash_attention_score_v4.cpp
 * \brief FlashAttentionScoreV4 L0 (level-0) 算子实现
 *
 * 参考来源：
 *   flash_attention_score/op_api/flash_attention_score.cpp
 *   — 复用 L0 op 注册模式（OP_TYPE_REGISTER / L0_DFX / AllocTensor /
 *     ConvertToTensor / INFER_SHAPE / ADD_TO_LAUNCHER_LIST_AICORE）。
 *   — 调整参数列表：移除 p_scale（非量化），调整可选输入顺序，
 *     属性顺序改为 V4 接口定义（out_dtype 调至 pse_type 前）。
 *
 * 说明：
 *   - 所有可选 tensor 为 nullptr 时自动分配空 tensor（大小为 0）。
 *   - 所有可选 aclIntArray 为 nullptr 时自动分配空 INT64 tensor。
 *   - 输出 tensor（softmaxMax/softmaxSum/softmaxOut/attentionOut）
 *     由框架在 INFER_SHAPE 阶段推导 shape 后分配。
 *   - V4 非量化不使用 p_scale，不进行 FP8 outputDtype 映射。
 */

#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(FlashAttentionScoreV4);

const std::array<const aclTensor *, 4> FlashAttentionScoreV4(
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
    int64_t            preTockens,
    int64_t            nextTockens,
    int64_t            headNum,
    const char        *inputLayout,
    int64_t            innerPrecise,
    int64_t            sparseMode,
    int64_t            outDtype,
    int64_t            pseType,
    const char        *softmaxOutLayout,
    int64_t            seed,
    int64_t            offset,
    aclOpExecutor     *executor)
{
    L0_DFX(FlashAttentionScoreV4,
            query, key, value,
            realShiftOptional, dropMaskOptional, paddingMaskOptional, attenMaskOptional,
            queryRopeOptional, keyRopeOptional,
            dScaleQOptional, dScaleKOptional, dScaleVOptional, sinkOptional,
            prefixOptional, actualSeqQLenOptional, actualSeqKvLenOptional,
            qStartIdxOptional, kvStartIdxOptional,
            scaleValue, keepProb, preTockens, nextTockens, headNum, inputLayout,
            innerPrecise, sparseMode, outDtype, pseType, softmaxOutLayout, seed, offset);

    /* ---- 可选 tensor 为 nullptr 时分配空 tensor ---- */
    if (realShiftOptional == nullptr) {
        realShiftOptional = executor->AllocTensor(query->GetDataType(), Format::FORMAT_ND, Format::FORMAT_ND);
    }
    if (dropMaskOptional == nullptr) {
        dropMaskOptional = executor->AllocTensor(DataType::DT_UINT8, Format::FORMAT_ND, Format::FORMAT_ND);
    }
    if (paddingMaskOptional == nullptr) {
        paddingMaskOptional = executor->AllocTensor(query->GetDataType(), Format::FORMAT_ND, Format::FORMAT_ND);
    }
    if (attenMaskOptional == nullptr) {
        attenMaskOptional = executor->AllocTensor(DataType::DT_BOOL, Format::FORMAT_ND, Format::FORMAT_ND);
    }
    if (queryRopeOptional == nullptr) {
        queryRopeOptional = executor->AllocTensor(query->GetDataType(), Format::FORMAT_ND, Format::FORMAT_ND);
    }
    if (keyRopeOptional == nullptr) {
        keyRopeOptional = executor->AllocTensor(key->GetDataType(), Format::FORMAT_ND, Format::FORMAT_ND);
    }
    /* 量化参数：非量化场景传 nullptr，统一分配为空 DT_FLOAT tensor */
    if (dScaleQOptional == nullptr) {
        dScaleQOptional = executor->AllocTensor(DataType::DT_FLOAT, Format::FORMAT_ND, Format::FORMAT_ND);
    }
    if (dScaleKOptional == nullptr) {
        dScaleKOptional = executor->AllocTensor(DataType::DT_FLOAT, Format::FORMAT_ND, Format::FORMAT_ND);
    }
    if (dScaleVOptional == nullptr) {
        dScaleVOptional = executor->AllocTensor(DataType::DT_FLOAT, Format::FORMAT_ND, Format::FORMAT_ND);
    }
    if (sinkOptional == nullptr) {
        sinkOptional = executor->AllocTensor(DataType::DT_FLOAT, Format::FORMAT_ND, Format::FORMAT_ND);
    }

    /* ---- 可选 IntArray 转 Tensor，nullptr 时分配空 INT64 tensor ---- */
    const aclTensor *prefixTensor = nullptr;
    if (prefixOptional) {
        prefixTensor = executor->ConvertToTensor(prefixOptional, DataType::DT_INT64);
        const_cast<aclTensor *>(prefixTensor)->SetStorageFormat(Format::FORMAT_ND);
        const_cast<aclTensor *>(prefixTensor)->SetViewFormat(Format::FORMAT_ND);
        const_cast<aclTensor *>(prefixTensor)->SetOriginalFormat(Format::FORMAT_ND);
    } else {
        prefixTensor = executor->AllocTensor(DataType::DT_INT64, Format::FORMAT_ND, Format::FORMAT_ND);
    }

    const aclTensor *actualSeqQLen = nullptr;
    if (actualSeqQLenOptional) {
        actualSeqQLen = executor->ConvertToTensor(actualSeqQLenOptional, DataType::DT_INT64);
        const_cast<aclTensor *>(actualSeqQLen)->SetStorageFormat(Format::FORMAT_ND);
        const_cast<aclTensor *>(actualSeqQLen)->SetViewFormat(Format::FORMAT_ND);
        const_cast<aclTensor *>(actualSeqQLen)->SetOriginalFormat(Format::FORMAT_ND);
    } else {
        actualSeqQLen = executor->AllocTensor(DataType::DT_INT64, Format::FORMAT_ND, Format::FORMAT_ND);
    }

    const aclTensor *actualSeqKvLen = nullptr;
    if (actualSeqKvLenOptional) {
        actualSeqKvLen = executor->ConvertToTensor(actualSeqKvLenOptional, DataType::DT_INT64);
        const_cast<aclTensor *>(actualSeqKvLen)->SetStorageFormat(Format::FORMAT_ND);
        const_cast<aclTensor *>(actualSeqKvLen)->SetViewFormat(Format::FORMAT_ND);
        const_cast<aclTensor *>(actualSeqKvLen)->SetOriginalFormat(Format::FORMAT_ND);
    } else {
        actualSeqKvLen = executor->AllocTensor(DataType::DT_INT64, Format::FORMAT_ND, Format::FORMAT_ND);
    }

    const aclTensor *qStartIdxTensor = nullptr;
    if (qStartIdxOptional) {
        qStartIdxTensor = executor->ConvertToTensor(qStartIdxOptional, DataType::DT_INT64);
        const_cast<aclTensor *>(qStartIdxTensor)->SetStorageFormat(Format::FORMAT_ND);
        const_cast<aclTensor *>(qStartIdxTensor)->SetViewFormat(Format::FORMAT_ND);
        const_cast<aclTensor *>(qStartIdxTensor)->SetOriginalFormat(Format::FORMAT_ND);
    } else {
        qStartIdxTensor = executor->AllocTensor(DataType::DT_INT64, Format::FORMAT_ND, Format::FORMAT_ND);
    }

    const aclTensor *kvStartIdxTensor = nullptr;
    if (kvStartIdxOptional) {
        kvStartIdxTensor = executor->ConvertToTensor(kvStartIdxOptional, DataType::DT_INT64);
        const_cast<aclTensor *>(kvStartIdxTensor)->SetStorageFormat(Format::FORMAT_ND);
        const_cast<aclTensor *>(kvStartIdxTensor)->SetViewFormat(Format::FORMAT_ND);
        const_cast<aclTensor *>(kvStartIdxTensor)->SetOriginalFormat(Format::FORMAT_ND);
    } else {
        kvStartIdxTensor = executor->AllocTensor(DataType::DT_INT64, Format::FORMAT_ND, Format::FORMAT_ND);
    }

    /* ---- 输出 tensor 预分配 ---- */
    auto softmaxMaxOut  = executor->AllocTensor(DataType::DT_FLOAT, Format::FORMAT_ND, Format::FORMAT_ND);
    auto softmaxSumOut  = executor->AllocTensor(DataType::DT_FLOAT, Format::FORMAT_ND, Format::FORMAT_ND);
    /* 非量化：outputDtype 直接使用 query 的数据类型 */
    DataType outputDtype = query->GetDataType();
    auto softmaxOutOut  = executor->AllocTensor(outputDtype, Format::FORMAT_ND, Format::FORMAT_ND);
    auto attentionOutOut = executor->AllocTensor(outputDtype, Format::FORMAT_ND, Format::FORMAT_ND);

    /* ---- InferShape ----
     * 输入顺序与 def.cpp 中的 Input 声明顺序严格一致：
     *   0:query, 1:key, 2:value, 3:real_shift, 4:drop_mask, 5:padding_mask,
     *   6:atten_mask, 7:query_rope, 8:key_rope,
     *   9:d_scale_q, 10:d_scale_k, 11:d_scale_v, 12:sink,
     *   13:prefix, 14:actual_seq_qlen, 15:actual_seq_kvlen,
     *   16:q_start_idx, 17:kv_start_idx
     * 属性顺序与 def.cpp 中的 Attr 声明顺序严格一致（idx 0-12）：
     *   scale_value(float), keep_prob(float), pre_tockens, next_tockens, head_num,
     *   input_layout, inner_precise, sparse_mode, out_dtype, pse_type,
     *   softmax_out_layout, seed, offset
     * ---- */
    auto ret = INFER_SHAPE(FlashAttentionScoreV4,
                           OP_INPUT(query, key, value,
                                    realShiftOptional, dropMaskOptional, paddingMaskOptional, attenMaskOptional,
                                    queryRopeOptional, keyRopeOptional,
                                    dScaleQOptional, dScaleKOptional, dScaleVOptional, sinkOptional,
                                    prefixTensor, actualSeqQLen, actualSeqKvLen,
                                    qStartIdxTensor, kvStartIdxTensor),
                           OP_OUTPUT(softmaxMaxOut, softmaxSumOut, softmaxOutOut, attentionOutOut),
                           OP_ATTR(static_cast<float>(scaleValue), static_cast<float>(keepProb),
                                   preTockens, nextTockens, headNum, inputLayout,
                                   innerPrecise, sparseMode, outDtype, pseType,
                                   softmaxOutLayout, seed, offset));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "FlashAttentionScoreV4 InferShape failed.");
        return {nullptr, nullptr, nullptr, nullptr};
    }

    /* ---- 将算子加入执行器 ---- */
    ret = ADD_TO_LAUNCHER_LIST_AICORE(
        FlashAttentionScoreV4,
        OP_INPUT(query, key, value,
                 realShiftOptional, dropMaskOptional, paddingMaskOptional, attenMaskOptional,
                 queryRopeOptional, keyRopeOptional,
                 dScaleQOptional, dScaleKOptional, dScaleVOptional, sinkOptional,
                 prefixTensor, actualSeqQLen, actualSeqKvLen,
                 qStartIdxTensor, kvStartIdxTensor),
        OP_OUTPUT(softmaxMaxOut, softmaxSumOut, softmaxOutOut, attentionOutOut),
        OP_ATTR(static_cast<float>(scaleValue), static_cast<float>(keepProb),
                preTockens, nextTockens, headNum, inputLayout,
                innerPrecise, sparseMode, outDtype, pseType,
                softmaxOutLayout, seed, offset));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "FlashAttentionScoreV4 launch kernel failed.");
        return {nullptr, nullptr, nullptr, nullptr};
    }

    return {softmaxMaxOut, softmaxSumOut, softmaxOutOut, attentionOutOut};
}

} // namespace l0op
