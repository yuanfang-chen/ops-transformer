/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
 * the software repository for the full text of the License.
 */

#include <iostream>
#include <vector>
#include <torch/library.h>
#include <ATen/Operators.h>
#include "torch_npu/csrc/framework/utils/OpPreparation.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "op_host/incre_flash_attention_tiling_impl.h"
#include "op_kernel/incre_flash_attention_arch32.h"
namespace custom {
#include <iostream>
#include <iomanip>
#include <sstream>

void print(const optiling::IncreFlashAttentionTilingData& data) {
    std::cout << "=== IncreFlashAttentionTilingData Debug Output ===\n";

    // Base Params
    std::cout << "\n--- Base Parameters ---\n";
    std::cout << "batchSize: " << data.baseParams.get_batchSize() << "\n";
    std::cout << "seqSize: " << data.baseParams.get_seqSize() << "\n";
    std::cout << "qSeqSize: " << data.baseParams.get_qSeqSize() << "\n";
    std::cout << "headSize: " << data.baseParams.get_headSize() << "\n";
    std::cout << "headSizeV: " << data.baseParams.get_headSizeV() << "\n";
    std::cout << "blockSize: " << data.baseParams.get_blockSize() << "\n";
    std::cout << "maxBlockNumPerBatch: " << data.baseParams.get_maxBlockNumPerBatch() << "\n";
    std::cout << "maxBlockNumPerSeq: " << data.baseParams.get_maxBlockNumPerSeq() << "\n";
    std::cout << "scaleValue: " << std::fixed << std::setprecision(6) << data.baseParams.get_scaleValue() << "\n";
    std::cout << "kvHeadNum: " << data.baseParams.get_kvHeadNum() << "\n";
    std::cout << "headNumRatio: " << data.baseParams.get_headNumRatio() << "\n";
    std::cout << "qHeadNum: " << data.baseParams.get_qHeadNum() << "\n";
    std::cout << "nNumOfQInOneGroup: " << data.baseParams.get_nNumOfQInOneGroup() << "\n";
    std::cout << "batchContinuousFlag: " << data.baseParams.get_batchContinuousFlag() << "\n";
    std::cout << "pseShiftFlag: " << data.baseParams.get_pseShiftFlag() << "\n";
    std::cout << "pseShiftB: " << data.baseParams.get_pseShiftB() << "\n";
    std::cout << "pseShiftS: " << data.baseParams.get_pseShiftS() << "\n";
    std::cout << "pseShiftS0: " << data.baseParams.get_pseShiftS0() << "\n";
    std::cout << "selectWithByteMaskTmpMinSize: " << data.baseParams.get_selectWithByteMaskTmpMinSize() << "\n";
    std::cout << "actualLenQDims: " << data.baseParams.get_actualLenQDims() << "\n";
    std::cout << "actualLenDims: " << data.baseParams.get_actualLenDims() << "\n";
    std::cout << "qPaddingFlag: " << data.baseParams.get_qPaddingFlag() << "\n";
    std::cout << "kvPaddingFlag: " << data.baseParams.get_kvPaddingFlag() << "\n";
    std::cout << "msdIterNum: " << data.baseParams.get_msdIterNum() << "\n";
    std::cout << "l2CacheOffFlag: " << data.baseParams.get_l2CacheOffFlag() << "\n";
    std::cout << "antiquantPerTensorFlag: " << data.baseParams.get_antiquantPerTensorFlag() << "\n";
    std::cout << "antiquantPerHeadFlag: " << data.baseParams.get_antiquantPerHeadFlag() << "\n";
    std::cout << "antiquantParamsInPagedAttentionFlag: " << data.baseParams.get_antiquantParamsInPagedAttentionFlag() << "\n";
    std::cout << "attenMaskFlag: " << data.baseParams.get_attenMaskFlag() << "\n";
    std::cout << "attenMaskBatch: " << data.baseParams.get_attenMaskBatch() << "\n";
    std::cout << "attenMaskQSize: " << data.baseParams.get_attenMaskQSize() << "\n";
    std::cout << "attenMaskSize: " << data.baseParams.get_attenMaskSize() << "\n";
    std::cout << "softmaxLseFlag: " << data.baseParams.get_softmaxLseFlag() << "\n";
    std::cout << "totalBlockNum: " << data.baseParams.get_totalBlockNum() << "\n";
    std::cout << "paKvShapeType: " << data.baseParams.get_paKvShapeType() << "\n";
    std::cout << "antiqSeqSize: " << data.baseParams.get_antiqSeqSize() << "\n";
    std::cout << "preToken: " << data.baseParams.get_preToken() << "\n";
    std::cout << "nextToken: " << data.baseParams.get_nextToken() << "\n";
    std::cout << "isRowInvalid: " << data.baseParams.get_isRowInvalid() << "\n";
    std::cout << "sparseMode: " << data.baseParams.get_sparseMode() << "\n";
    std::cout << "slidingFlag: " << data.baseParams.get_slidingFlag() << "\n";
    std::cout << "windowSize: " << data.baseParams.get_windowSize() << "\n";

    // Split KV Params
    std::cout << "\n--- Split KV Parameters ---\n";
    std::cout << "s2: " << data.splitKVParams.get_s2() << "\n";
    std::cout << "sInnerLoopSize: " << data.splitKVParams.get_sInnerLoopSize() << "\n";
    std::cout << "accumOutSize: " << data.splitKVParams.get_accumOutSize() << "\n";
    std::cout << "logSumExpSize: " << data.splitKVParams.get_logSumExpSize() << "\n";

    // Core Params
    std::cout << "\n--- Core Parameters ---\n";
    std::cout << "coreSidxEnd (first 5): ";
    for (int i = 0; i < 50; ++i) {
        std::cout << data.increFlashAttentionCoreParams.coreSidxEnd[i] << " ";
    }
    std::cout << "\n";

    // Single Core Params
    std::cout << "\n--- Single Core Parameters ---\n";
    std::cout << "sInnerLoopTimes: " << data.increFlashAttentionSingleCoreParams.get_sInnerLoopTimes() << "\n";
    std::cout << "singleProcessSInnerSize: " << data.increFlashAttentionSingleCoreParams.get_singleProcessSInnerSize() << "\n";
    std::cout << "singleProcessSInnerSizeTail: " << data.increFlashAttentionSingleCoreParams.get_singleProcessSInnerSizeTail() << "\n";
    std::cout << "usedCoreNum: " << data.increFlashAttentionSingleCoreParams.get_usedCoreNum() << "\n";
    std::cout << "formerCoreNum: " << data.increFlashAttentionSingleCoreParams.get_formerCoreNum() << "\n";
    std::cout << "blockSplitBn2Range: " << data.increFlashAttentionSingleCoreParams.get_blockSplitBn2Range() << "\n";
    std::cout << "tailSplitedBatchRange: " << data.increFlashAttentionSingleCoreParams.get_tailSplitedBatchRange() << "\n";
    std::cout << "groupSplitSize: " << data.increFlashAttentionSingleCoreParams.get_groupSplitSize() << "\n";
    std::cout << "s1SplitSize: " << data.increFlashAttentionSingleCoreParams.get_s1SplitSize() << "\n";

    // Tensor Size
    std::cout << "\n--- Single Core Tensor Size ---\n";
    std::cout << "mmResUbSize: " << data.increFlashAttentionSingleCoreTensorSize.get_mmResUbSize() << "\n";
    std::cout << "bmm2ResUbSize: " << data.increFlashAttentionSingleCoreTensorSize.get_bmm2ResUbSize() << "\n";

    // Output Params
    std::cout << "\n--- Output Parameters ---\n";
    std::cout << "isPerChnOut: " << data.outputParams.get_isPerChnOut() << "\n";
    std::cout << "isOutQuantTypeBf16: " << data.outputParams.get_isOutQuantTypeBf16() << "\n";
    std::cout << "singleCoreSize: " << data.outputParams.get_singleCoreSize() << "\n";
    std::cout << "singleCoreLseSize: " << data.outputParams.get_singleCoreLseSize() << "\n";
    std::cout << "totalOutputSize: " << data.outputParams.get_totalOutputSize() << "\n";
    std::cout << "totalLseOutputSize: " << data.outputParams.get_totalLseOutputSize() << "\n";
    std::cout << "needInit: " << data.outputParams.get_needInit() << "\n";
    std::cout << "isBSNDOut: " << data.outputParams.get_isBSNDOut() << "\n";

    std::cout << "\n=== End of Debug Output ===\n";
}
#define LAUNCH_INCRE_FA(FD, LAYOUT, ANTIQ)                                                                 \
    incre_flash_attention<FD, LAYOUT, ANTIQ><<<blockDim, nullptr, aclstream>>>(                            \
        (GM_ADDR)(query.data_ptr()),                                                                       \
        (GM_ADDR)(key.data_ptr()),                                                                         \
        (GM_ADDR)(value.data_ptr()),                                                                       \
        (GM_ADDR)(pse_shift.has_value() ? pse_shift->data_ptr() : nullptr),                                \
        (GM_ADDR)(atten_mask.has_value() ? atten_mask->data_ptr() : nullptr),                              \
        (GM_ADDR)(actual_seq_qlen.has_value() ? actual_seq_qlen->data_ptr() : nullptr),                     \
        (GM_ADDR)(actual_seq_kvlen.has_value() ? actual_seq_kvlen->data_ptr() : nullptr),                   \
        (GM_ADDR)(nullptr),  /* deqScale1 */                                                               \
        (GM_ADDR)(nullptr),  /* quantScale1 */                                                             \
        (GM_ADDR)(nullptr),  /* deqScale2 */                                                               \
        (GM_ADDR)(nullptr),  /* quantScale2 */                                                             \
        (GM_ADDR)(nullptr),  /* quantOffset2 */                                                            \
        (GM_ADDR)(nullptr),  /* antiquantScale */                                                          \
        (GM_ADDR)(nullptr),  /* antiquantOffset */                                                         \
        (GM_ADDR)(block_table.has_value() ? block_table->data_ptr() : nullptr),                            \
        (GM_ADDR)(nullptr),  /* queryPaddingSize */                                                        \
        (GM_ADDR)(nullptr),  /* kvPaddingSize */                                                           \
        (GM_ADDR)(dequant_scale_key.has_value() ? dequant_scale_key->data_ptr() : nullptr),                \
        (GM_ADDR)(nullptr),  /* keyAntiquantOffset */                                                      \
        (GM_ADDR)(dequant_scale_value.has_value() ? dequant_scale_value->data_ptr() : nullptr),            \
        (GM_ADDR)(nullptr),  /* valueAntiquantOffset */                                                    \
        (GM_ADDR)(nullptr),  /* keySharedPrefix */                                                         \
        (GM_ADDR)(nullptr),  /* valueSharedPrefix */                                                       \
        (GM_ADDR)(nullptr),  /* actualSharedPrefixLen */                                                   \
        (GM_ADDR)(query_rope.has_value() ? query_rope->data_ptr() : nullptr),                              \
        (GM_ADDR)(key_rope.has_value() ? key_rope->data_ptr() : nullptr),                                  \
        (GM_ADDR)(dequant_scale_key_rope.has_value() ? dequant_scale_key_rope->data_ptr() : nullptr),      \
        (GM_ADDR)(dequant_scale_query.has_value() ? dequant_scale_query->data_ptr() : nullptr),            \
        (GM_ADDR)(metadata.has_value() ? metadata->data_ptr() : nullptr),            \
        (GM_ADDR)(output.data_ptr()),                                                                      \
        (GM_ADDR)(softmax_lse.data_ptr()),                                                                 \
        (GM_ADDR)(workspaceTensor.data_ptr()),                                                             \
        tilingData)

const static int FLASH_THRESHOLD = 512;
const static int64_t PFA_SPARSE_HIGH_PRECISION_NO_MASK = 10;
const static int64_t PFA_SPARSE_HIGH_PRECISION_BAND = 14;
const static int64_t DIM_0 = 0;
const static int64_t DIM_1 = 1;
const static int64_t DIM_2 = 2;
const static int64_t DIM_3 = 3;
const static int64_t DIM_4 = 4;
const static int64_t PA_BBH_DIMS = 3;
const static int64_t PA_BNBD_DIMS = 4;
const static int64_t PA_NZ_DIMS = 5;
using namespace at_npu::native;
using namespace optiling;
using npu_preparation = at_npu::native::OpPreparation;

std::tuple<at::Tensor, at::Tensor>
construct_fia_output_tensor_v2(const at::Tensor &query, const at::Tensor &value, std::string input_layout_str,
                               const c10::optional<at::Tensor> &quant_scale_out,
                               const c10::optional<at::Tensor> &block_table, int64_t num_query_heads,
                               int64_t num_key_value_heads, // 增加kvhead用于计算BBH情况下的D
                               bool return_softmax_lse, const c10::optional<at::Tensor> &query_rope)
{
    at::Tensor output;
    int64_t batchSize = 1;
    int64_t qsSize = 1;
    at::Tensor tmp_output = at::empty_like(query);
    if (input_layout_str == "BNSD_BSND") {
        tmp_output = at::empty({query.size(DIM_0), query.size(DIM_2), query.size(DIM_1), query.size(DIM_3)},
                               query.options().dtype(query.dtype()));
        batchSize = query.size(DIM_0);
        qsSize = query.size(DIM_2);
    } else if (input_layout_str == "BNSD_NBSD") {
        tmp_output = at::empty({query.size(DIM_1), query.size(DIM_0), query.size(DIM_2), query.size(DIM_3)},
                               query.options().dtype(query.dtype()));
        batchSize = query.size(DIM_0);
        qsSize = query.size(DIM_2);
    } else if (input_layout_str == "BSND_NBSD") {
        tmp_output = at::empty({query.size(DIM_2), query.size(DIM_0), query.size(DIM_1), query.size(DIM_3)},
                               query.options().dtype(query.dtype()));
        batchSize = query.size(DIM_0);
        qsSize = query.size(DIM_1);
    } else if (input_layout_str == "BSH_NBSD") {
        // TORCH_CHECK(num_query_heads > 0, "The num_query_heads should be greater than zero, but got ",
        // num_query_heads, OPS_ERROR(ErrCode::PARAM));
        tmp_output =
            at::empty({num_query_heads, query.size(DIM_0), query.size(DIM_1), query.size(DIM_2) / num_query_heads},
                      query.options().dtype(query.dtype()));
        batchSize = query.size(DIM_0);
        qsSize = query.size(DIM_1);
    } else if (input_layout_str == "TND_NTD") {
        tmp_output =
            at::empty({query.size(DIM_1), query.size(DIM_0), query.size(DIM_2)}, query.options().dtype(query.dtype()));
    } else if (input_layout_str == "NSD") {
        batchSize = 1;
        qsSize = query.size(DIM_1);
    } else if (input_layout_str == "BSH") {
        batchSize = query.size(DIM_0);
        qsSize = query.size(DIM_1);
    } else if (input_layout_str == "BSND") {
        batchSize = query.size(DIM_0);
        qsSize = query.size(DIM_1);
    } else if (input_layout_str == "BNSD") {
        batchSize = query.size(DIM_0);
        qsSize = query.size(DIM_2);
    } else if (input_layout_str == "TND") {
        int64_t kv_dim = value.dim();
        if (block_table.has_value()) {   // IFA目前TND只支持PA场景，PFA目前TND只支持非PA场景
            if (kv_dim == PA_BBH_DIMS) { // BBH的情况下，D = H / N
                tmp_output = at::empty({query.size(DIM_0), query.size(DIM_1), value.size(DIM_2) / num_key_value_heads},
                                       query.options().dtype(query.dtype()));
            } else if (kv_dim == PA_BNBD_DIMS) { // BNBD情况下取D
                tmp_output = at::empty({query.size(DIM_0), query.size(DIM_1), value.size(DIM_3)},
                                       query.options().dtype(query.dtype()));
            } else if (kv_dim == PA_NZ_DIMS) { // blockNum, N, D / 16, blockSize, 16取DIM2*DIM4
                tmp_output = at::empty({query.size(DIM_0), query.size(DIM_1), value.size(DIM_2) * value.size(DIM_4)},
                                       query.options().dtype(query.dtype()));
            } else {
                tmp_output = at::empty({query.size(DIM_0), query.size(DIM_1), value.size(DIM_2)},
                                       query.options().dtype(query.dtype()));
            }
        } else {
            tmp_output = at::empty({query.size(DIM_0), query.size(DIM_1), value.size(DIM_2)},
                                   query.options().dtype(query.dtype()));
        }
    } else if (input_layout_str == "NTD_TND") {
        int64_t kv_dim = value.dim();
        if (kv_dim == 0) {
            kv_dim = query.dim();
        }
        if (block_table.has_value()) {   // pa场景
            if (kv_dim == PA_BBH_DIMS) { // BBH的情况下，D = H / N
                tmp_output = at::empty({query.size(DIM_1), query.size(DIM_0), value.size(DIM_2) / num_key_value_heads},
                                       query.options().dtype(query.dtype()));
            } else if (kv_dim == PA_BNBD_DIMS) { // BNBD情况下取D
                tmp_output = at::empty({query.size(DIM_1), query.size(DIM_0), value.size(DIM_3)},
                                       query.options().dtype(query.dtype()));
            } else if (kv_dim == PA_NZ_DIMS) { // blockNum, N, D / 16, blockSize, 16取DIM2*DIM4
                tmp_output = at::empty({query.size(DIM_1), query.size(DIM_0), value.size(DIM_2) * value.size(DIM_4)},
                                       query.options().dtype(query.dtype()));
            } else {
                tmp_output = at::empty({query.size(DIM_1), query.size(DIM_0), value.size(DIM_2)},
                                       query.options().dtype(query.dtype()));
            }
        } else {
            tmp_output = at::empty({query.size(DIM_1), query.size(DIM_0), value.size(DIM_2)},
                                   query.options().dtype(query.dtype()));
        }
    }
    if (quant_scale_out.has_value()) {
        output = at::empty(tmp_output.sizes(), c10::dtype(c10::ScalarType::Char));
    } else if (query.dtype() == at::kChar) {
        if (query_rope.has_value()) {
            const at::Tensor &query_rope_tensor = c10::value_or_else(query_rope, [] { return at::Tensor(); });
            output = at::empty(tmp_output.sizes(), c10::dtype(query_rope_tensor.dtype()));
        } else {
            output = at::empty(tmp_output.sizes(), c10::dtype(c10::ScalarType::Half));
        }
    } else {
        output = at::empty_like(tmp_output);
    }

    auto lse_opts = output.options().dtype(c10::ScalarType::Float);
    at::Tensor softmax_lse;
    if (input_layout_str == "TND") {
        if (block_table.has_value()) {    // IFA目前TND只支持PA场景，PFA目前TND只支持非PA场景
            if (query.size(DIM_2) == 0) { // 增加softmax lse的情况下，可能存在空tensor的分支
                softmax_lse = at::empty(
                    {
                        query.size(DIM_0),
                        num_query_heads,
                        0,
                    },
                    lse_opts);
            } else {
                softmax_lse = at::empty({query.size(DIM_0), num_query_heads, 1}, lse_opts);
            }
        } else {
            softmax_lse = at::empty({query.size(DIM_0), query.size(DIM_1), 1}, lse_opts);
        }
    } else if (input_layout_str == "NTD_TND") {
        if (block_table.has_value()) {    // pa场景
            if (query.size(DIM_2) == 0) { // 增加softmax lse的情况下，可能存在空tensor的分支
                softmax_lse = at::empty({query.size(DIM_1), query.size(DIM_0), 0}, lse_opts);
            } else {
                softmax_lse = at::empty({query.size(DIM_1), query.size(DIM_0), 1}, lse_opts);
            }
        } else {
            softmax_lse = at::empty({query.size(DIM_1), query.size(DIM_0), 1}, lse_opts);
        }
    } else {
        softmax_lse = at::empty({batchSize, num_query_heads, qsSize, 1}, lse_opts);
    }

    if (!return_softmax_lse) {
        softmax_lse = at::empty({0}, lse_opts);
    }
    return std::tuple<at::Tensor, at::Tensor>(output, softmax_lse);
}

RequiredParaInfo ToRequiredParaInfo(const at::Tensor &t)
{
    return RequiredParaInfo{.data = t.defined() ? t.data_ptr() : nullptr, .shape = t.sizes(), .dType = t.scalar_type()};
}

OptionalTensorParaInfo ToOptionalTensorParaInfo(const c10::optional<at::Tensor> &ot)
{
    OptionalTensorParaInfo info;
    info.hasValue = ot.has_value();
    if (info.hasValue && ot->defined()) {
        info.data = ot->data_ptr();
        info.shape = ot->sizes();
        info.dType = ot->scalar_type();
    } else {
        info.data = nullptr;
        // shape 保持默认空；dType 保持默认 Float（或按你需求改）
    }
    return info;
}

void ConvertContextToParamsIFA(
    IFAContext &ifaContext, const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
    const c10::optional<at::Tensor> &query_rope, const c10::optional<at::Tensor> &key_rope,
    const c10::optional<at::Tensor> &pse_shift, const c10::optional<at::Tensor> &atten_mask,
    const c10::optional<at::Tensor> &actual_seq_qlen, const c10::optional<at::Tensor> &actual_seq_kvlen,
    const c10::optional<at::Tensor> &block_table, const c10::optional<at::Tensor> &dequant_scale_query,
    const c10::optional<at::Tensor> &dequant_scale_key, const c10::optional<at::Tensor> &dequant_offset_key,
    const c10::optional<at::Tensor> &dequant_scale_value, const c10::optional<at::Tensor> &dequant_offset_value,
    const c10::optional<at::Tensor> &dequant_scale_key_rope, const c10::optional<at::Tensor> &quant_scale_out,
    const c10::optional<at::Tensor> &quant_offset_out, const c10::optional<at::Tensor> &learnable_sink,
    int64_t num_query_heads, int64_t num_key_value_heads, double softmax_scale, int64_t pre_tokens, int64_t next_tokens,
    const char *input_layout, int64_t sparse_mode, int64_t block_size, int64_t query_quant_mode, int64_t key_quant_mode,
    int64_t value_quant_mode, int64_t inner_precise, bool return_softmax_lse, const c10::optional<int64_t> &query_dtype,
    const c10::optional<int64_t> &key_dtype, const c10::optional<int64_t> &value_dtype,
    const c10::optional<int64_t> &query_rope_dtype, const c10::optional<int64_t> &key_rope_dtype,
    const c10::optional<int64_t> &key_shared_prefix_dtype, const c10::optional<int64_t> &value_shared_prefix_dtype,
    const c10::optional<int64_t> &dequant_scale_query_dtype, const c10::optional<int64_t> &dequant_scale_key_dtype,
    const c10::optional<int64_t> &dequant_scale_value_dtype, const c10::optional<int64_t> &dequant_scale_key_rope_dtype,
    const at::Tensor &attenOut, const at::Tensor &lseOut)
{
    // required input
    ifaContext.opName = "FusedInferAttentionScore";
    ifaContext.query = ToRequiredParaInfo(query);
    ifaContext.key = ToRequiredParaInfo(key);
    ifaContext.value = ToRequiredParaInfo(value);
    // optional input
    ifaContext.pseShift = ToOptionalTensorParaInfo(pse_shift);                           //
    ifaContext.attenMask = ToOptionalTensorParaInfo(atten_mask);                         //
    ifaContext.actualSeqLengthsQ = ToOptionalTensorParaInfo(actual_seq_qlen);             //
    ifaContext.actualSeqLengths = ToOptionalTensorParaInfo(actual_seq_kvlen);             //
    ifaContext.deqScale1 = ToOptionalTensorParaInfo(c10::nullopt);                       //
    ifaContext.quantScale1 = ToOptionalTensorParaInfo(c10::nullopt);                     //
    ifaContext.deqScale2 = ToOptionalTensorParaInfo(c10::nullopt);                       //
    ifaContext.quantScale2 = ToOptionalTensorParaInfo(quant_scale_out);                  //
    ifaContext.quantOffset2 = ToOptionalTensorParaInfo(quant_offset_out);                //
    ifaContext.antiquantScale = ToOptionalTensorParaInfo(c10::nullopt);                  //
    ifaContext.antiquantOffset = ToOptionalTensorParaInfo(c10::nullopt);                 //
    ifaContext.blockTable = ToOptionalTensorParaInfo(block_table);                       //
    ifaContext.queryPaddingSize = ToOptionalTensorParaInfo(c10::nullopt);                //
    ifaContext.kvPaddingSize = ToOptionalTensorParaInfo(c10::nullopt);                   //
    ifaContext.keyAntiquantScale = ToOptionalTensorParaInfo(dequant_scale_key);          //
    ifaContext.keyAntiquantOffset = ToOptionalTensorParaInfo(dequant_offset_key);        //
    ifaContext.valueAntiquantScale = ToOptionalTensorParaInfo(dequant_scale_value);      //
    ifaContext.valueAntiquantOffset = ToOptionalTensorParaInfo(dequant_offset_value);    //
    ifaContext.keySharedPrefix = ToOptionalTensorParaInfo(c10::nullopt);                 //
    ifaContext.valueSharedPrefix = ToOptionalTensorParaInfo(c10::nullopt);               //
    ifaContext.actualSharedPrefixLen = ToOptionalTensorParaInfo(c10::nullopt);            //?
    ifaContext.queryRope = ToOptionalTensorParaInfo(query_rope);                         //
    ifaContext.keyRope = ToOptionalTensorParaInfo(key_rope);                             //
    ifaContext.keyRopeAntiquantScale = ToOptionalTensorParaInfo(dequant_scale_key_rope); //
    ifaContext.dequantScaleQuery = ToOptionalTensorParaInfo(dequant_scale_query);        //
    ifaContext.qStartIdx = ToOptionalTensorParaInfo(c10::nullopt);                       //
    ifaContext.kvStartIdx = ToOptionalTensorParaInfo(c10::nullopt);                      //
    // attr
    ifaContext.numHeads = num_query_heads;            //
    ifaContext.preToken = pre_tokens;                 //
    ifaContext.nextToken = next_tokens;               //
    ifaContext.scaleValue = softmax_scale;            //
    ifaContext.kvHeadNums = num_key_value_heads;      //
    ifaContext.layOut = input_layout;                 //
    ifaContext.blockSize = block_size;                //
    ifaContext.innerPrecise = inner_precise;          //
    ifaContext.antiquantMode = 0;                     //
    ifaContext.softmaxLseFlag = return_softmax_lse;   //
    ifaContext.keyAntiquantMode = key_quant_mode;     //
    ifaContext.valueAntiquantMode = value_quant_mode; //
    ifaContext.sparseMode = sparse_mode;              //
    ifaContext.queryQuantMode = query_quant_mode;     //
    ifaContext.pseType = 0;                           //
    ifaContext.windowSize = 0;                        //
    // output
    ifaContext.attenOut = ToRequiredParaInfo(attenOut); //
    ifaContext.lseOut = ToRequiredParaInfo(lseOut);     //

    ifaContext.kCache.resize(1);
    ifaContext.vCache.resize(1);
    at::IntArrayRef keyShapeRef = key.sizes();
    ifaContext.kCache[0] = &keyShapeRef;
    at::IntArrayRef valueShapeRef = value.sizes();
    ifaContext.vCache[0] = &valueShapeRef;
}

std::tuple<at::Tensor, at::Tensor> npu_fused_infer_attention_score_npu(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
    const c10::optional<at::Tensor> &query_rope, const c10::optional<at::Tensor> &key_rope,
    const c10::optional<at::Tensor> &pse_shift, const c10::optional<at::Tensor> &atten_mask,
    const c10::optional<at::Tensor> &actual_seq_qlen, const c10::optional<at::Tensor> &actual_seq_kvlen,
    const c10::optional<at::Tensor> &block_table, const c10::optional<at::Tensor> &dequant_scale_query,
    const c10::optional<at::Tensor> &dequant_scale_key, const c10::optional<at::Tensor> &dequant_offset_key,
    const c10::optional<at::Tensor> &dequant_scale_value, const c10::optional<at::Tensor> &dequant_offset_value,
    const c10::optional<at::Tensor> &dequant_scale_key_rope, const c10::optional<at::Tensor> &quant_scale_out,
    const c10::optional<at::Tensor> &quant_offset_out, const c10::optional<at::Tensor> &learnable_sink, 
    const c10::optional<at::Tensor> &metadata,
    int64_t num_query_heads, int64_t num_key_value_heads, double softmax_scale, int64_t pre_tokens, int64_t next_tokens,
    c10::string_view input_layout, int64_t sparse_mode, int64_t block_size, int64_t query_quant_mode,
    int64_t key_quant_mode, int64_t value_quant_mode, int64_t inner_precise, bool return_softmax_lse,
    c10::optional<int64_t> query_dtype, c10::optional<int64_t> key_dtype, c10::optional<int64_t> value_dtype,
    c10::optional<int64_t> query_rope_dtype, c10::optional<int64_t> key_rope_dtype,
    c10::optional<int64_t> key_shared_prefix_dtype, c10::optional<int64_t> value_shared_prefix_dtype,
    c10::optional<int64_t> dequant_scale_query_dtype, c10::optional<int64_t> dequant_scale_key_dtype,
    c10::optional<int64_t> dequant_scale_value_dtype, c10::optional<int64_t> dequant_scale_key_rope_dtype)
{
    printf("start npu\n");
    // convert str
    std::string input_layout_str = std::string(input_layout);

    // construct the output tensor
    std::tuple<at::Tensor, at::Tensor> fia_output =
        construct_fia_output_tensor_v2(query, value, input_layout_str, quant_scale_out, block_table, num_query_heads,
                                       num_key_value_heads, return_softmax_lse, query_rope);
    at::Tensor output = std::get<0>(fia_output);
    at::Tensor softmax_lse = std::get<1>(fia_output);

    char input_layout_ptr[20];
    strncpy(input_layout_ptr, input_layout_str.c_str(), 20 - 1);

    IFAContext ifaContext;
    ConvertContextToParamsIFA(
        ifaContext, query, key, value, query_rope, key_rope, pse_shift, atten_mask, actual_seq_qlen, actual_seq_kvlen,
        block_table, dequant_scale_query, dequant_scale_key, dequant_offset_key, dequant_scale_value,
        dequant_offset_value, dequant_scale_key_rope, quant_scale_out, quant_offset_out, learnable_sink,
        num_query_heads, num_key_value_heads, softmax_scale, pre_tokens, next_tokens, input_layout_ptr, sparse_mode,
        block_size, query_quant_mode, key_quant_mode, value_quant_mode, inner_precise, return_softmax_lse, query_dtype,
        key_dtype, value_dtype, query_rope_dtype, key_rope_dtype, key_shared_prefix_dtype, value_shared_prefix_dtype,
        dequant_scale_query_dtype, dequant_scale_key_dtype, dequant_scale_value_dtype, dequant_scale_key_rope_dtype,
        output, softmax_lse);
    printf("covert params end\n");
    IFATiling ifaTiling;
    ifaTiling.DoSubOpTiling(ifaContext);
    // stream
    int devidx = query.device().index();
    c10_npu::NPUStream stream = c10_npu::getCurrentNPUStream(devidx);
    void *aclstream = stream.stream(false);
    // workspace
    auto workspaceTensor =
        at::empty({static_cast<long>(ifaContext.workSpaceSize)}, at::TensorOptions().dtype(at::kByte).device(query.options().device()));
    // tilingdata
    optiling::IncreFlashAttentionTilingData &tilingData = ifaContext.tilingData.tilingBase;

    uint8_t fdFlag = ifaContext.fdFlag;
    uint8_t layoutVal = ifaContext.layoutVal;
    uint8_t antiquantMode = ifaContext.antiquantMode_;
    // blockdim
    uint32_t blockDim = tilingData.increFlashAttentionSingleCoreParams.get_usedCoreNum();
    printf("blockDim set to :%d\n", blockDim);
    print(tilingData);
    auto aclCal = [&]() -> int {
        if (fdFlag == 0 && layoutVal == 0 && antiquantMode == 0) {
            LAUNCH_INCRE_FA(0, 0, 0);
        } else if (fdFlag == 0 && layoutVal == 0 && antiquantMode == 1) {
            LAUNCH_INCRE_FA(0, 0, 1);
        } else if (fdFlag == 0 && layoutVal == 1 && antiquantMode == 0) {
            LAUNCH_INCRE_FA(0, 1, 0);
        } else if (fdFlag == 0 && layoutVal == 1 && antiquantMode == 1) {
            LAUNCH_INCRE_FA(0, 1, 1);
        } else if (fdFlag == 1 && layoutVal == 0 && antiquantMode == 0) {
            LAUNCH_INCRE_FA(1, 0, 0);
        } else if (fdFlag == 1 && layoutVal == 0 && antiquantMode == 1) {
            LAUNCH_INCRE_FA(1, 0, 1);
        } else if (fdFlag == 1 && layoutVal == 1 && antiquantMode == 0) {
            LAUNCH_INCRE_FA(1, 1, 0);
        } else if (fdFlag == 1 && layoutVal == 1 && antiquantMode == 1) {
            LAUNCH_INCRE_FA(1, 1, 1);
        } else {
            printf("invalid flags: fd=%d layout=%d antiquantMode=%d\n",
                fdFlag, layoutVal, antiquantMode);
            return -1;
        }
        return 0;
    };
    at_npu::native::OpCommand::RunOpApi("FA", aclCal);

    return std::tuple<at::Tensor, at::Tensor>(output, softmax_lse);
}

// 为META设备实现前向接口
std::tuple<at::Tensor, at::Tensor> npu_fused_infer_attention_score_meta(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
    const c10::optional<at::Tensor> &query_rope, const c10::optional<at::Tensor> &key_rope,
    const c10::optional<at::Tensor> &pse_shift, const c10::optional<at::Tensor> &atten_mask,
    const c10::optional<at::Tensor> & actual_seq_qlen, const c10::optional<at::Tensor> & actual_seq_kvlen,
    const c10::optional<at::Tensor> &block_table, const c10::optional<at::Tensor> &dequant_scale_query,
    const c10::optional<at::Tensor> &dequant_scale_key, const c10::optional<at::Tensor> &dequant_offset_key,
    const c10::optional<at::Tensor> &dequant_scale_value, const c10::optional<at::Tensor> &dequant_offset_value,
    const c10::optional<at::Tensor> &dequant_scale_key_rope, const c10::optional<at::Tensor> &quant_scale_out,
    const c10::optional<at::Tensor> &quant_offset_out, const c10::optional<at::Tensor> &learnable_sink,
    const c10::optional<at::Tensor> &metadata,
    int64_t num_query_heads, int64_t num_key_value_heads, double softmax_scale, int64_t pre_tokens, int64_t next_tokens,
    c10::string_view input_layout, int64_t sparse_mode, int64_t block_size, int64_t query_quant_mode,
    int64_t key_quant_mode, int64_t value_quant_mode, int64_t inner_precise, bool return_softmax_lse,
    c10::optional<int64_t> query_dtype, c10::optional<int64_t> key_dtype, c10::optional<int64_t> value_dtype,
    c10::optional<int64_t> query_rope_dtype, c10::optional<int64_t> key_rope_dtype,
    c10::optional<int64_t> key_shared_prefix_dtype, c10::optional<int64_t> value_shared_prefix_dtype,
    c10::optional<int64_t> dequant_scale_query_dtype, c10::optional<int64_t> dequant_scale_key_dtype,
    c10::optional<int64_t> dequant_scale_value_dtype, c10::optional<int64_t> dequant_scale_key_rope_dtype)
{
    printf("start meta\n");
    // convert str
    std::string input_layout_str = std::string(input_layout);
    // construct the output tensor
    std::tuple<at::Tensor, at::Tensor> fia_output =
        construct_fia_output_tensor_v2(query, value, input_layout_str, quant_scale_out, block_table, num_query_heads,
                                       num_key_value_heads, return_softmax_lse, query_rope);
    at::Tensor output = std::get<0>(fia_output);
    at::Tensor softmax_lse = std::get<1>(fia_output);
    return std::tuple<at::Tensor, at::Tensor>(output, softmax_lse);
}
} // namespace custom

// 为NPU设备注册前向实现
TORCH_LIBRARY_IMPL(custom, PrivateUse1, m)
{
    m.impl("npu_fused_infer_attention_score", &custom::npu_fused_infer_attention_score_npu);
}

// 为META设备注册前向实现
TORCH_LIBRARY_IMPL(custom, Meta, m)
{
    m.impl("npu_fused_infer_attention_score", &custom::npu_fused_infer_attention_score_meta);
}
