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
 * \brief FlashAttentionScoreV4 内核入口（非量化，训练推理归一）
 *
 * 参考来源：
 *   flash_attention_score/op_kernel/flash_attention_score.cpp
 *   — 复用全部非量化（FLOAT16 / BFLOAT16 / FLOAT32）计算模板，
 *     包括 FlashAttentionScoreS1s2Bn2gs1、FlashAttentionScoreS1Bn2gs1、
 *     FlashAttentionScoreBn2gs1s2B、FlashAttentionVarLenScore 等。
 *     移除 FLOAT8（FP8）量化相关代码路径。
 *
 *   fused_floyd_attention/op_kernel/fused_floyd_attention.cpp
 *   — 参考推理算子 FusedFloydAttentionS1s2Bn2gs1 对 BNSD 格式、
 *     有 / 无 atten_mask 两种路径的 TILING_KEY_IS 分发方式，
 *     为 V4 统一算子提供等效的推理分发逻辑。
 *
 * 设计要点：
 *   1. 训练正向与推理共用同一套内核模板；区别在于调用方是否关注
 *      softmax_max / softmax_sum 输出（训练需要，推理忽略）。
 *   2. 新增 seed / offset 参数通过 tilingData->inputParams.seed /
 *      tilingData->inputParams.offset 传递给 dropout 生成逻辑。
 *      当 keepProb == 1.0 时 seed/offset 不生效（不执行 dropout）。
 *   3. 非量化：仅保留 DT_FLOAT16、DT_BF16、DT_FLOAT 分发分支。
 */

#ifdef KFC_L1_RESERVER_SIZE
#undef KFC_L1_RESERVER_SIZE
#define KFC_L1_RESERVER_SIZE 0
#else
#define KFC_L1_RESERVER_SIZE 0
#endif

#include "kernel_operator.h"

/* ---- 复用 flash_attention_score arch32 内核头文件 ---- */
#include "../../../flash_attention_score/op_kernel/arch32/flash_attention_score_empty_tensor.h"
#include "../../../flash_attention_score/op_kernel/arch32/flash_attention_score_drop_mask_adapter.h"
#include "../../../flash_attention_score/op_kernel/arch32/flash_attention_score_s1s2_bn2gs1.h"
#include "../../../flash_attention_score/op_kernel/arch32/flash_attention_score_s1s2_bn2gs1_sab.h"
#include "../../../flash_attention_score/op_kernel/arch32/flash_attention_score_s1_bn2gs1.h"
#include "../../../flash_attention_score/op_kernel/arch32/flash_attention_score_bn2gs1s2_b.h"
#include "../../../flash_attention_score/op_kernel/arch32/flash_attention_var_len_score.h"
#include "../../../flash_attention_score/op_kernel/arch32/flash_attention_var_len_score_sab.h"
#include "../../../flash_attention_score/op_kernel/arch32/flash_attention_score_template_tiling_key.h"
#include "../../../flash_attention_score/op_kernel/arch32/flash_attention_score_tiling.h"

using namespace AscendC;

/* ======================================================================
 * 宏定义（与 flash_attention_score.cpp 保持一致，仅更改注释说明）
 * ====================================================================== */

#ifdef __DAV_C220_CUBE__ // CUBE 实现

#define COPY_TILING_DATA(tiling)                                                                                       \
    GET_TILING_DATA_MEMBER(FlashAttentionScoreGeneralTilingData, bmm1TilingData, bmm1TilingDataVar, tiling);           \
    GET_TILING_DATA_MEMBER(FlashAttentionScoreGeneralTilingData, bmm2TilingData, bmm2TilingDataVar, tiling);           \
    const FlashAttentionScoreGeneralTilingData *__restrict tilingData = nullptr;                                       \
    const TCubeTiling *__restrict bmm1tiling = &bmm1TilingDataVar;                                                     \
    const TCubeTiling *__restrict bmm2tiling = &bmm2TilingDataVar;

#define COPY_TILING_DATA_SAMEAB(tiling)                                                                                \
    GET_TILING_DATA_WITH_STRUCT(FlashAttentionScoreGeneralTilingData, tilingDataIn, tiling);                           \
    const FlashAttentionScoreGeneralTilingData *__restrict tilingData = &tilingDataIn;                                 \
    const TCubeTiling *__restrict bmm1tiling = &(tilingData->bmm1TilingData);                                          \
    const TCubeTiling *__restrict bmm2tiling = &(tilingData->bmm2TilingData);

#define INVOKE_FA_GENERAL_OP_IMPL(templateClass, ...)                                                                  \
    do {                                                                                                               \
        templateClass<__VA_ARGS__> op;                                                                                 \
        COPY_TILING_DATA(tiling);                                                                                      \
        REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.bmm1, bmm1tiling, op.bmm2, bmm2tiling);                     \
    } while (0)

#define INVOKE_FA_GENERAL_OP_IMPL_SAMEAB(templateClass, ...)                                                           \
    do {                                                                                                               \
        __gm__ uint8_t *user = GetUserWorkspace(workspace);                                                            \
        COPY_TILING_DATA_SAMEAB(tiling);                                                                               \
        templateClass<__VA_ARGS__> op;                                                                                 \
        op.Init(query, queryRope, key, keyRope, value, pse, dropMask, paddingMask, prefix, attenMask, sink,            \
                softmaxMax, softmaxSum, softmaxOut, attentionOut, user, tilingData, &tPipe);                           \
        op.Process();                                                                                                  \
    } while (0)

#define INVOKE_FA_GENERAL_OP_IMPL_WITH_MMPOLICY(templateClass, ...)                                                    \
    do {                                                                                                               \
        __gm__ uint8_t *user = GetUserWorkspace(workspace);                                                            \
        COPY_TILING_DATA_SAMEAB(tiling);                                                                               \
        templateClass<__VA_ARGS__> op;                                                                                 \
        FaTscm::FaTscmArray tscmArray[FaTscm::TSCM_BUF_NUM];                                                           \
        FaTscm::TscmGlobal = tscmArray;                                                                                \
        FaTscm::InitTscmBuffer(&tPipe, FaTscm::TscmGlobal);                                                            \
        op.Init(query, queryRope, key, keyRope, value, pse, dropMask, paddingMask, prefix, attenMask, sink,            \
                softmaxMax, softmaxSum, softmaxOut, attentionOut, user, tilingData, &tPipe);                           \
        op.Process();                                                                                                  \
    } while (0)

#define INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(templateClass, ...)                                                           \
    do {                                                                                                               \
        templateClass<__VA_ARGS__> op;                                                                                 \
        COPY_TILING_DATA(tiling);                                                                                      \
        REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.bmm1, bmm1tiling, op.bmm1Nz, bmm1tiling, op.bmm2,           \
                          bmm2tiling);                                                                                 \
    } while (0)

#define INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ_SAMEAB(templateClass, ...)                                                    \
    do {                                                                                                               \
        __gm__ uint8_t *user = GetUserWorkspace(workspace);                                                            \
        COPY_TILING_DATA_SAMEAB(tiling);                                                                               \
        templateClass<__VA_ARGS__> op;                                                                                 \
        op.Init(query, queryRope, key, keyRope, value, pse, dropMask, paddingMask, prefix, attenMask, sink,            \
                softmaxMax, softmaxSum, softmaxOut, attentionOut, user, tilingData, &tPipe);                           \
        op.Process();                                                                                                  \
    } while (0)

#define INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ_WITH_MMPOLICY(templateClass, ...)                                             \
    do {                                                                                                               \
        templateClass<__VA_ARGS__> op;                                                                                 \
        COPY_TILING_DATA(tiling);                                                                                      \
        matmul::GlobalL1Array l1Array[matmul::L1_BUF_NUM];                                                             \
        matmul::l1Global = l1Array;                                                                                    \
        matmul::InitL1Buffer(&tPipe, matmul::l1Global);                                                                \
        REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.bmm1, bmm1tiling, op.bmm1Nz, bmm1tiling, op.bmm2,           \
                          bmm2tiling);                                                                                 \
    } while (0)

#define INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(templateClass, ...)                                                           \
    do {                                                                                                               \
        templateClass<__VA_ARGS__> op;                                                                                 \
        COPY_TILING_DATA(tiling);                                                                                      \
        REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.bmm1, bmm1tiling, op.bmm2, bmm2tiling, op.bmm2Nz,           \
                          bmm2tiling);                                                                                 \
    } while (0)

#define INVOKE_FA_GENERAL_OP_IMPL_VAR_LEN(templateClass, ...)                                                          \
    do {                                                                                                               \
        templateClass<__VA_ARGS__> op;                                                                                 \
        GET_TILING_DATA_MEMBER(FlashAttentionScoreGeneralTilingData, inputParams, inputParamsVar, tiling);             \
        if (inputParamsVar.needL1Carry) {                                                                              \
            COPY_TILING_DATA_SAMEAB(tiling);                                                                           \
            __gm__ uint8_t *user = GetUserWorkspace(workspace);                                                        \
            op.UnpackInit(query, queryRope, key, keyRope, value, pse, dropMask, paddingMask, prefix, attenMask, sink,  \
                          actualSeqLengths, actualSeqLengthsKv,                                                        \
                          softmaxMax, softmaxSum, softmaxOut, attentionOut, user, tilingData, &tPipe);                 \
            op.ProcessL1Carry();                                                                                       \
        } else {                                                                                                       \
            COPY_TILING_DATA(tiling);                                                                                  \
            REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.bmm1, bmm1tiling, op.bmm1Nz, bmm1tiling, op.bmm2,       \
                              bmm2tiling);                                                                             \
        }                                                                                                              \
    } while (0)

#define INVOKE_FA_GENERAL_OP_IMPL_VAR_LEN_SAMEAB(templateClass, ...)                                                   \
    do {                                                                                                               \
        COPY_TILING_DATA_SAMEAB(tiling);                                                                               \
        __gm__ uint8_t *user = GetUserWorkspace(workspace);                                                            \
        templateClass<__VA_ARGS__> op;                                                                                 \
        op.UnpackInit(query, queryRope, key, keyRope, value, pse, dropMask, paddingMask, prefix, attenMask, sink,      \
                      actualSeqLengths, actualSeqLengthsKv,                                                            \
                      softmaxMax, softmaxSum, softmaxOut, attentionOut, user, tilingData, &tPipe);                     \
        op.Process();                                                                                                  \
    } while (0)

#else // VECTOR 实现

#define COPY_TILING_DATA(tiling)                                                                                       \
    REGISTER_TILING_DEFAULT(FlashAttentionScoreGeneralTilingData);                                                     \
    GET_TILING_DATA_WITH_STRUCT(FlashAttentionScoreGeneralTilingData, tilingDataIn, tiling);                           \
    const FlashAttentionScoreGeneralTilingData *__restrict tilingData = &tilingDataIn;                                 \
    const TCubeTiling *__restrict bmm1tiling = &(tilingData->bmm1TilingData);                                          \
    const TCubeTiling *__restrict bmm2tiling = &(tilingData->bmm2TilingData);

#define INVOKE_FA_GENERAL_OP_IMPL(templateClass, ...)                                                                  \
    do {                                                                                                               \
        __gm__ uint8_t *user = GetUserWorkspace(workspace);                                                            \
        COPY_TILING_DATA(tiling);                                                                                      \
        if (tilingData->inputParams.needDropMaskOp) {                                                                  \
            FlashAttentionScoreDropMaskAdapter dropMaskAdapter;                                                        \
            dropMaskAdapter.Init(dropMask, user, tilingData, &tPipe);                                                  \
            dropMaskAdapter.Process();                                                                                 \
            tPipe.Reset();                                                                                             \
            templateClass<__VA_ARGS__> op;                                                                             \
            REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.bmm1, bmm1tiling, op.bmm2, bmm2tiling);                 \
            op.Init(query, queryRope, key, keyRope, value, pse, dropMask, paddingMask, prefix, attenMask,              \
                    sink, softmaxMax, softmaxSum, softmaxOut, attentionOut, user, tilingData, &tPipe);                 \
            op.Process();                                                                                              \
        } else {                                                                                                       \
            templateClass<__VA_ARGS__> op;                                                                             \
            REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.bmm1, bmm1tiling, op.bmm2, bmm2tiling);                 \
            op.Init(query, queryRope, key, keyRope, value, pse, dropMask, paddingMask, prefix, attenMask, sink,        \
                    softmaxMax, softmaxSum, softmaxOut, attentionOut, user, tilingData, &tPipe);                       \
            op.Process();                                                                                              \
        }                                                                                                              \
    } while (0)

#define INVOKE_FA_GENERAL_OP_IMPL_SAMEAB(templateClass, ...)                                                           \
    do {                                                                                                               \
        __gm__ uint8_t *user = GetUserWorkspace(workspace);                                                            \
        COPY_TILING_DATA(tiling);                                                                                      \
        if (tilingData->inputParams.needDropMaskOp) {                                                                  \
            FlashAttentionScoreDropMaskAdapter dropMaskAdapter;                                                        \
            dropMaskAdapter.Init(dropMask, user, tilingData, &tPipe);                                                  \
            dropMaskAdapter.Process();                                                                                 \
            tPipe.Reset();                                                                                             \
            templateClass<__VA_ARGS__> op;                                                                             \
            op.Init(query, queryRope, key, keyRope, value, pse, dropMask, paddingMask, prefix, attenMask,              \
                    sink, softmaxMax, softmaxSum, softmaxOut, attentionOut, user, tilingData, &tPipe);                 \
            op.Process();                                                                                              \
        } else {                                                                                                       \
            templateClass<__VA_ARGS__> op;                                                                             \
            op.Init(query, queryRope, key, keyRope, value, pse, dropMask, paddingMask, prefix, attenMask,              \
                    sink, softmaxMax, softmaxSum, softmaxOut, attentionOut, user, tilingData, &tPipe);                 \
            op.Process();                                                                                              \
        }                                                                                                              \
    } while (0)

#define INVOKE_FA_GENERAL_OP_IMPL_WITH_MMPOLICY(templateClass, ...)                                                    \
    do {                                                                                                               \
        __gm__ uint8_t *user = GetUserWorkspace(workspace);                                                            \
        COPY_TILING_DATA(tiling);                                                                                      \
        if (tilingData->inputParams.needDropMaskOp) {                                                                  \
            FlashAttentionScoreDropMaskAdapter dropMaskAdapter;                                                        \
            dropMaskAdapter.Init(dropMask, user, tilingData, &tPipe);                                                  \
            dropMaskAdapter.Process();                                                                                 \
            tPipe.Reset();                                                                                             \
            templateClass<__VA_ARGS__> op;                                                                             \
            FaTscm::FaTscmArray tscmArray[FaTscm::TSCM_BUF_NUM];                                                       \
            FaTscm::TscmGlobal = tscmArray;                                                                            \
            FaTscm::InitTscmBuffer(&tPipe, FaTscm::TscmGlobal);                                                        \
            op.Init(query, queryRope, key, keyRope, value, pse, dropMask, paddingMask, prefix, attenMask,              \
                    sink, softmaxMax, softmaxSum, softmaxOut, attentionOut, user, tilingData, &tPipe);                 \
            op.Process();                                                                                              \
        } else {                                                                                                       \
            templateClass<__VA_ARGS__> op;                                                                             \
            FaTscm::FaTscmArray tscmArray[FaTscm::TSCM_BUF_NUM];                                                       \
            FaTscm::TscmGlobal = tscmArray;                                                                            \
            FaTscm::InitTscmBuffer(&tPipe, FaTscm::TscmGlobal);                                                        \
            op.Init(query, queryRope, key, keyRope, value, pse, dropMask, paddingMask, prefix, attenMask,              \
                    sink, softmaxMax, softmaxSum, softmaxOut, attentionOut, user, tilingData, &tPipe);                 \
            op.Process();                                                                                              \
        }                                                                                                              \
    } while (0)

#define INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(templateClass, ...)                                                           \
    do {                                                                                                               \
        __gm__ uint8_t *user = GetUserWorkspace(workspace);                                                            \
        COPY_TILING_DATA(tiling);                                                                                      \
        if (tilingData->inputParams.needDropMaskOp) {                                                                  \
            FlashAttentionScoreDropMaskAdapter dropMaskAdapter;                                                        \
            dropMaskAdapter.Init(dropMask, user, tilingData, &tPipe);                                                  \
            dropMaskAdapter.Process();                                                                                 \
            tPipe.Destroy();                                                                                           \
            TPipe tPipeOp;                                                                                             \
            templateClass<__VA_ARGS__> op;                                                                             \
            REGIST_MATMUL_OBJ(&tPipeOp, GetSysWorkSpacePtr(), op.bmm1, bmm1tiling, op.bmm1Nz, bmm1tiling, op.bmm2,     \
                              bmm2tiling);                                                                             \
            op.Init(query, queryRope, key, keyRope, value, pse, dropMask, paddingMask, prefix, attenMask,              \
                    sink, softmaxMax, softmaxSum, softmaxOut, attentionOut, user, tilingData, &tPipeOp);               \
            op.Process();                                                                                              \
        } else {                                                                                                       \
            templateClass<__VA_ARGS__> op;                                                                             \
            REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.bmm1, bmm1tiling, op.bmm1Nz, bmm1tiling, op.bmm2,       \
                              bmm2tiling);                                                                             \
            op.Init(query, queryRope, key, keyRope, value, pse, dropMask, paddingMask, prefix, attenMask,              \
                    sink, softmaxMax, softmaxSum, softmaxOut, attentionOut, user, tilingData, &tPipe);                 \
            op.Process();                                                                                              \
        }                                                                                                              \
    } while (0)

#define INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ_SAMEAB(templateClass, ...)                                                    \
    do {                                                                                                               \
        __gm__ uint8_t *user = GetUserWorkspace(workspace);                                                            \
        COPY_TILING_DATA(tiling);                                                                                      \
        if (tilingData->inputParams.needDropMaskOp) {                                                                  \
            FlashAttentionScoreDropMaskAdapter dropMaskAdapter;                                                        \
            dropMaskAdapter.Init(dropMask, user, tilingData, &tPipe);                                                  \
            dropMaskAdapter.Process();                                                                                 \
            tPipe.Destroy();                                                                                           \
            TPipe tPipeOp;                                                                                             \
            templateClass<__VA_ARGS__> op;                                                                             \
            op.Init(query, queryRope, key, keyRope, value, pse, dropMask, paddingMask, prefix, attenMask,              \
                    sink, softmaxMax, softmaxSum, softmaxOut, attentionOut, user, tilingData, &tPipeOp);               \
            op.Process();                                                                                              \
        } else {                                                                                                       \
            templateClass<__VA_ARGS__> op;                                                                             \
            op.Init(query, queryRope, key, keyRope, value, pse, dropMask, paddingMask, prefix, attenMask,              \
                    sink, softmaxMax, softmaxSum, softmaxOut, attentionOut, user, tilingData, &tPipe);                 \
            op.Process();                                                                                              \
        }                                                                                                              \
    } while (0)

#define INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(templateClass, ...)                                                           \
    do {                                                                                                               \
        __gm__ uint8_t *user = GetUserWorkspace(workspace);                                                            \
        COPY_TILING_DATA(tiling);                                                                                      \
        if (tilingData->inputParams.needDropMaskOp) {                                                                  \
            FlashAttentionScoreDropMaskAdapter dropMaskAdapter;                                                        \
            dropMaskAdapter.Init(dropMask, user, tilingData, &tPipe);                                                  \
            dropMaskAdapter.Process();                                                                                 \
            tPipe.Destroy();                                                                                           \
            TPipe tPipeOp;                                                                                             \
            templateClass<__VA_ARGS__> op;                                                                             \
            REGIST_MATMUL_OBJ(&tPipeOp, GetSysWorkSpacePtr(), op.bmm1, bmm1tiling, op.bmm2, bmm2tiling, op.bmm2Nz,     \
                              bmm2tiling);                                                                             \
            op.Init(query, key, value, pse, dropMask, paddingMask, prefix, attenMask, sink,                            \
                    softmaxMax, softmaxSum, softmaxOut, attentionOut, user, tilingData, &tPipeOp);                     \
            op.Process();                                                                                              \
        } else {                                                                                                       \
            templateClass<__VA_ARGS__> op;                                                                             \
            REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.bmm1, bmm1tiling, op.bmm2, bmm2tiling, op.bmm2Nz,       \
                              bmm2tiling);                                                                             \
            op.Init(query, key, value, pse, dropMask, paddingMask, prefix, attenMask, sink,                            \
                    softmaxMax, softmaxSum, softmaxOut, attentionOut, user, tilingData, &tPipe);                       \
            op.Process();                                                                                              \
        }                                                                                                              \
    } while (0)

#define INVOKE_FA_GENERAL_OP_IMPL_VAR_LEN(templateClass, ...)                                                          \
    do {                                                                                                               \
        COPY_TILING_DATA(tiling);                                                                                      \
        __gm__ uint8_t *user = GetUserWorkspace(workspace);                                                            \
        if (tilingData->inputParams.needDropMaskOp) {                                                                  \
            FlashAttentionScoreDropMaskAdapter dropMaskAdapter;                                                        \
            dropMaskAdapter.Init(dropMask, user, tilingData, &tPipe);                                                  \
            dropMaskAdapter.Process();                                                                                 \
        }                                                                                                              \
        tPipe.Reset();                                                                                                 \
        templateClass<__VA_ARGS__> op;                                                                                 \
        REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.bmm1, bmm1tiling, op.bmm1Nz, bmm1tiling, op.bmm2,           \
                          bmm2tiling);                                                                                 \
        op.UnpackInit(query, queryRope, key, keyRope, value, pse, dropMask, paddingMask, prefix, attenMask, sink,      \
                      actualSeqLengths, actualSeqLengthsKv,                                                            \
                      softmaxMax, softmaxSum, softmaxOut, attentionOut, user, tilingData, &tPipe);                     \
        if (tilingData->inputParams.needL1Carry) {                                                                     \
            op.ProcessL1Carry();                                                                                       \
        } else {                                                                                                       \
            op.Process();                                                                                              \
        }                                                                                                              \
    } while (0)

#define INVOKE_FA_GENERAL_OP_IMPL_VAR_LEN_SAMEAB(templateClass, ...)                                                   \
    do {                                                                                                               \
        COPY_TILING_DATA(tiling);                                                                                      \
        __gm__ uint8_t *user = GetUserWorkspace(workspace);                                                            \
        if (tilingData->inputParams.needDropMaskOp) {                                                                  \
            FlashAttentionScoreDropMaskAdapter dropMaskAdapter;                                                        \
            dropMaskAdapter.Init(dropMask, user, tilingData, &tPipe);                                                  \
            dropMaskAdapter.Process();                                                                                 \
        }                                                                                                              \
        tPipe.Reset();                                                                                                 \
        templateClass<__VA_ARGS__> op;                                                                                 \
        op.UnpackInit(query, queryRope, key, keyRope, value, pse, dropMask, paddingMask, prefix, attenMask, sink,      \
                      actualSeqLengths, actualSeqLengthsKv,                                                            \
                      softmaxMax, softmaxSum, softmaxOut, attentionOut, user, tilingData, &tPipe);                     \
        op.Process();                                                                                                  \
    } while (0)

#endif /* __DAV_C220_CUBE__ */

/* ======================================================================
 * flash_attention_score_v4 内核入口
 *
 * 参数说明（按 aclnnFlashAttentionScoreV4 接口文档顺序）：
 *   query / key / value       — 必选输入
 *   pse                       — real_shift (PSE) 可选
 *   dropMask                  — drop_mask 可选；当 keepProb<1.0 且为 nullptr 时，
 *                               内核通过 tilingData->inputParams.seed/offset 内部生成
 *   paddingMask               — padding_mask 可选（暂未使用）
 *   attenMask                 — atten_mask 可选
 *   prefix                    — prefix 稀疏可选
 *   actualSeqLengths          — actual_seq_qlen 可选
 *   actualSeqLengthsKv        — actual_seq_kvlen 可选
 *   qStartIdx / kvStartIdx    — 外切起始索引可选
 *   deqScaleQ/K/V             — 量化参数，非量化场景传 nullptr（内核不使用）
 *   queryRope / keyRope       — RoPE 可选
 *   sink                      — 保留参数（nullptr）
 *   softmaxMax / softmaxSum   — 输出：训练反向所需，推理可忽略
 *   softmaxOut                — 输出：保留接口（空 tensor）
 *   attentionOut              — 输出：注意力计算结果
 * ====================================================================== */
KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

template<uint8_t KernelTypeKey, uint8_t UB0, uint8_t UB1, uint8_t Block, uint8_t ImplMode,
    uint8_t DataType, uint8_t Layout, uint8_t Bmm1Format, uint8_t Bmm2Source, uint8_t Sparse,
    uint8_t BigDoubleBuffer, bool HasDropOut, bool HasAttenMask, bool HasPse, bool EnableL1Reuse,
    bool HasRope, uint8_t MatmulPolicyType, uint8_t S1TemplateType, uint8_t S2TemplateType,
    uint8_t dTemplateSize>
__global__ __aicore__ void
flash_attention_score_v4(
    __gm__ uint8_t *query,           __gm__ uint8_t *key,             __gm__ uint8_t *value,
    __gm__ uint8_t *pse,             __gm__ uint8_t *dropMask,        __gm__ uint8_t *paddingMask,
    __gm__ uint8_t *attenMask,       __gm__ uint8_t *prefix,          __gm__ uint8_t *actualSeqLengths,
    __gm__ uint8_t *actualSeqLengthsKv, __gm__ uint8_t *qStartIdx,   __gm__ uint8_t *kvStartIdx,
    __gm__ uint8_t *deqScaleQ,       __gm__ uint8_t *deqScaleK,       __gm__ uint8_t *deqScaleV,
    __gm__ uint8_t *queryRope,       __gm__ uint8_t *keyRope,         __gm__ uint8_t *sink,
    __gm__ uint8_t *pScale,
    __gm__ uint8_t *softmaxMax,      __gm__ uint8_t *softmaxSum,      __gm__ uint8_t *softmaxOut,
    __gm__ uint8_t *attentionOut,
    __gm__ uint8_t *workspace,       __gm__ uint8_t *tiling)
{
    TPipe tPipe;
    AscendC::SetMaskNorm();
    REGISTER_TILING_DEFAULT(FlashAttentionScoreGeneralTilingData);

    /* ---- 空输入处理 ---- */
    if constexpr (KernelTypeKey == 1) {
        REGISTER_TILING_FOR_TILINGKEY("(TILING_KEY_VAR == 0x1)", FlashAttentionScoreTilingData);
        GET_TILING_DATA_WITH_STRUCT(FlashAttentionScoreTilingData, tiling_data_in, tiling);
        const FlashAttentionScoreTilingData *__restrict tiling_data = &tiling_data_in;
        /* 非量化：仅保留 FP16 / BF16 / FP32 三种空 tensor 分支 */
        if (ORIG_DTYPE_QUERY == DT_FLOAT16) {
            FlashAttentionScoreEmptyTensor<half> op;
            op.Init(softmaxMax, softmaxSum, attentionOut, tiling_data);
            op.Process();
        } else if (ORIG_DTYPE_QUERY == DT_FLOAT) {
            FlashAttentionScoreEmptyTensor<float> op;
            op.Init(softmaxMax, softmaxSum, attentionOut, tiling_data);
            op.Process();
#if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
        } else if (ORIG_DTYPE_QUERY == DT_BF16) {
            FlashAttentionScoreEmptyTensor<bfloat16_t> op;
            op.Init(softmaxMax, softmaxSum, attentionOut, tiling_data);
            op.Process();
#endif
        }
        return;
    }

    /* ----
     * TND sameAB 模板（推理 / 训练均支持）
     * 参考：flash_attention_score.cpp TND sameAB 分支
     * ---- */
    if constexpr (UB0 == 3 && UB1 == 9 && Block == 9 && Layout == 4) {
        if constexpr (DataType == 3) {
            if constexpr (MatmulPolicyType == 1) {
                INVOKE_FA_GENERAL_OP_IMPL_VAR_LEN_SAMEAB(FlashAttentionVarLenScoreSameAB,
                    LayOutTypeEnum::LAYOUT_TND, HasPse, HasAttenMask, HasDropOut,
                    half, float, CubeFormat::NZ, MmPolicyType::UNSPLITK, ImplModeEnum(ImplMode), HasRope);
            } else {
                INVOKE_FA_GENERAL_OP_IMPL_VAR_LEN_SAMEAB(FlashAttentionVarLenScoreSameAB,
                    LayOutTypeEnum::LAYOUT_TND, HasPse, HasAttenMask, HasDropOut,
                    half, float, CubeFormat::NZ, MmPolicyType::NORMAL, ImplModeEnum(ImplMode), HasRope);
            }
        } else if constexpr (DataType == 2) {
#if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
            if constexpr (MatmulPolicyType == 1) {
                INVOKE_FA_GENERAL_OP_IMPL_VAR_LEN_SAMEAB(FlashAttentionVarLenScoreSameAB,
                    LayOutTypeEnum::LAYOUT_TND, HasPse, HasAttenMask, HasDropOut,
                    bfloat16_t, float, CubeFormat::NZ, MmPolicyType::UNSPLITK, ImplModeEnum(ImplMode), HasRope);
            } else {
                INVOKE_FA_GENERAL_OP_IMPL_VAR_LEN_SAMEAB(FlashAttentionVarLenScoreSameAB,
                    LayOutTypeEnum::LAYOUT_TND, HasPse, HasAttenMask, HasDropOut,
                    bfloat16_t, float, CubeFormat::NZ, MmPolicyType::NORMAL, ImplModeEnum(ImplMode), HasRope);
            }
#endif
        } else if constexpr (DataType == 1) {
            if constexpr (MatmulPolicyType == 1) {
                INVOKE_FA_GENERAL_OP_IMPL_VAR_LEN_SAMEAB(FlashAttentionVarLenScoreSameAB,
                    LayOutTypeEnum::LAYOUT_TND, HasPse, HasAttenMask, HasDropOut,
                    float, float, CubeFormat::NZ, MmPolicyType::UNSPLITK, ImplModeEnum(ImplMode), HasRope);
            } else {
                INVOKE_FA_GENERAL_OP_IMPL_VAR_LEN_SAMEAB(FlashAttentionVarLenScoreSameAB,
                    LayOutTypeEnum::LAYOUT_TND, HasPse, HasAttenMask, HasDropOut,
                    float, float, CubeFormat::NZ, MmPolicyType::NORMAL, ImplModeEnum(ImplMode), HasRope);
            }
        }
    }

    /* ---- TND 标准模板 ---- */
    if constexpr (UB0 == 3 && UB1 == 4 && Block == 9 && Layout == 4) {
        if constexpr (DataType == 3) {
            INVOKE_FA_GENERAL_OP_IMPL_VAR_LEN(FlashAttentionVarLenScore, ImplModeEnum(ImplMode),
                LayOutTypeEnum::LAYOUT_TND, HasPse, HasAttenMask, HasDropOut,
                half, float, true, CubeFormat::NZ, HasRope);
        } else if constexpr (DataType == 2) {
#if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
            INVOKE_FA_GENERAL_OP_IMPL_VAR_LEN(FlashAttentionVarLenScore, ImplModeEnum(ImplMode),
                LayOutTypeEnum::LAYOUT_TND, HasPse, HasAttenMask, HasDropOut,
                bfloat16_t, float, true, CubeFormat::NZ, HasRope);
#endif
        } else if constexpr (DataType == 1) {
            INVOKE_FA_GENERAL_OP_IMPL_VAR_LEN(FlashAttentionVarLenScore, ImplModeEnum(ImplMode),
                LayOutTypeEnum::LAYOUT_TND, HasPse, HasAttenMask, HasDropOut,
                float, float, true, CubeFormat::NZ, HasRope);
        }
    }

    /* ----
     * sameAB 模板（BSH / SBH / BSND / BNSD）
     * 参考：flash_attention_score.cpp sameAB 分支
     * ---- */
    if constexpr (UB0 == 3 && UB1 == 9 && Block == 9 && Layout != 4) {
        if constexpr (DataType == 3) {
            if constexpr (MatmulPolicyType == 1) {
                INVOKE_FA_GENERAL_OP_IMPL_WITH_MMPOLICY(FlashAttentionScoreS1s2Bn2gs1SameAB,
                    ImplModeEnum(ImplMode), LayOutTypeEnum(Layout), HasPse, HasAttenMask, HasDropOut,
                    half, float, CubeFormat::ND, MmPolicyType::UNSPLITK, HasRope);
            } else if constexpr (Bmm1Format == 0) {
                INVOKE_FA_GENERAL_OP_IMPL_SAMEAB(FlashAttentionScoreS1s2Bn2gs1SameAB,
                    ImplModeEnum(ImplMode), LayOutTypeEnum(Layout), HasPse, HasAttenMask, HasDropOut,
                    half, float, CubeFormat::ND, MmPolicyType::NORMAL, HasRope);
            } else if constexpr (Bmm1Format == 1) {
                INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ_SAMEAB(FlashAttentionScoreS1s2Bn2gs1SameAB,
                    ImplModeEnum(ImplMode), LayOutTypeEnum(Layout), HasPse, HasAttenMask, HasDropOut,
                    half, float, CubeFormat::NZ, MmPolicyType::NORMAL, HasRope);
            }
        } else if constexpr (DataType == 2) {
#if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
            if constexpr (MatmulPolicyType == 1) {
                INVOKE_FA_GENERAL_OP_IMPL_WITH_MMPOLICY(FlashAttentionScoreS1s2Bn2gs1SameAB,
                    ImplModeEnum(ImplMode), LayOutTypeEnum(Layout), HasPse, HasAttenMask, HasDropOut,
                    bfloat16_t, float, CubeFormat::ND, MmPolicyType::UNSPLITK, HasRope);
            } else if constexpr (Bmm1Format == 0) {
                INVOKE_FA_GENERAL_OP_IMPL_SAMEAB(FlashAttentionScoreS1s2Bn2gs1SameAB,
                    ImplModeEnum(ImplMode), LayOutTypeEnum(Layout), HasPse, HasAttenMask, HasDropOut,
                    bfloat16_t, float, CubeFormat::ND, MmPolicyType::NORMAL, HasRope);
            } else if constexpr (Bmm1Format == 1) {
                INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ_SAMEAB(FlashAttentionScoreS1s2Bn2gs1SameAB,
                    ImplModeEnum(ImplMode), LayOutTypeEnum(Layout), HasPse, HasAttenMask, HasDropOut,
                    bfloat16_t, float, CubeFormat::NZ, MmPolicyType::NORMAL, HasRope);
            }
#endif
        }
    }

    /* ----
     * s1s2 标准模板（BSH / SBH / BSND / BNSD）
     * 参考：flash_attention_score.cpp s1s2 分支
     * 同时覆盖推理（inference）场景（fused_floyd_attention 参考）：
     *   训练：keepProb < 1.0 时 HasDropOut=true，需要 softmax_max/softmax_sum 供反向使用
     *   推理：keepProb=1.0 时 HasDropOut=false，softmax_max/softmax_sum 输出但调用方忽略
     * ---- */
    if constexpr (UB0 == 3 && UB1 == 4 && Block == 9 && Layout != 4) {
        if constexpr (DataType == 3) {
            if constexpr (Bmm1Format == 0) {
                INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1,
                    ImplModeEnum(ImplMode), LayOutTypeEnum(Layout),
                    bool(HasPse), bool(HasAttenMask), bool(HasDropOut),
                    half, float, true, CubeFormat::ND, bool(EnableL1Reuse));
            } else {
                INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1,
                    ImplModeEnum(ImplMode), LayOutTypeEnum(Layout),
                    bool(HasPse), bool(HasAttenMask), bool(HasDropOut),
                    half, float, true, CubeFormat::NZ, bool(EnableL1Reuse));
            }
        } else if constexpr (DataType == 2) {
#if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
            if constexpr (Bmm1Format == 0) {
                INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1,
                    ImplModeEnum(ImplMode), LayOutTypeEnum(Layout),
                    bool(HasPse), bool(HasAttenMask), bool(HasDropOut),
                    bfloat16_t, float, true, CubeFormat::ND, bool(EnableL1Reuse));
            } else {
                INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1,
                    ImplModeEnum(ImplMode), LayOutTypeEnum(Layout),
                    bool(HasPse), bool(HasAttenMask), bool(HasDropOut),
                    bfloat16_t, float, true, CubeFormat::NZ, bool(EnableL1Reuse));
            }
#endif
        } else if constexpr (DataType == 1) {
            if constexpr (Bmm1Format == 0) {
                INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1,
                    ImplModeEnum(ImplMode), LayOutTypeEnum(Layout),
                    bool(HasPse), bool(HasAttenMask), bool(HasDropOut),
                    float, float, true, CubeFormat::ND, bool(EnableL1Reuse));
            } else {
                INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1,
                    ImplModeEnum(ImplMode), LayOutTypeEnum(Layout),
                    bool(HasPse), bool(HasAttenMask), bool(HasDropOut),
                    float, float, true, CubeFormat::NZ, bool(EnableL1Reuse));
            }
        }
    }

    /* ---- S1 模板 ---- */
    if constexpr (UB0 == 3 && UB1 == 5 && Block == 9) {
        if constexpr (DataType == 3) {
            if constexpr (Bmm1Format == 0) {
                if constexpr (Bmm2Source > 0) {
                    INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1,
                        ImplModeEnum(ImplMode), LayOutTypeEnum(Layout),
                        bool(HasPse), bool(HasAttenMask), bool(HasDropOut),
                        half, float, true, CubeFormat::ND, TPosition::TSCM, CubeFormat::NZ,
                        bool(EnableL1Reuse),
                        STemplateType(S1TemplateType * 16),
                        STemplateType(S2TemplateType * 16),
                        DTemplateType(dTemplateSize * 16));
                } else {
                    INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1,
                        ImplModeEnum(ImplMode), LayOutTypeEnum(Layout),
                        bool(HasPse), bool(HasAttenMask), bool(HasDropOut),
                        half, float, true, CubeFormat::ND, TPosition::GM, CubeFormat::ND,
                        bool(EnableL1Reuse),
                        STemplateType(S1TemplateType * 16),
                        STemplateType(S2TemplateType * 16),
                        DTemplateType(dTemplateSize * 16));
                }
            } else {
                if constexpr (Bmm2Source > 0) {
                    INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1,
                        ImplModeEnum(ImplMode), LayOutTypeEnum(Layout),
                        bool(HasPse), bool(HasAttenMask), bool(HasDropOut),
                        half, float, true, CubeFormat::NZ, TPosition::TSCM, CubeFormat::NZ,
                        bool(EnableL1Reuse),
                        STemplateType(S1TemplateType * 16),
                        STemplateType(S2TemplateType * 16),
                        DTemplateType(dTemplateSize * 16));
                } else {
                    INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1,
                        ImplModeEnum(ImplMode), LayOutTypeEnum(Layout),
                        bool(HasPse), bool(HasAttenMask), bool(HasDropOut),
                        half, float, true, CubeFormat::NZ, TPosition::GM, CubeFormat::ND,
                        bool(EnableL1Reuse),
                        STemplateType(S1TemplateType * 16),
                        STemplateType(S2TemplateType * 16),
                        DTemplateType(dTemplateSize * 16));
                }
            }
        } else if constexpr (DataType == 2) {
#if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
            if constexpr (Bmm1Format == 0) {
                if constexpr (Bmm2Source > 0) {
                    INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1,
                        ImplModeEnum(ImplMode), LayOutTypeEnum(Layout),
                        bool(HasPse), bool(HasAttenMask), bool(HasDropOut),
                        bfloat16_t, float, true, CubeFormat::ND, TPosition::TSCM, CubeFormat::NZ,
                        bool(EnableL1Reuse),
                        STemplateType(S1TemplateType * 16),
                        STemplateType(S2TemplateType * 16),
                        DTemplateType(dTemplateSize * 16));
                } else {
                    INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1,
                        ImplModeEnum(ImplMode), LayOutTypeEnum(Layout),
                        bool(HasPse), bool(HasAttenMask), bool(HasDropOut),
                        bfloat16_t, float, true, CubeFormat::ND, TPosition::GM, CubeFormat::ND,
                        bool(EnableL1Reuse),
                        STemplateType(S1TemplateType * 16),
                        STemplateType(S2TemplateType * 16),
                        DTemplateType(dTemplateSize * 16));
                }
            } else {
                if constexpr (Bmm2Source > 0) {
                    INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1,
                        ImplModeEnum(ImplMode), LayOutTypeEnum(Layout),
                        bool(HasPse), bool(HasAttenMask), bool(HasDropOut),
                        bfloat16_t, float, true, CubeFormat::NZ, TPosition::TSCM, CubeFormat::NZ,
                        bool(EnableL1Reuse),
                        STemplateType(S1TemplateType * 16),
                        STemplateType(S2TemplateType * 16),
                        DTemplateType(dTemplateSize * 16));
                } else {
                    INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1,
                        ImplModeEnum(ImplMode), LayOutTypeEnum(Layout),
                        bool(HasPse), bool(HasAttenMask), bool(HasDropOut),
                        bfloat16_t, float, true, CubeFormat::NZ, TPosition::GM, CubeFormat::ND,
                        bool(EnableL1Reuse),
                        STemplateType(S1TemplateType * 16),
                        STemplateType(S2TemplateType * 16),
                        DTemplateType(dTemplateSize * 16));
                }
            }
#endif
        } else if constexpr (DataType == 1) {
            if constexpr (Bmm1Format == 0) {
                if constexpr (Bmm2Source > 0) {
                    INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1,
                        ImplModeEnum(ImplMode), LayOutTypeEnum(Layout),
                        bool(HasPse), bool(HasAttenMask), bool(HasDropOut),
                        float, float, true, CubeFormat::ND, TPosition::TSCM, CubeFormat::NZ,
                        bool(EnableL1Reuse),
                        STemplateType(S1TemplateType * 16),
                        STemplateType(S2TemplateType * 16),
                        DTemplateType(dTemplateSize * 16));
                } else {
                    INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1,
                        ImplModeEnum(ImplMode), LayOutTypeEnum(Layout),
                        bool(HasPse), bool(HasAttenMask), bool(HasDropOut),
                        float, float, true, CubeFormat::ND, TPosition::GM, CubeFormat::ND,
                        bool(EnableL1Reuse),
                        STemplateType(S1TemplateType * 16),
                        STemplateType(S2TemplateType * 16),
                        DTemplateType(dTemplateSize * 16));
                }
            } else {
                if constexpr (Bmm2Source > 0) {
                    INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1,
                        ImplModeEnum(ImplMode), LayOutTypeEnum(Layout),
                        bool(HasPse), bool(HasAttenMask), bool(HasDropOut),
                        float, float, true, CubeFormat::NZ, TPosition::TSCM, CubeFormat::NZ,
                        bool(EnableL1Reuse),
                        STemplateType(S1TemplateType * 16),
                        STemplateType(S2TemplateType * 16),
                        DTemplateType(dTemplateSize * 16));
                } else {
                    INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1,
                        ImplModeEnum(ImplMode), LayOutTypeEnum(Layout),
                        bool(HasPse), bool(HasAttenMask), bool(HasDropOut),
                        float, float, true, CubeFormat::NZ, TPosition::GM, CubeFormat::ND,
                        bool(EnableL1Reuse),
                        STemplateType(S1TemplateType * 16),
                        STemplateType(S2TemplateType * 16),
                        DTemplateType(dTemplateSize * 16));
                }
            }
        }
    }

    /* ----
     * B 模板（FlashAttentionScoreBn2gs1s2B）
     * 推理友好：适用于超长序列 BNSD / BSND / BSH 场景
     * 参考：fused_floyd_attention 中 BNSD 格式推理优化的 BN2GS1 分发方式
     * ---- */
    if constexpr (UB0 == 9 && UB1 == 9 && Block == 0) {
        if constexpr (DataType == 3) {
            if constexpr (Layout == 1) {
                INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B,
                    ImplModeEnum(ImplMode), LayOutTypeEnum(Layout),
                    bool(HasPse), bool(HasAttenMask), bool(HasDropOut),
                    half, float, true, LayoutMode::BSNGD,
                    STemplateType(S1TemplateType * 16),
                    STemplateType(S2TemplateType * 16),
                    DTemplateType(dTemplateSize * 16));
            } else if constexpr (Layout == 2) {
                INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B,
                    ImplModeEnum(ImplMode), LayOutTypeEnum(Layout),
                    bool(HasPse), bool(HasAttenMask), bool(HasDropOut),
                    half, float, true, LayoutMode::SBNGD,
                    STemplateType(S1TemplateType * 16),
                    STemplateType(S2TemplateType * 16),
                    DTemplateType(dTemplateSize * 16));
            } else if constexpr (Layout == 3) {
                INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B,
                    ImplModeEnum(ImplMode), LayOutTypeEnum(Layout),
                    bool(HasPse), bool(HasAttenMask), bool(HasDropOut),
                    half, float, true, LayoutMode::BNGS1S2,
                    STemplateType(S1TemplateType * 16),
                    STemplateType(S2TemplateType * 16),
                    DTemplateType(dTemplateSize * 16));
            }
        } else if constexpr (DataType == 2) {
#if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
            if constexpr (Layout == 1) {
                INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B,
                    ImplModeEnum(ImplMode), LayOutTypeEnum(Layout),
                    bool(HasPse), bool(HasAttenMask), bool(HasDropOut),
                    bfloat16_t, float, true, LayoutMode::BSNGD,
                    STemplateType(S1TemplateType * 16),
                    STemplateType(S2TemplateType * 16),
                    DTemplateType(dTemplateSize * 16));
            } else if constexpr (Layout == 2) {
                INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B,
                    ImplModeEnum(ImplMode), LayOutTypeEnum(Layout),
                    bool(HasPse), bool(HasAttenMask), bool(HasDropOut),
                    bfloat16_t, float, true, LayoutMode::SBNGD,
                    STemplateType(S1TemplateType * 16),
                    STemplateType(S2TemplateType * 16),
                    DTemplateType(dTemplateSize * 16));
            } else if constexpr (Layout == 3) {
                INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B,
                    ImplModeEnum(ImplMode), LayOutTypeEnum(Layout),
                    bool(HasPse), bool(HasAttenMask), bool(HasDropOut),
                    bfloat16_t, float, true, LayoutMode::BNGS1S2,
                    STemplateType(S1TemplateType * 16),
                    STemplateType(S2TemplateType * 16),
                    DTemplateType(dTemplateSize * 16));
            }
#endif
        } else if constexpr (DataType == 1) {
            if constexpr (Layout == 1) {
                INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B,
                    ImplModeEnum(ImplMode), LayOutTypeEnum(Layout),
                    bool(HasPse), bool(HasAttenMask), bool(HasDropOut),
                    float, float, true, LayoutMode::BSNGD,
                    STemplateType(S1TemplateType * 16),
                    STemplateType(S2TemplateType * 16),
                    DTemplateType(dTemplateSize * 16));
            } else if constexpr (Layout == 2) {
                INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B,
                    ImplModeEnum(ImplMode), LayOutTypeEnum(Layout),
                    bool(HasPse), bool(HasAttenMask), bool(HasDropOut),
                    float, float, true, LayoutMode::SBNGD,
                    STemplateType(S1TemplateType * 16),
                    STemplateType(S2TemplateType * 16),
                    DTemplateType(dTemplateSize * 16));
            } else if constexpr (Layout == 3) {
                INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B,
                    ImplModeEnum(ImplMode), LayOutTypeEnum(Layout),
                    bool(HasPse), bool(HasAttenMask), bool(HasDropOut),
                    float, float, true, LayoutMode::BNGS1S2,
                    STemplateType(S1TemplateType * 16),
                    STemplateType(S2TemplateType * 16),
                    DTemplateType(dTemplateSize * 16));
            }
        }
    }

    return;
}
