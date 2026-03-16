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
 * \file flash_attention_def.cpp
 * \brief FlashAttention算子定义（训练推理归一，仅非量化）
 *        输入数据类型仅支持FLOAT16和BFLOAT16。
 *        支持BSND/BNSD/TND三种layout，支持分页KV缓存（PA_ND/PA_Nz）。
 */

#include "register/op_def_registry.h"

namespace ops {

class FlashAttention : public OpDef {
public:
    explicit FlashAttention(const char *name) : OpDef(name)
    {
        // ----------------------------------------------------------------
        // 必选输入
        // ----------------------------------------------------------------
        this->Input("q")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16,
                       ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();

        this->Input("k")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16,
                       ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();

        this->Input("v")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16,
                       ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();

        // ----------------------------------------------------------------
        // 可选输入
        // ----------------------------------------------------------------
        // 分页KV缓存块映射表（与layoutKv=PA_ND/PA_Nz配合使用）
        this->Input("block_table")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32, ge::DT_INT32,
                       ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();

        // query的累积序列长度（TND变长场景）
        this->Input("cu_seqlens_q")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32, ge::DT_INT32,
                       ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND})
            .ValueDepend(OPTIONAL)
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND});

        // kv的累积序列长度（TND变长场景）
        this->Input("cu_seqlens_kv")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32, ge::DT_INT32,
                       ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND})
            .ValueDepend(OPTIONAL)
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND});

        // query各batch实际序列长度（padded batch场景）
        this->Input("seqused_q")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32, ge::DT_INT32,
                       ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND})
            .ValueDepend(OPTIONAL)
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND});

        // kv各batch实际序列长度（padded batch场景）
        this->Input("seqused_kv")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32, ge::DT_INT32,
                       ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND})
            .ValueDepend(OPTIONAL)
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND});

        // 可学习的sink注意力权重
        this->Input("sinks")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();

        // 预计算tiling方案（由上游算子传入，减少推理时tiling开销）
        this->Input("metadata")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32, ge::DT_INT32,
                       ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND})
            .ValueDepend(OPTIONAL)
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND});

        // ----------------------------------------------------------------
        // 必选输出
        // ----------------------------------------------------------------
        // attention计算结果，shape和dtype与q一致
        this->Output("attention_out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16,
                       ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND});

        // ----------------------------------------------------------------
        // 可选输出
        // ----------------------------------------------------------------
        // softmax的log-sum-exp，训练时（returnSoftmaxLse=1）用于反向传播
        this->Output("softmax_lse")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND});

        // ----------------------------------------------------------------
        // 属性定义
        // ----------------------------------------------------------------
        // softmax缩放系数，0.0时使用1/sqrt(D)，对应老接口scaleValue
        this->Attr("softmax_mode")
            .AttrType(OPTIONAL)
            .Float(0.0f);

        // 掩码/稀疏模式，对应老接口sparseMode：0=无掩码，1=因果，2=非因果，3=band，4=滑动窗口
        this->Attr("mask_mode")
            .AttrType(OPTIONAL)
            .Int(0);

        // 左侧注意力窗口（maskMode=4时生效，对应老接口preTokens）
        this->Attr("win_left")
            .AttrType(OPTIONAL)
            .Int(2147483647);

        // 右侧注意力窗口（maskMode=4时生效，对应老接口nextTokens）
        this->Attr("win_right")
            .AttrType(OPTIONAL)
            .Int(2147483647);

        // query的数据布局，从fused_infer_attention_score的inputLayout拆分而来
        this->Attr("layout_q")
            .AttrType(OPTIONAL)
            .String("BSND");

        // kv的数据布局，从fused_infer_attention_score的inputLayout拆分而来
        this->Attr("layout_kv")
            .AttrType(OPTIONAL)
            .String("BSND");

        // 输出的数据布局，从fused_infer_attention_score的inputLayout拆分而来
        this->Attr("layout_out")
            .AttrType(OPTIONAL)
            .String("BSND");

        // 是否输出softmax_lse：1=输出（训练前向），0=不输出（推理）
        this->Attr("return_softmax_lse")
            .AttrType(OPTIONAL)
            .Int(0);

        // 是否使用确定性计算
        this->Attr("deterministic")
            .AttrType(OPTIONAL)
            .Int(0);

    }
};

OP_ADD(FlashAttention);

}  // namespace ops
