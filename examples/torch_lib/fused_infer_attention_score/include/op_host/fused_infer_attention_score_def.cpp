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
 * \file fused_infer_attention_score_def.cpp
 * \brief
 */
#include "register/op_def_registry.h"

// namespace optiling {
// bool TilingFusedInferAttentionScore(optiling::TilingContext* context);
// }

namespace ops {
class FusedInferAttentionScore : public OpDef {
public:
    FusedInferAttentionScore(const char *name) : OpDef(name)
    {
        this->Input("query")
            .ParamType(REQUIRED)
            .DataType({DT_FLOAT16, DT_FLOAT16, DT_BF16,    DT_INT8,    DT_INT8,    DT_FLOAT16,
                       DT_FLOAT16, DT_BF16,    DT_BF16,    DT_FLOAT16, DT_BF16,    DT_INT8,
                       DT_INT8,    DT_FLOAT16, DT_FLOAT16, DT_BF16,    DT_BF16,    DT_FLOAT16,
                       DT_BF16,    DT_INT8,    DT_INT8,    DT_FLOAT16, DT_FLOAT16, DT_BF16,
                       DT_BF16,    DT_BF16,    DT_BF16,    DT_FLOAT16, DT_FLOAT16, DT_BF16,
                       DT_BF16,    DT_BF16,    DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                       DT_BF16,    DT_FLOAT16, DT_BF16,    DT_FLOAT16, DT_FLOAT16, DT_BF16,
                       DT_INT8,    DT_INT8,    DT_FLOAT16, DT_FLOAT16, DT_BF16,    DT_BF16,
                       DT_FLOAT16, DT_BF16,    DT_INT8,    DT_INT8,    DT_FLOAT16, DT_FLOAT16,
                       DT_BF16,    DT_BF16,    DT_FLOAT16, DT_BF16,    DT_INT8,    DT_INT8,
                       DT_FLOAT16, DT_FLOAT16, DT_BF16,    DT_BF16,    DT_BF16,    DT_BF16,
                       DT_FLOAT16, DT_FLOAT16, DT_BF16,    DT_BF16,    DT_BF16,    DT_FLOAT16,
                       DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_BF16,    DT_FLOAT16, DT_BF16,
                       DT_BF16,    DT_BF16,    DT_FLOAT16, DT_FLOAT16, DT_INT8,    DT_BF16,
                       DT_BF16,    DT_BF16,    DT_FLOAT16, DT_BF16,    DT_FLOAT16, DT_BF16,
                       DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                       DT_BF16,    DT_FLOAT16, DT_FLOAT16, DT_BF16,    DT_BF16,    DT_INT8,
                       DT_INT8,    DT_BF16,    DT_BF16})
            .FormatList({FORMAT_ND})
            .AutoContiguous();
        this->Input("key")
            .ParamType(DYNAMIC)
            .DataType({DT_FLOAT16, DT_FLOAT16, DT_BF16,    DT_INT8,    DT_INT8,    DT_INT8,
                       DT_INT8,    DT_BF16,    DT_BF16,    DT_FLOAT16, DT_BF16,    DT_INT8,
                       DT_INT8,    DT_INT8,    DT_INT8,    DT_BF16,    DT_BF16,    DT_FLOAT16,
                       DT_BF16,    DT_INT8,    DT_INT8,    DT_INT8,    DT_INT8,    DT_BF16,
                       DT_BF16,    DT_INT8,    DT_INT8,    DT_INT8,    DT_INT8,    DT_INT8,
                       DT_INT8,    DT_BF16,    DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                       DT_INT8,    DT_FLOAT16, DT_INT8,    DT_FLOAT16, DT_FLOAT16, DT_BF16,
                       DT_INT8,    DT_INT8,    DT_INT8,    DT_INT8,    DT_BF16,    DT_BF16, // key datatype
                       DT_FLOAT16, DT_BF16,    DT_INT8,    DT_INT8,    DT_INT8,    DT_INT8,
                       DT_BF16,    DT_BF16,    DT_FLOAT16, DT_BF16,    DT_INT8,    DT_INT8,
                       DT_INT8,    DT_INT8,    DT_BF16,    DT_BF16,    DT_INT8,    DT_INT8,
                       DT_INT8,    DT_INT8,    DT_INT8,    DT_INT8,    DT_BF16,    DT_FLOAT16,
                       DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_INT8,    DT_FLOAT16, DT_INT8,
                       DT_INT8,    DT_INT8,    DT_INT8,    DT_INT8,    DT_INT8,    DT_INT8,
                       DT_INT8,    DT_INT8,    DT_INT4,    DT_INT4,    DT_INT4,    DT_INT4,
                       DT_INT8,    DT_INT8,    DT_INT8,    DT_INT8,    DT_INT8,    DT_INT4,
                       DT_INT4,    DT_FLOAT16, DT_FLOAT16, DT_INT8,    DT_INT8,    DT_INT8,
                       DT_INT8,    DT_INT8,    DT_INT8})
            .FormatList({FORMAT_ND})
            .AutoContiguous();
        this->Input("value")
            .ParamType(DYNAMIC)
            .DataType({DT_FLOAT16, DT_FLOAT16, DT_BF16,    DT_INT8,    DT_INT8,    DT_INT8,
                       DT_INT8,    DT_BF16,    DT_BF16,    DT_FLOAT16, DT_BF16,    DT_INT8,
                       DT_INT8,    DT_INT8,    DT_INT8,    DT_BF16,    DT_BF16,    DT_FLOAT16,
                       DT_BF16,    DT_INT8,    DT_INT8,    DT_INT8,    DT_INT8,    DT_BF16,
                       DT_BF16,    DT_INT8,    DT_INT8,    DT_INT8,    DT_INT8,    DT_INT8,
                       DT_INT8,    DT_BF16,    DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                       DT_INT8,    DT_FLOAT16, DT_INT8,    DT_FLOAT16, DT_FLOAT16, DT_BF16,
                       DT_INT8,    DT_INT8,    DT_INT8,    DT_INT8,    DT_BF16,    DT_BF16, // value datatype
                       DT_FLOAT16, DT_BF16,    DT_INT8,    DT_INT8,    DT_INT8,    DT_INT8,
                       DT_BF16,    DT_BF16,    DT_FLOAT16, DT_BF16,    DT_INT8,    DT_INT8,
                       DT_INT8,    DT_INT8,    DT_BF16,    DT_BF16,    DT_INT8,    DT_INT8,
                       DT_INT8,    DT_INT8,    DT_INT8,    DT_INT8,    DT_BF16,    DT_FLOAT16,
                       DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_INT8,    DT_FLOAT16, DT_INT8,
                       DT_INT8,    DT_INT8,    DT_INT8,    DT_INT8,    DT_INT8,    DT_INT8,
                       DT_INT8,    DT_INT8,    DT_INT4,    DT_INT4,    DT_INT4,    DT_INT4,
                       DT_INT8,    DT_INT8,    DT_INT8,    DT_INT8,    DT_INT8,    DT_INT4,
                       DT_INT4,    DT_FLOAT16, DT_FLOAT16, DT_INT8,    DT_INT8,    DT_INT8,
                       DT_INT8,    DT_INT8,    DT_INT8})
            .FormatList({FORMAT_ND})
            .AutoContiguous();
        this->Input("pse_shift")
            .ParamType(OPTIONAL)
            .DataType({DT_FLOAT16, DT_FLOAT16, DT_BF16,    DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                       DT_FLOAT16, DT_BF16,    DT_BF16,    DT_FLOAT16, DT_BF16,    DT_FLOAT16,
                       DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_BF16,    DT_BF16,    DT_FLOAT16,
                       DT_BF16,    DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_BF16,
                       DT_BF16,    DT_BF16,    DT_BF16,    DT_FLOAT16, DT_FLOAT16, DT_BF16,
                       DT_BF16,    DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                       DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_BF16,
                       DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_BF16,    DT_BF16,
                       DT_FLOAT16, DT_BF16,    DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                       DT_BF16,    DT_BF16,    DT_FLOAT16, DT_BF16,    DT_FLOAT16, DT_FLOAT16,
                       DT_FLOAT16, DT_FLOAT16, DT_BF16,    DT_BF16,    DT_BF16,    DT_BF16,
                       DT_FLOAT16, DT_FLOAT16, DT_BF16,    DT_BF16,    DT_FLOAT16, DT_FLOAT16,
                       DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                       DT_BF16,    DT_BF16,    DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_BF16,
                       DT_BF16,    DT_BF16,    DT_FLOAT16, DT_BF16,    DT_FLOAT16, DT_BF16,
                       DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                       DT_BF16,    DT_FLOAT16, DT_FLOAT16, DT_BF16,    DT_BF16,    DT_FLOAT16,
                       DT_FLOAT16, DT_BF16,    DT_BF16})
            .FormatList({FORMAT_ND})
            .AutoContiguous();
        this->Input("atten_mask")
            .ParamType(OPTIONAL)
            .DataType({DT_FLOAT16, DT_BOOL,  DT_BOOL,    DT_UINT8,   DT_UINT8, DT_BOOL,
                       DT_BOOL,    DT_BOOL,  DT_BOOL,    DT_FLOAT16, DT_BOOL,  DT_UINT8,
                       DT_UINT8,   DT_BOOL,  DT_BOOL,    DT_BOOL,    DT_BOOL,  DT_FLOAT16,
                       DT_BOOL,    DT_UINT8, DT_UINT8,   DT_BOOL,    DT_BOOL,  DT_BOOL,
                       DT_BOOL,    DT_BOOL,  DT_BOOL,    DT_BOOL,    DT_BOOL,  DT_BOOL,
                       DT_BOOL,    DT_BOOL,  DT_UINT8,   DT_BOOL,    DT_BOOL,  DT_BOOL,
                       DT_BOOL,    DT_INT8,  DT_BOOL,    DT_FLOAT16, DT_BOOL,  DT_BOOL,
                       DT_UINT8,   DT_UINT8, DT_BOOL,    DT_BOOL,    DT_BOOL,  DT_BOOL,
                       DT_FLOAT16, DT_BOOL,  DT_UINT8,   DT_UINT8,   DT_BOOL,  DT_BOOL,
                       DT_BOOL,    DT_BOOL,  DT_FLOAT16, DT_BOOL,    DT_UINT8, DT_UINT8,
                       DT_BOOL,    DT_BOOL,  DT_BOOL,    DT_BOOL,    DT_BOOL,  DT_BOOL,
                       DT_BOOL,    DT_BOOL,  DT_BOOL,    DT_BOOL,    DT_BOOL,  DT_UINT8,
                       DT_BOOL,    DT_BOOL,  DT_BOOL,    DT_BOOL,    DT_INT8,  DT_BOOL,
                       DT_BOOL,    DT_BOOL,  DT_BOOL,    DT_BOOL,    DT_INT8,  DT_INT8,
                       DT_BOOL,    DT_UINT8, DT_BOOL,    DT_BOOL,    DT_BOOL,  DT_BOOL,
                       DT_BOOL,    DT_INT8,  DT_UINT8,   DT_BOOL,    DT_BOOL,  DT_BOOL,
                       DT_BOOL,    DT_INT8,  DT_UINT8,   DT_UINT8,   DT_INT8,  DT_UINT8,
                       DT_INT8,    DT_INT8,  DT_UINT8})
            .FormatList({FORMAT_ND})
            .AutoContiguous();
        this->Input("actual_seq_lengths")
            .ParamType(OPTIONAL)
            .ValueDepend(OPTIONAL)
            .DataTypeList({DT_INT64})
            .FormatList({FORMAT_ND})
            .AutoContiguous();
        this->Input("actual_seq_lengths_kv")
            .ParamType(OPTIONAL)
            .ValueDepend(OPTIONAL)
            .DataTypeList({DT_INT64})
            .FormatList({FORMAT_ND})
            .AutoContiguous();
        this->Input("dequant_scale1")
            .ParamType(OPTIONAL)
            .DataType({DT_UINT64, DT_UINT64, DT_UINT64, DT_UINT64, DT_UINT64, DT_UINT64,
                       DT_UINT64, DT_UINT64, DT_UINT64, DT_UINT64, DT_UINT64, DT_UINT64,
                       DT_UINT64, DT_UINT64, DT_UINT64, DT_UINT64, DT_UINT64, DT_UINT64,
                       DT_UINT64, DT_UINT64, DT_UINT64, DT_UINT64, DT_UINT64, DT_UINT64,
                       DT_UINT64, DT_UINT64, DT_UINT64, DT_UINT64, DT_UINT64, DT_UINT64,
                       DT_UINT64, DT_UINT64, DT_UINT64, DT_UINT64, DT_UINT64, DT_UINT64,
                       DT_UINT64, DT_UINT64, DT_UINT64, DT_FLOAT,  DT_FLOAT,  DT_FLOAT,
                       DT_FLOAT,  DT_FLOAT,  DT_FLOAT,  DT_FLOAT,  DT_FLOAT,  DT_FLOAT, // dequant scale1 datatype
                       DT_FLOAT,  DT_FLOAT,  DT_FLOAT,  DT_FLOAT,  DT_FLOAT,  DT_FLOAT,
                       DT_FLOAT,  DT_FLOAT,  DT_FLOAT,  DT_FLOAT,  DT_FLOAT,  DT_FLOAT,
                       DT_FLOAT,  DT_FLOAT,  DT_FLOAT,  DT_FLOAT,  DT_FLOAT,  DT_FLOAT,
                       DT_FLOAT,  DT_FLOAT,  DT_FLOAT,  DT_FLOAT,  DT_FLOAT,  DT_FLOAT,
                       DT_FLOAT,  DT_FLOAT,  DT_FLOAT,  DT_FLOAT,  DT_FLOAT,  DT_FLOAT,
                       DT_FLOAT,  DT_FLOAT,  DT_FLOAT,  DT_FLOAT,  DT_FLOAT,  DT_UINT64,
                       DT_UINT64, DT_UINT64, DT_FLOAT,  DT_FLOAT,  DT_FLOAT,  DT_FLOAT,
                       DT_UINT64, DT_UINT64, DT_UINT64, DT_FLOAT,  DT_FLOAT,  DT_FLOAT,
                       DT_FLOAT,  DT_FLOAT,  DT_FLOAT,  DT_UINT64, DT_UINT64, DT_FLOAT,
                       DT_FLOAT,  DT_UINT64, DT_UINT64})
            .FormatList({FORMAT_ND})
            .AutoContiguous();
        this->Input("quant_scale1")
            .ParamType(OPTIONAL)
            .DataTypeList({DT_FLOAT})
            .FormatList({FORMAT_ND})
            .AutoContiguous();
        this->Input("dequant_scale2")
            .ParamType(OPTIONAL)
            .DataType({DT_UINT64, DT_UINT64, DT_UINT64, DT_UINT64, DT_UINT64, DT_UINT64,
                       DT_UINT64, DT_UINT64, DT_UINT64, DT_UINT64, DT_UINT64, DT_UINT64,
                       DT_UINT64, DT_UINT64, DT_UINT64, DT_UINT64, DT_UINT64, DT_UINT64,
                       DT_UINT64, DT_UINT64, DT_UINT64, DT_UINT64, DT_UINT64, DT_UINT64,
                       DT_UINT64, DT_UINT64, DT_UINT64, DT_UINT64, DT_UINT64, DT_UINT64,
                       DT_UINT64, DT_UINT64, DT_UINT64, DT_UINT64, DT_UINT64, DT_UINT64,
                       DT_UINT64, DT_UINT64, DT_UINT64, DT_FLOAT,  DT_FLOAT,  DT_FLOAT,
                       DT_FLOAT,  DT_FLOAT,  DT_FLOAT,  DT_FLOAT,  DT_FLOAT,  DT_FLOAT, // dequant scale2 datatype
                       DT_FLOAT,  DT_FLOAT,  DT_FLOAT,  DT_FLOAT,  DT_FLOAT,  DT_FLOAT,
                       DT_FLOAT,  DT_FLOAT,  DT_FLOAT,  DT_FLOAT,  DT_FLOAT,  DT_FLOAT,
                       DT_FLOAT,  DT_FLOAT,  DT_FLOAT,  DT_FLOAT,  DT_FLOAT,  DT_FLOAT,
                       DT_FLOAT,  DT_FLOAT,  DT_FLOAT,  DT_FLOAT,  DT_FLOAT,  DT_FLOAT,
                       DT_FLOAT,  DT_FLOAT,  DT_FLOAT,  DT_FLOAT,  DT_FLOAT,  DT_FLOAT,
                       DT_FLOAT,  DT_FLOAT,  DT_FLOAT,  DT_FLOAT,  DT_FLOAT,  DT_UINT64,
                       DT_UINT64, DT_UINT64, DT_FLOAT,  DT_FLOAT,  DT_FLOAT,  DT_FLOAT,
                       DT_UINT64, DT_UINT64, DT_UINT64, DT_FLOAT,  DT_FLOAT,  DT_FLOAT,
                       DT_FLOAT,  DT_FLOAT,  DT_FLOAT,  DT_UINT64, DT_UINT64, DT_FLOAT,
                       DT_FLOAT,  DT_UINT64, DT_UINT64})
            .FormatList({FORMAT_ND})
            .AutoContiguous();
        this->Input("quant_scale2")
            .ParamType(OPTIONAL)
            .DataType({DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                       DT_FLOAT, DT_BF16,  DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                       DT_FLOAT, DT_FLOAT, DT_BF16,  DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                       DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_BF16,  DT_FLOAT, DT_FLOAT, DT_FLOAT,
                       DT_FLOAT, DT_BF16,  DT_BF16,  DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                       DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_BF16,  DT_FLOAT, DT_FLOAT, DT_FLOAT,
                       DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_BF16,  DT_FLOAT,
                       DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_BF16, // quant scale2 datatype
                       DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                       DT_BF16,  DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_BF16,  DT_BF16,
                       DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                       DT_BF16,  DT_BF16,  DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                       DT_FLOAT, DT_BF16,  DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                       DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT16, DT_FLOAT16, DT_FLOAT,
                       DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_BF16, DT_BF16,   DT_FLOAT,
                       DT_FLOAT, DT_BF16,  DT_BF16})
            .FormatList({FORMAT_ND})
            .AutoContiguous();
        this->Input("quant_offset2")
            .ParamType(OPTIONAL)
            .DataType({DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                       DT_FLOAT, DT_BF16,  DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                       DT_FLOAT, DT_FLOAT, DT_BF16,  DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                       DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_BF16,  DT_FLOAT, DT_FLOAT, DT_FLOAT,
                       DT_FLOAT, DT_BF16,  DT_BF16,  DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                       DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_BF16,  DT_FLOAT, DT_FLOAT, DT_FLOAT,
                       DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_BF16,  DT_FLOAT,
                       DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_BF16, // quant offset2 datatype
                       DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                       DT_BF16,  DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_BF16,  DT_BF16,
                       DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                       DT_BF16,  DT_BF16,  DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                       DT_FLOAT, DT_BF16,  DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                       DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT16, DT_FLOAT16, DT_FLOAT,
                       DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_BF16, DT_BF16,   DT_FLOAT,
                       DT_FLOAT, DT_BF16,  DT_BF16})
            .FormatList({FORMAT_ND})
            .AutoContiguous();
        this->Input("antiquant_scale")
            .ParamType(OPTIONAL)
            .DataType({DT_FLOAT16, DT_FLOAT16, DT_BF16,    DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                       DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_BF16,    DT_FLOAT16,
                       DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                       DT_BF16,    DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                       DT_FLOAT16, DT_BF16,    DT_BF16,    DT_FLOAT,   DT_FLOAT,   DT_FLOAT,
                       DT_FLOAT,   DT_FLOAT,   DT_FLOAT,   DT_FLOAT,   DT_FLOAT,   DT_FLOAT,
                       DT_FLOAT,   DT_FLOAT,   DT_BF16,    DT_FLOAT16, DT_FLOAT16, DT_BF16,
                       DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, // antiquant scale datatype
                       DT_FLOAT16, DT_BF16,    DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                       DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_BF16,    DT_FLOAT16, DT_FLOAT16,
                       DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_BF16,    DT_BF16,
                       DT_FLOAT,   DT_FLOAT,   DT_FLOAT,   DT_FLOAT,   DT_FLOAT,   DT_FLOAT,
                       DT_FLOAT,   DT_FLOAT,   DT_FLOAT,   DT_FLOAT,   DT_FLOAT,   DT_BF16,
                       DT_FLOAT,   DT_BF16,    DT_FLOAT,   DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                       DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_BF16,    DT_FLOAT,   DT_FLOAT,
                       DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                       DT_BF16,    DT_FLOAT,   DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT,
                       DT_FLOAT,   DT_FLOAT,   DT_FLOAT})
            .FormatList({FORMAT_ND})
            .AutoContiguous();
        this->Input("antiquant_offset")
            .ParamType(OPTIONAL)
            .DataType({DT_FLOAT16, DT_FLOAT16, DT_BF16,    DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                       DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_BF16,    DT_FLOAT16,
                       DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                       DT_BF16,    DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                       DT_FLOAT16, DT_BF16,    DT_BF16,    DT_FLOAT,   DT_FLOAT,   DT_FLOAT,
                       DT_FLOAT,   DT_FLOAT,   DT_FLOAT,   DT_FLOAT,   DT_FLOAT,   DT_FLOAT,
                       DT_FLOAT,   DT_FLOAT,   DT_BF16,    DT_FLOAT16, DT_FLOAT16, DT_BF16,
                       DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, // antiquant offset datatype
                       DT_FLOAT16, DT_BF16,    DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                       DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_BF16,    DT_FLOAT16, DT_FLOAT16,
                       DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_BF16,    DT_BF16,
                       DT_FLOAT,   DT_FLOAT,   DT_FLOAT,   DT_FLOAT,   DT_FLOAT,   DT_FLOAT,
                       DT_FLOAT,   DT_FLOAT,   DT_FLOAT,   DT_FLOAT,   DT_FLOAT,   DT_BF16,
                       DT_FLOAT,   DT_BF16,    DT_FLOAT,   DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                       DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_BF16,    DT_FLOAT,   DT_FLOAT,
                       DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                       DT_BF16,    DT_FLOAT,   DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT,
                       DT_FLOAT,   DT_FLOAT,   DT_FLOAT})
            .FormatList({FORMAT_ND})
            .AutoContiguous();
        this->Input("block_table")
            .ParamType(OPTIONAL)
            .DataTypeList({DT_INT32})
            .FormatList({FORMAT_ND})
            .AutoContiguous();
        this->Input("query_padding_size")
            .ParamType(OPTIONAL)
            .DataTypeList({DT_INT64})
            .FormatList({FORMAT_ND})
            .AutoContiguous();
        this->Input("kv_padding_size")
            .ParamType(OPTIONAL)
            .DataTypeList({DT_INT64})
            .FormatList({FORMAT_ND})
            .AutoContiguous();
        this->Input("key_antiquant_scale")
            .ParamType(OPTIONAL)
            .DataType({DT_FLOAT16, DT_FLOAT16, DT_BF16,    DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                       DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_BF16,    DT_FLOAT16,
                       DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                       DT_BF16,    DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                       DT_FLOAT16, DT_BF16,    DT_BF16,    DT_FLOAT,   DT_FLOAT,   DT_FLOAT,
                       DT_FLOAT,   DT_FLOAT,   DT_FLOAT,   DT_FLOAT,   DT_FLOAT,   DT_FLOAT,
                       DT_FLOAT,   DT_FLOAT,   DT_BF16,    DT_FLOAT16, DT_FLOAT16, DT_BF16,
                       DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, // key antiquant scale datatype
                       DT_FLOAT16, DT_BF16,    DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                       DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_BF16,    DT_FLOAT16, DT_FLOAT16,
                       DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_BF16,    DT_BF16,
                       DT_FLOAT,   DT_FLOAT,   DT_FLOAT,   DT_FLOAT,   DT_FLOAT,   DT_FLOAT,
                       DT_FLOAT,   DT_FLOAT,   DT_FLOAT,   DT_FLOAT,   DT_FLOAT,   DT_BF16,
                       DT_BF16,    DT_FLOAT,   DT_FLOAT16, DT_FLOAT,   DT_FLOAT16, DT_FLOAT,
                       DT_FLOAT,   DT_BF16,    DT_FLOAT16, DT_BF16,    DT_FLOAT,   DT_FLOAT,
                       DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT,   DT_FLOAT,   DT_FLOAT16,
                       DT_BF16,    DT_FLOAT,   DT_FLOAT16, DT_BF16,    DT_BF16,    DT_FLOAT,
                       DT_FLOAT,   DT_FLOAT,   DT_FLOAT})
            .FormatList({FORMAT_ND})
            .AutoContiguous();
        this->Input("key_antiquant_offset")
            .ParamType(OPTIONAL)
            .DataType({DT_FLOAT16, DT_FLOAT16, DT_BF16,    DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                       DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_BF16,    DT_FLOAT16,
                       DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                       DT_BF16,    DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                       DT_FLOAT16, DT_BF16,    DT_BF16,    DT_FLOAT,   DT_FLOAT,   DT_FLOAT,
                       DT_FLOAT,   DT_FLOAT,   DT_FLOAT,   DT_FLOAT,   DT_FLOAT,   DT_FLOAT,
                       DT_FLOAT,   DT_FLOAT,   DT_BF16,    DT_FLOAT16, DT_FLOAT16, DT_BF16,
                       DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, // key antiquant offset datatype
                       DT_FLOAT16, DT_BF16,    DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                       DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_BF16,    DT_FLOAT16, DT_FLOAT16,
                       DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_BF16,    DT_BF16,
                       DT_FLOAT,   DT_FLOAT,   DT_FLOAT,   DT_FLOAT,   DT_FLOAT,   DT_FLOAT,
                       DT_FLOAT,   DT_FLOAT,   DT_FLOAT,   DT_FLOAT,   DT_FLOAT,   DT_BF16,
                       DT_BF16,    DT_FLOAT,   DT_FLOAT16, DT_FLOAT,   DT_FLOAT16, DT_FLOAT,
                       DT_FLOAT,   DT_BF16,    DT_FLOAT16, DT_BF16,    DT_FLOAT,   DT_FLOAT,
                       DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT,   DT_FLOAT,   DT_FLOAT16,
                       DT_BF16,    DT_FLOAT,   DT_FLOAT16, DT_BF16,    DT_BF16,    DT_FLOAT,
                       DT_FLOAT,   DT_FLOAT,   DT_FLOAT})
            .FormatList({FORMAT_ND})
            .AutoContiguous();
        this->Input("value_antiquant_scale")
            .ParamType(OPTIONAL)
            .DataType({DT_FLOAT16, DT_FLOAT16, DT_BF16,    DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                       DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_BF16,    DT_FLOAT16,
                       DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                       DT_BF16,    DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                       DT_FLOAT16, DT_BF16,    DT_BF16,    DT_FLOAT,   DT_FLOAT,   DT_FLOAT,
                       DT_FLOAT,   DT_FLOAT,   DT_FLOAT,   DT_FLOAT,   DT_FLOAT,   DT_FLOAT,
                       DT_FLOAT,   DT_FLOAT,   DT_BF16,    DT_FLOAT16, DT_FLOAT16, DT_BF16,
                       DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, // value antiquant scale datatype
                       DT_FLOAT16, DT_BF16,    DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                       DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_BF16,    DT_FLOAT16, DT_FLOAT16,
                       DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_BF16,    DT_BF16,
                       DT_FLOAT,   DT_FLOAT,   DT_FLOAT,   DT_FLOAT,   DT_FLOAT,   DT_FLOAT,
                       DT_FLOAT,   DT_FLOAT,   DT_FLOAT,   DT_FLOAT,   DT_FLOAT,   DT_BF16,
                       DT_BF16,    DT_FLOAT,   DT_FLOAT16, DT_FLOAT,   DT_FLOAT16, DT_FLOAT,
                       DT_FLOAT,   DT_BF16,    DT_FLOAT16, DT_BF16,    DT_FLOAT,   DT_FLOAT,
                       DT_FLOAT,   DT_FLOAT,   DT_FLOAT,   DT_FLOAT,   DT_FLOAT,   DT_FLOAT,
                       DT_FLOAT,   DT_FLOAT,   DT_FLOAT16, DT_BF16,    DT_BF16,    DT_FLOAT,
                       DT_FLOAT,   DT_FLOAT,   DT_FLOAT})
            .FormatList({FORMAT_ND})
            .AutoContiguous();
        this->Input("value_antiquant_offset")
            .ParamType(OPTIONAL)
            .DataType({DT_FLOAT16, DT_FLOAT16, DT_BF16,    DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                       DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_BF16,    DT_FLOAT16,
                       DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                       DT_BF16,    DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                       DT_FLOAT16, DT_BF16,    DT_BF16,    DT_FLOAT,   DT_FLOAT,   DT_FLOAT,
                       DT_FLOAT,   DT_FLOAT,   DT_FLOAT,   DT_FLOAT,   DT_FLOAT,   DT_FLOAT,
                       DT_FLOAT,   DT_FLOAT,   DT_BF16,    DT_FLOAT16, DT_FLOAT16, DT_BF16,
                       DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, // value antiquant offset datatype
                       DT_FLOAT16, DT_BF16,    DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                       DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_BF16,    DT_FLOAT16, DT_FLOAT16,
                       DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_BF16,    DT_BF16,
                       DT_FLOAT,   DT_FLOAT,   DT_FLOAT,   DT_FLOAT,   DT_FLOAT,   DT_FLOAT,
                       DT_FLOAT,   DT_FLOAT,   DT_FLOAT,   DT_FLOAT,   DT_FLOAT,   DT_BF16,
                       DT_BF16,    DT_FLOAT,   DT_FLOAT16, DT_FLOAT,   DT_FLOAT16, DT_FLOAT,
                       DT_FLOAT,   DT_BF16,    DT_FLOAT16, DT_BF16,    DT_FLOAT,   DT_FLOAT,
                       DT_FLOAT,   DT_FLOAT,   DT_FLOAT,   DT_FLOAT,   DT_FLOAT,   DT_FLOAT,
                       DT_FLOAT,   DT_FLOAT,   DT_FLOAT16, DT_BF16,    DT_BF16,    DT_FLOAT,
                       DT_FLOAT,   DT_FLOAT,   DT_FLOAT})
            .FormatList({FORMAT_ND})
            .AutoContiguous();
        this->Input("key_shared_prefix")
            .ParamType(OPTIONAL)
            .DataType({DT_FLOAT16, DT_FLOAT16, DT_BF16,    DT_INT8,    DT_INT8,    DT_INT8,
                       DT_INT8,    DT_BF16,    DT_BF16,    DT_FLOAT16, DT_BF16,    DT_INT8,
                       DT_INT8,    DT_INT8,    DT_INT8,    DT_BF16,    DT_BF16,    DT_FLOAT16,
                       DT_BF16,    DT_INT8,    DT_INT8,    DT_INT8,    DT_INT8,    DT_BF16,
                       DT_BF16,    DT_INT8,    DT_INT8,    DT_INT8,    DT_INT8,    DT_INT8,
                       DT_INT8,    DT_BF16,    DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                       DT_INT8,    DT_FLOAT16, DT_INT8,    DT_FLOAT16, DT_FLOAT16, DT_BF16,
                       DT_INT8,    DT_INT8,    DT_INT8,    DT_INT8,    DT_BF16,    DT_BF16, // key shared prefix datatype
                       DT_FLOAT16, DT_BF16,    DT_INT8,    DT_INT8,    DT_INT8,    DT_INT8,
                       DT_BF16,    DT_BF16,    DT_FLOAT16, DT_BF16,    DT_INT8,    DT_INT8,
                       DT_INT8,    DT_INT8,    DT_BF16,    DT_BF16,    DT_INT8,    DT_INT8,
                       DT_INT8,    DT_INT8,    DT_INT8,    DT_INT8,    DT_BF16,    DT_FLOAT16,
                       DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_INT8,    DT_FLOAT16, DT_INT8,
                       DT_INT8,    DT_INT8,    DT_INT8,    DT_INT8,    DT_INT8,    DT_FLOAT16,
                       DT_FLOAT,   DT_FLOAT16, DT_INT4,    DT_INT4,    DT_INT4,    DT_INT4,
                       DT_INT8,    DT_INT8,    DT_INT8,    DT_INT8,    DT_INT8,    DT_INT4,
                       DT_INT4,    DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_INT8,
                       DT_INT8,    DT_INT8,    DT_INT8})
            .FormatList({FORMAT_ND})
            .AutoContiguous();
        this->Input("value_shared_prefix")
            .ParamType(OPTIONAL)
            .DataType({DT_FLOAT16, DT_FLOAT16, DT_BF16,    DT_INT8,    DT_INT8,    DT_INT8,
                       DT_INT8,    DT_BF16,    DT_BF16,    DT_FLOAT16, DT_BF16,    DT_INT8,
                       DT_INT8,    DT_INT8,    DT_INT8,    DT_BF16,    DT_BF16,    DT_FLOAT16,
                       DT_BF16,    DT_INT8,    DT_INT8,    DT_INT8,    DT_INT8,    DT_BF16,
                       DT_BF16,    DT_INT8,    DT_INT8,    DT_INT8,    DT_INT8,    DT_INT8,
                       DT_INT8,    DT_BF16,    DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                       DT_INT8,    DT_FLOAT16, DT_INT8,    DT_FLOAT16, DT_FLOAT16, DT_BF16,
                       DT_INT8,    DT_INT8,    DT_INT8,    DT_INT8,    DT_BF16,    DT_BF16, // value shared prefix datatype
                       DT_FLOAT16, DT_BF16,    DT_INT8,    DT_INT8,    DT_INT8,    DT_INT8,
                       DT_BF16,    DT_BF16,    DT_FLOAT16, DT_BF16,    DT_INT8,    DT_INT8,
                       DT_INT8,    DT_INT8,    DT_BF16,    DT_BF16,    DT_INT8,    DT_INT8,
                       DT_INT8,    DT_INT8,    DT_INT8,    DT_INT8,    DT_BF16,    DT_FLOAT16,
                       DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_INT8,    DT_FLOAT16, DT_INT8,
                       DT_INT8,    DT_INT8,    DT_INT8,    DT_INT8,    DT_INT8,    DT_FLOAT16,
                       DT_FLOAT,   DT_FLOAT16, DT_INT4,    DT_INT4,    DT_INT4,    DT_INT4,
                       DT_INT8,    DT_INT8,    DT_INT8,    DT_INT8,    DT_INT8,    DT_INT4,
                       DT_INT4,    DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_INT8,
                       DT_INT8,    DT_INT8,    DT_INT8})
            .FormatList({FORMAT_ND})
            .AutoContiguous();
        this->Input("actual_shared_prefix_len")
            .ParamType(OPTIONAL)
            .ValueDepend(OPTIONAL)
            .DataTypeList({DT_INT64})
            .FormatList({FORMAT_ND})
            .AutoContiguous();
        this->Input("query_rope")
            .ParamType(OPTIONAL)
            .DataType({DT_FLOAT16, DT_FLOAT16, DT_BF16,    DT_INT8,    DT_INT8,    DT_FLOAT16,
                       DT_FLOAT16, DT_BF16,    DT_BF16,    DT_FLOAT16, DT_BF16,    DT_INT8,
                       DT_INT8,    DT_FLOAT16, DT_FLOAT16, DT_BF16,    DT_BF16,    DT_FLOAT16,
                       DT_BF16,    DT_INT8,    DT_INT8,    DT_FLOAT16, DT_FLOAT16, DT_BF16,
                       DT_BF16,    DT_BF16,    DT_BF16,    DT_FLOAT16, DT_FLOAT16, DT_BF16,
                       DT_BF16,    DT_BF16,    DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                       DT_BF16,    DT_FLOAT16, DT_BF16,    DT_FLOAT16, DT_FLOAT16, DT_BF16,
                       DT_INT8,    DT_INT8,    DT_FLOAT16, DT_FLOAT16, DT_BF16,    DT_BF16,
                       DT_FLOAT16, DT_BF16,    DT_INT8,    DT_INT8,    DT_FLOAT16, DT_FLOAT16,
                       DT_BF16,    DT_BF16,    DT_FLOAT16, DT_BF16,    DT_INT8,    DT_INT8,
                       DT_FLOAT16, DT_FLOAT16, DT_BF16,    DT_BF16,    DT_BF16,    DT_BF16,
                       DT_FLOAT16, DT_FLOAT16, DT_BF16,    DT_BF16,    DT_BF16,    DT_FLOAT16,
                       DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_BF16,    DT_FLOAT16, DT_BF16,
                       DT_BF16,    DT_BF16,    DT_FLOAT16, DT_FLOAT16, DT_INT8,    DT_BF16,
                       DT_BF16,    DT_BF16,    DT_FLOAT16, DT_BF16,    DT_FLOAT16, DT_BF16,
                       DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                       DT_BF16,    DT_FLOAT16, DT_FLOAT16, DT_BF16,    DT_BF16,    DT_BF16,
                       DT_BF16,    DT_BF16,    DT_BF16})
            .FormatList({FORMAT_ND})
            .AutoContiguous();
        this->Input("key_rope")
            .ParamType(OPTIONAL)
            .DataType({DT_FLOAT16, DT_FLOAT16, DT_BF16,    DT_INT8,    DT_INT8,    DT_INT8,
                       DT_INT8,    DT_BF16,    DT_BF16,    DT_FLOAT16, DT_BF16,    DT_INT8,
                       DT_INT8,    DT_INT8,    DT_INT8,    DT_BF16,    DT_BF16,    DT_FLOAT16,
                       DT_BF16,    DT_INT8,    DT_INT8,    DT_INT8,    DT_INT8,    DT_BF16,
                       DT_BF16,    DT_INT8,    DT_INT8,    DT_INT8,    DT_INT8,    DT_INT8,
                       DT_INT8,    DT_BF16,    DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                       DT_INT8,    DT_FLOAT16, DT_INT8,    DT_FLOAT16, DT_FLOAT16, DT_BF16,
                       DT_INT8,    DT_INT8,    DT_INT8,    DT_INT8,    DT_BF16,    DT_BF16, // key datatype
                       DT_FLOAT16, DT_BF16,    DT_INT8,    DT_INT8,    DT_INT8,    DT_INT8,
                       DT_BF16,    DT_BF16,    DT_FLOAT16, DT_BF16,    DT_INT8,    DT_INT8,
                       DT_INT8,    DT_INT8,    DT_BF16,    DT_BF16,    DT_INT8,    DT_INT8,
                       DT_INT8,    DT_INT8,    DT_INT8,    DT_INT8,    DT_BF16,    DT_FLOAT16,
                       DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_INT8,    DT_FLOAT16, DT_INT8,
                       DT_INT8,    DT_INT8,    DT_INT8,    DT_INT8,    DT_INT8,    DT_INT8,
                       DT_INT8,    DT_INT8,    DT_INT4,    DT_INT4,    DT_INT4,    DT_INT4,
                       DT_INT8,    DT_INT8,    DT_INT8,    DT_INT8,    DT_INT8,    DT_INT4,
                       DT_INT4,    DT_FLOAT16, DT_FLOAT16, DT_INT8,    DT_INT8,    DT_BF16,
                       DT_BF16,    DT_INT8,    DT_INT8})
            .FormatList({FORMAT_ND})
            .AutoContiguous();
        this->Input("key_rope_antiquant_scale")
            .ParamType(OPTIONAL)
            .DataType({DT_FLOAT16, DT_FLOAT16, DT_BF16,    DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                       DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_BF16,    DT_FLOAT16,
                       DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                       DT_BF16,    DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                       DT_FLOAT16, DT_BF16,    DT_BF16,    DT_FLOAT,   DT_FLOAT,   DT_FLOAT,
                       DT_FLOAT,   DT_FLOAT,   DT_FLOAT,   DT_FLOAT,   DT_FLOAT,   DT_FLOAT,
                       DT_FLOAT,   DT_FLOAT,   DT_BF16,    DT_FLOAT16, DT_FLOAT16, DT_BF16,
                       DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, // key rope antiquant scale datatype
                       DT_FLOAT16, DT_BF16,    DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                       DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_BF16,    DT_FLOAT16, DT_FLOAT16,
                       DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_BF16,    DT_BF16,
                       DT_FLOAT,   DT_FLOAT,   DT_FLOAT,   DT_FLOAT,   DT_FLOAT,   DT_FLOAT,
                       DT_FLOAT,   DT_FLOAT,   DT_FLOAT,   DT_FLOAT,   DT_FLOAT,   DT_BF16,
                       DT_BF16,    DT_FLOAT,   DT_FLOAT16, DT_FLOAT,   DT_FLOAT16, DT_FLOAT,
                       DT_FLOAT,   DT_BF16,    DT_FLOAT16, DT_BF16,    DT_FLOAT,   DT_FLOAT,
                       DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT,   DT_FLOAT,   DT_FLOAT16,
                       DT_BF16,    DT_FLOAT,   DT_FLOAT16, DT_BF16,   DT_BF16,     DT_BF16,
                       DT_BF16,    DT_FLOAT,   DT_FLOAT})
            .FormatList({FORMAT_ND})
            .AutoContiguous();
        this->Input("dequant_scale_query")
            .ParamType(OPTIONAL)
            .DataTypeList({DT_FLOAT})
            .FormatList({FORMAT_ND})
            .AutoContiguous();
        this->Input("learnable_sink")
            .ParamType(OPTIONAL)
            .DataTypeList({DT_BF16, DT_FLOAT16})
            .FormatList({FORMAT_ND})
            .AutoContiguous();
        this->Input("q_start_idx")
            .ParamType(OPTIONAL)
            .ValueDepend(OPTIONAL)
            .DataTypeList({DT_INT64})
            .FormatList({FORMAT_ND})
            .AutoContiguous();
        this->Input("kv_start_idx")
            .ParamType(OPTIONAL)
            .ValueDepend(OPTIONAL)
            .DataTypeList({DT_INT64})
            .FormatList({FORMAT_ND})
            .AutoContiguous();
        this->Output("attention_out")
            .ParamType(REQUIRED)
            .DataType({DT_FLOAT16, DT_FLOAT16, DT_BF16,    DT_INT8,    DT_FLOAT16, DT_FLOAT16,
                       DT_INT8,    DT_INT8,    DT_INT8,    DT_FLOAT16, DT_BF16,    DT_INT8,
                       DT_FLOAT16, DT_FLOAT16, DT_INT8,    DT_INT8,    DT_INT8,    DT_FLOAT16,
                       DT_BF16,    DT_INT8,    DT_FLOAT16, DT_FLOAT16, DT_INT8,    DT_INT8,
                       DT_INT8,    DT_BF16,    DT_INT8,    DT_FLOAT16, DT_INT8,    DT_BF16,
                       DT_INT8,    DT_BF16,    DT_FLOAT16, DT_INT8,    DT_INT8,    DT_INT8,
                       DT_INT8,    DT_INT8,    DT_INT8,    DT_FLOAT16, DT_FLOAT16, DT_BF16,
                       DT_INT8,    DT_FLOAT16, DT_FLOAT16, DT_INT8,    DT_INT8,    DT_INT8,
                       DT_FLOAT16, DT_BF16,    DT_INT8,    DT_FLOAT16, DT_FLOAT16, DT_INT8,
                       DT_INT8,    DT_INT8,    DT_FLOAT16, DT_BF16,    DT_INT8,    DT_FLOAT16,
                       DT_FLOAT16, DT_INT8,    DT_INT8,    DT_INT8,    DT_BF16,    DT_INT8,
                       DT_FLOAT16, DT_INT8,    DT_BF16,    DT_INT8,    DT_BF16,    DT_FLOAT16,
                       DT_INT8,    DT_INT8,    DT_INT8,    DT_INT8,    DT_INT8,    DT_INT8,
                       DT_BF16,    DT_BF16,    DT_FLOAT16, DT_FLOAT16, DT_INT8,    DT_INT8,
                       DT_INT8,    DT_INT8,    DT_FLOAT16, DT_BF16,    DT_FLOAT16, DT_BF16,
                       DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_INT8,    DT_FLOAT16, DT_FLOAT16,
                       DT_BF16,    DT_FLOAT16, DT_INT8,    DT_BF16,    DT_BF16,    DT_BF16,
                       DT_BF16,    DT_BF16,    DT_BF16})
            .FormatList({FORMAT_ND});
        this->Output("softmax_lse").ParamType(REQUIRED).DataTypeList({DT_FLOAT}).FormatList({FORMAT_ND});
        this->Attr("num_heads").AttrType(REQUIRED).Int(1);
        this->Attr("scale").AttrType(OPTIONAL).Float(1.0);
        this->Attr("pre_tokens").AttrType(OPTIONAL).Int(2147483647); // 2147483647: Maximum value of int32_t.
        this->Attr("next_tokens").AttrType(OPTIONAL).Int(2147483647); // 2147483647: Maximum value of int32_t.
        this->Attr("input_layout").AttrType(OPTIONAL).String("BSH");
        this->Attr("num_key_value_heads").AttrType(OPTIONAL).Int(0);
        this->Attr("sparse_mode").AttrType(OPTIONAL).Int(0);
        this->Attr("inner_precise").AttrType(OPTIONAL).Int(1);
        this->Attr("block_size").AttrType(OPTIONAL).Int(0);
        this->Attr("antiquant_mode").AttrType(OPTIONAL).Int(0);
        this->Attr("softmax_lse_flag").AttrType(OPTIONAL).Bool(false);
        this->Attr("key_antiquant_mode").AttrType(OPTIONAL).Int(0);
        this->Attr("value_antiquant_mode").AttrType(OPTIONAL).Int(0);
        this->Attr("query_quant_mode").AttrType(OPTIONAL).Int(0);
        this->Attr("pse_type").AttrType(OPTIONAL).Int(0);
        this->Attr("out_dtype").AttrType(OPTIONAL).Int(0);
        OpAICoreConfig aicore_config;
        aicore_config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn")
            .ExtendCfgInfo("opFile.value", "fused_infer_attention_score")
            .ExtendCfgInfo("jitCompile.flag", "static_false,dynamic_false");
        this->AICore().AddConfig("ascend910b", aicore_config); // use 910B
        this->AICore().AddConfig("ascend910_93", aicore_config);

        OpAICoreConfig aicore_config_95;
        aicore_config_95.Input("query")
            .ParamType(REQUIRED)
            .DataType({DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_INT8, DT_INT8, DT_FLOAT16,
                        DT_FLOAT16, DT_BF16, DT_BF16, DT_FLOAT16, DT_BF16, DT_INT8,
                        DT_INT8, DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_BF16, DT_FLOAT16,
                        DT_BF16, DT_INT8, DT_INT8, DT_FLOAT16, DT_FLOAT16, DT_BF16,
                        DT_BF16, DT_BF16, DT_BF16, DT_FLOAT16, DT_FLOAT16, DT_BF16,
                        DT_BF16, DT_BF16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                        DT_BF16, DT_FLOAT16, DT_BF16, DT_FLOAT16, DT_FLOAT16, DT_BF16,
                        DT_INT8, DT_INT8, DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_BF16,
                        DT_FLOAT16, DT_BF16, DT_INT8, DT_INT8, DT_FLOAT16, DT_FLOAT16,
                        DT_BF16, DT_BF16, DT_FLOAT16, DT_BF16, DT_INT8, DT_INT8,
                        DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_BF16, DT_BF16, DT_BF16,
                        DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_BF16, DT_BF16, DT_FLOAT16,
                        DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_FLOAT16, DT_BF16,
                        DT_BF16, DT_BF16, DT_FLOAT16, DT_FLOAT16, DT_INT8, DT_BF16,
                        DT_BF16, DT_BF16, DT_FLOAT16, DT_BF16, DT_FLOAT16, DT_BF16,
                        DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                        DT_BF16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                        DT_BF16, DT_BF16, DT_BF16, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT8, DT_FLOAT16, DT_BF16, DT_BF16, DT_BF16, DT_BF16,
                        DT_BF16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_BF16,
                        DT_BF16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                        DT_FLOAT16, DT_BF16, DT_BF16, DT_BF16, DT_BF16, DT_BF16,
                        DT_BF16, DT_BF16, DT_BF16, DT_BF16, DT_BF16, DT_BF16,
                        DT_BF16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_BF16,
                        DT_BF16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_BF16,
                        DT_BF16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_BF16,
                        DT_BF16, DT_BF16, DT_BF16, DT_BF16, DT_BF16, DT_BF16,
                        DT_BF16, DT_BF16, DT_BF16, DT_BF16, DT_BF16, DT_BF16,
                        DT_BF16, DT_BF16, DT_BF16, DT_BF16, DT_BF16, DT_BF16,
                        DT_BF16, DT_BF16, DT_FLOAT8_E4M3FN, DT_FLOAT8_E4M3FN, DT_FLOAT16, DT_FLOAT16,
                        DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_BF16,
                        DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_BF16, DT_BF16, DT_BF16,
                        DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                        DT_BF16, DT_BF16, DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_BF16,
                        DT_BF16, DT_BF16, DT_BF16, DT_BF16, DT_FLOAT16, DT_FLOAT16,
                        DT_BF16, DT_BF16, DT_BF16, DT_FLOAT16, DT_BF16, DT_FLOAT16,
                        DT_BF16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                        DT_FLOAT16, DT_BF16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_BF16,
                        DT_BF16, DT_BF16, DT_FLOAT16, DT_BF16, DT_BF16, DT_BF16,
                        DT_BF16, DT_BF16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                        DT_BF16, DT_BF16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                        DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_BF16, DT_FLOAT16, DT_FLOAT16,
                        DT_BF16, DT_BF16, DT_BF16, DT_BF16, DT_FLOAT16, DT_FLOAT16,
                        DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_BF16,
                        DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_BF16, DT_BF16, DT_BF16,
                        DT_BF16, DT_BF16, DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_BF16,
                        DT_BF16, DT_FLOAT16, DT_BF16, DT_FLOAT16, DT_BF16, DT_FLOAT16,
                        DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_BF16,
                        DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_BF16, DT_BF16,
                        DT_FLOAT16, DT_BF16, DT_BF16, DT_BF16, DT_BF16, DT_BF16,
                        DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_BF16,
                        DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                        DT_BF16, DT_BF16, DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_BF16,
                        DT_BF16, DT_BF16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                        DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_BF16, DT_FLOAT16, DT_FLOAT16,
                        DT_BF16, DT_BF16, DT_BF16, DT_BF16, DT_BF16, DT_BF16,
                        DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_BF16, DT_BF16, DT_FLOAT16,
                        DT_BF16, DT_FLOAT16, DT_BF16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                        DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_FLOAT16, DT_FLOAT16,
                        DT_FLOAT16, DT_BF16, DT_BF16, DT_BF16, DT_FLOAT16, DT_BF16,
                        DT_BF16, DT_BF16, DT_BF16, DT_BF16, DT_FLOAT16, DT_FLOAT16,
                        DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_BF16, DT_BF16, DT_HIFLOAT8,
                        DT_HIFLOAT8, DT_BF16})
            .FormatList({FORMAT_ND})
            .AutoContiguous();
        aicore_config_95.Input("key")
            .ParamType(DYNAMIC)
            .DataType({DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT8, DT_BF16, DT_BF16, DT_FLOAT16, DT_BF16, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT8, DT_BF16, DT_BF16, DT_FLOAT16,
                        DT_BF16, DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_BF16,
                        DT_BF16, DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT8, DT_BF16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                        DT_INT8, DT_FLOAT16, DT_INT8, DT_FLOAT16, DT_FLOAT16, DT_BF16,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_BF16, DT_BF16,	// key datatype
                        DT_FLOAT16, DT_BF16, DT_INT8, DT_INT8, DT_INT8, DT_INT8,
                        DT_BF16, DT_BF16, DT_FLOAT16, DT_BF16, DT_INT8, DT_INT8,
                        DT_INT8, DT_INT8, DT_BF16, DT_BF16, DT_INT8, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_BF16, DT_FLOAT16,
                        DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_INT8, DT_FLOAT16, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT4, DT_INT4, DT_INT4, DT_INT4,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT4,
                        DT_INT4, DT_FLOAT16, DT_FLOAT16, DT_HIFLOAT8, DT_FLOAT8_E4M3FN, DT_FLOAT4_E2M1,
                        DT_HIFLOAT8, DT_FLOAT8_E4M3FN, DT_FLOAT4_E2M1, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT8, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                        DT_FLOAT16, DT_BF16, DT_BF16, DT_BF16, DT_BF16, DT_BF16,
                        DT_BF16, DT_BF16, DT_BF16, DT_BF16, DT_BF16, DT_BF16,
                        DT_BF16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_BF16,
                        DT_BF16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_BF16,
                        DT_BF16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_BF16,
                        DT_BF16, DT_BF16, DT_BF16, DT_BF16, DT_BF16, DT_BF16,
                        DT_BF16, DT_BF16, DT_BF16, DT_BF16, DT_BF16, DT_BF16,
                        DT_BF16, DT_BF16, DT_BF16, DT_BF16, DT_BF16, DT_BF16,
                        DT_BF16, DT_BF16, DT_FLOAT8_E4M3FN, DT_FLOAT8_E4M3FN, DT_INT8, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT4, DT_INT4, DT_INT4,
                        DT_INT4, DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT4, DT_INT4, DT_HIFLOAT8, DT_FLOAT8_E4M3FN, DT_FLOAT4_E2M1, DT_HIFLOAT8,
                        DT_FLOAT8_E4M3FN, DT_FLOAT4_E2M1, DT_INT8, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT8, DT_INT4, DT_INT4, DT_INT4, DT_INT4, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT4, DT_INT4,
                        DT_HIFLOAT8, DT_FLOAT8_E4M3FN, DT_FLOAT4_E2M1, DT_HIFLOAT8, DT_FLOAT8_E4M3FN, DT_FLOAT4_E2M1,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT4,
                        DT_INT4, DT_INT4, DT_INT4, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT4, DT_INT4, DT_HIFLOAT8, DT_FLOAT8_E4M3FN,
                        DT_FLOAT4_E2M1, DT_HIFLOAT8, DT_FLOAT8_E4M3FN, DT_FLOAT4_E2M1, DT_INT8, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_BF16, DT_HIFLOAT8,
                        DT_HIFLOAT8, DT_INT8})
            .FormatList({FORMAT_ND})
            .AutoContiguous();
        aicore_config_95.Input("value")
            .ParamType(DYNAMIC)
            .DataType({DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT8, DT_BF16, DT_BF16, DT_FLOAT16, DT_BF16, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT8, DT_BF16, DT_BF16, DT_FLOAT16,
                        DT_BF16, DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_BF16,
                        DT_BF16, DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT8, DT_BF16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                        DT_INT8, DT_FLOAT16, DT_INT8, DT_FLOAT16, DT_FLOAT16, DT_BF16,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_BF16, DT_BF16,	// value datatype
                        DT_FLOAT16, DT_BF16, DT_INT8, DT_INT8, DT_INT8, DT_INT8,
                        DT_BF16, DT_BF16, DT_FLOAT16, DT_BF16, DT_INT8, DT_INT8,
                        DT_INT8, DT_INT8, DT_BF16, DT_BF16, DT_INT8, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_BF16, DT_FLOAT16,
                        DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_INT8, DT_FLOAT16, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT4, DT_INT4, DT_INT4, DT_INT4,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT4,
                        DT_INT4, DT_FLOAT16, DT_FLOAT16, DT_HIFLOAT8, DT_FLOAT8_E4M3FN, DT_FLOAT4_E2M1,
                        DT_HIFLOAT8, DT_FLOAT8_E4M3FN, DT_FLOAT4_E2M1, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT8, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                        DT_FLOAT16, DT_BF16, DT_BF16, DT_BF16, DT_BF16, DT_BF16,
                        DT_BF16, DT_BF16, DT_BF16, DT_BF16, DT_BF16, DT_BF16,
                        DT_BF16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_BF16,
                        DT_BF16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_BF16,
                        DT_BF16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_BF16,
                        DT_BF16, DT_BF16, DT_BF16, DT_BF16, DT_BF16, DT_BF16,
                        DT_BF16, DT_BF16, DT_BF16, DT_BF16, DT_BF16, DT_BF16,
                        DT_BF16, DT_BF16, DT_BF16, DT_BF16, DT_BF16, DT_BF16,
                        DT_BF16, DT_BF16, DT_FLOAT8_E4M3FN, DT_FLOAT8_E4M3FN, DT_INT8, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT4, DT_INT4, DT_INT4,
                        DT_INT4, DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT4, DT_INT4, DT_HIFLOAT8, DT_FLOAT8_E4M3FN, DT_FLOAT4_E2M1, DT_HIFLOAT8,
                        DT_FLOAT8_E4M3FN, DT_FLOAT4_E2M1, DT_INT8, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT8, DT_INT4, DT_INT4, DT_INT4, DT_INT4, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT4, DT_INT4,
                        DT_HIFLOAT8, DT_FLOAT8_E4M3FN, DT_FLOAT4_E2M1, DT_HIFLOAT8, DT_FLOAT8_E4M3FN, DT_FLOAT4_E2M1,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT4,
                        DT_INT4, DT_INT4, DT_INT4, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT4, DT_INT4, DT_HIFLOAT8, DT_FLOAT8_E4M3FN,
                        DT_FLOAT4_E2M1, DT_HIFLOAT8, DT_FLOAT8_E4M3FN, DT_FLOAT4_E2M1, DT_INT8, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_BF16, DT_HIFLOAT8,
                        DT_HIFLOAT8, DT_INT8})
            .FormatList({FORMAT_ND})
            .AutoContiguous();
        aicore_config_95.Input("pse_shift")
            .ParamType(OPTIONAL)
            .DataType({DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                        DT_FLOAT16, DT_BF16, DT_BF16, DT_FLOAT16, DT_BF16, DT_FLOAT16,
                        DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_BF16, DT_FLOAT16,
                        DT_BF16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_BF16,
                        DT_BF16, DT_BF16, DT_BF16, DT_FLOAT16, DT_FLOAT16, DT_BF16,
                        DT_BF16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                        DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_BF16,
                        DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_BF16,
                        DT_FLOAT16, DT_BF16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                        DT_BF16, DT_BF16, DT_FLOAT16, DT_BF16, DT_FLOAT16, DT_FLOAT16,
                        DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_BF16, DT_BF16, DT_BF16,
                        DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_BF16, DT_FLOAT16, DT_FLOAT16,
                        DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                        DT_BF16, DT_BF16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_BF16,
                        DT_BF16, DT_BF16, DT_FLOAT16, DT_BF16, DT_FLOAT16, DT_BF16,
                        DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                        DT_BF16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                        DT_BF16, DT_BF16, DT_BF16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                        DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_BF16, DT_BF16, DT_BF16,
                        DT_BF16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_BF16,
                        DT_BF16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                        DT_FLOAT16, DT_BF16, DT_BF16, DT_BF16, DT_BF16, DT_BF16,
                        DT_BF16, DT_BF16, DT_BF16, DT_BF16, DT_BF16, DT_BF16,
                        DT_BF16, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_BF16, DT_BF16, DT_BF16, DT_BF16, DT_BF16,
                        DT_BF16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                        DT_FLOAT16, DT_BF16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                        DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_BF16,
                        DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_BF16, DT_FLOAT16, DT_FLOAT16,
                        DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                        DT_BF16, DT_BF16, DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_BF16,
                        DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_BF16, DT_FLOAT16, DT_FLOAT16,
                        DT_BF16, DT_BF16, DT_BF16, DT_FLOAT16, DT_BF16, DT_FLOAT16,
                        DT_BF16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                        DT_FLOAT16, DT_BF16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_BF16,
                        DT_BF16, DT_BF16, DT_FLOAT16, DT_BF16, DT_BF16, DT_BF16,
                        DT_BF16, DT_BF16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                        DT_BF16, DT_BF16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                        DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_BF16, DT_FLOAT16, DT_FLOAT16,
                        DT_BF16, DT_BF16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                        DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_BF16,
                        DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_BF16, DT_FLOAT16, DT_FLOAT16,
                        DT_BF16, DT_BF16, DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_BF16,
                        DT_BF16, DT_FLOAT16, DT_BF16, DT_FLOAT16, DT_BF16, DT_FLOAT16,
                        DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_BF16,
                        DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_BF16, DT_BF16,
                        DT_FLOAT16, DT_BF16, DT_BF16, DT_BF16, DT_BF16, DT_BF16,
                        DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_BF16,
                        DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                        DT_BF16, DT_BF16, DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_BF16,
                        DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                        DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_BF16, DT_FLOAT16, DT_FLOAT16,
                        DT_BF16, DT_BF16, DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_BF16,
                        DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_BF16, DT_BF16, DT_FLOAT16,
                        DT_BF16, DT_FLOAT16, DT_BF16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                        DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_FLOAT16, DT_FLOAT16,
                        DT_FLOAT16, DT_BF16, DT_BF16, DT_BF16, DT_FLOAT16, DT_BF16,
                        DT_BF16, DT_BF16, DT_BF16, DT_BF16, DT_FLOAT16, DT_FLOAT16,
                        DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_BF16, DT_BF16, DT_FLOAT16,
                        DT_FLOAT16, DT_BF16})
            .FormatList({FORMAT_ND})
            .AutoContiguous();
        aicore_config_95.Input("atten_mask")
            .ParamType(OPTIONAL)
            .DataType({DT_FLOAT16, DT_BOOL, DT_BOOL, DT_UINT8, DT_UINT8, DT_BOOL,
                        DT_BOOL, DT_BOOL, DT_BOOL, DT_FLOAT16, DT_BOOL, DT_UINT8,
                        DT_UINT8, DT_BOOL, DT_BOOL, DT_BOOL, DT_BOOL, DT_FLOAT16,
                        DT_BOOL, DT_UINT8, DT_UINT8, DT_BOOL, DT_BOOL, DT_BOOL,
                        DT_BOOL, DT_BOOL, DT_BOOL, DT_BOOL, DT_BOOL, DT_BOOL,
                        DT_BOOL, DT_BOOL, DT_UINT8, DT_BOOL, DT_BOOL, DT_BOOL,
                        DT_BOOL, DT_INT8, DT_BOOL, DT_FLOAT16, DT_BOOL, DT_BOOL,
                        DT_UINT8, DT_UINT8, DT_BOOL, DT_BOOL, DT_BOOL, DT_BOOL,
                        DT_FLOAT16, DT_BOOL, DT_UINT8, DT_UINT8, DT_BOOL, DT_BOOL,
                        DT_BOOL, DT_BOOL, DT_FLOAT16, DT_BOOL, DT_UINT8, DT_UINT8,
                        DT_BOOL, DT_BOOL, DT_BOOL, DT_BOOL, DT_BOOL, DT_BOOL,
                        DT_BOOL, DT_BOOL, DT_BOOL, DT_BOOL, DT_BOOL, DT_UINT8,
                        DT_BOOL, DT_BOOL, DT_BOOL, DT_BOOL, DT_INT8, DT_BOOL,
                        DT_BOOL, DT_BOOL, DT_BOOL, DT_BOOL, DT_INT8, DT_INT8,
                        DT_BOOL, DT_UINT8, DT_BOOL, DT_BOOL, DT_BOOL, DT_BOOL,
                        DT_BOOL, DT_INT8, DT_UINT8, DT_BOOL, DT_BOOL, DT_BOOL,
                        DT_BOOL, DT_INT8, DT_UINT8, DT_BOOL, DT_BOOL, DT_BOOL,
                        DT_BOOL, DT_BOOL, DT_BOOL, DT_INT8, DT_INT8, DT_UINT8,
                        DT_UINT8, DT_UINT8, DT_BOOL, DT_INT8, DT_INT8, DT_UINT8,
                        DT_UINT8, DT_BOOL, DT_INT8, DT_INT8, DT_UINT8, DT_INT8,
                        DT_UINT8, DT_BOOL, DT_BOOL, DT_UINT8, DT_UINT8, DT_INT8,
                        DT_INT8, DT_BOOL, DT_BOOL, DT_UINT8, DT_UINT8, DT_INT8,
                        DT_INT8, DT_BOOL, DT_BOOL, DT_UINT8, DT_UINT8, DT_INT8,
                        DT_INT8, DT_BOOL, DT_INT8, DT_UINT8, DT_BOOL, DT_INT8,
                        DT_UINT8, DT_BOOL, DT_INT8, DT_UINT8, DT_BOOL, DT_INT8,
                        DT_UINT8, DT_BOOL, DT_INT8, DT_UINT8, DT_BOOL, DT_INT8,
                        DT_UINT8, DT_BOOL, DT_INT8, DT_UINT8, DT_BOOL, DT_INT8,
                        DT_UINT8, DT_BOOL, DT_BOOL, DT_INT8, DT_INT8, DT_UINT8,
                        DT_UINT8, DT_BOOL, DT_BOOL, DT_INT8, DT_INT8, DT_UINT8,
                        DT_UINT8, DT_INT8, DT_BOOL, DT_BOOL, DT_BOOL, DT_BOOL,
                        DT_BOOL, DT_BOOL, DT_BOOL, DT_BOOL, DT_BOOL, DT_BOOL,
                        DT_BOOL, DT_BOOL, DT_BOOL, DT_BOOL, DT_BOOL, DT_BOOL,
                        DT_BOOL, DT_BOOL, DT_BOOL, DT_BOOL, DT_BOOL, DT_BOOL,
                        DT_BOOL, DT_BOOL, DT_BOOL, DT_BOOL, DT_BOOL, DT_BOOL,
                        DT_BOOL, DT_BOOL, DT_BOOL, DT_BOOL, DT_BOOL, DT_BOOL,
                        DT_INT8, DT_BOOL, DT_UINT8, DT_BOOL, DT_BOOL, DT_BOOL,
                        DT_BOOL, DT_BOOL, DT_INT8, DT_UINT8, DT_BOOL, DT_BOOL,
                        DT_BOOL, DT_BOOL, DT_BOOL, DT_BOOL, DT_BOOL, DT_BOOL,
                        DT_BOOL, DT_BOOL, DT_UINT8, DT_BOOL, DT_INT8, DT_INT8,
                        DT_UINT8, DT_UINT8, DT_BOOL, DT_INT8, DT_INT8, DT_UINT8,
                        DT_INT8, DT_UINT8, DT_BOOL, DT_BOOL, DT_BOOL, DT_BOOL,
                        DT_BOOL, DT_BOOL, DT_BOOL, DT_BOOL, DT_BOOL, DT_BOOL,
                        DT_BOOL, DT_BOOL, DT_BOOL, DT_BOOL, DT_BOOL, DT_BOOL,
                        DT_BOOL, DT_BOOL, DT_BOOL, DT_BOOL, DT_BOOL, DT_BOOL,
                        DT_BOOL, DT_BOOL, DT_BOOL, DT_BOOL, DT_BOOL, DT_BOOL,
                        DT_BOOL, DT_BOOL, DT_BOOL, DT_BOOL, DT_INT8, DT_BOOL,
                        DT_UINT8, DT_BOOL, DT_BOOL, DT_BOOL, DT_BOOL, DT_BOOL,
                        DT_INT8, DT_UINT8, DT_BOOL, DT_BOOL, DT_BOOL, DT_BOOL,
                        DT_BOOL, DT_BOOL, DT_BOOL, DT_BOOL, DT_BOOL, DT_BOOL,
                        DT_UINT8, DT_BOOL, DT_INT8, DT_INT8, DT_UINT8, DT_UINT8,
                        DT_BOOL, DT_INT8, DT_INT8, DT_UINT8, DT_INT8, DT_UINT8,
                        DT_BOOL, DT_BOOL, DT_BOOL, DT_BOOL, DT_BOOL, DT_BOOL,
                        DT_BOOL, DT_BOOL, DT_BOOL, DT_BOOL, DT_BOOL, DT_BOOL,
                        DT_BOOL, DT_BOOL, DT_BOOL, DT_BOOL, DT_BOOL, DT_BOOL,
                        DT_BOOL, DT_BOOL, DT_BOOL, DT_BOOL, DT_BOOL, DT_BOOL,
                        DT_BOOL, DT_BOOL, DT_BOOL, DT_BOOL, DT_BOOL, DT_BOOL,
                        DT_BOOL, DT_BOOL, DT_INT8, DT_BOOL, DT_UINT8, DT_BOOL,
                        DT_BOOL, DT_BOOL, DT_BOOL, DT_BOOL, DT_INT8, DT_UINT8,
                        DT_BOOL, DT_BOOL, DT_BOOL, DT_BOOL, DT_BOOL, DT_BOOL,
                        DT_BOOL, DT_BOOL, DT_BOOL, DT_BOOL, DT_UINT8, DT_BOOL,
                        DT_INT8, DT_INT8, DT_UINT8, DT_UINT8, DT_BOOL, DT_INT8,
                        DT_INT8, DT_UINT8, DT_INT8, DT_UINT8, DT_UINT8, DT_BOOL,
                        DT_BOOL, DT_BOOL})
            .FormatList({FORMAT_ND})
            .AutoContiguous();
        aicore_config_95.Input("actual_seq_lengths")
            .ParamType(OPTIONAL)
            .ValueDepend(OPTIONAL)
            .DataTypeList({DT_INT64})
            .FormatList({FORMAT_ND})
            .AutoContiguous();
        aicore_config_95.Input("actual_seq_lengths_kv")
            .ParamType(OPTIONAL)
            .ValueDepend(OPTIONAL)
            .DataTypeList({DT_INT64})
            .FormatList({FORMAT_ND})
            .AutoContiguous();
        aicore_config_95.Input("dequant_scale1")
            .ParamType(OPTIONAL)
            .DataType({DT_UINT64, DT_UINT64, DT_UINT64, DT_UINT64, DT_UINT64, DT_UINT64,
                        DT_UINT64, DT_UINT64, DT_UINT64, DT_UINT64, DT_UINT64, DT_UINT64,
                        DT_UINT64, DT_UINT64, DT_UINT64, DT_UINT64, DT_UINT64, DT_UINT64,
                        DT_UINT64, DT_UINT64, DT_UINT64, DT_UINT64, DT_UINT64, DT_UINT64,
                        DT_UINT64, DT_UINT64, DT_UINT64, DT_UINT64, DT_UINT64, DT_UINT64,
                        DT_UINT64, DT_UINT64, DT_UINT64, DT_UINT64, DT_UINT64, DT_UINT64,
                        DT_UINT64, DT_UINT64, DT_UINT64, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,	// dequant scale1 datatype
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_UINT64,
                        DT_UINT64, DT_UINT64, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_UINT64, DT_UINT64, DT_UINT64, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_UINT64, DT_FLOAT,
                        DT_UINT64, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_UINT64, DT_UINT64,
                        DT_UINT64, DT_UINT64, DT_UINT64, DT_UINT64, DT_UINT64, DT_UINT64,
                        DT_UINT64, DT_UINT64, DT_UINT64, DT_UINT64, DT_UINT64, DT_UINT64,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_UINT64, DT_UINT64, DT_UINT64, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_UINT64, DT_UINT64, DT_UINT64, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_UINT64, DT_UINT64, DT_UINT64, DT_UINT64,
                        DT_UINT64, DT_UINT64, DT_UINT64, DT_UINT64, DT_UINT64, DT_UINT64,
                        DT_UINT64, DT_UINT64, DT_UINT64, DT_UINT64, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_UINT64, DT_UINT64,
                        DT_UINT64, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_UINT64,
                        DT_UINT64, DT_UINT64, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_UINT64, DT_UINT64, DT_UINT64, DT_UINT64, DT_UINT64, DT_UINT64,
                        DT_UINT64, DT_UINT64, DT_UINT64, DT_UINT64, DT_UINT64, DT_UINT64,
                        DT_UINT64, DT_UINT64, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_UINT64, DT_UINT64, DT_UINT64, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_UINT64, DT_UINT64, DT_UINT64,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_UINT64, DT_FLOAT,
                        DT_FLOAT, DT_UINT64})
            .FormatList({FORMAT_ND})
            .AutoContiguous();
        aicore_config_95.Input("quant_scale1")
            .ParamType(OPTIONAL)
            .DataTypeList({DT_FLOAT})
            .FormatList({FORMAT_ND})
            .AutoContiguous();
        aicore_config_95.Input("dequant_scale2")
            .ParamType(OPTIONAL)
            .DataType({DT_UINT64, DT_UINT64, DT_UINT64, DT_UINT64, DT_UINT64, DT_UINT64,
                        DT_UINT64, DT_UINT64, DT_UINT64, DT_UINT64, DT_UINT64, DT_UINT64,
                        DT_UINT64, DT_UINT64, DT_UINT64, DT_UINT64, DT_UINT64, DT_UINT64,
                        DT_UINT64, DT_UINT64, DT_UINT64, DT_UINT64, DT_UINT64, DT_UINT64,
                        DT_UINT64, DT_UINT64, DT_UINT64, DT_UINT64, DT_UINT64, DT_UINT64,
                        DT_UINT64, DT_UINT64, DT_UINT64, DT_UINT64, DT_UINT64, DT_UINT64,
                        DT_UINT64, DT_UINT64, DT_UINT64, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,	// dequant scale2 datatype
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_UINT64,
                        DT_UINT64, DT_UINT64, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_UINT64, DT_UINT64, DT_UINT64, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_UINT64, DT_FLOAT,
                        DT_UINT64, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_UINT64, DT_UINT64,
                        DT_UINT64, DT_UINT64, DT_UINT64, DT_UINT64, DT_UINT64, DT_UINT64,
                        DT_UINT64, DT_UINT64, DT_UINT64, DT_UINT64, DT_UINT64, DT_UINT64,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_UINT64, DT_UINT64, DT_UINT64, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_UINT64, DT_UINT64, DT_UINT64, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_UINT64, DT_UINT64, DT_UINT64, DT_UINT64,
                        DT_UINT64, DT_UINT64, DT_UINT64, DT_UINT64, DT_UINT64, DT_UINT64,
                        DT_UINT64, DT_UINT64, DT_UINT64, DT_UINT64, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_UINT64, DT_UINT64,
                        DT_UINT64, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_UINT64,
                        DT_UINT64, DT_UINT64, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_UINT64, DT_UINT64, DT_UINT64, DT_UINT64, DT_UINT64, DT_UINT64,
                        DT_UINT64, DT_UINT64, DT_UINT64, DT_UINT64, DT_UINT64, DT_UINT64,
                        DT_UINT64, DT_UINT64, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_UINT64, DT_UINT64, DT_UINT64, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_UINT64, DT_UINT64, DT_UINT64,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_UINT64, DT_FLOAT,
                        DT_FLOAT, DT_UINT64})
            .FormatList({FORMAT_ND})
            .AutoContiguous();
        aicore_config_95.Input("quant_scale2")
            .ParamType(OPTIONAL)
            .DataType({DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_BF16, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_BF16, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_BF16, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_BF16,
                        DT_BF16, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_BF16, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_BF16,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,	// quant scale2 datatype
                        DT_FLOAT, DT_BF16, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_BF16, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_BF16, DT_BF16, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_BF16,
                        DT_BF16, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_BF16, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT16, DT_FLOAT16, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_BF16, DT_BF16, DT_BF16, DT_BF16, DT_BF16,
                        DT_BF16, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_BF16, DT_BF16, DT_BF16, DT_BF16, DT_BF16,
                        DT_BF16, DT_FLOAT, DT_BF16, DT_FLOAT, DT_BF16, DT_FLOAT,
                        DT_BF16, DT_FLOAT, DT_BF16, DT_FLOAT, DT_BF16, DT_FLOAT,
                        DT_BF16, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_BF16, DT_BF16, DT_FLOAT, DT_BF16,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_BF16, DT_BF16,
                        DT_FLOAT, DT_BF16, DT_BF16, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_BF16, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT16, DT_FLOAT16,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_BF16, DT_BF16, DT_FLOAT, DT_BF16, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_BF16, DT_BF16, DT_FLOAT, DT_BF16,
                        DT_BF16, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_BF16, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT16, DT_FLOAT16, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_BF16, DT_BF16,
                        DT_FLOAT, DT_BF16, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_BF16, DT_BF16, DT_FLOAT, DT_BF16, DT_BF16, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_BF16, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT16, DT_FLOAT16, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_BF16})
            .FormatList({FORMAT_ND})
            .AutoContiguous();
        aicore_config_95.Input("quant_offset2")
            .ParamType(OPTIONAL)
            .DataType({DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_BF16, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_BF16, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_BF16, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_BF16,
                        DT_BF16, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_BF16, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_BF16,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,	// quant offset2 datatype
                        DT_FLOAT, DT_BF16, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_BF16, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_BF16, DT_BF16, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_BF16,
                        DT_BF16, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_BF16, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT16, DT_FLOAT16, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_BF16, DT_BF16, DT_BF16, DT_BF16, DT_BF16,
                        DT_BF16, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_BF16, DT_BF16, DT_BF16, DT_BF16, DT_BF16,
                        DT_BF16, DT_FLOAT, DT_BF16, DT_FLOAT, DT_BF16, DT_FLOAT,
                        DT_BF16, DT_FLOAT, DT_BF16, DT_FLOAT, DT_BF16, DT_FLOAT,
                        DT_BF16, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_BF16, DT_BF16, DT_FLOAT, DT_BF16,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_BF16, DT_BF16,
                        DT_FLOAT, DT_BF16, DT_BF16, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_BF16, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT16, DT_FLOAT16,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_BF16, DT_BF16, DT_FLOAT, DT_BF16, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_BF16, DT_BF16, DT_FLOAT, DT_BF16,
                        DT_BF16, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_BF16, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT16, DT_FLOAT16, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_BF16, DT_BF16,
                        DT_FLOAT, DT_BF16, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_BF16, DT_BF16, DT_FLOAT, DT_BF16, DT_BF16, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_BF16, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT16, DT_FLOAT16, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT})
            .FormatList({FORMAT_ND})
            .AutoContiguous();
        aicore_config_95.Input("antiquant_scale")
            .ParamType(OPTIONAL)
            .DataType({DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                        DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_FLOAT16,
                        DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                        DT_BF16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                        DT_FLOAT16, DT_BF16, DT_BF16, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_BF16, DT_FLOAT16, DT_FLOAT16, DT_BF16,
                        DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,	// antiquant scale datatype
                        DT_FLOAT16, DT_BF16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                        DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_FLOAT16, DT_FLOAT16,
                        DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_BF16,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_BF16,
                        DT_FLOAT, DT_BF16, DT_FLOAT, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                        DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                        DT_BF16, DT_FLOAT, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT8_E8M0,
                        DT_BF16, DT_BF16, DT_FLOAT8_E8M0, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                        DT_FLOAT16, DT_FLOAT16, DT_FLOAT, DT_BF16, DT_FLOAT, DT_BF16,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT16, DT_FLOAT, DT_FLOAT16, DT_BF16,
                        DT_BF16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                        DT_FLOAT16, DT_BF16, DT_BF16, DT_BF16, DT_BF16, DT_BF16,
                        DT_BF16, DT_BF16, DT_BF16, DT_BF16, DT_BF16, DT_BF16,
                        DT_BF16, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT16, DT_FLOAT16,
                        DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_BF16,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_BF16,
                        DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                        DT_BF16, DT_BF16, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_BF16, DT_FLOAT, DT_BF16, DT_FLOAT, DT_FLOAT16,
                        DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                        DT_FLOAT16, DT_BF16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT8_E8M0, DT_BF16,
                        DT_BF16, DT_FLOAT8_E8M0, DT_FLOAT16, DT_FLOAT, DT_BF16, DT_FLOAT,
                        DT_BF16, DT_FLOAT, DT_FLOAT, DT_FLOAT16, DT_FLOAT, DT_FLOAT16,
                        DT_BF16, DT_BF16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                        DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_BF16, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_BF16, DT_FLOAT16, DT_FLOAT16,
                        DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_BF16,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_BF16,
                        DT_FLOAT, DT_BF16, DT_FLOAT, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                        DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_FLOAT, DT_FLOAT, DT_FLOAT16,
                        DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_BF16,
                        DT_FLOAT16, DT_FLOAT16, DT_FLOAT8_E8M0, DT_BF16, DT_BF16, DT_FLOAT8_E8M0,
                        DT_FLOAT16, DT_FLOAT, DT_BF16, DT_FLOAT, DT_BF16, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT16, DT_FLOAT, DT_FLOAT16, DT_BF16, DT_BF16,
                        DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                        DT_BF16, DT_BF16, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_BF16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                        DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_BF16, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_BF16, DT_FLOAT, DT_BF16,
                        DT_FLOAT, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                        DT_BF16, DT_FLOAT, DT_FLOAT, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                        DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_FLOAT16, DT_FLOAT16,
                        DT_FLOAT8_E8M0, DT_BF16, DT_BF16, DT_FLOAT8_E8M0, DT_FLOAT16, DT_FLOAT,
                        DT_BF16, DT_FLOAT, DT_BF16, DT_FLOAT, DT_FLOAT, DT_FLOAT16,
                        DT_FLOAT, DT_FLOAT16, DT_BF16, DT_BF16, DT_BF16, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT16})
            .FormatList({FORMAT_ND})
            .AutoContiguous();
        aicore_config_95.Input("antiquant_offset")
            .ParamType(OPTIONAL)
            .DataType({DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                        DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_FLOAT16,
                        DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                        DT_BF16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                        DT_FLOAT16, DT_BF16, DT_BF16, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_BF16, DT_FLOAT16, DT_FLOAT16, DT_BF16,
                        DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,	// antiquant offset datatype
                        DT_FLOAT16, DT_BF16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                        DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_FLOAT16, DT_FLOAT16,
                        DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_BF16,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_BF16,
                        DT_FLOAT, DT_BF16, DT_FLOAT, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                        DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                        DT_BF16, DT_FLOAT, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT8_E8M0,
                        DT_BF16, DT_BF16, DT_FLOAT8_E8M0, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                        DT_FLOAT16, DT_FLOAT16, DT_FLOAT, DT_BF16, DT_FLOAT, DT_BF16,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT16, DT_FLOAT, DT_FLOAT16, DT_BF16,
                        DT_BF16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                        DT_FLOAT16, DT_BF16, DT_BF16, DT_BF16, DT_BF16, DT_BF16,
                        DT_BF16, DT_BF16, DT_BF16, DT_BF16, DT_BF16, DT_BF16,
                        DT_BF16, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT16, DT_FLOAT16,
                        DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_BF16,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_BF16,
                        DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                        DT_BF16, DT_BF16, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_BF16, DT_FLOAT, DT_BF16, DT_FLOAT, DT_FLOAT16,
                        DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                        DT_FLOAT16, DT_BF16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT8_E8M0, DT_BF16,
                        DT_BF16, DT_FLOAT8_E8M0, DT_FLOAT16, DT_FLOAT, DT_BF16, DT_FLOAT,
                        DT_BF16, DT_FLOAT, DT_FLOAT, DT_FLOAT16, DT_FLOAT, DT_FLOAT16,
                        DT_BF16, DT_BF16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                        DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_BF16, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_BF16, DT_FLOAT16, DT_FLOAT16,
                        DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_BF16,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_BF16,
                        DT_FLOAT, DT_BF16, DT_FLOAT, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                        DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_FLOAT, DT_FLOAT, DT_FLOAT16,
                        DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_BF16,
                        DT_FLOAT16, DT_FLOAT16, DT_FLOAT8_E8M0, DT_BF16, DT_BF16, DT_FLOAT8_E8M0,
                        DT_FLOAT16, DT_FLOAT, DT_BF16, DT_FLOAT, DT_BF16, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT16, DT_FLOAT, DT_FLOAT16, DT_BF16, DT_BF16,
                        DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                        DT_BF16, DT_BF16, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_BF16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                        DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_BF16, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_BF16, DT_FLOAT, DT_BF16,
                        DT_FLOAT, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                        DT_BF16, DT_FLOAT, DT_FLOAT, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                        DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_FLOAT16, DT_FLOAT16,
                        DT_FLOAT8_E8M0, DT_BF16, DT_BF16, DT_FLOAT8_E8M0, DT_FLOAT16, DT_FLOAT,
                        DT_BF16, DT_FLOAT, DT_BF16, DT_FLOAT, DT_FLOAT, DT_FLOAT16,
                        DT_FLOAT, DT_FLOAT16, DT_BF16, DT_BF16, DT_BF16, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT16})
            .FormatList({FORMAT_ND})
            .AutoContiguous();
        aicore_config_95.Input("block_table")
            .ParamType(OPTIONAL)
            .DataTypeList({DT_INT32})
            .FormatList({FORMAT_ND})
            .AutoContiguous();
        aicore_config_95.Input("query_padding_size")
            .ParamType(OPTIONAL)
            .DataTypeList({DT_INT64})
            .FormatList({FORMAT_ND})
            .AutoContiguous();
        aicore_config_95.Input("kv_padding_size")
            .ParamType(OPTIONAL)
            .DataTypeList({DT_INT64})
            .FormatList({FORMAT_ND})
            .AutoContiguous();
        aicore_config_95.Input("key_antiquant_scale")
            .ParamType(OPTIONAL)
            .DataType({DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                        DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_FLOAT16,
                        DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                        DT_BF16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                        DT_FLOAT16, DT_BF16, DT_BF16, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_BF16, DT_FLOAT16, DT_FLOAT16, DT_BF16,
                        DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,	// key antiquant scale datatype
                        DT_FLOAT16, DT_BF16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                        DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_FLOAT16, DT_FLOAT16,
                        DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_BF16,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_BF16,
                        DT_BF16, DT_FLOAT, DT_FLOAT16, DT_FLOAT, DT_FLOAT16, DT_FLOAT,
                        DT_FLOAT, DT_BF16, DT_FLOAT16, DT_BF16, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT, DT_FLOAT, DT_FLOAT16,
                        DT_BF16, DT_FLOAT, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT8_E8M0,
                        DT_BF16, DT_BF16, DT_FLOAT8_E8M0, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                        DT_FLOAT16, DT_FLOAT, DT_BF16, DT_BF16, DT_BF16, DT_FLOAT,
                        DT_BF16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT, DT_FLOAT16, DT_FLOAT,
                        DT_BF16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                        DT_FLOAT16, DT_BF16, DT_BF16, DT_BF16, DT_BF16, DT_BF16,
                        DT_BF16, DT_BF16, DT_BF16, DT_BF16, DT_BF16, DT_BF16,
                        DT_BF16, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT16, DT_FLOAT16,
                        DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_BF16,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_BF16,
                        DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                        DT_BF16, DT_BF16, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_BF16, DT_BF16, DT_FLOAT, DT_FLOAT16, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_BF16, DT_FLOAT16, DT_BF16, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT16, DT_BF16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT8_E8M0, DT_BF16,
                        DT_BF16, DT_FLOAT8_E8M0, DT_FLOAT, DT_BF16, DT_BF16, DT_BF16,
                        DT_FLOAT, DT_BF16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT, DT_FLOAT16,
                        DT_FLOAT, DT_BF16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                        DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_BF16, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_BF16, DT_FLOAT16, DT_FLOAT16,
                        DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_BF16,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_BF16,
                        DT_BF16, DT_FLOAT, DT_FLOAT16, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_BF16, DT_FLOAT16, DT_BF16, DT_FLOAT, DT_FLOAT, DT_FLOAT16,
                        DT_FLOAT16, DT_FLOAT16, DT_FLOAT, DT_FLOAT, DT_FLOAT16, DT_BF16,
                        DT_FLOAT16, DT_FLOAT16, DT_FLOAT8_E8M0, DT_BF16, DT_BF16, DT_FLOAT8_E8M0,
                        DT_FLOAT, DT_BF16, DT_BF16, DT_BF16, DT_FLOAT, DT_BF16,
                        DT_FLOAT16, DT_FLOAT16, DT_FLOAT, DT_FLOAT16, DT_FLOAT, DT_BF16,
                        DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                        DT_BF16, DT_BF16, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_BF16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                        DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_BF16, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_BF16, DT_BF16, DT_FLOAT,
                        DT_FLOAT16, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_BF16, DT_FLOAT16,
                        DT_BF16, DT_FLOAT, DT_FLOAT, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT16, DT_BF16, DT_FLOAT16, DT_FLOAT16,
                        DT_FLOAT8_E8M0, DT_BF16, DT_BF16, DT_FLOAT8_E8M0, DT_FLOAT, DT_BF16,
                        DT_BF16, DT_BF16, DT_FLOAT, DT_BF16, DT_FLOAT16, DT_FLOAT16,
                        DT_FLOAT, DT_FLOAT16, DT_FLOAT, DT_BF16, DT_BF16, DT_FLOAT,
                        DT_FLOAT, DT_BF16})
            .FormatList({FORMAT_ND})
            .AutoContiguous();
        aicore_config_95.Input("key_antiquant_offset")
            .ParamType(OPTIONAL)
            .DataType({DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                        DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_FLOAT16,
                        DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                        DT_BF16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                        DT_FLOAT16, DT_BF16, DT_BF16, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_BF16, DT_FLOAT16, DT_FLOAT16, DT_BF16,
                        DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,	// key antiquant offset datatype
                        DT_FLOAT16, DT_BF16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                        DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_FLOAT16, DT_FLOAT16,
                        DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_BF16,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_BF16,
                        DT_BF16, DT_FLOAT, DT_FLOAT16, DT_FLOAT, DT_FLOAT16, DT_FLOAT,
                        DT_FLOAT, DT_BF16, DT_FLOAT16, DT_BF16, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT, DT_FLOAT, DT_FLOAT16,
                        DT_BF16, DT_FLOAT, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT8_E8M0,
                        DT_BF16, DT_BF16, DT_FLOAT8_E8M0, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                        DT_FLOAT16, DT_FLOAT, DT_BF16, DT_BF16, DT_BF16, DT_FLOAT,
                        DT_BF16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT, DT_FLOAT16, DT_FLOAT,
                        DT_BF16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                        DT_FLOAT16, DT_BF16, DT_BF16, DT_BF16, DT_BF16, DT_BF16,
                        DT_BF16, DT_BF16, DT_BF16, DT_BF16, DT_BF16, DT_BF16,
                        DT_BF16, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT16, DT_FLOAT16,
                        DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_BF16,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_BF16,
                        DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                        DT_BF16, DT_BF16, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_BF16, DT_BF16, DT_FLOAT, DT_FLOAT16, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_BF16, DT_FLOAT16, DT_BF16, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT16, DT_BF16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT8_E8M0, DT_BF16,
                        DT_BF16, DT_FLOAT8_E8M0, DT_FLOAT, DT_BF16, DT_BF16, DT_BF16,
                        DT_FLOAT, DT_BF16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT, DT_FLOAT16,
                        DT_FLOAT, DT_BF16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                        DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_BF16, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_BF16, DT_FLOAT16, DT_FLOAT16,
                        DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_BF16,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_BF16,
                        DT_BF16, DT_FLOAT, DT_FLOAT16, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_BF16, DT_FLOAT16, DT_BF16, DT_FLOAT, DT_FLOAT, DT_FLOAT16,
                        DT_FLOAT16, DT_FLOAT16, DT_FLOAT, DT_FLOAT, DT_FLOAT16, DT_BF16,
                        DT_FLOAT16, DT_FLOAT16, DT_FLOAT8_E8M0, DT_BF16, DT_BF16, DT_FLOAT8_E8M0,
                        DT_FLOAT, DT_BF16, DT_BF16, DT_BF16, DT_FLOAT, DT_BF16,
                        DT_FLOAT16, DT_FLOAT16, DT_FLOAT, DT_FLOAT16, DT_FLOAT, DT_BF16,
                        DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                        DT_BF16, DT_BF16, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_BF16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                        DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_BF16, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_BF16, DT_BF16, DT_FLOAT,
                        DT_FLOAT16, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_BF16, DT_FLOAT16,
                        DT_BF16, DT_FLOAT, DT_FLOAT, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT16, DT_BF16, DT_FLOAT16, DT_FLOAT16,
                        DT_FLOAT8_E8M0, DT_BF16, DT_BF16, DT_FLOAT8_E8M0, DT_FLOAT, DT_BF16,
                        DT_BF16, DT_BF16, DT_FLOAT, DT_BF16, DT_FLOAT16, DT_FLOAT16,
                        DT_FLOAT, DT_FLOAT16, DT_FLOAT, DT_BF16, DT_BF16, DT_FLOAT,
                        DT_FLOAT, DT_BF16})
            .FormatList({FORMAT_ND})
            .AutoContiguous();
        aicore_config_95.Input("value_antiquant_scale")
            .ParamType(OPTIONAL)
            .DataType({DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                        DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_FLOAT16,
                        DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                        DT_BF16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                        DT_FLOAT16, DT_BF16, DT_BF16, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_BF16, DT_FLOAT16, DT_FLOAT16, DT_BF16,
                        DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,	// value antiquant scale datatype
                        DT_FLOAT16, DT_BF16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                        DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_FLOAT16, DT_FLOAT16,
                        DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_BF16,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_BF16,
                        DT_BF16, DT_FLOAT, DT_FLOAT16, DT_FLOAT, DT_FLOAT16, DT_FLOAT,
                        DT_FLOAT, DT_BF16, DT_FLOAT16, DT_BF16, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT8_E8M0,
                        DT_BF16, DT_BF16, DT_FLOAT8_E8M0, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                        DT_FLOAT16, DT_FLOAT, DT_FLOAT, DT_BF16, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT16, DT_FLOAT, DT_FLOAT16, DT_FLOAT,
                        DT_BF16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                        DT_FLOAT16, DT_BF16, DT_BF16, DT_BF16, DT_BF16, DT_BF16,
                        DT_BF16, DT_BF16, DT_BF16, DT_BF16, DT_BF16, DT_BF16,
                        DT_BF16, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT16, DT_FLOAT16,
                        DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_BF16,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_BF16,
                        DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                        DT_BF16, DT_BF16, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_BF16, DT_BF16, DT_FLOAT, DT_FLOAT16, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_BF16, DT_FLOAT16, DT_BF16, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT16, DT_BF16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT8_E8M0, DT_BF16,
                        DT_BF16, DT_FLOAT8_E8M0, DT_FLOAT, DT_BF16, DT_BF16, DT_BF16,
                        DT_FLOAT, DT_BF16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT, DT_FLOAT16,
                        DT_FLOAT, DT_BF16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                        DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_BF16, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_BF16, DT_FLOAT16, DT_FLOAT16,
                        DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_BF16,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_BF16,
                        DT_BF16, DT_FLOAT, DT_FLOAT16, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_BF16, DT_FLOAT16, DT_BF16, DT_FLOAT, DT_FLOAT, DT_FLOAT16,
                        DT_FLOAT16, DT_FLOAT16, DT_FLOAT, DT_FLOAT, DT_FLOAT16, DT_BF16,
                        DT_FLOAT16, DT_FLOAT16, DT_FLOAT8_E8M0, DT_BF16, DT_BF16, DT_FLOAT8_E8M0,
                        DT_FLOAT, DT_BF16, DT_BF16, DT_BF16, DT_FLOAT, DT_BF16,
                        DT_FLOAT16, DT_FLOAT16, DT_FLOAT, DT_FLOAT16, DT_FLOAT, DT_BF16,
                        DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                        DT_BF16, DT_BF16, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_BF16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                        DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_BF16, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_BF16, DT_BF16, DT_FLOAT,
                        DT_FLOAT16, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_BF16, DT_FLOAT16,
                        DT_BF16, DT_FLOAT, DT_FLOAT, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT16, DT_BF16, DT_FLOAT16, DT_FLOAT16,
                        DT_FLOAT8_E8M0, DT_BF16, DT_BF16, DT_FLOAT8_E8M0, DT_FLOAT, DT_BF16,
                        DT_BF16, DT_BF16, DT_FLOAT, DT_BF16, DT_FLOAT16, DT_FLOAT16,
                        DT_FLOAT, DT_FLOAT16, DT_FLOAT, DT_BF16, DT_BF16, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT})
            .FormatList({FORMAT_ND})
            .AutoContiguous();
        aicore_config_95.Input("value_antiquant_offset")
            .ParamType(OPTIONAL)
            .DataType({DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                        DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_FLOAT16,
                        DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                        DT_BF16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                        DT_FLOAT16, DT_BF16, DT_BF16, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_BF16, DT_FLOAT16, DT_FLOAT16, DT_BF16,
                        DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,	// value antiquant offset datatype
                        DT_FLOAT16, DT_BF16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                        DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_FLOAT16, DT_FLOAT16,
                        DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_BF16,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_BF16,
                        DT_BF16, DT_FLOAT, DT_FLOAT16, DT_FLOAT, DT_FLOAT16, DT_FLOAT,
                        DT_FLOAT, DT_BF16, DT_FLOAT16, DT_BF16, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT8_E8M0,
                        DT_BF16, DT_BF16, DT_FLOAT8_E8M0, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                        DT_FLOAT16, DT_FLOAT, DT_FLOAT, DT_BF16, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT16, DT_FLOAT, DT_FLOAT16, DT_FLOAT,
                        DT_BF16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                        DT_FLOAT16, DT_BF16, DT_BF16, DT_BF16, DT_BF16, DT_BF16,
                        DT_BF16, DT_BF16, DT_BF16, DT_BF16, DT_BF16, DT_BF16,
                        DT_BF16, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT16, DT_FLOAT16,
                        DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_BF16,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_BF16,
                        DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                        DT_BF16, DT_BF16, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_BF16, DT_BF16, DT_FLOAT, DT_FLOAT16, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_BF16, DT_FLOAT16, DT_BF16, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT16, DT_FLOAT16, DT_FLOAT8_E8M0, DT_BF16,
                        DT_BF16, DT_FLOAT8_E8M0, DT_FLOAT, DT_FLOAT, DT_BF16, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT16, DT_FLOAT, DT_FLOAT16,
                        DT_FLOAT, DT_BF16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                        DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_BF16, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_BF16, DT_FLOAT16, DT_FLOAT16,
                        DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_BF16,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_BF16,
                        DT_BF16, DT_FLOAT, DT_FLOAT16, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_BF16, DT_FLOAT16, DT_BF16, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT16, DT_FLOAT16, DT_FLOAT8_E8M0, DT_BF16, DT_BF16, DT_FLOAT8_E8M0,
                        DT_FLOAT, DT_FLOAT, DT_BF16, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT16, DT_FLOAT, DT_FLOAT16, DT_FLOAT, DT_BF16,
                        DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                        DT_BF16, DT_BF16, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_BF16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                        DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_BF16, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_BF16, DT_BF16, DT_FLOAT,
                        DT_FLOAT16, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_BF16, DT_FLOAT16,
                        DT_BF16, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT16, DT_FLOAT16,
                        DT_FLOAT8_E8M0, DT_BF16, DT_BF16, DT_FLOAT8_E8M0, DT_FLOAT, DT_FLOAT,
                        DT_BF16, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT16,
                        DT_FLOAT, DT_FLOAT16, DT_FLOAT, DT_BF16, DT_BF16, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT})
            .FormatList({FORMAT_ND})
            .AutoContiguous();
        aicore_config_95.Input("key_shared_prefix")
            .ParamType(OPTIONAL)
            .DataType({DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT8, DT_BF16, DT_BF16, DT_FLOAT16, DT_BF16, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT8, DT_BF16, DT_BF16, DT_FLOAT16,
                        DT_BF16, DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_BF16,
                        DT_BF16, DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT8, DT_BF16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                        DT_INT8, DT_FLOAT16, DT_INT8, DT_FLOAT16, DT_FLOAT16, DT_BF16,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_BF16, DT_BF16,	// key shared prefix datatype
                        DT_FLOAT16, DT_BF16, DT_INT8, DT_INT8, DT_INT8, DT_INT8,
                        DT_BF16, DT_BF16, DT_FLOAT16, DT_BF16, DT_INT8, DT_INT8,
                        DT_INT8, DT_INT8, DT_BF16, DT_BF16, DT_INT8, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_BF16, DT_FLOAT16,
                        DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_INT8, DT_FLOAT16, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_FLOAT16,
                        DT_FLOAT, DT_FLOAT16, DT_INT4, DT_INT4, DT_INT4, DT_INT4,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT4,
                        DT_INT4, DT_FLOAT16, DT_FLOAT16, DT_HIFLOAT8, DT_FLOAT8_E4M3FN, DT_FLOAT4_E2M1,
                        DT_HIFLOAT8, DT_FLOAT8_E4M3FN, DT_FLOAT4_E2M1, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT8, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                        DT_FLOAT16, DT_BF16, DT_BF16, DT_BF16, DT_BF16, DT_BF16,
                        DT_BF16, DT_BF16, DT_BF16, DT_BF16, DT_BF16, DT_BF16,
                        DT_BF16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_BF16,
                        DT_BF16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_BF16,
                        DT_BF16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_BF16,
                        DT_BF16, DT_BF16, DT_BF16, DT_BF16, DT_BF16, DT_BF16,
                        DT_BF16, DT_BF16, DT_BF16, DT_BF16, DT_BF16, DT_BF16,
                        DT_BF16, DT_BF16, DT_BF16, DT_BF16, DT_BF16, DT_BF16,
                        DT_BF16, DT_BF16, DT_FLOAT8_E4M3FN, DT_FLOAT8_E4M3FN, DT_INT8, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8,
                        DT_FLOAT16, DT_FLOAT, DT_FLOAT16, DT_INT4, DT_INT4, DT_INT4,
                        DT_INT4, DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT4, DT_INT4, DT_HIFLOAT8, DT_FLOAT8_E4M3FN, DT_FLOAT4_E2M1, DT_HIFLOAT8,
                        DT_FLOAT8_E4M3FN, DT_FLOAT4_E2M1, DT_INT8, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_FLOAT16, DT_FLOAT,
                        DT_FLOAT16, DT_INT4, DT_INT4, DT_INT4, DT_INT4, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT4, DT_INT4,
                        DT_HIFLOAT8, DT_FLOAT8_E4M3FN, DT_FLOAT4_E2M1, DT_HIFLOAT8, DT_FLOAT8_E4M3FN, DT_FLOAT4_E2M1,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT8, DT_INT8, DT_FLOAT16, DT_FLOAT, DT_FLOAT16, DT_INT4,
                        DT_INT4, DT_INT4, DT_INT4, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT4, DT_INT4, DT_HIFLOAT8, DT_FLOAT8_E4M3FN,
                        DT_FLOAT4_E2M1, DT_HIFLOAT8, DT_FLOAT8_E4M3FN, DT_FLOAT4_E2M1, DT_INT8, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_BF16, DT_HIFLOAT8,
                        DT_HIFLOAT8, DT_INT8})
            .FormatList({FORMAT_ND})
            .AutoContiguous();
        aicore_config_95.Input("value_shared_prefix")
            .ParamType(OPTIONAL)
            .DataType({DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT8, DT_BF16, DT_BF16, DT_FLOAT16, DT_BF16, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT8, DT_BF16, DT_BF16, DT_FLOAT16,
                        DT_BF16, DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_BF16,
                        DT_BF16, DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT8, DT_BF16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                        DT_INT8, DT_FLOAT16, DT_INT8, DT_FLOAT16, DT_FLOAT16, DT_BF16,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_BF16, DT_BF16,	// value shared prefix datatype
                        DT_FLOAT16, DT_BF16, DT_INT8, DT_INT8, DT_INT8, DT_INT8,
                        DT_BF16, DT_BF16, DT_FLOAT16, DT_BF16, DT_INT8, DT_INT8,
                        DT_INT8, DT_INT8, DT_BF16, DT_BF16, DT_INT8, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_BF16, DT_FLOAT16,
                        DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_INT8, DT_FLOAT16, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_FLOAT16,
                        DT_FLOAT, DT_FLOAT16, DT_INT4, DT_INT4, DT_INT4, DT_INT4,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT4,
                        DT_INT4, DT_FLOAT16, DT_FLOAT16, DT_HIFLOAT8, DT_FLOAT8_E4M3FN, DT_FLOAT4_E2M1,
                        DT_HIFLOAT8, DT_FLOAT8_E4M3FN, DT_FLOAT4_E2M1, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT8, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                        DT_FLOAT16, DT_BF16, DT_BF16, DT_BF16, DT_BF16, DT_BF16,
                        DT_BF16, DT_BF16, DT_BF16, DT_BF16, DT_BF16, DT_BF16,
                        DT_BF16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_BF16,
                        DT_BF16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_BF16,
                        DT_BF16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_BF16,
                        DT_BF16, DT_BF16, DT_BF16, DT_BF16, DT_BF16, DT_BF16,
                        DT_BF16, DT_BF16, DT_BF16, DT_BF16, DT_BF16, DT_BF16,
                        DT_BF16, DT_BF16, DT_BF16, DT_BF16, DT_BF16, DT_BF16,
                        DT_BF16, DT_BF16, DT_FLOAT8_E4M3FN, DT_FLOAT8_E4M3FN, DT_INT8, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8,
                        DT_FLOAT16, DT_FLOAT, DT_FLOAT16, DT_INT4, DT_INT4, DT_INT4,
                        DT_INT4, DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT4, DT_INT4, DT_HIFLOAT8, DT_FLOAT8_E4M3FN, DT_FLOAT4_E2M1, DT_HIFLOAT8,
                        DT_FLOAT8_E4M3FN, DT_FLOAT4_E2M1, DT_INT8, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_FLOAT16, DT_FLOAT,
                        DT_FLOAT16, DT_INT4, DT_INT4, DT_INT4, DT_INT4, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT4, DT_INT4,
                        DT_HIFLOAT8, DT_FLOAT8_E4M3FN, DT_FLOAT4_E2M1, DT_HIFLOAT8, DT_FLOAT8_E4M3FN, DT_FLOAT4_E2M1,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT8, DT_INT8, DT_FLOAT16, DT_FLOAT, DT_FLOAT16, DT_INT4,
                        DT_INT4, DT_INT4, DT_INT4, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT4, DT_INT4, DT_HIFLOAT8, DT_FLOAT8_E4M3FN,
                        DT_FLOAT4_E2M1, DT_HIFLOAT8, DT_FLOAT8_E4M3FN, DT_FLOAT4_E2M1, DT_INT8, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_BF16, DT_HIFLOAT8,
                        DT_HIFLOAT8, DT_INT8})
            .FormatList({FORMAT_ND})
            .AutoContiguous();
        aicore_config_95.Input("actual_shared_prefix_len")
            .ParamType(OPTIONAL)
            .ValueDepend(OPTIONAL)
            .DataTypeList({DT_INT64})
            .FormatList({FORMAT_ND})
            .AutoContiguous();
        aicore_config_95.Input("query_rope")
            .ParamType(OPTIONAL)
            .DataType({DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_INT8, DT_INT8, DT_FLOAT16,
                        DT_FLOAT16, DT_BF16, DT_BF16, DT_FLOAT16, DT_BF16, DT_INT8,
                        DT_INT8, DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_BF16, DT_FLOAT16,
                        DT_BF16, DT_INT8, DT_INT8, DT_FLOAT16, DT_FLOAT16, DT_BF16,
                        DT_BF16, DT_BF16, DT_BF16, DT_FLOAT16, DT_FLOAT16, DT_BF16,
                        DT_BF16, DT_BF16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                        DT_BF16, DT_FLOAT16, DT_BF16, DT_FLOAT16, DT_FLOAT16, DT_BF16,
                        DT_INT8, DT_INT8, DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_BF16,
                        DT_FLOAT16, DT_BF16, DT_INT8, DT_INT8, DT_FLOAT16, DT_FLOAT16,
                        DT_BF16, DT_BF16, DT_FLOAT16, DT_BF16, DT_INT8, DT_INT8,
                        DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_BF16, DT_BF16, DT_BF16,
                        DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_BF16, DT_BF16, DT_FLOAT16,
                        DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_FLOAT16, DT_BF16,
                        DT_BF16, DT_BF16, DT_FLOAT16, DT_FLOAT16, DT_INT8, DT_BF16,
                        DT_BF16, DT_BF16, DT_FLOAT16, DT_BF16, DT_FLOAT16, DT_BF16,
                        DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                        DT_BF16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                        DT_BF16, DT_BF16, DT_BF16, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT8, DT_FLOAT16, DT_BF16, DT_BF16, DT_BF16, DT_BF16,
                        DT_BF16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_BF16,
                        DT_BF16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                        DT_FLOAT16, DT_BF16, DT_BF16, DT_BF16, DT_BF16, DT_BF16,
                        DT_BF16, DT_BF16, DT_BF16, DT_BF16, DT_BF16, DT_BF16,
                        DT_BF16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_BF16,
                        DT_BF16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_BF16,
                        DT_BF16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_BF16,
                        DT_BF16, DT_BF16, DT_BF16, DT_BF16, DT_BF16, DT_BF16,
                        DT_BF16, DT_BF16, DT_BF16, DT_BF16, DT_BF16, DT_BF16,
                        DT_BF16, DT_BF16, DT_BF16, DT_BF16, DT_BF16, DT_BF16,
                        DT_BF16, DT_BF16, DT_BF16, DT_BF16, DT_FLOAT16, DT_FLOAT16,
                        DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_BF16,
                        DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_BF16, DT_BF16, DT_BF16,
                        DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                        DT_BF16, DT_BF16, DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_BF16,
                        DT_BF16, DT_BF16, DT_BF16, DT_BF16, DT_FLOAT16, DT_FLOAT16,
                        DT_BF16, DT_BF16, DT_BF16, DT_FLOAT16, DT_BF16, DT_FLOAT16,
                        DT_BF16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                        DT_FLOAT16, DT_BF16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_BF16,
                        DT_BF16, DT_BF16, DT_FLOAT16, DT_BF16, DT_BF16, DT_BF16,
                        DT_BF16, DT_BF16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                        DT_BF16, DT_BF16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                        DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_BF16, DT_FLOAT16, DT_FLOAT16,
                        DT_BF16, DT_BF16, DT_BF16, DT_BF16, DT_FLOAT16, DT_FLOAT16,
                        DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_BF16,
                        DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_BF16, DT_BF16, DT_BF16,
                        DT_BF16, DT_BF16, DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_BF16,
                        DT_BF16, DT_FLOAT16, DT_BF16, DT_FLOAT16, DT_BF16, DT_FLOAT16,
                        DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_BF16,
                        DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_BF16, DT_BF16,
                        DT_FLOAT16, DT_BF16, DT_BF16, DT_BF16, DT_BF16, DT_BF16,
                        DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_BF16,
                        DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                        DT_BF16, DT_BF16, DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_BF16,
                        DT_BF16, DT_BF16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                        DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_BF16, DT_FLOAT16, DT_FLOAT16,
                        DT_BF16, DT_BF16, DT_BF16, DT_BF16, DT_BF16, DT_BF16,
                        DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_BF16, DT_BF16, DT_FLOAT16,
                        DT_BF16, DT_FLOAT16, DT_BF16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                        DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_FLOAT16, DT_FLOAT16,
                        DT_FLOAT16, DT_BF16, DT_BF16, DT_BF16, DT_FLOAT16, DT_BF16,
                        DT_BF16, DT_BF16, DT_BF16, DT_BF16, DT_FLOAT16, DT_FLOAT16,
                        DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_BF16, DT_BF16, DT_BF16,
                        DT_BF16, DT_BF16})
            .FormatList({FORMAT_ND})
            .AutoContiguous();
        aicore_config_95.Input("key_rope")
            .ParamType(OPTIONAL)
            .DataType({DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT8, DT_BF16, DT_BF16, DT_FLOAT16, DT_BF16, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT8, DT_BF16, DT_BF16, DT_FLOAT16,
                        DT_BF16, DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_BF16,
                        DT_BF16, DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT8, DT_BF16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                        DT_INT8, DT_FLOAT16, DT_INT8, DT_FLOAT16, DT_FLOAT16, DT_BF16,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_BF16, DT_BF16,	// key datatype
                        DT_FLOAT16, DT_BF16, DT_INT8, DT_INT8, DT_INT8, DT_INT8,
                        DT_BF16, DT_BF16, DT_FLOAT16, DT_BF16, DT_INT8, DT_INT8,
                        DT_INT8, DT_INT8, DT_BF16, DT_BF16, DT_INT8, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_BF16, DT_FLOAT16,
                        DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_INT8, DT_FLOAT16, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT4, DT_INT4, DT_INT4, DT_INT4,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT4,
                        DT_INT4, DT_FLOAT16, DT_FLOAT16, DT_HIFLOAT8, DT_FLOAT8_E4M3FN, DT_FLOAT4_E2M1,
                        DT_HIFLOAT8, DT_FLOAT8_E4M3FN, DT_FLOAT4_E2M1, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT8, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                        DT_FLOAT16, DT_BF16, DT_BF16, DT_BF16, DT_BF16, DT_BF16,
                        DT_BF16, DT_BF16, DT_BF16, DT_BF16, DT_BF16, DT_BF16,
                        DT_BF16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_BF16,
                        DT_BF16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_BF16,
                        DT_BF16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_BF16,
                        DT_BF16, DT_BF16, DT_BF16, DT_BF16, DT_BF16, DT_BF16,
                        DT_BF16, DT_BF16, DT_BF16, DT_BF16, DT_BF16, DT_BF16,
                        DT_BF16, DT_BF16, DT_BF16, DT_BF16, DT_BF16, DT_BF16,
                        DT_BF16, DT_BF16, DT_BF16, DT_BF16, DT_INT8, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT4, DT_INT4, DT_INT4,
                        DT_INT4, DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT4, DT_INT4, DT_HIFLOAT8, DT_FLOAT8_E4M3FN, DT_FLOAT4_E2M1, DT_HIFLOAT8,
                        DT_FLOAT8_E4M3FN, DT_FLOAT4_E2M1, DT_INT8, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT8, DT_INT4, DT_INT4, DT_INT4, DT_INT4, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT4, DT_INT4,
                        DT_HIFLOAT8, DT_FLOAT8_E4M3FN, DT_FLOAT4_E2M1, DT_HIFLOAT8, DT_FLOAT8_E4M3FN, DT_FLOAT4_E2M1,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT4,
                        DT_INT4, DT_INT4, DT_INT4, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT4, DT_INT4, DT_HIFLOAT8, DT_FLOAT8_E4M3FN,
                        DT_FLOAT4_E2M1, DT_HIFLOAT8, DT_FLOAT8_E4M3FN, DT_FLOAT4_E2M1, DT_INT8, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_BF16, DT_BF16,
                        DT_BF16, DT_INT8})
            .FormatList({FORMAT_ND})
            .AutoContiguous();
        aicore_config_95.Input("key_rope_antiquant_scale")
            .ParamType(OPTIONAL)
            .DataType({DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                        DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_FLOAT16,
                        DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                        DT_BF16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                        DT_FLOAT16, DT_BF16, DT_BF16, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_BF16, DT_FLOAT16, DT_FLOAT16, DT_BF16,
                        DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,	// key rope antiquant scale datatype
                        DT_FLOAT16, DT_BF16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                        DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_FLOAT16, DT_FLOAT16,
                        DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_BF16,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_BF16,
                        DT_BF16, DT_FLOAT, DT_FLOAT16, DT_FLOAT, DT_FLOAT16, DT_FLOAT,
                        DT_FLOAT, DT_BF16, DT_FLOAT16, DT_BF16, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT, DT_FLOAT, DT_FLOAT16,
                        DT_BF16, DT_FLOAT, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT8_E8M0,
                        DT_BF16, DT_BF16, DT_FLOAT8_E8M0, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                        DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                        DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                        DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                        DT_FLOAT16, DT_BF16, DT_BF16, DT_BF16, DT_BF16, DT_BF16,
                        DT_BF16, DT_BF16, DT_BF16, DT_BF16, DT_BF16, DT_BF16,
                        DT_BF16, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_BF16, DT_BF16, DT_FLOAT16, DT_FLOAT16,
                        DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_BF16,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_BF16,
                        DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                        DT_BF16, DT_BF16, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_BF16, DT_BF16, DT_FLOAT, DT_FLOAT16, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_BF16, DT_FLOAT16, DT_BF16, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT16, DT_BF16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT8_E8M0, DT_BF16,
                        DT_BF16, DT_FLOAT8_E8M0, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                        DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                        DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                        DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_BF16, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_BF16, DT_FLOAT16, DT_FLOAT16,
                        DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_BF16,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_BF16,
                        DT_BF16, DT_FLOAT, DT_FLOAT16, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_BF16, DT_FLOAT16, DT_BF16, DT_FLOAT, DT_FLOAT, DT_FLOAT16,
                        DT_FLOAT16, DT_FLOAT16, DT_FLOAT, DT_FLOAT, DT_FLOAT16, DT_BF16,
                        DT_FLOAT16, DT_FLOAT16, DT_FLOAT8_E8M0, DT_BF16, DT_BF16, DT_FLOAT8_E8M0,
                        DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                        DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                        DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                        DT_BF16, DT_BF16, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_BF16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                        DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_BF16, DT_FLOAT, DT_FLOAT,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_BF16, DT_BF16, DT_FLOAT,
                        DT_FLOAT16, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_BF16, DT_FLOAT16,
                        DT_BF16, DT_FLOAT, DT_FLOAT, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                        DT_FLOAT, DT_FLOAT, DT_FLOAT16, DT_BF16, DT_FLOAT16, DT_FLOAT16,
                        DT_FLOAT8_E8M0, DT_BF16, DT_BF16, DT_FLOAT8_E8M0, DT_FLOAT16, DT_FLOAT16,
                        DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                        DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_BF16,
                        DT_BF16, DT_INT8})
            .FormatList({FORMAT_ND})
            .AutoContiguous();
        aicore_config_95.Input("dequant_scale_query")
            .ParamType(OPTIONAL)
            .DataTypeList({DT_FLOAT})
            .FormatList({FORMAT_ND})
            .AutoContiguous();
        aicore_config_95.Input("learnable_sink")
            .ParamType(OPTIONAL)
            .DataTypeList({DT_BF16})
            .FormatList({FORMAT_ND})
            .AutoContiguous();
        aicore_config_95.Input("q_start_idx")
            .ParamType(OPTIONAL)
            .ValueDepend(OPTIONAL)
            .DataTypeList({DT_INT64})
            .FormatList({FORMAT_ND})
            .AutoContiguous();
        aicore_config_95.Input("kv_start_idx")
            .ParamType(OPTIONAL)
            .ValueDepend(OPTIONAL)
            .DataTypeList({DT_INT64})
            .FormatList({FORMAT_ND})
            .AutoContiguous();
        aicore_config_95.Output("attention_out")
            .ParamType(REQUIRED)
            .DataType({DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_INT8, DT_FLOAT16, DT_FLOAT16,
                        DT_INT8, DT_INT8, DT_INT8, DT_FLOAT16, DT_BF16, DT_INT8,
                        DT_FLOAT16, DT_FLOAT16, DT_INT8, DT_INT8, DT_INT8, DT_FLOAT16,
                        DT_BF16, DT_INT8, DT_FLOAT16, DT_FLOAT16, DT_INT8, DT_INT8,
                        DT_INT8, DT_BF16, DT_INT8, DT_FLOAT16, DT_INT8, DT_BF16,
                        DT_INT8, DT_BF16, DT_FLOAT16, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT8, DT_FLOAT16, DT_FLOAT16, DT_BF16,
                        DT_INT8, DT_FLOAT16, DT_FLOAT16, DT_INT8, DT_INT8, DT_INT8,
                        DT_FLOAT16, DT_BF16, DT_INT8, DT_FLOAT16, DT_FLOAT16, DT_INT8,
                        DT_INT8, DT_INT8, DT_FLOAT16, DT_BF16, DT_INT8, DT_FLOAT16,
                        DT_FLOAT16, DT_INT8, DT_INT8, DT_INT8, DT_BF16, DT_INT8,
                        DT_FLOAT16, DT_INT8, DT_BF16, DT_INT8, DT_BF16, DT_FLOAT16,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8,
                        DT_BF16, DT_BF16, DT_FLOAT16, DT_FLOAT16, DT_INT8, DT_INT8,
                        DT_INT8, DT_INT8, DT_FLOAT16, DT_BF16, DT_FLOAT16, DT_BF16,
                        DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_INT8, DT_FLOAT16, DT_FLOAT16,
                        DT_BF16, DT_FLOAT16, DT_INT8, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                        DT_BF16, DT_BF16, DT_BF16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                        DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_BF16, DT_BF16, DT_BF16,
                        DT_BF16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_BF16,
                        DT_BF16, DT_HIFLOAT8, DT_FLOAT8_E4M3FN, DT_HIFLOAT8, DT_FLOAT8_E4M3FN, DT_HIFLOAT8,
                        DT_FLOAT8_E4M3FN, DT_HIFLOAT8, DT_FLOAT8_E4M3FN, DT_HIFLOAT8, DT_FLOAT8_E4M3FN, DT_HIFLOAT8,
                        DT_FLOAT8_E4M3FN, DT_HIFLOAT8, DT_FLOAT8_E4M3FN, DT_HIFLOAT8, DT_FLOAT8_E4M3FN, DT_HIFLOAT8,
                        DT_FLOAT8_E4M3FN, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_BF16,
                        DT_BF16, DT_HIFLOAT8, DT_HIFLOAT8, DT_HIFLOAT8, DT_HIFLOAT8, DT_HIFLOAT8,
                        DT_HIFLOAT8, DT_FLOAT8_E4M3FN, DT_FLOAT8_E4M3FN, DT_FLOAT8_E4M3FN, DT_FLOAT8_E4M3FN, DT_FLOAT8_E4M3FN,
                        DT_FLOAT8_E4M3FN, DT_HIFLOAT8, DT_HIFLOAT8, DT_HIFLOAT8, DT_FLOAT8_E4M3FN, DT_FLOAT8_E4M3FN,
                        DT_FLOAT8_E4M3FN, DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT8, DT_BF16, DT_FLOAT16, DT_BF16, DT_INT8, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8, DT_INT8,
                        DT_INT8, DT_INT8, DT_FLOAT8_E4M3FN, DT_FLOAT8_E4M3FN, DT_FLOAT8_E4M3FN, DT_FLOAT8_E4M3FN,
                        DT_FLOAT8_E4M3FN, DT_FLOAT8_E4M3FN, DT_FLOAT8_E4M3FN, DT_FLOAT8_E4M3FN, DT_FLOAT8_E4M3FN, DT_FLOAT8_E4M3FN,
                        DT_FLOAT8_E4M3FN, DT_FLOAT8_E4M3FN, DT_FLOAT8_E4M3FN, DT_FLOAT8_E4M3FN, DT_FLOAT8_E4M3FN, DT_FLOAT8_E4M3FN,
                        DT_FLOAT8_E4M3FN, DT_FLOAT8_E4M3FN, DT_FLOAT8_E4M3FN, DT_FLOAT8_E4M3FN, DT_FLOAT8_E4M3FN, DT_FLOAT8_E4M3FN,
                        DT_FLOAT8_E4M3FN, DT_FLOAT8_E4M3FN, DT_FLOAT8_E4M3FN, DT_FLOAT8_E4M3FN, DT_FLOAT8_E4M3FN, DT_FLOAT8_E4M3FN,
                        DT_FLOAT8_E4M3FN, DT_FLOAT8_E4M3FN, DT_FLOAT8_E4M3FN, DT_FLOAT8_E4M3FN, DT_FLOAT8_E4M3FN, DT_FLOAT8_E4M3FN,
                        DT_FLOAT8_E4M3FN, DT_FLOAT8_E4M3FN, DT_FLOAT8_E4M3FN, DT_FLOAT8_E4M3FN, DT_FLOAT8_E4M3FN, DT_FLOAT8_E4M3FN,
                        DT_FLOAT8_E4M3FN, DT_FLOAT8_E4M3FN, DT_FLOAT8_E4M3FN, DT_FLOAT8_E4M3FN, DT_FLOAT8_E4M3FN, DT_FLOAT8_E4M3FN,
                        DT_FLOAT8_E4M3FN, DT_FLOAT8_E4M3FN, DT_FLOAT8_E4M3FN, DT_FLOAT8_E4M3FN, DT_FLOAT8_E4M3FN, DT_FLOAT8_E4M3FN,
                        DT_FLOAT8_E4M3FN, DT_FLOAT8_E4M3FN, DT_FLOAT8_E4M3FN, DT_FLOAT8_E4M3FN, DT_FLOAT8_E4M3FN, DT_FLOAT8_E4M3FN,
                        DT_FLOAT8_E4M3FN, DT_FLOAT8_E4M3FN, DT_FLOAT8_E4M3FN, DT_FLOAT8_E4M3FN, DT_FLOAT8_E4M3FN, DT_FLOAT8_E4M3FN,
                        DT_HIFLOAT8, DT_HIFLOAT8, DT_HIFLOAT8, DT_HIFLOAT8, DT_HIFLOAT8, DT_HIFLOAT8,
                        DT_HIFLOAT8, DT_HIFLOAT8, DT_HIFLOAT8, DT_HIFLOAT8, DT_HIFLOAT8, DT_HIFLOAT8,
                        DT_HIFLOAT8, DT_HIFLOAT8, DT_HIFLOAT8, DT_HIFLOAT8, DT_HIFLOAT8, DT_HIFLOAT8,
                        DT_HIFLOAT8, DT_HIFLOAT8, DT_HIFLOAT8, DT_HIFLOAT8, DT_HIFLOAT8, DT_HIFLOAT8,
                        DT_HIFLOAT8, DT_HIFLOAT8, DT_HIFLOAT8, DT_HIFLOAT8, DT_HIFLOAT8, DT_HIFLOAT8,
                        DT_HIFLOAT8, DT_HIFLOAT8, DT_HIFLOAT8, DT_HIFLOAT8, DT_HIFLOAT8, DT_HIFLOAT8,
                        DT_HIFLOAT8, DT_HIFLOAT8, DT_HIFLOAT8, DT_HIFLOAT8, DT_HIFLOAT8, DT_HIFLOAT8,
                        DT_HIFLOAT8, DT_HIFLOAT8, DT_HIFLOAT8, DT_HIFLOAT8, DT_HIFLOAT8, DT_HIFLOAT8,
                        DT_HIFLOAT8, DT_HIFLOAT8, DT_HIFLOAT8, DT_HIFLOAT8, DT_HIFLOAT8, DT_HIFLOAT8,
                        DT_HIFLOAT8, DT_HIFLOAT8, DT_HIFLOAT8, DT_HIFLOAT8, DT_HIFLOAT8, DT_HIFLOAT8,
                        DT_HIFLOAT8, DT_HIFLOAT8, DT_HIFLOAT8, DT_HIFLOAT8, DT_BF16, DT_FLOAT16,
                        DT_BF16, DT_INT8})
            .FormatList({FORMAT_ND});
        aicore_config_95.Output("softmax_lse").ParamType(REQUIRED).DataTypeList({DT_FLOAT}).FormatList({FORMAT_ND});
        aicore_config_95.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn")
            .ExtendCfgInfo("opFile.value", "fused_infer_attention_score_apt")
            .ExtendCfgInfo("jitCompile.flag", "static_false,dynamic_false");
        this->AICore().AddConfig("ascend950", aicore_config_95);
    }
};
OP_ADD(FusedInferAttentionScore, optiling::FusedInferAttentionScoreCompileInfo);
} // namespace ops