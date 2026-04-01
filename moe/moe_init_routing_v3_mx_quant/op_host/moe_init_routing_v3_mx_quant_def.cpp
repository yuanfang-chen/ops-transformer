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
 * \file moe_init_routing_v3_mx_quant_def.cpp
 * \brief MoeInitRoutingV3MxQuant operator definition
 */

#include "register/op_def_registry.h"

namespace ops {
class MoeInitRoutingV3MxQuant : public OpDef {
public:
    explicit MoeInitRoutingV3MxQuant(const char* name) : OpDef(name)
    {
        // Two dtype configs: config0 = dst_type=0 (e4m3fn), config1 = dst_type=1 (e5m2)
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();

        this->Input("expert_idx")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();

        this->Input("scale")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_BF16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();

        this->Input("offset")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_BF16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();

        // y: fp8_e4m3fn (ge::DT_FLOAT8_E4M3FN=36) for config0, fp8_e5m2 (ge::DT_FLOAT8_E5M2=35) for config1
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});

        // mxscale: fp8_e8m0 (ge::DT_FLOAT8_E8M0=37)
        this->Output("mxscale")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});

        this->Output("expanded_row_idx")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});

        this->Output("expert_tokens_count_or_cumsum")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});

        this->Output("expanded_scale")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_BF16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});

        this->Attr("active_num")
            .AttrType(OPTIONAL)
            .Int(-1);

        this->Attr("expert_capacity")
            .AttrType(OPTIONAL)
            .Int(-1);

        this->Attr("expert_num")
            .AttrType(OPTIONAL)
            .Int(-1);

        this->Attr("drop_pad_mode")
            .AttrType(OPTIONAL)
            .Int(0);

        this->Attr("expert_tokens_count_or_cumsum_flag")
            .AttrType(OPTIONAL)
            .Int(0);

        this->Attr("expert_tokens_before_capacity_flag")
            .AttrType(OPTIONAL)
            .Bool(false);

        this->Attr("axis")
            .AttrType(OPTIONAL)
            .Int(-1);

        this->Attr("round_mode")
            .AttrType(OPTIONAL)
            .String("rint");

        this->Attr("dst_type")
            .AttrType(OPTIONAL)
            .Int(0);

        this->Attr("blocksize")
            .AttrType(OPTIONAL)
            .Int(32);

        this->Attr("scale_alg")
            .AttrType(OPTIONAL)
            .Int(0);

        OpAICoreConfig aicoreConfig950;
        aicoreConfig950.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("opFile.value", "moe_init_routing_v3_mx_quant_apt");
        this->AICore().AddConfig("ascend950", aicoreConfig950);

    }
};

OP_ADD(MoeInitRoutingV3MxQuant);
}  // namespace ops
