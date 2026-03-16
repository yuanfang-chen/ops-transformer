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
 * \file chunk_gated_delta_rule_recurrence_def.cpp
 * \brief OpDef for ChunkGatedDeltaRuleRecurrence (Ascend 950 / DAV_3510 only)
 *
 * Inputs (all REQUIRED unless noted):
 *   0  initial_state  float32  [b, hv, dv, dk]  in-place read+write
 *   1  kgexp          float32  [hv, n_chunks, cs, dk]
 *   2  value          float32  [hv, n_chunks, cs, dv]
 *   3  k_cumdecay     float32  [hv, n_chunks, cs, dk]
 *   4  qgexp          float32  [hv, n_chunks, cs, dk]
 *   5  gexp           float32  [hv, n_chunks, cs, 1]
 *   6  cu_seqlens     int32    [b+1]
 *
 * Outputs:
 *   0  initial_state  float32  [b, hv, dv, dk]   (updated in-place, also listed as output)
 *   1  attn_inter_out float32  [hv, n_chunks, cs, dv]
 *   2  v_new_out      float32  [hv, n_chunks, cs, dv]
 *
 * Attrs:
 *   scale_value  OPTIONAL  Float  default=1.0
 */
#include "register/op_def_registry.h"

namespace ops {
class ChunkGatedDeltaRuleRecurrence : public OpDef {
public:
    explicit ChunkGatedDeltaRuleRecurrence(const char *name) : OpDef(name)
    {
        this->Input("initial_state")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("kgexp")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("value")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("k_cumdecay")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("qgexp")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("gexp")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("cu_seqlens")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("initial_state")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("attn_inter_out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("v_new_out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("scale_value").AttrType(OPTIONAL).Float(1.0);

        OpAICoreConfig aicCfg;
        aicCfg.DynamicCompileStaticFlag(true)
              .DynamicFormatFlag(true)
              .DynamicRankSupportFlag(true)
              .DynamicShapeSupportFlag(true)
              .NeedCheckSupportFlag(false)
              .ExtendCfgInfo("aclnnSupport.value", "support_aclnn")
              .ExtendCfgInfo("opFile.value", "chunk_gated_delta_rule_recurrence");
        this->AICore().AddConfig("ascend950", aicCfg);
    }
};

OP_ADD(ChunkGatedDeltaRuleRecurrence);

} // namespace ops
