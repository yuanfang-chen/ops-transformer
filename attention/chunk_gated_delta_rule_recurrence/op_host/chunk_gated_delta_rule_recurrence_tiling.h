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
 * \file chunk_gated_delta_rule_recurrence_tiling.h
 * \brief
 */
#ifndef __OP_HOST_CHUNK_GATED_DELTA_RULE_RECURRENCE_TILING_H__
#define __OP_HOST_CHUNK_GATED_DELTA_RULE_RECURRENCE_TILING_H__

#include <tiling/tiling_api.h>
#include "register/tilingdata_base.h"
#include "tiling_base/tiling_base.h"
#include "err/ops_err.h"
#include "../op_kernel/chunk_gated_delta_rule_recurrence_tiling_data.h"

namespace optiling {
using namespace ChunkGatedDeltaRuleRecurrence;

struct ChunkGatedDeltaRuleRecurrenceCompileInfo {
    uint64_t aivNum{0UL};
    uint64_t ubSize{0UL};
};

struct ChunkGatedDeltaRuleRecurrenceInfo {
    const char *opName = "ChunkGatedDeltaRuleRecurrence";
};

class ChunkGatedDeltaRuleRecurrenceTiling
    : public Ops::Transformer::OpTiling::TilingBaseClass {
public:
    explicit ChunkGatedDeltaRuleRecurrenceTiling(gert::TilingContext *context)
        : Ops::Transformer::OpTiling::TilingBaseClass(context)
    {
        InitCompileInfo();
    }
    ~ChunkGatedDeltaRuleRecurrenceTiling() override = default;

protected:
    bool IsCapable() override { return true; }
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;

protected:
    void InitCompileInfo();
    void PrintTilingData();

    ge::graphStatus CheckContext();
    ge::graphStatus AnalyzeDtype();
    ge::graphStatus AnalyzeShapes();
    ge::graphStatus GetScaleAttr();
    ge::graphStatus CalDvTile();

    bool CheckDim(const gert::Shape &shape, size_t expected, const std::string &name);

    ChunkGatedDeltaRuleRecurrenceCompileInfo compileInfo_;
    ChunkGatedDeltaRuleRecurrenceTilingData  tilingData_;
    ChunkGatedDeltaRuleRecurrenceInfo        inputParams_;
};

} // namespace optiling
#endif // __OP_HOST_CHUNK_GATED_DELTA_RULE_RECURRENCE_TILING_H__
