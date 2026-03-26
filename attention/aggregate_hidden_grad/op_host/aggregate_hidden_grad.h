/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!\n * \file aggregate_hidden_grad.h\n * \brief Host tiling class declaration for aggregate_hidden_grad\n */
#ifndef OPS_TRANSFORMER_ATTENTION_AGGREGATE_HIDDEN_GRAD_OP_HOST_H
#define OPS_TRANSFORMER_ATTENTION_AGGREGATE_HIDDEN_GRAD_OP_HOST_H

#include <tiling/tiling_api.h>
#include "register/tilingdata_base.h"
#include "tiling_base/tiling_base.h"

namespace optiling {

struct AggregateHiddenGradCompileInfo {
    uint64_t aivNum{0UL};
    uint64_t ubSize{0UL};
};

class AggregateHiddenGradTiling : public Ops::Transformer::OpTiling::TilingBaseClass {
public:
    explicit AggregateHiddenGradTiling(gert::TilingContext *context)
        : Ops::Transformer::OpTiling::TilingBaseClass(context) {}
    ~AggregateHiddenGradTiling() override = default;

protected:
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;

private:
    // helpers
    bool PreparePlatformInfo();
    void CalcInterCoreSplit();
    void CalcUbTiling();

    // cached
    uint64_t aivNum_{0};
    uint64_t ubSize_{0};

    // shapes and dtype
    int64_t S_{0}, B_{0}, H_{0}, W_{0}; // W must be 3
    uint32_t dtypeSize_{2};
    bool hasMask_{false};

    // inter-core split
    int64_t hUB_{64}, bUB_{1}, sUB_{1};
    int64_t hLoopCnt_{0}, bLoopCnt_{0}, sLoopCnt_{0};
    int64_t hUBTail_{0}, bUBTail_{0}, sUBTail_{0};

};

} // namespace optiling

#endif // OPS_TRANSFORMER_ATTENTION_AGGREGATE_HIDDEN_GRAD_OP_HOST_H
