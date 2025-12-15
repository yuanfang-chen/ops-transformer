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
 * \file matmul_allto_all_tiling_base.h
 * \brief
 */
#ifndef MATMUL_ALLTO_ALL_TILING_BASE_H
#define MATMUL_ALLTO_ALL_TILING_BASE_H

#include "tiling/tiling_api.h"
#include "tiling/mc2_tiling_utils.h"
#include "tiling_base/tiling_base.h"
#include "tiling_base/tiling_templates_registry.h"
#include "common/matmul_allto_all_util_tiling.h"
#include "common/allto_all_formulaic_tiling.h"
#include "../../op_kernel/arch35/matmul_allto_all_tiling_key.h"

namespace MC2Tiling {

class MatmulAllToAllTilingBase : public Ops::Transformer::OpTiling::TilingBaseClass {
public:
    explicit MatmulAllToAllTilingBase(gert::TilingContext *context) : TilingBaseClass(context)
    {
    }
    ~MatmulAllToAllTilingBase() override = default;

    void Reset(gert::TilingContext *context) override
    {
        TilingBaseClass::Reset(context);
    }

protected:
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetWorkspaceSize() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus CheckInput()
    {
        return ge::GRAPH_SUCCESS;
    }
    ge::graphStatus TileCommAndCompute();
    void SetUserWorkSpace();

    platform_ascendc::SocVersion socVersion_;
    const char *opName_{nullptr};
    uint32_t libApiWorkSpaceSize_{0};
    TilingContextInfo contextInfo;
    TilingInferredInfo inferredInfo;

private:
    // 功能后移，基类的GetShapeAttrsInfo在isCapable之前，当前将校验和参数获取放到子类的DoOptiling中
    ge::graphStatus GetShapeAttrsInfo() override;
    // 在框架中没有实际使用
    ge::graphStatus DoLibApiTiling() override;
};
} // namespace MC2Tiling
#endif
