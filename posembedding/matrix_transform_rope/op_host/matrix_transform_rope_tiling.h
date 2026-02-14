/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to License for details. You may not use this file except in compliance with License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file matrix_transform_rope_tiling.h
 * \brief Tiling数据结构定义
 */
#ifndef __OP_HOST_MATRIX_TRANSFORM_ROPE_TILING_H__
#define __OP_HOST_MATRIX_TRANSFORM_ROPE_TILING_H__

#include <tiling/tiling_api.h>
#include "register/tilingdata_base.h"
#include "../../../common/include/error_ops/error.h"

namespace optiling {

// 定义Tiling数据结构
BEGIN_TILING_DATA_DEF(MatrixTransformRopeTilingData)
    // TILING_DATA_FIELD_DEF_STRUCT(CubeTiling, cubeTiling);
    TILING_DATA_FIELD_DEF(uint32_t, coreNum);
    TILING_DATA_FIELD_DEF(uint64_t, totalElements);
END_TILING_DATA_DEF;

// 定义CompileInfo结构体，用于存储编译时平台信息
struct MatrixTransformRopeCompileInfo {
    uint64_t aicNum{0UL};
    uint64_t aivNum{0UL};
    uint64_t ubSize{0UL};
    uint64_t l1Size{0UL};
    uint64_t l0CSize{0UL};
    uint64_t l0ASize{0UL};
    uint64_t l0BSize{0UL};
};

class MatrixTransformRopeBaseTiling : public Ops::Transformer::OpTiling::TilingBaseClass {
public:
    explicit MatrixTransformRopeBaseTiling(gert::TilingContext* context) : Ops::Transformer::OpTiling::TilingBaseClass(context) {};

    ~MatrixTransformRopeBaseTiling() override = default;

protected:
    bool IsCapable() override
    {
        return true;
    }

    // 1、获取平台信息比如CoreNum、UB/L1/L0C资源大小
    ge::graphStatus GetPlatformInfo() override {return ge::GRAPH_SUCCESS;};
    // 2、获取INPUT/OUTPUT/ATTR信息
    ge::graphStatus GetShapeAttrsInfo() override{return ge::GRAPH_SUCCESS;};
    // 3、计算数据切分TilingData
    ge::graphStatus DoOpTiling() override;
    // 4、计算高阶API的TilingData
    ge::graphStatus DoLibApiTiling() override{return ge::GRAPH_SUCCESS;};
    // 5、计算TilingKey
    uint64_t GetTilingKey() const override;
    // 6、计算Workspace大小
    ge::graphStatus GetWorkspaceSize() override{return ge::GRAPH_SUCCESS;};
    // 7、保存Tiling数据
    ge::graphStatus PostTiling() override;

    // 添加自定义辅助方法
    ge::graphStatus ParseInputAndAttr();
    void PrintTilingData();
    void FillTilingData();
    ge::graphStatus TilingProcess();

private:
    MatrixTransformRopeTilingData tilingData_;
    uint32_t blockDim_;     // AICore

    // 输入tensor信息
    int64_t xB_{1};
    int64_t xN_{1};
    int64_t xS_{1};
    int64_t xD_{1};
    int64_t cosB_{1};
    int64_t cosS_{1};
    int64_t sinB_{1};
    int64_t sinS_{1};
    int64_t rotateD_{1};
};

} // namespace optiling

#endif // __OP_HOST_MATRIX_TRANSFORM_ROPE_TILING_H__
