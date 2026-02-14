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
 * \file matrix_transform_rope_def.cpp
 * \brief 算子定义文件，定义算子的输入输出、数据类型、格式等
 */
#include "register/op_def_registry.h"

namespace ops {
class MatrixTransformRope : public OpDef {
public:
    explicit MatrixTransformRope(const char* name) : OpDef(name)
    {
        // 定义输入tensor: x
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        // 定义输入tensor: cos
        this->Input("cos")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        // 定义输入tensor: sin
        this->Input("sin")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        // 定义输入tensor: rotate
        this->Input("rotate")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        // 定义输出tensor: out
        this->Output("out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        // OpAICoreConfig原型注册与管理
        OpAICoreConfig aicConfig;
        aicConfig.DynamicCompileStaticFlag(true) // 用于标识该算子是否支持构图时的静态Shape编译
            .DynamicFormatFlag(false)                // 用于标识是否自动推导算子输入输出支持的dtype和format
            .DynamicRankSupportFlag(true)            // 标识该算子是否支持dynamicRank（动态维度）
            .DynamicShapeSupportFlag(true)           // 标识该算子是否支持构图时的动态Shape场景
            .NeedCheckSupportFlag(false);            // 标识是否在算子融合阶段调用算子参数校验函数进行data type与Shape的校验

        // 添加支持的芯片型号
        this->AICore().AddConfig("ascend910b", aicConfig);
        this->AICore().AddConfig("ascend910_93", aicConfig);
    }
};

// 注册算子
OP_ADD(MatrixTransformRope);

} // namespace ops
