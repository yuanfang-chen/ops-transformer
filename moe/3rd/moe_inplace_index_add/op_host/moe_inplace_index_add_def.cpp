/* *
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/* !
 * \file moe_inplace_index_add_def.cpp
 * \brief
 */
#include "register/op_def_registry.h"

namespace ops {
static const std::vector<ge::DataType> varSupportedTypes = {
    ge::DT_BF16,  ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_UINT8, ge::DT_INT8,    ge::DT_INT16,
    ge::DT_INT32, ge::DT_INT64,   ge::DT_BOOL,  ge::DT_BF16,  ge::DT_FLOAT16, ge::DT_FLOAT,
    ge::DT_UINT8, ge::DT_INT8,    ge::DT_INT16, ge::DT_INT32, ge::DT_INT64,   ge::DT_BOOL};

static const std::vector<ge::DataType> indicesSupportedTypes = {
    ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32,
    ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
    ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64};

static const std::vector<ge::Format> format = {
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND};

class MoeInplaceIndexAdd : public OpDef {
public:
    explicit MoeInplaceIndexAdd(const char *name) : OpDef(name) {
        this->Input("var")
            .ParamType(REQUIRED)
            .DataType(varSupportedTypes)
            .Format(format)
            .UnknownShapeFormat(format);
        this->Input("indices")
            .ParamType(REQUIRED)
            .DataType(indicesSupportedTypes)
            .Format(format)
            .UnknownShapeFormat(format);
        this->Input("updates")
            .ParamType(REQUIRED)
            .DataType(varSupportedTypes)
            .Format(format)
            .UnknownShapeFormat(format);
        this->Input("alpha")
            .ParamType(OPTIONAL)
            .DataType(varSupportedTypes)
            .Format(format)
            .UnknownShapeFormat(format);
        this->Output("var")
            .ParamType(REQUIRED)
            .DataType(varSupportedTypes)
            .Format(format)
            .UnknownShapeFormat(format);
        this->Attr("axis").AttrType(REQUIRED).Int();
        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true);
        this->AICore().AddConfig("ascend950", aicoreConfig);
    }
};

OP_ADD(MoeInplaceIndexAdd);
}