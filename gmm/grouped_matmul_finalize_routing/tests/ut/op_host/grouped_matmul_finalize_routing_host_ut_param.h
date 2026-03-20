/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GROUPED_MATMUL_FINALIZE_ROUTING_HOST_UT_PARAM_H
#define GROUPED_MATMUL_FINALIZE_ROUTING_HOST_UT_PARAM_H

#include "op_host_csv_case_loader.h"

namespace GroupedMatmulFinalizeRoutingUT {

struct GroupedMatmulFinalizeRoutingInferShapeUtParam {
    std::string case_name;
    ge::graphStatus expectResult;
    std::vector<std::vector<int64_t>> expectOutputShape;

    gert::InfershapeContextPara::TensorDescription x = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription w = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription scale = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription bias = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription pertoken_scale = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription group_list = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription shared_input = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription logit = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription row_index = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription y = ID_DEFAULT;

    int64_t dtype = 0;
    float shared_input_weight = 1.0F;
    int64_t shared_input_offset = 0;
    bool transpose_x = false;
    bool transpose_w = false;
    int64_t output_bs = 0;
    int64_t group_list_type = 1;

    std::vector<uint32_t> inputInstance;
    std::vector<uint32_t> outputInstance;

    explicit GroupedMatmulFinalizeRoutingInferShapeUtParam(const csv_map &csvMap)
    {
        case_name = ReadMap(csvMap, "case_name");
        expectResult = Str2StatusGE(ReadMap(csvMap, "expectResult"));

        inputInstance.emplace_back(GetTensorGE(csvMap, "x_shape", "x_dtype", "x_format", x));
        inputInstance.emplace_back(GetTensorGE(csvMap, "w_shape", "w_dtype", "w_format", w));
        inputInstance.emplace_back(GetTensorGE(csvMap, "scale_shape", "scale_dtype", "scale_format", scale));
        inputInstance.emplace_back(GetTensorGE(csvMap, "bias_shape", "bias_dtype", "bias_format", bias));
        inputInstance.emplace_back(
            GetTensorGE(csvMap, "pertoken_scale_shape", "pertoken_scale_dtype", "pertoken_scale_format", pertoken_scale));
        inputInstance.emplace_back(GetTensorGE(csvMap, "group_list_shape", "group_list_dtype", "group_list_format", group_list));
        inputInstance.emplace_back(
            GetTensorGE(csvMap, "shared_input_shape", "shared_input_dtype", "shared_input_format", shared_input));
        inputInstance.emplace_back(GetTensorGE(csvMap, "logit_shape", "logit_dtype", "logit_format", logit));
        inputInstance.emplace_back(GetTensorGE(csvMap, "row_index_shape", "row_index_dtype", "row_index_format", row_index));

        outputInstance.emplace_back(GetTensorGE(csvMap, "y_shape", "y_dtype", "y_format", y));

        dtype = std::stoll(ReadMap(csvMap, "dtype"));
        shared_input_weight = std::stof(ReadMap(csvMap, "shared_input_weight"));
        shared_input_offset = std::stoll(ReadMap(csvMap, "shared_input_offset"));
        transpose_x = std::stoi(ReadMap(csvMap, "transpose_x")) != 0;
        transpose_w = std::stoi(ReadMap(csvMap, "transpose_w")) != 0;
        output_bs = std::stoll(ReadMap(csvMap, "output_bs"));
        group_list_type = std::stoll(ReadMap(csvMap, "group_list_type"));

        if (expectResult == ge::GRAPH_SUCCESS) {
            expectOutputShape = {GetShapeArr(ReadMap(csvMap, "expectOutputShape"))};
        }
    }
};

inline std::ostream &operator<<(std::ostream &os, const GroupedMatmulFinalizeRoutingInferShapeUtParam &param)
{
    return os << param.case_name;
}
}  // namespace GroupedMatmulFinalizeRoutingUT

#endif  // GROUPED_MATMUL_FINALIZE_ROUTING_HOST_UT_PARAM_H
