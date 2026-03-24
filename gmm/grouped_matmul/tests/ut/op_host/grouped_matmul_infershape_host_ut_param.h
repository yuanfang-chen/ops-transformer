/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GROUPED_MATMUL_INFERSHAPE_HOST_UT_PARAM_H
#define GROUPED_MATMUL_INFERSHAPE_HOST_UT_PARAM_H

#include "op_host_csv_case_loader.h"

namespace GroupedMatmulUT {

struct GroupedMatmulInferShapeUtParam {
    std::string case_name;
    std::string soc_version;
    ge::graphStatus expectResult;
    std::vector<std::vector<int64_t>> expectOutputShape;

    gert::InfershapeContextPara::TensorDescription x = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription weight = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription bias = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription scale = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription offset = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription antiquant_scale = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription antiquant_offset = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription group_list = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription pertoken_scale = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription y = ID_DEFAULT;

    int64_t split_item = 3;
    int64_t dtype = 0;
    bool transpose_weight = false;
    bool transpose_x = false;
    int64_t group_type = 0;
    int64_t group_list_type = 0;
    int64_t act_type = 0;

    std::vector<uint32_t> inputInstance;
    std::vector<uint32_t> outputInstance;

    explicit GroupedMatmulInferShapeUtParam(const csv_map &csvMap)
    {
        auto getTensor = [&csvMap](const std::string &shapeKey, const std::string &dtypeKey,
                                   const std::string &formatKey,
                                   gert::InfershapeContextPara::TensorDescription &out) -> uint32_t {
            std::string shapeStr = ReadMap(csvMap, shapeKey);
            std::string dtypeStr = ReadMap(csvMap, dtypeKey);
            std::string formatStr = ReadMap(csvMap, formatKey, "ND");
            if (shapeStr.empty() && dtypeStr.empty()) {
                return 0;
            }
            gert::StorageShape shape = GetStorageShape(shapeStr);
            ge::DataType dtype = Str2DTypeGE(dtypeStr);
            ge::Format format = ReadMap(GE_FORMAT, formatStr, ge::FORMAT_ND);
            out = gert::InfershapeContextPara::TensorDescription(shape, dtype, format);
            return 1;
        };

        case_name = ReadMap(csvMap, "case_name");
        soc_version = ReadMap(csvMap, "soc_version");
        expectResult = Str2StatusGE(ReadMap(csvMap, "expectResult"));

        inputInstance.emplace_back(getTensor("x_shape", "x_dtype", "x_format", x));
        inputInstance.emplace_back(getTensor("weight_shape", "weight_dtype", "weight_format", weight));
        inputInstance.emplace_back(getTensor("bias_shape", "bias_dtype", "bias_format", bias));
        inputInstance.emplace_back(getTensor("scale_shape", "scale_dtype", "scale_format", scale));
        inputInstance.emplace_back(getTensor("offset_shape", "offset_dtype", "offset_format", offset));
        inputInstance.emplace_back(getTensor("antiquant_scale_shape", "antiquant_scale_dtype",
                                             "antiquant_scale_format", antiquant_scale));
        inputInstance.emplace_back(getTensor("antiquant_offset_shape", "antiquant_offset_dtype",
                                             "antiquant_offset_format", antiquant_offset));
        inputInstance.emplace_back(getTensor("group_list_shape", "group_list_dtype", "group_list_format", group_list));
        inputInstance.emplace_back(getTensor("pertoken_scale_shape", "pertoken_scale_dtype",
                                             "pertoken_scale_format", pertoken_scale));

        outputInstance.emplace_back(getTensor("y_shape", "y_dtype", "y_format", y));

        split_item = std::stoll(ReadMap(csvMap, "split_item"));
        dtype = std::stoll(ReadMap(csvMap, "dtype"));
        transpose_weight = std::stoi(ReadMap(csvMap, "transpose_weight")) != 0;
        transpose_x = std::stoi(ReadMap(csvMap, "transpose_x")) != 0;
        group_type = std::stoll(ReadMap(csvMap, "group_type"));
        group_list_type = std::stoll(ReadMap(csvMap, "group_list_type"));
        act_type = std::stoll(ReadMap(csvMap, "act_type"));

        if (expectResult == ge::GRAPH_SUCCESS) {
            expectOutputShape = {GetShapeArr(ReadMap(csvMap, "expectOutputShape"))};
        }
    }
};

inline std::ostream &operator<<(std::ostream &os, const GroupedMatmulInferShapeUtParam &param)
{
    return os << param.case_name;
}

} // namespace GroupedMatmulUT

#endif // GROUPED_MATMUL_INFERSHAPE_HOST_UT_PARAM_H

