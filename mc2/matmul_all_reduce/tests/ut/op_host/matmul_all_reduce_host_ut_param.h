/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MATMUL_ALL_REDUCE_HOST_UT_PARAM_H
#define MATMUL_ALL_REDUCE_HOST_UT_PARAM_H

#include <cstdint>
#include <string>
#include <vector>
#include <sstream>
#include "tiling_context_faker.h"
#include "infer_shape_context_faker.h"
#include "mc2_csv_case_loader.h"

namespace matmul_all_reduce_ut {

struct MatmulAllReduceHostUtParamBase {
    std::string case_name;
    std::vector<uint32_t> inputInstance;
    std::vector<uint32_t> outputInstance;
    std::string group;
    std::string reduce_op;
    bool is_trans_a;
    bool is_trans_b;
    int64_t comm_turn;
    int64_t antiquant_group_size;
    int64_t group_size;
    int64_t y_dtype;
    int64_t comm_quant_mode;
    ge::graphStatus expectResult;

    MatmulAllReduceHostUtParamBase(const csv_map& csvMap)
    {
        this->case_name = ReadMap(csvMap, "case_name");
        this->group = ReadMap(csvMap, "group");
        this->reduce_op = ReadMap(csvMap, "reduce_op");
        this->is_trans_a = stoi(ReadMap(csvMap, "is_trans_a"));
        this->is_trans_b = stoi(ReadMap(csvMap, "is_trans_b"));
        this->comm_turn = stoi(ReadMap(csvMap, "comm_turn"));
        this->antiquant_group_size = stoi(ReadMap(csvMap, "antiquant_group_size"));
        this->group_size = stoi(ReadMap(csvMap, "group_size"));
        this->y_dtype = stoi(ReadMap(csvMap, "y_dtype"));
        this->comm_quant_mode = stoi(ReadMap(csvMap, "comm_quant_mode"));
        this->expectResult = stoi(ReadMap(csvMap, "expectResult")) ? ge::GRAPH_SUCCESS : ge::GRAPH_FAILED;
    }
};

inline std::ostream& operator<<(std::ostream& os, const MatmulAllReduceHostUtParamBase& param)
{
    return os << param.case_name;
}

template<typename T>
inline std::string GetCaseInfoString(const testing::TestParamInfo<T>& info)
{
    return info.param.case_name;
}

const gert::TilingContextPara::TensorDescription TD_DEFAULT = {{}, ge::DT_UNDEFINED, ge::FORMAT_NULL};
struct MatmulAllReduceTilingUtParam: public MatmulAllReduceHostUtParamBase {
    gert::TilingContextPara::TensorDescription x1 = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription x2 = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription bias = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription x3 = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription antiquant_scale = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription antiquant_offset = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription dequant_scale = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription pertoken_scale = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription comm_quant_scale_1 = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription comm_quant_scale_2 = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription y = TD_DEFAULT;
    uint64_t ranksize;
    uint64_t expectTilingKey;
    std::string expectTilingDataHash;

    MatmulAllReduceTilingUtParam(const csv_map& csvMap):
        MatmulAllReduceHostUtParamBase(csvMap)
    {
        this->inputInstance.emplace_back(
            CreateTensor(csvMap, "x1_shape", "x1_dtype", "x1_format",
                x1));
        this->inputInstance.emplace_back(
            CreateTensor(csvMap, "x2_shape", "x2_dtype", "x2_format",
                x2));
        this->inputInstance.emplace_back(
            CreateTensor(csvMap, "bias_shape", "bias_dtype", "bias_format",
                bias));
        this->inputInstance.emplace_back(
            CreateTensor(csvMap, "antiquant_scale_shape", "antiquant_scale_dtype", "antiquant_scale_format",
                x3));
        this->inputInstance.emplace_back(
            CreateTensor(csvMap, "antiquant_offset_shape", "antiquant_offset_dtype", "antiquant_offset_format",
                antiquant_offset));
        this->inputInstance.emplace_back(
            CreateTensor(csvMap, "dequant_scale_shape", "dequant_scale_dtype", "dequant_scale_format",
                dequant_scale));
        this->inputInstance.emplace_back(
            CreateTensor(csvMap, "pertoken_scale_shape", "pertoken_scale_dtype", "pertoken_scale_format",
                pertoken_scale));
        this->inputInstance.emplace_back(
            CreateTensor(csvMap, "comm_quant_scale_1_shape", "comm_quant_scale_1_dtype", "comm_quant_scale_1_format",
                comm_quant_scale_1));
        this->inputInstance.emplace_back(
            CreateTensor(csvMap, "comm_quant_scale_2_shape", "comm_quant_scale_2_dtype", "comm_quant_scale_2_format",
                comm_quant_scale_2));
        this->outputInstance.emplace_back(
            CreateTensor(csvMap, "output_y_shape", "output_y_dtype", "output_y_format",
                y));
        this->ranksize = stoi(ReadMap(csvMap, "ranksize"));
        if(this->expectResult == 1) {
            this->expectTilingKey = stoi(ReadMap(csvMap, "expectTilingKey"));
            this->expectTilingDataHash = ReadMap(csvMap, "expectTilingDataHash");
        }
    }

private:
    int CreateTensor(const csv_map& csvMap, const std::string& shapeKey, const std::string& dtypeKey,
        const std::string& formatKey, gert::TilingContextPara::TensorDescription& outTd)
    {
        std::string shapeStr = ReadMap(csvMap, shapeKey);
        if (shapeStr.empty()) return 0;
        std::string dtypeStr = ReadMap(csvMap, dtypeKey);
        if (dtypeStr.empty()) return 0;
        std::string formatStr = ReadMap(csvMap, formatKey);
        if (formatStr.empty()) return 0;

        gert::StorageShape shape = GetStorageShape(shapeStr);
        ge::DataType dtype = ReadMap(GE_DTYPE, dtypeStr, ge::DT_UNDEFINED);
        ge::Format format = ReadMap(GE_FORMAT, formatStr, ge::FORMAT_NULL);
        outTd = gert::TilingContextPara::TensorDescription(shape, dtype, format);
        return 1;
    }
};

} // namespace matmul_all_reduce_ut

#endif // MATMUL_ALL_REDUCE_HOST_UT_PARAM_H
