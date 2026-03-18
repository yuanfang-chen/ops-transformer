/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MATMUL_REDUCE_SCATTER_V2_HOST_UT_PARAM_H
#define MATMUL_REDUCE_SCATTER_V2_HOST_UT_PARAM_H

#include <sstream>
#include "op_host_csv_case_loader.h"

namespace MatmulReduceScatterV2UT {

struct MatmulReduceScatterV2HostUtParamBase {
    std::string case_name;
    std::vector<uint32_t> inputInstance;
    std::vector<uint32_t> outputInstance;
    std::string group;
    std::string reduce_op;
    bool is_trans_a;
    bool is_trans_b;
    int64_t comm_turn;
    int64_t rank_size;
    int64_t block_size;
    int64_t group_size;
    bool is_amax_out;
    ge::DataType y_dtype;
    std::string comm_mode;
    ge::graphStatus expectResult;

    MatmulReduceScatterV2HostUtParamBase(const csv_map& csvMap)
    {
        this->case_name = ReadMap(csvMap, "case_name");
        this->group = ReadMap(csvMap, "group");
        this->reduce_op = ReadMap(csvMap, "reduce_op");
        this->is_trans_a = stoi(ReadMap(csvMap, "is_trans_a"));
        this->is_trans_b = stoi(ReadMap(csvMap, "is_trans_b"));
        this->comm_turn = stoll(ReadMap(csvMap, "comm_turn"));
        this->rank_size = stoll(ReadMap(csvMap, "rank_size"));
        this->block_size = stoll(ReadMap(csvMap, "block_size"));
        this->group_size = stoll(ReadMap(csvMap, "group_size"));
        this->is_amax_out = stoi(ReadMap(csvMap, "is_amax_out"));
        this->y_dtype = Str2DTypeGE(ReadMap(csvMap, "y_dtype"));
        this->comm_mode = ReadMap(csvMap, "comm_mode");
        this->expectResult = Str2StatusGE(ReadMap(csvMap, "expectResult"));
    }
};

inline std::ostream& operator<<(std::ostream& os, const MatmulReduceScatterV2HostUtParamBase& param)
{
    return os << param.case_name;
}

struct MatmulReduceScatterV2TilingUtParam: public MatmulReduceScatterV2HostUtParamBase {
    gert::TilingContextPara::TensorDescription x1 = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription x2 = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription bias = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription x1_scale = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription x2_scale = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription quant_scale = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription y = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription amax_out = TD_DEFAULT;
    std::string soc;
    uint64_t coreNum;
    uint64_t ubsize;
    uint64_t rankNum;
    uint64_t expectTilingKey;
    std::string expectTilingDataHash;

    MatmulReduceScatterV2TilingUtParam(const csv_map& csvMap):
        MatmulReduceScatterV2HostUtParamBase(csvMap)
    {
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "x1_shape", "x1_dtype", "x1_format",
                x1));
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "x2_shape", "x2_dtype", "x2_format",
                x2));
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "bias_shape", "bias_dtype", "bias_format",
                bias));
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "x1_scale_shape", "x1_scale_dtype", "x1_scale_format",
                x1_scale));
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "x2_scale_shape", "x2_scale_dtype", "x2_scale_format",
                x2_scale));
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "quant_scale_shape", "quant_scale_dtype", "quant_scale_format",
                quant_scale));

        this->outputInstance.emplace_back(
            GetTensorGE(csvMap, "output_y_shape", "output_y_dtype", "output_y_format",
                y));
        this->outputInstance.emplace_back(
            GetTensorGE(csvMap, "amax_out_shape", "amax_out_dtype", "amax_out_format",
                amax_out));

        this->soc = ReadMap(csvMap, "soc");
        this->coreNum = stoull(ReadMap(csvMap, "core_num"));
        this->ubsize = stoull(ReadMap(csvMap, "ubsize"));
        this->rankNum = stoull(ReadMap(csvMap, "rankNum"));

        if(this->expectResult == ge::GRAPH_SUCCESS) {
            this->expectTilingKey = stoull(ReadMap(csvMap, "expectTilingKey"));
            this->expectTilingDataHash = ReadMap(csvMap, "expectTilingDataHash");
        }
    }
};
/*
struct MatmulReduceScatterV2InferShapeUtParam: public MatmulReduceScatterV2HostUtParamBase {
    gert::InfershapeContextPara::TensorDescription x1 = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription x2 = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription bias = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription x3 = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription antiquant_scale = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription antiquant_offset = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription dequant_scale = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription pertoken_scale = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription comm_quant_scale_1 = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription comm_quant_scale_2 = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription y = ID_DEFAULT;
    uint64_t ranksize;
    std::vector<std::vector<int64_t>> expectOutputShape;

    MatmulReduceScatterV2InferShapeUtParam(const csv_map& csvMap):
        MatmulReduceScatterV2HostUtParamBase(csvMap)
    {
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "x1_shape", "x1_dtype", "x1_format",
                x1));
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "x2_shape", "x2_dtype", "x2_format",
                x2));
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "bias_shape", "bias_dtype", "bias_format",
                bias));
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "x3_shape", "x3_dtype", "x3_format",
                x3));
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "antiquant_scale_shape", "antiquant_scale_dtype", "antiquant_scale_format",
                antiquant_scale));
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "antiquant_offset_shape", "antiquant_offset_dtype", "antiquant_offset_format",
                antiquant_offset));
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "dequant_scale_shape", "dequant_scale_dtype", "dequant_scale_format",
                dequant_scale));
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "pertoken_scale_shape", "pertoken_scale_dtype", "pertoken_scale_format",
                pertoken_scale));
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "comm_quant_scale_1_shape", "comm_quant_scale_1_dtype", "comm_quant_scale_1_format",
                comm_quant_scale_1));
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "comm_quant_scale_2_shape", "comm_quant_scale_2_dtype", "comm_quant_scale_2_format",
                comm_quant_scale_2));

        this->outputInstance.emplace_back(1);

        this->ranksize = stoull(ReadMap(csvMap, "ranksize"));
        if(this->expectResult == ge::GRAPH_SUCCESS) {
            this->expectOutputShape = {GetShapeArr(ReadMap(csvMap, "expectOutputShape"))};
        }
    }
};

struct MatmulReduceScatterV2InferDataTypeUtParam: public MatmulReduceScatterV2HostUtParamBase {
    ge::DataType x1 = ge::DT_UNDEFINED;
    ge::DataType x2 = ge::DT_UNDEFINED;
    ge::DataType bias = ge::DT_UNDEFINED;
    ge::DataType x3 = ge::DT_UNDEFINED;
    ge::DataType antiquant_scale = ge::DT_UNDEFINED;
    ge::DataType antiquant_offset = ge::DT_UNDEFINED;
    ge::DataType dequant_scale = ge::DT_UNDEFINED;
    ge::DataType pertoken_scale = ge::DT_UNDEFINED;
    ge::DataType comm_quant_scale_1 = ge::DT_UNDEFINED;
    ge::DataType comm_quant_scale_2 = ge::DT_UNDEFINED;
    ge::DataType y = ge::DT_UNDEFINED;

    MatmulReduceScatterV2InferDataTypeUtParam(const csv_map& csvMap):
        MatmulReduceScatterV2HostUtParamBase(csvMap)
    {
        this->inputInstance.emplace_back(GetDataTypeGE(csvMap, "x1_dtype", x1));
        this->inputInstance.emplace_back(GetDataTypeGE(csvMap, "x2_dtype", x2));
        this->inputInstance.emplace_back(GetDataTypeGE(csvMap, "bias_dtype", bias));
        this->inputInstance.emplace_back(GetDataTypeGE(csvMap, "x3_dtype", x3));
        this->inputInstance.emplace_back(GetDataTypeGE(csvMap, "antiquant_scale_dtype", antiquant_scale));
        this->inputInstance.emplace_back(GetDataTypeGE(csvMap, "antiquant_offset_dtype", antiquant_offset));
        this->inputInstance.emplace_back(GetDataTypeGE(csvMap, "dequant_scale_dtype", dequant_scale));
        this->inputInstance.emplace_back(GetDataTypeGE(csvMap, "pertoken_scale_dtype", pertoken_scale));
        this->inputInstance.emplace_back(GetDataTypeGE(csvMap, "comm_quant_scale_1_dtype", comm_quant_scale_1));
        this->inputInstance.emplace_back(GetDataTypeGE(csvMap, "comm_quant_scale_2_dtype", comm_quant_scale_2));

        this->outputInstance.emplace_back(1);

        if(this->expectResult == ge::GRAPH_SUCCESS) {
            this->y = Str2DTypeGE(ReadMap(csvMap, "expect_y_dtype"));
        }
    }
};
*/
} // namespace MatmulReduceScatterV2UT

#endif // MATMUL_REDUCE_SCATTER_V2_HOST_UT_PARAM_H
