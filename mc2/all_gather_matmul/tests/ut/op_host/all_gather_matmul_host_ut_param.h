/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ALL_GATHER_MATMUL_HOST_UT_PARAM_H
#define ALL_GATHER_MATMUL_HOST_UT_PARAM_H

#include <sstream>
#include "op_host_csv_case_loader.h"

namespace AllGatherMatmulUT {

struct AllGatherMatmulHostUtParamBase {
    std::string case_name;
    std::vector<uint32_t> inputInstance;
    std::vector<uint32_t> outputInstance;
    std::string group;
    bool is_trans_a;
    bool is_trans_b;
    int64_t gather_index;
    int64_t comm_turn;
    int64_t rank_size;
    bool is_gather_out;
    ge::graphStatus expectResult;

    cHostUtParamBase(const csv_map& csvMap)
    {
        this->case_name = ReadMap(csvMap, "case_name");
        this->group = ReadMap(csvMap, "group");
        this->is_trans_a = stoi(ReadMap(csvMap, "is_trans_a"));
        this->is_trans_b = stoi(ReadMap(csvMap, "is_trans_b"));
        this->gather_index = stoll(ReadMap(csvMap, "gather_index"));
        this->comm_turn = stoll(ReadMap(csvMap, "comm_turn"));
        this->rank_size = stoll(ReadMap(csvMap, "rank_size"));
        this->is_gather_out = stoi(ReadMap(csvMap, "is_gather_out"));
        this->expectResult = ReadMap(csvMap, "expectResult") == "SUCCESS" ? ge::GRAPH_SUCCESS : ge::GRAPH_FAILED;
    }
};

inline std::ostream& operator<<(std::ostream& os, const AllGatherMatmulHostUtParamBase& param)
{
    return os << param.case_name;
}

const gert::TilingContextPara::TensorDescription TD_DEFAULT = {{}, ge::DT_UNDEFINED, ge::FORMAT_NULL};
struct AllGatherMatmulTilingUtParam: public AllGatherMatmulHostUtParamBase {
    gert::TilingContextPara::TensorDescription x1 = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription x2 = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription bias = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription y = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription gather_out = TD_DEFAULT;
    std::string soc;
    uint64_t coreNum;
    uint64_t ubsize;
    uint64_t expectTilingKey;
    std::string expectTilingDataHash;

    AllGatherMatmulTilingUtParam(const csv_map& csvMap):
        AllGatherMatmulHostUtParamBase(csvMap)
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
        this->outputInstance.emplace_back(
            GetTensorGE(csvMap, "output_y_shape", "output_y_dtype", "output_y_format",
                y));
        this->outputInstance.emplace_back(
            GetTensorGE(csvMap, "output_gather_out_shape", "output_gather_out_dtype", "output_gather_out_format",
                gather_out));
        this->soc = ReadMap(csvMap, "soc");
        this->coreNum = stoull(ReadMap(csvMap, "core_num"));
        this->ubsize = stoull(ReadMap(csvMap, "ubsize"));
        if(this->expectResult == ge::GRAPH_SUCCESS) {
            this->expectTilingKey = stoull(ReadMap(csvMap, "expectTilingKey"));
            this->expectTilingDataHash = ReadMap(csvMap, "expectTilingDataHash");
        }
    }
};

const gert::InfershapeContextPara::TensorDescription ID_DEFAULT = {{}, ge::DT_UNDEFINED, ge::FORMAT_NULL};
struct MatmulAllReduceInferShapeUtParam: public MatmulAllReduceHostUtParamBase {
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

    MatmulAllReduceInferShapeUtParam(const csv_map& csvMap):
        MatmulAllReduceHostUtParamBase(csvMap)
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

struct MatmulAllReduceInferDataTypeUtParam: public MatmulAllReduceHostUtParamBase {
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

    MatmulAllReduceInferDataTypeUtParam(const csv_map& csvMap):
        MatmulAllReduceHostUtParamBase(csvMap)
    {
        this->inputInstance.emplace_back(GetDataType(csvMap, "x1_dtype", x1));
        this->inputInstance.emplace_back(GetDataType(csvMap, "x2_dtype", x2));
        this->inputInstance.emplace_back(GetDataType(csvMap, "bias_dtype", bias));
        this->inputInstance.emplace_back(GetDataType(csvMap, "x3_dtype", x3));
        this->inputInstance.emplace_back(GetDataType(csvMap, "antiquant_scale_dtype", antiquant_scale));
        this->inputInstance.emplace_back(GetDataType(csvMap, "antiquant_offset_dtype", antiquant_offset));
        this->inputInstance.emplace_back(GetDataType(csvMap, "dequant_scale_dtype", dequant_scale));
        this->inputInstance.emplace_back(GetDataType(csvMap, "pertoken_scale_dtype", pertoken_scale));
        this->inputInstance.emplace_back(GetDataType(csvMap, "comm_quant_scale_1_dtype", comm_quant_scale_1));
        this->inputInstance.emplace_back(GetDataType(csvMap, "comm_quant_scale_2_dtype", comm_quant_scale_2));
        this->outputInstance.emplace_back(1);
        if(this->expectResult == ge::GRAPH_SUCCESS) {
            GetDataType(csvMap, "expect_y_dtype", y);
        }
    }
};

} // namespace MatmulAllReduceUT

#endif // ALL_GATHER_MATMUL_HOST_UT_PARAM_H
