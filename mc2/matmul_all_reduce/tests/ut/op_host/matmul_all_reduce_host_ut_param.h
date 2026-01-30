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
#include <optional>
#include <sstream>
#include "mc2_csv_case_loader.h"
#include "tiling_context_faker.h"
#include "infer_shape_context_faker.h"

namespace matmul_all_reduce_ut {

struct MatmulAllReduceHostUtParamBase {
    std::string case_name;
    std::vector<uint32_t> inputInstanceNum;
    std::vector<uint32_t> outputInstanceNum;
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

    MatmulAllReduceHostUtParamBase(const std::string& case_name, const std::string& group, const std::string& reduce_op,
        bool is_trans_a, bool is_trans_b, int64_t comm_turn, int64_t antiquant_group_size, int64_t group_size,
        int64_t y_dtype, int64_t comm_quant_mode, ge::graphStatus expectResult) : case_name(case_name), group(group),
        reduce_op(reduce_op), is_trans_a(is_trans_a), is_trans_b(is_trans_b), comm_turn(comm_turn),
        antiquant_group_size(antiquant_group_size), group_size(group_size), y_dtype(y_dtype),
        comm_quant_mode(comm_quant_mode), expectResult(expectResult) {}
    MatmulAllReduceHostUtParamBase(const csv_map& csvMap)
    {
        this->case_name = ReadCsvMap(csvMap, "case_name", "");
        this->group = ReadCsvMap(csvMap, "group", "");
        this->reduce_op = ReadCsvMap(csvMap, "reduce_op", "");
        this->is_trans_a = stoi(ReadCsvMap(csvMap, "is_trans_a", "0"));
        this->is_trans_b = stoi(ReadCsvMap(csvMap, "is_trans_b", "0"));
        this->comm_turn = stoi(ReadCsvMap(csvMap, "comm_turn", "0"));
        this->antiquant_group_size = stoi(ReadCsvMap(csvMap, "antiquant_group_size", "0"));
        this->group_size = stoi(ReadCsvMap(csvMap, "group_size", "0"));
        this->y_dtype = stoi(ReadCsvMap(csvMap, "y_dtype", "0"));
        this->comm_quant_mode = stoi(ReadCsvMap(csvMap, "comm_quant_mode", "0"));
        this->expectResult = stoi(ReadCsvMap(csvMap, "expectResult", "1")) ? ge::GRAPH_SUCCESS : ge::GRAPH_FAILED;
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

template<typename T>
void ProcessOptional(std::optional<T> src, T& dst, std::vector<uint32_t>& instanceNum)
{
    if (src.has_value()) {
        dst = src.value();
        instanceNum.emplace_back(1);
    } else {
        instanceNum.emplace_back(0);
    }
}

// tiling 参数结构体 ====================================================================================================
using TD = gert::TilingContextPara::TensorDescription;
const TD DEFAULT_TD = TD({}, ge::DT_UNDEFINED, ge::FORMAT_ND);

struct MatmulAllReduceTilingUtParam: public MatmulAllReduceHostUtParamBase {
    TD x1;
    TD x2;
    TD bias;
    TD x3;
    TD antiquant_scale;
    TD antiquant_offset;
    TD dequant_scale;
    TD pertoken_scale;
    TD comm_quant_scale_1;
    TD comm_quant_scale_2;
    TD y;
    uint64_t ranksize;
    uint64_t expectTilingKey;
    std::string expectTilingDataHash;

    MatmulAllReduceTilingUtParam(const std::string& case_name, std::optional<TD> x1, std::optional<TD> x2,
        std::optional<TD> bias, std::optional<TD> x3, std::optional<TD> antiquant_scale,
        std::optional<TD> antiquant_offset, std::optional<TD> dequant_scale, std::optional<TD> pertoken_scale,
        std::optional<TD> comm_quant_scale_1, std::optional<TD> comm_quant_scale_2, std::optional<TD> y,
        const std::string& group, const std::string& reduce_op, bool is_trans_a, bool is_trans_b, int64_t comm_turn,
        int64_t antiquant_group_size, int64_t group_size, int64_t y_dtype, int64_t comm_quant_mode, uint64_t ranksize,
        ge::graphStatus expectResult, uint64_t expectTilingKey = 0, const std::string& expectTilingDataHash = "") :
        MatmulAllReduceHostUtParamBase(case_name, group, reduce_op, is_trans_a, is_trans_b, comm_turn,
        antiquant_group_size, group_size, y_dtype, comm_quant_mode, expectResult), x1(DEFAULT_TD), x2(DEFAULT_TD),
        bias(DEFAULT_TD), x3(DEFAULT_TD), antiquant_scale(DEFAULT_TD), antiquant_offset(DEFAULT_TD),
        dequant_scale(DEFAULT_TD), pertoken_scale(DEFAULT_TD), comm_quant_scale_1(DEFAULT_TD),
        comm_quant_scale_2(DEFAULT_TD), y(DEFAULT_TD), ranksize(ranksize), expectTilingKey(expectTilingKey),
        expectTilingDataHash(expectTilingDataHash)
    {
        ProcessOptional(x1, this->x1, this->inputInstanceNum);
        ProcessOptional(x2, this->x2, this->inputInstanceNum);
        ProcessOptional(bias, this->bias, this->inputInstanceNum);
        ProcessOptional(x3, this->x3, this->inputInstanceNum);
        ProcessOptional(antiquant_scale, this->antiquant_scale, this->inputInstanceNum);
        ProcessOptional(antiquant_offset, this->antiquant_offset, this->inputInstanceNum);
        ProcessOptional(dequant_scale, this->dequant_scale, this->inputInstanceNum);
        ProcessOptional(pertoken_scale, this->pertoken_scale, this->inputInstanceNum);
        ProcessOptional(comm_quant_scale_1, this->comm_quant_scale_1, this->inputInstanceNum);
        ProcessOptional(comm_quant_scale_2, this->comm_quant_scale_2, this->inputInstanceNum);
        ProcessOptional(y, this->y, this->outputInstanceNum);
    }
    MatmulAllReduceTilingUtParam(const csv_map& csvMap):
        MatmulAllReduceHostUtParamBase(csvMap), x1(DEFAULT_TD), x2(DEFAULT_TD),
        bias(DEFAULT_TD), x3(DEFAULT_TD), antiquant_scale(DEFAULT_TD), antiquant_offset(DEFAULT_TD),
        dequant_scale(DEFAULT_TD), pertoken_scale(DEFAULT_TD), comm_quant_scale_1(DEFAULT_TD),
        comm_quant_scale_2(DEFAULT_TD), y(DEFAULT_TD) {}
};

// inferShape 参数结构体 ================================================================================================
using ID = gert::InfershapeContextPara::TensorDescription;
const ID DEFAULT_ID = ID({}, ge::DT_UNDEFINED, ge::FORMAT_ND);
struct MatmulAllReduceInferShapeUtParam: public MatmulAllReduceHostUtParamBase {
    ID x1;
    ID x2;
    ID bias;
    ID x3;
    ID antiquant_scale;
    ID antiquant_offset;
    ID dequant_scale;
    ID pertoken_scale;
    ID comm_quant_scale_1;
    ID comm_quant_scale_2;
    ID y;
    uint64_t ranksize;
    std::vector<std::vector<int64_t>> expectOutputShape;

    MatmulAllReduceInferShapeUtParam(const std::string& case_name, std::optional<ID> x1, std::optional<ID> x2,
        std::optional<ID> bias, std::optional<ID> x3, std::optional<ID> antiquant_scale,
        std::optional<ID> antiquant_offset, std::optional<ID> dequant_scale, std::optional<ID> pertoken_scale,
        std::optional<ID> comm_quant_scale_1, std::optional<ID> comm_quant_scale_2, ID y,
        const std::string& group, const std::string& reduce_op, bool is_trans_a, bool is_trans_b, int64_t comm_turn,
        int64_t antiquant_group_size, int64_t group_size, int64_t y_dtype, int64_t comm_quant_mode, uint64_t ranksize,
        ge::graphStatus expectResult, std::vector<std::vector<int64_t>> expectOutputShape = {}) :
        MatmulAllReduceHostUtParamBase(case_name, group, reduce_op, is_trans_a, is_trans_b, comm_turn,
        antiquant_group_size, group_size, y_dtype, comm_quant_mode, expectResult), x1(DEFAULT_ID), x2(DEFAULT_ID),
        bias(DEFAULT_ID), x3(DEFAULT_ID), antiquant_scale(DEFAULT_ID), antiquant_offset(DEFAULT_ID),
        dequant_scale(DEFAULT_ID), pertoken_scale(DEFAULT_ID), comm_quant_scale_1(DEFAULT_ID),
        comm_quant_scale_2(DEFAULT_ID), y(y), ranksize(ranksize), expectOutputShape(expectOutputShape)
    {
        ProcessOptional(x1, this->x1, this->inputInstanceNum);
        ProcessOptional(x2, this->x2, this->inputInstanceNum);
        ProcessOptional(bias, this->bias, this->inputInstanceNum);
        ProcessOptional(x3, this->x3, this->inputInstanceNum);
        ProcessOptional(antiquant_scale, this->antiquant_scale, this->inputInstanceNum);
        ProcessOptional(antiquant_offset, this->antiquant_offset, this->inputInstanceNum);
        ProcessOptional(dequant_scale, this->dequant_scale, this->inputInstanceNum);
        ProcessOptional(pertoken_scale, this->pertoken_scale, this->inputInstanceNum);
        ProcessOptional(comm_quant_scale_1, this->comm_quant_scale_1, this->inputInstanceNum);
        ProcessOptional(comm_quant_scale_2, this->comm_quant_scale_2, this->inputInstanceNum);
        this->outputInstanceNum = {1};
    }
};

// inferDataType 参数结构体 =============================================================================================
using GD = ge::DataType;
const GD DEFAULT_GD = ge::DT_UNDEFINED;
struct MatmulAllReduceInferDataTypeUtParam: public MatmulAllReduceHostUtParamBase {
    GD x1;
    GD x2;
    GD bias;
    GD x3;
    GD antiquant_scale;
    GD antiquant_offset;
    GD dequant_scale;
    GD pertoken_scale;
    GD comm_quant_scale_1;
    GD comm_quant_scale_2;
    GD y;

    MatmulAllReduceInferDataTypeUtParam(const std::string& case_name, std::optional<GD> x1, std::optional<GD> x2,
        std::optional<GD> bias, std::optional<GD> x3, std::optional<GD> antiquant_scale,
        std::optional<GD> antiquant_offset, std::optional<GD> dequant_scale, std::optional<GD> pertoken_scale,
        std::optional<GD> comm_quant_scale_1, std::optional<GD> comm_quant_scale_2, const std::string& group,
        const std::string& reduce_op, bool is_trans_a, bool is_trans_b, int64_t comm_turn, int64_t antiquant_group_size,
        int64_t group_size, int64_t y_dtype, int64_t comm_quant_mode, ge::graphStatus expectResult, GD y = DEFAULT_GD) :
        MatmulAllReduceHostUtParamBase(case_name, group, reduce_op, is_trans_a, is_trans_b, comm_turn,
        antiquant_group_size, group_size, y_dtype, comm_quant_mode, expectResult), x1(DEFAULT_GD), x2(DEFAULT_GD),
        bias(DEFAULT_GD), x3(DEFAULT_GD), antiquant_scale(DEFAULT_GD), antiquant_offset(DEFAULT_GD),
        dequant_scale(DEFAULT_GD), pertoken_scale(DEFAULT_GD), comm_quant_scale_1(DEFAULT_GD),
        comm_quant_scale_2(DEFAULT_GD), y(y)
    {
        ProcessOptional(x1, this->x1, this->inputInstanceNum);
        ProcessOptional(x2, this->x2, this->inputInstanceNum);
        ProcessOptional(bias, this->bias, this->inputInstanceNum);
        ProcessOptional(x3, this->x3, this->inputInstanceNum);
        ProcessOptional(antiquant_scale, this->antiquant_scale, this->inputInstanceNum);
        ProcessOptional(antiquant_offset, this->antiquant_offset, this->inputInstanceNum);
        ProcessOptional(dequant_scale, this->dequant_scale, this->inputInstanceNum);
        ProcessOptional(pertoken_scale, this->pertoken_scale, this->inputInstanceNum);
        ProcessOptional(comm_quant_scale_1, this->comm_quant_scale_1, this->inputInstanceNum);
        ProcessOptional(comm_quant_scale_2, this->comm_quant_scale_2, this->inputInstanceNum);
        this->outputInstanceNum = {1};
    }
};

} // namespace matmul_all_reduce_ut

#endif // MATMUL_ALL_REDUCE_HOST_UT_PARAM_H
