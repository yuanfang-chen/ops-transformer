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
#include "tiling_context_faker.h"

namespace matmul_all_reduce_ut {

using TD = gert::TilingContextPara::TensorDescription;
const TD DEFAULT_TD = TD({}, ge::DT_UNDEFINED, ge::FORMAT_ND);

struct MatmulAllReduceHostUtParam {
    std::string case_name;
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
    std::vector<uint32_t> inputInstanceNum;
    TD y;
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
    uint64_t ranksize;
    ge::graphStatus expectResult;
    uint64_t expectTilingKey;
    std::string expectTilingDataHash;

    MatmulAllReduceHostUtParam(const std::string& case_name, std::optional<TD> x1, std::optional<TD> x2,
        std::optional<TD> bias, std::optional<TD> x3, std::optional<TD> antiquant_scale,
        std::optional<TD> antiquant_offset, std::optional<TD> dequant_scale, std::optional<TD> pertoken_scale,
        std::optional<TD> comm_quant_scale_1, std::optional<TD> comm_quant_scale_2, std::optional<TD> y,
        const std::string& group, const std::string& reduce_op, bool is_trans_a, bool is_trans_b, int64_t comm_turn,
        int64_t antiquant_group_size, int64_t group_size, int64_t y_dtype, int64_t comm_quant_mode, uint64_t ranksize,
        ge::graphStatus expectResult, uint64_t expectTilingKey = 0, const std::string& expectTilingDataHash = ""
    ) : case_name(case_name), x1(DEFAULT_TD), x2(DEFAULT_TD), bias(DEFAULT_TD), x3(DEFAULT_TD),
        antiquant_scale(DEFAULT_TD), antiquant_offset(DEFAULT_TD), dequant_scale(DEFAULT_TD),
        pertoken_scale(DEFAULT_TD), comm_quant_scale_1(DEFAULT_TD), comm_quant_scale_2(DEFAULT_TD), y(DEFAULT_TD),
        group(group), reduce_op(reduce_op), is_trans_a(is_trans_a), is_trans_b(is_trans_b), comm_turn(comm_turn),
        antiquant_group_size(antiquant_group_size), group_size(group_size), y_dtype(y_dtype),
        comm_quant_mode(comm_quant_mode), ranksize(ranksize), expectResult(expectResult),
        expectTilingKey(expectTilingKey), expectTilingDataHash(expectTilingDataHash)
    {
        processTensor(x1, this->x1, this->inputInstanceNum);
        processTensor(x2, this->x2, this->inputInstanceNum);
        processTensor(bias, this->bias, this->inputInstanceNum);
        processTensor(x3, this->x3, this->inputInstanceNum);
        processTensor(antiquant_scale, this->antiquant_scale, this->inputInstanceNum);
        processTensor(antiquant_offset, this->antiquant_offset, this->inputInstanceNum);
        processTensor(dequant_scale, this->dequant_scale, this->inputInstanceNum);
        processTensor(pertoken_scale, this->pertoken_scale, this->inputInstanceNum);
        processTensor(comm_quant_scale_1, this->comm_quant_scale_1, this->inputInstanceNum);
        processTensor(comm_quant_scale_2, this->comm_quant_scale_2, this->inputInstanceNum);
        processTensor(y, this->y, this->outputInstanceNum);
    }

private:
    void processTensor(std::optional<TD> src, TD& dst, std::vector<uint32_t>& instanceNum)
    {
        if (src.has_value()) {
            dst = src.value();
            instanceNum.emplace_back(1);
        } else {
            instanceNum.emplace_back(0);
        }
    }
};

inline std::ostream& operator<<(std::ostream& os, const MatmulAllReduceHostUtParam& param)
{
    return os << param.case_name;
}

inline std::string PrintMatmulAllReduceHostUtParam(const testing::TestParamInfo<MatmulAllReduceHostUtParam>& info)
{
    return info.param.case_name;
}

} // namespace matmul_all_reduce_ut

#endif // MATMUL_ALL_REDUCE_HOST_UT_PARAM_H
