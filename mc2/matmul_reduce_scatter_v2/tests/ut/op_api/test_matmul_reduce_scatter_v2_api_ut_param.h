/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MATMUL_REDUCE_SCATTER_API_UT_PARAM_H
#define MATMUL_REDUCE_SCATTER_API_UT_PARAM_H

#include <sstream>
#include "op_api_csv_case_loader.h"

namespace MatmulReduceScatterV2UT {

struct MatmulReduceScatterV2ApiUtParam {
    std::string case_name;
    TensorDesc x1;
    TensorDesc x2;
    TensorDesc bias;
    TensorDesc x1Scale;
    TensorDesc x2Scale;
    TensorDesc quantScale;
    TensorDesc output;
    TensorDesc amaxOutOptional;
    int64_t blockSize;
    std::string group;
    std::string reduceOp;
    int64_t commTurn;
    int64_t streamMode;
    int64_t groupSize;
    std::string commMode;
    op::SocVersion soc;
    aclnnStatus expectResult;

    MatmulReduceScatterV2ApiUtParam(const csv_map& csvMap)
    {
        this->case_name = ReadMap(csvMap, "case_name");
        this->x1 = GetTensorACL(csvMap, 
            "x1_shape", "x1_dtype", "x1_format");
        this->x2 = GetTensorACL(csvMap, 
            "x2_shape", "x2_dtype", "x2_format");
        this->bias = GetTensorACL(csvMap, 
            "bias_shape", "bias_dtype", "bias_format");
        this->x1Scale = GetTensorACL(csvMap, 
            "x1_scale_shape", "x1_scale_dtype", "x1_scale_format");
        this->x2Scale = GetTensorACL(csvMap, 
            "x2_scale_shape", "x2_scale_dtype", "x2_scale_format");
        this->quantScale = GetTensorACL(csvMap, 
            "quantScale_shape", "quantScale_dtype", "quantScale_format");
        this->output = GetTensorACL(csvMap, 
            "output_shape", "output_dtype", "output_format");
        this->amaxOutOptional = GetTensorACL(csvMap, 
            "amaxOutOptional_shape", "amaxOutOptional_dtype", "amaxOutOptional_format");
        this->blockSize = stoll(ReadMap(csvMap, "blockSize", "0"));
        this->group = ReadMap(csvMap, "group");
        this->reduceOp = ReadMap(csvMap, "reduceOp");
        this->commTurn = stoll(ReadMap(csvMap, "comm_turn", "8"));
        this->streamMode = stoll(ReadMap(csvMap, "stream_mode", "1"));
        this->groupSize = stoll(ReadMap(csvMap, "group_size", "0"));
        this->commMode = ReadMap(csvMap, "commMode");
        this->soc = GetCaseSocVersion(csvMap, "soc");
        this->expectResult = GetAclnnRet(csvMap, "expect_result");
    }
};

inline std::ostream& operator<<(std::ostream& os, const MatmulReduceScatterV2ApiUtParam& param)
{
    return os << param.case_name;
}

} // namespace MatmulReduceScatterV2UT

#endif // MATMUL_REDUCE_SCATTER_API_UT_PARAM_H
