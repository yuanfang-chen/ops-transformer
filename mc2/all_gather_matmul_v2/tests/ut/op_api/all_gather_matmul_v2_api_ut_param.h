/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ALL_GATHER_MATMUL_V2_API_UT_PARAM_H
#define ALL_GATHER_MATMUL_V2_API_UT_PARAM_H

#include <sstream>
#include "op_api_csv_case_loader.h"

namespace AllGatherMatmulV2UT {

struct AllGatherMatmulV2ApiUtParam {
    std::string case_name;
    TensorDesc x1;
    TensorDesc x2;
    TensorDesc bias;
    TensorDesc x1Scale;
    TensorDesc x2Scale;
    TensorDesc quantScale;
    TensorDesc output;
    TensorDesc gatherOut;
    TensorDesc amaxOut;
    std::string group;
    std::string commMode;
    int64_t blockSize;
    int64_t gatherIndex;
    int64_t commTurn;
    int64_t streamMode;
    int64_t groupSize;
    op::SocVersion soc;
    aclnnStatus expectResult;

    AllGatherMatmulV2ApiUtParam(const csv_map& csvMap)
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
            "quant_scale_shape", "quant_scale_dtype", "quant_scale_format");
        this->output = GetTensorACL(csvMap, 
            "output_shape", "output_dtype", "output_format");
        this->amaxOut = GetTensorACL(csvMap, 
            "amaxOut_shape", "amaxOut_dtype", "amaxOut_format");
        this->gatherOut = GetTensorACL(csvMap, 
            "gather_out_shape", "gather_out_dtype", "gather_out_format");
        this->group = ReadMap(csvMap, "group");
        this->commMode = ReadMap(csvMap, "comm_mode");
        this->blockSize = stoll(ReadMap(csvMap, "block_size", "0"));
        this->gatherIndex = stoll(ReadMap(csvMap, "gather_index", "0"));
        this->commTurn = stoll(ReadMap(csvMap, "comm_turn", "0"));
        this->streamMode = stoll(ReadMap(csvMap, "stream_mode", "0"));
        this->groupSize = stoll(ReadMap(csvMap, "group_size", "0"));
        this->soc = GetCaseSocVersion(csvMap, "soc");
        this->expectResult = GetAclnnRet(csvMap, "expect_result");
    }
};

inline std::ostream& operator<<(std::ostream& os, const AllGatherMatmulV2ApiUtParam& param)
{
    return os << param.case_name;
}

} // namespace AllGatherMatmulV2UT

#endif // ALL_GATHER_MATMUL_V2_API_UT_PARAM_H
