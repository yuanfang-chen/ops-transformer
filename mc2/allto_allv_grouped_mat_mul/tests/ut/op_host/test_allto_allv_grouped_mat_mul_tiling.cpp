/**
* This program is free software, you can redistribute it and/or modify.
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <gtest/gtest.h>
#include "mc2_tiling_case_executor.h"
#include "../../../op_host/op_tiling/arch35/allto_allv_grouped_mat_mul_quant_tiling_base.h"
#include "../../../op_host/op_tiling/arch35/allto_allv_grouped_mat_mul_tt_quant_tiling.h"
#include "../../../op_host/op_tiling/arch35/allto_allv_grouped_mat_mul_mx_quant_tiling.h"
#include "allto_allv_grouped_mat_mul_host_ut_param.h"

using namespace std;

namespace AlltoAllvGroupedMatMulUT {

class AlltoAllvGroupedMatMulTilingTest : public ::testing::TestWithParam<AlltoAllvGroupedMatMulTilingUTParam> {
protected:
    static void SetUpTestCase() {
        cout << "AlltoAllvGroupedMatMulTilingTest SetUp" << endl;
    }

    static void TearDownTestCase() {
        cout << "AlltoAllvGroupedMatMulTilingTest TearDown" << endl;
    }
};

TEST_P(AlltoAllvGroupedMatMulTilingTest, param) 
{
    auto param = GetParam();

    struct AlltoAllvGroupedMatMulCompileInfo {};
    AlltoAllvGroupedMatMulCompileInfo compileInfo;
    uint64_t coreNum = 36;
    uint64_t ubSize = 256 * 1024;
    size_t tilingDataSize = sizeof(QuantAlltoAllvGroupedMatmulTilingData);

    gert::TilingContextPara tilingContextPara(
        "AlltoAllvGroupedMatMul",
        {
            param.gmm_x,
            param.gmm_weight,
            {{}, ge::DT_INT64, ge::FORMAT_ND}, // send_counts_tensor
            {{}, ge::DT_INT64, ge::FORMAT_ND}, // recv_counts_tensor
            param.mm_x,
            param.mm_x_weight,
            param.gmm_x_scale,
            param.gmm_weight_scale,
            {{}, ge::DT_INT64, ge::FORMAT_ND}, // gmmXOffset
            {{}, ge::DT_INT64, ge::FORMAT_ND}, // gmmWeightOffset
            param.mm_x_scale,
            param.mm_x_weight_scale,
            {{}, ge::DT_INT64, ge::FORMAT_ND}, // gmmXOffset
            {{}, ge::DT_INT64, ge::FORMAT_ND}, // gmmWeightOffset
        },
        {
            param.gmm_y,
            param.mm_y,
            param.permute_out
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.ep_world_size)},
            {"send_counts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(param.sendCounts)},
            {"recv_counts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(param.recvCounts)},
            {"trans_gmm_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(param.trans_gmm_weight_flag)},
            {"trans_mm_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(param.trans_mm_weight_flag)},
            {"permute_out_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(param.permute_out_flag)},
            {"gmm_x_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.gmm_x_quant_mode)},
            {"gmm_weight_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.gmm_weight_quant_mode)},
            {"mm_x_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.mm_x_quant_mode)},
            {"mm_weight_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.mm_weight_quant_mode)},
            {"groupSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"mm_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo,
        "3510",
        coreNum,
        ubSize,
        tilingDataSize
    );

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};

    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, param.expectedStatus, param.expectTilingKey);
}

INSTANTIATE_TEST_SUITE_P(
    AlltoAllvGroupedMatMul,
    AlltoAllvGroupedMatMulTilingTest,
    testing::ValuesIn(GetCasesFromCsv<AlltoAllvGroupedMatMulTilingUTParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<AlltoAllvGroupedMatMulTilingUTParam>
);

} // namespace AlltoAllvGroupedMatMulUT