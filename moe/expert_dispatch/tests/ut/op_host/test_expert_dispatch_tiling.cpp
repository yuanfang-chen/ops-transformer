/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <gtest/gtest.h>
#include "op_log.h"
#define private public
#include "register/op_tiling_registry.h"
#include "test_common.h"
#include "pad_ops.h"
#include "array_ops.h"
#include "common/utils/ut_op_util.h"
#include "op_tiling/op_tiling_util.h"
#include "common_unittest.h"
#include "runtime/expert_dispatch/expert_dispatch_tiling.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "fusion_ops.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"

using namespace ut_util;
using namespace std;
using namespace ge;

class ExpertDispatchTiling : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "ExpertDispatchTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "ExpertDispatchTiling TearDown" << std::endl;
    }
};

static string TilingData2Str(const gert::TilingData* tiling_data)
{
    auto data = tiling_data->GetData();
    string result;
    for (size_t i = 0; i < tiling_data->GetDataSize(); i += sizeof(int64_t)) {
        result += std::to_string((reinterpret_cast<const int64_t*>(tiling_data->GetData())[i / sizeof(int64_t)]));
        result += " ";
    }

    return result;
}

void RunTestCase(gert::StorageShape x_shape, gert::StorageShape expert_id_shape, gert::StorageShape scale_shape,
                 gert::StorageShape dispatched_x_shape, gert::StorageShape dispatched_row_idx_shape,
                 gert::StorageShape expert_tokens_count_shape, gert::StorageShape expert_total_count_shape,
                 gert::StorageShape dispatched_scale_shape, ge::DataType xDataType, std::vector<int64_t> expertRange,
                 ge::graphStatus result, int64_t tilingKey)
{
    string compile_info_string = R"({
         "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                           "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                           "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                           "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                           "CORE_NUM": 40}
                           })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::ExpertDispatchCompileInfo compile_info;

    std::string op_type("ExpertDispatch");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                            intrinsics);
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);

    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(3, 5)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&x_shape, &expert_id_shape, &scale_shape})
                      .OutputShapes({&dispatched_x_shape, &dispatched_row_idx_shape, &expert_tokens_count_shape,
                                     &expert_total_count_shape, &dispatched_scale_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, xDataType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, xDataType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"expert_range", ge::AnyValue::CreateFrom<std::vector<int64_t>>(expertRange)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), result);
    if (result == ge::GRAPH_SUCCESS) {
        // todo check tiling result
        auto tiling_key = tiling_context->GetTilingKey();
        ASSERT_EQ(tiling_key, tilingKey);
        auto tiling_data_result = TilingData2Str(tiling_context->GetRawTilingData());
        std::cout << tiling_data_result << std::endl;
    }
}

void RunNormalCase(int64_t N, int64_t H, int64_t K, ge::DataType xDataType, std::vector<int64_t> expertRange,
                   ge::graphStatus result, int64_t tilingKey)
{
    gert::StorageShape x_shape = {{N, H}, {N, H}};
    gert::StorageShape expert_id_shape = {{N, K}, {N, K}};
    gert::StorageShape scale_shape = {{N}, {N}};

    int64_t E = expertRange[1] - expertRange[0];

    gert::StorageShape dispatched_x_shape = {{N * K, H}, {N * K, H}};
    gert::StorageShape dispatched_row_idx_shape = {{N * K}, {N * K}};
    gert::StorageShape expert_tokens_count_shape = {{E}, {E}};
    gert::StorageShape expert_total_count_shape = {{1}, {1}};
    gert::StorageShape dispatched_scale_shape = {{N * K}, {N * K}};
    RunTestCase(x_shape, expert_id_shape, scale_shape, dispatched_x_shape, dispatched_row_idx_shape,
                expert_tokens_count_shape, expert_total_count_shape, dispatched_scale_shape, xDataType, expertRange,
                result, tilingKey);
}

// 单核+直方图全载+GatherOut全载   1000000
TEST_F(ExpertDispatchTiling, expert_dispatch_tiling_01)
{
    RunNormalCase(4, 3, 5, ge::DT_FLOAT, {1, 7}, ge::GRAPH_SUCCESS, 1000000);
}

// 学员补充：其他Tilingkey模板