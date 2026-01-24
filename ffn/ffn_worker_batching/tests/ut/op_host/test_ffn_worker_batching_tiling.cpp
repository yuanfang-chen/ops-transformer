/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.You may not use this file except in compliance with the License.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 */
#include <iostream>
#include <fstream>
#include <vector>
#include <gtest/gtest.h>
#include "op_log.h"
#include "register/op_tiling_registry.h"
#include "test_common.h"
#include "pad_ops.h"
#include "array_ops.h"
#include "common/utils/ut_op_util.h"
#include "op_tiling/op_tiling_util.h"
#include "common_unittest.h"
#include "runtime/diag_util.h"
#include "runtime/ffn_worker_batching_tiling.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "experiment_ops.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "platform/platform_ascendc.h"

using namespace ge;

class FfnWorkerBatchingTilingTest : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "FfnWorkerBatchingTilingTest SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "FfnWorkerBatchingTilingTest TearDown" << std::endl;
    }
};

TEST_F(FfnWorkerBatchingTilingTest, basic_functionality) {
    dlog_setlevel(0, 0, 0);
    // 1. 定义输入输出张量形状
    gert::StorageShape schedule_context_shape = {{1024}, {1024}};
    gert::StorageShape y_shape = {{1024, 4096}, {1024, 4096}};
    gert::StorageShape group_list_shape = {{8, 2}, {8, 2}};
    gert::StorageShape session_ids_shape = {{1024}, {1024}};
    gert::StorageShape micro_batch_ids_shape = {{1024}, {1024}};
    gert::StorageShape token_ids_shape = {{1024}, {1024}};
    gert::StorageShape expert_offsets_shape = {{1024}, {1024}};
    gert::StorageShape dynamic_scale_shape = {{1024}, {1024}};
    gert::StorageShape actual_token_num_shape = {{1}, {1}};

    std::map<std::string, std::string> soc_version_infos = {{"Short_SoC_version", "Ascend910B"}};

    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 48}
                          })";
    
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // 3. 平台信息初始化
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    
    // 4. 编译信息结构体（根据实际需要定义）
    optiling::FfnWorkerBatchingCompileInfo compile_info;
    // struct FfnWorkerBatchingCompileInfo {
    //     uint32_t totalCoreNum = 40;
    //     uint64_t maxUbSize = 196608;
    //     platform_ascendc::SocVersion curSocVersion = platform_ascendc::SocVersion::ASCEND910B;
    // };
    // FfnWorkerBatchingCompileInfo compile_info;

    // 5. 获取算子Tiling函数
    std::string op_type("FfnWorkerBatching");
    auto op_impl = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str());
    ASSERT_NE(op_impl, nullptr);
    auto tiling_func = op_impl->tiling;
    auto tiling_parse_func = op_impl->tiling_parse;

    // 6. 模拟Tiling解析过程
    auto kernel_holder = gert::KernelRunContextFaker()
        .KernelIONum(2, 1)
        .Inputs({const_cast<char*>(compile_info_string.c_str()), 
                reinterpret_cast<void*>(&platform_info)})
        .Outputs({&compile_info})
        .Build();

    // 设置平台资源信息
    auto parse_ctx = kernel_holder.GetContext<gert::TilingParseContext>();
    ASSERT_TRUE(parse_ctx->GetPlatformInfo()->Init());
    parse_ctx->GetPlatformInfo()->SetPlatformRes("version", soc_version_infos);
    parse_ctx->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    parse_ctx->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    parse_ctx->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    parse_ctx->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    // 执行Tiling解析
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // 7. 准备Tiling计算上下文
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_holder = gert::ContinuousVector::Create<size_t>(409600);
    auto ws_sizes = reinterpret_cast<gert::ContinuousVector*>(workspace_holder.get());
    ASSERT_NE(param, nullptr);

    auto holder = gert::TilingContextFaker()
        .SetOpType("FfnWorkerBatching")
        .NodeIoNum(1, 8) // 1个输入，8个输出
        .IrInstanceNum({1})
        .InputShapes({&schedule_context_shape})
        .OutputShapes({
            &y_shape, &group_list_shape, &session_ids_shape, 
            &micro_batch_ids_shape, &token_ids_shape, &expert_offsets_shape,
            &dynamic_scale_shape, &actual_token_num_shape
        })
        .CompileInfo(&compile_info)
        .PlatformInfo(reinterpret_cast<char*>(&platform_info))
        // 设置输入数据类型和格式
        .NodeInputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
        // 设置输出数据类型和格式
        .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)   // y
        .NodeOutputTd(1, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)  // group_list
        .NodeOutputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)  // session_ids
        .NodeOutputTd(3, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)  // micro_batch_ids
        .NodeOutputTd(4, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)  // token_ids
        .NodeOutputTd(5, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)  // expert_offsets
        .NodeOutputTd(6, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND) // dynamic_scale
        .NodeOutputTd(7, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)   // actual_token_num
        // 设置算子属性
        .NodeAttrs({
            {"expert_num", ge::AnyValue::CreateFrom<int64_t>(8)},
            {"max_out_shape", ge::AnyValue::CreateFrom<std::vector<int64_t>>({16, 6, 9, 4096})},
            {"token_dtype", ge::AnyValue::CreateFrom<int64_t>(0)},
            {"need_schedule", ge::AnyValue::CreateFrom<int64_t>(1)},
            {"layer_num", ge::AnyValue::CreateFrom<int64_t>(1)}
        })
        .TilingData(param.get())
        .Workspace(ws_sizes)
        .Build();

    // 8. 执行Tiling计算
    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context, nullptr);
    auto status = tiling_func(tiling_context);
    
    // 9. 验证结果
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    
    // 验证Tiling Key（假设有特定策略ID）
    auto tiling_key = tiling_context->GetTilingKey();
    EXPECT_EQ(tiling_key, 101);
    dlog_setlevel(0, 3, 0);
}